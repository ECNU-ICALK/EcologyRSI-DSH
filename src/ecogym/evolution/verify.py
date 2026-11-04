"""One-time verification of an evolution run's result: holdout tasks, then sealed periods.

* ``verify_holdout``: the starting and the final harness each run one episode
  on every holdout task of the suite (tasks no evolving agent ever saw), and
  the final version is compared with the starting one by the suite's gates.
  Each (harness, task, model and budget) exposure is registered in the ledger
  right before its episode starts, so a version meets a holdout task once per
  model and budget across all runs; a later verification of the same version
  under the same conditions reuses the recorded episode. An
  exposure whose episode failed for service reasons, was interrupted or went
  stale may run again, and verifying a run again fills in exactly those.
* ``verify_sealed``: the programs those versions submitted (their latest
  scored episodes on this run's training tasks, and their holdout episodes)
  are fitted on every row before ``validation`` (or ``final_test``) and
  streamed through it, once per (harness, task, stage); ``final_test`` needs
  ``validation`` first. Nobody looked at these periods while the harness
  evolved, so they measure transfer forward in time. Each task's result is
  recorded as soon as it is judged (``sealed_task``): an interrupted
  verification keeps what it finished, and judges again only what this run
  claimed without a result (no score was ever revealed for those). A result
  recorded by another run for the same version is reused.
* ``check_adoptions``: each adoption of a run against the version it
  replaced, on the training tasks' ``validation`` period, with the programs
  the comparison used (judged once per version, task and stage, as above).
  The loop runs it after every adopting round. People read the result; the
  evolver never does, and ``final_test`` stays for the verification above.

A run must have finished to be verified; a stopped one can be finished where
it is (``loop.finish_run``).
"""

from __future__ import annotations

import contextlib
import fcntl
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..episodes.agent import Cancelled
from ..episodes.history import fold_episode
from ..episodes.judge import file_digest, judge_program, load_predictions, reference_scores, save_predictions
from ..episodes.providers import canonical
from ..episodes.runner import EpisodeConfig, run_episode
from ..episodes.sandbox import discover_runtime, ecogym_home, scratch_root
from ..episodes.trajectory import extract_submission
from ..episodes.workspace import LOCAL_STAGE, SCORED_STAGE
from ..evaluation.harness import Bench, Evaluation
from ..evaluation.suite import TaskPair, compare_suite
from ..kernel.canonical import digest
from ..kernel.ledger import Ledger
from ..protocol.partitions import SEALED_STAGES
from ..protocol.views import SealedAccess
from ..task.spec import task_spec
from ..task.suite import suite_spec
from .fold import SCORED_STATUSES, fold
from .loop import EvolveConfig

HOLDOUT = "holdout"
SEALED = "sealed"
RETRYABLE = ("infra_failed", "interrupted")
CLAIM_GRACE = 600.0  # seconds: a claim this young with no episode yet belongs to a verifier that is starting it


def exposure_key(bench: Bench, harness: str, stage: str, kind: str, *,
                 conditions: dict[str, Any] | None = None) -> str:
    """One version's look at one task's stage; ``conditions`` adds what else defines the measurement."""

    body = {"schema": "ecogym.harness-exposure/1", "kind": kind, "frame": bench.frame.digest,
            "task": bench.task.digest, "stage": stage, "interval": bench.partitions.intervals[stage],
            "harness": harness}
    if conditions:
        body["conditions"] = conditions
    return digest(body)


def holdout_conditions(config: EvolveConfig) -> dict[str, Any]:
    """The model and budget of a run's holdout episodes: part of their exposure, so a version verified under
    other conditions runs again instead of reusing an episode that measured something else (on 2026-11-01 the
    seed's 4-minute deepseek-v4-flash episode on Sonoma, from the smoke run, stood in an 8-minute pjlab one)."""

    model = canonical(config.model) if config.model else None
    return {"model": model, "minutes": float(config.minutes), "tokens": int(config.tokens)}


def _finished_view(ledger_path: Path, run_id: str) -> dict[str, Any]:
    with Ledger(ledger_path) as ledger:
        events = ledger.events(run_id)
    if not events:
        raise KeyError(f"no evolution run {run_id}")
    view = fold(events)
    if view["status"] != "finished":
        raise ValueError(f"{run_id} has not finished (status {view['status']}); a stopped run can be ended where it "
                         "is first (ecogym evolve finish)")
    return view


def versions(view: dict[str, Any]) -> list[tuple[str, str]]:
    """The starting and the final version (the one ``evolve_finished`` recorded) of a run."""

    seed, final = view["seed"]["digest"], (view.get("result") or {}).get("final") or view["incumbent"]
    return [("seed", seed)] if final == seed else [("seed", seed), ("final", final)]


def _open(task_id: str, benches: dict[str, Bench]) -> Bench:
    if task_id not in benches:
        bench = Bench.open(task_spec(task_id))
        for stage in (LOCAL_STAGE, SCORED_STAGE):
            bench.stage(stage)
            reference_scores(bench, stage)
        benches[task_id] = bench
    return benches[task_id]


def _age(at: str) -> float:
    return (datetime.now(UTC) - datetime.fromisoformat(at)).total_seconds()


def _entry(record: dict[str, Any], **extra: Any) -> dict[str, Any]:
    scored, finished = record.get("scored") or {}, record.get("finished") or {}
    trajectory = (finished.get("trajectory") or {}).get("dir")
    predictions = str(Path(trajectory) / scored["predictions"]) if trajectory and scored.get("predictions") else None
    return {"episode": record["episode"], "task": record["task_id"], "status": record["status"],
            "score": scored.get("score"), "interval": scored.get("interval"), "predictions": predictions,
            "predictions_digest": scored.get("predictions_digest"), **extra}


def _attempts(ledger: Ledger, keys: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Holdout episodes by exposure key (it is in their context), oldest first."""

    found: dict[str, list[dict[str, Any]]] = {}
    for event in ledger.events(kind="episode_started"):
        key = (event.body.get("context") or {}).get("exposure")
        if key and (keys is None or key in keys):
            found.setdefault(key, []).append(fold_episode(ledger.events(event.run_id)))
    return found


def complete(body: dict[str, Any] | None) -> bool:
    """A recorded holdout verification with a scored episode for every version and task."""

    if not body:
        return False
    return all((version.get("episodes") or {}).get(task, {}).get("status") in SCORED_STATUSES
               for version in (body.get("versions") or {}).values() for task in body.get("tasks") or [])


def _compare(evaluations: dict[str, dict[str, Evaluation]], gates: Any, seed: Any,
             labels: tuple[str, str] = ("final", "seed")) -> dict[str, Any] | None:
    """``labels[0]`` against ``labels[1]`` on the tasks both have, by the suite's gates."""

    after, before = labels
    if set(evaluations) != {after, before}:
        return None
    tasks = sorted(set(evaluations[before]) & set(evaluations[after]))
    if not tasks:
        return None
    pairs = [TaskPair(task, (evaluations[after][task],), (evaluations[before][task],),
                      group=task_spec(task).dataset) for task in tasks]
    return compare_suite(pairs, gates, seed).to_dict()


def _means(results: dict[str, dict[str, dict[str, Any]]]) -> dict[str, float | None]:
    means = {}
    for label, by_task in results.items():
        scores = [item["score"] for item in by_task.values() if isinstance(item.get("score"), (int, float))]
        means[label] = float(np.mean(scores)) if scores else None
    return means


def verify_holdout(ledger_path: Path, run_id: str, *, concurrency: int | None = None,
                   cancel: threading.Event | None = None) -> dict[str, Any]:
    """Setting ``cancel`` stops the running episodes (recorded as interrupted, so they may run again)."""

    view = _finished_view(ledger_path, run_id)
    if complete(view["verification"]["holdout"]):
        return view["verification"]["holdout"]
    config = EvolveConfig.from_dict(view["config"])
    suite = suite_spec(config.suite)
    if suite.digest != view["suite"]["digest"]:
        raise ValueError("the suite differs from the one this run used")
    tasks = suite.tasks("holdout")
    if not tasks:
        raise ValueError(f"suite {suite.id} has no holdout tasks")
    benches: dict[str, Bench] = {}
    for task in tasks:
        _open(task.id, benches)
    stale = config.minutes * 60 + 900
    conditions = holdout_conditions(config)
    results: dict[str, dict[str, dict[str, Any]]] = {label: {} for label, _ in versions(view)}
    jobs = []
    with Ledger(ledger_path) as ledger:
        attempts = _attempts(ledger)
    for label, harness in versions(view):
        for task in tasks:
            key = exposure_key(benches[task.id], harness, SCORED_STAGE, HOLDOUT, conditions=conditions)
            done = [record for record in attempts.get(key, []) if record["status"] in SCORED_STATUSES]
            if done:
                results[label][task.id] = _entry(done[-1], reused=True)
            else:
                jobs.append((label, harness, task, key))

    def run(job: tuple[str, str, Any, str]) -> tuple[str, str, dict[str, Any]]:
        label, harness, task, key = job
        if cancel is not None and cancel.is_set():
            raise Cancelled("the verification was cancelled")
        with Ledger(ledger_path) as ledger:  # claimed right before the episode, so a claim soon has an attempt
            if not ledger.claim_exposure(key, task=task.id, stage=HOLDOUT, method=harness, run_id=run_id):
                tried = _attempts(ledger, {key}).get(key, [])
                done = [record for record in tried if record["status"] in SCORED_STATUSES]
                if done:  # another verification finished it meanwhile
                    return label, task.id, _entry(done[-1], reused=True)
                claim = next((item for item in ledger.exposures(task=task.id) if item["key"] == key), None)
                live = [record for record in tried
                        if record["status"] not in RETRYABLE and _age(record["updated_at"]) < stale]
                if live or (not tried and claim is not None and _age(claim["at"]) < CLAIM_GRACE):
                    raise PermissionError(f"{label} {harness[:12]} on {task.id} is being verified elsewhere "
                                          f"({live[-1]['episode'] if live else 'starting'})")
        entry: dict[str, Any] = {}
        for attempt in range(config.retries + 1):
            episode = EpisodeConfig(task=task.id, harness=harness, minutes=config.minutes, tokens=config.tokens,
                                    model=config.model, label=f"{run_id} 留出验证 {label} {task.id}",
                                    limits=config.limits,
                                    context={"evolution": run_id, "role": HOLDOUT, "version": label,
                                             "exposure": key, "attempt": attempt})
            result = run_episode(episode, ledger_path, bench=benches[task.id], cancel=cancel)
            entry = {"episode": result["episode"], "task": task.id, "status": result["status"],
                     "score": result["score"], "interval": result["scored"]["interval"],
                     "predictions": result["predictions"],
                     "predictions_digest": result["scored"].get("predictions_digest"), "reused": False}
            if result["status"] != "infra_failed":
                break
        return label, task.id, entry

    with ThreadPoolExecutor(max_workers=concurrency or config.concurrency, thread_name_prefix="holdout") as pool:
        for label, task_id, entry in pool.map(run, jobs):
            results[label][task_id] = entry
    evaluations: dict[str, dict[str, Evaluation]] = {}
    for label, by_task in results.items():
        for task_id, entry in by_task.items():
            if entry["status"] in SCORED_STATUSES and entry.get("predictions"):
                evaluations.setdefault(label, {})[task_id] = load_predictions(
                    Path(entry["predictions"]), benches[task_id], expect=entry.get("predictions_digest"))
    body = {"tasks": [task.id for task in tasks],
            "versions": {label: {"digest": harness, "episodes": results[label]} for label, harness in versions(view)},
            "means": _means(results),
            "decision": _compare(evaluations, suite.gates, {"holdout": run_id})}
    with Ledger(ledger_path) as ledger:
        ledger.append(run_id, "holdout_verified", body)
    return body


def _judged(ledger: Ledger) -> dict[str, dict[str, Any]]:
    """Recorded sealed results by exposure key, from any run: per-task events and complete verifications."""

    found: dict[str, dict[str, Any]] = {}
    for event in ledger.events(kind="sealed_verified"):
        for version in (event.body.get("versions") or {}).values():
            for item in (version.get("tasks") or {}).values():
                if item.get("exposure_key"):
                    found[item["exposure_key"]] = {**item, "run_id": event.run_id}
    for event in ledger.events(kind="sealed_task"):
        found[event.body["exposure_key"]] = {**event.body, "run_id": event.run_id}
    return found


def verify_sealed(ledger_path: Path, run_id: str, stage: str, *,
                  cancel: threading.Event | None = None) -> dict[str, Any]:
    """Setting ``cancel`` stops between tasks or inside a judge; finished tasks stay recorded."""

    if stage not in SEALED_STAGES:
        raise ValueError(f"{stage} is not a sealed stage")
    view = _finished_view(ledger_path, run_id)
    if stage in view["verification"]["sealed"]:
        return view["verification"]["sealed"][stage]
    if stage == "final_test" and "validation" not in view["verification"]["sealed"]:
        raise PermissionError("verify validation before final_test")
    config = EvolveConfig.from_dict(view["config"])
    suite = suite_spec(config.suite)
    sources: dict[str, dict[str, dict[str, Any]]] = {label: {} for label, _ in versions(view)}
    for label, harness in versions(view):
        for round_ in view["rounds"]:
            for entry in round_["episodes"]:
                if entry["harness"] == harness and entry.get("status") in SCORED_STATUSES and entry.get("episode"):
                    sources[label][entry["task"]] = entry  # later rounds overwrite: the latest program per task
        holdout = view["verification"]["holdout"] or {}
        for task_id, entry in ((holdout.get("versions") or {}).get(label, {}).get("episodes") or {}).items():
            if entry.get("status") in SCORED_STATUSES:
                sources[label][task_id] = entry
    tag = run_id.partition(":")[2] or run_id
    results, evaluations = _seal(ledger_path, run_id, stage,
                                 [(label, harness, sources[label]) for label, harness in versions(view)],
                                 limits=config.limits, store=ecogym_home() / "verifications" / tag / stage,
                                 filename=lambda label, harness, task_id: f"{label}-{task_id.replace(':', '_')}.npz",
                                 cancel=cancel)
    with Ledger(ledger_path) as ledger:
        body = {"stage": stage,
                "versions": {label: {"digest": harness, "tasks": results[label]} for label, harness in versions(view)},
                "means": _means(results),
                "decision": _compare(evaluations, suite.gates, {"sealed": run_id, "stage": stage}),
                "methods_exposed_on_stage": len({item["method"] for item in ledger.exposures()
                                                 if item["stage"] == stage})}
        ledger.append(run_id, "sealed_verified", body)
    return body


def _seal(ledger_path: Path, run_id: str, stage: str,
          plan: list[tuple[str, str, dict[str, dict[str, Any]]]], *, limits: Any, store: Path,
          filename: Callable[[str, str, str], str], cancel: threading.Event | None = None
          ) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Evaluation]]]:
    """Judge the programs of ``plan`` ((label, harness, task -> the episode that submitted it)) on a sealed stage,
    once per (harness, task, stage) across all runs: a result recorded before, by any run, is reused (whichever
    program was judged first stands for the version: a final version checked when it was adopted keeps that
    check's result). Each new result is recorded as soon as it is judged (``sealed_task``). One process at a time
    judges a run's stage (``_stage_lock``), so this run's own claims without a result, left by an interruption,
    are judged again by nobody else meanwhile. Results and evaluations by label and task."""

    tag = run_id.partition(":")[2] or run_id
    with _stage_lock(ecogym_home() / "verifications" / tag / f"{stage}.lock", run_id, stage):
        return _seal_locked(ledger_path, run_id, stage, plan, limits=limits, store=store, filename=filename,
                            cancel=cancel)


@contextlib.contextmanager
def _stage_lock(path: Path, run_id: str, stage: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PermissionError(f"another process is judging {run_id} on {stage}; try again when it is done") \
                from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _seal_locked(ledger_path: Path, run_id: str, stage: str,
                 plan: list[tuple[str, str, dict[str, dict[str, Any]]]], *, limits: Any, store: Path,
                 filename: Callable[[str, str, str], str], cancel: threading.Event | None = None
                 ) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Evaluation]]]:
    runtime = discover_runtime()
    benches: dict[str, Bench] = {}
    store.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, dict[str, Any]]] = {label: {} for label, _, _ in plan}
    evaluations: dict[str, dict[str, Evaluation]] = {}
    for label, harness, sources in plan:
        for task_id, entry in sorted(sources.items()):
            if cancel is not None and cancel.is_set():
                raise Cancelled("the verification was cancelled")
            bench = _open(task_id, benches)
            with Ledger(ledger_path) as ledger:  # read again for every task: another run may have judged it since
                judged = _judged(ledger)
            if stage == "final_test":
                validated = judged.get(exposure_key(bench, harness, "validation", SEALED))
                if validated is None:
                    continue  # final_test only follows a recorded validation result
                entry = {"episode": validated["episode"]}  # the very program that was validated
            key = exposure_key(bench, harness, stage, SEALED)
            access = SealedAccess(stage, key)
            if key in judged:  # this version met this task's stage before: its one result stands
                result = {key_: value for key_, value in judged[key].items()
                          if key_ not in ("stage", "version", "harness", "task", "run_id")}
                result["reused"] = judged[key]["run_id"] != run_id or bool(result.get("reused"))
                result["judged_as"] = judged[key].get("version")  # e.g. "adopted": judged by an adoption check
                results[label][task_id] = result
                evaluations.setdefault(label, {})[task_id] = load_predictions(
                    Path(result["predictions"]), bench, bench.stage(stage, sealed=access),
                    expect=result.get("predictions_digest"))
                continue
            with Ledger(ledger_path) as ledger:
                if not ledger.claim_exposure(key, task=task_id, stage=stage, method=harness, run_id=run_id):
                    claim = next((item for item in ledger.exposures(task=task_id) if item["key"] == key), {})
                    if claim.get("run_id") != run_id:
                        other = claim.get("run_id")
                        raise PermissionError(
                            f"{label} {harness[:12]} on {task_id} was claimed for {stage} by {other}, which has no "
                            f"result yet (it was interrupted); complete that first: `ecogym evolve check-adoptions "
                            f"{other}` if it was an adoption check, else verify {other} on {stage}")
                    # claimed by this run before an interruption, never recorded: no score was revealed
                record = fold_episode(ledger.events(entry["episode"]))
            trajectory = Path(((record.get("finished") or {}).get("trajectory") or {}).get("dir", ""))
            scratch = scratch_root() / f"seal-{uuid.uuid4().hex[:12]}"
            try:
                program = extract_submission(trajectory, record.get("scored"), scratch / "program")
                report, evaluation = judge_program(bench, program, stage, runtime, limits=limits,
                                                   sealed=access, cancel=cancel)
            finally:
                subprocess.run(["rm", "-r", "-f", str(scratch)], check=False)
            path = save_predictions(store / filename(label, harness, task_id), bench, evaluation)
            result = {
                "episode": entry["episode"], "status": report["status"], "score": report["score"],
                "interval": report["interval"], "coverage": report["coverage"], "origins": report["origins"],
                "cells": report["cells"], "predictions": str(path), "predictions_digest": file_digest(path),
                "exposure_key": key, "judged_at": time.time(),
            }
            with Ledger(ledger_path) as ledger:
                ledger.append(run_id, "sealed_task", {"stage": stage, "version": label, "harness": harness,
                                                      "task": task_id, **result})
            results[label][task_id] = result
            evaluations.setdefault(label, {})[task_id] = evaluation
    return results, evaluations


ADOPTION_STAGE = "validation"  # final_test stays untouched until the run's own verification


def unchecked(view: dict[str, Any]) -> list[int]:
    """The generations whose adoption has no check recorded without an error."""

    done = {item.get("generation") for item in view.get("adoption_checks", []) if not item.get("error")}
    return [item["generation"] for item in view["adoptions"] if item["generation"] not in done]


def adoption_sources(view: dict[str, Any], generation: int) -> tuple[str, str, dict[str, dict[str, Any]],
                                                                     dict[str, dict[str, Any]]] | None:
    """The version adopted in ``generation`` and the incumbent it replaced, with the programs to judge for each:
    the adopted candidate's episodes of that round, and the replaced incumbent's latest episodes up to that round
    (the ones the comparison used), one per training task. None when the generation adopted nothing.

    A version judged on the stage before keeps that result (``_seal``): the incumbent replaced in a later
    generation was judged when it was adopted, so its check uses the programs of its own adoption."""

    last = {}
    for round_ in view["rounds"]:
        last[round_["generation"]] = round_
    round_ = last.get(generation)
    if not round_ or not round_.get("finished") or not round_.get("adopted"):
        return None
    adopted = round_["incumbent_after"]
    parent = round_["incumbent"]
    mine: dict[str, dict[str, Any]] = {}
    theirs: dict[str, dict[str, Any]] = {}
    for entry in round_["episodes"]:
        if (entry.get("role") == "candidate" and entry.get("candidate") == round_["adopted"]
                and entry.get("status") in SCORED_STATUSES and entry.get("episode")):
            mine[entry["task"]] = entry
    for key in sorted(last):
        if key > generation:
            break
        for entry in last[key]["episodes"]:
            if (entry.get("role") == "incumbent" and entry.get("harness") == parent
                    and entry.get("status") in SCORED_STATUSES and entry.get("episode")):
                theirs[entry["task"]] = entry  # later rounds overwrite: the latest program per task
    return parent, adopted, theirs, mine


def check_adoptions(ledger_path: Path, run_id: str, *, generations: list[int] | None = None,
                    cancel: threading.Event | None = None) -> list[dict[str, Any]]:
    """Each adoption of a run (or those of ``generations``) against the incumbent it replaced, on the training
    tasks' sealed ``validation`` period, where no adoption was checked yet; the new ``adoption_checked`` bodies.

    The gate adopts on the training tasks' hidden period, and a run adopts many small changes chosen on that one
    period: their sum can be partly selection on its noise. The later, sealed period was looked at by nobody, so
    each adoption is measured there once: only the programs are judged again (no episodes), every version meets
    each task's stage once across all runs (results are reused), and ``final_test`` is left for the run's own
    verification. The evolver never sees these results (the evidence never reads them); people do, in the panel."""

    with Ledger(ledger_path) as ledger:
        events = ledger.events(run_id)
    if not events:
        raise KeyError(f"no evolution run {run_id}")
    view = fold(events)
    config = EvolveConfig.from_dict(view["config"])
    gates = suite_spec(config.suite).gates
    wanted = [generation for generation in unchecked(view) if generations is None or generation in generations]
    tag = run_id.partition(":")[2] or run_id
    bodies = []
    for generation in wanted:
        found = adoption_sources(view, generation)
        if found is None:
            continue
        parent, adopted, theirs, mine = found
        tasks = sorted(set(theirs) & set(mine))
        body: dict[str, Any] = {"generation": generation, "digest": adopted, "parent": parent,
                                "stage": ADOPTION_STAGE, "tasks": tasks}
        try:
            results, evaluations = _seal(
                ledger_path, run_id, ADOPTION_STAGE,
                [("parent", parent, {task: theirs[task] for task in tasks}),
                 ("adopted", adopted, {task: mine[task] for task in tasks})],
                limits=config.limits, store=ecogym_home() / "verifications" / tag / ADOPTION_STAGE / "adoptions",
                filename=lambda label, harness, task_id: f"{harness[:12]}-{task_id.replace(':', '_')}.npz",
                cancel=cancel)
            body.update({"versions": {"parent": {"digest": parent, "tasks": results["parent"]},
                                      "adopted": {"digest": adopted, "tasks": results["adopted"]}},
                         "means": _means(results),
                         "decision": _compare(evaluations, gates, {"adoption": run_id, "generation": generation},
                                              labels=("adopted", "parent"))})
        except Cancelled:
            raise
        except Exception as error:  # recorded, and tried again by the next check of this run
            body["error"] = f"{type(error).__name__}: {error}"[:1000]
        with Ledger(ledger_path) as ledger:
            ledger.append(run_id, "adoption_checked", body)
        bodies.append(body)
    return bodies


__all__ = ["ADOPTION_STAGE", "HOLDOUT", "SEALED", "adoption_sources", "check_adoptions", "complete", "exposure_key",
           "unchecked", "verify_holdout", "verify_sealed", "versions"]
