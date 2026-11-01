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
"""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..episodes.agent import Cancelled
from ..episodes.history import fold_episode
from ..episodes.judge import file_digest, judge_program, load_predictions, reference_scores, save_predictions
from ..episodes.providers import split as split_model
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

    model = "/".join(split_model(config.model)) if config.model else None
    return {"model": model, "minutes": float(config.minutes), "tokens": int(config.tokens)}


def _finished_view(ledger_path: Path, run_id: str) -> dict[str, Any]:
    with Ledger(ledger_path) as ledger:
        events = ledger.events(run_id)
    if not events:
        raise KeyError(f"no evolution run {run_id}")
    view = fold(events)
    if view["status"] != "finished":
        raise ValueError(f"{run_id} has not finished (status {view['status']})")
    return view


def versions(view: dict[str, Any]) -> list[tuple[str, str]]:
    seed, final = view["seed"]["digest"], view["incumbent"]
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


def _compare(evaluations: dict[str, dict[str, Evaluation]], gates: Any, seed: Any) -> dict[str, Any] | None:
    if set(evaluations) != {"seed", "final"}:
        return None
    tasks = sorted(set(evaluations["seed"]) & set(evaluations["final"]))
    if not tasks:
        return None
    pairs = [TaskPair(task, (evaluations["final"][task],), (evaluations["seed"][task],),
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
    runtime = discover_runtime()
    benches: dict[str, Bench] = {}
    tag = run_id.partition(":")[2] or run_id
    store = ecogym_home() / "verifications" / tag / stage
    store.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in sources}
    evaluations: dict[str, dict[str, Evaluation]] = {}
    with Ledger(ledger_path) as ledger:
        judged = _judged(ledger)
    for label, harness in versions(view):
        for task_id, entry in sorted(sources[label].items()):
            if cancel is not None and cancel.is_set():
                raise Cancelled("the verification was cancelled")
            bench = _open(task_id, benches)
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
                results[label][task_id] = result
                evaluations.setdefault(label, {})[task_id] = load_predictions(
                    Path(result["predictions"]), bench, bench.stage(stage, sealed=access),
                    expect=result.get("predictions_digest"))
                continue
            with Ledger(ledger_path) as ledger:
                if not ledger.claim_exposure(key, task=task_id, stage=stage, method=harness, run_id=run_id):
                    claim = next((item for item in ledger.exposures(task=task_id) if item["key"] == key), {})
                    if claim.get("run_id") != run_id:
                        raise PermissionError(f"{label} {harness[:12]} on {task_id} was claimed for {stage} by "
                                              f"{claim.get('run_id')}, which has no result yet; verify that run first")
                    # claimed by this run before an interruption, never recorded: no score was revealed
                record = fold_episode(ledger.events(entry["episode"]))
            trajectory = Path(((record.get("finished") or {}).get("trajectory") or {}).get("dir", ""))
            scratch = scratch_root() / f"seal-{uuid.uuid4().hex[:12]}"
            try:
                program = extract_submission(trajectory, record.get("scored"), scratch / "program")
                report, evaluation = judge_program(bench, program, stage, runtime, limits=config.limits,
                                                   sealed=access, cancel=cancel)
            finally:
                subprocess.run(["rm", "-r", "-f", str(scratch)], check=False)
            path = save_predictions(store / f"{label}-{task_id.replace(':', '_')}.npz", bench, evaluation)
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
    with Ledger(ledger_path) as ledger:
        body = {"stage": stage,
                "versions": {label: {"digest": harness, "tasks": results[label]} for label, harness in versions(view)},
                "means": _means(results),
                "decision": _compare(evaluations, suite.gates, {"sealed": run_id, "stage": stage}),
                "methods_exposed_on_stage": len({item["method"] for item in ledger.exposures()
                                                 if item["stage"] == stage})}
        ledger.append(run_id, "sealed_verified", body)
    return body


__all__ = ["HOLDOUT", "SEALED", "complete", "exposure_key", "verify_holdout", "verify_sealed", "versions"]
