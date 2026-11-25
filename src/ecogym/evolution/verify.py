"""One-time verification of an evolution run's result: holdout tasks, then sealed periods.

* ``verify_holdout``: the starting and the final harness each run one episode
  on every holdout task of the suite (tasks no evolving agent ever saw), and
  the final version is compared with the starting one by the suite's gates.
  Each (harness, task, model and budget) exposure is registered in the ledger
  right before its episode starts, so a version meets a holdout task once per
  model and budget across all runs; a later verification of the same version
  under the same conditions reuses the recorded episode. An exposure whose
  episode failed for service reasons, was interrupted or went stale may run
  again, and verifying a run again fills in exactly those.
* ``verify_sealed``: the programs those versions submitted are fitted on
  every row before ``validation`` (or ``final_test``) and streamed through
  it; ``final_test`` needs all of ``validation`` first. Nobody looked at these
  periods while the harness evolved, so they measure transfer forward in
  time. The plan is fixed before anything is judged: every version on every
  task of the suite (training and holdout tasks), and for each the programs
  of every replicate in the run's first look, or its latest finished round if
  no look was taken (a holdout task: its holdout episode). A harness is a
  policy whose episodes write different programs; its last replicate alone
  would stand for whichever replicate happened to be stored last. Each task's
  result is recorded program by program (``sealed_task``): an interrupted
  verification keeps what it finished, and only resumes unjudged programs of
  the frozen look. What the plan could not judge is listed (``missing``) and
  the record says whether it is ``complete``; verifying again fills in what
  has become possible since (a holdout task verified later), never what was
  judged.
* ``check_adoptions``: each adoption of a run against the version it
  replaced, on the training tasks' ``validation`` period, with the programs
  of the adopting round on both sides (every replicate; the replaced version's
  latest earlier round where it has none there). When the run enables it
  (``EvolveConfig.check_adoptions``, off by default since 0.4.0, because each
  check spends a look at ``validation`` on programs chosen by the adoption),
  the loop runs it after every adopting round. People read the result; the
  evolver never does, and ``final_test`` stays for the verification above.

A sealed stage is looked at once per (version, model and budget, task, stage)
across all runs (``look_key``): the model and budget the programs' episodes
ran with are part of what a version's result means, so a version verified
under other conditions gets a look of its own instead of reusing one that
measured something else (2026-10-29: the seed's 4-minute deepseek-v4-flash
programs stood in for its 8-minute pjlab ones in evo:9eca5239360b). Only
those explicit conditions open a new look: changing programs, guidance or
code does not, so the sealed periods cannot be reopened at will. A previous
look is usable only when it covers the planned episode and program
identities. Every program is checked against the digest it was scored with
before it is judged again (``trajectory.extract_submission``).

A run must have finished to be verified; a stopped one can be finished where
it is (``loop.finish_run``).
"""

from __future__ import annotations

import contextlib
import fcntl
import subprocess
import tarfile
import threading
import uuid
import zipfile
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..episodes.agent import Cancelled
from ..episodes.history import fold_episode
from ..episodes.judge import (
    JUDGE_PROTOCOL,
    file_digest,
    judge_program,
    load_predictions,
    reference_scores,
    save_predictions,
)
from ..episodes.providers import canonical
from ..episodes.runner import EpisodeConfig, run_episode
from ..episodes.sandbox import discover_runtime, ecogym_home, scratch_root
from ..episodes.tools import SINGLE_WINDOW_SELECTION
from ..episodes.trajectory import ProgramMismatch, extract_submission
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
    return _selection_conditions({"model": model, "minutes": float(config.minutes), "tokens": int(config.tokens)},
                                 config.local_selection)


def _selection_conditions(conditions: dict[str, Any], policy: str | None) -> dict[str, Any]:
    # Keep historical exposure digests stable. The old policy was implicit in their records.
    if policy and policy != SINGLE_WINDOW_SELECTION:
        conditions["local_selection"] = policy
    return conditions


def conditions_of(record: dict[str, Any]) -> dict[str, Any]:
    """The model and budget an episode ran with (its folded record): what its program measured besides the
    harness. A sealed look is about a version under these conditions."""

    model = record.get("model")
    if model:
        with contextlib.suppress(ValueError):
            model = canonical(model)
    return _selection_conditions({"model": model or None, "minutes": float(record.get("minutes") or 0.0),
                                  "tokens": int(record.get("tokens") or 0)}, record.get("local_selection"))


@dataclass(slots=True)
class Look:
    """One version's programs on one task: what a sealed stage judges, once per (version, conditions, task,
    stage). ``episodes`` are the episodes whose scored programs are judged, all run under ``conditions``."""

    label: str
    harness: str
    task: str
    episodes: tuple[str, ...]
    conditions: dict[str, Any] = field(default_factory=dict)
    program_digests: dict[str, str | None] = field(default_factory=dict)
    bench_identity: str | None = None

    def plan(self) -> dict[str, Any]:
        return {"version": self.label, "harness": self.harness, "task": self.task, "conditions": self.conditions,
                "episodes": list(self.episodes), "program_digests": self.program_digests,
                "bench_identity": self.bench_identity}


def latest_programs(view: dict[str, Any], harness: str, *, until: int | None = None) -> dict[str, list[str]]:
    """Task -> every scored episode of a version in the latest finished round (up to generation ``until``) in
    which it has any on that task: the replicates the comparisons of that round used, not one of them."""

    found: dict[str, list[str]] = {}
    for round_ in view["rounds"]:  # in order: a later round replaces an earlier one, task by task
        if not round_.get("finished") or (until is not None and round_["generation"] > until):
            continue
        here: dict[str, list[str]] = {}
        for entry in round_["episodes"]:
            if (entry.get("harness") == harness and entry.get("status") in SCORED_STATUSES
                    and entry.get("episode")):
                here.setdefault(entry["task"], []).append(entry["episode"])
        found.update(here)
    return found


def _looks(ledger: Ledger, wanted: Sequence[tuple[str, str, str, Sequence[str]]]
           ) -> tuple[list[Look], list[dict[str, Any]]]:
    """Resolve the planned episode identities. A missing record or mixed conditions leave a gap."""

    looks, missing = [], []
    for label, harness, task, episodes in wanted:
        if len(set(episodes)) != len(episodes):
            missing.append({"version": label, "task": task, "reason": "duplicate_episode", "episodes": list(episodes)})
            continue
        records = [_episode(ledger, episode) for episode in episodes]
        if any(record is None or record["status"] not in SCORED_STATUSES or record["task_id"] != task
               or (record.get("harness") or {}).get("digest") != harness for record in records):
            missing.append({"version": label, "task": task, "reason": "episode_record_unavailable",
                            "episodes": list(episodes)})
            continue
        conditions = conditions_of(records[0])
        if any(conditions_of(record) != conditions for record in records):
            missing.append({"version": label, "task": task, "reason": "mixed_episode_conditions",
                            "episodes": list(episodes)})
            continue
        benches = {digest(record["bench"]) if record.get("bench") else None for record in records}
        if len(benches) != 1:
            missing.append({"version": label, "task": task, "reason": "mixed_bench_identities",
                            "episodes": list(episodes)})
            continue
        programs = {episode: (record.get("scored") or {}).get("digest")
                    for episode, record in zip(episodes, records, strict=True)}
        looks.append(Look(label, harness, task, tuple(episodes), conditions, programs, benches.pop()))
    return looks, missing


def look_key(bench: Bench, look: Look, stage: str) -> str:
    """The exposure of a sealed look: the version, the conditions its programs ran under, the task and stage."""

    return exposure_key(bench, look.harness, stage, SEALED, conditions=look.conditions)


def _key_bench(task_id: str, benches: dict[str, Bench]) -> Bench:
    """A task's bench for exposure keys only (its frame, task and partitions; nothing is evaluated)."""

    if task_id not in benches:
        benches[task_id] = Bench.open(task_spec(task_id))
    return benches[task_id]


def _first_looks(ledger: Ledger, looks: Sequence[Look], stage: str) -> tuple[list[Look], set[tuple[str, str]]]:
    """Each look as ``stage`` first saw it, and the (label, task) of the looks this replaced.

    A version judged on the stage before under the same conditions keeps that result, whichever run took it
    (``adoption_sources``): a run that starts from an earlier run's final version, or checks an adoption whose
    replaced version was judged before, plans the programs of that look. Planning its own instead would find the
    exposure taken by other programs (``prior_exposure_differs``) and could never complete: on the 2026-11-17
    ledger every run that started from a verified version was in that state."""

    judged = _judged(ledger)
    benches: dict[str, Bench] = {}
    first, replaced = [], set()
    for look in looks:
        found = _reused(ledger, judged, _key_bench(look.task, benches), look, stage)
        planned = (found or {}).get("plan")
        episodes = tuple(planned["episodes"]) if planned else tuple(program_episodes(found))
        if not episodes or set(episodes) == set(look.episodes):
            first.append(look)
            continue
        if planned:
            earlier: Look | None = Look(look.label, look.harness, look.task, episodes, look.conditions,
                                        dict(planned.get("program_digests") or {}), planned.get("bench_identity"))
        else:  # recorded before looks kept their plan: its program is read from the episode record
            resolved, _ = _looks(ledger, [(look.label, look.harness, look.task, episodes)])
            earlier = resolved[0] if resolved and resolved[0].conditions == look.conditions else None
        if earlier is None:  # unreadable: the seal reports the conflict (prior_exposure_differs)
            first.append(look)
            continue
        first.append(earlier)
        replaced.add((look.label, look.task))
    return first, replaced


def _result_conditions(ledger: Ledger, result: dict[str, Any]) -> dict[str, Any] | None:
    """The conditions a recorded sealed result measured: recorded with it, or (results recorded before looks
    named them) those of the episode whose program it judged."""

    if result.get("conditions"):
        return result["conditions"]
    events = ledger.events(result["episode"]) if result.get("episode") else []
    return conditions_of(fold_episode(events)) if events else None


def program_episodes(result: dict[str, Any] | None) -> list[str]:
    """The episodes whose programs a recorded sealed result judged (one, for results recorded before looks
    judged every replicate)."""

    if not result:
        return []
    found = [item["episode"] for item in result.get("programs") or [] if item.get("episode")]
    return found or ([result["episode"]] if result.get("episode") else [])


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


def _episode(ledger: Ledger, episode: str | None) -> dict[str, Any] | None:
    events = ledger.events(episode) if episode else []
    return fold_episode(events) if events and events[0].kind == "episode_started" else None


def _entry(record: dict[str, Any], **extra: Any) -> dict[str, Any]:
    scored, finished = record.get("scored") or {}, record.get("finished") or {}
    trajectory = (finished.get("trajectory") or {}).get("dir")
    predictions = str(Path(trajectory) / scored["predictions"]) if trajectory and scored.get("predictions") else None
    return {"episode": record["episode"], "task": record["task_id"], "status": record["status"],
            "score": scored.get("score"), "interval": scored.get("interval"), "predictions": predictions,
            "predictions_digest": scored.get("predictions_digest"), "judge_protocol": scored.get("judge_protocol"),
            **extra}


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

    if not body or body.get("complete") is False:
        return False
    tasks, held = body.get("tasks") or [], body.get("versions") or {}
    return bool(tasks and held) and all(
        (entry := (version.get("episodes") or {}).get(task, {})).get("status") in SCORED_STATUSES
        and entry.get("episode") and entry.get("predictions") for version in held.values() for task in tasks)


def _prediction_intact(entry: dict[str, Any]) -> bool:
    if not entry.get("predictions"):
        return False
    path = Path(entry["predictions"])
    expected = entry.get("predictions_digest")
    try:
        return file_digest(path) == expected if expected else path.is_file()
    except OSError:
        return False


def _holdout_record(ledger: Ledger, episode: str | None, harness: str, task: str,
                    expected: dict[str, Any]) -> dict[str, Any] | None:
    record = _episode(ledger, episode)
    if (record is not None and record["status"] in SCORED_STATUSES and record["task_id"] == task
            and (record.get("harness") or {}).get("digest") == harness
            and record["context"].get("role") == HOLDOUT and _ran_under(conditions_of(record), expected)):
        return record
    return None


def _compare(evaluations: dict[str, dict[str, tuple[Evaluation, ...]]], gates: Any, seed: Any,
             labels: tuple[str, str] = ("final", "seed")) -> dict[str, Any] | None:
    """``labels[0]`` against ``labels[1]`` on the tasks both have, by the suite's gates; each task's programs are
    the replicates of its version."""

    after, before = labels
    if set(evaluations) != {after, before}:
        return None
    tasks = sorted(set(evaluations[before]) & set(evaluations[after]))
    if not tasks:
        return None
    pairs = [TaskPair(task, tuple(evaluations[after][task]), tuple(evaluations[before][task]),
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
    config = EvolveConfig.from_dict(view["config"])
    suite = suite_spec(config.suite)
    if suite.digest != view["suite"]["digest"]:
        raise ValueError("the suite differs from the one this run used")
    tasks = suite.tasks("holdout")
    if not tasks:
        raise ValueError(f"suite {suite.id} has no holdout tasks")
    conditions = holdout_conditions(config)
    recorded = view["verification"]["holdout"] or {}
    version_list = versions(view)
    results: dict[str, dict[str, dict[str, Any]]] = {label: {} for label, _ in version_list}
    with Ledger(ledger_path) as ledger:
        for label, harness in version_list:
            held = ((recorded.get("versions") or {}).get(label) or {}).get("episodes") or {}
            for task in tasks:
                entry = held.get(task.id) or {}
                valid = _holdout_record(ledger, entry.get("episode"), harness, task.id, conditions)
                if valid:
                    results[label][task.id] = _entry(valid, reused=True)
    protocols = {entry.get("judge_protocol") for side in results.values() for entry in side.values()}
    if (complete(recorded) and recorded["tasks"] == [task.id for task in tasks] and len(protocols) == 1
            and {label: item.get("digest") for label, item in recorded["versions"].items()} == dict(version_list)
            and all(len(results[label]) == len(tasks) for label, _ in version_list)
            and all(_prediction_intact(entry) for side in results.values() for entry in side.values())):
        return recorded

    benches: dict[str, Bench] = {}
    for task in tasks:
        _open(task.id, benches)
    stale = config.minutes * 60 + 900
    jobs = []
    with Ledger(ledger_path) as ledger:
        attempts = _attempts(ledger)
    for label, harness in version_list:
        for task in tasks:
            if task.id in results[label]:
                continue
            key = exposure_key(benches[task.id], harness, SCORED_STAGE, HOLDOUT, conditions=conditions)
            legacy_key = exposure_key(benches[task.id], harness, SCORED_STAGE, HOLDOUT)
            done = [record for old in (legacy_key, key) for record in attempts.get(old, [])
                    if record["status"] in SCORED_STATUSES and _ran_under(conditions_of(record), conditions)]
            if done:
                results[label][task.id] = _entry(done[-1], reused=True)
            else:
                jobs.append((label, harness, task, key))

    def run(job: tuple[str, str, Any, str]) -> tuple[str, str, dict[str, Any]]:
        key = job[3]
        with _stage_lock(ecogym_home() / "verifications" / "holdout-exposures" / f"{key}.lock", run_id, HOLDOUT):
            return run_claimed(job)

    def run_claimed(job: tuple[str, str, Any, str]) -> tuple[str, str, dict[str, Any]]:
        label, harness, task, key = job
        if cancel is not None and cancel.is_set():
            raise Cancelled("the verification was cancelled")
        with Ledger(ledger_path) as ledger:  # claimed right before the episode, so a claim soon has an attempt
            if not ledger.claim_exposure(key, task=task.id, stage=HOLDOUT, method=harness, run_id=run_id):
                tried = _attempts(ledger, {key}).get(key, [])
                done = [record for record in tried if record["status"] in SCORED_STATUSES
                        and _ran_under(conditions_of(record), conditions)]
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
                                    limits=config.limits, local_selection=config.local_selection,
                                    context={"evolution": run_id, "role": HOLDOUT, "version": label,
                                             "exposure": key, "attempt": attempt})
            result = run_episode(episode, ledger_path, bench=benches[task.id], cancel=cancel)
            entry = {"episode": result["episode"], "task": task.id, "status": result["status"],
                     "score": result["score"], "interval": result["scored"]["interval"],
                     "predictions": result["predictions"],
                     "predictions_digest": result["scored"].get("predictions_digest"),
                     "judge_protocol": result["scored"].get("judge_protocol"), "reused": False}
            if result["status"] != "infra_failed":
                break
        return label, task.id, entry

    with ThreadPoolExecutor(max_workers=concurrency or config.concurrency, thread_name_prefix="holdout") as pool:
        for label, task_id, entry in pool.map(run, jobs):
            results[label][task_id] = entry
    evaluations: dict[str, dict[str, tuple[Evaluation, ...]]] = {}
    missing = []
    for label, by_task in results.items():
        for task_id, entry in by_task.items():
            if entry["status"] in SCORED_STATUSES and entry.get("predictions"):
                try:
                    evaluation = load_predictions(Path(entry["predictions"]), benches[task_id],
                                                  expect=entry.get("predictions_digest"))
                except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                    missing.append({"version": label, "task": task_id, "reason": "predictions_unavailable"})
                else:
                    evaluations.setdefault(label, {})[task_id] = (evaluation,)
            else:
                missing.append({"version": label, "task": task_id, "reason": "unscored_episode"})
    protocols = {results[label][task].get("judge_protocol") for label, tasks_ in evaluations.items()
                 for task in tasks_}
    if len(protocols) > 1:
        missing.extend({"version": label, "task": task, "reason": "mixed_judge_protocols"}
                       for label, tasks_ in evaluations.items() for task in tasks_
                       if results[label][task].get("judge_protocol") != JUDGE_PROTOCOL)
    body = {"tasks": [task.id for task in tasks],
            "versions": {label: {"digest": harness, "episodes": results[label]} for label, harness in version_list},
            "judge_protocol": next(iter(protocols)) if len(protocols) == 1 else None,
            "means": _means(results),
            "decision": None if missing else _compare(evaluations, suite.gates, {"holdout": run_id}),
            "missing": missing, "complete": not missing}
    with Ledger(ledger_path) as ledger:
        if recorded and body == {key: value for key, value in recorded.items() if key != "at"}:
            return recorded
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


def _frozen_validation(ledger: Ledger, run_id: str) -> dict[tuple[str, str], list[str]]:
    """The first plan this run opened for a harness and task, including plans still waiting for programs."""

    frozen = {}
    for event in ledger.events(run_id, "sealed_task"):
        body = event.body
        if (body.get("stage") == "validation" and body.get("judge_protocol") == JUDGE_PROTOCOL
                and (plan := body.get("plan"))):
            frozen.setdefault((body["harness"], body["task"]), plan["episodes"])
    return frozen


def sealed_plan(ledger: Ledger, view: dict[str, Any], suite: Any, stage: str
                ) -> tuple[list[Look], list[dict[str, Any]]]:
    """What a run's verification on ``stage`` judges, fixed before anything is judged: every version (seed and
    final) on every task of the suite. Use its first plan if it already opened a training task, or the programs
    in its latest finished round; a version some run already judged on ``validation`` under the same conditions
    keeps that look and its programs (``_first_looks``). A holdout task uses its holdout episode. On
    ``final_test``, use the programs judged on ``validation``. The (version, task) pairs without programs are
    listed with why: ``no_program`` (no scored episode on a training task: it will never have one),
    ``awaiting_holdout`` (holdout not verified, or not scored there yet) and ``awaiting_validation``."""

    holdout = view["verification"]["holdout"] or {}
    validation = view["verification"]["sealed"].get("validation") or {}
    config = EvolveConfig.from_dict(view["config"])
    expected = holdout_conditions(config)
    train = [task.id for task in suite.tasks("train")]
    frozen = _frozen_validation(ledger, view["run_id"]) if stage == "validation" else {}
    wanted, missing = [], []
    for label, harness in versions(view):
        trained = latest_programs(view, harness)
        for task in [*train, *(task.id for task in suite.tasks("holdout"))]:
            if stage == "final_test":
                result = ((validation.get("versions") or {}).get(label, {}).get("tasks") or {}).get(task)
                episodes, reason = program_episodes(result), "awaiting_validation"
            elif task in train:
                episodes, reason = frozen.get((harness, task), trained.get(task, [])), "no_program"
            else:
                entry = ((holdout.get("versions") or {}).get(label, {}).get("episodes") or {}).get(task) or {}
                scored = entry.get("status") in SCORED_STATUSES and entry.get("episode")
                episodes, reason = ([entry["episode"]] if scored else []), "awaiting_holdout"
                if episodes and not _holdout_record(ledger, entry["episode"], harness, task, expected):
                    # a holdout record from before holdout exposures named their conditions may hold an episode
                    # of another model or budget (2026-11-01: the seed's 4-minute episode on one holdout task)
                    episodes, reason = [], "holdout_conditions_differ"
            if episodes:
                wanted.append((label, harness, task, episodes))
            else:
                missing.append({"version": label, "task": task, "reason": reason})
    looks, unresolved = _looks(ledger, wanted)
    if stage == "validation":  # final_test takes the programs validation judged
        looks, replaced = _first_looks(ledger, looks, stage)
    else:
        replaced = set()
    missing.extend(_short(looks, set(train), config.repeats, replaced))
    return looks, [*missing, *unresolved]


def _short(looks: Sequence[Look], train: set[str], repeats: int,
           replaced: set[tuple[str, str]]) -> list[dict[str, Any]]:
    """The training-task looks with fewer programs than the run's replicates, as ``missing_replicates``: one
    program stands for whichever replicate happened to score, so such a look neither completes a plan nor opens a
    new look at a sealed stage (``reuse_only``). A look recorded before (``replaced``) counts the same way: a
    single-program result from an earlier run cannot complete a plan that asks for every replicate."""

    return [{"version": look.label, "task": look.task, "reason": "missing_replicates", "episodes": list(look.episodes),
             "expected": repeats, **({"recorded_before": True} if (look.label, look.task) in replaced else {})}
            for look in looks if look.task in train and len(set(look.episodes)) < repeats]


def reuse_only(missing: Sequence[dict[str, Any]]) -> frozenset[tuple[str, str]]:
    """(version, task) cells whose looks may show a recorded result but must not be judged anew."""

    return frozenset((item["version"], item["task"]) for item in missing if item.get("reason") == "missing_replicates")


def _ran_under(conditions: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Whether an episode's conditions are a run's (``holdout_conditions``: its model counts only when pinned)."""

    return (conditions["minutes"] == expected["minutes"] and conditions["tokens"] == expected["tokens"]
            and conditions.get("local_selection") == expected.get("local_selection")
            and (expected["model"] is None or conditions["model"] == expected["model"]))


def _matches(ledger: Ledger, result: dict[str, Any], look: Look) -> bool:
    if _result_conditions(ledger, result) != look.conditions:
        return False
    planned = result.get("plan")
    if planned is not None:
        return (set(planned["episodes"]) == set(look.episodes)
                and planned["program_digests"] == look.program_digests
                and planned.get("bench_identity") == look.bench_identity)
    # Legacy looks can be reused when the ledger proves they measured exactly the intended programs.
    return set(program_episodes(result)) == set(look.episodes) and all(
        ((_episode(ledger, episode) or {}).get("scored") or {}).get("digest") == program
        for episode, program in look.program_digests.items())


def gaps(ledger: Ledger, body: dict[str, Any] | None, looks: Sequence[Look]) -> list[Look]:
    """The planned looks a recorded verification lacks, or holds under other conditions (results recorded
    before looks named their conditions, judged on programs of other runs): what verifying again would add."""

    held = {(label, task): result for label, version in ((body or {}).get("versions") or {}).items()
            for task, result in (version.get("tasks") or {}).items() if result}
    return [look for look in looks
            if not (result := held.get((look.label, look.task))) or not _matches(ledger, result, look)
            or result.get("judge_protocol") != JUDGE_PROTOCOL or set(program_episodes(result)) != set(look.episodes)
            or not all(_prediction_intact(entry) for entry in result.get("programs") or [result])]


def verify_sealed(ledger_path: Path, run_id: str, stage: str, *,
                  cancel: threading.Event | None = None) -> dict[str, Any]:
    """Setting ``cancel`` stops between tasks or inside a judge; finished tasks stay recorded.

    A recorded verification of the stage is returned as it is unless the plan (``sealed_plan``) now holds looks
    it lacks; then the stage is verified again, reusing every look judged before, and recorded anew."""

    if stage not in SEALED_STAGES:
        raise ValueError(f"{stage} is not a sealed stage")
    view = _finished_view(ledger_path, run_id)
    config = EvolveConfig.from_dict(view["config"])
    suite = suite_spec(config.suite)
    if suite.digest != view["suite"]["digest"]:
        raise ValueError("the suite differs from the one this run used")
    tag = run_id.partition(":")[2] or run_id
    base = ecogym_home() / "verifications" / tag
    with _stage_lock(base / f"{stage}.lock", run_id, stage):
        with Ledger(ledger_path) as ledger:
            view = fold(ledger.events(run_id))
            if stage == "final_test":
                validation = view["verification"]["sealed"].get("validation")
                ahead, missing = sealed_plan(ledger, view, suite, "validation")
                if (validation is None or validation.get("complete") is False or missing
                        or gaps(ledger, validation, ahead)):
                    raise PermissionError("complete the full validation plan before final_test "
                                          "(ecogym evolve verify RUN --sealed validation)")
            looks, missing = sealed_plan(ledger, view, suite, stage)
            recorded = view["verification"]["sealed"].get(stage)
            if (recorded and recorded.get("complete") is not False and not missing
                    and not gaps(ledger, recorded, looks)):
                return recorded
        results, evaluations, unavailable = _seal_locked(
            ledger_path, run_id, stage, looks, limits=config.limits, store=base / stage,
            filename=lambda look, index: f"{look.label}-{look.task.replace(':', '_')}-{index}.npz", cancel=cancel,
            reuse_only=reuse_only(missing))
        missing = [*missing, *unavailable]
        results = {label: results.get(label, {}) for label, _ in versions(view)}
        with Ledger(ledger_path) as ledger:
            body = {"stage": stage, "suite": suite.digest,
                    "tasks": [task.id for task in [*suite.tasks("train"), *suite.tasks("holdout")]],
                    "plan": [look.plan() for look in looks], "scope": "fixed_programs",
                    "judge_protocol": JUDGE_PROTOCOL,
                    "versions": {label: {"digest": harness, "tasks": results[label]}
                                 for label, harness in versions(view)},
                    "means": _means(results),
                    "decision": None if missing else _compare(evaluations, suite.gates, {"sealed": run_id,
                                                                                        "stage": stage}),
                    "missing": missing, "complete": not missing,
                    "methods_exposed_on_stage": len({item["method"] for item in ledger.exposures()
                                                     if item["stage"] == stage})}
            if recorded and body == {key: value for key, value in recorded.items() if key != "at"}:
                return recorded
            ledger.append(run_id, "sealed_verified", body)
    return body


def _seal(ledger_path: Path, run_id: str, stage: str, looks: Sequence[Look], *, limits: Any, store: Path,
          filename: Callable[[Look, int], str], cancel: threading.Event | None = None,
          reuse_only: frozenset[tuple[str, str]] = frozenset()
          ) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, tuple[Evaluation, ...]]],
                     list[dict[str, Any]]]:
    """Judge ``looks`` on a sealed stage, once per (version, conditions, task, stage) across all runs: a result
    recorded before, by any run, can be reused if its plan matches. Each program result is recorded as soon as
    it is judged (``sealed_task``). The stage lock serializes work in the same run. Results and evaluations by
    label and task, and the unresolved cells of the plan. The (label, task) cells in ``reuse_only`` show a result
    recorded before but are never judged: no look at the stage is opened for them."""

    tag = run_id.partition(":")[2] or run_id
    with _stage_lock(ecogym_home() / "verifications" / tag / f"{stage}.lock", run_id, stage):
        return _seal_locked(ledger_path, run_id, stage, looks, limits=limits, store=store, filename=filename,
                            cancel=cancel, reuse_only=reuse_only)


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


def _reused(ledger: Ledger, judged: dict[str, dict[str, Any]], bench: Bench, look: Look,
            stage: str) -> dict[str, Any] | None:
    """The recorded result that stands for ``look``: one under its key, or one recorded before looks named their
    conditions whose program's episode ran under the same conditions; None when the look was never taken."""

    found = judged.get(look_key(bench, look, stage))
    if found is not None:
        return found
    old = judged.get(exposure_key(bench, look.harness, stage, SEALED))
    if old is not None and _result_conditions(ledger, old) == look.conditions:
        return old
    return None


def _evaluations(result: dict[str, Any], bench: Bench, data: Any) -> tuple[Evaluation, ...]:
    programs = result.get("programs") or [result]
    return tuple(load_predictions(Path(item["predictions"]), bench, data, expect=item.get("predictions_digest"))
                 for item in programs)


def _extract(ledger_path: Path, look: Look, scratch: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """Read back every program of a look (checked against the digest it was scored with), before the look is
    claimed: (episode, program directory) of those that could be read, and why the others could not."""

    found, problems = [], []
    for index, episode in enumerate(look.episodes):
        with Ledger(ledger_path) as ledger:
            record = _episode(ledger, episode) or {}
        scored = record.get("scored") or {}
        trajectory = Path(((record.get("finished") or {}).get("trajectory") or {}).get("dir", ""))
        try:
            if episode in look.program_digests and look.program_digests[episode] != scored.get("digest"):
                raise ProgramMismatch("the ledger no longer identifies the frozen program")
            found.append((episode, extract_submission(trajectory, scored, scratch / f"program-{index}")))
        except (ProgramMismatch, OSError, EOFError, tarfile.TarError) as error:
            problems.append(f"{episode}: {type(error).__name__}: {error}"[:300])
    return found, problems


def _seal_locked(ledger_path: Path, run_id: str, stage: str, looks: Sequence[Look], *, limits: Any, store: Path,
                 filename: Callable[[Look, int], str], cancel: threading.Event | None = None,
                 reuse_only: frozenset[tuple[str, str]] = frozenset()
                 ) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, tuple[Evaluation, ...]]],
                            list[dict[str, Any]]]:
    results: dict[str, dict[str, dict[str, Any]]] = {look.label: {} for look in looks}
    evaluations: dict[str, dict[str, tuple[Evaluation, ...]]] = {}
    unavailable: list[dict[str, Any]] = []
    if not looks:
        return results, evaluations, unavailable

    runtime = None
    benches: dict[str, Bench] = {}
    for look in sorted(looks, key=lambda item: (item.label, item.task)):
        if cancel is not None and cancel.is_set():
            raise Cancelled("the verification was cancelled")
        bench = _open(look.task, benches)
        cell = {"version": look.label, "task": look.task}
        if look.bench_identity and look.bench_identity != bench.digest:
            unavailable.append({**cell, "reason": "bench_identity_changed"})
            continue
        key = look_key(bench, look, stage)
        access = SealedAccess(stage, key)
        with Ledger(ledger_path) as ledger:
            found = _reused(ledger, _judged(ledger), bench, look, stage)
            matches = found is None or _matches(ledger, found, look)
        if not matches:
            unavailable.append({**cell, "reason": "prior_exposure_differs", "source_run": found["run_id"],
                                "expected_episodes": list(look.episodes), "judged_episodes": program_episodes(found)})
            continue
        if found is not None and found.get("judge_protocol") != JUDGE_PROTOCOL:
            unavailable.append({**cell, "reason": "prior_judge_protocol_differs", "source_run": found["run_id"],
                                "judge_protocol": found.get("judge_protocol")})
            continue
        recorded_programs = (found.get("programs") or [found]) if found else []
        judged = {item["episode"]: item for item in recorded_programs if item.get("episode")}
        remaining = tuple(episode for episode in look.episodes if episode not in judged)
        if remaining and (look.label, look.task) in reuse_only:
            continue  # too few programs for the plan (listed as missing_replicates): no new look at the stage
        if remaining and found is not None and found["run_id"] != run_id:
            unavailable.append({**cell, "reason": "pending_in_source_run", "source_run": found["run_id"],
                                "episodes": list(remaining)})
        elif remaining:
            scratch = scratch_root() / f"seal-{uuid.uuid4().hex[:12]}"
            try:
                programs, problems = _extract(ledger_path, replace(look, episodes=remaining), scratch)
                missing = [episode for episode in remaining if episode not in {name for name, _ in programs}]
                if missing:
                    unavailable.append({**cell, "reason": "program_unavailable",
                                        "episodes": missing,
                                        "detail": "; ".join(problems)[:1000]})
                if programs:
                    if runtime is None:
                        runtime = discover_runtime()
                    with Ledger(ledger_path) as ledger:
                        if not ledger.claim_exposure(key, task=look.task, stage=stage, method=look.harness,
                                                     run_id=run_id):
                            claim = next((item for item in ledger.exposures(task=look.task) if item["key"] == key), {})
                            if claim.get("run_id") != run_id:
                                raise PermissionError(f"{look.label} {look.harness[:12]} on {look.task} was claimed "
                                                      f"for {stage} by {claim.get('run_id')}; complete that run first")
                        if found is None:
                            # Freeze the entire look before judging its first program.
                            _record_look(ledger, run_id, stage, look, key, judged)
                    for episode, program in programs:
                        report, evaluation = judge_program(bench, program, stage, runtime, limits=limits,
                                                           sealed=access, cancel=cancel)
                        path = store / key / filename(look, look.episodes.index(episode))
                        path.parent.mkdir(parents=True, exist_ok=True)
                        save_predictions(path, bench, evaluation)
                        judged[episode] = {"episode": episode, "program_digest": look.program_digests.get(episode),
                                           "status": report["status"], "score": report["score"],
                                           "interval": report["interval"], "coverage": report["coverage"],
                                           "origins": report["origins"], "cells": report["cells"],
                                           "predictions": str(path), "predictions_digest": file_digest(path)}
                        with Ledger(ledger_path) as ledger:
                            _record_look(ledger, run_id, stage, look, key, judged)
            finally:
                subprocess.run(["rm", "-r", "-f", str(scratch)], check=False)
        if not judged:
            continue
        if remaining:
            result = _look_result(look, key, judged)
        else:
            result = {name: value for name, value in found.items()
                      if name not in ("stage", "version", "harness", "task", "run_id")}
            result["reused"] = True
            result["source_run"] = found["run_id"]
            result["judged_as"] = found.get("version")
            result["conditions"] = look.conditions
        results[look.label][look.task] = result
        if set(judged) == set(look.episodes):
            try:
                evaluations.setdefault(look.label, {})[look.task] = _evaluations(
                    result, bench, bench.stage(stage, sealed=access))
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                unavailable.append({**cell, "reason": "predictions_unavailable"})
    return results, evaluations, unavailable


def _look_result(look: Look, key: str, judged: dict[str, dict[str, Any]]) -> dict[str, Any]:
    programs = [judged[episode] for episode in look.episodes if episode in judged]
    result = {"plan": look.plan(), "exposure_key": key, "conditions": look.conditions, "judge_protocol": JUDGE_PROTOCOL,
              "episodes": [item["episode"] for item in programs], "programs": programs,
              "complete": len(programs) == len(look.episodes)}
    if not programs:
        return {**result, "status": "pending"}
    first = programs[0]
    return {**result, "episode": first["episode"], "origins": first["origins"],
            "status": next((item["status"] for item in programs if item["status"] != "ok"), "ok"),
            "score": float(np.mean([item["score"] for item in programs])),
            "coverage": float(np.mean([item["coverage"] for item in programs])),
            "interval": first["interval"] if len(programs) == 1 else None,
            "cells": first["cells"] if len(programs) == 1 else None,
            "predictions": first["predictions"], "predictions_digest": first["predictions_digest"]}


def _record_look(ledger: Ledger, run_id: str, stage: str, look: Look, key: str,
                 judged: dict[str, dict[str, Any]]) -> None:
    ledger.append(run_id, "sealed_task", {"stage": stage, "version": look.label, "harness": look.harness,
                                         "task": look.task, **_look_result(look, key, judged)})


ADOPTION_STAGE = "validation"  # final_test stays untouched until the run's own verification


def unchecked(view: dict[str, Any]) -> list[int]:
    """The generations whose adoption has no complete check."""

    done = {item.get("generation") for item in view.get("adoption_checks", [])
            if not item.get("error") and item.get("complete") is not False}
    return [item["generation"] for item in view["adoptions"] if item["generation"] not in done]


def adoption_sources(view: dict[str, Any], generation: int) -> tuple[str, str, dict[str, list[str]],
                                                                     dict[str, list[str]]] | None:
    """The version adopted in ``generation`` and the incumbent it replaced, with the programs to judge for each,
    by training task: every replicate of the adopted candidate in that round, and every replicate of the replaced
    incumbent in that round (in its latest earlier round, on a task where it has none there). None when the
    generation adopted nothing.

    A version judged on the stage before under the same conditions keeps that result (``_seal``): the incumbent
    replaced in a later generation was judged when it was adopted, so its check uses the programs of its own
    adoption."""

    last = {}
    for round_ in view["rounds"]:
        last[round_["generation"]] = round_
    round_ = last.get(generation)
    if not round_ or not round_.get("finished") or not round_.get("adopted"):
        return None
    adopted = round_["incumbent_after"]
    parent = round_["incumbent"]
    mine: dict[str, list[str]] = {}
    for entry in round_["episodes"]:
        if (entry.get("role") == "candidate" and entry.get("candidate") == round_["adopted"]
                and entry.get("status") in SCORED_STATUSES and entry.get("episode")):
            mine.setdefault(entry["task"], []).append(entry["episode"])
    return parent, adopted, latest_programs(view, parent, until=generation), mine


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
        frozen = _frozen_validation(ledger, run_id)
    if not events:
        raise KeyError(f"no evolution run {run_id}")
    view = fold(events)
    config = EvolveConfig.from_dict(view["config"])
    suite = suite_spec(config.suite)
    if suite.digest != view["suite"]["digest"]:
        raise ValueError("the suite differs from the one this run used")
    gates = suite.gates
    train = [task.id for task in suite.tasks("train")]
    wanted = [generation for generation in unchecked(view) if generations is None or generation in generations]
    tag = run_id.partition(":")[2] or run_id
    bodies = []
    for generation in wanted:
        found = adoption_sources(view, generation)
        if found is None:
            continue
        parent, adopted, theirs, mine = found
        for harness, side in ((parent, theirs), (adopted, mine)):
            side.update({task: frozen[(harness, task)] for task in train if (harness, task) in frozen})
        tasks = [task for task in train if theirs.get(task) and mine.get(task)]
        missing = [{"version": label, "task": task, "reason": "no_program"} for task in train
                   for label, side in (("parent", theirs), ("adopted", mine)) if not side.get(task)]
        body: dict[str, Any] = {"generation": generation, "digest": adopted, "parent": parent,
                                "stage": ADOPTION_STAGE, "tasks": tasks}
        try:
            with Ledger(ledger_path) as ledger:
                looks, unresolved = _looks(ledger, [(label, harness, task, side[task]) for task in tasks
                                                    for label, harness, side in (("parent", parent, theirs),
                                                                                 ("adopted", adopted, mine))])
                looks, replaced = _first_looks(ledger, looks, ADOPTION_STAGE)
            short = _short(looks, set(train), config.repeats, replaced)
            missing.extend(short)
            results, evaluations, unavailable = _seal(
                ledger_path, run_id, ADOPTION_STAGE, looks, limits=config.limits,
                store=ecogym_home() / "verifications" / tag / ADOPTION_STAGE / "adoptions",
                filename=lambda look, index: f"{look.harness[:12]}-{look.task.replace(':', '_')}-{index}.npz",
                cancel=cancel, reuse_only=reuse_only(short))
            missing = [*missing, *unresolved, *unavailable]
            frozen.update({(look.harness, look.task): list(look.episodes) for look in looks
                           if results.get(look.label, {}).get(look.task)})
            body.update({"plan": [look.plan() for look in looks], "scope": "fixed_programs",
                         "judge_protocol": JUDGE_PROTOCOL,
                         "versions": {"parent": {"digest": parent, "tasks": results.get("parent", {})},
                                      "adopted": {"digest": adopted, "tasks": results.get("adopted", {})}},
                         "means": _means(results),
                         "decision": None if missing else _compare(
                             evaluations, gates, {"adoption": run_id, "generation": generation},
                             labels=("adopted", "parent")),
                         "missing": missing, "complete": not missing})
        except Cancelled:
            raise
        except Exception as error:  # recorded, and tried again by the next check of this run
            body["error"] = f"{type(error).__name__}: {error}"[:1000]
        with Ledger(ledger_path) as ledger:
            ledger.append(run_id, "adoption_checked", body)
        bodies.append(body)
    return bodies


__all__ = ["ADOPTION_STAGE", "HOLDOUT", "SEALED", "Look", "adoption_sources", "check_adoptions", "complete",
           "conditions_of", "exposure_key", "gaps", "latest_programs", "look_key", "program_episodes", "sealed_plan",
           "unchecked", "verify_holdout", "verify_sealed", "versions"]
