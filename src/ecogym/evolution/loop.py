"""The outer loop: evolve an agent harness on a suite's training tasks.

Round 0 runs the starting harness on every training task (``repeats``
episodes each). Every later round:

1. an evolver session reviews the evidence of all earlier rounds and writes
   up to ``candidates`` candidate harnesses (``ecogym.evolution.evolver``);
   one that behaves like the incumbent or like a candidate compared before
   (``behavior_digest``) is not run;
2. every valid candidate and the incumbent run ``repeats`` episodes on every
   training task with the same budget, queued task by task so no version
   starts later than another, and with the same expert guidance: what the
   ledger held when the round started;
3. each candidate is compared with the incumbent's unbiased episodes (those it
   ran as the incumbent under the same guidance, never the episodes that got
   it adopted) under the suite's gates (``ecogym.evaluation.suite``), with
   the suite interval Bonferroni-adjusted for the number of candidates;
4. the passing candidate with the largest gain becomes the incumbent.

An episode whose agent never got a model reply is retried (``retries``) and
otherwise left out: service failures are not harness effects. A candidate
missing usable episodes on any training task is not adopted that round.
Failed candidates stay in the record; later evolvers read them in
``archive.md``. All loop state is recomputed from the ledger at every round,
so ``Evolution.run`` continues an interrupted run after its last finished
round. ``round_started`` names the process that runs it (``owner``), so
another process can tell a live run from a dead one. Holdout tasks and
sealed stages are touched only by ``ecogym.evolution.verify``.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__
from ..episodes import expert, providers
from ..episodes.agent import Cancelled
from ..episodes.judge import Limits, load_predictions, reference_scores
from ..episodes.runner import EpisodeConfig, run_episode
from ..episodes.workspace import LOCAL_STAGE, SCORED_STAGE
from ..evaluation.harness import Bench, Evaluation
from ..evaluation.suite import TaskPair, adjusted_confidence, compare_suite
from ..harness import Harness, behavior_digest, resolve
from ..kernel.canonical import digest
from ..kernel.ledger import Ledger
from ..task.spec import TaskSpec
from ..task.suite import suite_spec
from .evolver import EvolverSettings, run_evolver
from .fold import EVENT_KINDS, SCORED_STATUSES, fold

PATCH_LIMIT = 60_000  # characters of unified diff kept per candidate in the ledger


class RunStopped(RuntimeError):
    """The run was asked to stop between rounds."""


class ServiceDown(RuntimeError):
    """No model replies: every episode of a round, or the evolver, got none. The run fails and can be resumed."""


@dataclass(frozen=True, slots=True)
class EvolveConfig:
    suite: str
    harness: str = "seed"
    generations: int = 3
    candidates: int = 2
    repeats: int = 1
    minutes: float = 20.0
    tokens: int = 3_000_000
    model: str | None = None  # pin the inner agents' model (<provider>/<model>); None lets each harness choose
    evolver_minutes: float = 30.0
    evolver_tokens: int = 4_000_000
    evolver_model: str = "deepseek-v4-pro"
    concurrency: int = 3
    retries: int = 1
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self) -> None:
        if self.generations < 1 or self.repeats < 1 or self.concurrency < 1 or self.retries < 0:
            raise ValueError("generations, repeats and concurrency must be positive, retries non-negative")
        if not 1 <= self.candidates <= 6:
            raise ValueError("candidates must lie in 1..6")
        if self.minutes <= 0 or self.evolver_minutes <= 0 or self.tokens <= 0 or self.evolver_tokens <= 0:
            raise ValueError("budgets must be positive")
        for ref in (self.model, self.evolver_model):
            if ref:
                providers.split(ref)  # <provider>/<model> or a DeepSeek id; ModelError is a ValueError

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "ecogym.evolve-config/1", **asdict(self)}

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> EvolveConfig:
        values = {key: value for key, value in body.items() if key != "schema"}
        values["limits"] = Limits(**values.get("limits", {}))
        return cls(**values)

    def episodes_per_round(self, tasks: int) -> tuple[int, int]:
        """Episodes in round 0 and in each later round with every candidate valid."""

        return tasks * self.repeats, tasks * self.repeats * (self.candidates + 1)


@dataclass(slots=True)
class State:
    generation: int  # the next round to run
    incumbent: str
    pool: dict[str, list[dict[str, Any]]]  # task -> the incumbent's unbiased episodes
    tried: set[str]  # behavior digests of the seed and of every candidate compared in a finished round


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The expert guidance of one round: ledger events up to ``until``, and its digest per training task."""

    until: int
    guidance: dict[str, str | None]


def usable(entry: dict[str, Any]) -> bool:
    return entry.get("status") in SCORED_STATUSES and bool(entry.get("predictions"))


def owner() -> dict[str, Any]:
    """The process that runs a round."""

    return {"pid": os.getpid(), "host": socket.gethostname()}


def _guidance_digest(entries: list[dict[str, Any]]) -> str | None:
    return digest({"schema": "ecogym.expert-guidance/1", "entries": entries}) if entries else None


def _require_replies(entries: list[dict[str, Any]]) -> None:
    """Stop the run (resumably) when no episode of a round ran: no model reply (the provider is down or
    refusing) or a host error, after retries; the reasons say which."""

    if entries and not any(entry.get("status") != "infra_failed" and entry.get("status") != "error"
                           for entry in entries):
        reasons = sorted({entry.get("upstream_error") or entry.get("error") or entry.get("status", "")
                          for entry in entries})
        raise ServiceDown("no episode of this round ran (no model reply or a host error, after retries): "
                          + "; ".join(reasons)[:1500])


class Evolution:
    def __init__(self, config: EvolveConfig, ledger_path: Path, *,
                 on_event: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        """``on_event`` sees every logged event and an ``episode`` notice per finished episode (progress only)."""

        self.config = config
        self.ledger_path = Path(ledger_path)
        self.suite = suite_spec(config.suite)
        self.tasks: list[TaskSpec] = self.suite.tasks("train")
        self.benches: dict[str, Bench] = {}
        self.cancel = threading.Event()
        self.on_event = on_event
        self._evaluations: dict[str, Evaluation] = {}
        self._notify_lock = threading.Lock()

    # ---- plumbing ---------------------------------------------------------------------------------------------

    def _notify(self, kind: str, body: dict[str, Any]) -> None:
        if self.on_event is not None:
            with self._notify_lock, contextlib.suppress(Exception):  # progress display must never break a run
                self.on_event(kind, body)

    def _log(self, run_id: str, kind: str, body: dict[str, Any]) -> None:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind}")
        with Ledger(self.ledger_path) as ledger:
            ledger.append(run_id, kind, body)
        self._notify(kind, body)

    def view(self, run_id: str) -> dict[str, Any]:
        with Ledger(self.ledger_path) as ledger:
            return fold(ledger.events(run_id))

    def _open_benches(self) -> None:
        """Open every training bench in this thread and fill the caches episodes read concurrently."""

        for task in self.tasks:
            if task.id not in self.benches:
                bench = Bench.open(task)
                for stage in (LOCAL_STAGE, SCORED_STAGE):
                    bench.stage(stage)
                    reference_scores(bench, stage)
                self.benches[task.id] = bench

    def _evaluation(self, entry: dict[str, Any]) -> Evaluation:
        if entry["episode"] not in self._evaluations:
            self._evaluations[entry["episode"]] = load_predictions(
                Path(entry["predictions"]), self.benches[entry["task"]], expect=entry.get("predictions_digest"))
        return self._evaluations[entry["episode"]]

    # ---- the run ----------------------------------------------------------------------------------------------

    def start(self, *, label: str = "", run_id: str | None = None) -> str:
        run_id = run_id or f"evo:{uuid.uuid4().hex[:12]}"
        seed = resolve(self.config.harness)
        for ref in dict.fromkeys((self.config.model or seed.agent.model, self.config.evolver_model)):
            providers.check(ref)  # route, model and key: refuse a run whose agents could not get a reply
            providers.probe(ref)  # and one tiny request: settings may list a model the key cannot use
        self._log(run_id, "evolve_started", {
            "label": label, "config": self.config.to_dict(), "ecogym": __version__,
            "suite": {"id": self.suite.id, "title": self.suite.title, "digest": self.suite.digest,
                      "train": list(self.suite.train), "holdout_tasks": len(self.suite.holdout)},
            "gates": asdict(self.suite.gates),
            "seed": {"digest": seed.digest, "name": seed.name, "summary": seed.summary,
                     "behavior": behavior_digest(seed)},
        })
        return run_id

    def run(self, run_id: str, *, stop: Callable[[], bool] | None = None) -> dict[str, Any]:
        """Run, or continue, every remaining round; ``stop`` is polled between rounds."""

        started = time.monotonic()
        try:
            self._open_benches()
            while True:
                view = self.view(run_id)
                if view["status"] == "finished":
                    return view["result"]
                if view["suite"]["digest"] != self.suite.digest:
                    raise ValueError("the suite differs from the one this run started with")
                state = self.state(view)
                if state.generation > self.config.generations:
                    break
                if stop is not None and stop():
                    raise RunStopped(f"stopped before round {state.generation}")
                self._round(run_id, view, state)
        except (RunStopped, Cancelled) as error:  # asked to stop: resumable from the last finished round
            self.cancel.set()
            self._log(run_id, "evolve_failed", {"error": str(error), "stopped": True})
            raise
        except BaseException as error:
            self.cancel.set()
            self._log(run_id, "evolve_failed", {"error": f"{type(error).__name__}: {error}"[:2000]})
            raise
        view = self.view(run_id)
        episodes = [item for round_ in view["rounds"] for item in round_["episodes"]]
        evolvers = [round_["evolver"] for round_ in view["rounds"] if round_.get("evolver")]
        result = {
            "seed": view["seed"]["digest"], "final": view["incumbent"], "adoptions": len(view["adoptions"]),
            "rounds": sum(1 for round_ in view["rounds"] if round_["finished"]), "episodes": len(episodes),
            "tokens": sum(item.get("tokens") or 0 for item in [*episodes, *evolvers]),
            "metered_tokens": sum(item.get("metered_tokens") or 0 for item in [*episodes, *evolvers]),
            "seconds": round(time.monotonic() - started, 1),
        }
        self._log(run_id, "evolve_finished", result)
        return result

    def state(self, view: dict[str, Any]) -> State:
        finished = [round_ for round_ in view["rounds"] if round_["finished"]]
        generation = finished[-1]["generation"] + 1 if finished else 0
        incumbent = finished[-1]["incumbent_after"] if finished else view["seed"]["digest"]
        adopted_at = max((item["generation"] for item in view["adoptions"] if item["digest"] == incumbent),
                         default=-1)
        pool: dict[str, list[dict[str, Any]]] = {}
        for round_ in finished:
            if round_["generation"] <= adopted_at:
                continue
            for entry in round_["episodes"]:
                if entry["role"] == "incumbent" and entry["harness"] == incumbent and usable(entry):
                    pool.setdefault(entry["task"], []).append(entry)
        tried = {view["seed"].get("behavior") or view["seed"]["digest"]}
        for round_ in finished:
            for candidate in round_["candidates"]:
                decision = candidate.get("decision")
                if decision and not str(decision.get("reason", "")).startswith("missing_tasks"):
                    tried.add(candidate.get("behavior") or candidate["digest"])
        return State(generation, incumbent, pool, tried)

    def _round(self, run_id: str, view: dict[str, Any], state: State) -> None:
        generation = state.generation
        incumbent = resolve(state.incumbent)
        with Ledger(self.ledger_path) as ledger:
            until = ledger.last_seq()
            snapshot = Snapshot(until, {task.id: _guidance_digest(expert.guidance(ledger, task, until=until))
                                        for task in self.tasks})
        self._log(run_id, "round_started", {"generation": generation, "incumbent": incumbent.digest,
                                            "owner": owner(), "expert_until": until, "guidance": snapshot.guidance})
        if generation == 0:
            entries = self._episodes(run_id, generation, [(incumbent, "incumbent", None)], snapshot)
            self._log(run_id, "episodes_finished", {"generation": 0, "episodes": entries})
            _require_replies(entries)
            self._log(run_id, "round_finished", {"generation": 0, "incumbent": incumbent.digest, "adopted": None,
                                                 "outcome": "baseline"})
            return
        candidates = self._propose(run_id, generation, incumbent, view, state, snapshot)
        if not candidates:
            self._log(run_id, "round_finished", {"generation": generation, "incumbent": incumbent.digest,
                                                 "adopted": None, "outcome": "no_candidates"})
            return
        jobs = [(incumbent, "incumbent", None)] + [(item["harness"], "candidate", item["name"]) for item in candidates]
        entries = self._episodes(run_id, generation, jobs, snapshot)
        self._log(run_id, "episodes_finished", {"generation": generation, "episodes": entries})
        _require_replies(entries)
        # earlier incumbent episodes count only if they saw this round's guidance; otherwise only today's do
        pool = {task: [entry for entry in items if entry.get("guidance") == snapshot.guidance.get(task)]
                for task, items in state.pool.items()}
        for entry in entries:
            if entry["role"] == "incumbent" and usable(entry):
                pool.setdefault(entry["task"], []).append(entry)
        confidence = adjusted_confidence(self.suite.gates.confidence, len(candidates))
        passing = []
        for candidate in candidates:
            pairs, missing = [], []
            for task in self.tasks:
                own = [entry for entry in entries if entry["role"] == "candidate" and entry["task"] == task.id
                       and entry["candidate"] == candidate["name"] and usable(entry)]
                theirs = pool.get(task.id, [])
                if not own or not theirs:
                    missing.append(task.id)
                    continue
                pairs.append(TaskPair(task.id, tuple(self._evaluation(entry) for entry in own),
                                      tuple(self._evaluation(entry) for entry in theirs), group=task.dataset))
            body: dict[str, Any] = {"generation": generation, "name": candidate["name"], "digest": candidate["digest"],
                                    "missing": missing}
            if pairs:
                decision = compare_suite(pairs, self.suite.gates, {"run": run_id, "generation": generation,
                                                                   "candidate": candidate["name"]},
                                         confidence=confidence).to_dict()
                if missing and decision["adopted"]:
                    decision["adopted"], decision["reason"] = False, "missing_tasks:" + ",".join(missing)
            else:
                decision = {"adopted": False, "reason": "missing_tasks:" + ",".join(missing), "delta": None,
                            "interval": [None, None], "confidence": confidence, "tasks": [], "cells": {}}
            body["decision"] = decision
            self._log(run_id, "candidate_compared", body)
            if decision["adopted"]:
                passing.append((decision, candidate))
        adopted = None
        if passing:
            decision, adopted = max(passing, key=lambda item: item[0]["delta"])
            self._log(run_id, "harness_adopted", {"generation": generation, "name": adopted["name"],
                                                  "digest": adopted["digest"], "delta": decision["delta"],
                                                  "interval": decision["interval"], "parent": incumbent.digest,
                                                  "passing": len(passing)})
        self._log(run_id, "round_finished", {
            "generation": generation, "incumbent": adopted["digest"] if adopted else incumbent.digest,
            "adopted": adopted["name"] if adopted else None, "outcome": "adopted" if adopted else "kept",
        })

    def _propose(self, run_id: str, generation: int, incumbent: Harness, view: dict[str, Any],
                 state: State, snapshot: Snapshot) -> list[dict[str, Any]]:
        config = self.config
        settings = EvolverSettings(minutes=config.evolver_minutes, tokens=config.evolver_tokens,
                                   model=config.evolver_model, candidates=config.candidates)
        with Ledger(self.ledger_path) as ledger:  # the training tasks' guidance only: never a holdout task's
            advice = expert.guidance(ledger, self.tasks, until=snapshot.until)
        try:
            outcome = run_evolver(run_id=run_id, generation=generation, incumbent=incumbent, suite=self.suite,
                                  tasks=self.tasks, benches=self.benches, view=view, ledger_path=self.ledger_path,
                                  settings=settings, repeats=config.repeats, episode_minutes=config.minutes,
                                  episode_tokens=config.tokens, model=config.model, limits=config.limits,
                                  expert_text=expert.render(advice) if advice else "", cancel=self.cancel)
        except (Cancelled, KeyboardInterrupt):
            raise
        except Exception as error:  # a failed review costs one round, not the run
            self._log(run_id, "evolver_finished", {"generation": generation, "status": "error",
                                                   "error": f"{type(error).__name__}: {error}"[:2000]})
            return []
        session = outcome["session"]
        if session["status"] == "infra_failed":  # no reply at all: later rounds would fail the same way
            self._log(run_id, "evolver_finished", {
                "generation": generation, **{key: session.get(key) for key in (
                    "status", "seconds", "tokens", "metered_tokens", "llm_calls", "answered_calls", "failed_calls",
                    "trajectory", "upstream_error")}, "candidates": []})
            raise ServiceDown(f"the evolver got no model reply: {session.get('upstream_error') or 'service failure'}")
        found = []
        seen = state.tried | {behavior_digest(incumbent)}
        for candidate in outcome["candidates"]:
            behavior = behavior_digest(candidate["harness"]) if candidate["status"] == "proposed" else None
            if behavior is not None and behavior in seen:
                candidate["status"], candidate["ok"] = "duplicate", False
                candidate["problems"] = [*candidate["problems"],
                                         "与在任版本或以往比较过的候选内容相同（harness.toml 的名称、摘要不算）"]
            elif behavior is not None:
                seen.add(behavior)
            patches, budget = [], PATCH_LIMIT
            for change in candidate.get("changes", []):
                patch = change["patch"][: max(budget, 0)]
                budget -= len(patch)
                patches.append({**change, "patch": patch})
            body = {"generation": generation, "name": candidate["name"], "status": candidate["status"],
                    "digest": candidate.get("digest"), "behavior": behavior, "parent": incumbent.digest,
                    "summary": candidate.get("summary", ""), "proposal": candidate.get("proposal", ""),
                    "agent": candidate.get("agent"), "changes": patches, "smoke": candidate.get("smoke"),
                    "problems": candidate.get("problems", [])}
            if candidate["status"] == "proposed":
                self._log(run_id, "candidate_proposed", body)
                found.append({"name": candidate["name"], "digest": candidate["digest"],
                              "harness": candidate["harness"]})
            else:
                self._log(run_id, "candidate_rejected", body)
        self._log(run_id, "evolver_finished", {
            "generation": generation,
            **{key: session.get(key) for key in ("status", "seconds", "tokens", "metered_tokens", "llm_calls",
                                                 "answered_calls", "failed_calls", "trajectory", "activity",
                                                 "upstream_error")},
            "final_text": session["final_text"][-4000:],
            "candidates": [{"name": item["name"], "status": item["status"]} for item in outcome["candidates"]],
        })
        return found

    def _episodes(self, run_id: str, generation: int, jobs: list[tuple[Harness, str, str | None]],
                  snapshot: Snapshot) -> list[dict[str, Any]]:
        # replicate by replicate, task by task, every version in turn: no version waits behind another's queue
        specs = [(harness, role, name, task, replicate) for replicate in range(self.config.repeats)
                 for task in self.tasks for harness, role, name in jobs]
        with ThreadPoolExecutor(max_workers=self.config.concurrency, thread_name_prefix="episode") as pool:
            futures = [pool.submit(self._episode, run_id, generation, *spec, snapshot) for spec in specs]
            try:
                return [future.result() for future in futures]
            except BaseException:
                self.cancel.set()
                for future in futures:
                    future.cancel()
                raise

    def _episode(self, run_id: str, generation: int, harness: Harness, role: str, name: str | None,
                 task: TaskSpec, replicate: int, snapshot: Snapshot) -> dict[str, Any]:
        entry: dict[str, Any] = {"task": task.id, "harness": harness.digest, "role": role, "candidate": name,
                                 "replicate": replicate, "guidance": snapshot.guidance[task.id]}
        for attempt in range(self.config.retries + 1):
            if self.cancel.is_set():
                raise Cancelled("the run was cancelled")
            label = f"{run_id} 第{generation}代 {'在任' if role == 'incumbent' else f'候选{name}'} {task.id}"
            config = EpisodeConfig(
                task=task.id, harness=harness.digest, minutes=self.config.minutes, tokens=self.config.tokens,
                model=self.config.model, label=label, limits=self.config.limits, expert_until=snapshot.until,
                context={"evolution": run_id, "generation": generation, "role": role, "candidate": name,
                         "replicate": replicate, "attempt": attempt},
            )
            try:
                result = run_episode(config, self.ledger_path, bench=self.benches[task.id], cancel=self.cancel)
            except (Cancelled, KeyboardInterrupt):
                raise
            except Exception as error:  # a broken episode is recorded and retried, never fatal to the run
                entry.update({"episode": None, "status": "error", "error": f"{type(error).__name__}: {error}"[:1000],
                              "attempt": attempt})
                continue
            entry.update({"episode": result["episode"], "status": result["status"], "score": result["score"],
                          "predictions": result["predictions"],
                          "predictions_digest": (result.get("scored") or {}).get("predictions_digest"),
                          "attempt": attempt, "tokens": result["tokens"], "metered_tokens": result["metered_tokens"],
                          "seconds": result["seconds"]})
            entry.pop("error", None)
            if result["status"] != "infra_failed":
                entry.pop("upstream_error", None)
                break
            entry["upstream_error"] = (result.get("upstream_error") or "")[:500]
        self._notify("episode", entry)
        return entry


__all__ = ["Evolution", "EvolveConfig", "RunStopped", "ServiceDown", "Snapshot", "State", "owner", "usable"]
