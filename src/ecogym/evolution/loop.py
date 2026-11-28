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
3. each candidate is compared with the incumbent's episodes from the same
   round under the suite's gates (``ecogym.evaluation.suite``), with
   the suite interval Bonferroni-adjusted for the number of candidates;
4. the passing candidate with the largest gain becomes the incumbent;
5. optionally (``check_adoptions``, off by default since 0.4.0), after an
   adoption, the programs of the new incumbent and of the version it replaced
   are judged again on the training tasks' sealed validation period
   (``verify.check_adoptions``): a run adopts small gains chosen on one hidden
   period, and this says how much of each holds on a period nobody saw.
   People read it; the evolver never does.

An episode whose agent never got a model reply is retried (``retries``) and
otherwise left out: service failures are not harness effects. Before a retry
the loop sends the model one test request (``ServiceGate``): when that gets no
answer either, the service is down, no session starts until it answers again
(for up to ``outage_minutes``), and the episode the outage cut short runs
again without using up its retries. When the test request is answered but the
episode still ended ``infra_failed`` (an unstable provider failing many calls),
it also runs again without using up its retries, up to ``SERVICE_RERUNS``
times. A candidate missing usable episodes on
any training task is not adopted that round. An evolver session that got no
reply, or that failed model calls kept from working for more than
``STALL_SHARE`` of its time (the builder's: generation 1's research session
has its own) while it left no candidate to compare, is reviewed
once more, after the outage; a second such session, or an outage longer than
``outage_minutes``, stops the run (``ServiceDown``, resumable, and resumed by
the service itself with ``auto_resume``): its generation runs again rather
than being spent on the outage.
Failed candidates stay in the record; later evolvers read them in
``archive.md``. All loop state is recomputed from the ledger at every round,
so ``Evolution.run`` continues an interrupted run after its last finished
round. Each round's plan (``round_planned``: candidates, incumbent, expert
guidance and the identity of the data, code and runtime) is recorded before
its episodes run, so a resumed round keeps its candidates and reuses its
verified episodes; when the data, code or runtime changed meanwhile, the same
candidates run again, every episode, under a new plan that names the one it
replaces. ``round_started`` names the process that runs it (``owner``), so
another process can tell a live run from a dead one (``stale``). A stopped
run can be ended where it is (``finish_run``), so its result can be verified.
Holdout tasks and sealed stages are touched only by
``ecogym.evolution.verify``.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..episodes import expert, providers
from ..episodes.agent import Cancelled
from ..episodes.governor import configure_judge_limit
from ..episodes.history import fold_episode
from ..episodes.judge import (
    JUDGE_PROTOCOL,
    Limits,
    calibration_pre_stage,
    file_digest,
    judge_program,
    load_predictions,
    reference_scores,
    save_predictions,
)
from ..episodes.mcp import ToolError
from ..episodes.runner import STALL_SHARE, EpisodeConfig, run_episode
from ..episodes.sandbox import discover_runtime, ecogym_home, scratch_root
from ..episodes.tools import DEFAULT_LOCAL_SELECTION, LOCAL_SELECTION_POLICIES, SINGLE_WINDOW_SELECTION, StartCache
from ..episodes.tools import snapshot as snapshot_program
from ..episodes.workspace import LOCAL_STAGE, SCORED_STAGE
from ..evaluation.harness import Bench, Evaluation
from ..evaluation.suite import TaskPair, adjusted_confidence, compare_suite
from ..harness import Harness, HarnessError, behavior_digest, resolve
from ..kernel import clock
from ..kernel.canonical import digest
from ..kernel.ledger import Ledger
from ..task.spec import TaskSpec
from ..task.suite import signed_rank_reachable, suite_spec
from .evolver import METHOD_RESEARCH_MODES, NAMES, EvolverSettings, run_evolver
from .fold import EVENT_KINDS, SCORED_STATUSES, fold
from .lifecycle import (
    OUTAGE_RERUNS,
    PROBE_EVERY,
    SERVICE_RERUNS,
    RunBusy,
    RunStopped,
    ServiceDown,
    ServiceGate,
    claim_resume,
    finish_run,
    model_key,
    owner,
    result_of,
    run_models,
    spent_by,
    stale,
)
from .relay import (
    effective_relay_batch,
    effective_relay_candidates,
    pick_stage_leader,
    rank_stage_candidates,
    render_relay_brief,
    select_finalists,
    select_stage_tasks,
)

PATCH_LIMIT = 60_000  # characters of unified diff kept per candidate in the ledger
PUBLIC_REFERENCE_SECONDS = 120.0  # shared by all tasks in this optional development diagnostic
# shared across all fixed replays in one round, split evenly between them. Scoring a program on model_selection
# takes 49 s at the median and 85 s at the 90th percentile (400 scorings in the ledger, 2026-11-17); with one worker,
# 120 s gave each replay less than the median once three were pending. Most rounds need few replays (an episode that
# kept its template already scored it), and the budget is a cap, not a wait
TEMPLATE_ATTRIBUTION_SECONDS = 600.0


_LEGACY_CONFIG_DEFAULTS = {
    "auto_resume": False,
    "method_research": "internal",
    "local_selection": SINGLE_WINDOW_SELECTION,
}


@dataclass(frozen=True, slots=True)
class EvolveConfig:
    suite: str
    harness: str = "seed"
    generations: int = 3
    candidates: int = 1
    repeats: int = 2  # one episode per version and task makes adoptions fragile (docs/evolution.md, section 16)
    minutes: float = 10.0
    tokens: int = 3_000_000
    model: str | None = None  # pin the inner agents' model (<provider>/<model>); None lets each harness choose
    evolver_minutes: float = 20.0
    evolver_tokens: int = 4_000_000
    evolver_model: str = "deepseek-v4-pro"
    concurrency: int = 4
    retries: int = 1
    check_adoptions: bool = False  # optional diagnostic on sealed validation, for people only
    outage_minutes: float = 120.0  # how long a round waits for the model service to answer again (0: not at all)
    auto_resume: bool = True  # the service continues the run by itself after it restarts or the service recovers
    limits: Limits = field(default_factory=Limits)
    method_research: str = "autonomous"
    method_registry: str | None = None
    local_selection: str = DEFAULT_LOCAL_SELECTION
    template_fast_track: bool = False  # evaluate template-only candidates deterministically without LLM episodes
    early_prune: bool = False  # prune clearly dominated candidates after replicate 0 (when repeats >= 2)
    relay_stages: int = 1  # 1: standard single-stage round; 2-3: intra-generation mini-batch relay stages
    relay_batch: int = 0  # tasks per relay stage (0: auto stratified ceil(tasks / relay_stages))
    relay_candidates: int = 0  # candidates explored per relay stage (0: auto min(4, max(2, candidates)))

    def __post_init__(self) -> None:
        if self.generations < 1 or self.repeats < 1 or self.concurrency < 1 or self.retries < 0:
            raise ValueError("generations, repeats and concurrency must be positive, retries non-negative")
        if not 1 <= self.candidates <= 6:
            raise ValueError("candidates must lie in 1..6")
        if not 1 <= self.relay_stages <= 3:
            raise ValueError("relay_stages must lie in 1..3")
        if self.relay_batch < 0:
            raise ValueError("relay_batch must be non-negative")
        if not 0 <= self.relay_candidates <= 6:
            raise ValueError("relay_candidates must lie in 0..6")
        if self.minutes <= 0 or self.evolver_minutes <= 0 or self.tokens <= 0 or self.evolver_tokens <= 0:
            raise ValueError("budgets must be positive")
        if not (math.isfinite(self.outage_minutes) and self.outage_minutes >= 0):
            raise ValueError("outage_minutes must be a finite number of minutes, 0 or more")
        if self.method_research not in METHOD_RESEARCH_MODES:
            raise ValueError(f"method_research must be one of {', '.join(METHOD_RESEARCH_MODES)}")
        if self.local_selection not in LOCAL_SELECTION_POLICIES:
            raise ValueError(f"local_selection must be one of {', '.join(LOCAL_SELECTION_POLICIES)}")
        if self.method_research == "supplied" and not self.method_registry:
            raise ValueError("supplied method research needs a host method registry")
        if self.method_registry:
            object.__setattr__(self, "method_registry", str(Path(self.method_registry).expanduser().resolve()))
        for ref in (self.model, self.evolver_model):
            if ref:
                providers.split(ref)  # <provider>/<model> or a DeepSeek id; ModelError is a ValueError

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "ecogym.evolve-config/1", **asdict(self)}

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> EvolveConfig:
        valid = {f.name for f in fields(cls)}
        values = {key: value for key, value in body.items() if key in valid}
        values["limits"] = Limits(**values.get("limits", {}))
        # Runs recorded before 0.3.2 keep their historical defaults when omitted from the ledger.
        for key, default in _LEGACY_CONFIG_DEFAULTS.items():
            values.setdefault(key, default)
        return cls(**values)

    def episodes_per_round(self, tasks: int) -> tuple[int, int]:
        """Episodes in round 0 and in each later round with every candidate valid."""

        later = tasks * self.repeats * (self.candidates + 1)
        if self.relay_stages > 1:
            batch = effective_relay_batch(tasks, total_stages=self.relay_stages, batch_size=self.relay_batch)
            relay_cands = effective_relay_candidates(self.candidates, self.relay_candidates)
            later += self.relay_stages * batch * relay_cands
        return tasks * self.repeats, later


@dataclass(slots=True)
class State:
    generation: int  # the next round to run
    incumbent: str
    tried: set[str]  # behavior digests of the seed and of every candidate compared in a finished round


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The expert guidance of one round: ledger events up to ``until``, and its digest per training task."""

    until: int
    guidance: dict[str, str | None]


def _slot(entry: dict[str, Any]) -> tuple[str, str, str | None, str, int]:
    return entry["harness"], entry["role"], entry.get("candidate"), entry["task"], entry["replicate"]


def usable(entry: dict[str, Any]) -> bool:
    return entry.get("status") in SCORED_STATUSES and bool(entry.get("predictions"))


def _scored_program(scored: dict[str, Any]) -> dict[str, Any]:
    return {"program_digest": scored.get("digest"), "scored_start": bool(scored.get("start")),
            "score_stage": scored.get("stage"), "judge_protocol": scored.get("judge_protocol"),
            "score_status": scored.get("status")}


def _builder_stalled(outcome: dict[str, Any]) -> float:
    """Seconds an evolver's builder waited on failed model calls. The round's session record also holds generation
    1's research session (``evolver._include_discovery``), which has its own time on top of the builder's
    (``evolver.DISCOVERY_SHARE``), so its waiting does not count against the builder's budget."""

    research = ((outcome.get("method_research") or {}).get("discovery") or {}).get("session") or {}
    stalled = float(outcome["session"].get("stalled_seconds") or 0.0)
    return max(0.0, stalled - float(research.get("stalled_seconds") or 0.0))


def _forecast_fingerprint(evaluation: Evaluation) -> str:
    predicted = np.array(evaluation.predicted, dtype="<f8", copy=True, order="C")
    predicted[np.isnan(predicted)] = np.nan
    predicted[predicted == 0] = 0.0
    return hashlib.sha256(predicted.tobytes()).hexdigest()


def _horizon_scores(evaluation: Evaluation) -> dict[str, float]:
    return {str(horizon): score for horizon, score in evaluation.horizon_scores().items()}


def _attribution_summary(arms: dict[str, Any], tasks: Sequence[TaskSpec]) -> dict[str, Any]:
    comparisons = {}
    for label, arm in arms.items():
        if label == "incumbent":
            continue
        matched, missing = {}, []
        for task in tasks:
            own, incumbent = arm["tasks"][task.id], arms["incumbent"]["tasks"][task.id]
            if own["episode_over_template"] is None or incumbent["episode_over_template"] is None:
                missing.append(task.id)
                continue
            matched[task.id] = {"template_gain": own["template"]["score"] - incumbent["template"]["score"],
                                "episode_over_template_gain": (own["episode_over_template"]
                                                               - incumbent["episode_over_template"]),
                                "episode_gain": own["episode_mean"] - incumbent["episode_mean"]}
            horizon_gains = {}
            for horizon, score in own.get("episode_horizons", {}).items():
                before = incumbent.get("episode_horizons", {}).get(horizon)
                prior = incumbent["template"].get("horizons", {}).get(horizon)
                next_score = own["template"].get("horizons", {}).get(horizon)
                if None not in (before, prior, next_score):
                    horizon_gains[horizon] = {
                        "episode_gain": score - before, "template_gain": next_score - prior,
                        "episode_over_template_gain": (score - next_score) - (before - prior)}
            matched[task.id]["horizons"] = horizon_gains
        means = {key: sum(row[key] for row in matched.values()) / len(matched) if matched else None
                 for key in ("template_gain", "episode_over_template_gain", "episode_gain")}
        comparisons[label] = {"matched_tasks": len(matched), "expected_tasks": len(tasks), "missing_tasks": missing,
                              "status": "complete" if not missing else "partial" if matched else "unavailable",
                              "mean_on_matched_tasks": means, "tasks": matched}
    return comparisons


def _guidance_digest(entries: list[dict[str, Any]]) -> str | None:
    return digest({"schema": "ecogym.expert-guidance/1", "entries": entries}) if entries else None


# every file of these layers is part of a round's measurement: the code that runs the episodes and the judge, and
# what the agents read (a domain pack's task background, presets, knowledge and expert; the seed harness). Until
# 0.4.2 only the *.py files of data, episodes, evaluation, harness, protocol and solvers counted, so a resumed round
# could mix episodes that read another task background or ran under other task or ledger code (2026-11-18 review)
MEASURED_LAYERS = ("data", "domains", "episodes", "evaluation", "harness", "kernel", "protocol", "solvers", "task")


def measured_sources(package: Path) -> list[Path]:
    """The files whose bytes a round's measurement identity (``Evolution._measurement``) covers."""

    sources = [package / "evolution" / name for name in ("loop.py", "evolver.py", "evolver.md", "discovery.py",
                                                         "lifecycle.py", "methods.py", "pareto.py", "relay.py")]
    for layer in MEASURED_LAYERS:
        root = package / layer
        # hidden files (.DS_Store) and byte code come and go without changing anything
        sources.extend(sorted(path for path in root.rglob("*") if path.is_file() and path.suffix != ".pyc"
                              and not any(part.startswith(".") or part == "__pycache__"
                                          for part in path.relative_to(root).parts)))
    return sources


def _require_replies(entries: list[dict[str, Any]]) -> None:
    """Stop the run (resumably) when no episode of a round ran: no model reply (the provider is down or
    refusing) or a host error, after retries; the reasons say which."""

    if entries and not any(entry.get("status") != "infra_failed" and entry.get("status") != "error"
                           for entry in entries):
        reasons = sorted({entry.get("upstream_error") or entry.get("error") or entry.get("status", "")
                          for entry in entries})
        raise ServiceDown("no episode of this round ran (no model reply or a host error, after retries): "
                          + "; ".join(reasons)[:1500],
                          outage=all(entry.get("status") == "infra_failed" for entry in entries))


_model_key = model_key


class Evolution:
    def __init__(self, config: EvolveConfig, ledger_path: Path, *,
                 on_event: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        """``on_event`` sees every logged event and an ``episode`` notice per finished episode (progress only)."""

        self.config = config
        self.ledger_path = Path(ledger_path)
        self.suite = suite_spec(config.suite)
        gates = self.suite.gates
        if gates.test == "signed_rank" and not signed_rank_reachable(len(self.suite.train), gates.confidence,
                                                                     config.candidates):
            raise ValueError(f"with {config.candidates} candidates a round, the signed-rank gate of {self.suite.id} "
                             f"cannot pass on {len(self.suite.train)} training tasks; use fewer candidates")
        self.tasks: list[TaskSpec] = self.suite.tasks("train")
        self.benches: dict[str, Bench] = {}
        self.cancel = threading.Event()
        self.stop_reason: str | None = None  # why it is asked to stop (user, shutdown); set by whoever stops it
        self.on_event = on_event
        self._evaluations: dict[str, Evaluation] = {}
        self._start_cache = StartCache()
        self._notify_lock = threading.Lock()
        self._gates: dict[str, ServiceGate] = {}  # model -> its gate, for the models the run was started with
        self.method_registry = None
        if config.method_registry and config.method_research != "internal":
            from .methods import MethodRegistry

            self.method_registry = MethodRegistry(Path(config.method_registry)).pin()
            self.method_registry.check_agent_isolation(scratch_root() / "_method_registry_preflight",
                                                       discover_runtime())
            if config.method_research == "supplied" and not self.method_registry.available():
                raise ValueError("supplied method research needs at least one approved method")

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
                calibration_pre_stage(bench)
                self.benches[task.id] = bench

    def _evaluation(self, entry: dict[str, Any]) -> Evaluation:
        if entry["episode"] not in self._evaluations:
            bench = self.benches[entry["task"]]
            self._evaluations[entry["episode"]] = load_predictions(
                Path(entry["predictions"]), bench, bench.stage(SCORED_STAGE), expect=entry.get("predictions_digest"))
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
            # the process that runs it, from the start: one that dies before its first round (opening the benches
            # takes a while) reads as stale and is resumed by the service, instead of looking ownerless forever
            "owner": owner(),
            **({"method_registry_digest": self.method_registry.identity()} if self.method_registry is not None else {}),
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
        configure_judge_limit(self.config.concurrency)
        try:
            with clock.awake():  # judging and the gaps between sessions too, not only the agents
                result = self._rounds(run_id, stop)
            if result is not None:
                return result
        except (RunStopped, Cancelled, KeyboardInterrupt) as error:  # asked to stop (Ctrl-C included): resumable
            self.cancel.set()
            self._log(run_id, "evolve_failed", {"error": str(error) or "interrupted (Ctrl-C)", "stopped": True,
                                                "reason": self.stop_reason or "user"})
            raise
        except ServiceDown as error:  # the model service stayed down: resumable, by the service once it answers
            self.cancel.set()
            self._log(run_id, "evolve_failed", {"error": f"ServiceDown: {error}"[:2000],
                                                "reason": "service_down" if error.outage else None})
            raise
        except BaseException as error:
            self.cancel.set()
            self._log(run_id, "evolve_failed", {"error": f"{type(error).__name__}: {error}"[:2000]})
            raise
        with Ledger(self.ledger_path) as ledger:
            view = fold(ledger.events(run_id))
            spent = spent_by(ledger, view)
        result = {**result_of(view, spent), "seconds": round(time.monotonic() - started, 1)}
        self._log(run_id, "evolve_finished", result)
        return result

    def _rounds(self, run_id: str, stop: Callable[[], bool] | None) -> dict[str, Any] | None:
        """Every remaining round; the recorded result when the run had already finished, else None."""

        self._open_benches()
        resumed = True
        while True:
            view = self.view(run_id)
            if view["status"] == "finished":
                return view["result"]
            if view["suite"]["digest"] != self.suite.digest:
                raise ValueError("the suite differs from the one this run started with")
            if (self.method_registry is not None
                    and view.get("method_registry_digest") != self.method_registry.identity()):
                raise ValueError("the approved method registry differs from the one this run started with")
            if resumed:  # checks an interruption cut short, or that failed, are tried again once per resume
                resumed = False
                self._watch(run_id, view)
                from .verify import unchecked

                self._check_adoptions(run_id, unchecked(view))
            state = self.state(view)
            if state.generation > self.config.generations:
                return None
            if stop is not None and stop():
                raise RunStopped(f"stopped before round {state.generation}")
            self._round(run_id, view, state)

    def _watch(self, run_id: str, view: dict[str, Any]) -> None:
        """A gate for each model the run was started with: its agents' (pinned, or the seed's) and the evolver's.
        Sessions on another model (a candidate that names its own) retry as before, without waiting."""

        for key in run_models(self.config, view["seed"]["digest"]):
            if key not in self._gates:
                self._gates[key] = ServiceGate(key, self.config.outage_minutes * 60,
                                               lambda kind, body: self._log(run_id, kind, body))

    def _gate(self, ref: str | None) -> ServiceGate | None:
        key = _model_key(ref)
        return self._gates.get(key) if key else None

    def state(self, view: dict[str, Any]) -> State:
        finished = [round_ for round_ in view["rounds"] if round_["finished"]]
        generation = finished[-1]["generation"] + 1 if finished else 0
        incumbent = finished[-1]["incumbent_after"] if finished else view["seed"]["digest"]
        tried = {view["seed"].get("behavior") or view["seed"]["digest"]}
        for round_ in finished:
            for candidate in round_["candidates"]:
                decision = candidate.get("decision")
                if decision and not candidate.get("missing") and not decision["reason"].startswith("missing_tasks"):
                    tried.add(candidate.get("behavior") or candidate["digest"])
        return State(generation, incumbent, tried)

    def _measurement(self) -> str:
        """Identity of the inputs and local software allowed to fill the same round plan."""

        package = Path(__file__).resolve().parents[1]
        sources = measured_sources(package)
        runtime = discover_runtime()
        interpreters = {}
        for executable in (runtime.python, runtime.dsh):
            if executable is not None:
                target = executable.resolve()
                stat = target.stat()
                interpreters[str(target)] = [stat.st_size, stat.st_mtime_ns]
        libraries = sorted(str(path.relative_to(root)) for root in runtime.python_roots
                           for path in root.glob("lib/python*/site-packages/*.dist-info"))
        registry_identity = self.method_registry.identity() if self.method_registry is not None else None
        return digest({"config": self.config.to_dict(), "suite": self.suite.digest, "version": __version__,
                       "method_registry": registry_identity,
                       "bench": {task.id: self.benches[task.id].identity for task in self.tasks},
                       "sources": {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sources},
                       "interpreters": interpreters, "libraries": libraries})

    def _round(self, run_id: str, view: dict[str, Any], state: State) -> None:
        generation = state.generation
        incumbent = resolve(state.incumbent)
        last = view["rounds"][-1] if view["rounds"] else {}
        unfinished_same = last if last.get("generation") == generation and not last["finished"] else {}
        plan = unfinished_same.get("plan")
        resuming_relay = (
            not plan
            and generation > 0
            and self.config.relay_stages > 1
            and bool(unfinished_same.get("relay_stages"))
            and unfinished_same.get("incumbent") == incumbent.digest
            and unfinished_same.get("expert_until") is not None
            and isinstance(unfinished_same.get("guidance"), dict)
        )
        measurement = self._measurement()
        if plan:
            if plan["incumbent"] != incumbent.digest:
                raise ValueError("the interrupted round was planned for another incumbent than the run's finished "
                                 "rounds name; the ledger is inconsistent")
            snapshot = Snapshot(plan["expert_until"], plan["guidance"])
        elif resuming_relay:
            snapshot = Snapshot(int(unfinished_same["expert_until"]), dict(unfinished_same["guidance"]))
        else:
            with Ledger(self.ledger_path) as ledger:
                until = ledger.last_seq()
                snapshot = Snapshot(until, {task.id: _guidance_digest(expert.guidance(ledger, task, until=until))
                                            for task in self.tasks})
        self._log(run_id, "round_started", {"generation": generation, "incumbent": incumbent.digest, "owner": owner(),
                                            "expert_until": snapshot.until, "guidance": snapshot.guidance,
                                            **({"resume_plan": plan["id"]} if plan else {}),
                                            **({"resume_relay": True} if resuming_relay else {})})
        if plan:
            candidates = [{**item, "harness": resolve(item["digest"])} for item in plan["candidates"]]
            if plan["measurement"] != measurement:
                # the data, code or runtime changed while the round was interrupted (an upgrade, a new DSH): its
                # episodes so far measured something else. The same candidates under the same expert guidance are
                # measured again, every slot, under a new plan; nothing of the old one is reused, and its episodes
                # stay in the ledger and in the run's cost.
                plan = {**{key: value for key, value in plan.items() if key not in ("id", "replaces", "reason")},
                        "measurement": measurement, "replaces": plan["id"], "reason": "environment_changed"}
                plan["id"] = digest({"run": run_id, **plan})
                self._log(run_id, "round_planned", plan)
        else:
            if not generation:
                candidates = []
            elif self.config.relay_stages > 1:
                candidates = self._relay_propose(run_id, generation, incumbent, view, state, snapshot, measurement)
            else:
                candidates = self._propose(run_id, generation, incumbent, view, state, snapshot)
            plan_candidates = []
            for item in candidates:
                row = {key: item[key] for key in ("name", "digest")}
                if item.get("template_only"):
                    row["template_only"] = True
                if item.get("relay_origin"):
                    row["relay_origin"] = item["relay_origin"]
                plan_candidates.append(row)
            plan = {"schema": "ecogym.round-plan/1", "generation": generation, "incumbent": incumbent.digest,
                    "measurement": measurement, "expert_until": snapshot.until, "guidance": snapshot.guidance,
                    "candidates": plan_candidates}
            plan["id"] = digest({"run": run_id, **plan})
            self._log(run_id, "round_planned", plan)
        if generation and not candidates:
            outcome = (
                "relay_no_viable_candidate"
                if self.config.relay_stages > 1 and any(
                    st.get("ranking") for st in (self.view(run_id)["rounds"][-1].get("relay_stages") or ())
                )
                else "no_candidates"
            )
            self._log(run_id, "round_finished", {"generation": generation, "incumbent": incumbent.digest,
                                                 "adopted": None, "outcome": outcome})
            return
        jobs = [(incumbent, "incumbent", None)] + [(item["harness"], "candidate", item["name"]) for item in candidates]
        fast_tracked = (
            generation > 0
            and self.config.template_fast_track
            and bool(candidates)
            and all(item.get("template_only") for item in candidates)
        )
        if fast_tracked:
            entries = self._fast_track_templates(run_id, generation, jobs, snapshot, plan["id"])
        else:
            entries = self._episodes(run_id, generation, jobs, snapshot, plan["id"])
        self._log(run_id, "episodes_finished", {
            "generation": generation, "episodes": entries,
            **({"fast_track": True} if fast_tracked else {}),
        })
        _require_replies(entries)
        if generation == 0:
            self._log(run_id, "round_finished", {"generation": 0, "incumbent": incumbent.digest, "adopted": None,
                                                 "outcome": "baseline"})
            self._record_template_attribution(run_id, generation, plan, jobs, entries, view)
            return
        research = None
        if self.method_registry is not None:
            research = (self.view(run_id)["rounds"][-1].get("evolver") or {}).get("method_research")
        pool: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            if entry["role"] == "incumbent" and usable(entry):
                pool.setdefault(entry["task"], []).append(entry)
        confidence = adjusted_confidence(self.suite.gates.confidence, len(candidates))
        required_replicates = 1 if fast_tracked else self.config.repeats
        max_under_replicated = max(1, len(self.tasks) // 4) if required_replicates >= 2 else 0
        passing = []
        for candidate in candidates:
            pruned = any(entry.get("status") == "pruned" and entry.get("candidate") == candidate["name"]
                         for entry in entries)
            pairs, missing, under_replicated = [], [], []
            for task in self.tasks:
                own = [entry for entry in entries if entry["role"] == "candidate" and entry["task"] == task.id
                       and entry["candidate"] == candidate["name"] and usable(entry)]
                theirs = pool.get(task.id, [])
                have = min(len(own), len(theirs))
                needed = 1 if pruned else required_replicates
                if have < 1:
                    missing.append(task.id)
                    continue
                if have < needed:
                    under_replicated.append(task.id)
                pairs.append(TaskPair(task.id, tuple(self._evaluation(entry) for entry in own),
                                      tuple(self._evaluation(entry) for entry in theirs), group=task.dataset))
            too_few_replicates = not pruned and len(under_replicated) > max_under_replicated
            blocking_missing = [*missing, *(under_replicated if too_few_replicates else [])]
            body: dict[str, Any] = {"generation": generation, "name": candidate["name"], "digest": candidate["digest"],
                                    "missing": blocking_missing}
            if under_replicated and not too_few_replicated:
                body["under_replicated"] = under_replicated
            if fast_tracked:
                body["fast_track"] = True
            if pairs:
                decision = compare_suite(pairs, self.suite.gates, {"run": run_id, "generation": generation,
                                                                   "candidate": candidate["name"]},
                                         confidence=confidence).to_dict()
                if under_replicated and not too_few_replicated:
                    decision["under_replicated"] = under_replicated
                if pruned:
                    decision["adopted"], decision["reason"] = False, "pruned_after_replicate_0"
                    decision["scope"] = "replicate_0_pruned"
                elif blocking_missing:
                    decision["adopted"], decision["reason"] = False, "missing_tasks:" + ",".join(blocking_missing)
                    decision["scope"] = "partial_task_suite"
            else:
                reason = "pruned_after_replicate_0" if pruned else "missing_tasks:" + ",".join(blocking_missing)
                decision = {"adopted": False, "reason": reason, "delta": None,
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
        self._record_template_attribution(run_id, generation, plan, jobs, entries, view)
        references = self._public_reference(research, entries, adopted["name"] if adopted else None)
        if references:
            self._log(run_id, "method_reference_checked", {"generation": generation, **references})
        if adopted:
            self._check_adoptions(run_id, [generation])

    def _record_template_attribution(self, run_id: str, generation: int, plan: dict[str, Any],
                                     jobs: list[tuple[Harness, str, str | None]], entries: list[dict[str, Any]],
                                     view: dict[str, Any]) -> None:
        try:
            result = self._template_attribution(jobs, entries, plan["measurement"], view["rounds"])
        except Cancelled:
            raise
        except Exception as error:  # a diagnostic must not block a committed round or the next one
            result = {"status": "error", "error": f"{type(error).__name__}: {error}"[:1000]}
        self._log(run_id, "template_attribution_checked", {
            "schema": "ecogym.template-attribution/1", "generation": generation, "round_plan": plan["id"],
            "stage": SCORED_STAGE, "judge_protocol": JUDGE_PROTOCOL, **result})

    def _template_attribution(self, jobs: list[tuple[Harness, str, str | None]], entries: list[dict[str, Any]],
                              measurement: str, rounds: Sequence[dict[str, Any]]) -> dict[str, Any]:
        # A complete reference can be reused across identical programs, even when the surrounding harness changes.
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for entry in entries:
            label = "incumbent" if entry["role"] == "incumbent" else entry["candidate"]
            groups.setdefault((label, entry["task"]), []).append(entry)
        prior: dict[str, tuple[int, dict[str, Any]]] = {}
        for round_ in rounds:
            for arm in (round_.get("template_attribution") or {}).get("arms", {}).values():
                for task in arm["tasks"].values():
                    reference = task["template"]
                    if reference["status"] in ("ok", "drift"):
                        prior[reference["key"]] = (round_["generation"], reference)

        started = clock.now()
        arms: dict[str, Any] = {}
        pending: dict[str, tuple[Bench, Path]] = {}
        references: dict[str, dict[str, Any]] = {}
        observed_starts: dict[str, list[dict[str, Any]]] = {}
        with tempfile.TemporaryDirectory(prefix="evolution-templates-", dir=scratch_root()) as directory:
            for harness, _, name in jobs:
                label = name or "incumbent"
                template = Path(directory) / label
                template.mkdir()
                program, problem = None, None
                try:
                    program = snapshot_program(harness.root / "template", template)
                except (OSError, ToolError) as error:
                    problem = str(error)[:500]
                rows = {}
                for task in self.tasks:
                    bench = self.benches[task.id]
                    key = digest([measurement, bench.digest, SCORED_STAGE, JUDGE_PROTOCOL, program])
                    case = groups.get((label, task.id), [])
                    row, starts = self._episode_attribution(case, program)
                    reference = {"key": key, "program_digest": program, "bench": bench.digest}
                    if problem:
                        references[key] = {**reference, "status": "invalid_template", "error": problem}
                    else:
                        pending.setdefault(key, (bench, template))
                        observed_starts.setdefault(key, []).extend(starts)
                        references.setdefault(key, reference)
                    rows[task.id] = {**row, "template_key": key}
                arms[label] = {"harness": harness.digest, "tasks": rows}

            for key, starts in observed_starts.items():
                previous = prior.get(key)
                if previous is not None:
                    generation, saved = previous
                    if saved["status"] == "drift":
                        references[key] = {**saved, "source": "prior_round", "source_generation": generation}
                        pending.pop(key)
                        continue
                    starts.append({**saved, "source": "prior_round", "source_generation": generation})
                if not starts:
                    continue
                fingerprints = {item["forecast_digest"] for item in starts}
                if len(fingerprints) == 1:
                    references[key] = {**references[key], **starts[0],
                                       "matching_observations": len(starts)}
                else:
                    scores = [item["score"] for item in starts]
                    references[key].update({"status": "drift", "observations": len(starts),
                                            "score_range": [min(scores), max(scores)],
                                            "distinct_forecasts": len(fingerprints)})
                pending.pop(key)
            self._replay_templates(pending, references, started)

        for arm in arms.values():
            for row in arm["tasks"].values():
                reference = references[row.pop("template_key")]
                row["template"] = reference
                row["episode_over_template"] = (row["episode_mean"] - reference["score"]
                                                if reference["status"] == "ok" and row["episode_mean"] is not None
                                                else None)
        matched = sum(row["episode_over_template"] is not None
                      for arm in arms.values() for row in arm["tasks"].values())
        expected = len(arms) * len(self.tasks)
        return {"status": "complete" if matched == expected else "partial" if matched else "unavailable",
                "matched_arm_tasks": matched, "expected_arm_tasks": expected, "expected_episodes": self.config.repeats,
                "replay_budget_seconds": TEMPLATE_ATTRIBUTION_SECONDS, "seconds": round(clock.now() - started, 3),
                "arms": arms, "comparisons": _attribution_summary(arms, self.tasks)}

    def _episode_attribution(self, case: Sequence[dict[str, Any]], program: str | None
                             ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        scores, starts, coverage = [], [], []
        horizons: dict[str, list[float]] = {}
        available = set()
        failed = 0
        for entry in case:
            if (not usable(entry) or not entry.get("predictions_digest") or entry.get("score_stage") != SCORED_STAGE
                    or entry.get("judge_protocol") != JUDGE_PROTOCOL or entry["replicate"] in available):
                continue
            try:
                evaluation = self._evaluation(entry)
                score = evaluation.score()
                covered = float(np.isfinite(evaluation.predicted).mean())
                by_horizon = _horizon_scores(evaluation)
            except (OSError, ValueError, KeyError):
                failed += 1
                continue
            if not math.isfinite(score):
                failed += 1
                continue
            scores.append(score)
            coverage.append(covered)
            available.add(entry["replicate"])
            for horizon, value in by_horizon.items():
                horizons.setdefault(horizon, []).append(value)
            if (program and entry.get("scored_start") and entry.get("program_digest") == program
                    and entry.get("score_status") == "ok" and entry.get("score_limits_match")):
                starts.append({"status": "ok", "source": "episode_start", "episode": entry["episode"],
                               "score": score, "coverage": covered, "origins": evaluation.stage.size,
                               "horizons": by_horizon,
                               "predictions_digest": entry["predictions_digest"],
                               "forecast_digest": _forecast_fingerprint(evaluation)})
        complete = available == set(range(self.config.repeats))
        return {"episode_count": len(scores), "episode_mean": sum(scores) / len(scores) if complete else None,
                "episode_horizons": {key: sum(values) / len(values) for key, values in horizons.items()
                                     if complete and len(values) == len(scores)},
                "episode_coverage": sum(coverage) / len(coverage) if coverage else None,
                "scored_start_count": sum(bool(item.get("scored_start")) for item in case if usable(item)),
                "episodes_unreadable": failed}, starts

    def _replay_templates(self, pending: dict[str, tuple[Bench, Path]], references: dict[str, dict[str, Any]],
                          started: float) -> None:
        if not pending:
            return
        runtime = discover_runtime()
        limits = self.config.limits
        workers = min(2, self.config.concurrency, len(pending))
        deadline = started + TEMPLATE_ATTRIBUTION_SECONDS
        per_job = TEMPLATE_ATTRIBUTION_SECONDS / math.ceil(len(pending) / workers)

        def replay(bench: Bench, template: Path) -> dict[str, Any]:
            seconds = min(limits.total_seconds, per_job, deadline - clock.now())
            if seconds <= 0:
                return {"status": "budget_exhausted"}
            capped = replace(limits, total_seconds=seconds, fit_seconds=min(limits.fit_seconds, seconds),
                             forecast_seconds=min(limits.forecast_seconds, seconds))
            try:
                report, evaluation = judge_program(bench, template, SCORED_STAGE, runtime,
                                                   limits=capped, cancel=self.cancel)
            except Cancelled:
                raise
            except (OSError, ValueError, RuntimeError) as error:
                return {"status": "error", "error": f"{type(error).__name__}: {error}"[:500]}
            result = {"status": report["status"], "source": "fixed_replay", "coverage": report["coverage"],
                      "origins": report["origins"], "seconds": report["seconds"], "limit_seconds": seconds}
            if report["status"] == "ok" and math.isfinite(report["score"]):
                result.update({"score": report["score"], "forecast_digest": _forecast_fingerprint(evaluation),
                               "horizons": _horizon_scores(evaluation)})
            elif report["status"] == "ok":
                result["status"] = "invalid_score"
            return result

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="template-attribution") as pool:
            futures = {key: pool.submit(replay, *item) for key, item in pending.items()}
            for key, future in futures.items():
                references[key].update(future.result())

    def _public_reference(self, research: dict[str, Any] | None, entries: list[dict[str, Any]],
                          adopted: str | None) -> dict[str, Any] | None:
        """Compare one method actually probed this round on development, independently of adoption."""

        if self.method_registry is None or not research:
            return None
        completed = [probe for probe in research.get("probes", ())
                     if probe.get("status") == "evaluated" and probe.get("method")
                     and probe.get("result", {}).get("status") == "ok"]
        if not completed:
            return None
        method = max(completed, key=lambda probe: probe["result"]["mean"])["method"]
        runtime = discover_runtime()
        limits = self.config.limits
        budget = min(limits.total_seconds, PUBLIC_REFERENCE_SECONDS)
        deadline = clock.now() + budget
        tasks = {}
        for task in self.tasks:
            remaining = deadline - clock.now()
            row: dict[str, Any] = {"status": "budget_exhausted", "external": None}
            try:
                if remaining > 0:
                    capped = replace(limits, total_seconds=remaining, fit_seconds=min(limits.fit_seconds, remaining),
                                     forecast_seconds=min(limits.forecast_seconds, remaining))
                    report, _ = self.method_registry.judge(method["digest"], self.benches[task.id], SCORED_STAGE,
                                                           runtime, limits=capped, cancel=self.cancel)
                    row.update({key: report.get(key) for key in ("status", "coverage", "bench", "judge_protocol")})
                    if report.get("status") == "ok" and report.get("coverage") == 1:
                        row["external"] = report.get("score")
            except Cancelled:
                raise
            except (OSError, ValueError, RuntimeError) as error:
                row = {"status": type(error).__name__, "external": None}
            scores = [entry["score"] for entry in entries if entry["task"] == task.id and usable(entry)
                      and (entry.get("candidate") == adopted if adopted is not None else entry["role"] == "incumbent")]
            row["selected"] = sum(scores) / len(scores) if len(scores) >= self.config.repeats else None
            tasks[task.id] = row
        gaps = [row["selected"] - row["external"] for row in tasks.values()
                if row["external"] is not None and row["selected"] is not None]
        gap = sum(gaps) / len(gaps) if gaps and len(gaps) == len(tasks) else None
        return {"method": method, "stage": SCORED_STAGE, "tasks": tasks, "selected_minus_external": gap,
                "budget_seconds": budget}

    def _check_adoptions(self, run_id: str, generations: list[int]) -> None:
        """Measure these generations' adoptions on the training tasks' sealed validation period
        (``verify.check_adoptions``), between rounds: for people judging the run, never for the evolver.

        Only the programs are judged again (no agent runs), so a check takes minutes. A failed check is recorded
        and never stops the run; asking the run to stop stops it (the next resume checks again)."""

        if not self.config.check_adoptions or not generations:
            return
        from .verify import ADOPTION_STAGE, check_adoptions  # verify imports this module

        for generation in generations:
            try:
                bodies = check_adoptions(self.ledger_path, run_id, generations=[generation], cancel=self.cancel)
            except (Cancelled, KeyboardInterrupt):
                raise
            except Exception as error:
                self._log(run_id, "adoption_checked", {"generation": generation, "stage": ADOPTION_STAGE,
                                                       "error": f"{type(error).__name__}: {error}"[:1000]})
                continue
            for body in bodies:
                self._notify("adoption_checked", body)

    def _incumbent_baseline(self, view: dict[str, Any], incumbent: Harness, snapshot: Snapshot) -> dict[str, float]:
        """Per-task mean score of ``incumbent`` from the most recent finished round under the same expert guidance.

        Used strictly for cheap intra-generation relay screening (reducing screening variance while avoiding
        redundant incumbent episodes); the final adoption gate always runs fresh contemporary paired episodes.
        """

        found: dict[str, float] = {}
        for round_ in reversed(view.get("rounds") or ()):
            if not round_.get("finished"):
                continue
            by_task: dict[str, list[float]] = {}
            for entry in round_.get("episodes") or ():
                if (entry.get("harness") == incumbent.digest and usable(entry)
                        and isinstance(entry.get("score"), (int, float))
                        and math.isfinite(float(entry["score"]))
                        and (entry.get("guidance") or None) == snapshot.guidance.get(entry["task"])):
                    by_task.setdefault(entry["task"], []).append(float(entry["score"]))
            for task_id, scores in by_task.items():
                if task_id not in found and scores:
                    found[task_id] = sum(scores) / len(scores)
            if len(found) >= len(self.tasks):
                break
        return found

    def _task_difficulty(
        self, view: dict[str, Any], incumbent: Harness, baseline: dict[str, float]
    ) -> dict[str, float]:
        """Priority score per training task for high-regret relay Stage-1 screening.

        Combines gap to the best non-naive reference model (regret) with incumbent replicate spread so Stage 1
        screens the most informative/high-regret tasks across every dataset group first.
        """

        if not baseline:
            return {}
        spread: dict[str, float] = {}
        for round_ in reversed(view.get("rounds") or ()):
            if not round_.get("finished"):
                continue
            by_task: dict[str, list[float]] = {}
            for entry in round_.get("episodes") or ():
                if (entry.get("harness") == incumbent.digest and usable(entry)
                        and isinstance(entry.get("score"), (int, float))
                        and math.isfinite(float(entry["score"]))):
                    by_task.setdefault(entry["task"], []).append(float(entry["score"]))
            for task_id, scores in by_task.items():
                if task_id not in spread and len(scores) >= 2:
                    spread[task_id] = max(scores) - min(scores)
            if len(spread) >= len(self.tasks):
                break
        difficulty: dict[str, float] = {}
        for task in self.tasks:
            base = baseline.get(task.id)
            if base is None:
                continue
            bench = self.benches.get(task.id)
            refs = reference_scores(bench, SCORED_STAGE) if bench is not None else {}
            non_naive = [float(v) for k, v in refs.items() if k != "naive" and isinstance(v, (int, float))
                         and math.isfinite(float(v))]
            ref_target = max(non_naive) if non_naive else 0.0
            regret = max(0.0, ref_target - base)
            difficulty[task.id] = round(regret + 0.5 * spread.get(task.id, 0.0), 6)
        return difficulty

    def _run_evolver_session(
        self,
        run_id: str,
        generation: int,
        incumbent: Harness,
        view: dict[str, Any],
        snapshot: Snapshot,
        *,
        settings: EvolverSettings,
        relay_stage: int | None = None,
        starting: Harness | None = None,
        starting_label: str | None = None,
        extra_archived: dict[str, Harness] | None = None,
        extra_known: dict[str, str] | None = None,
        relay_note: str = "",
        allow_discovery: bool = True,
        stage_tag: str = "",
    ) -> dict[str, Any]:
        config = self.config
        with Ledger(self.ledger_path) as ledger:  # the training tasks' guidance only: never a holdout task's
            advice = expert.guidance(ledger, self.tasks, until=snapshot.until)
        gate, reruns = self._gate(config.evolver_model), 0
        stage_extra = {"relay_stage": relay_stage} if relay_stage is not None else {}
        while True:
            try:
                outcome = run_evolver(
                    run_id=run_id, generation=generation, incumbent=incumbent, suite=self.suite,
                    tasks=self.tasks, benches=self.benches, view=view, ledger_path=self.ledger_path,
                    settings=settings, repeats=config.repeats, episode_minutes=config.minutes,
                    episode_tokens=config.tokens, model=config.model, limits=config.limits,
                    expert_text=expert.render(advice) if advice else "", cancel=self.cancel,
                    workers=config.concurrency, method_research=config.method_research,
                    method_registry=self.method_registry, local_selection=config.local_selection,
                    starting=starting, starting_label=starting_label, extra_archived=extra_archived,
                    extra_known=extra_known, relay_note=relay_note, allow_discovery=allow_discovery,
                    stage_tag=stage_tag,
                )
            except (Cancelled, KeyboardInterrupt) as error:
                stopped = getattr(error, "session", None)
                if stopped is not None:  # what the stopped session spent stays on the record (fold.spending)
                    body = stopped.to_dict()
                    self._log(run_id, "evolver_finished", {
                        "generation": generation, **stage_extra, "status": "interrupted", "candidates": [],
                        **{key: body.get(key) for key in ("seconds", "tokens", "metered_tokens", "llm_calls",
                                                          "answered_calls", "failed_calls", "trajectory",
                                                          "upstream_error", "unsettled")}})
                raise
            except Exception as error:
                self._log(run_id, "evolver_finished", {"generation": generation, **stage_extra, "status": "error",
                                                       "error": f"{type(error).__name__}: {error}"[:2000]})
                raise
            session = outcome["session"]
            budget = settings.minutes * 60
            stalled = _builder_stalled(outcome)
            # failed calls kept the evolver from working for most of its time and it left nothing to compare: the
            # provider decided this generation, not the evidence (evo:0f0d4068b8b1 generation 5: 871 of 900 s, two
            # duplicates without proposals), so the generation is reviewed again rather than spent
            crippled = stalled > STALL_SHARE * budget and not any(item["status"] == "proposed"
                                                                  for item in outcome["candidates"])
            if session["status"] != "infra_failed" and not crippled:
                return outcome
            body = {"generation": generation, **stage_extra, **{key: session.get(key) for key in (
                "status", "seconds", "tokens", "metered_tokens", "llm_calls", "answered_calls", "failed_calls",
                "trajectory", "upstream_error", "stalled_seconds")}, "candidates": []}
            if crippled and session["status"] != "infra_failed":
                body["status"] = "infra_failed"
                body["stalled"] = (f"failed model calls kept the evolver from working for {stalled:.0f} of its "
                                   f"{budget:.0f} seconds and it left no candidate to compare")
            if gate is not None and gate.limit > 0 and not reruns:
                body["rerun"] = True
            self._log(run_id, "evolver_finished", body)
            if body.get("rerun"):  # one more session, after the outage if the service is down
                reruns += 1
                if gate.confirm(session.get("upstream_error") or body.get("stalled") or ""):
                    gate.wait(self.cancel)  # ServiceDown once the outage outlasts outage_minutes
                continue
            raise ServiceDown(f"the evolver got no model reply: {session.get('upstream_error') or 'service failure'}"
                              if not body.get("stalled") else f"the evolver could not work: {body['stalled']} "
                              f"({session.get('upstream_error') or 'service failure'})")

    def _candidate_record(self, generation: int, incumbent: Harness, candidate: dict[str, Any],
                          seen: set[str], *, duplicate_msg: str | None = None) -> tuple[dict[str, Any], bool]:
        behavior = behavior_digest(candidate["harness"]) if candidate["status"] == "proposed" else None
        if behavior is not None and behavior in seen:
            candidate["status"], candidate["ok"] = "duplicate", False
            candidate["problems"] = [
                *candidate["problems"],
                duplicate_msg or "与在任版本或以往比较过的候选内容相同（harness.toml 的名称、摘要不算）",
            ]
        elif behavior is not None:
            seen.add(behavior)
        patches, budget = [], PATCH_LIMIT
        for change in candidate.get("changes", []):
            patch = change["patch"][: max(budget, 0)]
            budget -= len(patch)
            patches.append({**change, "patch": patch})
        body = {"generation": generation, "name": candidate["name"], "status": candidate["status"],
                "digest": candidate.get("digest"), "behavior": behavior, "parent": incumbent.digest,
                "base": candidate.get("base"),  # the archived version it was reset to; None: the incumbent
                "summary": candidate.get("summary", ""), "proposal": candidate.get("proposal", ""),
                "agent": candidate.get("agent"), "changes": patches, "smoke": candidate.get("smoke"),
                "problems": candidate.get("problems", []), "warnings": candidate.get("warnings", [])}
        if candidate.get("method_research"):
            body["method_research"] = candidate["method_research"]
        changes = candidate.get("changes", [])
        agent_changed = candidate.get("agent") != asdict(incumbent.agent)
        template_only = (
            candidate["status"] == "proposed"
            and not agent_changed
            and any(item["path"].startswith("template/") for item in changes)
            and all(item["path"].startswith("template/") or item["path"] == "harness.toml" for item in changes)
        )
        return body, template_only

    def _log_evolver_finished(self, run_id: str, generation: int, outcome: dict[str, Any],
                              *, relay_stage: int | None = None) -> None:
        session = outcome["session"]
        self._log(run_id, "evolver_finished", {
            "generation": generation,
            **({"relay_stage": relay_stage} if relay_stage is not None else {}),
            **{key: session.get(key) for key in ("status", "seconds", "tokens", "metered_tokens", "llm_calls",
                                                 "answered_calls", "failed_calls", "trajectory", "activity",
                                                 "upstream_error", "continuations")},
            "final_text": session["final_text"][-4000:],
            "candidates": [{"name": item["name"], "status": item["status"]} for item in outcome["candidates"]],
            **({"method_research": outcome["method_research"]} if "method_research" in outcome else {}),
        })

    def _propose(self, run_id: str, generation: int, incumbent: Harness, view: dict[str, Any],
                 state: State, snapshot: Snapshot) -> list[dict[str, Any]]:
        config = self.config
        settings = EvolverSettings(minutes=config.evolver_minutes, tokens=config.evolver_tokens,
                                   model=config.evolver_model, candidates=config.candidates)
        outcome = self._run_evolver_session(run_id, generation, incumbent, view, snapshot, settings=settings)
        found = []
        seen = state.tried | {behavior_digest(incumbent)}
        for candidate in outcome["candidates"]:
            body, template_only = self._candidate_record(generation, incumbent, candidate, seen)
            if candidate["status"] == "proposed":
                self._log(run_id, "candidate_proposed", body)
                found.append({"name": candidate["name"], "digest": candidate["digest"],
                              "harness": candidate["harness"], "template_only": template_only})
            else:
                self._log(run_id, "candidate_rejected", body)
        self._log_evolver_finished(run_id, generation, outcome)
        return found

    def _plan_relay_stage(
        self,
        run_id: str,
        generation: int,
        stage: int,
        total_stages: int,
        incumbent: Harness,
        starting: Harness,
        starting_label: str,
        state: State,
        snapshot: Snapshot,
        measurement: str,
        completed_stages: Sequence[dict[str, Any]],
        extra_archived: dict[str, Harness],
        stage_seen: set[str],
        *,
        stage_cands: int,
        stage_minutes: float,
        stage_tokens: int,
        task_difficulty: dict[str, float] | None = None,
    ) -> tuple[str, list[TaskSpec], list[dict[str, Any]]]:
        """Run one relay stage's evolver session, carry prior leader for cross-batch checks, and log its plan."""

        config = self.config
        stage_tasks = select_stage_tasks(
            self.tasks, generation=generation, stage=stage,
            total_stages=total_stages, batch_size=config.relay_batch,
            task_difficulty=task_difficulty,
        )
        live_view = self.view(run_id)
        relay_note = render_relay_brief(
            generation=generation, stage=stage, total_stages=total_stages,
            stage_tasks=stage_tasks, prior_stages=completed_stages,
            starting_label=starting_label, starting_digest=starting.digest,
            incumbent_digest=incumbent.digest, final_limit=config.candidates,
        )
        extra_known: dict[str, str] = {}
        if starting.digest != incumbent.digest:
            extra_known[behavior_digest(starting)] = (
                f"本阶段出发的优胜版本 {starting_label}（已自动纳入本阶段互补对照与终审池，请在此基础上进一步改进）"
            )
        for prev in completed_stages:
            for cand in prev.get("candidates") or ():
                beh = cand.get("behavior")
                if beh and beh not in extra_known:
                    extra_known[beh] = f"本代前序接力阶段候选 {cand.get('label') or cand.get('name')}"
        settings = EvolverSettings(minutes=stage_minutes, tokens=stage_tokens,
                                   model=config.evolver_model, candidates=stage_cands)
        outcome = self._run_evolver_session(
            run_id, generation, incumbent, live_view, snapshot,
            settings=settings, relay_stage=stage, starting=starting,
            starting_label=starting_label if starting.digest != incumbent.digest else None,
            extra_archived=extra_archived, extra_known=extra_known,
            relay_note=relay_note, allow_discovery=(stage == 1), stage_tag=f"-r{stage}",
        )
        stage_found: list[dict[str, Any]] = []
        stage_rejected: list[dict[str, Any]] = []
        seen = state.tried | {behavior_digest(incumbent)} | stage_seen
        if starting.digest != incumbent.digest:
            seen.add(behavior_digest(starting))
        dup_msg = (
            f"与在任版本、出发版本 {starting_label} 或已比较过的候选内容相同"
            if starting.digest != incumbent.digest
            else None
        )
        for candidate in outcome["candidates"]:
            body, template_only = self._candidate_record(
                generation, incumbent, candidate, seen, duplicate_msg=dup_msg
            )
            if candidate["status"] == "proposed":
                r_name = f"r{stage}{candidate['name']}"
                record = {
                    **body, "name": r_name, "local_name": candidate["name"],
                    "label": f"g{generation}{r_name}", "template_only": template_only,
                }
                stage_found.append({**record, "harness": candidate["harness"]})
                if body.get("behavior"):
                    stage_seen.add(body["behavior"])
            else:
                stage_rejected.append({
                    "name": f"r{stage}{candidate['name']}", "status": candidate["status"],
                    "problems": candidate.get("problems", []),
                })
        # Cross-batch verification: carry the prior stage's advancing leader onto stage s's complementary batch
        # so multi-stage finalist selection scores its genuine cross-batch generalization.
        if stage > 1 and stage_found and starting.digest != incumbent.digest:
            lead_beh = behavior_digest(starting)
            if all(item.get("behavior") != lead_beh for item in stage_found):
                prior_cand = next(
                    (c for st in reversed(completed_stages) for c in st.get("candidates") or ()
                     if c.get("digest") == starting.digest),
                    {},
                )
                stage_found.append({
                    **prior_cand,
                    "name": f"r{stage}_lead",
                    "label": starting_label,
                    "digest": starting.digest,
                    "behavior": lead_beh,
                    "carried_leader": True,
                    "harness": starting,
                })
        self._log_evolver_finished(run_id, generation, outcome, relay_stage=stage)
        stage_plan_id = digest({
            "schema": "ecogym.relay-stage-plan/1",
            "run": run_id, "generation": generation, "stage": stage, "measurement": measurement,
            "tasks": [task.id for task in stage_tasks],
            "candidates": [item["digest"] for item in stage_found],
        })
        plan_cands = [{key: val for key, val in item.items() if key != "harness"} for item in stage_found]
        self._log(run_id, "relay_stage_planned", {
            "generation": generation, "stage": stage, "total_stages": total_stages,
            "plan_id": stage_plan_id, "measurement": measurement,
            "tasks": [task.id for task in stage_tasks],
            "starting": {"digest": starting.digest, "label": starting_label},
            "candidates": plan_cands, "rejected": stage_rejected,
        })
        return stage_plan_id, stage_tasks, stage_found

    def _promote_relay_finalists(
        self,
        run_id: str,
        generation: int,
        incumbent: Harness,
        completed_stages: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Promote top cross-stage relay finalists as official round candidates (or early-exit if all worse)."""

        finalists = select_finalists(
            completed_stages,
            limit=self.config.candidates,
            min_delta=float(getattr(self.suite.gates, "min_delta", 0.005)),
            skip_if_all_worse=self.config.early_prune,
        )
        promoted: list[dict[str, Any]] = []
        for idx, item in enumerate(finalists):
            final_name = NAMES[idx]
            origin = item["relay_origin"]
            harness_obj = resolve(item["digest"])
            body = {
                "generation": generation, "name": final_name, "status": "proposed",
                "digest": item["digest"], "behavior": item.get("behavior") or behavior_digest(harness_obj),
                "parent": incumbent.digest, "base": item.get("base"),
                "summary": item.get("summary", ""), "proposal": item.get("proposal", ""),
                "agent": item.get("agent"), "changes": item.get("changes", []), "smoke": item.get("smoke"),
                "problems": [], "warnings": item.get("warnings", []),
                "relay_origin": origin,
            }
            if item.get("method_research"):
                body["method_research"] = item["method_research"]
            self._log(run_id, "candidate_proposed", body)
            promoted.append({
                "name": final_name, "digest": item["digest"], "harness": harness_obj,
                "template_only": bool(item.get("template_only")), "relay_origin": origin,
            })
        return promoted

    def _relay_propose(self, run_id: str, generation: int, incumbent: Harness, view: dict[str, Any],
                       state: State, snapshot: Snapshot, measurement: str) -> list[dict[str, Any]]:
        """Run ``config.relay_stages`` mini-batch screening stages and promote the best finalist(s)."""

        config = self.config
        total_stages = config.relay_stages
        stage_cands = effective_relay_candidates(config.candidates, config.relay_candidates)
        stage_minutes = max(min(config.evolver_minutes, 8.0), round(config.evolver_minutes / total_stages, 1))
        stage_tokens = max(min(config.evolver_tokens, 1_500_000), int(config.evolver_tokens / total_stages))
        task_by_id = {task.id: task for task in self.tasks}
        baseline = self._incumbent_baseline(view, incumbent, snapshot)
        difficulty = self._task_difficulty(view, incumbent, baseline)
        current_round = (self.view(run_id).get("rounds") or [{}])[-1]
        saved_stages = {
            int(item["stage"]): item
            for item in current_round.get("relay_stages") or ()
            if item.get("measurement") == measurement
        }
        starting, starting_label = incumbent, "incumbent"
        extra_archived: dict[str, Harness] = {}
        stage_seen: set[str] = set()
        completed_stages: list[dict[str, Any]] = []

        for stage in range(1, total_stages + 1):
            saved = saved_stages.get(stage)
            if saved and saved.get("finished"):
                completed_stages.append(saved)
                for cand in saved.get("candidates") or ():
                    if cand.get("digest"):
                        with contextlib.suppress(HarnessError):
                            h_obj = resolve(cand["digest"])
                            extra_archived[str(cand.get("label") or cand["name"])] = h_obj
                            stage_seen.add(str(cand.get("behavior") or behavior_digest(h_obj)))
                leader = saved.get("leader") or {}
                if leader.get("advanced") and leader.get("digest"):
                    with contextlib.suppress(HarnessError):
                        starting = resolve(leader["digest"])
                        starting_label = str(leader.get("label") or leader.get("name") or starting.name)
                continue

            if saved and saved.get("plan_id"):
                stage_plan_id = str(saved["plan_id"])
                stage_tasks = [task_by_id[tid] for tid in saved.get("tasks") or () if tid in task_by_id]
                stage_found = [{**cand, "harness": resolve(cand["digest"])} for cand in saved.get("candidates") or ()
                               if cand.get("digest")]
            else:
                stage_plan_id, stage_tasks, stage_found = self._plan_relay_stage(
                    run_id, generation, stage, total_stages, incumbent, starting, starting_label,
                    state, snapshot, measurement, completed_stages, extra_archived, stage_seen,
                    stage_cands=stage_cands, stage_minutes=stage_minutes, stage_tokens=stage_tokens,
                    task_difficulty=difficulty,
                )

            stage_fast = (
                bool(stage_found) and config.template_fast_track
                and all(item.get("template_only") for item in stage_found)
            )
            if not stage_found:
                stage_entries: list[dict[str, Any]] = []
                ranked: list[dict[str, Any]] = []
            elif stage_fast:
                stage_jobs = [(incumbent, "incumbent", None)] + [
                    (item["harness"], "candidate", item["name"]) for item in stage_found
                ]
                stage_entries = self._fast_track_templates(
                    run_id, generation, stage_jobs, snapshot, stage_plan_id,
                    tasks=stage_tasks, relay_stage=stage,
                )
                _require_replies(stage_entries)
                ranked = rank_stage_candidates(stage_found, stage_entries, stage_tasks, incumbent_baseline=baseline)
            else:
                need_inc_tasks = [task for task in stage_tasks if task.id not in baseline]
                stage_entries = self._relay_episodes(
                    run_id, generation, stage, stage_found, incumbent,
                    stage_tasks, need_inc_tasks, snapshot, stage_plan_id,
                )
                _require_replies(stage_entries)
                ranked = rank_stage_candidates(stage_found, stage_entries, stage_tasks, incumbent_baseline=baseline)

            leader = pick_stage_leader(ranked, fallback_digest=starting.digest, fallback_label=starting_label)
            stage_body = {
                "generation": generation, "stage": stage, "total_stages": total_stages,
                "plan_id": stage_plan_id, "tasks": [task.id for task in stage_tasks],
                "episodes": stage_entries, "ranking": ranked, "leader": leader,
                **({"fast_track": True} if stage_fast else {}),
            }
            self._log(run_id, "relay_stage_finished", stage_body)
            completed_stages.append({
                **stage_body,
                "candidates": [{key: val for key, val in item.items() if key != "harness"} for item in stage_found],
                "finished": True,
            })
            for item in stage_found:
                if not item.get("carried_leader"):
                    extra_archived[str(item.get("label") or item["name"])] = item["harness"]
            if leader.get("advanced") and leader.get("digest"):
                starting = resolve(leader["digest"])
                starting_label = str(leader.get("label") or leader.get("name") or starting.name)

        return self._promote_relay_finalists(run_id, generation, incumbent, completed_stages)

    def _run_episode_pool(
        self,
        batch: Sequence[tuple[Harness, str, str | None, TaskSpec, int]],
        recovered: dict[tuple[str, str, str | None, str, int], dict[str, Any]],
        attempts: dict[tuple[str, str, str | None, str, int], int],
        run_id: str,
        generation: int,
        snapshot: Snapshot,
        plan_id: str,
        *,
        relay_stage: int | None = None,
        prefix: str = "episode",
    ) -> list[dict[str, Any]]:
        """Run a batch of episode specs concurrently, reusing verified recovered slots and cancelling on error."""

        with ThreadPoolExecutor(max_workers=self.config.concurrency, thread_name_prefix=prefix) as pool:
            pending = []
            for spec in batch:
                harness, role, name, task, replicate = spec
                slot = (harness.digest, role, name, task.id, replicate)
                future = None if slot in recovered else pool.submit(
                    self._episode, run_id, generation, *spec, snapshot, plan_id, attempts.get(slot, 0),
                    **({"relay_stage": relay_stage} if relay_stage is not None else {}),
                )
                pending.append((slot, future))
            try:
                return [recovered[slot] if future is None else future.result() for slot, future in pending]
            except BaseException:
                self.cancel.set()
                for _, future in pending:
                    if future is not None:
                        future.cancel()
                raise

    def _relay_episodes(
        self,
        run_id: str,
        generation: int,
        stage: int,
        stage_candidates: Sequence[dict[str, Any]],
        incumbent: Harness,
        stage_tasks: Sequence[TaskSpec],
        need_inc_tasks: Sequence[TaskSpec],
        snapshot: Snapshot,
        plan_id: str,
    ) -> list[dict[str, Any]]:
        """Run 1 replicate per task for a relay stage's candidates (plus incumbent on tasks lacking a baseline)."""

        need_inc_ids = {task.id for task in need_inc_tasks}
        specs: list[tuple[Harness, str, str | None, TaskSpec, int]] = []
        for task in stage_tasks:
            if task.id in need_inc_ids:
                specs.append((incumbent, "incumbent", None, task, 0))
            for item in stage_candidates:
                specs.append((item["harness"], "candidate", str(item["name"]), task, 0))
        recovered, attempts = self._recover_episodes(run_id, generation, plan_id, snapshot)
        return self._run_episode_pool(
            specs, recovered, attempts, run_id, generation, snapshot, plan_id,
            relay_stage=stage, prefix=f"relay-s{stage}",
        )

    def _recover_episodes(self, run_id: str, generation: int, plan_id: str, snapshot: Snapshot
                          ) -> tuple[dict, dict]:
        """Find verified completed slots and the next attempt numbers from the episode events."""

        with Ledger(self.ledger_path) as ledger:
            events = ledger.evolution_episode_events(
                (run_id,), generation=generation, kinds=("episode_started", "episode_scored", "episode_finished"))
        by_episode: dict[str, list] = {}
        for event in events:
            by_episode.setdefault(event.run_id, []).append(event)
        recovered, attempts = {}, {}
        for history in by_episode.values():
            started = history[0].body
            context = started.get("context") or {}
            if context.get("round_plan") != plan_id:
                continue
            task = started["task_id"]
            entry = {"task": task, "harness": started["harness"]["digest"], "role": context["role"],
                     "candidate": context.get("candidate"), "replicate": context["replicate"],
                     "guidance": snapshot.guidance[task], "attempt": context["attempt"]}
            slot = _slot(entry)
            attempts[slot] = max(attempts.get(slot, 0), context["attempt"] + 1)
            if slot in recovered:
                continue
            record = fold_episode(history)
            finished, scored = record.get("finished") or {}, record.get("scored") or {}
            artifact_dir = (finished.get("trajectory") or {}).get("dir")
            if (record["status"] not in SCORED_STATUSES or not artifact_dir or not scored.get("predictions")
                    or not scored.get("predictions_digest")):
                continue
            entry.update({"episode": record["episode"], "status": record["status"], "score": scored["score"],
                          "predictions": str(Path(artifact_dir) / scored["predictions"]),
                          "predictions_digest": scored["predictions_digest"], "tokens": finished.get("tokens"),
                          "metered_tokens": finished.get("metered_tokens"), "seconds": finished.get("seconds"),
                          **_scored_program(scored),
                          "score_limits_match": started.get("limits") == asdict(self.config.limits)})
            try:
                self._evaluation(entry)
            except (OSError, ValueError, KeyError):
                continue
            recovered[slot] = entry
        return recovered, attempts

    def _prunable_candidates(self, wave1: Sequence[dict[str, Any]], jobs: Sequence[tuple[Harness, str, str | None]]
                             ) -> set[str]:
        """Candidates clearly dominated after replicate 0 (worse on >50% of tasks and mean delta < -min_delta)."""

        if len(self.tasks) < 3:
            return set()
        incumbent_scores = {
            entry["task"]: float(entry["score"])
            for entry in wave1
            if entry.get("role") == "incumbent" and usable(entry) and isinstance(entry.get("score"), (int, float))
            and math.isfinite(float(entry["score"]))
        }
        if len(incumbent_scores) < len(self.tasks):
            return set()
        pruned: set[str] = set()
        min_delta = float(getattr(self.suite.gates, "min_delta", 0.005))
        for _, role, name in jobs:
            if role != "candidate" or not name:
                continue
            deltas: list[float] = []
            for entry in wave1:
                if (entry.get("role") == "candidate" and entry.get("candidate") == name and usable(entry)
                        and entry["task"] in incumbent_scores and isinstance(entry.get("score"), (int, float))
                        and math.isfinite(float(entry["score"]))):
                    deltas.append(float(entry["score"]) - incumbent_scores[entry["task"]])
            if len(deltas) == len(self.tasks):
                worse = sum(1 for d in deltas if d < 0.0)
                if worse > len(deltas) / 2.0 and (sum(deltas) / len(deltas)) < -min_delta:
                    pruned.add(name)
        return pruned

    def _fast_track_templates(
        self,
        run_id: str,
        generation: int,
        jobs: list[tuple[Harness, str, str | None]],
        snapshot: Snapshot,
        plan_id: str,
        *,
        tasks: Sequence[TaskSpec] | None = None,
        relay_stage: int | None = None,
    ) -> list[dict[str, Any]]:
        """Deterministically evaluate pure template-only rounds on SCORED_STAGE without running LLM agent episodes."""

        runtime = discover_runtime()
        tag = run_id.partition(":")[2] or run_id
        stage_part = f"-r{relay_stage}" if relay_stage is not None else ""
        out_dir = ecogym_home() / "trajectories" / f"fast-{tag}-g{generation}{stage_part}-{plan_id[:8]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        chosen_tasks = list(tasks) if tasks is not None else self.tasks
        work_items = [(harness, role, name, task) for task in chosen_tasks for harness, role, name in jobs]

        def run_one(item: tuple[Harness, str, str | None, TaskSpec]) -> dict[str, Any]:
            harness, role, name, task = item
            label = "incumbent" if role == "incumbent" else str(name)
            ep_id = f"fast:{tag}-g{generation}{stage_part}-{label}-{digest([task.id, harness.digest])[:8]}"
            entry: dict[str, Any] = {
                "episode": ep_id, "task": task.id, "harness": harness.digest, "role": role,
                "candidate": name, "replicate": 0, "guidance": snapshot.guidance[task.id],
                "attempt": 0, "tokens": 0, "metered_tokens": 0, "fast_track": True,
                **({"relay_stage": relay_stage} if relay_stage is not None else {}),
            }
            bench = self.benches[task.id]
            with tempfile.TemporaryDirectory(prefix="fast-template-", dir=scratch_root()) as tmp:
                program_dir = Path(tmp) / "template"
                program_dir.mkdir()
                try:
                    prog_digest = snapshot_program(harness.root / "template", program_dir)
                    report, evaluation = judge_program(
                        bench, program_dir, SCORED_STAGE, runtime, limits=self.config.limits, cancel=self.cancel
                    )
                except Cancelled:
                    raise
                except (OSError, ToolError, ValueError, RuntimeError) as error:
                    entry.update({"status": "error", "error": f"{type(error).__name__}: {error}"[:1000],
                                  "seconds": 0.0})
                    return entry
            pred_path = out_dir / f"{label}-{task.id.replace(':', '_').replace('/', '_')}.npz"
            save_predictions(pred_path, bench, evaluation)
            pred_digest = file_digest(pred_path)
            entry.update({
                "status": "finished" if report["status"] == "ok" else "error",
                "score": report["score"],
                "predictions": str(pred_path),
                "predictions_digest": pred_digest,
                "program_digest": prog_digest,
                "scored_start": True,
                "score_stage": SCORED_STAGE,
                "judge_protocol": JUDGE_PROTOCOL,
                "score_status": report["status"],
                "score_limits_match": True,
                "seconds": report["seconds"],
            })
            self._evaluations[ep_id] = evaluation
            self._notify("episode", entry)
            return entry

        with ThreadPoolExecutor(max_workers=min(self.config.concurrency, max(1, len(work_items))),
                                thread_name_prefix="fast-track") as pool:
            return list(pool.map(run_one, work_items))

    def _episodes(self, run_id: str, generation: int, jobs: list[tuple[Harness, str, str | None]],
                  snapshot: Snapshot, plan_id: str) -> list[dict[str, Any]]:
        # replicate by replicate, task by task, every version in turn: no version waits behind another's queue
        specs = [(harness, role, name, task, replicate) for replicate in range(self.config.repeats)
                 for task in self.tasks for harness, role, name in jobs]
        recovered, attempts = self._recover_episodes(run_id, generation, plan_id, snapshot)

        def run_batch(batch: Sequence[tuple[Harness, str, str | None, TaskSpec, int]]) -> list[dict[str, Any]]:
            return self._run_episode_pool(batch, recovered, attempts, run_id, generation, snapshot, plan_id)

        if not (self.config.early_prune and generation > 0 and self.config.repeats >= 2):
            return run_batch(specs)

        wave1_specs = [spec for spec in specs if spec[4] == 0]
        wave2_specs = [spec for spec in specs if spec[4] > 0]
        wave1_results = run_batch(wave1_specs)
        pruned = self._prunable_candidates(wave1_results, jobs)
        if not pruned:
            return [*wave1_results, *run_batch(wave2_specs)]
        kept_specs = [spec for spec in wave2_specs if spec[2] not in pruned]
        pruned_entries = [
            {"episode": None, "task": task.id, "harness": harness.digest, "role": role, "candidate": name,
             "replicate": replicate, "guidance": snapshot.guidance[task.id], "status": "pruned",
             "tokens": 0, "metered_tokens": 0, "seconds": 0.0}
            for harness, role, name, task, replicate in wave2_specs
            if name in pruned
        ]
        return [*wave1_results, *run_batch(kept_specs), *pruned_entries]

    def _episode(self, run_id: str, generation: int, harness: Harness, role: str, name: str | None,
                 task: TaskSpec, replicate: int, snapshot: Snapshot, plan_id: str,
                 first_attempt: int, *, relay_stage: int | None = None) -> dict[str, Any]:
        entry: dict[str, Any] = {"task": task.id, "harness": harness.digest, "role": role, "candidate": name,
                                 "replicate": replicate, "guidance": snapshot.guidance[task.id],
                                 **({"relay_stage": relay_stage} if relay_stage is not None else {})}
        gate = self._gate(self.config.model or harness.agent.model)
        # attempts that count against ``retries``; reruns after a confirmed outage; reruns after a service failure
        # that no outage explains
        failures = outages = unstable = 0
        while True:
            if self.cancel.is_set():
                raise Cancelled("the run was cancelled")
            if gate is not None:
                gate.wait(self.cancel)  # no episode starts while the service is down
            attempt = first_attempt + failures + outages + unstable
            stage_text = f"接力{relay_stage}" if relay_stage is not None else ""
            label = f"{run_id} 第{generation}代{stage_text} {'在任' if role == 'incumbent' else f'候选{name}'} {task.id}"
            config = EpisodeConfig(
                task=task.id, harness=harness.digest, minutes=self.config.minutes, tokens=self.config.tokens,
                model=self.config.model, label=label, limits=self.config.limits, expert_until=snapshot.until,
                local_selection=self.config.local_selection,
                context={"evolution": run_id, "generation": generation, "role": role, "candidate": name,
                         "replicate": replicate, "attempt": attempt, "round_plan": plan_id,
                         **({"relay_stage": relay_stage} if relay_stage is not None else {})},
            )
            try:
                result = run_episode(config, self.ledger_path, bench=self.benches[task.id], cancel=self.cancel,
                                     start_cache=self._start_cache)
            except (Cancelled, KeyboardInterrupt):
                raise
            except Exception as error:  # a broken episode is recorded and retried, never fatal to the run
                entry.update({"episode": None, "status": "error", "error": f"{type(error).__name__}: {error}"[:1000],
                              "attempt": attempt})
                failures += 1
                if failures > self.config.retries:
                    break
                continue
            scored = result.get("scored") or {}
            entry.update({"episode": result["episode"], "status": result["status"], "score": result["score"],
                          "predictions": result["predictions"],
                          "predictions_digest": scored.get("predictions_digest"), **_scored_program(scored),
                          "score_limits_match": True,
                          "attempt": attempt, "tokens": result["tokens"], "metered_tokens": result["metered_tokens"],
                          "seconds": result["seconds"]})
            entry.pop("error", None)
            if result["status"] != "infra_failed":
                entry.pop("upstream_error", None)
                break
            entry["upstream_error"] = (result.get("upstream_error") or "")[:500]
            if gate is not None and outages < OUTAGE_RERUNS and gate.confirm(entry["upstream_error"]):
                outages += 1  # the service was down: this attempt says nothing about the harness
                continue
            if unstable < SERVICE_RERUNS:
                # the service answers a test request, yet the session got no reply or spent most of its time on
                # failed calls: an unstable provider, not the harness. Counted against ``retries`` these lost the
                # slot after two attempts, and with it the round's comparison on that task (both sides need every
                # replicate): on 2026-11-17/18 pjlab failed most calls for hours and 31-46 % of the attempts of a
                # round ended this way
                unstable += 1
                continue
            failures += 1
            if failures > self.config.retries:
                break
        if outages:
            entry["outage_reruns"] = outages
        if unstable:
            entry["service_reruns"] = unstable
        self._notify("episode", entry)
        return entry


__all__ = ["Evolution", "EvolveConfig", "RunBusy", "RunStopped", "ServiceDown", "Snapshot", "State", "claim_resume",
           "finish_run", "owner", "result_of", "run_models", "spent_by", "stale", "usable"]
