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
4. the passing candidate with the largest gain becomes the incumbent;
5. after an adoption, the programs of the new incumbent and of the version it
   replaced are judged again on the training tasks' sealed validation period
   (``verify.check_adoptions``): a run adopts small gains chosen on one hidden
   period, and this says how much of each holds on a period nobody saw.
   People read it; the evolver never does.

An episode whose agent never got a model reply is retried (``retries``) and
otherwise left out: service failures are not harness effects. Before a retry
the loop sends the model one test request (``ServiceGate``): when that gets no
answer either, the service is down, no session starts until it answers again
(for up to ``outage_minutes``), and the episode the outage cut short runs
again without using up its retries. A candidate missing usable episodes on
any training task is not adopted that round. An evolver session that got no
reply, or that failed model calls kept from working for more than
``STALL_SHARE`` of its time while it left no candidate to compare, is reviewed
once more, after the outage; a second such session, or an outage longer than
``outage_minutes``, stops the run (``ServiceDown``, resumable, and resumed by
the service itself with ``auto_resume``): its generation runs again rather
than being spent on the outage.
Failed candidates stay in the record; later evolvers read them in
``archive.md``. All loop state is recomputed from the ledger at every round,
so ``Evolution.run`` continues an interrupted run after its last finished
round. ``round_started`` names the process that runs it (``owner``), so
another process can tell a live run from a dead one (``stale``). A stopped
run can be ended where it is (``finish_run``), so its result can be verified.
Holdout tasks and sealed stages are touched only by
``ecogym.evolution.verify``.
"""

from __future__ import annotations

import contextlib
import functools
import math
import os
import re
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__
from ..episodes import expert, providers
from ..episodes.agent import Cancelled
from ..episodes.judge import Limits, load_predictions, reference_scores
from ..episodes.runner import STALL_SHARE, EpisodeConfig, run_episode
from ..episodes.workspace import LOCAL_STAGE, SCORED_STAGE
from ..evaluation.harness import Bench, Evaluation
from ..evaluation.suite import TaskPair, adjusted_confidence, compare_suite
from ..harness import Harness, HarnessError, behavior_digest, resolve
from ..kernel import clock
from ..kernel.canonical import digest
from ..kernel.ledger import Ledger, LedgerConflict
from ..task.spec import TaskSpec
from ..task.suite import signed_rank_reachable, suite_spec
from .evolver import EvolverSettings, run_evolver
from .fold import EVENT_KINDS, SCORED_STATUSES, evolver_sessions, fold

PATCH_LIMIT = 60_000  # characters of unified diff kept per candidate in the ledger


class RunStopped(RuntimeError):
    """The run was asked to stop between rounds."""


class ServiceDown(RuntimeError):
    """No model replies: every episode of a round, or the evolver, got none. The run fails and can be resumed.

    ``outage`` is False when the episodes failed on the host instead (every one an ``error``, or a mix): waiting
    for the model service would not help, so the run is not resumed by itself (``EvolveConfig.auto_resume``)."""

    def __init__(self, message: str, *, outage: bool = True) -> None:
        super().__init__(message)
        self.outage = outage


@dataclass(frozen=True, slots=True)
class EvolveConfig:
    suite: str
    harness: str = "seed"
    generations: int = 3
    candidates: int = 2
    repeats: int = 2  # one episode per version and task makes adoptions fragile (docs/evolution.md, section 16)
    minutes: float = 20.0
    tokens: int = 3_000_000
    model: str | None = None  # pin the inner agents' model (<provider>/<model>); None lets each harness choose
    evolver_minutes: float = 30.0
    evolver_tokens: int = 4_000_000
    evolver_model: str = "deepseek-v4-pro"
    concurrency: int = 3
    retries: int = 1
    check_adoptions: bool = True  # each adoption against the version it replaced, on the sealed validation period
    outage_minutes: float = 120.0  # how long a round waits for the model service to answer again (0: not at all)
    auto_resume: bool = True  # the service continues the run by itself after it restarts or the service recovers
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self) -> None:
        if self.generations < 1 or self.repeats < 1 or self.concurrency < 1 or self.retries < 0:
            raise ValueError("generations, repeats and concurrency must be positive, retries non-negative")
        if not 1 <= self.candidates <= 6:
            raise ValueError("candidates must lie in 1..6")
        if self.minutes <= 0 or self.evolver_minutes <= 0 or self.tokens <= 0 or self.evolver_tokens <= 0:
            raise ValueError("budgets must be positive")
        if not (math.isfinite(self.outage_minutes) and self.outage_minutes >= 0):
            raise ValueError("outage_minutes must be a finite number of minutes, 0 or more")
        for ref in (self.model, self.evolver_model):
            if ref:
                providers.split(ref)  # <provider>/<model> or a DeepSeek id; ModelError is a ValueError

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "ecogym.evolve-config/1", **asdict(self)}

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> EvolveConfig:
        values = {key: value for key, value in body.items() if key != "schema"}
        values["limits"] = Limits(**values.get("limits", {}))
        # a run recorded before 0.3.2 keeps its behaviour: it never resumes by itself and spends no sealed look on
        # adoption checks (an outage is waited out: that changes no measurement)
        values.setdefault("auto_resume", False)
        values.setdefault("check_adoptions", False)
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
                          + "; ".join(reasons)[:1500],
                          outage=all(entry.get("status") == "infra_failed" for entry in entries))


PROBE_EVERY = 60.0  # seconds between test requests while a model service is down
OUTAGE_RERUNS = 3  # reruns of one episode after confirmed outages, on top of ``retries``


class ServiceGate:
    """One model service as a run's sessions see it: up, or down since a confirmed outage.

    A session that got no reply asks ``confirm``: a test request (``providers.answers``) tells an outage from one
    failed session. While the service is down no session starts (``wait``); one waiter sends a test request every
    ``PROBE_EVERY`` seconds and all go on once it answers. A session cut short by the outage runs again without
    using up its retries. After ``limit`` seconds down, every waiter raises ``ServiceDown``: the run stops and can
    be resumed (by the service itself, ``EvolveConfig.auto_resume``). A provider that refuses the test request
    (bad key, no access, unknown model) is not waited for: the run stops at once, not to be resumed by itself.
    ``log`` records the outage's start and end.
    """

    def __init__(self, ref: str, limit: float, log: Callable[[str, dict[str, Any]], None]) -> None:
        self.ref, self.limit, self.log = ref, limit, log
        self._lock = threading.Lock()
        self._up = threading.Event()
        self._up.set()
        self._since = 0.0
        self._probes = 0
        self._probing = False
        self._gave_up: tuple[str, bool] | None = None  # why every waiter stops, and whether it was an outage

    def _refused(self, note: str) -> ServiceDown:
        return ServiceDown(f"{self.ref} refuses requests ({note}); fix the key or the model, then resume",
                           outage=False)

    def confirm(self, error: str) -> bool:
        """After a session got no reply: True when the service is down (now, or already), so the caller waits and
        runs the session again; False when the service answers (the failure was the session's own).
        ``ServiceDown`` when the provider refuses: no session of this model could get a reply."""

        if self.limit <= 0:
            return False
        if self._gave_up is not None or not self._up.is_set():
            return True
        state, note = providers.answers(self.ref)
        if state == "refused":
            raise self._refused(note)
        if state == "ok":
            return False
        with self._lock:
            if self._up.is_set():
                self._up.clear()
                self._since, self._probes = time.monotonic(), 1
                self.log("outage_started", {"model": self.ref, "error": (error or "")[:500], "probe": note,
                                            "limit_minutes": round(self.limit / 60, 1)})
        return True

    def wait(self, cancel: threading.Event) -> None:
        """Return at once while the service is up, else once it answers again; ``ServiceDown`` when the outage
        has lasted ``limit`` seconds, ``Cancelled`` when the run is cancelled."""

        while True:
            if self._gave_up is not None:
                raise ServiceDown(self._gave_up[0], outage=self._gave_up[1])
            if self._up.is_set():
                return
            if cancel.is_set():
                raise Cancelled("the run was cancelled")
            with self._lock:
                mine, self._probing = not self._probing, True
            if not mine:
                self._up.wait(5.0)
                continue
            try:
                if cancel.wait(PROBE_EVERY):
                    raise Cancelled("the run was cancelled")
                state, note = providers.answers(self.ref)
                with self._lock:
                    self._probes += 1
                    waited = time.monotonic() - self._since
                    if state == "ok":
                        self._up.set()
                    elif state == "refused":
                        self._gave_up = (str(self._refused(note)), False)
                    elif waited >= self.limit:
                        self._gave_up = (f"{self.ref} gave no reply for {waited / 60:.0f} minutes "
                                         f"({self._probes} test requests; the last: {note})", True)
                    if state == "ok" or self._gave_up is not None:
                        self.log("outage_ended", {"model": self.ref, "recovered": state == "ok",
                                                  "seconds": round(waited), "probes": self._probes,
                                                  **({} if state == "ok" else {"error": note})})
            finally:
                with self._lock:
                    self._probing = False


def _model_key(ref: str | None) -> str | None:
    if not ref:
        return None
    try:
        return providers.canonical(ref)
    except ValueError:
        return None


def run_models(config: EvolveConfig, seed: str) -> list[str]:
    """The models a run was started with, canonical: its inner agents' (pinned, or the starting harness's) and its
    evolver's. The loop waits out these models' outages; the service resumes a run once they answer again."""

    agents = config.model
    if not agents:
        with contextlib.suppress(HarnessError, OSError, ValueError):
            agents = resolve(seed).agent.model
    return [key for key in dict.fromkeys(_model_key(ref) for ref in (agents, config.evolver_model)) if key]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists and belongs to someone else
        return True
    return True


@functools.cache
def _booted_at() -> float | None:
    """When this machine started (seconds since the epoch), if the system says."""

    with contextlib.suppress(OSError, ValueError, subprocess.SubprocessError):
        if os.path.exists("/proc/stat"):
            for line in Path("/proc/stat").read_text().splitlines():
                if line.startswith("btime "):
                    return float(line.split()[1])
        text = subprocess.run(["sysctl", "-n", "kern.boottime"], capture_output=True, text=True, timeout=5,
                              check=True).stdout
        found = re.search(r"sec = (\d+)", text)
        if found:
            return float(found.group(1))
    return None


def stale(view: dict[str, Any]) -> bool:
    """Says running, but the process named in its latest round (``owner``) is gone.

    The calling process counts as gone: callers check their own threads first. A run that logged its last event
    before this machine started is gone whatever process has its pid now. Without an owner on this host (older
    runs, another machine) the run is stale once it has logged nothing for longer than any of its steps can take.
    """

    if view["status"] != "running":
        return False
    owner = view.get("owner") or {}
    if owner.get("host") == socket.gethostname() and isinstance(owner.get("pid"), int):
        booted = _booted_at()
        if booted is not None and datetime.fromisoformat(view["updated_at"]).timestamp() < booted:
            return True
        return owner["pid"] == os.getpid() or not _pid_alive(owner["pid"])
    config = view["config"]
    quiet = (datetime.now(UTC) - datetime.fromisoformat(view["updated_at"])).total_seconds()
    longest = max(float(config.get("minutes") or 0), float(config.get("evolver_minutes") or 0)) * 60
    return quiet > longest * 2 + 1800 + float(config.get("outage_minutes") or 0) * 60  # an outage wait logs nothing


class RunBusy(RuntimeError):
    """The ledger says the run is running, in a process that is still alive."""


def claim_resume(ledger_path: Path, run_id: str, *, by: str, why: str | None = None, force: bool = False,
                 expect: int | None = None) -> dict[str, Any]:
    """Record that this process continues a stopped or failed run (or one whose process is gone), before it starts:
    ``evolve_resumed`` names this process as the run's owner, so the run reads as running at once and no other
    process continues it too. The event is appended only if nothing was logged since the run was read here
    (``Ledger.append(after=...)``; ``expect``: nor since the caller's own reading), so two processes deciding at
    the same moment cannot both win. Returns the view the decision was taken on.

    ``KeyError``: no such run; ``ValueError``: it has finished; ``RunBusy``: it runs in a live process (``force``
    skips this check); ``LedgerConflict``: it changed meanwhile."""

    with Ledger(ledger_path) as ledger:
        events = ledger.events(run_id)
        if not events or events[0].kind != "evolve_started":
            raise KeyError(f"no evolution run {run_id}")
        view = fold(events)
        if expect is not None and events[-1].seq != expect:
            raise LedgerConflict(f"{run_id} changed meanwhile")
        if view["status"] == "finished":
            raise ValueError(f"{run_id} has finished")
        if view["status"] == "running" and not force and not stale(view):
            raise RunBusy(f"{run_id} may still be running in another process (owner {view.get('owner') or 'unknown'}"
                          f", last event {view['updated_at']})")
        ledger.append(run_id, "evolve_resumed",
                      {"by": by, "why": why or view.get("stop_reason") or ("stale" if view["status"] == "running"
                                                                           else None),
                       "status": view["status"], "owner": owner()}, after=events[-1].seq)
    return view


def result_of(view: dict[str, Any]) -> dict[str, Any]:
    """The body of ``evolve_finished`` without its timing: the final version and what the run spent."""

    episodes = [item for round_ in view["rounds"] for item in round_["episodes"]]
    evolvers = evolver_sessions(view)
    return {
        "seed": view["seed"]["digest"], "final": view["incumbent"], "adoptions": len(view["adoptions"]),
        "rounds": sum(1 for round_ in view["rounds"] if round_["finished"]), "episodes": len(episodes),
        "tokens": sum(item.get("tokens") or 0 for item in [*episodes, *evolvers]),
        "metered_tokens": sum(item.get("metered_tokens") or 0 for item in [*episodes, *evolvers]),
    }


def finish_run(ledger_path: Path, run_id: str, *, force: bool = False) -> dict[str, Any]:
    """End a stopped or failed run where it is, so it can be verified like a run that ran all its generations.

    The incumbent after its last finished round is the final version and the remaining generations never run
    (``ended_early``; the run cannot be resumed afterwards, and ``fold`` ignores whatever a straggling process
    logs later). Verification needs a finished run because a run that could still continue has no final version
    yet. ``RunBusy`` when it runs in a live process (``force`` skips that check; a caller running it in its own
    threads must stop it first), ``LedgerConflict`` when it changed while this was deciding.
    """

    with Ledger(ledger_path) as ledger:
        events = ledger.events(run_id)
        if not events or events[0].kind != "evolve_started":
            raise KeyError(f"no evolution run {run_id}")
        view = fold(events)
        if view["status"] == "finished":
            raise ValueError(f"{run_id} has already finished")
        if view["status"] == "running" and not force and not stale(view):
            raise RunBusy(f"{run_id} may still be running in another process (owner {view.get('owner') or 'unknown'}"
                          "); stop it there first")
        finished = [round_["generation"] for round_ in view["rounds"] if round_["finished"]]
        if not finished:
            raise ValueError(f"{run_id} has no finished round: it has nothing to verify")
        body = {**result_of(view), "ended_early": True, "planned_generations": view["config"]["generations"],
                "last_generation": max(finished), "was": view["status"]}
        ledger.append(run_id, "evolve_finished", body, after=events[-1].seq)
    return body


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
        self._notify_lock = threading.Lock()
        self._gates: dict[str, ServiceGate] = {}  # model -> its gate, for the models the run was started with

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
        result = {**result_of(self.view(run_id)), "seconds": round(time.monotonic() - started, 1)}
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
        if adopted:
            self._check_adoptions(run_id, [generation])

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

    def _propose(self, run_id: str, generation: int, incumbent: Harness, view: dict[str, Any],
                 state: State, snapshot: Snapshot) -> list[dict[str, Any]]:
        config = self.config
        settings = EvolverSettings(minutes=config.evolver_minutes, tokens=config.evolver_tokens,
                                   model=config.evolver_model, candidates=config.candidates)
        with Ledger(self.ledger_path) as ledger:  # the training tasks' guidance only: never a holdout task's
            advice = expert.guidance(ledger, self.tasks, until=snapshot.until)
        gate, reruns = self._gate(config.evolver_model), 0
        while True:
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
            budget = config.evolver_minutes * 60
            stalled = float(session.get("stalled_seconds") or 0.0)
            # failed calls kept the evolver from working for most of its time and it left nothing to compare: the
            # provider decided this generation, not the evidence (evo:0f0d4068b8b1 generation 5: 871 of 900 s, two
            # duplicates without proposals), so the generation is reviewed again rather than spent
            crippled = stalled > STALL_SHARE * budget and not any(item["status"] == "proposed"
                                                                  for item in outcome["candidates"])
            if session["status"] != "infra_failed" and not crippled:
                break
            body = {"generation": generation, **{key: session.get(key) for key in (
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
                    "base": candidate.get("base"),  # the archived version it was reset to; None: the incumbent
                    "summary": candidate.get("summary", ""), "proposal": candidate.get("proposal", ""),
                    "agent": candidate.get("agent"), "changes": patches, "smoke": candidate.get("smoke"),
                    "problems": candidate.get("problems", []), "warnings": candidate.get("warnings", [])}
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
                                                 "upstream_error", "continuations")},
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
        gate = self._gate(self.config.model or harness.agent.model)
        failures = outages = 0  # attempts that count against ``retries``; reruns after a confirmed outage
        while True:
            if self.cancel.is_set():
                raise Cancelled("the run was cancelled")
            if gate is not None:
                gate.wait(self.cancel)  # no episode starts while the service is down
            attempt = failures + outages
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
                failures += 1
                if failures > self.config.retries:
                    break
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
            if gate is not None and outages < OUTAGE_RERUNS and gate.confirm(entry["upstream_error"]):
                outages += 1  # the service was down: this attempt says nothing about the harness
                continue
            failures += 1
            if failures > self.config.retries:
                break
        if outages:
            entry["outage_reruns"] = outages
        self._notify("episode", entry)
        return entry


__all__ = ["Evolution", "EvolveConfig", "RunBusy", "RunStopped", "ServiceDown", "Snapshot", "State", "claim_resume",
           "finish_run", "owner", "result_of", "run_models", "stale", "usable"]
