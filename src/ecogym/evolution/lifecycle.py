"""Evolution run lifecycle primitives: outage gating, process liveness, resume claiming, and early finish."""

from __future__ import annotations

import contextlib
import functools
import os
import re
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..episodes import providers
from ..episodes.agent import Cancelled
from ..harness import HarnessError, resolve
from ..kernel.ledger import Ledger
from .fold import evolver_sessions, fold, spending

if TYPE_CHECKING:
    from .loop import EvolveConfig

PROBE_EVERY = 60.0  # seconds between test requests while a model service is down
OUTAGE_RERUNS = 2  # reruns of one episode after confirmed outages, on top of ``retries``
# reruns of one episode that ended ``infra_failed`` while the service answered a test request (a provider that
# fails many calls, not all), on top of ``retries``: bounded so one congested slot does not stall a whole round
SERVICE_RERUNS = 1


class RunStopped(RuntimeError):
    """The run was asked to stop between rounds."""


class ServiceDown(RuntimeError):
    """No model replies: every episode of a round, or the evolver, got none. The run fails and can be resumed.

    ``outage`` is False when the episodes failed on the host instead (every one an ``error``, or a mix): waiting
    for the model service would not help, so the run is not resumed by itself (``EvolveConfig.auto_resume``)."""

    def __init__(self, message: str, *, outage: bool = True) -> None:
        super().__init__(message)
        self.outage = outage


class RunBusy(RuntimeError):
    """The ledger says the run is running, in a process that is still alive."""


def owner() -> dict[str, Any]:
    """The process that runs a round."""

    return {"pid": os.getpid(), "host": socket.gethostname()}


class ServiceGate:
    """One model service as a run's sessions see it: up, or down since a confirmed outage."""

    def __init__(self, ref: str, limit: float, log: Callable[[str, dict[str, Any]], None]) -> None:
        self.ref, self.limit, self.log = ref, limit, log
        self._lock = threading.Lock()
        self._up = threading.Event()
        self._up.set()
        self._since = 0.0
        self._probes = 0
        self._probing = False
        self._gave_up: tuple[str, bool] | None = None

    def _refused(self, note: str) -> ServiceDown:
        return ServiceDown(f"{self.ref} refuses requests ({note}); fix the key or the model, then resume",
                           outage=False)

    def confirm(self, error: str) -> bool:
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


def model_key(ref: str | None) -> str | None:
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
    return [key for key in dict.fromkeys(model_key(ref) for ref in (agents, config.evolver_model)) if key]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
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
    """Says running, but the process named in its latest round (``owner``) is gone."""

    if view["status"] != "running":
        return False
    proc_owner = view.get("owner") or {}
    if proc_owner.get("host") == socket.gethostname() and isinstance(proc_owner.get("pid"), int):
        booted = _booted_at()
        if booted is not None and datetime.fromisoformat(view["updated_at"]).timestamp() < booted:
            return True
        return proc_owner["pid"] == os.getpid() or not _pid_alive(proc_owner["pid"])
    config = view["config"]
    quiet = (datetime.now(UTC) - datetime.fromisoformat(view["updated_at"])).total_seconds()
    longest = max(float(config.get("minutes") or 0), float(config.get("evolver_minutes") or 0)) * 60
    return quiet > longest * 2 + 1800 + float(config.get("outage_minutes") or 0) * 60


def claim_resume(ledger_path: Path, run_id: str, *, by: str, why: str | None = None, force: bool = False,
                 expect: int | None = None) -> dict[str, Any]:
    """Record that this process continues a stopped or failed run (or one whose process is gone), before it starts."""

    from ..kernel.ledger import LedgerConflict

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


def result_of(view: dict[str, Any], spent: dict[str, Any] | None = None) -> dict[str, Any]:
    """The body of ``evolve_finished`` without its timing: the final version and what the run spent."""

    episodes = [item for round_ in view["rounds"] for item in round_["episodes"]]
    evolvers = evolver_sessions(view)
    body = {
        "seed": view["seed"]["digest"], "final": view["incumbent"], "adoptions": len(view["adoptions"]),
        "rounds": sum(1 for round_ in view["rounds"] if round_["finished"]), "episodes": len(episodes),
        "tokens": sum(item.get("tokens") or 0 for item in [*episodes, *evolvers]),
        "metered_tokens": sum(item.get("metered_tokens") or 0 for item in [*episodes, *evolvers]),
    }
    if spent is not None:
        body.update({"tokens": spent["tokens"], "metered_tokens": spent["metered_tokens"],
                     "attempts": spent["episodes"]["attempts"] + spent["verification"]["attempts"], "spent": spent})
    return body


def spent_by(ledger: Ledger, view: dict[str, Any]) -> dict[str, Any]:
    """What one run spent (``fold.spending``)."""

    return spending(ledger, {view["run_id"]: view})[view["run_id"]]


def finish_run(ledger_path: Path, run_id: str, *, force: bool = False) -> dict[str, Any]:
    """End a stopped or failed run where it is, so it can be verified like a run that ran all its generations."""

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
        body = {**result_of(view, spent_by(ledger, view)), "ended_early": True,
                "planned_generations": view["config"]["generations"], "last_generation": max(finished),
                "was": view["status"]}
        ledger.append(run_id, "evolve_finished", body, after=events[-1].seq)
    return body


__all__ = [
    "OUTAGE_RERUNS",
    "PROBE_EVERY",
    "SERVICE_RERUNS",
    "RunBusy",
    "RunStopped",
    "ServiceDown",
    "ServiceGate",
    "claim_resume",
    "finish_run",
    "model_key",
    "owner",
    "result_of",
    "run_models",
    "spent_by",
    "stale",
]
