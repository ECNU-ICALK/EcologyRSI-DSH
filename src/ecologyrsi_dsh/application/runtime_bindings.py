"""Validated candidate identity and event cursors shared by application runtimes."""
from __future__ import annotations
import threading
from typing import Any
from collections.abc import Callable, Mapping
from ..core.ledger import EventLedger
from ..core.state import validate_identity_binding

def dsh_revision_snapshot(ledger: EventLedger, run_id: str) -> dict[str, int]:
    """Read DSH cursors without materializing the run projection."""

    run_state_revision = ledger.latest_run_seq(run_id)
    if run_state_revision == 0:
        raise KeyError(f"unknown run: {run_id}")
    return {
        "run_state_revision": run_state_revision,
        "ledger_expected_revision": ledger.latest_seq(),
    }


class ValidatedCandidateIdentityCache:
    """Cache immutable candidate bindings after one trusted state replay.

    Candidate identity is fixed by ``CandidateSpawned`` and every later event
    can only repeat that exact binding.  The first miss for a run therefore
    uses the full projector (retaining all genome, compiler, and tamper
    checks), then serves defensive copies.  A miss after the run cursor moves
    refreshes the snapshot so candidates spawned in later generations remain
    discoverable after normal execution or process recovery.
    """

    def __init__(
        self,
        ledger: EventLedger,
        state_provider: Callable[[str], Any],
    ) -> None:
        self._ledger = ledger
        self._state_provider = state_provider
        self._guard = threading.Lock()
        self._run_locks: dict[str, threading.RLock] = {}
        self._snapshots: dict[
            str, tuple[int, dict[str, dict[str, str]]]
        ] = {}

    @staticmethod
    def _key(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    def _run_lock(self, run_id: str) -> threading.RLock:
        with self._guard:
            lock = self._run_locks.get(run_id)
            if lock is None:
                lock = threading.RLock()
                self._run_locks[run_id] = lock
            return lock

    def get(self, run_id: str, candidate_id: str) -> dict[str, str] | None:
        run_id = self._key(run_id, "run_id")
        candidate_id = self._key(candidate_id, "candidate_id")
        with self._run_lock(run_id):
            with self._guard:
                snapshot = self._snapshots.get(run_id)
                cached = None if snapshot is None else snapshot[1].get(candidate_id)
            if cached is not None:
                return dict(cached)

            latest_run_seq = self._ledger.latest_run_seq(run_id)
            if latest_run_seq == 0:
                raise KeyError(f"unknown run: {run_id}")
            if snapshot is not None and snapshot[0] == latest_run_seq:
                return None

            # Keep the per-run lock across replay: concurrent sample workers
            # share one validation pass instead of replaying the same stream.
            state = self._state_provider(run_id)
            if state.run.run_id != run_id or not state.events:
                raise ValueError("candidate identity projection belongs to another run")
            projected_seq = int(state.events[-1].seq)
            if projected_seq < latest_run_seq:
                raise RuntimeError("candidate identity projection is behind the ledger")
            bindings: dict[str, dict[str, str]] = {}
            for item in state.candidate_identity_bindings:
                if not isinstance(item, Mapping):
                    raise ValueError("candidate identity cache entry is invalid")
                item_candidate_id = self._key(
                    item.get("candidate_id"), "candidate_id"
                )
                binding = validate_identity_binding(item.get("identity_binding"))
                existing = bindings.get(item_candidate_id)
                if existing is not None and existing != binding:
                    raise ValueError("candidate has conflicting identity bindings")
                bindings[item_candidate_id] = binding
            with self._guard:
                self._snapshots[run_id] = (projected_seq, bindings)
            result = bindings.get(candidate_id)
            return dict(result) if result is not None else None

    def forget(self, run_id: str) -> bool:
        run_id = self._key(run_id, "run_id")
        with self._run_lock(run_id), self._guard:
            return self._snapshots.pop(run_id, None) is not None

    def clear(self) -> None:
        with self._guard:
            self._snapshots.clear()
