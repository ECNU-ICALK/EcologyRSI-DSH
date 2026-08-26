"""Host-owned per-run admission for complete strict-origin sample chains."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


DEFAULT_SAMPLE_CONCURRENCY = 64
MAX_SAMPLE_CONCURRENCY = 128
HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK = 4


def validate_sample_concurrency(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_SAMPLE_CONCURRENCY
    ):
        raise ValueError(
            "sample_concurrency must be between 1 and "
            f"{MAX_SAMPLE_CONCURRENCY}"
        )
    return value


@dataclass
class _RunAdmissionState:
    limit: int
    semaphore: threading.BoundedSemaphore
    active: int = 0
    waiting: int = 0


class RunSampleAdmission:
    """Enforce one immutable sample-chain concurrency limit per run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, _RunAdmissionState] = {}

    def _state_for(self, run_id: str, limit: int) -> _RunAdmissionState:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be non-empty text")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("admission limit must be a positive integer")
        with self._lock:
            state = self._states.get(run_id)
            if state is None:
                state = _RunAdmissionState(
                    limit=limit,
                    semaphore=threading.BoundedSemaphore(limit),
                )
                self._states[run_id] = state
            elif state.limit != limit:
                raise ValueError(
                    f"run {run_id!r} has frozen admission limit {state.limit}, "
                    f"not {limit}"
                )
            return state

    @contextmanager
    def admit(self, run_id: str, limit: int) -> Iterator[None]:
        state = self._state_for(run_id, limit)
        with self._lock:
            state.waiting += 1
        try:
            state.semaphore.acquire()
        except BaseException:
            with self._lock:
                state.waiting -= 1
            raise
        with self._lock:
            state.waiting -= 1
            state.active += 1
        try:
            yield
        finally:
            with self._lock:
                state.active -= 1
            state.semaphore.release()

    def snapshot(self, run_id: str) -> dict[str, int]:
        with self._lock:
            state = self._states.get(run_id)
            if state is None:
                return {"limit": 0, "active": 0, "waiting": 0}
            return {
                "limit": state.limit,
                "active": state.active,
                "waiting": state.waiting,
            }


__all__ = [
    "DEFAULT_SAMPLE_CONCURRENCY",
    "HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK",
    "MAX_SAMPLE_CONCURRENCY",
    "RunSampleAdmission",
    "validate_sample_concurrency",
]
