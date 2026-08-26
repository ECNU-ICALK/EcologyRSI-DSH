"""Host-owned per-run admission for complete strict-origin sample chains."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from ..core.errors import (
    dsh_native_runtime_error_in_chain,
    dsh_native_runtime_retryable,
)


DEFAULT_SAMPLE_CONCURRENCY = 64
MAX_SAMPLE_CONCURRENCY = 128
HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK = 4
_INITIAL_ADAPTIVE_CONCURRENCY = 8


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
    adaptive_limit: int
    active: int = 0
    waiting: int = 0
    successful_since_adjustment: int = 0
    congestion_events: int = 0
    adjustment_epoch: int = 0
    congestion_epoch: int = 0


class RunSampleAdmission:
    """Enforce one immutable logical limit with adaptive physical admission."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
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
                    adaptive_limit=min(limit, _INITIAL_ADAPTIVE_CONCURRENCY),
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
        with self._condition:
            state.waiting += 1
            try:
                while state.active >= state.adaptive_limit:
                    self._condition.wait()
            except BaseException:
                state.waiting -= 1
                raise
            state.waiting -= 1
            state.active += 1
            admission_adjustment_epoch = state.adjustment_epoch
            admission_congestion_epoch = state.congestion_epoch
        try:
            yield
        except BaseException as exc:
            with self._condition:
                state.active -= 1
                dsh_error = dsh_native_runtime_error_in_chain(exc)
                if (
                    dsh_error is not None
                    and dsh_native_runtime_retryable(dsh_error)
                    and admission_congestion_epoch == state.congestion_epoch
                ):
                    state.adaptive_limit = max(1, state.adaptive_limit // 2)
                    state.successful_since_adjustment = 0
                    state.congestion_events += 1
                    state.adjustment_epoch += 1
                    state.congestion_epoch += 1
                self._condition.notify_all()
            raise
        else:
            with self._condition:
                state.active -= 1
                if admission_adjustment_epoch == state.adjustment_epoch:
                    state.successful_since_adjustment += 1
                    if (
                        state.adaptive_limit < state.limit
                        and state.successful_since_adjustment
                        >= state.adaptive_limit
                    ):
                        state.adaptive_limit = min(
                            state.limit,
                            state.adaptive_limit + 1,
                        )
                        state.successful_since_adjustment = 0
                        state.adjustment_epoch += 1
                self._condition.notify_all()

    def snapshot(self, run_id: str) -> dict[str, int]:
        with self._lock:
            state = self._states.get(run_id)
            if state is None:
                return {
                    "limit": 0,
                    "adaptive_limit": 0,
                    "active": 0,
                    "waiting": 0,
                    "congestion_events": 0,
                }
            return {
                "limit": state.limit,
                "adaptive_limit": state.adaptive_limit,
                "active": state.active,
                "waiting": state.waiting,
                "congestion_events": state.congestion_events,
            }


__all__ = [
    "DEFAULT_SAMPLE_CONCURRENCY",
    "HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK",
    "MAX_SAMPLE_CONCURRENCY",
    "RunSampleAdmission",
    "validate_sample_concurrency",
]
