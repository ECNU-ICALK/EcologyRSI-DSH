"""Host-owned per-run admission for complete strict-origin sample chains."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

from ..core.errors import (
    dsh_native_runtime_error_in_chain,
    dsh_native_runtime_retryable,
    dsh_native_runtime_evaluation_fatal,
)


DEFAULT_SAMPLE_CONCURRENCY = 64
MAX_SAMPLE_CONCURRENCY = 128
INITIAL_SAMPLE_CONCURRENCY = 8
HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK = 4
# Healthy admissions required before the adaptive limit grows by one. This used
# to be the adaptive limit itself, which made recovery unreachable rather than
# merely cautious: climbing from 8 to a configured 64 needed sum(8..63) = 1988
# consecutive healthy origins, and one halving to 1 needed sum(1..63) = 2016.
# A 200-origin epoch cannot spend that, so run:e4332050-18c1-4562-8f03-3c4c8ee3a8bf
# sat at adaptive_limit 1 -- effectively serial -- against a frozen limit of 64
# for a whole epoch after its congestion halvings. A small constant keeps growth
# strictly additive (one step per window, never a jump to the ceiling) while
# making the ceiling reachable inside a single epoch.
ADMISSION_GROWTH_HEALTHY_ADMISSIONS = 2


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
class AdmissionOutcome:
    healthy: bool = True


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
    execution_failure: BaseException | None = None


class RunSampleAdmission:
    """Enforce one immutable logical limit with adaptive physical admission."""

    def __init__(self, *, on_execution_failure: Callable[[str], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._states: dict[str, _RunAdmissionState] = {}
        self._on_execution_failure = on_execution_failure

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
                    # Probe capacity before filling the configured ceiling.
                    # The existing additive recovery also drives warmup.
                    adaptive_limit=min(limit, INITIAL_SAMPLE_CONCURRENCY),
                )
                self._states[run_id] = state
            elif state.limit != limit:
                raise ValueError(
                    f"run {run_id!r} has frozen admission limit {state.limit}, "
                    f"not {limit}"
                )
            return state

    def forget(self, run_id: str) -> bool:
        """Forget a purged run after all of its admissions have quiesced."""

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be non-empty text")
        with self._condition:
            state = self._states.get(run_id)
            if state is None:
                return False
            if state.active or state.waiting:
                raise RuntimeError("cannot forget a run with active or waiting admissions")
            del self._states[run_id]
            self._condition.notify_all()
            return True

    @contextmanager
    def admit(self, run_id: str, limit: int) -> Iterator[AdmissionOutcome]:
        state = self._state_for(run_id, limit)
        with self._condition:
            state.waiting += 1
            try:
                if state.execution_failure is not None:
                    raise state.execution_failure
                while state.active >= state.adaptive_limit:
                    self._condition.wait()
                    if state.execution_failure is not None:
                        raise state.execution_failure
            except BaseException:
                state.waiting -= 1
                raise
            state.waiting -= 1
            state.active += 1
            admission_adjustment_epoch = state.adjustment_epoch
            admission_congestion_epoch = state.congestion_epoch
        outcome = AdmissionOutcome()
        try:
            yield outcome
        except BaseException as exc:
            notify_failure = False
            with self._condition:
                state.active -= 1
                dsh_error = dsh_native_runtime_error_in_chain(exc)
                if state.execution_failure is None and dsh_error is not None and dsh_native_runtime_evaluation_fatal(dsh_error):
                    state.execution_failure = dsh_error
                    notify_failure = True
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
                failure = state.execution_failure
            # Cancel remote siblings before executor shutdown waits for them.
            # Never hold the admission lock while calling another service.
            if notify_failure and self._on_execution_failure is not None:
                try:
                    self._on_execution_failure(run_id)
                except Exception:
                    # The original failure still reaches the run owner, whose
                    # terminal transition retries native quiescence if needed.
                    pass
            if failure is not None and failure is not exc:
                raise failure from exc
            raise
        else:
            with self._condition:
                state.active -= 1
                if not outcome.healthy:
                    state.successful_since_adjustment = 0
                elif admission_adjustment_epoch == state.adjustment_epoch:
                    state.successful_since_adjustment += 1
                    if (
                        state.adaptive_limit < state.limit
                        and state.successful_since_adjustment
                        >= ADMISSION_GROWTH_HEALTHY_ADMISSIONS
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
