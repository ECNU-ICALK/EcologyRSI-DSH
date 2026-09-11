"""Durable, bounded background progression for autonomous evolution runs.

The HTTP request that creates a run should not have to stay open for the whole
search budget.  ``AutoProgressManager`` owns a bounded multi-run worker pool.
Ordinary protocols advance through ``execute_generation``; the Top-2 adaptive
protocol yields at durable phase boundaries through its lane scheduler.  In
both cases the event ledger remains the source of truth and a process restart
can resume a partially written generation.

Only runs whose immutable task manifest contains ``auto_progress=true`` are
scheduled.  Legacy runs and explicit bounded ``auto_advance`` requests retain
their previous manual-step semantics.  The Top-2 adaptive epoch protocol is
scheduled as one durable scheduler turn at a time (screening, one boundary per
admitted finalist lane, or epoch closeout), rather than holding the worker for
an entire 500-origin finalist trajectory.
"""

from __future__ import annotations

from ..evolution.schedule import ADAPTIVE_PROTOCOLS

import math
import os
import sqlite3
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Any

from ..core.errors import (
    FrozenRuntimeBindingDriftError,
    dsh_native_runtime_error_in_chain,
    dsh_native_runtime_retryable,
    find_exception,
)
from ..core.ledger import ConcurrentRunMutationError
from ..core.models import RunStatus, digest
from ..core.redaction import public_exception_summary
from ..core.state import gateway_retry_error_code
from ..evaluators.sample_execution import (
    SampleExecutionControlUnavailableError,
    SampleResultCallbackError,
)
from ..evolution.batches import ResearchResponseContractError
from ..evolution.schedule import (
    PAIRED_LOCAL_EVALUATION_MODE,
    OptimizationSchedule,
)
from ..integrations.dsh_native_runtime import DSH_NATIVE_EXECUTION_PROTOCOL
from ..integrations.model_gateway import gateway_error_in_chain
from ..application.generation_execution import complete_if_budget_exhausted, execute_generation

_AUTO_PROGRESS_METADATA_KEY = "auto_progress"
_DEFAULT_RETRY_LIMIT = 3
_DEFAULT_WORKER_COUNT = 4
_MAX_WORKER_COUNT = 8
_FAILURE_PERSISTENCE_RETRY_SECONDS = 1.0
_RETRY_TIMER_START_RETRY_SECONDS = 1.0
_GATEWAY_RETRY_BASE_SECONDS = 15.0
_GATEWAY_RETRY_MAX_SECONDS = 300.0
_GATEWAY_RETRY_DEADLINE_MAX_SECONDS = 3600.0
_NATIVE_QUIESCENCE_RETRY_BASE_SECONDS = 1.0
_NATIVE_QUIESCENCE_RETRY_MAX_SECONDS = 30.0
_SCHEDULER_COOLDOWN_MAINTENANCE_SECONDS = 5.0
_SCHEDULER_ORPHAN_MAINTENANCE_SECONDS = 30.0
_RESEARCH_RETRYABLE_RESPONSE_CODES = frozenset(
    {
        "gateway_response_error",
        "research_algorithm_contract_invalid",
    }
)
_TERMINAL_RUN_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.CANCELLED, RunStatus.FAILED}
)
_WorkItem = tuple[str, int]


class _RunIncarnationChanged(KeyError):
    """Raised when queued work no longer names the current RunCreated event."""


@dataclass(frozen=True, slots=True)
class _PendingGatewayRetry:
    run_incarnation: int
    generation: int
    stage: str
    retry_class: str
    failure_id: str
    attempt_anchor_seq: int
    delay_seconds: float
    last_error_code: str


@dataclass(frozen=True, slots=True)
class _DeferredFailure:
    reason: str
    error_code: str
    failure_context: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _DeferredPause:
    reason: str
    code: str = "auto_progress_host_fault"


@dataclass(frozen=True, slots=True)
class _NativeQuiescence:
    request: dict[str, Any]
    runtime: Any
    marker: int | tuple[int, object]


def _run_incarnation(state: Any) -> int:
    events = tuple(getattr(state, "events", ()))
    if not events or getattr(events[0], "kind", None) != "RunCreated":
        raise ValueError("run projection has no leading RunCreated event")
    return int(events[0].seq)


def auto_progress_enabled(state: Any) -> bool:
    """Return whether a replayed run opted into continuous progression."""

    metadata = getattr(getattr(state, "task_manifest", None), "metadata", {})
    return metadata.get(_AUTO_PROGRESS_METADATA_KEY) is True


def _failure_diagnostics(
    state: Any,
    exc: BaseException,
    *,
    stage: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return only host-owned, bounded failure classification and location."""

    generation = int(state.run.generation)
    adaptive = (
        state.task_manifest.metadata.get("optimization_protocol")
        in ADAPTIVE_PROTOCOLS
    )
    context: dict[str, Any] = {
        "generation": generation,
        "stage": str(stage or "generation")[:80],
        "work_unit_kind": "generation",
    }
    if stage in {"preflight", "search", "research", "reflection", "generation.judge"}:
        # These stages precede or follow the sample scheduler. Zero screening
        # records do not mean that a research failure happened in screening.
        context["work_unit_kind"] = stage
    elif adaptive:
        schedule = OptimizationSchedule.from_dict(
            state.task_manifest.metadata["optimization_schedule"]
        )
        screening_count = sum(
            int(event.payload.get("generation", -1)) == generation
            for event in state.candidate_screening_events
        )
        if not schedule.quick and screening_count < 4:
            context.update(stage="screening", work_unit_kind="candidate_screening")
        else:
            formal_total = sum(
                item.scope.origin_count
                for item in state.formal_batch_evaluations
                if item.scope.generation == generation
            )
            expected_formal = schedule.generation_execution_budget(
                cells_per_origin=1
            )["formal_candidate_origins"]
            context.update(
                formal_origin_occurrences_completed=formal_total,
                formal_origin_occurrences_upper_bound=expected_formal,
            )
            completed_trajectories = sum(
                item.generation == generation
                and getattr(item.status, "value", item.status) == "completed"
                for item in state.formal_trajectories
            )
            if completed_trajectories < schedule.finalist_count:
                batch = next(
                    (
                        item
                        for item in reversed(state.formal_batches)
                        if item.generation == generation
                        and state.revision_activation_for(
                            item.candidate_id, item.batch_index
                        )
                        is None
                    ),
                    None,
                )
                batch_evaluation = (
                    state.batch_evaluation_for(
                        batch.candidate_id, batch.batch_index
                    )
                    if batch is not None
                    else None
                )
                comparison = (
                    state.batch_comparison_for(
                        batch.candidate_id,
                        batch.batch_index,
                    )
                    if batch is not None
                    and callable(getattr(state, "batch_comparison_for", None))
                    else None
                )
                paired = (
                    schedule.local_evaluation_mode
                    == PAIRED_LOCAL_EVALUATION_MODE
                )
                needs_local_edit = bool(
                    batch is not None
                    and batch.batch_index < batch.batch_count - 1
                    and (
                        comparison is not None
                        if paired
                        else batch_evaluation is not None
                    )
                )
                context.update(
                    stage="local_edit" if needs_local_edit else "formal_batch",
                    work_unit_kind=(
                        "local_edit" if needs_local_edit else "formal_batch"
                    ),
                )
                if batch is not None:
                    context.update(
                        candidate_id=batch.candidate_id,
                        batch_id=batch.batch_id,
                        batch_index=batch.batch_index,
                        batch_count=batch.batch_count,
                    )
            elif len(
                [item for item in state.holdout_evaluations if item.scope.generation == generation]
            ) < schedule.finalist_count + 1:
                context.update(stage="holdout", work_unit_kind="selection_holdout")
            else:
                context.update(stage="decision", work_unit_kind="epoch_closeout")
    # Preserve stable codes only for concrete Host-owned exception classes.
    # Reading an arbitrary ``error_code`` attribute here would let an
    # untrusted provider exception choose the public terminal classification.
    native_error = dsh_native_runtime_error_in_chain(exc)
    if native_error is not None and native_error.error_code in {
        "structured_child_tool_protocol_error", "structured_child_output_budget_exhausted",
        "structured_result_missing", "structured_child_output_schema_invalid",
        "evaluation_execution_incomplete",
        "structured_child_execution_budget_exhausted",
    }:
        context["failure_domain"] = "model_execution"
        return native_error.error_code, context
    binding_drift = find_exception(exc, FrozenRuntimeBindingDriftError)
    if binding_drift is not None:
        return FrozenRuntimeBindingDriftError.error_code, context
    suffixes = {
        ValueError: "host_value_error",
        TypeError: "host_type_error",
        KeyError: "host_key_error",
        RuntimeError: "host_runtime_error",
        TimeoutError: "host_timeout_error",
    }
    suffix = next(
        (name for kind, name in suffixes.items() if isinstance(exc, kind)),
        "unexpected_error",
    )
    safe_stage = "".join(
        char if char.isalnum() else "_" for char in str(context["stage"]).lower()
    ).strip("_") or "generation"
    return f"auto_progress_{safe_stage}_{suffix}"[:120], context


def _host_fault_pause(
    state: Any,
    exc: BaseException,
    *,
    stage: str | None = None,
) -> _DeferredPause:
    """Build a bounded public pause without retaining exception messages."""

    try:
        _failure_code, context = _failure_diagnostics(state, exc, stage=stage)
    except Exception:  # noqa: BLE001 - diagnostics must not mask the fault
        context = {
            "stage": str(stage or "generation")[:80],
            "work_unit_kind": "generation",
        }
    safe_stage = "".join(
        character
        for character in str(context.get("stage") or "generation")[:80]
        if character.isalnum() or character in {"_", "-"}
    ) or "generation"
    safe_work_unit = "".join(
        character
        for character in str(context.get("work_unit_kind") or "generation")[:80]
        if character.isalnum() or character in {"_", "-"}
    ) or "generation"
    native_error = dsh_native_runtime_error_in_chain(exc)
    fault_detail = public_exception_summary(exc)
    if native_error is not None:
        native_code = str(getattr(native_error, "error_code", "") or "")[:80]
        native_status = getattr(native_error, "status_code", None)
        if native_code:
            fault_detail = native_code
            if isinstance(native_status, int):
                fault_detail += f"/HTTP{native_status}"
    return _DeferredPause(
        reason=(
            "自动推进检测到宿主异常，已暂停并保留当前检查点；"
            f"阶段={safe_stage}，工作单元={safe_work_unit}，"
            f"异常={fault_detail}。"
        )[:500]
    )


class AutoProgressManager:
    """Fairly advance autonomous runs with a bounded background worker pool.

    A worker owns at most one run generation.  A still-running run returns to
    the FIFO queue after that generation, which gives other queued runs a turn.
    The per-run generation lease remains the final guard shared with explicit
    ``/advance`` requests.  Queue and active-run de-duplication are kept here
    as well, so a long Retry-After or model call cannot cause duplicate work.
    When the pool is explicitly configured with more than one worker, one
    blocked model call also cannot stall every pending run.
    """

    def __init__(self, server: Any) -> None:
        self.server = server
        self._queue: Queue[_WorkItem | None] = Queue()
        self._scheduled: set[_WorkItem] = set()
        self._running: set[_WorkItem] = set()
        self._reschedule_requested: set[_WorkItem] = set()
        # A failed generation must not be executed again merely because writing
        # its RunFailed event hit a transient ledger error.  Keep the bounded
        # public reason in memory and let the FIFO worker retry only that
        # terminal transition.  A restart recovers the still-running durable run
        # through ``recover_running`` if the process exits before persistence.
        self._deferred_failures: dict[_WorkItem, _DeferredFailure] = {}
        # Host faults pause instead of consuming the scientific checkpoint. If
        # the lifecycle append is temporarily unavailable, retain only this
        # bounded transition and never replay the work unit first.
        self._deferred_pauses: dict[_WorkItem, _DeferredPause] = {}
        # A request-local gateway retry may already have exhausted its small
        # transport budget while the provider is still queueing work.  Keep
        # the run alive and delay its next generation attempt instead of
        # converting this recoverable boundary into RunFailed.
        self._retry_not_before: dict[_WorkItem, float] = {}
        self._retry_authority: dict[_WorkItem, tuple[int, int]] = {}
        # If the durable retry decision cannot commit, retry that exact report
        # before another gateway invocation.  No cooldown exists until commit.
        self._pending_gateway_retries: dict[_WorkItem, _PendingGatewayRetry] = {}
        # Cooldowns are timer-backed rather than worker sleeps.  A busy
        # provider must not occupy the only worker and block unrelated runs.
        self._retry_timers: dict[_WorkItem, threading.Timer] = {}
        self._state_lock = threading.RLock()
        self._stop = threading.Event()
        self._maintenance_failure_count = 0
        self._retry_limit = self._read_retry_limit()
        self._worker_count = self._read_worker_count()
        self._threads = tuple(
            threading.Thread(
                target=self._worker,
                name=(
                    "ecologyrsi-auto-progress"
                    if index == 0
                    else f"ecologyrsi-auto-progress-{index + 1}"
                ),
                daemon=True,
            )
            for index in range(self._worker_count)
        )
        # Retain the historical attribute for integrations which inspect the
        # primary worker's liveness.  Shutdown uses every pool member.
        self._thread = self._threads[0]
        for worker in self._threads:
            worker.start()

    @staticmethod
    def _read_retry_limit() -> int:
        raw = os.environ.get("ECOLOGYRSI_AUTO_PROGRESS_RETRIES", "")
        if not raw.strip():
            return _DEFAULT_RETRY_LIMIT
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_RETRY_LIMIT
        return max(1, min(value, 8))

    @staticmethod
    def _read_worker_count() -> int:
        raw = os.environ.get("ECOLOGYRSI_AUTO_PROGRESS_WORKERS", "")
        if not raw.strip():
            return _DEFAULT_WORKER_COUNT
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_WORKER_COUNT
        return max(1, min(value, _MAX_WORKER_COUNT))

    def diagnostics(self, run_id: str) -> dict[str, Any]:
        """Return a bounded, side-effect-free scheduler snapshot."""

        target_run_id = str(run_id).strip()
        with self._state_lock:
            running_items = set(self._running)
            scheduled_items = set(self._scheduled)
            cooldown_items = set(self._retry_timers) | set(self._retry_not_before)
            # Queue.Queue exposes its deque only while ``mutex`` is held.  The
            # snapshot is advisory; exact-incarnation de-duplication below
            # filters stale physical entries left behind by purge/recovery.
            with self._queue.mutex:
                physical_queue = tuple(self._queue.queue)

            queued_items: list[_WorkItem] = []
            seen: set[_WorkItem] = set()
            for item in physical_queue:
                if (
                    item is None
                    or item in seen
                    or item not in scheduled_items
                    or item in running_items
                ):
                    continue
                seen.add(item)
                queued_items.append(item)

            target_running = next(
                (item for item in running_items if item[0] == target_run_id),
                None,
            )
            target_queued_index = next(
                (
                    index
                    for index, item in enumerate(queued_items)
                    if item[0] == target_run_id
                ),
                None,
            )
            target_cooldown = any(
                item[0] == target_run_id for item in cooldown_items
            )
            target_scheduled = any(
                item[0] == target_run_id for item in scheduled_items
            )

            if target_running is not None:
                run_state = "running"
            elif target_cooldown:
                run_state = "cooldown"
            elif target_queued_index is not None or target_scheduled:
                run_state = "queued"
            else:
                run_state = "idle"

            active_worker_count = len(running_items)
            queue_position = (
                target_queued_index + 1
                if target_queued_index is not None
                else None
            )
            return {
                "schema_version": "ecologyrsi-dsh.execution-scheduler/1",
                "run_state": run_state,
                "worker_count": self._worker_count,
                "active_worker_count": active_worker_count,
                "available_worker_count": max(
                    0, self._worker_count - active_worker_count
                ),
                "queued_run_count": len(queued_items),
                "cooldown_run_count": len(cooldown_items),
                "queue_position": queue_position,
                "queued_ahead": target_queued_index,
                "waiting_for_worker": bool(
                    run_state == "queued"
                    and active_worker_count >= self._worker_count
                ),
            }

    def _reconcile_idle_running_run(self, run_id: str) -> bool:
        """Requeue an enabled durable run missing all scheduler ownership."""

        if not run_id or self._stop.is_set():
            return False
        with self._state_lock:
            scheduler_owned = any(
                item[0] == run_id
                for item in (
                    *self._scheduled,
                    *self._running,
                    *self._retry_timers,
                    *self._retry_not_before,
                )
            )
        if scheduler_owned:
            return False
        try:
            state = self.server.director.state(run_id)
        except (KeyError, ValueError):
            return False
        if (
            state.run.status is not RunStatus.RUNNING
            or not auto_progress_enabled(state)
        ):
            return False
        return self._schedule_work_item((run_id, _run_incarnation(state)))

    def _reconcile_idle_running_runs(self) -> int:
        """Recover orphaned automatic runs without relying on GET side effects."""

        recovered = 0
        for run_id in self.server.ledger.run_ids(include_archived=False):
            try:
                if self.server.director.run_status(run_id) is not RunStatus.RUNNING:
                    continue
            except (KeyError, ValueError):
                continue
            recovered += int(self._reconcile_idle_running_run(run_id))
        return recovered

    def schedule(self, run_id: str) -> bool:
        """Queue a run once; return ``True`` when a new work item was added."""

        run_id = str(run_id).strip()
        if not run_id or self._stop.is_set():
            return False
        try:
            state = self.server.director.state(run_id)
        except (KeyError, ValueError):
            return False
        return self._schedule_work_item((run_id, _run_incarnation(state)))

    def _schedule_work_item(self, work_item: _WorkItem) -> bool:
        """Queue one exact durable run incarnation."""

        if self._stop.is_set():
            return False
        with self._state_lock:
            if work_item in self._running:
                # A resume/control request may arrive exactly as a worker is
                # leaving a paused boundary.  Remember it so that transition
                # cannot be lost when the active generation releases its slot.
                self._reschedule_requested.add(work_item)
                return False
            if work_item in self._scheduled:
                return False
            self._scheduled.add(work_item)
            self._queue.put(work_item)
            return True

    def schedule_if_enabled(self, run_id: str) -> bool:
        """Read the durable projection before scheduling a control transition."""

        try:
            state = self.server.director.state(run_id)
        except KeyError:
            return False
        if (
            state.run.status is not RunStatus.RUNNING
            or not auto_progress_enabled(state)
        ):
            return False
        work_item = (run_id, _run_incarnation(state))
        if state.events[-1].kind == "RunResumed":
            # Explicit resume begins a new breaker epoch.  An old timer must
            # never wake a prior epoch after the operator authorizes recovery.
            with self._state_lock:
                self._clear_retry_cooldown_locked(work_item)
                self._pending_gateway_retries.pop(work_item, None)
        return self._schedule_work_item(work_item)

    def forget(self, state: Any) -> None:
        """Discard in-memory bookkeeping for one successfully purged incarnation."""

        work_item = (str(state.run.run_id), _run_incarnation(state))
        with self._state_lock:
            self._clear_retry_cooldown_locked(work_item)
            self._scheduled.discard(work_item)
            self._running.discard(work_item)
            self._reschedule_requested.discard(work_item)
            self._deferred_failures.pop(work_item, None)
            self._deferred_pauses.pop(work_item, None)
            self._pending_gateway_retries.pop(work_item, None)

    def _clear_retry_cooldown_locked(
        self,
        work_item: _WorkItem,
        *,
        clear_deadline: bool = True,
    ) -> None:
        """Remove one cooldown while ``_state_lock`` is held."""

        timer = self._retry_timers.pop(work_item, None)
        if timer is not None:
            timer.cancel()
        if clear_deadline:
            self._retry_not_before.pop(work_item, None)
            self._retry_authority.pop(work_item, None)

    def _clear_terminal_retry_cooldown(self, work_item: _WorkItem) -> None:
        """Drop retry state once the durable incarnation reaches a terminal state."""

        try:
            state = self._state_for_work_item(work_item)
        except KeyError:
            with self._state_lock:
                self._clear_retry_cooldown_locked(work_item)
                self._reschedule_requested.discard(work_item)
                self._deferred_failures.pop(work_item, None)
                self._deferred_pauses.pop(work_item, None)
                self._pending_gateway_retries.pop(work_item, None)
            return
        except Exception:  # noqa: BLE001 - diagnostics must remain best effort
            return
        if state.run.status not in _TERMINAL_RUN_STATUSES:
            return
        with self._state_lock:
            self._clear_retry_cooldown_locked(work_item)
            self._reschedule_requested.discard(work_item)
            self._deferred_failures.pop(work_item, None)
            self._deferred_pauses.pop(work_item, None)
            self._pending_gateway_retries.pop(work_item, None)

    def _reconcile_retry_cooldowns(self) -> None:
        """Repair stale cooldown bookkeeping without blocking scheduler locks on I/O."""

        with self._state_lock:
            items = tuple(set(self._retry_timers) | set(self._retry_not_before))

        now = time.monotonic()
        for work_item in items:
            with self._state_lock:
                deadline = self._retry_not_before.get(work_item)
                timer = self._retry_timers.get(work_item)
                scheduled = work_item in self._scheduled
                running = work_item in self._running

            try:
                state = self._state_for_work_item(work_item)
            except KeyError:
                state = None
            except Exception:  # noqa: BLE001 - preserve cooldown on transient reads
                continue

            terminal = (
                state is None or state.run.status in _TERMINAL_RUN_STATUSES
            )
            expired = deadline is not None and deadline <= now
            # ``is_alive`` is also false before ``start``.  ``ident`` proves the
            # timer was started and has since exited or was cancelled.
            dead_timer = (
                timer is not None
                and timer.ident is not None
                and not timer.is_alive()
            )
            enabled_running = bool(
                state is not None
                and state.run.status is RunStatus.RUNNING
                and auto_progress_enabled(state)
            )

            should_schedule = False
            with self._state_lock:
                # Do not erase a newer cooldown installed while the durable
                # projection was read outside the scheduler lock.
                if (
                    self._retry_not_before.get(work_item) != deadline
                    or self._retry_timers.get(work_item) is not timer
                ):
                    continue
                if terminal:
                    self._clear_retry_cooldown_locked(work_item)
                    if work_item not in self._running:
                        self._scheduled.discard(work_item)
                    self._reschedule_requested.discard(work_item)
                    self._deferred_failures.pop(work_item, None)
                    self._deferred_pauses.pop(work_item, None)
                elif expired:
                    self._clear_retry_cooldown_locked(work_item)
                    should_schedule = enabled_running and not (scheduled or running)
                elif dead_timer:
                    # Preserve a future provider deadline, especially while a
                    # run is paused. A running run is queued so the worker can
                    # install a replacement timer for the remaining interval.
                    self._clear_retry_cooldown_locked(
                        work_item,
                        clear_deadline=False,
                    )
                    should_schedule = enabled_running and not (scheduled or running)
                elif (
                    timer is None
                    and deadline is not None
                    and enabled_running
                    and not (scheduled or running)
                ):
                    # This covers a process-local wakeup that disappeared while
                    # its durable run is still eligible for auto progression.
                    should_schedule = True

            if should_schedule:
                self._schedule_work_item(work_item)

    def recover_running(self) -> int:
        """Queue enabled running runs found in the ledger after service start."""

        recovered = 0
        # Archived runs are intentionally outside the active work queue.  The
        # public API only archives terminal runs, but filtering here also keeps
        # recovery safe for migrated or administratively repaired ledgers.
        for run_id in self.server.ledger.run_ids(include_archived=False):
            try:
                if self.server.director.run_status(run_id) is not RunStatus.RUNNING:
                    continue
                state = self.server.director.state(run_id)
            except (KeyError, ValueError):
                continue
            if state.run.status is RunStatus.RUNNING and auto_progress_enabled(state):
                # A provider cooldown is an operational fact, not merely an
                # in-memory timer.  Restore it before queueing so a service
                # restart does not immediately replay a request whose
                # Retry-After window is still open.
                self._restore_retry_deadline(
                    (run_id, _run_incarnation(state)), state
                )
                recovered += int(
                    self._schedule_work_item((run_id, _run_incarnation(state)))
                )
        return recovered

    def recover_native_quiescence(self) -> int:
        """Restore Host-owned native pause/cancel boundaries after a restart."""

        actions = {
            RunStatus.PAUSED: "pause",
            RunStatus.FAILED: "cancel",
            RunStatus.CANCELLED: "cancel",
        }
        recovered = 0
        # Include archived terminal runs: archival removes a run from normal
        # work queues but cannot authorize a still-live DSH process to continue.
        for run_id in self.server.ledger.run_ids(include_archived=True):
            try:
                status = self.server.director.run_status(run_id)
                action = actions.get(status)
                if action is None:
                    continue
                state = self.server.director.state(run_id)
                if (
                    state.run.status is not status
                    or state.task_manifest.metadata.get("execution_protocol")
                    != DSH_NATIVE_EXECUTION_PROTOCOL
                ):
                    continue
                quiescence = self._close_native_admission(state, action=action)
            except (KeyError, ValueError):
                continue
            if quiescence is None:
                continue
            self._start_native_quiescence(
                action,
                quiescence,
                reconcile_runtime_status=True,
            )
            recovered += 1
        return recovered

    def close(self, *, timeout: float = 30.0) -> bool:
        """Stop every worker before the owning server closes its ledger."""

        if self._stop.is_set():
            return not any(worker.is_alive() for worker in self._threads)
        self._stop.set()
        for _ in self._threads:
            self._queue.put(None)
        current = threading.current_thread()
        deadline = time.monotonic() + max(0.0, float(timeout))
        for worker in self._threads:
            if worker is current:
                continue
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._state_lock:
            for timer in self._retry_timers.values():
                timer.cancel()
            self._retry_timers.clear()
            self._scheduled.clear()
            self._running.clear()
            self._reschedule_requested.clear()
            self._deferred_failures.clear()
            self._deferred_pauses.clear()
            self._retry_not_before.clear()
            self._retry_authority.clear()
            self._pending_gateway_retries.clear()
        return not any(worker.is_alive() for worker in self._threads)

    def _schedule_retry_wakeup(self, work_item: _WorkItem, delay: float) -> bool:
        """Wake one delayed run without occupying a progression worker."""

        delay = max(
            0.0,
            min(float(delay), _GATEWAY_RETRY_DEADLINE_MAX_SECONDS),
        )
        with self._state_lock:
            authority = self._retry_authority.get(work_item)

        def wake() -> None:
            with self._state_lock:
                if self._retry_authority.get(work_item) != authority:
                    return
                self._retry_timers.pop(work_item, None)
                if self._stop.is_set():
                    return
                self._retry_not_before.pop(work_item, None)
                self._retry_authority.pop(work_item, None)
            try:
                state = self.server.director.state(work_item[0])
                if (
                    _run_incarnation(state) != work_item[1]
                    or state.run.status is not RunStatus.RUNNING
                    or not auto_progress_enabled(state)
                ):
                    return
                if authority is not None:
                    event = next(
                        (
                            item
                            for item in state.events
                            if item.seq == authority[0]
                            and int(item.payload.get("breaker_epoch", -1))
                            == authority[1]
                        ),
                        None,
                    )
                    if event is None or not self._retry_event_is_authoritative(
                        state,
                        event,
                    ):
                        return
            except (KeyError, ValueError):
                return
            self._schedule_work_item(work_item)

        with self._state_lock:
            if self._stop.is_set():
                return False
            existing = self._retry_timers.get(work_item)
            if existing is not None and existing.is_alive():
                return True
            timer = threading.Timer(delay, wake)
            timer.daemon = True
            self._retry_timers[work_item] = timer
        try:
            timer.start()
        except Exception:
            with self._state_lock:
                if self._retry_timers.get(work_item) is timer:
                    self._retry_timers.pop(work_item, None)
            return False
        return True

    def _worker(self) -> None:
        maintenance_worker = threading.current_thread() is self._thread
        next_cooldown_maintenance = (
            time.monotonic() + _SCHEDULER_COOLDOWN_MAINTENANCE_SECONDS
        )
        next_orphan_maintenance = (
            time.monotonic() + _SCHEDULER_ORPHAN_MAINTENANCE_SECONDS
        )
        while not self._stop.is_set():
            try:
                work_item = self._queue.get(timeout=0.25)
            except Empty:
                if maintenance_worker:
                    now = time.monotonic()
                    if now >= next_cooldown_maintenance:
                        next_cooldown_maintenance = (
                            now + _SCHEDULER_COOLDOWN_MAINTENANCE_SECONDS
                        )
                        try:
                            self._reconcile_retry_cooldowns()
                        except Exception:  # noqa: BLE001 - maintenance is best effort
                            # Maintenance is a repair path, never a reason to
                            # terminate the only worker that can make progress.
                            self._maintenance_failure_count += 1
                    if now >= next_orphan_maintenance:
                        next_orphan_maintenance = (
                            now + _SCHEDULER_ORPHAN_MAINTENANCE_SECONDS
                        )
                        try:
                            self._reconcile_idle_running_runs()
                        except Exception:  # noqa: BLE001 - maintenance is best effort
                            # A transient ledger read is retried at the next
                            # bounded tick; normal FIFO work remains available.
                            self._maintenance_failure_count += 1
                continue
            continue_running = False
            reschedule_requested = False
            dispatch = False
            retry_wait = 0.0
            retry_wakeup_scheduled = False
            retry_wakeup_failed = False
            try:
                if work_item is None:
                    return
                with self._state_lock:
                    # A successful purge withdraws the exact incarnation from
                    # ``_scheduled``. Queue.Queue has no safe arbitrary removal,
                    # so discard that stale physical item when it reaches the
                    # head instead of recreating a per-run lock for it.
                    if work_item not in self._scheduled:
                        continue
                    retry_deadline = self._retry_not_before.get(work_item)
                    if retry_deadline is not None:
                        retry_wait = retry_deadline - time.monotonic()
                        if retry_wait <= 0.0:
                            # A queue item can wait behind other work until its
                            # cooldown has already elapsed.  It will dispatch
                            # immediately, so the expired deadline must not
                            # survive as a false cooldown diagnostic.
                            self._retry_not_before.pop(work_item, None)
                            retry_wait = 0.0
                # Do not hold either ledger or state locks while waiting for a
                # busy gateway.  A timer-backed wakeup lets this worker serve
                # other runs while the provider recovers.
                if retry_wait:
                    with self._state_lock:
                        # This queue item is no longer active; the wakeup will
                        # create a fresh exact-incarnation item later.
                        self._scheduled.discard(work_item)
                        retry_wakeup_scheduled = self._schedule_retry_wakeup(
                            work_item, retry_wait
                        )
                        if not retry_wakeup_scheduled and not self._stop.is_set():
                            retry_wakeup_failed = True
                    if retry_wakeup_scheduled or self._stop.is_set():
                        continue
                    if retry_wakeup_failed:
                        # Resource pressure must not authorize an early remote
                        # call. Retry installing the wakeup from the FIFO while
                        # retaining the durable deadline and authority.
                        if self._stop.wait(
                            min(
                                retry_wait,
                                _RETRY_TIMER_START_RETRY_SECONDS,
                            )
                        ):
                            continue
                        continue_running = True
                else:
                    with self._state_lock:
                        # ``_scheduled`` spans both queueing and execution.  That
                        # prevents another worker from dequeuing this run while
                        # its current generation still holds the shared lease.
                        self._running.add(work_item)
                        dispatch = True
                if not retry_wakeup_failed:
                    continue_running = self._run_one_generation(
                        work_item[0], expected_incarnation=work_item[1]
                    )
                    if not continue_running:
                        self._clear_terminal_retry_cooldown(work_item)
            except Exception as exc:  # noqa: BLE001 - isolate one queue item
                if work_item is not None:
                    try:
                        failure_state = self._state_for_work_item(work_item)
                        deferred_error = _retry_later_error(exc, stage="worker")
                        if deferred_error is not None:
                            continue_running = self._defer_retry(
                                work_item,
                                self._gateway_retry_delay(
                                    work_item,
                                    1,
                                    deferred_error,
                                    stage="worker",
                                ),
                                deferred_error,
                                stage="worker",
                                attempt_anchor_seq=self._attempt_authority_seq(
                                    failure_state
                                ),
                            )
                        elif _progress_failure_irrecoverable(exc, stage="worker"):
                            code, context = _failure_diagnostics(
                                failure_state, exc, stage="worker"
                            )
                            self._defer_failure(
                                work_item,
                                _DeferredFailure(
                                    "自动推进工作器异常："
                                    f"{public_exception_summary(exc)}",
                                    code,
                                    context,
                                ),
                            )
                            # The next FIFO turn persists only RunFailed.
                            continue_running = True
                        else:
                            continue_running = self._persist_or_defer_pause(
                                work_item,
                                _host_fault_pause(
                                    failure_state,
                                    exc,
                                    stage="worker",
                                ),
                            )
                    except (KeyError, ValueError):
                        continue_running = False
                        continue
            finally:
                if work_item is not None:
                    with self._state_lock:
                        if dispatch:
                            self._running.discard(work_item)
                        self._scheduled.discard(work_item)
                        reschedule_requested = (
                            work_item in self._reschedule_requested
                        )
                        self._reschedule_requested.discard(work_item)
                self._queue.task_done()
            if (
                (continue_running or reschedule_requested)
                and not self._stop.is_set()
            ):
                # Round-robin at generation granularity.  If a control request
                # already queued the same run while this item was executing,
                # the remembered reschedule request restores it after the
                # active lease is released.
                self._schedule_work_item(work_item)

    def _run_one_generation(
        self,
        run_id: str,
        *,
        expected_incarnation: int | None = None,
    ) -> bool:
        """Advance one generation and report whether the run should be requeued."""

        if expected_incarnation is None:
            try:
                expected_incarnation = _run_incarnation(
                    self.server.director.state(run_id)
                )
            except (KeyError, ValueError):
                return False
        work_item = (run_id, expected_incarnation)
        generation_lease = self.server.acquire_generation_lease(run_id)
        if generation_lease is None:
            return False
        retire_idle_lease = False
        try:
            try:
                self._state_for_work_item(work_item)
            except KeyError:
                self._clear_deferred_failure(work_item)
                self._clear_deferred_pause(work_item)
                retire_idle_lease = True
                return False
            return self._run_one_generation_locked(work_item)
        finally:
            generation_lease.release()
            if retire_idle_lease:
                self.server.retire_generation_lease_if_idle(generation_lease)

    def _run_one_generation_locked(self, work_item: _WorkItem) -> bool:
        """Execute one generation while holding only its per-run lock."""

        run_id, _expected_incarnation = work_item
        with self._state_lock:
            pending_gateway_retry = self._pending_gateway_retries.get(work_item)
        if pending_gateway_retry is not None:
            try:
                keep_running = self._persist_gateway_retry_report(
                    work_item,
                    pending_gateway_retry,
                )
            except Exception:
                if self._stop.wait(_FAILURE_PERSISTENCE_RETRY_SECONDS):
                    return False
                return True
            with self._state_lock:
                self._pending_gateway_retries.pop(work_item, None)
            return keep_running
        deferred_failure = self._deferred_failure(work_item)
        deferred_pause = self._deferred_pause(work_item)
        if deferred_pause is not None:
            return self._persist_or_defer_pause(work_item, deferred_pause)
        if deferred_failure is not None:
            with self.server.mutation_lock:
                retry_terminal_write = self._persist_run_failure(
                    work_item, deferred_failure
                )
            if not retry_terminal_write:
                return False
            if self._stop.wait(_FAILURE_PERSISTENCE_RETRY_SECONDS):
                return False
            return True

        failures = 0
        while not self._stop.is_set():
            retry_delay = 0.0
            preflight_failed = False
            preflight_pause: _DeferredPause | None = None
            terminal_failure = False
            retry_terminal_write = False
            # Only the short state/binding boundary needs the HTTP mutation
            # lock. Remote model waits and scientific execution happen outside
            # it so pause/cancel and unrelated runs remain controllable.
            with self.server.mutation_lock:
                try:
                    state = self._state_for_work_item(work_item)
                except KeyError:
                    return False
                if state.run.status is not RunStatus.RUNNING:
                    return False
                if not auto_progress_enabled(state):
                    return False
                attempt_anchor_seq = self._attempt_authority_seq(state)
                try:
                    state = complete_if_budget_exhausted(self.server, run_id, state)
                    if state.run.status is not RunStatus.RUNNING:
                        return False
                    # Explicit /advance requests validate the frozen dataset,
                    # evaluator, predictor, and model bindings before running.
                    # The background path must enforce the same boundary on
                    # every generation, especially after a service restart.
                    self.server.validate_frozen_runtime_bindings(
                        state.task_manifest,
                        run_id=state.run.run_id,
                    )
                    generation = int(state.run.generation)
                except Exception as exc:  # noqa: BLE001 - isolate one run preflight
                    preflight_failed = True
                    failures += 1
                    retryable = _progress_failure_retryable(exc)
                    gateway_error = gateway_error_in_chain(exc)
                    deferred_error = _retry_later_error(exc)
                    if deferred_error is not None:
                        return self._defer_retry(
                            work_item,
                            self._gateway_retry_delay(
                                work_item,
                                failures,
                                deferred_error,
                                stage="preflight",
                            ),
                            deferred_error,
                            stage="preflight",
                            attempt_anchor_seq=attempt_anchor_seq,
                        )
                    if retryable and failures < self._retry_limit:
                        # The generation executor is event-idempotent and can
                        # reconcile a partially written batch on the next
                        # attempt.  Retry transient gateway/ledger boundaries a
                        # small number of times without busy-spinning.
                        retry_delay = min(2.0, 0.25 * (2 ** (failures - 1)))
                    if (
                        deferred_error is None
                        and not _progress_failure_irrecoverable(
                            exc,
                            stage="preflight",
                        )
                    ):
                        preflight_pause = _host_fault_pause(
                            state,
                            exc,
                            stage="preflight",
                        )
                    elif not retryable or failures >= self._retry_limit:
                        terminal_failure = True
                        disposition = (
                            "网关请求已完成内部重试"
                            if gateway_error is not None
                            else "不可重试"
                            if not retryable
                            else f"已尝试 {failures} 次"
                        )
                        failure_reason = (
                            f"自动推进失败（{disposition}）："
                            f"{public_exception_summary(exc)}"
                        )
                        failure_code, failure_context = _failure_diagnostics(
                            state, exc, stage="preflight"
                        )
                        failure = _DeferredFailure(
                            failure_reason, failure_code, failure_context
                        )
                        try:
                            retry_terminal_write = self._persist_run_failure(
                                work_item,
                                failure,
                            )
                        except Exception:  # noqa: BLE001
                            # ``_persist_run_failure`` owns ordinary ledger and
                            # terminal-race handling.  If its in-memory recovery
                            # bookkeeping itself fails, retain the original
                            # bounded failure and keep the run queued.
                            self._defer_failure(work_item, failure)
                            retry_terminal_write = True

            if preflight_pause is not None:
                return self._persist_or_defer_pause(work_item, preflight_pause)

            if terminal_failure:
                if retry_terminal_write:
                    if self._stop.wait(_FAILURE_PERSISTENCE_RETRY_SECONDS):
                        return False
                    return True
                return False

            if preflight_failed:
                if self._stop.wait(retry_delay):
                    return False
                continue

            try:
                # Adaptive epochs deliberately yield after every lane-bounded
                # scheduler turn. This keeps pause/cancel responsive while two
                # finalist batches may share the same run-level origin permits.
                # The explicit /advance path still uses the synchronous
                # executor and is not changed here.
                adaptive_protocol = (
                    state.task_manifest.metadata.get("optimization_protocol")
                    in ADAPTIVE_PROTOCOLS
                )
                if adaptive_protocol:
                    from ..application.work_units import execute_next_adaptive_work_unit

                    before_work_unit_seq = int(state.events[-1].seq)
                    progressed = execute_next_adaptive_work_unit(self.server, run_id)
                    latest = self._state_for_work_item(work_item)
                    if _run_incarnation(latest) != work_item[1]:
                        raise _RunIncarnationChanged(run_id)
                    if latest.run.status is not RunStatus.RUNNING:
                        return False
                    if not progressed:
                        raise RuntimeError(
                            "adaptive work unit returned without durable progress"
                        )
                    if int(latest.events[-1].seq) <= before_work_unit_seq:
                        raise RuntimeError(
                            "adaptive work unit reported progress without "
                            "advancing the durable event sequence"
                        )
                    return True

                result = execute_generation(self.server, run_id)
                latest = (
                    result
                    if getattr(getattr(result, "run", None), "status", None)
                    is not None
                    else self._state_for_work_item(work_item)
                )
                if _run_incarnation(latest) != work_item[1]:
                    raise _RunIncarnationChanged(run_id)
                if latest.run.status is not RunStatus.RUNNING:
                    return False
                if int(latest.run.generation) <= generation:
                    raise RuntimeError(
                        "generation execution returned without advancing"
                    )
                return True
            except Exception as exc:  # noqa: BLE001 - isolate one generation
                failures += 1
                try:
                    recovery_state = self._state_for_work_item(work_item)
                except (KeyError, ValueError):
                    recovery_state = None
                retry_attempt_anchor_seq = (
                    self._attempt_authority_seq(recovery_state)
                    if recovery_state is not None
                    else attempt_anchor_seq
                )
                recovery_stage = _latest_failed_stage(recovery_state)
                research_contract_error = find_exception(
                    exc,
                    ResearchResponseContractError,
                )
                if (
                    recovery_stage == "research"
                    and isinstance(research_contract_error, ResearchResponseContractError)
                ):
                    validation_detail = str(
                        getattr(
                            research_contract_error,
                            "validation_detail",
                            None,
                        )
                        or "宿主语义合同校验未通过"
                    )[:500]
                    return self._persist_or_defer_pause(
                        work_item,
                        _DeferredPause(
                            reason=(
                                "研究响应已用尽本轮语义修复预算："
                                f"{validation_detail}"
                            )[:500],
                            code="research_contract_fallback_unavailable",
                        ),
                    )
                retryable = _progress_failure_retryable(
                    exc,
                    stage=recovery_stage,
                )
                # ModelGateway has already retried the individual request.
                # That budget is intentionally small, because a provider can
                # remain queued or rate-limited for much longer.  Preserve the
                # running generation and put it back on the FIFO after a
                # bounded cooldown; never mark the whole run failed merely
                # because this one call was temporarily unavailable.
                gateway_error = gateway_error_in_chain(exc)
                deferred_error = _retry_later_error(
                    exc,
                    stage=recovery_stage,
                )
                if deferred_error is not None:
                    retry_delay = self._gateway_retry_delay(
                        work_item,
                        failures,
                        deferred_error,
                        stage=recovery_stage,
                    )
                    return self._defer_retry(
                        work_item,
                        retry_delay,
                        deferred_error,
                        stage=recovery_stage,
                        attempt_anchor_seq=retry_attempt_anchor_seq,
                    )
                if not _progress_failure_irrecoverable(
                    exc,
                    stage=recovery_stage,
                ):
                    return self._persist_or_defer_pause(
                        work_item,
                        _host_fault_pause(
                            recovery_state or state,
                            exc,
                            stage=recovery_stage,
                        ),
                    )
                terminal_failure = not retryable or failures >= self._retry_limit
                retry_terminal_write = False
                failure_reason: str | None = None
                if retryable and failures < self._retry_limit:
                    retry_delay = min(300.0, 15.0 * (2 ** (failures - 1)))
                if terminal_failure:
                    disposition = (
                        "网关请求已完成内部重试，进化层不再重放整代"
                        if gateway_error is not None
                        else "不可重试"
                        if not retryable
                        else f"已尝试 {failures} 次"
                    )
                    failure_reason = (
                        f"自动推进失败（{disposition}）："
                        f"{public_exception_summary(exc)}"
                    )
                    diagnostics_state = recovery_state or state
                    failure_code, failure_context = _failure_diagnostics(
                        diagnostics_state,
                        exc,
                        stage=recovery_stage,
                    )
                    failure = _DeferredFailure(
                        failure_reason, failure_code, failure_context
                    )
                with self.server.mutation_lock:
                    try:
                        latest = self._state_for_work_item(work_item)
                        if latest.run.status is not RunStatus.RUNNING:
                            return False
                        if terminal_failure:
                            assert failure_reason is not None
                            retry_terminal_write = self._persist_run_failure(
                                work_item,
                                failure,
                            )
                    except KeyError:
                        self._clear_deferred_failure(work_item)
                        return False
                    except Exception:  # noqa: BLE001
                        if terminal_failure:
                            assert failure_reason is not None
                            self._defer_failure(work_item, failure)
                            retry_terminal_write = True
                if terminal_failure:
                    if retry_terminal_write:
                        if self._stop.wait(_FAILURE_PERSISTENCE_RETRY_SECONDS):
                            return False
                        return True
                    return False

            # Wait outside the mutation lock so pause/cancel requests are not
            # blocked by retry backoff.
            if self._stop.wait(retry_delay):
                return False
        return False

    def _state_for_work_item(self, work_item: _WorkItem) -> Any:
        run_id, expected_incarnation = work_item
        state = self.server.director.state(run_id)
        if _run_incarnation(state) != expected_incarnation:
            raise _RunIncarnationChanged(run_id)
        return state

    def _deferred_failure(
        self, work_item_or_run_id: _WorkItem | str
    ) -> _DeferredFailure | None:
        work_item: _WorkItem | None
        if isinstance(work_item_or_run_id, tuple):
            work_item = work_item_or_run_id
        else:
            try:
                state = self.server.director.state(work_item_or_run_id)
            except (KeyError, ValueError):
                return None
            work_item = (work_item_or_run_id, _run_incarnation(state))
        with self._state_lock:
            return self._deferred_failures.get(work_item)

    def _defer_failure(
        self, work_item: _WorkItem, failure: _DeferredFailure
    ) -> None:
        with self._state_lock:
            self._deferred_failures[work_item] = failure

    def _deferred_pause(
        self, work_item_or_run_id: _WorkItem | str
    ) -> _DeferredPause | None:
        work_item: _WorkItem | None
        if isinstance(work_item_or_run_id, tuple):
            work_item = work_item_or_run_id
        else:
            try:
                state = self.server.director.state(work_item_or_run_id)
            except (KeyError, ValueError):
                return None
            work_item = (work_item_or_run_id, _run_incarnation(state))
        with self._state_lock:
            return self._deferred_pauses.get(work_item)

    def _defer_pause(self, work_item: _WorkItem, pause: _DeferredPause) -> None:
        with self._state_lock:
            self._deferred_pauses[work_item] = pause

    def _clear_deferred_pause(self, work_item: _WorkItem) -> None:
        with self._state_lock:
            self._deferred_pauses.pop(work_item, None)

    def _persist_or_defer_pause(
        self,
        work_item: _WorkItem,
        pause: _DeferredPause,
    ) -> bool:
        """Persist a checkpoint-preserving pause before any work-unit replay."""

        try:
            with self.server.mutation_lock:
                retry_pause_write = self._persist_run_pause(work_item, pause)
        except Exception:  # noqa: BLE001 - retain the original transition intent
            self._defer_pause(work_item, pause)
            retry_pause_write = True
        if not retry_pause_write:
            return False
        if self._stop.wait(_FAILURE_PERSISTENCE_RETRY_SECONDS):
            return False
        return True

    def _defer_retry(
        self,
        work_item: _WorkItem,
        delay_seconds: float,
        exc: BaseException | None = None,
        *,
        stage: str | None = None,
        attempt_anchor_seq: int,
    ) -> bool:
        """Commit a retry decision before installing process-local scheduling."""

        retry_class, last_error_code = _retry_class_and_error_code(exc, stage=stage)
        try:
            state = self._state_for_work_item(work_item)
        except (KeyError, ValueError):
            return False
        normalized_stage = str(stage or "generation").strip().lower()
        report = _PendingGatewayRetry(
            run_incarnation=work_item[1],
            generation=int(state.run.generation),
            stage=normalized_stage,
            retry_class=retry_class,
            failure_id=digest(
                {
                    "schema_version": "ecologyrsi-dsh.gateway-failure-id/1",
                    "run_id": work_item[0],
                    "run_incarnation": work_item[1],
                    "generation": int(state.run.generation),
                    "stage": normalized_stage,
                    "retry_class": retry_class,
                    "attempt_anchor_seq": int(attempt_anchor_seq),
                }
            ),
            attempt_anchor_seq=int(attempt_anchor_seq),
            delay_seconds=max(
                0.0,
                min(float(delay_seconds), _GATEWAY_RETRY_DEADLINE_MAX_SECONDS),
            ),
            last_error_code=last_error_code,
        )
        try:
            keep_running = self._persist_gateway_retry_report(work_item, report)
        except Exception:
            # The report, not the remote generation call, is now pending.  A
            # later worker turn retries this exact immutable failure identity.
            with self._state_lock:
                self._pending_gateway_retries[work_item] = report
                self._clear_retry_cooldown_locked(work_item)
            return True
        with self._state_lock:
            self._pending_gateway_retries.pop(work_item, None)
        return keep_running

    def _persist_gateway_retry_report(
        self,
        work_item: _WorkItem,
        report: _PendingGatewayRetry,
    ) -> bool:
        decision = self.server.director.schedule_gateway_retry_or_pause(
            work_item[0],
            run_incarnation=report.run_incarnation,
            generation=report.generation,
            stage=report.stage,
            retry_class=report.retry_class,
            failure_id=report.failure_id,
            attempt_anchor_seq=report.attempt_anchor_seq,
            delay_seconds=report.delay_seconds,
            last_error_code=report.last_error_code,
        )
        event = decision.event
        if decision.outcome == "paused":
            # The retry circuit writes RunPaused atomically inside the
            # director. Complete the same native-runtime fence used by every
            # other automatic pause so Host and DSH cannot diverge.
            try:
                paused_state = self._state_for_work_item(work_item)
            except (KeyError, ValueError):
                paused_state = None
            if paused_state is not None:
                native_quiescence = self._close_native_admission(
                    paused_state,
                    action="pause",
                )
                self._start_native_quiescence("pause", native_quiescence)
            with self._state_lock:
                self._clear_retry_cooldown_locked(work_item)
            return False
        if decision.outcome != "scheduled" or event is None:
            with self._state_lock:
                self._clear_retry_cooldown_locked(work_item)
            return False
        try:
            state = self._state_for_work_item(work_item)
        except (KeyError, ValueError):
            return False
        if not self._retry_event_is_authoritative(state, event):
            with self._state_lock:
                self._clear_retry_cooldown_locked(work_item)
            return False
        retry_at = _parse_retry_at(event.payload.get("retry_at"))
        if retry_at is None:
            raise ValueError("durable gateway retry has an invalid retry_at")
        delay = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        delay = min(delay, _GATEWAY_RETRY_DEADLINE_MAX_SECONDS)
        with self._state_lock:
            self._retry_not_before[work_item] = time.monotonic() + delay
            self._retry_authority[work_item] = (
                int(event.seq),
                int(event.payload["breaker_epoch"]),
            )
        return True

    @staticmethod
    def _retry_event_is_authoritative(state: Any, event: Any) -> bool:
        if (
            getattr(getattr(state, "run", None), "status", None)
            is not RunStatus.RUNNING
            or int(getattr(state.run, "generation", -1))
            != int(event.payload.get("generation", -2))
            or _run_incarnation(state)
            != int(event.payload.get("run_incarnation", -1))
            or not any(item.seq == event.seq for item in state.events)
        ):
            return False
        for later in state.events:
            if later.seq <= event.seq:
                continue
            if later.kind in {
                "RunPaused",
                "RunResumed",
                "RunCancelled",
                "RunFailed",
                "RunCompleted",
                "GenerationAdvanced",
            }:
                return False
            if (
                later.kind == "GatewayRetryScheduled"
                and later.payload.get("schema_version")
                == "ecologyrsi-dsh.gateway-retry-scheduled/2"
                and later.payload.get("retry_class")
                == event.payload.get("retry_class")
                and later.payload.get("stage") == event.payload.get("stage")
            ):
                return False
            if (
                later.kind == "EvolutionStageRecorded"
                and int(later.payload.get("generation", -1))
                == int(event.payload.get("generation", -2))
                and later.payload.get("stage") == event.payload.get("stage")
                and later.payload.get("status") in {"started", "completed"}
            ):
                return False
        return True

    @staticmethod
    def _attempt_authority_seq(state: Any) -> int:
        """Recover the durable authority for one restarted logical attempt.

        A stage ``started`` or ``failed`` observation after a retry heartbeat
        proves the scheduled attempt began; it does not mint a second failure
        slot.  Reusing the heartbeat sequence keeps a post-crash failure on
        the same count chain.  Success and lifecycle transitions consume that
        authority and fall back to the latest event.
        """

        events = tuple(getattr(state, "events", ()) or ())
        if not events:
            raise ValueError("run has no durable attempt authority")
        generation = int(getattr(getattr(state, "run", None), "generation", -1))
        retry = next(
            (
                event
                for event in reversed(events)
                if event.kind == "GatewayRetryScheduled"
                and event.payload.get("schema_version")
                == "ecologyrsi-dsh.gateway-retry-scheduled/2"
                and int(event.payload.get("generation", -2)) == generation
            ),
            None,
        )
        if retry is None:
            return int(events[-1].seq)
        for later in events:
            if later.seq <= retry.seq:
                continue
            if later.kind in {
                "RunPaused",
                "RunResumed",
                "RunStarted",
                "RunCancelled",
                "RunFailed",
                "RunCompleted",
                "GenerationAdvanced",
            }:
                return int(events[-1].seq)
            if (
                later.kind == "EvolutionStageRecorded"
                and int(later.payload.get("generation", -1)) == generation
                and later.payload.get("stage") == retry.payload.get("stage")
                and later.payload.get("status") == "completed"
            ):
                return int(events[-1].seq)
            if (
                retry.payload.get("retry_class") == "dsh_native_runtime"
                and later.kind == "DshStructuredResultAccepted"
            ):
                # Candidate workers drain every admitted sibling before their
                # failure reaches this scheduler. A later accepted DSH child
                # therefore belongs to the settled retry attempt, not to a
                # future call. Anchor the failure report after that sibling so
                # its success resets the old breaker epoch without swallowing
                # the still-failed parallel work.
                return int(events[-1].seq)
        return int(retry.seq)

    def _restore_retry_deadline(self, work_item: _WorkItem, state: Any) -> None:
        """Restore a future retry deadline from the append-only heartbeat.

        The event is advisory and malformed/legacy timestamps are ignored;
        normal run recovery remains available for those ledgers.  Only the
        latest heartbeat for the current generation is considered, so an old
        generation can never delay a newly advanced one.
        """

        latest = next(
            (
                event
                for event in reversed(getattr(state, "events", ()))
                if event.kind == "GatewayRetryScheduled"
                and int(event.payload.get("generation", -1))
                == int(state.run.generation)
            ),
            None,
        )
        if latest is None:
            return
        is_v2 = (
            latest.payload.get("schema_version")
            == "ecologyrsi-dsh.gateway-retry-scheduled/2"
        )
        if is_v2 and not self._retry_event_is_authoritative(state, latest):
            return
        # A later stage event proves that the worker already resumed the
        # generation after this heartbeat (the process may have crashed before
        # the next durable boundary).  In that case restoring the old cooldown
        # would add an unnecessary multi-minute pause after restart.
        if not is_v2 and any(
            event.seq > latest.seq
            and (
                event.kind
                in {
                    "RunPaused",
                    "RunResumed",
                    "RunCancelled",
                    "RunFailed",
                    "RunCompleted",
                    "GenerationAdvanced",
                }
                or (
                    event.kind == "EvolutionStageRecorded"
                    and int(event.payload.get("generation", -1))
                    == int(state.run.generation)
                )
            )
            for event in getattr(state, "events", ())
        ):
            return
        retry_at = _parse_retry_at(latest.payload.get("retry_at"))
        if retry_at is None:
            return
        delay = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        with self._state_lock:
            self._retry_not_before[work_item] = time.monotonic() + min(
                delay, _GATEWAY_RETRY_DEADLINE_MAX_SECONDS
            )
            if is_v2:
                self._retry_authority[work_item] = (
                    int(latest.seq),
                    int(latest.payload["breaker_epoch"]),
                )

    def _retry_attempt(
        self,
        work_item: _WorkItem,
        *,
        retry_class: str | None = None,
        stage: str | None = None,
    ) -> int:
        """Return a bounded, scoped attempt for backoff only."""

        # The manager stores deadlines rather than counters.  Counting recent
        # durable heartbeats gives a restart-safe approximation and is enough
        # for the UI; exact transport attempts remain gateway diagnostics.
        try:
            state = self._state_for_work_item(work_item)
            current_generation = int(state.run.generation)
            scoped_v2 = [
                event
                for event in state.events
                if event.kind == "GatewayRetryScheduled"
                and event.payload.get("schema_version")
                == "ecologyrsi-dsh.gateway-retry-scheduled/2"
                and int(event.payload.get("generation", -1)) == current_generation
                and (retry_class is None or event.payload.get("retry_class") == retry_class)
                and (stage is None or event.payload.get("stage") == stage)
            ]
            if scoped_v2:
                return min(
                    12,
                    max(1, int(scoped_v2[-1].payload["consecutive_failures"]) + 1),
                )
            legacy_count = sum(
                1
                for event in state.events
                if event.kind == "GatewayRetryScheduled"
                and event.payload.get("schema_version")
                != "ecologyrsi-dsh.gateway-retry-scheduled/2"
                and int(event.payload.get("generation", -1)) == current_generation
            )
            return min(12, max(1, legacy_count + 1))
        except Exception:
            return 1

    def _gateway_retry_delay(
        self,
        work_item: _WorkItem,
        failures: int,
        exc: BaseException | None = None,
        *,
        stage: str | None = None,
    ) -> float:
        """Combine durable backoff with a bounded provider retry deadline."""

        retry_class, _error_code = _retry_class_and_error_code(exc, stage=stage)
        durable_attempt = max(
            1,
            self._retry_attempt(
                work_item,
                retry_class=retry_class,
                stage=str(stage or "generation").strip().lower(),
            ),
        )
        local_attempt = max(1, int(failures))
        exponent = max(0, durable_attempt + local_attempt - 2)
        retry_base = max(0.0, float(_GATEWAY_RETRY_BASE_SECONDS))
        retry_max = max(0.0, float(_GATEWAY_RETRY_MAX_SECONDS))
        if retry_base == 0.0 or retry_max == 0.0:
            local_delay = 0.0
        elif retry_base >= retry_max:
            local_delay = retry_max
        else:
            saturation_exponent = max(
                0,
                math.ceil(math.log2(retry_max) - math.log2(retry_base)),
            )
            bounded_exponent = min(exponent, saturation_exponent)
            local_delay = min(
                retry_max,
                math.ldexp(retry_base, bounded_exponent),
            )
        dsh_error = (
            dsh_native_runtime_error_in_chain(exc) if exc is not None else None
        )
        gateway_error = (
            gateway_error_in_chain(exc)
            if exc is not None and dsh_error is None
            else None
        )
        retry_after = (
            getattr(dsh_error or gateway_error, "retry_after_seconds", None)
            if dsh_error is not None or gateway_error is not None
            else None
        )
        if (
            isinstance(retry_after, (int, float))
            and not isinstance(retry_after, bool)
            and math.isfinite(float(retry_after))
            and float(retry_after) >= 0
        ):
            return min(
                _GATEWAY_RETRY_DEADLINE_MAX_SECONDS,
                max(local_delay, float(retry_after)),
            )
        return local_delay

    def _clear_deferred_failure(self, work_item: _WorkItem) -> None:
        with self._state_lock:
            self._deferred_failures.pop(work_item, None)

    def _persist_run_pause(
        self,
        work_item: _WorkItem,
        pause: _DeferredPause,
    ) -> bool:
        """Persist RunPaused, or request a retry of only that transition."""

        run_id = work_item[0]
        try:
            latest = self._state_for_work_item(work_item)
        except KeyError:
            self._clear_deferred_pause(work_item)
            return False
        except Exception:  # noqa: BLE001 - retry an unavailable ledger boundary
            self._defer_pause(work_item, pause)
            return True
        if latest.run.status is not RunStatus.RUNNING:
            self._clear_deferred_pause(work_item)
            return False
        native_quiescence = self._close_native_admission(latest, action="pause")
        try:
            self.server.director.pause_run(
                run_id,
                reason=str(pause.reason)[:500],
                code=pause.code,
            )
        except Exception:  # noqa: BLE001 - distinguish a lifecycle race by replay
            try:
                latest = self._state_for_work_item(work_item)
            except KeyError:
                self._clear_native_quiescence(native_quiescence)
                self._clear_deferred_pause(work_item)
                return False
            except Exception:  # noqa: BLE001 - ledger still unavailable
                self._clear_native_quiescence(native_quiescence)
                self._defer_pause(work_item, pause)
                return True
            if latest.run.status is not RunStatus.RUNNING:
                if latest.run.status is RunStatus.PAUSED:
                    self._start_native_quiescence("pause", native_quiescence)
                else:
                    self._clear_native_quiescence(native_quiescence)
                self._clear_deferred_pause(work_item)
                return False
            self._clear_native_quiescence(native_quiescence)
            self._defer_pause(work_item, pause)
            return True
        self._start_native_quiescence("pause", native_quiescence)
        with self._state_lock:
            self._deferred_pauses.pop(work_item, None)
            self._deferred_failures.pop(work_item, None)
            self._pending_gateway_retries.pop(work_item, None)
            self._clear_retry_cooldown_locked(work_item)
        return False

    def abort_native_evaluation(self, run_id: str) -> None:
        """Drain failed native work before candidate executors join siblings.

        The generation owner retains responsibility for persisting RunFailed.
        Reuse the control fence so concurrent user cancellation cannot race a
        second remote control request.
        """

        with self.server.mutation_lock:
            state = self.server.director.state(run_id)
            quiescence = self._close_native_admission(state, action="cancel")
        self._start_native_quiescence("cancel", quiescence)

    def _close_native_admission(
        self,
        state: Any,
        *,
        action: str,
    ) -> _NativeQuiescence | None:
        """Fence native tool writes before a terminal Host transition."""

        if (
            state.task_manifest.metadata.get("execution_protocol")
            != DSH_NATIVE_EXECUTION_PROTOCOL
        ):
            return None
        run_id = str(state.run.run_id)
        self.server.dsh_tools.close_run_admissions(run_id)
        runtime = getattr(self.server, "dsh_native_runtime", None)
        method = getattr(runtime, action, None)
        if not callable(method):
            return None
        run_revision = int(state.events[-1].seq)
        request = {
            "run_id": run_id,
            "run_state_revision": run_revision,
            "stage_attempt": 0,
            "ledger_expected_revision": int(self.server.ledger.latest_seq()),
            "idempotency_key": (
                f"auto-progress-{action}:{run_id}:{run_revision}"
            ),
        }
        marker = (id(runtime), object())
        if not self.server.claim_native_control_inflight(run_id, marker):
            # A control already owns remote quiescence for this run. Admission
            # is closed, so starting a second out-of-order pause/cancel would
            # only make a later explicit resume race the older operation.
            return None
        return _NativeQuiescence(
            request=request,
            runtime=runtime,
            marker=marker,
        )

    def _clear_native_quiescence(
        self,
        quiescence: _NativeQuiescence | None,
    ) -> None:
        if quiescence is None:
            return
        run_id = str(quiescence.request["run_id"])
        self.server.clear_native_control_inflight(run_id, quiescence.marker)

    def _native_quiescence_owned(
        self,
        quiescence: _NativeQuiescence,
    ) -> bool:
        return self.server.native_control_inflight_owned(
            str(quiescence.request["run_id"]),
            quiescence.marker,
        )

    @staticmethod
    def _native_quiescence_retry_delay(attempt: int) -> float:
        retry_base = max(0.0, float(_NATIVE_QUIESCENCE_RETRY_BASE_SECONDS))
        retry_max = max(0.0, float(_NATIVE_QUIESCENCE_RETRY_MAX_SECONDS))
        if retry_base == 0.0 or retry_max == 0.0:
            return 0.0
        exponent = min(30, max(0, int(attempt) - 1))
        return min(retry_max, math.ldexp(retry_base, exponent))

    @staticmethod
    def _native_status_disposition(action: str, payload: Any) -> str:
        """Classify a runtime status without trusting unknown states."""

        if not isinstance(payload, Mapping):
            return "blocked"
        status = payload.get("status")
        if not isinstance(status, str):
            return "blocked"
        normalized = status.strip().lower()
        if action == "pause" and normalized in {"paused", "cancelled"}:
            return "quiesced"
        if action == "cancel" and normalized == "cancelled":
            return "quiesced"
        if normalized in {"pausing", "cancelling"}:
            return "wait"
        allowed = (
            {"created", "running", "resuming"}
            if action == "pause"
            else {"created", "running", "paused", "resuming"}
        )
        return "invoke" if normalized in allowed else "blocked"

    def _finish_native_quiescence(
        self,
        action: str,
        quiescence: _NativeQuiescence,
        *,
        reconcile_runtime_status: bool = False,
    ) -> BaseException | None:
        """Synchronously drain one runtime, retaining ownership until safe."""

        method = getattr(quiescence.runtime, action, None)
        if not callable(method):
            # Admission remains closed. Keep the marker so start/resume cannot
            # overtake a runtime that has not demonstrated quiescence.
            return RuntimeError("native runtime has no quiescence method")
        status_method = getattr(quiescence.runtime, "status", None)
        attempt = 0
        check_status = reconcile_runtime_status and callable(status_method)
        while not self._stop.is_set() and self._native_quiescence_owned(
            quiescence
        ):
            if check_status:
                try:
                    status_payload = status_method(
                        str(quiescence.request["run_id"])
                    )
                except Exception as exc:  # noqa: BLE001 - classify boundary
                    dsh_error = dsh_native_runtime_error_in_chain(exc)
                    if (
                        dsh_error is not None
                        and getattr(dsh_error, "status_code", None) == 404
                    ):
                        # A missing runtime run has no live children to drain.
                        return None
                    if dsh_error is None or not dsh_native_runtime_retryable(
                        dsh_error
                    ):
                        return exc
                    disposition = "wait"
                else:
                    disposition = self._native_status_disposition(
                        action,
                        status_payload,
                    )
                if disposition == "quiesced":
                    return None
                if disposition == "blocked":
                    return RuntimeError(
                        "native runtime status did not prove quiescence"
                    )
                if disposition == "wait":
                    attempt += 1
                    if self._stop.wait(
                        self._native_quiescence_retry_delay(attempt)
                    ):
                        return RuntimeError("native quiescence was stopped")
                    continue
            try:
                method(dict(quiescence.request))
            except Exception as exc:  # noqa: BLE001 - classify boundary
                dsh_error = dsh_native_runtime_error_in_chain(exc)
                if dsh_error is None or not dsh_native_runtime_retryable(
                    dsh_error
                ):
                    # An unclassified/contract failure cannot prove silence.
                    # Retain the marker until runtime replacement or recovery.
                    return exc
                attempt += 1
                check_status = callable(status_method)
                if self._stop.wait(
                    self._native_quiescence_retry_delay(attempt)
                ):
                    return RuntimeError("native quiescence was stopped")
                continue
            return None
        if self._stop.is_set():
            return RuntimeError("native quiescence was stopped")
        return RuntimeError("native quiescence ownership was superseded")

    def finish_native_quiescence(
        self,
        action: str,
        request: dict[str, Any],
        runtime: Any,
        marker: int | tuple[int, object],
        *,
        reconcile_runtime_status: bool = False,
    ) -> BaseException | None:
        """Share the safe drain loop with explicit native controls."""

        return self._finish_native_quiescence(
            action,
            _NativeQuiescence(
                request=dict(request),
                runtime=runtime,
                marker=marker,
            ),
            reconcile_runtime_status=reconcile_runtime_status,
        )

    def _start_native_quiescence(
        self,
        action: str,
        quiescence: _NativeQuiescence | None,
        *,
        reconcile_runtime_status: bool = False,
    ) -> None:
        """Start the shared native drain outside the mutation lock."""

        if quiescence is None:
            return

        def quiesce() -> None:
            error = self._finish_native_quiescence(
                action,
                quiescence,
                reconcile_runtime_status=reconcile_runtime_status,
            )
            if error is not None:
                return
            generation_barrier = self.server.acquire_generation_lease(
                str(quiescence.request["run_id"])
            )
            if generation_barrier is None:
                return
            generation_barrier.release()
            self._clear_native_quiescence(quiescence)

        thread = threading.Thread(
            target=quiesce,
            name=(
                f"ecologyrsi-auto-{action}-"
                f"{str(quiescence.request['run_id'])[:8]}"
            ),
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            # Failing to start the drain cannot authorize resume. The closed
            # admission fence and marker remain for startup/operator recovery.
            return

    def _persist_run_failure(
        self, work_item: _WorkItem, failure: _DeferredFailure
    ) -> bool:
        """Persist RunFailed, or return ``True`` to retry only that write later.

        A pause, cancellation, completion, or another failure may win between
        the state read and the compare-and-swap transition.  Re-reading a
        non-running durable state proves that expected race and ends the worker.
        If the ledger cannot be read or still reports ``running``, retain the
        failure for a FIFO retry so the run cannot become an orphaned running
        projection and the failed generation is not replayed.
        """

        bounded_reason = str(failure.reason)[:500]
        run_id = work_item[0]
        try:
            latest = self._state_for_work_item(work_item)
        except KeyError:
            self._clear_deferred_failure(work_item)
            return False
        except Exception:  # noqa: BLE001 - retry an unavailable ledger boundary
            self._defer_failure(work_item, failure)
            return True
        if latest.run.status is not RunStatus.RUNNING:
            self._clear_deferred_failure(work_item)
            return False
        native_quiescence = self._close_native_admission(latest, action="cancel")
        try:
            self.server.director.fail_run(
                run_id,
                bounded_reason,
                error_code=failure.error_code,
                failure_context=failure.failure_context,
            )
        except Exception:  # noqa: BLE001 - distinguish race via durable reread
            try:
                latest = self._state_for_work_item(work_item)
            except KeyError:
                self._clear_native_quiescence(native_quiescence)
                self._clear_deferred_failure(work_item)
                return False
            except Exception:  # noqa: BLE001 - ledger still unavailable
                self._clear_native_quiescence(native_quiescence)
                self._defer_failure(work_item, failure)
                return True
            if latest.run.status is not RunStatus.RUNNING:
                if latest.run.status is RunStatus.FAILED:
                    self._start_native_quiescence("cancel", native_quiescence)
                else:
                    self._clear_native_quiescence(native_quiescence)
                self._clear_deferred_failure(work_item)
                return False
            self._clear_native_quiescence(native_quiescence)
            self._defer_failure(work_item, failure)
            return True
        self._start_native_quiescence("cancel", native_quiescence)
        self._clear_deferred_failure(work_item)
        return False


def _parse_retry_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _transient_persistence_error_in_chain(
    exc: BaseException,
) -> BaseException | None:
    """Return only explicit Host persistence/control infrastructure faults."""

    for error_type in (
        SampleResultCallbackError,
        SampleExecutionControlUnavailableError,
        ConcurrentRunMutationError,
        sqlite3.OperationalError,
    ):
        matched = find_exception(exc, error_type)
        if matched is not None:
            return matched
    return None


def _retry_class_and_error_code(
    exc: BaseException | None,
    *,
    stage: str | None,
) -> tuple[str, str]:
    """Classify one public retry scope without retaining exception text."""

    dsh_error = (
        dsh_native_runtime_error_in_chain(exc) if exc is not None else None
    )
    if dsh_error is not None:
        return (
            "dsh_native_runtime",
            gateway_retry_error_code(
                "dsh_native_runtime",
                getattr(dsh_error, "error_code", None),
            ),
        )
    gateway_error = gateway_error_in_chain(exc) if exc is not None else None
    if gateway_error is not None:
        return (
            "model_gateway",
            gateway_retry_error_code(
                "model_gateway",
                getattr(gateway_error, "error_code", None),
            ),
        )
    if (
        exc is not None
        and stage == "research"
        and find_exception(exc, TimeoutError) is not None
    ):
        # Keep the established public error code while separating timeout
        # failures into their own finite breaker scope.
        return "research_timeout", "timeout"
    if (
        exc is not None
        and _transient_persistence_error_in_chain(exc) is not None
    ):
        return "sample_result_persistence", "sample_result_callback_error"
    return "model_gateway", "gateway_unavailable"


def _latest_failed_stage(state: Any | None) -> str | None:
    """Return the current generation's latest explicitly failed stage."""

    if state is None:
        return None
    generation = int(getattr(getattr(state, "run", None), "generation", -1))
    for event in reversed(tuple(getattr(state, "events", ()) or ())):
        if (
            getattr(event, "kind", None) != "EvolutionStageRecorded"
            or int(event.payload.get("generation", -1)) != generation
        ):
            continue
        if event.payload.get("status") != "failed":
            return None
        stage = event.payload.get("stage")
        return stage if isinstance(stage, str) and stage.strip() else None
    return None


def _research_gateway_failure_retryable_later(
    gateway_error: BaseException,
    *,
    stage: str | None,
) -> bool:
    """Classify response failures that merit a new research call after cooldown.

    The gateway has already exhausted its request-local retry budget. A
    truncated or malformed research response is not evidence that credentials
    or routing are invalid, so retry it as a fresh stage attempt later. Definite
    non-retryable HTTP responses remain terminal and cannot enter an endless
    cooldown loop.
    """

    if stage != "research":
        return False
    status_code = getattr(gateway_error, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return status_code in {408, 425, 429} or 500 <= status_code <= 599
    error_code = str(getattr(gateway_error, "error_code", "") or "")
    return bool(getattr(gateway_error, "split_eligible", False)) or (
        error_code in _RESEARCH_RETRYABLE_RESPONSE_CODES
    )


def _progress_failure_retryable(
    exc: BaseException,
    *,
    stage: str | None = None,
) -> bool:
    """Retry only local recoverable boundaries at the orchestration layer."""

    dsh_error = dsh_native_runtime_error_in_chain(exc)
    if dsh_error is not None:
        return dsh_native_runtime_retryable(dsh_error)
    gateway_error = gateway_error_in_chain(exc)
    if gateway_error is not None:
        # ModelGateway owns request-local retries, but a provider can remain
        # queued/rate-limited after that budget is exhausted.  The caller
        # handles retryable gateway errors with a delayed generation retry;
        # non-retryable contract/configuration errors still terminate safely.
        return bool(gateway_error.retryable) or (
            _research_gateway_failure_retryable_later(
                gateway_error,
                stage=stage,
            )
        )
    if _transient_persistence_error_in_chain(exc) is not None:
        # Sample-result writes are part of the durable evaluation boundary.
        # A transient ledger/IPC failure must be retried after a cooldown, not
        # converted into a terminal generation failure after three attempts.
        return True
    # Built-in exceptions are ambiguous at this boundary. A RuntimeError,
    # ValueError, KeyError, AssertionError, or similar host fault must not be
    # guessed to be transient and replay a potentially non-idempotent work
    # unit. The caller durably pauses these faults at the current checkpoint.
    return False


def _progress_failure_irrecoverable(
    exc: BaseException,
    *,
    stage: str | None = None,
) -> bool:
    """Identify only explicit fail-closed boundaries owned by the Host."""

    if find_exception(exc, FrozenRuntimeBindingDriftError) is not None:
        return True
    dsh_error = dsh_native_runtime_error_in_chain(exc)
    if dsh_error is not None:
        return not dsh_native_runtime_retryable(dsh_error)
    gateway_error = gateway_error_in_chain(exc)
    if gateway_error is not None:
        return not (
            bool(gateway_error.retryable)
            or _research_gateway_failure_retryable_later(
                gateway_error,
                stage=stage,
            )
        )
    return False


def _retry_later_error(
    exc: BaseException,
    *,
    stage: str | None = None,
) -> BaseException | None:
    """Return a boundary that should keep the run alive for a later retry."""

    dsh_error = dsh_native_runtime_error_in_chain(exc)
    if dsh_error is not None:
        return dsh_error if dsh_native_runtime_retryable(dsh_error) else None
    gateway_error = gateway_error_in_chain(exc)
    if gateway_error is not None:
        if gateway_error.retryable or _research_gateway_failure_retryable_later(
            gateway_error,
            stage=stage,
        ):
            return gateway_error
        return None
    if stage == "research" and find_exception(exc, TimeoutError) is not None:
        return exc
    persistence_error = _transient_persistence_error_in_chain(exc)
    if persistence_error is not None:
        return persistence_error
    return None


__all__ = ["AutoProgressManager", "auto_progress_enabled"]
