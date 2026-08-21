"""Durable, bounded background progression for autonomous evolution runs.

The HTTP request that creates a run should not have to stay open for the whole
search budget.  ``AutoProgressManager`` owns a small bounded worker pool and
advances one complete generation at a time.  Every generation still goes
through the same
``execute_generation`` path used by the explicit ``/advance`` endpoint, so the
event ledger remains the source of truth and a process restart can resume a
partially written generation.

Only runs whose immutable task manifest contains ``auto_progress=true`` are
scheduled.  Legacy runs and explicit bounded ``auto_advance`` requests retain
their previous manual-step semantics.
"""

from __future__ import annotations

import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from queue import Empty, Queue
from types import SimpleNamespace
from typing import Any

from ..core.models import RunStatus
from ..core.models import digest
from ..core.errors import (
    dsh_native_runtime_error_in_chain,
    dsh_native_runtime_retryable,
)
from ..core.redaction import public_exception_summary
from ..evaluators.sample_execution import SampleResultCallbackError
from ..evolution.batches import ResearchResponseContractError
from ..integrations.model_gateway import gateway_error_in_chain
from .generation_execution import complete_if_budget_exhausted, execute_generation

_AUTO_PROGRESS_METADATA_KEY = "auto_progress"
_DEFAULT_RETRY_LIMIT = 3
_DEFAULT_WORKER_COUNT = 1
_MAX_WORKER_COUNT = 8
_FAILURE_PERSISTENCE_RETRY_SECONDS = 1.0
_GATEWAY_RETRY_BASE_SECONDS = 15.0
_GATEWAY_RETRY_MAX_SECONDS = 300.0
_GATEWAY_RETRY_DEADLINE_MAX_SECONDS = 3600.0
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


def _run_incarnation(state: Any) -> int:
    events = tuple(getattr(state, "events", ()))
    if not events or getattr(events[0], "kind", None) != "RunCreated":
        raise ValueError("run projection has no leading RunCreated event")
    return int(events[0].seq)


def auto_progress_enabled(state: Any) -> bool:
    """Return whether a replayed run opted into continuous progression."""

    metadata = getattr(getattr(state, "task_manifest", None), "metadata", {})
    return metadata.get(_AUTO_PROGRESS_METADATA_KEY) is True


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
        self._deferred_failures: dict[_WorkItem, str] = {}
        # A request-local gateway retry may already have exhausted its small
        # transport budget while the provider is still queueing work.  Keep
        # the run alive and delay its next generation attempt instead of
        # converting this recoverable boundary into RunFailed.
        self._retry_not_before: dict[_WorkItem, float] = {}
        # Cooldowns are timer-backed rather than worker sleeps.  A busy
        # provider must not occupy the only worker and block unrelated runs.
        self._retry_timers: dict[_WorkItem, threading.Timer] = {}
        self._state_lock = threading.RLock()
        self._stop = threading.Event()
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
        """Return a bounded scheduler snapshot after repairing stale cooldowns."""

        target_run_id = str(run_id).strip()
        # Cooldowns are an in-memory scheduling aid, while the ledger remains
        # authoritative for run lifecycle.  Reconcile before reporting so an
        # externally completed run or a dead Timer cannot remain visible as a
        # permanent cooldown (and cannot orphan a still-running run).
        self._reconcile_retry_cooldowns()
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
        return self._schedule_work_item((run_id, _run_incarnation(state)))

    def forget(self, state: Any) -> None:
        """Discard in-memory bookkeeping for one successfully purged incarnation."""

        work_item = (str(state.run.run_id), _run_incarnation(state))
        with self._state_lock:
            self._clear_retry_cooldown_locked(work_item)
            self._scheduled.discard(work_item)
            self._running.discard(work_item)
            self._reschedule_requested.discard(work_item)
            self._deferred_failures.pop(work_item, None)

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

    def _clear_terminal_retry_cooldown(self, work_item: _WorkItem) -> None:
        """Drop retry state once the durable incarnation reaches a terminal state."""

        try:
            state = self._state_for_work_item(work_item)
        except KeyError:
            with self._state_lock:
                self._clear_retry_cooldown_locked(work_item)
                self._reschedule_requested.discard(work_item)
                self._deferred_failures.pop(work_item, None)
            return
        except Exception:  # noqa: BLE001 - diagnostics must remain best effort
            return
        if state.run.status not in _TERMINAL_RUN_STATUSES:
            return
        with self._state_lock:
            self._clear_retry_cooldown_locked(work_item)
            self._reschedule_requested.discard(work_item)
            self._deferred_failures.pop(work_item, None)

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
            self._retry_not_before.clear()
        return not any(worker.is_alive() for worker in self._threads)

    def _schedule_retry_wakeup(self, work_item: _WorkItem, delay: float) -> bool:
        """Wake one delayed run without occupying a progression worker."""

        delay = max(
            0.0,
            min(float(delay), _GATEWAY_RETRY_DEADLINE_MAX_SECONDS),
        )

        def wake() -> None:
            with self._state_lock:
                self._retry_timers.pop(work_item, None)
                if self._stop.is_set():
                    return
                self._retry_not_before.pop(work_item, None)
            try:
                state = self.server.director.state(work_item[0])
                if (
                    _run_incarnation(state) != work_item[1]
                    or state.run.status is not RunStatus.RUNNING
                    or not auto_progress_enabled(state)
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
                    self._clear_retry_cooldown_locked(work_item)
            return False
        return True

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                work_item = self._queue.get(timeout=0.25)
            except Empty:
                continue
            continue_running = False
            reschedule_requested = False
            dispatch = False
            retry_wait = 0.0
            retry_wakeup_scheduled = False
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
                            # Timer.start() can fail under resource pressure.
                            # The current worker still owns this item, so execute
                            # it now instead of leaving a running run orphaned.
                            self._running.add(work_item)
                            dispatch = True
                    if retry_wakeup_scheduled or self._stop.is_set():
                        continue
                else:
                    with self._state_lock:
                        # ``_scheduled`` spans both queueing and execution.  That
                        # prevents another worker from dequeuing this run while
                        # its current generation still holds the shared lease.
                        self._running.add(work_item)
                        dispatch = True
                continue_running = self._run_one_generation(
                    work_item[0], expected_incarnation=work_item[1]
                )
                if not continue_running:
                    self._clear_terminal_retry_cooldown(work_item)
            except Exception as exc:  # noqa: BLE001 - isolate one queue item
                if work_item is not None:
                    self._defer_failure(
                        work_item,
                        "自动推进工作器异常："
                        f"{public_exception_summary(exc)}",
                    )
                    # Put the failed item at the FIFO tail. Its next turn only
                    # persists the bounded terminal failure, while later work
                    # can proceed on this still-live worker.
                    continue_running = True
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
        # ``generation_execution`` deliberately depends on a tiny endpoint
        # protocol (``endpoint.server``) shared with the HTTP mixin.  A simple
        # namespace keeps that code path identical without constructing a fake
        # BaseHTTPRequestHandler instance.
        endpoint = SimpleNamespace(server=self.server)
        deferred_reason = self._deferred_failure(work_item)
        if deferred_reason is not None:
            with self.server.mutation_lock:
                retry_terminal_write = self._persist_run_failure(
                    work_item, deferred_reason
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
                try:
                    state = complete_if_budget_exhausted(endpoint, run_id, state)
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
                        self._defer_retry(
                            work_item,
                            self._gateway_retry_delay(
                                work_item, failures, deferred_error
                            ),
                            deferred_error,
                        )
                        return True
                    if retryable and failures < self._retry_limit:
                        # The generation executor is event-idempotent and can
                        # reconcile a partially written batch on the next
                        # attempt.  Retry transient gateway/ledger boundaries a
                        # small number of times without busy-spinning.
                        retry_delay = min(2.0, 0.25 * (2 ** (failures - 1)))
                    if not retryable or failures >= self._retry_limit:
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
                        try:
                            retry_terminal_write = self._persist_run_failure(
                                work_item,
                                failure_reason,
                            )
                        except Exception:  # noqa: BLE001
                            # ``_persist_run_failure`` owns ordinary ledger and
                            # terminal-race handling.  If its in-memory recovery
                            # bookkeeping itself fails, retain the original
                            # bounded failure and keep the run queued.
                            self._defer_failure(work_item, failure_reason)
                            retry_terminal_write = True

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
                result = execute_generation(endpoint, run_id)
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
                recovery_stage = _latest_failed_stage(recovery_state)
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
                        work_item, failures, deferred_error
                    )
                    self._defer_retry(
                        work_item,
                        retry_delay,
                        deferred_error,
                        stage=recovery_stage,
                    )
                    return True
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
                with self.server.mutation_lock:
                    try:
                        latest = self._state_for_work_item(work_item)
                        if latest.run.status is not RunStatus.RUNNING:
                            return False
                        if terminal_failure:
                            assert failure_reason is not None
                            retry_terminal_write = self._persist_run_failure(
                                work_item,
                                failure_reason,
                            )
                    except KeyError:
                        self._clear_deferred_failure(work_item)
                        return False
                    except Exception:  # noqa: BLE001
                        if terminal_failure:
                            assert failure_reason is not None
                            self._defer_failure(work_item, failure_reason)
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
    ) -> str | None:
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

    def _defer_failure(self, work_item: _WorkItem, reason: str) -> None:
        with self._state_lock:
            self._deferred_failures[work_item] = str(reason)[:500]

    def _defer_retry(
        self,
        work_item: _WorkItem,
        delay_seconds: float,
        exc: BaseException | None = None,
        *,
        stage: str | None = None,
    ) -> None:
        """Schedule a recoverable gateway retry without busy-spinning."""

        delay = max(
            0.0,
            min(float(delay_seconds), _GATEWAY_RETRY_DEADLINE_MAX_SECONDS),
        )
        retry_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
        with self._state_lock:
            self._retry_not_before[work_item] = time.monotonic() + delay
        # Keep the cooldown visible in the append-only public projection.  The
        # in-memory deadline still owns scheduling; this event is deliberately
        # observational and does not alter the run state.
        try:
            state = self._state_for_work_item(work_item)
            attempt = int(self._retry_attempt(work_item))
            callback_failure = (
                exc is not None
                and _contains_exception_type(exc, SampleResultCallbackError)
            )
            gateway_error = (
                gateway_error_in_chain(exc) if exc is not None else None
            )
            dsh_error = (
                dsh_native_runtime_error_in_chain(exc)
                if exc is not None
                else None
            )
            timeout_failure = (
                exc is not None
                and gateway_error is None
                and _contains_exception_type(exc, TimeoutError)
            )
            error_code = (
                getattr(gateway_error, "error_code", None)
                or (getattr(exc, "error_code", None) if exc is not None else None)
            )
            research_contract_retry = error_code in {
                "research_algorithm_contract_invalid",
                "research_response_contract_invalid",
            }
            payload = {
                "generation": int(state.run.generation),
                "retry_at": retry_at.isoformat(),
                "delay_seconds": round(delay, 3),
                "attempt": attempt,
                "error_code": error_code
                or ("sample_result_callback_error" if callback_failure else None)
                or ("timeout" if timeout_failure else None),
                **({"stage": stage} if stage is not None else {}),
                "reason": (
                    "DSH 智能体运行时暂时不可用，"
                    "任务保持运行并将在服务恢复后自动重试"
                    if dsh_error is not None
                    else "样本结果写入暂时不可用，等待持久化边界恢复后再次调用"
                    if callback_failure
                    else (
                        "研究方案响应未通过宿主算法合同，"
                        "等待后重新请求本轮研究计划"
                    )
                    if research_contract_retry
                    else (
                        "研究计划输出被截断，"
                        "等待冷却后重新请求本轮研究计划"
                    )
                    if stage == "research" and error_code == "output_truncated"
                    else (
                        "研究阶段模型请求暂时不可用，"
                        "等待冷却后重新请求本轮研究计划"
                    )
                    if stage == "research"
                    else (
                        "网关请求已完成本地重试，"
                        "等待服务端队列恢复后再次调用"
                    )
                ),
            }
            # Keep the heartbeat idempotent and bounded.  A millisecond clock
            # is not sufficient when several worker/control paths report the
            # same generation in quick succession.
            event_id = (
                f"{work_item[0]}:gateway-retry:{state.run.generation}:"
                f"{attempt}:{digest(payload)[:16]}"
            )
            lock = getattr(self.server, "mutation_lock", None)
            if lock is None:
                self.server.ledger.append(
                    work_item[0],
                    "GatewayRetryScheduled",
                    payload,
                    event_id=event_id,
                )
            else:
                with lock:
                    self.server.ledger.append(
                        work_item[0],
                        "GatewayRetryScheduled",
                        payload,
                        event_id=event_id,
                    )
        except Exception:
            # Progress reporting must never turn a recoverable gateway error
            # into a terminal run failure (or mask the original exception).
            return

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
        # A later stage event proves that the worker already resumed the
        # generation after this heartbeat (the process may have crashed before
        # the next durable boundary).  In that case restoring the old cooldown
        # would add an unnecessary multi-minute pause after restart.
        if any(
            event.kind == "EvolutionStageRecorded"
            and event.seq > latest.seq
            and int(event.payload.get("generation", -1))
            == int(state.run.generation)
            for event in getattr(state, "events", ())
        ):
            return
        raw_retry_at = latest.payload.get("retry_at")
        if not isinstance(raw_retry_at, str) or not raw_retry_at.strip():
            return
        try:
            retry_at = datetime.fromisoformat(raw_retry_at)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return
        with self._state_lock:
            self._retry_not_before[work_item] = time.monotonic() + min(
                delay, _GATEWAY_RETRY_DEADLINE_MAX_SECONDS
            )

    def _retry_attempt(self, work_item: _WorkItem) -> int:
        """Return a bounded, best-effort retry count for public diagnostics."""

        # The manager stores deadlines rather than counters.  Counting recent
        # durable heartbeats gives a restart-safe approximation and is enough
        # for the UI; exact transport attempts remain gateway diagnostics.
        try:
            state = self._state_for_work_item(work_item)
            return max(
                1,
                sum(
                    1
                    for event in state.events
                    if event.kind == "GatewayRetryScheduled"
                    and int(event.payload.get("generation", -1)) == int(state.run.generation)
                )
                + 1,
            )
        except Exception:
            return 1

    def _gateway_retry_delay(
        self,
        work_item: _WorkItem,
        failures: int,
        exc: BaseException | None = None,
    ) -> float:
        """Combine durable backoff with a bounded provider retry deadline."""

        durable_attempt = max(1, self._retry_attempt(work_item))
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
        gateway_error = gateway_error_in_chain(exc) if exc is not None else None
        retry_after = (
            getattr(gateway_error, "retry_after_seconds", None)
            if gateway_error is not None
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

    def _persist_run_failure(self, work_item: _WorkItem, reason: str) -> bool:
        """Persist RunFailed, or return ``True`` to retry only that write later.

        A pause, cancellation, completion, or another failure may win between
        the state read and the compare-and-swap transition.  Re-reading a
        non-running durable state proves that expected race and ends the worker.
        If the ledger cannot be read or still reports ``running``, retain the
        failure for a FIFO retry so the run cannot become an orphaned running
        projection and the failed generation is not replayed.
        """

        bounded_reason = str(reason)[:500]
        run_id = work_item[0]
        try:
            latest = self._state_for_work_item(work_item)
        except KeyError:
            self._clear_deferred_failure(work_item)
            return False
        except Exception:  # noqa: BLE001 - retry an unavailable ledger boundary
            self._defer_failure(work_item, bounded_reason)
            return True
        if latest.run.status is not RunStatus.RUNNING:
            self._clear_deferred_failure(work_item)
            return False
        try:
            self.server.director.fail_run(run_id, bounded_reason)
        except Exception:  # noqa: BLE001 - distinguish race via durable reread
            try:
                latest = self._state_for_work_item(work_item)
            except KeyError:
                self._clear_deferred_failure(work_item)
                return False
            except Exception:  # noqa: BLE001 - ledger still unavailable
                self._defer_failure(work_item, bounded_reason)
                return True
            if latest.run.status is not RunStatus.RUNNING:
                self._clear_deferred_failure(work_item)
                return False
            self._defer_failure(work_item, bounded_reason)
            return True
        self._clear_deferred_failure(work_item)
        return False


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
    dsh_error = dsh_native_runtime_error_in_chain(exc)
    if dsh_error is not None:
        return dsh_native_runtime_retryable(dsh_error)
    if _contains_exception_type(exc, SampleResultCallbackError):
        # Sample-result writes are part of the durable evaluation boundary.
        # A transient ledger/IPC failure must be retried after a cooldown, not
        # converted into a terminal generation failure after three attempts.
        return True
    if stage == "research" and _contains_exception_type(
        exc,
        ResearchResponseContractError,
    ):
        return True
    return not isinstance(exc, (KeyError, TypeError, ValueError))


def _retry_later_error(
    exc: BaseException,
    *,
    stage: str | None = None,
) -> BaseException | None:
    """Return a boundary that should keep the run alive for a later retry."""

    gateway_error = gateway_error_in_chain(exc)
    if gateway_error is not None:
        if gateway_error.retryable or _research_gateway_failure_retryable_later(
            gateway_error,
            stage=stage,
        ):
            return gateway_error
        return None
    dsh_error = dsh_native_runtime_error_in_chain(exc)
    if dsh_error is not None:
        return dsh_error if dsh_native_runtime_retryable(dsh_error) else None
    if stage == "research" and _contains_exception_type(exc, TimeoutError):
        return exc
    if _contains_exception_type(exc, SampleResultCallbackError):
        return exc
    if stage == "research" and _contains_exception_type(
        exc,
        ResearchResponseContractError,
    ):
        return exc
    return None


def _contains_exception_type(
    exc: BaseException,
    expected_type: type[BaseException],
    *,
    max_depth: int = 32,
) -> bool:
    """Return whether an exception chain contains ``expected_type``."""

    pending: list[tuple[BaseException, int]] = [(exc, 0)]
    seen: set[int] = set()
    while pending:
        current, depth = pending.pop()
        identity = id(current)
        if identity in seen or depth > max_depth:
            continue
        seen.add(identity)
        if isinstance(current, expected_type):
            return True
        for related in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(related, BaseException):
                pending.append((related, depth + 1))
        related_group = getattr(current, "exceptions", None)
        if isinstance(related_group, (tuple, list)):
            for related in related_group:
                if isinstance(related, BaseException):
                    pending.append((related, depth + 1))
    return False


__all__ = ["AutoProgressManager", "auto_progress_enabled"]
