from __future__ import annotations

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import TaskManifest, digest
from ecologyrsi_dsh.evolution.strategies import FakeDSHAdapter
from ecologyrsi_dsh.api.events import EventEndpointsMixin
from ecologyrsi_dsh.api.projection import _projection_json


class MutableClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta


class GatewayRetryCircuitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.ledger = EventLedger(Path(self.directory.name) / "events.sqlite3")
        self.director = EvolutionDirector(self.ledger, FakeDSHAdapter())
        self.task = TaskManifest(
            task_id="gateway-retry-circuit",
            objective="bound orchestration-level model gateway retries",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={"max_candidates": 1, "max_generations": 1},
            metadata={"strategy_id": "parameter_sweep@1"},
        )
        self.run_id = self.director.start_evolution(
            self.task,
            run_id="run:gateway-retry-circuit",
        ).run.run_id

    def tearDown(self) -> None:
        self.ledger.close()
        self.directory.cleanup()

    def _failure_kwargs(self, attempt: int, *, failure_epoch: int = 1):
        state = self.director.state(self.run_id)
        return {
            "run_incarnation": state.events[0].seq,
            "generation": state.run.generation,
            "stage": "research",
            "retry_class": "model_gateway",
            "failure_id": digest(
                {
                    "run_id": self.run_id,
                    "generation": state.run.generation,
                    "stage": "research",
                    "retry_class": "model_gateway",
                    "failure_epoch": failure_epoch,
                    "attempt": attempt,
                }
            ),
            "attempt_anchor_seq": state.events[-1].seq,
            "delay_seconds": 0.0,
            "last_error_code": "gateway_unavailable",
        }

    def _report_failure(self, attempt: int, *, failure_epoch: int = 1):
        return self.director.schedule_gateway_retry_or_pause(
            self.run_id,
            **self._failure_kwargs(attempt, failure_epoch=failure_epoch),
        )

    def _record_dsh_success(self, *, generation: int = 0):
        structured = {
            "schema_version": "ecology-sample-reflection@1",
            "summary": "The DSH runtime returned a valid structured result.",
        }
        return self.ledger.append(
            self.run_id,
            "DshStructuredResultAccepted",
            {
                "schema_version": "ecologyrsi-dsh.structured-result-accepted/1",
                "identity": {
                    "run_id": self.run_id,
                    "generation": generation,
                    "role": "sample-critic",
                    "stage": "sample.reflect",
                    "session_id": "dsh-child-retry-recovery",
                },
                "output_schema_id": "ecology-sample-reflection@1",
                "result_digest": digest(structured),
                "structured": structured,
                "skill_invocation_evidence": {
                    "schema_version": "ecologyrsi-dsh.skill-invocation-evidence/1",
                    "stage": "sample.reflect",
                    "skill_name": "origin-vector-review",
                    "call_count": 1,
                    "successful_call_count": 1,
                    "call_seq": 1,
                    "result_seq": 2,
                    "first_tool_call_verified": True,
                    "next_tool_name": "structured_output",
                    "next_tool_call_seq": 3,
                    "order_verified": True,
                    "source": "dsh_session_event_log",
                },
            },
            event_id=f"{self.run_id}:dsh-success:{generation}",
        )

    def _race_failure_against(self, failure: dict, competing_action):
        entered_append = threading.Event()
        release_append = threading.Event()
        original_append = self.ledger.append
        target_ids = {
            f"{self.run_id}:gateway-retry:{failure['failure_id']}",
            f"{self.run_id}:gateway-circuit:{failure['failure_id']}",
        }

        def blocked_append(run_id, kind, payload, **kwargs):
            if kwargs.get("event_id") in target_ids:
                entered_append.set()
                if not release_append.wait(timeout=3):
                    raise TimeoutError("retry decision race was not released")
            return original_append(run_id, kind, payload, **kwargs)

        with (
            patch.object(self.ledger, "append", side_effect=blocked_append),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            future = pool.submit(
                self.director.schedule_gateway_retry_or_pause,
                self.run_id,
                **failure,
            )
            self.assertTrue(entered_append.wait(timeout=2))
            try:
                competing_action()
            finally:
                release_append.set()
            return future.result(timeout=3)

    def test_sixth_model_gateway_failure_opens_checkpoint_preserving_pause(
        self,
    ) -> None:
        for attempt in range(1, 6):
            decision = self._report_failure(attempt)
            self.assertEqual(decision.outcome, "scheduled")
            self.assertEqual(decision.event.kind, "GatewayRetryScheduled")
            self.assertEqual(
                self.director.state(self.run_id).run.status.value,
                "running",
            )

        decision = self._report_failure(6)

        self.assertEqual(decision.outcome, "paused")
        self.assertEqual(decision.event.kind, "RunPaused")
        state = self.director.state(self.run_id)
        self.assertEqual(state.run.status.value, "paused")
        self.assertEqual(state.run.generation, 0)
        self.assertEqual(
            decision.event.payload,
            {
                "code": "gateway_retry_circuit_open",
                "retry_class": "model_gateway",
                "generation": 0,
                "stage": "research",
                "breaker_epoch": 1,
                "consecutive_failures": 6,
                "retry_limit": 6,
                "first_failure_at": decision.event.payload["first_failure_at"],
                "last_failure_at": decision.event.payload["last_failure_at"],
                "last_error_code": "gateway_unavailable",
                "suggested_action": "check_gateway_then_resume",
                "pause_trigger": "failure_limit",
                "epoch_seconds": 30 * 60,
                "epoch_deadline_at": decision.event.payload[
                    "epoch_deadline_at"
                ],
                "proposed_retry_at": decision.event.payload[
                    "proposed_retry_at"
                ],
                "retry_delay_seconds": 0.0,
            },
        )
        self.assertEqual(
            len(
                [
                    event
                    for event in state.events
                    if event.kind == "GatewayRetryScheduled"
                ]
            ),
            5,
        )
        self.assertFalse(
            any(
                event.kind in {"RunFailed", "CandidateFailed"}
                for event in state.events
            )
        )
        self.assertNotIn("raw", str(decision.event.payload).lower())

    def test_duplicate_failure_id_is_replayed_without_incrementing(self) -> None:
        failure = self._failure_kwargs(1)

        first = self.director.schedule_gateway_retry_or_pause(
            self.run_id,
            **failure,
        )
        duplicate = self.director.schedule_gateway_retry_or_pause(
            self.run_id,
            **failure,
        )

        self.assertEqual(first.outcome, "scheduled")
        self.assertEqual(duplicate.outcome, "scheduled")
        self.assertTrue(duplicate.replayed)
        self.assertEqual(duplicate.event, first.event)
        state = self.director.state(self.run_id)
        retry_events = [
            event for event in state.events if event.kind == "GatewayRetryScheduled"
        ]
        self.assertEqual(len(retry_events), 1)
        self.assertEqual(retry_events[0].payload["consecutive_failures"], 1)

    def test_one_attempt_anchor_cannot_count_two_distinct_failure_ids(self) -> None:
        first_failure = self._failure_kwargs(1)
        second_failure = dict(first_failure)
        second_failure["failure_id"] = digest("same-anchor-distinct-report")

        first = self.director.schedule_gateway_retry_or_pause(
            self.run_id,
            **first_failure,
        )
        stale = self.director.schedule_gateway_retry_or_pause(
            self.run_id,
            **second_failure,
        )

        self.assertEqual(first.outcome, "scheduled")
        self.assertEqual(stale.outcome, "superseded")
        self.assertIsNone(stale.event)
        retry_events = [
            event
            for event in self.director.state(self.run_id).events
            if event.kind == "GatewayRetryScheduled"
        ]
        self.assertEqual(len(retry_events), 1)

    def test_two_workers_with_same_failure_id_linearize_to_one_decision(self) -> None:
        failure = self._failure_kwargs(1)
        barrier = threading.Barrier(2)
        append_lock = threading.Lock()
        decision_appends = 0
        original_append = self.ledger.append

        def append_with_barrier(run_id, kind, payload, **kwargs):
            nonlocal decision_appends
            should_wait = False
            if kind in {"GatewayRetryScheduled", "RunPaused"}:
                with append_lock:
                    decision_appends += 1
                    should_wait = decision_appends <= 2
            if should_wait:
                barrier.wait(timeout=2)
            return original_append(run_id, kind, payload, **kwargs)

        with (
            patch.object(self.ledger, "append", side_effect=append_with_barrier),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            decisions = list(
                pool.map(
                    lambda _index: self.director.schedule_gateway_retry_or_pause(
                        self.run_id,
                        **failure,
                    ),
                    range(2),
                )
            )

        self.assertEqual([item.outcome for item in decisions], ["scheduled"] * 2)
        self.assertEqual(sum(item.replayed for item in decisions), 1)
        self.assertEqual(decisions[0].event, decisions[1].event)
        self.assertEqual(
            len(
                [
                    event
                    for event in self.director.state(self.run_id).events
                    if event.kind == "GatewayRetryScheduled"
                ]
            ),
            1,
        )

    def test_two_workers_with_same_anchor_cannot_double_count_distinct_ids(
        self,
    ) -> None:
        first = self._failure_kwargs(1)
        second = dict(first)
        second["failure_id"] = digest("concurrent-distinct-failure-id")
        barrier = threading.Barrier(2)
        append_lock = threading.Lock()
        decision_appends = 0
        original_append = self.ledger.append

        def append_with_barrier(run_id, kind, payload, **kwargs):
            nonlocal decision_appends
            should_wait = False
            if kind in {"GatewayRetryScheduled", "RunPaused"}:
                with append_lock:
                    decision_appends += 1
                    should_wait = decision_appends <= 2
            if should_wait:
                barrier.wait(timeout=2)
            return original_append(run_id, kind, payload, **kwargs)

        with (
            patch.object(self.ledger, "append", side_effect=append_with_barrier),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            futures = [
                pool.submit(
                    self.director.schedule_gateway_retry_or_pause,
                    self.run_id,
                    **failure,
                )
                for failure in (first, second)
            ]
            decisions = [future.result(timeout=3) for future in futures]

        self.assertEqual(
            sorted(item.outcome for item in decisions),
            ["scheduled", "superseded"],
        )
        retry_events = [
            event
            for event in self.director.state(self.run_id).events
            if event.kind == "GatewayRetryScheduled"
        ]
        self.assertEqual(len(retry_events), 1)
        self.assertEqual(retry_events[0].payload["consecutive_failures"], 1)

    def test_durable_stage_success_resets_epoch_but_stage_start_does_not(self) -> None:
        for attempt in range(1, 6):
            self.assertEqual(self._report_failure(attempt).outcome, "scheduled")
        state = self.director.state(self.run_id)
        self.director.record_evolution_stage(
            self.run_id,
            generation=state.run.generation,
            stage="research",
            status="started",
            attempt=2,
        )

        sixth_failure = self._failure_kwargs(6)
        sixth_failure["attempt_anchor_seq"] = next(
            event.seq
            for event in reversed(self.director.state(self.run_id).events)
            if event.kind == "GatewayRetryScheduled"
        )
        paused = self.director.schedule_gateway_retry_or_pause(
            self.run_id,
            **sixth_failure,
        )

        self.assertEqual(paused.outcome, "paused")
        self.director.resume_run(self.run_id)
        self.director.record_evolution_stage(
            self.run_id,
            generation=0,
            stage="research",
            status="completed",
            attempt=2,
        )
        reset = self._report_failure(1, failure_epoch=2)
        self.assertEqual(reset.outcome, "scheduled")
        self.assertEqual(reset.event.payload["consecutive_failures"], 1)
        self.assertEqual(reset.event.payload["breaker_epoch"], 2)

    def test_success_at_n_minus_one_resets_before_next_failure(self) -> None:
        for attempt in range(1, 6):
            self.assertEqual(self._report_failure(attempt).outcome, "scheduled")
        self.director.record_evolution_stage(
            self.run_id,
            generation=0,
            stage="research",
            status="completed",
            attempt=2,
        )

        recovered_then_failed = self._report_failure(1, failure_epoch=2)

        self.assertEqual(recovered_then_failed.outcome, "scheduled")
        self.assertEqual(
            recovered_then_failed.event.payload["consecutive_failures"],
            1,
        )
        self.assertEqual(recovered_then_failed.event.payload["breaker_epoch"], 2)

    def test_dsh_structured_success_breaks_native_runtime_retry_epoch(self) -> None:
        for attempt in range(1, 6):
            state = self.director.state(self.run_id)
            decision = self.director.schedule_gateway_retry_or_pause(
                self.run_id,
                run_incarnation=state.events[0].seq,
                generation=0,
                stage="evaluation",
                retry_class="dsh_native_runtime",
                failure_id=digest({"native-failure": attempt}),
                attempt_anchor_seq=state.events[-1].seq,
                delay_seconds=0.0,
                last_error_code="dsh_native_runtime_unavailable",
            )
            self.assertEqual(decision.outcome, "scheduled")

        accepted = self._record_dsh_success()
        state = self.director.state(self.run_id)
        recovered_then_failed = self.director.schedule_gateway_retry_or_pause(
            self.run_id,
            run_incarnation=state.events[0].seq,
            generation=0,
            stage="evaluation",
            retry_class="dsh_native_runtime",
            failure_id=digest("native-failure-after-accepted-result"),
            attempt_anchor_seq=accepted.seq,
            delay_seconds=0.0,
            last_error_code="dsh_native_runtime_unavailable",
        )

        self.assertEqual(recovered_then_failed.outcome, "scheduled")
        self.assertEqual(
            recovered_then_failed.event.payload["consecutive_failures"],
            1,
        )
        self.assertEqual(recovered_then_failed.event.payload["breaker_epoch"], 2)

    def test_legacy_v2_native_retry_after_structured_success_still_replays(self) -> None:
        """0.3.27 did not treat unrelated DSH success as an epoch reset."""

        state = self.director.state(self.run_id)
        attempt_anchor_seq = state.events[-1].seq
        self._record_dsh_success()
        timestamp = datetime(2026, 8, 26, 4, 0, tzinfo=timezone.utc).isoformat()
        legacy_payload = {
            "schema_version": "ecologyrsi-dsh.gateway-retry-scheduled/2",
            "run_incarnation": state.events[0].seq,
            "generation": 0,
            "stage": "generation",
            "retry_class": "dsh_native_runtime",
            "breaker_epoch": 1,
            "failure_id": digest("legacy-native-retry-after-dsh-success"),
            "attempt_anchor_seq": attempt_anchor_seq,
            "consecutive_failures": 1,
            "retry_limit": 6,
            "first_failure_at": timestamp,
            "last_failure_at": timestamp,
            "last_error_code": "dsh_native_runtime_unavailable",
            "retry_at": timestamp,
            "delay_seconds": 0.0,
            "attempt": 1,
            "error_code": "dsh_native_runtime_unavailable",
            "reason": "DSH 智能体运行时暂时不可用，已安排有界延迟重试。",
        }
        self.ledger.append(
            self.run_id,
            "GatewayRetryScheduled",
            legacy_payload,
            event_id=f"{self.run_id}:legacy-native-retry-after-success",
        )

        replayed = self.director.state(self.run_id)

        self.assertEqual(replayed.run.status.value, "running")
        self.assertEqual(replayed.events[-1].payload, legacy_payload)

    def test_threshold_cas_loses_cleanly_to_cancellation(self) -> None:
        for attempt in range(1, 6):
            self._report_failure(attempt)
        sixth = self._failure_kwargs(6)

        decision = self._race_failure_against(
            sixth,
            lambda: self.director.cancel_run(self.run_id, "operator cancelled"),
        )

        self.assertEqual(decision.outcome, "superseded")
        state = self.director.state(self.run_id)
        self.assertEqual(state.run.status.value, "cancelled")
        cancelled = next(
            event for event in reversed(state.events) if event.kind == "RunCancelled"
        )
        self.assertFalse(
            any(event.kind == "RunPaused" and event.seq > cancelled.seq for event in state.events)
        )

    def test_threshold_cas_loses_cleanly_to_manual_pause(self) -> None:
        for attempt in range(1, 6):
            self._report_failure(attempt)
        sixth = self._failure_kwargs(6)

        decision = self._race_failure_against(
            sixth,
            lambda: self.director.pause_run(
                self.run_id,
                code="operator_backpressure",
            ),
        )

        self.assertEqual(decision.outcome, "superseded")
        state = self.director.state(self.run_id)
        self.assertEqual(state.run.status.value, "paused")
        pauses = [event for event in state.events if event.kind == "RunPaused"]
        self.assertEqual(len(pauses), 1)
        self.assertEqual(pauses[0].payload["code"], "operator_backpressure")

    def test_retry_cas_loses_cleanly_to_scoped_stage_success(self) -> None:
        first = self._report_failure(1).event
        second = self._failure_kwargs(2)
        second["attempt_anchor_seq"] = first.seq

        decision = self._race_failure_against(
            second,
            lambda: self.director.record_evolution_stage(
                self.run_id,
                generation=0,
                stage="research",
                status="completed",
                attempt=2,
            ),
        )

        self.assertEqual(decision.outcome, "superseded")
        state = self.director.state(self.run_id)
        self.assertEqual(state.run.status.value, "running")
        self.assertEqual(
            len([event for event in state.events if event.kind == "GatewayRetryScheduled"]),
            1,
        )

    def test_retry_cas_loses_cleanly_to_generation_advance(self) -> None:
        failure = self._failure_kwargs(1)

        decision = self._race_failure_against(
            failure,
            lambda: self.director.advance_generation(self.run_id),
        )

        self.assertEqual(decision.outcome, "superseded")
        state = self.director.state(self.run_id)
        self.assertEqual(state.run.generation, 1)
        self.assertFalse(
            any(event.kind == "GatewayRetryScheduled" for event in state.events)
        )

    def test_old_worker_report_after_resume_cannot_pollute_new_epoch(self) -> None:
        for attempt in range(1, 6):
            self._report_failure(attempt)
        stale = self._failure_kwargs(6)
        current = dict(stale)
        current["failure_id"] = digest("winner-opens-first-epoch-circuit")
        self.assertEqual(
            self.director.schedule_gateway_retry_or_pause(
                self.run_id,
                **current,
            ).outcome,
            "paused",
        )
        self.director.resume_run(self.run_id)

        late = self.director.schedule_gateway_retry_or_pause(
            self.run_id,
            **stale,
        )

        self.assertEqual(late.outcome, "superseded")
        self.assertIsNone(late.event)
        state = self.director.state(self.run_id)
        self.assertEqual(state.run.status.value, "running")
        self.assertEqual(state.events[-1].kind, "RunResumed")

    def test_circuit_resume_records_origin_and_second_epoch_is_bounded(self) -> None:
        for attempt in range(1, 7):
            decision = self._report_failure(attempt)
        self.assertEqual(decision.outcome, "paused")

        self.director.resume_run(self.run_id)

        resumed = self.director.state(self.run_id).events[-1]
        self.assertEqual(resumed.kind, "RunResumed")
        self.assertEqual(
            resumed.payload,
            {
                "resume_origin": "gateway_retry_circuit_open",
                "origin_pause_event_id": decision.event.event_id,
                "origin_breaker_epoch": 1,
                "retry_class": "model_gateway",
                "generation": 0,
                "stage": "research",
                "breaker_epoch": 2,
            },
        )
        for attempt in range(1, 7):
            decision = self._report_failure(attempt, failure_epoch=2)
        self.assertEqual(decision.outcome, "paused")
        self.assertEqual(decision.event.payload["breaker_epoch"], 2)
        self.assertEqual(decision.event.payload["consecutive_failures"], 6)
        state = self.director.state(self.run_id)
        self.assertEqual(
            len([event for event in state.events if event.kind == "RunPaused"]),
            2,
        )

    def test_circuit_pause_cannot_be_bypassed_with_start_transition(self) -> None:
        for attempt in range(1, 7):
            self._report_failure(attempt)
        before = self.director.state(self.run_id)

        with self.assertRaisesRegex(
            RuntimeError,
            "gateway circuit requires an explicit resume",
        ):
            self.director.start_run(self.run_id)

        after = self.director.state(self.run_id)
        self.assertEqual(after.run.status.value, "paused")
        self.assertEqual(after.events, before.events)

    def test_manual_pause_keeps_legacy_start_transition_compatibility(self) -> None:
        manual_run_id = "run:manual-pause-start"
        self.director.start_evolution(self.task, run_id=manual_run_id)
        self.director.pause_run(manual_run_id, code="operator_backpressure")

        restarted = self.director.start_run(manual_run_id)

        self.assertEqual(restarted.status.value, "running")
        self.assertEqual(self.director.state(manual_run_id).events[-1].kind, "RunStarted")

    def test_thirty_minute_epoch_opens_before_six_failures(self) -> None:
        clock = MutableClock(datetime(2026, 8, 26, 8, 0, tzinfo=timezone.utc))
        self.director = EvolutionDirector(
            self.ledger,
            FakeDSHAdapter(),
            clock=clock,
        )
        first = self._report_failure(1)
        self.assertEqual(first.outcome, "scheduled")

        clock.advance(timedelta(minutes=30))
        expired = self._report_failure(2)

        self.assertEqual(expired.outcome, "paused")
        self.assertEqual(expired.event.payload["consecutive_failures"], 2)
        self.assertEqual(
            expired.event.payload["first_failure_at"],
            "2026-08-26T08:00:00+00:00",
        )
        self.assertEqual(
            expired.event.payload["last_failure_at"],
            "2026-08-26T08:30:00+00:00",
        )
        self.assertEqual(expired.event.payload["pause_trigger"], "epoch_elapsed")
        self.assertEqual(expired.event.payload["epoch_seconds"], 30 * 60)
        self.assertEqual(
            expired.event.payload["epoch_deadline_at"],
            "2026-08-26T08:30:00+00:00",
        )
        self.assertEqual(
            expired.event.payload["proposed_retry_at"],
            "2026-08-26T08:30:00+00:00",
        )

    def test_retry_delay_reaching_epoch_deadline_pauses_without_late_probe(self) -> None:
        failure = self._failure_kwargs(1)
        failure["delay_seconds"] = 60 * 60

        decision = self.director.schedule_gateway_retry_or_pause(
            self.run_id,
            **failure,
        )

        self.assertEqual(decision.outcome, "paused")
        self.assertEqual(decision.event.payload["consecutive_failures"], 1)
        self.assertEqual(
            decision.event.payload["pause_trigger"],
            "retry_deadline_reaches_epoch",
        )
        first_failure_at = datetime.fromisoformat(
            decision.event.payload["first_failure_at"]
        )
        self.assertEqual(
            datetime.fromisoformat(decision.event.payload["epoch_deadline_at"]),
            first_failure_at + timedelta(minutes=30),
        )
        self.assertEqual(
            datetime.fromisoformat(decision.event.payload["proposed_retry_at"]),
            first_failure_at + timedelta(hours=1),
        )
        state = self.director.state(self.run_id)
        self.assertEqual(state.run.status.value, "paused")
        self.assertFalse(
            any(event.kind == "GatewayRetryScheduled" for event in state.events)
        )

        self.director.resume_run(self.run_id)
        resumed_retry = self._report_failure(1, failure_epoch=2)
        self.assertEqual(resumed_retry.outcome, "scheduled")
        self.assertEqual(resumed_retry.event.payload["breaker_epoch"], 2)

    def test_replay_rejects_count_two_pause_without_trigger_evidence(self) -> None:
        first = self._report_failure(1).event
        self.ledger.append(
            self.run_id,
            "RunPaused",
            {
                "code": "gateway_retry_circuit_open",
                "retry_class": "model_gateway",
                "generation": 0,
                "stage": "research",
                "breaker_epoch": first.payload["breaker_epoch"],
                "consecutive_failures": 2,
                "retry_limit": 6,
                "first_failure_at": first.payload["first_failure_at"],
                "last_failure_at": first.payload["last_failure_at"],
                "last_error_code": "gateway_unavailable",
                "suggested_action": "check_gateway_then_resume",
            },
            event_id=f"{self.run_id}:forged-count-two-pause",
        )

        with self.assertRaisesRegex(ValueError, "circuit pause"):
            self.director.state(self.run_id)

    def test_replay_rejects_inconsistent_pause_trigger_evidence(self) -> None:
        first = self._report_failure(1).event
        first_failure_at = datetime.fromisoformat(first.payload["first_failure_at"])
        self.ledger.append(
            self.run_id,
            "RunPaused",
            {
                "code": "gateway_retry_circuit_open",
                "retry_class": "model_gateway",
                "generation": 0,
                "stage": "research",
                "breaker_epoch": first.payload["breaker_epoch"],
                "consecutive_failures": 2,
                "retry_limit": 6,
                "first_failure_at": first.payload["first_failure_at"],
                "last_failure_at": first.payload["last_failure_at"],
                "last_error_code": "gateway_unavailable",
                "suggested_action": "check_gateway_then_resume",
                "pause_trigger": "epoch_elapsed",
                "epoch_seconds": 30 * 60,
                "epoch_deadline_at": (
                    first_failure_at + timedelta(minutes=30)
                ).isoformat(),
                "proposed_retry_at": first.payload["last_failure_at"],
                "retry_delay_seconds": 0.0,
            },
            event_id=f"{self.run_id}:forged-trigger-evidence",
        )

        with self.assertRaisesRegex(ValueError, "trigger evidence"):
            self.director.state(self.run_id)

    def test_replay_rejects_payload_forged_elapsed_count_two_pause(self) -> None:
        first = self._report_failure(1).event
        first_created_at = datetime.fromisoformat(first.created_at)
        forged_payload_last = first_created_at + timedelta(minutes=30)
        self.ledger.append(
            self.run_id,
            "RunPaused",
            {
                "code": "gateway_retry_circuit_open",
                "retry_class": "model_gateway",
                "generation": 0,
                "stage": "research",
                "breaker_epoch": first.payload["breaker_epoch"],
                "consecutive_failures": 2,
                "retry_limit": 6,
                "first_failure_at": first_created_at.isoformat(),
                "last_failure_at": forged_payload_last.isoformat(),
                "last_error_code": "gateway_unavailable",
                "suggested_action": "check_gateway_then_resume",
                "pause_trigger": "epoch_elapsed",
                "epoch_seconds": 30 * 60,
                "epoch_deadline_at": forged_payload_last.isoformat(),
                "proposed_retry_at": forged_payload_last.isoformat(),
                "retry_delay_seconds": 0.0,
            },
            event_id=f"{self.run_id}:payload-forged-elapsed-pause",
            created_at=(first_created_at + timedelta(minutes=1)).isoformat(),
        )

        with self.assertRaisesRegex(ValueError, "trigger evidence"):
            self.director.state(self.run_id)

    def test_replay_rejects_payload_forged_deadline_count_two_pause(self) -> None:
        first = self._report_failure(1).event
        first_created_at = datetime.fromisoformat(first.created_at)
        pause_created_at = first_created_at + timedelta(minutes=1)
        epoch_deadline = first_created_at + timedelta(minutes=30)
        self.ledger.append(
            self.run_id,
            "RunPaused",
            {
                "code": "gateway_retry_circuit_open",
                "retry_class": "model_gateway",
                "generation": 0,
                "stage": "research",
                "breaker_epoch": first.payload["breaker_epoch"],
                "consecutive_failures": 2,
                "retry_limit": 6,
                "first_failure_at": first_created_at.isoformat(),
                "last_failure_at": pause_created_at.isoformat(),
                "last_error_code": "gateway_unavailable",
                "suggested_action": "check_gateway_then_resume",
                "pause_trigger": "retry_deadline_reaches_epoch",
                "epoch_seconds": 30 * 60,
                "epoch_deadline_at": epoch_deadline.isoformat(),
                "proposed_retry_at": epoch_deadline.isoformat(),
                "retry_delay_seconds": 0.0,
            },
            event_id=f"{self.run_id}:payload-forged-deadline-pause",
            created_at=pause_created_at.isoformat(),
        )

        with self.assertRaisesRegex(ValueError, "trigger evidence"):
            self.director.state(self.run_id)

    def test_replay_rejects_out_of_range_pause_retry_delay(self) -> None:
        first = self._report_failure(1).event
        first_created_at = datetime.fromisoformat(first.created_at)
        self.ledger.append(
            self.run_id,
            "RunPaused",
            {
                "code": "gateway_retry_circuit_open",
                "retry_class": "model_gateway",
                "generation": 0,
                "stage": "research",
                "breaker_epoch": first.payload["breaker_epoch"],
                "consecutive_failures": 2,
                "retry_limit": 6,
                "first_failure_at": first_created_at.isoformat(),
                "last_failure_at": first_created_at.isoformat(),
                "last_error_code": "gateway_unavailable",
                "suggested_action": "check_gateway_then_resume",
                "pause_trigger": "retry_deadline_reaches_epoch",
                "epoch_seconds": 30 * 60,
                "epoch_deadline_at": (
                    first_created_at + timedelta(minutes=30)
                ).isoformat(),
                "proposed_retry_at": (
                    first_created_at + timedelta(seconds=3600.001)
                ).isoformat(),
                "retry_delay_seconds": 3600.001,
            },
            event_id=f"{self.run_id}:out-of-range-pause-delay",
            created_at=first_created_at.isoformat(),
        )

        with self.assertRaisesRegex(ValueError, "circuit pause"):
            self.director.state(self.run_id)

    def test_director_maps_untrusted_error_codes_per_retry_class(self) -> None:
        secret = "sk_live_abc123credential"
        expected_codes = {
            "model_gateway": "gateway_response_error",
            "dsh_native_runtime": "dsh_native_runtime_unavailable",
            "research_timeout": "timeout",
            "sample_result_persistence": "sample_result_callback_error",
        }

        for retry_class, expected_code in expected_codes.items():
            state = self.director.state(self.run_id)
            decision = self.director.schedule_gateway_retry_or_pause(
                self.run_id,
                run_incarnation=state.events[0].seq,
                generation=0,
                stage="research",
                retry_class=retry_class,
                failure_id=digest({"untrusted-code-class": retry_class}),
                attempt_anchor_seq=state.events[-1].seq,
                delay_seconds=0.0,
                last_error_code=secret,
            )
            self.assertEqual(decision.outcome, "scheduled")
            self.assertEqual(decision.event.payload["last_error_code"], expected_code)
            self.assertNotIn(secret, str(decision.event.payload))

    def test_replay_rejects_unowned_retry_error_code(self) -> None:
        first = self._report_failure(1).event
        malformed = dict(first.payload)
        malformed.update(
            {
                "failure_id": digest("unowned-retry-error-code"),
                "attempt_anchor_seq": first.seq,
                "consecutive_failures": 2,
                "attempt": 2,
                "last_error_code": "sk_live_abc123credential",
                "error_code": "sk_live_abc123credential",
            }
        )
        self.ledger.append(
            self.run_id,
            "GatewayRetryScheduled",
            malformed,
            event_id=f"{self.run_id}:unowned-retry-error-code",
        )

        with self.assertRaisesRegex(ValueError, "payload is invalid"):
            self.director.state(self.run_id)

    def test_replay_rejects_extra_sensitive_fields_in_v2_retry_event(self) -> None:
        first = self._report_failure(1)
        malformed = dict(first.event.payload)
        malformed["failure_id"] = digest("malformed-retry-event")
        malformed["raw_provider_body"] = "Bearer production-secret"
        self.ledger.append(
            self.run_id,
            "GatewayRetryScheduled",
            malformed,
            event_id=f"{self.run_id}:malformed-gateway-retry",
        )

        with self.assertRaisesRegex(
            ValueError,
            "GatewayRetryScheduled v2 payload is invalid",
        ):
            self.director.state(self.run_id)

    def test_replay_rejects_malformed_circuit_pause_payload(self) -> None:
        malformed = {
            "code": "gateway_retry_circuit_open",
            "retry_class": "model_gateway",
            "generation": 0,
            "stage": "research",
            "breaker_epoch": 1,
            "consecutive_failures": 6,
            "retry_limit": 6,
            "first_failure_at": "2026-08-26T08:00:00+00:00",
            "last_failure_at": "2026-08-26T08:05:00+00:00",
            "last_error_code": "gateway_unavailable",
            "raw_error": "credential=production-secret",
        }
        self.ledger.append(
            self.run_id,
            "RunPaused",
            malformed,
            event_id=f"{self.run_id}:malformed-circuit-pause",
        )

        with self.assertRaisesRegex(
            ValueError,
            "gateway circuit pause payload is invalid",
        ):
            self.director.state(self.run_id)

    def test_replay_rejects_circuit_pause_before_run_is_running(self) -> None:
        created_run_id = "run:circuit-before-start"
        self.director.create_run(self.task, run_id=created_run_id)
        self.ledger.append(
            created_run_id,
            "RunPaused",
            {
                "code": "gateway_retry_circuit_open",
                "retry_class": "model_gateway",
                "generation": 0,
                "stage": "research",
                "breaker_epoch": 1,
                "consecutive_failures": 6,
                "retry_limit": 6,
                "first_failure_at": "2026-08-26T08:00:00+00:00",
                "last_failure_at": "2026-08-26T08:05:00+00:00",
                "last_error_code": "gateway_unavailable",
                "suggested_action": "check_gateway_then_resume",
                "pause_trigger": "failure_limit",
                "epoch_seconds": 30 * 60,
                "epoch_deadline_at": "2026-08-26T08:30:00+00:00",
                "proposed_retry_at": "2026-08-26T08:05:00+00:00",
                "retry_delay_seconds": 0.0,
            },
            event_id=f"{created_run_id}:forged-circuit-pause",
        )

        with self.assertRaisesRegex(
            ValueError,
            "gateway circuit pause requires a running current generation",
        ):
            self.director.state(created_run_id)

    def test_replay_rejects_wrong_incarnation_v2_retry_event(self) -> None:
        first = self._report_failure(1)
        malformed = dict(first.event.payload)
        malformed["run_incarnation"] += 1
        malformed["failure_id"] = digest("wrong-incarnation-retry")
        self.ledger.append(
            self.run_id,
            "GatewayRetryScheduled",
            malformed,
            event_id=f"{self.run_id}:wrong-incarnation-retry",
        )

        with self.assertRaisesRegex(
            ValueError,
            "GatewayRetryScheduled scope does not match the running run",
        ):
            self.director.state(self.run_id)

    def test_replay_rejects_run_started_after_active_circuit_pause(self) -> None:
        for attempt in range(1, 7):
            self._report_failure(attempt)
        state = self.director.state(self.run_id)
        self.ledger.append(
            self.run_id,
            "RunStarted",
            {"session_id": state.run.session_id},
            event_id=f"{self.run_id}:forged-start-after-circuit",
        )

        with self.assertRaisesRegex(
            ValueError,
            "gateway circuit requires an exact RunResumed origin",
        ):
            self.director.state(self.run_id)

    def test_replay_rejects_generic_resume_of_active_circuit_pause(self) -> None:
        for attempt in range(1, 7):
            self._report_failure(attempt)
        self.ledger.append(
            self.run_id,
            "RunResumed",
            {},
            event_id=f"{self.run_id}:forged-generic-resume",
        )

        with self.assertRaisesRegex(
            ValueError,
            "gateway circuit requires an exact RunResumed origin",
        ):
            self.director.state(self.run_id)

    def test_replay_rejects_manual_pause_masking_an_active_circuit(self) -> None:
        for attempt in range(1, 7):
            self._report_failure(attempt)
        state = self.director.state(self.run_id)
        self.ledger.append(
            self.run_id,
            "RunPaused",
            {"code": "operator_backpressure"},
            event_id=f"{self.run_id}:forged-manual-pause-mask",
        )
        self.ledger.append(
            self.run_id,
            "RunStarted",
            {"session_id": state.run.session_id},
            event_id=f"{self.run_id}:forged-start-after-mask",
        )

        with self.assertRaisesRegex(
            ValueError,
            "gateway circuit requires an exact RunResumed origin",
        ):
            self.director.state(self.run_id)

    def test_replay_rejects_v2_retry_count_that_skips_chain(self) -> None:
        first = self._report_failure(1).event
        malformed = dict(first.payload)
        malformed.update(
            {
                "failure_id": digest("skipped-retry-count"),
                "attempt_anchor_seq": first.seq,
                "consecutive_failures": 3,
                "attempt": 3,
            }
        )
        self.ledger.append(
            self.run_id,
            "GatewayRetryScheduled",
            malformed,
            event_id=f"{self.run_id}:skipped-retry-count",
        )

        with self.assertRaisesRegex(ValueError, "retry chain"):
            self.director.state(self.run_id)

    def test_replay_rejects_v2_retry_reusing_an_old_attempt_anchor(self) -> None:
        first = self._report_failure(1).event
        malformed = dict(first.payload)
        malformed.update(
            {
                "failure_id": digest("reused-old-attempt-anchor"),
                "consecutive_failures": 2,
                "attempt": 2,
            }
        )
        self.ledger.append(
            self.run_id,
            "GatewayRetryScheduled",
            malformed,
            event_id=f"{self.run_id}:reused-old-attempt-anchor",
        )

        with self.assertRaisesRegex(ValueError, "retry chain"):
            self.director.state(self.run_id)

    def test_replay_rejects_duplicate_failure_id_under_a_new_event_id(self) -> None:
        first = self._report_failure(1).event
        malformed = dict(first.payload)
        malformed.update(
            {
                "attempt_anchor_seq": first.seq,
                "consecutive_failures": 2,
                "attempt": 2,
            }
        )
        self.ledger.append(
            self.run_id,
            "GatewayRetryScheduled",
            malformed,
            event_id=f"{self.run_id}:duplicate-failure-new-event",
        )

        with self.assertRaisesRegex(ValueError, "failure_id"):
            self.director.state(self.run_id)

    def test_replay_rejects_circuit_pause_not_derived_from_retry_chain(self) -> None:
        first = self._report_failure(1).event
        self.ledger.append(
            self.run_id,
            "RunPaused",
            {
                "code": "gateway_retry_circuit_open",
                "retry_class": "model_gateway",
                "generation": 0,
                "stage": "research",
                "breaker_epoch": first.payload["breaker_epoch"],
                "consecutive_failures": 6,
                "retry_limit": 6,
                "first_failure_at": first.payload["first_failure_at"],
                "last_failure_at": first.payload["last_failure_at"],
                "last_error_code": "gateway_unavailable",
                "suggested_action": "check_gateway_then_resume",
                "pause_trigger": "failure_limit",
                "epoch_seconds": 30 * 60,
                "epoch_deadline_at": (
                    datetime.fromisoformat(first.payload["first_failure_at"])
                    + timedelta(minutes=30)
                ).isoformat(),
                "proposed_retry_at": first.payload["last_failure_at"],
                "retry_delay_seconds": 0.0,
            },
            event_id=f"{self.run_id}:forged-threshold-pause",
            created_at=first.payload["last_failure_at"],
        )

        with self.assertRaisesRegex(ValueError, "retry chain"):
            self.director.state(self.run_id)

    def test_legacy_retry_heartbeat_replays_but_does_not_inflate_new_epoch(self) -> None:
        legacy_run_id = "run:legacy-gateway-retry"
        self.director.start_evolution(self.task, run_id=legacy_run_id)
        self.ledger.append(
            legacy_run_id,
            "GatewayRetryScheduled",
            {
                "generation": 0,
                "retry_at": "2026-08-26T08:05:00+00:00",
                "attempt": 2_147_483_647,
            },
            event_id=f"{legacy_run_id}:legacy-retry",
        )
        state = self.director.state(legacy_run_id)

        decision = self.director.schedule_gateway_retry_or_pause(
            legacy_run_id,
            run_incarnation=state.events[0].seq,
            generation=0,
            stage="research",
            retry_class="model_gateway",
            failure_id=digest("first-v2-failure-after-legacy"),
            attempt_anchor_seq=state.events[-1].seq,
            delay_seconds=0.0,
            last_error_code="gateway_unavailable",
        )

        self.assertEqual(decision.outcome, "scheduled")
        self.assertEqual(decision.event.payload["consecutive_failures"], 1)
        self.assertEqual(decision.event.payload["breaker_epoch"], 1)

    def test_running_projection_clears_retry_details_after_stage_start(
        self,
    ) -> None:
        retry = self._report_failure(1).event

        projected = _projection_json(self.director.state(self.run_id))
        retry_wait = projected["execution_progress"]["retry_wait"]
        self.assertEqual(
            {
                name: retry_wait[name]
                for name in (
                    "retry_class",
                    "stage",
                    "breaker_epoch",
                    "consecutive_failures",
                    "retry_limit",
                    "first_failure_at",
                    "last_failure_at",
                    "last_error_code",
                    "suggested_action",
                )
            },
            {
                "retry_class": "model_gateway",
                "stage": "research",
                "breaker_epoch": 1,
                "consecutive_failures": 1,
                "retry_limit": 6,
                "first_failure_at": retry.payload["first_failure_at"],
                "last_failure_at": retry.payload["last_failure_at"],
                "last_error_code": "gateway_unavailable",
                "suggested_action": "wait_for_scheduled_retry",
            },
        )
        self.assertNotIn("failure_id", retry_wait)
        self.assertNotIn("attempt_anchor_seq", retry_wait)

        self.director.record_evolution_stage(
            self.run_id,
            generation=0,
            stage="research",
            status="started",
            attempt=2,
        )

        after_start = _projection_json(self.director.state(self.run_id))
        self.assertIsNone(after_start["execution_progress"].get("retry_wait"))
        self.assertNotEqual(
            after_start["execution_progress"]["phase"],
            "gateway_retry",
        )

    def test_paused_projection_and_events_expose_only_actionable_circuit_fields(
        self,
    ) -> None:
        for attempt in range(1, 7):
            decision = self._report_failure(attempt)

        projected = _projection_json(self.director.state(self.run_id))
        self.assertEqual(projected["pause_code"], "gateway_retry_circuit_open")
        self.assertIn("检查模型网关", projected["pause_reason"])
        self.assertEqual(
            projected["retry_circuit"],
            {
                "open": True,
                "code": "gateway_retry_circuit_open",
                "retry_class": "model_gateway",
                "generation": 0,
                "stage": "research",
                "breaker_epoch": 1,
                "consecutive_failures": 6,
                "retry_limit": 6,
                "first_failure_at": decision.event.payload["first_failure_at"],
                "last_failure_at": decision.event.payload["last_failure_at"],
                "last_error_code": "gateway_unavailable",
                "suggested_action": "check_gateway_then_resume",
                "pause_event_id": decision.event.event_id,
                "updated_at": decision.event.created_at,
            },
        )

        retry_event = next(
            event
            for event in self.director.state(self.run_id).events
            if event.kind == "GatewayRetryScheduled"
        )
        public_retry = EventEndpointsMixin._event_json(retry_event)["payload"]
        public_pause = EventEndpointsMixin._event_json(decision.event)["payload"]
        for name in (
            "retry_class",
            "stage",
            "breaker_epoch",
            "consecutive_failures",
            "retry_limit",
            "first_failure_at",
            "last_failure_at",
            "last_error_code",
        ):
            self.assertIn(name, public_retry)
            self.assertIn(name, public_pause)
        self.assertEqual(
            public_pause["suggested_action"],
            "check_gateway_then_resume",
        )
        for private_name in ("failure_id", "attempt_anchor_seq"):
            self.assertNotIn(private_name, public_retry)
            self.assertNotIn(private_name, public_pause)


if __name__ == "__main__":
    unittest.main()
