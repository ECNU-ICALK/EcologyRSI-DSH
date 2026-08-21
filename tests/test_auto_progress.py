from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecologyrsi_dsh.api import auto_progress as auto_progress_module
from ecologyrsi_dsh.api import generation_execution as generation_execution_module
from ecologyrsi_dsh.evaluators.sample_execution import SampleResultCallbackError
from ecologyrsi_dsh.evolution.batches import ResearchResponseContractError
from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
from ecologyrsi_dsh.model_gateway import GatewayResponseError
from ecologyrsi_dsh.server import EvolutionHTTPServer


class AutoProgressHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.server = EvolutionHTTPServer(
            ("127.0.0.1", 0), Path(self.directory.name) / "events.sqlite3"
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.directory.cleanup()

    def request(
        self, path: str, method: str = "GET", body: dict | None = None
    ) -> tuple[int, dict]:
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self.base + path,
            data=encoded,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_retry_classifier_requeues_retryable_gateway_calls(self) -> None:
        self.assertTrue(
            auto_progress_module._progress_failure_retryable(
                GatewayResponseError(
                    "HTTP 503 after request-local retries", retryable=True, attempts=4
                )
            )
        )
        self.assertFalse(
            auto_progress_module._progress_failure_retryable(
                GatewayResponseError("invalid JSON contract", retryable=False)
            )
        )
        self.assertFalse(
            auto_progress_module._progress_failure_retryable(
                ValueError("frozen binding is invalid")
            )
        )
        research_contract = ResearchResponseContractError(
            "research response failed host contract validation"
        )
        self.assertFalse(
            auto_progress_module._progress_failure_retryable(research_contract)
        )
        self.assertIsNone(
            auto_progress_module._retry_later_error(research_contract)
        )
        self.assertTrue(
            auto_progress_module._progress_failure_retryable(
                research_contract,
                stage="research",
            )
        )
        self.assertIs(
            auto_progress_module._retry_later_error(
                research_contract,
                stage="research",
            ),
            research_contract,
        )
        self.assertTrue(
            auto_progress_module._progress_failure_retryable(
                TimeoutError("transient ledger boundary")
            )
        )

        truncated = GatewayResponseError(
            "research response reached its output limit",
            retryable=False,
            error_code="output_truncated",
            split_eligible=True,
            finish_reason="length",
        )
        self.assertFalse(
            auto_progress_module._progress_failure_retryable(truncated)
        )
        self.assertTrue(
            auto_progress_module._progress_failure_retryable(
                truncated,
                stage="research",
            )
        )
        self.assertIs(
            auto_progress_module._retry_later_error(
                truncated,
                stage="research",
            ),
            truncated,
        )

        malformed_research = GatewayResponseError(
            "research response violated its semantic contract",
            retryable=False,
        )
        self.assertTrue(
            auto_progress_module._progress_failure_retryable(
                malformed_research,
                stage="research",
            )
        )
        research_timeout = TimeoutError("research request timed out")
        self.assertIs(
            auto_progress_module._retry_later_error(
                research_timeout,
                stage="research",
            ),
            research_timeout,
        )

        invalid_credentials = GatewayResponseError(
            "HTTP 401",
            retryable=False,
            status_code=401,
        )
        self.assertFalse(
            auto_progress_module._progress_failure_retryable(
                invalid_credentials,
                stage="research",
            )
        )
        self.assertIsNone(
            auto_progress_module._retry_later_error(
                invalid_credentials,
                stage="research",
            )
        )

        transport_failure = DshNativeRuntimeUnavailableError(
            "DSH connection was interrupted during maintenance",
            error_code="dsh_native_runtime_transport_error",
        )
        self.assertTrue(
            auto_progress_module._progress_failure_retryable(transport_failure)
        )
        self.assertIs(
            auto_progress_module._retry_later_error(transport_failure),
            transport_failure,
        )

        dsh_stage_timeout = DshNativeRuntimeUnavailableError(
            "DSH returned its bounded stage timeout",
            error_code="runtime_controller_failed",
            status_code=502,
        )
        self.assertTrue(
            auto_progress_module._progress_failure_retryable(dsh_stage_timeout)
        )
        self.assertIs(
            auto_progress_module._retry_later_error(dsh_stage_timeout),
            dsh_stage_timeout,
        )

        contract_failure = DshNativeRuntimeUnavailableError(
            "DSH response violated its frozen schema",
            error_code="dsh_native_runtime_contract_error",
        )
        self.assertFalse(
            auto_progress_module._progress_failure_retryable(contract_failure)
        )
        self.assertIsNone(
            auto_progress_module._retry_later_error(contract_failure)
        )

    def test_gateway_retry_delay_saturates_before_large_exponent(self) -> None:
        work_item = ("run:large-retry-attempt", 1)
        with patch.object(
            self.server.auto_progress,
            "_retry_attempt",
            return_value=1025,
        ):
            delay = self.server.auto_progress._gateway_retry_delay(work_item, 1)

        self.assertEqual(delay, auto_progress_module._GATEWAY_RETRY_MAX_SECONDS)

        with (
            patch.object(
                self.server.auto_progress,
                "_retry_attempt",
                return_value=1025,
            ),
            patch.object(auto_progress_module, "_GATEWAY_RETRY_BASE_SECONDS", 0.0),
        ):
            zero_delay = self.server.auto_progress._gateway_retry_delay(work_item, 1)

        self.assertEqual(zero_delay, 0.0)

    def test_retry_recovery_ignores_later_stage_from_other_generation(self) -> None:
        retry_at = datetime.now(timezone.utc) + timedelta(seconds=60)
        state = SimpleNamespace(
            run=SimpleNamespace(generation=2),
            events=(
                SimpleNamespace(
                    kind="GatewayRetryScheduled",
                    seq=10,
                    payload={"generation": 2, "retry_at": retry_at.isoformat()},
                ),
                SimpleNamespace(
                    kind="EvolutionStageRecorded",
                    seq=11,
                    payload={"generation": 1, "stage": "decision"},
                ),
            ),
        )
        work_item = ("run:retry-generation-scope", 1)

        self.server.auto_progress._restore_retry_deadline(work_item, state)

        self.assertIn(work_item, self.server.auto_progress._retry_not_before)

    def test_exhausted_gateway_request_keeps_run_running_for_delayed_retry(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "gateway-request-not-generation-retry",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)

        exhausted = GatewayResponseError(
            "HTTP 503 after request-local retries; Bearer generation-audit-secret",
            retryable=True,
            attempts=4,
            status_code=503,
            retry_after_seconds=900,
        )
        with patch.object(
            auto_progress_module,
            "execute_generation",
            side_effect=exhausted,
        ) as execute_generation:
            with patch.object(
                auto_progress_module,
                "_GATEWAY_RETRY_BASE_SECONDS",
                0.0,
            ):
                keep_running = self.server.auto_progress._run_one_generation(run_id)

        self.assertTrue(keep_running)
        self.assertEqual(execute_generation.call_count, 1)
        state = self.server.director.state(run_id)
        self.assertEqual(state.run.status.value, "running")
        self.assertIsNotNone(self.server.auto_progress._retry_not_before)
        self.assertFalse(any(event.kind == "RunFailed" for event in state.events))
        retry_event = next(
            event
            for event in reversed(state.events)
            if event.kind == "GatewayRetryScheduled"
        )
        self.assertEqual(retry_event.payload["delay_seconds"], 900.0)

    def test_transient_dsh_outage_keeps_run_running_for_delayed_retry(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "dsh-outage-delayed-retry",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)

        outage = DshNativeRuntimeUnavailableError(
            "DSH was restarted during a stage call",
            error_code="dsh_native_runtime_transport_error",
        )
        with (
            patch.object(
                auto_progress_module,
                "execute_generation",
                side_effect=outage,
            ) as execute_generation,
            patch.object(auto_progress_module, "_GATEWAY_RETRY_BASE_SECONDS", 0.0),
        ):
            keep_running = self.server.auto_progress._run_one_generation(run_id)

        self.assertTrue(keep_running)
        self.assertEqual(execute_generation.call_count, 1)
        state = self.server.director.state(run_id)
        self.assertEqual(state.run.status.value, "running")
        self.assertFalse(any(event.kind == "RunFailed" for event in state.events))
        retry_event = next(
            event
            for event in reversed(state.events)
            if event.kind == "GatewayRetryScheduled"
        )
        self.assertEqual(
            retry_event.payload["error_code"],
            "dsh_native_runtime_transport_error",
        )
        self.assertIn("DSH 智能体运行时暂时不可用", retry_event.payload["reason"])

    def test_dsh_proposal_timeout_is_not_converted_to_terminal_batch_failure(
        self,
    ) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "dsh-proposal-timeout-delayed-retry",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)

        timeout = DshNativeRuntimeUnavailableError(
            "DSH structured proposal reached its operational timeout",
            error_code="runtime_controller_failed",
            status_code=502,
        )
        with (
            patch.object(
                self.server.director,
                "request_proposal",
                side_effect=timeout,
            ),
            patch.object(auto_progress_module, "_GATEWAY_RETRY_BASE_SECONDS", 0.0),
        ):
            keep_running = self.server.auto_progress._run_one_generation(run_id)

        self.assertTrue(keep_running)
        state = self.server.director.state(run_id)
        self.assertEqual(state.run.status.value, "running")
        self.assertFalse(any(event.kind == "RunFailed" for event in state.events))
        self.assertTrue(
            any(event.kind == "GatewayRetryScheduled" for event in state.events)
        )

    def test_research_contract_failure_keeps_run_running_for_delayed_retry(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "research-contract-delayed-retry",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)

        contract_error = GatewayResponseError(
            "research algorithm contract invalid after semantic retries",
            retryable=True,
            attempts=2,
            error_code="research_algorithm_contract_invalid",
        )
        with (
            patch.object(
                auto_progress_module,
                "execute_generation",
                side_effect=contract_error,
            ) as execute_generation,
            patch.object(auto_progress_module, "_GATEWAY_RETRY_BASE_SECONDS", 0.0),
        ):
            keep_running = self.server.auto_progress._run_one_generation(run_id)

        self.assertTrue(keep_running)
        self.assertEqual(execute_generation.call_count, 1)
        state = self.server.director.state(run_id)
        self.assertEqual(state.run.status.value, "running")
        retry_events = [
            event for event in state.events if event.kind == "GatewayRetryScheduled"
        ]
        self.assertTrue(retry_events)
        self.assertEqual(
            retry_events[-1].payload["error_code"],
            "research_algorithm_contract_invalid",
        )
        self.assertFalse(any(event.kind == "RunFailed" for event in state.events))

    def test_research_response_failures_persist_cooldown_without_failing_run(
        self,
    ) -> None:
        cases = (
            (
                "output-truncated",
                GatewayResponseError(
                    "research response ended at max_tokens",
                    retryable=False,
                    error_code="output_truncated",
                    split_eligible=True,
                    finish_reason="length",
                ),
                "output_truncated",
            ),
            (
                "gateway-response",
                GatewayResponseError(
                    "research response violated a bounded contract",
                    retryable=False,
                ),
                "gateway_response_error",
            ),
            (
                "timeout",
                TimeoutError("research request timed out after queueing"),
                "timeout",
            ),
        )
        for label, research_error, expected_code in cases:
            with self.subTest(label=label):
                status, created = self.request(
                    "/runs",
                    "POST",
                    {
                        "domain_pack_id": "crop_soil_water",
                        "dataset_id": "generated-toy-series@1",
                        "rounds": 1,
                        "candidates_per_generation": 1,
                        "max_candidates": 1,
                        "auto_progress": True,
                        "auto_advance": 0,
                        "start": False,
                        "idempotency_key": f"research-cooldown-{label}",
                    },
                )
                self.assertEqual(status, 201, created)
                run_id = created["projection"]["run_id"]
                with self.server.mutation_lock:
                    self.server.director.start_run(run_id)

                def fail_research(_endpoint: object, target_run_id: str) -> object:
                    state = self.server.director.state(target_run_id)
                    generation = int(state.run.generation)
                    self.server.director.record_evolution_stage(
                        target_run_id,
                        generation=generation,
                        stage="research",
                        status="started",
                        attempt=1,
                    )
                    self.server.director.record_evolution_stage(
                        target_run_id,
                        generation=generation,
                        stage="research",
                        status="failed",
                        attempt=1,
                        public_error="研究请求暂时失败。",
                    )
                    raise research_error

                with (
                    patch.object(
                        auto_progress_module,
                        "execute_generation",
                        side_effect=fail_research,
                    ) as execute_generation,
                    patch.object(
                        auto_progress_module,
                        "_GATEWAY_RETRY_BASE_SECONDS",
                        30.0,
                    ),
                ):
                    keep_running = self.server.auto_progress._run_one_generation(
                        run_id
                    )

                self.assertTrue(keep_running)
                self.assertEqual(execute_generation.call_count, 1)
                state = self.server.director.state(run_id)
                self.assertEqual(state.run.status.value, "running")
                self.assertFalse(
                    any(event.kind == "RunFailed" for event in state.events)
                )
                retry_event = next(
                    event
                    for event in reversed(state.events)
                    if event.kind == "GatewayRetryScheduled"
                )
                self.assertEqual(retry_event.payload["stage"], "research")
                self.assertEqual(
                    retry_event.payload["error_code"],
                    expected_code,
                )
                self.assertEqual(retry_event.payload["delay_seconds"], 30.0)

                work_item = (
                    run_id,
                    auto_progress_module._run_incarnation(state),
                )
                with self.server.auto_progress._state_lock:
                    self.server.auto_progress._retry_not_before.pop(
                        work_item,
                        None,
                    )
                self.server.auto_progress._restore_retry_deadline(
                    work_item,
                    state,
                )
                self.assertIn(
                    work_item,
                    self.server.auto_progress._retry_not_before,
                )

    def test_wrapped_gateway_error_keeps_run_running_for_delayed_retry(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "wrapped-gateway-retry",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)

        def wrapped_failure(*_args: object, **_kwargs: object) -> object:
            try:
                raise GatewayResponseError(
                    "provider queue is still busy",
                    retryable=True,
                    status_code=503,
                    attempts=4,
                )
            except GatewayResponseError as cause:
                raise RuntimeError("sample adapter wrapped gateway failure") from cause

        with (
            patch.object(auto_progress_module, "execute_generation", side_effect=wrapped_failure),
            patch.object(auto_progress_module, "_GATEWAY_RETRY_BASE_SECONDS", 0.0),
        ):
            self.assertTrue(self.server.auto_progress._run_one_generation(run_id))

        state = self.server.director.state(run_id)
        self.assertEqual(state.run.status.value, "running")
        self.assertTrue(any(event.kind == "GatewayRetryScheduled" for event in state.events))
        self.assertFalse(any(event.kind == "RunFailed" for event in state.events))

    def test_sample_result_callback_failure_is_deferred(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "sample-result-callback-retry",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)

        with (
            patch.object(
                auto_progress_module,
                "execute_generation",
                side_effect=SampleResultCallbackError("ledger temporarily busy"),
            ),
            patch.object(auto_progress_module, "_GATEWAY_RETRY_BASE_SECONDS", 0.0),
        ):
            self.assertTrue(self.server.auto_progress._run_one_generation(run_id))

        state = self.server.director.state(run_id)
        self.assertEqual(state.run.status.value, "running")
        self.assertTrue(any(event.kind == "GatewayRetryScheduled" for event in state.events))
        self.assertFalse(any(event.kind == "RunFailed" for event in state.events))

    def test_final_budget_candidate_checkpoint_is_not_completed_before_retry(
        self,
    ) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 2,
                "candidates_per_generation": 1,
                "max_candidates": 2,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "final-budget-candidate-callback-retry",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        endpoint = SimpleNamespace(server=self.server)
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)

        first_generation = generation_execution_module.execute_generation(
            endpoint,
            run_id,
        )
        self.assertEqual(first_generation.run.generation, 1)
        self.assertEqual(first_generation.run.status.value, "running")

        batch = generation_execution_module.start_generation_batch(
            self.server.director,
            run_id,
        )
        spawned = generation_execution_module._spawn_generation_candidates(
            endpoint,
            run_id,
            batch,
        )
        self.assertTrue(spawned)
        checkpoint = self.server.director.state(run_id)
        self.assertEqual(len(checkpoint.candidates), 2)
        self.assertEqual(checkpoint.candidates[-1].generation, 1)
        self.assertEqual(checkpoint.candidates[-1].status.value, "spawned")

        with (
            patch.object(
                auto_progress_module,
                "execute_generation",
                side_effect=SampleResultCallbackError("ledger temporarily busy"),
            ) as execute_generation,
            patch.object(auto_progress_module, "_GATEWAY_RETRY_BASE_SECONDS", 0.0),
        ):
            self.assertTrue(self.server.auto_progress._run_one_generation(run_id))

        execute_generation.assert_called_once()
        recovered = self.server.director.state(run_id)
        self.assertEqual(recovered.run.status.value, "running")
        self.assertFalse(any(event.kind == "RunCompleted" for event in recovered.events))
        self.assertTrue(
            any(event.kind == "GatewayRetryScheduled" for event in recovered.events)
        )

    def test_preflight_failure_write_is_requeued_without_reexecuting(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "preflight-failure-ledger-requeue",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)

        with (
            patch.object(
                self.server,
                "validate_frozen_runtime_bindings",
                side_effect=ValueError(
                    "invalid frozen binding; Bearer preflight-audit-secret"
                ),
            ),
            patch.object(
                self.server.director,
                "fail_run",
                side_effect=sqlite3.OperationalError("database is temporarily busy"),
            ),
            patch.object(auto_progress_module, "execute_generation") as execute_generation,
            patch.object(
                auto_progress_module,
                "_FAILURE_PERSISTENCE_RETRY_SECONDS",
                0.0,
            ),
        ):
            keep_running = self.server.auto_progress._run_one_generation(run_id)

        self.assertTrue(keep_running)
        self.assertEqual(
            self.server.director.state(run_id).run.status.value, "running"
        )
        execute_generation.assert_not_called()

        with patch.object(
            auto_progress_module, "execute_generation"
        ) as execute_generation:
            keep_running = self.server.auto_progress._run_one_generation(run_id)

        self.assertFalse(keep_running)
        state = self.server.director.state(run_id)
        self.assertEqual(state.run.status.value, "failed")
        failure = next(
            event for event in reversed(state.events) if event.kind == "RunFailed"
        )
        self.assertIn("ValueError", failure.payload["reason"])
        self.assertNotIn("preflight-audit-secret", failure.payload["reason"])
        execute_generation.assert_not_called()

    def test_generation_failure_write_is_requeued_without_replaying_generation(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "generation-failure-ledger-requeue",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)
        self.server.auto_progress._retry_limit = 1

        with (
            patch.object(
                auto_progress_module,
                "execute_generation",
                side_effect=RuntimeError("generation execution failed"),
            ) as execute_generation,
            patch.object(
                self.server.director,
                "fail_run",
                side_effect=sqlite3.OperationalError("database is temporarily busy"),
            ),
            patch.object(
                auto_progress_module,
                "_FAILURE_PERSISTENCE_RETRY_SECONDS",
                0.0,
            ),
        ):
            keep_running = self.server.auto_progress._run_one_generation(run_id)

        self.assertTrue(keep_running)
        self.assertEqual(execute_generation.call_count, 1)
        self.assertEqual(
            self.server.director.state(run_id).run.status.value, "running"
        )

        with patch.object(
            auto_progress_module, "execute_generation"
        ) as execute_generation:
            keep_running = self.server.auto_progress._run_one_generation(run_id)

        self.assertFalse(keep_running)
        self.assertEqual(self.server.director.state(run_id).run.status.value, "failed")
        execute_generation.assert_not_called()

    def test_worker_requeues_until_deferred_failure_is_persisted(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "worker-deferred-failure-requeue",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        persisted_fail_run = self.server.director.fail_run
        failure_write_calls = 0

        def flaky_fail_run(target_run_id: str, reason: str):
            nonlocal failure_write_calls
            failure_write_calls += 1
            if failure_write_calls == 1:
                raise sqlite3.OperationalError("database is temporarily busy")
            return persisted_fail_run(target_run_id, reason)

        with (
            patch.object(
                auto_progress_module,
                "execute_generation",
                side_effect=ValueError("terminal generation failure"),
            ) as execute_generation,
            patch.object(
                self.server.director,
                "fail_run",
                side_effect=flaky_fail_run,
            ),
            patch.object(
                auto_progress_module,
                "_FAILURE_PERSISTENCE_RETRY_SECONDS",
                0.0,
            ),
        ):
            with self.server.mutation_lock:
                self.server.director.start_run(run_id)
                self.server.auto_progress.schedule(run_id)

            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if self.server.director.state(run_id).run.status.value == "failed":
                    break
                time.sleep(0.01)

        state = self.server.director.state(run_id)
        self.assertEqual(state.run.status.value, "failed")
        self.assertEqual(execute_generation.call_count, 1)
        self.assertEqual(failure_write_calls, 2)
        self.assertIsNone(self.server.auto_progress._deferred_failure(run_id))
        self.assertEqual(
            [event.kind for event in state.events].count("RunFailed"),
            1,
        )

    def test_worker_survives_unexpected_state_error_and_processes_next_run(self) -> None:
        run_ids: list[str] = []
        for label in ("faulty", "next"):
            status, created = self.request(
                "/runs",
                "POST",
                {
                    "domain_pack_id": "crop_soil_water",
                    "dataset_id": "generated-toy-series@1",
                    "rounds": 1,
                    "candidates_per_generation": 1,
                    "max_candidates": 1,
                    "auto_progress": True,
                    "auto_advance": 0,
                    "start": False,
                    "idempotency_key": f"worker-state-error-{label}",
                },
            )
            self.assertEqual(status, 201, created)
            run_ids.append(created["projection"]["run_id"])

        faulty_run_id, next_run_id = run_ids
        original_state = self.server.auto_progress._state_for_work_item
        state_reads = 0

        def flaky_state(work_item: tuple[str, int]):
            nonlocal state_reads
            if work_item[0] == faulty_run_id and state_reads == 0:
                state_reads += 1
                raise sqlite3.OperationalError(
                    "database busy; Bearer worker-exception-secret"
                )
            return original_state(work_item)

        with (
            patch.object(
                self.server.auto_progress,
                "_state_for_work_item",
                side_effect=flaky_state,
            ),
            patch.object(
                auto_progress_module,
                "_FAILURE_PERSISTENCE_RETRY_SECONDS",
                0.0,
            ),
        ):
            with self.server.mutation_lock:
                for run_id in run_ids:
                    self.server.director.start_run(run_id)
                    self.assertTrue(self.server.auto_progress.schedule(run_id))

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                statuses = {
                    run_id: self.server.director.state(run_id).run.status.value
                    for run_id in run_ids
                }
                if statuses == {
                    faulty_run_id: "failed",
                    next_run_id: "completed",
                } and self.server.auto_progress._queue.unfinished_tasks == 0:
                    break
                time.sleep(0.01)

        faulty_state = self.server.director.state(faulty_run_id)
        self.assertEqual(faulty_state.run.status.value, "failed")
        self.assertEqual(
            self.server.director.state(next_run_id).run.status.value,
            "completed",
        )
        failure = next(
            event
            for event in reversed(faulty_state.events)
            if event.kind == "RunFailed"
        )
        self.assertIn("OperationalError", failure.payload["reason"])
        self.assertNotIn("worker-exception-secret", failure.payload["reason"])
        self.assertTrue(self.server.auto_progress._thread.is_alive())
        self.assertEqual(self.server.auto_progress._queue.unfinished_tasks, 0)
        with self.server.auto_progress._state_lock:
            for work_set in (
                self.server.auto_progress._scheduled,
                self.server.auto_progress._running,
                self.server.auto_progress._reschedule_requested,
            ):
                self.assertFalse(any(item[0] in run_ids for item in work_set))

    def test_worker_count_defaults_and_stays_bounded(self) -> None:
        with patch.dict(auto_progress_module.os.environ, {}, clear=True):
            self.assertEqual(
                auto_progress_module.AutoProgressManager._read_worker_count(), 1
            )
        with patch.dict(
            auto_progress_module.os.environ,
            {"ECOLOGYRSI_AUTO_PROGRESS_WORKERS": "2"},
            clear=True,
        ):
            self.assertEqual(
                auto_progress_module.AutoProgressManager._read_worker_count(), 2
            )
        with patch.dict(
            auto_progress_module.os.environ,
            {"ECOLOGYRSI_AUTO_PROGRESS_WORKERS": "999"},
            clear=True,
        ):
            self.assertEqual(
                auto_progress_module.AutoProgressManager._read_worker_count(), 8
            )
        with patch.dict(
            auto_progress_module.os.environ,
            {"ECOLOGYRSI_AUTO_PROGRESS_WORKERS": "0"},
            clear=True,
        ):
            self.assertEqual(
                auto_progress_module.AutoProgressManager._read_worker_count(), 1
            )
        with patch.dict(
            auto_progress_module.os.environ,
            {"ECOLOGYRSI_AUTO_PROGRESS_WORKERS": "invalid"},
            clear=True,
        ):
            self.assertEqual(
                auto_progress_module.AutoProgressManager._read_worker_count(), 1
            )

    def test_worker_pool_allows_other_run_while_one_generation_blocks(self) -> None:
        self.server.auto_progress.close()
        with patch.dict(
            auto_progress_module.os.environ,
            {"ECOLOGYRSI_AUTO_PROGRESS_WORKERS": "2"},
            clear=False,
        ):
            self.server.auto_progress = auto_progress_module.AutoProgressManager(
                self.server
            )
        self.assertEqual(self.server.auto_progress._worker_count, 2)

        run_ids: list[str] = []
        for label in ("slow", "fast"):
            status, created = self.request(
                "/runs",
                "POST",
                {
                    "domain_pack_id": "crop_soil_water",
                    "dataset_id": "generated-toy-series@1",
                    "rounds": 1,
                    "candidates_per_generation": 1,
                    "max_candidates": 1,
                    "auto_progress": True,
                    "auto_advance": 0,
                    "start": False,
                    "idempotency_key": f"worker-pool-{label}",
                },
            )
            self.assertEqual(status, 201, created)
            run_ids.append(created["projection"]["run_id"])

        slow_run_id, fast_run_id = run_ids
        slow_started = threading.Event()
        release_slow = threading.Event()
        fast_completed = threading.Event()
        execute_generation = auto_progress_module.execute_generation

        def controlled_execute(endpoint: object, target_run_id: str) -> object:
            if target_run_id == slow_run_id:
                slow_started.set()
                release_slow.wait(timeout=3)
            else:
                fast_completed.set()
            return execute_generation(endpoint, target_run_id)

        with patch.object(
            auto_progress_module,
            "execute_generation",
            side_effect=controlled_execute,
        ):
            with self.server.mutation_lock:
                for run_id in run_ids:
                    self.server.director.start_run(run_id)
                    self.server.auto_progress.schedule(run_id)

            self.assertTrue(slow_started.wait(timeout=1))
            # Active work retains the same queue lease, so duplicate schedule
            # requests cannot dispatch a second generation concurrently.
            self.assertFalse(self.server.auto_progress.schedule(slow_run_id))
            self.assertTrue(fast_completed.wait(timeout=2))

            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if (
                    self.server.director.state(fast_run_id).run.status.value
                    == "completed"
                ):
                    break
                time.sleep(0.01)

            self.assertEqual(
                self.server.director.state(fast_run_id).run.status.value,
                "completed",
            )
            release_slow.set()

            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if (
                    self.server.director.state(slow_run_id).run.status.value
                    == "completed"
                ):
                    break
                time.sleep(0.01)

        self.assertEqual(
            self.server.director.state(slow_run_id).run.status.value, "completed"
        )

    def test_gateway_cooldown_releases_worker_for_other_runs(self) -> None:
        run_ids: list[str] = []
        for label in ("cooldown", "independent"):
            status, created = self.request(
                "/runs",
                "POST",
                {
                    "domain_pack_id": "crop_soil_water",
                    "dataset_id": "generated-toy-series@1",
                    "rounds": 1,
                    "candidates_per_generation": 1,
                    "max_candidates": 1,
                    "auto_progress": True,
                    "auto_advance": 0,
                    "start": False,
                    "idempotency_key": f"cooldown-worker-{label}",
                },
            )
            self.assertEqual(status, 201, created)
            run_ids.append(created["projection"]["run_id"])
        cooldown_run_id, independent_run_id = run_ids
        first_attempt = threading.Event()
        original_execute = auto_progress_module.execute_generation

        def controlled_execute(endpoint: object, target_run_id: str) -> object:
            if target_run_id == cooldown_run_id:
                first_attempt.set()
                raise GatewayResponseError(
                    "provider queue is busy",
                    retryable=True,
                    status_code=503,
                    attempts=4,
                )
            return original_execute(endpoint, target_run_id)

        with (
            patch.object(auto_progress_module, "execute_generation", side_effect=controlled_execute),
            patch.object(auto_progress_module, "_GATEWAY_RETRY_BASE_SECONDS", 30.0),
        ):
            with self.server.mutation_lock:
                for run_id in run_ids:
                    self.server.director.start_run(run_id)
                    self.server.auto_progress.schedule(run_id)
            self.assertTrue(first_attempt.wait(timeout=2))
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if self.server.director.state(independent_run_id).run.status.value == "completed":
                    break
                time.sleep(0.01)
            self.assertEqual(
                self.server.director.state(independent_run_id).run.status.value,
                "completed",
            )
            self.assertEqual(
                self.server.director.state(cooldown_run_id).run.status.value,
                "running",
            )

    def test_expired_cooldown_is_removed_before_terminal_dispatch(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "expired-cooldown-terminal-dispatch",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)
        state = self.server.director.state(run_id)
        work_item = (run_id, auto_progress_module._run_incarnation(state))
        with self.server.auto_progress._state_lock:
            self.server.auto_progress._retry_not_before[work_item] = (
                time.monotonic() - 1.0
            )

        self.assertTrue(self.server.auto_progress.schedule(run_id))
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if (
                self.server.director.state(run_id).run.status.value == "completed"
                and self.server.auto_progress._queue.unfinished_tasks == 0
            ):
                break
            time.sleep(0.01)

        self.assertEqual(
            self.server.director.state(run_id).run.status.value,
            "completed",
        )
        self.assertEqual(self.server.auto_progress._queue.unfinished_tasks, 0)
        with self.server.auto_progress._state_lock:
            self.assertNotIn(
                work_item,
                self.server.auto_progress._retry_not_before,
            )
        diagnostics = self.server.auto_progress.diagnostics(run_id)
        self.assertEqual(diagnostics["run_state"], "idle")
        self.assertEqual(diagnostics["cooldown_run_count"], 0)

    def test_diagnostics_clears_cooldown_for_every_terminal_status(self) -> None:
        transitions = (
            ("completed", lambda run_id: self.server.director.complete_run(run_id)),
            ("cancelled", lambda run_id: self.server.director.cancel_run(run_id)),
            ("failed", lambda run_id: self.server.director.fail_run(run_id, "test")),
        )
        for label, transition in transitions:
            with self.subTest(status=label):
                status, created = self.request(
                    "/runs",
                    "POST",
                    {
                        "domain_pack_id": "crop_soil_water",
                        "dataset_id": "generated-toy-series@1",
                        "rounds": 1,
                        "candidates_per_generation": 1,
                        "max_candidates": 1,
                        "auto_progress": True,
                        "auto_advance": 0,
                        "start": False,
                        "idempotency_key": f"terminal-cooldown-{label}",
                    },
                )
                self.assertEqual(status, 201, created)
                run_id = created["projection"]["run_id"]
                with self.server.mutation_lock:
                    self.server.director.start_run(run_id)
                state = self.server.director.state(run_id)
                work_item = (run_id, auto_progress_module._run_incarnation(state))
                timer = threading.Timer(60.0, lambda: None)
                timer.daemon = True
                timer.start()
                with self.server.auto_progress._state_lock:
                    self.server.auto_progress._retry_not_before[work_item] = (
                        time.monotonic() + 60.0
                    )
                    self.server.auto_progress._retry_timers[work_item] = timer

                with self.server.mutation_lock:
                    transition(run_id)

                diagnostics = self.server.auto_progress.diagnostics(run_id)
                self.assertEqual(diagnostics["run_state"], "idle")
                self.assertEqual(diagnostics["cooldown_run_count"], 0)
                with self.server.auto_progress._state_lock:
                    self.assertNotIn(
                        work_item,
                        self.server.auto_progress._retry_not_before,
                    )
                    self.assertNotIn(
                        work_item,
                        self.server.auto_progress._retry_timers,
                    )

    def test_future_cooldown_is_preserved_while_running_or_paused(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "future-cooldown-preserved",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)
        state = self.server.director.state(run_id)
        work_item = (run_id, auto_progress_module._run_incarnation(state))
        timer = threading.Timer(60.0, lambda: None)
        timer.daemon = True
        timer.start()
        with self.server.auto_progress._state_lock:
            self.server.auto_progress._retry_not_before[work_item] = (
                time.monotonic() + 60.0
            )
            self.server.auto_progress._retry_timers[work_item] = timer

        diagnostics = self.server.auto_progress.diagnostics(run_id)
        self.assertEqual(diagnostics["run_state"], "cooldown")
        self.assertEqual(diagnostics["cooldown_run_count"], 1)
        with self.server.mutation_lock:
            self.server.director.pause_run(run_id)
        paused_diagnostics = self.server.auto_progress.diagnostics(run_id)
        self.assertEqual(paused_diagnostics["run_state"], "cooldown")
        with self.server.auto_progress._state_lock:
            self.assertIn(work_item, self.server.auto_progress._retry_not_before)
            self.assertIs(
                self.server.auto_progress._retry_timers.get(work_item),
                timer,
            )

    def test_dead_retry_timer_is_replaced_for_running_run(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "dead-cooldown-timer-replaced",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)
        state = self.server.director.state(run_id)
        work_item = (run_id, auto_progress_module._run_incarnation(state))
        dead_timer = threading.Timer(0.0, lambda: None)
        dead_timer.start()
        dead_timer.join(timeout=1)
        self.assertFalse(dead_timer.is_alive())
        with self.server.auto_progress._state_lock:
            self.server.auto_progress._retry_not_before[work_item] = (
                time.monotonic() + 60.0
            )
            self.server.auto_progress._retry_timers[work_item] = dead_timer

        diagnostics = self.server.auto_progress.diagnostics(run_id)
        self.assertEqual(diagnostics["run_state"], "cooldown")
        deadline = time.monotonic() + 2.0
        replacement = None
        while time.monotonic() < deadline:
            with self.server.auto_progress._state_lock:
                replacement = self.server.auto_progress._retry_timers.get(work_item)
            if replacement is not None and replacement is not dead_timer:
                break
            time.sleep(0.01)

        self.assertIsNotNone(replacement)
        self.assertIsNot(replacement, dead_timer)
        self.assertTrue(replacement.is_alive())
        with self.server.auto_progress._state_lock:
            self.assertIn(work_item, self.server.auto_progress._retry_not_before)

    def test_timer_start_failure_does_not_orphan_run(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "timer-start-failure-fallback",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)
        state = self.server.director.state(run_id)
        work_item = (run_id, auto_progress_module._run_incarnation(state))
        with self.server.auto_progress._state_lock:
            self.server.auto_progress._retry_not_before[work_item] = (
                time.monotonic() + 60.0
            )
        executed = threading.Event()
        execute_generation = auto_progress_module.execute_generation

        def controlled_execute(endpoint: object, target_run_id: str) -> object:
            executed.set()
            return execute_generation(endpoint, target_run_id)

        with (
            patch.object(
                auto_progress_module,
                "execute_generation",
                side_effect=controlled_execute,
            ),
            patch.object(
                auto_progress_module.threading.Timer,
                "start",
                side_effect=RuntimeError("thread limit reached"),
            ) as timer_start,
        ):
            self.assertTrue(self.server.auto_progress.schedule(run_id))
            self.assertTrue(executed.wait(timeout=2))

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if (
                self.server.director.state(run_id).run.status.value == "completed"
                and self.server.auto_progress._queue.unfinished_tasks == 0
            ):
                break
            time.sleep(0.01)

        timer_start.assert_called_once()
        self.assertEqual(
            self.server.director.state(run_id).run.status.value,
            "completed",
        )
        with self.server.auto_progress._state_lock:
            self.assertNotIn(work_item, self.server.auto_progress._retry_not_before)
            self.assertNotIn(work_item, self.server.auto_progress._retry_timers)

    def test_worker_clears_timer_when_generation_reaches_terminal_state(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "worker-terminal-clears-timer",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)
        state = self.server.director.state(run_id)
        work_item = (run_id, auto_progress_module._run_incarnation(state))
        timer = threading.Timer(60.0, lambda: None)
        timer.daemon = True
        timer.start()
        with self.server.auto_progress._state_lock:
            self.server.auto_progress._retry_timers[work_item] = timer

        self.assertTrue(self.server.auto_progress.schedule(run_id))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if (
                self.server.director.state(run_id).run.status.value == "completed"
                and self.server.auto_progress._queue.unfinished_tasks == 0
            ):
                break
            time.sleep(0.01)

        self.assertEqual(
            self.server.director.state(run_id).run.status.value,
            "completed",
        )
        with self.server.auto_progress._state_lock:
            self.assertNotIn(work_item, self.server.auto_progress._retry_timers)

    def test_purge_withdraws_queued_incarnation_before_run_id_reuse(self) -> None:
        blocker_id = "run-queued-purge-blocker"
        target_id = "run-queued-purge-target"
        for run_id, label in ((blocker_id, "blocker"), (target_id, "target")):
            status, created = self.request(
                "/runs",
                "POST",
                {
                    "run_id": run_id,
                    "domain_pack_id": "crop_soil_water",
                    "dataset_id": "generated-toy-series@1",
                    "rounds": 1,
                    "candidates_per_generation": 1,
                    "max_candidates": 1,
                    "auto_progress": True,
                    "auto_advance": 0,
                    "start": False,
                    "idempotency_key": f"queued-purge-{label}",
                },
            )
            self.assertEqual(status, 201, created)

        blocker_started = threading.Event()
        release_blocker = threading.Event()
        calls: list[str] = []
        execute_generation = auto_progress_module.execute_generation

        def controlled_execute(endpoint: object, run_id: str) -> object:
            calls.append(run_id)
            if run_id == blocker_id:
                blocker_started.set()
                release_blocker.wait(timeout=5)
            return execute_generation(endpoint, run_id)

        with patch.object(
            auto_progress_module,
            "execute_generation",
            side_effect=controlled_execute,
        ):
            try:
                with self.server.mutation_lock:
                    self.server.director.start_run(blocker_id)
                    self.assertTrue(self.server.auto_progress.schedule(blocker_id))
                self.assertTrue(blocker_started.wait(timeout=2))

                with self.server.mutation_lock:
                    self.server.director.start_run(target_id)
                    self.assertTrue(self.server.auto_progress.schedule(target_id))
                scheduler = self.server.auto_progress.diagnostics(target_id)
                self.assertEqual(scheduler["run_state"], "queued")
                self.assertEqual(scheduler["worker_count"], 1)
                self.assertEqual(scheduler["active_worker_count"], 1)
                self.assertEqual(scheduler["queue_position"], 1)
                self.assertEqual(scheduler["queued_ahead"], 0)
                self.assertTrue(scheduler["waiting_for_worker"])
                status, queued_projection = self.request(f"/runs/{target_id}")
                self.assertEqual(status, 200, queued_projection)
                self.assertEqual(
                    queued_projection["projection"]["execution_scheduler"],
                    scheduler,
                )
                old_state = self.server.director.state(target_id)
                old_work_item = (target_id, old_state.events[0].seq)

                status, cancelled = self.request(
                    f"/runs/{target_id}/control",
                    "POST",
                    {"action": "cancel"},
                )
                self.assertEqual(status, 200, cancelled)
                status, archived = self.request(
                    f"/runs/{target_id}/archive", "POST", {}
                )
                self.assertEqual(status, 200, archived)
                status, deleted = self.request(
                    f"/runs/{target_id}",
                    "DELETE",
                    {"confirm_run_id": target_id},
                )
                self.assertEqual(status, 200, deleted)
                with self.server.auto_progress._state_lock:
                    self.assertNotIn(
                        old_work_item, self.server.auto_progress._scheduled
                    )
                with self.server._generation_locks_guard:
                    self.assertNotIn(target_id, self.server._generation_locks)

                status, recreated = self.request(
                    "/runs",
                    "POST",
                    {
                        "run_id": target_id,
                        "domain_pack_id": "crop_soil_water",
                        "dataset_id": "generated-toy-series@1",
                        "rounds": 1,
                        "candidates_per_generation": 1,
                        "max_candidates": 1,
                        "auto_progress": False,
                        "auto_advance": 0,
                        "start": True,
                        "idempotency_key": "queued-purge-recreated",
                    },
                )
                self.assertEqual(status, 201, recreated)
            finally:
                release_blocker.set()

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with self.server.auto_progress._state_lock:
                    old_pending = (
                        old_work_item in self.server.auto_progress._scheduled
                        or old_work_item in self.server.auto_progress._running
                    )
                if (
                    not old_pending
                    and self.server.auto_progress._queue.unfinished_tasks == 0
                ):
                    break
                time.sleep(0.01)
            self.assertFalse(old_pending)
            self.assertEqual(self.server.auto_progress._queue.unfinished_tasks, 0)

        self.assertEqual(calls, [blocker_id])
        self.assertEqual(
            [event.kind for event in self.server.ledger.events(target_id)],
            ["RunCreated", "RunStarted"],
        )
        with self.server._generation_locks_guard:
            self.assertNotIn(target_id, self.server._generation_locks)

    def test_long_generation_does_not_block_pause_control(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 2,
                "candidates_per_generation": 1,
                "max_candidates": 2,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "pause-during-long-generation",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        generation_started = threading.Event()
        release_generation = threading.Event()

        def blocked_generation(_endpoint: object, target_run_id: str) -> object:
            self.assertEqual(target_run_id, run_id)
            generation_started.set()
            release_generation.wait(timeout=3)
            return self.server.director.state(run_id)

        timer = threading.Timer(2.0, release_generation.set)
        with patch.object(
            auto_progress_module,
            "execute_generation",
            side_effect=blocked_generation,
        ):
            with self.server.mutation_lock:
                self.server.director.start_run(run_id)
                self.server.auto_progress.schedule(run_id)
            self.assertTrue(generation_started.wait(timeout=1))
            timer.start()
            advance_started_at = time.monotonic()
            advance_status, advance_payload = self.request(
                f"/runs/{run_id}/advance", "POST", {"steps": 1}
            )
            advance_elapsed = time.monotonic() - advance_started_at
            started_at = time.monotonic()
            status, paused = self.request(
                f"/runs/{run_id}/action", "POST", {"action": "pause"}
            )
            elapsed = time.monotonic() - started_at
            release_generation.set()
            timer.cancel()

        self.assertEqual(status, 200, paused)
        self.assertEqual(paused["projection"]["status"], "paused")
        self.assertEqual(advance_status, 409, advance_payload)
        self.assertIn("already being advanced", advance_payload["error"])
        self.assertLess(advance_elapsed, 1.0)
        self.assertLess(elapsed, 1.0)

    def test_background_and_explicit_advance_share_one_generation_lease(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "one-generation-lease",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        proposal_started = threading.Event()
        release_proposal = threading.Event()
        propose = self.server.strategy_router.propose

        def blocked_propose(*args, **kwargs):
            proposal_started.set()
            release_proposal.wait(timeout=3)
            return propose(*args, **kwargs)

        timer = threading.Timer(2.0, release_proposal.set)
        with patch.object(
            self.server.strategy_router,
            "propose",
            side_effect=blocked_propose,
        ):
            with self.server.mutation_lock:
                self.server.director.start_run(run_id)
                self.server.auto_progress.schedule(run_id)
            self.assertTrue(proposal_started.wait(timeout=1))
            timer.start()
            advance_status, advance_payload = self.request(
                f"/runs/{run_id}/advance", "POST", {"steps": 1}
            )
            self.assertEqual(advance_status, 409, advance_payload)
            release_proposal.set()
            timer.cancel()

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = self.server.director.state(run_id)
                if state.run.status.value in {"completed", "failed"}:
                    break
                time.sleep(0.01)

        state = self.server.director.state(run_id)
        self.assertEqual(state.run.status.value, "completed")
        self.assertEqual(len(state.candidates), 1)
        event_kinds = [event.kind for event in state.events]
        self.assertEqual(event_kinds.count("ProposalSubmitted"), 1)
        self.assertEqual(event_kinds.count("CandidateSpawned"), 1)
        self.assertEqual(event_kinds.count("GenerationAdvanced"), 1)

    def test_pause_during_real_proposal_wait_is_not_recorded_as_failure(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 2,
                "candidates_per_generation": 1,
                "max_candidates": 2,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "pause-real-proposal-wait",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        proposal_started = threading.Event()
        release_proposal = threading.Event()
        propose = self.server.strategy_router.propose

        def blocked_propose(*args, **kwargs):
            proposal_started.set()
            release_proposal.wait(timeout=3)
            return propose(*args, **kwargs)

        timer = threading.Timer(2.0, release_proposal.set)
        with patch.object(
            self.server.strategy_router,
            "propose",
            side_effect=blocked_propose,
        ):
            with self.server.mutation_lock:
                self.server.director.start_run(run_id)
                self.server.auto_progress.schedule(run_id)
            self.assertTrue(proposal_started.wait(timeout=1))
            timer.start()
            pause_status, paused = self.request(
                f"/runs/{run_id}/action", "POST", {"action": "pause"}
            )
            self.assertEqual(pause_status, 200, paused)
            release_proposal.set()
            timer.cancel()

            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                state = self.server.director.state(run_id)
                if not self.server.generation_lock(run_id).acquire(blocking=False):
                    time.sleep(0.01)
                    continue
                self.server.generation_lock(run_id).release()
                break

        state = self.server.director.state(run_id)
        self.assertEqual(state.run.status.value, "paused")
        self.assertNotIn("RunFailed", [event.kind for event in state.events])

    def test_continuous_mode_finishes_without_an_advance_request(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 3,
                "candidates_per_generation": 1,
                "max_candidates": 3,
                "auto_progress": True,
                "auto_advance": 0,
                "idempotency_key": "continuous-regression",
            },
        )
        self.assertEqual(status, 201, created)
        initial = created["projection"]
        self.assertTrue(initial["auto_progress"])
        self.assertEqual(initial["status"], "running")
        self.assertEqual(initial["generation"], 0)
        self.assertEqual(initial["candidates"], [])

        run_id = initial["run_id"]
        deadline = time.monotonic() + 5
        latest = initial
        while time.monotonic() < deadline:
            _status, payload = self.request(f"/runs/{run_id}")
            latest = payload["projection"]
            if latest["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.02)

        self.assertEqual(latest["status"], "completed", latest)
        self.assertEqual(latest["generation"], 3)
        self.assertEqual(
            sum(
                event["kind"] == "GenerationAdvanced"
                for event in self.request(f"/runs/{run_id}/events")[1]["events"]
            ),
            3,
        )
        self.assertEqual(
            latest["execution_diagnostics"]["execution_mode"],
            "registered_lightweight",
        )
        diagnostics = latest["execution_diagnostics"]
        self.assertEqual(diagnostics["candidate_artifacts_count"], 3)
        self.assertEqual(diagnostics["candidate_evaluations_count"], 3)
        self.assertEqual(diagnostics["fit_passes_completed"], 3)
        self.assertEqual(diagnostics["evolution_rounds_completed"], 3)
        self.assertEqual(diagnostics["evolution_rounds_configured"], 3)
        self.assertGreater(diagnostics["training_used_examples"], 0)
        self.assertGreater(diagnostics["evaluation_used_examples"], 0)
        self.assertEqual(
            diagnostics["candidate_work_items"],
            diagnostics["training_used_examples"]
            + diagnostics["evaluation_used_examples"],
        )
        self.assertEqual(diagnostics["proposal_sources"], {"host_strategy": 3})
        self.assertEqual(diagnostics["remote_strategy_status"], "not_called")
        self.assertTrue(all("timing" in round_item for round_item in latest["rounds"]))

    def test_numeric_auto_advance_keeps_single_step_compatibility(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 2,
                "max_candidates": 2,
                "auto_advance": 1,
                "idempotency_key": "bounded-step-regression",
            },
        )
        self.assertEqual(status, 201, created)
        projection = created["projection"]
        self.assertFalse(projection["auto_progress"])
        self.assertEqual(projection["generation"], 1)
        self.assertEqual(projection["status"], "running")

    def test_multiple_continuous_runs_take_turns_by_generation(self) -> None:
        run_ids: list[str] = []
        for index in range(2):
            status, created = self.request(
                "/runs",
                "POST",
                {
                    "domain_pack_id": "crop_soil_water",
                    "dataset_id": "generated-toy-series@1",
                    "rounds": 3,
                    "candidates_per_generation": 1,
                    "max_candidates": 3,
                    "auto_progress": True,
                    "auto_advance": 0,
                    "start": False,
                    "idempotency_key": f"fair-continuous-{index}",
                },
            )
            self.assertEqual(status, 201, created)
            run_ids.append(created["projection"]["run_id"])

        calls: list[str] = []
        execute_generation = auto_progress_module.execute_generation

        def traced_execute(endpoint, run_id):
            calls.append(run_id)
            return execute_generation(endpoint, run_id)

        with patch.object(
            auto_progress_module,
            "execute_generation",
            side_effect=traced_execute,
        ):
            with self.server.mutation_lock:
                for run_id in run_ids:
                    self.server.director.start_run(run_id)
                    self.server.auto_progress.schedule(run_id)

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                states = [self.server.director.state(run_id) for run_id in run_ids]
                if all(state.run.status.value == "completed" for state in states):
                    break
                time.sleep(0.01)

        self.assertEqual(
            [self.server.director.state(run_id).run.status.value for run_id in run_ids],
            ["completed", "completed"],
        )
        # The worker pool may overlap different runs, but a run is dispatched
        # exactly once per generation and no run can monopolize the queue.
        self.assertEqual(Counter(calls), Counter({run_ids[0]: 3, run_ids[1]: 3}))

    def test_worker_revalidates_frozen_bindings_before_execution(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 2,
                "candidates_per_generation": 1,
                "max_candidates": 2,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "worker-binding-drift",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]

        with patch.object(
            self.server.strategy_router,
            "configuration_digest",
            return_value="0" * 64,
        ):
            status, started = self.request(
                f"/runs/{run_id}/action", "POST", {"action": "start"}
            )
            self.assertEqual(status, 200, started)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = self.server.director.state(run_id)
                if state.run.status.value == "failed":
                    break
                time.sleep(0.01)

        state = self.server.director.state(run_id)
        self.assertEqual(state.run.status.value, "failed")
        self.assertEqual(state.candidates, ())
        failure = next(
            event for event in reversed(state.events) if event.kind == "RunFailed"
        )
        self.assertIn("FrozenRuntimeBindingDriftError", failure.payload["reason"])
        self.assertIn("[frozen_runtime_binding_drift]", failure.payload["reason"])
        self.assertNotIn("进化策略实现发生漂移", failure.payload["reason"])
        self.assertNotIn("0" * 64, failure.payload["reason"])
        status, payload = self.request(f"/runs/{run_id}")
        self.assertEqual(status, 200, payload)
        projection = payload["projection"]
        self.assertEqual(
            projection["failure_code"], "frozen_runtime_binding_drift"
        )
        self.assertIn("请使用当前配置新建进化运行", projection["failure_reason"])
        self.assertNotIn("0" * 64, json.dumps(projection, ensure_ascii=False))

    def test_recovery_does_not_queue_an_archived_running_run(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 2,
                "auto_progress": True,
                "auto_advance": 0,
                "start": False,
                "idempotency_key": "archived-running-recovery",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        with self.server.mutation_lock:
            self.server.director.start_run(run_id)
            self.server.ledger.archive_run(run_id)

        with patch.object(
            self.server.auto_progress, "schedule", return_value=True
        ) as schedule:
            recovered = self.server.auto_progress.recover_running()

        self.assertEqual(recovered, 0)
        schedule.assert_not_called()


if __name__ == "__main__":
    unittest.main()
