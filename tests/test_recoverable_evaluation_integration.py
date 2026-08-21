from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ecologyrsi_dsh.api import auto_progress as auto_progress_module
from ecologyrsi_dsh.api.generation_execution import (
    _spawn_generation_candidates,
    execute_generation,
)
from ecologyrsi_dsh.api.projection import _projection_json
from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.models import CandidateStatus, TaskManifest
from ecologyrsi_dsh.core.sample_results import decode_sample_result_batch
from ecologyrsi_dsh.evaluators.registry import (
    RULE_JUDGE_ID,
    TOY_DATASET_ID,
    TOY_EVALUATOR_ID,
    TOY_PREDICTOR_MODEL_ID,
    EvaluatorRegistry,
)
from ecologyrsi_dsh.evolution.batches import start_generation_batch
from ecologyrsi_dsh.evolution.strategies import FakeDSHAdapter
from ecologyrsi_dsh.integrations.model_gateway import GatewayResponseError
from ecologyrsi_dsh.server import EvolutionHTTPServer


class _RetryableSiblingGateway:
    """Fail one planner wave only after a sibling wave is in flight."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._planner_calls = 0
        self.second_planner_started = threading.Event()
        self.error = GatewayResponseError(
            "provider queue remains busy after request-local retries",
            retryable=True,
            attempts=4,
            status_code=503,
        )

    @property
    def planner_calls(self) -> int:
        with self._lock:
            return self._planner_calls

    def sample_decide(
        self,
        _model_id: str,
        *,
        role: str,
        samples: list[dict],
        context: dict,
        available_tools: list[dict],
    ) -> dict:
        del context, available_tools
        if role == "planner":
            with self._lock:
                self._planner_calls += 1
                planner_call = self._planner_calls
            if planner_call == 1:
                if not self.second_planner_started.wait(timeout=2):
                    raise AssertionError("sibling planner wave never entered the gateway")
                raise self.error
            self.second_planner_started.set()

        decisions = []
        for item in samples:
            decisions.append(
                {
                    "sample_id": item["sample_id"],
                    "next_tool": (
                        item["sample"]["algorithm_id"]
                        if role == "planner"
                        else "accept"
                    ),
                    "reason_code": (
                        "registered_candidate_route"
                        if role == "planner"
                        else "remote_review_accept"
                    ),
                    "confidence": 0.9,
                }
            )
        return {"decisions": decisions}


class _RetryOnceGateway:
    """Fail the first planner wave after four physical HTTP attempts."""

    def __init__(self) -> None:
        self.planner_calls = 0

    def sample_decide(
        self,
        _model_id: str,
        *,
        role: str,
        samples: list[dict],
        context: dict,
        available_tools: list[dict],
    ) -> dict:
        del context, available_tools
        if role == "planner":
            self.planner_calls += 1
            if self.planner_calls == 1:
                raise GatewayResponseError(
                    "provider queue remains busy after request-local retries",
                    retryable=True,
                    attempts=4,
                    status_code=503,
                )

        return {
            "decisions": [
                {
                    "sample_id": item["sample_id"],
                    "next_tool": (
                        item["sample"]["algorithm_id"]
                        if role == "planner"
                        else "accept"
                    ),
                    "reason_code": (
                        "registered_candidate_route"
                        if role == "planner"
                        else "remote_review_accept"
                    ),
                    "confidence": 0.9,
                }
                for item in samples
            ]
        }


class _BudgetedSiblingGateway:
    """Return complete usage for two concurrent waves under a hard budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[dict[str, object]] = []

    def _decide(
        self,
        model_id: str,
        *,
        role: str,
        samples: list[dict],
        context: dict,
        available_tools: list[dict],
    ) -> dict:
        del available_tools
        decision_context = context.get("decision_context")
        with self._lock:
            self.calls.append(
                {
                    "model_id": model_id,
                    "role": role,
                    "candidate_id": (
                        decision_context.get("candidate_id")
                        if isinstance(decision_context, dict)
                        else None
                    ),
                    "sample_ids": tuple(item["sample_id"] for item in samples),
                }
            )
        return {
            "decisions": [
                {
                    "sample_id": item["sample_id"],
                    "next_tool": (
                        item["sample"]["algorithm_id"]
                        if role == "planner"
                        else "accept"
                    ),
                    "reason_code": (
                        "registered_candidate_route"
                        if role == "planner"
                        else "remote_review_accept"
                    ),
                    "confidence": 0.9,
                }
                for item in samples
            ]
        }

    def sample_decide(self, model_id: str, **kwargs: object) -> dict:
        return self._decide(model_id, **kwargs)  # type: ignore[arg-type]

    def sample_decide_with_diagnostics(
        self,
        model_id: str,
        *,
        role: str,
        samples: list[dict],
        context: dict,
        available_tools: list[dict],
        allow_format_retry: bool,
    ) -> tuple[dict, dict]:
        del allow_format_retry
        response = self._decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )
        return response, {
            "usage_reported": True,
            "http_attempts": 1,
            "prompt_tokens": 6,
            "completion_tokens": 4,
            "total_tokens": 10,
        }


class _ControlledInFlightGateway:
    """Hold one physical planner call across a run-control transition."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[dict[str, object]] = []
        self.first_planner_started = threading.Event()
        self.release_first_planner = threading.Event()

    def _decide(
        self,
        model_id: str,
        *,
        role: str,
        samples: list[dict],
        context: dict,
        available_tools: list[dict],
    ) -> dict:
        del context, available_tools
        with self._lock:
            self.calls.append(
                {
                    "model_id": model_id,
                    "role": role,
                    "sample_ids": tuple(item["sample_id"] for item in samples),
                }
            )
            call_index = len(self.calls)
        if role == "planner" and call_index == 1:
            self.first_planner_started.set()
            if not self.release_first_planner.wait(timeout=5):
                raise AssertionError("controlled planner call was not released")
        return {
            "decisions": [
                {
                    "sample_id": item["sample_id"],
                    "next_tool": (
                        item["sample"]["algorithm_id"]
                        if role == "planner"
                        else "accept"
                    ),
                    "reason_code": (
                        "registered_candidate_route"
                        if role == "planner"
                        else "remote_review_accept"
                    ),
                    "confidence": 0.9,
                }
                for item in samples
            ]
        }

    def sample_decide(self, model_id: str, **kwargs: object) -> dict:
        return self._decide(model_id, **kwargs)  # type: ignore[arg-type]

    def sample_decide_with_diagnostics(
        self,
        model_id: str,
        *,
        role: str,
        samples: list[dict],
        context: dict,
        available_tools: list[dict],
        allow_format_retry: bool,
    ) -> tuple[dict, dict]:
        del allow_format_retry
        response = self._decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )
        return response, {
            "usage_reported": True,
            "http_attempts": 1,
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        }


class RecoverableEvaluationIntegrationTests(unittest.TestCase):
    def test_cancel_during_gateway_call_records_only_late_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = EvolutionHTTPServer(
                ("127.0.0.1", 0), Path(directory) / "events.sqlite3"
            )
            execution_errors: list[BaseException] = []
            execution_states: list[object] = []
            worker: threading.Thread | None = None
            gateway = _ControlledInFlightGateway()
            try:
                server.director = EvolutionDirector(
                    server.ledger, FakeDSHAdapter(max_proposals=1)
                )
                server.evaluators = EvaluatorRegistry(
                    server.datasets, model_gateway=gateway  # type: ignore[arg-type]
                )
                series = server.datasets.series(TOY_DATASET_ID)
                task = TaskManifest(
                    task_id="cancel-inflight-sample-integration",
                    objective="cancel without losing already incurred model usage",
                    domain_pack="crop-soil-water@toy",
                    visible_datasets=(TOY_DATASET_ID,),
                    budget={
                        "max_candidates": 1,
                        "max_generations": 1,
                        "candidates_per_generation": 1,
                    },
                    metadata={
                        "domain": "toy",
                        "strategy_id": "parameter_sweep@1",
                        "prediction_model_id": TOY_PREDICTOR_MODEL_ID,
                        "evaluator_id": TOY_EVALUATOR_ID,
                        "judge_model_id": RULE_JUDGE_ID,
                        "dataset_digest": series.digest,
                        "split_manifest_digest": (
                            series.split_manifest_digest_sha256
                        ),
                        "sample_agent_mode": "gateway_microbatch",
                        "sample_agent_batch_size": 1,
                        "sample_concurrency": 1,
                        "strategy_model_id": "strategy-model",
                        "review_model_id": "review-model",
                        "knowledge_online_enabled": False,
                    },
                )
                run_id = server.director.start_evolution(
                    task, run_id="run:cancel-inflight-sample-integration"
                ).run.run_id
                endpoint = SimpleNamespace(server=server)
                batch = start_generation_batch(server.director, run_id)
                self.assertTrue(
                    _spawn_generation_candidates(endpoint, run_id, batch)
                )

                def execute() -> None:
                    try:
                        execution_states.append(execute_generation(endpoint, run_id))
                    except BaseException as exc:  # noqa: BLE001 - test thread boundary
                        execution_errors.append(exc)

                worker = threading.Thread(target=execute, daemon=True)
                worker.start()
                self.assertTrue(gateway.first_planner_started.wait(timeout=5))
                before_cancel = server.director.state(run_id)
                self.assertTrue(
                    any(
                        event.kind == "EvaluationSampleResultsStarted"
                        for event in before_cancel.events
                    )
                )

                server.director.cancel_run(run_id, "integration control race")
                cancelled = server.director.state(run_id)
                cancel_event = next(
                    event
                    for event in reversed(cancelled.events)
                    if event.kind == "RunCancelled"
                )
                gateway.release_first_planner.set()
                worker.join(timeout=10)
                self.assertFalse(worker.is_alive())
                self.assertEqual(execution_errors, [])
                self.assertEqual(len(execution_states), 1)

                final = server.director.state(run_id)
                candidate = final.candidates[0]
                late_usage = [
                    event
                    for event in final.events
                    if event.kind == "ModelUsageRecorded"
                    and event.seq > cancel_event.seq
                ]
                late_sample_publications = [
                    event
                    for event in final.events
                    if event.seq > cancel_event.seq
                    and event.kind
                    in {
                        "EvaluationSampleResultBatchRecorded",
                        "EvaluationProgressRecorded",
                    }
                ]

                self.assertEqual(final.run.status.value, "cancelled")
                self.assertIs(candidate.status, CandidateStatus.SPAWNED)
                self.assertEqual(len(gateway.calls), 1)
                self.assertEqual(len(late_usage), 1)
                self.assertEqual(late_usage[0].payload["total_tokens"], 10)
                self.assertEqual(late_sample_publications, [])
                self.assertFalse(
                    any(event.kind == "CandidateFailed" for event in final.events)
                )
                self.assertFalse(any(event.kind == "RunFailed" for event in final.events))
            finally:
                gateway.release_first_planner.set()
                if worker is not None:
                    worker.join(timeout=5)
                server.close()

    def test_hard_token_budget_pauses_after_persisting_inflight_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = EvolutionHTTPServer(
                ("127.0.0.1", 0), Path(directory) / "events.sqlite3"
            )
            try:
                gateway = _BudgetedSiblingGateway()
                server.director = EvolutionDirector(
                    server.ledger, FakeDSHAdapter(max_proposals=2)
                )
                server.evaluators = EvaluatorRegistry(
                    server.datasets, model_gateway=gateway  # type: ignore[arg-type]
                )
                series = server.datasets.series(TOY_DATASET_ID)
                task = TaskManifest(
                    task_id="hard-token-budget-integration",
                    objective="pause after durable usage exhausts the wave budget",
                    domain_pack="crop-soil-water@toy",
                    visible_datasets=(TOY_DATASET_ID,),
                    budget={
                        "max_candidates": 2,
                        "max_generations": 1,
                        "candidates_per_generation": 2,
                        "token_limit": 100,
                        "token_reservation_per_wave": 20,
                    },
                    metadata={
                        "domain": "toy",
                        "strategy_id": "parameter_sweep@1",
                        "prediction_model_id": TOY_PREDICTOR_MODEL_ID,
                        "evaluator_id": TOY_EVALUATOR_ID,
                        "judge_model_id": RULE_JUDGE_ID,
                        "dataset_digest": series.digest,
                        "split_manifest_digest": (
                            series.split_manifest_digest_sha256
                        ),
                        "sample_agent_mode": "gateway_microbatch",
                        "sample_agent_batch_size": 1,
                        "sample_concurrency": 2,
                        "strategy_model_id": "strategy-model",
                        "review_model_id": "review-model",
                        "knowledge_online_enabled": False,
                    },
                )
                run_id = server.director.start_evolution(
                    task, run_id="run:hard-token-budget-integration"
                ).run.run_id
                endpoint = SimpleNamespace(server=server)
                batch = start_generation_batch(server.director, run_id)
                self.assertTrue(
                    _spawn_generation_candidates(endpoint, run_id, batch)
                )
                seeded = server.director.state(run_id)
                candidates = sorted(seeded.candidates, key=lambda item: item.slot_index)
                first, second = candidates
                server.director.record_model_usage(
                    run_id,
                    generation=first.generation,
                    candidate_id=first.candidate_id,
                    role="planner",
                    model_id="historical-model",
                    usage={
                        "prompt_tokens": 50,
                        "completion_tokens": 10,
                        "total_tokens": 60,
                    },
                    gateway_request_count=1,
                    revision="historical:durable-usage",
                    usage_index=0,
                )

                paused = execute_generation(endpoint, run_id)
                projection = _projection_json(paused)
                pause_event = next(
                    event
                    for event in reversed(paused.events)
                    if event.kind == "RunPaused"
                )
                first_starts = [
                    event
                    for event in paused.events
                    if event.kind == "EvaluationSampleResultsStarted"
                    and event.payload.get("candidate_id") == first.candidate_id
                ]
                first_batches = [
                    event
                    for event in paused.events
                    if event.kind == "EvaluationSampleResultBatchRecorded"
                    and event.payload.get("candidate_id") == first.candidate_id
                    and event.payload.get("revision")
                    == first_starts[-1].payload.get("revision")
                ]
                persisted_rows = [
                    row
                    for event in first_batches
                    for row in decode_sample_result_batch(event.payload)
                ]
                usage_events = [
                    event
                    for event in paused.events
                    if event.kind == "ModelUsageRecorded"
                ]

                self.assertEqual(paused.run.status.value, "paused")
                self.assertEqual(projection["pause_code"], "model_token_budget_exhausted")
                self.assertIn("Token", projection["pause_reason"])
                # Two concurrent planners are admitted from the historical
                # baseline. Only one subsequent critic fits its frozen
                # per-call bound; the other is rejected before submission even
                # though the admitted critic ultimately reports fewer tokens.
                self.assertEqual(projection["tokens_used"], 90)
                self.assertEqual(projection["token_limit"], 100)
                self.assertEqual(len(usage_events), 4)
                self.assertEqual(len(persisted_rows), 1)
                self.assertTrue(all(event.seq < pause_event.seq for event in first_batches))
                self.assertIs(
                    paused.candidate(first.candidate_id).status,
                    CandidateStatus.SPAWNED,
                )
                self.assertIs(
                    paused.candidate(second.candidate_id).status,
                    CandidateStatus.SPAWNED,
                )
                self.assertEqual(
                    {
                        call["candidate_id"]
                        for call in gateway.calls
                    },
                    {first.candidate_id},
                )
                self.assertEqual(
                    [call["role"] for call in gateway.calls].count("planner"),
                    2,
                )
                self.assertEqual(
                    [call["role"] for call in gateway.calls].count("critic"),
                    1,
                )
                self.assertFalse(
                    any(
                        event.kind == "EvolutionStageRecorded"
                        and event.payload.get("candidate_id") == second.candidate_id
                        and event.payload.get("stage") == "training"
                        for event in paused.events
                    )
                )
                self.assertFalse(
                    any(event.kind == "CandidateFailed" for event in paused.events)
                )
                self.assertFalse(any(event.kind == "RunFailed" for event in paused.events))

                calls_before_paused_reentry = len(gateway.calls)
                paused_again = execute_generation(endpoint, run_id)
                self.assertEqual(paused_again.run.status.value, "paused")
                self.assertEqual(len(gateway.calls), calls_before_paused_reentry)
            finally:
                server.close()

    def test_retryable_sample_gateway_keeps_open_candidate_and_defers_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = EvolutionHTTPServer(
                ("127.0.0.1", 0), Path(directory) / "events.sqlite3"
            )
            try:
                gateway = _RetryableSiblingGateway()
                server.director = EvolutionDirector(
                    server.ledger, FakeDSHAdapter(max_proposals=1)
                )
                server.evaluators = EvaluatorRegistry(
                    server.datasets, model_gateway=gateway  # type: ignore[arg-type]
                )
                series = server.datasets.series(TOY_DATASET_ID)
                task = TaskManifest(
                    task_id="recoverable-sample-gateway-integration",
                    objective="preserve durable sample work across provider cooldown",
                    domain_pack="crop-soil-water@toy",
                    visible_datasets=(TOY_DATASET_ID,),
                    budget={
                        "max_candidates": 1,
                        "max_generations": 1,
                        "candidates_per_generation": 1,
                    },
                    metadata={
                        "domain": "toy",
                        "strategy_id": "parameter_sweep@1",
                        "prediction_model_id": TOY_PREDICTOR_MODEL_ID,
                        "evaluator_id": TOY_EVALUATOR_ID,
                        "judge_model_id": RULE_JUDGE_ID,
                        "dataset_digest": series.digest,
                        "split_manifest_digest": (
                            series.split_manifest_digest_sha256
                        ),
                        "sample_agent_mode": "gateway_microbatch",
                        "sample_agent_batch_size": 1,
                        "sample_concurrency": 2,
                        "strategy_model_id": "strategy-model",
                        "review_model_id": "review-model",
                        "knowledge_online_enabled": False,
                        "auto_progress": True,
                    },
                )
                run_id = server.director.start_evolution(
                    task, run_id="run:recoverable-sample-gateway-integration"
                ).run.run_id

                with (
                    patch.object(
                        server,
                        "validate_frozen_runtime_bindings",
                        return_value=None,
                    ),
                    patch.object(
                        auto_progress_module,
                        "_GATEWAY_RETRY_BASE_SECONDS",
                        30.0,
                    ),
                ):
                    keep_running = server.auto_progress._run_one_generation(run_id)

                state = server.director.state(run_id)
                candidate = state.candidates[0]
                starts = [
                    event
                    for event in state.events
                    if event.kind == "EvaluationSampleResultsStarted"
                    and event.payload.get("candidate_id") == candidate.candidate_id
                ]
                batches = [
                    event
                    for event in state.events
                    if event.kind == "EvaluationSampleResultBatchRecorded"
                    and event.payload.get("candidate_id") == candidate.candidate_id
                    and event.payload.get("revision")
                    == starts[-1].payload.get("revision")
                ]
                completions = [
                    event
                    for event in state.events
                    if event.kind == "EvaluationSampleResultsRecorded"
                    and event.payload.get("candidate_id") == candidate.candidate_id
                ]
                projection = _projection_json(state)

                self.assertTrue(keep_running)
                self.assertEqual(state.run.status.value, "running")
                self.assertIs(candidate.status, CandidateStatus.SPAWNED)
                self.assertEqual(len(starts), 1)
                self.assertEqual(
                    starts[0].payload["schema_version"],
                    "ecologyrsi-dsh.evaluation-sample-results-start/2",
                )
                self.assertTrue(batches)
                self.assertEqual(completions, [])
                self.assertIsNone(state.evaluation_for(candidate.candidate_id))
                self.assertFalse(
                    any(event.kind == "CandidateFailed" for event in state.events)
                )
                self.assertFalse(any(event.kind == "RunFailed" for event in state.events))
                self.assertTrue(
                    any(event.kind == "GatewayRetryScheduled" for event in state.events)
                )
                self.assertTrue(projection["execution_progress"]["retry_wait"]["waiting"])
                self.assertGreaterEqual(gateway.planner_calls, 2)

                revision = str(starts[0].payload["revision"])
                persisted_sample_ids = {
                    str(row["sample_id"])
                    for event in batches
                    for row in decode_sample_result_batch(event.payload)
                }
                expected_pending = (
                    int(starts[0].payload["checkpoint"]["sample_count"])
                    - len(persisted_sample_ids)
                )
                planner_calls_before_resume = gateway.planner_calls
                work_item = (run_id, state.events[0].seq)
                with server.auto_progress._state_lock:
                    server.auto_progress._retry_not_before[work_item] = 0.0

                with patch.object(
                    server,
                    "validate_frozen_runtime_bindings",
                    return_value=None,
                ):
                    keep_running_after_resume = (
                        server.auto_progress._run_one_generation(run_id)
                    )

                resumed = server.director.state(run_id)
                resumed_candidate = resumed.candidate(candidate.candidate_id)
                resumed_starts = [
                    event
                    for event in resumed.events
                    if event.kind == "EvaluationSampleResultsStarted"
                    and event.payload.get("candidate_id") == candidate.candidate_id
                ]
                resume_events = [
                    event
                    for event in resumed.events
                    if event.kind == "EvaluationSampleResultsResumed"
                    and event.payload.get("candidate_id") == candidate.candidate_id
                ]

                self.assertFalse(keep_running_after_resume)
                self.assertEqual(len(resumed_starts), 1)
                self.assertEqual(resumed_starts[0].payload["revision"], revision)
                self.assertTrue(resume_events)
                self.assertEqual(resume_events[-1].payload["revision"], revision)
                self.assertEqual(
                    gateway.planner_calls - planner_calls_before_resume,
                    expected_pending,
                )
                self.assertIsNotNone(
                    resumed.evaluation_for(resumed_candidate.candidate_id)
                )
                self.assertTrue(
                    any(
                        event.kind == "EvaluationSampleResultsRecorded"
                        and event.payload.get("revision") == revision
                        for event in resumed.events
                    )
                )
                self.assertFalse(
                    any(event.kind == "CandidateFailed" for event in resumed.events)
                )
            finally:
                server.close()

    def test_resume_progress_includes_all_durable_http_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = EvolutionHTTPServer(
                ("127.0.0.1", 0), Path(directory) / "events.sqlite3"
            )
            try:
                gateway = _RetryOnceGateway()
                server.director = EvolutionDirector(
                    server.ledger, FakeDSHAdapter(max_proposals=1)
                )
                server.evaluators = EvaluatorRegistry(
                    server.datasets, model_gateway=gateway  # type: ignore[arg-type]
                )
                series = server.datasets.series(TOY_DATASET_ID)
                task = TaskManifest(
                    task_id="resume-durable-http-attempts-integration",
                    objective="restore the complete durable gateway request count",
                    domain_pack="crop-soil-water@toy",
                    visible_datasets=(TOY_DATASET_ID,),
                    budget={
                        "max_candidates": 1,
                        "max_generations": 1,
                        "candidates_per_generation": 1,
                    },
                    metadata={
                        "domain": "toy",
                        "strategy_id": "parameter_sweep@1",
                        "prediction_model_id": TOY_PREDICTOR_MODEL_ID,
                        "evaluator_id": TOY_EVALUATOR_ID,
                        "judge_model_id": RULE_JUDGE_ID,
                        "dataset_digest": series.digest,
                        "split_manifest_digest": (
                            series.split_manifest_digest_sha256
                        ),
                        "sample_agent_mode": "gateway_microbatch",
                        "sample_agent_batch_size": 128,
                        "sample_concurrency": 1,
                        "strategy_model_id": "strategy-model",
                        "review_model_id": "review-model",
                        "knowledge_online_enabled": False,
                        "auto_progress": True,
                    },
                )
                run_id = server.director.start_evolution(
                    task, run_id="run:resume-durable-http-attempts-integration"
                ).run.run_id

                with (
                    patch.object(
                        server,
                        "validate_frozen_runtime_bindings",
                        return_value=None,
                    ),
                    patch.object(
                        auto_progress_module,
                        "_GATEWAY_RETRY_BASE_SECONDS",
                        30.0,
                    ),
                ):
                    self.assertTrue(
                        server.auto_progress._run_one_generation(run_id)
                    )

                deferred = server.director.state(run_id)
                candidate = deferred.candidates[0]
                starts = [
                    event
                    for event in deferred.events
                    if event.kind == "EvaluationSampleResultsStarted"
                    and event.payload.get("candidate_id") == candidate.candidate_id
                ]
                self.assertEqual(len(starts), 1)
                revision = str(starts[0].payload["revision"])
                failed_usage = [
                    event
                    for event in deferred.events
                    if event.kind == "ModelUsageRecorded"
                    and event.payload.get("schema_version")
                    == "ecologyrsi-dsh.model-usage/2"
                    and event.payload.get("candidate_id") == candidate.candidate_id
                    and event.payload.get("revision") == revision
                ]
                self.assertEqual(
                    sum(int(event.payload["http_attempts"]) for event in failed_usage),
                    4,
                )

                work_item = (run_id, deferred.events[0].seq)
                with server.auto_progress._state_lock:
                    server.auto_progress._retry_not_before[work_item] = 0.0
                with patch.object(
                    server,
                    "validate_frozen_runtime_bindings",
                    return_value=None,
                ):
                    self.assertFalse(
                        server.auto_progress._run_one_generation(run_id)
                    )

                resumed = server.director.state(run_id)
                resumed_starts = [
                    event
                    for event in resumed.events
                    if event.kind == "EvaluationSampleResultsStarted"
                    and event.payload.get("candidate_id") == candidate.candidate_id
                ]
                resume_events = [
                    event
                    for event in resumed.events
                    if event.kind == "EvaluationSampleResultsResumed"
                    and event.payload.get("candidate_id") == candidate.candidate_id
                    and event.payload.get("revision") == revision
                ]
                usage_events = [
                    event
                    for event in resumed.events
                    if event.kind == "ModelUsageRecorded"
                    and event.payload.get("schema_version")
                    == "ecologyrsi-dsh.model-usage/2"
                    and event.payload.get("candidate_id") == candidate.candidate_id
                    and event.payload.get("revision") == revision
                ]
                progress_events = [
                    event
                    for event in resumed.events
                    if event.kind == "EvaluationProgressRecorded"
                    and event.payload.get("schema_version")
                    == "ecologyrsi-dsh.evaluation-progress/3"
                    and event.payload.get("candidate_id") == candidate.candidate_id
                    and event.payload.get("revision") == revision
                    and event.payload.get("role") == "planner"
                ]
                durable_attempts = sum(
                    int(event.payload["http_attempts"])
                    for event in usage_events
                )

                self.assertEqual(len(resumed_starts), 1)
                self.assertTrue(resume_events)
                self.assertTrue(progress_events)
                self.assertEqual(
                    progress_events[-1].payload["gateway_request_count"],
                    durable_attempts,
                )
                self.assertEqual(usage_events[0].payload["outcome"], "failed")
                self.assertEqual(usage_events[0].payload["http_attempts"], 4)
                self.assertTrue(
                    all(
                        event.payload["outcome"] == "succeeded"
                        and event.payload["http_attempts"] == 1
                        for event in usage_events[1:]
                    )
                )
                self.assertGreater(durable_attempts, 4)
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
