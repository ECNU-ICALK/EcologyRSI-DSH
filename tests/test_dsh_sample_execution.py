from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import threading
import unittest

from ecologyrsi_dsh.api.dsh_tools import DshPredictionToolBinding, DshToolService
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.models import TaskManifest
from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
from ecologyrsi_dsh.core.redaction import REMOTE_REASON_CODES
from ecologyrsi_dsh.core.sample_results import build_sample_results
from ecologyrsi_dsh.data.registry import DatasetRegistry
from ecologyrsi_dsh.evaluators.dsh_sample_adapter import (
    DshSampleCollaborationAdapter,
)
from ecologyrsi_dsh.evaluators.gateway_sample_adapter import (
    GatewaySampleCollaborationAdapter,
)
from ecologyrsi_dsh.evaluators.sample_execution import (
    CollaborativeSampleExecutor,
    SamplePredictionRequest,
    classify_sample_failure,
)
from ecologyrsi_dsh.evaluators.registry import EvaluatorRegistry
from ecologyrsi_dsh.integrations.dsh_native_runtime import (
    DshNativeAgentRuntimeClient,
)
from ecologyrsi_dsh.integrations.dsh_structured_roles import DshStructuredRoleRuntime


def _skill_evidence(stage: str) -> dict:
    return {
        "schema_version": "ecologyrsi-dsh.skill-invocation-evidence/1",
        "stage": stage,
        "skill_name": (
            "origin-vector-forecasting-balanced"
            if stage == "sample.plan"
            else "origin-vector-review"
        ),
        "call_count": 1,
        "successful_call_count": 1,
        "call_seq": 1,
        "result_seq": 2,
        "first_tool_call_verified": True,
        "next_tool_name": (
            "ecology_execute_prediction_tool"
            if stage == "sample.plan"
            else "structured_output"
        ),
        "next_tool_call_seq": 3,
        "order_verified": True,
        "source": "dsh_session_event_log",
    }


class _SampleRuntime:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def run_stage(self, request: dict) -> dict:
        self.requests.append(request)
        stage_context = request["request"]["context"]
        if request["stage"] == "sample.reflect":
            structured = {
                "schema_version": "ecology-sample-reflection@1",
                "wave_digest": stage_context["wave_digest"],
                "sample_id": stage_context["sample"]["sample_id"],
                "outcome_class": "neutral",
                "error_source": "unknown",
                "next_action": "keep",
                "confidence": 0.8,
                "summary": "Retain the bounded tool configuration for later evidence.",
            }
            return {"structured": structured, "result_digest": digest(structured)}
        samples = stage_context["samples"]
        tools = stage_context["available_tools"]
        if request["stage"] == "sample.critic":
            next_tool = next(
                item["tool_id"] for item in tools if item["tool_id"] == "accept"
            )
            version = "ecology-sample-review@1"
        else:
            next_tool = tools[0]["tool_id"]
            version = "ecology-sample-decisions@1"
        structured = {
            "schema_version": version,
            "wave_digest": stage_context["wave_digest"],
            "decisions": [
                {
                    "sample_id": item["sample_id"],
                    "next_tool": next_tool,
                    "reason_code": "initial_registered_route",
                    "confidence": 0.9,
                }
                for item in samples
            ],
        }
        return {"structured": structured, "result_digest": digest(structured)}


class _FailureReviewRuntime(_SampleRuntime):
    def run_stage(self, request: dict) -> dict:
        if request["stage"] != "sample.critic":
            return super().run_stage(request)
        self.requests.append(request)
        self.assert_critic_request(request)
        stage_context = request["request"]["context"]
        next_tool = next(
            item["tool_id"]
            for item in stage_context["available_tools"]
            if item["tool_id"] == "bounded-persistence-fallback"
        )
        structured = {
            "schema_version": "ecology-sample-review@1",
            "wave_digest": stage_context["wave_digest"],
            "decisions": [
                {
                    "sample_id": item["sample_id"],
                    "next_tool": next_tool,
                    "reason_code": "critic_selects_alternate_tool",
                    "confidence": 1.0,
                }
                for item in stage_context["samples"]
            ],
        }
        return {"structured": structured, "result_digest": digest(structured)}

    @staticmethod
    def assert_critic_request(request: dict) -> None:
        if request["stage"] != "sample.critic":
            raise AssertionError("tool failure recovery must be reviewed by DSH critic")


class _CriticUnavailableRuntime(_SampleRuntime):
    def run_stage(self, request: dict) -> dict:
        if request["stage"] == "sample.critic":
            self.requests.append(request)
            raise RuntimeError("critic unavailable")
        return super().run_stage(request)


class _PersistingRetryRuntime:
    def __init__(self, service: DshToolService, ledger: EventLedger) -> None:
        self.service = service
        self.ledger = ledger
        self.requests: list[dict] = []
        self.fail_next_critic = True

    def run_stage(self, request: dict) -> dict:
        self.requests.append(request)
        stage = request["stage"]
        context = request["request"]["context"]
        if stage == "sample.critic" and self.fail_next_critic:
            self.fail_next_critic = False
            self.ledger.append(request["run_id"], "TestCriticAttemptFailed", {})
            raise DshNativeRuntimeUnavailableError(
                "critic temporarily unavailable",
                error_code="dsh_native_runtime_http_error",
                status_code=502,
            )

        session_id = f"session:{stage}:{len(self.requests)}"
        reservation = self.service.allocate_child_reservation(
            {
                "request_id": f"persisting-retry-{len(self.requests)}",
                "run_id": request["run_id"],
                "parent_session_id": "session:role-host",
                "role": request["request"]["role"],
                "stage": stage,
                "run_state_revision": request["run_state_revision"],
                "stage_attempt": request["stage_attempt"],
                "admission_id": request["admission_id"],
                "timeout_ms": 1_000,
                "item_digest": request["request"]["context_digest"],
                "idempotency_key": request["idempotency_key"],
            }
        )
        identity = {
            "run_id": request["run_id"],
            "role": request["request"]["role"],
            "stage": stage,
            "run_state_revision": request["run_state_revision"],
            "stage_attempt": request["stage_attempt"],
            "ledger_expected_revision": self.ledger.latest_seq(),
            "session_id": session_id,
            "idempotency_key": request["idempotency_key"],
            "child_reservation_id": reservation["launch"]["reservation_id"],
            "activation_lease_id": f"lease:{len(self.requests)}",
            **request["request"]["identity_digests"],
        }
        if stage == "sample.plan":
            tool_id = context["available_tools"][0]["tool_id"]
            self.service.execute(
                "ecology_execute_prediction_tool",
                {
                    "identity": identity,
                    "arguments": {
                        "tool_id": tool_id,
                        "wave_digest": context["wave_digest"],
                    },
                },
            )
            schema_version = "ecology-sample-decisions@1"
            next_tool = tool_id
        else:
            schema_version = "ecology-sample-review@1"
            next_tool = "accept"
        structured = {
            "schema_version": schema_version,
            "wave_digest": context["wave_digest"],
            "decisions": [
                {
                    "sample_id": item["sample_id"],
                    "next_tool": next_tool,
                    "reason_code": "accept_prediction",
                    "confidence": 0.9,
                }
                for item in context["samples"]
            ],
        }
        self.service.accept_structured(
            {
                "identity": identity,
                "output_schema_id": request["request"]["output_schema_id"],
                "structured": structured,
                "result_digest": digest(structured),
                "skill_invocation_evidence": _skill_evidence(stage),
                "admission_id": request["admission_id"],
            }
        )
        return {"structured": structured, "result_digest": digest(structured)}


class _EquivalentDecisionGateway:
    def sample_decide(self, _model_id: str, *, role: str, samples, available_tools, **_kwargs):
        next_tool = (
            next(item["tool_id"] for item in available_tools if item["tool_id"] == "accept")
            if role == "critic"
            else available_tools[0]["tool_id"]
        )
        return {
            "decisions": [
                {
                    "sample_id": item["sample_id"],
                    "next_tool": next_tool,
                    "reason_code": "initial_registered_route",
                    "confidence": 0.9,
                }
                for item in samples
            ]
        }


class _FakeAgentPredictionBinding:
    def __init__(self, values: dict, *, event_id: str) -> None:
        self.values = values
        self.event_id = event_id
        self.output_digest = digest(values)

    def prediction_bundle(self) -> dict:
        return self.values

    def audit_receipt(self) -> dict:
        return {
            "event_id": self.event_id,
            "output_digest": self.output_digest,
            "execution_owner": "dsh_agent_tool_call",
        }


@contextmanager
def _fake_agent_prediction_binder(**binding):
    values = binding["executor"]()
    yield _FakeAgentPredictionBinding(
        values,
        event_id=f"{binding['run_id']}:tool:{binding['wave_digest']}",
    )


def _constant_forecast_bundle(value: float):
    def execute(requests):
        return {
            request.sample_id: {
                "predicted": value,
                "metadata": {"source_model_id": request.algorithm_id},
            }
            for request in requests
        }

    return execute


def _request(sample_id: str) -> SamplePredictionRequest:
    return SamplePredictionRequest(
        sample_id=sample_id,
        candidate_id="candidate-1",
        dataset_digest="d" * 64,
        partition="training_feedback",
        target="air_temperature",
        unit="degC",
        horizon_hours=1,
        origin_timestamp=10,
        target_timestamp=11,
        baseline=20.0,
        proposed_prediction=None,
        minimum=-20.0,
        maximum=80.0,
        algorithm_id="registered-predictor",
        algorithm_version="1",
        label_free_context={
            "schema_version": "ecologyrsi-dsh.label-free-sample-context/1",
            "history_window": [19.0, 20.0],
            "causal_provenance": {
                "schema_version": "ecologyrsi-dsh.causal-sample-provenance/1",
                "origin_cutoff_timestamp": 10,
                "latest_context_timestamp": 10,
                "history_timestamps": [9, 10],
            },
        },
    )


class DshSampleExecutionTests(unittest.TestCase):
    def test_fresh_planner_children_reuse_one_prediction_result_and_receipt(self) -> None:
        calls = 0

        def execute_prediction() -> dict:
            nonlocal calls
            calls += 1
            return {
                "origin-a": {
                    "predicted": 21.5,
                    "metadata": {"source_model_id": "registered-predictor"},
                }
            }

        binding = DshPredictionToolBinding(
            run_id="run-planner-missing-retry",
            stage_attempt=1,
            idempotency_key="sample-plan-wave-1",
            wave_digest="f" * 64,
            tool_id="registered-predictor@1",
            sample_ids=("origin-a",),
            executor=execute_prediction,
        )
        arguments = {
            "tool_id": "registered-predictor@1",
            "wave_digest": "f" * 64,
        }

        first = binding.execute(arguments, session_id="planner-child-1")
        binding.set_receipt({
            "event_id": "prediction-event-1",
            "event_seq": 12,
            "request_digest": binding.request_digest(),
            "output_digest": first["output_digest"],
            "execution_owner": "dsh_agent_tool_call",
        })
        second = binding.execute(arguments, session_id="planner-child-2")

        self.assertEqual(calls, 1)
        self.assertEqual(second, first)
        self.assertEqual(binding.audit_receipt()["event_id"], "prediction-event-1")

    def test_raw_native_sample_provider_requires_host_admission(self) -> None:
        native = DshNativeAgentRuntimeClient(
            "http://127.0.0.1:9",
            token="unused-test-token",
        )
        adapter = DshSampleCollaborationAdapter(
            run_id="run:raw-native-provider",
            runtime_provider=lambda: native,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
        )

        with self.assertRaisesRegex(
            DshNativeRuntimeUnavailableError,
            "Host-local admission service",
        ):
            adapter._decision_client.sample_decide(
                "dsh/strategy",
                role="planner",
                samples=(),
                context={},
                available_tools=(),
            )

    def test_dsh_runtime_5xx_is_a_retryable_remote_failure(self) -> None:
        error = DshNativeRuntimeUnavailableError(
            "runtime temporarily unavailable",
            error_code="dsh_native_runtime_http_error",
            status_code=502,
        )

        self.assertEqual(
            classify_sample_failure(error),
            ("remote_transient", True, "DshNativeRuntimeUnavailableError"),
        )

    def test_retryable_dsh_critic_failure_crosses_the_sample_boundary(self) -> None:
        class TransientCriticRuntime(_SampleRuntime):
            def run_stage(self, request: dict) -> dict:
                if request["stage"] == "sample.critic":
                    raise DshNativeRuntimeUnavailableError(
                        "runtime temporarily unavailable",
                        error_code="dsh_native_runtime_http_error",
                        status_code=502,
                    )
                return super().run_stage(request)

        runtime = TransientCriticRuntime()
        adapter = DshSampleCollaborationAdapter(
            run_id="run-transient-critic",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
        )
        plan = adapter.plan_batch(
            {
                "run_id": "run-transient-critic",
                "candidate_id": "candidate-1",
                "dataset_digest": "d" * 64,
                "algorithm_id": "registered-predictor",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(DshNativeRuntimeUnavailableError):
            adapter.predict_samples(
                (_request("sample-transient-critic"),),
                (plan,),
                attempts=(1,),
            )

    def test_critic_retry_replays_planner_and_durable_tool_result(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        run_id = "run-persisted-planner-retry"
        ledger.append(run_id, "RunCreated", {"test": True})
        service = DshToolService(ledger)
        native = _PersistingRetryRuntime(service, ledger)
        forecast_calls = 0

        def forecast_bundle(requests):
            nonlocal forecast_calls
            forecast_calls += 1
            return _constant_forecast_bundle(21.5)(requests)

        adapter = DshSampleCollaborationAdapter(
            run_id=run_id,
            runtime_provider=lambda: DshStructuredRoleRuntime(
                native,
                admission=service,
            ),
            revision_provider=lambda _run_id: {
                "run_state_revision": ledger.latest_seq(),
                "ledger_expected_revision": ledger.latest_seq(),
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=forecast_bundle,
            prediction_tool_binder=service.bind_prediction_tool,
        )
        plan = adapter.plan_batch(
            {
                "run_id": run_id,
                "candidate_id": "candidate-1",
                "dataset_digest": "d" * 64,
                "algorithm_id": "registered-predictor",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(DshNativeRuntimeUnavailableError):
            adapter.predict_samples(
                (_request("sample-retry"),),
                (plan,),
                attempts=(1,),
            )
        outcome = adapter.predict_samples(
            (_request("sample-retry"),),
            (plan,),
            attempts=(1,),
        )[0]

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.result["predicted"], 21.5)
        self.assertEqual(
            [request["stage"] for request in native.requests],
            ["sample.plan", "sample.critic", "sample.critic"],
        )
        self.assertEqual(forecast_calls, 2)
        kinds = [event.kind for event in ledger.events(run_id)]
        self.assertEqual(kinds.count("DshPredictionToolExecuted"), 1)
        self.assertEqual(
            sum(
                event.kind == "DshStructuredResultAccepted"
                and event.payload["identity"]["stage"] == "sample.plan"
                for event in ledger.events(run_id)
            ),
            1,
        )

    def test_pre_tool_remote_retry_rebinds_the_registered_prediction_tool(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        run_id = "run-pre-tool-retry-binding"
        ledger.append(run_id, "RunCreated", {"test": True})
        service = DshToolService(ledger)
        native = _PersistingRetryRuntime(service, ledger)
        native.fail_next_critic = False
        forecast_calls = 0

        def forecast_bundle(requests):
            nonlocal forecast_calls
            forecast_calls += 1
            return _constant_forecast_bundle(21.5)(requests)

        adapter = DshSampleCollaborationAdapter(
            run_id=run_id,
            runtime_provider=lambda: DshStructuredRoleRuntime(
                native,
                admission=service,
            ),
            revision_provider=lambda _run_id: {
                "run_state_revision": ledger.latest_seq(),
                "ledger_expected_revision": ledger.latest_seq(),
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=forecast_bundle,
            prediction_tool_binder=service.bind_prediction_tool,
        )
        plan = adapter.plan_batch(
            {
                "run_id": run_id,
                "candidate_id": "candidate-1",
                "dataset_digest": "d" * 64,
                "algorithm_id": "registered-predictor",
                "algorithm_version": "1",
            }
        )

        outcome = adapter.predict_samples(
            (_request("sample-pre-tool-retry"),),
            (plan,),
            attempts=(2,),
        )[0]

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.result["predicted"], 21.5)
        self.assertEqual(forecast_calls, 1)
        planner_context = native.requests[0]["request"]["context"]
        self.assertEqual(planner_context["role"], "repair")
        self.assertEqual(
            [item["tool_id"] for item in planner_context["available_tools"]],
            ["registered-predictor"],
        )
        self.assertEqual(
            [event.kind for event in ledger.events(run_id)].count(
                "DshPredictionToolExecuted"
            ),
            1,
        )

    def test_single_registered_prediction_tool_still_uses_remote_agents(self) -> None:
        runtime = _SampleRuntime()
        adapter = DshSampleCollaborationAdapter(
            run_id="run-deterministic",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
            remote_critic_policy={
                "version": "uncertain_or_failure@1",
                "min_planner_confidence": 0.9,
            },
        )
        plan = adapter.plan_batch(
            {
                "run_id": "run-deterministic",
                "candidate_id": "candidate-1",
                "dataset_digest": "d" * 64,
                "algorithm_id": "registered-predictor",
                "algorithm_version": "1",
            }
        )

        outcome = adapter.predict_samples(
            (_request("sample-deterministic"),), (plan,), attempts=(1,)
        )[0]

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.result["predicted"], 21.5)
        self.assertEqual(
            [item["stage"] for item in runtime.requests],
            ["sample.plan"],
        )
        self.assertEqual(
            [item["role"] for item in outcome.result["agent_decisions"]],
            ["remote_planner_agent"],
        )

    def test_planner_exposes_only_the_frozen_candidate_tool(self) -> None:
        runtime = _SampleRuntime()
        adapter = DshSampleCollaborationAdapter(
            run_id="run-multi-tool",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
            remote_critic_policy={
                "version": "uncertain_or_failure@1",
                "min_planner_confidence": 0.9,
            },
        )
        plan = adapter.plan_batch(
            {
                "run_id": "run-multi-tool",
                "candidate_id": "candidate-1",
                "dataset_digest": "d" * 64,
                "algorithm_id": "registered-predictor",
                "algorithm_version": "1",
            }
        )

        outcome = adapter.predict_samples(
            (_request("sample-multi-tool"),), (plan,), attempts=(1,)
        )[0]

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.result["predicted"], 21.5)
        self.assertEqual(
            [item["stage"] for item in runtime.requests],
            ["sample.plan"],
        )
        planner_tools = runtime.requests[0]["request"]["context"][
            "available_tools"
        ]
        self.assertEqual(
            [item["tool_id"] for item in planner_tools],
            ["registered-predictor"],
        )

    def test_vector_tool_failure_fails_closed_before_critic(self) -> None:
        runtime = _FailureReviewRuntime()

        def fail_forecast(_requests):
            raise ArithmeticError("synthetic registered predictor failure")

        adapter = DshSampleCollaborationAdapter(
            run_id="run-failure-review",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=fail_forecast,
            prediction_tool_binder=_fake_agent_prediction_binder,
            remote_critic_policy={
                "version": "uncertain_or_failure@1",
                "min_planner_confidence": 0.9,
            },
        )
        plan = adapter.plan_batch(
            {
                "run_id": "run-failure-review",
                "candidate_id": "candidate-1",
                "dataset_digest": "d" * 64,
                "algorithm_id": "registered-predictor",
                "algorithm_version": "1",
            }
        )

        outcome = adapter.predict_samples(
            (_request("sample-failure-review"),), (plan,), attempts=(1,)
        )[0]

        self.assertIsNotNone(outcome.error)
        self.assertFalse(classify_sample_failure(outcome.error)[1])
        self.assertEqual(
            [item["stage"] for item in runtime.requests],
            [],
        )

    def test_registry_routes_native_mode_without_model_gateway(self) -> None:
        task = TaskManifest(
            task_id="native-samples",
            objective="route samples through DSH",
            domain_pack="greenhouse_environment@1",
            visible_datasets=("agc_cucumber_2018",),
            budget={"max_candidates": 1},
            metadata={
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "sample_agent_mode": "dsh_native_workflow",
                "sample_agent_protocol": "dsh-strict-origin-bundle@3",
                "sample_agent_batch_size": 8,
                "sample_concurrency": 2,
                "prediction_cells_per_origin": 1,
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
            },
        )
        registry = EvaluatorRegistry(
            DatasetRegistry(),
            object(),
            dsh_runtime_provider=lambda: _SampleRuntime(),
            dsh_revision_provider=lambda _run_id: {
                "run_state_revision": 1,
                "ledger_expected_revision": 1,
            },
            dsh_identity_provider=lambda _run_id, _candidate_id: {
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            dsh_prediction_tool_binder=_fake_agent_prediction_binder,
        )
        executor = registry._sample_executor_for_task(
            task,
            run_id="run-1",
            candidate_id="candidate-1",
            forecast_bundle_tool=_constant_forecast_bundle(1.0),
        )
        self.assertIsInstance(executor.adapter, DshSampleCollaborationAdapter)

    def test_planner_host_tool_and_independent_critic_preserve_result_contract(self) -> None:
        runtime = _SampleRuntime()
        adapter = DshSampleCollaborationAdapter(
            run_id="run-1",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
            microbatch_size=8,
            sample_concurrency=2,
        )
        context = {
            "run_id": "run-1",
            "candidate_id": "candidate-1",
            "dataset_digest": "d" * 64,
            "partition": "training_feedback",
            "algorithm_id": "registered-predictor",
            "algorithm_version": "1",
            "strategy_model_id": "dsh/strategy",
            "review_model_id": "dsh/review",
        }
        plan = adapter.plan_batch(context)
        outcome = adapter.predict_samples((_request("sample-1"),), (plan,), attempts=(1,))[0]

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.result["predicted"], 21.5)
        self.assertEqual(
            [item["stage"] for item in runtime.requests],
            ["sample.plan", "sample.critic"],
        )
        encoded = json.dumps(runtime.requests)
        self.assertNotIn("observed", encoded)
        self.assertNotIn("ground_truth", encoded)
        self.assertNotIn("max_tokens", encoded)
        critic_sample = runtime.requests[1]["request"]["context"]["samples"][0]
        self.assertNotIn("sample", critic_sample)
        self.assertNotIn("history_window", json.dumps(critic_sample))
        self.assertEqual(critic_sample["baseline"], 20.0)
        self.assertEqual(critic_sample["predicted"], 21.5)
        self.assertEqual(adapter.adapter_id, "dsh-native-sample-collaboration")
        self.assertEqual(
            runtime.requests[1]["request"]["context"]["allowed_reason_codes"],
            sorted(REMOTE_REASON_CODES),
        )

    def test_executor_runs_one_post_score_reflection_per_origin(self) -> None:
        runtime = _SampleRuntime()
        adapter = DshSampleCollaborationAdapter(
            run_id="run-reflection",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
        )
        batch = CollaborativeSampleExecutor(adapter).execute(
            (
                {
                    "sample_id": "sample-reflection",
                    "candidate_id": "candidate-1",
                    "dataset_digest": "d" * 64,
                    "partition": "training_feedback",
                    "target": "air_temperature",
                    "unit": "degC",
                    "horizon_hours": 1,
                    "origin_timestamp": 10,
                    "target_timestamp": 11,
                    "baseline": 20.0,
                    "observed": 21.0,
                    "label_free_context": _request("sample-reflection").label_free_context,
                },
            ),
            context={
                "run_id": "run-reflection",
                "candidate_id": "candidate-1",
                "dataset_digest": "d" * 64,
                "partition": "training_feedback",
                "algorithm_id": "registered-predictor",
                "algorithm_version": "1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
            },
            target_bounds={
                "air_temperature": {"minimum": -20.0, "maximum": 80.0}
            },
            algorithm_id="registered-predictor",
            algorithm_version="1",
        )

        self.assertEqual(
            [item["stage"] for item in runtime.requests],
            ["sample.plan", "sample.critic", "sample.reflect"],
        )
        self.assertTrue(batch.summary["strict_agent_chain_pass"])
        self.assertEqual(batch.summary["complete_agent_chains"], 1)
        self.assertEqual(batch.summary["host_route_bypass_count"], 0)
        reflection_context = runtime.requests[2]["request"]["context"]
        self.assertEqual(
            reflection_context["outcome"]["cells"][0]["observed"], 21.0
        )
        self.assertEqual(
            reflection_context["outcome"]["cells"][0]["predicted"], 21.5
        )

    def test_normal_origin_uses_planner_only_under_sparse_review_policy(self) -> None:
        runtime = _SampleRuntime()
        adapter = DshSampleCollaborationAdapter(
            run_id="run-planner-only",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
            remote_critic_policy={
                "version": "uncertain_or_failure@1",
                "min_planner_confidence": 0.5,
            },
            sample_reflection_policy="candidate_aggregate_post_score@1",
        )
        request = _request("sample-planner-only")
        row = request.to_dict()
        row["observed"] = 21.0

        batch = CollaborativeSampleExecutor(adapter).execute(
            (row,),
            context={
                "run_id": "run-planner-only",
                "candidate_id": "candidate-1",
                "dataset_digest": "d" * 64,
                "partition": "training_feedback",
                "algorithm_id": "registered-predictor",
                "algorithm_version": "1",
            },
            target_bounds={
                "air_temperature": {"minimum": -20.0, "maximum": 80.0}
            },
            algorithm_id="registered-predictor",
            algorithm_version="1",
        )

        self.assertEqual(
            [item["stage"] for item in runtime.requests],
            ["sample.plan"],
        )
        self.assertEqual(batch.summary["remote_planner_invocations"], 1)
        self.assertEqual(batch.summary["remote_critic_invocations"], 0)
        self.assertEqual(batch.summary["remote_reflection_invocations"], 0)
        self.assertTrue(batch.summary["strict_agent_chain_pass"])

    def test_origin_bundle_predicts_all_nine_cells_in_one_agent_chain(self) -> None:
        runtime = _SampleRuntime()
        bundle_calls: list[list[str]] = []
        progress: list[dict] = []

        def forecast_bundle(requests):
            bundle_calls.append([request.sample_id for request in requests])
            return {
                request.sample_id: {
                    "predicted": request.baseline + 0.5,
                    "metadata": {"source_model_id": "registered-predictor"},
                }
                for request in requests
            }

        adapter = DshSampleCollaborationAdapter(
            run_id="run-origin-bundle",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=forecast_bundle,
            prediction_tool_binder=_fake_agent_prediction_binder,
            progress_callback=progress.append,
            microbatch_size=9,
        )
        targets = (
            ("air_temperature", "degC", 20.0),
            ("relative_humidity", "percent", 60.0),
            ("co2_concentration", "ppm", 600.0),
        )
        rows = []
        for target, unit, baseline in targets:
            for horizon in (1, 6, 24):
                rows.append(
                    {
                        "partition": "training_feedback",
                        "target": target,
                        "unit": unit,
                        "horizon_hours": horizon,
                        "origin_timestamp": 100,
                        "target_timestamp": 100 + horizon,
                        "baseline": baseline,
                        "observed": baseline + 1.0,
                        "label_free_context": {
                            "schema_version": (
                                "ecologyrsi-dsh.label-free-sample-context/1"
                            ),
                            "history_window": [baseline],
                            "causal_provenance": {
                                "schema_version": (
                                    "ecologyrsi-dsh.causal-sample-provenance/1"
                                ),
                                "origin_cutoff_timestamp": 100,
                                "latest_context_timestamp": 100,
                                "history_timestamps": [100],
                            },
                        },
                    }
                )
        publications: list[list[dict]] = []
        batch = CollaborativeSampleExecutor(adapter).execute(
            rows,
            context={
                "run_id": "run-origin-bundle",
                "candidate_id": "candidate-1",
                "dataset_digest": "d" * 64,
                "partition": "training_feedback",
                "algorithm_id": "registered-predictor",
                "algorithm_version": "1",
            },
            target_bounds={
                "air_temperature": {
                    "unit": "degC",
                    "minimum": -20.0,
                    "maximum": 80.0,
                },
                "relative_humidity": {
                    "unit": "percent",
                    "minimum": 0.0,
                    "maximum": 100.0,
                },
                "co2_concentration": {
                    "unit": "ppm",
                    "minimum": 0.0,
                    "maximum": 5000.0,
                },
            },
            algorithm_id="registered-predictor",
            algorithm_version="1",
            result_callback=lambda values: publications.append(
                [dict(item) for item in values]
            ),
        )

        self.assertEqual(len(bundle_calls), 1)
        self.assertEqual(len(bundle_calls[0]), 9)
        self.assertEqual(
            [item["stage"] for item in runtime.requests],
            ["sample.plan", "sample.critic", "sample.reflect"],
        )
        self.assertEqual(
            len(runtime.requests[0]["request"]["context"]["samples"]), 9
        )
        self.assertEqual(
            runtime.requests[0]["request"]["context"]["available_tools"],
            [
                {
                    "tool_id": "registered-predictor",
                    "version": "1",
                    "purpose": "registered_candidate_prediction",
                }
            ],
        )
        reflection_context = runtime.requests[2]["request"]["context"]
        self.assertEqual(
            len(reflection_context["sample"]["prediction_cells"]), 9
        )
        self.assertEqual(len(reflection_context["outcome"]["cells"]), 9)
        self.assertEqual([len(item) for item in publications], [9])
        self.assertEqual(len(batch.scoring_rows), 9)
        self.assertEqual(batch.summary["attempted_origin_samples"], 1)
        self.assertEqual(batch.summary["prediction_cell_count"], 9)
        self.assertEqual(batch.summary["prediction_cells_per_origin"], 9)
        self.assertEqual(batch.summary["complete_origin_agent_chains"], 1)
        self.assertEqual(batch.summary["remote_planner_invocations"], 1)
        self.assertEqual(batch.summary["remote_critic_invocations"], 1)
        self.assertEqual(batch.summary["remote_reflection_invocations"], 1)
        self.assertEqual(
            batch.summary["reflection_outcome_counts"], {"neutral": 1}
        )
        self.assertEqual(
            batch.summary["reflection_error_source_counts"], {"unknown": 1}
        )
        self.assertEqual(
            batch.summary["reflection_next_action_counts"], {"keep": 1}
        )
        self.assertEqual(
            batch.summary["registered_prediction_tool_invocations"], 1
        )
        self.assertEqual(
            batch.summary["dsh_agent_prediction_tool_invocations"], 1
        )
        self.assertTrue(batch.summary["strict_agent_chain_pass"])
        completed = [
            item for item in progress if item["progress_kind"] == "completed_batch"
        ]
        self.assertEqual(
            [(item["completed_samples"], item["total_samples"]) for item in completed],
            [(1, 1)],
        )

    def test_v4_origin_protocol_runs_bounded_origin_chains_concurrently(self) -> None:
        class ConcurrentRuntime(_SampleRuntime):
            def __init__(self) -> None:
                super().__init__()
                self.lock = threading.Lock()
                self.release = threading.Event()
                self.active_planners = 0
                self.maximum_planners = 0

            def run_stage(self, request: dict) -> dict:
                if request["stage"] == "sample.plan":
                    with self.lock:
                        self.active_planners += 1
                        self.maximum_planners = max(
                            self.maximum_planners, self.active_planners
                        )
                        if self.active_planners == 2:
                            self.release.set()
                    self.release.wait(timeout=0.25)
                    try:
                        return super().run_stage(request)
                    finally:
                        with self.lock:
                            self.active_planners -= 1
                return super().run_stage(request)

        runtime = ConcurrentRuntime()
        adapter = DshSampleCollaborationAdapter(
            run_id="run-origin-concurrency",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 1,
                "ledger_expected_revision": 1,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
            sample_concurrency=2,
            microbatch_size=9,
        )
        rows = [
            {
                "partition": "training_feedback",
                "target": "air_temperature",
                "unit": "degC",
                "horizon_hours": 1,
                "origin_timestamp": origin,
                "target_timestamp": origin + 1,
                "baseline": 20.0,
                "observed": 21.0,
                "label_free_context": {
                    "schema_version": "ecologyrsi-dsh.label-free-sample-context/1",
                    "history_window": [20.0],
                    "causal_provenance": {
                        "schema_version": "ecologyrsi-dsh.causal-sample-provenance/1",
                        "origin_cutoff_timestamp": origin,
                        "latest_context_timestamp": origin,
                        "history_timestamps": [origin],
                    },
                },
            }
            for origin in (100, 200)
        ]

        batch = CollaborativeSampleExecutor(adapter).execute(
            rows,
            context={
                "run_id": "run-origin-concurrency",
                "candidate_id": "candidate-1",
                "dataset_digest": "d" * 64,
                "partition": "training_feedback",
                "algorithm_id": "registered-predictor",
                "algorithm_version": "1",
                "sample_concurrency": 2,
                "candidate_concurrency": 1,
            },
            target_bounds={
                "air_temperature": {
                    "unit": "degC",
                    "minimum": -20.0,
                    "maximum": 80.0,
                }
            },
            algorithm_id="registered-predictor",
            algorithm_version="1",
        )

        self.assertEqual(adapter.plan_batch({})["sample_agent_protocol"], "dsh-strict-origin-bundle@4")
        self.assertEqual(runtime.maximum_planners, 2)
        self.assertEqual(batch.summary["attempted_origin_samples"], 2)

    def test_v4_origin_protocol_publishes_completed_origin_without_waiting_for_slow_predecessor(
        self,
    ) -> None:
        class OutOfOrderRuntime(_SampleRuntime):
            def __init__(self) -> None:
                super().__init__()
                self.first_started = threading.Event()
                self.release_first = threading.Event()

            def run_stage(self, request: dict) -> dict:
                sample = request["request"]["context"].get("samples", [{}])[0]
                target_timestamp = sample.get("target_timestamp")
                if target_timestamp is None and isinstance(
                    sample.get("sample"), dict
                ):
                    target_timestamp = sample["sample"].get("target_timestamp")
                if (
                    request["stage"] == "sample.plan"
                    and target_timestamp == 101
                ):
                    self.first_started.set()
                    self.release_first.wait(timeout=2.0)
                return super().run_stage(request)

        runtime = OutOfOrderRuntime()
        adapter = DshSampleCollaborationAdapter(
            run_id="run-origin-completion-order",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 1,
                "ledger_expected_revision": 1,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
            sample_concurrency=2,
            microbatch_size=9,
        )
        rows = [
            {
                "partition": "training_feedback",
                "target": "air_temperature",
                "unit": "degC",
                "horizon_hours": 1,
                "origin_timestamp": origin,
                "target_timestamp": origin + 1,
                "baseline": 20.0,
                "observed": 21.0,
                "label_free_context": {
                    "schema_version": "ecologyrsi-dsh.label-free-sample-context/1",
                    "history_window": [20.0],
                    "causal_provenance": {
                        "schema_version": "ecologyrsi-dsh.causal-sample-provenance/1",
                        "origin_cutoff_timestamp": origin,
                        "latest_context_timestamp": origin,
                        "history_timestamps": [origin],
                    },
                },
            }
            for origin in (100, 200)
        ]
        completed_origin_published = threading.Event()
        publications: list[int] = []
        execution_errors: list[BaseException] = []

        def publish(finalized_rows) -> None:
            origin = int(finalized_rows[0]["origin_timestamp"])
            publications.append(origin)
            completed_origin_published.set()

        def execute() -> None:
            try:
                CollaborativeSampleExecutor(adapter).execute(
                    rows,
                    context={
                        "run_id": "run-origin-completion-order",
                        "candidate_id": "candidate-1",
                        "dataset_digest": "d" * 64,
                        "partition": "training_feedback",
                        "algorithm_id": "registered-predictor",
                        "algorithm_version": "1",
                        "sample_concurrency": 2,
                        "candidate_concurrency": 1,
                    },
                    target_bounds={
                        "air_temperature": {
                            "unit": "degC",
                            "minimum": -20.0,
                            "maximum": 80.0,
                        }
                    },
                    algorithm_id="registered-predictor",
                    algorithm_version="1",
                    result_callback=publish,
                )
            except Exception as exc:  # pragma: no cover - asserted below
                execution_errors.append(exc)

        worker = threading.Thread(target=execute)
        worker.start()
        first_started = runtime.first_started.wait(timeout=1.0)
        published_before_release = (
            completed_origin_published.wait(timeout=0.5) if first_started else False
        )
        runtime.release_first.set()
        worker.join(timeout=3.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(execution_errors, [])
        self.assertTrue(first_started, runtime.requests)
        self.assertTrue(published_before_release)
        self.assertEqual(sorted(publications), [100, 200])

    def test_origin_bundle_progress_does_not_mix_cell_split_counts_with_origins(
        self,
    ) -> None:
        progress: list[dict] = []
        adapter = DshSampleCollaborationAdapter(
            run_id="run-origin-progress",
            runtime_provider=lambda: _SampleRuntime(),
            revision_provider=lambda _run_id: {
                "run_state_revision": 1,
                "ledger_expected_revision": 1,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
            progress_callback=progress.append,
            microbatch_size=9,
        )
        adapter.set_resume_checkpoint(
            {
                "completed_samples": 0,
                "succeeded_samples": 0,
                "failed_samples": 0,
                "total_samples": 9,
                "completed_origin_samples": 0,
                "succeeded_origin_samples": 0,
                "failed_origin_samples": 0,
                "total_origin_samples": 2,
                "batch_index": 0,
                "batch_count": 9,
                "progress_id": 0,
                "gateway_request_count": 0,
                "adaptive_split_trigger_count": 0,
                "adaptive_split_count": 0,
                "adaptive_split_max_depth": 0,
                "adaptive_split_recovered_samples": 0,
                "adaptive_split_failed_samples": 0,
                "tasks": [
                    {
                        "target": "origin-vector",
                        "horizon_hours": 1,
                        "total_samples": 9,
                        "resumed_failed_samples": 0,
                    }
                ],
            }
        )

        adapter._handle_strict_gateway_progress(
            {
                "schema_version": "ecologyrsi-dsh.sample-microbatch-progress/3",
                "role": "planner",
                "model_id": "dsh/strategy",
                "progress_id": 1,
                "progress_kind": "completed_batch",
                "batch_index": 1,
                "batch_count": 1,
                "batch_size": 9,
                "completed_samples": 9,
                "total_samples": 9,
                "succeeded_samples": 9,
                "failed_samples": 0,
                "in_flight_batches": 0,
                "queued_batches": 0,
                "gateway_request_count": 3,
                "adaptive_split_trigger_count": 1,
                "adaptive_split_count": 1,
                "adaptive_split_max_depth": 1,
                "adaptive_split_recovered_samples": 9,
                "adaptive_split_failed_samples": 0,
            }
        )

        self.assertEqual(progress[-1]["completed_samples"], 0)
        self.assertEqual(progress[-1]["total_samples"], 2)
        self.assertEqual(progress[-1]["adaptive_split_trigger_count"], 1)
        self.assertEqual(progress[-1]["adaptive_split_count"], 1)
        self.assertEqual(progress[-1]["adaptive_split_recovered_samples"], 0)
        self.assertEqual(progress[-1]["adaptive_split_failed_samples"], 0)
        self.assertEqual(progress[-1]["in_flight_batches"], 0)
        self.assertEqual(progress[-1]["queued_batches"], 0)
        self.assertEqual(progress[-1]["awaiting_submission_batches"], 2)

        adapter.record_finalized_origin_progress(
            status="succeeded", prediction_cell_count=9
        )
        self.assertEqual(progress[-1]["completed_samples"], 1)
        self.assertEqual(progress[-1]["succeeded_samples"], 1)
        self.assertEqual(progress[-1]["adaptive_split_recovered_samples"], 0)
        self.assertEqual(progress[-1]["adaptive_split_failed_samples"], 0)
        self.assertEqual(progress[-1]["in_flight_batches"], 0)
        self.assertEqual(progress[-1]["queued_batches"], 0)
        self.assertEqual(progress[-1]["awaiting_submission_batches"], 1)

    def test_strict_progress_uses_run_admission_snapshot_for_actual_in_flight(
        self,
    ) -> None:
        progress: list[dict] = []
        adapter = DshSampleCollaborationAdapter(
            run_id="run-origin-admission-progress",
            runtime_provider=lambda: _SampleRuntime(),
            revision_provider=lambda _run_id: {
                "run_state_revision": 1,
                "ledger_expected_revision": 1,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
            progress_callback=progress.append,
            admission_snapshot_provider=lambda: {
                "limit": 64,
                "active": 3,
                "waiting": 4,
            },
        )
        adapter.set_resume_checkpoint(
            {
                "completed_samples": 0,
                "succeeded_samples": 0,
                "failed_samples": 0,
                "total_samples": 90,
                "completed_origin_samples": 0,
                "succeeded_origin_samples": 0,
                "failed_origin_samples": 0,
                "total_origin_samples": 10,
                "batch_index": 0,
                "batch_count": 90,
                "progress_id": 0,
                "gateway_request_count": 0,
                "adaptive_split_trigger_count": 0,
                "adaptive_split_count": 0,
                "adaptive_split_max_depth": 0,
                "adaptive_split_recovered_samples": 0,
                "adaptive_split_failed_samples": 0,
                "tasks": [
                    {
                        "target": "origin-vector",
                        "horizon_hours": 1,
                        "total_samples": 90,
                        "resumed_failed_samples": 0,
                    }
                ],
            }
        )

        adapter._handle_strict_gateway_progress(
            {
                "progress_kind": "waiting",
                "in_flight_batches": 1,
            }
        )

        self.assertEqual(progress[-1]["in_flight_batches"], 3)
        self.assertEqual(progress[-1]["queued_batches"], 0)
        self.assertEqual(progress[-1]["awaiting_submission_batches"], 7)

        adapter.record_finalized_origin_progress(
            status="succeeded",
            prediction_cell_count=9,
        )

        self.assertEqual(progress[-1]["completed_samples"], 1)
        self.assertEqual(progress[-1]["in_flight_batches"], 3)
        self.assertEqual(progress[-1]["queued_batches"], 0)
        self.assertEqual(progress[-1]["awaiting_submission_batches"], 6)

    def test_strict_executor_streams_and_checkpoints_each_complete_chain(
        self,
    ) -> None:
        runtime = _SampleRuntime()
        progress: list[dict] = []
        progress_stage_counts: list[int] = []

        def record_progress(item) -> None:
            progress.append(item)
            if (
                item["role"] == "planner"
                and item["progress_kind"] == "completed_batch"
            ):
                progress_stage_counts.append(len(runtime.requests))

        adapter = DshSampleCollaborationAdapter(
            run_id="run-streaming-chain",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
            progress_callback=record_progress,
            sample_concurrency=1,
        )
        rows = []
        for index in (1, 2):
            row = _request(f"sample-stream-{index}").to_dict()
            origin = 10 + index
            row.update(
                {
                    "sample_index": index,
                    "origin_timestamp": origin,
                    "target_timestamp": origin + 1,
                    "observed": 21.0 + index,
                    "label_free_context": {
                        **row["label_free_context"],
                        "causal_provenance": {
                            **row["label_free_context"]["causal_provenance"],
                            "origin_cutoff_timestamp": origin,
                            "latest_context_timestamp": origin,
                            "history_timestamps": [origin - 1, origin],
                        },
                    },
                }
            )
            rows.append(row)
        publications: list[tuple[int, str]] = []

        def publish(finalized_rows) -> None:
            publications.append(
                (len(runtime.requests), str(finalized_rows[0]["sample_id"]))
            )

        batch = CollaborativeSampleExecutor(adapter).execute(
            tuple(rows),
            context={
                "run_id": "run-streaming-chain",
                "candidate_id": "candidate-1",
                "dataset_digest": "d" * 64,
                "partition": "training_feedback",
                "algorithm_id": "registered-predictor",
                "algorithm_version": "1",
            },
            target_bounds={
                "air_temperature": {"minimum": -20.0, "maximum": 80.0}
            },
            algorithm_id="registered-predictor",
            algorithm_version="1",
            result_callback=publish,
            result_batch_size=50,
        )

        self.assertEqual(
            [item["stage"] for item in runtime.requests],
            [
                "sample.plan",
                "sample.critic",
                "sample.reflect",
                "sample.plan",
                "sample.critic",
                "sample.reflect",
            ],
        )
        self.assertEqual([item[0] for item in publications], [3, 6])
        self.assertEqual(progress_stage_counts, [3, 6])
        self.assertTrue(
            all(
                item["batch_size"] == 0
                for item in progress
                if item["progress_kind"] != "completed_batch"
            )
        )
        self.assertEqual(len({item[1] for item in publications}), 2)
        completed = [
            item
            for item in progress
            if item["role"] == "planner"
            and item["progress_kind"] == "completed_batch"
        ]
        self.assertEqual(
            [
                (
                    item["batch_index"],
                    item["completed_samples"],
                    item["total_samples"],
                    item["in_flight_batches"],
                    item["queued_batches"],
                    item["awaiting_submission_batches"],
                )
                for item in completed
            ],
            [(1, 1, 2, 0, 0, 1), (2, 2, 2, 0, 0, 0)],
        )
        self.assertEqual(batch.summary["complete_agent_chains"], 2)
        self.assertTrue(batch.summary["strict_agent_chain_pass"])

    def test_strict_checkpoint_is_published_only_after_reflection_and_resumes_proof(
        self,
    ) -> None:
        runtime = _SampleRuntime()
        adapter = DshSampleCollaborationAdapter(
            run_id="run-checkpoint",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
        )
        row = {
            "sample_index": 1,
            "candidate_id": "candidate-1",
            "dataset_digest": "d" * 64,
            "partition": "training_feedback",
            "target": "air_temperature",
            "unit": "degC",
            "horizon_hours": 1,
            "origin_timestamp": 10,
            "target_timestamp": 11,
            "baseline": 20.0,
            "observed": 21.0,
            "label_free_context": _request("sample-checkpoint").label_free_context,
        }
        context = {
            "run_id": "run-checkpoint",
            "candidate_id": "candidate-1",
            "dataset_digest": "d" * 64,
            "partition": "training_feedback",
            "algorithm_id": "registered-predictor",
            "algorithm_version": "1",
        }
        persisted: list[dict] = []
        publication_stage_counts: list[int] = []

        def publish(rows) -> None:
            publication_stage_counts.append(len(runtime.requests))
            persisted.extend(build_sample_results("candidate-1", rows))

        first = CollaborativeSampleExecutor(adapter).execute(
            (row,),
            context=context,
            target_bounds={
                "air_temperature": {"minimum": -20.0, "maximum": 80.0}
            },
            algorithm_id="registered-predictor",
            algorithm_version="1",
            result_callback=publish,
            checkpoint_callback=lambda _checkpoint: {"rows": ()},
        )

        self.assertEqual(publication_stage_counts, [3])
        self.assertTrue(persisted[0]["sample_agent_chain"]["complete"])
        self.assertTrue(first.summary["strict_agent_chain_pass"])
        request_count = len(runtime.requests)

        resumed = CollaborativeSampleExecutor(adapter).execute(
            (row,),
            context=context,
            target_bounds={
                "air_temperature": {"minimum": -20.0, "maximum": 80.0}
            },
            algorithm_id="registered-predictor",
            algorithm_version="1",
            checkpoint_callback=lambda _checkpoint: {"rows": tuple(persisted)},
        )

        self.assertEqual(len(runtime.requests), request_count)
        self.assertEqual(resumed.summary["checkpoint_resumed_examples"], 1)
        self.assertEqual(resumed.summary["complete_agent_chains"], 1)
        self.assertTrue(resumed.summary["strict_agent_chain_pass"])

    def test_strict_critic_failure_never_returns_a_prediction(self) -> None:
        runtime = _CriticUnavailableRuntime()
        adapter = DshSampleCollaborationAdapter(
            run_id="run-critic-failure",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
        )
        plan = adapter.plan_batch(
            {
                "run_id": "run-critic-failure",
                "candidate_id": "candidate-1",
                "dataset_digest": "d" * 64,
                "algorithm_id": "registered-predictor",
                "algorithm_version": "1",
            }
        )

        outcome = adapter.predict_samples(
            (_request("sample-critic-failure"),), (plan,), attempts=(1,)
        )[0]

        self.assertIsNone(outcome.result)
        self.assertIsNotNone(outcome.error)
        self.assertEqual(
            [item["stage"] for item in runtime.requests],
            ["sample.plan", "sample.critic"],
        )

    def test_dsh_reason_code_schemas_exactly_match_host_enum(self) -> None:
        schema_root = (
            Path(__file__).resolve().parents[1]
            / "integrations"
            / "dsh_ecology_plugin"
            / "schemas"
        )
        for name in ("sample-decisions", "sample-review"):
            with self.subTest(schema=name):
                schema = json.loads((schema_root / f"{name}.schema.json").read_text())
                reason = schema["properties"]["decisions"]["items"]["properties"][
                    "reason_code"
                ]
                self.assertEqual(set(reason["enum"]), set(REMOTE_REASON_CODES))

        reflection = json.loads(
            (schema_root / "sample-reflection.schema.json").read_text()
        )
        self.assertEqual(
            reflection["properties"]["schema_version"]["const"],
            "ecology-sample-reflection@1",
        )
        self.assertIn("wave_digest", reflection["required"])
        self.assertIn("sample_id", reflection["required"])

    def test_strict_route_keeps_prediction_compatible_without_host_bypass(self) -> None:
        runtime = _SampleRuntime()
        dsh = DshSampleCollaborationAdapter(
            run_id="run-1",
            runtime_provider=lambda: runtime,
            revision_provider=lambda _run_id: {
                "run_state_revision": 7,
                "ledger_expected_revision": 11,
            },
            identity_digests={
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            forecast_bundle_tool=_constant_forecast_bundle(21.5),
            prediction_tool_binder=_fake_agent_prediction_binder,
        )
        legacy = GatewaySampleCollaborationAdapter(
            _EquivalentDecisionGateway(),
            strategy_model_id="dsh/strategy",
            review_model_id="dsh/review",
            remote_review_enabled=True,
            forecast_tool=lambda _request: 21.5,
        )
        context = {
            "run_id": "run-1",
            "candidate_id": "candidate-1",
            "dataset_digest": "d" * 64,
            "partition": "training_feedback",
            "algorithm_id": "registered-predictor",
            "algorithm_version": "1",
            "strategy_model_id": "dsh/strategy",
            "review_model_id": "dsh/review",
        }
        request = _request("sample-compatible")
        dsh_result = dsh.predict_sample(request, dsh.plan_batch(context), attempt=1)
        legacy_result = legacy.predict_sample(request, legacy.plan_batch(context), attempt=1)
        self.assertEqual(dsh_result["predicted"], legacy_result["predicted"])
        self.assertEqual(
            [item["tool_id"] for item in dsh_result["tool_calls"]],
            [item["tool_id"] for item in legacy_result["tool_calls"]],
        )
        self.assertEqual(
            dsh_result["agent_decisions"][0]["role"],
            "remote_planner_agent",
        )


if __name__ == "__main__":
    unittest.main()
