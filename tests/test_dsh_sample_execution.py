from __future__ import annotations

import json
from pathlib import Path
import unittest

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.models import TaskManifest
from ecologyrsi_dsh.core.redaction import REMOTE_REASON_CODES
from ecologyrsi_dsh.data.registry import DatasetRegistry
from ecologyrsi_dsh.evaluators.dsh_sample_adapter import (
    DshSampleCollaborationAdapter,
)
from ecologyrsi_dsh.evaluators.gateway_sample_adapter import (
    GatewaySampleCollaborationAdapter,
)
from ecologyrsi_dsh.evaluators.sample_execution import SamplePredictionRequest
from ecologyrsi_dsh.evaluators.registry import EvaluatorRegistry


class _SampleRuntime:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def run_stage(self, request: dict) -> dict:
        self.requests.append(request)
        stage_context = request["request"]["context"]
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
                "sample_agent_batch_size": 8,
                "sample_concurrency": 2,
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
        )
        executor = registry._sample_executor_for_task(
            task,
            run_id="run-1",
            candidate_id="candidate-1",
            forecast_tool=lambda _request: 1.0,
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
            forecast_tool=lambda _request: 21.5,
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
        self.assertEqual([item["stage"] for item in runtime.requests], ["sample.plan", "sample.critic"])
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
            runtime.requests[0]["request"]["context"]["allowed_reason_codes"],
            sorted(REMOTE_REASON_CODES),
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

    def test_same_decisions_keep_prediction_and_reward_inputs_byte_compatible(self) -> None:
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
            forecast_tool=lambda _request: 21.5,
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
        self.assertEqual(
            json.dumps(dsh_result, sort_keys=True, separators=(",", ":")),
            json.dumps(legacy_result, sort_keys=True, separators=(",", ":")),
        )


if __name__ == "__main__":
    unittest.main()
