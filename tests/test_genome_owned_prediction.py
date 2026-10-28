"""Genome-owned prediction: copy-only sample Agents, bounded plans, label-safe panel."""
from __future__ import annotations

from dataclasses import replace
import unittest

from ecologyrsi_dsh.core.agent_prediction import EVIDENCE_FORMULA_POLICY
from ecologyrsi_dsh.core.models import canonical_json, digest
from ecologyrsi_dsh.evaluators.dsh_sample_adapter import DshSampleCollaborationAdapter
from ecologyrsi_dsh.evaluators.sample_execution import _MAX_PLAN_BYTES
from ecologyrsi_dsh.evolution import genome_owned_prediction as ownership
from ecologyrsi_dsh.evolution.agent_policy import build_agent_policy
from ecologyrsi_dsh.science.residual_panel import (
    SCALES, SCHEMA, fit_partition_residual_panel, validated_panel,
)
from tests.agent_prediction_fixtures import agent_binding, model_result
from tests.test_dsh_sample_execution import _SampleRuntime, _constant_forecast_bundle, _request
from tests.test_training_fit_diagnostics import series


IDENTITY = {"genome_digest": "a" * 64, "compiled_behavior_digest": "b" * 64,
            "phenotype_instance_digest": "c" * 64}


def _experience_row(index: int) -> dict:
    return {
        "tool_id": "greenhouse-recipe-ridge@1", "version": "1",
        "parameters": {"ridge_alpha": 0.1, "history_steps": 6},
        "target": "air_temperature", "horizon_hours": (1, 6, 24)[index % 3],
        "parameter_identity_schema": "ecologyrsi-dsh.tool-parameter-identity/1",
        "parameters_semantics": "request_overrides_only",
        "observation_unit": "tool_calls_per_scoring_cell_including_retries",
        "adjustment_gain_semantics": "cited_accepted_attempts_observational_not_causal",
        "resolved_predictor_id": "greenhouse-recipe-ridge@1",
        "effective_parameters_digest": digest({"row": index}),
        "effective_parameters_source": "tool_receipt",
        "parameters_digest": digest({"parameters": index}),
        "source_phase": "formal_batch", "evidence_kind": "raw_tool_output",
        "calls": 17, "failed": 0, "n": 17, "cited": 16, "adjusted_n": 16,
        "mean_tool_latency_ms": 12.3, "rmse": 0.7385682326786831, "mae": 0.5855848462018552,
        "agent_adjustment_mae_gain": 0.002912370935025388,
        "source_scope_digest": digest({"scope": index}),
    }


def _policy(*, rows: int) -> dict:
    policy = build_agent_policy(genome_digest="a" * 64, profile={}, parameters={}, previous_analysis=None,
                                generation=0, prediction_formula_policy=EVIDENCE_FORMULA_POLICY, joint=True)
    body = {key: value for key, value in policy.items() if key != "policy_digest"}
    body["experience"] = {**body["experience"], "rows": [_experience_row(i) for i in range(rows)]}
    return {**body, "policy_digest": digest(body)}


class _ScriptedRuntime(_SampleRuntime):
    def __init__(self, *, confidence=0.9, offset=0.0, method="model"):
        super().__init__()
        self.confidence, self.offset, self.method = confidence, offset, method

    def run_stage(self, request):
        if request["stage"] != "sample.plan":
            return super().run_stage(request)
        self.requests.append(request)
        context = request["request"]["context"]
        result = model_result(context)
        decisions = [dict(sample_id=row["sample_id"], predicted=row["predicted"] + self.offset,
                          confidence=self.confidence, reason_code="candidate_forecast",
                          method=self.method, evidence_call_ids=[result["call_id"]])
                     for row in result["outputs"]]
        structured = {"schema_version": "ecology-sample-predictions@2",
                      "wave_digest": context["wave_digest"], "decisions": decisions}
        return {"structured": structured, "result_digest": digest(structured)}


def _adapter(runtime, *, ownership_policy=ownership.POLICY, threshold=0.95):
    return DshSampleCollaborationAdapter(
        run_id="run-owned", runtime_provider=lambda: runtime,
        revision_provider=lambda _run_id: {"run_state_revision": 7, "ledger_expected_revision": 11},
        identity_digests=IDENTITY, strategy_model_id="dsh/strategy", review_model_id="dsh/review",
        forecast_bundle_tool=_constant_forecast_bundle(21.5), prediction_tool_binder=agent_binding,
        prediction_tool_catalog=[{"tool_id": "greenhouse-recipe-ridge@1", "version": "1",
                                  "purpose": "test ridge", "parameters": {}}],
        prediction_tool_executor=lambda requests, tool_id, parameters: {
            request.sample_id: {"predicted": 30.0} for request in requests},
        remote_critic_policy={"version": "uncertain_or_failure@1", "min_planner_confidence": threshold},
        prediction_ownership_policy=ownership_policy,
    )


def _context(policy):
    return {"run_id": "run-owned", "candidate_id": "candidate-1", "dataset_digest": "d" * 64,
            "algorithm_id": "registered-predictor", "algorithm_version": "1",
            "agent_policy": policy, "tool_experience": policy["experience"]["rows"],
            "candidate_agent_profile": {}}


class GenomeOwnedPredictionTests(unittest.TestCase):
    def test_copy_mode_exposes_only_the_candidate_tool_and_exact_model_copies(self):
        runtime = _ScriptedRuntime(confidence=0.6)
        adapter = _adapter(runtime)
        plan = adapter.plan_batch(_context(_policy(rows=32)))
        self.assertEqual(plan["prediction_ownership_policy"], ownership.POLICY)
        self.assertNotIn("tool_experience", plan["decision_context"])
        self.assertEqual(plan["decision_context"]["agent_policy"]["experience"]["rows"], [])
        self.assertLessEqual(len(canonical_json(plan).encode()), _MAX_PLAN_BYTES)
        outcome = adapter.predict_samples((_request("owned-1"),), (plan,), attempts=(1,))[0]
        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.result["predicted"], 21.5)
        # Confidence 0.6 is below the advisory threshold; copy mode skips the critic.
        self.assertEqual([item["stage"] for item in runtime.requests], ["sample.plan"])
        context = runtime.requests[0]["request"]["context"]
        self.assertEqual([tool["tool_id"] for tool in context["available_tools"]], ["candidate-model"])
        contract = context["context"]["prediction_contract"]
        self.assertEqual(contract["allowed_methods"], ["model"])
        self.assertEqual(contract["prediction_ownership_policy"], ownership.POLICY)
        self.assertNotIn("tool_experience", context["context"]["evolution_context"])

    def test_copy_mode_rejects_a_value_that_is_not_an_exact_copy(self):
        runtime = _ScriptedRuntime(offset=0.25)
        adapter = _adapter(runtime)
        plan = adapter.plan_batch(_context(_policy(rows=0)))
        outcome = adapter.predict_samples((_request("owned-2"),), (plan,), attempts=(1,))[0]
        self.assertIsNotNone(outcome.error)

    def test_copy_mode_requires_the_frozen_evidence_formula(self):
        runtime = _ScriptedRuntime()
        adapter = _adapter(runtime)
        policy = build_agent_policy(genome_digest="a" * 64, profile={}, parameters={}, previous_analysis=None,
                                    generation=0, joint=True)
        plan = adapter.plan_batch(_context(policy))
        # A Host contract fault, not a scientific failure: no Agent is called.
        from ecologyrsi_dsh.evaluators.sample_execution import SampleExecutionContractError
        with self.assertRaises(SampleExecutionContractError):
            adapter.predict_samples((_request("owned-3"),), (plan,), attempts=(1,))
        self.assertEqual(runtime.requests, [])

    def test_legacy_plan_is_unchanged_within_bound_and_compacted_beyond_it(self):
        legacy = _adapter(_ScriptedRuntime(), ownership_policy=None)
        small = legacy.plan_batch(_context(_policy(rows=1)))
        self.assertNotIn("decision_context_projection", small)
        self.assertIn("tool_experience", small["decision_context"])
        self.assertNotIn("prediction_ownership_policy", small)
        large = legacy.plan_batch(_context(_policy(rows=32)))
        self.assertEqual(large["decision_context_projection"], "bounded_sample_policy_view@1")
        self.assertNotIn("tool_experience", large["decision_context"])
        self.assertLessEqual(len(canonical_json(large).encode()), _MAX_PLAN_BYTES)
        rows = large["decision_context"]["agent_policy"]["experience"]["rows"]
        self.assertTrue(rows)
        self.assertNotIn("source_scope_digest", rows[0])
        self.assertEqual(large["decision_context"]["agent_policy"]["experience"]["source_row_count"], 32)

    def test_targets_and_operations_are_restricted_to_executable_output(self):
        targets = {"scientific_parameter": ("a",), "registered_predictor": ("p",),
                   "instruction_profile": ("t",), "strategy_program": ("s",), "skill_program": ("k",)}
        self.assertEqual(ownership.restrict_targets(targets),
                         {"scientific_parameter": ("a",), "strategy_program": ("s",)})
        self.assertEqual(ownership.non_prediction_operations([
            {"op": "set_bounded_parameter"}, {"op": "set_instruction_parameter"},
            {"op": "author_strategy_program"}]), ["set_instruction_parameter"])
        self.assertEqual(ownership.candidate_tool_id({}), "candidate-model")
        self.assertEqual(ownership.candidate_tool_id({"strategy_program": {"calls": []}}), "candidate-program")
        self.assertFalse(ownership.enabled({}))
        self.assertTrue(ownership.enabled({ownership.POLICY_KEY: ownership.POLICY}))
        with self.assertRaises(ValueError):
            ownership.enabled({ownership.POLICY_KEY: "unknown@1"})


class ResidualPanelTests(unittest.TestCase):
    def test_panel_reads_training_fit_only_and_is_digest_bound(self):
        source = series()
        changed = replace(source, values={name: column[:720] + (999999.0,) * (len(column) - 720)
                                          for name, column in source.values.items()})
        first = fit_partition_residual_panel(source, initial_fit_fraction=0.25)
        second = fit_partition_residual_panel(changed, initial_fit_fraction=0.25)
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], SCHEMA)
        self.assertEqual(first["label_scope"], "training_fit_only_forward_folds")
        self.assertEqual(len(first["families"]), 2)
        for family in first["families"]:
            self.assertEqual(len(family["cells"]), 9)
            for cell in family["cells"].values():
                self.assertEqual(len(cell["skill_by_scale"]), len(SCALES))
        self.assertEqual(validated_panel(first), first)
        tampered = {**first, "evaluated_origins": first["evaluated_origins"] + 1}
        with self.assertRaises(ValueError):
            validated_panel(tampered)
        self.assertIsNone(validated_panel(None))


if __name__ == "__main__":
    unittest.main()
