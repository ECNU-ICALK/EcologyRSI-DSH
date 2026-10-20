import unittest
from ecologyrsi_dsh.evolution.context import safe_aggregate_feedback
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
from ecologyrsi_dsh.integrations.model_directory import ModelDirectory
from ecologyrsi_dsh.integrations.dsh_native_runtime import DshNativeAgentRuntimeClient
from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
from tests.native_fixtures import native_manifest

class AggregateFeedbackBoundaryTests(unittest.TestCase):
    def test_skill_outcomes_keep_program_and_provenance_across_reflection_envelope(self):
        from ecologyrsi_dsh.evaluators.skill_program import seed_skill_program

        feedback = {"reflection_digest": "a" * 64, "canonical_candidate_outcomes": [{
            "candidate_id": "candidate:skill", "score": None, "eligible": False,
            "mutation_operations": [{"op": "author_skill_program", "role": "sample-planner",
                                     "skill_program": seed_skill_program()}],
        }]}
        self.assertEqual(safe_aggregate_feedback(feedback, name="reflection"), feedback)

    def test_outcome_envelope_preserves_sensitive_field_and_resource_limits(self):
        from copy import deepcopy
        from ecologyrsi_dsh.evaluators.skill_program import seed_skill_program

        row = {"mutation_operations": [{"op": "author_skill_program", "skill_program": seed_skill_program()}]}
        poisoned = deepcopy(row)
        step = poisoned["mutation_operations"][0]["skill_program"]["steps"][0]
        step.update({"GroundTruth": "private-label", "source_code": "private-code",
                     "sample_execution_trace_archive": "private-trace"})
        self.assertEqual(safe_aggregate_feedback({"canonical_candidate_outcomes": [poisoned]},
                         name="reflection"), {"canonical_candidate_outcomes": [row]})
        deep = {"safe": 1}
        for _ in range(8):
            deep = {"canonical_candidate_outcomes": [deep]}
        for payload in (deep, {"text": "x" * 4001}, {"items": list(range(129))}):
            with self.subTest(payload_type=next(iter(payload))), self.assertRaises(ValueError):
                safe_aggregate_feedback({"canonical_candidate_outcomes": [payload]}, name="reflection")
        with self.assertRaises(ValueError):
            safe_aggregate_feedback({"canonical_candidate_outcomes": [row] * 9}, name="reflection")
        with self.assertRaises(TypeError):
            safe_aggregate_feedback({"canonical_candidate_outcomes": [None]}, name="reflection")

    def test_sample_execution_detail_is_removed_before_agent_feedback(self) -> None:
        projected = safe_aggregate_feedback(
            {
                "scientific_pass": False,
                "sample_execution": {
                    "failure_preview": [
                        {
                            "attempt_trace": [
                                {"critic_decisions": [{"reason_code": "private"}]}
                            ]
                        }
                    ]
                },
                "sample_execution_trace_archive": "private-archive",
                "sample_execution_coverage": 0.0,
            },
            name="generation judge aggregate metrics",
        )

        self.assertEqual(
            projected,
            {"scientific_pass": False, "sample_execution_coverage": 0.0},
        )

    def test_research_iteration_envelope_does_not_consume_plan_depth(self) -> None:
        feedback = {
            "plan": {
                "dsh_research_evidence": [
                    {
                        "grounding": {
                            "historical_generations": [
                                {"run_id": "run:prior", "generation": 2}
                            ]
                        }
                    }
                ]
            }
        }

        self.assertEqual(
            safe_aggregate_feedback(feedback, name="research iteration"),
            feedback,
        )

class NativeStrategyBoundaryTests(unittest.TestCase):
    def test_directory_only_catalog_and_current_strategy(self):
        adapter=StrategyRouterDSHAdapter(gateway=ModelDirectory(()))
        self.assertEqual(adapter.connection_catalog,[])
        self.assertEqual([row['id'] for row in adapter.catalog()],["autonomous_model@1"])
        for old in ('parameter_sweep@1','adaptive_local@1','dsh_authenticated@1'):
            with self.assertRaises(ValueError):adapter.configuration_digest(old)

    def test_raw_runtime_requires_host_admission(self):
        adapter=StrategyRouterDSHAdapter(gateway=ModelDirectory(()),
            native_runtime_provider=lambda:DshNativeAgentRuntimeClient('http://127.0.0.1:9',token='synthetic'))
        with self.assertRaisesRegex(DshNativeRuntimeUnavailableError,'Host-local admission service'):
            adapter._native_runtime()

    def test_parameter_catalog_matches_supported_predictors(self):
        from dataclasses import replace
        from ecologyrsi_dsh.evaluators.registry import EvaluatorRegistry
        from tests.test_evaluation import _DatasetStub,_cohort_series
        catalog=EvaluatorRegistry(_DatasetStub(_cohort_series())).predictor_catalog()
        for row in catalog:
            task=native_manifest(metadata={'prediction_model_id':row['id']})
            schemas=StrategyRouterDSHAdapter.parameter_schemas_for_task(task)
            self.assertEqual(set(schemas),set() if row['id']=='greenhouse-recipe-ridge@1' else set(row['parameter_names']))
        for old in ('toy-rolling-water@1','greenhouse-rolling-residual@1'):
            with self.assertRaises(ValueError):StrategyRouterDSHAdapter.parameter_schemas_for_task(native_manifest(metadata={'prediction_model_id':old}))
