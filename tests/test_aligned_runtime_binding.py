"""Current native creation freezes the aligned predictor and causal history."""
from ecologyrsi_dsh.core.prediction_policy import RUNTIME_EVALUATOR_ID
from ecologyrsi_dsh.evaluators.epoch_cohorts import estimate_epoch_capacity
from ecologyrsi_dsh.evaluators.greenhouse_prediction import (
    BASELINE_ALIGNED_RIDGE_MODEL_ID, COHORT_HISTORY_HOURS, origin_history_alignment_hours,
)
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
from ecologyrsi_dsh.knowledge.algorithms import compile_algorithm_spec
from tests.native_http_fixture import NativeBindingFixture

PARAMETERS = {"history_steps", "ridge_alpha", "residual_scale_1h",
              "residual_scale_6h", "residual_scale_24h"}


class AlignedRuntimeBindingTests(NativeBindingFixture):
    def test_aligned_five_parameter_contract_and_seed_compile(self):
        for explicit in (False, True):
            with self.subTest(explicit_evaluator=explicit):
                fields = {"prediction_model_id": BASELINE_ALIGNED_RIDGE_MODEL_ID}
                if explicit:
                    fields["evaluator_id"] = RUNTIME_EVALUATOR_ID
                state, result = self.create(f"aligned-{explicit}", **fields)
                metadata = state.task_manifest.metadata
                self.assertEqual(metadata["evaluator_id"], RUNTIME_EVALUATOR_ID)
                self.assertEqual(metadata["seed_genome_template_id"], "greenhouse-baseline-aligned-default@2")
                self.assertEqual(metadata["evolution_diversity_policy"], "joint_behavior_search@1")
                self.assertEqual(result["projection"]["evolution_diversity"]["proposed_by_family"], {})
                from ecologyrsi_dsh.core.model_execution_policy import RESEARCH_EXECUTION_POLICY
                self.assertEqual(dict(metadata["research_execution_policy"]), dict(RESEARCH_EXECUTION_POLICY))
                self.assertEqual(set(StrategyRouterDSHAdapter.parameter_schemas_for_task(state.task_manifest)), PARAMETERS)
                semantics = StrategyRouterDSHAdapter.parameter_semantics_for_task(state.task_manifest)
                self.assertEqual(set(semantics), PARAMETERS)
                for horizon in (1, 6, 24):
                    self.assertIn("baseline", semantics[f"residual_scale_{horizon}h"].casefold())
                catalog = metadata["runtime_component_catalog"]["prediction_models"]
                self.assertIn(BASELINE_ALIGNED_RIDGE_MODEL_ID, [item["id"] for item in catalog])
                self.assertTrue(all(item["id"].startswith("greenhouse-") for item in catalog))
                self.server.director.start_run(state.run.run_id)
                seed = self.server.director.ensure_seed_incumbent_control(state.run.run_id)
                current = self.server.director.state(state.run.run_id)
                proposal = current.proposal(seed.proposal_id)
                self.assertEqual(set(proposal.changes), PARAMETERS)
                compiled = compile_algorithm_spec(current.task_manifest, proposal, None)
                self.assertEqual(compiled.adapter_id, BASELINE_ALIGNED_RIDGE_MODEL_ID)
                self.assertEqual(compiled.evaluator_id, RUNTIME_EVALUATOR_ID)

    def test_native_history_alignment_is_frozen_and_rederived_without_drift(self):
        state, _ = self.create("alignment-frozen")
        metadata = state.task_manifest.metadata
        self.assertEqual(metadata["origin_history_alignment_hours"], COHORT_HISTORY_HOURS)
        self.assertEqual(origin_history_alignment_hours(metadata), COHORT_HISTORY_HOURS)
        report = metadata["cohort_capacity_report"]
        view = self.server.datasets.selection_view("agc_cucumber_2018", "agc_cucumber_2018:TeamA")
        arguments = dict(schedule=OptimizationSchedule.from_dict(metadata["optimization_schedule"]),
                         planned_generations=1, seed=state.task_manifest.seed, scoring_cells_per_origin=9)
        expected = estimate_epoch_capacity(view, **arguments, history_steps=COHORT_HISTORY_HOURS)
        self.assertEqual(report["planner_digest"], expected.planner_digest)
        self.assertNotEqual(report["planner_digest"], estimate_epoch_capacity(
            view, **arguments, history_steps=COHORT_HISTORY_HOURS - 1).planner_digest)

    def test_retired_evaluator_and_toy_requests_are_rejected(self):
        for fields in ({"prediction_model_id": BASELINE_ALIGNED_RIDGE_MODEL_ID,
                        "evaluator_id": "greenhouse_multihorizon_time_forward@3"},
                       {"dataset_id": "generated-toy-series@1"}):
            status, result = self._post_path("/runs", self.body("retired-binding", **fields))
            self.assertIn(status, (400, 404), result)
        self.assertEqual(self.native.created, [])
