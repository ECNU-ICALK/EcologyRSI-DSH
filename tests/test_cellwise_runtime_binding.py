"""Cellwise opt-in remains distinct from standard native scientific bindings."""
from ecologyrsi_dsh.core.prediction_policy import CELLWISE_EVALUATOR_ID, TARGETED_SEARCH_POLICY, CONTRIBUTION_POLICY
from ecologyrsi_dsh.knowledge.algorithms import compile_algorithm_spec
from tests.native_http_fixture import NativeBindingFixture


class CellwiseRuntimeBindingTests(NativeBindingFixture):
    def test_opt_in_frozen_new_contract_and_compilable_seed(self):
        state, _ = self.create("cellwise-test", prediction_selection_policy="model_during_run@1",
                               targeted_search_policy=TARGETED_SEARCH_POLICY)
        metadata = state.task_manifest.metadata
        self.assertEqual(metadata["evaluator_id"], CELLWISE_EVALUATOR_ID)
        self.assertEqual(metadata["targeted_search_policy"], TARGETED_SEARCH_POLICY)
        self.assertEqual(metadata["behavior_diagnostics_policy"], CONTRIBUTION_POLICY)
        self.assertEqual(metadata["candidate_novelty_policy"], "run_behavior_novelty@1")
        self.assertEqual(metadata["seed_genome_template_id"], "greenhouse-baseline-aligned-cellwise-default@2")
        self.assertIn("greenhouse-baseline-aligned-ridge@2",
                      [item["id"] for item in metadata["runtime_component_catalog"]["prediction_models"]])
        self.server.director.start_run(state.run.run_id)
        seed = self.server.director.ensure_seed_incumbent_control(state.run.run_id)
        current = self.server.director.state(state.run.run_id)
        spec = compile_algorithm_spec(current.task_manifest, current.proposal(seed.proposal_id), None)
        self.assertEqual(spec.adapter_id, "greenhouse-baseline-aligned-ridge@2")
        self.assertEqual(len(spec.parameters), 11)
        for field in ("targeted_search_policy", "candidate_novelty_policy"):
            status, result = self._post_path("/runs", self.body(f"invalid-{field}", **{field: "unknown"}))
            self.assertEqual(status, 400, result)
        standard, _ = self.create("standard-novelty")
        self.assertEqual(standard.task_manifest.metadata["candidate_novelty_policy"], "run_behavior_novelty@1")
        self.assertIsNone(standard.task_manifest.metadata.get("targeted_search_policy"))
        self.assertEqual(standard.task_manifest.metadata["evaluator_id"], "greenhouse_multihorizon_time_forward@4")
