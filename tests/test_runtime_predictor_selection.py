from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from ecologyrsi_dsh.core.models import Candidate
from ecologyrsi_dsh.core.prediction_policy import (
    BASELINE_REFERENCE_PREDICTOR_ID, RUNTIME_EVALUATOR_ID,
    RUNTIME_PREDICTION_POLICY, prediction_usage,
)
from ecologyrsi_dsh.data.adapters import CUCUMBER_2018
from ecologyrsi_dsh.evaluators.greenhouse_prediction import BaselineAlignedRidgeConfig
from ecologyrsi_dsh.evolution.genome import (
    GenomeMutationContextV1, TRUST_REGION_MUTATION_OPERATOR_ID, apply_genome_mutation,
)
from ecologyrsi_dsh.evolution.strategies import (
    _genome_parameter_boundary,
)
from ecologyrsi_dsh.knowledge.algorithms import compile_algorithm_spec
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_baseline_aligned_ridge import periodic_series, run
from tests.test_dsh_native_runtime import _FakeNativeRuntime
from tests.test_evaluation import _native_registry, _scope_cohort
from tests import test_runtime_integration as runtime_helpers
from tests.native_fixtures import native_manifest
from tests.native_candidate_fixture import seed_proposal


PREDICTORS = (
    BASELINE_REFERENCE_PREDICTOR_ID, "greenhouse-exogenous-ridge@1",
    "greenhouse-targetwise-ridge@1", "greenhouse-horizon-targetwise-ridge@1",
)


class RuntimeChoiceApiTests(unittest.TestCase):
    setUp = runtime_helpers.RuntimeIntegrationTests.setUp
    tearDown = runtime_helpers.RuntimeIntegrationTests.tearDown
    request = runtime_helpers.RuntimeIntegrationTests.request

    def test_creation_freezes_registered_choices_before_research(self):
        native = _FakeNativeRuntime()
        native.run_stage = Mock(side_effect=AssertionError("creation must not call research models"))
        self.server.dsh_native_runtime = native
        body = {"execution_protocol": "dsh_native_plugin_evolution@1", "dataset_id": "agc_cucumber_2018",
            "strategy_model_id": "stub/strategy", "review_model_id": "stub/review", "start": False,
            "knowledge_online_enabled": False, "rounds": 1}
        for explicit in (False, True):
            request = {**body, "idempotency_key": f"runtime-choice-{explicit}"}
            if explicit:
                request["prediction_selection_policy"] = RUNTIME_PREDICTION_POLICY
            status, created = self.request("/runs", "POST", request)
            self.assertEqual(status, 201, created)
            state = self.server.director.state(created["projection"]["run_id"])
            metadata = state.task_manifest.metadata
            self.assertEqual(metadata["prediction_selection_policy"], RUNTIME_PREDICTION_POLICY)
            self.assertEqual(metadata["evaluator_id"], RUNTIME_EVALUATOR_ID)
            self.assertEqual(metadata["dataset_task"], CUCUMBER_2018.contract())
            self.assertTrue(set(PREDICTORS).issubset(metadata["prediction_selection"]["allowed_predictor_ids"]))
            genome = state.materialized_seed_genome()
            self.assertEqual(prediction_usage(BASELINE_REFERENCE_PREDICTOR_ID,
                genome.scientific_program["parameter_overrides"])["mode"], "baseline_only")
            self.assertTrue(any(e.kind == "DshRuntimeBound" for e in state.events))
        for field, value in (("task_manifest", {}), ("prediction_selection_policy", "unsupported"),
                             ("seed_genome_template_id", "greenhouse-default@1")):
            status, _ = self.request("/runs", "POST", {**body, field: value})
            self.assertEqual(status, 400)
        native.run_stage.assert_not_called()


class RuntimeChoiceEvolutionTests(unittest.TestCase):
    def test_selected_registered_genome_can_be_tuned_and_disabled(self):
        from tests.test_evolution_genome import _seed_genome
        from tests.native_fixtures import native_manifest
        parent = _seed_genome()
        for predictor in PREDICTORS[1:]:
            genome = parent if parent.scientific_program["predictor_ref"]["id"] == predictor else self.mutate(parent, {"op": "select_registered_pipeline", "predictor_id": predictor})
            task = native_manifest()
            _, schemas = _genome_parameter_boundary(task, genome)
            self.assertNotIn("residual_scale_6h", schemas)
            tuned = self.mutate(genome, {"op": "set_bounded_parameter", "name": "ridge_alpha", "value": .2})
            self.assertEqual(tuned.scientific_program["parameter_overrides"]["ridge_alpha"], .2)
            baseline = self.mutate(tuned, {"op": "select_registered_pipeline", "predictor_id": BASELINE_REFERENCE_PREDICTOR_ID})
            self.assertEqual(prediction_usage(BASELINE_REFERENCE_PREDICTOR_ID,
                baseline.scientific_program["parameter_overrides"])["mode"], "baseline_only")

    def mutate(self, parent, operation):
        context = GenomeMutationContextV1(
            run_id="run:runtime-model-choice", generation=1, slot_index=0, slot_seed=1,
            parent_candidate_id="candidate:parent", parent_genome_digest=parent.genome_digest,
            generation_batch_digest="a" * 64, research_iteration_digest="b" * 64,
            knowledge_snapshot_digest="c" * 64, mutation_budget_digest="d" * 64,
            mutation_operator_id=TRUST_REGION_MUTATION_OPERATOR_ID,
        )
        return apply_genome_mutation(parent, {"schema_version": "ecologyrsi-dsh.genome-mutation/1",
                                    "operations": [operation]}, context, current_program_registry())


class RuntimeChoiceNumericalTests(unittest.TestCase):
    def test_baseline_only_never_fits_residuals_and_partial_use_fits_only_active_cells(self):
        from ecologyrsi_dsh.evaluators.greenhouse_prediction import _ridge_coefficients
        with patch("ecologyrsi_dsh.evaluators.greenhouse_prediction._ridge_coefficients",
                   wraps=_ridge_coefficients) as solver:
            result = run(periodic_series(), 0)
            solver.assert_not_called()
            self.assertTrue(all(m["status"] == "baseline_only" for m in result["models"]))
            self.assertTrue(all(not r["used_zero_residual_fallback"] for r in result["prediction_rows"]))
            partial = run(periodic_series(), config=BaselineAlignedRidgeConfig(
                6, .1, 0, residual_scale_6h=.1))
            self.assertEqual(solver.call_count, 3)
            self.assertEqual(sum(m["status"] == "baseline_only" for m in partial["models"]), 6)

    def test_all_registered_models_share_the_measurement_grid_and_scoring_baseline(self):
        series = periodic_series()
        registry = _native_registry(series)
        digests = []
        for predictor in PREDICTORS:
            with self.subTest(predictor=predictor):
                template = {
                    BASELINE_REFERENCE_PREDICTOR_ID: "greenhouse-baseline-aligned-default@2",
                    "greenhouse-exogenous-ridge@1": "greenhouse-exogenous-default@2",
                    "greenhouse-targetwise-ridge@1": "greenhouse-targetwise-default@2",
                    "greenhouse-horizon-targetwise-ridge@1": "greenhouse-default@2",
                }[predictor]
                task = native_manifest(task_id="choice-grid", metadata={
                    "seed_genome_template_id": template,
                    "episode_id": series.episode_id, "prediction_model_id": predictor,
                    "evaluator_id": RUNTIME_EVALUATOR_ID, "dataset_digest": series.digest,
                    "data_protocol_digest": "f" * 64,
                    "split_manifest_digest": series.split_manifest_digest_sha256})
                proposal = seed_proposal(task, run_id="run:grid")
                candidate = Candidate(candidate_id="candidate:grid", run_id="run:grid",
                                      proposal_id=proposal.proposal_id, generation=0)
                compiled = compile_algorithm_spec(task, proposal, None)
                self.assertEqual(compiled.evaluator_id, RUNTIME_EVALUATOR_ID)
                self.assertEqual(compiled.adapter_id, predictor)
                registry.dsh_identity_provider = lambda *_: {
                    "genome_digest": proposal.metadata["genome_digest"],
                    "compiled_behavior_digest": "b" * 64, "phenotype_instance_digest": "c" * 64}
                bundle = registry.evaluate_scientific(task, candidate, proposal, **_scope_cohort(series,candidate,count=3))
                metrics = bundle.evaluation.metrics
                digests.append((metrics["evaluation_index_digest"], metrics["baseline_metrics_digest"]))
                self.assertEqual(len(metrics["targets"]), 9)
        self.assertEqual(len(set(digests)), 1)


if __name__ == "__main__":
    unittest.main()
