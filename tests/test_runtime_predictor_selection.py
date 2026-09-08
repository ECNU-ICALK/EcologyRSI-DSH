from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import Candidate, Proposal, TaskManifest
from ecologyrsi_dsh.core.prediction_policy import (
    BASELINE_REFERENCE_PREDICTOR_ID, RUNTIME_EVALUATOR_ID,
    RUNTIME_PREDICTION_POLICY, prediction_usage,
)
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.data.adapters import CUCUMBER_2018
from ecologyrsi_dsh.evaluators.greenhouse_prediction import BaselineAlignedRidgeConfig
from ecologyrsi_dsh.evaluators.registry import EvaluatorRegistry
from ecologyrsi_dsh.evolution.batches import start_generation_batch
from ecologyrsi_dsh.evolution.genome import (
    GenomeMutationContextV1, TRUST_REGION_MUTATION_OPERATOR_ID, apply_genome_mutation,
)
from ecologyrsi_dsh.evolution.strategies import (
    StrategyRouterDSHAdapter, _genome_parameter_boundary, _mutation_contract_catalog,
)
from ecologyrsi_dsh.knowledge.algorithms import compile_algorithm_spec
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_autonomous_search_reflection_cycle import _CycleRuntime, _task
from tests.test_baseline_aligned_ridge import periodic_series, run
from tests.test_dsh_native_runtime import _FakeNativeRuntime
from tests.test_evaluation import _DatasetStub
from tests import test_runtime_integration as runtime_helpers


PREDICTORS = (
    BASELINE_REFERENCE_PREDICTOR_ID, "greenhouse-exogenous-ridge@1",
    "greenhouse-targetwise-ridge@1", "greenhouse-horizon-targetwise-ridge@1",
)


class RuntimeChoiceApiTests(unittest.TestCase):
    setUp = runtime_helpers.RuntimeIntegrationTests.setUp
    tearDown = runtime_helpers.RuntimeIntegrationTests.tearDown
    request = runtime_helpers.RuntimeIntegrationTests.request

    def test_creation_freezes_choices_without_requiring_or_calling_a_model(self):
        series = SimpleNamespace(digest="d" * 64, split_manifest_digest_sha256="b" * 64,
                                 episode_id="agc_cucumber_2018:test")
        selection = SimpleNamespace(
            dataset_id="agc_cucumber_2018", episode_id=series.episode_id,
            timestamps=tuple(range(2000)), partitions={"model_selection": IndexRange(0, 2000)},
            data_protocol_digest="c" * 64, selection_view_digest="e" * 64,
        )
        description = {"descriptor": {"runnable": True, "display_name_zh": "测试温室",
                       "domain_id": "greenhouse_test", "adapter_id": "greenhouse_test"},
                       "readiness": {"ready": True}}
        native = _FakeNativeRuntime()
        native.run_stage = Mock(side_effect=AssertionError("creation must not call research models"))
        self.server.dsh_native_runtime = native
        body = {
            "execution_protocol": "dsh_native_plugin_evolution@1",
            "domain_pack_id": "greenhouse_environment@1", "dataset_id": "agc_cucumber_2018",
            "strategy_model_id": "stub/strategy", "review_model_id": "stub/review",
            "autonomous_mode": True, "auto_advance": 0,
            "budget": {"max_generations": 1, "max_candidates": 4, "candidates_per_generation": 4},
        }
        with (patch.object(self.server.datasets, "describe", return_value=description),
              patch.object(self.server.datasets, "series", return_value=series),
              patch.object(self.server.datasets, "selection_view", return_value=selection)):
            for explicit in (False, True):
                request = {**body, "idempotency_key": f"runtime-choice-{explicit}"}
                if explicit:
                    request["prediction_selection_policy"] = RUNTIME_PREDICTION_POLICY
                status, created = self.request("/runs", "POST", request)
                self.assertEqual(status, 201, created)
                state = self.server.director.state(created["projection"]["run_id"])
                metadata = state.task_manifest.metadata
                self.assertEqual(created["projection"]["configuration"]["dataset_task"], CUCUMBER_2018.contract())
                self.assertEqual(metadata["prediction_selection_policy"], RUNTIME_PREDICTION_POLICY)
                self.assertEqual(metadata["evaluator_id"], RUNTIME_EVALUATOR_ID)
                self.assertEqual(metadata["dataset_task"], CUCUMBER_2018.contract())
                self.assertEqual(metadata["dataset_task_digest"], CUCUMBER_2018.contract()["contract_digest"])
                self.assertEqual(metadata["evaluator_digest"], self.server.evaluators.evaluator_configuration_digest(RUNTIME_EVALUATOR_ID, CUCUMBER_2018.dataset_id))
                self.assertEqual(set(metadata["prediction_selection"]["allowed_predictor_ids"]), set(PREDICTORS))
                self.assertEqual({p["id"] for p in metadata["runtime_component_catalog"]["prediction_models"]}, set(PREDICTORS))
                genome = state.materialized_seed_genome()
                self.assertEqual(prediction_usage(BASELINE_REFERENCE_PREDICTOR_ID,
                                 genome.scientific_program["parameter_overrides"])["mode"], "baseline_only")
            full = TaskManifest(task_id="manifest-choice", objective="runtime choice from a full manifest",
                domain_pack="greenhouse_environment@1", visible_datasets=("agc_cucumber_2018",),
                budget=body["budget"], metadata={
                    "execution_protocol": body["execution_protocol"],
                    "strategy_model_id": "stub/strategy", "review_model_id": "stub/review",
                    "autonomous_mode": True,
                }).to_dict()
            for field, value in (("dataset_task", {"dataset_id": "wrong"}), ("dataset_task_digest", "bad"),
                                 ("objective_targets", ["marketable_yield"]),
                                 ("objective_horizons", [72])):
                invalid = {**full, "metadata": {**full["metadata"], field: value}}
                status, error = self.request("/runs", "POST", {
                    "task_manifest": invalid, "auto_advance": 0,
                    "idempotency_key": "reject-dataset-task-" + field,
                })
                self.assertEqual(status, 400, error)
            full_request = {"task_manifest": full, "auto_advance": 0,
                            "idempotency_key": "full-manifest-choice",
                            "prediction_selection_policy": RUNTIME_PREDICTION_POLICY}
            status, created = self.request("/runs", "POST", full_request)
            self.assertEqual(status, 201, created)
            state = self.server.director.state(created["projection"]["run_id"])
            self.assertEqual(state.task_manifest.metadata["prediction_selection_policy"], RUNTIME_PREDICTION_POLICY)
            self.assertEqual(native.created[-1]["run_id"], state.run.run_id)
            self.assertTrue(any(e.kind == "DshRuntimeBound" for e in state.events))
            for overrides in (
                {"execution_protocol": "different-protocol"},
                {"prediction_selection_policy": "unsupported"},
                {"prediction_model_id": PREDICTORS[1]},
            ):
                status, error = self.request("/runs", "POST", {
                    **full_request, **overrides, "idempotency_key": "reject-full-" + next(iter(overrides)),
                })
                self.assertEqual(status, 400, error)
            for field, value in (("prediction_model_id", PREDICTORS[1]),
                                 ("evaluator_id", RUNTIME_EVALUATOR_ID),
                                 ("seed_genome_template_id", "greenhouse-default@1")):
                status, error = self.request("/runs", "POST", {
                    **body, "prediction_selection_policy": RUNTIME_PREDICTION_POLICY,
                    field: value, "idempotency_key": f"reject-{field}",
                })
                self.assertEqual(status, 400, error)
                self.assertIn("preselected", str(error))
        native.run_stage.assert_not_called()


class _ChoiceRuntime(_CycleRuntime):
    def run_stage(self, request):
        result = super().run_stage(request)
        structured = result["structured"]
        if request["stage"] == "generation.research-synthesis":
            structured["summary"] = "Compare registered forecast structures on the frozen task cells."
            for index, direction in enumerate(structured["candidate_directions"]):
                direction.update(mutation_axis="registered_predictor", mutation_target=PREDICTORS[index + 1],
                                 mutation_direction="select", title=f"Forecast structure {index + 1}",
                                 hypothesis=f"The structure in {PREDICTORS[index + 1]} may improve the weak cells.")
        elif request["stage"] == "candidate.propose":
            slot = request["request"]["context"]["mutation_context"]["slot_index"]
            structured["operations"] = [{"op": "select_registered_pipeline", "predictor_id": PREDICTORS[slot + 1]}]
        from ecologyrsi_dsh.core.models import digest
        return {"structured": structured, "result_digest": digest(structured)}


class RuntimeChoiceEvolutionTests(unittest.TestCase):
    def test_research_selects_models_and_selected_genome_can_be_tuned_or_disabled(self):
        registry = current_program_registry()
        evaluator = EvaluatorRegistry(_DatasetStub(periodic_series()))
        catalog = [p for p in evaluator.predictor_catalog() if p["id"] in PREDICTORS]
        for item in catalog:
            item["parameter_schemas"] = registry.program("predictors", item["id"])["parameters"]
        base = _task()
        task = replace(base, metadata={**base.metadata,
            "prediction_model_id": BASELINE_REFERENCE_PREDICTOR_ID,
            "evaluator_id": RUNTIME_EVALUATOR_ID,
            "prediction_selection_policy": RUNTIME_PREDICTION_POLICY,
            "seed_genome_template_id": "greenhouse-baseline-aligned-default@1",
            "runtime_component_catalog": {"prediction_models": catalog,
                "selected_evaluator_id": RUNTIME_EVALUATOR_ID,
                "selected_prediction_model_id": BASELINE_REFERENCE_PREDICTOR_ID},
        })
        ledger = EventLedger(":memory:")
        self.addCleanup(ledger.close)
        runtime = _ChoiceRuntime()
        director = EvolutionDirector(ledger, StrategyRouterDSHAdapter(
            gateway=object(), native_runtime_provider=lambda: runtime))
        run_id = "run:runtime-model-choice"
        director.create_run(task, run_id=run_id)
        director.start_run(run_id)
        batch = start_generation_batch(director, run_id)
        for slot in range(batch.batch_size):
            proposal = director.request_proposal(run_id, generation_batch=batch, slot_index=slot)
            director.spawn_candidate(run_id, proposal, slot_index=slot)
        state = director.state(run_id)
        for request in runtime.requests:
            if request["stage"] in {"generation.research-synthesis", "candidate.propose"}:
                context = request["request"]["context"]
                contract = context["synthesis_contract"] if request["stage"] == "generation.research-synthesis" else context["mutation_contract"]
                choice = contract["prediction_selection"]
                self.assertEqual(choice["current_usage"]["mode"], "baseline_only")
                self.assertEqual({p["id"] for p in choice["predictor_options"]}, set(PREDICTORS))
        for index, candidate in enumerate(state.candidates):
            genome = state.persisted_genome_for(candidate.candidate_id)
            self.assertEqual(genome.scientific_program["predictor_ref"]["id"], PREDICTORS[index + 1])
            from ecologyrsi_dsh.api.projection import _candidate_projection
            # Selection must be visible immediately after spawning, before any
            # compilation/evaluation attempt can supply a predictor identifier.
            self.assertEqual(state.algorithm_attempts_for(candidate.candidate_id), ())
            projected = _candidate_projection(state, candidate, summary_only=True)
            self.assertEqual(projected["execution"]["prediction_model_id"], PREDICTORS[index + 1])
            self.assertEqual(projected["execution"]["prediction_usage"]["mode"], "registered_model")
            compiled = compile_algorithm_spec(task, state.proposal(candidate.proposal_id), None)
            self.assertEqual(compiled.adapter_id, PREDICTORS[index + 1])
            self.assertEqual(compiled.evaluator_id, RUNTIME_EVALUATOR_ID)
            _, schemas = _genome_parameter_boundary(task, genome)
            # The new model's parameters, not the baseline anchor's, govern later tuning.
            self.assertNotIn("residual_scale_6h", schemas)
            tuned = self.mutate(genome, {"op": "set_bounded_parameter", "name": "ridge_alpha", "value": .2})
            self.assertEqual(tuned.scientific_program["parameter_overrides"]["ridge_alpha"], .2)
            baseline = self.mutate(tuned, {"op": "select_registered_pipeline", "predictor_id": BASELINE_REFERENCE_PREDICTOR_ID})
            self.assertEqual(prediction_usage(BASELINE_REFERENCE_PREDICTOR_ID,
                baseline.scientific_program["parameter_overrides"])["mode"], "baseline_only")
            enabled = self.mutate(baseline, {"op": "set_bounded_parameter", "name": "residual_scale_6h", "value": .1})
            choices = _mutation_contract_catalog(task, enabled)["allowed_mutation_targets"]
            self.assertIn(BASELINE_REFERENCE_PREDICTOR_ID, choices["registered_predictor"])
            reset = self.mutate(enabled, {"op": "select_registered_pipeline", "predictor_id": BASELINE_REFERENCE_PREDICTOR_ID})
            self.assertEqual(reset.scientific_program["parameter_overrides"], baseline.scientific_program["parameter_overrides"])
        self.assertEqual(len(state.candidates), 2)

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
        registry = EvaluatorRegistry(_DatasetStub(series))
        digests = []
        for predictor in PREDICTORS:
            with self.subTest(predictor=predictor):
                task = TaskManifest(task_id="choice-grid", objective="equal measurement",
                    domain_pack="greenhouse_environment@1", visible_datasets=(series.dataset_id,),
                    budget={"max_candidates": 1}, metadata={
                        "episode_id": series.episode_id, "prediction_model_id": predictor,
                        "evaluator_id": RUNTIME_EVALUATOR_ID, "dataset_digest": series.digest,
                        "split_manifest_digest": series.split_manifest_digest_sha256})
                proposal = Proposal(proposal_id="proposal:grid", run_id="run:grid", generation=0,
                    title=predictor, changes=current_program_registry().predictor_defaults(predictor))
                candidate = Candidate(candidate_id="candidate:grid", run_id="run:grid",
                                      proposal_id=proposal.proposal_id, generation=0)
                compiled = compile_algorithm_spec(task, proposal, None)
                self.assertEqual(compiled.evaluator_id, RUNTIME_EVALUATOR_ID)
                bundle = registry.evaluate_scientific(task, candidate, proposal)
                metrics = bundle.evaluation.metrics
                digests.append((metrics["evaluation_index_digest"], metrics["baseline_metrics_digest"]))
                self.assertEqual(len(metrics["targets"]), 9)
                if predictor == BASELINE_REFERENCE_PREDICTOR_ID:
                    self.assertEqual(bundle.evaluation.score, 0)
                    self.assertEqual(bundle.artifact.metrics["fit_passes_completed"], 0)
                    self.assertEqual(bundle.artifact.metrics["solver_fallback_count"], 0)
                    self.assertEqual(metrics["prediction_usage"]["mode"], "baseline_only")
        self.assertEqual(len(set(digests)), 1)


if __name__ == "__main__":
    unittest.main()
