from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ecologyrsi_dsh.core.models import Run, TaskManifest
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.epoch_cohorts import estimate_epoch_capacity
from ecologyrsi_dsh.evaluators.greenhouse_prediction import (
    BASELINE_ALIGNED_RIDGE_MODEL_ID,
    COHORT_HISTORY_HOURS,
    MAX_EXOGENOUS_RIDGE_HISTORY_STEPS,
    origin_history_alignment_hours,
)
from ecologyrsi_dsh.evolution.schedule import OPTIMIZATION_PROTOCOL, OptimizationSchedule
from ecologyrsi_dsh.evaluators.registry import GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
from ecologyrsi_dsh.integrations.dsh_native_runtime import DSH_NATIVE_EXECUTION_PROTOCOL
from ecologyrsi_dsh.knowledge.algorithms import compile_algorithm_spec, resolve_predictor_adoption
from ecologyrsi_dsh.knowledge.algorithm_ir import registered_algorithm_blueprint
from tests import test_runtime_integration as runtime_test_helpers


PARAMETERS = {"history_steps", "ridge_alpha", "residual_scale_1h",
              "residual_scale_6h", "residual_scale_24h"}


class AlignedRuntimeBindingTests(unittest.TestCase):
    # Reuse only the fixture lifecycle, without inheriting unrelated tests.
    setUp = runtime_test_helpers.RuntimeIntegrationTests.setUp
    tearDown = runtime_test_helpers.RuntimeIntegrationTests.tearDown
    request = runtime_test_helpers.RuntimeIntegrationTests.request

    def test_five_parameter_schema_sweep_and_completed_parent_compile(self):
        task = TaskManifest(
            task_id="aligned-sweep", objective="aligned parameter sweep",
            domain_pack="greenhouse_environment@1", visible_datasets=("agc_cucumber_2018",),
            budget={"max_candidates": 4}, metadata={
                "domain": "greenhouse", "strategy_id": "parameter_sweep@1",
                "prediction_model_id": BASELINE_ALIGNED_RIDGE_MODEL_ID,
                "evaluator_id": GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID,
            })
        run = Run(run_id="run:aligned-sweep", task_id=task.task_id,
                  task_manifest_digest=task.digest)
        adapter = StrategyRouterDSHAdapter(max_proposals=4)
        schemas = adapter.parameter_schemas_for_task(task)
        self.assertEqual(set(schemas), PARAMETERS)
        semantics = adapter.parameter_semantics_for_task(task)
        self.assertEqual(set(semantics), PARAMETERS)
        self.assertTrue(all("Bounded numeric parameter" not in text for text in semantics.values()))
        for horizon in (1, 6, 24):
            self.assertIn("baseline", semantics[f"residual_scale_{horizon}h"].casefold())
        session = adapter.open_session(run, task)
        seed = adapter.propose(run, task, session)
        self.assertEqual(set(seed.changes), PARAMETERS)
        # The sweep deliberately starts with an active residual learner.
        self.assertTrue(all(seed.changes[f"residual_scale_{h}h"] == 0.5 for h in (1, 6, 24)))
        compiled = compile_algorithm_spec(task, seed, None)
        self.assertEqual(compiled.adapter_id, BASELINE_ALIGNED_RIDGE_MODEL_ID)
        parent = {
            "candidate_id": "candidate:aligned-parent", "status": "promoted",
            "proposal_id": seed.proposal_id, "proposal_parameters": dict(seed.changes),
            "evaluation": {"score": 0.0, "passed": False}, "artifact": {}, "judge": {},
        }
        proposal = adapter.propose(run, task, session,
                                   parent_candidate_id=parent["candidate_id"], parent_context=parent)
        self.assertEqual(set(proposal.changes), PARAMETERS)
        self.assertEqual(compile_algorithm_spec(task, proposal, None).evaluator_id,
                         GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID)

    def test_native_api_defaults_to_v3_and_freezes_compilable_aligned_seed(self):
        description = {
            "descriptor": {"runnable": True, "display_name_zh": "测试温室序列",
                           "domain_id": "greenhouse_test", "adapter_id": "greenhouse_test"},
            "readiness": {"ready": True},
        }
        series = SimpleNamespace(digest="d" * 64, split_manifest_digest_sha256="b" * 64,
                                 episode_id="agc_cucumber_2018:test")
        selection_view = SimpleNamespace(
            dataset_id="agc_cucumber_2018", episode_id=series.episode_id,
            timestamps=tuple(range(2000)), partitions={"model_selection": IndexRange(0, 2000)},
            data_protocol_digest="c" * 64, selection_view_digest="e" * 64,
        )
        native_runtime = Mock()
        native_runtime.capabilities.return_value = {
            "schema_version": "ecology-agent-runtime-capabilities/1", "ready": True,
            "root_services": {"required": ["agents"], "missing": [], "declared": True},
            "presets": [{"preset_id": preset, "declared": True, "preset_mountable": True,
                         "tool_surface_verified": True, "route_resolvable": True,
                         "live_agent_service_ready": True, "first_call_verified": False}
                        for preset in ("ecology-coordinator-v5", "ecology-researcher-v13",
                                       "ecology-candidate-proposer-v5", "ecology-sample-planner-v11",
                                       "ecology-sample-critic-v5", "ecology-generation-judge-v8")],
            "live_agent_service_ready": True, "first_call_verified": False,
        }
        self.server.dsh_native_runtime = native_runtime
        with (patch.object(self.server.datasets, "describe", return_value=description),
              patch.object(self.server.datasets, "series", return_value=series),
              patch.object(self.server.datasets, "selection_view", return_value=selection_view)):
            for explicit in (False, True):
                with self.subTest(explicit_evaluator=explicit):
                    body = {
                        "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                        "domain_pack_id": "greenhouse_environment@1", "dataset_id": "agc_cucumber_2018",
                        "strategy_model_id": "stub/strategy", "review_model_id": "stub/review",
                        "autonomous_mode": True, "prediction_model_id": BASELINE_ALIGNED_RIDGE_MODEL_ID,
                        "budget": {"max_generations": 1, "max_candidates": 4, "candidates_per_generation": 4},
                        "auto_advance": 0, "idempotency_key": f"aligned-runtime-{explicit}",
                    }
                    if explicit:
                        body["evaluator_id"] = GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID
                    status, created = self.request("/runs", "POST", body)
                    self.assertEqual(status, 201, created)
                    run_id = created["projection"]["run_id"]
                    state = self.server.director.state(run_id)
                    metadata = state.task_manifest.metadata
                    self.assertEqual(metadata["evolution_diversity_policy"], "executed_family_rotation@1")
                    self.assertEqual(created["projection"]["evolution_diversity"]["proposed_by_family"], {})
                    from ecologyrsi_dsh.core.model_execution_policy import RESEARCH_EXECUTION_POLICY
                    self.assertEqual(dict(metadata["research_execution_policy"]), dict(RESEARCH_EXECUTION_POLICY))
                    self.assertEqual(created["projection"]["research_execution_policy"], dict(RESEARCH_EXECUTION_POLICY))
                    self.assertEqual(metadata["evaluator_id"], GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID)
                    self.assertEqual(metadata["seed_genome_template_id"], "greenhouse-baseline-aligned-default@1")
                    catalog = metadata["runtime_component_catalog"]["prediction_models"]
                    self.assertEqual([item["id"] for item in catalog], [BASELINE_ALIGNED_RIDGE_MODEL_ID])
                    self.assertEqual(set(catalog[0]["parameter_schemas"]), PARAMETERS)
                    for requested in ("greenhouse-exogenous-ridge@1", "greenhouse-targetwise-ridge@1"):
                        adoption = resolve_predictor_adoption(state.task_manifest,
                            {"prediction_model": {"id": requested},
                             "algorithm_blueprint": {**registered_algorithm_blueprint(requested),
                                                     "evidence_refs": ["knowledge:test-baseline"]}})
                        self.assertEqual(adoption.adopted_id, BASELINE_ALIGNED_RIDGE_MODEL_ID)
                        self.assertEqual(adoption.status, "research_only")
                        self.assertEqual(adoption.reason, "requested_predictor_is_not_in_frozen_compatible_catalog")
                    if state.run.status.value == "created":
                        self.server.director.start_run(run_id)
                    seed_candidate = self.server.director.ensure_seed_incumbent_control(run_id)
                    state = self.server.director.state(run_id)
                    seed_proposal = state.proposal(seed_candidate.proposal_id)
                    self.assertEqual(set(seed_proposal.changes), PARAMETERS)
                    compiled = compile_algorithm_spec(state.task_manifest, seed_proposal, None)
                    self.assertEqual(compiled.adapter_id, BASELINE_ALIGNED_RIDGE_MODEL_ID)
                    self.assertEqual(compiled.evaluator_id, GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID)
        native_runtime.run_stage.assert_not_called()

    def test_a_native_greenhouse_run_freezes_the_deep_alignment_and_others_keep_12(self):
        """One value, frozen once, read by both the planner and the evaluator.

        The alignment decides which origins exist (``_eligible_origins`` bakes it
        into every ``origin_id``) *and* which recipes are servable, so it cannot
        be a package default either side may re-read later. Native greenhouse
        runs freeze the cohort ceiling so recipes are reachable; every other mode
        keeps 12, and the frozen report omits the key at that depth precisely so
        archived reports -- and the ``planner_digest`` a paused run must match on
        resume -- stay byte-identical.
        """
        description = {
            "descriptor": {"runnable": True, "display_name_zh": "测试温室序列",
                           "domain_id": "greenhouse_test", "adapter_id": "greenhouse_test"},
            "readiness": {"ready": True},
        }
        series = SimpleNamespace(digest="d" * 64, split_manifest_digest_sha256="b" * 64,
                                 episode_id="agc_cucumber_2018:test")
        selection_view = SimpleNamespace(
            dataset_id="agc_cucumber_2018", episode_id=series.episode_id,
            timestamps=tuple(range(2000)), partitions={"model_selection": IndexRange(0, 2000)},
            data_protocol_digest="c" * 64, selection_view_digest="e" * 64,
        )
        native_runtime = Mock()
        native_runtime.capabilities.return_value = {
            "schema_version": "ecology-agent-runtime-capabilities/1", "ready": True,
            "root_services": {"required": ["agents"], "missing": [], "declared": True},
            "presets": [{"preset_id": preset, "declared": True, "preset_mountable": True,
                         "tool_surface_verified": True, "route_resolvable": True,
                         "live_agent_service_ready": True, "first_call_verified": False}
                        for preset in ("ecology-coordinator-v5", "ecology-researcher-v13",
                                       "ecology-candidate-proposer-v5", "ecology-sample-planner-v11",
                                       "ecology-sample-critic-v5", "ecology-generation-judge-v8")],
            "live_agent_service_ready": True, "first_call_verified": False,
        }
        self.server.dsh_native_runtime = native_runtime
        with (patch.object(self.server.datasets, "describe", return_value=description),
              patch.object(self.server.datasets, "series", return_value=series),
              patch.object(self.server.datasets, "selection_view", return_value=selection_view)):
            status, created = self.request("/runs", "POST", {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "domain_pack_id": "greenhouse_environment@1", "dataset_id": "agc_cucumber_2018",
                "strategy_model_id": "stub/strategy", "review_model_id": "stub/review",
                "autonomous_mode": True, "prediction_model_id": BASELINE_ALIGNED_RIDGE_MODEL_ID,
                "budget": {"max_generations": 1, "max_candidates": 4, "candidates_per_generation": 4},
                "auto_advance": 0, "idempotency_key": "alignment-freeze-native",
            })
            self.assertEqual(status, 201, created)
            state = self.server.director.state(created["projection"]["run_id"])
            metadata = state.task_manifest.metadata
            self.assertEqual(metadata["origin_history_alignment_hours"], COHORT_HISTORY_HOURS)
            # The single reader agrees with the manifest, so the value-blind
            # planner and the evaluator cannot drift apart on this run.
            self.assertEqual(origin_history_alignment_hours(metadata), COHORT_HISTORY_HOURS)
            report = metadata["cohort_capacity_report"]
            self.assertEqual(report["origin_history_alignment_hours"], COHORT_HISTORY_HOURS)
            # The frozen report is the one the planner re-derives at that depth,
            # which is exactly what the resume-time drift check compares.
            rederived = estimate_epoch_capacity(
                selection_view,
                schedule=OptimizationSchedule.from_dict(metadata["optimization_schedule"]),
                planned_generations=state.task_manifest.max_generations,
                seed=state.task_manifest.seed,
                scoring_cells_per_origin=int(metadata["prediction_cells_per_origin"]),
                history_steps=COHORT_HISTORY_HOURS,
            )
            self.assertEqual(report["planner_digest"], rederived.planner_digest)
            # Planning it one hour shallower is a different cohort, and the
            # digest has to say so rather than rely on the eligible count.
            self.assertNotEqual(
                report["planner_digest"],
                estimate_epoch_capacity(
                    selection_view,
                    schedule=OptimizationSchedule.from_dict(metadata["optimization_schedule"]),
                    planned_generations=state.task_manifest.max_generations,
                    seed=state.task_manifest.seed,
                    scoring_cells_per_origin=int(metadata["prediction_cells_per_origin"]),
                    history_steps=COHORT_HISTORY_HOURS - 1,
                ).planner_digest,
            )
        # A toy run is the other half of the rule: it keeps 12, and its frozen
        # report omits the key entirely so archived reports stay byte-identical.
        status, created = self.request("/runs", "POST", {
            "dataset_id": "generated-toy-series@1", "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "rounds": 1, "candidates_per_generation": 4, "max_candidates": 4,
            "auto_advance": 0, "idempotency_key": "alignment-freeze-toy",
        })
        self.assertEqual(status, 201, created)
        metadata = self.server.director.state(
            created["projection"]["run_id"]).task_manifest.metadata
        self.assertEqual(metadata["origin_history_alignment_hours"],
                         MAX_EXOGENOUS_RIDGE_HISTORY_STEPS)
        self.assertEqual(origin_history_alignment_hours(metadata),
                         MAX_EXOGENOUS_RIDGE_HISTORY_STEPS)
        self.assertNotIn("origin_history_alignment_hours", metadata["cohort_capacity_report"])


if __name__ == "__main__":
    unittest.main()
