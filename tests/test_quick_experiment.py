"""Quick epochs retain causal evidence while removing repeated candidate work."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import replace
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evaluators.epoch_cohorts import plan_run_adaptation_cohort, plan_generation_selection_cohorts
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from ecologyrsi_dsh.application import formal_trajectory
from ecologyrsi_dsh.core.trajectory import HoldoutArm, TrajectoryStatus
from ecologyrsi_dsh.core.models import TaskManifest
from ecologyrsi_dsh.core.state import validate_generation_comparison_binding
from ecologyrsi_dsh.core.trajectory import GenerationHoldout
from tests.test_epoch_cohort_planning import dataset_fixture
from tests.test_run_parameter_consistency import small_holdout
from tests.test_formal_trajectory import PairedFormalTrajectoryTests


class QuickPlanTests(unittest.TestCase):
    def test_first_quick_batch_failure_is_not_reported_as_screening(self):
        from ecologyrsi_dsh.api.auto_progress import _failure_diagnostics
        from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
        schedule = OptimizationSchedule.for_new_run()
        state = SimpleNamespace(
            run=SimpleNamespace(generation=0),
            task_manifest=SimpleNamespace(metadata={
                "optimization_protocol": schedule.protocol,
                "optimization_schedule": schedule.to_dict(),
            }),
            candidate_screening_events=[], formal_batch_evaluations=[],
            formal_trajectories=[], formal_batches=[],
        )
        error = DshNativeRuntimeUnavailableError(
            error_code="structured_child_output_budget_exhausted", status_code=422,
        )
        code, context = _failure_diagnostics(state, error)
        self.assertEqual(code, error.error_code)
        self.assertEqual(context["stage"], "formal_batch")
        self.assertEqual(context["work_unit_kind"], "formal_batch")

    def test_default_budget_and_fresh_value_blind_holdouts(self):
        schedule = OptimizationSchedule.for_new_run()
        plan = schedule.execution_plan(5, cells_per_origin=9)
        self.assertEqual(plan["generation_budget"]["total_candidate_origins"], 200)
        self.assertEqual(plan["run_budget"]["total_candidate_origins"], 1000)
        self.assertEqual(plan["required_unique_origins"], 350)
        self.assertEqual(schedule.planned_origin_occurrences(5), 750)
        all_plans = []
        for changed in (False, True):
            dataset = dataset_fixture(3000, changed_labels=changed)
            adaptation = plan_run_adaptation_cohort(dataset, schedule=schedule, seed=0)
            holdouts = [plan_generation_selection_cohorts(dataset, schedule=schedule, generation=g, adaptation=adaptation, seed=0) for g in range(5)]
            self.assertTrue(all(h.screening.origin_count == 0 for h in holdouts))
            origins = [o.origin_id for h in holdouts for o in h.holdout.origins]
            self.assertEqual(len(set(origins)), 250)
            previous = adaptation.origins[-1].maximum_target_timestamp
            for h in holdouts:
                self.assertGreater(h.holdout.origins[0].origin_timestamp, previous)
                previous = h.holdout.origins[-1].maximum_target_timestamp
            all_plans.append((adaptation, holdouts))
        self.assertEqual(all_plans[0], all_plans[1])

    def test_quick_selection_never_claims_certification(self):
        candidate = small_holdout(HoldoutArm.FINALIST_1, .2)
        incumbent = small_holdout(HoldoutArm.INCUMBENT, .1)
        decision = build_generation_comparison(run_id=candidate.scope.run_id, generation=0,
            cohort_digest=candidate.scope.cohort_digest, holdout_evaluations=(candidate, incumbent), quick_experiment=True)
        self.assertEqual(decision.selected_candidate_id, candidate.scope.candidate_id)
        self.assertIsNone(decision.gate_results["certification_selected_arm"])
        self.assertEqual(decision.gate_results["experiment_protocol"], "quick_adaptive_epoch@1")

    def test_exploratory_protocol_identity_is_frozen_and_checked_on_replay(self):
        candidate = small_holdout(HoldoutArm.FINALIST_1, .2)
        incumbent = small_holdout(HoldoutArm.INCUMBENT, .1)
        inputs = dict(run_id=candidate.scope.run_id, generation=0,
            cohort_digest=candidate.scope.cohort_digest,
            holdout_evaluations=(candidate, incumbent), quick_experiment=True)
        holdout = GenerationHoldout(holdout_id="holdout:protocol-binding", run_id=candidate.scope.run_id,
            generation=0, cohort_digest=candidate.scope.cohort_digest, origin_count=50,
            arm_bindings={item.scope.holdout_arm.value: {"candidate_id": item.scope.candidate_id,
                "candidate_revision_id": item.scope.candidate_revision_id} for item in (candidate, incumbent)})
        for schedule in (OptimizationSchedule.for_new_run(), OptimizationSchedule.for_evidence_guided_run()):
            with self.subTest(protocol=schedule.protocol):
                decision = build_generation_comparison(**inputs, experiment_protocol=schedule.protocol)
                self.assertEqual(decision.gate_results["experiment_protocol"], schedule.protocol)
                self.assertIsNone(decision.gate_results["certification_selected_arm"])
                task = TaskManifest(task_id="task:protocol-binding", objective="replay frozen protocol",
                    domain_pack="greenhouse_environment@1", visible_datasets=("dataset",),
                    budget={"max_candidates": 4, "max_generations": 1}, metadata={
                        "execution_protocol": "dsh_native_plugin_evolution@1",
                        "host_runtime_build": {"evolution_runtime_schema": "ecologyrsi-dsh.evolution-runtime/3"},
                        "optimization_protocol": schedule.protocol,
                        "optimization_schedule": schedule.to_dict()})
                validate_generation_comparison_binding(task, candidate.scope.run_id, holdout, {}, decision,
                    persisted_evaluations={item.scope.holdout_arm: item for item in (candidate, incumbent)})
                wrong = "quick_adaptive_epoch@1" if schedule.race else "evidence_guided_epoch@1"
                with self.assertRaisesRegex(ValueError, "deterministic Host comparison"):
                    validate_generation_comparison_binding(task, candidate.scope.run_id, holdout, {},
                        build_generation_comparison(**inputs, experiment_protocol=wrong),
                        persisted_evaluations={item.scope.holdout_arm: item for item in (candidate, incumbent)})


class QuickTrajectoryTests(unittest.TestCase):
    quick_protocol = True
    setUp = PairedFormalTrajectoryTests.setUp
    tearDown = PairedFormalTrajectoryTests.tearDown
    _revision_inputs = staticmethod(PairedFormalTrajectoryTests._revision_inputs)
    _local_context = staticmethod(PairedFormalTrajectoryTests._local_context)
    _mutate_proposal = staticmethod(PairedFormalTrajectoryTests._mutate_proposal)
    _execution_patches = PairedFormalTrajectoryTests._execution_patches

    def test_single_lane_prequential_updates_and_two_arm_freeze_replay(self):
        candidate_id = self.finalist.candidate_id
        patches = self._execution_patches(self._mutate_proposal())
        with patches[0], patches[1], patches[2], patches[3], patches[4] as editor:
            for _ in range(self.schedule.batch_count):
                self.assertTrue(formal_trajectory.execute_next_formal_batch(self.endpoint.server, self.run_id, candidate_id))
                self.assertTrue(formal_trajectory.execute_next_local_edit(self.endpoint.server, self.run_id, candidate_id))
            self.assertGreaterEqual(editor.call_count, 1)
        state = self.director.state(self.run_id)
        trajectory = state.trajectory_for(candidate_id)
        self.assertEqual(trajectory.status, TrajectoryStatus.COMPLETED)
        self.assertEqual(len(state.formal_batch_evaluations), self.schedule.batch_count)
        self.assertFalse(state.formal_batch_comparisons)
        control = self.candidates[1]
        holdout = self.director.freeze_generation_holdout(self.run_id, 0, state.generation_cohort_for(0).holdout.cohort_digest, {
            "finalist_1": {"candidate_id": candidate_id, "candidate_revision_id": trajectory.final_revision_id},
            "incumbent": {"candidate_id": control.candidate_id, "candidate_revision_id": self.revisions[control.candidate_id].revision_id},
        })
        self.assertEqual(len(holdout.arm_bindings), 2)
        self.assertEqual(self.director.state(self.run_id).generation_holdout_for(0), holdout)
        # The final batch must not author a revision that this trajectory never
        # scored, so completion can only promote the last measured revision.
        self.assertFalse([
            revision.revision_id
            for revision in state.candidate_revisions
            if revision.candidate_id == candidate_id
            and revision.source_batch_index is not None
            and revision.source_batch_index >= self.schedule.batch_count - 1
        ])
        self.assertEqual(
            trajectory.final_revision_id,
            state.formal_batch_for(candidate_id, self.schedule.batch_count - 1).revision_id,
        )

    def test_prequential_score_regression_rolls_back_through_the_real_ledger(self):
        """A real ledger accepts the regression rollback and keeps a measured revision."""
        candidate_id = self.finalist.candidate_id
        scores = {0: 0.50, 1: 0.50, 2: 0.10}
        evaluate = self.evaluator.evaluate_scientific

        def scored_batch(*args, **kwargs):
            result = evaluate(*args, **kwargs)
            score = scores[int(kwargs["scope"].batch_index)]
            result.evaluation.score = score
            result.evaluation.metrics["objective_score"] = score
            for target in result.evaluation.metrics["targets"]:
                target["skill_score"] = score
            return result

        def distinct_edit(*_args, **_kwargs):
            from ecologyrsi_dsh.evolution.local_edits import LocalEditProposal

            distinct_edit.calls += 1
            return LocalEditProposal(
                decision="mutate",
                operations=(
                    {
                        "op": "set_bounded_parameter",
                        "name": "ridge_alpha",
                        "value": 0.1 * distinct_edit.calls,
                    },
                ),
                evidence_refs=("batch:score",),
                expected_effect_cells=("air_temperature@1h",),
                risk_cells=(),
            )

        distinct_edit.calls = 0
        patches = self._execution_patches(self._mutate_proposal())
        with patches[0], patches[1], patches[2], patches[3], patch.object(
            formal_trajectory, "_local_edit_proposal", side_effect=distinct_edit
        ), patch.object(
            self.evaluator, "evaluate_scientific", side_effect=scored_batch
        ):
            for _ in range(self.schedule.batch_count):
                self.assertTrue(formal_trajectory.execute_next_formal_batch(self.endpoint.server, self.run_id, candidate_id))
                self.assertTrue(formal_trajectory.execute_next_local_edit(self.endpoint.server, self.run_id, candidate_id))

        # Batch 2 fell far outside the lane's own realized volatility, so the
        # edit authored on batch 1 is undone instead of being built upon.
        self.assertEqual(distinct_edit.calls, 2)
        state = self.director.state(self.run_id)
        final_batch = self.schedule.batch_count - 1
        parent_revision_id = state.formal_batch_for(candidate_id, final_batch - 1).revision_id
        outcome = next(
            item
            for item in state.local_edit_outcomes
            if item["candidate_id"] == candidate_id
            and item["batch_index"] == final_batch
        )
        self.assertEqual(outcome["outcome"], "rolled_back")
        self.assertEqual(outcome["reason"], "score_regression_guardrail_failed")
        self.assertEqual(
            state.local_edit_proposal_for(candidate_id, final_batch)["safety_reason"],
            "score_regression_guardrail_failed",
        )
        activation = state.revision_activation_for(candidate_id, final_batch)
        self.assertEqual(activation.reason.value, "prequential_safety_rollback")
        self.assertEqual(activation.to_revision_id, parent_revision_id)
        trajectory = state.trajectory_for(candidate_id)
        self.assertEqual(trajectory.status, TrajectoryStatus.COMPLETED)
        # The promoted revision is the one batch 1 actually measured.
        self.assertEqual(trajectory.final_revision_id, parent_revision_id)
        self.assertIsNotNone(state.batch_evaluation_for(candidate_id, final_batch - 1))
