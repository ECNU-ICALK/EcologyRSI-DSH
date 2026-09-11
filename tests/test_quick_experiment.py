"""Quick epochs retain causal evidence while removing repeated candidate work."""
import unittest
from types import SimpleNamespace
from dataclasses import replace
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evaluators.epoch_cohorts import plan_run_adaptation_cohort, plan_generation_selection_cohorts
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from ecologyrsi_dsh.application import formal_trajectory
from ecologyrsi_dsh.core.trajectory import HoldoutArm, TrajectoryStatus
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
