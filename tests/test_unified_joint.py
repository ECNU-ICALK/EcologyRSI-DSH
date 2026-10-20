"""Unified admission, bounded confirmation and frozen historical schedules."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.trajectory import HoldoutArm
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from tests.test_run_parameter_consistency import small_holdout


def pair(gain=.108, *, replica=0):
    values = [small_holdout(HoldoutArm.FINALIST_1, gain), small_holdout(HoldoutArm.INCUMBENT, .1)]
    return tuple(replace(item, metrics=item.to_dict()["metrics"], scope=replace(item.scope, inference_replica=replica)) for item in values)


def compare(values, repeats=()):
    return build_generation_comparison(run_id=values[0].scope.run_id, generation=0,
        cohort_digest=values[0].scope.cohort_digest, holdout_evaluations=values,
        quick_experiment=True, experiment_protocol="joint_plugin_epoch@2", holdout_rechecks=repeats)


class UnifiedJointTests(unittest.TestCase):
    def test_default_reserves_one_confirmation_without_inventing_observations(self):
        schedule = OptimizationSchedule.for_new_run()
        self.assertEqual(schedule.protocol, "joint_plugin_epoch@2")
        plan = schedule.execution_plan(3, cells_per_origin=9)
        budget = plan['generation_budget']
        self.assertEqual((budget['base_candidate_origins'], budget['recheck_candidate_origins'],
                          budget['total_candidate_origins']), (200, 100, 300))
        self.assertEqual(plan['run_budget']['total_candidate_origins'], 900)
        self.assertEqual(plan['required_unique_origins'], 255)
        self.assertEqual(plan['maximum_rechecks_per_generation'], 1)
        self.assertEqual(OptimizationSchedule.from_dict(schedule.to_dict()), schedule)
        for legacy in (OptimizationSchedule.for_quick_run(), OptimizationSchedule.for_joint_run(),
                       OptimizationSchedule.for_evidence_guided_run()):
            self.assertFalse(legacy.unified)
            self.assertEqual(legacy.generation_execution_budget(cells_per_origin=9)['total_candidate_origins'], 200)
            self.assertEqual(OptimizationSchedule.from_dict(legacy.to_dict()), legacy)

    def test_clear_consistent_gain_needs_no_repeat_and_is_not_certification(self):
        result = compare(pair(.2))
        gate = result.gate_results['arms']['finalist_1']
        self.assertTrue(gate['search_eligible'])
        self.assertFalse(gate['joint_recheck']['required'])
        self.assertFalse(gate['certification_eligible'])
        self.assertIsNone(result.gate_results['certification_selected_arm'])

    def test_marginal_gain_requires_both_complete_paired_results(self):
        pending = compare(pair())
        gate = pending.gate_results['arms']['finalist_1']
        self.assertTrue(gate['joint_recheck']['required'])
        self.assertFalse(gate['search_eligible'])
        self.assertEqual(pending.selected_candidate_id, 'candidate:incumbent')
        confirmed = compare(pair(), pair(.11, replica=1))
        self.assertTrue(confirmed.gate_results['arms']['finalist_1']['search_eligible'])
        self.assertEqual(confirmed.gate_results['arms']['finalist_1']['joint_recheck']['status'], 'confirmed')
        failed = compare(pair(), pair(.099, replica=1))
        self.assertEqual(failed.selected_candidate_id, 'candidate:incumbent')
        self.assertEqual(failed.gate_results['arms']['finalist_1']['joint_recheck']['status'], 'not_confirmed')

    def test_failed_first_comparison_cannot_be_rescued_by_repeated_sampling(self):
        values = pair(.099)
        self.assertFalse(compare(values).gate_results['arms']['finalist_1']['joint_recheck']['required'])
        with self.assertRaisesRegex(ValueError, 'not triggered'):
            compare(values, pair(.2, replica=1))

    def test_confirmation_rejects_another_revision_or_extra_attempt(self):
        repeated = pair(.11, replica=1)
        for invalid in ((replace(repeated[0], metrics=repeated[0].to_dict()["metrics"], scope=replace(repeated[0].scope,
                           candidate_revision_id='another-revision')), repeated[1]),
                        (*repeated, repeated[0]), (repeated[0],)):
            with self.subTest(count=len(invalid)), self.assertRaises(ValueError):
                compare(pair(), invalid)

    def test_missing_time_coverage_retains_incumbent_without_wasting_repeats(self):
        values = []
        for item in pair(.2):
            metrics = item.to_dict()['metrics']
            evidence = metrics['promotion_block_evidence']
            evidence['blocks'] = evidence['blocks'][:2]
            evidence['block_count'] = 2
            evidence['evidence_digest'] = digest({k: v for k, v in evidence.items() if k != 'evidence_digest'})
            values.append(replace(item, metrics=metrics))
        gate = compare(values).gate_results['arms']['finalist_1']
        self.assertFalse(gate['search_eligible'])
        self.assertFalse(gate['joint_recheck']['required'])
        self.assertIn('additional_time_blocks_required', gate['search_failures'])

    def test_failed_repeat_cannot_hide_behind_its_positive_overall_score(self):
        repeated = list(pair(.11, replica=1))
        metrics = repeated[0].to_dict()['metrics']
        metrics['targets'][0]['skill_score'] = -.2
        repeated[0] = replace(repeated[0], metrics=metrics)
        gate = compare(pair(), repeated).gate_results['arms']['finalist_1']
        self.assertFalse(gate['search_eligible'])
        self.assertIn('cell_regression', gate['joint_recheck']['repeated_failures'])

    def test_recheck_resumes_its_own_checkpoint_and_seals_once_in_real_ledger(self):
        from ecologyrsi_dsh import EvolutionDirector, FakeDSHAdapter
        from ecologyrsi_dsh.application import generation_execution as execution
        from ecologyrsi_dsh.core.sample_results import build_sample_results
        from tests.test_evidence_guided_epoch import EvidenceGuidedRaceTests

        fixture = EvidenceGuidedRaceTests()
        self.addCleanup(fixture.doCleanups)
        with patch.object(OptimizationSchedule, 'for_evidence_guided_run', OptimizationSchedule.for_new_run):
            fixture.setUp()
        fixture.test_five_arms_freeze_and_resume_without_rescoring_warmup()
        state = fixture.director.state(fixture.run_id)
        finalist = fixture.candidates[0]
        bindings = {
            'finalist_1': {'candidate_id': finalist.candidate_id,
                'candidate_revision_id': state.trajectory_for(finalist.candidate_id).final_revision_id},
            'incumbent': {'candidate_id': fixture.control.candidate_id,
                'candidate_revision_id': state.initial_revision_for(fixture.control.candidate_id).revision_id},
        }
        fixture.director.freeze_generation_holdout(fixture.run_id, 0,
            fixture.cohorts.holdout.cohort_digest, bindings)
        originals = []
        for value in pair():
            bound = bindings[value.scope.holdout_arm.value]
            scope = replace(value.scope, run_id=fixture.run_id, **bound,
                            cohort_digest=fixture.cohorts.holdout.cohort_digest)
            original = replace(value, scope=scope, metrics=value.to_dict()['metrics'])
            fixture.director.record_holdout_evaluation(fixture.run_id, original)
            originals.append(original)
        original = originals[1]
        from ecologyrsi_dsh.api.projection import _execution_diagnostics
        before_counts = _execution_diagnostics(fixture.director.state(fixture.run_id))
        rows = [{'sample_id': f'recheck:{i}', 'sample_index': i + 1, 'target': 'air_temperature',
                 'horizon_hours': 1, 'origin_timestamp': i, 'target_timestamp': i + 1,
                 'observed': 2., 'predicted': 1.5, 'baseline': 1.} for i in range(50)]
        restored_counts = []

        def evaluate(task, candidate, proposal, *, scope, **kwargs):
            restored = kwargs['on_sample_checkpoint']({
                'schema_version': 'ecologyrsi-dsh.sample-checkpoint/2',
                'candidate_revision_id': scope.candidate_revision_id, 'evaluation_phase': 'holdout',
                'formal_batch_index': None, 'holdout_arm': scope.holdout_arm.value,
                'cohort_digest': scope.cohort_digest, 'execution_scope_digest': scope.scope_key,
                'inference_replica': 1, 'sample_cohort_digest': 'a' * 64,
                'execution_context_digest': 'b' * 64, 'sample_count': 50})
            restored_counts.append(len(restored['rows']))
            if not restored['rows']:
                kwargs['on_sample_results'](rows[:10])
                raise RuntimeError('simulated interruption')
            kwargs['on_sample_results'](rows[10:])
            return SimpleNamespace(evaluation=original,
                sample_results=build_sample_results(candidate.candidate_id, rows))

        services = SimpleNamespace(director=fixture.director, ledger=fixture.ledger,
                                   evaluators=SimpleNamespace(evaluate_scientific=evaluate))
        with patch.object(execution, '_holdout_replay_inputs',
                          return_value=(fixture.control, object(), object())):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                execution._execute_joint_holdout_recheck(services, fixture.run_id, original, fixture.cohorts.holdout)
            from ecologyrsi_dsh.api.projection import _holdout_arm_live_projection
            live = _holdout_arm_live_projection(services.director.state(fixture.run_id))
            self.assertEqual(live['outcome_scope'], 'active_holdout_recheck')
            self.assertEqual(live['completed_origins'], 10)
            services.director = EvolutionDirector(fixture.ledger, FakeDSHAdapter())
            sealed = execution._execute_joint_holdout_recheck(services, fixture.run_id, original, fixture.cohorts.holdout)
            again = execution._execute_joint_holdout_recheck(services, fixture.run_id, original, fixture.cohorts.holdout)
        self.assertEqual(restored_counts, [0, 10])
        self.assertEqual(sealed, again)
        replayed = services.director.replay(fixture.run_id)
        self.assertEqual(len(replayed.holdout_evaluations), 2)
        self.assertEqual(len(replayed.holdout_rechecks), 1)
        self.assertEqual(sum(event.kind == 'HoldoutRecheckStarted' for event in replayed.events), 1)
        self.assertEqual(replayed.holdout_evaluation_for(0, HoldoutArm.INCUMBENT), original)
        after_counts = _execution_diagnostics(replayed)
        # This fixture has one score cell per origin. A durable second
        # inference is 50 additional executions, not merely another view of
        # the first result. Resume and exact replay add no third execution.
        self.assertEqual(before_counts["adaptive_all_settled_origins"], 200)
        self.assertEqual(after_counts["adaptive_all_settled_origins"], 250)
        self.assertEqual(after_counts["adaptive_scored_cells"], 250)
        self.assertEqual(after_counts["candidate_work_items"], 250)
        self.assertEqual(after_counts["adaptive_evaluation_count"], 6)
        self.assertEqual(after_counts["adaptive_succeeded_origins"], 200)
        self.assertEqual(after_counts["adaptive_failed_origins"], 0)
        self.assertTrue(after_counts["adaptive_execution_counts_known"])
        self.assertTrue(after_counts["adaptive_outcome_counts_known"])
        self.assertEqual(after_counts, _execution_diagnostics(services.director.replay(fixture.run_id)))
        # A restart must not discard a completed confirmation arm and seal the
        # first comparison as though that durable evidence never existed.
        from ecologyrsi_dsh.core.state import validate_generation_comparison_binding
        with self.assertRaisesRegex(ValueError, 'durable recheck evidence'):
            validate_generation_comparison_binding(replayed.task_manifest, fixture.run_id,
                replayed.generation_holdout_for(0), replayed.formal_selection_for(0), compare(originals),
                persisted_evaluations={item.scope.holdout_arm: item for item in originals},
                persisted_rechecks=replayed.holdout_rechecks)


if __name__ == '__main__':
    unittest.main()
