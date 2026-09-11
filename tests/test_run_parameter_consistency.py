"""Default, admission and actual epoch decisions share one parameter contract."""
from dataclasses import replace
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.evolution.parameters import run_parameter_contract, validate_run_parameter
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evolution.evidence_capacity import require_guarded_cohort_evidence_capacity
from ecologyrsi_dsh.evolution.promotion import build_promotion_block_evidence
from ecologyrsi_dsh.evaluators.fitness import FitnessProfile
from ecologyrsi_dsh.evaluators.agent_stability import replica_summary, stability_evidence
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from ecologyrsi_dsh.core.trajectory import HoldoutArm
from tests.test_epoch_cohort_planning import dataset_fixture
from tests.test_generation_comparison import _evaluation, TARGETS, HORIZONS


class ParameterConsistencyTests(unittest.TestCase):
    def test_browser_bootstrap_matches_the_published_contract(self):
        source = (Path(__file__).resolve().parents[1] / 'plugins/ecology_evolution/assets/js/commands.js').read_text()
        raw = re.search(r'var bootstrapRunParameters = (.*);', source).group(1)
        self.assertEqual(json.loads(raw), run_parameter_contract())
        for name, value in [('rounds', 5.5), ('rounds', 51), ('max_candidates', 257),
                            ('candidates_per_generation', 3), ('local_batch_origin_count', True)]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_run_parameter(name, value)

    def test_training_rules_do_not_change_at_49_or_169_samples(self):
        profile = FitnessProfile()
        default = OptimizationSchedule.for_comparison_run()
        for count in (10, 48, 49, 50, 72):
            schedule = replace(default, formal_origin_count_per_finalist=count * 2, local_batch_origin_count=count)
            self.assertTrue(schedule.exploratory_local_comparison)
        for count in (40, 50, 168, 169, 200):
            self.assertEqual(profile.minimum_origins_for_schedule(replace(default, selection_holdout_origin_count=count)), 40)
        custom = profile.with_overrides(selection_minimum_cell_samples=60)
        self.assertEqual(run_parameter_contract(custom)['parameters']['selection_holdout_origin_count']['minimum'], 60)
        with self.assertRaisesRegex(ValueError, '60'):
            validate_run_parameter('selection_holdout_origin_count', 50, profile=custom)
        with self.assertRaisesRegex(ValueError, '60 个不同起点'):
            require_guarded_cohort_evidence_capacity(dataset=dataset_fixture(1000), schedule=default,
                                                     planned_generations=1, seed=0, profile=custom)

    def test_eight_disconnected_days_cannot_supply_moving_block_evidence(self):
        holdout = SimpleNamespace(cohort_digest='c' * 64, origin_count=50,
            origin_ids=tuple(f'o:{i}' for i in range(50)),
            origins=tuple(SimpleNamespace(origin_timestamp=(i * 8 // 50) * 48 + i % 6) for i in range(50)))
        with patch('ecologyrsi_dsh.evolution.evidence_capacity.plan_generation_selection_cohorts',
                   return_value=SimpleNamespace(holdout=holdout)):
            with self.assertRaisesRegex(ValueError, '连续 3 日窗口.*0 个窗口'):
                require_guarded_cohort_evidence_capacity(dataset=dataset_fixture(1000),
                    schedule=OptimizationSchedule.for_comparison_run(), planned_generations=1, seed=0)


def small_holdout(arm, gain, *, second_gain=None):
    item = _evaluation(arm, 'candidate:' + arm.value, 'revision:' + arm.value, gain, skill=gain)
    scope = replace(item.scope, origin_count=50)
    metrics = item.to_dict()['metrics']
    metrics['sample_execution'].update(coverage=1., attempted_origin_samples=50, succeeded_origin_samples=50,
        complete_origin_agent_chains=50, failed_origin_samples=0, failed_examples=0, scoring_fallback_examples=0)
    rows = [dict(target=target, horizon_hours=horizon, origin_timestamp=i * 168 // 49,
                 observed=0., predicted=1. - gain, baseline=1., normalization_scale=1., status='succeeded')
            for i in range(50) for target in TARGETS for horizon in HORIZONS]
    metrics['promotion_block_evidence'] = build_promotion_block_evidence(rows,
        horizons=HORIZONS, target_weights=metrics['objective_target_weights'],
        dataset_digest='d' * 64, split_manifest_digest_sha256='e' * 64)
    for cell in metrics['targets']:
        cell.update(n=50, paired_block_count=8)
    result = replace(item, scope=scope, metrics=metrics)
    second = replace(result, metrics=metrics, score=gain if second_gain is None else second_gain)
    metrics['prediction_owner'] = 'sample_agent'
    metrics['agent_inference_stability'] = stability_evidence([
        replica_summary(scope, result), replica_summary(replace(scope, inference_replica=1), second)])
    return replace(result, metrics=metrics)


class SmallEpochDecisionTests(unittest.TestCase):
    def compare(self, second_gain):
        a = small_holdout(HoldoutArm.FINALIST_1, .2, second_gain=second_gain)
        b = small_holdout(HoldoutArm.FINALIST_2, 0.)
        incumbent = small_holdout(HoldoutArm.INCUMBENT, 0.)
        return build_generation_comparison(run_id=a.scope.run_id, generation=0,
            cohort_digest=a.scope.cohort_digest, holdout_evaluations=(a, b, incumbent),
            fitness_profile=FitnessProfile(), require_paired_strict_chain=True)

    def test_fifty_real_scored_origins_can_select_the_next_epoch_parent(self):
        comparison = self.compare(.2)
        self.assertEqual(comparison.selected_candidate_id, 'candidate:finalist_1')

    def test_a_failed_second_inference_cannot_supply_the_next_epoch_parent(self):
        comparison = self.compare(-.01)
        self.assertEqual(comparison.selected_candidate_id, 'candidate:incumbent')
        gate = comparison.gate_results['arms']['finalist_1']
        self.assertFalse(gate['eligible'])
        self.assertFalse(gate['certification_eligible'])
        self.assertIn('gain_not_stable_across_agent_inference', gate['failures'])
