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
    def test_defaults_construct_current_protocol_and_cover_candidate_budget(self):
        defaults = {name: rule['default'] for name, rule in run_parameter_contract()['parameters'].items()}
        self.assertEqual(defaults['formal_origin_count'], 10 + defaults['local_batch_origin_count'])
        self.assertGreaterEqual(defaults['max_candidates'], defaults['rounds'] * defaults['candidates_per_generation'])
        self.assertEqual(OptimizationSchedule.for_new_run().protocol, 'joint_plugin_epoch@2')

    def test_browser_bootstrap_matches_the_published_contract(self):
        source = (Path(__file__).resolve().parents[1] / 'plugins/ecology_evolution/assets/js/commands.js').read_text()
        raw = re.search(r'var bootstrapRunParameters = (.*);', source).group(1)
        self.assertEqual(json.loads(raw), run_parameter_contract())
        for name, value in [('rounds', 5.5), ('rounds', 51), ('max_candidates', 257),
                            ('candidates_per_generation', 3), ('local_batch_origin_count', True), ('local_batch_origin_count', 10), ('formal_origin_count', 100), ('selection_holdout_origin_count', 169)]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_run_parameter(name, value)


    def test_current_temporal_capacity_uses_unique_origins_and_frozen_day_blocks(self):
        from ecologyrsi_dsh.evolution.evidence_capacity import guarded_cohort_evidence_capacity
        report=guarded_cohort_evidence_capacity(dataset_fixture(3000),
            schedule=OptimizationSchedule.for_new_run(),planned_generations=2,seed=0)
        self.assertTrue(report['sufficient'])
        for holdout in report['selection_holdouts']:
            self.assertEqual(holdout['unique_origin_count'],50)
            self.assertEqual(holdout['minimum_day_blocks'],3)
            self.assertGreaterEqual(holdout['day_block_count'],3)
        self.assertEqual(FitnessProfile().minimum_origins_for_schedule(OptimizationSchedule.for_new_run()),40)
        profile=FitnessProfile().with_overrides(selection_minimum_cell_samples=60)
        self.assertEqual(profile.minimum_origins_for_schedule(OptimizationSchedule.for_new_run()),60)
        with self.assertRaisesRegex(ValueError,'below the fitness sample minimum'):
            run_parameter_contract(profile)
        with self.assertRaisesRegex(ValueError,'60 个不同起点'):
            require_guarded_cohort_evidence_capacity(dataset=dataset_fixture(3000),
                schedule=OptimizationSchedule.for_new_run(),planned_generations=1,seed=0,profile=profile)


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
