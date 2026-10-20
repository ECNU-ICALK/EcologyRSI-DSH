"""Value-blind admission and joint retention share their temporal requirement."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.evaluators.joint_recheck import MINIMUM_TIME_BLOCKS
from ecologyrsi_dsh.evolution.evidence_capacity import (
    guarded_cohort_evidence_capacity,
    require_guarded_cohort_evidence_capacity,
)
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from tests.test_epoch_cohort_planning import dataset_fixture


class JointCapacityAlignmentTests(unittest.TestCase):
    def test_unified_report_uses_retention_requirement_and_legacy_reports_stay_frozen(self):
        for schedule, expected in (
            (OptimizationSchedule.for_new_run(), MINIMUM_TIME_BLOCKS),
            (OptimizationSchedule.for_joint_run(), 2),
            (OptimizationSchedule.for_evidence_guided_run(), 2),
            (OptimizationSchedule.for_quick_run(), 2),
        ):
            with self.subTest(protocol=schedule.protocol):
                report = require_guarded_cohort_evidence_capacity(
                    dataset=dataset_fixture(2000), schedule=schedule,
                    planned_generations=2, seed=7)
                self.assertEqual([row['minimum_day_blocks'] for row in report['selection_holdouts']],
                                 [expected, expected])
                self.assertTrue(report['sufficient'])

    def test_two_day_planner_output_cannot_pass_unified_admission(self):
        # Exercise the admission boundary independently of the current fixed
        # 50-origin planner, whose hourly output already spans at least 3 days.
        # This plausible reduced cohort has enough distinct origins for the
        # generic count guard, but can never pass the joint temporal gate.
        holdout = SimpleNamespace(cohort_digest='c'*64, origin_count=40,
            origin_ids=tuple(f'o:{index}' for index in range(40)),
            origins=tuple(SimpleNamespace(origin_timestamp=index) for index in range(40)))
        selection = SimpleNamespace(holdout=holdout, adaptation=SimpleNamespace(batches=()))
        with patch('ecologyrsi_dsh.evolution.evidence_capacity.plan_generation_selection_cohorts',
                   return_value=selection):
            report = guarded_cohort_evidence_capacity(dataset_fixture(2000),
                schedule=OptimizationSchedule.for_new_run(), planned_generations=1, seed=7)
            evidence = report['selection_holdouts'][0]
            self.assertEqual(evidence['day_block_count'], 2)
            self.assertGreaterEqual(evidence['unique_origin_count'], evidence['minimum_origin_count'])
            self.assertFalse(report['sufficient'])
            with self.assertRaisesRegex(ValueError, '选择留出 cohort 仅覆盖 2 个日块.*至少需要 3'):
                require_guarded_cohort_evidence_capacity(dataset=dataset_fixture(2000),
                    schedule=OptimizationSchedule.for_new_run(), planned_generations=1, seed=7)
            for legacy in (OptimizationSchedule.for_joint_run(), OptimizationSchedule.for_evidence_guided_run()):
                with self.subTest(legacy=legacy.protocol):
                    unchanged = require_guarded_cohort_evidence_capacity(dataset=dataset_fixture(2000),
                        schedule=legacy, planned_generations=1, seed=7)
                    self.assertTrue(unchanged['sufficient'])
                    self.assertEqual(unchanged['selection_holdouts'][0]['minimum_day_blocks'], 2)


if __name__ == '__main__':
    unittest.main()
