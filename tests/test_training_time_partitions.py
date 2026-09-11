from dataclasses import replace
import unittest

from ecologyrsi_dsh.data.splits import build_split_manifest
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evaluators.epoch_cohorts import (
    estimate_epoch_capacity, plan_run_adaptation_cohort, plan_generation_selection_cohorts,
)
from tests.test_epoch_cohort_planning import dataset_fixture
from tests.test_formal_stage_protocol import _Episode


class TrainingEpochTests(unittest.TestCase):
    def test_small_default_is_runnable_and_spreads_comparison_over_eight_days(self):
        from ecologyrsi_dsh.evolution.evidence_capacity import require_guarded_cohort_evidence_capacity
        from ecologyrsi_dsh.core.search_policy import SEARCH_GUARD_POLICY, local_challenger_policy
        from ecologyrsi_dsh.evaluators.fitness import FitnessProfile
        schedule = OptimizationSchedule.for_comparison_run()
        self.assertEqual((schedule.formal_origin_count_per_finalist, schedule.local_batch_origin_count,
                          schedule.selection_holdout_origin_count, schedule.batch_count), (100, 10, 50, 10))
        data = dataset_fixture(800)
        report = require_guarded_cohort_evidence_capacity(dataset=data, schedule=schedule,
                                                        planned_generations=5, seed=7)
        self.assertTrue(report["sufficient"])
        self.assertEqual(report["local_comparison_mode"], "exploratory_paired_point_comparison")
        self.assertTrue(all(b["origin_count"] == 10 for b in report["formal_batches"]))
        self.assertTrue(all(h["origin_count"] == 50 and h["day_block_count"] >= 8
                            for h in report["selection_holdouts"]))
        profile = FitnessProfile()
        self.assertEqual(profile.minimum_origins_for_schedule(schedule), 40)
        self.assertEqual(profile.selection_minimum_paired_blocks, 8)
        policy = local_challenger_policy({"search_guard_policy": SEARCH_GUARD_POLICY,
                                          "local_comparison_policy": "exploratory_paired_point_comparison",
                                          "fitness_profile": profile.to_dict(),
                                          "optimization_schedule": schedule.to_dict()})
        self.assertFalse(policy["require_paired_evidence"])
        self.assertTrue(policy["require_paired_strict_chain"])
        self.assertTrue(policy["cell_regression_blocks"])
        self.assertEqual(policy["minimum_score_delta"], .005)
        budget = schedule.run_execution_budget(5, cells_per_origin=9, holdout_inference_replicas=2)
        self.assertEqual(budget["total_candidate_origins"], 4680)

    def test_small_cohorts_remain_value_blind_and_purged_between_batches(self):
        schedule = OptimizationSchedule.for_comparison_run()
        plans = [plan_run_adaptation_cohort(dataset_fixture(800, changed_labels=changed),
                                            schedule=schedule, seed=7) for changed in (False, True)]
        self.assertEqual(plans[0], plans[1])
        for left, right in zip(plans[0].batches, plans[0].batches[1:]):
            self.assertLess(left.cohort.origins[-1].maximum_target_timestamp,
                            right.cohort.origins[0].origin_timestamp)

    def test_more_epochs_cost_more_executions_without_using_more_data(self):
        data = dataset_fixture(800)
        schedule = OptimizationSchedule.for_comparison_run()
        first = estimate_epoch_capacity(data, schedule=schedule, planned_generations=1, seed=7)
        ten = estimate_epoch_capacity(data, schedule=schedule, planned_generations=10, seed=7)
        self.assertTrue(first.sufficient)
        self.assertTrue(ten.sufficient)
        self.assertEqual(first.required_unique_origins, ten.required_unique_origins)
        self.assertEqual(schedule.required_unique_origins(10), ten.required_unique_origins)
        self.assertEqual(schedule.planned_origin_occurrences(10), ten.to_dict()["planned_origin_occurrences"])
        self.assertEqual(ten.candidate_origin_executions_for_run, first.candidate_origin_executions_for_run * 10)
        self.assertEqual(ten.reused_origin_occurrences, first.required_unique_origins * 9)
        self.assertFalse(ten.to_dict()["independent_evaluation_included"])

    def test_fixed_sources_are_purged_within_epoch_and_repetition_is_explicit(self):
        data = dataset_fixture(800)
        schedule = OptimizationSchedule.for_comparison_run()
        adaptation = plan_run_adaptation_cohort(data, schedule=schedule, seed=7)
        plans = [plan_generation_selection_cohorts(data, schedule=schedule, generation=g,
                    adaptation=adaptation, seed=7) for g in (0, 9)]
        for name in ("screening", "holdout"):
            left, right = (getattr(p, name) for p in plans)
            self.assertEqual(left.origin_ids, right.origin_ids)
            self.assertEqual({o.reuse_index for o in right.origins}, {9})
        groups = [adaptation.origins, plans[0].screening.origins, plans[0].holdout.origins]
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                self.assertTrue(all(abs(a.origin_timestamp - b.origin_timestamp) >= 24
                                    for a in left for b in right))

    def test_single_epoch_still_rejects_insufficient_data(self):
        report = estimate_epoch_capacity(dataset_fixture(300), schedule=OptimizationSchedule.for_comparison_run(),
                                        planned_generations=10, seed=0)
        self.assertFalse(report.sufficient)


class CalendarSplitTests(unittest.TestCase):
    def test_missing_rows_do_not_shift_shared_calendar_boundaries(self):
        complete = _Episode(episode_id="data:full", timestamps=tuple(range(1000)))
        gaps = replace(complete, episode_id="data:gaps", timestamps=tuple(t for t in range(1000) if t % 5))
        late = replace(complete, episode_id="data:late", timestamps=tuple(range(400, 1000)))
        manifest = build_split_manifest("data", (complete, gaps, late))
        for episode in (complete, gaps, late):
            split = manifest.split_for(episode.episode_id)
            for partition, start, end in ((split.training_fit, 0, 300),
                                          (split.training_feedback, 301, 600),
                                          (split.development, 624, 800), (split.gate, 824, 1000)):
                self.assertEqual(episode.timestamps[partition.start:partition.end],
                                 tuple(t for t in episode.timestamps if start <= t < end))
        self.assertEqual(manifest.split_for(late.episode_id).training_fit.size, 0)

    def test_reference_team_does_not_change_training_calendar(self):
        train = _Episode(episode_id="data:train", timestamps=tuple(range(1000)))
        reference = replace(train, episode_id="data:reference", timestamps=tuple(range(2000)))
        alone = build_split_manifest("data", (train,)).split_for(train.episode_id)
        together = build_split_manifest("data", (train, reference)).split_for(train.episode_id)
        self.assertEqual(alone, together)
