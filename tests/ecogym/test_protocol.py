from __future__ import annotations

import unittest

import numpy as np

from ecogym.protocol import (
    LeakageError,
    SealedAccess,
    eligible,
    forward_fill,
    four_stage,
    open_stage,
    plan_generation,
    select_baselines,
)
from ecogym.protocol.partitions import STAGES

from .helpers import synthetic_frame, synthetic_task


class ForwardFillTests(unittest.TestCase):
    def test_max_age_and_episode_boundary(self):
        values = np.array([1.0, np.nan, np.nan, np.nan, 5.0, np.nan, np.nan])
        time = np.array([0, 1, 2, 3, 4, 5, 6])
        episode = np.array([0, 0, 0, 0, 0, 1, 1])
        filled = forward_fill(values, time, episode, 2)
        np.testing.assert_array_equal(filled[:5], [1.0, 1.0, 1.0, np.nan, 5.0])
        self.assertTrue(np.all(np.isnan(filled[5:])))  # the next episode starts empty


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_frame()
        self.task = synthetic_task()
        self.partitions = four_stage(self.frame, self.task, ["A", "B"])

    def test_stages_are_ordered_disjoint_and_embargoed(self):
        intervals = [self.partitions.intervals[stage]["A"] for stage in STAGES]
        for (start, end), (next_start, _) in zip(intervals, intervals[1:]):
            self.assertLess(start, end)
            self.assertGreaterEqual(next_start - end, self.task.protocol.embargo - 1)
        masks = [self.partitions.mask(self.frame, stage) for stage in STAGES]
        self.assertEqual(int(sum(mask.astype(int) for mask in masks).max()), 1)

    def test_sealed_stages_need_access(self):
        with self.assertRaises(PermissionError):
            open_stage(self.frame, self.partitions, self.task, "validation")
        with self.assertRaises(PermissionError):
            open_stage(self.frame, self.partitions, self.task, "final_test", sealed=SealedAccess("validation", "k"))
        opened = open_stage(self.frame, self.partitions, self.task, "validation", sealed=SealedAccess("validation", "k"))
        self.assertGreater(opened.size, 0)


class OriginTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_frame()
        self.task = synthetic_task()
        self.partitions = four_stage(self.frame, self.task, ["A", "B"])
        self.selection = open_stage(self.frame, self.partitions, self.task, "model_selection")

    def test_forecast_origins_carry_no_labels_and_no_future(self):
        forecast = self.selection.without_labels()
        with self.assertRaises(LeakageError):
            forecast.label("y", 1)
        with self.assertRaises(LeakageError):
            forecast.past("y", -1)

    def test_partition_scope_hides_earlier_rows(self):
        first = self.selection.take(np.array([0]))
        self.assertTrue(np.isfinite(first.past("y", 0))[0])
        self.assertTrue(np.isnan(first.past("y", 1))[0])  # the row before lies in the embargo

    def test_reads_never_cross_episodes(self):
        frame = synthetic_frame()
        task = synthetic_task(episodes=("A", "B"), history_scope="causal")
        partitions = four_stage(frame, task, ["A", "B"])
        opened = open_stage(frame, partitions, task, "calibration_fit")
        start_b = frame.episode_rows("B").start
        first_b = np.flatnonzero(opened._rows == start_b)
        self.assertTrue(np.isnan(opened.take(first_b).past("y", 1))[0])

    def test_eligible_origins_have_full_history_and_labels(self):
        mask = eligible(self.selection, self.task)
        chosen = self.selection.take(mask)
        self.assertGreater(chosen.size, 100)
        for k in range(self.task.history_steps):
            self.assertTrue(np.all(np.isfinite(chosen.past("y", k))))
        for horizon in self.task.horizons:
            self.assertTrue(np.all(np.isfinite(chosen.label("y", horizon))))
        self.assertLess(chosen.size, self.selection.size)


class BaselineTests(unittest.TestCase):
    def test_seasonal_wins_mid_cycle(self):
        frame = synthetic_frame()
        task = synthetic_task(horizons=(1, 12))
        partitions = four_stage(frame, task, ["A", "B"])
        profile = select_baselines(open_stage(frame, partitions, task, "calibration_fit"), task)
        self.assertEqual(profile.kinds, {"y@1": "persistence", "y@12": "seasonal"})


class CohortTests(unittest.TestCase):
    def test_disjoint_days_and_deterministic(self):
        time = np.arange(1_000_000, 1_000_000 + 24 * 30)
        screening, holdout = plan_generation(time, block_steps=24, chunk_blocks=3, screen_fraction=1 / 3,
                                             seed=7, generation=0)
        self.assertFalse(set(screening.blocks) & set(holdout.blocks))
        self.assertEqual(screening.size + holdout.size, time.size)
        again, _ = plan_generation(time, block_steps=24, chunk_blocks=3, screen_fraction=1 / 3, seed=7, generation=0)
        np.testing.assert_array_equal(screening.index, again.index)
        other, _ = plan_generation(time, block_steps=24, chunk_blocks=3, screen_fraction=1 / 3, seed=7, generation=1)
        self.assertNotEqual(screening.blocks, other.blocks)


if __name__ == "__main__":
    unittest.main()
