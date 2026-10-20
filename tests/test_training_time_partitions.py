from dataclasses import replace
import unittest

from ecologyrsi_dsh.data.splits import build_split_manifest
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evaluators.epoch_cohorts import (
    estimate_epoch_capacity, plan_run_adaptation_cohort, plan_generation_selection_cohorts,
)
from tests.test_epoch_cohort_planning import dataset_fixture
from tests.test_formal_stage_protocol import _Episode


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
