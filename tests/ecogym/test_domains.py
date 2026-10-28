"""Domain conformance: every registered task card works end to end on its data.

For each task whose data is present: the card matches its dataset spec, the
protocol yields eligible origins in every development stage, every preset
normalizes, fits on calibration_fit and forecasts model_selection with
finite values, and the naive-baseline method scores exactly zero.
"""

from __future__ import annotations

import unittest

import numpy as np

from ecogym.data.catalog import dataset_dir, dataset_spec
from ecogym.domains import load_domains
from ecogym.evaluation import Bench
from ecogym.protocol import DEVELOPMENT_STAGES
from ecogym.task.spec import check_task, task_specs

load_domains()
NAIVE = {"solver": "recipe_ridge@1", "params": {"terms": [{"op": "target_lag", "k": 0}], "scale": 0.0}}


class DomainConformance(unittest.TestCase):
    def test_every_task_card_matches_its_dataset(self):
        for task in task_specs():
            with self.subTest(task=task.id):
                check_task(task, dataset_spec(task.dataset))
                self.assertTrue(task.presets, "a task card should declare at least one seed method")

    def test_every_task_runs_on_its_data(self):
        for task in task_specs():
            if not dataset_dir(dataset_spec(task.dataset)).is_dir():
                continue
            with self.subTest(task=task.id):
                bench = Bench.open(task)
                for stage in DEVELOPMENT_STAGES:
                    self.assertGreater(bench.stage(stage).size, 0, stage)
                naive = bench.evaluate(bench.genome(NAIVE), "model_selection")
                self.assertEqual(naive.score(), 0.0)
                for name in task.presets:
                    evaluation = bench.evaluate(bench.preset(name), "model_selection")
                    self.assertTrue(np.all(np.isfinite(evaluation.predicted)), name)
                    self.assertGreater(evaluation.score(), -1.0, name)


if __name__ == "__main__":
    unittest.main()
