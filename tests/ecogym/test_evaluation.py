from __future__ import annotations

import unittest

import numpy as np

from ecogym.evaluation import compare, objective_interval
from ecogym.evaluation.bootstrap import iid_counts, legal_starts, moving_counts
from ecogym.evaluation.harness import Evaluation
from ecogym.evaluation.scoring import block_sums, cell_objectives, objective, skill
from ecogym.evaluation.suite import TaskPair, compare_suite
from ecogym.task.suite import SuiteGates

from .helpers import synthetic_bench
from .test_solvers import RECIPE

PERSISTENCE = {"solver": "recipe_ridge@1", "params": {"terms": [{"op": "target_lag", "k": 0}], "scale": 0.0}}


class ScoringTests(unittest.TestCase):
    def test_skill_is_bounded_and_symmetric(self):
        np.testing.assert_allclose(skill(np.array([1.0, 2.0, 0.0]), np.array([2.0, 1.0, 0.0])), [0.5, -0.5, 0.0])

    def test_missing_forecasts_are_penalized(self):
        method = np.array([[0.1, 0.1, np.nan, np.nan]])
        base = np.array([[1.0, 1.0, 1.0, 1.0]])
        sums = block_sums(method, base, np.array([0, 0, 1, 1]), np.array([1.0]))
        # skill 0.9 on the forecast half, -1 on the missing half
        self.assertAlmostEqual(float(cell_objectives(sums.total())[0, 0]), 0.5 * 0.9 - 0.5)

    def test_block_sums_are_additive(self):
        rng = np.random.default_rng(0)
        method, base = rng.normal(size=(2, 50)), rng.normal(size=(2, 50))
        blocks = np.repeat(np.arange(5), 10)
        total = block_sums(method, base, blocks, np.ones(2)).total()
        np.testing.assert_allclose(total.sse_method[:, 0], (method**2).sum(axis=1))
        whole = objective(total, np.array([0.5, 0.5]))[0]
        expected = np.mean(skill(np.sqrt((method**2).mean(axis=1)), np.sqrt((base**2).mean(axis=1))))
        self.assertAlmostEqual(float(whole), float(expected))

    def test_suite_separates_horizon_and_group_gains_and_can_guard_a_horizon(self):
        bench = synthetic_bench()
        stage = bench.stage("model_selection")
        genome = bench.genome(PERSISTENCE)
        incumbent = Evaluation(genome, stage, stage.baseline, np.zeros(len(stage.cells)))
        forecast = stage.observed.copy()
        forecast[1] = stage.observed[1] + 1.25 * (stage.baseline[1] - stage.observed[1])
        candidate = Evaluation(genome, stage, forecast, np.zeros(len(stage.cells)))
        pairs = [TaskPair(f"task-{i}", (candidate,), (incumbent,), group=f"season-{i // 4}") for i in range(8)]
        unrestricted = compare_suite(pairs, SuiteGates(test="signed_rank"), "seed")
        decision = compare_suite(pairs, SuiteGates(test="signed_rank", max_horizon_loss=((6, 0.1),)), "seed")
        self.assertTrue(unrestricted.adopted)
        self.assertFalse(decision.adopted)
        self.assertEqual(decision.reason, "horizon_regression:6")
        self.assertEqual(decision.interval_target, "hodges_lehmann_task_gain")
        self.assertEqual(decision.scope, "observed_task_suite")
        self.assertEqual(decision.groups["season-0"]["tasks"], 4)
        self.assertAlmostEqual(decision.horizons[1], 1.0)
        self.assertAlmostEqual(decision.horizons[6], -0.2)


class BootstrapTests(unittest.TestCase):
    def test_legal_starts_need_consecutive_blocks(self):
        self.assertEqual(legal_starts([1, 2, 3, 5, 6, 7, 8], 3), (0, 3, 4))
        self.assertIsNone(moving_counts([1, 3, 5], 3, 10, "seed"))

    def test_draws_are_reproducible_and_complete(self):
        first = iid_counts(10, 50, {"a": 1})
        np.testing.assert_array_equal(first, iid_counts(10, 50, {"a": 1}))
        self.assertTrue(np.all(first.sum(axis=1) == 10))
        moving = moving_counts(list(range(10)), 3, 50, "x")
        self.assertTrue(np.all(moving.sum(axis=1) == 10))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.bench = synthetic_bench()
        self.stage = self.bench.stage("model_selection")
        self.all = np.arange(self.stage.size)

    def test_identical_methods_are_not_promoted(self):
        evaluation = self.bench.evaluate(self.bench.genome(RECIPE), "model_selection")
        decision = compare(evaluation, evaluation, self.all, self.bench.task.gates, "seed")
        self.assertFalse(decision.promoted)
        self.assertEqual(decision.reason, "below_practical_delta")
        self.assertEqual(decision.delta, 0.0)

    def test_a_clearly_better_method_is_promoted(self):
        better = self.bench.evaluate(self.bench.genome(RECIPE), "model_selection")
        naive = self.bench.evaluate(self.bench.genome(PERSISTENCE), "model_selection")
        decision = compare(better, naive, self.all, self.bench.task.gates, "seed")
        self.assertTrue(decision.promoted, decision.reason)
        self.assertGreater(decision.interval[0], 0)
        self.assertTrue(all(verdict.passed for verdict in decision.cells))
        backwards = compare(naive, better, self.all, self.bench.task.gates, "seed")
        self.assertFalse(backwards.promoted)

    def test_objective_interval_brackets_the_score(self):
        evaluation = self.bench.evaluate(self.bench.genome(RECIPE), "model_selection")
        low, high = objective_interval(evaluation, None, self.bench.task.gates, "seed")
        self.assertLessEqual(low, evaluation.score())
        self.assertGreaterEqual(high, evaluation.score())


if __name__ == "__main__":
    unittest.main()
