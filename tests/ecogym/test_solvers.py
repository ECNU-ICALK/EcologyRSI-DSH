from __future__ import annotations

import unittest

import numpy as np

from ecogym.data.frame import Frame
from ecogym.solvers import Genome, SolverError
from ecogym.solvers.recipe import validate_terms

from .helpers import synthetic_bench, synthetic_frame, synthetic_task

RECIPE = {
    "solver": "recipe_ridge@1",
    "params": {
        "terms": [
            {"op": "target_lag", "k": 0},
            {"op": "target_lag", "k": 1},
            {"op": "seasonal_reference", "period": 24},
            {"op": "diurnal_sin_cos", "period": 24},
            {"op": "exogenous", "col": "x", "lag": 0},
            {"op": "exogenous_rolling_mean", "col": "x", "w": 3},
        ],
        "alpha": 0.1,
        "scale": 1.0,
    },
}


class RecipeGrammarTests(unittest.TestCase):
    def setUp(self):
        self.task = synthetic_task()

    def test_rejects_unknown_ops_future_reads_and_overreach(self):
        for terms in (
            [{"op": "magic"}],
            [{"op": "target_lag", "k": -1}],
            [{"op": "target_lag", "k": self.task.history_steps}],
            [{"op": "seasonal_reference", "period": 7}],
            [{"op": "target_lag", "k": 1}, {"op": "target_lag", "k": 1}],
            [{"op": "rolling_mean", "w": 3, "extra": 1}],
        ):
            with self.subTest(terms=terms), self.assertRaises(SolverError):
                validate_terms(terms, self.task)

    def test_seasonal_reference_refuses_horizons_beyond_the_period(self):
        task = synthetic_task(horizons=(1, 30), history_steps=40, periods=(24,))
        with self.assertRaises(SolverError):
            validate_terms([{"op": "seasonal_reference", "period": 24}], task)


class GenomeTests(unittest.TestCase):
    def test_normalization_broadcasts_cells_and_rejects_unknown(self):
        import ecogym.solvers  # noqa: F401

        task = synthetic_task()
        genome = Genome.from_dict(RECIPE, task)
        self.assertEqual(genome.params["scale"], {"y@1": 1.0, "y@6": 1.0})
        self.assertEqual(genome.params["anchor"], "baseline")
        params = dict(reversed(list(RECIPE["params"].items())))
        reordered = Genome.from_dict({"params": params, "solver": RECIPE["solver"]}, task)
        self.assertEqual(genome.digest, reordered.digest)
        with self.assertRaises(SolverError):
            Genome.from_dict({"solver": "recipe_ridge@1", "params": {"alpah": 0.1}}, task)
        with self.assertRaises(SolverError):
            Genome.from_dict({"solver": "recipe_ridge@1", "params": {"alpha": 5.0}}, task)


class RecipeSolverTests(unittest.TestCase):
    def test_forecasts_beat_persistence_on_a_driven_signal(self):
        bench = synthetic_bench()
        evaluation = bench.evaluate(bench.genome(RECIPE), "model_selection")
        self.assertTrue(np.all(np.isfinite(evaluation.predicted)))
        self.assertGreater(evaluation.score(), 0.1)

    def test_zero_scale_returns_the_anchor(self):
        bench = synthetic_bench()
        body = {"solver": "recipe_ridge@1", "params": {**RECIPE["params"], "scale": 0.0}}
        evaluation = bench.evaluate(bench.genome(body), "model_selection")
        np.testing.assert_array_equal(evaluation.predicted, evaluation.stage.baseline)
        self.assertEqual(evaluation.score(), 0.0)

    def test_forecasts_ignore_everything_after_the_origin(self):
        """Corrupting every value after one origin must leave its forecast unchanged."""

        frame = synthetic_frame()
        bench = synthetic_bench(frame=frame)
        genome = bench.genome(RECIPE)
        before = bench.evaluate(genome, "model_selection")
        position = before.stage.size // 2
        cutoff = int(before.stage.time[position])
        rng = np.random.default_rng(1)
        corrupted = {
            name: np.where((frame.time > cutoff) & (frame.episode == 0), rng.normal(500, 100, len(frame)), values)
            for name, values in frame.columns.items()
        }
        episodes = []
        for name in frame.episode_ids:
            rows = frame.episode_rows(name)
            keep = frame.present[rows]
            episodes.append((name, frame.time[rows][keep], {c: v[rows][keep] for c, v in corrupted.items()}))
        changed = Frame.build(frame.dataset_id, frame.step, episodes, frame.specs)
        after = synthetic_bench(frame=changed).evaluate(genome, "model_selection")
        index = int(np.flatnonzero(after.stage.time == cutoff)[0])
        np.testing.assert_array_equal(after.predicted[:, index], before.predicted[:, position])
        self.assertFalse(np.array_equal(after.stage.observed[:, index], before.stage.observed[:, position]))


if __name__ == "__main__":
    unittest.main()
