"""Golden parity with v0.10.33 on AGC 2018 cucumber, team AiCU.

The fixtures in ``golden/`` were produced from the retired implementation by
``tests/ecogym/golden/make_golden.py``, which needs that package and is no
longer in the tree (see commit 6c51c8c). Data, partitions, baselines and
origin sets must match exactly; forecasts within 1e-9.
"""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np

from ecogym.protocol import eligible, open_stage

from .helpers import agc_bench, dataset_available, golden, requires_agc

TOLERANCE = 1e-9


def sha(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


@requires_agc
class DataParity(unittest.TestCase):
    def test_every_column_matches_bit_for_bit(self):
        from ecogym.data.catalog import dataset_spec, load_frame

        expected = golden()["dataset"]
        frame = load_frame(dataset_spec("agc_cucumber_2018"))
        for item in expected["episodes"]:
            name = item["episode_id"].split(":", 1)[1]
            times = frame.present_times(name)
            self.assertEqual((times.size, int(times[0]), int(times[-1])),
                             (item["row_count"], item["first_timestamp"], item["last_timestamp"]))
        rows = frame.episode_rows("AiCU")
        present = frame.present[rows]
        aicu = expected["aicu"]
        self.assertEqual(sha([int(t) for t in frame.time[rows][present]]), aicu["timestamps_digest"])
        self.assertEqual(set(frame.columns), set(aicu["columns"]))
        for name, column in aicu["columns"].items():
            values = frame.columns[name][rows][present]
            as_json = [None if math.isnan(value) else float(value) for value in values]
            self.assertEqual(sha(as_json), column["digest"], name)


TOMATO = Path(__file__).resolve().parent / "golden" / "agc_tomato_2019.json"


@unittest.skipUnless(dataset_available("agc_tomato_2019"), "AGC 2019 data not found (set ECOGYM_DATA_ROOT)")
class TomatoDataParity(unittest.TestCase):
    """The second greenhouse dataset loads through the same generic loader, bit for bit."""

    def test_every_column_matches_bit_for_bit(self):
        from ecogym.data.catalog import dataset_spec, load_frame

        expected = json.loads(TOMATO.read_text(encoding="utf-8"))
        frame = load_frame(dataset_spec("agc_tomato_2019"))
        for item in expected["episodes"]:
            times = frame.present_times(item["episode_id"].split(":", 1)[1])
            self.assertEqual((times.size, int(times[0]), int(times[-1])),
                             (item["row_count"], item["first_timestamp"], item["last_timestamp"]))
        episode = expected["episode"]
        rows = frame.episode_rows(episode["episode_id"].split(":", 1)[1])
        present = frame.present[rows]
        self.assertEqual(sha([int(t) for t in frame.time[rows][present]]), episode["timestamps_digest"])
        self.assertEqual(set(frame.columns), set(episode["columns"]))
        for name, column in episode["columns"].items():
            values = frame.columns[name][rows][present]
            self.assertEqual(sha([None if math.isnan(v) else float(v) for v in values]), column["digest"], name)


@requires_agc
class ProtocolParity(unittest.TestCase):
    def test_stage_boundaries(self):
        bench = agc_bench()
        for stage, expected in golden()["protocol"]["stages"].items():
            interval = (expected["time_start"], expected["time_end"])
            self.assertEqual(bench.partitions.intervals[stage]["AiCU"], interval, stage)
            self.assertEqual(bench.partitions.rows[stage]["AiCU"], expected["rows"], stage)

    def test_fit_selected_baselines(self):
        bench = agc_bench()
        for item in golden()["baselines"]:
            kind = {"seasonal_24h": "seasonal"}.get(item["kind"], item["kind"])
            evidence = bench.baselines.evidence[item["cell"]]
            self.assertEqual(bench.baselines.kinds[item["cell"]], kind, item["cell"])
            self.assertEqual(evidence["comparison_n"], item["comparison_n"])
            self.assertAlmostEqual(evidence["persistence_rmse"], item["persistence_rmse"], delta=1e-12)
            self.assertAlmostEqual(evidence["seasonal_rmse"], item["seasonal_rmse"], delta=1e-12)

    def test_eligible_origins(self):
        bench = agc_bench()
        for stage, expected in golden()["origins"].items():
            opened = open_stage(bench.frame, bench.partitions, bench.task, stage)
            times = [int(t) for t in opened.time[eligible(opened, bench.task)]]
            self.assertEqual(len(times), expected["count"], stage)
            self.assertEqual(sha(times), expected["digest"], stage)

    def test_normalization_scales(self):
        bench = agc_bench()
        for target, scale in golden()["results"]["expert"]["model_selection"]["normalization_scales"].items():
            self.assertAlmostEqual(bench.scales[target], scale, delta=1e-12)


@requires_agc
class SolverParity(unittest.TestCase):
    def check(self, preset: str, stage: str) -> None:
        bench = agc_bench()
        evaluation = bench.evaluate(bench.preset(preset), stage)
        expected = golden()["results"][preset][stage]
        self.assertEqual(evaluation.stage.size, expected["origins"])
        self.assertEqual(sha([int(t) for t in evaluation.stage.time]), expected["origins_digest"])
        self.assertAlmostEqual(evaluation.score(), expected["overall_skill"], delta=TOLERANCE)
        position_of = {int(t): index for index, t in enumerate(evaluation.stage.time)}
        for row, cell in zip(evaluation.cells(), evaluation.stage.cells, strict=True):
            golden_cell = expected["cells"][cell.key]
            self.assertEqual(row["n"], golden_cell["n"])
            self.assertAlmostEqual(row["rmse"], golden_cell["model_rmse"], delta=TOLERANCE, msg=cell.key)
            self.assertAlmostEqual(row["baseline_rmse"], golden_cell["baseline_rmse"], delta=TOLERANCE, msg=cell.key)
            predicted = evaluation.predicted[evaluation.stage.cells.index(cell)]
            for sample in golden_cell["samples"]:
                self.assertAlmostEqual(predicted[position_of[sample["origin"]]], sample["predicted"], delta=TOLERANCE)
        self.assertEqual(int(evaluation.clipped.sum()), 0)

    def test_expert_model_selection(self):
        self.check("expert", "model_selection")

    def test_expert_calibration_uq(self):
        self.check("expert", "calibration_uq")

    def test_recipe_seed_model_selection(self):
        self.check("recipe_seed", "model_selection")

    def test_recipe_seed_calibration_uq(self):
        self.check("recipe_seed", "calibration_uq")

    def test_expert_constrained_member(self):
        bench = agc_bench()
        fit = bench.evaluate(bench.preset("expert"), "model_selection").fit
        for model in golden()["results"]["expert"]["model_selection"]["models"]:
            mine = fit[model["cell"]]
            self.assertEqual(mine["rows"], model["training_rows"])
            self.assertAlmostEqual(mine["constrained"]["profile_weight"],
                                   model["constrained_member"]["profile_anomaly_weight"], delta=1e-12)
            self.assertAlmostEqual(mine["constrained"]["previous_day_weight"],
                                   model["constrained_member"]["previous_day_weight"], delta=1e-12)

    def test_zero_scale_returns_the_baseline(self):
        bench = agc_bench()
        genome = bench.preset("expert")
        params = dict(genome.params, scale={**genome.params["scale"], "air_temperature@24": 0.0})
        evaluation = bench.evaluate(bench.genome({"solver": genome.solver, "params": params}), "model_selection")
        position = [cell.key for cell in evaluation.stage.cells].index("air_temperature@24")
        np.testing.assert_array_equal(evaluation.predicted[position], evaluation.stage.baseline[position])


if __name__ == "__main__":
    unittest.main()
