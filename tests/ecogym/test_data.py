from __future__ import annotations

import tempfile
import textwrap
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ecogym.data.loaders import load_csv_tables
from ecogym.data.spec import load_dataset_spec

from .helpers import synthetic_frame

SPEC = """
id = "tiny"
title = "tiny"
path = "tiny"
loader = "csv_tables"
step = "1h"
episodes = "*/climate.csv"
external_episodes = ["ref"]

[[columns]]
name = "temp"
role = "environment"
unit = "degC"
aliases = ["T"]
required = true

[[columns]]
name = "rh"
role = "environment"
unit = "percent"
aliases = ["RH"]
valid = [0.0, 100.0]

[[columns]]
name = "setpoint"
role = "action"
unit = "degC"
aliases = ["SP"]

[[columns]]
name = "outside"
role = "outside_weather"
unit = "degC"
aliases = ["Tout"]

[[columns]]
name = "water"
role = "resource"
unit = "L"
aliases = ["W"]

[[tables]]
file = "climate.csv"
time = ["time"]
columns = ["temp", "rh"]

[[tables]]
file = "vip.csv"
time = ["time"]
allow_suffix = true
columns = ["setpoint"]

[[tables]]
file = "meteo.csv"
scope = "dataset"
time = ["time"]
columns = ["outside"]

[[tables]]
file = "daily.csv"
time = ["time"]
interval_minutes = 1440
delay = 24
columns = ["water"]
"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.root = root / "tiny"
        write(root / "tiny.toml", SPEC)
        self.spec = load_dataset_spec(root / "tiny.toml")
        day = 43000  # serial day; hour 0 of that day is step 43000 * 24
        # Episode A: 5-minute rows in hours 0, 1 and 23. Hour 0 averages 10 and 12, hour 1 is 20;
        # one RH reading of 150 is invalid and must be dropped before averaging.
        rows = ["time,T,RH"]
        for minute in [*range(0, 120, 5), *range(1380, 1440, 5)]:
            temp = 10.0 if minute < 30 else 12.0 if minute < 60 else 20.0
            rh = "150" if minute == 65 else "50"
            rows.append(f"{day + minute / 1440:.10f},{temp},{rh}")
        write(self.root / "A" / "climate.csv", "\n".join(rows))
        write(self.root / "A" / "vip.csv", f"time,SP_1,SP_2\n{day},18,20\n{day + 10 / 1440:.10f},19,21\n")
        write(self.root / "A" / "daily.csv", f"time,W\n{day},3.5\n")
        write(self.root / "ref" / "climate.csv", f"time,T,RH\n{day},1,2\n")
        # Date-only stamps on a partial day of 12 five-minute rows: aligned to the day's last hour.
        stamp = (datetime(1899, 12, 30) + timedelta(days=day)).strftime("%m/%d/%Y")
        write(self.root / "meteo.csv", "\n".join(["time,Tout"] + [f"{stamp},7"] * 12))
        self.hour0 = day * 24

    def tearDown(self):
        self.directory.cleanup()

    def test_hourly_means_suffix_columns_and_validity(self):
        frame = load_csv_tables(self.spec, self.root)
        self.assertEqual(frame.episode_ids, ("A", "ref"))
        rows = frame.episode_rows("A")
        self.assertEqual(frame.time[rows].tolist(), list(range(self.hour0, self.hour0 + 24)))
        self.assertEqual(frame.present[rows].tolist(), [True, True] + [False] * 21 + [True])
        temp = frame.columns["temp"][rows]
        self.assertAlmostEqual(temp[0], 11.0)
        self.assertAlmostEqual(temp[1], 20.0)
        self.assertTrue(np.isnan(temp[2]))
        rh = frame.columns["rh"][rows]
        self.assertEqual(rh[[0, 1, 23]].tolist(), [50.0, 50.0, 50.0])
        setpoint = frame.columns["setpoint"][rows]
        self.assertAlmostEqual(setpoint[0], 19.5)  # mean over SP_1, SP_2 and both rows of hour 0
        self.assertTrue(np.isnan(setpoint[1]))

    def test_availability_delay_and_date_only_rows(self):
        frame = load_csv_tables(self.spec, self.root)
        rows = frame.episode_rows("A")
        # The daily value stamped at hour 0 is usable from hour 24, after this episode ends.
        self.assertNotIn("water", frame.columns)
        outside = frame.columns["outside"][rows]
        self.assertEqual(outside[23], 7.0)
        self.assertTrue(np.all(np.isnan(outside[:23])))
        self.assertTrue(self.spec.is_external("ref"))

    def test_spec_rejects_unknown_columns(self):
        broken = SPEC.replace('columns = ["water"]', 'columns = ["nope"]')
        path = Path(self.directory.name) / "broken.toml"
        write(path, broken)
        with self.assertRaises(ValueError):
            load_dataset_spec(path)


class FrameTests(unittest.TestCase):
    def test_gaps_become_absent_rows(self):
        frame = synthetic_frame(days=3, gap_at=10)
        rows = frame.episode_rows("A")
        self.assertEqual(rows.stop - rows.start, 72)
        present = frame.present[rows]
        self.assertEqual(int((~present).sum()), 3)
        self.assertTrue(np.all(np.isnan(frame.columns["y"][rows][~present])))
        self.assertTrue(np.all(np.diff(frame.time[rows]) == 1))

    def test_digest_tracks_content(self):
        self.assertEqual(synthetic_frame(days=3).digest, synthetic_frame(days=3).digest)
        self.assertNotEqual(synthetic_frame(days=3).digest, synthetic_frame(days=3, seed=1).digest)

    def test_select_keeps_named_episodes(self):
        frame = synthetic_frame(days=3)
        subset = frame.select(["B"])
        self.assertEqual(subset.episode_ids, ("B",))
        np.testing.assert_array_equal(subset.columns["y"], frame.columns["y"][frame.episode_rows("B")])


if __name__ == "__main__":
    unittest.main()
