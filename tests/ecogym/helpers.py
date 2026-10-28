"""Shared fixtures: a synthetic task on a synthetic frame, and the real AGC bench."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from ecogym.data.catalog import data_root, dataset_dir, dataset_spec
from ecogym.data.frame import Frame
from ecogym.data.spec import ColumnSpec
from ecogym.domains import load_domains
from ecogym.task.spec import GateSpec, ProtocolSpec, Target, TaskSpec

GOLDEN = Path(__file__).resolve().parent / "golden" / "agc_cucumber_2018.json"
TASK_ID = "greenhouse.agc2018.cucumber.climate@1"
# Tests never write to the user's cache.
os.environ.setdefault("ECOGYM_CACHE_DIR", str(Path(tempfile.gettempdir()) / "ecogym-test-cache"))


def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def agc_available() -> bool:
    load_domains()
    return dataset_dir(dataset_spec("agc_cucumber_2018")).is_dir()


requires_agc = unittest.skipUnless(
    agc_available(), f"AGC 2018 data not found under {data_root()} (set ECOGYM_DATA_ROOT)"
)


@lru_cache(maxsize=1)
def agc_bench():
    from ecogym.evaluation import Bench

    load_domains()
    return Bench.open(TASK_ID)


def synthetic_bench(task: TaskSpec | None = None, frame: Frame | None = None):
    from ecogym.evaluation import Bench

    import ecogym.solvers  # noqa: F401  (registers recipe_ridge@1)

    frame = frame if frame is not None else synthetic_frame()
    return Bench(task or synthetic_task(), frame, optimization=list(frame.episode_ids))


def synthetic_frame(days: int = 60, *, seed: int = 0, gap_at: int | None = None) -> Frame:
    """Two episodes of an hourly diurnal signal y with a driver x; optional missing hours."""

    rng = np.random.default_rng(seed)
    episodes = []
    for name, shift in (("A", 0.0), ("B", 1.5)):
        hours = np.arange(1_000_000, 1_000_000 + days * 24, dtype=np.int64)
        phase = 2 * np.pi * (hours % 24) / 24
        x = 10 + 5 * np.sin(phase) + rng.normal(0, 0.5, hours.size)
        drift = np.zeros(hours.size)  # AR(1) weather-like drift: persistence wins at short horizons
        shocks = rng.normal(0, 0.4, hours.size)
        for index in range(1, hours.size):
            drift[index] = 0.97 * drift[index - 1] + shocks[index]
        y = 20 + shift + 4 * np.sin(phase - 0.5) + 0.3 * x + drift + rng.normal(0, 0.3, hours.size)
        if gap_at is not None:
            keep = np.ones(hours.size, dtype=bool)
            keep[gap_at : gap_at + 3] = False
            hours, x, y = hours[keep], x[keep], y[keep]
        episodes.append((name, hours, {"y": y, "x": x}))
    specs = {
        "y": ColumnSpec("y", "environment", "u"),
        "x": ColumnSpec("x", "action", "u"),
    }
    return Frame.build("synthetic", "1h", episodes, specs)


def synthetic_task(**changes) -> TaskSpec:
    task = TaskSpec(
        id="synthetic@1",
        title="synthetic",
        dataset="synthetic",
        episodes=("A",),
        targets=(Target("y", 1.0),),
        horizons=(1, 6),
        history_steps=30,
        history_scope="partition",
        exogenous_roles=("action",),
        fill_age={"action": 6},
        min_coverage=0.2,
        block_steps=24,
        season_steps=24,
        periods=(24,),
        baselines=("persistence", "seasonal"),
        tie_tolerance=1e-12,
        protocol=ProtocolSpec(min_fit_origins=10, min_fit_blocks=3, min_uq_rows=10, min_uq_blocks=2),
        gates=GateSpec(resamples=200),
    )
    return replace(task, **changes)
