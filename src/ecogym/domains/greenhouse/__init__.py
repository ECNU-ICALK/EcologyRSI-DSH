"""Greenhouse climate domain pack: Autonomous Greenhouse Challenge data.

Ships the AGC 2018 cucumber dataset spec, its climate task card (with the
expert and recipe seed presets) and the diurnal-profile expert solver.
"""

from pathlib import Path

from . import expert  # noqa: F401  (registers greenhouse.diurnal_profile@1)

HERE = Path(__file__).resolve().parent
DATASETS = (HERE / "agc_cucumber_2018.toml",)
TASKS = (HERE / "tasks" / "cucumber_climate.toml",)
