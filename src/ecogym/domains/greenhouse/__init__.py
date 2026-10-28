"""Greenhouse climate domain pack: Autonomous Greenhouse Challenge data.

Ships the AGC 2018 cucumber and AGC 2019 tomato dataset specs, a climate
task card for each (with expert and recipe-seed presets), the domain
knowledge notes shown to research agents, and the diurnal-profile expert
solver.
"""

from pathlib import Path

from . import expert  # noqa: F401  (registers greenhouse.diurnal_profile@1)

HERE = Path(__file__).resolve().parent
DATASETS = (HERE / "agc_cucumber_2018.toml", HERE / "agc_tomato_2019.toml")
TASKS = (HERE / "tasks" / "cucumber_climate.toml", HERE / "tasks" / "tomato_climate.toml")
