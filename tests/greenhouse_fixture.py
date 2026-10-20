"""Small observed-format fixture exercising the production dataset adapter."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from ecologyrsi_dsh.data.registry import DatasetRegistry

DATASET_ID = "agc_cucumber_2018"
EPISODE_ID = f"{DATASET_ID}:TeamA"


def greenhouse_registry(root: Path, *, hours: int = 2048) -> DatasetRegistry:
    registry = DatasetRegistry(data_root=root)
    descriptor = registry._descriptor(DATASET_ID)
    folder = root / DATASET_ID
    for pattern in descriptor.required_globs:
        target = folder / pattern.replace("*/", "TeamA/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    climate = folder / "TeamA" / "Greenhouse_climate.csv"
    with climate.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["GHtime", "Tair", "RHair", "CO2air"])
        writer.writeheader()
        for hour in range(hours):
            phase = 2 * math.pi * hour / 24
            writer.writerow({"GHtime": hour / 24, "Tair": 21 + 3 * math.sin(phase),
                             "RHair": 70 - 5 * math.sin(phase), "CO2air": 700 + 30 * math.cos(phase)})
    return registry
