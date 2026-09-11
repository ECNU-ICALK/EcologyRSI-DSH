"""Dataset catalog location and built-in toy dataset definitions."""

from __future__ import annotations

import os
from pathlib import Path

from ..core.errors import FrozenRuntimeBindingDriftError
from .definitions import default_catalog_path as _default_catalog_path
from .contracts import _TOY_DATASET_ID, DatasetDescriptor
from .greenhouse import CanonicalEpisode, CanonicalSeries, FeatureSpec
from .toy import ToyCropSoilWater


def _assert_snapshot_digest(
    label: str,
    actual: str,
    expected: str | None,
) -> None:
    if expected is None:
        return
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError(f"{label}的冻结校验值必须是非空字符串")
    normalized = expected.strip()
    if normalized != actual:
        raise FrozenRuntimeBindingDriftError(label)


def _default_data_root() -> Path:
    configured = os.environ.get("ECOLOGYRSI_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    # Downloads and every normal service start share one durable location.
    # Do not change roots based on which old checkout happens to exist.
    return (Path.home() / ".ecologyrsi-dsh" / "data" / "greenhouse").resolve()


def _toy_descriptor() -> DatasetDescriptor:
    return DatasetDescriptor(
        dataset_id=_TOY_DATASET_ID,
        display_name_zh="作物—土壤—水分合成演示序列",
        domain_id="crop_soil_water",
        adapter_id="toy_crop_soil_water",
        title="Generated toy crop-soil-water series",
        subject="engineering_fixture",
        edition="generated",
        version="1",
        publisher="EcologyRSI-DSH",
        doi="not-applicable",
        landing_page="local-generated-fixture",
        license="LicenseRef-EcologyRSI-Project",
        modalities=("time_series",),
        runnable=True,
        required_globs=(),
        source_files=(),
        notes_zh=("本地确定性生成，仅用于工程演示。",),
    )


def _toy_series() -> CanonicalSeries:
    fixture = ToyCropSoilWater(seed=0)
    specs = {
        "rainfall": FeatureSpec("rainfall", "降雨", "forcing", "normalized_water_depth_day", True),
        "evapotranspiration": FeatureSpec(
            "evapotranspiration", "蒸散量", "forcing", "normalized_water_depth_day", True
        ),
        "soil_water": FeatureSpec("soil_water", "土壤含水量", "state", "fraction", True),
    }
    timestamps = tuple(item.day * 24 for item in fixture.observations)
    values: dict[str, tuple[float | None, ...]] = {
        "rainfall": tuple(item.rainfall for item in fixture.observations),
        "evapotranspiration": tuple(item.evapotranspiration for item in fixture.observations),
        "soil_water": tuple(item.soil_water for item in fixture.observations),
    }
    identity = {
        "dataset_id": _TOY_DATASET_ID,
        "episode_id": "generated-toy-series@1:seed-0",
        "timestamps": list(timestamps),
        "values": {name: list(items) for name, items in values.items()},
    }
    episode = CanonicalEpisode(
        dataset_id=_TOY_DATASET_ID,
        domain_id="crop_soil_water",
        episode_id=identity["episode_id"],
        timestamps=timestamps,
        values=values,
        features=specs,
        source_files=(),
        # Use the exact fixture identity consumed by the evaluator.  The
        # browser sample page and every fitted artifact now bind to the same
        # immutable digest instead of parallel hashes of equivalent rows.
        content_sha256=fixture.dataset_digest,
    )
    return CanonicalSeries(_TOY_DATASET_ID, "crop_soil_water", (episode,))
