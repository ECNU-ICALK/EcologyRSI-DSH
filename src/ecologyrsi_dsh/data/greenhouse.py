"""Dependency-free canonical loaders for the 2018/2019 greenhouse datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .definitions import definition_for_domain

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    display_name_zh: str
    role: str
    unit: str
    required: bool = False
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name_zh": self.display_name_zh,
            "role": self.role,
            "unit": self.unit,
            "required": self.required,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True, slots=True)
class CanonicalEpisode:
    dataset_id: str
    domain_id: str
    episode_id: str
    timestamps: tuple[int, ...]
    values: Mapping[str, tuple[float | None, ...]]
    features: Mapping[str, FeatureSpec]
    source_files: tuple[str, ...]
    content_sha256: str

    @property
    def row_count(self) -> int:
        return len(self.timestamps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ecologyrsi-dsh.canonical-episode/1",
            "dataset_id": self.dataset_id,
            "domain_id": self.domain_id,
            "episode_id": self.episode_id,
            "row_count": self.row_count,
            "timestamps": list(self.timestamps),
            "values": {name: list(items) for name, items in self.values.items()},
            "features": {name: item.to_dict() for name, item in self.features.items()},
            "source_files": list(self.source_files),
            "content_sha256": self.content_sha256,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "row_count": self.row_count,
            "timestamp_start": self.timestamps[0],
            "timestamp_end": self.timestamps[-1],
            "feature_names": sorted(self.values),
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class CanonicalSeries:
    dataset_id: str
    domain_id: str
    episodes: tuple[CanonicalEpisode, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ecologyrsi-dsh.canonical-series/1",
            "dataset_id": self.dataset_id,
            "domain_id": self.domain_id,
            "episodes": [item.to_dict() for item in self.episodes],
        }


def feature_specs(domain_id: str) -> dict[str, FeatureSpec]:
    return {
        item["name"]: FeatureSpec(**{**item, "aliases": tuple(item["aliases"])})
        for item in definition_for_domain(domain_id)["features"]
    }


class GreenhouseDatasetAdapter:
    """Map official challenge CSV layouts into hourly canonical episodes."""

    def __init__(self, dataset_id: str, domain_id: str, dataset_dir: str | Path) -> None:
        self.dataset_id = dataset_id
        self.domain_id = domain_id
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.definition = definition_for_domain(domain_id)
        self.features = feature_specs(domain_id)

    def profile(self) -> dict[str, Any]:
        return {
            "schema_version": "ecologyrsi-dsh.dataset-profile/1",
            "adapter_id": "greenhouse_timeseries",
            "domain_id": self.domain_id,
            "evaluation_mode": "offline_logged",
            "sampling": "hourly_mean",
            "required_features": [f.name for f in self.features.values() if f.required],
            "features": [item.to_dict() for item in self.features.values()],
            "observation_tables": self.definition["tables"],
            "scientific_limits_zh": [
                "历史记录只能支持离线预测、回放和支持域分析。",
                "不能把历史下一状态解释为候选控制动作造成的反事实结果。",
                "低频作物、品质和产量标签保持稀疏，不扩散到高频时间点。",
            ],
        }

    def load(self) -> CanonicalSeries:
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {self.dataset_dir}")
        primary = self.definition["tables"][0]["filename"]
        climate_paths = sorted(self.dataset_dir.glob(f"*/{primary}"))
        if not climate_paths:
            raise FileNotFoundError(f"no {primary} files were found")
        shared = {}
        episodes = []
        for climate_path in climate_paths:
            tables, sources = [], []
            for table in self.definition["tables"]:
                scope = self.dataset_dir if table.get("scope") == "dataset" else climate_path.parent
                path = scope / table["filename"]
                if path not in shared:
                    rows = self._read_features(
                        path, {name: self.features[name].aliases for name in table["features"]},
                        tuple(table["timestamp_aliases"]),
                        allow_suffix=table.get("allow_suffix", False),
                        week_year=table.get("week_year"),
                        source_interval=table["source_interval"],
                        availability_delay_hours=table["availability_delay_hours"],
                    )
                    if table.get("scope") == "dataset":
                        shared[path] = rows
                else:
                    rows = shared[path]
                tables.append(rows)
                sources.append(path)
            episodes.append(self._episode(climate_path.parent.name, tables, sources,
                            derive_tomato=self.definition["derive_tomato_resources"]))
        return CanonicalSeries(self.dataset_id, self.domain_id, tuple(episodes))

    def _read_features(
        self,
        path: Path,
        mapping: Mapping[str, tuple[str, ...]],
        timestamp_aliases: tuple[str, ...],
        *,
        allow_suffix: bool = False,
        week_year: int | None = None,
        source_interval: str = "5_minutes",
        availability_delay_hours: int = 0,
    ) -> dict[int, dict[str, float]]:
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                return {}
            normalized = {_normalize(name): name for name in reader.fieldnames if name}
            timestamp_columns = [
                normalized[_normalize(alias)]
                for alias in timestamp_aliases
                if _normalize(alias) in normalized
            ]
            if not timestamp_columns:
                raise ValueError(f"no timestamp column found in {path}")
            columns: dict[str, tuple[str, ...]] = {}
            for canonical, aliases in mapping.items():
                matches: list[str] = []
                for alias in aliases:
                    normalized_alias = _normalize(alias)
                    for normalized_name, original in normalized.items():
                        if normalized_name == normalized_alias or (
                            allow_suffix and normalized_name.startswith(normalized_alias + "_")
                        ):
                            matches.append(original)
                columns[canonical] = tuple(sorted(set(matches)))
            rows = list(reader)

        date_offsets: dict[str, int] = {}
        first_timestamp = _row_timestamp(rows[0], timestamp_columns) if rows else None
        if source_interval == "5_minutes" and _is_date_only(first_timestamp):
            counts = Counter(str(_row_timestamp(row, timestamp_columns) or "").strip() for row in rows)
            date_offsets = {value: max(0, 288 - count) for value, count in counts.items() if value}
        occurrences: dict[str, int] = defaultdict(int)
        accumulator: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            raw_timestamp = _row_timestamp(row, timestamp_columns)
            hour = _parse_week_hour(raw_timestamp, week_year) if week_year is not None else _parse_hour(raw_timestamp)
            if hour is None:
                continue
            if source_interval == "5_minutes" and _is_date_only(raw_timestamp):
                key = str(raw_timestamp).strip()
                hour += (date_offsets.get(key, 0) + occurrences[key]) // 12
                occurrences[key] += 1
            hour += availability_delay_hours
            for canonical, source_columns in columns.items():
                parsed = [_parse_float(row.get(column)) for column in source_columns]
                values = [value for value in parsed if value is not None]
                if not values:
                    continue
                value = _validated_value(canonical, fmean(values))
                if value is not None:
                    accumulator[hour][canonical].append(value)
        return {
            hour: {name: fmean(values) for name, values in feature_values.items() if values}
            for hour, feature_values in accumulator.items()
        }

    def _episode(
        self,
        team: str,
        tables: Sequence[dict[int, dict[str, float]]],
        source_files: Iterable[Path],
        *,
        derive_tomato: bool = False,
    ) -> CanonicalEpisode:
        if not tables or not tables[0]:
            raise ValueError(f"primary climate series is empty for {team}")
        timestamps = tuple(sorted(tables[0]))
        names: set[str] = set()
        for table in tables:
            for row in table.values():
                names.update(row)
        mutable = {name: [None] * len(timestamps) for name in sorted(names)}
        for index, hour in enumerate(timestamps):
            for table in tables:
                for name, value in table.get(hour, {}).items():
                    mutable[name][index] = value
        if derive_tomato:
            _derive_tomato_resource_totals(mutable)
        values = {
            name: tuple(items)
            for name, items in mutable.items()
            if any(item is not None for item in items)
        }
        missing = [f.name for f in self.features.values() if f.required and f.name not in values]
        if missing:
            raise ValueError(f"episode {team} is missing required features: {', '.join(missing)}")
        identity = {
            "schema_version": "ecologyrsi-dsh.canonical-episode/1",
            "dataset_id": self.dataset_id,
            "domain_id": self.domain_id,
            "episode_id": f"{self.dataset_id}:{team}",
            "timestamps": list(timestamps),
            "values": {name: list(items) for name, items in values.items()},
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        present_sources = tuple(str(path.resolve()) for path in source_files if path.is_file())
        return CanonicalEpisode(
            dataset_id=self.dataset_id,
            domain_id=self.domain_id,
            episode_id=identity["episode_id"],
            timestamps=timestamps,
            values=values,
            features={name: self.features[name] for name in values},
            source_files=present_sources,
            content_sha256=digest,
        )


def _derive_tomato_resource_totals(values: dict[str, list[float | None]]) -> None:
    peak = values.get("electricity_peak_use")
    offpeak = values.get("electricity_offpeak_use")
    if peak is not None or offpeak is not None:
        length = len(peak or offpeak or ())
        total: list[float | None] = []
        for index in range(length):
            parts = [items[index] if items is not None else None for items in (peak, offpeak)]
            total.append(sum(parts) if all(part is not None for part in parts) else None)
        values["electricity_use"] = total
    irrigation = values.get("irrigation_water")
    drain = values.get("drain_water")
    if irrigation is not None:
        net: list[float | None] = []
        for index, supplied in enumerate(irrigation):
            if supplied is None or drain is None or drain[index] is None:
                net.append(None)
                continue
            discharged = drain[index]
            net.append(supplied - discharged)
        values["water_use"] = net


def _row_timestamp(row: Mapping[str, str | None], columns: Sequence[str]) -> str | None:
    return next((row.get(name) for name in columns if row.get(name)), None)


def _normalize(value: str) -> str:
    return _NORMALIZE_RE.sub("_", value.strip().lower().lstrip("%")).strip("_")


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.casefold() in {"nan", "na", "none", "null"}:
        return None
    try:
        number = float(stripped)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _parse_hour(value: str | None) -> int | None:
    number = _parse_float(value)
    if number is not None:
        return math.floor(number * 24 + 1e-7)
    if value is None:
        return None
    for pattern in ("%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            timestamp = datetime.strptime(value.strip(), pattern)
        except ValueError:
            continue
        return math.floor((timestamp - datetime(1899, 12, 30)).total_seconds() / 3600)
    return None


def _parse_week_hour(value: str | None, year: int) -> int | None:
    week = _parse_float(value)
    if week is None:
        return None
    try:
        timestamp = datetime.fromisocalendar(year, int(week), 1)
    except ValueError:
        return None
    return math.floor((timestamp - datetime(1899, 12, 30)).total_seconds() / 3600)


def _is_date_only(value: str | None) -> bool:
    if value is None or _parse_float(value) is not None or ":" in value:
        return False
    for pattern in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            datetime.strptime(value.strip(), pattern)
            return True
        except ValueError:
            pass
    return False


def _validated_value(name: str, value: float) -> float | None:
    limits: dict[str, tuple[float | None, float | None]] = {
        "relative_humidity": (0.0, 100.0),
        "outside_relative_humidity": (0.0, 100.0),
        "co2_concentration": (0.0, 5000.0),
        "root_zone_water_content": (0.0, 100.0),
        "drain_ph": (2.0, 14.0),
        "irrigation_ph": (2.0, 14.0),
        "drain_ec": (0.0, 30.0),
        "irrigation_ec": (0.0, 30.0),
        "root_zone_ec": (0.0, 30.0),
    }
    minimum, maximum = limits.get(name, (None, None))
    if minimum is not None and value < minimum:
        return None
    if maximum is not None and value > maximum:
        return None
    return value


__all__ = [
    "CanonicalEpisode",
    "CanonicalSeries",
    "FeatureSpec",
    "GreenhouseDatasetAdapter",
    "feature_specs",
]
