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


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_REQUIRED_FEATURES = ("air_temperature", "relative_humidity", "co2_concentration")


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


def _feature(
    name: str,
    display_name_zh: str,
    role: str,
    unit: str,
    *aliases: str,
    required: bool = False,
) -> FeatureSpec:
    return FeatureSpec(name, display_name_zh, role, unit, required, tuple(aliases))


_COMMON_FEATURES = (
    _feature("air_temperature", "室内气温", "environment", "degC", "Tair", required=True),
    _feature("relative_humidity", "室内相对湿度", "environment", "percent", "RHair", "Rhair", required=True),
    _feature("co2_concentration", "室内 CO2 浓度", "environment", "ppm", "CO2air", required=True),
    _feature("humidity_deficit", "湿度亏缺", "environment", "g_m3", "HumDef"),
    _feature("supplemental_light", "补光比例", "environment", "percent", "AssimLight"),
    _feature("heating_pipe_temperature", "加热管温度", "environment", "degC", "PipeLow"),
    _feature("ventilation_leeward", "背风侧通风开度", "environment", "percent", "VentLee"),
    _feature("ventilation_windward", "迎风侧通风开度", "environment", "percent", "Ventwind"),
    _feature("outside_temperature", "室外气温", "outside_weather", "degC", "Tout"),
    _feature("outside_relative_humidity", "室外相对湿度", "outside_weather", "percent", "Rhout"),
    _feature("outside_radiation", "室外辐射", "outside_weather", "W_m2", "Iglob"),
    _feature("outside_wind_speed", "室外风速", "outside_weather", "m_s", "Windsp"),
)

_CUCUMBER_FEATURES = (
    *_COMMON_FEATURES,
    _feature("irrigation_water", "灌溉水量", "resource", "L_m2_day", "Water", "water"),
    _feature("drain_water", "排水量", "root_zone", "L_m2_day", "drain"),
    _feature("drain_ph", "排液 pH", "root_zone", "pH", "pH_Drain"),
    _feature("drain_ec", "排液电导率", "root_zone", "dS_m", "EC_Drain"),
    _feature("marketable_yield", "一级果累计产量", "outcome", "kg_m2_cumulative", "ProdA_cum"),
    _feature("class_b_yield", "二级果累计产量", "outcome", "kg_m2_cumulative", "ProdB_cum"),
    _feature("total_yield", "累计总产量", "outcome", "kg_m2_cumulative", "Total_Prod_cum"),
    _feature("production_value", "累计产值", "outcome", "EUR_cumulative", "Prod_value_cum"),
    _feature("water_use", "净用水量", "outcome", "L_m2_day", "net_water"),
    _feature("co2_use", "CO2 用量", "outcome", "kg_m2_day", "CO2_dosage"),
    _feature("heating_energy_use", "供热能耗", "outcome", "kWh_m2_day", "Heating_Energy", "Heating_Eenrgy"),
    _feature("electricity_use", "照明用电", "outcome", "kWh_m2_day", "Electricity_Lamp", "Electricity_Lamps"),
    _feature("labour_use", "人工投入", "outcome", "hours_day", "Labour"),
    _feature("fruit_development_time", "果实发育时间", "crop", "days", "FruitGrw"),
    _feature("leaf_formation_rate", "叶片形成速率", "crop", "leaves_stem_week", "LeafFormRate"),
    _feature("cumulative_leaves", "累计叶片数", "crop", "leaves_stem", "N_leaves"),
    _feature("pruning_fraction", "修剪比例", "crop", "percent", "Pruning"),
    _feature("stem_elongation", "茎伸长量", "crop", "cm_week", "Stem_elong"),
    _feature("co2_setpoint", "CO2 设定值", "action", "ppm", "CO2_Vip"),
    _feature("heating_setpoint", "加热设定值", "action", "degC", "HeatTemp_Vip"),
    _feature("humidity_deficit_setpoint", "湿差设定值", "action", "g_m3", "HumDef_Vip"),
    _feature("ventilation_setpoint", "通风设定值", "action", "degC", "VentLeew_Vip", "VentWind_Vip"),
    _feature("irrigation_interval", "灌溉间隔", "action", "minutes", "WaterSupInt_Vip"),
)

_TOMATO_FEATURES = (
    *_COMMON_FEATURES,
    _feature("root_zone_water_content", "根区含水量", "root_zone", "percent", "WC_slab1", "WC_slab2"),
    _feature("root_zone_ec", "根区电导率", "root_zone", "mS_cm", "EC_slab1", "EC_slab2"),
    _feature("cumulative_irrigation", "累计灌溉量", "resource", "L_m2", "Cum_irr"),
    _feature("marketable_yield", "一级果产量", "outcome", "kg_m2_harvest", "ProdA"),
    _feature("class_b_yield", "二级果产量", "outcome", "kg_m2_harvest", "ProdB"),
    _feature("harvested_trusses", "已采收果穗数", "crop", "trusses_stem", "avg_nr_harvested_trusses"),
    _feature("truss_development_time", "果穗发育时间", "crop", "days", "Truss development time"),
    _feature("stem_elongation", "茎伸长量", "crop", "cm_week", "Stem_elong"),
    _feature("stem_thickness", "茎粗", "crop", "mm", "Stem_thick"),
    _feature("cumulative_trusses", "累计果穗数", "crop", "trusses_stem", "Cum_trusses"),
    _feature("stem_density", "茎密度", "crop", "stems_m2", "stem_dens"),
    _feature("plant_density", "种植密度", "crop", "plants_m2", "plant_dens"),
    _feature("fruit_flavour", "果实风味评分", "outcome", "score_0_100", "Flavour"),
    _feature("fruit_tss", "可溶性固形物", "outcome", "degBrix", "TSS"),
    _feature("fruit_acid", "果实酸度", "outcome", "mmol_H3O_100g", "Acid"),
    _feature("fruit_juice", "果汁比例", "outcome", "percent", "%Juice"),
    _feature("fruit_bite", "果实咬合力", "outcome", "N", "Bite"),
    _feature("fruit_weight", "单果重", "outcome", "g", "Weight"),
    _feature("fruit_dry_matter", "果实干物质比例", "outcome", "percent", "DMC_fruit"),
    _feature("heating_energy_use", "供热能耗", "outcome", "MJ_m2_day", "Heat_cons"),
    _feature("electricity_peak_use", "峰时用电", "resource", "kWh_m2_day", "ElecHigh"),
    _feature("electricity_offpeak_use", "谷时用电", "resource", "kWh_m2_day", "ElecLow"),
    _feature("electricity_use", "总用电", "outcome", "kWh_m2_day"),
    _feature("co2_use", "CO2 用量", "outcome", "kg_m2_day", "CO2_cons"),
    _feature("irrigation_water", "灌溉水量", "resource", "L_m2_day", "Irr"),
    _feature("drain_water", "排水量", "root_zone", "L_m2_day", "Drain"),
    _feature("water_use", "净用水量", "outcome", "L_m2_day"),
    _feature("irrigation_ph", "灌溉液 pH", "root_zone", "pH", "irr_PH"),
    _feature("irrigation_ec", "灌溉液电导率", "root_zone", "dS_m", "irr_EC"),
    _feature("irrigation_nitrate", "灌溉液硝酸盐", "root_zone", "mmol_L", "irr_NO3"),
    _feature("irrigation_potassium", "灌溉液钾", "root_zone", "mmol_L", "irr_K"),
    _feature("drain_ph", "排液 pH", "root_zone", "pH", "drain_PH"),
    _feature("drain_ec", "排液电导率", "root_zone", "dS_m", "drain_EC"),
    _feature("drain_nitrate", "排液硝酸盐", "root_zone", "mmol_L", "drain_NO3"),
    _feature("drain_potassium", "排液钾", "root_zone", "mmol_L", "drain_K"),
    _feature("co2_setpoint", "CO2 设定值", "action", "ppm", "co2_sp", "co2_vip"),
    _feature("heating_setpoint", "加热设定值", "action", "degC", "t_heat_sp", "t_heat_vip"),
    _feature("ventilation_temperature_setpoint", "通风温度设定值", "action", "degC", "t_vent_sp"),
    _feature("humidity_deficit_setpoint", "湿差设定值", "action", "g_m3", "dx_sp", "dx_vip"),
    _feature("lighting_setpoint", "照明设定值", "action", "percent", "assim_sp", "assim_vip"),
    _feature("irrigation_interval", "灌溉间隔", "action", "minutes", "water_sup_intervals_sp_min", "water_sup_intervals_vip_min"),
)


def feature_specs(domain_id: str) -> dict[str, FeatureSpec]:
    if domain_id == "greenhouse_cucumber_2018":
        values = _CUCUMBER_FEATURES
    elif domain_id == "greenhouse_tomato_2019":
        values = _TOMATO_FEATURES
    else:
        raise ValueError(f"unsupported greenhouse domain: {domain_id}")
    return {item.name: item for item in values}


class GreenhouseDatasetAdapter:
    """Map official challenge CSV layouts into hourly canonical episodes."""

    def __init__(self, dataset_id: str, domain_id: str, dataset_dir: str | Path) -> None:
        self.dataset_id = dataset_id
        self.domain_id = domain_id
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.features = feature_specs(domain_id)

    def profile(self) -> dict[str, Any]:
        return {
            "schema_version": "ecologyrsi-dsh.dataset-profile/1",
            "adapter_id": "greenhouse_timeseries",
            "domain_id": self.domain_id,
            "evaluation_mode": "offline_logged",
            "sampling": "hourly_mean",
            "required_features": list(_REQUIRED_FEATURES),
            "features": [item.to_dict() for item in self.features.values()],
            "scientific_limits_zh": [
                "历史记录只能支持离线预测、回放和支持域分析。",
                "不能把历史下一状态解释为候选控制动作造成的反事实结果。",
                "低频作物、品质和产量标签保持稀疏，不扩散到高频时间点。",
            ],
        }

    def load(self) -> CanonicalSeries:
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {self.dataset_dir}")
        if self.domain_id == "greenhouse_cucumber_2018":
            episodes = self._load_cucumber()
        elif self.domain_id == "greenhouse_tomato_2019":
            episodes = self._load_tomato()
        else:  # pragma: no cover - guarded by feature_specs
            raise ValueError(f"unsupported greenhouse domain: {self.domain_id}")
        return CanonicalSeries(self.dataset_id, self.domain_id, episodes)

    def _load_cucumber(self) -> tuple[CanonicalEpisode, ...]:
        climate_paths = sorted(self.dataset_dir.glob("*/Greenhouse_climate.csv"))
        if not climate_paths:
            raise FileNotFoundError("no cucumber Greenhouse_climate.csv files were found")
        weather_path = self.dataset_dir / "meteo.csv"
        weather = self._read_features(
            weather_path,
            self._mapping_for_roles("outside_weather"),
            ("time",),
        )
        episodes = []
        for climate_path in climate_paths:
            team_dir = climate_path.parent
            tables = [
                self._read_features(climate_path, self._mapping_for_roles("environment"), ("GHtime", "time")),
                self._read_features(team_dir / "vip.csv", self._mapping_for_roles("action"), ("time",), allow_suffix=True),
                weather,
                self._read_features(team_dir / "Irrigation.csv", self._mapping_for_roles("root_zone", "resource"), ("time",)),
                self._read_features(team_dir / "Production.csv", self._mapping_for_roles("outcome"), ("time",)),
                self._read_features(team_dir / "ResourceCalculations.csv", self._mapping_for_roles("outcome"), ("time",)),
                self._read_features(team_dir / "CropManagement.csv", self._mapping_for_roles("crop"), ("weeks",), week_year=2018),
            ]
            source_files = (
                climate_path,
                team_dir / "vip.csv",
                weather_path,
                team_dir / "Irrigation.csv",
                team_dir / "Production.csv",
                team_dir / "ResourceCalculations.csv",
                team_dir / "CropManagement.csv",
            )
            episodes.append(self._episode(climate_path.parent.name, tables, source_files))
        return tuple(episodes)

    def _load_tomato(self) -> tuple[CanonicalEpisode, ...]:
        climate_paths = sorted(self.dataset_dir.glob("*/GreenhouseClimate.csv"))
        if not climate_paths:
            raise FileNotFoundError("no tomato GreenhouseClimate.csv files were found")
        weather_path = self.dataset_dir / "Weather" / "Weather.csv"
        weather = self._read_features(
            weather_path,
            self._mapping_for_roles("outside_weather"),
            ("%time", "time"),
        )
        episodes = []
        for climate_path in climate_paths:
            team_dir = climate_path.parent
            tables = [
                self._read_features(
                    climate_path,
                    self._mapping_for_roles("environment", "resource", "action"),
                    ("%time", "time"),
                ),
                weather,
                self._read_features(team_dir / "GrodanSens.csv", self._mapping_for_roles("root_zone"), ("%time", "time")),
                self._read_features(team_dir / "CropParameters.csv", self._mapping_for_roles("crop"), ("%time", "time")),
                self._read_features(team_dir / "Production.csv", self._mapping_for_roles("outcome", "crop"), ("%time", "time")),
                self._read_features(team_dir / "TomQuality.csv", self._mapping_for_roles("outcome"), ("%time", "time")),
                self._read_features(team_dir / "Resources.csv", self._mapping_for_roles("outcome", "resource", "root_zone"), ("%time", "time")),
                self._read_features(team_dir / "LabAnalysis.csv", self._mapping_for_roles("root_zone"), ("%time", "time")),
            ]
            source_files = (
                climate_path,
                weather_path,
                team_dir / "GrodanSens.csv",
                team_dir / "CropParameters.csv",
                team_dir / "Production.csv",
                team_dir / "TomQuality.csv",
                team_dir / "Resources.csv",
                team_dir / "LabAnalysis.csv",
            )
            episodes.append(self._episode(climate_path.parent.name, tables, source_files, derive_tomato=True))
        return tuple(episodes)

    def _mapping_for_roles(self, *roles: str) -> dict[str, tuple[str, ...]]:
        selected = set(roles)
        return {
            item.name: item.aliases
            for item in self.features.values()
            if item.role in selected and item.aliases
        }

    def _read_features(
        self,
        path: Path,
        mapping: Mapping[str, tuple[str, ...]],
        timestamp_aliases: tuple[str, ...],
        *,
        allow_suffix: bool = False,
        week_year: int | None = None,
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
        if _is_date_only(first_timestamp):
            counts = Counter(str(_row_timestamp(row, timestamp_columns) or "").strip() for row in rows)
            date_offsets = {value: max(0, 288 - count) for value, count in counts.items() if value}
        occurrences: dict[str, int] = defaultdict(int)
        accumulator: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            raw_timestamp = _row_timestamp(row, timestamp_columns)
            hour = _parse_week_hour(raw_timestamp, week_year) if week_year is not None else _parse_hour(raw_timestamp)
            if hour is None:
                continue
            if _is_date_only(raw_timestamp):
                key = str(raw_timestamp).strip()
                hour += (date_offsets.get(key, 0) + occurrences[key]) // 12
                occurrences[key] += 1
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
        missing = [name for name in _REQUIRED_FEATURES if name not in values]
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
            parts = [items[index] for items in (peak, offpeak) if items is not None and items[index] is not None]
            total.append(sum(parts) if parts else None)
        values["electricity_use"] = total
    irrigation = values.get("irrigation_water")
    drain = values.get("drain_water")
    if irrigation is not None:
        net: list[float | None] = []
        for index, supplied in enumerate(irrigation):
            if supplied is None:
                net.append(None)
                continue
            discharged = drain[index] if drain is not None and drain[index] is not None else 0.0
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
