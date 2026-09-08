"""Dataset-owned prediction contracts, independent of Agent model selection.

An adapter binds source semantics to one supported evaluation task. Adding a
new crop task requires an explicit contract; merely finding a numeric column
does not make it a prediction label. Contracts contain no observed values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..core.models import digest
from ..core.prediction_policy import RUNTIME_EVALUATOR_ID
from .greenhouse import GreenhouseDatasetAdapter


@dataclass(frozen=True, slots=True)
class PredictionTarget:
    name: str
    label: str
    unit: str
    minimum: float
    maximum: float
    weight: float

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.name, self.label, self.unit)):
            raise ValueError("prediction target requires a name, label and unit")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (self.minimum, self.maximum, self.weight)):
            raise ValueError("prediction target bounds and weight must be finite")
        if self.minimum >= self.maximum or self.weight <= 0:
            raise ValueError("prediction target bounds or weight are invalid")


@dataclass(frozen=True, slots=True)
class DatasetAdapter:
    adapter_id: str
    dataset_id: str
    domain_id: str
    label: str
    climate_filename: str
    targets: tuple[PredictionTarget, ...]
    label_semantics: tuple[str, ...]
    horizons_hours: tuple[int, ...] = (1, 6, 24)
    evaluator_id: str = RUNTIME_EVALUATOR_ID
    domain_pack_id: str = "greenhouse_environment@1"
    minimum_coverage: float = 0.80
    selection_minimum_coverage: float = 0.90
    minimum_skill: float = 1e-9
    no_regression_tolerance: float = 1e-12
    maximum_constraint_violations: int = 0
    selection_minimum_score_delta: float = 0.005

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "horizons_hours", tuple(self.horizons_hours))
        object.__setattr__(self, "label_semantics", tuple(self.label_semantics))
        if not self.targets or len(set(self.target_names)) != len(self.targets):
            raise ValueError("dataset adapter must declare unique prediction targets")
        if not math.isclose(sum(self.target_weights.values()), 1.0):
            raise ValueError("dataset adapter target weights must sum to one")
        if (not self.horizons_hours or len(set(self.horizons_hours)) != len(self.horizons_hours)
                or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in self.horizons_hours)):
            raise ValueError("dataset adapter horizons must be unique positive hours")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not 0 < v <= 1
               for v in (self.minimum_coverage, self.selection_minimum_coverage)):
            raise ValueError("dataset adapter coverage must be in (0, 1]")
        if self.selection_minimum_coverage < self.minimum_coverage:
            raise ValueError("selection coverage cannot weaken prediction coverage")
        if any(not math.isfinite(v) or v < 0 for v in (
                self.minimum_skill, self.no_regression_tolerance, self.selection_minimum_score_delta)):
            raise ValueError("dataset adapter scoring thresholds must be finite and nonnegative")
        if (isinstance(self.maximum_constraint_violations, bool)
                or not isinstance(self.maximum_constraint_violations, int)
                or self.maximum_constraint_violations < 0):
            raise ValueError("dataset adapter constraint budget must be a nonnegative integer")

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.targets)

    @property
    def target_weights(self) -> dict[str, float]:
        return {target.name: target.weight for target in self.targets}

    @property
    def target_bounds(self) -> tuple[tuple[str, str, float, float], ...]:
        return tuple((t.name, t.unit, t.minimum, t.maximum) for t in self.targets)

    def contract(self) -> dict[str, Any]:
        body = {
            "schema_version": "ecologyrsi-dsh.dataset-task/1",
            **asdict(self),
            "targets": [asdict(target) for target in self.targets],
            "horizons_hours": list(self.horizons_hours),
            "label_semantics": list(self.label_semantics),
            "evaluation_mode": "historical_replay_prediction_non_causal",
            "sampling": "hourly_mean",
            "primary_metric": "weighted_symmetric_rmse_skill@1",
            "horizon_weighting": "equal",
            "missing_label_policy": "keep_missing_no_interpolation",
            "fit_partition": "calibration_fit",
            "selection_partition": "model_selection",
            "prediction_cells_per_origin": len(self.targets) * len(self.horizons_hours),
        }
        return {**body, "contract_digest": digest(body)}

    def loader(self, dataset_dir: Path) -> GreenhouseDatasetAdapter:
        return GreenhouseDatasetAdapter(self.dataset_id, self.domain_id, dataset_dir)

    def episodes(self, dataset_dir: Path) -> list[dict[str, Any]]:
        return [
            {"id": f"{self.dataset_id}:{path.parent.name}",
             "episode_id": f"{self.dataset_id}:{path.parent.name}",
             "label": path.parent.name, "row_count": None}
            for path in sorted(dataset_dir.glob(f"*/{self.climate_filename}"))
            if "reference" not in path.parent.name.casefold()
        ]

    def validate_series(self, series: Any) -> None:
        if series.dataset_id != self.dataset_id or series.domain_id != self.domain_id:
            raise ValueError("dataset adapter does not match the loaded dataset/domain")
        for target in self.targets:
            feature = series.features.get(target.name)
            if target.name not in series.values or feature is None or feature.unit != target.unit:
                raise ValueError(f"dataset adapter target missing or unit mismatch: {target.name}")


_CLIMATE_TARGETS = (
    PredictionTarget("air_temperature", "室内气温", "degC", -10.0, 60.0, 1 / 3),
    PredictionTarget("relative_humidity", "室内相对湿度", "percent", 0.0, 100.0, 1 / 3),
    PredictionTarget("co2_concentration", "室内 CO₂ 浓度", "ppm", 0.0, 5000.0, 1 / 3),
)

CUCUMBER_2018 = DatasetAdapter(
    adapter_id="agc_cucumber_2018@1", dataset_id="agc_cucumber_2018",
    domain_id="greenhouse_cucumber_2018", label="2018 黄瓜 · 环境预测",
    climate_filename="Greenhouse_climate.csv", targets=_CLIMATE_TARGETS,
    label_semantics=("黄瓜一级果产量为累计产量（kg/m²），保留原始稀疏观测。",
                     "供热能耗单位为 kWh/m²/日；产量、资源和作物数据不计入本任务评分。"),
)
TOMATO_2019 = DatasetAdapter(
    adapter_id="agc_tomato_2019@1", dataset_id="agc_tomato_2019",
    domain_id="greenhouse_tomato_2019", label="2019 番茄 · 环境预测",
    climate_filename="GreenhouseClimate.csv", targets=_CLIMATE_TARGETS,
    label_semantics=("番茄一级果产量为单次采收量（kg/m²），不能与黄瓜累计产量混用。",
                     "供热能耗单位为 MJ/m²/日；稀疏产量和品质标签不插值，也不计入本任务评分。"),
)

DATASET_ADAPTERS = MappingProxyType({item.dataset_id: item for item in (CUCUMBER_2018, TOMATO_2019)})


def dataset_adapter(dataset_id: str) -> DatasetAdapter:
    try:
        return DATASET_ADAPTERS[dataset_id]
    except KeyError:
        raise ValueError(f"no prediction task adapter registered for dataset: {dataset_id}") from None
