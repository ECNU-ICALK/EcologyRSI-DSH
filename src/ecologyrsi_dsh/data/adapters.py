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
from .definitions import DATASET_DEFINITIONS


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
    # Retained, but no longer decides anything. The per-cell gate is now
    # `per_cell_noninferiority@1`; this tolerance survives so archived runs stay
    # replayable and so `would_pass_under_legacy_zero_tolerance` can be computed
    # against the exact rule it names.
    no_regression_tolerance: float = 1e-12
    # The absolute ceiling on a per-cell regression, as a fraction of that
    # cell's baseline normalized RMSE. A wide bootstrap interval can excuse a
    # small regression; nothing excuses one above this line.
    cell_regression_hard_cap: float = 0.03
    # One minus the confidence level of the per-cell paired interval.
    cell_noninferiority_alpha: float = 0.05
    maximum_constraint_violations: int = 0
    selection_minimum_score_delta: float = 0.005

    primary_metric: str = "weighted_symmetric_rmse_skill@1"
    diagnostic_metrics: tuple[str, ...] = ("mae", "rmse", "bias", "normalized_rmse", "skill_score", "sample_execution_coverage", "constraint_violations")
    definition_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostic_metrics", tuple(self.diagnostic_metrics))
        supported = {"mae", "rmse", "bias", "normalized_rmse", "skill_score", "sample_execution_coverage", "constraint_violations"}
        if self.primary_metric != "weighted_symmetric_rmse_skill@1":
            raise ValueError("unsupported dataset primary metric")
        if not self.diagnostic_metrics or not set(self.diagnostic_metrics) <= supported or len(set(self.diagnostic_metrics)) != len(self.diagnostic_metrics):
            raise ValueError("unsupported or duplicate dataset diagnostic metrics")
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
                self.minimum_skill, self.no_regression_tolerance, self.selection_minimum_score_delta,
                self.cell_regression_hard_cap)):
            raise ValueError("dataset adapter scoring thresholds must be finite and nonnegative")
        if (isinstance(self.cell_noninferiority_alpha, bool)
                or not isinstance(self.cell_noninferiority_alpha, (int, float))
                or not 0 < float(self.cell_noninferiority_alpha) < 0.5):
            raise ValueError("dataset adapter noninferiority alpha must lie in (0, 0.5)")
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
            "primary_metric": self.primary_metric,
            "diagnostic_metrics": list(self.diagnostic_metrics),
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


def adapter_from_definition(definition: dict[str, Any]) -> DatasetAdapter:
    task = definition["prediction_task"]
    return DatasetAdapter(
        **{**task, "targets": tuple(PredictionTarget(**t) for t in task["targets"])},
        definition_digest=digest(definition),
    )


DATASET_ADAPTERS = MappingProxyType({
    key: adapter_from_definition(value) for key, value in DATASET_DEFINITIONS.items()
})
CUCUMBER_2018 = DATASET_ADAPTERS["agc_cucumber_2018"]
TOMATO_2019 = DATASET_ADAPTERS["agc_tomato_2019"]


def dataset_adapter(dataset_id: str) -> DatasetAdapter:
    try:
        return DATASET_ADAPTERS[dataset_id]
    except KeyError:
        raise ValueError(f"no prediction task adapter registered for dataset: {dataset_id}") from None
