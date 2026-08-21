"""Leakage-resistant exogenous ridge forecasts for greenhouse time series.

The predictor issues a forecast at an observed timestamp and resolves labels by
an exact timestamp offset.  Fit and feedback rows are prepared independently;
neither lag lookup nor causal forward filling is allowed to cross a partition.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import fmean, median
from typing import Any, Callable, ClassVar, Mapping, Sequence

from ..data.registry import DatasetSeries
from ..data.splits import IndexRange

EXOGENOUS_RIDGE_MODEL_ID = "greenhouse-exogenous-ridge@1"
TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID = "greenhouse-targetwise-ridge@1"
HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID = (
    "greenhouse-horizon-targetwise-ridge@1"
)

_RESULT_SCHEMA = "ecologyrsi-dsh.greenhouse-exogenous-ridge-result/1"
_TARGETWISE_RESULT_SCHEMA = "ecologyrsi-dsh.greenhouse-targetwise-ridge-result/1"
_HORIZON_TARGETWISE_RESULT_SCHEMA = (
    "ecologyrsi-dsh.greenhouse-horizon-targetwise-ridge-result/1"
)
_ALLOWED_EXOGENOUS_ROLES = frozenset(
    {"environment", "outside_weather", "action", "crop", "root_zone", "resource"}
)
_SHORT_FORWARD_FILL_ROLES = frozenset({"environment", "outside_weather", "action"})
_LONG_FORWARD_FILL_ROLES = frozenset({"crop", "root_zone", "resource"})
_FEATURE_COVERAGE_THRESHOLD = 0.2
_MAX_EXOGENOUS_FEATURES = 32
_CONSTANT_EPSILON = 1e-12
MAX_EXOGENOUS_RIDGE_HISTORY_STEPS = 12


@dataclass(frozen=True, slots=True)
class ExogenousRidgeConfig:
    """Validated candidate-controlled ridge parameters."""

    history_steps: int
    ridge_alpha: float
    residual_scale: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.history_steps, bool)
            or not isinstance(self.history_steps, int)
            or not 1 <= self.history_steps <= MAX_EXOGENOUS_RIDGE_HISTORY_STEPS
        ):
            raise ValueError(
                "history_steps must be an integer between 1 and "
                f"{MAX_EXOGENOUS_RIDGE_HISTORY_STEPS}"
            )
        _bounded_number("ridge_alpha", self.ridge_alpha, 1e-4, 1.0)
        _bounded_number("residual_scale", self.residual_scale, 0.0, 1.0)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExogenousRidgeConfig":
        if not isinstance(value, Mapping):
            raise TypeError("exogenous ridge parameters must be an object")
        required = {"history_steps", "ridge_alpha", "residual_scale"}
        missing = required.difference(value)
        unknown = set(value).difference(required)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unsupported " + ", ".join(sorted(unknown)))
            raise ValueError(
                "exogenous ridge parameters are invalid: " + "; ".join(details)
            )
        return cls(
            history_steps=value["history_steps"],
            ridge_alpha=value["ridge_alpha"],
            residual_scale=value["residual_scale"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_steps": self.history_steps,
            "ridge_alpha": float(self.ridge_alpha),
            "residual_scale": float(self.residual_scale),
        }

    def residual_scale_for(self, target: str, horizon_hours: int) -> float:
        del target, horizon_hours
        return float(self.residual_scale)


@dataclass(frozen=True, slots=True)
class TargetwiseExogenousRidgeConfig:
    """Registered ridge parameters with an independent correction per target.

    A zero target scale is the registered persistence fallback.  The model may
    choose only these bounded numbers; feature engineering, fitting, and the
    fallback formula remain host owned.
    """

    history_steps: int
    ridge_alpha: float
    air_temperature_residual_scale: float
    relative_humidity_residual_scale: float
    co2_concentration_residual_scale: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.history_steps, bool)
            or not isinstance(self.history_steps, int)
            or not 1 <= self.history_steps <= MAX_EXOGENOUS_RIDGE_HISTORY_STEPS
        ):
            raise ValueError(
                "history_steps must be an integer between 1 and "
                f"{MAX_EXOGENOUS_RIDGE_HISTORY_STEPS}"
            )
        _bounded_number("ridge_alpha", self.ridge_alpha, 1e-4, 1.0)
        for name in (
            "air_temperature_residual_scale",
            "relative_humidity_residual_scale",
            "co2_concentration_residual_scale",
        ):
            _bounded_number(name, getattr(self, name), 0.0, 1.0)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "TargetwiseExogenousRidgeConfig":
        if not isinstance(value, Mapping):
            raise TypeError("targetwise exogenous ridge parameters must be an object")
        required = {
            "history_steps",
            "ridge_alpha",
            "air_temperature_residual_scale",
            "relative_humidity_residual_scale",
            "co2_concentration_residual_scale",
        }
        missing = required.difference(value)
        unknown = set(value).difference(required)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unsupported " + ", ".join(sorted(unknown)))
            raise ValueError(
                "targetwise exogenous ridge parameters are invalid: "
                + "; ".join(details)
            )
        return cls(**{name: value[name] for name in required})

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_steps": self.history_steps,
            "ridge_alpha": float(self.ridge_alpha),
            "air_temperature_residual_scale": float(
                self.air_temperature_residual_scale
            ),
            "relative_humidity_residual_scale": float(
                self.relative_humidity_residual_scale
            ),
            "co2_concentration_residual_scale": float(
                self.co2_concentration_residual_scale
            ),
        }

    def residual_scale_for(self, target: str, horizon_hours: int) -> float:
        del horizon_hours
        try:
            value = {
                "air_temperature": self.air_temperature_residual_scale,
                "relative_humidity": self.relative_humidity_residual_scale,
                "co2_concentration": self.co2_concentration_residual_scale,
            }[target]
        except KeyError:
            raise ValueError(f"targetwise ridge target is not registered: {target}") from None
        return float(value)


@dataclass(frozen=True, slots=True)
class HorizonTargetwiseExogenousRidgeConfig:
    """Registered residual correction scale for each target and horizon.

    The supported target-horizon grid is deliberately fixed to the registered
    greenhouse multihorizon evaluator. A zero scale selects persistence for
    only that cell while leaving the other cells' ridge corrections active.
    """

    history_steps: int
    ridge_alpha: float
    air_temperature_1h_residual_scale: float
    air_temperature_6h_residual_scale: float
    air_temperature_24h_residual_scale: float
    relative_humidity_1h_residual_scale: float
    relative_humidity_6h_residual_scale: float
    relative_humidity_24h_residual_scale: float
    co2_concentration_1h_residual_scale: float
    co2_concentration_6h_residual_scale: float
    co2_concentration_24h_residual_scale: float

    _SCALE_FIELDS: ClassVar[dict[tuple[str, int], str]] = {
        ("air_temperature", 1): "air_temperature_1h_residual_scale",
        ("air_temperature", 6): "air_temperature_6h_residual_scale",
        ("air_temperature", 24): "air_temperature_24h_residual_scale",
        ("relative_humidity", 1): "relative_humidity_1h_residual_scale",
        ("relative_humidity", 6): "relative_humidity_6h_residual_scale",
        ("relative_humidity", 24): "relative_humidity_24h_residual_scale",
        ("co2_concentration", 1): "co2_concentration_1h_residual_scale",
        ("co2_concentration", 6): "co2_concentration_6h_residual_scale",
        ("co2_concentration", 24): "co2_concentration_24h_residual_scale",
    }

    def __post_init__(self) -> None:
        if (
            isinstance(self.history_steps, bool)
            or not isinstance(self.history_steps, int)
            or not 1 <= self.history_steps <= MAX_EXOGENOUS_RIDGE_HISTORY_STEPS
        ):
            raise ValueError(
                "history_steps must be an integer between 1 and "
                f"{MAX_EXOGENOUS_RIDGE_HISTORY_STEPS}"
            )
        _bounded_number("ridge_alpha", self.ridge_alpha, 1e-4, 1.0)
        for name in self._SCALE_FIELDS.values():
            _bounded_number(name, getattr(self, name), 0.0, 1.0)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "HorizonTargetwiseExogenousRidgeConfig":
        if not isinstance(value, Mapping):
            raise TypeError(
                "horizon-targetwise exogenous ridge parameters must be an object"
            )
        required = {"history_steps", "ridge_alpha", *cls._SCALE_FIELDS.values()}
        missing = required.difference(value)
        unknown = set(value).difference(required)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unsupported " + ", ".join(sorted(unknown)))
            raise ValueError(
                "horizon-targetwise exogenous ridge parameters are invalid: "
                + "; ".join(details)
            )
        return cls(**{name: value[name] for name in required})

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_steps": self.history_steps,
            "ridge_alpha": float(self.ridge_alpha),
            **{
                name: float(getattr(self, name))
                for name in self._SCALE_FIELDS.values()
            },
        }

    def residual_scale_for(self, target: str, horizon_hours: int) -> float:
        try:
            name = self._SCALE_FIELDS[(target, horizon_hours)]
        except KeyError:
            raise ValueError(
                "horizon-targetwise ridge target-horizon is not registered: "
                f"{target}@{horizon_hours}h"
            ) from None
        return float(getattr(self, name))


RidgeConfig = (
    ExogenousRidgeConfig
    | TargetwiseExogenousRidgeConfig
    | HorizonTargetwiseExogenousRidgeConfig
)


@dataclass(frozen=True, slots=True)
class _BaseSample:
    origin_index: int
    label_index: int
    target_lags: tuple[float, ...]
    target_lag_timestamps: tuple[int, ...]
    observed: float
    baseline: float


@dataclass(frozen=True, slots=True)
class _FeatureStatistic:
    name: str
    kind: str
    source_feature: str | None
    coverage: float
    median: float
    mean: float
    scale: float
    constant: bool

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "coverage": self.coverage,
            "median": self.median,
            "mean": self.mean,
            "scale": self.scale,
            "constant": self.constant,
        }
        if self.source_feature is not None:
            value["source_feature"] = self.source_feature
        return value


@dataclass(frozen=True, slots=True)
class _FittedTask:
    target: str
    horizon: int
    fit_samples: tuple[_BaseSample, ...]
    feedback_samples: tuple[_BaseSample, ...]
    model: dict[str, Any]
    coefficients: tuple[float, ...]
    statistics: tuple[_FeatureStatistic, ...]


def validate_exogenous_ridge_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized JSON-safe parameter mapping or raise ``ValueError``."""

    return ExogenousRidgeConfig.from_mapping(value).to_dict()


def validate_targetwise_exogenous_ridge_parameters(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a normalized targetwise ridge parameter mapping."""

    return TargetwiseExogenousRidgeConfig.from_mapping(value).to_dict()


def validate_horizon_targetwise_exogenous_ridge_parameters(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a normalized target-horizon ridge parameter mapping."""

    return HorizonTargetwiseExogenousRidgeConfig.from_mapping(value).to_dict()


def fit_predict_exogenous_ridge(
    series: DatasetSeries,
    *,
    targets: Sequence[str],
    horizons: Sequence[int],
    config: RidgeConfig | Mapping[str, Any],
    evaluation_history_steps: int | None = None,
    defer_prediction_partitions: Sequence[str] = (),
    on_fit_complete: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Fit persistence-residual ridge models and predict both visible partitions.

    Every ``target`` x ``horizon`` model learns feature selection, imputation,
    centering, scaling, and ridge coefficients solely from ``training_fit``.
    Returned rows include both ``training_fit`` and ``training_feedback``. A
    caller may defer selected partitions so those rows contain only the
    label-free inputs needed by a registered host tool; prediction then occurs
    only after an agent selects that tool.
    """

    resolved_config: RidgeConfig = (
        config
        if isinstance(
            config,
            (
                ExogenousRidgeConfig,
                TargetwiseExogenousRidgeConfig,
                HorizonTargetwiseExogenousRidgeConfig,
            ),
        )
        else ExogenousRidgeConfig.from_mapping(config)
    )
    model_id = (
        HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
        if isinstance(resolved_config, HorizonTargetwiseExogenousRidgeConfig)
        else TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
        if isinstance(resolved_config, TargetwiseExogenousRidgeConfig)
        else EXOGENOUS_RIDGE_MODEL_ID
    )
    resolved_targets = _validate_targets(targets, series)
    resolved_horizons = _validate_horizons(horizons)
    _validate_series(series)
    if isinstance(defer_prediction_partitions, (str, bytes)) or not isinstance(
        defer_prediction_partitions, Sequence
    ):
        raise TypeError("defer_prediction_partitions must be a sequence")
    deferred_partitions = frozenset(defer_prediction_partitions)
    unknown_deferred = deferred_partitions.difference(
        {"training_fit", "training_feedback"}
    )
    if unknown_deferred:
        raise ValueError(
            "defer_prediction_partitions contains unsupported partitions: "
            + ", ".join(sorted(unknown_deferred))
        )
    if evaluation_history_steps is not None and (
        isinstance(evaluation_history_steps, bool)
        or not isinstance(evaluation_history_steps, int)
        or not resolved_config.history_steps
        <= evaluation_history_steps
        <= MAX_EXOGENOUS_RIDGE_HISTORY_STEPS
    ):
        raise ValueError(
            "evaluation_history_steps must be an integer between the candidate "
            "history_steps and the supported maximum"
        )

    ranges = {
        "training_fit": series.partitions["training_fit"],
        "training_feedback": series.partitions["training_feedback"],
    }
    external_features = _external_feature_roles(series)
    filled_by_partition = {
        partition: _causal_forward_fill(series, selected, external_features)
        for partition, selected in ranges.items()
    }

    fitted_tasks: list[_FittedTask] = []
    for target in resolved_targets:
        target_external_features = {
            name: role for name, role in external_features.items() if name != target
        }
        for horizon in resolved_horizons:
            fit_samples = _base_samples(
                series,
                target,
                ranges["training_fit"],
                horizon,
                resolved_config.history_steps,
            )
            feedback_samples = _base_samples(
                series,
                target,
                ranges["training_feedback"],
                horizon,
                resolved_config.history_steps,
            )
            if evaluation_history_steps is not None:
                cohort_pairs = {
                    (sample.origin_index, sample.label_index)
                    for sample in _base_samples(
                        series,
                        target,
                        ranges["training_feedback"],
                        horizon,
                        evaluation_history_steps,
                    )
                }
                feedback_samples = tuple(
                    sample
                    for sample in feedback_samples
                    if (sample.origin_index, sample.label_index) in cohort_pairs
                )
            model, coefficients, statistics = _fit_model(
                series,
                target,
                horizon,
                fit_samples,
                target_external_features,
                filled_by_partition["training_fit"],
                resolved_config,
            )
            fitted_tasks.append(
                _FittedTask(
                    target=target,
                    horizon=horizon,
                    fit_samples=fit_samples,
                    feedback_samples=feedback_samples,
                    model=model,
                    coefficients=coefficients,
                    statistics=statistics,
                )
            )

    if on_fit_complete is not None:
        on_fit_complete()

    models: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for fitted in fitted_tasks:
        prediction_fallback_rows = 0
        for partition, samples in (
            ("training_fit", fitted.fit_samples),
            ("training_feedback", fitted.feedback_samples),
        ):
            rows, fallback_rows = _predict_rows(
                series,
                partition,
                fitted.target,
                fitted.horizon,
                samples,
                filled_by_partition[partition],
                fitted.statistics,
                fitted.coefficients,
                resolved_config,
                defer_prediction=partition in deferred_partitions,
            )
            prediction_rows.extend(rows)
            prediction_fallback_rows += fallback_rows

        fitted.model["training_rows"] = len(fitted.fit_samples)
        fitted.model["feedback_rows"] = len(fitted.feedback_samples)
        fitted.model["prediction_fallback_rows"] = prediction_fallback_rows
        fitted.model["fit_digest_sha256"] = _digest(fitted.model)
        models.append(fitted.model)

    result = {
        "schema_version": (
            _HORIZON_TARGETWISE_RESULT_SCHEMA
            if model_id == HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
            else _TARGETWISE_RESULT_SCHEMA
            if model_id == TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
            else _RESULT_SCHEMA
        ),
        "model_id": model_id,
        "parameters": resolved_config.to_dict(),
        "training_partition": "training_fit",
        "evaluation_partition": "training_feedback",
        "feature_policy": {
            "coverage_threshold": _FEATURE_COVERAGE_THRESHOLD,
            "maximum_exogenous_features": _MAX_EXOGENOUS_FEATURES,
            "short_forward_fill_hours": 6,
            "long_forward_fill_hours": 168,
            "target_lag_imputation": False,
            "label_imputation": False,
            "baseline_imputation": False,
        },
        "models": models,
        "prediction_rows": prediction_rows,
    }
    if deferred_partitions:
        result["deferred_prediction_partitions"] = sorted(deferred_partitions)
    if evaluation_history_steps is not None:
        result["evaluation_history_steps"] = evaluation_history_steps
    # Keep this boundary strict even if future feature engineering changes.
    json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return result


def predict_fitted_exogenous_ridge(
    *,
    target: str,
    horizon_hours: int,
    baseline: float,
    label_free_context: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    config: RidgeConfig | Mapping[str, Any],
) -> float:
    """Execute one fitted ridge prediction from a prepared label-free sample.

    This function is the host-owned numerical tool boundary used after a
    planner selects the candidate ridge model. It never consumes the future
    observation and does not require predictions to be materialized in advance.
    """

    if not isinstance(target, str) or not target.strip():
        raise ValueError("ridge tool target must be non-empty text")
    if (
        isinstance(horizon_hours, bool)
        or not isinstance(horizon_hours, int)
        or horizon_hours < 1
    ):
        raise ValueError("ridge tool horizon_hours must be a positive integer")
    baseline_value = _finite_number(baseline, "ridge tool baseline")
    if not isinstance(label_free_context, Mapping):
        raise TypeError("ridge tool label_free_context must be an object")
    resolved_config: RidgeConfig = (
        config
        if isinstance(
            config,
            (
                ExogenousRidgeConfig,
                TargetwiseExogenousRidgeConfig,
                HorizonTargetwiseExogenousRidgeConfig,
            ),
        )
        else ExogenousRidgeConfig.from_mapping(config)
    )
    matching = [
        model
        for model in models
        if isinstance(model, Mapping)
        and model.get("target") == target
        and model.get("horizon_hours") == horizon_hours
    ]
    if len(matching) != 1:
        raise ValueError("ridge tool requires exactly one fitted target-horizon model")
    model = matching[0]
    if model.get("status") != "fitted":
        return baseline_value

    raw_statistics = model.get("feature_statistics")
    raw_coefficients = model.get("coefficients")
    raw_snapshot = label_free_context.get("feature_snapshot")
    if not isinstance(raw_statistics, (list, tuple)) or not isinstance(
        raw_coefficients, Mapping
    ):
        raise ValueError("ridge tool fitted model metadata is incomplete")
    if not isinstance(raw_snapshot, (list, tuple)):
        raise ValueError("ridge tool sample feature_snapshot is missing")
    snapshot: dict[str, float] = {}
    for item in raw_snapshot:
        if not isinstance(item, Mapping):
            raise ValueError("ridge tool feature_snapshot item must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or name in snapshot:
            raise ValueError("ridge tool feature_snapshot names must be unique text")
        snapshot[name] = _finite_number(
            item.get("value"), f"ridge tool feature {name}"
        )

    intercept = _finite_number(model.get("intercept"), "ridge tool intercept")
    predicted_residual = intercept
    expected_names: set[str] = set()
    for item in raw_statistics:
        if not isinstance(item, Mapping):
            raise ValueError("ridge tool feature statistic must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or name in expected_names:
            raise ValueError("ridge tool feature statistic names must be unique text")
        expected_names.add(name)
        if name not in snapshot or name not in raw_coefficients:
            raise ValueError("ridge tool sample does not match the fitted feature schema")
        center = _finite_number(item.get("mean"), f"ridge tool mean {name}")
        scale = _finite_number(item.get("scale"), f"ridge tool scale {name}")
        if scale <= 0:
            raise ValueError("ridge tool feature scale must be positive")
        coefficient = _finite_number(
            raw_coefficients[name], f"ridge tool coefficient {name}"
        )
        predicted_residual += coefficient * ((snapshot[name] - center) / scale)
    if set(snapshot) != expected_names or set(raw_coefficients) != expected_names:
        raise ValueError("ridge tool sample and fitted feature schemas differ")
    if not math.isfinite(predicted_residual):
        raise ArithmeticError("ridge tool residual prediction is not finite")
    predicted = baseline_value + resolved_config.residual_scale_for(
        target, horizon_hours
    ) * predicted_residual
    if not math.isfinite(predicted):
        raise ArithmeticError("ridge tool prediction is not finite")
    return float(predicted)


def _fit_model(
    series: DatasetSeries,
    target: str,
    horizon: int,
    samples: Sequence[_BaseSample],
    external_features: Mapping[str, str],
    filled: Mapping[str, tuple[float | None, ...]],
    config: RidgeConfig,
) -> tuple[dict[str, Any], tuple[float, ...], tuple[_FeatureStatistic, ...]]:
    selected, coverage, medians = _select_external_features(
        samples, external_features, filled
    )
    feature_names = _feature_names(target, config.history_steps, selected)
    feature_kinds = (
        [("target_lag", None)] * config.history_steps
        + [("exogenous", name) for name in selected]
        + [("hour_cycle", None), ("hour_cycle", None)]
    )

    if not samples:
        model = _model_metadata(
            target,
            horizon,
            "fallback_zero_residual",
            "no_usable_training_rows",
            selected,
            (),
            (),
        )
        return model, (), ()

    try:
        raw_rows = [
            _raw_feature_row(series, sample, target, selected, filled, medians)
            for sample in samples
        ]
        statistics: list[_FeatureStatistic] = []
        for column, (name, (kind, source_feature)) in enumerate(
            zip(feature_names, feature_kinds)
        ):
            values = [row[column] for row in raw_rows]
            center = fmean(values)
            variance = fmean((value - center) ** 2 for value in values)
            raw_scale = math.sqrt(max(variance, 0.0))
            if not all(math.isfinite(value) for value in (center, raw_scale)):
                raise ArithmeticError("feature statistics are not finite")
            constant = raw_scale <= _CONSTANT_EPSILON
            feature_coverage = (
                coverage[source_feature] if source_feature is not None else 1.0
            )
            feature_median = (
                medians[source_feature]
                if source_feature is not None
                else float(median(values))
            )
            statistics.append(
                _FeatureStatistic(
                    name=name,
                    kind=kind,
                    source_feature=source_feature,
                    coverage=feature_coverage,
                    median=feature_median,
                    mean=center,
                    scale=1.0 if constant else raw_scale,
                    constant=constant,
                )
            )

        normalized_rows = [_standardize(row, statistics) for row in raw_rows]
        residuals = [sample.observed - sample.baseline for sample in samples]
        if not all(math.isfinite(value) for value in residuals):
            raise ArithmeticError("training residuals are not finite")
        coefficients = _ridge_coefficients(
            normalized_rows, residuals, float(config.ridge_alpha)
        )
        if not all(math.isfinite(value) for value in coefficients):
            raise ArithmeticError("ridge coefficients are not finite")
        status = "fitted"
        fallback_reason = None
    except (ArithmeticError, OverflowError, ValueError):
        coefficients = ()
        statistics = []
        status = "fallback_zero_residual"
        fallback_reason = "numerical_fit_failure"

    model = _model_metadata(
        target,
        horizon,
        status,
        fallback_reason,
        selected,
        statistics,
        coefficients,
    )
    return model, coefficients, tuple(statistics)


def _model_metadata(
    target: str,
    horizon: int,
    status: str,
    fallback_reason: str | None,
    selected: Sequence[str],
    statistics: Sequence[_FeatureStatistic],
    coefficients: Sequence[float],
) -> dict[str, Any]:
    feature_names = [item.name for item in statistics]
    coefficient_values = list(coefficients[1:]) if coefficients else []
    return {
        "target": target,
        "horizon_hours": horizon,
        "status": status,
        "fallback_reason": fallback_reason,
        "selected_exogenous_features": list(selected),
        "feature_names": feature_names,
        "feature_statistics": [item.to_dict() for item in statistics],
        "coefficient_space": "standardized",
        "intercept": float(coefficients[0]) if coefficients else 0.0,
        "coefficients": {
            name: float(value) for name, value in zip(feature_names, coefficient_values)
        },
    }


def _predict_rows(
    series: DatasetSeries,
    partition: str,
    target: str,
    horizon: int,
    samples: Sequence[_BaseSample],
    filled: Mapping[str, tuple[float | None, ...]],
    statistics: Sequence[_FeatureStatistic],
    coefficients: Sequence[float],
    config: RidgeConfig,
    *,
    defer_prediction: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    fallback_rows = 0
    selected = tuple(
        item.source_feature
        for item in statistics
        if item.kind == "exogenous" and item.source_feature is not None
    )
    medians = {
        item.source_feature: item.median
        for item in statistics
        if item.kind == "exogenous" and item.source_feature is not None
    }
    target_residual_scale = config.residual_scale_for(target, horizon)
    predictor_state: dict[str, Any] = {
        "predictor_id": (
            HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
            if isinstance(config, HorizonTargetwiseExogenousRidgeConfig)
            else TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
            if isinstance(config, TargetwiseExogenousRidgeConfig)
            else EXOGENOUS_RIDGE_MODEL_ID
        ),
        "history_steps": int(config.history_steps),
        "ridge_alpha": float(config.ridge_alpha),
        "residual_scale": target_residual_scale,
        "target_residual_scale": target_residual_scale,
        "used_target_persistence": target_residual_scale == 0.0,
    }
    if isinstance(config, TargetwiseExogenousRidgeConfig):
        predictor_state["targetwise_residual_scales"] = {
            name: value
            for name, value in config.to_dict().items()
            if name.endswith("_residual_scale")
        }
    elif isinstance(config, HorizonTargetwiseExogenousRidgeConfig):
        predictor_state["horizon_targetwise_residual_scales"] = {
            name: value
            for name, value in config.to_dict().items()
            if name.endswith("_residual_scale")
        }
    for sample in samples:
        used_fallback = not coefficients or not statistics
        predicted_residual = 0.0
        raw_features: tuple[float, ...] = ()
        if not used_fallback:
            raw_features = _raw_feature_row(
                series, sample, target, selected, filled, medians
            )
            if not defer_prediction:
                normalized = _standardize(raw_features, statistics)
                predicted_residual = coefficients[0] + sum(
                    coefficient * value
                    for coefficient, value in zip(coefficients[1:], normalized)
                )
                if not math.isfinite(predicted_residual):
                    predicted_residual = 0.0
                    used_fallback = True
        predicted = sample.baseline + target_residual_scale * predicted_residual
        if not defer_prediction and not math.isfinite(predicted):
            predicted = sample.baseline
            used_fallback = True
        if not defer_prediction and used_fallback:
            fallback_rows += 1
        row: dict[str, Any] = {
                "partition": partition,
                "target": target,
                "horizon_hours": horizon,
                "origin_index": sample.origin_index,
                "origin_timestamp": series.timestamps[sample.origin_index],
                "label_index": sample.label_index,
                "timestamp": series.timestamps[sample.label_index],
                "observed": sample.observed,
                "baseline": sample.baseline,
                "used_zero_residual_fallback": used_fallback,
                "used_target_persistence": target_residual_scale == 0.0,
                # This context is the only sample-specific data exposed to a
                # collaboration adapter. It contains information available at
                # the forecast origin, never the future observation.
                "label_free_context": {
                    "schema_version": "ecologyrsi-dsh.label-free-sample-context/1",
                    "history_window": [float(value) for value in sample.target_lags],
                    "causal_provenance": {
                        "schema_version": (
                            "ecologyrsi-dsh.causal-sample-provenance/1"
                        ),
                        "origin_cutoff_timestamp": series.timestamps[
                            sample.origin_index
                        ],
                        "latest_context_timestamp": series.timestamps[
                            sample.origin_index
                        ],
                        "history_timestamps": list(sample.target_lag_timestamps),
                    },
                    "feature_snapshot": [
                        {"name": statistic.name, "value": float(value)}
                        for statistic, value in zip(statistics, raw_features)
                    ],
                    "predictor_state": {
                        **predictor_state,
                        "used_registered_fallback": used_fallback,
                    },
                },
            }
        if not defer_prediction:
            row["predicted"] = predicted
        rows.append(row)
    return rows, fallback_rows


def _select_external_features(
    samples: Sequence[_BaseSample],
    external_features: Mapping[str, str],
    filled: Mapping[str, tuple[float | None, ...]],
) -> tuple[tuple[str, ...], dict[str, float], dict[str, float]]:
    if not samples:
        return (), {}, {}
    coverage = {
        name: sum(filled[name][sample.origin_index] is not None for sample in samples)
        / len(samples)
        for name in external_features
    }
    eligible = [
        name
        for name in external_features
        if coverage[name] >= _FEATURE_COVERAGE_THRESHOLD
    ]
    eligible.sort(key=lambda name: (-coverage[name], name))
    selected = tuple(eligible[:_MAX_EXOGENOUS_FEATURES])
    medians = {
        name: float(
            median(
                value
                for sample in samples
                if (value := filled[name][sample.origin_index]) is not None
            )
        )
        for name in selected
    }
    return selected, coverage, medians


def _raw_feature_row(
    series: DatasetSeries,
    sample: _BaseSample,
    target: str,
    selected: Sequence[str],
    filled: Mapping[str, tuple[float | None, ...]],
    medians: Mapping[str, float],
) -> tuple[float, ...]:
    external_values = [
        filled[name][sample.origin_index]
        if filled[name][sample.origin_index] is not None
        else medians[name]
        for name in selected
    ]
    hour_angle = 2.0 * math.pi * (series.timestamps[sample.origin_index] % 24) / 24.0
    result = (
        *sample.target_lags,
        *(float(value) for value in external_values),
        math.sin(hour_angle),
        math.cos(hour_angle),
    )
    if not all(math.isfinite(value) for value in result):
        raise ArithmeticError(f"non-finite prepared feature for {target}")
    return tuple(result)


def _feature_names(
    target: str, history_steps: int, selected: Sequence[str]
) -> tuple[str, ...]:
    return (
        *(f"target:{target}:lag_{lag}h" for lag in range(history_steps)),
        *(f"exogenous:{name}" for name in selected),
        "time:hour_sin",
        "time:hour_cos",
    )


def _standardize(
    values: Sequence[float],
    statistics: Sequence[_FeatureStatistic],
) -> tuple[float, ...]:
    return tuple(
        (value - statistic.mean) / statistic.scale
        for value, statistic in zip(values, statistics)
    )


def _ridge_coefficients(
    feature_rows: Sequence[Sequence[float]],
    labels: Sequence[float],
    alpha: float,
) -> tuple[float, ...]:
    if not feature_rows or len(feature_rows) != len(labels):
        raise ValueError("ridge fit requires aligned non-empty rows")
    width = len(feature_rows[0])
    if any(len(row) != width for row in feature_rows):
        raise ValueError("ridge feature rows have inconsistent widths")
    dimension = width + 1
    gram = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    right = [0.0 for _ in range(dimension)]
    for row, label in zip(feature_rows, labels):
        augmented = (1.0, *row)
        for left_index, left_value in enumerate(augmented):
            right[left_index] += left_value * label
            for right_index in range(left_index, dimension):
                gram[left_index][right_index] += left_value * augmented[right_index]
    for left_index in range(dimension):
        for right_index in range(left_index):
            gram[left_index][right_index] = gram[right_index][left_index]
    for index in range(1, dimension):
        gram[index][index] += alpha
    return _solve_with_partial_pivoting(gram, right)


def _solve_with_partial_pivoting(
    coefficients: Sequence[Sequence[float]],
    right: Sequence[float],
) -> tuple[float, ...]:
    size = len(right)
    if (
        size == 0
        or len(coefficients) != size
        or any(len(row) != size for row in coefficients)
    ):
        raise ValueError("linear system must be non-empty and square")
    matrix = [list(row) + [float(value)] for row, value in zip(coefficients, right)]
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ArithmeticError("linear system contains non-finite values")

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        pivot = matrix[pivot_row][column]
        row_scale = max(abs(value) for value in matrix[pivot_row][:-1])
        if abs(pivot) <= _CONSTANT_EPSILON * max(1.0, row_scale):
            raise ArithmeticError("linear system is singular")
        if pivot_row != column:
            matrix[column], matrix[pivot_row] = matrix[pivot_row], matrix[column]
        for row in range(column + 1, size):
            factor = matrix[row][column] / matrix[column][column]
            matrix[row][column] = 0.0
            for item in range(column + 1, size + 1):
                matrix[row][item] -= factor * matrix[column][item]

    solution = [0.0 for _ in range(size)]
    for row in range(size - 1, -1, -1):
        remainder = matrix[row][size] - sum(
            matrix[row][column] * solution[column] for column in range(row + 1, size)
        )
        pivot = matrix[row][row]
        if abs(pivot) <= _CONSTANT_EPSILON:
            raise ArithmeticError("linear system is singular")
        solution[row] = remainder / pivot
    if not all(math.isfinite(value) for value in solution):
        raise ArithmeticError("linear solution is not finite")
    return tuple(solution)


def _base_samples(
    series: DatasetSeries,
    target: str,
    selected: IndexRange,
    horizon: int,
    history_steps: int,
) -> tuple[_BaseSample, ...]:
    timestamp_to_index = {
        series.timestamps[index]: index for index in range(selected.start, selected.end)
    }
    target_values = series.values[target]
    samples: list[_BaseSample] = []
    for origin_index in range(selected.start, selected.end):
        origin_timestamp = series.timestamps[origin_index]
        label_index = timestamp_to_index.get(origin_timestamp + horizon)
        if label_index is None:
            continue
        lag_indices = [
            timestamp_to_index.get(origin_timestamp - lag)
            for lag in range(history_steps)
        ]
        if any(index is None for index in lag_indices):
            continue
        target_lags = tuple(
            _finite_value(target_values[index])  # type: ignore[index]
            for index in lag_indices
        )
        observed = _finite_value(target_values[label_index])
        if observed is None or any(value is None for value in target_lags):
            continue
        finite_lags = tuple(float(value) for value in target_lags if value is not None)
        finite_lag_timestamps = tuple(
            series.timestamps[index] for index in lag_indices if index is not None
        )
        samples.append(
            _BaseSample(
                origin_index=origin_index,
                label_index=label_index,
                target_lags=finite_lags,
                target_lag_timestamps=finite_lag_timestamps,
                observed=observed,
                baseline=finite_lags[0],
            )
        )
    return tuple(samples)


def _causal_forward_fill(
    series: DatasetSeries,
    selected: IndexRange,
    external_features: Mapping[str, str],
) -> dict[str, tuple[float | None, ...]]:
    output: dict[str, tuple[float | None, ...]] = {}
    for name, role in external_features.items():
        values = series.values[name]
        filled: list[float | None] = [None] * len(series.timestamps)
        last_value: float | None = None
        last_timestamp: int | None = None
        maximum_age = 6 if role in _SHORT_FORWARD_FILL_ROLES else 168
        for index in range(selected.start, selected.end):
            timestamp = series.timestamps[index]
            value = _finite_value(values[index])
            if value is not None:
                last_value = value
                last_timestamp = timestamp
                filled[index] = value
            elif (
                last_value is not None
                and last_timestamp is not None
                and timestamp - last_timestamp <= maximum_age
            ):
                filled[index] = last_value
        output[name] = tuple(filled)
    return output


def _external_feature_roles(series: DatasetSeries) -> dict[str, str]:
    selected: dict[str, str] = {}
    for name in sorted(series.values):
        specification = series.features.get(name)
        role = getattr(specification, "role", None)
        if role in _ALLOWED_EXOGENOUS_ROLES:
            selected[name] = str(role)
    return selected


def _validate_series(series: DatasetSeries) -> None:
    timestamps = series.timestamps
    if not timestamps:
        raise ValueError("dataset series must contain timestamps")
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in timestamps
    ):
        raise ValueError("dataset timestamps must be integer hours")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("dataset timestamps must increase strictly")
    for name, values in series.values.items():
        if len(values) != len(timestamps):
            raise ValueError(f"feature length does not match timestamps: {name}")
    for partition in ("training_fit", "training_feedback"):
        if partition not in series.partitions:
            raise ValueError(f"dataset series is missing partition: {partition}")
        selected = series.partitions[partition]
        if selected.end > len(timestamps):
            raise ValueError(f"dataset partition exceeds timestamps: {partition}")
    fit_range = series.partitions["training_fit"]
    feedback_range = series.partitions["training_feedback"]
    if fit_range.end > feedback_range.start:
        raise ValueError("training_fit must not overlap training_feedback")


def _validate_targets(targets: Sequence[str], series: DatasetSeries) -> tuple[str, ...]:
    if isinstance(targets, (str, bytes)) or not targets:
        raise ValueError("targets must be a non-empty sequence")
    resolved: list[str] = []
    for value in targets:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("target names must be non-empty strings")
        name = value.strip()
        if name not in series.values:
            raise ValueError(f"dataset is missing prediction target: {name}")
        if name in resolved:
            raise ValueError(f"duplicate prediction target: {name}")
        resolved.append(name)
    return tuple(resolved)


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    if isinstance(horizons, (str, bytes)) or not horizons:
        raise ValueError("horizons must be a non-empty sequence")
    resolved: list[int] = []
    for value in horizons:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("horizons must contain positive integer hours")
        if value in resolved:
            raise ValueError(f"duplicate prediction horizon: {value}")
        resolved.append(value)
    return tuple(resolved)


def _bounded_number(name: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def _finite_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite_number(value: Any, name: str) -> float:
    number = _finite_value(value)
    if number is None:
        raise ValueError(f"{name} must be a finite number")
    return number


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "EXOGENOUS_RIDGE_MODEL_ID",
    "HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID",
    "TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID",
    "ExogenousRidgeConfig",
    "HorizonTargetwiseExogenousRidgeConfig",
    "TargetwiseExogenousRidgeConfig",
    "fit_predict_exogenous_ridge",
    "predict_fitted_exogenous_ridge",
    "validate_exogenous_ridge_parameters",
    "validate_horizon_targetwise_exogenous_ridge_parameters",
    "validate_targetwise_exogenous_ridge_parameters",
]
