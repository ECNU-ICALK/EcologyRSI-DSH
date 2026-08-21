"""Shared fitting, scoring, and digest helpers for evaluator services."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Mapping

from ..core.models import digest
from ..data.registry import DatasetSeries
from .objectives import (
    OBJECTIVE_COMPONENT_BOUND,
    clip_normalized_objective,
    normalized_absolute_error_reward,
    skill_score,
)

MAX_ROLLING_WINDOW_HOURS = 48

# Evaluation objectives are compared across targets with different physical
# units.  Keep the scale rule explicit and versioned in every evaluation so a
# later change cannot silently make historical scores incomparable.
NORMALIZATION_SCALE_METHOD = "training_fit_std_floor@1"
NORMALIZED_OBJECTIVE_BOUND = OBJECTIVE_COMPONENT_BOUND


def _greenhouse_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"blend", "window", "bias_scale"}
    if set(value) != required:
        missing = required - set(value)
        unknown = set(value) - required
        detail = []
        if missing:
            detail.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            detail.append("unsupported " + ", ".join(sorted(unknown)))
        raise ValueError("greenhouse parameters are invalid: " + "; ".join(detail))
    blend = value["blend"]
    window = value["window"]
    bias_scale = value["bias_scale"]
    if (
        isinstance(blend, bool)
        or not isinstance(blend, (int, float))
        or not 0 <= float(blend) <= 1
    ):
        raise ValueError("blend must be between 0 and 1")
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or not 1 <= window <= MAX_ROLLING_WINDOW_HOURS
    ):
        raise ValueError(
            f"window must be an integer between 1 and {MAX_ROLLING_WINDOW_HOURS}"
        )
    if (
        isinstance(bias_scale, bool)
        or not isinstance(bias_scale, (int, float))
        or not 0 <= float(bias_scale) <= 2
    ):
        raise ValueError("bias_scale must be between 0 and 2")
    return {"blend": float(blend), "window": window, "bias_scale": float(bias_scale)}


def _finite_value(value: float | None) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _base_prediction(
    values: tuple[float | None, ...],
    index: int,
    parameters: Mapping[str, Any],
    *,
    history_start: int = 0,
    timestamps: tuple[int, ...] | None = None,
    timestamp_index: Mapping[int, int] | None = None,
) -> float | None:
    window = int(parameters["window"])
    if timestamps is not None:
        if len(timestamps) != len(values):
            raise ValueError("timestamps and values must have the same length")
        lookup = timestamp_index or {
            timestamp: position
            for position, timestamp in enumerate(
                timestamps[history_start:index], history_start
            )
        }
        origin_timestamp = timestamps[index]
        previous_index = lookup.get(origin_timestamp - 1)
        if (
            previous_index is None
            or previous_index < history_start
            or previous_index >= index
        ):
            return None
        previous = _finite_value(values[previous_index])
        if previous is None:
            return None
        history_indices = [
            lookup.get(origin_timestamp - lag) for lag in range(1, window + 1)
        ]
        history = [
            item
            for position in history_indices
            if position is not None and history_start <= position < index
            for item in (_finite_value(values[position]),)
            if item is not None
        ]
        if not history:
            return None
        rolling = fmean(history)
        return (
            float(parameters["blend"]) * previous
            + (1 - float(parameters["blend"])) * rolling
        )
    start = max(history_start, index - window)
    previous = _finite_value(values[index - 1]) if index - 1 >= history_start else None
    history = [
        item
        for item in (_finite_value(value) for value in values[start:index])
        if item is not None
    ]
    if previous is None or not history:
        return None
    rolling = fmean(history)
    return (
        float(parameters["blend"]) * previous
        + (1 - float(parameters["blend"])) * rolling
    )


def _fit_bias(
    values: tuple[float | None, ...],
    start: int,
    end: int,
    parameters: Mapping[str, Any],
    *,
    timestamps: tuple[int, ...] | None = None,
    timestamp_index: Mapping[int, int] | None = None,
) -> tuple[float, list[float], list[int]]:
    usable: list[tuple[int, float, float]] = []
    first_timestamp = timestamps[start] if timestamps is not None else None
    for index in range(start, end):
        if first_timestamp is not None and timestamps[index] - first_timestamp < int(
            parameters["window"]
        ):
            continue
        observed = _finite_value(values[index])
        base_prediction = _base_prediction(
            values,
            index,
            parameters,
            history_start=start,
            timestamps=timestamps,
            timestamp_index=timestamp_index,
        )
        if observed is None or base_prediction is None:
            continue
        usable.append((index, observed, base_prediction))
    residuals = [observed - base_prediction for _, observed, base_prediction in usable]
    if not residuals:
        raise ValueError("training_fit has no usable rows")
    learned_bias = fmean(residuals) * float(parameters["bias_scale"])
    errors = [
        base_prediction + learned_bias - observed
        for _, observed, base_prediction in usable
    ]
    return learned_bias, errors, [index for index, _, _ in usable]


def _rolling_eligible_rows(
    timestamps: tuple[int, ...],
    start: int,
    end: int,
    window: int,
    *,
    history_start: int | None = None,
    timestamp_index: Mapping[int, int] | None = None,
    minimum_history_hours: int | None = None,
) -> int:
    """Count timestamp-valid rolling origins before value-level filtering.

    A missing hour must not be silently treated as one array position.  This
    count is used only for the diagnostic missing-row denominator; evaluator
    scoring still checks whether observed and predicted values are usable.
    """

    resolved_history_start = start if history_start is None else history_start
    if not 0 <= resolved_history_start <= start:
        raise ValueError("history_start must be between zero and the score range start")
    if minimum_history_hours is not None and minimum_history_hours < 0:
        raise ValueError("minimum_history_hours must be non-negative")
    lookup = (
        timestamp_index
        if timestamp_index is not None
        else {
            timestamp: position
            for position, timestamp in enumerate(
                timestamps[resolved_history_start:end], resolved_history_start
            )
        }
    )
    first_timestamp = timestamps[resolved_history_start]
    warmup_hours = max(window, int(minimum_history_hours or 0))
    count = 0
    for index in range(start, end):
        origin_timestamp = timestamps[index]
        if origin_timestamp - first_timestamp < warmup_hours:
            continue
        previous = lookup.get(origin_timestamp - 1)
        if previous is None or previous < resolved_history_start or previous >= index:
            continue
        if not any(
            (position := lookup.get(origin_timestamp - lag)) is not None
            and resolved_history_start <= position < index
            for lag in range(1, window + 1)
        ):
            continue
        count += 1
    return count


def _mae(errors: list[float]) -> float:
    return fmean(abs(item) for item in errors)


def _rmse(errors: list[float]) -> float:
    return math.sqrt(fmean(item * item for item in errors))


def _skill_score(candidate_nrmse: float, baseline_nrmse: float) -> float:
    """Return a bounded relative improvement over the baseline.

    The unbounded ratio is useful as a diagnostic, but it is unsafe as a
    cross-target objective: a nearly perfect or nearly constant baseline can
    otherwise dominate the aggregate.  Callers retain the raw nRMSE values
    and can reconstruct the unbounded ratio when needed.
    """

    return skill_score(candidate_nrmse, baseline_nrmse)


def _clip_normalized_objective(value: float, *, bound: float = NORMALIZED_OBJECTIVE_BOUND) -> float:
    """Bound one normalized objective component while preserving finiteness."""

    return clip_normalized_objective(value, bound=bound)


def _normalized_absolute_error_reward(
    baseline_errors: list[float],
    candidate_errors: list[float],
    scale: float,
) -> tuple[float, float]:
    """Return ``(raw_mean, bounded_mean)`` normalized sample rewards.

    Raw reward remains available for diagnostics.  The bounded mean is used
    by the aggregate objective so one outlier cannot overwhelm another target
    merely because its training variance is small.
    """

    return normalized_absolute_error_reward(
        baseline_errors,
        candidate_errors,
        scale,
    )


def _exact_time_eligible_rows(
    series: DatasetSeries,
    start: int,
    end: int,
    horizon: int,
    history_steps: int,
) -> int:
    timestamps = {series.timestamps[index] for index in range(start, end)}
    return sum(
        1
        for origin in timestamps
        if origin + horizon in timestamps
        and all(origin - lag in timestamps for lag in range(history_steps))
    )


def _standard_deviation(values: tuple[float | None, ...]) -> float:
    finite = [
        item for item in (_finite_value(value) for value in values) if item is not None
    ]
    if not finite:
        raise ValueError("cannot scale against an empty training range")
    mean = fmean(finite)
    variance = fmean((item - mean) ** 2 for item in finite)
    return max(math.sqrt(variance), 1e-6)


def _normalization_scale(values: tuple[float | None, ...]) -> tuple[float, str]:
    """Return the frozen cross-unit scale and its audit label."""

    return _standard_deviation(values), NORMALIZATION_SCALE_METHOD


def _judge_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "mae",
        "rmse",
        "normalized_rmse",
        "baseline_normalized_rmse",
        "skill_score",
        "improvement",
        "n",
        "constraint_violations",
        "scientific_pass",
    )
    return {key: metrics[key] for key in allowed if key in metrics}


def artifact_set_digest(bundle: Any) -> str:
    """Return a stable digest useful in tests and read projections."""

    return digest(
        {
            "artifact": bundle.artifact.to_dict(),
            "evaluation": bundle.evaluation.to_dict(),
        }
    )
