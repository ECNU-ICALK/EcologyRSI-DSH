"""Shared fitting, scoring, and digest helpers for evaluator services."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Mapping

from ..core.models import digest
from ..data.registry import DatasetSeries

MAX_ROLLING_WINDOW_HOURS = 48

# Evaluation objectives are compared across targets with different physical
# units.  Keep the scale rule explicit and versioned in every evaluation so a
# later change cannot silently make historical scores incomparable.
NORMALIZATION_SCALE_METHOD = "training_fit_std_floor@1"


def _finite_value(value: float | None) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None








def _mae(errors: list[float]) -> float:
    return fmean(abs(item) for item in errors)


def _rmse(errors: list[float]) -> float:
    return math.sqrt(fmean(item * item for item in errors))


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




def artifact_set_digest(bundle: Any) -> str:
    """Return a stable digest useful in tests and read projections."""

    return digest(
        {
            "artifact": bundle.artifact.to_dict(),
            "evaluation": bundle.evaluation.to_dict(),
        }
    )
