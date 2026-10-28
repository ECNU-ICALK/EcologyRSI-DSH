"""Diurnal-profile mechanistic ensemble: the research-derived expert predictor.

Why this structure (offline studies on the development partitions only; fit on
calibration_fit, choices on model_selection, untouched check on calibration_uq):

* The controlled greenhouse climate repeats a daily course, but the regime
  drifts strongly between the August fit period and the October selection
  period (vents mostly closed, lamps on in the morning, shorter days). Models
  that learn hour-of-day or weather relationships from August do not transfer.
  The most recent two days of the target itself carry the current course, so
  every cell reads the target at t, t-24, t+h-24 and t+h-48 (all within the
  47 hours a 48-hour cohort alignment serves).
* A few mechanism terms transfer: the last hourly change for air temperature
  and humidity, yesterday's setpoint steps over the same clock window (the
  controller's schedule repeats day to day), and CO2 mass-balance proxies
  (ventilation exchange with outside air, light-driven uptake, the dosing gap
  to the setpoint, yesterday's CO2-setpoint step).
* Two members with different biases are averaged: a free ridge over all
  terms (pulls toward a constant level) and a constrained profile-anomaly
  regression whose weights sum to one with no intercept (keeps the current
  level). The average beat both members in every period checked.

The fitted result is one linear model per target-horizon cell over the plan's
features, stored in the same standardized-coefficient form as every other
registered ridge, so fit, deferred inference and Agent tools share one path.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..core.models import digest
from .feature_recipe import FeaturePlan, _CompiledTerm

DIURNAL_PROFILE_MODEL_ID = "greenhouse-diurnal-profile-ensemble@1"
DIURNAL_PROFILE_PLAN_VERSION = "greenhouse-diurnal-profile-plan/1"
CO2_OUTSIDE_REFERENCE_PPM = 400.0
PROFILE_DAYS = (1, 2)

_AIR_TEMPERATURE = "air_temperature"
_RELATIVE_HUMIDITY = "relative_humidity"
_CO2 = "co2_concentration"
_TREND_TARGETS = frozenset({_AIR_TEMPERATURE, _RELATIVE_HUMIDITY})
_TEMPERATURE_SCHEDULE_COLUMNS = ("heating_setpoint", "ventilation_setpoint")
_CO2_VENT_COLUMNS = ("ventilation_leeward", "ventilation_windward")
_CO2_LIGHT_COLUMNS = ("outside_radiation", "supplemental_light")
_CO2_SETPOINT_COLUMN = "co2_setpoint"


def lag_name(target: str, lag: int) -> str:
    return f"target:{target}:lag_{lag}h"


def profile_name(target: str, days: int) -> str:
    return f"target:{target}:seasonal_{24 * days}h_reference"


def _target_term(name: str, offset: int) -> _CompiledTerm:
    return _CompiledTerm(
        feature_names=(name,),
        target_offsets=(offset,),
        exogenous_offsets=(),
        evaluate=lambda origin, reads, external: (reads[offset],),
    )


def _schedule_step_term(column: str, horizon: int) -> _CompiledTerm:
    """Yesterday's change of a schedule over the same clock window."""

    later, earlier = (column, horizon - 24), (column, -24)
    return _CompiledTerm(
        feature_names=(f"exogenous:{column}:schedule_step_{horizon}h",),
        target_offsets=(),
        exogenous_offsets=(later, earlier),
        evaluate=lambda origin, reads, external: (external[later] - external[earlier],),
    )


def _exogenous_now_term(column: str) -> _CompiledTerm:
    key = (column, 0)
    return _CompiledTerm(
        feature_names=(f"exogenous:{column}:lag_0h",),
        target_offsets=(),
        exogenous_offsets=(key,),
        evaluate=lambda origin, reads, external: (external[key],),
    )


def _ventilation_exchange_term(column: str) -> _CompiledTerm:
    """Opening times the indoor-outdoor CO2 difference: the ventilation loss proxy."""

    key = (column, 0)
    return _CompiledTerm(
        feature_names=(f"exogenous:{column}:co2_exchange",),
        target_offsets=(0,),
        exogenous_offsets=(key,),
        evaluate=lambda origin, reads, external: (
            external[key] * (reads[0] - CO2_OUTSIDE_REFERENCE_PPM),
        ),
    )


def _dosing_gap_term() -> _CompiledTerm:
    key = (_CO2_SETPOINT_COLUMN, 0)
    return _CompiledTerm(
        feature_names=(f"exogenous:{_CO2_SETPOINT_COLUMN}:dosing_gap",),
        target_offsets=(0,),
        exogenous_offsets=(key,),
        evaluate=lambda origin, reads, external: (external[key] - reads[0],),
    )


def _column_available(series: Any, column: str, allowed_roles: frozenset[str]) -> bool:
    if series is None:
        return True
    specification = getattr(series, "features", {}).get(column)
    return column in getattr(series, "values", {}) and getattr(specification, "role", None) in allowed_roles


def compile_diurnal_profile_plan(
    *,
    target: str,
    horizon_hours: int,
    series: Any = None,
    allowed_roles: frozenset[str] = frozenset(),
    origin_history_alignment: int | None = None,
) -> FeaturePlan:
    """Compile one cell's causal read plan. Every offset is <= 0 by construction."""

    if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, int) or horizon_hours < 1:
        raise ValueError("diurnal profile horizon must be a positive whole number of hours")
    if series is not None and target not in getattr(series, "values", {}):
        raise ValueError(f"diurnal profile target is not present in the dataset: {target}")
    terms: list[_CompiledTerm] = [_target_term(lag_name(target, 0), 0), _target_term(lag_name(target, 24), -24)]
    lag_offsets = {0, -24}
    for days in PROFILE_DAYS:
        offset = horizon_hours - 24 * days
        # A same-clock read of a past day; skipped when it is already a lag
        # term (h=24) or would lie after the origin (h>24*days).
        if offset < 0 and offset not in lag_offsets:
            terms.append(_target_term(profile_name(target, days), offset))
    if target in _TREND_TARGETS:
        terms.append(_target_term(lag_name(target, 1), -1))
    columns: list[str] = []
    if target == _AIR_TEMPERATURE and horizon_hours <= 24:
        for column in _TEMPERATURE_SCHEDULE_COLUMNS:
            if _column_available(series, column, allowed_roles):
                terms.append(_schedule_step_term(column, horizon_hours))
                columns.append(column)
    if target == _CO2:
        for column in _CO2_VENT_COLUMNS:
            if _column_available(series, column, allowed_roles):
                terms.append(_ventilation_exchange_term(column))
        for column in _CO2_LIGHT_COLUMNS:
            if _column_available(series, column, allowed_roles):
                terms.append(_exogenous_now_term(column))
        if _column_available(series, _CO2_SETPOINT_COLUMN, allowed_roles):
            terms.append(_dosing_gap_term())
            if horizon_hours <= 24:
                terms.append(_schedule_step_term(_CO2_SETPOINT_COLUMN, horizon_hours))
    feature_names: list[str] = []
    target_offsets: set[int] = {0}
    exogenous_offsets: dict[str, set[int]] = {}
    for term in terms:
        if any(offset > 0 for offset in term.target_offsets) or any(
                offset > 0 for _, offset in term.exogenous_offsets):
            raise ValueError("diurnal profile term would read after the origin")
        feature_names.extend(term.feature_names)
        target_offsets.update(term.target_offsets)
        for column, offset in term.exogenous_offsets:
            exogenous_offsets.setdefault(column, set()).add(offset)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("diurnal profile plan produced duplicate feature names")
    plan_identity = {
        "version": DIURNAL_PROFILE_PLAN_VERSION,
        "target": target,
        "horizon_hours": horizon_hours,
        "feature_names": feature_names,
        "co2_outside_reference_ppm": CO2_OUTSIDE_REFERENCE_PPM,
    }
    plan = FeaturePlan(
        target=target,
        horizon_hours=horizon_hours,
        feature_names=tuple(feature_names),
        required_timestamp_offsets=tuple(sorted(target_offsets, reverse=True)),
        exogenous_timestamp_offsets={
            column: tuple(sorted(offsets, reverse=True))
            for column, offsets in sorted(exogenous_offsets.items())
        },
        recipe_digest=digest(plan_identity),
        _terms=tuple(terms),
    )
    if origin_history_alignment is not None and plan.max_history_hours > origin_history_alignment - 1:
        raise ValueError(
            f"diurnal profile needs {plan.max_history_hours}h of history for {target}@{horizon_hours}h "
            f"but the cohort only guarantees {origin_history_alignment - 1}h"
        )
    return plan


def _solve_small(matrix: list[list[float]], right: list[float]) -> list[float] | None:
    """Gauss-Jordan for the 1x1 / 2x2 constrained member; None if singular."""

    size = len(right)
    a = [row[:] + [right[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(a[row][column]))
        if abs(a[pivot][column]) <= 1e-12 * max(1.0, max(abs(v) for v in a[pivot][:size])):
            return None
        a[column], a[pivot] = a[pivot], a[column]
        for row in range(size):
            if row != column:
                factor = a[row][column] / a[column][column]
                for item in range(column, size + 1):
                    a[row][item] -= factor * a[column][item]
    return [a[index][size] / a[index][index] for index in range(size)]


def constrained_member_raw_coefficients(
    plan: FeaturePlan,
    raw_rows: Sequence[Sequence[float]],
    observed: Sequence[float],
) -> tuple[list[float], dict[str, float]]:
    """Sum-to-one profile anomaly member, no intercept, in raw feature space.

    y(t+h) - y(t) = c1 * (P(t+h) - y(t)) + c2 * (y(t-24) - y(t)), where P is the
    mean of the available same-clock past-day reads. Returns the raw coefficient
    vector over ``plan.feature_names`` (the prediction is beta . x) and the fitted c.
    """

    names = list(plan.feature_names)
    target = plan.target
    i0 = names.index(lag_name(target, 0))
    i24 = names.index(lag_name(target, 24))
    profile_indices = [names.index(profile_name(target, days)) for days in PROFILE_DAYS
                       if profile_name(target, days) in names]
    regressors: list[list[float]] = []
    responses: list[float] = []
    for row, value in zip(raw_rows, observed):
        current = row[i0]
        columns = []
        if profile_indices:
            columns.append(math.fsum(row[i] for i in profile_indices) / len(profile_indices) - current)
        columns.append(row[i24] - current)
        regressors.append(columns)
        responses.append(value - current)
    width = len(regressors[0]) if regressors else (2 if profile_indices else 1)
    gram = [[math.fsum(r[a] * r[b] for r in regressors) for b in range(width)] for a in range(width)]
    rhs = [math.fsum(r[a] * v for r, v in zip(regressors, responses)) for a in range(width)]
    solution = _solve_small(gram, rhs) if regressors else None
    if solution is None:
        solution = [0.0] * width
    beta = [0.0] * len(names)
    beta[i0] = 1.0 - math.fsum(solution)
    if profile_indices:
        for index in profile_indices:
            beta[index] += solution[0] / len(profile_indices)
        beta[i24] += solution[1]
    else:
        beta[i24] += solution[0]
    fitted = {"profile_anomaly_weight": solution[0] if profile_indices else 0.0,
              "previous_day_weight": solution[-1]}
    return beta, fitted


def ensemble_standardized_coefficients(
    *,
    plan: FeaturePlan,
    raw_rows: Sequence[Sequence[float]],
    residuals: Sequence[float],
    baselines: Sequence[float],
    means: Sequence[float],
    scales: Sequence[float],
    free_coefficients: Sequence[float],
    anchor_feature: str,
    ensemble_weight: float,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Blend the free ridge (standardized) with the constrained member.

    Both members predict the residual ``y(t+h) - baseline`` linearly in the
    plan's features. The constrained raw coefficients are re-expressed in the
    standardized space (coef_j * scale_j, intercept sum coef_j * mean_j) after
    subtracting the anchor feature, which is the baseline itself.
    """

    names = list(plan.feature_names)
    if anchor_feature not in names:
        raise ValueError("diurnal profile anchor is not a plan feature")
    observed = [residual + baseline for residual, baseline in zip(residuals, baselines)]
    beta, fitted = constrained_member_raw_coefficients(plan, raw_rows, observed)
    beta[names.index(anchor_feature)] -= 1.0
    constrained = [math.fsum(b * m for b, m in zip(beta, means))] + [b * s for b, s in zip(beta, scales)]
    if len(free_coefficients) != len(constrained):
        raise ValueError("diurnal profile members disagree on the feature width")
    weight = float(ensemble_weight)
    blended = tuple(weight * f + (1.0 - weight) * c for f, c in zip(free_coefficients, constrained))
    if not all(math.isfinite(value) for value in blended):
        raise ArithmeticError("diurnal profile coefficients are not finite")
    return blended, {"constrained_member": fitted, "ensemble_weight": weight}


def plan_summary(plans: Mapping[tuple[str, int], FeaturePlan]) -> list[dict[str, Any]]:
    return [plans[key].summary() for key in sorted(plans)]


__all__ = [
    "CO2_OUTSIDE_REFERENCE_PPM",
    "DIURNAL_PROFILE_MODEL_ID",
    "DIURNAL_PROFILE_PLAN_VERSION",
    "compile_diurnal_profile_plan",
    "constrained_member_raw_coefficients",
    "ensemble_standardized_coefficients",
    "lag_name",
    "plan_summary",
    "profile_name",
]
