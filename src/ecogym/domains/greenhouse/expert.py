"""Diurnal-profile mechanism ensemble: the greenhouse expert solver.

Derived in the October 2026 study (fit on calibration_fit, choices on
model_selection, checked on the untouched calibration_uq):

* Controlled greenhouse climate repeats a daily course, but the regime
  drifts between the August fit period and the October selection period, so
  hour-of-day or weather relationships learned in August transfer poorly.
  The target's own last two days carry the current course: every cell reads
  y(t), y(t-S) and the same-clock values of the past ``profile_days``.
* A few mechanism terms transfer and are declared per target in ``terms``:
  ``trend`` (last change), ``schedule_step`` (yesterday's setpoint change
  over the same clock window), ``exchange`` (vent opening x indoor-outdoor
  CO2 difference), ``now`` (a driver at the origin) and ``gap``
  (setpoint minus target).
* Two members with different biases are averaged: a free standardized ridge
  over all features (pulls toward a constant level) and a constrained
  profile-anomaly regression whose weights sum to one with no intercept
  (keeps the current level): y(t+h) - y(t) = c1 (P - y(t)) + c2 (y(t-S) - y(t)),
  P the mean of the profile reads. ``ensemble_weight`` is the free share.

The forecast is baseline + scale[cell] * residual, with the fit-selected
baseline as anchor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from ...protocol.views import Origins
from ...solvers.base import Param, SolverContext, SolverError, SolverSpec, register_solver
from ...solvers.recipe import CellModel, Column, fit_linear_cell
from ...task.spec import Cell, TaskSpec

MECHANISMS = {
    "trend": (),
    "schedule_step": ("col",),
    "exchange": ("col", "reference"),
    "now": ("col",),
    "gap": ("col",),
}


def validate_mechanisms(value: Any, task: TaskSpec) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, Mapping):
        raise SolverError("terms maps target names to lists of mechanism terms")
    unknown = sorted(set(value) - set(task.target_names))
    if unknown:
        raise SolverError(f"terms names unknown targets {unknown}")
    result: dict[str, list[dict[str, Any]]] = {}
    for target in task.target_names:
        items = value.get(target, [])
        if not isinstance(items, (list, tuple)):
            raise SolverError(f"terms.{target} must be a list")
        normalized = []
        for index, raw in enumerate(items):
            where = f"terms.{target}[{index}]"
            if not isinstance(raw, Mapping) or raw.get("op") not in MECHANISMS:
                raise SolverError(f"{where}.op must be one of {sorted(MECHANISMS)}")
            fields = MECHANISMS[raw["op"]]
            if set(raw) != {"op", *fields}:
                raise SolverError(f"{where} ({raw['op']}) takes exactly {list(fields)}")
            term = {"op": raw["op"]}
            if "col" in fields:
                if not isinstance(raw["col"], str) or not raw["col"]:
                    raise SolverError(f"{where}.col must be a column name")
                term["col"] = raw["col"]
            if "reference" in fields:
                if isinstance(raw["reference"], bool) or not isinstance(raw["reference"], (int, float)):
                    raise SolverError(f"{where}.reference must be a number")
                term["reference"] = float(raw["reference"])
            normalized.append(term)
        result[target] = normalized
    return result


def validate_profile_days(value: Any, task: TaskSpec) -> list[int]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(isinstance(day, bool) or not isinstance(day, int) or day < 1 for day in value)
        or len(set(value)) != len(value)
    ):
        raise SolverError("profile_days is a list of distinct positive integers")
    reach = task.history_steps - 1
    for day in value:
        if task.season_steps * day - min(task.horizons) > reach:
            raise SolverError(f"profile day {day} reaches beyond the task history ({reach} steps)")
    return sorted(value)


def _lag(target: str, k: int) -> Column:
    return lambda origins, ctx, exo: origins.past(target, k)


def _step(column: str, later: int, earlier: int) -> Column:
    return lambda origins, ctx, exo: exo.read(origins, ctx, column, later) - exo.read(origins, ctx, column, earlier)


def _exchange(target: str, column: str, reference: float) -> Column:
    return lambda origins, ctx, exo: exo.read(origins, ctx, column, 0) * (origins.past(target, 0) - reference)


def _now(column: str) -> Column:
    return lambda origins, ctx, exo: exo.read(origins, ctx, column, 0)


def _gap(target: str, column: str) -> Column:
    return lambda origins, ctx, exo: exo.read(origins, ctx, column, 0) - origins.past(target, 0)


Plan = tuple[list[str], list[Column], set[int], set[str], list[int]]


def _plan(cell: Cell, params: Mapping[str, Any], season: int) -> Plan:
    """Feature names, builders, target reads, input columns and profile reads of one cell."""

    target, horizon = cell.target, cell.horizon
    names, builders, reads = ["lag_0", f"lag_{season}"], [_lag(target, 0), _lag(target, season)], {0, season}
    profiles = []
    for day in params["profile_days"]:
        k = season * day - horizon
        if k > 0 and k not in reads:
            names.append(f"profile_{day}")
            builders.append(_lag(target, k))
            reads.add(k)
            profiles.append(k)
    columns: set[str] = set()
    for term in params["terms"].get(target, []):
        op = term["op"]
        if op == "trend":
            names.append("lag_1")
            builders.append(_lag(target, 1))
            reads.add(1)
            continue
        column = term["col"]
        columns.add(column)
        if op == "schedule_step":
            if horizon > season:
                continue  # would read the setpoint after the origin
            names.append(f"{column}:step")
            builders.append(_step(column, season - horizon, season))
        elif op == "exchange":
            names.append(f"{column}:exchange")
            builders.append(_exchange(target, column, term["reference"]))
        elif op == "now":
            names.append(f"{column}@0")
            builders.append(_now(column))
        else:  # gap
            names.append(f"{column}:gap")
            builders.append(_gap(target, column))
    return names, builders, reads, columns, profiles


def _solve_small(matrix: list[list[float]], right: list[float]) -> list[float] | None:
    """Gauss-Jordan for the 1x1 or 2x2 constrained system; None if singular."""

    size = len(right)
    a = [[*row, right[index]] for index, row in enumerate(matrix)]
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


def _constrained(
    names: list[str], features: np.ndarray, observed: np.ndarray, season: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Raw coefficients (over ``names``) of the sum-to-one member predicting y(t+h)."""

    i0, i_season = names.index("lag_0"), names.index(f"lag_{season}")
    profile_columns = [index for index, name in enumerate(names) if name.startswith("profile_")]
    current = features[:, i0]
    regressors = []
    if profile_columns:
        reads = features[:, profile_columns].tolist()
        profile = np.array([math.fsum(row) / len(profile_columns) for row in reads])
        regressors.append(profile - current)
    regressors.append(features[:, i_season] - current)
    response = observed - current
    width = len(regressors)
    gram = [[math.fsum(regressors[a] * regressors[b]) for b in range(width)] for a in range(width)]
    rhs = [math.fsum(regressors[a] * response) for a in range(width)]
    solution = _solve_small(gram, rhs) if features.shape[0] else None
    if solution is None:
        solution = [0.0] * width
    beta = np.zeros(len(names))
    beta[i0] = 1.0 - math.fsum(solution)
    if profile_columns:
        for index in profile_columns:
            beta[index] += solution[0] / len(profile_columns)
    beta[i_season] += solution[-1]
    fitted = {"profile_weight": solution[0] if profile_columns else 0.0, "previous_day_weight": solution[-1]}
    return beta, fitted


def _fit(data: Origins, ctx: SolverContext, params: Mapping[str, Any]) -> dict[str, Any]:
    season = ctx.task.season_steps
    models: dict[str, Any] = {}
    for cell in ctx.task.cells:
        names, builders, reads, columns, profiles = _plan(cell, params, season)
        model, _rows, features, observed = fit_linear_cell(
            data, ctx, cell,
            names=tuple(names), builders=tuple(builders), target_reads=reads, columns=columns,
            anchor="baseline", scale=params["scale"][cell.key], alpha=params["alpha"],
        )
        audit: dict[str, Any] = {}
        if model.status == "fitted":
            if ctx.baselines.kind(cell.key) == "seasonal" and cell.horizon < season:
                if season - cell.horizon not in profiles:
                    raise SolverError(f"{cell.key}: a seasonal anchor needs profile day 1")
                anchor = names.index("profile_1")
            else:
                anchor = names.index("lag_0")
            beta, audit = _constrained(names, features, observed, season)
            beta[anchor] -= 1.0
            standardizer = model.standardizer
            constrained = np.concatenate(([math.fsum(beta * standardizer.mean)], beta * standardizer.scale))
            free = np.concatenate(([model.intercept], model.coefficients))
            weight = params["ensemble_weight"]
            blended = weight * free + (1.0 - weight) * constrained
            if not np.all(np.isfinite(blended)):
                raise SolverError(f"{cell.key}: ensemble coefficients are not finite")
            model = CellModel(
                model.status, model.anchor, model.scale, model.names, model.builders, model.exogenous,
                standardizer, float(blended[0]), blended[1:], model.rows,
            )
            audit["anchor"] = names[anchor]
        models[cell.key] = (model, audit)
    return models


def _predict(models: dict[str, Any], origins: Origins, ctx: SolverContext) -> dict[str, np.ndarray]:
    return {cell.key: models[cell.key][0].predict(origins, ctx, cell) for cell in ctx.task.cells}


def _describe(models: dict[str, Any]) -> dict[str, Any]:
    return {key: {**model.describe(), "constrained": audit} for key, (model, audit) in models.items()}


DIURNAL_PROFILE = register_solver(
    SolverSpec(
        id="greenhouse.diurnal_profile",
        version=1,
        title="Diurnal-profile mechanism ensemble",
        doc=__doc__.strip(),
        params=(
            Param("alpha", "float", 0.5, lo=1e-4, hi=1.0, doc="ridge penalty of the free member"),
            Param("ensemble_weight", "float", 0.5, lo=0.0, hi=1.0, doc="share of the free member in the blend"),
            Param("scale", "cells", 1.0, lo=0.0, hi=1.0, doc="per-cell shrinkage of the residual (0 = baseline)"),
            Param("profile_days", "json", [1, 2], doc="past days read at the label's clock time",
                  validate=validate_profile_days),
            Param("terms", "json", {}, doc="per-target mechanism terms: trend, schedule_step(col), "
                  "exchange(col, reference), now(col), gap(col)", validate=validate_mechanisms),
        ),
        fit=_fit,
        predict=_predict,
        describe_fit=_describe,
    )
)

__all__ = ["DIURNAL_PROFILE", "MECHANISMS", "validate_mechanisms"]
