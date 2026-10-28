"""Feature-recipe ridge: a declarative model a proposer can edit safely.

A recipe is a list of terms naming host primitives; it never contains code.
For every cell the host builds the terms' features from causal reads, fits a
standardized ridge (``alpha``) to the residual against an anchor (the
fit-selected baseline or persistence), and forecasts

    anchor + scale[cell] * (b0 + b . z)

A scale of 0 returns the anchor exactly. Term bounds come from the task:
target reads may reach back at most ``history_steps - 1`` steps, so every
recipe is scored on the same origins, and seasonal terms use the task's
periods (a seasonal reference is refused when a horizon exceeds the period,
since it would read the future).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..kernel.canonical import canonical_json
from ..protocol.views import Origins
from ..task.spec import Cell, TaskSpec
from .base import Param, SolverContext, SolverError, SolverSpec, register_solver
from .features import ExogenousReads, finite
from .ridge import Standardizer, ridge

MAX_TERMS = 12

# op -> {parameter: kind}; kinds: "lag", "window", "slope", "period", "column"
PRIMITIVES: dict[str, dict[str, str]] = {
    "target_lag": {"k": "lag"},
    "seasonal_reference": {"period": "period"},
    "seasonal_delta": {"period": "period"},
    "diurnal_sin_cos": {"period": "period"},
    "rolling_mean": {"w": "window"},
    "rolling_std": {"w": "window"},
    "target_slope": {"w": "slope"},
    "exogenous": {"col": "column", "lag": "lag"},
    "exogenous_rolling_mean": {"col": "column", "w": "window"},
}

PRIMITIVE_DOCS = {
    "target_lag": "target k steps before the origin",
    "seasonal_reference": "target one period before the label time (what the seasonal baseline reads)",
    "seasonal_delta": "target change between the origin and one period earlier",
    "diurnal_sin_cos": "sine and cosine of the origin's phase in the period",
    "rolling_mean": "mean of the target over the w steps ending at the origin",
    "rolling_std": "population standard deviation of the target over w steps",
    "target_slope": "average change per step of the target over the last w steps",
    "exogenous": "forward-filled input column at a lag (missing -> fit median)",
    "exogenous_rolling_mean": "mean of a forward-filled input column over w steps",
}


def _periods(task: TaskSpec) -> tuple[int, ...]:
    return task.periods or (task.season_steps,)


def _target_reads(term: Mapping[str, Any], horizon: int) -> tuple[int, ...]:
    """Steps back from the origin that a term reads the target at."""

    op = term["op"]
    if op == "target_lag":
        return (term["k"],)
    if op == "seasonal_reference":
        return (term["period"] - horizon,)
    if op == "seasonal_delta":
        return (0, term["period"])
    if op in ("rolling_mean", "rolling_std"):
        return tuple(range(term["w"]))
    if op == "target_slope":
        return (0, term["w"])
    return ()


def validate_terms(value: Any, task: TaskSpec) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_TERMS:
        raise SolverError(f"a recipe is a list of 1..{MAX_TERMS} terms")
    reach = task.history_steps - 1
    terms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        where = f"terms[{index}]"
        if not isinstance(raw, Mapping) or raw.get("op") not in PRIMITIVES:
            raise SolverError(f"{where}.op must be one of {sorted(PRIMITIVES)}")
        schema = PRIMITIVES[raw["op"]]
        if set(raw) != {"op", *schema}:
            raise SolverError(f"{where} ({raw['op']}) takes exactly {sorted(schema)}")
        term: dict[str, Any] = {"op": raw["op"]}
        for name, kind in schema.items():
            item = raw[name]
            if kind == "column":
                if not isinstance(item, str) or not item:
                    raise SolverError(f"{where}.{name} must be a column name")
            else:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise SolverError(f"{where}.{name} must be an integer")
                bounds = {"lag": (0, reach), "window": (2, reach + 1), "slope": (2, reach), "period": (1, None)}
                low, high = bounds[kind]
                if kind == "period" and item not in _periods(task):
                    raise SolverError(f"{where}.{name} must be one of the task periods {_periods(task)}")
                if item < low or (high is not None and item > high):
                    raise SolverError(f"{where}.{name} must lie in [{low}, {high}]")
            term[name] = item
        for horizon in task.horizons:
            reads = _target_reads(term, horizon)
            if any(k < 0 for k in reads):
                raise SolverError(f"{where} would read the target after the origin at horizon {horizon}")
            if any(k > reach for k in reads):
                raise SolverError(f"{where} reaches more than {reach} steps back at horizon {horizon}")
        key = canonical_json(term)
        if key in seen:
            raise SolverError(f"{where} repeats an earlier term")
        seen.add(key)
        terms.append(term)
    return terms


Column = Callable[[Origins, SolverContext, ExogenousReads], np.ndarray]


def _compile(term: Mapping[str, Any], cell: Cell) -> tuple[tuple[str, ...], list[Column]]:
    """Feature names and builders of one term for one cell."""

    op, target = term["op"], cell.target

    def lag(k: int) -> Column:
        return lambda origins, ctx, exo: origins.past(target, k)

    if op == "target_lag":
        return (f"lag_{term['k']}",), [lag(term["k"])]
    if op == "seasonal_reference":
        return (f"seasonal_ref_{term['period']}",), [lag(term["period"] - cell.horizon)]
    if op == "seasonal_delta":
        period = term["period"]
        return (f"seasonal_delta_{period}",), [
            lambda origins, ctx, exo: origins.past(target, 0) - origins.past(target, period)
        ]
    if op == "diurnal_sin_cos":
        period = term["period"]

        def phase(origins: Origins) -> np.ndarray:
            return 2.0 * np.pi * np.mod(origins.time, period) / period

        return (f"sin_{period}", f"cos_{period}"), [
            lambda origins, ctx, exo: np.sin(phase(origins)),
            lambda origins, ctx, exo: np.cos(phase(origins)),
        ]
    if op in ("rolling_mean", "rolling_std"):
        window = term["w"]
        reduce = np.mean if op == "rolling_mean" else np.std
        return (f"{op}_{window}",), [
            lambda origins, ctx, exo: reduce(np.stack([origins.past(target, k) for k in range(window)]), axis=0)
        ]
    if op == "target_slope":
        window = term["w"]
        return (f"slope_{window}",), [
            lambda origins, ctx, exo: (origins.past(target, 0) - origins.past(target, window)) / window
        ]
    column = term["col"]
    if op == "exogenous":
        k = term["lag"]
        return (f"{column}@{k}",), [lambda origins, ctx, exo: exo.read(origins, ctx, column, k)]
    window = term["w"]
    return (f"{column}:mean_{window}",), [
        lambda origins, ctx, exo: np.mean(np.stack([exo.read(origins, ctx, column, k) for k in range(window)]), axis=0)
    ]


@dataclass(frozen=True, slots=True)
class CellModel:
    """One cell's fitted residual model (``status`` "fitted", "baseline" or "fallback")."""

    status: str
    anchor: str
    scale: float
    names: tuple[str, ...] = ()
    builders: tuple[Column, ...] = ()
    exogenous: ExogenousReads | None = None
    standardizer: Standardizer | None = None
    intercept: float = 0.0
    coefficients: np.ndarray | None = None
    rows: int = 0

    def anchor_values(self, origins: Origins, ctx: SolverContext, cell: Cell) -> np.ndarray:
        if self.anchor == "baseline":
            return ctx.baseline(origins, cell.key)
        return origins.past(cell.target, 0)

    def features(self, origins: Origins, ctx: SolverContext) -> np.ndarray:
        if not self.builders:
            return np.zeros((origins.size, 0))
        return np.column_stack([build(origins, ctx, self.exogenous) for build in self.builders])

    def predict(self, origins: Origins, ctx: SolverContext, cell: Cell) -> np.ndarray:
        anchor = self.anchor_values(origins, ctx, cell)
        if self.status != "fitted":
            return anchor
        features = self.features(origins, ctx)
        residual = self.intercept + self.standardizer.transform(features) @ self.coefficients
        residual = np.where(np.isfinite(residual), residual, 0.0)
        forecast = anchor + self.scale * residual
        return np.where(np.all(np.isfinite(features), axis=1), forecast, np.nan)

    def describe(self) -> dict[str, Any]:
        body: dict[str, Any] = {"status": self.status, "anchor": self.anchor, "scale": self.scale, "rows": self.rows}
        if self.status == "fitted":
            body["intercept"] = self.intercept
            body["coefficients"] = dict(zip(self.names, self.coefficients.tolist(), strict=True))
        return body


def fit_linear_cell(
    data: Origins,
    ctx: SolverContext,
    cell: Cell,
    *,
    names: tuple[str, ...],
    builders: tuple[Column, ...],
    target_reads: set[int],
    columns: set[str],
    anchor: str,
    scale: float,
    alpha: float,
) -> tuple[CellModel, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one cell; also return (rows mask, raw features, labels) for derived solvers."""

    empty = np.zeros((0, len(names)))
    if scale == 0.0:
        return CellModel("baseline", anchor, scale), np.zeros(data.size, bool), empty, np.zeros(0)
    labels = data.label(cell.target, cell.horizon)
    reads = [data.past(cell.target, k) for k in sorted(target_reads | {0})]
    rows = finite(labels, *reads)
    if not rows.any():
        return CellModel("fallback", anchor, scale), rows, empty, np.zeros(0)
    exogenous = ExogenousReads.fit(data, rows, ctx, columns)
    model = CellModel("fitted", anchor, scale, names, builders, exogenous)
    subset = data.take(rows)
    features = model.features(subset, ctx)
    residual = labels[rows] - model.anchor_values(subset, ctx, cell)
    try:
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(residual)):
            raise ArithmeticError("non-finite training features")
        standardizer = Standardizer.fit(features)
        intercept, coefficients = ridge(standardizer.transform(features), residual, alpha)
    except ArithmeticError:
        return CellModel("fallback", anchor, scale, rows=int(rows.sum())), rows, features, labels[rows]
    fitted = CellModel(
        "fitted", anchor, scale, names, builders, exogenous, standardizer, intercept, coefficients, int(rows.sum())
    )
    return fitted, rows, features, labels[rows]


def _fit(data: Origins, ctx: SolverContext, params: Mapping[str, Any]) -> dict[str, CellModel]:
    models = {}
    for cell in ctx.task.cells:
        names: list[str] = []
        builders: list[Column] = []
        reads: set[int] = set()
        columns: set[str] = set()
        for term in params["terms"]:
            term_names, term_builders = _compile(term, cell)
            names.extend(term_names)
            builders.extend(term_builders)
            reads.update(_target_reads(term, cell.horizon))
            if "col" in term:
                columns.add(term["col"])
        models[cell.key], *_ = fit_linear_cell(
            data, ctx, cell,
            names=tuple(names), builders=tuple(builders), target_reads=reads, columns=columns,
            anchor=params["anchor"], scale=params["scale"][cell.key], alpha=params["alpha"],
        )
    return models


def _predict(models: dict[str, CellModel], origins: Origins, ctx: SolverContext) -> dict[str, np.ndarray]:
    return {cell.key: models[cell.key].predict(origins, ctx, cell) for cell in ctx.task.cells}


RECIPE_RIDGE = register_solver(
    SolverSpec(
        id="recipe_ridge",
        version=1,
        title="Feature-recipe ridge",
        doc=__doc__.strip(),
        params=(
            Param(
                "terms", "json", [{"op": "target_lag", "k": 0}, {"op": "target_lag", "k": 1}],
                doc="list of terms: " + "; ".join(f"{op}({', '.join(args)}): {PRIMITIVE_DOCS[op]}"
                                                    for op, args in PRIMITIVES.items()),
                validate=validate_terms,
            ),
            Param("alpha", "float", 0.1, lo=1e-4, hi=1.0, doc="ridge penalty on standardized features"),
            Param("anchor", "choice", "baseline", choices=("baseline", "persistence"),
                  doc="residual target: the fit-selected baseline or persistence"),
            Param("scale", "cells", 1.0, lo=0.0, hi=1.0, doc="per-cell shrinkage of the residual (0 = anchor only)"),
        ),
        fit=_fit,
        predict=_predict,
        describe_fit=lambda models: {key: model.describe() for key, model in models.items()},
    )
)

__all__ = ["MAX_TERMS", "PRIMITIVES", "RECIPE_RIDGE", "CellModel", "fit_linear_cell", "validate_terms"]
