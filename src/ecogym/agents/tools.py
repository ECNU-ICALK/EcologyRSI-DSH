"""Research tools an agent may call while preparing proposals.

Data access is limited to calibration_fit (the fit period) and the current
generation's screening days of model_selection. The holdout days that will
judge this generation's finalist, calibration_uq and the sealed stages are
never touched, so the holdout comparison stays independent of everything
the agent looked at in this generation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..arena.genome import Edit, apply_edit, check_edit
from ..evaluation.harness import Bench, StageData
from ..solvers.base import Genome, SolverError, get_solver

STAGES = ("calibration_fit", "screening")


def _round(value: float, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    return round(value, max(0, digits - 1 - math.floor(math.log10(abs(value)))))


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return _round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3)


class ResearchTools:
    def __init__(self, bench: Bench, incumbent: Genome, screening: np.ndarray, *, max_step: float,
                 max_changes: int) -> None:
        self.bench = bench
        self.incumbent = incumbent
        self.screening = np.asarray(screening, dtype=np.int64)
        self.max_step = max_step
        self.max_changes = max_changes

    @staticmethod
    def catalog() -> list[dict[str, Any]]:
        return [
            {"name": "overview", "args": {},
             "returns": "cells, naive baselines, input columns with role/unit/coverage, incumbent skill per cell"},
            {"name": "column_profile", "args": {"column": "name", "stage": "calibration_fit | screening"},
             "returns": "summary statistics, mean by hour of day, correlation with each cell's baseline residual"},
            {"name": "residual_profile", "args": {"cell": "target@horizon", "stage": "calibration_fit | screening"},
             "returns": "incumbent error: RMSE vs baseline, bias, mean by hour of day and by day, "
                        "inputs most correlated with the error"},
            {"name": "try_edit", "args": {"changes": {"path": "value"}},
             "returns": "trust-region check, then the edited method's skill change on this generation's "
                        "screening days, overall and per cell"},
        ]

    def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        handlers = {"overview": self.overview, "column_profile": self.column_profile,
                    "residual_profile": self.residual_profile, "try_edit": self.try_edit}
        if name not in handlers:
            return {"error": f"unknown tool {name}; tools: {sorted(handlers)}"}
        try:
            return handlers[name](**dict(args))
        except (TypeError, ValueError, KeyError) as error:
            return {"error": f"{type(error).__name__}: {error}"}

    def _stage(self, stage: str) -> tuple[StageData, np.ndarray]:
        if stage == "calibration_fit":
            data = self.bench.stage("calibration_fit")
            return data, np.arange(data.size)
        if stage == "screening":
            return self.bench.stage("model_selection"), self.screening
        raise ValueError(f"stage must be one of {STAGES}")

    def _inputs(self) -> list[str]:
        task = self.bench.task
        frame = self.bench.frame
        return [name for name in sorted(frame.columns) if frame.specs[name].role in task.exogenous_roles]

    def overview(self) -> dict[str, Any]:
        task, bench = self.bench.task, self.bench
        fit = bench.stage("calibration_fit")
        columns = []
        for name in self._inputs():
            spec = bench.frame.specs[name]
            values = fit.origins.past(name, 0)
            columns.append({"column": name, "role": spec.role, "unit": spec.unit, "label": spec.label,
                            "coverage_fit": _round(float(np.isfinite(values).mean()), 3)})
        incumbent = bench.evaluate(self.incumbent, "model_selection")
        return {
            "task": task.id,
            "horizons_steps": list(task.horizons),
            "season_steps": task.season_steps,
            "history_steps": task.history_steps,
            "baselines": bench.baselines.kinds,
            "inputs": columns,
            "screening_origins": int(self.screening.size),
            "incumbent_screening_skill": {row["cell"]: _round(row["skill"], 3)
                                          for row in incumbent.cells(self.screening)},
        }

    def column_profile(self, column: str, stage: str = "calibration_fit") -> dict[str, Any]:
        if column not in self.bench.frame.columns:
            raise KeyError(f"unknown column {column}")
        data, index = self._stage(stage)
        origins = data.origins.take(index)
        values = origins.past(column, 0)
        finite = values[np.isfinite(values)]
        season = self.bench.task.season_steps
        phase = np.mod(origins.time, season)
        profile = [_round(float(np.nanmean(values[phase == hour]))) if np.isfinite(values[phase == hour]).any()
                   else None for hour in range(season)]
        residual = data.observed[:, index] - data.baseline[:, index]
        spec = self.bench.frame.specs[column]
        return {
            "column": column, "role": spec.role, "unit": spec.unit, "stage": stage, "origins": int(index.size),
            "coverage": _round(float(finite.size / max(1, values.size)), 3),
            "mean": _round(float(finite.mean())) if finite.size else None,
            "sd": _round(float(finite.std())) if finite.size else None,
            "p05_p50_p95": [_round(float(q)) for q in np.percentile(finite, [5, 50, 95])] if finite.size else None,
            "mean_by_phase": profile,
            "corr_with_baseline_residual": {cell.key: _corr(values, residual[position])
                                            for position, cell in enumerate(data.cells)},
        }

    def residual_profile(self, cell: str, stage: str = "screening") -> dict[str, Any]:
        data, index = self._stage(stage)
        keys = [item.key for item in data.cells]
        if cell not in keys:
            raise KeyError(f"unknown cell {cell}; cells: {keys}")
        position = keys.index(cell)
        evaluation = self.bench.evaluate(self.incumbent, data.name)
        error = evaluation.errors[position, index]
        base = data.baseline[position, index] - data.observed[position, index]
        origins = data.origins.take(index)
        season = self.bench.task.season_steps
        phase = np.mod(origins.time, season)
        days = np.floor_divide(origins.time, self.bench.task.block_steps)
        inputs = {name: _corr(origins.past(name, 0), error) for name in self._inputs()}
        ranked = sorted(((name, r) for name, r in inputs.items() if r is not None), key=lambda item: -abs(item[1]))
        rmse = float(np.sqrt(np.nanmean(error**2)))
        base_rmse = float(np.sqrt(np.nanmean(base**2)))
        return {
            "cell": cell, "stage": stage, "origins": int(index.size), "note": "error = forecast - observed",
            "rmse": _round(rmse), "baseline_rmse": _round(base_rmse),
            "skill": _round((base_rmse - rmse) / max(rmse, base_rmse, 1e-12), 3),
            "bias": _round(float(np.nanmean(error))),
            "mean_error_by_phase": [_round(float(np.nanmean(error[phase == hour]))) if (phase == hour).any() else None
                                    for hour in range(season)],
            "mean_error_by_block": [_round(float(np.nanmean(error[days == day]))) for day in np.unique(days)],
            "inputs_most_correlated_with_error": ranked[:8],
        }

    def try_edit(self, changes: Mapping[str, Any]) -> dict[str, Any]:
        try:
            child = apply_edit(self.incumbent, Edit(dict(changes)), self.bench.task)
        except SolverError as error:
            return {"valid": False, "problems": [str(error)]}
        problems = check_edit(self.incumbent, child, max_step=self.max_step, max_changes=self.max_changes)
        if problems:
            return {"valid": False, "problems": problems}
        try:
            candidate = self.bench.evaluate(child, "model_selection")
        except SolverError as error:
            return {"valid": False, "problems": [str(error)]}
        incumbent = self.bench.evaluate(self.incumbent, "model_selection")
        before = {row["cell"]: row["objective"] for row in incumbent.cells(self.screening)}
        after = {row["cell"]: row["objective"] for row in candidate.cells(self.screening)}
        return {
            "valid": True,
            "solver": get_solver(child.solver).ref,
            "screening_delta": _round(candidate.score(self.screening) - incumbent.score(self.screening), 3),
            "cell_deltas": {key: _round(after[key] - before[key], 3) for key in after},
            "note": "screening days only; the holdout decides promotion",
        }


__all__ = ["STAGES", "ResearchTools"]
