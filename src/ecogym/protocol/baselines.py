"""Naive comparators, chosen per cell on calibration_fit.

Every skill score in ecogym is relative to the better of two naive forecasts,
picked once per target and horizon on calibration_fit:

* persistence: y(t+h) = y(t)
* seasonal:    y(t+h) = y(t+h-S), S = the task's season length; causal only
  when h <= S

Both are compared on the same fit rows; seasonal wins only if its RMSE is
lower by more than ``tie_tolerance``. At forecast time a missing seasonal
value falls back to persistence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..kernel.canonical import digest
from ..task.spec import TaskSpec
from .views import Origins


@dataclass(frozen=True, slots=True)
class BaselineProfile:
    kinds: dict[str, str]  # cell key -> "persistence" | "seasonal"
    evidence: dict[str, dict[str, float | int | None]]
    digest: str

    def kind(self, cell_key: str) -> str:
        return self.kinds[cell_key]


def _rmse(errors: np.ndarray) -> float:
    return math.sqrt(math.fsum(errors * errors) / errors.size)


def select_baselines(fit: Origins, task: TaskSpec) -> BaselineProfile:
    if not fit.has_labels:
        raise ValueError("baseline selection needs labelled calibration origins")
    season = task.season_steps
    kinds: dict[str, str] = {}
    evidence: dict[str, dict[str, float | int | None]] = {}
    for cell in task.cells:
        observed = fit.label(cell.target, cell.horizon)
        persistence = fit.past(cell.target, 0)
        kind, record = "persistence", {"comparison_n": 0, "persistence_rmse": None, "seasonal_rmse": None}
        if "seasonal" in task.baselines and cell.horizon <= season:
            seasonal = fit.past(cell.target, season - cell.horizon)
            rows = np.isfinite(observed) & np.isfinite(persistence) & np.isfinite(seasonal)
            if rows.any():
                p_rmse = _rmse(persistence[rows] - observed[rows])
                s_rmse = _rmse(seasonal[rows] - observed[rows])
                record = {"comparison_n": int(rows.sum()), "persistence_rmse": p_rmse, "seasonal_rmse": s_rmse}
                if s_rmse < p_rmse - task.tie_tolerance:
                    kind = "seasonal"
        kinds[cell.key] = kind
        evidence[cell.key] = record
    return BaselineProfile(kinds, evidence, digest({"kinds": kinds, "evidence": evidence}))


def baseline_values(origins: Origins, task: TaskSpec, profile: BaselineProfile) -> dict[str, np.ndarray]:
    """The selected naive forecast for every cell at every origin."""

    values: dict[str, np.ndarray] = {}
    for cell in task.cells:
        persistence = origins.past(cell.target, 0)
        if profile.kind(cell.key) == "seasonal":
            seasonal = origins.past(cell.target, task.season_steps - cell.horizon)
            values[cell.key] = np.where(np.isfinite(seasonal), seasonal, persistence)
        else:
            values[cell.key] = persistence
    return values


__all__ = ["BaselineProfile", "baseline_values", "select_baselines"]
