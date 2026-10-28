"""Which origins of a stage are scored.

An origin is eligible when, for every target, the last ``history_steps``
values (lags 0 .. history_steps-1) are finite and readable and every horizon's
label is finite inside the stage. Fixing one origin set per stage means every
method is scored on the same forecasts, however much history it reads, and
no method can buy an easier cohort by needing less history.
"""

from __future__ import annotations

import numpy as np

from ..task.spec import TaskSpec
from .views import Origins


def eligible(origins: Origins, task: TaskSpec) -> np.ndarray:
    """Boolean mask over ``origins`` (which must carry labels)."""

    ok = np.ones(origins.size, dtype=bool)
    for target in task.target_names:
        for k in range(task.history_steps):
            ok &= np.isfinite(origins.past(target, k))
        for horizon in task.horizons:
            ok &= np.isfinite(origins.label(target, horizon))
    return ok


def blocks(time: np.ndarray, block_steps: int) -> np.ndarray:
    """Calendar block (for hourly data with 24-step blocks: the day) of each origin."""

    return np.floor_divide(time, block_steps)


__all__ = ["blocks", "eligible"]
