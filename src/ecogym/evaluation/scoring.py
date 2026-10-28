"""Per-cell skill and the task objective.

For each cell, over the scored origins of a cohort:

* RMSE of the method and of the fit-selected naive baseline on the rows the
  method forecast (n of the cohort's ``eligible`` rows);
* skill = (RMSE_base - RMSE_method) / max(RMSE_method, RMSE_base), in [-1, 1]
  (0 when both are zero), so a positive value is the familiar relative gain;
* coverage q = n / eligible; the cell objective is q * skill + (1 - q) * (-1),
  and a cell with no forecasts scores -1.

The task objective is the weighted mean of cell objectives, each target's
weight split equally over horizons (``weighted_task_skill_reward@3`` in
v0.10.33). Everything is computed from additive per-block sums, so the block
bootstrap resamples sums instead of re-scoring rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SKILL_EPSILON = 1e-12
MISSING_PENALTY = -1.0


@dataclass(frozen=True, slots=True)
class BlockSums:
    """Additive statistics, shape (cells, blocks)."""

    blocks: np.ndarray  # block ids, sorted
    eligible: np.ndarray
    n: np.ndarray
    sse_method: np.ndarray
    sse_base: np.ndarray

    def total(self, counts: np.ndarray | None = None) -> BlockSums:
        """Collapse blocks; ``counts`` (resamples, blocks) weights each block per resample."""

        if counts is None:
            counts = np.ones((1, self.blocks.size))
        reduce = lambda values: values @ counts.T  # noqa: E731  (cells, resamples)
        return BlockSums(
            np.zeros(0), reduce(self.eligible), reduce(self.n), reduce(self.sse_method), reduce(self.sse_base)
        )


def block_sums(
    method_errors: np.ndarray, base_errors: np.ndarray, block_ids: np.ndarray, scales: np.ndarray
) -> BlockSums:
    """Sum errors (cells, rows) by block; errors are divided by the per-cell ``scales``."""

    blocks, position = np.unique(block_ids, return_inverse=True)
    cells = method_errors.shape[0]
    forecast = np.isfinite(method_errors)
    method = np.where(forecast, method_errors / scales[:, None], 0.0)
    base = np.where(forecast, base_errors / scales[:, None], 0.0)
    shape = (cells, blocks.size)
    eligible, n, sse_method, sse_base = (np.zeros(shape) for _ in range(4))
    for cell in range(cells):
        eligible[cell] = np.bincount(position, minlength=blocks.size)
        n[cell] = np.bincount(position, weights=forecast[cell].astype(float), minlength=blocks.size)
        sse_method[cell] = np.bincount(position, weights=method[cell] ** 2, minlength=blocks.size)
        sse_base[cell] = np.bincount(position, weights=base[cell] ** 2, minlength=blocks.size)
    return BlockSums(blocks, eligible, n, sse_method, sse_base)


def skill(method_rmse: np.ndarray, base_rmse: np.ndarray) -> np.ndarray:
    denominator = np.maximum(method_rmse, base_rmse)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = (base_rmse - method_rmse) / denominator
    return np.where(denominator <= SKILL_EPSILON, 0.0, value)


def cell_objectives(sums: BlockSums) -> np.ndarray:
    """Cell objectives from collapsed sums, shape (cells, resamples)."""

    n = sums.n
    with np.errstate(invalid="ignore", divide="ignore"):
        method = np.sqrt(sums.sse_method / n)
        base = np.sqrt(sums.sse_base / n)
        quality = np.clip(sums.n / sums.eligible, 0.0, 1.0)
    value = quality * np.clip(skill(method, base), -1.0, 1.0) + (1.0 - quality) * MISSING_PENALTY
    return np.where(n > 0, value, MISSING_PENALTY)


def objective(sums: BlockSums, weights: np.ndarray) -> np.ndarray:
    """Weighted task objective of collapsed sums, shape (resamples,)."""

    return weights @ cell_objectives(sums) / weights.sum()


__all__ = ["MISSING_PENALTY", "BlockSums", "block_sums", "cell_objectives", "objective", "skill"]
