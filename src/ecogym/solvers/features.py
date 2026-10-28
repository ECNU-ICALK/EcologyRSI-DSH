"""Shared helpers for building feature matrices from causal reads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from ..protocol.views import Origins
from .base import SolverContext, SolverError


def finite(*arrays: np.ndarray) -> np.ndarray:
    """Rows where every array is finite."""

    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


class ExogenousReads:
    """Forward-filled input columns with fit-sample median imputation.

    Coverage and the median are taken from the column's value at the origin
    over the fit rows; a column observed at fewer than the task's
    ``min_coverage`` share of them is refused rather than silently imputed.
    """

    def __init__(self, columns: Iterable[str], medians: Mapping[str, float]) -> None:
        self.columns = tuple(columns)
        self.medians = dict(medians)

    @classmethod
    def fit(cls, origins: Origins, rows: np.ndarray, ctx: SolverContext, columns: Iterable[str]) -> ExogenousReads:
        medians = {}
        for column in sorted(set(columns)):
            if column not in origins.columns:
                raise SolverError(f"column {column} is not in the dataset")
            values = ctx.exogenous(origins, column, 0)[rows]
            observed = values[np.isfinite(values)]
            coverage = observed.size / values.size if values.size else 0.0
            if coverage < ctx.task.min_coverage:
                raise SolverError(
                    f"column {column} is observed at {coverage:.3f} of fit origins, below {ctx.task.min_coverage}"
                )
            medians[column] = float(np.median(observed))
        return cls(medians.keys(), medians)

    def read(self, origins: Origins, ctx: SolverContext, column: str, k: int) -> np.ndarray:
        values = ctx.exogenous(origins, column, k)
        return np.where(np.isfinite(values), values, self.medians[column])


__all__ = ["ExogenousReads", "finite"]
