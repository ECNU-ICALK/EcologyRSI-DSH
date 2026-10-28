"""Standardized ridge regression on numpy.

Features are centred and divided by their population standard deviation on
the fit rows (a constant feature keeps scale 1, so it centres to zero and its
coefficient vanishes). The intercept is unpenalized; ``alpha`` is added to
the diagonal of the unnormalized Gram matrix of the standardized features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONSTANT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> Standardizer:
        mean = features.mean(axis=0)
        spread = np.sqrt(np.maximum(((features - mean) ** 2).mean(axis=0), 0.0))
        scale = np.where(spread <= CONSTANT_EPSILON, 1.0, spread)
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(scale))):
            raise ArithmeticError("feature statistics are not finite")
        return cls(mean, scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.scale


def ridge(standardized: np.ndarray, response: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """Return (intercept, coefficients) minimizing ||y - b0 - Zb||^2 + alpha ||b||^2."""

    rows, width = standardized.shape
    if rows == 0:
        raise ArithmeticError("ridge fit has no rows")
    design = np.empty((rows, width + 1))
    design[:, 0] = 1.0
    design[:, 1:] = standardized
    gram = design.T @ design
    gram[np.arange(1, width + 1), np.arange(1, width + 1)] += alpha
    try:
        solution = np.linalg.solve(gram, design.T @ response)
    except np.linalg.LinAlgError:
        raise ArithmeticError("ridge system is singular") from None
    if not np.all(np.isfinite(solution)):
        raise ArithmeticError("ridge solution is not finite")
    return float(solution[0]), solution[1:]


__all__ = ["CONSTANT_EPSILON", "Standardizer", "ridge"]
