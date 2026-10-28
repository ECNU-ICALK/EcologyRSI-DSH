"""Ridge regression in numpy (fast and stable on small samples)."""

import numpy as np


class Ridge:
    """Ridge on standardized features; the intercept is not penalized. Rows with NaN are skipped in fit."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.mean = self.scale = self.coef = None
        self.intercept = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        if len(y) < X.shape[1] + 2:
            raise ValueError(f"only {len(y)} complete rows for {X.shape[1]} features")
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        Z = (X - self.mean) / self.scale
        self.intercept = float(y.mean())
        gram = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef = np.linalg.solve(gram, Z.T @ (y - self.intercept))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept + ((X - self.mean) / self.scale) @ self.coef
