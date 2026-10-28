"""The submission: the judge imports this file and uses ``Predictor``.

Protocol (see TASK.md):

    p = Predictor(spec)          # spec: targets, horizons (steps), step, season_steps, baselines, columns
    p.fit(history)               # DataFrame (DatetimeIndex) of every row before the scored period
    p.forecast(history)          # history up to and including the origin row (its last row)
        -> {target: [forecast for each horizon in spec["horizons"]]}

This starting version is the per-cell naive baseline the judge scores against
(persistence, or the same time one season ago), so its skill is 0.
"""

import numpy as np


class Predictor:
    def __init__(self, spec):
        self.spec = spec
        self.targets = spec["targets"]
        self.horizons = spec["horizons"]
        self.season = spec["season_steps"]
        self.baselines = spec["baselines"]  # {"target@h": "persistence" | "seasonal"}

    def fit(self, history):
        pass

    def baseline(self, values, target, horizon):
        now = values[-1]
        if self.baselines.get(f"{target}@{horizon}") == "seasonal" and horizon <= self.season:
            back = self.season - horizon
            if len(values) > back and np.isfinite(values[-1 - back]):
                return float(values[-1 - back])
        return float(now)

    def forecast(self, history):
        out = {}
        for target in self.targets:
            values = history[target].to_numpy(dtype=float)
            out[target] = [self.baseline(values, target, h) for h in self.horizons]
        return out
