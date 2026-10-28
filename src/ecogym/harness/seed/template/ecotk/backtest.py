"""Replay the judge's protocol in your own process, for quick experiments on visible data.

The official local score comes from the ``evaluate_local`` tool; this is for fast
iteration over several windows and many variants. Same rules as the judge: fit
on the rows before ``start``, then forecast every origin from the rows up to and
including it; skill per cell = (RMSE_baseline - RMSE) / max(both).

    import json, pandas as pd
    from predictor import Predictor
    from ecotk import backtest
    spec = json.load(open("../data/spec.json"))
    history = pd.read_csv("../data/history.csv", index_col="time", parse_dates=True)
    print(backtest.rolling(Predictor, spec, history, count=3, days=5))
"""

import numpy as np
import pandas as pd


def _baseline(values, position, spec, target, horizon):
    now = values[position]
    season = spec["season_steps"]
    if spec["baselines"].get(f"{target}@{horizon}") == "seasonal" and horizon <= season:
        back = position - (season - horizon)
        if back >= 0 and np.isfinite(values[back]):
            return values[back]
    return now


def run(predictor_class, spec, history: pd.DataFrame, start, end) -> dict:
    """Score one window [start, end) of ``history``."""

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    fit_part = history[history.index < start]
    window = np.flatnonzero((history.index >= start) & (history.index < end))
    predictor = predictor_class(spec)
    predictor.fit(fit_part)
    horizons, targets = spec["horizons"], spec["targets"]
    errors = {(t, h): [] for t in targets for h in horizons}
    base = {(t, h): [] for t in targets for h in horizons}
    columns = {t: history[t].to_numpy(dtype=float) for t in targets}
    last = window[-1] if window.size else -1
    for position in window:
        if position + max(horizons) > last:
            break
        if not all(np.isfinite(columns[t][position]) for t in targets):
            continue
        forecast = predictor.forecast(history.iloc[: position + 1])
        for t in targets:
            for i, h in enumerate(horizons):
                truth = columns[t][position + h]
                value = forecast.get(t, [np.nan] * len(horizons))[i]
                if np.isfinite(truth) and value is not None and np.isfinite(value):
                    errors[(t, h)].append(value - truth)
                    base[(t, h)].append(_baseline(columns[t], position, spec, t, h) - truth)
    cells = {}
    for (t, h), values in errors.items():
        if not values:
            cells[f"{t}@{h}"] = None
            continue
        rmse = float(np.sqrt(np.mean(np.square(values))))
        base_rmse = float(np.sqrt(np.mean(np.square(base[(t, h)]))))
        denominator = max(rmse, base_rmse)
        cells[f"{t}@{h}"] = {"n": len(values), "rmse": rmse, "baseline_rmse": base_rmse,
                             "skill": 0.0 if denominator == 0 else (base_rmse - rmse) / denominator}
    scored = [cell["skill"] for cell in cells.values() if cell]
    return {"start": str(start), "end": str(end), "score": float(np.mean(scored)) if scored else None,
            "cells": cells}


def windows(history: pd.DataFrame, count: int = 3, days: float = 5.0):
    """``count`` consecutive windows of ``days`` at the end of ``history``."""

    end = history.index[-1] + pd.Timedelta(hours=1)
    length = pd.Timedelta(days=days)
    return [(end - length * (count - i), end - length * (count - i - 1)) for i in range(count)]


def rolling(predictor_class, spec, history: pd.DataFrame, count: int = 3, days: float = 5.0) -> dict:
    results = [run(predictor_class, spec, history, start, end) for start, end in windows(history, count, days)]
    scores = [item["score"] for item in results if item["score"] is not None]
    return {"mean": float(np.mean(scores)) if scores else None, "windows": results}
