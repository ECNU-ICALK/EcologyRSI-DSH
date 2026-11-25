"""Replay the judge's protocol in your own process, for quick experiments on visible data.

The official local score comes from the ``evaluate_local`` tool; this is for fast
iteration over non-overlapping causal windows inside the fitting period (excluding
the ``evaluate_local`` check tail by default so you do not fit to the local check
twice). Same rules as the judge: fit on the rows before ``start``, stream every row
of ``[start, end)`` through ``forecast`` in order, and score eligible origins:
skill per cell = (RMSE_baseline - RMSE) / max(both), and a missing forecast counts
like the judge counts it (coverage q: objective = q * skill - (1 - q)).

    import json, pandas as pd
    from predictor import Predictor
    from ecotk import backtest
    spec = json.load(open("../data/spec.json"))
    history = pd.read_csv("../data/history.csv", index_col="time", parse_dates=True)
    print(backtest.rolling(Predictor, spec, history, count=3, days=5))
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

_FIT_END_RE = re.compile(r"拟合用\s+([0-9T:-]+)\s+之前的行")


def _baseline(values, position, spec, target, horizon):
    now = values[position]
    season = spec["season_steps"]
    if spec["baselines"].get(f"{target}@{horizon}") == "seasonal" and horizon <= season:
        back = position - (season - horizon)
        if back >= 0 and np.isfinite(values[back]):
            return values[back]
    return now


def _detect_fit_end() -> pd.Timestamp | None:
    for candidate in (Path("data/README.md"), Path("../data/README.md")):
        if candidate.is_file():
            match = _FIT_END_RE.search(candidate.read_text(encoding="utf-8", errors="ignore"))
            if match:
                return pd.Timestamp(match.group(1))
    return None


def run(predictor_class, spec, history: pd.DataFrame, start, end) -> dict:
    """Score one causal window [start, end) of ``history`` using the streaming protocol."""

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    fit_part = history[history.index < start]
    window = np.flatnonzero((history.index >= start) & (history.index < end))
    predictor = predictor_class(spec)
    predictor.fit(fit_part)
    horizons, targets = spec["horizons"], spec["targets"]
    history_steps = max(1, int(spec.get("history_steps", 1)))
    errors = {(t, h): [] for t in targets for h in horizons}
    base = {(t, h): [] for t in targets for h in horizons}
    asked = {(t, h): 0 for t in targets for h in horizons}
    columns = {t: history[t].to_numpy(dtype=float) for t in targets}
    last = window[-1] if window.size else -1
    for position in window:
        forecast = predictor.forecast(history.iloc[: position + 1])
        if position + max(horizons) > last or position + 1 < history_steps:
            continue
        if not all(np.all(np.isfinite(columns[t][position - history_steps + 1 : position + 1])) for t in targets):
            continue
        if not all(np.isfinite(columns[t][position + h]) for t in targets for h in horizons):
            continue
        for t in targets:
            values = list(forecast.get(t, [])) if isinstance(forecast, dict) else []
            for i, h in enumerate(horizons):
                truth = columns[t][position + h]
                asked[(t, h)] += 1
                value = values[i] if i < len(values) else None
                if value is not None and np.isfinite(value):
                    errors[(t, h)].append(value - truth)
                    base[(t, h)].append(_baseline(columns[t], position, spec, t, h) - truth)
    cells = {}
    by_horizon: dict[int, list[float]] = {h: [] for h in horizons}
    for (t, h), values in errors.items():
        if not asked[(t, h)]:
            cells[f"{t}@{h}"] = None
            continue
        coverage = len(values) / asked[(t, h)]
        if not values:
            obj = -1.0
            cells[f"{t}@{h}"] = {"n": 0, "coverage": 0.0, "skill": None, "objective": obj}
            by_horizon[h].append(obj)
            continue
        rmse = float(np.sqrt(np.mean(np.square(values))))
        base_rmse = float(np.sqrt(np.mean(np.square(base[(t, h)]))))
        denominator = max(rmse, base_rmse)
        skill = 0.0 if denominator == 0 else (base_rmse - rmse) / denominator
        obj = coverage * skill - (1.0 - coverage)
        cells[f"{t}@{h}"] = {
            "n": len(values),
            "coverage": coverage,
            "rmse": rmse,
            "baseline_rmse": base_rmse,
            "skill": skill,
            "objective": obj,
        }
        by_horizon[h].append(obj)
    scored = [cell["objective"] for cell in cells.values() if cell]
    horizons_summary = {str(h): float(np.mean(vals)) for h, vals in by_horizon.items() if vals}
    return {
        "start": str(start),
        "end": str(end),
        "score": float(np.mean(scored)) if scored else None,
        "horizons": horizons_summary,
        "cells": cells,
    }


def windows(
    history: pd.DataFrame,
    count: int = 3,
    days: float = 5.0,
    step: str | None = None,
    *,
    before=None,
    fit_only: bool = True,
):
    """``count`` consecutive causal windows of ``days`` before the local check period when ``fit_only`` is true."""

    one = pd.Timedelta(step) if step else pd.Series(history.index).diff().median()
    full_end = history.index[-1] + one
    length = pd.Timedelta(days=days)
    min_train = max(length, pd.Timedelta(days=5))
    end = full_end
    if before is not None:
        end = min(full_end, pd.Timestamp(before))
    elif fit_only:
        detected = _detect_fit_end()
        if detected is not None and detected - length * count >= history.index[0] + min_train:
            end = min(full_end, detected)
        else:
            span = full_end - history.index[0]
            tail = max(length, span * 0.2)
            if full_end - tail - length * count >= history.index[0] + min_train:
                end = full_end - tail
    return [(end - length * (count - i), end - length * (count - i - 1)) for i in range(count)]


def rolling(
    predictor_class,
    spec,
    history: pd.DataFrame,
    count: int = 3,
    days: float = 5.0,
    *,
    before=None,
    fit_only: bool = True,
) -> dict:
    spans = windows(history, count, days, spec.get("step"), before=before, fit_only=fit_only)
    results = [run(predictor_class, spec, history, start, end) for start, end in spans]
    scores = [item["score"] for item in results if item["score"] is not None]
    horizons: dict[str, list[float]] = {}
    for item in results:
        for h, val in (item.get("horizons") or {}).items():
            horizons.setdefault(h, []).append(val)
    return {
        "mean": float(np.mean(scores)) if scores else None,
        "horizons": {h: float(np.mean(vals)) for h, vals in horizons.items()},
        "windows": results,
    }
