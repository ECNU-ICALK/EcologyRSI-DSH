"""Causal features on a time-indexed history (every value uses rows at or before its own row)."""

import numpy as np
import pandas as pd


def lag(series: pd.Series, k: int) -> pd.Series:
    """The value ``k`` steps before each row (k >= 0)."""

    if k < 0:
        raise ValueError("lags look back only")
    return series.shift(k)


def seasonal(series: pd.Series, period: int, horizon: int, cycles: int = 1) -> pd.Series:
    """y(t + horizon - cycles * period): the same phase as the target time, ``cycles`` periods back."""

    back = cycles * period - horizon
    if back < 0:
        raise ValueError("cycles * period must be at least the horizon")
    return series.shift(back)


def profile(series: pd.Series, period: int, horizon: int, cycles=(1, 2)) -> pd.Series:
    """Mean of ``seasonal`` over several cycles (a diurnal profile for hourly data)."""

    parts = [seasonal(series, period, horizon, cycle) for cycle in cycles]
    return pd.concat(parts, axis=1).mean(axis=1)


def time_of_day(index: pd.DatetimeIndex, period_hours: float = 24.0) -> pd.DataFrame:
    hours = index.hour + index.minute / 60.0
    angle = 2 * np.pi * hours / period_hours
    return pd.DataFrame({"tod_sin": np.sin(angle), "tod_cos": np.cos(angle)}, index=index)


def direct_target(series: pd.Series, horizon: int) -> pd.Series:
    """y(t + horizon) aligned to origin t, for building training sets (NaN in the last rows)."""

    return series.shift(-horizon)
