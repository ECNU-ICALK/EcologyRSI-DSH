"""Runs a submitted predictor for the judge; executed inside the program sandbox.

This file is copied next to the submission and run by the agent runtime's
Python (stdlib, numpy and pandas only; it never imports ecogym). It speaks
JSON lines on stdin/stdout:

    -> {"op": "start", "spec": {...}, "columns": [...], "time": [...], "rows": [[...], ...], "stream": N}
    <- {"op": "ready", "seconds": s}                     fit finished
    -> {"op": "step", "time": "...", "values": [...]}    append one observation and forecast (single IPC)
    <- {"op": "forecast", "values": {target: [v_h, ...]}, "seconds": s}
    -> {"op": "row", "time": "...", "values": [...]}     legacy two-message form (still supported)
    -> {"op": "forecast"}
    <- {"op": "forecast", "values": {target: [v_h, ...]}, "seconds": s}
    -> {"op": "end"}

Any failure is answered with {"op": "error", "message": ..., "trace": ...}.
The predictor's own prints go to stderr so they cannot corrupt the protocol.
"""

import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd

PROTOCOL = sys.stdout
sys.stdout = sys.stderr


def send(message):
    PROTOCOL.write(json.dumps(message, allow_nan=False) + "\n")
    PROTOCOL.flush()


def failure(error):
    return {"op": "error", "message": f"{type(error).__name__}: {error}"[:2000],
            "trace": traceback.format_exc()[-4000:]}


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _to_ns(when):
    if isinstance(when, str) and "Z" not in when and "+" not in when and "-" not in when.partition("T")[2]:
        try:
            return np.datetime64(when, "ns")
        except ValueError:
            pass
    ts = pd.Timestamp(when)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.asm8


class History:
    """Rows seen so far, as a time-indexed DataFrame backed by preallocated NumPy buffers."""

    def __init__(self, columns, times, rows, capacity):
        self.columns = list(columns)
        total = len(times) + capacity
        self.values = np.full((total, len(self.columns)), np.nan, dtype=float)
        if rows:
            block = np.array([[np.nan if item is None else item for item in row] for row in rows], dtype=float)
            self.values[: len(rows)] = block
        self.times = np.empty(total, dtype="datetime64[ns]")
        self.tz = None
        if times:
            parsed = pd.to_datetime(times)
            if parsed.tz is not None:
                self.tz = parsed.tz
                parsed = parsed.tz_convert("UTC").tz_localize(None)
            self.times[: len(times)] = parsed.to_numpy(dtype="datetime64[ns]")
        self.size = len(times)
        self.last_time = times[-1] if times else None
        self.last_values = None

    def append(self, when, values):
        row = [np.nan if item is None else float(item) for item in values]
        self.values[self.size] = row
        self.times[self.size] = _to_ns(when)
        self.size += 1
        self.last_time = when
        self.last_values = row

    def frame(self):
        # Contiguous datetime64[ns] avoids O(N) Python Timestamp conversion on every streaming step.
        index = pd.DatetimeIndex(self.times[: self.size], name="time")
        if self.tz is not None:
            index = index.tz_localize("UTC").tz_convert(self.tz)
        return pd.DataFrame(self.values[: self.size].copy(), index=index, columns=self.columns)


def _run_forecast(predictor, history, spec, *, use_step=False):
    started = time.monotonic()
    try:
        if use_step and history.last_values is not None:
            row_map = dict(zip(history.columns, history.last_values, strict=True))
            result = predictor.forecast_step(pd.Timestamp(history.last_time), row_map)
        else:
            result = predictor.forecast(history.frame())
        values = {}
        width = len(spec["horizons"])
        for target in spec["targets"]:
            items = list(result.get(target, [])) if isinstance(result, dict) else []
            values[target] = [number(items[i]) if i < len(items) else None for i in range(width)]
        send({"op": "forecast", "values": values, "seconds": round(time.monotonic() - started, 4)})
    except Exception as error:  # every failure is reported to the judge, never raised
        send(failure(error))


def main():
    random.seed(0)
    np.random.seed(0)
    sys.path.insert(0, os.getcwd())
    history = None
    predictor = None
    spec = None
    has_step = False
    for line in sys.stdin:
        message = json.loads(line)
        op = message.get("op")
        if op == "start":
            started = time.monotonic()
            try:
                spec = message["spec"]
                loaded = importlib.util.spec_from_file_location("predictor", os.path.join(os.getcwd(), "predictor.py"))
                module = importlib.util.module_from_spec(loaded)
                loaded.loader.exec_module(module)
                history = History(message["columns"], message["time"], message["rows"], int(message["stream"]))
                predictor = module.Predictor(spec)
                has_step = callable(getattr(predictor, "forecast_step", None))
                predictor.fit(history.frame())
                send({"op": "ready", "seconds": round(time.monotonic() - started, 3)})
            except Exception as error:  # every failure is reported to the judge, never raised
                send(failure(error))
                return
        elif op == "step":
            history.append(message["time"], message["values"])
            _run_forecast(predictor, history, spec, use_step=has_step)
        elif op == "row":
            history.append(message["time"], message["values"])
        elif op == "forecast":
            _run_forecast(predictor, history, spec, use_step=has_step)
        elif op == "end":
            return


if __name__ == "__main__":
    main()
