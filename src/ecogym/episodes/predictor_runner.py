"""Runs a submitted predictor for the judge; executed inside the program sandbox.

This file is copied next to the submission and run by the agent runtime's
Python (stdlib, numpy and pandas only; it never imports ecogym). It speaks
JSON lines on stdin/stdout:

    -> {"op": "start", "spec": {...}, "columns": [...], "time": [...], "rows": [[...], ...], "stream": N}
    <- {"op": "ready", "seconds": s}                     fit finished
    -> {"op": "row", "time": "...", "values": [...]}     one new observation (no reply)
    -> {"op": "forecast"}                                 forecast from the last row sent
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


class History:
    """Rows seen so far, as a time-indexed DataFrame view of a preallocated array."""

    def __init__(self, columns, times, rows, capacity):
        self.columns = list(columns)
        self.values = np.full((len(times) + capacity, len(self.columns)), np.nan)
        if rows:
            block = np.array([[np.nan if item is None else item for item in row] for row in rows], dtype=float)
            self.values[: len(rows)] = block
        self.times = list(pd.to_datetime(times)) + [None] * capacity
        self.size = len(times)

    def append(self, when, values):
        self.values[self.size] = [np.nan if item is None else item for item in values]
        self.times[self.size] = pd.Timestamp(when)
        self.size += 1

    def frame(self):
        index = pd.DatetimeIndex(self.times[: self.size], name="time")
        return pd.DataFrame(self.values[: self.size].copy(), index=index, columns=self.columns)


def main():
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(0)
    np.random.seed(0)
    sys.path.insert(0, os.getcwd())
    history = None
    predictor = None
    spec = None
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
                predictor.fit(history.frame())
                send({"op": "ready", "seconds": round(time.monotonic() - started, 3)})
            except Exception as error:  # every failure is reported to the judge, never raised
                send(failure(error))
                return
        elif op == "row":
            history.append(message["time"], message["values"])
        elif op == "forecast":
            started = time.monotonic()
            try:
                result = predictor.forecast(history.frame())
                values = {}
                width = len(spec["horizons"])
                for target in spec["targets"]:
                    items = list(result.get(target, [])) if isinstance(result, dict) else []
                    values[target] = [number(items[i]) if i < len(items) else None for i in range(width)]
                send({"op": "forecast", "values": values, "seconds": round(time.monotonic() - started, 4)})
            except Exception as error:  # every failure is reported to the judge, never raised
                send(failure(error))
        elif op == "end":
            return


if __name__ == "__main__":
    main()
