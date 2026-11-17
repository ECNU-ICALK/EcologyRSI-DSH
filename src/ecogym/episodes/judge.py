"""The judge: run a submitted predictor on a stage it has never seen, then score it.

For stage S of a single-episode task the predictor is fitted on every row of
the episode before S starts, then receives S one time step at a time (rows
are dense; missing values are nulls). After each row of S it is asked for its
forecasts, regardless of whether that origin has the future labels for scoring.
Scores are the bench's own: per-cell skill against the fit-selected naive
baseline on S's eligible origins, -1 for missing forecasts, forecasts clipped
to the targets' physical bounds, a day-block bootstrap interval.

The predictor runs in the program sandbox (no network, reads only its own
directory); hidden values exist only in this process and the pipe.

Every run gets a fresh copy of the program (``run_predictor``): what a run
writes into its directory never reaches the stored program, so a program
judged again later (a sealed check, a re-judged submission) is byte for byte
the one its digest names. A watchdog holds the run to its total time and to
``cancel`` even where the host blocks writing to a program that stopped
reading its input: it kills the program's whole process group.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.gates import objective_interval
from ..evaluation.harness import Bench, Evaluation, StageData
from ..protocol.views import SealedAccess
from ..solvers.base import Genome
from .agent import Cancelled
from .sandbox import Runtime, confine, program_profile, scratch_root, thread_limits

JUDGE_PROTOCOL = "causal_stream/2"
RUNNER = "_ecogym_runner.py"
WRITE_GRACE = 5.0  # seconds past the total deadline before the watchdog kills a program the host cannot reach
CLOSE_SECONDS = 10.0  # a finished run's program gets this long to exit after "end"
REPLY_LIMIT = 1_000_000  # bytes of one reply line; longer ones are an error
REPLY_QUEUE = 64  # replies read ahead of the judge; a program that floods its output waits
# A judged program's numeric threads: fixed, so no score depends on how hot the machine may run. One: these
# programs fit small models, and with two a template's judging took 1.4–1.8 times as long and 3.6–4.6 times the
# CPU (waiting OpenMP threads spin), for the same score to the last digit (2026-11-05, the 2fb9cb template, two
# tasks judged twice each way).
JUDGE_THREADS = 1
EPOCH = np.datetime64("1899-12-30T00:00")
STEP_UNITS = {"1h": "h", "1d": "D"}


@dataclass(frozen=True, slots=True)
class Limits:
    fit_seconds: float = 300.0
    forecast_seconds: float = 30.0  # one forecast
    total_seconds: float = 900.0  # fit plus the whole stream
    max_error_share: float = 0.2


@dataclass(frozen=True, slots=True, eq=False)
class Stream:
    stage: str
    columns: tuple[str, ...]
    fit_time: np.ndarray
    fit_values: np.ndarray
    time: np.ndarray
    values: np.ndarray
    ask: np.ndarray  # stream positions scored by the host, never sent to the predictor


def episode_of(bench: Bench) -> str:
    if len(bench.task.episodes) != 1:
        raise ValueError(f"{bench.task.id}: the judge supports single-episode tasks")
    return bench.task.episodes[0]


def visible_columns(bench: Bench) -> tuple[str, ...]:
    """Targets first, then every column whose role the task admits as an input."""

    task, frame = bench.task, bench.frame
    inputs = sorted(name for name, spec in frame.specs.items()
                    if spec.role in task.exogenous_roles and name not in task.target_names)
    return (*task.target_names, *inputs)


def timestamps(steps: np.ndarray, step: str) -> np.ndarray:
    return EPOCH + steps.astype(np.int64).astype(f"timedelta64[{STEP_UNITS[step]}]")


def iso(steps: np.ndarray, step: str) -> list[str]:
    return [str(item) for item in np.datetime_as_string(timestamps(steps, step), unit="m")]


def predictor_spec(bench: Bench) -> dict[str, Any]:
    task, frame = bench.task, bench.frame
    return {
        "task": task.id,
        "title": task.title,
        "targets": list(task.target_names),
        "horizons": list(task.horizons),
        "step": frame.step,
        "season_steps": task.season_steps,
        "history_steps": task.history_steps,
        "baselines": dict(bench.baselines.kinds),
        "bounds": {target.name: [target.lo, target.hi] for target in task.targets},
        "columns": [{"name": name, "role": frame.specs[name].role, "unit": frame.specs[name].unit,
                     "label": frame.specs[name].label, "target": name in task.target_names}
                    for name in visible_columns(bench)],
    }


def stage_stream(bench: Bench, stage: str, *, sealed: SealedAccess | None = None) -> tuple[Stream, StageData]:
    data = bench.stage(stage, sealed=sealed)  # raises for a sealed stage without access
    episode = episode_of(bench)
    start, end = bench.partitions.intervals[stage][episode]
    rows = bench.frame.episode_rows(episode)
    times = bench.frame.time[rows]
    columns = visible_columns(bench)
    values = np.column_stack([bench.frame.columns[name][rows] for name in columns])
    before = times < start
    inside = (times >= start) & (times < end)
    stream_time = times[inside]
    ask = np.searchsorted(stream_time, data.time)
    if ask.size and (ask.max() >= stream_time.size or np.any(stream_time[ask] != data.time)):
        raise ValueError(f"{stage}: eligible origins are not rows of the stage")
    return Stream(stage, columns, times[before], values[before], stream_time, values[inside], ask), data


def _rows(values: np.ndarray) -> list[list[float | None]]:
    return [[float(item) if np.isfinite(item) else None for item in row] for row in values]


@dataclass(slots=True)
class Run:
    predicted: np.ndarray  # (targets, horizons, origins)
    status: str
    errors: list[str]
    fit_seconds: float | None
    seconds: float
    stderr: str


class _Program:
    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str], stderr: Path) -> None:
        self._stderr = stderr.open("wb")
        self.process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self._stderr, start_new_session=True)
        self.replies: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=REPLY_QUEUE)
        self._closed = threading.Event()
        threading.Thread(target=self._read, daemon=True, name="judge-reader").start()

    def _put(self, item: dict[str, Any] | None) -> bool:
        """Queue one reply, waiting while the queue is full; False once the run is over (nobody reads it)."""

        while not self._closed.is_set():
            try:
                self.replies.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _read(self) -> None:
        assert self.process.stdout is not None
        skipping = False  # inside a line longer than REPLY_LIMIT
        while True:
            try:
                chunk = self.process.stdout.readline(REPLY_LIMIT)
            except (OSError, ValueError):  # the pipe was closed under a reader that a leftover process kept waiting
                return
            if not chunk:
                break
            complete = chunk.endswith(b"\n")
            if skipping or (not complete and len(chunk) >= REPLY_LIMIT):
                skipping = not complete
                if not skipping and not self._put({"op": "error", "message": "an oversized reply from the runner"}):
                    return
                continue
            try:
                item = json.loads(chunk)
            except ValueError:  # not JSON, or not UTF-8
                item = None
            if not isinstance(item, dict):
                item = {"op": "error", "message": "unreadable reply from the runner"}
            if not self._put(item):
                return
        self._put(None)

    def kill(self) -> None:
        """Kill the program and whatever it started in its process group (it runs in a session of its own)."""

        with contextlib.suppress(OSError):
            os.killpg(self.process.pid, signal.SIGKILL)
        with contextlib.suppress(OSError):
            self.process.kill()

    def send(self, message: dict[str, Any]) -> bool:
        assert self.process.stdin is not None
        try:
            self.process.stdin.write((json.dumps(message, allow_nan=False) + "\n").encode())
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def reply(self, timeout: float, cancel: threading.Event | None = None) -> dict[str, Any] | None:
        """The next reply, ``{"op": "timeout"}`` after ``timeout`` or ``{"op": "cancelled"}`` once ``cancel`` is set."""

        end = time.monotonic() + max(timeout, 0.01)
        while True:
            try:
                return self.replies.get(timeout=max(0.01, min(end - time.monotonic(), 0.5)))
            except queue.Empty:
                if cancel is not None and cancel.is_set():
                    return {"op": "cancelled"}
                if time.monotonic() >= end:
                    return {"op": "timeout"}

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.send({"op": "end"})
                self.process.wait(timeout=CLOSE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        finally:
            if self.process.poll() is None:
                self.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=CLOSE_SECONDS)
            self._closed.set()
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe is not None:
                    with contextlib.suppress(OSError):
                        pipe.close()
            self._stderr.close()


def _watch(program: _Program, limit: list[float], cancel: threading.Event | None, stopped: list[str],
           over: threading.Event) -> None:
    """Kill the program once ``cancel`` is set or the clock passes ``limit[0]``, whatever the judge is doing: a
    write to a program that does not read its input blocks, and only the program's death ends that write."""

    while not over.wait(0.2):
        if cancel is not None and cancel.is_set():
            stopped.append("cancelled")
        elif time.monotonic() > limit[0]:
            stopped.append("timeout")
        else:
            continue
        program.kill()
        return


def _fresh_copy(source: Path, target: Path) -> None:
    """Copy a program directory's regular files and folders (never links), so a run cannot change the original."""

    target.mkdir(parents=True)
    for directory, names, files in os.walk(source, followlinks=False):
        here = Path(directory)
        relative = here.relative_to(source)
        for name in list(names):
            if (here / name).is_symlink():
                names.remove(name)
            else:
                (target / relative / name).mkdir()
        for name in files:
            path = here / name
            if path.is_file() and not path.is_symlink():
                shutil.copyfile(path, target / relative / name)


def run_predictor(program_dir: Path, stream: Stream, spec: dict[str, Any], runtime: Runtime, *,
                  limits: Limits | None = None, step: str = "1h", cancel: threading.Event | None = None) -> Run:
    """Fit and stream one predictor; forecasts for origins it misses are NaN.

    The predictor runs in a fresh copy of ``program_dir``, removed afterwards; ``program_dir`` itself is never
    written. Setting ``cancel`` kills the predictor within a second and raises ``Cancelled``: a run cut short is
    never scored. The run is killed ``WRITE_GRACE`` seconds after its total time even where the host blocks on it.
    """

    limits = limits or Limits()
    work = Path(tempfile.mkdtemp(prefix="judge-", dir=scratch_root()))
    try:
        run_dir = work / "program"
        _fresh_copy(Path(program_dir).resolve(), run_dir)
        return _run_in(run_dir, work / "program.sb", stream, spec, runtime, limits=limits, step=step, cancel=cancel)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_in(run_dir: Path, profile: Path, stream: Stream, spec: dict[str, Any], runtime: Runtime, *,
            limits: Limits, step: str, cancel: threading.Event | None) -> Run:
    shutil.copyfile(resources.files("ecogym.episodes").joinpath("predictor_runner.py"), run_dir / RUNNER)
    (run_dir / "tmp").mkdir(exist_ok=True)
    # The allowlisted environment below sets PYTHONHASHSEED. -I would ignore it. Keep -I's safe import path
    # and disabled user site explicitly, without inheriting any other PYTHON* environment variables.
    argv = confine(program_profile(run_dir, runtime), profile, [str(runtime.python), "-P", "-s", "-B", RUNNER])
    env = {"PATH": "/usr/bin:/bin", "HOME": str(run_dir), "TMPDIR": str(run_dir / "tmp"), "LANG": "en_US.UTF-8",
           "PYTHONHASHSEED": "0", **thread_limits(JUDGE_THREADS)}
    targets, horizons = spec["targets"], spec["horizons"]
    predicted = np.full((len(targets), len(horizons), stream.ask.size), np.nan)
    errors: list[str] = []
    started = time.monotonic()
    deadline = started + limits.total_seconds
    stderr = profile.with_suffix(".stderr")
    program = _Program(argv, run_dir, env, stderr)
    limit, stopped, over = [deadline + WRITE_GRACE], [], threading.Event()
    threading.Thread(target=_watch, args=(program, limit, cancel, stopped, over), daemon=True,
                     name="judge-watchdog").start()
    status, fit_seconds = "ok", None

    def stop_if_cancelled() -> None:
        nonlocal status
        if cancel is not None and cancel.is_set():
            status = "cancelled"
            raise Cancelled("judging was cancelled")

    try:
        program.send({"op": "start", "spec": spec, "columns": list(stream.columns), "time": iso(stream.fit_time, step),
                      "rows": _rows(stream.fit_values), "stream": int(stream.time.size)})
        reply = program.reply(min(limits.fit_seconds, deadline - time.monotonic()), cancel)
        stop_if_cancelled()
        if not reply or reply.get("op") != "ready":
            status = "fit_failed" if reply and reply.get("op") == "error" else "fit_timeout" if reply else "crashed"
            if reply and reply.get("op") == "error":
                errors.append(str(reply.get("message")))
        else:
            fit_seconds = float(reply.get("seconds", 0.0))
            asked = {int(position): index for index, position in enumerate(stream.ask)}
            stamps = iso(stream.time, step)
            rows = _rows(stream.values)
            for position, row in enumerate(rows):
                stop_if_cancelled()
                if not program.send({"op": "row", "time": stamps[position], "values": row}):
                    status = "crashed"
                    break
                program.send({"op": "forecast"})
                reply = program.reply(min(limits.forecast_seconds, deadline - time.monotonic()), cancel)
                if reply is None:
                    status = "crashed"
                    break
                if reply.get("op") == "timeout":
                    status = "timeout"
                    break
                if reply.get("op") == "error":
                    errors.append(str(reply.get("message")))
                    if len(errors) > limits.max_error_share * stream.time.size:
                        status = "too_many_errors"
                        break
                    continue
                if position not in asked:
                    continue
                values = reply.get("values") if isinstance(reply.get("values"), dict) else {}
                for t, target in enumerate(targets):
                    found = values.get(target)
                    for h, value in enumerate(found[: len(horizons)] if isinstance(found, list) else []):
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            predicted[t, h, asked[position]] = float(value)
            stop_if_cancelled()  # a cancel during the last forecast leaves it missing: not a result either
    finally:
        if program.process.poll() is None and status in ("timeout", "too_many_errors", "fit_timeout", "cancelled"):
            program.kill()
        limit[0] = min(limit[0], time.monotonic() + CLOSE_SECONDS + WRITE_GRACE)  # closing is bounded too
        program.close()
        over.set()
    if stopped and status not in ("ok", "cancelled"):  # the watchdog ended it: say why, not "crashed"
        status = "cancelled" if stopped[0] == "cancelled" else "fit_timeout" if fit_seconds is None else "timeout"
    if status == "cancelled":
        raise Cancelled("judging was cancelled")
    return Run(predicted, status, errors[:20], fit_seconds, time.monotonic() - started, _tail(stderr))


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            return stream.read(limit).decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


SUBMISSION = "submission@1"


def score(bench: Bench, data: StageData, run: Run, label: str) -> Evaluation:
    """An ``Evaluation`` of the run's forecasts, clipped like any solver's."""

    task = bench.task
    rows, clipped = [], []
    for cell in data.cells:
        values = run.predicted[task.target_names.index(cell.target), task.horizons.index(cell.horizon)]
        target = task.target(cell.target)
        lo = -np.inf if target.lo is None else target.lo
        hi = np.inf if target.hi is None else target.hi
        bounded = np.clip(values, lo, hi)
        clipped.append(int(np.sum(np.isfinite(values) & (bounded != values))))
        rows.append(bounded)
    return Evaluation(Genome(SUBMISSION, {"label": label}), data, np.stack(rows), np.array(clipped))


def missing(data: StageData, label: str = "missing") -> Evaluation:
    """The evaluation of no submission at all: every forecast missing (each cell scores -1)."""

    shape = (len(data.cells), data.size)
    return Evaluation(Genome(SUBMISSION, {"label": label}), data, np.full(shape, np.nan),
                      np.zeros(len(data.cells), dtype=np.int64))


def average(evaluations: Sequence[Evaluation], label: str = "average") -> Evaluation:
    """Forecasts averaged over evaluations of one stage; an origin a member missed takes the others' mean."""

    data = evaluations[0].stage
    keys = [cell.key for cell in data.cells]
    if any(item.stage.name != data.name or not np.array_equal(item.stage.time, data.time)
           or [cell.key for cell in item.stage.cells] != keys for item in evaluations):
        raise ValueError("only evaluations of the same stage, origins and cells can be averaged")
    stack = np.stack([item.predicted for item in evaluations])
    count = np.isfinite(stack).sum(axis=0)
    total = np.where(np.isfinite(stack), stack, 0.0).sum(axis=0)
    return Evaluation(Genome(SUBMISSION, {"label": label, "members": len(evaluations)}), data,
                      np.where(count > 0, total / np.maximum(count, 1), np.nan),
                      np.zeros(len(data.cells), dtype=np.int64))


def report_of(bench: Bench, evaluation: Evaluation, run: Run | None) -> dict[str, Any]:
    stage = evaluation.stage.name
    low, high = objective_interval(evaluation, None, bench.task.gates, {"judge": stage})
    return {
        "stage": stage,
        "judge_protocol": JUDGE_PROTOCOL,
        "status": run.status if run is not None else "no_program",
        "origins": int(evaluation.stage.size),
        "score": evaluation.score(),
        "interval": [low, high],
        "coverage": float(np.isfinite(evaluation.predicted).mean()),
        "cells": evaluation.cells(),
        "fit_seconds": run.fit_seconds if run is not None else None,
        "seconds": round(run.seconds, 3) if run is not None else 0.0,
        "errors": run.errors if run is not None else [],
        "stderr": run.stderr if run is not None else "",
    }


def judge_program(bench: Bench, program_dir: Path, stage: str, runtime: Runtime, *, limits: Limits | None = None,
                  sealed: SealedAccess | None = None,
                  cancel: threading.Event | None = None) -> tuple[dict[str, Any], Evaluation]:
    """Fit and stream one program on ``stage``; the report and the evaluation behind it."""

    limits = limits or Limits()
    stream, data = stage_stream(bench, stage, sealed=sealed)
    run = run_predictor(program_dir, stream, predictor_spec(bench), runtime, limits=limits, step=bench.frame.step,
                        cancel=cancel)
    evaluation = score(bench, data, run, str(program_dir))
    return report_of(bench, evaluation, run), evaluation


def judge(bench: Bench, program_dir: Path, stage: str, runtime: Runtime, *, limits: Limits | None = None,
          sealed: SealedAccess | None = None, references: bool = True,
          cancel: threading.Event | None = None) -> dict[str, Any]:
    report, _ = judge_program(bench, program_dir, stage, runtime, limits=limits, sealed=sealed, cancel=cancel)
    if references and sealed is None:
        report["references"] = reference_scores(bench, stage)
    return report


def save_predictions(path: Path, bench: Bench, evaluation: Evaluation) -> Path:
    """Store a judged program's forecasts, so comparisons can be recomputed from the record."""

    stage = evaluation.stage
    with path.open("wb") as stream:
        np.savez_compressed(stream, predicted=evaluation.predicted, clipped=evaluation.clipped, time=stage.time,
                            cells=np.array([cell.key for cell in stage.cells]), stage=np.array(stage.name),
                            bench=np.array(bench.digest))
    return path


def file_digest(path: Path) -> str:
    """sha256 of a stored predictions file, recorded next to its path so a changed file is noticed."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_predictions(path: Path, bench: Bench, data: StageData | None = None, *,
                     expect: str | None = None) -> Evaluation:
    """An evaluation from stored forecasts; ``data`` must be given for a sealed stage.

    ``expect`` is the file's recorded digest (``file_digest``); a file that no longer matches is refused.
    """

    if expect and file_digest(path) != expect:
        raise ValueError(f"{path} changed after it was recorded")
    with np.load(path, allow_pickle=False) as stored:
        if str(stored["bench"]) != bench.digest:
            raise ValueError(f"{path} was judged on another bench (data, task or partitions changed)")
        data = data if data is not None else bench.stage(str(stored["stage"]))
        if (str(stored["stage"]) != data.name or not np.array_equal(stored["time"], data.time)
                or [str(key) for key in stored["cells"]] != [cell.key for cell in data.cells]):
            raise ValueError(f"{path} does not match the scored origins of {data.name}")
        return Evaluation(Genome(SUBMISSION, {"predictions": str(path)}), data, stored["predicted"].copy(),
                          stored["clipped"].copy())


def reference_scores(bench: Bench, stage: str) -> dict[str, float]:
    """Built-in methods on the same stage (fitted on calibration_fit only), naive baseline = 0."""

    scores = {"naive": 0.0}
    for name in sorted(bench.task.presets):
        scores[name] = bench.evaluate(bench.preset(name), stage).score()
    return scores


__all__ = ["JUDGE_PROTOCOL", "SUBMISSION", "Limits", "Run", "Stream", "average", "file_digest", "judge",
           "judge_program", "load_predictions", "missing", "predictor_spec", "reference_scores", "report_of",
           "run_predictor", "save_predictions", "score", "stage_stream", "timestamps", "visible_columns"]
