"""Time-forward partitions.

``four_stage`` cuts every episode into five stages separated by embargoes:

    calibration_fit | calibration_uq | model_selection | validation | final_test

Methods are fitted on calibration_fit and compared on model_selection;
calibration_uq is an untouched development check. validation and final_test
are sealed: only ``ecogym.arena.sealed`` may open them, once per method.

The cut rules reproduce ``time-forward-four-stage@2`` of v0.10.33 exactly.
A calendar computed over all optimization episodes of the dataset fixes the
training, development and test boundaries for every episode, so one team's
test dates are never another team's training dates. Within the training
range, calibration_fit takes the first ``calibration_percent`` of the
episode's observed rows. Each stage is stored as a half-open time interval
per episode.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.frame import Frame
from ..kernel.canonical import digest
from ..task.spec import TaskSpec

STAGES = ("calibration_fit", "calibration_uq", "model_selection", "validation", "final_test")
DEVELOPMENT_STAGES = ("calibration_fit", "calibration_uq", "model_selection")
SEALED_STAGES = frozenset({"validation", "final_test"})


class ProtocolError(ValueError):
    """The data cannot support the protocol (too short, too sparse)."""


@dataclass(frozen=True, slots=True)
class Partitions:
    task_id: str
    intervals: dict[str, dict[str, tuple[int, int]]]  # stage -> episode -> [start, end)
    rows: dict[str, dict[str, int]]  # observed rows per stage and episode
    calendar: dict[str, int]
    evidence: dict[str, dict[str, int]]
    digest: str

    def mask(self, frame: Frame, stage: str) -> np.ndarray:
        if stage not in self.intervals:
            raise KeyError(f"unknown stage {stage}")
        mask = np.zeros(len(frame), dtype=bool)
        for episode, (start, end) in self.intervals[stage].items():
            code = frame.episode_code(episode)
            mask |= (frame.episode == code) & (frame.time >= start) & (frame.time < end)
        return mask

    def summary(self) -> dict[str, Any]:
        return {
            stage: {
                episode: {"start": start, "end": end, "rows": self.rows[stage][episode]}
                for episode, (start, end) in by_episode.items()
            }
            for stage, by_episode in self.intervals.items()
        }


def _calendar(times: Sequence[np.ndarray], task: TaskSpec) -> dict[str, int]:
    protocol = task.protocol
    start = min(int(item[0]) for item in times)
    end = max(int(item[-1]) for item in times) + 1
    duration = end - start
    return {
        "start": start,
        "end": end,
        "train": start + int(duration * protocol.train_fraction),
        "fit": start + int(duration * protocol.train_fraction * (1 - protocol.feedback_fraction)),
        "development": start + int(duration * (protocol.train_fraction + protocol.development_fraction)),
    }


def _episode_ranges(
    ts: list[int], calendar: dict[str, int], task: TaskSpec
) -> tuple[dict[str, tuple[int, int]], dict[str, int]]:
    """Row ranges into one episode's observed times ``ts``."""

    protocol = task.protocol
    n = len(ts)
    train_end = bisect_left(ts, calendar["train"])
    development_end = bisect_left(ts, calendar["development"])
    fit_end = max(0, min(bisect_left(ts, calendar["fit"]), train_end))
    feedback = (min(bisect_left(ts, calendar["fit"] + protocol.feedback_gap), train_end), train_end)
    development = (min(bisect_left(ts, calendar["train"] + protocol.embargo), development_end), development_end)
    final = (min(bisect_left(ts, calendar["development"] + protocol.embargo), n), n)
    a, b = 0, fit_end
    cut = a + (b - a) * protocol.calibration_percent // 100
    if not a < cut < b:
        raise ProtocolError("calibration range is too short to split")
    uq_start = bisect_left(ts, ts[cut - 1] + protocol.embargo)
    if uq_start >= b:
        raise ProtocolError("calibration_uq is empty after the embargo")
    selection_start = max(feedback[0], bisect_left(ts, ts[b - 1] + protocol.embargo))
    if selection_start >= feedback[1]:
        raise ProtocolError("model_selection is empty after the embargo")
    ranges = {
        "calibration_fit": (a, cut),
        "calibration_uq": (uq_start, b),
        "model_selection": (selection_start, feedback[1]),
        "validation": development,
        "final_test": final,
    }
    block = task.block_steps
    reach = task.history_steps + task.max_horizon
    evidence = {
        "fit_origins": max(0, (cut - a) - reach + 1),
        "fit_blocks": len({ts[index] // block for index in range(min(cut, a + reach), cut)}),
        "uq_rows": b - uq_start,
        "uq_blocks": len({ts[index] // block for index in range(uq_start, b)}),
    }
    if evidence["fit_origins"] < protocol.min_fit_origins or evidence["fit_blocks"] < protocol.min_fit_blocks:
        raise ProtocolError(f"calibration_fit evidence is insufficient: {evidence}")
    if evidence["uq_rows"] < protocol.min_uq_rows or evidence["uq_blocks"] < protocol.min_uq_blocks:
        raise ProtocolError(f"calibration_uq evidence is insufficient: {evidence}")
    return ranges, evidence


def four_stage(frame: Frame, task: TaskSpec, optimization_episodes: Sequence[str]) -> Partitions:
    """Partition the task's episodes; the calendar spans ``optimization_episodes``."""

    if not optimization_episodes:
        raise ProtocolError("the dataset has no optimization episodes")
    calendar = _calendar([frame.present_times(name) for name in optimization_episodes], task)
    intervals: dict[str, dict[str, tuple[int, int]]] = {stage: {} for stage in STAGES}
    rows: dict[str, dict[str, int]] = {stage: {} for stage in STAGES}
    evidence: dict[str, dict[str, int]] = {}
    for episode in task.episodes:
        ts = [int(value) for value in frame.present_times(episode)]
        ranges, evidence[episode] = _episode_ranges(ts, calendar, task)
        after_last = ts[-1] + 1
        for stage, (start, end) in ranges.items():
            intervals[stage][episode] = (
                ts[start] if start < len(ts) else after_last,
                ts[end] if end < len(ts) else after_last,
            )
            rows[stage][episode] = end - start
    body = {
        "schema": "ecogym.partitions/1",
        "protocol": task.protocol,
        "frame": frame.digest,
        "calendar": calendar,
        "intervals": intervals,
    }
    return Partitions(task.id, intervals, rows, calendar, evidence, digest(body))


__all__ = [
    "DEVELOPMENT_STAGES",
    "SEALED_STAGES",
    "STAGES",
    "Partitions",
    "ProtocolError",
    "four_stage",
]
