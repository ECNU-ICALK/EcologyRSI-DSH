"""Causal access to data at forecast origins.

``Origins`` is the only way solvers see data. It holds a set of origin rows
and answers ``past(column, k)``: the value k >= 0 steps before each origin,
or NaN when that row lies outside the readable history (another episode,
outside the partition when the task's history scope is "partition", or
before the data). Future values are reachable only through ``label``, which
exists on training origins (calibration_fit) and on the scorer's private
copy; forecast origins handed to a solver carry no labels.

Sealed stages (validation, final_test) are opened only with a
``SealedAccess`` issued by ``ecogym.arena.sealed`` after the exposure has
been recorded in the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.frame import Frame
from ..data.spec import ColumnSpec
from ..task.spec import TaskSpec
from .partitions import SEALED_STAGES, Partitions


class LeakageError(PermissionError):
    """A read would use information not available at the forecast origin."""


@dataclass(frozen=True, slots=True)
class SealedAccess:
    """Permission to open one sealed stage, issued after the exposure is recorded."""

    stage: str
    exposure_key: str


def forward_fill(values: np.ndarray, time: np.ndarray, episode: np.ndarray, max_age: int) -> np.ndarray:
    """Carry the last finite value forward for at most ``max_age`` steps, within episodes."""

    n = values.shape[0]
    if n == 0:
        return values.copy()
    index = np.where(np.isfinite(values), np.arange(n), -1)
    last = np.maximum.accumulate(index)
    first_row = np.zeros(n, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, episode[1:] != episode[:-1]])
    first_row[starts] = starts
    first_row = np.maximum.accumulate(first_row)
    usable = (last >= first_row) & (last >= 0)
    safe = np.where(usable, last, 0)
    usable &= (time - time[safe]) <= max_age
    return np.where(usable, values[safe], np.nan)


class Origins:
    """Forecast origins with causal reads; see the module docstring."""

    def __init__(
        self,
        frame: Frame,
        rows: np.ndarray,
        *,
        stage: str,
        history: np.ndarray,
        labels: np.ndarray | None = None,
        _fill_cache: dict | None = None,
    ) -> None:
        self._frame = frame
        self._rows = np.asarray(rows, dtype=np.int64)
        self._rows.setflags(write=False)
        self._episode = frame.episode[self._rows]
        self._history = history
        self._labels = labels
        self._fill_cache = {} if _fill_cache is None else _fill_cache
        self.stage = stage

    @property
    def size(self) -> int:
        return int(self._rows.shape[0])

    def __len__(self) -> int:
        return self.size

    @property
    def time(self) -> np.ndarray:
        return self._frame.time[self._rows]

    @property
    def episode(self) -> np.ndarray:
        return self._episode

    @property
    def has_labels(self) -> bool:
        return self._labels is not None

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(sorted(self._frame.columns))

    def spec(self, column: str) -> ColumnSpec:
        return self._frame.specs[column]

    def identity(self) -> dict:
        """Which rows these are, for digests and cache keys."""

        return {"stage": self.stage, "time": self.time, "episode": self._episode}

    def _read(self, values: np.ndarray, offset: int, region: np.ndarray) -> np.ndarray:
        n = values.shape[0]
        source = self._rows + offset
        inside = (source >= 0) & (source < n)
        clipped = np.clip(source, 0, max(n - 1, 0))
        ok = inside & (self._frame.episode[clipped] == self._episode) & region[clipped]
        out = np.full(self.size, np.nan)
        out[ok] = values[clipped[ok]]
        return out

    def _column(self, column: str) -> np.ndarray:
        try:
            return self._frame.columns[column]
        except KeyError:
            raise KeyError(f"unknown column {column}") from None

    def past(self, column: str, k: int) -> np.ndarray:
        """Value ``k`` steps before each origin (k >= 0), NaN if not readable."""

        if k < 0:
            raise LeakageError(f"past() needs k >= 0, got {k}")
        return self._read(self._column(column), -int(k), self._history)

    def past_filled(self, column: str, k: int, max_age: int) -> np.ndarray:
        """Like ``past`` on the column forward-filled (within the history) for ``max_age`` steps."""

        if k < 0:
            raise LeakageError(f"past_filled() needs k >= 0, got {k}")
        key = (column, int(max_age))
        filled = self._fill_cache.get(key)
        if filled is None:
            values = np.where(self._history, self._column(column), np.nan)
            filled = forward_fill(values, self._frame.time, self._frame.episode, int(max_age))
            self._fill_cache[key] = filled
        return self._read(filled, -int(k), self._history)

    def label(self, column: str, h: int) -> np.ndarray:
        """Value ``h`` steps after each origin; only on origins that carry labels."""

        if self._labels is None:
            raise LeakageError("labels are not available at forecast origins")
        if h < 1:
            raise ValueError("label horizons start at 1")
        return self._read(self._column(column), int(h), self._labels)

    def take(self, selection: np.ndarray) -> Origins:
        """A subset by boolean mask or integer positions, sharing fill caches."""

        rows = self._rows[selection]
        return Origins(
            self._frame, rows, stage=self.stage, history=self._history, labels=self._labels,
            _fill_cache=self._fill_cache,
        )

    def without_labels(self) -> Origins:
        return Origins(self._frame, self._rows, stage=self.stage, history=self._history, _fill_cache=self._fill_cache)


def stage_rows(frame: Frame, partitions: Partitions, stage: str) -> np.ndarray:
    return np.flatnonzero(partitions.mask(frame, stage))


def open_stage(
    frame: Frame,
    partitions: Partitions,
    task: TaskSpec,
    stage: str,
    *,
    sealed: SealedAccess | None = None,
) -> Origins:
    """Every row of ``stage`` as origins with labels inside the stage.

    History is the stage itself ("partition" scope) or everything before
    ("causal" scope). Sealed stages require a matching ``SealedAccess``.
    """

    if stage in SEALED_STAGES and (sealed is None or sealed.stage != stage):
        raise PermissionError(f"{stage} is sealed; open it through ecogym.arena.sealed")
    mask = partitions.mask(frame, stage)
    history = mask if task.history_scope == "partition" else np.ones(len(frame), dtype=bool)
    return Origins(frame, np.flatnonzero(mask), stage=stage, history=history, labels=mask)


__all__ = ["LeakageError", "Origins", "SealedAccess", "forward_fill", "open_stage", "stage_rows"]
