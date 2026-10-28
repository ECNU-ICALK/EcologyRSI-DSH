"""The dense time grid every task reads from.

A ``Frame`` concatenates episodes (teams, sites) end to end. Within an
episode every time step from its first to its last observation has a row, so
a lag of k steps is always k rows back; steps missing from the source are
rows with ``present`` false and NaN values. ``time`` holds the true step index
(for hourly data, hours since 1899-12-30) and ``episode`` the episode code.
Reads never cross an episode boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..kernel.canonical import canonical_json
from .spec import ColumnSpec


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    dataset_id: str
    step: str
    time: np.ndarray
    episode: np.ndarray
    episode_ids: tuple[str, ...]
    present: np.ndarray
    columns: Mapping[str, np.ndarray]
    specs: Mapping[str, ColumnSpec]
    digest: str

    def __len__(self) -> int:
        return int(self.time.shape[0])

    @classmethod
    def build(
        cls,
        dataset_id: str,
        step: str,
        episodes: Sequence[tuple[str, np.ndarray, Mapping[str, np.ndarray]]],
        specs: Mapping[str, ColumnSpec],
    ) -> Frame:
        """Assemble episodes given as ``(name, sorted unique times, values)``."""

        names = sorted(set().union(*(values.keys() for _, _, values in episodes))) if episodes else []
        times, codes, present = [], [], []
        columns: dict[str, list[np.ndarray]] = {name: [] for name in names}
        for code, (name, episode_times, values) in enumerate(episodes):
            episode_times = np.asarray(episode_times, dtype=np.int64)
            if episode_times.size == 0 or np.any(np.diff(episode_times) <= 0):
                raise ValueError(f"episode {name} needs strictly increasing times")
            dense = np.arange(episode_times[0], episode_times[-1] + 1, dtype=np.int64)
            position = episode_times - episode_times[0]
            mask = np.zeros(dense.size, dtype=bool)
            mask[position] = True
            times.append(dense)
            codes.append(np.full(dense.size, code, dtype=np.int32))
            present.append(mask)
            for column in names:
                grid = np.full(dense.size, np.nan)
                if column in values:
                    grid[position] = np.asarray(values[column], dtype=np.float64)
                columns[column].append(grid)
        frame_columns = {name: _readonly(np.concatenate(parts)) for name, parts in columns.items()}
        time = _readonly(np.concatenate(times) if times else np.zeros(0, np.int64))
        episode = _readonly(np.concatenate(codes) if codes else np.zeros(0, np.int32))
        present_rows = _readonly(np.concatenate(present) if present else np.zeros(0, bool))
        episode_ids = tuple(name for name, _, _ in episodes)
        kept_specs = {name: specs[name] for name in names}
        return cls(
            dataset_id=dataset_id,
            step=step,
            time=time,
            episode=episode,
            episode_ids=episode_ids,
            present=present_rows,
            columns=frame_columns,
            specs=kept_specs,
            digest=_frame_digest(dataset_id, step, episode_ids, time, episode, present_rows, frame_columns, kept_specs),
        )

    def episode_code(self, name: str) -> int:
        try:
            return self.episode_ids.index(name)
        except ValueError:
            raise KeyError(f"{self.dataset_id} has no episode {name}") from None

    def episode_rows(self, name: str) -> slice:
        rows = np.flatnonzero(self.episode == self.episode_code(name))
        return slice(int(rows[0]), int(rows[-1]) + 1)

    def present_times(self, name: str) -> np.ndarray:
        rows = self.episode_rows(name)
        return self.time[rows][self.present[rows]]

    def select(self, names: Sequence[str]) -> Frame:
        """A frame holding only the named episodes, in the given order."""

        episodes = []
        for name in names:
            rows = self.episode_rows(name)
            keep = self.present[rows]
            episodes.append(
                (
                    name,
                    self.time[rows][keep],
                    {column: values[rows][keep] for column, values in self.columns.items()},
                )
            )
        return Frame.build(self.dataset_id, self.step, episodes, self.specs)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _frame_digest(
    dataset_id: str,
    step: str,
    episode_ids: tuple[str, ...],
    time: np.ndarray,
    episode: np.ndarray,
    present: np.ndarray,
    columns: Mapping[str, np.ndarray],
    specs: Mapping[str, ColumnSpec],
) -> str:
    sha = hashlib.sha256()
    header = {
        "schema": "ecogym.frame/1",
        "dataset_id": dataset_id,
        "step": step,
        "episodes": list(episode_ids),
        "columns": {name: [specs[name].role, specs[name].unit] for name in sorted(columns)},
    }
    sha.update(canonical_json(header).encode("utf-8"))
    sha.update(time.astype("<i8").tobytes())
    sha.update(episode.astype("<i4").tobytes())
    sha.update(present.astype("u1").tobytes())
    for name in sorted(columns):
        sha.update(name.encode("utf-8"))
        sha.update(columns[name].astype("<f8").tobytes())
    return sha.hexdigest()


__all__ = ["Frame"]
