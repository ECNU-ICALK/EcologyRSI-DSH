"""Block bootstrap draws over calendar blocks.

Forecast errors are strongly correlated within a day and between adjacent
days, so resampling is over blocks, never rows. Two schemes, as in v0.10.33:

* ``iid_counts``: blocks drawn with replacement (the overall promotion
  interval);
* ``moving_counts``: non-circular runs of ``length`` consecutive blocks,
  concatenated and truncated to the original block count (the per-cell
  non-inferiority interval). Runs may only start where ``length``
  consecutive block ids are present.

Draws are seeded from a digest of the evidence they judge, so every verdict
can be re-derived from the ledger. Each draw is returned as counts per block.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..kernel.canonical import digest


def _rng(seed_material: Any) -> random.Random:
    return random.Random(int(digest(seed_material)[:16], 16))


def legal_starts(blocks: Sequence[int], length: int) -> tuple[int, ...]:
    """Positions where ``length`` consecutive block ids begin."""

    ids = [int(block) for block in blocks]
    return tuple(
        position
        for position in range(0, len(ids) - length + 1)
        if ids[position : position + length] == list(range(ids[position], ids[position] + length))
    )


def iid_counts(block_count: int, resamples: int, seed_material: Any) -> np.ndarray:
    rng = _rng(seed_material)
    counts = np.zeros((resamples, block_count))
    for row in range(resamples):
        for _ in range(block_count):
            counts[row, rng.randrange(block_count)] += 1
    return counts


def moving_counts(blocks: Sequence[int], length: int, resamples: int, seed_material: Any) -> np.ndarray | None:
    starts = legal_starts(blocks, length)
    if not starts:
        return None
    rng = _rng(seed_material)
    count = len(blocks)
    counts = np.zeros((resamples, count))
    for row in range(resamples):
        selected: list[int] = []
        while len(selected) < count:
            start = starts[rng.randrange(len(starts))]
            selected.extend(range(start, start + length))
        for position in selected[:count]:
            counts[row, position] += 1
    return counts


def percentile(sorted_values: np.ndarray, probability: float) -> float:
    """The v0.10.33 index rule: ``sorted[int(p * (n - 1))]``."""

    return float(sorted_values[int(probability * (sorted_values.size - 1))])


__all__ = ["iid_counts", "legal_starts", "moving_counts", "percentile"]
