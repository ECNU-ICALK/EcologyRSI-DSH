"""Screening and holdout cohorts for one generation.

The model_selection origins are grouped into calendar blocks (days), and the
blocks into consecutive chunks of ``chunk_blocks``. Each generation shuffles
the chunks with a seed derived from the run seed and the generation number:
the first ``screen_fraction`` of them form the screening cohort (rank the
candidates, pick a finalist) and the rest the holdout cohort (finalist
against incumbent). The two cohorts share no day, so the finalist's luck on
screening days cannot carry into its holdout comparison; consecutive days
inside a chunk keep the moving-block bootstrap usable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..kernel.canonical import digest
from .origins import blocks


@dataclass(frozen=True, slots=True)
class Cohort:
    role: str
    index: np.ndarray  # positions into the stage's eligible origins
    blocks: tuple[int, ...]
    digest: str

    @property
    def size(self) -> int:
        return int(self.index.shape[0])


def plan_generation(
    time: np.ndarray,
    *,
    block_steps: int,
    chunk_blocks: int,
    screen_fraction: float,
    seed: int,
    generation: int,
) -> tuple[Cohort, Cohort]:
    """Split eligible origin times into (screening, holdout) for ``generation``."""

    if not 0 < screen_fraction < 1:
        raise ValueError("screen_fraction must lie in (0, 1)")
    origin_blocks = blocks(time, block_steps)
    unique = np.unique(origin_blocks)
    chunk_of_block = {int(block): position // chunk_blocks for position, block in enumerate(unique)}
    chunks = np.array([chunk_of_block[int(block)] for block in origin_blocks], dtype=np.int64)
    count = int(chunks.max()) + 1 if chunks.size else 0
    if count < 2:
        raise ValueError("model_selection is too short for separate screening and holdout cohorts")
    material = digest({"schema": "ecogym.cohorts/1", "seed": seed, "generation": generation, "blocks": unique})
    order = np.random.default_rng(int(material[:16], 16)).permutation(count)
    screen_count = min(count - 1, max(1, round(count * screen_fraction)))
    screen = np.isin(chunks, order[:screen_count])
    cohorts = []
    for role, mask in (("screening", screen), ("holdout", ~screen)):
        index = np.flatnonzero(mask)
        cohort_blocks = tuple(int(block) for block in np.unique(origin_blocks[index]))
        cohorts.append(Cohort(role, index, cohort_blocks, digest({"role": role, "material": material, "index": index})))
    return cohorts[0], cohorts[1]


__all__ = ["Cohort", "plan_generation"]
