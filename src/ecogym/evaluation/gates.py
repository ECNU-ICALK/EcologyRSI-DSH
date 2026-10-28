"""Promotion gates: is a candidate really better than the incumbent?

On a cohort's origins, a candidate replaces the incumbent only if all hold:

1. practical gain: objective(candidate) - objective(incumbent) > ``min_delta``;
2. enough evidence: at least ``min_blocks`` calendar blocks;
3. the paired block bootstrap interval of that gain (``confidence``, blocks
   drawn with replacement) lies above zero;
4. no cell regresses: for every cell, with d = nRMSE(candidate) -
   nRMSE(incumbent) on rows both forecast, the lower ``cell_alpha``/2 bound
   of a moving-block bootstrap of d is <= 0 (no credible regression) and
   d <= ``cell_cap`` * nRMSE(incumbent) (a small point loss is the most a
   wide interval may excuse). This needs ``min_starts`` legal block starts.

These are the v0.10.33 rules with one change of reference: the per-cell
check compares against the incumbent, not the naive baseline, so a promotion
can never trade away a cell the incumbent had already won.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..task.spec import GateSpec
from .bootstrap import iid_counts, legal_starts, moving_counts, percentile
from .harness import Evaluation
from .scoring import objective


@dataclass(frozen=True, slots=True)
class CellVerdict:
    cell: str
    n: int
    nrmse_candidate: float
    nrmse_incumbent: float
    d: float
    d_lower: float | None
    d_upper: float | None
    cap: float
    passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class Decision:
    promoted: bool
    reason: str
    candidate: float
    incumbent: float
    delta: float
    interval: tuple[float, float]
    blocks: int
    starts: int
    cells: tuple[CellVerdict, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def objective_interval(
    evaluation: Evaluation, subset: np.ndarray | None, gates: GateSpec, seed: Any
) -> tuple[float, float]:
    """Block bootstrap interval of one method's objective (its skill against the baseline)."""

    sums = evaluation.sums(subset)
    counts = iid_counts(sums.blocks.size, gates.resamples, {"interval": seed, "blocks": sums.blocks})
    values = np.sort(objective(sums.total(counts), evaluation.stage.weights))
    tail = (1.0 - gates.confidence) / 2.0
    return percentile(values, tail), percentile(values, 1.0 - tail)


def _paired_cell_sums(candidate: Evaluation, incumbent: Evaluation, subset: np.ndarray, blocks: np.ndarray):
    stage = candidate.stage
    scales = stage.scales[:, None]
    a = candidate.errors[:, subset] / scales
    b = incumbent.errors[:, subset] / scales
    both = np.isfinite(a) & np.isfinite(b)
    position = np.searchsorted(blocks, stage.blocks[subset])
    shape = (len(stage.cells), blocks.size)
    n, sse_a, sse_b = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for cell in range(len(stage.cells)):
        n[cell] = np.bincount(position, weights=both[cell].astype(float), minlength=blocks.size)
        sse_a[cell] = np.bincount(position, weights=np.where(both[cell], a[cell], 0.0) ** 2, minlength=blocks.size)
        sse_b[cell] = np.bincount(position, weights=np.where(both[cell], b[cell], 0.0) ** 2, minlength=blocks.size)
    return n, sse_a, sse_b


def compare(
    candidate: Evaluation, incumbent: Evaluation, subset: np.ndarray, gates: GateSpec, seed: Any
) -> Decision:
    if candidate.stage is not incumbent.stage:
        raise ValueError("candidate and incumbent must be evaluated on the same stage")
    stage = candidate.stage
    cand_sums, inc_sums = candidate.sums(subset), incumbent.sums(subset)
    blocks = cand_sums.blocks
    cand_score = float(objective(cand_sums.total(), stage.weights)[0])
    inc_score = float(objective(inc_sums.total(), stage.weights)[0])
    delta = cand_score - inc_score
    material = {"candidate": candidate.genome.digest, "incumbent": incumbent.genome.digest,
                "blocks": blocks, "seed": seed}

    counts = iid_counts(blocks.size, gates.resamples, {"gate": "overall", **material})
    resampled = objective(cand_sums.total(counts), stage.weights) - objective(inc_sums.total(counts), stage.weights)
    deltas = np.sort(resampled)
    tail = (1.0 - gates.confidence) / 2.0
    interval = (percentile(deltas, tail), percentile(deltas, 1.0 - tail))

    starts = legal_starts(blocks.tolist(), gates.moving_block)
    sufficient = blocks.size >= gates.min_blocks and len(starts) >= gates.min_starts
    draws = moving_counts(blocks.tolist(), gates.moving_block, gates.resamples, {"gate": "cells", **material})
    n, sse_a, sse_b = _paired_cell_sums(candidate, incumbent, subset, blocks)
    verdicts = []
    for position, cell in enumerate(stage.cells):
        total = n[position].sum()
        if total <= 0:
            nan = float("nan")
            verdicts.append(CellVerdict(cell.key, 0, nan, nan, nan, None, None, nan, False, "no_paired_forecasts"))
            continue
        nrmse_a = float(np.sqrt(sse_a[position].sum() / total))
        nrmse_b = float(np.sqrt(sse_b[position].sum() / total))
        d = nrmse_a - nrmse_b
        cap = gates.cell_cap * nrmse_b
        lower = upper = None
        if sufficient and draws is not None:
            drawn_n = draws @ n[position]
            ok = drawn_n > 0
            resampled = np.sort(np.sqrt((draws @ sse_a[position])[ok] / drawn_n[ok])
                                - np.sqrt((draws @ sse_b[position])[ok] / drawn_n[ok]))
            if resampled.size:
                lower = percentile(resampled, gates.cell_alpha / 2.0)
                upper = percentile(resampled, 1.0 - gates.cell_alpha / 2.0)
        if lower is None:
            passed, reason = False, "insufficient_evidence"
        elif d > cap:
            passed, reason = False, "regression_above_cap"
        elif lower > 0.0:
            passed, reason = False, "credible_regression"
        else:
            passed, reason = True, "better" if d <= 0 else "within_uncertainty"
        verdicts.append(CellVerdict(cell.key, int(total), nrmse_a, nrmse_b, d, lower, upper, cap, passed, reason))

    if delta <= gates.min_delta:
        promoted, reason = False, "below_practical_delta"
    elif blocks.size < gates.min_blocks:
        promoted, reason = False, "insufficient_evidence"
    elif interval[0] <= 0.0:
        promoted, reason = False, "interval_crosses_zero"
    elif not all(verdict.passed for verdict in verdicts):
        failed = [verdict.cell for verdict in verdicts if not verdict.passed]
        promoted, reason = False, "cell_regression:" + ",".join(failed)
    else:
        promoted, reason = True, "improved"
    return Decision(
        promoted, reason, cand_score, inc_score, delta, interval, int(blocks.size), len(starts), tuple(verdicts)
    )


__all__ = ["CellVerdict", "Decision", "compare", "objective_interval"]
