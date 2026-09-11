"""Per-cell statistical non-inferiority, replacing the per-cell 1e-12 knife edge.

The gate this module implements answers a different question than the one it
replaces. ``all_targets_no_regression`` asked "is every cell's normalized RMSE
no worse than the baseline's, to within 1e-12" — a deterministic threshold on a
quantity that is itself a noisy estimate from a few dozen paired origins. Paired
with the overall requirement ``objective_score > 1e-9``, that made the feasible
set empty in practice: a candidate identical to the baseline scores exactly zero
skill and fails the overall gate, while any candidate that actually differs
perturbs at least one of the nine cells upward by more than 1e-12 and fails the
per-cell gate. Nothing improved because nothing *could*.

What replaces it is a two-part test, and both parts have to hold:

1. ``d_lcb(cell) <= 0`` — the regression is only called a regression when it is
   credible at the declared confidence level. Noisy cells get a wider tolerance
   automatically; adding paired blocks tightens it automatically. The boundary
   is derived from the data, not chosen by hand.
2. ``d(cell) <= hard_cap_ratio * nRMSE_base(cell)`` — an absolute ceiling, so a
   genuinely large regression can never hide behind a wide interval. The
   2.25x blow-up at ``air_temperature@6h`` that sank the last run is refused by
   this clause no matter how few blocks were collected.

Two rejected alternatives, for the record. ``d_ucb <= 0`` demands that every
cell be *significantly better*, which is stricter than the 1e-12 rule and would
deepen the deadlock. A fixed additive margin cannot adapt to each cell's noise
scale and is exactly the "quietly lowered the bar" move this design is meant to
rule out.

The statistic costs no new evaluation rows. ``build_promotion_block_evidence``
already writes per-block per-cell ``succeeded`` counts and normalized squared
error sums, and both are additive over blocks, so a paired interval is a
resampling of numbers the evaluator has always produced. Because
``PROMOTION_BLOCK_EVIDENCE_VERSION`` does not move, this gate can also be
back-computed against archived runs to ask what it would have decided then.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from ..core.models import digest
from ..evolution.promotion import (
    PROMOTION_BLOCK_HOURS,
    PROMOTION_BOOTSTRAP_RESAMPLES,
    PROMOTION_CONFIDENCE_LEVEL,
    PROMOTION_MINIMUM_PAIRED_BLOCKS,
)
from .fitness import _legal_starts, build_moving_block_resample_indices


CELL_NONINFERIORITY_GATE_ID = "per_cell_noninferiority@1"
EVIDENCE_SUFFICIENCY_GATE_ID = "noninferiority_evidence_sufficient@1"
RELAXATION_DEBT_GATE_ID = "noninferiority_relaxation_debt@1"
CELL_NONINFERIORITY_EVIDENCE_VERSION = "ecologyrsi-dsh.cell-noninferiority/1"
CELL_NONINFERIORITY_STATISTIC = "paired_moving_block_bootstrap_nrmse_difference@1"

#: Verdict labels. ``strictly_better`` is the subset the retired gate would also
#: have admitted, which is what makes the relaxation auditable rather than
#: merely asserted.
ADMITTED_STRICTLY_BETTER = "strictly_better"
ADMITTED_WITHIN_UNCERTAINTY = "within_bootstrap_uncertainty"
ADMITTED_FAILED = "failed"

DEFAULT_HARD_CAP_RATIO = 0.03
DEFAULT_ALPHA = 0.05
DEFAULT_MOVING_BLOCK_DAYS = 3
DEFAULT_MINIMUM_VALID_THREE_DAY_STARTS = 4


@dataclass(frozen=True, slots=True)
class NoninferiorityPolicy:
    """Every constant that can change the gate's verdict, in one place.

    Instances are projected into the scoring contract, so a change here moves
    ``evaluator_digest`` and ``_common_contract_matches`` stops candidates
    judged under different semantics from being compared. That mechanism
    already exists; this dataclass only has to be honest about its inputs.
    """

    hard_cap_ratio: float = DEFAULT_HARD_CAP_RATIO
    alpha: float = DEFAULT_ALPHA
    bootstrap_resamples: int = PROMOTION_BOOTSTRAP_RESAMPLES
    minimum_paired_blocks: int = PROMOTION_MINIMUM_PAIRED_BLOCKS
    minimum_valid_three_day_starts: int = DEFAULT_MINIMUM_VALID_THREE_DAY_STARTS
    moving_block_days: int = DEFAULT_MOVING_BLOCK_DAYS

    def __post_init__(self) -> None:
        for name in ("hard_cap_ratio", "alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"noninferiority {name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"noninferiority {name} must be finite")
        if self.hard_cap_ratio < 0:
            raise ValueError("noninferiority hard cap ratio must be nonnegative")
        if not 0 < float(self.alpha) < 0.5:
            raise ValueError("noninferiority alpha must lie in (0, 0.5)")
        for name in (
            "bootstrap_resamples",
            "minimum_paired_blocks",
            "minimum_valid_three_day_starts",
            "moving_block_days",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"noninferiority {name} must be a positive integer")

    @property
    def confidence_level(self) -> float:
        return 1.0 - float(self.alpha)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_semantics_id": CELL_NONINFERIORITY_GATE_ID,
            "statistic": CELL_NONINFERIORITY_STATISTIC,
            "hard_cap_ratio": float(self.hard_cap_ratio),
            "alpha": float(self.alpha),
            "confidence_level": self.confidence_level,
            "bootstrap_resamples": int(self.bootstrap_resamples),
            "minimum_paired_blocks": int(self.minimum_paired_blocks),
            "minimum_valid_three_day_starts": int(
                self.minimum_valid_three_day_starts
            ),
            "moving_block_days": int(self.moving_block_days),
            "block_hours": PROMOTION_BLOCK_HOURS,
        }


@dataclass(frozen=True, slots=True)
class CellNoninferiorityAssessment:
    """The gate's verdict plus the evidence that makes it checkable."""

    passed: bool
    evidence_sufficient: bool
    evidence: dict[str, Any]
    audit: dict[str, Any]

    @property
    def failed_cells(self) -> tuple[str, ...]:
        return tuple(
            f"{cell['target']}@{cell['horizon_hours']}h"
            for cell in self.evidence["cells"]
            if not cell["verdict"]
        )


def _cell_label(target: str, horizon_hours: int) -> str:
    return f"{target}@{horizon_hours}h"


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _parse_blocks(
    block_evidence: Mapping[str, Any],
) -> tuple[tuple[int, ...], dict[int, str], dict[tuple[str, int], dict[int, tuple[int, float, float]]]] | None:
    """Index the raw block evidence by calendar block, then by cell.

    Returns ``None`` on any structural surprise rather than raising: the gate's
    contract is to fail closed, and a caller that cannot parse its own evidence
    has no business certifying anything. The evidence is normally produced a few
    hundred lines earlier in the same function, so this is a guard against
    drift, not against hostile input.
    """

    raw_blocks = block_evidence.get("blocks")
    if not isinstance(raw_blocks, (list, tuple)) or not raw_blocks:
        return None
    block_ids: dict[int, str] = {}
    stats: dict[tuple[str, int], dict[int, tuple[int, float, float]]] = {}
    for block in raw_blocks:
        if not isinstance(block, Mapping):
            return None
        index = block.get("origin_block_index")
        block_id = block.get("block_id")
        cells = block.get("cells")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not isinstance(block_id, str)
            or index in block_ids
            or not isinstance(cells, (list, tuple))
        ):
            return None
        block_ids[index] = block_id
        for cell in cells:
            if not isinstance(cell, Mapping):
                return None
            horizon = cell.get("horizon_hours")
            succeeded = cell.get("succeeded")
            if (
                isinstance(horizon, bool)
                or not isinstance(horizon, int)
                or isinstance(succeeded, bool)
                or not isinstance(succeeded, int)
                or succeeded < 0
            ):
                return None
            candidate_sq = _numeric(cell.get("candidate_squared_error_sum"))
            baseline_sq = _numeric(cell.get("baseline_squared_error_sum"))
            if candidate_sq is None or baseline_sq is None:
                return None
            if candidate_sq < 0 or baseline_sq < 0:
                return None
            key = (str(cell.get("target")), horizon)
            per_block = stats.setdefault(key, {})
            if index in per_block:
                return None
            per_block[index] = (succeeded, candidate_sq, baseline_sq)
    indices = tuple(sorted(block_ids))
    for per_block in stats.values():
        if set(per_block) != set(indices):
            return None
    return indices, block_ids, stats


def _difference(
    per_block: Mapping[int, tuple[int, float, float]],
    selection: Sequence[int],
) -> tuple[int, float, float, float] | None:
    """``(n, nRMSE_cand, nRMSE_base, d)`` over one multiset of calendar blocks.

    Both sums and the count are additive over blocks, which is the whole reason
    a paired per-cell interval needs no new evaluation rows.
    """

    total = 0
    candidate_sq = 0.0
    baseline_sq = 0.0
    for index in selection:
        succeeded, block_candidate, block_baseline = per_block[index]
        total += succeeded
        candidate_sq += block_candidate
        baseline_sq += block_baseline
    if total <= 0:
        return None
    candidate_nrmse = math.sqrt(candidate_sq / total)
    baseline_nrmse = math.sqrt(baseline_sq / total)
    return total, candidate_nrmse, baseline_nrmse, candidate_nrmse - baseline_nrmse


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    # The same index convention as `_paired_bootstrap_interval`, so the per-cell
    # interval and the overall promotion interval cannot disagree about what a
    # 95% percentile bootstrap bound means.
    return sorted_values[int(probability * (len(sorted_values) - 1))]


def assess_cell_noninferiority(
    block_evidence: Mapping[str, Any],
    *,
    cell_metrics: Sequence[Mapping[str, Any]],
    policy: NoninferiorityPolicy,
    legacy_tolerance: float,
    effective_degrees_of_freedom: int | None = None,
) -> CellNoninferiorityAssessment:
    """Decide the per-cell gate and emit the evidence a reviewer needs.

    ``cell_metrics`` is the evaluator's own per-cell row list — ``task_results``
    on the multi-horizon path, ``target_results`` on the single-horizon one. It
    supplies two things the block evidence does not: the grid the evaluator
    believes it scored, and the exact ``normalized_rmse`` pair the retired gate
    compared, so ``would_pass_under_legacy_zero_tolerance`` is a statement about
    the old quantity under the old tolerance rather than a re-derivation that
    could flatter the new gate.

    ``legacy_tolerance`` is the adapter's surviving ``no_regression_tolerance``.
    It no longer decides anything; it is kept so the audit line stays truthful.
    """

    policy_body = policy.to_dict()
    expected: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in cell_metrics:
        horizon = row.get("horizon_hours")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            continue
        expected[(str(row.get("target")), horizon)] = row
    parsed = _parse_blocks(block_evidence)
    if parsed is None or not expected or set(parsed[2]) != set(expected):
        # Fail closed and say which way it broke. A grid mismatch means the
        # block evidence and the scored rows disagree about what was measured,
        # which is never a condition under which a promotion should be granted.
        reason = (
            "promotion_block_evidence_unparseable"
            if parsed is None
            else "promotion_block_grid_mismatch"
        )
        return CellNoninferiorityAssessment(
            passed=False,
            evidence_sufficient=False,
            evidence={
                "schema_version": CELL_NONINFERIORITY_EVIDENCE_VERSION,
                **policy_body,
                "evidence_sufficient": False,
                "evidence_failure": reason,
                "paired_block_count": 0,
                "valid_three_day_start_count": 0,
                "block_ids_digest": None,
                "resample_family_digest": None,
                "cells": [],
            },
            audit={
                "schema_version": CELL_NONINFERIORITY_EVIDENCE_VERSION,
                "gate_semantics_id": CELL_NONINFERIORITY_GATE_ID,
                "legacy_gate_id": "all_targets_no_regression",
                "legacy_tolerance": float(legacy_tolerance),
                "cells_strictly_better": 0,
                "cells_admitted_only_by_noninferiority": 0,
                "cells_failed": 0,
                "would_pass_under_legacy_zero_tolerance": False,
                "requires_stronger_overall_evidence": False,
            },
        )
    indices, block_ids, stats = parsed
    starts = _legal_starts(indices, policy.moving_block_days)
    block_ids_digest = digest(
        {
            "indices": list(indices),
            "block_ids": [block_ids[index] for index in indices],
        }
    )
    evidence_sufficient = (
        len(indices) >= policy.minimum_paired_blocks
        and len(starts) >= policy.minimum_valid_three_day_starts
    )

    # The resample family is drawn once and shared by every cell. Cells scored
    # on different random families could contradict each other about the same
    # calendar days, and the shared family is also what makes the seed material
    # -- and therefore the whole verdict -- reproducible from the ledger.
    draws: tuple[tuple[int, ...], ...] = ()
    if evidence_sufficient:
        draws = build_moving_block_resample_indices(
            indices,
            resamples=policy.bootstrap_resamples,
            seed_material={
                "gate_semantics_id": CELL_NONINFERIORITY_GATE_ID,
                "statistic": CELL_NONINFERIORITY_STATISTIC,
                "block_ids_digest": block_ids_digest,
                "alpha": float(policy.alpha),
            },
            moving_block_days=policy.moving_block_days,
        )
        if not draws:
            evidence_sufficient = False

    cells: list[dict[str, Any]] = []
    strictly_better = 0
    admitted_only_by_noninferiority = 0
    failed = 0
    legacy_pass = True
    for key in sorted(expected):
        target, horizon = key
        per_block = stats[key]
        row = expected[key]
        point = _difference(per_block, indices)
        paired_block_count = sum(
            1 for index in indices if per_block[index][0] > 0
        )
        legacy_candidate = _numeric(row.get("normalized_rmse"))
        legacy_baseline = _numeric(row.get("baseline_normalized_rmse"))
        legacy_cell_pass = (
            legacy_candidate is not None
            and legacy_baseline is not None
            and legacy_candidate <= legacy_baseline + float(legacy_tolerance)
        )
        legacy_pass = legacy_pass and legacy_cell_pass
        if point is None:
            # No paired samples at all in this cell. There is nothing to be
            # non-inferior to, so it cannot be certified.
            cells.append(
                {
                    "target": target,
                    "horizon_hours": horizon,
                    "cell": _cell_label(target, horizon),
                    "n": 0,
                    "paired_block_count": paired_block_count,
                    "nrmse_base": None,
                    "nrmse_cand": None,
                    "d": None,
                    "d_lcb": None,
                    "d_ucb": None,
                    "hard_cap_value": None,
                    "verdict": False,
                    "admitted_by": ADMITTED_FAILED,
                    "failure_reason": "no_paired_samples",
                    "legacy_no_regression_pass": legacy_cell_pass,
                }
            )
            failed += 1
            continue
        total, candidate_nrmse, baseline_nrmse, delta = point
        hard_cap_value = float(policy.hard_cap_ratio) * baseline_nrmse
        lower: float | None = None
        upper: float | None = None
        if draws:
            resampled: list[float] = []
            for draw in draws:
                drawn = _difference(per_block, draw)
                if drawn is None:
                    continue
                resampled.append(drawn[3])
            if resampled:
                resampled.sort()
                half = float(policy.alpha) / 2.0
                lower = _quantile(resampled, half)
                upper = _quantile(resampled, 1.0 - half)
        within_cap = delta <= hard_cap_value
        # `d_lcb <= 0` is the pass condition: a regression is only credible when
        # the whole interval sits above zero. With insufficient evidence there is
        # no interval, and the gate refuses rather than treating "no interval" as
        # "no credible regression" -- otherwise shrinking the cohort would buy a
        # pass, which is precisely the loophole this ordering closes.
        interval_pass = lower is not None and lower <= 0.0
        verdict = bool(evidence_sufficient and interval_pass and within_cap)
        if verdict and delta <= 0.0:
            admitted_by = ADMITTED_STRICTLY_BETTER
            strictly_better += 1
        elif verdict:
            admitted_by = ADMITTED_WITHIN_UNCERTAINTY
            admitted_only_by_noninferiority += 1
        else:
            admitted_by = ADMITTED_FAILED
            failed += 1
        if verdict:
            failure_reason = None
        elif not evidence_sufficient:
            failure_reason = "insufficient_paired_evidence"
        elif not within_cap:
            failure_reason = "absolute_regression_cap_exceeded"
        else:
            failure_reason = "credible_regression"
        cells.append(
            {
                "target": target,
                "horizon_hours": horizon,
                "cell": _cell_label(target, horizon),
                "n": total,
                "paired_block_count": paired_block_count,
                "nrmse_base": baseline_nrmse,
                "nrmse_cand": candidate_nrmse,
                "d": delta,
                "d_lcb": lower,
                "d_ucb": upper,
                "hard_cap_value": hard_cap_value,
                "verdict": verdict,
                "admitted_by": admitted_by,
                "failure_reason": failure_reason,
                "legacy_no_regression_pass": legacy_cell_pass,
            }
        )

    passed = bool(evidence_sufficient and failed == 0)
    evidence: dict[str, Any] = {
        "schema_version": CELL_NONINFERIORITY_EVIDENCE_VERSION,
        **policy_body,
        "evidence_sufficient": evidence_sufficient,
        "evidence_failure": None
        if evidence_sufficient
        else "insufficient_paired_evidence",
        "paired_block_count": len(indices),
        "valid_three_day_start_count": len(starts),
        "block_ids_digest": block_ids_digest,
        "resample_family_digest": digest(
            {"draws": [list(draw) for draw in draws]}
        )
        if draws
        else None,
        "cells": cells,
    }
    if effective_degrees_of_freedom is not None:
        # The reviewer's expressiveness-versus-skill ratio: a recipe that bought
        # its cells with many more terms is a different claim than one that did
        # it with the same term count.
        evidence["effective_degrees_of_freedom"] = int(effective_degrees_of_freedom)
    audit = {
        "schema_version": CELL_NONINFERIORITY_EVIDENCE_VERSION,
        "gate_semantics_id": CELL_NONINFERIORITY_GATE_ID,
        "legacy_gate_id": "all_targets_no_regression",
        "legacy_tolerance": float(legacy_tolerance),
        "cells_strictly_better": strictly_better,
        "cells_admitted_only_by_noninferiority": admitted_only_by_noninferiority,
        "cells_failed": failed,
        "would_pass_under_legacy_zero_tolerance": bool(legacy_pass),
        # When the relaxation did any work, the promotion path owes the stronger
        # overall evidence described in the plan: a positive selection stability
        # floor and an overall gain above the practical delta, not merely above
        # `minimum_skill`. Recorded here so the requirement travels with the
        # evidence instead of living only in a reviewer's head.
        "requires_stronger_overall_evidence": admitted_only_by_noninferiority > 0,
    }
    return CellNoninferiorityAssessment(
        passed=passed,
        evidence_sufficient=evidence_sufficient,
        evidence=evidence,
        audit=audit,
    )


def assess_relaxation_debt(
    assessment: CellNoninferiorityAssessment,
    *,
    objective_score: float,
    escalated_minimum_skill: float,
) -> dict[str, Any]:
    """Charge a candidate that used the relaxation for the privilege.

    A cell admitted only because its regression sat inside the bootstrap
    interval is a cell where we could not prove the candidate is not worse. That
    is a defensible thing to allow, but it should not be free: the retired 1e-12
    rule would have refused it outright. So when any cell was admitted that way,
    the overall skill requirement escalates from ``minimum_skill`` (1e-9, which
    a candidate can clear by being trivially different from the baseline) to the
    practical delta the selection layer already demands.

    The certification path in ``evaluators/generation_comparison.py`` requires
    ``primary_selection_gate`` -- a positive stability floor *and* a gain above
    ``selection_minimum_score_delta`` -- for every candidate unconditionally, so
    this adds nothing there. It bites on ``scientific_pass``, whose only overall
    requirement is ``objective_score > minimum_skill``. Without this, a
    candidate could bank nine cells' worth of unprovable non-inferiority against
    an overall gain of 1e-9 and call it an improvement.

    ``applies`` is false for a strictly-better candidate, which is why this
    cannot function as a general tightening: it is a conditional debt, payable
    only by whoever draws on the relaxation.
    """

    applies = bool(assessment.audit.get("requires_stronger_overall_evidence"))
    score = _numeric(objective_score)
    threshold = _numeric(escalated_minimum_skill)
    if applies:
        passed = score is not None and threshold is not None and score > threshold
    else:
        passed = True
    return {
        "gate_semantics_id": RELAXATION_DEBT_GATE_ID,
        "applies": applies,
        "cells_admitted_only_by_noninferiority": int(
            assessment.audit.get("cells_admitted_only_by_noninferiority", 0)
        ),
        "objective_score": score,
        "escalated_minimum_skill": threshold,
        "passed": passed,
    }


__all__ = [
    "ADMITTED_FAILED",
    "ADMITTED_STRICTLY_BETTER",
    "ADMITTED_WITHIN_UNCERTAINTY",
    "CELL_NONINFERIORITY_EVIDENCE_VERSION",
    "CELL_NONINFERIORITY_GATE_ID",
    "CELL_NONINFERIORITY_STATISTIC",
    "EVIDENCE_SUFFICIENCY_GATE_ID",
    "RELAXATION_DEBT_GATE_ID",
    "CellNoninferiorityAssessment",
    "NoninferiorityPolicy",
    "assess_cell_noninferiority",
    "assess_relaxation_debt",
]
