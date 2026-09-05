"""Deterministic same-cohort selection for formal trajectory challengers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from ..core.models import digest
from ..core.trajectory import (
    BatchEvaluation,
    EvaluationPhase,
    FormalBatchArm,
    FormalBatchComparison,
    FormalBatchComparisonDecision,
)
from .promotion import V2_MINIMUM_SCORE_DELTA


LOCAL_MINIMUM_SCORE_DELTA = V2_MINIMUM_SCORE_DELTA
POSITIVE_DELTA_MINIMUM_SCORE_DELTA = 1e-12
LOCAL_CELL_REGRESSION_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class LocalChallengerAssessment:
    score_delta: float
    comparison_contract_digest: str
    safety_gate_passed: bool
    cell_regression_gate_passed: bool
    decision: FormalBatchComparisonDecision
    champion_after_revision_id: str
    reason: str


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(child) for child in value]
    return value


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def local_challenger_safety_reason(metrics: Mapping[str, Any]) -> str | None:
    """Derive the v2 challenger safety result from durable batch metrics."""

    if not isinstance(metrics, Mapping):
        return "batch_metrics_invalid"
    if "constraint_violations" not in metrics:
        return "constraint_guardrail_missing"
    violations = metrics["constraint_violations"]
    if isinstance(violations, bool) or not isinstance(violations, (int, float)):
        return "constraint_guardrail_invalid"
    if not math.isfinite(float(violations)) or float(violations) < 0:
        return "constraint_guardrail_invalid"
    if violations > 0:
        return "constraint_guardrail_failed"

    coverage = metrics.get("sample_execution_coverage_pass")
    sample = metrics.get("sample_execution")
    sample_coverage = (
        sample.get("coverage_pass") if isinstance(sample, Mapping) else None
    )
    failures = sample.get("failure_counts") if isinstance(sample, Mapping) else None
    rejected = (
        failures.get("constraint_rejected", 0)
        if isinstance(failures, Mapping)
        else 0
    )
    constraint_rejected = bool(
        not isinstance(rejected, bool)
        and isinstance(rejected, (int, float))
        and math.isfinite(float(rejected))
        and rejected > 0
    )
    if isinstance(sample, Mapping):
        strict_chain_pass = sample.get("strict_agent_chain_pass")
        if strict_chain_pass is not True:
            if constraint_rejected:
                return "sample_constraint_guardrail_failed"
            if strict_chain_pass is False:
                return "strict_origin_chain_guardrail_failed"
            if strict_chain_pass is None:
                return "strict_origin_chain_guardrail_missing"
            return "strict_origin_chain_guardrail_invalid"
    if coverage is not None and not isinstance(coverage, bool):
        return "coverage_guardrail_invalid"
    if sample_coverage is not None and not isinstance(sample_coverage, bool):
        return "coverage_guardrail_invalid"
    if coverage is False or sample_coverage is False:
        if constraint_rejected:
            return "sample_constraint_guardrail_failed"
        return "coverage_guardrail_failed"
    if coverage is not True and sample_coverage is not True:
        return "coverage_guardrail_missing"
    if not isinstance(sample, Mapping):
        return "strict_origin_chain_guardrail_missing"

    required_counts = (
        "attempted_origin_samples",
        "succeeded_origin_samples",
        "minimum_coverage",
    )
    if any(name not in sample for name in required_counts):
        return "coverage_guardrail_missing"
    attempted = sample["attempted_origin_samples"]
    succeeded = sample["succeeded_origin_samples"]
    minimum = sample["minimum_coverage"]
    if (
        isinstance(attempted, bool)
        or not isinstance(attempted, (int, float))
        or not math.isfinite(float(attempted))
        or attempted <= 0
        or isinstance(succeeded, bool)
        or not isinstance(succeeded, (int, float))
        or not math.isfinite(float(succeeded))
        or not 0 <= float(succeeded) <= float(attempted)
        or isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or not 0 <= float(minimum) <= 1
    ):
        return "coverage_guardrail_invalid"
    if float(succeeded) / float(attempted) < float(minimum):
        return (
            "sample_constraint_guardrail_failed"
            if constraint_rejected
            else "origin_coverage_guardrail_failed"
        )
    return None


def _metric_contract(evaluation: BatchEvaluation) -> dict[str, Any] | None:
    metrics = evaluation.metrics
    objective_version = metrics.get("objective_aggregation_version")
    weights = metrics.get("objective_target_weights")
    horizons = metrics.get("objective_horizons")
    if not isinstance(objective_version, str) or not objective_version.strip():
        return None
    if not isinstance(weights, Mapping) or not weights:
        return None
    normalized_weights: dict[str, float] = {}
    for target, raw_weight in weights.items():
        weight = _finite_number(raw_weight)
        if not isinstance(target, str) or not target or weight is None or weight < 0:
            return None
        normalized_weights[target] = weight
    if sum(normalized_weights.values()) <= 0:
        return None
    if not isinstance(horizons, (list, tuple)) or not horizons:
        return None
    normalized_horizons: list[int] = []
    for horizon in horizons:
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon < 1
            or horizon in normalized_horizons
        ):
            return None
        normalized_horizons.append(horizon)
    for name in (
        "baseline_profile_digest",
        "evaluation_index_digest",
        "dataset_digest",
        "split_manifest_digest_sha256",
    ):
        if not _is_sha256(metrics.get(name)):
            return None
    return {
        "objective_aggregation_version": objective_version,
        "objective_target_weights": normalized_weights,
        "objective_horizons": normalized_horizons,
        "baseline_profile_digest": metrics["baseline_profile_digest"],
        "evaluation_index_digest": metrics["evaluation_index_digest"],
        "dataset_digest": metrics["dataset_digest"],
        "split_manifest_digest_sha256": metrics[
            "split_manifest_digest_sha256"
        ],
    }


def _comparison_contract(
    champion: BatchEvaluation,
    challenger: BatchEvaluation,
) -> tuple[bool, str, dict[str, Any] | None]:
    champion_contract = _metric_contract(champion)
    challenger_contract = _metric_contract(challenger)
    contract_digest = digest(
        {
            "champion": _plain_json(champion_contract),
            "challenger": _plain_json(challenger_contract),
            "champion_evaluator_digest": champion.evaluator_digest,
            "challenger_evaluator_digest": challenger.evaluator_digest,
        }
    )
    champion_scope = champion.scope
    challenger_scope = challenger.scope
    same_scope = bool(
        champion_scope.phase is EvaluationPhase.FORMAL_BATCH
        and challenger_scope.phase is EvaluationPhase.FORMAL_BATCH
        and champion_scope.run_id == challenger_scope.run_id
        and champion_scope.generation == challenger_scope.generation
        and champion_scope.candidate_id == challenger_scope.candidate_id
        and champion_scope.batch_index == challenger_scope.batch_index
        and champion_scope.cohort_digest == challenger_scope.cohort_digest
        and champion_scope.origin_count == challenger_scope.origin_count
    )
    arms_valid = bool(
        champion is challenger
        or (
            champion_scope.formal_batch_arm is FormalBatchArm.CHAMPION
            and challenger_scope.formal_batch_arm is FormalBatchArm.CHALLENGER
        )
    )
    matches = bool(
        same_scope
        and arms_valid
        and champion.evaluator_digest == challenger.evaluator_digest
        and champion_contract is not None
        and champion_contract == challenger_contract
    )
    return matches, contract_digest, champion_contract if matches else None


def _cell_map(
    evaluation: BatchEvaluation,
    expected: set[tuple[str, int]],
) -> dict[tuple[str, int], float] | None:
    rows = evaluation.metrics.get("targets")
    if not isinstance(rows, (list, tuple)):
        return None
    cells: dict[tuple[str, int], float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        target = row.get("target")
        if not isinstance(target, str) or not target:
            return None
        if "horizon_hours" in row:
            raw_cells = (row,)
        else:
            raw_cells = row.get("horizons")
            if not isinstance(raw_cells, (list, tuple)):
                return None
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, Mapping):
                return None
            horizon = raw_cell.get("horizon_hours", raw_cell.get("hours"))
            if isinstance(horizon, bool) or not isinstance(horizon, int):
                return None
            key = (target, horizon)
            skill = _finite_number(
                raw_cell.get("skill_score", raw_cell.get("skill"))
            )
            if key not in expected or key in cells or skill is None:
                return None
            cells[key] = skill
    return cells if set(cells) == expected else None


def assess_local_challenger(
    champion: BatchEvaluation,
    challenger: BatchEvaluation,
    *,
    challenger_safety_gate_passed: bool,
    minimum_score_delta: float = LOCAL_MINIMUM_SCORE_DELTA,
    cell_regression_blocks: bool = True,
) -> LocalChallengerAssessment:
    """Select a challenger only from complete, paired, Host-owned evidence."""

    if not isinstance(champion, BatchEvaluation):
        raise TypeError("champion must be a BatchEvaluation")
    if not isinstance(challenger, BatchEvaluation):
        raise TypeError("challenger must be a BatchEvaluation")
    if not isinstance(challenger_safety_gate_passed, bool):
        raise TypeError("challenger_safety_gate_passed must be a bool")
    if (
        isinstance(minimum_score_delta, bool)
        or not isinstance(minimum_score_delta, (int, float))
        or not math.isfinite(float(minimum_score_delta))
        or float(minimum_score_delta) < 0
    ):
        raise ValueError("minimum_score_delta must be a finite non-negative number")
    if not isinstance(cell_regression_blocks, bool):
        raise TypeError("cell_regression_blocks must be a bool")

    score_delta = challenger.score - champion.score
    contract_matches, contract_digest, contract = _comparison_contract(
        champion,
        challenger,
    )
    champion_cells: dict[tuple[str, int], float] | None = None
    challenger_cells: dict[tuple[str, int], float] | None = None
    if contract is not None:
        expected = {
            (target, horizon)
            for target in contract["objective_target_weights"]
            for horizon in contract["objective_horizons"]
        }
        champion_cells = _cell_map(champion, expected)
        challenger_cells = _cell_map(challenger, expected)

    cell_regression_gate_passed = bool(
        champion_cells is not None
        and challenger_cells is not None
        and all(
            challenger_cells[key] + LOCAL_CELL_REGRESSION_TOLERANCE
            >= champion_cells[key]
            for key in champion_cells
        )
    )

    if not contract_matches:
        reason = "incompatible_comparison_contract"
    elif champion_cells is None:
        reason = "champion_evaluation_incomplete"
    elif challenger_cells is None:
        reason = "challenger_evaluation_incomplete"
    elif not challenger_safety_gate_passed:
        reason = "challenger_safety_gate_failed"
    elif cell_regression_blocks and not cell_regression_gate_passed:
        reason = "challenger_cell_regression"
    elif score_delta <= float(minimum_score_delta):
        reason = (
            "no_positive_score_delta"
            if math.isclose(
                float(minimum_score_delta),
                POSITIVE_DELTA_MINIMUM_SCORE_DELTA,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            else "below_practical_delta"
        )
    else:
        reason = "challenger_improved"

    promoted = reason == "challenger_improved"
    return LocalChallengerAssessment(
        score_delta=score_delta,
        comparison_contract_digest=contract_digest,
        safety_gate_passed=challenger_safety_gate_passed,
        cell_regression_gate_passed=cell_regression_gate_passed,
        decision=(
            FormalBatchComparisonDecision.CHALLENGER_PROMOTED
            if promoted
            else FormalBatchComparisonDecision.CHAMPION_RETAINED
        ),
        champion_after_revision_id=(
            challenger.scope.candidate_revision_id
            if promoted
            else champion.scope.candidate_revision_id
        ),
        reason=reason,
    )


def validate_formal_batch_comparison(
    comparison: FormalBatchComparison,
    champion: BatchEvaluation,
    challenger: BatchEvaluation,
    *,
    minimum_score_delta: float = LOCAL_MINIMUM_SCORE_DELTA,
    cell_regression_blocks: bool = True,
) -> None:
    """Reject comparison fields that differ from Host-derived evidence."""

    if not isinstance(comparison, FormalBatchComparison):
        raise TypeError("comparison must be a FormalBatchComparison")
    safety_gate_passed = (
        local_challenger_safety_reason(challenger.metrics) is None
    )
    assessment = assess_local_challenger(
        champion,
        challenger,
        challenger_safety_gate_passed=safety_gate_passed,
        minimum_score_delta=minimum_score_delta,
        cell_regression_blocks=cell_regression_blocks,
    )
    if comparison.batch_index == 0:
        expected_decision = FormalBatchComparisonDecision.INITIAL_CHAMPION
        expected_champion_after = champion.scope.candidate_revision_id
        expected_reason = "initial_champion"
    else:
        expected_decision = assessment.decision
        expected_champion_after = assessment.champion_after_revision_id
        expected_reason = assessment.reason

    invalid_fields: list[str] = []
    if not math.isclose(
        comparison.score_delta,
        assessment.score_delta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        invalid_fields.append("score_delta")
    if comparison.comparison_contract_digest != assessment.comparison_contract_digest:
        invalid_fields.append("comparison_contract_digest")
    if comparison.safety_gate_passed is not assessment.safety_gate_passed:
        invalid_fields.append("safety_gate_passed")
    if (
        comparison.cell_regression_gate_passed
        is not assessment.cell_regression_gate_passed
    ):
        invalid_fields.append("cell_regression_gate_passed")
    # This value is a frozen policy constant, not a noisy measurement.  An
    # approximate comparison would let the entire v3 epsilon (1e-12) disappear
    # from, or be doubled in, durable audit evidence without replay noticing.
    if comparison.minimum_score_delta != float(minimum_score_delta):
        invalid_fields.append("minimum_score_delta")
    if comparison.decision is not expected_decision:
        invalid_fields.append("decision")
    if comparison.champion_after_revision_id != expected_champion_after:
        invalid_fields.append("champion_after_revision_id")
    if comparison.reason != expected_reason:
        invalid_fields.append("reason")
    if invalid_fields:
        raise ValueError(
            "formal batch comparison host-owned assessment is invalid: "
            + ", ".join(invalid_fields)
        )


__all__ = [
    "LOCAL_CELL_REGRESSION_TOLERANCE",
    "LOCAL_MINIMUM_SCORE_DELTA",
    "POSITIVE_DELTA_MINIMUM_SCORE_DELTA",
    "LocalChallengerAssessment",
    "assess_local_challenger",
    "local_challenger_safety_reason",
    "validate_formal_batch_comparison",
]
