"""Deterministic same-cohort comparison for an adaptive generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..core.trajectory import (
    GenerationComparison,
    HoldoutArm,
    HoldoutEvaluation,
)


def _constraint_violations(evaluation: HoldoutEvaluation) -> int:
    value = evaluation.metrics.get("constraint_violations", 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1
    return max(0, int(value))


def _coverage_pass(evaluation: HoldoutEvaluation) -> bool:
    metrics = evaluation.metrics
    sample_execution = metrics.get("sample_execution")
    if isinstance(sample_execution, Mapping):
        coverage = sample_execution.get("coverage_pass")
        if coverage is False:
            return False
    explicit = metrics.get("sample_execution_coverage_pass")
    return explicit is not False


def _gate(evaluation: HoldoutEvaluation) -> dict[str, Any]:
    constraints = _constraint_violations(evaluation)
    coverage_pass = _coverage_pass(evaluation)
    passed = bool(evaluation.passed)
    return {
        "passed": passed,
        "constraint_violations": constraints,
        "coverage_pass": coverage_pass,
        "eligible": bool(passed and constraints == 0 and coverage_pass),
        "score": evaluation.score,
    }


def build_generation_comparison(
    *,
    run_id: str,
    generation: int,
    cohort_digest: str,
    holdout_evaluations: Sequence[HoldoutEvaluation],
    incumbent_candidate_id: str | None = None,
) -> GenerationComparison:
    """Build one immutable three-arm comparison without model-authored ranking."""

    evaluations = tuple(holdout_evaluations)
    if len(evaluations) != len(HoldoutArm):
        raise ValueError("generation comparison requires exactly three holdout evaluations")
    arms = {item.scope.holdout_arm for item in evaluations}
    if arms != set(HoldoutArm):
        raise ValueError("generation comparison requires finalist_1, finalist_2, and incumbent")
    if any(item.scope.run_id != run_id for item in evaluations):
        raise ValueError("holdout evaluation belongs to another run")
    if any(item.scope.generation != generation for item in evaluations):
        raise ValueError("holdout evaluation belongs to another generation")
    if any(item.scope.cohort_digest != cohort_digest for item in evaluations):
        raise ValueError("holdout evaluations must use one frozen cohort")

    by_arm = {item.scope.holdout_arm: item for item in evaluations}
    finalist_evaluations = [
        by_arm[HoldoutArm.FINALIST_1],
        by_arm[HoldoutArm.FINALIST_2],
    ]
    eligible_finalists = [
        item for item in finalist_evaluations if _gate(item)["eligible"]
    ]
    eligible_finalists.sort(
        key=lambda item: (-item.score, item.scope.candidate_id, item.scope.candidate_revision_id)
    )
    incumbent = by_arm[HoldoutArm.INCUMBENT]
    incumbent_gate = _gate(incumbent)
    selected = eligible_finalists[0] if eligible_finalists else incumbent
    if incumbent_candidate_id is not None and incumbent.scope.candidate_id != incumbent_candidate_id:
        raise ValueError("incumbent holdout arm does not match incumbent candidate")

    incumbent_delta = selected.score - incumbent.score
    gate_results = {
        "schema_version": "ecologyrsi-dsh.generation-comparison/1",
        "arms": {
            arm.value: _gate(by_arm[arm]) for arm in HoldoutArm
        },
        "eligible_finalist_count": len(eligible_finalists),
        "selected_arm": selected.scope.holdout_arm.value if selected.scope.holdout_arm else None,
        "selected_score": selected.score,
        "incumbent_score": incumbent.score,
        "delta_to_incumbent": incumbent_delta,
        "incumbent_gate_pass": incumbent_gate["eligible"],
        "selection_rule": "eligible_finalist_max_score_then_candidate_revision_id_else_incumbent",
    }
    return GenerationComparison(
        comparison_id=f"generation-comparison:{run_id}:{generation}",
        run_id=run_id,
        generation=generation,
        cohort_digest=cohort_digest,
        holdout_evaluations=evaluations,
        selected_candidate_id=selected.scope.candidate_id,
        selected_revision_id=selected.scope.candidate_revision_id,
        gate_results=gate_results,
    )


__all__ = ["build_generation_comparison"]
