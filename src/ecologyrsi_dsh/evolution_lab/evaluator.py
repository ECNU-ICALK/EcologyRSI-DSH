"""Promotion rules for external plugin experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    cohort_id: str
    scientific_score: float
    per_cell_scores: Mapping[str, float]
    physical_violations: int
    skill_success_rate: float
    tool_success_rate: float
    latency_ms: float
    cost_units: float
    stable: bool
    sample_count: int
    failure_attribution: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.cohort_id:
            raise ValueError("cohort_id is required")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.physical_violations < 0:
            raise ValueError("physical_violations cannot be negative")
        for name in ("skill_success_rate", "tool_success_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if any(not isinstance(key, str) for key in self.per_cell_scores):
            raise ValueError("per_cell_scores keys must be strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "scientific_score": self.scientific_score,
            "per_cell_scores": dict(self.per_cell_scores),
            "physical_violations": self.physical_violations,
            "skill_success_rate": self.skill_success_rate,
            "tool_success_rate": self.tool_success_rate,
            "latency_ms": self.latency_ms,
            "cost_units": self.cost_units,
            "stable": self.stable,
            "sample_count": self.sample_count,
            "failure_attribution": dict(self.failure_attribution or {}),
        }


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    status: str
    delta: float
    certification_eligible: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "delta": self.delta,
            "certification_eligible": self.certification_eligible,
            "reasons": list(self.reasons),
        }


def evaluate_candidate(
    candidate: EvaluationReport,
    incumbent: EvaluationReport,
    *,
    practical_delta: float = 0.005,
    min_skill_success: float = 0.95,
    min_tool_success: float = 0.95,
) -> PromotionDecision:
    """Compare two reports without mixing cohorts or hiding regressions."""

    reasons: list[str] = []
    delta = float(candidate.scientific_score) - float(incumbent.scientific_score)
    if candidate.cohort_id != incumbent.cohort_id:
        return PromotionDecision("rejected", delta, False, ("cohort_mismatch",))
    if candidate.physical_violations:
        reasons.append("physical_constraint_violation")
    if not candidate.stable:
        reasons.append("unstable_resampling")
    if candidate.skill_success_rate < min_skill_success:
        reasons.append("skill_reliability_below_threshold")
    if candidate.tool_success_rate < min_tool_success:
        reasons.append("tool_reliability_below_threshold")
    for cell, baseline in incumbent.per_cell_scores.items():
        if cell not in candidate.per_cell_scores or candidate.per_cell_scores[cell] < baseline:
            reasons.append(f"cell_regression:{cell}")
    if delta <= practical_delta:
        reasons.append("practical_delta_not_reached")
    if reasons:
        return PromotionDecision("rejected", delta, False, tuple(reasons))
    if candidate.scientific_score < 0:
        return PromotionDecision(
            "search_winner",
            delta,
            False,
            ("relative_improvement_only", "absolute_score_below_zero"),
        )
    return PromotionDecision(
        "certification_eligible",
        delta,
        True,
        ("same_cohort_improvement", "all_cells_non_regressed", "reliability_gate_passed"),
    )
