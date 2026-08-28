"""Deterministic same-cohort comparison for an adaptive generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from ..core.trajectory import (
    GenerationComparison,
    HoldoutArm,
    HoldoutEvaluation,
)
from .fitness import FitnessProfile, assess_generation_selection


PROMOTION_CELL_REGRESSION_FLOOR = -0.01
PROMOTION_REQUIRED_COVERAGE = 0.95


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _cell_map(
    evaluation: HoldoutEvaluation,
) -> tuple[dict[tuple[str, int], Mapping[str, Any]], bool]:
    """Return the reported cells and whether their row identity is unambiguous.

    A mapping alone cannot distinguish an exact grid from one containing a
    duplicate key because the later row overwrites the earlier one.  Promotion
    is fail-closed, so malformed, duplicate, and non-cell rows make the grid
    inexact even when the surviving keys happen to look complete.
    """

    rows = evaluation.metrics.get("targets")
    if not isinstance(rows, (list, tuple)):
        return {}, False
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return {}, False
        target = row.get("target")
        horizon = row.get("horizon_hours")
        if (
            not isinstance(target, str)
            or not target
            or isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon < 1
        ):
            return {}, False
        key = (target, horizon)
        if key in result:
            return {}, False
        result[key] = row
    return result, True


def _plain_json(value: Any) -> Any:
    """Thaw immutable MappingProxyType metrics for the legacy promotion helper."""

    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _promotion_view(evaluation: HoldoutEvaluation) -> Any:
    return SimpleNamespace(
        evaluation_id=evaluation.evaluation_id,
        candidate_id=evaluation.scope.candidate_id,
        score=evaluation.score,
        evaluator_digest=evaluation.evaluator_digest,
        metrics=_plain_json(evaluation.metrics),
    )


def _cell_gate(
    evaluation: HoldoutEvaluation,
    incumbent: HoldoutEvaluation,
    expected: set[tuple[str, int]],
) -> dict[str, Any]:
    current, current_well_formed = _cell_map(evaluation)
    baseline, baseline_well_formed = _cell_map(incumbent)
    deltas: dict[str, float] = {}
    failures: list[str] = []
    current_complete = current_well_formed and set(current) == expected
    baseline_complete = baseline_well_formed and set(baseline) == expected
    if not expected or not current_complete or not baseline_complete:
        if not current_complete:
            failures.append("objective_grid_incomplete")
        if not baseline_complete:
            failures.append("incumbent_objective_grid_incomplete")
        return {
            "complete": False,
            "coverage_pass": False,
            "no_regression": False,
            "worst_cell_delta": None,
            "failures": failures or ["objective_grid_incomplete"],
        }
    for key in sorted(expected):
        candidate_row = current[key]
        incumbent_row = baseline[key]
        candidate_skill = _finite_number(candidate_row.get("skill_score"))
        incumbent_skill = _finite_number(incumbent_row.get("skill_score"))
        candidate_coverage = _finite_number(
            candidate_row.get("sample_execution_coverage", candidate_row.get("coverage"))
        )
        incumbent_coverage = _finite_number(
            incumbent_row.get("sample_execution_coverage", incumbent_row.get("coverage"))
        )
        if candidate_skill is None or incumbent_skill is None:
            failures.append(f"cell_skill_missing:{key[0]}:{key[1]}")
        else:
            deltas[f"{key[0]}@{key[1]}h"] = candidate_skill - incumbent_skill
        if candidate_coverage is None or candidate_coverage < PROMOTION_REQUIRED_COVERAGE:
            failures.append(f"cell_coverage_insufficient:{key[0]}:{key[1]}")
        if incumbent_coverage is None or incumbent_coverage < PROMOTION_REQUIRED_COVERAGE:
            failures.append(f"incumbent_cell_coverage_insufficient:{key[0]}:{key[1]}")
    worst = min(deltas.values()) if deltas else None
    if worst is None or worst < PROMOTION_CELL_REGRESSION_FLOOR:
        failures.append("cell_regression")
    return {
        "complete": True,
        "coverage_pass": not any("coverage" in item for item in failures),
        "no_regression": worst is not None and worst >= PROMOTION_CELL_REGRESSION_FLOOR,
        "worst_cell_delta": worst,
        "cell_deltas": deltas,
        "failures": failures,
    }


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
    overall_coverage = _finite_number(
        evaluation.metrics.get(
            "objective_weight_coverage",
            evaluation.metrics.get("sample_execution_coverage"),
        )
    )
    if overall_coverage is None or overall_coverage < PROMOTION_REQUIRED_COVERAGE:
        coverage_pass = False
    passed = bool(evaluation.passed)
    return {
        "passed": passed,
        "constraint_violations": constraints,
        "overall_coverage": overall_coverage,
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
    fitness_profile: FitnessProfile | None = None,
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
    incumbent = by_arm[HoldoutArm.INCUMBENT]
    if len({item.evaluator_digest for item in evaluations}) != 1:
        raise ValueError("holdout evaluations must share one evaluator digest")
    profile = fitness_profile or FitnessProfile()
    expected_grid = {
        (target, horizon)
        for target in profile.expected_targets
        for horizon in profile.expected_horizons
    }
    assessments = assess_generation_selection(
        tuple(_promotion_view(item) for item in finalist_evaluations),
        _promotion_view(incumbent),
        profile,
    )
    assessment_by_candidate = {item.candidate_id: item for item in assessments}
    finalist_gates: dict[str, dict[str, Any]] = {}
    eligible_finalists = []
    for item in finalist_evaluations:
        scientific_gate = _gate(item)
        cell_gate = _cell_gate(item, incumbent, expected_grid)
        selection = assessment_by_candidate[item.scope.candidate_id]
        stability_floor = selection.selection_stability_floor
        delta = item.score - incumbent.score
        eligible = bool(
            scientific_gate["eligible"]
            and cell_gate["complete"]
            and cell_gate["coverage_pass"]
            and cell_gate["no_regression"]
            and selection.primary_selection_gate
            # The shared max-T family controls uncertainty across siblings;
            # retain the predeclared practical score delta as a separate
            # effect-size gate so a statistically stable tiny gain is not
            # promoted.
            and delta > profile.selection_minimum_score_delta
        )
        gate = {
            **scientific_gate,
            "eligible": eligible,
            "complete_objective_grid": cell_gate["complete"],
            "coverage_pass": cell_gate["coverage_pass"],
            "no_cell_regression": cell_gate["no_regression"],
            "worst_cell_delta": cell_gate["worst_cell_delta"],
            "cell_deltas": cell_gate.get("cell_deltas", {}),
            "promotion_assessment": selection.to_dict(),
            "stability_lower_bound": stability_floor,
            "failures": list(cell_gate.get("failures", ()))
            + (
                []
                if selection.primary_selection_gate
                and delta > profile.selection_minimum_score_delta
                else [
                    selection.status
                    if not selection.primary_selection_gate
                    else "below_practical_score_delta"
                ]
            ),
        }
        finalist_gates[item.scope.holdout_arm.value] = gate
        if eligible:
            eligible_finalists.append((item, gate))
    eligible_finalists.sort(
        key=lambda pair: (
            -(pair[1]["stability_lower_bound"] if pair[1]["stability_lower_bound"] is not None else float("-inf")),
            -(pair[0].score - incumbent.score),
            -(pair[1]["worst_cell_delta"] if pair[1]["worst_cell_delta"] is not None else float("-inf")),
            pair[0].scope.candidate_id,
            pair[0].scope.candidate_revision_id,
        )
    )
    incumbent_gate = _gate(incumbent)
    selected = eligible_finalists[0][0] if eligible_finalists else incumbent
    if incumbent_candidate_id is not None and incumbent.scope.candidate_id != incumbent_candidate_id:
        raise ValueError("incumbent holdout arm does not match incumbent candidate")

    incumbent_delta = selected.score - incumbent.score
    gate_results = {
        "schema_version": "ecologyrsi-dsh.generation-comparison/1",
        "arms": {
            arm.value: (
                finalist_gates[arm.value]
                if arm in {HoldoutArm.FINALIST_1, HoldoutArm.FINALIST_2}
                else incumbent_gate
            )
            for arm in HoldoutArm
        },
        "eligible_finalist_count": len(eligible_finalists),
        "selected_arm": selected.scope.holdout_arm.value if selected.scope.holdout_arm else None,
        "selected_score": selected.score,
        "incumbent_score": incumbent.score,
        "delta_to_incumbent": incumbent_delta,
        "incumbent_gate_pass": incumbent_gate["eligible"],
        "fitness_profile_digest": profile.profile_digest,
        "selection_rule": "shared_centered_max_t_then_scientific_cell_gates_then_stability_lower_bound_delta_worst_cell_candidate_revision_else_incumbent",
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
