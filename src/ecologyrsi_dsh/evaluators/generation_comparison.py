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
POSITIVE_SEARCH_MINIMUM_SCORE_DELTA = 1e-12


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


def _constraint_violations(
    evaluation: HoldoutEvaluation,
    *,
    legacy_runtime_v2_shape: bool = False,
) -> int:
    value = evaluation.metrics.get(
        "constraint_violations",
        0 if legacy_runtime_v2_shape else None,
    )
    if legacy_runtime_v2_shape:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 1
        return max(0, int(value))
    number = _finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        # Evaluator counts are a fail-closed safety input.  Fractional,
        # negative, boolean, non-finite, and non-numeric values cannot mean
        # "zero violations".
        return 1
    return int(number)


def _coverage_pass(evaluation: HoldoutEvaluation) -> bool:
    metrics = evaluation.metrics
    sample_execution = metrics.get("sample_execution")
    if isinstance(sample_execution, Mapping):
        coverage = sample_execution.get("coverage_pass")
        if coverage is False:
            return False
    explicit = metrics.get("sample_execution_coverage_pass")
    return explicit is not False


def _strict_chain_pass(evaluation: HoldoutEvaluation) -> bool:
    sample_execution = evaluation.metrics.get("sample_execution")
    return bool(
        isinstance(sample_execution, Mapping)
        and sample_execution.get("strict_agent_chain_pass") is True
    )


def _gate(
    evaluation: HoldoutEvaluation,
    *,
    legacy_runtime_v2_shape: bool = False,
) -> dict[str, Any]:
    constraints = _constraint_violations(
        evaluation,
        legacy_runtime_v2_shape=legacy_runtime_v2_shape,
    )
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
    challenger_promotion_allowed: bool = True,
    positive_delta_search: bool = False,
    legacy_runtime_v2_shape: bool = False,
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
    search_eligible_finalists = []
    certification_eligible_finalists = []
    for item in finalist_evaluations:
        scientific_gate = _gate(
            item,
            legacy_runtime_v2_shape=legacy_runtime_v2_shape,
        )
        cell_gate = _cell_gate(item, incumbent, expected_grid)
        selection = assessment_by_candidate[item.scope.candidate_id]
        stability_floor = selection.selection_stability_floor
        delta = item.score - incumbent.score
        certification_eligible = bool(
            challenger_promotion_allowed
            and scientific_gate["eligible"]
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
        strict_chain_pass = bool(
            _strict_chain_pass(item) and _strict_chain_pass(incumbent)
        )
        cell_values_complete = bool(
            cell_gate["complete"]
            and len(cell_gate.get("cell_deltas", {})) == len(expected_grid)
        )
        search_failures: list[str] = []
        if scientific_gate["constraint_violations"] != 0:
            search_failures.append("constraint_violations")
        if not scientific_gate["coverage_pass"] or not cell_gate["coverage_pass"]:
            search_failures.append("coverage_failed")
        if not cell_gate["complete"] or not cell_values_complete:
            search_failures.append("objective_grid_incomplete")
        if not strict_chain_pass:
            search_failures.append("strict_agent_chain_failed")
        if delta <= POSITIVE_SEARCH_MINIMUM_SCORE_DELTA:
            search_failures.append("no_positive_score_delta")
        search_eligible = bool(
            not search_failures
            if positive_delta_search
            else certification_eligible
        )
        certification_failures = list(cell_gate.get("failures", ()))
        if not challenger_promotion_allowed:
            certification_failures.append("screening_exploration_only")
        if not scientific_gate["passed"]:
            certification_failures.append("scientific_gate_failed")
        if scientific_gate["constraint_violations"] != 0:
            certification_failures.append("constraint_violations")
        if not scientific_gate["coverage_pass"]:
            certification_failures.append("coverage_failed")
        if not selection.primary_selection_gate:
            certification_failures.append(selection.status)
        elif delta <= profile.selection_minimum_score_delta:
            certification_failures.append("below_practical_score_delta")
        if legacy_runtime_v2_shape:
            legacy_failures = list(cell_gate.get("failures", ()))
            if not challenger_promotion_allowed:
                legacy_failures.append("screening_exploration_only")
            if not selection.primary_selection_gate:
                legacy_failures.append(selection.status)
            elif delta <= profile.selection_minimum_score_delta:
                legacy_failures.append("below_practical_score_delta")
            gate = {
                **scientific_gate,
                "eligible": certification_eligible,
                "complete_objective_grid": cell_gate["complete"],
                "coverage_pass": cell_gate["coverage_pass"],
                "no_cell_regression": cell_gate["no_regression"],
                "worst_cell_delta": cell_gate["worst_cell_delta"],
                "cell_deltas": cell_gate.get("cell_deltas", {}),
                "promotion_assessment": selection.to_dict(),
                "stability_lower_bound": stability_floor,
                "failures": legacy_failures,
            }
        else:
            gate = {
                **scientific_gate,
                "eligible": search_eligible,
                "search_eligible": search_eligible,
                "certification_eligible": certification_eligible,
                "complete_objective_grid": cell_gate["complete"],
                "coverage_pass": cell_gate["coverage_pass"],
                "no_cell_regression": cell_gate["no_regression"],
                "worst_cell_delta": cell_gate["worst_cell_delta"],
                "cell_deltas": cell_gate.get("cell_deltas", {}),
                "promotion_assessment": selection.to_dict(),
                "stability_lower_bound": stability_floor,
                "strict_agent_chain_pass": strict_chain_pass,
                "search_failures": search_failures,
                "certification_failures": certification_failures,
                "failures": (
                    search_failures
                    if positive_delta_search
                    else certification_failures
                ),
            }
        finalist_gates[item.scope.holdout_arm.value] = gate
        if search_eligible:
            search_eligible_finalists.append((item, gate))
        if certification_eligible:
            certification_eligible_finalists.append((item, gate))
    certification_eligible_finalists.sort(
        key=lambda pair: (
            -(pair[1]["stability_lower_bound"] if pair[1]["stability_lower_bound"] is not None else float("-inf")),
            -(pair[0].score - incumbent.score),
            -(pair[1]["worst_cell_delta"] if pair[1]["worst_cell_delta"] is not None else float("-inf")),
            pair[0].scope.candidate_id,
            pair[0].scope.candidate_revision_id,
        )
    )
    search_eligible_finalists.sort(
        key=lambda pair: (
            -(pair[0].score - incumbent.score),
            -(
                pair[1]["stability_lower_bound"]
                if pair[1]["stability_lower_bound"] is not None
                else float("-inf")
            ),
            -(
                pair[1]["worst_cell_delta"]
                if pair[1]["worst_cell_delta"] is not None
                else float("-inf")
            ),
            pair[0].scope.candidate_id,
            pair[0].scope.candidate_revision_id,
        )
        if positive_delta_search
        else (
            -(
                pair[1]["stability_lower_bound"]
                if pair[1]["stability_lower_bound"] is not None
                else float("-inf")
            ),
            -(pair[0].score - incumbent.score),
            -(
                pair[1]["worst_cell_delta"]
                if pair[1]["worst_cell_delta"] is not None
                else float("-inf")
            ),
            pair[0].scope.candidate_id,
            pair[0].scope.candidate_revision_id,
        )
    )
    incumbent_scientific_gate = _gate(
        incumbent,
        legacy_runtime_v2_shape=legacy_runtime_v2_shape,
    )
    incumbent_gate = (
        incumbent_scientific_gate
        if legacy_runtime_v2_shape
        else {
            **incumbent_scientific_gate,
            "search_eligible": True,
            "certification_eligible": incumbent_scientific_gate["eligible"],
            "search_failures": [],
            "certification_failures": [],
            "strict_agent_chain_pass": _strict_chain_pass(incumbent),
        }
    )
    selected = (
        search_eligible_finalists[0][0]
        if search_eligible_finalists
        else incumbent
    )
    certification_selected = (
        certification_eligible_finalists[0][0]
        if certification_eligible_finalists
        else None
    )
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
        "eligible_finalist_count": len(search_eligible_finalists),
        "search_eligible_finalist_count": len(search_eligible_finalists),
        "certification_eligible_finalist_count": len(
            certification_eligible_finalists
        ),
        "selected_arm": selected.scope.holdout_arm.value if selected.scope.holdout_arm else None,
        "selected_score": selected.score,
        "incumbent_score": incumbent.score,
        "delta_to_incumbent": incumbent_delta,
        "incumbent_gate_pass": incumbent_gate["eligible"],
        "fitness_profile_digest": profile.profile_digest,
        "selection_policy": (
            "positive_delta_search@1"
            if positive_delta_search
            else "strict_certification@1"
        ),
        "selection_rule": (
            "positive_same_holdout_delta_then_stability_lower_bound_then_"
            "worst_cell_then_candidate_revision_else_incumbent"
            if positive_delta_search
            else "shared_centered_max_t_then_scientific_cell_gates_then_stability_lower_bound_delta_worst_cell_candidate_revision_else_incumbent"
        ),
        "challenger_promotion_allowed": challenger_promotion_allowed,
    }
    if not legacy_runtime_v2_shape:
        gate_results.update(
            {
                "certification_selected_arm": (
                    certification_selected.scope.holdout_arm.value
                    if certification_selected is not None
                    and certification_selected.scope.holdout_arm is not None
                    else None
                ),
                "selected_search_certification_status": (
                    "incumbent_retained"
                    if selected is incumbent
                    else "certification_selected"
                    if certification_selected is selected
                    else "certification_eligible_not_selected"
                    if finalist_gates[selected.scope.holdout_arm.value][
                        "certification_eligible"
                    ]
                    else "search_only"
                ),
            }
        )
    else:
        for field in (
            "search_eligible_finalist_count",
            "certification_eligible_finalist_count",
            "selection_policy",
        ):
            gate_results.pop(field, None)
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


__all__ = [
    "POSITIVE_SEARCH_MINIMUM_SCORE_DELTA",
    "build_generation_comparison",
]
