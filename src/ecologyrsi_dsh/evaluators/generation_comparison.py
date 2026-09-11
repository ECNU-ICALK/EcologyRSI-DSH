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
from .agent_stability import paired_stability_gate
from .fitness import FitnessProfile, assess_generation_selection
from ..core.search_policy import PAIRED_EXECUTION_QUALIFICATION
from ..core.finalist_review import FINALIST_REVIEW_QUALIFICATION
from ..core.agent_prediction import successful_agent_provenance_passes
from ..evolution.execution_qualification import paired_scoring_evidence_complete


PROMOTION_CELL_REGRESSION_FLOOR = -FitnessProfile().selection_cell_regression_tolerance
PROMOTION_REQUIRED_COVERAGE = FitnessProfile().selection_minimum_coverage
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
    profile: FitnessProfile,
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
        if candidate_coverage is None or candidate_coverage < profile.selection_minimum_coverage:
            failures.append(f"cell_coverage_insufficient:{key[0]}:{key[1]}")
        if incumbent_coverage is None or incumbent_coverage < profile.selection_minimum_coverage:
            failures.append(f"incumbent_cell_coverage_insufficient:{key[0]}:{key[1]}")
    worst = min(deltas.values()) if deltas else None
    # This is the *selection* no-regression check: a per-cell skill delta against
    # the incumbent, tolerated up to `selection_cell_regression_tolerance`. It is
    # a heuristic for choosing among siblings, and it is deliberately not the
    # certification gate -- that is `per_cell_noninferiority@1`, which compares
    # each cell against the frozen baseline and derives its own boundary from the
    # paired bootstrap. Two different references, two different purposes; the
    # scope key below exists so a reader of the evidence never has to guess which
    # one produced a given `no_regression`.
    if worst is None or worst < -profile.selection_cell_regression_tolerance:
        failures.append("cell_regression")
    return {
        "complete": True,
        "coverage_pass": not any("coverage" in item for item in failures),
        "no_regression": worst is not None and worst >= -profile.selection_cell_regression_tolerance,
        "no_regression_scope": "generation_selection_heuristic",
        "no_regression_reference": "incumbent_cell_skill",
        "cell_regression_tolerance": profile.selection_cell_regression_tolerance,
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
    return successful_agent_provenance_passes(evaluation.metrics.get("sample_execution"))


def _gate(
    evaluation: HoldoutEvaluation,
    *,
    legacy_runtime_v2_shape: bool = False,
    minimum_coverage: float = PROMOTION_REQUIRED_COVERAGE,
) -> dict[str, Any]:
    constraints = _constraint_violations(
        evaluation,
        legacy_runtime_v2_shape=legacy_runtime_v2_shape,
    )
    coverage_pass = _coverage_pass(evaluation)
    sample = evaluation.metrics.get("sample_execution")
    # Objective weights can sum to one even when most executions failed.
    overall_coverage = _finite_number(
        sample.get("coverage") if isinstance(sample, Mapping)
        else evaluation.metrics.get("sample_execution_coverage")
    )
    if isinstance(sample, Mapping):
        attempted = _finite_number(sample.get("attempted_origin_samples"))
        succeeded = _finite_number(sample.get("succeeded_origin_samples"))
        if (attempted is not None and succeeded is not None and attempted > 0
                and attempted.is_integer() and succeeded.is_integer() and 0 <= succeeded <= attempted):
            overall_coverage = succeeded / attempted
        elif 'attempted_origin_samples' in sample or 'succeeded_origin_samples' in sample:
            overall_coverage = None
    if legacy_runtime_v2_shape:
        overall_coverage = _finite_number(evaluation.metrics.get("objective_weight_coverage"))
    if overall_coverage is None or overall_coverage < minimum_coverage:
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
    require_paired_strict_chain: bool = False,
    finalist_reviews: Mapping[str, Mapping[str, Any]] | None = None,
    quick_experiment: bool = False,
) -> GenerationComparison:
    """Build one immutable three-arm comparison without model-authored ranking."""

    if not isinstance(require_paired_strict_chain, bool):
        raise TypeError("require_paired_strict_chain must be a bool")
    if require_paired_strict_chain and legacy_runtime_v2_shape:
        raise ValueError("paired execution qualification cannot use a legacy gate shape")
    evaluations = tuple(holdout_evaluations)
    expected_arms = {HoldoutArm.FINALIST_1, HoldoutArm.INCUMBENT} if quick_experiment else set(HoldoutArm)
    if quick_experiment:
        positive_delta_search = True
    if len(evaluations) != len(expected_arms):
        raise ValueError("generation comparison requires exactly three holdout evaluations")
    arms = {item.scope.holdout_arm for item in evaluations}
    if arms != expected_arms:
        raise ValueError("generation comparison requires finalist_1, finalist_2, and incumbent")
    if any(item.scope.run_id != run_id for item in evaluations):
        raise ValueError("holdout evaluation belongs to another run")
    if any(item.scope.generation != generation for item in evaluations):
        raise ValueError("holdout evaluation belongs to another generation")
    if any(item.scope.cohort_digest != cohort_digest for item in evaluations):
        raise ValueError("holdout evaluations must use one frozen cohort")

    by_arm = {item.scope.holdout_arm: item for item in evaluations}
    finalist_evaluations = [by_arm[arm] for arm in HoldoutArm if arm in arms and arm is not HoldoutArm.INCUMBENT]
    incumbent = by_arm[HoldoutArm.INCUMBENT]
    if len({item.evaluator_digest for item in evaluations}) != 1:
        raise ValueError("holdout evaluations must share one evaluator digest")
    profile = fitness_profile or FitnessProfile()
    expected_grid = {
        (target, horizon)
        for target in profile.expected_targets
        for horizon in profile.expected_horizons
    }
    if finalist_reviews is not None:
        if legacy_runtime_v2_shape or set(finalist_reviews) != {item.scope.holdout_arm.value for item in finalist_evaluations}:
            raise ValueError("independent review requires the two current finalist arms")
        for item in finalist_evaluations:
            review = finalist_reviews[item.scope.holdout_arm.value]
            if (review.get("candidate_id") != item.scope.candidate_id
                    or review.get("candidate_revision_id") != item.scope.candidate_revision_id
                    or review.get("evaluation_scope_digest") != item.scope.scope_key):
                raise ValueError("independent review belongs to another finalist revision")
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
            minimum_coverage=profile.selection_minimum_coverage,
        )
        cell_gate = _cell_gate(item, incumbent, expected_grid, profile)
        selection = assessment_by_candidate[item.scope.candidate_id]
        stability_floor = selection.selection_stability_floor
        delta = item.score - incumbent.score
        strict_chain_pass = bool(
            _strict_chain_pass(item) and _strict_chain_pass(incumbent)
        )
        scoring_evidence_complete = (
            paired_scoring_evidence_complete(incumbent, item)
            if require_paired_strict_chain else True
        )
        review = finalist_reviews[item.scope.holdout_arm.value] if finalist_reviews is not None else None
        review_pass = review is None or (
            review.get("judge_status") == "completed" and review.get("judge_accepted") is True
        )
        certification_eligible = bool(
            review_pass and challenger_promotion_allowed
            and scientific_gate["eligible"]
            # Guarded scientific comparison cannot benefit from an incumbent's
            # transport/chain failure through the coverage-penalized objective.
            and (not require_paired_strict_chain or strict_chain_pass)
            and scoring_evidence_complete
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
        cell_values_complete = bool(
            cell_gate["complete"]
            and len(cell_gate.get("cell_deltas", {})) == len(expected_grid)
        )
        search_failures: list[str] = []
        if not review_pass:
            search_failures.append("independent_review_not_accepted")
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
        if quick_experiment:
            if delta <= profile.selection_minimum_score_delta:
                search_failures.append("below_practical_score_delta")
            if not cell_gate["no_regression"]:
                search_failures.append("cell_regression")
            if not paired_scoring_evidence_complete(incumbent, item):
                search_failures.append("paired_scoring_evidence_incomplete")
        search_eligible = bool(
            not search_failures
            if positive_delta_search
            else certification_eligible
        )
        inference_stability = None
        if item.metrics.get("prediction_owner") == "sample_agent" or item.metrics.get("agent_inference_stability") is not None or incumbent.metrics.get("agent_inference_stability") is not None:
            inference_stability = paired_stability_gate(item, incumbent, minimum_delta=profile.selection_minimum_score_delta)
            certification_eligible = certification_eligible and inference_stability["passed"]
        if not positive_delta_search:
            # All epoch gates must settle before selecting the next parent.
            # A successful first replica cannot bypass a failing second one.
            search_eligible = certification_eligible
        certification_failures = list(cell_gate.get("failures", ()))
        if quick_experiment:
            certification_eligible = False
            certification_failures.append("quick_experiment_requires_independent_certification")
        if inference_stability is not None and not inference_stability["passed"]:
            certification_failures.append(inference_stability["reason"])
        if not review_pass:
            certification_failures.append("independent_review_not_accepted")
        if require_paired_strict_chain and not strict_chain_pass:
            certification_failures.append("paired_strict_agent_chain_failed")
        if require_paired_strict_chain and not scoring_evidence_complete:
            certification_failures.append("paired_scoring_evidence_incomplete")
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
                **({"paired_scoring_evidence_complete": scoring_evidence_complete}
                   if require_paired_strict_chain else {}),
                "search_failures": search_failures,
                "certification_failures": certification_failures,
                "agent_inference_stability": inference_stability,
                "failures": (
                    search_failures
                    if positive_delta_search
                    else certification_failures
                ),
            }
        if review is not None:
            gate["judge_available"] = review.get("judge_status") == "completed"
            gate["judge_accepted"] = review_pass
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
        minimum_coverage=profile.selection_minimum_coverage,
    )
    incumbent_scoring_complete = (
        paired_scoring_evidence_complete(incumbent, incumbent)
        if require_paired_strict_chain else True
    )
    if require_paired_strict_chain and (not _strict_chain_pass(incumbent) or not incumbent_scoring_complete):
        incumbent_scientific_gate["eligible"] = False
    incumbent_gate = (
        incumbent_scientific_gate
        if legacy_runtime_v2_shape
        else {
            **incumbent_scientific_gate,
            "search_eligible": True,
            "certification_eligible": incumbent_scientific_gate["eligible"],
            "search_failures": [],
            "certification_failures": (
                (["paired_strict_agent_chain_failed"] if not _strict_chain_pass(incumbent) else [])
                + (["paired_scoring_evidence_incomplete"] if not incumbent_scoring_complete else [])
                if require_paired_strict_chain else []
            ),
            **({"paired_scoring_evidence_complete": incumbent_scoring_complete}
               if require_paired_strict_chain else {}),
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
        **({"finalist_review_qualification": FINALIST_REVIEW_QUALIFICATION,
            "finalist_reviews": _plain_json(finalist_reviews)} if finalist_reviews is not None else {}),
        **({"paired_execution_qualification": PAIRED_EXECUTION_QUALIFICATION}
           if require_paired_strict_chain else {}),
        "arms": {
            arm.value: (
                finalist_gates[arm.value]
                if arm in {HoldoutArm.FINALIST_1, HoldoutArm.FINALIST_2}
                else incumbent_gate
            )
            for arm in HoldoutArm if arm in arms
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
    if quick_experiment:
        gate_results["experiment_protocol"] = "quick_adaptive_epoch@1"
        gate_results["selection_rule"] = "complete_paired_practical_delta_cell_nonregression_else_incumbent"
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
