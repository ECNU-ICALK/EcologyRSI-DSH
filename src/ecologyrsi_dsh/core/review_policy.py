"""Purpose-bound review: the model explains evidence; the Host owns decisions."""

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from .immutable import thaw_json
from .models import digest
from .trajectory import EvaluationScope


REVIEW_POLICY_VERSION = "ecologyrsi-dsh.candidate-review-policy/1"
EXPLORATION_REVIEW = "exploratory_candidate_evidence"
CERTIFICATION_REVIEW = "certification_candidate_evidence"


def _binding(evaluation: Any) -> dict[str, Any]:
    scope = getattr(evaluation, "scope", None)
    if scope is None:
        raw_scope = getattr(evaluation, "evaluation_scope", None)
        if raw_scope is None:
            # Non-adaptive native evaluations have no trajectory scope. Bind
            # their immutable receipt identity without inventing one; their
            # existing scientific and judgment gates remain unchanged.
            return {
                "evaluation_id": evaluation.evaluation_id,
                "run_id": evaluation.run_id,
                "candidate_id": evaluation.candidate_id,
                "candidate_revision_id": evaluation.candidate_revision_id,
                "evaluation_scope_digest": None,
                "evaluator_digest": evaluation.evaluator_digest,
                "artifact_digest": evaluation.artifact_digest,
            }
        scope = EvaluationScope.from_dict(raw_scope)
    return {
        "candidate_id": scope.candidate_id,
        "candidate_revision_id": scope.candidate_revision_id,
        "evaluation_scope_digest": scope.scope_key,
        "evaluator_digest": evaluation.evaluator_digest,
    }


def build_host_gate_summary(task: Any, evaluation: Any, incumbent: Any = None,
                            *, training_effect_evidence: Mapping | None = None) -> dict[str, Any]:
    """Expose the Host's actual gate inputs, without new statistical claims."""
    from ..evaluators.fitness import FitnessProfile
    from ..evaluators.generation_comparison import _cell_gate, _gate, _strict_chain_pass
    from ..evolution.execution_qualification import paired_scoring_evidence_complete
    from ..evolution.effect_contracts import hard_effect_failure
    from ..evolution.schedule import OptimizationSchedule

    raw_profile = task.metadata.get("fitness_profile")
    profile = FitnessProfile(**dict(raw_profile or {}))
    if raw_profile is not None and task.metadata.get("fitness_profile_digest", profile.profile_digest) != profile.profile_digest:
        raise ValueError("review fitness profile binding mismatch")
    raw_schedule = task.metadata.get("optimization_schedule")
    exploratory = bool(raw_schedule and OptimizationSchedule.from_dict(raw_schedule).quick)
    gate = _gate(evaluation, minimum_coverage=profile.selection_minimum_coverage)
    summary = {
        "schema_version": "ecologyrsi-dsh.review-host-gate-summary/1",
        "evaluation_binding": _binding(evaluation),
        "candidate_score": evaluation.score,
        "reported_scientific_pass": evaluation.metrics.get("scientific_pass", evaluation.passed),
        "certification_applicability": "not_applicable_to_exploration" if exploratory else "requires_separate_locked_holdout",
        "minimum_score_delta": profile.selection_minimum_score_delta,
        "minimum_coverage": profile.selection_minimum_coverage,
        "cell_regression_tolerance": profile.selection_cell_regression_tolerance,
        "candidate_coverage": gate["overall_coverage"],
        "constraint_violations": gate["constraint_violations"],
        "same_cohort_comparison": "unavailable",
    }
    if incumbent is None:
        return summary
    def scope_of(item):
        return getattr(item, "scope", None) or EvaluationScope.from_dict(item.evaluation_scope)
    scope, reference = scope_of(evaluation), scope_of(incumbent)
    if (scope.run_id, scope.generation, scope.cohort_digest, scope.origin_count, scope.phase) != (
            reference.run_id, reference.generation, reference.cohort_digest, reference.origin_count, reference.phase
    ) or evaluation.evaluator_digest != incumbent.evaluator_digest:
        raise ValueError("review comparison must use the same frozen cohort and evaluator")
    training_effect = (training_effect_evidence or {}).get(scope.holdout_arm.value, {})
    if training_effect and (training_effect.get("generation"), training_effect.get("candidate_id"), training_effect.get("candidate_revision_id")) != (
            scope.generation, scope.candidate_id, scope.candidate_revision_id):
        raise ValueError("review training effect evidence belongs to another revision")
    expected = {(target, horizon) for target in profile.expected_targets for horizon in profile.expected_horizons}
    cells = _cell_gate(evaluation, incumbent, expected, profile)
    delta = evaluation.score - incumbent.score
    def scoped_view(item, bound_scope):
        return SimpleNamespace(scope=bound_scope, metrics=item.metrics, score=item.score,
                               evaluator_digest=item.evaluator_digest)
    gates = {
        "hard_mutation_effect": hard_effect_failure(evaluation.metrics) is None and not training_effect.get("failure"),
        "practical_gain": delta > profile.selection_minimum_score_delta,
        "constraints": gate["constraint_violations"] == 0,
        "coverage": gate["coverage_pass"] and cells["coverage_pass"],
        "complete_objective_grid": cells["complete"] and len(cells.get("cell_deltas", {})) == len(expected),
        "cell_nonregression": cells["no_regression"],
        "strict_agent_chain": _strict_chain_pass(evaluation) and _strict_chain_pass(incumbent),
        "complete_paired_scoring_evidence": paired_scoring_evidence_complete(
            scoped_view(incumbent, reference), scoped_view(evaluation, scope)),
    }
    summary.update({
        "same_cohort_comparison": "available",
        "incumbent_binding": _binding(incumbent),
        "incumbent_score": incumbent.score,
        "score_delta": delta,
        "worst_cell_delta": cells["worst_cell_delta"],
        "cell_deltas": cells.get("cell_deltas", {}),
        "applicable_exploration_gates": gates,
        "exploration_gates_pass": all(gates.values()) if exploratory else None,
        "review_gate": "pending_advisory_review_availability_and_binding",
        **({"training_effect_evidence": thaw_json(training_effect)} if training_effect else {}),
    })
    return summary


def build_review_policy(task: Any, evaluation: Any, incumbent: Any = None,
                        *, training_effect_evidence: Mapping | None = None) -> dict[str, Any]:
    """Build the review envelope before calling the model, without exposing rows."""
    from ..evolution.schedule import OptimizationSchedule

    raw_schedule = task.metadata.get("optimization_schedule")
    exploratory = bool(raw_schedule and OptimizationSchedule.from_dict(raw_schedule).quick)
    body = {
        "schema_version": REVIEW_POLICY_VERSION,
        "purpose": EXPLORATION_REVIEW if exploratory else CERTIFICATION_REVIEW,
        "evidence_class": "exploratory_adaptive_data",
        "binding": _binding(evaluation),
        "host_gate_summary": build_host_gate_summary(task, evaluation, incumbent,
                                                     training_effect_evidence=training_effect_evidence),
        "decision_authority": "host_deterministic_gates",
        "review_authority": "advisory_evidence_findings",
        "scientific_pass_semantics": "certification_evidence_not_exploration_admission",
        "independent_certification": "not_applicable" if exploratory else "separate_locked_holdout_required",
        "instructions": [
            "Assess whether claims are supported by the supplied evidence; identify specific evidence for every concern.",
            "accepted is an advisory assessment, not permission to select or promote a candidate.",
            "Missing certification blocks, inference replicas, or scientific_pass alone cannot disqualify exploratory evidence.",
            "The Host enforces paired execution, coverage, physical constraints, practical gain, and cell regression limits.",
            "Do not invent missing sibling evidence, change a Host threshold, or claim independent certification.",
        ],
    }
    return {**body, "policy_digest": digest(body)}


def review_result_metrics(policy: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the Host-owned policy alongside the advisory model response."""
    if type(review.get("accepted")) is not bool:
        raise ValueError("candidate review accepted must be boolean")
    return {"judge_review_policy": thaw_json(policy),
            "judge_review_advisory_accepted": review["accepted"]}


def review_allows_host_decision(review: Mapping[str, Any] | None, evaluation: Any,
                               *, exploratory: bool) -> bool:
    """Keep review availability and identity strict, without delegating a veto.

    Unmarked archived judgments retain their recorded meaning. New judgments
    bind their explicit purpose and exact evaluation scope before opinions can
    be treated as advisory. A forged or mismatched envelope fails closed.
    """
    if review is None:
        return True
    policy = review.get("review_policy")
    if policy is None:
        return review.get("judge_status") == "completed" and review.get("judge_accepted") is True
    if not isinstance(policy, Mapping):
        raise ValueError("candidate review policy must be an object")
    policy = thaw_json(policy)
    body = {key: value for key, value in policy.items() if key != "policy_digest"}
    expected = EXPLORATION_REVIEW if exploratory else CERTIFICATION_REVIEW
    if (policy.get("schema_version") != REVIEW_POLICY_VERSION
            or policy.get("policy_digest") != digest(body)
            or policy.get("purpose") != expected
            or policy.get("binding") != _binding(evaluation)
            or policy.get("decision_authority") != "host_deterministic_gates"
            or policy.get("review_authority") != "advisory_evidence_findings"):
        raise ValueError("candidate review purpose or evidence binding mismatch")
    return review.get("judge_status") == "completed"
