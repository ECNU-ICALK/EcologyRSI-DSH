"""Bind the final decision to independent judgments of the actual finalists."""

from collections.abc import Mapping, Sequence
from typing import Any

from .models import Evaluation, digest
from .trajectory import EvaluationScope, HoldoutArm, HoldoutEvaluation


FINALIST_REVIEW_QUALIFICATION = "independent_finalist_review@1"


def requires_finalist_review(metadata: Mapping[str, Any]) -> bool:
    return bool(
        metadata.get("execution_protocol") == "dsh_native_plugin_evolution@1"
        and isinstance(metadata.get("review_model_id"), str)
        and metadata["review_model_id"].strip()
    )


def finalist_review_qualification_required(metadata, marker, *, new_decision=False):
    required = requires_finalist_review(metadata)
    if marker is not None and marker != FINALIST_REVIEW_QUALIFICATION:
        raise ValueError("unsupported finalist review qualification")
    if marker is not None and not required:
        raise ValueError("finalist review requires the frozen independent model")
    if new_decision and required and marker is None:
        raise ValueError("new generation comparison requires independent finalist review")
    return marker is not None


def finalist_review_evidence(
    holdouts: Sequence[HoldoutEvaluation],
    judgments: Mapping[str, Evaluation],
    review_model_id: str,
) -> dict[str, dict[str, Any]]:
    """Read already persisted judgments; never infer review from a score/pass."""
    evidence = {}
    for holdout in holdouts:
        arm = holdout.scope.holdout_arm
        if arm is HoldoutArm.INCUMBENT:
            continue
        evaluation = judgments.get(holdout.scope.candidate_id)
        if evaluation is None:
            raise ValueError("finalist is missing its durable judgment")
        if (
            evaluation.run_id != holdout.scope.run_id
            or evaluation.candidate_revision_id != holdout.scope.candidate_revision_id
            or EvaluationScope.from_dict(evaluation.evaluation_scope).scope_key != holdout.scope.scope_key
            or evaluation.score != holdout.score
            or evaluation.evaluator_digest != holdout.evaluator_digest
        ):
            raise ValueError("finalist judgment differs from the frozen holdout identity")
        metrics = evaluation.metrics
        status = metrics.get("judge_status")
        accepted = metrics.get("judge_accepted")
        if status not in {"completed", "unavailable"} or type(accepted) is not bool:
            raise ValueError("finalist judgment is not a terminal review")
        if metrics.get("judge_model_id") != review_model_id:
            raise ValueError("finalist judgment uses a different independent model")
        if status != "completed" and accepted:
            raise ValueError("unavailable finalist review cannot accept a candidate")
        evidence[arm.value] = {
            "candidate_id": evaluation.candidate_id,
            "candidate_revision_id": evaluation.candidate_revision_id,
            "evaluation_scope_digest": holdout.scope.scope_key,
            "artifact_digest": evaluation.artifact_digest,
            "judged_evaluation_digest": digest(evaluation.to_dict()),
            "judge_status": status,
            "judge_model_id": review_model_id,
            "judge_accepted": accepted,
            "judge_result_digest": metrics.get("judge_result_digest"),
        }
    if set(evidence) != {HoldoutArm.FINALIST_1.value, HoldoutArm.FINALIST_2.value}:
        raise ValueError("independent review requires both finalists")
    return evidence
