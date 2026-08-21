"""Small, deterministic reporting helpers for a replayed run.

The reporting layer deliberately consumes a ``RunState`` projection and never
queries SQLite directly.  This keeps exported artifacts explainable and makes
the same summary usable by the CLI, HTTP adapter, and plugin.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..integrations.model_bindings import (
    HOST_PARAMETER_GENERATOR_ID,
    RULE_JUDGE_ID,
    builtin_model_configuration_digest,
)
from ..core.models import digest
from ..core.redaction import sanitize_public_value
from .trajectory import build_training_trajectory

_TRAINING_REDACTED_FIELDS = frozenset(
    {
        "baseline",
        "chain_of_thought",
        "api_key_env",
        "messages",
        "observations",
        "observed",
        "predicted",
        "prediction_preview",
        "private_reasoning",
        "prompt",
        "raw",
        "raw_rows",
        "reasoning",
        "rows",
        "samples",
        "timestamps",
        "原始数据",
        "原始行",
        "思维链",
        "私密推理",
    }
)


def _training_safe_value(value: Any, *, depth: int = 0) -> Any:
    """Copy bounded JSON values while removing fields unsafe for training projection."""

    return sanitize_public_value(
        value,
        extra_sensitive_keys=_TRAINING_REDACTED_FIELDS,
        depth=depth,
        max_depth=8,
        text_limit=None,
        sequence_limit=128,
    )
def _applied_interventions(state: Any, proposal_id: str) -> list[dict[str, Any]]:
    receipt_by_id: dict[str, dict[str, Any]] = {}
    for event in state.events:
        if event.kind == "ProposalSubmitted":
            proposal = event.payload.get("proposal")
            if not isinstance(proposal, Mapping) or proposal.get("proposal_id") != proposal_id:
                continue
            for receipt in event.payload.get("intervention_receipts", []):
                if isinstance(receipt, Mapping) and receipt.get("intervention_id"):
                    receipt_by_id[str(receipt["intervention_id"])] = _training_safe_value(receipt)
            continue
        if event.kind != "HumanInterventionApplied":
            continue
        if event.payload.get("proposal_id") != proposal_id:
            continue
        intervention_id = str(event.payload.get("intervention_id", ""))
        receipt_by_id[intervention_id] = _training_safe_value(event.payload)
    result = []
    for item in state.interventions:
        if item.applied_proposal_id != proposal_id and item.intervention_id not in receipt_by_id:
            continue
        receipt = receipt_by_id.get(item.intervention_id, {})
        if "application_status" not in receipt:
            enforced = item.kind.value in {"parameter_override", "parent_selection"}
            receipt = {
                "recorded": True,
                "applied": enforced,
                "enforced": enforced,
                "application_status": "enforced" if enforced else "recorded",
                "reason": (
                    "旧版事件可验证该人工选择已执行。"
                    if enforced
                    else "旧版事件没有参数作用收据，不能验证该自由文本已执行。"
                ),
            }
        result.append(
            {
                "intervention_id": item.intervention_id,
                "kind": item.kind.value,
                "message": _training_safe_value(item.message),
                "parameter_overrides": _training_safe_value(item.parameter_overrides),
                "target_candidate_id": item.target_candidate_id,
                "recorded": bool(receipt.get("recorded", True)),
                "applied": bool(receipt.get("applied", False)),
                "enforced": bool(receipt.get("enforced", False)),
                "application_status": str(
                    receipt.get("application_status", "recorded")
                ),
                "application": {
                    key: value
                    for key, value in receipt.items()
                    if key
                    not in {
                        "intervention_id",
                        "proposal_id",
                        "recorded",
                        "applied",
                        "enforced",
                        "application_status",
                    }
                },
            }
        )
    return result


def _parent_parameters(state: Any, parent_candidate_id: str | None) -> dict[str, Any] | None:
    if parent_candidate_id is None:
        return None
    parent = state.candidate(parent_candidate_id)
    proposal = state.proposal(parent.proposal_id)
    return _training_safe_value(proposal.changes)


def _candidate_source_event_seq(state: Any, candidate_id: str, proposal_id: str) -> int:
    relevant = _candidate_source_events(state, candidate_id, proposal_id)
    return max((event.seq for event in relevant), default=0)


def _candidate_source_events(
    state: Any, candidate_id: str, proposal_id: str
) -> list[Any]:
    generation = state.candidate(candidate_id).generation
    intervention_ids = {
        str(event.payload.get("intervention_id"))
        for event in state.events
        if event.kind == "HumanInterventionApplied"
        and event.payload.get("proposal_id") == proposal_id
    }
    relevant = []
    for event in state.events:
        payload = event.payload
        nested = None
        if event.kind == "ProposalSubmitted":
            nested = payload.get("proposal")
        elif event.kind == "CandidateSpawned":
            nested = payload.get("candidate")
        elif event.kind == "ArtifactRecorded":
            nested = payload.get("artifact")
        elif event.kind == "EvaluationRecorded":
            nested = payload.get("evaluation")
        elif event.kind == "PromotionDecided":
            nested = payload.get("promotion")
        matches_nested = isinstance(nested, Mapping) and (
            nested.get("candidate_id") == candidate_id
            or nested.get("proposal_id") == proposal_id
        )
        matches_failure = (
            event.kind in {"CandidateFailed", "CandidateMarkedDuplicate"}
            and payload.get("candidate_id") == candidate_id
        )
        matches_generation = (
            event.kind in {
                "GenerationBatchStarted",
                "GenerationKnowledgeRetrieved",
                "GenerationKnowledgeAssessed",
                "GenerationAnalyzed",
                "GenerationChampionSelected",
            }
            and (
                payload.get("generation") == generation
                or isinstance(payload.get("batch"), Mapping)
                and payload["batch"].get("generation") == generation
                or isinstance(payload.get("analysis"), Mapping)
                and payload["analysis"].get("generation") == generation
                or isinstance(payload.get("knowledge_snapshot"), Mapping)
                and payload["knowledge_snapshot"].get("generation") == generation
                or isinstance(payload.get("knowledge_assessment"), Mapping)
                and payload["knowledge_assessment"].get("generation") == generation
            )
        )
        matches_intervention = (
            event.kind == "HumanInterventionApplied"
            and payload.get("proposal_id") == proposal_id
        )
        matches_recorded_intervention = (
            event.kind == "HumanInterventionRecorded"
            and isinstance(payload.get("intervention"), Mapping)
            and str(payload["intervention"].get("intervention_id"))
            in intervention_ids
        )
        matches_stage = (
            event.kind == "EvolutionStageRecorded"
            and payload.get("generation") == generation
            and payload.get("proposal_id") in (None, proposal_id)
            and payload.get("candidate_id") in (None, candidate_id)
        )
        if (
            matches_nested
            or matches_failure
            or matches_generation
            or matches_intervention
            or matches_recorded_intervention
            or matches_stage
        ):
            relevant.append(event)
    return relevant


def _event_receipts(
    state: Any, candidate_id: str, proposal_id: str
) -> list[dict[str, Any]]:
    return [
        {
            "seq": event.seq,
            "event_id": event.event_id,
            "kind": event.kind,
            "payload_digest": digest(event.payload),
        }
        for event in _candidate_source_events(state, candidate_id, proposal_id)
    ]


def _candidate_failure_reason(state: Any, candidate_id: str) -> str | None:
    for event in reversed(state.events):
        if (
            event.kind == "CandidateFailed"
            and event.payload.get("candidate_id") == candidate_id
        ):
            return _training_safe_value(event.payload.get("reason"))
    return None


def training_assets(state: Any) -> list[dict[str, Any]]:
    """Derive one immutable, redacted evolution trajectory per candidate."""

    task = state.task_manifest
    metadata = dict(task.metadata)
    result: list[dict[str, Any]] = []
    for candidate in state.candidates:
        proposal = state.proposal(candidate.proposal_id)
        artifact = state.artifact_for(candidate.candidate_id)
        evaluation = state.evaluation_for(candidate.candidate_id)
        promotion = state.promotion_for(candidate.candidate_id)
        interventions = _applied_interventions(state, proposal.proposal_id)

        analysis = state.analysis_for(candidate.generation)
        batch = state.batch_for(candidate.generation)
        knowledge = state.knowledge_for(candidate.generation)
        knowledge_summary = (
            {
                "snapshot_digest": knowledge.snapshot_digest,
                "query_terms": list(knowledge.query_terms),
                "adopted_knowledge": [
                    {
                        "knowledge_id": item.knowledge_id,
                        "title": item.title,
                        "source_url": item.source_url,
                        "capability_kind": item.capability_kind,
                        "capability_id": item.capability_id,
                    }
                    for item in knowledge.executable_cards
                ],
            }
            if knowledge is not None
            else None
        )
        ranking = next(
            (
                dict(item)
                for item in (analysis.ranking if analysis is not None else ())
                if item.get("candidate_id") == candidate.candidate_id
            ),
            {},
        )
        if candidate.status.value in {"failed", "duplicate"}:
            tier = "quarantine"
        elif promotion is None or evaluation is None:
            tier = "pending"
        elif promotion.decision.value == "approved":
            tier = "iterative_positive"
        else:
            tier = "iterative_negative"

        judge = None
        if evaluation is not None:
            judge = {
                "model_id": _training_safe_value(
                    evaluation.metrics.get("judge_model_id")
                ),
                "accepted": evaluation.metrics.get("judge_accepted"),
            }
        artifact_summary = None
        if artifact is not None:
            artifact_summary = {
                "artifact_id": artifact.artifact_id,
                "artifact_digest": artifact.digest,
                "model_id": artifact.model_id,
                "training_partition": artifact.training_partition,
                "training_rows": artifact.training_rows,
            }
        evaluation_summary = None
        if evaluation is not None:
            evaluation_summary = {
                "score": evaluation.score,
                "passed": evaluation.passed,
                "partition": evaluation.partition,
                "scientific_pass": evaluation.metrics.get("scientific_pass"),
                "judge": judge,
            }
        decision = {
            "status": (
                promotion.decision.value
                if promotion is not None
                else "failed"
                if candidate.status.value == "failed"
                else "duplicate"
                if candidate.status.value == "duplicate"
                else "pending"
            ),
            "reason": (
                _training_safe_value(promotion.reason)
                if promotion is not None
                else None
            ),
            "iterative_only": True,
            "formal_validation": False,
        }
        admission = {
            "tier": tier,
            "formal_training_ready": False,
            "requires_governance_review": True,
            "reason": "仅有迭代阶段证据，尚未经过独立治理审查。",
        }
        dataset_digest = (
            artifact.dataset_digest
            if artifact is not None
            else metadata.get("dataset_digest")
        )
        strategy_id = metadata.get("strategy_id", "parameter_sweep@1")
        strategy_digest = metadata.get("strategy_digest")
        prediction_model_id = metadata.get("prediction_model_id")
        prediction_model_digest = metadata.get("prediction_model_digest")
        policy_model_id = metadata.get(
            "policy_model_id", HOST_PARAMETER_GENERATOR_ID
        )
        judge_model_id = metadata.get("judge_model_id", RULE_JUDGE_ID)
        strategy_model_id = metadata.get("strategy_model_id", policy_model_id)
        review_model_id = metadata.get("review_model_id", judge_model_id)
        model_workflow = metadata.get("model_workflow")
        research_domain = metadata.get("research_domain", task.domain_pack)
        model_plan = getattr(proposal, "metadata", {})
        policy_digest = metadata.get("policy_model_digest")
        if not isinstance(policy_digest, str) or not policy_digest:
            policy_digest = builtin_model_configuration_digest(
                str(policy_model_id)
            )
        judge_digest = metadata.get("judge_model_digest")
        if not isinstance(judge_digest, str) or not judge_digest:
            judge_digest = builtin_model_configuration_digest(str(judge_model_id))
        policy_binding_source = metadata.get("policy_model_binding_source") or (
            "builtin_implementation" if policy_digest is not None else "legacy_missing"
        )
        judge_binding_source = metadata.get("judge_model_binding_source") or (
            "builtin_implementation" if judge_digest is not None else "legacy_missing"
        )
        event_receipts = _event_receipts(
            state, candidate.candidate_id, proposal.proposal_id
        )
        event_chain_digest = digest(event_receipts)
        training_stage = {
            "status": (
                "completed"
                if artifact is not None
                else "skipped"
                if candidate.status.value == "duplicate"
                else "failed"
                if candidate.status.value == "failed"
                else "pending"
            ),
            "artifact_id": artifact.artifact_id if artifact is not None else None,
            "artifact_digest": artifact.digest if artifact is not None else None,
            "model_id": artifact.model_id if artifact is not None else None,
            "training_partition": (
                artifact.training_partition if artifact is not None else None
            ),
            "training_rows": artifact.training_rows if artifact is not None else None,
            "parameters": (
                _training_safe_value(artifact.parameters)
                if artifact is not None
                else None
            ),
            "learned_parameters": (
                _training_safe_value(artifact.learned_parameters)
                if artifact is not None
                else None
            ),
            "metrics": (
                _training_safe_value(artifact.metrics)
                if artifact is not None
                else None
            ),
        }
        evaluation_stage = {
            "status": (
                "completed"
                if evaluation is not None
                else "skipped"
                if candidate.status.value == "duplicate"
                else "failed"
                if candidate.status.value == "failed"
                else "pending"
            ),
            "evaluation_id": (
                evaluation.evaluation_id if evaluation is not None else None
            ),
            "evaluator_id": metadata.get("evaluator_id"),
            "evaluator_digest": (
                evaluation.evaluator_digest if evaluation is not None else None
            ),
            "artifact_digest": (
                evaluation.artifact_digest if evaluation is not None else None
            ),
            "partition": evaluation.partition if evaluation is not None else None,
            "score": evaluation.score if evaluation is not None else None,
            "passed": evaluation.passed if evaluation is not None else None,
            "metrics": (
                _training_safe_value(evaluation.metrics)
                if evaluation is not None
                else None
            ),
        }
        episode = {
            "schema_version": "ecologyrsi-dsh.evolution-episode/1",
            "stages": {
                "strategy_input": {
                    "objective": _training_safe_value(task.objective),
                    "domain_pack": task.domain_pack,
                    "dataset_id": task.dataset,
                    "episode_id": metadata.get("episode_id"),
                    "strategy_id": strategy_id,
                    "strategy_digest": strategy_digest,
                    "prediction_model_id": prediction_model_id,
                    "prediction_model_digest": prediction_model_digest,
                    "evaluator_id": metadata.get("evaluator_id"),
                    "evaluator_digest": metadata.get("evaluator_digest"),
                    "policy_model_id": policy_model_id,
                    "judge_model_id": judge_model_id,
                    "strategy_model_id": strategy_model_id,
                    "review_model_id": review_model_id,
                    "autonomous_mode": metadata.get("autonomous_mode") is True,
                    "model_workflow": model_workflow,
                    "research_domain": research_domain,
                    "model_plan": _training_safe_value(model_plan),
                    "policy_model_digest": policy_digest,
                    "judge_model_digest": judge_digest,
                    "policy_model_binding_source": policy_binding_source,
                    "judge_model_binding_source": judge_binding_source,
                    "seed": task.seed,
                    "slot_index": candidate.slot_index,
                    "batch_size": batch.batch_size if batch is not None else 1,
                    "generation_context_digest": (
                        batch.context_digest if batch is not None else None
                    ),
                    "previous_generation_analysis_digest": (
                        batch.previous_analysis_digest if batch is not None else None
                    ),
                    "knowledge_snapshot": knowledge_summary,
                    "parent_candidate_id": proposal.parent_candidate_id,
                    "model_plan": _training_safe_value(model_plan),
                    "parent_parameters": _parent_parameters(
                        state, proposal.parent_candidate_id
                    ),
                    "human_interventions": interventions,
                },
                "proposal_response": {
                    "proposal_id": proposal.proposal_id,
                    "title": _training_safe_value(proposal.title),
                    "rationale": _training_safe_value(proposal.rationale),
                    "parameters": _training_safe_value(proposal.changes),
                    "parent_candidate_id": proposal.parent_candidate_id,
                },
                "training": training_stage,
                "evaluation": evaluation_stage,
                "decision": {
                    **decision,
                    "candidate_status": candidate.status.value,
                    "generation_rank": ranking.get("rank"),
                    "classification": ranking.get("classification"),
                    "selection_reason": ranking.get("selection_reason"),
                    "promotion_id": (
                        promotion.promotion_id if promotion is not None else None
                    ),
                    "failure_reason": _candidate_failure_reason(
                        state, candidate.candidate_id
                    ),
                    "admission": admission,
                },
            },
            "lineage": {
                "run_id": candidate.run_id,
                "candidate_id": candidate.candidate_id,
                "slot_index": candidate.slot_index,
                "proposal_id": proposal.proposal_id,
                "parent_candidate_id": proposal.parent_candidate_id,
                "artifact_id": artifact.artifact_id if artifact is not None else None,
                "evaluation_id": (
                    evaluation.evaluation_id if evaluation is not None else None
                ),
                "evaluator_id": metadata.get("evaluator_id"),
                "prediction_model_id": prediction_model_id,
            },
            "reproducibility": {
                "seed": task.seed,
                "seed_policy": task.seed_policy,
                "policy_version": task.policy_version,
                "manifest_digest": task.digest,
                "dataset_digest": dataset_digest,
                "split_manifest_digest": metadata.get("split_manifest_digest"),
                "proposal_digest": proposal.digest,
                "artifact_digest": artifact.digest if artifact is not None else None,
                "policy_digest": policy_digest,
                "strategy_digest": strategy_digest,
                "prediction_model_digest": prediction_model_digest,
                "strategy_model_id": strategy_model_id,
                "review_model_id": review_model_id,
                "model_workflow": model_workflow,
                "research_domain": research_domain,
                "model_plan": _training_safe_value(model_plan),
                "policy_model_digest": policy_digest,
                "judge_model_digest": judge_digest,
                "policy_model_binding_source": policy_binding_source,
                "judge_model_binding_source": judge_binding_source,
                "evaluator_digest": metadata.get("evaluator_digest")
                or (evaluation.evaluator_digest if evaluation is not None else None),
                "horizons_hours": (
                    [
                        item.get("horizon_hours")
                        for item in evaluation.metrics.get("horizons", [])
                        if isinstance(item, Mapping)
                    ]
                    if evaluation is not None
                    else []
                ),
            },
            "event_receipts": event_receipts,
            "event_chain_digest": event_chain_digest,
        }
        # Keep the historical five-stage episode intact while adding one
        # ordered trajectory that joins agent interaction, host compilation,
        # sample predictions, feedback, and parent/child optimization links.
        # The trajectory builder is a browser-safe read model and does not
        # expose prompts, raw rows, credentials, or private reasoning.
        trajectory, trajectory_summary = build_training_trajectory(
            state,
            candidate,
            proposal,
            artifact,
            evaluation,
            promotion,
            batch=batch,
            knowledge=knowledge,
            interventions=interventions,
            ranking=ranking,
        )
        episode["trajectory"] = trajectory
        episode["trajectory_summary"] = trajectory_summary
        episode["episode_digest_sha256"] = digest(episode)
        sample_identity = {
            "run_id": candidate.run_id,
            "candidate_id": candidate.candidate_id,
            "proposal_digest": proposal.digest,
        }
        result.append(
            {
                "schema_version": "ecologyrsi-dsh.evolution-training-sample/1",
                "sample_id": "training-sample:" + digest(sample_identity)[:32],
                "run_id": candidate.run_id,
                "generation": candidate.generation + 1,
                "slot_index": candidate.slot_index,
                "candidate_id": candidate.candidate_id,
                "parent_candidate_id": proposal.parent_candidate_id,
                "proposal_id": proposal.proposal_id,
                "input": {
                    "dataset_id": task.dataset,
                    "episode_id": metadata.get("episode_id"),
                    "strategy_id": strategy_id,
                    "strategy_digest": strategy_digest,
                    "prediction_model_id": prediction_model_id,
                    "prediction_model_digest": prediction_model_digest,
                    "policy_model_id": policy_model_id,
                    "judge_model_id": judge_model_id,
                    "strategy_model_id": strategy_model_id,
                    "review_model_id": review_model_id,
                    "model_workflow": model_workflow,
                    "research_domain": research_domain,
                    "model_plan": _training_safe_value(model_plan),
                    "batch_size": batch.batch_size if batch is not None else 1,
                    "generation_context_digest": (
                        batch.context_digest if batch is not None else None
                    ),
                    "previous_generation_analysis": (
                        _training_safe_value(
                            state.analysis_for(candidate.generation - 1).identity_dict()
                        )
                        if candidate.generation > 0
                        and state.analysis_for(candidate.generation - 1) is not None
                        else None
                    ),
                    "knowledge_snapshot": knowledge_summary,
                    "parent_parameters": _parent_parameters(
                        state, proposal.parent_candidate_id
                    ),
                    "applied_interventions": interventions,
                },
                "output": {
                    "title": _training_safe_value(proposal.title),
                    "rationale": _training_safe_value(proposal.rationale),
                    "changes": _training_safe_value(proposal.changes),
                    "artifact": artifact_summary,
                    "model_id": artifact.model_id if artifact is not None else None,
                },
                "evaluation": evaluation_summary,
                "decision": decision,
                "admission": admission,
                "trajectory": trajectory,
                "trajectory_summary": trajectory_summary,
                "episode": episode,
                "provenance": {
                    "manifest_digest": task.digest,
                    "dataset_digest": dataset_digest,
                    "split_manifest_digest": metadata.get("split_manifest_digest"),
                    "proposal_digest": proposal.digest,
                    "knowledge_snapshot_digest": (
                        knowledge.snapshot_digest if knowledge is not None else None
                    ),
                    "artifact_digest": artifact.digest if artifact is not None else None,
                    "evaluator_digest": (
                        metadata.get("evaluator_digest")
                        or (evaluation.evaluator_digest if evaluation is not None else None)
                    ),
                    "strategy_digest": strategy_digest,
                    "prediction_model_digest": prediction_model_digest,
                    "policy_model_digest": policy_digest,
                    "judge_model_digest": judge_digest,
                    "policy_model_binding_source": policy_binding_source,
                    "judge_model_binding_source": judge_binding_source,
                    "source_event_seq": _candidate_source_event_seq(
                        state, candidate.candidate_id, proposal.proposal_id
                    ),
                    "derived_from": "append_only_event_projection",
                    "formal_validation_status": "not_run",
                    "event_chain_digest": event_chain_digest,
                    "episode_digest_sha256": episode["episode_digest_sha256"],
                },
            }
        )
    return result
