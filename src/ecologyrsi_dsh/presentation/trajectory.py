"""Browser-safe complete evolution trajectory assets.

The durable run state intentionally stores facts in separate records.  This
module joins those records into one bounded, ordered episode for training-data
inspection: task input, agent research/proposal, host compilation, prediction,
feedback, optimization, and the resulting candidate decision.  It is a read
model only; it never changes the event ledger and never exports prompts,
credentials, raw rows, or private reasoning.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..core.models import digest
from ..core.redaction import (
    is_sensitive_key,
    public_error_summary,
    redact_sensitive_text,
    sanitize_public_value,
)

TRAJECTORY_SCHEMA_VERSION = "ecologyrsi-dsh.evolution-trajectory/1"
TRAJECTORY_LIMIT = 48
TRAJECTORY_STAGE_ORDER = (
    "input_context",
    "agent_research",
    "agent_proposal",
    "host_compile",
    "training_prediction",
    "agent_feedback",
    "agent_optimization",
    "final_result",
)

_SENSITIVE_KEYS = frozenset(
    {
        "chain_of_thought",
        "messages",
        "private_reasoning",
        "prompt",
        "raw",
        "raw_rows",
        "reasoning",
        "rows",
        "samples",
        "prediction_preview",
        "prediction_records",
        "sample_predictions",
        "inference_trace",
        "observations",
        "observed",
        "predicted",
        "baseline",
        "思维链",
        "私密推理",
        "原始数据",
        "原始行",
    }
)


def _safe_key(key: Any) -> bool:
    return not is_sensitive_key(key, extra_keys=_SENSITIVE_KEYS)


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    """Bound JSON values while filtering fields unsafe for a training asset."""

    return sanitize_public_value(
        value,
        extra_sensitive_keys=_SENSITIVE_KEYS,
        depth=depth,
        max_depth=7,
        text_limit=1000,
        mapping_limit=64,
        sequence_limit=32,
    )


def _finite(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _scalar(value: Any, *, limit: int = 300) -> str | int | float | None:
    number = _finite(value)
    if number is not None:
        return number
    if isinstance(value, str):
        return value[:limit]
    return None


def _count(value: Any, fallback: int = 0) -> int:
    number = _finite(value)
    if number is None:
        return fallback
    return max(0, int(number))


def _status_from_candidate(candidate: Any) -> str:
    return str(getattr(getattr(candidate, "status", None), "value", "pending"))


def _candidate_events(
    state: Any,
    candidate_id: str,
    proposal_id: str,
    generation: int,
) -> list[Any]:
    """Return only events that can explain this candidate's trajectory."""

    relevant: list[Any] = []
    for event in state.events:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        nested = None
        if event.kind in {
            "ProposalSubmitted",
            "CandidateSpawned",
            "ArtifactRecorded",
            "EvaluationRecorded",
            "PromotionDecided",
        }:
            nested_key = {
                "ProposalSubmitted": "proposal",
                "CandidateSpawned": "candidate",
                "ArtifactRecorded": "artifact",
                "EvaluationRecorded": "evaluation",
                "PromotionDecided": "promotion",
            }[event.kind]
            nested = payload.get(nested_key)
        matches = isinstance(nested, Mapping) and (
            nested.get("candidate_id") == candidate_id
            or nested.get("proposal_id") == proposal_id
        )
        matches = matches or (
            event.kind in {"CandidateFailed", "CandidateMarkedDuplicate"}
            and payload.get("candidate_id") == candidate_id
        )
        matches = matches or (
            event.kind == "HumanInterventionApplied"
            and payload.get("proposal_id") == proposal_id
        )
        if event.kind == "EvolutionStageRecorded":
            if payload.get("generation") != generation:
                continue
            event_candidate = payload.get("candidate_id")
            event_proposal = payload.get("proposal_id")
            # A candidate must never inherit a sibling's stage event.  Legacy
            # records without candidate_id are accepted only when proposal_id
            # identifies this candidate.
            matches = (
                event_candidate == candidate_id
                if event_candidate is not None
                else event_proposal == proposal_id
            )
        if matches:
            relevant.append(event)
    return sorted(relevant, key=lambda item: item.seq)


def _stage_event(
    events: list[Any],
    stage: str,
) -> dict[str, Any] | None:
    stage_events = [
        event
        for event in events
        if event.kind == "EvolutionStageRecorded"
        and isinstance(event.payload, Mapping)
        and event.payload.get("stage") == stage
    ]
    if not stage_events:
        return None
    started = next(
        (event for event in stage_events if event.payload.get("status") == "started"),
        stage_events[0],
    )
    ended = next(
        (
            event
            for event in reversed(stage_events)
            if event.payload.get("status") in {"completed", "failed"}
        ),
        None,
    )
    status = str((ended or stage_events[-1]).payload.get("status") or "pending")
    if status == "started":
        status = "running"
    result: dict[str, Any] = {
        "status": status,
        "start_seq": started.seq,
        "started_at": started.created_at,
        "end_seq": ended.seq if ended is not None else None,
        "completed_at": ended.created_at if ended is not None else None,
        "evidence": "stage_event",
    }
    error = (ended or stage_events[-1]).payload.get("public_error")
    if error:
        result["public_error"] = public_error_summary(error)
    return result


def _stage(
    stage_name: str,
    fallback_status: str,
    events: list[Any],
    **fields: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": fallback_status}
    event_info = _stage_event(events, stage_name)
    if event_info is not None:
        result.update(event_info)
    result.update(fields)
    return result


def _prediction_source(evaluation: Any | None) -> tuple[list[Any], int]:
    metrics = dict(getattr(evaluation, "metrics", {}) or {}) if evaluation else {}
    raw: Any = None
    # These are legacy evaluator fields.  They are read here and renamed to a
    # public training-asset vocabulary; the unsafe source key is never emitted.
    for key in (
        "prediction_preview",
        "prediction_records",
        "sample_predictions",
        "inference_trace",
        "predictions",
    ):
        candidate = metrics.get(key)
        if isinstance(candidate, list):
            raw = candidate
            break
    source = raw if isinstance(raw, list) else []
    total = max(
        _count(metrics.get("sample_count", metrics.get("n")), len(source)),
        len(source),
    )
    return source, total


def _prediction_records(evaluation: Any | None) -> dict[str, Any]:
    source, total = _prediction_source(evaluation)
    records: list[dict[str, Any]] = []
    for index, item in enumerate(source[:TRAJECTORY_LIMIT]):
        if not isinstance(item, Mapping):
            continue
        observed = _finite(
            item.get(
                "observed",
                item.get("observed_value", item.get("actual", item.get("actual_value"))),
            )
        )
        predicted = _finite(
            item.get(
                "predicted",
                item.get(
                    "predicted_value",
                    item.get("prediction", item.get("forecast", item.get("value"))),
                ),
            )
        )
        baseline = _finite(
            item.get("baseline", item.get("baseline_value", item.get("persistence")))
        )
        record: dict[str, Any] = {
            "sample_index": index + 1,
            "origin_timestamp": _scalar(item.get("origin_timestamp")),
            "target_timestamp": _scalar(
                item.get("target_timestamp", item.get("timestamp"))
            ),
            "target": _scalar(item.get("target")),
            "unit": _scalar(item.get("unit")),
            "horizon_hours": _finite(item.get("horizon_hours")),
            "observed_value": observed,
            "predicted_value": predicted,
            "baseline_value": baseline,
            "value": predicted,
            # Compact aliases make the record convenient for generic episode
            # consumers while the explicit *_value fields stay self-describing.
            "reference": observed,
        }
        if observed is not None and predicted is not None:
            record["error"] = predicted - observed
        if observed is not None and baseline is not None:
            record["baseline_error"] = baseline - observed
        input_summary = item.get("input_summary", item.get("features"))
        if isinstance(input_summary, Mapping):
            record["input_summary"] = _safe_value(input_summary)
        step = item.get("step_summary", item.get("method_step", item.get("method")))
        if isinstance(step, str) and step.strip():
            record["step_summary"] = redact_sensitive_text(step, limit=500)
        records.append(record)
    return {
        "prediction_records": records,
        "sample_count": total,
        "shown_count": len(records),
        "truncated": total > len(records),
    }


def _aggregate_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        return {}
    blocked = {
        "prediction_preview",
        "inference_trace",
        "predictions",
        "prediction_records",
        "sample_predictions",
        "rows",
        "raw_rows",
        "raw",
        "private_reasoning",
        "prompt",
        "messages",
    }
    return {
        str(key): _safe_value(value)
        for key, value in list(metrics.items())[:80]
        if isinstance(key, str) and key not in blocked and _safe_key(key)
    }


def _analysis_summary(analysis: Any | None) -> dict[str, Any] | None:
    if analysis is None:
        return None
    return {
        "generation": getattr(analysis, "generation", None),
        "outcome": _safe_value(getattr(analysis, "outcome", None)),
        "champion_candidate_id": _safe_value(
            getattr(analysis, "champion_candidate_id", None)
        ),
        "incumbent_before_candidate_id": _safe_value(
            getattr(analysis, "incumbent_before_candidate_id", None)
        ),
        "incumbent_after_candidate_id": _safe_value(
            getattr(analysis, "incumbent_after_candidate_id", None)
        ),
        "selection_reason": _safe_value(getattr(analysis, "selection_reason", "")),
        "next_generation_focus": _safe_value(
            getattr(analysis, "next_generation_focus", "")
        ),
        "common_failures": _safe_value(list(getattr(analysis, "common_failures", ()))),
        "target_weaknesses": _safe_value(
            list(getattr(analysis, "target_weaknesses", ()))
        ),
        "horizon_weaknesses": _safe_value(
            list(getattr(analysis, "horizon_weaknesses", ()))
        ),
        "parameter_effects": _safe_value(
            list(getattr(analysis, "parameter_effects", ()))
        ),
    }


def _knowledge_summary(knowledge: Any | None) -> dict[str, Any] | None:
    if knowledge is None:
        return None
    cards = []
    for card in tuple(getattr(knowledge, "cards", ()))[:12]:
        cards.append(
            {
                "knowledge_id": _safe_value(getattr(card, "knowledge_id", None)),
                "title": _safe_value(getattr(card, "title", None)),
                "summary": _safe_value(getattr(card, "summary", None)),
                "source_url": _safe_value(getattr(card, "source_url", None)),
                "execution_status": _safe_value(
                    getattr(card, "execution_status", None)
                ),
                "capability_kind": _safe_value(getattr(card, "capability_kind", None)),
                "capability_id": _safe_value(getattr(card, "capability_id", None)),
                "selection_reason": _safe_value(
                    getattr(card, "selection_reason", None)
                ),
            }
        )
    return {
        "snapshot_digest": _safe_value(getattr(knowledge, "snapshot_digest", None)),
        "query_terms": _safe_value(list(getattr(knowledge, "query_terms", ()))),
        "provider": _safe_value(getattr(knowledge, "provider", None)),
        "retrieval_status": _safe_value(getattr(knowledge, "retrieval_status", None)),
        "online_enabled": bool(getattr(knowledge, "online_enabled", False)),
        "cards": cards,
        "adopted_knowledge_ids": [
            _safe_value(getattr(card, "knowledge_id", None))
            for card in tuple(getattr(knowledge, "executable_cards", ()))[:12]
        ],
        "warnings": _safe_value(list(getattr(knowledge, "warnings", ()))),
    }


def _parent_context(state: Any, parent_id: str | None) -> dict[str, Any] | None:
    if not parent_id:
        return None
    try:
        parent = state.candidate(parent_id)
        parent_proposal = state.proposal(parent.proposal_id)
    except (KeyError, AttributeError):
        return {"candidate_id": parent_id, "status": "unavailable"}
    parent_artifact = state.artifact_for(parent_id)
    parent_evaluation = state.evaluation_for(parent_id)
    parent_promotion = state.promotion_for(parent_id)
    result: dict[str, Any] = {
        "candidate_id": parent.candidate_id,
        "proposal_id": parent.proposal_id,
        "generation": parent.generation + 1,
        "status": _status_from_candidate(parent),
        "parameters": _safe_value(dict(parent_proposal.changes)),
        "title": _safe_value(parent_proposal.title),
    }
    if parent_artifact is not None:
        result["artifact"] = {
            "artifact_id": parent_artifact.artifact_id,
            "model_id": parent_artifact.model_id,
            "training_partition": parent_artifact.training_partition,
            "training_rows": parent_artifact.training_rows,
            "artifact_digest": parent_artifact.digest,
        }
    if parent_evaluation is not None:
        result["evaluation"] = {
            "evaluation_id": parent_evaluation.evaluation_id,
            "score": parent_evaluation.score,
            "passed": parent_evaluation.passed,
            "partition": parent_evaluation.partition,
            "metrics": _aggregate_metrics(parent_evaluation.metrics),
        }
        judge = {
            key: _safe_value(parent_evaluation.metrics.get(key))
            for key in (
                "judge_model_id",
                "judge_accepted",
                "judge_guidance",
                "judge_parameter_override",
            )
            if key in parent_evaluation.metrics
        }
        if judge:
            result["judge"] = judge
    if parent_promotion is not None:
        result["promotion"] = {
            "decision": parent_promotion.decision.value,
            "reason": _safe_value(parent_promotion.reason),
            "promotion_id": parent_promotion.promotion_id,
        }
    return result


def _parameter_changes(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any],
) -> list[dict[str, Any]]:
    before_map = dict(before or {})
    names = list(dict.fromkeys([*before_map.keys(), *after.keys()]))[:48]
    result = []
    for name in names:
        old = before_map.get(name)
        new = after.get(name)
        changed = old != new
        result.append(
            {
                "parameter": str(name)[:120],
                "before": _safe_value(old),
                "after": _safe_value(new),
                "changed": changed,
                "direction": (
                    "increased"
                    if isinstance(old, (int, float))
                    and isinstance(new, (int, float))
                    and new > old
                    else "decreased"
                    if isinstance(old, (int, float))
                    and isinstance(new, (int, float))
                    and new < old
                    else "changed"
                    if changed
                    else "unchanged"
                ),
            }
        )
    return result


def _failure_reason(state: Any, candidate_id: str) -> str | None:
    for event in reversed(state.events):
        if (
            event.kind == "CandidateFailed"
            and isinstance(event.payload, Mapping)
            and event.payload.get("candidate_id") == candidate_id
        ):
            return _safe_value(event.payload.get("reason"))
    return None


def build_training_trajectory(
    state: Any,
    candidate: Any,
    proposal: Any,
    artifact: Any | None,
    evaluation: Any | None,
    promotion: Any | None,
    *,
    batch: Any | None = None,
    knowledge: Any | None = None,
    interventions: list[Mapping[str, Any]] | None = None,
    ranking: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one complete, bounded evolution trajectory and its summary."""

    task = state.task_manifest
    metadata = dict(task.metadata)
    generation = int(candidate.generation)
    parent_id = getattr(proposal, "parent_candidate_id", None)
    parent = _parent_context(state, parent_id)
    analysis = state.analysis_for(generation)
    event_list = _candidate_events(
        state, candidate.candidate_id, proposal.proposal_id, generation
    )
    prediction = _prediction_records(evaluation)
    # Link every bounded prediction record back to the frozen input episode
    # without copying source rows or hidden evaluation data.
    for record in prediction["prediction_records"]:
        record["input_reference"] = {
            "dataset_id": _safe_value(task.dataset),
            "episode_id": _safe_value(metadata.get("episode_id")),
            "partition": _safe_value(
                evaluation.partition if evaluation is not None else None
            ),
            "origin_timestamp": record.get("origin_timestamp"),
            "target_timestamp": record.get("target_timestamp"),
            "target": record.get("target"),
            "horizon_hours": record.get("horizon_hours"),
        }
    plan = getattr(proposal, "metadata", {}).get("plan") if getattr(proposal, "metadata", None) else None
    if not isinstance(plan, Mapping):
        plan = metadata.get("autonomous_plan")
    if not isinstance(plan, Mapping):
        plan = {}
    plan = _safe_value(plan)
    strategy_id = metadata.get("strategy_id", "parameter_sweep@1")
    strategy_model_id = metadata.get(
        "strategy_model_id", metadata.get("policy_model_id", "host_parameter_generator@1")
    )
    proposal_metadata = (
        proposal.metadata if isinstance(proposal.metadata, Mapping) else {}
    )
    proposal_source = proposal_metadata.get("proposal_source", "legacy_unknown")
    knowledge_data = _knowledge_summary(knowledge)
    intervention_data = _safe_value(interventions or [])
    parent_parameters = (
        parent.get("parameters") if isinstance(parent, Mapping) else None
    )
    parameter_changes = _parameter_changes(
        parent_parameters if isinstance(parent_parameters, Mapping) else None,
        dict(proposal.changes),
    )
    # A training asset is an immutable snapshot of the candidate's own
    # episode.  Looking up descendants on every later projection would mutate
    # an already-exported parent asset and break replay/digest stability.  The
    # child carries the durable reverse link (parent_candidate_id) and its
    # parent feedback in the stages below; run-level candidate/round views can
    # still show the forward graph.
    children: list[dict[str, Any]] = []
    current_status = _status_from_candidate(candidate)
    terminal = current_status in {"evaluated", "promoted", "rejected", "failed", "duplicate"}
    score = evaluation.score if evaluation is not None else None
    parent_score = (
        parent.get("evaluation", {}).get("score")
        if isinstance(parent, Mapping) and isinstance(parent.get("evaluation"), Mapping)
        else None
    )
    score_delta = (
        score - parent_score
        if isinstance(score, (int, float)) and isinstance(parent_score, (int, float))
        else None
    )
    judge = {}
    if evaluation is not None:
        judge = {
            key: _safe_value(evaluation.metrics.get(key))
            for key in (
                "judge_model_id",
                "judge_accepted",
                "judge_guidance",
                "judge_parameter_override",
            )
            if key in evaluation.metrics
        }
    model_plan_status = plan.get("status") if isinstance(plan, Mapping) else None
    if not model_plan_status:
        model_plan_status = "host_strategy" if strategy_id not in {
            "autonomous_model@1",
            "dsh_authenticated@1",
        } else "not_recorded"
    compile_status = (
        "skipped"
        if current_status == "duplicate"
        else "completed"
        if artifact is not None
        else "failed"
        if current_status == "failed"
        else "pending"
    )
    prediction_status = (
        "skipped"
        if current_status == "duplicate"
        else "completed"
        if evaluation is not None
        else "failed"
        if current_status == "failed"
        else "pending"
    )
    feedback_status = prediction_status
    final_status = (
        "failed"
        if current_status == "failed"
        else "completed"
        if terminal
        else "pending"
    )

    # Stage-specific payloads intentionally use direct fields so a UI can
    # render the chain without understanding a generic event envelope.
    trajectory: dict[str, Any] = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "stage_order": list(TRAJECTORY_STAGE_ORDER),
        "input_context": _stage(
            "input_context",
            "completed",
            event_list,
            run_id=candidate.run_id,
            generation=generation + 1,
            objective=_safe_value(task.objective),
            domain_pack=_safe_value(task.domain_pack),
            dataset_id=_safe_value(task.dataset),
            episode_id=_safe_value(metadata.get("episode_id")),
            dataset_digest=_safe_value(
                artifact.dataset_digest
                if artifact is not None
                else metadata.get("dataset_digest")
            ),
            split_manifest_digest=_safe_value(metadata.get("split_manifest_digest")),
            batch_context_digest=_safe_value(
                batch.context_digest if batch is not None else None
            ),
            batch_size=batch.batch_size if batch is not None else 1,
            slot_index=candidate.slot_index,
            parent_candidate_id=parent_id,
            parent_parameters=_safe_value(parent_parameters),
            previous_generation_analysis=_analysis_summary(
                state.analysis_for(generation - 1) if generation > 0 else None
            ),
            knowledge=knowledge_data,
            human_interventions=intervention_data,
            strategy_id=_safe_value(strategy_id),
            proposal_source=_safe_value(proposal_source),
            strategy_digest=_safe_value(metadata.get("strategy_digest")),
            prediction_model_id=_safe_value(metadata.get("prediction_model_id")),
            evaluator_id=_safe_value(metadata.get("evaluator_id")),
            model_workflow=_safe_value(metadata.get("model_workflow")),
        ),
        "agent_research": _stage(
            "proposal",
            "completed",
            event_list,
            status_detail=_safe_value(model_plan_status),
            model_id=_safe_value(strategy_model_id),
            interaction={
                "mode": "structured_json_boundary",
                "request_summary": {
                    "objective": _safe_value(task.objective),
                    "generation": generation + 1,
                    "parent_candidate_id": parent_id,
                    "knowledge_snapshot_digest": (
                        knowledge_data.get("snapshot_digest")
                        if isinstance(knowledge_data, Mapping)
                        else None
                    ),
                    "feedback_available": bool(parent and parent.get("evaluation")),
                    "human_input_available": bool(interventions),
                },
                "response_recorded": True,
                "private_content_status": "excluded",
            },
            team=_safe_value(plan.get("team")) if isinstance(plan, Mapping) else None,
            prediction_model=_safe_value(plan.get("prediction_model"))
            if isinstance(plan, Mapping)
            else None,
            strategy=_safe_value(plan.get("strategy")) if isinstance(plan, Mapping) else None,
            research=_safe_value(plan.get("research", [])) if isinstance(plan, Mapping) else [],
            implementation_notes=_safe_value(plan.get("implementation_notes"))
            if isinstance(plan, Mapping)
            else None,
            confidence=_finite(plan.get("confidence")) if isinstance(plan, Mapping) else None,
            proposal_id=_safe_value(proposal.proposal_id),
            proposal_title=_safe_value(proposal.title),
            proposal_parameters=_safe_value(dict(proposal.changes)),
            proposal_source=_safe_value(proposal_source),
            remote_strategy_called=(
                proposal_metadata.get("remote_strategy_called") is True
            ),
            remote_strategy_succeeded=(
                proposal_metadata.get("remote_strategy_succeeded") is True
            ),
            knowledge_used=(
                knowledge_data.get("adopted_knowledge_ids", [])
                if isinstance(knowledge_data, Mapping)
                else []
            ),
        ),
        "agent_proposal": _stage(
            "proposal",
            "completed",
            event_list,
            proposal_id=proposal.proposal_id,
            title=_safe_value(proposal.title),
            rationale=_safe_value(proposal.rationale),
            parent_candidate_id=parent_id,
            requested_parameters=_safe_value(dict(proposal.changes)),
            strategy_id=_safe_value(strategy_id),
            strategy_model_id=_safe_value(strategy_model_id),
            proposal_source=_safe_value(proposal_source),
            remote_strategy_called=(
                proposal_metadata.get("remote_strategy_called") is True
            ),
            model_plan=_safe_value(plan),
        ),
        "host_compile": _stage(
            "candidate",
            compile_status,
            event_list,
            candidate_id=candidate.candidate_id,
            proposal_id=proposal.proposal_id,
            accepted_parameters=_safe_value(dict(proposal.changes)),
            parameter_changes=parameter_changes,
            predictor_id=_safe_value(artifact.model_id if artifact is not None else metadata.get("prediction_model_id")),
            implementation_plan=_safe_value(plan.get("prediction_model"))
            if isinstance(plan, Mapping)
            else None,
            artifact_id=_safe_value(artifact.artifact_id if artifact is not None else None),
            artifact_digest=_safe_value(artifact.digest if artifact is not None else None),
            training_partition=_safe_value(artifact.training_partition if artifact is not None else None),
            training_rows=artifact.training_rows if artifact is not None else None,
            boundary={
                "parameters_range_checked": True,
                "registered_predictor_only": True,
                "scientific_gate_host_controlled": True,
                "arbitrary_code_executed": False,
            },
        ),
        "training_prediction": _stage(
            "evaluation",
            prediction_status,
            event_list,
            model_id=_safe_value(artifact.model_id if artifact is not None else metadata.get("prediction_model_id")),
            partition=_safe_value(evaluation.partition if evaluation is not None else None),
            evaluator_digest=_safe_value(
                evaluation.evaluator_digest
                if evaluation is not None
                else metadata.get("evaluator_digest")
            ),
            training_fit={
                "artifact_id": _safe_value(artifact.artifact_id if artifact is not None else None),
                "training_rows": artifact.training_rows if artifact is not None else None,
                "training_partition": _safe_value(artifact.training_partition if artifact is not None else None),
                "metrics": _aggregate_metrics(artifact.metrics if artifact is not None else None),
            },
            prediction_records=prediction["prediction_records"],
            sample_count=prediction["sample_count"],
            shown_count=prediction["shown_count"],
            truncated=prediction["truncated"],
            result_metrics=_aggregate_metrics(evaluation.metrics if evaluation is not None else None),
        ),
        "agent_feedback": _stage(
            "judge",
            feedback_status,
            event_list,
            current_candidate={
                "candidate_id": candidate.candidate_id,
                "score": score,
                "passed": evaluation.passed if evaluation is not None else None,
                "partition": evaluation.partition if evaluation is not None else None,
                "metrics": _aggregate_metrics(evaluation.metrics if evaluation is not None else None),
            },
            parent_candidate=parent,
            score_delta_from_parent=score_delta,
            improvement=(
                _finite(evaluation.metrics.get("improvement"))
                if evaluation is not None
                else None
            ),
            reviewer_model_id=_safe_value(
                evaluation.metrics.get("judge_model_id")
                if evaluation is not None
                else metadata.get("judge_model_id")
            ),
            judge=judge,
            interaction={
                "mode": "aggregate_feedback_boundary",
                "request_summary": {
                    "proposal_id": proposal.proposal_id,
                    "metric_keys": sorted(
                        str(key)
                        for key in (
                            evaluation.metrics.keys() if evaluation is not None else ()
                        )
                        if str(key)
                        not in {"prediction_preview", "inference_trace", "predictions"}
                        and _safe_key(str(key))
                    )[:48],
                },
                "response_recorded": evaluation is not None,
            },
            generation_analysis=_analysis_summary(analysis),
            ranking=_safe_value(dict(ranking or {})),
            feedback_available=evaluation is not None or bool(parent and parent.get("evaluation")),
        ),
        "agent_optimization": _stage(
            "proposal",
            "completed",
            event_list,
            strategy_id=_safe_value(strategy_id),
            strategy_model_id=_safe_value(strategy_model_id),
            model_workflow=_safe_value(metadata.get("model_workflow")),
            proposal_digest=_safe_value(proposal.digest),
            parent_candidate_id=parent_id,
            feedback_source=(
                "parent_evaluation" if parent and parent.get("evaluation") else "initial_context"
            ),
            parameter_changes=parameter_changes,
            applied_interventions=intervention_data,
            next_candidates=children,
            next_candidate_ids=[item["candidate_id"] for item in children],
            feedback={
                "parent_score": parent_score,
                "current_score": score,
                "score_delta": score_delta,
                "improvement": (
                    _finite(evaluation.metrics.get("improvement"))
                    if evaluation is not None
                    else None
                ),
                "judge_guidance": judge.get("judge_guidance"),
            },
            optimization_signal={
                "current_score": score,
                "parent_score": parent_score,
                "score_delta": score_delta,
                "next_generation_focus": (
                    getattr(analysis, "next_generation_focus", "") if analysis is not None else ""
                ),
            },
        ),
        "final_result": _stage(
            "decision",
            final_status,
            event_list,
            candidate_id=candidate.candidate_id,
            candidate_status=current_status,
            decision=(
                promotion.decision.value
                if promotion is not None
                else "failed"
                if current_status == "failed"
                else "duplicate"
                if current_status == "duplicate"
                else "pending"
            ),
            score=score,
            passed=evaluation.passed if evaluation is not None else None,
            metrics=_aggregate_metrics(evaluation.metrics if evaluation is not None else None),
            promotion_id=_safe_value(promotion.promotion_id if promotion is not None else None),
            decision_reason=_safe_value(promotion.reason if promotion is not None else None),
            generation_rank=_safe_value((ranking or {}).get("rank")),
            eligible=_safe_value((ranking or {}).get("eligible")),
            classification=_safe_value((ranking or {}).get("classification")),
            selection_reason=_safe_value((ranking or {}).get("selection_reason")),
            artifact_digest=_safe_value(artifact.digest if artifact is not None else None),
            evaluator_digest=_safe_value(
                evaluation.evaluator_digest
                if evaluation is not None
                else metadata.get("evaluator_digest")
            ),
            prediction_ref="training_prediction.prediction_records",
            prediction_summary={
                "sample_count": prediction["sample_count"],
                "shown_count": prediction["shown_count"],
                "truncated": prediction["truncated"],
            },
            is_champion=bool(
                promotion is not None and promotion.decision.value == "approved"
            ),
            failure_reason=_failure_reason(state, candidate.candidate_id),
            next_candidate_ids=[item["candidate_id"] for item in children],
        ),
    }
    stage_statuses = {
        name: str(trajectory[name].get("status", "pending"))
        for name in TRAJECTORY_STAGE_ORDER
    }
    completed_count = sum(status == "completed" for status in stage_statuses.values())
    summary: dict[str, Any] = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "stage_order": list(TRAJECTORY_STAGE_ORDER),
        "stage_statuses": stage_statuses,
        "completed_stage_count": completed_count,
        "total_stage_count": len(TRAJECTORY_STAGE_ORDER),
        "stage_count": len(TRAJECTORY_STAGE_ORDER),
        "overall_status": (
            "failed"
            if current_status == "failed"
            else "completed"
            if terminal
            else "running"
            if event_list
            else "pending"
        ),
        "sample_count": prediction["sample_count"],
        "shown_count": prediction["shown_count"],
        "prediction_count": prediction["sample_count"],
        "truncated": prediction["truncated"],
        "feedback_available": bool(
            evaluation is not None or (parent and parent.get("evaluation"))
        ),
        "parent_candidate_id": parent_id,
        "child_count": len(children),
        "child_link_mode": "reverse_parent_reference",
        "event_count": len(event_list),
        "source_event_seq_start": event_list[0].seq if event_list else None,
        "source_event_seq_end": event_list[-1].seq if event_list else None,
        "source_event_seq": event_list[-1].seq if event_list else None,
        "final_score": score,
        "final_decision": trajectory["final_result"]["decision"],
        "trajectory_digest": digest(trajectory),
    }
    return trajectory, summary


__all__ = [
    "TRAJECTORY_LIMIT",
    "TRAJECTORY_SCHEMA_VERSION",
    "TRAJECTORY_STAGE_ORDER",
    "build_training_trajectory",
]
