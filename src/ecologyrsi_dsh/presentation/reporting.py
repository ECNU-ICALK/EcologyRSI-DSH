"""Run summaries, stage projections, and export helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.models import digest
from ..integrations.model_bindings import (
    HOST_PARAMETER_GENERATOR_ID,
    RULE_JUDGE_ID,
    builtin_model_configuration_digest,
)
from ..version import __version__
from .training_assets import _applied_interventions, training_assets


_EVOLUTION_STAGE_ORDER = (
    "proposal",
    "candidate",
    "training",
    "evaluation",
    "judge",
    "decision",
)


def _round_timing(state: Any, generation: int) -> dict[str, Any]:
    """Derive wall-clock duration from immutable generation boundary events."""

    started = next(
        (
            event
            for event in state.events
            if (
                event.kind == "EvolutionStageRecorded"
                and event.payload.get("stage") == "research"
                and event.payload.get("status") == "started"
                and int(event.payload.get("generation", -1)) == generation
            )
            or (
                event.kind == "GenerationBatchStarted"
                and isinstance(event.payload.get("batch"), Mapping)
                and int(event.payload["batch"].get("generation", -1))
                == generation
            )
        ),
        None,
    )
    if started is None:
        return {"status": "not_started", "duration_ms": None}
    finished = next(
        (
            event
            for event in state.events
            if event.kind == "GenerationAdvanced"
            and int(event.payload.get("generation", -1)) == generation + 1
            and event.seq >= started.seq
        ),
        None,
    )
    duration_ms = None
    try:
        start_at = datetime.fromisoformat(
            str(started.created_at).replace("Z", "+00:00")
        )
        end_at = (
            datetime.fromisoformat(str(finished.created_at).replace("Z", "+00:00"))
            if finished is not None
            else None
        )
        if end_at is not None:
            if start_at.tzinfo is None:
                start_at = start_at.replace(tzinfo=timezone.utc)
            if end_at.tzinfo is None:
                end_at = end_at.replace(tzinfo=timezone.utc)
            duration_ms = max(0, round((end_at - start_at).total_seconds() * 1000, 3))
    except (TypeError, ValueError):
        pass
    return {
        "status": "completed" if finished is not None else "running",
        "started_at": started.created_at,
        "completed_at": finished.created_at if finished is not None else None,
        "duration_ms": duration_ms,
    }


def judge_stage_status(evaluation: Any | None) -> str:
    """Map persisted judge evidence to the shared execution-stage vocabulary."""

    if evaluation is None:
        return "pending"
    metrics = evaluation.metrics
    status = metrics.get("judge_status")
    if status == "completed":
        return "completed"
    if status == "unavailable":
        return "failed"
    if "judge_model_id" in metrics or isinstance(metrics.get("judge_accepted"), bool):
        return "completed"
    return "not_recorded"


def _latest_stage_statuses(
    state: Any,
    generation: int,
    *,
    proposal_id: str | None,
    candidate_id: str | None,
) -> dict[str, str]:
    latest: dict[str, str] = {}
    for event in state.events:
        if event.kind != "EvolutionStageRecorded":
            continue
        payload = event.payload
        if payload.get("generation") != generation:
            continue
        recorded_proposal_id = payload.get("proposal_id")
        recorded_candidate_id = payload.get("candidate_id")
        # Candidate projections must never inherit a batch-level event.  A
        # few early ledger records omitted one of the identifiers, so accept
        # a missing candidate id only when the proposal id still identifies
        # the requested candidate.  For a proposal-only projection the same
        # rule applies in the other direction.
        if candidate_id is not None:
            if recorded_candidate_id is not None:
                if recorded_candidate_id != candidate_id:
                    continue
            elif proposal_id is None or recorded_proposal_id != proposal_id:
                continue
        elif proposal_id is not None:
            if recorded_proposal_id != proposal_id:
                continue
        elif recorded_proposal_id is None and recorded_candidate_id is None:
            # Keep batch-level events only for the unscoped aggregate view.
            pass
        status = str(payload["status"])
        latest[str(payload["stage"])] = "running" if status == "started" else status
    return latest


def _candidate_stage_statuses(
    state: Any,
    candidate: Any,
    proposal: Any,
    evaluation: Any | None,
    promotion: Any | None,
) -> dict[str, str]:
    """Resolve one candidate's stages from state and scoped durable events."""

    candidate_status = candidate.status.value
    statuses = {
        "proposal": "completed",
        "candidate": "completed",
        "training": (
            "completed" if state.artifact_for(candidate.candidate_id) else "pending"
        ),
        "evaluation": "completed" if evaluation is not None else "pending",
        "judge": judge_stage_status(evaluation),
        "decision": "completed" if promotion is not None else "pending",
    }
    statuses.update(
        {
            stage: status
            for stage, status in _latest_stage_statuses(
                state,
                candidate.generation,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
            ).items()
            if stage in _EVOLUTION_STAGE_ORDER
        }
    )

    # Terminal core state is stronger than an older "started" observation.
    # Preserve a recorded failed stage when available; legacy failures without
    # stage evidence conservatively failed during training.
    if candidate_status == "duplicate":
        statuses.update(
            {
                "training": "skipped",
                "evaluation": "skipped",
                "judge": "skipped",
                "decision": "skipped",
            }
        )
    elif candidate_status == "failed":
        failed_stage = next(
            (
                stage
                for stage in reversed(_EVOLUTION_STAGE_ORDER)
                if statuses[stage] == "failed"
            ),
            "training",
        )
        failed_index = _EVOLUTION_STAGE_ORDER.index(failed_stage)
        for index, stage in enumerate(_EVOLUTION_STAGE_ORDER):
            if index < failed_index and statuses[stage] in {"pending", "running"}:
                statuses[stage] = "completed"
            elif index == failed_index:
                statuses[stage] = "failed"
            elif index > failed_index:
                statuses[stage] = "skipped"
    return statuses


def _latest_stage_identifier(
    state: Any, generation: int, field_name: str
) -> str | None:
    for event in reversed(state.events):
        if event.kind != "EvolutionStageRecorded":
            continue
        if event.payload.get("generation") != generation:
            continue
        value = event.payload.get(field_name)
        if isinstance(value, str):
            return value
    return None


def _nonnegative_metric(metrics: Any, key: str) -> int | None:
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0 or number != number or number in {float("inf"), float("-inf")}:
        return None
    return int(number)


def _round_execution_diagnostics(state: Any, candidates: list[Any]) -> dict[str, Any]:
    """Summarize actual work for one generation, separate from run totals."""

    totals = {
        "training_partition_rows": 0,
        "training_eligible_examples": 0,
        "training_used_examples": 0,
        "training_skipped_examples": 0,
        "evaluation_partition_rows": 0,
        "evaluation_eligible_examples": 0,
        "evaluation_used_examples": 0,
        "evaluation_skipped_examples": 0,
        "fit_passes_completed": 0,
    }
    source_counts: dict[str, int] = {}
    iterative_flags: list[bool] = []
    legacy_estimate = False
    artifact_count = 0
    evaluation_count = 0
    fallback_count = 0
    remote_calls = 0
    remote_successes = 0
    for candidate in candidates:
        proposal = state.proposal(candidate.proposal_id)
        metadata = proposal.metadata if isinstance(proposal.metadata, Mapping) else {}
        fallback = metadata.get("host_fallback")
        source = metadata.get("proposal_source")
        if not isinstance(source, str):
            source = "legacy_unknown"
        if isinstance(fallback, Mapping) and fallback.get("applied") is True:
            source = "host_fallback"
            fallback_count += 1
        source_counts[source] = source_counts.get(source, 0) + 1
        remote_calls += int(
            metadata.get("remote_strategy_called") is True
            or source in {"remote_model", "dsh_native_agent", "host_fallback"}
        )
        remote_successes += int(
            metadata.get("remote_strategy_succeeded") is True
            or source in {"remote_model", "dsh_native_agent"}
        )

        artifact = state.artifact_for(candidate.candidate_id)
        if artifact is not None:
            artifact_count += 1
            metrics = artifact.metrics
            partition_rows = _nonnegative_metric(metrics, "training_partition_rows")
            if partition_rows is None:
                partition_rows = max(0, int(artifact.training_rows))
                legacy_estimate = True
            used = _nonnegative_metric(metrics, "training_used_examples")
            if used is None:
                legacy_counts = [
                    _nonnegative_metric(metrics, key)
                    for key in metrics
                    if isinstance(key, str)
                    and key.endswith("_n")
                    and not key.startswith("evaluation_")
                ]
                legacy_counts = [item for item in legacy_counts if item is not None]
                used = sum(legacy_counts) if legacy_counts else partition_rows
                legacy_estimate = True
            skipped = _nonnegative_metric(metrics, "training_skipped_examples")
            skipped = 0 if skipped is None else skipped
            eligible = _nonnegative_metric(metrics, "training_eligible_examples")
            eligible = used + skipped if eligible is None else eligible
            passes = _nonnegative_metric(metrics, "fit_passes_completed")
            if passes is None:
                passes = _nonnegative_metric(metrics, "epochs_completed")
                passes = 1 if passes is None else passes
                legacy_estimate = True
            totals["training_partition_rows"] += partition_rows
            totals["training_eligible_examples"] += eligible
            totals["training_used_examples"] += used
            totals["training_skipped_examples"] += skipped
            totals["fit_passes_completed"] += passes
            iterative = metrics.get("iterative_epoch_training")
            if isinstance(iterative, bool):
                iterative_flags.append(iterative)

        evaluation = state.evaluation_for(candidate.candidate_id)
        if evaluation is not None:
            evaluation_count += 1
            metrics = evaluation.metrics
            used = _nonnegative_metric(metrics, "evaluation_used_examples")
            used = _nonnegative_metric(metrics, "n") if used is None else used
            used = 0 if used is None else used
            skipped = _nonnegative_metric(metrics, "evaluation_skipped_examples")
            if skipped is None:
                skipped = _nonnegative_metric(metrics, "missing_or_nonfinite_rows")
            skipped = 0 if skipped is None else skipped
            eligible = _nonnegative_metric(metrics, "evaluation_eligible_examples")
            eligible = used + skipped if eligible is None else eligible
            partition_rows = _nonnegative_metric(metrics, "evaluation_partition_rows")
            if partition_rows is None:
                partition_rows = used
                legacy_estimate = True
            totals["evaluation_partition_rows"] += partition_rows
            totals["evaluation_eligible_examples"] += eligible
            totals["evaluation_used_examples"] += used
            totals["evaluation_skipped_examples"] += skipped

    return {
        "proposal_attempts": len(candidates),
        "unique_candidates": sum(
            item.status.value != "duplicate" for item in candidates
        ),
        "failed_candidates": sum(item.status.value == "failed" for item in candidates),
        "duplicate_candidates": sum(
            item.status.value == "duplicate" for item in candidates
        ),
        "candidate_artifacts_count": artifact_count,
        "candidate_evaluations_count": evaluation_count,
        **totals,
        "candidate_work_items": (
            totals["training_used_examples"] + totals["evaluation_used_examples"]
        ),
        "iterative_epoch_training": any(iterative_flags),
        "legacy_workload_estimate_used": legacy_estimate,
        "proposal_sources": source_counts,
        "remote_strategy_calls": remote_calls,
        "remote_strategy_successes": remote_successes,
        "fallback_count": fallback_count,
    }


def _research_iteration_summary(iteration: Any | None) -> dict[str, Any] | None:
    """Project the per-generation research decision without raw prompt content."""

    if iteration is None:
        return None
    plan = dict(iteration.plan)
    adoption = dict(iteration.prediction_model_adoption)
    raw_historical_provenance = getattr(
        iteration,
        "historical_provenance",
        None,
    )
    historical_provenance = None
    if isinstance(raw_historical_provenance, Mapping):
        historical_provenance = {
            field_name: raw_historical_provenance[field_name]
            for field_name in (
                "source_digest",
                "scanned_run_count",
                "compatible_source_run_count",
                "included_source_run_count",
                "available_generation_count",
                "included_generation_count",
                "omitted_generation_summaries",
                "omitted_detail_count",
                "history_cutoff_seq",
            )
            if field_name in raw_historical_provenance
        }
    algorithm: dict[str, Any] = {}
    section_fields = {
        "algorithm_blueprint": (
            "schema_version",
            "pipeline_id",
            "operator_ids",
            "parameter_names",
            "evidence_refs",
        ),
        "algorithm_synthesis": (
            "schema_version",
            "pipeline_id",
            "evidence_refs",
            "parameter_focus",
            "rationale",
        ),
        "algorithm_synthesis_degradation": (
            "schema_version",
            "reason_code",
            "rationale",
        ),
    }
    for section_name, allowed_fields in section_fields.items():
        raw_section = plan.get(section_name)
        if not isinstance(raw_section, Mapping):
            continue
        section: dict[str, Any] = {}
        for field_name in allowed_fields:
            value = raw_section.get(field_name)
            if isinstance(value, str):
                section[field_name] = value[:1000]
            elif isinstance(value, (list, tuple)):
                section[field_name] = [
                    str(item)[:500]
                    for item in value[:32]
                    if isinstance(item, str) and item.strip()
                ]
        algorithm[section_name] = section

    synthesis = algorithm.get("algorithm_synthesis")
    degradation = algorithm.get("algorithm_synthesis_degradation")
    blueprint = algorithm.get("algorithm_blueprint")
    raw_research = plan.get("research")
    key_findings: list[dict[str, str]] = []
    if isinstance(raw_research, (list, tuple)):
        for raw_item in raw_research[:8]:
            if not isinstance(raw_item, Mapping):
                continue
            finding = raw_item.get("finding")
            if not isinstance(finding, str) or not finding.strip():
                continue
            item = {"finding": finding.strip()[:1000]}
            for field_name in ("title", "source", "relevance"):
                value = raw_item.get(field_name)
                if isinstance(value, str) and value.strip():
                    item[field_name] = value.strip()[:500]
            key_findings.append(item)

    analysis_source = (
        synthesis if isinstance(synthesis, Mapping) else degradation
        if isinstance(degradation, Mapping) else None
    )
    analysis_summary = {
        "schema_version": "ecologyrsi-dsh.literature-analysis-summary/1",
        "status": (
            "completed"
            if isinstance(synthesis, Mapping)
            else "degraded"
            if isinstance(degradation, Mapping)
            else "pending"
        ),
        "summary": (
            str(analysis_source.get("rationale"))[:2000]
            if isinstance(analysis_source, Mapping)
            and isinstance(analysis_source.get("rationale"), str)
            else None
        ),
        "evidence_refs": (
            list(analysis_source.get("evidence_refs", []))[:16]
            if isinstance(analysis_source, Mapping)
            and isinstance(analysis_source.get("evidence_refs"), list)
            else []
        ),
        "key_findings": key_findings,
        "source": "model_research_plan",
    }
    final_plan = {
        "schema_version": "ecologyrsi-dsh.final-implementation-plan/1",
        "status": (
            "ready_for_host_compilation"
            if isinstance(blueprint, Mapping) and isinstance(synthesis, Mapping)
            else "research_only"
            if isinstance(degradation, Mapping)
            else "pending"
        ),
        "predictor_id": adoption.get("adopted_id"),
        "pipeline_id": (
            blueprint.get("pipeline_id") if isinstance(blueprint, Mapping) else None
        ),
        "operator_ids": (
            list(blueprint.get("operator_ids", []))[:32]
            if isinstance(blueprint, Mapping)
            and isinstance(blueprint.get("operator_ids"), list)
            else []
        ),
        "parameter_names": (
            list(blueprint.get("parameter_names", []))[:32]
            if isinstance(blueprint, Mapping)
            and isinstance(blueprint.get("parameter_names"), list)
            else []
        ),
        "parameter_focus": (
            list(synthesis.get("parameter_focus", []))[:32]
            if isinstance(synthesis, Mapping)
            and isinstance(synthesis.get("parameter_focus"), list)
            else []
        ),
        "rationale": analysis_summary["summary"],
        "implementation_mode": "registered_host_components_only",
        "validation_sequence": [
            "compile_registered_ir",
            "training_fit_smoke_test",
            "training_feedback_evaluation",
            "independent_model_review",
        ],
    }
    result = {
        "schema_version": "ecologyrsi-dsh.research-iteration-summary/2",
        "status": iteration.status,
        "iteration_digest": iteration.iteration_digest,
        "plan_digest": adoption.get("plan_digest") or digest(plan),
        "knowledge_snapshot_digest": iteration.knowledge_snapshot_digest,
        "source_analysis_digest": iteration.source_analysis_digest,
        "source_assessment_digest": iteration.source_assessment_digest,
        "model_id": iteration.model_id,
        "predictor_adoption": {
            field_name: adoption.get(field_name)
            for field_name in (
                "status",
                "requested_id",
                "adopted_id",
                "reason",
                "adoption_digest",
                "parameter_schema_digest",
            )
        },
        "analysis_summary": analysis_summary,
        "final_plan": final_plan,
        **algorithm,
    }
    if historical_provenance is not None:
        result["historical_provenance"] = historical_provenance
    return result


def rounds(state: Any) -> list[dict[str, Any]]:
    """Derive compact batch and per-candidate stage records by generation."""

    stage_generations = {
        int(event.payload["generation"])
        for event in state.events
        if event.kind == "EvolutionStageRecorded"
    }
    generations = stage_generations | {item.generation for item in state.proposals}
    generations |= {item.generation for item in state.generation_batches}
    generations |= {item.generation for item in state.knowledge_snapshots}
    generations |= {item.generation for item in state.research_iterations}
    result: list[dict[str, Any]] = []
    for generation in sorted(generations):
        batch = state.batch_for(generation)
        analysis = state.analysis_for(generation)
        knowledge = state.knowledge_for(generation)
        research_iteration = state.research_iteration_for(generation)
        knowledge_assessment = state.knowledge_assessment_for(generation)
        candidates = sorted(
            (item for item in state.candidates if item.generation == generation),
            key=lambda item: item.slot_index,
        )
        candidate_rows = []
        intervention_ids: set[str] = set()
        for candidate in candidates:
            proposal = state.proposal(candidate.proposal_id)
            artifact = state.artifact_for(candidate.candidate_id)
            evaluation = state.evaluation_for(candidate.candidate_id)
            promotion = state.promotion_for(candidate.candidate_id)
            failed = candidate.status.value == "failed"
            duplicate = candidate.status.value == "duplicate"
            interventions = _applied_interventions(state, proposal.proposal_id)
            intervention_ids.update(item["intervention_id"] for item in interventions)
            stage_statuses = _candidate_stage_statuses(
                state,
                candidate,
                proposal,
                evaluation,
                promotion,
            )
            rank = next(
                (
                    dict(item)
                    for item in (analysis.ranking if analysis is not None else ())
                    if item.get("candidate_id") == candidate.candidate_id
                ),
                {},
            )
            candidate_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "proposal_id": proposal.proposal_id,
                    "slot_index": candidate.slot_index,
                    "stages": stage_statuses,
                    "score": evaluation.score if evaluation is not None else None,
                    "passed": evaluation.passed if evaluation is not None else None,
                    "decision": (
                        promotion.decision.value
                        if promotion is not None
                        else "duplicate"
                        if duplicate
                        else "failed"
                        if failed
                        else "pending"
                    ),
                    "rank": rank.get("rank"),
                    "eligible": rank.get("eligible"),
                    "classification": rank.get("classification"),
                    "selection_reason": rank.get("selection_reason"),
                    "proposal_source": proposal.metadata.get("proposal_source"),
                    "execution": {
                        "model_id": artifact.model_id if artifact is not None else None,
                        "training_rows": artifact.training_rows if artifact is not None else None,
                        "evaluation_rows": (
                            evaluation.metrics.get("n")
                            if evaluation is not None
                            and isinstance(evaluation.metrics, Mapping)
                            else None
                        ),
                        "execution_mode": (
                            artifact.metrics.get("execution_mode")
                            if artifact is not None
                            and isinstance(artifact.metrics, Mapping)
                            else None
                        ),
                        "fit_method": (
                            artifact.metrics.get("fit_method")
                            if artifact is not None
                            and isinstance(artifact.metrics, Mapping)
                            else None
                        ),
                        "training_used_examples": (
                            artifact.metrics.get("training_used_examples")
                            if artifact is not None
                            and isinstance(artifact.metrics, Mapping)
                            else None
                        ),
                        "training_skipped_examples": (
                            artifact.metrics.get("training_skipped_examples")
                            if artifact is not None
                            and isinstance(artifact.metrics, Mapping)
                            else None
                        ),
                        "evaluation_eligible_examples": (
                            evaluation.metrics.get("evaluation_eligible_examples")
                            if evaluation is not None
                            and isinstance(evaluation.metrics, Mapping)
                            else None
                        ),
                        "evaluation_used_examples": (
                            evaluation.metrics.get("evaluation_used_examples")
                            if evaluation is not None
                            and isinstance(evaluation.metrics, Mapping)
                            else None
                        ),
                        "fallback_used": (
                            isinstance(proposal.metadata.get("host_fallback"), Mapping)
                            and proposal.metadata.get("host_fallback", {}).get("applied")
                            is True
                        ),
                    },
                }
            )
        representative = (
            analysis.champion_candidate_id or analysis.selected_candidate_id
            if analysis is not None
            else candidates[0].candidate_id
            if candidates
            else None
        )
        representative_row = next(
            (item for item in candidate_rows if item["candidate_id"] == representative),
            candidate_rows[0] if candidate_rows else None,
        )
        representative_proposal_id = (
            representative_row.get("proposal_id")
            if representative_row is not None
            else _latest_stage_identifier(state, generation, "proposal_id")
        )
        representative_stages = (
            dict(representative_row["stages"])
            if representative_row is not None
            else {
                **{
                    "proposal": "pending",
                    "candidate": "pending",
                    "training": "pending",
                    "evaluation": "pending",
                    "judge": "pending",
                    "decision": "pending",
                },
                **_latest_stage_statuses(
                    state,
                    generation,
                    proposal_id=None,
                    candidate_id=None,
                ),
            }
        )
        result.append(
            {
                "schema_version": "ecologyrsi-dsh.evolution-round/2",
                "generation": generation + 1,
                "batch_size": batch.batch_size if batch is not None else max(1, len(candidates)),
                "parent_candidate_id": batch.parent_candidate_id if batch is not None else None,
                "context_digest": batch.context_digest if batch is not None else None,
                "knowledge": knowledge.to_dict() if knowledge is not None else None,
                "research_iteration": _research_iteration_summary(
                    research_iteration
                ),
                "knowledge_assessment": (
                    knowledge_assessment.to_dict()
                    if knowledge_assessment is not None
                    else None
                ),
                "candidate_count": len(candidates),
                "eligible_count": analysis.eligible_count if analysis is not None else 0,
                "proposal_id": representative_proposal_id,
                "candidate_id": representative,
                "stages": representative_stages,
                "score": representative_row.get("score") if representative_row else None,
                "decision": analysis.outcome if analysis is not None else "pending",
                "champion_candidate_id": analysis.champion_candidate_id if analysis is not None else None,
                "incumbent_after_candidate_id": analysis.incumbent_after_candidate_id if analysis is not None else None,
                "selection_reason": analysis.selection_reason if analysis is not None else "",
                "next_generation_focus": analysis.next_generation_focus if analysis is not None else "",
                "common_failures": list(analysis.common_failures) if analysis is not None else [],
                "target_weaknesses": [dict(item) for item in analysis.target_weaknesses] if analysis is not None else [],
                "horizon_weaknesses": [dict(item) for item in analysis.horizon_weaknesses] if analysis is not None else [],
                "parameter_effects": [dict(item) for item in analysis.parameter_effects] if analysis is not None else [],
                "insufficient_evidence": analysis.insufficient_evidence if analysis is not None else True,
                "candidates": candidate_rows,
                "execution_diagnostics": _round_execution_diagnostics(
                    state, candidates
                ),
                "applied_intervention_count": len(intervention_ids),
                "applied_intervention_ids": sorted(intervention_ids),
                "timing": _round_timing(state, generation),
            }
        )
    return result


def state_snapshot(state: Any) -> dict[str, Any]:
    """Return the complete durable projection without its event list."""

    return {
        "run": state.run.to_dict(),
        "task_manifest": state.task_manifest.to_dict(),
        "proposals": [item.to_dict() for item in state.proposals],
        "candidates": [item.to_dict() for item in state.candidates],
        "artifacts": [item.to_dict() for item in state.artifacts],
        "evaluations": [item.to_dict() for item in state.evaluations],
        "promotions": [item.to_dict() for item in state.promotions],
        "interventions": [item.to_dict() for item in state.interventions],
        "generation_batches": [item.to_dict() for item in state.generation_batches],
        "generation_analyses": [item.to_dict() for item in state.generation_analyses],
        "knowledge_snapshots": [item.to_dict() for item in state.knowledge_snapshots],
        "knowledge_assessments": [
            item.to_dict() for item in state.knowledge_assessments
        ],
        "event_count": len(state.events),
    }


def _event_snapshot(event: Any) -> dict[str, Any]:
    return {
        "seq": event.seq,
        "event_id": event.event_id,
        "run_id": event.run_id,
        "kind": event.kind,
        "payload": event.payload,
        "created_at": event.created_at,
    }


_PRIVATE_SAMPLE_EVENT_KINDS = frozenset(
    {
        "EvaluationSampleResultsStarted",
        "EvaluationSampleResultBatchRecorded",
        "EvaluationSampleResultsRecorded",
    }
)


def best_observed_evaluation(state: Any) -> Any | None:
    """Return the highest-scoring evaluation, independent of gate acceptance."""

    if not state.evaluations:
        return None
    return max(
        state.evaluations,
        key=lambda item: (item.score, item.created_at, item.evaluation_id),
    )


def run_completion_outcome(state: Any) -> tuple[str | None, str | None]:
    """Project a terminal result without changing the durable run-status enum."""

    if state.run.status.value != "completed":
        return None, None
    completed_event = next(
        (event for event in reversed(state.events) if event.kind == "RunCompleted"),
        None,
    )
    if completed_event is not None:
        raw_outcome = completed_event.payload.get("outcome")
        if isinstance(raw_outcome, str) and raw_outcome.strip():
            outcome = {
                "accepted": "completed_with_acceptable_candidate",
                "completed_with_search_retained_candidate": (
                    "completed_with_acceptable_candidate"
                ),
            }.get(raw_outcome.strip(), raw_outcome.strip())
            raw_reason = completed_event.payload.get("termination_reason")
            termination_reason = (
                raw_reason.strip()
                if isinstance(raw_reason, str) and raw_reason.strip()
                else None
            )
            return outcome, termination_reason

    # Legacy completion events had an empty payload.  Retain a deterministic
    # read-only inference for those ledgers without rewriting their history.
    outcome = (
        "completed_with_acceptable_candidate"
        if state.run.best_candidate_id is not None
        else "budget_exhausted_without_acceptable_candidate"
    )
    return outcome, outcome


def run_summary(state: Any) -> dict[str, Any]:
    """Build a compact, audit-friendly summary for one run."""

    task = state.task_manifest
    metadata = dict(task.metadata)
    evaluations = tuple(state.evaluations)
    approved = sum(1 for item in state.promotions if item.decision.value == "approved")
    rejected = sum(1 for item in state.promotions if item.decision.value == "rejected")
    failed = sum(1 for item in state.candidates if item.status.value == "failed")
    partitions = sorted({item.partition for item in evaluations})
    policy_model_id = metadata.get(
        "policy_model_id", HOST_PARAMETER_GENERATOR_ID
    )
    judge_model_id = metadata.get("judge_model_id", RULE_JUDGE_ID)
    policy_model_digest = metadata.get("policy_model_digest") or (
        builtin_model_configuration_digest(str(policy_model_id))
    )
    judge_model_digest = metadata.get("judge_model_digest") or (
        builtin_model_configuration_digest(str(judge_model_id))
    )

    acceptable_evaluation = None
    if state.run.best_candidate_id:
        acceptable_evaluation = state.evaluation_for(state.run.best_candidate_id)
    observed_evaluation = best_observed_evaluation(state)
    outcome, termination_reason = run_completion_outcome(state)

    summary: dict[str, Any] = {
        "run_id": state.run.run_id,
        "status": state.run.status.value,
        "task_id": task.task_id,
        "objective": task.objective,
        "domain_pack": task.domain_pack,
        "dataset_id": task.dataset,
        "manifest_digest": task.digest,
        "dataset_digest": metadata.get("dataset_digest"),
        "split_manifest_digest": metadata.get("split_manifest_digest"),
        "strategy_id": metadata.get("strategy_id", "parameter_sweep@1"),
        "strategy_digest": metadata.get("strategy_digest"),
        "prediction_model_id": metadata.get("prediction_model_id"),
        "prediction_model_digest": metadata.get("prediction_model_digest"),
        "evaluator_id": metadata.get("evaluator_id"),
        "evaluator_digest": metadata.get("evaluator_digest"),
        "policy_model_id": policy_model_id,
        "policy_model_digest": policy_model_digest,
        "judge_model_id": judge_model_id,
        "judge_model_digest": judge_model_digest,
        "seed": task.seed,
        "seed_policy": task.seed_policy,
        "policy_version": task.policy_version,
        "scientific_scope": metadata.get("scientific_scope", "unspecified"),
        "selection_scope": "iterative_training_feedback_only",
        "formal_validation_status": "not_run",
        "best_candidate_scope": (
            "iterative_training_feedback_only"
            if state.run.best_candidate_id is not None
            else None
        ),
        "evaluation_partition": partitions or [metadata.get("evaluation_partition", "unspecified")],
        "generation": state.run.generation,
        "candidate_count": len(state.candidates),
        "candidates_per_generation": task.candidates_per_generation,
        "generation_analysis_count": len(state.generation_analyses),
        "knowledge_snapshot_count": len(state.knowledge_snapshots),
        "knowledge_assessment_count": len(state.knowledge_assessments),
        "artifact_count": len(state.artifacts),
        "evaluation_count": len(evaluations),
        "intervention_count": len(state.interventions),
        "training_asset_count": len(state.candidates),
        "round_count": len(rounds(state)),
        "approved_count": approved,
        "rejected_count": rejected,
        "failed_count": failed,
        "outcome": outcome,
        "termination_reason": termination_reason,
        "best_candidate_id": state.run.best_candidate_id,
        "best_candidate_score": (
            acceptable_evaluation.score if acceptable_evaluation is not None else None
        ),
        "best_score": (
            acceptable_evaluation.score if acceptable_evaluation is not None else None
        ),
        "best_metrics": (
            dict(acceptable_evaluation.metrics)
            if acceptable_evaluation is not None
            else {}
        ),
        "best_observed_candidate_id": (
            observed_evaluation.candidate_id
            if observed_evaluation is not None
            else None
        ),
        "best_observed_score": (
            observed_evaluation.score if observed_evaluation is not None else None
        ),
        "best_observed_metrics": (
            dict(observed_evaluation.metrics)
            if observed_evaluation is not None
            else {}
        ),
        "event_count": len(state.events),
        "last_event_seq": state.events[-1].seq if state.events else 0,
    }
    return summary


def run_export(state: Any) -> dict[str, Any]:
    """Return a self-contained JSON export with a content digest."""

    payload: dict[str, Any] = {
        "format": "ecologyrsi-dsh.run-export",
        "format_version": 1,
        "package_version": __version__,
        "redaction": "projection-only",
        "summary": run_summary(state),
        "training_assets": training_assets(state),
        "rounds": rounds(state),
        "state": state_snapshot(state),
        "events": [
            _event_snapshot(item)
            for item in state.events
            if item.kind not in _PRIVATE_SAMPLE_EVENT_KINDS
        ],
    }
    payload["export_digest"] = digest(payload)
    return payload


def export_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return structural or digest errors in an exported run bundle."""

    errors: list[str] = []
    if payload.get("format") != "ecologyrsi-dsh.run-export":
        errors.append("unsupported export format")
    if payload.get("format_version") != 1:
        errors.append("unsupported export format version")
    if payload.get("redaction") != "projection-only":
        errors.append("export redaction marker is missing or unsupported")
    stored = payload.get("export_digest")
    unsigned = dict(payload)
    unsigned.pop("export_digest", None)
    try:
        computed = digest(unsigned)
    except (TypeError, ValueError) as exc:
        errors.append(f"export is not canonical JSON: {exc}")
    else:
        if not isinstance(stored, str) or stored != computed:
            errors.append("export digest mismatch")

    summary = payload.get("summary")
    if not isinstance(summary, Mapping) or not isinstance(summary.get("run_id"), str):
        errors.append("export summary is missing run_id")
        summary_run_id = None
    else:
        summary_run_id = str(summary["run_id"])
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        errors.append("export events must be a non-empty list")
        events = []
    previous_seq: int | None = None
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            errors.append(f"event {index} is not an object")
            continue
        for field_name in ("event_id", "run_id", "kind", "created_at"):
            if not isinstance(event.get(field_name), str) or not str(event[field_name]).strip():
                errors.append(f"event {index} has invalid {field_name}")
        if not isinstance(event.get("payload"), Mapping):
            errors.append(f"event {index} payload is not an object")
        if summary_run_id is not None and event.get("run_id") != summary_run_id:
            errors.append(f"event {index} belongs to a different run")
        seq = event.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            errors.append(f"event {index} has invalid seq")
        elif previous_seq is not None and seq <= previous_seq:
            errors.append(f"event {index} sequence is not strictly increasing")
        else:
            previous_seq = seq
    return errors


def write_json_atomic(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    force: bool = False,
) -> Path:
    """Write JSON beside its target, then atomically replace the target."""

    target = Path(path).expanduser().resolve()
    if target.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return target
