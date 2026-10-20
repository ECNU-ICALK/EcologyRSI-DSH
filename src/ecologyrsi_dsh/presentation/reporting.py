"""Run summaries, stage projections, and export helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.models import CandidateRole, digest
from ..evolution.context import safe_aggregate_feedback


_EVOLUTION_STAGE_ORDER = (
    "proposal",
    "candidate",
    "training",
    "evaluation",
    "judge",
    "decision",
)


def _search_design_audit(value: Any) -> dict[str, Any] | None:
    """Expose only bounded, aggregate candidate-design diagnostics."""

    if not isinstance(value, Mapping):
        return None
    allowed = {
        "policy",
        "adopted_parameter",
        "parameter_source",
        "host_projection_applied",
        "sibling_unique",
        "target_focus",
        "remote_changed_parameters",
    }
    result = {name: value[name] for name in allowed if name in value}
    reference = value.get("shared_reference_parameters")
    if isinstance(reference, Mapping):
        result["shared_reference_parameters"] = {
            str(name): item
            for name, item in list(reference.items())[:32]
            if isinstance(name, str)
            and isinstance(item, (int, float))
            and not isinstance(item, bool)
        }
    return result


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
    """Sum explicit native measurements; absent measurements remain unknown."""
    training_keys = ("training_partition_rows", "training_eligible_examples",
                     "training_used_examples", "training_skipped_examples", "fit_passes_completed")
    evaluation_keys = ("evaluation_partition_rows", "evaluation_eligible_examples",
                       "evaluation_used_examples", "evaluation_skipped_examples")
    artifacts = [a for c in candidates if (a := state.artifact_for(c.candidate_id)) is not None]
    evaluations = [e for c in candidates if (e := state.evaluation_for(c.candidate_id)) is not None]
    totals = {}
    for records, keys in ((artifacts, training_keys), (evaluations, evaluation_keys)):
        for key in keys:
            values = [_nonnegative_metric(record.metrics, key) for record in records]
            totals[key] = sum(values) if values and all(v is not None for v in values) else None
    sources = {}
    for candidate in candidates:
        source = state.proposal(candidate.proposal_id).metadata.get("proposal_source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    used = [totals["training_used_examples"], totals["evaluation_used_examples"]]
    return {
        "proposal_attempts": len(candidates),
        "unique_candidates": sum(c.status.value != "duplicate" for c in candidates),
        "failed_candidates": sum(c.status.value == "failed" for c in candidates),
        "duplicate_candidates": sum(c.status.value == "duplicate" for c in candidates),
        "candidate_artifacts_count": len(artifacts),
        "candidate_evaluations_count": len(evaluations),
        **totals,
        "candidate_work_items": sum(used) if all(v is not None for v in used) else None,
        "iterative_epoch_training": any(a.metrics.get("iterative_epoch_training") is True for a in artifacts),
        "proposal_sources": sources,
        "remote_strategy_calls": sources.get("dsh_native_agent", 0),
        "remote_strategy_successes": sources.get("dsh_native_agent", 0),
    }


def _research_iteration_summary(iteration: Any | None) -> dict[str, Any] | None:
    """Expose only the native research contract's bounded public fields."""
    if iteration is None:
        return None
    plan = iteration.plan
    def fields(value, names):
        return {name: value.get(name) for name in names.split()} if isinstance(value, Mapping) else {}
    summary = plan.get("dsh_research_summary")
    summary = summary.strip()[:2000] if isinstance(summary, str) and summary.strip() else None
    findings, references = [], []
    for value in plan.get("dsh_research_evidence", ())[:8]:
        if not isinstance(value, Mapping) or not isinstance(value.get("finding"), str):
            continue
        item = {key: value[key].strip()[:1000 if key == "finding" else 500]
                for key in ("finding", "title", "source", "relevance", "knowledge_id", "evidence_digest")
                if isinstance(value.get(key), str) and value[key].strip()}
        findings.append(item)
        for key in ("evidence_ref", "knowledge_id", "evidence_digest"):
            reference = value.get(key)
            if isinstance(reference, str) and reference.strip() and reference not in references:
                references.append(reference.strip()[:500])
    result = {
        "schema_version": "ecologyrsi-dsh.research-iteration-summary/2",
        **fields(iteration.to_dict(), "status iteration_digest knowledge_snapshot_digest source_analysis_digest source_assessment_digest model_id"),
        "iteration_digest": iteration.iteration_digest,
        "plan_digest": digest(plan),
        "analysis_summary": {
            "schema_version": "ecologyrsi-dsh.literature-analysis-summary/1",
            "status": "completed" if summary is not None else "pending",
            "summary": summary, "evidence_refs": references[:16], "key_findings": findings,
            "source": "dsh_native_research",
        },
        "final_plan": {
            "schema_version": "ecologyrsi-dsh.final-implementation-plan/1",
            "status": "ready_for_host_compilation" if summary is not None else "pending",
            "predictor_id": None, "pipeline_id": None,
            "operator_ids": [], "parameter_names": [], "parameter_focus": [], "rationale": summary,
            "implementation_mode": "registered_host_components_only",
            "validation_sequence": ["compile_registered_ir", "training_fit_smoke_test", "training_feedback_evaluation", "independent_model_review"],
        },
        "candidate_directions": [fields(item, "direction_id direction_digest title hypothesis target_weakness capability_focus mutation_axis mutation_target evidence_refs expected_tradeoff success_criterion")
            for item in plan.get("candidate_directions", ())[:8] if isinstance(item, Mapping)],
    }
    provenance = getattr(iteration, "historical_provenance", None)
    if isinstance(provenance, Mapping):
        result["historical_provenance"] = fields(provenance, "source_digest scanned_run_count compatible_source_run_count included_source_run_count available_generation_count included_generation_count omitted_generation_summaries omitted_detail_count history_cutoff_seq")
    reflection = plan.get("dsh_evolution_reflection")
    if isinstance(reflection, Mapping):
        reflection = safe_aggregate_feedback(reflection, name="DSH evolution reflection reporting projection")
        if reflection is not None:
            result["evolution_reflection"] = {
                "schema_version": reflection.get("schema_version"),
                "avoid_behavior_count": len(reflection.get("avoid_behaviors", ())),
                "review_behaviors": [fields(item, "behavior_digest reason source_run_id source_generation") for item in reflection.get("review_behaviors", ())[:8]],
                "avoid_behaviors": [fields(item, "behavior_digest prediction_model_id parameters_digest reason source_run_id source_generation") for item in reflection.get("avoid_behaviors", ())[:8]],
                "active_unresolved": list(reflection.get("active_unresolved", ()))[:8],
                "policy": dict(reflection.get("policy", {})),
            }
    if getattr(iteration, "diagnostic_report", None) is not None:
        from ..core.research import DiagnosticReport
        diagnostic = DiagnosticReport(**dict(iteration.diagnostic_report))
        result["diagnostic_report"] = {**diagnostic.to_dict(), "report_id": diagnostic.report_id}
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
    generations |= {item.generation for item in state.generation_search_plans}
    generations |= {item.generation for item in state.generation_reflections}
    result: list[dict[str, Any]] = []
    for generation in sorted(generations):
        batch = state.batch_for(generation)
        analysis = state.analysis_for(generation)
        knowledge = state.knowledge_for(generation)
        search_plan = state.search_plan_for(generation)
        research_iteration = state.research_iteration_for(generation)
        generation_reflection = state.reflection_for(generation)
        knowledge_assessment = state.knowledge_assessment_for(generation)
        candidates = sorted(
            (
                item
                for item in state.candidates
                if item.generation == generation
                and getattr(item, "role", CandidateRole.SEARCH)
                in {CandidateRole.SEARCH, CandidateRole.SEARCH.value}
            ),
            key=lambda item: item.slot_index,
        )
        candidate_rows = []
        for candidate in candidates:
            proposal = state.proposal(candidate.proposal_id)
            artifact = state.artifact_for(candidate.candidate_id)
            evaluation = state.evaluation_for(candidate.candidate_id)
            promotion = state.promotion_for(candidate.candidate_id)
            failed = candidate.status.value == "failed"
            duplicate = candidate.status.value == "duplicate"
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
                    "search_design_audit": _search_design_audit(
                        proposal.metadata.get("search_design_audit")
                    ),
                    "candidate_direction_id": proposal.metadata.get(
                        "candidate_direction_id"
                    ),
                    "candidate_direction_digest": proposal.metadata.get(
                        "candidate_direction_digest"
                    ),
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
            analysis.champion_candidate_id
            or analysis.selected_candidate_id
            or analysis.search_parent_candidate_id
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
                "search_plan": (
                    search_plan.to_dict() if search_plan is not None else None
                ),
                "research_iteration": _research_iteration_summary(
                    research_iteration
                ),
                "generation_reflection": (
                    generation_reflection.to_dict()
                    if generation_reflection is not None
                    else None
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
        "generation_batches": [item.to_dict() for item in state.generation_batches],
        "generation_analyses": [item.to_dict() for item in state.generation_analyses],
        "knowledge_snapshots": [item.to_dict() for item in state.knowledge_snapshots],
        "knowledge_assessments": [
            item.to_dict() for item in state.knowledge_assessments
        ],
        "generation_search_plans": [
            item.to_dict() for item in state.generation_search_plans
        ],
        "generation_reflections": [
            item.to_dict() for item in state.generation_reflections
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
    search_candidates = tuple(
        item
        for item in state.candidates
        if getattr(item, "role", CandidateRole.SEARCH)
        in {CandidateRole.SEARCH, CandidateRole.SEARCH.value}
    )
    failed = sum(
        1 for item in search_candidates if item.status.value == "failed"
    )
    partitions = sorted({item.partition for item in evaluations})
    policy_model_id = metadata.get("strategy_model_id")
    judge_model_id = metadata.get("review_model_id")
    policy_model_digest = metadata.get("policy_model_digest")
    judge_model_digest = metadata.get("judge_model_digest")

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
        "candidate_count": len(search_candidates),
        "candidates_per_generation": task.candidates_per_generation,
        "generation_analysis_count": len(state.generation_analyses),
        "knowledge_snapshot_count": len(state.knowledge_snapshots),
        "knowledge_assessment_count": len(state.knowledge_assessments),
        "artifact_count": len(state.artifacts),
        "evaluation_count": len(evaluations),
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
