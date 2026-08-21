"""Browser-safe run, candidate, and intervention projections."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ..core.errors import (
    FROZEN_RUNTIME_BINDING_DRIFT_CODE,
    FROZEN_RUNTIME_BINDING_DRIFT_PUBLIC_MESSAGE,
)
from ..core.models import HumanIntervention, InterventionKind
from ..core.redaction import (
    public_error_summary,
    safe_error_code,
    sanitize_public_value,
)
from ..core.sample_results import (
    MAX_SAMPLE_RESULTS_RECORDS,
    SAMPLE_REWARD_DEFINITION_V1,
)
from ..evaluators.registry import (
    EXOGENOUS_RIDGE_MODEL_ID,
    GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID,
    TOY_DATASET_ID,
    TOY_EVALUATOR_ID,
    TOY_PREDICTOR_MODEL_ID,
)
from ..evolution.analysis import (
    evaluation_cohort_comparison,
    evaluation_cohort_digest,
    sample_update_windows_enabled,
)
from ..integrations.model_bindings import HOST_PARAMETER_GENERATOR_ID, RULE_JUDGE_ID
from ..presentation.reporting import (
    _EVOLUTION_STAGE_ORDER,
    _candidate_stage_statuses,
    best_observed_evaluation,
    rounds,
    run_completion_outcome,
    training_assets,
)
from .shared import (
    _assert_http_scope,
    _budget_value,
    _expected_partition,
    _max_generations,
)

# The browser projection is intentionally a compact operational trace.  It is
# not a model chain-of-thought export: only values already produced by the
# evaluator and a fixed list of host-controlled steps are exposed.
_INFERENCE_TRACE_LIMIT = 48
_TERMINAL_STAGE_STATUSES = frozenset({"completed", "failed", "skipped", "not_recorded"})
_TERMINAL_RUN_STATUSES = frozenset({"completed", "cancelled", "failed"})
_SAMPLE_TOKEN_BUDGET_POLICY = "hard_gateway_call_reservation@1"
_SAMPLE_TOKEN_BUDGET_SCOPE = "sample_agent_gateway_calls_only@1"
_SENSITIVE_PLAN_KEYS = frozenset(
    {
        "private_reasoning",
        "prompt",
        "raw",
        "reasoning",
        "思维链",
        "私密推理",
    }
)


def _run_failure_code(state: Any) -> str | None:
    """Recover a host-owned code from current or legacy RunFailed payloads."""

    failed_event = next(
        (event for event in reversed(state.events) if event.kind == "RunFailed"),
        None,
    )
    if failed_event is None:
        return None
    direct = safe_error_code(failed_event.payload.get("error_code"))
    if direct is not None:
        return direct
    reason = failed_event.payload.get("reason")
    if (
        isinstance(reason, str)
        and f"[{FROZEN_RUNTIME_BINDING_DRIFT_CODE}]" in reason
    ):
        return FROZEN_RUNTIME_BINDING_DRIFT_CODE
    return None


def _run_failure_projection(state: Any) -> tuple[str | None, dict[str, Any] | None]:
    """Expose the durable public failure reason without requiring event joins."""

    failed_event = next(
        (event for event in reversed(state.events) if event.kind == "RunFailed"),
        None,
    )
    reason = None
    failure_code = _run_failure_code(state)
    if failed_event is not None:
        raw_reason = failed_event.payload.get("reason")
        if isinstance(raw_reason, str) and raw_reason.strip():
            reason = public_error_summary(raw_reason)
    if failure_code == FROZEN_RUNTIME_BINDING_DRIFT_CODE:
        reason = FROZEN_RUNTIME_BINDING_DRIFT_PUBLIC_MESSAGE
    stage_event = next(
        (
            event
            for event in reversed(state.events)
            if event.kind == "EvolutionStageRecorded"
            and event.payload.get("status") == "failed"
        ),
        None,
    )
    if stage_event is None:
        return reason, None
    payload = stage_event.payload
    return reason, {
        "generation": payload.get("generation"),
        "stage": payload.get("stage"),
        "attempt": payload.get("attempt"),
        "proposal_id": payload.get("proposal_id"),
        "candidate_id": payload.get("candidate_id"),
        "public_error": public_error_summary(payload.get("public_error")),
        "created_at": stage_event.created_at,
    }


def _run_pause_projection(state: Any) -> tuple[str | None, str | None]:
    """Return the active pause cause without reviving an older pause event."""

    if state.run.status.value != "paused":
        return None, None
    paused_event = next(
        (event for event in reversed(state.events) if event.kind == "RunPaused"),
        None,
    )
    if paused_event is None:
        return None, None
    raw_reason = paused_event.payload.get("reason")
    reason = (
        public_error_summary(raw_reason)
        if isinstance(raw_reason, str) and raw_reason.strip()
        else None
    )
    raw_code = paused_event.payload.get("code")
    code = str(raw_code) if isinstance(raw_code, str) and raw_code else None
    return reason, code


def _safe_plan_value(value: Any, *, depth: int = 0) -> Any:
    """Bound model-plan values before placing them in a browser response."""

    return sanitize_public_value(
        value,
        extra_sensitive_keys=_SENSITIVE_PLAN_KEYS,
        depth=depth,
        max_depth=6,
        text_limit=1000,
        sequence_limit=32,
    )


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _effective_stage_statuses(
    state: Any,
    statuses: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve stale started observations against the durable run lifecycle."""

    run_status = state.run.status.value
    replacement = (
        "paused"
        if run_status == "paused"
        else "not_recorded"
        if run_status in _TERMINAL_RUN_STATUSES
        else None
    )
    if replacement is None:
        return dict(statuses)
    return {
        name: replacement if status == "running" else status
        for name, status in statuses.items()
    }


def _effective_candidate_status(
    state: Any,
    candidate: Any,
    default: str,
) -> str:
    """Avoid presenting an unfinished candidate as active after its run stops."""

    if candidate.status.value != "spawned":
        return default
    run_status = state.run.status.value
    if run_status == "paused":
        return "paused"
    if run_status in _TERMINAL_RUN_STATUSES:
        return "aborted"
    return default


def _rounds_projection(state: Any) -> list[dict[str, Any]]:
    """Apply operational run lifecycle semantics to the audit-derived rounds."""

    projected = rounds(state)
    for round_item in projected:
        stage_statuses = round_item.get("stages")
        if isinstance(stage_statuses, Mapping):
            round_item["stages"] = _effective_stage_statuses(
                state,
                stage_statuses,
            )
        candidate_rows = round_item.get("candidates")
        if not isinstance(candidate_rows, list):
            continue
        for candidate_row in candidate_rows:
            if not isinstance(candidate_row, dict):
                continue
            candidate_stages = candidate_row.get("stages")
            if isinstance(candidate_stages, Mapping):
                candidate_row["stages"] = _effective_stage_statuses(
                    state,
                    candidate_stages,
                )
    return projected


def _execution_diagnostics(state: Any) -> dict[str, Any]:
    """Summarize what actually ran, including lightweight fits and fallbacks."""

    artifacts = list(state.artifacts)
    evaluations = list(state.evaluations)
    fallback_reasons: list[str] = []
    fallback_count = 0
    source_counts: dict[str, int] = {}
    remote_strategy_calls = 0
    remote_strategy_successes = 0
    research_attempts: dict[tuple[int, int], str] = {}
    for event in state.events:
        if (
            event.kind != "EvolutionStageRecorded"
            or event.payload.get("stage") != "research"
        ):
            continue
        generation = event.payload.get("generation")
        attempt = event.payload.get("attempt")
        status = event.payload.get("status")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or status not in {"started", "completed", "failed"}
        ):
            continue
        key = (generation, attempt)
        if status == "started":
            research_attempts.setdefault(key, status)
        elif key in research_attempts:
            research_attempts[key] = status
    remote_research_calls = len(research_attempts)
    remote_research_successes = sum(
        status == "completed" for status in research_attempts.values()
    )
    remote_research_running = any(
        status == "started" for status in research_attempts.values()
    )
    remote_strategy_calls += remote_research_calls
    remote_strategy_successes += remote_research_successes
    for proposal in state.proposals:
        metadata = proposal.metadata if isinstance(proposal.metadata, Mapping) else {}
        fallback = metadata.get("host_fallback") if isinstance(metadata, Mapping) else None
        if isinstance(fallback, Mapping) and fallback.get("applied") is True:
            fallback_count += 1
            reason = public_error_summary(
                fallback.get("public_error")
                or fallback.get("reason")
                or "remote_strategy_gateway_error"
            )
            assert reason is not None
            reason = reason[:300]
            if reason not in fallback_reasons:
                fallback_reasons.append(reason)
        source = metadata.get("proposal_source")
        if not isinstance(source, str) or source not in {
            "remote_model",
            "dsh_native_agent",
            "host_reserved_seed",
            "host_fallback",
            "host_strategy",
        }:
            source = "legacy_unknown"
        if isinstance(fallback, Mapping) and fallback.get("applied") is True:
            source = "host_fallback"
        source_counts[source] = source_counts.get(source, 0) + 1
        called = metadata.get("remote_strategy_called") is True or source in {
            "remote_model",
            "dsh_native_agent",
            "host_fallback",
        }
        succeeded = metadata.get("remote_strategy_succeeded") is True or source in {
            "remote_model",
            "dsh_native_agent",
        }
        remote_strategy_calls += int(called)
        remote_strategy_successes += int(succeeded)
    modes = sorted(
        {
            str(item.metrics.get("execution_mode"))
            for item in artifacts
            if isinstance(item.metrics, Mapping) and item.metrics.get("execution_mode")
        }
    )
    fit_methods = sorted(
        {
            str(item.metrics.get("fit_method"))
            for item in artifacts
            if isinstance(item.metrics, Mapping) and item.metrics.get("fit_method")
        }
    )
    model_ids = sorted({str(item.model_id) for item in artifacts if item.model_id})

    def metric_count(metrics: Any, key: str) -> int | None:
        if not isinstance(metrics, Mapping):
            return None
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(float(value)):
            return None
        return max(0, int(value))

    training_partition_rows = 0
    training_eligible_examples = 0
    training_used_examples = 0
    training_skipped_examples = 0
    fit_passes_completed = 0
    per_artifact_fit_passes: list[int] = []
    iterative_training_flags: list[bool] = []
    legacy_workload_estimate_used = False
    for artifact in artifacts:
        metrics = artifact.metrics
        partition_rows = metric_count(metrics, "training_partition_rows")
        partition_rows = (
            max(0, int(artifact.training_rows))
            if partition_rows is None
            else partition_rows
        )
        used = metric_count(metrics, "training_used_examples")
        if used is None:
            legacy_target_counts = [
                metric_count(metrics, key)
                for key in metrics
                if isinstance(key, str)
                and key.endswith("_n")
                and not key.startswith("evaluation_")
            ]
            legacy_target_counts = [
                item for item in legacy_target_counts if item is not None
            ]
            used = sum(legacy_target_counts) if legacy_target_counts else partition_rows
            legacy_workload_estimate_used = True
        skipped = metric_count(metrics, "training_skipped_examples")
        if skipped is None:
            skipped = 0
        eligible = metric_count(metrics, "training_eligible_examples")
        if eligible is None:
            eligible = used + skipped
        passes = metric_count(metrics, "fit_passes_completed")
        if passes is None:
            passes = metric_count(metrics, "epochs_completed")
            legacy_workload_estimate_used = True
        passes = 1 if passes is None else passes
        iterative_flag = (
            metrics.get("iterative_epoch_training")
            if isinstance(metrics, Mapping)
            else None
        )
        if isinstance(iterative_flag, bool):
            iterative_training_flags.append(iterative_flag)
        training_partition_rows += partition_rows
        training_eligible_examples += eligible
        training_used_examples += used
        training_skipped_examples += skipped
        fit_passes_completed += passes
        per_artifact_fit_passes.append(passes)

    evaluation_partition_rows = 0
    evaluation_eligible_examples = 0
    evaluation_used_examples = 0
    evaluation_skipped_examples = 0
    evaluation_available_examples = 0
    evaluation_selected_examples = 0
    evaluation_deferred_examples = 0
    for evaluation in evaluations:
        metrics = evaluation.metrics
        used = metric_count(metrics, "evaluation_used_examples")
        if used is None:
            used = metric_count(metrics, "n") or 0
        skipped = metric_count(metrics, "evaluation_skipped_examples")
        eligible = metric_count(metrics, "evaluation_eligible_examples")
        targets = metrics.get("targets") if isinstance(metrics, Mapping) else None
        if eligible is None and isinstance(targets, list):
            target_eligible = [
                metric_count(item, "eligible_rows") for item in targets
            ]
            if target_eligible and all(item is not None for item in target_eligible):
                eligible = sum(int(item) for item in target_eligible if item is not None)
        if skipped is None:
            skipped = metric_count(metrics, "missing_or_nonfinite_rows")
        skipped = 0 if skipped is None else skipped
        eligible = used + skipped if eligible is None else eligible
        partition_rows = metric_count(metrics, "evaluation_partition_rows")
        if partition_rows is None:
            # Legacy evaluators did not distinguish physical partition rows
            # from target/horizon work items.  Preserve a conservative value.
            partition_rows = used
        evaluation_partition_rows += partition_rows
        evaluation_eligible_examples += eligible
        evaluation_used_examples += used
        evaluation_skipped_examples += skipped
        available = metric_count(metrics, "evaluation_available_examples")
        selected_count = metric_count(metrics, "evaluation_selected_examples")
        deferred = metric_count(metrics, "evaluation_deferred_examples")
        evaluation_available_examples += eligible if available is None else available
        evaluation_selected_examples += eligible if selected_count is None else selected_count
        evaluation_deferred_examples += 0 if deferred is None else deferred

    # Evaluation progress is durable before the artifact and EvaluationRecorded
    # event are sealed. Count only the latest active revision for candidates
    # without a final evaluation, so live work is visible without double
    # counting it once the evaluator's authoritative metrics arrive.
    evaluated_candidate_ids = {item.candidate_id for item in evaluations}
    live_evaluation_completed_examples = 0
    live_evaluation_total_examples = 0
    live_evaluation_succeeded_examples = 0
    live_evaluation_failed_examples = 0
    live_evaluation_candidate_count = 0
    live_evaluation_outcome_counts_known = True
    partial_evaluation_active_candidate_count = 0
    partial_evaluation_retained_candidate_count = 0
    partial_evaluation_aborted_candidate_count = 0
    partial_evaluation_sources: set[str] = set()
    for candidate in state.candidates:
        if candidate.candidate_id in evaluated_candidate_ids:
            continue
        progress = _evaluation_progress_projection(state, candidate.candidate_id)
        batch_progress = _evaluation_batch_progress_projection(
            state, candidate.candidate_id
        )
        if batch_progress is not None and (
            progress is None
            or int(batch_progress["completed_samples"])
            > int(progress.get("completed_samples") or 0)
        ):
            heartbeat_total = (
                metric_count(progress, "total_samples") if progress is not None else None
            )
            progress = dict(batch_progress)
            progress["total_samples"] = max(
                int(batch_progress["total_samples"]), heartbeat_total or 0
            )
        if progress is None:
            continue
        completed = metric_count(progress, "completed_samples")
        if completed is None or completed <= 0:
            continue
        total = metric_count(progress, "total_samples")
        succeeded = metric_count(progress, "succeeded_samples")
        failed = metric_count(progress, "failed_samples")
        live_evaluation_completed_examples += completed
        live_evaluation_total_examples += max(completed, total or completed)
        if succeeded is None or failed is None:
            live_evaluation_outcome_counts_known = False
        else:
            live_evaluation_succeeded_examples += succeeded
            live_evaluation_failed_examples += failed
        live_evaluation_candidate_count += 1
        partial_evaluation_sources.add(
            str(progress.get("evidence_source") or "planner_progress_heartbeat")
        )
        run_status = state.run.status.value
        candidate_status = candidate.status.value
        if run_status == "running" and candidate_status == "spawned":
            partial_evaluation_active_candidate_count += 1
        elif run_status in {"completed", "cancelled", "failed"} or candidate_status in {
            "failed",
            "duplicate",
        }:
            partial_evaluation_aborted_candidate_count += 1
        else:
            partial_evaluation_retained_candidate_count += 1

    partial_status_category_count = sum(
        count > 0
        for count in (
            partial_evaluation_active_candidate_count,
            partial_evaluation_retained_candidate_count,
            partial_evaluation_aborted_candidate_count,
        )
    )
    if partial_status_category_count > 1:
        execution_evidence_status = "mixed_partial"
    elif partial_evaluation_active_candidate_count:
        execution_evidence_status = "partial_live"
    elif partial_evaluation_aborted_candidate_count:
        execution_evidence_status = "aborted_partial"
    elif partial_evaluation_retained_candidate_count:
        execution_evidence_status = "retained_partial"
    elif artifacts or evaluations:
        execution_evidence_status = "recorded"
    else:
        execution_evidence_status = "none"

    if remote_research_running:
        remote_strategy_status = "running"
    elif not state.proposals and remote_strategy_calls == 0:
        remote_strategy_status = "not_started"
    elif remote_strategy_calls == 0:
        remote_strategy_status = (
            "unknown"
            if source_counts.get("legacy_unknown", 0)
            else "not_called"
        )
    elif fallback_count and remote_strategy_successes:
        remote_strategy_status = "partial_host_fallback"
    elif fallback_count:
        remote_strategy_status = "host_fallback"
    elif remote_strategy_successes == remote_strategy_calls:
        remote_strategy_status = "completed"
    else:
        remote_strategy_status = "incomplete"

    single_pass_methods = {"bias_fit", "closed_form_ridge", "toy_score"}
    iterative_epoch_training = (
        any(iterative_training_flags)
        if iterative_training_flags
        else bool(fit_methods)
        and not set(fit_methods).issubset(single_pass_methods)
    )
    return {
        "execution_mode": modes[0] if len(modes) == 1 else modes or "pending",
        "fit_method": fit_methods[0] if len(fit_methods) == 1 else fit_methods or None,
        "model_ids": model_ids,
        "candidate_artifacts_count": len(artifacts),
        "candidate_evaluations_count": len(evaluations),
        "partition_scan_policy": (
            "full_training_fit_rotating_bounded_training_feedback_per_generation"
            if state.task_manifest.metadata.get("samples_per_update") is not None
            else "full_frozen_partition_per_candidate"
        ),
        "samples_per_update": state.task_manifest.metadata.get(
            "samples_per_update"
        ),
        "sample_concurrency": state.task_manifest.metadata.get(
            "sample_concurrency"
        ),
        "training_partition_rows": training_partition_rows,
        "training_eligible_examples": training_eligible_examples,
        "training_used_examples": training_used_examples,
        "training_skipped_examples": training_skipped_examples,
        "evaluation_partition_rows": evaluation_partition_rows,
        "evaluation_eligible_examples": evaluation_eligible_examples,
        "evaluation_used_examples": evaluation_used_examples,
        "evaluation_skipped_examples": evaluation_skipped_examples,
        "evaluation_available_examples": evaluation_available_examples,
        "evaluation_selected_examples": evaluation_selected_examples,
        "evaluation_deferred_examples": evaluation_deferred_examples,
        "live_evaluation_completed_examples": live_evaluation_completed_examples,
        "live_evaluation_total_examples": live_evaluation_total_examples,
        "live_evaluation_succeeded_examples": (
            live_evaluation_succeeded_examples
            if live_evaluation_outcome_counts_known
            else None
        ),
        "live_evaluation_failed_examples": (
            live_evaluation_failed_examples
            if live_evaluation_outcome_counts_known
            else None
        ),
        "live_evaluation_candidate_count": live_evaluation_candidate_count,
        "partial_evaluation_active_candidate_count": (
            partial_evaluation_active_candidate_count
        ),
        "partial_evaluation_retained_candidate_count": (
            partial_evaluation_retained_candidate_count
        ),
        "partial_evaluation_aborted_candidate_count": (
            partial_evaluation_aborted_candidate_count
        ),
        "partial_evaluation_sources": sorted(partial_evaluation_sources),
        "execution_evidence_status": execution_evidence_status,
        "candidate_work_items": (
            training_used_examples
            + evaluation_used_examples
            + live_evaluation_completed_examples
        ),
        "fit_passes_completed": fit_passes_completed,
        "fit_passes_per_candidate": max(per_artifact_fit_passes or [0]),
        "iterative_epoch_training": iterative_epoch_training,
        "legacy_workload_estimate_used": legacy_workload_estimate_used,
        "evolution_rounds_completed": int(state.run.generation),
        "evolution_rounds_configured": _max_generations(state.task_manifest),
        # Backward-compatible aliases.  These are cumulative physical training
        # rows and evaluation work items, respectively; new clients should use
        # the explicit fields above.
        "training_rows": training_partition_rows,
        "evaluation_rows": evaluation_used_examples,
        "epochs_completed": max(
            per_artifact_fit_passes or [0]
        ),
        "fallback_used": fallback_count > 0,
        "fallback_count": fallback_count,
        "fallback_reasons": fallback_reasons[:8],
        "proposal_sources": source_counts,
        "remote_research_calls": remote_research_calls,
        "remote_research_successes": remote_research_successes,
        "remote_strategy_calls": remote_strategy_calls,
        "remote_strategy_successes": remote_strategy_successes,
        "remote_strategy_status": remote_strategy_status,
    }


def _trace_scalar(value: Any, *, limit: int = 120) -> str | int | float | None:
    """Keep timestamps and labels primitive and bounded for the UI."""

    numeric = _finite_number(value)
    if numeric is not None:
        return numeric
    if isinstance(value, str):
        return value[:limit]
    return None


def _public_evaluation_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Hide the durable all-sample trace from routine browser responses."""

    result = dict(metrics)
    result.pop("sample_execution_records", None)
    result.pop("sample_execution_trace_archive", None)
    result.pop("promotion_block_evidence", None)
    return result


def _public_inference_trace(
    state: Any,
    candidate: Any,
    proposal: Any,
    evaluation: Any | None,
    artifact: Any | None,
) -> dict[str, Any]:
    """Project a bounded, non-chain-of-thought sample inference trace.

    Evaluators currently persist a ``prediction_preview`` in their metrics.
    This helper gives the browser one stable shape while retaining backwards
    compatibility with evaluators that have not produced a preview yet.
    """

    metrics = dict(evaluation.metrics) if evaluation is not None else {}
    raw_rows = metrics.get("prediction_preview")
    if not isinstance(raw_rows, list):
        raw_rows = metrics.get("inference_trace")
    if not isinstance(raw_rows, list):
        raw_rows = []

    sample_execution = metrics.get("sample_execution")
    if not isinstance(sample_execution, Mapping):
        sample_execution = {}
    raw_records = metrics.get("sample_execution_records")
    records_by_id = {
        str(item.get("sample_id")): item
        for item in raw_records
        if isinstance(item, Mapping) and item.get("sample_id") is not None
    } if isinstance(raw_records, list) else {}
    raw_actions = sample_execution.get("action_catalog")
    actions_by_digest = {
        str(item.get("action_digest")): item
        for item in raw_actions
        if isinstance(item, Mapping) and item.get("action_digest") is not None
    } if isinstance(raw_actions, list) else {}

    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows[:_INFERENCE_TRACE_LIMIT]):
        if not isinstance(raw, Mapping):
            continue
        observed = _finite_number(raw.get("observed"))
        predicted = _finite_number(raw.get("predicted"))
        baseline = _finite_number(raw.get("baseline"))
        row: dict[str, Any] = {
            "sample_index": index + 1,
            "timestamp": _trace_scalar(raw.get("timestamp")),
            "origin_timestamp": _trace_scalar(raw.get("origin_timestamp")),
            "target_timestamp": _trace_scalar(
                raw.get("target_timestamp", raw.get("timestamp"))
            ),
            "target": _trace_scalar(raw.get("target")),
            "unit": _trace_scalar(raw.get("unit")),
            "horizon_hours": _finite_number(raw.get("horizon_hours")),
            "observed": observed,
            "predicted": predicted,
            "baseline": baseline,
        }
        sample_id = _trace_scalar(raw.get("sample_id"))
        if sample_id is not None:
            row["sample_id"] = sample_id
            record = records_by_id.get(str(sample_id))
            evidence = record if isinstance(record, Mapping) else raw
            if isinstance(evidence, Mapping):
                row["status"] = _trace_scalar(
                    evidence.get("status", evidence.get("sample_execution_status"))
                )
                row["attempts"] = _finite_number(
                    evidence.get("attempts", evidence.get("sample_execution_attempts"))
                )
                row["retry_count"] = _finite_number(
                    evidence.get(
                        "retry_count", evidence.get("sample_execution_retry_count")
                    )
                )
                action_digest = _trace_scalar(evidence.get("action_digest"))
                row["action_digest"] = action_digest
                failure = evidence.get(
                    "failure", evidence.get("sample_execution_failure")
                )
                if failure is not None:
                    row["failure"] = _safe_plan_value(failure)
                    row["failure_action"] = _trace_scalar(
                        evidence.get("failure_action")
                    )
                action = actions_by_digest.get(str(action_digest))
                if isinstance(action, Mapping):
                    row["algorithm"] = _safe_plan_value(action.get("algorithm"))
                    row["agent_decisions"] = _safe_plan_value(
                        action.get("agent_decisions")
                    )
                    row["tool_calls"] = _safe_plan_value(action.get("tool_calls"))
        if observed is not None and predicted is not None:
            row["error"] = predicted - observed
        if observed is not None and baseline is not None:
            row["baseline_error"] = baseline - observed
        if observed is not None and predicted is not None and baseline is not None:
            row["reward"] = abs(baseline - observed) - abs(predicted - observed)
        rows.append(row)

    failure_preview = sample_execution.get("failure_preview")
    if isinstance(failure_preview, list):
        shown_sample_ids = {
            str(row.get("sample_id"))
            for row in rows
            if row.get("sample_id") is not None
        }
        for raw in failure_preview[: max(0, _INFERENCE_TRACE_LIMIT - len(rows))]:
            if not isinstance(raw, Mapping):
                continue
            if str(raw.get("sample_id")) in shown_sample_ids:
                continue
            rows.append(
                {
                    "sample_index": len(rows) + 1,
                    "sample_id": _trace_scalar(raw.get("sample_id")),
                    "status": "failed",
                    "attempts": _finite_number(raw.get("attempts")),
                    "retry_count": _finite_number(raw.get("retry_count")),
                    "target": _trace_scalar(raw.get("target")),
                    "horizon_hours": _finite_number(raw.get("horizon_hours")),
                    "predicted": None,
                    "failure_action": _trace_scalar(raw.get("failure_action")),
                    "failure": _safe_plan_value(raw.get("failure")),
                }
            )

    candidate_status = candidate.status.value
    if evaluation is not None:
        trace_status = "completed"
    elif candidate_status == "failed":
        trace_status = "failed"
    elif candidate_status == "duplicate":
        trace_status = "skipped"
    else:
        trace_status = "pending"
    total_samples = _finite_number(sample_execution.get("eligible_examples"))
    if total_samples is None:
        total_samples = _finite_number(metrics.get("n"))
    if total_samples is None:
        total_samples = len(rows)
    model_id = artifact.model_id if artifact is not None else None
    evaluator_id = state.task_manifest.metadata.get("evaluator_id")
    if evaluator_id is None:
        evaluator_id = (
            TOY_EVALUATOR_ID
            if state.task_manifest.visible_datasets
            and state.task_manifest.visible_datasets[0] == TOY_DATASET_ID
            else GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID
        )
    return {
        "schema_version": "ecologyrsi-dsh.inference-trace/1",
        "status": trace_status,
        "model_id": model_id,
        "evaluator_id": str(evaluator_id) if evaluator_id is not None else None,
        "partition": evaluation.partition if evaluation is not None else None,
        "reward_definition": metrics.get(
            "reward_definition", SAMPLE_REWARD_DEFINITION_V1
        ),
        "positive_is_better": True,
        "sample_count": int(total_samples) if float(total_samples).is_integer() else total_samples,
        "shown_count": len(rows),
        "truncated": bool(total_samples > len(rows)),
        # These are host-controlled operational steps, not model private
        # reasoning.  They make the execution auditable without exposing a
        # prompt or hidden chain of thought.
        "method_steps": [
            "冻结训练拟合分区",
            "应用候选参数与已登记预测器",
            "在训练反馈分区逐样本生成预测",
            "与观测值和冻结评分基线计算误差",
        ],
        "parameter_keys": sorted(str(key) for key in proposal.changes),
        "sample_execution": _safe_plan_value(sample_execution),
        "rows": rows,
    }


def _evaluation_batch_progress_projection(
    state: Any,
    candidate_id: str | None,
) -> dict[str, Any] | None:
    """Count host-finalized rows in the active private result revision."""

    if not candidate_id:
        return None
    start = next(
        (
            event
            for event in reversed(state.events)
            if event.kind == "EvaluationSampleResultsStarted"
            and event.payload.get("candidate_id") == candidate_id
        ),
        None,
    )
    if start is None:
        return None
    revision = start.payload.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        return None
    events = sorted(
        (
            event
            for event in state.events
            if event.seq > start.seq
            and event.kind == "EvaluationSampleResultBatchRecorded"
            and event.payload.get("candidate_id") == candidate_id
            and event.payload.get("revision") == revision
            and event.payload.get("run_id") == state.run.run_id
        ),
        key=lambda item: item.seq,
    )
    if not events:
        return None
    indices = [item.payload.get("batch_index") for item in events]
    if indices != list(range(1, len(events) + 1)):
        return None
    counts = [item.payload.get("record_count") for item in events]
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        for value in counts
    ):
        return None
    completed = sum(int(value) for value in counts)
    if completed > MAX_SAMPLE_RESULTS_RECORDS:
        return None
    checkpoint = start.payload.get("checkpoint")
    expected = checkpoint.get("sample_count") if isinstance(checkpoint, Mapping) else None
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or not 1 <= expected <= MAX_SAMPLE_RESULTS_RECORDS
    ):
        expected = completed
    if completed > expected:
        return None
    latest = events[-1]
    return {
        "revision": revision,
        "completed_samples": completed,
        "total_samples": expected,
        "batch_index": len(events),
        "batch_size": int(counts[-1]),
        "updated_at": latest.created_at,
        "event_seq": latest.seq,
        "evidence_source": "durable_sample_result_batches",
    }


def _evaluation_progress_projection(
    state: Any,
    candidate_id: str | None,
) -> dict[str, Any] | None:
    """Return the latest revision's durable, label-free planner heartbeat."""

    if not candidate_id:
        return None
    events = [
        event
        for event in state.events
        if event.kind == "EvaluationProgressRecorded"
        and event.payload.get("candidate_id") == candidate_id
        and event.payload.get("role") == "planner"
    ]
    latest_revision_start = next(
        (
            event
            for event in reversed(state.events)
            if event.kind == "EvaluationSampleResultsStarted"
            and event.payload.get("candidate_id") == candidate_id
        ),
        None,
    )
    active_revision = (
        latest_revision_start.payload.get("revision")
        if latest_revision_start is not None
        else None
    )
    if isinstance(active_revision, str) and active_revision.strip():
        # A sidecar recovery historically started a fresh sample-result
        # revision and reset its counters.  Comparing that revision with an
        # older, larger completed_samples value made a live run appear frozen
        # until it caught up.  Revision starts are durable ordering barriers,
        # so progress before the latest one must never drive the active view.
        v3_events = [
            event
            for event in events
            if event.payload.get("schema_version")
            == "ecologyrsi-dsh.evaluation-progress/3"
            and event.payload.get("revision") == active_revision
        ]
        # A v2 writer can resume an older checkpoint. Keep that historical
        # path readable until the writer emits its first revision-aware event.
        events = v3_events or [
            event for event in events if event.seq > latest_revision_start.seq
        ]
    elif latest_revision_start is not None:
        events = [event for event in events if event.seq > latest_revision_start.seq]
    if not events:
        return None
    is_v3_revision = bool(
        isinstance(active_revision, str)
        and active_revision.strip()
        and all(
            item.payload.get("schema_version")
            == "ecologyrsi-dsh.evaluation-progress/3"
            and item.payload.get("revision") == active_revision
            for item in events
        )
    )
    if is_v3_revision:
        # On sidecar restart a durable checkpoint can temporarily contain
        # fewer rows than a prior in-memory attempt.  The v3 progress identity
        # is monotonic within a revision, so it identifies the live heartbeat.
        event = max(
            events,
            key=lambda item: (int(item.payload.get("progress_id") or 0), item.seq),
        )
    else:
        event = max(
            events,
            key=lambda item: (
                int(item.payload.get("completed_samples") or 0),
                item.seq,
            ),
        )
    payload = event.payload
    completed = int(payload.get("completed_samples") or 0)
    total = max(1, int(payload.get("total_samples") or 0))
    samples_per_minute, gateway_calls_per_minute = _evaluation_progress_rates(
        events, event
    )
    estimated_remaining_seconds = (
        round(60.0 * max(0, total - completed) / samples_per_minute)
        if samples_per_minute is not None and samples_per_minute > 0
        else None
    )
    metadata = state.task_manifest.metadata
    configured_concurrency = metadata.get("sample_concurrency", 4)
    if (
        isinstance(configured_concurrency, bool)
        or not isinstance(configured_concurrency, int)
        or not 1 <= configured_concurrency <= 8
    ):
        configured_concurrency = None
    return {
        "schema_version": payload.get("schema_version"),
        "revision": payload.get("revision"),
        "progress_id": payload.get("progress_id"),
        "progress_kind": payload.get("progress_kind", "completed_batch"),
        "role": "planner",
        "model_id": payload.get("model_id"),
        "batch_index": payload.get("batch_index"),
        "batch_count": payload.get("batch_count"),
        "batch_size": payload.get("batch_size"),
        "completed_samples": completed,
        "total_samples": total,
        "succeeded_samples": payload.get("succeeded_samples"),
        "failed_samples": payload.get("failed_samples"),
        "gateway_request_count": payload.get(
            "gateway_request_count", payload.get("batch_index")
        ),
        "adaptive_split_trigger_count": payload.get(
            "adaptive_split_trigger_count", 0
        ),
        "adaptive_split_count": payload.get("adaptive_split_count", 0),
        "adaptive_split_max_depth": payload.get("adaptive_split_max_depth", 0),
        "adaptive_split_recovered_samples": payload.get(
            "adaptive_split_recovered_samples", 0
        ),
        "adaptive_split_failed_samples": payload.get(
            "adaptive_split_failed_samples", 0
        ),
        "causal_wave_sample_count": payload.get("batch_size"),
        "in_flight_batches": payload.get("in_flight_batches"),
        "queued_batches": payload.get("queued_batches"),
        "configured_concurrency": configured_concurrency,
        "samples_per_minute": samples_per_minute,
        "gateway_calls_per_minute": gateway_calls_per_minute,
        "estimated_remaining_seconds": estimated_remaining_seconds,
        "progress_percent": round(100.0 * min(completed, total) / total, 1),
        "updated_at": event.created_at,
        "event_seq": event.seq,
    }


def _superseded_sample_revision_projection(
    state: Any,
    candidate_id: str | None,
) -> dict[str, Any] | None:
    """Expose aggregate evidence for a checkpoint revision fenced from reuse.

    A fresh checkpoint deliberately starts its active counters at zero when an
    earlier revision cannot prove that its cohort and execution context match.
    Keep that active progress isolated, but retain a compact explanation of
    the work visible before the fence so an operator does not mistake the
    reset for lost evidence.
    """

    if not candidate_id:
        return None
    starts = [
        event
        for event in state.events
        if event.kind == "EvaluationSampleResultsStarted"
        and event.payload.get("candidate_id") == candidate_id
    ]
    if not starts:
        return None
    current_start = starts[-1]
    superseded_revision = current_start.payload.get("supersedes_revision")
    resume_disposition = current_start.payload.get("resume_disposition")
    if not (
        isinstance(superseded_revision, str)
        and superseded_revision.strip()
        and isinstance(resume_disposition, str)
        and resume_disposition.strip()
    ):
        return None
    previous_start = next(
        (
            event
            for event in reversed(starts[:-1])
            if event.payload.get("revision") == superseded_revision
        ),
        None,
    )
    if previous_start is None:
        return None

    progress_events = [
        event
        for event in state.events
        if previous_start.seq < event.seq < current_start.seq
        and event.kind == "EvaluationProgressRecorded"
        and event.payload.get("candidate_id") == candidate_id
        and event.payload.get("role") == "planner"
        # Revision-aware heartbeats must match the superseded revision. Older
        # v1/v2 writers have no revision field, so their bounded start window
        # is the only safe association available.
        and (
            event.payload.get("schema_version")
            != "ecologyrsi-dsh.evaluation-progress/3"
            or event.payload.get("revision") == superseded_revision
        )
    ]
    if not progress_events:
        return None
    latest = max(
        progress_events,
        key=lambda event: (int(event.payload.get("completed_samples") or 0), event.seq),
    )
    payload = latest.payload
    return {
        "revision": superseded_revision,
        "resume_disposition": resume_disposition,
        "completed_samples": int(payload.get("completed_samples") or 0),
        "succeeded_samples": int(payload.get("succeeded_samples") or 0),
        "failed_samples": int(payload.get("failed_samples") or 0),
        "total_samples": int(payload.get("total_samples") or 0),
        "superseded_at": current_start.created_at,
    }


def _evaluation_progress_rates(
    events: list[Any], latest: Any
) -> tuple[float | None, float | None]:
    """Estimate recent durable throughput from at most ten heartbeat intervals."""

    latest_completed = int(latest.payload.get("completed_samples") or 0)
    latest_calls = int(
        latest.payload.get("gateway_request_count")
        or latest.payload.get("batch_index")
        or 0
    )
    latest_timestamp = _event_timestamp(latest.created_at)
    if latest_timestamp is None:
        return None, None
    previous = sorted(
        (
            item
            for item in events
            if item.seq < latest.seq
            and int(item.payload.get("completed_samples") or 0) < latest_completed
        ),
        key=lambda item: item.seq,
    )
    if not previous:
        return None, None
    baseline = previous[max(0, len(previous) - 10)]
    baseline_timestamp = _event_timestamp(baseline.created_at)
    if baseline_timestamp is None:
        return None, None
    elapsed_minutes = (latest_timestamp - baseline_timestamp) / 60.0
    if elapsed_minutes <= 0:
        return None, None
    sample_delta = latest_completed - int(
        baseline.payload.get("completed_samples") or 0
    )
    baseline_calls = int(
        baseline.payload.get("gateway_request_count")
        or baseline.payload.get("batch_index")
        or 0
    )
    call_delta = max(0, latest_calls - baseline_calls)
    return (
        round(sample_delta / elapsed_minutes, 2),
        round(call_delta / elapsed_minutes, 2),
    )


def _event_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _gateway_retry_projection(state: Any) -> dict[str, Any] | None:
    """Expose the latest provider cooldown while a run remains live."""

    event = next(
        (
            item
            for item in reversed(state.events)
            if item.kind == "GatewayRetryScheduled"
            and int(item.payload.get("generation", -1)) == int(state.run.generation)
        ),
        None,
    )
    if event is None or state.run.status.value != "running":
        return None
    newer_stage = next(
        (
            item
            for item in reversed(state.events)
            if item.kind == "EvolutionStageRecorded"
            and int(item.payload.get("generation", -1)) == int(state.run.generation)
            and item.seq > event.seq
        ),
        None,
    )
    if newer_stage is not None:
        return None
    payload = event.payload
    return {
        "waiting": True,
        "generation": payload.get("generation"),
        "retry_at": payload.get("retry_at"),
        "delay_seconds": payload.get("delay_seconds"),
        "attempt": payload.get("attempt"),
        "error_code": payload.get("error_code"),
        "reason": payload.get("reason"),
        "event_seq": event.seq,
        "updated_at": event.created_at,
    }


def _algorithm_execution_projection(state: Any, candidate: Any) -> dict[str, Any]:
    attempts = state.algorithm_attempts_for(candidate.candidate_id)
    compiled = next(
        (
            item
            for item in reversed(attempts)
            if item.phase == "compile" and item.status == "passed"
        ),
        None,
    )
    failed = next((item for item in reversed(attempts) if item.status == "failed"), None)
    debug_passed = any(
        item.phase == "debug" and item.status == "passed" for item in attempts
    )
    status = (
        "debug_passed"
        if debug_passed
        else f"{failed.phase}_failed"
        if failed is not None
        else "compiled"
        if compiled is not None
        else "pending"
    )
    return {
        "status": status,
        "training_authorized": debug_passed,
        "algorithm_spec": (
            _safe_plan_value(dict(compiled.algorithm_spec))
            if compiled is not None and compiled.algorithm_spec is not None
            else None
        ),
        "algorithm_spec_digest": (
            compiled.algorithm_spec_digest if compiled is not None else None
        ),
        "attempts": [
            {
                "phase": item.phase,
                "attempt": item.attempt,
                "status": item.status,
                "algorithm_spec_digest": item.algorithm_spec_digest,
                "evidence": _safe_plan_value(dict(item.evidence)),
                "failure_code": item.failure_code,
                "public_error": public_error_summary(item.public_error),
                "created_at": item.created_at,
            }
            for item in attempts[:16]
        ],
        "security_boundary": {
            "external_code_execution": False,
            "registered_adapters_only": True,
        },
    }


def _candidate_execution_projection(
    state: Any,
    candidate: Any,
    proposal: Any,
    evaluation: Any | None,
    promotion: Any | None,
) -> dict[str, Any]:
    stages = _effective_stage_statuses(
        state,
        _candidate_stage_statuses(
            state, candidate, proposal, evaluation, promotion
        ),
    )
    terminal = sum(value in _TERMINAL_STAGE_STATUSES for value in stages.values())
    run_status = state.run.status.value
    if run_status in _TERMINAL_RUN_STATUSES:
        active_stage = None
    else:
        active_stage = next(
            (
                name
                for name in _EVOLUTION_STAGE_ORDER
                if stages[name] in {"running", "paused"}
            ),
            next(
                (name for name in _EVOLUTION_STAGE_ORDER if stages[name] == "failed"),
                next(
                    (
                        name
                        for name in _EVOLUTION_STAGE_ORDER
                        if stages[name] == "pending"
                    ),
                    None,
                ),
            ),
        )
    stage_progress = _evaluation_progress_projection(
        state,
        candidate.candidate_id,
    )
    superseded_sample_revision = _superseded_sample_revision_projection(
        state,
        candidate.candidate_id,
    )
    intra_stage_fraction = 0.0
    if (
        stages.get("evaluation") in {"running", "paused"}
        and stage_progress is not None
    ):
        intra_stage_fraction = min(
            1.0,
            float(stage_progress["completed_samples"])
            / max(1.0, float(stage_progress["total_samples"])),
        )
    prediction_model_id = state.task_manifest.metadata.get("prediction_model_id")
    proposal_metadata = getattr(proposal, "metadata", None)
    if isinstance(proposal_metadata, Mapping):
        adoption = proposal_metadata.get("prediction_model_adoption")
        if isinstance(adoption, Mapping) and adoption.get("status") == "adopted":
            adopted_id = adoption.get("adopted_id")
            if isinstance(adopted_id, str) and adopted_id.strip():
                prediction_model_id = adopted_id.strip()
    compiled = next(
        (
            item
            for item in reversed(state.algorithm_attempts_for(candidate.candidate_id))
            if item.phase == "compile"
            and item.status == "passed"
            and isinstance(item.algorithm_spec, Mapping)
        ),
        None,
    )
    if compiled is not None:
        adapter_id = compiled.algorithm_spec.get("adapter_id")
        if isinstance(adapter_id, str) and adapter_id.strip():
            prediction_model_id = adapter_id.strip()
    if evaluation is not None and isinstance(evaluation.metrics, Mapping):
        evaluated_model_id = evaluation.metrics.get("prediction_model_id")
        if isinstance(evaluated_model_id, str) and evaluated_model_id.strip():
            prediction_model_id = evaluated_model_id.strip()

    return {
        "status": _effective_candidate_status(
            state,
            candidate,
            (
                "completed"
                if candidate.status.value in {"promoted", "rejected", "duplicate"}
                else candidate.status.value
            ),
        ),
        "current_stage": active_stage,
        "stages": stages,
        "completed_stages": terminal,
        "total_stages": len(_EVOLUTION_STAGE_ORDER),
        "progress_percent": round(
            100.0
            * (terminal + intra_stage_fraction)
            / len(_EVOLUTION_STAGE_ORDER),
            1,
        ),
        "stage_progress": stage_progress,
        "superseded_sample_revision": superseded_sample_revision,
        "prediction_model_id": prediction_model_id,
        "strategy_id": state.task_manifest.metadata.get("strategy_id"),
        "evaluator_id": state.task_manifest.metadata.get("evaluator_id"),
        "evidence": "stage_events" if any(event.kind == "EvolutionStageRecorded" and event.payload.get("candidate_id") == candidate.candidate_id for event in state.events) else "state_projection",
    }


def _run_execution_progress(state: Any) -> dict[str, Any]:
    """Summarize durable execution evidence for a compact progress bar."""

    task = state.task_manifest
    total_generations = max(1, _max_generations(task))
    candidates_per_generation = max(1, task.candidates_per_generation)
    target_candidates = min(
        max(1, task.max_candidates), total_generations * candidates_per_generation
    )
    total_steps = target_candidates * len(_EVOLUTION_STAGE_ORDER)
    completed_steps = 0
    terminal_candidates = 0
    current_candidates = [
        item for item in state.candidates if item.generation == state.run.generation
    ]
    current_stage_rows: list[tuple[Any, dict[str, str]]] = []
    for candidate in state.candidates:
        proposal = state.proposal(candidate.proposal_id)
        evaluation = state.evaluation_for(candidate.candidate_id)
        promotion = state.promotion_for(candidate.candidate_id)
        execution = _candidate_execution_projection(
            state, candidate, proposal, evaluation, promotion
        )
        completed_steps += int(execution["completed_stages"])
        if candidate.status.value in {
            "promoted",
            "rejected",
            "failed",
            "duplicate",
        }:
            terminal_candidates += 1
        if candidate.generation == state.run.generation:
            current_stage_rows.append((candidate, execution["stages"]))

    status = state.run.status.value
    completed_steps = min(total_steps, completed_steps)
    progress_percent = round(
        min(100.0, max(0.0, 100.0 * completed_steps / max(1, total_steps))),
        1,
    )

    active_candidate_id: str | None = None
    current_stage: str | None = None
    # Prefer the most recent running stage, which is the strongest evidence
    # that a model call or evaluator is active at the time of polling.
    for candidate, stages in reversed(current_stage_rows):
        current_stage = next(
            (name for name in _EVOLUTION_STAGE_ORDER if stages.get(name) == "running"),
            None,
        )
        if current_stage is not None:
            active_candidate_id = candidate.candidate_id
            break
    retry_wait = _gateway_retry_projection(state)
    if retry_wait is not None:
        # A cooldown is more actionable than a generic queued boundary.  Keep
        # the last stage/candidate context so the UI can explain what is being
        # retried without pretending that a generation completed.
        current_stage = "gateway_retry"

    if current_stage is None:
        for candidate, stages in current_stage_rows:
            current_stage = next(
                (name for name in _EVOLUTION_STAGE_ORDER if stages.get(name) == "pending"),
                None,
            )
            if current_stage is not None:
                active_candidate_id = candidate.candidate_id
                break

    stage_event = next(
        (
            event
            for event in reversed(state.events)
            if event.kind == "EvolutionStageRecorded"
            and event.payload.get("generation") == state.run.generation
        ),
        None,
    )
    # A retry heartbeat is written after the failed gateway stage and is the
    # latest durable explanation of why execution is waiting.  Do not let an
    # older ``started``/``failed`` stage overwrite ``gateway_retry`` here; the
    # browser needs a stable phase while the provider cooldown is active.
    if stage_event is not None and retry_wait is None:
        stage_status = str(stage_event.payload.get("status") or "").lower()
        if stage_status in {"started", "running", "failed"}:
            current_stage = str(stage_event.payload.get("stage") or current_stage)
            active_candidate_id = stage_event.payload.get("candidate_id") or active_candidate_id

    if status in _TERMINAL_RUN_STATUSES:
        # A terminal run can retain pending or stale started stage evidence,
        # but it must not be projected as if execution were still active.
        current_stage = None
        active_candidate_id = None

    stage_progress = _evaluation_progress_projection(state, active_candidate_id)
    superseded_sample_revision = _superseded_sample_revision_projection(
        state, active_candidate_id
    )
    if current_stage == "evaluation" and stage_progress is not None:
        fraction = min(
            1.0,
            float(stage_progress["completed_samples"])
            / max(1.0, float(stage_progress["total_samples"])),
        )
        progress_percent = round(
            min(
                100.0,
                max(
                    0.0,
                    100.0 * (completed_steps + fraction) / max(1, total_steps),
                ),
            ),
            1,
        )

    if status == "paused":
        # Preserve ``current_stage`` as historical context for the pause while
        # making the effective phase unambiguously non-running.
        phase = "paused"
    elif current_stage is None:
        if status == "completed":
            phase = "completed"
        elif status in {"failed", "cancelled"}:
            phase = status
        elif status == "paused":
            phase = "paused"
        elif state.run.generation >= total_generations:
            phase = "finalizing"
        elif state.task_manifest.metadata.get("auto_progress") is True:
            # A continuous run is expected to cross generation boundaries on
            # its own.  Distinguish this queued boundary from the legacy
            # manual ``waiting`` state so the UI never asks the user to click
            # an advance button for an autonomous run.
            phase = "queued"
        else:
            phase = "waiting"
    else:
        phase = current_stage

    return {
        "schema_version": "ecologyrsi-dsh.execution-progress/1",
        "status": status,
        "phase": phase,
        "progress_percent": progress_percent,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "completed_candidates": terminal_candidates,
        "total_candidates": target_candidates,
        "completed_generations": min(state.run.generation, total_generations),
        "total_generations": total_generations,
        "current_generation": min(state.run.generation + 1, total_generations),
        "current_candidate_id": active_candidate_id,
        "current_stage": current_stage,
        "stage_progress": stage_progress,
        "superseded_sample_revision": superseded_sample_revision,
        "retry_wait": retry_wait,
        "current_generation_candidate_count": len(current_candidates),
        "candidates_per_generation": candidates_per_generation,
        "auto_progress": state.task_manifest.metadata.get("auto_progress") is True,
        "auto_progress_policy": state.task_manifest.metadata.get(
            "auto_progress_policy"
        ),
        "last_event_seq": state.events[-1].seq if state.events else 0,
        "last_event_at": state.events[-1].created_at if state.events else state.run.created_at,
        "evidence": (
            "append_only_stage_and_progress_events"
            if stage_progress is not None
            else "append_only_stage_events"
        ),
    }


def _token_budget_scope(task: Any, metadata: Mapping[str, Any]) -> str | None:
    """Describe the hard budget without overstating run-wide accounting."""

    if (
        metadata.get("sample_token_budget_policy") == _SAMPLE_TOKEN_BUDGET_POLICY
        and _budget_value(task, "token_limit", 0) > 0
    ):
        # Infer the scope for already-created manifests that froze the current
        # hard-budget policy before the explicit scope marker was introduced.
        return _SAMPLE_TOKEN_BUDGET_SCOPE
    return None


def _model_usage_summary(state: Any) -> dict[str, Any]:
    """Aggregate reported usage and conservative hard-budget accounting."""

    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "request_count": 0,
    }
    by_role: dict[str, dict[str, int]] = {}
    event_count = 0
    reported_call_count = 0
    reported_total_tokens = 0
    has_v2_receipt = False
    physical_call_count = 0
    logical_call_digests: set[str] = set()
    outcome_counts: dict[str, int] = {}
    scope_candidate_ids: set[str] = set()
    scope_revisions: set[str] = set()
    scope_complete = True
    for event in state.events:
        if event.kind != "ModelUsageRecorded":
            continue
        payload = event.payload
        role = payload.get("role")
        if not isinstance(role, str):
            # Replay validates all current events.  Retaining this guard keeps
            # standalone diagnostic projections bounded for legacy fixtures.
            continue
        event_count += 1
        candidate_id = payload.get("candidate_id")
        revision = payload.get("revision")
        if isinstance(candidate_id, str) and candidate_id.strip():
            scope_candidate_ids.add(candidate_id)
        else:
            scope_complete = False
        if isinstance(revision, str) and revision.strip():
            scope_revisions.add(revision)
        else:
            scope_complete = False
        is_v2_receipt = (
            payload.get("schema_version") == "ecologyrsi-dsh.model-usage/2"
        )
        usage_reported = (
            payload.get("usage_reported") is True
            if is_v2_receipt
            else True
        )
        has_v2_receipt = has_v2_receipt or is_v2_receipt
        if is_v2_receipt:
            physical_call_count += 1
            logical_call_digest = payload.get("logical_call_digest")
            if isinstance(logical_call_digest, str) and logical_call_digest:
                logical_call_digests.add(logical_call_digest)
            outcome = payload.get("outcome")
            if isinstance(outcome, str) and outcome:
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if usage_reported:
            reported_call_count += 1
            reported_total = payload.get("total_tokens")
            if (
                isinstance(reported_total, int)
                and not isinstance(reported_total, bool)
                and reported_total >= 0
            ):
                reported_total_tokens += reported_total
        role_totals = by_role.setdefault(
            role,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "request_count": 0,
                "call_count": 0,
                "reported_call_count": 0,
                "missing_call_count": 0,
            },
        )
        role_totals["call_count"] += 1
        if usage_reported:
            role_totals["reported_call_count"] += 1
        else:
            role_totals["missing_call_count"] += 1
        request_count = payload.get(
            "http_attempts", payload.get("gateway_request_count", 0)
        )
        if isinstance(request_count, int) and not isinstance(request_count, bool):
            totals["request_count"] += request_count
            role_totals["request_count"] += request_count
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = payload.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[name] += value
                role_totals[name] += value
    if not has_v2_receipt:
        legacy_roles = {
            role: {
                "prompt_tokens": values["prompt_tokens"],
                "completion_tokens": values["completion_tokens"],
                "total_tokens": values["total_tokens"],
                "gateway_request_count": values["request_count"],
                "call_count": values["call_count"],
            }
            for role, values in sorted(by_role.items())
        }
        return {
            "schema_version": "ecologyrsi-dsh.model-usage-summary/1",
            "available": event_count > 0,
            "call_count": event_count,
            "prompt_tokens": totals["prompt_tokens"],
            "completion_tokens": totals["completion_tokens"],
            "total_tokens": totals["total_tokens"],
            "gateway_request_count": totals["request_count"],
            "by_role": legacy_roles,
        }
    missing_call_count = event_count - reported_call_count
    task_manifest = getattr(state, "task_manifest", None)
    missing_call_reservation = int(
        getattr(task_manifest, "token_reservation_per_wave", 0) or 0
    )
    # Keep the raw counters visible for diagnostics, but mirror hard-budget
    # admission here: reported totals charge reported calls, while an
    # unreported call charges its frozen reservation instead of any
    # placeholder estimate. This is budget accounting, not actual consumption.
    budget_accounted_tokens = (
        reported_total_tokens + missing_call_count * missing_call_reservation
    )
    summary = {
        "schema_version": "ecologyrsi-dsh.model-usage-summary/2",
        # Receipts are durable usage evidence even when a provider omits its
        # token counters. ``complete`` / ``missing_call_count`` distinguish a
        # complete raw total from conservative reservation-based accounting.
        "available": event_count > 0,
        "call_count": event_count,
        # These four fields deliberately cover v2 physical receipts only.
        # Legacy v1 events remain in call_count and token totals but cannot be
        # assigned a stable logical-call identity after the fact.
        "physical_call_count": physical_call_count,
        "logical_call_count": len(logical_call_digests),
        "replayed_call_count": max(
            0, physical_call_count - len(logical_call_digests)
        ),
        "outcome_counts": {
            outcome: outcome_counts[outcome] for outcome in sorted(outcome_counts)
        },
        "reported_call_count": reported_call_count,
        "missing_call_count": missing_call_count,
        "complete": missing_call_count == 0,
        "usage_coverage": (
            round(reported_call_count / event_count, 4) if event_count else 0.0
        ),
        "budget_accounted_tokens": budget_accounted_tokens,
        "missing_call_reservation": missing_call_reservation,
        **totals,
        # Compatibility alias for earlier public clients.
        "gateway_request_count": totals["request_count"],
        "by_role": {role: by_role[role] for role in sorted(by_role)},
    }
    # A linear live-stage estimate is defensible only when every counter in
    # this run-wide summary belongs to one known candidate revision. Omit the
    # scope otherwise so clients cannot accidentally combine unrelated work.
    if (
        scope_complete
        and len(scope_candidate_ids) == 1
        and len(scope_revisions) == 1
    ):
        summary["scope_candidate_id"] = next(iter(scope_candidate_ids))
        summary["scope_revision"] = next(iter(scope_revisions))
    return summary


def _candidate_projection(state: Any, candidate: Any) -> dict[str, Any]:
    proposal = state.proposal(candidate.proposal_id)
    evaluation = state.evaluation_for(candidate.candidate_id)
    artifact = state.artifact_for(candidate.candidate_id)
    promotion = state.promotion_for(candidate.candidate_id)
    status = {
        "spawned": "evaluating",
        "evaluated": "evaluated",
        "promoted": "retained",
        "rejected": "rejected",
        "failed": "failed",
        "duplicate": "duplicate",
    }.get(candidate.status.value, candidate.status.value)
    status = _effective_candidate_status(state, candidate, status)
    analysis = state.analysis_for(candidate.generation)
    ranking = next(
        (
            dict(item)
            for item in analysis.ranking
            if item.get("candidate_id") == candidate.candidate_id
        ),
        None,
    ) if analysis is not None else None
    result: dict[str, Any] = {
        "id": candidate.candidate_id,
        "candidate_id": candidate.candidate_id,
        "parent_id": proposal.parent_candidate_id,
        "proposal_id": proposal.proposal_id,
        "generation": candidate.generation + 1,
        "slot_index": candidate.slot_index,
        "status": status,
        "created_at": candidate.created_at,
        "title": proposal.title,
        "rationale": proposal.rationale,
        "changes": dict(proposal.changes),
        "proposal_source": (
            proposal.metadata.get("proposal_source")
            if isinstance(proposal.metadata, Mapping)
            else None
        ),
    }
    result["execution"] = _candidate_execution_projection(
        state, candidate, proposal, evaluation, promotion
    )
    result["algorithm_execution"] = _algorithm_execution_projection(state, candidate)
    result["inference_trace"] = _public_inference_trace(
        state, candidate, proposal, evaluation, artifact
    )
    try:
        genome = state.persisted_genome_for(candidate.candidate_id)
    except (KeyError, TypeError, ValueError):
        result["genome"] = {
            "available": False,
            "source": "historical_legacy_projection",
        }
    else:
        genome_value = genome.to_dict()
        scientific = genome_value["scientific_program"]
        lineage = genome_value["lineage"]
        identity = state.candidate_identity_binding(candidate.candidate_id) or {}
        result["genome"] = {
            "available": True,
            "source": "persisted_dsh_native_genome",
            "genome_id": genome.genome_id,
            "genome_digest": genome.genome_digest,
            "behavior_digest": genome.behavior_digest,
            "lineage": {
                name: lineage.get(name)
                for name in (
                    "origin_kind",
                    "parent_candidate_id",
                    "parent_genome_digest",
                    "mutation_operator_id",
                    "mutation_digest",
                    "generation",
                    "slot_index",
                )
            },
            "programs": {
                "predictor": scientific["predictor_ref"]["id"],
                "feature_policy": scientific["feature_policy_ref"]["id"],
                "fit_policy": scientific["fit_policy_ref"]["id"],
                "uncertainty_policy": scientific["uncertainty_policy_ref"]["id"],
                "parameter_names": sorted(scientific["parameter_overrides"]),
            },
            "compiled_identity": {
                name: identity.get(name)
                for name in (
                    "compiled_behavior_digest",
                    "phenotype_instance_digest",
                    "compiler_digest",
                    "workflow_ir_digest",
                    "tool_policy_digest",
                )
                if identity.get(name) is not None
            },
        }
    if getattr(proposal, "metadata", None):
        # The model plan is an advisory, JSON-only trace.  It is intentionally
        # projected separately from executable parameter changes.
        result["model_plan"] = _safe_plan_value(dict(proposal.metadata))
    if evaluation is not None:
        result.update(
            {
                "score": evaluation.score,
                "passed": evaluation.passed,
                "metrics": _public_evaluation_metrics(evaluation.metrics),
                "partition": evaluation.partition,
                "evaluator_digest": evaluation.evaluator_digest,
                "artifact_digest": evaluation.artifact_digest,
            }
        )
    if promotion is not None:
        public_promotion = promotion.to_dict()
        public_promotion.update(
            {
                "stage": "iterative_search",
                "formal_validation": False,
            }
        )
        result["promotion"] = public_promotion
    if ranking is not None:
        result.update(
            {
                "generation_rank": ranking.get("rank"),
                "eligible": ranking.get("eligible"),
                "classification": ranking.get("classification"),
                "selection_reason": ranking.get("selection_reason"),
                "worst_skill_score": ranking.get("worst_skill_score"),
                "parameter_distance": ranking.get("parameter_distance"),
            }
        )
    result["fitness"] = {
        "evidence_class": (
            ranking.get("evidence_class")
            if ranking is not None
            else "exploratory_adaptive_data"
        ),
        "label": "探索性自适应证据",
        "eligible": ranking.get("eligible") if ranking is not None else None,
        "classification": (
            ranking.get("classification") if ranking is not None else None
        ),
        "primary_score": evaluation.score if evaluation is not None else None,
        "worst_cell_skill": (
            ranking.get("worst_skill_score") if ranking is not None else None
        ),
        "formal_confirmation": False,
    }
    if candidate.status.value == "failed":
        failure_event = next(
            (
                event
                for event in reversed(state.events)
                if event.kind == "CandidateFailed"
                and event.payload.get("candidate_id") == candidate.candidate_id
            ),
            None,
        )
        failed_stage = next(
            (
                event.payload.get("stage")
                for event in reversed(state.events)
                if event.kind == "EvolutionStageRecorded"
                and event.payload.get("candidate_id") == candidate.candidate_id
                and event.payload.get("status") == "failed"
            ),
            None,
        )
        if failed_stage is None:
            failed_algorithm = next(
                (
                    item
                    for item in reversed(
                        state.algorithm_attempts_for(candidate.candidate_id)
                    )
                    if item.status == "failed"
                ),
                None,
            )
            if failed_algorithm is not None:
                failed_stage = f"algorithm_{failed_algorithm.phase}"
        result["failure_reason"] = public_error_summary(
            failure_event.payload.get("reason", "候选训练或评测失败")
            if failure_event is not None
            else "候选训练或评测失败"
        ) or "候选训练或评测失败"
        result["failed_stage"] = failed_stage
    return result


def _dsh_runtime_projection(state: Any) -> dict[str, Any]:
    bound = next(
        (event for event in state.events if event.kind == "DshRuntimeBound"),
        None,
    )
    first_call_verified = any(
        event.kind == "DshStructuredResultAccepted" for event in state.events
    )
    if bound is None:
        return {
            "execution_protocol": "legacy_read_only",
            "native": False,
            "capability_verified": False,
            "first_call_verified": False,
            "context_pressure": {"available": False, "source": "not_dsh_native"},
            "provider_usage": {"available": False, "source": "not_dsh_native"},
        }
    metrics_by_session: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for event in state.events:
        if event.kind != "DshStructuredResultAccepted":
            continue
        payload = event.payload
        metrics = payload.get("session_metrics") if isinstance(payload, Mapping) else None
        identity = payload.get("identity") if isinstance(payload, Mapping) else None
        session_id = (
            identity.get("session_id") if isinstance(identity, Mapping) else None
        )
        if (
            not isinstance(metrics, Mapping)
            or metrics.get("schema_version")
            != "ecologyrsi-dsh.dsh-session-metrics/1"
            or not isinstance(session_id, str)
            or metrics.get("session_id") != session_id
        ):
            continue
        seq = int(getattr(event, "seq", 0) or 0)
        prior = metrics_by_session.get(session_id)
        if prior is None or seq >= prior[0]:
            metrics_by_session[session_id] = (seq, metrics)

    pressure_rows = [
        metrics["context_pressure"]
        for _seq, metrics in metrics_by_session.values()
        if isinstance(metrics.get("context_pressure"), Mapping)
        and metrics["context_pressure"].get("available") is True
    ]
    usage_rows = [
        metrics["provider_usage"]["totals"]
        for _seq, metrics in metrics_by_session.values()
        if isinstance(metrics.get("provider_usage"), Mapping)
        and metrics["provider_usage"].get("available") is True
        and isinstance(metrics["provider_usage"].get("totals"), Mapping)
    ]
    context_pressure = {
        "available": False,
        "source": "dsh_token_meter",
        "measurement": "current_context_pressure",
    }
    if pressure_rows:
        context_pressure = {
            "available": True,
            "source": "dsh_token_meter",
            "measurement": "current_context_pressure",
            "session_count": len(pressure_rows),
            "maximum_total_tokens": max(
                int(item.get("total_tokens", 0)) for item in pressure_rows
            ),
            "maximum_surface_tokens": max(
                int(item.get("surface_tokens", 0)) for item in pressure_rows
            ),
        }
    provider_usage = {
        "available": False,
        "source": "dsh_session_projection_token_usage",
        "measurement": "cumulative_provider_reported_usage",
    }
    if usage_rows:
        names = (
            "uncached_input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "total_tokens",
        )
        totals = {
            name: sum(int(item.get(name, 0)) for item in usage_rows)
            for name in names
        }
        provider_usage = {
            "available": True,
            "source": "dsh_session_projection_token_usage",
            "measurement": "cumulative_provider_reported_usage",
            "session_count": len(usage_rows),
            **totals,
        }
    return {
        "execution_protocol": bound.payload["execution_protocol"],
        "native": True,
        "capability_verified": True,
        "capabilities_digest": bound.payload["capabilities_digest"],
        "preset_ids": list(bound.payload["preset_ids"]),
        "root_session_id": state.run.session_id,
        "first_call_verified": first_call_verified,
        # DSH reports the TokenMeter pressure and Session projection usage as
        # distinct Host-owned measurements. Python never infers one from the
        # other and never relabels legacy ModelUsageRecorded receipts as DSH.
        "context_pressure": context_pressure,
        "provider_usage": provider_usage,
    }


_INTERVENTION_RECEIPT_DETAIL_FIELDS = (
    "parameter",
    "direction",
    "step",
    "previous_value",
    "result_value",
    "parameters",
    "previous_values",
    "result_values",
    "operator",
    "bound",
    "target_candidate_id",
)


def _public_intervention_receipt(
    payload: Any,
    *,
    kind: str,
) -> dict[str, Any]:
    """Normalize new and legacy application receipts for browser projection."""

    source = payload if isinstance(payload, dict) else {}
    status = source.get("application_status")
    if status not in {"recorded", "applied", "enforced"}:
        status = (
            "enforced"
            if kind in {
                InterventionKind.PARAMETER_OVERRIDE.value,
                InterventionKind.PARENT_SELECTION.value,
            }
            else "recorded"
        )
        legacy_reason = (
            "旧版事件未保存执行明细；根据干预类型推断为宿主强制执行"
            if status == "enforced"
            else "旧版事件未保存执行收据；仅确认意见已被本轮提案消费"
        )
    else:
        legacy_reason = {
            "recorded": "意见已记录，但未执行",
            "applied": "意见已应用到本轮提案",
            "enforced": "意见已由宿主边界强制执行",
        }[status]
    reason = public_error_summary(source.get("reason") or legacy_reason) or legacy_reason
    receipt: dict[str, Any] = {
        "recorded": True,
        "applied": status in {"applied", "enforced"},
        "enforced": status == "enforced",
        "application_status": status,
        "reason": reason,
    }
    for name in _INTERVENTION_RECEIPT_DETAIL_FIELDS:
        if name in source:
            receipt[name] = source[name]
    return receipt


def _intervention_projection(state: Any, item: HumanIntervention) -> dict[str, Any]:
    effective_generation = state.run.generation + 1
    receipt: dict[str, Any]
    if item.applied_proposal_id is None:
        receipt = {
            "recorded": True,
            "applied": False,
            "enforced": False,
            "application_status": "recorded",
            "reason": "意见已记录，等待下一轮提案处理",
        }
        status_text = "等待下一轮"
    else:
        proposal = state.proposal(item.applied_proposal_id)
        effective_generation = proposal.generation + 1
        application_event = next(
            (
                event
                for event in reversed(state.events)
                if event.kind == "HumanInterventionApplied"
                and event.payload.get("intervention_id") == item.intervention_id
                and event.payload.get("proposal_id") == item.applied_proposal_id
            ),
            None,
        )
        receipt = _public_intervention_receipt(
            application_event.payload if application_event is not None else {},
            kind=item.kind.value,
        )
        status_text = {
            "recorded": "仅记录（未执行）",
            "applied": "已应用",
            "enforced": "已强制执行",
        }[receipt["application_status"]]
    return {
        **item.to_dict(),
        "id": item.intervention_id,
        "effective_generation": effective_generation,
        **receipt,
        "status": status_text,
    }


def _expert_consultation_projection(state: Any, item: Any) -> dict[str, Any]:
    """Return the bounded, flat collaboration contract used by the browser."""

    answer = state.answer_for_consultation(item.consultation_id)

    def public_text(value: Any, *, limit: int) -> str | None:
        if value is None:
            return None
        projected = sanitize_public_value(str(value), text_limit=limit)
        return projected if isinstance(projected, str) else None

    def display_generation(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value + 1

    return {
        "id": item.consultation_id,
        "consultation_id": item.consultation_id,
        "status": "answered" if answer is not None else "pending",
        "question": public_text(item.question, limit=4000),
        "context": public_text(item.context, limit=4000),
        "fallback_assumption": public_text(
            item.fallback_assumption,
            limit=4000,
        ),
        "options": [
            public_text(option, limit=500)
            for option in item.options
        ],
        "requested_expertise": [
            public_text(expertise, limit=500)
            for expertise in item.requested_expertise
        ],
        "uncertainty_type": public_text(
            getattr(item.uncertainty_type, "value", item.uncertainty_type),
            limit=200,
        ),
        "confidence": sanitize_public_value(item.confidence),
        "source_generation": display_generation(item.generation),
        "candidate_id": item.candidate_id,
        "model_id": public_text(item.requested_by_model_id, limit=500),
        "non_blocking": item.non_blocking is True,
        "created_at": item.created_at,
        "answer_id": answer.answer_id if answer is not None else None,
        "answer": (
            public_text(answer.answer, limit=4000)
            if answer is not None
            else None
        ),
        "answered_by": (
            public_text(answer.answered_by, limit=120)
            if answer is not None
            else None
        ),
        "answered_at": answer.created_at if answer is not None else None,
        "selected_option": (
            public_text(answer.selected_option, limit=500)
            if answer is not None
            else None
        ),
        "effective_generation": (
            display_generation(answer.effective_generation)
            if answer is not None
            else None
        ),
        "applied_generation": (
            display_generation(answer.applied_generation)
            if answer is not None
            else None
        ),
    }


def _projection_json(state: Any) -> dict[str, Any]:
    """Build the small browser-safe read model from the event projection."""

    # Keep this helper safe even when called outside ``_state_payload`` (for
    # example by a list or diagnostic path).  A projection must never be
    # assembled before its scientific partition has been authorized.
    _assert_http_scope(state)
    task = state.task_manifest
    run = state.run
    latest_event = state.events[-1].created_at if state.events else run.created_at
    selected = best_observed_evaluation(state)
    acceptable_evaluation = (
        state.evaluation_for(run.best_candidate_id)
        if run.best_candidate_id is not None
        else None
    )
    outcome, termination_reason = run_completion_outcome(state)
    failure_reason, failed_stage = _run_failure_projection(state)
    failure_code = _run_failure_code(state)
    metrics = (
        _public_evaluation_metrics(acceptable_evaluation.metrics)
        if acceptable_evaluation is not None
        else {}
    )
    if acceptable_evaluation is not None:
        metrics.update(
            {
                "score": acceptable_evaluation.score,
                "passed": acceptable_evaluation.passed,
                "partition": acceptable_evaluation.partition,
            }
        )
    visible_pass = any(item.passed for item in state.evaluations)
    all_evaluated = bool(state.candidates) and all(
        item.status.value in {"evaluated", "promoted", "rejected", "failed", "duplicate"}
        for item in state.candidates
    )
    dataset_id = task.visible_datasets[0] if task.visible_datasets else None
    metadata = dict(task.metadata)
    slot = metadata.get("slot")
    trajectory = []
    best_observed_score: float | None = None
    incumbent_score: float | None = None
    projected_incumbent_evaluation = None
    windowed_scores = sample_update_windows_enabled(task)
    observed_score_scope = (
        "observation_only_cross_cohort_not_comparable"
        if windowed_scores
        else "observation_only_full_cohort"
    )
    ordered_candidates = sorted(
        state.candidates,
        key=lambda item: (item.generation, item.slot_index, item.candidate_id),
    )
    for candidate in ordered_candidates:
        evaluation = state.evaluation_for(candidate.candidate_id)
        if evaluation is None:
            continue
        promotion = state.promotion_for(candidate.candidate_id)
        is_champion = bool(
            promotion is not None and promotion.decision.value == "approved"
        )
        analysis = state.analysis_for(candidate.generation)
        incumbent_before_evaluation = projected_incumbent_evaluation
        if analysis is not None and analysis.incumbent_before_candidate_id is not None:
            incumbent_before_evaluation = state.evaluation_for(
                analysis.incumbent_before_candidate_id
            )
        cohort_boundary = (
            evaluation_cohort_comparison(
                task,
                evaluation,
                incumbent_before_evaluation,
            )
            if incumbent_before_evaluation is not None
            else "no_incumbent"
        )
        if is_champion:
            # A different-cohort champion becomes the actual current incumbent
            # even when its raw score is lower than the previous window's score.
            incumbent_score = evaluation.score
            projected_incumbent_evaluation = evaluation
        best_observed_score = (
            evaluation.score
            if best_observed_score is None
            else max(best_observed_score, evaluation.score)
        )
        trajectory_metrics = dict(evaluation.metrics)
        trajectory.append(
            {
                "generation": candidate.generation + 1,
                "candidate_id": candidate.candidate_id,
                "candidate_score": evaluation.score,
                "score": evaluation.score,
                # ``best_score`` remains the accepted incumbent for older
                # clients.  The observed search trajectory is separate so a
                # near miss is never presented as an acceptable candidate.
                "best_score": incumbent_score,
                "incumbent_score": incumbent_score,
                "best_observed_score": best_observed_score,
                "best_observed_score_scope": observed_score_scope,
                "evaluation_cohort_digest": evaluation_cohort_digest(evaluation),
                "incumbent_before_cohort_digest": (
                    evaluation_cohort_digest(incumbent_before_evaluation)
                    if incumbent_before_evaluation is not None
                    else None
                ),
                "score_comparison_boundary": cohort_boundary,
                "score_comparable_to_incumbent_before": cohort_boundary
                in {"legacy_full_cohort", "same_cohort"},
                "is_champion": is_champion,
                "passed": evaluation.passed,
                "scientific_pass": trajectory_metrics.get("scientific_pass"),
                "judge_accepted": trajectory_metrics.get("judge_accepted"),
                "constraint_violations": trajectory_metrics.get(
                    "constraint_violations", 0
                ),
                "baseline_score": 0.0,
                "slot_index": candidate.slot_index,
                "generation_rank": (
                    next(
                        (
                            item.get("rank")
                            for item in (state.analysis_for(candidate.generation).ranking)
                            if item.get("candidate_id") == candidate.candidate_id
                        ),
                        None,
                    )
                    if state.analysis_for(candidate.generation) is not None
                    else None
                ),
            }
        )
    token_budget_scope = _token_budget_scope(task, metadata)
    configuration = {
        "execution_protocol": metadata.get("execution_protocol", "legacy_read_only"),
        "domain_pack_id": task.domain_pack,
        "dataset_id": dataset_id,
        "episode_id": metadata.get("episode_id"),
        "strategy_id": metadata.get("strategy_id", "parameter_sweep@1"),
        "strategy_digest": metadata.get("strategy_digest"),
        "prediction_model_id": metadata.get(
            "prediction_model_id",
            TOY_PREDICTOR_MODEL_ID
            if dataset_id == TOY_DATASET_ID
            else EXOGENOUS_RIDGE_MODEL_ID,
        ),
        "prediction_model_digest": metadata.get("prediction_model_digest"),
        "evaluator_id": metadata.get(
            "evaluator_id",
            TOY_EVALUATOR_ID
            if dataset_id == TOY_DATASET_ID
            else GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID,
        ),
        "evaluator_digest": metadata.get("evaluator_digest"),
        "objective_profile": metadata.get("objective_profile"),
        "policy_model_id": metadata.get("policy_model_id", HOST_PARAMETER_GENERATOR_ID),
        "judge_model_id": metadata.get("judge_model_id", RULE_JUDGE_ID),
        "strategy_model_id": metadata.get(
            "strategy_model_id",
            metadata.get("policy_model_id", HOST_PARAMETER_GENERATOR_ID),
        ),
        "review_model_id": metadata.get(
            "review_model_id",
            metadata.get("judge_model_id", RULE_JUDGE_ID),
        ),
        "sample_agent_mode": metadata.get(
            "sample_agent_mode", "host_feedback_state_machine"
        ),
        "sample_agent_batch_size": metadata.get("sample_agent_batch_size"),
        "samples_per_update": metadata.get("samples_per_update"),
        "sample_concurrency": metadata.get("sample_concurrency", 4),
        "sample_operation_max_tokens": metadata.get("sample_operation_max_tokens"),
        "sample_remote_critic_policy": metadata.get(
            "sample_remote_critic_policy"
        ),
        "sample_planner_prompt_profile": metadata.get(
            "sample_planner_prompt_profile"
        ),
        "sample_truncation_retry_policy": metadata.get(
            "sample_truncation_retry_policy"
        ),
        "sample_token_budget_policy": metadata.get("sample_token_budget_policy"),
        "token_budget_scope": token_budget_scope,
        # The current ledger does not include research, proposal, or judge
        # calls, even when every recorded sample-agent receipt is complete.
        "run_wide_accounting_complete": False,
        "autonomous_mode": bool(metadata.get("autonomous_mode", False)),
        "auto_progress": metadata.get("auto_progress") is True,
        "auto_progress_policy": metadata.get("auto_progress_policy"),
        "allow_host_fallback": metadata.get("allow_host_fallback") is True,
        "remote_fallback_policy": metadata.get("remote_fallback_policy"),
        "model_selection_policy": metadata.get("model_selection_policy"),
        "model_workflow": metadata.get("model_workflow"),
        "autonomous_plan_execution": metadata.get("autonomous_plan_execution"),
        "research_domain": metadata.get("research_domain", task.domain_pack),
        "research_domain_id": metadata.get("research_domain", task.domain_pack),
        "autonomous_plan": metadata.get("autonomous_plan"),
        "autonomous_plan_digest": metadata.get("autonomous_plan_digest"),
        "model_team": (
            metadata.get("autonomous_plan", {}).get("team")
            if isinstance(metadata.get("autonomous_plan"), dict)
            else None
        ),
        "model_selected_prediction": (
            metadata.get("autonomous_plan", {}).get("prediction_model")
            if isinstance(metadata.get("autonomous_plan"), dict)
            else None
        ),
        "model_selected_strategy": (
            metadata.get("autonomous_plan", {}).get("strategy")
            if isinstance(metadata.get("autonomous_plan"), dict)
            else None
        ),
        "policy_model_digest": metadata.get("policy_model_digest"),
        "judge_model_digest": metadata.get("judge_model_digest"),
        "policy_model_binding_source": metadata.get("policy_model_binding_source"),
        "judge_model_binding_source": metadata.get("judge_model_binding_source"),
        "slot": slot,
        "candidates_per_generation": task.candidates_per_generation,
        "candidates_per_round": task.candidates_per_generation,
        "variants_per_round": task.candidates_per_generation,
        "knowledge_online_enabled": bool(metadata.get("knowledge_online_enabled", False)),
    }
    execution_progress = _run_execution_progress(state)
    execution_diagnostics = _execution_diagnostics(state)
    model_usage = _model_usage_summary(state)
    dsh_runtime = _dsh_runtime_projection(state)
    pause_reason, pause_code = _run_pause_projection(state)
    return {
        "id": run.run_id,
        "run_id": run.run_id,
        "status": run.status.value,
        "outcome": outcome,
        "termination_reason": termination_reason,
        "failure_reason": failure_reason,
        "failure_code": failure_code,
        "failed_stage": failed_stage,
        "pause_reason": pause_reason,
        "pause_code": pause_code,
        "created_at": run.created_at,
        "updated_at": latest_event,
        "generation": run.generation,
        "total_generations": _max_generations(task),
        "candidates_count": len(state.candidates),
        "max_candidates": task.max_candidates,
        "candidates_per_generation": task.candidates_per_generation,
        "samples_per_update": metadata.get("samples_per_update"),
        "sample_agent_batch_size": metadata.get("sample_agent_batch_size"),
        "sample_concurrency": metadata.get("sample_concurrency"),
        "budget": dict(task.budget),
        "token_usage_available": model_usage["available"],
        "tokens_used": model_usage.get(
            "budget_accounted_tokens", model_usage["total_tokens"]
        ),
        "token_limit": _budget_value(task, "token_limit", 0),
        "token_reservation_per_wave": _budget_value(
            task, "token_reservation_per_wave", 0
        ),
        "token_budget_scope": token_budget_scope,
        "run_wide_accounting_complete": False,
        "model_usage": model_usage,
        "dsh_runtime": dsh_runtime,
        "manifest_digest": task.digest,
        "task_manifest_digest": task.digest,
        "dataset_digest": metadata.get("dataset_digest"),
        "seed": task.seed,
        "seed_policy": task.seed_policy,
        "policy_version": task.policy_version,
        "scientific_scope": metadata.get("scientific_scope", "prediction_demo_non_causal"),
        "selection_scope": "iterative_training_feedback_only",
        "formal_validation_status": "not_run",
        "best_candidate_scope": (
            "iterative_training_feedback_only"
            if run.best_candidate_id is not None
            else None
        ),
        "evaluation_partition": metadata.get("evaluation_partition", _expected_partition(task)),
        "research_domain": metadata.get("research_domain", task.domain_pack),
        "research_domain_id": metadata.get("research_domain", task.domain_pack),
        "model_workflow": metadata.get("model_workflow"),
        "auto_progress": metadata.get("auto_progress") is True,
        "auto_progress_policy": metadata.get("auto_progress_policy"),
        "allow_host_fallback": metadata.get("allow_host_fallback") is True,
        "remote_fallback_policy": metadata.get("remote_fallback_policy"),
        "projection_revision": state.events[-1].seq if state.events else 0,
        "task": {
            "task_id": task.task_id,
            "objective": task.objective,
            "domain_pack_id": task.domain_pack,
            "dataset_id": dataset_id,
            "slot": slot,
            "autonomous_mode": bool(metadata.get("autonomous_mode", False)),
            "auto_progress": metadata.get("auto_progress") is True,
            "research_domain_id": metadata.get("research_domain", task.domain_pack),
        },
        "configuration": configuration,
        "dataset": {
            "id": dataset_id,
            "display_name": metadata.get("dataset_display_name", dataset_id),
            "episode_id": metadata.get("episode_id"),
            "digest": metadata.get("dataset_digest"),
            "split_manifest_digest": metadata.get("split_manifest_digest"),
            "partition": metadata.get("evaluation_partition", _expected_partition(task)),
        },
        "metrics": metrics,
        "metrics_candidate_id": (
            acceptable_evaluation.candidate_id
            if acceptable_evaluation is not None
            else None
        ),
        "metrics_scope": (
            "current_search_incumbent"
            if acceptable_evaluation is not None
            else "no_current_search_incumbent"
        ),
        "gate": {
            "visible": "通过" if visible_pass else ("未通过" if state.evaluations else "未开始"),
            "process": "通过" if all_evaluated and state.evaluations else "等待",
            "hidden": "未开放",
            "release": "待审批",
        },
        "trajectory": trajectory,
        "execution_progress": execution_progress,
        "execution_diagnostics": execution_diagnostics,
        "rounds": _rounds_projection(state),
        "generation_batches": [item.to_dict() for item in state.generation_batches],
        "generation_analyses": [item.to_dict() for item in state.generation_analyses],
        "knowledge_snapshots": [item.to_dict() for item in state.knowledge_snapshots],
        "knowledge_assessments": [
            item.to_dict() for item in state.knowledge_assessments
        ],
        "algorithm_attempts": [
            item.to_dict() for item in state.algorithm_attempts
        ],
        "training_assets": training_assets(state),
        "artifacts": [item.to_dict() for item in reversed(state.artifacts)],
        "interventions": [
            _intervention_projection(state, item) for item in reversed(state.interventions)
        ],
        "expert_consultations": [
            _expert_consultation_projection(state, item)
            for item in reversed(state.expert_consultations)
        ],
        "best_candidate_id": run.best_candidate_id,
        "selection_incumbent_id": run.selection_incumbent_id,
        "search_parent_candidate_id": (
            state.generation_batches[-1].parent_candidate_id
            if state.generation_batches
            else None
        ),
        "validated_candidate_id": run.validated_candidate_id,
        "final_test_candidate_id": run.final_test_candidate_id,
        "best_candidate_score": (
            acceptable_evaluation.score
            if acceptable_evaluation is not None
            else None
        ),
        "best_observed_candidate_id": (
            selected.candidate_id if selected is not None else None
        ),
        "best_observed_score": selected.score if selected is not None else None,
        "best_observed_score_scope": observed_score_scope,
        "best_observed_drives_current_metrics": False,
        "candidates": [_candidate_projection(state, item) for item in reversed(state.candidates)],
    }


def _run_summary_projection(state: Any) -> dict[str, Any]:
    """Build the bounded run-list row without materializing full evidence."""

    _assert_http_scope(state)
    task = state.task_manifest
    run = state.run
    metadata = dict(task.metadata)
    latest_event = state.events[-1].created_at if state.events else run.created_at
    outcome, termination_reason = run_completion_outcome(state)
    failure_reason, failed_stage = _run_failure_projection(state)
    pause_reason, pause_code = _run_pause_projection(state)
    observed = best_observed_evaluation(state)
    acceptable = (
        state.evaluation_for(run.best_candidate_id)
        if run.best_candidate_id is not None
        else None
    )
    dataset_id = task.visible_datasets[0] if task.visible_datasets else None
    configuration = {
        "dataset_id": dataset_id,
        "episode_id": metadata.get("episode_id"),
        "strategy_model_id": metadata.get(
            "strategy_model_id",
            metadata.get("policy_model_id", HOST_PARAMETER_GENERATOR_ID),
        ),
        "review_model_id": metadata.get(
            "review_model_id",
            metadata.get("judge_model_id", RULE_JUDGE_ID),
        ),
        "policy_model_id": metadata.get(
            "policy_model_id", HOST_PARAMETER_GENERATOR_ID
        ),
        "judge_model_id": metadata.get("judge_model_id", RULE_JUDGE_ID),
        "autonomous_mode": bool(metadata.get("autonomous_mode", False)),
        "model_workflow": metadata.get("model_workflow"),
        "knowledge_online_enabled": bool(
            metadata.get("knowledge_online_enabled", False)
        ),
        "samples_per_update": metadata.get("samples_per_update"),
        "sample_agent_batch_size": metadata.get("sample_agent_batch_size"),
        "sample_concurrency": metadata.get("sample_concurrency"),
    }
    return {
        "schema_version": "ecologyrsi-dsh.browser-run-summary/1",
        "id": run.run_id,
        "run_id": run.run_id,
        "status": run.status.value,
        "outcome": outcome,
        "termination_reason": termination_reason,
        "failure_reason": failure_reason,
        "failure_code": _run_failure_code(state),
        "failed_stage": failed_stage,
        "pause_reason": pause_reason,
        "pause_code": pause_code,
        "created_at": run.created_at,
        "updated_at": latest_event,
        "projection_revision": state.events[-1].seq if state.events else 0,
        "generation": run.generation,
        "total_generations": _max_generations(task),
        "candidates_count": len(state.candidates),
        "max_candidates": task.max_candidates,
        "candidates_per_generation": task.candidates_per_generation,
        "samples_per_update": metadata.get("samples_per_update"),
        "sample_agent_batch_size": metadata.get("sample_agent_batch_size"),
        "sample_concurrency": metadata.get("sample_concurrency"),
        "token_limit": _budget_value(task, "token_limit", 0),
        "budget": dict(task.budget),
        "seed_policy": task.seed_policy,
        "auto_progress": metadata.get("auto_progress") is True,
        "auto_progress_policy": metadata.get("auto_progress_policy"),
        "configuration": configuration,
        "best_candidate_id": run.best_candidate_id,
        "best_candidate_score": (
            acceptable.score if acceptable is not None else None
        ),
        "best_observed_candidate_id": (
            observed.candidate_id if observed is not None else None
        ),
        "best_observed_score": observed.score if observed is not None else None,
    }


def _state_payload(state: Any) -> dict[str, Any]:
    _assert_http_scope(state)
    return {
        "schema_version": "ecologyrsi-dsh.browser-run/2",
        "projection": _projection_json(state),
    }
