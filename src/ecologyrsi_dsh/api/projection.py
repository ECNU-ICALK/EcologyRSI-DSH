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
from ..core.models import Evaluation, HumanIntervention, InterventionKind
from ..core.protocols import supports_two_stage_screening
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
from .sample_admission import (
    HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK,
    MAX_SAMPLE_CONCURRENCY,
)

_TWO_STAGE_SCREENING_ORIGINS = 64
_HISTORICAL_PREDICTION_CELLS_PER_ORIGIN = 9

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
_RETRY_CIRCUIT_PUBLIC_REASONS = {
    "gateway_retry_circuit_open": (
        "模型网关连续重试已达到安全上限；"
        "请检查模型网关后恢复，以重试当前检查点。"
    ),
    "dsh_runtime_retry_circuit_open": (
        "DSH 智能体运行时连续重试已达到安全上限；"
        "请检查 DSH 运行时后恢复，以重试当前检查点。"
    ),
    "research_timeout_retry_circuit_open": (
        "研究阶段请求连续超时已达到安全上限；"
        "请检查模型网关后恢复，以重试当前检查点。"
    ),
    "sample_persistence_retry_circuit_open": (
        "样本结果持久化连续失败已达到安全上限；"
        "请检查持久化服务后恢复，以重试当前检查点。"
    ),
}


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

    # Stage failures are retry observations, not terminal run failures.  Once a
    # later attempt completes the run, keeping the earlier stage receipt in the
    # top-level failure fields makes a healthy run look internally inconsistent.
    run_status = getattr(
        getattr(getattr(state, "run", None), "status", None), "value", None
    )
    if run_status == "completed":
        return None, None

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
    terminal_context = (
        failed_event.payload.get("failure_context")
        if failed_event is not None
        else None
    )
    if isinstance(terminal_context, Mapping):
        return reason, {
            "generation": terminal_context.get("generation"),
            "stage": terminal_context.get("stage"),
            "work_unit_kind": terminal_context.get("work_unit_kind"),
            "candidate_id": terminal_context.get("candidate_id"),
            "batch_id": terminal_context.get("batch_id"),
            "batch_index": terminal_context.get("batch_index"),
            "batch_count": terminal_context.get("batch_count"),
            "created_at": failed_event.created_at,
            "evidence": "terminal_run_failure_context",
        }
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


def _run_pause_projection(
    state: Any,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Return the active pause cause without reviving an older pause event."""

    if state.run.status.value != "paused":
        return None, None, None
    paused_event = next(
        (event for event in reversed(state.events) if event.kind == "RunPaused"),
        None,
    )
    if paused_event is None:
        return None, None, None
    raw_reason = paused_event.payload.get("reason")
    reason = (
        public_error_summary(raw_reason)
        if isinstance(raw_reason, str) and raw_reason.strip()
        else None
    )
    raw_code = paused_event.payload.get("code")
    code = str(raw_code) if isinstance(raw_code, str) and raw_code else None
    circuit = None
    if code in _RETRY_CIRCUIT_PUBLIC_REASONS:
        reason = _RETRY_CIRCUIT_PUBLIC_REASONS[code]
        payload = paused_event.payload
        circuit = {
            "open": True,
            "code": code,
            "retry_class": payload.get("retry_class"),
            "generation": payload.get("generation"),
            "stage": payload.get("stage"),
            "breaker_epoch": payload.get("breaker_epoch"),
            "consecutive_failures": payload.get("consecutive_failures"),
            "retry_limit": payload.get("retry_limit"),
            "first_failure_at": payload.get("first_failure_at"),
            "last_failure_at": payload.get("last_failure_at"),
            "last_error_code": payload.get("last_error_code"),
            "suggested_action": payload.get("suggested_action"),
            "pause_event_id": paused_event.event_id,
            "updated_at": paused_event.created_at,
        }
    return reason, code, circuit


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
    adaptive_evaluations = [
        *state.formal_batch_evaluations,
        *state.holdout_evaluations,
    ]
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
    elif artifacts or evaluations or adaptive_evaluations:
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
    adaptive_settled_origins = sum(
        int(item.scope.origin_count) for item in adaptive_evaluations
    )
    adaptive_succeeded_origins = 0
    adaptive_failed_origins = 0
    adaptive_failed_scoring_cells = 0
    adaptive_outcome_counts_known = True
    for evaluation in adaptive_evaluations:
        sample = (
            evaluation.metrics.get("sample_execution")
            if isinstance(evaluation.metrics, Mapping)
            else None
        )
        if not isinstance(sample, Mapping):
            adaptive_outcome_counts_known = False
            continue
        succeeded = metric_count(sample, "succeeded_origin_samples")
        failed = metric_count(sample, "failed_origin_samples")
        failed_cells = metric_count(sample, "failed_examples")
        if succeeded is None or failed is None:
            adaptive_outcome_counts_known = False
        else:
            adaptive_succeeded_origins += succeeded
            adaptive_failed_origins += failed
        adaptive_failed_scoring_cells += failed_cells or 0
    adaptive_progress = _adaptive_progress_projection(state)
    adaptive_live_completed_origins = (
        metric_count(adaptive_progress, "completed_origins")
        if adaptive_progress is not None
        else None
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
        "candidate_concurrency": state.task_manifest.metadata.get(
            "candidate_concurrency"
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
        "adaptive_settled_origins": adaptive_settled_origins,
        "adaptive_live_completed_origins": adaptive_live_completed_origins,
        "adaptive_succeeded_origins": (
            adaptive_succeeded_origins if adaptive_outcome_counts_known else None
        ),
        "adaptive_failed_origins": (
            adaptive_failed_origins if adaptive_outcome_counts_known else None
        ),
        "adaptive_failed_scoring_cells": adaptive_failed_scoring_cells,
        "adaptive_evaluation_count": len(adaptive_evaluations),
        "candidate_work_items": (
            training_used_examples
            + evaluation_used_examples
            + live_evaluation_completed_examples
            + adaptive_settled_origins
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
    raw_controls = result.pop("generation_control_evaluations", None)
    if isinstance(raw_controls, list):
        controls: list[dict[str, Any]] = []
        for raw in raw_controls:
            if not isinstance(raw, Mapping):
                continue
            evaluation = raw.get("evaluation")
            evaluation = evaluation if isinstance(evaluation, Mapping) else {}
            control_metrics = evaluation.get("metrics")
            control_metrics = (
                control_metrics if isinstance(control_metrics, Mapping) else {}
            )
            sample_summary = control_metrics.get("sample_execution")
            sample_summary = (
                sample_summary if isinstance(sample_summary, Mapping) else {}
            )
            control_cohort_digest = None
            if evaluation:
                try:
                    control_cohort_digest = evaluation_cohort_digest(
                        Evaluation.from_dict(evaluation)
                    )
                except (TypeError, ValueError):
                    # Historical or damaged control evidence must not make the
                    # read-only UI fail.  The selection path validates the same
                    # payload strictly and will refuse to advance the round.
                    control_cohort_digest = None
            controls.append(
                {
                    "comparison_role": raw.get("comparison_role"),
                    "candidate_id": raw.get("candidate_id"),
                    "score": _finite_number(evaluation.get("score")),
                    "scientific_pass": control_metrics.get("scientific_pass"),
                    "evaluation_cohort_digest": control_cohort_digest,
                    "strict_agent_chain_pass": sample_summary.get(
                        "strict_agent_chain_pass"
                    ),
                    "complete_agent_chains": sample_summary.get(
                        "complete_agent_chains"
                    ),
                    "complete_origin_agent_chains": sample_summary.get(
                        "complete_origin_agent_chains"
                    ),
                    "remote_planner_invocations": sample_summary.get(
                        "remote_planner_invocations"
                    ),
                    "remote_critic_invocations": sample_summary.get(
                        "remote_critic_invocations"
                    ),
                    "remote_reflection_invocations": sample_summary.get(
                        "remote_reflection_invocations"
                    ),
                    "attempted_examples": sample_summary.get(
                        "attempted_examples"
                    ),
                    "attempted_origin_samples": sample_summary.get(
                        "attempted_origin_samples"
                    ),
                    "prediction_cell_count": sample_summary.get(
                        "prediction_cell_count"
                    ),
                    "host_route_bypass_count": sample_summary.get(
                        "host_route_bypass_count"
                    ),
                }
            )
        result["generation_controls"] = controls
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
    latest_resume = next(
        (
            item
            for item in reversed(state.events)
            if item.kind == "EvaluationSampleResultsResumed"
            and item.payload.get("candidate_id") == candidate_id
            and item.payload.get("revision") == active_revision
        ),
        None,
    )
    samples_per_minute, gateway_calls_per_minute = _evaluation_progress_rates(
        events,
        event,
        after_seq=int(latest_resume.seq) if latest_resume is not None else None,
    )
    estimated_remaining_seconds = (
        round(60.0 * max(0, total - completed) / samples_per_minute)
        if samples_per_minute is not None and samples_per_minute > 0
        else None
    )
    metadata = state.task_manifest.metadata
    configured_concurrency = metadata.get(
        "sample_concurrency",
        HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK,
    )
    if (
        isinstance(configured_concurrency, bool)
        or not isinstance(configured_concurrency, int)
        or not 1 <= configured_concurrency <= MAX_SAMPLE_CONCURRENCY
    ):
        configured_concurrency = None
    in_flight_batches = payload.get("in_flight_batches")
    queued_batches = payload.get("queued_batches")
    awaiting_submission_batches = payload.get("awaiting_submission_batches")
    if not (
        isinstance(awaiting_submission_batches, int)
        and not isinstance(awaiting_submission_batches, bool)
        and awaiting_submission_batches >= 0
    ):
        awaiting_submission_batches = (
            max(
                0,
                total
                - completed
                - in_flight_batches
                - queued_batches,
            )
            if all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in (in_flight_batches, queued_batches)
            )
            else None
        )
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
        "in_flight_batches": in_flight_batches,
        "queued_batches": queued_batches,
        "awaiting_submission_batches": awaiting_submission_batches,
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
    events: list[Any], latest: Any, *, after_seq: int | None = None
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
            and (after_seq is None or item.seq > after_seq)
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
    for newer in state.events:
        if newer.seq <= event.seq:
            continue
        if newer.kind in {"RunPaused", "RunResumed", "RunStarted"}:
            return None
        if (
            event.payload.get("continuity_reset_contract")
            == "dsh_structured_success@1"
            and newer.kind == "DshStructuredResultAccepted"
        ):
            # Native DSH retries are generation-scoped, so their synthetic
            # stage name need not match an EvolutionStageRecorded row.  The
            # durable structured receipt is the explicit v2 continuity-reset
            # witness and must clear the stale cooldown banner immediately.
            return None
        if (
            newer.kind == "EvolutionStageRecorded"
            and newer.payload.get("status") in {"started", "completed"}
            and int(newer.payload.get("generation", -1))
            == int(event.payload.get("generation", -2))
            and newer.payload.get("stage") == event.payload.get("stage")
        ):
            return None
    payload = event.payload
    projected = {
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
    if payload.get("schema_version") == "ecologyrsi-dsh.gateway-retry-scheduled/2":
        projected.update(
            {
                "retry_class": payload.get("retry_class"),
                "stage": payload.get("stage"),
                "breaker_epoch": payload.get("breaker_epoch"),
                "consecutive_failures": payload.get("consecutive_failures"),
                "retry_limit": payload.get("retry_limit"),
                "first_failure_at": payload.get("first_failure_at"),
                "last_failure_at": payload.get("last_failure_at"),
                "last_error_code": payload.get("last_error_code"),
                "suggested_action": "wait_for_scheduled_retry",
            }
        )
    return projected


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


def _dsh_evolution_stage(stage: str | None) -> str | None:
    value = str(stage or "").strip().lower()
    if value == "generation.search-plan":
        return "search"
    if value in {"generation.research", "generation.research-synthesis"}:
        return "research"
    if value == "candidate.propose":
        return "proposal"
    if value == "candidate.local_edit":
        return "evaluation"
    if value.startswith("sample."):
        return "evaluation"
    if value == "generation.judge":
        return "judge"
    if value == "generation.reflect":
        return "reflection"
    return None


def _sample_member_digest_set(value: Any) -> frozenset[str] | None:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 128:
        return None
    members = tuple(value)
    if any(
        not isinstance(member, str)
        or len(member) != 64
        or any(character not in "0123456789abcdef" for character in member)
        for member in members
    ) or len(members) != len(set(members)):
        return None
    return frozenset(members)


def _legacy_sample_launch_key(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    for stage in ("plan", "critic", "reflect"):
        marker = f"sample.{stage}:"
        if marker not in value:
            continue
        prefix, suffix = value.rsplit(marker, 1)
        if suffix:
            return prefix.rstrip(":"), suffix
    return None


def _screening_progress_projection(state: Any) -> dict[str, Any] | None:
    """Project live two-stage screening from durable DSH child events.

    Screening intentionally does not publish formal sample rows.  The child
    ledger still proves completed reflections and currently active provider
    calls, so expose those aggregate counts without treating not-yet-submitted
    origins as provider-queued requests.
    """

    metadata = state.task_manifest.metadata
    configured_cells_per_origin = metadata.get(
        "prediction_cells_per_origin",
        _HISTORICAL_PREDICTION_CELLS_PER_ORIGIN,
    )
    cells_per_origin = (
        configured_cells_per_origin
        if isinstance(configured_cells_per_origin, int)
        and not isinstance(configured_cells_per_origin, bool)
        and configured_cells_per_origin > 0
        else _HISTORICAL_PREDICTION_CELLS_PER_ORIGIN
    )
    if (
        not supports_two_stage_screening(metadata.get("sample_agent_protocol"))
        or metadata.get("two_stage_evaluation_enabled", True) is not True
        or state.run.status.value != "running"
    ):
        return None
    generation = int(state.run.generation)
    candidates = tuple(
        candidate
        for candidate in state.candidates
        if int(candidate.generation) == generation
    )
    if len(candidates) <= 2:
        return None
    batch_start = next(
        (
            event
            for event in reversed(state.events)
            if event.kind == "GenerationBatchStarted"
            and isinstance(event.payload.get("batch"), Mapping)
            and int(event.payload["batch"].get("generation", -1)) == generation
        ),
        None,
    )
    if batch_start is None:
        return None
    batch_seq = int(batch_start.seq)
    if any(
        event.kind == "FormalSelectionCohortFrozen"
        and int(event.seq) > batch_seq
        and int(event.payload.get("generation", -1)) == generation
        for event in state.events
    ):
        return None

    sample_events: list[Any] = []
    latest_launch_by_key: dict[str, tuple[Any, Mapping[str, Any]]] = {}
    launch_history_by_key: dict[str, tuple[Any, Mapping[str, Any]]] = {}
    launch_by_reservation_id: dict[str, tuple[Any, Mapping[str, Any]]] = {}
    accepted_reservations: set[str] = set()
    failed_reservations: set[str] = set()
    reflection_enabled = (
        metadata.get("sample_reflection_policy")
        != "candidate_aggregate_post_score@1"
    )
    terminal_stage = "sample.reflect" if reflection_enabled else "sample.plan"
    completed_terminals: dict[str, Any] = {}
    launch_count = 0
    primary_launch_count = 0
    repair_launch_count = 0
    for event in state.events:
        if int(event.seq) <= batch_seq:
            continue
        if event.kind in {"RunPaused", "RunResumed"} or (
            event.kind == "GatewayRetryScheduled"
            and int(event.payload.get("generation", -1)) == generation
        ):
            # Every admitted worker from the preceding process/control attempt
            # has settled before these durable boundaries.  Reservations that
            # never produced a result are terminal failures, not live provider
            # work in the new attempt.
            latest_launch_by_key.clear()
            continue
        if event.kind == "DshChildLaunchReserved":
            launch = event.payload.get("launch")
            if not isinstance(launch, Mapping) or not str(
                launch.get("stage") or ""
            ).startswith("sample."):
                continue
            key = str(launch.get("idempotency_key") or "").strip()
            if not key:
                continue
            launch_count += 1
            launch_members = _sample_member_digest_set(
                launch.get("sample_member_digests")
            )
            if launch.get("stage") == "sample.plan" and launch_members is not None:
                if len(launch_members) == cells_per_origin:
                    primary_launch_count += 1
                elif len(launch_members) < cells_per_origin:
                    repair_launch_count += 1
            latest_launch_by_key[key] = (event, launch)
            launch_history_by_key[key] = (event, launch)
            reservation_id = str(launch.get("reservation_id") or "").strip()
            if reservation_id:
                launch_by_reservation_id[reservation_id] = (event, launch)
            sample_events.append(event)
        elif event.kind == "DshStructuredResultAccepted":
            identity = event.payload.get("identity")
            if not isinstance(identity, Mapping) or not str(
                identity.get("stage") or ""
            ).startswith("sample."):
                continue
            reservation_id = str(
                identity.get("child_reservation_id") or ""
            ).strip()
            if reservation_id:
                accepted_reservations.add(reservation_id)
            if identity.get("stage") == terminal_stage:
                key = str(identity.get("idempotency_key") or "").strip()
                if key:
                    completed_terminals[key] = event
            sample_events.append(event)
        elif event.kind == "DshChildExecutionFailed":
            identity = event.payload.get("identity")
            if not isinstance(identity, Mapping) or not str(
                identity.get("stage") or ""
            ).startswith("sample."):
                continue
            reservation_id = str(
                identity.get("child_reservation_id") or ""
            ).strip()
            if reservation_id:
                failed_reservations.add(reservation_id)
            sample_events.append(event)
        elif event.kind == "DshPredictionToolExecuted" and not reflection_enabled:
            # For aggregate post-score reflection this receipt proves that the
            # registered predictor produced the complete origin vector. It can
            # advance prediction progress, but the structured candidate result
            # below remains the only boundary that advances the workflow from
            # screening into formal evaluation.
            stage = str(event.payload.get("stage") or "")
            key = str(event.payload.get("idempotency_key") or "").strip()
            if stage == terminal_stage and key:
                completed_terminals[key] = event
            sample_events.append(event)
    if not sample_events:
        return None

    completed_origins: dict[
        tuple[str, frozenset[str] | tuple[str, str] | str],
        tuple[Any, frozenset[str] | None, tuple[str, str] | None],
    ] = {}
    completed_repair_waves: dict[str, Any] = {}
    for terminal_key, terminal_event in completed_terminals.items():
        identity = terminal_event.payload.get("identity")
        reflection_reservation = str(
            identity.get("child_reservation_id") or ""
            if isinstance(identity, Mapping)
            else ""
        ).strip()
        reflection_launch = launch_by_reservation_id.get(reflection_reservation)
        terminal_launch = launch_history_by_key.get(terminal_key)
        correlated_launch = reflection_launch or terminal_launch
        terminal_idempotency_key = (
            identity.get("idempotency_key")
            if isinstance(identity, Mapping)
            else terminal_event.payload.get("idempotency_key")
        )
        member_digests = _sample_member_digest_set(
            correlated_launch[1].get("sample_member_digests")
            if correlated_launch is not None
            else None
        )
        legacy_key = _legacy_sample_launch_key(terminal_idempotency_key)
        if not reflection_enabled:
            # Aggregate-reflection screening uses sample.plan as the remote
            # origin boundary.  Sparse retry/repair waves use the same stage,
            # but they do not prove that one complete configured origin returned.
            # Fail closed when durable launch membership is missing or has an
            # unexpected shape; otherwise repair traffic can make screening
            # appear remotely complete before any primary origin has settled.
            if (
                correlated_launch is None
                or correlated_launch[1].get("stage") != "sample.plan"
                or member_digests is None
            ):
                continue
            if len(member_digests) < cells_per_origin:
                completed_repair_waves[terminal_key] = terminal_event
                continue
            if len(member_digests) != cells_per_origin:
                continue
            origin_key = ("members", member_digests)
        elif member_digests is not None:
            origin_key = ("members", member_digests)
        elif legacy_key is not None:
            origin_key = ("legacy", legacy_key)
        else:
            origin_key = ("terminal", terminal_key)
        previous = completed_origins.get(origin_key)
        if previous is None or int(terminal_event.seq) > int(previous[0].seq):
            completed_origins[origin_key] = (
                terminal_event,
                member_digests,
                legacy_key,
            )

    completed_launches = [
        (member_digests, legacy_key)
        for _, member_digests, legacy_key in completed_origins.values()
    ]
    completed_terminal_events = [
        event for event, _, _ in completed_origins.values()
    ]

    total = len(candidates) * _TWO_STAGE_SCREENING_ORIGINS
    completed = min(total, len(completed_origins))
    if reflection_enabled:
        failed = sum(
            event.payload.get("structured", {}).get("outcome_class") == "failed"
            for event in completed_terminal_events
        )
        succeeded = sum(
            event.payload.get("structured", {}).get("outcome_class")
            in {"improved", "degraded", "neutral"}
            for event in completed_terminal_events
        )
    else:
        # Under aggregate post-score reflection, a durable Planner result is
        # the terminal remote stage for one complete forecast origin.
        succeeded = completed
        failed = 0
    remaining = max(0, total - completed)
    configured_concurrency = metadata.get(
        "sample_concurrency",
        HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK,
    )
    if (
        isinstance(configured_concurrency, bool)
        or not isinstance(configured_concurrency, int)
        or not 1 <= configured_concurrency <= MAX_SAMPLE_CONCURRENCY
    ):
        configured_concurrency = None
    active_origins: set[
        tuple[str, frozenset[str] | tuple[str, str] | str]
    ] = set()
    active_repair_waves: set[str] = set()
    outstanding_origins: set[
        tuple[str, frozenset[str] | tuple[str, str] | str]
    ] = set()
    for _, launch in latest_launch_by_key.values():
        launch_key = str(launch.get("idempotency_key") or "").strip()
        reservation_id = str(launch.get("reservation_id") or "").strip()
        if (
            reservation_id in accepted_reservations
            or reservation_id in failed_reservations
        ):
            continue
        member_digests = _sample_member_digest_set(
            launch.get("sample_member_digests")
        )
        legacy_key = _legacy_sample_launch_key(launch.get("idempotency_key"))
        is_aggregate_repair = (
            not reflection_enabled
            and launch.get("stage") == "sample.plan"
            and member_digests is not None
            and len(member_digests) < cells_per_origin
        )
        if is_aggregate_repair:
            repair_key = reservation_id or launch_key
            active_repair_waves.add(repair_key)
            # This is still a real admitted provider request, so retain it in
            # generic request activity while keeping it out of origin progress.
            active_origins.add(("repair", repair_key))
            continue
        if member_digests is not None:
            active_key = ("members", member_digests)
        elif legacy_key is not None:
            active_key = ("legacy", legacy_key)
        else:
            active_key = ("launch", reservation_id or launch_key)
        completed_by_terminal_reflection = reflection_enabled and any(
            (
                (
                    member_digests is not None
                    and completed_members is not None
                    and member_digests <= completed_members
                )
                or (
                    legacy_key is not None
                    and completed_legacy_key is not None
                    and legacy_key == completed_legacy_key
                )
            )
            for completed_members, completed_legacy_key in completed_launches
        )
        if not completed_by_terminal_reflection:
            active_origins.add(active_key)
        if (
            not reflection_enabled
            and launch.get("stage") != terminal_stage
        ):
            # Aggregate post-score reflection declares sample.plan as the
            # completed origin boundary. A later/parallel critic does not
            # advance forecast progress, but it remains real remote activity
            # that can block host settlement and must stay observable.
            continue
        if (
            not reflection_enabled
            and (
                member_digests is None
                or len(member_digests) != cells_per_origin
            )
        ):
            # Only a complete primary wave reserves one origin slot. Unknown,
            # malformed, and sparse repair launches remain request evidence but
            # must not reduce the number of origins awaiting submission.
            continue
        if launch_key in completed_terminals:
            continue
        if any(
            (
                (
                    member_digests is not None
                    and completed_members is not None
                    and member_digests <= completed_members
                )
                or (
                    legacy_key is not None
                    and completed_legacy_key is not None
                    and legacy_key == completed_legacy_key
                )
            )
            for completed_members, completed_legacy_key in completed_launches
        ):
            continue
        if member_digests is not None:
            origin_key = ("members", member_digests)
        elif legacy_key is not None:
            origin_key = ("legacy", legacy_key)
        else:
            origin_key = ("launch", reservation_id or launch_key)
        outstanding_origins.add(origin_key)
    outstanding = min(remaining, len(outstanding_origins))
    in_flight = min(
        len(active_origins),
        configured_concurrency
        if configured_concurrency is not None
        else HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK,
    )
    provider_queued = max(0, len(active_origins) - in_flight)
    awaiting_submission = remaining - outstanding
    terminal_events = sorted(
        completed_terminal_events, key=lambda event: int(event.seq)
    )
    samples_per_minute = None
    if len(terminal_events) >= 2:
        started = datetime.fromisoformat(terminal_events[0].created_at)
        ended = datetime.fromisoformat(terminal_events[-1].created_at)
        elapsed_minutes = max(0.0, (ended - started).total_seconds() / 60.0)
        if elapsed_minutes > 0:
            samples_per_minute = round(
                (len(terminal_events) - 1) / elapsed_minutes, 3
            )
    estimated_remaining_seconds = (
        round(60.0 * remaining / samples_per_minute)
        if samples_per_minute is not None and samples_per_minute > 0
        else None
    )
    latest = max(sample_events, key=lambda event: int(event.seq))
    return {
        "schema_version": "ecologyrsi-dsh.screening-progress/1",
        "evaluation_phase": "screening",
        "progress_kind": "waiting" if in_flight else "completed_batch",
        "role": "planner",
        "model_id": None,
        "batch_index": completed,
        "batch_count": total,
        "batch_size": 1 if completed else 0,
        "completed_samples": completed,
        "remote_completed_origins": completed,
        "total_samples": total,
        "succeeded_samples": succeeded,
        "failed_samples": failed,
        "gateway_request_count": launch_count,
        "primary_gateway_request_count": primary_launch_count,
        "repair_gateway_request_count": repair_launch_count,
        "completed_repair_waves": len(completed_repair_waves),
        "active_repair_waves": len(active_repair_waves),
        "adaptive_split_trigger_count": 0,
        "adaptive_split_count": 0,
        "adaptive_split_max_depth": 0,
        "adaptive_split_recovered_samples": 0,
        "adaptive_split_failed_samples": 0,
        "causal_wave_sample_count": 1,
        "in_flight_batches": in_flight,
        "queued_batches": provider_queued,
        "awaiting_submission_batches": awaiting_submission,
        "queue_semantics": "provider_admission_only",
        "configured_concurrency": configured_concurrency,
        "samples_per_minute": samples_per_minute,
        "gateway_calls_per_minute": None,
        "estimated_remaining_seconds": estimated_remaining_seconds,
        "progress_percent": round(100.0 * completed / max(1, total), 1),
        "updated_at": latest.created_at,
        "event_seq": latest.seq,
        "evidence_source": "durable_dsh_screening_child_events",
    }


def _dsh_activity_projection(
    state: Any,
    *,
    current_stage: str | None,
    run_status: str,
) -> dict[str, Any] | None:
    """Project bounded DSH child activity without inventing model progress."""

    if run_status != "running" or current_stage not in {
        "search",
        "research",
        "proposal",
        "evaluation",
        "judge",
        "reflection",
    }:
        return None
    stage_started = next(
        (
            event
            for event in reversed(state.events)
            if event.kind == "EvolutionStageRecorded"
            and event.payload.get("stage") == current_stage
            and str(event.payload.get("status") or "").lower()
            in {"started", "running"}
        ),
        None,
    )
    stage_seq = int(getattr(stage_started, "seq", 0) or 0)
    control_seq = max(
        (
            int(event.seq)
            for event in state.events
            if event.kind in {"RunPaused", "RunResumed", "RunCancelled"}
        ),
        default=0,
    )
    activity_floor_seq = max(stage_seq, control_seq)
    launches: list[tuple[Any, Mapping[str, Any]]] = []
    for event in state.events:
        if (
            event.kind != "DshChildLaunchReserved"
            or int(event.seq) < activity_floor_seq
        ):
            continue
        launch = event.payload.get("launch")
        if not isinstance(launch, Mapping):
            continue
        if _dsh_evolution_stage(launch.get("stage")) == current_stage:
            launches.append((event, launch))
    terminal_reservations = {
        str(identity.get("child_reservation_id") or "").strip()
        for event in state.events
        if event.kind in {"DshStructuredResultAccepted", "DshChildExecutionFailed"}
        and isinstance((identity := event.payload.get("identity")), Mapping)
        and str(identity.get("child_reservation_id") or "").strip()
    }
    unresolved_launches = [
        (event, launch)
        for event, launch in launches
        if str(launch.get("reservation_id") or "").strip()
        not in terminal_reservations
    ]
    if stage_started is None and not unresolved_launches:
        return None
    if not launches:
        assert stage_started is not None
        return {
            "schema_version": "ecologyrsi-dsh.dsh-activity/1",
            "state": "waiting_for_model_slot",
            "evolution_stage": current_stage,
            "dsh_stage": None,
            "role": None,
            "launch_attempt": None,
            "started_at": stage_started.created_at,
            "updated_at": state.events[-1].created_at,
            "event_seq": stage_seq,
            "evidence": "append_only_stage_event",
        }

    launch_event, launch = (
        unresolved_launches[-1] if unresolved_launches else launches[-1]
    )
    reservation_id = launch.get("reservation_id")
    accepted_event = next(
        (
            event
            for event in reversed(state.events)
            if event.kind == "DshStructuredResultAccepted"
            and int(event.seq) > int(launch_event.seq)
            and isinstance(event.payload.get("identity"), Mapping)
            and event.payload["identity"].get("child_reservation_id")
            == reservation_id
        ),
        None,
    )
    attempt = int(launch.get("launch_attempt") or 1)
    if accepted_event is None:
        activity_state = "model_retry_running" if attempt > 1 else "model_running"
        updated_at = state.events[-1].created_at
        event_seq = int(launch_event.seq)
    elif int(accepted_event.seq) == int(state.events[-1].seq):
        activity_state = "host_validating"
        updated_at = accepted_event.created_at
        event_seq = int(accepted_event.seq)
    else:
        activity_state = "waiting_for_model_slot"
        updated_at = state.events[-1].created_at
        event_seq = int(accepted_event.seq)
    return {
        "schema_version": "ecologyrsi-dsh.dsh-activity/1",
        "state": activity_state,
        "evolution_stage": current_stage,
        "dsh_stage": launch.get("stage"),
        "role": launch.get("role"),
        "launch_attempt": attempt,
        "started_at": launch_event.created_at,
        "updated_at": updated_at,
        "event_seq": event_seq,
        "evidence": "append_only_dsh_child_events",
    }


def _origin_live_projection_after(
    state: Any,
    started: Any,
    origin_total: int,
) -> dict[str, Any] | None:
    """Count durable DSH origin receipts after one explicit work boundary."""

    events = tuple(getattr(state, "events", ()))
    metadata = state.task_manifest.metadata
    cells_per_origin = int(metadata.get("prediction_cells_per_origin") or 1)
    configured_concurrency = metadata.get(
        "sample_concurrency",
        HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK,
    )
    if (
        isinstance(configured_concurrency, bool)
        or not isinstance(configured_concurrency, int)
        or not 1 <= configured_concurrency <= MAX_SAMPLE_CONCURRENCY
    ):
        configured_concurrency = HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK

    launch_by_key: dict[str, tuple[Any, Mapping[str, Any]]] = {}
    latest_launch_by_key: dict[str, tuple[Any, Mapping[str, Any]]] = {}
    accepted_reservations: set[str] = set()
    failed_reservations: set[str] = set()
    completed_keys: set[str] = set()
    sample_events: list[Any] = []
    launch_count = 0
    for event in events:
        if int(event.seq) <= int(started.seq):
            continue
        if event.kind in {"RunPaused", "RunResumed"}:
            latest_launch_by_key.clear()
            continue
        if event.kind == "DshChildLaunchReserved":
            launch = event.payload.get("launch")
            if not isinstance(launch, Mapping) or launch.get("stage") != "sample.plan":
                continue
            key = str(launch.get("idempotency_key") or "").strip()
            if not key:
                continue
            launch_count += 1
            launch_by_key[key] = (event, launch)
            latest_launch_by_key[key] = (event, launch)
            reservation = str(launch.get("reservation_id") or "").strip()
            sample_events.append(event)
        elif event.kind == "DshPredictionToolExecuted":
            if event.payload.get("stage") != "sample.plan":
                continue
            key = str(event.payload.get("idempotency_key") or "").strip()
            if key:
                completed_keys.add(key)
            sample_events.append(event)
        elif event.kind == "DshStructuredResultAccepted":
            identity = event.payload.get("identity")
            if not isinstance(identity, Mapping) or identity.get("stage") != "sample.plan":
                continue
            key = str(identity.get("idempotency_key") or "").strip()
            if key:
                completed_keys.add(key)
            reservation = str(identity.get("child_reservation_id") or "").strip()
            if reservation:
                accepted_reservations.add(reservation)
            sample_events.append(event)
        elif event.kind == "DshChildExecutionFailed":
            identity = event.payload.get("identity")
            if not isinstance(identity, Mapping) or identity.get("stage") != "sample.plan":
                continue
            reservation = str(identity.get("child_reservation_id") or "").strip()
            if reservation:
                failed_reservations.add(reservation)
            sample_events.append(event)
    if not sample_events:
        return None

    source_origins = {
        members
        for _, launch in launch_by_key.values()
        if (members := _sample_member_digest_set(launch.get("sample_member_digests")))
        is not None
        and len(members) == cells_per_origin
    }
    completed_origins: set[frozenset[str]] = set()
    for key in completed_keys:
        correlated = launch_by_key.get(key)
        if correlated is None:
            continue
        members = _sample_member_digest_set(
            correlated[1].get("sample_member_digests")
        )
        if members is not None and len(members) == cells_per_origin:
            completed_origins.add(members)

    active_origins: set[frozenset[str]] = set()
    for _, launch in latest_launch_by_key.values():
        reservation = str(launch.get("reservation_id") or "").strip()
        if reservation in accepted_reservations or reservation in failed_reservations:
            continue
        members = _sample_member_digest_set(launch.get("sample_member_digests"))
        if members is None:
            continue
        source = next(
            (origin for origin in source_origins if members <= origin),
            members,
        )
        active_origins.add(source)

    remotely_completed = min(origin_total, len(completed_origins))
    in_flight = min(len(active_origins), configured_concurrency)
    queued = max(0, len(active_origins) - in_flight)
    submitted = min(origin_total, len(source_origins))
    latest = max(sample_events, key=lambda event: int(event.seq))
    return {
        "completed_origins": remotely_completed,
        "progress_kind": "settling" if remotely_completed else "waiting",
        "in_flight_batches": in_flight,
        "queued_batches": queued,
        "awaiting_submission_batches": max(0, origin_total - submitted),
        "awaiting_settlement_batches": remotely_completed,
        "gateway_request_count": launch_count,
        "configured_concurrency": configured_concurrency,
        "queue_semantics": "provider_admission_only",
        "updated_at": latest.created_at,
        "event_seq": latest.seq,
    }


def _formal_batch_live_projection(state: Any) -> dict[str, Any] | None:
    """Project the active formal batch from durable DSH origin events."""

    status = getattr(getattr(state.run, "status", None), "value", None)
    if status not in {None, "running"}:
        return None
    events = tuple(getattr(state, "events", ()))
    if not events:
        return None
    generation = int(state.run.generation)
    started = next(
        (
            event
            for event in reversed(events)
            if event.kind == "FormalBatchStarted"
            and isinstance(event.payload.get("batch"), Mapping)
            and int(event.payload["batch"].get("generation", -1)) == generation
        ),
        None,
    )
    if started is None:
        return None
    batch = started.payload["batch"]
    candidate_id = str(batch.get("candidate_id") or "")
    batch_index = int(batch.get("batch_index", -1))
    cohort_digest = str(batch.get("cohort_digest") or "")
    if any(
        item.scope.generation == generation
        and item.scope.candidate_id == candidate_id
        and item.scope.batch_index == batch_index
        and item.scope.cohort_digest == cohort_digest
        for item in state.formal_batch_evaluations
    ):
        return None
    return _origin_live_projection_after(
        state,
        started,
        max(1, int(batch.get("origin_count") or 0)),
    )


def _holdout_arm_live_projection(state: Any) -> dict[str, Any] | None:
    """Project the currently evaluating 169-origin holdout arm."""

    status = getattr(getattr(state.run, "status", None), "value", None)
    if status not in {None, "running"}:
        return None
    generation = int(state.run.generation)
    started = next(
        (
            event
            for event in reversed(tuple(getattr(state, "events", ())))
            if event.kind == "HoldoutArmStarted"
            and int(event.payload.get("generation", -1)) == generation
        ),
        None,
    )
    if started is None:
        return None
    arm = str(started.payload.get("holdout_arm") or "")
    if any(
        item.scope.generation == generation
        and getattr(item.scope.holdout_arm, "value", None) == arm
        for item in state.holdout_evaluations
    ):
        return None
    live = _origin_live_projection_after(
        state,
        started,
        max(1, int(started.payload.get("origin_count") or 0)),
    )
    if live is not None:
        live["holdout_arm"] = arm
        live["current_candidate_id"] = started.payload.get("candidate_id")
    return live


def _adaptive_progress_projection(
    state: Any,
    admission_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Project origin-level progress for the Top-2 adaptive protocol.

    The legacy six-stage bar has no representation for ten 50-origin batches,
    so it can sit at a misleading percentage while the formal lane is active.
    This projection counts only durable cohort/batch boundaries and therefore
    remains monotonic across a process restart.
    """

    metadata = state.task_manifest.metadata
    if metadata.get("optimization_protocol") != "top2_adaptive_epoch@1":
        return None
    schedule = metadata.get("optimization_schedule")
    if not isinstance(schedule, Mapping):
        return None
    try:
        screening_total = 4 * int(schedule["screening_origin_count"])
        formal_total = 2 * int(schedule["formal_origin_count_per_finalist"])
        holdout_total = 3 * int(schedule["selection_holdout_origin_count"])
    except (KeyError, TypeError, ValueError):
        return None
    generation = state.run.generation
    settled_screening_completed = sum(
        int(event.payload.get("origin_count") or 0)
        for event in state.candidate_screening_events
        if int(event.payload.get("generation", -1)) == generation
    )
    # Completed progress is the host-settled candidate boundary. A predictor
    # receipt can be followed by schema repair, optional critic work, or host
    # scoring, so counting child receipts here can reach 256 while the
    # screening workers are still active.
    screening_completed = settled_screening_completed
    live_screening = _screening_progress_projection(state)
    has_adaptive_boundary = (
        live_screening is not None
        or any(
            int(event.payload.get("generation", -1)) == generation
            for event in state.candidate_screening_events
        )
        or any(
            item.scope.generation == generation
            for item in state.formal_batch_evaluations
        )
        or any(
            item.scope.generation == generation
            for item in state.holdout_evaluations
        )
        or any(
            event.kind == "HoldoutArmStarted"
            and int(event.payload.get("generation", -1)) == generation
            for event in state.events
        )
        or any(batch.generation == generation for batch in state.formal_batches)
    )
    if not has_adaptive_boundary:
        # The schedule exists from run creation, but it is not progress
        # evidence.  Before screening cohorts start, preserve the durable
        # search/research/proposal stage instead of projecting a synthetic
        # zero-percent evaluation phase.
        return None
    formal_completed = sum(
        int(item.scope.origin_count)
        for item in state.formal_batch_evaluations
        if item.scope.generation == generation
    )
    holdout_completed = sum(
        int(item.scope.origin_count)
        for item in state.holdout_evaluations
        if item.scope.generation == generation
    )
    total = screening_total + formal_total + holdout_total
    completed = min(
        total,
        screening_completed + formal_completed + holdout_completed,
    )
    throughput_rows: list[tuple[datetime, int]] = []
    for event in state.candidate_screening_events:
        if int(event.payload.get("generation", -1)) != generation:
            continue
        try:
            throughput_rows.append(
                (
                    datetime.fromisoformat(
                        str(getattr(event, "created_at", "")).replace(
                            "Z", "+00:00"
                        )
                    ),
                    int(event.payload.get("origin_count") or 0),
                )
            )
        except (TypeError, ValueError):
            continue
    for item in (*state.formal_batch_evaluations, *state.holdout_evaluations):
        if item.scope.generation != generation:
            continue
        try:
            throughput_rows.append(
                (
                    datetime.fromisoformat(
                        str(getattr(item, "created_at", "")).replace(
                            "Z", "+00:00"
                        )
                    ),
                    int(item.scope.origin_count),
                )
            )
        except (TypeError, ValueError):
            continue
    throughput_rows.sort(key=lambda row: row[0])
    rolling_rate: float | None = None
    rolling_eta: int | None = None
    if len(throughput_rows) >= 2:
        window = throughput_rows[-8:]
        latest_time = window[-1][0]
        now = datetime.now(latest_time.tzinfo)
        elapsed_minutes = max(
            1.0 / 60.0, (max(now, latest_time) - window[0][0]).total_seconds() / 60.0
        )
        rolling_rate = round(sum(count for _at, count in window[1:]) / elapsed_minutes, 3)
        if rolling_rate > 0:
            rolling_eta = int(math.ceil(max(0, total - completed) / rolling_rate * 60.0))
    phase = "screening"
    if settled_screening_completed >= screening_total:
        phase = "formal_batch"
    if formal_completed >= formal_total:
        phase = "holdout"
    if holdout_completed >= holdout_total:
        phase = "decision"
    batch_count = max(1, int(schedule.get("formal_origin_count_per_finalist", 500))) // max(
        1, int(schedule.get("local_batch_origin_count", 50))
    )
    active_batch = None
    active_candidate = None
    for batch in reversed(state.formal_batches):
        if batch.generation != generation:
            continue
        active_batch = batch.batch_index + 1
        active_candidate = batch.candidate_id
        break
    if phase != "formal_batch":
        active_batch = None
    live_fields: dict[str, Any] = {}
    live_holdout_completed = 0
    if live_screening is not None and phase == "screening":
        # Child events remain useful activity evidence, but cannot increase
        # host-settled origin progress. Keep submitted-but-unsettled origins
        # separate from origins that have not reached the gateway yet; calling
        # every remaining origin "awaiting settlement" hides real throughput.
        for key in (
            "in_flight_batches",
            "queued_batches",
            "configured_concurrency",
            "queue_semantics",
            "gateway_request_count",
            "primary_gateway_request_count",
            "repair_gateway_request_count",
            "remote_completed_origins",
            "completed_repair_waves",
            "active_repair_waves",
            "updated_at",
            "event_seq",
        ):
            if key in live_screening:
                live_fields[key] = live_screening[key]
        in_flight = max(0, int(live_fields.get("in_flight_batches") or 0))
        queued = max(0, int(live_fields.get("queued_batches") or 0))
        remotely_completed = max(
            0, int(live_screening.get("remote_completed_origins") or 0)
        )
        awaiting_settlement = max(
            0, remotely_completed - settled_screening_completed
        )
        awaiting_submission = max(
            0, int(live_screening.get("awaiting_submission_batches") or 0)
        )
        live_fields.update(
            {
                "progress_kind": (
                    "settling" if awaiting_settlement else "waiting"
                ),
                "settled_origins": settled_screening_completed,
                "awaiting_settlement_batches": awaiting_settlement,
                "awaiting_submission_batches": awaiting_submission,
                "samples_per_minute": None,
                "estimated_remaining_seconds": None,
            }
        )
    elif phase == "formal_batch":
        live_formal = _formal_batch_live_projection(state)
        if live_formal is not None:
            remote_completed = int(live_formal.pop("completed_origins"))
            completed = min(total, completed + remote_completed)
            live_fields.update(live_formal)
            live_fields.update(
                {
                    "settled_origins": (
                        screening_completed
                        + formal_completed
                        + holdout_completed
                    ),
                    "awaiting_settlement_batches": remote_completed,
                    "samples_per_minute": None,
                    "estimated_remaining_seconds": None,
                }
            )
    elif phase == "holdout":
        live_holdout = _holdout_arm_live_projection(state)
        if live_holdout is not None:
            live_holdout_completed = int(
                live_holdout.pop("completed_origins")
            )
            completed = min(total, completed + live_holdout_completed)
            live_fields.update(live_holdout)
            live_fields.update(
                {
                    "settled_origins": (
                        screening_completed
                        + formal_completed
                        + holdout_completed
                    ),
                    "samples_per_minute": None,
                    "estimated_remaining_seconds": None,
                }
            )
    if "in_flight_batches" in live_fields:
        live_fields["in_flight_requests"] = live_fields["in_flight_batches"]
    if "queued_batches" in live_fields:
        live_fields["provider_queued_requests"] = live_fields["queued_batches"]
    live_fields["samples_per_minute"] = rolling_rate
    live_fields["estimated_remaining_seconds"] = rolling_eta
    live_fields["throughput_semantics"] = "host_settled_origins_rolling_8_boundaries"
    if isinstance(admission_snapshot, Mapping):
        live_fields.update(
            {
                "admission_limit": admission_snapshot.get("limit"),
                "adaptive_admission_limit": admission_snapshot.get(
                    "adaptive_limit"
                ),
                "admission_active": admission_snapshot.get("active"),
                "admission_waiting": admission_snapshot.get("waiting"),
                "admission_congestion_events": admission_snapshot.get(
                    "congestion_events"
                ),
                "admission_semantics": "host_origin_admission_live_snapshot",
            }
        )
    epoch_progress_percent = round(100.0 * completed / max(1, total), 1)
    return {
        "schema_version": "ecologyrsi-dsh.adaptive-progress/2",
        "evaluation_phase": phase,
        "completed_origins": completed,
        "total_origins": total,
        "completed_samples": completed,
        "total_samples": total,
        "progress_percent": epoch_progress_percent,
        "epoch_progress_percent": epoch_progress_percent,
        "screening_completed_origins": min(screening_completed, screening_total),
        "screening_total_origins": screening_total,
        "formal_completed_origins": min(formal_completed, formal_total),
        "formal_total_origins": formal_total,
        "holdout_completed_origins": min(
            holdout_completed + live_holdout_completed,
            holdout_total,
        ),
        "holdout_total_origins": holdout_total,
        # The existing browser progress renderer consumes the generic
        # ``batch_index``/``batch_count`` pair.  Keep the human-facing index
        # one-based here while durable FormalBatch state remains zero-based.
        "batch_index": active_batch,
        "current_batch": active_batch,
        "batch_count": batch_count if phase == "formal_batch" else None,
        "current_candidate_id": active_candidate,
        "evidence": "durable_adaptive_cohort_and_trajectory_events",
        **live_fields,
    }


def _adaptive_trajectory_projection(state: Any) -> list[dict[str, Any]]:
    """Expose bounded, truthful formal-batch evidence for the process UI.

    Each batch uses a different cohort window.  Scores are therefore retained
    as diagnostics and explicitly marked non-comparable; promotion remains a
    separate same-cohort holdout decision.
    """

    proposals = {
        (str(item.get("candidate_id")), int(item.get("batch_index"))): item
        for item in state.local_edit_proposals
        if isinstance(item, Mapping)
        and isinstance(item.get("candidate_id"), str)
        and isinstance(item.get("batch_index"), int)
    }
    outcomes = {
        (str(item.get("candidate_id")), int(item.get("batch_index"))): item
        for item in state.local_edit_outcomes
        if isinstance(item, Mapping)
        and isinstance(item.get("candidate_id"), str)
        and isinstance(item.get("batch_index"), int)
    }
    activations = {
        (item.candidate_id, item.batch_index): item
        for item in state.trajectory_revision_activations
    }
    batches_by_candidate: dict[str, list[Any]] = {}
    for batch in state.formal_batches:
        batches_by_candidate.setdefault(batch.candidate_id, []).append(batch)

    lanes: list[dict[str, Any]] = []
    for trajectory in sorted(
        state.formal_trajectories,
        key=lambda item: (item.generation, item.candidate_id),
    ):
        rows: list[dict[str, Any]] = []
        batches = sorted(
            batches_by_candidate.get(trajectory.candidate_id, ()),
            key=lambda item: item.batch_index,
        )
        for batch in batches:
            key = (batch.candidate_id, batch.batch_index)
            evaluation = state.batch_evaluation_for(*key)
            proposal = proposals.get(key)
            outcome = outcomes.get(key)
            activation = activations.get(key)
            activation_reason = getattr(activation, "reason", None)
            activation_reason_value = getattr(
                activation_reason, "value", activation_reason
            )
            metrics = evaluation.metrics if evaluation is not None else {}
            sample = metrics.get("sample_execution") if isinstance(metrics, Mapping) else None
            if not isinstance(sample, Mapping):
                sample = {}
            operations: list[dict[str, Any]] = []
            proposal_detail = proposal.get("proposal", proposal) if proposal else {}
            if isinstance(proposal_detail, Mapping) and isinstance(proposal_detail.get("operations"), (list, tuple)):
                for operation in proposal_detail["operations"][:5]:
                    if not isinstance(operation, Mapping):
                        continue
                    operations.append(
                        {
                            name: sanitize_public_value(operation.get(name))
                            for name in ("op", "name", "value", "target", "program_id")
                            if name in operation
                        }
                    )
            coverage = _finite_number(
                sample.get("coverage", metrics.get("sample_execution_coverage"))
            )
            attempted_origins = _finite_number(
                sample.get("attempted_origin_samples")
            )
            succeeded_origins = _finite_number(
                sample.get("succeeded_origin_samples")
            )
            origin_success_rate = (
                min(1.0, max(0.0, succeeded_origins / attempted_origins))
                if attempted_origins is not None
                and attempted_origins > 0
                and succeeded_origins is not None
                else None
            )
            rows.append(
                {
                    "batch_index": batch.batch_index + 1,
                    "batch_count": batch.batch_count,
                    "origin_count": batch.origin_count,
                    "candidate_revision_id": batch.revision_id,
                    "cohort_digest": batch.cohort_digest,
                    "status": (
                        "rolled_back"
                        if activation is not None
                        and activation_reason_value == "prequential_safety_rollback"
                        else "safety_kept"
                        if outcome is not None and outcome.get("reason")
                        else
                        "edited"
                        if activation is not None
                        else "evaluated"
                        if evaluation is not None
                        else "running"
                    ),
                    "score": evaluation.score if evaluation is not None else None,
                    "passed": evaluation.passed if evaluation is not None else None,
                    "coverage": coverage,
                    "prediction_cell_coverage": coverage,
                    "origin_success_rate": origin_success_rate,
                    "coverage_pass": (
                        metrics.get("sample_execution_coverage_pass")
                        if isinstance(metrics, Mapping)
                        else None
                    ),
                    "succeeded_origins": succeeded_origins,
                    "failed_origins": _finite_number(
                        sample.get("failed_origin_samples")
                    ),
                    "failed_scoring_cells": _finite_number(
                        sample.get("failed_examples")
                    ),
                    "fallback_scoring_cells": _finite_number(
                        sample.get("scoring_fallback_examples")
                    ),
                    "edit_decision": proposal_detail.get("decision") if isinstance(proposal_detail, Mapping) else None,
                    "edit_outcome": outcome.get("outcome") if outcome else None,
                    "edit_reason": outcome.get("reason") if outcome else proposal.get("safety_reason") if proposal else None,
                    "operations": operations,
                    "active_revision_id": (
                        activation.to_revision_id
                        if activation is not None
                        else outcome.get("active_revision_id")
                        if outcome is not None
                        else batch.revision_id
                    ),
                    "created_at": (
                        activation.created_at
                        if activation is not None
                        else evaluation.created_at
                        if evaluation is not None
                        else batch.created_at
                    ),
                    "score_comparability": "different_batch_cohort_diagnostic_only",
                }
            )
        lanes.append(
            {
                "candidate_id": trajectory.candidate_id,
                "generation": trajectory.generation,
                "status": trajectory.status.value,
                "initial_revision_id": trajectory.initial_revision_id,
                "final_revision_id": trajectory.final_revision_id,
                "batch_count": trajectory.batch_count,
                "completed_batch_count": sum(
                    row["edit_outcome"] is not None for row in rows
                ),
                "batches": rows,
                "score_comparability": "different_batch_cohort_diagnostic_only",
            }
        )
    return lanes


def _run_execution_progress(
    state: Any,
    admission_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
            "screened_out",
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
    if stage_progress is None:
        stage_progress = _screening_progress_projection(state)
        if stage_progress is not None:
            # Two-stage screening runs before formal Artifact/Evaluation rows
            # exist, but durable sample child events prove that evaluation is
            # active.  Do not mislabel this interval as a scheduler queue.
            current_stage = "evaluation"
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

    adaptive_progress = _adaptive_progress_projection(state, admission_snapshot)
    epoch_progress_percent: float | None = None
    if adaptive_progress is not None:
        stage_progress = adaptive_progress
        current_stage = (
            "evaluation"
            if adaptive_progress["evaluation_phase"]
            in {"screening", "formal_batch", "holdout"}
            else adaptive_progress["evaluation_phase"]
        )
        active_candidate_id = (
            adaptive_progress.get("current_candidate_id") or active_candidate_id
        )
        epoch_progress_percent = float(adaptive_progress["epoch_progress_percent"])
        progress_percent = round(
            min(
                100.0,
                100.0
                * (
                    min(state.run.generation, total_generations)
                    + epoch_progress_percent / 100.0
                )
                / total_generations,
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

    dsh_activity = _dsh_activity_projection(
        state,
        current_stage=current_stage,
        run_status=status,
    )

    return {
        "schema_version": "ecologyrsi-dsh.execution-progress/2",
        "status": status,
        "phase": phase,
        "progress_percent": progress_percent,
        "overall_progress_percent": progress_percent,
        "epoch_progress_percent": epoch_progress_percent,
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
        "dsh_activity": dsh_activity,
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


def _run_wide_token_usage(
    task: Any,
    metadata: Mapping[str, Any],
    model_usage: Mapping[str, Any],
    dsh_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the frozen accounting source used by the public run budget."""

    limit = _budget_value(task, "token_limit", 0)
    provider = dsh_runtime.get("provider_usage")
    provider_available = (
        isinstance(provider, Mapping) and provider.get("available") is True
    )
    if provider_available:
        available = True
        used = int(provider.get("total_tokens", 0)) if available else 0
        source = "dsh_provider_reported_sessions"
    else:
        available = model_usage.get("available") is True
        used = int(
            model_usage.get(
                "budget_accounted_tokens", model_usage.get("total_tokens", 0)
            )
        )
        source = "model_usage_receipts"
    ratio = (used / limit) if limit > 0 else None
    return {
        "available": available,
        "source": source,
        "tokens_used": used,
        "token_limit": limit,
        "remaining_tokens": max(0, limit - used) if limit > 0 else None,
        "utilization_ratio": round(ratio, 6) if ratio is not None else None,
        "warning": (
            "exhausted"
            if ratio is not None and ratio >= 1.0
            else "approaching_limit"
            if ratio is not None and ratio >= 0.8
            else None
        ),
        "enforcement": None,
        "scope": _token_budget_scope(task, metadata),
    }


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
    structured_events = [
        event
        for event in state.events
        if event.kind == "DshStructuredResultAccepted"
    ]
    skill_evidence = [
        event.payload.get("skill_invocation_evidence")
        for event in structured_events
        if isinstance(event.payload, Mapping)
        and isinstance(event.payload.get("skill_invocation_evidence"), Mapping)
    ]
    first_call_verified = bool(structured_events) and (
        len(skill_evidence) == len(structured_events)
        and all(
            item.get("first_tool_call_verified") is True
            and item.get("order_verified") is True
            for item in skill_evidence
        )
    )
    if bound is None:
        return {
            "execution_protocol": "legacy_read_only",
            "native": False,
            "capability_verified": False,
            "first_call_verified": False,
            "skill_invocation": {
                "required": False,
                "all_verified": False,
                "verified_call_count": 0,
                "stages": [],
                "skills": [],
            },
            "context_pressure": {"available": False, "source": "not_dsh_native"},
            "provider_usage": {"available": False, "source": "not_dsh_native"},
            "retrieval": {
                "available": False,
                "call_count": 0,
                "fallback_count": 0,
                "routes": {},
                "stages": [],
                "roles": [],
                "result_digests": [],
            },
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
    retrieval_events = [
        event
        for event in state.events
        if event.kind == "DshRetrievalExecuted"
        and isinstance(event.payload, Mapping)
    ]
    route_counts: dict[str, int] = {}
    fallback_count = 0
    stages: set[str] = set()
    roles: set[str] = set()
    result_digests: list[str] = []
    for event in retrieval_events:
        route = event.payload.get("provider_route")
        if isinstance(route, str) and route:
            route_counts[route] = route_counts.get(route, 0) + 1
        if event.payload.get("fallback_reason") is not None:
            fallback_count += 1
        identity = event.payload.get("identity")
        if isinstance(identity, Mapping):
            if isinstance(identity.get("stage"), str) and identity["stage"]:
                stages.add(identity["stage"])
            if isinstance(identity.get("role"), str) and identity["role"]:
                roles.add(identity["role"])
        result_digest = event.payload.get("result_digest")
        if (
            isinstance(result_digest, str)
            and len(result_digest) == 64
            and all(character in "0123456789abcdef" for character in result_digest)
        ):
            result_digests.append(result_digest)
    retrieval = {
        "available": True,
        "call_count": len(retrieval_events),
        "fallback_count": fallback_count,
        "routes": dict(sorted(route_counts.items())),
        "stages": sorted(stages),
        "roles": sorted(roles),
        # Keep the public read model bounded even for long-running evolution.
        "result_digests": result_digests[-64:],
    }
    return {
        "execution_protocol": bound.payload["execution_protocol"],
        "native": True,
        "capability_verified": True,
        "capabilities_digest": bound.payload["capabilities_digest"],
        "preset_ids": list(bound.payload["preset_ids"]),
        "root_session_id": state.run.session_id,
        "first_call_verified": first_call_verified,
        "skill_invocation": {
            "required": True,
            "all_verified": first_call_verified,
            "verified_call_count": len(skill_evidence),
            "stages": sorted(
                {str(item.get("stage")) for item in skill_evidence}
            ),
            "skills": sorted(
                {str(item.get("skill_name")) for item in skill_evidence}
            ),
        },
        # DSH reports the TokenMeter pressure and Session projection usage as
        # distinct Host-owned measurements. Python never infers one from the
        # other and never relabels legacy ModelUsageRecorded receipts as DSH.
        "context_pressure": context_pressure,
        "provider_usage": provider_usage,
        "retrieval": retrieval,
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


def _projection_json(
    state: Any,
    admission_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
        "optimization_protocol": metadata.get("optimization_protocol"),
        "optimization_schedule": metadata.get("optimization_schedule"),
        "derived_execution_budget": metadata.get("derived_execution_budget"),
        "derived_run_execution_budget": metadata.get(
            "derived_run_execution_budget"
        ),
        "cohort_capacity_report": metadata.get("cohort_capacity_report"),
        "cohort_capacity_enforced": metadata.get("cohort_capacity_enforced"),
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
        "fitness_profile": metadata.get("fitness_profile"),
        "fitness_profile_digest": metadata.get("fitness_profile_digest"),
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
        "minimum_selection_samples_per_update": metadata.get(
            "minimum_selection_samples_per_update"
        ),
        "minimum_selection_origin_samples_per_update": metadata.get(
            "minimum_selection_origin_samples_per_update"
        ),
        "prediction_cells_per_origin": metadata.get(
            "prediction_cells_per_origin"
        ),
        "sample_agent_protocol": metadata.get("sample_agent_protocol"),
        "sample_budget_class": metadata.get("sample_budget_class"),
        "sample_concurrency": metadata.get("sample_concurrency", 4),
        "candidate_concurrency": metadata.get("candidate_concurrency"),
        "two_stage_evaluation_enabled": metadata.get(
            "two_stage_evaluation_enabled", True
        ),
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
        "optimization_protocol": metadata.get("optimization_protocol"),
        "optimization_schedule": metadata.get("optimization_schedule"),
        "cohort_capacity_report": metadata.get("cohort_capacity_report"),
        "cohort_capacity_enforced": metadata.get("cohort_capacity_enforced"),
        "candidates_per_round": task.candidates_per_generation,
        "variants_per_round": task.candidates_per_generation,
        "knowledge_online_enabled": bool(metadata.get("knowledge_online_enabled", False)),
    }
    execution_progress = _run_execution_progress(state, admission_snapshot)
    execution_diagnostics = _execution_diagnostics(state)
    model_usage = _model_usage_summary(state)
    dsh_runtime = _dsh_runtime_projection(state)
    run_wide_usage = _run_wide_token_usage(
        task, metadata, model_usage, dsh_runtime
    )
    pause_reason, pause_code, retry_circuit = _run_pause_projection(state)
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
        "retry_circuit": retry_circuit,
        "created_at": run.created_at,
        "updated_at": latest_event,
        "generation": run.generation,
        "total_generations": _max_generations(task),
        "candidates_count": len(state.candidates),
        "max_candidates": task.max_candidates,
        "candidates_per_generation": task.candidates_per_generation,
        "optimization_protocol": metadata.get("optimization_protocol"),
        "optimization_schedule": metadata.get("optimization_schedule"),
        "samples_per_update": metadata.get("samples_per_update"),
        "minimum_selection_samples_per_update": metadata.get(
            "minimum_selection_samples_per_update"
        ),
        "minimum_selection_origin_samples_per_update": metadata.get(
            "minimum_selection_origin_samples_per_update"
        ),
        "prediction_cells_per_origin": metadata.get(
            "prediction_cells_per_origin"
        ),
        "sample_agent_protocol": metadata.get("sample_agent_protocol"),
        "sample_budget_class": metadata.get("sample_budget_class"),
        "sample_agent_batch_size": metadata.get("sample_agent_batch_size"),
        "sample_concurrency": metadata.get("sample_concurrency"),
        "candidate_concurrency": metadata.get("candidate_concurrency"),
        "two_stage_evaluation_enabled": metadata.get(
            "two_stage_evaluation_enabled", True
        ),
        "budget": dict(task.budget),
        "token_usage_available": run_wide_usage["available"],
        "tokens_used": run_wide_usage["tokens_used"],
        "token_limit": _budget_value(task, "token_limit", 0),
        "token_reservation_per_wave": _budget_value(
            task, "token_reservation_per_wave", 0
        ),
        "token_budget_scope": token_budget_scope,
        "run_wide_accounting_complete": False,
        "run_wide_usage": run_wide_usage,
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
        "adaptive_trajectories": _adaptive_trajectory_projection(state),
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
    pause_reason, pause_code, retry_circuit = _run_pause_projection(state)
    observed = best_observed_evaluation(state)
    acceptable = (
        state.evaluation_for(run.best_candidate_id)
        if run.best_candidate_id is not None
        else None
    )
    dataset_id = task.visible_datasets[0] if task.visible_datasets else None
    configuration = {
        "dataset_id": dataset_id,
        "optimization_protocol": metadata.get("optimization_protocol"),
        "optimization_schedule": metadata.get("optimization_schedule"),
        "derived_execution_budget": metadata.get("derived_execution_budget"),
        "derived_run_execution_budget": metadata.get(
            "derived_run_execution_budget"
        ),
        "cohort_capacity_report": metadata.get("cohort_capacity_report"),
        "cohort_capacity_enforced": metadata.get("cohort_capacity_enforced"),
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
        "minimum_selection_samples_per_update": metadata.get(
            "minimum_selection_samples_per_update"
        ),
        "minimum_selection_origin_samples_per_update": metadata.get(
            "minimum_selection_origin_samples_per_update"
        ),
        "prediction_cells_per_origin": metadata.get(
            "prediction_cells_per_origin"
        ),
        "sample_agent_protocol": metadata.get("sample_agent_protocol"),
        "sample_budget_class": metadata.get("sample_budget_class"),
        "sample_agent_batch_size": metadata.get("sample_agent_batch_size"),
        "sample_concurrency": metadata.get("sample_concurrency"),
        "candidate_concurrency": metadata.get("candidate_concurrency"),
        "two_stage_evaluation_enabled": metadata.get(
            "two_stage_evaluation_enabled", True
        ),
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
        "retry_circuit": retry_circuit,
        "created_at": run.created_at,
        "updated_at": latest_event,
        "projection_revision": state.events[-1].seq if state.events else 0,
        "generation": run.generation,
        "total_generations": _max_generations(task),
        "candidates_count": len(state.candidates),
        "max_candidates": task.max_candidates,
        "candidates_per_generation": task.candidates_per_generation,
        "optimization_protocol": metadata.get("optimization_protocol"),
        "optimization_schedule": metadata.get("optimization_schedule"),
        "cohort_capacity_report": metadata.get("cohort_capacity_report"),
        "cohort_capacity_enforced": metadata.get("cohort_capacity_enforced"),
        "samples_per_update": metadata.get("samples_per_update"),
        "minimum_selection_samples_per_update": metadata.get(
            "minimum_selection_samples_per_update"
        ),
        "minimum_selection_origin_samples_per_update": metadata.get(
            "minimum_selection_origin_samples_per_update"
        ),
        "prediction_cells_per_origin": metadata.get(
            "prediction_cells_per_origin"
        ),
        "sample_agent_protocol": metadata.get("sample_agent_protocol"),
        "sample_budget_class": metadata.get("sample_budget_class"),
        "sample_agent_batch_size": metadata.get("sample_agent_batch_size"),
        "sample_concurrency": metadata.get("sample_concurrency"),
        "candidate_concurrency": metadata.get("candidate_concurrency"),
        "two_stage_evaluation_enabled": metadata.get(
            "two_stage_evaluation_enabled", True
        ),
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


def _state_payload(
    state: Any,
    admission_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _assert_http_scope(state)
    return {
        "schema_version": "ecologyrsi-dsh.browser-run/3",
        "projection": _projection_json(state, admission_snapshot),
    }


def _monitor_payload(
    state: Any,
    admission_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a polling-safe projection without candidate/evidence payloads."""

    _assert_http_scope(state)
    task = state.task_manifest
    metadata = dict(task.metadata)
    outcome, termination_reason = run_completion_outcome(state)
    failure_reason, failed_stage = _run_failure_projection(state)
    pause_reason, pause_code, retry_circuit = _run_pause_projection(state)
    model_usage = _model_usage_summary(state)
    dsh_runtime = _dsh_runtime_projection(state)
    run_wide_usage = _run_wide_token_usage(
        task, metadata, model_usage, dsh_runtime
    )
    latest_at = state.events[-1].created_at if state.events else state.run.created_at
    return {
        "schema_version": "ecologyrsi-dsh.browser-run-monitor/1",
        "projection": {
            "id": state.run.run_id,
            "run_id": state.run.run_id,
            "status": state.run.status.value,
            "outcome": outcome,
            "termination_reason": termination_reason,
            "failure_reason": failure_reason,
            "failure_code": _run_failure_code(state),
            "failed_stage": failed_stage,
            "pause_reason": pause_reason,
            "pause_code": pause_code,
            "retry_circuit": retry_circuit,
            "updated_at": latest_at,
            "projection_revision": state.events[-1].seq if state.events else 0,
            "generation": state.run.generation,
            "total_generations": _max_generations(task),
            "candidates_count": len(state.candidates),
            "max_candidates": task.max_candidates,
            "execution_progress": _run_execution_progress(
                state, admission_snapshot
            ),
            # This is a bounded public aggregate (two finalist lanes per
            # generation), so the batch table can stay live without returning
            # candidate metrics or per-origin evidence on every poll.
            "adaptive_trajectories": _adaptive_trajectory_projection(state),
            "token_usage_available": run_wide_usage["available"],
            "tokens_used": run_wide_usage["tokens_used"],
            "token_limit": run_wide_usage["token_limit"],
            "token_budget_scope": run_wide_usage["scope"],
            "run_wide_usage": run_wide_usage,
            "model_usage": model_usage,
            "dsh_runtime": dsh_runtime,
        },
    }
