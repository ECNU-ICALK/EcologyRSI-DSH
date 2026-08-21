"""Execution of one resumable, frozen sibling-candidate generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.errors import (
    dsh_native_runtime_error_in_chain,
    dsh_native_runtime_retryable,
)
from ..core.models import CandidateStatus, Evaluation, RunStatus, canonical_json, digest
from ..core.redaction import (
    public_error_summary,
    public_exception_summary,
    safe_error_code,
)
from ..core.sample_results import (
    build_sample_results,
    sample_result_batch_event_payload,
    sample_results_event_payload,
)
from ..evaluators.registry import RULE_JUDGE_ID, EvaluationBundle, EvaluatorRegistry
from ..evaluators.gateway_sample_adapter import ModelTokenBudgetExhaustedError
from ..evaluators.sample_execution import (
    SampleExecutionCancelledError,
    SampleExecutionControlError,
    SampleExecutionControlUnavailableError,
    SampleExecutionPausedError,
    SampleResultCallbackError,
)
from ..evolution.batches import finalize_generation_batch, start_generation_batch
from ..evolution.context import safe_aggregate_feedback
from ..integrations.dsh_native_runtime import (
    DSH_NATIVE_EXECUTION_PROTOCOL,
    DshNativeRuntimeUnavailableError,
)
from ..integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ..integrations.model_gateway import (
    GatewayConfigurationError,
    GatewayResponseError,
    gateway_error_in_chain,
)
from ..knowledge.algorithm_smoke import (
    AlgorithmSmokeError,
    smoke_test_algorithm_spec,
)
from ..knowledge.algorithms import (
    AlgorithmAttempt,
    AlgorithmSpec,
    compile_algorithm_spec,
    debug_algorithm_spec,
)

_ALGORITHM_SMOKE_MAX_ATTEMPTS = 3


def _quiesce_native_terminal(endpoint: Any, state: Any, terminal: str) -> None:
    """Close Host admission and dispose DSH role agents before Python terminal state."""

    if (
        state.task_manifest.metadata.get("execution_protocol")
        != DSH_NATIVE_EXECUTION_PROTOCOL
    ):
        return
    dsh_tools = getattr(endpoint.server, "dsh_tools", None)
    if dsh_tools is not None:
        dsh_tools.close_run_admissions(state.run.run_id)
    runtime = getattr(endpoint.server, "dsh_native_runtime", None)
    if runtime is None:
        return
    revision = state.events[-1].seq
    try:
        runtime.cancel(
            {
                "run_id": state.run.run_id,
                "run_state_revision": revision,
                "stage_attempt": 0,
                "ledger_expected_revision": endpoint.server.ledger.latest_seq(),
                "idempotency_key": (
                    f"terminal:{terminal}:{state.run.run_id}:{revision}"
                ),
            }
        )
    except DshNativeRuntimeUnavailableError:
        # The durable Host admission fence remains closed. A missing DSH
        # process has no live children to accept; a later process restart does
        # not restore terminal run hosts.
        return


def _model_token_budget_error_in_chain(
    exc: BaseException,
) -> ModelTokenBudgetExhaustedError | None:
    pending: list[tuple[BaseException, int]] = [(exc, 0)]
    seen: set[int] = set()
    while pending:
        current, depth = pending.pop()
        identity = id(current)
        if identity in seen or depth > 32:
            continue
        seen.add(identity)
        if isinstance(current, ModelTokenBudgetExhaustedError):
            return current
        for related in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(related, BaseException):
                pending.append((related, depth + 1))
        grouped = getattr(current, "exceptions", None)
        if isinstance(grouped, (tuple, list)):
            pending.extend(
                (related, depth + 1)
                for related in grouped
                if isinstance(related, BaseException)
            )
    return None


def _model_token_budget_state(state: Any) -> dict[str, int]:
    """Summarize durable run-wide receipts for scheduler admission."""

    tokens_used = 0
    missing_call_count = 0
    missing_call_reservation = int(
        state.task_manifest.token_reservation_per_wave
    )
    for event in state.events:
        if event.kind != "ModelUsageRecorded":
            continue
        payload = event.payload
        usage_reported = (
            payload.get("usage_reported") is True
            if payload.get("schema_version") == "ecologyrsi-dsh.model-usage/2"
            else True
        )
        if not usage_reported:
            if missing_call_reservation > 0:
                # The exact amount is unavailable, but strict per-call
                # admission proves this frozen reservation is an upper bound.
                # Charge it in full so a retryable gateway failure can keep its
                # delayed-retry semantics without under-accounting the budget.
                tokens_used += missing_call_reservation
            else:
                missing_call_count += 1
            continue
        total_tokens = payload.get("total_tokens")
        if (
            isinstance(total_tokens, int)
            and not isinstance(total_tokens, bool)
            and total_tokens >= 0
        ):
            tokens_used += total_tokens
    return {
        "token_limit": state.task_manifest.token_limit,
        "tokens_used": tokens_used,
        "missing_call_count": missing_call_count,
    }


def _pause_for_model_token_budget(
    endpoint: Any,
    run_id: str,
    error: ModelTokenBudgetExhaustedError,
) -> Any:
    if error.reason == "usage_unreported":
        reason = (
            "模型 Token 用量回执不完整，硬预算无法继续安全核算；"
            f"已记录 {error.tokens_used} tokens，缺失 "
            f"{error.missing_usage_call_count} 次调用用量。"
        )
    else:
        reason = (
            "模型 Token 硬预算已停止新调用："
            f"已用 {error.tokens_used}，在途预留 {error.reserved_tokens}，"
            f"单次网关调用预留上限 {error.token_reservation_per_wave}，"
            f"上限 {error.token_limit}。"
        )
    latest = endpoint.server.director.state(run_id)
    if latest.run.status is RunStatus.RUNNING:
        endpoint.server.director.pause_run(
            run_id,
            reason=reason,
            code="model_token_budget_exhausted",
        )
        latest = endpoint.server.director.state(run_id)
    return latest


def _recoverable_evaluation_error(exc: BaseException) -> bool:
    """Keep transient gateway and result-ledger failures resumable.

    A candidate must only become permanently failed after a deterministic
    scientific or contract error.  Remote gateway errors can be wrapped by a
    sample adapter, and the result callback deliberately wraps ledger write
    failures, so inspect both boundaries before applying ``CandidateFailed``.
    """

    if _model_token_budget_error_in_chain(exc) is not None:
        return True
    gateway_error = gateway_error_in_chain(exc)
    if gateway_error is not None and gateway_error.retryable:
        return True
    pending: list[tuple[BaseException, int]] = [(exc, 0)]
    seen: set[int] = set()
    while pending:
        current, depth = pending.pop()
        identity = id(current)
        if identity in seen or depth > 32:
            continue
        seen.add(identity)
        if isinstance(
            current,
            (SampleResultCallbackError, SampleExecutionControlError),
        ):
            return True
        for related in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(related, BaseException):
                pending.append((related, depth + 1))
        grouped = getattr(current, "exceptions", None)
        if isinstance(grouped, (tuple, list)):
            for related in grouped:
                if isinstance(related, BaseException):
                    pending.append((related, depth + 1))
    return False


def _record_stage(
    endpoint: Any,
    run_id: str,
    generation: int,
    stage: str,
    status: str,
    *,
    attempt: int = 1,
    proposal_id: str | None = None,
    candidate_id: str | None = None,
    public_error: str | None = None,
    event_id: str | None = None,
) -> None:
    endpoint.server.director.record_evolution_stage(
        run_id,
        generation=generation,
        stage=stage,
        status=status,
        attempt=attempt,
        proposal_id=proposal_id,
        candidate_id=candidate_id,
        public_error=public_error_summary(public_error),
        event_id=event_id,
    )


def _restore_orphan_interventions(endpoint: Any, state: Any, proposal: Any) -> None:
    proposal_event = next(
        (
            event
            for event in reversed(state.events)
            if event.kind == "ProposalSubmitted"
            and isinstance(event.payload.get("proposal"), dict)
            and event.payload["proposal"].get("proposal_id") == proposal.proposal_id
        ),
        None,
    )
    receipts = (
        proposal_event.payload.get("intervention_receipts", [])
        if proposal_event is not None
        else []
    )
    receipts_by_id = {
        str(item.get("intervention_id")): dict(item)
        for item in receipts
        if isinstance(item, dict) and item.get("intervention_id")
    }
    for intervention in state.pending_interventions:
        receipt = receipts_by_id.get(intervention.intervention_id)
        if receipt is None:
            receipt = {
                "kind": intervention.kind.value,
                "recorded": True,
                "applied": False,
                "enforced": False,
                "application_status": "recorded",
                "reason": "恢复提案时缺少原执行收据，仅保留审计记录",
            }
        endpoint.server.ledger.append(
            state.run.run_id,
            "HumanInterventionApplied",
            {
                **receipt,
                "intervention_id": intervention.intervention_id,
                "proposal_id": proposal.proposal_id,
            },
            event_id=(
                f"{state.run.run_id}:intervention:{intervention.intervention_id}:"
                f"applied:{proposal.proposal_id}"
            ),
        )


def _candidate_signature(state: Any, proposal: Any) -> str:
    """Identify equivalent executable candidates across the whole run."""

    metadata = state.task_manifest.metadata
    proposal_metadata = (
        proposal.metadata if isinstance(proposal.metadata, Mapping) else {}
    )
    execution_plan = proposal_metadata.get("derived_execution_plan")
    execution_plan_digest = (
        execution_plan.get("execution_digest") or execution_plan.get("plan_digest")
        if isinstance(execution_plan, Mapping)
        else None
    )
    predictor_adoption = proposal_metadata.get("prediction_model_adoption")
    adopted_predictor_id = (
        predictor_adoption.get("adopted_id")
        if isinstance(predictor_adoption, Mapping)
        else metadata.get("prediction_model_id")
    )
    adopted_predictor_digest = (
        predictor_adoption.get("adopted_digest")
        if isinstance(predictor_adoption, Mapping)
        else metadata.get("prediction_model_digest")
    )
    return canonical_json(
        {
            "parameters": dict(proposal.changes),
            "derived_execution_semantics_digest": execution_plan_digest,
            "prediction_model_id": adopted_predictor_id,
            "prediction_model_digest": adopted_predictor_digest,
            "dataset_digest": metadata.get("dataset_digest")
            or list(state.task_manifest.visible_datasets),
            "evaluator_digest": metadata.get("evaluator_digest"),
        }
    )


def _spawn_generation_candidates(endpoint: Any, run_id: str, batch: Any) -> bool:
    for slot_index in range(batch.batch_size):
        state = endpoint.server.director.state(run_id)
        by_slot = {
            item.slot_index: item
            for item in state.candidates
            if item.generation == batch.generation
        }
        if slot_index in by_slot:
            continue
        linked = {item.proposal_id for item in state.candidates}
        orphan = next(
            (
                item
                for item in state.proposals
                if item.generation == batch.generation
                and item.proposal_id not in linked
            ),
            None,
        )
        _record_stage(
            endpoint,
            run_id,
            batch.generation,
            "proposal",
            "started",
            event_id=f"{run_id}:stage:{batch.generation}:slot:{slot_index}:proposal:started",
        )
        if orphan is None:
            proposal = endpoint.server.director.request_proposal(
                run_id,
                parent_candidate_id=batch.parent_candidate_id,
                generation_batch=batch,
                slot_index=slot_index,
                consume_interventions=slot_index == batch.batch_size - 1,
            )
        else:
            proposal = orphan
            if slot_index == batch.batch_size - 1:
                _restore_orphan_interventions(endpoint, state, proposal)
        fallback = (
            proposal.metadata.get("host_fallback")
            if isinstance(proposal.metadata, Mapping)
            else None
        )
        proposal_attempt = 1
        if isinstance(fallback, Mapping) and fallback.get("applied") is True:
            _record_stage(
                endpoint,
                run_id,
                batch.generation,
                "proposal",
                "failed",
                attempt=1,
                proposal_id=proposal.proposal_id,
                public_error=public_error_summary(
                    fallback.get("public_error")
                    or "远程策略提案不可用，已切换宿主有界回退"
                ),
                event_id=(
                    f"{run_id}:stage:{batch.generation}:slot:{slot_index}:"
                    "proposal:remote-failed"
                ),
            )
            strict_fallback = (
                endpoint.server.director.state(run_id)
                .task_manifest.metadata.get("remote_fallback_policy")
                == "fail_run"
            )
            if strict_fallback:
                endpoint.server.director.fail_run(
                    run_id,
                    "远程策略 API 不可用，连续进化已停止；"
                    + str(
                        public_error_summary(
                            fallback.get("public_error")
                            or "宿主回退被连续模式禁止"
                        )
                        or "宿主回退被连续模式禁止"
                    )[:400],
                )
                return False
            proposal_attempt = 2
        _record_stage(
            endpoint,
            run_id,
            batch.generation,
            "proposal",
            "completed",
            attempt=proposal_attempt,
            proposal_id=proposal.proposal_id,
        )
        _record_stage(
            endpoint,
            run_id,
            batch.generation,
            "candidate",
            "started",
            proposal_id=proposal.proposal_id,
        )
        candidate = endpoint.server.director.spawn_candidate(
            run_id, proposal, slot_index=slot_index
        )
        _record_stage(
            endpoint,
            run_id,
            batch.generation,
            "candidate",
            "completed",
            proposal_id=proposal.proposal_id,
            candidate_id=candidate.candidate_id,
        )
        signature = _candidate_signature(state, proposal)
        state = endpoint.server.director.state(run_id)
        duplicate_of = next(
            (
                item
                for item in state.candidates
                if item.candidate_id != candidate.candidate_id
                and item.status is not CandidateStatus.DUPLICATE
                and _candidate_signature(state, state.proposal(item.proposal_id))
                == signature
            ),
            None,
        )
        if duplicate_of is not None:
            endpoint.server.director.mark_candidate_duplicate(
                run_id, candidate.candidate_id, duplicate_of.candidate_id
            )
    return True


def _scientific_pass(evaluation: Evaluation) -> bool:
    return bool(evaluation.metrics.get("scientific_pass", evaluation.passed))


def _completed_judgment(
    scientific_evaluation: Evaluation,
    judged_evaluation: Evaluation,
) -> Evaluation:
    scientific_pass = _scientific_pass(scientific_evaluation)
    metrics = dict(judged_evaluation.metrics)
    # The append-only scientific evaluation owns the compressed full-sample
    # archive.  The judgment event keeps its digest and bounded preview without
    # duplicating hundreds of kilobytes per candidate.
    metrics.pop("sample_execution_trace_archive", None)
    metrics.update(
        {
            "scientific_pass": scientific_pass,
            "judge_status": "completed",
        }
    )
    data = judged_evaluation.to_dict()
    data.update(
        {
            "passed": scientific_pass and judged_evaluation.passed,
            "metrics": metrics,
        }
    )
    return Evaluation.from_dict(data)


def _unavailable_judgment(
    state: Any,
    evaluation: Evaluation,
    exc: BaseException,
) -> Evaluation:
    gateway_error = gateway_error_in_chain(exc)
    if gateway_error is not None:
        failure_class = "transient" if gateway_error.retryable else "permanent"
        error_code = gateway_error.error_code
    else:
        pending: list[tuple[BaseException, int]] = [(exc, 0)]
        seen: set[int] = set()
        transient = False
        permanent = False
        while pending:
            current, depth = pending.pop()
            identity = id(current)
            if identity in seen or depth > 32:
                continue
            seen.add(identity)
            permanent = permanent or isinstance(
                current,
                (
                    GatewayConfigurationError,
                    KeyError,
                    PermissionError,
                    TypeError,
                    ValueError,
                ),
            )
            transient = transient or isinstance(
                current,
                (ConnectionError, TimeoutError),
            )
            for related in (
                getattr(current, "__cause__", None),
                getattr(current, "__context__", None),
            ):
                if isinstance(related, BaseException):
                    pending.append((related, depth + 1))
            grouped = getattr(current, "exceptions", None)
            if isinstance(grouped, (tuple, list)):
                pending.extend(
                    (related, depth + 1)
                    for related in grouped
                    if isinstance(related, BaseException)
                )
        failure_class = (
            "permanent" if permanent else "transient" if transient else "unknown"
        )
        error_code = safe_error_code(type(exc).__name__, "judge_unavailable")
    metrics = dict(evaluation.metrics)
    metrics.pop("sample_execution_trace_archive", None)
    metrics.update(
        {
            "scientific_pass": _scientific_pass(evaluation),
            "judge_model_id": str(
                state.task_manifest.metadata.get("judge_model_id") or RULE_JUDGE_ID
            ),
            "judge_status": "unavailable",
            "judge_accepted": False,
            "judge_guidance": "独立评审暂不可用；保留科学评测结果，但禁止正式晋升。",
            "judge_parameter_override": {},
            "judge_error_type": type(exc).__name__,
            "judge_error_code": error_code,
            "judge_failure_class": failure_class,
        }
    )
    data = evaluation.to_dict()
    data.update({"passed": False, "metrics": metrics})
    return Evaluation.from_dict(data)


def _apply_candidate_judge(
    endpoint: Any,
    state: Any,
    proposal: Any,
    artifact: Any,
    evaluation: Evaluation,
) -> None:
    candidate = state.candidate(evaluation.candidate_id)
    _record_stage(
        endpoint,
        evaluation.run_id,
        candidate.generation,
        "judge",
        "started",
        proposal_id=proposal.proposal_id,
        candidate_id=candidate.candidate_id,
    )
    try:
        if (
            state.task_manifest.metadata.get("execution_protocol")
            == DSH_NATIVE_EXECUTION_PROTOCOL
        ):
            latest = endpoint.server.director.state(evaluation.run_id)
            identity_binding = latest.candidate_identity_binding(candidate.candidate_id)
            if identity_binding is None:
                raise ValueError("DSH generation judge requires candidate identity binding")
            aggregate_metrics = safe_aggregate_feedback(
                dict(evaluation.metrics),
                name="generation judge aggregate metrics",
            )
            context = {
                "candidate_id": candidate.candidate_id,
                "proposal_id": proposal.proposal_id,
                "generation": candidate.generation,
                "scientific_evaluation": {
                    "score": evaluation.score,
                    "passed": _scientific_pass(evaluation),
                    "metrics": aggregate_metrics,
                    "evaluation_digest": digest(evaluation.to_dict()),
                    "artifact_digest": artifact.digest,
                },
                "fitness_profile_digest": state.task_manifest.metadata.get(
                    "fitness_profile_digest"
                ),
                "evaluation_cohort_digest": state.task_manifest.metadata.get(
                    "evaluation_cohort_digest"
                ),
            }
            review = DshStructuredRoleRuntime(
                endpoint.server.dsh_native_runtime,
                admission=endpoint.server.dsh_tools,
            ).run(
                run_id=evaluation.run_id,
                stage="generation.judge",
                role="generation-judge",
                context=context,
                output_schema_id="ecology-generation-review@1",
                run_state_revision=latest.events[-1].seq,
                stage_attempt=1,
                ledger_expected_revision=endpoint.server.ledger.latest_seq(),
                idempotency_key=(
                    f"{evaluation.run_id}:candidate:{candidate.candidate_id}:judge"
                ),
                identity_digests={
                    "genome_digest": str(identity_binding["genome_digest"]),
                    "compiled_behavior_digest": str(
                        identity_binding["compiled_behavior_digest"]
                    ),
                    "phenotype_instance_digest": str(
                        identity_binding["phenotype_instance_digest"]
                    ),
                },
            )
            if review.get("schema_version") != "ecology-generation-review@1":
                raise ValueError("DSH generation judge returned an unsupported schema")
            if not isinstance(review.get("accepted"), bool):
                raise TypeError("DSH generation judge accepted must be boolean")
            rationale = review.get("rationale")
            flags = review.get("flags")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("DSH generation judge rationale is required")
            if not isinstance(flags, list) or any(
                not isinstance(item, str) for item in flags
            ):
                raise TypeError("DSH generation judge flags must be text")
            data = evaluation.to_dict()
            data["passed"] = bool(review["accepted"])
            data["metrics"] = {
                **dict(evaluation.metrics),
                "judge_model_id": str(
                    state.task_manifest.metadata.get("review_model_id")
                    or "dsh-generation-judge"
                ),
                "judge_accepted": bool(review["accepted"]),
                "judge_guidance": rationale.strip(),
                "judge_flags": list(flags),
                "judge_parameter_override": {},
                "judge_result_digest": digest(review),
            }
            judged = EvaluationBundle(
                artifact=artifact,
                evaluation=Evaluation.from_dict(data),
            )
        else:
            judged = endpoint.server.evaluators.apply_judge(
                state.task_manifest,
                proposal,
                EvaluationBundle(artifact=artifact, evaluation=evaluation),
            )
    except Exception as exc:
        gateway_error = gateway_error_in_chain(exc)
        if gateway_error is not None and gateway_error.retryable:
            # Keep the started stage resumable.  Reusing a ``failed`` stage
            # event with attempt=1 would collide with the next retry's
            # idempotent ``started`` event and strand the candidate.
            raise
        endpoint.server.director.record_judgment(
            _unavailable_judgment(state, evaluation, exc)
        )
        _record_stage(
            endpoint,
            evaluation.run_id,
            candidate.generation,
            "judge",
            "failed",
            proposal_id=proposal.proposal_id,
            candidate_id=candidate.candidate_id,
            public_error=public_exception_summary(exc),
        )
        return
    endpoint.server.director.record_judgment(
        _completed_judgment(evaluation, judged.evaluation)
    )
    _record_stage(
        endpoint,
        evaluation.run_id,
        candidate.generation,
        "judge",
        "completed",
        proposal_id=proposal.proposal_id,
        candidate_id=candidate.candidate_id,
    )


def _algorithm_failure_code(exc: BaseException, fallback: str) -> str:
    code = getattr(exc, "code", None)
    return safe_error_code(code, fallback) or fallback


def _smoke_failure_feedback(exc: BaseException, attempt: int) -> dict[str, Any]:
    retryable = bool(getattr(exc, "retryable", False)) or isinstance(
        exc,
        (ConnectionError, TimeoutError),
    )
    result: dict[str, Any] = {
        "attempt": attempt,
        "failure_code": _algorithm_failure_code(exc, "algorithm_smoke_failed"),
        "retryable": retryable,
        "exception_type": type(exc).__name__,
        "public_error": public_exception_summary(exc),
    }
    details = getattr(exc, "evidence", None)
    if isinstance(details, Mapping):
        result["details"] = dict(details)
    canonical_json(result)
    return result


def _validate_smoke_evidence(
    value: Any,
    spec: AlgorithmSpec,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AlgorithmSmokeError(
            "smoke_invalid_evidence",
            "algorithm smoke runner must return an evidence object",
        )
    result = dict(value)
    canonical_json(result)
    if len(canonical_json(result)) > 24_000:
        raise AlgorithmSmokeError(
            "smoke_invalid_evidence",
            "algorithm smoke evidence exceeds the bounded contract",
        )
    stored_smoke_digest = result.pop("smoke_digest", None)
    expected_smoke_digest = digest(result)
    result["smoke_digest"] = stored_smoke_digest
    if stored_smoke_digest != expected_smoke_digest:
        raise AlgorithmSmokeError(
            "smoke_digest_mismatch",
            "algorithm smoke evidence digest does not match its payload",
            evidence={
                "expected_smoke_digest": expected_smoke_digest,
                "observed_smoke_digest": stored_smoke_digest,
            },
        )
    if result.get("status") not in {"passed", "compatibility_skipped"}:
        raise AlgorithmSmokeError(
            "smoke_not_passed",
            "algorithm smoke runner did not report a passing status",
        )
    if result.get("source_partition") != "training_fit":
        raise AlgorithmSmokeError(
            "smoke_forbidden_partition",
            "algorithm smoke execution must use training_fit only",
        )
    if result.get("restricted_partition_access") is not False:
        raise AlgorithmSmokeError(
            "smoke_forbidden_partition",
            "algorithm smoke evidence did not prove the restricted partition boundary",
        )
    if result.get("status") == "passed":
        algorithm_ir = spec.algorithm_ir
        expected_ir_digest = (
            algorithm_ir.get("ir_digest")
            if isinstance(algorithm_ir, Mapping)
            else None
        )
        if result.get("algorithm_ir_digest") != expected_ir_digest:
            raise AlgorithmSmokeError(
                "smoke_ir_mismatch",
                "algorithm smoke evidence does not match the compiled IR",
            )
    return result


def _run_algorithm_smoke(
    endpoint: Any,
    spec: AlgorithmSpec,
    task: Any,
    proposal: Any,
    *,
    attempt: int,
    failure_feedback: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    runner = getattr(endpoint.server, "algorithm_smoke_runner", None)
    if callable(runner):
        evidence = runner(
            spec,
            task,
            proposal,
            attempt=attempt,
            failure_feedback=failure_feedback,
        )
    else:
        evaluator = endpoint.server.evaluators
        datasets = getattr(evaluator, "datasets", None)
        if datasets is None:
            datasets = getattr(endpoint.server, "datasets", None)
        evidence = smoke_test_algorithm_spec(
            spec,
            task,
            datasets,
            attempt=attempt,
            failure_feedback=failure_feedback,
        )
    return _validate_smoke_evidence(evidence, spec)


def _ensure_candidate_algorithm_ready(
    endpoint: Any,
    state: Any,
    proposal: Any,
    candidate: Any,
) -> bool:
    """Compile and debug only a host-registered pipeline before training."""

    attempts = state.algorithm_attempts_for(candidate.candidate_id)
    if any(item.phase == "debug" and item.status == "passed" for item in attempts):
        return True
    if any(item.phase == "compile" and item.status == "failed" for item in attempts):
        return False

    compiled = state.compiled_algorithm_for(candidate.candidate_id)
    spec: AlgorithmSpec | None = None
    if compiled is not None:
        spec = AlgorithmSpec.from_dict(compiled)
    else:
        try:
            spec = compile_algorithm_spec(
                state.task_manifest,
                proposal,
                state.knowledge_for(candidate.generation),
            )
        except Exception as exc:  # noqa: BLE001 - isolate candidate compilation
            endpoint.server.director.record_algorithm_attempt(
                AlgorithmAttempt(
                    run_id=candidate.run_id,
                    generation=candidate.generation,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    phase="compile",
                    attempt=1,
                    status="failed",
                    evidence={
                        "registered_adapters_only": True,
                        "exception_type": type(exc).__name__,
                    },
                    failure_code=_algorithm_failure_code(
                        exc, "algorithm_compile_failed"
                    ),
                    public_error=public_exception_summary(exc),
                )
            )
            endpoint.server.director.fail_candidate(
                candidate.run_id,
                candidate.candidate_id,
                f"候选算法编译失败：{public_exception_summary(exc)}",
            )
            return False
        endpoint.server.director.record_algorithm_attempt(
            AlgorithmAttempt(
                run_id=candidate.run_id,
                generation=candidate.generation,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
                phase="compile",
                attempt=1,
                status="passed",
                algorithm_spec_digest=spec.spec_digest,
                algorithm_spec=spec.to_dict(),
                evidence={
                    "registered_adapters_only": True,
                    "tool_count": len(spec.tool_ids),
                    "knowledge_mapping_count": sum(
                        item.get("decision") == "adopted"
                        for item in spec.knowledge_mappings
                    ),
                    "knowledge_not_selected_count": sum(
                        item.get("decision") != "adopted"
                        for item in spec.knowledge_mappings
                    ),
                },
            )
        )

    debug_attempts = [item for item in attempts if item.phase == "debug"]
    failure_feedback = tuple(
        dict(item.evidence["failure_feedback"])
        for item in debug_attempts
        if item.status == "failed"
        and isinstance(item.evidence.get("failure_feedback"), Mapping)
    )
    next_attempt = max((item.attempt for item in debug_attempts), default=0) + 1
    if next_attempt > _ALGORITHM_SMOKE_MAX_ATTEMPTS:
        endpoint.server.director.fail_candidate(
            candidate.run_id,
            candidate.candidate_id,
            "候选算法 training_fit smoke 重试预算已耗尽。",
        )
        return False

    for attempt_number in range(
        next_attempt,
        _ALGORITHM_SMOKE_MAX_ATTEMPTS + 1,
    ):
        try:
            debug_evidence = debug_algorithm_spec(
                spec,
                state.task_manifest,
                proposal,
                state.knowledge_for(candidate.generation),
            )
        except Exception as exc:  # noqa: BLE001 - isolate static debug validation
            endpoint.server.director.record_algorithm_attempt(
                AlgorithmAttempt(
                    run_id=candidate.run_id,
                    generation=candidate.generation,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    phase="debug",
                    attempt=attempt_number,
                    status="failed",
                    algorithm_spec_digest=spec.spec_digest,
                    evidence={
                        "stage": "static_debug",
                        "registered_adapters_only": True,
                        "exception_type": type(exc).__name__,
                    },
                    failure_code=_algorithm_failure_code(
                        exc, "algorithm_debug_failed"
                    ),
                    public_error=public_exception_summary(exc),
                )
            )
            endpoint.server.director.fail_candidate(
                candidate.run_id,
                candidate.candidate_id,
                f"候选算法静态调试失败：{public_exception_summary(exc)}",
            )
            return False

        try:
            smoke_evidence = _run_algorithm_smoke(
                endpoint,
                spec,
                state.task_manifest,
                proposal,
                attempt=attempt_number,
                failure_feedback=failure_feedback,
            )
        except Exception as exc:  # noqa: BLE001 - isolate registered smoke tools
            gateway_error = gateway_error_in_chain(exc)
            if gateway_error is not None and gateway_error.retryable:
                # A registered smoke tool may consult the same remote provider
                # as the proposal/evaluator.  Preserve the candidate's
                # resumable state instead of spending the algorithm smoke
                # budget and permanently rejecting it during a provider
                # cooldown.
                raise
            feedback = _smoke_failure_feedback(exc, attempt_number)
            endpoint.server.director.record_algorithm_attempt(
                AlgorithmAttempt(
                    run_id=candidate.run_id,
                    generation=candidate.generation,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    phase="debug",
                    attempt=attempt_number,
                    status="failed",
                    algorithm_spec_digest=spec.spec_digest,
                    evidence={
                        "stage": "training_fit_smoke",
                        "registered_operators_only": True,
                        "algorithm_ir_digest": (
                            spec.algorithm_ir.get("ir_digest")
                            if isinstance(spec.algorithm_ir, Mapping)
                            else None
                        ),
                        "failure_feedback": feedback,
                    },
                    failure_code=str(feedback["failure_code"]),
                    public_error=str(feedback["public_error"]),
                )
            )
            failure_feedback = (*failure_feedback, feedback)
            if (
                not bool(feedback["retryable"])
                or attempt_number >= _ALGORITHM_SMOKE_MAX_ATTEMPTS
            ):
                endpoint.server.director.fail_candidate(
                    candidate.run_id,
                    candidate.candidate_id,
                    "候选算法 training_fit smoke 失败："
                    + str(feedback["public_error"]),
                )
                return False
            continue

        endpoint.server.director.record_algorithm_attempt(
            AlgorithmAttempt(
                run_id=candidate.run_id,
                generation=candidate.generation,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
                phase="debug",
                attempt=attempt_number,
                status="passed",
                algorithm_spec_digest=spec.spec_digest,
                evidence={
                    **debug_evidence,
                    "stage": "training_fit_smoke",
                    "smoke": smoke_evidence,
                    "prior_smoke_failures": [
                        dict(item) for item in failure_feedback
                    ],
                },
            )
        )
        return True
    return False


def _evaluate_candidate(endpoint: Any, run_id: str, candidate_id: str) -> None:
    state = endpoint.server.director.state(run_id)
    candidate = state.candidate(candidate_id)
    proposal = state.proposal(candidate.proposal_id)
    existing_artifact = state.artifact_for(candidate.candidate_id)
    existing_evaluation = state.evaluation_for(candidate.candidate_id)
    if candidate.status is CandidateStatus.EVALUATED:
        judge_status = (
            existing_evaluation.metrics.get("judge_status")
            if existing_evaluation is not None
            else None
        )
        if (
            existing_artifact is not None
            and existing_evaluation is not None
            and judge_status != "completed"
            and state.promotion_for(candidate.candidate_id) is None
        ):
            _apply_candidate_judge(
                endpoint,
                state,
                proposal,
                existing_artifact,
                existing_evaluation,
            )
        return
    if candidate.status is not CandidateStatus.SPAWNED:
        return
    if not _ensure_candidate_algorithm_ready(
        endpoint, state, proposal, candidate
    ):
        return
    state = endpoint.server.director.state(run_id)
    compiled_algorithm = state.compiled_algorithm_for(candidate.candidate_id)
    if compiled_algorithm is None:
        raise RuntimeError("debugged candidate is missing its compiled algorithm spec")
    algorithm_spec = AlgorithmSpec.from_dict(compiled_algorithm)
    active_stage = "training"
    _record_stage(
        endpoint,
        run_id,
        candidate.generation,
        "training",
        "started",
        proposal_id=proposal.proposal_id,
        candidate_id=candidate.candidate_id,
    )
    try:
        evaluation_started = False

        def start_evaluation() -> None:
            nonlocal active_stage, evaluation_started
            if evaluation_started:
                return
            _record_stage(
                endpoint,
                run_id,
                candidate.generation,
                "training",
                "completed",
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
            )
            active_stage = "evaluation"
            _record_stage(
                endpoint,
                run_id,
                candidate.generation,
                "evaluation",
                "started",
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
            )
            evaluation_started = True

        evaluation_kwargs: dict[str, Any] = {
            "on_training_complete": start_evaluation,
        }
        sample_results_revision: str | None = None
        if isinstance(endpoint.server.evaluators, EvaluatorRegistry):
            evaluation_kwargs["algorithm_spec"] = algorithm_spec
            sample_result_batch_index = 0

            def sample_run_control() -> str:
                """Expose only the owning run's current scheduling state."""

                run_status = endpoint.server.director.state(run_id).run.status
                if run_status is RunStatus.RUNNING:
                    return "running"
                if run_status is RunStatus.PAUSED:
                    return "paused"
                # A sample invocation cannot make progress in created or terminal
                # states. Treat every such transition as a terminal cancellation
                # without exposing unrelated run state through the adapter API.
                return "cancelled"

            def accepts_sample_publication() -> bool:
                return endpoint.server.director.state(run_id).run.status in {
                    RunStatus.RUNNING,
                    RunStatus.PAUSED,
                }

            evaluation_kwargs["on_sample_control"] = sample_run_control

            def prepare_sample_checkpoint(
                checkpoint: Mapping[str, Any],
            ) -> Mapping[str, Any]:
                """Open or resume one exact evaluation cohort before API work."""

                nonlocal sample_results_revision, sample_result_batch_index
                if not evaluation_started:
                    start_evaluation()
                prepared = endpoint.server.director.prepare_evaluation_sample_checkpoint(
                    run_id,
                    generation=candidate.generation,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    checkpoint=checkpoint,
                )
                revision = prepared.get("revision")
                next_batch_index = prepared.get("next_batch_index")
                if not isinstance(revision, str) or not revision.strip():
                    raise RuntimeError("sample checkpoint did not return a revision")
                if (
                    isinstance(next_batch_index, bool)
                    or not isinstance(next_batch_index, int)
                    or next_batch_index < 1
                ):
                    raise RuntimeError("sample checkpoint returned an invalid batch index")
                if (
                    sample_results_revision is not None
                    and sample_results_revision != revision
                ):
                    raise RuntimeError("sample checkpoint changed revisions mid-evaluation")
                sample_results_revision = revision
                sample_result_batch_index = next_batch_index - 1
                return {
                    **prepared,
                    "token_budget_state": _model_token_budget_state(
                        endpoint.server.director.state(run_id)
                    ),
                }

            def record_sample_results(
                scoring_rows: Any,
            ) -> None:
                nonlocal sample_result_batch_index
                if not accepts_sample_publication():
                    return
                if not evaluation_started:
                    start_evaluation()
                if sample_results_revision is None:
                    raise RuntimeError("sample result callback ran before checkpoint")
                projected = build_sample_results(
                    candidate.candidate_id, scoring_rows
                )
                if not projected:
                    return
                sample_result_batch_index += 1
                try:
                    endpoint.server.director.record_evaluation_sample_result_batch(
                        run_id,
                        sample_result_batch_event_payload(
                            run_id,
                            candidate.candidate_id,
                            projected,
                            revision=sample_results_revision,
                            batch_index=sample_result_batch_index,
                        ),
                    )
                except Exception:
                    # Cancellation may win between the read-only guard and the
                    # ledger append. The completed physical call is accounted by
                    # its separate usage callback, while its outcome is discarded.
                    if not accepts_sample_publication():
                        return
                    raise

            evaluation_kwargs["on_sample_results"] = record_sample_results
            evaluation_kwargs["on_sample_checkpoint"] = prepare_sample_checkpoint

            def record_model_usage(receipts: Any) -> Mapping[str, Any]:
                """Durably publish physical gateway receipts before continuing."""

                if sample_results_revision is None:
                    raise SampleResultCallbackError(
                        "model usage callback ran before checkpoint"
                    )
                try:
                    endpoint.server.director.record_model_usage_batch(
                        run_id,
                        generation=candidate.generation,
                        candidate_id=candidate.candidate_id,
                        revision=sample_results_revision,
                        receipts=receipts,
                    )
                except Exception as exc:  # noqa: BLE001 - retain retry boundary
                    raise SampleResultCallbackError(
                        "model usage receipts could not be persisted"
                    ) from exc
                return _model_token_budget_state(
                    endpoint.server.director.state(run_id)
                )

            evaluation_kwargs["on_model_usage"] = record_model_usage

            def record_evaluation_progress(progress: Mapping[str, Any]) -> None:
                # Planner microbatches cover the complete evaluation cohort.
                # Sparse repair calls are reflected in the final aggregate
                # metrics and would reset their own per-call denominator.
                if progress.get("role") != "planner":
                    return
                if not accepts_sample_publication():
                    return
                if not evaluation_started:
                    start_evaluation()
                if sample_results_revision is None:
                    raise SampleResultCallbackError(
                        "evaluation progress callback ran before checkpoint"
                    )
                try:
                    endpoint.server.director.record_evaluation_progress(
                        run_id,
                        generation=candidate.generation,
                        proposal_id=proposal.proposal_id,
                        candidate_id=candidate.candidate_id,
                        progress=progress,
                        revision=sample_results_revision,
                    )
                except Exception as exc:  # noqa: BLE001 - retry ledger boundary
                    if not accepts_sample_publication():
                        return
                    raise SampleResultCallbackError(
                        "evaluation progress heartbeat could not be persisted"
                    ) from exc

            evaluation_kwargs["on_evaluation_progress"] = (
                record_evaluation_progress
            )
        bundle = endpoint.server.evaluators.evaluate_scientific(
            state.task_manifest,
            candidate,
            proposal,
            **evaluation_kwargs,
        )
        if not evaluation_started:
            start_evaluation()
        evaluation = bundle.evaluation
        if existing_artifact is None:
            existing_artifact = endpoint.server.director.record_artifact(bundle.artifact)
        else:
            expected = bundle.artifact.to_dict()
            actual = existing_artifact.to_dict()
            expected.pop("created_at", None)
            actual.pop("created_at", None)
            if expected != actual:
                raise RuntimeError("恢复评测时训练产物与已记录产物不一致")
            data = evaluation.to_dict()
            data["artifact_digest"] = existing_artifact.digest
            evaluation = Evaluation.from_dict(data)
            bundle = EvaluationBundle(
                artifact=existing_artifact,
                evaluation=evaluation,
                sample_results=bundle.sample_results,
            )
        _record_stage(
            endpoint,
            run_id,
            candidate.generation,
            "evaluation",
            "completed",
            proposal_id=proposal.proposal_id,
            candidate_id=candidate.candidate_id,
        )
        completed_sample_results = (
            sample_results_event_payload(
                bundle.evaluation,
                bundle.sample_results,
                revision=sample_results_revision,
            )
            if bundle.sample_results is not None
            and sample_results_revision is not None
            else None
        )
        scientific_evaluation = endpoint.server.director.record_evaluation(
            bundle.evaluation,
            sample_results=completed_sample_results,
        )
    except (SampleExecutionPausedError, SampleExecutionCancelledError):
        # Run control owns these transitions. Keep the exact sample checkpoint
        # and candidate open so pause can resume and cancel remains terminal.
        return
    except SampleExecutionControlUnavailableError:
        # Fail closed before another API call, but keep the candidate recoverable.
        raise
    except Exception as exc:  # noqa: BLE001 - isolate one candidate evaluation
        token_budget_error = _model_token_budget_error_in_chain(exc)
        if token_budget_error is not None:
            _pause_for_model_token_budget(endpoint, run_id, token_budget_error)
            return
        # Leave the candidate resumable when a provider or the durable sample
        # result callback is temporarily unavailable.  A CandidateFailed
        # event here would make the background worker unable to retry the
        # interrupted evaluation after its gateway cooldown.
        if _recoverable_evaluation_error(exc):
            raise
        _record_stage(
            endpoint,
            run_id,
            candidate.generation,
            active_stage,
            "failed",
            proposal_id=proposal.proposal_id,
            candidate_id=candidate.candidate_id,
            public_error=public_exception_summary(exc),
        )
        endpoint.server.director.fail_candidate(
            run_id,
            candidate.candidate_id,
            f"候选训练或评测失败：{public_exception_summary(exc)}",
        )
        return
    state = endpoint.server.director.state(run_id)
    artifact = state.artifact_for(candidate.candidate_id)
    if artifact is None:
        raise RuntimeError("recorded scientific evaluation is missing its artifact")
    _apply_candidate_judge(
        endpoint,
        state,
        proposal,
        artifact,
        scientific_evaluation,
    )


def _has_advanced_to_current_generation(state: Any) -> bool:
    """Return whether the event stream proves the current generation was closed.

    Candidate count alone is not enough to infer budget exhaustion: a process can
    stop after spawning the last candidate but before training and evaluation.
    ``GenerationAdvanced`` is emitted only after the batch decision barrier, so
    it is the durable evidence needed to recover a lost finalization write.
    """

    generation = int(state.run.generation)
    return generation > 0 and any(
        event.kind == "GenerationAdvanced"
        and int(event.payload.get("generation", -1)) == generation
        for event in state.events
    )


def _generation_all_duplicates(state: Any, generation: int) -> bool:
    """Return whether a complete generation only rediscovered prior work."""

    candidates = tuple(
        candidate
        for candidate in state.candidates
        if candidate.generation == generation
    )
    if not candidates:
        return False
    batch = state.batch_for(generation)
    if batch is not None and len(candidates) != batch.batch_size:
        return False
    return all(
        candidate.status is CandidateStatus.DUPLICATE for candidate in candidates
    )


def _generation_decision_finalized(state: Any, generation: int) -> bool:
    """Return whether the duplicate batch crossed its durable decision barrier."""

    analysis = state.analysis_for(generation)
    if analysis is None:
        return False
    return any(
        event.kind == "GenerationChampionSelected"
        and int(event.payload.get("generation", -1)) == generation
        and event.payload.get("analysis_digest") == analysis.analysis_digest
        for event in state.events
    )


def _complete_converged_run(endpoint: Any, run_id: str, state: Any) -> Any:
    _quiesce_native_terminal(endpoint, state, "converged")
    endpoint.server.director.complete_run(
        run_id,
        termination_reason="search_space_converged_all_candidates_duplicate",
        outcome=(
            "completed_with_search_retained_candidate"
            if state.run.best_candidate_id is not None
            else "completed_without_acceptable_candidate"
        ),
    )
    return endpoint.server.director.state(run_id)


def complete_if_budget_exhausted(
    endpoint: Any, run_id: str, state: Any | None = None
) -> Any:
    """Idempotently finish a run whose final budget event was already written.

    A crash can occur after a generation decision commits and before
    ``RunCompleted`` commits. On retry, starting another batch would either
    reject the exhausted budget or repeat finalized work. This helper
    reconstructs the same termination decision from the durable projection
    and writes the missing terminal event.
    """

    state = state or endpoint.server.director.state(run_id)
    if state.run.status is not RunStatus.RUNNING:
        return state

    # Duplicate-only convergence has no new scientific evaluation, so it is
    # terminal without consuming an epoch or writing GenerationAdvanced. The
    # finalized decision event prevents a partially analyzed batch from being
    # mistaken for completed convergence during recovery.
    current_generation = int(state.run.generation)
    if (
        _generation_all_duplicates(state, current_generation)
        and _generation_decision_finalized(state, current_generation)
    ):
        return _complete_converged_run(endpoint, run_id, state)

    # Preserve recovery for ledgers written by older releases, which advanced
    # an all-duplicate generation before attempting RunCompleted.
    previous_generation = int(state.run.generation) - 1
    if (
        _has_advanced_to_current_generation(state)
        and previous_generation >= 0
        and _generation_all_duplicates(state, previous_generation)
    ):
        return _complete_converged_run(endpoint, run_id, state)

    max_generations = max(1, int(state.task_manifest.max_generations))
    generation_exhausted = state.run.generation >= max_generations
    current_generation_has_candidates = any(
        candidate.generation == int(state.run.generation)
        for candidate in state.candidates
    )
    candidate_exhausted = (
        len(state.candidates) >= int(state.task_manifest.max_candidates)
        and _has_advanced_to_current_generation(state)
        # A retry can re-enter this preflight after the final candidate slot
        # has been spawned but before its evaluation and generation decision
        # have committed.  The occupied slot is not a completed budget until
        # GenerationAdvanced moves past that candidate's generation.
        and not current_generation_has_candidates
    )
    if not generation_exhausted and not candidate_exhausted:
        return state

    reasons: list[str] = []
    if candidate_exhausted:
        reasons.append("candidate_budget_exhausted")
    if generation_exhausted:
        reasons.append("generation_budget_exhausted")
    _quiesce_native_terminal(endpoint, state, "budget_exhausted")
    endpoint.server.director.complete_run(
        run_id,
        termination_reason="+".join(reasons) or "budget_exhausted",
        outcome=(
            "completed_with_search_retained_candidate"
            if state.run.best_candidate_id is not None
            else "budget_exhausted_without_acceptable_candidate"
        ),
    )
    return endpoint.server.director.state(run_id)


def _generation_evidence_failure(state: Any, generation: int) -> str | None:
    """Reject a round that cannot provide feedback for the next proposal."""

    evaluations = tuple(
        evaluation
        for candidate in state.candidates
        if candidate.generation == generation
        if (evaluation := state.evaluation_for(candidate.candidate_id)) is not None
    )
    if not evaluations:
        return (
            "本轮证据门禁失败（generation_evidence_missing）："
            "未产生任何新的科学评测，候选可能全部执行失败或与历史候选重复；"
            "已停止连续进化，未推进到下一轮。"
        )
    if all(
        evaluation.metrics.get("judge_status") == "unavailable"
        for evaluation in evaluations
    ):
        return (
            "本轮证据门禁失败（generation_judges_unavailable）："
            f"已完成的 {len(evaluations)} 个科学评测的独立评审均不可用；"
            "已停止连续进化，未推进到下一轮。"
        )
    return None


def _generation_judges_should_retry(state: Any, generation: int) -> bool:
    """Keep a generation resumable while every judge result is unavailable."""

    evaluations = tuple(
        evaluation
        for candidate in state.candidates
        if candidate.generation == generation
        if (evaluation := state.evaluation_for(candidate.candidate_id)) is not None
    )
    return bool(evaluations) and all(
        evaluation.metrics.get("judge_status") == "unavailable"
        for evaluation in evaluations
    ) and any(
        evaluation.metrics.get("judge_failure_class") == "transient"
        for evaluation in evaluations
    )


def execute_generation(endpoint: Any, run_id: str) -> Any:
    """Complete exactly one generation including its unified decision barrier."""

    state = complete_if_budget_exhausted(endpoint, run_id)
    if state.run.status is not RunStatus.RUNNING:
        return state
    batch = start_generation_batch(endpoint.server.director, run_id)
    try:
        spawned = _spawn_generation_candidates(endpoint, run_id, batch)
        if not spawned:
            return endpoint.server.director.state(run_id)
    except Exception as exc:
        # Control commands are intentionally allowed while a remote proposal is
        # waiting.  If pause/cancel wins that race, preserve the partially
        # written batch for resume instead of converting the control transition
        # into a terminal execution failure.
        latest = endpoint.server.director.state(run_id)
        if latest.run.status is not RunStatus.RUNNING:
            return latest
        gateway_error = gateway_error_in_chain(exc)
        dsh_error = dsh_native_runtime_error_in_chain(exc)
        if (
            gateway_error is not None
            and gateway_error.retryable
        ) or (
            dsh_error is not None
            and dsh_native_runtime_retryable(dsh_error)
        ):
            _record_stage(
                endpoint,
                run_id,
                batch.generation,
                "proposal",
                "failed",
                public_error=public_exception_summary(exc),
            )
            raise
        _record_stage(
            endpoint,
            run_id,
            batch.generation,
            "proposal",
            "failed",
            public_error=public_exception_summary(exc),
            event_id=f"{run_id}:stage:{batch.generation}:batch-generation:failed",
        )
        endpoint.server.director.fail_run(
            run_id, f"候选批次生成失败：{public_exception_summary(exc)}"
        )
        return endpoint.server.director.state(run_id)

    state = endpoint.server.director.state(run_id)
    current = sorted(
        (item for item in state.candidates if item.generation == batch.generation),
        key=lambda item: item.slot_index,
    )
    for candidate in current:
        _evaluate_candidate(endpoint, run_id, candidate.candidate_id)
        latest = endpoint.server.director.state(run_id)
        if latest.run.status is not RunStatus.RUNNING:
            return latest

    state = endpoint.server.director.state(run_id)
    if _generation_judges_should_retry(state, batch.generation):
        raise GatewayResponseError(
            "all generation judges are temporarily unavailable",
            retryable=True,
            error_code="generation_judges_unavailable",
        )
    evidence_failure = _generation_evidence_failure(state, batch.generation)
    if (
        evidence_failure is not None
        and "generation_judges_unavailable" in evidence_failure
    ):
        _record_stage(
            endpoint,
            run_id,
            batch.generation,
            "decision",
            "failed",
            public_error=evidence_failure[:500],
            event_id=f"{run_id}:stage:{batch.generation}:batch:decision:evidence-failed",
        )
        state = endpoint.server.director.state(run_id)
        _quiesce_native_terminal(endpoint, state, "judges_unavailable")
        endpoint.server.director.fail_run(run_id, evidence_failure)
        return endpoint.server.director.state(run_id)
    for candidate in current:
        refreshed = state.candidate(candidate.candidate_id)
        if refreshed.status is CandidateStatus.EVALUATED:
            _record_stage(
                endpoint,
                run_id,
                batch.generation,
                "decision",
                "started",
                proposal_id=refreshed.proposal_id,
                candidate_id=refreshed.candidate_id,
            )
    try:
        finalize_generation_batch(endpoint.server.director, run_id)
    except Exception as exc:
        state = endpoint.server.director.state(run_id)
        gateway_error = gateway_error_in_chain(exc)
        if gateway_error is None or not gateway_error.retryable:
            for candidate in current:
                refreshed = state.candidate(candidate.candidate_id)
                if (
                    refreshed.status is CandidateStatus.EVALUATED
                    and state.promotion_for(refreshed.candidate_id) is None
                ):
                    _record_stage(
                        endpoint,
                        run_id,
                        batch.generation,
                        "decision",
                        "failed",
                        proposal_id=refreshed.proposal_id,
                        candidate_id=refreshed.candidate_id,
                        public_error=public_exception_summary(exc),
                    )
        raise
    state = endpoint.server.director.state(run_id)
    for candidate in current:
        refreshed = state.candidate(candidate.candidate_id)
        if refreshed.status in {CandidateStatus.PROMOTED, CandidateStatus.REJECTED}:
            _record_stage(
                endpoint,
                run_id,
                batch.generation,
                "decision",
                "completed",
                proposal_id=refreshed.proposal_id,
                candidate_id=refreshed.candidate_id,
            )
    if _generation_all_duplicates(state, batch.generation):
        return _complete_converged_run(endpoint, run_id, state)
    evidence_failure = _generation_evidence_failure(state, batch.generation)
    if evidence_failure is not None:
        _record_stage(
            endpoint,
            run_id,
            batch.generation,
            "decision",
            "failed",
            public_error=evidence_failure[:500],
            event_id=f"{run_id}:stage:{batch.generation}:batch:decision:evidence-failed",
        )
        state = endpoint.server.director.state(run_id)
        _quiesce_native_terminal(endpoint, state, "evidence_failed")
        endpoint.server.director.fail_run(run_id, evidence_failure)
        return endpoint.server.director.state(run_id)
    endpoint.server.director.advance_generation(run_id)
    return complete_if_budget_exhausted(endpoint, run_id)


__all__ = [
    "_ensure_candidate_algorithm_ready",
    "complete_if_budget_exhausted",
    "execute_generation",
]
