"""Public, redacted event projections for browser clients."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..core.models import CandidateStatus, RunStatus
from ..core.redaction import (
    public_error_summary,
    redact_sensitive_text,
    sanitize_public_value,
)
from ..core.sample_results import (
    SAMPLE_REWARD_DEFINITION,
    decode_sample_result_batch,
    decode_sample_results,
)
from .shared import _assert_http_scope, _event_type, public_fields


_PRIVATE_SAMPLE_EVENT_KINDS = frozenset(
    {
        "EvaluationSampleResultsStarted",
        "EvaluationSampleResultsResumed",
        "EvaluationSampleResultBatchRecorded",
        "EvaluationSampleResultsRecorded",
    }
)
_TERMINAL_RUN_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.CANCELLED, RunStatus.FAILED}
)


def _browser_sample_result_row(value: dict[str, Any]) -> dict[str, Any]:
    """Hide private scoring penalties from the model-prediction surface.

    The private archive retains a finite worst-case value so a failed model
    call cannot gain an advantage and checkpoint resume stays deterministic.
    That host-generated penalty is not a model prediction, error, or reward.
    """

    row = dict(value)
    status = str(row.get("status") or "").strip().casefold()
    scoring_fallback = row.get("scoring_fallback")
    failed = status == "failed" or bool(scoring_fallback)
    if not failed:
        row["model_prediction_available"] = row.get("predicted") is not None
        row["scoring_penalty_applied"] = False
        return row

    row.update(
        {
            "status": "failed",
            "predicted": None,
            "prediction_source": "failed_no_model_prediction",
            "absolute_error": None,
            "reward": None,
            "model_prediction_available": False,
            "scoring_penalty_applied": bool(scoring_fallback),
        }
    )
    for name in ("raw_normalized_reward", "normalized_reward"):
        if name in row:
            row[name] = None
    return row


def _screening_completion_matches(state, candidate, start, completed):
    checkpoint = start.payload.get('checkpoint') if start is not None else None
    if not isinstance(checkpoint, Mapping) or checkpoint.get('evaluation_phase') != 'screening':
        return False
    generation = start.payload.get('generation')
    return (
        generation == candidate.generation
        and start.payload.get('proposal_id') == candidate.proposal_id
        and completed.payload.get('evaluation_id') == f'screening-evaluation:{candidate.candidate_id}:{generation}'
        and any(
            event.kind == 'CandidateScreeningRecorded'
            and event.payload.get('candidate_id') == candidate.candidate_id
            and event.payload.get('generation') == generation
            and event.payload.get('cohort_digest') == checkpoint.get('cohort_digest')
            and event.payload.get('prediction_cell_count') == completed.payload.get('record_count')
            for event in state.events
        )
    )


_EVENT_FIELDS = {
    'GenerationBatchStarted': ('batch', 'generation batch_size parent_candidate_id context_digest'),
    'CandidateRevisionCreated': ('revision', 'generation candidate_id revision_id parent_revision_id source_batch_index status revision_digest genome_digest behavior_digest mutation_digest'),
    'FormalTrajectoryStarted': ('trajectory', 'trajectory_id generation candidate_id initial_revision_id batch_count status'),
    'FormalBatchStarted': ('batch', 'batch_id trajectory_id generation candidate_id revision_id batch_index batch_count cohort_digest origin_count'),
    'TrajectoryRevisionAdvanced': ('activation', 'activation_id generation candidate_id batch_index from_revision_id to_revision_id reason'),
    'FormalTrajectoryCompleted': ('', 'candidate_id final_revision_id'),
    'HoldoutArmStarted': ('', 'generation holdout_arm candidate_id candidate_revision_id origin_count cohort_digest'),
    'HoldoutRecheckStarted': ('', 'generation holdout_arm candidate_id candidate_revision_id origin_count cohort_digest'),
    'CandidateEffectiveRevisionFrozen': ('', 'generation selected_candidate_id selected_revision_id comparison_digest'),
    'CandidateMarkedDuplicate': ('', 'candidate_id duplicate_of_candidate_id'),
    'ArtifactRecorded': ('artifact', 'artifact_id candidate_id'),
    'EvaluationRecorded': ('evaluation', 'candidate_id score passed'),
    'EvaluationSampleResultsStarted': ('', 'generation candidate_id revision'),
    'EvaluationSampleResultBatchRecorded': ('', 'candidate_id revision batch_index record_count result_digest'),
    'EvaluationSampleResultsRecorded': ('', 'candidate_id evaluation_id revision record_count result_digest'),
}


def _duplicate_evaluation_context(events):
    """Report later execution evidence without rewriting a duplicate event."""
    marked = {event.payload.get("candidate_id"): event.seq for event in events
              if event.kind == "CandidateMarkedDuplicate"}
    if not marked:
        return {}
    result = {}
    ranks = {"started": 1, "evaluation_recorded": 2, "screening_recorded": 3}
    for event in events:
        if event.kind not in {"EvaluationSampleResultsStarted", "CandidateScreeningRecorded", "EvaluationRecorded"}:
            continue
        source = event.payload.get("evaluation", {}) if event.kind == "EvaluationRecorded" else event.payload
        candidate_id = source.get("candidate_id")
        if candidate_id not in marked or event.seq <= marked[candidate_id]:
            continue
        disposition = {"EvaluationSampleResultsStarted": "started", "CandidateScreeningRecorded": "screening_recorded",
                       "EvaluationRecorded": "evaluation_recorded"}[event.kind]
        previous = result.get(candidate_id, {})
        if ranks[disposition] >= ranks.get(previous.get("evaluation_disposition"), 0):
            result[candidate_id] = {"evaluation_disposition": disposition, "evaluation_evidence_seq": event.seq}
    return result


class EventEndpointsMixin:
    @staticmethod
    def _event_json(
        event: Any,
        *,
        duplicate_evaluations: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if event.kind in _PRIVATE_SAMPLE_EVENT_KINDS:
            raise ValueError("private sample-result events cannot be publicly projected")
        payload = event.payload
        message = {
            "RunCreated": "进化运行实例已创建，任务配置已冻结。",
            "RunStarted": "进化运行实例已启动。",
            "RunPaused": "进化运行实例已暂停。",
            "RunResumed": "进化运行实例已恢复。",
            "RunCancelled": "进化运行实例已取消。",
            "RunFailed": "进化运行实例失败。",
            "RunCompleted": "进化运行实例已完成。",
            "GenerationAdvanced": "进化轮次已推进。",
            "GenerationBatchStarted": "本轮候选批次与共享上下文已冻结。",
            "GenerationSearchPlanned": "策略模型已生成并冻结本轮资料检索计划。",
            "GenerationKnowledgeRetrieved": "本轮公开知识与算法元数据已检索并冻结。",
            "GenerationKnowledgeAssessed": "本轮知识指导的联合搜索结果已完成非因果判断。",
            "GenerationResearchIterated": "本代研究模型已结合上一轮反馈更新并冻结研究计划。",
            "GenerationAnalyzed": "本轮候选结果与弱点已统一分析。",
            "GenerationReflected": "反思模型已基于本轮聚合结果生成下一轮多套方向。",
            "GenerationChampionSelected": "本轮下一轮有效搜索版本已冻结。",
            "CandidateRevisionCreated": "候选方案修订已冻结。",
            "FormalTrajectoryStarted": "入围候选的优化轨迹已启动。",
            "FormalBatchStarted": "正式轨迹的下一局部批次已启动。",
            "FormalBatchEvaluated": "局部批次评测已完成。",
            "FormalBatchCompared": "同 cohort 冠军—挑战者比较已完成。",
            "LocalEditProposalRecorded": "局部小改动提案已记录。",
            "LocalEditDecided": "局部小改动的宿主决策已记录。",
            "TrajectoryRevisionAdvanced": "下一批次的有效修订已冻结。",
            "FormalTrajectoryCompleted": "入围候选的优化轨迹已完成。",
            "GenerationHoldoutFrozen": "本轮独立留出评测队列已冻结。",
            "HoldoutArmStarted": "本轮一个留出评测臂已启动。",
            "HoldoutRecheckStarted": "本轮一个条件复核评测臂已启动。",
            "HoldoutEvaluationRecorded": "本轮一个留出评测臂已完成。",
            "GenerationComparisonRecorded": "本轮留出比较已完成。",
            "CandidateEffectiveRevisionFrozen": "本轮有效修订已冻结。",
            "RunAdaptationCohortFrozen": "本次运行的自适应队列已冻结。",
            "GenerationCohortsFrozen": "本轮初筛与独立留出队列已冻结。",
            "ProposalSubmitted": "变更提案已提交。",
            "CandidateSpawned": "候选方案已生成。",
            "CandidateFailed": "候选方案训练或评测失败。",
            "CandidateMarkedDuplicate": "候选已标记为本次运行中的等价方案；是否评测以本轮执行记录为准。",
            "ArtifactRecorded": "候选训练产物已记录。",
            "EvaluationRecorded": "候选方案科学评测已完成。",
            "EvaluationProgressRecorded": "候选方案正在逐微批执行真实样本评测。",
            "EvaluationSampleResultsStarted": "候选方案逐样本结果修订已开始。",
            "EvaluationSampleResultBatchRecorded": "候选方案已完成一批逐样本结果。",
            "EvaluationSampleResultsRecorded": "候选方案完整逐样本结果已封存。",
            "EvaluationJudged": "候选方案独立评审结论已记录。",
            "PromotionDecided": "候选方案搜索保留决策已记录。",
            "HumanInterventionRecorded": "人工意见已记录。",
            "HumanInterventionApplied": "人工意见执行状态已记录。",
            "ExpertConsultationRequested": "模型已提出非阻塞专家咨询。",
            "ExpertConsultationAnswered": "专家已异步答复模型咨询。",
            "ExpertConsultationApplied": "专家答复已纳入后续轮次的研究上下文。",
            "EvolutionStageRecorded": "进化阶段执行状态已记录。",
            "GatewayRetryScheduled": "网关暂时繁忙，已安排延迟重试。",
            "DshChildExecutionFailed": (
                "DSH 子任务请求失败并已释放预约；"
                "该请求不计为模型成功，最终失败 origin 由宿主结算记录。"
            ),
            "AlgorithmAttemptRecorded": "候选算法编译或调试证据已记录。",
        }.get(event.kind, "运行记录已更新。")
        public_payload: dict[str, Any] = {"message": message}
        if event.kind == "CandidateMarkedDuplicate":
            context = (duplicate_evaluations or {}).get(payload.get("candidate_id"), {})
            disposition = context.get("evaluation_disposition")
            if disposition in {"started", "screening_recorded", "evaluation_recorded"}:
                public_payload.update(context)
                public_payload["message"] = {
                    "started": "候选已标记为等价方案，随后仍启动了本轮评测；完成情况待确认。",
                    "screening_recorded": "候选已标记为等价方案，随后仍记录了本轮同组初筛结果；不能按跳过评测统计。",
                    "evaluation_recorded": "候选已标记为等价方案，随后仍记录了评测结果；不能按跳过评测统计。",
                }[disposition]
        if event.kind in _EVENT_FIELDS:
            path, fields = _EVENT_FIELDS[event.kind]
            source = payload.get(path, {}) if path else payload
            public_payload.update(public_fields(source, fields))
        elif event.kind == "GenerationAdvanced":
            public_payload["generation"] = payload.get("generation")
        elif event.kind == "GenerationSearchPlanned":
            search_plan = payload.get("search_plan", {})
            public_payload.update(
                {
                    **public_fields(search_plan, 'generation search_plan_digest source_analysis_digest source_reflection_digest'),
                    "search_queries": sanitize_public_value(
                        search_plan.get("search_queries", []),
                        max_depth=2,
                        text_limit=180,
                        sequence_limit=6,
                    ),
                    "focus_areas": sanitize_public_value(
                        search_plan.get("focus_areas", []),
                        max_depth=2,
                        text_limit=240,
                        sequence_limit=8,
                    ),
                }
            )
        elif event.kind == "GenerationKnowledgeRetrieved":
            snapshot = payload.get("knowledge_snapshot", {})
            cards = snapshot.get("cards", [])
            public_payload.update(
                {
                    **public_fields(snapshot, 'generation snapshot_digest retrieval_status'),
                    "source_count": len(cards) if isinstance(cards, list) else 0,
                    "adopted_count": sum(
                        1
                        for item in cards
                        if isinstance(item, dict)
                        and item.get("execution_status") == "adopted"
                    )
                    if isinstance(cards, list)
                    else 0,
                }
            )
        elif event.kind == "GenerationKnowledgeAssessed":
            assessment = payload.get("knowledge_assessment", {})
            public_payload.update(
                {
                    "generation": assessment.get("generation"),
                    "outcome": assessment.get("outcome"),
                    "conclusion": redact_sensitive_text(
                        str(assessment.get("conclusion", "")), limit=500
                    ),
                    "next_action": redact_sensitive_text(
                        str(assessment.get("next_action", "")), limit=500
                    ),
                }
            )
        elif event.kind == "GenerationResearchIterated":
            iteration = payload.get("research_iteration", {})
            historical_provenance = iteration.get("historical_provenance", {})
            public_payload.update(
                {
                    **public_fields(iteration, 'generation status iteration_digest knowledge_snapshot_digest source_analysis_digest'),
                    "historical_source_digest": (
                        historical_provenance.get("source_digest")
                        if isinstance(historical_provenance, dict)
                        else None
                    ),
                }
            )
        elif event.kind == "GenerationAnalyzed":
            analysis = payload.get("analysis", {})
            public_payload.update(
                {
                    **public_fields(analysis, 'generation analysis_digest candidate_count eligible_count outcome champion_candidate_id'),
                    "next_generation_focus": sanitize_public_value(
                        analysis.get("next_generation_focus"),
                        max_depth=4,
                        text_limit=500,
                        sequence_limit=16,
                    ),
                    "algorithm_failures": sanitize_public_value(
                        analysis.get("algorithm_failures", []),
                        max_depth=6,
                        text_limit=500,
                        sequence_limit=32,
                    ),
                    "sample_failures": sanitize_public_value(
                        analysis.get("sample_failures", []),
                        max_depth=6,
                        text_limit=500,
                        sequence_limit=32,
                    ),
                }
            )
        elif event.kind == "GenerationReflected":
            reflection = payload.get("reflection", {})
            directions = reflection.get("candidate_directions", [])
            public_payload.update(
                {
                    **public_fields(reflection, 'generation reflection_digest analysis_digest'),
                    "summary": redact_sensitive_text(
                        str(reflection.get("summary", "")), limit=1000
                    ),
                    "stop_recommendation": reflection.get(
                        "stop_recommendation"
                    ),
                    "direction_count": (
                        len(directions) if isinstance(directions, list) else 0
                    ),
                }
            )
        elif event.kind == "GenerationChampionSelected":
            public_payload.update(
                {
                    "generation": payload.get("generation"),
                    "outcome": payload.get("outcome"),
                    "champion_candidate_id": payload.get(
                        "champion_candidate_id",
                        payload.get("selected_candidate_id"),
                    ),
                    **public_fields(payload, 'selected_candidate_id selected_revision_id incumbent_after_candidate_id'),
                    "selection_reason": redact_sensitive_text(
                        str(payload.get("selection_reason", "")), limit=500
                    ),
                }
            )
        elif event.kind == "FormalBatchEvaluated":
            evaluation = payload.get("evaluation", {})
            scope = evaluation.get("scope", {})
            public_payload.update(
                {
                    "evaluation_id": evaluation.get("evaluation_id"),
                    **public_fields(scope, 'generation candidate_id candidate_revision_id batch_index origin_count cohort_digest'),
                    **public_fields(evaluation, 'score passed evaluator_digest'),
                }
            )
            if scope.get("formal_batch_arm") is not None:
                public_payload["formal_batch_arm"] = scope.get(
                    "formal_batch_arm"
                )
        elif event.kind == "FormalBatchCompared":
            comparison = payload.get("comparison", {})
            reason = redact_sensitive_text(
                str(comparison.get("reason", "")), limit=120
            )
            public_payload.update(
                {
                    **public_fields(comparison, 'comparison_id generation candidate_id batch_index cohort_digest champion_before_revision_id challenger_revision_id champion_score challenger_score score_delta minimum_score_delta safety_gate_passed cell_regression_gate_passed decision champion_after_revision_id'),
                    "reason": reason,
                }
            )
        elif event.kind == "LocalEditProposalRecorded":
            proposal = payload.get("proposal")
            if not isinstance(proposal, dict):
                proposal = payload
            operations = proposal.get("operations")
            public_payload.update(
                {
                    **public_fields(payload, 'proposal_id candidate_id batch_index evidence_scope_digest'),
                    "decision": proposal.get("decision"),
                    "operation_count": len(operations)
                    if isinstance(operations, list)
                    else 0,
                }
            )
        elif event.kind == "LocalEditDecided":
            raw_reason = payload.get("reason")
            public_payload.update(
                {
                    **public_fields(payload, 'proposal_id candidate_id batch_index outcome active_revision_id'),
                    "reason": (
                        redact_sensitive_text(str(raw_reason), limit=300)
                        if raw_reason is not None
                        else None
                    ),
                }
            )
        elif event.kind == "GenerationHoldoutFrozen":
            holdout = payload.get("holdout", {})
            arm_bindings = holdout.get("arm_bindings")
            public_payload.update(
                {
                    **public_fields(holdout, 'holdout_id generation cohort_digest origin_count'),
                    "arm_count": len(arm_bindings)
                    if isinstance(arm_bindings, dict)
                    else 0,
                }
            )
        elif event.kind == "HoldoutEvaluationRecorded":
            evaluation = payload.get("evaluation", {})
            scope = evaluation.get("scope", {})
            public_payload.update(
                {
                    "evaluation_id": evaluation.get("evaluation_id"),
                    **public_fields(scope, 'generation holdout_arm candidate_id candidate_revision_id origin_count cohort_digest inference_replica'),
                    **public_fields(evaluation, 'score passed evaluator_digest'),
                }
            )
        elif event.kind == "GenerationComparisonRecorded":
            comparison = payload.get("comparison", {})
            gates = comparison.get("gate_results", {})
            if not isinstance(gates, dict):
                gates = {}
            public_payload.update(
                {
                    **public_fields(comparison, 'comparison_id comparison_digest generation cohort_digest selected_candidate_id selected_revision_id'),
                    **public_fields(gates, 'selection_policy certification_selected_arm selected_search_certification_status delta_to_incumbent'),
                }
            )
        elif event.kind == "RunAdaptationCohortFrozen":
            adaptation = payload.get("adaptation", {})
            cohort = adaptation.get("cohort", {})
            batches = adaptation.get("batches")
            public_payload.update(
                {
                    **public_fields(adaptation, 'dataset_id episode_id adaptation_digest'),
                    "cohort_digest": cohort.get("cohort_digest"),
                    "origin_count": cohort.get("origin_count"),
                    "batch_count": len(batches) if isinstance(batches, list) else 0,
                }
            )
        elif event.kind == "GenerationCohortsFrozen":
            planned = payload.get("generation_cohorts", {})
            screening = planned.get("screening", {})
            holdout = planned.get("holdout", {})
            batch_digests = planned.get("adaptation_batch_digests")
            public_payload.update(
                {
                    **public_fields(planned, 'generation generation_cohorts_digest adaptation_digest'),
                    "adaptation_batch_count": len(batch_digests)
                    if isinstance(batch_digests, list)
                    else 0,
                    "screening_cohort_digest": screening.get("cohort_digest"),
                    "screening_origin_count": screening.get("origin_count"),
                    "holdout_cohort_digest": holdout.get("cohort_digest"),
                    "holdout_origin_count": holdout.get("origin_count"),
                }
            )
        elif event.kind == "ProposalSubmitted":
            proposal = payload.get("proposal", {})
            public_payload.update(
                {
                    "proposal_id": proposal.get("proposal_id"),
                    "title": redact_sensitive_text(
                        str(proposal.get("title", "")), limit=500
                    ),
                }
            )
        elif event.kind == "CandidateSpawned":
            candidate = payload.get("candidate", {})
            role = str(candidate.get("role") or "search")
            public_payload.update(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "role": (
                        role
                        if role in {"search", "incumbent_control"}
                        else "search"
                    ),
                }
            )
        elif event.kind == "CandidateFailed":
            public_payload.update(
                {
                    "candidate_id": payload.get("candidate_id"),
                    "reason": public_error_summary(
                        payload.get("reason", "评测失败")
                    ),
                }
            )
        elif event.kind == "EvaluationProgressRecorded":
            public_payload.update(
                {
                    **public_fields(payload, 'generation proposal_id candidate_id role model_id revision progress_id progress_kind batch_index batch_count batch_size completed_samples total_samples succeeded_samples failed_samples in_flight_batches queued_batches'),
                    "gateway_request_count": payload.get(
                        "gateway_request_count", payload.get("batch_index")
                    ),
                    "adaptive_split_trigger_count": payload.get(
                        "adaptive_split_trigger_count", 0
                    ),
                    "adaptive_split_count": payload.get("adaptive_split_count", 0),
                    "adaptive_split_max_depth": payload.get(
                        "adaptive_split_max_depth", 0
                    ),
                    "adaptive_split_recovered_samples": payload.get(
                        "adaptive_split_recovered_samples", 0
                    ),
                    "adaptive_split_failed_samples": payload.get(
                        "adaptive_split_failed_samples", 0
                    ),
                }
            )
        elif event.kind == "EvaluationJudged":
            evaluation = payload.get("evaluation", {})
            metrics = evaluation.get("metrics", {})
            public_payload.update(
                {
                    "candidate_id": evaluation.get("candidate_id"),
                    "passed": evaluation.get("passed"),
                    "judge_status": metrics.get("judge_status"),
                    "judge_accepted": metrics.get("judge_accepted"),
                }
            )
        elif event.kind == "PromotionDecided":
            promotion = payload.get("promotion", {})
            public_payload.update(
                {
                    "candidate_id": promotion.get("candidate_id"),
                    "decision": promotion.get("decision"),
                    "reason": redact_sensitive_text(
                        str(promotion.get("reason", "")), limit=500
                    ),
                }
            )
        elif event.kind == "EvolutionStageRecorded":
            public_payload.update(
                {
                    **public_fields(payload, 'generation stage status attempt proposal_id candidate_id'),
                    "public_error": public_error_summary(
                        payload.get("public_error")
                    ),
                }
            )
        elif event.kind == "GatewayRetryScheduled":
            public_payload.update(
                {
                    **public_fields(payload, 'generation retry_at delay_seconds attempt error_code'),
                    "reason": redact_sensitive_text(str(payload.get("reason") or "网关暂时繁忙，等待后重试"), limit=240),
                }
            )
            if (
                payload.get("schema_version")
                == "ecologyrsi-dsh.gateway-retry-scheduled/2"
            ):
                public_payload.update(
                    {
                        **public_fields(payload, 'retry_class stage breaker_epoch consecutive_failures retry_limit first_failure_at last_failure_at last_error_code'),
                        "suggested_action": "wait_for_scheduled_retry",
                    }
                )
        elif event.kind == "DshChildExecutionFailed":
            identity = payload.get("identity", {})
            public_payload.update(
                {
                    "stage": identity.get("stage"),
                    "error_code": payload.get("error_code"),
                }
            )
        elif event.kind == "RunPaused":
            public_payload.update(
                {
                    "code": payload.get("code"),
                    "reason": (
                        redact_sensitive_text(str(payload.get("reason")), limit=500)
                        if payload.get("reason") is not None
                        else None
                    ),
                }
            )
            if payload.get("code") in {
                "gateway_retry_circuit_open",
                "dsh_runtime_retry_circuit_open",
                "research_timeout_retry_circuit_open",
                "sample_persistence_retry_circuit_open",
            }:
                public_payload.update(
                    {
                        **public_fields(payload, 'retry_class generation stage breaker_epoch consecutive_failures retry_limit first_failure_at last_failure_at last_error_code suggested_action'),
                    }
                )
        elif event.kind == "AlgorithmAttemptRecorded":
            attempt = payload.get("algorithm_attempt", {})
            public_payload.update(
                {
                    **public_fields(attempt, 'generation proposal_id candidate_id phase attempt status algorithm_spec_digest failure_code'),
                    "public_error": public_error_summary(
                        attempt.get("public_error")
                    ),
                }
            )
        return {
            "seq": event.seq,
            "id": event.event_id,
            "event_id": event.event_id,
            "run_id": event.run_id,
            "type": _event_type(event.kind),
            "event_type": _event_type(event.kind),
            "kind": event.kind,
            "payload": public_payload,
            "occurred_at": event.created_at,
            "created_at": event.created_at,
        }

    def _events_payload(self, run_id: str) -> dict[str, Any]:
        from ..application.queries import RunQueries
        return RunQueries(self.server.director).completed_read(
            run_id, lambda state: self._events_payload_from_state(run_id, state),
            cache_key='public-events:' + urlparse(self.path).query)

    def _events_payload_from_state(self, run_id: str, state) -> dict[str, Any]:
        _assert_http_scope(state)
        query = parse_qs(urlparse(self.path).query)
        raw_after = query.get("after", ["0"])[0]
        try:
            after = int(raw_after)
        except (TypeError, ValueError) as exc:
            raise ValueError("after must be a non-negative integer cursor") from exc
        if after < 0:
            raise ValueError("after must be a non-negative integer cursor")
        raw_tail = query.get("tail", [None])[0]
        tail: int | None = None
        if raw_tail is not None:
            try:
                tail = int(raw_tail)
            except (TypeError, ValueError) as exc:
                raise ValueError("tail must be an integer between 1 and 500") from exc
            if not 1 <= tail <= 500:
                raise ValueError("tail must be an integer between 1 and 500")
        # State already contains the validated immutable stream. A tail read
        # must not fetch and JSON-decode the entire ledger a second time.
        events = [event for event in state.events if event.seq > after]
        next_cursor = events[-1].seq if events else after
        public_events = [
            event for event in events if event.kind not in _PRIVATE_SAMPLE_EVENT_KINDS
        ]
        total_public_events = sum(
            event.kind not in _PRIVATE_SAMPLE_EVENT_KINDS for event in state.events
        )
        truncated = tail is not None and len(public_events) > tail
        if tail is not None:
            public_events = public_events[-tail:]
        duplicate_evaluations = _duplicate_evaluation_context(state.events)
        return {
            "run_id": run_id,
            "events": [
                self._event_json(
                    event,
                    duplicate_evaluations=duplicate_evaluations,
                )
                for event in public_events
            ],
            "after": after,
            "cursor": str(next_cursor),
            "next_cursor": str(next_cursor),
            "has_more": False,
            "total_public_events": total_public_events,
            "truncated": truncated,
        }

    def _cached_sample_result_rows(
        self,
        event: Any,
        *,
        batch: bool,
    ) -> tuple[dict[str, Any], ...]:
        """Decode each immutable archive once within a bounded process LRU."""

        result_digest = str(event.payload.get("result_digest") or "")
        cache_key = (event.event_id, result_digest)
        with self.server.sample_result_cache_lock:
            cached = self.server.sample_result_cache.get(cache_key)
            if cached is not None:
                self.server.sample_result_cache.move_to_end(cache_key)
                return cached[1]
        rows = (
            decode_sample_result_batch(event.payload)
            if batch
            else decode_sample_results(event.payload)
        )
        archive = event.payload.get("archive")
        weight = (
            int(archive.get("uncompressed_bytes", 0))
            if isinstance(archive, dict)
            else 0
        )
        if not 0 < weight <= self.server.sample_result_cache_max_bytes:
            return rows
        with self.server.sample_result_cache_lock:
            existing = self.server.sample_result_cache.get(cache_key)
            if existing is not None:
                self.server.sample_result_cache.move_to_end(cache_key)
                return existing[1]
            while (
                self.server.sample_result_cache
                and self.server.sample_result_cache_bytes + weight
                > self.server.sample_result_cache_max_bytes
            ):
                _old_key, (old_weight, _old_rows) = (
                    self.server.sample_result_cache.popitem(last=False)
                )
                self.server.sample_result_cache_bytes -= old_weight
            self.server.sample_result_cache[cache_key] = (weight, rows)
            self.server.sample_result_cache_bytes += weight
        return rows

    def _sample_results_payload(self, run_id: str) -> dict[str, Any]:
        """Return one candidate's host-finalized rows with bounded pagination."""
        from ..application.queries import RunQueries
        return RunQueries(self.server.director).completed_read(
            run_id, lambda state: self._sample_results_payload_from_state(run_id, state),
            cache_key='public-samples:' + urlparse(self.path).query)

    def _sample_results_payload_from_state(self, run_id: str, state) -> dict[str, Any]:
        _assert_http_scope(state)
        query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
        unknown = set(query) - {"candidate_id", "offset", "limit"}
        if unknown:
            raise ValueError(
                "unknown sample results query: " + ", ".join(sorted(unknown))
            )
        candidate_values = query.get("candidate_id", [])
        if (
            len(candidate_values) != 1
            or not isinstance(candidate_values[0], str)
            or not candidate_values[0].strip()
        ):
            raise ValueError("candidate_id must be supplied exactly once")
        candidate_id = candidate_values[0].strip()
        candidate = state.candidate(candidate_id)
        try:
            offset = int(query.get("offset", ["0"])[0])
            limit = int(query.get("limit", ["50"])[0])
        except (TypeError, ValueError) as exc:
            raise ValueError("offset and limit must be integers") from exc
        if offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")

        starts = [
            event
            for event in state.events
            if event.kind == "EvaluationSampleResultsStarted"
            and event.payload.get("candidate_id") == candidate_id
        ]
        start = starts[-1] if starts else None
        revision = (
            str(start.payload.get("revision"))
            if start is not None and start.payload.get("revision") is not None
            else None
        )
        completed_events = [
            event
            for event in state.events
            if event.kind == "EvaluationSampleResultsRecorded"
            and event.payload.get("candidate_id") == candidate_id
            and revision is not None
            and event.payload.get("revision") == revision
        ]
        completed = completed_events[-1] if completed_events else None

        rows: list[dict[str, Any]] = []
        terminal = (
            candidate.status is not CandidateStatus.SPAWNED
            or state.run.status in _TERMINAL_RUN_STATUSES
        )
        expected_count: int | None = None
        batch_events: list[Any] = []
        if completed is not None:
            evaluation_id = completed.payload.get("evaluation_id")
            scientific_event_exists = any(
                event.kind == "EvaluationRecorded"
                and event.payload.get("evaluation", {}).get("candidate_id")
                == candidate_id
                and event.payload.get("evaluation", {}).get("evaluation_id")
                == evaluation_id
                for event in state.events
            )
            if not scientific_event_exists:
                # Screening archives are sealed against CandidateScreeningRecorded,
                # not EvaluationRecorded. Bind the explicit phase, generation,
                # frozen cohort and cell count before exposing their sample page.
                scientific_event_exists = _screening_completion_matches(state, candidate, start, completed)
            if not scientific_event_exists:
                raise ValueError(
                    "sample results completion is missing its scientific evaluation"
                )
            rows = [
                dict(row)
                for row in self._cached_sample_result_rows(completed, batch=False)
            ]
            revision = str(completed.payload["revision"])
            expected_count = int(completed.payload["record_count"])
        elif revision is not None:
            batch_events = sorted(
                (
                    event
                    for event in state.events
                    if event.kind == "EvaluationSampleResultBatchRecorded"
                    and event.payload.get("candidate_id") == candidate_id
                    and event.payload.get("revision") == revision
                ),
                key=lambda event: (int(event.payload.get("batch_index", 0)), event.seq),
            )
            seen_ids: set[str] = set()
            seen_indices: set[int] = set()
            for event in batch_events:
                for row in self._cached_sample_result_rows(event, batch=True):
                    sample_id = str(row["sample_id"])
                    sample_index = int(row["sample_index"])
                    if sample_id in seen_ids:
                        raise ValueError(
                            "sample result revision contains a duplicate sample_id"
                        )
                    if sample_index in seen_indices:
                        raise ValueError(
                            "sample result revision contains a duplicate sample_index"
                        )
                    seen_ids.add(sample_id)
                    seen_indices.add(sample_index)
                    rows.append(dict(row))
            progress_events = [
                event
                for event in state.events
                if start is not None
                and event.seq > start.seq
                and event.kind == "EvaluationProgressRecorded"
                and event.payload.get("candidate_id") == candidate_id
                and event.payload.get("role") == "planner"
            ]
            if progress_events:
                expected_count = int(progress_events[-1].payload["total_samples"])

        rows.sort(key=lambda row: (int(row["sample_index"]), str(row["sample_id"])))
        unavailable = revision is None and terminal
        if completed is not None:
            result_status = "completed"
        elif revision is not None and terminal:
            result_status = "aborted"
        elif revision is not None and state.run.status is RunStatus.PAUSED:
            result_status = "paused"
        elif revision is not None:
            result_status = "running"
        elif unavailable:
            result_status = "unavailable"
        else:
            result_status = "pending"
        complete = completed is not None or terminal

        total = len(rows)
        page_rows = [
            _browser_sample_result_row(row)
            for row in rows[offset : offset + limit]
        ]
        consumed = offset + len(page_rows)
        reward_definition = (
            str(completed.payload.get("reward_definition"))
            if completed is not None
            and completed.payload.get("reward_definition") is not None
            else str(batch_events[-1].payload.get("reward_definition"))
            if revision is not None
            and batch_events
            and batch_events[-1].payload.get("reward_definition") is not None
            else SAMPLE_REWARD_DEFINITION
        )
        return {
            "schema_version": "ecologyrsi-dsh.browser-sample-results/1",
            "run_id": run_id,
            "reward_definition": reward_definition,
            "positive_is_better": True,
            "candidate_id": candidate_id,
            "supported": not unavailable,
            "status": result_status,
            "rows": page_rows,
            "offset": offset,
            "limit": limit,
            "total": total,
            "available_count": total,
            "expected_count": expected_count,
            "partial": completed is None and bool(rows),
            "has_more": consumed < total,
            "next_offset": consumed if consumed < total else None,
            "complete": complete,
            "revision": revision,
        }
