"""Public, redacted event projections for browser clients."""

from __future__ import annotations

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
    SAMPLE_REWARD_DEFINITION_V1,
    decode_sample_result_batch,
    decode_sample_results,
)
from .projection import _public_intervention_receipt
from .shared import _assert_http_scope, _event_type


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


class EventEndpointsMixin:
    @staticmethod
    def _event_json(
        event: Any,
        *,
        intervention_kinds: dict[str, str] | None = None,
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
            "GenerationKnowledgeRetrieved": "本轮公开知识与算法元数据已检索并冻结。",
            "GenerationKnowledgeAssessed": "本轮知识指导的联合搜索结果已完成非因果判断。",
            "GenerationResearchIterated": "本代研究模型已结合上一轮反馈更新并冻结研究计划。",
            "GenerationAnalyzed": "本轮候选结果与弱点已统一分析。",
            "GenerationChampionSelected": "本轮单一冠军选择已完成。",
            "ProposalSubmitted": "变更提案已提交。",
            "CandidateSpawned": "候选方案已生成。",
            "CandidateFailed": "候选方案训练或评测失败。",
            "CandidateMarkedDuplicate": "候选与本次运行已有等价方案重复，已跳过评测。",
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
            "ModelUsageRecorded": "模型调用 token 用量已记录。",
            "AlgorithmAttemptRecorded": "候选算法编译或调试证据已记录。",
        }.get(event.kind, "运行记录已更新。")
        public_payload: dict[str, Any] = {"message": message}
        if event.kind == "GenerationAdvanced":
            public_payload["generation"] = payload.get("generation")
        elif event.kind == "GenerationBatchStarted":
            batch = payload.get("batch", {})
            public_payload.update(
                {
                    "generation": batch.get("generation"),
                    "batch_size": batch.get("batch_size"),
                    "parent_candidate_id": batch.get("parent_candidate_id"),
                    "context_digest": batch.get("context_digest"),
                }
            )
        elif event.kind == "GenerationKnowledgeRetrieved":
            snapshot = payload.get("knowledge_snapshot", {})
            cards = snapshot.get("cards", [])
            public_payload.update(
                {
                    "generation": snapshot.get("generation"),
                    "snapshot_digest": snapshot.get("snapshot_digest"),
                    "retrieval_status": snapshot.get("retrieval_status"),
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
            adoption = iteration.get("prediction_model_adoption", {})
            historical_provenance = iteration.get("historical_provenance", {})
            public_payload.update(
                {
                    "generation": iteration.get("generation"),
                    "status": iteration.get("status"),
                    "iteration_digest": iteration.get("iteration_digest"),
                    "knowledge_snapshot_digest": iteration.get(
                        "knowledge_snapshot_digest"
                    ),
                    "source_analysis_digest": iteration.get(
                        "source_analysis_digest"
                    ),
                    "historical_source_digest": (
                        historical_provenance.get("source_digest")
                        if isinstance(historical_provenance, dict)
                        else None
                    ),
                    "adopted_predictor_id": adoption.get("adopted_id"),
                }
            )
        elif event.kind == "GenerationAnalyzed":
            analysis = payload.get("analysis", {})
            public_payload.update(
                {
                    "generation": analysis.get("generation"),
                    "candidate_count": analysis.get("candidate_count"),
                    "eligible_count": analysis.get("eligible_count"),
                    "outcome": analysis.get("outcome"),
                    "champion_candidate_id": analysis.get("champion_candidate_id"),
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
        elif event.kind == "GenerationChampionSelected":
            public_payload.update(
                {
                    "generation": payload.get("generation"),
                    "outcome": payload.get("outcome"),
                    "champion_candidate_id": payload.get("champion_candidate_id"),
                    "incumbent_after_candidate_id": payload.get("incumbent_after_candidate_id"),
                    "selection_reason": redact_sensitive_text(
                        str(payload.get("selection_reason", "")), limit=500
                    ),
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
            public_payload["candidate_id"] = candidate.get("candidate_id")
        elif event.kind == "CandidateFailed":
            public_payload.update(
                {
                    "candidate_id": payload.get("candidate_id"),
                    "reason": public_error_summary(
                        payload.get("reason", "评测失败")
                    ),
                }
            )
        elif event.kind == "CandidateMarkedDuplicate":
            public_payload.update(
                {
                    "candidate_id": payload.get("candidate_id"),
                    "duplicate_of_candidate_id": payload.get("duplicate_of_candidate_id"),
                }
            )
        elif event.kind == "ArtifactRecorded":
            artifact = payload.get("artifact", {})
            public_payload.update(
                {
                    "artifact_id": artifact.get("artifact_id"),
                    "candidate_id": artifact.get("candidate_id"),
                }
            )
        elif event.kind == "EvaluationRecorded":
            evaluation = payload.get("evaluation", {})
            public_payload.update(
                {
                    "candidate_id": evaluation.get("candidate_id"),
                    "score": evaluation.get("score"),
                    "passed": evaluation.get("passed"),
                }
            )
        elif event.kind == "EvaluationProgressRecorded":
            public_payload.update(
                {
                    "generation": payload.get("generation"),
                    "proposal_id": payload.get("proposal_id"),
                    "candidate_id": payload.get("candidate_id"),
                    "role": payload.get("role"),
                    "model_id": payload.get("model_id"),
                    "revision": payload.get("revision"),
                    "progress_id": payload.get("progress_id"),
                    "progress_kind": payload.get("progress_kind"),
                    "batch_index": payload.get("batch_index"),
                    "batch_count": payload.get("batch_count"),
                    "batch_size": payload.get("batch_size"),
                    "completed_samples": payload.get("completed_samples"),
                    "total_samples": payload.get("total_samples"),
                    "succeeded_samples": payload.get("succeeded_samples"),
                    "failed_samples": payload.get("failed_samples"),
                    "in_flight_batches": payload.get("in_flight_batches"),
                    "queued_batches": payload.get("queued_batches"),
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
        elif event.kind == "EvaluationSampleResultsStarted":
            public_payload.update(
                {
                    "generation": payload.get("generation"),
                    "candidate_id": payload.get("candidate_id"),
                    "revision": payload.get("revision"),
                }
            )
        elif event.kind == "EvaluationSampleResultBatchRecorded":
            public_payload.update(
                {
                    "candidate_id": payload.get("candidate_id"),
                    "revision": payload.get("revision"),
                    "batch_index": payload.get("batch_index"),
                    "record_count": payload.get("record_count"),
                    "result_digest": payload.get("result_digest"),
                }
            )
        elif event.kind == "EvaluationSampleResultsRecorded":
            public_payload.update(
                {
                    "candidate_id": payload.get("candidate_id"),
                    "evaluation_id": payload.get("evaluation_id"),
                    "revision": payload.get("revision"),
                    "record_count": payload.get("record_count"),
                    "result_digest": payload.get("result_digest"),
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
        elif event.kind == "HumanInterventionRecorded":
            intervention = payload.get("intervention", {})
            public_payload.update(
                {
                    "intervention_id": intervention.get("intervention_id"),
                    "kind": intervention.get("kind"),
                }
            )
        elif event.kind == "HumanInterventionApplied":
            intervention_id = str(payload.get("intervention_id", ""))
            kind = str(
                payload.get("kind")
                or (intervention_kinds or {}).get(intervention_id, "")
            )
            receipt = _public_intervention_receipt(payload, kind=kind)
            public_payload.update(
                {
                    "message": {
                        "recorded": "人工意见仅记录到本轮提案，未执行。",
                        "applied": "人工意见已应用到本轮提案。",
                        "enforced": "人工意见已由宿主边界强制执行。",
                    }[receipt["application_status"]],
                    "intervention_id": intervention_id,
                    "proposal_id": payload.get("proposal_id"),
                    "kind": kind or None,
                    **receipt,
                }
            )
        elif event.kind == "ExpertConsultationRequested":
            consultation = payload.get("consultation", {})
            generation = consultation.get("generation")
            public_payload.update(
                {
                    "consultation_id": consultation.get("consultation_id"),
                    "source_generation": (
                        generation + 1
                        if isinstance(generation, int)
                        and not isinstance(generation, bool)
                        and generation >= 0
                        else None
                    ),
                    "uncertainty_type": sanitize_public_value(
                        consultation.get("uncertainty_type"),
                        text_limit=200,
                    ),
                    "question": sanitize_public_value(
                        consultation.get("question"),
                        text_limit=4000,
                    ),
                    "context": sanitize_public_value(
                        consultation.get("context"),
                        text_limit=4000,
                    ),
                    "fallback_assumption": sanitize_public_value(
                        consultation.get("fallback_assumption"),
                        text_limit=4000,
                    ),
                    "requested_expertise": sanitize_public_value(
                        consultation.get("requested_expertise", []),
                        text_limit=500,
                        sequence_limit=16,
                    ),
                    "options": sanitize_public_value(
                        consultation.get("options", []),
                        text_limit=500,
                        sequence_limit=16,
                    ),
                    "confidence": sanitize_public_value(
                        consultation.get("confidence")
                    ),
                    "candidate_id": consultation.get("candidate_id"),
                    "model_id": sanitize_public_value(
                        consultation.get("requested_by_model_id"),
                        text_limit=500,
                    ),
                    "non_blocking": consultation.get("non_blocking") is True,
                }
            )
        elif event.kind == "ExpertConsultationAnswered":
            answer = payload.get("answer", {})
            effective_generation = answer.get("effective_generation")
            public_payload.update(
                {
                    "consultation_id": answer.get("consultation_id"),
                    "answer_id": answer.get("answer_id"),
                    "answer": sanitize_public_value(
                        answer.get("answer"),
                        text_limit=4000,
                    ),
                    "answered_by": sanitize_public_value(
                        answer.get("answered_by"),
                        text_limit=120,
                    ),
                    "selected_option": sanitize_public_value(
                        answer.get("selected_option"),
                        text_limit=500,
                    ),
                    "effective_generation": (
                        effective_generation + 1
                        if isinstance(effective_generation, int)
                        and not isinstance(effective_generation, bool)
                        and effective_generation >= 0
                        else None
                    ),
                    "audit_only": effective_generation is None,
                }
            )
        elif event.kind == "ExpertConsultationApplied":
            application = payload.get("application", payload)
            applied_generation = application.get(
                "applied_generation",
                application.get("generation"),
            )
            public_payload.update(
                {
                    "consultation_id": application.get("consultation_id"),
                    "answer_id": application.get("answer_id"),
                    "research_iteration_digest": application.get(
                        "research_iteration_digest",
                        application.get("iteration_digest"),
                    ),
                    "applied_generation": (
                        applied_generation + 1
                        if isinstance(applied_generation, int)
                        and not isinstance(applied_generation, bool)
                        and applied_generation >= 0
                        else None
                    ),
                }
            )
        elif event.kind == "EvolutionStageRecorded":
            public_payload.update(
                {
                    "generation": payload.get("generation"),
                    "stage": payload.get("stage"),
                    "status": payload.get("status"),
                    "attempt": payload.get("attempt"),
                    "proposal_id": payload.get("proposal_id"),
                    "candidate_id": payload.get("candidate_id"),
                    "public_error": public_error_summary(
                        payload.get("public_error")
                    ),
                }
            )
        elif event.kind == "GatewayRetryScheduled":
            public_payload.update(
                {
                    "generation": payload.get("generation"),
                    "retry_at": payload.get("retry_at"),
                    "delay_seconds": payload.get("delay_seconds"),
                    "attempt": payload.get("attempt"),
                    "error_code": payload.get("error_code"),
                    "reason": redact_sensitive_text(str(payload.get("reason") or "网关暂时繁忙，等待后重试"), limit=240),
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
        elif event.kind == "ModelUsageRecorded":
            public_payload.update(
                {
                    "schema_version": payload.get("schema_version"),
                    "generation": payload.get("generation"),
                    "candidate_id": payload.get("candidate_id"),
                    "role": payload.get("role"),
                    "model_id": payload.get("model_id"),
                    "prompt_tokens": payload.get("prompt_tokens"),
                    "completion_tokens": payload.get("completion_tokens"),
                    "total_tokens": payload.get("total_tokens"),
                    "usage_reported": payload.get("usage_reported", True),
                    "outcome": payload.get("outcome", "succeeded"),
                    "call_count": 1,
                    "http_attempts": payload.get(
                        "http_attempts", payload.get("gateway_request_count")
                    ),
                    "revision": payload.get("revision"),
                    "usage_index": payload.get("usage_index"),
                }
            )
        elif event.kind == "AlgorithmAttemptRecorded":
            attempt = payload.get("algorithm_attempt", {})
            public_payload.update(
                {
                    "generation": attempt.get("generation"),
                    "proposal_id": attempt.get("proposal_id"),
                    "candidate_id": attempt.get("candidate_id"),
                    "phase": attempt.get("phase"),
                    "attempt": attempt.get("attempt"),
                    "status": attempt.get("status"),
                    "algorithm_spec_digest": attempt.get("algorithm_spec_digest"),
                    "failure_code": attempt.get("failure_code"),
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
        if run_id not in self.server.ledger.run_ids():
            raise KeyError(f"unknown run: {run_id}")
        state = self.server.director.state(run_id)
        _assert_http_scope(state)
        query = parse_qs(urlparse(self.path).query)
        raw_after = query.get("after", ["0"])[0]
        try:
            after = int(raw_after)
        except (TypeError, ValueError) as exc:
            raise ValueError("after must be a non-negative integer cursor") from exc
        if after < 0:
            raise ValueError("after must be a non-negative integer cursor")
        events = self.server.ledger.events(run_id, after_seq=after)
        next_cursor = events[-1].seq if events else after
        public_events = [
            event for event in events if event.kind not in _PRIVATE_SAMPLE_EVENT_KINDS
        ]
        intervention_kinds = {
            item.intervention_id: item.kind.value for item in state.interventions
        }
        return {
            "run_id": run_id,
            "events": [
                self._event_json(
                    event,
                    intervention_kinds=intervention_kinds,
                )
                for event in public_events
            ],
            "after": after,
            "cursor": str(next_cursor),
            "next_cursor": str(next_cursor),
            "has_more": False,
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

        if run_id not in self.server.ledger.run_ids():
            raise KeyError(f"unknown run: {run_id}")
        state = self.server.director.state(run_id)
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
        legacy = revision is None and terminal
        supported = not legacy
        if completed is not None:
            result_status = "completed"
        elif revision is not None and terminal:
            result_status = "aborted"
        elif revision is not None and state.run.status is RunStatus.PAUSED:
            result_status = "paused"
        elif revision is not None:
            result_status = "running"
        elif legacy:
            result_status = "legacy"
        else:
            result_status = "pending"
        complete = completed is not None or terminal

        total = len(rows)
        page_rows = rows[offset : offset + limit]
        consumed = offset + len(page_rows)
        reward_definition = (
            str(completed.payload.get("reward_definition"))
            if completed is not None
            and completed.payload.get("reward_definition") is not None
            else str(batch_events[-1].payload.get("reward_definition"))
            if revision is not None
            and batch_events
            and batch_events[-1].payload.get("reward_definition") is not None
            else SAMPLE_REWARD_DEFINITION_V1
            if legacy
            else SAMPLE_REWARD_DEFINITION
        )
        return {
            "schema_version": "ecologyrsi-dsh.browser-sample-results/1",
            "run_id": run_id,
            "reward_definition": reward_definition,
            "positive_is_better": True,
            "candidate_id": candidate_id,
            "supported": supported,
            "legacy": legacy,
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
