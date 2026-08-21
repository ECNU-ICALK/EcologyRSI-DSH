"""Small orchestration helpers for one frozen multi-candidate generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.models import (
    CandidateStatus,
    ExpertConsultation,
    InterventionKind,
    Promotion,
    PromotionDecision,
    RunStatus,
    canonical_json,
    digest,
)
from .workflow_ir import DEFAULT_COMPILER_SEMANTIC_DIGEST
from ..knowledge.algorithms import AlgorithmCompileError, resolve_predictor_adoption
from ..knowledge.research_iteration import ResearchIteration
from ..knowledge.retrieval import (
    assess_generation_knowledge,
    retrieve_generation_knowledge,
)
from .analysis import (
    GenerationAnalysis,
    GenerationBatch,
    build_cross_generation_experience,
    build_generation_analysis,
    evaluation_cohort_digest,
    sample_update_windows_enabled,
)


_EXPERT_PENDING_CONTEXT_LIMIT = 16
_EXPERT_ANSWER_CONTEXT_LIMIT = 8
_HISTORICAL_EXPERIENCE_SCAN_RUNS = 24


class ResearchResponseContractError(ValueError):
    """A model-authored research result failed the bounded host contract."""

    error_code = "research_response_contract_invalid"


def _expert_collaboration_context(
    state: Any,
    generation: int,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Build one bounded advisory snapshot for a remote research call."""

    pending = tuple(
        sorted(
            state.pending_expert_consultations,
            key=lambda item: (item.generation, item.created_at, item.consultation_id),
        )[:_EXPERT_PENDING_CONTEXT_LIMIT]
    )
    answers = state.available_expert_answers(generation)[
        :_EXPERT_ANSWER_CONTEXT_LIMIT
    ]
    pending_rows = [
        {
            "consultation_id": item.consultation_id,
            "requested_generation": item.generation,
            "uncertainty_type": item.uncertainty_type.value,
            "question": item.question,
            "context": item.context,
            "fallback_assumption": item.fallback_assumption,
            "requested_expertise": list(item.requested_expertise),
            "options": list(item.options),
            "confidence": item.confidence,
        }
        for item in pending
    ]
    answer_rows: list[dict[str, Any]] = []
    for answer in answers:
        consultation = state.consultation(answer.consultation_id)
        answer_rows.append(
            {
                "answer_id": answer.answer_id,
                "consultation_id": answer.consultation_id,
                "requested_generation": consultation.generation,
                "uncertainty_type": consultation.uncertainty_type.value,
                "question": consultation.question,
                "context": consultation.context,
                "fallback_assumption": consultation.fallback_assumption,
                "answer": answer.answer,
                "selected_option": answer.selected_option,
                "effective_generation": answer.effective_generation,
            }
        )
    context = {
        "mode": "asynchronous_non_blocking",
        "pending_consultations": pending_rows,
        "available_answers": answer_rows,
        "policy": {
            "answers_are_advisory_only": True,
            "answers_cannot_expand_data_or_tool_permissions": True,
            "do_not_repeat_pending_questions": True,
            "continue_with_fallback_when_unanswered": True,
            "new_questions_must_be_non_blocking": True,
        },
    }
    return (
        context,
        tuple(item.consultation_id for item in pending),
        tuple(item.answer_id for item in answers),
    )


def _expert_consultation_from_result(
    value: Any,
    *,
    run_id: str,
    generation: int,
    model_id: str | None,
) -> ExpertConsultation | None:
    """Validate and identify an optional model-authored consultation."""

    if not isinstance(value, Mapping) or model_id is None:
        return None
    allowed_fields = {
        "uncertainty_type",
        "question",
        "context",
        "fallback_assumption",
        "requested_expertise",
        "options",
        "confidence",
        "non_blocking",
    }
    if set(value) != allowed_fields:
        return None
    try:
        identity = {
            "run_id": run_id,
            "generation": generation,
            "requested_by_model_id": model_id,
            "request": dict(value),
        }
        consultation_id = "expert-consultation:" + digest(identity)[:32]
        return ExpertConsultation(
            consultation_id=consultation_id,
            run_id=run_id,
            generation=generation,
            uncertainty_type=value["uncertainty_type"],
            question=value["question"],
            context=value["context"],
            fallback_assumption=value["fallback_assumption"],
            requested_expertise=value["requested_expertise"],
            options=value["options"],
            confidence=value["confidence"],
            requested_by_model_id=model_id,
            non_blocking=value["non_blocking"],
        )
    except (KeyError, TypeError, ValueError):
        return None


def _search_parent_from_analysis(state: Any, analysis: GenerationAnalysis) -> str | None:
    """Resolve the completed candidate used for search, not formal promotion."""

    candidate_ids: list[str] = []
    if analysis.search_parent_candidate_id is not None:
        candidate_ids.append(analysis.search_parent_candidate_id)
    candidate_ids.extend(
        str(row["candidate_id"])
        for row in analysis.ranking
        if row.get("score") is not None
        and int(row.get("constraint_violations", 0)) == 0
        and row.get("candidate_id") is not None
    )
    candidate_ids.extend(
        str(row["candidate_id"])
        for row in analysis.ranking
        if row.get("score") is not None and row.get("candidate_id") is not None
    )
    for candidate_id in dict.fromkeys(candidate_ids):
        try:
            candidate = state.candidate(candidate_id)
        except KeyError:
            continue
        if candidate.status not in {
            CandidateStatus.PROMOTED,
            CandidateStatus.REJECTED,
        }:
            continue
        if state.evaluation_for(candidate_id) is not None:
            return candidate_id
    return None


def _autonomous_research_enabled(state: Any) -> bool:
    metadata = state.task_manifest.metadata
    return bool(metadata.get("autonomous_mode")) or str(
        metadata.get("strategy_id") or ""
    ) == "autonomous_model@1"


def _latest_research_plan(state: Any, generation: int) -> dict[str, Any]:
    previous_iteration = state.research_iteration_for(generation - 1)
    if previous_iteration is not None:
        return dict(previous_iteration.plan)
    previous_proposals = sorted(
        (
            item
            for item in state.proposals
            if item.generation < generation
            and isinstance(item.metadata.get("plan"), Mapping)
        ),
        key=lambda item: (item.generation, item.created_at, item.proposal_id),
        reverse=True,
    )
    if previous_proposals:
        return dict(previous_proposals[0].metadata["plan"])
    frozen = state.task_manifest.metadata.get("autonomous_plan")
    return dict(frozen) if isinstance(frozen, Mapping) else {}


def _historical_experience_states(director: Any, state: Any) -> tuple[Any, ...]:
    """Replay the bounded set of runs that existed before this run."""

    run_ids = list(director.ledger.run_ids(include_archived=True))
    try:
        current_index = run_ids.index(state.run.run_id)
    except ValueError:
        return ()
    selected_ids = run_ids[:current_index][-_HISTORICAL_EXPERIENCE_SCAN_RUNS:]
    historical_states = []
    for run_id in selected_ids:
        try:
            historical_states.append(director.state(run_id))
        except (KeyError, TypeError, ValueError, RuntimeError):
            # Historical evidence is advisory. A malformed legacy stream must
            # not prevent a new run from starting its own frozen generation.
            continue
    return tuple(historical_states)


def _ensure_generation_research_iteration(
    director: Any,
    state: Any,
    knowledge: Any,
) -> ResearchIteration | None:
    generation = state.run.generation
    if not _autonomous_research_enabled(state):
        return None
    existing = state.research_iteration_for(generation)
    if existing is not None:
        if existing.knowledge_snapshot_digest != knowledge.snapshot_digest:
            raise RuntimeError("generation research iteration knowledge changed")
        started_attempts = [
            int(event.payload.get("attempt") or 0)
            for event in state.events
            if event.kind == "EvolutionStageRecorded"
            and event.payload.get("generation") == generation
            and event.payload.get("stage") == "research"
            and event.payload.get("status") == "started"
        ]
        if started_attempts:
            latest_attempt = max(started_attempts)
            completed = any(
                event.kind == "EvolutionStageRecorded"
                and event.payload.get("generation") == generation
                and event.payload.get("stage") == "research"
                and event.payload.get("status") == "completed"
                and event.payload.get("attempt") == latest_attempt
                for event in state.events
            )
            if not completed:
                director.record_evolution_stage(
                    state.run.run_id,
                    generation=generation,
                    stage="research",
                    status="completed",
                    attempt=latest_attempt,
                )
        return existing

    previous = state.analysis_for(generation - 1) if generation > 0 else None
    assessment = (
        state.knowledge_assessment_for(generation - 1) if generation > 0 else None
    )
    model_id_value = state.task_manifest.metadata.get(
        "strategy_model_id",
        state.task_manifest.metadata.get("policy_model_id"),
    )
    model_id = (
        str(model_id_value).strip() or None
        if model_id_value is not None
        else None
    )
    current_plan = _latest_research_plan(state, generation)
    history_cutoff_seq = (
        state.events[0].seq
        if state.events and state.events[0].kind == "RunCreated"
        else None
    )
    cross_generation_experience = build_cross_generation_experience(
        state,
        generation,
        historical_states=_historical_experience_states(director, state),
        history_cutoff_seq=history_cutoff_seq,
    )
    research_attempt: int | None = None
    visible_pending_ids: tuple[str, ...] = ()
    consumed_answer_ids: tuple[str, ...] = ()
    expert_consultation: ExpertConsultation | None = None

    def record_research_failure(public_error: str) -> None:
        if research_attempt is None:
            return
        director.record_evolution_stage(
            state.run.run_id,
            generation=generation,
            stage="research",
            status="failed",
            attempt=research_attempt,
            public_error=public_error,
        )

    current_proposals = sorted(
        (item for item in state.proposals if item.generation == generation),
        key=lambda item: (item.created_at, item.proposal_id),
    )
    if current_proposals:
        raw_plan = current_proposals[0].metadata.get("plan")
        if not isinstance(raw_plan, Mapping):
            raise RuntimeError(
                "existing autonomous generation has no recoverable research plan"
            )
        plan = dict(raw_plan)
        status = "recovered_existing_proposal"
    elif generation == 0 and current_plan:
        # The creation-time model call is generation zero's research call.
        plan = current_plan
        status = "initial_frozen"
    else:
        planner = getattr(director.dsh, "research_iteration", None)
        if callable(planner):
            (
                expert_collaboration,
                selected_pending_ids,
                selected_answer_ids,
            ) = _expert_collaboration_context(state, generation)
            failed_attempts = [
                int(event.payload.get("attempt") or 0)
                for event in state.events
                if event.kind == "EvolutionStageRecorded"
                and event.payload.get("generation") == generation
                and event.payload.get("stage") == "research"
                and event.payload.get("status") == "failed"
            ]
            research_attempt = max(failed_attempts, default=0) + 1
            director.record_evolution_stage(
                state.run.run_id,
                generation=generation,
                stage="research",
                status="started",
                attempt=research_attempt,
            )
            try:
                planner_kwargs = {
                    "run": state.run,
                    "task": state.task_manifest,
                    "previous_generation_analysis": (
                        previous.to_dict() if previous is not None else None
                    ),
                    "knowledge_snapshot": knowledge.proposal_context(),
                    "previous_knowledge_assessment": (
                        assessment.to_dict() if assessment is not None else None
                    ),
                    "current_plan": current_plan,
                    "cross_generation_experience": cross_generation_experience,
                    "expert_collaboration": expert_collaboration,
                }
                if (
                    state.task_manifest.metadata.get("execution_protocol")
                    == "dsh_native_plugin_evolution@1"
                ):
                    parent = state.parent_genome_for_generation(generation)
                    planner_kwargs.update(
                        {
                            "parent_genome": parent.to_dict(),
                            "run_state_revision": state.events[-1].seq,
                            "stage_attempt": research_attempt,
                            "ledger_expected_revision": director.ledger.latest_seq(),
                        }
                    )
                raw_result = planner(
                    **planner_kwargs,
                )
            except Exception:
                record_research_failure(
                    "远程研究计划请求失败；运行将按既定重试与失败策略处理。"
                )
                raise
            try:
                if not isinstance(raw_result, Mapping):
                    raise TypeError("research iteration adapter must return an object")
                raw_plan = raw_result.get("plan")
                if not isinstance(raw_plan, Mapping):
                    raise TypeError("research iteration adapter result requires plan")
                plan = dict(raw_plan)
                status = str(raw_result.get("status") or "model_generated")
                raw_model_id = raw_result.get("model_id")
                if raw_model_id is not None:
                    model_id = str(raw_model_id).strip() or None
                if status == "model_generated":
                    visible_pending_ids = selected_pending_ids
                    consumed_answer_ids = selected_answer_ids
                    expert_consultation = _expert_consultation_from_result(
                        raw_result.get("expert_consultation"),
                        run_id=state.run.run_id,
                        generation=generation,
                        model_id=model_id,
                    )
            except (TypeError, ValueError) as exc:
                record_research_failure(
                    "远程研究计划响应未通过宿主契约校验。"
                )
                raise ResearchResponseContractError(
                    "research response failed host contract validation"
                ) from exc
        else:
            plan = current_plan
            status = "host_fallback"

    try:
        adoption = resolve_predictor_adoption(state.task_manifest, plan)
    except AlgorithmCompileError:
        record_research_failure("宿主冻结的预测器配置未通过校验。")
        raise
    except (TypeError, ValueError) as exc:
        record_research_failure("远程研究计划响应未通过宿主契约校验。")
        raise ResearchResponseContractError(
            "research response failed host contract validation"
        ) from exc
    except Exception:
        record_research_failure("研究计划的宿主解析过程失败。")
        raise

    # Probe only model-authored fields with known-valid host metadata. The
    # actual construction below is outside this conversion so corrupt host
    # state or provenance remains a terminal local error.
    try:
        response_contract = ResearchIteration(
            run_id="research-response-contract",
            generation=0,
            status=status,
            plan=plan,
            prediction_model_adoption=adoption.to_dict(),
            knowledge_snapshot_digest="research-response-contract",
            model_id=model_id,
        )
    except (TypeError, ValueError) as exc:
        record_research_failure("远程研究计划响应未通过宿主契约校验。")
        raise ResearchResponseContractError(
            "research response failed host contract validation"
        ) from exc

    try:
        iteration = ResearchIteration(
            run_id=state.run.run_id,
            generation=generation,
            status=response_contract.status,
            plan=response_contract.plan,
            prediction_model_adoption=adoption.to_dict(),
            knowledge_snapshot_digest=knowledge.snapshot_digest,
            source_analysis_digest=(previous.analysis_digest if previous else None),
            historical_provenance=(
                cross_generation_experience.get("historical_provenance")
                if isinstance(
                    cross_generation_experience.get("historical_provenance"),
                    Mapping,
                )
                else None
            ),
            source_assessment_digest=(
                assessment.assessment_digest if assessment is not None else None
            ),
            previous_next_action=(assessment.next_action if assessment else None),
            model_id=response_contract.model_id,
            pending_consultation_ids=visible_pending_ids,
            expert_answer_ids=consumed_answer_ids,
        )
        recorded = director.record_research_iteration(
            iteration,
            expert_consultation=expert_consultation,
        )
    except Exception:
        record_research_failure("研究结果的宿主状态校验或持久化失败。")
        raise
    if research_attempt is not None:
        director.record_evolution_stage(
            state.run.run_id,
            generation=generation,
            stage="research",
            status="completed",
            attempt=research_attempt,
        )
    return recorded


def start_generation_batch(director: Any, run_id: str) -> GenerationBatch:
    """Create or restore the immutable batch contract for the current round."""

    state = director.state(run_id)
    if state.run.status is not RunStatus.RUNNING:
        raise RuntimeError("run must be running to start a generation batch")
    existing = state.batch_for(state.run.generation)
    if existing is not None:
        knowledge = state.knowledge_for(existing.generation)
        if (
            knowledge is None
            or knowledge.snapshot_digest != existing.knowledge_snapshot_digest
        ):
            raise RuntimeError("generation batch knowledge snapshot is missing")
        _ensure_generation_research_iteration(director, state, knowledge)
        return existing
    current_count = sum(
        item.generation == state.run.generation for item in state.candidates
    )
    remaining = state.task_manifest.max_candidates - len(state.candidates)
    if remaining < 1 and current_count < 1:
        raise RuntimeError("run candidate budget exhausted")
    requested = state.task_manifest.candidates_per_generation
    batch_size = min(max(current_count, requested), remaining + current_count)

    knowledge = state.knowledge_for(state.run.generation)
    if knowledge is None:
        knowledge = retrieve_generation_knowledge(state)
        director.ledger.append(
            run_id,
            "GenerationKnowledgeRetrieved",
            {"knowledge_snapshot": knowledge.to_dict()},
            event_id=f"{run_id}:generation:{state.run.generation}:knowledge-retrieved",
        )
        state = director.state(run_id)
        knowledge = state.knowledge_for(state.run.generation) or knowledge

    _ensure_generation_research_iteration(director, state, knowledge)
    state = director.state(run_id)

    previous = (
        state.analysis_for(state.run.generation - 1)
        if state.run.generation > 0
        else None
    )
    parent_candidate_id = state.run.best_candidate_id
    if previous is not None:
        parent_candidate_id = (
            _search_parent_from_analysis(state, previous) or parent_candidate_id
        )
    for intervention in state.pending_interventions:
        if intervention.kind is InterventionKind.PARENT_SELECTION:
            parent_candidate_id = intervention.target_candidate_id
    if parent_candidate_id is not None:
        director._completed_parent_context(state, parent_candidate_id)

    parent_genome = None
    stage_context_digests = None
    if (
        state.task_manifest.metadata.get("execution_protocol")
        == "dsh_native_plugin_evolution@1"
    ):
        parent_genome = (
            state.materialized_seed_genome()
            if state.run.generation == 0
            else state.persisted_genome_for(parent_candidate_id)
        )
        parent_data = parent_genome.to_dict()
        runtime_binding = dict(parent_data["runtime_binding"])
        frozen_contracts = dict(parent_data["frozen_contract_refs"])
        research = state.research_iteration_for(state.run.generation)
        stage_context_digests = {
            "knowledge_snapshot_digest": knowledge.snapshot_digest,
            "research_iteration_digest": (
                research.iteration_digest
                if research is not None
                else digest({"research_iteration": None})
            ),
            "evaluation_cohort_digest": str(
                state.task_manifest.metadata["evaluation_cohort_digest"]
            ),
            "fitness_profile_digest": str(frozen_contracts["fitness_profile_digest"]),
            "compiler_semantic_digest": str(
                state.task_manifest.metadata.get("compiler_semantic_digest")
                or DEFAULT_COMPILER_SEMANTIC_DIGEST
            ),
            "registry_catalog_digest": str(runtime_binding["registry_catalog_digest"]),
            "security_kernel_digest": str(frozen_contracts["security_kernel_digest"]),
        }

    batch = GenerationBatch(
        run_id=run_id,
        generation=state.run.generation,
        batch_size=batch_size,
        task_manifest_digest=state.task_manifest.digest,
        parent_candidate_id=parent_candidate_id,
        parent_genome_digest=(
            parent_genome.genome_digest if parent_genome is not None else None
        ),
        parent_genome_canonical_json=(
            canonical_json(parent_genome.to_dict())
            if parent_genome is not None
            else None
        ),
        stage_context_digests=stage_context_digests,
        previous_analysis_digest=(previous.analysis_digest if previous else None),
        knowledge_snapshot_digest=knowledge.snapshot_digest,
        intervention_ids=tuple(
            item.intervention_id for item in state.pending_interventions
        ),
    )
    director.ledger.append(
        run_id,
        "GenerationBatchStarted",
        {"batch": batch.to_dict()},
        event_id=f"{run_id}:generation:{batch.generation}:batch-started",
    )
    return director.state(run_id).batch_for(batch.generation) or batch


def _validate_frozen_evidence(state: Any, batch: GenerationBatch) -> None:
    if batch.task_manifest_digest != state.task_manifest.digest:
        raise RuntimeError("generation batch task manifest changed")
    if batch.knowledge_snapshot_digest is not None:
        snapshot = state.knowledge_for(batch.generation)
        if snapshot is None or snapshot.snapshot_digest != batch.knowledge_snapshot_digest:
            raise RuntimeError("generation batch knowledge snapshot changed")
    evaluations = []
    for candidate in state.candidates:
        if candidate.generation != batch.generation:
            continue
        evaluation = state.evaluation_for(candidate.candidate_id)
        if evaluation is not None:
            evaluations.append(evaluation)
            artifact = state.artifact_for(candidate.candidate_id)
            if artifact is None:
                raise RuntimeError("evaluated candidate is missing its training artifact")
            expected_dataset = state.task_manifest.metadata.get("dataset_digest")
            if expected_dataset and artifact.dataset_digest != expected_dataset:
                raise RuntimeError("generation candidates used different dataset snapshots")
    if evaluations:
        partitions = {item.partition for item in evaluations}
        evaluators = {item.evaluator_digest for item in evaluations}
        if len(partitions) != 1:
            raise RuntimeError("generation candidates used different evaluation partitions")
        if len(evaluators) != 1:
            raise RuntimeError("generation candidates used different evaluators")
        if sample_update_windows_enabled(state.task_manifest):
            cohort_digests = [
                evaluation_cohort_digest(evaluation) for evaluation in evaluations
            ]
            if any(item is None for item in cohort_digests):
                raise RuntimeError(
                    "generation candidate is missing its bounded evaluation cohort"
                )
            if len(set(cohort_digests)) != 1:
                raise RuntimeError(
                    "generation candidates used different bounded evaluation cohorts"
                )


def _decision_reason(analysis: GenerationAnalysis, candidate_id: str) -> str:
    row = next(
        (item for item in analysis.ranking if item.get("candidate_id") == candidate_id),
        None,
    )
    if candidate_id == analysis.champion_candidate_id:
        return analysis.selection_reason
    if row is None:
        return "候选未进入本轮稳定排名。"
    reason = str(row.get("selection_reason") or "not_selected")
    labels = {
        "generation_best_did_not_improve_incumbent": "本轮排名第一，但未严格优于运行当前最优方案。",
        "cohort_changed_batch_champion": "本轮固定样本窗口排名第一；因窗口变化，未比较跨窗口原始分数。",
        "cohort_changed_search_parent_only": "本轮固定样本窗口排名第一；因窗口变化，仅作为后续搜索父方案，未作正式晋升。",
        "cohort_digest_unavailable": "候选缺少可验证的固定样本窗口摘要，未作正式晋升。",
        "lower_stable_rank_than_generation_best": "候选通过门禁，但同轮稳定排名低于本轮最佳候选。",
        "scientific_gate_failed": "候选未通过固定科学评测门槛。",
        "judge_rejected": "候选通过科学门槛，但未通过独立评审。",
        "judge_unavailable": "候选完成科学评测，但独立评审不可用，未作正式晋升。",
        "execution_failed": "候选训练或评测失败。",
        "duplicate": "候选参数与同轮较早候选重复，未重复评测。",
    }
    return labels.get(reason, f"候选未保留：{reason}。")


def finalize_generation_batch(director: Any, run_id: str) -> GenerationAnalysis:
    """Analyze all siblings, persist one-champion decisions, and remain resumable."""

    state = director.state(run_id)
    batch = state.batch_for(state.run.generation)
    if batch is None:
        raise RuntimeError("generation batch has not started")
    _validate_frozen_evidence(state, batch)
    analysis = state.analysis_for(batch.generation)
    if analysis is None:
        analysis = build_generation_analysis(state, batch)
        director.ledger.append(
            run_id,
            "GenerationAnalyzed",
            {"analysis": analysis.to_dict()},
            event_id=f"{run_id}:generation:{batch.generation}:analyzed",
        )
        state = director.state(run_id)
        analysis = state.analysis_for(batch.generation) or analysis

    if (
        state.task_manifest.metadata.get("execution_protocol")
        == "dsh_native_plugin_evolution@1"
    ):
        from ..core.exposure_registry import ScientificExposureRegistry
        from ..evaluators.fitness import FitnessProfile

        ScientificExposureRegistry(director.ledger).record_adaptive_evidence(
            run_id=run_id,
            evidence_digest=digest(
                {
                    "generation": batch.generation,
                    "batch_context_digest": batch.context_digest,
                    "ranking": list(analysis.ranking),
                }
            ),
            fitness_profile_digest=FitnessProfile.from_task(
                state.task_manifest
            ).profile_digest,
        )

    snapshot = state.knowledge_for(batch.generation)
    if snapshot is not None and state.knowledge_assessment_for(batch.generation) is None:
        assessment = assess_generation_knowledge(state, analysis, snapshot)
        director.ledger.append(
            run_id,
            "GenerationKnowledgeAssessed",
            {"knowledge_assessment": assessment.to_dict()},
            event_id=f"{run_id}:generation:{batch.generation}:knowledge-assessed",
        )
        state = director.state(run_id)

    ranked_ids = [str(item["candidate_id"]) for item in analysis.ranking]
    candidates = [state.candidate(candidate_id) for candidate_id in ranked_ids]
    candidates.sort(
        key=lambda item: item.candidate_id == analysis.champion_candidate_id
    )
    for candidate in candidates:
        if state.evaluation_for(candidate.candidate_id) is None:
            continue
        if state.promotion_for(candidate.candidate_id) is not None:
            continue
        approved = candidate.candidate_id == analysis.champion_candidate_id
        director.decide_promotion(
            Promotion(
                promotion_id=(
                    f"promotion:{run_id}:{batch.generation}:{candidate.candidate_id}"
                ),
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                decision=(
                    PromotionDecision.APPROVED
                    if approved
                    else PromotionDecision.REJECTED
                ),
                reason=_decision_reason(analysis, candidate.candidate_id),
            )
        )
        state = director.state(run_id)

    director.ledger.append(
        run_id,
        "GenerationChampionSelected",
        {
            "generation": batch.generation,
            "analysis_digest": analysis.analysis_digest,
            "outcome": analysis.outcome,
            "selected_candidate_id": analysis.selected_candidate_id,
            "champion_candidate_id": analysis.champion_candidate_id,
            "incumbent_before_candidate_id": analysis.incumbent_before_candidate_id,
            "incumbent_after_candidate_id": analysis.incumbent_after_candidate_id,
            "search_parent_candidate_id": analysis.search_parent_candidate_id,
            "selection_reason": analysis.selection_reason,
        },
        event_id=f"{run_id}:generation:{batch.generation}:champion-selected",
    )
    return analysis


def candidate_is_terminal(candidate: Any) -> bool:
    return candidate.status in {
        CandidateStatus.EVALUATED,
        CandidateStatus.PROMOTED,
        CandidateStatus.REJECTED,
        CandidateStatus.FAILED,
        CandidateStatus.DUPLICATE,
    }


__all__ = [
    "ResearchResponseContractError",
    "candidate_is_terminal",
    "finalize_generation_batch",
    "start_generation_batch",
]
