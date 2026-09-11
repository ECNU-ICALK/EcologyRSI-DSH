"""Small orchestration helpers for one frozen multi-candidate generation."""

from __future__ import annotations

from ..evolution.schedule import ADAPTIVE_PROTOCOLS

from .diagnosis import diagnose_generation

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ..core.models import (
    CandidateRole,
    CandidateStatus,
    ExpertConsultation,
    InterventionKind,
    Promotion,
    PromotionDecision,
    RunStatus,
    canonical_json,
    digest,
)
from ..core.redaction import public_error_summary
from .workflow_ir import DEFAULT_COMPILER_SEMANTIC_DIGEST
from ..knowledge.algorithms import AlgorithmCompileError, resolve_predictor_adoption
from ..knowledge.research_iteration import (
    ResearchIteration,
    build_deterministic_research_fallback_plan,
)
from ..knowledge.autonomous_cycle import (
    AUTONOMOUS_RESEARCH_PROTOCOL,
    GenerationReflection,
    GenerationSearchPlan,
)
from ..knowledge.retrieval import (
    assess_generation_knowledge,
    generation_query_hints,
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
from .genome import EcologyEvolutionPluginGenome, deep_thaw_json
from .schedule import (
    OPTIMIZATION_PROTOCOL,
    QUICK_OPTIMIZATION_PROTOCOL,
    PAIRED_LOCAL_EVALUATION_MODE,
    OptimizationSchedule,
)


_EXPERT_PENDING_CONTEXT_LIMIT = 16
_EXPERT_ANSWER_CONTEXT_LIMIT = 8
_HISTORICAL_EXPERIENCE_SCAN_RUNS = 24


class ResearchResponseContractError(ValueError):
    """A model-authored research result failed the bounded host contract."""

    error_code = "research_response_contract_invalid"

    def __init__(
        self,
        message: str,
        *,
        validation_detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.validation_detail = public_error_summary(
            validation_detail if validation_detail is not None else message,
            limit=500,
        )


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


def _model_search_cycle_enabled(state: Any) -> bool:
    return (
        state.task_manifest.metadata.get("execution_protocol")
        == "dsh_native_plugin_evolution@1"
        and state.task_manifest.metadata.get("autonomous_research_protocol")
        == AUTONOMOUS_RESEARCH_PROTOCOL
    )


def _adaptive_effective_parent_revision(state: Any):
    generation = state.run.generation
    if (
        generation == 0
        or state.task_manifest.metadata.get("optimization_protocol")
        not in ADAPTIVE_PROTOCOLS
    ):
        return None
    selected_revision_id = state.effective_revision_for(generation - 1)
    if selected_revision_id is None:
        raise RuntimeError(
            "adaptive generation is missing the previous effective revision"
        )
    return state.revision(selected_revision_id)


def _generation_parent_candidate_id(state: Any) -> str | None:
    generation = state.run.generation
    parent_candidate_id = state.run.best_candidate_id
    effective_parent_revision = _adaptive_effective_parent_revision(state)
    if effective_parent_revision is not None:
        parent_candidate_id = effective_parent_revision.candidate_id
    else:
        previous = state.analysis_for(generation - 1) if generation > 0 else None
        if previous is not None:
            parent_candidate_id = (
                _search_parent_from_analysis(state, previous) or parent_candidate_id
            )
    for intervention in state.pending_interventions:
        if intervention.kind is InterventionKind.PARENT_SELECTION:
            parent_candidate_id = intervention.target_candidate_id
    return parent_candidate_id


def _generation_parent_genome(state: Any, parent_candidate_id: str | None):
    """Resolve the exact frozen source genome for the current generation."""

    generation = state.run.generation
    if generation == 0:
        return state.materialized_seed_genome()
    if parent_candidate_id is None:
        raise RuntimeError("generation is missing its parent candidate")
    effective_parent_revision = _adaptive_effective_parent_revision(state)
    if effective_parent_revision is not None:
        if effective_parent_revision.candidate_id == parent_candidate_id:
            return EcologyEvolutionPluginGenome.from_dict(
                deep_thaw_json(effective_parent_revision.genome)
            )
        # A pending, explicitly validated parent-selection intervention may
        # deliberately replace the automatic effective-revision parent.
        overridden_parent_ids = {
            item.target_candidate_id
            for item in state.pending_interventions
            if item.kind is InterventionKind.PARENT_SELECTION
        }
        if parent_candidate_id not in overridden_parent_ids:
            raise RuntimeError(
                "adaptive parent candidate differs from the frozen effective revision"
            )
    return state.persisted_genome_for(parent_candidate_id)


def _next_stage_attempt(state: Any, stage: str) -> int:
    attempts = [
        int(event.payload.get("attempt") or 0)
        for event in state.events
        if event.kind == "EvolutionStageRecorded"
        and event.payload.get("generation") == state.run.generation
        and event.payload.get("stage") == stage
        and event.payload.get("status") in {"started", "failed"}
    ]
    return max(attempts, default=0) + 1


def _validate_required_search_replan(
    state: Any,
    search_plan: GenerationSearchPlan,
) -> None:
    """Require a machine-visible direction change after repeated exploration."""

    generation = state.run.generation
    if generation < 1:
        return
    previous = state.analysis_for(generation - 1)
    if previous is None or getattr(previous, "replan_required", False) is not True:
        return
    if search_plan.source_analysis_digest != previous.analysis_digest:
        raise ValueError("required search replan must bind the triggering analysis")
    prior_plan = state.search_plan_for(generation - 1)
    if prior_plan is not None and (
        search_plan.search_queries == prior_plan.search_queries
        and search_plan.focus_areas == prior_plan.focus_areas
    ):
        raise ValueError(
            "required search replan must change queries or focus areas"
        )


def _ensure_generation_search_plan(
    director: Any,
    state: Any,
    *,
    candidate_count: int,
) -> GenerationSearchPlan | None:
    """Obtain and persist model-authored queries before any online retrieval."""

    if not _model_search_cycle_enabled(state):
        return None
    existing = state.search_plan_for(state.run.generation)
    if existing is not None:
        _validate_required_search_replan(state, existing)
        return existing
    planner = getattr(director.dsh, "plan_generation_search", None)
    if not callable(planner):
        raise RuntimeError("DSH strategy adapter has no generation search planner")
    generation = state.run.generation
    previous = state.analysis_for(generation - 1) if generation > 0 else None
    previous_reflection = (
        state.reflection_for(generation - 1) if generation > 0 else None
    )
    parent_candidate_id = _generation_parent_candidate_id(state)
    parent = _generation_parent_genome(state, parent_candidate_id)
    attempt = _next_stage_attempt(state, "search")
    director.record_evolution_stage(
        state.run.run_id,
        generation=generation,
        stage="search",
        status="started",
        attempt=attempt,
    )
    state = director.state(state.run.run_id)
    try:
        search_plan = planner(
            run=state.run,
            task=state.task_manifest,
            parent_genome=parent.to_dict(),
            previous_generation_analysis=(
                previous.to_dict() if previous is not None else None
            ),
            previous_generation_reflection=(
                previous_reflection.to_dict()
                if previous_reflection is not None
                else None
            ),
            current_plan=_latest_research_plan(state, generation),
            host_query_hints=generation_query_hints(state),
            candidate_count=candidate_count,
            run_state_revision=state.events[-1].seq,
            stage_attempt=attempt,
            ledger_expected_revision=director.ledger.latest_seq(),
        )
        if not isinstance(search_plan, GenerationSearchPlan):
            raise TypeError("generation search planner must return GenerationSearchPlan")
        _validate_required_search_replan(state, search_plan)
        recorded = director.record_generation_search_plan(search_plan)
    except Exception:
        director.record_evolution_stage(
            state.run.run_id,
            generation=generation,
            stage="search",
            status="failed",
            attempt=attempt,
            public_error="模型检索规划失败或未通过宿主契约校验。",
        )
        raise
    director.record_evolution_stage(
        state.run.run_id,
        generation=generation,
        stage="search",
        status="completed",
        attempt=attempt,
    )
    return recorded


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


def _research_contract_fallback_enabled(state: Any) -> bool:
    # A native research result must have valid model-authored directions.
    # The legacy deterministic plan has no such directions and cannot feed
    # the native proposer. A runtime version alone is never fallback consent.
    metadata = state.task_manifest.metadata
    if metadata.get("execution_protocol") == "dsh_native_plugin_evolution@1":
        return False
    if metadata.get("allow_host_fallback") is not True:
        return False
    runtime = metadata.get("host_runtime_build")
    return bool(
        isinstance(runtime, Mapping)
        and runtime.get("evolution_runtime_schema")
        == "ecologyrsi-dsh.evolution-runtime/3"
    )


def _research_contract_fallback_allowed(
    state: Any, validation_error: BaseException | None = None
) -> bool:
    """Allow recovery for bounded payload overflow in native quick runs.

    Native runs deliberately reject gateway/model fallback.  A model can still
    return a syntactically valid but oversized research plan, though; treating
    that local contract violation as a fatal run pause makes the small-batch
    experiment unnecessarily brittle.  In the quick protocol we retain the
    current plan and record a deterministic host fallback, while malformed
    contracts and all remote failures remain governed by the existing policy.
    """

    if _research_contract_fallback_enabled(state):
        return True
    metadata = state.task_manifest.metadata
    if metadata.get("execution_protocol") != "dsh_native_plugin_evolution@1":
        return False
    if metadata.get("optimization_protocol") != QUICK_OPTIMIZATION_PROTOCOL:
        return False
    if validation_error is None:
        return False
    detail = str(validation_error).lower()
    return "exceeds the bounded contract" in detail


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
    *,
    candidate_count: int,
    search_plan: GenerationSearchPlan | None = None,
) -> ResearchIteration | None:
    generation = state.run.generation
    if not _autonomous_research_enabled(state):
        return None
    existing = state.research_iteration_for(generation)
    if existing is not None:
        if existing.knowledge_snapshot_digest != knowledge.snapshot_digest:
            raise RuntimeError("generation research iteration knowledge changed")
        if _model_search_cycle_enabled(state):
            if search_plan is None:
                raise RuntimeError("generation research iteration has no search plan")
            recorded_search = existing.plan.get("generation_search_plan")
            if (
                not isinstance(recorded_search, Mapping)
                or recorded_search.get("search_plan_digest")
                != search_plan.search_plan_digest
            ):
                raise RuntimeError("generation research iteration search plan changed")
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
    previous_reflection = (
        state.reflection_for(generation - 1) if generation > 0 else None
    )
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
    diagnostic = diagnose_generation(state, knowledge)
    cross_generation_experience["diagnostic_report"] = diagnostic.to_dict()
    research_attempt: int | None = None
    visible_pending_ids: tuple[str, ...] = ()
    consumed_answer_ids: tuple[str, ...] = ()
    expert_consultation: ExpertConsultation | None = None

    def fallback_result(validation_detail: str) -> dict[str, Any]:
        nonlocal model_id, visible_pending_ids, consumed_answer_ids
        nonlocal expert_consultation
        model_id = None
        visible_pending_ids = ()
        consumed_answer_ids = ()
        expert_consultation = None
        return {
            "status": "host_fallback",
            "model_id": None,
            "plan": build_deterministic_research_fallback_plan(
                current_plan=current_plan,
                validation_detail=validation_detail,
                source_analysis_digest=(
                    previous.analysis_digest if previous is not None else None
                ),
                source_reflection_digest=(
                    previous_reflection.reflection_digest
                    if previous_reflection is not None
                    else None
                ),
                search_plan_digest=(
                    search_plan.search_plan_digest
                    if search_plan is not None
                    else None
                ),
            ),
        }

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
    elif generation == 0 and current_plan and not _model_search_cycle_enabled(state):
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
            research_attempt = _next_stage_attempt(state, "research")
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
                    "candidate_count": candidate_count,
                    "generation_search_plan": (
                        search_plan.to_dict() if search_plan is not None else None
                    ),
                    "previous_generation_reflection": (
                        previous_reflection.to_dict()
                        if previous_reflection is not None
                        else None
                    ),
                }
                if (
                    state.task_manifest.metadata.get("execution_protocol")
                    == "dsh_native_plugin_evolution@1"
                ):
                    parent_candidate_id = _generation_parent_candidate_id(state)
                    parent = _generation_parent_genome(
                        state, parent_candidate_id
                    )
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
            except (TypeError, ValueError) as exc:
                record_research_failure(
                    "远程研究计划响应未通过宿主契约校验。"
                )
                if _research_contract_fallback_allowed(state, exc):
                    raw_result = fallback_result(str(exc))
                else:
                    raise ResearchResponseContractError(
                        "research response failed host contract validation",
                        validation_detail=str(exc),
                    ) from exc
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
                if _research_contract_fallback_allowed(state, exc):
                    fallback = fallback_result(str(exc))
                    plan = dict(fallback["plan"])
                    status = "host_fallback"
                else:
                    raise ResearchResponseContractError(
                        "research response failed host contract validation",
                        validation_detail=str(exc),
                    ) from exc
        else:
            plan = current_plan
            status = "host_fallback"

    try:
        adoption = resolve_predictor_adoption(state.task_manifest, plan)
    except AlgorithmCompileError as exc:
        record_research_failure("宿主冻结的预测器配置未通过校验。")
        if _research_contract_fallback_allowed(state, exc):
            fallback = fallback_result(str(exc))
            plan = dict(fallback["plan"])
            status = "host_fallback"
            adoption = resolve_predictor_adoption(state.task_manifest, plan)
        else:
            raise
    except (TypeError, ValueError) as exc:
        record_research_failure("远程研究计划响应未通过宿主契约校验。")
        if _research_contract_fallback_allowed(state, exc):
            fallback = fallback_result(str(exc))
            plan = dict(fallback["plan"])
            status = "host_fallback"
            adoption = resolve_predictor_adoption(state.task_manifest, plan)
        else:
            raise ResearchResponseContractError(
                "research response failed host contract validation",
                validation_detail=str(exc),
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
        if not _research_contract_fallback_allowed(state, exc):
            raise ResearchResponseContractError(
                "research response failed host contract validation",
                validation_detail=str(exc),
            ) from exc
        fallback = fallback_result(str(exc))
        plan = dict(fallback["plan"])
        status = "host_fallback"
        adoption = resolve_predictor_adoption(state.task_manifest, plan)
        response_contract = ResearchIteration(
            run_id="research-response-contract",
            generation=0,
            status=status,
            plan=plan,
            prediction_model_adoption=adoption.to_dict(),
            knowledge_snapshot_digest="research-response-contract",
            model_id=None,
        )

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
            diagnostic_report=diagnostic.to_dict(),
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
        search_plan = state.search_plan_for(existing.generation)
        if _model_search_cycle_enabled(state) and search_plan is None:
            raise RuntimeError("generation batch search plan is missing")
        _ensure_generation_research_iteration(
            director,
            state,
            knowledge,
            candidate_count=existing.batch_size,
            search_plan=search_plan,
        )
        return existing
    search_candidates = tuple(
        item
        for item in state.candidates
        if getattr(item, "role", CandidateRole.SEARCH)
        in {CandidateRole.SEARCH, CandidateRole.SEARCH.value}
    )
    current_count = sum(
        item.generation == state.run.generation for item in search_candidates
    )
    remaining = state.task_manifest.max_candidates - len(search_candidates)
    if remaining < 1 and current_count < 1:
        raise RuntimeError("run candidate budget exhausted")
    requested = state.task_manifest.candidates_per_generation
    batch_size = min(max(current_count, requested), remaining + current_count)

    search_plan = _ensure_generation_search_plan(
        director,
        state,
        candidate_count=batch_size,
    )
    state = director.state(run_id)

    knowledge = state.knowledge_for(state.run.generation)
    if knowledge is None:
        knowledge = retrieve_generation_knowledge(
            state,
            query_terms=(
                list(search_plan.search_queries)
                if search_plan is not None
                else None
            ),
        )
        # Retrieval is intentionally outside the mutation boundary. Re-check
        # the run before committing its result: an operator may have cancelled
        # or completed the run while the remote knowledge request was in
        # flight. The expected revision closes the remaining check/append race.
        latest = director.state(run_id)
        already_recorded = latest.knowledge_for(state.run.generation)
        if already_recorded is not None:
            state = latest
            knowledge = already_recorded
        elif (
            latest.run.status is not RunStatus.RUNNING
            or latest.run.generation != state.run.generation
        ):
            raise RuntimeError(
                "run stopped or advanced while generation knowledge was retrieved"
            )
        else:
            director.ledger.append(
                run_id,
                "GenerationKnowledgeRetrieved",
                {"knowledge_snapshot": knowledge.to_dict()},
                event_id=f"{run_id}:generation:{state.run.generation}:knowledge-retrieved",
                expected_run_seq=latest.events[-1].seq,
            )
            state = director.state(run_id)
            knowledge = state.knowledge_for(state.run.generation) or knowledge

    _ensure_generation_research_iteration(
        director,
        state,
        knowledge,
        candidate_count=batch_size,
        search_plan=search_plan,
    )
    state = director.state(run_id)

    previous = (
        state.analysis_for(state.run.generation - 1)
        if state.run.generation > 0
        else None
    )
    parent_candidate_id = _generation_parent_candidate_id(state)
    effective_parent_revision = _adaptive_effective_parent_revision(state)
    adaptive_effective_parent = (
        effective_parent_revision is not None
        and effective_parent_revision.candidate_id == parent_candidate_id
    )
    if parent_candidate_id is not None and not adaptive_effective_parent:
        director._completed_parent_context(state, parent_candidate_id)

    parent_genome = None
    stage_context_digests = None
    if (
        state.task_manifest.metadata.get("execution_protocol")
        == "dsh_native_plugin_evolution@1"
    ):
        parent_genome = _generation_parent_genome(state, parent_candidate_id)
        parent_data = parent_genome.to_dict()
        runtime_binding = dict(parent_data["runtime_binding"])
        frozen_contracts = dict(parent_data["frozen_contract_refs"])
        research = state.research_iteration_for(state.run.generation)
        stage_context_digests = {
            "knowledge_snapshot_digest": knowledge.snapshot_digest,
            **(
                {"generation_search_plan_digest": search_plan.search_plan_digest}
                if search_plan is not None
                else {}
            ),
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
        if (
            candidate.generation != batch.generation
            or getattr(candidate, "role", CandidateRole.SEARCH)
            not in {CandidateRole.SEARCH, CandidateRole.SEARCH.value}
        ):
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
        "diagnostic_smoke_search_parent_only": "候选仅在诊断样本上排名第一，可供后续搜索参考，但证据不足以晋级。",
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


def _canonical_candidate_outcomes(
    state: Any,
    *,
    generation: int,
    analysis: GenerationAnalysis,
) -> tuple[dict[str, Any], ...]:
    """Bind each ranked result to its durable candidate and direction identity."""

    candidates = {
        item.candidate_id: item
        for item in state.candidates
        if item.generation == generation
        and getattr(item, "role", CandidateRole.SEARCH)
        in {CandidateRole.SEARCH, CandidateRole.SEARCH.value}
    }
    rows: dict[str, Mapping[str, Any]] = {}
    for row in analysis.ranking:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise RuntimeError("generation ranking contains an invalid candidate id")
        if candidate_id in rows:
            raise RuntimeError("generation ranking contains a duplicate candidate")
        rows[candidate_id] = row
    if len(candidates) != analysis.candidate_count or set(rows) != set(candidates):
        raise RuntimeError(
            "generation ranking does not match the generation candidate set"
        )

    outcomes: list[dict[str, Any]] = []
    for candidate_id, row in rows.items():
        candidate = candidates[candidate_id]
        proposal = state.proposal(candidate.proposal_id)
        metadata = proposal.metadata
        outcomes.append(
            {
                "rank": row.get("rank"),
                "candidate_id": candidate_id,
                "direction_id": metadata.get("candidate_direction_id"),
                "direction_digest": metadata.get("candidate_direction_digest"),
                "slot_index": candidate.slot_index,
                "status": candidate.status.value,
                "score": row.get("score"),
                "eligible": row.get("eligible"),
                "classification": row.get("classification"),
                "selection_reason": row.get("selection_reason"),
                "mutation_operations": metadata.get("mutation_operations"),
                "behavior_digest": metadata.get("behavior_digest"),
            }
        )
    outcomes.sort(
        key=lambda item: (
            item["rank"] is None,
            item["rank"] if item["rank"] is not None else 0,
            item["slot_index"],
            item["candidate_id"],
        )
    )
    return tuple(outcomes)


def _adaptive_trajectory_comparison_evidence(
    ranking: tuple[Mapping[str, Any], ...],
) -> list[dict[str, Any]]:
    """Whitelist bounded lane decisions for the generation reflector."""

    comparison_fields = (
        "batch_index",
        "decision",
        "reason",
        "score_delta",
        "minimum_score_delta",
        "safety_gate_passed",
        "cell_regression_gate_passed",
        "champion_before_revision_id",
        "challenger_revision_id",
        "champion_after_revision_id",
        "challenger_operation_category",
        "challenger_operation_targets",
        "rejected_operation_category",
        "rejected_operation_targets",
        "next_mutation_parent_revision_id",
        "next_challenger_revision_id",
    )
    lanes: list[dict[str, Any]] = []
    for candidate in ranking:
        local_evidence = candidate.get("local_edit_evidence")
        if not isinstance(local_evidence, Mapping):
            continue
        raw_comparisons = local_evidence.get("recent_comparisons")
        if not isinstance(raw_comparisons, (list, tuple)):
            continue
        comparisons = [
            {
                name: item.get(name)
                for name in comparison_fields
            }
            for item in raw_comparisons[-10:]
            if isinstance(item, Mapping)
        ]
        if not comparisons:
            continue
        lanes.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "final_revision_id": local_evidence.get("final_revision_id"),
                "recent_comparisons": comparisons,
            }
        )
    return lanes


def _adaptive_candidate_result_evidence(
    ranking: tuple[Mapping[str, Any], ...],
) -> list[dict[str, Any]]:
    """Project fixed aggregate fields and omit detailed lane histories."""

    candidate_fields = (
        "rank",
        "candidate_id",
        "slot_index",
        "score",
        "eligible",
        "scientific_pass",
        "constraint_violations",
        "classification",
        "primary_selection_gate",
        "selection_status",
        "selection_reason",
        "judge_available",
        "judge_accepted",
        "holdout_arm",
        "delta_to_incumbent",
        "failure_reasons",
        "final_revision_id",
        "final_revision_digest",
        "final_genome_digest",
        "final_behavior_digest",
        "comparison_gate",
        "outer_mutation_evidence",
    )
    local_fields = (
        "kind",
        "local_evaluation_mode",
        "batch_count",
        "local_edit_decision_count",
        "included_batch_count",
        "truncated_batch_count",
        "comparison_count",
        "included_comparison_count",
        "truncated_comparison_count",
        "initial_revision_id",
        "final_revision_id",
        "revision_chain",
        "outcome_counts",
        "operation_category_counts",
    )
    results: list[dict[str, Any]] = []
    for candidate in ranking:
        result = {
            name: candidate.get(name)
            for name in candidate_fields
            if name in candidate
        }
        local_evidence = candidate.get("local_edit_evidence")
        if isinstance(local_evidence, Mapping):
            result["local_edit_evidence"] = {
                name: local_evidence.get(name)
                for name in local_fields
                if name in local_evidence
            }
        results.append(result)
    return json.loads(canonical_json(results))


def _adaptive_reflection_analysis(
    state: Any,
    analysis: GenerationAnalysis,
) -> dict[str, Any]:
    """Expose adaptive epoch evidence while keeping legacy reflection compact."""

    reflection_analysis = analysis.to_dict()
    reflection_analysis.pop("ranking", None)
    if (
        state.task_manifest.metadata.get("optimization_protocol")
        not in ADAPTIVE_PROTOCOLS
    ):
        return reflection_analysis
    schedule_value = state.task_manifest.metadata.get("optimization_schedule")
    paired_mode = bool(
        isinstance(schedule_value, Mapping)
        and OptimizationSchedule.from_dict(schedule_value).local_evaluation_mode
        == PAIRED_LOCAL_EVALUATION_MODE
    )
    if paired_mode:
        reflection_analysis.pop("created_at", None)
    comparison = state.comparison_for(analysis.generation)
    if comparison is None:
        raise RuntimeError("adaptive reflection is missing generation comparison")
    incumbent = next(
        item
        for item in comparison.holdout_evaluations
        if item.scope.holdout_arm.value == "incumbent"
    )
    selected_revision = state.revision(comparison.selected_revision_id)
    incumbent_revision = state.revision(incumbent.scope.candidate_revision_id)

    def revision_summary(revision: Any, *, score: float) -> dict[str, Any]:
        return {
            "candidate_id": revision.candidate_id,
            "revision_id": revision.revision_id,
            "revision_digest": revision.revision_digest,
            "genome_digest": revision.genome_digest,
            "behavior_digest": revision.behavior_digest,
            "holdout_score": score,
        }

    selected_holdout = next(
        item
        for item in comparison.holdout_evaluations
        if item.scope.candidate_id == comparison.selected_candidate_id
        and item.scope.candidate_revision_id == comparison.selected_revision_id
    )
    gate_results = comparison.gate_results
    candidate_results = (
        _adaptive_candidate_result_evidence(analysis.ranking)
        if paired_mode
        else json.loads(canonical_json(list(analysis.ranking)))
    )
    adaptive_evidence = {
        "schema_version": "ecologyrsi-dsh.adaptive-reflection-evidence/1",
        "generation": analysis.generation,
        "comparison_digest": comparison.comparison_digest,
        "selected_arm": gate_results.get("selected_arm"),
        "selected": revision_summary(
            selected_revision,
            score=selected_holdout.score,
        ),
        "incumbent": revision_summary(
            incumbent_revision,
            score=incumbent.score,
        ),
        "delta_to_incumbent": gate_results.get("delta_to_incumbent"),
        # These are Host-authored aggregate rows.  Their two distinct evidence
        # blocks make the once-per-generation outer mutation and the ten
        # within-candidate batch edits impossible to conflate during reflection.
        "candidate_results": candidate_results,
    }
    if paired_mode:
        adaptive_evidence["trajectory_comparisons"] = (
            _adaptive_trajectory_comparison_evidence(analysis.ranking)
        )
    reflection_analysis["adaptive_epoch_evidence"] = adaptive_evidence
    return reflection_analysis


def _ensure_generation_reflection(
    director: Any,
    state: Any,
    batch: GenerationBatch,
    analysis: GenerationAnalysis,
) -> GenerationReflection | None:
    """Reflect once on aggregate batch evidence and seed the next generation."""

    if not _model_search_cycle_enabled(state):
        return None
    candidate_outcomes = _canonical_candidate_outcomes(
        state,
        generation=batch.generation,
        analysis=analysis,
    )
    existing = state.reflection_for(batch.generation)
    if existing is not None:
        if existing.analysis_digest != analysis.analysis_digest:
            raise RuntimeError("generation reflection analysis changed")
        if existing.canonical_candidate_outcomes != candidate_outcomes:
            raise RuntimeError("generation reflection candidate outcomes changed")
        return existing
    reflector = getattr(director.dsh, "reflect_generation", None)
    if not callable(reflector):
        raise RuntimeError("DSH strategy adapter has no generation reflector")
    if not isinstance(batch.parent_genome_canonical_json, str):
        raise RuntimeError("generation reflection is missing the frozen parent genome")
    parent_genome = json.loads(batch.parent_genome_canonical_json)
    reflection_analysis = _adaptive_reflection_analysis(state, analysis)
    attempt = _next_stage_attempt(state, "reflection")
    director.record_evolution_stage(
        state.run.run_id,
        generation=batch.generation,
        stage="reflection",
        status="started",
        attempt=attempt,
    )
    state = director.state(state.run.run_id)
    try:
        reflection = reflector(
            run=state.run,
            task=state.task_manifest,
            parent_genome=parent_genome,
            generation_analysis=reflection_analysis,
            knowledge_snapshot=(
                state.knowledge_for(batch.generation).proposal_context()
                if state.knowledge_for(batch.generation) is not None
                else None
            ),
            research_iteration=(
                state.research_iteration_for(batch.generation).to_dict()
                if state.research_iteration_for(batch.generation) is not None
                else None
            ),
            candidate_outcomes=list(candidate_outcomes),
            direction_count=min(
                8,
                max(2, int(state.task_manifest.candidates_per_generation)),
            ),
            run_state_revision=state.events[-1].seq,
            stage_attempt=attempt,
            ledger_expected_revision=director.ledger.latest_seq(),
        )
        if not isinstance(reflection, GenerationReflection):
            raise TypeError(
                "generation reflector must return GenerationReflection"
            )
        if reflection.canonical_candidate_outcomes:
            raise ValueError(
                "generation reflector cannot author canonical candidate outcomes"
            )
        reflection = replace(
            reflection,
            canonical_candidate_outcomes=candidate_outcomes,
            reflection_digest="",
        )
        recorded = director.record_generation_reflection(reflection)
    except Exception:
        director.record_evolution_stage(
            state.run.run_id,
            generation=batch.generation,
            stage="reflection",
            status="failed",
            attempt=attempt,
            public_error="批次反思失败或未通过宿主契约校验。",
        )
        raise
    director.record_evolution_stage(
        state.run.run_id,
        generation=batch.generation,
        stage="reflection",
        status="completed",
        attempt=attempt,
    )
    return recorded


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

    _ensure_generation_reflection(director, state, batch, analysis)
    state = director.state(run_id)

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
        CandidateStatus.SCREENED_OUT,
    }


__all__ = [
    "ResearchResponseContractError",
    "candidate_is_terminal",
    "finalize_generation_batch",
    "start_generation_batch",
]
