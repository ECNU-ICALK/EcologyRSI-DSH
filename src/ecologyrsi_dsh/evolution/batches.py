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
    RunStatus,
    canonical_json,
    digest,
)
from ..core.redaction import public_error_summary
from .workflow_ir import DEFAULT_COMPILER_SEMANTIC_DIGEST
from ..knowledge.research_iteration import (
    ResearchIteration,
)
from ..knowledge.autonomous_cycle import (
    AUTONOMOUS_RESEARCH_PROTOCOL,
    GenerationReflection,
    GenerationSearchPlan,
)
from ..knowledge.retrieval import (
    generation_query_hints,
    retrieve_generation_knowledge,
)
from .analysis import (
    GenerationAnalysis,
    GenerationBatch,
    build_cross_generation_experience,
    evaluation_cohort_digest,
    sample_update_windows_enabled,
)
from .genome import EcologyEvolutionPluginGenome, deep_thaw_json
from .research_recovery import research_checkpoint
from . import novelty
from .schedule import (
    PAIRED_LOCAL_EVALUATION_MODE,
    OptimizationSchedule,
)


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
    revision = _adaptive_effective_parent_revision(state)
    return revision.candidate_id if revision is not None else state.run.best_candidate_id


def _generation_parent_genome(state: Any, parent_candidate_id: str | None):
    """Resolve the selected revision, never the last rejected local proposal."""
    if state.run.generation == 0:
        return state.materialized_seed_genome()
    revision = _adaptive_effective_parent_revision(state)
    if revision is None or revision.candidate_id != parent_candidate_id:
        raise RuntimeError("parent differs from the frozen effective revision")
    return EcologyEvolutionPluginGenome.from_dict(deep_thaw_json(revision.genome))


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
    if novelty.enabled(state.task_manifest.metadata):
        cross_generation_experience[novelty.ARCHIVE_KEY] = novelty.archive_reference(
            novelty.build_archive(state, generation)
        )
    diagnostic = diagnose_generation(state, knowledge)
    cross_generation_experience["diagnostic_report"] = diagnostic.to_dict()
    research_attempt: int | None = None


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
                            "research_recovery": research_checkpoint(state, parent),
                        }
                    )
                raw_result = planner(
                    **planner_kwargs,
                )
            except (TypeError, ValueError) as exc:
                record_research_failure(
                    "远程研究计划响应未通过宿主契约校验。"
                )
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
            except (TypeError, ValueError) as exc:
                record_research_failure(
                    "远程研究计划响应未通过宿主契约校验。"
                )
                raise ResearchResponseContractError(
                    "research response failed host contract validation",
                    validation_detail=str(exc),
                ) from exc
        else:
            raise RuntimeError("native research iteration adapter is required")

    # Probe only model-authored fields with known-valid host metadata. The
    # actual construction below is outside this conversion so corrupt host
    # state or provenance remains a terminal local error.
    try:
        response_contract = ResearchIteration(
            run_id="research-response-contract",
            generation=0,
            status=status,
            plan=plan,
            knowledge_snapshot_digest="research-response-contract",
            model_id=model_id,
        )
    except (TypeError, ValueError) as exc:
        record_research_failure("远程研究计划响应未通过宿主契约校验。")
        raise ResearchResponseContractError(
            "research response failed host contract validation",
            validation_detail=str(exc),
        ) from exc

    try:
        iteration = ResearchIteration(
            run_id=state.run.run_id,
            generation=generation,
            status=response_contract.status,
            plan=response_contract.plan,
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
            diagnostic_report=diagnostic.to_dict(),
        )
        recorded = director.record_research_iteration(
            iteration,
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
    )
    director.ledger.append(
        run_id,
        "GenerationBatchStarted",
        {"batch": batch.to_dict()},
        event_id=f"{run_id}:generation:{batch.generation}:batch-started",
    )
    return director.state(run_id).batch_for(batch.generation) or batch






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
                **({"mutation_operator_id": "joint-plugin-mutation@1"}
                   if getattr(getattr(state, "task_manifest", None), "metadata", {}).get("optimization_protocol") in {"joint_plugin_epoch@1", "joint_plugin_epoch@2"} else {}),
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
    "start_generation_batch",
]
