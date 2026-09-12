"""Read-only run state and deterministic event-stream replay."""

from __future__ import annotations

from ..evolution.schedule import ADAPTIVE_PROTOCOLS

from .dsh_usage import validate_session_metrics as _validate_dsh_session_metrics
from .dsh_usage import validate_session_usage, check_usage_binding

from ..evolution.diagnosis import diagnose_evidence
from .research import DiagnosticReport

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import math
from typing import Any

from ..evolution.analysis import (
    GenerationAnalysis,
    GenerationBatch,
    sample_update_windows_enabled,
)
from .search_policy import guarded_search, local_challenger_policy, paired_execution_qualification_required
from .model_preflight import AUDIT_EVENT, preflight_audit_required, validate_preflight_audit
from .artifact_identity import (
    ARTIFACT_EVENT_V2, EVALUATION_EVENT_V2, FORMAL_STAGE_V2,
    resolve_artifact_scope, validate_artifact_revision_binding,
    validate_evaluation_artifact_binding,
)
from ..evolution.champion_challenger import (
    LOCAL_MINIMUM_SCORE_DELTA,
    POSITIVE_DELTA_MINIMUM_SCORE_DELTA,
    validate_formal_batch_comparison,
)
from ..evolution.schedule import (
    OPTIMIZATION_PROTOCOL,
    PAIRED_LOCAL_EVALUATION_MODE,
    OptimizationSchedule,
)
from ..evaluators.epoch_cohorts import GenerationCohorts, RunAdaptationCohort
from ..knowledge.algorithms import AlgorithmAttempt
from ..knowledge.autonomous_cycle import (
    GenerationReflection,
    GenerationSearchPlan,
)
from ..knowledge.models import KnowledgeAssessment, KnowledgeSnapshot
from ..knowledge.research_iteration import ResearchIteration
from .ledger import Event
from .models import (
    Candidate,
    CandidateRole,
    CandidateStatus,
    Evaluation,
    ExpertConsultation,
    ExpertConsultationAnswer,
    HumanIntervention,
    ModelArtifact,
    Promotion,
    PromotionDecision,
    Proposal,
    Run,
    RunStatus,
    TaskManifest,
    canonical_json,
    digest,
)
from .protocols import supports_two_stage_screening
from .retry_policy import (
    GATEWAY_CIRCUIT_CODES,
    GATEWAY_RETRY_CLASSES,
    GATEWAY_RETRY_EPOCH_SECONDS,
    GATEWAY_RETRY_MAX_DELAY_SECONDS,
    GATEWAY_RETRY_POLICIES,
    GATEWAY_RETRY_SCHEMA_VERSION,
)
from .sample_budget import complete_origin_count
from .screening import (
    FORMAL_SELECTION_SCHEMA_V1,
    FORMAL_SELECTION_SCHEMA_V2,
    FORMAL_SELECTION_SCHEMA_V3,
    SCREENED_OUT_SCHEMA_V1,
    SCREENING_SCHEMA_V1,
    SCREENING_SCHEMA_V2,
    screening_cohort_digest,
    screening_record_digest,
)
from .trajectory import (
    BatchEvaluation,
    CandidateRevision,
    FormalBatch,
    FormalBatchArm,
    FormalBatchComparison,
    FormalTrajectory,
    GenerationComparison,
    GenerationHoldout,
    HoldoutArm,
    HoldoutEvaluation,
    LocalEditOutcome,
    LocalEditProposalDecision,
    RevisionAdvanceReason,
    TrajectoryRevisionActivation,
    TrajectoryStatus,
)

DSH_NATIVE_EVOLUTION_PROTOCOL = "dsh_native_plugin_evolution@1"
EVOLUTION_RUNTIME_SCHEMA_V2 = "ecologyrsi-dsh.evolution-runtime/2"
EVOLUTION_RUNTIME_SCHEMA_V3 = "ecologyrsi-dsh.evolution-runtime/3"

_DSH_STAGE_SKILLS: dict[str, frozenset[str]] = {
    "generation.research": frozenset({"autonomous-ecology-research"}),
    "generation.search-plan": frozenset({"autonomous-ecology-research"}),
    "generation.research-synthesis": frozenset({"autonomous-ecology-research"}),
    "generation.reflect": frozenset({"batch-scientific-reflection"}),
    "candidate.propose": frozenset({"bounded-plugin-experiment"}),
    "candidate.local_edit": frozenset({"bounded-plugin-experiment"}),
    # Replay must remain compatible with results accepted before the judge and
    # batch-reflection stages were split.  The write-side DSH tool contract
    # still accepts only candidate-scientific-review for new judge results.
    "generation.judge": frozenset(
        {"candidate-scientific-review", "batch-scientific-reflection"}
    ),
    # Replay must keep accepting the three per-strategy Skill names that existed
    # before the strategy moved out of the Skill file into the registry
    # directive.  The write-side DSH tool contract accepts only the shared
    # origin-vector-forecasting Skill for new sample results.
    "sample.plan": frozenset(
        {
            "origin-vector-forecasting",
            "origin-vector-forecasting-balanced",
            "origin-vector-forecasting-anomaly-aware",
            "origin-vector-forecasting-horizon-aware",
        }
    ),
    "sample.critic": frozenset({"origin-vector-review"}),
    "sample.reflect": frozenset({"origin-vector-review"}),
}

_DSH_RETRIEVAL_ROLES = frozenset(
    {
        "coordinator",
        "researcher",
        "candidate-proposer",
        "sample-planner",
        "sample-critic",
        "generation-judge",
    }
)
_DSH_RETRIEVAL_FALLBACK_REASONS = frozenset(
    {
        "primary_provider_unavailable",
        "primary_provider_error",
        "primary_timeout",
        "primary_malformed",
        "primary_empty",
        "insufficient_distinct_sources",
        "insufficient_evidence_sources",
        "insufficient_query_overlap",
    }
)


def _validate_dsh_skill_evidence(value: Any, *, stage: str) -> None:
    fields = {
        "schema_version",
        "stage",
        "skill_name",
        "call_count",
        "successful_call_count",
        "call_seq",
        "result_seq",
        "first_tool_call_verified",
        "next_tool_name",
        "next_tool_call_seq",
        "order_verified",
        "source",
    }
    expected_next = "structured_output"
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema_version")
        != "ecologyrsi-dsh.skill-invocation-evidence/1"
        or value.get("stage") != stage
        or value.get("skill_name") not in _DSH_STAGE_SKILLS.get(stage, frozenset())
        or value.get("call_count") != 1
        or value.get("successful_call_count") != 1
        or value.get("first_tool_call_verified") is not True
        or value.get("next_tool_name") != expected_next
        or value.get("order_verified") is not True
        or value.get("source") != "dsh_session_event_log"
    ):
        raise ValueError("DSH Skill invocation evidence is invalid")
    sequence = [value.get(name) for name in ("call_seq", "result_seq", "next_tool_call_seq")]
    if (
        any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in sequence)
        or not sequence[0] < sequence[1] < sequence[2]
    ):
        raise ValueError("DSH Skill invocation evidence order is invalid")


def _validate_dsh_retrieval_event(value: Any, *, run_id: str) -> None:
    fields = {
        "schema_version",
        "execution_owner",
        "identity",
        "retrieval_idempotency_key",
        "query_digest",
        "queries",
        "provider_route",
        "fallback_reason",
        "primary_quality",
        "result",
        "result_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("DshRetrievalExecuted payload is invalid")
    identity = value.get("identity")
    identity_fields = {
        "run_id",
        "role",
        "stage",
        "run_state_revision",
        "stage_attempt",
        "ledger_expected_revision",
        "session_id",
        "idempotency_key",
        "child_reservation_id",
        "activation_lease_id",
        "genome_digest",
        "compiled_behavior_digest",
        "phenotype_instance_digest",
    }
    if (
        value.get("schema_version") != "ecologyrsi-dsh.retrieval-executed/1"
        or value.get("execution_owner") != "dsh_agent_web_search"
        or not isinstance(identity, Mapping)
        or set(identity) != identity_fields
        or identity.get("run_id") != run_id
        or identity.get("role") not in _DSH_RETRIEVAL_ROLES
        or not isinstance(identity.get("stage"), str)
        or not identity["stage"]
    ):
        raise ValueError("DshRetrievalExecuted identity is invalid")
    for name in ("run_state_revision", "stage_attempt", "ledger_expected_revision"):
        item = identity.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError("DshRetrievalExecuted identity revision is invalid")
    for name in (
        "session_id",
        "idempotency_key",
        "child_reservation_id",
        "activation_lease_id",
        "genome_digest",
        "compiled_behavior_digest",
        "phenotype_instance_digest",
    ):
        if not isinstance(identity.get(name), str) or not identity[name]:
            raise ValueError("DshRetrievalExecuted identity field is invalid")
    for name in (
        "genome_digest",
        "compiled_behavior_digest",
        "phenotype_instance_digest",
    ):
        if len(identity[name]) != 64 or any(
            character not in "0123456789abcdef" for character in identity[name]
        ):
            raise ValueError("DshRetrievalExecuted identity digest is invalid")
    retrieval_key = value.get("retrieval_idempotency_key")
    queries = value.get("queries")
    if (
        not isinstance(retrieval_key, str)
        or not retrieval_key
        or len(retrieval_key) > 120
        or not isinstance(queries, list)
        or not 1 <= len(queries) <= 4
        or not all(
            isinstance(item, str) and item and len(item) <= 180 for item in queries
        )
        or value.get("query_digest") != digest(queries)
    ):
        raise ValueError("DshRetrievalExecuted query contract is invalid")
    quality = value.get("primary_quality")
    quality_fields = {
        "distinct_source_count",
        "evidence_source_count",
        "query_token_count",
        "overlap_token_count",
        "sufficient",
        "fallback_reason",
    }
    fallback_reason = value.get("fallback_reason")
    route = value.get("provider_route")
    if (
        not isinstance(quality, Mapping)
        or set(quality) != quality_fields
        or any(
            isinstance(quality.get(name), bool)
            or not isinstance(quality.get(name), int)
            or quality[name] < 0
            for name in (
                "distinct_source_count",
                "evidence_source_count",
                "query_token_count",
                "overlap_token_count",
            )
        )
        or not isinstance(quality.get("sufficient"), bool)
        or quality.get("fallback_reason") != fallback_reason
        or fallback_reason not in (_DSH_RETRIEVAL_FALLBACK_REASONS | {None})
        or route not in {
            "dsh_primary",
            "dsh_primary_then_openalex_fallback",
        }
        or (route == "dsh_primary") != (fallback_reason is None)
        or quality["sufficient"] != (fallback_reason is None)
    ):
        raise ValueError("DshRetrievalExecuted quality contract is invalid")
    result = value.get("result")
    if (
        not isinstance(result, Mapping)
        or set(result) - {"content", "sources", "truncated"}
        or not isinstance(result.get("sources"), list)
        or len(result["sources"]) > 8
        or not isinstance(result.get("truncated"), bool)
        or (
            "content" in result
            and (
                not isinstance(result["content"], str)
                or not result["content"]
                or len(result["content"]) > 4_000
            )
        )
    ):
        raise ValueError("DshRetrievalExecuted result is invalid")
    for source in result["sources"]:
        if (
            not isinstance(source, Mapping)
            or set(source) - {"url", "title", "snippet", "publishedAt"}
            or not isinstance(source.get("url"), str)
            or not source["url"].startswith("https://")
            or any(
                name in source
                and (not isinstance(source[name], str) or not source[name])
                for name in ("title", "snippet", "publishedAt")
            )
        ):
            raise ValueError("DshRetrievalExecuted source is invalid")
    if value.get("result_digest") != digest(result):
        raise ValueError("DshRetrievalExecuted retrieval result digest mismatch")


def is_dsh_native_protocol(task: TaskManifest) -> bool:
    return task.metadata.get("execution_protocol") == DSH_NATIVE_EVOLUTION_PROTOCOL


def uses_global_incumbent_protocol(task: TaskManifest) -> bool:
    """Return whether a frozen run opted into the explicit global incumbent."""

    runtime = task.metadata.get("host_runtime_build")
    return is_dsh_native_protocol(task) and isinstance(runtime, Mapping) and (
        runtime.get("evolution_runtime_schema")
        in {EVOLUTION_RUNTIME_SCHEMA_V2, EVOLUTION_RUNTIME_SCHEMA_V3}
    )


def uses_positive_delta_search_protocol(task: TaskManifest) -> bool:
    """Return whether the frozen run advances on any comparable positive delta."""

    runtime = task.metadata.get("host_runtime_build")
    return is_dsh_native_protocol(task) and isinstance(runtime, Mapping) and (
        runtime.get("evolution_runtime_schema") == EVOLUTION_RUNTIME_SCHEMA_V3
        and not guarded_search(task.metadata)
    )


def validate_generation_zero_incumbent_binding(
    task: TaskManifest,
    generation: int,
    arm_bindings: Mapping[str, Mapping[str, str]],
    *,
    candidates: Mapping[str, Candidate],
    revisions: Mapping[str, CandidateRevision],
    materialized_seed_canonical: str | None,
    prior_effective_revision_id: str | None = None,
    prior_champion_selected: bool = False,
) -> None:
    """Fail closed unless every runtime-v2 incumbent continues the champion."""

    if not uses_global_incumbent_protocol(task):
        return
    binding = arm_bindings.get(HoldoutArm.INCUMBENT.value)
    if not isinstance(binding, Mapping):
        raise ValueError("runtime-v2 holdout requires the global incumbent")
    candidate = candidates.get(str(binding.get("candidate_id")))
    revision = revisions.get(str(binding.get("candidate_revision_id")))
    if generation > 0:
        expected_revision = revisions.get(str(prior_effective_revision_id))
        if (
            not prior_champion_selected
            or prior_effective_revision_id is None
            or expected_revision is None
            or candidate is None
            or revision is None
            or revision.revision_id != prior_effective_revision_id
            or candidate.candidate_id != expected_revision.candidate_id
            or revision.candidate_id != expected_revision.candidate_id
        ):
            raise ValueError(
                "runtime-v2 incumbent must bind the prior effective champion revision"
            )
        return
    if (
        candidate is None
        or candidate.generation != 0
        or candidate.role is not CandidateRole.INCUMBENT_CONTROL
        or revision is None
        or revision.candidate_id != candidate.candidate_id
        or revision.generation != 0
        or revision.parent_revision_id is not None
        or revision.source_batch_index is not None
        or materialized_seed_canonical is None
    ):
        raise ValueError("runtime-v2 holdout requires the seed incumbent control R0")
    import json

    from ..evolution.genome import EcologyEvolutionPluginGenome

    seed = EcologyEvolutionPluginGenome.from_dict(
        json.loads(materialized_seed_canonical)
    )
    if (
        revision.genome_digest != seed.genome_digest
        or revision.behavior_digest != seed.behavior_digest
        or canonical_json(revision.identity_dict()["genome"])
        != canonical_json(seed.to_dict())
    ):
        raise ValueError("runtime-v2 seed incumbent control R0 differs from materialized seed")


def validate_exploration_generation_comparison(
    task: TaskManifest,
    formal: Any,
    comparison: GenerationComparison,
) -> None:
    """Ensure an exploration-only generation cannot durably promote a challenger."""

    payload = getattr(formal, "payload", formal)
    if uses_positive_delta_search_protocol(task):
        return
    if not isinstance(payload, Mapping) or (
        payload.get("schema_version") != FORMAL_SELECTION_SCHEMA_V3
        or payload.get("exploration_only") is not True
    ):
        return
    incumbent = next(
        (
            evaluation
            for evaluation in comparison.holdout_evaluations
            if evaluation.scope.holdout_arm is HoldoutArm.INCUMBENT
        ),
        None,
    )
    gates = comparison.gate_results
    if (
        incumbent is None
        or gates.get("challenger_promotion_allowed") is not False
        or gates.get("selected_arm") != HoldoutArm.INCUMBENT.value
        or comparison.selected_candidate_id != incumbent.scope.candidate_id
        or comparison.selected_revision_id
        != incumbent.scope.candidate_revision_id
    ):
        raise ValueError(
            "exploration comparison must retain the incumbent and disable promotion"
        )


def validate_runtime_promotion_decision(
    task: TaskManifest,
    candidate: Candidate,
    promotion: Promotion,
    comparison: GenerationComparison | None,
    analysis: GenerationAnalysis | None,
) -> None:
    """Bind a runtime-v2/v3 decision to the frozen generation authority."""

    if not uses_global_incumbent_protocol(task):
        return
    if comparison is None or analysis is None:
        raise ValueError(
            "runtime promotion requires the recorded generation comparison and analysis"
        )
    if (
        comparison.generation != candidate.generation
        or analysis.generation != candidate.generation
        or comparison.run_id != candidate.run_id
        or analysis.run_id != candidate.run_id
    ):
        raise ValueError("runtime promotion generation binding is invalid")

    gate_results = comparison.gate_results
    arms = gate_results.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("runtime promotion comparison gates are missing")

    if uses_positive_delta_search_protocol(task):
        selected_arm = gate_results.get("selected_arm")
        selected_evaluation = next(
            (
                item
                for item in comparison.holdout_evaluations
                if item.scope.holdout_arm is not None
                and item.scope.holdout_arm.value == selected_arm
            ),
            None,
        )
        expected_analysis_selected = (
            selected_evaluation.scope.candidate_id
            if selected_evaluation is not None
            and selected_evaluation.scope.holdout_arm
            in {HoldoutArm.FINALIST_1, HoldoutArm.FINALIST_2}
            else None
        )
        certification_arm = gate_results.get("certification_selected_arm")
        certification_evaluation = next(
            (
                item
                for item in comparison.holdout_evaluations
                if item.scope.holdout_arm is not None
                and item.scope.holdout_arm.value == certification_arm
            ),
            None,
        )
        certification_gate = (
            arms.get(certification_arm)
            if isinstance(certification_arm, str)
            else None
        )
        certified_candidate_id = (
            certification_evaluation.scope.candidate_id
            if certification_evaluation is not None
            and certification_evaluation.scope.holdout_arm
            in {HoldoutArm.FINALIST_1, HoldoutArm.FINALIST_2}
            and isinstance(certification_gate, Mapping)
            and certification_gate.get("certification_eligible") is True
            else None
        )
        expected_approved = candidate.candidate_id == certified_candidate_id
        actual_approved = promotion.decision is PromotionDecision.APPROVED
        champion_row = next(
            (
                row
                for row in analysis.ranking
                if row.get("candidate_id") == certified_candidate_id
            ),
            None,
        )
        valid_analysis = bool(
            selected_evaluation is not None
            and comparison.selected_candidate_id
            == selected_evaluation.scope.candidate_id
            and analysis.selected_candidate_id == expected_analysis_selected
            and analysis.search_parent_candidate_id
            == comparison.selected_candidate_id
            and (
                certified_candidate_id is None
                and analysis.outcome != "promoted"
                and analysis.champion_candidate_id is None
                or certified_candidate_id is not None
                and analysis.outcome == "promoted"
                and analysis.champion_candidate_id == certified_candidate_id
                and analysis.incumbent_after_candidate_id
                == certified_candidate_id
                and isinstance(champion_row, Mapping)
                and champion_row.get("certification_eligible") is True
                and champion_row.get("primary_selection_gate") is True
            )
        )
        if not valid_analysis or actual_approved is not expected_approved:
            raise ValueError(
                "runtime-v3 promotion differs from the strict-certification decision"
            )
        return

    selected_is_finalist = any(
        item.scope.candidate_id == comparison.selected_candidate_id
        and item.scope.holdout_arm
        in {HoldoutArm.FINALIST_1, HoldoutArm.FINALIST_2}
        for item in comparison.holdout_evaluations
    )
    expected_approved = bool(
        selected_is_finalist
        and candidate.candidate_id == comparison.selected_candidate_id
    )
    if (promotion.decision is PromotionDecision.APPROVED) is not expected_approved:
        raise ValueError(
            "runtime-v2 promotion differs from the recorded generation champion"
        )
    if expected_approved:
        champion_row = next(
            (
                row
                for row in analysis.ranking
                if row.get("candidate_id") == candidate.candidate_id
            ),
            None,
        )
        if (
            analysis.outcome != "promoted"
            or analysis.selected_candidate_id != candidate.candidate_id
            or analysis.champion_candidate_id != candidate.candidate_id
            or analysis.incumbent_after_candidate_id != candidate.candidate_id
            or champion_row is None
            or champion_row.get("primary_selection_gate") is not True
        ):
            raise ValueError(
                "runtime-v2 approval is not bound to the generation analysis"
            )


def validate_generation_comparison_binding(
    task: TaskManifest,
    run_id: str,
    holdout: GenerationHoldout,
    formal: Any,
    comparison: GenerationComparison,
    *,
    persisted_evaluations: Mapping[HoldoutArm, HoldoutEvaluation],
    persisted_judgments: Mapping[str, Evaluation] | None = None,
) -> None:
    """Bind a comparison to this run's exact durable evidence and Host gates."""

    if holdout.run_id != run_id or comparison.run_id != run_id:
        raise ValueError("generation comparison belongs to another run")
    if (
        comparison.generation != holdout.generation
        or comparison.cohort_digest != holdout.cohort_digest
    ):
        raise ValueError("generation comparison holdout is missing")
    embedded = {
        item.scope.holdout_arm: item for item in comparison.holdout_evaluations
    }
    arms = set(map(HoldoutArm, holdout.arm_bindings))
    if set(embedded) != arms or set(persisted_evaluations) != arms:
        raise ValueError("generation comparison requires all three holdout arms")
    if any(
        embedded[arm].to_dict() != persisted_evaluations[arm].to_dict()
        for arm in arms
    ):
        raise ValueError(
            "generation comparison evaluations differ from durable holdout evidence"
        )
    validate_exploration_generation_comparison(task, formal, comparison)
    from .finalist_review import finalist_review_evidence, finalist_review_qualification_required
    review_required = finalist_review_qualification_required(
        task.metadata, comparison.gate_results.get("finalist_review_qualification")
    )
    reviews = (
        finalist_review_evidence(comparison.holdout_evaluations, persisted_judgments or {},
                                task.metadata["review_model_id"])
        if review_required else None
    )
    if canonical_json(comparison.to_dict()["gate_results"].get("finalist_reviews")) != canonical_json(reviews):
        raise ValueError("generation comparison differs from durable independent judgments")
    if not uses_global_incumbent_protocol(task):
        return

    from ..evaluators.fitness import FitnessProfile
    from ..evaluators.generation_comparison import build_generation_comparison

    formal_payload = getattr(formal, "payload", formal)
    exploration_only = bool(
        isinstance(formal_payload, Mapping)
        and formal_payload.get("exploration_only") is True
    )
    legacy_runtime_v2_shape = bool(
        not uses_positive_delta_search_protocol(task)
        and "selection_policy" not in comparison.gate_results
    )
    expected = build_generation_comparison(
        quick_experiment=OptimizationSchedule.from_dict(task.metadata.get("optimization_schedule", OptimizationSchedule.default().to_dict())).quick,
        finalist_reviews=reviews,
        run_id=run_id,
        generation=comparison.generation,
        cohort_digest=comparison.cohort_digest,
        holdout_evaluations=comparison.holdout_evaluations,
        incumbent_candidate_id=holdout.arm_bindings[
            HoldoutArm.INCUMBENT.value
        ]["candidate_id"],
        fitness_profile=FitnessProfile.from_task(task),
        challenger_promotion_allowed=not exploration_only,
        positive_delta_search=uses_positive_delta_search_protocol(task),
        legacy_runtime_v2_shape=legacy_runtime_v2_shape,
        require_paired_strict_chain=paired_execution_qualification_required(
            task.metadata, comparison.gate_results.get("paired_execution_qualification")
        ),
    )
    if comparison.identity_dict() != expected.identity_dict():
        raise ValueError(
            "generation comparison differs from deterministic Host comparison"
        )


def persisted_genome_from_proposal(proposal: Proposal):
    """Parse and verify the canonical source genome stored on a new proposal."""

    from ..evolution.genome import EcologyEvolutionPluginGenome

    metadata = proposal.metadata
    if metadata.get("execution_protocol") != DSH_NATIVE_EVOLUTION_PROTOCOL:
        return None
    canonical = metadata.get("evolution_genome_canonical_json")
    if not isinstance(canonical, str) or not canonical:
        raise ValueError("DSH-native proposal requires canonical genome JSON")
    try:
        import json

        raw = json.loads(canonical)
    except (TypeError, ValueError) as exc:
        raise ValueError("proposal canonical genome JSON is invalid") from exc
    try:
        genome = EcologyEvolutionPluginGenome.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("proposal genome schema is invalid; legacy projection requires migration") from exc
    if canonical_json(genome.to_dict()) != canonical:
        raise ValueError("proposal genome JSON is not canonical")
    if metadata.get("genome_digest") != genome.genome_digest:
        raise ValueError("proposal genome_digest mismatch")
    if metadata.get("behavior_digest") != genome.behavior_digest:
        raise ValueError("proposal behavior_digest mismatch")
    return genome


_IDENTITY_BINDING_FIELDS = frozenset(
    {
        "execution_protocol",
        "genome_digest",
        "behavior_digest",
        "compiled_behavior_digest",
        "phenotype_instance_digest",
        "compiler_semantic_digest",
        "registry_catalog_digest",
        "security_semantic_digest",
        "runtime_execution_digest",
        "evaluation_cohort_digest",
    }
)


def validate_identity_binding(
    value: Any,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_BINDING_FIELDS:
        raise ValueError("DSH-native identity binding is incomplete")
    result: dict[str, str] = {}
    for name in _IDENTITY_BINDING_FIELDS:
        item = value[name]
        if name == "execution_protocol":
            if item != DSH_NATIVE_EVOLUTION_PROTOCOL:
                raise ValueError("identity binding execution protocol mismatch")
            result[name] = str(item)
            continue
        if (
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
        ):
            raise ValueError(f"identity binding {name} must be a SHA-256 digest")
        result[name] = item
    if expected is not None and dict(expected) != result:
        differing = sorted(
            name for name in _IDENTITY_BINDING_FIELDS if expected.get(name) != result.get(name)
        )
        label = differing[0] if differing else "unknown"
        raise ValueError(f"identity binding mismatch: {label}")
    return result


def _expected_seed_canonical(created_payload: Mapping[str, Any], task: TaskManifest) -> str | None:
    if not is_dsh_native_protocol(task):
        return None
    from ..evolution.genome import EcologyEvolutionPluginGenome

    initialization = created_payload.get("genome_initialization")
    if not isinstance(initialization, Mapping):
        raise ValueError("DSH-native RunCreated is missing genome initialization")
    required = {
        "schema_version",
        "materializer_version",
        "seed_template_canonical_json",
        "seed_template_digest",
        "materialization_input",
        "expected_seed_canonical_json",
        "expected_seed_genome_digest",
    }
    if set(initialization) != required:
        raise ValueError("RunCreated genome initialization fields are invalid")
    canonical = initialization["expected_seed_canonical_json"]
    if not isinstance(canonical, str):
        raise ValueError("RunCreated expected seed must be canonical JSON")
    try:
        import json

        genome = EcologyEvolutionPluginGenome.from_dict(json.loads(canonical))
    except (TypeError, ValueError) as exc:
        raise ValueError("RunCreated expected seed genome is invalid") from exc
    if canonical_json(genome.to_dict()) != canonical:
        raise ValueError("RunCreated expected seed JSON is not canonical")
    if initialization["expected_seed_genome_digest"] != genome.genome_digest:
        raise ValueError("RunCreated expected seed genome digest mismatch")
    return canonical

_TERMINAL_RUN_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.CANCELLED, RunStatus.FAILED}
)
_EVOLUTION_STAGES = frozenset(
    {
        "search",
        "research",
        "proposal",
        "candidate",
        "training",
        "evaluation",
        "judge",
        "decision",
        "reflection",
    }
)
_EVOLUTION_STAGE_STATUSES = frozenset({"started", "completed", "failed"})
_EVOLUTION_STAGE_PAYLOAD_FIELDS = frozenset(
    {
        "generation",
        "proposal_id",
        "candidate_id",
        "stage",
        "status",
        "attempt",
        "public_error",
    }
)
_DSH_CONTINUITY_RESET_CONTRACT = "dsh_structured_success@1"
_GATEWAY_RETRY_ERROR_CODE_POLICIES = {
    "model_gateway": (
        frozenset(
            {
                "gateway_response_error",
                "gateway_unavailable",
                "output_truncated",
                "research_algorithm_contract_invalid",
                "response_choices_invalid",
                "response_content_type_invalid",
                "response_envelope_invalid",
                "response_format_invalid",
                "response_message_missing",
                "final_content_missing",
            }
        ),
        "gateway_response_error",
    ),
    "dsh_native_runtime": (
        frozenset(
            {
                "dsh_native_runtime_http_error",
                "dsh_native_runtime_not_ready",
                "dsh_native_runtime_transport_error",
                "dsh_native_runtime_unavailable",
                "structured_child_model_error",
                "structured_child_tool_protocol_error",
                "structured_role_operational_timeout",
                "provider_route_cooling_down",
            }
        ),
        "dsh_native_runtime_unavailable",
    ),
    "research_timeout": (frozenset({"timeout"}), "timeout"),
    "sample_result_persistence": (
        frozenset({"sample_result_callback_error"}),
        "sample_result_callback_error",
    ),
}
_GATEWAY_RETRY_V2_FIELDS = frozenset(
    {
        "schema_version",
        "run_incarnation",
        "generation",
        "stage",
        "retry_class",
        "breaker_epoch",
        "failure_id",
        "attempt_anchor_seq",
        "consecutive_failures",
        "retry_limit",
        "first_failure_at",
        "last_failure_at",
        "last_error_code",
        "retry_at",
        "delay_seconds",
        "attempt",
        "error_code",
        "reason",
    }
)
_GATEWAY_RETRY_V2_CONTINUITY_FIELDS = (
    _GATEWAY_RETRY_V2_FIELDS | {"continuity_reset_contract"}
)
_GATEWAY_CIRCUIT_PAUSE_FIELDS = frozenset(
    {
        "code",
        "retry_class",
        "generation",
        "stage",
        "breaker_epoch",
        "consecutive_failures",
        "retry_limit",
        "first_failure_at",
        "last_failure_at",
        "last_error_code",
        "suggested_action",
        "pause_trigger",
        "epoch_seconds",
        "epoch_deadline_at",
        "proposed_retry_at",
        "retry_delay_seconds",
    }
)
_GATEWAY_CIRCUIT_RESUME_FIELDS = frozenset(
    {
        "resume_origin",
        "origin_pause_event_id",
        "origin_breaker_epoch",
        "retry_class",
        "generation",
        "stage",
        "breaker_epoch",
    }
)
_EVALUATION_PROGRESS_SCHEMA_VERSION_V1 = "ecologyrsi-dsh.evaluation-progress/1"
_EVALUATION_PROGRESS_SCHEMA_VERSION_V2 = "ecologyrsi-dsh.evaluation-progress/2"
_EVALUATION_PROGRESS_SCHEMA_VERSION_V3 = "ecologyrsi-dsh.evaluation-progress/3"
_EVALUATION_PROGRESS_ROLES = frozenset({"planner", "repair", "critic"})
_EVALUATION_PROGRESS_PAYLOAD_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "generation",
        "proposal_id",
        "candidate_id",
        "role",
        "model_id",
        "batch_index",
        "batch_count",
        "batch_size",
        "completed_samples",
        "total_samples",
        "succeeded_samples",
        "failed_samples",
    }
)
_EVALUATION_PROGRESS_DIAGNOSTIC_FIELDS = frozenset(
    {
        "gateway_request_count",
        "adaptive_split_trigger_count",
        "adaptive_split_count",
        "adaptive_split_max_depth",
        "adaptive_split_recovered_samples",
        "adaptive_split_failed_samples",
    }
)
_EVALUATION_PROGRESS_PAYLOAD_FIELDS_V2 = (
    _EVALUATION_PROGRESS_PAYLOAD_FIELDS_V1
    | _EVALUATION_PROGRESS_DIAGNOSTIC_FIELDS
)
_EVALUATION_PROGRESS_PAYLOAD_FIELDS_V3 = (
    _EVALUATION_PROGRESS_PAYLOAD_FIELDS_V2
    | {
        "revision",
        "progress_id",
        "progress_kind",
        "in_flight_batches",
        "queued_batches",
    }
)
_MODEL_USAGE_SCHEMA_VERSION_V1 = "ecologyrsi-dsh.model-usage/1"
_MODEL_USAGE_SCHEMA_VERSION_V2 = "ecologyrsi-dsh.model-usage/2"
_MODEL_USAGE_ROLES = frozenset(
    {"planner", "repair", "critic", "proposal", "research", "judge"}
)
_MODEL_USAGE_PAYLOAD_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "generation",
        "candidate_id",
        "role",
        "model_id",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "gateway_request_count",
        "revision",
        "usage_index",
    }
)
_MODEL_USAGE_PAYLOAD_FIELDS_V2 = frozenset(
    {
        "schema_version",
        "generation",
        "candidate_id",
        "role",
        "model_id",
        "call_id",
        "logical_call_digest",
        "outcome",
        "usage_reported",
        "http_attempts",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "revision",
        "usage_index",
    }
)
_MAX_MODEL_USAGE_TOKENS = 1_000_000_000_000


def _bounded_machine_code(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and value
        and value == value.strip()
        and len(value) <= 80
        and value.replace("_", "").replace("-", "").isalnum()
    )


def gateway_retry_error_code(retry_class: str, value: Any) -> str:
    """Map untrusted boundary codes onto a Host-owned per-class vocabulary."""

    policy = _GATEWAY_RETRY_ERROR_CODE_POLICIES.get(retry_class)
    if policy is None:
        raise ValueError("unknown gateway retry class")
    allowed, generic = policy
    candidate = str(value or "").strip().lower()
    return candidate if candidate in allowed else generic


def _aware_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _validate_gateway_retry_v2_payload(payload: Mapping[str, Any]) -> None:
    payload_fields = set(payload)
    if payload_fields not in {
        _GATEWAY_RETRY_V2_FIELDS,
        _GATEWAY_RETRY_V2_CONTINUITY_FIELDS,
    }:
        raise ValueError("GatewayRetryScheduled v2 payload is invalid")
    run_incarnation = payload.get("run_incarnation")
    generation = payload.get("generation")
    breaker_epoch = payload.get("breaker_epoch")
    anchor = payload.get("attempt_anchor_seq")
    consecutive = payload.get("consecutive_failures")
    retry_limit = payload.get("retry_limit")
    delay = payload.get("delay_seconds")
    failure_id = payload.get("failure_id")
    if (
        payload.get("schema_version") != GATEWAY_RETRY_SCHEMA_VERSION
        or isinstance(run_incarnation, bool)
        or not isinstance(run_incarnation, int)
        or run_incarnation < 1
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
        or not _bounded_machine_code(payload.get("stage"))
        or payload.get("retry_class") not in GATEWAY_RETRY_CLASSES
        or isinstance(breaker_epoch, bool)
        or not isinstance(breaker_epoch, int)
        or breaker_epoch < 1
        or not isinstance(failure_id, str)
        or len(failure_id) != 64
        or any(character not in "0123456789abcdef" for character in failure_id)
        or isinstance(anchor, bool)
        or not isinstance(anchor, int)
        or anchor < run_incarnation
        or isinstance(consecutive, bool)
        or not isinstance(consecutive, int)
        or consecutive < 1
        or isinstance(retry_limit, bool)
        or not isinstance(retry_limit, int)
        or not 3 <= retry_limit <= 12
        or consecutive >= retry_limit
        or payload.get("attempt") != consecutive
        or not _bounded_machine_code(payload.get("last_error_code"))
        or gateway_retry_error_code(
            str(payload.get("retry_class")),
            payload.get("last_error_code"),
        )
        != payload.get("last_error_code")
        or payload.get("error_code") != payload.get("last_error_code")
        or isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or not math.isfinite(float(delay))
        or float(delay) < 0
        or payload.get("reason")
        != GATEWAY_RETRY_POLICIES[payload.get("retry_class")].public_reason
        or (
            "continuity_reset_contract" in payload
            and (
                payload.get("retry_class") != "dsh_native_runtime"
                or payload.get("continuity_reset_contract")
                != _DSH_CONTINUITY_RESET_CONTRACT
            )
        )
    ):
        raise ValueError("GatewayRetryScheduled v2 payload is invalid")
    first = _aware_timestamp(payload.get("first_failure_at"))
    last = _aware_timestamp(payload.get("last_failure_at"))
    retry_at = _aware_timestamp(payload.get("retry_at"))
    if (
        first is None
        or last is None
        or retry_at is None
        or last < first
        or retry_at < last
        or abs((retry_at - last).total_seconds() - float(delay)) > 0.0015
        or (retry_at - first).total_seconds() >= GATEWAY_RETRY_EPOCH_SECONDS
    ):
        raise ValueError("GatewayRetryScheduled v2 payload is invalid")


def _validate_gateway_circuit_pause_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != _GATEWAY_CIRCUIT_PAUSE_FIELDS:
        raise ValueError("gateway circuit pause payload is invalid")
    generation = payload.get("generation")
    breaker_epoch = payload.get("breaker_epoch")
    consecutive = payload.get("consecutive_failures")
    retry_limit = payload.get("retry_limit")
    pause_trigger = payload.get("pause_trigger")
    epoch_seconds = payload.get("epoch_seconds")
    retry_delay = payload.get("retry_delay_seconds")
    policy = GATEWAY_RETRY_POLICIES.get(payload.get("retry_class"))
    if (
        policy is None
        or payload.get("code") != policy.circuit_code
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
        or not _bounded_machine_code(payload.get("stage"))
        or isinstance(breaker_epoch, bool)
        or not isinstance(breaker_epoch, int)
        or breaker_epoch < 1
        or isinstance(consecutive, bool)
        or not isinstance(consecutive, int)
        or consecutive < 1
        or isinstance(retry_limit, bool)
        or not isinstance(retry_limit, int)
        or not 3 <= retry_limit <= 12
        or consecutive > retry_limit
        or not _bounded_machine_code(payload.get("last_error_code"))
        or gateway_retry_error_code(
            str(payload.get("retry_class")),
            payload.get("last_error_code"),
        )
        != payload.get("last_error_code")
        or payload.get("suggested_action") != policy.suggested_action
        or pause_trigger
        not in {
            "failure_limit",
            "epoch_elapsed",
            "retry_deadline_reaches_epoch",
        }
        or isinstance(epoch_seconds, bool)
        or not isinstance(epoch_seconds, int)
        or isinstance(retry_delay, bool)
        or not isinstance(retry_delay, (int, float))
        or not math.isfinite(float(retry_delay))
        or not 0 <= float(retry_delay) <= GATEWAY_RETRY_MAX_DELAY_SECONDS
    ):
        raise ValueError("gateway circuit pause payload is invalid")
    first = _aware_timestamp(payload.get("first_failure_at"))
    last = _aware_timestamp(payload.get("last_failure_at"))
    epoch_deadline = _aware_timestamp(payload.get("epoch_deadline_at"))
    proposed_retry = _aware_timestamp(payload.get("proposed_retry_at"))
    if (
        first is None
        or last is None
        or epoch_deadline is None
        or proposed_retry is None
    ):
        raise ValueError("gateway circuit pause payload is invalid")


def _validate_gateway_circuit_pause_trigger_evidence(
    payload: Mapping[str, Any],
    *,
    first_failure_event: Event,
    pause_event: Event,
) -> None:
    first = _aware_timestamp(first_failure_event.created_at)
    last = _aware_timestamp(pause_event.created_at)
    payload_first = _aware_timestamp(payload.get("first_failure_at"))
    payload_last = _aware_timestamp(payload.get("last_failure_at"))
    payload_epoch_deadline = _aware_timestamp(payload.get("epoch_deadline_at"))
    payload_proposed_retry = _aware_timestamp(payload.get("proposed_retry_at"))
    epoch_seconds = int(payload["epoch_seconds"])
    retry_delay = float(payload["retry_delay_seconds"])
    if first is None or last is None or last < first:
        raise ValueError("gateway circuit pause trigger evidence is invalid")
    epoch_deadline = first + timedelta(seconds=epoch_seconds)
    proposed_retry = last + timedelta(seconds=retry_delay)
    elapsed_seconds = (last - first).total_seconds()
    evidence_valid = (
        epoch_seconds == GATEWAY_RETRY_EPOCH_SECONDS
        and payload_first == first
        and payload_last == last
        and payload_epoch_deadline == epoch_deadline
        and payload_proposed_retry == proposed_retry
    )
    pause_trigger = payload["pause_trigger"]
    consecutive = int(payload["consecutive_failures"])
    retry_limit = int(payload["retry_limit"])
    if pause_trigger == "failure_limit":
        evidence_valid = evidence_valid and consecutive == retry_limit
    elif pause_trigger == "epoch_elapsed":
        evidence_valid = (
            evidence_valid
            and consecutive < retry_limit
            and elapsed_seconds >= epoch_seconds
        )
    else:
        evidence_valid = (
            evidence_valid
            and consecutive < retry_limit
            and elapsed_seconds < epoch_seconds
            and proposed_retry >= epoch_deadline
        )
    if not evidence_valid:
        raise ValueError("gateway circuit pause trigger evidence is invalid")


def _validate_gateway_circuit_resume_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != _GATEWAY_CIRCUIT_RESUME_FIELDS:
        raise ValueError("gateway circuit resume payload is invalid")
    generation = payload.get("generation")
    breaker_epoch = payload.get("breaker_epoch")
    origin_epoch = payload.get("origin_breaker_epoch")
    policy = GATEWAY_RETRY_POLICIES.get(payload.get("retry_class"))
    if (
        policy is None
        or payload.get("resume_origin") != policy.circuit_code
        or not isinstance(payload.get("origin_pause_event_id"), str)
        or not payload["origin_pause_event_id"].strip()
        or payload.get("retry_class") not in GATEWAY_RETRY_CLASSES
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
        or not _bounded_machine_code(payload.get("stage"))
        or isinstance(breaker_epoch, bool)
        or not isinstance(breaker_epoch, int)
        or breaker_epoch < 2
        or isinstance(origin_epoch, bool)
        or not isinstance(origin_epoch, int)
        or origin_epoch < 1
        or breaker_epoch != origin_epoch + 1
    ):
        raise ValueError("gateway circuit resume payload is invalid")


def validate_evolution_stage_payload(payload: Mapping[str, Any]) -> None:
    fields = set(payload)
    if fields != _EVOLUTION_STAGE_PAYLOAD_FIELDS:
        missing = sorted(_EVOLUTION_STAGE_PAYLOAD_FIELDS - fields)
        unexpected = sorted(fields - _EVOLUTION_STAGE_PAYLOAD_FIELDS)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(
            "invalid EvolutionStageRecorded payload fields: " + "; ".join(details)
        )
    generation = payload["generation"]
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("EvolutionStageRecorded generation must be a non-negative integer")
    for field_name in ("proposal_id", "candidate_id"):
        value = payload[field_name]
        if value is not None and (
            not isinstance(value, str) or not value.strip() or value != value.strip()
        ):
            raise ValueError(
                f"EvolutionStageRecorded {field_name} must be null or a non-empty trimmed string"
            )
    if payload["stage"] not in _EVOLUTION_STAGES:
        raise ValueError(f"unknown evolution stage: {payload['stage']}")
    if payload["status"] not in _EVOLUTION_STAGE_STATUSES:
        raise ValueError(f"unknown evolution stage status: {payload['status']}")
    attempt = payload["attempt"]
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("EvolutionStageRecorded attempt must be an integer greater than zero")
    public_error = payload["public_error"]
    if public_error is not None and not isinstance(public_error, str):
        raise ValueError("EvolutionStageRecorded public_error must be null or a string")


def validate_evaluation_progress_payload(payload: Mapping[str, Any]) -> None:
    """Validate a label-free, aggregate progress heartbeat."""

    schema_version = payload.get("schema_version")
    if schema_version == _EVALUATION_PROGRESS_SCHEMA_VERSION_V1:
        expected_fields = _EVALUATION_PROGRESS_PAYLOAD_FIELDS_V1
    elif schema_version == _EVALUATION_PROGRESS_SCHEMA_VERSION_V2:
        expected_fields = _EVALUATION_PROGRESS_PAYLOAD_FIELDS_V2
    elif schema_version == _EVALUATION_PROGRESS_SCHEMA_VERSION_V3:
        expected_fields = _EVALUATION_PROGRESS_PAYLOAD_FIELDS_V3
    else:
        raise ValueError("unknown evaluation progress schema_version")
    if set(payload) != expected_fields:
        raise ValueError("invalid EvaluationProgressRecorded payload fields")
    generation = payload["generation"]
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("evaluation progress generation must be non-negative")
    for name in ("proposal_id", "candidate_id", "model_id"):
        value = payload[name]
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"evaluation progress {name} must be trimmed text")
    if payload["role"] not in _EVALUATION_PROGRESS_ROLES:
        raise ValueError("evaluation progress role is unsupported")
    counts: dict[str, int] = {}
    for name in (
        "batch_index",
        "batch_count",
        "batch_size",
        "completed_samples",
        "total_samples",
        "succeeded_samples",
        "failed_samples",
    ):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"evaluation progress {name} must be non-negative")
        counts[name] = value
    if schema_version != _EVALUATION_PROGRESS_SCHEMA_VERSION_V3 and not 1 <= counts[
        "batch_index"
    ] <= counts["batch_count"]:
        raise ValueError("evaluation progress batch position is invalid")
    if (
        (schema_version != _EVALUATION_PROGRESS_SCHEMA_VERSION_V3 and counts["batch_size"] < 1)
        or counts["total_samples"] < 1
    ):
        raise ValueError("evaluation progress batch and total sizes must be positive")
    if not 0 <= counts["batch_index"] <= counts["batch_count"]:
        raise ValueError("evaluation progress batch position is invalid")
    if not counts["completed_samples"] <= counts["total_samples"]:
        raise ValueError("evaluation progress completed sample count is invalid")
    if (
        schema_version != _EVALUATION_PROGRESS_SCHEMA_VERSION_V3
        and counts["batch_size"] > counts["completed_samples"]
    ):
        raise ValueError("evaluation progress batch size is invalid")
    if counts["succeeded_samples"] + counts["failed_samples"] != counts[
        "completed_samples"
    ]:
        raise ValueError("evaluation progress outcomes do not cover completed samples")
    if schema_version == _EVALUATION_PROGRESS_SCHEMA_VERSION_V1:
        return

    diagnostics: dict[str, int] = {}
    for name in sorted(_EVALUATION_PROGRESS_DIAGNOSTIC_FIELDS):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"evaluation progress {name} must be non-negative")
        diagnostics[name] = value
    if diagnostics["adaptive_split_max_depth"] > 8:
        raise ValueError("evaluation progress adaptive split depth is invalid")
    if diagnostics["adaptive_split_count"] > diagnostics[
        "adaptive_split_trigger_count"
    ]:
        raise ValueError("evaluation progress split count exceeds its triggers")
    # V3 batch_index counts durable outcome publications, not HTTP calls. One
    # remote response can be published in several planner/repair result batches,
    # so the two counters are intentionally independent for resumable execution.
    if (
        schema_version != _EVALUATION_PROGRESS_SCHEMA_VERSION_V3
        and diagnostics["gateway_request_count"] < counts["batch_index"]
    ):
        raise ValueError("evaluation progress gateway request count is invalid")
    if (
        diagnostics["adaptive_split_recovered_samples"]
        + diagnostics["adaptive_split_failed_samples"]
        > counts["completed_samples"]
    ):
        raise ValueError("evaluation progress split sample counts are invalid")
    if schema_version != _EVALUATION_PROGRESS_SCHEMA_VERSION_V3:
        return
    revision = payload["revision"]
    if not isinstance(revision, str) or not revision.strip() or revision != revision.strip():
        raise ValueError("evaluation progress revision must be trimmed text")
    for name in ("progress_id", "in_flight_batches", "queued_batches"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"evaluation progress {name} must be non-negative")
    if payload["progress_kind"] not in {"waiting", "completed_batch", "drained"}:
        raise ValueError("evaluation progress kind is unsupported")
    if payload["progress_kind"] == "completed_batch":
        if not 1 <= counts["batch_index"] <= counts["batch_count"]:
            raise ValueError("evaluation progress completed batch position is invalid")
        if counts["batch_size"] < 1 or counts["batch_size"] > counts["completed_samples"]:
            raise ValueError("evaluation progress completed batch size is invalid")
    elif counts["batch_size"] != 0:
        raise ValueError("evaluation progress waiting heartbeat batch size must be zero")


def validate_model_usage_payload(payload: Mapping[str, Any]) -> None:
    """Validate one public, call-level model usage delta."""

    schema_version = payload.get("schema_version")
    if schema_version == _MODEL_USAGE_SCHEMA_VERSION_V1:
        expected_fields = _MODEL_USAGE_PAYLOAD_FIELDS_V1
    elif schema_version == _MODEL_USAGE_SCHEMA_VERSION_V2:
        expected_fields = _MODEL_USAGE_PAYLOAD_FIELDS_V2
    else:
        raise ValueError("unknown model usage schema_version")
    if set(payload) != expected_fields:
        raise ValueError("invalid ModelUsageRecorded payload fields")
    generation = payload["generation"]
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("model usage generation must be non-negative")
    for name, maximum in (
        ("candidate_id", 320),
        ("model_id", 320),
        ("revision", 320),
    ):
        value = payload[name]
        if (
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            or len(value) > maximum
        ):
            raise ValueError(f"model usage {name} must be bounded trimmed text")
    if payload["role"] not in _MODEL_USAGE_ROLES:
        raise ValueError("model usage role is unsupported")
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = payload[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= _MAX_MODEL_USAGE_TOKENS
        ):
            raise ValueError(f"model usage {name} must be a bounded non-negative integer")
    request_count = (
        payload["gateway_request_count"]
        if schema_version == _MODEL_USAGE_SCHEMA_VERSION_V1
        else payload["http_attempts"]
    )
    if (
        isinstance(request_count, bool)
        or not isinstance(request_count, int)
        or request_count < (1 if schema_version == _MODEL_USAGE_SCHEMA_VERSION_V1 else 0)
    ):
        if schema_version == _MODEL_USAGE_SCHEMA_VERSION_V1:
            raise ValueError("model usage gateway_request_count must be positive")
        raise ValueError("model usage http_attempts is invalid")
    if schema_version == _MODEL_USAGE_SCHEMA_VERSION_V2:
        for name, maximum in (("call_id", 320), ("outcome", 64)):
            value = payload[name]
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
                or len(value) > maximum
            ):
                raise ValueError(f"model usage {name} must be bounded trimmed text")
        logical_call_digest = payload["logical_call_digest"]
        if (
            not isinstance(logical_call_digest, str)
            or len(logical_call_digest) != 64
            or any(character not in "0123456789abcdef" for character in logical_call_digest)
        ):
            raise ValueError("model usage logical_call_digest must be a SHA-256 digest")
        if payload["outcome"] not in {"succeeded", "failed"}:
            raise ValueError("model usage outcome is unsupported")
        if not isinstance(payload["usage_reported"], bool):
            raise ValueError("model usage usage_reported must be a boolean")
    usage_index = payload["usage_index"]
    if isinstance(usage_index, bool) or not isinstance(usage_index, int) or usage_index < 0:
        raise ValueError("model usage usage_index must be non-negative")


@dataclass(frozen=True, slots=True)
class RunState:
    run: Run
    task_manifest: TaskManifest
    proposals: tuple[Proposal, ...]
    candidates: tuple[Candidate, ...]
    artifacts: tuple[ModelArtifact, ...]
    evaluations: tuple[Evaluation, ...]
    promotions: tuple[Promotion, ...]
    interventions: tuple[HumanIntervention, ...]
    generation_batches: tuple[GenerationBatch, ...]
    generation_analyses: tuple[GenerationAnalysis, ...]
    knowledge_snapshots: tuple[KnowledgeSnapshot, ...]
    knowledge_assessments: tuple[KnowledgeAssessment, ...]
    research_iterations: tuple[ResearchIteration, ...]
    algorithm_attempts: tuple[AlgorithmAttempt, ...]
    events: tuple[Event, ...]
    generation_search_plans: tuple[GenerationSearchPlan, ...] = ()
    generation_reflections: tuple[GenerationReflection, ...] = ()
    expert_consultations: tuple[ExpertConsultation, ...] = ()
    expert_consultation_answers: tuple[ExpertConsultationAnswer, ...] = ()
    materialized_seed_genome_canonical_json: str | None = None
    candidate_identity_bindings: tuple[Mapping[str, Any], ...] = ()
    formal_stage_seals: tuple[Mapping[str, Any], ...] = ()
    candidate_screening_events: tuple[Event, ...] = ()
    formal_selection_events: tuple[Event, ...] = ()
    screened_out_events: tuple[Event, ...] = ()
    candidate_revisions: tuple[CandidateRevision, ...] = ()
    formal_trajectories: tuple[FormalTrajectory, ...] = ()
    formal_batches: tuple[FormalBatch, ...] = ()
    formal_batch_evaluations: tuple[BatchEvaluation, ...] = ()
    formal_batch_comparisons: tuple[FormalBatchComparison, ...] = ()
    local_edit_proposals: tuple[Mapping[str, Any], ...] = ()
    local_edit_outcomes: tuple[Mapping[str, Any], ...] = ()
    trajectory_revision_activations: tuple[TrajectoryRevisionActivation, ...] = ()
    generation_holdouts: tuple[GenerationHoldout, ...] = ()
    holdout_evaluations: tuple[HoldoutEvaluation, ...] = ()
    generation_comparisons: tuple[GenerationComparison, ...] = ()
    effective_revision_bindings: tuple[Mapping[str, Any], ...] = ()
    run_adaptation_cohort: RunAdaptationCohort | None = None
    generation_selection_cohorts: tuple[GenerationCohorts, ...] = ()

    def proposal(self, proposal_id: str) -> Proposal:
        for item in self.proposals:
            if item.proposal_id == proposal_id:
                return item
        raise KeyError(f"unknown proposal: {proposal_id}")

    def candidate(self, candidate_id: str) -> Candidate:
        for item in self.candidates:
            if item.candidate_id == candidate_id:
                return item
        raise KeyError(f"unknown candidate: {candidate_id}")

    def screening_for(self, generation: int, candidate_id: str) -> Event | None:
        return next(
            (
                event
                for event in reversed(self.candidate_screening_events)
                if event.payload["generation"] == generation
                and event.payload["candidate_id"] == candidate_id
            ),
            None,
        )

    def generation_cohort_for(self, generation: int) -> GenerationCohorts | None:
        return next(
            (
                item
                for item in reversed(self.generation_selection_cohorts)
                if item.generation == generation
            ),
            None,
        )

    def formal_selection_for(self, generation: int) -> Event | None:
        return next(
            (
                event
                for event in reversed(self.formal_selection_events)
                if event.payload["generation"] == generation
            ),
            None,
        )

    def revision(self, revision_id: str) -> CandidateRevision:
        for item in self.candidate_revisions:
            if item.revision_id == revision_id:
                return item
        raise KeyError(f"unknown candidate revision: {revision_id}")

    def initial_revision_for(self, candidate_id: str) -> CandidateRevision | None:
        return next(
            (
                item
                for item in self.candidate_revisions
                if item.candidate_id == candidate_id
                and item.parent_revision_id is None
            ),
            None,
        )

    def trajectory_for(self, candidate_id: str) -> FormalTrajectory | None:
        return next(
            (
                item
                for item in reversed(self.formal_trajectories)
                if item.candidate_id == candidate_id
            ),
            None,
        )

    def formal_batch_for(
        self, candidate_id: str, batch_index: int
    ) -> FormalBatch | None:
        return next(
            (
                item
                for item in reversed(self.formal_batches)
                if item.candidate_id == candidate_id
                and item.batch_index == batch_index
            ),
            None,
        )

    def batch_evaluation_for(
        self,
        candidate_id: str,
        batch_index: int,
        arm: FormalBatchArm | None = None,
    ) -> BatchEvaluation | None:
        matches = tuple(
            item
            for item in reversed(self.formal_batch_evaluations)
            if item.scope.candidate_id == candidate_id
            and item.scope.batch_index == batch_index
        )
        if arm is not None:
            return next(
                (
                    item
                    for item in matches
                    if item.scope.formal_batch_arm is arm
                ),
                None,
            )
        legacy = next(
            (
                item
                for item in matches
                if item.scope.formal_batch_arm is None
            ),
            None,
        )
        if legacy is not None:
            return legacy
        scheduled_arm = (
            FormalBatchArm.CHAMPION
            if batch_index == 0
            else FormalBatchArm.CHALLENGER
        )
        scheduled = next(
            (
                item
                for item in matches
                if item.scope.formal_batch_arm is scheduled_arm
            ),
            None,
        )
        if scheduled is not None:
            return scheduled
        comparison = self.batch_comparison_for(candidate_id, batch_index)
        if (
            comparison is not None
            and comparison.champion_evaluation_id
            == comparison.challenger_evaluation_id
        ):
            return next(
                (
                    item
                    for item in matches
                    if item.evaluation_id == comparison.champion_evaluation_id
                ),
                None,
            )
        return None

    def batch_comparison_for(
        self,
        candidate_id: str,
        batch_index: int,
    ) -> FormalBatchComparison | None:
        return next(
            (
                item
                for item in reversed(self.formal_batch_comparisons)
                if item.candidate_id == candidate_id
                and item.batch_index == batch_index
            ),
            None,
        )

    def trajectory_champion_revision_id(self, candidate_id: str) -> str:
        trajectory = self.trajectory_for(candidate_id)
        if trajectory is None:
            raise KeyError(f"unknown formal trajectory: {candidate_id}")
        comparisons = sorted(
            (
                item
                for item in self.formal_batch_comparisons
                if item.candidate_id == candidate_id
            ),
            key=lambda item: item.batch_index,
        )
        return (
            comparisons[-1].champion_after_revision_id
            if comparisons
            else trajectory.initial_revision_id
        )

    def local_edit_proposal_for(
        self, candidate_id: str, batch_index: int
    ) -> Mapping[str, Any] | None:
        return next(
            (
                item
                for item in reversed(self.local_edit_proposals)
                if item.get("candidate_id") == candidate_id
                and item.get("batch_index") == batch_index
            ),
            None,
        )

    def revision_activation_for(
        self, candidate_id: str, batch_index: int
    ) -> TrajectoryRevisionActivation | None:
        return next(
            (
                item
                for item in reversed(self.trajectory_revision_activations)
                if item.candidate_id == candidate_id
                and item.batch_index == batch_index
            ),
            None,
        )

    def generation_holdout_for(self, generation: int) -> GenerationHoldout | None:
        return next(
            (
                item
                for item in reversed(self.generation_holdouts)
                if item.generation == generation
            ),
            None,
        )

    def holdout_evaluation_for(
        self, generation: int, arm: HoldoutArm
    ) -> HoldoutEvaluation | None:
        return next(
            (
                item
                for item in reversed(self.holdout_evaluations)
                if item.scope.generation == generation
                and item.scope.holdout_arm == arm
            ),
            None,
        )

    def comparison_for(self, generation: int) -> GenerationComparison | None:
        return next(
            (
                item
                for item in reversed(self.generation_comparisons)
                if item.generation == generation
            ),
            None,
        )

    def effective_revision_for(self, generation: int) -> str | None:
        binding = next(
            (
                item
                for item in reversed(self.effective_revision_bindings)
                if item.get("generation") == generation
            ),
            None,
        )
        return str(binding["selected_revision_id"]) if binding is not None else None

    def evaluation_for(self, candidate_id: str) -> Evaluation | None:
        return next(
            (item for item in reversed(self.evaluations) if item.candidate_id == candidate_id),
            None,
        )

    def artifact_for(self, candidate_id: str) -> ModelArtifact | None:
        return next(
            (item for item in reversed(self.artifacts) if item.candidate_id == candidate_id),
            None,
        )

    def promotion_for(self, candidate_id: str) -> Promotion | None:
        return next(
            (item for item in reversed(self.promotions) if item.candidate_id == candidate_id),
            None,
        )

    def batch_for(self, generation: int) -> GenerationBatch | None:
        return next(
            (item for item in reversed(self.generation_batches) if item.generation == generation),
            None,
        )

    def analysis_for(self, generation: int) -> GenerationAnalysis | None:
        return next(
            (item for item in reversed(self.generation_analyses) if item.generation == generation),
            None,
        )

    def knowledge_for(self, generation: int) -> KnowledgeSnapshot | None:
        return next(
            (
                item
                for item in reversed(self.knowledge_snapshots)
                if item.generation == generation
            ),
            None,
        )

    def search_plan_for(self, generation: int) -> GenerationSearchPlan | None:
        return next(
            (
                item
                for item in reversed(self.generation_search_plans)
                if item.generation == generation
            ),
            None,
        )

    def reflection_for(self, generation: int) -> GenerationReflection | None:
        return next(
            (
                item
                for item in reversed(self.generation_reflections)
                if item.generation == generation
            ),
            None,
        )

    def knowledge_assessment_for(
        self, generation: int
    ) -> KnowledgeAssessment | None:
        return next(
            (
                item
                for item in reversed(self.knowledge_assessments)
                if item.generation == generation
            ),
            None,
        )

    def algorithm_attempts_for(self, candidate_id: str) -> tuple[AlgorithmAttempt, ...]:
        return tuple(
            item for item in self.algorithm_attempts if item.candidate_id == candidate_id
        )

    def research_iteration_for(self, generation: int) -> ResearchIteration | None:
        return next(
            (
                item
                for item in reversed(self.research_iterations)
                if item.generation == generation
            ),
            None,
        )

    def compiled_algorithm_for(self, candidate_id: str) -> Mapping[str, Any] | None:
        for item in reversed(self.algorithm_attempts_for(candidate_id)):
            if item.phase == "compile" and item.status == "passed":
                return item.algorithm_spec
        return None

    def materialized_seed_genome(self):
        from ..evolution.genome import EcologyEvolutionPluginGenome

        if self.materialized_seed_genome_canonical_json is None:
            raise RuntimeError("run seed genome has not been materialized")
        import json

        return EcologyEvolutionPluginGenome.from_dict(
            json.loads(self.materialized_seed_genome_canonical_json)
        )

    def persisted_genome_for(self, candidate_id: str):
        candidate = self.candidate(candidate_id)
        proposal = self.proposal(candidate.proposal_id)
        genome = persisted_genome_from_proposal(proposal)
        if genome is None:
            raise ValueError("historical candidate has no persisted DSH-native genome")
        return genome

    def parent_genome_for_generation(self, generation: int):
        batch = self.batch_for(generation)
        if batch is not None and batch.parent_genome_canonical_json is not None:
            from ..evolution.genome import EcologyEvolutionPluginGenome

            import json

            return EcologyEvolutionPluginGenome.from_dict(
                json.loads(batch.parent_genome_canonical_json)
            )
        if generation == 0:
            return self.materialized_seed_genome()
        if batch is None:
            if (
                self.task_manifest.metadata.get("optimization_protocol")
                in ADAPTIVE_PROTOCOLS
            ):
                from ..evolution.genome import (
                    EcologyEvolutionPluginGenome,
                    deep_thaw_json,
                )

                selected_revision_id = self.effective_revision_for(generation - 1)
                if selected_revision_id is None:
                    raise ValueError(
                        "adaptive generation has no frozen effective parent revision"
                    )
                selected_revision = self.revision(selected_revision_id)
                return EcologyEvolutionPluginGenome.from_dict(
                    deep_thaw_json(selected_revision.genome)
                )
            previous = self.analysis_for(generation - 1)
            if (
                previous is not None
                and previous.search_parent_candidate_id is not None
            ):
                return self.persisted_genome_for(
                    previous.search_parent_candidate_id
                )
            raise ValueError("generation has no persisted parent genome")
        if batch.parent_candidate_id is None:
            raise ValueError("generation has no persisted parent genome")
        return self.persisted_genome_for(batch.parent_candidate_id)

    def candidate_identity_binding(
        self, candidate_id: str
    ) -> Mapping[str, Any] | None:
        return next(
            (
                dict(item["identity_binding"])
                for item in reversed(self.candidate_identity_bindings)
                if item.get("candidate_id") == candidate_id
            ),
            None,
        )

    def artifact_revision_binding(self, artifact_id: str) -> Mapping[str, Any] | None:
        return next((dict(event.payload["artifact_revision_binding"])
                     for event in reversed(self.events)
                     if event.kind == "ArtifactRecorded"
                     and event.payload.get("schema_version") == ARTIFACT_EVENT_V2
                     and event.payload.get("artifact", {}).get("artifact_id") == artifact_id), None)

    def candidate_duplicate_signature(self, candidate_id: str) -> str:
        binding = self.candidate_identity_binding(candidate_id)
        if binding is None:
            raise ValueError("candidate has no compiled behavior identity")
        return digest(
            {
                "compiled_behavior_digest": binding["compiled_behavior_digest"],
                "evaluation_cohort_digest": binding["evaluation_cohort_digest"],
            }
        )

    @property
    def pending_interventions(self) -> tuple[HumanIntervention, ...]:
        return tuple(item for item in self.interventions if item.applied_proposal_id is None)

    def consultation(self, consultation_id: str) -> ExpertConsultation:
        for item in self.expert_consultations:
            if item.consultation_id == consultation_id:
                return item
        raise KeyError(f"unknown expert consultation: {consultation_id}")

    def answer_for_consultation(
        self, consultation_id: str
    ) -> ExpertConsultationAnswer | None:
        return next(
            (
                item
                for item in reversed(self.expert_consultation_answers)
                if item.consultation_id == consultation_id
            ),
            None,
        )

    @property
    def pending_expert_consultations(self) -> tuple[ExpertConsultation, ...]:
        return tuple(
            item
            for item in self.expert_consultations
            if self.answer_for_consultation(item.consultation_id) is None
        )

    def available_expert_answers(
        self, generation: int
    ) -> tuple[ExpertConsultationAnswer, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self.expert_consultation_answers
                    if item.effective_generation is not None
                    and item.effective_generation <= generation
                    and item.applied_generation is None
                ),
                key=lambda item: (item.created_at, item.answer_id),
            )
        )


class RunStateReducer:
    """Incremental projection of one append-only run stream.

    State is private to the director's synchronized cache. Public snapshots are
    immutable domain records. Invalid tails discard the reducer; authoritative
    commands and audit exports continue to read the event ledger.
    """
    schema_version = "ecologyrsi-dsh.run-state-reducer/1"

    def __init__(self, created: Event) -> None:
        self.events = (created,)
        self.created = created
        if self.created.kind != "RunCreated":
            raise ValueError("run event stream must start with RunCreated")
        self.task = TaskManifest.from_dict(self.created.payload["task_manifest"])
        self.run = Run.from_dict(self.created.payload["run"])
        self.expected_seed_canonical = _expected_seed_canonical(self.created.payload, self.task)
        self.materialized_seed_canonical: str | None = None
        self.model_contract_preflight_recorded = False
        self.dsh_usage_launches: dict[str, Event] = {}
        self.dsh_usage_by_reservation: dict[str, list[Event]] = {}
        self.dsh_usage_by_session: dict[str, list[Event]] = {}
        self.proposals: dict[str, Proposal] = {}
        self.candidates: dict[str, Candidate] = {}
        self.artifacts: dict[str, ModelArtifact] = {}
        self.evaluations: dict[str, Evaluation] = {}
        self.promotions: dict[str, Promotion] = {}
        self.interventions: dict[str, HumanIntervention] = {}
        self.expert_consultations: dict[str, ExpertConsultation] = {}
        self.expert_consultation_answers: dict[str, ExpertConsultationAnswer] = {}
        self.generation_batches: dict[int, GenerationBatch] = {}
        self.generation_analyses: dict[int, GenerationAnalysis] = {}
        self.knowledge_snapshots: dict[int, KnowledgeSnapshot] = {}
        self.knowledge_assessments: dict[int, KnowledgeAssessment] = {}
        self.research_iterations: dict[int, ResearchIteration] = {}
        self.generation_search_plans: dict[int, GenerationSearchPlan] = {}
        self.generation_reflections: dict[int, GenerationReflection] = {}
        self.algorithm_attempts: list[AlgorithmAttempt] = []
        self.candidate_identity_bindings: dict[str, dict[str, Any]] = {}
        self.artifact_revision_bindings: dict[str, dict[str, Any]] = {}
        self.formal_stage_seals: dict[str, dict[str, Any]] = {}
        self.candidate_screening_events: dict[tuple[int, str], Event] = {}
        self.formal_selection_events: dict[int, Event] = {}
        self.screened_out_events: dict[tuple[int, str], Event] = {}
        self.candidate_revisions: dict[str, CandidateRevision] = {}
        self.formal_trajectories: dict[str, FormalTrajectory] = {}
        self.formal_batches: dict[tuple[str, int], FormalBatch] = {}
        self.formal_batch_evaluations: dict[
            tuple[str, int, FormalBatchArm | None], BatchEvaluation
        ] = {}
        self.formal_batch_comparisons: dict[
            tuple[str, int], FormalBatchComparison
        ] = {}
        self.local_edit_proposals: dict[tuple[str, int], dict[str, Any]] = {}
        self.local_edit_outcomes: dict[tuple[str, int], dict[str, Any]] = {}
        self.local_edit_proposal_events: dict[tuple[str, int], Event] = {}
        self.local_edit_outcome_events: dict[tuple[str, int], Event] = {}
        self.trajectory_revision_activations: dict[
            tuple[str, int], TrajectoryRevisionActivation
        ] = {}
        self.generation_holdouts: dict[int, GenerationHoldout] = {}
        self.holdout_evaluations: dict[tuple[int, HoldoutArm], HoldoutEvaluation] = {}
        self.generation_comparisons: dict[int, GenerationComparison] = {}
        self.effective_revision_bindings: dict[int, dict[str, Any]] = {}
        self.champion_generations: set[int] = set()
        self.run_adaptation_cohort: RunAdaptationCohort | None = None
        self.generation_selection_cohorts: dict[int, GenerationCohorts] = {}
        self.dsh_prediction_tool_events: dict[str, tuple[int, dict[str, Any]]] = {}
        self.formal_stage_started = False
        self.active_gateway_circuit_pause: Event | None = None
        self.gateway_retry_last_by_scope: dict[tuple[int, int, str, str], Event] = {}
        self.gateway_retry_first_by_scope: dict[tuple[int, int, str, str], Event] = {}
        self.gateway_retry_max_epoch: dict[tuple[int, int, str, str], int] = {}
        self.gateway_retry_failure_ids: set[str] = set()
        self.gateway_retry_stage_success_seq: dict[tuple[int, str], int] = {}
        self.gateway_retry_dsh_success_seq: dict[int, int] = {}
        self.latest_run_resumed: Event | None = None
        self.events_by_seq = {created.seq: created}


    def replay_batch_evaluation_for(
        self,
        candidate_id: str,
        batch_index: int,
        arm: FormalBatchArm | None = None,
    ) -> BatchEvaluation | None:
        if arm is not None:
            return self.formal_batch_evaluations.get(
                (candidate_id, batch_index, arm)
            )
        legacy = self.formal_batch_evaluations.get((candidate_id, batch_index, None))
        if legacy is not None:
            return legacy
        scheduled_arm = (
            FormalBatchArm.CHAMPION
            if batch_index == 0
            else FormalBatchArm.CHALLENGER
        )
        scheduled = self.formal_batch_evaluations.get(
            (candidate_id, batch_index, scheduled_arm)
        )
        if scheduled is not None:
            return scheduled
        comparison = self.formal_batch_comparisons.get((candidate_id, batch_index))
        if (
            comparison is not None
            and comparison.champion_evaluation_id
            == comparison.challenger_evaluation_id
        ):
            return next(
                (
                    item
                    for (key_candidate, key_index, _key_arm), item
                    in self.formal_batch_evaluations.items()
                    if key_candidate == candidate_id
                    and key_index == batch_index
                    and item.evaluation_id
                    == comparison.champion_evaluation_id
                ),
                None,
            )
        return None

    def gateway_retry_scope(self, payload: Mapping[str, Any]) -> tuple[int, int, str, str]:
        return (
            int(self.created.seq),
            int(payload["generation"]),
            str(payload["stage"]),
            str(payload["retry_class"]),
        )

    def gateway_retry_reset_seq(
        self,
        scope: tuple[int, int, str, str],
        *,
        dsh_continuity_reset: bool = False,
    ) -> int:
        _incarnation, generation, stage, retry_class = scope
        return max(
            int(self.created.seq),
            int(self.latest_run_resumed.seq) if self.latest_run_resumed is not None else 0,
            self.gateway_retry_stage_success_seq.get((generation, stage), 0),
            (
                self.gateway_retry_dsh_success_seq.get(generation, 0)
                if retry_class == "dsh_native_runtime"
                and dsh_continuity_reset
                else 0
            ),
        )

    def gateway_retry_expected_epoch(
        self,
        scope: tuple[int, int, str, str],
        *,
        reset_seq: int,
    ) -> int:
        _incarnation, generation, stage, retry_class = scope
        if (
            self.latest_run_resumed is not None
            and self.latest_run_resumed.seq == reset_seq
            and self.latest_run_resumed.payload.get("retry_class") == retry_class
            and self.latest_run_resumed.payload.get("generation") == generation
            and self.latest_run_resumed.payload.get("stage") == stage
            and self.latest_run_resumed.payload.get("resume_origin")
            == GATEWAY_RETRY_POLICIES[retry_class].circuit_code
        ):
            return int(self.latest_run_resumed.payload["breaker_epoch"])
        return self.gateway_retry_max_epoch.get(scope, 0) + 1

    def active_gateway_retry(
        self,
        scope: tuple[int, int, str, str],
        *,
        reset_seq: int,
    ) -> Event | None:
        latest = self.gateway_retry_last_by_scope.get(scope)
        return latest if latest is not None and latest.seq > reset_seq else None

    def apply(self, incoming: tuple[Event, ...]) -> None:
        if not incoming:
            return
        previous = self.events[-1].seq
        for item in incoming:
            if item.run_id != self.created.run_id or item.seq <= previous:
                raise ValueError("incremental event identity/order mismatch")
            previous = item.seq
        self.events_by_seq.update((item.seq, item) for item in incoming)
        self.events += incoming
        for event in incoming:
            payload = event.payload
            if event.kind == "RunSeedGenomeMaterialized":
                if self.expected_seed_canonical is None:
                    raise ValueError("historical run cannot materialize a DSH-native seed")
                if self.materialized_seed_canonical is not None:
                    raise ValueError("run has multiple seed materialization events")
                if not isinstance(payload, Mapping) or set(payload) != {
                    "schema_version",
                    "materializer_version",
                    "genome_canonical_json",
                    "genome_digest",
                }:
                    raise ValueError("RunSeedGenomeMaterialized payload is invalid")
                if payload["genome_canonical_json"] != self.expected_seed_canonical:
                    raise ValueError("materialized seed differs from RunCreated expectation")
                from ..evolution.genome import EcologyEvolutionPluginGenome

                import json

                seed = EcologyEvolutionPluginGenome.from_dict(
                    json.loads(self.expected_seed_canonical)
                )
                if payload["genome_digest"] != seed.genome_digest:
                    raise ValueError("materialized seed genome digest mismatch")
                self.materialized_seed_canonical = self.expected_seed_canonical
            elif event.kind == "DshRuntimeBound":
                if set(payload) != {
                    "schema_version",
                    "execution_protocol",
                    "capabilities_digest",
                    "preset_ids",
                    "first_call_verified",
                }:
                    raise ValueError("DshRuntimeBound payload is invalid")
                if (
                    payload["schema_version"] != "ecologyrsi-dsh.runtime-bound/1"
                    or payload["execution_protocol"] != DSH_NATIVE_EVOLUTION_PROTOCOL
                    or payload["first_call_verified"] is not False
                    or not isinstance(payload["preset_ids"], list)
                    or not isinstance(payload["capabilities_digest"], str)
                    or len(payload["capabilities_digest"]) != 64
                ):
                    raise ValueError("DshRuntimeBound contract is invalid")
            elif event.kind == AUDIT_EVENT:
                if self.run.status is not RunStatus.CREATED or self.model_contract_preflight_recorded:
                    raise ValueError("model preflight audit must be recorded once before start")
                validate_preflight_audit(
                    payload, self.task, self.run.run_id,
                    recorded_at=event.created_at, created_at=self.created.created_at,
                )
                self.model_contract_preflight_recorded = True
            elif event.kind == "RunStarted":
                if (preflight_audit_required(self.task.metadata)
                        and not self.model_contract_preflight_recorded):
                    raise ValueError("run started without model contract preflight audit")
                if self.active_gateway_circuit_pause is not None:
                    raise ValueError(
                        "gateway circuit requires an exact RunResumed origin"
                    )
                if self.expected_seed_canonical is not None and self.materialized_seed_canonical is None:
                    raise ValueError("DSH-native run started before seed initialization")
                if self.run.status not in _TERMINAL_RUN_STATUSES:
                    self.run = replace(
                        self.run, status=RunStatus.RUNNING, session_id=payload["session_id"]
                    )
            elif event.kind == "RunPaused":
                if payload.get("code") in GATEWAY_CIRCUIT_CODES:
                    _validate_gateway_circuit_pause_payload(payload)
                    if (
                        self.run.status is not RunStatus.RUNNING
                        or int(payload["generation"]) != int(self.run.generation)
                    ):
                        raise ValueError(
                            "gateway circuit pause requires a running current generation"
                        )
                    scope = self.gateway_retry_scope(payload)
                    reset_seq = self.gateway_retry_reset_seq(
                        scope,
                        dsh_continuity_reset=(
                            payload.get("continuity_reset_contract")
                            == _DSH_CONTINUITY_RESET_CONTRACT
                        ),
                    )
                    prior_retry = self.active_gateway_retry(scope, reset_seq=reset_seq)
                    first_retry = self.gateway_retry_first_by_scope.get(scope)
                    first_failure_event = (
                        first_retry
                        if first_retry is not None and first_retry.seq > reset_seq
                        else event
                    )
                    _validate_gateway_circuit_pause_trigger_evidence(
                        payload,
                        first_failure_event=first_failure_event,
                        pause_event=event,
                    )
                    if prior_retry is None:
                        chain_valid = (
                            payload["consecutive_failures"] == 1
                            and payload["breaker_epoch"]
                            == self.gateway_retry_expected_epoch(scope, reset_seq=reset_seq)
                            and payload["first_failure_at"]
                            == payload["last_failure_at"]
                        )
                    else:
                        chain_valid = (
                            payload["consecutive_failures"]
                            == prior_retry.payload["consecutive_failures"] + 1
                            and payload["breaker_epoch"]
                            == prior_retry.payload["breaker_epoch"]
                            and payload["retry_limit"]
                            == prior_retry.payload["retry_limit"]
                            and payload["first_failure_at"]
                            == prior_retry.payload["first_failure_at"]
                            and _aware_timestamp(payload["last_failure_at"])
                            >= _aware_timestamp(prior_retry.payload["last_failure_at"])
                        )
                    if not chain_valid:
                        raise ValueError(
                            "gateway circuit pause does not match the retry chain"
                        )
                    self.gateway_retry_max_epoch[scope] = max(
                        self.gateway_retry_max_epoch.get(scope, 0),
                        int(payload["breaker_epoch"]),
                    )
                    self.active_gateway_circuit_pause = event
                else:
                    if self.active_gateway_circuit_pause is not None:
                        raise ValueError(
                            "gateway circuit requires an exact RunResumed origin"
                        )
                if self.run.status not in _TERMINAL_RUN_STATUSES:
                    self.run = replace(self.run, status=RunStatus.PAUSED)
            elif event.kind == "RunResumed":
                if self.active_gateway_circuit_pause is not None:
                    if payload.get("resume_origin") not in GATEWAY_CIRCUIT_CODES:
                        raise ValueError(
                            "gateway circuit requires an exact RunResumed origin"
                        )
                    _validate_gateway_circuit_resume_payload(payload)
                    origin = self.active_gateway_circuit_pause
                    if (
                        self.run.status is not RunStatus.PAUSED
                        or origin is None
                        or payload["origin_pause_event_id"] != origin.event_id
                        or payload["origin_breaker_epoch"]
                        != origin.payload.get("breaker_epoch")
                        or payload["retry_class"] != origin.payload.get("retry_class")
                        or payload["generation"] != origin.payload.get("generation")
                        or payload["stage"] != origin.payload.get("stage")
                    ):
                        raise ValueError(
                            "gateway circuit resume does not match the active pause"
                        )
                elif payload.get("resume_origin") in GATEWAY_CIRCUIT_CODES:
                    _validate_gateway_circuit_resume_payload(payload)
                    raise ValueError(
                        "gateway circuit resume does not match the active pause"
                    )
                self.active_gateway_circuit_pause = None
                if self.run.status not in _TERMINAL_RUN_STATUSES:
                    self.run = replace(self.run, status=RunStatus.RUNNING)
                self.latest_run_resumed = event
            elif event.kind == "RunCancelled":
                if self.run.status not in _TERMINAL_RUN_STATUSES:
                    self.run = replace(self.run, status=RunStatus.CANCELLED)
            elif event.kind == "RunFailed":
                if self.run.status not in _TERMINAL_RUN_STATUSES:
                    self.run = replace(self.run, status=RunStatus.FAILED)
            elif event.kind == "RunCompleted":
                if self.run.status not in _TERMINAL_RUN_STATUSES:
                    self.run = replace(self.run, status=RunStatus.COMPLETED)
            elif event.kind == "GenerationAdvanced":
                if self.run.status not in _TERMINAL_RUN_STATUSES:
                    next_generation = payload.get("generation")
                    if uses_global_incumbent_protocol(self.task) and (
                        isinstance(next_generation, bool)
                        or not isinstance(next_generation, int)
                        or next_generation != self.run.generation + 1
                        or self.run.generation not in self.generation_comparisons
                        or self.run.generation not in self.effective_revision_bindings
                        or self.run.generation not in self.champion_generations
                    ):
                        raise ValueError(
                            "runtime-v2 generation advance requires the sequential "
                            "global incumbent decision"
                        )
                    self.run = replace(self.run, generation=int(payload["generation"]))
            elif event.kind == "ProposalSubmitted":
                if is_dsh_native_protocol(self.task) and self.formal_stage_started:
                    raise ValueError("formal stage results cannot feed a new proposal")
                item = Proposal.from_dict(payload["proposal"])
                if is_dsh_native_protocol(self.task):
                    genome = persisted_genome_from_proposal(item)
                    if genome is None:
                        raise ValueError("DSH-native proposal is missing its genome")
                    lineage = dict(genome.lineage)
                    seed_control = (
                        item.metadata.get("candidate_role")
                        == CandidateRole.INCUMBENT_CONTROL.value
                    )
                    if seed_control:
                        if self.materialized_seed_canonical is None:
                            raise ValueError("seed incumbent cannot precede seed materialization")
                        import json

                        from ..evolution.genome import EcologyEvolutionPluginGenome

                        seed = EcologyEvolutionPluginGenome.from_dict(
                            json.loads(self.materialized_seed_canonical)
                        )
                        if (
                            item.generation != 0
                            or item.parent_candidate_id is not None
                            or genome.genome_digest != seed.genome_digest
                            or lineage["origin_kind"] != "seed_catalog"
                            or lineage["generation"] is not None
                            or lineage["slot_index"] is not None
                        ):
                            raise ValueError(
                                "seed incumbent proposal does not match materialized seed"
                            )
                    elif (
                        lineage["generation"] != item.generation
                        or lineage["parent_candidate_id"] != item.parent_candidate_id
                    ):
                        raise ValueError("proposal scope does not match genome lineage")
                    if item.generation == 0 and not seed_control:
                        if self.materialized_seed_canonical is None:
                            raise ValueError("proposal cannot precede seed materialization")
                        import json

                        from ..evolution.genome import EcologyEvolutionPluginGenome

                        seed = EcologyEvolutionPluginGenome.from_dict(
                            json.loads(self.materialized_seed_canonical)
                        )
                        if lineage["parent_genome_digest"] != seed.genome_digest:
                            raise ValueError("first-generation proposal parent genome mismatch")
                    elif not seed_control and item.parent_candidate_id is None:
                        raise ValueError("later DSH-native proposal requires a parent candidate")
                self.proposals[item.proposal_id] = item
            elif event.kind == "CandidateSpawned":
                item = Candidate.from_dict(payload["candidate"])
                if item.run_id != self.run.run_id:
                    raise ValueError("candidate ownership does not match run")
                if item.candidate_id in self.candidates:
                    raise ValueError("candidate_id has multiple spawn events")
                if is_dsh_native_protocol(self.task):
                    proposal = self.proposals.get(item.proposal_id)
                    if proposal is None:
                        raise ValueError("candidate is missing its DSH-native proposal")
                    genome = persisted_genome_from_proposal(proposal)
                    if genome is None:
                        raise ValueError("candidate proposal is missing its genome")
                    lineage = dict(genome.lineage)
                    if item.role is CandidateRole.INCUMBENT_CONTROL:
                        if self.materialized_seed_canonical is None:
                            raise ValueError("incumbent control cannot precede seed")
                        import json

                        from ..evolution.genome import EcologyEvolutionPluginGenome

                        seed = EcologyEvolutionPluginGenome.from_dict(
                            json.loads(self.materialized_seed_canonical)
                        )
                        if (
                            item.proposal_id
                            != f"proposal:{self.run.run_id}:seed-incumbent-control"
                            or item.candidate_id
                            != f"candidate:{self.run.run_id}:seed-incumbent-control"
                            or item.slot_index != 0
                            or any(
                                candidate.role is CandidateRole.INCUMBENT_CONTROL
                                for candidate in self.candidates.values()
                            )
                            or
                            proposal.metadata.get("candidate_role") != item.role.value
                            or genome.genome_digest != seed.genome_digest
                            or lineage["origin_kind"] != "seed_catalog"
                            or lineage["generation"] is not None
                            or lineage["slot_index"] is not None
                        ):
                            raise ValueError(
                                "deterministic seed control does not match materialized seed"
                            )
                    elif (
                        item.generation != lineage["generation"]
                        or item.slot_index != lineage["slot_index"]
                    ):
                        raise ValueError("candidate coordinates do not match genome lineage")
                    binding = validate_identity_binding(payload.get("identity_binding"))
                    if binding["genome_digest"] != genome.genome_digest:
                        raise ValueError("candidate identity binding genome mismatch")
                    if binding["behavior_digest"] != genome.behavior_digest:
                        raise ValueError("candidate identity binding behavior mismatch")
                    self.candidate_identity_bindings[item.candidate_id] = {
                        "candidate_id": item.candidate_id,
                        "identity_binding": binding,
                    }
                self.candidates[item.candidate_id] = item
            elif event.kind == "RunAdaptationCohortFrozen":
                if set(payload) != {"adaptation"}:
                    raise ValueError("RunAdaptationCohortFrozen payload is invalid")
                adaptation = RunAdaptationCohort.from_dict(payload["adaptation"])
                schedule = OptimizationSchedule.from_dict(
                    self.task.metadata["optimization_schedule"]
                )
                if (
                    self.task.metadata.get("optimization_protocol") not in ADAPTIVE_PROTOCOLS
                    or adaptation.dataset_id != self.task.visible_datasets[0]
                    or adaptation.episode_id != str(self.task.metadata.get("episode_id"))
                    or adaptation.seed != self.task.seed
                    or adaptation.origin_count
                    != schedule.formal_origin_count_per_finalist
                    or len(adaptation.batches) != schedule.batch_count
                    or any(
                        batch.origin_count != schedule.local_batch_origin_count
                        for batch in adaptation.batches
                    )
                ):
                    raise ValueError("run adaptation cohort differs from frozen task")
                if (
                    self.run_adaptation_cohort is not None
                    and self.run_adaptation_cohort.to_dict() != adaptation.to_dict()
                ):
                    raise ValueError("conflicting run adaptation cohort")
                self.run_adaptation_cohort = adaptation
            elif event.kind == "GenerationCohortsFrozen":
                if set(payload) != {"generation_cohorts"}:
                    raise ValueError("GenerationCohortsFrozen payload is invalid")
                planned = GenerationCohorts.from_dict(payload["generation_cohorts"])
                schedule = OptimizationSchedule.from_dict(
                    self.task.metadata["optimization_schedule"]
                )
                generation_candidates = [
                    item for item in self.candidates.values()
                    if item.generation == planned.generation
                    and item.role is CandidateRole.SEARCH
                ]
                if self.run_adaptation_cohort is None:
                    raise ValueError("generation cohorts require run adaptation cohort")
                if len(generation_candidates) != 4:
                    raise ValueError("generation cohorts require exactly four candidates")
                if any(
                    generation == planned.generation
                    for generation, _candidate_id in self.candidate_screening_events
                ):
                    raise ValueError("generation cohorts must precede screening")
                if (
                    planned.dataset_id != self.run_adaptation_cohort.dataset_id
                    or planned.episode_id != self.run_adaptation_cohort.episode_id
                    or planned.seed != self.task.seed
                    or planned.adaptation_digest
                    != self.run_adaptation_cohort.adaptation_digest
                    or planned.adaptation_batch_digests
                    != self.run_adaptation_cohort.batch_digests
                    or planned.screening.origin_count
                    != schedule.screening_origin_count
                    or planned.screening.shared_candidate_count != 4
                    or planned.holdout.origin_count
                    != schedule.selection_holdout_origin_count
                    or planned.holdout.shared_arm_count != schedule.finalist_count + 1
                    or set(planned.screening.origin_occurrence_keys)
                    & set(self.run_adaptation_cohort.origin_occurrence_keys)
                    or set(planned.holdout.origin_occurrence_keys)
                    & set(self.run_adaptation_cohort.origin_occurrence_keys)
                ):
                    raise ValueError("generation cohorts differ from frozen task")
                prior_origins = {
                    origin_id
                    for item in self.generation_selection_cohorts.values()
                    for origin_id in (
                        *item.screening.origin_occurrence_keys,
                        *item.holdout.origin_occurrence_keys,
                    )
                }
                if prior_origins & set(
                    (
                        *planned.screening.origin_occurrence_keys,
                        *planned.holdout.origin_occurrence_keys,
                    )
                ):
                    raise ValueError("generation selection origins cannot be reused")
                existing = self.generation_selection_cohorts.get(planned.generation)
                if existing is not None and existing.to_dict() != planned.to_dict():
                    raise ValueError("conflicting generation cohorts")
                self.generation_selection_cohorts.setdefault(planned.generation, planned)
            elif event.kind == "CandidateScreeningRecorded":
                schema_version = payload.get("schema_version")
                expected_fields = {
                    "schema_version",
                    "generation",
                    "candidate_id",
                    "score",
                    "passed",
                    "constraint_violations",
                    "origin_count",
                    "prediction_cell_count",
                    "cohort_digest",
                }
                if schema_version == SCREENING_SCHEMA_V2:
                    expected_fields.add("record_digest")
                elif schema_version != SCREENING_SCHEMA_V1:
                    raise ValueError("unsupported candidate screening schema")
                if set(payload) != expected_fields:
                    raise ValueError("candidate screening payload is invalid")
                generation = payload["generation"]
                candidate_id = payload["candidate_id"]
                if (
                    isinstance(generation, bool)
                    or not isinstance(generation, int)
                    or generation < 0
                    or not isinstance(candidate_id, str)
                    or not candidate_id.strip()
                ):
                    raise ValueError("candidate screening identity is invalid")
                candidate = self.candidates.get(candidate_id)
                if candidate is None:
                    raise ValueError("screening candidate is missing")
                if candidate.generation != generation:
                    raise ValueError("screening generation does not match candidate")
                if candidate.role is not CandidateRole.SEARCH:
                    raise ValueError("incumbent control cannot enter candidate screening")
                if (
                    self.task.metadata.get("optimization_protocol")
                    in ADAPTIVE_PROTOCOLS
                    and not any(
                        revision.candidate_id == candidate_id
                        and revision.parent_revision_id is None
                        for revision in self.candidate_revisions.values()
                    )
                ):
                    raise ValueError(
                        "adaptive candidate screening requires frozen initial revision R0"
                    )
                if self.task.metadata.get("optimization_protocol") in ADAPTIVE_PROTOCOLS:
                    planned = self.generation_selection_cohorts.get(generation)
                    if planned is None:
                        raise ValueError(
                            "adaptive candidate screening requires frozen generation cohorts"
                        )
                    if payload.get("cohort_digest") != planned.screening.cohort_digest:
                        raise ValueError(
                            "candidate screening differs from frozen screening cohort"
                        )
                score = payload["score"]
                constraint_violations = payload["constraint_violations"]
                origin_count = payload["origin_count"]
                prediction_cell_count = payload["prediction_cell_count"]
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                    or not isinstance(payload["passed"], bool)
                    or isinstance(constraint_violations, bool)
                    or not isinstance(constraint_violations, int)
                    or constraint_violations < 0
                    or isinstance(origin_count, bool)
                    or not isinstance(origin_count, int)
                    or origin_count < 1
                    or isinstance(prediction_cell_count, bool)
                    or not isinstance(prediction_cell_count, int)
                    or prediction_cell_count < origin_count
                ):
                    raise ValueError("candidate screening evidence is invalid")
                cohort_digest = payload["cohort_digest"]
                strict_selection = bool(
                    supports_two_stage_screening(
                        self.task.metadata.get("sample_agent_protocol")
                    )
                    and self.task.metadata.get("sample_budget_class") == "selection_eligible"
                    and self.task.metadata.get("two_stage_evaluation_enabled", True) is True
                )
                if (
                    schema_version == SCREENING_SCHEMA_V2 or cohort_digest is not None
                ) and (
                    not isinstance(cohort_digest, str)
                    or len(cohort_digest) != 64
                    or any(character not in "0123456789abcdef" for character in cohort_digest)
                ):
                    raise ValueError("screening cohort_digest must be a SHA-256 digest")
                if strict_selection:
                    cells_per_origin = self.task.metadata.get("prediction_cells_per_origin")
                    if (
                        cohort_digest is None
                        or isinstance(cells_per_origin, bool)
                        or not isinstance(cells_per_origin, int)
                        or cells_per_origin < 1
                        or origin_count != 64
                        or complete_origin_count(
                            prediction_cell_count, cells_per_origin
                        )
                        != 64
                    ):
                        raise ValueError("strict-v4 screening evidence is incomplete")
                if schema_version == SCREENING_SCHEMA_V2:
                    record_digest = payload["record_digest"]
                    if (
                        not isinstance(record_digest, str)
                        or len(record_digest) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in record_digest
                        )
                        or record_digest != screening_record_digest(payload)
                    ):
                        raise ValueError("candidate screening record_digest is invalid")
                key = (generation, candidate_id)
                existing = self.candidate_screening_events.get(key)
                if existing is not None:
                    if canonical_json(existing.payload) != canonical_json(payload):
                        raise ValueError("conflicting screening record")
                else:
                    if candidate.status is not CandidateStatus.SPAWNED:
                        raise ValueError("only a new candidate can be screened")
                    self.candidate_screening_events[key] = event
            elif event.kind == "FormalSelectionCohortFrozen":
                if payload.get("schema_version") not in {
                    FORMAL_SELECTION_SCHEMA_V1,
                    FORMAL_SELECTION_SCHEMA_V2,
                    FORMAL_SELECTION_SCHEMA_V3,
                }:
                    raise ValueError("formal selection payload is invalid")
                expected_formal_fields = {
                    "schema_version",
                    "generation",
                    "selected_candidate_ids",
                    "screening_digest",
                }
                if payload.get("schema_version") == FORMAL_SELECTION_SCHEMA_V3:
                    expected_formal_fields.update(
                        {
                            "screening_pass_count",
                            "exploration_only",
                            "consecutive_exploration_generations",
                        }
                    )
                if set(payload) != expected_formal_fields:
                    raise ValueError("formal selection payload is invalid")
                generation = payload["generation"]
                selected = payload["selected_candidate_ids"]
                schedule = OptimizationSchedule.from_dict(self.task.metadata.get("optimization_schedule", OptimizationSchedule.default().to_dict()))
                if schedule.quick:
                    first = min((c for c in self.candidates.values() if c.generation == generation and c.role is CandidateRole.SEARCH), key=lambda c: c.slot_index)
                    if selected != [first.candidate_id]:
                        raise ValueError("quick trajectory must preregister the first proposal")
                if (
                    isinstance(generation, bool)
                    or not isinstance(generation, int)
                    or generation < 0
                    or not isinstance(selected, list)
                    or len(selected) != schedule.finalist_count
                    or any(not isinstance(item, str) or not item.strip() for item in selected)
                    or len(set(selected)) != schedule.finalist_count
                ):
                    raise ValueError("formal selected candidate count is invalid")
                for candidate_id in selected:
                    candidate = self.candidates.get(candidate_id)
                    if (
                        candidate is None
                        or candidate.generation != generation
                        or candidate.role is not CandidateRole.SEARCH
                    ):
                        raise ValueError("formal selected candidate is outside generation")
                    if not schedule.quick and (generation, candidate_id) not in self.candidate_screening_events:
                        raise ValueError("formal selection is missing screening evidence")
                screening_digest = payload["screening_digest"]
                if (
                    not isinstance(screening_digest, str)
                    or len(screening_digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in screening_digest
                    )
                ):
                    raise ValueError("formal screening_digest must be a SHA-256 digest")
                generation_records = [
                    screening_event.payload
                    for (item_generation, _candidate_id), screening_event in self.candidate_screening_events.items()
                    if item_generation == generation
                ]
                if screening_digest != screening_cohort_digest(generation_records):
                    raise ValueError("formal screening digest does not match screening cohort")
                if payload.get("schema_version") == FORMAL_SELECTION_SCHEMA_V3:
                    pass_count = sum(
                        record.get("passed") is True for record in generation_records
                    )
                    exploration_only = pass_count == 0
                    consecutive = payload["consecutive_exploration_generations"]
                    expected_consecutive = 0
                    if exploration_only:
                        expected_consecutive = 1
                        expected_generation = generation - 1
                        while expected_generation >= 0:
                            prior = self.formal_selection_events.get(expected_generation)
                            if (
                                prior is None
                                or prior.payload.get("schema_version")
                                != FORMAL_SELECTION_SCHEMA_V3
                                or prior.payload.get("exploration_only") is not True
                            ):
                                break
                            expected_consecutive += 1
                            expected_generation -= 1
                    supplied_pass_count = payload["screening_pass_count"]
                    if (
                        isinstance(consecutive, bool)
                        or not isinstance(consecutive, int)
                        or consecutive < 0
                        or isinstance(supplied_pass_count, bool)
                        or not isinstance(supplied_pass_count, int)
                        or supplied_pass_count != pass_count
                        or payload["exploration_only"] is not exploration_only
                        or consecutive != expected_consecutive
                    ):
                        raise ValueError("formal exploration state is invalid")
                existing = self.formal_selection_events.get(generation)
                if existing is not None:
                    if canonical_json(existing.payload) != canonical_json(payload):
                        raise ValueError("conflicting formal selection")
                else:
                    self.formal_selection_events[generation] = event
            elif event.kind == "ArtifactRecorded":
                item = ModelArtifact.from_dict(payload["artifact"])
                if is_dsh_native_protocol(self.task):
                    expected_binding = self.candidate_identity_bindings.get(item.candidate_id)
                    if expected_binding is None:
                        raise ValueError("artifact candidate has no identity binding")
                    if payload.get("schema_version") == ARTIFACT_EVENT_V2:
                        if set(payload) != {"schema_version", "artifact", "proposal_identity_binding", "artifact_revision_binding"}:
                            raise ValueError("artifact v2 envelope fields are invalid")
                        validate_identity_binding(payload.get("proposal_identity_binding"), expected=expected_binding["identity_binding"])
                        revision = self.candidate_revisions.get(item.candidate_revision_id)
                        if revision is None:
                            raise ValueError("artifact actual revision is missing")
                        scope = resolve_artifact_scope(item, holdouts=self.generation_holdouts.values(),
                                                       evaluations=(*self.formal_batch_evaluations.values(), *self.holdout_evaluations.values()))
                        self.artifact_revision_bindings[item.artifact_id] = validate_artifact_revision_binding(
                            payload.get("artifact_revision_binding"), artifact=item, revision=revision, scope=scope)
                    else:
                        if payload.get("schema_version") is not None or item.artifact_id in self.artifact_revision_bindings:
                            raise ValueError("artifact envelope version is invalid")
                        validate_identity_binding(payload.get("identity_binding"), expected=expected_binding["identity_binding"])
                    if payload["artifact"].get("artifact_digest") != item.digest:
                        raise ValueError("artifact digest mismatch")
                existing = self.artifacts.get(item.artifact_id)
                if existing is not None and existing.to_dict() != item.to_dict():
                    raise ValueError("artifact_id belongs to multiple candidates")
                self.artifacts[item.artifact_id] = item
            elif event.kind == "EvaluationRecorded":
                item = Evaluation.from_dict(payload["evaluation"])
                if is_dsh_native_protocol(self.task):
                    expected_binding = self.candidate_identity_bindings.get(item.candidate_id)
                    if expected_binding is None:
                        raise ValueError("evaluation candidate has no identity binding")
                    artifact = next(
                        (
                            artifact
                            for artifact in self.artifacts.values()
                            if artifact.candidate_id == item.candidate_id
                        ),
                        None,
                    )
                    if artifact is None or item.artifact_digest != artifact.digest:
                        raise ValueError("evaluation artifact binding mismatch")
                    actual_binding = self.artifact_revision_bindings.get(artifact.artifact_id)
                    if payload.get("schema_version") == EVALUATION_EVENT_V2:
                        if set(payload) != {"schema_version", "evaluation", "proposal_identity_binding", "artifact_revision_binding", "artifact_digest", "evaluation_digest"}:
                            raise ValueError("evaluation v2 envelope fields are invalid")
                        validate_identity_binding(payload.get("proposal_identity_binding"), expected=expected_binding["identity_binding"])
                        if actual_binding is None or canonical_json(payload.get("artifact_revision_binding")) != canonical_json(actual_binding):
                            raise ValueError("evaluation artifact revision binding mismatch")
                        validate_evaluation_artifact_binding(item, artifact, actual_binding)
                    else:
                        if payload.get("schema_version") is not None or actual_binding is not None:
                            raise ValueError("evaluation envelope does not match artifact version")
                        validate_identity_binding(payload.get("identity_binding"), expected=expected_binding["identity_binding"])
                    if payload.get("artifact_digest") != artifact.digest:
                        raise ValueError("evaluation event artifact digest mismatch")
                    if payload.get("evaluation_digest") != digest(item.to_dict()):
                        raise ValueError("evaluation event digest mismatch")
                existing = self.evaluations.get(item.evaluation_id)
                if existing is not None and existing.to_dict() != item.to_dict():
                    raise ValueError("evaluation_id belongs to multiple candidates")
                self.evaluations[item.evaluation_id] = item
                candidate = self.candidates.get(item.candidate_id)
                if candidate is not None:
                    self.candidates[item.candidate_id] = replace(
                        candidate,
                        status=CandidateStatus.EVALUATED,
                        evaluation_id=item.evaluation_id,
                    )
            elif event.kind == "EvaluationJudged":
                item = Evaluation.from_dict(payload["evaluation"])
                if is_dsh_native_protocol(self.task):
                    expected_binding = self.candidate_identity_bindings.get(item.candidate_id)
                    if expected_binding is None:
                        raise ValueError("judgment candidate has no identity binding")
                    validate_identity_binding(
                        payload.get("identity_binding"),
                        expected=expected_binding["identity_binding"],
                    )
                    artifact = next(
                        (
                            artifact
                            for artifact in self.artifacts.values()
                            if artifact.candidate_id == item.candidate_id
                        ),
                        None,
                    )
                    if artifact is None or item.artifact_digest != artifact.digest:
                        raise ValueError("judgment artifact binding mismatch")
                    actual_binding = self.artifact_revision_bindings.get(artifact.artifact_id)
                    if actual_binding is not None:
                        validate_evaluation_artifact_binding(item, artifact, actual_binding)
                        scientific = self.evaluations.get(item.evaluation_id)
                        immutable_fields = (
                            "evaluation_id", "run_id", "candidate_id", "candidate_revision_id",
                            "evaluation_scope", "score", "partition", "evaluator_digest",
                            "artifact_digest", "created_at",
                        )
                        if scientific is None or any(
                            getattr(scientific, name) != getattr(item, name)
                            for name in immutable_fields
                        ):
                            raise ValueError("judgment cannot change scientific evaluation fields")
                    if payload.get("artifact_digest") != artifact.digest:
                        raise ValueError("judgment event artifact digest mismatch")
                    if payload.get("evaluation_digest") != digest(item.to_dict()):
                        raise ValueError("judgment event evaluation digest mismatch")
                existing = self.evaluations.get(item.evaluation_id)
                if existing is None:
                    raise ValueError("judged evaluation is missing its scientific evaluation")
                if existing.candidate_id != item.candidate_id:
                    raise ValueError("judged evaluation belongs to another candidate")
                self.evaluations[item.evaluation_id] = item
            elif event.kind == "PromotionDecided":
                item = Promotion.from_dict(payload["promotion"])
                if is_dsh_native_protocol(self.task):
                    expected_binding = self.candidate_identity_bindings.get(item.candidate_id)
                    if expected_binding is None:
                        raise ValueError("promotion candidate has no identity binding")
                    validate_identity_binding(
                        payload.get("identity_binding"),
                        expected=expected_binding["identity_binding"],
                    )
                    evaluation = next(
                        (
                            evaluation
                            for evaluation in self.evaluations.values()
                            if evaluation.candidate_id == item.candidate_id
                        ),
                        None,
                    )
                    artifact = next(
                        (
                            artifact
                            for artifact in self.artifacts.values()
                            if artifact.candidate_id == item.candidate_id
                        ),
                        None,
                    )
                    if evaluation is None or artifact is None:
                        raise ValueError("promotion is missing evaluation or artifact binding")
                    if payload.get("evaluation_id") != evaluation.evaluation_id:
                        raise ValueError("promotion evaluation binding mismatch")
                    if payload.get("evaluation_digest") != digest(evaluation.to_dict()):
                        raise ValueError("promotion evaluation digest mismatch")
                    if payload.get("artifact_digest") != artifact.digest:
                        raise ValueError("promotion artifact digest mismatch")
                    candidate = self.candidates.get(item.candidate_id)
                    if candidate is None:
                        raise ValueError("promotion candidate is missing")
                    validate_runtime_promotion_decision(
                        self.task,
                        candidate,
                        item,
                        self.generation_comparisons.get(candidate.generation),
                        self.generation_analyses.get(candidate.generation),
                    )
                existing = self.promotions.get(item.promotion_id)
                if existing is not None and existing.to_dict() != item.to_dict():
                    raise ValueError("promotion_id belongs to multiple candidates")
                self.promotions[item.promotion_id] = item
                candidate = self.candidates.get(item.candidate_id)
                if candidate is not None:
                    status = (
                        CandidateStatus.PROMOTED
                        if item.decision is PromotionDecision.APPROVED
                        else CandidateStatus.REJECTED
                    )
                    self.candidates[item.candidate_id] = replace(
                        candidate, status=status, promotion_id=item.promotion_id
                    )
            elif event.kind == "CandidateScreenedOut":
                legacy_fields = {
                    "candidate_id",
                    "generation",
                    "formal_selection_event_id",
                    "reason",
                }
                if set(payload) == legacy_fields:
                    pass
                elif (
                    payload.get("schema_version") != SCREENED_OUT_SCHEMA_V1
                    or set(payload) != legacy_fields | {"schema_version"}
                ):
                    raise ValueError("candidate screened-out payload is invalid")
                candidate_id = payload["candidate_id"]
                generation = payload["generation"]
                candidate = self.candidates.get(candidate_id) if isinstance(candidate_id, str) else None
                if (
                    candidate is None
                    or isinstance(generation, bool)
                    or not isinstance(generation, int)
                    or candidate.generation != generation
                    or candidate.status
                    not in {CandidateStatus.SPAWNED, CandidateStatus.SCREENED_OUT}
                ):
                    raise ValueError("screened-out candidate generation is invalid")
                formal = self.formal_selection_events.get(generation)
                if formal is None or payload["formal_selection_event_id"] != formal.event_id:
                    raise ValueError("screened-out candidate formal selection is missing")
                if candidate.candidate_id in formal.payload["selected_candidate_ids"]:
                    raise ValueError("selected candidate cannot be screened out")
                quick = OptimizationSchedule.from_dict(self.task.metadata.get("optimization_schedule", OptimizationSchedule.default().to_dict())).quick
                if not quick and (generation, candidate.candidate_id) not in self.candidate_screening_events:
                    raise ValueError("screened-out candidate is missing screening evidence")
                if payload["reason"] != ("outside_preregistered_quick_trajectory" if quick else "not_selected_by_screening_top_k"):
                    raise ValueError("screened-out candidate reason is invalid")
                key = (generation, candidate.candidate_id)
                existing = self.screened_out_events.get(key)
                if existing is not None:
                    if canonical_json(existing.payload) != canonical_json(payload):
                        raise ValueError("conflicting screened-out candidate")
                else:
                    self.screened_out_events[key] = event
                self.candidates[candidate.candidate_id] = replace(
                    candidate, status=CandidateStatus.SCREENED_OUT
                )
            elif event.kind in {
                "CandidateFailed",
                "CandidateMarkedDuplicate",
            }:
                candidate_id = str(payload["candidate_id"])
                candidate = self.candidates.get(candidate_id)
                if candidate is not None:
                    status = {
                        "CandidateFailed": CandidateStatus.FAILED,
                        "CandidateMarkedDuplicate": CandidateStatus.DUPLICATE,
                    }[event.kind]
                    self.candidates[candidate_id] = replace(candidate, status=status)
            elif event.kind == "GenerationSearchPlanned":
                item = GenerationSearchPlan.from_dict(payload["search_plan"])
                if item.run_id != self.run.run_id:
                    raise ValueError("generation search plan belongs to another run")
                previous = self.generation_analyses.get(item.generation - 1)
                previous_reflection = self.generation_reflections.get(item.generation - 1)
                if item.source_analysis_digest != (
                    previous.analysis_digest if previous is not None else None
                ):
                    raise ValueError("generation search plan previous analysis mismatch")
                if item.source_reflection_digest != (
                    previous_reflection.reflection_digest
                    if previous_reflection is not None
                    else None
                ):
                    raise ValueError("generation search plan previous reflection mismatch")
                if previous is not None and previous.replan_required:
                    prior_plan = self.generation_search_plans.get(item.generation - 1)
                    if prior_plan is not None and (
                        item.search_queries == prior_plan.search_queries
                        and item.focus_areas == prior_plan.focus_areas
                    ):
                        raise ValueError(
                            "required search replan must change queries or focus areas"
                        )
                existing = self.generation_search_plans.get(item.generation)
                if existing is not None and existing.to_dict() != item.to_dict():
                    raise ValueError("generation has multiple search plans")
                self.generation_search_plans[item.generation] = item
            elif event.kind == "GenerationBatchStarted":
                if self.expected_seed_canonical is not None and self.materialized_seed_canonical is None:
                    raise ValueError("generation batch cannot precede seed materialization")
                item = GenerationBatch.from_dict(payload["batch"])
                if (
                    item.run_id != self.run.run_id
                    or item.generation != self.run.generation
                    or item.task_manifest_digest != self.task.digest
                ):
                    raise ValueError("generation batch scope differs from the active run")
                frozen_interventions = []
                expected_pending_intervention_ids = tuple(
                    intervention.intervention_id
                    for intervention in self.interventions.values()
                    if intervention.applied_proposal_id is None
                )
                if item.intervention_ids != expected_pending_intervention_ids:
                    raise ValueError(
                        "generation batch intervention set differs from pending guidance"
                    )
                for intervention_id in item.intervention_ids:
                    intervention = self.interventions.get(intervention_id)
                    if (
                        intervention is None
                        or intervention.run_id != self.run.run_id
                        or intervention.applied_proposal_id is not None
                    ):
                        raise ValueError(
                            "generation batch intervention set is not pending and frozen"
                        )
                    frozen_interventions.append(intervention)
                if is_dsh_native_protocol(self.task):
                    if item.parent_genome_canonical_json is None:
                        raise ValueError("DSH-native batch is missing its parent genome")
                    if item.generation == 0:
                        if item.parent_candidate_id is not None:
                            raise ValueError("first generation cannot bind a parent candidate")
                        if item.parent_genome_canonical_json != self.materialized_seed_canonical:
                            raise ValueError("first generation parent differs from materialized seed")
                    else:
                        if item.parent_candidate_id is None:
                            raise ValueError("later generation batch requires a parent candidate")
                        parent_candidate = self.candidates.get(item.parent_candidate_id)
                        if parent_candidate is None:
                            raise ValueError("generation parent candidate is missing")
                        parent_overrides = [
                            intervention
                            for intervention in frozen_interventions
                            if intervention.kind is InterventionKind.PARENT_SELECTION
                        ]
                        parent_override = (
                            parent_overrides[-1] if parent_overrides else None
                        )
                        if (
                            parent_override is not None
                            and parent_override.target_candidate_id
                            != item.parent_candidate_id
                        ):
                            raise ValueError(
                                "generation parent differs from the frozen parent-selection "
                                "intervention"
                            )
                        if (
                            uses_global_incumbent_protocol(self.task)
                            and parent_override is None
                        ):
                            prior_binding = self.effective_revision_bindings.get(
                                item.generation - 1
                            )
                            prior_revision = (
                                self.candidate_revisions.get(
                                    str(prior_binding.get("selected_revision_id"))
                                )
                                if isinstance(prior_binding, Mapping)
                                else None
                            )
                            if (
                                prior_revision is None
                                or prior_binding.get("selected_candidate_id")
                                != item.parent_candidate_id
                                or prior_revision.candidate_id
                                != item.parent_candidate_id
                                or prior_revision.genome_digest
                                != item.parent_genome_digest
                                or canonical_json(prior_revision.identity_dict()["genome"])
                                != item.parent_genome_canonical_json
                            ):
                                raise ValueError(
                                    "runtime generation parent differs from the prior "
                                    "effective revision"
                                )
                        else:
                            parent_proposal = self.proposals.get(parent_candidate.proposal_id)
                            if parent_proposal is None:
                                raise ValueError("generation parent proposal is missing")
                            parent_genome = persisted_genome_from_proposal(parent_proposal)
                            if (
                                parent_genome is None
                                or canonical_json(parent_genome.to_dict())
                                != item.parent_genome_canonical_json
                            ):
                                raise ValueError("generation parent genome binding mismatch")
                existing_batch = self.generation_batches.get(item.generation)
                if existing_batch is not None and existing_batch.to_dict() != item.to_dict():
                    raise ValueError("generation has multiple conflicting batches")
                self.generation_batches.setdefault(item.generation, item)
            elif event.kind == "GenerationAnalyzed":
                item = GenerationAnalysis.from_dict(payload["analysis"])
                if item.run_id != self.run.run_id:
                    raise ValueError("generation analysis belongs to another run")
                formal = self.formal_selection_events.get(item.generation)
                formal_payload = getattr(formal, "payload", {})
                if (
                    isinstance(formal_payload, Mapping)
                    and formal_payload.get("schema_version")
                    == FORMAL_SELECTION_SCHEMA_V3
                ):
                    exploration_only = formal_payload.get("exploration_only") is True
                    consecutive = int(
                        formal_payload.get("consecutive_exploration_generations") or 0
                    )
                    if (
                        item.consecutive_exploration_generations
                        != (consecutive if exploration_only else 0)
                        or item.replan_required
                        is not (exploration_only and consecutive >= 2)
                    ):
                        raise ValueError(
                            "generation analysis exploration replan state is invalid"
                        )
                existing = self.generation_analyses.get(item.generation)
                if existing is not None and existing.to_dict() != item.to_dict():
                    raise ValueError("generation has multiple analyses")
                self.generation_analyses[item.generation] = item
            elif event.kind == "GenerationReflected":
                reflection_payload = payload["reflection"]
                if "canonical_candidate_outcomes" in reflection_payload:
                    item = GenerationReflection.from_dict(reflection_payload)
                else:
                    item = GenerationReflection.from_legacy_dict(reflection_payload)
                if item.run_id != self.run.run_id:
                    raise ValueError("generation reflection belongs to another run")
                analysis = self.generation_analyses.get(item.generation)
                if analysis is None or analysis.analysis_digest != item.analysis_digest:
                    raise ValueError("generation reflection analysis mismatch")
                existing = self.generation_reflections.get(item.generation)
                if existing is not None and existing.to_dict() != item.to_dict():
                    raise ValueError("generation has multiple reflections")
                self.generation_reflections[item.generation] = item
            elif event.kind == "GenerationKnowledgeRetrieved":
                item = KnowledgeSnapshot.from_dict(payload["knowledge_snapshot"])
                self.knowledge_snapshots[item.generation] = item
            elif event.kind == "GenerationKnowledgeAssessed":
                item = KnowledgeAssessment.from_dict(payload["knowledge_assessment"])
                self.knowledge_assessments[item.generation] = item
            elif event.kind == "GenerationResearchIterated":
                item = ResearchIteration.from_dict(payload["research_iteration"])
                if item.run_id != self.run.run_id:
                    raise ValueError("research iteration belongs to another run")
                if item.diagnostic_report is not None:
                    knowledge = self.knowledge_snapshots.get(item.generation)
                    if knowledge is None or knowledge.snapshot_digest != item.knowledge_snapshot_digest:
                        raise ValueError("research diagnosis knowledge is not frozen")
                    expected = diagnose_evidence(
                        self.task, self.generation_analyses.get(item.generation - 1),
                        self.generation_comparisons.get(item.generation - 1), knowledge,
                    )
                    if DiagnosticReport(**dict(item.diagnostic_report)) != expected:
                        raise ValueError("research diagnosis does not match visible host evidence")
                existing = self.research_iterations.get(item.generation)
                if existing is not None and existing.to_dict() != item.to_dict():
                    raise ValueError("generation has multiple research iterations")
                self.research_iterations[item.generation] = item
            elif event.kind == "AlgorithmAttemptRecorded":
                item = AlgorithmAttempt.from_dict(payload["algorithm_attempt"])
                if item.run_id != self.run.run_id:
                    raise ValueError("algorithm attempt belongs to another run")
                candidate = self.candidates.get(item.candidate_id)
                if candidate is None:
                    raise ValueError("algorithm attempt is missing its candidate")
                if (
                    candidate.proposal_id != item.proposal_id
                    or candidate.generation != item.generation
                ):
                    raise ValueError("algorithm attempt scope does not match its candidate")
                if is_dsh_native_protocol(self.task):
                    expected_binding = self.candidate_identity_bindings.get(item.candidate_id)
                    if expected_binding is None:
                        raise ValueError("algorithm attempt candidate has no identity binding")
                    validate_identity_binding(
                        payload.get("identity_binding"),
                        expected=expected_binding["identity_binding"],
                    )
                duplicate = next(
                    (
                        existing
                        for existing in self.algorithm_attempts
                        if existing.candidate_id == item.candidate_id
                        and existing.phase == item.phase
                        and existing.attempt == item.attempt
                    ),
                    None,
                )
                if duplicate is not None and duplicate.to_dict() != item.to_dict():
                    raise ValueError("algorithm attempt identity was reused")
                if duplicate is None:
                    self.algorithm_attempts.append(item)
            elif event.kind == "FormalStageFrozen":
                required = {
                    "stage",
                    "candidate_id",
                    "artifact_digest",
                    "genome_digest",
                    "analysis_plan_digest",
                    "objective_family_digest",
                    "partition_digest",
                    "holdout_exposure_key",
                    "token_digest",
                }
                revision_aware = payload.get("schema_version") == FORMAL_STAGE_V2
                if revision_aware:
                    required |= {"schema_version", "candidate_revision_id", "artifact_revision_binding"}
                if not is_dsh_native_protocol(self.task) or set(payload) != required:
                    raise ValueError("formal stage frozen payload is invalid")
                stage = str(payload["stage"])
                if stage not in {"validation", "final_test"}:
                    raise ValueError("formal stage is invalid")
                if stage in self.formal_stage_seals:
                    raise ValueError("formal stage is already sealed")
                candidate = self.candidates.get(str(payload["candidate_id"]))
                if candidate is None:
                    raise ValueError("formal stage candidate is missing")
                artifact = next(
                    (
                        item
                        for item in self.artifacts.values()
                        if item.candidate_id == candidate.candidate_id
                    ),
                    None,
                )
                if artifact is None or artifact.digest != payload["artifact_digest"]:
                    raise ValueError("formal stage artifact binding mismatch")
                if revision_aware:
                    revision = self.candidate_revisions.get(artifact.candidate_revision_id)
                    if revision is None or payload["candidate_revision_id"] != revision.revision_id:
                        raise ValueError("formal stage actual artifact revision is missing")
                    scope = resolve_artifact_scope(artifact, holdouts=self.generation_holdouts.values(),
                                                   evaluations=(*self.formal_batch_evaluations.values(), *self.holdout_evaluations.values()))
                    binding = validate_artifact_revision_binding(payload["artifact_revision_binding"], artifact=artifact, revision=revision, scope=scope)
                    if binding["genome_digest"] != payload["genome_digest"]:
                        raise ValueError("formal stage actual genome binding mismatch")
                else:
                    if artifact.artifact_id in self.artifact_revision_bindings:
                        raise ValueError("formal stage envelope cannot downgrade a v2 artifact")
                    binding = self.candidate_identity_bindings.get(candidate.candidate_id)
                    if binding is None or binding["identity_binding"]["genome_digest"] != payload["genome_digest"]:
                        raise ValueError("formal stage genome binding mismatch")
                if stage == "final_test" and self.run.validated_candidate_id != candidate.candidate_id:
                    raise ValueError("final-test requires the validated candidate")
                self.formal_stage_started = True
            elif event.kind == "FormalStageCompleted":
                if not self.formal_stage_started:
                    raise ValueError("formal stage completion has no frozen stage")
                stage = str(payload.get("stage") or "")
                candidate_id = str(payload.get("candidate_id") or "")
                outcome = str(payload.get("outcome") or "")
                if stage not in {"validation", "final_test"} or outcome not in {
                    "passed",
                    "failed",
                    "inconclusive",
                }:
                    raise ValueError("formal stage completion is invalid")
                if candidate_id not in self.candidates:
                    raise ValueError("formal stage completion candidate is missing")
                if outcome == "passed":
                    self.run = replace(
                        self.run,
                        **(
                            {"validated_candidate_id": candidate_id}
                            if stage == "validation"
                            else {"final_test_candidate_id": candidate_id}
                        ),
                    )
            elif event.kind == "FormalStageSealed":
                stage = str(payload.get("stage") or "")
                if stage not in {"validation", "final_test"}:
                    raise ValueError("formal stage seal is invalid")
                if stage in self.formal_stage_seals:
                    raise ValueError("formal stage has multiple seals")
                self.formal_stage_seals[stage] = dict(payload)
            # Legacy generations used the same event name for an aggregate
            # analysis payload.  Adaptive generations use the new three-field
            # revision binding below; only the legacy shape is ignored here so the
            # authoritative binding is still validated and replayed.
            elif event.kind == "GenerationChampionSelected" and set(payload) != {
                "generation",
                "selected_candidate_id",
                "selected_revision_id",
            }:
                continue
            elif event.kind == "HumanInterventionRecorded":
                item = HumanIntervention.from_dict(payload["intervention"])
                existing = self.interventions.get(item.intervention_id)
                if existing is not None and existing.to_dict() != item.to_dict():
                    raise ValueError("intervention_id belongs to multiple interventions")
                self.interventions[item.intervention_id] = item
            elif event.kind == "HumanInterventionApplied":
                intervention_id = str(payload["intervention_id"])
                item = self.interventions.get(intervention_id)
                if item is None:
                    raise ValueError("applied intervention is missing its recorded event")
                self.interventions[intervention_id] = replace(
                    item, applied_proposal_id=str(payload["proposal_id"])
                )
            elif event.kind == "ExpertConsultationRequested":
                item = ExpertConsultation.from_dict(payload["consultation"])
                if item.run_id != self.run.run_id:
                    raise ValueError("expert consultation belongs to another run")
                existing = self.expert_consultations.get(item.consultation_id)
                if existing is not None and existing.to_dict() != item.to_dict():
                    raise ValueError("consultation_id belongs to multiple consultations")
                self.expert_consultations[item.consultation_id] = item
            elif event.kind == "ExpertConsultationAnswered":
                item = ExpertConsultationAnswer.from_dict(payload["answer"])
                if item.run_id != self.run.run_id:
                    raise ValueError("expert consultation answer belongs to another run")
                consultation = self.expert_consultations.get(item.consultation_id)
                if consultation is None:
                    raise ValueError("expert answer is missing its consultation request")
                existing = self.expert_consultation_answers.get(item.consultation_id)
                if existing is not None and existing.to_dict() != item.to_dict():
                    raise ValueError("expert consultation already has a different answer")
                self.expert_consultation_answers[item.consultation_id] = item
            elif event.kind == "ExpertConsultationApplied":
                consultation_id = str(payload.get("consultation_id") or "")
                answer_id = str(payload.get("answer_id") or "")
                generation = payload.get("generation")
                iteration_digest = str(payload.get("research_iteration_digest") or "")
                if (
                    not consultation_id
                    or not answer_id
                    or isinstance(generation, bool)
                    or not isinstance(generation, int)
                    or generation < 0
                    or not iteration_digest
                ):
                    raise ValueError("expert consultation application payload is invalid")
                if consultation_id not in self.expert_consultations:
                    raise ValueError("applied expert answer is missing its consultation")
                answer = self.expert_consultation_answers.get(consultation_id)
                if answer is None or answer.answer_id != answer_id:
                    raise ValueError("applied expert answer is missing its answer event")
                iteration = self.research_iterations.get(generation)
                if (
                    iteration is None
                    or iteration.iteration_digest != iteration_digest
                    or answer_id not in iteration.expert_answer_ids
                ):
                    raise ValueError("expert answer application does not match research iteration")
                if (
                    answer.effective_generation is None
                    or answer.effective_generation > generation
                ):
                    raise ValueError("expert answer was applied before it became effective")
                if answer.applied_generation not in (None, generation):
                    raise ValueError("expert answer was applied to multiple generations")
                self.expert_consultation_answers[consultation_id] = replace(
                    answer, applied_generation=generation
                )
            elif event.kind == "EvolutionStageRecorded":
                validate_evolution_stage_payload(payload)
                if payload.get("status") == "completed":
                    self.gateway_retry_stage_success_seq[
                        (int(payload["generation"]), str(payload["stage"]))
                    ] = event.seq
            elif event.kind == "DshStructuredResultAccepted":
                required_fields = {
                    "schema_version",
                    "identity",
                    "output_schema_id",
                    "result_digest",
                    "structured",
                    "skill_invocation_evidence",
                }
                if not required_fields.issubset(payload) or set(payload) - (
                    required_fields | {"session_metrics", "required_tool_receipt"}
                ):
                    raise ValueError("DshStructuredResultAccepted payload is invalid")
                if payload["schema_version"] != "ecologyrsi-dsh.structured-result-accepted/1":
                    raise ValueError("unsupported DSH structured-result event version")
                identity = payload["identity"]
                structured = payload["structured"]
                if not isinstance(identity, Mapping) or identity.get("run_id") != self.run.run_id:
                    raise ValueError("DSH structured-result run identity mismatch")
                if not isinstance(structured, Mapping) or digest(structured) != payload["result_digest"]:
                    raise ValueError("DSH structured-result digest mismatch")
                stage_contracts = {
                    "generation.research": (
                        "researcher",
                        "ecology-research-result@1",
                    ),
                    "generation.search-plan": (
                        "researcher",
                        "ecology-research-search-plan@1",
                    ),
                    "generation.research-synthesis": (
                        "researcher",
                        "ecology-research-synthesis@1",
                    ),
                    "generation.reflect": (
                        "generation-judge",
                        "ecology-generation-reflection@1",
                    ),
                    "candidate.propose": (
                        "candidate-proposer",
                        "ecology-genome-mutation@1",
                    ),
                    "candidate.local_edit": (
                        "candidate-proposer",
                        "ecology-local-edit@1",
                    ),
                    "generation.judge": (
                        "generation-judge",
                        "ecology-generation-review@1",
                    ),
                    "sample.plan": (
                        "sample-planner",
                        "ecology-sample-predictions@2",
                    ),
                    "sample.critic": (
                        "sample-critic",
                        "ecology-sample-review@2",
                    ),
                    "sample.reflect": (
                        "sample-critic",
                        "ecology-sample-reflection@1",
                    ),
                }
                contract = stage_contracts.get(identity.get("stage"))
                if contract is None or (
                    identity.get("role"), payload["output_schema_id"]
                ) != contract:
                    raise ValueError("DSH structured-result stage contract mismatch")
                _validate_dsh_skill_evidence(
                    payload["skill_invocation_evidence"],
                    stage=str(identity.get("stage") or ""),
                )
                if "session_metrics" in payload:
                    session_id = identity.get("session_id")
                    if not isinstance(session_id, str) or not session_id:
                        raise ValueError("DSH structured-result session identity is invalid")
                    _validate_dsh_session_metrics(
                        payload["session_metrics"], session_id=session_id
                    )
                tool_receipt = payload.get("required_tool_receipt")
                if identity.get("stage") == "sample.plan":
                    from types import SimpleNamespace
                    from .agent_prediction import validate_prediction_receipt
                    def lookup_prediction_event(event_id):
                        record = self.dsh_prediction_tool_events.get(event_id)
                        return None if record is None else SimpleNamespace(
                            kind="DshPredictionToolExecuted", seq=record[0], payload=record[1]
                        )
                    validate_prediction_receipt(
                        structured, tool_receipt, event_lookup=lookup_prediction_event,
                        identity=identity, before_seq=event.seq,
                    )
                elif tool_receipt is not None:
                    raise ValueError(
                        "non-Planner structured result cannot claim a prediction-tool receipt"
                    )
                # A successfully admitted DSH result proves runtime continuity.
                # It closes only the native-runtime retry epoch for the generation
                # in which it was accepted; model/persistence retry classes remain
                # independent.
                self.gateway_retry_dsh_success_seq[int(self.run.generation)] = event.seq
            elif event.kind == "DshRetrievalExecuted":
                _validate_dsh_retrieval_event(payload, run_id=self.run.run_id)
            elif event.kind == "DshPredictionToolExecuted":
                from .agent_prediction import validate_tool_event
                validate_tool_event(payload)
                self.dsh_prediction_tool_events[event.event_id] = (event.seq, dict(payload))
            elif event.kind == "DshChildLaunchReserved":
                legacy_fields = {
                    "schema_version",
                    "request_id",
                    "parent_session_id",
                    "business_key_digest",
                    "launch",
                }
                if set(payload) not in (
                    legacy_fields,
                    legacy_fields | {"request_contract_digest"},
                ):
                    raise ValueError("DshChildLaunchReserved payload is invalid")
                launch = payload["launch"]
                request_contract_digest = payload.get("request_contract_digest")
                if (
                    payload["schema_version"]
                    != "ecologyrsi-dsh.child-launch-reserved/1"
                    or not isinstance(launch, Mapping)
                    or launch.get("run_id") != self.run.run_id
                    or isinstance(launch.get("launch_attempt"), bool)
                    or not isinstance(launch.get("launch_attempt"), int)
                    or launch["launch_attempt"] < 1
                    or (
                        request_contract_digest is not None
                        and (
                            not isinstance(request_contract_digest, str)
                            or len(request_contract_digest) != 64
                            or any(
                                character not in "0123456789abcdef"
                                for character in request_contract_digest
                            )
                        )
                    )
                ):
                    raise ValueError("DshChildLaunchReserved contract is invalid")
            elif event.kind == "DshSessionUsageRecorded":
                validate_session_usage(payload, run_id=event.run_id)
                # These indexes contain only already validated earlier events,
                # even when apply() received a whole history in one batch.
                identity = payload["identity"]
                reservation_id = identity["child_reservation_id"]
                launch = self.dsh_usage_launches.get(reservation_id)
                prior = {
                    item.seq: item
                    for item in (
                        self.dsh_usage_by_reservation.get(reservation_id, [])
                        + self.dsh_usage_by_session.get(identity["session_id"], [])
                    )
                }
                if launch is not None:
                    prior[launch.seq] = launch
                check_usage_binding(payload, (prior[seq] for seq in sorted(prior)))
            elif event.kind == "DshChildExecutionFailed":
                from .runtime_failure import validate_runtime_failure
                identity = payload.get("identity")
                failure_v2 = payload.get("schema_version") == "ecologyrsi-dsh.child-execution-failed/2"
                if (
                    set(payload)
                    != ({"schema_version", "identity", "error_code"} | ({"runtime_failure"} if failure_v2 else set()))
                    or payload.get("schema_version")
                    not in {"ecologyrsi-dsh.child-execution-failed/1", "ecologyrsi-dsh.child-execution-failed/2"}
                    or not isinstance(identity, Mapping)
                    or set(identity)
                    != {"child_reservation_id", "stage", "idempotency_key"}
                    or any(
                        not isinstance(identity.get(name), str)
                        or not identity.get(name)
                        for name in identity
                    )
                    or not isinstance(payload.get("error_code"), str)
                    or not payload.get("error_code")
                ):
                    raise ValueError("DshChildExecutionFailed payload is invalid")
                if failure_v2:
                    validate_runtime_failure(payload["runtime_failure"], error_code=payload["error_code"])
            elif event.kind == "GatewayRetryScheduled":
                # A gateway cooldown is an operational heartbeat only.  It must
                # survive replay so a browser refresh can distinguish a live run
                # waiting on a busy provider from a stalled/failed run.
                if payload.get("schema_version") == GATEWAY_RETRY_SCHEMA_VERSION:
                    _validate_gateway_retry_v2_payload(payload)
                    if (
                        self.run.status is not RunStatus.RUNNING
                        or int(payload["run_incarnation"]) != int(self.created.seq)
                        or int(payload["generation"]) != int(self.run.generation)
                    ):
                        raise ValueError(
                            "GatewayRetryScheduled scope does not match the running run"
                        )
                    scope = self.gateway_retry_scope(payload)
                    reset_seq = self.gateway_retry_reset_seq(
                        scope,
                        dsh_continuity_reset=(
                            payload.get("continuity_reset_contract")
                            == _DSH_CONTINUITY_RESET_CONTRACT
                        ),
                    )
                    prior_retry = self.active_gateway_retry(scope, reset_seq=reset_seq)
                    anchor = int(payload["attempt_anchor_seq"])
                    anchor_event = self.events_by_seq.get(anchor)
                    if (
                        anchor_event is None
                        or anchor >= event.seq
                        or anchor < reset_seq
                        or payload["failure_id"] in self.gateway_retry_failure_ids
                    ):
                        if payload["failure_id"] in self.gateway_retry_failure_ids:
                            raise ValueError(
                                "GatewayRetryScheduled failure_id is not unique"
                            )
                        raise ValueError("GatewayRetryScheduled retry chain is invalid")
                    if prior_retry is None:
                        chain_valid = (
                            payload["consecutive_failures"] == 1
                            and payload["breaker_epoch"]
                            == self.gateway_retry_expected_epoch(scope, reset_seq=reset_seq)
                            and payload["first_failure_at"]
                            == payload["last_failure_at"]
                        )
                    else:
                        chain_valid = (
                            anchor == prior_retry.seq
                            and payload["consecutive_failures"]
                            == prior_retry.payload["consecutive_failures"] + 1
                            and payload["breaker_epoch"]
                            == prior_retry.payload["breaker_epoch"]
                            and payload["retry_limit"]
                            == prior_retry.payload["retry_limit"]
                            and payload["first_failure_at"]
                            == prior_retry.payload["first_failure_at"]
                            and _aware_timestamp(payload["last_failure_at"])
                            >= _aware_timestamp(prior_retry.payload["last_failure_at"])
                        )
                    if not chain_valid:
                        raise ValueError("GatewayRetryScheduled retry chain is invalid")
                    first_retry = self.gateway_retry_first_by_scope.get(scope)
                    if first_retry is None or first_retry.seq <= reset_seq:
                        self.gateway_retry_first_by_scope[scope] = event
                    self.gateway_retry_last_by_scope[scope] = event
                    self.gateway_retry_max_epoch[scope] = max(
                        self.gateway_retry_max_epoch.get(scope, 0),
                        int(payload["breaker_epoch"]),
                    )
                    self.gateway_retry_failure_ids.add(str(payload["failure_id"]))
                if not isinstance(payload.get("generation"), int) or payload["generation"] < 0:
                    raise ValueError("GatewayRetryScheduled generation must be non-negative")
                if not isinstance(payload.get("retry_at"), str) or not payload["retry_at"].strip():
                    raise ValueError("GatewayRetryScheduled retry_at must be text")
            elif event.kind == "CandidateRevisionCreated":
                if set(payload) != {"revision"}:
                    raise ValueError("CandidateRevisionCreated payload is invalid")
                revision = CandidateRevision.from_dict(payload["revision"])
                candidate = self.candidates.get(revision.candidate_id)
                if (
                    revision.run_id != self.run.run_id
                    or candidate is None
                    or candidate.generation != revision.generation
                ):
                    raise ValueError("candidate revision ownership is invalid")
                if (
                    candidate.role is CandidateRole.INCUMBENT_CONTROL
                    and is_dsh_native_protocol(self.task)
                    and (
                    candidate.candidate_id
                    != f"candidate:{self.run.run_id}:seed-incumbent-control"
                    or revision.revision_id
                    != f"revision:{candidate.candidate_id}:r0"
                    or revision.parent_revision_id is not None
                    or revision.source_batch_index is not None
                    )
                ):
                    raise ValueError(
                        "incumbent control must use the deterministic seed control R0"
                    )
                existing = self.candidate_revisions.get(revision.revision_id)
                if existing is not None and existing.to_dict() != revision.to_dict():
                    raise ValueError("conflicting candidate revision")
                if existing is None:
                    if revision.parent_revision_id is None:
                        if any(
                            item.candidate_id == revision.candidate_id
                            and item.parent_revision_id is None
                            for item in self.candidate_revisions.values()
                        ):
                            raise ValueError("candidate already has an initial revision")
                    else:
                        parent = self.candidate_revisions.get(revision.parent_revision_id)
                        if parent is None or parent.candidate_id != revision.candidate_id:
                            raise ValueError("revision parent is missing or cross-candidate")
                        schedule = OptimizationSchedule.from_dict(
                            self.task.metadata["optimization_schedule"]
                        )
                        if (
                            schedule.local_evaluation_mode
                            == PAIRED_LOCAL_EVALUATION_MODE
                        ):
                            source_batch_index = revision.source_batch_index
                            assert source_batch_index is not None
                            trajectory = self.formal_trajectories.get(
                                revision.candidate_id
                            )
                            if (
                                trajectory is None
                                or trajectory.status is not TrajectoryStatus.RUNNING
                            ):
                                raise ValueError(
                                    "paired local edit child requires a running trajectory"
                                )
                            if source_batch_index >= trajectory.batch_count - 1:
                                raise ValueError(
                                    "paired final batch cannot create a local edit child"
                                )
                            comparison = self.formal_batch_comparisons.get(
                                (revision.candidate_id, source_batch_index)
                            )
                            proposal = self.local_edit_proposals.get(
                                (revision.candidate_id, source_batch_index)
                            )
                            if comparison is None or proposal is None:
                                raise ValueError(
                                    "paired local edit child requires comparison and proposal"
                                )
                            if (
                                revision.parent_revision_id
                                != comparison.champion_after_revision_id
                            ):
                                raise ValueError(
                                    "paired local edit child must descend from selected champion"
                                )
                            if (
                                proposal.get("decision")
                                != LocalEditProposalDecision.MUTATE.value
                            ):
                                raise ValueError(
                                    "paired local edit child requires a mutate proposal"
                                )
                            local_key = (
                                revision.candidate_id,
                                source_batch_index,
                            )
                            if (
                                local_key in self.local_edit_outcomes
                                or local_key in self.trajectory_revision_activations
                            ):
                                raise ValueError(
                                    "paired local edit child must be created before local decision"
                                )
                            if any(
                                item.candidate_id == revision.candidate_id
                                and item.source_batch_index == source_batch_index
                                for item in self.candidate_revisions.values()
                            ):
                                raise ValueError(
                                    "paired local edit batch can create only one challenger"
                                )
                    self.candidate_revisions[revision.revision_id] = revision
            elif event.kind == "FormalTrajectoryStarted":
                if set(payload) != {"trajectory"}:
                    raise ValueError("FormalTrajectoryStarted payload is invalid")
                trajectory = FormalTrajectory.from_dict(payload["trajectory"])
                if trajectory.status is not TrajectoryStatus.RUNNING:
                    raise ValueError("started formal trajectory must be running")
                formal = self.formal_selection_events.get(trajectory.generation)
                if (
                    formal is None
                    or trajectory.candidate_id
                    not in formal.payload["selected_candidate_ids"]
                ):
                    raise ValueError("formal trajectory requires frozen Top 2 selection")
                revision = self.candidate_revisions.get(trajectory.initial_revision_id)
                if revision is None or revision.candidate_id != trajectory.candidate_id:
                    raise ValueError("formal trajectory initial revision is invalid")
                schedule = OptimizationSchedule.from_dict(
                    self.task.metadata["optimization_schedule"]
                )
                if trajectory.batch_count != schedule.batch_count:
                    raise ValueError("formal trajectory batch_count differs from schedule")
                existing = self.formal_trajectories.get(trajectory.candidate_id)
                if existing is not None and existing.to_dict() != trajectory.to_dict():
                    raise ValueError("conflicting formal trajectory")
                self.formal_trajectories.setdefault(trajectory.candidate_id, trajectory)
            elif event.kind == "FormalBatchStarted":
                if set(payload) != {"batch"}:
                    raise ValueError("FormalBatchStarted payload is invalid")
                batch = FormalBatch.from_dict(payload["batch"])
                trajectory = self.formal_trajectories.get(batch.candidate_id)
                if (
                    trajectory is None
                    or trajectory.status is not TrajectoryStatus.RUNNING
                    or trajectory.trajectory_id != batch.trajectory_id
                    or trajectory.generation != batch.generation
                    or trajectory.batch_count != batch.batch_count
                ):
                    raise ValueError("formal batch trajectory is invalid")
                prior_batches = [
                    item
                    for (candidate_id, _index), item in self.formal_batches.items()
                    if candidate_id == batch.candidate_id
                ]
                expected_index = len(prior_batches)
                if batch.batch_index != expected_index:
                    raise ValueError("formal batch must use the next batch index")
                active_revision_id = (
                    trajectory.initial_revision_id
                    if batch.batch_index == 0
                    else self.trajectory_revision_activations[
                        (batch.candidate_id, batch.batch_index - 1)
                    ].to_revision_id
                )
                if batch.revision_id != active_revision_id:
                    raise ValueError("formal batch revision is not the active revision")
                self.formal_batches[(batch.candidate_id, batch.batch_index)] = batch
            elif event.kind == "FormalBatchEvaluated":
                if set(payload) != {"evaluation"}:
                    raise ValueError("FormalBatchEvaluated payload is invalid")
                evaluation = BatchEvaluation.from_dict(payload["evaluation"])
                candidate_id = evaluation.scope.candidate_id
                batch_index = int(evaluation.scope.batch_index)
                batch_key = (candidate_id, batch_index)
                batch = self.formal_batches.get(batch_key)
                schedule = OptimizationSchedule.from_dict(
                    self.task.metadata["optimization_schedule"]
                )
                arm = evaluation.scope.formal_batch_arm
                expected_revision_id = batch.revision_id if batch is not None else None
                if schedule.local_evaluation_mode == PAIRED_LOCAL_EVALUATION_MODE:
                    if arm is None:
                        raise ValueError("paired formal batch evaluation requires an arm")
                    if batch_index == 0:
                        if arm is not FormalBatchArm.CHAMPION:
                            raise ValueError("paired batch 0 only accepts the champion arm")
                    elif arm is FormalBatchArm.CHAMPION:
                        prior_comparison = self.formal_batch_comparisons.get(
                            (candidate_id, batch_index - 1)
                        )
                        if prior_comparison is None:
                            raise ValueError(
                                "paired champion evaluation requires prior comparison"
                            )
                        expected_revision_id = (
                            prior_comparison.champion_after_revision_id
                        )
                elif arm is not None:
                    raise ValueError("prequential formal evaluation cannot have an arm")
                key = (candidate_id, batch_index, arm)
                if (
                    batch is None
                    or evaluation.scope.run_id != batch.run_id
                    or evaluation.scope.generation != batch.generation
                    or evaluation.scope.candidate_revision_id
                    != expected_revision_id
                    or evaluation.scope.cohort_digest != batch.cohort_digest
                    or evaluation.scope.origin_count != batch.origin_count
                ):
                    raise ValueError("formal batch evaluation scope does not match batch")
                existing = self.formal_batch_evaluations.get(key)
                if existing is not None and existing.to_dict() != evaluation.to_dict():
                    raise ValueError("conflicting formal batch evaluation")
                self.formal_batch_evaluations.setdefault(key, evaluation)
            elif event.kind == "FormalBatchCompared":
                if set(payload) != {"comparison"}:
                    raise ValueError("FormalBatchCompared payload is invalid")
                comparison = FormalBatchComparison.from_dict(payload["comparison"])
                schedule = OptimizationSchedule.from_dict(
                    self.task.metadata["optimization_schedule"]
                )
                if schedule.local_evaluation_mode != PAIRED_LOCAL_EVALUATION_MODE:
                    raise ValueError("formal batch comparison requires paired schedule")
                key = (comparison.candidate_id, comparison.batch_index)
                batch = self.formal_batches.get(key)
                trajectory = self.formal_trajectories.get(comparison.candidate_id)
                evaluations_for_batch = tuple(
                    item
                    for (candidate_id, batch_index, _arm), item
                    in self.formal_batch_evaluations.items()
                    if candidate_id == comparison.candidate_id
                    and batch_index == comparison.batch_index
                )
                champion_evaluation = next(
                    (
                        item
                        for item in evaluations_for_batch
                        if item.evaluation_id
                        == comparison.champion_evaluation_id
                    ),
                    None,
                )
                challenger_evaluation = next(
                    (
                        item
                        for item in evaluations_for_batch
                        if item.evaluation_id
                        == comparison.challenger_evaluation_id
                    ),
                    None,
                )
                prior_comparison = self.formal_batch_comparisons.get(
                    (comparison.candidate_id, comparison.batch_index - 1)
                )
                expected_champion_id = (
                    trajectory.initial_revision_id
                    if comparison.batch_index == 0 and trajectory is not None
                    else prior_comparison.champion_after_revision_id
                    if prior_comparison is not None
                    else None
                )
                if (
                    batch is None
                    or trajectory is None
                    or comparison.run_id != batch.run_id
                    or comparison.generation != batch.generation
                    or comparison.cohort_digest != batch.cohort_digest
                    or comparison.champion_before_revision_id
                    != expected_champion_id
                    or comparison.challenger_revision_id != batch.revision_id
                    or champion_evaluation is None
                    or challenger_evaluation is None
                    or champion_evaluation.scope.candidate_revision_id
                    != comparison.champion_before_revision_id
                    or challenger_evaluation.scope.candidate_revision_id
                    != comparison.challenger_revision_id
                    or champion_evaluation.evaluation_digest
                    != comparison.champion_evaluation_digest
                    or challenger_evaluation.evaluation_digest
                    != comparison.challenger_evaluation_digest
                    or not math.isclose(
                        champion_evaluation.score,
                        comparison.champion_score,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        challenger_evaluation.score,
                        comparison.challenger_score,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        "formal batch comparison evaluation digest, score, "
                        "revision, or cohort is invalid"
                    )
                validate_formal_batch_comparison(
                    comparison,
                    champion_evaluation,
                    challenger_evaluation,
                    **local_challenger_policy(self.task.metadata),
                )
                existing = self.formal_batch_comparisons.get(key)
                if existing is not None and existing.to_dict() != comparison.to_dict():
                    raise ValueError("conflicting formal batch comparison")
                self.formal_batch_comparisons.setdefault(key, comparison)
            elif event.kind == "LocalEditProposalRecorded":
                legacy_fields = {
                    "proposal_id",
                    "candidate_id",
                    "batch_index",
                    "evidence_scope_digest",
                    "decision",
                    "operations",
                }
                fields = {
                    "proposal_id",
                    "candidate_id",
                    "batch_index",
                    "evidence_scope_digest",
                    "proposal",
                }
                if set(payload) not in (legacy_fields, fields, fields | {"safety_reason"}):
                    raise ValueError("local edit proposal payload is invalid")
                candidate_id = payload["candidate_id"]
                batch_index = payload["batch_index"]
                if (
                    not isinstance(candidate_id, str)
                    or not candidate_id
                    or isinstance(batch_index, bool)
                    or not isinstance(batch_index, int)
                    or batch_index < 0
                ):
                    raise ValueError("local edit proposal scope is invalid")
                key = (candidate_id, batch_index)
                existing_event = self.local_edit_proposal_events.get(key)
                if existing_event is not None:
                    if canonical_json(existing_event.payload) == canonical_json(payload):
                        continue
                    raise ValueError("conflicting local edit proposal")
                evaluation = self.replay_batch_evaluation_for(candidate_id, batch_index)
                schedule = OptimizationSchedule.from_dict(
                    self.task.metadata["optimization_schedule"]
                )
                if (
                    schedule.local_evaluation_mode
                    == PAIRED_LOCAL_EVALUATION_MODE
                ):
                    trajectory = self.formal_trajectories.get(candidate_id)
                    if (
                        trajectory is None
                        or trajectory.status is not TrajectoryStatus.RUNNING
                    ):
                        raise ValueError(
                            "paired local edit requires a running trajectory"
                        )
                    if batch_index >= trajectory.batch_count - 1:
                        raise ValueError(
                            "paired final batch cannot record a local edit proposal"
                        )
                if "proposal" in payload:
                    from ..evolution.local_edits import LocalEditProposal

                    proposal_value = LocalEditProposal.from_dict(payload["proposal"])
                    decision = proposal_value.decision
                    operations = list(proposal_value.operations)
                else:
                    decision = LocalEditProposalDecision(payload["decision"])
                    operations = payload["operations"]
                if (
                    evaluation is None
                    or (
                        schedule.local_evaluation_mode
                        == PAIRED_LOCAL_EVALUATION_MODE
                        and key not in self.formal_batch_comparisons
                    )
                    or payload["evidence_scope_digest"] != evaluation.scope.scope_key
                    or not isinstance(operations, list)
                    or len(operations) > schedule.max_local_edits_per_batch
                    or (decision is LocalEditProposalDecision.KEEP and operations)
                    or (decision is LocalEditProposalDecision.MUTATE and not operations)
                ):
                    raise ValueError("local edit proposal evidence is invalid")
                if "safety_reason" in payload and (
                    decision is not LocalEditProposalDecision.KEEP
                    or not isinstance(payload["safety_reason"], str)
                    or not payload["safety_reason"].strip()
                ):
                    raise ValueError("local edit safety reason is invalid")
                normalized = {
                    "proposal_id": payload["proposal_id"],
                    "candidate_id": candidate_id,
                    "batch_index": batch_index,
                    "evidence_scope_digest": payload["evidence_scope_digest"],
                    "proposal": (
                        proposal_value.to_dict()
                        if "proposal" in payload
                        else {
                            "schema_version": "ecology-local-edit@1",
                            "decision": decision.value,
                            "operations": [dict(item) for item in operations],
                            "evidence_refs": ["batch:score"],
                            "expected_effect_cells": [],
                            "risk_cells": [],
                        }
                    ),
                    "decision": decision.value,
                    "operations": [dict(item) for item in operations],
                }
                if "safety_reason" in payload:
                    normalized["safety_reason"] = payload["safety_reason"]
                existing = self.local_edit_proposals.get(key)
                if existing is not None and canonical_json(existing) != canonical_json(normalized):
                    raise ValueError("conflicting local edit proposal")
                self.local_edit_proposals.setdefault(key, normalized)
                self.local_edit_proposal_events.setdefault(key, event)
            elif event.kind == "LocalEditDecided":
                fields = {
                    "proposal_id",
                    "candidate_id",
                    "batch_index",
                    "outcome",
                    "active_revision_id",
                }
                if set(payload) not in (fields, fields | {"reason"}):
                    raise ValueError("local edit outcome payload is invalid")
                key = (payload["candidate_id"], payload["batch_index"])
                existing_event = self.local_edit_outcome_events.get(key)
                if existing_event is not None:
                    if canonical_json(existing_event.payload) == canonical_json(payload):
                        continue
                    raise ValueError("conflicting local edit outcome")
                proposal = self.local_edit_proposals.get(key)
                outcome = LocalEditOutcome(payload["outcome"])
                revision = self.candidate_revisions.get(payload["active_revision_id"])
                schedule = OptimizationSchedule.from_dict(
                    self.task.metadata["optimization_schedule"]
                )
                if (
                    schedule.local_evaluation_mode
                    == PAIRED_LOCAL_EVALUATION_MODE
                ):
                    trajectory = self.formal_trajectories.get(payload["candidate_id"])
                    if (
                        trajectory is None
                        or trajectory.status is not TrajectoryStatus.RUNNING
                    ):
                        raise ValueError(
                            "paired local edit requires a running trajectory"
                        )
                    if payload["batch_index"] >= trajectory.batch_count - 1:
                        raise ValueError(
                            "paired final batch cannot record a local edit decision"
                        )
                    comparison = self.formal_batch_comparisons.get(key)
                    if comparison is None:
                        raise ValueError(
                            "paired local edit decision requires a comparison"
                        )
                    proposal_decision = (
                        proposal.get("decision") if proposal is not None else None
                    )
                    if not (
                        (
                            proposal_decision
                            == LocalEditProposalDecision.KEEP.value
                            and outcome is LocalEditOutcome.KEPT
                        )
                        or (
                            proposal_decision
                            == LocalEditProposalDecision.MUTATE.value
                            and outcome
                            in (LocalEditOutcome.APPLIED, LocalEditOutcome.REJECTED)
                        )
                    ):
                        raise ValueError(
                            "paired local edit outcome does not match proposal decision"
                        )
                    selected_revision_id = comparison.champion_after_revision_id
                    children = [
                        item
                        for item in self.candidate_revisions.values()
                        if item.candidate_id == payload["candidate_id"]
                        and item.source_batch_index == payload["batch_index"]
                    ]
                    if outcome is LocalEditOutcome.APPLIED:
                        if (
                            proposal is None
                            or proposal.get("decision")
                            != LocalEditProposalDecision.MUTATE.value
                            or revision is None
                            or len(children) != 1
                            or revision.revision_id != children[0].revision_id
                            or revision.parent_revision_id != selected_revision_id
                        ):
                            raise ValueError(
                                "paired applied edit must create one child from selected champion"
                            )
                    elif (
                        outcome is LocalEditOutcome.ROLLED_BACK
                        or revision is None
                        or revision.revision_id != selected_revision_id
                        or children
                    ):
                        raise ValueError(
                            "paired retained edit must keep the selected champion "
                            "without a child"
                        )
                if (
                    proposal is None
                    or proposal["proposal_id"] != payload["proposal_id"]
                    or revision is None
                    or revision.candidate_id != payload["candidate_id"]
                    or (
                        proposal["decision"] == LocalEditProposalDecision.KEEP.value
                        and outcome is not LocalEditOutcome.KEPT
                        and not (
                            outcome is LocalEditOutcome.ROLLED_BACK
                            and payload.get("reason") == proposal.get("safety_reason")
                        )
                    )
                ):
                    raise ValueError("local edit outcome is inconsistent")
                if "reason" in payload and (
                    not isinstance(payload["reason"], str)
                    or not payload["reason"].strip()
                ):
                    raise ValueError("local edit outcome reason is invalid")
                normalized = dict(payload)
                existing = self.local_edit_outcomes.get(key)
                if existing is not None and canonical_json(existing) != canonical_json(normalized):
                    raise ValueError("conflicting local edit outcome")
                self.local_edit_outcomes.setdefault(key, normalized)
                self.local_edit_outcome_events.setdefault(key, event)
            elif event.kind == "TrajectoryRevisionAdvanced":
                if set(payload) != {"activation"}:
                    raise ValueError("TrajectoryRevisionAdvanced payload is invalid")
                activation = TrajectoryRevisionActivation.from_dict(payload["activation"])
                key = (activation.candidate_id, activation.batch_index)
                batch = self.formal_batches.get(key)
                outcome = self.local_edit_outcomes.get(key)
                destination = self.candidate_revisions.get(activation.to_revision_id)
                schedule = OptimizationSchedule.from_dict(
                    self.task.metadata["optimization_schedule"]
                )
                comparison = self.formal_batch_comparisons.get(key)
                trajectory = self.formal_trajectories.get(activation.candidate_id)
                if (
                    schedule.local_evaluation_mode
                    == PAIRED_LOCAL_EVALUATION_MODE
                ):
                    if (
                        trajectory is None
                        or trajectory.status is not TrajectoryStatus.RUNNING
                    ):
                        raise ValueError(
                            "paired revision activation requires a running trajectory"
                        )
                    if activation.batch_index >= trajectory.batch_count - 1:
                        raise ValueError(
                            "paired final batch cannot activate a local edit revision"
                        )
                source_revision_id = (
                    comparison.champion_after_revision_id
                    if schedule.local_evaluation_mode
                    == PAIRED_LOCAL_EVALUATION_MODE
                    and comparison is not None
                    else batch.revision_id
                    if batch is not None
                    else None
                )
                if (
                    batch is None
                    or outcome is None
                    or (
                        schedule.local_evaluation_mode
                        == PAIRED_LOCAL_EVALUATION_MODE
                        and comparison is None
                    )
                    or self.replay_batch_evaluation_for(
                        activation.candidate_id,
                        activation.batch_index,
                    )
                    is None
                    or activation.run_id != batch.run_id
                    or activation.generation != batch.generation
                    or activation.from_revision_id != source_revision_id
                    or destination is None
                    or destination.candidate_id != activation.candidate_id
                    or outcome["active_revision_id"] != activation.to_revision_id
                ):
                    raise ValueError("trajectory revision activation is invalid")
                expected_reason = {
                    LocalEditOutcome.KEPT.value: RevisionAdvanceReason.KEPT,
                    LocalEditOutcome.APPLIED.value: RevisionAdvanceReason.LOCAL_EDIT_APPLIED,
                    LocalEditOutcome.REJECTED.value: RevisionAdvanceReason.LOCAL_EDIT_REJECTED,
                    LocalEditOutcome.ROLLED_BACK.value: (
                        RevisionAdvanceReason.PREQUENTIAL_SAFETY_ROLLBACK
                    ),
                }[outcome["outcome"]]
                if activation.reason is not expected_reason:
                    raise ValueError("trajectory revision activation reason is inconsistent")
                if (
                    schedule.local_evaluation_mode == PAIRED_LOCAL_EVALUATION_MODE
                    and activation.reason
                    is RevisionAdvanceReason.PREQUENTIAL_SAFETY_ROLLBACK
                ):
                    raise ValueError("paired trajectory cannot use prequential rollback")
                if activation.reason is RevisionAdvanceReason.LOCAL_EDIT_APPLIED:
                    if (
                        destination.source_batch_index != activation.batch_index
                        or destination.parent_revision_id != source_revision_id
                    ):
                        raise ValueError("new revision source batch is inconsistent")
                elif activation.reason is RevisionAdvanceReason.PREQUENTIAL_SAFETY_ROLLBACK:
                    source = self.candidate_revisions.get(batch.revision_id)
                    if source is None or source.parent_revision_id != activation.to_revision_id:
                        raise ValueError("safety rollback must activate the batch revision parent")
                elif activation.to_revision_id != source_revision_id:
                    raise ValueError("kept/rejected edit cannot change active revision")
                existing = self.trajectory_revision_activations.get(key)
                if existing is not None and existing.to_dict() != activation.to_dict():
                    raise ValueError("conflicting trajectory revision activation")
                self.trajectory_revision_activations.setdefault(key, activation)
            elif event.kind == "FormalTrajectoryCompleted":
                if set(payload) != {"candidate_id", "final_revision_id"}:
                    raise ValueError("FormalTrajectoryCompleted payload is invalid")
                candidate_id = payload["candidate_id"]
                trajectory = self.formal_trajectories.get(candidate_id)
                if trajectory is None or trajectory.status is not TrajectoryStatus.RUNNING:
                    raise ValueError("formal trajectory is not running")
                required = {
                    (candidate_id, index) for index in range(trajectory.batch_count)
                }
                schedule = OptimizationSchedule.from_dict(
                    self.task.metadata["optimization_schedule"]
                )
                if schedule.local_evaluation_mode == PAIRED_LOCAL_EVALUATION_MODE:
                    local_required = {
                        (candidate_id, index)
                        for index in range(trajectory.batch_count - 1)
                    }
                    candidate_batches = {
                        key for key in self.formal_batches if key[0] == candidate_id
                    }
                    candidate_comparisons = {
                        key
                        for key in self.formal_batch_comparisons
                        if key[0] == candidate_id
                    }
                    candidate_proposals = {
                        key for key in self.local_edit_proposals if key[0] == candidate_id
                    }
                    candidate_outcomes = {
                        key for key in self.local_edit_outcomes if key[0] == candidate_id
                    }
                    candidate_activations = {
                        key
                        for key in self.trajectory_revision_activations
                        if key[0] == candidate_id
                    }
                    local_bindings_are_exact = True
                    for key in local_required:
                        comparison = self.formal_batch_comparisons.get(key)
                        proposal = self.local_edit_proposals.get(key)
                        outcome = self.local_edit_outcomes.get(key)
                        activation = self.trajectory_revision_activations.get(key)
                        children = [
                            item
                            for item in self.candidate_revisions.values()
                            if item.candidate_id == candidate_id
                            and item.source_batch_index == key[1]
                        ]
                        if (
                            comparison is None
                            or proposal is None
                            or outcome is None
                            or activation is None
                        ):
                            local_bindings_are_exact = False
                            break
                        selected_revision_id = (
                            comparison.champion_after_revision_id
                        )
                        if outcome.get("outcome") == LocalEditOutcome.APPLIED.value:
                            local_bindings_are_exact = (
                                proposal.get("decision")
                                == LocalEditProposalDecision.MUTATE.value
                                and len(children) == 1
                                and children[0].parent_revision_id
                                == selected_revision_id
                                and outcome.get("active_revision_id")
                                == children[0].revision_id
                                and activation.to_revision_id
                                == children[0].revision_id
                                and activation.reason
                                is RevisionAdvanceReason.LOCAL_EDIT_APPLIED
                            )
                        else:
                            expected = {
                                LocalEditOutcome.KEPT.value: (
                                    LocalEditProposalDecision.KEEP.value,
                                    RevisionAdvanceReason.KEPT
                                ),
                                LocalEditOutcome.REJECTED.value: (
                                    LocalEditProposalDecision.MUTATE.value,
                                    RevisionAdvanceReason.LOCAL_EDIT_REJECTED,
                                ),
                            }.get(outcome.get("outcome"))
                            local_bindings_are_exact = (
                                not children
                                and expected is not None
                                and proposal.get("decision") == expected[0]
                                and outcome.get("active_revision_id")
                                == selected_revision_id
                                and activation.to_revision_id
                                == selected_revision_id
                                and activation.reason is expected[1]
                            )
                        if not local_bindings_are_exact:
                            break
                    has_post_final_child = any(
                        item.candidate_id == candidate_id
                        and item.source_batch_index is not None
                        and item.source_batch_index >= trajectory.batch_count - 1
                        for item in self.candidate_revisions.values()
                    )
                    if not (
                        required == candidate_batches
                        and required == candidate_comparisons
                        and local_required == candidate_proposals
                        and local_required == candidate_outcomes
                        and local_required == candidate_activations
                        and local_bindings_are_exact
                        and not has_post_final_child
                    ):
                        raise ValueError(
                            "formal trajectory has incomplete paired batches or invalid "
                            "child/outcome binding"
                        )
                    final_revision_id = self.formal_batch_comparisons[
                        (candidate_id, trajectory.batch_count - 1)
                    ].champion_after_revision_id
                else:
                    legacy_evaluation_keys = {
                        (candidate_id, index, None)
                        for index in range(trajectory.batch_count)
                    }
                    if not (
                        required <= set(self.formal_batches)
                        and legacy_evaluation_keys <= set(self.formal_batch_evaluations)
                        and required <= set(self.local_edit_proposals)
                        and required <= set(self.local_edit_outcomes)
                        and required <= set(self.trajectory_revision_activations)
                    ):
                        raise ValueError("formal trajectory has incomplete batches")
                    final_revision_id = self.trajectory_revision_activations[
                        (candidate_id, trajectory.batch_count - 1)
                    ].to_revision_id
                if payload["final_revision_id"] != final_revision_id:
                    raise ValueError("formal trajectory final revision is invalid")
                self.formal_trajectories[candidate_id] = replace(
                    trajectory,
                    status=TrajectoryStatus.COMPLETED,
                    final_revision_id=final_revision_id,
                )
            elif event.kind == "GenerationHoldoutFrozen":
                if set(payload) != {"holdout"}:
                    raise ValueError("GenerationHoldoutFrozen payload is invalid")
                holdout = GenerationHoldout.from_dict(payload["holdout"])
                if holdout.run_id != self.run.run_id:
                    raise ValueError("generation holdout belongs to another run")
                completed = [
                    item
                    for item in self.formal_trajectories.values()
                    if item.generation == holdout.generation
                    and item.status is TrajectoryStatus.COMPLETED
                ]
                schedule = OptimizationSchedule.from_dict(self.task.metadata.get("optimization_schedule", OptimizationSchedule.default().to_dict()))
                if len(completed) != schedule.finalist_count or len(holdout.arm_bindings) != schedule.finalist_count + 1:
                    raise ValueError("holdout requires two completed trajectories")
                formal = self.formal_selection_events.get(holdout.generation)
                finalist_candidates = {
                    holdout.arm_bindings[arm.value]["candidate_id"]
                    for arm in map(HoldoutArm, holdout.arm_bindings) if arm is not HoldoutArm.INCUMBENT
                }
                if formal is None or finalist_candidates != set(
                    formal.payload["selected_candidate_ids"]
                ):
                    raise ValueError("holdout finalist arms do not match frozen Top 2")
                for binding in holdout.arm_bindings.values():
                    revision = self.candidate_revisions.get(binding["candidate_revision_id"])
                    if revision is None or revision.candidate_id != binding["candidate_id"]:
                        raise ValueError("holdout arm revision binding is invalid")
                completed_by_candidate = {
                    item.candidate_id: item for item in completed
                }
                for arm in map(HoldoutArm, holdout.arm_bindings):
                    if arm is HoldoutArm.INCUMBENT:
                        continue
                    binding = holdout.arm_bindings[arm.value]
                    trajectory = completed_by_candidate.get(
                        binding["candidate_id"]
                    )
                    if (
                        trajectory is None
                        or binding["candidate_revision_id"]
                        != trajectory.final_revision_id
                    ):
                        raise ValueError(
                            "holdout finalist arm must bind trajectory final revision"
                        )
                validate_generation_zero_incumbent_binding(
                    self.task,
                    holdout.generation,
                    holdout.arm_bindings,
                    candidates=self.candidates,
                    revisions=self.candidate_revisions,
                    materialized_seed_canonical=self.materialized_seed_canonical,
                    prior_effective_revision_id=(
                        str(
                            self.effective_revision_bindings[holdout.generation - 1][
                                "selected_revision_id"
                            ]
                        )
                        if holdout.generation > 0
                        and holdout.generation - 1 in self.effective_revision_bindings
                        else None
                    ),
                    prior_champion_selected=(
                        holdout.generation == 0
                        or holdout.generation - 1 in self.champion_generations
                    ),
                )
                existing = self.generation_holdouts.get(holdout.generation)
                if existing is not None and existing.to_dict() != holdout.to_dict():
                    raise ValueError("conflicting generation holdout")
                self.generation_holdouts.setdefault(holdout.generation, holdout)
            elif event.kind == "HoldoutArmStarted":
                expected_fields = {
                    "schema_version",
                    "generation",
                    "holdout_arm",
                    "candidate_id",
                    "candidate_revision_id",
                    "cohort_digest",
                    "origin_count",
                }
                if (
                    set(payload) != expected_fields
                    or payload.get("schema_version")
                    != "ecologyrsi-dsh.holdout-arm-started/1"
                ):
                    raise ValueError("HoldoutArmStarted payload is invalid")
                try:
                    arm = HoldoutArm(str(payload["holdout_arm"]))
                except ValueError:
                    raise ValueError("HoldoutArmStarted arm is invalid") from None
                generation = payload["generation"]
                holdout = (
                    self.generation_holdouts.get(generation)
                    if isinstance(generation, int) and not isinstance(generation, bool)
                    else None
                )
                binding = holdout.arm_bindings[arm.value] if holdout is not None else None
                if (
                    holdout is None
                    or binding is None
                    or payload["candidate_id"] != binding["candidate_id"]
                    or payload["candidate_revision_id"]
                    != binding["candidate_revision_id"]
                    or payload["cohort_digest"] != holdout.cohort_digest
                    or payload["origin_count"] != holdout.origin_count
                ):
                    raise ValueError("HoldoutArmStarted scope is invalid")
            elif event.kind == "HoldoutEvaluationRecorded":
                if set(payload) != {"evaluation"}:
                    raise ValueError("HoldoutEvaluationRecorded payload is invalid")
                evaluation = HoldoutEvaluation.from_dict(payload["evaluation"])
                arm = evaluation.scope.holdout_arm
                assert arm is not None
                holdout = self.generation_holdouts.get(evaluation.scope.generation)
                binding = holdout.arm_bindings[arm.value] if holdout is not None else None
                if (
                    holdout is None
                    or evaluation.scope.run_id != holdout.run_id
                    or evaluation.scope.cohort_digest != holdout.cohort_digest
                    or evaluation.scope.origin_count != holdout.origin_count
                    or binding is None
                    or evaluation.scope.candidate_id != binding["candidate_id"]
                    or evaluation.scope.candidate_revision_id
                    != binding["candidate_revision_id"]
                ):
                    raise ValueError("holdout evaluation scope is invalid")
                key = (holdout.generation, arm)
                existing = self.holdout_evaluations.get(key)
                if existing is not None and existing.to_dict() != evaluation.to_dict():
                    raise ValueError("conflicting holdout evaluation")
                self.holdout_evaluations.setdefault(key, evaluation)
            elif event.kind == "GenerationComparisonRecorded":
                if set(payload) != {"comparison"}:
                    raise ValueError("GenerationComparisonRecorded payload is invalid")
                comparison = GenerationComparison.from_dict(payload["comparison"])
                holdout = self.generation_holdouts.get(comparison.generation)
                if holdout is None or holdout.cohort_digest != comparison.cohort_digest:
                    raise ValueError("generation comparison holdout is missing")
                persisted = {
                    arm: self.holdout_evaluations.get((comparison.generation, arm))
                    for arm in map(HoldoutArm, holdout.arm_bindings)
                }
                if any(item is None for item in persisted.values()):
                    raise ValueError("generation comparison requires all three holdout arms")
                validate_generation_comparison_binding(
                    self.task,
                    self.run.run_id,
                    holdout,
                    self.formal_selection_events.get(comparison.generation),
                    comparison,
                    persisted_evaluations={
                        arm: item
                        for arm, item in persisted.items()
                        if item is not None
                    },
                    persisted_judgments={item.candidate_id: item for item in self.evaluations.values()},
                )
                existing = self.generation_comparisons.get(comparison.generation)
                if existing is not None and existing.to_dict() != comparison.to_dict():
                    raise ValueError("conflicting generation comparison")
                self.generation_comparisons.setdefault(comparison.generation, comparison)
            elif event.kind == "CandidateEffectiveRevisionFrozen":
                fields = {
                    "generation",
                    "selected_candidate_id",
                    "selected_revision_id",
                    "comparison_digest",
                }
                if set(payload) != fields:
                    raise ValueError("effective revision payload is invalid")
                generation = payload["generation"]
                comparison = self.generation_comparisons.get(generation)
                if (
                    comparison is None
                    or payload["comparison_digest"] != comparison.comparison_digest
                    or payload["selected_candidate_id"]
                    != comparison.selected_candidate_id
                    or payload["selected_revision_id"]
                    != comparison.selected_revision_id
                ):
                    raise ValueError("effective revision differs from Host comparison")
                existing = self.effective_revision_bindings.get(generation)
                if existing is not None and canonical_json(existing) != canonical_json(payload):
                    raise ValueError("conflicting effective revision")
                self.effective_revision_bindings.setdefault(generation, dict(payload))
            elif event.kind == "GenerationChampionSelected":
                fields = {"generation", "selected_candidate_id", "selected_revision_id"}
                if set(payload) != fields:
                    raise ValueError("generation champion payload is invalid")
                binding = self.effective_revision_bindings.get(payload["generation"])
                if (
                    binding is None
                    or payload["selected_candidate_id"]
                    != binding["selected_candidate_id"]
                    or payload["selected_revision_id"]
                    != binding["selected_revision_id"]
                    or payload["generation"] in self.champion_generations
                ):
                    raise ValueError("generation champion requires effective revision")
                self.champion_generations.add(payload["generation"])
            elif event.kind == "ModelUsageRecorded":
                validate_model_usage_payload(payload)
            elif event.kind == "EvaluationProgressRecorded":
                validate_evaluation_progress_payload(payload)
            elif event.kind in {
                "EvaluationSampleResultsStarted",
                "EvaluationSampleResultsResumed",
                "EvaluationSampleResultBatchRecorded",
                "EvaluationSampleResultsRecorded",
            }:
                # Full result rows are a private paginated read model. They must
                # never enter evaluations or later-generation strategy context.
                continue
            else:
                raise ValueError(f"unknown event kind: {event.kind}")

            if event.kind == "DshChildLaunchReserved":
                reservation_id = payload["launch"].get("reservation_id")
                if isinstance(reservation_id, str):
                    # Preserve the original validator's first-launch semantics.
                    self.dsh_usage_launches.setdefault(reservation_id, event)
            elif event.kind in {"DshSessionUsageRecorded", "DshStructuredResultAccepted"}:
                identity = payload.get("identity", {})
                reservation_id = identity.get("child_reservation_id")
                session_id = identity.get("session_id")
                if isinstance(reservation_id, str):
                    self.dsh_usage_by_reservation.setdefault(reservation_id, []).append(event)
                if isinstance(session_id, str):
                    self.dsh_usage_by_session.setdefault(session_id, []).append(event)

    def snapshot(self) -> RunState:

        for generation, formal in self.formal_selection_events.items():
            generation_records = [
                screening_event.payload
                for (
                    item_generation,
                    _candidate_id,
                ), screening_event in self.candidate_screening_events.items()
                if item_generation == generation
            ]
            if formal.payload["screening_digest"] != screening_cohort_digest(
                generation_records
            ):
                raise ValueError("formal screening digest does not match screening cohort")

        approved = []
        for candidate in self.candidates.values():
            if candidate.status is not CandidateStatus.PROMOTED:
                continue
            evaluation = self.evaluations.get(candidate.evaluation_id or "")
            if evaluation is not None:
                approved.append((candidate, evaluation))
        if approved and sample_update_windows_enabled(self.task):
            # Scores from different rotating windows are not a single global
            # ordering. The latest approved batch champion is the active parent;
            # score and slot only provide a deterministic same-generation tie-break.
            best = max(
                approved,
                key=lambda item: (
                    item[0].generation,
                    item[1].score,
                    -item[0].slot_index,
                    item[0].candidate_id,
                ),
            )[0].candidate_id
        else:
            best = (
                max(
                    approved,
                    key=lambda item: (
                        item[1].score,
                        item[0].created_at,
                        item[0].candidate_id,
                    ),
                )[0].candidate_id
                if approved
                else None
            )
        return RunState(
            run=replace(
                self.run,
                best_candidate_id=best,
                selection_incumbent_id=(
                    best if is_dsh_native_protocol(self.task) else self.run.selection_incumbent_id
                ),
            ),
            task_manifest=self.task,
            proposals=tuple(self.proposals.values()),
            candidates=tuple(self.candidates.values()),
            artifacts=tuple(self.artifacts.values()),
            evaluations=tuple(self.evaluations.values()),
            promotions=tuple(self.promotions.values()),
            interventions=tuple(self.interventions.values()),
            generation_batches=tuple(self.generation_batches.values()),
            generation_analyses=tuple(self.generation_analyses.values()),
            knowledge_snapshots=tuple(self.knowledge_snapshots.values()),
            knowledge_assessments=tuple(self.knowledge_assessments.values()),
            research_iterations=tuple(self.research_iterations.values()),
            algorithm_attempts=tuple(self.algorithm_attempts),
            events=self.events,
            generation_search_plans=tuple(self.generation_search_plans.values()),
            generation_reflections=tuple(self.generation_reflections.values()),
            expert_consultations=tuple(self.expert_consultations.values()),
            expert_consultation_answers=tuple(self.expert_consultation_answers.values()),
            materialized_seed_genome_canonical_json=self.materialized_seed_canonical,
            candidate_identity_bindings=tuple(self.candidate_identity_bindings.values()),
            formal_stage_seals=tuple(self.formal_stage_seals.values()),
            candidate_screening_events=tuple(self.candidate_screening_events.values()),
            formal_selection_events=tuple(self.formal_selection_events.values()),
            screened_out_events=tuple(self.screened_out_events.values()),
            candidate_revisions=tuple(self.candidate_revisions.values()),
            formal_trajectories=tuple(self.formal_trajectories.values()),
            formal_batches=tuple(self.formal_batches.values()),
            formal_batch_evaluations=tuple(self.formal_batch_evaluations.values()),
            formal_batch_comparisons=tuple(self.formal_batch_comparisons.values()),
            local_edit_proposals=tuple(self.local_edit_proposals.values()),
            local_edit_outcomes=tuple(self.local_edit_outcomes.values()),
            trajectory_revision_activations=tuple(
                self.trajectory_revision_activations.values()
            ),
            generation_holdouts=tuple(self.generation_holdouts.values()),
            holdout_evaluations=tuple(self.holdout_evaluations.values()),
            generation_comparisons=tuple(self.generation_comparisons.values()),
            effective_revision_bindings=tuple(self.effective_revision_bindings.values()),
            run_adaptation_cohort=self.run_adaptation_cohort,
            generation_selection_cohorts=tuple(
                self.generation_selection_cohorts.values()
            ),
        )


def project_run_state(events: tuple[Event, ...]) -> RunState:
    """Full replay remains available for independent cache verification."""
    if not events:
        raise ValueError("run event stream must not be empty")
    reducer = RunStateReducer(events[0])
    reducer.apply(events[1:])
    return reducer.snapshot()


__all__ = [
    "DSH_NATIVE_EVOLUTION_PROTOCOL",
    "RunState",
    "is_dsh_native_protocol",
    "persisted_genome_from_proposal",
    "project_run_state",
    "validate_evolution_stage_payload",
    "validate_identity_binding",
]
