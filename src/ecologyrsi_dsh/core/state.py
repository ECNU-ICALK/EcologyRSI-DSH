"""Read-only run state and deterministic event-stream replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ..evolution.analysis import (
    GenerationAnalysis,
    GenerationBatch,
    sample_update_windows_enabled,
)
from ..knowledge.algorithms import AlgorithmAttempt
from ..knowledge.models import KnowledgeAssessment, KnowledgeSnapshot
from ..knowledge.research_iteration import ResearchIteration
from .ledger import Event
from .models import (
    Candidate,
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

DSH_NATIVE_EVOLUTION_PROTOCOL = "dsh_native_plugin_evolution@1"


def _validate_dsh_session_metrics(value: Any, *, session_id: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "session_id",
        "context_pressure",
        "provider_usage",
    }:
        raise ValueError("DSH session metrics have an invalid shape")
    if value.get("schema_version") != "ecologyrsi-dsh.dsh-session-metrics/1":
        raise ValueError("unsupported DSH session metrics schema")
    if value.get("session_id") != session_id:
        raise ValueError("DSH session metrics identity mismatch")

    pressure = value.get("context_pressure")
    if not isinstance(pressure, Mapping):
        raise ValueError("DSH context pressure must be an object")
    if pressure.get("available") is True:
        if set(pressure) != {
            "available",
            "source",
            "measurement",
            "log_revision",
            "baseline_kind",
            "total_tokens",
            "surface_tokens",
        }:
            raise ValueError("DSH context pressure has an invalid shape")
        if (
            pressure.get("source") != "dsh_token_meter"
            or pressure.get("measurement") != "current_context_pressure"
            or pressure.get("baseline_kind") not in {"none", "estimated", "usage"}
        ):
            raise ValueError("DSH context pressure semantics are invalid")
        for name in ("log_revision", "total_tokens", "surface_tokens"):
            item = pressure.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"DSH context pressure {name} is invalid")
    elif dict(pressure) != {"available": False, "source": "dsh_token_meter"}:
        raise ValueError("unavailable DSH context pressure is invalid")

    usage = value.get("provider_usage")
    if not isinstance(usage, Mapping):
        raise ValueError("DSH provider usage must be an object")
    if usage.get("available") is True:
        if set(usage) != {"available", "source", "measurement", "totals"}:
            raise ValueError("DSH provider usage has an invalid shape")
        if (
            usage.get("source") != "dsh_session_projection_token_usage"
            or usage.get("measurement") != "cumulative_provider_reported_usage"
        ):
            raise ValueError("DSH provider usage semantics are invalid")
        totals = usage.get("totals")
        fields = {
            "uncached_input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "total_tokens",
        }
        if not isinstance(totals, Mapping) or set(totals) != fields:
            raise ValueError("DSH provider usage totals have an invalid shape")
        for name in fields:
            item = totals.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"DSH provider usage {name} is invalid")
        if totals["total_tokens"] != sum(
            int(totals[name]) for name in fields - {"total_tokens"}
        ):
            raise ValueError("DSH provider total token count is inconsistent")
    elif dict(usage) != {
        "available": False,
        "source": "dsh_session_projection_token_usage",
    }:
        raise ValueError("unavailable DSH provider usage is invalid")


def is_dsh_native_protocol(task: TaskManifest) -> bool:
    return task.metadata.get("execution_protocol") == DSH_NATIVE_EVOLUTION_PROTOCOL


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
        "research",
        "proposal",
        "candidate",
        "training",
        "evaluation",
        "judge",
        "decision",
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
    expert_consultations: tuple[ExpertConsultation, ...] = ()
    expert_consultation_answers: tuple[ExpertConsultationAnswer, ...] = ()
    materialized_seed_genome_canonical_json: str | None = None
    candidate_identity_bindings: tuple[Mapping[str, Any], ...] = ()
    formal_stage_seals: tuple[Mapping[str, Any], ...] = ()

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

    def projected_legacy_genome_for(self, candidate_id: str):
        from ..evolution.genome import legacy_genome_from_proposal

        candidate = self.candidate(candidate_id)
        proposal = self.proposal(candidate.proposal_id)
        if persisted_genome_from_proposal(proposal) is not None:
            raise ValueError("DSH-native candidates do not use legacy projection")
        return legacy_genome_from_proposal(
            proposal,
            self.task_manifest,
            self.knowledge_for(candidate.generation),
        )

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

    def algorithm_debug_passed(self, candidate_id: str) -> bool:
        return any(
            item.phase == "debug" and item.status == "passed"
            for item in self.algorithm_attempts_for(candidate_id)
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


def project_run_state(events: tuple[Event, ...]) -> RunState:
    created = events[0]
    if created.kind != "RunCreated":
        raise ValueError("run event stream must start with RunCreated")
    task = TaskManifest.from_dict(created.payload["task_manifest"])
    run = Run.from_dict(created.payload["run"])
    expected_seed_canonical = _expected_seed_canonical(created.payload, task)
    materialized_seed_canonical: str | None = None
    proposals: dict[str, Proposal] = {}
    candidates: dict[str, Candidate] = {}
    artifacts: dict[str, ModelArtifact] = {}
    evaluations: dict[str, Evaluation] = {}
    promotions: dict[str, Promotion] = {}
    interventions: dict[str, HumanIntervention] = {}
    expert_consultations: dict[str, ExpertConsultation] = {}
    expert_consultation_answers: dict[str, ExpertConsultationAnswer] = {}
    generation_batches: dict[int, GenerationBatch] = {}
    generation_analyses: dict[int, GenerationAnalysis] = {}
    knowledge_snapshots: dict[int, KnowledgeSnapshot] = {}
    knowledge_assessments: dict[int, KnowledgeAssessment] = {}
    research_iterations: dict[int, ResearchIteration] = {}
    algorithm_attempts: list[AlgorithmAttempt] = []
    candidate_identity_bindings: dict[str, dict[str, Any]] = {}
    formal_stage_seals: dict[str, dict[str, Any]] = {}
    formal_stage_started = False

    for event in events[1:]:
        payload = event.payload
        if event.kind == "RunSeedGenomeMaterialized":
            if expected_seed_canonical is None:
                raise ValueError("historical run cannot materialize a DSH-native seed")
            if materialized_seed_canonical is not None:
                raise ValueError("run has multiple seed materialization events")
            if not isinstance(payload, Mapping) or set(payload) != {
                "schema_version",
                "materializer_version",
                "genome_canonical_json",
                "genome_digest",
            }:
                raise ValueError("RunSeedGenomeMaterialized payload is invalid")
            if payload["genome_canonical_json"] != expected_seed_canonical:
                raise ValueError("materialized seed differs from RunCreated expectation")
            from ..evolution.genome import EcologyEvolutionPluginGenome

            import json

            seed = EcologyEvolutionPluginGenome.from_dict(
                json.loads(expected_seed_canonical)
            )
            if payload["genome_digest"] != seed.genome_digest:
                raise ValueError("materialized seed genome digest mismatch")
            materialized_seed_canonical = expected_seed_canonical
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
        elif event.kind == "RunStarted":
            if expected_seed_canonical is not None and materialized_seed_canonical is None:
                raise ValueError("DSH-native run started before seed initialization")
            if run.status not in _TERMINAL_RUN_STATUSES:
                run = replace(
                    run, status=RunStatus.RUNNING, session_id=payload["session_id"]
                )
        elif event.kind == "RunPaused":
            if run.status not in _TERMINAL_RUN_STATUSES:
                run = replace(run, status=RunStatus.PAUSED)
        elif event.kind == "RunResumed":
            if run.status not in _TERMINAL_RUN_STATUSES:
                run = replace(run, status=RunStatus.RUNNING)
        elif event.kind == "RunCancelled":
            if run.status not in _TERMINAL_RUN_STATUSES:
                run = replace(run, status=RunStatus.CANCELLED)
        elif event.kind == "RunFailed":
            if run.status not in _TERMINAL_RUN_STATUSES:
                run = replace(run, status=RunStatus.FAILED)
        elif event.kind == "RunCompleted":
            if run.status not in _TERMINAL_RUN_STATUSES:
                run = replace(run, status=RunStatus.COMPLETED)
        elif event.kind == "GenerationAdvanced":
            if run.status not in _TERMINAL_RUN_STATUSES:
                run = replace(run, generation=int(payload["generation"]))
        elif event.kind == "ProposalSubmitted":
            if is_dsh_native_protocol(task) and formal_stage_started:
                raise ValueError("formal stage results cannot feed a new proposal")
            item = Proposal.from_dict(payload["proposal"])
            if is_dsh_native_protocol(task):
                genome = persisted_genome_from_proposal(item)
                if genome is None:
                    raise ValueError("DSH-native proposal is missing its genome")
                lineage = dict(genome.lineage)
                if (
                    lineage["generation"] != item.generation
                    or lineage["parent_candidate_id"] != item.parent_candidate_id
                ):
                    raise ValueError("proposal scope does not match genome lineage")
                if item.generation == 0:
                    if materialized_seed_canonical is None:
                        raise ValueError("proposal cannot precede seed materialization")
                    import json

                    from ..evolution.genome import EcologyEvolutionPluginGenome

                    seed = EcologyEvolutionPluginGenome.from_dict(
                        json.loads(materialized_seed_canonical)
                    )
                    if lineage["parent_genome_digest"] != seed.genome_digest:
                        raise ValueError("first-generation proposal parent genome mismatch")
                elif item.parent_candidate_id is None:
                    raise ValueError("later DSH-native proposal requires a parent candidate")
            proposals[item.proposal_id] = item
        elif event.kind == "CandidateSpawned":
            item = Candidate.from_dict(payload["candidate"])
            if is_dsh_native_protocol(task):
                proposal = proposals.get(item.proposal_id)
                if proposal is None:
                    raise ValueError("candidate is missing its DSH-native proposal")
                genome = persisted_genome_from_proposal(proposal)
                if genome is None:
                    raise ValueError("candidate proposal is missing its genome")
                lineage = dict(genome.lineage)
                if (
                    item.generation != lineage["generation"]
                    or item.slot_index != lineage["slot_index"]
                ):
                    raise ValueError("candidate coordinates do not match genome lineage")
                binding = validate_identity_binding(payload.get("identity_binding"))
                if binding["genome_digest"] != genome.genome_digest:
                    raise ValueError("candidate identity binding genome mismatch")
                if binding["behavior_digest"] != genome.behavior_digest:
                    raise ValueError("candidate identity binding behavior mismatch")
                candidate_identity_bindings[item.candidate_id] = {
                    "candidate_id": item.candidate_id,
                    "identity_binding": binding,
                }
            candidates[item.candidate_id] = item
        elif event.kind == "ArtifactRecorded":
            item = ModelArtifact.from_dict(payload["artifact"])
            if is_dsh_native_protocol(task):
                expected_binding = candidate_identity_bindings.get(item.candidate_id)
                if expected_binding is None:
                    raise ValueError("artifact candidate has no identity binding")
                validate_identity_binding(
                    payload.get("identity_binding"),
                    expected=expected_binding["identity_binding"],
                )
                if payload["artifact"].get("artifact_digest") != item.digest:
                    raise ValueError("artifact digest mismatch")
            existing = artifacts.get(item.artifact_id)
            if existing is not None and existing.to_dict() != item.to_dict():
                raise ValueError("artifact_id belongs to multiple candidates")
            artifacts[item.artifact_id] = item
        elif event.kind == "EvaluationRecorded":
            item = Evaluation.from_dict(payload["evaluation"])
            if is_dsh_native_protocol(task):
                expected_binding = candidate_identity_bindings.get(item.candidate_id)
                if expected_binding is None:
                    raise ValueError("evaluation candidate has no identity binding")
                validate_identity_binding(
                    payload.get("identity_binding"),
                    expected=expected_binding["identity_binding"],
                )
                artifact = next(
                    (
                        artifact
                        for artifact in artifacts.values()
                        if artifact.candidate_id == item.candidate_id
                    ),
                    None,
                )
                if artifact is None or item.artifact_digest != artifact.digest:
                    raise ValueError("evaluation artifact binding mismatch")
                if payload.get("artifact_digest") != artifact.digest:
                    raise ValueError("evaluation event artifact digest mismatch")
                if payload.get("evaluation_digest") != digest(item.to_dict()):
                    raise ValueError("evaluation event digest mismatch")
            existing = evaluations.get(item.evaluation_id)
            if existing is not None and existing.to_dict() != item.to_dict():
                raise ValueError("evaluation_id belongs to multiple candidates")
            evaluations[item.evaluation_id] = item
            candidate = candidates.get(item.candidate_id)
            if candidate is not None:
                candidates[item.candidate_id] = replace(
                    candidate,
                    status=CandidateStatus.EVALUATED,
                    evaluation_id=item.evaluation_id,
                )
        elif event.kind == "EvaluationJudged":
            item = Evaluation.from_dict(payload["evaluation"])
            if is_dsh_native_protocol(task):
                expected_binding = candidate_identity_bindings.get(item.candidate_id)
                if expected_binding is None:
                    raise ValueError("judgment candidate has no identity binding")
                validate_identity_binding(
                    payload.get("identity_binding"),
                    expected=expected_binding["identity_binding"],
                )
                artifact = next(
                    (
                        artifact
                        for artifact in artifacts.values()
                        if artifact.candidate_id == item.candidate_id
                    ),
                    None,
                )
                if artifact is None or item.artifact_digest != artifact.digest:
                    raise ValueError("judgment artifact binding mismatch")
                if payload.get("artifact_digest") != artifact.digest:
                    raise ValueError("judgment event artifact digest mismatch")
                if payload.get("evaluation_digest") != digest(item.to_dict()):
                    raise ValueError("judgment event evaluation digest mismatch")
            existing = evaluations.get(item.evaluation_id)
            if existing is None:
                raise ValueError("judged evaluation is missing its scientific evaluation")
            if existing.candidate_id != item.candidate_id:
                raise ValueError("judged evaluation belongs to another candidate")
            evaluations[item.evaluation_id] = item
        elif event.kind == "PromotionDecided":
            item = Promotion.from_dict(payload["promotion"])
            if is_dsh_native_protocol(task):
                expected_binding = candidate_identity_bindings.get(item.candidate_id)
                if expected_binding is None:
                    raise ValueError("promotion candidate has no identity binding")
                validate_identity_binding(
                    payload.get("identity_binding"),
                    expected=expected_binding["identity_binding"],
                )
                evaluation = next(
                    (
                        evaluation
                        for evaluation in evaluations.values()
                        if evaluation.candidate_id == item.candidate_id
                    ),
                    None,
                )
                artifact = next(
                    (
                        artifact
                        for artifact in artifacts.values()
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
            existing = promotions.get(item.promotion_id)
            if existing is not None and existing.to_dict() != item.to_dict():
                raise ValueError("promotion_id belongs to multiple candidates")
            promotions[item.promotion_id] = item
            candidate = candidates.get(item.candidate_id)
            if candidate is not None:
                status = (
                    CandidateStatus.PROMOTED
                    if item.decision is PromotionDecision.APPROVED
                    else CandidateStatus.REJECTED
                )
                candidates[item.candidate_id] = replace(
                    candidate, status=status, promotion_id=item.promotion_id
                )
        elif event.kind in {"CandidateFailed", "CandidateMarkedDuplicate"}:
            candidate_id = str(payload["candidate_id"])
            candidate = candidates.get(candidate_id)
            if candidate is not None:
                status = (
                    CandidateStatus.FAILED
                    if event.kind == "CandidateFailed"
                    else CandidateStatus.DUPLICATE
                )
                candidates[candidate_id] = replace(candidate, status=status)
        elif event.kind == "GenerationBatchStarted":
            if expected_seed_canonical is not None and materialized_seed_canonical is None:
                raise ValueError("generation batch cannot precede seed materialization")
            item = GenerationBatch.from_dict(payload["batch"])
            if is_dsh_native_protocol(task):
                if item.parent_genome_canonical_json is None:
                    raise ValueError("DSH-native batch is missing its parent genome")
                if item.generation == 0:
                    if item.parent_candidate_id is not None:
                        raise ValueError("first generation cannot bind a parent candidate")
                    if item.parent_genome_canonical_json != materialized_seed_canonical:
                        raise ValueError("first generation parent differs from materialized seed")
                else:
                    if item.parent_candidate_id is None:
                        raise ValueError("later generation batch requires a parent candidate")
                    parent_candidate = candidates.get(item.parent_candidate_id)
                    if parent_candidate is None:
                        raise ValueError("generation parent candidate is missing")
                    parent_proposal = proposals.get(parent_candidate.proposal_id)
                    if parent_proposal is None:
                        raise ValueError("generation parent proposal is missing")
                    parent_genome = persisted_genome_from_proposal(parent_proposal)
                    if (
                        parent_genome is None
                        or canonical_json(parent_genome.to_dict())
                        != item.parent_genome_canonical_json
                    ):
                        raise ValueError("generation parent genome binding mismatch")
            generation_batches[item.generation] = item
        elif event.kind == "GenerationAnalyzed":
            item = GenerationAnalysis.from_dict(payload["analysis"])
            generation_analyses[item.generation] = item
        elif event.kind == "GenerationKnowledgeRetrieved":
            item = KnowledgeSnapshot.from_dict(payload["knowledge_snapshot"])
            knowledge_snapshots[item.generation] = item
        elif event.kind == "GenerationKnowledgeAssessed":
            item = KnowledgeAssessment.from_dict(payload["knowledge_assessment"])
            knowledge_assessments[item.generation] = item
        elif event.kind == "GenerationResearchIterated":
            item = ResearchIteration.from_dict(payload["research_iteration"])
            if item.run_id != run.run_id:
                raise ValueError("research iteration belongs to another run")
            existing = research_iterations.get(item.generation)
            if existing is not None and existing.to_dict() != item.to_dict():
                raise ValueError("generation has multiple research iterations")
            research_iterations[item.generation] = item
        elif event.kind == "AlgorithmAttemptRecorded":
            item = AlgorithmAttempt.from_dict(payload["algorithm_attempt"])
            if item.run_id != run.run_id:
                raise ValueError("algorithm attempt belongs to another run")
            candidate = candidates.get(item.candidate_id)
            if candidate is None:
                raise ValueError("algorithm attempt is missing its candidate")
            if (
                candidate.proposal_id != item.proposal_id
                or candidate.generation != item.generation
            ):
                raise ValueError("algorithm attempt scope does not match its candidate")
            if is_dsh_native_protocol(task):
                expected_binding = candidate_identity_bindings.get(item.candidate_id)
                if expected_binding is None:
                    raise ValueError("algorithm attempt candidate has no identity binding")
                validate_identity_binding(
                    payload.get("identity_binding"),
                    expected=expected_binding["identity_binding"],
                )
            duplicate = next(
                (
                    existing
                    for existing in algorithm_attempts
                    if existing.candidate_id == item.candidate_id
                    and existing.phase == item.phase
                    and existing.attempt == item.attempt
                ),
                None,
            )
            if duplicate is not None and duplicate.to_dict() != item.to_dict():
                raise ValueError("algorithm attempt identity was reused")
            if duplicate is None:
                algorithm_attempts.append(item)
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
            if not is_dsh_native_protocol(task) or set(payload) != required:
                raise ValueError("formal stage frozen payload is invalid")
            stage = str(payload["stage"])
            if stage not in {"validation", "final_test"}:
                raise ValueError("formal stage is invalid")
            if stage in formal_stage_seals:
                raise ValueError("formal stage is already sealed")
            candidate = candidates.get(str(payload["candidate_id"]))
            if candidate is None:
                raise ValueError("formal stage candidate is missing")
            artifact = next(
                (
                    item
                    for item in artifacts.values()
                    if item.candidate_id == candidate.candidate_id
                ),
                None,
            )
            if artifact is None or artifact.digest != payload["artifact_digest"]:
                raise ValueError("formal stage artifact binding mismatch")
            binding = candidate_identity_bindings.get(candidate.candidate_id)
            if (
                binding is None
                or binding["identity_binding"]["genome_digest"]
                != payload["genome_digest"]
            ):
                raise ValueError("formal stage genome binding mismatch")
            if stage == "final_test" and run.validated_candidate_id != candidate.candidate_id:
                raise ValueError("final-test requires the validated candidate")
            formal_stage_started = True
        elif event.kind == "FormalStageCompleted":
            if not formal_stage_started:
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
            if candidate_id not in candidates:
                raise ValueError("formal stage completion candidate is missing")
            if outcome == "passed":
                run = replace(
                    run,
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
            if stage in formal_stage_seals:
                raise ValueError("formal stage has multiple seals")
            formal_stage_seals[stage] = dict(payload)
        elif event.kind == "GenerationChampionSelected":
            continue
        elif event.kind == "HumanInterventionRecorded":
            item = HumanIntervention.from_dict(payload["intervention"])
            existing = interventions.get(item.intervention_id)
            if existing is not None and existing.to_dict() != item.to_dict():
                raise ValueError("intervention_id belongs to multiple interventions")
            interventions[item.intervention_id] = item
        elif event.kind == "HumanInterventionApplied":
            intervention_id = str(payload["intervention_id"])
            item = interventions.get(intervention_id)
            if item is None:
                raise ValueError("applied intervention is missing its recorded event")
            interventions[intervention_id] = replace(
                item, applied_proposal_id=str(payload["proposal_id"])
            )
        elif event.kind == "ExpertConsultationRequested":
            item = ExpertConsultation.from_dict(payload["consultation"])
            if item.run_id != run.run_id:
                raise ValueError("expert consultation belongs to another run")
            existing = expert_consultations.get(item.consultation_id)
            if existing is not None and existing.to_dict() != item.to_dict():
                raise ValueError("consultation_id belongs to multiple consultations")
            expert_consultations[item.consultation_id] = item
        elif event.kind == "ExpertConsultationAnswered":
            item = ExpertConsultationAnswer.from_dict(payload["answer"])
            if item.run_id != run.run_id:
                raise ValueError("expert consultation answer belongs to another run")
            consultation = expert_consultations.get(item.consultation_id)
            if consultation is None:
                raise ValueError("expert answer is missing its consultation request")
            existing = expert_consultation_answers.get(item.consultation_id)
            if existing is not None and existing.to_dict() != item.to_dict():
                raise ValueError("expert consultation already has a different answer")
            expert_consultation_answers[item.consultation_id] = item
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
            if consultation_id not in expert_consultations:
                raise ValueError("applied expert answer is missing its consultation")
            answer = expert_consultation_answers.get(consultation_id)
            if answer is None or answer.answer_id != answer_id:
                raise ValueError("applied expert answer is missing its answer event")
            iteration = research_iterations.get(generation)
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
            expert_consultation_answers[consultation_id] = replace(
                answer, applied_generation=generation
            )
        elif event.kind == "EvolutionStageRecorded":
            validate_evolution_stage_payload(payload)
        elif event.kind == "DshStructuredResultAccepted":
            required_fields = {
                "schema_version",
                "identity",
                "output_schema_id",
                "result_digest",
                "structured",
            }
            if not required_fields.issubset(payload) or set(payload) - (
                required_fields | {"session_metrics"}
            ):
                raise ValueError("DshStructuredResultAccepted payload is invalid")
            if payload["schema_version"] != "ecologyrsi-dsh.structured-result-accepted/1":
                raise ValueError("unsupported DSH structured-result event version")
            identity = payload["identity"]
            structured = payload["structured"]
            if not isinstance(identity, Mapping) or identity.get("run_id") != run.run_id:
                raise ValueError("DSH structured-result run identity mismatch")
            if not isinstance(structured, Mapping) or digest(structured) != payload["result_digest"]:
                raise ValueError("DSH structured-result digest mismatch")
            stage_contracts = {
                "generation.research": (
                    "researcher",
                    "ecology-research-result@1",
                ),
                "candidate.propose": (
                    "candidate-proposer",
                    "ecology-genome-mutation@1",
                ),
                "generation.judge": (
                    "generation-judge",
                    "ecology-generation-review@1",
                ),
                "sample.plan": (
                    "sample-planner",
                    "ecology-sample-decisions@1",
                ),
                "sample.critic": (
                    "sample-critic",
                    "ecology-sample-review@1",
                ),
            }
            contract = stage_contracts.get(identity.get("stage"))
            if contract is None or (
                identity.get("role"), payload["output_schema_id"]
            ) != contract:
                raise ValueError("DSH structured-result stage contract mismatch")
            if "session_metrics" in payload:
                session_id = identity.get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError("DSH structured-result session identity is invalid")
                _validate_dsh_session_metrics(
                    payload["session_metrics"], session_id=session_id
                )
        elif event.kind == "DshChildLaunchReserved":
            if set(payload) != {
                "schema_version",
                "request_id",
                "parent_session_id",
                "business_key_digest",
                "launch",
            }:
                raise ValueError("DshChildLaunchReserved payload is invalid")
            launch = payload["launch"]
            if (
                payload["schema_version"]
                != "ecologyrsi-dsh.child-launch-reserved/1"
                or not isinstance(launch, Mapping)
                or launch.get("run_id") != run.run_id
                or isinstance(launch.get("launch_attempt"), bool)
                or not isinstance(launch.get("launch_attempt"), int)
                or launch["launch_attempt"] < 1
            ):
                raise ValueError("DshChildLaunchReserved contract is invalid")
        elif event.kind == "GatewayRetryScheduled":
            # A gateway cooldown is an operational heartbeat only.  It must
            # survive replay so a browser refresh can distinguish a live run
            # waiting on a busy provider from a stalled/failed run.
            if not isinstance(payload.get("generation"), int) or payload["generation"] < 0:
                raise ValueError("GatewayRetryScheduled generation must be non-negative")
            if not isinstance(payload.get("retry_at"), str) or not payload["retry_at"].strip():
                raise ValueError("GatewayRetryScheduled retry_at must be text")
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

    approved = []
    for candidate in candidates.values():
        if candidate.status is not CandidateStatus.PROMOTED:
            continue
        evaluation = evaluations.get(candidate.evaluation_id or "")
        if evaluation is not None:
            approved.append((candidate, evaluation))
    if approved and sample_update_windows_enabled(task):
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
            run,
            best_candidate_id=best,
            selection_incumbent_id=(
                best if is_dsh_native_protocol(task) else run.selection_incumbent_id
            ),
        ),
        task_manifest=task,
        proposals=tuple(proposals.values()),
        candidates=tuple(candidates.values()),
        artifacts=tuple(artifacts.values()),
        evaluations=tuple(evaluations.values()),
        promotions=tuple(promotions.values()),
        interventions=tuple(interventions.values()),
        generation_batches=tuple(generation_batches.values()),
        generation_analyses=tuple(generation_analyses.values()),
        knowledge_snapshots=tuple(knowledge_snapshots.values()),
        knowledge_assessments=tuple(knowledge_assessments.values()),
        research_iterations=tuple(research_iterations.values()),
        algorithm_attempts=tuple(algorithm_attempts),
        events=events,
        expert_consultations=tuple(expert_consultations.values()),
        expert_consultation_answers=tuple(expert_consultation_answers.values()),
        materialized_seed_genome_canonical_json=materialized_seed_canonical,
        candidate_identity_bindings=tuple(candidate_identity_bindings.values()),
        formal_stage_seals=tuple(formal_stage_seals.values()),
    )


__all__ = [
    "DSH_NATIVE_EVOLUTION_PROTOCOL",
    "RunState",
    "is_dsh_native_protocol",
    "persisted_genome_from_proposal",
    "project_run_state",
    "validate_evolution_stage_payload",
    "validate_identity_binding",
]
