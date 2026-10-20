"""Compile research evidence into host-registered executable algorithms.

The compiler deliberately has no dynamic import or source-code execution path.
It binds a proposal to an adapter and tool chain already shipped by the host,
then emits an immutable digest that can be checked before training starts.
"""

from __future__ import annotations

from ..core.prediction_policy import RUNTIME_EVALUATOR_ID, CELLWISE_EVALUATOR_ID

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core.models import (
    JsonObject,
    Proposal,
    TaskManifest,
    canonical_json,
    digest,
    utc_now,
)
from ..evolution.execution_plan import DerivedExecutionPlan
from .algorithm_ir import (
    AlgorithmIR,
)
from .models import KnowledgeSnapshot

ALGORITHM_SPEC_VERSION = "ecologyrsi-dsh.algorithm-spec/1"
EVOLUTION_ALLOWED_PARTITIONS = ("training_fit", "training_feedback")
_FORBIDDEN_PARTITIONS = frozenset(
    {"hidden", "final", "test", "development", "gate", "external"}
)


class AlgorithmCompileError(ValueError):
    """A proposal cannot be mapped to a registered host algorithm."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_CAPABILITIES: dict[str, dict[str, dict[str, Any]]] = {
    "strategy": {
        "autonomous_model@1": {
            "version": "per-generation-research-runtime-adoption/9"
        },
    },
    "predictor": {
        "greenhouse-exogenous-ridge@1": {
            "version": "greenhouse-exogenous-ridge/1",
            "parameter_names": ("history_steps", "ridge_alpha", "residual_scale"),
            "tool_ids": ("host.predictor.greenhouse-exogenous-ridge.fit-predict@1",),
            "evaluator_ids": (RUNTIME_EVALUATOR_ID, CELLWISE_EVALUATOR_ID),
        },
        "greenhouse-baseline-aligned-ridge@2": {
            "version": "greenhouse-baseline-aligned-ridge/2",
            "parameter_names": ("history_steps", "ridge_alpha", *tuple(f"{t}_{h}h_residual_scale" for t in ("air_temperature", "relative_humidity", "co2_concentration") for h in (1, 6, 24))),
            "tool_ids": ("host.predictor.greenhouse-baseline-aligned-ridge.fit-predict@2",),
            "evaluator_ids": (RUNTIME_EVALUATOR_ID, CELLWISE_EVALUATOR_ID),
        },
        "greenhouse-baseline-aligned-ridge@1": {
            "version": "greenhouse-baseline-aligned-ridge/1",
            "parameter_names": ("history_steps", "ridge_alpha",
                                "residual_scale_1h", "residual_scale_6h", "residual_scale_24h"),
            "tool_ids": ("host.predictor.greenhouse-baseline-aligned-ridge.fit-predict@1",),
            # Also admissible on the recipe evaluator, where it is the
            # incumbent a recipe challenger is compared against.
            "evaluator_ids": (RUNTIME_EVALUATOR_ID, CELLWISE_EVALUATOR_ID),
        },
        "greenhouse-targetwise-ridge@1": {
            "version": "greenhouse-targetwise-ridge/1",
            "parameter_names": (
                "history_steps",
                "ridge_alpha",
                "air_temperature_residual_scale",
                "relative_humidity_residual_scale",
                "co2_concentration_residual_scale",
            ),
            "tool_ids": (
                "host.predictor.greenhouse-targetwise-ridge.fit-predict@1",
            ),
            "evaluator_ids": (RUNTIME_EVALUATOR_ID, CELLWISE_EVALUATOR_ID),
        },
        "greenhouse-horizon-targetwise-ridge@1": {
            "version": "greenhouse-horizon-targetwise-ridge/1",
            "parameter_names": (
                "history_steps",
                "ridge_alpha",
                "air_temperature_1h_residual_scale",
                "air_temperature_6h_residual_scale",
                "air_temperature_24h_residual_scale",
                "relative_humidity_1h_residual_scale",
                "relative_humidity_6h_residual_scale",
                "relative_humidity_24h_residual_scale",
                "co2_concentration_1h_residual_scale",
                "co2_concentration_6h_residual_scale",
                "co2_concentration_24h_residual_scale",
            ),
            "tool_ids": (
                "host.predictor.greenhouse-horizon-targetwise-ridge.fit-predict@1",
            ),
            "evaluator_ids": (RUNTIME_EVALUATOR_ID, CELLWISE_EVALUATOR_ID),
        },
        "greenhouse-recipe-ridge@1": {
            "version": "greenhouse-recipe-ridge/1",
            # Deliberately empty. Every tunable — ridge alpha, the anchor and
            # each per-cell residual scale — lives inside the declarative
            # recipe, which the feature policy grammar bounds. Declaring a
            # scalar here would promise an override the config cannot accept,
            # and _validated_catalog_predictor requires this key set to equal
            # the registry's parameter_schemas exactly.
            "parameter_names": (),
            "tool_ids": (
                "host.predictor.greenhouse-recipe-ridge.fit-predict@1",
            ),
            "evaluator_ids": (RUNTIME_EVALUATOR_ID, CELLWISE_EVALUATOR_ID),
        },
    },
    "evaluator": {
        CELLWISE_EVALUATOR_ID: {"version": "greenhouse-cellwise-aligned-forward/1",
            "tool_ids": ("host.evaluator.greenhouse-cellwise-aligned.score@1",)},
        RUNTIME_EVALUATOR_ID: {"version": "greenhouse-runtime-model-selection-forward/1",
                               "tool_ids": ("host.evaluator.greenhouse-runtime-model-selection.score@1",)},
    },
}


def registered_capability_ids(kind: str) -> frozenset[str]:
    """Return the immutable public ID set for one host capability kind."""

    return frozenset(_CAPABILITIES.get(str(kind), {}))


def registered_predictor_evaluator_ids(predictor_id: str) -> tuple[str, ...]:
    """Return the evaluator allowlist for one registered predictor.

    Predictor/evaluator compatibility is a scientific execution boundary. Keep
    its lookup next to the canonical capability registry so compilers, research
    planning, and runtime admission cannot drift into separate maps.
    """

    capability = _CAPABILITIES["predictor"].get(str(predictor_id))
    if capability is None:
        return ()
    return tuple(str(item) for item in capability.get("evaluator_ids", ()))


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    result = dict(value)
    encoded = canonical_json(result)
    if len(encoded) > 24_000:
        raise ValueError(f"{name} is too large")
    return result


def _canonical_partition(value: Any) -> str:
    return str(value or "").strip().casefold()


def _validate_task_partition_boundary(task: TaskManifest) -> None:
    """Reject any task request that expands evolution beyond training data."""

    metadata = task.metadata
    requested: list[str] = []
    for field_name in ("training_partition", "evaluation_partition"):
        raw = metadata.get(field_name)
        if raw is not None:
            requested.append(_canonical_partition(raw))
    raw_allowed = metadata.get("allowed_partitions")
    if raw_allowed is not None:
        if not isinstance(raw_allowed, (list, tuple)):
            raise AlgorithmCompileError(
                "invalid_partition_contract",
                "task allowed_partitions must be an array",
            )
        requested.extend(
            _canonical_partition(item) for item in raw_allowed
        )
    forbidden = sorted({item for item in requested if item in _FORBIDDEN_PARTITIONS})
    if forbidden:
        raise AlgorithmCompileError(
            "forbidden_data_partition",
            "evolution cannot access forbidden partitions: " + ", ".join(forbidden),
        )
    unsupported = sorted(
        {item for item in requested if item not in EVOLUTION_ALLOWED_PARTITIONS}
    )
    if unsupported:
        raise AlgorithmCompileError(
            "unsupported_data_partition",
            "evolution only supports training_fit and training_feedback partitions",
        )


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """A candidate pipeline compiled entirely from host-owned registrations."""

    run_id: str
    generation: int
    proposal_id: str
    algorithm_id: str
    algorithm_version: str
    adapter_id: str
    adapter_version: str
    evaluator_id: str
    evaluator_version: str
    strategy_id: str
    tool_ids: tuple[str, ...]
    parameters: Mapping[str, Any]
    visible_datasets: tuple[str, ...]
    dataset_digest: str | None = None
    split_manifest_digest: str | None = None
    allowed_partitions: tuple[str, ...] = ()
    knowledge_snapshot_digest: str | None = None
    knowledge_mappings: tuple[Mapping[str, Any], ...] = ()
    algorithm_ir: Mapping[str, Any] | None = None
    derived_execution_plan: Mapping[str, Any] | None = None
    spec_version: str = ALGORITHM_SPEC_VERSION
    spec_digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "proposal_id",
            "algorithm_id",
            "algorithm_version",
            "adapter_id",
            "adapter_version",
            "evaluator_id",
            "evaluator_version",
            "strategy_id",
            "spec_version",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("generation must be a non-negative integer")
        if self.spec_version != ALGORITHM_SPEC_VERSION:
            raise ValueError("unsupported algorithm spec version")
        object.__setattr__(
            self, "tool_ids", tuple(_text(item, "tool_id") for item in self.tool_ids)
        )
        if not self.tool_ids:
            raise ValueError("algorithm spec requires registered tools")
        object.__setattr__(
            self, "parameters", _json_mapping(self.parameters, "parameters")
        )
        object.__setattr__(
            self,
            "visible_datasets",
            tuple(_text(item, "visible_dataset") for item in self.visible_datasets),
        )
        if not self.visible_datasets:
            raise ValueError("algorithm spec requires a visible dataset")
        boundary_present = any(
            (
                self.dataset_digest is not None,
                self.split_manifest_digest is not None,
                bool(self.allowed_partitions),
            )
        )
        if boundary_present:
            object.__setattr__(
                self,
                "dataset_digest",
                _text(self.dataset_digest, "dataset_digest"),
            )
            object.__setattr__(
                self,
                "split_manifest_digest",
                _text(self.split_manifest_digest, "split_manifest_digest"),
            )
            partitions = tuple(
                _text(item, "allowed_partition") for item in self.allowed_partitions
            )
            if partitions != EVOLUTION_ALLOWED_PARTITIONS:
                raise ValueError(
                    "algorithm spec may access only training_fit and training_feedback"
                )
            object.__setattr__(self, "allowed_partitions", partitions)
        if self.knowledge_snapshot_digest is not None:
            object.__setattr__(
                self,
                "knowledge_snapshot_digest",
                _text(self.knowledge_snapshot_digest, "knowledge_snapshot_digest"),
            )
        object.__setattr__(
            self,
            "knowledge_mappings",
            tuple(
                _json_mapping(item, "knowledge_mapping")
                for item in self.knowledge_mappings
            ),
        )
        if self.algorithm_ir is not None:
            algorithm_ir = AlgorithmIR.from_dict(self.algorithm_ir)
            expected_evidence = tuple(
                {
                    "knowledge_id": str(item["knowledge_id"]),
                    "decision": str(item["decision"]),
                }
                for item in self.knowledge_mappings
                if item.get("knowledge_id") is not None
                and item.get("decision")
                in {"adopted", "not_selected", "research_only"}
            )
            if (
                algorithm_ir.predictor_id != self.adapter_id
                or algorithm_ir.evaluator_id != self.evaluator_id
                or dict(algorithm_ir.parameters) != dict(self.parameters)
                or algorithm_ir.dataset_digest != self.dataset_digest
                or algorithm_ir.split_manifest_digest != self.split_manifest_digest
                or algorithm_ir.allowed_partitions != self.allowed_partitions
                or algorithm_ir.knowledge_snapshot_digest
                != self.knowledge_snapshot_digest
                or algorithm_ir.evidence_mappings != expected_evidence
            ):
                raise ValueError("algorithm IR does not match the compiled algorithm spec")
            object.__setattr__(self, "algorithm_ir", algorithm_ir.to_dict())
        if self.derived_execution_plan is not None:
            plan = DerivedExecutionPlan.from_dict(self.derived_execution_plan)
            expected_generation = self.generation - 1 if self.generation > 0 else None
            if plan.source_generation != expected_generation:
                raise ValueError(
                    "derived execution plan source does not precede the candidate generation"
                )
            object.__setattr__(self, "derived_execution_plan", plan.to_dict())
        expected = digest(self.identity_dict())
        if self.spec_digest and self.spec_digest != expected:
            raise ValueError("algorithm spec digest mismatch")
        object.__setattr__(self, "spec_digest", expected)

    def identity_dict(self) -> JsonObject:
        result: JsonObject = {
            "spec_version": self.spec_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "proposal_id": self.proposal_id,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "strategy_id": self.strategy_id,
            "tool_ids": list(self.tool_ids),
            "parameters": dict(self.parameters),
            "visible_datasets": list(self.visible_datasets),
            "knowledge_snapshot_digest": self.knowledge_snapshot_digest,
            "knowledge_mappings": [dict(item) for item in self.knowledge_mappings],
            "security_boundary": {
                "external_code_execution": False,
                "dynamic_imports": False,
                "registered_adapters_only": True,
            },
        }
        # Keep historical algorithm-spec digests replayable. New candidate
        # specs always include this host-derived execution input.
        if self.derived_execution_plan is not None:
            result["derived_execution_plan"] = dict(self.derived_execution_plan)
        if self.algorithm_ir is not None:
            result["algorithm_ir"] = dict(self.algorithm_ir)
        if self.dataset_digest is not None:
            result.update(
                {
                    "dataset_digest": self.dataset_digest,
                    "split_manifest_digest": self.split_manifest_digest,
                    "allowed_partitions": list(self.allowed_partitions),
                }
            )
        return result

    def to_dict(self) -> JsonObject:
        return {**self.identity_dict(), "spec_digest": self.spec_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AlgorithmSpec:
        data = dict(value)
        boundary = data.pop("security_boundary", None)
        if not isinstance(boundary, Mapping) or boundary != {
            "external_code_execution": False,
            "dynamic_imports": False,
            "registered_adapters_only": True,
        }:
            raise ValueError("algorithm spec security boundary is invalid")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class AlgorithmAttempt:
    """One append-only compile or debug attempt for a candidate."""

    run_id: str
    generation: int
    proposal_id: str
    candidate_id: str
    phase: str
    attempt: int
    status: str
    algorithm_spec_digest: str | None = None
    algorithm_spec: Mapping[str, Any] | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    failure_code: str | None = None
    public_error: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "proposal_id",
            "candidate_id",
            "phase",
            "status",
            "created_at",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.phase not in {"compile", "debug"}:
            raise ValueError("algorithm attempt phase must be compile or debug")
        if self.status not in {"passed", "failed"}:
            raise ValueError("algorithm attempt status must be passed or failed")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("generation must be a non-negative integer")
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("attempt must be a positive integer")
        if self.algorithm_spec is not None:
            spec = AlgorithmSpec.from_dict(self.algorithm_spec)
            object.__setattr__(self, "algorithm_spec", spec.to_dict())
            if self.algorithm_spec_digest not in {None, spec.spec_digest}:
                raise ValueError("algorithm attempt spec digest mismatch")
            object.__setattr__(self, "algorithm_spec_digest", spec.spec_digest)
        elif self.algorithm_spec_digest is not None:
            object.__setattr__(
                self,
                "algorithm_spec_digest",
                _text(self.algorithm_spec_digest, "algorithm_spec_digest"),
            )
        object.__setattr__(self, "evidence", _json_mapping(self.evidence, "evidence"))
        for name in ("failure_code", "public_error"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name)[:500])
        if self.status == "failed" and (
            self.failure_code is None or self.public_error is None
        ):
            raise ValueError("failed algorithm attempts require failure evidence")
        if self.status == "passed" and self.failure_code is not None:
            raise ValueError("passed algorithm attempts cannot carry a failure code")

    def to_dict(self) -> JsonObject:
        return {
            "run_id": self.run_id,
            "generation": self.generation,
            "proposal_id": self.proposal_id,
            "candidate_id": self.candidate_id,
            "phase": self.phase,
            "attempt": self.attempt,
            "status": self.status,
            "algorithm_spec_digest": self.algorithm_spec_digest,
            "algorithm_spec": dict(self.algorithm_spec)
            if self.algorithm_spec is not None
            else None,
            "evidence": dict(self.evidence),
            "failure_code": self.failure_code,
            "public_error": self.public_error,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AlgorithmAttempt:
        return cls(**dict(value))


def compile_algorithm_spec(
    task: TaskManifest, proposal: Proposal, snapshot: KnowledgeSnapshot | None,
) -> AlgorithmSpec:
    """Use the Genome compiler as the single source of executable semantics."""
    from ..evolution.genome import deep_thaw_json
    from ..core.state import persisted_genome_from_proposal
    from ..evolution.workflow_ir import compile_plugin_behavior
    from .program_registry import current_program_registry

    _validate_task_partition_boundary(task)
    genome = persisted_genome_from_proposal(proposal)
    if genome is None:
        raise AlgorithmCompileError("plugin_genome_missing", "native compilation requires a verified Genome")
    if snapshot is not None and (snapshot.run_id != proposal.run_id or snapshot.generation != proposal.generation):
        raise AlgorithmCompileError("knowledge_scope_mismatch", "knowledge snapshot is outside the candidate scope")
    compiled = compile_plugin_behavior(genome, task, snapshot, current_program_registry())
    template = deep_thaw_json(compiled.algorithm_spec_template)
    if canonical_json(template["parameters"]) != canonical_json(proposal.changes):
        raise AlgorithmCompileError("parameter_contract_mismatch", "proposal parameters differ from the verified Genome")
    return AlgorithmSpec(
        run_id=proposal.run_id, generation=proposal.generation, proposal_id=proposal.proposal_id,
        derived_execution_plan=proposal.metadata.get("derived_execution_plan"), **template,
    )


def debug_algorithm_spec(
    spec: AlgorithmSpec, task: TaskManifest, proposal: Proposal,
    snapshot: KnowledgeSnapshot | None = None,
) -> dict[str, Any]:
    """Recompile the frozen Genome and reject any executable binding drift."""
    rebuilt = compile_algorithm_spec(task, proposal, snapshot)
    if rebuilt.spec_digest != spec.spec_digest:
        raise AlgorithmCompileError("algorithm_spec_binding_mismatch",
                                    "algorithm spec differs from the frozen task and Genome")
    checks = ["verified_plugin_genome", "host_registered_adapter", "registered_tool_chain",
              "frozen_runtime_bindings", "bounded_parameter_contract", "knowledge_capability_mapping",
              "registered_operator_algorithm_ir", "algorithm_ir_digest", "frozen_derived_execution_plan",
              "frozen_dataset_snapshot", "training_partition_allowlist", "no_external_code_execution"]
    return {"checks": checks, "check_count": len(checks), "passed": True}


__all__ = ["ALGORITHM_SPEC_VERSION", "EVOLUTION_ALLOWED_PARTITIONS", "AlgorithmAttempt",
           "AlgorithmCompileError", "AlgorithmSpec", "compile_algorithm_spec", "debug_algorithm_spec",
           "registered_capability_ids", "registered_predictor_evaluator_ids"]
