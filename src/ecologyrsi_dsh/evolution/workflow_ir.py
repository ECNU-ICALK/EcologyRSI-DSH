"""Closed compilation from source genomes to DSH workflow behavior.

No function in this module accepts JavaScript, workflow nodes, shell commands
or model-authored executable text.  Graphs and scripts are resolved only from
the immutable host registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

from ..core.models import TaskManifest
from ..knowledge.algorithm_ir import (
    algorithm_behavior_projection,
    build_registered_algorithm_ir,
    registered_operator_tool_ids,
)
from ..knowledge.algorithms import AlgorithmSpec, EVOLUTION_ALLOWED_PARTITIONS
from ..knowledge.models import KnowledgeSnapshot
from ..knowledge.program_registry import ProgramRegistrySnapshot
from .genome import (
    EcologyEvolutionPluginGenome,
    FrozenJsonObject,
    ProjectedLegacyGenome,
    _domain_digest,
    deep_freeze_json,
    deep_thaw_json,
)


COMPILER_VERSION = "ecology-plugin-behavior-compiler@1"
DEFAULT_COMPILER_SEMANTIC_DIGEST = _domain_digest(
    "ecologyrsi-dsh/plugin-compiler-semantics/1",
    {
        "compiler_version": COMPILER_VERSION,
        "algorithm_behavior_projection": "algorithm_behavior_projection@1",
        "workflow_ir": "compiled-dsh-workflow@1",
        "defaults": "registry-resolved-before-digest@1",
    },
)
SECURITY_SEMANTIC_DIGEST = _domain_digest(
    "ecologyrsi-dsh/plugin-security-semantics/1",
    {
        "registered_graphs_only": True,
        "candidate_tool_policy_monotonic": "narrow-only",
        "reviewer_program_owner": "fitness-security-kernel",
        "arbitrary_code": False,
    },
)
RUNTIME_SEMANTIC_DIGEST = _domain_digest(
    "ecologyrsi-dsh/plugin-runtime-semantics/1",
    {
        "protocol": "dsh_native_plugin_evolution@1",
        "workflow_child_inherits_role_preset": True,
        "reviewer_session": "fresh",
        "context_owner": "dsh-session",
    },
)


_EVALUATOR_VERSIONS = {
    "toy_time_forward@1": "toy-time-forward/3",
    "greenhouse_time_forward@1": "greenhouse-time-forward/5",
    "greenhouse_multihorizon_time_forward@1": (
        "greenhouse-multihorizon-time-forward/4"
    ),
    "greenhouse_multihorizon_time_forward@2": (
        "greenhouse-multihorizon-time-forward/5"
    ),
}
_PREDICTOR_EVALUATORS = {
    "toy-rolling-water@1": {"toy_time_forward@1"},
    "greenhouse-rolling-residual@1": {
        "greenhouse_time_forward@1",
        "greenhouse_multihorizon_time_forward@1",
    },
    "greenhouse-exogenous-ridge@1": {
        "greenhouse_time_forward@1",
        "greenhouse_multihorizon_time_forward@1",
        "greenhouse_multihorizon_time_forward@2",
    },
    "greenhouse-targetwise-ridge@1": {
        "greenhouse_multihorizon_time_forward@1"
    },
    "greenhouse-horizon-targetwise-ridge@1": {
        "greenhouse_multihorizon_time_forward@2"
    },
}


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _frozen_object(value: Mapping[str, Any]) -> FrozenJsonObject:
    frozen = deep_freeze_json(value)
    assert isinstance(frozen, FrozenJsonObject)
    return frozen


def _require_ref(
    registry: ProgramRegistrySnapshot,
    category: str,
    value: Any,
    name: str,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"id", "catalog_digest"}:
        raise ValueError(f"{name} reference is invalid")
    program_id = _text(value.get("id"), f"{name}.id")
    expected = registry.program_ref(category, program_id)
    if dict(value) != expected:
        raise ValueError(f"{name} reference digest does not match the registry")
    return program_id, registry.program(category, program_id)


def _effective_parameters(
    registry: ProgramRegistrySnapshot,
    category: str,
    program_id: str,
    overrides: Any,
) -> dict[str, Any]:
    if not isinstance(overrides, Mapping):
        raise TypeError("program overrides must be an object")
    program = registry.program(category, program_id)
    contracts = program.get("parameters")
    if not isinstance(contracts, Mapping):
        raise ValueError("registered program parameter schema is invalid")
    unknown = set(overrides) - set(contracts)
    if unknown:
        raise ValueError(
            "program overrides contain unregistered parameters: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    effective = {
        name: contract["default"] for name, contract in contracts.items()
    }
    effective.update(dict(overrides))
    for name, value in effective.items():
        contract = contracts[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"program parameter {name} must be numeric and not bool")
        if not math.isfinite(float(value)):
            raise ValueError(f"program parameter {name} must be finite")
        if contract.get("integer") and not isinstance(value, int):
            raise TypeError(f"program parameter {name} must be an integer")
        if not float(contract["minimum"]) <= float(value) <= float(contract["maximum"]):
            raise ValueError(f"program parameter {name} is outside registered bounds")
    return {name: effective[name] for name in sorted(effective)}


@dataclass(frozen=True, slots=True)
class CompiledDshWorkflowSpec:
    _value: FrozenJsonObject

    @property
    def workflow_digest(self) -> str:
        return str(self._value["workflow_digest"])

    @property
    def template_id(self) -> str:
        return str(self._value["template_id"])

    def to_dict(self) -> dict[str, Any]:
        return deep_thaw_json(self._value)


def compile_dsh_workflow_spec(
    template_ref: Mapping[str, Any],
    overrides: Mapping[str, Any],
    role_profiles: Sequence[Mapping[str, Any]],
    registry: ProgramRegistrySnapshot,
) -> CompiledDshWorkflowSpec:
    """Resolve one candidate workflow only from its trusted registry graph."""

    if not isinstance(registry, ProgramRegistrySnapshot):
        raise TypeError("registry must be ProgramRegistrySnapshot")
    template_id, template = _require_ref(
        registry, "workflow_templates", template_ref, "workflow template"
    )
    if template_id != "candidate-sample-execution@1":
        raise ValueError("candidate execution requires its registered workflow class")
    effective = _effective_parameters(
        registry, "workflow_templates", template_id, overrides
    )
    graph = template.get("graph")
    if not isinstance(graph, Mapping):
        raise ValueError("registered workflow graph is invalid")
    allowed_roles = set(str(item) for item in graph["allowed_roles"])
    if isinstance(role_profiles, (str, bytes)) or not isinstance(
        role_profiles, Sequence
    ):
        raise TypeError("role_profiles must be an array")
    compiled_profiles: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for raw_profile in role_profiles:
        if not isinstance(raw_profile, Mapping):
            raise TypeError("role profile must be an object")
        required = {
            "role",
            "preset_id",
            "instruction_template_ref",
            "instruction_parameters",
            "response_schema_id",
            "base_tool_policy_id",
            "enabled_tool_ids",
        }
        if set(raw_profile) != required:
            raise ValueError("role profile has unsupported or missing fields")
        role = _text(raw_profile["role"], "role")
        if role not in allowed_roles or role in seen_roles:
            raise ValueError("role profile is outside or duplicates workflow roles")
        seen_roles.add(role)
        instruction_id, instruction = _require_ref(
            registry,
            "instruction_templates",
            raw_profile["instruction_template_ref"],
            "instruction template",
        )
        instruction_parameters = _effective_parameters(
            registry,
            "instruction_templates",
            instruction_id,
            raw_profile["instruction_parameters"],
        )
        policy_id = _text(raw_profile["base_tool_policy_id"], "base_tool_policy_id")
        base_tools = set(registry.tool_policy(policy_id))
        raw_tools = raw_profile["enabled_tool_ids"]
        if isinstance(raw_tools, (str, bytes)) or not isinstance(raw_tools, Sequence):
            raise TypeError("enabled_tool_ids must be an array")
        enabled_tools = [
            _text(item, "enabled tool id") for item in raw_tools
        ]
        if len(enabled_tools) != len(set(enabled_tools)):
            raise ValueError("enabled tool ids must be unique")
        if not set(enabled_tools).issubset(base_tools):
            raise ValueError("enabled tool ids must be a subset of the base tool policy")
        compiled_profiles.append(
            {
                "role": role,
                "preset_id": _text(raw_profile["preset_id"], "preset_id"),
                "instruction_template_id": instruction_id,
                "instruction_template_digest": registry.program_ref(
                    "instruction_templates", instruction_id
                )["catalog_digest"],
                "instruction_version": instruction["version"],
                "instruction_parameters": instruction_parameters,
                "response_schema_id": _text(
                    raw_profile["response_schema_id"], "response_schema_id"
                ),
                "base_tool_policy_id": policy_id,
                "base_tool_policy_digest": registry.program_ref(
                    "tool_policies", policy_id
                )["catalog_digest"],
                "enabled_tool_ids": sorted(enabled_tools),
            }
        )
    if seen_roles != allowed_roles:
        raise ValueError("role profiles do not cover the registered workflow roles")
    compiled_profiles.sort(key=lambda item: item["role"])
    identity = {
        "schema_version": "ecologyrsi-dsh.compiled-dsh-workflow/1",
        "template_id": template_id,
        "template_digest": registry.program_ref(
            "workflow_templates", template_id
        )["catalog_digest"],
        "template_version": template["version"],
        "effective_parameters": effective,
        "graph": dict(graph),
        "role_profiles": compiled_profiles,
        "security_boundary": {
            "host_registered_graph_only": True,
            "host_registered_scripts_only": True,
            "model_supplied_javascript": False,
        },
    }
    identity["workflow_digest"] = _domain_digest(
        "ecologyrsi-dsh/compiled-dsh-workflow/1", identity
    )
    return CompiledDshWorkflowSpec(_frozen_object(identity))


def _compile_policy_ref(
    registry: ProgramRegistrySnapshot,
    category: str,
    raw_ref: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    if not isinstance(raw_ref, Mapping) or set(raw_ref) != {
        "id",
        "catalog_digest",
        "overrides",
    }:
        raise ValueError(f"{name} reference is invalid")
    program_id, program = _require_ref(
        registry,
        category,
        {"id": raw_ref["id"], "catalog_digest": raw_ref["catalog_digest"]},
        name,
    )
    effective = _effective_parameters(
        registry, category, program_id, raw_ref["overrides"]
    )
    return {
        "id": program_id,
        "catalog_digest": registry.program_ref(category, program_id)[
            "catalog_digest"
        ],
        "version": program["version"],
        "effective_parameters": effective,
    }


def _compile_reproduction_program(
    source: Mapping[str, Any], registry: ProgramRegistrySnapshot
) -> dict[str, Any]:
    template_id, template = _require_ref(
        registry,
        "workflow_templates",
        source["workflow_template_ref"],
        "reproduction workflow",
    )
    if template_id != "research-and-propose@1":
        raise ValueError("reproduction program must use the registered workflow class")
    effective = _effective_parameters(
        registry, "workflow_templates", template_id, source["workflow_overrides"]
    )
    role_refs = source["role_template_refs"]
    if isinstance(role_refs, (str, bytes)) or not isinstance(role_refs, Sequence):
        raise TypeError("reproduction role_template_refs must be an array")
    roles = [str(item) for item in role_refs]
    if set(roles) != {"researcher@1", "candidate-proposer@1"}:
        raise ValueError("reproduction roles do not match the fixed registry graph")
    role_specs = []
    for instruction_id in sorted(roles):
        instruction = registry.program("instruction_templates", instruction_id)
        role_specs.append(
            {
                "instruction_template_id": instruction_id,
                "instruction_template_digest": registry.program_ref(
                    "instruction_templates", instruction_id
                )["catalog_digest"],
                "instruction_version": instruction["version"],
            }
        )
    return {
        "schema_version": "ecologyrsi-dsh.compiled-reproduction-program/1",
        "template_id": template_id,
        "template_digest": registry.program_ref(
            "workflow_templates", template_id
        )["catalog_digest"],
        "template_version": template["version"],
        "effective_parameters": effective,
        "graph": template["graph"],
        "role_specs": role_specs,
        "mutation_mask": {"reproduction_program": False},
    }


def _expected_task_binding(task: TaskManifest) -> dict[str, str]:
    metadata = task.metadata
    fields = {
        "task_manifest_digest": task.digest,
        "dataset_snapshot_set_digest": metadata.get(
            "dataset_snapshot_set_digest", metadata.get("dataset_digest")
        ),
        "split_manifest_digest": metadata.get("split_manifest_digest"),
        "data_protocol_digest": metadata.get("data_protocol_digest"),
        "stage_policy_digest": metadata.get("stage_policy_digest"),
        "evaluator_digest": metadata.get("evaluator_digest"),
        "fitness_profile_digest": metadata.get("fitness_profile_digest"),
        "security_kernel_digest": metadata.get("security_kernel_digest"),
        "selection_reviewer_program_digest": metadata.get(
            "selection_reviewer_program_digest"
        ),
    }
    return {name: _sha(value, name) for name, value in fields.items()}


def _knowledge_mappings(
    snapshot: KnowledgeSnapshot | None, predictor_id: str
) -> tuple[dict[str, str], ...]:
    if snapshot is None:
        return ()
    mappings: list[dict[str, str]] = []
    for card in snapshot.cards:
        capability_ids = tuple(card.capability_ids) or (
            ((card.capability_id,) if card.capability_id else ())
        )
        if predictor_id in capability_ids and card.executable:
            decision = "adopted"
        elif not capability_ids:
            decision = "research_only"
        else:
            decision = "not_selected"
        mappings.append({"knowledge_id": card.knowledge_id, "decision": decision})
    return tuple(mappings)


@dataclass(frozen=True, slots=True)
class CompiledEcologyBehaviorSpec:
    source_genome_digest: str
    source_genome_id: str
    source_lineage: FrozenJsonObject
    algorithm_ir: FrozenJsonObject
    algorithm_behavior: FrozenJsonObject
    feature_training_spec: FrozenJsonObject
    candidate_workflow: CompiledDshWorkflowSpec
    reproduction_spec: FrozenJsonObject
    algorithm_spec_template: FrozenJsonObject
    frozen_source_binding: FrozenJsonObject
    compiler_semantic_digest: str
    registry_catalog_digest: str
    security_semantic_digest: str
    runtime_semantic_digest: str
    compiled_behavior_digest: str

    def behavior_identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ecologyrsi-dsh.compiled-plugin-behavior/1",
            "algorithm_behavior": deep_thaw_json(self.algorithm_behavior),
            "feature_training_spec": deep_thaw_json(self.feature_training_spec),
            "candidate_workflow": self.candidate_workflow.to_dict(),
            "reproduction_spec": deep_thaw_json(self.reproduction_spec),
            "compiler_semantic_digest": self.compiler_semantic_digest,
            "registry_catalog_digest": self.registry_catalog_digest,
            "security_semantic_digest": self.security_semantic_digest,
            "runtime_semantic_digest": self.runtime_semantic_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.behavior_identity_dict(),
            "source_genome_digest": self.source_genome_digest,
            "source_genome_id": self.source_genome_id,
            "source_lineage": deep_thaw_json(self.source_lineage),
            "algorithm_ir": deep_thaw_json(self.algorithm_ir),
            "algorithm_spec_template": deep_thaw_json(self.algorithm_spec_template),
            "frozen_source_binding": deep_thaw_json(self.frozen_source_binding),
            "compiled_behavior_digest": self.compiled_behavior_digest,
        }


def compile_plugin_behavior(
    genome: EcologyEvolutionPluginGenome,
    task: TaskManifest,
    knowledge_snapshot: KnowledgeSnapshot | None,
    registry: ProgramRegistrySnapshot,
    *,
    compiler_semantic_digest: str = DEFAULT_COMPILER_SEMANTIC_DIGEST,
) -> CompiledEcologyBehaviorSpec:
    """Resolve a source genome into one closed, instance-free behavior."""

    if not isinstance(genome, EcologyEvolutionPluginGenome):
        raise TypeError("compile_plugin_behavior requires a materialized source genome")
    if not isinstance(task, TaskManifest):
        raise TypeError("task must be TaskManifest")
    if knowledge_snapshot is not None and not isinstance(
        knowledge_snapshot, KnowledgeSnapshot
    ):
        raise TypeError("knowledge_snapshot must be KnowledgeSnapshot or null")
    if not isinstance(registry, ProgramRegistrySnapshot):
        raise TypeError("registry must be ProgramRegistrySnapshot")
    compiler_semantic_digest = _sha(
        compiler_semantic_digest, "compiler_semantic_digest"
    )
    genome_data = genome.to_dict()
    runtime_binding = genome_data["runtime_binding"]
    if runtime_binding["registry_catalog_digest"] != registry.catalog_digest:
        raise ValueError("genome registry catalog binding mismatch")
    expected_binding = _expected_task_binding(task)
    if genome_data["frozen_contract_refs"] != expected_binding:
        raise ValueError("genome task/data/stage/security binding mismatch")

    scientific = genome_data["scientific_program"]
    predictor_id, predictor = _require_ref(
        registry, "predictors", scientific["predictor_ref"], "predictor"
    )
    parameters = _effective_parameters(
        registry,
        "predictors",
        predictor_id,
        scientific["parameter_overrides"],
    )
    evaluator_id = _text(task.metadata.get("evaluator_id"), "evaluator_id")
    if evaluator_id not in _PREDICTOR_EVALUATORS.get(predictor_id, set()):
        raise ValueError("predictor and evaluator bindings are incompatible")
    if evaluator_id not in _EVALUATOR_VERSIONS:
        raise ValueError("evaluator is not registered")
    mappings = _knowledge_mappings(knowledge_snapshot, predictor_id)
    algorithm_ir = build_registered_algorithm_ir(
        predictor_id=predictor_id,
        evaluator_id=evaluator_id,
        parameters=parameters,
        dataset_digest=_text(task.metadata.get("dataset_digest"), "dataset_digest"),
        split_manifest_digest=expected_binding["split_manifest_digest"],
        allowed_partitions=EVOLUTION_ALLOWED_PARTITIONS,
        evidence_mappings=mappings,
        knowledge_snapshot_digest=(
            knowledge_snapshot.snapshot_digest if knowledge_snapshot is not None else None
        ),
    )
    behavior_projection = algorithm_behavior_projection(algorithm_ir)
    feature_training = {
        "schema_version": "ecologyrsi-dsh.compiled-feature-training/1",
        "feature_policy": _compile_policy_ref(
            registry,
            "feature_policies",
            scientific["feature_policy_ref"],
            "feature policy",
        ),
        "fit_policy": _compile_policy_ref(
            registry,
            "fit_policies",
            scientific["fit_policy_ref"],
            "fit policy",
        ),
        "uncertainty_policy": _compile_policy_ref(
            registry,
            "uncertainty_policies",
            scientific["uncertainty_policy_ref"],
            "uncertainty policy",
        ),
        "allowed_partitions": list(EVOLUTION_ALLOWED_PARTITIONS),
    }
    agent = genome_data["agent_program"]
    execution = agent["candidate_execution_program"]
    candidate_workflow = compile_dsh_workflow_spec(
        execution["workflow_template_ref"],
        execution["workflow_overrides"],
        execution["role_profiles"],
        registry,
    )
    reproduction = _compile_reproduction_program(
        agent["reproduction_program"], registry
    )
    template = {
        "algorithm_id": f"{predictor_id}+{evaluator_id}",
        "algorithm_version": (
            f"registered-pipeline/{_domain_digest('ecologyrsi-dsh/algorithm-version/1', [predictor['version'], _EVALUATOR_VERSIONS[evaluator_id]])[:12]}"
        ),
        "adapter_id": predictor_id,
        "adapter_version": predictor["version"],
        "evaluator_id": evaluator_id,
        "evaluator_version": _EVALUATOR_VERSIONS[evaluator_id],
        "strategy_id": str(task.metadata.get("strategy_id") or "autonomous_model@1"),
        "tool_ids": list(registered_operator_tool_ids(predictor_id)),
        "parameters": parameters,
        "visible_datasets": list(task.visible_datasets),
        "dataset_digest": task.metadata["dataset_digest"],
        "split_manifest_digest": expected_binding["split_manifest_digest"],
        "allowed_partitions": list(EVOLUTION_ALLOWED_PARTITIONS),
        "knowledge_snapshot_digest": (
            knowledge_snapshot.snapshot_digest if knowledge_snapshot is not None else None
        ),
        "knowledge_mappings": list(mappings),
        "algorithm_ir": algorithm_ir.to_dict(),
    }
    source_binding = {
        **expected_binding,
        "required_capability_digest": runtime_binding["required_capability_digest"],
        "resolved_policy_route_digest": runtime_binding[
            "resolved_policy_route_digest"
        ],
        "resolved_review_route_digest": runtime_binding[
            "resolved_review_route_digest"
        ],
        "protocol": runtime_binding["protocol"],
    }
    provisional = CompiledEcologyBehaviorSpec(
        source_genome_digest=genome.genome_digest,
        source_genome_id=genome.genome_id,
        source_lineage=_frozen_object(genome_data["lineage"]),
        algorithm_ir=_frozen_object(algorithm_ir.to_dict()),
        algorithm_behavior=_frozen_object(behavior_projection),
        feature_training_spec=_frozen_object(feature_training),
        candidate_workflow=candidate_workflow,
        reproduction_spec=_frozen_object(reproduction),
        algorithm_spec_template=_frozen_object(template),
        frozen_source_binding=_frozen_object(source_binding),
        compiler_semantic_digest=compiler_semantic_digest,
        registry_catalog_digest=registry.catalog_digest,
        security_semantic_digest=SECURITY_SEMANTIC_DIGEST,
        runtime_semantic_digest=RUNTIME_SEMANTIC_DIGEST,
        compiled_behavior_digest="",
    )
    compiled_digest = _domain_digest(
        "ecologyrsi-dsh/compiled-plugin-behavior/1",
        provisional.behavior_identity_dict(),
    )
    return replace(provisional, compiled_behavior_digest=compiled_digest)


@dataclass(frozen=True, slots=True)
class CompilationInstanceContext:
    run_id: str
    proposal_id: str
    candidate_id: str
    generation: int
    slot_index: int
    task_manifest_digest: str
    dataset_snapshot_set_digest: str
    split_manifest_digest: str
    data_protocol_digest: str
    stage_policy_digest: str
    evaluator_digest: str
    evaluation_cohort_digest: str
    required_capability_digest: str
    resolved_policy_route_config_digest: str
    resolved_review_route_config_digest: str
    preset_content_digest: str
    standing_tool_surface_digest: str
    security_kernel_digest: str

    def __post_init__(self) -> None:
        for name in ("run_id", "proposal_id", "candidate_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("generation", "slot_index"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in (
            "task_manifest_digest",
            "dataset_snapshot_set_digest",
            "split_manifest_digest",
            "data_protocol_digest",
            "stage_policy_digest",
            "evaluator_digest",
            "evaluation_cohort_digest",
            "required_capability_digest",
            "resolved_policy_route_config_digest",
            "resolved_review_route_config_digest",
            "preset_content_digest",
            "standing_tool_surface_digest",
            "security_kernel_digest",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompilationInstanceContext":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class BoundEcologyPluginSpec:
    compiled_behavior_digest: str
    source_genome_digest: str
    runtime_execution_digest: str
    phenotype_instance_digest: str
    instance_context: FrozenJsonObject
    algorithm_spec: FrozenJsonObject

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ecologyrsi-dsh.bound-plugin-spec/1",
            "compiled_behavior_digest": self.compiled_behavior_digest,
            "source_genome_digest": self.source_genome_digest,
            "runtime_execution_digest": self.runtime_execution_digest,
            "phenotype_instance_digest": self.phenotype_instance_digest,
            "instance_context": deep_thaw_json(self.instance_context),
            "algorithm_spec": deep_thaw_json(self.algorithm_spec),
        }


def bind_phenotype_instance(
    compiled_behavior: CompiledEcologyBehaviorSpec,
    context: CompilationInstanceContext,
) -> BoundEcologyPluginSpec:
    if not isinstance(compiled_behavior, CompiledEcologyBehaviorSpec):
        raise TypeError("compiled_behavior must be CompiledEcologyBehaviorSpec")
    if not isinstance(context, CompilationInstanceContext):
        raise TypeError("context must be CompilationInstanceContext")
    binding = deep_thaw_json(compiled_behavior.frozen_source_binding)
    comparisons = {
        "task_manifest_digest": context.task_manifest_digest,
        "dataset_snapshot_set_digest": context.dataset_snapshot_set_digest,
        "split_manifest_digest": context.split_manifest_digest,
        "data_protocol_digest": context.data_protocol_digest,
        "stage_policy_digest": context.stage_policy_digest,
        "evaluator_digest": context.evaluator_digest,
        "security_kernel_digest": context.security_kernel_digest,
        "required_capability_digest": context.required_capability_digest,
    }
    for name, actual in comparisons.items():
        if binding[name] != actual:
            raise ValueError(f"phenotype instance binding mismatch: {name}")
    lineage = deep_thaw_json(compiled_behavior.source_lineage)
    if lineage["origin_kind"] == "bounded_mutation" and (
        lineage["generation"] != context.generation
        or lineage["slot_index"] != context.slot_index
    ):
        raise ValueError("phenotype instance binding mismatch: lineage coordinates")
    template = deep_thaw_json(compiled_behavior.algorithm_spec_template)
    algorithm_spec = AlgorithmSpec(
        run_id=context.run_id,
        generation=context.generation,
        proposal_id=context.proposal_id,
        **template,
    )
    runtime_identity = {
        "schema_version": "ecologyrsi-dsh.runtime-execution/1",
        "compiled_behavior_digest": compiled_behavior.compiled_behavior_digest,
        "compiler_semantic_digest": compiled_behavior.compiler_semantic_digest,
        "registry_catalog_digest": compiled_behavior.registry_catalog_digest,
        "security_semantic_digest": compiled_behavior.security_semantic_digest,
        "runtime_semantic_digest": compiled_behavior.runtime_semantic_digest,
        "source_binding": binding,
        "resolved_policy_route_config_digest": context.resolved_policy_route_config_digest,
        "resolved_review_route_config_digest": context.resolved_review_route_config_digest,
        "preset_content_digest": context.preset_content_digest,
        "standing_tool_surface_digest": context.standing_tool_surface_digest,
    }
    runtime_execution_digest = _domain_digest(
        "ecologyrsi-dsh/runtime-execution/1", runtime_identity
    )
    phenotype_identity = {
        "schema_version": "ecologyrsi-dsh.phenotype-instance/1",
        "compiled_behavior_digest": compiled_behavior.compiled_behavior_digest,
        "source_genome_digest": compiled_behavior.source_genome_digest,
        "runtime_execution_digest": runtime_execution_digest,
        "instance_context": context.to_dict(),
        "algorithm_spec_digest": algorithm_spec.spec_digest,
    }
    phenotype_instance_digest = _domain_digest(
        "ecologyrsi-dsh/phenotype-instance/1", phenotype_identity
    )
    return BoundEcologyPluginSpec(
        compiled_behavior_digest=compiled_behavior.compiled_behavior_digest,
        source_genome_digest=compiled_behavior.source_genome_digest,
        runtime_execution_digest=runtime_execution_digest,
        phenotype_instance_digest=phenotype_instance_digest,
        instance_context=_frozen_object(context.to_dict()),
        algorithm_spec=_frozen_object(algorithm_spec.to_dict()),
    )


def compile_legacy_algorithm_ir(projected: ProjectedLegacyGenome) -> dict[str, Any]:
    """Compatibility wrapper: return the frozen legacy compiler result exactly."""

    if not isinstance(projected, ProjectedLegacyGenome):
        raise TypeError("legacy compilation requires a projected legacy genome")
    return deep_thaw_json(projected.legacy_algorithm_ir)


__all__ = [
    "BoundEcologyPluginSpec",
    "CompilationInstanceContext",
    "CompiledDshWorkflowSpec",
    "CompiledEcologyBehaviorSpec",
    "DEFAULT_COMPILER_SEMANTIC_DIGEST",
    "bind_phenotype_instance",
    "compile_dsh_workflow_spec",
    "compile_legacy_algorithm_ir",
    "compile_plugin_behavior",
]
