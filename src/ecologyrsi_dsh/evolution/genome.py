"""Canonical, immutable source genomes for the DSH-native ecology plugin.

The genome deliberately contains registered source references rather than an
``AlgorithmIR`` or an executable workflow graph.  Compilation is a separate
boundary; this module owns only canonical identity, lineage and bounded source
mutation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import TYPE_CHECKING, Any, TypeAlias
import unicodedata

from ..core.models import Proposal, TaskManifest, digest

if TYPE_CHECKING:
    from ..knowledge.models import KnowledgeSnapshot


GENOME_SCHEMA_VERSION = "ecologyrsi-dsh.plugin-genome/1"
GENOME_CANONICAL_VERSION = "plugin-genome-canonical-json@1"
SEED_TEMPLATE_SCHEMA_VERSION = "ecologyrsi-dsh.seed-genome-template/1"
MUTATION_SCHEMA_VERSION = "ecologyrsi-dsh.genome-mutation/1"
LEGACY_PROJECTION_SCHEMA_VERSION = "ecologyrsi-dsh.legacy-genome-projection/1"
LEGACY_ADAPTER_VERSION = "legacy-proposal-genome-adapter@0.2.2"
MATERIALIZER_VERSION = "seed-genome-materializer@1"
MIGRATION_MATERIALIZER_VERSION = "legacy-genome-migration@1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROGRAM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*(?:@[A-Za-z0-9][A-Za-z0-9._-]*)?$")
_PRESET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True, slots=True)
class FrozenJsonObject(Mapping[str, "FrozenJson"]):
    """Tuple-backed immutable JSON object with deterministic key order."""

    _items: tuple[tuple[str, "FrozenJson"], ...]

    def __getitem__(self, key: str) -> "FrozenJson":
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        return deep_thaw_json(self) == deep_thaw_json(other)

    def __hash__(self) -> int:
        return hash(self._items)


@dataclass(frozen=True, slots=True)
class FrozenJsonArray(Sequence["FrozenJson"]):
    """Tuple-backed immutable JSON array."""

    _items: tuple["FrozenJson", ...]

    def __getitem__(self, index: int | slice) -> "FrozenJson | FrozenJsonArray":
        value = self._items[index]
        if isinstance(index, slice):
            return FrozenJsonArray(value)
        return value

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator["FrozenJson"]:
        return iter(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (str, bytes)) or not isinstance(other, Sequence):
            return False
        return deep_thaw_json(self) == deep_thaw_json(other)

    def __hash__(self) -> int:
        return hash(self._items)


FrozenJson: TypeAlias = (
    None | bool | int | float | str | FrozenJsonObject | FrozenJsonArray
)


def deep_freeze_json(value: Any) -> FrozenJson:
    """Copy, normalize and recursively freeze a JSON-compatible value."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: dict[str, FrozenJson] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise TypeError("JSON object keys must be strings")
            key = unicodedata.normalize("NFC", raw_key)
            if key in normalized:
                raise ValueError("JSON object keys must be unique after NFC normalization")
            normalized[key] = deep_freeze_json(raw_value)
        return FrozenJsonObject(tuple((key, normalized[key]) for key in sorted(normalized)))
    if isinstance(value, (list, tuple)):
        return FrozenJsonArray(tuple(deep_freeze_json(item) for item in value))
    raise TypeError("value must contain only JSON-compatible types")


def deep_thaw_json(value: FrozenJson | Any) -> Any:
    """Return a detached mutable JSON copy of a frozen value."""

    if isinstance(value, FrozenJsonObject):
        return {key: deep_thaw_json(item) for key, item in value._items}
    if isinstance(value, FrozenJsonArray):
        return [deep_thaw_json(item) for item in value._items]
    if isinstance(value, Mapping):
        return {str(key): deep_thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [deep_thaw_json(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    frozen = deep_freeze_json(value)
    return json.dumps(
        deep_thaw_json(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _domain_digest(domain: str, value: Any) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + _canonical_bytes(value)).hexdigest()


def _text(value: Any, name: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = unicodedata.normalize("NFC", value).strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(result) is None:
        raise ValueError(f"{name} has an invalid format")
    return result


def _digest_text(value: Any, name: str) -> str:
    result = _text(value, name)
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _finite_number(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number and not bool")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


def _exact_mapping(value: Any, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    result = dict(value)
    unknown = set(result) - allowed
    missing = allowed - set(result)
    if unknown:
        raise ValueError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    return result


def _unique_sorted_texts(
    value: Any,
    name: str,
    *,
    pattern: re.Pattern[str] | None = None,
    maximum: int = 64,
) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    if len(value) > maximum:
        raise ValueError(f"{name} contains too many items")
    items = [_text(item, f"{name} item", pattern=pattern) for item in value]
    if len(items) != len(set(items)):
        raise ValueError(f"{name} items must be unique")
    return sorted(items)


def _program_ref(value: Any, name: str) -> dict[str, str]:
    raw = _exact_mapping(value, name, {"id", "catalog_digest"})
    return {
        "id": _text(raw["id"], f"{name}.id", pattern=_PROGRAM_ID_RE),
        "catalog_digest": _digest_text(
            raw["catalog_digest"], f"{name}.catalog_digest"
        ),
    }


def _bounded_overrides(value: Any, name: str) -> dict[str, int | float | str | bool]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if len(value) > 64:
        raise ValueError(f"{name} contains too many fields")
    result: dict[str, int | float | str | bool] = {}
    for raw_key, raw_value in value.items():
        key = _text(raw_key, f"{name} key", pattern=_PROGRAM_ID_RE)
        if isinstance(raw_value, bool):
            result[key] = raw_value
        elif isinstance(raw_value, (int, float)):
            result[key] = _finite_number(raw_value, f"{name}.{key}")
        elif isinstance(raw_value, str):
            if len(raw_value) > 500:
                raise ValueError(f"{name}.{key} is too long")
            result[key] = unicodedata.normalize("NFC", raw_value)
        else:
            raise TypeError(f"{name}.{key} must be a bounded scalar")
    return {key: result[key] for key in sorted(result)}


def _scientific_program(value: Any) -> dict[str, Any]:
    raw = _exact_mapping(
        value,
        "scientific_program",
        {
            "predictor_ref",
            "parameter_overrides",
            "feature_policy_ref",
            "fit_policy_ref",
            "uncertainty_policy_ref",
        },
    )
    parameters = _bounded_overrides(
        raw["parameter_overrides"], "parameter_overrides"
    )
    parameters = {
        name: _finite_number(item, f"parameter_overrides.{name}")
        for name, item in parameters.items()
    }
    result: dict[str, Any] = {
        "predictor_ref": _program_ref(raw["predictor_ref"], "predictor_ref"),
        "parameter_overrides": parameters,
    }
    for field_name in (
        "feature_policy_ref",
        "fit_policy_ref",
        "uncertainty_policy_ref",
    ):
        ref = _exact_mapping(raw[field_name], field_name, {"id", "catalog_digest", "overrides"})
        result[field_name] = {
            **_program_ref(
                {"id": ref["id"], "catalog_digest": ref["catalog_digest"]},
                field_name,
            ),
            "overrides": _bounded_overrides(ref["overrides"], f"{field_name}.overrides"),
        }
    return result


def _role_profile(value: Any) -> dict[str, Any]:
    raw = _exact_mapping(
        value,
        "role_profile",
        {
            "role",
            "preset_id",
            "instruction_template_ref",
            "instruction_parameters",
            "response_schema_id",
            "base_tool_policy_id",
            "enabled_tool_ids",
        },
    )
    role = _text(raw["role"], "role_profile.role", pattern=_PRESET_ID_RE)
    if role not in {"sample-planner", "sample-repair"}:
        raise ValueError("candidate role_profile cannot select reproduction or reviewer roles")
    instruction_parameters = _bounded_overrides(
        raw["instruction_parameters"], "instruction_parameters"
    )
    allowed_instruction_parameters = (
        {"confidence_threshold"} if role == "sample-planner" else set()
    )
    unsupported_parameters = set(instruction_parameters) - allowed_instruction_parameters
    if unsupported_parameters:
        raise ValueError(
            "instruction_parameters has unsupported fields: "
            + ", ".join(sorted(unsupported_parameters))
        )
    if "confidence_threshold" in instruction_parameters:
        threshold = _finite_number(
            instruction_parameters["confidence_threshold"],
            "instruction_parameters.confidence_threshold",
        )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("instruction confidence_threshold must be between 0 and 1")
        instruction_parameters["confidence_threshold"] = threshold
    return {
        "role": role,
        "preset_id": _text(raw["preset_id"], "role_profile.preset_id", pattern=_PRESET_ID_RE),
        "instruction_template_ref": _program_ref(
            raw["instruction_template_ref"], "instruction_template_ref"
        ),
        "instruction_parameters": instruction_parameters,
        "response_schema_id": _text(
            raw["response_schema_id"], "response_schema_id", pattern=_PROGRAM_ID_RE
        ),
        "base_tool_policy_id": _text(
            raw["base_tool_policy_id"], "base_tool_policy_id", pattern=_PROGRAM_ID_RE
        ),
        "enabled_tool_ids": _unique_sorted_texts(
            raw["enabled_tool_ids"], "enabled_tool_ids", pattern=_PROGRAM_ID_RE
        ),
    }


def _agent_program(value: Any) -> dict[str, Any]:
    raw = _exact_mapping(
        value,
        "agent_program",
        {"candidate_execution_program", "reproduction_program"},
    )
    execution = _exact_mapping(
        raw["candidate_execution_program"],
        "candidate_execution_program",
        {"workflow_template_ref", "workflow_overrides", "role_profiles"},
    )
    profiles_raw = execution["role_profiles"]
    if isinstance(profiles_raw, (str, bytes)) or not isinstance(profiles_raw, Sequence):
        raise TypeError("role_profiles must be an array")
    profiles = [_role_profile(item) for item in profiles_raw]
    roles = [item["role"] for item in profiles]
    if not profiles or len(roles) != len(set(roles)):
        raise ValueError("role_profiles roles must be non-empty and unique")
    profiles.sort(key=lambda item: item["role"])

    reproduction = _exact_mapping(
        raw["reproduction_program"],
        "reproduction_program",
        {"workflow_template_ref", "workflow_overrides", "role_template_refs"},
    )
    role_template_refs = _unique_sorted_texts(
        reproduction["role_template_refs"],
        "reproduction_program.role_template_refs",
        pattern=_PROGRAM_ID_RE,
    )
    allowed_reproduction = {"researcher@1", "candidate-proposer@1"}
    if set(role_template_refs) != allowed_reproduction:
        raise ValueError("reproduction_program must use the fixed researcher/proposer roles")
    return {
        "candidate_execution_program": {
            "workflow_template_ref": _program_ref(
                execution["workflow_template_ref"], "candidate workflow_template_ref"
            ),
            "workflow_overrides": _bounded_overrides(
                execution["workflow_overrides"], "candidate workflow_overrides"
            ),
            "role_profiles": profiles,
        },
        "reproduction_program": {
            "workflow_template_ref": _program_ref(
                reproduction["workflow_template_ref"],
                "reproduction workflow_template_ref",
            ),
            "workflow_overrides": _bounded_overrides(
                reproduction["workflow_overrides"], "reproduction workflow_overrides"
            ),
            "role_template_refs": role_template_refs,
        },
    }


@dataclass(frozen=True, slots=True)
class SeedGenomeTemplate:
    """Task-independent registered source used to materialize a run seed."""

    _value: FrozenJsonObject

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SeedGenomeTemplate":
        raw = dict(value)
        allowed = {
            "schema_version",
            "template_id",
            "scientific_program",
            "agent_program",
            "evidence_refs",
            "template_digest",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                "seed template has unsupported or run-specific fields: "
                + ", ".join(sorted(unknown))
            )
        missing = allowed - {"template_digest"} - set(raw)
        if missing:
            raise ValueError("seed template is missing fields: " + ", ".join(sorted(missing)))
        normalized = {
            "schema_version": _text(raw["schema_version"], "seed schema_version"),
            "template_id": _text(raw["template_id"], "template_id", pattern=_PROGRAM_ID_RE),
            "scientific_program": _scientific_program(raw["scientific_program"]),
            "agent_program": _agent_program(raw["agent_program"]),
            "evidence_refs": _unique_sorted_texts(
                raw["evidence_refs"], "seed evidence_refs", pattern=_PROGRAM_ID_RE
            ),
        }
        if normalized["schema_version"] != SEED_TEMPLATE_SCHEMA_VERSION:
            raise ValueError("unsupported seed template schema_version")
        expected = _domain_digest("ecologyrsi-dsh/seed-genome-template/1", normalized)
        supplied = raw.get("template_digest")
        if supplied is not None and _digest_text(supplied, "template_digest") != expected:
            raise ValueError("seed template digest mismatch")
        normalized["template_digest"] = expected
        frozen = deep_freeze_json(normalized)
        assert isinstance(frozen, FrozenJsonObject)
        return cls(frozen)

    @property
    def template_id(self) -> str:
        return str(self._value["template_id"])

    @property
    def template_digest(self) -> str:
        return str(self._value["template_digest"])

    def to_dict(self) -> dict[str, Any]:
        return deep_thaw_json(self._value)


@dataclass(frozen=True, slots=True)
class GenomeBindingSubset:
    runtime_binding: FrozenJsonObject
    frozen_contract_refs: FrozenJsonObject

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_binding": deep_thaw_json(self.runtime_binding),
            "frozen_contract_refs": deep_thaw_json(self.frozen_contract_refs),
        }


@dataclass(frozen=True, slots=True)
class FrozenRunInitialization:
    """Complete run initialization; only ``bindings`` enter source identity."""

    run_id: str
    task_manifest_digest: str
    dataset_snapshot_set_digest: str
    split_manifest_digest: str
    data_protocol_digest: str
    stage_policy_digest: str
    evaluator_digest: str
    fitness_profile_digest: str
    security_kernel_digest: str
    selection_reviewer_program_digest: str
    protocol: str
    required_capability_digest: str
    resolved_policy_route_digest: str
    resolved_review_route_digest: str
    registry_catalog_digest: str
    compiler_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(self, "protocol", _text(self.protocol, "protocol", pattern=_PROGRAM_ID_RE))
        if self.protocol != "dsh_native_plugin_evolution@1":
            raise ValueError("unsupported runtime protocol")
        for name in (
            "task_manifest_digest",
            "dataset_snapshot_set_digest",
            "split_manifest_digest",
            "data_protocol_digest",
            "stage_policy_digest",
            "evaluator_digest",
            "fitness_profile_digest",
            "security_kernel_digest",
            "selection_reviewer_program_digest",
            "required_capability_digest",
            "resolved_policy_route_digest",
            "resolved_review_route_digest",
            "registry_catalog_digest",
            "compiler_digest",
        ):
            object.__setattr__(self, name, _digest_text(getattr(self, name), name))

    @property
    def bindings(self) -> GenomeBindingSubset:
        runtime = deep_freeze_json(
            {
                "protocol": self.protocol,
                "required_capability_digest": self.required_capability_digest,
                "resolved_policy_route_digest": self.resolved_policy_route_digest,
                "resolved_review_route_digest": self.resolved_review_route_digest,
                "registry_catalog_digest": self.registry_catalog_digest,
            }
        )
        contracts = deep_freeze_json(
            {
                "task_manifest_digest": self.task_manifest_digest,
                "dataset_snapshot_set_digest": self.dataset_snapshot_set_digest,
                "split_manifest_digest": self.split_manifest_digest,
                "data_protocol_digest": self.data_protocol_digest,
                "stage_policy_digest": self.stage_policy_digest,
                "evaluator_digest": self.evaluator_digest,
                "fitness_profile_digest": self.fitness_profile_digest,
                "security_kernel_digest": self.security_kernel_digest,
                "selection_reviewer_program_digest": self.selection_reviewer_program_digest,
            }
        )
        assert isinstance(runtime, FrozenJsonObject)
        assert isinstance(contracts, FrozenJsonObject)
        return GenomeBindingSubset(runtime, contracts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            **self.bindings.to_dict(),
            "compiler_digest": self.compiler_digest,
        }


def _lineage(value: Any) -> dict[str, Any]:
    fields = {
        "origin_kind",
        "parent_candidate_id",
        "parent_genome_digest",
        "generation",
        "slot_index",
        "slot_seed",
        "generation_batch_digest",
        "mutation_budget_digest",
        "mutation_operator_id",
        "mutation_digest",
        "source_research_iteration_digest",
        "source_knowledge_snapshot_digest",
        "migration_source",
    }
    raw = _exact_mapping(value, "lineage", fields)
    origin = _text(raw["origin_kind"], "lineage.origin_kind")
    if origin not in {"seed_catalog", "legacy_migration", "bounded_mutation"}:
        raise ValueError("unsupported lineage origin_kind")
    result: dict[str, Any] = {"origin_kind": origin}
    nullable_texts = {"parent_candidate_id"}
    nullable_digests = {
        "parent_genome_digest",
        "generation_batch_digest",
        "mutation_budget_digest",
        "mutation_digest",
        "source_research_iteration_digest",
        "source_knowledge_snapshot_digest",
    }
    for name in nullable_texts:
        result[name] = None if raw[name] is None else _text(raw[name], f"lineage.{name}")
    for name in nullable_digests:
        result[name] = None if raw[name] is None else _digest_text(raw[name], f"lineage.{name}")
    result["mutation_operator_id"] = (
        None
        if raw["mutation_operator_id"] is None
        else _text(raw["mutation_operator_id"], "lineage.mutation_operator_id", pattern=_PROGRAM_ID_RE)
    )
    for name in ("generation", "slot_index", "slot_seed"):
        result[name] = None if raw[name] is None else _integer(raw[name], f"lineage.{name}", minimum=0)
    migration = raw["migration_source"]
    if migration is not None:
        if not isinstance(migration, Mapping):
            raise TypeError("lineage.migration_source must be an object or null")
        migration = deep_thaw_json(deep_freeze_json(migration))
    result["migration_source"] = migration

    if origin in {"seed_catalog", "legacy_migration"}:
        for name in (
            "parent_candidate_id",
            "parent_genome_digest",
            "generation",
            "slot_index",
            "slot_seed",
            "generation_batch_digest",
            "mutation_budget_digest",
            "mutation_operator_id",
            "mutation_digest",
            "source_research_iteration_digest",
            "source_knowledge_snapshot_digest",
        ):
            if result[name] is not None:
                raise ValueError(f"{origin} lineage cannot contain {name}")
        if origin == "seed_catalog" and migration is not None:
            raise ValueError("seed_catalog lineage cannot contain migration_source")
        if origin == "legacy_migration" and migration is None:
            raise ValueError("legacy_migration lineage requires migration_source")
    else:
        required = (
            "parent_genome_digest",
            "generation",
            "slot_index",
            "slot_seed",
            "generation_batch_digest",
            "mutation_budget_digest",
            "mutation_operator_id",
            "mutation_digest",
            "source_research_iteration_digest",
            "source_knowledge_snapshot_digest",
        )
        if any(result[name] is None for name in required):
            raise ValueError("bounded_mutation lineage is incomplete")
        if result["generation"] == 0 and result["parent_candidate_id"] is not None:
            raise ValueError("first generation parent_candidate_id must be null")
        if result["generation"] > 0 and result["parent_candidate_id"] is None:
            raise ValueError("later generations require parent_candidate_id")
        if migration is not None:
            raise ValueError("bounded_mutation lineage cannot contain migration_source")
    return result


def _runtime_binding(value: Any) -> dict[str, str]:
    fields = {
        "protocol",
        "required_capability_digest",
        "resolved_policy_route_digest",
        "resolved_review_route_digest",
        "registry_catalog_digest",
    }
    raw = _exact_mapping(value, "runtime_binding", fields)
    result = {"protocol": _text(raw["protocol"], "runtime_binding.protocol", pattern=_PROGRAM_ID_RE)}
    if result["protocol"] != "dsh_native_plugin_evolution@1":
        raise ValueError("unsupported runtime binding protocol")
    for name in fields - {"protocol"}:
        result[name] = _digest_text(raw[name], f"runtime_binding.{name}")
    return result


def _frozen_contract_refs(value: Any) -> dict[str, str]:
    fields = {
        "task_manifest_digest",
        "dataset_snapshot_set_digest",
        "split_manifest_digest",
        "data_protocol_digest",
        "stage_policy_digest",
        "evaluator_digest",
        "fitness_profile_digest",
        "security_kernel_digest",
        "selection_reviewer_program_digest",
    }
    raw = _exact_mapping(value, "frozen_contract_refs", fields)
    return {name: _digest_text(raw[name], f"frozen_contract_refs.{name}") for name in sorted(fields)}


@dataclass(frozen=True, slots=True)
class EcologyEvolutionPluginGenome:
    """A fully materialized, recursively immutable source genome."""

    _value: FrozenJsonObject

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EcologyEvolutionPluginGenome":
        if not isinstance(value, Mapping):
            raise TypeError("plugin genome must be an object")
        raw = dict(value)
        allowed = {
            "schema_version",
            "genome_id",
            "genome_revision",
            "lineage",
            "scientific_program",
            "agent_program",
            "runtime_binding",
            "frozen_contract_refs",
            "evidence_refs",
            "behavior_digest",
            "genome_digest",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError("plugin genome has unsupported fields: " + ", ".join(sorted(unknown)))
        required = allowed - {"genome_id", "behavior_digest", "genome_digest"}
        missing = required - set(raw)
        if missing:
            raise ValueError("plugin genome is missing fields: " + ", ".join(sorted(missing)))
        normalized: dict[str, Any] = {
            "schema_version": _text(raw["schema_version"], "genome schema_version"),
            "genome_revision": _integer(raw["genome_revision"], "genome_revision", minimum=1),
            "lineage": _lineage(raw["lineage"]),
            "scientific_program": _scientific_program(raw["scientific_program"]),
            "agent_program": _agent_program(raw["agent_program"]),
            "runtime_binding": _runtime_binding(raw["runtime_binding"]),
            "frozen_contract_refs": _frozen_contract_refs(raw["frozen_contract_refs"]),
            "evidence_refs": _unique_sorted_texts(
                raw["evidence_refs"], "evidence_refs", pattern=_PROGRAM_ID_RE
            ),
        }
        if normalized["schema_version"] != GENOME_SCHEMA_VERSION:
            raise ValueError("unsupported plugin genome schema_version")
        if normalized["genome_revision"] != 1:
            raise ValueError("unsupported genome_revision")
        behavior_projection = {
            "schema_version": "ecologyrsi-dsh.plugin-behavior-source/1",
            "scientific_program": normalized["scientific_program"],
            "agent_program": normalized["agent_program"],
        }
        behavior_digest = _domain_digest(
            "ecologyrsi-dsh/plugin-behavior-source/1", behavior_projection
        )
        if "behavior_digest" in raw and _digest_text(
            raw["behavior_digest"], "behavior_digest"
        ) != behavior_digest:
            raise ValueError("plugin genome behavior_digest mismatch")
        normalized["behavior_digest"] = behavior_digest
        identity = dict(normalized)
        genome_digest = _domain_digest("ecologyrsi-dsh/plugin-genome/1", identity)
        if "genome_digest" in raw and _digest_text(
            raw["genome_digest"], "genome_digest"
        ) != genome_digest:
            raise ValueError("plugin genome genome_digest mismatch")
        genome_id = f"genome:{genome_digest[:24]}"
        if "genome_id" in raw and _text(raw["genome_id"], "genome_id") != genome_id:
            raise ValueError("plugin genome genome_id mismatch")
        normalized["genome_id"] = genome_id
        normalized["genome_digest"] = genome_digest
        frozen = deep_freeze_json(normalized)
        assert isinstance(frozen, FrozenJsonObject)
        return cls(frozen)

    @property
    def genome_id(self) -> str:
        return str(self._value["genome_id"])

    @property
    def genome_digest(self) -> str:
        return str(self._value["genome_digest"])

    @property
    def behavior_digest(self) -> str:
        return str(self._value["behavior_digest"])

    @property
    def lineage(self) -> FrozenJsonObject:
        value = self._value["lineage"]
        assert isinstance(value, FrozenJsonObject)
        return value

    @property
    def scientific_program(self) -> FrozenJsonObject:
        value = self._value["scientific_program"]
        assert isinstance(value, FrozenJsonObject)
        return value

    @property
    def agent_program(self) -> FrozenJsonObject:
        value = self._value["agent_program"]
        assert isinstance(value, FrozenJsonObject)
        return value

    def to_dict(self) -> dict[str, Any]:
        return deep_thaw_json(self._value)


def _empty_root_lineage(origin_kind: str, migration_source: Any = None) -> dict[str, Any]:
    return {
        "origin_kind": origin_kind,
        "parent_candidate_id": None,
        "parent_genome_digest": None,
        "generation": None,
        "slot_index": None,
        "slot_seed": None,
        "generation_batch_digest": None,
        "mutation_budget_digest": None,
        "mutation_operator_id": None,
        "mutation_digest": None,
        "source_research_iteration_digest": None,
        "source_knowledge_snapshot_digest": None,
        "migration_source": migration_source,
    }


def materialize_seed_genome(
    template: SeedGenomeTemplate,
    initialization: FrozenRunInitialization,
) -> EcologyEvolutionPluginGenome:
    if not isinstance(template, SeedGenomeTemplate):
        raise TypeError("template must be a SeedGenomeTemplate")
    if not isinstance(initialization, FrozenRunInitialization):
        raise TypeError("initialization must be FrozenRunInitialization")
    source = template.to_dict()
    return EcologyEvolutionPluginGenome.from_dict(
        {
            "schema_version": GENOME_SCHEMA_VERSION,
            "genome_revision": 1,
            "lineage": _empty_root_lineage("seed_catalog"),
            "scientific_program": source["scientific_program"],
            "agent_program": source["agent_program"],
            **initialization.bindings.to_dict(),
            "evidence_refs": source["evidence_refs"],
        }
    )


@dataclass(frozen=True, slots=True)
class GenomeMutationContextV1:
    run_id: str
    generation: int
    slot_index: int
    slot_seed: int
    parent_candidate_id: str | None
    parent_genome_digest: str
    generation_batch_digest: str
    research_iteration_digest: str
    knowledge_snapshot_digest: str
    mutation_budget_digest: str
    mutation_operator_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        for name in ("generation", "slot_index", "slot_seed"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=0))
        if self.parent_candidate_id is not None:
            object.__setattr__(
                self,
                "parent_candidate_id",
                _text(self.parent_candidate_id, "parent_candidate_id"),
            )
        if self.generation == 0 and self.parent_candidate_id is not None:
            raise ValueError("first generation parent_candidate_id must be null")
        if self.generation > 0 and self.parent_candidate_id is None:
            raise ValueError("later generation parent_candidate_id is required")
        for name in (
            "parent_genome_digest",
            "generation_batch_digest",
            "research_iteration_digest",
            "knowledge_snapshot_digest",
            "mutation_budget_digest",
        ):
            object.__setattr__(self, name, _digest_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "mutation_operator_id",
            _text(self.mutation_operator_id, "mutation_operator_id", pattern=_PROGRAM_ID_RE),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ecologyrsi-dsh.genome-mutation-context/1",
            "run_id": self.run_id,
            "generation": self.generation,
            "slot_index": self.slot_index,
            "slot_seed": self.slot_seed,
            "parent_candidate_id": self.parent_candidate_id,
            "parent_genome_digest": self.parent_genome_digest,
            "generation_batch_digest": self.generation_batch_digest,
            "research_iteration_digest": self.research_iteration_digest,
            "knowledge_snapshot_digest": self.knowledge_snapshot_digest,
            "mutation_budget_digest": self.mutation_budget_digest,
            "mutation_operator_id": self.mutation_operator_id,
        }


def _operation(raw_value: Any, fields: set[str], op_name: str) -> dict[str, Any]:
    raw = _exact_mapping(raw_value, f"mutation operation {op_name}", fields | {"op"})
    if raw["op"] != op_name:
        raise ValueError("mutation operation name mismatch")
    return raw


def apply_genome_mutation(
    parent: EcologyEvolutionPluginGenome,
    accepted_mutation: Mapping[str, Any],
    context: GenomeMutationContextV1,
    registry: Any,
) -> EcologyEvolutionPluginGenome:
    """Apply one host-validated mutation without ever accepting executable code."""

    if not isinstance(parent, EcologyEvolutionPluginGenome):
        raise TypeError("projected legacy views require explicit migration before mutation")
    if not isinstance(context, GenomeMutationContextV1):
        raise TypeError("context must be GenomeMutationContextV1")
    if context.parent_genome_digest != parent.genome_digest:
        raise ValueError("mutation context parent_genome_digest mismatch")
    raw = _exact_mapping(
        accepted_mutation,
        "accepted_mutation",
        {"schema_version", "operations"},
    )
    if raw["schema_version"] != MUTATION_SCHEMA_VERSION:
        raise ValueError("unsupported genome mutation schema_version")
    operations = raw["operations"]
    if isinstance(operations, (str, bytes)) or not isinstance(operations, Sequence):
        raise TypeError("mutation operations must be an array")
    if len(operations) > 32:
        raise ValueError("mutation contains too many operations")
    normalized_mutation = deep_thaw_json(deep_freeze_json(raw))
    result = parent.to_dict()
    scientific = result["scientific_program"]
    agent = result["agent_program"]
    for raw_operation in operations:
        if not isinstance(raw_operation, Mapping):
            raise TypeError("mutation operation must be an object")
        op = _text(raw_operation.get("op"), "mutation op", pattern=_PROGRAM_ID_RE)
        if op == "set_bounded_parameter":
            item = _operation(raw_operation, {"name", "value"}, op)
            name = _text(item["name"], "parameter name", pattern=_PROGRAM_ID_RE)
            value = _finite_number(item["value"], f"parameter {name}")
            registry.validate_parameter(
                scientific["predictor_ref"]["id"], name, value
            )
            scientific["parameter_overrides"][name] = value
        elif op == "select_registered_pipeline":
            item = _operation(raw_operation, {"predictor_id"}, op)
            predictor_id = _text(item["predictor_id"], "predictor_id", pattern=_PROGRAM_ID_RE)
            scientific["predictor_ref"] = registry.program_ref("predictors", predictor_id)
            scientific["parameter_overrides"] = registry.predictor_defaults(predictor_id)
        elif op in {
            "select_registered_feature_policy",
            "select_registered_fit_policy",
            "select_registered_uncertainty_policy",
        }:
            field_by_op = {
                "select_registered_feature_policy": ("feature_policy_ref", "feature_policies"),
                "select_registered_fit_policy": ("fit_policy_ref", "fit_policies"),
                "select_registered_uncertainty_policy": (
                    "uncertainty_policy_ref",
                    "uncertainty_policies",
                ),
            }
            item = _operation(raw_operation, {"program_id"}, op)
            field_name, category = field_by_op[op]
            ref = registry.program_ref(
                category,
                _text(item["program_id"], "program_id", pattern=_PROGRAM_ID_RE),
            )
            scientific[field_name] = {**ref, "overrides": {}}
        elif op == "select_registered_workflow_template":
            item = _operation(raw_operation, {"workflow_template_id"}, op)
            workflow_id = _text(
                item["workflow_template_id"], "workflow_template_id", pattern=_PROGRAM_ID_RE
            )
            agent["candidate_execution_program"]["workflow_template_ref"] = registry.program_ref(
                "workflow_templates", workflow_id
            )
            agent["candidate_execution_program"]["workflow_overrides"] = registry.workflow_defaults(
                workflow_id
            )
        elif op == "set_bounded_workflow_parameter":
            item = _operation(raw_operation, {"name", "value"}, op)
            name = _text(item["name"], "workflow parameter", pattern=_PROGRAM_ID_RE)
            value = _finite_number(item["value"], f"workflow parameter {name}")
            workflow_id = agent["candidate_execution_program"]["workflow_template_ref"]["id"]
            registry.validate_workflow_parameter(workflow_id, name, value)
            agent["candidate_execution_program"]["workflow_overrides"][name] = value
        elif op == "select_instruction_template":
            item = _operation(raw_operation, {"role", "instruction_template_id"}, op)
            role = _text(item["role"], "mutation role", pattern=_PRESET_ID_RE)
            if role not in {"sample-planner", "sample-repair"}:
                raise ValueError("candidate cannot mutate reproduction or reviewer role instructions")
            profile = next(
                (
                    profile
                    for profile in agent["candidate_execution_program"]["role_profiles"]
                    if profile["role"] == role
                ),
                None,
            )
            if profile is None:
                raise ValueError("mutation role is not registered in candidate execution")
            profile["instruction_template_ref"] = registry.program_ref(
                "instruction_templates",
                _text(
                    item["instruction_template_id"],
                    "instruction_template_id",
                    pattern=_PROGRAM_ID_RE,
                ),
            )
            profile["instruction_parameters"] = {}
        elif op == "set_instruction_parameter":
            item = _operation(raw_operation, {"role", "name", "value"}, op)
            role = _text(item["role"], "mutation role", pattern=_PRESET_ID_RE)
            if role not in {"sample-planner", "sample-repair"}:
                raise ValueError("candidate cannot mutate reproduction or reviewer role instructions")
            profile = next(
                (
                    profile
                    for profile in agent["candidate_execution_program"]["role_profiles"]
                    if profile["role"] == role
                ),
                None,
            )
            if profile is None:
                raise ValueError("mutation role is not registered in candidate execution")
            name = _text(item["name"], "instruction parameter", pattern=_PROGRAM_ID_RE)
            value = item["value"]
            registry.validate_instruction_parameter(
                profile["instruction_template_ref"]["id"], name, value
            )
            profile["instruction_parameters"][name] = value
        elif op == "narrow_role_tool_policy":
            item = _operation(raw_operation, {"role", "enabled_tool_ids"}, op)
            role = _text(item["role"], "mutation role", pattern=_PRESET_ID_RE)
            if role not in {"sample-planner", "sample-repair"}:
                raise ValueError("candidate cannot mutate reviewer tool policy")
            profile = next(
                (
                    profile
                    for profile in agent["candidate_execution_program"]["role_profiles"]
                    if profile["role"] == role
                ),
                None,
            )
            if profile is None:
                raise ValueError("mutation role is not registered in candidate execution")
            requested = _unique_sorted_texts(
                item["enabled_tool_ids"], "enabled_tool_ids", pattern=_PROGRAM_ID_RE
            )
            base = set(registry.tool_policy(profile["base_tool_policy_id"]))
            inherited = set(profile["enabled_tool_ids"])
            if not set(requested).issubset(base & inherited):
                raise ValueError("enabled tool policy must be a subset of the inherited base tools")
            profile["enabled_tool_ids"] = requested
        else:
            raise ValueError(f"unsupported genome mutation operation: {op}")

    mutation_digest = _domain_digest(
        "ecologyrsi-dsh/genome-mutation/1",
        {"accepted_mutation": normalized_mutation, "context": context.to_dict()},
    )
    result.pop("genome_id", None)
    result.pop("genome_digest", None)
    result.pop("behavior_digest", None)
    result["lineage"] = {
        "origin_kind": "bounded_mutation",
        "parent_candidate_id": context.parent_candidate_id,
        "parent_genome_digest": context.parent_genome_digest,
        "generation": context.generation,
        "slot_index": context.slot_index,
        "slot_seed": context.slot_seed,
        "generation_batch_digest": context.generation_batch_digest,
        "mutation_budget_digest": context.mutation_budget_digest,
        "mutation_operator_id": context.mutation_operator_id,
        "mutation_digest": mutation_digest,
        "source_research_iteration_digest": context.research_iteration_digest,
        "source_knowledge_snapshot_digest": context.knowledge_snapshot_digest,
        "migration_source": None,
    }
    result["scientific_program"] = scientific
    result["agent_program"] = agent
    return EcologyEvolutionPluginGenome.from_dict(result)


@dataclass(frozen=True, slots=True)
class ProjectedLegacyGenome:
    _value: FrozenJsonObject

    @property
    def projection_digest(self) -> str:
        return str(self._value["projection_digest"])

    @property
    def legacy_algorithm_ir(self) -> FrozenJsonObject:
        value = self._value["legacy_algorithm_ir"]
        assert isinstance(value, FrozenJsonObject)
        return value

    def to_dict(self) -> dict[str, Any]:
        return deep_thaw_json(self._value)


def _legacy_catalog_dict(legacy_catalog: Any) -> dict[str, Any]:
    if hasattr(legacy_catalog, "to_dict"):
        value = legacy_catalog.to_dict()
    else:
        value = deep_thaw_json(legacy_catalog)
    if not isinstance(value, Mapping):
        raise TypeError("legacy catalog must be an immutable mapping")
    return dict(value)


def legacy_genome_from_proposal(
    proposal: Proposal,
    task: TaskManifest,
    knowledge_snapshot: KnowledgeSnapshot | None,
    legacy_catalog: Any = None,
    *,
    current_registry: Any = None,
) -> ProjectedLegacyGenome:
    """Project a historical proposal using only the shipped 0.2.2 catalog."""

    from ..knowledge.models import KnowledgeSnapshot

    del current_registry  # Explicitly excluded from legacy replay identity.
    if legacy_catalog is None:
        from ..knowledge.program_registry import LEGACY_PROGRAM_CATALOG_0_2_2

        legacy_catalog = LEGACY_PROGRAM_CATALOG_0_2_2
    if not isinstance(proposal, Proposal) or not isinstance(task, TaskManifest):
        raise TypeError("legacy projection requires Proposal and TaskManifest")
    if knowledge_snapshot is not None and not isinstance(
        knowledge_snapshot, KnowledgeSnapshot
    ):
        raise TypeError("knowledge_snapshot must be KnowledgeSnapshot or null")
    if proposal.run_id != (knowledge_snapshot.run_id if knowledge_snapshot else proposal.run_id):
        raise ValueError("legacy knowledge snapshot is outside the proposal run")
    if knowledge_snapshot is not None and proposal.generation != knowledge_snapshot.generation:
        raise ValueError("legacy knowledge snapshot is outside the proposal generation")
    catalog = _legacy_catalog_dict(legacy_catalog)
    if catalog.get("schema_version") != "ecologyrsi-dsh.legacy-program-catalog/0.2.2":
        raise ValueError("unsupported legacy program catalog")
    plan = proposal.metadata.get("plan")
    plan = dict(plan) if isinstance(plan, Mapping) else {}
    adoption = proposal.metadata.get("prediction_model_adoption")
    adoption = dict(adoption) if isinstance(adoption, Mapping) else {}
    prediction_model = plan.get("prediction_model")
    requested = (
        str(prediction_model.get("id") or "").strip()
        if isinstance(prediction_model, Mapping)
        else ""
    )
    predictor_id = requested or str(adoption.get("adopted_id") or "").strip() or str(
        task.metadata.get("prediction_model_id") or "toy-rolling-water@1"
    ).strip()
    predictors = catalog.get("predictors")
    if not isinstance(predictors, Mapping) or predictor_id not in predictors:
        raise ValueError("legacy predictor is not in the frozen 0.2.2 catalog")
    predictor = dict(predictors[predictor_id])
    evaluator_id = str(
        task.metadata.get("evaluator_id") or predictor.get("default_evaluator_id") or ""
    ).strip()
    parameters = dict(proposal.changes)
    expected_parameters = set(predictor.get("parameter_names", ()))
    if set(parameters) != expected_parameters:
        raise ValueError("legacy proposal parameters do not match the frozen predictor")
    for name, value in parameters.items():
        _finite_number(value, f"legacy parameter {name}")
    dataset_digest = task.metadata.get("dataset_digest")
    if not isinstance(dataset_digest, str) or not dataset_digest.strip():
        dataset_digest = digest({"legacy_visible_datasets": list(task.visible_datasets)})
    split_manifest_digest = task.metadata.get("split_manifest_digest")
    if not isinstance(split_manifest_digest, str) or not split_manifest_digest.strip():
        split_manifest_digest = digest(
            {
                "legacy_task_manifest_digest": task.digest,
                "allowed_partitions": ["training_fit", "training_feedback"],
            }
        )
    blueprint = plan.get("algorithm_blueprint")
    synthesis = plan.get("algorithm_synthesis")
    blueprint_refs = set(
        str(item)
        for item in (
            blueprint.get("evidence_refs", ()) if isinstance(blueprint, Mapping) else ()
        )
    )
    evidence_mappings: list[dict[str, str]] = []
    if knowledge_snapshot is not None:
        for card in knowledge_snapshot.cards:
            capability_ids = tuple(card.capability_ids) or (
                ((card.capability_id,) if card.capability_id else ())
            )
            if card.knowledge_id in blueprint_refs and predictor_id in capability_ids:
                decision = "adopted"
            elif card.knowledge_id in blueprint_refs or not capability_ids:
                decision = "research_only"
            elif predictor_id in capability_ids and card.executable:
                decision = "adopted"
            else:
                decision = "not_selected"
            evidence_mappings.append(
                {"knowledge_id": card.knowledge_id, "decision": decision}
            )
    if isinstance(blueprint, Mapping):
        blueprint_pipeline = str(blueprint.get("pipeline_id") or "").strip()
        if blueprint_pipeline:
            evidence_mappings.append(
                {
                    "knowledge_id": f"model-blueprint:pipeline:{blueprint_pipeline}"[:300],
                    "decision": (
                        "adopted" if blueprint_pipeline == predictor_id else "not_selected"
                    ),
                }
            )
    if isinstance(prediction_model, Mapping):
        planned_predictor = str(prediction_model.get("id") or "").strip()
        if planned_predictor:
            evidence_mappings.append(
                {
                    "knowledge_id": f"model-plan:predictor:{planned_predictor}"[:300],
                    "decision": (
                        "adopted" if planned_predictor == predictor_id else "not_selected"
                    ),
                }
            )
    identity: dict[str, Any] = {
        "schema_version": "ecologyrsi-dsh.algorithm-ir/1",
        "predictor_id": predictor_id,
        "evaluator_id": evaluator_id,
        "pipeline_version": predictor["pipeline_version"],
        "parameters": parameters,
        "dataset_digest": str(dataset_digest),
        "split_manifest_digest": str(split_manifest_digest),
        "allowed_partitions": ["training_fit", "training_feedback"],
        "operators": list(predictor["operators"]),
        "evidence_mappings": evidence_mappings,
        "source_plan_digest": digest(plan) if plan else None,
        "knowledge_snapshot_digest": (
            knowledge_snapshot.snapshot_digest if knowledge_snapshot is not None else None
        ),
        "lowering_policy": "host-registered-operator-lowering@1",
        "security_boundary": {
            "registered_operators_only": True,
            "model_generated_code_execution": False,
            "dynamic_imports": False,
            "shell_execution": False,
        },
    }
    if isinstance(blueprint, Mapping):
        normalized_blueprint = dict(blueprint)
        identity["source_blueprint_digest"] = digest(normalized_blueprint)
    if isinstance(synthesis, Mapping):
        identity["source_synthesis_digest"] = digest(dict(synthesis))
    legacy_ir = {**identity, "ir_digest": digest(identity)}
    catalog_digest = str(catalog.get("catalog_digest") or "")
    if _SHA256_RE.fullmatch(catalog_digest) is None:
        catalog_without_digest = dict(catalog)
        catalog_without_digest.pop("catalog_digest", None)
        catalog_digest = _domain_digest(
            "ecologyrsi-dsh/legacy-program-catalog/0.2.2", catalog_without_digest
        )
    compiler_source = {
        "algorithm_blueprint": blueprint if isinstance(blueprint, Mapping) else None,
        "algorithm_synthesis": synthesis if isinstance(synthesis, Mapping) else None,
        "prediction_model_adoption": adoption or None,
        "proposal_changes": parameters,
    }
    projection = {
        "schema_version": LEGACY_PROJECTION_SCHEMA_VERSION,
        "projected": True,
        "inheritable": False,
        "adapter_version": LEGACY_ADAPTER_VERSION,
        "legacy_catalog_digest": catalog_digest,
        "source_proposal_digest": proposal.digest,
        "task_manifest_digest": task.digest,
        "knowledge_snapshot_digest": (
            knowledge_snapshot.snapshot_digest if knowledge_snapshot is not None else None
        ),
        "legacy_compiler_source": compiler_source,
        "legacy_algorithm_ir": legacy_ir,
        "projection_identity_knowledge": (
            knowledge_snapshot.snapshot_digest
            if knowledge_snapshot is not None
            else str(catalog.get("no_snapshot_sentinel"))
        ),
    }
    projection_digest = _domain_digest(
        "ecologyrsi-dsh/legacy-genome-projection/1", projection
    )
    projection["projection_digest"] = projection_digest
    frozen = deep_freeze_json(projection)
    assert isinstance(frozen, FrozenJsonObject)
    return ProjectedLegacyGenome(frozen)


def migrate_legacy_seed(
    projected_legacy: ProjectedLegacyGenome,
    bindings: FrozenRunInitialization,
    frozen_migration_template: Mapping[str, Any] | FrozenJsonObject,
) -> EcologyEvolutionPluginGenome:
    if not isinstance(projected_legacy, ProjectedLegacyGenome):
        raise TypeError("legacy migration requires a projected legacy genome")
    if not isinstance(bindings, FrozenRunInitialization):
        raise TypeError("legacy migration requires FrozenRunInitialization")
    template = deep_thaw_json(frozen_migration_template)
    if not isinstance(template, Mapping):
        raise TypeError("frozen migration template must be an object")
    required = {
        "schema_version",
        "template_id",
        "template_digest",
        "predictor_refs",
        "feature_policy_ref",
        "fit_policy_ref",
        "uncertainty_policy_ref",
        "agent_program",
        "evidence_refs",
    }
    if set(template) != required:
        raise ValueError("frozen migration template has unsupported or missing fields")
    projected = projected_legacy.to_dict()
    legacy_ir = projected["legacy_algorithm_ir"]
    predictor_id = legacy_ir["predictor_id"]
    predictor_refs = template["predictor_refs"]
    if not isinstance(predictor_refs, Mapping) or predictor_id not in predictor_refs:
        raise ValueError("migration template cannot map the legacy predictor")
    migration_source = {
        "projection_digest": projected_legacy.projection_digest,
        "source_proposal_digest": projected["source_proposal_digest"],
        "legacy_adapter_version": projected["adapter_version"],
        "legacy_catalog_digest": projected["legacy_catalog_digest"],
        "migration_template_id": template["template_id"],
        "migration_template_digest": template["template_digest"],
    }
    scientific = {
        "predictor_ref": predictor_refs[predictor_id],
        "parameter_overrides": legacy_ir["parameters"],
        "feature_policy_ref": template["feature_policy_ref"],
        "fit_policy_ref": template["fit_policy_ref"],
        "uncertainty_policy_ref": template["uncertainty_policy_ref"],
    }
    evidence_refs = list(template["evidence_refs"])
    evidence_refs.append("legacy:" + projected_legacy.projection_digest)
    return EcologyEvolutionPluginGenome.from_dict(
        {
            "schema_version": GENOME_SCHEMA_VERSION,
            "genome_revision": 1,
            "lineage": _empty_root_lineage("legacy_migration", migration_source),
            "scientific_program": scientific,
            "agent_program": template["agent_program"],
            **bindings.bindings.to_dict(),
            "evidence_refs": evidence_refs,
        }
    )


__all__ = [
    "EcologyEvolutionPluginGenome",
    "FrozenJson",
    "FrozenJsonArray",
    "FrozenJsonObject",
    "FrozenRunInitialization",
    "GENOME_CANONICAL_VERSION",
    "GENOME_SCHEMA_VERSION",
    "GenomeBindingSubset",
    "GenomeMutationContextV1",
    "ProjectedLegacyGenome",
    "SeedGenomeTemplate",
    "apply_genome_mutation",
    "deep_freeze_json",
    "deep_thaw_json",
    "legacy_genome_from_proposal",
    "materialize_seed_genome",
    "migrate_legacy_seed",
]
