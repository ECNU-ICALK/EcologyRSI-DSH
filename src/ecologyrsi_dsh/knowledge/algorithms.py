"""Compile research evidence into host-registered executable algorithms.

The compiler deliberately has no dynamic import or source-code execution path.
It binds a proposal to an adapter and tool chain already shipped by the host,
then emits an immutable digest that can be checked before training starts.
"""

from __future__ import annotations

import math
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
    build_registered_algorithm_ir,
    validate_bounded_algorithm_synthesis,
    validate_registered_algorithm_blueprint,
)
from .models import KnowledgeSnapshot

ALGORITHM_SPEC_VERSION = "ecologyrsi-dsh.algorithm-spec/1"
PREDICTOR_ADOPTION_VERSION = "ecologyrsi-dsh.predictor-adoption/1"
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
        "parameter_sweep@1": {"version": "bounded-parent-sweep/6"},
        "adaptive_local@1": {"version": "bounded-feedback-local-search/6"},
        "dsh_authenticated@1": {"version": "authenticated-structured-proposal/7"},
        "autonomous_model@1": {
            "version": "per-generation-research-runtime-adoption/8"
        },
    },
    "predictor": {
        "toy-rolling-water@1": {
            "version": "toy-rolling-water/1",
            "parameter_names": ("alpha", "window", "water_threshold"),
            "tool_ids": ("host.predictor.toy-rolling-water.fit-predict@1",),
            "evaluator_ids": ("toy_time_forward@1",),
        },
        "greenhouse-rolling-residual@1": {
            "version": "greenhouse-rolling-residual/1",
            "parameter_names": ("blend", "window", "bias_scale"),
            "tool_ids": ("host.predictor.greenhouse-rolling-residual.fit-predict@1",),
            "evaluator_ids": ("greenhouse_time_forward@1",),
        },
        "greenhouse-exogenous-ridge@1": {
            "version": "greenhouse-exogenous-ridge/1",
            "parameter_names": ("history_steps", "ridge_alpha", "residual_scale"),
            "tool_ids": ("host.predictor.greenhouse-exogenous-ridge.fit-predict@1",),
            "evaluator_ids": (
                "greenhouse_time_forward@1",
                "greenhouse_multihorizon_time_forward@1",
                "greenhouse_multihorizon_time_forward@2",
            ),
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
            "evaluator_ids": (
                "greenhouse_time_forward@1",
                "greenhouse_multihorizon_time_forward@1",
                "greenhouse_multihorizon_time_forward@2",
            ),
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
            "evaluator_ids": ("greenhouse_multihorizon_time_forward@2",),
        },
    },
    "evaluator": {
        "toy_time_forward@1": {
            "version": "toy-time-forward/3",
            "tool_ids": ("host.evaluator.toy-time-forward.score@1",),
        },
        "greenhouse_time_forward@1": {
            "version": "greenhouse-time-forward/5",
            "tool_ids": ("host.evaluator.greenhouse-time-forward.score@1",),
        },
        "greenhouse_multihorizon_time_forward@1": {
            "version": "greenhouse-multihorizon-time-forward/4",
            "tool_ids": (
                "host.evaluator.greenhouse-multihorizon-time-forward.score@1",
            ),
        },
        "greenhouse_multihorizon_time_forward@2": {
            "version": "greenhouse-multihorizon-time-forward/5",
            "tool_ids": (
                "host.evaluator.greenhouse-multihorizon-time-forward.score@2",
            ),
        },
    },
}


def registered_capability_ids(kind: str) -> frozenset[str]:
    """Return the immutable public ID set for one host capability kind."""

    return frozenset(_CAPABILITIES.get(str(kind), {}))


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


def _active_bindings(
    task: TaskManifest,
    proposal: Proposal | None = None,
) -> dict[str, str]:
    metadata = task.metadata
    toy = any("toy" in str(item).casefold() for item in task.visible_datasets)
    predictor_default = "toy-rolling-water@1" if toy else "greenhouse-exogenous-ridge@1"
    evaluator_default = (
        "toy_time_forward@1" if toy else "greenhouse_multihorizon_time_forward@2"
    )
    result = {
        "strategy": str(metadata.get("strategy_id") or "parameter_sweep@1"),
        "predictor": str(metadata.get("prediction_model_id") or predictor_default),
        "evaluator": str(metadata.get("evaluator_id") or evaluator_default),
    }
    if proposal is not None:
        if proposal.metadata.get("execution_protocol") == "dsh_native_plugin_evolution@1":
            from ..core.state import persisted_genome_from_proposal
            from .program_registry import current_program_registry

            genome = persisted_genome_from_proposal(proposal)
            if genome is None:  # pragma: no cover - guarded by protocol above
                raise AlgorithmCompileError(
                    "plugin_genome_missing",
                    "DSH-native proposal has no verified plugin genome",
                )
            predictor_ref = genome.scientific_program["predictor_ref"]
            predictor_id = str(predictor_ref["id"])
            expected_ref = current_program_registry().program_ref(
                "predictors", predictor_id
            )
            if dict(predictor_ref) != expected_ref:
                raise AlgorithmCompileError(
                    "plugin_genome_predictor_mismatch",
                    "plugin genome predictor is outside the frozen Host registry",
                )
            result["predictor"] = predictor_id
        raw_adoption = proposal.metadata.get("prediction_model_adoption")
        raw_plan = proposal.metadata.get("plan")
        if raw_adoption is not None:
            if not isinstance(raw_plan, Mapping):
                raise AlgorithmCompileError(
                    "predictor_adoption_missing_plan",
                    "proposal predictor adoption requires its frozen research plan",
                )
            expected = resolve_predictor_adoption(task, raw_plan)
            stored = PredictorAdoption.from_dict(raw_adoption)
            if stored.to_dict() != expected.to_dict():
                raise AlgorithmCompileError(
                    "predictor_adoption_mismatch",
                    "proposal predictor adoption does not match the frozen host catalog",
                )
            result["predictor"] = stored.adopted_id
    return result


def _canonical_partition(value: Any, *, toy: bool) -> str:
    name = str(value or "").strip().casefold()
    if toy and name == "train":
        return "training_fit"
    if toy and name == "validation":
        return "training_feedback"
    return name


def _validate_task_partition_boundary(task: TaskManifest) -> None:
    """Reject any task request that expands evolution beyond training data."""

    metadata = task.metadata
    toy = any("toy" in str(item).casefold() for item in task.visible_datasets)
    requested: list[str] = []
    for field_name in ("training_partition", "evaluation_partition"):
        raw = metadata.get(field_name)
        if raw is not None:
            requested.append(_canonical_partition(raw, toy=toy))
    raw_allowed = metadata.get("allowed_partitions")
    if raw_allowed is not None:
        if not isinstance(raw_allowed, (list, tuple)):
            raise AlgorithmCompileError(
                "invalid_partition_contract",
                "task allowed_partitions must be an array",
            )
        requested.extend(
            _canonical_partition(item, toy=toy) for item in raw_allowed
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


def _proposal_research_mappings(
    proposal: Proposal,
    bindings: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Map the model's bounded research plan onto executable host bindings."""

    raw_plan = proposal.metadata.get("plan")
    if not isinstance(raw_plan, Mapping):
        return []
    mappings: list[dict[str, Any]] = []
    raw_blueprint = raw_plan.get("algorithm_blueprint")
    blueprint = (
        validate_registered_algorithm_blueprint(raw_blueprint)
        if isinstance(raw_blueprint, Mapping)
        else None
    )
    if blueprint is not None:
        pipeline_id = str(blueprint["pipeline_id"])
        mappings.append(
            {
                "knowledge_id": f"model-blueprint:pipeline:{pipeline_id}"[:300],
                "capability_kind": "predictor",
                "capability_id": pipeline_id,
                "decision": (
                    "adopted"
                    if bindings.get("predictor") == pipeline_id
                    else "not_selected"
                ),
                "source_url": None,
                "source": "strategy_model_algorithm_blueprint",
            }
        )
    for component_field, kind in (
        ("prediction_model", "predictor"),
        ("strategy", "strategy"),
    ):
        component = raw_plan.get(component_field)
        if not isinstance(component, Mapping):
            continue
        capability_id = str(component.get("id") or "").strip()
        if not capability_id:
            continue
        registered = capability_id in registered_capability_ids(kind)
        decision = (
            "adopted"
            if registered and bindings.get(kind) == capability_id
            else "not_selected"
            if registered
            else "research_only"
        )
        mappings.append(
            {
                "knowledge_id": f"model-plan:{kind}:{capability_id}"[:300],
                "capability_kind": kind,
                "capability_id": capability_id,
                "decision": decision,
                "source_url": None,
                "source": "strategy_model_research_plan",
            }
        )

    research = raw_plan.get("research")
    if isinstance(research, list):
        for item in research[:16]:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or item.get("source") or "").strip()[:300]
            url = str(item.get("url") or "").strip()[:1000]
            if not title and not url:
                continue
            identity = {"title": title, "url": url}
            mappings.append(
                {
                    "knowledge_id": "model-research:" + digest(identity)[:20],
                    "capability_kind": None,
                    "capability_id": None,
                    "decision": "research_only",
                    "source_url": url or None,
                    "source": "strategy_model_research_plan",
                    "title": title or None,
                }
            )
    return mappings


@dataclass(frozen=True, slots=True)
class PredictorAdoption:
    """Host-verifiable result of mapping a model plan to a registered predictor."""

    status: str
    adopted_id: str
    adopted_digest: str
    default_id: str
    evaluator_id: str
    dataset_id: str
    catalog_digest: str
    plan_digest: str
    parameter_schema_digest: str
    reason: str
    requested_id: str | None = None
    schema_version: str = PREDICTOR_ADOPTION_VERSION
    adoption_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTOR_ADOPTION_VERSION:
            raise ValueError("unsupported predictor adoption version")
        if self.status not in {"adopted", "host_default", "research_only"}:
            raise ValueError("unsupported predictor adoption status")
        for name in (
            "adopted_id",
            "adopted_digest",
            "default_id",
            "evaluator_id",
            "dataset_id",
            "catalog_digest",
            "plan_digest",
            "parameter_schema_digest",
            "reason",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.requested_id is not None:
            object.__setattr__(
                self,
                "requested_id",
                _text(self.requested_id, "requested_id"),
            )
        if self.status == "adopted" and self.requested_id != self.adopted_id:
            raise ValueError("adopted predictor must match the requested predictor")
        if self.status != "adopted" and self.adopted_id != self.default_id:
            raise ValueError("rejected research must preserve the host default predictor")
        expected = digest(self.identity_dict())
        if self.adoption_digest and self.adoption_digest != expected:
            raise ValueError("predictor adoption digest mismatch")
        object.__setattr__(self, "adoption_digest", expected)

    def identity_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "requested_id": self.requested_id,
            "adopted_id": self.adopted_id,
            "adopted_digest": self.adopted_digest,
            "default_id": self.default_id,
            "evaluator_id": self.evaluator_id,
            "dataset_id": self.dataset_id,
            "catalog_digest": self.catalog_digest,
            "plan_digest": self.plan_digest,
            "parameter_schema_digest": self.parameter_schema_digest,
            "reason": self.reason,
            "security_boundary": {
                "registered_predictor_only": True,
                "model_generated_code_execution": False,
                "task_partitions_unchanged": True,
            },
        }

    def to_dict(self) -> JsonObject:
        return {**self.identity_dict(), "adoption_digest": self.adoption_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PredictorAdoption:
        if not isinstance(value, Mapping):
            raise TypeError("predictor adoption must be an object")
        data = dict(value)
        boundary = data.pop("security_boundary", None)
        if boundary != {
            "registered_predictor_only": True,
            "model_generated_code_execution": False,
            "task_partitions_unchanged": True,
        }:
            raise ValueError("predictor adoption security boundary is invalid")
        return cls(**data)


def _predictor_catalog_entry(
    task: TaskManifest,
    predictor_id: str,
) -> Mapping[str, Any] | None:
    catalog = task.metadata.get("runtime_component_catalog")
    if not isinstance(catalog, Mapping):
        return None
    raw_entries = catalog.get("prediction_models")
    if not isinstance(raw_entries, list):
        raise AlgorithmCompileError(
            "invalid_runtime_component_catalog",
            "runtime predictor catalog must be an array",
        )
    return next(
        (
            item
            for item in raw_entries
            if isinstance(item, Mapping) and item.get("id") == predictor_id
        ),
        None,
    )


def _validated_catalog_predictor(
    task: TaskManifest,
    predictor_id: str,
) -> tuple[str, str] | None:
    entry = _predictor_catalog_entry(task, predictor_id)
    if entry is None:
        return None
    evaluator_id = str(task.metadata.get("evaluator_id") or "")
    dataset_id = str(task.dataset or "")
    capability = _CAPABILITIES["predictor"].get(predictor_id)
    if capability is None or evaluator_id not in capability.get("evaluator_ids", ()):
        return None
    dataset_ids = entry.get("dataset_ids")
    if not isinstance(dataset_ids, list) or dataset_id not in dataset_ids:
        return None
    schemas = entry.get("parameter_schemas")
    if not isinstance(schemas, Mapping) or frozenset(schemas) != frozenset(
        capability.get("parameter_names", ())
    ):
        return None
    configuration_digest = entry.get("configuration_digest")
    if not isinstance(configuration_digest, str) or not configuration_digest.strip():
        return None
    return configuration_digest.strip(), digest(dict(schemas))


def resolve_predictor_adoption(
    task: TaskManifest,
    plan: Mapping[str, Any],
) -> PredictorAdoption:
    """Map an advisory plan onto the frozen compatible host component catalog."""

    if not isinstance(plan, Mapping):
        raise TypeError("research plan must be an object")
    canonical_json(dict(plan))
    metadata = task.metadata
    toy = any("toy" in str(item).casefold() for item in task.visible_datasets)
    default_id = str(
        metadata.get("prediction_model_id")
        or (
            "toy-rolling-water@1"
            if toy
            else "greenhouse-exogenous-ridge@1"
        )
    ).strip()
    evaluator_id = str(
        metadata.get("evaluator_id")
        or (
            "toy_time_forward@1"
            if toy
            else "greenhouse_multihorizon_time_forward@2"
        )
    ).strip()
    dataset_id = str(task.dataset or "").strip()
    if not default_id or not evaluator_id or not dataset_id:
        raise AlgorithmCompileError(
            "incomplete_default_predictor_binding",
            "task is missing its frozen predictor, evaluator, or dataset binding",
        )
    default_capability = _CAPABILITIES["predictor"].get(default_id)
    if default_capability is None or evaluator_id not in default_capability.get(
        "evaluator_ids", ()
    ):
        raise AlgorithmCompileError(
            "invalid_default_predictor_binding",
            "task default predictor is not registered for the frozen evaluator",
        )
    catalog = metadata.get("runtime_component_catalog")
    catalog_data = dict(catalog) if isinstance(catalog, Mapping) else {}
    if catalog_data:
        if catalog_data.get("selected_evaluator_id") != evaluator_id:
            raise AlgorithmCompileError(
                "runtime_catalog_evaluator_mismatch",
                "runtime component catalog does not match the frozen evaluator",
            )
        if catalog_data.get("selected_prediction_model_id") != default_id:
            raise AlgorithmCompileError(
                "runtime_catalog_default_mismatch",
                "runtime component catalog does not match the default predictor",
            )
    default_catalog = _validated_catalog_predictor(task, default_id)
    frozen_default_digest = metadata.get("prediction_model_digest")
    default_digest = (
        str(frozen_default_digest).strip()
        if isinstance(frozen_default_digest, str) and frozen_default_digest.strip()
        else default_catalog[0]
        if default_catalog is not None
        else digest(
            {
                "predictor_id": default_id,
                "version": default_capability["version"],
            }
        )
    )
    if default_catalog is not None and default_catalog[0] != default_digest:
        raise AlgorithmCompileError(
            "runtime_catalog_predictor_digest_mismatch",
            "runtime component catalog changed the frozen default predictor digest",
        )
    default_schema_digest = (
        default_catalog[1]
        if default_catalog is not None
        else digest(list(default_capability.get("parameter_names", ())))
    )

    component = plan.get("prediction_model")
    component_requested_id = (
        str(component.get("id") or "").strip()
        if isinstance(component, Mapping)
        else ""
    )
    raw_blueprint = plan.get("algorithm_blueprint")
    blueprint_requested_id = ""
    blueprint_error: str | None = None
    if raw_blueprint is not None:
        if not isinstance(raw_blueprint, Mapping):
            blueprint_error = "requested_algorithm_blueprint_is_invalid"
        else:
            try:
                blueprint = validate_registered_algorithm_blueprint(raw_blueprint)
            except (TypeError, ValueError):
                blueprint_error = "requested_algorithm_blueprint_is_not_host_registered"
                raw_pipeline_id = raw_blueprint.get("pipeline_id")
                if isinstance(raw_pipeline_id, str):
                    blueprint_requested_id = raw_pipeline_id.strip()
            else:
                blueprint_requested_id = str(blueprint["pipeline_id"])
    requested_id = blueprint_requested_id or component_requested_id
    blueprint_required = bool(
        requested_id
        and requested_id != default_id
        and metadata.get("model_workflow") == "research_compile_evolve@1"
    )
    status = "host_default"
    adopted_id = default_id
    adopted_digest = default_digest
    schema_digest = default_schema_digest
    reason = "research_plan_did_not_request_a_predictor"
    if blueprint_error is not None:
        status = "research_only"
        requested_id = requested_id or "invalid-algorithm-blueprint"
        reason = blueprint_error
    elif blueprint_required and not blueprint_requested_id:
        status = "research_only"
        reason = "requested_predictor_requires_registered_algorithm_blueprint"
    elif (
        blueprint_requested_id
        and component_requested_id
        and blueprint_requested_id != component_requested_id
    ):
        status = "research_only"
        reason = "algorithm_blueprint_conflicts_with_prediction_model"
    elif requested_id:
        requested_capability = _CAPABILITIES["predictor"].get(requested_id)
        if requested_capability is None:
            status = "research_only"
            reason = "requested_predictor_is_not_host_registered"
        else:
            selected = _validated_catalog_predictor(task, requested_id)
            if selected is None:
                status = "research_only"
                reason = "requested_predictor_is_not_in_frozen_compatible_catalog"
            else:
                status = "adopted"
                adopted_id = requested_id
                adopted_digest, schema_digest = selected
                reason = "requested_predictor_is_registered_and_compatible"
    return PredictorAdoption(
        status=status,
        requested_id=requested_id or None,
        adopted_id=adopted_id,
        adopted_digest=adopted_digest,
        default_id=default_id,
        evaluator_id=evaluator_id,
        dataset_id=dataset_id,
        catalog_digest=digest(catalog_data),
        plan_digest=digest(dict(plan)),
        parameter_schema_digest=schema_digest,
        reason=reason,
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
    predictor_adoption: Mapping[str, Any] | None = None
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
        if self.predictor_adoption is not None:
            adoption = PredictorAdoption.from_dict(self.predictor_adoption)
            if adoption.adopted_id != self.adapter_id:
                raise ValueError(
                    "predictor adoption does not match algorithm adapter binding"
                )
            if (
                self.algorithm_ir is not None
                and AlgorithmIR.from_dict(self.algorithm_ir).source_plan_digest
                != adoption.plan_digest
            ):
                raise ValueError(
                    "algorithm IR plan digest does not match predictor adoption"
                )
            object.__setattr__(self, "predictor_adoption", adoption.to_dict())
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
        if self.predictor_adoption is not None:
            result["predictor_adoption"] = dict(self.predictor_adoption)
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
    task: TaskManifest,
    proposal: Proposal,
    snapshot: KnowledgeSnapshot | None,
) -> AlgorithmSpec:
    """Compile a proposal without reading samples or accepting model code."""

    _validate_task_partition_boundary(task)
    raw_plan = proposal.metadata.get("plan")
    algorithm_blueprint: dict[str, Any] | None = None
    algorithm_synthesis: dict[str, Any] | None = None
    if isinstance(raw_plan, Mapping) and "algorithm_blueprint" in raw_plan:
        raw_blueprint = raw_plan.get("algorithm_blueprint")
        if not isinstance(raw_blueprint, Mapping):
            raise AlgorithmCompileError(
                "invalid_algorithm_blueprint",
                "algorithm blueprint must be a host-registered object",
            )
        try:
            algorithm_blueprint = validate_registered_algorithm_blueprint(
                raw_blueprint
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmCompileError(
                "invalid_algorithm_blueprint",
                f"algorithm blueprint failed the host registry contract: {exc}",
            ) from exc
        raw_prediction_model = raw_plan.get("prediction_model")
        requested_prediction_model = (
            str(raw_prediction_model.get("id") or "").strip()
            if isinstance(raw_prediction_model, Mapping)
            else ""
        )
        if requested_prediction_model and requested_prediction_model != str(
            algorithm_blueprint["pipeline_id"]
        ):
            raise AlgorithmCompileError(
                "algorithm_blueprint_predictor_conflict",
                "algorithm blueprint conflicts with the requested prediction model",
            )
    if isinstance(raw_plan, Mapping) and "algorithm_synthesis" in raw_plan:
        raw_synthesis = raw_plan.get("algorithm_synthesis")
        if algorithm_blueprint is None or not isinstance(raw_synthesis, Mapping):
            raise AlgorithmCompileError(
                "invalid_algorithm_synthesis",
                "algorithm synthesis requires a host-registered blueprint",
            )
        try:
            algorithm_synthesis = validate_bounded_algorithm_synthesis(
                raw_synthesis,
                algorithm_blueprint=algorithm_blueprint,
                allowed_evidence_refs=tuple(
                    dict.fromkeys(
                        [
                            *algorithm_blueprint.get("evidence_refs", ()),
                            *(
                                tuple(card.knowledge_id for card in snapshot.cards)
                                if snapshot is not None
                                else ()
                            ),
                        ]
                    )
                ),
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmCompileError(
                "invalid_algorithm_synthesis",
                f"algorithm synthesis failed the bounded host contract: {exc}",
            ) from exc
    bindings = _active_bindings(task, proposal)
    if (
        algorithm_blueprint is not None
        and algorithm_blueprint["pipeline_id"] != bindings["predictor"]
    ):
        raise AlgorithmCompileError(
            "algorithm_blueprint_binding_mismatch",
            "algorithm blueprint does not match the frozen predictor adoption",
        )
    resolved: dict[str, Mapping[str, Any]] = {}
    for kind, capability_id in bindings.items():
        capability = _CAPABILITIES.get(kind, {}).get(capability_id)
        if capability is None:
            raise AlgorithmCompileError(
                "unregistered_capability",
                f"{kind} capability is not registered: {capability_id}",
            )
        resolved[kind] = capability
    predictor = resolved["predictor"]
    if bindings["evaluator"] not in predictor.get("evaluator_ids", ()):
        raise AlgorithmCompileError(
            "incompatible_evaluator",
            "registered predictor and evaluator are incompatible",
        )
    expected_parameters = frozenset(predictor.get("parameter_names", ()))
    actual_parameters = frozenset(proposal.changes)
    if actual_parameters != expected_parameters:
        raise AlgorithmCompileError(
            "parameter_contract_mismatch",
            "proposal parameters do not match the registered adapter contract",
        )
    for name, value in proposal.changes.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise AlgorithmCompileError(
                "invalid_parameter_value", f"proposal parameter is not finite: {name}"
            )

    raw_execution_plan = proposal.metadata.get("derived_execution_plan")
    execution_plan: DerivedExecutionPlan | None = None
    if raw_execution_plan is not None:
        try:
            execution_plan = DerivedExecutionPlan.from_dict(raw_execution_plan)
        except (TypeError, ValueError) as exc:
            raise AlgorithmCompileError(
                "invalid_derived_execution_plan",
                f"proposal derived execution plan is invalid: {exc}",
            ) from exc
        expected_source_generation = (
            proposal.generation - 1 if proposal.generation > 0 else None
        )
        if execution_plan.source_generation != expected_source_generation:
            raise AlgorithmCompileError(
                "derived_execution_plan_scope_mismatch",
                "proposal execution plan does not reference the immediately previous generation",
            )

    raw_predictor_adoption = proposal.metadata.get("prediction_model_adoption")
    predictor_adoption = (
        PredictorAdoption.from_dict(raw_predictor_adoption)
        if isinstance(raw_predictor_adoption, Mapping)
        else None
    )

    blueprint_evidence_refs = frozenset(
        str(item)
        for item in (
            algorithm_blueprint.get("evidence_refs", ())
            if algorithm_blueprint is not None
            else ()
        )
    )
    synthesis_evidence_refs = frozenset(
        str(item)
        for item in (
            algorithm_synthesis.get("evidence_refs", ())
            if algorithm_synthesis is not None
            else ()
        )
    )
    if blueprint_evidence_refs and snapshot is None:
        raise AlgorithmCompileError(
            "algorithm_blueprint_evidence_snapshot_missing",
            "algorithm blueprint evidence requires the frozen generation snapshot",
        )
    mappings: list[dict[str, Any]] = []
    if snapshot is not None:
        if (
            snapshot.run_id != proposal.run_id
            or snapshot.generation != proposal.generation
        ):
            raise AlgorithmCompileError(
                "knowledge_scope_mismatch",
                "knowledge snapshot is outside the candidate scope",
            )
        known_evidence_refs = {card.knowledge_id for card in snapshot.cards}
        unknown_evidence_refs = sorted(
            blueprint_evidence_refs - known_evidence_refs
        )
        if unknown_evidence_refs:
            raise AlgorithmCompileError(
                "unknown_algorithm_blueprint_evidence",
                "algorithm blueprint references evidence outside the frozen snapshot: "
                + ", ".join(unknown_evidence_refs),
            )
        executable_blueprint_evidence = 0
        for card in snapshot.cards:
            kind = str(card.capability_kind or "")
            capability_id = str(card.capability_id or "")
            raw_capability_ids = getattr(card, "capability_ids", ())
            capability_ids = tuple(str(item) for item in raw_capability_ids) or (
                (capability_id,) if capability_id else ()
            )
            registered = any(
                item in registered_capability_ids(kind) for item in capability_ids
            )
            active = registered and bindings.get(kind) in capability_ids
            cited_by_blueprint = card.knowledge_id in blueprint_evidence_refs
            cited_executable = bool(
                cited_by_blueprint
                and kind == "predictor"
                and bindings["predictor"] in capability_ids
                and card.execution_status
                in {"adopted", "available_not_selected"}
            )
            if cited_executable:
                executable_blueprint_evidence += 1
                decision = "adopted"
            elif cited_by_blueprint:
                decision = "research_only"
            elif card.executable and active:
                decision = "adopted"
            elif not registered:
                decision = "research_only"
            else:
                decision = "not_selected"
            mappings.append(
                {
                    "knowledge_id": card.knowledge_id,
                    "capability_kind": kind,
                    "capability_id": capability_id or None,
                    "capability_ids": list(capability_ids),
                    "decision": decision,
                    "source_url": card.source_url,
                    "source_authority": card.source_authority,
                    "title": card.title,
                    "summary": card.summary,
                    "evidence_role": (
                        "algorithm_synthesis_source"
                        if card.knowledge_id in synthesis_evidence_refs
                        else "algorithm_blueprint_source"
                        if cited_by_blueprint
                        else "capability_catalog"
                    ),
                }
            )
        if blueprint_evidence_refs and executable_blueprint_evidence == 0:
            raise AlgorithmCompileError(
                "algorithm_blueprint_has_no_executable_evidence",
                "algorithm blueprint has no frozen executable evidence for its predictor",
            )
    mappings.extend(_proposal_research_mappings(proposal, bindings))

    predictor_tools = tuple(str(item) for item in predictor.get("tool_ids", ()))
    evaluator_tools = tuple(
        str(item) for item in resolved["evaluator"].get("tool_ids", ())
    )
    adapter_version = str(predictor["version"])
    evaluator_version = str(resolved["evaluator"]["version"])
    dataset_digest = task.metadata.get("dataset_digest")
    if not isinstance(dataset_digest, str) or not dataset_digest.strip():
        # Compatibility for direct core tests that bypass the API's runtime
        # binding. Production manifests always carry the real snapshot digest.
        dataset_digest = digest(
            {"legacy_visible_datasets": list(task.visible_datasets)}
        )
    split_manifest_digest = task.metadata.get("split_manifest_digest")
    if not isinstance(split_manifest_digest, str) or not split_manifest_digest.strip():
        split_manifest_digest = digest(
            {
                "legacy_task_manifest_digest": task.digest,
                "allowed_partitions": list(EVOLUTION_ALLOWED_PARTITIONS),
            }
        )
    source_plan_digest = (
        digest(dict(raw_plan)) if isinstance(raw_plan, Mapping) else None
    )
    algorithm_ir = build_registered_algorithm_ir(
        predictor_id=bindings["predictor"],
        evaluator_id=bindings["evaluator"],
        parameters=proposal.changes,
        dataset_digest=dataset_digest,
        split_manifest_digest=split_manifest_digest,
        allowed_partitions=EVOLUTION_ALLOWED_PARTITIONS,
        evidence_mappings=tuple(
            {
                "knowledge_id": str(item["knowledge_id"]),
                "decision": str(item["decision"]),
            }
            for item in mappings
            if item.get("knowledge_id") is not None
            and item.get("decision") in {"adopted", "not_selected", "research_only"}
        ),
        source_plan_digest=source_plan_digest,
        algorithm_blueprint=algorithm_blueprint,
        algorithm_synthesis=algorithm_synthesis,
        knowledge_snapshot_digest=(
            snapshot.snapshot_digest if snapshot is not None else None
        ),
    )
    base_algorithm_version = (
        f"registered-pipeline/{digest([adapter_version, evaluator_version])[:12]}"
    )
    algorithm_version = (
        f"{base_algorithm_version}+synthesis/{digest(algorithm_synthesis)[:12]}"
        if algorithm_synthesis is not None
        else base_algorithm_version
    )
    return AlgorithmSpec(
        run_id=proposal.run_id,
        generation=proposal.generation,
        proposal_id=proposal.proposal_id,
        algorithm_id=f"{bindings['predictor']}+{bindings['evaluator']}",
        algorithm_version=algorithm_version,
        adapter_id=bindings["predictor"],
        adapter_version=adapter_version,
        evaluator_id=bindings["evaluator"],
        evaluator_version=evaluator_version,
        strategy_id=bindings["strategy"],
        tool_ids=predictor_tools + evaluator_tools,
        parameters=dict(proposal.changes),
        visible_datasets=tuple(task.visible_datasets),
        dataset_digest=dataset_digest,
        split_manifest_digest=split_manifest_digest,
        allowed_partitions=EVOLUTION_ALLOWED_PARTITIONS,
        knowledge_snapshot_digest=(
            snapshot.snapshot_digest if snapshot is not None else None
        ),
        knowledge_mappings=tuple(mappings),
        algorithm_ir=algorithm_ir.to_dict(),
        predictor_adoption=(
            predictor_adoption.to_dict() if predictor_adoption is not None else None
        ),
        derived_execution_plan=(
            execution_plan.to_dict() if execution_plan is not None else None
        ),
    )


def debug_algorithm_spec(
    spec: AlgorithmSpec,
    task: TaskManifest,
    proposal: Proposal,
    snapshot: KnowledgeSnapshot | None = None,
) -> dict[str, Any]:
    """Run deterministic preflight checks against the registered adapter."""

    rebuilt = compile_algorithm_spec(task, proposal, snapshot)
    comparable = spec.identity_dict()
    rebuilt_identity = rebuilt.identity_dict()
    for name in ("knowledge_snapshot_digest", "knowledge_mappings"):
        rebuilt_identity[name] = comparable[name]
    if spec.dataset_digest is None:
        for name in (
            "dataset_digest",
            "split_manifest_digest",
            "allowed_partitions",
        ):
            rebuilt_identity.pop(name, None)
    if spec.algorithm_ir is None:
        rebuilt_identity.pop("algorithm_ir", None)
    else:
        raw_plan = proposal.metadata.get("plan")
        expected_ir = build_registered_algorithm_ir(
            predictor_id=spec.adapter_id,
            evaluator_id=spec.evaluator_id,
            parameters=spec.parameters,
            dataset_digest=str(spec.dataset_digest),
            split_manifest_digest=str(spec.split_manifest_digest),
            allowed_partitions=spec.allowed_partitions,
            evidence_mappings=tuple(
                {
                    "knowledge_id": str(item["knowledge_id"]),
                    "decision": str(item["decision"]),
                }
                for item in spec.knowledge_mappings
                if item.get("knowledge_id") is not None
                and item.get("decision")
                in {"adopted", "not_selected", "research_only"}
            ),
            source_plan_digest=(
                digest(dict(raw_plan)) if isinstance(raw_plan, Mapping) else None
            ),
            algorithm_blueprint=(
                raw_plan.get("algorithm_blueprint")
                if isinstance(raw_plan, Mapping)
                and isinstance(raw_plan.get("algorithm_blueprint"), Mapping)
                else None
            ),
            algorithm_synthesis=(
                raw_plan.get("algorithm_synthesis")
                if isinstance(raw_plan, Mapping)
                and isinstance(raw_plan.get("algorithm_synthesis"), Mapping)
                else None
            ),
            knowledge_snapshot_digest=spec.knowledge_snapshot_digest,
        )
        rebuilt_identity["algorithm_ir"] = expected_ir.to_dict()
    if digest(rebuilt_identity) != spec.spec_digest:
        raise AlgorithmCompileError(
            "algorithm_spec_binding_mismatch",
            "algorithm spec does not match the frozen task and proposal bindings",
        )
    return {
        "checks": [
            "host_registered_adapter",
            "registered_tool_chain",
            "frozen_runtime_bindings",
            "bounded_parameter_contract",
            "knowledge_capability_mapping",
            "registered_operator_algorithm_ir",
            "bounded_evidence_backed_algorithm_synthesis",
            "algorithm_ir_digest",
            "frozen_registered_predictor_adoption",
            "frozen_derived_execution_plan",
            "frozen_dataset_snapshot",
            "training_partition_allowlist",
            "no_external_code_execution",
        ],
        "check_count": 13,
        "passed": True,
    }


__all__ = [
    "ALGORITHM_SPEC_VERSION",
    "EVOLUTION_ALLOWED_PARTITIONS",
    "PREDICTOR_ADOPTION_VERSION",
    "AlgorithmAttempt",
    "AlgorithmCompileError",
    "AlgorithmSpec",
    "PredictorAdoption",
    "compile_algorithm_spec",
    "debug_algorithm_spec",
    "registered_capability_ids",
    "resolve_predictor_adoption",
]
