"""Restricted algorithm IR lowered only to host-registered operators.

Research models may select a registered predictor and propose bounded numeric
parameters.  They never provide executable source.  This module lowers that
selection to an immutable operator graph owned by the host so compilation,
smoke testing, and replay all refer to the same executable semantics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.models import JsonObject, canonical_json, digest

ALGORITHM_IR_VERSION = "ecologyrsi-dsh.algorithm-ir/1"
ALGORITHM_IR_LOWERING_POLICY = "host-registered-operator-lowering@1"
ALGORITHM_BLUEPRINT_VERSION = "ecologyrsi-dsh.algorithm-blueprint-request/1"
ALGORITHM_SYNTHESIS_VERSION = "ecologyrsi-dsh.algorithm-synthesis/1"
IR_ALLOWED_PARTITIONS = ("training_fit", "training_feedback")

_REGISTERED_PIPELINES: dict[str, dict[str, Any]] = {
    "toy-rolling-water@1": {
        "pipeline_version": "toy-water-operator-graph/1",
        "parameter_names": ("alpha", "window", "water_threshold"),
        "operators": (
            ("host.feature.causal-window@1", "feature", ("window",)),
            (
                "host.predictor.toy-water-balance@1",
                "predict",
                ("alpha", "water_threshold"),
            ),
            ("host.postprocess.physical-bounds@1", "postprocess", ()),
        ),
    },
    "greenhouse-rolling-residual@1": {
        "pipeline_version": "greenhouse-rolling-operator-graph/1",
        "parameter_names": ("blend", "window", "bias_scale"),
        "operators": (
            ("host.feature.causal-rolling-window@1", "feature", ("window",)),
            ("host.fit.mean-residual-bias@1", "fit", ("bias_scale",)),
            ("host.predictor.rolling-residual@1", "predict", ("blend",)),
            ("host.postprocess.physical-bounds@1", "postprocess", ()),
        ),
    },
    "greenhouse-exogenous-ridge@1": {
        "pipeline_version": "greenhouse-ridge-operator-graph/1",
        "parameter_names": ("history_steps", "ridge_alpha", "residual_scale"),
        "operators": (
            (
                "host.feature.causal-lag-exogenous@1",
                "feature",
                ("history_steps",),
            ),
            ("host.fit.partition-statistics@1", "fit", ()),
            ("host.fit.closed-form-ridge@1", "fit", ("ridge_alpha",)),
            (
                "host.predictor.residual-correction@1",
                "predict",
                ("residual_scale",),
            ),
            ("host.postprocess.physical-bounds@1", "postprocess", ()),
        ),
    },
    "greenhouse-targetwise-ridge@1": {
        "pipeline_version": "greenhouse-targetwise-ridge-operator-graph/1",
        "parameter_names": (
            "history_steps",
            "ridge_alpha",
            "air_temperature_residual_scale",
            "relative_humidity_residual_scale",
            "co2_concentration_residual_scale",
        ),
        "operators": (
            (
                "host.feature.causal-lag-exogenous@1",
                "feature",
                ("history_steps",),
            ),
            ("host.fit.partition-statistics@1", "fit", ()),
            ("host.fit.closed-form-ridge@1", "fit", ("ridge_alpha",)),
            (
                "host.predictor.targetwise-residual-or-persistence@1",
                "predict",
                (
                    "air_temperature_residual_scale",
                    "relative_humidity_residual_scale",
                    "co2_concentration_residual_scale",
                ),
            ),
            ("host.postprocess.physical-bounds@1", "postprocess", ()),
        ),
    },
    "greenhouse-horizon-targetwise-ridge@1": {
        "pipeline_version": "greenhouse-horizon-targetwise-ridge-operator-graph/1",
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
        "operators": (
            (
                "host.feature.causal-lag-exogenous@1",
                "feature",
                ("history_steps",),
            ),
            ("host.fit.partition-statistics@1", "fit", ()),
            ("host.fit.closed-form-ridge@1", "fit", ("ridge_alpha",)),
            (
                "host.predictor.horizon-targetwise-residual-or-persistence@1",
                "predict",
                (
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
            ),
            ("host.postprocess.physical-bounds@1", "postprocess", ()),
        ),
    },
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _registered_operators(predictor_id: str) -> tuple[dict[str, Any], ...]:
    try:
        pipeline = _REGISTERED_PIPELINES[predictor_id]
    except KeyError:
        raise ValueError(f"algorithm IR predictor is not registered: {predictor_id}") from None
    return tuple(
        {
            "operator_id": operator_id,
            "stage": stage,
            "parameter_names": list(parameter_names),
        }
        for operator_id, stage, parameter_names in pipeline["operators"]
    )


def registered_operator_tool_ids(predictor_id: str) -> tuple[str, ...]:
    """Return the immutable operator IDs for one registered predictor."""

    return tuple(item["operator_id"] for item in _registered_operators(predictor_id))


def registered_algorithm_blueprint(predictor_id: str) -> dict[str, Any]:
    """Return the model-visible executable blueprint for one host pipeline."""

    try:
        pipeline = _REGISTERED_PIPELINES[predictor_id]
    except KeyError:
        raise ValueError(
            f"algorithm blueprint pipeline is not registered: {predictor_id}"
        ) from None
    return {
        "schema_version": ALGORITHM_BLUEPRINT_VERSION,
        "pipeline_id": predictor_id,
        "operator_ids": list(registered_operator_tool_ids(predictor_id)),
        "parameter_names": list(pipeline["parameter_names"]),
    }


def registered_algorithm_blueprint_catalog(
    predictor_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return immutable blueprints, optionally limited to compatible predictors."""

    selected = tuple(_REGISTERED_PIPELINES) if predictor_ids is None else tuple(predictor_ids)
    return tuple(registered_algorithm_blueprint(item) for item in selected)


def validate_registered_algorithm_blueprint(
    value: Mapping[str, Any],
    *,
    expected_predictor_id: str | None = None,
) -> dict[str, Any]:
    """Validate a model request against one exact registered operator graph."""

    if not isinstance(value, Mapping):
        raise TypeError("algorithm blueprint must be an object")
    allowed = {
        "schema_version",
        "pipeline_id",
        "operator_ids",
        "parameter_names",
        "evidence_refs",
        "rationale",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            "algorithm blueprint has unsupported fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    if value.get("schema_version") != ALGORITHM_BLUEPRINT_VERSION:
        raise ValueError("unsupported algorithm blueprint version")
    pipeline_id = _text(value.get("pipeline_id"), "algorithm blueprint pipeline_id")
    if expected_predictor_id is not None and pipeline_id != expected_predictor_id:
        raise ValueError("algorithm blueprint does not match the adopted predictor")
    registered = registered_algorithm_blueprint(pipeline_id)
    for name in ("operator_ids", "parameter_names"):
        raw = value.get(name)
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError(f"algorithm blueprint {name} must be an array")
        normalized = [_text(item, f"algorithm blueprint {name} item") for item in raw]
        if normalized != registered[name]:
            raise ValueError(
                f"algorithm blueprint {name} do not match the registered pipeline"
            )

    raw_refs = value.get("evidence_refs")
    if isinstance(raw_refs, (str, bytes)) or not isinstance(raw_refs, Sequence):
        raise TypeError("algorithm blueprint evidence_refs must be an array")
    if not raw_refs or len(raw_refs) > 16:
        raise ValueError("algorithm blueprint evidence_refs must contain 1 to 16 items")
    evidence_refs: list[str] = []
    for raw_ref in raw_refs:
        evidence_ref = _text(raw_ref, "algorithm blueprint evidence ref")
        if len(evidence_ref) > 300:
            raise ValueError("algorithm blueprint evidence ref is too long")
        if evidence_ref not in evidence_refs:
            evidence_refs.append(evidence_ref)
    result = {
        **registered,
        "evidence_refs": evidence_refs,
    }
    if value.get("rationale") is not None:
        result["rationale"] = _text(
            value.get("rationale"), "algorithm blueprint rationale"
        )[:4000]
    canonical_json(result)
    return result


def validate_bounded_algorithm_synthesis(
    value: Mapping[str, Any],
    *,
    algorithm_blueprint: Mapping[str, Any],
    allowed_evidence_refs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate an evidence-backed design over one registered operator graph.

    A synthesis can select which registered parameters deserve exploration, but
    it cannot add operators, code, dependencies, or parameters.  Its source
    evidence must belong to the frozen evidence scope supplied by the caller.
    Compatibility callers that omit that scope retain the stricter historical
    rule that synthesis evidence must also appear on the executable blueprint.
    """

    if not isinstance(value, Mapping):
        raise TypeError("algorithm synthesis must be an object")
    allowed = {
        "schema_version",
        "pipeline_id",
        "evidence_refs",
        "parameter_focus",
        "rationale",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            "algorithm synthesis has unsupported fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    missing = allowed - set(value)
    if missing:
        raise ValueError(
            "algorithm synthesis is missing fields: "
            + ", ".join(sorted(missing))
        )
    if value.get("schema_version") != ALGORITHM_SYNTHESIS_VERSION:
        raise ValueError("unsupported algorithm synthesis version")
    blueprint = validate_registered_algorithm_blueprint(algorithm_blueprint)
    pipeline_id = _text(value.get("pipeline_id"), "algorithm synthesis pipeline_id")
    if pipeline_id != blueprint["pipeline_id"]:
        raise ValueError("algorithm synthesis does not match the registered blueprint")

    raw_refs = value.get("evidence_refs")
    if isinstance(raw_refs, (str, bytes)) or not isinstance(raw_refs, Sequence):
        raise TypeError("algorithm synthesis evidence_refs must be an array")
    if not raw_refs or len(raw_refs) > 16:
        raise ValueError("algorithm synthesis evidence_refs must contain 1 to 16 items")
    evidence_refs = list(
        dict.fromkeys(
            _text(item, "algorithm synthesis evidence ref") for item in raw_refs
        )
    )
    if allowed_evidence_refs is None:
        evidence_scope = set(blueprint["evidence_refs"])
    else:
        if isinstance(allowed_evidence_refs, (str, bytes)) or not isinstance(
            allowed_evidence_refs, Sequence
        ):
            raise TypeError("algorithm synthesis evidence scope must be an array")
        evidence_scope = {
            _text(item, "algorithm synthesis evidence scope item")
            for item in allowed_evidence_refs
        }
    if not set(evidence_refs).issubset(evidence_scope):
        raise ValueError(
            "algorithm synthesis evidence is outside the frozen evidence scope"
        )

    raw_focus = value.get("parameter_focus")
    if isinstance(raw_focus, (str, bytes)) or not isinstance(raw_focus, Sequence):
        raise TypeError("algorithm synthesis parameter_focus must be an array")
    if not raw_focus or len(raw_focus) > len(blueprint["parameter_names"]):
        raise ValueError("algorithm synthesis parameter_focus has an invalid size")
    parameter_focus = list(
        dict.fromkeys(
            _text(item, "algorithm synthesis parameter focus") for item in raw_focus
        )
    )
    if not set(parameter_focus).issubset(set(blueprint["parameter_names"])):
        raise ValueError("algorithm synthesis references an unregistered parameter")

    result = {
        "schema_version": ALGORITHM_SYNTHESIS_VERSION,
        "pipeline_id": pipeline_id,
        "evidence_refs": evidence_refs,
        "parameter_focus": parameter_focus,
        "rationale": _text(value.get("rationale"), "algorithm synthesis rationale")[:4000],
    }
    canonical_json(result)
    return result


def _validated_parameters(
    predictor_id: str,
    value: Mapping[str, Any],
) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise TypeError("algorithm IR parameters must be an object")
    expected = frozenset(_REGISTERED_PIPELINES[predictor_id]["parameter_names"])
    if frozenset(value) != expected:
        raise ValueError("algorithm IR parameters do not match the predictor contract")
    result: dict[str, int | float] = {}
    for name, raw in value.items():
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise ValueError(f"algorithm IR parameter is not finite: {name}")
        result[str(name)] = raw
    canonical_json(result)
    return result


def _validated_mappings(
    value: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    if isinstance(value, (str, bytes)) or len(value) > 32:
        raise ValueError("algorithm IR evidence mappings must contain at most 32 items")
    result: list[dict[str, str]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise TypeError("algorithm IR evidence mapping must be an object")
        if set(raw) != {"knowledge_id", "decision"}:
            raise ValueError("algorithm IR evidence mapping has unsupported fields")
        decision = _text(raw.get("decision"), "evidence decision")
        if decision not in {"adopted", "not_selected", "research_only"}:
            raise ValueError("algorithm IR evidence decision is unsupported")
        result.append(
            {
                "knowledge_id": _text(raw.get("knowledge_id"), "knowledge_id")[:300],
                "decision": decision,
            }
        )
    canonical_json(result)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class AlgorithmIR:
    """Immutable executable graph containing no model-supplied code."""

    predictor_id: str
    evaluator_id: str
    pipeline_version: str
    parameters: Mapping[str, Any]
    dataset_digest: str
    split_manifest_digest: str
    allowed_partitions: tuple[str, ...]
    operators: tuple[Mapping[str, Any], ...]
    evidence_mappings: tuple[Mapping[str, Any], ...] = ()
    source_plan_digest: str | None = None
    source_blueprint_digest: str | None = None
    source_synthesis_digest: str | None = None
    knowledge_snapshot_digest: str | None = None
    lowering_policy: str = ALGORITHM_IR_LOWERING_POLICY
    schema_version: str = ALGORITHM_IR_VERSION
    ir_digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "predictor_id",
            "evaluator_id",
            "pipeline_version",
            "dataset_digest",
            "split_manifest_digest",
            "lowering_policy",
            "schema_version",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.schema_version != ALGORITHM_IR_VERSION:
            raise ValueError("unsupported algorithm IR version")
        if self.lowering_policy != ALGORITHM_IR_LOWERING_POLICY:
            raise ValueError("unsupported algorithm IR lowering policy")
        try:
            pipeline = _REGISTERED_PIPELINES[self.predictor_id]
        except KeyError:
            raise ValueError(
                f"algorithm IR predictor is not registered: {self.predictor_id}"
            ) from None
        if self.pipeline_version != pipeline["pipeline_version"]:
            raise ValueError("algorithm IR pipeline version is not registered")
        parameters = _validated_parameters(self.predictor_id, self.parameters)
        object.__setattr__(self, "parameters", parameters)
        partitions = tuple(
            _text(item, "algorithm IR partition") for item in self.allowed_partitions
        )
        if partitions != IR_ALLOWED_PARTITIONS:
            raise ValueError(
                "algorithm IR may access only training_fit and training_feedback"
            )
        object.__setattr__(self, "allowed_partitions", partitions)
        operators = tuple(dict(item) for item in self.operators)
        if operators != _registered_operators(self.predictor_id):
            raise ValueError("algorithm IR operator graph is not host registered")
        object.__setattr__(self, "operators", operators)
        object.__setattr__(
            self,
            "evidence_mappings",
            _validated_mappings(self.evidence_mappings),
        )
        for name in (
            "source_plan_digest",
            "source_blueprint_digest",
            "source_synthesis_digest",
            "knowledge_snapshot_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        expected = digest(self.identity_dict())
        if self.ir_digest and self.ir_digest != expected:
            raise ValueError("algorithm IR digest mismatch")
        object.__setattr__(self, "ir_digest", expected)

    def identity_dict(self) -> JsonObject:
        result: JsonObject = {
            "schema_version": self.schema_version,
            "predictor_id": self.predictor_id,
            "evaluator_id": self.evaluator_id,
            "pipeline_version": self.pipeline_version,
            "parameters": dict(self.parameters),
            "dataset_digest": self.dataset_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "allowed_partitions": list(self.allowed_partitions),
            "operators": [dict(item) for item in self.operators],
            "evidence_mappings": [dict(item) for item in self.evidence_mappings],
            "source_plan_digest": self.source_plan_digest,
            "knowledge_snapshot_digest": self.knowledge_snapshot_digest,
            "lowering_policy": self.lowering_policy,
            "security_boundary": {
                "registered_operators_only": True,
                "model_generated_code_execution": False,
                "dynamic_imports": False,
                "shell_execution": False,
            },
        }
        if self.source_blueprint_digest is not None:
            result["source_blueprint_digest"] = self.source_blueprint_digest
        if self.source_synthesis_digest is not None:
            result["source_synthesis_digest"] = self.source_synthesis_digest
        return result

    def to_dict(self) -> JsonObject:
        return {**self.identity_dict(), "ir_digest": self.ir_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AlgorithmIR:
        if not isinstance(value, Mapping):
            raise TypeError("algorithm IR must be an object")
        data = dict(value)
        boundary = data.pop("security_boundary", None)
        if boundary != {
            "registered_operators_only": True,
            "model_generated_code_execution": False,
            "dynamic_imports": False,
            "shell_execution": False,
        }:
            raise ValueError("algorithm IR security boundary is invalid")
        return cls(**data)


def build_registered_algorithm_ir(
    *,
    predictor_id: str,
    evaluator_id: str,
    parameters: Mapping[str, Any],
    dataset_digest: str,
    split_manifest_digest: str,
    allowed_partitions: Sequence[str],
    evidence_mappings: Sequence[Mapping[str, Any]] = (),
    source_plan_digest: str | None = None,
    algorithm_blueprint: Mapping[str, Any] | None = None,
    algorithm_synthesis: Mapping[str, Any] | None = None,
    knowledge_snapshot_digest: str | None = None,
) -> AlgorithmIR:
    """Lower research evidence to one registered operator pipeline."""

    try:
        pipeline_version = str(_REGISTERED_PIPELINES[predictor_id]["pipeline_version"])
    except KeyError:
        raise ValueError(f"algorithm IR predictor is not registered: {predictor_id}") from None
    normalized_blueprint = (
        validate_registered_algorithm_blueprint(
            algorithm_blueprint,
            expected_predictor_id=predictor_id,
        )
        if algorithm_blueprint is not None
        else None
    )
    if algorithm_synthesis is not None and normalized_blueprint is None:
        raise ValueError("algorithm synthesis requires a registered blueprint")
    normalized_synthesis = (
        validate_bounded_algorithm_synthesis(
            algorithm_synthesis,
            algorithm_blueprint=normalized_blueprint,
            allowed_evidence_refs=tuple(
                dict.fromkeys(
                    [
                        *(
                            normalized_blueprint.get("evidence_refs", ())
                            if normalized_blueprint is not None
                            else ()
                        ),
                        *(
                            str(item.get("knowledge_id"))
                            for item in evidence_mappings
                            if isinstance(item, Mapping)
                            and isinstance(item.get("knowledge_id"), str)
                            and str(item.get("knowledge_id")).strip()
                        ),
                    ]
                )
            ),
        )
        if algorithm_synthesis is not None and normalized_blueprint is not None
        else None
    )
    return AlgorithmIR(
        predictor_id=predictor_id,
        evaluator_id=evaluator_id,
        pipeline_version=pipeline_version,
        parameters=dict(parameters),
        dataset_digest=dataset_digest,
        split_manifest_digest=split_manifest_digest,
        allowed_partitions=tuple(allowed_partitions),
        operators=_registered_operators(predictor_id),
        evidence_mappings=tuple(dict(item) for item in evidence_mappings),
        source_plan_digest=source_plan_digest,
        source_blueprint_digest=(
            digest(normalized_blueprint) if normalized_blueprint is not None else None
        ),
        source_synthesis_digest=(
            digest(normalized_synthesis) if normalized_synthesis is not None else None
        ),
        knowledge_snapshot_digest=knowledge_snapshot_digest,
    )


def algorithm_behavior_projection(value: AlgorithmIR | Mapping[str, Any]) -> JsonObject:
    """Return the positive allowlist used for behavior-level identity.

    Dataset instances, plans, evidence, lineage and display metadata are
    intentionally absent.  Defaults must be resolved before this function is
    called, so one effective operator behavior has one projection.
    """

    algorithm_ir = value if isinstance(value, AlgorithmIR) else AlgorithmIR.from_dict(value)
    return {
        "schema_version": "algorithm_behavior_projection@1",
        "predictor_id": algorithm_ir.predictor_id,
        "evaluator_id": algorithm_ir.evaluator_id,
        "pipeline_version": algorithm_ir.pipeline_version,
        "parameters": dict(algorithm_ir.parameters),
        "allowed_partitions": list(algorithm_ir.allowed_partitions),
        "operators": [dict(item) for item in algorithm_ir.operators],
        "lowering_policy": algorithm_ir.lowering_policy,
    }


__all__ = [
    "ALGORITHM_BLUEPRINT_VERSION",
    "ALGORITHM_IR_LOWERING_POLICY",
    "ALGORITHM_IR_VERSION",
    "ALGORITHM_SYNTHESIS_VERSION",
    "AlgorithmIR",
    "algorithm_behavior_projection",
    "build_registered_algorithm_ir",
    "registered_algorithm_blueprint",
    "registered_algorithm_blueprint_catalog",
    "registered_operator_tool_ids",
    "validate_bounded_algorithm_synthesis",
    "validate_registered_algorithm_blueprint",
]
