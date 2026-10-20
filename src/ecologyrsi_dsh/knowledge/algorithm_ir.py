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
IR_ALLOWED_PARTITIONS = ("training_fit", "training_feedback")

_REGISTERED_PIPELINES: dict[str, dict[str, Any]] = {
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
    "greenhouse-baseline-aligned-ridge@2": {
        "pipeline_version": "greenhouse-baseline-aligned-ridge-operator-graph/2",
        "parameter_names": ("history_steps", "ridge_alpha", *tuple(f"{t}_{h}h_residual_scale" for t in ("air_temperature", "relative_humidity", "co2_concentration") for h in (1, 6, 24))),
        "operators": (
            ("host.feature.causal-lag-exogenous@1", "feature", ("history_steps",)),
            ("host.fit.partition-selected-baseline@1", "fit", ()),
            ("host.fit.partition-statistics@1", "fit", ()),
            ("host.fit.baseline-aligned-closed-form-ridge@1", "fit", ("ridge_alpha",)),
            ("host.predictor.fit-selected-baseline-cellwise-residual@1", "predict", tuple(f"{t}_{h}h_residual_scale" for t in ("air_temperature", "relative_humidity", "co2_concentration") for h in (1, 6, 24))),
            ("host.postprocess.physical-bounds@1", "postprocess", ()),
        ),
    },
    "greenhouse-baseline-aligned-ridge@1": {
        "pipeline_version": "greenhouse-baseline-aligned-ridge-operator-graph/1",
        "parameter_names": ("history_steps", "ridge_alpha",
                            "residual_scale_1h", "residual_scale_6h", "residual_scale_24h"),
        "operators": (
            ("host.feature.causal-lag-exogenous@1", "feature", ("history_steps",)),
            ("host.fit.partition-selected-baseline@1", "fit", ()),
            ("host.fit.partition-statistics@1", "fit", ()),
            ("host.fit.baseline-aligned-closed-form-ridge@1", "fit", ("ridge_alpha",)),
            ("host.predictor.fit-selected-baseline-residual@1", "predict",
             ("residual_scale_1h", "residual_scale_6h", "residual_scale_24h")),
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
    "greenhouse-recipe-ridge@1": {
        "pipeline_version": "greenhouse-recipe-operator-graph/1",
        # Deliberately empty, and every operator below declares no scalars for
        # the same reason: the ridge alpha, the anchor and each per-cell
        # residual scale live in scientific_program.feature_recipe, which the
        # compiler projects into feature_training_spec.feature_recipe. The IR
        # therefore names the operator graph, and the recipe names the knobs;
        # neither restates the other, so they cannot disagree.
        "parameter_names": (),
        "operators": (
            ("host.feature.compiled-feature-recipe@1", "feature", ()),
            ("host.fit.partition-selected-baseline@1", "fit", ()),
            ("host.fit.partition-statistics@1", "fit", ()),
            ("host.fit.baseline-aligned-closed-form-ridge@1", "fit", ()),
            ("host.predictor.fit-selected-baseline-residual@1", "predict", ()),
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
            "knowledge_snapshot_digest": self.knowledge_snapshot_digest,
            "lowering_policy": self.lowering_policy,
            "security_boundary": {
                "registered_operators_only": True,
                "model_generated_code_execution": False,
                "dynamic_imports": False,
                "shell_execution": False,
            },
        }
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
    knowledge_snapshot_digest: str | None = None,
) -> AlgorithmIR:
    """Lower research evidence to one registered operator pipeline."""

    try:
        pipeline_version = str(_REGISTERED_PIPELINES[predictor_id]["pipeline_version"])
    except KeyError:
        raise ValueError(f"algorithm IR predictor is not registered: {predictor_id}") from None
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
    "ALGORITHM_IR_LOWERING_POLICY",
    "ALGORITHM_IR_VERSION",
    "AlgorithmIR",
    "algorithm_behavior_projection",
    "build_registered_algorithm_ir",
    "registered_operator_tool_ids",
]
