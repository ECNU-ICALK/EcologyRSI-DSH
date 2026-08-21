"""Lossless shared-context encoding for one causal sample-routing wave."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..core.models import canonical_json, digest
from .sample_execution import SampleExecutionContractError, SamplePredictionRequest

ORIGIN_SHARED_CONTEXT_PROFILE = "origin_shared_context@1"
ORIGIN_SHARED_CONTEXT_SCHEMA = "ecologyrsi-dsh.origin-shared-sample-context/1"
SIBLING_STAGE_CONTEXT_SCHEMA = "ecologyrsi-dsh.sibling-sample-stage-context/1"

_REMOTE_SAMPLE_DETAIL_FIELDS = (
    "target",
    "unit",
    "origin_timestamp",
    "baseline",
    "minimum",
    "maximum",
)


def normalized_sample_planner_prompt_profile(
    value: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Validate the frozen opt-in profile without changing legacy runs."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("sample_planner_prompt_profile must be an object")
    if set(value) != {"version"}:
        raise ValueError(
            "sample_planner_prompt_profile must define exactly version"
        )
    version = value.get("version")
    if version != ORIGIN_SHARED_CONTEXT_PROFILE:
        raise ValueError(
            "sample_planner_prompt_profile.version must be "
            + ORIGIN_SHARED_CONTEXT_PROFILE
        )
    return {"version": ORIGIN_SHARED_CONTEXT_PROFILE}


def sibling_stage_context_digest(
    *,
    task_manifest_digest: str,
    generation: int,
    frozen_contract_digests: Mapping[str, str],
) -> str:
    """Digest the candidate-independent context shared by sibling genomes."""

    if not isinstance(task_manifest_digest, str) or len(task_manifest_digest) != 64:
        raise ValueError("task_manifest_digest must be a SHA-256 digest")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    if not isinstance(frozen_contract_digests, Mapping) or not frozen_contract_digests:
        raise ValueError("frozen_contract_digests must be a non-empty object")
    normalized: dict[str, str] = {}
    for name, value in frozen_contract_digests.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("sibling stage contract digests must be SHA-256 values")
        normalized[name] = value
    return digest(
        {
            "schema_version": SIBLING_STAGE_CONTEXT_SCHEMA,
            "task_manifest_digest": task_manifest_digest,
            "generation": generation,
            "frozen_contract_digests": {
                name: normalized[name] for name in sorted(normalized)
            },
        }
    )


def build_origin_shared_routing_payload(
    requests: Sequence[SamplePredictionRequest],
    attempts: Sequence[int],
    failure_feedbacks: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Factor repeated request data while preserving every remote-visible value."""

    if not requests or not (
        len(requests) == len(attempts) == len(failure_feedbacks)
    ):
        raise ValueError(
            "shared sample requests, attempts, and failure feedback must align"
        )
    sample_ids = [request.sample_id for request in requests]
    if len(sample_ids) != len(set(sample_ids)):
        raise SampleExecutionContractError(
            "shared sample context requires unique sample identifiers"
        )

    detail_rows = [
        {
            field: request.to_dict()[field]
            for field in _REMOTE_SAMPLE_DETAIL_FIELDS
        }
        for request in requests
    ]
    label_contexts = [dict(request.label_free_context) for request in requests]
    sample_defaults = _exact_common_fields(detail_rows)
    label_defaults = _exact_common_fields(label_contexts)
    detail_variants = [
        {
            **_without_keys(details, sample_defaults),
            **(
                {"label_free_context": _without_keys(label_context, label_defaults)}
                if _without_keys(label_context, label_defaults)
                else {}
            ),
        }
        for details, label_context in zip(detail_rows, label_contexts)
    ]

    origin_guard = {
        "origin_timestamps": _unique_values(
            request.origin_timestamp for request in requests
        ),
        "causal_provenance_digests": sorted(
            {
                digest(request.label_free_context["causal_provenance"])
                for request in requests
                if isinstance(
                    request.label_free_context.get("causal_provenance"), Mapping
                )
            }
        ),
    }
    variants: dict[str, dict[str, Any]] = {}
    sample_variant_refs: dict[str, str] = {}
    for sample_id, variant in zip(sample_ids, detail_variants):
        variant_ref = digest(
            {
                "schema_version": ORIGIN_SHARED_CONTEXT_SCHEMA,
                "origin_guard": origin_guard,
                "variant": variant,
            }
        )
        variants.setdefault(variant_ref, variant)
        sample_variant_refs[sample_id] = variant_ref

    shared_context = {
        "schema_version": ORIGIN_SHARED_CONTEXT_SCHEMA,
        "origin_guard": origin_guard,
        "sample_defaults": sample_defaults,
        "label_free_context_defaults": label_defaults,
        "sample_variants": variants,
        "sample_variant_refs": sample_variant_refs,
        "sample_count": len(requests),
    }
    context_ref = digest(shared_context)
    remote_samples = [
        {
            "sample_id": request.sample_id,
            "context_ref": context_ref,
            "horizon_hours": request.horizon_hours,
            "target_timestamp": request.target_timestamp,
            "attempt": attempt,
            "failure_feedback": [dict(item) for item in failure_feedback],
        }
        for request, attempt, failure_feedback in zip(
            requests, attempts, failure_feedbacks
        )
    ]
    return remote_samples, {context_ref: shared_context}


def expand_origin_shared_routing_payload(
    samples: Sequence[Mapping[str, Any]],
    shared_contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate references and reconstruct the lossless routing view."""

    expanded: dict[str, dict[str, Any]] = {}
    for raw_sample in samples:
        sample = dict(raw_sample)
        sample_id = sample.get("sample_id")
        context_ref = sample.get("context_ref")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("shared routing sample_id must be non-empty text")
        if not isinstance(context_ref, str) or context_ref not in shared_contexts:
            raise ValueError("shared routing sample context_ref is unresolved")
        shared = dict(shared_contexts[context_ref])
        if digest(shared) != context_ref:
            raise ValueError("shared routing context digest does not match context_ref")
        if shared.get("schema_version") != ORIGIN_SHARED_CONTEXT_SCHEMA:
            raise ValueError("shared routing context schema is unsupported")
        origin_guard = shared.get("origin_guard")
        sample_refs = shared.get("sample_variant_refs")
        variants = shared.get("sample_variants")
        defaults = shared.get("sample_defaults")
        label_defaults = shared.get("label_free_context_defaults")
        if not all(
            isinstance(value, Mapping)
            for value in (
                origin_guard,
                sample_refs,
                variants,
                defaults,
                label_defaults,
            )
        ):
            raise ValueError("shared routing context is malformed")
        variant_ref = sample_refs.get(sample_id)
        if not isinstance(variant_ref, str) or variant_ref not in variants:
            raise ValueError("shared routing sample variant is unresolved")
        raw_variant = variants[variant_ref]
        if not isinstance(raw_variant, Mapping):
            raise ValueError("shared routing sample variant is malformed")
        variant = dict(raw_variant)
        if digest(
            {
                "schema_version": ORIGIN_SHARED_CONTEXT_SCHEMA,
                "origin_guard": dict(origin_guard),
                "variant": variant,
            }
        ) != variant_ref:
            raise ValueError("shared routing sample variant digest does not match")
        label_variant = variant.pop("label_free_context", {})
        if not isinstance(label_variant, Mapping):
            raise ValueError("shared routing label context variant is malformed")
        details = {**dict(defaults), **variant}
        label_context = {**dict(label_defaults), **dict(label_variant)}
        details["label_free_context"] = label_context
        expanded[sample_id] = {
            "sample_id": sample_id,
            **details,
            "horizon_hours": sample.get("horizon_hours"),
            "target_timestamp": sample.get("target_timestamp"),
            "attempt": sample.get("attempt"),
            "failure_feedback": sample.get("failure_feedback"),
        }
    return expanded


def _exact_common_fields(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not values:
        return {}
    common_keys = set(values[0])
    for value in values[1:]:
        common_keys.intersection_update(value)
    common: dict[str, Any] = {}
    for key in sorted(common_keys):
        encoded = canonical_json(values[0][key])
        if all(canonical_json(value[key]) == encoded for value in values[1:]):
            common[key] = values[0][key]
    return common


def _without_keys(
    value: Mapping[str, Any], removed: Mapping[str, Any]
) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in removed}


def _unique_values(values: Any) -> list[Any]:
    indexed: dict[str, Any] = {}
    for value in values:
        indexed.setdefault(canonical_json(value), value)
    return [indexed[key] for key in sorted(indexed)]
