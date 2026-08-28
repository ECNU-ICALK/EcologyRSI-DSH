"""Bounded shared-context encoding for one causal sample-routing wave."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.models import canonical_json, digest
from .sample_execution import SampleExecutionContractError, SamplePredictionRequest

ORIGIN_SHARED_CONTEXT_PROFILE = "origin_shared_context@1"
# Keep the transport envelope at ``/1`` because ModelGateway validates that
# frozen profile before transport.  The payload itself advertises the bounded
# routing-manifest revision independently; old envelopes without that marker
# remain readable for ledger replay and tests.
ORIGIN_SHARED_CONTEXT_SCHEMA = "ecologyrsi-dsh.origin-shared-sample-context/1"
ORIGIN_ROUTING_MANIFEST_VERSION = "origin_routing_manifest@2"
LEGACY_ORIGIN_ROUTING_MANIFEST_VERSION = "origin_shared_context_manifest@1"
ORIGIN_ANOMALY_SUMMARY_SCHEMA = "ecologyrsi-dsh.origin-anomaly-summary/1"
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
    *,
    manifest_version: str = ORIGIN_ROUTING_MANIFEST_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build the bounded v2 manifest, with an explicit legacy fixture seam."""

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

    if manifest_version == LEGACY_ORIGIN_ROUTING_MANIFEST_VERSION:
        return _build_legacy_origin_shared_routing_payload(
            requests,
            attempts,
            failure_feedbacks,
        )
    if manifest_version != ORIGIN_ROUTING_MANIFEST_VERSION:
        raise ValueError("unsupported origin routing manifest version")

    return _build_origin_routing_manifest_v2(
        requests,
        attempts,
        failure_feedbacks,
    )


def _build_origin_routing_manifest_v2(
    requests: Sequence[SamplePredictionRequest],
    attempts: Sequence[int],
    failure_feedbacks: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    context_digests = {
        request.sample_id: digest(request.label_free_context)
        for request in requests
    }
    causal_digests = {
        request.sample_id: digest(
            request.label_free_context.get("causal_provenance")
            if isinstance(
                request.label_free_context.get("causal_provenance"), Mapping
            )
            else None
        )
        for request in requests
    }
    origin_guard = {
        "origin_timestamps": _unique_values(
            request.origin_timestamp for request in requests
        ),
        "causal_provenance_digests": sorted(set(causal_digests.values())),
        "sample_context_digests": sorted(set(context_digests.values())),
    }
    variants: dict[str, dict[str, Any]] = {}
    sample_variant_refs: dict[str, str] = {}
    for request in requests:
        variant = {
            "target": request.target,
            "unit": request.unit,
            "origin_timestamp": request.origin_timestamp,
            "baseline": request.baseline,
            "minimum": request.minimum,
            "maximum": request.maximum,
            "causal_provenance_digest": causal_digests[request.sample_id],
            "sample_context_digest": context_digests[request.sample_id],
            "anomaly_summary": _origin_anomaly_summary(request),
        }
        variant_ref = digest(
            {
                "schema_version": ORIGIN_SHARED_CONTEXT_SCHEMA,
                "origin_guard": origin_guard,
                "variant": variant,
            }
        )
        variants.setdefault(variant_ref, variant)
        sample_variant_refs[request.sample_id] = variant_ref

    shared_context = {
        "schema_version": ORIGIN_SHARED_CONTEXT_SCHEMA,
        "routing_manifest_version": ORIGIN_ROUTING_MANIFEST_VERSION,
        "origin_guard": origin_guard,
        "sample_variants": variants,
        "sample_variant_refs": sample_variant_refs,
        "sample_count": len(requests),
    }
    context_ref = digest(shared_context)
    remote_samples = [
        {
            "sample_id": request.sample_id,
            "context_ref": context_ref,
            "target": request.target,
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


def _build_legacy_origin_shared_routing_payload(
    requests: Sequence[SamplePredictionRequest],
    attempts: Sequence[int],
    failure_feedbacks: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Construct a v1 fixture for replay validation; new calls default to v2."""

    sample_ids = [request.sample_id for request in requests]
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
    """Validate and expand legacy v1 or bounded routing-manifest v2."""

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
        manifest_version = shared.get("routing_manifest_version")
        if manifest_version == ORIGIN_ROUTING_MANIFEST_VERSION:
            expanded[sample_id] = _expand_origin_routing_manifest_v2_sample(
                sample,
                shared,
                origin_guard=origin_guard,
                sample_refs=sample_refs,
                variants=variants,
            )
            continue
        if manifest_version is not None:
            raise ValueError("shared routing manifest version is unsupported")
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


def _expand_origin_routing_manifest_v2_sample(
    sample: Mapping[str, Any],
    shared: Mapping[str, Any],
    *,
    origin_guard: Any,
    sample_refs: Any,
    variants: Any,
) -> dict[str, Any]:
    if not all(
        isinstance(value, Mapping)
        for value in (origin_guard, sample_refs, variants)
    ):
        raise ValueError("shared routing context is malformed")
    sample_count = shared.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or sample_count != len(sample_refs)
    ):
        raise ValueError("shared routing sample count is malformed")
    sample_id = str(sample["sample_id"])
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
    for name in ("causal_provenance_digest", "sample_context_digest"):
        value = variant.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"shared routing {name} is malformed")
    anomaly_summary = variant.get("anomaly_summary")
    if (
        not isinstance(anomaly_summary, Mapping)
        or anomaly_summary.get("schema_version") != ORIGIN_ANOMALY_SUMMARY_SCHEMA
    ):
        raise ValueError("shared routing anomaly summary is malformed")
    if sample.get("target") != variant.get("target"):
        raise ValueError("shared routing sample target does not match its variant")
    return {
        "sample_id": sample_id,
        **variant,
        "target": sample.get("target"),
        "horizon_hours": sample.get("horizon_hours"),
        "target_timestamp": sample.get("target_timestamp"),
        "attempt": sample.get("attempt"),
        "failure_feedback": sample.get("failure_feedback"),
    }


def _origin_anomaly_summary(
    request: SamplePredictionRequest,
) -> dict[str, Any]:
    """Summarize Host-visible inputs without exposing raw feature/history rows."""

    context = request.label_free_context
    raw_history = context.get("history_window")
    history_values = _finite_numeric_values(raw_history)
    history_count = _sequence_length(raw_history)
    raw_features = context.get("feature_snapshot")
    feature_rows = (
        list(raw_features)
        if isinstance(raw_features, Sequence)
        and not isinstance(raw_features, (str, bytes, bytearray))
        else []
    )
    feature_values = _finite_numeric_values(
        [
            row.get("value")
            for row in feature_rows
            if isinstance(row, Mapping)
        ]
    )
    feature_names = [
        str(row.get("name"))
        for row in feature_rows
        if isinstance(row, Mapping)
        and isinstance(row.get("name"), str)
        and str(row.get("name")).strip()
    ]
    flags: list[str] = []
    if history_count == 0:
        flags.append("history_missing")
    elif len(history_values) != history_count:
        flags.append("history_contains_non_numeric_values")
    if not feature_rows:
        flags.append("features_missing")
    elif len(feature_values) != len(feature_rows):
        flags.append("features_contain_non_numeric_values")
    if len(feature_names) != len(set(feature_names)):
        flags.append("duplicate_feature_names")
    if not request.minimum <= request.baseline <= request.maximum:
        flags.append("baseline_outside_bounds")
    if history_values and (
        min(history_values) < request.minimum
        or max(history_values) > request.maximum
    ):
        flags.append("history_outside_bounds")
    if not isinstance(context.get("causal_provenance"), Mapping):
        flags.append("causal_provenance_missing")

    return {
        "schema_version": ORIGIN_ANOMALY_SUMMARY_SCHEMA,
        "flags": sorted(flags),
        "history": _numeric_summary(
            history_values,
            observed_count=history_count,
            include_step_change=True,
        ),
        "features": {
            **_numeric_summary(
                feature_values,
                observed_count=len(feature_rows),
                include_step_change=False,
            ),
            "named_count": len(feature_names),
            "duplicate_name_count": len(feature_names) - len(set(feature_names)),
        },
        "context_field_count": len(context),
        "causal_provenance_present": isinstance(
            context.get("causal_provenance"), Mapping
        ),
    }


def _sequence_length(value: Any) -> int:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return len(value)
    return 0


def _finite_numeric_values(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return []
    return [
        float(item)
        for item in value
        if not isinstance(item, bool)
        and isinstance(item, (int, float))
        and math.isfinite(float(item))
    ]


def _numeric_summary(
    values: Sequence[float],
    *,
    observed_count: int,
    include_step_change: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "observed_count": observed_count,
        "numeric_count": len(values),
    }
    if not values:
        return summary
    summary.update(
        {
            "minimum": min(values),
            "maximum": max(values),
            "mean": sum(values) / len(values),
            "maximum_absolute_value": max(abs(value) for value in values),
        }
    )
    if include_step_change:
        summary["latest"] = values[0]
        summary["maximum_absolute_step_change"] = max(
            (abs(current - previous) for previous, current in zip(values, values[1:])),
            default=0.0,
        )
    return summary


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
