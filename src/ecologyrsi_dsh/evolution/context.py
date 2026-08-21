"""Validation and routing for aggregate-only generation feedback."""

from __future__ import annotations

import math
from collections.abc import Mapping
import re
from typing import Any

from ..core.models import Run
from ..knowledge.models import validate_knowledge_context
from ..knowledge.research_iteration import ResearchIteration

_BLOCKED_FIELDS = frozenset(
    {
        "baseline",
        "actual",
        "actual_value",
        "code",
        "command",
        "entrypoint",
        "ground_truth",
        "label",
        "module",
        "observation",
        "observations",
        "observed",
        "prediction",
        "predicted",
        "prediction_preview",
        "raw",
        "rows",
        "samples",
        "sample_execution",
        "sample_execution_records",
        "sample_execution_trace_archive",
        "script",
        "source_code",
        "prediction_records",
        "labels",
        "target_value",
        "timestamps",
    }
)
_BLOCKED_FIELD_TOKENS = frozenset(
    name.replace("_", "") for name in _BLOCKED_FIELDS
)


def is_sensitive_context_field(value: str) -> bool:
    """Match sensitive field aliases independent of case and separators."""

    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return (
        normalized in _BLOCKED_FIELDS
        or normalized.replace("_", "") in _BLOCKED_FIELD_TOKENS
    )

_PARAMETER_SEMANTICS: dict[str, dict[str, str]] = {
    "greenhouse": {
        "blend": (
            "base_prediction = blend * latest_observation + (1 - blend) * "
            "rolling_window_mean. Increasing blend gives more weight to the latest "
            "observation and less weight to the rolling mean."
        ),
        "window": (
            "Number of prior hourly observations in rolling_window_mean. A larger "
            "window uses a longer history; it does not make the latest observation "
            "more important."
        ),
        "bias_scale": (
            "learned_bias = mean(training_observation - base_prediction) * "
            "bias_scale, and learned_bias is added to each prediction. Zero disables "
            "the learned bias correction."
        ),
    },
    "greenhouse_ridge": {
        "history_steps": "Number of prior hourly feature steps supplied to the ridge model.",
        "ridge_alpha": "L2 regularization strength; increasing it shrinks fitted coefficients more.",
        "residual_scale": "Scale applied to the fitted residual correction; zero disables that correction.",
    },
    "greenhouse_targetwise_ridge": {
        "history_steps": "Number of prior hourly feature steps supplied to each ridge model.",
        "ridge_alpha": "Shared L2 regularization strength for the registered target models.",
        "air_temperature_residual_scale": "Scale for the fitted air-temperature residual; zero selects persistence.",
        "relative_humidity_residual_scale": "Scale for the fitted humidity residual; zero selects persistence.",
        "co2_concentration_residual_scale": "Scale for the fitted CO2 residual; zero selects persistence.",
    },
    "greenhouse_horizon_targetwise_ridge": {
        "history_steps": "Number of prior hourly feature steps supplied to each ridge model.",
        "ridge_alpha": "Shared L2 regularization strength for all registered target-horizon models.",
        "air_temperature_1h_residual_scale": "Scale for the 1-hour air-temperature residual; zero selects persistence for this cell.",
        "air_temperature_6h_residual_scale": "Scale for the 6-hour air-temperature residual; zero selects persistence for this cell.",
        "air_temperature_24h_residual_scale": "Scale for the 24-hour air-temperature residual; zero selects persistence for this cell.",
        "relative_humidity_1h_residual_scale": "Scale for the 1-hour humidity residual; zero selects persistence for this cell.",
        "relative_humidity_6h_residual_scale": "Scale for the 6-hour humidity residual; zero selects persistence for this cell.",
        "relative_humidity_24h_residual_scale": "Scale for the 24-hour humidity residual; zero selects persistence for this cell.",
        "co2_concentration_1h_residual_scale": "Scale for the 1-hour CO2 residual; zero selects persistence for this cell.",
        "co2_concentration_6h_residual_scale": "Scale for the 6-hour CO2 residual; zero selects persistence for this cell.",
        "co2_concentration_24h_residual_scale": "Scale for the 24-hour CO2 residual; zero selects persistence for this cell.",
    },
    "toy": {
        "alpha": "Bounded smoothing coefficient used by the toy predictor.",
        "window": "Number of prior observations used by the toy rolling summary.",
        "water_threshold": "Bounded soil-water decision threshold used by the toy predictor.",
    },
}


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _aggregate_value(value: Any, name: str, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > 4000:
            raise ValueError(f"{name} contains an overlong string")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return value
    if depth >= 6:
        raise ValueError(f"{name} is nested too deeply")
    if isinstance(value, Mapping):
        return {
            key: _aggregate_value(item, f"{name}.{key}", depth=depth + 1)
            for key, item in value.items()
            if isinstance(key, str)
            and key
            and not is_sensitive_context_field(key)
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 128:
            raise ValueError(f"{name} contains too many items")
        return [
            _aggregate_value(item, f"{name} item", depth=depth + 1)
            for item in value
        ]
    raise TypeError(f"{name} must contain only JSON-compatible summary values")


def batch_context(value: Mapping[str, Any] | None, run: Run) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("batch_context must be a mapping")
    allowed = {
        "generation",
        "slot_index",
        "batch_size",
        "round_parent_candidate_id",
        "previous_generation_analysis",
        "knowledge_snapshot",
        "knowledge_snapshot_digest",
        "research_iteration",
        "frozen_runtime_binding",
        "context_digest",
        "parent_genome_digest",
        "parent_genome_canonical_json",
        "stage_context_digests",
        "run_state_revision",
        "stage_attempt",
        "ledger_expected_revision",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            "batch_context contains unsupported fields: "
            + ", ".join(sorted(unknown))
        )
    generation = value.get("generation")
    slot_index = value.get("slot_index")
    batch_size = value.get("batch_size")
    if isinstance(generation, bool) or generation != run.generation:
        raise ValueError("batch_context generation does not match the run")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 8
    ):
        raise ValueError("batch_context batch_size must be between 1 and 8")
    if (
        isinstance(slot_index, bool)
        or not isinstance(slot_index, int)
        or not 0 <= slot_index < batch_size
    ):
        raise ValueError("batch_context slot_index is outside the batch")
    context_digest = _optional_text(
        value.get("context_digest"), "batch_context.context_digest"
    )
    if context_digest is None:
        raise ValueError("batch_context requires context_digest")
    previous = _aggregate_value(
        value.get("previous_generation_analysis"),
        "batch_context.previous_generation_analysis",
    )
    if previous is not None and not isinstance(previous, Mapping):
        raise TypeError("previous_generation_analysis must be an object or null")
    knowledge = validate_knowledge_context(value.get("knowledge_snapshot"))
    knowledge_digest = _optional_text(
        value.get("knowledge_snapshot_digest"),
        "batch_context.knowledge_snapshot_digest",
    )
    if knowledge is not None and knowledge.get("snapshot_digest") != knowledge_digest:
        raise ValueError("knowledge_snapshot digest does not match the frozen batch")
    research_iteration = None
    raw_research_iteration = value.get("research_iteration")
    if raw_research_iteration is not None:
        if not isinstance(raw_research_iteration, Mapping):
            raise TypeError("research_iteration must be an object or null")
        item = ResearchIteration.from_dict(raw_research_iteration)
        if item.run_id != run.run_id or item.generation != generation:
            raise ValueError("research_iteration is outside the batch scope")
        if item.knowledge_snapshot_digest != knowledge_digest:
            raise ValueError("research_iteration knowledge snapshot does not match")
        expected_analysis_digest = (
            previous.get("analysis_digest")
            if isinstance(previous, Mapping)
            else None
        )
        if item.source_analysis_digest != expected_analysis_digest:
            raise ValueError("research_iteration previous analysis does not match")
        research_iteration = item.to_dict()
    runtime_binding = _aggregate_value(
        value.get("frozen_runtime_binding"),
        "batch_context.frozen_runtime_binding",
    )
    if runtime_binding is not None and not isinstance(runtime_binding, Mapping):
        raise TypeError("frozen_runtime_binding must be an object or null")
    parent_digest = _optional_text(
        value.get("parent_genome_digest"),
        "batch_context.parent_genome_digest",
    )
    parent_canonical = _optional_text(
        value.get("parent_genome_canonical_json"),
        "batch_context.parent_genome_canonical_json",
    )
    raw_stage_digests = value.get("stage_context_digests")
    if raw_stage_digests == {} and parent_digest is None and parent_canonical is None:
        raw_stage_digests = None
    stage_digests: dict[str, str] | None = None
    if any(item is not None for item in (parent_digest, parent_canonical, raw_stage_digests)):
        if any(item is None for item in (parent_digest, parent_canonical, raw_stage_digests)):
            raise ValueError("batch_context parent genome binding is incomplete")
        if len(parent_digest) != 64 or not re.fullmatch(r"[0-9a-f]{64}", parent_digest):
            raise ValueError("batch_context.parent_genome_digest must be a SHA-256 digest")
        if not isinstance(raw_stage_digests, Mapping) or not raw_stage_digests:
            raise TypeError("batch_context.stage_context_digests must be a non-empty object")
        stage_digests = {}
        for raw_name, raw_digest in raw_stage_digests.items():
            name = _optional_text(raw_name, "batch_context stage context name")
            item = _optional_text(
                raw_digest,
                f"batch_context.stage_context_digests.{name}",
            )
            if name is None or item is None or not re.fullmatch(r"[0-9a-f]{64}", item):
                raise ValueError("batch_context stage context digests must be SHA-256 digests")
            stage_digests[name] = item

    revisions: dict[str, int] = {}
    for name in ("run_state_revision", "stage_attempt", "ledger_expected_revision"):
        raw_revision = value.get(name)
        if raw_revision is None:
            continue
        if isinstance(raw_revision, bool) or not isinstance(raw_revision, int) or raw_revision < 0:
            raise ValueError(f"batch_context.{name} must be a non-negative integer")
        revisions[name] = raw_revision
    return {
        "generation": generation,
        "slot_index": slot_index,
        "batch_size": batch_size,
        "round_parent_candidate_id": _optional_text(
            value.get("round_parent_candidate_id"),
            "batch_context.round_parent_candidate_id",
        ),
        "previous_generation_analysis": previous,
        "knowledge_snapshot": knowledge,
        "knowledge_snapshot_digest": knowledge_digest,
        "research_iteration": research_iteration,
        "frozen_runtime_binding": runtime_binding,
        "context_digest": context_digest,
        "parent_genome_digest": parent_digest,
        "parent_genome_canonical_json": parent_canonical,
        "stage_context_digests": stage_digests,
        **revisions,
    }


def safe_aggregate_feedback(
    value: Mapping[str, Any] | None,
    *,
    name: str,
) -> dict[str, Any] | None:
    """Project aggregate feedback while removing sample- and code-level fields."""

    if value is None:
        return None
    result = _aggregate_value(value, name)
    if not isinstance(result, Mapping):  # pragma: no cover - input contract above
        raise TypeError(f"{name} must be an object or null")
    return dict(result)


def analysis_focus_parameter(
    analysis: Mapping[str, Any] | None,
    schemas: Mapping[str, Mapping[str, Any]],
) -> str | None:
    if not isinstance(analysis, Mapping):
        return None
    effects = analysis.get("parameter_effects")
    if isinstance(effects, list):
        for item in effects:
            if isinstance(item, Mapping) and item.get("parameter") in schemas:
                return str(item["parameter"])
    weaknesses = analysis.get("target_weaknesses")
    target = ""
    horizon: int | None = None
    if isinstance(weaknesses, list) and weaknesses and isinstance(weaknesses[0], Mapping):
        target = str(weaknesses[0].get("target") or "").casefold()
        raw_horizon = weaknesses[0].get("horizon_hours")
        if (
            isinstance(raw_horizon, int)
            and not isinstance(raw_horizon, bool)
            and raw_horizon > 0
        ):
            horizon = raw_horizon
    horizon_specific = (
        ("co2", "co2_concentration"),
        ("humidity", "relative_humidity"),
        ("temperature", "air_temperature"),
    )
    if horizon is not None:
        for marker, prefix in horizon_specific:
            name = f"{prefix}_{horizon}h_residual_scale"
            if marker in target and name in schemas:
                return name
    preferred = (
        (
            "co2",
            (
                "co2_concentration_residual_scale",
                "residual_scale",
                "ridge_alpha",
                "history_steps",
                "bias_scale",
                "window",
            ),
        ),
        (
            "humidity",
            (
                "relative_humidity_residual_scale",
                "residual_scale",
                "history_steps",
                "ridge_alpha",
                "blend",
                "window",
            ),
        ),
        (
            "temperature",
            (
                "air_temperature_residual_scale",
                "residual_scale",
                "history_steps",
                "ridge_alpha",
                "window",
                "blend",
            ),
        ),
        ("water", ("water_threshold", "alpha", "window")),
    )
    for marker, names in preferred:
        if marker in target:
            return next((name for name in names if name in schemas), None)
    return None


def parameter_semantics(
    boundary: str,
    schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Return host-authored parameter meaning for the exact active boundary."""

    known = _PARAMETER_SEMANTICS.get(boundary, {})
    return {
        name: known.get(name, f"Bounded numeric parameter: {name}.")
        for name in schemas
    }


__all__ = [
    "analysis_focus_parameter",
    "batch_context",
    "is_sensitive_context_field",
    "parameter_semantics",
    "safe_aggregate_feedback",
]
