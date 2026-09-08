"""Shared causal and bounded request contracts for sample transports."""
from collections.abc import Mapping
from typing import Any
from datetime import datetime, timezone
from decimal import Decimal
import math
from ..core.models import canonical_json
from .sample_execution import SamplePredictionRequest, SampleExecutionContractError

_MAX_GATEWAY_PAYLOAD_BYTES = 4_000_000

_SAMPLE_OPERATION_MAX_TOKENS = frozenset(
    {"sample.planner", "sample.repair", "sample.critic"}
)

_ALWAYS_CRITIC_POLICY = "always@1"

_UNCERTAIN_OR_FAILURE_CRITIC_POLICY = "uncertain_or_failure@1"

_CAUSAL_PROVENANCE_SCHEMA_VERSION = "ecologyrsi-dsh.causal-sample-provenance/1"

_FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "actual",
        "actual_value",
        "ground_truth",
        "label",
        "labels",
        "observed",
        "observation",
        "target_value",
    }
)

_FORBIDDEN_OUTCOME_TOKENS = frozenset(
    "".join(character for character in name if character.isalnum())
    for name in _FORBIDDEN_OUTCOME_KEYS
)

def _normalized_operation_max_tokens(value: Mapping[str, int] | None) -> dict[str, int]:
    """Validate per-operation output limits frozen into a new task manifest."""

    if value is None:
        return {}
    if not isinstance(value, Mapping) or set(value) != _SAMPLE_OPERATION_MAX_TOKENS:
        raise ValueError(
            "operation_max_tokens must define sample.planner, sample.repair, and sample.critic"
        )
    normalized: dict[str, int] = {}
    for operation in sorted(_SAMPLE_OPERATION_MAX_TOKENS):
        max_tokens = value[operation]
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 512 <= max_tokens <= 8_192
        ):
            raise ValueError(
                f"operation_max_tokens.{operation} must be an integer between 512 and 8192"
            )
        normalized[operation] = max_tokens
    return normalized

def _normalized_remote_critic_policy(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the frozen policy controlling independent sample review."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("remote_critic_policy must be an object")
    version = value.get("version")
    if version == _ALWAYS_CRITIC_POLICY:
        if set(value) != {"version"}:
            raise ValueError("always remote critic policy must define only version")
        return {"version": _ALWAYS_CRITIC_POLICY}
    if set(value) != {"version", "min_planner_confidence"}:
        raise ValueError(
            "remote_critic_policy must define version and min_planner_confidence"
        )
    if version != _UNCERTAIN_OR_FAILURE_CRITIC_POLICY:
        raise ValueError(
            "remote_critic_policy.version must be always@1 or uncertain_or_failure@1"
        )
    threshold = value.get("min_planner_confidence")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0 <= float(threshold) <= 1
    ):
        raise ValueError(
            "remote_critic_policy.min_planner_confidence must be in [0, 1]"
        )
    return {
        "version": _UNCERTAIN_OR_FAILURE_CRITIC_POLICY,
        "min_planner_confidence": float(threshold),
    }

def _causal_wave_identity(
    request: SamplePredictionRequest,
    *,
    index: int,
) -> tuple[Any, ...]:
    """Return a verified origin wave, or a request-local singleton fallback."""

    singleton = ("singleton", index)
    origin = _normalized_timestamp(request.origin_timestamp)
    target = _normalized_timestamp(request.target_timestamp)
    provenance = request.label_free_context.get("causal_provenance")
    if provenance is None:
        return singleton
    if not isinstance(provenance, Mapping):
        raise SampleExecutionContractError(
            "sample causal_provenance must be an object"
        )
    if origin is None or target is None or origin[0] != target[0]:
        return singleton
    if not origin[1] < target[1]:
        raise SampleExecutionContractError(
            "sample causal provenance requires origin_timestamp before target_timestamp"
        )
    if provenance.get("schema_version") != _CAUSAL_PROVENANCE_SCHEMA_VERSION:
        raise SampleExecutionContractError(
            "sample causal_provenance schema is unsupported"
        )
    cutoff = _normalized_timestamp(provenance.get("origin_cutoff_timestamp"))
    latest = _normalized_timestamp(provenance.get("latest_context_timestamp"))
    if cutoff != origin:
        raise SampleExecutionContractError(
            "sample causal provenance does not match origin_timestamp"
        )
    if latest is None or latest[0] != origin[0] or latest[1] > origin[1]:
        raise SampleExecutionContractError(
            "sample context contains or claims information after its origin cutoff"
        )

    raw_history = request.label_free_context.get("history_window")
    raw_history_timestamps = provenance.get("history_timestamps")
    if raw_history is not None:
        if not isinstance(raw_history, (list, tuple)) or not isinstance(
            raw_history_timestamps, (list, tuple)
        ):
            raise SampleExecutionContractError(
                "sample history requires aligned causal timestamps"
            )
        if len(raw_history) != len(raw_history_timestamps):
            raise SampleExecutionContractError(
                "sample history and causal timestamps must have equal lengths"
            )
    elif raw_history_timestamps not in (None, [], ()):
        raise SampleExecutionContractError(
            "sample causal history timestamps require a history_window"
        )
    for raw_timestamp in raw_history_timestamps or ():
        history_timestamp = _normalized_timestamp(raw_timestamp)
        if (
            history_timestamp is None
            or history_timestamp[0] != origin[0]
            or history_timestamp[1] > origin[1]
        ):
            raise SampleExecutionContractError(
                "sample history contains a timestamp after its origin cutoff"
            )
    return ("verified_origin", *origin)

def _normalized_timestamp(value: Any) -> tuple[str, Any] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return ("number", Decimal(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return ("number", Decimal(str(value)))
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return ("datetime", parsed.astimezone(timezone.utc))

def _safe_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SampleExecutionContractError(f"{name} must be an object")
    normalized = _safe_value(value, name, depth=0)
    assert isinstance(normalized, dict)
    if len(canonical_json(normalized).encode("utf-8")) > _MAX_GATEWAY_PAYLOAD_BYTES:
        raise SampleExecutionContractError(f"{name} exceeds the gateway payload bound")
    return normalized

def _safe_value(value: Any, path: str, *, depth: int) -> Any:
    if depth > 10:
        raise SampleExecutionContractError(f"{path} is nested too deeply")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise SampleExecutionContractError(f"{path} keys must be non-empty text")
            key = raw_key.strip()
            normalized_key = key.casefold().replace("-", "_").replace(" ", "_")
            normalized_token = "".join(
                character for character in key.casefold() if character.isalnum()
            )
            if (
                normalized_key in _FORBIDDEN_OUTCOME_KEYS
                or normalized_token in _FORBIDDEN_OUTCOME_TOKENS
            ):
                raise SampleExecutionContractError(
                    f"{path} contains forbidden outcome field {key!r}"
                )
            result[key] = _safe_value(
                raw_item,
                f"{path}.{key}",
                depth=depth + 1,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _safe_value(item, f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SampleExecutionContractError(f"{path} must contain finite numbers")
        return value
    raise SampleExecutionContractError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )
