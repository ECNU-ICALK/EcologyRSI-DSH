"""Bounded real DSH tool/schema canaries, with independent expiring receipts.

This is a transport gate, not a scientific evaluation or a substitute for the
sample.plan prediction-tool admission chain. No evolution event is written.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from ..core.models import digest

REQUEST_SCHEMA = "ecologyrsi-dsh.model-contract-canary/1"
RECEIPT_SCHEMA = "ecologyrsi-dsh.model-contract-canary-receipt/1"
SCOPE = "tool_and_schema_transport_only"
_ROLES = (
    ("strategy_model_id", "resolved_policy_route_config_digest", "researcher", "ecology-researcher-v12", "generation.search-plan", "ecology-research-search-plan@1"),
    ("review_model_id", "resolved_review_route_config_digest", "generation-judge", "ecology-generation-judge-v8", "generation.reflect", "ecology-generation-reflection@1"),
    ("review_model_id", "resolved_review_route_config_digest", "sample-critic", "ecology-sample-critic-v5", "sample.critic", "ecology-sample-review@2"),
    ("strategy_model_id", "resolved_policy_route_config_digest", "sample-planner", "ecology-sample-planner-v9", "sample.plan", "ecology-sample-predictions@2"),
)
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_ROUTE_PART = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:@-]{0,119}$")


@dataclass(frozen=True)
class CanaryBounds:
    max_attempts: int = 4
    max_output_tokens: int = 2048
    max_reported_tokens: int = 30000
    # Raising this to the validated maximum of 180000 was measured against
    # pjlab/deepseek-v4-flash-0731 and changed nothing: its sample.critic probe
    # exhausted 120000 and 180000 alike while reporting ~1.4k tokens, so the
    # bound separates unusable transports from usable ones already.
    total_timeout_ms: int = 120000
    ttl_seconds: int = 3600

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("max_attempts", 1, 4), ("max_output_tokens", 512, 2048),
            ("max_reported_tokens", 1024, 50000), ("total_timeout_ms", 1000, 180000),
            ("ttl_seconds", 60, 86400),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"invalid model canary bound: {name}")


def required_canary_identities(metadata: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    """Derive the role transport identities from the Host-bound task metadata."""
    result = []
    for route_key, digest_key, role, preset, stage, schema in _ROLES:
        route = str(metadata.get(route_key) or "")
        provider, separator, model = route.partition("/")
        if not separator or not _ROUTE_PART.fullmatch(provider) or not _ROUTE_PART.fullmatch(model):
            raise ValueError(f"model canary requires a frozen provider/model route: {route_key}")
        identity = {
            "provider_id": provider, "model_id": model, "role": role,
            "preset_id": preset, "stage": stage, "output_schema_id": schema,
            "preset_content_digest": str(metadata.get("preset_content_digest") or ""),
            "standing_tool_surface_digest": str(metadata.get("standing_tool_surface_digest") or ""),
            "route_config_digest": str(metadata.get(digest_key) or ""),
        }
        _validate_identity(identity)
        result.append(identity)
    return tuple(result)


def _validate_identity(identity: Mapping[str, Any]) -> None:
    expected = {"provider_id", "model_id", "role", "preset_id", "stage", "output_schema_id", "preset_content_digest", "standing_tool_surface_digest", "route_config_digest"}
    if set(identity) != expected:
        raise ValueError("invalid model canary identity fields")
    if not all(isinstance(identity[k], str) and _ROUTE_PART.fullmatch(identity[k]) for k in ("provider_id", "model_id")):
        raise ValueError("invalid model canary provider/model")
    if not all(isinstance(identity[k], str) and _DIGEST.fullmatch(identity[k]) for k in ("preset_content_digest", "standing_tool_surface_digest", "route_config_digest")):
        raise ValueError("model canary requires Host-frozen content and route digests")
    if (identity["role"], identity["preset_id"], identity["stage"], identity["output_schema_id"]) not in {r[2:] for r in _ROLES}:
        raise ValueError("model canary stage unsupported")


def canary_request(identity: Mapping[str, Any], bounds: CanaryBounds | None = None) -> dict[str, Any]:
    _validate_identity(identity)
    return {"schema_version": REQUEST_SCHEMA, "identity": dict(identity), "bounds": asdict(bounds or CanaryBounds())}


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("invalid model canary timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("model canary timestamps must be timezone aware")
    return result


def validate_canary_receipt(receipt: Mapping[str, Any], identity: Mapping[str, Any], *, now: datetime | None = None, require_passed: bool = False) -> None:
    _validate_identity(identity)
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("scope") != SCOPE:
        raise ValueError("model canary receipt scope/schema mismatch")
    if receipt.get("identity") != dict(identity) or receipt.get("identity_digest") != digest(identity):
        raise ValueError("model canary receipt identity mismatch")
    bounds = CanaryBounds(**dict(receipt.get("bounds") or {}))
    started, completed, expires = (_timestamp(receipt.get(k)) for k in ("started_at", "completed_at", "expires_at"))
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or started > completed or completed > current + timedelta(seconds=5) or expires <= current or expires > completed + timedelta(seconds=bounds.ttl_seconds, milliseconds=1):
        raise ValueError("model canary receipt expired or timestamp invalid")
    attempts = receipt.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= bounds.max_attempts:
        raise ValueError("model canary receipt attempt count invalid")
    if not isinstance(receipt.get("passed"), bool):
        raise ValueError("model canary receipt outcome invalid")
    if require_passed and receipt["passed"] is not True:
        raise ValueError("model canary has not passed")
    if receipt["passed"]:
        ev = receipt.get("tool_evidence") or {}
        usage = receipt.get("usage") or {}
        count = usage.get("reported_tokens")
        observed = usage.get("observed_session_count")
        launched = usage.get("launched_session_count")
        complete = usage.get("complete_session_count")
        valid_coverage = (
            all(isinstance(value, int) and not isinstance(value, bool) for value in (observed, launched, complete))
            and 0 < observed == launched <= bounds.max_attempts and 0 <= complete <= launched
            and isinstance(usage.get("complete"), bool) and usage["complete"] == (complete == launched)
        )
        if (receipt.get("route_binding") != "role_host_explicit_provider_and_model"
            or not all(ev.get(k) is True for k in ("first_tool_call_verified", "order_verified"))
            or ev.get("source") != "dsh_session_event_log" or ev.get("stage") != identity["stage"]
            or ev.get("successful_call_count") != 1 or ev.get("next_tool_name") != "structured_output"
            or not _DIGEST.fullmatch(str(receipt.get("output_schema_digest") or ""))
            or not _DIGEST.fullmatch(str(receipt.get("result_digest") or ""))
            or not attempts[-1].get("accepted") or receipt.get("failure") is not None
            or not valid_coverage or isinstance(count, bool) or not isinstance(count, int)
            or not 0 < count < bounds.max_reported_tokens):
            raise ValueError("model canary passed without durable tool/schema/usage evidence")


class ModelCanaryReceiptStore:
    """Local trusted receipt directory; never accepts user-uploaded attestations."""
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _path(self, identity: Mapping[str, Any]) -> Path:
        _validate_identity(identity)
        return self.directory / f"{digest(identity)}.json"

    def save(self, receipt: Mapping[str, Any]) -> Path:
        identity = receipt.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("model canary receipt has no identity")
        validate_canary_receipt(receipt, identity)
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._path(identity)
        temporary = self.directory / f".{uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                os.chmod(temporary, 0o600)
                json.dump(receipt, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def require_fresh(self, identity: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
        path = self._path(identity)
        if path.stat().st_size > 131072:
            raise ValueError("model canary receipt exceeds bound")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("model canary receipt is not an object")
        validate_canary_receipt(receipt, identity, now=now, require_passed=True)
        return receipt


def require_model_preflight(metadata: Mapping[str, Any], receipt_directory: str | Path, *, now: datetime | None = None) -> tuple[dict[str, Any], ...]:
    """Read-only gate for new runs; an absent/expired/failed receipt fails closed."""
    store = ModelCanaryReceiptStore(receipt_directory)
    return tuple(store.require_fresh(identity, now=now) for identity in required_canary_identities(metadata))


def run_preflight(client: Any, *, metadata: Mapping[str, Any], receipt_directory: str | Path, bounds: CanaryBounds | None = None, force: bool = False) -> dict[str, Any]:
    """Check four execution roles, stopping at the first exhausted role.

    The bounds apply per role; all four roles share no additional retries.
    At most 4*max_attempts children and 4*total_timeout_ms are permitted.
    """
    store = ModelCanaryReceiptStore(receipt_directory)
    receipts = []
    for identity in required_canary_identities(metadata):
        cached = None
        if not force:
            try:
                cached = store.require_fresh(identity)
            except (OSError, ValueError, TypeError):
                pass
        if cached is not None:
            receipts.append(cached)
            continue
        request = canary_request(identity, bounds)
        receipt = client.run_canary(request)
        validate_canary_receipt(receipt, identity)
        # The client is a trusted local runtime seam, not a model response.
        store.save(receipt)
        receipts.append(receipt)
        if receipt["passed"] is not True:
            break
    return {"scope": SCOPE, "passed": len(receipts) == len(_ROLES) and all(r["passed"] for r in receipts), "receipts": receipts,
            "unsupported_scope": ["scientific_qualification", "research_semantic_quality", "sample.plan_prediction_tool_admission"]}


__all__ = ["CanaryBounds", "ModelCanaryReceiptStore", "required_canary_identities", "canary_request", "validate_canary_receipt", "require_model_preflight", "run_preflight"]
