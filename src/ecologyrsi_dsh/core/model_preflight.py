"""Durable, run-bound evidence for the creation-time model transport gate.

Receipts are checked at their ledger recording time. Replaying a historical
run never requires a fresh external receipt and never upgrades a transport
check into a scientific qualification.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import re
from typing import Any

from .models import digest
from ..integrations.model_canary import (
    SCOPE, required_canary_identities, validate_canary_receipt,
)

LEGACY_AUDIT_SCHEMA = "ecologyrsi-dsh.model-contract-preflight-recorded/1"
AUDIT_SCHEMA = "ecologyrsi-dsh.model-contract-preflight-recorded/2"
AUDIT_METADATA_KEY = "model_contract_preflight_audit_schema"
AUDIT_EVENT = "ModelContractPreflightRecorded"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
_RECEIPT_FIELDS = (
    "schema_version", "receipt_id", "scope", "identity", "identity_digest",
    "bounds", "started_at", "completed_at", "expires_at", "passed",
    "output_schema_digest", "result_digest", "route_binding",
)
_TOOL_FIELDS = (
    "first_tool_call_verified", "order_verified", "successful_call_count",
    "next_tool_name", "stage", "source",
)
_USAGE_FIELDS = (
    "complete", "reported_tokens", "observed_session_count",
    "launched_session_count", "complete_session_count",
)


def preflight_audit_required(metadata: Mapping[str, Any]) -> bool:
    schema = metadata.get(AUDIT_METADATA_KEY)
    if schema is None:
        return False
    if (schema not in {LEGACY_AUDIT_SCHEMA, AUDIT_SCHEMA}
            or metadata.get("require_model_contract_preflight") is not True
            or metadata.get("execution_protocol") != "dsh_native_plugin_evolution@1"):
        raise ValueError("invalid model contract preflight audit policy")
    return True


def _audit_identities(metadata: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    identities = required_canary_identities(metadata)
    # Historical receipts prove only the roles checked at creation. Never
    # silently add critic evidence while replaying an immutable run.
    return identities[:2] if metadata.get(AUDIT_METADATA_KEY) == LEGACY_AUDIT_SCHEMA else identities


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("invalid model preflight audit timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("model preflight audit timestamp requires timezone")
    return result


def _safe_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Retain evidence fields only, never prompts, diagnostics or extra text."""
    receipt_id = receipt.get("receipt_id")
    if not isinstance(receipt_id, str) or not _IDENTIFIER.fullmatch(receipt_id):
        raise ValueError("invalid model preflight receipt id")
    result = {key: receipt.get(key) for key in _RECEIPT_FIELDS}
    result["tool_evidence"] = {key: (receipt.get("tool_evidence") or {}).get(key) for key in _TOOL_FIELDS}
    result["usage"] = {key: (receipt.get("usage") or {}).get(key) for key in _USAGE_FIELDS}
    attempts = receipt.get("attempts")
    if not isinstance(attempts, (list, tuple)):
        raise ValueError("invalid model preflight receipt attempts")
    result["attempts"] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError("invalid model preflight receipt attempt")
        session_id = attempt.get("session_id")
        number = attempt.get("attempt")
        if (not isinstance(session_id, str) or not _IDENTIFIER.fullmatch(session_id)
                or isinstance(number, bool) or not isinstance(number, int)
                or number != len(result["attempts"]) + 1
                or not isinstance(attempt.get("accepted"), bool)):
            raise ValueError("invalid model preflight attempt identity")
        result["attempts"].append({"attempt": number, "session_id": session_id,
                                   "accepted": attempt["accepted"]})
    result["failure"] = receipt.get("failure")
    return result


def build_preflight_audit(task: Any, run_id: str, receipts: Sequence[Mapping[str, Any]],
                          *, checked_at: str | None = None) -> dict[str, Any]:
    if not preflight_audit_required(task.metadata):
        raise ValueError("run does not opt into durable model preflight audit")
    checked_at = checked_at or datetime.now(timezone.utc).isoformat()
    now = _time(checked_at)
    identities = _audit_identities(task.metadata)
    if len(receipts) != len(identities):
        raise ValueError("model preflight requires every frozen role")
    safe = []
    for receipt, identity in zip(receipts, identities, strict=True):
        validate_canary_receipt(receipt, identity, now=now, require_passed=True)
        summary = _safe_receipt(receipt)
        validate_canary_receipt(summary, identity, now=now, require_passed=True)
        safe.append({"receipt": summary, "receipt_digest": digest(summary)})
    body = {"schema_version": task.metadata[AUDIT_METADATA_KEY], "scope": SCOPE, "run_id": run_id,
            "task_manifest_digest": task.digest, "checked_at": checked_at, "receipts": safe}
    return {**body, "audit_digest": digest(body)}


def validate_preflight_audit(payload: Mapping[str, Any], task: Any, run_id: str,
                             *, recorded_at: str, created_at: str) -> None:
    if set(payload) != {"schema_version", "scope", "run_id", "task_manifest_digest",
                        "checked_at", "receipts", "audit_digest"}:
        raise ValueError("model preflight audit fields are invalid")
    checked, recorded, created = map(_time, (payload["checked_at"], recorded_at, created_at))
    if checked < created or checked > recorded + timedelta(seconds=5):
        raise ValueError("model preflight audit is outside its recording boundary")
    entries = payload["receipts"]
    if not isinstance(entries, (list, tuple)) or any(
        not isinstance(entry, Mapping) or set(entry) != {"receipt", "receipt_digest"}
        for entry in entries
    ):
        raise ValueError("model preflight audit receipts are invalid")
    # The event timestamp is authoritative: a receipt must still be valid when
    # persisted, even when runtime setup took longer than the initial gate.
    expected = build_preflight_audit(task, run_id, [entry["receipt"] for entry in entries],
                                     checked_at=payload["checked_at"])
    for entry, identity in zip(entries, _audit_identities(task.metadata), strict=True):
        validate_canary_receipt(entry["receipt"], identity, now=recorded, require_passed=True)
    if digest(payload) != digest(expected):
        raise ValueError("model preflight audit identity or digest mismatch")


def model_preflight_projection(state: Any) -> dict[str, Any]:
    event = next((event for event in state.events if event.kind == AUDIT_EVENT), None)
    required = state.task_manifest.metadata.get("require_model_contract_preflight") is True
    if event is None:
        return {"status": "missing" if required else "not_required", "scope": SCOPE,
                "checked_at": None, "roles": [], "audit_digest": None,
                "audit_required": preflight_audit_required(state.task_manifest.metadata)}
    payload = event.payload
    roles = []
    for entry in payload["receipts"]:
        receipt = entry["receipt"]
        identity = receipt["identity"]
        roles.append({"role": identity["role"],
                      "model_id": f"{identity['provider_id']}/{identity['model_id']}",
                      "receipt_id": receipt["receipt_id"],
                      "checked_at": receipt["completed_at"], "expires_at": receipt["expires_at"],
                      "identity_digest": receipt["identity_digest"],
                      "receipt_digest": entry["receipt_digest"]})
    return {"status": "verified", "scope": SCOPE, "checked_at": payload["checked_at"],
            "roles": roles, "audit_digest": payload["audit_digest"], "audit_required": True}
