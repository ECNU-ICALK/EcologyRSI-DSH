"""Shared credential redaction and bounded public error summaries."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping
from typing import Any


REDACTED = "[REDACTED]"
REMOTE_REASON_INVALID = "remote_reason_invalid"

# Remote reason text is advisory and must never become a credential reflection
# channel. Only host-defined, semantically generic codes may cross a persistence
# boundary; arbitrary model-provided machine-looking strings are still rejected.
REMOTE_REASON_CODES = frozenset(
    {
        "accept_prediction",
        "baseline",
        "bounded_retry_recovered",
        "candidate_forecast",
        "critic_requested_tool",
        "critic_selects_alternate_tool",
        "critic_terminates_failed_tool",
        "critic_unavailable_connection",
        "critic_unavailable_constraint_rejected",
        "critic_unavailable_invalid_output",
        "critic_unavailable_numerical",
        "critic_unavailable_rate_limited",
        "critic_unavailable_remote_rejected",
        "critic_unavailable_remote_transient",
        "critic_unavailable_timeout",
        "critic_unavailable_tool_error",
        "fallback",
        "initial_registered_route",
        "insufficient_context",
        "persistence_baseline",
        "projection_repair",
        "registered_forecast",
        "registered_tool",
        "remote_review_accept",
        "repair_after_host_rejection",
        "request_registered_repair",
        "route",
        "stable_baseline",
        "terminate_after_tool_failure",
        "use_persistence_fallback",
        "use_projection_repair",
        "use_registered_forecast",
        "use_registered_tool",
    }
)

_SENSITIVE_KEY_TOKENS = frozenset(
    {
        "accesstoken",
        "apikey",
        "apikeys",
        "authtoken",
        "authorization",
        "authorizationtoken",
        "capabilitytoken",
        "clientsecret",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "idtoken",
        "password",
        "refreshtoken",
        "secret",
        "secrets",
        "servicetoken",
        "sessiontoken",
        "token",
    }
)
_SENSITIVE_KEY_SUFFIXES = (
    "apikey",
    "apikeys",
    "password",
    "secret",
    "secrets",
    "token",
    "credential",
    "credentials",
    "cookie",
    "cookies",
)
_SECRET_TEXT = re.compile(
    r"(?:\bBearer\s+[^\s,;]+|"
    r"(?:api(?:[\W_]*key)?|access[\W_]*token|auth[\W_]*token|"
    r"capability[\W_]*token|session[\W_]*token|service[\W_]*token|"
    r"refresh[\W_]*token|id[\W_]*token|authorization(?:[\W_]*token)?|"
    r"client[\W_]*secret|password|credential|secret)\s*[:=]\s*"
    r"(?:\"[^\"]+\"|'[^']+'|\S+))",
    re.IGNORECASE,
)
_EXCEPTION_TYPE_PREFIX = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.]{0,127})(?=\s*[:\[])"
)
_PRIVATE_LOCATION_TEXT = re.compile(
    r"(?:\b[a-z][a-z0-9+.-]*://|"
    r"(?<![A-Za-z0-9._-])/[A-Za-z0-9._~%+-][^\s,;]*|"
    r"\b[A-Za-z]:[\\/][^\s,;]+)",
    re.IGNORECASE,
)
_SAFE_ERROR_CODE = re.compile(r"[A-Za-z0-9_.-]{1,100}")


def folded_key(value: str) -> str:
    """Fold key spelling so case and punctuation cannot bypass redaction."""

    return "".join(character for character in value.casefold() if character.isalnum())


def is_sensitive_key(
    key: Any,
    *,
    extra_keys: Collection[str] = (),
) -> bool:
    """Return whether a mapping key can carry credentials or private content."""

    if not isinstance(key, str):
        return True
    normalized = folded_key(key)
    if not normalized:
        return True
    extra_tokens = {folded_key(item) for item in extra_keys}
    return (
        normalized in _SENSITIVE_KEY_TOKENS
        or normalized in extra_tokens
        or normalized.startswith("authorization")
        or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)
    )


def redact_sensitive_text(value: str, *, limit: int | None = None) -> str:
    """Redact a whole text value when it contains an inline credential."""

    result = REDACTED if _SECRET_TEXT.search(value) else value
    return result if limit is None else result[:limit]


def sanitize_public_value(
    value: Any,
    *,
    extra_sensitive_keys: Collection[str] = (),
    depth: int = 0,
    max_depth: int = 7,
    text_limit: int | None = 1000,
    mapping_limit: int | None = None,
    sequence_limit: int = 32,
) -> Any:
    """Copy a bounded JSON value while recursively removing sensitive fields."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return redact_sensitive_text(value, limit=text_limit)
    if depth >= max_depth:
        return None
    if isinstance(value, Mapping):
        items = value.items()
        if mapping_limit is not None:
            items = list(items)[:mapping_limit]
        return {
            key: sanitize_public_value(
                item,
                extra_sensitive_keys=extra_sensitive_keys,
                depth=depth + 1,
                max_depth=max_depth,
                text_limit=text_limit,
                mapping_limit=mapping_limit,
                sequence_limit=sequence_limit,
            )
            for key, item in items
            if isinstance(key, str)
            and not is_sensitive_key(key, extra_keys=extra_sensitive_keys)
        }
    if isinstance(value, (list, tuple)):
        return [
            sanitize_public_value(
                item,
                extra_sensitive_keys=extra_sensitive_keys,
                depth=depth + 1,
                max_depth=max_depth,
                text_limit=text_limit,
                mapping_limit=mapping_limit,
                sequence_limit=sequence_limit,
            )
            for item in value[:sequence_limit]
        ]
    return None


def safe_error_code(value: Any, fallback: str | None = None) -> str | None:
    """Accept only a compact identifier as a public error code."""

    candidate = str(value).strip() if isinstance(value, (str, int)) else ""
    if candidate and _SAFE_ERROR_CODE.fullmatch(candidate):
        return candidate
    return fallback


def safe_remote_reason_code(value: Any) -> str:
    """Map untrusted model rationale to a host-owned non-secret code."""

    candidate = value.strip() if isinstance(value, str) else ""
    if candidate in REMOTE_REASON_CODES:
        return candidate
    return REMOTE_REASON_INVALID


def public_exception_summary(exc: BaseException) -> str:
    """Describe an exception publicly without including its message."""

    type_name = safe_error_code(type(exc).__name__, "Exception") or "Exception"
    code = safe_error_code(
        getattr(exc, "error_code", None) or getattr(exc, "code", None)
    )
    return f"{type_name} [{code}]" if code is not None else type_name


def public_error_summary(value: Any, *, limit: int = 500) -> str | None:
    """Bound stored public errors and redact credentials in legacy messages."""

    if value is None:
        return None
    if isinstance(value, BaseException):
        return public_exception_summary(value)[:limit]
    text = str(value).strip()
    if not text:
        return None
    redacted = redact_sensitive_text(text, limit=limit)
    if redacted != REDACTED and not _PRIVATE_LOCATION_TEXT.search(redacted):
        return redacted
    match = _EXCEPTION_TYPE_PREFIX.match(text)
    if match is None:
        return REDACTED
    return (safe_error_code(match.group(1).rsplit(".", 1)[-1], "Exception") or "Exception")[
        :limit
    ]
