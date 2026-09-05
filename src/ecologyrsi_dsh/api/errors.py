"""Stable public error contract for HTTP and DSH adapters."""

from __future__ import annotations

from enum import Enum
from http import HTTPStatus
from typing import Any

from ..core.redaction import public_error_summary, safe_error_code


class ErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    COMMAND_IN_PROGRESS = "command_in_progress"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    PROVIDER_QUEUE_TIMEOUT = "provider_queue_timeout"
    DRAINING_TIMEOUT = "draining_timeout"
    STALE_MANIFEST = "stale_manifest"
    INTERNAL_ERROR = "internal_error"


_MESSAGES = {
    ErrorCode.INVALID_REQUEST: "invalid request",
    ErrorCode.NOT_FOUND: "resource not found",
    ErrorCode.COMMAND_IN_PROGRESS: "command is already in progress",
    ErrorCode.RUNTIME_UNAVAILABLE: "runtime is unavailable",
    ErrorCode.PROVIDER_QUEUE_TIMEOUT: "provider queue timed out",
    ErrorCode.DRAINING_TIMEOUT: "runtime draining timed out",
    ErrorCode.STALE_MANIFEST: "runtime manifest is stale",
    ErrorCode.INTERNAL_ERROR: "internal server error",
}


def error_payload(
    code: ErrorCode, *, retryable: bool, command_id: str | None = None
) -> dict[str, object]:
    """Build the sole stable public error shape."""

    code = ErrorCode(code)
    payload: dict[str, object] = {
        "error": _MESSAGES[code],
        "error_code": code.value,
        "retryable": bool(retryable),
    }
    if command_id is not None:
        payload["command_id"] = str(command_id)
    return payload


def error_code_for_exception(
    exc: BaseException, status: int | HTTPStatus | None = None
) -> ErrorCode:
    raw = safe_error_code(getattr(exc, "error_code", None))
    if raw:
        try:
            return ErrorCode(raw)
        except ValueError:
            pass
    name = type(exc).__name__.lower()
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return ErrorCode.NOT_FOUND
    if name in {"commandinprogressexception", "commandinprogresseerror"}:
        return ErrorCode.COMMAND_IN_PROGRESS
    if "stale" in name or "manifest" in name and "drift" in name:
        return ErrorCode.STALE_MANIFEST
    if "drain" in name and "timeout" in name:
        return ErrorCode.DRAINING_TIMEOUT
    if isinstance(exc, TimeoutError) or "queue" in name and "timeout" in name:
        return ErrorCode.PROVIDER_QUEUE_TIMEOUT
    if status is not None:
        status_value = int(status)
        if status_value == 404:
            return ErrorCode.NOT_FOUND
        if status_value == 503:
            return ErrorCode.RUNTIME_UNAVAILABLE
        if 400 <= status_value < 500:
            return ErrorCode.INVALID_REQUEST
    return ErrorCode.INTERNAL_ERROR


def public_error_payload(
    exc: BaseException,
    *,
    status: int | HTTPStatus | None = None,
    command_id: str | None = None,
    retryable: bool | None = None,
) -> dict[str, object]:
    """Map an exception while retaining only redacted human-readable text."""

    raw_code = safe_error_code(getattr(exc, "error_code", None))
    code = error_code_for_exception(exc, status)
    if retryable is None:
        retryable = bool(getattr(exc, "retryable", False)) or code in {
            ErrorCode.COMMAND_IN_PROGRESS,
            ErrorCode.RUNTIME_UNAVAILABLE,
            ErrorCode.PROVIDER_QUEUE_TIMEOUT,
            ErrorCode.DRAINING_TIMEOUT,
        }
    payload = error_payload(code, retryable=retryable, command_id=command_id)
    # Domain adapters may define additional, already-normalized public codes
    # (for example dsh_tool_*). Preserve those codes while keeping this module
    # the sole serializer and never exposing exception text as a code.
    if raw_code and raw_code not in {item.value for item in ErrorCode}:
        payload["error_code"] = raw_code
    summary = public_error_summary(str(exc))
    if summary:
        payload["error"] = summary
    return payload


__all__ = ["ErrorCode", "error_code_for_exception", "error_payload", "public_error_payload"]
