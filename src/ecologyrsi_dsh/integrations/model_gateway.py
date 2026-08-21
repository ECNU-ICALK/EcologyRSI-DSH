"""Authenticated, bounded access to OpenAI-compatible policy models.

The gateway owns credentials and network access.  Callers receive only a
redacted catalog and strictly validated JSON objects, never raw model output.
"""

from __future__ import annotations

import errno
import json
import math
import os
import random
import re
import socket
import ssl
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
from threading import Lock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..core.models import ExpertUncertaintyType, digest
from ..core.redaction import (
    REMOTE_REASON_CODES,
    public_error_summary,
    redact_sensitive_text,
    safe_remote_reason_code,
)
from ..knowledge.algorithm_ir import validate_bounded_algorithm_synthesis
from .dsh_directory import discover_model_entries
from .gateway_url_policy import assess_gateway_url
from .model_bindings import MODEL_BINDING_SCHEMA_VERSION, canonical_model_roles


_ALLOWED_FINISH_REASONS = frozenset(
    {
        "stop",
        "length",
        "max_tokens",
        "content_filter",
        "tool_calls",
        "function_call",
    }
)
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})
_RETRYABLE_STAGE_OUTPUT_ERROR_CODES = frozenset(
    {
        "output_truncated",
        "response_choices_invalid",
        "response_content_type_invalid",
        "response_envelope_invalid",
        "response_format_invalid",
        "response_message_missing",
        "final_content_missing",
    }
)
_SAFE_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
_LEGACY_RECOVERABLE_RESPONSE_MARKERS = (
    "model response content must contain one json object",
    "model response content must contain only valid json",
    "model response final content must contain one json object",
    "sample decision response must contain exactly one decision per sample",
    "sample decision response must use every input sample_id exactly once",
)


class GatewayConfigurationError(ValueError):
    """Raised when a configured model connection is unsafe or incomplete."""


class GatewayResponseError(RuntimeError):
    """Raised when a policy-model call fails or violates its response contract.

    ``retryable`` describes the original failure class for orchestration code.
    Network retries remain owned by :class:`ModelGateway`; callers can use the
    flag to avoid replaying an entire generation after a definite 4xx or JSON
    contract error.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        attempts: int = 1,
        status_code: int | None = None,
        error_code: str = "gateway_response_error",
        split_eligible: bool = False,
        finish_reason: str | None = None,
        usage: Mapping[str, int] | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.attempts = max(1, int(attempts))
        self.status_code = status_code
        normalized_error_code = str(error_code).strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", normalized_error_code):
            raise ValueError("gateway response error_code must be a bounded machine code")
        self.error_code = normalized_error_code
        self.split_eligible = bool(split_eligible)
        self.finish_reason = (
            finish_reason if finish_reason in _ALLOWED_FINISH_REASONS else None
        )
        self.usage = _safe_token_usage(usage)
        if retry_after_seconds is None:
            self.retry_after_seconds = None
        else:
            retry_after = _nonnegative_retry_value(
                retry_after_seconds, "retry_after_seconds"
            )
            self.retry_after_seconds = min(
                retry_after, _MAX_SERVER_RETRY_AFTER_SECONDS
            )


def gateway_error_in_chain(
    exc: BaseException,
    *,
    retryable_only: bool = False,
    max_depth: int = 32,
) -> GatewayResponseError | None:
    """Find a gateway response error retained inside an exception boundary.

    Evaluation adapters and persistence callbacks may wrap the original
    gateway exception to add context.  Orchestration must still see the
    gateway's retry classification; otherwise a transient provider outage can
    be mistaken for a permanent local failure.  The traversal is bounded and
    cycle-safe because third-party exceptions can expose arbitrary cause and
    context chains (and Python 3.11 exception groups).
    """

    if not isinstance(exc, BaseException):
        return None
    pending: list[tuple[BaseException, int]] = [(exc, 0)]
    seen: set[int] = set()
    while pending:
        current, depth = pending.pop()
        identity = id(current)
        if identity in seen or depth > max_depth:
            continue
        seen.add(identity)
        if isinstance(current, GatewayResponseError):
            if not retryable_only or current.retryable:
                return current
        for related in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(related, BaseException):
                pending.append((related, depth + 1))
        exceptions = getattr(current, "exceptions", None)
        if isinstance(exceptions, (tuple, list)):
            for related in exceptions:
                if isinstance(related, BaseException):
                    pending.append((related, depth + 1))
    return None


_UNSET = object()


def _wrap_gateway_error(
    message: str,
    exc: GatewayResponseError,
    *,
    retryable: bool | None = None,
    attempts: int | None = None,
    status_code: int | None | object = _UNSET,
    error_code: str | None = None,
    split_eligible: bool | None = None,
    finish_reason: str | None | object = _UNSET,
    usage: Mapping[str, int] | None | object = _UNSET,
    retry_after_seconds: float | None | object = _UNSET,
) -> GatewayResponseError:
    """Reclassify an error without losing gateway retry/response metadata.

    Several validation layers need to provide a more specific public error
    code, but orchestration still relies on the original ``retryable`` and
    request metadata.  Centralizing this construction prevents wrappers from
    accidentally turning a transient gateway failure into a permanent one.
    """

    return GatewayResponseError(
        message,
        retryable=exc.retryable if retryable is None else retryable,
        attempts=exc.attempts if attempts is None else attempts,
        status_code=(exc.status_code if status_code is _UNSET else status_code),
        error_code=exc.error_code if error_code is None else error_code,
        split_eligible=(
            exc.split_eligible if split_eligible is None else split_eligible
        ),
        finish_reason=(
            exc.finish_reason if finish_reason is _UNSET else finish_reason
        ),
        usage=exc.usage if usage is _UNSET else usage,
        retry_after_seconds=(
            exc.retry_after_seconds
            if retry_after_seconds is _UNSET
            else retry_after_seconds
        ),
    )


def _retryable_stage_output_error(exc: GatewayResponseError) -> bool:
    """Return whether a completed call failed only at model-output decoding."""

    return exc.error_code in _RETRYABLE_STAGE_OUTPUT_ERROR_CODES


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GatewayConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _safe_gateway_url(
    value: Any,
    *,
    allow_insecure_http: bool = False,
) -> tuple[str, bool]:
    assessment = assess_gateway_url(
        value,
        allow_insecure_http=allow_insecure_http,
    )
    if assessment.gateway_error is not None:
        raise GatewayConfigurationError(assessment.gateway_error)
    return assessment.url, assessment.insecure_http_exception


def _chat_completions_url(gateway_url: str) -> str:
    if gateway_url.endswith("/chat/completions"):
        return gateway_url
    return f"{gateway_url}/chat/completions"


_DEFAULT_MODEL_TIMEOUT = 900.0
_DEFAULT_MODEL_MAX_ATTEMPTS = 4
_DEFAULT_RETRY_BASE_SECONDS = 1.0
_DEFAULT_RETRY_MAX_SECONDS = 600.0
_MAX_SERVER_RETRY_AFTER_SECONDS = 3600.0
_MAX_INLINE_SERVER_RETRY_AFTER_SECONDS = 15.0
_SAMPLE_DECISION_MAX_BATCH = 128
_MIN_SAMPLE_OPERATION_MAX_TOKENS = 512
_MAX_SAMPLE_OPERATION_MAX_TOKENS = 8_192
_RESEARCH_PLAN_INITIAL_MAX_TOKENS = 16_384
_RESEARCH_PLAN_MAX_TOKENS = 32_768
_COMPACT_JSON_RETRY_INSTRUCTION = (
    "The previous final output was incomplete or invalid. Return exactly one "
    "complete, compact JSON object. Preserve every field required by the response "
    "contract; omit optional detail and shorten optional strings before required "
    "content. Emit no Markdown or explanatory text."
)
_ORIGIN_SHARED_CONTEXT_PROFILE = "origin_shared_context@1"
_ORIGIN_SHARED_CONTEXT_SCHEMA = "ecologyrsi-dsh.origin-shared-sample-context/1"
# OpenAI-compatible chat tokenizers are byte based, so prompt tokens cannot
# outnumber the UTF-8 bytes in the serialized request.  Reserve an additional
# fixed allowance for provider-specific chat framing that is not represented
# in the JSON body.
_CHAT_TOKEN_FRAMING_UPPER_BOUND = 1_024
_SAMPLE_AGENT_ROLE_TO_MODEL_ROLE = {
    "planner": "propose",
    "repair": "propose",
    # Keep the first experimental role name callable while newer runtimes use
    # planner/repair explicitly.
    "forecast_agent": "propose",
    "critic": "judge",
}
_FORBIDDEN_REMOTE_SAMPLE_FIELDS = frozenset(
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
_FORBIDDEN_REMOTE_SAMPLE_FIELD_TOKENS = frozenset(
    name.replace("_", "") for name in _FORBIDDEN_REMOTE_SAMPLE_FIELDS
)
_RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429})
_RETRYABLE_OS_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }
)
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_ALGORITHM_BLUEPRINT_VERSION = "ecologyrsi-dsh.algorithm-blueprint-request/1"
_ALGORITHM_SYNTHESIS_REQUIREMENT_VERSION = (
    "ecologyrsi-dsh.algorithm-synthesis-requirement/1"
)
_ALGORITHM_SYNTHESIS_DEGRADATION_VERSION = (
    "ecologyrsi-dsh.algorithm-synthesis-degradation/1"
)


def _positive_timeout(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GatewayConfigurationError(f"{name} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise GatewayConfigurationError(f"{name} must be a positive number")
    return number


def _model_timeout_from_env(environ: Mapping[str, str], explicit: float | None) -> float:
    if explicit is not None:
        return _positive_timeout(explicit, "timeout")
    raw = str(environ.get("ECOLOGYRSI_DSH_MODEL_TIMEOUT", "")).strip()
    if not raw:
        return _DEFAULT_MODEL_TIMEOUT
    try:
        value: Any = float(raw)
    except ValueError as exc:
        raise GatewayConfigurationError(
            "ECOLOGYRSI_DSH_MODEL_TIMEOUT must be a positive number"
        ) from exc
    return _positive_timeout(value, "ECOLOGYRSI_DSH_MODEL_TIMEOUT")


def _bounded_integer_setting(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = str(environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise GatewayConfigurationError(
            f"{name} must be an integer between {minimum} and {maximum}"
        ) from exc
    if value < minimum or value > maximum:
        raise GatewayConfigurationError(
            f"{name} must be an integer between {minimum} and {maximum}"
        )
    return value


def _nonnegative_number_setting(
    environ: Mapping[str, str], name: str, default: float
) -> float:
    raw = str(environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise GatewayConfigurationError(f"{name} must be a non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise GatewayConfigurationError(f"{name} must be a non-negative number")
    return value


def _nonnegative_retry_value(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return number


@dataclass(frozen=True, slots=True)
class ModelConnection:
    model_id: str
    gateway_url: str
    model: str
    token: str | None = None
    label: str | None = None
    roles: tuple[str, ...] = ("propose", "judge")
    allow_insecure_http: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _required_text(self.model_id, "model_id"))
        if not isinstance(self.allow_insecure_http, bool):
            raise GatewayConfigurationError("allow_insecure_http must be a boolean")
        gateway_url, insecure_http_exception = _safe_gateway_url(
            self.gateway_url,
            allow_insecure_http=self.allow_insecure_http,
        )
        object.__setattr__(self, "gateway_url", gateway_url)
        object.__setattr__(
            self,
            "allow_insecure_http",
            insecure_http_exception,
        )
        object.__setattr__(self, "model", _required_text(self.model, "model"))
        if self.token is not None:
            object.__setattr__(self, "token", _required_text(self.token, "token"))
        label = self.model_id if self.label is None else _required_text(self.label, "label")
        object.__setattr__(self, "label", label)
        if not isinstance(self.roles, (tuple, list)):
            raise GatewayConfigurationError("roles must be a non-empty array")
        try:
            roles = canonical_model_roles(self.roles)
        except ValueError as exc:
            raise GatewayConfigurationError(str(exc)) from exc
        object.__setattr__(self, "roles", roles)

    @property
    def configuration_digest(self) -> str:
        """Identify the callable model configuration without hashing credentials."""

        return digest(
            {
                "schema_version": MODEL_BINDING_SCHEMA_VERSION,
                "model_id": self.model_id,
                "gateway_url": self.gateway_url,
                "model": self.model,
                "roles": sorted(self.roles),
                "allow_insecure_http": self.allow_insecure_http,
            }
        )

    @property
    def credential_fingerprint(self) -> str | None:
        """Bind persisted verification to a credential without exposing it."""

        if self.token is None:
            return None
        return digest(
            {
                "schema_version": "ecologyrsi-dsh.credential-binding/1",
                "configuration_digest": self.configuration_digest,
                "credential": self.token,
            }
        )


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        # Model credentials are origin-bound.  urllib otherwise copies the
        # Authorization header to a redirect target, including another
        # loopback port, so every redirect is rejected and must be configured
        # explicitly as the gateway URL.
        return None


class ModelGateway:
    """Load policy-model connections and execute bounded JSON operations."""

    def __init__(
        self,
        connections: Sequence[ModelConnection],
        *,
        timeout: float = _DEFAULT_MODEL_TIMEOUT,
        max_attempts: int = _DEFAULT_MODEL_MAX_ATTEMPTS,
        retry_base_seconds: float = _DEFAULT_RETRY_BASE_SECONDS,
        retry_max_seconds: float = _DEFAULT_RETRY_MAX_SECONDS,
        verification_store: Any | None = None,
        directory_unavailable: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a positive number")
        timeout_number = float(timeout)
        if not math.isfinite(timeout_number) or timeout_number <= 0:
            raise ValueError("timeout must be a positive number")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise TypeError("max_attempts must be an integer between 1 and 8")
        if max_attempts < 1 or max_attempts > 8:
            raise ValueError("max_attempts must be an integer between 1 and 8")
        retry_base = _nonnegative_retry_value(
            retry_base_seconds, "retry_base_seconds"
        )
        retry_max = _nonnegative_retry_value(
            retry_max_seconds, "retry_max_seconds"
        )
        if retry_max < retry_base:
            raise ValueError(
                "retry_max_seconds must be greater than or equal to retry_base_seconds"
            )
        indexed: dict[str, ModelConnection] = {}
        for connection in connections:
            if not isinstance(connection, ModelConnection):
                raise TypeError("connections must contain ModelConnection values")
            if connection.model_id in indexed:
                raise GatewayConfigurationError(f"duplicate model_id: {connection.model_id}")
            indexed[connection.model_id] = connection
        self._connections = indexed
        self.timeout = timeout_number
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base
        self.retry_max_seconds = retry_max
        self._status: dict[str, dict[str, Any]] = {
            model_id: {"state": "configured", "last_checked_at": None, "last_error": None}
            for model_id in indexed
        }
        # Explicit verification is a service-process trust decision.  Keep it
        # separate from the latest request health so a transient proposal or
        # judge timeout does not rewrite a previously verified model as if it
        # had never been verified.
        self._verified: dict[str, bool] = {model_id: False for model_id in indexed}
        self._verification_persisted: dict[str, bool] = {
            model_id: False for model_id in indexed
        }
        self._verification_store = verification_store
        self._request_diagnostics: dict[str, dict[str, Any] | None] = {
            model_id: None for model_id in indexed
        }
        self._directory_unavailable = tuple(
            self._unavailable_catalog_item(item) for item in directory_unavailable
        )
        self._lock = Lock()
        self._opener = build_opener(_NoRedirectHandler())
        self._sleep = time.sleep
        self._random = random.random
        self._restore_verifications()

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        timeout: float | None = None,
        verification_store: Any | None = None,
    ) -> ModelGateway:
        env = os.environ if environ is None else environ
        resolved_timeout = _model_timeout_from_env(env, timeout)
        retry_options = {
            "max_attempts": _bounded_integer_setting(
                env,
                "ECOLOGYRSI_DSH_MODEL_MAX_ATTEMPTS",
                _DEFAULT_MODEL_MAX_ATTEMPTS,
                minimum=1,
                maximum=8,
            ),
            "retry_base_seconds": _nonnegative_number_setting(
                env,
                "ECOLOGYRSI_DSH_MODEL_RETRY_BASE_SECONDS",
                _DEFAULT_RETRY_BASE_SECONDS,
            ),
            "retry_max_seconds": _nonnegative_number_setting(
                env,
                "ECOLOGYRSI_DSH_MODEL_RETRY_MAX_SECONDS",
                _DEFAULT_RETRY_MAX_SECONDS,
            ),
        }
        if retry_options["retry_max_seconds"] < retry_options["retry_base_seconds"]:
            raise GatewayConfigurationError(
                "ECOLOGYRSI_DSH_MODEL_RETRY_MAX_SECONDS must be greater than or equal to "
                "ECOLOGYRSI_DSH_MODEL_RETRY_BASE_SECONDS"
            )
        raw_catalog = env.get("ECOLOGYRSI_DSH_MODELS_JSON", "").strip()
        if raw_catalog:
            try:
                entries = json.loads(raw_catalog)
            except json.JSONDecodeError as exc:
                raise GatewayConfigurationError("ECOLOGYRSI_DSH_MODELS_JSON must be valid JSON") from exc
            if not isinstance(entries, list):
                raise GatewayConfigurationError("ECOLOGYRSI_DSH_MODELS_JSON must be a JSON array")
            connections = [cls._connection_from_entry(item, env, index) for index, item in enumerate(entries)]
            return cls(
                connections,
                timeout=resolved_timeout,
                **retry_options,
                verification_store=verification_store,
            )

        gateway_url = env.get("ECOLOGYRSI_DSH_GATEWAY_URL", "").strip()
        model = env.get("ECOLOGYRSI_DSH_MODEL", "").strip()
        token = env.get("ECOLOGYRSI_DSH_TOKEN", "").strip() or None
        if not gateway_url and not model and token is None:
            discovered = discover_model_entries(environ)
            if discovered:
                runnable = [
                    item
                    for item in discovered
                    if item.get("directory_available") is not False
                ]
                unavailable = [
                    item
                    for item in discovered
                    if item.get("directory_available") is False
                ]
                connections = [
                    cls._connection_from_entry(
                        {
                            key: value
                            for key, value in item.items()
                            if key
                            in {
                                "id",
                                "model_id",
                                "gateway_url",
                                "url",
                                "model",
                                "token",
                                "secret",
                                "api_key",
                                "api_key_env",
                                "label",
                                "roles",
                            }
                        },
                        env,
                        index,
                        allow_insecure_http=bool(item.get("allow_insecure_http")),
                    )
                    for index, item in enumerate(runnable)
                ]
                return cls(
                    connections,
                    timeout=resolved_timeout,
                    **retry_options,
                    verification_store=verification_store,
                    directory_unavailable=unavailable,
                )
            return cls(
                (),
                timeout=resolved_timeout,
                **retry_options,
                verification_store=verification_store,
            )
        if not gateway_url or not model:
            raise GatewayConfigurationError(
                "ECOLOGYRSI_DSH_GATEWAY_URL and ECOLOGYRSI_DSH_MODEL must be configured together"
            )
        return cls(
            (ModelConnection(model_id=model, gateway_url=gateway_url, model=model, token=token),),
            timeout=resolved_timeout,
            **retry_options,
            verification_store=verification_store,
        )

    from_environment = from_env

    @staticmethod
    def _connection_from_entry(
        value: Any,
        env: Mapping[str, str],
        index: int,
        *,
        allow_insecure_http: bool = False,
    ) -> ModelConnection:
        if not isinstance(value, Mapping):
            raise GatewayConfigurationError(f"model catalog item {index} must be an object")
        allowed = {
            "id",
            "model_id",
            "gateway_url",
            "url",
            "model",
            "token",
            "secret",
            "api_key",
            "api_key_env",
            "label",
            "roles",
        }
        unknown = set(value) - allowed
        if unknown:
            raise GatewayConfigurationError(
                f"model catalog item {index} has unsupported fields: {', '.join(sorted(unknown))}"
            )
        model = _required_text(value.get("model"), f"model catalog item {index}.model")
        model_id = value.get("model_id", value.get("id", model))
        gateway_url = value.get("gateway_url", value.get("url"))
        token_values = [value.get("token"), value.get("secret"), value.get("api_key")]
        configured_tokens = [item for item in token_values if item not in (None, "")]
        if len(configured_tokens) > 1:
            raise GatewayConfigurationError(f"model catalog item {index} configures multiple secrets")
        token = configured_tokens[0] if configured_tokens else None
        api_key_env = value.get("api_key_env")
        if api_key_env not in (None, ""):
            if token is not None:
                raise GatewayConfigurationError(
                    f"model catalog item {index} cannot combine a secret with api_key_env"
                )
            env_name = _required_text(api_key_env, f"model catalog item {index}.api_key_env")
            token = env.get(env_name, "").strip() or None
            if token is None:
                raise GatewayConfigurationError(f"credential environment variable is not set: {env_name}")
        raw_roles = value.get("roles", ("propose", "judge"))
        if not isinstance(raw_roles, (list, tuple)):
            raise GatewayConfigurationError(f"model catalog item {index}.roles must be an array")
        return ModelConnection(
            model_id=_required_text(model_id, f"model catalog item {index}.model_id"),
            gateway_url=gateway_url,
            model=model,
            token=token,
            label=value.get("label"),
            roles=tuple(raw_roles),
            allow_insecure_http=allow_insecure_http,
        )

    def catalog(self) -> list[dict[str, Any]]:
        """Return a redacted, UI-safe connection catalog."""

        self._refresh_verifications()
        with self._lock:
            statuses = {key: dict(value) for key, value in self._status.items()}
            verified = dict(self._verified)
            persisted = dict(self._verification_persisted)
            request_diagnostics = {
                key: dict(value) if value is not None else None
                for key, value in self._request_diagnostics.items()
            }
        configured_catalog = [
            {
                "model_id": item.model_id,
                "label": item.label,
                "model": item.model,
                "roles": list(item.roles),
                "configuration_digest": item.configuration_digest,
                "binding_source": "server_gateway_configuration",
                "credential_configured": item.token is not None,
                "configured": item.token is not None,
                "directory_available": True,
                "execution_available": item.token is not None,
                "authentication_verified": (
                    item.token is not None and verified[item.model_id]
                ),
                "verification_persisted": persisted[item.model_id],
                "authentication_state": (
                    "verified"
                    if item.token is not None and verified[item.model_id]
                    else "verification_failed"
                    if item.token is not None
                    and _connection_phase(
                        statuses[item.model_id], request_diagnostics[item.model_id]
                    )
                    == "permanent_error"
                    else "configured_unverified"
                    if item.token is not None
                    else "missing_credential"
                ),
                # Compatibility field.  Unlike the first draft, it now means
                # a successful authenticated request, not merely token presence.
                "authenticated": (
                    item.token is not None and verified[item.model_id]
                ),
                "available": (
                    item.token is not None
                    and verified[item.model_id]
                    and statuses[item.model_id]["state"] == "available"
                ),
                "connection_phase": _connection_phase(
                    statuses[item.model_id], request_diagnostics[item.model_id]
                ),
                "temporarily_unavailable": _connection_phase(
                    statuses[item.model_id], request_diagnostics[item.model_id]
                )
                in {"retrying", "transient_error"},
                "api_invalid": _connection_phase(
                    statuses[item.model_id], request_diagnostics[item.model_id]
                )
                == "permanent_error",
                "connection": {
                    **statuses[item.model_id],
                    "request_policy": self._request_policy(),
                    "last_request": request_diagnostics[item.model_id],
                },
            }
            for item in self._connections.values()
        ]
        return configured_catalog + [dict(item) for item in self._directory_unavailable]

    @staticmethod
    def _unavailable_catalog_item(value: Mapping[str, Any]) -> dict[str, Any]:
        model_id = _required_text(
            value.get("model_id", value.get("id")), "directory model_id"
        )
        model = _required_text(value.get("model"), "directory model")
        raw_roles = value.get("roles", ("propose", "judge"))
        if not isinstance(raw_roles, (list, tuple)):
            raw_roles = ("propose", "judge")
        try:
            roles = list(canonical_model_roles(raw_roles))
        except ValueError:
            roles = ["propose", "judge"]
        reason = value.get("unavailable_reason")
        if not isinstance(reason, Mapping):
            reason = {
                "code": "directory_unavailable",
                "message": "The DSH directory entry is not callable by this backend.",
            }
        safe_reason = {
            "code": str(reason.get("code") or "directory_unavailable")[:80],
            "message": str(reason.get("message") or "Directory entry unavailable")[:300],
        }
        return {
            "model_id": model_id,
            "label": str(value.get("label") or model_id),
            "model": model,
            "roles": roles,
            "configuration_digest": digest(
                {
                    "schema_version": "ecologyrsi-dsh.unavailable-model-binding/1",
                    "model_id": model_id,
                    "model": model,
                    "roles": roles,
                    "reason": safe_reason["code"],
                }
            ),
            "binding_source": "dsh_directory_unavailable",
            "credential_configured": any(
                value.get(key) not in (None, "")
                for key in ("token", "secret", "api_key")
            ),
            "configured": False,
            "directory_available": False,
            "authentication_verified": False,
            "verification_persisted": False,
            "authentication_state": "unavailable",
            "authenticated": False,
            "available": False,
            "connection_phase": "unavailable",
            "temporarily_unavailable": False,
            "api_invalid": True,
            "unavailable_reason": safe_reason,
            "connection": {
                "state": "unavailable",
                "last_checked_at": None,
                "last_error": safe_reason["code"],
            },
        }

    def connection_status(self, model_id: str) -> dict[str, Any]:
        self._connection(model_id)
        self._refresh_verification(model_id)
        with self._lock:
            status = dict(self._status[model_id])
            diagnostics = self._request_diagnostics[model_id]
        status["request_policy"] = self._request_policy()
        status["last_request"] = (
            dict(diagnostics) if diagnostics is not None else None
        )
        return status

    def _request_policy(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout,
            "max_attempts": self.max_attempts,
            "retry_base_seconds": self.retry_base_seconds,
            "retry_max_seconds": self.retry_max_seconds,
            "server_retry_after_cap_seconds": _MAX_SERVER_RETRY_AFTER_SECONDS,
            "inline_retry_after_cap_seconds": (
                _MAX_INLINE_SERVER_RETRY_AFTER_SECONDS
            ),
            "retryable_http_statuses": [408, 425, 429, "5xx"],
        }

    def configuration_digest(self, model_id: str) -> str:
        """Return the stable, credential-free identity bound into a run."""

        return self._connection(model_id).configuration_digest

    def verify_connection(self, model_id: str) -> dict[str, Any]:
        """Verify credentials and the bounded JSON chat contract end to end."""

        connection = self._connection(model_id)
        if connection.token is None:
            raise GatewayConfigurationError(
                f"policy model {model_id} has no authentication credential"
            )
        try:
            content = self._chat(
                model_id,
                operation="connection.verify",
                payload={
                    "purpose": "Verify authenticated access and JSON response compatibility."
                },
                response_contract={"ok": "boolean; must be true"},
            )
            _reject_unknown_fields(
                content, {"ok"}, "connection verification response"
            )
            if content.get("ok") is not True:
                raise GatewayResponseError(
                    "connection verification response.ok must be true"
                )
        except GatewayResponseError as exc:
            # A queue timeout or transient upstream outage is not evidence that
            # credentials became invalid.  Only a definite request/contract
            # failure revokes explicit verification.
            if not exc.retryable:
                self._set_status(model_id, "error", _public_error(exc))
                self._set_verified(model_id, False)
            raise
        self._set_verified(model_id, True)
        return self.connection_status(model_id)

    def propose(
        self,
        model_id: str,
        context: Mapping[str, Any],
        allowed_parameters: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        schemas = _normalize_parameter_schemas(allowed_parameters)
        try:
            content = self._chat(
                model_id,
                operation="propose",
                payload={"context": _json_object(context, "context"), "allowed_parameters": schemas},
                response_contract={
                    "parameters": "object containing only allowed parameter names",
                    "title": "optional string",
                    "rationale": "optional string",
                    "guidance": "optional string",
                },
            )
        except GatewayResponseError as exc:
            if not _retryable_stage_output_error(exc):
                raise
            raise _wrap_gateway_error(
                str(exc),
                exc,
                retryable=True,
                error_code="proposal_response_contract_invalid",
            ) from exc
        try:
            allowed_fields = {"parameters", "title", "rationale", "guidance"}
            _reject_unknown_fields(content, allowed_fields, "proposal response")
            if "parameters" not in content:
                raise GatewayResponseError("proposal response must contain parameters")
            parameters = _validate_parameters(content["parameters"], schemas, partial=True)
            result: dict[str, Any] = {"parameters": parameters}
            for name in ("title", "rationale", "guidance"):
                if name in content:
                    result[name] = _response_text(
                        content[name], f"proposal response.{name}"
                    )
            return result
        except GatewayResponseError as exc:
            raise _wrap_gateway_error(
                str(exc),
                exc,
                retryable=True,
                error_code="proposal_response_contract_invalid",
            ) from exc

    def research_plan(
        self,
        model_id: str,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Ask a strategy model for a bounded, advisory implementation plan.

        The plan is deliberately separate from ``propose``.  A model may
        research candidate teams, predictor families, and search strategies,
        but the host only records the description and can adopt identifiers
        that are present in its own evaluator registry.  No source code,
        shell command, or hidden/private chain-of-thought is accepted here.
        """

        normalized_context = _json_object(context, "context")
        # This is a host-owned decision derived from the frozen catalogs.  A
        # caller-provided value must never be able to weaken the response
        # contract sent to the research model.
        normalized_context.pop("algorithm_synthesis_requirement", None)
        normalized_context.pop("algorithm_synthesis_correction", None)
        synthesis_requirement = _research_algorithm_synthesis_requirement(
            normalized_context
        )
        if synthesis_requirement is not None:
            normalized_context["algorithm_synthesis_requirement"] = (
                synthesis_requirement
            )
        requirement_mode = (
            synthesis_requirement.get("mode")
            if synthesis_requirement is not None
            else None
        )
        algorithm_section_requirement = {
            "synthesis_required": "required",
            "degradation_required": "must be omitted",
        }.get(requirement_mode, "optional")
        degradation_section_requirement = {
            "synthesis_required": "must be omitted",
            "degradation_required": "required",
        }.get(requirement_mode, "optional")
        response_contract = {
            "team": "optional object with name and roles",
            "prediction_model": "optional object describing a predictor family",
            "strategy": "optional object describing the search strategy",
            "research": (
                "optional array of source-summary objects using only source, "
                "title, url, finding, and relevance string fields"
            ),
            "algorithm_blueprint": (
                algorithm_section_requirement
                + " object selecting one exact pipeline from "
                "context.algorithm_blueprint_catalog, with schema_version, "
                "pipeline_id, ordered operator_ids, parameter_names, and "
                "evidence_refs drawn from context.knowledge_snapshot.evidence_catalog"
            ),
            "algorithm_synthesis": (
                algorithm_section_requirement
                + " evidence-backed design over algorithm_blueprint with "
                "schema_version ecologyrsi-dsh.algorithm-synthesis/1, matching "
                "pipeline_id, cited evidence_refs, registered parameter_focus, "
                "and a bounded rationale; never source code"
            ),
            "algorithm_synthesis_degradation": (
                degradation_section_requirement
                + " object used only when the host requirement says no compatible "
                "executable evidence exists; must contain schema_version, "
                "reason_code no_compatible_executable_evidence, and rationale"
            ),
            "implementation_notes": "optional string",
            "confidence": "optional number between 0 and 1",
            "expert_consultation": (
                "optional non-blocking object used only for material uncertainty; "
                "must contain uncertainty_type, question, context, "
                "fallback_assumption, requested_expertise, options, confidence, "
                "and non_blocking=true; it cannot request tools, data, or permissions"
            ),
        }
        request_context = normalized_context
        for semantic_attempt in range(1, 3):
            content = self._chat(
                model_id,
                operation="research_plan",
                payload={"context": request_context},
                response_contract=response_contract,
                max_tokens=_RESEARCH_PLAN_INITIAL_MAX_TOKENS,
            )
            try:
                _validate_research_algorithm_response_requirement(
                    content,
                    synthesis_requirement,
                )
            except GatewayResponseError as exc:
                if semantic_attempt >= 2:
                    # A bounded semantic correction budget is distinct from a
                    # transport retry.  The provider still returned two valid
                    # JSON responses, so this is not evidence that its route or
                    # credentials are permanently broken.  Let orchestration
                    # release the worker and retry the research stage after a
                    # durable cooldown instead of terminating the whole run.
                    raise _wrap_gateway_error(
                        "research plan algorithm synthesis contract failed after "
                        f"{semantic_attempt} semantic attempts: {exc}",
                        exc,
                        retryable=True,
                        attempts=semantic_attempt,
                        error_code="research_algorithm_contract_invalid",
                    ) from exc
                request_context = {
                    **normalized_context,
                    "algorithm_synthesis_correction": {
                        "schema_version": (
                            "ecologyrsi-dsh.algorithm-synthesis-correction/1"
                        ),
                        "attempt": semantic_attempt + 1,
                        "previous_contract_error": str(exc)[:500],
                        "instruction": (
                            "Return a new complete response satisfying the host-owned "
                            "algorithm_synthesis_requirement. Do not reuse an omitted "
                            "or conflicting algorithm section."
                        ),
                    },
                }
                continue
            break
        allowed = {
            "team",
            "prediction_model",
            "strategy",
            "research",
            "algorithm_blueprint",
            "algorithm_synthesis",
            "algorithm_synthesis_degradation",
            "implementation_notes",
            "confidence",
            "expert_consultation",
        }
        _reject_unknown_fields(content, allowed, "research plan response")
        result: dict[str, Any] = {}
        if "team" in content:
            result["team"] = _project_bounded_plan_object(
                content["team"],
                "team",
                allowed_fields={"id", "name", "roles", "rationale"},
                list_fields={"roles"},
            )
        if "prediction_model" in content:
            result["prediction_model"] = _project_bounded_plan_object(
                content["prediction_model"],
                "prediction_model",
                allowed_fields={
                    "id",
                    "name",
                    "family",
                    "implementation",
                    "rationale",
                    "parameter_names",
                },
                list_fields={"parameter_names"},
            )
        if "strategy" in content:
            result["strategy"] = _project_bounded_plan_object(
                content["strategy"],
                "strategy",
                allowed_fields={"id", "name", "steps", "rationale"},
                list_fields={"steps"},
            )
        if "research" in content:
            raw_research = content["research"]
            if not isinstance(raw_research, list):
                raise GatewayResponseError("research plan response.research must be an array")
            if len(raw_research) > 16:
                raise GatewayResponseError("research plan response.research contains too many items")
            result["research"] = []
            for item in raw_research:
                if not isinstance(item, Mapping):
                    continue
                projected = _project_bounded_research_summary(
                    item,
                )
                if projected:
                    result["research"].append(projected)
        if "algorithm_blueprint" in content:
            result["algorithm_blueprint"] = _validate_research_algorithm_blueprint(
                content["algorithm_blueprint"],
                context=normalized_context,
            )
            prediction_model = result.get("prediction_model")
            requested_predictor = (
                str(prediction_model.get("id") or "").strip()
                if isinstance(prediction_model, Mapping)
                else ""
            )
            if requested_predictor and requested_predictor != result[
                "algorithm_blueprint"
            ]["pipeline_id"]:
                raise GatewayResponseError(
                    "research plan algorithm_blueprint conflicts with prediction_model.id"
                )
        if "algorithm_synthesis" in content:
            blueprint = result.get("algorithm_blueprint")
            if not isinstance(blueprint, Mapping):
                raise GatewayResponseError(
                    "research plan algorithm_synthesis requires algorithm_blueprint"
                )
            try:
                knowledge_snapshot = normalized_context.get("knowledge_snapshot")
                evidence_catalog = (
                    knowledge_snapshot.get("evidence_catalog")
                    if isinstance(knowledge_snapshot, Mapping)
                    else ()
                )
                frozen_evidence_refs = tuple(
                    str(item.get("knowledge_id")).strip()
                    for item in evidence_catalog
                    if isinstance(item, Mapping)
                    and isinstance(item.get("knowledge_id"), str)
                    and str(item.get("knowledge_id")).strip()
                )
                result["algorithm_synthesis"] = validate_bounded_algorithm_synthesis(
                    content["algorithm_synthesis"],
                    algorithm_blueprint=blueprint,
                    allowed_evidence_refs=frozen_evidence_refs,
                )
            except (TypeError, ValueError) as exc:
                raise GatewayResponseError(
                    f"research plan algorithm_synthesis is invalid: {exc}"
                ) from exc
        if "algorithm_synthesis_degradation" in content:
            result["algorithm_synthesis_degradation"] = (
                _validate_algorithm_synthesis_degradation(
                    content["algorithm_synthesis_degradation"]
                )
            )
        if "implementation_notes" in content:
            result["implementation_notes"] = _response_text(
                content["implementation_notes"],
                "research plan response.implementation_notes",
            )[:4000]
        if "confidence" in content:
            confidence = content["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise GatewayResponseError("research plan response.confidence must be a number")
            if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
                raise GatewayResponseError("research plan response.confidence must be between 0 and 1")
            result["confidence"] = float(confidence)
        if "expert_consultation" in content:
            try:
                result["expert_consultation"] = _validate_expert_consultation_request(
                    content["expert_consultation"]
                )
            except GatewayResponseError:
                # A malformed optional question must not discard an otherwise
                # valid, host-bounded research plan. It is omitted rather than
                # persisted or sent to an expert.
                pass
        return result

    def sample_decide(
        self,
        model_id: str,
        *,
        role: str,
        samples: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        available_tools: Sequence[Mapping[str, Any]],
        max_tokens: int | None = None,
        _sample_format_retry_limit: int = 1,
        _attempts_out: list[int] | None = None,
        _usage_out: list[dict[str, int]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Request bounded, label-free decisions for one sample micro-batch.

        This is deliberately a planning API.  The remote model can select a
        host-registered tool for each unlabeled sample, but it never receives
        observed values and cannot execute tools itself.  The executor owns
        repair, scoring, and failure accounting.
        """

        if (
            isinstance(_sample_format_retry_limit, bool)
            or not isinstance(_sample_format_retry_limit, int)
            or not 0 <= _sample_format_retry_limit <= 1
        ):
            raise ValueError("sample format retry limit must be 0 or 1")
        if _attempts_out is not None and not isinstance(_attempts_out, list):
            raise TypeError("attempts output must be a list")
        if _usage_out is not None and not isinstance(_usage_out, list):
            raise TypeError("usage output must be a list")
        if max_tokens is not None:
            if (
                isinstance(max_tokens, bool)
                or not isinstance(max_tokens, int)
                or not _MIN_SAMPLE_OPERATION_MAX_TOKENS
                <= max_tokens
                <= _MAX_SAMPLE_OPERATION_MAX_TOKENS
            ):
                raise ValueError("sample max_tokens must be an integer between 512 and 8192")

        if not isinstance(role, str) or role.strip() not in _SAMPLE_AGENT_ROLE_TO_MODEL_ROLE:
            allowed = ", ".join(sorted(_SAMPLE_AGENT_ROLE_TO_MODEL_ROLE))
            raise ValueError(f"sample decision role must be one of: {allowed}")
        normalized_role = role.strip()
        compact_critic = (
            normalized_role == "critic"
            and isinstance(context, Mapping)
            and context.get("critic_prompt_profile")
            == "uncertain_or_failure_compact@1"
        )
        if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
            raise TypeError("samples must be a sequence of JSON objects")
        if not samples:
            raise ValueError("samples must not be empty")
        if len(samples) > _SAMPLE_DECISION_MAX_BATCH:
            raise ValueError(
                f"samples must contain at most {_SAMPLE_DECISION_MAX_BATCH} items"
            )
        if isinstance(available_tools, (str, bytes)) or not isinstance(available_tools, Sequence):
            raise TypeError("available_tools must be a sequence of JSON objects")
        if not available_tools:
            raise ValueError("available_tools must not be empty")

        normalized_samples: list[dict[str, Any]] = []
        sample_ids: list[str] = []
        for index, sample in enumerate(samples):
            normalized = _label_free_remote_object(sample, f"samples[{index}]")
            sample_id = normalized.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"samples[{index}].sample_id must be a non-empty string")
            if len(sample_id.strip()) > 300:
                raise ValueError(f"samples[{index}].sample_id is too long")
            normalized["sample_id"] = sample_id.strip()
            normalized_samples.append(normalized)
            sample_ids.append(sample_id.strip())
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("samples must not contain duplicate sample_id values")
        normalized_context = _label_free_remote_object(context, "context")
        origin_shared_context = _validate_origin_shared_context_references(
            normalized_samples,
            normalized_context,
        )

        normalized_tools: list[dict[str, Any]] = []
        tool_ids: set[str] = set()
        for index, tool in enumerate(available_tools):
            normalized = _label_free_remote_object(tool, f"available_tools[{index}]")
            tool_id = normalized.get("tool_id")
            if not isinstance(tool_id, str) or not tool_id.strip():
                raise ValueError(f"available_tools[{index}].tool_id must be a non-empty string")
            if len(tool_id.strip()) > 160:
                raise ValueError(f"available_tools[{index}].tool_id is too long")
            if tool_id.strip() in tool_ids:
                raise ValueError("available_tools must not contain duplicate tool_id values")
            normalized["tool_id"] = tool_id.strip()
            tool_ids.add(tool_id.strip())
            normalized_tools.append(normalized)

        content = self._chat(
            model_id,
            operation=f"sample.{normalized_role}",
            payload={
                "samples": normalized_samples,
                "context": normalized_context,
                "available_tools": normalized_tools,
                "allowed_tool_ids": [tool["tool_id"] for tool in normalized_tools],
                "allowed_reason_codes": sorted(REMOTE_REASON_CODES),
                "decision_policy": _sample_decision_policy(
                    normalized_role,
                    compact_critic=compact_critic,
                    origin_shared_context=origin_shared_context,
                ),
            },
            response_contract={
                "decisions": (
                    "array with exactly one object per input sample: sample_id string, "
                    "next_tool copied character-for-character from input.allowed_tool_ids, "
                    "reason_code copied character-for-character from "
                    "input.allowed_reason_codes, and confidence number between 0 and 1; "
                    "version is metadata and must never be appended to next_tool"
                )
            },
            sample_format_retry_limit=_sample_format_retry_limit,
            attempts_out=_attempts_out,
            usage_out=_usage_out,
            max_tokens=max_tokens,
        )
        try:
            _reject_forbidden_remote_fields(
                content, "sample decision response", response=True
            )
        except GatewayResponseError as exc:
            # The decoded decision content is still rejected. A smaller batch may
            # avoid one malformed decision without treating the provider as bad.
            raise _wrap_gateway_error(
                "sample decision response contains a forbidden outcome field",
                exc,
                error_code="sample_decisions_forbidden_field",
                split_eligible=True,
            ) from exc
        try:
            _reject_unknown_fields(content, {"decisions"}, "sample decision response")
        except GatewayResponseError as exc:
            raise _wrap_gateway_error(
                "sample decision response fields do not match the contract",
                exc,
                error_code="sample_decisions_format_invalid",
                split_eligible=True,
            ) from exc
        raw_decisions = content.get("decisions")
        if not isinstance(raw_decisions, list):
            raise GatewayResponseError(
                "sample decision response.decisions must be an array",
                error_code="sample_decisions_format_invalid",
                split_eligible=True,
            )
        if len(raw_decisions) != len(sample_ids):
            raise GatewayResponseError(
                "sample decision response must contain exactly one decision per sample",
                error_code="sample_decisions_incomplete",
                split_eligible=True,
            )

        decisions_by_id: dict[str, dict[str, Any]] = {}
        for index, raw_decision in enumerate(raw_decisions):
            if not isinstance(raw_decision, Mapping):
                raise GatewayResponseError(
                    f"sample decision response.decisions[{index}] must be an object",
                    error_code="sample_decision_format_invalid",
                    split_eligible=True,
                )
            try:
                _reject_unknown_fields(
                    raw_decision,
                    {"sample_id", "next_tool", "reason_code", "confidence"},
                    f"sample decision response.decisions[{index}]",
                )
            except GatewayResponseError as exc:
                raise _wrap_gateway_error(
                    "sample decision response decision fields do not match the contract",
                    exc,
                    error_code="sample_decision_format_invalid",
                    split_eligible=True,
                ) from exc
            required = {"sample_id", "next_tool", "reason_code", "confidence"}
            missing = required - set(raw_decision)
            if missing:
                raise GatewayResponseError(
                    "sample decision response decision is missing fields: "
                    + ", ".join(sorted(missing)),
                    error_code="sample_decision_format_invalid",
                    split_eligible=True,
                )
            try:
                sample_id = _response_text(
                    raw_decision["sample_id"],
                    f"sample decision response.decisions[{index}].sample_id",
                )
            except GatewayResponseError as exc:
                raise _wrap_gateway_error(
                    "sample decision response sample_id is invalid",
                    exc,
                    error_code="sample_decision_mapping_invalid",
                    split_eligible=True,
                ) from exc
            if sample_id not in sample_ids or sample_id in decisions_by_id:
                raise GatewayResponseError(
                    "sample decision response must use every input sample_id exactly once",
                    error_code="sample_decision_mapping_invalid",
                    split_eligible=True,
                )
            try:
                next_tool = _response_text(
                    raw_decision["next_tool"],
                    f"sample decision response.decisions[{index}].next_tool",
                )
            except GatewayResponseError as exc:
                raise _wrap_gateway_error(
                    "sample decision response next_tool is invalid",
                    exc,
                    error_code="sample_decision_tool_invalid",
                    split_eligible=True,
                ) from exc
            if next_tool not in tool_ids:
                raise GatewayResponseError(
                    "sample decision response selects an unavailable tool",
                    error_code="sample_decision_tool_invalid",
                    split_eligible=True,
                )
            reason_code = safe_remote_reason_code(raw_decision["reason_code"])
            confidence = raw_decision["confidence"]
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1
            ):
                raise GatewayResponseError(
                    "sample decision response decision confidence must be between 0 and 1",
                    error_code="sample_decision_confidence_invalid",
                    split_eligible=True,
                )
            decisions_by_id[sample_id] = {
                "sample_id": sample_id,
                "next_tool": next_tool,
                "reason_code": reason_code,
                "confidence": float(confidence),
            }
        if set(decisions_by_id) != set(sample_ids):
            raise GatewayResponseError(
                "sample decision response must use every input sample_id exactly once",
                error_code="sample_decision_mapping_invalid",
                split_eligible=True,
            )
        return {"decisions": [decisions_by_id[sample_id] for sample_id in sample_ids]}

    def sample_decide_with_diagnostics(
        self,
        model_id: str,
        *,
        role: str,
        samples: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        available_tools: Sequence[Mapping[str, Any]],
        allow_format_retry: bool,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int | bool]]:
        """Return one decision batch with call-local attempt and token receipts."""

        if not isinstance(allow_format_retry, bool):
            raise TypeError("allow_format_retry must be a boolean")
        attempts: list[int] = []
        usages: list[dict[str, int]] = []
        try:
            result = self.sample_decide(
                model_id,
                role=role,
                samples=samples,
                context=context,
                available_tools=available_tools,
                _sample_format_retry_limit=1 if allow_format_retry else 0,
                _attempts_out=attempts,
                _usage_out=usages,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            http_attempts = attempts[-1] if attempts else 0
            setattr(exc, "http_attempts", http_attempts)
            setattr(
                exc,
                "token_usage",
                _token_usage_receipt(usages, http_attempts=http_attempts),
            )
            if not isinstance(exc, GatewayResponseError):
                raise
            if attempts:
                exc.attempts = max(exc.attempts, attempts[-1])
            raise
        http_attempts = attempts[-1] if attempts else 0
        return result, {
            "http_attempts": http_attempts,
            **_token_usage_receipt(usages, http_attempts=http_attempts),
        }

    def sample_decide_token_upper_bound(
        self,
        model_id: str,
        *,
        role: str,
        samples: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        available_tools: Sequence[Mapping[str, Any]],
        max_tokens: int | None = None,
    ) -> int:
        """Return a conservative aggregate bound for one logical sample call.

        One logical call can replay the same HTTP request during transport or
        narrow format recovery.  Bounding every possible attempt is necessary
        because a response can be billed even when its connection is lost
        before usage metadata reaches the client.
        """

        normalized_role = str(role).strip()
        if normalized_role not in _SAMPLE_AGENT_ROLE_TO_MODEL_ROLE:
            raise ValueError("unsupported sample decision role")
        resolved_max_tokens = (
            _MAX_SAMPLE_OPERATION_MAX_TOKENS
            if max_tokens is None
            else max_tokens
        )
        if (
            isinstance(resolved_max_tokens, bool)
            or not isinstance(resolved_max_tokens, int)
            or not _MIN_SAMPLE_OPERATION_MAX_TOKENS
            <= resolved_max_tokens
            <= _MAX_SAMPLE_OPERATION_MAX_TOKENS
        ):
            raise ValueError(
                "sample max_tokens must be an integer between 512 and 8192"
            )
        compact_critic = (
            normalized_role == "critic"
            and isinstance(context, Mapping)
            and context.get("critic_prompt_profile")
            == "uncertain_or_failure_compact@1"
        )
        normalized_samples = [
            _label_free_remote_object(sample, f"samples[{index}]")
            for index, sample in enumerate(samples)
        ]
        normalized_context = _label_free_remote_object(context, "context")
        origin_shared_context = _validate_origin_shared_context_references(
            normalized_samples,
            normalized_context,
        )
        normalized_tools = [
            _label_free_remote_object(tool, f"available_tools[{index}]")
            for index, tool in enumerate(available_tools)
        ]
        operation = f"sample.{normalized_role}"
        body = _chat_request_body(
            self._connection(model_id),
            operation=operation,
            payload={
                "samples": normalized_samples,
                "context": normalized_context,
                "available_tools": normalized_tools,
                "allowed_tool_ids": [
                    str(tool.get("tool_id", "")).strip()
                    for tool in normalized_tools
                ],
                "allowed_reason_codes": sorted(REMOTE_REASON_CODES),
                "decision_policy": _sample_decision_policy(
                    normalized_role,
                    compact_critic=compact_critic,
                    origin_shared_context=origin_shared_context,
                ),
            },
            response_contract={
                "decisions": (
                    "array with exactly one object per input sample: sample_id string, "
                    "next_tool copied character-for-character from input.allowed_tool_ids, "
                    "reason_code copied character-for-character from "
                    "input.allowed_reason_codes, and confidence number between 0 and 1; "
                    "version is metadata and must never be appended to next_tool"
                )
            },
            max_tokens=resolved_max_tokens,
        )
        per_attempt = (
            len(body)
            + resolved_max_tokens
            + _CHAT_TOKEN_FRAMING_UPPER_BOUND
        )
        return self.max_attempts * per_attempt

    def judge(
        self,
        model_id: str,
        proposal: Mapping[str, Any],
        aggregate_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            content = self._chat(
                model_id,
                operation="judge",
                payload={
                    "proposal": _json_object(proposal, "proposal"),
                    "aggregate_metrics": _json_object(aggregate_metrics, "aggregate_metrics"),
                },
                response_contract={
                    "accepted": "boolean",
                    "guidance": "optional string",
                    "parameter_override": "optional JSON object; validated by the strategy router",
                },
            )
        except GatewayResponseError as exc:
            if not _retryable_stage_output_error(exc):
                raise
            raise _wrap_gateway_error(
                str(exc),
                exc,
                retryable=True,
                error_code="judge_response_contract_invalid",
            ) from exc
        try:
            _reject_unknown_fields(
                content,
                {"accepted", "guidance", "parameter_override"},
                "judge response",
            )
            if not isinstance(content.get("accepted"), bool):
                raise GatewayResponseError("judge response.accepted must be a boolean")
            result: dict[str, Any] = {"accepted": content["accepted"]}
            if "guidance" in content:
                try:
                    result["guidance"] = _response_text(
                        content["guidance"], "judge response.guidance"
                    )
                except GatewayResponseError:
                    # Guidance is advisory metadata.  Preserve the mandatory
                    # accept/reject decision when a gateway serializes this
                    # optional field with an unexpected type.
                    pass
            if "parameter_override" in content:
                override = content["parameter_override"]
                if isinstance(override, Mapping):
                    try:
                        result["parameter_override"] = _json_object(
                            override, "judge response.parameter_override"
                        )
                    except (TypeError, ValueError):
                        # The strategy router applies its own parameter allowlist;
                        # a malformed optional override must never invalidate a
                        # completed scientific evaluation.
                        pass
            return result
        except GatewayResponseError as exc:
            raise _wrap_gateway_error(
                str(exc),
                exc,
                retryable=True,
                error_code="judge_response_contract_invalid",
            ) from exc

    def _connection(self, model_id: str) -> ModelConnection:
        try:
            return self._connections[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown policy model: {model_id}") from exc

    def _chat(
        self,
        model_id: str,
        *,
        operation: str,
        payload: Mapping[str, Any],
        response_contract: Mapping[str, Any],
        sample_format_retry_limit: int = 0,
        attempts_out: list[int] | None = None,
        usage_out: list[dict[str, int]] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        connection = self._connection(model_id)
        # Another sidecar process may have verified this exact callable and
        # credential since this instance started.  Refresh before any health
        # update so this instance neither reports stale trust nor attempts to
        # overwrite it with its old in-memory value.
        self._refresh_verification(model_id)
        required_role = _required_model_role(operation)
        if operation != "connection.verify" and required_role not in connection.roles:
            raise GatewayConfigurationError(
                f"policy model {model_id} does not allow the {operation} role"
            )
        current_max_tokens = max_tokens
        body = _chat_request_body(
            connection,
            operation=operation,
            payload=payload,
            response_contract=response_contract,
            max_tokens=current_max_tokens,
        )
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if connection.token is not None:
            headers["Authorization"] = f"Bearer {connection.token}"
        request = Request(
            _chat_completions_url(connection.gateway_url),
            data=body,
            headers=headers,
            method="POST",
        )
        # Calls can legitimately wait behind other model jobs.  Retries remain
        # request-local and bounded: only throttling, timeout, and transient
        # gateway failures are replayed; authentication, routing, and response
        # contract errors fail immediately.
        started_at = _now()
        retries: list[dict[str, Any]] = []
        sample_format_retries = 0
        self._record_request_diagnostics(
            model_id,
            operation=operation,
            started_at=started_at,
            attempts=0,
            outcome="in_progress",
            retries=retries,
            last_error=None,
            next_retry_seconds=None,
            classification="none",
        )
        for attempt in range(1, self.max_attempts + 1):
            if attempts_out is not None:
                attempts_out[:] = [attempt]
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    raw = response.read()
                envelope = json.loads(raw.decode("utf-8"))
                content, response_metadata = _extract_message_content(envelope)
            except (
                IncompleteRead,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                retryable = _is_retryable_gateway_error(exc)
                public_error = _public_error(exc)
                server_retry_after = (
                    _retry_after_seconds(exc) if retryable else None
                )
                if (
                    server_retry_after is not None
                    and server_retry_after
                    > _MAX_INLINE_SERVER_RETRY_AFTER_SECONDS
                ):
                    delay = _gateway_retry_delay(
                        exc,
                        attempt=attempt,
                        base_seconds=self.retry_base_seconds,
                        max_seconds=self.retry_max_seconds,
                        random_unit=self._random(),
                    )
                    failure = (
                        "policy model request deferred after "
                        f"{attempt} attempt{'s' if attempt != 1 else ''}: "
                        f"{public_error} (server retry delay)"
                    )
                    self._record_request_diagnostics(
                        model_id,
                        operation=operation,
                        started_at=started_at,
                        attempts=attempt,
                        outcome="failed",
                        retries=retries,
                        last_error=public_error,
                        next_retry_seconds=delay,
                        classification="transient",
                    )
                    self._set_status(model_id, "configured", failure)
                    raise GatewayResponseError(
                        failure,
                        retryable=True,
                        attempts=attempt,
                        status_code=(
                            exc.code if isinstance(exc, HTTPError) else None
                        ),
                        retry_after_seconds=delay,
                    ) from exc
                if retryable and attempt < self.max_attempts:
                    delay = _gateway_retry_delay(
                        exc,
                        attempt=attempt,
                        base_seconds=self.retry_base_seconds,
                        max_seconds=self.retry_max_seconds,
                        random_unit=self._random(),
                    )
                    retries.append(
                        {
                            "attempt": attempt,
                            "reason": public_error,
                            "delay_seconds": delay,
                        }
                    )
                    self._record_request_diagnostics(
                        model_id,
                        operation=operation,
                        started_at=started_at,
                        attempts=attempt,
                        outcome="retrying",
                        retries=retries,
                        last_error=public_error,
                        next_retry_seconds=delay,
                        classification="transient",
                    )
                    self._sleep(delay)
                    continue
                suffix = "retry budget exhausted" if retryable else "not retryable"
                failure = (
                    f"policy model request failed after {attempt} attempt"
                    f"{'s' if attempt != 1 else ''}: {public_error} ({suffix})"
                )
                self._record_request_diagnostics(
                    model_id,
                    operation=operation,
                    started_at=started_at,
                    attempts=attempt,
                    outcome="failed",
                    retries=retries,
                    last_error=public_error,
                    next_retry_seconds=server_retry_after,
                    classification="transient" if retryable else "permanent",
                )
                self._set_status(
                    model_id,
                    "configured" if retryable else "error",
                    failure,
                )
                raise GatewayResponseError(
                    failure,
                    retryable=retryable,
                    attempts=attempt,
                    status_code=exc.code if isinstance(exc, HTTPError) else None,
                    retry_after_seconds=server_retry_after,
                ) from exc
            except GatewayResponseError as exc:
                _append_token_usage(usage_out, exc.usage)
                # A complete response may contain an intermittently malformed
                # root JSON object even when the provider reports ``stop``.
                # Retry that narrow syntax failure inside the gateway so the
                # adapter does not misclassify a healthy sample as a tool
                # failure. Truncation and decoded sample-contract failures still
                # belong to adaptive splitting/repair to avoid replaying a large
                # expensive batch before reducing it.
                sample_json_retry = (
                    operation.startswith("sample.")
                    and exc.error_code == "response_format_invalid"
                    and exc.finish_reason in {None, "stop"}
                    and sample_format_retries < sample_format_retry_limit
                )
                stage_output_retry = operation in {"propose", "judge"} and (
                    _retryable_stage_output_error(exc)
                )
                retry_output_contract = (exc.split_eligible or stage_output_retry) and (
                    not operation.startswith("sample.") or sample_json_retry
                )
                truncation_retry_max_tokens: int | None = None
                if exc.finish_reason in _TRUNCATED_FINISH_REASONS:
                    # Replaying a truncated completion with the same output cap
                    # only repeats the deterministic failure and can consume the
                    # entire transport budget.  Research has a deliberately
                    # larger bounded escalation; sample truncation remains owned
                    # by adaptive batch splitting/repair.
                    if (
                        operation == "research_plan"
                        and current_max_tokens is not None
                        and current_max_tokens < _RESEARCH_PLAN_MAX_TOKENS
                    ):
                        truncation_retry_max_tokens = min(
                            current_max_tokens * 2,
                            _RESEARCH_PLAN_MAX_TOKENS,
                        )
                    retry_output_contract = truncation_retry_max_tokens is not None
                if retry_output_contract and attempt < self.max_attempts:
                    if sample_json_retry:
                        sample_format_retries += 1
                    delay = _gateway_retry_delay(
                        exc,
                        attempt=attempt,
                        base_seconds=self.retry_base_seconds,
                        max_seconds=self.retry_max_seconds,
                        random_unit=self._random(),
                    )
                    retries.append(
                        {
                            "attempt": attempt,
                            "reason": _public_error(exc),
                            "delay_seconds": delay,
                        }
                    )
                    self._record_request_diagnostics(
                        model_id,
                        operation=operation,
                        started_at=started_at,
                        attempts=attempt,
                        outcome="retrying",
                        retries=retries,
                        last_error=_public_error(exc),
                        next_retry_seconds=delay,
                        classification="transient",
                        response_metadata=_gateway_error_response_metadata(exc),
                    )
                    self._sleep(delay)
                    if truncation_retry_max_tokens is not None:
                        current_max_tokens = truncation_retry_max_tokens
                    if not operation.startswith("sample."):
                        body = _chat_request_body(
                            connection,
                            operation=operation,
                            payload=payload,
                            response_contract=response_contract,
                            max_tokens=current_max_tokens,
                            retry_instruction=_COMPACT_JSON_RETRY_INSTRUCTION,
                        )
                        request = Request(
                            _chat_completions_url(connection.gateway_url),
                            data=body,
                            headers=headers,
                            method="POST",
                        )
                    continue
                exc.attempts = attempt
                recoverable_contract = (
                    exc.retryable or exc.split_eligible or stage_output_retry
                )
                self._record_request_diagnostics(
                    model_id,
                    operation=operation,
                    started_at=started_at,
                    attempts=attempt,
                    outcome="failed",
                    retries=retries,
                    last_error=_public_error(exc),
                    next_retry_seconds=None,
                    classification=(
                        "transient" if recoverable_contract else "permanent"
                    ),
                    response_metadata=_gateway_error_response_metadata(exc),
                )
                self._set_status(
                    model_id,
                    "configured" if recoverable_contract else "error",
                    _public_error(exc),
                )
                raise
            _append_token_usage(usage_out, response_metadata.get("usage"))
            self._record_request_diagnostics(
                model_id,
                operation=operation,
                started_at=started_at,
                attempts=attempt,
                outcome="succeeded",
                retries=retries,
                last_error=None,
                next_retry_seconds=None,
                classification="recovered_transient" if retries else "none",
                response_metadata=response_metadata,
            )
            self._set_status(model_id, "available", None)
            return content
        # The loop either returns a validated object or raises on its final
        # attempt.  Keep a defensive guard for static type checkers.
        raise GatewayResponseError("policy model request failed")

    def _record_request_diagnostics(
        self,
        model_id: str,
        *,
        operation: str,
        started_at: str,
        attempts: int,
        outcome: str,
        retries: Sequence[Mapping[str, Any]],
        last_error: str | None,
        next_retry_seconds: float | None,
        classification: str,
        response_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        diagnostic = {
            "operation": operation,
            "started_at": started_at,
            "updated_at": _now(),
            "outcome": outcome,
            "attempts": attempts,
            "retry_count": len(retries),
            "retries": [dict(item) for item in retries],
            "last_error": last_error,
            "next_retry_seconds": next_retry_seconds,
            "classification": classification,
            "timeout_seconds": self.timeout,
        }
        safe_response_metadata = _safe_response_metadata(response_metadata)
        if safe_response_metadata:
            diagnostic["response_metadata"] = safe_response_metadata
        with self._lock:
            self._request_diagnostics[model_id] = diagnostic

    def _set_status(self, model_id: str, state: str, error: str | None) -> None:
        with self._lock:
            self._status[model_id] = {
                "state": state,
                "last_checked_at": _now(),
                "last_error": error,
            }
        self._persist_verification(model_id, verified=None)

    def _set_verified(self, model_id: str, verified: bool) -> None:
        with self._lock:
            self._verified[model_id] = verified
        self._persist_verification(model_id, verified=verified)

    def _restore_verifications(self) -> None:
        self._refresh_verifications()

    def _refresh_verifications(self) -> None:
        for model_id in self._connections:
            self._refresh_verification(model_id)

    def _refresh_verification(self, model_id: str) -> None:
        store = self._verification_store
        loader = getattr(store, "model_verification", None)
        if not callable(loader):
            return
        connection = self._connection(model_id)
        fingerprint = connection.credential_fingerprint
        if fingerprint is None:
            return
        record = loader(
            model_id,
            connection.configuration_digest,
            fingerprint,
        )
        if not isinstance(record, Mapping):
            return
        state = str(record.get("state", "configured"))
        if state not in {"configured", "available", "error"}:
            return
        last_checked_at = record.get("last_checked_at")
        last_error = record.get("last_error")
        with self._lock:
            self._status[model_id] = {
                "state": state,
                "last_checked_at": (
                    str(last_checked_at) if last_checked_at is not None else None
                ),
                "last_error": (
                    public_error_summary(str(last_error))
                    if last_error is not None
                    else None
                ),
            }
            self._verified[model_id] = bool(record.get("verified"))
            self._verification_persisted[model_id] = True

    def _persist_verification(
        self, model_id: str, *, verified: bool | None
    ) -> None:
        store = self._verification_store
        recorder = getattr(store, "record_model_verification", None)
        if not callable(recorder):
            return
        connection = self._connections[model_id]
        fingerprint = connection.credential_fingerprint
        if fingerprint is None:
            return
        with self._lock:
            status = dict(self._status[model_id])
        recorder(
            model_id,
            connection.configuration_digest,
            fingerprint,
            verified=verified,
            state=str(status["state"]),
            last_checked_at=status.get("last_checked_at"),
            last_error=status.get("last_error"),
        )
        with self._lock:
            self._verification_persisted[model_id] = True


def _public_error(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, URLError):
        reason = exc.reason
        return (
            reason.__class__.__name__
            if isinstance(reason, BaseException)
            else "connection error"
        )
    return exc.__class__.__name__


def _validate_origin_shared_context_references(
    samples: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> bool:
    """Fail before transport when a compact sample cannot resolve its context."""

    raw_profile = context.get("sample_planner_prompt_profile")
    if raw_profile is None:
        return False
    if not isinstance(raw_profile, Mapping) or dict(raw_profile) != {
        "version": _ORIGIN_SHARED_CONTEXT_PROFILE
    }:
        raise ValueError("sample planner prompt profile is unsupported")
    shared_contexts = context.get("shared_sample_contexts")
    if not isinstance(shared_contexts, Mapping) or not shared_contexts:
        raise ValueError("origin-shared prompt requires shared_sample_contexts")
    for sample in samples:
        sample_id = sample.get("sample_id")
        context_ref = sample.get("context_ref")
        if (
            not isinstance(context_ref, str)
            or not re.fullmatch(r"[0-9a-f]{64}", context_ref)
            or context_ref not in shared_contexts
        ):
            raise ValueError("origin-shared sample context_ref is unresolved")
        shared = shared_contexts[context_ref]
        if (
            not isinstance(shared, Mapping)
            or shared.get("schema_version") != _ORIGIN_SHARED_CONTEXT_SCHEMA
        ):
            raise ValueError("origin-shared sample context schema is unsupported")
        if digest(shared) != context_ref:
            raise ValueError(
                "origin-shared sample context digest does not match context_ref"
            )
        origin_guard = shared.get("origin_guard")
        sample_refs = shared.get("sample_variant_refs")
        variants = shared.get("sample_variants")
        if not all(
            isinstance(value, Mapping)
            for value in (origin_guard, sample_refs, variants)
        ):
            raise ValueError("origin-shared sample variants are malformed")
        variant_ref = sample_refs.get(sample_id)
        if not isinstance(variant_ref, str) or variant_ref not in variants:
            raise ValueError("origin-shared sample variant is unresolved")
        variant = variants[variant_ref]
        if not isinstance(variant, Mapping):
            raise ValueError("origin-shared sample variant is malformed")
        if digest(
            {
                "schema_version": _ORIGIN_SHARED_CONTEXT_SCHEMA,
                "origin_guard": dict(origin_guard),
                "variant": dict(variant),
            }
        ) != variant_ref:
            raise ValueError("origin-shared sample variant digest does not match")
    return True


def _sample_decision_policy(
    role: str,
    *,
    compact_critic: bool = False,
    origin_shared_context: bool = False,
) -> dict[str, Any]:
    shared = {
        "tool_execution": "host_executes_only_the_selected_registered_tool",
        "outcome_access": "future_observations_are_unavailable",
        "reasoning_output": "return_only_bounded_reason_code_and_confidence",
        "tool_id_contract": (
            "copy_next_tool_exactly_from_allowed_tool_ids_without_adding_"
            "version_or_at_version_suffix"
        ),
    }
    if origin_shared_context:
        shared["sample_context_resolution"] = (
            "resolve_context_ref_in_context.shared_sample_contexts_then_use_"
            "sample_id_to_apply_defaults_and_the_referenced_variant"
        )
    if role == "planner":
        return {
            **shared,
            "goal": "select_the_most_suitable_prediction_tool_for_each_sample",
            "selection_basis": (
                "sample_context_target_horizon_baseline_and_tool_purpose"
            ),
        }
    if role == "critic":
        if compact_critic:
            return {
                **shared,
                "goal": "review_only_the_triggered_uncertain_or_failed_sample",
                "acceptance": "accept_credible_predictions_or_choose_one_allowed_recovery_tool",
                "recovery": "never invent_tools_or_data; terminate_when_no_tool_is_credible",
            }
        return {
            **shared,
            "goal": "accept_the_tool_output_or_select_one_different_recovery_tool",
            "acceptance": "accept_only_when_the_selected_output_is_credible",
            "recovery": "prefer_an_untried_tool_or_terminate_when_none_is_credible",
        }
    return {
        **shared,
        "goal": "select_an_untried_tool_using_bounded_failure_feedback",
        "recovery": "do_not_repeat_a_tool_listed_in_failure_feedback",
    }


def _chat_request_body(
    connection: ModelConnection,
    *,
    operation: str,
    payload: Mapping[str, Any],
    response_contract: Mapping[str, Any],
    max_tokens: int | None,
    retry_instruction: str | None = None,
) -> bytes:
    """Serialize the exact request body used for admission and transport."""

    sample_json_instruction = (
        " For sample operations, use the exact ASCII field names decisions, "
        "sample_id, next_tool, reason_code, and confidence. Keep every JSON "
        "colon and quote; never translate field names."
        if operation.startswith("sample.")
        else ""
    )
    compact_retry_instruction = (
        " " + retry_instruction.strip()
        if isinstance(retry_instruction, str) and retry_instruction.strip()
        else ""
    )
    return json.dumps(
        {
            "model": connection.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a bounded ecology policy model. Return exactly one JSON object, "
                        "without Markdown or explanatory text. Do not invent parameters."
                        + sample_json_instruction
                        + compact_retry_instruction
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "operation": operation,
                            "input": payload,
                            "response_contract": response_contract,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    ),
                },
            ],
        },
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _required_model_role(operation: str) -> str:
    if operation in {"propose", "research_plan"}:
        return "propose"
    if operation.startswith("sample."):
        sample_role = operation.removeprefix("sample.")
        try:
            return _SAMPLE_AGENT_ROLE_TO_MODEL_ROLE[sample_role]
        except KeyError as exc:
            raise GatewayConfigurationError(
                f"unsupported sample operation: {operation}"
            ) from exc
    return operation


def _label_free_remote_object(value: Any, name: str) -> dict[str, Any]:
    """Validate an outbound sample-agent object has no target information."""

    result = _json_object(value, name)
    _reject_forbidden_remote_fields(result, name, response=False)
    return result


def _reject_forbidden_remote_fields(
    value: Any,
    name: str,
    *,
    response: bool,
) -> None:
    """Reject labels at every depth before any remote boundary is crossed."""

    error_type = GatewayResponseError if response else ValueError

    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalized_key = (
                    re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
                    if isinstance(key, str)
                    else None
                )
                if normalized_key is not None and (
                    normalized_key in _FORBIDDEN_REMOTE_SAMPLE_FIELDS
                    or normalized_key.replace("_", "")
                    in _FORBIDDEN_REMOTE_SAMPLE_FIELD_TOKENS
                ):
                    raise error_type(f"{path} must not contain {key!r}")
                walk(nested, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                walk(nested, f"{path}[{index}]")

    walk(value, name)


def _connection_phase(
    status: Mapping[str, Any], diagnostic: Mapping[str, Any] | None
) -> str:
    if diagnostic is not None:
        outcome = str(diagnostic.get("outcome") or "")
        classification = str(diagnostic.get("classification") or "")
        if outcome == "retrying":
            return "retrying"
        if outcome == "failed" and classification == "transient":
            return "transient_error"
        if outcome == "failed" and classification == "permanent":
            return "permanent_error"
    state = str(status.get("state") or "configured")
    if state == "available":
        return "available"
    if state != "error":
        return "configured"
    # Classify records written by earlier releases without request diagnostics.
    # This prevents a historical queue timeout/5xx from being rendered as a
    # permanently invalid API after a service restart.
    error = str(status.get("last_error") or "")
    if re.search(r"\bHTTP (?:408|425|429|5\d\d)\b", error, re.IGNORECASE) or any(
        marker in error.casefold()
        for marker in (
            "timeout",
            "timed out",
            "connection reset",
            "connection refused",
            "connectionrefusederror",
            "gaierror",
            "temporarily",
        )
    ):
        return "transient_error"
    if any(
        marker in error.casefold()
        for marker in _LEGACY_RECOVERABLE_RESPONSE_MARKERS
    ):
        # Before response errors carried machine-readable split eligibility,
        # decoded truncation/contract failures were persisted as generic
        # errors. They do not prove that credentials, routing, or the API are
        # invalid and must remain callable after restart.
        return "transient_error"
    return "permanent_error"


def _is_retryable_gateway_error(exc: BaseException) -> bool:
    if isinstance(exc, (UnicodeDecodeError, json.JSONDecodeError)):
        # A 200 response can still be truncated while an upstream proxy streams
        # it, or contain a temporarily broken byte sequence. Replay the bounded
        # request; fully decoded contract violations remain GatewayResponseError.
        return True
    if isinstance(exc, HTTPError):
        return exc.code in _RETRYABLE_HTTP_STATUS or 500 <= exc.code <= 599
    if isinstance(exc, (IncompleteRead, ssl.SSLEOFError)):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, socket.gaierror):
        return exc.errno == socket.EAI_AGAIN
    if isinstance(exc, URLError):
        reason = exc.reason
        if isinstance(reason, (TimeoutError, ConnectionError)):
            return True
        if isinstance(reason, socket.gaierror):
            return reason.errno == socket.EAI_AGAIN
        if isinstance(
            reason,
            (ConnectionResetError, ConnectionAbortedError, BrokenPipeError),
        ):
            return True
        text = str(reason).casefold()
        return any(
            marker in text
            for marker in (
                "timed out",
                "timeout",
                "connection reset",
                "connection aborted",
                "temporarily unavailable",
                "try again",
            )
        )
    if isinstance(exc, OSError):
        return exc.errno in _RETRYABLE_OS_ERRNOS
    return False


def _gateway_retry_delay(
    exc: BaseException,
    *,
    attempt: int,
    base_seconds: float,
    max_seconds: float,
    random_unit: float,
) -> float:
    exponential = base_seconds * (2 ** max(0, attempt - 1))
    # A bounded +/-20% jitter prevents several DSH processes released by the
    # same upstream limit from immediately forming another synchronized burst.
    jitter_unit = min(1.0, max(0.0, float(random_unit)))
    jittered = exponential * (0.8 + 0.4 * jitter_unit)
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        # ``retry_max_seconds`` bounds client-generated exponential backoff. A
        # gateway's Retry-After describes its actual queue and must not be
        # shortened to that client cap, otherwise several workers wake early
        # and immediately hit the same limit again. Keep one independent,
        # finite safety ceiling for malformed or hostile headers.
        return min(
            _MAX_SERVER_RETRY_AFTER_SECONDS,
            max(min(max_seconds, jittered), retry_after),
        )
    return min(max_seconds, jittered)


def _retry_after_seconds(exc: BaseException) -> float | None:
    if not isinstance(exc, HTTPError):
        return None
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = retry_at.timestamp() - time.time()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _extract_message_content(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise GatewayResponseError(
            "model response must be a JSON object",
            error_code="response_envelope_invalid",
        )
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise GatewayResponseError(
            "model response must contain exactly one choice",
            error_code="response_choices_invalid",
        )
    choice = choices[0]
    if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
        raise GatewayResponseError(
            "model response choice must contain a message",
            error_code="response_message_missing",
        )
    finish_reason = _validated_finish_reason(choice.get("finish_reason"))
    usage = _safe_token_usage(value.get("usage"))
    response_metadata = _safe_response_metadata(
        {"finish_reason": finish_reason, "usage": usage}
    )
    if finish_reason in _TRUNCATED_FINISH_REASONS:
        raise GatewayResponseError(
            "model response ended before a complete final JSON object",
            error_code="output_truncated",
            split_eligible=True,
            finish_reason=finish_reason,
            usage=usage,
        )
    if finish_reason == "content_filter":
        raise GatewayResponseError(
            "model response final content was filtered",
            error_code="content_filtered",
            finish_reason=finish_reason,
            usage=usage,
        )
    if finish_reason in {"tool_calls", "function_call"}:
        raise GatewayResponseError(
            "model response used an unsupported completion mode",
            error_code="unsupported_completion_mode",
            finish_reason=finish_reason,
            usage=usage,
        )
    content = choice["message"].get("content")
    if content is None:
        raise GatewayResponseError(
            "model response did not contain final answer content",
            error_code="final_content_missing",
            split_eligible=True,
            finish_reason=finish_reason,
            usage=usage,
        )
    if not isinstance(content, str):
        raise GatewayResponseError(
            "model response message content must be a JSON string",
            error_code="response_content_type_invalid",
            finish_reason=finish_reason,
            usage=usage,
        )
    try:
        parsed = _extract_single_json_object(content)
    except GatewayResponseError as exc:
        raise _wrap_gateway_error(
            "model response final content must contain one JSON object",
            exc,
            error_code="response_format_invalid",
            split_eligible=True,
            finish_reason=finish_reason,
            usage=usage,
        ) from exc
    return parsed, response_metadata


def _validated_finish_reason(value: Any) -> str | None:
    # Several compatible gateways omit this optional envelope field. If it is
    # present, accept only known protocol values and never echo an unknown value.
    if value is None:
        return None
    if not isinstance(value, str) or value not in _ALLOWED_FINISH_REASONS:
        raise GatewayResponseError(
            "model response finish reason is unsupported",
            error_code="finish_reason_invalid",
        )
    return value


def _safe_token_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    usage: dict[str, int] = {}
    for name in _SAFE_USAGE_FIELDS:
        count = value.get(name)
        if (
            isinstance(count, int)
            and not isinstance(count, bool)
            and 0 <= count <= 1_000_000_000_000
        ):
            usage[name] = count
    return usage


def _append_token_usage(
    output: list[dict[str, int]] | None,
    value: Mapping[str, int] | None,
) -> None:
    """Capture one HTTP response's public usage in its owning call frame."""

    if output is not None:
        output.append(_safe_token_usage(value))


def _token_usage_receipt(
    usages: Sequence[Mapping[str, int]],
    *,
    http_attempts: int,
) -> dict[str, int | bool]:
    """Aggregate allowlisted token usage without reading shared diagnostics."""

    receipt: dict[str, int | bool] = {
        name: sum(
            _safe_token_usage(item).get(name, 0)
            for item in usages
        )
        for name in _SAFE_USAGE_FIELDS
    }
    # An omitted provider envelope is not evidence of zero token use.  Keep
    # that distinction at the gateway boundary so accounting projections never
    # present a missing receipt as a measured zero.
    receipt["usage_reported"] = (
        len(usages) == http_attempts
        and http_attempts > 0
        and all(
            set(_safe_token_usage(item)) == set(_SAFE_USAGE_FIELDS)
            for item in usages
        )
    )
    return receipt


def _safe_response_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    metadata: dict[str, Any] = {}
    finish_reason = value.get("finish_reason")
    if finish_reason in _ALLOWED_FINISH_REASONS:
        metadata["finish_reason"] = finish_reason
    usage = _safe_token_usage(value.get("usage"))
    if usage:
        metadata["usage"] = usage
    return metadata


def _gateway_error_response_metadata(exc: GatewayResponseError) -> dict[str, Any]:
    return _safe_response_metadata(
        {"finish_reason": exc.finish_reason, "usage": exc.usage}
    )


def _extract_single_json_object(content: str) -> dict[str, Any]:
    """Extract one outer JSON object without retaining model narration.

    Some OpenAI-compatible gateways ignore ``response_format`` and wrap the
    object in Markdown or a ``<think>`` block.  Strip thought blocks before
    scanning, then accept exactly one non-nested object.  The returned value is
    immediately passed through the existing finite-value object validator; no
    source text is included in errors or status metadata.
    """

    sanitized = _THINK_BLOCK.sub("", content)
    # An incomplete thought tag must not allow hidden reasoning to be scanned
    # for executable-looking JSON.  Discard everything after the opening tag.
    sanitized = _THINK_OPEN.sub("", sanitized)
    decoder = json.JSONDecoder()
    stripped_start = len(sanitized) - len(sanitized.lstrip())
    if sanitized[stripped_start:].startswith("{"):
        try:
            decoder.raw_decode(sanitized, stripped_start)
        except json.JSONDecodeError as exc:
            repaired = _repair_redundant_object_prefix(
                sanitized,
                start=stripped_start,
                decoder=decoder,
            )
            if repaired is not None:
                sanitized = repaired
                stripped_start = len(sanitized) - len(sanitized.lstrip())
            else:
                # Do not mistake a completed nested decision for the outer response
                # when a JSON object that starts at the beginning was truncated.
                raise GatewayResponseError(
                    "model response content must contain one JSON object"
                ) from exc
    candidates: list[tuple[int, int, Any]] = []
    for start, character in enumerate(sanitized):
        if character != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(sanitized, start)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            candidates.append((start, end, parsed))

    # Nested objects are expected; retain only candidates that are not wholly
    # contained by another candidate.  More than one outer object is rejected
    # so an explanatory example cannot silently become an executable payload.
    outer = [
        candidate
        for candidate in candidates
        if not any(
            other[0] < candidate[0] and other[1] >= candidate[1]
            for other in candidates
        )
    ]
    if len(outer) != 1:
        raise GatewayResponseError("model response content must contain one JSON object")
    start, end, parsed = outer[0]
    outside = sanitized[:start] + sanitized[end:]
    if any(character in "{}[]" for character in outside):
        # Structural delimiters outside the candidate mean it was embedded in
        # another (possibly truncated) JSON value rather than prose or a fence.
        raise GatewayResponseError("model response content must contain one JSON object")
    return _json_object(parsed, "model response content", response=True)


def _repair_redundant_object_prefix(
    content: str,
    *,
    start: int,
    decoder: json.JSONDecoder,
) -> str | None:
    """Repair one observed gateway corruption without enabling fuzzy JSON parsing."""

    # Some OpenAI-compatible JSON-mode gateways return {"{"decisions... when
    # the intended object is {"decisions"... . Accept only that exact prefix,
    # and only when removing the redundant quoted brace yields one complete
    # object with no trailing content. Downstream operation contracts still
    # validate every field and value.
    if not content.startswith('{"{"', start):
        return None
    repaired = content[:start] + "{" + content[start + 3 :]
    try:
        parsed, end = decoder.raw_decode(repaired, start)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, Mapping) or repaired[end:].strip():
        return None
    return repaired


def _json_object(value: Any, name: str, *, response: bool = False) -> dict[str, Any]:
    error_type = GatewayResponseError if response else TypeError
    if not isinstance(value, Mapping):
        raise error_type(f"{name} must be a JSON object")
    result = dict(value)
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise error_type(f"{name} must contain finite JSON values") from exc
    return result


def _response_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GatewayResponseError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_expert_consultation_request(value: Any) -> dict[str, Any]:
    """Project one optional model question onto a non-executable schema."""

    if not isinstance(value, Mapping):
        raise GatewayResponseError("expert_consultation must be an object")
    allowed = {
        "uncertainty_type",
        "question",
        "context",
        "fallback_assumption",
        "requested_expertise",
        "options",
        "confidence",
        "non_blocking",
    }
    _reject_unknown_fields(value, allowed, "expert_consultation")
    missing = allowed - set(value)
    if missing:
        raise GatewayResponseError(
            "expert_consultation is missing fields: " + ", ".join(sorted(missing))
        )
    uncertainty_type = value["uncertainty_type"]
    allowed_types = {item.value for item in ExpertUncertaintyType}
    if not isinstance(uncertainty_type, str) or uncertainty_type not in allowed_types:
        raise GatewayResponseError(
            "expert_consultation uncertainty_type is unsupported"
        )
    if value["non_blocking"] is not True:
        raise GatewayResponseError("expert_consultation must be non-blocking")
    confidence = value["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0 <= float(confidence) <= 1
    ):
        raise GatewayResponseError(
            "expert_consultation confidence must be between 0 and 1"
        )

    result: dict[str, Any] = {
        "uncertainty_type": uncertainty_type,
        "confidence": float(confidence),
        "non_blocking": True,
    }
    for name, maximum in (
        ("question", 2000),
        ("context", 4000),
        ("fallback_assumption", 2000),
    ):
        text = _response_text(value[name], f"expert_consultation.{name}")
        if len(text) > maximum:
            raise GatewayResponseError(f"expert_consultation.{name} is too long")
        result[name] = text
    for name, maximum_items, maximum_length in (
        ("requested_expertise", 8, 160),
        ("options", 8, 500),
    ):
        raw_items = value[name]
        if not isinstance(raw_items, list) or len(raw_items) > maximum_items:
            raise GatewayResponseError(
                f"expert_consultation.{name} must be a bounded array"
            )
        items: list[str] = []
        for item in raw_items:
            text = _response_text(item, f"expert_consultation.{name} item")
            if len(text) > maximum_length:
                raise GatewayResponseError(
                    f"expert_consultation.{name} item is too long"
                )
            if text not in items:
                items.append(text)
        if name == "requested_expertise" and not items:
            raise GatewayResponseError(
                "expert_consultation.requested_expertise must not be empty"
            )
        result[name] = items
    return result


def _project_bounded_plan_object(
    value: Any,
    name: str,
    *,
    allowed_fields: set[str],
    list_fields: set[str],
) -> dict[str, Any]:
    """Project advisory plan metadata onto the small host-owned schema.

    OpenAI-compatible models often add descriptive fields (for example
    ``outputs`` or ``hyperparameters``) even when the response contract asks
    for a compact summary.  Those fields are never executable and are safely
    discarded here.  Malformed optional list entries are skipped rather than
    making an otherwise useful research plan unavailable; all values retained
    by this function are still bounded strings or finite numbers.
    """

    if not isinstance(value, Mapping):
        raise GatewayResponseError(f"{name} must be an object")
    result: dict[str, Any] = {}
    for key, raw in value.items():
        if key not in allowed_fields:
            continue
        if key in list_fields:
            if not isinstance(raw, list):
                continue
            items: list[str] = []
            for item in raw[:32]:
                text: str | None = None
                if isinstance(item, str):
                    text = item.strip()
                elif isinstance(item, Mapping):
                    # Some models render a role/step as a small descriptive
                    # object.  Extract only a conventional display label.
                    for label_key in ("role", "name", "title", "label"):
                        label = item.get(label_key)
                        if isinstance(label, str) and label.strip():
                            text = label.strip()
                            break
                if text:
                    items.append(text[:500])
            result[key] = items
        elif (
            key == "relevance"
            and isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and math.isfinite(float(raw))
        ):
            result[key] = float(raw)
        elif isinstance(raw, str) and raw.strip():
            result[key] = raw.strip()[:1000]
    return result


def _project_bounded_research_summary(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize a small explicit set of non-executable summary aliases."""

    aliases = {
        "source": ("source", "evidence_ref", "knowledge_id", "citation"),
        "title": ("title", "name"),
        "url": ("url",),
        "finding": ("finding", "summary", "key_finding"),
        "relevance": ("relevance", "rationale"),
    }
    result: dict[str, Any] = {}
    for canonical_name, accepted_names in aliases.items():
        for accepted_name in accepted_names:
            if accepted_name not in value:
                continue
            raw = value[accepted_name]
            if (
                canonical_name == "relevance"
                and isinstance(raw, (int, float))
                and not isinstance(raw, bool)
                and math.isfinite(float(raw))
            ):
                result[canonical_name] = float(raw)
                break
            if isinstance(raw, str) and raw.strip():
                result[canonical_name] = redact_sensitive_text(
                    raw.strip(),
                    limit=1000,
                )
                break
        # Unknown keys and malformed aliases are deliberately omitted. An
        # entirely empty projection is filtered by the caller rather than
        # persisted as misleading research evidence.
    return result


def _evidence_predictor_ids(value: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    capability_id = value.get("capability_id")
    if isinstance(capability_id, str) and capability_id.strip():
        result.add(capability_id.strip())
    capability_ids = value.get("capability_ids")
    if isinstance(capability_ids, list):
        result.update(
            item.strip()
            for item in capability_ids
            if isinstance(item, str) and item.strip()
        )
    return result


def _compatible_executable_evidence_refs(
    evidence_catalog: Sequence[Any],
    pipeline_id: str,
) -> list[str]:
    refs: list[str] = []
    for raw_item in evidence_catalog:
        if not isinstance(raw_item, Mapping):
            continue
        knowledge_id = raw_item.get("knowledge_id")
        if not isinstance(knowledge_id, str) or not knowledge_id.strip():
            continue
        if raw_item.get("execution_status") not in {
            "adopted",
            "available_not_selected",
        }:
            continue
        if raw_item.get("capability_kind") != "predictor":
            continue
        if pipeline_id not in _evidence_predictor_ids(raw_item):
            continue
        normalized_ref = knowledge_id.strip()
        if normalized_ref not in refs:
            refs.append(normalized_ref)
    return refs


def _research_algorithm_synthesis_requirement(
    context: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Derive the non-bypassable synthesis contract from frozen host data."""

    blueprint_catalog = context.get("algorithm_blueprint_catalog")
    knowledge_snapshot = context.get("knowledge_snapshot")
    evidence_catalog = (
        knowledge_snapshot.get("evidence_catalog")
        if isinstance(knowledge_snapshot, Mapping)
        else None
    )
    # Compatibility callers that do not yet provide both frozen catalogs keep
    # the legacy optional contract.  Real generation research supplies both.
    if not isinstance(blueprint_catalog, list) or not isinstance(
        evidence_catalog, list
    ):
        return None

    all_research_evidence_options = [
        {
            "knowledge_id": str(raw_item["knowledge_id"]).strip(),
            "evidence_digest": raw_item.get("evidence_digest"),
            "execution_status": raw_item.get("execution_status"),
        }
        for raw_item in evidence_catalog
        if isinstance(raw_item, Mapping)
        and isinstance(raw_item.get("knowledge_id"), str)
        and str(raw_item["knowledge_id"]).strip()
        and raw_item.get("execution_status") in {"research_only", "metadata_only"}
    ]
    metadata_evidence_options = [
        item
        for item in all_research_evidence_options
        if item["execution_status"] == "metadata_only"
    ]
    catalog_research_evidence_options = [
        item
        for item in all_research_evidence_options
        if item["execution_status"] == "research_only"
    ]
    research_evidence_options = [
        *metadata_evidence_options,
        *catalog_research_evidence_options,
    ][:16]
    required_research_evidence_options = (
        metadata_evidence_options[:16]
        if metadata_evidence_options
        else catalog_research_evidence_options[:16]
    )
    compatible_options: list[dict[str, Any]] = []
    seen_pipeline_ids: set[str] = set()
    for raw_blueprint in blueprint_catalog:
        if not isinstance(raw_blueprint, Mapping):
            continue
        pipeline_id = raw_blueprint.get("pipeline_id")
        if (
            not isinstance(pipeline_id, str)
            or not pipeline_id.strip()
            or pipeline_id.strip() in seen_pipeline_ids
        ):
            continue
        normalized_pipeline_id = pipeline_id.strip()
        evidence_refs = _compatible_executable_evidence_refs(
            evidence_catalog,
            normalized_pipeline_id,
        )
        if not evidence_refs:
            continue
        seen_pipeline_ids.add(normalized_pipeline_id)
        compatible_options.append(
            {
                "pipeline_id": normalized_pipeline_id,
                "evidence_refs": evidence_refs[:16],
            }
        )

    if compatible_options:
        return {
            "schema_version": _ALGORITHM_SYNTHESIS_REQUIREMENT_VERSION,
            "mode": "synthesis_required",
            "compatible_options": compatible_options,
            "research_evidence_options": research_evidence_options,
            "required_research_evidence_options": (
                required_research_evidence_options
            ),
            "evidence_requirements": {
                "compatible_executable_predictor_minimum": 1,
                "frozen_research_direction_minimum": (
                    1 if required_research_evidence_options else 0
                ),
            },
            "required_response_fields": [
                "algorithm_blueprint",
                "algorithm_synthesis",
            ],
            "degradation_allowed": False,
        }
    return {
        "schema_version": _ALGORITHM_SYNTHESIS_REQUIREMENT_VERSION,
        "mode": "degradation_required",
        "compatible_options": [],
        "research_evidence_options": research_evidence_options,
        "required_research_evidence_options": required_research_evidence_options,
        "required_response_fields": ["algorithm_synthesis_degradation"],
        "degradation_allowed": True,
        "reason_code": "no_compatible_executable_evidence",
    }


def _validate_research_algorithm_response_requirement(
    content: Mapping[str, Any],
    requirement: Mapping[str, Any] | None,
) -> None:
    if requirement is None:
        if "algorithm_synthesis_degradation" in content:
            raise GatewayResponseError(
                "research plan response cannot report algorithm synthesis degradation "
                "without a frozen host requirement"
            )
        return

    mode = requirement.get("mode")
    if mode == "synthesis_required":
        if "algorithm_synthesis_degradation" in content:
            raise GatewayResponseError(
                "research plan response must not degrade algorithm synthesis when "
                "compatible executable evidence is available",
            )
        missing = {"algorithm_blueprint", "algorithm_synthesis"} - set(content)
        if missing:
            raise GatewayResponseError(
                "research plan response must provide algorithm_blueprint and "
                "algorithm_synthesis when compatible executable evidence is available; "
                "missing: "
                + ", ".join(sorted(missing)),
            )
        research_refs = {
            str(item.get("knowledge_id")).strip()
            for item in requirement.get("required_research_evidence_options", [])
            if isinstance(item, Mapping)
            and isinstance(item.get("knowledge_id"), str)
            and str(item.get("knowledge_id")).strip()
        }
        if research_refs:
            # The executable blueprint is bound separately to exact host
            # operators and compatible executable evidence. Research-only
            # literature belongs in the model synthesis/rationale; forcing it
            # into the executable blueprint creates a brittle and misleading
            # duplicate citation requirement.
            raw_synthesis = content.get("algorithm_synthesis")
            raw_refs = (
                raw_synthesis.get("evidence_refs")
                if isinstance(raw_synthesis, Mapping)
                else None
            )
            cited_refs = (
                {
                    item.strip()
                    for item in raw_refs
                    if isinstance(item, str) and item.strip()
                }
                if isinstance(raw_refs, list)
                else set()
            )
            if not cited_refs.intersection(research_refs):
                raise GatewayResponseError(
                    "research plan response.algorithm_synthesis must cite at least "
                    "one frozen research or metadata evidence item",
                )
        return
    if mode == "degradation_required":
        if "algorithm_blueprint" in content or "algorithm_synthesis" in content:
            raise GatewayResponseError(
                "research plan response must report algorithm_synthesis_degradation "
                "instead of an unsupported blueprint or synthesis",
            )
        if "algorithm_synthesis_degradation" not in content:
            raise GatewayResponseError(
                "research plan response must explicitly report "
                "algorithm_synthesis_degradation when no compatible executable "
                "evidence is available",
            )
        return
    raise GatewayResponseError("host algorithm synthesis requirement mode is invalid")


def _validate_algorithm_synthesis_degradation(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise GatewayResponseError(
            "research plan algorithm_synthesis_degradation must be an object"
        )
    required = {"schema_version", "reason_code", "rationale"}
    if set(value) != required:
        raise GatewayResponseError(
            "research plan algorithm_synthesis_degradation fields do not match "
            "the degradation schema"
        )
    if value.get("schema_version") != _ALGORITHM_SYNTHESIS_DEGRADATION_VERSION:
        raise GatewayResponseError(
            "research plan algorithm_synthesis_degradation has an unsupported "
            "schema_version"
        )
    if value.get("reason_code") != "no_compatible_executable_evidence":
        raise GatewayResponseError(
            "research plan algorithm_synthesis_degradation has an unsupported "
            "reason_code"
        )
    return {
        "schema_version": _ALGORITHM_SYNTHESIS_DEGRADATION_VERSION,
        "reason_code": "no_compatible_executable_evidence",
        "rationale": _response_text(
            value.get("rationale"),
            "research plan algorithm_synthesis_degradation.rationale",
        )[:4000],
    }


def _validate_research_algorithm_blueprint(
    value: Any,
    *,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a model blueprint to one exact host-advertised operator graph."""

    if not isinstance(value, Mapping):
        raise GatewayResponseError("research plan algorithm_blueprint must be an object")
    allowed = {
        "schema_version",
        "pipeline_id",
        "operator_ids",
        "parameter_names",
        "evidence_refs",
        "rationale",
    }
    _reject_unknown_fields(value, allowed, "research plan algorithm_blueprint")
    required = {
        "schema_version",
        "pipeline_id",
        "operator_ids",
        "parameter_names",
        "evidence_refs",
    }
    missing = required - set(value)
    if missing:
        raise GatewayResponseError(
            "research plan algorithm_blueprint is missing fields: "
            + ", ".join(sorted(missing))
        )
    if value.get("schema_version") != _ALGORITHM_BLUEPRINT_VERSION:
        raise GatewayResponseError(
            "research plan algorithm_blueprint has an unsupported schema_version"
        )
    pipeline_id = _response_text(
        value.get("pipeline_id"), "research plan algorithm_blueprint.pipeline_id"
    )
    catalog = context.get("algorithm_blueprint_catalog")
    if not isinstance(catalog, list):
        raise GatewayResponseError(
            "research plan cannot request an algorithm blueprint without a host catalog"
        )
    registered = next(
        (
            item
            for item in catalog
            if isinstance(item, Mapping) and item.get("pipeline_id") == pipeline_id
        ),
        None,
    )
    if registered is None:
        raise GatewayResponseError(
            "research plan algorithm_blueprint pipeline_id is not host registered"
        )
    result: dict[str, Any] = {
        "schema_version": _ALGORITHM_BLUEPRINT_VERSION,
        "pipeline_id": pipeline_id,
    }
    for name in ("operator_ids", "parameter_names"):
        raw = value.get(name)
        if not isinstance(raw, list):
            raise GatewayResponseError(
                f"research plan algorithm_blueprint.{name} must be an array"
            )
        normalized = [
            _response_text(
                item, f"research plan algorithm_blueprint.{name} item"
            )[:500]
            for item in raw
        ]
        registered_values = registered.get(name)
        if not isinstance(registered_values, list) or normalized != registered_values:
            raise GatewayResponseError(
                f"research plan algorithm_blueprint.{name} does not match the host catalog"
            )
        result[name] = normalized

    knowledge_snapshot = context.get("knowledge_snapshot")
    evidence_catalog = (
        knowledge_snapshot.get("evidence_catalog")
        if isinstance(knowledge_snapshot, Mapping)
        else None
    )
    if not isinstance(evidence_catalog, list):
        raise GatewayResponseError(
            "research plan algorithm_blueprint requires a frozen evidence catalog"
        )
    evidence_by_ref = {
        str(item.get("knowledge_id")): item
        for item in evidence_catalog
        if isinstance(item, Mapping)
        and isinstance(item.get("knowledge_id"), str)
        and item.get("knowledge_id")
    }
    available_refs = set(evidence_by_ref)
    raw_evidence = value.get("evidence_refs")
    if (
        not isinstance(raw_evidence, list)
        or not raw_evidence
        or len(raw_evidence) > 16
    ):
        raise GatewayResponseError(
            "research plan algorithm_blueprint.evidence_refs must contain 1 to 16 items"
        )
    evidence_refs: list[str] = []
    for raw_ref in raw_evidence:
        evidence_ref = _response_text(
            raw_ref, "research plan algorithm_blueprint.evidence_refs item"
        )
        if evidence_ref not in available_refs:
            raise GatewayResponseError(
                "research plan algorithm_blueprint evidence ref is outside the frozen snapshot"
            )
        if evidence_ref not in evidence_refs:
            evidence_refs.append(evidence_ref)
    referenced_evidence = [evidence_by_ref[ref] for ref in evidence_refs]
    if not _compatible_executable_evidence_refs(
        referenced_evidence,
        pipeline_id,
    ):
        raise GatewayResponseError(
            "research plan algorithm_blueprint requires frozen executable evidence "
            "mapped to its predictor"
        )
    result["evidence_refs"] = evidence_refs
    if "rationale" in value:
        result["rationale"] = _response_text(
            value["rationale"], "research plan algorithm_blueprint.rationale"
        )[:4000]
    return result


def _reject_unknown_fields(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise GatewayResponseError(f"{name} contains unsupported fields: {', '.join(sorted(unknown))}")


def _normalize_parameter_schemas(
    value: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("allowed_parameters must be a non-empty mapping")
    result: dict[str, dict[str, Any]] = {}
    for name, raw_schema in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("allowed parameter names must be non-empty strings")
        if not isinstance(raw_schema, Mapping):
            raise TypeError(f"allowed parameter schema for {name} must be a mapping")
        schema = dict(raw_schema)
        _reject_schema_fields(schema, name)
        kind = schema.get("type", "number")
        if kind not in {"number", "integer"}:
            raise ValueError(f"allowed parameter {name} has unsupported type")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        for boundary_name, boundary in (("minimum", minimum), ("maximum", maximum)):
            if isinstance(boundary, bool) or not isinstance(boundary, (int, float)) or not math.isfinite(float(boundary)):
                raise ValueError(f"allowed parameter {name}.{boundary_name} must be finite")
        if float(minimum) > float(maximum):
            raise ValueError(f"allowed parameter {name} has an inverted range")
        result[name] = {"type": kind, "minimum": minimum, "maximum": maximum}
    return result


def _reject_schema_fields(schema: Mapping[str, Any], name: str) -> None:
    unknown = set(schema) - {"type", "minimum", "maximum"}
    if unknown:
        raise ValueError(f"allowed parameter {name} has unsupported schema fields")
    if "minimum" not in schema or "maximum" not in schema:
        raise ValueError(f"allowed parameter {name} requires minimum and maximum")


def _validate_parameters(
    value: Any,
    schemas: Mapping[str, Mapping[str, Any]],
    *,
    partial: bool,
) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise GatewayResponseError("parameters must be a JSON object")
    unknown = set(value) - set(schemas)
    if unknown:
        raise GatewayResponseError(f"parameters contain unsupported names: {', '.join(sorted(unknown))}")
    if not partial and set(value) != set(schemas):
        raise GatewayResponseError("parameters must contain every allowed parameter")
    result: dict[str, int | float] = {}
    for name, raw in value.items():
        schema = schemas[name]
        kind = schema["type"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise GatewayResponseError(f"parameter {name} must be a finite number")
        if kind == "integer" and not isinstance(raw, int):
            raise GatewayResponseError(f"parameter {name} must be an integer")
        if float(raw) < float(schema["minimum"]) or float(raw) > float(schema["maximum"]):
            raise GatewayResponseError(f"parameter {name} is outside the allowed range")
        result[name] = int(raw) if kind == "integer" else float(raw)
    return result
