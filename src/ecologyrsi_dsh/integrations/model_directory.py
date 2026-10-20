"""Read-only Harness model inventory; all inference belongs to native roles."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from ..core.models import digest
from .dsh_directory import discover_model_entries
from .gateway_url_policy import assess_gateway_url
from .model_bindings import MODEL_BINDING_SCHEMA_VERSION, canonical_model_roles


class ModelConfigurationError(ValueError):
    """A model directory entry is unsafe or incomplete."""


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _safe_gateway_url(value: Any, *, allow_insecure_http: bool = False) -> tuple[str, bool]:
    assessment = assess_gateway_url(value, allow_insecure_http=allow_insecure_http)
    if assessment.gateway_error is not None:
        raise ModelConfigurationError(assessment.gateway_error)
    return assessment.url, assessment.insecure_http_exception


@dataclass(frozen=True, slots=True)
class ModelConnection:
    model_id: str
    gateway_url: str
    model: str
    token: str | None = field(default=None, repr=False)
    label: str | None = None
    roles: tuple[str, ...] = ("propose", "judge")
    allow_insecure_http: bool = False
    native_configuration_digest: str | None = None

    def __post_init__(self) -> None:
        if self.native_configuration_digest is not None and (
            not isinstance(self.native_configuration_digest, str)
            or len(self.native_configuration_digest) != 64
            or any(char not in "0123456789abcdef" for char in self.native_configuration_digest)
        ):
            raise ModelConfigurationError("native_configuration_digest must be a SHA-256 digest")
        object.__setattr__(self, "model_id", _required_text(self.model_id, "model_id"))
        if not isinstance(self.allow_insecure_http, bool):
            raise ModelConfigurationError("allow_insecure_http must be a boolean")
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
            raise ModelConfigurationError("roles must be a non-empty array")
        try:
            roles = canonical_model_roles(self.roles)
        except ValueError as exc:
            raise ModelConfigurationError(str(exc)) from exc
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
                "native_configuration_digest": self.native_configuration_digest,
            }
        )

class ModelDirectory:
    """Expose redacted routes and immutable bindings, without network requests.

    Credential presence is configuration evidence only. Execution readiness is
    established separately by four role-bound native canaries before creation.
    """

    def __init__(self, connections: Sequence[ModelConnection] = (), *,
                 directory_unavailable: Sequence[Mapping[str, Any]] = ()) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        for connection in connections:
            if not isinstance(connection, ModelConnection):
                raise TypeError("connections must contain ModelConnection values")
            self._add({
                "model_id": connection.model_id,
                "label": connection.label,
                "model": connection.model,
                "roles": list(connection.roles),
                "configuration_digest": connection.configuration_digest,
                "binding_source": "dsh_native_configuration",
                "credential_configured": connection.token is not None,
                "configured": connection.token is not None,
                "directory_available": True,
                "execution_available": connection.token is not None,
                "authentication_verified": False,
                "authentication_state": ("configured_unverified" if connection.token else "missing_credential"),
                "available": False,
            })
        for raw in directory_unavailable:
            model_id = _required_text(raw.get("id"), "directory model id")
            model = _required_text(raw.get("model"), "directory model")
            reason = raw["unavailable_reason"]
            if not isinstance(reason, Mapping):
                raise ModelConfigurationError("unavailable entry requires a reason")
            self._add({
                "model_id": model_id,
                "label": str(raw.get("label") or model_id),
                "model": model,
                "roles": list(raw.get("roles", [])),
                "configuration_digest": digest({"model_id": model_id, "reason": reason["code"]}),
                "binding_source": "dsh_directory_unavailable",
                "credential_configured": bool(raw.get("token")),
                "configured": False,
                "directory_available": False,
                "execution_available": False,
                "authentication_verified": False,
                "authentication_state": "unavailable",
                "available": False,
                "unavailable_reason": {"code": str(reason["code"])[:80],
                                       "message": str(reason["message"])[:300]},
            })

    def _add(self, entry: dict[str, Any]) -> None:
        model_id = entry["model_id"]
        if model_id in self._entries:
            raise ModelConfigurationError(f"duplicate model_id: {model_id}")
        self._entries[model_id] = entry

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ModelDirectory:
        connections = []
        unavailable = []
        for entry in discover_model_entries(environ):
            if entry["directory_available"] is False:
                unavailable.append(entry)
                continue
            connections.append(ModelConnection(
                model_id=entry["id"], model=entry["model"], label=entry["label"],
                gateway_url=entry["gateway_url"], token=entry.get("token"),
                roles=tuple(entry["roles"]),
                allow_insecure_http=entry["allow_insecure_http"],
                native_configuration_digest=entry["native_configuration_digest"],
            ))
        return cls(connections, directory_unavailable=unavailable)

    def catalog(self) -> list[dict[str, Any]]:
        from copy import deepcopy
        return deepcopy(list(self._entries.values()))

    def configuration_digest(self, model_id: str) -> str:
        try:
            return self._entries[model_id]["configuration_digest"]
        except KeyError:
            raise ModelConfigurationError(f"unknown model_id: {model_id}") from None
