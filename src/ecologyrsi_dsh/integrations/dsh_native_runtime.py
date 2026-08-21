"""Narrow loopback client for the DSH-owned Agent runtime.

This module deliberately has no completion/chat API and imports no model
credential or gateway implementation. DSH owns Agent context, compaction,
subagents and workflows; Python owns durable scientific state.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from ..core.errors import DshNativeRuntimeUnavailableError


DSH_NATIVE_EXECUTION_PROTOCOL = "dsh_native_plugin_evolution@1"
_CAPABILITY_KEYS = frozenset(
    {
        "schema_version",
        "ready",
        "root_services",
        "presets",
        "live_agent_service_ready",
        "first_call_verified",
    }
)
_MUTATION_KEYS = frozenset(
    {
        "accepted",
        "run_id",
        "run_state_revision",
        "stage_attempt",
        "ledger_expected_revision",
        "idempotency_key",
    }
)
_STATUS_KEYS = frozenset(
    {
        "run_id",
        "status",
        "run_state_revision",
        "stage_attempt",
        "ledger_expected_revision",
        "idempotency_key",
    }
)
_STAGE_KEYS = _MUTATION_KEYS | frozenset(
    {"structured", "result_digest", "session_id", "first_call_verified"}
)


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DshNativeRuntimeUnavailableError(
            f"DSH 运行时响应缺少 {name}。",
            error_code="dsh_native_runtime_contract_error",
        )
    return value.strip()


def _revision(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DshNativeRuntimeUnavailableError(
            f"DSH 运行时响应中 {name} 无效。",
            error_code="dsh_native_runtime_contract_error",
        )
    return value


class DshNativeAgentRuntimeClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        timeout: float = 660.0,
        max_response_bytes: int = 1_048_576,
    ) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("DSH native runtime must use a literal loopback HTTP origin")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("DSH native runtime token is required")
        if timeout <= 0:
            raise ValueError("DSH native runtime timeout must be positive")
        if not 1 <= max_response_bytes <= 8 * 1024 * 1024:
            raise ValueError("DSH native runtime response bound is invalid")
        self.base_url = base_url.rstrip("/")
        self.__token = token.strip()
        self.timeout = float(timeout)
        self.max_response_bytes = int(max_response_bytes)

    def capabilities(self, *, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        payload = self._request("GET", "/api/ecology-agent-runtime/v1/capabilities", cancelled=cancelled)
        self._exact_keys(payload, _CAPABILITY_KEYS, "capabilities")
        if payload.get("schema_version") != "ecology-agent-runtime-capabilities/1":
            raise DshNativeRuntimeUnavailableError(
                "DSH 运行时 capability 版本不匹配。",
                error_code="dsh_native_runtime_contract_error",
            )
        if not isinstance(payload.get("ready"), bool):
            raise DshNativeRuntimeUnavailableError(
                "DSH 运行时 capability ready 无效。",
                error_code="dsh_native_runtime_contract_error",
            )
        if not isinstance(payload.get("root_services"), Mapping) or not isinstance(payload.get("presets"), list):
            raise DshNativeRuntimeUnavailableError(
                "DSH 运行时 capability 结构无效。",
                error_code="dsh_native_runtime_contract_error",
            )
        return payload

    def require_capabilities(
        self,
        payload: Mapping[str, Any],
        required_presets: Sequence[str],
        *,
        require_live: bool = True,
    ) -> None:
        roots = payload.get("root_services")
        presets = payload.get("presets")
        by_id = {
            item.get("preset_id"): item
            for item in presets if isinstance(item, Mapping)
        } if isinstance(presets, list) else {}
        good_roots = (
            isinstance(roots, Mapping)
            and roots.get("declared") is True
            and roots.get("missing") == []
        )
        required_fields = [
            "declared",
            "preset_mountable",
            "tool_surface_verified",
            "route_resolvable",
        ]
        if require_live:
            required_fields.append("live_agent_service_ready")
        good_presets = all(
            isinstance(by_id.get(preset_id), Mapping)
            and all(
                by_id[preset_id].get(field) is True
                for field in required_fields
            )
            for preset_id in required_presets
        )
        if not (
            payload.get("ready") is True
            and good_roots
            and good_presets
            and (
                not require_live
                or payload.get("live_agent_service_ready") is True
            )
        ):
            raise DshNativeRuntimeUnavailableError(
                "DSH 原生智能体预设、工具或运行时服务尚未就绪。",
                error_code="dsh_native_runtime_not_ready",
            )

    def create_run(self, binding: Mapping[str, Any], *, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        return self._mutation("/api/ecology-agent-runtime/v1/runs", binding, cancelled)

    def run_stage(self, request: Mapping[str, Any], *, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        run_id = _nonempty_text(request.get("run_id"), "run_id")
        self._validate_identity(request)
        payload = self._request(
            "POST",
            f"/api/ecology-agent-runtime/v1/runs/{quote(run_id, safe='')}/stages",
            body=request,
            cancelled=cancelled,
        )
        self._exact_keys(payload, _STAGE_KEYS, "stage mutation")
        self._validate_identity(payload)
        if payload.get("accepted") is not True or payload.get("run_id") != run_id:
            raise DshNativeRuntimeUnavailableError(
                "DSH stage was not accepted for the frozen run.",
                error_code="dsh_native_runtime_stage_rejected",
            )
        if not isinstance(payload.get("structured"), Mapping):
            raise DshNativeRuntimeUnavailableError(
                "DSH stage returned no structured result.",
                error_code="dsh_native_runtime_contract_error",
            )
        _nonempty_text(payload.get("result_digest"), "result_digest")
        _nonempty_text(payload.get("session_id"), "session_id")
        if payload.get("first_call_verified") is not True:
            raise DshNativeRuntimeUnavailableError(
                "DSH stage did not verify its frozen model route.",
                error_code="dsh_native_runtime_route_unverified",
            )
        return payload

    def cancel(self, request: Mapping[str, Any], *, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        run_id = _nonempty_text(request.get("run_id"), "run_id")
        return self._mutation(f"/api/ecology-agent-runtime/v1/runs/{quote(run_id, safe='')}/cancel", request, cancelled)

    def pause(self, request: Mapping[str, Any], *, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        run_id = _nonempty_text(request.get("run_id"), "run_id")
        return self._mutation(f"/api/ecology-agent-runtime/v1/runs/{quote(run_id, safe='')}/pause", request, cancelled)

    def resume(self, request: Mapping[str, Any], *, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        run_id = _nonempty_text(request.get("run_id"), "run_id")
        return self._mutation(f"/api/ecology-agent-runtime/v1/runs/{quote(run_id, safe='')}/resume", request, cancelled)

    def status(self, run_id: str, *, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        target = _nonempty_text(run_id, "run_id")
        payload = self._request("GET", f"/api/ecology-agent-runtime/v1/runs/{quote(target, safe='')}", cancelled=cancelled)
        self._exact_keys(payload, _STATUS_KEYS, "status")
        self._validate_identity(payload)
        _nonempty_text(payload.get("status"), "status")
        return payload

    def _mutation(
        self,
        path: str,
        request: Mapping[str, Any],
        cancelled: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        self._validate_identity(request)
        payload = self._request("POST", path, body=request, cancelled=cancelled)
        self._exact_keys(payload, _MUTATION_KEYS, "mutation")
        self._validate_identity(payload)
        if payload.get("accepted") is not True:
            raise DshNativeRuntimeUnavailableError(
                "DSH 运行时未接受请求。",
                error_code="dsh_native_runtime_stage_rejected",
            )
        for field in ("run_id", "idempotency_key"):
            if payload.get(field) != request.get(field):
                raise DshNativeRuntimeUnavailableError(
                    "DSH 运行时返回的冻结身份不匹配。",
                    error_code="dsh_native_runtime_identity_mismatch",
                )
        return payload

    @staticmethod
    def _exact_keys(payload: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
        if not isinstance(payload, Mapping) or frozenset(payload) != expected:
            raise DshNativeRuntimeUnavailableError(
                f"DSH 运行时 {name} 响应 schema 不匹配。",
                error_code="dsh_native_runtime_contract_error",
            )

    @staticmethod
    def _validate_identity(payload: Mapping[str, Any]) -> None:
        _nonempty_text(payload.get("run_id"), "run_id")
        _nonempty_text(payload.get("idempotency_key"), "idempotency_key")
        _revision(payload.get("run_state_revision"), "run_state_revision")
        _revision(payload.get("stage_attempt"), "stage_attempt")
        _revision(payload.get("ledger_expected_revision"), "ledger_expected_revision")

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        if cancelled is not None and cancelled():
            raise DshNativeRuntimeUnavailableError(
                "DSH 运行时请求已取消。",
                error_code="dsh_native_runtime_cancelled",
            )
        encoded = None
        if body is not None:
            try:
                encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError("DSH native runtime request must be finite JSON") from exc
            if len(encoded) > 1_048_576:
                raise ValueError("DSH native runtime request is too large")
        request = Request(
            self.base_url + path,
            data=encoded,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.__token}",
                **({"Content-Type": "application/json"} if encoded is not None else {}),
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                if response.headers.get_content_type() != "application/json":
                    raise DshNativeRuntimeUnavailableError(
                        "DSH 运行时返回了非 JSON 响应。",
                        error_code="dsh_native_runtime_contract_error",
                    )
                raw = response.read(self.max_response_bytes + 1)
                if len(raw) > self.max_response_bytes:
                    raise DshNativeRuntimeUnavailableError(
                        "DSH 运行时响应超过大小上限。",
                        error_code="dsh_native_runtime_contract_error",
                    )
        except HTTPError as exc:
            raw = exc.read(self.max_response_bytes + 1)
            code = "dsh_native_runtime_http_error"
            try:
                parsed = json.loads(raw)
                supplied = parsed.get("error_code") if isinstance(parsed, Mapping) else None
                if isinstance(supplied, str) and supplied.replace("_", "").isalnum():
                    code = supplied[:80]
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            raise DshNativeRuntimeUnavailableError(
                "DSH 原生智能体运行时拒绝了请求。",
                error_code=code,
                status_code=exc.code,
            ) from None
        except (URLError, TimeoutError, socket.timeout, OSError):
            raise DshNativeRuntimeUnavailableError(
                "DSH 原生智能体运行时连接失败。",
                error_code="dsh_native_runtime_transport_error",
            ) from None
        if cancelled is not None and cancelled():
            raise DshNativeRuntimeUnavailableError(
                "DSH 运行时请求已取消。",
                error_code="dsh_native_runtime_cancelled",
            )
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise DshNativeRuntimeUnavailableError(
                "DSH 运行时返回了无效 JSON。",
                error_code="dsh_native_runtime_contract_error",
            ) from None
        if not isinstance(payload, dict):
            raise DshNativeRuntimeUnavailableError(
                "DSH 运行时响应必须是 JSON object。",
                error_code="dsh_native_runtime_contract_error",
            )
        return payload
