"""HTTP parsing, authorization, static files, and JSON responses."""

from __future__ import annotations

import hmac
import json
from http import HTTPStatus
from typing import Any
from urllib.parse import unquote, urlparse

from .shared import _PLUGIN_FILES, _plugin_root, _public_http_error


class TransportMixin:
    def _route(self) -> list[str]:
        path = [unquote(part) for part in urlparse(self.path).path.split("/") if part]
        if path and path[0] == "api":
            path = path[1:]
            if path and path[0] == "ecology-evolution":
                path = path[1:]
            if path and path[0] in {"v1", "v2"}:
                path = path[1:]
        return path

    def _authorize_api(self) -> bool:
        expected = self.server.capability_token
        if expected is None:
            return True
        header = self.headers.get("Authorization", "")
        supplied = header[7:].strip() if header.startswith("Bearer ") else ""
        if supplied and hmac.compare_digest(supplied, expected):
            return True
        self._send(HTTPStatus.UNAUTHORIZED, {"error": "DSH 能力令牌无效或缺失"})
        return False

    def _authorize_dsh_tool(self) -> bool:
        expected = self.server.dsh_tool_token
        if expected is None:
            self._send(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "DSH role-tool boundary is not configured"},
            )
            return False
        header = self.headers.get("Authorization", "")
        supplied = header[7:].strip() if header.startswith("Bearer ") else ""
        if supplied and hmac.compare_digest(supplied, expected):
            return True
        self._send(HTTPStatus.UNAUTHORIZED, {"error": "invalid DSH role-tool token"})
        return False

    def _serve_plugin(self, raw_path: str) -> None:
        prefix = "/plugins/ecology/evolution"
        relative = raw_path[len(prefix):].lstrip("/") or "index.html"
        if relative not in _PLUGIN_FILES:
            self._send(HTTPStatus.NOT_FOUND, {"error": "plugin file not found"})
            return
        root = _plugin_root()
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            self._send(HTTPStatus.NOT_FOUND, {"error": "plugin file not found"})
            return
        if not target.is_file():
            self._send(HTTPStatus.NOT_FOUND, {"error": "plugin file not found"})
            return
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", _PLUGIN_FILES[relative])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'self'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # A browser refresh or bounded CLI read may close the socket after
            # headers; this is not an application or evolution failure.
            return

    def _call(self, callback: Any) -> None:
        try:
            self._send(HTTPStatus.OK, callback())
        except PermissionError as exc:
            self._send(HTTPStatus.FORBIDDEN, {"error": _public_http_error(exc)})
        except FileNotFoundError as exc:
            self._send(HTTPStatus.NOT_FOUND, {"error": _public_http_error(exc)})
        except KeyError as exc:
            self._send(HTTPStatus.NOT_FOUND, {"error": _public_http_error(exc)})
        except (RuntimeError, TypeError, ValueError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": _public_http_error(exc)})

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _send(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Do not turn a client-side disconnect into a noisy sidecar trace.
            return
