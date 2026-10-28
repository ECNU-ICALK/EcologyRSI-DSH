"""Loopback LLM gateway: the agent never holds a real API key.

DSH inside the sandbox talks to ``http://127.0.0.1:<port>/deepseek/...`` with a
per-episode token; the gateway checks it, swaps in the real key and forwards
to the upstream API, streaming responses back unchanged. It also

* meters tokens from each response's usage and refuses calls once the
  episode's token budget or deadline is spent;
* appends every call (request body, response text, usage, timing) to a JSON
  lines file outside the sandbox: the exact model-level trajectory.

Keys never appear in the log: only bodies are recorded, never headers.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import ssl
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PREFIX = "/deepseek"
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding",
       "upgrade", "content-length", "content-encoding", "host", "authorization", "x-api-key", "accept-encoding"}
LOG_RESPONSE_LIMIT = 4_000_000


class _QuietServer(ThreadingHTTPServer):
    """Dropped connections are normal when the episode is torn down; don't dump their tracebacks."""

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            super().handle_error(request, client_address)


def tokens_of(usage: dict[str, Any] | None) -> int:
    if not usage:
        return 0
    if isinstance(usage.get("total_tokens"), int):
        return int(usage["total_tokens"])
    pairs = (("prompt_tokens", "completion_tokens"), ("input_tokens", "output_tokens"))
    return max(sum(int(usage.get(key) or 0) for key in pair) for pair in pairs)


def usage_of(text: str, content_type: str) -> dict[str, Any] | None:
    """Usage from a JSON body or the events of a server-sent stream (last value per key wins)."""

    if "event-stream" not in content_type:
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            return None
        return body.get("usage") if isinstance(body, dict) else None
    usage: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            for found in (event.get("usage"), (event.get("message") or {}).get("usage")):
                if isinstance(found, dict):
                    usage.update({key: value for key, value in found.items() if value is not None})
    return usage or None


class Gateway:
    def __init__(self, *, key: str, token: str, log_path: Path, upstream: str = "https://api.deepseek.com",
                 token_budget: int = 3_000_000, deadline: float | None = None, proxy: str | None = None) -> None:
        self._key = key
        self.token = token
        self.log_path = log_path
        self.upstream = urlparse(upstream)
        self.token_budget = token_budget
        self.deadline = deadline
        self.proxy = urlparse(proxy) if proxy else None
        self.tokens = 0
        self.calls = 0
        self.refused = 0
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self.on_call: Callable[[dict[str, Any]], None] | None = None

    @property
    def port(self) -> int:
        assert self._server is not None
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}{PREFIX}"

    def start(self) -> Gateway:
        self._server = _QuietServer(("127.0.0.1", 0), _handler(self))
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def refusal(self) -> str | None:
        if self.deadline is not None and time.time() > self.deadline:
            return "the episode deadline has passed"
        if self.tokens >= self.token_budget:
            return f"the episode token budget ({self.token_budget}) is spent"
        return None

    def connection(self) -> http.client.HTTPConnection:
        host = self.upstream.hostname or ""
        port = self.upstream.port or (443 if self.upstream.scheme == "https" else 80)
        if self.upstream.scheme == "http":
            return http.client.HTTPConnection(host, port, timeout=900)
        context = ssl.create_default_context()
        if self.proxy is not None:
            connection = http.client.HTTPSConnection(self.proxy.hostname or "", self.proxy.port or 80,
                                                     timeout=900, context=context)
            connection.set_tunnel(host, port)
            return connection
        return http.client.HTTPSConnection(host, port, timeout=900, context=context)

    def record(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self.calls += 1
            self.tokens += tokens_of(entry.get("usage"))
            entry["seq"] = self.calls
            entry["tokens_total"] = self.tokens
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if self.on_call is not None:
            self.on_call({key: entry.get(key) for key in ("seq", "path", "status", "usage", "seconds", "model")})


def _handler(gateway: Gateway) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # quiet
            return None

        def _json(self, status: int, message: str) -> None:
            data = json.dumps({"error": {"message": f"ecogym gateway: {message}", "type": "ecogym_gateway"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _token(self) -> str:
            bearer = self.headers.get("Authorization", "")
            return bearer[7:].strip() if bearer.startswith("Bearer ") else self.headers.get("x-api-key", "").strip()

        def do_GET(self) -> None:
            self._forward("GET")

        def do_POST(self) -> None:
            self._forward("POST")

        def _forward(self, method: str) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else b""
            if self._token() != gateway.token:
                return self._json(401, "unauthorized")
            if not self.path.startswith(PREFIX + "/"):
                return self._json(404, f"unknown route {self.path}")
            reason = gateway.refusal()
            if reason:
                with gateway._lock:
                    gateway.refused += 1
                return self._json(429, reason)
            path = self.path[len(PREFIX):]
            headers = {key: value for key, value in self.headers.items() if key.lower() not in HOP}
            if self.headers.get("Authorization"):
                headers["Authorization"] = f"Bearer {gateway._key}"
            if self.headers.get("x-api-key"):
                headers["x-api-key"] = gateway._key
            headers["Accept-Encoding"] = "identity"
            started = time.monotonic()
            entry: dict[str, Any] = {"at": time.time(), "path": path, "method": method}
            try:
                entry["request"] = json.loads(body) if body else None
            except json.JSONDecodeError:
                entry["request"] = body.decode("utf-8", "replace")[:100000]
            if isinstance(entry["request"], dict):
                entry["model"] = entry["request"].get("model")
            captured = bytearray()
            try:
                connection = gateway.connection()
                connection.request(method, path, body=body or None, headers=headers)
                response = connection.getresponse()
                entry["status"] = response.status
                content_type = response.getheader("Content-Type", "")
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in HOP:
                        self.send_header(key, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    if len(captured) < LOG_RESPONSE_LIMIT:
                        captured.extend(chunk[: LOG_RESPONSE_LIMIT - len(captured)])
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                connection.close()
                text = captured.decode("utf-8", "replace")
                entry["response"] = text
                entry["usage"] = usage_of(text, content_type)
            except (OSError, http.client.HTTPException) as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                if "status" not in entry:
                    with contextlib.suppress(OSError):
                        self._json(502, f"upstream failed: {type(error).__name__}")
            finally:
                entry["seconds"] = round(time.monotonic() - started, 3)
                gateway.record(entry)

    return Handler


__all__ = ["Gateway", "tokens_of", "usage_of"]
