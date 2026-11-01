"""Loopback LLM gateway: the agent never holds a real API key.

DSH inside the sandbox talks to ``http://127.0.0.1:<port>/deepseek/...`` (the
DeepSeek API) or ``.../route/<provider>/...`` (another provider route, see
``ecogym.episodes.providers``) with a per-session token; the gateway checks
it, swaps in that route's real key (and adds the route's own headers) and
forwards to the route's upstream, streaming responses back unchanged. It also

* meters tokens from each response's usage and refuses calls once the
  session's token budget or deadline is spent. The budget counts
  cost-weighted tokens: prompt tokens served from the provider's cache count
  ``CACHE_WEIGHT`` each (DeepSeek bills them at about a tenth), so a long
  session that re-reads its own history is not cut off long before its time;
* counts answered and failed upstream calls, so a session that never got a
  model reply is told apart from one whose method failed;
* appends every call to a JSON lines file outside the sandbox: the request
  body and, for chat completions, the assembled reply (content, reasoning,
  tool calls, finish reason). That is the exact model-level trajectory, ready
  to be turned into training data. Other responses (web search) are stored as
  raw text, truncated.

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

from ..kernel import clock

PREFIX = "/deepseek"
ROUTE_PREFIX = "/route/"
SEARCH_PATH = "/anthropic/v1/"  # DeepSeek's web search, under PREFIX (DSH's web-search-deepseek)
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding",
       "upgrade", "content-length", "content-encoding", "host", "authorization", "x-api-key", "accept-encoding"}
REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")  # stored as reasoning_content
CAPTURE_LIMIT = 16_000_000  # bytes of one response kept for assembly
RAW_LOG_LIMIT = 200_000  # characters of a response that cannot be assembled
CACHE_WEIGHT = 0.1  # budget weight of a prompt token served from the provider's cache


class _QuietServer(ThreadingHTTPServer):
    """Dropped connections are normal when a session is torn down; don't dump their tracebacks."""

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


def metered_tokens(usage: dict[str, Any] | None) -> int:
    """Cost-weighted tokens: cached prompt tokens count ``CACHE_WEIGHT`` each."""

    if not usage:
        return 0
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    cached = next((value for value in (usage.get("prompt_cache_hit_tokens"), details.get("cached_tokens"),
                                       usage.get("cache_read_input_tokens"), usage.get("cached_tokens"))
                   if isinstance(value, int)), 0)  # DeepSeek, OpenAI, Anthropic, Kimi spellings
    total = tokens_of(usage)
    return max(0, round(total - cached + CACHE_WEIGHT * cached))


def _events(text: str) -> list[Any]:
    found = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            found.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return found


def usage_of(text: str, content_type: str) -> dict[str, Any] | None:
    """Usage from a JSON body or the events of a server-sent stream (last value per key wins)."""

    if "event-stream" not in content_type:
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            return None
        return body.get("usage") if isinstance(body, dict) else None
    usage: dict[str, Any] = {}
    for event in _events(text):
        if isinstance(event, dict):
            for found in (event.get("usage"), (event.get("message") or {}).get("usage")):
                if isinstance(found, dict):
                    usage.update({key: value for key, value in found.items() if value is not None})
    return usage or None


def assemble(text: str, content_type: str) -> dict[str, Any] | None:
    """The assistant message of a chat-completions response (streamed or not); None for other formats."""

    if "event-stream" not in content_type:
        try:
            body = json.loads(text)
            choice = body["choices"][0]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return None
        message = {key: value for key, value in (choice.get("message") or {}).items() if value is not None}
        alias = next((key for key in REASONING_FIELDS[1:] if isinstance(message.get(key), str) and message[key]), None)
        if alias and not message.get("reasoning_content"):
            message["reasoning_content"] = message.pop(alias)
        return {"message": {"role": "assistant", **message}, "finish_reason": choice.get("finish_reason")}
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish = None
    events = _events(text)
    if not events or not all(isinstance(event, dict) and "choices" in event for event in events):
        return None
    for event in events:
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for key in REASONING_FIELDS:  # the first field a provider uses, as pi-ai reads it
                if isinstance(delta.get(key), str) and delta[key]:
                    reasoning.append(delta[key])
                    break
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(int(call.get("index", 0)),
                                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if call.get("id"):
                    slot["id"] = call["id"]
                function = call.get("function") or {}
                slot["function"]["name"] += function.get("name") or ""
                slot["function"]["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return {"message": message, "finish_reason": finish}


class Gateway:
    """``key`` serves ``/deepseek`` (None: no DeepSeek route); ``routes``: provider -> (base URL, key, headers).

    ``chat`` names the one route prefix (``/deepseek`` or ``/route/<provider>``) whose chat completions the
    session may call; with it set, nothing else is forwarded but DeepSeek's web search (``/deepseek/anthropic/
    v1/…``), so an agent holding the session token cannot reach other endpoints the real key opens, nor another
    route's models. None forwards any GET or POST under a known prefix (tests and tools).
    """

    def __init__(self, *, key: str | None, token: str, log_path: Path, upstream: str = "https://api.deepseek.com",
                 token_budget: int = 3_000_000, deadline: float | None = None, proxy: str | None = None,
                 routes: dict[str, tuple[str, str, dict[str, str]]] | None = None, chat: str | None = None) -> None:
        self.token = token
        self.chat = chat
        self.log_path = log_path
        self._upstreams: dict[str, tuple[Any, str, dict[str, str]]] = {}
        if key:
            self._upstreams[PREFIX] = (urlparse(upstream), key, {})
        for name, (url, route_key, headers) in (routes or {}).items():
            self._upstreams[ROUTE_PREFIX + name] = (urlparse(url), route_key, dict(headers))
        self.token_budget = token_budget
        self.deadline = deadline
        self.proxy = urlparse(proxy) if proxy else None
        self.tokens = 0  # every token the provider reported
        self.metered = 0  # cost-weighted tokens, what the budget counts
        self.calls = 0
        self.answered = 0  # upstream replied 200
        self.failed = 0  # upstream error status or no response
        self.refused = 0
        self.last_failure = ""  # status and provider message of the latest failed call, for the session record
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

    @property
    def has_deepseek(self) -> bool:
        return PREFIX in self._upstreams

    def route_url(self, provider: str) -> str:
        return f"http://127.0.0.1:{self.port}{ROUTE_PREFIX}{provider}"

    def upstream_for(self, path: str) -> tuple[str, tuple[Any, str, dict[str, str]]] | None:
        """The (prefix, upstream) serving a request path, or None."""

        for prefix in sorted(self._upstreams, key=len, reverse=True):
            if path.startswith(prefix + "/"):
                return prefix, self._upstreams[prefix]
        return None

    def permits(self, method: str, prefix: str, rest: str) -> bool:
        """Whether a request for ``rest`` (the path after ``prefix``, without the query) may go upstream."""

        if self.chat is None:
            return True
        if method != "POST":
            return False
        if rest == "/chat/completions":
            return prefix == self.chat
        return prefix == PREFIX and rest.startswith(SEARCH_PATH)

    def start(self) -> Gateway:
        self._server = _QuietServer(("127.0.0.1", 0), _handler(self))
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def usage(self) -> dict[str, Any]:
        return {"tokens_used": self.metered, "tokens_left": max(0, self.token_budget - self.metered),
                "tokens_total": self.tokens, "llm_calls": self.calls}

    def refusal(self) -> str | None:
        if self.deadline is not None and clock.now() > self.deadline:  # awake time: a sleep spends no budget
            return "the session deadline has passed"
        if self.metered >= self.token_budget:
            return f"the session token budget ({self.token_budget} cost-weighted tokens) is spent"
        return None

    def connection(self, upstream: Any) -> http.client.HTTPConnection:
        host = upstream.hostname or ""
        port = upstream.port or (443 if upstream.scheme == "https" else 80)
        if upstream.scheme == "http":
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
            self.metered += metered_tokens(entry.get("usage"))
            if entry.get("status") == 200 and "error" not in entry:
                self.answered += 1
            else:
                self.failed += 1
                detail = entry.get("error") or entry.get("response") or ""
                self.last_failure = f"{entry.get('status') or 'no response'}: {str(detail).strip()[:400]}"
            entry["seq"] = self.calls
            entry["tokens_total"] = self.tokens
            entry["metered_total"] = self.metered
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
            found = gateway.upstream_for(self.path)
            if found is None:
                return self._json(404, f"unknown route {self.path}")
            if not gateway.permits(method, found[0], self.path[len(found[0]):].split("?", 1)[0]):
                with gateway._lock:
                    gateway.refused += 1
                return self._json(403, f"{method} {self.path} is not open in this session: only chat completions "
                                       "on the session's own route, and web search")
            reason = gateway.refusal()
            if reason:
                with gateway._lock:
                    gateway.refused += 1
                # "quota exhausted" is what DSH classifies as QUOTA, which it does not retry: waiting changes
                # nothing here, and the session's retry policy would otherwise keep it for minutes
                return self._json(429, f"session quota exhausted: {reason}")
            prefix, (upstream, key, extra) = found
            path = upstream.path.rstrip("/") + self.path[len(prefix):]
            headers = {name: value for name, value in self.headers.items() if name.lower() not in HOP}
            if self.headers.get("Authorization"):
                headers["Authorization"] = f"Bearer {key}"
            if self.headers.get("x-api-key"):
                headers["x-api-key"] = key
            headers.update(extra)
            headers["Accept-Encoding"] = "identity"
            started = time.monotonic()
            entry: dict[str, Any] = {"at": time.time(), "path": path, "method": method,
                                     "route": prefix.removeprefix(ROUTE_PREFIX).lstrip("/")}
            try:
                entry["request"] = json.loads(body) if body else None
            except json.JSONDecodeError:
                entry["request"] = body.decode("utf-8", "replace")[:100000]
            if isinstance(entry["request"], dict):
                entry["model"] = entry["request"].get("model")
            captured = bytearray()
            content_type = ""
            try:
                connection = gateway.connection(upstream)
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
                    if len(captured) < CAPTURE_LIMIT:
                        captured.extend(chunk[: CAPTURE_LIMIT - len(captured)])
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                connection.close()
            except (OSError, http.client.HTTPException) as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                if "status" not in entry:
                    with contextlib.suppress(OSError):
                        self._json(502, f"upstream failed: {type(error).__name__}")
            finally:
                text = captured.decode("utf-8", "replace")
                entry["usage"] = usage_of(text, content_type)
                reply = assemble(text, content_type) if entry.get("status") == 200 else None
                if reply is not None:
                    entry["reply"] = reply
                elif text:
                    entry["response"] = text[:RAW_LOG_LIMIT]
                entry["seconds"] = round(time.monotonic() - started, 3)
                gateway.record(entry)

    return Handler


__all__ = ["CACHE_WEIGHT", "PREFIX", "ROUTE_PREFIX", "SEARCH_PATH", "Gateway", "assemble", "metered_tokens",
           "tokens_of", "usage_of"]
