"""A minimal MCP tool server over Streamable HTTP with plain JSON responses.

Enough of MCP for DSH's client: ``initialize``, ``notifications/initialized``,
``tools/list``, ``tools/call`` and ``ping``; GET (server-initiated streams)
answers 405. Every request needs ``Authorization: Bearer <token>``. Tool
handlers run on the server's threads, so sub-agents can call concurrently.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..kernel.canonical import jsonable

PROTOCOL_VERSION = "2025-06-18"


class _QuietServer(ThreadingHTTPServer):
    """Dropped connections are normal when the episode is torn down; don't dump their tracebacks."""

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            super().handle_error(request, client_address)


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[..., Any]


class ToolError(Exception):
    """A tool refused the call; the message goes back to the model."""


class McpServer:
    def __init__(self, tools: list[Tool], *, token: str, name: str = "ecogym",
                 log: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.token = token
        self.name = name
        self.log = log
        self.session = uuid.uuid4().hex
        self._server: ThreadingHTTPServer | None = None

    @property
    def port(self) -> int:
        assert self._server is not None
        return int(self._server.server_address[1])

    def start(self) -> McpServer:
        self._server = _QuietServer(("127.0.0.1", 0), _handler(self))
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def dispatch(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method, params, ident = message.get("method"), message.get("params") or {}, message.get("id")
        if ident is None:
            return None
        if method == "initialize":
            result: Any = {"protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                           "capabilities": {"tools": {"listChanged": False}},
                           "serverInfo": {"name": self.name, "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": tool.name, "description": tool.description, "inputSchema": tool.schema}
                                for tool in self.tools.values()]}
        elif method == "tools/call":
            result = self.call(str(params.get("name")), params.get("arguments") or {})
        elif method == "ping":
            result = {}
        else:
            return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": f"no method {method}"}}
        return {"jsonrpc": "2.0", "id": ident, "result": result}

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        tool = self.tools.get(name)
        error = None
        try:
            if tool is None:
                raise ToolError(f"unknown tool {name}; tools: {sorted(self.tools)}")
            if not isinstance(arguments, dict):
                raise ToolError("arguments must be an object")
            value = tool.handler(**arguments)
            text = value if isinstance(value, str) else json.dumps(jsonable(value), ensure_ascii=False, indent=1)
        except ToolError as failure:
            error = text = str(failure)
        except TypeError as failure:
            error = text = f"bad arguments for {name}: {failure}"
        except Exception as failure:  # reported to the model and logged, never raised
            error = text = f"{type(failure).__name__}: {failure}"
        if self.log is not None:
            self.log({"tool": name, "arguments": arguments, "error": error, "result": text[:20000],
                      "seconds": round(time.monotonic() - started, 3)})
        return {"content": [{"type": "text", "text": text}], "isError": error is not None}


def _handler(server: McpServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # quiet
            return None

        def _reply(self, status: int, body: Any = None, headers: dict[str, str] | None = None) -> None:
            data = b"" if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            if body is not None:
                self.send_header("Content-Type", "application/json")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            return self.headers.get("Authorization") == f"Bearer {server.token}"

        def do_GET(self) -> None:
            self._reply(405)

        def do_DELETE(self) -> None:
            self._reply(200 if self._authorized() else 401)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            if not self._authorized():
                return self._reply(401, {"error": "unauthorized"})
            try:
                message = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return self._reply(400, {"jsonrpc": "2.0", "id": None,
                                         "error": {"code": -32700, "message": "parse error"}})
            if isinstance(message, list):
                replies = [reply for item in message if (reply := server.dispatch(item)) is not None]
                return self._reply(200, replies) if replies else self._reply(202)
            reply = server.dispatch(message)
            if reply is None:
                return self._reply(202)
            headers = {"Mcp-Session-Id": server.session} if message.get("method") == "initialize" else None
            self._reply(200, reply, headers)

    return Handler


__all__ = ["McpServer", "Tool", "ToolError"]
