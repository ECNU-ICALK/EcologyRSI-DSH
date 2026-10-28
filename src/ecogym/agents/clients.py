"""LLM transports.

Both transports implement ``complete(messages) -> Reply`` over a plain chat
transcript; tool use is handled above them with a JSON reply protocol, so any
chat model works.

* ``http``: an OpenAI-compatible ``/chat/completions`` endpoint. The API key
  comes from the environment variable named by ``key_env``, or, if that is
  unset, from a ``NAME: value`` line in the file named by
  ``$ECOGYM_CREDENTIALS_FILE``. Keys are never logged.
* ``dsh``: one ``dsh --profile headless`` process per call, using DSH's
  provider configuration and credentials. A patch selects the model and
  disables every DSH tool (shell, files, web, sub-agents), and the process
  runs in an empty scratch directory, so the model can only answer.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from ..kernel.secrets import SecretMissing
from ..kernel.secrets import resolve_secret as _resolve_secret

Message = Mapping[str, str]  # {"role": "system" | "user" | "assistant", "content": str}
LOG = logging.getLogger("ecogym.agents")

DSH_DISABLED_TOOLS = (
    "tool-bash", "tool-pwsh", "tool-fs", "tool-fs-search", "tool-web", "tool-subagent", "tool-subagent-fork",
    "tool-subagent-control", "tool-subagent-list-agents", "tool-workflow", "tool-jobs", "tool-skill",
    "tool-todo", "tool-goal",
)


class AgentError(RuntimeError):
    """An LLM call failed after retries, or a reply could not be used."""


@dataclass(frozen=True, slots=True)
class Reply:
    text: str
    model: str
    seconds: float
    usage: dict[str, Any] = field(default_factory=dict)


class ChatClient(Protocol):
    def describe(self) -> dict[str, Any]: ...

    def complete(self, messages: Sequence[Message]) -> Reply: ...


def resolve_secret(name: str) -> str:
    try:
        return _resolve_secret(name)
    except SecretMissing as error:
        raise AgentError(str(error)) from None


@dataclass
class OpenAICompatibleClient:
    base_url: str
    model: str
    key_env: str = "ECOGYM_LLM_API_KEY"
    temperature: float | None = 0.4
    max_tokens: int = 16000
    timeout: float = 300.0  # per socket read; a streamed reply resets it with every chunk
    retries: int = 4
    stream: bool = True
    min_interval: float = 0.0  # seconds between call starts, for rate-limited providers
    max_backoff: float = 300.0
    proxy: str = "system"  # "system" (environment/OS settings), "none", or a proxy URL
    max_tokens_cap: int = 64000  # a reply cut off by the token limit is retried with up to this budget
    _last_call: float = field(default=0.0, init=False, repr=False)
    _finish: str | None = field(default=None, init=False, repr=False)

    # 410 is what some provider firewalls return for rate-based blocks; it clears after a pause.
    RETRYABLE = (408, 409, 410, 425, 429, 500, 502, 503, 504)

    def describe(self) -> dict[str, Any]:
        return {"transport": "http", "base_url": self.base_url, "model": self.model, "key_env": self.key_env,
                "temperature": self.temperature, "max_tokens": self.max_tokens}

    def _pace(self) -> None:
        wait = self._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _backoff(self, attempt: int, error: Exception | None) -> float:
        delay = min(self.max_backoff, 10.0 * 2**attempt)
        if isinstance(error, urllib.error.HTTPError):
            retry_after = error.headers.get("Retry-After") if error.headers else None
            if retry_after and retry_after.strip().isdigit():
                delay = min(self.max_backoff, max(delay, float(retry_after)))
        return delay

    def _opener(self) -> urllib.request.OpenerDirector:
        host = urlparse(self.base_url).hostname or ""
        if self.proxy == "none" or host in ("127.0.0.1", "localhost", "::1"):
            return urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if self.proxy != "system":
            return urllib.request.build_opener(urllib.request.ProxyHandler({"http": self.proxy, "https": self.proxy}))
        return urllib.request.build_opener()

    def _read(self, response: Any) -> tuple[str, str, dict[str, Any]]:
        """Collect a streamed (server-sent events) or plain completion."""

        if "text/event-stream" not in (response.headers.get("Content-Type") or ""):
            payload = json.loads(response.read().decode("utf-8"))
            choice = payload["choices"][0]
            self._finish = choice.get("finish_reason")
            return (choice["message"].get("content") or "", str(payload.get("model", self.model)),
                    dict(payload.get("usage") or {}))
        parts, model, usage, finished = [], self.model, {}, False
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                finished = True
                break
            chunk = json.loads(data)
            model = str(chunk.get("model") or model)
            usage = dict(chunk.get("usage") or usage)
            for choice in chunk.get("choices") or []:
                parts.append((choice.get("delta") or {}).get("content") or "")
                if choice.get("finish_reason"):
                    finished = True
                    self._finish = choice["finish_reason"]
        if not finished:
            # The connection closed mid-reply; a partial answer must not pass as a whole one.
            raise http.client.IncompleteRead("".join(parts).encode("utf-8"))
        return "".join(parts), model, usage

    def _request(self, messages: Sequence[Message], max_tokens: int) -> urllib.request.Request:
        body: dict[str, Any] = {"model": self.model, "messages": [dict(m) for m in messages],
                                "max_tokens": max_tokens, "stream": self.stream}
        if self.stream:
            body["stream_options"] = {"include_usage": True}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {resolve_secret(self.key_env)}"},
        )

    def complete(self, messages: Sequence[Message]) -> Reply:
        opener = self._opener()
        budget = self.max_tokens
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            self._pace()
            started = time.monotonic()
            self._finish = None
            try:
                with opener.open(self._request(messages, budget), timeout=self.timeout) as response:
                    text, model, usage = self._read(response)
                if text.strip() and self._finish != "length":
                    return Reply(text, model, round(time.monotonic() - started, 2), usage)
                if self._finish == "length":
                    # A reasoning model spent the whole budget before (or while) answering: give it more room.
                    budget = min(self.max_tokens_cap, budget * 2)
                    last = AgentError(f"the reply hit the {budget // 2}-token limit; retrying with {budget}")
                else:
                    last = AgentError("the model returned an empty message")
            except urllib.error.HTTPError as error:
                last = error
                if error.code not in self.RETRYABLE:
                    detail = error.read().decode("utf-8", "replace")[:300]
                    raise AgentError(f"HTTP {error.code} from {self.base_url}: {detail}") from None
            except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError, TypeError) as error:
                last = error  # network failures, cut streams, malformed payloads
            if attempt < self.retries:
                delay = self._backoff(attempt, last)
                LOG.warning("%s %s: attempt %d failed (%s: %s); retrying in %.0f s", self.base_url, self.model,
                            attempt + 1, type(last).__name__, str(last)[:120], delay)
                time.sleep(delay)
        raise AgentError(f"LLM call failed after {self.retries + 1} attempts: {type(last).__name__}: {last}")


@dataclass
class FallbackClient:
    """Tries each client in turn for every call; the first answer wins."""

    clients: tuple[ChatClient, ...]

    def describe(self) -> dict[str, Any]:
        first = self.clients[0].describe()
        return {**first, "fallbacks": [client.describe() for client in self.clients[1:]]}

    def complete(self, messages: Sequence[Message]) -> Reply:
        errors = []
        for client in self.clients:
            try:
                return client.complete(messages)
            except AgentError as error:
                errors.append(f"{client.describe().get('model')}: {error}")
        raise AgentError("every endpoint failed: " + " | ".join(errors))


def render_transcript(messages: Sequence[Message]) -> str:
    parts = [f"[{m['role']}]\n{m['content']}" for m in messages]
    return "\n\n".join(parts) + "\n\n[assistant]\nReply now as the assistant, with the reply only."


@dataclass
class DshHeadlessClient:
    model: str  # "provider/model" as configured in DSH
    dsh_bin: str = "dsh"
    dsh_home: str | None = None
    patches: tuple[str, ...] = ()
    path_prepend: str | None = None  # e.g. the directory holding node
    timeout: float = 900.0

    def describe(self) -> dict[str, Any]:
        return {"transport": "dsh", "model": self.model, "dsh_bin": self.dsh_bin, "patches": list(self.patches)}

    def _patch(self, directory: Path) -> Path:
        provider, _, model = self.model.partition("/")
        if not provider or not model:
            raise AgentError("the dsh transport needs model='provider/model'")
        lines = ["- id: agent-default-model", "  config:", f"    provider: {provider}", f"    model: {model}"]
        for tool in DSH_DISABLED_TOOLS:
            lines += [f"- id: {tool}", "  disabled: true"]
        path = directory / "ecogym-dsh.patch.yml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def complete(self, messages: Sequence[Message]) -> Reply:
        env = dict(os.environ)
        if self.dsh_home:
            env["DSH_HOME"] = self.dsh_home
        if self.path_prepend:
            env["PATH"] = self.path_prepend + os.pathsep + env.get("PATH", "")
        env.setdefault("NO_PROXY", "127.0.0.1,localhost")
        env.setdefault("no_proxy", "127.0.0.1,localhost")
        with tempfile.TemporaryDirectory(prefix="ecogym-dsh-") as scratch:
            command = [self.dsh_bin, "--profile", "headless"]
            for patch in (*self.patches, str(self._patch(Path(scratch)))):
                command += ["--patch", str(Path(patch).expanduser())]
            command.append("-")
            started = time.monotonic()
            try:
                done = subprocess.run(command, input=render_transcript(messages), capture_output=True, text=True,
                                      cwd=scratch, env=env, timeout=self.timeout, check=False)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise AgentError(f"dsh headless failed: {type(error).__name__}: {error}") from None
        if done.returncode != 0 or not done.stdout.strip():
            raise AgentError(f"dsh headless exited {done.returncode}: {done.stderr.strip()[-400:]}")
        return Reply(done.stdout.strip(), self.model, round(time.monotonic() - started, 2))


def make_client(options: Mapping[str, Any]) -> ChatClient:
    """A client from proposer options; ``fallbacks`` lists further option sets tried in order."""

    fallbacks = options.get("fallbacks") or ()
    if fallbacks:
        primary = {key: value for key, value in options.items() if key != "fallbacks"}
        return FallbackClient(tuple(make_client({**primary, **dict(item)}) for item in [{}, *fallbacks]))
    transport = options.get("transport", "http")
    if transport == "http":
        return OpenAICompatibleClient(
            base_url=str(options.get("base_url") or os.environ.get("ECOGYM_LLM_BASE_URL", "")),
            model=str(options.get("model") or os.environ.get("ECOGYM_LLM_MODEL", "")),
            key_env=str(options.get("key_env", "ECOGYM_LLM_API_KEY")),
            temperature=options.get("temperature", 0.4),
            max_tokens=int(options.get("max_tokens", 16000)),
            timeout=float(options.get("timeout", 300.0)),
            retries=int(options.get("retries", 4)),
            stream=bool(options.get("stream", True)),
            min_interval=float(options.get("min_interval", 0.0)),
            max_backoff=float(options.get("max_backoff", 300.0)),
            proxy=str(options.get("proxy", "system")),
        )
    if transport == "dsh":
        return DshHeadlessClient(
            model=str(options.get("model", "")),
            dsh_bin=str(options.get("dsh_bin") or os.environ.get("ECOGYM_DSH_BIN", "dsh")),
            dsh_home=options.get("dsh_home") or os.environ.get("DSH_HOME"),
            patches=tuple(options.get("patches", ())),
            path_prepend=options.get("path_prepend"),
            timeout=float(options.get("timeout", 900.0)),
        )
    raise AgentError(f"unknown transport {transport}; use http or dsh")


__all__ = [
    "AgentError",
    "ChatClient",
    "DshHeadlessClient",
    "FallbackClient",
    "Message",
    "OpenAICompatibleClient",
    "Reply",
    "make_client",
    "render_transcript",
    "resolve_secret",
]
