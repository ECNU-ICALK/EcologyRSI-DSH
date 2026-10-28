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

import json
import os
import re
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

Message = Mapping[str, str]  # {"role": "system" | "user" | "assistant", "content": str}

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
    value = os.environ.get(name, "").strip()
    if value:
        return value
    path = os.environ.get("ECOGYM_CREDENTIALS_FILE", "").strip()
    if path:
        pattern = re.compile(rf"^\s*{re.escape(name)}\s*:\s*(.+?)\s*$")
        for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            if match:
                return match.group(1).strip("'\"")
    raise AgentError(f"no API key: set {name} or point ECOGYM_CREDENTIALS_FILE at a file with a '{name}: ...' line")


@dataclass
class OpenAICompatibleClient:
    base_url: str
    model: str
    key_env: str = "ECOGYM_LLM_API_KEY"
    temperature: float | None = 0.4
    max_tokens: int = 16000
    timeout: float = 300.0
    retries: int = 4

    # 410 is what some provider firewalls return for rate-based blocks; it clears after a pause.
    RETRYABLE = (408, 409, 410, 425, 429, 500, 502, 503, 504)

    def describe(self) -> dict[str, Any]:
        return {"transport": "http", "base_url": self.base_url, "model": self.model, "key_env": self.key_env,
                "temperature": self.temperature, "max_tokens": self.max_tokens}

    def _opener(self) -> urllib.request.OpenerDirector:
        host = urlparse(self.base_url).hostname or ""
        if host in ("127.0.0.1", "localhost", "::1"):
            return urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return urllib.request.build_opener()

    def complete(self, messages: Sequence[Message]) -> Reply:
        body: dict[str, Any] = {"model": self.model, "messages": [dict(m) for m in messages],
                                "max_tokens": self.max_tokens}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {resolve_secret(self.key_env)}"},
        )
        opener = self._opener()
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                message = payload["choices"][0]["message"]
                text = message.get("content") or ""
                if not text.strip():
                    raise AgentError("the model returned an empty message (raise max_tokens for reasoning models)")
                return Reply(text, str(payload.get("model", self.model)), round(time.monotonic() - started, 2),
                             dict(payload.get("usage") or {}))
            except urllib.error.HTTPError as error:
                last = error
                if error.code not in self.RETRYABLE:
                    detail = error.read().decode("utf-8", "replace")[:300]
                    raise AgentError(f"HTTP {error.code} from {self.base_url}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError) as error:
                last = error
            if attempt < self.retries:
                time.sleep(min(120.0, 10.0 * 2**attempt))
        raise AgentError(f"LLM call failed after {self.retries + 1} attempts: {type(last).__name__}: {last}")


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
    transport = options.get("transport", "http")
    if transport == "http":
        return OpenAICompatibleClient(
            base_url=str(options.get("base_url") or os.environ.get("ECOGYM_LLM_BASE_URL", "")),
            model=str(options.get("model") or os.environ.get("ECOGYM_LLM_MODEL", "")),
            key_env=str(options.get("key_env", "ECOGYM_LLM_API_KEY")),
            temperature=options.get("temperature", 0.4),
            max_tokens=int(options.get("max_tokens", 16000)),
            timeout=float(options.get("timeout", 300.0)),
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
    "Message",
    "OpenAICompatibleClient",
    "Reply",
    "make_client",
    "render_transcript",
    "resolve_secret",
]
