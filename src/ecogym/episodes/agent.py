"""One sandboxed DSH agent session: gateway, tool server, process and trajectory.

The caller lays out ``root/work`` (the agent's working directory) and
``root/harness/skills``, creates the ``Gateway`` (``make_gateway``) so its tools
can report usage, and passes the MCP tools. ``run_agent`` starts the gateway
and the tool server on loopback, writes a DSH patch that routes models and web
search through the gateway and mounts the tool server, launches ``dsh
--profile headless`` in the agent sandbox with a per-session DSH home, waits
until the agent finishes or the deadline passes, and stops everything, also
when interrupted. The trajectory (DSH sessions, every model call, every tool
call, the workspace) is stored in every case.

The DSH kernel is never modified: a harness only chooses the options in
``AgentSpec``; credentials, sandbox, disabled tools and the tool server are
fixed by this module. The model may be any runnable route of the user's DSH
(``ecogym.episodes.providers``): the session's DSH then knows that one route,
pointed at the gateway, and the real endpoint and key stay on the host. Web
search uses DeepSeek's API and is switched off when no DeepSeek key exists.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..harness import AgentOptions
from ..kernel.secrets import has_secret, resolve_secret
from . import providers
from .gateway import PREFIX, ROUTE_PREFIX, Gateway
from .mcp import McpServer, Tool
from .sandbox import Runtime, agent_profile, confine
from .trajectory import store

ROUTE_KEY = "ECOGYM_ROUTE_API_KEY"  # the session's token for a provider route (never a real key)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """The DSH options of one session (a harness's [agent] table, optionally with a pinned model)."""

    model: str  # <provider>/<model> of the DSH settings, or a DeepSeek model id (ecogym.episodes.providers)
    reasoning_effort: str = "high"
    web_search: bool = True
    subagent_depth: int = 1
    parallel_tool_calls: int = 10

    @classmethod
    def of(cls, options: AgentOptions, model: str | None = None) -> AgentSpec:
        return cls(model or options.model, options.reasoning_effort, options.web_search, options.subagent_depth,
                   options.parallel_tool_calls)


@dataclass(slots=True)
class Session:
    status: str  # finished | timeout | infra_failed (no model call was answered)
    exit_code: int | None
    killed: bool
    seconds: float
    tokens: int  # every token the provider reported
    metered_tokens: int  # cost-weighted (cached prompt tokens at CACHE_WEIGHT), what the budget counts
    llm_calls: int
    answered_calls: int
    failed_calls: int
    refused_calls: int
    final_text: str
    trajectory: dict[str, Any]
    activity: dict[str, Any]
    upstream_error: str = ""  # the provider's answer to the latest failed model call (why infra_failed)
    continuations: int = 0  # follow-up sessions after the agent ended early (see run_agent's follow_up)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_gateway(trajectory: Path, *, tokens: int, deadline: float, model: str) -> Gateway:
    """The session's gateway for ``model`` (``<provider>/<model>`` or a DeepSeek id); fails if it cannot run.

    The DeepSeek route also serves web search, so it is kept, when a key exists, for other providers too; the
    gateway then forwards only chat completions on the session's own route and web search (``Gateway.chat``).
    """

    route, _ = providers.resolve(model)
    extra = {}
    if route.provider != providers.DEEPSEEK:
        extra[route.provider] = (route.base_url, resolve_secret(route.key_env), route.headers)
    needs_deepseek = route.provider == providers.DEEPSEEK or has_secret(providers.DEEPSEEK_KEY)
    chat = PREFIX if route.provider == providers.DEEPSEEK else ROUTE_PREFIX + route.provider
    return Gateway(key=resolve_secret(providers.DEEPSEEK_KEY) if needs_deepseek else None,
                   token=secrets.token_urlsafe(24), log_path=trajectory / "llm_calls.jsonl", token_budget=tokens,
                   deadline=deadline, proxy=os.environ.get("ECOGYM_UPSTREAM_PROXY") or None, routes=extra, chat=chat)


def _model_rows(agent: AgentSpec, gateway: Gateway) -> str:
    """The session's default model and, for a provider route, that route alone, through the gateway."""

    provider, model = providers.split(agent.model)
    rows = (f"- id: agent-default-model\n  config:\n    provider: {json.dumps(provider)}\n"
            f"    model: {json.dumps(model)}\n")
    if provider != providers.DEEPSEEK:
        route, model = providers.resolve(agent.model)
        profile = route.session_profile(model, agent.reasoning_effort, gateway.route_url(provider), ROUTE_KEY)
        rows += f"- id: llm-pi-ai\n  config: {json.dumps({'providers': {provider: profile}}, ensure_ascii=False)}\n"
    return rows


def dsh_patch(agent: AgentSpec, *, home: Path, skills: Path, gateway: Gateway, mcp: McpServer) -> str:
    thinking = "disabled" if agent.reasoning_effort == "off" else "enabled"
    search = agent.web_search and gateway.has_deepseek  # web search runs on DeepSeek's API
    rows = f"""- id: credentials
  config:
    path: {json.dumps(str(home / '.credentials.yaml'))}
    watch: false
{_model_rows(agent, gateway)}- id: llm-deepseek
  config:
    baseURL: {json.dumps(gateway.base_url)}
    apiKeyEnv: DEEPSEEK_API_KEY
    thinking: {thinking}
    reasoningEffort: {json.dumps(agent.reasoning_effort)}
- id: web-search-deepseek
  config:
    baseURL: {json.dumps(gateway.base_url + '/anthropic/v1')}
    apiKeyEnv: DEEPSEEK_API_KEY
- id: tool-web
  config:
    search: {"true" if search else "false"}
    fetch: false
    searchTimeoutMs: 90000
- id: web-fetch-http
  disabled: true
- id: session-title-llm
  disabled: true
- id: session-telemetry-otel
  disabled: true
- id: tool-workflow
  disabled: true
- id: skill-filesystem
  config:
    includeDefaultRoots: false
    customSkillDirs:
      - {json.dumps(str(skills))}
- id: agent-loop
  config:
    agents: []
    maxParallelToolCalls: {int(agent.parallel_tool_calls)}
"""
    if agent.subagent_depth > 0:
        for row, provider, mode in (("tool-subagent", "spawn", "continuable"), ("tool-subagent-fork", "fork",
                                                                                 "one-shot")):
            tool = row.removeprefix("tool-").replace("-", "_")
            rows += (f"- id: {row}\n  config:\n    provider: {provider}\n    toolName: {tool}\n"
                     f"    backgroundMode: {mode}\n    maxDepth: {int(agent.subagent_depth)}\n")
    else:
        for row in ("tool-subagent", "tool-subagent-fork", "tool-subagent-control", "tool-subagent-list-agents",
                    "tool-ralph"):
            rows += f"- id: {row}\n  disabled: true\n"
    rows += f"""- insert:
    - id: mcp-ecogym
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: ecogym
        transport: streamable-http
        url: {json.dumps(f'http://127.0.0.1:{mcp.port}/mcp')}
        headers:
          Authorization: {json.dumps(f'Bearer {mcp.token}')}
        failOnStartupError: true
        toolCallTimeoutMs: 1200000
"""
    return rows


def dsh_version(dsh: Path, env: dict[str, str]) -> str:
    try:
        done = subprocess.run([str(dsh), "--version"], capture_output=True, text=True, timeout=60, env=env)
        return done.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _stop(process: subprocess.Popen[bytes], root: Path) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            continue
    subprocess.run(["pkill", "-f", str(root)], capture_output=True, check=False)  # agent jobs in other groups


def session_env(root: Path, runtime: Runtime, gateway: Gateway) -> dict[str, str]:
    assert runtime.dsh is not None
    path = os.pathsep.join([str(runtime.dsh.parent), str(runtime.python.parent), "/usr/bin", "/bin", "/usr/sbin",
                            "/sbin"])
    return {"PATH": path, "HOME": str(root / "home"), "DSH_HOME": str(root / "home"), "TMPDIR": str(root / "tmp"),
            "LANG": "en_US.UTF-8", "DSH_PERMISSION_MODE": "danger-full-access", "DEEPSEEK_API_KEY": gateway.token,
            ROUTE_KEY: gateway.token,
            "DEEPSEEK_BASE_URL": gateway.base_url, "DEEPSEEK_SEARCH_BASE_URL": gateway.base_url + "/anthropic/v1",
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost", "PYTHONDONTWRITEBYTECODE": "1"}


class Cancelled(RuntimeError):
    """The session was cancelled from outside (the process has been stopped)."""


def run_agent(*, root: Path, prompt: str, agent: AgentSpec, tools: list[Tool], gateway: Gateway, deadline: float,
              trajectory: Path, runtime: Runtime, on_start: Callable[[dict[str, Any]], None] | None = None,
              cancel: threading.Event | None = None,
              follow_up: Callable[[int, str], str | None] | None = None) -> Session:
    """Run one session to completion or the deadline; ``on_start`` gets the DSH version and sandbox profile.

    Setting ``cancel`` stops the agent within seconds and raises ``Cancelled`` (after the trajectory is stored).
    When the agent ends on its own, ``follow_up(n, last_reply)`` may return the prompt of continuation ``n``
    (a new DSH session in the same workspace, with the same gateway, tools and deadline), or None to stop.
    """

    if runtime.dsh is None or runtime.node_root is None:
        raise FileNotFoundError("dsh not found; install DeepSeek Harness or set ECOGYM_DSH_BIN")
    for path in (root / "work", root / "home", root / "tmp", root / "harness" / "skills", trajectory):
        path.mkdir(parents=True, exist_ok=True)
    started = time.time()
    mcp_log = (trajectory / "mcp_calls.jsonl").open("a", encoding="utf-8")
    log_lock = threading.Lock()

    def log_call(entry: dict[str, Any]) -> None:
        with log_lock:
            mcp_log.write(json.dumps({"at": time.time(), **entry}, ensure_ascii=False) + "\n")
            mcp_log.flush()

    gateway.start()
    mcp = McpServer(tools, token=secrets.token_urlsafe(24), log=log_call).start()
    exit_code, killed, continuations = None, False, 0
    stdout_path, stderr_path = trajectory / "agent_stdout.txt", trajectory / "agent_stderr.txt"
    try:
        patch = root / "session.patch.yml"
        patch.write_text(dsh_patch(agent, home=root / "home", skills=root / "harness" / "skills", gateway=gateway,
                                   mcp=mcp), encoding="utf-8")
        env = session_env(root, runtime, gateway)
        profile = agent_profile(root, [gateway.port, mcp.port], runtime)
        if on_start is not None:
            on_start({"dsh": dsh_version(runtime.dsh, env), "profile": profile})
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            text: str | None = prompt
            while text is not None:
                offset = stdout_path.stat().st_size  # this session's reply starts here
                argv = confine(profile, root / "agent.sb",
                               [str(runtime.dsh), "--profile", "headless", "--patch", str(patch), text])
                process = subprocess.Popen(argv, cwd=root / "work", env=env, stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                exit_code = None
                try:
                    while exit_code is None:
                        if cancel is not None and cancel.is_set():
                            raise Cancelled("the session was cancelled")
                        if time.time() > deadline + 30:
                            killed = True
                            break
                        try:
                            exit_code = process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            continue
                finally:
                    _stop(process, root)
                    exit_code = process.returncode if exit_code is None else exit_code
                text = None
                if follow_up is not None and not killed and exit_code == 0 and gateway.answered:
                    stdout.flush()
                    reply = stdout_path.read_bytes()[offset:].decode("utf-8", "replace")
                    text = follow_up(continuations + 1, reply[-3000:])
                if text is not None:
                    continuations += 1
                    stdout.write(f"\n\n---- 续跑 {continuations} ----\n\n".encode())
                    stdout.flush()
    finally:
        mcp.stop()
        gateway.stop()
        mcp_log.close()
        try:
            stored = store(root, trajectory, runtime.node_root / "bin" / "node")
        except Exception as error:  # a broken trajectory must not hide the session's own outcome
            stored = {"files": {}, "summary": {"error": f"{type(error).__name__}: {error}"}}
    status = "infra_failed" if gateway.answered == 0 else "timeout" if killed else "finished"
    final_text = stdout_path.read_text(encoding="utf-8", errors="replace")[-8000:] if stdout_path.exists() else ""
    return Session(status=status, exit_code=exit_code, killed=killed, seconds=round(time.time() - started, 1),
                   tokens=gateway.tokens, metered_tokens=gateway.metered, llm_calls=gateway.calls,
                   answered_calls=gateway.answered,
                   failed_calls=gateway.failed, refused_calls=gateway.refused, final_text=final_text,
                   trajectory={"dir": str(trajectory), "files": stored["files"]}, activity=stored["summary"],
                   upstream_error=gateway.last_failure, continuations=continuations)


__all__ = ["AgentSpec", "Cancelled", "Session", "dsh_patch", "dsh_version", "make_gateway", "run_agent",
           "session_env"]
