"""Run one episode: a DSH agent solves a task in a sandbox, the judge scores its last submission.

Steps: build the workspace (task contract, visible data, the harness), start
the LLM gateway and the ecogym tool server on loopback, launch ``dsh
--profile headless`` inside the agent sandbox with a per-episode DSH home and
a patch that routes models and search through the gateway, wait until the
agent finishes or the deadline passes, judge the last accepted submission on
the hidden scored stage, then store the trajectory and log every step to the
ledger. Nothing the agent writes is trusted: submissions are snapshotted
without following links, and only the judge's numbers are recorded as scores.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__
from ..evaluation.harness import Bench
from ..harness import Harness, resolve
from ..kernel.canonical import digest
from ..kernel.ledger import Ledger
from ..kernel.secrets import resolve_secret
from .gateway import Gateway
from .judge import Limits, judge
from .mcp import McpServer, ToolError
from .sandbox import agent_profile, confine, discover_runtime, ecogym_home, scratch_root
from .tools import EpisodeTools, snapshot
from .trajectory import store
from .workspace import LOCAL_STAGE, SCORED_STAGE, export_visible, task_brief

EVENT_KINDS = frozenset({"episode_started", "episode_evaluated", "episode_submitted", "episode_note",
                         "expert_question", "episode_scored", "episode_finished"})
PROMPT = ("请先阅读当前目录下的 TASK.md 和 AGENTS.md，然后独立完成任务：研究数据，编写并改进 submission/ 中的预测程序，"
          "用 ecogym 工具检验和提交。截止前务必至少成功提交一次。")


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    task: str
    harness: str = "seed"
    minutes: float = 30.0
    tokens: int = 4_000_000
    model: str | None = None
    label: str = ""
    keep: bool = False
    limits: Limits = field(default_factory=Limits)


def dsh_patch(harness: Harness, *, model: str, home: Path, skills: Path, gateway: Gateway, mcp: McpServer) -> str:
    depth = harness.agent.subagent_depth
    search = "true" if harness.agent.web_search else "false"
    rows = f"""- id: credentials
  config:
    path: {json.dumps(str(home / '.credentials.yaml'))}
    watch: false
- id: agent-default-model
  config:
    provider: deepseek-official
    model: {json.dumps(model)}
- id: llm-deepseek
  config:
    baseURL: {json.dumps(gateway.base_url)}
    apiKeyEnv: DEEPSEEK_API_KEY
- id: web-search-deepseek
  config:
    baseURL: {json.dumps(gateway.base_url + '/anthropic/v1')}
    apiKeyEnv: DEEPSEEK_API_KEY
- id: tool-web
  config:
    search: {search}
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
- id: tool-subagent
  config:
    provider: spawn
    toolName: subagent
    backgroundMode: continuable
    maxDepth: {depth}
- id: tool-subagent-fork
  config:
    provider: fork
    toolName: subagent_fork
    backgroundMode: one-shot
    maxDepth: {depth}
- insert:
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


class _Events:
    """Ledger appends from any thread (one connection per append)."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self._lock = threading.Lock()

    def __call__(self, kind: str, body: dict[str, Any]) -> None:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown episode event {kind}")
        with self._lock, Ledger(self.path) as ledger:
            ledger.append(self.run_id, kind, body)


def _dsh_version(dsh: Path, env: dict[str, str]) -> str:
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


def run_episode(config: EpisodeConfig, ledger_path: Path, *, bench: Bench | None = None) -> dict[str, Any]:
    bench = bench or Bench.open(config.task)
    runtime = discover_runtime()
    if runtime.dsh is None or runtime.node_root is None:
        raise FileNotFoundError("dsh not found; install DeepSeek Harness or set ECOGYM_DSH_BIN")
    harness = resolve(config.harness)
    model = config.model or harness.agent.model
    episode_id = f"ep:{uuid.uuid4().hex[:12]}"
    name = episode_id.replace(":", "-")
    root = scratch_root() / name
    trajectory = ecogym_home() / "trajectories" / name
    for path in (root / "work" / "data", root / "home", root / "tmp", trajectory):
        path.mkdir(parents=True, exist_ok=True)
    events = _Events(ledger_path, episode_id)
    started = time.time()
    deadline = started + config.minutes * 60
    harness.install(root)
    visible = export_visible(bench, root / "work" / "data")
    (root / "work" / "TASK.md").write_text(task_brief(bench, visible["boundaries"], minutes=config.minutes,
                                                      tokens=config.tokens, deadline=deadline, limits=config.limits),
                                           encoding="utf-8")
    (root / "work" / "notes").mkdir(exist_ok=True)

    gateway = Gateway(key=resolve_secret("DEEPSEEK_API_KEY"), token=secrets.token_urlsafe(24),
                      log_path=trajectory / "llm_calls.jsonl", token_budget=config.tokens, deadline=deadline,
                      proxy=os.environ.get("ECOGYM_UPSTREAM_PROXY") or None).start()
    usage = lambda: {"tokens_used": gateway.tokens, "tokens_left": max(0, config.tokens - gateway.tokens),  # noqa: E731
                     "llm_calls": gateway.calls}
    tools = EpisodeTools(bench=bench, root=root, runtime=runtime, limits=config.limits, deadline=deadline,
                         usage=usage, event=events)
    mcp_log = (trajectory / "mcp_calls.jsonl").open("a", encoding="utf-8")
    log_lock = threading.Lock()

    def log_call(entry: dict[str, Any]) -> None:
        with log_lock:
            mcp_log.write(json.dumps({"at": time.time(), **entry}, ensure_ascii=False) + "\n")
            mcp_log.flush()

    mcp = McpServer(tools.catalog(), token=secrets.token_urlsafe(24), log=log_call).start()
    patch = root / "episode.patch.yml"
    patch.write_text(dsh_patch(harness, model=model, home=root / "home", skills=root / "harness" / "skills",
                               gateway=gateway, mcp=mcp), encoding="utf-8")
    path = os.pathsep.join([str(runtime.dsh.parent), str(runtime.python.parent), "/usr/bin", "/bin", "/usr/sbin",
                            "/sbin"])
    env = {"PATH": path, "HOME": str(root / "home"), "DSH_HOME": str(root / "home"), "TMPDIR": str(root / "tmp"),
           "LANG": "en_US.UTF-8", "DSH_PERMISSION_MODE": "danger-full-access", "DEEPSEEK_API_KEY": gateway.token,
           "DEEPSEEK_BASE_URL": gateway.base_url, "DEEPSEEK_SEARCH_BASE_URL": gateway.base_url + "/anthropic/v1",
           "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost", "PYTHONDONTWRITEBYTECODE": "1"}
    profile = agent_profile(root, [gateway.port, mcp.port], runtime)
    argv = confine(profile, root / "agent.sb",
                   [str(runtime.dsh), "--profile", "headless", "--patch", str(patch), PROMPT])
    events("episode_started", {
        "task_id": bench.task.id, "label": config.label, "bench": bench.identity, "ecogym": __version__,
        "harness": {"digest": harness.digest, "name": harness.name, "parent": harness.parent},
        "model": model, "minutes": config.minutes, "tokens": config.tokens, "limits": asdict(config.limits),
        "dsh": _dsh_version(runtime.dsh, env), "sandbox": digest(profile.replace(str(root), "<episode>")),
        "visible_rows": visible["rows"], "boundaries": visible["boundaries"],
    })
    status, exit_code, killed = "finished", None, False
    stdout_path, stderr_path = trajectory / "agent_stdout.txt", trajectory / "agent_stderr.txt"
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(argv, cwd=root / "work", env=env, stdin=subprocess.DEVNULL, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            try:
                exit_code = process.wait(timeout=max(1.0, deadline - time.time()) + 30)
            except subprocess.TimeoutExpired:
                killed = True
            finally:
                _stop(process, root)
                exit_code = process.returncode if exit_code is None else exit_code
    finally:
        mcp.stop()
        gateway.stop()
        mcp_log.close()

    final = tools.final()
    scored = None
    if final is None:
        fallback = root / "submissions" / "final-fallback" / "program"
        fallback.mkdir(parents=True, exist_ok=True)
        try:
            final = {"n": None, "digest": snapshot(root / "work" / "submission", fallback), "program": str(fallback),
                     "auto": True}
        except ToolError:
            final = None
    if final is not None:
        report = judge(bench, Path(final["program"]), SCORED_STAGE, runtime, limits=config.limits)
        scored = {key: report[key] for key in ("stage", "status", "origins", "score", "interval", "coverage", "cells",
                                               "fit_seconds", "seconds", "errors", "references")}
        scored.update({"n": final.get("n"), "digest": final["digest"], "auto": bool(final.get("auto"))})
        events("episode_scored", scored)
    else:
        status = "no_submission"
    stored = store(root, trajectory, runtime.node_root / "bin" / "node")
    final_text = stdout_path.read_text(encoding="utf-8", errors="replace")[-8000:]
    finished = {
        "status": status, "seconds": round(time.time() - started, 1), "agent_exit": exit_code, "killed": killed,
        "tokens": gateway.tokens, "llm_calls": gateway.calls, "refused_calls": gateway.refused,
        "local_evaluations": tools.evaluations, "submissions": len(tools.submissions),
        "notes": len(tools.notes), "questions": len(tools.questions), "activity": stored["summary"],
        "final_text": final_text, "trajectory": {"dir": str(trajectory), "files": stored["files"]},
        "score": None if scored is None else scored["score"], "local_check": LOCAL_STAGE,
    }
    events("episode_finished", finished)
    if not config.keep:
        subprocess.run(["rm", "-r", "-f", str(root)], check=False)
    return {"episode": episode_id, **finished, "scored": scored}


__all__ = ["EVENT_KINDS", "EpisodeConfig", "dsh_patch", "run_episode"]
