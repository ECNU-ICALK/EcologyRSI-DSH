"""Run one episode: a DSH agent solves a task in a sandbox, the judge scores its last submission.

Steps: build the workspace (task contract, visible data, expert guidance, the
harness), judge the starting program on the local check, run one agent
session (``ecogym.episodes.agent``) with the ecogym tools, judge the scored
program on the hidden scored stage (the latest accepted submission that beats
the starting program locally by ``REPLACE_MARGIN``, else the starting program;
see ``ecogym.episodes.tools``), store its forecasts next to the trajectory,
and log every step to the ledger.
Nothing the agent writes is trusted: submissions are snapshotted without
following links, and only the judge's numbers are recorded as scores.

The time budget counts awake time (``ecogym.kernel.clock``): a machine that
sleeps pauses the episode instead of spending its minutes.

Agents tend to declare the task done long before their time is up. An agent
that ends with more than a quarter of its time left gets a follow-up session
in the same workspace (at most three), told how much time is left, what its
current submission is and what it last replied; every harness version is
treated the same, so this does not bias comparisons.

An episode whose agent never got a model reply is ``infra_failed``: a
failure of the service, not of the harness, which callers may retry. An
episode stopped during its session or its judging (``cancel``) is recorded as
``interrupted``, unscored, before the interruption propagates.
"""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__
from ..evaluation.harness import Bench
from ..harness import resolve
from ..kernel import clock
from ..kernel.canonical import digest
from ..kernel.ledger import Ledger
from ..task.spec import task_spec
from . import expert
from .agent import AgentSpec, make_gateway, run_agent
from .judge import Limits, file_digest, judge_program, missing, reference_scores, report_of, save_predictions
from .mcp import ToolError
from .sandbox import discover_runtime, ecogym_home, scratch_root
from .tools import REPLACE_MARGIN, EpisodeTools, snapshot, starting_program
from .workspace import LOCAL_STAGE, SCORED_STAGE, export_visible, task_brief

EVENT_KINDS = frozenset({"episode_started", "episode_evaluated", "episode_submitted", "episode_note",
                         "expert_question", "episode_scored", "episode_finished"})
PROMPT = ("请先阅读当前目录下的 TASK.md 和 AGENTS.md，然后独立完成任务：研究数据，编写并改进 submission/ 中的预测程序，"
          "用 ecogym 工具检验和提交。截止前务必至少成功提交一次。")
PREDICTIONS = "scored.npz"
CONTINUE_SHARE = 0.25  # an agent that ends with more than this share of its time left gets a follow-up session
MAX_CONTINUATIONS = 3


def _follow_up(tools: EpisodeTools, *, deadline: float, minutes: float):
    """Prompts for follow-up sessions: agents tend to stop early, so the rest of the time would go unused.

    A follow-up is a fresh DSH session in the same workspace: it sees the files, the submissions (task_info)
    and the previous session's final reply, which this prompt carries. Every harness version gets the same.
    A follow-up that neither evaluates nor submits ends the series: the agent judged itself done, and further
    sessions only repeat its report.
    """

    started_at = [-1]  # local evaluations + submissions when the latest follow-up started

    def ask(n: int, reply: str) -> str | None:
        left = deadline - clock.now()
        work = tools.evaluations + len(tools.submissions)
        if n > MAX_CONTINUATIONS or left < CONTINUE_SHARE * minutes * 60 or work == started_at[0]:
            return None
        started_at[0] = work
        best = tools.best_local()
        state = (f"现在评分用{tools.scored_text()}；已接受的提交里本地最好 {best['local']['score']:+.4f}。"
                 if best else f"还没有被接受的提交，现在评分用{tools.scored_text()}。")
        return "\n".join([
            f"这是同一个回合的第 {n + 1} 段会话。上一段会话已经结束，但这个回合还剩约 {left / 60:.0f} 分钟"
            f"（共 {minutes:g} 分钟）。工作区里的文件和提交都还在，用 task_info 查看。{state}",
            "请继续改进：先读 TASK.md 和 submission/ 回顾已经做过的事，再找在拟合期内部多个滚动窗口上都稳定的改进；"
            f"只有确实更好时才 submit（评分用最后一次被接受、且本地比起始程序高至少 {REPLACE_MARGIN} 的提交）。"
            "截止前留出时间收尾，写最终回复。",
            "", "上一段会话的最终回复（节选）：", "", reply.strip()[-2500:] or "（没有）",
        ])

    return ask


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    task: str
    harness: str = "seed"
    minutes: float = 30.0
    tokens: int = 4_000_000  # cost-weighted, see ecogym.episodes.gateway
    model: str | None = None  # pin the model; None uses the harness's choice
    label: str = ""
    keep: bool = False
    limits: Limits = field(default_factory=Limits)
    context: dict[str, Any] = field(default_factory=dict)  # who ran it: evolution run, round, role, candidate
    expert_until: int | None = None  # expert guidance recorded up to this ledger sequence number (None: all)


class Events:
    """Ledger appends from any thread (one connection per append)."""

    def __init__(self, path: Path, run_id: str, kinds: frozenset[str] = EVENT_KINDS) -> None:
        self.path = path
        self.run_id = run_id
        self.kinds = kinds
        self._lock = threading.Lock()

    def __call__(self, kind: str, body: dict[str, Any]) -> None:
        if kind not in self.kinds:
            raise ValueError(f"unknown event {kind}")
        with self._lock, Ledger(self.path) as ledger:
            ledger.append(self.run_id, kind, body)


def _final_program(tools: EpisodeTools, root: Path) -> dict[str, Any] | None:
    """The latest accepted submission that replaces the starting program, else the starting program; with no
    starting program (a harness without a template), a snapshot of the current submission/ directory."""

    final = tools.final()
    if final is not None:
        return final
    if tools.start is not None:
        return tools.start
    fallback = root / "submissions" / "final-fallback" / "program"
    fallback.mkdir(parents=True, exist_ok=True)
    try:
        return {"n": None, "digest": snapshot(root / "work" / "submission", fallback), "program": str(fallback),
                "auto": True}
    except ToolError:
        return None


def run_episode(config: EpisodeConfig, ledger_path: Path, *, bench: Bench | None = None,
                cancel: threading.Event | None = None) -> dict[str, Any]:
    bench = bench or Bench.open(task_spec(config.task))
    runtime = discover_runtime()
    if runtime.dsh is None or runtime.node_root is None:
        raise FileNotFoundError("dsh not found; install DeepSeek Harness or set ECOGYM_DSH_BIN")
    harness = resolve(config.harness)
    agent = AgentSpec.of(harness.agent, config.model)
    episode_id = f"ep:{uuid.uuid4().hex[:12]}"
    name = episode_id.replace(":", "-")
    root = scratch_root() / name
    trajectory = ecogym_home() / "trajectories" / name
    started = time.time()
    deadline = clock.now() + config.minutes * 60  # awake time: a sleeping machine spends none of the budget
    gateway = make_gateway(trajectory, tokens=config.tokens, deadline=deadline,
                           model=agent.model)  # fails (unknown route, missing key) before anything is created
    for path in (root / "work" / "data", trajectory):
        path.mkdir(parents=True, exist_ok=True)
    events = Events(ledger_path, episode_id)
    harness.install(root)
    start = starting_program(bench, root, runtime, config.limits)
    deadline = clock.now() + config.minutes * 60  # the clock starts after the starting program was judged
    gateway.deadline = deadline
    with Ledger(ledger_path) as ledger:
        advice = expert.guidance(ledger, bench.task, until=config.expert_until)
    visible = export_visible(bench, root / "work" / "data", expert=expert.render(advice) if advice else "")
    (root / "work" / "TASK.md").write_text(task_brief(bench, visible["boundaries"], minutes=config.minutes,
                                                      tokens=config.tokens, deadline=deadline, limits=config.limits,
                                                      start_local=start["local"]["score"] if start else None),
                                           encoding="utf-8")
    (root / "work" / "notes").mkdir(exist_ok=True)
    tools = EpisodeTools(bench=bench, root=root, runtime=runtime, limits=config.limits, deadline=deadline,
                         usage=gateway.usage, event=events, start=start)

    logged = []

    def started_event(facts: dict[str, Any]) -> None:
        logged.append(True)
        events("episode_started", {
            "task_id": bench.task.id, "label": config.label, "bench": bench.identity, "ecogym": __version__,
            "harness": {"digest": harness.digest, "name": harness.name, "parent": harness.parent},
            "model": agent.model, "agent": asdict(agent), "minutes": config.minutes, "tokens": config.tokens,
            "limits": asdict(config.limits), "context": config.context, "root": str(root),
            "dsh": facts["dsh"], "sandbox": digest(facts["profile"].replace(str(root), "<episode>")),
            "visible_rows": visible["rows"], "boundaries": visible["boundaries"], "expert_entries": len(advice),
            "expert_until": config.expert_until, "replace_margin": REPLACE_MARGIN,
            "start": None if start is None else {"digest": start["digest"], "local": start["local"]},
        })

    try:
        session = run_agent(root=root, prompt=PROMPT, agent=agent, tools=tools.catalog(), gateway=gateway,
                            deadline=deadline, trajectory=trajectory, runtime=runtime, on_start=started_event,
                            cancel=cancel, follow_up=_follow_up(tools, deadline=deadline, minutes=config.minutes))
        final = _final_program(tools, root)
        if final is not None:
            report, evaluation = judge_program(bench, Path(final["program"]), SCORED_STAGE, runtime,
                                               limits=config.limits, cancel=cancel)
        else:
            evaluation = missing(bench.stage(SCORED_STAGE))
            report = report_of(bench, evaluation, None)
    except BaseException as error:  # the agent or the judge was stopped (or broke): nothing is scored
        if logged:
            events("episode_finished", {"status": "interrupted", "error": f"{type(error).__name__}: {error}"[:2000],
                                        "seconds": round(time.time() - started, 1), "tokens": gateway.tokens,
                                        "metered_tokens": gateway.metered, "llm_calls": gateway.calls,
                                        "submissions": len(tools.submissions),
                                        "trajectory": {"dir": str(trajectory)}})
        if not config.keep:
            subprocess.run(["rm", "-r", "-f", str(root)], check=False)
        raise

    report["references"] = reference_scores(bench, SCORED_STAGE)
    stored = save_predictions(trajectory / PREDICTIONS, bench, evaluation)
    scored = {key: report[key] for key in ("stage", "status", "origins", "score", "interval", "coverage", "cells",
                                           "fit_seconds", "seconds", "errors", "references")}
    scored.update({"n": final.get("n") if final else None, "digest": final["digest"] if final else None,
                   "auto": bool(final.get("auto")) if final else False,
                   "start": bool(final.get("start")) if final else False, "predictions": PREDICTIONS,
                   "predictions_digest": file_digest(stored)})
    events("episode_scored", scored)
    finished = {
        **session.to_dict(), "status": session.status, "local_evaluations": tools.evaluations,
        "submissions": len(tools.submissions), "notes": len(tools.notes), "questions": len(tools.questions),
        "score": scored["score"], "local_check": LOCAL_STAGE,
    }
    events("episode_finished", finished)
    if not config.keep:
        subprocess.run(["rm", "-r", "-f", str(root)], check=False)
    return {"episode": episode_id, **finished, "scored": scored, "predictions": str(trajectory / PREDICTIONS)}


__all__ = ["EVENT_KINDS", "PREDICTIONS", "PROMPT", "EpisodeConfig", "Events", "run_episode"]
