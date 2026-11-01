"""The ecogym tools an agent calls during an episode (served over MCP).

``evaluate_local`` and ``submit`` snapshot ``work/submission`` and run it with
the judge's protocol on the local check (calibration_uq). Snapshots are taken
outside the sandbox, so they refuse symlinks, hard links and special files:
a submission can only contain bytes the agent wrote inside its workspace.

The starting program (the harness's template, as ``submission/`` holds it
when the episode begins) is judged on the local check before the agent
starts. A submission replaces it only if its local score is at least
``REPLACE_MARGIN`` higher; the latest such submission is scored, else the
starting program. Over 74 episodes that began from a modelling template,
agents raised the local score by 0.028 on average and the hidden one by
0.0005, and 36% ended below their template (down to −0.132); keeping the
template unless a submission gains 0.02 locally would have kept 16% below
it (worst −0.051) and cost nothing on episodes that began from the naive
template, where agents gain far more than that.
"""

from __future__ import annotations

import hashlib
import os
import stat
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..evaluation.harness import Bench
from ..kernel import clock
from .judge import Limits, judge
from .mcp import Tool, ToolError
from .sandbox import Runtime
from .workspace import LOCAL_STAGE

MAX_FILES = 300
MAX_BYTES = 20_000_000
SKIP = {"__pycache__", ".git", "tmp", ".ipynb_checkpoints"}
NOTE_KINDS = ("finding", "lesson", "dead_end")
REPLACE_MARGIN = 0.02  # local-score gain a submission needs over the starting program to replace it


def snapshot(source: Path, target: Path) -> str:
    """Copy a submission directory without following links; return its content digest."""

    if source.is_symlink() or not source.is_dir():
        raise ToolError(f"{source.name} must be a directory")
    if not (source / "predictor.py").is_file():
        raise ToolError("submission/predictor.py is missing")
    sha = hashlib.sha256(b"ecogym.submission/1\n")
    count = total = 0
    for directory, names, filenames in os.walk(source, followlinks=False):
        for name in list(names):
            if os.path.islink(os.path.join(directory, name)):
                raise ToolError(f"{name}: symlinks are not allowed in a submission")
        names[:] = sorted(name for name in names if name not in SKIP and not name.startswith("."))
        for name in sorted(filenames):
            path = Path(directory) / name
            if name.startswith(".") or name.startswith("_ecogym") or name.endswith(".pyc") or name == "stderr.log":
                continue
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ToolError(f"{path.relative_to(source)}: only regular files are allowed")
            if info.st_nlink > 1:
                raise ToolError(f"{path.relative_to(source)}: hard links are not allowed")
            count += 1
            total += info.st_size
            if count > MAX_FILES or total > MAX_BYTES:
                raise ToolError(f"submission exceeds {MAX_FILES} files or {MAX_BYTES // 1_000_000} MB")
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                data = stream.read(MAX_BYTES + 1)
            relative = path.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            sha.update(f"{relative.as_posix()}\n{len(data)}\n".encode())
            sha.update(data)
    return sha.hexdigest()


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "score": round(report["score"], 4),
        "interval": [round(value, 4) for value in report["interval"]],
        "coverage": round(report["coverage"], 3),
        "cells": {cell["cell"]: {"skill": round(cell["skill"], 4), "rmse": round(cell["rmse"], 4),
                                 "baseline_rmse": round(cell["baseline_rmse"], 4), "n": cell["n"]}
                  for cell in report["cells"]},
        "fit_seconds": report["fit_seconds"],
        "seconds": report["seconds"],
        "errors": report["errors"][:3],
        "stderr_tail": report["stderr"][-1500:],
    }


def starting_program(bench: Bench, root: Path, runtime: Runtime, limits: Limits) -> dict[str, Any] | None:
    """Snapshot and judge ``submission/`` as the episode begins (``submissions/start``); None without a program."""

    program = root / "submissions" / "start" / "program"
    program.mkdir(parents=True)
    try:
        digest = snapshot(root / "work" / "submission", program)
    except ToolError:
        return None
    report = judge(bench, program, LOCAL_STAGE, runtime, limits=limits, references=False)
    return {"n": None, "digest": digest, "program": str(program), "start": True, "local": _summary(report)}


class EpisodeTools:
    def __init__(self, *, bench: Bench, root: Path, runtime: Runtime, limits: Limits, deadline: float,
                 usage: Callable[[], dict[str, Any]], event: Callable[[str, dict[str, Any]], None],
                 start: dict[str, Any] | None = None) -> None:
        """``start``: the judged starting program (``starting_program``); None scores the latest accepted
        submission whatever its local score."""

        self.bench = bench
        self.root = root
        self.runtime = runtime
        self.limits = limits
        self.deadline = deadline
        self.usage = usage
        self.event = event
        self.start = start
        self.submissions: list[dict[str, Any]] = []
        self.evaluations = 0
        self.notes: list[dict[str, Any]] = []
        self.questions: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(2)
        self._counter = 0

    def catalog(self) -> list[Tool]:
        return [
            Tool("task_info", "剩余时间与 token、已提交的版本、最好的本地成绩。",
                 {"type": "object", "properties": {}}, self.task_info),
            Tool("evaluate_local",
                 "按裁判协议在本地检验期上运行当前的 submission/ 目录（快照），返回总分、95% 区间、逐格技能、覆盖率、"
                 "用时和错误信息。不会提交。",
                 {"type": "object", "properties": {}}, self.evaluate_local),
            Tool("submit",
                 "提交当前 submission/ 目录的快照：先在本地检验期上运行，成功则被接受。评分用最后一次被接受、"
                 f"且本地总分比起始程序高至少 {REPLACE_MARGIN} 的提交，没有时用起始程序（见 TASK.md）。",
                 {"type": "object", "properties": {"note": {"type": "string", "description": "这一版的方法与改动"}}},
                 self.submit),
            Tool("note", "记录发现、经验或无效方向，供后续回合使用。",
                 {"type": "object", "required": ["kind", "text"],
                  "properties": {"kind": {"type": "string", "enum": list(NOTE_KINDS)},
                                 "text": {"type": "string"},
                                 "sources": {"type": "array", "items": {"type": "string"},
                                             "description": "引用的来源（链接或文献）"}}},
                 self.note),
            Tool("ask_expert", "给领域专家留言。不会立即回答，请继续工作；回答会出现在以后回合的 data/expert.md 里。",
                 {"type": "object", "required": ["question"],
                  "properties": {"question": {"type": "string"}, "context": {"type": "string"}}},
                 self.ask_expert),
        ]

    def _next(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def _run_local(self, kind: str) -> tuple[int, str, Path, dict[str, Any]]:
        if clock.now() > self.deadline:
            raise ToolError("the episode deadline has passed")
        number = self._next()
        program = self.root / kind / f"{number:03d}" / "program"
        program.mkdir(parents=True)
        digest = snapshot(self.root / "work" / "submission", program)
        with self._slots:
            report = judge(self.bench, program, LOCAL_STAGE, self.runtime, limits=self.limits, references=False)
        return number, digest, program, _summary(report)

    def best_local(self) -> dict[str, Any] | None:
        accepted = [item for item in self.submissions if item.get("accepted")]
        return max(accepted, key=lambda item: item["local"]["score"], default=None)

    def replaces_start(self, item: dict[str, Any]) -> bool:
        """An accepted submission whose local score beats the starting program's by ``REPLACE_MARGIN``."""

        return bool(item["accepted"]) and (self.start is None or
                                           item["local"]["score"] >= self.start["local"]["score"] + REPLACE_MARGIN)

    def scored_text(self) -> str:
        """Which program would be scored now, in words for the agent."""

        final = self.final()
        if final is not None:
            score = final["local"]["score"]
            gain = "" if self.start is None else f"，比起始程序高 {score - self.start['local']['score']:+.4f}"
            return f"第 {final['n']} 次提交（本地 {score:+.4f}{gain}）"
        if self.start is not None:
            return f"起始程序（本地 {self.start['local']['score']:+.4f}）"
        return "还没有被接受的提交"

    def task_info(self) -> dict[str, Any]:
        best, final = self.best_local(), self.final()
        return {
            "minutes_left": round(max(0.0, self.deadline - clock.now()) / 60, 1),
            **self.usage(),
            "local_evaluations": self.evaluations,
            "start_local_score": None if self.start is None else self.start["local"]["score"],
            "submissions": [{"n": item["n"], "digest": item["digest"][:12], "local_score": item["local"]["score"],
                             "accepted": item["accepted"], "replaces_start": self.replaces_start(item)}
                            for item in self.submissions],
            "latest_accepted": next((item["n"] for item in reversed(self.submissions) if item["accepted"]), None),
            "scored_now": final["n"] if final is not None else "start" if self.start is not None else None,
            "best_local": None if best is None else {"n": best["n"], "score": best["local"]["score"]},
        }

    def evaluate_local(self) -> dict[str, Any]:
        number, digest, _, summary = self._run_local("evals")
        with self._lock:
            self.evaluations += 1
        self.event("episode_evaluated", {"n": number, "digest": digest, "local": summary})
        return {"digest": digest[:12], **summary}

    def submit(self, note: str = "") -> dict[str, Any]:
        number, digest, program, summary = self._run_local("submissions")
        accepted = summary["status"] == "ok" and summary["coverage"] > 0
        record = {"n": number, "digest": digest, "program": str(program), "accepted": accepted,
                  "note": str(note)[:4000], "at": time.time(), "local": summary}
        with self._lock:
            self.submissions.append(record)
        replaces = self.replaces_start(record)
        self.event("episode_submitted", {**{key: record[key] for key in ("n", "digest", "accepted", "note", "local")},
                                         "replaces_start": replaces})
        if not accepted:
            verdict = f"未接受：程序没有正常完成本地检验。现在评分用{self.scored_text()}"
        elif replaces:
            verdict = f"已接受，现在评分用这次提交：{self.scored_text()}"
        else:
            gain = summary["score"] - self.start["local"]["score"]
            verdict = ((f"已接受，但本地只比起始程序高 {gain:+.4f}" if gain > 0 else
                        f"已接受，但本地没有超过起始程序（{gain:+.4f}）")
                       + f"，替换它至少要高 {REPLACE_MARGIN}。现在评分用{self.scored_text()}。"
                       + ("本地检验期比隐藏期短得多，这么小的提升常常不能在隐藏期上成立。" if gain > 0 else ""))
        return {"n": number, "digest": digest[:12], "accepted": accepted, "replaces_start": replaces,
                "verdict": verdict, **summary}

    def final(self) -> dict[str, Any] | None:
        """The latest accepted submission that replaces the starting program, or None (the start is scored)."""

        return next((item for item in reversed(self.submissions) if self.replaces_start(item)), None)

    def note(self, kind: str, text: str, sources: list[str] | None = None) -> dict[str, Any]:
        if kind not in NOTE_KINDS:
            raise ToolError(f"kind must be one of {NOTE_KINDS}")
        record = {"kind": kind, "text": str(text)[:4000], "sources": [str(item)[:500] for item in (sources or [])][:20],
                  "at": time.time()}
        with self._lock:
            self.notes.append(record)
        self.event("episode_note", record)
        return {"recorded": len(self.notes)}

    def ask_expert(self, question: str, context: str = "") -> dict[str, Any]:
        record = {"question": str(question)[:4000], "context": str(context)[:4000], "at": time.time()}
        with self._lock:
            self.questions.append(record)
        self.event("expert_question", record)
        return {"recorded": True, "message": "已转给专家；不会立即回答，请继续工作。"}


__all__ = ["EpisodeTools", "snapshot"]
