"""The ecogym tools an agent calls during an episode (served over MCP).

``evaluate_local`` and ``submit`` snapshot ``work/submission`` and run it with
the judge's protocol on the local check (calibration_uq). Snapshots are taken
outside the sandbox, so they refuse symlinks, hard links and special files:
a submission can only contain bytes the agent wrote inside its workspace.

The starting program and each submission receive the same causal local check.
The default policy keeps the highest-scoring complete submission that clears
the replacement margin. Three chronological windows give additional diagnostics
from the same causal predictions. Runs may opt into a window consistency gate
or retain the legacy single-window policy.

``close`` (called when the session ends, before its record is sealed) cancels
local checks still running and makes every tool refuse: a submission judged
after the agent was stopped is never recorded, so the program scored is always
the latest accepted submission the session itself saw.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.harness import Bench, Evaluation
from ..kernel import clock
from ..kernel.canonical import digest as canonical_digest
from ..kernel.fs import CopyError, copy_regular_tree
from .classical import classical_reference_tool
from .judge import JUDGE_PROTOCOL, PRE_LOCAL_STAGE, Limits, calibration_pre_stage, judge_program
from .mcp import Tool, ToolError
from .sandbox import Runtime, SandboxUnavailable, confine_program, scratch_root, thread_limits
from .workspace import LOCAL_STAGE

MAX_FILES = 300
MAX_BYTES = 20_000_000
SKIP = {"__pycache__", ".git", "tmp", ".ipynb_checkpoints"}
NOTE_KINDS = ("finding", "lesson", "dead_end")
REPLACE_MARGIN = 0.02  # local-score gain a submission needs over the starting program to replace it
SINGLE_WINDOW_SELECTION = "single_window/1"
BEST_LOCAL_SELECTION = "best_local/1"
ROLLING_SELECTION = "rolling_blocks/1"
WALK_FORWARD_SELECTION = "walk_forward/1"
DEFAULT_LOCAL_SELECTION = WALK_FORWARD_SELECTION
LOCAL_SELECTION_POLICIES = (
    SINGLE_WINDOW_SELECTION,
    BEST_LOCAL_SELECTION,
    ROLLING_SELECTION,
    WALK_FORWARD_SELECTION,
)
WINDOW_COUNT = 3
MIN_BLOCKS = 2 * WINDOW_COUNT
WINDOW_LOSS_CAP = 0.02
WALK_FORWARD_MEAN_MARGIN = 0.015
WALK_FORWARD_PRE_FLOOR = 0.0
WALK_FORWARD_HORIZON_CAP = 0.05
SCORE_TOLERANCE = 1e-12


def selection_rule_text(policy: str) -> str:
    if policy == SINGLE_WINDOW_SELECTION:
        return f"评分用最后一次被接受、且本地总分比起始程序高至少 {REPLACE_MARGIN:g} 的提交，否则用起始程序。"
    if policy == WALK_FORWARD_SELECTION:
        return (
            f"候选须完成本地检验并覆盖所有计分格；相对有效起始程序，近期本地窗口总分须提高至少 {REPLACE_MARGIN:g}，"
            f"且在拟合期内部留出的更早因果前滚折上不退化（≥ {WALK_FORWARD_PRE_FLOOR:+.2g}）、"
            f"两折平均增益至少 {WALK_FORWARD_MEAN_MARGIN:g}、"
            f"近期折及双折平均任一时距退化均不超过 {WALK_FORWARD_HORIZON_CAP:g}。"
            "满足条件的提交中双折均分最高者用于评分，没有通过则继续用起始程序；"
            "拟合期过短不足以切分前滚折时退回近期窗口最高分规则；起始程序无效时可用通过完整性检验的提交恢复。"
        )
    if policy not in (BEST_LOCAL_SELECTION, ROLLING_SELECTION):
        raise ValueError(f"unknown local selection policy {policy!r}")
    windows = (f"将至少 {MIN_BLOCKS} 个时间块顺序分成 {WINDOW_COUNT} 个窗口，至少两个窗口进步，"
               f"每个窗口的退化不得超过 {WINDOW_LOSS_CAP:g}。" if policy == ROLLING_SELECTION
               else f"本地结果附带 {WINDOW_COUNT} 个窗口的稳定性诊断。")
    return (f"候选须完成本地检验并覆盖所有计分格；相对有效起始程序，本地总分须提高至少 {REPLACE_MARGIN:g}。"
            f"{windows}满足条件的提交中本地总分最高者用于评分，"
            "没有通过的提交则继续用起始程序。起始程序无效时，可用通过完整性检验的提交恢复。")


def _healthy(report: dict[str, Any]) -> bool:
    return (report["status"] == "ok" and report["origins"] > 0 and report["coverage"] == 1
            and math.isfinite(report["score"]))


def rolling_selection(candidate: Evaluation, start: Evaluation) -> dict[str, Any]:
    """Paired local-stage selection, with whole calendar blocks shared by the two programs."""

    stage = candidate.stage
    if stage is not start.stage or stage.name != LOCAL_STAGE:
        raise ValueError("rolling selection requires the same local stage for both programs")
    blocks = np.unique(stage.blocks)
    delta = candidate.score() - start.score()
    result = {"policy": ROLLING_SELECTION, "passed": False, "eligible_blocks": int(blocks.size),
              "delta": delta, "windows": []}
    if blocks.size < MIN_BLOCKS:
        return {**result, "reason": "insufficient_blocks"}
    windows = result["windows"]
    for window in np.array_split(blocks, WINDOW_COUNT):
        mask = (stage.blocks >= window[0]) & (stage.blocks <= window[-1])
        first, last = stage.time[mask][[0, -1]]
        score, baseline = candidate.score(mask), start.score(mask)
        windows.append({"first_origin": int(first), "last_origin": int(last), "blocks": len(window),
                        "origins": int(mask.sum()), "candidate": score, "start": baseline,
                        "delta": score - baseline})
    gains = [window["delta"] for window in windows]
    if not math.isfinite(delta) or not all(math.isfinite(gain) for gain in gains):
        return {**result, "reason": "nonfinite_score"}
    if delta + SCORE_TOLERANCE < REPLACE_MARGIN:
        return {**result, "reason": "overall_gain_below_margin"}
    if min(gains) < -WINDOW_LOSS_CAP - SCORE_TOLERANCE:
        return {**result, "reason": "window_loss_above_cap"}
    if sum(gain > SCORE_TOLERANCE for gain in gains) <= WINDOW_COUNT // 2:
        return {**result, "reason": "insufficient_window_wins"}
    return {**result, "passed": True, "reason": "passed"}


def horizon_objectives(evaluation: Evaluation) -> dict[int, float]:
    """Weighted mean cell objective per forecast horizon on ``evaluation``."""

    return evaluation.horizon_scores()


def walk_forward_selection(
    candidate: Evaluation,
    start: Evaluation,
    *,
    candidate_pre: Evaluation | None = None,
    start_pre: Evaluation | None = None,
    pre_available: bool = True,
    pre_invalid: bool = False,
) -> dict[str, Any]:
    """Two-fold causal walk-forward selection across ``calibration_pre`` and ``calibration_uq``."""

    diagnostic = rolling_selection(candidate, start)
    uq_delta = diagnostic["delta"]
    cand_h = horizon_objectives(candidate)
    start_h = horizon_objectives(start)
    uq_h_deltas = {str(h): round(cand_h[h] - start_h[h], 4) for h in start_h if h in cand_h}
    min_uq_h = min((cand_h[h] - start_h[h] for h in start_h if h in cand_h), default=0.0)
    base = {
        **diagnostic,
        "policy": WALK_FORWARD_SELECTION,
        "passed": False,
        "window_passed": diagnostic["passed"],
        "window_reason": diagnostic["reason"],
        "uq_horizons": uq_h_deltas,
        "pre_available": pre_available,
    }
    if not math.isfinite(uq_delta) or not math.isfinite(min_uq_h):
        return {**base, "reason": "nonfinite_score"}
    if uq_delta + SCORE_TOLERANCE < REPLACE_MARGIN:
        return {**base, "reason": "overall_gain_below_margin"}
    if min_uq_h < -WALK_FORWARD_HORIZON_CAP - SCORE_TOLERANCE:
        return {**base, "reason": "horizon_loss_above_cap"}
    if not pre_available:
        return {**base, "passed": True, "reason": "passed_single_fold", "walk_forward_score": candidate.score()}
    if pre_invalid or candidate_pre is None or start_pre is None:
        return {**base, "reason": "pre_fold_invalid"}
    pre_score = candidate_pre.score()
    pre_base = start_pre.score()
    pre_delta = pre_score - pre_base
    cand_pre_h = horizon_objectives(candidate_pre)
    start_pre_h = horizon_objectives(start_pre)
    pre_h_deltas = {str(h): round(cand_pre_h[h] - start_pre_h[h], 4) for h in start_pre_h if h in cand_pre_h}
    min_pre_h = min((cand_pre_h[h] - start_pre_h[h] for h in start_pre_h if h in cand_pre_h), default=0.0)
    shared_h = [h for h in start_h if h in cand_h and h in start_pre_h and h in cand_pre_h]
    wf_h_raw = {h: 0.5 * ((cand_h[h] - start_h[h]) + (cand_pre_h[h] - start_pre_h[h])) for h in shared_h}
    wf_h_deltas = {str(h): round(val, 4) for h, val in wf_h_raw.items()}
    min_wf_h = min(wf_h_raw.values(), default=0.0)
    wf_delta = 0.5 * (uq_delta + pre_delta)
    wf_score = 0.5 * (candidate.score() + pre_score)
    enriched = {
        **base,
        "pre_score": round(pre_score, 4),
        "pre_start": round(pre_base, 4),
        "pre_delta": pre_delta,
        "pre_horizons": pre_h_deltas,
        "walk_forward_horizons": wf_h_deltas,
        "walk_forward_delta": wf_delta,
        "walk_forward_score": wf_score,
    }
    if not math.isfinite(pre_delta) or not math.isfinite(min_pre_h) or not math.isfinite(min_wf_h):
        return {**enriched, "reason": "nonfinite_score"}
    if pre_delta + SCORE_TOLERANCE < WALK_FORWARD_PRE_FLOOR:
        return {**enriched, "reason": "pre_fold_regression"}
    if min_wf_h < -WALK_FORWARD_HORIZON_CAP - SCORE_TOLERANCE:
        return {**enriched, "reason": "horizon_loss_above_cap"}
    if wf_delta + SCORE_TOLERANCE < WALK_FORWARD_MEAN_MARGIN:
        return {**enriched, "reason": "walk_forward_gain_below_margin"}
    return {**enriched, "passed": True, "reason": "passed"}


def snapshot(source: Path, target: Path) -> str:
    """Copy a submission directory without following links; return its content digest."""

    if source.is_symlink() or not source.is_dir():
        raise ToolError(f"{source.name} must be a directory")
    if not (source / "predictor.py").is_file():
        raise ToolError("submission/predictor.py is missing")
    try:
        return copy_regular_tree(
            source,
            target,
            max_files=MAX_FILES,
            max_bytes=MAX_BYTES,
            skip_dirs=SKIP,
            skip_file=lambda name: (
                name.startswith(".") or name.startswith("_ecogym") or name.endswith(".pyc") or name == "stderr.log"
            ),
            hash_header=b"ecogym.submission/1\n",
        )
    except CopyError as error:
        raise ToolError(str(error)) from None


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


class StartCache:
    """Thread-safe deterministic cache for starting-template local checks shared across replicates and rounds.

    Keyed by ``(bench.digest, program_digest, limits_digest, JUDGE_PROTOCOL)``. Only healthy ``LOCAL_STAGE``
    judgements (and their optional ``PRE_LOCAL_STAGE`` walk-forward evaluations) are cached; unhealthy runs
    are never cached so retries can recover.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._key_locks: dict[tuple[str, str, str, str], threading.Lock] = {}
        self._entries: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(bench: Bench, program_digest: str, limits: Limits) -> tuple[str, str, str, str]:
        return (bench.digest, program_digest, canonical_digest(asdict(limits)), JUDGE_PROTOCOL)

    def _lock_for(self, key: tuple[str, str, str, str]) -> threading.Lock:
        with self._guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = self._key_locks[key] = threading.Lock()
            return lock

    def get_or_judge(
        self,
        bench: Bench,
        program_digest: str,
        limits: Limits,
        compute: Callable[[], tuple[dict[str, Any], Evaluation]],
    ) -> tuple[dict[str, Any], Evaluation, dict[str, Any]]:
        key = self.key(bench, program_digest, limits)
        with self._lock_for(key):
            cached = self._entries.get(key)
            if cached is not None:
                with self._guard:
                    self.hits += 1
                return cached["report"], cached["evaluation"], cached
            report, evaluation = compute()
            entry: dict[str, Any] = {"report": report, "evaluation": evaluation, "local": _summary(report)}
            with self._guard:
                self.misses += 1
                if _healthy(report):
                    self._entries[key] = entry
            return report, evaluation, entry

    def get_pre(self, bench: Bench, program_digest: str, limits: Limits) -> tuple[bool, Evaluation | None]:
        key = self.key(bench, program_digest, limits)
        with self._guard:
            cached = self._entries.get(key)
            if cached is not None and "pre_evaluation" in cached:
                return True, cached["pre_evaluation"]
        return False, None

    def set_pre(self, bench: Bench, program_digest: str, limits: Limits, evaluation: Evaluation | None) -> None:
        if evaluation is None:
            return
        key = self.key(bench, program_digest, limits)
        with self._guard:
            cached = self._entries.get(key)
            if cached is not None and "pre_evaluation" not in cached:
                cached["pre_evaluation"] = evaluation


def starting_program(bench: Bench, root: Path, runtime: Runtime, limits: Limits, *,
                     programs: Path | None = None, cache: StartCache | None = None) -> dict[str, Any] | None:
    """Snapshot and judge ``submission/`` as the episode begins (``submissions/start``); None without a program."""

    program = (programs or root) / "submissions" / "start" / "program"
    program.mkdir(parents=True)
    try:
        digest = snapshot(root / "work" / "submission", program)
    except ToolError:
        return None
    if cache is not None:
        report, evaluation, entry = cache.get_or_judge(
            bench, digest, limits, lambda: judge_program(bench, program, LOCAL_STAGE, runtime, limits=limits)
        )
        local = copy.deepcopy(entry["local"])
        extra = {"pre_evaluation": entry["pre_evaluation"]} if "pre_evaluation" in entry else {}
    else:
        report, evaluation = judge_program(bench, program, LOCAL_STAGE, runtime, limits=limits)
        local = _summary(report)
        extra = {}
    return {"n": None, "digest": digest, "program": str(program), "start": True, "local": local,
            "report": report, "evaluation": evaluation, "healthy": _healthy(report), **extra}


HOOK_TIMEOUT = 5.0
HOOK_MAX_WARNINGS = 8
HOOK_WARNING_CHARS = 300
RESCUE_MAX_SECONDS = 300.0
RESCUE_SHARE = 0.25
_HOOK_RUNNER = (
    "import importlib.util, json, sys\n"
    "payload = json.load(sys.stdin)\n"
    "spec = importlib.util.spec_from_file_location('audit_submission', sys.argv[1])\n"
    "mod = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(mod)\n"
    "fn = getattr(mod, 'audit', None)\n"
    "if not callable(fn):\n"
    "    raise RuntimeError('hooks/audit_submission.py must define audit(summary, selection, files)')\n"
    "out = fn(payload['summary'], payload['selection'], payload['files'])\n"
    "if out is None:\n"
    "    out = []\n"
    "if not isinstance(out, (list, tuple)):\n"
    "    raise TypeError('audit() must return a list of warning strings')\n"
    "print(json.dumps([str(x)[:300] for x in out[:8] if str(x).strip()], ensure_ascii=False))\n"
)


def builtin_submission_audit(
    summary: dict[str, Any],
    selection: dict[str, Any],
    *,
    start_digest: str | None = None,
    digest: str | None = None,
) -> list[str]:
    """Deterministic local-check warnings surfaced directly to the inner agent on evaluate_local/submit."""

    warnings: list[str] = []
    if start_digest and digest and digest == start_digest:
        warnings.append("当前 submission/ 与起始模板内容完全相同，未产生有效代码改动。")
    if summary.get("status") == "ok":
        cells = summary.get("cells") or {}
        if cells and all(abs(float(item.get("skill") or 0.0)) < 1e-4 for item in cells.values()):
            warnings.append("全部计分格技能接近 0（与朴素基线无差），请检查残差方向或模型预测是否生效。")
    uq_h = selection.get("uq_horizons") or {}
    bad_h = [(k, float(v)) for k, v in uq_h.items() if isinstance(v, (int, float)) and float(v) < -0.03]
    if bad_h:
        desc = "、".join(f"h={k} ({v:+.4f})" for k, v in sorted(bad_h, key=lambda kv: kv[1]))
        warnings.append(f"存在显著退化时距：{desc}，建议按短/长时距分开建模或增强长时距收缩正则。")
    delta = selection.get("delta")
    pre_delta = selection.get("pre_delta")
    if (isinstance(delta, (int, float)) and isinstance(pre_delta, (int, float))
            and float(delta) > REPLACE_MARGIN and float(pre_delta) < 0.0):
        warnings.append(
            f"近期检验窗增益 {delta:+.4f}，但更早拟合折前滚变化 {pre_delta:+.4f}，疑似过拟合近期窗口。"
        )
    return warnings


def _read_program_texts(program: Path, *, max_files: int = 12, max_chars: int = 16_000) -> dict[str, str]:
    files: dict[str, str] = {}
    if not program.is_dir():
        return files
    for path in sorted(program.rglob("*.py"))[:max_files]:
        if path.is_symlink() or not path.is_file():
            continue
        try:
            files[path.relative_to(program).as_posix()] = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        except OSError:
            continue
    return files


def run_audit_hook(
    hook_path: Path,
    runtime: Runtime | None,
    *,
    summary: dict[str, Any],
    selection: dict[str, Any],
    program: Path | None = None,
    timeout: float = HOOK_TIMEOUT,
) -> tuple[list[str], str | None]:
    """Run an optional Harness-provided ``hooks/audit_submission.py`` inside a networkless sandbox."""

    if hook_path.is_symlink() or not hook_path.is_file():
        return [], None
    if runtime is None or not getattr(runtime, "python", None):
        return [], "audit hook skipped: no sandbox runtime"
    try:
        descriptor = os.open(hook_path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as handle:
            source = handle.read(64_000)
    except OSError as error:
        return [], f"cannot read audit hook: {error}"
    payload = {
        "summary": summary,
        "selection": selection,
        "files": _read_program_texts(program) if program is not None else {},
    }
    with tempfile.TemporaryDirectory(prefix="ecogym-hook-", dir=scratch_root()) as tmp:
        stage_dir = Path(tmp)
        target = stage_dir / "audit_submission.py"
        target.write_bytes(source)
        argv = [str(runtime.python), "-I", "-c", _HOOK_RUNNER, str(target)]
        try:
            cmd = confine_program(stage_dir, runtime, stage_dir / "hook.sb", argv)
        except SandboxUnavailable as error:
            return [], str(error)
        env = {**os.environ, **thread_limits(1), "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            done = subprocess.run(
                cmd,
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=stage_dir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return [], f"hooks/audit_submission.py timed out after {timeout:g}s"
        except OSError as error:
            return [], f"hooks/audit_submission.py failed to launch: {error}"
    if done.returncode != 0:
        err = (done.stderr or done.stdout or f"exit {done.returncode}").strip()[-400:]
        return [], err
    try:
        parsed = json.loads(done.stdout.strip() or "[]")
    except ValueError as error:
        return [], f"hooks/audit_submission.py returned invalid JSON: {error}"
    if not isinstance(parsed, list):
        return [], "hooks/audit_submission.py output must be a JSON list"
    cleaned = [str(item).strip()[:HOOK_WARNING_CHARS] for item in parsed[:HOOK_MAX_WARNINGS] if str(item).strip()]
    return cleaned, None


class EpisodeTools:
    def __init__(self, *, bench: Bench, root: Path, runtime: Runtime, limits: Limits, deadline: float,
                 usage: Callable[[], dict[str, Any]], event: Callable[[str, dict[str, Any]], None],
                 start: dict[str, Any] | None = None, programs: Path | None = None,
                 selection_policy: str = DEFAULT_LOCAL_SELECTION,
                 reuse_local: bool | None = None, start_cache: StartCache | None = None,
                 budget_minutes: float | None = None) -> None:
        """``start`` is the judged starting program (``starting_program``), if any."""

        if selection_policy not in LOCAL_SELECTION_POLICIES:
            raise ValueError(f"unknown local selection policy {selection_policy!r}")

        self.bench = bench
        self.root = root
        self.programs = programs or root
        self.runtime = runtime
        self.limits = limits
        self.deadline = deadline
        self.budget_minutes = budget_minutes
        self.usage = usage
        self.event = event
        self.start = start
        self.start_cache = start_cache
        self.selection_policy = selection_policy
        self._reuse_local = (runtime is not None) if reuse_local is None else reuse_local
        self._local_cache: dict[tuple[str, str], tuple[dict[str, Any], Evaluation]] = {}
        if (self._reuse_local and start and start.get("healthy")
                and "digest" in start and "report" in start and "evaluation" in start):
            self._local_cache[(start["digest"], LOCAL_STAGE)] = (start["report"], start["evaluation"])
        self.submissions: list[dict[str, Any]] = []
        self.evaluations = 0
        self.notes: list[dict[str, Any]] = []
        self.questions: list[dict[str, Any]] = []
        self.rescue_triggered = False
        self.hook_fired = False
        self._lock = threading.Lock()
        self._pre_lock = threading.Lock()
        self._slots = threading.Semaphore(2)
        self._counter = 0
        self.closing = threading.Event()  # set when the session ends: local checks stop, nothing more is recorded

    def close(self) -> None:
        """The session has ended: cancel running local checks; every tool refuses from now on."""

        with self._lock:
            self.closing.set()

    def _open(self) -> None:
        if self.closing.is_set():
            raise ToolError("the session has ended")

    def _record(self, items: list[dict[str, Any]], record: dict[str, Any]) -> None:
        """Append a record unless the session ended meanwhile (checked under the lock ``close`` races with)."""

        with self._lock:
            self._open()
            items.append(record)

    def catalog(self) -> list[Tool]:
        return [
            Tool("task_info", "剩余时间与 token、已提交的版本、最好的本地成绩。",
                 {"type": "object", "properties": {}}, self.task_info),
            classical_reference_tool((self.bench,), check_open=self._open),
            Tool("evaluate_local",
                 "按裁判协议在本地检验期上运行当前的 submission/ 目录（快照），返回总分、95% 区间、逐格技能、覆盖率、"
                 "替换条件、用时和错误信息。不会提交。",
                 {"type": "object", "properties": {}}, self.evaluate_local),
            Tool("submit",
                 "提交当前 submission/ 目录的快照：先在本地检验期上运行，成功则被接受。"
                 + selection_rule_text(self.selection_policy),
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

    def _judge_stage(self, program: Path, digest: str | None, stage: str) -> tuple[dict[str, Any], Evaluation, bool]:
        if self._reuse_local and digest:
            with self._lock:
                cached = self._local_cache.get((digest, stage))
            if cached is not None:
                return cached[0], cached[1], True
        with self._slots:
            self._open()
            if self._reuse_local and digest:
                with self._lock:
                    cached = self._local_cache.get((digest, stage))
                if cached is not None:
                    return cached[0], cached[1], True
            report, evaluation = judge_program(self.bench, program, stage, self.runtime, limits=self.limits,
                                               cancel=self.closing)
            if self._reuse_local and digest and _healthy(report):
                with self._lock:
                    self._local_cache[(digest, stage)] = (report, evaluation)
        return report, evaluation, False

    def _start_pre_evaluation(self) -> Evaluation | None:
        if self.start is None:
            return None
        with self._pre_lock:
            if "pre_evaluation" in self.start:
                return self.start["pre_evaluation"]
            start_digest = self.start.get("digest")
            if self.start_cache is not None and start_digest:
                found, cached_pre = self.start_cache.get_pre(self.bench, start_digest, self.limits)
                if found:
                    self.start["pre_evaluation"] = cached_pre
                    return cached_pre
            start_program = self.start.get("program")
            if not start_program:
                return None
            report, evaluation, _ = self._judge_stage(Path(start_program), start_digest, PRE_LOCAL_STAGE)
            result = evaluation if _healthy(report) else None
            self.start["pre_evaluation"] = result
            if self.start_cache is not None and start_digest and result is not None:
                self.start_cache.set_pre(self.bench, start_digest, self.limits, result)
            return result

    def _run_local(self, kind: str) -> tuple[int, str, Path, dict[str, Any], bool, dict[str, Any]]:
        self._open()
        if clock.now() > self.deadline:
            raise ToolError("the episode deadline has passed")
        number = self._next()
        program = self.programs / kind / f"{number:03d}" / "program"
        program.mkdir(parents=True)
        digest = snapshot(self.root / "work" / "submission", program)
        report, evaluation, reused = self._judge_stage(program, digest, LOCAL_STAGE)
        summary = _summary(report)
        accepted, selection = self._selection(report, evaluation, summary, program=program, digest=digest)
        selection["candidate_score"] = report["score"]
        if reused:
            selection["reused_local"] = True
        return number, digest, program, summary, accepted, selection

    def _selection(self, report: dict[str, Any], evaluation: Evaluation, summary: dict[str, Any], *,
                   program: Path | None = None, digest: str | None = None) -> tuple[bool, dict[str, Any]]:
        policy = self.selection_policy
        if policy == SINGLE_WINDOW_SELECTION:
            accepted = summary["status"] == "ok" and summary["coverage"] > 0
            gain = None if self.start is None else summary["score"] - self.start["local"]["score"]
            passed = accepted and (self.start is None or
                                   summary["score"] >= self.start["local"]["score"] + REPLACE_MARGIN)
            return accepted, {"policy": policy, "passed": passed, "delta": gain,
                              "reason": "passed" if passed else "below_margin" if accepted else "invalid_submission"}
        accepted = _healthy(report)
        if not accepted:
            return False, {"policy": policy, "passed": False, "reason": "invalid_submission"}
        if self.start is None or self.start.get("healthy") is False:
            reason = "no_start" if self.start is None else "recover_invalid_start"
            return True, {"policy": policy, "passed": True, "reason": reason}
        if "evaluation" not in self.start:
            return True, {"policy": policy, "passed": False, "reason": "missing_start_predictions"}
        if policy == WALK_FORWARD_SELECTION:
            pre_stage = calibration_pre_stage(self.bench)
            if pre_stage is None:
                return True, walk_forward_selection(evaluation, self.start["evaluation"], pre_available=False)
            prelim = walk_forward_selection(evaluation, self.start["evaluation"], pre_available=False)
            if not prelim["passed"]:
                return True, {**prelim, "passed": False, "pre_available": True}
            start_pre = self._start_pre_evaluation()
            if start_pre is None or program is None:
                return True, walk_forward_selection(evaluation, self.start["evaluation"], pre_available=False)
            pre_report, cand_pre, _ = self._judge_stage(program, digest, PRE_LOCAL_STAGE)
            return True, walk_forward_selection(
                evaluation,
                self.start["evaluation"],
                candidate_pre=cand_pre if _healthy(pre_report) else None,
                start_pre=start_pre,
                pre_available=True,
                pre_invalid=not _healthy(pre_report),
            )
        diagnostic = rolling_selection(evaluation, self.start["evaluation"])
        if policy == ROLLING_SELECTION:
            return True, diagnostic
        gain = diagnostic["delta"]
        passed = gain >= REPLACE_MARGIN
        return True, {**diagnostic, "policy": policy, "passed": passed,
                      "reason": "passed" if passed else "overall_gain_below_margin",
                      "window_passed": diagnostic["passed"], "window_reason": diagnostic["reason"]}

    def best_local(self) -> dict[str, Any] | None:
        accepted = [item for item in self.submissions if item.get("accepted")]
        return max(accepted, key=lambda item: item["local"]["score"], default=None)

    def replaces_start(self, item: dict[str, Any]) -> bool:
        """Whether the submission passed the replacement policy at the time of its local check."""

        return bool(item["local_selection"]["passed"])

    def scored_text(self) -> str:
        """Which program would be scored now, in words for the agent."""

        final = self.final()
        if final is not None:
            score = final["local"]["score"]
            gain = "" if self.start is None else f"，相对起始程序 {score - self.start['local']['score']:+.4f}"
            wf = final.get("local_selection", {}).get("walk_forward_delta")
            wf_text = f"，双折均增 {wf:+.4f}" if isinstance(wf, (int, float)) and math.isfinite(wf) else ""
            return f"第 {final['n']} 次提交（本地 {score:+.4f}{gain}{wf_text}）"
        if self.start is not None:
            return f"起始程序（本地 {self.start['local']['score']:+.4f}）"
        return "还没有被接受的提交"

    def _pacing_notice(self) -> str | None:
        """Budget-rescue nudge (Self-HPO pattern) when time is running low with no qualifying submission."""

        if self.final() is not None:
            return None
        left = self.deadline - clock.now()
        window = min(RESCUE_MAX_SECONDS, RESCUE_SHARE * self.budget_minutes * 60.0) if self.budget_minutes else 180.0
        if left <= 0 or left > window:
            return None
        self.rescue_triggered = True
        custom = self.root / "harness" / "hooks" / "rescue.md"
        if custom.is_file() and not custom.is_symlink():
            try:
                text = custom.read_text(encoding="utf-8", errors="replace").strip()[:1500]
                if text:
                    return f"[预算告急救援 · 剩 {left / 60:.1f} 分钟]\n{text}"
            except OSError:
                pass
        return (
            f"[预算告急救援 · 剩 {left / 60:.1f} 分钟] 尚无通过替换条件的合格提交！"
            "请立即停止开放式探索，保留当前本地表现最稳的改动，调用 evaluate_local 确认覆盖率与增益后立刻 submit 锁定。"
        )

    def _audit_submission(self, program: Path, digest: str, summary: dict[str, Any],
                          selection: dict[str, Any]) -> tuple[list[str], str | None]:
        start_digest = self.start.get("digest") if self.start else None
        warnings = builtin_submission_audit(summary, selection, start_digest=start_digest, digest=digest)
        hook = self.root / "harness" / "hooks" / "audit_submission.py"
        hook_error: str | None = None
        if hook.is_file() and not hook.is_symlink():
            extra, error = run_audit_hook(hook, self.runtime, summary=summary, selection=selection, program=program)
            if extra:
                self.hook_fired = True
                warnings.extend(extra)
            elif error and not error.startswith("audit hook skipped:"):
                hook_error = error
        return warnings[:HOOK_MAX_WARNINGS], hook_error

    def task_info(self) -> dict[str, Any]:
        best, final = self.best_local(), self.final()
        rescue = self._pacing_notice()
        return {
            "minutes_left": round(max(0.0, self.deadline - clock.now()) / 60, 1),
            **self.usage(),
            "local_evaluations": self.evaluations,
            "start_local_score": None if self.start is None else self.start["local"]["score"],
            "submissions": [{"n": item["n"], "digest": item["digest"][:12], "local_score": item["local"]["score"],
                             "accepted": item["accepted"], "replaces_start": self.replaces_start(item),
                             "local_selection": item["local_selection"]}
                            for item in self.submissions],
            "latest_accepted": next((item["n"] for item in reversed(self.submissions) if item["accepted"]), None),
            "scored_now": final["n"] if final is not None else "start" if self.start is not None else None,
            "best_local": None if best is None else {"n": best["n"], "score": best["local"]["score"]},
            **({"budget_rescue": rescue} if rescue else {}),
        }

    def evaluate_local(self) -> dict[str, Any]:
        number, digest, program, summary, _, selection = self._run_local("evals")
        with self._lock:
            self._open()
            self.evaluations += 1
        audit_warnings, hook_error = self._audit_submission(program, digest, summary, selection)
        rescue = self._pacing_notice()
        self.event("episode_evaluated", {"n": number, "digest": digest, "local": summary,
                                         "local_selection": selection,
                                         **({"audit_warnings": audit_warnings} if audit_warnings else {}),
                                         **({"hook_error": hook_error} if hook_error else {})})
        return {"digest": digest[:12], "local_selection": selection, **summary,
                **({"audit_warnings": audit_warnings} if audit_warnings else {}),
                **({"budget_rescue": rescue} if rescue else {})}

    def submit(self, note: str = "") -> dict[str, Any]:
        number, digest, program, summary, accepted, selection = self._run_local("submissions")
        record = {"n": number, "digest": digest, "program": str(program), "accepted": accepted,
                  "note": str(note)[:4000], "at": time.time(), "local": summary, "local_selection": selection}
        self._record(self.submissions, record)
        replaces = self.replaces_start(record)
        audit_warnings, hook_error = self._audit_submission(program, digest, summary, selection)
        rescue = self._pacing_notice()
        self.event("episode_submitted", {**{key: record[key] for key in ("n", "digest", "accepted", "note", "local",
                                                                        "local_selection")},
                                         "replaces_start": replaces,
                                         **({"audit_warnings": audit_warnings} if audit_warnings else {}),
                                         **({"hook_error": hook_error} if hook_error else {})})
        if not accepted:
            verdict = f"未接受。请检查本地检验状态、成绩和预报覆盖率。现在评分用{self.scored_text()}。"
        elif replaces and self.final() is record:
            verdict = f"已接受，现在评分用这次提交：{self.scored_text()}"
        elif replaces:
            verdict = f"已接受，满足替换条件，但已有本地成绩更好的提交。现在评分用{self.scored_text()}"
        else:
            gain = selection.get("delta")
            reason = selection.get("reason")
            small = (reason in ("below_margin", "overall_gain_below_margin")
                     and isinstance(gain, (int, float)) and math.isfinite(gain))
            caution = "本地检验期比隐藏期短得多，这么小的提升常常不能在隐藏期上成立。" if small and gain > 0 else ""
            detail = ""
            if small:
                detail = f"：本地比起始程序 {gain:+.4f}，替换至少要高 {REPLACE_MARGIN:g}"
            elif reason == "pre_fold_regression" and isinstance(selection.get("pre_delta"), (int, float)):
                detail = (f"：近期窗口 {gain:+.4f}，但更早拟合折前滚检验 {selection['pre_delta']:+.4f}"
                          "（发生跨时间段退化，疑似过拟合近期窗口）")
            elif reason == "horizon_loss_above_cap":
                detail = f"：单时距退化超过上限 {WALK_FORWARD_HORIZON_CAP:g}"
            elif reason == "walk_forward_gain_below_margin" and isinstance(
                selection.get("walk_forward_delta"), (int, float)
            ):
                detail = (f"：双折平均增益 {selection['walk_forward_delta']:+.4f}，"
                          f"至少需要 {WALK_FORWARD_MEAN_MARGIN:g}")
            verdict = f"已接受，未达到本地替换条件{detail}。现在评分用{self.scored_text()}。{caution}"
        return {"n": number, "digest": digest[:12], "accepted": accepted, "replaces_start": replaces,
                "verdict": verdict, "local_selection": selection, **summary,
                **({"audit_warnings": audit_warnings} if audit_warnings else {}),
                **({"budget_rescue": rescue} if rescue else {})}

    def final(self) -> dict[str, Any] | None:
        """Pick a qualifying submission under the recorded policy."""

        passing = [item for item in self.submissions if self.replaces_start(item)]
        if self.selection_policy == SINGLE_WINDOW_SELECTION:
            return passing[-1] if passing else None
        if self.selection_policy == WALK_FORWARD_SELECTION:
            return max(
                passing,
                key=lambda item: (
                    item["local_selection"].get("walk_forward_score", item["local_selection"]["candidate_score"]),
                    item["local_selection"]["candidate_score"],
                    -item["n"],
                ),
                default=None,
            )
        return max(passing, key=lambda item: (item["local_selection"]["candidate_score"], -item["n"]), default=None)

    def note(self, kind: str, text: str, sources: list[str] | None = None) -> dict[str, Any]:
        if kind not in NOTE_KINDS:
            raise ToolError(f"kind must be one of {NOTE_KINDS}")
        record = {"kind": kind, "text": str(text)[:4000], "sources": [str(item)[:500] for item in (sources or [])][:20],
                  "at": time.time()}
        self._record(self.notes, record)
        self.event("episode_note", record)
        return {"recorded": len(self.notes)}

    def ask_expert(self, question: str, context: str = "") -> dict[str, Any]:
        record = {"question": str(question)[:4000], "context": str(context)[:4000], "at": time.time()}
        self._record(self.questions, record)
        self.event("expert_question", record)
        return {"recorded": True, "message": "已转给专家；不会立即回答，请继续工作。"}


__all__ = ["BEST_LOCAL_SELECTION", "DEFAULT_LOCAL_SELECTION", "LOCAL_SELECTION_POLICIES", "REPLACE_MARGIN",
           "ROLLING_SELECTION", "SINGLE_WINDOW_SELECTION", "WALK_FORWARD_HORIZON_CAP", "WALK_FORWARD_MEAN_MARGIN",
           "WALK_FORWARD_PRE_FLOOR", "WALK_FORWARD_SELECTION", "EpisodeTools", "StartCache",
           "builtin_submission_audit", "horizon_objectives", "rolling_selection", "run_audit_hook",
           "selection_rule_text", "snapshot", "starting_program", "walk_forward_selection"]
