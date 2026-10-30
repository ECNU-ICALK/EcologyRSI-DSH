"""One evolver session: a DSH agent reviews the evidence and writes candidate harnesses.

Workspace, inside the agent sandbox:

    work/AGENTS.md            the evolver's fixed instructions (``evolver.md``, host-owned)
    work/EVOLVE.md            this round's brief: tasks, comparison rules, options, limits, budget
    work/incumbent/           the incumbent harness, for reference
    work/candidates/<name>/   one editable copy of the incumbent per candidate
    work/proposals/<name>.md  each candidate's rationale
    work/evidence/            overview, archive, lessons, expert notes, one folder per episode

Tools: ``task_info``, ``check_candidate``, ``note`` and ``ask_expert``. When the
session ends the host checks every candidate again on its own: it copies the
directory without following links, writes ``name``, ``parent`` and ``author``,
loads and digests it, refuses names of training tasks, diffs it against the
incumbent, and runs its template program on a training task's local check if
the template changed. Only candidates that pass are stored and run. Setting
``cancel`` also stops these checks.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from ..episodes import providers
from ..episodes.agent import AgentSpec, Cancelled, make_gateway, run_agent
from ..episodes.judge import Limits, judge_program, reference_scores
from ..episodes.mcp import Tool, ToolError
from ..episodes.runner import Events
from ..episodes.sandbox import Runtime, discover_runtime, ecogym_home, scratch_root
from ..episodes.workspace import LOCAL_STAGE, SCORED_STAGE
from ..evaluation.harness import Bench
from ..evaluation.suite import adjusted_confidence
from ..harness import MAX_BYTES, MAX_FILES, Harness, HarnessError, diff, fork, normalize, store, term_problems
from ..kernel.ledger import Ledger
from ..task.spec import TaskSpec, base_id
from ..task.suite import SuiteSpec
from .ensemble import ensemble_scores
from .evidence import write_evidence

NAMES = "abcdefghij"
LAYOUT = frozenset({"harness.toml", "instructions.md", "skills", "template", "memory"})  # what a harness may hold
NOTE_KINDS = ("lesson", "finding", "dead_end")
PROPOSAL_LIMIT = 20_000
PROMPT = ("请先阅读当前目录下的 EVOLVE.md 和 AGENTS.md，然后评审 evidence/ 中的证据，"
          "在 candidates/ 下完成各个候选 Harness，写好 proposals/，并用 check_candidate 逐个检查通过。")


@dataclass(frozen=True, slots=True)
class EvolverSettings:
    minutes: float = 30.0
    tokens: int = 4_000_000
    model: str = "deepseek-v4-pro"
    reasoning_effort: str = "high"
    candidates: int = 2


def copy_tree(source: Path, target: Path) -> None:
    """Copy a directory the agent wrote, without following links; regular files only."""

    if source.is_symlink() or not source.is_dir():
        raise HarnessError(f"{source.name} must be a directory")
    count = total = 0
    for directory, names, filenames in os.walk(source, followlinks=False):
        for name in names:
            if os.path.islink(os.path.join(directory, name)):
                raise HarnessError(f"{name}: links are not allowed")
        names[:] = sorted(name for name in names if not name.startswith(".") and name != "__pycache__")
        for name in sorted(filenames):
            path = Path(directory) / name
            if name.startswith(".") or name.endswith(".pyc"):
                continue
            relative = path.relative_to(source)
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode):
                raise HarnessError(f"{relative}: only regular files are allowed")
            if info.st_nlink > 1:
                raise HarnessError(f"{relative}: hard links are not allowed")
            count += 1
            total += info.st_size
            if count > MAX_FILES or total > MAX_BYTES:
                raise HarnessError(f"more than {MAX_FILES} files or {MAX_BYTES} bytes")
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                data = stream.read(MAX_BYTES + 1)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)


def _read_text(path: Path, limit: int) -> str:
    if path.is_symlink() or not path.is_file():
        return ""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read(limit).decode("utf-8", "replace")


def forbidden_terms(tasks: Sequence[TaskSpec]) -> list[str]:
    """Names of the training tasks: dataset, card, task and episode ids, and their identifier-like parts.

    A part that mixes letters and digits (``site2018`` in ``domain.site2018.crop.target@1``) names a
    dataset as surely as the full id; plain words and bare numbers stay allowed, they are domain vocabulary.
    """

    terms = set()
    for task in tasks:
        names = {task.dataset, base_id(task.id), task.id, *task.episodes}
        terms |= names
        terms |= {part for name in names for part in re.split(r"[._@:/-]", name)
                  if re.search(r"[A-Za-z]", part) and re.search(r"\d", part)}
    return sorted(terms)


class EvolverTools:
    def __init__(self, *, root: Path, incumbent: Harness, names: Sequence[str], tag: str, smoke: Bench,
                 runtime: Runtime, limits: Limits, forbidden: Sequence[str], deadline: float,
                 usage: Callable[[], dict[str, Any]], event: Callable[[str, dict[str, Any]], None],
                 generation: int, cancel: threading.Event | None = None, pinned: bool = False) -> None:
        """``pinned``: the run fixes the episodes' model, so a candidate's ``[agent].model`` is never used."""

        self.root = root
        self.cancel = cancel
        self.pinned = pinned
        self._models: dict[str, str] = {}  # model reference -> why it cannot run ("" when it can)
        self.work = root / "work"
        self.incumbent = incumbent
        self.names = tuple(names)
        self.tag = tag
        self.smoke = smoke
        self.runtime = runtime
        self.limits = limits
        self.forbidden = tuple(forbidden)
        self.deadline = deadline
        self.usage = usage
        self.event = event
        self.generation = generation
        self.checks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def catalog(self) -> list[Tool]:
        return [
            Tool("task_info", "剩余时间与 token，以及每个候选最近一次检查的结果。",
                 {"type": "object", "properties": {}}, self.task_info),
            Tool("check_candidate",
                 "按宿主的规则检查一个候选：能否加载、是否超出限制、是否含有训练任务的专有名称、相对在任版本改了哪些文件；"
                 "模板代码有改动时，在一个训练任务的本地检验期上按裁判协议实际运行模板程序。最终运行哪些候选以宿主在会话结束后"
                 "的同样检查为准。",
                 {"type": "object", "required": ["name"],
                  "properties": {"name": {"type": "string", "enum": list(self.names)}}},
                 self.check_candidate),
            Tool("note", "记下关于怎样改进 Harness 的经验、发现或无效方向，供以后各代参考。",
                 {"type": "object", "required": ["kind", "text"],
                  "properties": {"kind": {"type": "string", "enum": list(NOTE_KINDS)}, "text": {"type": "string"}}},
                 self.note),
            Tool("ask_expert", "给领域专家留言。不会立即回答，请继续工作；回答会出现在以后的证据里。",
                 {"type": "object", "required": ["question"],
                  "properties": {"question": {"type": "string"}, "context": {"type": "string"}}},
                 self.ask_expert),
        ]

    def task_info(self) -> dict[str, Any]:
        return {"minutes_left": round(max(0.0, self.deadline - time.time()) / 60, 1), **self.usage(),
                "candidates": {name: {"ok": item["ok"], "problems": item["problems"]} if (item := self.checks.get(name))
                               else "not checked yet" for name in self.names}}

    def _next(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def inspect(self, name: str) -> tuple[dict[str, Any], Path | None]:
        """Check one candidate; the result and the normalized copy (None when it cannot be loaded)."""

        if name not in self.names:
            raise ToolError(f"name must be one of {list(self.names)}")
        folder = self.root / "checks" / f"{name}-{self._next():03d}"
        staging, baseline = folder / "harness", folder / "baseline"
        meta = {"name": f"{self.tag}-g{self.generation}{name}", "parent": self.incumbent.digest, "author": "evolver"}
        try:
            copy_tree(self.work / "candidates" / name, staging)
            ignored = sorted(item.name for item in staging.iterdir() if item.name not in LAYOUT)
            for item in ignored:  # nothing outside the harness layout reaches the inner agent; drop it
                path = staging / item
                shutil.rmtree(path) if path.is_dir() else path.unlink()
            harness = normalize(staging, **meta)
            normalize(fork(self.incumbent, baseline), **meta)  # same meta, so only real edits show in the diff
        except HarnessError as error:
            return {"name": name, "ok": False, "status": "invalid", "problems": [str(error)]}, None
        problems = term_problems(staging, self.forbidden)
        if (not self.pinned and harness.agent.model != self.incumbent.agent.model
                and (problem := self.model_problem(harness.agent.model))):
            problems.append(problem)
        changes = diff(baseline, staging)
        if not changes:
            problems.append("与在任版本完全相同")
        smoke = None
        if any(item["path"].startswith("template/") for item in changes):
            program = folder / "program"
            shutil.copytree(staging / "template", program)
            report, _ = judge_program(self.smoke, program, LOCAL_STAGE, self.runtime, limits=self.limits,
                                      cancel=self.cancel)
            smoke = {key: report[key] for key in ("status", "score", "coverage", "fit_seconds", "seconds")}
            smoke["errors"] = report["errors"][:3]
            smoke["stderr_tail"] = report["stderr"][-1200:]
            if report["status"] != "ok" or report["coverage"] < 0.999:
                problems.append(f"模板程序没有在本地检验期上正常完成：状态 {report['status']}，"
                                f"覆盖率 {report['coverage']:.0%}，错误 {report['errors'][:1]}")
        proposal = _read_text(self.work / "proposals" / f"{name}.md", PROPOSAL_LIMIT).strip()
        if len(proposal) < 80:
            problems.append(f"proposals/{name}.md 还没有写（至少写清改动、依据、预期和风险）")
        status = "unchanged" if not changes else "invalid" if problems else "proposed"
        result = {"name": name, "ok": status == "proposed", "status": status, "digest": harness.digest,
                  "summary": harness.summary, "agent": asdict(harness.agent), "problems": problems,
                  "changes": changes, "smoke": smoke, "proposal": proposal,
                  "ignored": [f"{item}（不属于 Harness 的目录结构，已忽略）" for item in ignored]}
        with self._lock:
            self.checks[name] = {key: value for key, value in result.items() if key != "changes"}
        return result, staging

    def model_problem(self, ref: str) -> str:
        """Why a candidate's own model cannot run (route, key, one tiny request; cached per run), or ""."""

        with self._lock:
            known = self._models.get(ref)
        if known is None:
            try:
                providers.check(ref)
                providers.probe(ref)
                known = ""
            except providers.ModelError as error:
                known = f"harness.toml 的 [agent].model = {ref} 不能用：{error}"
            with self._lock:
                self._models[ref] = known
        return known

    def check_candidate(self, name: str) -> dict[str, Any]:
        result, _ = self.inspect(name)
        shown = {key: value for key, value in result.items() if key not in ("changes", "proposal")}
        shown["changes"] = [{key: item[key] for key in ("path", "status", "added", "removed")}
                            for item in result.get("changes", [])]
        return shown

    def note(self, kind: str, text: str) -> dict[str, Any]:
        if kind not in NOTE_KINDS:
            raise ToolError(f"kind must be one of {NOTE_KINDS}")
        self.event("evolver_note", {"generation": self.generation, "kind": kind, "text": str(text)[:4000],
                                    "at_unix": time.time()})
        return {"recorded": True}

    def ask_expert(self, question: str, context: str = "") -> dict[str, Any]:
        self.event("expert_question", {"generation": self.generation, "question": str(question)[:4000],
                                       "context": str(context)[:4000], "at_unix": time.time()})
        return {"recorded": True, "message": "已转给专家；不会立即回答，请继续工作。"}

    def collect(self) -> list[dict[str, Any]]:
        """The host's own final check of every candidate; valid ones are stored."""

        found = []
        for name in self.names:
            if self.cancel is not None and self.cancel.is_set():
                raise Cancelled("the run was cancelled")
            result, staging = self.inspect(name)
            if result["ok"] and staging is not None:
                stored = store(Harness.load(staging))
                result["digest"] = stored.digest
                result["harness"] = stored
            found.append(result)
        return found


def _task_rows(tasks: Sequence[TaskSpec], benches: dict[str, Bench]) -> list[str]:
    rows = []
    for task in tasks:
        bench = benches[task.id]
        references = reference_scores(bench, SCORED_STAGE)
        shown = "、".join(f"{name} {value:+.3f}" for name, value in references.items() if name != "naive")
        start = bench.partitions.intervals[SCORED_STAGE][task.episodes[0]][0]
        rows.append(f"| `{task.id}` | {', '.join(task.target_names)} | {list(task.horizons)} | {bench.frame.step} | "
                    f"{bench.stage(SCORED_STAGE).size}（起于第 {start} 步） | {shown or '–'} |")
    return rows


def brief(*, generation: int, suite: SuiteSpec, tasks: Sequence[TaskSpec], benches: dict[str, Bench],
          names: Sequence[str], repeats: int, episode_minutes: float, episode_tokens: int, model: str | None,
          incumbent: Harness, limits: Limits, forbidden: Sequence[str], minutes: float, tokens: int,
          deadline: float) -> str:
    gates = suite.gates
    confidence = adjusted_confidence(gates.confidence, len(names))
    ends = time.strftime("%Y-%m-%d %H:%M", time.localtime(deadline))
    pinned = (f"本次运行把内循环的模型固定为 `{model}`，修改 `model` 不起作用。" if model
              else f"在任版本用 `{incumbent.agent.model}`；可以改，但更强的模型也更贵、更慢。")
    candidate_list = "、".join(f"`candidates/{name}/`" for name in names)
    return "\n".join([
        f"# 第 {generation} 代：改进 Harness",
        "",
        "## 内循环",
        "",
        "每个回合，DSH 智能体按一版 Harness 独立解一个训练任务：读可见数据（隐藏期之前的部分），"
        "可以用 `web_search` 查资料，"
        f"自己写代码、做实验，用 ecogym 工具做本地检验和提交；每个回合限时 {episode_minutes:g} 分钟、"
        f"{episode_tokens:,} 计费 token。宿主在隐藏期（`model_selection`）上逐时刻运行最后一次被接受的提交并打分："
        "每格技能 = (RMSE基线 − RMSE) / max(RMSE, RMSE基线)，缺失的预报按 −1 计，总分是逐格加权平均。",
        "",
        "## 训练任务",
        "",
        "| 任务 | 目标 | 时距（步） | 步长 | 隐藏期计分起点 | 参考成绩（隐藏期，宿主内置方法） |",
        "|---|---|---|---|---|---|",
        *_task_rows(tasks, benches),
        "",
        "## 比较与采纳",
        "",
        f"本代每个通过检查的候选和在任版本在每个训练任务上各跑 {repeats} 个回合，按任务交替排队、预算相同，"
        "看到的专家意见也相同（本代开始时账本里的内容）。候选 − 在任的增益按任务配对，"
        + ("区间是各任务增益的 t 区间，回合间和任务间的差异都会让它变宽。" if len(tasks) >= 3 else
           "任务少于 3 个时区间由分层自助法给出（依次重抽任务、回合和天），只反映任务内的随机差异。")
        + "采纳需要同时满足：",
        "",
        f"1. 各任务平均增益大于 {gates.min_delta}；",
        f"2. 区间下限大于 0（置信度按实际比较的候选数 K 做 Bonferroni 调整，为 1 − {1 - gates.confidence:.2g}/K："
        f"{len(names)} 个候选都通过检查时是 {confidence:.1%}，即单侧 {(1 - confidence) / 2:.2%}）；",
        f"3. 没有任务的损失超过 {gates.task_cap} 且它自己的 {gates.confidence:.0%} 区间整体低于 0"
        "（两个版本在这个任务上都至少有 2 个回合时才判断：只有一个回合时区间不含回合间的随机差异）。",
        "",
        "通过的候选中增益最大的一个成为新的在任版本。在任版本在以后的每一代都会重新运行，它的成绩估计会越来越准。"
        "与在任版本或以往比较过的候选内容相同的候选不会运行。"
        "被采纳的版本最后还要在你看不到的留出任务和更晚的封存时期上各验证一次。",
        "",
        "## 你的产出",
        "",
        f"- 候选：{candidate_list}，每个都已是在任版本 `{incumbent.digest[:12]}` 的完整副本，直接修改；",
        "- 理由：写在工作区根目录的 `proposals/<名字>.md`（已有模板），不要放进候选目录——候选目录里的每个文件都会成为"
        "这个 Harness 的一部分，交给内循环智能体；",
        "- 用 `check_candidate` 检查每个候选。没有通过检查或与在任版本相同的候选不会运行。",
        "",
        "## harness.toml 的 [agent]（DSH 设置）",
        "",
        f"- `model`：DeepSeek 模型 id。{pinned}",
        "- `reasoning_effort`：off、low、high、max",
        "- `web_search`：true 或 false",
        "- `subagent_depth`：0–2，0 表示不用子智能体",
        "- `parallel_tool_calls`：1–16，一步内并行的工具调用数",
        "",
        "## 限制",
        "",
        "- 文件类型 .md .py .toml .txt .json .yaml .yml .csv；最多 300 个文件、4 MB；不能有符号链接。",
        "- 预测程序在裁判的沙箱里运行：只有 Python 标准库和 numpy、pandas、scipy、scikit-learn、statsmodels，不能联网；"
        f"fit ≤ {limits.fit_seconds:.0f} 秒，单次 forecast ≤ {limits.forecast_seconds:.0f} 秒，"
        f"拟合加全部预报 ≤ {limits.total_seconds:.0f} 秒。",
        f"- 不能出现训练任务的专有名称：{'、'.join(f'`{term}`' for term in forbidden)}。",
        "",
        "## 证据（`evidence/`）",
        "",
        "- `overview.md`：各版本在各训练任务上的隐藏期成绩、在任版本的逐格技能、各版本几个回合的预报平均后的成绩、"
        "上一代的比较结果；",
        "- `archive.md`：以往所有候选的改动、提案、结果和原因，以及评审员的笔记；",
        "- `lessons.md`：内循环智能体在各回合记下的发现、经验和无效方向；",
        "- `expert.md`：领域专家的意见（如果有）；",
        "- `episodes/<回合>/summary.md` 和 `submission/`：每个回合的成绩、时间线、笔记、工具使用、错误、"
        "最终回复和被评分的代码。",
        "",
        "## 预算",
        "",
        f"{minutes:g} 分钟，截止 {ends}；最多 {tokens:,} 个计费 token。"
        "到截止时间时会话会被终止，以当时的候选目录为准。",
        "",
    ])


def run_evolver(*, run_id: str, generation: int, incumbent: Harness, suite: SuiteSpec, tasks: Sequence[TaskSpec],
                benches: dict[str, Bench], view: dict[str, Any], ledger_path: Path, settings: EvolverSettings,
                repeats: int, episode_minutes: float, episode_tokens: int, model: str | None, limits: Limits,
                expert_text: str = "", cancel: threading.Event | None = None) -> dict[str, Any]:
    runtime = discover_runtime()
    tag = run_id.partition(":")[2] or run_id
    stamp = time.strftime("%Y%m%d%H%M%S")
    name = f"evo-{tag}-g{generation}-{stamp}"
    root = scratch_root() / name
    work = root / "work"
    trajectory = ecogym_home() / "trajectories" / name
    names = NAMES[: settings.candidates]
    deadline = time.time() + settings.minutes * 60
    gateway = make_gateway(trajectory, tokens=settings.tokens, deadline=deadline,
                           model=settings.model)  # an unusable model fails before anything is created
    for path in (work / "candidates", work / "proposals", root / "harness" / "skills", trajectory):
        path.mkdir(parents=True, exist_ok=True)
    (work / "AGENTS.md").write_text(resources.files("ecogym.evolution").joinpath("evolver.md").read_text("utf-8"),
                                    encoding="utf-8")
    fork(incumbent, work / "incumbent")
    for candidate in names:
        fork(incumbent, work / "candidates" / candidate)
        (work / "proposals" / f"{candidate}.md").write_text(
            f"# 候选 {candidate}\n\n## 改了什么\n\n## 依据\n\n## 预期\n\n## 风险\n", encoding="utf-8")
    forbidden = forbidden_terms(tasks)
    with Ledger(ledger_path) as ledger:
        write_evidence(work / "evidence", ledger=ledger, view=view, tasks=[task.id for task in tasks],
                       names=_names(view, incumbent), expert_text=expert_text,
                       ensembles=ensemble_scores(view, benches.__getitem__))
    (work / "EVOLVE.md").write_text(brief(
        generation=generation, suite=suite, tasks=tasks, benches=benches, names=names, repeats=repeats,
        episode_minutes=episode_minutes, episode_tokens=episode_tokens, model=model, incumbent=incumbent,
        limits=limits, forbidden=forbidden, minutes=settings.minutes, tokens=settings.tokens, deadline=deadline,
    ), encoding="utf-8")
    events = Events(ledger_path, run_id, frozenset({"evolver_note", "expert_question"}))
    tools = EvolverTools(root=root, incumbent=incumbent, names=names, tag=tag, smoke=benches[tasks[0].id],
                         runtime=runtime, limits=limits, forbidden=forbidden, deadline=deadline, usage=gateway.usage,
                         event=events, generation=generation, cancel=cancel, pinned=model is not None)
    agent = AgentSpec(model=settings.model, reasoning_effort=settings.reasoning_effort, web_search=True,
                      subagent_depth=1, parallel_tool_calls=10)
    try:
        session = run_agent(root=root, prompt=PROMPT, agent=agent, tools=tools.catalog(), gateway=gateway,
                            deadline=deadline, trajectory=trajectory, runtime=runtime, cancel=cancel)
        candidates = tools.collect()
    finally:
        subprocess.run(["rm", "-r", "-f", str(root)], check=False)
    return {"session": session.to_dict(), "candidates": candidates}


def _names(view: dict[str, Any], incumbent: Harness) -> dict[str, str]:
    names = {view["seed"]["digest"]: view["seed"].get("name", "seed"), incumbent.digest: incumbent.name}
    for round_ in view["rounds"]:
        for candidate in round_["candidates"]:
            if candidate.get("digest"):
                names[candidate["digest"]] = f"g{round_['generation']}{candidate['name']}"
    return names


__all__ = ["NAMES", "EvolverSettings", "EvolverTools", "brief", "copy_tree", "forbidden_terms", "run_evolver"]
