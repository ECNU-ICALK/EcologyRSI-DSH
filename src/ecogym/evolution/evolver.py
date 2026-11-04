"""One evolver session: a DSH agent reviews the evidence and writes candidate harnesses.

Workspace, inside the agent sandbox:

    work/AGENTS.md            the evolver's fixed instructions (``evolver.md``, host-owned)
    work/EVOLVE.md            this round's brief: tasks, comparison rules, options, limits, budget
    work/incumbent/           the incumbent harness, for reference
    work/archive/<label>/     versions compared before (this run's, other runs' on training tasks all among
                              this run's, the seed)
    work/candidates/<name>/   one editable copy of the incumbent per candidate
    work/proposals/<name>.md  each candidate's rationale
    work/variants/<name>/     template variants to screen (not part of any harness)
    work/evidence/            overview, archive, lessons, contrasts, prior runs, expert notes, one folder per episode

Tools: ``task_info``, ``check_candidate``, ``screen_templates``,
``reset_candidate``, ``note`` and ``ask_expert``. ``screen_templates`` runs
many template variants on every training task's local check before any of them
becomes a candidate. When the session ends the host checks every candidate again on
its own: it copies the directory without following links, writes ``name``,
``parent`` and ``author``, loads and digests it, refuses names of training
tasks and versions compared before, diffs it against the incumbent, and runs
its template program on the training tasks' local checks if the template
changed. Only candidates that pass are stored and run. Setting ``cancel`` also
stops these checks.

A candidate may start from any archived version (``reset_candidate``), not
only from the incumbent; it is still compared with the incumbent. Search that
keeps non-adopted versions as stepping stones beats always building on the
current best (the Darwin Gödel Machine's open-ended-exploration ablation,
Zhang et al., 2025; Huxley-Gödel Machine; GEPA's Pareto candidate pool;
AIRA-2's population): a candidate that was better on most tasks but missed the
gate is often a better parent than the incumbent. The archive also holds the
candidates of other runs whose training tasks are all among this run's (no
other run can carry hidden-period knowledge of this run's holdout tasks).
None of the versions compared on exactly this run's training tasks runs again
unchanged: that would only retest it until it passed by chance; a version
compared on part of them may run once more, on all of them. Two candidates of one round
whose edits mostly coincide get a warning: they split one round's power
(Bonferroni) without testing two directions (ShinkaEvolve's novelty
rejection; on 2026-10-29 and 11-01 three of nine rounds had two candidates
sharing 52 to 96 % of their added lines).
"""

from __future__ import annotations

import contextlib
import difflib
import os
import re
import shutil
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from ..episodes import providers
from ..episodes.agent import AgentSpec, Cancelled, make_gateway, run_agent
from ..episodes.judge import Limits, judge_program, reference_scores
from ..episodes.mcp import Tool, ToolError
from ..episodes.runner import Events
from ..episodes.sandbox import Runtime, check_workers, discover_runtime, ecogym_home, scratch_root
from ..episodes.tools import REPLACE_MARGIN
from ..episodes.workspace import LOCAL_STAGE, SCORED_STAGE
from ..evaluation.harness import Bench
from ..evaluation.suite import adjusted_confidence, signed_rank_critical
from ..harness import (
    MAX_BYTES,
    MAX_FILES,
    Harness,
    HarnessError,
    behavior_digest,
    content_files,
    diff,
    fork,
    normalize,
    resolve,
    store,
    term_problems,
)
from ..kernel import clock
from ..kernel.ledger import Ledger
from ..task.spec import TaskSpec, base_id
from ..task.suite import SuiteSpec
from .ensemble import ensemble_scores
from .evidence import related_runs, same_tasks, version_label, write_evidence

NAMES = "abcdefghij"
SCREEN_LIMIT = 12  # template variants screen_templates judges in one call
SCREEN_TOTAL = 48  # ... and in one evolver session
SCREEN_TIE = 0.0005  # a task's local change this small counts as neither better nor worse
VARIANT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,40}$")
ARCHIVE_LIMIT = 24  # earlier candidates copied into the evolver's archive/
SIMILAR_EDITS = 0.5  # two candidates sharing this share of their added lines get a warning
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


def _compared(view: dict[str, Any], *, finished_only: bool = False) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(round, candidate) for every candidate of a folded run that was compared on all training tasks."""

    return [(round_, candidate) for round_ in view["rounds"] if round_["finished"] or not finished_only
            for candidate in round_["candidates"] if candidate.get("decision")
            and not str(candidate["decision"].get("reason", "")).startswith("missing_tasks")]


def archive_versions(view: dict[str, Any], incumbent: Harness,
                     prior: Sequence[dict[str, Any]] = ()) -> dict[str, Harness]:
    """Earlier versions a candidate may start from, keyed by label: every stored candidate this run compared
    (``g2b``, newest first), then those of ``prior`` runs on the same training tasks (``35ad33-g1b``: first the
    ones compared with this run's incumbent, by their gain), then the seed. The incumbent is left out
    (``work/incumbent`` holds it), and so is any version already listed under another label."""

    found: dict[str, Harness] = {}

    def add(label: str, digest: str | None) -> None:
        # a round interrupted and run again repeats its labels: the newer one, seen first, keeps the label
        if (len(found) < ARCHIVE_LIMIT and digest and digest != incumbent.digest and label not in found
                and all(harness.digest != digest for harness in found.values())):
            with contextlib.suppress(HarnessError):  # not in this machine's store: listed, not copied
                found[label] = resolve(digest)

    for round_, candidate in reversed(_compared(view)):
        add(version_label(view["run_id"], round_["generation"], candidate["name"], current=view["run_id"]),
            candidate.get("digest"))
    earlier = [(other, round_, candidate) for other in prior for round_, candidate in _compared(other)]
    earlier.sort(key=lambda item: (item[2].get("parent") != incumbent.digest, -(item[2]["decision"].get("delta") or 0)))
    for other, round_, candidate in earlier:
        add(version_label(other["run_id"], round_["generation"], candidate["name"], current=view["run_id"]),
            candidate.get("digest"))
    if view["seed"]["digest"] != incumbent.digest and len(found) < ARCHIVE_LIMIT:
        with contextlib.suppress(HarnessError):
            found["seed"] = resolve(view["seed"]["digest"])
    return found


def known_versions(view: dict[str, Any], incumbent: Harness, prior: Sequence[dict[str, Any]] = ()) -> dict[str, str]:
    """Behavior digest -> label of every version that will not run again: the seed, the incumbent, every candidate
    compared in a finished round of this run (the loop's ``State.tried``) and of the ``prior`` runs on exactly
    the same training tasks. Running a compared version again on the same tasks only retests it until it passes
    by chance. A version compared on fewer tasks (a prior run on part of this run's training tasks) may run once
    here: the test on more tasks is a stronger one, as in staged evaluation (the Darwin Gödel Machine evaluates a
    promising agent on more tasks), and once compared here it is known like any other."""

    known = {view["seed"].get("behavior") or "": "起点版本"}
    for other in prior:
        if not same_tasks(view, other):
            continue
        for round_, candidate in _compared(other, finished_only=True):
            if candidate.get("behavior"):
                label = version_label(other["run_id"], round_["generation"], candidate["name"], current=view["run_id"])
                known[candidate["behavior"]] = (f"以往运行 {other['run_id']} 第 {round_['generation']} 代的候选 "
                                                f"{candidate['name']}（{label}）")
    for round_, candidate in _compared(view, finished_only=True):
        if candidate.get("behavior"):
            label = version_label(view["run_id"], round_["generation"], candidate["name"], current=view["run_id"])
            known[candidate["behavior"]] = f"第 {round_['generation']} 代的候选 {candidate['name']}（{label}）"
    known.pop("", None)
    known[behavior_digest(incumbent)] = "在任版本"
    return known


def edit_lines(root: Path, base: Path) -> set[str]:
    """Lines a harness adds or changes relative to ``base``, each prefixed with its file; ``harness.toml``
    (names and summary) is left out. Empty when either tree cannot be read as a harness."""

    try:
        before = {path.as_posix() for path in content_files(base)}
        files = [path for path in content_files(root) if path.as_posix() != "harness.toml"]
    except HarnessError:
        return set()
    found = set()
    for relative in files:
        key = relative.as_posix()
        new = (root / relative).read_text(encoding="utf-8", errors="replace").splitlines()
        old = (base / relative).read_text(encoding="utf-8", errors="replace").splitlines() if key in before else []
        if new == old:
            continue
        for line in difflib.unified_diff(old, new, lineterm="", n=0):
            if line.startswith("+") and not line.startswith("+++") and line[1:].strip():
                found.add(f"{key}:{line[1:].strip()}")
    return found


def edit_overlap(first: set[str], second: set[str]) -> float:
    """Share of added or changed lines two candidates have in common (Jaccard); 0 when either has none."""

    return len(first & second) / len(first | second) if first and second else 0.0


def archive_readme(view: dict[str, Any], archived: dict[str, Harness], prior: Sequence[dict[str, Any]] = (),
                   incumbent: str = "") -> str:
    """``work/archive/README.md``: what each archived version is and how it did."""

    ours = len((view.get("suite") or {}).get("train") or [])
    fewer = "、".join(f"`{other['run_id']}`" for other in prior if not same_tasks(view, other))
    lines = ["# 存档：以往比较过的版本", "",
             "每个目录是一个以往版本的完整副本（只读参考）：`g2b` 这样的是本次运行的候选，`35ad33-g1b` 这样带前缀的来自"
             "其他运行（前缀是运行编号的开头），它们的训练任务都在本次之内。新候选可以从其中任何一个出发：调用 "
             "`reset_candidate(name, base)` 把 `candidates/<name>/` 重置为它的副本再修改，也可以把两个版本在不同任务上"
             "有效的部分合到一个候选里。比较对象始终是在任版本。与本次或训练任务相同的运行里比较过的版本内容相同的候选"
             "不会运行。"
             + (f"来自训练任务比本次少的运行（{fewer}）的版本只在那几个任务上比较过，"
                f"可以原样在本次的 {ours} 个训练任务上再比较一次：任务多了，以前分不清的小增益可能分得清。"
                if fewer else "")
             + "下表的增益是相对它当时的在任版本、在当时的训练任务上算的，"
             "“当时的在任”一列标出它是否就是本次的在任版本；"
             "本次运行的细节见 `evidence/archive.md`，其他运行见 `evidence/prior_runs.md`。"
             "标 ★ 的版本在当时的全部训练任务上都好于在任版本，只是平均增益不够大或区间跨 0，没被采纳："
             "这样的改动多半是真的但很小，在它上面再加一处有效的改动，或把两个 ★ 版本的改动合在一起，"
             "最可能通过门槛（先用 `screen_templates` 在本地检验期上看叠加后是否还有效）。", "",
             "| 目录 | 摘要 | 当时的结果 | 好于当时在任的任务 | 当时的在任 |", "|---|---|---|---|---|"]
    by_label = {version_label(other["run_id"], round_["generation"], candidate["name"], current=view["run_id"]):
                candidate for other in (*prior, view) for round_, candidate in _compared(other)}
    for label, harness in archived.items():
        candidate = by_label.get(label) or {}
        decision = candidate.get("decision") or {}
        tasks = decision.get("tasks") or []
        everywhere = bool(tasks) and not candidate.get("missing") and all((item.get("delta") or 0) > 0
                                                                          for item in tasks)
        stack = everywhere and candidate.get("status") not in ("adopted", "passed")
        result = (("★ " if stack else "") + f"Δ {decision['delta']:+.4f}，{decision.get('reason')}"
                  + ("（采纳）" if candidate.get("status") == "adopted" else "")
                  if isinstance(decision.get("delta"), (int, float)) else "起点版本" if label == "seed" else "–")
        better = f"{sum(1 for item in tasks if (item.get('delta') or 0) > 0)}/{len(tasks)}" if tasks else "–"
        parent = str(candidate.get("parent") or "")
        against = "–" if not parent else "本次的在任版本" if parent == incumbent else f"`{parent[:12]}`"
        summary = " ".join((harness.summary or "").split())[:200].replace("|", "/")
        lines.append(f"| `{label}/`（`{harness.digest[:12]}`） | {summary} | {result} | {better} | {against} |")
    return "\n".join(lines) + "\n"


class EvolverTools:
    def __init__(self, *, root: Path, incumbent: Harness, names: Sequence[str], tag: str, smoke: Bench,
                 runtime: Runtime, limits: Limits, forbidden: Sequence[str], deadline: float,
                 usage: Callable[[], dict[str, Any]], event: Callable[[str, dict[str, Any]], None],
                 generation: int, cancel: threading.Event | None = None, pinned: bool = False,
                 others: Sequence[Bench] = (), archived: dict[str, Harness] | None = None,
                 known: dict[str, str] | None = None) -> None:
        """``pinned``: the run fixes the episodes' model, so a candidate's ``[agent].model`` is never used.

        A changed template runs on ``smoke``'s local check (it must finish there) and on each of ``others``
        (the other training tasks; problems there are warnings). ``archived`` are the versions a candidate may
        be reset to (label -> harness); ``known`` the behavior digests the loop would not run again (-> label)."""

        self.root = root
        self.cancel = cancel
        self.pinned = pinned
        self._models: dict[str, str] = {}  # model reference -> why it cannot run ("" when it can)
        self.work = root / "work"
        self.incumbent = incumbent
        self.archived = dict(archived or {})
        self.known = dict(known or {})
        self.bases: dict[str, str] = {}  # candidate -> archive label it was last reset to
        # candidate -> its added lines (against the incumbent, against its base) at its latest check
        self._edits: dict[str, tuple[set[str], set[str]]] = {}
        self.names = tuple(names)
        self.tag = tag
        self.smoke = smoke
        self.others = tuple(others)
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
        self._incumbent_smoke: dict[str, dict[str, Any]] | None = None  # the incumbent template's checks, once
        self._incumbent_lock = threading.Lock()  # held while it is judged
        self._screen_lock = threading.Lock()  # one screen_templates call at a time: each already judges several tasks
        self._screened = 0  # variants screened in this session

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
            Tool("screen_templates",
                 "批量试模板：variants/ 下每个目录是一个完整的 template/ 副本（含 predictor.py），"
                 "在每个训练任务的本地检验期上按裁判协议运行，和在任版本的模板比较，"
                 "返回每个变体的平均分、平均增益、更好和更差的任务数与各任务分数，按平均增益排序。"
                 f"每次最多 {SCREEN_LIMIT} 个、每个会话最多 {SCREEN_TOTAL} 个，一次调用约一两分钟；"
                 "只用本地检验期，不涉及隐藏期。用它在做成候选之前多试几个方向或参数，"
                 "再把选中的变体复制进候选的 template/。",
                 {"type": "object", "required": ["variants"],
                  "properties": {"variants": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                                              "maxItems": SCREEN_LIMIT}}},
                 self.screen_templates),
            Tool("reset_candidate",
                 "把一个候选目录重置为在任版本（base 为 incumbent）或 archive/ 里某个以往版本的完整副本，"
                 "原有改动全部丢弃，然后在它上面修改。比较对象始终是在任版本；"
                 "与以往比较过的版本内容完全相同的候选不会运行。",
                 {"type": "object", "required": ["name", "base"],
                  "properties": {"name": {"type": "string", "enum": list(self.names)},
                                 "base": {"type": "string", "enum": ["incumbent", *self.archived]}}},
                 self.reset_candidate),
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
        return {"minutes_left": round(max(0.0, self.deadline - clock.now()) / 60, 1), **self.usage(),
                "candidates": {name: {"ok": item["ok"], "problems": item["problems"]} if (item := self.checks.get(name))
                               else "not checked yet" for name in self.names}}

    def _next(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def reset_candidate(self, name: str, base: str) -> dict[str, Any]:
        if name not in self.names:
            raise ToolError(f"name must be one of {list(self.names)}")
        if base != "incumbent" and base not in self.archived:
            raise ToolError(f"base must be incumbent or one of {list(self.archived)}")
        source = self.incumbent if base == "incumbent" else self.archived[base]
        target = self.work / "candidates" / name
        with self._lock:  # never two resets of one directory at once
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.exists():
                shutil.rmtree(target)
            fork(source, target)
            if base == "incumbent":
                self.bases.pop(name, None)
            else:
                self.bases[name] = base
            self.checks.pop(name, None)
        return {"reset": True, "name": name, "base": base, "digest": source.digest,
                "summary": source.summary,
                "message": f"candidates/{name}/ 现在是 {base} 的副本。在它上面修改；改动和比较都相对在任版本计算，"
                           f"proposals/{name}.md 里写明从哪个版本出发、为什么。"}

    def _edit_sets(self, name: str, root: Path) -> tuple[set[str], set[str]]:
        """A candidate's added lines relative to the incumbent, and relative to the archived version it was reset
        to (the same set when it starts from the incumbent)."""

        against_incumbent = edit_lines(root, self.incumbent.root)
        base = self.bases.get(name)
        return against_incumbent, edit_lines(root, self.archived[base].root) if base else against_incumbent

    def _current_edits(self, name: str) -> tuple[set[str], set[str]]:
        """Another candidate's edit sets as its directory stands now (copied without following links)."""

        folder = self.root / "checks" / f"{name}-peek-{self._next():03d}"
        try:
            copy_tree(self.work / "candidates" / name, folder)
            return self._edit_sets(name, folder)
        except (HarnessError, OSError):
            return set(), set()
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def similar(self, name: str, edits: dict[str, tuple[set[str], set[str]]]) -> list[str]:
        """Warnings for the candidates of this round whose edits mostly coincide with ``name``'s.

        Overlap is measured relative to the incumbent, which is what both are compared with: two candidates reset
        to the same archived version share that version's whole diff, and their comparisons then mostly measure
        it twice, whatever each adds on top (2026-11-01: two candidates from one archived version, 58 %)."""

        found = []
        mine = edits.get(name, (set(), set()))
        for other in self.names:
            theirs = edits.get(other, (set(), set()))
            share = edit_overlap(mine[0], theirs[0]) if other != name else 0.0
            if share < SIMILAR_EDITS:
                continue
            base = self.bases.get(name)
            if base and self.bases.get(other) == base:
                own = edit_overlap(mine[1], theirs[1])
                found.append(f"候选 {name} 和 {other} 都从 {base} 出发，相对在任版本新增或改动的行有 {share:.0%} 相同"
                             f"（在 {base} 之上各自的改动有 {own:.0%} 相同）："
                             "两次比较测到的大部分是同一个起点版本的效果，会分走同一代的检验力"
                             "（区间按候选数做 Bonferroni 调整）。如果是有意做的两个强度，在提案里写明；"
                             "否则让其中一个换一个起点或方向。")
            else:
                found.append(f"候选 {name} 和 {other} 新增或改动的行有 {share:.0%} 相同：两个候选的方向几乎一样，"
                             "会分走同一代的检验力（区间按候选数做 Bonferroni 调整），比较也说明不了哪个方向有效。"
                             "让其中一个换一个方向，或者只保留一个。")
        return found

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
            self._edits.pop(name, None)
            return {"name": name, "ok": False, "status": "invalid", "problems": [str(error)]}, None
        problems = term_problems(staging, self.forbidden)
        if (not self.pinned and harness.agent.model != self.incumbent.agent.model
                and (problem := self.model_problem(harness.agent.model))):
            problems.append(problem)
        changes = diff(baseline, staging)
        same = self.known.get(behavior_digest(harness)) if changes else None
        if not changes:
            problems.append("与在任版本完全相同")
        elif same is not None:
            problems.append(f"与{same}内容相同（harness.toml 的名称和摘要不算）：比较过的版本不会再运行，"
                            "在它上面做出新的改动")
        self._edits[name] = self._edit_sets(name, staging)
        smoke, warnings = None, []
        if any(item["path"].startswith("template/") for item in changes):
            reports = self._judge_template(staging / "template", folder)
            report = reports[self.smoke.task.id]
            smoke = {key: report[key] for key in ("status", "score", "coverage", "fit_seconds", "seconds")}
            smoke["errors"] = report["errors"][:3]
            smoke["stderr_tail"] = report["stderr"][-1200:]
            smoke["cells"] = {cell["cell"]: round(cell["skill"], 4) for cell in report["cells"]}
            if report["status"] != "ok" or report["coverage"] < 0.999:
                problems.append(f"模板程序没有在本地检验期上正常完成：状态 {report['status']}，"
                                f"覆盖率 {report['coverage']:.0%}，错误 {report['errors'][:1]}")
            else:
                reference = self.incumbent_smoke()
                smoke["incumbent_score"] = reference[self.smoke.task.id].get("score")
                smoke["tasks"] = {task: {"status": item["status"], "score": item["score"],
                                         "coverage": item["coverage"], "incumbent_score": reference[task].get("score")}
                                  for task, item in reports.items()}
                warnings = template_warnings(reports, reference, main=self.smoke.task.id, predictor_changed=any(
                    item["path"] == "template/predictor.py" for item in changes))
        proposal = _read_text(self.work / "proposals" / f"{name}.md", PROPOSAL_LIMIT).strip()
        if len(proposal) < 80:
            problems.append(f"proposals/{name}.md 还没有写（至少写清改动、依据、预期和风险）")
        status = ("unchanged" if not changes else "duplicate" if same is not None else "invalid" if problems
                  else "proposed")
        base = self.bases.get(name)
        result = {"name": name, "ok": status == "proposed", "status": status, "digest": harness.digest,
                  "summary": harness.summary, "agent": asdict(harness.agent), "problems": problems,
                  "warnings": warnings, "changes": changes, "smoke": smoke, "proposal": proposal,
                  "base": {"label": base, "digest": self.archived[base].digest} if base else None,
                  "ignored": [f"{item}（不属于 Harness 的目录结构，已忽略）" for item in ignored]}
        with self._lock:
            self.checks[name] = {key: value for key, value in result.items() if key != "changes"}
        return result, staging

    def _judge_template(self, template: Path, folder: Path) -> dict[str, dict[str, Any]]:
        """Judge reports of a template on every check task's local check, ``check_workers()`` tasks at a time: task
        id -> report.

        Only the first task's report decides whether the candidate may run; the others feed warnings, since a
        template can break on one task alone (the template adopted on 2026-11-01 forecast nonsense on one of the
        four training tasks, local skill −0.957, and scored +0.11 to +0.22 on every other task of its suite)."""

        benches = [self.smoke, *self.others]

        def one(index: int) -> dict[str, Any]:
            program = folder / f"program-{index}"
            shutil.rmtree(program, ignore_errors=True)
            shutil.copytree(template, program)
            report, _ = judge_program(benches[index], program, LOCAL_STAGE, self.runtime, limits=self.limits,
                                      cancel=self.cancel)
            return report

        # a few at a time, not all: eight programs fitting at once heat a laptop for little gain in time
        with ThreadPoolExecutor(max_workers=min(len(benches), check_workers()), thread_name_prefix="template-check") \
                as pool:
            reports = list(pool.map(one, range(len(benches))))
        return {bench.task.id: report for bench, report in zip(benches, reports, strict=True)}

    def incumbent_smoke(self) -> dict[str, dict[str, Any]]:
        """The incumbent's own template on the same local checks, by task (judged once per session; concurrent
        checks wait for it): the reference for a candidate's template scores."""

        with self._incumbent_lock:
            if self._incumbent_smoke is None:
                template = self.incumbent.root / "template"
                if not (template / "predictor.py").is_file():
                    self._incumbent_smoke = {bench.task.id: {"status": "no_template", "score": None}
                                             for bench in (self.smoke, *self.others)}
                else:
                    reports = self._judge_template(template, self.root / "checks" / "incumbent")
                    self._incumbent_smoke = {task: {"status": report["status"], "score": report["score"]
                                                    if report["status"] == "ok" else None}
                                             for task, report in reports.items()}
            return self._incumbent_smoke

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
        if name in self._edits:  # the other candidates as their directories stand now, not at their last check
            edits = {other: self._current_edits(other) for other in self.names if other != name}
            shown["warnings"] = [*shown.get("warnings", []), *self.similar(name, {**edits, name: self._edits[name]})]
        return shown

    def screen_templates(self, variants: Sequence[str]) -> dict[str, Any]:
        """Judge template variants (directories under ``work/variants/``) on every training task's local check and
        compare each with the incumbent's template, best mean gain first.

        A check_candidate costs a candidate slot and an hour of episodes to learn what one template does; the local
        check of a whole template takes seconds, and its mean over the training tasks predicts the hidden-period
        gain well (2026-11-04: correlation +0.86 over nine candidates), its per-task changes poorly (+0.21). So the
        evolver can try many directions or settings here and spend its candidates on the best. Only the local check
        period is used: nothing here tells the hidden period's values. Calls run one at a time."""

        names = [str(item).strip().removeprefix("variants/").strip("/") for item in (variants or [])]
        if not names:
            raise ToolError("variants must name at least one directory under variants/")
        if len(names) > SCREEN_LIMIT:
            raise ToolError(f"at most {SCREEN_LIMIT} variants per call")
        if len(set(names)) != len(names):
            raise ToolError("name each variant once")
        bad = [name for name in names if not VARIANT_NAME.fullmatch(name)]
        if bad:
            raise ToolError(f"{bad[0]}: a variant is the name of a directory under variants/ (letters, digits, . _ -)")
        with self._screen_lock:
            if self._screened + len(names) > SCREEN_TOTAL:
                left = SCREEN_TOTAL - self._screened
                raise ToolError(f"this session may screen {SCREEN_TOTAL} variants; {left} left")
            reference = self.incumbent_smoke()
            scores = [item["score"] for item in reference.values() if isinstance(item.get("score"), float)]
            base = float(sum(scores) / len(scores)) if scores else None
            results = []
            for name in names:
                if self.cancel is not None and self.cancel.is_set():
                    raise Cancelled("the run was cancelled")
                results.append(self._screen_one(name, reference))
                if results[-1]["status"] in ("ok", "failed"):  # judged: a missing or broken directory costs nothing
                    self._screened += 1
        results.sort(key=lambda item: (item.get("gain") is None, -(item.get("gain") or 0.0)))
        without = sorted(task for task, item in reference.items() if not isinstance(item.get("score"), float))
        return {"incumbent_mean": base, "tasks": [bench.task.id for bench in (self.smoke, *self.others)],
                "variants": results, "left": SCREEN_TOTAL - self._screened,
                **({"incumbent_failed": without} if without else {}),
                "note": "本地检验期比隐藏期短得多：以多个任务的平均增益和更好的任务数为准，"
                        "单个任务上的本地差别常常在隐藏期上不成立。选中的变体复制进 candidates/<名字>/template/，"
                        "再用 check_candidate 检查；最终由隐藏期上的比较决定。"
                        + ("在任版本的模板在 incumbent_failed 列出的任务上没有正常完成，增益只在其他任务上算。"
                           if without else "")}

    def _screen_one(self, name: str, reference: dict[str, dict[str, Any]]) -> dict[str, Any]:
        source = self.work / "variants" / name
        if source.is_symlink() or not source.is_dir():
            return {"name": name, "status": "missing", "problem": f"variants/{name}/ 不存在或不是目录"}
        folder = self.root / "checks" / f"screen-{self._next():03d}-{name}"
        template = folder / "template"
        try:
            try:
                copy_tree(source, template)
            except (HarnessError, OSError) as error:
                return {"name": name, "status": "invalid", "problem": str(error)}
            if not (template / "predictor.py").is_file():
                return {"name": name, "status": "invalid",
                        "problem": "没有 predictor.py：变体应是一个完整的 template/ 目录副本，例如 "
                                   f"cp -r candidates/a/template variants/{name}"}
            reports = self._judge_template(template, folder)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        tasks, failed = {}, []
        for task, report in reports.items():
            incumbent = (reference.get(task) or {}).get("score")
            ok = report["status"] == "ok" and report["coverage"] >= 0.999
            score = report["score"] if ok else None
            tasks[task] = {"score": score, "incumbent": incumbent,
                           "delta": score - incumbent if isinstance(score, float) and isinstance(incumbent, float)
                           else None}
            if not ok:
                failed.append(f"{task}：状态 {report['status']}，覆盖率 {report['coverage']:.0%}，"
                              f"错误 {report['errors'][:1]}")
        if failed:
            return {"name": name, "status": "failed", "problem": "；".join(failed[:3]), "tasks": tasks}
        values = [item["score"] for item in tasks.values()]
        deltas = [item["delta"] for item in tasks.values() if item["delta"] is not None]
        return {"name": name, "status": "ok", "mean": float(sum(values) / len(values)),
                "gain": float(sum(deltas) / len(deltas)) if deltas else None,  # over the tasks the incumbent scored
                "better": sum(1 for delta in deltas if delta > SCREEN_TIE),
                "worse": sum(1 for delta in deltas if delta < -SCREEN_TIE),
                "worst": min(deltas) if deltas else None, "tasks": tasks}

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
        edits = {result["name"]: self._edits[result["name"]] for result in found
                 if result["ok"] and result["name"] in self._edits}  # every directory final: the overlap is exact
        for result in found:
            if result["name"] in edits:
                result["warnings"] = [*result.get("warnings", []), *self.similar(result["name"], edits)]
        return found


FOLLOW_UP_SHARE = 0.5  # an evolver that ends with more than this share of its time left gets one follow-up session


def evolver_follow_up(*, deadline: float, minutes: float, widest: Sequence[str] | None
                      ) -> Callable[[int, str], str | None]:
    """The prompt of the one follow-up an evolver gets when it ends early: read the two sessions of the widest
    contrast, then keep or revise the candidates.

    Evolvers end early, like the inner agents (2026-11-01: 271 s and 278 s of 15 minutes), and neither of those
    sessions opened a single transcript although both read ``contrasts.md`` and the instructions asked for it.
    The raw sessions are what Meta-Harness found a proposer needs most, so the spare time goes to them."""

    def ask(n: int, reply: str) -> str | None:
        left = deadline - clock.now()
        if n > 1 or not widest or left < FOLLOW_UP_SHARE * minutes * 60:
            return None
        best, worst = widest
        return "\n".join([
            f"这是同一个评审会话的第 2 段。上一段已经结束，但还剩约 {left / 60:.0f} 分钟（共 {minutes:g} 分钟）；"
            "候选目录、提案和检查结果都还在，用 task_info 查看。",
            f"请先对照 `evidence/{best}`（好回合）和 `evidence/{worst}`（差回合）：它们是 `evidence/contrasts.md` 里"
            "差距最大的一组，同一版 Harness、同一个任务。看两个智能体在哪一步做了不同的选择、各自把时间花在哪里、"
            "差回合为什么没有做对。",
            "然后判断候选要不要据此修改：要改就改，并重新用 check_candidate 检查；不需要改就不要改。"
            "最终回复里先写一段你从这两份会话记录里看到的差别，再写各候选的方向、依据和预期。",
            "", "上一段会话的最终回复（节选）：", "", reply.strip()[-1500:] or "（没有）",
        ])

    return ask


TEMPLATE_DROP = 0.02  # a changed template this far below the incumbent's on a task's check gets a warning
TEMPLATE_FLOOR = -0.2  # ... and one this far below the naive baseline gets one whatever the incumbent scored


def template_warnings(reports: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]], *, main: str,
                      predictor_changed: bool = True) -> list[str]:
    """What a changed template's check runs (judge reports by task, ``main`` first) suggest is wrong, without
    blocking the candidate; ``reference`` holds the incumbent template's status and score by task.

    On 2026-11-01 both candidates of a round fitted their residual model with the sign reversed (baseline −
    truth, then added to the baseline); their own validation gate then switched every cell off, the
    templates forecast the naive baseline exactly, and the evolver took the 0.000 for a safe lower bound.
    The template adopted that day scored −0.957 on one training task's check and well everywhere else.
    """

    found = []
    cells = reports[main].get("cells") or []
    if predictor_changed and cells and all(abs(cell["skill"]) < 1e-9 for cell in cells):
        found.append("模板在每一格的预报都和朴素基线完全相同（本地技能恰好为 0）：在基线上加的修正一格也没有生效。"
                     "如果模板本该建模，检查残差的方向（目标应为 真值 − 基线，预报 = 基线 + 修正）、"
                     "启用门槛和回退条件。")
    for task, report in reports.items():
        if report["status"] != "ok" or report["coverage"] < 0.999:
            if task != main:  # on the main task this is a problem, not a warning
                found.append(f"模板在 {task} 的本地检验期上没有正常完成：状态 {report['status']}，"
                             f"覆盖率 {report['coverage']:.0%}，错误 {report['errors'][:1]}。")
            continue
        score, incumbent = report.get("score"), (reference.get(task) or {}).get("score")
        versus = f"（在任版本的模板 {incumbent:+.3f}）" if isinstance(incumbent, float) else ""
        if isinstance(score, float) and isinstance(incumbent, float) and score < incumbent - TEMPLATE_DROP:
            found.append(f"模板在 {task} 上本地 {score:+.3f}，低于在任版本模板的 {incumbent:+.3f}："
                         "这个任务的回合从更差的起点开始。")
        elif isinstance(score, float) and score < TEMPLATE_FLOOR:
            found.append(f"模板在 {task} 上本地 {score:+.3f}{versus}，比朴素基线差很多：这个任务的回合从比朴素基线"
                         "还差的起点开始。常见原因是数值问题，例如近乎常数的列标准化后被放大、外推失控或缺失值处理。")
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


def _archive_rows(archived: dict[str, Harness]) -> list[str]:
    if not archived:
        return ["还没有比较过的候选，`archive/` 是空的；候选从在任版本出发。", ""]
    return ["`archive/` 里是以往比较过的版本的完整副本（`archive/README.md` 列出每个版本当时的结果）："
            + "、".join(f"`{label}`" for label in archived) + "。"
            "`g2b` 这样的是本次运行的候选，带前缀的来自训练任务都在本次之内的其他运行。"
            "候选不必从在任版本出发：没被采纳、但在多数任务上好于当时在任版本的候选，常常是更好的起点。"
            "`reset_candidate(name, base)` 把 `candidates/<name>/` 重置为在任版本或某个存档版本，再在上面修改；"
            "也可以把两个版本在不同任务上有效的部分合进一个候选。比较对象始终是在任版本。"
            "与在同样的训练任务上比较过的版本内容完全相同的候选不会运行；只在更少的任务上比较过的版本"
            "（`archive/README.md` 写明是哪些）可以原样提交，在本次的全部训练任务上再比较一次。",
            ""]


def brief(*, generation: int, suite: SuiteSpec, tasks: Sequence[TaskSpec], benches: dict[str, Bench],
          names: Sequence[str], repeats: int, episode_minutes: float, episode_tokens: int, model: str | None,
          incumbent: Harness, limits: Limits, forbidden: Sequence[str], minutes: float, tokens: int,
          deadline: float, archived: dict[str, Harness] | None = None) -> str:
    gates = suite.gates
    confidence = adjusted_confidence(gates.confidence, len(names))
    critical = signed_rank_critical(len(tasks), (1 - confidence) / 2) if gates.test == "signed_rank" else -1
    if len(tasks) < 3:
        interval_text = "任务少于 3 个时区间由分层自助法给出（依次重抽任务、回合和天），只反映任务内的随机差异。"
        rank_rule = ""
    elif critical >= 0:
        interval_text = ("区间是各任务增益的符号秩区间（Wilcoxon 符号秩检验对应的 Hodges–Lehmann 区间）：只看每个任务"
                         "是变好还是变差、变化的大小在各任务里排第几，不看具体差多少。所以一个任务上特别大的增益不会把区间"
                         "拉宽，但几乎每个任务都要变好；平均增益（条件 1）仍按各任务增益的平均算。")
        worse = "在所有任务里最小" if critical == 1 else f"是最小的 {critical} 个之一"
        rank_rule = (f"这时要 {len(tasks)} 个训练任务全部更好" if critical == 0 else
                     f"这时要 {len(tasks)} 个训练任务全部更好，或者变差的任务按变化大小（绝对值从小到大）排的名次加起来"
                     f"不超过 {critical}，例如只有一个任务变差、且它的变化{worse}")
    else:
        interval_text = "区间是各任务增益的 t 区间，回合间和任务间的差异都会让它变宽。"
        rank_rule = ""
    ends = time.strftime("%Y-%m-%d %H:%M", time.localtime(clock.wall(deadline)))
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
        f"{episode_tokens:,} 计费 token。宿主在隐藏期（`model_selection`）上逐时刻运行被评分的程序并打分："
        "每格技能 = (RMSE基线 − RMSE) / max(RMSE, RMSE基线)，缺失的预报按 −1 计，总分是逐格加权平均。",
        "",
        f"被评分的程序是最后一次被接受、且本地总分比起始程序（Harness 的 `template/`）高至少 {REPLACE_MARGIN} 的提交，"
        "没有这样的提交时就是模板本身：本地检验期比隐藏期短得多，智能体在本地的小幅提升常常不能在隐藏期上成立。"
        "所以模板的好坏直接决定回合成绩的下限，智能体只在本地明显好过模板时才会改变它。"
        "回合摘要里写明了每个回合评的是哪个程序。",
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
        + interval_text + "采纳需要同时满足：",
        "",
        f"1. 各任务平均增益大于 {gates.min_delta}；",
        f"2. 区间下限大于 0（置信度按实际比较的候选数 K 做 Bonferroni 调整，为 1 − {1 - gates.confidence:.2g}/K："
        f"{len(names)} 个候选都通过检查时是 {confidence:.1%}，即单侧 {(1 - confidence) / 2:.2%}"
        + (f"；{rank_rule}" if rank_rule else "") + "）；",
        f"3. 没有任务的损失超过 {gates.task_cap} 且它自己的 {gates.confidence:.0%} 区间整体低于 0"
        "（两个版本在这个任务上都至少有 2 个回合时才判断：只有一个回合时区间不含回合间的随机差异）。",
        "",
        "通过的候选中增益最大的一个成为新的在任版本。在任版本在以后的每一代都会重新运行，它的成绩估计会越来越准。"
        "与在任版本或以往比较过的候选内容相同的候选不会运行。"
        "被采纳的版本最后还要在你看不到的留出任务和更晚的封存时期上各验证一次。",
        "",
        "## 你的产出",
        "",
        f"- 候选：{candidate_list}，每个都已是在任版本 `{incumbent.digest[:12]}` 的完整副本，直接修改，"
        "或先用 `reset_candidate` 换成某个存档版本；",
        "- 理由：写在工作区根目录的 `proposals/<名字>.md`（已有模板），不要放进候选目录——候选目录里的每个文件都会成为"
        "这个 Harness 的一部分，交给内循环智能体；",
        "- 用 `check_candidate` 检查每个候选。没有通过检查、与在任版本或以往比较过的版本相同的候选不会运行；"
        "两个候选的改动大部分相同时会有提醒。",
        "",
        "## 批量试模板（`variants/`）",
        "",
        f"每个候选要 {repeats} × {len(tasks)} 个回合、约一小时才知道结果，每代只有 {len(names)} 个；"
        "模板在本地检验期上跑一遍只要几秒。所以先在 `variants/` 里多试：把模板复制成几个目录"
        "（`cp -r candidates/a/template variants/<名字>`），各改一处，调用 `screen_templates`，一次在全部训练任务的"
        f"本地检验期上比较最多 {SCREEN_LIMIT} 个（每个会话最多 {SCREEN_TOTAL} 个），再把最好的复制进候选。"
        "可以用它扫一个参数的几个取值、比较几种结构，或检查两个改动叠在一起是否还有效。"
        "以多个任务的平均增益和更好的任务数为准：单个任务上的本地差别常常在隐藏期上不成立。"
        "`variants/` 不属于任何 Harness，不会交给内循环智能体。",
        "",
        "## 存档（`archive/`）",
        "",
        *_archive_rows(archived or {}),
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
        "上一代的比较结果，以及在任版本一路被采纳时的增益（采纳记录）；",
        "- `archive.md`：以往所有候选的改动、提案、结果和原因，以及评审员的笔记；",
        "- `lessons.md`：内循环智能体在各回合记下的发现、经验和无效方向；",
        "- `contrasts.md`：同一版本在同一任务上成绩最好和最差的回合、两者评分程序的差别；",
        "- `prior_runs.md`（如果有）：训练任务都在本次之内的以往运行里各候选的结果和评审员的笔记；",
        "- `expert.md`：领域专家的意见（如果有）；",
        "- `episodes/<回合>/summary.md` 和 `submission/`：每个回合的成绩、时间线、笔记、工具使用、错误、"
        "最终回复和被评分的代码；",
        "- `episodes/<回合>/transcript.md`：这个回合智能体的会话记录（推理、工具调用和结果，过长处保留首尾）。"
        "摘要只告诉你结果，会话记录告诉你智能体是怎么做的：时间花在哪里、在哪一步走偏、为什么本地分和隐藏期不一致。"
        "文件较大，先看摘要和 `contrasts.md`，再针对具体问题读或用 grep 查。",
        "",
        "## 预算",
        "",
        f"{minutes:g} 分钟，截止约 {ends}（机器休眠的时间不计，剩余时间以 `task_info` 为准）；"
        f"最多 {tokens:,} 个计费 token。"
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
    deadline = clock.now() + settings.minutes * 60  # awake time: a sleeping machine spends none of it
    gateway = make_gateway(trajectory, tokens=settings.tokens, deadline=deadline,
                           model=settings.model)  # an unusable model fails before anything is created
    for path in (work / "candidates", work / "proposals", work / "variants", root / "harness" / "skills", trajectory):
        path.mkdir(parents=True, exist_ok=True)
    (work / "AGENTS.md").write_text(resources.files("ecogym.evolution").joinpath("evolver.md").read_text("utf-8"),
                                    encoding="utf-8")
    fork(incumbent, work / "incumbent")
    with Ledger(ledger_path) as ledger:
        prior = related_runs(ledger, view)
    archived = archive_versions(view, incumbent, prior)
    for label, version in archived.items():
        fork(version, work / "archive" / label)
    if archived:
        (work / "archive" / "README.md").write_text(archive_readme(view, archived, prior, incumbent.digest),
                                                    encoding="utf-8")
    for candidate in names:
        fork(incumbent, work / "candidates" / candidate)
        (work / "proposals" / f"{candidate}.md").write_text(
            f"# 候选 {candidate}\n\n## 起点\n\n## 改了什么\n\n## 依据\n\n## 预期\n\n## 风险\n", encoding="utf-8")
    forbidden = forbidden_terms(tasks)
    with Ledger(ledger_path) as ledger:
        evidence = write_evidence(work / "evidence", ledger=ledger, view=view, tasks=[task.id for task in tasks],
                                  names=_names(view, incumbent), expert_text=expert_text,
                                  ensembles=ensemble_scores(view, benches.__getitem__), archived=archived, prior=prior)
    (work / "EVOLVE.md").write_text(brief(
        generation=generation, suite=suite, tasks=tasks, benches=benches, names=names, repeats=repeats,
        episode_minutes=episode_minutes, episode_tokens=episode_tokens, model=model, incumbent=incumbent,
        limits=limits, forbidden=forbidden, minutes=settings.minutes, tokens=settings.tokens, deadline=deadline,
        archived=archived,
    ), encoding="utf-8")
    events = Events(ledger_path, run_id, frozenset({"evolver_note", "expert_question"}))
    tools = EvolverTools(root=root, incumbent=incumbent, names=names, tag=tag, smoke=benches[tasks[0].id],
                         others=[benches[task.id] for task in tasks[1:]], runtime=runtime, limits=limits,
                         forbidden=forbidden, deadline=deadline, usage=gateway.usage, event=events,
                         generation=generation, cancel=cancel, pinned=model is not None, archived=archived,
                         known=known_versions(view, incumbent, prior))
    agent = AgentSpec(model=settings.model, reasoning_effort=settings.reasoning_effort, web_search=True,
                      subagent_depth=1, parallel_tool_calls=10)
    follow_up = evolver_follow_up(deadline=deadline, minutes=settings.minutes, widest=evidence.get("widest"))
    try:
        session = run_agent(root=root, prompt=PROMPT, agent=agent, tools=tools.catalog(), gateway=gateway,
                            deadline=deadline, trajectory=trajectory, runtime=runtime, cancel=cancel,
                            follow_up=follow_up)
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


__all__ = ["NAMES", "EvolverSettings", "EvolverTools", "archive_readme", "archive_versions", "brief", "copy_tree",
           "edit_lines", "edit_overlap", "evolver_follow_up", "forbidden_terms", "known_versions", "run_evolver"]
