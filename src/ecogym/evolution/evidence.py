"""The evidence an evolver reviews, written as files into its workspace.

Everything comes from the ledger and the stored trajectories of this run's
episodes on the training tasks: hidden-stage scores (model_selection of the
training tasks, which the inner agents never saw), per-cell skill, each
episode's timeline of local checks and submissions, its notes, expert
questions, tool use, errors, final report and the code it was scored with;
the record of every earlier candidate (changes, proposal, outcome) and the
evolver's own lessons. Holdout tasks and sealed stages never appear (the
loop passes only the training tasks' expert guidance). The hidden stage also
decides adoption, so the evidence leaves out what would point straight at
its values: the signed per-cell bias and the numbers in the predictor's
error messages.

Three files follow published findings on what a harness proposer needs:

- ``episodes/<episode>/transcript.md``: the agent's own sessions (reasoning,
  tool calls and results, cut where long). Meta-Harness (Lee et al., 2026)
  found raw execution traces to be the ingredient that makes harness search
  work, with score-only or summarized feedback far behind. The inner agents
  never see the hidden stage, so their sessions hold nothing of it.
- ``contrasts.md``: the best and the worst episode of one version on one task
  and the diff between their scored programs. Episodes of one version differ
  as much as candidates do; contrasting rollouts of the same prompt is the
  learning signal of Training-Free GRPO and ReasoningBank (2025).
- ``prior_runs.md``: earlier runs on the same training tasks, their
  candidates' outcomes and their evolvers' notes, so failed directions are
  not rediscovered run after run (MetaShare, Group-Evolving Agents, 2026).
  Runs on any other set of training tasks are left out: their evidence may
  cover this run's holdout tasks.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..episodes.export import sanitize, threads
from ..episodes.history import fold_episode
from ..episodes.trajectory import submission_files
from ..kernel.ledger import Ledger
from .fold import SCORED_STATUSES, fold

FINAL_REPORT_LIMIT = 6000
NOTE_LIMIT = 1200
NUMBER = re.compile(r"(?<!line )(?<![\w.])\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
TRANSCRIPT_LIMIT = 200_000  # characters of one episode's transcript
TURN_LIMITS = {"prompt": 4000, "reasoning": 1500, "reply": 3000, "call": 2500, "result": 1800}
FENCE = "~~~~~"
INJECTED = ("<system-reminder>", "Current runtime context")
CONTINUED = "这是同一个回合的第"  # how the host opens a follow-up session (``episodes.runner``)
CONTRAST_GAP = 0.02  # hidden-score gap below which two episodes of one version are not contrasted
CONTRAST_PAIRS = 24
CONTRAST_DIFF = 8000  # characters of program diff per pair
PRIOR_LIMIT = 40_000  # characters of prior_runs.md
RUN_STATUS = {"running": "运行中或已中断", "finished": "已完成", "failed": "失败", "stopped": "已停止"}


def _signed(value: Any, digits: int = 3) -> str:
    return f"{value:+.{digits}f}" if isinstance(value, (int, float)) and np.isfinite(value) else "–"


def _minutes(at: str, start: str) -> str:
    try:
        delta = datetime.fromisoformat(at) - datetime.fromisoformat(start)
    except ValueError:
        return "–"
    return f"{delta.total_seconds() / 60:.1f}"


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " …（截断）"


def _masked(text: str) -> str:
    """A hidden-stage error message without its numbers (they may be hidden observations); line numbers stay."""

    return NUMBER.sub("#", str(text))


def episode_summary(record: dict[str, Any], *, role: str) -> str:
    scored, finished = record.get("scored") or {}, record.get("finished") or {}
    harness = record.get("harness") or {}
    lines = [f"# 回合 {record['episode']}", "",
             f"- 任务：`{record['task_id']}`",
             f"- Harness：{harness.get('name', '')} `{str(harness.get('digest', ''))[:12]}`（{role}）",
             f"- 模型：{record.get('model')}；预算 {record.get('minutes')} 分钟、"
             f"{record.get('tokens') or 0:,} 计费 token",
             f"- 状态：{record['status']}；用时 {finished.get('seconds', '–')} 秒"
             + (f"（其中机器休眠 {asleep:.0f} 秒，不计入时限）" if (asleep := finished.get("asleep_seconds") or 0) >= 60
                else "") + f"；token {finished.get('tokens', '–')}"
             f"（计费 {finished.get('metered_tokens', '–')}）；模型调用 {finished.get('llm_calls', '–')}"
             f"（失败 {finished.get('failed_calls', 0)}）",
             f"- 提交 {len(record['submissions'])} 次，本地检验 {len(record['evaluations'])} 次"
             + (f"；起始程序（模板）本地 {_signed(start_local, 4)}"
                if isinstance(start_local := ((record.get("start") or {}).get("local") or {}).get("score"), float)
                else ""), ""]
    if scored:
        low, high = scored.get("interval") or [None, None]
        references = "、".join(f"{name} {_signed(value)}" for name, value in (scored.get("references") or {}).items())
        source = ("自动快照（没有被接受的提交）" if scored.get("auto") else
                  f"起始程序（没有被接受的提交在本地比它高 {record.get('replace_margin')} 以上）" if scored.get("start")
                  else f"第 {scored.get('n')} 次提交")
        lines += [f"## 隐藏期成绩（model_selection，{scored.get('origins')} 个起点）", "",
                  f"总分 **{_signed(scored.get('score'), 4)}**，95% 区间 [{_signed(low)}, {_signed(high)}]，"
                  f"覆盖率 {scored.get('coverage', 0):.0%}，程序状态 {scored.get('status')}，评分程序：{source}。",
                  f"同一批起点上的参考（宿主内置方法，只用拟合期拟合）：{references}。", "",
                  "| 格 | 技能 | RMSE | 基线 RMSE |", "|---|---|---|---|"]
        for cell in scored.get("cells", []):
            lines.append(f"| {cell['cell']} | {_signed(cell['skill'])} | {cell['rmse']:.4g} | "
                         f"{cell['baseline_rmse']:.4g} |")
        local = ((record.get("start") or {}).get("local") or {}) if scored.get("start") else next(
            (item.get("local") or {} for item in record["submissions"] if item.get("n") == scored.get("n")), {})
        if (scored.get("cells") and all(abs(cell["skill"]) < 1e-9 for cell in scored["cells"])
                and abs(local.get("score") or 0.0) > 1e-9):
            # 2026-11-01: a program at +0.105 on the local check forecast the baseline exactly in the hidden period
            lines += ["", f"注意：隐藏期每一格的预报都和朴素基线完全相同（技能恰好为 0），而同一程序在本地检验期上是 "
                          f"{_signed(local['score'], 4)}。它在隐藏期之前的全部数据上拟合时关掉了所有修正，"
                          "多半是启用门槛或回退条件在更长、更新的拟合数据上不再通过。"]
        if scored.get("errors"):
            lines += ["", "程序错误（节选，数值已隐去）：",
                      *[f"- `{_clip(_masked(error), 300)}`" for error in scored["errors"][:5]]]
        lines.append("")
    timeline = sorted([("检验", item) for item in record["evaluations"]] + [("提交", item) for item in
                                                                             record["submissions"]],
                      key=lambda pair: pair[1]["at"])
    if timeline:
        lines += ["## 时间线（本地检验期 calibration_uq）", "", "| 分钟 | 动作 | 本地总分 | 状态 | 说明 |",
                  "|---|---|---|---|---|"]
        for action, item in timeline:
            local = item.get("local") or {}
            accepted = "" if action == "检验" else (
                "，未接受" if not item.get("accepted") else
                "，已接受，不替换起始程序" if item.get("replaces_start") is False else "，已接受")
            note = _clip(item.get("note", ""), 200).replace("|", "/").replace("\n", " ")
            lines.append(f"| {_minutes(item['at'], record['started_at'])} | {action}{accepted} | "
                         f"{_signed(local.get('score'), 4)} | {local.get('status', '')} | {note} |")
        lines.append("")
    if record["notes"]:
        lines += ["## 智能体的笔记", ""]
        for note in record["notes"]:
            sources = f"（来源：{'；'.join(note.get('sources') or [])}）" if note.get("sources") else ""
            lines.append(f"- [{note.get('kind')}] {_clip(note.get('text', ''), NOTE_LIMIT)}{sources}")
        lines.append("")
    if record["questions"]:
        lines += ["## 给专家的问题", "", *[f"- {_clip(item.get('question', ''), 600)}" for item in record["questions"]],
                  ""]
    activity = finished.get("activity") or {}
    if activity:
        tools = "、".join(f"{name} {count}" for name, count in (activity.get("tool_calls") or {}).items())
        lines += ["## 工具使用", "", f"- 调用：{tools or '无'}", f"- 子智能体：{activity.get('subagents', 0)} 个"]
        queries = activity.get("web_queries") or []
        if queries:
            lines.append(f"- 网页搜索：{'；'.join(_clip(str(query), 120) for query in queries[:20])}")
        lines.append("")
    if finished.get("final_text"):
        lines += ["## 最终回复（智能体）", "", _clip(finished["final_text"], FINAL_REPORT_LIMIT), ""]
    return "\n".join(lines)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _cell_order(key: str) -> tuple[str, int]:
    target, _, horizon = key.rpartition("@")
    return (target, int(horizon)) if horizon.isdigit() else (key, 0)


def overview(view: dict[str, Any], records: dict[str, dict[str, Any]], tasks: Sequence[str],
             names: dict[str, str], ensembles: dict[str, dict[str, dict[str, Any]]] | None = None) -> str:
    """Scores of every version on every training task, its episodes' averaged forecasts (``ensembles``, see
    ``evolution.ensemble``) and the last round's comparisons."""

    scores: dict[str, dict[str, list[float]]] = {}
    cells: dict[str, dict[str, dict[str, list[float]]]] = {}
    kept: dict[str, list[int]] = {}  # harness -> [scored on the starting program, episodes that had one]
    for round_ in view["rounds"]:
        for item in round_["episodes"]:
            record = records.get(item["episode"])
            scored = (record or {}).get("scored")
            if not scored or item.get("status") not in ("finished", "timeout"):
                continue
            scores.setdefault(item["harness"], {}).setdefault(item["task"], []).append(scored["score"])
            if (record or {}).get("start"):
                counts = kept.setdefault(item["harness"], [0, 0])
                counts[0] += bool(scored.get("start"))
                counts[1] += 1
            for cell in scored["cells"]:
                cells.setdefault(item["harness"], {}).setdefault(item["task"], {}).setdefault(
                    cell["cell"], []).append(cell["skill"])
    incumbent = view["incumbent"]
    lines = ["# 证据总览", "",
             f"在任版本：{names.get(incumbent, '')} `{incumbent[:12]}`。"
             "表中是训练任务上隐藏期（model_selection）的总分，括号里是回合数；多个回合时给出均值和最小–最大。", "",
             "| 版本 | " + " | ".join(f"`{task}`" for task in tasks) + " | 平均 |",
             "|---|" + "---|" * (len(tasks) + 1)]
    for harness, by_task in scores.items():
        row, means = [], []
        for task in tasks:
            values = by_task.get(task, [])
            if not values:
                row.append("–")
                continue
            means.append(float(np.mean(values)))
            spread = f" [{_signed(min(values))}, {_signed(max(values))}]" if len(values) > 1 else ""
            row.append(f"{_signed(np.mean(values))} ({len(values)}){spread}")
        mark = "（在任）" if harness == incumbent else ""
        lines.append(f"| {names.get(harness, '')} `{harness[:12]}`{mark} | " + " | ".join(row)
                     + f" | {_signed(_mean(means))} |")
    if kept:
        lines += ["", "被评分的是最后一次被接受、且本地比起始程序（模板）高出足够多的提交，否则是模板本身。"
                      "按模板评分的回合：" + "；".join(f"{names.get(harness, '')} `{harness[:12]}` {done}/{total}"
                                                for harness, (done, total) in kept.items()) + "。"]
    if incumbent in cells:
        keys = sorted({key for by_cell in cells[incumbent].values() for key in by_cell}, key=_cell_order)
        header = "| 格 | " + " | ".join(f"`{task}`" for task in tasks) + " |"
        lines += ["", "## 在任版本的逐格技能（各回合均值）", "", header, "|---|" + "---|" * len(tasks)]
        for key in keys:
            values = [_mean(cells[incumbent].get(task, {}).get(key, [])) for task in tasks]
            lines.append(f"| {key} | " + " | ".join(_signed(value) for value in values) + " |")
    averaged = [harness for harness in scores if (ensembles or {}).get(harness)]
    if averaged:
        lines += ["", "## 平均各回合的预报", "",
                  "同一版本在同一任务上的几个回合各自写程序。把它们在隐藏期上的预报逐时刻平均后的总分，括号里是回合数和"
                  "这些回合各自的平均分；只用每个起点都有预报的回合，所以平均后高出的部分来自各回合误差中互相独立的部分。"
                  "采纳仍按单个回合比较。", "",
                  "| 版本 | " + " | ".join(f"`{task}`" for task in tasks) + " |", "|---|" + "---|" * len(tasks)]
        for harness in averaged:
            row = [f"{_signed(item['score'])}（{item['n']} 个，单个 {_signed(item['mean'])}"
                   + (f"，另有 {item['left_out']} 个缺预报未计入" if item.get("left_out") else "") + "）"
                   if (item := ensembles[harness].get(task)) else "–" for task in tasks]
            lines.append(f"| {names.get(harness, '')} `{harness[:12]}` | " + " | ".join(row) + " |")
    finished = [round_ for round_ in view["rounds"] if round_["finished"] and round_["generation"] > 0]
    if finished:
        last = finished[-1]
        lines += ["", f"## 第 {last['generation']} 代的比较（候选 − 在任，训练任务）", ""]
        for candidate in last["candidates"]:
            decision = candidate.get("decision")
            if not decision:
                problems = "；".join(candidate.get("problems", []))
                lines.append(f"- 候选 {candidate['name']}：{candidate['status']}；{problems}")
                continue
            per_task = "；".join(f"`{item['task']}` {_signed(item['delta'])}" for item in decision["tasks"])
            low, high = decision["interval"]
            lines.append(f"- 候选 {candidate['name']} `{candidate.get('digest', '')[:12]}`：{candidate['status']}，"
                         f"Δ {_signed(decision['delta'], 4)}，区间 [{_signed(low)}, {_signed(high)}]，"
                         f"{decision['reason']}。各任务：{per_task}")
    return "\n".join(lines) + "\n"


def _archived(archived: Mapping[str, Any] | None) -> dict[str, str]:
    """Label -> digest of the versions copied into the evolver's ``archive/`` (values: harnesses or digests)."""

    return {label: getattr(item, "digest", item) for label, item in (archived or {}).items()}


def archive(view: dict[str, Any], archived: Mapping[str, Any] | None = None) -> str:
    """Every earlier candidate of the run; ``archived`` maps the labels (``g1b``) whose full copy is in the
    evolver's ``archive/`` to their harnesses or digests."""

    archived = _archived(archived)
    lines = ["# 以往候选与教训", "",
             "每一代的候选、改动、结果和原因。没有被采纳的候选同样是证据：不要重复已证明无效的方向，"
             "但可以吸收其中在部分任务或格上有效的部分。", ""]
    for round_ in view["rounds"]:
        if round_["generation"] == 0:
            continue
        evolver = round_.get("evolver") or {}
        lines += [f"## 第 {round_['generation']} 代（在任 `{round_['incumbent'][:12]}`）", ""]
        if evolver.get("status") and evolver["status"] not in ("finished", "timeout"):
            lines += [f"评审会话状态：{evolver['status']} {evolver.get('error', '')}", ""]
        for candidate in round_["candidates"]:
            lines.append(f"### 候选 {candidate['name']} `{candidate.get('digest', '')[:12]}` —— {candidate['status']}")
            if candidate.get("summary"):
                lines.append(f"- 摘要：{_clip(candidate['summary'], 600)}")
            changes = candidate.get("changes") or []
            if changes:
                listed = (f"{item['path']}（{item['status']}，+{item['added']} −{item['removed']}）"
                          for item in changes[:20])
                lines.append("- 改动：" + "，".join(listed))
            if candidate.get("problems"):
                lines.append(f"- 问题：{'；'.join(candidate['problems'][:5])}")
            if candidate.get("warnings"):
                lines.append(f"- 宿主检查的提醒：{'；'.join(candidate['warnings'][:5])}")
            smoke = candidate.get("smoke") or {}
            if smoke.get("tasks"):
                listed = (f"{task} {_signed(item['score']) if item.get('status') == 'ok' else item.get('status')}"
                          + (f"（在任版本的模板 {_signed(item['incumbent_score'])}）"
                             if isinstance(item.get("incumbent_score"), (int, float)) else "")
                          for task, item in smoke["tasks"].items())
                lines.append(f"- 模板试运行（本地检验期）：{'；'.join(listed)}")
            elif isinstance(smoke.get("score"), (int, float)):
                reference = smoke.get("incumbent_score")
                versus = f"，在任版本的模板 {_signed(reference)}" if isinstance(reference, (int, float)) else ""
                lines.append(f"- 模板试运行（本地检验期）：{_signed(smoke['score'])}{versus}")
            decision = candidate.get("decision")
            if decision:
                low, high = decision["interval"]
                per_task = "；".join(f"{item['task']} {_signed(item['delta'])}" for item in decision["tasks"])
                better = sum(1 for item in decision["tasks"] if (item.get("delta") or 0) > 0)
                lines.append(f"- 结果：Δ {_signed(decision['delta'], 4)} [{_signed(low)}, {_signed(high)}]，"
                             f"{decision['reason']}；各任务 {per_task}"
                             f"（{better}/{len(decision['tasks'])} 个任务好于当时的在任版本）")
                label = f"g{round_['generation']}{candidate['name']}"
                if candidate.get("digest") and archived.get(label) == candidate["digest"]:
                    lines.append(f"- 完整内容在工作区的 `archive/{label}/`，"
                                 "可以作为新候选的起点（`reset_candidate`）。")
                gains = decision.get("cells") or {}
                if gains:
                    ranked = sorted(gains.items(), key=lambda item: item[1])
                    worst = "，".join(f"{key} {_signed(value)}" for key, value in ranked[:3])
                    best = "，".join(f"{key} {_signed(value)}" for key, value in ranked[-3:])
                    lines.append(f"- 逐格平均变化：最差 {worst}；最好 {best}")
            if candidate.get("proposal"):
                lines += ["", "提案（节选）：", "", "> " + _clip(candidate["proposal"], 1500).replace("\n", "\n> ")]
            lines.append("")
        if round_["notes"]:
            lines += ["评审员的笔记：", *[f"- [{note.get('kind')}] {_clip(note.get('text', ''), NOTE_LIMIT)}"
                                    for note in round_["notes"]], ""]
    return "\n".join(lines) + "\n"


def lessons(records: Iterable[dict[str, Any]], names: dict[str, str]) -> str:
    groups: dict[str, list[str]] = {"lesson": [], "dead_end": [], "finding": []}
    seen: set[str] = set()
    for record in records:
        harness = (record.get("harness") or {}).get("digest", "")
        for note in record["notes"]:
            text = _clip(note.get("text", ""), NOTE_LIMIT)
            if not text or text in seen:
                continue
            seen.add(text)
            tag = f"（`{record['task_id']}`，{names.get(harness, '')} `{harness[:12]}`，{record['episode']}）"
            groups.setdefault(note.get("kind", "finding"), []).append(f"- {text} {tag}")
    titles = {"lesson": "经验（lesson）", "dead_end": "无效方向（dead_end）", "finding": "发现（finding）"}
    lines = ["# 内循环智能体的笔记", "", "智能体在回合中用 note 记下的内容，按类型汇总、去重。", ""]
    for kind, items in groups.items():
        if items:
            lines += [f"## {titles.get(kind, kind)}", "", *items, ""]
    return "\n".join(lines)


def _cut(text: Any, limit: int) -> str:
    """Head and tail of a long text (what an agent read first and what it ended on)."""

    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]}\n…（中间略去 {len(text) - limit} 字）…\n{text[len(text) - (limit - head):]}"


def _block(text: str) -> list[str]:
    return [FENCE, text.replace(FENCE, "~~~~"), FENCE, ""]


def _text(content: Any) -> str:
    """A message's content: a string, or a list of parts."""

    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def _arguments(function: dict[str, Any]) -> tuple[str, str]:
    """A tool call's name and its arguments, long strings (code, commands, file contents) written out."""

    name, raw = str(function.get("name") or "?"), function.get("arguments")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return name, str(raw)
    if not isinstance(arguments, dict):
        return name, json.dumps(arguments, ensure_ascii=False)
    lines = []
    for key, value in arguments.items():
        if isinstance(value, str) and ("\n" in value or len(value) > 120):
            lines.append(f"[{key}]\n{value}")
        else:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    return name, "\n".join(lines)


def _injected(text: str) -> str:
    """One line for context DSH injects as a user message (AGENTS.md, the skill list, runtime facts, reminders)."""

    if "Instructions from: AGENTS.md" in text:
        what = "AGENTS.md（Harness 的指令和经验）"
    elif "available_skills" in text:
        what = "技能列表"
    elif text.startswith("Current runtime context"):
        what = "运行环境说明"
    else:
        what = "提醒：" + _clip(text.replace("<system-reminder>", "").replace("</system-reminder>", "").strip(),
                             160).replace("\n", " ")
    return f"（DSH 注入的上下文，{len(text)} 字，略去：{what}）"


def _thread_label(index: int, messages: list[dict[str, Any]]) -> str:
    first = next((_text(item.get("content")) for item in messages if item.get("role") == "user"), "")
    if first.startswith(CONTINUED):
        return "续跑会话"
    return "主会话" if index == 0 else "其他线程（子智能体，或上下文压缩后的续写）"


def transcript(record: dict[str, Any], *, role: str) -> str:
    """The agent's sessions in one episode, from its recorded model calls ("" when none are stored)."""

    directory = Path(((record.get("finished") or {}).get("trajectory") or {}).get("dir", ""))
    path = directory / "llm_calls.jsonl.gz"
    if not directory.name or not path.is_file():
        return ""
    try:
        found = threads(path)
    except (OSError, EOFError, ValueError):  # a damaged log costs this file only
        return ""
    if not found:
        return ""
    lines = [f"# 回合 {record['episode']} 的会话记录", "",
             f"- 任务 `{record['task_id']}`；{role}；模型 {record.get('model')}；{len(found)} 个线程。",
             "- 节选：系统提示和 DSH 注入的上下文只留一行说明；推理、回复、工具参数和结果过长时保留首尾。"
             "会话目录写成 `/workspace`。",
             "- 智能体在会话中看不到隐藏期；隐藏期成绩见同目录的 `summary.md`。", ""]
    for index, chain in enumerate(found):
        messages = sanitize(chain["messages"], [record.get("root") or ""])
        label = _thread_label(index, messages)
        lines += [f"## 线程 {index + 1}/{len(found)}：{label}（{chain['calls']} 次模型调用）", ""]
        called: dict[str, str] = {}
        step = 0
        for message in messages:
            kind, text = message.get("role"), _text(message.get("content"))
            if kind == "system":
                lines += [f"（系统提示 {len(text)} 字，略去）", ""]
            elif kind == "user" and text.lstrip().startswith(INJECTED):
                lines += [_injected(text.strip()), ""]
            elif kind == "user":
                lines += ["### 宿主或上级智能体的消息", "", *_block(_cut(text, TURN_LIMITS["prompt"]))]
            elif kind == "assistant":
                step += 1
                lines += [f"### 第 {step} 步", ""]
                if message.get("reasoning_content"):
                    lines += ["推理：", "", *_block(_cut(message["reasoning_content"], TURN_LIMITS["reasoning"]))]
                if text.strip():
                    lines += ["回复：", "", *_block(_cut(text, TURN_LIMITS["reply"]))]
                for call in message.get("tool_calls") or []:
                    name, arguments = _arguments(call.get("function") or {})
                    called[str(call.get("id"))] = name
                    lines += [f"调用 `{name}`：", "", *_block(_cut(arguments, TURN_LIMITS["call"]))]
            elif kind == "tool":
                name = called.get(str(message.get("tool_call_id")), message.get("name") or "工具")
                lines += [f"结果（`{name}`）：", "", *_block(_cut(text, TURN_LIMITS["result"]))]
    text = "\n".join(lines)
    if len(text) > TRANSCRIPT_LIMIT:
        text = text[:TRANSCRIPT_LIMIT] + "\n\n…（记录过长，其余部分略去）\n"
    return text


def _scored_local(record: dict[str, Any]) -> float | None:
    """The local-check score of the program an episode was scored with."""

    scored = record.get("scored") or {}
    if scored.get("start"):
        return ((record.get("start") or {}).get("local") or {}).get("score")
    found = next((item for item in record["submissions"] if item.get("n") == scored.get("n")), None)
    return ((found or {}).get("local") or {}).get("score")


def _scored_source(record: dict[str, Any]) -> str:
    scored = record.get("scored") or {}
    return ("自动快照" if scored.get("auto") else "起始程序（模板）" if scored.get("start")
            else f"第 {scored.get('n')} 次提交")


def program_diff(before: dict[str, bytes], after: dict[str, bytes], limit: int = CONTRAST_DIFF) -> str:
    """A unified diff from one scored program to another (file name -> bytes, as ``submission_files`` reads)."""

    chunks = []
    for relative in sorted(before.keys() | after.keys()):
        old, new = before.get(relative), after.get(relative)
        if old == new:
            continue
        if b"\x00" in (old or b"") or b"\x00" in (new or b""):
            where = "只在好回合中有" if old is None else "只在差回合中有" if new is None else "不同"
            chunks.append(f"（{relative}：二进制文件，{where}）")
            continue
        chunks.append("\n".join(difflib.unified_diff(
            (old or b"").decode("utf-8", "replace").splitlines(), (new or b"").decode("utf-8", "replace").splitlines(),
            f"差/{relative}", f"好/{relative}", n=2, lineterm="")))
    return _clip("\n".join(chunks), limit)


def contrast_pairs(view: dict[str, Any], records: dict[str, dict[str, Any]]) -> list[tuple[float, str, str,
                                                                                         list[dict[str, Any]]]]:
    """(gap, harness, task, episodes from worst to best) for every version and task whose best and worst scored
    episodes differ by at least ``CONTRAST_GAP``, the widest gap first."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for round_ in view["rounds"]:
        for item in round_["episodes"]:
            record = records.get(item.get("episode") or "")
            if (record is not None and item.get("status") in SCORED_STATUSES
                    and isinstance((record.get("scored") or {}).get("score"), float)):
                groups.setdefault((item["harness"], item["task"]), []).append(record)
    pairs = []
    for (harness, task), members in groups.items():
        ordered = sorted(members, key=lambda record: record["scored"]["score"])
        gap = ordered[-1]["scored"]["score"] - ordered[0]["scored"]["score"] if len(ordered) > 1 else 0.0
        if gap >= CONTRAST_GAP:
            pairs.append((gap, harness, task, ordered))
    pairs.sort(key=lambda item: -item[0])
    return pairs


def contrasts(view: dict[str, Any], records: dict[str, dict[str, Any]], names: dict[str, str],
              programs: dict[str, dict[str, bytes]]) -> str:
    """The best and the worst episode of one version on one task, for every such pair far enough apart."""

    pairs = contrast_pairs(view, records)
    lines = ["# 同一版本、同一任务的回合对照", "",
             "同一版 Harness 在同一个任务上的几个回合，提示完全相同，成绩却常常相差很大："
             "这个差别就是比较里回合间的噪声，也说明同一份说明下智能体有时做对、有时做错。"
             "下面每一组列出成绩最好和最差的回合、它们评分程序的差别"
             f"（差 → 好）和会话记录的位置；只列相差至少 {CONTRAST_GAP} 的组，按差距从大到小。"
             "可以对照两份会话记录，找出好回合做对了什么、差回合在哪一步走偏，"
             "再判断能否用说明、技能或模板让好的做法成为常态。",
             ""]
    if not pairs:
        lines.append(f"（目前没有相差 {CONTRAST_GAP} 以上的回合。）")
    for gap, harness, task, ordered in pairs[:CONTRAST_PAIRS]:
        best, worst = ordered[-1], ordered[0]
        lines += [f"## `{task}` · {names.get(harness, '')} `{harness[:12]}`：相差 {_signed(gap)}", "",
                  f"这个版本在这个任务上共 {len(ordered)} 个回合，隐藏期 "
                  + "、".join(_signed(record["scored"]["score"]) for record in reversed(ordered)) + "。", "",
                  "| | 回合 | 隐藏期 | 评分程序 | 评分程序的本地分 | 本地检验 / 提交 | 用时（秒） | token |",
                  "|---|---|---|---|---|---|---|---|"]
        for label, record in (("好", best), ("差", worst)):
            finished = record.get("finished") or {}
            lines.append(f"| {label} | `{record['episode']}` | {_signed(record['scored']['score'], 4)} | "
                         f"{_scored_source(record)} | {_signed(_scored_local(record), 4)} | "
                         f"{len(record['evaluations'])} / {len(record['submissions'])} | "
                         f"{finished.get('seconds', '–')} | {finished.get('tokens') or 0:,} |")
        patch = program_diff(programs.get(worst["episode"], {}), programs.get(best["episode"], {}))
        lines += ["", "评分程序的差别（差 → 好）：", ""]
        lines += [FENCE + "diff", patch.replace(FENCE, "~~~~"), FENCE, ""] if patch else [
            "（两个回合评分的程序相同，或其中一个没有存下程序。）", ""]
        lines += ["会话记录：" + "、".join(f"`episodes/{record['episode'].replace(':', '-')}/transcript.md`"
                                     for record in (best, worst)), ""]
    return "\n".join(lines) + "\n"


def same_task_runs(ledger: Ledger, view: dict[str, Any]) -> list[dict[str, Any]]:
    """Every other evolution run on exactly this run's training tasks, folded, newest first.

    A run on any other set of training tasks is left out: its evidence and its candidates were shaped by hidden
    periods of tasks that may be this run's holdout tasks."""

    train = sorted((view.get("suite") or {}).get("train") or [])
    found = []
    for run_id in reversed(ledger.runs(first="evolve_started")):
        if not train or run_id == view["run_id"]:
            continue
        try:
            other = fold(ledger.events(run_id))
        except (ValueError, KeyError):
            continue
        if sorted((other.get("suite") or {}).get("train") or []) == train:
            found.append(other)
    return found


def version_label(run_id: str, generation: int, name: str, *, current: str) -> str:
    """How the evolver's workspace names a candidate: ``g2b`` in this run, ``35ad33-g1b`` in an earlier one."""

    own = f"g{generation}{name}"
    return own if run_id == current else f"{run_id.partition(':')[2][:6]}-{own}"


def prior_runs(ledger: Ledger, view: dict[str, Any], runs: Sequence[dict[str, Any]] | None = None,
               archived: Mapping[str, Any] | None = None) -> str:
    """Earlier runs on exactly this run's training tasks (``same_task_runs``): their candidates' outcomes and
    their evolvers' notes; ``archived`` says which candidates the evolver's ``archive/`` holds (see ``archive``)."""

    archived = _archived(archived)
    blocks = []
    for other in same_task_runs(ledger, view) if runs is None else runs:
        run_id = other["run_id"]
        rounds = [round_ for round_ in other["rounds"] if round_["generation"] > 0]
        if not rounds and not other["notes"]:
            continue
        seed = other.get("seed") or {}
        status = RUN_STATUS.get(other["status"], other["status"])
        lines = [f"## `{run_id}`（{str(other.get('started_at', ''))[:10]}，起点 {seed.get('name', '')} "
                 f"`{str(seed.get('digest', ''))[:12]}`，{status}，采纳 {len(other['adoptions'])} 次）", ""]
        for round_ in rounds:
            for candidate in round_["candidates"]:
                decision = candidate.get("decision") or {}
                result = (f"Δ {_signed(decision['delta'], 4)}，{decision.get('reason')}"
                          if isinstance(decision.get("delta"), (int, float)) else candidate.get("status", ""))
                adopted = "（采纳）" if candidate.get("status") == "adopted" else ""
                label = version_label(run_id, round_["generation"], candidate["name"], current=view["run_id"])
                stored = candidate.get("digest") and archived.get(label) == candidate["digest"]
                copy = f"完整内容在 `archive/{label}/`。" if stored else ""
                lines.append(f"- 第 {round_['generation']} 代候选 {candidate['name']} "
                             f"`{str(candidate.get('digest') or '')[:12]}`：{result}{adopted}。"
                             f"{_clip(candidate.get('summary', ''), 300)}{copy}")
        if other["notes"]:
            lines += ["", "评审员的笔记：", *[f"- [{note.get('kind')}] {_clip(note.get('text', ''), 600)}"
                                         for note in other["notes"]]]
        blocks.append("\n".join(lines))
    if not blocks:
        return ""
    head = ["# 以往运行（训练任务完全相同）的候选与评审笔记", "",
            "同一组训练任务上的其他进化运行，新的在前：每个候选的结果和一句话摘要，以及那些运行的评审员记下的经验和无效方向。"
            "它们的在任版本可能和本次不同，增益是相对当时的在任版本算的。用来避免重复已经失败的方向，"
            "也可以从其中的候选出发（在 `archive/` 里的那些）；在这些运行里比较过的版本原样提交不会再运行。"
            "训练任务不同的运行不在这里（它们的证据可能涉及本次的留出任务）。", ""]
    return _clip("\n".join(head) + "\n" + "\n\n".join(blocks), PRIOR_LIMIT) + "\n"


def write_evidence(target: Path, *, ledger: Ledger, view: dict[str, Any], tasks: Sequence[str],
                   names: dict[str, str], expert_text: str = "",
                   ensembles: dict[str, dict[str, dict[str, Any]]] | None = None,
                   archived: Mapping[str, Any] | None = None,
                   prior: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Write ``overview.md``, ``archive.md``, ``lessons.md``, ``contrasts.md``, ``expert.md``, ``prior_runs.md``
    and one folder per episode (``summary.md``, ``transcript.md``, the scored ``submission/``); ``archived``
    maps the labels of the versions copied into the evolver's ``archive/`` to their harnesses, ``prior`` lists
    the runs on the same training tasks (``same_task_runs``, read from the ledger when None)."""

    archived = _archived(archived)

    target.mkdir(parents=True, exist_ok=True)
    roles = {}
    for round_ in view["rounds"]:
        for item in round_["episodes"]:
            if not item.get("episode"):
                continue  # the episode raised before it could start; nothing was recorded
            role = "在任" if item["role"] == "incumbent" else f"候选 {item.get('candidate')}"
            roles[item["episode"]] = f"第 {round_['generation']} 代，{role}"
    records = {episode: fold_episode(ledger.events(episode)) for episode in roles}
    episodes_dir = target / "episodes"
    programs: dict[str, dict[str, bytes]] = {}
    transcripts = 0
    for episode, record in records.items():
        folder = episodes_dir / episode.replace(":", "-")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.md").write_text(episode_summary(record, role=roles[episode]), encoding="utf-8")
        trajectory = Path(((record.get("finished") or {}).get("trajectory") or {}).get("dir", ""))
        programs[episode] = submission_files(trajectory, record.get("scored"))
        for relative, data in programs[episode].items():
            path = folder / "submission" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if text := transcript(record, role=roles[episode]):
            (folder / "transcript.md").write_text(text, encoding="utf-8")
            transcripts += 1
    (target / "overview.md").write_text(overview(view, records, tasks, names, ensembles), encoding="utf-8")
    (target / "archive.md").write_text(archive(view, archived), encoding="utf-8")
    (target / "lessons.md").write_text(lessons(records.values(), names), encoding="utf-8")
    (target / "contrasts.md").write_text(contrasts(view, records, names, programs), encoding="utf-8")
    if expert_text.strip():
        (target / "expert.md").write_text(expert_text, encoding="utf-8")
    if earlier := prior_runs(ledger, view, prior, archived):
        (target / "prior_runs.md").write_text(earlier, encoding="utf-8")
    widest = None  # the widest contrast whose two sessions are both on file (best first): a follow-up reads them
    for _, _, _, ordered in contrast_pairs(view, records):
        files = [f"episodes/{record['episode'].replace(':', '-')}/transcript.md"
                 for record in (ordered[-1], ordered[0])]
        if all((target / name).is_file() for name in files):
            widest = files
            break
    return {"episodes": len(records), "transcripts": transcripts, "prior_runs": bool(earlier), "widest": widest}


__all__ = ["archive", "contrast_pairs", "contrasts", "episode_summary", "lessons", "overview", "prior_runs",
           "program_diff", "same_task_runs", "transcript", "version_label", "write_evidence"]
