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
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..episodes.history import fold_episode
from ..episodes.trajectory import submission_files
from ..kernel.ledger import Ledger

FINAL_REPORT_LIMIT = 6000
NOTE_LIMIT = 1200
NUMBER = re.compile(r"(?<!line )(?<![\w.])\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


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
             f"- 状态：{record['status']}；用时 {finished.get('seconds', '–')} 秒；token {finished.get('tokens', '–')}"
             f"（计费 {finished.get('metered_tokens', '–')}）；模型调用 {finished.get('llm_calls', '–')}"
             f"（失败 {finished.get('failed_calls', 0)}）",
             f"- 提交 {len(record['submissions'])} 次，本地检验 {len(record['evaluations'])} 次", ""]
    if scored:
        low, high = scored.get("interval") or [None, None]
        references = "、".join(f"{name} {_signed(value)}" for name, value in (scored.get("references") or {}).items())
        source = "自动快照（没有被接受的提交）" if scored.get("auto") else f"第 {scored.get('n')} 次提交"
        lines += [f"## 隐藏期成绩（model_selection，{scored.get('origins')} 个起点）", "",
                  f"总分 **{_signed(scored.get('score'), 4)}**，95% 区间 [{_signed(low)}, {_signed(high)}]，"
                  f"覆盖率 {scored.get('coverage', 0):.0%}，程序状态 {scored.get('status')}，评分程序：{source}。",
                  f"同一批起点上的参考（宿主内置方法，只用拟合期拟合）：{references}。", "",
                  "| 格 | 技能 | RMSE | 基线 RMSE |", "|---|---|---|---|"]
        for cell in scored.get("cells", []):
            lines.append(f"| {cell['cell']} | {_signed(cell['skill'])} | {cell['rmse']:.4g} | "
                         f"{cell['baseline_rmse']:.4g} |")
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
            accepted = "" if action == "检验" else ("，已接受" if item.get("accepted") else "，未接受")
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
    for round_ in view["rounds"]:
        for item in round_["episodes"]:
            record = records.get(item["episode"])
            scored = (record or {}).get("scored")
            if not scored or item.get("status") not in ("finished", "timeout"):
                continue
            scores.setdefault(item["harness"], {}).setdefault(item["task"], []).append(scored["score"])
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


def archive(view: dict[str, Any]) -> str:
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
            decision = candidate.get("decision")
            if decision:
                low, high = decision["interval"]
                per_task = "；".join(f"{item['task']} {_signed(item['delta'])}" for item in decision["tasks"])
                lines.append(f"- 结果：Δ {_signed(decision['delta'], 4)} [{_signed(low)}, {_signed(high)}]，"
                             f"{decision['reason']}；各任务 {per_task}")
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


def write_evidence(target: Path, *, ledger: Ledger, view: dict[str, Any], tasks: Sequence[str],
                   names: dict[str, str], expert_text: str = "",
                   ensembles: dict[str, dict[str, dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Write ``overview.md``, ``archive.md``, ``lessons.md``, ``expert.md`` and one folder per episode."""

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
    for episode, record in records.items():
        folder = episodes_dir / episode.replace(":", "-")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.md").write_text(episode_summary(record, role=roles[episode]), encoding="utf-8")
        trajectory = Path(((record.get("finished") or {}).get("trajectory") or {}).get("dir", ""))
        for relative, data in submission_files(trajectory, record.get("scored")).items():
            path = folder / "submission" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    (target / "overview.md").write_text(overview(view, records, tasks, names, ensembles), encoding="utf-8")
    (target / "archive.md").write_text(archive(view), encoding="utf-8")
    (target / "lessons.md").write_text(lessons(records.values(), names), encoding="utf-8")
    if expert_text.strip():
        (target / "expert.md").write_text(expert_text, encoding="utf-8")
    return {"episodes": len(records)}


__all__ = ["archive", "episode_summary", "lessons", "overview", "write_evidence"]
