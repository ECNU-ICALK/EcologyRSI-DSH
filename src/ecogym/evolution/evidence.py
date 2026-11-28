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
its values: the signed per-cell bias and the text of the predictor's errors
on the hidden stage. Those messages are written by the program while it reads
hidden rows, so no filter can prove they carry none (a masked number survives
as "line 23.5" or in base64): the evidence names only the error types from a
fixed list and how often they occurred; the messages stay in the ledger for
people.

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
- ``prior_runs.md``: earlier runs whose training tasks are all among this
  run's (the same set, or part of it), their candidates' outcomes and their
  evolvers' notes, so failed directions are not rediscovered run after run
  (MetaShare, Group-Evolving Agents, 2026). A run with any other training
  task is left out: its evidence may cover this run's holdout tasks.
"""

from __future__ import annotations

import builtins
import collections
import difflib
import gzip
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..episodes.history import fold_episode
from ..episodes.tools import REPLACE_MARGIN, SINGLE_WINDOW_SELECTION, selection_rule_text
from ..episodes.trajectory import submission_files
from ..episodes.transcripts import session_threads
from ..kernel.ledger import Ledger
from .fold import SCORED_STATUSES, fold

FINAL_REPORT_LIMIT = 6000
NOTE_LIMIT = 1200
# the error types the evidence may name: Python's own exceptions and numpy's linear-algebra error; any other name
# (a program may define a class called anything) reads as "其他错误"
KNOWN_ERRORS = frozenset(name for name, value in vars(builtins).items()
                         if isinstance(value, type) and issubclass(value, BaseException)) | {"LinAlgError"}
ERROR_TYPE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?::|$)")
TRANSCRIPT_LIMIT = 200_000  # characters of one episode's transcript
TURN_LIMITS = {"prompt": 4000, "reasoning": 1500, "reply": 3000, "call": 2500, "result": 1800}
FENCE = "~~~~~"
CONTRAST_GAP = 0.02  # hidden-score gap below which two episodes of one version are not contrasted
CONTRAST_PAIRS = 24
CONTRAST_DIFF = 8000  # characters of program diff per pair
PRIOR_LIMIT = 40_000  # characters of prior_runs.md
OVERFIT_GAP = 0.04  # local minus hidden score above which a replacing submission is flagged as overfit
UNSUBMITTED_GAIN = 0.02  # local eval gain over start that was never submitted
HDA_COST_RATIO_WARN = 1.25  # candidate token ratio above which small gains are flagged as inefficient
HDA_MIN_GAIN = 0.015
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


def error_kinds(errors: Sequence[Any]) -> str:
    """Hidden-stage errors as types from ``KNOWN_ERRORS`` and counts: never their messages, which the program wrote
    while it read hidden rows."""

    counts: collections.Counter[str] = collections.Counter()
    for error in errors:
        found = ERROR_TYPE.match(str(error).strip())
        name = found.group(1) if found else ""
        counts[name if name in KNOWN_ERRORS else "其他错误"] += 1
    return "、".join(f"{name} ×{count}" for name, count in counts.most_common())


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
             f"（失败 {finished.get('failed_calls', 0)}"
             + (f"，复读熔断 ×{aborts}" if (aborts := finished.get("degenerate_aborts") or 0) else "") + "）",
             f"- 提交 {len(record['submissions'])} 次，本地检验 {len(record['evaluations'])} 次"
             + (f"；起始程序（模板）本地 {_signed(start_local, 4)}"
                if isinstance(start_local := ((record.get("start") or {}).get("local") or {}).get("score"), float)
                else "")
             + ("；触发预算救援" if finished.get("rescue_triggered") else "")
             + ("；触发审计钩子" if finished.get("hook_fired") else ""), ""]
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
            first = "（只记了前 20 条）" if len(scored["errors"]) >= 20 else ""
            lines += ["", f"程序在隐藏期上报错{first}：{error_kinds(scored['errors'])}。这里只给错误类型和次数："
                          "错误信息是程序读着隐藏期的数据写的，可能带有隐藏期的数值，所以不给出。"
                          "要看具体信息，在本地检验期上运行同一个程序（check_candidate、screen_templates）。"]
        lines.append("")
    timeline = sorted([("检验", item) for item in record["evaluations"]] + [("提交", item) for item in
                                                                             record["submissions"]],
                      key=lambda pair: pair[1]["at"])
    if timeline:
        lines += ["## 时间线（本地检验期 calibration_uq）", "", "| 分钟 | 动作 | 本地总分 | 状态 | 说明 |",
                  "|---|---|---|---|---|"]
        for action, item in timeline:
            local = item.get("local") or {}
            sel = item.get("local_selection") or {}
            accepted = "" if action == "检验" else (
                "，未接受" if not item.get("accepted") else
                "，已接受，不替换起始程序" if item.get("replaces_start") is False else "，已接受")
            pre_d = sel.get("pre_delta")
            pre_tag = f"（早折 {_signed(pre_d, 4)}）" if isinstance(pre_d, (int, float)) else ""
            reused = " [复用缓存]" if sel.get("reused_local") else ""
            note = _clip(item.get("note", ""), 200).replace("|", "/").replace("\n", " ")
            lines.append(f"| {_minutes(item['at'], record['started_at'])} | {action}{accepted}{reused} | "
                         f"{_signed(local.get('score'), 4)}{pre_tag} | {local.get('status', '')} | {note} |")
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


REPLACEMENT_LINES = 12  # episodes whose submission replaced the template, listed worst first


def _attribution_rows(attribution: dict[str, Any], round_: dict[str, Any], records: dict[str, dict[str, Any]],
                      names: dict[str, str]) -> list[str]:
    """Per version and task: the fixed template's score, how far the episodes ended above it, how many replicates
    scored and how many kept the template; then every episode whose submission replaced the template, its local
    gain against the template next to its hidden-stage one. The baseline round has no comparison, so without these
    rows its diagnostic read only "matched 5/8" (2026-11-18 review), and the evolver never saw that the three
    submissions that replaced the template were each worse on the hidden stage."""

    arms = attribution.get("arms") or {}
    if not arms:
        return []
    rows = [row for arm in arms.values() for row in arm["tasks"].values()]
    expected = attribution.get("expected_episodes")
    planned = f"/{len(rows) * expected}" if isinstance(expected, int) else ""
    lines = ["", f"有成绩的重复回合 {sum(row.get('episode_count', 0) for row in rows)}{planned}"
                 "（服务故障或中断的回合不算）。",
             "", "| 版本 | 任务 | 固定模板 | 回合均分 − 模板 | 有成绩的重复 | 其中按模板评分 |",
             "|---|---|---|---|---|---|"]
    templates: dict[tuple[str, str], float] = {}
    for label, arm in arms.items():
        version = f"{label} `{arm.get('harness', '')[:12]}`" if label != "incumbent" else "在任"
        for task, row in arm["tasks"].items():
            reference = row.get("template") or {}
            if reference.get("status") == "ok" and isinstance(reference.get("score"), (int, float)):
                templates[(label, task)] = reference["score"]
            template = _signed(reference.get("score"), 4) if reference.get("status") == "ok" else \
                f"–（{reference.get('status', '无')}）"
            gain = _signed(row.get("episode_over_template"), 4) if row.get("episode_over_template") is not None \
                else "–"
            lines.append(f"| {version} | `{task}` | {template} | {gain} | {row.get('episode_count', 0)}/"
                         f"{expected if isinstance(expected, int) else '?'} | {row.get('scored_start_count', 0)} |")
    replaced = []
    for entry in round_.get("episodes") or []:
        record = records.get(entry.get("episode") or "") or {}
        scored = record.get("scored") or {}
        label = "incumbent" if entry.get("role") == "incumbent" else entry.get("candidate")
        template = templates.get((label, entry.get("task")))
        if (entry.get("status") not in SCORED_STATUSES or scored.get("start") or scored.get("auto")
                or type(scored.get("n")) is not int or template is None
                or not isinstance(scored.get("score"), (int, float))):
            continue
        submission = next((item for item in record.get("submissions") or [] if item.get("n") == scored["n"]), {})
        local = (submission.get("local_selection") or {}).get("delta")
        if not isinstance(local, (int, float)):
            start = ((record.get("start") or {}).get("local") or {}).get("score")
            score = (submission.get("local") or {}).get("score")
            local = score - start if isinstance(score, (int, float)) and isinstance(start, (int, float)) else None
        replaced.append((scored["score"] - template, entry, scored["n"], local))
    if replaced:
        replaced.sort(key=lambda item: item[0])
        worse = sum(1 for hidden, *_ in replaced if hidden < 0)
        lines += ["", f"提交替换了模板的回合 {len(replaced)} 个：隐藏期上比模板好 {len(replaced) - worse} 个、"
                      f"差 {worse} 个，平均 {_signed(_mean([hidden for hidden, *_ in replaced]), 4)}。从最差的开始列出"
                      "（本地增益是这次提交在本地检验期上相对模板的增益）："]
        for hidden, entry, number, local in replaced[:REPLACEMENT_LINES]:
            version = "在任" if entry.get("role") == "incumbent" else f"候选 {entry.get('candidate')}"
            lines.append(f"- {version} `{entry.get('task')}` 重复 {entry.get('replicate', 0)}"
                         f"（`{entry.get('episode')}`）：第 {number} 次提交，本地 {_signed(local, 4)} → "
                         f"隐藏期 {_signed(hidden, 4)}")
        if len(replaced) > REPLACEMENT_LINES:
            lines.append(f"- 另有 {len(replaced) - REPLACEMENT_LINES} 个，见各回合的 summary.md。")
    return lines


def _round_all_episodes(round_: Mapping[str, Any]) -> list[dict[str, Any]]:
    """All scored/recorded episodes of a round, including its intra-generation relay stages (deduplicated)."""

    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for stage in round_.get("relay_stages") or ():
        for entry in stage.get("episodes") or ():
            ep = entry.get("episode")
            if ep and ep in seen:
                continue
            if ep:
                seen.add(ep)
            items.append(entry)
    for entry in round_.get("episodes") or ():
        ep = entry.get("episode")
        if ep and ep in seen:
            continue
        if ep:
            seen.add(ep)
        items.append(entry)
    return items


def overview(view: dict[str, Any], records: dict[str, dict[str, Any]], tasks: Sequence[str],
             names: dict[str, str], ensembles: dict[str, dict[str, dict[str, Any]]] | None = None) -> str:
    """Scores of every version on every training task, its episodes' averaged forecasts (``ensembles``, see
    ``evolution.ensemble``) and the last round's comparisons."""

    scores: dict[str, dict[str, list[float]]] = {}
    cells: dict[str, dict[str, dict[str, list[float]]]] = {}
    kept: dict[str, list[int]] = {}  # harness -> [scored on the starting program, episodes that had one]
    for round_ in view["rounds"]:
        for item in _round_all_episodes(round_):
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
        # the rule the run's episodes were scored by, as the agents read it (a run before 0.4.0: the latest
        # passing submission); 0.4.0 wrote "the latest" here for every run, while best_local/1 picks the highest
        policy = (view.get("config") or {}).get("local_selection") or SINGLE_WINDOW_SELECTION
        lines += ["", f"回合评分用哪个程序（`{policy}`）：{selection_rule_text(policy)}"
                      "按模板评分的回合：" + "；".join(f"{names.get(harness, '')} `{harness[:12]}` {done}/{total}"
                                                for harness, (done, total) in kept.items()) + "。"]
    measured = [round_ for round_ in view["rounds"] if round_["finished"] and round_.get("template_attribution")]
    if measured:
        round_ = measured[-1]
        attribution = round_["template_attribution"]
        lines += ["", f"## 第 {round_['generation']} 代的起始模板与提交表现", "",
                  "固定模板和各回合提交在相同的开发期预测起点评分。回合超出模板的分差含候选选择与运行波动，"
                  "仅作诊断。"]
        if attribution.get("status") == "error":
            lines.append("本代诊断未能完成。")
        else:
            lines.append(f"匹配 {attribution['matched_arm_tasks']}/{attribution['expected_arm_tasks']} "
                         "个版本与任务组合（版本在任务上的全部重复回合都有成绩、模板有稳定参考时才算匹配）。")
            for name, comparison in attribution.get("comparisons", {}).items():
                mean = comparison["mean_on_matched_tasks"]
                lines.append(f"- 候选 {name} 的回合分差 {_signed(mean['episode_gain'], 4)}，其中模板分差 "
                             f"{_signed(mean['template_gain'], 4)}，相对各自模板的回合增益差 "
                             f"{_signed(mean['episode_over_template_gain'], 4)}。"
                             f"匹配任务 {comparison['matched_tasks']}/{comparison['expected_tasks']}。")
            for name, arm in attribution.get("arms", {}).items():
                drift = [task for task, row in arm["tasks"].items() if row["template"]["status"] == "drift"]
                if drift:
                    lines.append(f"- {name} 的固定模板在 {', '.join(drift)} 产生不同预测，该处缺少稳定参考。")
            lines += _attribution_rows(attribution, round_, records, names)
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
            horizons = decision.get("horizons") or {}
            ordered_h = sorted(
                ((int(k), float(v)) for k, v in horizons.items() if str(k).isdigit() and isinstance(v, (int, float))),
                key=lambda kv: kv[0],
            )
            per_horizon = "；各时距：" + "，".join(f"h={h} {_signed(val)}" for h, val in ordered_h) if ordered_h else ""
            lines.append(f"- 候选 {candidate['name']} `{candidate.get('digest', '')[:12]}`：{candidate['status']}，"
                         f"Δ {_signed(decision['delta'], 4)}，区间 [{_signed(low)}, {_signed(high)}]，"
                         f"{decision['reason']}。各任务：{per_task}{per_horizon}")
    return "\n".join(lines) + "\n"


def lineage(view: dict[str, Any], prior: Sequence[dict[str, Any]] = ()) -> str:
    """How the incumbent came about, oldest adoption first: each adoption on its line in this run and in the
    ``prior`` runs (``related_runs``: training tasks all among this run's), with the gain that adopted it.

    Successive gains say whether the current family of changes is running out (on 2026-11-04 they had fallen from
    +0.075 to +0.014, +0.007 and +0.006 along one line). Runs with another training task are never read, and the
    independent verifications of a run are not shown: they cover holdout tasks and sealed periods."""

    adopted: dict[str, dict[str, Any]] = {}  # digest -> how it was adopted
    for other in (*prior, view):
        last = {}
        for round_ in other["rounds"]:
            last[round_["generation"]] = round_  # the attempt that counted
        for adoption in other.get("adoptions") or []:
            round_ = last.get(adoption["generation"]) or {}
            candidate = next((item for item in round_.get("candidates") or [] if item["name"] == adoption["name"]), {})
            tasks = (candidate.get("decision") or {}).get("tasks") or []
            adopted[adoption["digest"]] = {
                "run": other["run_id"], "current": other is view, "generation": adoption["generation"],
                "name": adoption["name"], "parent": round_.get("incumbent"), "delta": adoption.get("delta"),
                "better": sum(1 for item in tasks if (item.get("delta") or 0) > 0), "tasks": len(tasks),
                "summary": candidate.get("summary", "")}
    chain, digest = [], view["incumbent"]
    while digest in adopted and len(chain) < 12:
        chain.append((digest, adopted[digest]))
        digest = adopted[digest]["parent"]
    if not chain:
        return ""
    lines = ["## 在任版本的来历（采纳记录）", "",
             "在任版本一路被采纳的过程，最早的在前。增益是相对当时的在任版本、在那次运行的训练任务上算的。"
             "最近几次的增益越来越小、候选又多是调一两个参数时，见 `AGENTS.md`「增益越来越小时」。", "",
             "| 运行 | 代 | 采纳的版本 | 增益 | 更好的任务 | 摘要 |", "|---|---|---|---|---|---|"]
    for digest, item in reversed(chain):
        where = "本次" if item["current"] else f"`{item['run']}`"
        summary = " ".join(item["summary"].split())[:160].replace("|", "/")
        lines.append(f"| {where} | {item['generation']} | {item['name']} `{digest[:12]}` | "
                     f"{_signed(item['delta'], 4)} | {item['better']}/{item['tasks']} | {summary} |")
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
            version = (candidate.get("digest") or "无有效版本")[:12]
            lines.append(f"### 候选 {candidate['name']} `{version}` —— {candidate['status']}")
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
                horizons = decision.get("horizons") or {}
                ordered_h = sorted(
                    (
                        (int(k), float(v))
                        for k, v in horizons.items()
                        if str(k).isdigit() and isinstance(v, (int, float))
                    ),
                    key=lambda kv: kv[0],
                )
                if ordered_h:
                    h_text = "，".join(f"h={h} {_signed(val)}" for h, val in ordered_h)
                    lines.append(f"- 分时距平均变化：{h_text}")
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


def _block(text: str) -> list[str]:
    return [FENCE, text.replace(FENCE, "~~~~"), FENCE, ""]


def transcript(record: dict[str, Any], *, role: str) -> str:
    """The agent's sessions in one episode, from its recorded model calls ("" when none are stored)."""

    directory = Path(((record.get("finished") or {}).get("trajectory") or {}).get("dir", ""))
    if not directory.name:
        return ""
    found = session_threads(directory / "llm_calls.jsonl.gz", [record.get("root") or ""], TURN_LIMITS)
    if not found:
        return ""
    lines = [f"# 回合 {record['episode']} 的会话记录", "",
             f"- 任务 `{record['task_id']}`；{role}；模型 {record.get('model')}；{len(found)} 个线程。",
             "- 节选：系统提示和 DSH 注入的上下文只留一行说明；推理、回复、工具参数和结果过长时保留首尾。"
             "会话目录写成 `/workspace`。",
             "- 智能体在会话中看不到隐藏期；隐藏期成绩见同目录的 `summary.md`。", ""]
    for index, thread in enumerate(found):
        lines += [f"## 线程 {index + 1}/{len(found)}：{thread['label']}（{thread['calls']} 次模型调用）", ""]
        step = None
        for item in thread["steps"]:
            kind = item["kind"]
            if item.get("step") is not None and item["step"] != step:
                step = item["step"]
                lines += [f"### 第 {step} 步", ""]
            if kind == "system":
                lines += [f"（系统提示 {item['chars']} 字，略去）", ""]
            elif kind == "context":
                lines += [f"（DSH 注入的上下文，{item['chars']} 字，略去：{item['text']}）", ""]
            elif kind == "prompt":
                lines += ["### 宿主或上级智能体的消息", "", *_block(item["text"])]
            elif kind == "reasoning":
                lines += ["推理：", "", *_block(item["text"])]
            elif kind == "reply":
                lines += ["回复：", "", *_block(item["text"])]
            elif kind == "call":
                lines += [f"调用 `{item['name']}`：", "", *_block(item["text"])]
            elif kind == "result":
                lines += [f"结果（`{item['name'] or '工具'}`）：", "", *_block(item["text"])]
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
        for item in _round_all_episodes(round_):
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


def related_runs(ledger: Ledger, view: dict[str, Any]) -> list[dict[str, Any]]:
    """Every other evolution run whose training tasks are all among this run's (the same set, or part of it),
    folded, newest first.

    Such a run's evolvers saw the hidden periods of those tasks only, all of them training tasks here, so its
    candidates and notes say nothing about this run's holdout tasks. A run with any other training task is left
    out: that task may be one of this run's holdout tasks."""

    train = set((view.get("suite") or {}).get("train") or [])
    mode = (view.get("config") or {}).get("method_research", "internal")
    found = []
    for run_id in reversed(ledger.runs(first="evolve_started")):
        if not train or run_id == view["run_id"]:
            continue
        try:
            other = fold(ledger.events(run_id))
        except (ValueError, KeyError):
            continue
        other_mode = (other.get("config") or {}).get("method_research", "internal")
        if other_mode != "internal" and (mode != other_mode or
                                         view.get("method_registry_digest") != other.get("method_registry_digest")):
            continue  # a supplied source must not leak into a run set up for blind discovery
        theirs = set((other.get("suite") or {}).get("train") or [])
        if theirs and theirs <= train:
            found.append(other)
    return found


def same_tasks(view: dict[str, Any], other: dict[str, Any]) -> bool:
    """Whether two runs compared their versions on exactly the same training tasks."""

    return (sorted((view.get("suite") or {}).get("train") or [])
            == sorted((other.get("suite") or {}).get("train") or []))


def version_label(run_id: str, generation: int, name: str, *, current: str) -> str:
    """How the evolver's workspace names a candidate: ``g2b`` in this run, ``35ad33-g1b`` in an earlier one."""

    own = f"g{generation}{name}"
    return own if run_id == current else f"{run_id.partition(':')[2][:6]}-{own}"


def prior_runs(ledger: Ledger, view: dict[str, Any], runs: Sequence[dict[str, Any]] | None = None,
               archived: Mapping[str, Any] | None = None) -> str:
    """Earlier runs whose training tasks are all among this run's (``related_runs``): their candidates' outcomes
    and their evolvers' notes; ``archived`` says which candidates the evolver's ``archive/`` holds (see
    ``archive``)."""

    archived = _archived(archived)
    ours = len((view.get("suite") or {}).get("train") or [])
    blocks = []
    for other in related_runs(ledger, view) if runs is None else runs:
        run_id = other["run_id"]
        rounds = [round_ for round_ in other["rounds"] if round_["generation"] > 0]
        if not rounds and not other["notes"]:
            continue
        seed = other.get("seed") or {}
        status = RUN_STATUS.get(other["status"], other["status"])
        theirs = len((other.get("suite") or {}).get("train") or [])
        tasks = ("训练任务与本次相同" if same_tasks(view, other) else
                 f"训练任务 {theirs} 个，都是本次 {ours} 个训练任务中的；它比较过的版本可以在本次原样再比较一次")
        lines = [f"## `{run_id}`（{str(other.get('started_at', ''))[:10]}，起点 {seed.get('name', '')} "
                 f"`{str(seed.get('digest', ''))[:12]}`，{status}，采纳 {len(other['adoptions'])} 次；{tasks}）", ""]
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
    head = ["# 以往运行的候选与评审笔记", "",
            "训练任务都在本次之内（与本次相同，或是本次训练任务的一部分）的其他进化运行，新的在前：每个候选的结果和一句话摘要，"
            "以及那些运行的评审员记下的经验和无效方向。它们的在任版本可能和本次不同，增益是相对当时的在任版本、"
            "在那次的训练任务上算的。用来避免重复已经失败的方向，也可以从其中的候选出发（在 `archive/` 里的那些）。"
            "训练任务与本次相同的运行里比较过的版本，原样提交不会再运行；训练任务比本次少的运行里比较过的版本，"
            "可以原样在本次的全部训练任务上再比较一次：任务更多，检验更有力。"
            "有本次训练任务之外的任务的运行不在这里（它们的证据可能涉及本次的留出任务）。", ""]
    return _clip("\n".join(head) + "\n" + "\n\n".join(blocks), PRIOR_LIMIT) + "\n"


def _skill_names(harness: Any) -> list[str]:
    root = getattr(harness, "root", None)
    if root is None:
        return []
    skills_dir = Path(root) / "skills"
    if not skills_dir.is_dir():
        return []
    return sorted(p.name for p in skills_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def _harness_skill_map(harnesses: Mapping[str, Any] | None) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for key, item in (harnesses or {}).items():
        digest = getattr(item, "digest", None) or str(key)
        skills = _skill_names(item)
        if skills and digest not in found:
            found[digest] = skills
    return found


def _invoked_skills_from_calls(record: dict[str, Any]) -> set[str] | None:
    """Parse exact ``skill`` tool invocations from ``llm_calls.jsonl.gz`` when ``activity.skills`` is absent."""

    tdir = Path(((record.get("finished") or {}).get("trajectory") or {}).get("dir", ""))
    calls_path = tdir / "llm_calls.jsonl.gz"
    if not tdir.name or not calls_path.is_file():
        return None
    found: set[str] = set()
    try:
        with gzip.open(calls_path, "rt", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if '"skill"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                msg = (row.get("reply") or {}).get("message") or {}
                for tc in msg.get("tool_calls") or ():
                    fn = tc.get("function") or {}
                    if (fn.get("name") or tc.get("name")) != "skill":
                        continue
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        continue
                    if isinstance(args, dict) and args.get("name"):
                        found.add(str(args["name"]).lower())
    except OSError:
        return None
    return found


def _episode_skill_set(record: dict[str, Any], known_skills: Sequence[str], transcript_text: str = "") -> set[str]:
    finished = record.get("finished") or {}
    activity = finished.get("activity") or {}
    if isinstance(activity.get("skills"), dict) and (activity.get("sessions") or activity["skills"]):
        return {str(k).lower() for k, v in activity["skills"].items() if v}
    parsed = _invoked_skills_from_calls(record)
    if parsed is not None:
        return parsed
    # Fallback when no trajectory files exist (e.g. unit tests): inspect transcript/final_text/notes
    haystack = " ".join([
        transcript_text,
        str(finished.get("final_text") or ""),
        *(str(note.get("text") or "") for note in record.get("notes") or ()),
    ]).lower()
    return {s.lower() for s in known_skills if s.lower() in haystack}


def _best_eval_detail(record: dict[str, Any]) -> tuple[float | None, bool, str]:
    best_score: float | None = None
    any_passed = False
    best_reason = ""
    for ev in record.get("evaluations") or ():
        sc = (ev.get("local") or {}).get("score")
        sel = ev.get("local_selection") or {}
        if sel.get("passed"):
            any_passed = True
        if isinstance(sc, (int, float)) and np.isfinite(sc):
            val = float(sc)
            if best_score is None or val > best_score:
                best_score = val
                best_reason = str(sel.get("reason") or "")
    for sub in record.get("submissions") or ():
        sel = sub.get("local_selection") or {}
        sc = (sub.get("local") or {}).get("score")
        if isinstance(sc, (int, float)) and np.isfinite(sc):
            val = float(sc)
            if best_score is None or val >= best_score:
                best_score = val
                if sel.get("reason"):
                    best_reason = str(sel["reason"])
    return best_score, any_passed, best_reason


def triage(
    view: dict[str, Any],
    records: dict[str, dict[str, Any]],
    names: dict[str, str],
    *,
    harnesses: Mapping[str, Any] | None = None,
    transcripts_by_ep: Mapping[str, str] | None = None,
) -> str:
    """Zero-LLM failure triage, ACT skill/hook activation attribution, and HDA cost-efficiency metrics."""

    seen_eps: set[str] = set()
    scored_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for round_ in view.get("rounds") or ():
        for item in _round_all_episodes(round_):
            ep = item.get("episode")
            if not ep or ep in seen_eps:
                continue
            record = records.get(ep)
            if not record:
                continue
            seen_eps.add(ep)
            scored = record.get("scored") or {}
            if item.get("status") in SCORED_STATUSES and isinstance(scored.get("score"), (int, float)):
                scored_items.append((item, record))

    lines = [
        "# 病灶分诊、激活归因与效率分析（零 LLM 自动生成）",
        "",
        "基于规则自动汇总失败模式（提交损耗、门控拦截与局部过拟合）、分时距瓶颈、技能/钩子激活归因（ACT）与"
        "成本效率（HDA），用于优先锁定最确定、最高杠杆的 Harness 改进点。",
        "",
        "## 1. 失败模式与提交损耗分诊",
        "",
    ]
    if not scored_items:
        lines.append("（暂无已评分回合可供分诊。）")
        return "\n".join(lines) + "\n"

    incumbent = str(view.get("incumbent") or "")
    # (missed_gain, kind_tag, item, record, start_local, max_eval)
    sub_loss: list[tuple[float, str, dict[str, Any], dict[str, Any], float | None, float | None]] = []
    overfits: list[tuple[float, dict[str, Any], dict[str, Any], float, float]] = []
    horizon_skills: dict[int, list[float]] = {}
    inc_horizon_skills: dict[int, list[float]] = {}
    rescued = 0
    hooked = 0
    aborts = 0
    timeouts = 0
    zero_subs_weak = 0
    kept_strong_template = 0

    for item, record in scored_items:
        scored = record["scored"]
        finished = record.get("finished") or {}
        if finished.get("rescue_triggered"):
            rescued += 1
        if finished.get("hook_fired"):
            hooked += 1
        if (finished.get("degenerate_aborts") or 0) > 0:
            aborts += 1
        hit_timeout = item.get("status") == "timeout" or (finished.get("refused_calls") or 0) > 0
        if hit_timeout:
            timeouts += 1

        for cell in scored.get("cells") or ():
            _, _, h_str = str(cell.get("cell") or "").rpartition("@")
            skill = cell.get("skill")
            if h_str.isdigit() and isinstance(skill, (int, float)) and np.isfinite(skill):
                h = int(h_str)
                horizon_skills.setdefault(h, []).append(float(skill))
                if item.get("harness") == incumbent:
                    inc_horizon_skills.setdefault(h, []).append(float(skill))

        start_raw = ((record.get("start") or {}).get("local") or {}).get("score")
        start_local = float(start_raw) if isinstance(start_raw, (int, float)) and np.isfinite(start_raw) else None
        strong_template = start_local is not None and start_local > REPLACE_MARGIN
        max_eval, any_passed, best_reason = _best_eval_detail(record)
        subs = record.get("submissions") or ()
        if not subs and not strong_template:
            zero_subs_weak += 1

        if scored.get("start") or scored.get("auto") or not subs:
            missed_gain = (max_eval - start_local) if (max_eval is not None and start_local is not None) else 0.0
            has_high_eval = any_passed or missed_gain >= max(REPLACE_MARGIN, UNSUBMITTED_GAIN)
            if any_passed:
                sub_loss.append((missed_gain, "过门槛未交", item, record, start_local, max_eval))
            elif has_high_eval:
                tag = f"门控拦截:{best_reason}" if best_reason and best_reason != "passed" else "高分未交"
                sub_loss.append((missed_gain, tag, item, record, start_local, max_eval))
            elif not strong_template and not subs:
                sub_loss.append((missed_gain, "弱模板零提交", item, record, start_local, max_eval))
            elif strong_template and scored.get("start"):
                kept_strong_template += 1
        else:
            local_val = _scored_local(record)
            hidden_val = float(scored["score"])
            if isinstance(local_val, (int, float)) and np.isfinite(local_val):
                gap = float(local_val) - hidden_val
                if gap > OVERFIT_GAP:
                    overfits.append((gap, item, record, float(local_val), hidden_val))

    total_scored = len(scored_items)
    lines.append(
        f"- 统计范围：{total_scored} 个已评分回合（触顶超时 {timeouts}，弱模板零提交 {zero_subs_weak}，"
        f"强模板合规沿用 {kept_strong_template}，触发预算救援 {rescued}，触发审计钩子 {hooked}，复读熔断 {aborts}）。"
    )
    if sub_loss:
        sub_loss.sort(key=lambda row: (-row[0], row[2].get("episode", "")))
        missed_unsub = sum(1 for _, tag, *_ in sub_loss if tag in ("过门槛未交", "高分未交"))
        gate_blocked = sum(1 for _, tag, *_ in sub_loss if tag.startswith("门控拦截:"))
        detail_bits = []
        if missed_unsub:
            detail_bits.append(f"高分/过门槛漏交 {missed_unsub} 个")
        if gate_blocked:
            detail_bits.append(f"被本地替换门控拦截 {gate_blocked} 个")
        weak_zero = sum(1 for _, tag, *_ in sub_loss if tag == "弱模板零提交")
        if weak_zero:
            detail_bits.append(f"弱模板零提交 {weak_zero} 个")
        detail = f"（{'，'.join(detail_bits)}；优先通过 `hooks/rescue.md` 或门控针对性优化修复）" if detail_bits else ""
        lines += [
            f"- **提交损耗与门控拦截**：发现 {len(sub_loss)} 个异常回合{detail}：",
            "",
            "| 回合 | 版本 | 任务 | 模板本地分 | 最高本地分 | 提交数 | 诊断类型 | 预算救援 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, tag, item, record, start_l, max_e in sub_loss[:10]:
            h_id = str(item.get("harness") or "")
            fin = record.get("finished") or {}
            lines.append(
                f"| `{record['episode']}` | {names.get(h_id, '')} `{h_id[:8]}` | `{record['task_id']}` | "
                f"{_signed(start_l, 4)} | {_signed(max_e, 4)} | {len(record.get('submissions') or ())} | "
                f"{tag} | {'是' if fin.get('rescue_triggered') else '否'} |"
            )
        lines.append("")
    else:
        lines.append("- **提交损耗与门控拦截**：未发现弱模板零提交、漏提交或高分门控拦截。")

    if overfits:
        overfits.sort(key=lambda row: (-row[0], row[1].get("episode", "")))
        lines += [
            f"- **本地→隐藏期过拟合反转（Local − Hidden > {OVERFIT_GAP:.2f}）**：共 {len(overfits)} 个替换模板的回合"
            "出现显著回撤（建议检查特征过拟合、正则强度或在 `hooks/audit_submission.py` 中增加门禁审计）：",
            "",
            "| 回合 | 版本 | 任务 | 评分程序本地分 | 隐藏期得分 | 回撤 (本地−隐藏) |",
            "|---|---|---|---|---|---|",
        ]
        for gap, item, record, local_v, hidden_v in overfits[:10]:
            h_id = str(item.get("harness") or "")
            lines.append(
                f"| `{record['episode']}` | {names.get(h_id, '')} `{h_id[:8]}` | `{record['task_id']}` | "
                f"{_signed(local_v, 4)} | {_signed(hidden_v, 4)} | {gap:+.4f} |"
            )
        lines.append("")
    else:
        lines.append(f"- **本地→隐藏期过拟合反转**：无回撤超过 {OVERFIT_GAP:.2f} 的提交。")

    if horizon_skills:
        lines += ["", "### 分时距隐藏期瓶颈（由弱到强）", "", "| 预报时距 | 在任版本均分 | 全部回合均分 | 样本数 |", "|---|---|---|---|"]
        ordered_h = sorted(horizon_skills, key=lambda h: float(np.mean(inc_horizon_skills.get(h) or horizon_skills[h])))
        for h in ordered_h:
            inc_v = _mean(inc_horizon_skills.get(h, []))
            all_v = _mean(horizon_skills[h])
            lines.append(f"| `h={h}` | {_signed(inc_v, 4)} | {_signed(all_v, 4)} | {len(horizon_skills[h])} |")
        lines.append("")

    # Section 2: ACT Skill & Hook Activation Attribution
    lines += ["## 2. 技能与钩子激活归因（ACT Attribution）", ""]
    hook_acted = [
        float(rec["scored"]["score"]) for _, rec in scored_items if (rec.get("finished") or {}).get("hook_fired")
    ]
    hook_idle = [float(rec["scored"]["score"]) for _, rec in scored_items
                 if not (rec.get("finished") or {}).get("hook_fired")]
    if hook_acted:
        mean_a, mean_i = _mean(hook_acted), _mean(hook_idle)
        diff_txt = f"，差值 {_signed((mean_a or 0.0) - (mean_i or 0.0), 4)}" if mean_i is not None else ""
        lines.append(
            f"- **提交审计钩子（`hook_fired`）**：触发告警 {len(hook_acted)}/{total_scored} 回合"
            f"（{len(hook_acted) / total_scored:.0%}），触发均分 {_signed(mean_a, 4)} vs 未触发 {_signed(mean_i, 4)}{diff_txt}。"
        )
    else:
        lines.append("- **提交审计钩子（`hooks/audit_submission.py`）**：尚未在任何回合触发告警（当前 Harness 未配置或未命中告警）。")

    rescue_acted = [float(rec["scored"]["score"]) for _, rec in scored_items
                    if (rec.get("finished") or {}).get("rescue_triggered")]
    rescue_idle = [float(rec["scored"]["score"]) for _, rec in scored_items
                   if not (rec.get("finished") or {}).get("rescue_triggered")]
    if rescue_acted:
        lines.append(
            f"- **预算救援（`rescue_triggered`）**：触发 {len(rescue_acted)}/{total_scored} 回合，"
            f"触发均分 {_signed(_mean(rescue_acted), 4)} vs 未触发 {_signed(_mean(rescue_idle), 4)}。"
        )

    skill_map = _harness_skill_map(harnesses)
    if skill_map:
        tx_map = transcripts_by_ep or {}
        ep_skills: dict[str, set[str]] = {}
        skill_totals: dict[str, list[int]] = {}  # skill -> [acted_eps, total_eps_with_skill]
        skill_rows: list[str] = []
        for h_digest, skill_list in skill_map.items():
            h_eps = [(item, rec) for item, rec in scored_items if item.get("harness") == h_digest]
            if not h_eps:
                continue
            label = f"{names.get(h_digest, '')} `{h_digest[:8]}`"
            for _, rec in h_eps:
                if rec["episode"] not in ep_skills:
                    ep_skills[rec["episode"]] = _episode_skill_set(
                        rec, skill_list, tx_map.get(rec["episode"], "")
                    )
            for skill in skill_list:
                key = skill.lower()
                acted = [float(rec["scored"]["score"]) for _, rec in h_eps if key in ep_skills[rec["episode"]]]
                not_acted = [float(rec["scored"]["score"]) for _, rec in h_eps if key not in ep_skills[rec["episode"]]]
                counts = skill_totals.setdefault(skill, [0, 0])
                counts[0] += len(acted)
                counts[1] += len(h_eps)
                rate = len(acted) / len(h_eps)
                skill_rows.append(
                    f"| {label} | `{skill}` | {len(acted)}/{len(h_eps)} ({rate:.0%}) | "
                    f"{_signed(_mean(acted), 4)} | {_signed(_mean(not_acted), 4)} |"
                )
        if skill_rows:
            lines += [
                "",
                "| 版本 | 技能目录 | 激活率 | 激活回合均分 | 未激活回合均分 |",
                "|---|---|---|---|---|",
                *skill_rows,
            ]
        dead_skills = [f"`{name}`（0/{tot} 回合）" for name, (hit, tot) in sorted(skill_totals.items())
                       if hit == 0 and tot > 0]
        if dead_skills:
            lines += ["", f"- ⚠ **零激活技能（建议合并或剪枝，避免上下文膨胀）**：{'、'.join(dead_skills)}。"]
    lines.append("")

    # Section 3: HDA Cost-Efficiency Attribution
    lines += [
        "## 3. 成本与效率归因（HDA: Harness Delta Attribution）",
        "",
        "对比各版本的隐藏期得分、相对在任版本增益、平均 Token 消耗与单位 Token 收益（`ΔScore / MToken`）。"
        f"若成本倍率 > {HDA_COST_RATIO_WARN:.2f}× 且增益 < {HDA_MIN_GAIN:+.3f}，标记为高成本低增益。",
        "",
    ]
    by_harness: dict[str, dict[str, list[float]]] = {}
    for item, rec in scored_items:
        h_id = str(item.get("harness") or "")
        fin = rec.get("finished") or {}
        bucket = by_harness.setdefault(h_id, {"scores": [], "tokens": [], "seconds": []})
        bucket["scores"].append(float(rec["scored"]["score"]))
        tok = fin.get("tokens")
        if isinstance(tok, (int, float)) and tok > 0:
            bucket["tokens"].append(float(tok))
        sec = fin.get("seconds")
        if isinstance(sec, (int, float)) and sec >= 0:
            bucket["seconds"].append(float(sec))

    decision_delta: dict[str, float] = {}
    for round_ in view.get("rounds") or ():
        for cand in round_.get("candidates") or ():
            dec = cand.get("decision") or {}
            if cand.get("digest") and isinstance(dec.get("delta"), (int, float)):
                decision_delta[str(cand["digest"])] = float(dec["delta"])

    inc_bucket = by_harness.get(incumbent) or {}
    inc_score = _mean(inc_bucket.get("scores", []))
    inc_tokens = _mean(inc_bucket.get("tokens", []))

    lines += [
        "| 版本 | 回合数 | 隐藏均分 | 相对在任 Δ | 均值 Token (K) | 均耗 (秒) | 成本倍率 | Δ / MToken | 效率诊断 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for h_id, bucket in by_harness.items():
        m_score = _mean(bucket["scores"])
        m_tok = _mean(bucket["tokens"])
        m_sec = _mean(bucket["seconds"])
        delta = (
            0.0 if h_id == incumbent else decision_delta.get(
                h_id, (m_score - inc_score) if (m_score is not None and inc_score is not None) else None
            )
        )
        ratio = (m_tok / inc_tokens) if (m_tok and inc_tokens and inc_tokens > 0) else None
        mtok_gain = (delta / max(m_tok / 1_000_000.0, 0.01)) if (delta is not None and m_tok) else None
        mark = "（在任）" if h_id == incumbent else ""
        if h_id == incumbent:
            diag = "基线"
        elif ratio is not None and delta is not None and ratio > HDA_COST_RATIO_WARN and delta < HDA_MIN_GAIN:
            diag = "⚠ 高成本低增益"
        elif ratio is not None and delta is not None and ratio <= 1.05 and delta > 0.005:
            diag = "★ 提效且增分"
        elif delta is not None and delta > 0.0:
            diag = "正增益"
        else:
            diag = "无净增益"
        tok_str = f"{m_tok / 1000:.1f}K" if m_tok is not None else "–"
        sec_str = f"{m_sec:.0f}s" if m_sec is not None else "–"
        ratio_str = f"{ratio:.2f}×" if ratio is not None else "–"
        lines.append(
            f"| {names.get(h_id, '')} `{h_id[:8]}`{mark} | {len(bucket['scores'])} | {_signed(m_score, 4)} | "
            f"{_signed(delta, 4)} | {tok_str} | {sec_str} | {ratio_str} | {_signed(mtok_gain, 3)} | {diag} |"
        )
    return "\n".join(lines) + "\n"


def write_evidence(target: Path, *, ledger: Ledger, view: dict[str, Any], tasks: Sequence[str],
                   names: dict[str, str], expert_text: str = "",
                   ensembles: dict[str, dict[str, dict[str, Any]]] | None = None,
                   archived: Mapping[str, Any] | None = None,
                   prior: Sequence[dict[str, Any]] | None = None,
                   incumbent: Any = None) -> dict[str, Any]:
    """Write ``overview.md``, ``triage.md``, ``archive.md``, ``lessons.md``, ``contrasts.md``, ``expert.md``,
    ``prior_runs.md`` and one folder per episode (``summary.md``, ``transcript.md``, the scored ``submission/``);
    ``archived`` maps the labels of the versions copied into the evolver's ``archive/`` to their harnesses,
    ``prior`` lists the runs whose training tasks are all among this run's (``related_runs``, read from the
    ledger when None)."""

    harness_map: dict[str, Any] = {}
    if incumbent is not None and getattr(incumbent, "digest", None):
        harness_map[str(incumbent.digest)] = incumbent
    for item in (archived or {}).values():
        if getattr(item, "digest", None):
            harness_map[str(item.digest)] = item
    archived = _archived(archived)

    target.mkdir(parents=True, exist_ok=True)
    roles = {}
    for round_ in view["rounds"]:
        for stage in round_.get("relay_stages") or ():
            for item in stage.get("episodes") or ():
                if not item.get("episode"):
                    continue
                role = "在任" if item["role"] == "incumbent" else f"接力候选 {item.get('candidate')}"
                roles[item["episode"]] = f"第 {round_['generation']} 代接力阶段 {stage.get('stage')}，{role}"
        for item in round_["episodes"]:
            if not item.get("episode"):
                continue  # the episode raised before it could start; nothing was recorded
            role = "在任" if item["role"] == "incumbent" else f"候选 {item.get('candidate')}"
            roles[item["episode"]] = f"第 {round_['generation']} 代，{role}"
    records = {episode: fold_episode(ledger.events(episode)) for episode in roles}
    episodes_dir = target / "episodes"
    programs: dict[str, dict[str, bytes]] = {}
    transcripts_by_ep: dict[str, str] = {}
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
            transcripts_by_ep[episode] = text
            transcripts += 1
    record = lineage(view, related_runs(ledger, view) if prior is None else prior)
    (target / "overview.md").write_text(overview(view, records, tasks, names, ensembles)
                                        + (f"\n{record}" if record else ""), encoding="utf-8")
    (target / "triage.md").write_text(
        triage(view, records, names, harnesses=harness_map, transcripts_by_ep=transcripts_by_ep),
        encoding="utf-8",
    )
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
           "program_diff", "related_runs", "same_tasks", "transcript", "triage", "version_label", "write_evidence"]
