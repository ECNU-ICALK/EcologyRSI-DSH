"""Intra-generation mini-batch relay evolution: stratified sampling, stage ranking, and finalist selection.

A generation with ``relay_stages >= 2`` explores multiple candidates cheaply on complementary mini-batches of
training tasks (1 replicate each), immediately feeds the winning harness and raw trajectories into the next
intra-generation evolver stage, and promotes only the best distinct finalist(s) into the clean full-suite
adoption gate. This increases effective evolution steps per generation while preserving full Bonferroni
statistical power and zero selection-bias contamination at the final adoption gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..task.spec import TaskSpec
from .fold import SCORED_STATUSES


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def effective_relay_candidates(candidates: int, relay_candidates: int) -> int:
    """Candidates explored per relay screening stage (auto: at least 2, capped at 4)."""

    if relay_candidates > 0:
        return min(6, max(1, relay_candidates))
    return min(4, max(2, candidates))


def effective_relay_batch(total_tasks: int, *, total_stages: int, batch_size: int = 0) -> int:
    """Number of training tasks screened per relay stage."""

    if total_tasks <= 0:
        return 0
    if batch_size > 0:
        return min(total_tasks, max(1, batch_size))
    return max(1, min(total_tasks, math.ceil(total_tasks / max(1, total_stages))))


def select_stage_tasks(
    tasks: Sequence[TaskSpec],
    *,
    generation: int,
    stage: int,
    total_stages: int,
    batch_size: int = 0,
    task_difficulty: Mapping[str, float] | None = None,
) -> list[TaskSpec]:
    """Deterministically pick a stratified, complementary mini-batch of training tasks for a relay stage.

    Tasks are interleaved across ``task.dataset`` groups before slicing so every mini-batch balances groups and
    consecutive stages within a generation draw complementary slices. When ``task_difficulty`` is provided,
    each dataset bucket is sorted by descending difficulty/regret so Stage 1 screens the highest-regret tasks
    across every dataset first; otherwise successive generations rotate their starting offset.
    """

    if not tasks:
        return []
    groups: dict[str, list[tuple[int, TaskSpec]]] = {}
    for idx, task in enumerate(tasks):
        groups.setdefault(task.dataset, []).append((idx, task))
    use_diff = bool(task_difficulty)
    ordered_groups: list[list[TaskSpec]] = []
    for bucket in groups.values():
        if use_diff:
            bucket = sorted(
                bucket,
                key=lambda pair: (
                    -float(task_difficulty[pair[1].id])
                    if pair[1].id in task_difficulty and _finite(task_difficulty[pair[1].id])
                    else 0.0,
                    pair[0],
                ),
            )
        ordered_groups.append([task for _, task in bucket])
    ring: list[TaskSpec] = []
    max_len = max(len(bucket) for bucket in ordered_groups)
    for index in range(max_len):
        for bucket in ordered_groups:
            if index < len(bucket):
                ring.append(bucket[index])
    count = effective_relay_batch(len(ring), total_stages=total_stages, batch_size=batch_size)
    if count >= len(ring):
        return list(ring)
    if use_diff:
        offset = ((max(1, stage) - 1) * count) % len(ring)
    else:
        offset = ((max(1, generation) - 1) + (max(1, stage) - 1) * count) % len(ring)
    chosen: list[TaskSpec] = []
    seen: set[str] = set()
    for step in range(len(ring)):
        task = ring[(offset + step) % len(ring)]
        if task.id not in seen:
            seen.add(task.id)
            chosen.append(task)
            if len(chosen) >= count:
                break
    return chosen


def rank_stage_candidates(
    candidates: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    tasks: Sequence[TaskSpec],
    *,
    incumbent_baseline: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Score and rank one relay stage's candidates by paired delta against the incumbent on ``tasks``."""

    task_ids = [task.id for task in tasks]
    inc_scores: dict[str, list[float]] = {}
    cand_scores: dict[tuple[str, str], list[float]] = {}
    for entry in entries:
        if entry.get("status") not in SCORED_STATUSES or not _finite(entry.get("score")):
            continue
        task_id = str(entry.get("task") or "")
        if task_id not in task_ids:
            continue
        score = float(entry["score"])
        if entry.get("role") == "incumbent":
            inc_scores.setdefault(task_id, []).append(score)
        elif entry.get("role") == "candidate" and entry.get("candidate"):
            cand_scores.setdefault((str(entry["candidate"]), task_id), []).append(score)

    inc_mean_by_task: dict[str, tuple[float, bool]] = {}
    for task_id in task_ids:
        contemporary = inc_scores.get(task_id)
        if contemporary:
            inc_mean_by_task[task_id] = (sum(contemporary) / len(contemporary), False)
        elif incumbent_baseline and _finite(incumbent_baseline.get(task_id)):
            inc_mean_by_task[task_id] = (float(incumbent_baseline[task_id]), True)

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        name = str(candidate["name"])
        task_deltas: list[dict[str, Any]] = []
        raw_means: list[float] = []
        paired_deltas: list[float] = []
        inc_means: list[float] = []
        for task_id in task_ids:
            values = cand_scores.get((name, task_id))
            if not values:
                continue
            cand_mean = sum(values) / len(values)
            raw_means.append(cand_mean)
            inc_ref = inc_mean_by_task.get(task_id)
            if inc_ref is not None:
                inc_val, from_prior = inc_ref
                delta = cand_mean - inc_val
                paired_deltas.append(delta)
                inc_means.append(inc_val)
                task_deltas.append({
                    "task": task_id,
                    "candidate": round(cand_mean, 6),
                    "incumbent": round(inc_val, 6),
                    "delta": round(delta, 6),
                    "incumbent_from_prior": from_prior,
                })
            else:
                task_deltas.append({
                    "task": task_id,
                    "candidate": round(cand_mean, 6),
                    "incumbent": None,
                    "delta": None,
                    "incumbent_from_prior": False,
                })
        paired = len(paired_deltas)
        better = sum(1 for d in paired_deltas if d > 0.0)
        worse = sum(1 for d in paired_deltas if d < 0.0)
        mean_delta = sum(paired_deltas) / paired if paired else None
        mean_score = sum(raw_means) / len(raw_means) if raw_means else None
        incumbent_mean = sum(inc_means) / len(inc_means) if inc_means else None
        min_delta = min(paired_deltas) if paired_deltas else None
        win_rate = better / paired if paired else 0.0
        beats = bool(paired > 0 and mean_delta is not None and mean_delta > 0.0 and better >= paired / 2.0)
        ranked.append({
            "name": name,
            "label": str(candidate.get("label") or name),
            "digest": candidate.get("digest"),
            "behavior": candidate.get("behavior"),
            "summary": candidate.get("summary", ""),
            "paired": paired,
            "total_tasks": len(task_ids),
            "better": better,
            "worse": worse,
            "mean_delta": round(mean_delta, 6) if mean_delta is not None else None,
            "mean_score": round(mean_score, 6) if mean_score is not None else None,
            "incumbent_mean": round(incumbent_mean, 6) if incumbent_mean is not None else None,
            "min_delta": round(min_delta, 6) if min_delta is not None else None,
            "win_rate": round(win_rate, 4),
            "beats_incumbent": beats,
            "task_deltas": task_deltas,
        })

    ranked.sort(
        key=lambda row: (
            row["paired"] > 0,
            row["beats_incumbent"],
            row["mean_delta"] if row["mean_delta"] is not None else -math.inf,
            row["win_rate"],
            row["min_delta"] if row["min_delta"] is not None else -math.inf,
            row["mean_score"] if row["mean_score"] is not None else -math.inf,
        ),
        reverse=True,
    )
    return ranked


def pick_stage_leader(
    ranked: Sequence[Mapping[str, Any]],
    *,
    fallback_digest: str,
    fallback_label: str,
) -> dict[str, Any]:
    """Choose the next relay stage's starting harness: advance only when the top candidate beats the incumbent."""

    top = ranked[0] if ranked else None
    if top and top.get("beats_incumbent") and top.get("digest"):
        return {
            "advanced": True,
            "name": top["name"],
            "label": top.get("label") or top["name"],
            "digest": top["digest"],
            "behavior": top.get("behavior"),
            "mean_delta": top.get("mean_delta"),
            "better": top.get("better", 0),
            "paired": top.get("paired", 0),
            "total_tasks": top.get("total_tasks", 0),
        }
    return {
        "advanced": False,
        "name": None,
        "label": fallback_label,
        "digest": fallback_digest,
        "behavior": None,
        "mean_delta": top.get("mean_delta") if top else None,
        "better": top.get("better", 0) if top else 0,
        "paired": top.get("paired", 0) if top else 0,
        "total_tasks": top.get("total_tasks", 0) if top else 0,
    }


def select_finalists(
    stage_records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    min_delta: float = 0.0,
    skip_if_all_worse: bool = False,
) -> list[dict[str, Any]]:
    """Select up to ``limit`` best distinct candidates across all completed relay stages for the final gate.

    When the same candidate (matched by behavior/digest, e.g. a stage-1 leader cross-checked on stage-2's
    complementary tasks) was evaluated in multiple stages, its per-task paired deltas are merged across stages
    so cross-batch generalization directly governs finalist selection. When ``skip_if_all_worse`` is True and
    every scored candidate falls below ``-min_delta`` without beating the incumbent, returns ``[]`` to avoid
    wasting a full-suite final gate on clearly dominated candidates.
    """

    if limit <= 0:
        return []
    merged: dict[str, dict[str, Any]] = {}
    for stage_rec in stage_records:
        stage_num = int(stage_rec.get("stage") or 0)
        by_name = {str(item["name"]): item for item in stage_rec.get("candidates") or [] if item.get("digest")}
        for row in stage_rec.get("ranking") or []:
            cand = by_name.get(str(row.get("name") or ""))
            if not cand or not cand.get("digest"):
                continue
            key = str(cand.get("behavior") or cand["digest"])
            bucket = merged.get(key)
            if bucket is None:
                bucket = {
                    "cand": dict(cand),
                    "stage": stage_num,
                    "label": str(row.get("label") or cand.get("label") or cand["name"]),
                    "name": str(cand["name"]),
                    "task_deltas": {},  # task_id -> list[float]
                    "raw_scores": {},  # task_id -> list[float]
                    "stages_seen": [],
                    "fallback_row": row,
                }
                merged[key] = bucket
            else:
                if stage_num > bucket["stage"] and not cand.get("carried_leader"):
                    bucket["cand"] = dict(cand)
                    bucket["stage"] = stage_num
                    bucket["label"] = str(row.get("label") or cand.get("label") or cand["name"])
                    bucket["name"] = str(cand["name"])
            if stage_num not in bucket["stages_seen"]:
                bucket["stages_seen"].append(stage_num)
            for td in row.get("task_deltas") or ():
                tid = str(td.get("task") or "")
                if not tid:
                    continue
                if _finite(td.get("delta")):
                    bucket["task_deltas"].setdefault(tid, []).append(float(td["delta"]))
                if _finite(td.get("candidate")):
                    bucket["raw_scores"].setdefault(tid, []).append(float(td["candidate"]))

    scored: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for bucket in merged.values():
        fb = bucket["fallback_row"]
        if bucket["task_deltas"]:
            per_task = [sum(vals) / len(vals) for vals in bucket["task_deltas"].values()]
            paired = len(per_task)
            better = sum(1 for d in per_task if d > 0.0)
            mean_delta: float | None = round(sum(per_task) / paired, 6)
            win_rate = better / paired
            beats = bool(mean_delta > 0.0 and better >= paired / 2.0)
            total_tasks = max(paired, int(fb.get("total_tasks") or paired))
        else:
            paired = int(fb.get("paired") or 0)
            better = int(fb.get("better") or 0)
            mean_delta = float(fb["mean_delta"]) if _finite(fb.get("mean_delta")) else None
            win_rate = float(fb.get("win_rate") or 0.0)
            beats = bool(fb.get("beats_incumbent"))
            total_tasks = int(fb.get("total_tasks") or 0)
        if bucket["raw_scores"]:
            raw_means = [sum(vals) / len(vals) for vals in bucket["raw_scores"].values()]
            mean_score: float | None = sum(raw_means) / len(raw_means)
        else:
            mean_score = float(fb["mean_score"]) if _finite(fb.get("mean_score")) else None
        score_key = (
            beats,
            paired > 0,
            mean_delta if mean_delta is not None else -math.inf,
            len(bucket["stages_seen"]),
            bucket["stage"],
            win_rate,
            mean_score if mean_score is not None else -math.inf,
        )
        item = {
            **bucket["cand"],
            "relay_origin": {
                "stage": bucket["stage"],
                "stages_tested": list(bucket["stages_seen"]),
                "name": bucket["name"],
                "label": bucket["label"],
                "mean_delta": mean_delta,
                "better": better,
                "paired": paired,
                "total_tasks": total_tasks,
                "beats_incumbent": beats,
            },
        }
        scored.append((score_key, item))

    ordered = sorted(scored, key=lambda pair: pair[0], reverse=True)
    if skip_if_all_worse and ordered:
        def _viable(item: dict[str, Any]) -> bool:
            org = item["relay_origin"]
            if org["beats_incumbent"] or org["paired"] <= 0:
                return True
            md = org["mean_delta"]
            return md is not None and md >= -abs(min_delta)

        if not any(_viable(item) for _, item in ordered):
            return []
    return [item for _, item in ordered[:limit]]


def _fmt_delta(value: Any) -> str:
    return f"{float(value):+.4f}" if _finite(value) else "–"


def render_relay_brief(
    *,
    generation: int,
    stage: int,
    total_stages: int,
    stage_tasks: Sequence[TaskSpec],
    prior_stages: Sequence[Mapping[str, Any]],
    starting_label: str,
    starting_digest: str,
    incumbent_digest: str,
    final_limit: int,
) -> str:
    """Markdown guidance explaining the current intra-generation relay stage to the evolver."""

    task_list = "、".join(f"`{task.id}`" for task in stage_tasks)
    lines = [
        f"## 代内小批接力进化（第 {generation} 代 · 阶段 {stage}/{total_stages}）",
        "",
        f"本代启用了 **{total_stages} 阶段小批接力筛选 + 全量终审**。",
        f"- **本阶段评测子批（{len(stage_tasks)} 个分层任务，每任务 1 重复）**：{task_list}。",
        f"- **终审晋级名额**：全部 {total_stages} 个接力阶段结束后，宿主从各阶段优胜候选中选出最好的 {final_limit} 个，"
        "与本代在任版本在**全部训练任务**上做干净的正式配对门控。",
    ]
    if starting_digest != incumbent_digest:
        lines += [
            f"- **本阶段默认出发版本**：上一阶段胜出的 `{starting_label}`（`{starting_digest[:12]}`）。"
            f"`candidates/` 下的每个候选目录已经初始化为 `{starting_label}` 的完整副本；`incumbent/` 仍保留本代真正在任版本"
            f"（`{incumbent_digest[:12]}`），所有阶段候选也保存在 `archive/` 中。",
            "- **你的目标**：先阅读 `evidence/` 中上一阶段各回合的摘要、程序与会话记录，针对薄弱任务或可泛化方向在 "
            f"`{starting_label}` 之上进一步改进（如要回退或组合，可用 `reset_candidate` 切回 `incumbent` 或其他阶段版本）。"
            f"不要原样提交未修改的 `{starting_label}`（它已保留在终审候选池中）。",
        ]
    else:
        lines += [
            f"- **本阶段默认出发版本**：本代在任版本 `{incumbent_digest[:12]}`。",
            "- **你的目标**：提出结构互补、各有侧重的候选，在当前小批任务上并行试错，挑出更有潜力的方向交给下一阶段继续精修。",
        ]
    if prior_stages:
        lines += ["", "### 本代前序接力阶段结果", ""]
        for rec in prior_stages:
            s_num = rec.get("stage")
            s_tasks = "、".join(f"`{t}`" for t in rec.get("tasks") or [])
            leader = rec.get("leader") or {}
            lead_note = (
                f"胜出并晋级下段起点：`{leader.get('label')}`（Δ {_fmt_delta(leader.get('mean_delta'))}，"
                f"{leader.get('better', 0)}/{leader.get('paired', 0)} 更好）"
                if leader.get("advanced")
                else f"未超过在任版本，下段保持从 `{leader.get('label') or '在任'}` 出发"
            )
            lines.append(f"#### 阶段 {s_num}/{total_stages}（子批：{s_tasks}）— {lead_note}")
            lines.append("")
            ranking = rec.get("ranking") or []
            if not ranking:
                lines.append("- 本阶段没有产出可比较的候选。")
            else:
                for row in ranking:
                    per_task = "；".join(
                        f"`{item['task']}` {_fmt_delta(item['delta'])}"
                        for item in row.get("task_deltas") or []
                    )
                    tag = "✓ 优于在任" if row.get("beats_incumbent") else "未超过在任"
                    summary = " ".join(str(row.get("summary") or "").split())[:160]
                    lines.append(
                        f"- 候选 `{row.get('label')}`（`{str(row.get('digest') or '')[:12]}`，{tag}）："
                        f"配对平均 Δ {_fmt_delta(row.get('mean_delta'))}，"
                        f"{row.get('better', 0)}/{row.get('paired', 0)} 任务更好；"
                        f"各任务：{per_task or '无'}"
                        + (f"。摘要：{summary}" if summary else "")
                    )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "effective_relay_batch",
    "effective_relay_candidates",
    "pick_stage_leader",
    "rank_stage_candidates",
    "render_relay_brief",
    "select_finalists",
    "select_stage_tasks",
]
