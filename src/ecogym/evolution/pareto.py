"""Multi-objective Pareto archive and complementary crossover selection for harness evolution.

Single-trunk hill climbing (comparing every candidate solely against one incumbent across
all tasks) struggles with trade-offs across forecast horizons (short vs long) and across
task groups. This module computes a non-dominated Pareto frontier over compared versions
and surfaces complementary parent pairs suitable for structural recombination (crossover).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ArchivePoint:
    """One compared archived candidate projected into multi-objective behavior space."""

    label: str
    digest: str
    delta: float
    win_share: float
    horizons: dict[int, float]
    groups: dict[str, float]
    status: str
    parent: str


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def extract_points(
    by_label: Mapping[str, dict[str, Any]],
    digests: Mapping[str, str],
) -> list[ArchivePoint]:
    """Build multi-objective points for archived versions that have a complete suite decision."""

    points: list[ArchivePoint] = []
    for label, digest_id in digests.items():
        candidate = by_label.get(label)
        if not isinstance(candidate, dict):
            continue
        decision = candidate.get("decision")
        if not isinstance(decision, dict) or candidate.get("missing"):
            continue
        delta = _finite(decision.get("delta"))
        if delta is None:
            continue
        tasks = decision.get("tasks") or []
        better = sum(1 for item in tasks if (_finite(item.get("delta")) or 0.0) > 0.0)
        win_share = better / len(tasks) if tasks else 0.0
        horizons: dict[int, float] = {}
        for key, value in (decision.get("horizons") or {}).items():
            if str(key).isdigit() and (val := _finite(value)) is not None:
                horizons[int(key)] = val
        groups: dict[str, float] = {}
        for key, entry in (decision.get("groups") or {}).items():
            if isinstance(entry, dict) and (val := _finite(entry.get("delta"))) is not None:
                groups[str(key)] = val
        points.append(
            ArchivePoint(
                label=label,
                digest=digest_id,
                delta=delta,
                win_share=win_share,
                horizons=horizons,
                groups=groups,
                status=str(candidate.get("status") or ""),
                parent=str(candidate.get("parent") or ""),
            )
        )
    return points


def _objectives(point: ArchivePoint, horizons: Sequence[int], groups: Sequence[str]) -> tuple[float, ...]:
    vals = [point.delta, point.win_share]
    vals.extend(point.horizons.get(h, -math.inf) for h in horizons)
    vals.extend(point.groups.get(g, -math.inf) for g in groups)
    return tuple(vals)


def pareto_front(points: Sequence[ArchivePoint]) -> list[ArchivePoint]:
    """Return the non-dominated subset across overall gain, win-rate, per-horizon, and per-group gains."""

    if len(points) <= 1:
        return list(points)
    horizons = sorted({h for pt in points for h in pt.horizons})
    groups = sorted({g for pt in points for g in pt.groups})
    vectors = [_objectives(pt, horizons, groups) for pt in points]
    front: list[ArchivePoint] = []
    for idx, vec_a in enumerate(vectors):
        dominated = False
        for jdx, vec_b in enumerate(vectors):
            if idx == jdx:
                continue
            if all(b >= a for a, b in zip(vec_a, vec_b, strict=True)) and any(
                b > a for a, b in zip(vec_a, vec_b, strict=True)
            ):
                dominated = True
                break
        if not dominated:
            front.append(points[idx])
    front.sort(key=lambda pt: (-pt.delta, -pt.win_share, pt.label))
    return front


def niche_leaders(points: Sequence[ArchivePoint]) -> dict[str, ArchivePoint]:
    """MAP-Elites style specialist leaders: best overall, best per horizon, and best per task group."""

    leaders: dict[str, ArchivePoint] = {}
    if not points:
        return leaders
    leaders["overall"] = max(points, key=lambda pt: (pt.delta, pt.win_share))
    for h in sorted({h for pt in points for h in pt.horizons}):
        eligible = [pt for pt in points if h in pt.horizons]
        if eligible:
            leaders[f"h={h}"] = max(eligible, key=lambda pt: (pt.horizons[h], pt.delta))
    for g in sorted({g for pt in points for g in pt.groups}):
        eligible = [pt for pt in points if g in pt.groups]
        if eligible:
            leaders[f"group:{g}"] = max(eligible, key=lambda pt: (pt.groups[g], pt.delta))
    return leaders


def crossover_pairs(
    points: Sequence[ArchivePoint],
    *,
    max_pairs: int = 3,
    min_niche_advantage: float = 0.003,
    max_overall_drop: float = -0.015,
) -> list[dict[str, Any]]:
    """Suggest complementary parent pairs whose strengths lie on different horizons or task groups."""

    viable = [pt for pt in points if pt.delta > max_overall_drop]
    if len(viable) < 2:
        return []
    horizons = sorted({h for pt in viable for h in pt.horizons})
    groups = sorted({g for pt in viable for g in pt.groups})
    scored: list[tuple[float, dict[str, Any]]] = []
    for i in range(len(viable)):
        for j in range(i + 1, len(viable)):
            a, b = viable[i], viable[j]
            if a.digest == b.digest:
                continue
            a_wins: list[tuple[str, float]] = []
            b_wins: list[tuple[str, float]] = []
            for h in horizons:
                if h in a.horizons and h in b.horizons:
                    diff = a.horizons[h] - b.horizons[h]
                    if diff >= min_niche_advantage:
                        a_wins.append((f"h={h}", diff))
                    elif -diff >= min_niche_advantage:
                        b_wins.append((f"h={h}", -diff))
            for g in groups:
                if g in a.groups and g in b.groups:
                    diff = a.groups[g] - b.groups[g]
                    if diff >= min_niche_advantage:
                        a_wins.append((f"组 {g}", diff))
                    elif -diff >= min_niche_advantage:
                        b_wins.append((f"组 {g}", -diff))
            if a_wins and b_wins:
                best_a = max(a_wins, key=lambda item: item[1])
                best_b = max(b_wins, key=lambda item: item[1])
                synergy = min(best_a[1], best_b[1]) + 0.25 * (a.delta + b.delta)
                scored.append((
                    synergy,
                    {
                        "a": a.label,
                        "b": b.label,
                        "a_strength": f"{best_a[0]}（+{best_a[1]:.4f}）",
                        "b_strength": f"{best_b[0]}（+{best_b[1]:.4f}）",
                        "synergy": round(synergy, 4),
                    },
                ))
    scored.sort(key=lambda item: (-item[0], item[1]["a"], item[1]["b"]))
    return [item for _, item in scored[:max_pairs]]


def dgm_parent_weights(
    points: Sequence[ArchivePoint],
    children_counts: Mapping[str, int] | None = None,
    *,
    base_weight: float = 0.02,
) -> list[dict[str, Any]]:
    """Darwin-Godel decaying exploration weights: ``quality / (1 + children)``.

    Stepping-stone versions that performed well yet have spawned few or zero children receive higher sampling
    weight than heavily exploited branches, discouraging premature single-trunk collapse.
    """

    if not points:
        return []
    counts = children_counts or {}
    floor = min(0.0, min(pt.delta for pt in points))
    raw: list[tuple[ArchivePoint, int, float]] = []
    for pt in points:
        children = max(0, int(counts.get(pt.digest, 0)))
        quality = max(base_weight, (pt.delta - floor) + base_weight)
        weight = quality / (1.0 + children)
        raw.append((pt, children, weight))
    total = sum(w for _, _, w in raw) or 1.0
    ranked = [
        {
            "label": pt.label,
            "digest": pt.digest,
            "delta": round(pt.delta, 6),
            "win_share": round(pt.win_share, 4),
            "children": children,
            "weight": round(weight, 6),
            "share": round(weight / total, 4),
        }
        for pt, children, weight in raw
    ]
    ranked.sort(key=lambda item: (-item["weight"], -item["delta"], item["label"]))
    return ranked


def operator_schedule_section(
    names: Sequence[str],
    *,
    points: Sequence[ArchivePoint] = (),
    children_counts: Mapping[str, int] | None = None,
    pairs: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Explicit orthogonal mutation roles across parallel candidate slots in one generation."""

    if len(names) < 2:
        return ""
    dgm = dgm_parent_weights(points, children_counts)
    unexplored = next((item for item in dgm if item["children"] == 0), dgm[0] if dgm else None)
    top_pair = pairs[0] if pairs else None
    lines = ["", "### 本代多候选正交分工建议（Operator Schedule）", ""]
    lines.append(
        f"- **候选 `{names[0]}`（定向开发 / Exploit）**：基于当前起点，优先针对 `evidence/triage.md` 排在首位的"
        "单点病灶做精准修复或单一机制升级，严格保留在任版本已验证有效的核心机制。"
    )
    if top_pair:
        explore_hint = (
            f"优先尝试帕累托交叉重组 `{top_pair['a']}`（强于 {top_pair['a_strength']}）× "
            f"`{top_pair['b']}`（强于 {top_pair['b_strength']}），抽取两者互补结构合并。"
        )
    elif unexplored:
        explore_hint = (
            f"优先从高权未充分探索的踏脚石 `{unexplored['label']}`（Δ {unexplored['delta']:+.4f}，"
            f"衍生 {unexplored['children']} 次）出发（`reset_candidate`），尝试不同模型族、分时距架构或特征范式。"
        )
    else:
        explore_hint = "尝试与第一个候选明显不同的结构性方案（不同模型族、分时距建模或新特征来源），避免只改同类超参。"
    lines.append(f"- **候选 `{names[1]}`（结构探索 / Explore or Crossover）**：{explore_hint}")
    for extra in names[2:]:
        lines.append(
            f"- **候选 `{extra}`（精简消融或护栏加固 / Ablation & Guardrail）**：清理 `evidence/triage.md` "
            "中零激活技能与冗余指令（反膨胀），或通过 `hooks/audit_submission.py` / `hooks/rescue.md` 加固提交门禁与预算节奏。"
        )
    return "\n".join(lines) + "\n"


def render_pareto_section(
    by_label: Mapping[str, dict[str, Any]],
    digests: Mapping[str, str],
    *,
    children_counts: Mapping[str, int] | None = None,
) -> str:
    """Markdown section summarizing the multi-objective Pareto frontier, DGM stepping stones, and crossover pairs."""

    points = extract_points(by_label, digests)
    if len(points) < 2:
        return ""
    front = pareto_front(points)
    leaders = niche_leaders(points)
    pairs = crossover_pairs(points)
    dgm = dgm_parent_weights(points, children_counts)
    lines = ["", "## 多目标帕累托前沿与互补组合（MAP-Elites / DGM / Crossover）", ""]
    front_text = "、".join(f"`{pt.label}`（Δ {pt.delta:+.4f}，胜率 {pt.win_share:.0%}）" for pt in front[:8])
    lines.append(f"- **非支配帕累托前沿**：{front_text}")
    if dgm:
        dgm_text = "、".join(
            f"`{item['label']}`（权重 {item['share']:.0%}，Δ {item['delta']:+.4f}，衍生 {item['children']} 次）"
            for item in dgm[:5]
        )
        lines.append(f"- **DGM 踏脚石父本推荐（`quality / (1 + children)` 衰减探索权）**：{dgm_text}")
    leader_bits = []
    for niche, pt in leaders.items():
        if niche == "overall":
            leader_bits.append(f"综合最优 `{pt.label}`（{pt.delta:+.4f}）")
        elif niche.startswith("h="):
            h = int(niche[2:])
            leader_bits.append(f"{niche} 最优 `{pt.label}`（{pt.horizons[h]:+.4f}）")
        elif niche.startswith("group:"):
            g = niche[6:]
            leader_bits.append(f"组 `{g}` 最优 `{pt.label}`（{pt.groups[g]:+.4f}）")
    if leader_bits:
        lines.append("- **分维度专长领跑者**：" + "；".join(leader_bits))
    if pairs:
        lines.append("- **推荐交叉重组父本对（Crossover）**（在不同时距或任务组上互补，可抽取各自有效结构合并）：")
        for pair in pairs:
            lines.append(
                f"  - `{pair['a']}`（强于 {pair['a_strength']}） × `{pair['b']}`（强于 {pair['b_strength']}）"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "ArchivePoint",
    "crossover_pairs",
    "dgm_parent_weights",
    "extract_points",
    "niche_leaders",
    "operator_schedule_section",
    "pareto_front",
    "render_pareto_section",
]
