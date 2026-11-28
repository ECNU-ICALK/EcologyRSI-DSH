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


def render_pareto_section(
    by_label: Mapping[str, dict[str, Any]],
    digests: Mapping[str, str],
) -> str:
    """Markdown section summarizing the multi-objective Pareto frontier and crossover pairs."""

    points = extract_points(by_label, digests)
    if len(points) < 2:
        return ""
    front = pareto_front(points)
    leaders = niche_leaders(points)
    pairs = crossover_pairs(points)
    lines = ["", "## 多目标帕累托前沿与互补组合（MAP-Elites / Crossover）", ""]
    front_text = "、".join(f"`{pt.label}`（Δ {pt.delta:+.4f}，胜率 {pt.win_share:.0%}）" for pt in front[:8])
    lines.append(f"- **非支配帕累托前沿**：{front_text}")
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
    "extract_points",
    "niche_leaders",
    "pareto_front",
    "render_pareto_section",
]
