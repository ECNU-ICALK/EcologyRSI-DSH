"""The agent's workspace for one episode: the task contract and only the visible data.

Visible data are the task episode's rows before the scored stage starts
(model_selection): calibration_fit, calibration_uq and the gaps between them,
restricted to the targets and the input columns the task admits. Nothing from
other episodes, the scored stage or the sealed stages is written. The local
check (``evaluate_local``) fits on the rows before calibration_uq and streams
calibration_uq, exactly as the judge streams the scored stage.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.harness import Bench
from ..kernel import clock
from .judge import Limits, episode_of, iso, predictor_spec, reference_scores, visible_columns

SCORED_STAGE = "model_selection"
LOCAL_STAGE = "calibration_uq"
UNIT_LABELS = {"degC": "°C", "percent": "%"}


def boundaries(bench: Bench) -> dict[str, Any]:
    episode = episode_of(bench)
    step = bench.frame.step
    intervals = {stage: bench.partitions.intervals[stage][episode] for stage in bench.partitions.intervals}
    rows = bench.frame.episode_rows(episode)
    first = int(bench.frame.time[rows][0])
    visible_end = intervals[SCORED_STAGE][0]
    local_start, local_end = intervals[LOCAL_STAGE]
    scored_start, scored_end = intervals[SCORED_STAGE]
    per_day = 24 if step == "1h" else 1
    stamp = lambda value: iso(np.array([value]), step)[0]  # noqa: E731
    return {
        "episode": episode,
        "visible": [stamp(first), stamp(visible_end)],
        "local_fit_end": stamp(local_start),
        "local_check": [stamp(local_start), stamp(local_end)],
        "scored": [stamp(scored_start), stamp(scored_end)],
        "scored_days": round((scored_end - scored_start) / per_day, 1),
        "visible_end_step": visible_end,
    }


def _autocorr(series: np.ndarray, lag: int) -> float | None:
    if lag <= 0 or lag >= series.size:
        return None
    a, b = series[:-lag], series[lag:]
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 8:
        return None
    x, y = a[mask], b[mask]
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def data_profile(
    bench: Bench,
    bounds: dict[str, Any],
    columns: list[str],
    values: list[np.ndarray],
    *,
    start_report: dict[str, Any] | None = None,
) -> str:
    """Precomputed visible-data statistics and starting-template local diagnostics (zero hidden data)."""

    task = bench.task
    season = int(task.season_steps)
    lines = [
        "# 预计算数据与基线画像（仅可见期）",
        "",
        f"自动生成于回合启动前（共 {values[0].size if values else 0} 行，"
        f"步长 `{bench.frame.step}`，季节周期 `{season}` 步；"
        f"本地检验 `{bounds['local_check'][0]}` → `{bounds['local_check'][1]}`）。"
        "可直接据此跳过基础探索，优先切入建模与特征工程。",
        "",
        "## 列分布与自相关（仅可见数据）",
        "",
        "| 列 | 角色 | 有效率 | 均值 | 标准差 | P5 / P50 / P95 | Lag-1 自相关 | 季节 Lag 自相关 |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for name, arr in zip(columns, values, strict=True):
        finite = arr[np.isfinite(arr)]
        rate = float(finite.size / arr.size) if arr.size else 0.0
        spec = bench.frame.specs[name]
        role = f"{spec.role}*" if name in task.target_names else spec.role
        if finite.size == 0:
            lines.append(f"| `{name}` | {role} | 0% | – | – | – | – | – |")
            continue
        p5, p50, p95 = (float(v) for v in np.percentile(finite, [5, 50, 95]))
        ac1 = _autocorr(arr, 1) if name in task.target_names else None
        acs = _autocorr(arr, season) if name in task.target_names and season > 1 else None
        ac1_s = f"{ac1:+.3f}" if ac1 is not None else "–"
        acs_s = f"{acs:+.3f}" if acs is not None else "–"
        lines.append(
            f"| `{name}` | {role} | {rate:.0%} | {float(np.mean(finite)):.3g} | "
            f"{float(np.std(finite)):.3g} | {p5:.3g} / {p50:.3g} / {p95:.3g} | {ac1_s} | {acs_s} |"
        )
    if start_report and start_report.get("cells"):
        cells = [cell for cell in start_report["cells"] if isinstance(cell.get("skill"), (int, float))]
        score = start_report.get("score")
        score_s = f"{float(score):+.4f}" if isinstance(score, (int, float)) else "–"
        lines += ["", f"## 起始模板本地检验短板（总分 {score_s}，状态 `{start_report.get('status')}`）", ""]
        by_h: dict[int, list[float]] = {}
        for cell in cells:
            _, _, hz = str(cell.get("cell", "")).partition("@")
            if hz.isdigit():
                by_h.setdefault(int(hz), []).append(float(cell["skill"]))
        if by_h:
            h_parts = [f"h={h}: `{sum(vals) / len(vals):+.4f}`" for h, vals in sorted(by_h.items())]
            lines.append("- **分时距均值**：" + " · ".join(h_parts))
        weakest = sorted(cells, key=lambda item: float(item["skill"]))[:4]
        if weakest:
            lines.append("- **最弱计分格（优先攻克）**：")
            for cell in weakest:
                lines.append(
                    f"  - `{cell['cell']}`：技能 `{float(cell['skill']):+.4f}`"
                    f"（RMSE `{float(cell.get('rmse', 0)):.4g}` vs 基线 `{float(cell.get('baseline_rmse', 0)):.4g}`）"
                )
    return "\n".join(lines) + "\n"


def export_visible(bench: Bench, data_dir: Path, *, expert: str = "",
                   start_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the visible data, its documentation, precomputed profile, and (if any) expert guidance."""

    data_dir.mkdir(parents=True, exist_ok=True)
    if expert.strip():
        (data_dir / "expert.md").write_text(expert.strip() + "\n", encoding="utf-8")
    bounds = boundaries(bench)
    rows = bench.frame.episode_rows(bounds["episode"])
    times = bench.frame.time[rows]
    keep = times < bounds["visible_end_step"]
    columns = visible_columns(bench)
    stamps = iso(times[keep], bench.frame.step)
    values = [bench.frame.columns[name][rows][keep] for name in columns]
    with (data_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", *columns])
        for index, stamp in enumerate(stamps):
            cells = (f"{column[index]:.6g}" if np.isfinite(column[index]) else "" for column in values)
            writer.writerow([stamp, *cells])
    task = bench.task
    with (data_dir / "columns.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", "label", "role", "unit", "target", "valid_min", "valid_max", "max_fill_age_steps"])
        for name in columns:
            spec = bench.frame.specs[name]
            writer.writerow([name, spec.label, spec.role, spec.unit, name in task.target_names,
                             "" if spec.valid_min is None else spec.valid_min,
                             "" if spec.valid_max is None else spec.valid_max,
                             task.fill_age.get(spec.role, "")])
    spec = {
        **predictor_spec(bench),
        "local_fit_end": bounds["local_fit_end"],
        "target_weights": {target.name: float(target.weight) for target in task.targets},
    }
    (data_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    background = [f"# 背景资料（任务提供）\n\n{task.notes.strip()}" if task.notes.strip() else "# 背景资料（任务提供）"]
    if task.knowledge.strip():
        background.append(task.knowledge.strip())
    (data_dir / "background.md").write_text("\n\n".join(background) + "\n", encoding="utf-8")
    coverage = {name: round(float(np.isfinite(column).mean()), 3) for name, column in zip(columns, values, strict=True)}
    readme = _data_readme(bench, bounds, int(keep.sum()), coverage, expert=bool(expert.strip()))
    (data_dir / "README.md").write_text(readme, encoding="utf-8")
    profile_text = data_profile(bench, bounds, list(columns), values, start_report=start_report)
    (data_dir / "profile.md").write_text(profile_text, encoding="utf-8")
    return {"rows": int(keep.sum()), "columns": list(columns), "boundaries": bounds, "expert": bool(expert.strip())}


def _data_readme(bench: Bench, bounds: dict[str, Any], rows: int, coverage: dict[str, float], *,
                 expert: bool = False) -> str:
    frame = bench.frame
    lines = [
        "# 可见数据",
        "",
        f"- `history.csv`：{rows} 行，时间步长 {frame.step}，"
        f"从 {bounds['visible'][0]} 到 {bounds['visible'][1]}（不含）。"
        "每个时间步一行，缺失值为空。",
        "- `columns.csv`：每列的含义、角色、单位、有效范围，以及读数保持可用的最长步数（用于前向填充）。",
        "- `spec.json`：裁判传给 `Predictor(spec)` 的同一份说明。",
        "- `profile.md`：预计算的列分布、自相关与起始模板本地弱项画像（可直接参考，免去重复写基础 EDA）。",
        "- `background.md`：任务提供的背景资料。",
        *(["- `expert.md`：领域专家的提示，以及对以往回合中问题的回答。"] if expert else []),
        "",
        f"本地检验（`evaluate_local`）：拟合用 {bounds['local_fit_end']} 之前的行，检验期为 "
        f"{bounds['local_check'][0]} 至 {bounds['local_check'][1]}。",
        f"隐藏的评分时期从 {bounds['scored'][0]} 开始，约 {bounds['scored_days']} 天，这里没有它的数据。",
        "",
        "日表和周表的数值已按可得时间对齐：一行里出现的值，在这一时刻确实已经可知。",
        "",
        "| 列 | 含义 | 角色 | 单位 | 覆盖率 |",
        "|---|---|---|---|---|",
    ]
    for name, share in coverage.items():
        spec = frame.specs[name]
        mark = "（目标）" if name in bench.task.target_names else ""
        unit = UNIT_LABELS.get(spec.unit, spec.unit)
        lines.append(f"| `{name}` | {spec.label}{mark} | {spec.role} | {unit} | {share:.0%} |")
    return "\n".join(lines) + "\n"


def task_brief(bench: Bench, bounds: dict[str, Any], *, minutes: float, tokens: int, deadline: float,
               limits: Limits, start_local: float | None = None, selection_policy: str | None = None) -> str:
    """Describe the task, including the active submission policy and any starting score."""

    from .tools import (  # tools imports this module
        DEFAULT_LOCAL_SELECTION,
        SINGLE_WINDOW_SELECTION,
        selection_rule_text,
    )

    task = bench.task
    known_now = ("`action` 角色的列（控制设定值等）在起点已知；其余输入列也只用到起点为止。"
                 if "action" in task.exogenous_roles else "输入列只用到起点为止。")
    targets = "、".join(f"`{target}`（{bench.frame.specs[target].label}，"
                        f"{UNIT_LABELS.get(bench.frame.specs[target].unit, bench.frame.specs[target].unit)}）"
                        for target in task.target_names)
    step = bench.frame.step
    horizons = "、".join(f"{h} 步" for h in task.horizons)
    references = reference_scores(bench, LOCAL_STAGE)
    reference_text = "；".join(f"{name} {value:+.3f}" for name, value in references.items())
    kinds = "，".join(f"{cell} 用{'季节值' if kind == 'seasonal' else '持续性'}"
                     for cell, kind in bench.baselines.kinds.items())
    ends = time.strftime("%Y-%m-%d %H:%M", time.localtime(clock.wall(deadline)))
    skill_line = (f"- 每个目标与时距组成一个格，格内技能 = (RMSE基线 − RMSE) / max(RMSE, RMSE基线)，"
                  f"大于 0 表示优于基线。基线是在拟合期上逐格选定的朴素预报：{kinds}。")
    scored_line = (f"- **正式成绩**来自可见数据之后的隐藏时期，约 {bounds['scored_days']} 天，你看不到这段数据。"
                   f"裁判先用隐藏时期之前的全部数据调用 `fit`，然后逐个时间步送入新观测，"
                   f"每收到一个时间步就调用 `forecast`，宿主按规定的起点计分。")
    submit_line = ("- 提交内容是 `submission/` 目录：`predictor.py` 中的 `Predictor` 类，接口见文件头；"
                   "同目录下的其他 Python 模块可以导入（如 `ecotk/`）。")
    limit_line = (f"- 时限：`fit` ≤ {limits.fit_seconds:.0f} 秒；"
                  f"单次 `forecast` ≤ {limits.forecast_seconds:.0f} 秒；"
                  f"拟合加全部预报 ≤ {limits.total_seconds:.0f} 秒。结果要确定（固定随机种子）。")
    policy = selection_policy or DEFAULT_LOCAL_SELECTION
    rule = "按上面的规则评分"
    if start_local is None:
        rule_line = ("- **评分用哪个程序**：没有起始程序。"
                     + ("以最后一次成功的提交为准。\n" if policy == SINGLE_WINDOW_SELECTION
                        else "在完成本地检验且覆盖完整的提交中，使用本地总分最高者。\n"))
    else:
        rule_line = (f"- **评分用哪个程序**：`submission/` 里原有的程序是起始程序，"
                     f"其本地总分为 {start_local:+.4f}。{selection_rule_text(policy)}"
                     "本地检验期比隐藏期短得多，本地只高一点的改动常常不能在隐藏期上成立。\n")
    return f"""# 任务：{task.title}

## 目标

在每个预报起点 t，预测 {targets} 在 t+h 的值，h = {horizons}（时间步长 {step}）。

## 评分

{skill_line}
- 总分是各格技能的加权平均（各目标等权，每个目标的时距等权）。缺失的预报按 −1 计入；预报会被裁剪到目标的物理范围。
{scored_line}
- 本地检验期上的参考成绩（宿主内置方法，只用拟合期拟合）：{reference_text}。

## 数据与信息集

- 数据都在 `data/` 下，先读 `data/README.md` 和 `data/profile.md`（已预计算各列分布与起始模板弱项）。
- 预报只能使用起点及以前的观测。{known_now}
- 工作区以外的文件和网络都不可访问；`web_search` 可以用来查资料。

## 提交

{submit_line}
{rule_line}- 运行环境：Python 3.12，已装 numpy、pandas、scipy、scikit-learn、statsmodels；不能安装其他包，不能联网。
{limit_line}
- `submission/` 最多 20 MB、300 个文件，不能有符号链接。

## ecogym 工具（MCP）

- `task_info`：剩余时间与 token、已提交的版本、最好的本地成绩、现在评分用哪个程序。
- `classical_reference`：列出持久、季节朴素和任务登记的经典方法；填 `name` 试算一个，只返回本地检验期的汇总成绩。
- `evaluate_local`：按裁判协议在本地检验期上运行 `submission/`，返回总分和逐格技能。
- `submit`：保存当前 `submission/` 的快照并在本地检验期上运行；{rule}。
- `note`：记录发现、经验、无效方向（kind = finding / lesson / dead_end）。
- `ask_expert`：给领域专家留言；不会立即回答，不要等待。回答会出现在以后回合的 `data/expert.md` 里。

## 预算

{minutes:.0f} 分钟，截止约 {ends}（机器休眠的时间不计，剩余时间以 `task_info` 为准）；
最多 {tokens:,} 个计费 token（命中缓存的输入 token 按 1/10 计）。
到截止时间时，进程会被终止，{rule}。

## 最终回复

用中文写：最终方法、本地成绩（总分与逐格）、有效与无效的尝试、给后续回合的经验。
"""


__all__ = ["LOCAL_STAGE", "SCORED_STAGE", "boundaries", "data_profile", "export_visible", "task_brief"]
