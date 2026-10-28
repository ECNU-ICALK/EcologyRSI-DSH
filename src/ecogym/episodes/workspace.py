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


def export_visible(bench: Bench, data_dir: Path) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
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
    spec = predictor_spec(bench)
    (data_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    background = [f"# 背景资料（任务提供）\n\n{task.notes.strip()}" if task.notes.strip() else "# 背景资料（任务提供）"]
    if task.knowledge.strip():
        background.append(task.knowledge.strip())
    (data_dir / "background.md").write_text("\n\n".join(background) + "\n", encoding="utf-8")
    coverage = {name: round(float(np.isfinite(column).mean()), 3) for name, column in zip(columns, values, strict=True)}
    (data_dir / "README.md").write_text(_data_readme(bench, bounds, int(keep.sum()), coverage), encoding="utf-8")
    return {"rows": int(keep.sum()), "columns": list(columns), "boundaries": bounds}


def _data_readme(bench: Bench, bounds: dict[str, Any], rows: int, coverage: dict[str, float]) -> str:
    frame = bench.frame
    lines = [
        "# 可见数据",
        "",
        f"- `history.csv`：{rows} 行，时间步长 {frame.step}，"
        f"从 {bounds['visible'][0]} 到 {bounds['visible'][1]}（不含）。"
        "每个时间步一行，缺失值为空。",
        "- `columns.csv`：每列的含义、角色、单位、有效范围，以及读数保持可用的最长步数（用于前向填充）。",
        "- `spec.json`：裁判传给 `Predictor(spec)` 的同一份说明。",
        "- `background.md`：任务提供的背景资料。",
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
               limits: Limits) -> str:
    task = bench.task
    targets = "、".join(f"`{target}`（{bench.frame.specs[target].label}，"
                        f"{UNIT_LABELS.get(bench.frame.specs[target].unit, bench.frame.specs[target].unit)}）"
                        for target in task.target_names)
    step = bench.frame.step
    horizons = "、".join(f"{h} 步" for h in task.horizons)
    references = reference_scores(bench, LOCAL_STAGE)
    reference_text = "；".join(f"{name} {value:+.3f}" for name, value in references.items())
    kinds = "，".join(f"{cell} 用{'季节值' if kind == 'seasonal' else '持续性'}"
                     for cell, kind in bench.baselines.kinds.items())
    ends = time.strftime("%Y-%m-%d %H:%M", time.localtime(deadline))
    skill_line = (f"- 每个目标与时距组成一个格，格内技能 = (RMSE基线 − RMSE) / max(RMSE, RMSE基线)，"
                  f"大于 0 表示优于基线。基线是在拟合期上逐格选定的朴素预报：{kinds}。")
    scored_line = (f"- **正式成绩**来自可见数据之后的隐藏时期，约 {bounds['scored_days']} 天，你看不到这段数据。"
                   f"裁判先用隐藏时期之前的全部数据调用 `fit`，然后逐个时间步送入新观测，"
                   f"在每个计分起点调用 `forecast`。")
    submit_line = ("- 提交内容是 `submission/` 目录：`predictor.py` 中的 `Predictor` 类，接口见文件头；"
                   "同目录下的其他 Python 模块可以导入（如 `ecotk/`）。")
    limit_line = (f"- 时限：`fit` ≤ {limits.fit_seconds:.0f} 秒；"
                  f"单次 `forecast` ≤ {limits.forecast_seconds:.0f} 秒；"
                  f"拟合加全部预报 ≤ {limits.total_seconds:.0f} 秒。结果要确定（固定随机种子）。")
    return f"""# 任务：{task.title}

## 目标

在每个预报起点 t，预测 {targets} 在 t+h 的值，h = {horizons}（时间步长 {step}）。

## 评分

{skill_line}
- 总分是各格技能的加权平均（各目标等权，每个目标的时距等权）。缺失的预报按 −1 计入；预报会被裁剪到目标的物理范围。
{scored_line}
- 本地检验期上的参考成绩（宿主内置方法，只用拟合期拟合）：{reference_text}。

## 数据与信息集

- 数据都在 `data/` 下，先读 `data/README.md`。
- 预报只能使用起点及以前的观测。`action` 角色的列（控制设定值）在起点已知；其余输入列也只用到起点为止。
- 工作区以外的文件和网络都不可访问；`web_search` 可以用来查资料。

## 提交

{submit_line}
- 运行环境：Python 3.12，已装 numpy、pandas、scipy、scikit-learn、statsmodels；不能安装其他包，不能联网。
{limit_line}
- `submission/` 最多 20 MB、300 个文件，不能有符号链接。

## ecogym 工具（MCP）

- `task_info`：剩余时间与 token、已提交的版本、最好的本地成绩。
- `evaluate_local`：按裁判协议在本地检验期上运行 `submission/`，返回总分和逐格技能。
- `submit`：保存当前 `submission/` 的快照并在本地检验期上运行；以最后一次成功的提交为准。
- `note`：记录发现、经验、无效方向（kind = finding / lesson / dead_end）。
- `ask_expert`：给领域专家留言；不会立即回答，不要等待。

## 预算

{minutes:.0f} 分钟，截止 {ends}；最多 {tokens:,} 个 token。到截止时间时，进程会被终止，以最后一次成功的提交为准。

## 最终回复

用中文写：最终方法、本地成绩（总分与逐格）、有效与无效的尝试、给后续回合的经验。
"""


__all__ = ["LOCAL_STAGE", "SCORED_STAGE", "boundaries", "export_visible", "task_brief"]
