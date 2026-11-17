"""Optional, local-only calls to the task's classical forecasting references."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..evaluation.harness import Bench, Evaluation
from ..protocol.baselines import BaselineProfile, baseline_values
from ..solvers import Genome
from .mcp import Tool, ToolError
from .workspace import LOCAL_STAGE

BASELINES = ("selected_naive", "persistence", "seasonal_naive")


def _reference(bench: Bench, name: str) -> Evaluation | None:
    if name.startswith("preset:"):
        preset = name.removeprefix("preset:")
        return bench.evaluate(bench.preset(preset), LOCAL_STAGE) if preset in bench.task.presets else None

    stage = bench.stage(LOCAL_STAGE)
    if name == "selected_naive":
        predictions = stage.baseline
    else:
        seasonal = name == "seasonal_naive"
        kinds = {cell.key: "seasonal" if seasonal and cell.horizon <= bench.task.season_steps else "persistence"
                 for cell in stage.cells}
        values = baseline_values(stage.origins.without_labels(), bench.task, BaselineProfile(kinds, {}, ""))
        predictions = np.stack([values[cell.key] for cell in stage.cells])
    return Evaluation(Genome(name), stage, predictions, np.zeros(len(stage.cells), dtype=int))


def classical_reference_tool(benches: Sequence[Bench], *, check_open: Callable[[], None]) -> Tool:
    """List available references or run one against local origins. Return aggregate metrics only."""

    def call(name: str = "") -> dict[str, Any]:
        check_open()
        catalogs = {bench.task.id: [*BASELINES, *(f"preset:{key}" for key in sorted(bench.task.presets))]
                    for bench in benches}
        if not name:
            return {"stage": LOCAL_STAGE, "methods": catalogs}
        if not any(name in methods for methods in catalogs.values()):
            raise ToolError("unknown reference method; omit name to list the available methods")
        rows = {}
        for bench in benches:
            check_open()
            evaluation = _reference(bench, name)
            if evaluation is None:
                rows[bench.task.id] = {"status": "unavailable"}
                continue
            rows[bench.task.id] = {
                "status": "ok", "solver": evaluation.genome.solver, "score": evaluation.score(),
                "origins": evaluation.stage.size,
                "cells": {row["cell"]: {"objective": row["objective"], "n": row["n"]} for row in evaluation.cells()},
            }
        return {"stage": LOCAL_STAGE, "name": name, "fit_scope": "calibration_fit", "tasks": rows}

    return Tool("classical_reference",
                "列出持久、季节朴素、按拟合期选出的朴素基线，以及任务登记的经典方法。填 name 可试算一个方法，"
                "只返回本地检验期的分数和各格覆盖。登记预设只在 calibration_fit 拟合。工具不会提交或改动候选。",
                {"type": "object", "properties": {"name": {"type": "string"}}}, call)
