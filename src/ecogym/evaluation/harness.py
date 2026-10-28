"""The bench: one task's data, partitions and comparators, ready to score methods.

``Bench.open(task)`` loads the dataset, cuts the partitions, selects the
naive baselines on calibration_fit and fixes the eligible origins of each
stage. ``evaluate(genome, stage)`` fits the genome's solver on
calibration_fit, forecasts the stage's eligible origins (the solver sees no
labels), clips forecasts to the targets' physical bounds and returns an
``Evaluation`` whose errors any cohort subset can be scored from. Fits and
development-stage evaluations are cached by genome digest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..data.catalog import dataset_spec, load_frame
from ..data.frame import Frame
from ..kernel.canonical import digest
from ..protocol.baselines import BaselineProfile, baseline_values, select_baselines
from ..protocol.origins import blocks, eligible
from ..protocol.partitions import SEALED_STAGES, Partitions, four_stage
from ..protocol.views import Origins, SealedAccess, open_stage
from ..solvers.base import Genome, SolverContext, SolverError, get_solver
from ..task.spec import Cell, TaskSpec, task_spec
from .scoring import BlockSums, block_sums, cell_objectives, objective, skill

SCALE_FLOOR = 1e-6


@dataclass(frozen=True, slots=True, eq=False)
class StageData:
    """Eligible origins of one stage with labels and naive forecasts, shape (cells, origins)."""

    name: str
    origins: Origins
    cells: tuple[Cell, ...]
    observed: np.ndarray
    baseline: np.ndarray
    blocks: np.ndarray
    scales: np.ndarray
    weights: np.ndarray

    @property
    def size(self) -> int:
        return self.origins.size

    @property
    def time(self) -> np.ndarray:
        return self.origins.time


@dataclass(frozen=True, slots=True, eq=False)
class Evaluation:
    genome: Genome
    stage: StageData
    predicted: np.ndarray
    clipped: np.ndarray
    fit: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> np.ndarray:
        return self.predicted - self.stage.observed

    def sums(self, subset: np.ndarray | None = None) -> BlockSums:
        index = slice(None) if subset is None else subset
        stage = self.stage
        return block_sums(self.errors[:, index], stage.baseline[:, index] - stage.observed[:, index],
                          stage.blocks[index], stage.scales)

    def score(self, subset: np.ndarray | None = None) -> float:
        return float(objective(self.sums(subset).total(), self.stage.weights)[0])

    def cells(self, subset: np.ndarray | None = None) -> list[dict[str, Any]]:
        """Per-cell metrics in physical units."""

        index = slice(None) if subset is None else subset
        stage = self.stage
        rows = []
        objectives = cell_objectives(self.sums(subset).total())[:, 0]
        for position, cell in enumerate(stage.cells):
            error = self.errors[position, index]
            base = (stage.baseline[position, index] - stage.observed[position, index])
            forecast = np.isfinite(error)
            n = int(forecast.sum())
            rmse = float(np.sqrt(np.mean(error[forecast] ** 2))) if n else float("nan")
            base_rmse = float(np.sqrt(np.mean(base[forecast] ** 2))) if n else float("nan")
            rows.append({
                "cell": cell.key,
                "n": n,
                "eligible": int(error.shape[0]),
                "rmse": rmse,
                "baseline_rmse": base_rmse,
                "mae": float(np.mean(np.abs(error[forecast]))) if n else float("nan"),
                "bias": float(np.mean(error[forecast])) if n else float("nan"),
                "nrmse": rmse / float(stage.scales[position]),
                "skill": float(skill(np.array(rmse), np.array(base_rmse))) if n else float("nan"),
                "objective": float(objectives[position]),
                "clipped": int(self.clipped[position]),
            })
        return rows


class Bench:
    def __init__(self, task: TaskSpec, full: Frame, *, optimization: list[str], dataset_digest: str = "") -> None:
        """``full`` holds every episode; the partition calendar spans ``optimization``."""

        self.task = task
        self.dataset_digest = dataset_digest
        self.partitions: Partitions = four_stage(full, task, optimization)
        self.frame = full.select(task.episodes)
        self.fit_data = open_stage(self.frame, self.partitions, task, "calibration_fit")
        self.baselines: BaselineProfile = select_baselines(self.fit_data, task)
        self.context = SolverContext(task, self.baselines)
        fit_mask = self.partitions.mask(self.frame, "calibration_fit")
        self.scales = {}
        for name in task.target_names:
            values = self.frame.columns[name][fit_mask]
            values = values[np.isfinite(values)]
            self.scales[name] = max(float(np.sqrt(np.mean((values - values.mean()) ** 2))), SCALE_FLOOR)
        self._stages: dict[str, StageData] = {}
        self._fits: dict[str, Any] = {}
        self._evaluations: dict[tuple[str, str], Evaluation] = {}

    @classmethod
    def open(cls, task: str | TaskSpec, *, root: Path | None = None) -> Bench:
        spec = task if isinstance(task, TaskSpec) else task_spec(task)
        dataset = dataset_spec(spec.dataset)
        full = load_frame(dataset, root=root)
        optimization = [name for name in full.episode_ids if not dataset.is_external(name)]
        return cls(spec, full, optimization=optimization, dataset_digest=dataset.digest)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "task": self.task.digest,
            "dataset": self.dataset_digest,
            "frame": self.frame.digest,
            "partitions": self.partitions.digest,
            "baselines": self.baselines.digest,
        }

    @property
    def digest(self) -> str:
        return digest(self.identity)

    def genome(self, body: dict[str, Any]) -> Genome:
        return Genome.from_dict(body, self.task)

    def preset(self, name: str) -> Genome:
        try:
            body = self.task.presets[name]
        except KeyError:
            raise KeyError(f"task {self.task.id} has no preset {name}; presets: {sorted(self.task.presets)}") from None
        return self.genome({"solver": body["solver"], "params": body.get("params", {})})

    def stage(self, name: str, *, sealed: SealedAccess | None = None) -> StageData:
        if name in self._stages:
            return self._stages[name]
        opened = open_stage(self.frame, self.partitions, self.task, name, sealed=sealed)
        origins = opened.take(eligible(opened, self.task))
        cells = self.task.cells
        observed = np.stack([origins.label(cell.target, cell.horizon) for cell in cells])
        naive = baseline_values(origins, self.task, self.baselines)
        data = StageData(
            name=name,
            origins=origins,
            cells=cells,
            observed=observed,
            baseline=np.stack([naive[cell.key] for cell in cells]),
            blocks=blocks(origins.time, self.task.block_steps),
            scales=np.array([self.scales[cell.target] for cell in cells]),
            weights=np.array([self.task.cell_weight(cell) for cell in cells]),
        )
        if name not in SEALED_STAGES:
            self._stages[name] = data
        return data

    def fitted(self, genome: Genome) -> Any:
        if genome.digest not in self._fits:
            spec = get_solver(genome.solver)
            self._fits[genome.digest] = spec.fit(self.fit_data, self.context, genome.params)
        return self._fits[genome.digest]

    def evaluate(self, genome: Genome, stage: str, *, sealed: SealedAccess | None = None) -> Evaluation:
        key = (genome.digest, stage)
        if key in self._evaluations:
            return self._evaluations[key]
        spec = get_solver(genome.solver)
        state = self.fitted(genome)
        data = self.stage(stage, sealed=sealed)
        forecasts = spec.predict(state, data.origins.without_labels(), self.context)
        rows = []
        clipped = []
        for cell in data.cells:
            values = np.asarray(forecasts.get(cell.key, np.full(data.size, np.nan)), dtype=np.float64)
            if values.shape != (data.size,):
                raise SolverError(
                    f"{genome.solver} returned {values.shape} forecasts for {cell.key}, expected {data.size}"
                )
            target = self.task.target(cell.target)
            low = -np.inf if target.lo is None else target.lo
            high = np.inf if target.hi is None else target.hi
            bounded = np.clip(values, low, high)
            clipped.append(int(np.sum(np.isfinite(values) & (bounded != values))))
            rows.append(bounded)
        describe = spec.describe_fit(state) if spec.describe_fit else {}
        evaluation = Evaluation(genome, data, np.stack(rows), np.array(clipped), describe)
        if stage not in SEALED_STAGES:
            self._evaluations[key] = evaluation
        return evaluation


__all__ = ["Bench", "Evaluation", "StageData"]
