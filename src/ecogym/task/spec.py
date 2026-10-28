"""Task cards: what to forecast, from what, and how success is judged.

A task card (TOML) binds a dataset to an objective grid (targets x horizons
in steps), an information set (history length and scope, admissible input
roles, forward-fill limits), a baseline menu, the partition protocol and the
promotion gates. Seed genomes ("presets") may be declared alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..data.spec import DatasetSpec
from ..kernel.canonical import digest
from ..kernel.config import ConfigError, Fields, load_toml

HISTORY_SCOPES = ("partition", "causal")
BASELINE_KINDS = ("persistence", "seasonal")


@dataclass(frozen=True, slots=True)
class Target:
    name: str
    weight: float
    lo: float | None = None
    hi: float | None = None


@dataclass(frozen=True, slots=True, order=True)
class Cell:
    target: str
    horizon: int

    @property
    def key(self) -> str:
        return f"{self.target}@{self.horizon}"


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    """Four-stage time-forward partitions (all durations in steps)."""

    kind: str = "four_stage"
    train_fraction: float = 0.6
    feedback_fraction: float = 0.5
    development_fraction: float = 0.2
    feedback_gap: int = 1
    embargo: int = 24
    calibration_percent: int = 70
    min_fit_origins: int = 80
    min_fit_blocks: int = 14
    min_uq_rows: int = 40
    min_uq_blocks: int = 8


@dataclass(frozen=True, slots=True)
class GateSpec:
    """Promotion gates; see ``ecogym.evaluation.gates``."""

    min_delta: float = 0.005
    confidence: float = 0.95
    resamples: int = 1000
    min_blocks: int = 8
    moving_block: int = 3
    min_starts: int = 4
    cell_alpha: float = 0.05
    cell_cap: float = 0.03


@dataclass(frozen=True, slots=True)
class TaskSpec:
    id: str
    title: str
    dataset: str
    episodes: tuple[str, ...]
    targets: tuple[Target, ...]
    horizons: tuple[int, ...]
    history_steps: int
    history_scope: str
    exogenous_roles: tuple[str, ...]
    fill_age: dict[str, int]
    min_coverage: float
    block_steps: int
    season_steps: int
    periods: tuple[int, ...]
    baselines: tuple[str, ...]
    tie_tolerance: float
    protocol: ProtocolSpec
    gates: GateSpec
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: str = ""
    knowledge: str = ""  # domain knowledge for research agents; does not affect scores
    digest: str = ""

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.targets)

    @property
    def cells(self) -> tuple[Cell, ...]:
        return tuple(Cell(target.name, horizon) for target in self.targets for horizon in self.horizons)

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def target(self, name: str) -> Target:
        for target in self.targets:
            if target.name == name:
                return target
        raise KeyError(f"task {self.id} has no target {name}")

    def cell_weight(self, cell: Cell) -> float:
        """Target weight times an equal share of the horizons."""

        return self.target(cell.target).weight / len(self.horizons)

    def identity(self) -> dict[str, Any]:
        """Everything that changes what a score means (presets excluded)."""

        return {
            "schema": "ecogym.task/1",
            "id": self.id,
            "dataset": self.dataset,
            "episodes": list(self.episodes),
            "targets": [[t.name, t.weight, t.lo, t.hi] for t in self.targets],
            "horizons": list(self.horizons),
            "history_steps": self.history_steps,
            "history_scope": self.history_scope,
            "exogenous_roles": list(self.exogenous_roles),
            "fill_age": self.fill_age,
            "min_coverage": self.min_coverage,
            "block_steps": self.block_steps,
            "season_steps": self.season_steps,
            "periods": list(self.periods),
            "baselines": list(self.baselines),
            "tie_tolerance": self.tie_tolerance,
            "protocol": self.protocol,
            "gates": self.gates,
        }


def _protocol(fields: Fields, horizon: int) -> ProtocolSpec:
    spec = ProtocolSpec(
        kind=fields.str("kind", "four_stage", choices=("four_stage",)),
        train_fraction=fields.float("train_fraction", 0.6, lo=0.0, hi=1.0),
        feedback_fraction=fields.float("feedback_fraction", 0.5, lo=0.0, hi=1.0),
        development_fraction=fields.float("development_fraction", 0.2, lo=0.0, hi=1.0),
        feedback_gap=fields.int("feedback_gap", 1, lo=0),
        embargo=fields.int("embargo", horizon, lo=0),
        calibration_percent=fields.int("calibration_percent", 70, lo=1, hi=99),
        min_fit_origins=fields.int("min_fit_origins", 80, lo=1),
        min_fit_blocks=fields.int("min_fit_blocks", 14, lo=1),
        min_uq_rows=fields.int("min_uq_rows", 40, lo=1),
        min_uq_blocks=fields.int("min_uq_blocks", 8, lo=1),
    )
    fields.done()
    if spec.train_fraction + spec.development_fraction >= 1:
        raise ConfigError(f"{fields.where}: train and development fractions must leave a final test")
    return spec


def _gates(fields: Fields) -> GateSpec:
    spec = GateSpec(
        min_delta=fields.float("min_delta", 0.005, lo=0.0),
        confidence=fields.float("confidence", 0.95, lo=0.5, hi=0.999),
        resamples=fields.int("resamples", 1000, lo=100),
        min_blocks=fields.int("min_blocks", 8, lo=3),
        moving_block=fields.int("moving_block", 3, lo=1),
        min_starts=fields.int("min_starts", 4, lo=1),
        cell_alpha=fields.float("cell_alpha", 0.05, lo=0.001, hi=0.5),
        cell_cap=fields.float("cell_cap", 0.03, lo=0.0),
    )
    fields.done()
    return spec


def load_task_spec(path: str | Path) -> TaskSpec:
    raw = load_toml(path)
    fields = Fields(raw, Path(path).name)
    targets = []
    for item in fields.tables("targets"):
        bounds = item.raw("bounds", None)
        if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 2):
            raise ConfigError(f"{item.where}.bounds must be [lo, hi]")
        targets.append(
            Target(
                name=item.str("name"),
                weight=item.float("weight", lo=0.0),
                lo=float(bounds[0]) if bounds else None,
                hi=float(bounds[1]) if bounds else None,
            )
        )
        item.done()
    horizons = fields.ints("horizons", lo=1)
    if not targets or not horizons or len(set(horizons)) != len(horizons):
        raise ConfigError(f"{fields.where}: needs targets and unique horizons")
    if sum(target.weight for target in targets) <= 0:
        raise ConfigError(f"{fields.where}: target weights must sum to a positive value")
    exogenous = fields.table("exogenous")
    fill_age = exogenous.table("max_fill_age")
    ages = {key: fill_age.int(key, lo=0) for key in fill_age}
    fill_age.done()
    roles = exogenous.strs("roles", ())
    min_coverage = exogenous.float("min_coverage", 0.2, lo=0.0, hi=1.0)
    exogenous.done()
    baseline = fields.table("baselines")
    baselines = baseline.strs("menu", BASELINE_KINDS)
    if not baselines or baselines[0] != "persistence" or not set(baselines) <= set(BASELINE_KINDS):
        raise ConfigError(f"{baseline.where}.menu must start with persistence and use {BASELINE_KINDS}")
    tie_tolerance = baseline.float("tie_tolerance", 1e-12, lo=0.0)
    baseline.done()
    presets_table = fields.table("presets")
    presets = {name: dict(presets_table.raw(name)) for name in presets_table}
    presets_table.done()
    max_horizon = max(horizons)
    spec = TaskSpec(
        id=fields.str("id"),
        title=fields.str("title"),
        dataset=fields.str("dataset"),
        episodes=fields.strs("episodes"),
        targets=tuple(targets),
        horizons=horizons,
        history_steps=fields.int("history_steps", lo=1),
        history_scope=fields.str("history_scope", "partition", choices=HISTORY_SCOPES),
        exogenous_roles=roles,
        fill_age=ages,
        min_coverage=min_coverage,
        block_steps=fields.int("block_steps", lo=1),
        season_steps=fields.int("season_steps", lo=1),
        periods=fields.ints("periods", (), lo=1),
        baselines=baselines,
        tie_tolerance=tie_tolerance,
        protocol=_protocol(fields.table("protocol"), max_horizon),
        gates=_gates(fields.table("gates")),
        presets=presets,
        notes=fields.raw("notes", ""),
        knowledge=_knowledge(Path(path), fields.strs("knowledge_files", ())),
    )
    fields.done()
    missing_ages = sorted(set(spec.exogenous_roles) - set(spec.fill_age))
    if missing_ages:
        raise ConfigError(f"{spec.id}: exogenous.max_fill_age lacks roles {missing_ages}")
    return _with_digest(spec)


def _knowledge(card: Path, files: tuple[str, ...]) -> str:
    """Concatenated knowledge notes, paths relative to the card."""

    texts = []
    for relative in files:
        path = (card.parent / relative).resolve()
        if not path.is_file():
            raise ConfigError(f"{card.name}: knowledge file {relative} not found")
        texts.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(texts)


def _with_digest(spec: TaskSpec) -> TaskSpec:
    return replace(spec, digest=digest(spec.identity()))


def check_task(spec: TaskSpec, dataset: DatasetSpec) -> None:
    """Fail if the card names columns or roles the dataset does not have."""

    if spec.dataset != dataset.id:
        raise ConfigError(f"task {spec.id} is for {spec.dataset}, not {dataset.id}")
    names = {column.name: column for column in dataset.columns}
    for target in spec.targets:
        if target.name not in names:
            raise ConfigError(f"task {spec.id}: target {target.name} is not a column of {dataset.id}")
    roles = {column.role for column in dataset.columns}
    unknown = sorted(set(spec.exogenous_roles) - roles)
    if unknown:
        raise ConfigError(f"task {spec.id}: dataset {dataset.id} has no roles {unknown}")
    if spec.history_steps <= 0 or spec.season_steps > spec.history_steps:
        raise ConfigError(f"task {spec.id}: history_steps must cover one season ({spec.season_steps})")
    if any(dataset.is_external(name) for name in spec.episodes):
        raise ConfigError(f"task {spec.id}: external episodes cannot be optimized on")


_TASKS: dict[str, TaskSpec] = {}


def register_task(path: str | Path) -> TaskSpec:
    spec = load_task_spec(path)
    existing = _TASKS.get(spec.id)
    if existing is not None and existing.digest != spec.digest:
        raise ValueError(f"task {spec.id} is registered twice with different cards")
    _TASKS[spec.id] = spec
    return spec


def task_spec(task_id: str) -> TaskSpec:
    try:
        return _TASKS[task_id]
    except KeyError:
        raise KeyError(f"unknown task {task_id}; known: {', '.join(sorted(_TASKS)) or 'none'}") from None


def task_specs() -> list[TaskSpec]:
    return [_TASKS[key] for key in sorted(_TASKS)]


__all__ = [
    "BASELINE_KINDS",
    "HISTORY_SCOPES",
    "Cell",
    "GateSpec",
    "ProtocolSpec",
    "Target",
    "TaskSpec",
    "check_task",
    "load_task_spec",
    "register_task",
    "task_spec",
    "task_specs",
]
