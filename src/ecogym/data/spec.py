"""Dataset specs: what a dataset contains and how to read it.

A spec is a TOML file shipped by a domain pack. It names the columns (with
role, unit and source aliases), the source tables and how they are timed, and
which episodes are external (never used for optimization). It says nothing
about forecasting; that is the task card's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..kernel.canonical import digest
from ..kernel.config import ConfigError, Fields, load_toml

STEPS = {"1h": 24, "1d": 1}  # step name -> steps per day


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    name: str
    role: str
    unit: str
    label: str = ""
    aliases: tuple[str, ...] = ()
    required: bool = False
    valid_min: float | None = None
    valid_max: float | None = None

    def admits(self, value: float) -> bool:
        return not (
            (self.valid_min is not None and value < self.valid_min)
            or (self.valid_max is not None and value > self.valid_max)
        )


@dataclass(frozen=True, slots=True)
class TableSpec:
    file: str
    columns: tuple[str, ...]
    time_columns: tuple[str, ...]
    scope: str = "episode"  # "episode": one file per episode directory; "dataset": shared
    interval_minutes: int = 5
    delay_steps: int = 0  # availability delay: a value is usable this many steps later
    allow_suffix: bool = False  # alias "x" also matches "x_1", "x_2", ... (averaged)
    time_format: str = "auto"  # "auto" (serial days or dates) or "iso_week"
    week_year: int | None = None


@dataclass(frozen=True, slots=True)
class DerivedSpec:
    name: str
    op: str  # "sum" or "difference"
    inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    id: str
    title: str
    source: str
    path: str
    loader: str
    step: str
    episodes: str
    external_episodes: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    tables: tuple[TableSpec, ...]
    derived: tuple[DerivedSpec, ...]
    origin: str
    digest: str

    @property
    def steps_per_day(self) -> int:
        return STEPS[self.step]

    def column(self, name: str) -> ColumnSpec:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"dataset {self.id} has no column {name}")

    def is_external(self, episode: str) -> bool:
        lowered = episode.casefold()
        return any(pattern.casefold() in lowered for pattern in self.external_episodes)


def _column(fields: Fields) -> ColumnSpec:
    valid = fields.raw("valid", None)
    if valid is not None and (
        not isinstance(valid, list) or len(valid) != 2 or not all(isinstance(v, (int, float)) for v in valid)
    ):
        raise ConfigError(f"{fields.where}.valid must be [min, max]")
    column = ColumnSpec(
        name=fields.str("name"),
        role=fields.str("role"),
        unit=fields.str("unit"),
        label=fields.raw("label", ""),
        aliases=fields.strs("aliases", ()),
        required=fields.bool("required", False),
        valid_min=float(valid[0]) if valid else None,
        valid_max=float(valid[1]) if valid else None,
    )
    fields.done()
    return column


def _table(fields: Fields) -> TableSpec:
    table = TableSpec(
        file=fields.str("file"),
        columns=fields.strs("columns"),
        time_columns=fields.strs("time"),
        scope=fields.str("scope", "episode", choices=("episode", "dataset")),
        interval_minutes=fields.int("interval_minutes", 5, lo=1),
        delay_steps=fields.int("delay", 0, lo=0),
        allow_suffix=fields.bool("allow_suffix", False),
        time_format=fields.str("time_format", "auto", choices=("auto", "iso_week")),
        week_year=fields.raw("week_year", None),
    )
    fields.done()
    if table.time_format == "iso_week" and not isinstance(table.week_year, int):
        raise ConfigError(f"{fields.where}: iso_week tables need an integer week_year")
    return table


def _derived(fields: Fields) -> DerivedSpec:
    derived = DerivedSpec(
        name=fields.str("name"),
        op=fields.str("op", choices=("sum", "difference")),
        inputs=fields.strs("inputs"),
    )
    fields.done()
    if derived.op == "difference" and len(derived.inputs) != 2:
        raise ConfigError(f"{fields.where}: a difference takes exactly two inputs")
    return derived


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    raw = load_toml(path)
    fields = Fields(raw, Path(path).name)
    columns = tuple(_column(item) for item in fields.tables("columns"))
    tables = tuple(_table(item) for item in fields.tables("tables", []))
    derived = tuple(_derived(item) for item in fields.tables("derived", []))
    spec = DatasetSpec(
        id=fields.str("id"),
        title=fields.str("title"),
        source=fields.raw("source", ""),
        path=fields.str("path"),
        loader=fields.str("loader"),
        step=fields.str("step", choices=tuple(STEPS)),
        episodes=fields.str("episodes"),
        external_episodes=fields.strs("external_episodes", ()),
        columns=columns,
        tables=tables,
        derived=derived,
        origin=str(path),
        digest=digest(raw),
    )
    fields.done()
    names = [column.name for column in columns]
    if len(set(names)) != len(names):
        raise ConfigError(f"{spec.id}: duplicate column names")
    known = set(names)
    for table in tables:
        missing = sorted(set(table.columns) - known)
        if missing:
            raise ConfigError(f"{spec.id}: table {table.file} names unknown columns {missing}")
    for item in derived:
        if item.name not in known or not set(item.inputs) <= known:
            raise ConfigError(f"{spec.id}: derived column {item.name} uses undeclared columns")
    return spec


__all__ = ["STEPS", "ColumnSpec", "DatasetSpec", "DerivedSpec", "TableSpec", "load_dataset_spec"]
