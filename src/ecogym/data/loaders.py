"""Loaders turn source files into a ``Frame``.

``csv_tables`` reads logged time series stored as one directory per episode,
each holding CSV tables at their own sampling interval (5-minute climate,
daily resources, weekly crop records), plus optional dataset-wide tables such
as a shared weather station. Every table is averaged onto the dataset step;
values are shifted by the table's availability delay so a daily summary is
only usable once the day is over. The first table's steps define an
episode's rows; values outside the declared valid range are dropped.

The arithmetic (``statistics.fmean`` at both averaging levels, the time
codecs, alias matching) reproduces the v0.10.33 greenhouse loader bit for bit.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from statistics import fmean

import numpy as np

from .frame import Frame
from .spec import DatasetSpec, TableSpec

_EPOCH = datetime(1899, 12, 30)
_DATE_PATTERNS = ("%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
_DATE_ONLY_PATTERNS = ("%m/%d/%Y", "%Y-%m-%d")
_NORMALIZE = re.compile(r"[^a-z0-9]+")
_MISSING_TOKENS = {"nan", "na", "none", "null"}

Loader = Callable[[DatasetSpec, Path], Frame]


def _normalize(name: str) -> str:
    return _NORMALIZE.sub("_", name.strip().lower().lstrip("%")).strip("_")


def _number(text: str | None) -> float | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped or stripped.casefold() in _MISSING_TOKENS:
        return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _date_only(text: str | None) -> bool:
    if text is None or _number(text) is not None or ":" in text:
        return False
    for pattern in _DATE_ONLY_PATTERNS:
        try:
            datetime.strptime(text.strip(), pattern)
            return True
        except ValueError:
            pass
    return False


def _step_of(text: str | None, steps_per_day: int) -> int | None:
    """Serial days (spreadsheet epoch 1899-12-30) or a date string -> step index."""

    value = _number(text)
    if value is not None:
        return math.floor(value * steps_per_day + 1e-7)
    if text is None:
        return None
    for pattern in _DATE_PATTERNS:
        try:
            moment = datetime.strptime(text.strip(), pattern)
        except ValueError:
            continue
        return math.floor((moment - _EPOCH).total_seconds() / (86400 / steps_per_day))
    return None


def _week_step(text: str | None, year: int, steps_per_day: int) -> int | None:
    week = _number(text)
    if week is None:
        return None
    try:
        monday = datetime.fromisocalendar(year, int(week), 1)
    except ValueError:
        return None
    return math.floor((monday - _EPOCH).total_seconds() / (86400 / steps_per_day))


def _read_table(path: Path, table: TableSpec, spec: DatasetSpec) -> dict[int, dict[str, float]]:
    """Average one CSV table onto the dataset step: ``{step: {column: value}}``."""

    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            return {}
        normalized = {_normalize(name): name for name in reader.fieldnames if name}
        aliases = [_normalize(alias) for alias in table.time_columns]
        time_columns = [normalized[alias] for alias in aliases if alias in normalized]
        if not time_columns:
            raise ValueError(f"no time column {table.time_columns} in {path}")
        sources: dict[str, tuple[str, ...]] = {}
        for name in table.columns:
            matches: set[str] = set()
            for alias in spec.column(name).aliases:
                key = _normalize(alias)
                for candidate, original in normalized.items():
                    if candidate == key or (table.allow_suffix and candidate.startswith(key + "_")):
                        matches.add(original)
            sources[name] = tuple(sorted(matches))
        rows = list(reader)

    def stamp(row: Mapping[str, str | None]) -> str | None:
        return next((row.get(name) for name in time_columns if row.get(name)), None)

    per_day = spec.steps_per_day
    sub_daily = table.interval_minutes < 1440
    rows_per_day = 1440 // table.interval_minutes
    step_minutes = 1440 // per_day
    day_offsets: dict[str, int] = {}
    if rows and sub_daily and _date_only(stamp(rows[0])):
        # Date-only stamps on sub-daily rows: spread the rows through the day,
        # aligning a partial first day to its end.
        counts = Counter(str(stamp(row) or "").strip() for row in rows)
        day_offsets = {day: max(0, rows_per_day - count) for day, count in counts.items() if day}
    occurrences: dict[str, int] = defaultdict(int)
    accumulator: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    columns = [(name, spec.column(name), sources[name]) for name in table.columns]
    for row in rows:
        text = stamp(row)
        if table.time_format == "iso_week":
            step = _week_step(text, int(table.week_year or 0), per_day)
        else:
            step = _step_of(text, per_day)
        if step is None:
            continue
        if sub_daily and _date_only(text):
            day = str(text).strip()
            step += (day_offsets.get(day, 0) + occurrences[day]) * table.interval_minutes // step_minutes
            occurrences[day] += 1
        step += table.delay_steps
        for name, column, matched in columns:
            values = [value for value in (_number(row.get(source)) for source in matched) if value is not None]
            if not values:
                continue
            value = fmean(values)
            if column.admits(value):
                accumulator[step][name].append(value)
    return {
        step: {name: fmean(values) for name, values in by_column.items() if values}
        for step, by_column in accumulator.items()
    }


def _derive(values: dict[str, list[float | None]], spec: DatasetSpec, length: int) -> None:
    for item in spec.derived:
        parts = [values.get(name) for name in item.inputs]
        if item.op == "sum":
            if all(part is None for part in parts):
                continue
            result: list[float | None] = []
            for index in range(length):
                terms = [part[index] if part is not None else None for part in parts]
                result.append(sum(terms) if all(term is not None for term in terms) else None)
        else:
            left, right = parts
            if left is None:
                continue
            result = [
                None if a is None or right is None or right[index] is None else a - right[index]
                for index, a in enumerate(left)
            ]
        values[item.name] = result


def load_csv_tables(spec: DatasetSpec, root: Path) -> Frame:
    if not spec.tables:
        raise ValueError(f"{spec.id} declares no tables")
    primary = spec.tables[0]
    if primary.scope != "episode":
        raise ValueError(f"{spec.id}: the first table must be per episode")
    directories = sorted(path.parent for path in root.glob(spec.episodes))
    if not directories:
        raise FileNotFoundError(f"no episodes match {spec.episodes} under {root}")
    shared: dict[Path, dict[int, dict[str, float]]] = {}
    episodes = []
    for directory in directories:
        tables = []
        for table in spec.tables:
            path = (root if table.scope == "dataset" else directory) / table.file
            if table.scope == "dataset":
                if path not in shared:
                    shared[path] = _read_table(path, table, spec)
                tables.append(shared[path])
            else:
                tables.append(_read_table(path, table, spec))
        if not tables[0]:
            raise ValueError(f"primary table {primary.file} is empty for {directory.name}")
        steps = sorted(tables[0])
        position = {step: index for index, step in enumerate(steps)}
        values: dict[str, list[float | None]] = {}
        for table_values in tables:
            for step, by_column in table_values.items():
                index = position.get(step)
                if index is None:
                    continue
                for name, value in by_column.items():
                    values.setdefault(name, [None] * len(steps))[index] = value
        _derive(values, spec, len(steps))
        kept = {name: column for name, column in values.items() if any(item is not None for item in column)}
        missing = [column.name for column in spec.columns if column.required and column.name not in kept]
        if missing:
            raise ValueError(f"episode {directory.name} lacks required columns: {', '.join(missing)}")
        arrays = {
            name: np.array([np.nan if item is None else item for item in column], dtype=np.float64)
            for name, column in kept.items()
        }
        episodes.append((directory.name, np.asarray(steps, dtype=np.int64), arrays))
    specs = {column.name: column for column in spec.columns}
    return Frame.build(spec.id, spec.step, episodes, specs)


LOADERS: dict[str, Loader] = {"csv_tables": load_csv_tables}


def source_files(spec: DatasetSpec, root: Path) -> list[Path]:
    """Every file a load would read, for cache keys."""

    files = []
    for directory in sorted(path.parent for path in root.glob(spec.episodes)):
        for table in spec.tables:
            path = (root if table.scope == "dataset" else directory) / table.file
            if path.is_file():
                files.append(path)
    return sorted(set(files))


__all__ = ["LOADERS", "Loader", "load_csv_tables", "source_files"]
