"""Solver contract, parameter schemas, genomes and the solver registry.

A solver is registered once, as a ``SolverSpec``: an id and version, a
parameter schema, and two functions::

    fit(data: Origins, ctx: SolverContext, params) -> fitted state
    predict(fitted, origins: Origins, ctx: SolverContext) -> {cell key: forecasts}

``fit`` sees calibration_fit origins with labels; ``predict`` sees forecast
origins without labels and may only read the past. A forecast of NaN counts
as missing. Everything else (catalog entries, defaults, bounds for search,
validation of proposed edits) is derived from the schema.

A ``Genome`` is a solver reference plus a complete, normalized parameter set;
its digest identifies a method.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..kernel.canonical import digest, jsonable
from ..protocol.baselines import BaselineProfile, baseline_values
from ..protocol.views import Origins
from ..task.spec import TaskSpec

PARAM_KINDS = ("float", "int", "choice", "cells", "json")


class SolverError(ValueError):
    """A genome is invalid or its solver cannot fit."""


@dataclass(frozen=True, slots=True)
class Param:
    """One tunable. ``cells`` is a float per task cell (edit path ``name.<cell>``)."""

    name: str
    kind: str
    default: Any
    lo: float | None = None
    hi: float | None = None
    choices: tuple[Any, ...] = ()
    doc: str = ""
    validate: Callable[[Any, TaskSpec], Any] | None = None  # json params: normalize or raise

    @property
    def log_scale(self) -> bool:
        """Search coordinates are logarithmic for positive ranges spanning >= 100x."""

        return self.lo is not None and self.hi is not None and self.lo > 0 and self.hi / self.lo >= 100

    def describe(self) -> dict[str, Any]:
        body: dict[str, Any] = {"kind": self.kind, "default": jsonable(self.default), "doc": self.doc}
        if self.lo is not None:
            body["lo"] = self.lo
        if self.hi is not None:
            body["hi"] = self.hi
        if self.choices:
            body["choices"] = list(self.choices)
        if self.log_scale:
            body["scale"] = "log"
        return body

    def _number(self, value: Any, where: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SolverError(f"{where} must be a finite number")
        if (self.lo is not None and value < self.lo) or (self.hi is not None and value > self.hi):
            raise SolverError(f"{where} must lie in [{self.lo}, {self.hi}]")
        return float(value)

    def normalize(self, value: Any, task: TaskSpec) -> Any:
        where = f"parameter {self.name}"
        if self.kind == "float":
            return self._number(value, where)
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise SolverError(f"{where} must be an integer")
            self._number(value, where)
            return int(value)
        if self.kind == "choice":
            if value not in self.choices:
                raise SolverError(f"{where} must be one of {list(self.choices)}")
            return value
        if self.kind == "cells":
            keys = [cell.key for cell in task.cells]
            if isinstance(value, Mapping):
                unknown = sorted(set(value) - set(keys))
                if unknown:
                    raise SolverError(f"{where} names unknown cells {unknown}")
                default = self.default if not isinstance(self.default, Mapping) else None
                result = {}
                for key in keys:
                    item = value.get(key, default)
                    if item is None:
                        raise SolverError(f"{where} lacks cell {key}")
                    result[key] = self._number(item, f"{where}.{key}")
                return result
            number = self._number(value, where)
            return {key: number for key in keys}
        if self.kind == "json":
            return jsonable(self.validate(value, task) if self.validate else value)
        raise SolverError(f"{where} has unknown kind {self.kind}")


@dataclass(frozen=True, slots=True)
class SolverContext:
    """What a solver may know besides the data: the task and its comparators."""

    task: TaskSpec
    baselines: BaselineProfile

    def baseline(self, origins: Origins, cell_key: str) -> np.ndarray:
        """The fit-selected naive forecast for one cell (seasonal falls back to persistence)."""

        return _single_baseline(origins, self.task, self.baselines, cell_key)

    def fill_age(self, origins: Origins, column: str) -> int:
        role = origins.spec(column).role
        if role not in self.task.exogenous_roles:
            raise SolverError(f"column {column} has role {role}, which this task does not admit as an input")
        return self.task.fill_age[role]

    def exogenous(self, origins: Origins, column: str, k: int) -> np.ndarray:
        """Forward-filled value of an admissible input column ``k`` steps back."""

        return origins.past_filled(column, k, self.fill_age(origins, column))


def _single_baseline(origins: Origins, task: TaskSpec, profile: BaselineProfile, cell_key: str) -> np.ndarray:
    cell = next(cell for cell in task.cells if cell.key == cell_key)
    persistence = origins.past(cell.target, 0)
    if profile.kind(cell_key) == "seasonal":
        seasonal = origins.past(cell.target, task.season_steps - cell.horizon)
        return np.where(np.isfinite(seasonal), seasonal, persistence)
    return persistence


FitFn = Callable[[Origins, SolverContext, Mapping[str, Any]], Any]
PredictFn = Callable[[Any, Origins, SolverContext], dict[str, np.ndarray]]


@dataclass(frozen=True, slots=True)
class SolverSpec:
    id: str
    version: int
    title: str
    doc: str
    params: tuple[Param, ...]
    fit: FitFn
    predict: PredictFn
    describe_fit: Callable[[Any], dict[str, Any]] | None = None

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    def param(self, name: str) -> Param:
        for param in self.params:
            if param.name == name:
                return param
        raise SolverError(f"{self.ref} has no parameter {name}")

    def normalize(self, params: Mapping[str, Any], task: TaskSpec) -> dict[str, Any]:
        unknown = sorted(set(params) - {param.name for param in self.params})
        if unknown:
            raise SolverError(f"{self.ref} has no parameters {unknown}")
        return {
            param.name: param.normalize(params.get(param.name, param.default), task) for param in self.params
        }

    def describe(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "title": self.title,
            "doc": self.doc,
            "params": {param.name: param.describe() for param in self.params},
        }


@dataclass(frozen=True, slots=True)
class Genome:
    """A method: one solver and a complete, normalized parameter set."""

    solver: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return digest({"schema": "ecogym.genome/1", "solver": self.solver, "params": self.params})

    def to_dict(self) -> dict[str, Any]:
        return {"solver": self.solver, "params": jsonable(self.params)}

    @staticmethod
    def build(solver: str, params: Mapping[str, Any], task: TaskSpec) -> Genome:
        spec = get_solver(solver)
        return Genome(spec.ref, spec.normalize(params, task))

    @staticmethod
    def from_dict(body: Mapping[str, Any], task: TaskSpec) -> Genome:
        if set(body) - {"solver", "params"} or "solver" not in body:
            raise SolverError("a genome is {solver, params}")
        return Genome.build(str(body["solver"]), dict(body.get("params") or {}), task)


_SOLVERS: dict[str, SolverSpec] = {}


def register_solver(spec: SolverSpec) -> SolverSpec:
    existing = _SOLVERS.get(spec.ref)
    if existing is not None and existing is not spec:
        raise ValueError(f"solver {spec.ref} is already registered")
    _SOLVERS[spec.ref] = spec
    return spec


def get_solver(ref: str) -> SolverSpec:
    try:
        return _SOLVERS[ref]
    except KeyError:
        raise SolverError(f"unknown solver {ref}; known: {', '.join(sorted(_SOLVERS)) or 'none'}") from None


def solver_specs() -> list[SolverSpec]:
    return [_SOLVERS[key] for key in sorted(_SOLVERS)]


__all__ = [
    "PARAM_KINDS",
    "Genome",
    "Param",
    "SolverContext",
    "SolverError",
    "SolverSpec",
    "baseline_values",
    "get_solver",
    "register_solver",
    "solver_specs",
]
