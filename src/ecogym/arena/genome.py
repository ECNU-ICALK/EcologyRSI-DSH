"""Bounded edits of a genome.

An ``Edit`` sets parameters by path: a parameter name (``alpha``,
``terms``) or, for per-cell parameters, ``name.<cell>`` (``scale.<target>@<horizon>``).
Every edited genome is re-normalized by its solver's schema, and numeric
moves must stay inside a trust region: each changed coordinate may move at
most ``max_step`` of its range, measured on a log scale for positive ranges
spanning 100x or more (v0.10.33's ``bounded-trust-region-mutation@2``).
Structured parameters (recipes, mechanism terms) are replaced whole.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..solvers.base import Genome, Param, SolverError, get_solver
from ..task.spec import TaskSpec


@dataclass(frozen=True, slots=True)
class Edit:
    changes: dict[str, Any]
    rationale: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"changes": self.changes, "rationale": self.rationale, "meta": self.meta}


@dataclass(frozen=True, slots=True)
class Axis:
    """One numeric coordinate of a genome."""

    path: str
    param: Param
    value: float

    @property
    def lo(self) -> float:
        return float(self.param.lo)

    @property
    def hi(self) -> float:
        return float(self.param.hi)

    def to_unit(self, value: float) -> float:
        if self.param.log_scale:
            return (math.log(value) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (value - self.lo) / (self.hi - self.lo)

    def from_unit(self, unit: float) -> float:
        unit = min(1.0, max(0.0, unit))
        if self.param.log_scale:
            return math.exp(math.log(self.lo) + unit * (math.log(self.hi) - math.log(self.lo)))
        value = self.lo + unit * (self.hi - self.lo)
        return float(round(value)) if self.param.kind == "int" else value

    def describe(self, max_step: float) -> dict[str, Any]:
        unit = self.to_unit(self.value)
        return {
            "path": self.path,
            "value": self.value,
            "range": [self.lo, self.hi],
            "scale": "log" if self.param.log_scale else "linear",
            "reachable": [self.from_unit(unit - max_step), self.from_unit(unit + max_step)],
        }


def axes(genome: Genome) -> list[Axis]:
    """Every bounded numeric coordinate, in schema order."""

    spec = get_solver(genome.solver)
    result = []
    for param in spec.params:
        if param.lo is None or param.hi is None or param.hi <= param.lo:
            continue
        if param.kind in ("float", "int"):
            result.append(Axis(param.name, param, float(genome.params[param.name])))
        elif param.kind == "cells":
            for key, value in genome.params[param.name].items():
                result.append(Axis(f"{param.name}.{key}", param, float(value)))
    return result


def _set(params: dict[str, Any], path: str, value: Any, genome: Genome) -> None:
    """Set ``name`` or ``name.key`` (a cell of a per-cell parameter, or a key of a mapping parameter)."""

    spec = get_solver(genome.solver)
    name, _, key = path.partition(".")
    param = spec.param(name)
    if not key:
        params[name] = value
    elif param.kind == "cells":
        if key not in params[name]:
            raise SolverError(f"{path}: unknown cell {key}")
        params[name] = {**params[name], key: value}
    elif param.kind == "json" and isinstance(params[name], Mapping):
        params[name] = {**params[name], key: value}
    else:
        raise SolverError(f"{path}: {name} does not take a key suffix")


def apply_edit(genome: Genome, edit: Edit, task: TaskSpec) -> Genome:
    params = dict(genome.params)
    for path, value in edit.changes.items():
        _set(params, path, value, genome)
    return Genome.build(genome.solver, params, task)


def check_edit(parent: Genome, child: Genome, *, max_step: float, max_changes: int) -> list[str]:
    """Trust-region violations of ``child`` relative to ``parent`` (empty if allowed)."""

    problems = []
    parent_axes = {axis.path: axis for axis in axes(parent)}
    changed = 0
    for axis in axes(child):
        before = parent_axes[axis.path]
        if axis.value == before.value:
            continue
        changed += 1
        step = abs(axis.to_unit(axis.value) - before.to_unit(before.value))
        if step > max_step + 1e-12:
            problems.append(f"{axis.path} moves {step:.3f} of its range (limit {max_step})")
    spec = get_solver(parent.solver)
    structured = [param.name for param in spec.params if param.kind in ("json", "choice")]
    changed += sum(1 for name in structured if parent.params[name] != child.params[name])
    if changed == 0:
        problems.append("the edit changes nothing")
    if changed > max_changes:
        problems.append(f"the edit changes {changed} coordinates (limit {max_changes})")
    return problems


def edit_from_dict(body: Mapping[str, Any]) -> Edit:
    if not isinstance(body, Mapping) or not isinstance(body.get("changes"), Mapping):
        raise SolverError("an edit is {changes: {path: value}, rationale}")
    return Edit(dict(body["changes"]), str(body.get("rationale", "")), dict(body.get("meta", {})))


__all__ = ["Axis", "Edit", "apply_edit", "axes", "check_edit", "edit_from_dict"]
