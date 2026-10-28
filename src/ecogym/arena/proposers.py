"""Proposers: where candidate edits come from.

Every source of method changes (a parameter sweep, random search, an LLM
research agent, a person) implements one interface::

    propose(context, k) -> list[Edit]     # up to k edits of context.incumbent
    observe(report)                        # what happened to them

The arena validates each edit against the solver schema and the trust
region; a proposer cannot change how candidates are scored.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..kernel.canonical import digest
from ..solvers.base import Genome
from ..task.spec import TaskSpec
from .genome import Edit, axes


@dataclass(frozen=True, slots=True)
class ProposalContext:
    task: TaskSpec
    incumbent: Genome
    generation: int
    seed: int
    max_step: float
    max_changes: int
    history: tuple[dict[str, Any], ...] = ()  # previous generation reports, oldest first
    bench: Any = None  # the run's Bench, for proposers that study development data
    screening: Any = None  # this generation's screening positions in the selection stage
    log: Callable[[dict[str, Any]], None] | None = None  # records an agent exchange in the ledger

    def axes(self) -> list[dict[str, Any]]:
        return [axis.describe(self.max_step) for axis in axes(self.incumbent)]


class Proposer(Protocol):
    name: str

    def options(self) -> dict[str, Any]: ...

    def propose(self, context: ProposalContext, k: int) -> list[Edit]: ...

    def observe(self, report: Mapping[str, Any]) -> None: ...


@dataclass
class SweepProposer:
    """Coordinate sweep: one axis per edit, +step then -step, cycling through axes.

    After a promotion the sweep restarts around the new incumbent.
    """

    step: float | None = None  # fraction of an axis range; defaults to the trust region
    name: str = "sweep"
    _cursor: int = field(default=0, init=False)

    def options(self) -> dict[str, Any]:
        return {"step": self.step}

    def propose(self, context: ProposalContext, k: int) -> list[Edit]:
        step = min(self.step or context.max_step, context.max_step)
        moves = [(axis, sign) for axis in axes(context.incumbent) for sign in (1.0, -1.0)]
        edits: list[Edit] = []
        tried = 0
        while len(edits) < k and tried < len(moves):
            axis, sign = moves[self._cursor % len(moves)]
            self._cursor += 1
            tried += 1
            value = axis.from_unit(axis.to_unit(axis.value) + sign * step)
            if value != axis.value:
                direction = "up" if sign > 0 else "down"
                edits.append(Edit({axis.path: value}, f"sweep {axis.path} {direction} by {step:.2f} of its range"))
        return edits

    def observe(self, report: Mapping[str, Any]) -> None:
        if report.get("promoted"):
            self._cursor = 0


@dataclass
class RandomProposer:
    """Random search inside the trust region: 1..max_changes axes per edit."""

    name: str = "random"

    def options(self) -> dict[str, Any]:
        return {}

    def propose(self, context: ProposalContext, k: int) -> list[Edit]:
        material = {"proposer": self.name, "seed": context.seed, "generation": context.generation,
                    "incumbent": context.incumbent.digest}
        rng = random.Random(int(digest(material)[:16], 16))
        coordinates = axes(context.incumbent)
        edits = []
        for _ in range(k):
            count = rng.randint(1, min(context.max_changes, len(coordinates)))
            changes = {}
            for axis in rng.sample(coordinates, count):
                unit = axis.to_unit(axis.value) + rng.uniform(-context.max_step, context.max_step)
                value = axis.from_unit(unit)
                if value != axis.value:
                    changes[axis.path] = value
            if changes:
                edits.append(Edit(changes, f"random move of {', '.join(sorted(changes))}"))
        return edits

    def observe(self, report: Mapping[str, Any]) -> None:
        return None


@dataclass
class RecordedProposer:
    """Plays back the edits a run logged, generation by generation (used by ``replay``)."""

    edits: dict[int, list[Edit]]
    name: str = "recorded"

    def options(self) -> dict[str, Any]:
        return {}

    def propose(self, context: ProposalContext, k: int) -> list[Edit]:
        return list(self.edits.get(context.generation, []))

    def observe(self, report: Mapping[str, Any]) -> None:
        return None


PROPOSERS: dict[str, Callable[..., Proposer]] = {"sweep": SweepProposer, "random": RandomProposer}


def register_proposer(name: str, factory: Callable[..., Proposer]) -> None:
    PROPOSERS[name] = factory


def make_proposer(name: str, options: Mapping[str, Any] | None = None) -> Proposer:
    try:
        factory = PROPOSERS[name]
    except KeyError:
        raise ValueError(f"unknown proposer {name}; known: {', '.join(sorted(PROPOSERS))}") from None
    return factory(**dict(options or {}))


__all__ = [
    "PROPOSERS",
    "ProposalContext",
    "Proposer",
    "RandomProposer",
    "RecordedProposer",
    "SweepProposer",
    "make_proposer",
    "register_proposer",
]
