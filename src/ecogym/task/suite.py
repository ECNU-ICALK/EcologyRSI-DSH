"""Task suites: where harness versions are compared, and where the winner is verified.

A suite (TOML, shipped by a domain pack or given as a file) lists training
tasks, on which candidate harnesses are compared with the incumbent, and
holdout tasks, which only the one-time verification of an adopted version
may run; the evolving agents never see them. Entries are task references
(``<task id>`` or ``<task id>:<episode>``), so one card serves several sites
or teams. Every entry must be a single-episode task, and no (dataset,
episode) pair may appear twice or on both sides.

``gates`` decide when a candidate is credibly better across the training
tasks; see ``ecogym.evaluation.suite``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ..kernel.canonical import digest
from ..kernel.config import ConfigError, Fields, load_toml
from .spec import TaskSpec, task_spec


@dataclass(frozen=True, slots=True)
class SuiteGates:
    """Adoption gates for a candidate harness; see ``ecogym.evaluation.suite``."""

    min_delta: float = 0.005  # smallest mean skill gain over the training tasks worth adopting
    confidence: float = 0.95  # the interval of the mean task gain must exclude zero (see evaluation.suite)
    resamples: int = 2000
    task_cap: float = 0.03  # a task may not lose more than this with its own interval below zero


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    id: str
    title: str
    train: tuple[str, ...]
    holdout: tuple[str, ...]
    gates: SuiteGates
    notes: str = ""
    digest: str = ""

    def tasks(self, side: str = "train") -> list[TaskSpec]:
        refs = self.train if side == "train" else self.holdout
        return [task_spec(ref) for ref in refs]


def load_suite_spec(path: str | Path) -> SuiteSpec:
    fields = Fields(load_toml(path), Path(path).name)
    table = fields.table("gates")
    default = SuiteGates()  # slots dataclass: class attributes are descriptors, not defaults
    gates = SuiteGates(
        min_delta=table.float("min_delta", default.min_delta, lo=0.0),
        confidence=table.float("confidence", default.confidence, lo=0.5, hi=0.999),
        resamples=table.int("resamples", default.resamples, lo=200),
        task_cap=table.float("task_cap", default.task_cap, lo=0.0),
    )
    table.done()
    spec = SuiteSpec(
        id=fields.str("id"),
        title=fields.str("title"),
        train=fields.strs("train"),
        holdout=fields.strs("holdout", ()),
        gates=gates,
        notes=fields.raw("notes", ""),
    )
    fields.done()
    body = {"schema": "ecogym.suite/1", "id": spec.id, "train": spec.train, "holdout": spec.holdout,
            "gates": spec.gates}
    return replace(spec, digest=digest(body))


def check_suite(spec: SuiteSpec) -> None:
    """Fail unless every entry resolves to a distinct single-episode task."""

    seen: dict[tuple[str, str], str] = {}
    for side, refs in (("train", spec.train), ("holdout", spec.holdout)):
        for ref in refs:
            try:
                task = task_spec(ref)
            except KeyError as error:
                raise ConfigError(f"suite {spec.id}: {error.args[0]}") from None
            if len(task.episodes) != 1:
                raise ConfigError(f"suite {spec.id}: {ref} has {len(task.episodes)} episodes; use <task>:<episode>")
            key = (task.dataset, task.episodes[0])
            if key in seen:
                raise ConfigError(f"suite {spec.id}: {ref} ({side}) repeats {seen[key]}")
            seen[key] = f"{ref} ({side})"
    if not spec.train:
        raise ConfigError(f"suite {spec.id} has no training tasks")


_SUITES: dict[str, SuiteSpec] = {}


def register_suite(path: str | Path) -> SuiteSpec:
    spec = load_suite_spec(path)
    existing = _SUITES.get(spec.id)
    if existing is not None and existing.digest != spec.digest:
        raise ValueError(f"suite {spec.id} is registered twice with different contents")
    _SUITES[spec.id] = spec
    return spec


def suite_spec(ref: str) -> SuiteSpec:
    """A registered suite, or a suite TOML file (checked on load)."""

    if ref in _SUITES:
        return _SUITES[ref]
    path = Path(ref).expanduser()
    if path.suffix == ".toml" and path.is_file():
        spec = load_suite_spec(path)
        check_suite(spec)
        return spec
    raise KeyError(f"unknown suite {ref}; known: {', '.join(sorted(_SUITES)) or 'none'} (or give a .toml file)")


def suite_specs() -> list[SuiteSpec]:
    return [_SUITES[key] for key in sorted(_SUITES)]


__all__ = ["SuiteGates", "SuiteSpec", "check_suite", "load_suite_spec", "register_suite", "suite_spec",
           "suite_specs"]
