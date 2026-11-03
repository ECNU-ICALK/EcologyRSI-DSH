"""Task suites: where harness versions are compared, and where the winner is verified.

A suite (TOML, shipped by a domain pack or given as a file) lists training
tasks, on which candidate harnesses are compared with the incumbent, and
holdout tasks, which only the one-time verification of an adopted version
may run; the evolving agents never see them. Entries are task references
(``<task id>`` or ``<task id>:<episode>``), so one card serves several sites
or teams. Every entry must be a single-episode task, and no (dataset,
episode) pair may appear twice or on both sides.

``gates`` decide when a candidate is credibly better across the training
tasks; see ``ecogym.evaluation.suite``. ``gates.test`` picks the interval
over the task gains: ``t`` (the default) or ``signed_rank``, which needs
enough training tasks for the exact test to reject at all (2^-N at most the
one-sided level). ``test = "t"`` stays out of the digest, so suites written
before the option keep theirs. ``repeats`` is the number of episodes
per version and training task a suite is designed for (a run that does not
set its own uses it), and ``recommended`` marks the suite a new run starts
from in the panel. Neither enters the suite's digest: they suggest how to run
it and do not change what a comparison on it means.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..kernel.canonical import digest
from ..kernel.config import ConfigError, Fields, load_toml
from .spec import TaskSpec, task_spec

GATE_TESTS = ("t", "signed_rank")


@dataclass(frozen=True, slots=True)
class SuiteGates:
    """Adoption gates for a candidate harness; see ``ecogym.evaluation.suite``."""

    min_delta: float = 0.005  # smallest mean skill gain over the training tasks worth adopting
    confidence: float = 0.95  # the interval of the mean task gain must exclude zero (see evaluation.suite)
    resamples: int = 2000
    task_cap: float = 0.03  # a task may not lose more than this with its own interval below zero
    test: str = "t"  # the interval over the task gains with three or more tasks: t | signed_rank


def gates_body(gates: SuiteGates) -> dict[str, float | int | str]:
    """The gates as they enter a suite's digest: without ``test`` when it is the default t interval."""

    body = asdict(gates)
    if body["test"] == "t":
        del body["test"]
    return body


def signed_rank_reachable(tasks: int, confidence: float, candidates: int = 1) -> bool:
    """Whether the exact signed-rank test over ``tasks`` gains can reject at the one-sided level of ``confidence``
    shared by ``candidates`` (Bonferroni): every gain positive has probability 2^-tasks under the null."""

    return tasks >= 3 and 0.5**tasks <= (1.0 - confidence) / (2.0 * max(1, candidates)) * (1.0 + 1e-9)


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    id: str
    title: str
    train: tuple[str, ...]
    holdout: tuple[str, ...]
    gates: SuiteGates
    notes: str = ""
    digest: str = ""
    repeats: int = 2  # episodes per version and training task per round, unless the run says otherwise
    recommended: bool = False

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
        test=table.str("test", default.test, choices=GATE_TESTS),
    )
    table.done()
    spec = SuiteSpec(
        id=fields.str("id"),
        title=fields.str("title"),
        train=fields.strs("train"),
        holdout=fields.strs("holdout", ()),
        gates=gates,
        notes=fields.raw("notes", ""),
        repeats=fields.int("repeats", 2, lo=1, hi=10),
        recommended=fields.bool("recommended", False),
    )
    fields.done()
    if gates.test == "signed_rank" and not signed_rank_reachable(len(spec.train), gates.confidence):
        raise ConfigError(f"suite {spec.id}: {len(spec.train)} training tasks are too few for the signed-rank test "
                          f"at confidence {gates.confidence} (it needs 0.5^N <= {(1 - gates.confidence) / 2:.4g})")
    body = {"schema": "ecogym.suite/1", "id": spec.id, "train": spec.train, "holdout": spec.holdout,
            "gates": gates_body(spec.gates)}
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


__all__ = ["GATE_TESTS", "SuiteGates", "SuiteSpec", "check_suite", "gates_body", "load_suite_spec", "register_suite",
           "signed_rank_reachable", "suite_spec", "suite_specs"]
