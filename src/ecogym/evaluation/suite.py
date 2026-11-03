"""Is one agent harness credibly better than another across a suite of tasks?

Each task contributes replicate evaluations of both versions (one per agent
episode) on the same scored origins. The task gain is the mean objective of
the candidate's replicates minus the incumbent's; its interval comes from a
day-block bootstrap of the stage with the replicates resampled too.

The suite gain is the mean of the task gains. With three or more tasks its
interval is a t interval over the task gains (``basis = "tasks_t"``): every
task gain already carries its episodes' run-to-run noise, and with a handful
of tasks a bootstrap over tasks is too narrow (its variance is (N − 1)/N of
the sample variance, and its quantiles are normal-like where t with N − 1
degrees of freedom applies). With one or two tasks the spread between tasks
cannot be estimated, so the interval resamples tasks, then replicates, then
days (``basis = "bootstrap"``), and tasks of one group (dataset) share their
day draws because they share weather and calendar. That interval covers
within-task noise only; a real test needs three or more tasks.

Gates with ``test = "signed_rank"`` replace the t interval with the interval
that belongs to the exact Wilcoxon signed-rank test over the task gains
(``basis = "tasks_signed_rank"``; Demšar 2006 recommends this test for two
methods compared over several data sets). The t interval grows with the
spread of the gains, so one task with an exceptionally large gain (a crash
fixed) can push a candidate that is better on every task below the gate; the
signed-rank test only uses the signs and the ranks of the gains' sizes. The
interval runs between order statistics of the Walsh averages (the means of
every pair of task gains, each gain paired with itself included): its lower
bound lies above zero exactly when the one-sided test at (1 − level)/2
rejects, a zero gain counting against the candidate. The exact test cannot
reject with too few tasks (all N gains positive has probability 2^-N under
the null); there the t interval is used instead, so the gate's suite loader
and the evolution loop refuse such a configuration.

A candidate is adopted only if all hold:

1. the mean gain over the tasks exceeds ``min_delta``;
2. the suite interval's lower bound lies above zero. The interval is
   two-sided at ``confidence``, so this is a one-sided test at
   (1 − confidence)/2: 2.5 % at the default 0.95;
3. no task regresses credibly: a task whose own interval lies below zero and
   whose loss exceeds ``task_cap`` blocks adoption. Task intervals always use
   the gates' own ``confidence``. The veto needs two replicates of each
   version on the task: with a single candidate episode the interval covers
   day-block noise only, and one unlucky run would read as a credible loss
   (such a task is reported as ``unconfirmed_regression``). Verification
   (``ecogym.evolution.verify``) has one result per version and task, so its
   verdicts never carry a per-task veto.

Several candidates compared with one incumbent in the same round should pass
a Bonferroni-adjusted ``confidence`` (see ``adjusted_confidence``); it widens
the suite interval only, so more candidates never loosen the per-task veto.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..kernel.canonical import digest
from ..task.suite import SuiteGates
from .bootstrap import iid_counts, percentile
from .harness import Evaluation
from .scoring import objective

MIN_TAIL_DRAWS = 10  # bootstrap draws beyond the quantile that bounds an interval
VETO_REPLICATES = 2  # replicates each version needs on a task before its regression can block adoption


@dataclass(frozen=True, slots=True)
class TaskPair:
    task: str
    candidate: tuple[Evaluation, ...]  # replicates
    incumbent: tuple[Evaluation, ...]
    group: str = ""  # tasks of one group (their dataset) share day draws; empty: the task alone


@dataclass(frozen=True, slots=True)
class TaskVerdict:
    task: str
    candidate: float
    incumbent: float
    delta: float
    interval: tuple[float, float]
    replicates: tuple[int, int]  # candidate, incumbent
    passed: bool
    reason: str  # better | within_uncertainty | credible_regression | unconfirmed_regression


@dataclass(frozen=True, slots=True)
class SuiteDecision:
    adopted: bool
    reason: str
    delta: float
    interval: tuple[float, float]
    confidence: float  # of the suite interval (Bonferroni-adjusted when given)
    tasks: tuple[TaskVerdict, ...]
    cells: dict[str, float]  # mean per-cell objective gain over the tasks that have the cell
    basis: str = "tasks_t"  # tasks_t | tasks_signed_rank | bootstrap
    task_confidence: float = 0.95  # of the per-task intervals

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def adjusted_confidence(confidence: float, comparisons: int) -> float:
    """Bonferroni: k candidates against one incumbent share the error rate."""

    return 1.0 - (1.0 - confidence) / max(1, comparisons)


def _t_central(theta: float, df: int) -> float:
    """P(|T| <= sqrt(df)·tan(theta)) for Student's t with integer ``df`` (Abramowitz & Stegun 26.7.3, 26.7.4)."""

    sine, cos2 = math.sin(theta), math.cos(theta) ** 2
    if df % 2 == 0:
        term = total = 1.0
        for k in range(2, df, 2):
            term *= cos2 * (k - 1) / k
            total += term
        return sine * total
    term = total = math.cos(theta) if df > 1 else 0.0
    for k in range(3, df, 2):
        term *= cos2 * (k - 1) / k
        total += term
    return 2.0 / math.pi * (theta + sine * total)


def t_quantile(probability: float, df: int) -> float:
    """The ``probability`` quantile (at least 0.5) of Student's t with integer ``df`` >= 1."""

    if df < 1 or not 0.5 <= probability < 1.0:
        raise ValueError("t_quantile needs df >= 1 and 0.5 <= probability < 1")
    target, low, high = 2.0 * probability - 1.0, 0.0, math.pi / 2
    for _ in range(80):  # bisection on the angle: exact to double precision, no bracket to guess
        middle = (low + high) / 2
        low, high = (middle, high) if _t_central(middle, df) < target else (low, middle)
    return math.sqrt(df) * math.tan((low + high) / 2)


def signed_rank_critical(n: int, tail: float) -> int:
    """The largest c with P(T ≤ c) ≤ ``tail`` for the Wilcoxon signed-rank statistic T of n untied, nonzero values
    under the null (all 2^n sign patterns equally likely); −1 when even P(T = 0) = 2^-n exceeds it, so that the
    one-sided test at ``tail`` cannot reject with n values."""

    if n < 1:
        return -1
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1.0
    for rank in range(1, n + 1):  # subsets of the ranks 1..n by their sum
        shifted = np.zeros_like(counts)
        shifted[rank:] = counts[:-rank]
        counts += shifted
    below = np.flatnonzero(np.cumsum(counts) / 2.0**n <= tail * (1.0 + 1e-9))
    return int(below[-1]) if below.size else -1


def signed_rank_interval(gains: np.ndarray, tail: float) -> tuple[float, float]:
    """Interval of the signed-rank test: from the (c + 1)-th smallest to the (c + 1)-th largest Walsh average,
    c = ``signed_rank_critical(n, tail)``. Its lower bound lies above zero exactly when at most c Walsh averages
    are ≤ 0, which for untied nonzero gains is the one-sided test rejecting at ``tail``. When the test cannot
    reject, the range of the Walsh averages widened to include zero."""

    upper = np.triu_indices(gains.size)
    walsh = np.sort((gains[:, None] + gains[None, :])[upper] / 2.0)
    c = signed_rank_critical(gains.size, tail)
    if c < 0:
        return min(float(walsh[0]), 0.0), max(float(walsh[-1]), 0.0)
    return float(walsh[c]), float(walsh[-1 - c])


def _check(pair: TaskPair) -> None:
    if not pair.candidate or not pair.incumbent:
        raise ValueError(f"{pair.task}: both versions need at least one evaluation")
    first = pair.candidate[0].stage
    for evaluation in (*pair.candidate, *pair.incumbent):
        stage = evaluation.stage
        if stage is not first and (stage.name != first.name or not np.array_equal(stage.time, first.time)
                                   or [cell.key for cell in stage.cells] != [cell.key for cell in first.cells]):
            raise ValueError(f"{pair.task}: evaluations are not on the same scored origins")


def _objectives(evaluations: Sequence[Evaluation], counts: np.ndarray) -> np.ndarray:
    """(replicates, resamples) objective of each replicate under each block resample."""

    return np.stack([objective(item.sums().total(counts), item.stage.weights) for item in evaluations])


def _cells(evaluations: Sequence[Evaluation]) -> dict[str, float]:
    rows = [{row["cell"]: row["objective"] for row in item.cells()} for item in evaluations]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def compare_suite(pairs: Sequence[TaskPair], gates: SuiteGates, seed: Any, *,
                  confidence: float | None = None) -> SuiteDecision:
    if not pairs:
        raise ValueError("a suite comparison needs at least one task")
    level = gates.confidence if confidence is None else confidence
    tail, task_tail = (1.0 - level) / 2.0, (1.0 - gates.confidence) / 2.0
    if len(pairs) < 3:
        basis = "bootstrap"
    elif gates.test == "signed_rank" and signed_rank_critical(len(pairs), tail) >= 0:
        basis = "tasks_signed_rank"
    else:  # the t interval, also where the signed-rank test cannot reject with this many tasks
        basis = "tasks_t"
    smallest = min(tail, task_tail) if basis == "bootstrap" else task_tail
    resamples = max(gates.resamples, math.ceil(MIN_TAIL_DRAWS / smallest) + 1)
    rng = np.random.default_rng(int(digest({"schema": "ecogym.suite-compare/2", "seed": seed})[:16], 16))
    columns = np.arange(resamples)[:, None]
    draws, verdicts = [], []
    cell_gains: dict[str, list[float]] = {}
    for pair in pairs:
        _check(pair)
        blocks = pair.candidate[0].sums().blocks
        counts = iid_counts(blocks.size, resamples, {"suite": seed, "group": pair.group or pair.task,
                                                     "blocks": blocks})
        candidate, incumbent = _objectives(pair.candidate, counts), _objectives(pair.incumbent, counts)
        picks_c = rng.integers(0, candidate.shape[0], size=(resamples, candidate.shape[0]))
        picks_i = rng.integers(0, incumbent.shape[0], size=(resamples, incumbent.shape[0]))
        gain = candidate[picks_c, columns].mean(axis=1) - incumbent[picks_i, columns].mean(axis=1)
        draws.append(gain)
        ordered = np.sort(gain)
        low, high = percentile(ordered, task_tail), percentile(ordered, 1.0 - task_tail)
        point_c = float(np.mean([item.score() for item in pair.candidate]))
        point_i = float(np.mean([item.score() for item in pair.incumbent]))
        delta = point_c - point_i
        if high < 0.0 and -delta > gates.task_cap:
            # one run per side: the interval holds day-block noise only, not the agent's run-to-run spread
            confirmed = min(len(pair.candidate), len(pair.incumbent)) >= VETO_REPLICATES
            passed, reason = (False, "credible_regression") if confirmed else (True, "unconfirmed_regression")
        else:
            passed, reason = True, "better" if delta > 0 else "within_uncertainty"
        verdicts.append(TaskVerdict(pair.task, point_c, point_i, delta, (low, high),
                                    (len(pair.candidate), len(pair.incumbent)), passed, reason))
        before, after = _cells(pair.incumbent), _cells(pair.candidate)
        for key in after.keys() & before.keys():
            cell_gains.setdefault(key, []).append(after[key] - before[key])
    gains = np.array([verdict.delta for verdict in verdicts])
    delta = float(gains.mean())
    if basis == "tasks_t":
        half = t_quantile(1.0 - tail, gains.size - 1) * float(gains.std(ddof=1)) / math.sqrt(gains.size)
        interval = (delta - half, delta + half)
    elif basis == "tasks_signed_rank":
        interval = signed_rank_interval(gains, tail)
    else:  # one column per resample for every drawn task: tasks of one group keep their shared day draw
        picks = rng.integers(0, len(pairs), size=(resamples, len(pairs)))
        overall = np.sort(np.stack(draws)[picks, columns].mean(axis=1))
        interval = (percentile(overall, tail), percentile(overall, 1.0 - tail))
    failed = [verdict.task for verdict in verdicts if not verdict.passed]
    if delta <= gates.min_delta:
        adopted, reason = False, "below_practical_delta"
    elif interval[0] <= 0.0:
        adopted, reason = False, "interval_crosses_zero"
    elif failed:
        adopted, reason = False, "task_regression:" + ",".join(failed)
    else:
        adopted, reason = True, "improved"
    cells = {key: float(np.mean(values)) for key, values in sorted(cell_gains.items())}
    return SuiteDecision(adopted, reason, delta, interval, level, tuple(verdicts), cells, basis, gates.confidence)


__all__ = ["SuiteDecision", "TaskPair", "TaskVerdict", "adjusted_confidence", "compare_suite", "signed_rank_critical",
           "signed_rank_interval", "t_quantile"]
