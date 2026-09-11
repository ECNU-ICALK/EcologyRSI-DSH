"""Bounded sibling-candidate evaluation with drain-before-return semantics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

from ..core.errors import preferred_execution_failure


@dataclass(frozen=True, slots=True, order=True)
class CandidateEvaluationTask:
    """One generation-local candidate ordered by its frozen sibling slot."""

    slot_index: int
    candidate_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.slot_index, bool)
            or not isinstance(self.slot_index, int)
            or self.slot_index < 0
        ):
            raise ValueError("candidate evaluation slot_index must be non-negative")
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate evaluation candidate_id must be non-empty")


def run_candidate_evaluations(
    tasks: Sequence[CandidateEvaluationTask],
    *,
    max_concurrency: int,
    evaluate: Callable[[str], None],
    admission_open: Callable[[], bool],
) -> None:
    """Run admitted tasks concurrently and drain every in-flight worker.

    Work is submitted only up to ``max_concurrency``. The first worker error
    or a closed run admission stops queue refill, but work already admitted is
    allowed to settle before the fatal or lowest-slot error is re-raised. A concurrency
    value of one deliberately stays on the caller thread so legacy manifests
    preserve their historical execution boundary.
    """

    if (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or max_concurrency < 1
    ):
        raise ValueError("candidate evaluation concurrency must be positive")
    ordered = tuple(sorted(tasks))
    if len({task.candidate_id for task in ordered}) != len(ordered):
        raise ValueError("candidate evaluation tasks must have unique candidate ids")
    if len({task.slot_index for task in ordered}) != len(ordered):
        raise ValueError("candidate evaluation tasks must have unique slot indices")

    if max_concurrency == 1 or len(ordered) <= 1:
        for task in ordered:
            if not admission_open():
                break
            evaluate(task.candidate_id)
        return

    task_iter = iter(ordered)
    active: dict[Future[None], CandidateEvaluationTask] = {}
    failures: list[tuple[int, Exception]] = []
    worker_count = min(max_concurrency, len(ordered))

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="ecology-candidate",
    ) as executor:
        exhausted = False

        def fill() -> None:
            nonlocal exhausted
            while (
                not exhausted
                and not failures
                and len(active) < worker_count
                and admission_open()
            ):
                try:
                    task = next(task_iter)
                except StopIteration:
                    exhausted = True
                    return
                active[executor.submit(evaluate, task.candidate_id)] = task

        fill()
        while active:
            completed, _pending = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda item: active[item].slot_index):
                task = active.pop(future)
                try:
                    future.result()
                except Exception as exc:  # drain siblings before propagation
                    failures.append((task.slot_index, exc))
            fill()

    if failures:
        failures.sort(key=lambda item: item[0])
        failure = failures[0][1]
        for _slot, error in failures[1:]:
            failure = preferred_execution_failure(failure, error)
        raise failure


__all__ = ["CandidateEvaluationTask", "run_candidate_evaluations"]
