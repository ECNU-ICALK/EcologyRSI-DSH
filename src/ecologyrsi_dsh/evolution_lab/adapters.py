"""Adapters that keep the external lab independent from DSH internals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .evaluator import EvaluationReport
from .genome import PluginGenome


class ExperimentBackend(Protocol):
    def evaluate(self, genome: PluginGenome, cohort_id: str) -> EvaluationReport: ...


class CallableBackend:
    """Deterministic adapter used by tests and local experiments."""

    def __init__(self, function: Callable[[PluginGenome, str], EvaluationReport]) -> None:
        self._function = function

    def evaluate(self, genome: PluginGenome, cohort_id: str) -> EvaluationReport:
        report = self._function(genome, cohort_id)
        if report.cohort_id != cohort_id:
            raise ValueError("backend returned a report for the wrong cohort")
        return report


class DshExperimentAdapter:
    """Thin production seam: the caller supplies an existing DSH API runner.

    The adapter intentionally knows no DSH implementation details.  A host
    integration maps ``run_genome`` to the existing create/status/projection
    endpoints and returns the public metrics as ``EvaluationReport``.
    """

    def __init__(self, run_genome: Callable[[PluginGenome, str], EvaluationReport]) -> None:
        self._runner = CallableBackend(run_genome)

    def evaluate(self, genome: PluginGenome, cohort_id: str) -> EvaluationReport:
        return self._runner.evaluate(genome, cohort_id)
