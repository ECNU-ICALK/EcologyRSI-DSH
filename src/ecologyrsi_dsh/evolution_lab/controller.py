"""Orchestrate external plugin experiments without changing DSH."""

from __future__ import annotations

from collections.abc import Callable

from .capabilities import CapabilityRegistry
from .evaluator import EvaluationReport, PromotionDecision, evaluate_candidate
from .genome import Mutation, MutationPlanner, PluginGenome
from .store import EvolutionStore


Backend = Callable[[PluginGenome, str], EvaluationReport]


class EvolutionController:
    def __init__(self, store: EvolutionStore, registry: CapabilityRegistry) -> None:
        self.store = store
        self.planner = MutationPlanner(registry)

    def seed(self, genome: PluginGenome) -> None:
        self.store.put_genome(genome, "incumbent")

    def evaluate_and_record(
        self,
        parent_digest: str,
        mutation: Mutation,
        cohort_id: str,
        backend: Backend,
        *,
        practical_delta: float = 0.005,
    ) -> PromotionDecision:
        parent = self.store.get_genome(parent_digest)
        child = self.planner.apply(parent, mutation)
        self.store.put_genome(child)
        incumbent_digest = self.store.incumbent_digest() or parent_digest
        incumbent = self.store.get_genome(incumbent_digest)
        candidate_report = backend(child, cohort_id)
        incumbent_report = backend(incumbent, cohort_id)
        decision = evaluate_candidate(candidate_report, incumbent_report, practical_delta=practical_delta)
        self.store.put_evaluation(child, cohort_id, candidate_report, decision)
        return decision

    def latest_candidate_digest(self) -> str:
        return self.store.latest_digest()

    def promote(self, digest: str, *, idempotency_key: str | None = None) -> None:
        if self.store.get_status(digest) != "eligible":
            raise ValueError("only certification-eligible candidates can become incumbent")
        current = self.store.incumbent_digest()
        key = idempotency_key or f"promote:{digest}"
        if not self.store.record_transition("promote", current, digest, key):
            return
        if current and current != digest:
            self.store.set_status(current, "superseded")
        self.store.set_status(digest, "incumbent")

    def rollback(self, digest: str, *, idempotency_key: str) -> None:
        self.store.get_genome(digest)
        current = self.store.incumbent_digest()
        if not self.store.record_transition("rollback", current, digest, idempotency_key):
            return
        if current and current != digest:
            self.store.set_status(current, "rolled_back")
        self.store.set_status(digest, "incumbent")

    def current_incumbent(self) -> PluginGenome:
        digest = self.store.incumbent_digest()
        if digest is None:
            raise KeyError("no incumbent Genome")
        return self.store.get_genome(digest)
