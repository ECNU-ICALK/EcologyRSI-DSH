"""Carry sealed training constraints into selection without relabeling evidence."""
from collections.abc import Mapping

from .trajectory import HoldoutArm
from ..evolution.effect_contracts import hard_effect_failure


def finalist_training_effect_evidence(holdouts, *, screening_events=(), batch_evaluations=()):
    """Read only the finalist's exact revision in this generation.

    A parent failure cannot taint a repaired child. Conversely, returning to a
    parent returns to that parent's own constraints. Receipts stay references
    to their training scopes; they never become holdout observations.
    """
    result = {}
    for item in holdouts:
        scope = item.scope
        if scope.holdout_arm is HoldoutArm.INCUMBENT:
            continue
        identity = (scope.generation, scope.candidate_id, scope.candidate_revision_id)
        sources = []

        def add(kind, source_id, source_digest, metrics):
            sample = metrics.get("sample_execution", {})
            receipt = sample.get("mutation_effect_receipt") if isinstance(sample, Mapping) else None
            if not isinstance(receipt, Mapping):
                return
            failure = hard_effect_failure(metrics)  # Validate the original receipt.
            sources.append({"source_kind": kind, "source_id": source_id,
                "source_evidence_digest": source_digest,
                "source_scope_digest": receipt.get("scope_digest"),
                "receipt_digest": receipt["receipt_digest"], "failure": failure})

        for event in screening_events:
            row = event.payload
            if event.run_id == scope.run_id and (row.get("generation"), row.get("candidate_id"), row.get("candidate_revision_id")) == identity:
                add("screening", event.event_id, row.get("record_digest"), row.get("metrics", {}))
        for evaluation in batch_evaluations:
            original = evaluation.scope
            if original.run_id != scope.run_id or (original.generation, original.candidate_id, original.candidate_revision_id) != identity:
                continue
            # Warmup already has its authoritative screening receipt above.
            if evaluation.metrics.get("reused_screening_evidence") is True:
                continue
            add("formal_batch", evaluation.evaluation_id, evaluation.evaluation_digest, evaluation.metrics)
        if sources:
            result[scope.holdout_arm.value] = {
                "generation": scope.generation, "candidate_id": scope.candidate_id,
                "candidate_revision_id": scope.candidate_revision_id,
                "qualification": "sealed_training_effect_constraints_not_holdout_observations",
                "failure": next((row["failure"] for row in sources if row["failure"]), None),
                "sources": sorted(sources, key=lambda row: (row["source_kind"], row["source_id"])),
            }
    return result or None
