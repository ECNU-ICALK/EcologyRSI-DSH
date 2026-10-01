"""Local edit invariants used both before append and during ledger replay."""
from __future__ import annotations

from .trajectory import LocalEditProposalDecision, TrajectoryStatus
from ..evolution.schedule import PAIRED_LOCAL_EVALUATION_MODE


def local_edit_key(payload):
    fields = {"proposal_id", "candidate_id", "batch_index", "evidence_scope_digest", "proposal"}
    if set(payload) not in (fields, fields | {"safety_reason"}):
        raise ValueError("local edit proposal payload is invalid")
    candidate_id, batch_index = payload["candidate_id"], payload["batch_index"]
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("local edit proposal scope is invalid")
    if isinstance(batch_index, bool) or not isinstance(batch_index, int):
        raise TypeError("batch_index must be an integer")
    if batch_index < 0:
        raise ValueError("local edit proposal scope is invalid")
    return candidate_id, batch_index


def validate_local_edit_event(payload, *, evaluation, schedule, trajectory, comparison):
    from ..evolution.local_edits import LocalEditProposal

    _, batch_index = local_edit_key(payload)
    paired = schedule.local_evaluation_mode == PAIRED_LOCAL_EVALUATION_MODE
    if paired:
        if trajectory is None or trajectory.status is not TrajectoryStatus.RUNNING:
            raise ValueError("paired local edit requires a running trajectory")
        if batch_index >= trajectory.batch_count - 1:
            raise ValueError("paired final batch cannot record a local edit proposal")
    proposal = LocalEditProposal.from_dict(payload["proposal"])
    operations = [dict(item) for item in proposal.operations]
    decision = proposal.decision
    if (evaluation is None or (paired and comparison is None)
        or payload["evidence_scope_digest"] != evaluation.scope.scope_key
        or len(operations) > schedule.max_local_edits_per_batch
        or (decision is LocalEditProposalDecision.KEEP and operations)
        or (decision is LocalEditProposalDecision.MUTATE and not operations)):
        raise ValueError("local edit proposal evidence or operation count is invalid")
    if "safety_reason" in payload and (
        decision is not LocalEditProposalDecision.KEEP
        or not isinstance(payload["safety_reason"], str)
        or not payload["safety_reason"].strip()
    ):
        raise ValueError("local edit safety reason requires a keep proposal")
    return {**payload, "proposal": proposal.to_dict(),
            "decision": decision.value, "operations": operations}
