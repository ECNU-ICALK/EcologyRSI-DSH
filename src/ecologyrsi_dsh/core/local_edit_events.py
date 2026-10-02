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


def validate_local_effect_event(payload, *, revision, parent_revision, proposal, allowed_effect_cells=()):
    """Bind persisted effect evidence to the immutable parent, child and edit.

    Older/manual producers may omit both fields. A partially supplied contract
    is never accepted, and replay repeats the same validation as append.
    """
    from .models import canonical_json
    from .immutable import thaw_json
    from ..evolution.effect_contracts import build_edit_effect_contract, resolve_mutation_effects, mutation_effect_cells
    from ..evolution.genome import EcologyEvolutionPluginGenome
    from ..knowledge.program_registry import current_program_registry

    if not ({"effect_contract", "effect_resolution"} & set(payload)):
        return
    if (not {"effect_contract", "effect_resolution"}.issubset(payload)
            or payload["outcome"] != "applied" or parent_revision is None):
        raise ValueError("effect contract requires one applied child and its parent")
    from collections.abc import Mapping
    resolution = payload["effect_resolution"]
    if not isinstance(resolution, Mapping) or not isinstance(payload["effect_contract"], Mapping):
        raise ValueError("effect contract and resolution must be objects")
    resolved = resolve_mutation_effects(
        EcologyEvolutionPluginGenome.from_dict(thaw_json(parent_revision.genome)),
        EcologyEvolutionPluginGenome.from_dict(thaw_json(revision.genome)),
        thaw_json(proposal["operations"]), current_program_registry(),
        runtime_constraints=resolution.get("runtime_constraints"),
    )
    if canonical_json(resolution) != canonical_json(resolved):
        raise ValueError("persisted mutation effect does not match its immutable genomes")
    source = proposal["proposal"]
    expected = build_edit_effect_contract(
        parent_revision_id=parent_revision.revision_id,
        evidence_scope_digest=proposal["evidence_scope_digest"],
        evidence_refs=source["evidence_refs"],
        affected_cells=mutation_effect_cells(proposal["operations"], allowed_effect_cells),
        resolution=resolved,
    )
    if (revision.parent_revision_id != parent_revision.revision_id
            or canonical_json(payload["effect_contract"]) != canonical_json(expected)):
        raise ValueError("persisted effect contract does not match its local edit evidence")
