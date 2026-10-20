"""Recover only real, scope-bound synthesis submissions from the event ledger."""
from collections.abc import Mapping

from ..core.models import digest
from .genome import deep_thaw_json


def research_checkpoint(state, parent):
    """Find the last submission after this generation's search/knowledge freeze.

    A structured-tool receipt proves submission, not semantic validity. The
    caller must revalidate the entire object before using it or requesting a
    correction. Plain assistant text and other generations are never eligible.
    """
    run = state.run
    search = state.search_plan_for(run.generation)
    knowledge = state.knowledge_for(run.generation)
    if search is None or knowledge is None:
        return None
    freeze_seq = 0
    for event in state.events:
        field = {"GenerationSearchPlanned": "search_plan",
                 "GenerationKnowledgeRetrieved": "knowledge_snapshot"}.get(event.kind)
        if field and event.payload.get(field, {}).get("generation") == run.generation:
            freeze_seq = max(freeze_seq, event.seq)
    if not freeze_seq:
        return None
    prefix = f"{run.run_id}:generation:{run.generation}:generation.research-synthesis:"
    failed_requests = {
        event.payload.get("identity", {}).get("idempotency_key")
        for event in state.events if event.kind == "DshChildExecutionFailed"
    }
    for event in reversed(state.events):
        if event.seq <= freeze_seq:
            break
        if event.kind != "DshStructuredResultAccepted":
            continue
        payload = event.payload
        identity = payload.get("identity", {})
        if not (
            event.run_id == run.run_id
            and identity.get("run_id") == run.run_id
            and identity.get("stage") == "generation.research-synthesis"
            and identity.get("role") == "researcher"
            and identity.get("genome_digest") == parent.genome_digest
            and str(identity.get("idempotency_key", "")).startswith(prefix)
            and identity.get("idempotency_key") not in failed_requests
            and type(identity.get("run_state_revision")) is int
            and freeze_seq <= identity["run_state_revision"] < event.seq
            and payload.get("output_schema_id") == "ecology-research-synthesis@1"
        ):
            continue
        structured = payload.get("structured")
        if not isinstance(structured, Mapping) or payload.get("result_digest") != digest(structured):
            raise ValueError("research checkpoint structured result digest mismatch")
        return {
            "run_id": run.run_id, "generation": run.generation,
            "parent_genome_digest": parent.genome_digest,
            "search_plan_digest": search.search_plan_digest,
            "knowledge_snapshot_digest": knowledge.snapshot_digest,
            "event_id": event.event_id, "event_seq": event.seq,
            "result_digest": payload["result_digest"],
            "structured": deep_thaw_json(structured),
        }
    return None


def validate_research_checkpoint(value, *, run, parent, search, knowledge):
    """Recheck scope at the adapter boundary; never accept a caller's raw text."""
    if value is None:
        return None
    expected = {
        "run_id": run.run_id, "generation": run.generation,
        "parent_genome_digest": parent.genome_digest,
        "search_plan_digest": search.search_plan_digest,
        "knowledge_snapshot_digest": knowledge["snapshot_digest"],
    }
    if not isinstance(value, Mapping) or any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("research checkpoint scope mismatch")
    structured = value.get("structured")
    if not isinstance(structured, Mapping) or value.get("result_digest") != digest(structured):
        raise ValueError("research checkpoint structured result digest mismatch")
    return deep_thaw_json(structured)
