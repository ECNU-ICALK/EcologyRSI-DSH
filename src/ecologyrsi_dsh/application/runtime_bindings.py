"""Validated candidate identity and event cursors shared by application runtimes."""
from __future__ import annotations
import threading
from typing import Any
from collections.abc import Callable, Mapping
from ..core.ledger import EventLedger
from ..core.state import validate_identity_binding


def candidate_revision_identity_binding(state, candidate_id: str, revision_id: str):
    """Compile a persisted revision under the run's original frozen contracts.

    CandidateSpawned remains the immutable R0 identity. A local revision must
    never overwrite it or borrow its compiled/phenotype identity.
    """
    from .formal_trajectory import _revision_evaluation_inputs
    from ..evolution.workflow_ir import (
        CompilationInstanceContext, bind_phenotype_instance, compile_plugin_behavior,
    )
    from ..evolution.genome import EcologyEvolutionPluginGenome
    from ..knowledge.program_registry import current_program_registry

    candidate = state.candidate(candidate_id)
    revision = state.revision(revision_id)
    if revision.candidate_id != candidate_id:
        raise ValueError("candidate revision belongs to another candidate")
    original = state.candidate_identity_binding(candidate_id)
    if original is None:
        raise ValueError("candidate has no native identity binding")
    original = validate_identity_binding(original)
    if revision.genome_digest == original["genome_digest"]:
        return dict(original)
    task = state.task_manifest
    revision, proposal, _ = _revision_evaluation_inputs(state, candidate, revision_id, task)
    genome = EcologyEvolutionPluginGenome.from_dict(dict(revision.genome))
    compiled = compile_plugin_behavior(
        genome, task, state.knowledge_for(candidate.generation), current_program_registry(),
        compiler_semantic_digest=original["compiler_semantic_digest"],
    )
    coordinates = {
        "run_id": state.run.run_id, "proposal_id": proposal.proposal_id,
        "candidate_id": candidate_id, "generation": candidate.generation,
        "slot_index": candidate.slot_index, "task_manifest_digest": task.digest,
        "agent_policy_digest": proposal.metadata["agent_policy"]["policy_digest"],
    }
    context = CompilationInstanceContext(**{
        name: coordinates[name] if name in coordinates else task.metadata[name]
        for name in CompilationInstanceContext.__dataclass_fields__
    })
    bound = bind_phenotype_instance(compiled, context)
    return validate_identity_binding({
        **original,
        "genome_digest": genome.genome_digest,
        "behavior_digest": genome.behavior_digest,
        "compiled_behavior_digest": compiled.compiled_behavior_digest,
        "phenotype_instance_digest": bound.phenotype_instance_digest,
        "registry_catalog_digest": compiled.registry_catalog_digest,
        "security_semantic_digest": compiled.security_semantic_digest,
        "runtime_execution_digest": bound.runtime_execution_digest,
    })

def dsh_revision_snapshot(ledger: EventLedger, run_id: str) -> dict[str, int]:
    """Read DSH cursors without materializing the run projection."""

    run_state_revision = ledger.latest_run_seq(run_id)
    if run_state_revision == 0:
        raise KeyError(f"unknown run: {run_id}")
    return {
        "run_state_revision": run_state_revision,
        "ledger_expected_revision": ledger.latest_seq(),
    }


class ValidatedCandidateIdentityCache:
    """Cache immutable candidate bindings after one trusted state replay.

    Candidate identity is fixed by ``CandidateSpawned`` and every later event
    can only repeat that exact binding.  The first miss for a run therefore
    uses the full projector (retaining all genome, compiler, and tamper
    checks), then serves defensive copies.  A miss after the run cursor moves
    refreshes the snapshot so candidates spawned in later generations remain
    discoverable after normal execution or process recovery.
    """

    def __init__(
        self,
        ledger: EventLedger,
        state_provider: Callable[[str], Any],
    ) -> None:
        self._ledger = ledger
        self._state_provider = state_provider
        self._guard = threading.Lock()
        self._run_locks: dict[str, threading.RLock] = {}
        self._snapshots: dict[
            str, tuple[int, dict[str, dict[str, str]]]
        ] = {}
        self._revision_bindings: dict[tuple[str, str, str], dict[str, str]] = {}

    @staticmethod
    def _key(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    def _run_lock(self, run_id: str) -> threading.RLock:
        with self._guard:
            lock = self._run_locks.get(run_id)
            if lock is None:
                lock = threading.RLock()
                self._run_locks[run_id] = lock
            return lock

    def get(self, run_id: str, candidate_id: str,
            revision_id: str | None = None) -> dict[str, str] | None:
        run_id = self._key(run_id, "run_id")
        candidate_id = self._key(candidate_id, "candidate_id")
        if revision_id is not None:
            revision_id = self._key(revision_id, "revision_id")
            key = (run_id, candidate_id, revision_id)
            with self._run_lock(run_id):
                with self._guard:
                    cached = self._revision_bindings.get(key)
                if cached is None:
                    state = self._state_provider(run_id)
                    if state.run.run_id != run_id:
                        raise ValueError("revision identity projection belongs to another run")
                    cached = candidate_revision_identity_binding(state, candidate_id, revision_id)
                    with self._guard:
                        self._revision_bindings[key] = dict(cached)
                return dict(cached)
        with self._run_lock(run_id):
            with self._guard:
                snapshot = self._snapshots.get(run_id)
                cached = None if snapshot is None else snapshot[1].get(candidate_id)
            if cached is not None:
                return dict(cached)

            latest_run_seq = self._ledger.latest_run_seq(run_id)
            if latest_run_seq == 0:
                raise KeyError(f"unknown run: {run_id}")
            if snapshot is not None and snapshot[0] == latest_run_seq:
                return None

            # Keep the per-run lock across replay: concurrent sample workers
            # share one validation pass instead of replaying the same stream.
            state = self._state_provider(run_id)
            if state.run.run_id != run_id or not state.events:
                raise ValueError("candidate identity projection belongs to another run")
            projected_seq = int(state.events[-1].seq)
            if projected_seq < latest_run_seq:
                raise RuntimeError("candidate identity projection is behind the ledger")
            bindings: dict[str, dict[str, str]] = {}
            for item in state.candidate_identity_bindings:
                if not isinstance(item, Mapping):
                    raise ValueError("candidate identity cache entry is invalid")
                item_candidate_id = self._key(
                    item.get("candidate_id"), "candidate_id"
                )
                binding = validate_identity_binding(item.get("identity_binding"))
                existing = bindings.get(item_candidate_id)
                if existing is not None and existing != binding:
                    raise ValueError("candidate has conflicting identity bindings")
                bindings[item_candidate_id] = binding
            with self._guard:
                self._snapshots[run_id] = (projected_seq, bindings)
            result = bindings.get(candidate_id)
            return dict(result) if result is not None else None

    def forget(self, run_id: str) -> bool:
        run_id = self._key(run_id, "run_id")
        with self._run_lock(run_id), self._guard:
            keys = [key for key in self._revision_bindings if key[0] == run_id]
            for key in keys:
                del self._revision_bindings[key]
            return self._snapshots.pop(run_id, None) is not None or bool(keys)

    def clear(self) -> None:
        with self._guard:
            self._snapshots.clear()
            self._revision_bindings.clear()
