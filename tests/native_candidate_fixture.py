"""Materialize a real bounded native candidate for persistence/query tests."""
from ecologyrsi_dsh.core.models import Proposal, canonical_json
from ecologyrsi_dsh.evolution.genome import (
    GenomeMutationContextV1, TRUST_REGION_MUTATION_OPERATOR_ID, apply_genome_mutation,
)
from ecologyrsi_dsh.evolution.workflow_ir import resolve_candidate_agent_profile
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.native_fixtures import native_manifest, start_native_run


def seed_candidate(director, run_id, candidate_id="candidate:fixture", *, task=None, slot=0):
    state = start_native_run(director, task=task or native_manifest(), run_id=run_id)
    parent = state.materialized_seed_genome()
    context = GenomeMutationContextV1(run_id=run_id, generation=0, slot_index=slot,
        slot_seed=17, parent_candidate_id=None, parent_genome_digest=parent.genome_digest,
        generation_batch_digest="a" * 64, research_iteration_digest="b" * 64,
        knowledge_snapshot_digest="c" * 64, mutation_budget_digest="d" * 64,
        mutation_operator_id=TRUST_REGION_MUTATION_OPERATOR_ID)
    registry = current_program_registry()
    genome = apply_genome_mutation(parent, {"schema_version": "ecologyrsi-dsh.genome-mutation/1",
        "operations": [{"op": "set_bounded_parameter", "name": "ridge_alpha", "value": .2}]},
        context, registry)
    proposal = Proposal(proposal_id=f"proposal:{run_id}:{slot}", run_id=run_id,
        generation=0, title="Bounded native fixture", changes=dict(genome.scientific_program["parameter_overrides"]),
        metadata={"execution_protocol": "dsh_native_plugin_evolution@1",
            "evolution_genome_canonical_json": canonical_json(genome.to_dict()),
            "genome_digest": genome.genome_digest, "behavior_digest": genome.behavior_digest,
            "candidate_agent_profile": resolve_candidate_agent_profile(genome, registry)})
    director.submit_proposal(proposal)
    return proposal, director.spawn_candidate(run_id, proposal, candidate_id=candidate_id, slot_index=slot)


def seed_proposal(task, run_id="run:compiler-fixture"):
    """A valid current control Genome without executing remote research."""
    from ecologyrsi_dsh.core.director import EvolutionDirector
    from ecologyrsi_dsh.core.ledger import EventLedger
    from tests.native_fixtures import NativeRoleStub
    with EventLedger() as ledger:
        director = EvolutionDirector(ledger, NativeRoleStub())
        start_native_run(director, task=task, run_id=run_id)
        candidate = director.ensure_seed_incumbent_control(run_id)
        return director.state(run_id).proposal(candidate.proposal_id)
