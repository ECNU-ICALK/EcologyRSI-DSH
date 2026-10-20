"""Explicit local doubles for the supported native contract; never production fallbacks."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from ecologyrsi_dsh.core.models import TaskManifest, digest
from ecologyrsi_dsh.core.search_policy import SEARCH_GUARD_POLICY, PAIRED_EXECUTION_QUALIFICATION
from ecologyrsi_dsh.core.model_preflight import AUDIT_METADATA_KEY, AUDIT_SCHEMA
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.integrations.dsh_native_runtime import DSH_NATIVE_EXECUTION_PROTOCOL
from ecologyrsi_dsh.integrations.model_canary import (
    RECEIPT_SCHEMA, SCOPE, canary_request, required_canary_identities,
)
from ecologyrsi_dsh.version import __version__


def native_manifest(*, task_id="native-fixture", generations=1, metadata=None):
    """A frozen current contract for ledger tests, without opening dataset labels."""
    values = {
        "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
        "optimization_protocol": "joint_plugin_epoch@2",
        "search_guard_policy": SEARCH_GUARD_POLICY,
        "paired_execution_qualification": PAIRED_EXECUTION_QUALIFICATION,
        "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
        "seed_genome_template_id": "greenhouse-default@2",
        "prediction_model_id": "greenhouse-baseline-aligned-ridge@1",
        "evaluator_id": "greenhouse_multihorizon_time_forward@4",
        "strategy_id": "autonomous_model@1", "autonomous_mode": True,
        "strategy_model_id": "fixture/strategy", "review_model_id": "fixture/review",
        "sample_agent_protocol": "dsh-strict-origin-bundle@4",
        "sample_agent_mode": "dsh_native_agent", "sample_budget_class": "selection_eligible",
        "prediction_cells_per_origin": 9,
        "require_model_contract_preflight": True,
        AUDIT_METADATA_KEY: AUDIT_SCHEMA,
        "host_runtime_build": {"package_version": __version__,
            "evolution_runtime_schema": "ecologyrsi-dsh.evolution-runtime/3",
            "generation_comparison_schema": "ecologyrsi-dsh.generation-comparison/1",
            "projection_schema": "ecologyrsi-dsh.execution-projection/2"},
    }
    for index, key in enumerate((
        "dataset_digest", "dataset_snapshot_set_digest", "split_manifest_digest",
        "data_protocol_digest", "stage_policy_digest", "evaluator_digest",
        "fitness_profile_digest", "security_kernel_digest", "selection_reviewer_program_digest",
        "required_capability_digest", "resolved_policy_route_digest", "resolved_review_route_digest",
        "resolved_policy_route_config_digest", "resolved_review_route_config_digest",
        "preset_content_digest", "standing_tool_surface_digest", "evaluation_cohort_digest",
    )):
        values[key] = digest({"fixture": key, "index": index})
    values.update(metadata or {})
    return TaskManifest(task_id=task_id, objective="predict greenhouse climate",
        domain_pack="greenhouse_environment@1", visible_datasets=("agc_cucumber_2018",),
        budget={"max_candidates": 4 * generations, "max_generations": generations,
                "candidates_per_generation": 4}, metadata=values)


def canary_receipts(metadata, *_args, **_kwargs):
    """Four distinct, validator-compatible local receipts; no model request."""
    now = datetime.now(timezone.utc)
    receipts = []
    for identity in required_canary_identities(metadata):
        request = canary_request(identity)
        role = identity["role"]
        receipts.append({
            "schema_version": RECEIPT_SCHEMA, "scope": SCOPE,
            "receipt_id": f"canary:fixture-{role}", "identity": identity,
            "identity_digest": digest(identity), "bounds": request["bounds"],
            "started_at": (now - timedelta(seconds=1)).isoformat(),
            "completed_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=request["bounds"]["ttl_seconds"])).isoformat(),
            "elapsed_ms": 1000, "passed": True,
            "output_schema_digest": digest({"schema": identity["output_schema_id"]}),
            "result_digest": digest({"fixture": role}),
            "route_binding": "role_host_explicit_provider_and_model",
            "tool_evidence": {"first_tool_call_verified": True, "order_verified": True,
                "successful_call_count": 1, "next_tool_name": "structured_output",
                "stage": identity["stage"], "source": "dsh_session_event_log"},
            "attempts": [{"attempt": 1, "session_id": f"child:fixture-{role}",
                "elapsed_ms": 1000, "accepted": True}], "failure": None,
            "usage": {"complete": True, "reported_tokens": 1, "observed_session_count": 1,
                "launched_session_count": 1, "complete_session_count": 1},
        })
    return tuple(receipts)


class NativeRoleStub:
    """Lifecycle and explicit structured replies only; no synthetic evolution policy."""
    def __init__(self, replies=()):
        self.replies = list(replies)
        self.requests = []
        self.opened = []
        self.closed = []

    def open_session(self, run, task):
        self.opened.append((run.run_id, task.digest))
        return f"session:{run.run_id}"

    def close_session(self, session_id):
        self.closed.append(session_id)

    def run_stage(self, request):
        self.requests.append(deepcopy(request))
        if not self.replies:
            raise AssertionError("test must explicitly provide the next native role reply")
        structured = self.replies.pop(0)
        if callable(structured):
            structured = structured(request)
        if not isinstance(structured, Mapping):
            raise TypeError("native fixture reply must be structured")
        structured = deepcopy(dict(structured))
        return {"structured": structured, "result_digest": digest(structured)}

    def propose(self, *_args, **_kwargs):
        raise AssertionError("use StrategyRouterDSHAdapter with explicit native role replies")


def start_native_run(director, *, task=None, run_id="run:native-fixture"):
    task = task or native_manifest()
    director.create_run(task, run_id=run_id)
    director.record_model_contract_preflight(run_id, canary_receipts(task.metadata))
    director.start_run(run_id)
    return director.state(run_id)


def spawn_native_candidate(director, run_id, *, slot=None):
    """Append an explicit bounded Genome fixture to an already started run."""
    from ecologyrsi_dsh.core.models import Proposal, canonical_json, CandidateRole
    from ecologyrsi_dsh.evolution.genome import GenomeMutationContextV1, apply_genome_mutation
    from ecologyrsi_dsh.evolution.workflow_ir import resolve_candidate_agent_profile
    from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
    state = director.state(run_id)
    if state.run.generation != 0:
        raise ValueError("fixture requires an explicit later-generation parent")
    if slot is None:
        slot = len([c for c in state.candidates if c.role is CandidateRole.SEARCH])
    parent = state.materialized_seed_genome()
    context = GenomeMutationContextV1(run_id,0,slot,slot,None,parent.genome_digest,
        *('a'*64,)*4,'test-mutation@1')
    registry = current_program_registry()
    genome = apply_genome_mutation(parent, {'schema_version':'ecologyrsi-dsh.genome-mutation/1',
        'operations':[{'op':'set_bounded_parameter','name':'ridge_alpha','value':.2+.1*slot}]},context,registry)
    proposal = Proposal(f'proposal:{run_id}:{slot}',run_id,0,'bounded fixture',
        dict(genome.scientific_program['parameter_overrides']), metadata={
            'execution_protocol':'dsh_native_plugin_evolution@1',
            'evolution_genome_canonical_json':canonical_json(genome.to_dict()),
            'genome_digest':genome.genome_digest,'behavior_digest':genome.behavior_digest,
            'candidate_agent_profile':resolve_candidate_agent_profile(genome,registry)})
    director.submit_proposal(proposal)
    return director.spawn_candidate(run_id,proposal,candidate_id=f'candidate:{run_id}:{slot}',slot_index=slot)
