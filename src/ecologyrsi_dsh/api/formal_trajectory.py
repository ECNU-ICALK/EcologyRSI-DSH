"""Restart-safe execution of one Top-2 finalist's adaptive formal lane."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..core.models import Candidate, RunStatus, canonical_json, digest
from ..core.trajectory import (
    BatchEvaluation,
    CandidateRevision,
    EvaluationPhase,
    EvaluationScope,
    LocalEditOutcome,
    RevisionAdvanceReason,
    RevisionStatus,
    TrajectoryStatus,
)
from ..evolution.genome import EcologyEvolutionPluginGenome, deep_thaw_json
from ..evolution.local_edits import (
    LocalEditContext,
    LocalEditProposal,
    apply_or_reject_local_edit_bundle,
)
from ..evolution.strategies import (
    _genome_parameter_boundary,
    _registered_mutation_targets,
)
from ..evolution.schedule import OptimizationSchedule
from ..evolution.workflow_ir import resolve_candidate_agent_profile
from ..knowledge.algorithms import AlgorithmSpec, compile_algorithm_spec
from ..knowledge.program_registry import current_program_registry
from ..evaluators.registry import EvaluatorRegistry
from ..integrations.dsh_native_runtime import DshNativeRuntimeUnavailableError
from ..integrations.dsh_structured_roles import DshStructuredRoleRuntime
from .generation_execution import _director_mutation, _phase_task_manifest


def _candidate_revision(state: Any, candidate_id: str, revision_id: str) -> CandidateRevision:
    revision = state.revision(revision_id)
    if revision.candidate_id != candidate_id:
        raise ValueError("candidate revision belongs to another finalist")
    return revision


def _revision_evaluation_inputs(
    state: Any,
    candidate: Candidate,
    revision_id: str,
    task: Any,
) -> tuple[CandidateRevision, Any, AlgorithmSpec]:
    """Materialize evaluator inputs from the active immutable revision."""

    revision = _candidate_revision(state, candidate.candidate_id, revision_id)
    genome = EcologyEvolutionPluginGenome.from_dict(dict(revision.genome))
    base = state.proposal(candidate.proposal_id)
    metadata = dict(base.metadata)
    metadata.update(
        {
            "evolution_genome_canonical_json": canonical_json(genome.to_dict()),
            "genome_digest": genome.genome_digest,
            "behavior_digest": genome.behavior_digest,
            "mutation_digest": genome.lineage["mutation_digest"],
            "candidate_agent_profile": resolve_candidate_agent_profile(
                genome, current_program_registry()
            ),
        }
    )
    proposal = replace(
        base,
        generation=candidate.generation,
        changes=dict(genome.scientific_program["parameter_overrides"]),
        metadata=metadata,
    )
    spec = compile_algorithm_spec(
        task,
        proposal,
        state.knowledge_for(candidate.generation),
    )
    return revision, proposal, spec


def ensure_formal_trajectory(endpoint: Any, run_id: str, candidate_id: str):
    state = endpoint.server.director.state(run_id)
    candidate = state.candidate(candidate_id)
    formal = state.formal_selection_for(candidate.generation)
    if formal is None or candidate_id not in formal.payload["selected_candidate_ids"]:
        raise ValueError("formal trajectory requires frozen Top-2 selection")
    existing = state.trajectory_for(candidate_id)
    if existing is not None:
        return existing
    revision = state.initial_revision_for(candidate_id)
    if revision is None:
        raise RuntimeError("finalist is missing immutable R0")
    schedule = OptimizationSchedule.from_dict(
        state.task_manifest.metadata["optimization_schedule"]
    )
    return _director_mutation(
        endpoint,
        "start_formal_trajectory",
        run_id,
        candidate_id,
        revision.revision_id,
        schedule.batch_count,
    )


def _scope_for_batch(state: Any, candidate: Candidate, revision_id: str, batch: Any):
    return EvaluationScope(
        run_id=state.run.run_id,
        generation=candidate.generation,
        candidate_id=candidate.candidate_id,
        candidate_revision_id=revision_id,
        phase=EvaluationPhase.FORMAL_BATCH,
        cohort_digest=batch.cohort_digest,
        origin_count=batch.origin_count,
        batch_index=batch.batch_index,
    )


def execute_next_formal_batch(endpoint: Any, run_id: str, candidate_id: str) -> bool:
    state = endpoint.server.director.state(run_id)
    trajectory = ensure_formal_trajectory(endpoint, run_id, candidate_id)
    if trajectory.status is TrajectoryStatus.COMPLETED:
        return False
    candidate = state.candidate(candidate_id)
    adaptation = state.run_adaptation_cohort
    cohorts = state.generation_cohort_for(candidate.generation)
    if adaptation is None or cohorts is None:
        raise RuntimeError("formal batch requires frozen adaptation and generation cohorts")
    batch_index = None
    for index in range(trajectory.batch_count):
        existing_batch = state.formal_batch_for(candidate_id, index)
        evaluation = state.batch_evaluation_for(candidate_id, index)
        if existing_batch is None or evaluation is None:
            batch_index = index
            break
        # A completed batch must be locally decided and activated before the
        # next revision can be evaluated.  Returning here lets the scheduler
        # spend its next turn on the pending local-edit/recovery boundary.
        if (
            state.local_edit_proposal_for(candidate_id, index) is None
            or next(
                (
                    item
                    for item in state.local_edit_outcomes
                    if item.get("candidate_id") == candidate_id
                    and item.get("batch_index") == index
                ),
                None,
            )
            is None
            or state.revision_activation_for(candidate_id, index) is None
        ):
            return False
    if batch_index is None:
        return False
    active_revision_id = (
        trajectory.initial_revision_id
        if batch_index == 0
        else state.revision_activation_for(candidate_id, batch_index - 1).to_revision_id
    )
    planned_batch = adaptation.batches[batch_index]
    formal_batch = _director_mutation(
        endpoint,
        "start_formal_batch",
        run_id,
        candidate_id,
        active_revision_id,
        batch_index,
    )
    state = endpoint.server.director.state(run_id)
    if state.batch_evaluation_for(candidate_id, batch_index) is not None:
        return False
    candidate = state.candidate(candidate_id)
    scope = _scope_for_batch(state, candidate, active_revision_id, formal_batch)
    task = _phase_task_manifest(state.task_manifest, candidate.generation, "formal")
    _revision, proposal, compiled = _revision_evaluation_inputs(
        state, candidate, active_revision_id, task
    )

    def sample_run_control() -> str:
        status = endpoint.server.director.state(run_id).run.status
        if status is RunStatus.RUNNING:
            return "running"
        if status is RunStatus.PAUSED:
            return "paused"
        return "cancelled"

    if isinstance(endpoint.server.evaluators, EvaluatorRegistry):
        bundle = endpoint.server.evaluators.evaluate_scientific(
            task,
            candidate,
            proposal,
            scope=scope,
            cohort=planned_batch.cohort,
            algorithm_spec=compiled,
            on_sample_control=sample_run_control,
        )
    else:
        bundle = endpoint.server.evaluators.evaluate_scientific(
            task,
            candidate,
            proposal,
            scope=scope,
            cohort=planned_batch.cohort,
            on_sample_control=sample_run_control,
    )
    metrics = dict(bundle.evaluation.metrics)
    summary = metrics.get("sample_execution")
    if not isinstance(summary, Mapping) or int(
        summary.get("attempted_origin_samples", 0)
    ) < scope.origin_count:
        raise RuntimeError("formal batch did not complete its frozen origin cohort")
    _director_mutation(
        endpoint,
        "record_formal_batch_evaluation",
        run_id,
        BatchEvaluation(
            evaluation_id=f"formal-evaluation:{candidate_id}:{batch_index}",
            scope=scope,
            score=bundle.evaluation.score,
            passed=bundle.evaluation.passed,
            metrics=metrics,
            evaluator_digest=digest({"evaluator": bundle.evaluation.evaluator_digest}),
        ),
    )
    return True


def _local_edit_context(state: Any, candidate: Candidate, revision: CandidateRevision, batch: Any) -> LocalEditContext:
    genome = EcologyEvolutionPluginGenome.from_dict(dict(revision.genome))
    # Use the same Host-owned registered target catalog exposed to the proposer.
    targets = _registered_mutation_targets(state.task_manifest, genome)
    registry = current_program_registry()
    workflow_id = str(
        genome.agent_program["candidate_execution_program"]["workflow_template_ref"]["id"]
    )
    targets = {
        **targets,
        "workflow_template": tuple(
            item
            for item in registry.program_ids("workflow_templates")
            if item != workflow_id
            and "sample-planner"
            in registry.program("workflow_templates", item)["graph"]["allowed_roles"]
        ),
        "workflow_parameter": tuple(
            sorted(registry.program("workflow_templates", workflow_id)["parameters"])
        ),
        "feature_policy": registry.program_ids("feature_policies"),
        "fit_policy": registry.program_ids("fit_policies"),
        "uncertainty_policy": registry.program_ids("uncertainty_policies"),
        "instruction_tool_policy": tuple(
            sorted(
                str(profile["role"])
                for profile in genome.agent_program["candidate_execution_program"]["role_profiles"]
            )
        ),
        "instruction_parameter": tuple(
            sorted(
                {
                    name
                    for profile in genome.agent_program["candidate_execution_program"]["role_profiles"]
                    for name in registry.program(
                        "instruction_templates", profile["instruction_template_ref"]["id"]
                    )["parameters"]
                }
            )
        ),
    }
    _boundary, parameter_schemas = _genome_parameter_boundary(state.task_manifest, genome)
    cells = tuple(
        f"{target}@{horizon}h"
        for target in ("air_temperature", "relative_humidity", "co2_concentration")
        for horizon in (1, 6, 24)
    )
    return LocalEditContext(
        run_id=state.run.run_id,
        generation=candidate.generation,
        candidate_id=candidate.candidate_id,
        candidate_revision_id=revision.revision_id,
        batch_index=batch.batch_index,
        evidence_scope_digest=state.batch_evaluation_for(
            candidate.candidate_id, batch.batch_index
        ).scope.scope_key,
        parent_genome_digest=revision.genome_digest,
        maximum_operations=OptimizationSchedule.from_dict(
            state.task_manifest.metadata["optimization_schedule"]
        ).max_local_edits_per_batch,
        allowed_mutation_targets=targets,
        allowed_evidence_refs=("batch:score", "batch:metrics"),
        allowed_effect_cells=cells,
        parameter_schemas=parameter_schemas,
    )


def execute_next_local_edit(endpoint: Any, run_id: str, candidate_id: str) -> bool:
    state = endpoint.server.director.state(run_id)
    candidate = state.candidate(candidate_id)
    trajectory = state.trajectory_for(candidate_id)
    if trajectory is None:
        raise RuntimeError("local edit requires a formal trajectory")
    # Recover the narrow crash window after ``LocalEditDecided`` but before
    # ``TrajectoryRevisionAdvanced``.  The outcome contains the exact active
    # revision and is therefore sufficient to replay the activation without
    # asking DSH for a second proposal.
    recovery = next(
        (
            (batch, outcome)
            for batch in state.formal_batches
            if batch.candidate_id == candidate_id
            and state.batch_evaluation_for(candidate_id, batch.batch_index) is not None
            and state.revision_activation_for(candidate_id, batch.batch_index) is None
            for outcome in state.local_edit_outcomes
            if outcome.get("candidate_id") == candidate_id
            and outcome.get("batch_index") == batch.batch_index
        ),
        None,
    )
    if recovery is not None:
        pending_batch, outcome = recovery
        reason = {
            LocalEditOutcome.KEPT.value: RevisionAdvanceReason.KEPT,
            LocalEditOutcome.APPLIED.value: RevisionAdvanceReason.LOCAL_EDIT_APPLIED,
            LocalEditOutcome.REJECTED.value: RevisionAdvanceReason.LOCAL_EDIT_REJECTED,
        }[outcome["outcome"]]
        _director_mutation(
            endpoint,
            "advance_trajectory_revision",
            run_id,
            candidate_id,
            pending_batch.batch_index,
            outcome["active_revision_id"],
            reason,
        )
        if pending_batch.batch_index == trajectory.batch_count - 1:
            _director_mutation(
                endpoint,
                "complete_formal_trajectory",
                run_id,
                candidate_id,
                outcome["active_revision_id"],
            )
        return True
    pending = next(
        (
            batch
            for batch in state.formal_batches
            if batch.candidate_id == candidate_id
            and state.batch_evaluation_for(candidate_id, batch.batch_index) is not None
            and next(
                (
                    item
                    for item in state.local_edit_outcomes
                    if item.get("candidate_id") == candidate_id
                    and item.get("batch_index") == batch.batch_index
                ),
                None,
            )
            is None
        ),
        None,
    )
    if pending is None:
        return False
    revision = state.revision(pending.revision_id)
    context = _local_edit_context(state, candidate, revision, pending)
    recorded = state.local_edit_proposal_for(candidate_id, pending.batch_index)
    if recorded is None:
        proposal = _local_edit_proposal(endpoint, state, candidate, pending, context)
    else:
        # A process/ledger failure may have committed the proposal before the
        # child revision or decision.  Reconstruct the validated, minimal
        # proposal from the durable event and finish that exact work unit
        # instead of asking DSH to author a different edit on resume.
        proposal = LocalEditProposal(
            decision=recorded["decision"],
            operations=tuple(recorded["operations"]),
            evidence_refs=("batch:score",),
            expected_effect_cells=(),
            risk_cells=(),
        )
    validated = apply_or_reject_local_edit_bundle(
        EcologyEvolutionPluginGenome.from_dict(dict(revision.genome)),
        proposal,
        context,
        current_program_registry(),
    )
    active_revision_id = revision.revision_id
    advance_reason = RevisionAdvanceReason.KEPT
    if validated.child is not None:
        child = validated.child
        child_revision = CandidateRevision(
            revision_id=f"revision:{candidate_id}:batch:{pending.batch_index + 1}",
            run_id=run_id,
            generation=candidate.generation,
            candidate_id=candidate_id,
            genome=child.to_dict(),
            genome_digest=child.genome_digest,
            behavior_digest=child.behavior_digest,
            mutation_digest=str(child.lineage["mutation_digest"]),
            parent_revision_id=revision.revision_id,
            source_batch_index=pending.batch_index,
            status=RevisionStatus.ACTIVE,
        )
        _director_mutation(endpoint, "create_candidate_revision", run_id, child_revision)
        active_revision_id = child_revision.revision_id
        advance_reason = RevisionAdvanceReason.LOCAL_EDIT_APPLIED
    elif validated.outcome is LocalEditOutcome.REJECTED:
        advance_reason = RevisionAdvanceReason.LOCAL_EDIT_REJECTED
    if recorded is None:
        _director_mutation(
            endpoint,
            "record_local_edit_proposal",
            run_id,
            {
                "proposal_id": f"local-edit:{candidate_id}:{pending.batch_index}",
                "candidate_id": candidate_id,
                "batch_index": pending.batch_index,
                "evidence_scope_digest": context.evidence_scope_digest,
                "decision": proposal.decision.value,
                "operations": [dict(item) for item in proposal.operations],
            },
        )
    _director_mutation(
        endpoint,
        "decide_local_edit",
        run_id,
        {
            "proposal_id": f"local-edit:{candidate_id}:{pending.batch_index}",
            "candidate_id": candidate_id,
            "batch_index": pending.batch_index,
            "outcome": validated.outcome.value,
            "active_revision_id": active_revision_id,
        },
    )
    _director_mutation(
        endpoint,
        "advance_trajectory_revision",
        run_id,
        candidate_id,
        pending.batch_index,
        active_revision_id,
        advance_reason,
    )
    if pending.batch_index == trajectory.batch_count - 1:
        _director_mutation(
            endpoint,
            "complete_formal_trajectory",
            run_id,
            candidate_id,
            active_revision_id,
        )
    return True


def _local_edit_proposal(
    endpoint: Any,
    state: Any,
    candidate: Candidate,
    batch: Any,
    context: LocalEditContext,
) -> LocalEditProposal:
    """Ask the native local editor when present, with a safe local fallback."""

    # Zero is an explicit Host-owned switch that disables local mutation for
    # this schedule.  Resolve it before inspecting native-runtime state so a
    # disabled editor cannot reserve a DSH child request or consume tokens.
    if context.maximum_operations == 0:
        return LocalEditProposal(
            decision="keep",
            operations=(),
            evidence_refs=("batch:score",),
            expected_effect_cells=(),
            risk_cells=(),
        )
    runtime = getattr(endpoint.server, "dsh_native_runtime", None)
    admission = getattr(endpoint.server, "dsh_tools", None)
    identity = state.candidate_identity_binding(candidate.candidate_id)
    if runtime is None or identity is None:
        return LocalEditProposal(
            decision="keep",
            operations=(),
            evidence_refs=("batch:score",),
            expected_effect_cells=(),
            risk_cells=(),
        )
    evaluation = state.batch_evaluation_for(candidate.candidate_id, batch.batch_index)
    if evaluation is None:
        raise RuntimeError("local editor requires completed batch evidence")
    metrics = _local_edit_evidence_metrics(evaluation.metrics)
    context_payload = {
        **context.to_dict(),
        "batch_evidence": {
            "score": evaluation.score,
            "passed": evaluation.passed,
            "metrics": metrics,
            "scope_digest": evaluation.scope.scope_key,
        },
    }
    try:
        result = DshStructuredRoleRuntime(runtime, admission=admission).run(
            run_id=state.run.run_id,
            stage="candidate.local_edit",
            role="candidate-proposer",
            context=context_payload,
            output_schema_id="ecology-local-edit@1",
            run_state_revision=state.events[-1].seq,
            stage_attempt=1,
            ledger_expected_revision=endpoint.server.ledger.latest_seq(),
            idempotency_key=(
                f"{state.run.run_id}:candidate:{candidate.candidate_id}:"
                f"batch:{batch.batch_index}:local-edit"
            ),
            identity_digests={
                key: str(identity[key])
                for key in ("genome_digest", "compiled_behavior_digest", "phenotype_instance_digest")
                if key in identity
            },
        )
    except DshNativeRuntimeUnavailableError:
        # A test/local run without the native process keeps the trajectory
        # resumable. Real native runs surface other failures to auto-progress.
        return LocalEditProposal(
            decision="keep",
            operations=(),
            evidence_refs=("batch:score",),
            expected_effect_cells=(),
            risk_cells=(),
        )
    return LocalEditProposal.from_dict(result)


def _local_edit_evidence_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return detached aggregate evidence safe for the native JSON boundary.

    Trajectory models recursively freeze their metrics, so a shallow ``dict``
    leaves nested ``MappingProxyType`` instances behind.  The native DSH
    request encoder correctly rejects those non-JSON objects.  Thaw the whole
    tree, then remove sample-level evidence that the bounded local editor is
    neither allowed nor expected to inspect.
    """

    detached = deep_thaw_json(metrics)
    if not isinstance(detached, dict):
        raise TypeError("local edit metrics must be a JSON object")
    for key in (
        "sample_execution_records",
        "sample_execution_trace_archive",
        "prediction_preview",
    ):
        detached.pop(key, None)
    return detached


def execute_formal_trajectory(endpoint: Any, run_id: str, candidate_id: str) -> None:
    ensure_formal_trajectory(endpoint, run_id, candidate_id)
    while execute_next_formal_batch(endpoint, run_id, candidate_id):
        execute_next_local_edit(endpoint, run_id, candidate_id)
    while execute_next_local_edit(endpoint, run_id, candidate_id):
        pass


__all__ = [
    "ensure_formal_trajectory",
    "execute_formal_trajectory",
    "execute_next_formal_batch",
    "execute_next_local_edit",
]
