"""Restart-safe execution of one Top-2 finalist's adaptive formal lane."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

from ..core.models import Candidate, canonical_json, digest
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
    LocalEditResult,
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
from .generation_execution import (
    _ScopedEvaluationCallbacks,
    _director_mutation,
    _phase_task_manifest,
)


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


def _durable_batch_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Persist aggregate trajectory evidence without duplicate sample traces.

    DSH already records every structured child/tool boundary in the run ledger.
    Scientific evaluators also return an expanded record list, a UI preview,
    and one compressed canonical trace.  The first two are derived copies; the
    compressed trace is the only self-contained evidence from which the batch
    can be audited.  Retain that archive plus aggregate decision inputs, while
    dropping the expanded copies that made every status replay parse the same
    sample data several times.
    """

    detached = deep_thaw_json(metrics)
    if not isinstance(detached, dict):
        raise TypeError("formal batch metrics must be a JSON object")
    result = _local_edit_evidence_metrics(detached)
    for name in (
        "baseline_metrics_digest",
        "baseline_profile_digest",
        "dataset_digest",
        "evaluation_index_digest",
        "execution_scope_digest",
        "feedback_update_cohort_digest",
        "sample_execution_trace_digest",
        "split_manifest_digest_sha256",
        "reward_definition",
        "primary_fitness_definition",
        "objective_profile",
        "prediction_model_id",
        "causal_interpretation",
    ):
        if name in detached:
            result[name] = detached[name]
    archive = detached.get("sample_execution_trace_archive")
    if isinstance(archive, Mapping):
        # Do not reduce scientific auditability to a digest-only assertion:
        # DSH tool events intentionally store output digests, not the complete
        # predictions/observations used by the evaluator.
        result["sample_execution_trace_archive"] = deep_thaw_json(archive)
    return result


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

    callbacks = _ScopedEvaluationCallbacks(
        endpoint,
        run_id=run_id,
        generation=candidate.generation,
        proposal_id=proposal.proposal_id,
        candidate_id=candidate_id,
        scope=scope,
    )

    if isinstance(endpoint.server.evaluators, EvaluatorRegistry):
        bundle = endpoint.server.evaluators.evaluate_scientific(
            task,
            candidate,
            proposal,
            scope=scope,
            cohort=planned_batch.cohort,
            algorithm_spec=compiled,
            **callbacks.evaluation_kwargs(),
        )
    else:
        bundle = endpoint.server.evaluators.evaluate_scientific(
            task,
            candidate,
            proposal,
            scope=scope,
            cohort=planned_batch.cohort,
            **callbacks.evaluation_kwargs(),
        )
    metrics = dict(bundle.evaluation.metrics)
    summary = metrics.get("sample_execution")
    if not isinstance(summary, Mapping) or int(
        summary.get("attempted_origin_samples", 0)
    ) < scope.origin_count:
        raise RuntimeError("formal batch did not complete its frozen origin cohort")
    batch_evaluation = BatchEvaluation(
        evaluation_id=f"formal-evaluation:{candidate_id}:{batch_index}",
        scope=scope,
        score=bundle.evaluation.score,
        passed=bundle.evaluation.passed,
        metrics=_durable_batch_metrics(metrics),
        evaluator_digest=digest({"evaluator": bundle.evaluation.evaluator_digest}),
    )
    _director_mutation(
        endpoint,
        "record_formal_batch_evaluation",
        run_id,
        batch_evaluation,
        sample_results=callbacks.completion_payload(
            batch_evaluation.evaluation_id,
            bundle.sample_results,
        ),
    )
    return True


def _local_edit_context(state: Any, candidate: Candidate, revision: CandidateRevision, batch: Any) -> LocalEditContext:
    genome = EcologyEvolutionPluginGenome.from_dict(dict(revision.genome))
    # Use the same Host-owned registered target catalog exposed to the proposer.
    targets = {
        axis: values
        for axis, values in _registered_mutation_targets(
            state.task_manifest,
            genome,
        ).items()
        # Replacing the complete prediction pipeline is a generation-level
        # search direction, not a small batch-local adjustment. Keeping it in
        # the local catalog allowed one 50-origin diagnostic window to replace
        # the finalist's whole algorithm and destabilize the remaining epoch.
        # The outer four-candidate proposer still owns this axis unchanged.
        if axis != "registered_predictor"
    }
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
            LocalEditOutcome.ROLLED_BACK.value: (
                RevisionAdvanceReason.PREQUENTIAL_SAFETY_ROLLBACK
            ),
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
        safety_reason = _prequential_safety_reason(
            state.batch_evaluation_for(candidate_id, pending.batch_index).metrics
        )
        proposal = (
            LocalEditProposal(
                decision="keep",
                operations=(),
                evidence_refs=("batch:score",),
                expected_effect_cells=(),
                risk_cells=(),
            )
            if safety_reason is not None
            else _local_edit_proposal(endpoint, state, candidate, pending, context)
        )
    else:
        # A process/ledger failure may have committed the proposal before the
        # child revision or decision.  Reconstruct the validated, minimal
        # proposal from the durable event and finish that exact work unit
        # instead of asking DSH to author a different edit on resume.
        proposal = LocalEditProposal.from_dict(recorded["proposal"])
        safety_reason = recorded.get("safety_reason")
    policy_rejection_reason = (
        None
        if safety_reason is not None
        else _local_edit_policy_rejection_reason(
            state,
            candidate_id,
            pending.batch_index,
            context.candidate_revision_id,
            proposal,
        )
    )
    safety_rollback = _safety_requires_rollback(safety_reason)
    # Commit the complete, schema-validated proposal before deriving a child
    # revision.  This closes the old crash window where replay only had the
    # child genome and a lossy operation list, and might ask for a new edit.
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
                "proposal": proposal.to_dict(),
                **({"safety_reason": safety_reason} if safety_reason is not None else {}),
            },
        )
    if safety_reason is not None:
        # The just-recorded batch is authoritative: do not mutate a lineage
        # after a coverage or constraint breach.  The outcome rolls back to
        # the parent when available and remains explicitly explained on R0.
        validated = None
    elif policy_rejection_reason is not None:
        # Preserve the authored proposal for audit, but never re-apply an
        # exact bundle that this same immutable parent revision already had
        # rejected.  This decision is derived only from durable prior batches,
        # so a crash after proposal persistence replays the same outcome.
        validated = LocalEditResult(
            outcome=LocalEditOutcome.REJECTED,
            operations=tuple(proposal.operations),
            child=None,
            proposal_digest=digest(proposal.to_dict()),
        )
    else:
        validated = apply_or_reject_local_edit_bundle(
            EcologyEvolutionPluginGenome.from_dict(dict(revision.genome)),
            proposal,
            context,
            current_program_registry(),
        )
    active_revision_id = revision.revision_id
    advance_reason = RevisionAdvanceReason.KEPT
    if validated is not None and validated.child is not None:
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
    elif safety_reason is not None:
        if safety_rollback and revision.parent_revision_id is not None:
            active_revision_id = revision.parent_revision_id
            advance_reason = RevisionAdvanceReason.PREQUENTIAL_SAFETY_ROLLBACK
    elif (
        validated is not None and validated.outcome is LocalEditOutcome.REJECTED
    ):
        advance_reason = RevisionAdvanceReason.LOCAL_EDIT_REJECTED
    _director_mutation(
        endpoint,
        "decide_local_edit",
        run_id,
        {
            "proposal_id": f"local-edit:{candidate_id}:{pending.batch_index}",
            "candidate_id": candidate_id,
            "batch_index": pending.batch_index,
            "outcome": (
                LocalEditOutcome.ROLLED_BACK.value
                if safety_rollback and revision.parent_revision_id is not None
                else LocalEditOutcome.KEPT.value
                if safety_reason is not None
                else validated.outcome.value
            ),
            "active_revision_id": active_revision_id,
            **(
                {"reason": safety_reason or policy_rejection_reason}
                if safety_reason is not None or policy_rejection_reason is not None
                else {}
            ),
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


def _prequential_safety_reason(metrics: Mapping[str, Any]) -> str | None:
    """Return a durable reason when a batch is unsafe to adapt from."""

    if not isinstance(metrics, Mapping):
        return "batch_metrics_invalid"
    violations = metrics.get("constraint_violations", 0)
    if isinstance(violations, bool) or not isinstance(violations, (int, float)):
        return "constraint_guardrail_invalid"
    if not math.isfinite(float(violations)) or float(violations) < 0:
        return "constraint_guardrail_invalid"
    if violations > 0:
        return "constraint_guardrail_failed"
    coverage = metrics.get("sample_execution_coverage_pass")
    sample = metrics.get("sample_execution")
    sample_coverage = sample.get("coverage_pass") if isinstance(sample, Mapping) else None
    failures = sample.get("failure_counts") if isinstance(sample, Mapping) else None
    rejected = (
        failures.get("constraint_rejected", 0)
        if isinstance(failures, Mapping)
        else 0
    )
    constraint_rejected = bool(
        not isinstance(rejected, bool)
        and isinstance(rejected, (int, float))
        and math.isfinite(float(rejected))
        and rejected > 0
    )
    if isinstance(sample, Mapping):
        strict_chain_pass = sample.get("strict_agent_chain_pass")
        if strict_chain_pass is False:
            return (
                "sample_constraint_guardrail_failed"
                if constraint_rejected
                else "strict_origin_chain_guardrail_failed"
            )
        attempted = sample.get("attempted_origin_samples")
        succeeded = sample.get("succeeded_origin_samples")
        minimum = sample.get("minimum_coverage")
        if (
            not isinstance(attempted, bool)
            and isinstance(attempted, (int, float))
            and math.isfinite(float(attempted))
            and attempted > 0
            and not isinstance(succeeded, bool)
            and isinstance(succeeded, (int, float))
            and math.isfinite(float(succeeded))
            and not isinstance(minimum, bool)
            and isinstance(minimum, (int, float))
            and math.isfinite(float(minimum))
            and 0 <= float(minimum) <= 1
            and float(succeeded) / float(attempted) < float(minimum)
        ):
            return (
                "sample_constraint_guardrail_failed"
                if constraint_rejected
                else "origin_coverage_guardrail_failed"
            )
    if coverage is False or sample_coverage is False:
        if constraint_rejected:
            return "sample_constraint_guardrail_failed"
        return "coverage_guardrail_failed"
    if coverage is not True and sample_coverage is not True:
        return "coverage_guardrail_missing"
    return None


def _safety_requires_rollback(reason: Any) -> bool:
    return reason in {
        "constraint_guardrail_failed",
        "sample_constraint_guardrail_failed",
    }


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
    revision = state.revision(context.candidate_revision_id)
    context_payload = {
        **context.to_dict(),
        "current_candidate_state": _local_edit_current_state(revision),
        "recent_edit_history": _recent_local_edit_history(
            state, candidate.candidate_id, batch.batch_index
        ),
        "decision_policy": {
            "prefer_smallest_effective_change": True,
            "reject_exact_current_value": True,
            "avoid_repeating_recent_rejected_operation": True,
            "require_batch_evidence_for_structural_change": True,
        },
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


def _local_edit_current_state(revision: CandidateRevision) -> dict[str, Any]:
    """Expose bounded current values so the editor can avoid no-op proposals."""

    genome = EcologyEvolutionPluginGenome.from_dict(dict(revision.genome))
    scientific = genome.scientific_program
    execution = genome.agent_program["candidate_execution_program"]
    roles = execution["role_profiles"]
    result = {
        "candidate_revision_id": revision.revision_id,
        "genome_digest": genome.genome_digest,
        "scientific_program": {
            "predictor_id": scientific["predictor_ref"]["id"],
            "feature_policy_id": scientific["feature_policy_ref"]["id"],
            "fit_policy_id": scientific["fit_policy_ref"]["id"],
            "uncertainty_policy_id": scientific["uncertainty_policy_ref"]["id"],
            "parameter_values": scientific["parameter_overrides"],
        },
        "candidate_execution_program": {
            "workflow_template_id": execution["workflow_template_ref"]["id"],
            "workflow_parameters": execution["workflow_overrides"],
            "roles": [
                {
                    "role": profile["role"],
                    "instruction_template_id": profile["instruction_template_ref"][
                        "id"
                    ],
                    "instruction_parameters": profile["instruction_parameters"],
                    "enabled_tool_ids": profile["enabled_tool_ids"],
                }
                for profile in roles
            ],
        },
    }
    return deep_thaw_json(result)


def _recent_local_edit_history(
    state: Any,
    candidate_id: str,
    before_batch_index: int,
) -> list[dict[str, Any]]:
    """Return only bounded proposal outcomes from earlier batches in this lane."""

    revision_ids = {
        int(item.batch_index): str(item.revision_id)
        for item in getattr(state, "formal_batches", ())
        if getattr(item, "candidate_id", None) == candidate_id
        and isinstance(getattr(item, "batch_index", None), int)
        and not isinstance(getattr(item, "batch_index", None), bool)
        and isinstance(getattr(item, "revision_id", None), str)
        and str(item.revision_id).strip()
    }
    outcomes = {
        int(item["batch_index"]): item
        for item in state.local_edit_outcomes
        if item.get("candidate_id") == candidate_id
        and isinstance(item.get("batch_index"), int)
        and not isinstance(item.get("batch_index"), bool)
    }
    rows: list[dict[str, Any]] = []
    for batch_index in range(max(0, before_batch_index)):
        proposal = state.local_edit_proposal_for(candidate_id, batch_index)
        outcome = outcomes.get(batch_index)
        if proposal is None or outcome is None:
            continue
        detail = proposal.get("proposal", proposal)
        rows.append(
            {
                "batch_index": batch_index,
                "candidate_revision_id": revision_ids.get(batch_index),
                "decision": detail.get("decision"),
                "operations": [
                    dict(operation)
                    for operation in detail.get("operations", ())
                    if isinstance(operation, Mapping)
                ][:5],
                "outcome": outcome.get("outcome"),
                "reason": outcome.get("reason"),
            }
        )
    return deep_thaw_json(rows[-8:])


def _local_edit_bundle_signature(operations: Any) -> str:
    """Return an order-independent signature for one atomic operation bundle."""

    if isinstance(operations, (str, bytes)) or not isinstance(
        operations, (list, tuple)
    ):
        raise TypeError("local edit operations must be an array")
    canonical_operations: list[str] = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise TypeError("local edit operation must be an object")
        canonical_operations.append(canonical_json(dict(operation)))
    return digest({"operations": sorted(canonical_operations)})


def _local_edit_policy_rejection_reason(
    state: Any,
    candidate_id: str,
    before_batch_index: int,
    candidate_revision_id: str,
    proposal: LocalEditProposal,
) -> str | None:
    """Reject a repeated failed bundle only while its parent revision is unchanged."""

    if proposal.decision.value != "mutate":
        return None
    proposed_signature = _local_edit_bundle_signature(proposal.operations)
    for row in _recent_local_edit_history(state, candidate_id, before_batch_index):
        if (
            row.get("outcome") != LocalEditOutcome.REJECTED.value
            or row.get("candidate_revision_id") != candidate_revision_id
            or row.get("decision") != "mutate"
        ):
            continue
        if _local_edit_bundle_signature(row.get("operations")) == proposed_signature:
            return "duplicate_recent_rejected_bundle"
    return None


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
    scalar_keys = (
        "objective_score",
        "objective_weight_coverage",
        "objective_aggregation_version",
        "objective_target_weights",
        "objective_horizons",
        "constraint_violations",
        "failed_cell_count",
        "raw_out_of_range_count",
        "prediction_clipped_count",
        "sample_execution_coverage",
        "sample_execution_coverage_pass",
        "scientific_pass",
        "baseline_score",
    )
    safe: dict[str, Any] = {
        key: detached[key] for key in scalar_keys if key in detached
    }
    sample = detached.get("sample_execution")
    if isinstance(sample, Mapping):
        safe["sample_execution"] = {
            key: sample[key]
            for key in (
                "attempted_origin_samples",
                "succeeded_origin_samples",
                "failed_origin_samples",
                "coverage",
                "coverage_pass",
                "minimum_coverage",
                "strict_agent_chain_pass",
                "strict_agent_chain_coverage",
                "complete_origin_agent_chains",
                "failed_examples",
                "scoring_fallback_examples",
                "failure_counts",
                "reason_code_counts",
                "recovered_by_failure_class",
                "critic_outcome_counts",
            )
            if key in sample
        }
    target_rows: list[dict[str, Any]] = []
    targets = detached.get("targets")
    if isinstance(targets, (list, tuple)):
        for target in targets[:64]:
            if not isinstance(target, Mapping):
                continue
            row = {
                key: target[key]
                for key in (
                    "target",
                    "horizon_hours",
                    "skill_score",
                    "coverage",
                    "sample_execution_coverage",
                    "sample_execution_coverage_pass",
                )
                if key in target
            }
            horizons = target.get("horizons")
            if isinstance(horizons, (list, tuple)):
                row["horizons"] = [
                    {
                        key: horizon[key]
                        for key in (
                            "hours",
                            "horizon_hours",
                            "skill",
                            "skill_score",
                            "coverage",
                        )
                        if key in horizon
                    }
                    for horizon in horizons[:32]
                    if isinstance(horizon, Mapping)
                ]
            target_rows.append(row)
    if target_rows:
        safe["targets"] = target_rows
    # Re-encode after projection to enforce JSON safety and a bounded native
    # context; potentially large diagnostic previews never cross this boundary.
    encoded = canonical_json(safe)
    if len(encoded.encode("utf-8")) > 16 * 1024:
        raise ValueError("aggregate local edit evidence exceeds 16KB")
    return deep_thaw_json(safe)


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
