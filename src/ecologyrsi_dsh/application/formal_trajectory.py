"""Restart-safe execution of one Top-2 finalist's adaptive formal lane."""

from __future__ import annotations

from ..evaluators.skill_program import skill_grammar

from ..evolution.agent_policy import rebind_agent_policy

import math
from dataclasses import replace
from statistics import median
from typing import Any, Mapping
from .ports import GenerationRuntime

from ..core.models import Candidate, canonical_json, digest
from ..core.model_execution_policy import LOCAL_EDIT_CONTEXT_POLICY
from ..core.search_policy import PAIRED_EXECUTION_QUALIFICATION, local_challenger_policy
from ..core.trajectory import (
    BatchEvaluation,
    CandidateRevision,
    EvaluationPhase,
    EvaluationScope,
    FormalBatchArm,
    FormalBatchComparison,
    FormalBatchComparisonDecision,
    LocalEditOutcome,
    RevisionAdvanceReason,
    RevisionStatus,
    TrajectoryStatus,
)
from ..evolution.genome import (
    EcologyEvolutionPluginGenome,
    deep_thaw_json,
    parameter_trust_region_neighborhood,
)
from ..evolution.champion_challenger import (
    LOCAL_MINIMUM_SCORE_DELTA,
    POSITIVE_DELTA_MINIMUM_SCORE_DELTA,
    assess_local_challenger,
    local_challenger_safety_reason,
)
from ..evolution.local_edits import (
    LocalEditContext,
    LocalEditProposal,
    LocalEditResult,
    _operation_target,
    apply_or_reject_local_edit_bundle,
)
from ..evolution.strategies import (
    _directive_authoring_contract,
    _genome_parameter_boundary,
    _registered_mutation_targets,
    _registered_scalar_axis_targets,
    mutation_axis_contract,
)
from ..evolution.parameter_activity import parameter_activity_contract
from ..evolution.schedule import (
    PREQUENTIAL_PERFORMANCE_POLICY,
    PAIRED_LOCAL_EVALUATION_MODE,
    OptimizationSchedule,
)
from ..evolution.workflow_ir import (
    resolve_candidate_agent_profile,
)
from ..knowledge.algorithms import AlgorithmSpec, compile_algorithm_spec
from ..knowledge.program_registry import current_program_registry
from ..evaluators.authored_directive import AUTHORED_DIRECTIVE_POLICY_ID
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
    metadata["agent_policy"] = rebind_agent_policy(metadata.get("agent_policy"),
        genome_digest=genome.genome_digest, profile=metadata["candidate_agent_profile"],
        parameters=genome.scientific_program["parameter_overrides"])
    metadata["tool_experience"] = metadata["agent_policy"]["experience"]["rows"]
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


def ensure_formal_trajectory(services: GenerationRuntime, run_id: str, candidate_id: str):
    state = services.director.state(run_id)
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
        services,
        "start_formal_trajectory",
        run_id,
        candidate_id,
        revision.revision_id,
        schedule.batch_count,
    )


def _scope_for_batch(
    state: Any,
    candidate: Candidate,
    revision_id: str,
    batch: Any,
    *,
    arm: FormalBatchArm | None = None,
):
    return EvaluationScope(
        run_id=state.run.run_id,
        generation=candidate.generation,
        candidate_id=candidate.candidate_id,
        candidate_revision_id=revision_id,
        phase=EvaluationPhase.FORMAL_BATCH,
        cohort_digest=batch.cohort_digest,
        origin_count=batch.origin_count,
        batch_index=batch.batch_index,
        formal_batch_arm=arm,
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
        "promotion_block_evidence",
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


def _evaluate_formal_batch_arm(
    services: GenerationRuntime,
    run_id: str,
    candidate_id: str,
    revision_id: str,
    formal_batch: Any,
    *,
    arm: FormalBatchArm | None,
) -> BatchEvaluation:
    state = services.director.state(run_id)
    candidate = state.candidate(candidate_id)
    adaptation = state.run_adaptation_cohort
    if adaptation is None:
        raise RuntimeError("formal evaluation requires frozen adaptation cohorts")
    planned_batch = adaptation.batches[formal_batch.batch_index]
    scope = _scope_for_batch(
        state,
        candidate,
        revision_id,
        formal_batch,
        arm=arm,
    )
    task = _phase_task_manifest(state.task_manifest, candidate.generation, "formal")
    _revision, proposal, compiled = _revision_evaluation_inputs(
        state,
        candidate,
        revision_id,
        task,
    )
    callbacks = _ScopedEvaluationCallbacks(
        services,
        run_id=run_id,
        generation=candidate.generation,
        proposal_id=proposal.proposal_id,
        candidate_id=candidate_id,
        scope=scope,
    )
    if isinstance(services.evaluators, EvaluatorRegistry):
        bundle = services.evaluators.evaluate_scientific(
            task,
            candidate,
            proposal,
            scope=scope,
            cohort=planned_batch.cohort,
            algorithm_spec=compiled,
            **callbacks.evaluation_kwargs(),
        )
    else:
        bundle = services.evaluators.evaluate_scientific(
            task,
            candidate,
            proposal,
            scope=scope,
            cohort=planned_batch.cohort,
            **callbacks.evaluation_kwargs(),
        )
    metrics = dict(bundle.evaluation.metrics)
    from ..evaluators.execution_validity import require_valid_execution
    require_valid_execution(metrics, scope.origin_count, phase="formal_batch")
    suffix = f":{arm.value}" if arm is not None else ""
    evaluation = BatchEvaluation(
        evaluation_id=(
            f"formal-evaluation:{candidate_id}:{formal_batch.batch_index}{suffix}"
        ),
        scope=scope,
        score=bundle.evaluation.score,
        passed=bundle.evaluation.passed,
        metrics=_durable_batch_metrics(metrics),
        evaluator_digest=digest(
            {"evaluator": bundle.evaluation.evaluator_digest}
        ),
    )
    return _director_mutation(
        services,
        "record_formal_batch_evaluation",
        run_id,
        evaluation,
        sample_results=callbacks.completion_payload(
            evaluation.evaluation_id,
            bundle.sample_results,
        ),
    )


def _execute_next_prequential_formal_batch(
    services: GenerationRuntime,
    run_id: str,
    candidate_id: str,
) -> bool:
    state = services.director.state(run_id)
    trajectory = ensure_formal_trajectory(services, run_id, candidate_id)
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
    formal_batch = _director_mutation(
        services,
        "start_formal_batch",
        run_id,
        candidate_id,
        active_revision_id,
        batch_index,
    )
    state = services.director.state(run_id)
    if state.batch_evaluation_for(candidate_id, batch_index) is not None:
        return False
    _evaluate_formal_batch_arm(
        services,
        run_id,
        candidate_id,
        active_revision_id,
        formal_batch,
        arm=None,
    )
    return True


def _local_challenger_policy(services: GenerationRuntime, run_id: str) -> dict[str, Any]:
    return local_challenger_policy(services.director.state(run_id).task_manifest.metadata)


def _record_initial_champion(
    services: GenerationRuntime,
    run_id: str,
    evaluation: BatchEvaluation,
) -> FormalBatchComparison:
    policy = _local_challenger_policy(
        services,
        run_id,
    )
    safety_passed = local_challenger_safety_reason(evaluation.metrics) is None
    assessment = assess_local_challenger(
        evaluation,
        evaluation,
        challenger_safety_gate_passed=safety_passed,
        **policy,
    )
    scope = evaluation.scope
    comparison = FormalBatchComparison(
        comparison_id=(
            f"comparison:{scope.candidate_id}:{scope.batch_index}"
        ),
        run_id=run_id,
        generation=scope.generation,
        candidate_id=scope.candidate_id,
        batch_index=int(scope.batch_index),
        cohort_digest=scope.cohort_digest,
        champion_before_revision_id=scope.candidate_revision_id,
        challenger_revision_id=scope.candidate_revision_id,
        champion_evaluation_id=evaluation.evaluation_id,
        challenger_evaluation_id=evaluation.evaluation_id,
        champion_evaluation_digest=evaluation.evaluation_digest,
        challenger_evaluation_digest=evaluation.evaluation_digest,
        champion_score=evaluation.score,
        challenger_score=evaluation.score,
        score_delta=0.0,
        comparison_contract_digest=assessment.comparison_contract_digest,
        safety_gate_passed=safety_passed,
        cell_regression_gate_passed=assessment.cell_regression_gate_passed,
        minimum_score_delta=policy["minimum_score_delta"],
        decision=FormalBatchComparisonDecision.INITIAL_CHAMPION,
        champion_after_revision_id=scope.candidate_revision_id,
        reason="initial_champion",
        paired_execution_qualification=(
            PAIRED_EXECUTION_QUALIFICATION if policy["require_paired_strict_chain"] else None
        ),
    )
    return _director_mutation(
        services,
        "record_formal_batch_comparison",
        run_id,
        comparison,
    )


def _record_challenger_comparison(
    services: GenerationRuntime,
    run_id: str,
    champion: BatchEvaluation,
    challenger: BatchEvaluation,
) -> FormalBatchComparison:
    policy = _local_challenger_policy(
        services,
        run_id,
    )
    safety_passed = local_challenger_safety_reason(challenger.metrics) is None
    assessment = assess_local_challenger(
        champion,
        challenger,
        challenger_safety_gate_passed=safety_passed,
        **policy,
    )
    scope = challenger.scope
    comparison = FormalBatchComparison(
        comparison_id=(
            f"comparison:{scope.candidate_id}:{scope.batch_index}"
        ),
        run_id=run_id,
        generation=scope.generation,
        candidate_id=scope.candidate_id,
        batch_index=int(scope.batch_index),
        cohort_digest=scope.cohort_digest,
        champion_before_revision_id=(
            champion.scope.candidate_revision_id
        ),
        challenger_revision_id=challenger.scope.candidate_revision_id,
        champion_evaluation_id=champion.evaluation_id,
        challenger_evaluation_id=challenger.evaluation_id,
        champion_evaluation_digest=champion.evaluation_digest,
        challenger_evaluation_digest=challenger.evaluation_digest,
        champion_score=champion.score,
        challenger_score=challenger.score,
        score_delta=assessment.score_delta,
        comparison_contract_digest=assessment.comparison_contract_digest,
        safety_gate_passed=assessment.safety_gate_passed,
        cell_regression_gate_passed=(
            assessment.cell_regression_gate_passed
        ),
        minimum_score_delta=policy["minimum_score_delta"],
        decision=assessment.decision,
        champion_after_revision_id=(
            assessment.champion_after_revision_id
        ),
        reason=assessment.reason,
        paired_execution_qualification=(
            PAIRED_EXECUTION_QUALIFICATION if policy["require_paired_strict_chain"] else None
        ),
    )
    return _director_mutation(
        services,
        "record_formal_batch_comparison",
        run_id,
        comparison,
    )


def _execute_next_paired_formal_batch(
    services: GenerationRuntime,
    run_id: str,
    candidate_id: str,
) -> bool:
    trajectory = ensure_formal_trajectory(services, run_id, candidate_id)
    if trajectory.status is TrajectoryStatus.COMPLETED:
        return False
    state = services.director.state(run_id)
    candidate = state.candidate(candidate_id)
    if (
        state.run_adaptation_cohort is None
        or state.generation_cohort_for(candidate.generation) is None
    ):
        raise RuntimeError(
            "formal batch requires frozen adaptation and generation cohorts"
        )
    for batch_index in range(trajectory.batch_count):
        state = services.director.state(run_id)
        trajectory = state.trajectory_for(candidate_id)
        comparison = state.batch_comparison_for(candidate_id, batch_index)
        batch = state.formal_batch_for(candidate_id, batch_index)
        if comparison is not None:
            if batch_index == trajectory.batch_count - 1:
                if trajectory.status is not TrajectoryStatus.COMPLETED:
                    _director_mutation(
                        services,
                        "complete_formal_trajectory",
                        run_id,
                        candidate_id,
                        comparison.champion_after_revision_id,
                    )
                    return True
                return False
            if state.revision_activation_for(candidate_id, batch_index) is None:
                return False
            continue
        if batch is None:
            if batch_index == 0:
                challenger_revision_id = trajectory.initial_revision_id
            else:
                activation = state.revision_activation_for(
                    candidate_id,
                    batch_index - 1,
                )
                if activation is None:
                    return False
                challenger_revision_id = activation.to_revision_id
            batch = _director_mutation(
                services,
                "start_formal_batch",
                run_id,
                candidate_id,
                challenger_revision_id,
                batch_index,
            )
            state = services.director.state(run_id)
        if batch_index == 0:
            warmup = state.batch_evaluation_for(
                candidate_id,
                batch_index,
                FormalBatchArm.CHAMPION,
            )
            if warmup is None:
                _evaluate_formal_batch_arm(
                    services,
                    run_id,
                    candidate_id,
                    trajectory.initial_revision_id,
                    batch,
                    arm=FormalBatchArm.CHAMPION,
                )
                return True
            _record_initial_champion(services, run_id, warmup)
            if batch_index == trajectory.batch_count - 1:
                _director_mutation(
                    services,
                    "complete_formal_trajectory",
                    run_id,
                    candidate_id,
                    trajectory.initial_revision_id,
                )
            return True

        prior_comparison = state.batch_comparison_for(
            candidate_id,
            batch_index - 1,
        )
        if prior_comparison is None:
            raise RuntimeError("paired batch requires prior champion comparison")
        champion_revision_id = prior_comparison.champion_after_revision_id
        challenger_revision_id = batch.revision_id
        champion_evaluation = state.batch_evaluation_for(
            candidate_id,
            batch_index,
            FormalBatchArm.CHAMPION,
        )
        if champion_evaluation is None:
            _evaluate_formal_batch_arm(
                services,
                run_id,
                candidate_id,
                champion_revision_id,
                batch,
                arm=FormalBatchArm.CHAMPION,
            )
            return True
        if challenger_revision_id == champion_revision_id:
            challenger_evaluation = champion_evaluation
        else:
            challenger_evaluation = state.batch_evaluation_for(
                candidate_id,
                batch_index,
                FormalBatchArm.CHALLENGER,
            )
            if challenger_evaluation is None:
                _evaluate_formal_batch_arm(
                    services,
                    run_id,
                    candidate_id,
                    challenger_revision_id,
                    batch,
                    arm=FormalBatchArm.CHALLENGER,
                )
                return True
        comparison = _record_challenger_comparison(
            services,
            run_id,
            champion_evaluation,
            challenger_evaluation,
        )
        if batch_index == trajectory.batch_count - 1:
            _director_mutation(
                services,
                "complete_formal_trajectory",
                run_id,
                candidate_id,
                comparison.champion_after_revision_id,
            )
        return True
    return False


def execute_next_formal_batch(
    services: GenerationRuntime,
    run_id: str,
    candidate_id: str,
) -> bool:
    state = services.director.state(run_id)
    schedule = OptimizationSchedule.from_dict(
        state.task_manifest.metadata["optimization_schedule"]
    )
    if schedule.local_evaluation_mode == PAIRED_LOCAL_EVALUATION_MODE:
        return _execute_next_paired_formal_batch(
            services,
            run_id,
            candidate_id,
        )
    return _execute_next_prequential_formal_batch(
        services,
        run_id,
        candidate_id,
    )


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
        # Every axis the batch-local editor may spend its one operation on is
        # named here, including the two the proposer catalog above already
        # supplies. Restating them off the shared helper costs nothing and keeps
        # this one site a complete, readable answer to "what can a local edit
        # touch" -- the alternative is a reader having to subtract
        # `registered_predictor` from another function's return value and then
        # guess which of the remaining axes are open.
        "workflow_template": tuple(
            item
            for item in registry.program_ids("workflow_templates")
            if item != workflow_id
            and "sample-planner"
            in registry.program("workflow_templates", item)["graph"]["allowed_roles"]
        ),
        "workflow_parameter": _registered_scalar_axis_targets(
            genome, "workflow_parameter"
        ),
        # Current feature/fit policies have no alternative implementation;
        # the registered UQ program is not connected to native prediction.
        "feature_policy": (),
        "fit_policy": (),
        "uncertainty_policy": (),
        # Closed on purpose, not for lack of an implementation. The operator can
        # only narrow an inherited tool policy, and `sample-planner-tools@1`
        # holds exactly one tool -- so the single legal edit is to switch the
        # prediction tool off, which removes the only path by which the
        # scientific parameters reach a prediction at all. Reopen this when the
        # planner has a second tool and narrowing can express a real strategy.
        "instruction_tool_policy": (),
        "instruction_parameter": _registered_scalar_axis_targets(
            genome, "instruction_parameter"
        ),
        # Open to the batch-local editor too, and it is the one axis that never
        # exhausts: `instruction_profile` runs out of unused templates after a
        # few edits, while a directive can always be rewritten. The single
        # target is the grammar that bounds the text, so the allowed-target
        # check asks whether this candidate may author at all, not whether a
        # particular sentence is legal.
        "instruction_directive": (AUTHORED_DIRECTIVE_POLICY_ID,),
    }
    _boundary, parameter_schemas = _genome_parameter_boundary(state.task_manifest, genome)
    from ..evolution.diversity import enabled as diversity_enabled, candidate_family, FAMILY_AXES
    if diversity_enabled(state.task_manifest.metadata):
        family = candidate_family(state.proposal(candidate.proposal_id))
        if family in FAMILY_AXES:
            targets = {axis: values if axis in FAMILY_AXES[family] else () for axis, values in targets.items()}
    cells = tuple(
        f"{target}@{horizon}h"
        for target in state.task_manifest.metadata["fitness_profile"]["expected_targets"]
        for horizon in state.task_manifest.metadata["fitness_profile"]["expected_horizons"]
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


def _execute_next_prequential_local_edit(
    services: GenerationRuntime,
    run_id: str,
    candidate_id: str,
) -> bool:
    state = services.director.state(run_id)
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
            services,
            "advance_trajectory_revision",
            run_id,
            candidate_id,
            pending_batch.batch_index,
            outcome["active_revision_id"],
            reason,
        )
        if pending_batch.batch_index == trajectory.batch_count - 1:
            _director_mutation(
                services,
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
    pending_metrics = state.batch_evaluation_for(
        candidate_id, pending.batch_index
    ).metrics
    recorded = state.local_edit_proposal_for(candidate_id, pending.batch_index)
    if recorded is None:
        safety_reason = _prequential_safety_reason(
            pending_metrics
        ) or _prequential_regression_reason(
            state,
            candidate_id,
            revision,
            pending.batch_index,
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
            or pending.batch_index == trajectory.batch_count - 1
            # A child authored after the final batch is never evaluated in this
            # trajectory, yet completion promotes the last active revision into
            # the holdout arm.  Keep the measured revision instead, exactly as
            # the paired mode already requires of its final batch.
            else _local_edit_proposal(services, state, candidate, pending, context)
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
        or _inactive_axis_rejection_reason(pending_metrics, proposal)
    )
    safety_rollback = _safety_requires_rollback(safety_reason)
    # Commit the complete, schema-validated proposal before deriving a child
    # revision.  This closes the old crash window where replay only had the
    # child genome and a lossy operation list, and might ask for a new edit.
    if recorded is None:
        _director_mutation(
            services,
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
    validated = _check_behavior_revisit(state, candidate_id, proposal, validated, batch_index=pending.batch_index)
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
        _director_mutation(services, "create_candidate_revision", run_id, child_revision)
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
        services,
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
                {
                    "reason": (
                        safety_reason
                        or policy_rejection_reason
                        or (
                            validated.rejection_reason
                            if validated is not None
                            else None
                        )
                    )
                }
                if safety_reason is not None
                or policy_rejection_reason is not None
                or (
                    validated is not None
                    and validated.rejection_reason is not None
                )
                else {}
            ),
        },
    )
    _director_mutation(
        services,
        "advance_trajectory_revision",
        run_id,
        candidate_id,
        pending.batch_index,
        active_revision_id,
        advance_reason,
    )
    if pending.batch_index == trajectory.batch_count - 1:
        _director_mutation(
            services,
            "complete_formal_trajectory",
            run_id,
            candidate_id,
            active_revision_id,
        )
    return True


def _execute_next_paired_local_edit(
    services: GenerationRuntime,
    run_id: str,
    candidate_id: str,
) -> bool:
    state = services.director.state(run_id)
    candidate = state.candidate(candidate_id)
    trajectory = state.trajectory_for(candidate_id)
    if trajectory is None:
        raise RuntimeError("local edit requires a formal trajectory")
    if trajectory.status is TrajectoryStatus.COMPLETED:
        return False
    recovery = next(
        (
            (batch, outcome)
            for batch in state.formal_batches
            if batch.candidate_id == candidate_id
            and batch.batch_index < trajectory.batch_count - 1
            and state.batch_comparison_for(candidate_id, batch.batch_index)
            is not None
            and state.revision_activation_for(candidate_id, batch.batch_index)
            is None
            for outcome in state.local_edit_outcomes
            if outcome.get("candidate_id") == candidate_id
            and outcome.get("batch_index") == batch.batch_index
        ),
        None,
    )
    if recovery is not None:
        batch, outcome = recovery
        reason = {
            LocalEditOutcome.KEPT.value: RevisionAdvanceReason.KEPT,
            LocalEditOutcome.APPLIED.value: (
                RevisionAdvanceReason.LOCAL_EDIT_APPLIED
            ),
            LocalEditOutcome.REJECTED.value: (
                RevisionAdvanceReason.LOCAL_EDIT_REJECTED
            ),
        }[outcome["outcome"]]
        _director_mutation(
            services,
            "advance_trajectory_revision",
            run_id,
            candidate_id,
            batch.batch_index,
            outcome["active_revision_id"],
            reason,
        )
        return True
    pending = next(
        (
            batch
            for batch in state.formal_batches
            if batch.candidate_id == candidate_id
            and batch.batch_index < trajectory.batch_count - 1
            and state.batch_comparison_for(candidate_id, batch.batch_index)
            is not None
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
    comparison = state.batch_comparison_for(
        candidate_id,
        pending.batch_index,
    )
    revision = state.revision(comparison.champion_after_revision_id)
    context = _local_edit_context(state, candidate, revision, pending)
    recorded = state.local_edit_proposal_for(candidate_id, pending.batch_index)
    evaluation = next(
        item for item in state.formal_batch_evaluations
        if item.evaluation_id == (
            comparison.challenger_evaluation_id
            if comparison.champion_after_revision_id == comparison.challenger_revision_id
            else comparison.champion_evaluation_id
        )
    )
    execution_reason = _local_edit_execution_reason(evaluation.metrics)
    proposal = (
        LocalEditProposal(decision="keep", operations=(), evidence_refs=context.allowed_evidence_refs[:1],
                          expected_effect_cells=(), risk_cells=())
        if recorded is None and execution_reason is not None
        else _local_edit_proposal(services, state, candidate, pending, context)
        if recorded is None
        else LocalEditProposal.from_dict(recorded["proposal"])
    )
    policy_rejection_reason = _local_edit_policy_rejection_reason(
        state,
        candidate_id,
        pending.batch_index,
        revision.revision_id,
        proposal,
    ) or _inactive_axis_rejection_reason(evaluation.metrics, proposal)
    if recorded is None:
        _director_mutation(
            services,
            "record_local_edit_proposal",
            run_id,
            {
                "proposal_id": (
                    f"local-edit:{candidate_id}:{pending.batch_index}"
                ),
                "candidate_id": candidate_id,
                "batch_index": pending.batch_index,
                "evidence_scope_digest": context.evidence_scope_digest,
                "proposal": proposal.to_dict(),
            },
        )
    if execution_reason is not None and proposal.decision.value == "mutate":
        policy_rejection_reason = execution_reason
    if policy_rejection_reason is not None:
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
    validated = _check_behavior_revisit(state, candidate_id, proposal, validated, batch_index=pending.batch_index)
    active_revision_id = revision.revision_id
    advance_reason = RevisionAdvanceReason.KEPT
    if validated.child is not None:
        child = validated.child
        child_revision = CandidateRevision(
            revision_id=(
                f"revision:{candidate_id}:batch:{pending.batch_index + 1}"
            ),
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
        _director_mutation(
            services,
            "create_candidate_revision",
            run_id,
            child_revision,
        )
        active_revision_id = child_revision.revision_id
        advance_reason = RevisionAdvanceReason.LOCAL_EDIT_APPLIED
    elif validated.outcome is LocalEditOutcome.REJECTED:
        advance_reason = RevisionAdvanceReason.LOCAL_EDIT_REJECTED
    _director_mutation(
        services,
        "decide_local_edit",
        run_id,
        {
            "proposal_id": f"local-edit:{candidate_id}:{pending.batch_index}",
            "candidate_id": candidate_id,
            "batch_index": pending.batch_index,
            "outcome": validated.outcome.value,
            "active_revision_id": active_revision_id,
            **(
                {
                    "reason": (
                        policy_rejection_reason or validated.rejection_reason
                    )
                }
                if policy_rejection_reason is not None
                or validated.rejection_reason is not None
                else {}
            ),
            **({"reason": execution_reason} if execution_reason is not None else {}),
        },
    )
    _director_mutation(
        services,
        "advance_trajectory_revision",
        run_id,
        candidate_id,
        pending.batch_index,
        active_revision_id,
        advance_reason,
    )
    return True


def execute_next_local_edit(
    services: GenerationRuntime,
    run_id: str,
    candidate_id: str,
) -> bool:
    state = services.director.state(run_id)
    metadata = getattr(getattr(state, "task_manifest", None), "metadata", {})
    raw_schedule = metadata.get("optimization_schedule")
    if (
        isinstance(raw_schedule, Mapping)
        and OptimizationSchedule.from_dict(raw_schedule).local_evaluation_mode
        == PAIRED_LOCAL_EVALUATION_MODE
    ):
        return _execute_next_paired_local_edit(
            services,
            run_id,
            candidate_id,
        )
    return _execute_next_prequential_local_edit(
        services,
        run_id,
        candidate_id,
    )


_REGRESSION_GUARDRAIL_REASON = "score_regression_guardrail_failed"


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


def _cohort_scoped_editor(state: Any) -> bool:
    metadata = getattr(getattr(state, "task_manifest", None), "metadata", {})
    return metadata.get("local_edit_context_policy") == LOCAL_EDIT_CONTEXT_POLICY


def _recent_behavior_states(state: Any, candidate_id: str) -> list[dict[str, Any]]:
    """Expose executable prior states, without treating their windows as trials."""
    revisions = [r for r in getattr(state, "candidate_revisions", ())
                 if r.candidate_id == candidate_id]
    return [{"revision_id": r.revision_id, "behavior_digest": r.behavior_digest,
             "candidate_state": _local_edit_current_state(r)} for r in revisions[-8:]]


def _check_behavior_revisit(state: Any, candidate_id: str, proposal: LocalEditProposal,
                            result: LocalEditResult | None, *, batch_index: int) -> LocalEditResult | None:
    if not _cohort_scoped_editor(state) or result is None or result.child is None:
        return result
    # Replay may already contain this batch's child after a crash between the
    # child and decision events. It is not a previous behavior revisit.
    repeated = any(r.candidate_id == candidate_id
                   and (r.source_batch_index is None or r.source_batch_index < batch_index)
                   and r.behavior_digest == result.child.behavior_digest
                   for r in getattr(state, "candidate_revisions", ()))
    if repeated and proposal.revisit is None:
        return LocalEditResult(outcome=LocalEditOutcome.REJECTED, operations=result.operations,
                               child=None, proposal_digest=result.proposal_digest,
                               rejection_reason="repeated_behavior_requires_revisit_reason")
    return result


def _batch_score(state: Any, candidate_id: str, batch_index: int) -> float | None:
    """Return one finite prequential batch score, or None when unavailable."""

    accessor = getattr(state, "batch_evaluation_for", None)
    if not callable(accessor):
        return None
    score = getattr(accessor(candidate_id, batch_index), "score", None)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return float(score) if math.isfinite(float(score)) else None


def _batch_metrics(state: Any, candidate_id: str, batch_index: int) -> Any:
    """Return one batch's recorded metrics mapping, or None."""

    accessor = getattr(state, "batch_evaluation_for", None)
    if not callable(accessor):
        return None
    return getattr(accessor(candidate_id, batch_index), "metrics", None)


def _batch_score_coverage_gap(metrics: Any) -> str | None:
    """Name the coverage damage that makes one batch score incomparable.

    A (target, horizon) group with no scored cell contributes the frozen
    ``OBJECTIVE_MISSING_PENALTY`` instead of a skill, which moves the batch
    score by roughly a ninth of that penalty -- an order of magnitude more than
    any bounded parameter edit can.  Such a batch measures availability, not
    the edit, so it must neither trigger the regression rollback, define the
    band that judges other batches, nor stand as the best block observed.  The
    penalty itself is part of the frozen objective aggregation version and is
    deliberately left unchanged: rescaling it would invalidate the archived
    scores that replay reproduces.
    """

    if not isinstance(metrics, Mapping):
        return "batch_metrics_unavailable"
    weight_coverage = metrics.get("objective_weight_coverage")
    if (
        not isinstance(weight_coverage, bool)
        and isinstance(weight_coverage, (int, float))
        and math.isfinite(float(weight_coverage))
        and float(weight_coverage) < 1.0
    ):
        return "objective_weight_coverage_incomplete"
    failed_cells = metrics.get("failed_cell_count")
    if (
        not isinstance(failed_cells, bool)
        and isinstance(failed_cells, (int, float))
        and math.isfinite(float(failed_cells))
        and float(failed_cells) > 0
    ):
        return "failed_scoring_cells"
    if metrics.get("sample_execution_coverage_pass") is False:
        return "sample_execution_coverage_failed"
    sample = metrics.get("sample_execution")
    if isinstance(sample, Mapping) and sample.get("coverage_pass") is False:
        return "sample_execution_coverage_failed"
    return None


def _trajectory_batch_scores(
    state: Any,
    candidate_id: str,
    through_batch_index: int,
    *,
    comparable_only: bool = False,
) -> dict[int, float]:
    """Return the finite scores this lane has actually measured so far."""

    scores: dict[int, float] = {}
    for index in range(int(through_batch_index) + 1):
        score = _batch_score(state, candidate_id, index)
        if score is None:
            continue
        if comparable_only and _batch_score_coverage_gap(
            _batch_metrics(state, candidate_id, index)
        ):
            continue
        scores[index] = score
    return scores


def _batch_revision_ids(state: Any, candidate_id: str) -> dict[int, str]:
    """Map each formal batch index to the revision that batch actually scored."""

    return {
        int(item.batch_index): str(item.revision_id)
        for item in getattr(state, "formal_batches", ())
        if getattr(item, "candidate_id", None) == candidate_id
        and isinstance(getattr(item, "batch_index", None), int)
        and not isinstance(getattr(item, "batch_index", None), bool)
        and isinstance(getattr(item, "revision_id", None), str)
        and str(item.revision_id).strip()
    }


def _trajectory_score_history(
    state: Any, candidate_id: str, through_batch_index: int
) -> dict[str, Any]:
    """Expose this lane's realized score series and its best measured block."""

    scores = _trajectory_batch_scores(state, candidate_id, through_batch_index)
    revision_ids = _batch_revision_ids(state, candidate_id)
    series = []
    for index in sorted(scores):
        gap = _batch_score_coverage_gap(
            _batch_metrics(state, candidate_id, index)
        )
        row = {
            "batch_index": index,
            "candidate_revision_id": revision_ids.get(index),
            "score": scores[index],
            "comparable": gap is None and not _cohort_scoped_editor(state),
            **({"coverage_complete": gap is None,
                "cohort_digest": (_batch_metrics(state, candidate_id, index) or {}).get("feedback_update_cohort_digest"),
                "comparison_scope": "different_batch_cohort_diagnostic_only"}
               if _cohort_scoped_editor(state) else {}),
        }
        if gap is not None:
            row["incomparable_reason"] = gap
        series.append(row)
    comparable = [row for row in series if row["comparable"]]
    return {
        "score_history": series,
        # A coverage-damaged block carries a missing-group penalty, so it is
        # never the bar a later batch is asked to beat.
        "best_observed": max(comparable, key=lambda row: row["score"], default=None),
    }


def _regression_band_floor(state: Any) -> float:
    """Never react below the run's own frozen selection tolerance."""

    metadata = getattr(getattr(state, "task_manifest", None), "metadata", None)
    if not isinstance(metadata, Mapping):
        return LOCAL_MINIMUM_SCORE_DELTA
    try:
        floor = float(local_challenger_policy(metadata)["minimum_score_delta"])
    except (KeyError, TypeError, ValueError):
        return LOCAL_MINIMUM_SCORE_DELTA
    return floor if math.isfinite(floor) and floor > 0 else LOCAL_MINIMUM_SCORE_DELTA


def _prequential_regression_reason(
    state: Any,
    candidate_id: str,
    revision: Any,
    batch_index: int,
) -> str | None:
    """Undo the most recent local edit when its own block score clearly fell.

    Prequential batches score different, time-forward blocks, so one drop can
    be block difficulty rather than the edit.  The band therefore comes from
    this lane's own realized volatility -- the median absolute consecutive
    delta over the *earlier* batches, so the drop under test cannot widen the
    band that judges it -- and never falls below the frozen selection
    tolerance.  Only the edit authored on the previous batch is undone, and
    only onto its parent, which is the exact revision a prequential rollback
    is allowed to reactivate.

    Batches whose objective weight coverage is incomplete are excluded on both
    sides: their missing-group penalty is not a measurement of the edit, so it
    may neither trigger a rollback nor widen the band.
    """

    metadata = getattr(getattr(state, "task_manifest", None), "metadata", {})
    if metadata.get("prequential_performance_policy") == PREQUENTIAL_PERFORMANCE_POLICY:
        # The parent was scored on a different window. Keep the new revision
        # provisional until the paired epoch comparison; physical/execution
        # guardrails are checked separately and still cause immediate rollback.
        return None
    parent_revision_id = getattr(revision, "parent_revision_id", None)
    if not isinstance(parent_revision_id, str) or not parent_revision_id.strip():
        # Without a parent there is no revision to return to, so a regression
        # here is not actionable and the lane keeps exploring.
        return None
    source_batch_index = getattr(revision, "source_batch_index", None)
    if isinstance(source_batch_index, bool) or not isinstance(source_batch_index, int):
        return None
    index = int(batch_index)
    if source_batch_index != index - 1:
        # The active revision predates the previous batch, so this drop is not
        # attributable to a single edit and there is nothing to undo.
        return None
    scores = _trajectory_batch_scores(
        state, candidate_id, index, comparable_only=True
    )
    current = scores.get(index)
    parent = scores.get(index - 1)
    if current is None or parent is None:
        return None
    deltas = [
        abs(scores[position] - scores[position - 1])
        for position in sorted(scores)
        if position < index and position - 1 in scores
    ]
    if not deltas:
        # One observed block cannot separate an edit effect from block noise.
        return None
    band = max(_regression_band_floor(state), median(deltas))
    return _REGRESSION_GUARDRAIL_REASON if current < parent - band else None


def _safety_requires_rollback(reason: Any) -> bool:
    return reason in {
        "constraint_guardrail_failed",
        "sample_constraint_guardrail_failed",
        _REGRESSION_GUARDRAIL_REASON,
    }


def _local_edit_proposal(
    services: GenerationRuntime,
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
    runtime = getattr(services, "dsh_native_runtime", None)
    admission = getattr(services, "dsh_tools", None)
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
    tool_usage = _agent_tool_usage(evaluation.metrics)
    inert_axes = _inert_mutation_axes(tool_usage)
    advertised = {
        axis: list(values)
        for axis, values in context.allowed_mutation_targets.items()
        if axis not in inert_axes
    }
    decision_policy = {
        "prefer_smallest_effective_change": True,
        "reject_exact_current_value": True,
        "avoid_repeating_recent_rejected_operation": True,
        "require_batch_evidence_for_structural_change": True,
        "avoid_repeating_an_edit_whose_next_batch_score_regressed": True,
        "host_rolls_back_a_clear_score_regression": True,
    }
    if _cohort_scoped_editor(state):
        decision_policy.update({
            "avoid_repeating_an_edit_whose_next_batch_score_regressed": False,
            "host_rolls_back_a_clear_score_regression": False,
            "cross_batch_scores_measure_edit_effect": False,
            "performance_decision": "use_same_cohort_paired_evidence_only",
            "physical_and_execution_guards_remain_active": True,
            "repeated_behavior_requires_revisit_reason": True,
            "revisit_reasons": ["safety_recovery", "new_batch_evidence", "paired_recheck"],
        })
    if inert_axes:
        decision_policy["inert_mutation_axes"] = list(inert_axes)
        decision_policy["inert_mutation_axes_are_rejected"] = True
        decision_policy["inert_mutation_axis_explanation"] = (
            "The sample Agent submitted every prediction of this batch without "
            "calling a prediction tool, so these axes only changed a default "
            "nothing read. Spend the operation on the Agent's instructions "
            "instead, so a later batch can measure the scientific parameters."
        )
    context_payload = {
        **context.to_dict(),
        "allowed_mutation_targets": advertised,
        # An axis name alone was not usable: the editor was shown six axes and
        # the operation names for three of them (the Skill documents only those),
        # so the other three could only be reached by guessing an operation id.
        # Projected onto the axes that have at least one legal target, so an
        # operation id never appears for an axis the editor cannot spend it on --
        # the axis itself still appears above, with an empty target list, which is
        # how a closed axis stays visible instead of silently missing.
        **mutation_axis_contract(
            axis for axis, values in advertised.items() if values
        ),
        "current_candidate_state": _local_edit_current_state(revision),
        "legal_parameter_neighborhoods": (
            {}
            if "scientific_parameter" in inert_axes
            else _legal_parameter_neighborhoods(revision, context)
        ),
        "legal_workflow_parameter_bounds": _legal_workflow_parameter_bounds(
            revision, context
        ),
        "legal_instruction_parameter_bounds": _legal_instruction_parameter_bounds(
            revision, context
        ),
        # Only when the axis is actually open for this candidate. A grammar with
        # no reachable operation is the kind of advertising this axis exists to
        # replace: the editor would learn how to write a directive it cannot
        # submit.
        **(
            {"legal_directive_grammar": _legal_directive_grammar(revision)}
            if advertised.get("instruction_directive")
            else {}
        ),
        **({"legal_skill_grammar": skill_grammar()} if advertised.get("skill_program") else {}),
        "recent_edit_history": _recent_local_edit_history(
            state, candidate.candidate_id, batch.batch_index
        ),
        **_trajectory_score_history(state, candidate.candidate_id, batch.batch_index),
        "decision_policy": decision_policy,
        **({"recent_behavior_states": _recent_behavior_states(state, candidate.candidate_id)}
           if _cohort_scoped_editor(state) else {}),
        "batch_evidence": {
            "score": evaluation.score,
            "passed": evaluation.passed,
            "metrics": metrics,
            "scope_digest": evaluation.scope.scope_key,
            **({"agent_tool_usage": tool_usage} if tool_usage is not None else {}),
        },
    }
    comparison = state.batch_comparison_for(
        candidate.candidate_id,
        batch.batch_index,
    )
    if comparison is not None:
        evaluations = tuple(
            item
            for item in state.formal_batch_evaluations
            if item.scope.candidate_id == candidate.candidate_id
            and item.scope.batch_index == batch.batch_index
        )
        champion_evaluation = next(
            (
                item
                for item in evaluations
                if item.evaluation_id == comparison.champion_evaluation_id
            ),
            None,
        )
        challenger_evaluation = next(
            (
                item
                for item in evaluations
                if item.evaluation_id == comparison.challenger_evaluation_id
            ),
            None,
        )
        if champion_evaluation is None or challenger_evaluation is None:
            raise RuntimeError("local editor comparison evidence is incomplete")
        context_payload["paired_batch_evidence"] = {
            "champion_before_revision_id": (
                comparison.champion_before_revision_id
            ),
            "challenger_revision_id": comparison.challenger_revision_id,
            "champion_score": comparison.champion_score,
            "challenger_score": comparison.challenger_score,
            "score_delta": comparison.score_delta,
            "minimum_score_delta": comparison.minimum_score_delta,
            "decision": comparison.decision.value,
            "reason": comparison.reason,
            "champion_after_revision_id": (
                comparison.champion_after_revision_id
            ),
            "safety_gate_passed": comparison.safety_gate_passed,
            "cell_regression_gate_passed": (
                comparison.cell_regression_gate_passed
            ),
            "champion_metrics": _local_edit_evidence_metrics(
                champion_evaluation.metrics
            ),
            "challenger_metrics": _local_edit_evidence_metrics(
                challenger_evaluation.metrics
            ),
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
            ledger_expected_revision=services.ledger.latest_seq(),
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
                    **({"skill_program": profile["skill_program"]} if "skill_program" in profile else {}),
                    # Present only when this candidate authored one, which is
                    # exactly the distinction between writing a first directive
                    # and revising an existing one. An editor that cannot see
                    # the current clauses can only propose a rewrite blind, and
                    # a rewrite identical to the parent is refused.
                    **(
                        {"authored_directive": profile["authored_directive"]}
                        if "authored_directive" in profile
                        else {}
                    ),
                }
                for profile in roles
            ],
        },
    }
    activity = parameter_activity_contract(genome)
    if activity:
        result["scientific_parameter_activity"] = activity
    return deep_thaw_json(result)


def _legal_parameter_neighborhoods(
    revision: CandidateRevision,
    context: LocalEditContext,
) -> dict[str, dict[str, Any]]:
    """Expose exact legal one-step ranges without altering authored values."""

    genome = EcologyEvolutionPluginGenome.from_dict(dict(revision.genome))
    values = genome.scientific_program["parameter_overrides"]
    result: dict[str, dict[str, Any]] = {}
    allowed = set(
        context.allowed_mutation_targets.get("scientific_parameter", ())
    )
    for name, contract in sorted(context.parameter_schemas.items()):
        if name not in allowed or name not in values:
            continue
        result[name] = parameter_trust_region_neighborhood(
            name=name,
            previous=values[name],
            contract=contract,
        )
    return deep_thaw_json(result)


def _legal_workflow_parameter_bounds(
    revision: CandidateRevision,
    context: LocalEditContext,
) -> dict[str, dict[str, Any]]:
    """Publish the registered bounds for the workflow knobs on offer.

    Scientific parameters travel with a trust region; workflow parameters have
    only the registered contract, and the editor was previously shown the target
    name with no bounds at all -- so every attempt was a guess, and a guess that
    lands outside the contract costs the batch its single operation.

    ``current_value`` is the effective one: the candidate's override if it has
    any, otherwise the template default, which is what the run would use.
    """

    genome = EcologyEvolutionPluginGenome.from_dict(dict(revision.genome))
    execution = genome.agent_program["candidate_execution_program"]
    registry = current_program_registry()
    template = registry.program(
        "workflow_templates", str(execution["workflow_template_ref"]["id"])
    )
    contracts = template["parameters"]
    overrides = execution["workflow_overrides"]
    result: dict[str, dict[str, Any]] = {}
    for name in context.allowed_mutation_targets.get("workflow_parameter", ()):
        contract = contracts.get(str(name))
        if not isinstance(contract, Mapping):
            continue
        result[str(name)] = {
            "current_value": overrides.get(str(name), contract["default"]),
            **dict(contract),
        }
    return deep_thaw_json(result)


def _legal_instruction_parameter_bounds(
    revision: CandidateRevision,
    context: LocalEditContext,
) -> dict[str, dict[str, Any]]:
    """Publish the registered bounds per role for the instruction knobs on offer.

    Keyed by role, because `set_instruction_parameter` names a role as well as a
    parameter, and two roles can declare the same parameter with different
    bounds. `current_value` is the *effective* value: the candidate's override if
    it has one, otherwise the template default, which is what the run would use.
    Without this the editor knew the parameter's name and nothing else -- not its
    range, not what it is set to now -- so it could only propose a value and hope.
    """

    genome = EcologyEvolutionPluginGenome.from_dict(dict(revision.genome))
    registry = current_program_registry()
    advertised = {
        str(name)
        for name in context.allowed_mutation_targets.get("instruction_parameter", ())
    }
    result: dict[str, dict[str, Any]] = {}
    for profile in genome.agent_program["candidate_execution_program"][
        "role_profiles"
    ]:
        template = registry.program(
            "instruction_templates", str(profile["instruction_template_ref"]["id"])
        )
        overrides = profile["instruction_parameters"]
        bounds = {
            name: {
                "current_value": overrides.get(name, contract["default"]),
                **dict(contract),
            }
            for name, contract in template["parameters"].items()
            if name in advertised
        }
        if bounds:
            result[str(profile["role"])] = bounds
    return deep_thaw_json(result)


def _legal_directive_grammar(revision: CandidateRevision) -> dict[str, Any]:
    """Publish the authoring grammar and the directive the candidate has now.

    Same intent as the bounds publishers above: an axis name plus an operation
    id was not enough here, because this operation's payload is a structure
    rather than a number.  The directive the candidate holds today is not
    repeated here -- `current_candidate_state` already carries it per role, and
    the operator rejects an authored value equal to it -- so this publishes only
    `current_directive_present`, which is the difference between writing a
    first directive and revising one.
    """

    genome = EcologyEvolutionPluginGenome.from_dict(dict(revision.genome))
    return deep_thaw_json(_directive_authoring_contract(genome))


def _recent_local_edit_history(
    state: Any,
    candidate_id: str,
    before_batch_index: int,
) -> list[dict[str, Any]]:
    """Return only bounded proposal outcomes from earlier batches in this lane."""

    revision_ids = _batch_revision_ids(state, candidate_id)
    for activation in getattr(state, "trajectory_revision_activations", ()):
        if (
            getattr(activation, "candidate_id", None) == candidate_id
            and isinstance(getattr(activation, "batch_index", None), int)
            and not isinstance(getattr(activation, "batch_index", None), bool)
            and isinstance(getattr(activation, "from_revision_id", None), str)
            and str(activation.from_revision_id).strip()
        ):
            revision_ids[int(activation.batch_index)] = str(
                activation.from_revision_id
            )
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
                # Preserve observed scores but keep different batch cohorts
                # distinct from a same-cohort comparison of edit effects.
                **({"revisit": dict(detail["revisit"])} if isinstance(detail.get("revisit"), Mapping) else {}),
                **({"score_comparability": "different_batch_cohort_diagnostic_only",
                    "edit_effect_established": False} if _cohort_scoped_editor(state) else {}),
                "batch_score": _batch_score(state, candidate_id, batch_index),
                "next_batch_score": _batch_score(
                    state, candidate_id, batch_index + 1
                ),
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


def _local_edit_execution_reason(metrics: Mapping[str, Any]) -> str | None:
    """Do not ask the scientific editor to learn from unavailable predictions."""
    sample = metrics.get("sample_execution")
    if not isinstance(sample, Mapping):
        return None
    failed = sample.get("failed_origin_samples", sample.get("failed_examples", 0))
    if isinstance(failed, (int, float)) and not isinstance(failed, bool) and failed > 0:
        return "batch_execution_incomplete_keep_champion"
    if sample.get("successful_agent_provenance_pass") is False:
        return "batch_agent_provenance_invalid"
    if sample.get("strict_agent_chain_pass") is False:
        return "batch_agent_provenance_invalid"
    if sample.get("coverage_pass") is False:
        return "batch_execution_coverage_insufficient"
    return None


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
        if (_cohort_scoped_editor(state) and proposal.revisit is not None
                and row.get("reason") == "repeated_behavior_requires_revisit_reason"):
            # Supplying the missing rationale may repair this specific
            # rejection. It cannot override schema, safety or no-op guards.
            continue
        if _local_edit_bundle_signature(row.get("operations")) == proposed_signature:
            return "duplicate_recent_rejected_bundle"
    return None


# Axes whose evolved values only reach a prediction when the sample Agent
# chooses to call a prediction tool.  ``instruction_profile`` and the other
# Agent-policy axes always reach it, because the Host installs them.
_TOOL_DEPENDENT_MUTATION_AXES = (
    "scientific_parameter",
    "registered_predictor",
    "feature_policy",
    "fit_policy",
    "uncertainty_policy",
)
_INACTIVE_AXIS_REJECTION_REASON = "inactive_mutation_axis_without_tool_usage"


def _agent_tool_usage(metrics: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return this batch's measured Agent prediction-tool usage, when known."""

    sample = metrics.get("sample_execution")
    if not isinstance(sample, Mapping):
        return None
    cells = sample.get("agent_prediction_cells")
    rate = sample.get("agent_tool_usage_rate")
    if isinstance(cells, bool) or not isinstance(cells, int) or cells <= 0:
        return None
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        return None

    def count(key: str) -> int:
        value = sample.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0
        return int(value)

    methods = sample.get("agent_prediction_method_counts")
    result = {
        "agent_prediction_cells": cells,
        "agent_tool_invocation_cells": count("agent_tool_invocation_cells"),
        "agent_tool_invocations": count("agent_tool_invocations"),
        "agent_tool_usage_rate": float(rate),
    }
    if isinstance(methods, Mapping):
        result["prediction_method_counts"] = {
            str(name): int(value)
            for name, value in methods.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
    return result


def _inert_mutation_axes(tool_usage: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Name the axes that provably could not have moved this batch's score.

    A ``scientific_parameter`` or registered-pipeline edit only changes the
    default fit the prediction tools return.  When the batch shows the Agent
    never called one, the whole axis is a dead coordinate for the next batch
    too, and an operation spent there buys a guaranteed no-signal step.  The
    measurement is required: an unmeasured batch keeps every axis open.
    """

    if tool_usage is None:
        return ()
    rate = tool_usage.get("agent_tool_usage_rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate > 0.0:
        return ()
    return _TOOL_DEPENDENT_MUTATION_AXES


def _inactive_axis_rejection_reason(
    metrics: Mapping[str, Any],
    proposal: LocalEditProposal,
) -> str | None:
    """Reject an edit authored on an axis this batch proved the Agent ignores."""

    if proposal.decision.value != "mutate":
        return None
    inert = set(_inert_mutation_axes(_agent_tool_usage(metrics)))
    if not inert:
        return None
    for operation in proposal.operations:
        try:
            axis, _target, _path = _operation_target(operation)
        except (TypeError, ValueError):
            return None
        if axis in inert:
            return _INACTIVE_AXIS_REJECTION_REASON
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
                "successful_agent_provenance_pass",
                "successful_agent_provenance_coverage",
                "execution_complete",
                "strict_agent_chain_coverage",
                "complete_origin_agent_chains",
                "failed_examples",
                "scoring_fallback_examples",
                "failure_counts",
                "reason_code_counts",
                "recovered_by_failure_class",
                "critic_outcome_counts",
                "feedback_diagnostic_scope",
                "feedback_diagnostics_complete",
                "feedback_executed_scoring_cells",
                "feedback_resumed_scoring_cells",
                # Whether the Agent ever reached a prediction tool decides
                # which mutation axes can move the score at all.
                "agent_prediction_cells",
                "agent_tool_invocation_cells",
                "agent_tool_invocations",
                "agent_tool_usage_rate",
                "agent_prediction_method_counts",
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


def execute_formal_trajectory(services: GenerationRuntime, run_id: str, candidate_id: str) -> None:
    ensure_formal_trajectory(services, run_id, candidate_id)
    while execute_next_formal_batch(services, run_id, candidate_id):
        execute_next_local_edit(services, run_id, candidate_id)
    while execute_next_local_edit(services, run_id, candidate_id):
        pass


__all__ = [
    "ensure_formal_trajectory",
    "execute_formal_trajectory",
    "execute_next_formal_batch",
    "execute_next_local_edit",
]
