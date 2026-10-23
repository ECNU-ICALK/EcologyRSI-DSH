"""Restart-safe A/A control before interpreting the generation's candidates."""
from ..core.models import RunStatus
from ..core.trajectory import EvaluationScope, EvaluationPhase
from ..evaluators.repeatability import enabled, sealed_reference, summary_for
from ..evaluators.execution_validity import require_valid_execution
from ..evaluators.sample_execution import SampleExecutionPausedError, SampleExecutionCancelledError


def ensure_screening_repeatability(services, run_id, generation):
    from .generation_execution import _phase_task_manifest, _holdout_replay_inputs, EvaluationSession, _director_mutation
    from .formal_trajectory import _durable_batch_metrics
    state = services.director.state(run_id)
    if not enabled(state.task_manifest.metadata) or summary_for(state, generation) is not None:
        return True
    if state.run.status is not RunStatus.RUNNING:
        return False
    cid, rid, _reference = sealed_reference(state, generation)
    cohort = state.generation_cohort_for(generation).screening
    scope = EvaluationScope(run_id=run_id, generation=generation, candidate_id=cid,
                            candidate_revision_id=rid, phase=EvaluationPhase.SCREENING,
                            cohort_digest=cohort.cohort_digest, origin_count=cohort.origin_count,
                            inference_replica=1)
    task = _phase_task_manifest(state.task_manifest, generation, "screening")
    candidate, proposal, spec = _holdout_replay_inputs(state, state.candidate(cid), generation, rid, task)
    session = EvaluationSession(services, run_id=run_id, generation=generation,
                                proposal_id=proposal.proposal_id, candidate_id=cid, scope=scope)
    try:
        bundle = services.evaluators.evaluate_scientific(task, candidate, proposal, scope=scope,
                    cohort=cohort, algorithm_spec=spec, **session.evaluation_kwargs())
    except (SampleExecutionPausedError, SampleExecutionCancelledError):
        return False
    require_valid_execution(bundle.evaluation.metrics, cohort.origin_count,
                            phase="screening_repeatability", metadata=task.metadata)
    if bundle.sample_results is None:
        raise RuntimeError("repeatability is missing durable sample results")
    evaluation_id = f"{run_id}:generation:{generation}:screening-repeatability"
    metrics = {**_durable_batch_metrics(bundle.evaluation.metrics),
               "screening_evaluator_digest": bundle.evaluation.evaluator_digest}
    _director_mutation(services, "record_screening_repeatability", run_id, scope, metrics,
                      bundle.evaluation.score, session.completion_payload(evaluation_id, bundle.sample_results))
    return True
