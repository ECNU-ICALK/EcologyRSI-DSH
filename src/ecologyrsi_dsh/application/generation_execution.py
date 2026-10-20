"""Execution of one resumable, frozen sibling-candidate generation."""

from __future__ import annotations

from ..evolution.effect_contracts import hard_effect_failure
from ..core.training_race import race_bindings
from ..core.training_effect_evidence import finalist_training_effect_evidence
from ..core.review_policy import build_review_policy, review_result_metrics

from ..evolution.schedule import ADAPTIVE_PROTOCOLS

from ..evaluators.agent_stability import (
    capacity_with_inference_replicas,
)

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any
from .ports import GenerationRuntime

from ..core.errors import (
    dsh_native_runtime_error_in_chain,
    dsh_native_runtime_retryable,
    dsh_native_runtime_evaluation_fatal,
    walk_exception_graph,
)
from ..core.models import (
    Candidate,
    CandidateRole,
    CandidateStatus,
    Evaluation,
    Promotion,
    PromotionDecision,
    Proposal,
    RunStatus,
    TaskManifest,
    canonical_json,
    digest,
)
from ..core.protocols import (
    is_strict_origin_protocol,
    supports_two_stage_screening,
)
from ..core.state import (
    uses_global_incumbent_protocol,
    uses_positive_delta_search_protocol,
)
from ..core.redaction import (
    public_error_summary,
    public_exception_summary,
    safe_error_code,
)
from ..core.sample_results import (
    build_sample_results,
    sample_result_batch_event_payload,
    sample_results_completion_payload,
    sample_results_event_payload,
)
from ..core.screening import screening_cohort_digest
from ..core.search_policy import guarded_search
from ..core.trajectory import (
    CandidateRevision,
    EvaluationPhase,
    EvaluationScope,
    FormalBatchComparisonDecision,
    GenerationComparison,
    HoldoutArm,
    HoldoutEvaluation,
    RevisionStatus,
    TrajectoryStatus,
)
from ..evaluators.epoch_cohorts import (
    estimate_epoch_capacity,
    plan_generation_selection_cohorts,
    plan_run_adaptation_cohort,
)
from ..evaluators.fitness import FitnessProfile
from ..evaluators.greenhouse_prediction import origin_history_alignment_hours
from ..evaluators.registry import EvaluationBundle
from ..evaluators.generation_comparison import build_generation_comparison
from ..evaluators.sample_execution import (
    SampleExecutionCancelledError,
    SampleExecutionControlError,
    SampleExecutionPausedError,
    SampleResultCallbackError,
)
from ..evolution.batches import start_generation_batch
from ..evolution.analysis import (
    GenerationAnalysis,
)
from ..evolution.context import safe_aggregate_feedback
from ..evolution.genome import deep_thaw_json
from ..evolution.schedule import (
    PAIRED_LOCAL_EVALUATION_MODE,
    OptimizationSchedule,
)
from ..integrations.dsh_native_runtime import (
    DSH_NATIVE_EXECUTION_PROTOCOL,
    DshNativeRuntimeUnavailableError,
)
from ..integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ..knowledge.algorithm_smoke import (
    AlgorithmSmokeError,
    smoke_test_algorithm_spec,
)
from ..knowledge.algorithms import (
    AlgorithmAttempt,
    AlgorithmSpec,
    compile_algorithm_spec,
    debug_algorithm_spec,
)
from ..core.errors import DshToolAdmissionClosedError
from .candidate_scheduler import (
    CandidateEvaluationTask,
    run_candidate_evaluations,
)

_ALGORITHM_SMOKE_MAX_ATTEMPTS = 3
_MAX_CANDIDATE_CONCURRENCY = 8
_SCREENING_ORIGIN_COUNT = 64
_FORMAL_FINALIST_COUNT = 2
_ADAPTIVE_REFLECTION_BATCH_LIMIT = 10
_ADAPTIVE_CELL_LIMIT = 16
_ADAPTIVE_FAILURE_LIMIT = 16


def _is_search_candidate(candidate: Any) -> bool:
    """Keep legacy/mocked candidates searchable while excluding controls."""

    return getattr(candidate, "role", CandidateRole.SEARCH) in {
        CandidateRole.SEARCH,
        CandidateRole.SEARCH.value,
    }


def _sample_run_control(director: Any, run_id: str) -> str:
    """Map the indexed durable lifecycle status to evaluator control."""

    status = director.run_status(run_id)
    if status is RunStatus.RUNNING:
        return "running"
    if status is RunStatus.PAUSED:
        return "paused"
    return "cancelled"


def _run_admission_open(director: Any, run_id: str) -> bool:
    """Allow new work only while the durable run status is running."""

    return director.run_status(run_id) is RunStatus.RUNNING


def _sample_publication_open(director: Any, run_id: str) -> bool:
    """Allow in-flight sample checkpoints while running or draining a pause."""

    return director.run_status(run_id) in {RunStatus.RUNNING, RunStatus.PAUSED}


def _select_screening_finalists(
    candidates: Any,
    screening_by_candidate_id: Mapping[str, Mapping[str, Any]],
    *,
    top_k: int = _FORMAL_FINALIST_COUNT,
) -> tuple[Any, ...]:
    """Freeze finalists with an evidence-only, deterministic tie break."""

    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.candidate_id in screening_by_candidate_id
    )
    ranked = sorted(
        eligible,
        key=lambda candidate: (
            bool(hard_effect_failure(screening_by_candidate_id[candidate.candidate_id].get("metrics", {}))),
            int(
                screening_by_candidate_id[candidate.candidate_id].get(
                    "constraint_violations", 0
                )
            ),
            -float(screening_by_candidate_id[candidate.candidate_id]["score"]),
            int(candidate.slot_index),
            str(candidate.candidate_id),
        ),
    )
    return tuple(ranked[: max(1, int(top_k))])


def _formal_selection_event(state: Any, generation: int) -> Any | None:
    lookup = getattr(state, "formal_selection_for", None)
    return lookup(generation) if callable(lookup) else None


def _screening_records(
    state: Any, generation: int
) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for candidate in state.candidates:
        event = state.screening_for(generation, candidate.candidate_id)
        if event is not None:
            records[candidate.candidate_id] = event.payload
    return records


def _two_stage_screening_enabled(state: Any, candidates: Any) -> bool:
    metadata = state.task_manifest.metadata
    return bool(
        len(tuple(candidates)) > _FORMAL_FINALIST_COUNT
        and metadata.get("optimization_protocol") in ADAPTIVE_PROTOCOLS
        and metadata.get("cohort_capacity_enforced") is True
        and supports_two_stage_screening(metadata.get("sample_agent_protocol"))
        and metadata.get("sample_budget_class") == "selection_eligible"
        and metadata.get("two_stage_evaluation_enabled", True) is True
    )


def _freeze_adaptive_generation_inputs(
    services: GenerationRuntime,
    run_id: str,
    generation: int,
) -> None:
    """Freeze R0 and value-blind cohort identities before the first sample call."""

    state = services.director.state(run_id)
    metadata = state.task_manifest.metadata
    if (
        metadata.get("optimization_protocol") not in ADAPTIVE_PROTOCOLS
        or metadata.get("cohort_capacity_enforced") is not True
    ):
        return
    schedule = OptimizationSchedule.from_dict(metadata["optimization_schedule"])
    if generation == 0 and uses_global_incumbent_protocol(state.task_manifest):
        _director_mutation(
            services,
            "ensure_seed_incumbent_control",
            run_id,
        )
        state = services.director.state(run_id)
    candidates = sorted(
        (
            item
            for item in state.candidates
            if item.generation == generation
            and _is_search_candidate(item)
        ),
        key=lambda item: (item.slot_index, item.candidate_id),
    )
    if len(candidates) != 4:
        raise RuntimeError(
            "adaptive generation inputs require exactly four candidates"
        )
    for candidate in candidates:
        state = services.director.state(run_id)
        if state.initial_revision_for(candidate.candidate_id) is not None:
            continue
        genome = state.persisted_genome_for(candidate.candidate_id)
        lineage = dict(genome.lineage)
        mutation_digest = lineage.get("mutation_digest")
        if not isinstance(mutation_digest, str) or len(mutation_digest) != 64:
            mutation_digest = digest(
                {
                    "kind": "candidate-initial-revision",
                    "candidate_id": candidate.candidate_id,
                    "genome_digest": genome.genome_digest,
                }
            )
        revision = CandidateRevision(
            revision_id=f"revision:{candidate.candidate_id}:r0",
            run_id=run_id,
            generation=generation,
            candidate_id=candidate.candidate_id,
            genome=genome.to_dict(),
            genome_digest=genome.genome_digest,
            behavior_digest=genome.behavior_digest,
            mutation_digest=mutation_digest,
            status=RevisionStatus.ACTIVE,
        )
        _director_mutation(
            services,
            "create_candidate_revision",
            run_id,
            revision,
        )

    dataset = services.datasets.selection_view(
        state.task_manifest.visible_datasets[0],
        metadata.get("episode_id"),
        expected_dataset_digest=metadata.get("dataset_digest"),
        expected_split_manifest_digest=metadata.get("split_manifest_digest"),
        expected_data_protocol_digest=metadata.get("data_protocol_digest"),
    )
    # Every planned origin bakes in the history depth it was planned at, so the
    # planner has to read the alignment this run froze -- the same value the
    # evaluator intersects the cohort down to. Planning deeper than the
    # evaluator serves (or shallower) is the defect that turns one unservable
    # origin into a non-retryable run failure.
    history_steps = origin_history_alignment_hours(metadata)
    report = estimate_epoch_capacity(
        dataset,
        schedule=schedule,
        planned_generations=state.task_manifest.max_generations,
        seed=state.task_manifest.seed,
        scoring_cells_per_origin=int(metadata["prediction_cells_per_origin"]),
        history_steps=history_steps,
    )
    frozen_report = metadata.get("cohort_capacity_report")
    expected_report = (capacity_with_inference_replicas(report, schedule)
                       if metadata.get("sample_agent_mode") == "dsh_native_agent" else report.to_dict())
    if not isinstance(frozen_report, Mapping) or frozen_report.get(
        "planner_digest"
    ) != expected_report["planner_digest"]:
        raise RuntimeError("frozen cohort capacity report no longer matches dataset")
    adaptation = plan_run_adaptation_cohort(
        dataset,
        schedule=schedule,
        seed=state.task_manifest.seed,
        history_steps=history_steps,
    )
    planned = plan_generation_selection_cohorts(
        dataset,
        schedule=schedule,
        generation=generation,
        adaptation=adaptation,
        seed=state.task_manifest.seed,
        history_steps=history_steps,
    )
    planned = replace(planned, revision_bindings=race_bindings(services.director.state(run_id), generation))
    _director_mutation(
        services,
        "freeze_run_adaptation_cohort",
        run_id,
        adaptation,
    )
    _director_mutation(
        services,
        "freeze_generation_selection_cohorts",
        run_id,
        planned,
    )


def _phase_task_manifest(task: Any, generation: int, phase: str) -> Any:
    if phase not in {"screening", "formal", "holdout"}:
        raise ValueError("unknown evaluation phase")
    return replace(task, metadata={**dict(task.metadata), "evaluation_phase": phase,
                                   "two_stage_evaluation_enabled": True})


def _screen_candidate(services: GenerationRuntime, run_id: str, candidate_id: str, *, generation: int | None = None) -> None:
    """Evaluate the frozen ten-origin screening cohort with restart safety."""

    state = services.director.state(run_id)
    if state.run.status is not RunStatus.RUNNING:
        return
    candidate = state.candidate(candidate_id)
    generation = candidate.generation if generation is None else generation
    schedule = OptimizationSchedule.from_dict(state.task_manifest.metadata["optimization_schedule"])
    if candidate.candidate_id in _screening_records(state, generation):
        return
    pass
    generation_cohorts = state.generation_cohort_for(generation)
    if generation_cohorts is None:
        raise RuntimeError("screening requires frozen generation cohorts")
    screening_task = _phase_task_manifest(state.task_manifest, generation, "screening")
    bound = (generation_cohorts.revision_bindings or {}).get(candidate_id)
    if bound is None:
        raise ValueError("candidate is not admitted to frozen training race")
    revision = state.revision(bound)
    candidate, proposal, algorithm = _holdout_replay_inputs(state, candidate, generation, bound, screening_task)
    compiled = algorithm.to_dict()
    if revision is None or compiled is None:
        raise RuntimeError("screening requires compiled frozen revision")
    screening_cohort = generation_cohorts.screening
    screening_scope = EvaluationScope(
        run_id=run_id,
        generation=generation,
        candidate_id=candidate_id,
        candidate_revision_id=revision.revision_id,
        phase=EvaluationPhase.SCREENING,
        cohort_digest=screening_cohort.cohort_digest,
        origin_count=screening_cohort.origin_count,
    )

    callbacks = EvaluationSession(
        services,
        run_id=run_id,
        generation=generation,
        proposal_id=proposal.proposal_id,
        candidate_id=candidate_id,
        scope=screening_scope,
    )

    try:
        bundle = services.evaluators.evaluate_scientific(
            screening_task,
            candidate,
            proposal,
            scope=screening_scope,
            cohort=screening_cohort,
            algorithm_spec=AlgorithmSpec.from_dict(compiled),
            on_training_complete=lambda: None,
            **callbacks.evaluation_kwargs(),
        )
    except (SampleExecutionPausedError, SampleExecutionCancelledError):
        return
    except Exception as exc:  # noqa: BLE001 - preserve remote retry boundary
        if _recoverable_evaluation_error(exc):
            raise
        raise  # Preserve a resumable five-arm race; never hide missing control evidence.
        _director_mutation(
            services,
            "fail_candidate",
            run_id,
            candidate_id,
            f"候选筛选评测失败：{public_exception_summary(exc)}",
        )
        return

    from ..evaluators.execution_validity import require_valid_execution
    require_valid_execution(bundle.evaluation.metrics, screening_cohort.origin_count, phase="screening")
    summary = bundle.evaluation.metrics.get("sample_execution")
    attempted_origins = (
        int(summary.get("attempted_origin_samples", 0))
        if isinstance(summary, Mapping)
        else 0
    )
    prediction_cells = (
        int(summary.get("prediction_cell_count", 0))
        if isinstance(summary, Mapping)
        else 0
    )
    if attempted_origins != screening_cohort.origin_count:
        raise RuntimeError("screening did not cover the frozen 64-origin cohort")
    screening_evaluation_id = (
        f"screening-evaluation:{candidate_id}:{generation}"
    )
    from .formal_trajectory import _durable_batch_metrics
    _director_mutation(
        services,
        "record_candidate_screening",
        run_id,
        candidate_id=candidate_id,
        generation=generation,
        **({"metrics": {**_durable_batch_metrics(bundle.evaluation.metrics), "screening_evaluator_digest": bundle.evaluation.evaluator_digest},
            "candidate_revision_id": revision.revision_id}),
        score=bundle.evaluation.score,
        passed=bundle.evaluation.passed,
        constraint_violations=max(
            0, int(bundle.evaluation.metrics.get("constraint_violations", 0))
        ),
        origin_count=attempted_origins,
        prediction_cell_count=prediction_cells,
        cohort_digest=(
            str(bundle.evaluation.metrics.get("feedback_update_cohort_digest"))
            if bundle.evaluation.metrics.get("feedback_update_cohort_digest")
            else None
        ),
        sample_results=callbacks.completion_payload(
            screening_evaluation_id,
            bundle.sample_results,
        ),
    )


def _prepare_formal_finalists(
    services: GenerationRuntime,
    run_id: str,
    candidates: Any,
    *,
    max_concurrency: int,
) -> tuple[Any, ...]:
    state = services.director.state(run_id)
    generation = int(candidates[0].generation)
    frozen = _formal_selection_event(state, generation)
    schedule = OptimizationSchedule.from_dict(state.task_manifest.metadata.get("optimization_schedule") or OptimizationSchedule.for_new_run().to_dict())
    pass
    if frozen is None:
        race_ids = state.generation_cohort_for(generation).revision_bindings
        evaluation_candidates = tuple(state.candidate(cid) for cid in race_ids) if race_ids else candidates
        tasks = tuple(
            CandidateEvaluationTask(
                slot_index=index,
                candidate_id=str(candidate.candidate_id),
            )
            for index, candidate in enumerate(evaluation_candidates)
            if candidate.status is CandidateStatus.SPAWNED or schedule.race
        )
        run_candidate_evaluations(
            tasks,
            max_concurrency=max_concurrency,
            evaluate=lambda candidate_id: _screen_candidate(
                services, run_id, candidate_id, generation=generation
            ),
            admission_open=lambda: _run_admission_open(
                services.director, run_id
            ),
        )
        state = services.director.state(run_id)
        if state.run.status is not RunStatus.RUNNING:
            return ()
        screening = _screening_records(state, generation)
        refreshed = tuple(state.candidate(item.candidate_id) for item in candidates)
        finalists = _select_screening_finalists(
            refreshed,
            screening,
            top_k=schedule.finalist_count,
        )
        if len(finalists) < schedule.finalist_count:
            _director_mutation(
                services,
                "fail_run",
                run_id,
                f"两阶段评估失败：筛选阶段不足 {schedule.finalist_count} 个可用候选。",
            )
            return ()
        selected_ids = tuple(item.candidate_id for item in finalists)
        frozen = _director_mutation(
            services,
            "freeze_formal_selection_cohort",
            run_id,
            generation=generation,
            selected_candidate_ids=selected_ids,
            screening_digest=screening_cohort_digest(tuple(screening.values())),
            include_exploration_state=uses_global_incumbent_protocol(
                state.task_manifest
            ) and not schedule.race,
        )
        state = services.director.state(run_id)
    selected_ids = tuple(frozen.payload["selected_candidate_ids"])
    for candidate in tuple(state.candidate(item.candidate_id) for item in candidates):
        if (
            candidate.candidate_id not in selected_ids
            and candidate.status is CandidateStatus.SPAWNED
        ):
            _director_mutation(
                services,
                "screen_out_candidate",
                run_id,
                candidate.candidate_id,
                generation=generation,
                formal_selection_event_id=frozen.event_id,
            )
    state = services.director.state(run_id)
    return tuple(state.candidate(candidate_id) for candidate_id in selected_ids)


def _director_mutation(
    services: GenerationRuntime,
    method_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Serialize one short Director/ledger mutation, never remote work."""

    method = getattr(services.director, method_name)
    lock = getattr(services, "mutation_lock", None)
    if lock is None:
        return method(*args, **kwargs)
    with lock:
        return method(*args, **kwargs)


def _quiesce_native_terminal(services: GenerationRuntime, state: Any, terminal: str) -> None:
    """Close Host admission and dispose DSH role agents before Python terminal state."""

    if (
        state.task_manifest.metadata.get("execution_protocol")
        != DSH_NATIVE_EXECUTION_PROTOCOL
    ):
        return
    dsh_tools = getattr(services, "dsh_tools", None)
    if dsh_tools is not None:
        dsh_tools.close_run_admissions(state.run.run_id)
    runtime = getattr(services, "dsh_native_runtime", None)
    if runtime is None:
        return
    revision = state.events[-1].seq
    try:
        runtime.cancel(
            {
                "run_id": state.run.run_id,
                "run_state_revision": revision,
                "stage_attempt": 0,
                "ledger_expected_revision": services.ledger.latest_seq(),
                "idempotency_key": (
                    f"terminal:{terminal}:{state.run.run_id}:{revision}"
                ),
            }
        )
    except DshNativeRuntimeUnavailableError:
        # The durable Host admission fence remains closed. A missing DSH
        # process has no live children to accept; a later process restart does
        # not restore terminal run hosts.
        return










class EvaluationSession:
    """Durable per-origin callbacks shared by every adaptive evaluation scope.

    Screening, one formal batch, and one holdout arm still publish their
    scientific score only at the complete frozen-scope boundary.  The rows,
    progress, and usage leading to that score are checkpointed independently
    so a pause or process restart never has to replay already-settled origins.
    """

    def __init__(
        self,
        services: GenerationRuntime,
        *,
        run_id: str,
        generation: int,
        proposal_id: str,
        candidate_id: str,
        scope: EvaluationScope | None = None,
        on_evaluation_started: Any = None,
    ) -> None:
        self.services = services
        self.run_id = run_id
        self.generation = generation
        self.proposal_id = proposal_id
        self.candidate_id = candidate_id
        self.scope = scope
        self.on_evaluation_started = on_evaluation_started
        self.revision: str | None = None
        self.batch_index = 0
        self._started = False

    def _ensure_started(self) -> None:
        if self._started:
            return
        if callable(self.on_evaluation_started):
            self.on_evaluation_started()
        self._started = True

    def run_control(self) -> str:
        return _sample_run_control(self.services.director, self.run_id)

    def accepts_publication(self) -> bool:
        return _sample_publication_open(
            self.services.director, self.run_id
        )

    def prepare_checkpoint(
        self, checkpoint: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self._ensure_started()
        prepared = _director_mutation(
            self.services,
            "prepare_evaluation_sample_checkpoint",
            self.run_id,
            generation=self.generation,
            proposal_id=self.proposal_id,
            candidate_id=self.candidate_id,
            checkpoint=checkpoint,
            **({"scope": self.scope} if self.scope is not None else {}),
        )
        revision = prepared.get("revision")
        next_batch_index = prepared.get("next_batch_index")
        if not isinstance(revision, str) or not revision.strip():
            raise RuntimeError("sample checkpoint did not return a revision")
        if (
            isinstance(next_batch_index, bool)
            or not isinstance(next_batch_index, int)
            or next_batch_index < 1
        ):
            raise RuntimeError("sample checkpoint returned an invalid batch index")
        if self.revision is not None and self.revision != revision:
            raise RuntimeError("sample checkpoint changed revisions mid-evaluation")
        self.revision = revision
        self.batch_index = next_batch_index - 1
        return {
            **prepared,
        }

    def record_results(self, scoring_rows: Any) -> None:
        if not self.accepts_publication():
            return
        self._ensure_started()
        if self.revision is None:
            raise RuntimeError("sample result callback ran before checkpoint")
        projected = build_sample_results(self.candidate_id, scoring_rows)
        if not projected:
            return
        self.batch_index += 1
        try:
            _director_mutation(
                self.services,
                "record_evaluation_sample_result_batch",
                self.run_id,
                sample_result_batch_event_payload(
                    self.run_id,
                    self.candidate_id,
                    projected,
                    revision=self.revision,
                    batch_index=self.batch_index,
                ),
            )
        except Exception:
            if not self.accepts_publication():
                return
            raise


    def record_progress(self, progress: Mapping[str, Any]) -> None:
        if progress.get("role") != "planner" or not self.accepts_publication():
            return
        self._ensure_started()
        if self.revision is None:
            raise SampleResultCallbackError(
                "evaluation progress callback ran before checkpoint"
            )
        try:
            _director_mutation(
                self.services,
                "record_evaluation_progress",
                self.run_id,
                generation=self.generation,
                proposal_id=self.proposal_id,
                candidate_id=self.candidate_id,
                progress=progress,
                revision=self.revision,
            )
        except Exception as exc:
            if not self.accepts_publication():
                return
            raise SampleResultCallbackError(
                "evaluation progress heartbeat could not be persisted"
            ) from exc

    def evaluation_kwargs(self) -> dict[str, Any]:
        return {
            "on_sample_control": self.run_control,
            "on_sample_checkpoint": self.prepare_checkpoint,
            "on_sample_results": self.record_results,
            "on_evaluation_progress": self.record_progress,
        }

    def completion_payload(
        self,
        evaluation_id: str,
        sample_results: Any,
    ) -> Mapping[str, Any] | None:
        """Seal a scope only when its durable row stream actually exists."""

        if sample_results is None and self.revision is None:
            return None
        if sample_results is None or self.revision is None:
            raise RuntimeError(
                "scoped evaluation has incomplete durable sample-result state"
            )
        return sample_results_completion_payload(
            run_id=self.run_id,
            evaluation_id=evaluation_id,
            candidate_id=self.candidate_id,
            rows=sample_results,
            revision=self.revision,
        )




def _recoverable_evaluation_error(exc: BaseException) -> bool:
    """Propagate operational failures to the run owner for retry or termination.

    A candidate must only become permanently failed after a deterministic
    scientific or contract error.  Remote gateway errors can be wrapped by a
    sample adapter, and the result callback deliberately wraps ledger write
    failures, so inspect both boundaries before applying ``CandidateFailed``.
    """

    dsh_error = dsh_native_runtime_error_in_chain(exc)
    if dsh_error is not None and (
        dsh_native_runtime_retryable(dsh_error)
        or dsh_native_runtime_evaluation_fatal(dsh_error)
    ):
        return True
    for current in walk_exception_graph(exc):
        if isinstance(
            current,
            (
                DshToolAdmissionClosedError,
                SampleResultCallbackError,
                SampleExecutionControlError,
            ),
        ):
            return True
    return False


def _record_stage(
    services: GenerationRuntime,
    run_id: str,
    generation: int,
    stage: str,
    status: str,
    *,
    attempt: int = 1,
    proposal_id: str | None = None,
    candidate_id: str | None = None,
    public_error: str | None = None,
    event_id: str | None = None,
) -> None:
    _director_mutation(
        services,
        "record_evolution_stage",
        run_id,
        generation=generation,
        stage=stage,
        status=status,
        attempt=attempt,
        proposal_id=proposal_id,
        candidate_id=candidate_id,
        public_error=public_error_summary(public_error),
        event_id=event_id,
    )




def _candidate_signature(state: Any, proposal: Any) -> str:
    """Identify equivalent executable candidates across the whole run."""

    metadata = state.task_manifest.metadata
    proposal_metadata = (
        proposal.metadata if isinstance(proposal.metadata, Mapping) else {}
    )
    behavior_digest = proposal_metadata.get("behavior_digest")
    if not (
        isinstance(behavior_digest, str)
        and len(behavior_digest) == 64
        and all(character in "0123456789abcdef" for character in behavior_digest)
    ):
        behavior_digest = None
    execution_plan = proposal_metadata.get("derived_execution_plan")
    execution_plan_digest = (
        execution_plan.get("execution_digest") or execution_plan.get("plan_digest")
        if isinstance(execution_plan, Mapping)
        else None
    )
    from ..core.state import persisted_genome_from_proposal
    genome = persisted_genome_from_proposal(proposal)
    if genome is None:
        raise ValueError("candidate signature requires a frozen native genome")
    predictor_ref = genome.scientific_program["predictor_ref"]
    return canonical_json(
        {
            "parameters": dict(proposal.changes),
            "behavior_digest": behavior_digest,
            "derived_execution_semantics_digest": execution_plan_digest,
            "prediction_model_id": predictor_ref["id"],
            "prediction_model_digest": predictor_ref["catalog_digest"],
            "dataset_digest": metadata.get("dataset_digest")
            or list(state.task_manifest.visible_datasets),
            "evaluator_digest": metadata.get("evaluator_digest"),
        }
    )


def _spawn_generation_candidates(services: GenerationRuntime, run_id: str, batch: Any) -> bool:
    for slot_index in range(batch.batch_size):
        state = services.director.state(run_id)
        by_slot = {
            item.slot_index: item
            for item in state.candidates
            if item.generation == batch.generation
            and _is_search_candidate(item)
        }
        if slot_index in by_slot:
            continue
        linked = {item.proposal_id for item in state.candidates}
        orphan = next(
            (
                item
                for item in state.proposals
                if item.generation == batch.generation
                and item.proposal_id not in linked
            ),
            None,
        )
        _record_stage(
            services,
            run_id,
            batch.generation,
            "proposal",
            "started",
            event_id=f"{run_id}:stage:{batch.generation}:slot:{slot_index}:proposal:started",
        )
        if orphan is None:
            proposal = services.director.request_proposal(
                run_id,
                parent_candidate_id=batch.parent_candidate_id,
                generation_batch=batch,
                slot_index=slot_index,
            )
        else:
            proposal = orphan
        fallback = (
            proposal.metadata.get("host_fallback")
            if isinstance(proposal.metadata, Mapping)
            else None
        )
        proposal_attempt = 1
        if isinstance(fallback, Mapping) and fallback.get("applied") is True:
            _record_stage(
                services,
                run_id,
                batch.generation,
                "proposal",
                "failed",
                attempt=1,
                proposal_id=proposal.proposal_id,
                public_error=public_error_summary(
                    fallback.get("public_error")
                    or "远程策略提案不可用，已切换宿主有界回退"
                ),
                event_id=(
                    f"{run_id}:stage:{batch.generation}:slot:{slot_index}:"
                    "proposal:remote-failed"
                ),
            )
            strict_fallback = (
                services.director.state(run_id)
                .task_manifest.metadata.get("remote_fallback_policy")
                == "fail_run"
            )
            if strict_fallback:
                services.director.fail_run(
                    run_id,
                    "远程策略 API 不可用，连续进化已停止；"
                    + str(
                        public_error_summary(
                            fallback.get("public_error")
                            or "宿主回退被连续模式禁止"
                        )
                        or "宿主回退被连续模式禁止"
                    )[:400],
                )
                return False
            proposal_attempt = 2
        _record_stage(
            services,
            run_id,
            batch.generation,
            "proposal",
            "completed",
            attempt=proposal_attempt,
            proposal_id=proposal.proposal_id,
        )
        _record_stage(
            services,
            run_id,
            batch.generation,
            "candidate",
            "started",
            proposal_id=proposal.proposal_id,
        )
        candidate = services.director.spawn_candidate(
            run_id, proposal, slot_index=slot_index
        )
        _record_stage(
            services,
            run_id,
            batch.generation,
            "candidate",
            "completed",
            proposal_id=proposal.proposal_id,
            candidate_id=candidate.candidate_id,
        )
        signature = _candidate_signature(state, proposal)
        state = services.director.state(run_id)
        duplicate_of = next(
            (
                item
                for item in state.candidates
                if item.candidate_id != candidate.candidate_id
                and _is_search_candidate(item)
                and item.status is not CandidateStatus.DUPLICATE
                and _candidate_signature(state, state.proposal(item.proposal_id))
                == signature
            ),
            None,
        )
        if duplicate_of is not None:
            services.director.mark_candidate_duplicate(
                run_id, candidate.candidate_id, duplicate_of.candidate_id
            )
    return True


def _scientific_pass(evaluation: Evaluation) -> bool:
    return bool(evaluation.metrics.get("scientific_pass", evaluation.passed))


def _completed_judgment(
    scientific_evaluation: Evaluation,
    judged_evaluation: Evaluation,
) -> Evaluation:
    scientific_pass = _scientific_pass(scientific_evaluation)
    metrics = dict(judged_evaluation.metrics)
    # The append-only scientific evaluation owns the compressed full-sample
    # archive.  The judgment event keeps its digest and bounded preview without
    # duplicating hundreds of kilobytes per candidate.
    metrics.pop("sample_execution_trace_archive", None)
    metrics.update(
        {
            "scientific_pass": scientific_pass,
            "judge_status": "completed",
        }
    )
    data = judged_evaluation.to_dict()
    data.update(
        {
            "passed": scientific_pass and judged_evaluation.passed,
            "metrics": metrics,
        }
    )
    return Evaluation.from_dict(data)


def _unavailable_judgment(
    state: Any,
    evaluation: Evaluation,
    exc: BaseException,
) -> Evaluation:
    dsh_error = dsh_native_runtime_error_in_chain(exc)
    if dsh_error is not None:
        failure_class = (
            "transient"
            if dsh_native_runtime_retryable(dsh_error)
            else "permanent"
        )
        error_code = str(
            getattr(dsh_error, "error_code", "dsh_native_runtime_unavailable")
        )
    else:
        pending: list[tuple[BaseException, int]] = [(exc, 0)]
        seen: set[int] = set()
        transient = False
        permanent = False
        while pending:
            current, depth = pending.pop()
            identity = id(current)
            if identity in seen or depth > 32:
                continue
            seen.add(identity)
            permanent = permanent or isinstance(
                current,
                (
                    KeyError,
                    PermissionError,
                    TypeError,
                    ValueError,
                ),
            )
            transient = transient or isinstance(
                current,
                (ConnectionError, TimeoutError),
            )
            for related in (
                getattr(current, "__cause__", None),
                getattr(current, "__context__", None),
            ):
                if isinstance(related, BaseException):
                    pending.append((related, depth + 1))
            grouped = getattr(current, "exceptions", None)
            if isinstance(grouped, (tuple, list)):
                pending.extend(
                    (related, depth + 1)
                    for related in grouped
                    if isinstance(related, BaseException)
                )
        failure_class = (
            "permanent"
            if permanent
            else "transient"
            if transient
            else "unknown"
        )
        error_code = safe_error_code(type(exc).__name__, "judge_unavailable")
    metrics = dict(evaluation.metrics)
    metrics.pop("sample_execution_trace_archive", None)
    metrics.update(
        {
            "scientific_pass": _scientific_pass(evaluation),
            "judge_model_id": str(
                state.task_manifest.metadata.get("review_model_id")
                or state.task_manifest.metadata.get("judge_model_id") or "dsh-generation-judge"
            ),
            "judge_status": "unavailable",
            "judge_accepted": False,
            "judge_guidance": "独立评审暂不可用；保留科学评测结果，但禁止正式晋升。",
            "judge_parameter_override": {},
            "judge_error_type": type(exc).__name__,
            "judge_error_code": error_code,
            "judge_failure_class": failure_class,
        }
    )
    data = evaluation.to_dict()
    data.update({"passed": False, "metrics": metrics})
    return Evaluation.from_dict(data)


def _apply_candidate_judge(
    services: GenerationRuntime,
    state: Any,
    proposal: Any,
    artifact: Any,
    evaluation: Evaluation,
    *,
    incumbent: Any = None,
    training_effect_evidence: Mapping | None = None,
) -> None:
    candidate = state.candidate(evaluation.candidate_id)
    _record_stage(
        services,
        evaluation.run_id,
        candidate.generation,
        "judge",
        "started",
        proposal_id=proposal.proposal_id,
        candidate_id=candidate.candidate_id,
    )
    try:
        latest = services.director.state(evaluation.run_id)
        identity_binding = latest.candidate_identity_binding(candidate.candidate_id)
        if identity_binding is None:
            raise ValueError("DSH generation judge requires candidate identity binding")
        aggregate_metrics = safe_aggregate_feedback(
            dict(evaluation.metrics),
            name="generation judge aggregate metrics",
        )
        review_policy = build_review_policy(state.task_manifest, evaluation, incumbent,
                                            training_effect_evidence=training_effect_evidence)
        context = {
            "review_policy": review_policy,
            "candidate_id": candidate.candidate_id,
            "proposal_id": proposal.proposal_id,
            "generation": candidate.generation,
            "scientific_evaluation": {
                "score": evaluation.score,
                "passed": _scientific_pass(evaluation),
                "metrics": aggregate_metrics,
                "evaluation_digest": digest(evaluation.to_dict()),
                "artifact_digest": artifact.digest,
                "candidate_revision_id": evaluation.candidate_revision_id,
                "evaluation_scope": deep_thaw_json(evaluation.evaluation_scope),
            },
            "fitness_profile_digest": state.task_manifest.metadata.get(
                "fitness_profile_digest"
            ),
            "evaluation_cohort_digest": state.task_manifest.metadata.get(
                "evaluation_cohort_digest"
            ),
        }
        review = DshStructuredRoleRuntime(
            services.dsh_native_runtime,
            admission=services.dsh_tools,
        ).run(
            run_id=evaluation.run_id,
            stage="generation.judge",
            role="generation-judge",
            context=context,
            output_schema_id="ecology-generation-review@1",
            run_state_revision=latest.events[-1].seq,
            stage_attempt=1,
            ledger_expected_revision=services.ledger.latest_seq(),
            idempotency_key=(
                f"{evaluation.run_id}:candidate:{candidate.candidate_id}:judge"
            ),
            identity_digests={
                "genome_digest": str(identity_binding["genome_digest"]),
                "compiled_behavior_digest": str(
                    identity_binding["compiled_behavior_digest"]
                ),
                "phenotype_instance_digest": str(
                    identity_binding["phenotype_instance_digest"]
                ),
            },
        )
        if review.get("schema_version") != "ecology-generation-review@1":
            raise ValueError("DSH generation judge returned an unsupported schema")
        if not isinstance(review.get("accepted"), bool):
            raise TypeError("DSH generation judge accepted must be boolean")
        rationale = review.get("rationale")
        flags = review.get("flags")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("DSH generation judge rationale is required")
        if not isinstance(flags, list) or any(
            not isinstance(item, str) for item in flags
        ):
            raise TypeError("DSH generation judge flags must be text")
        data = evaluation.to_dict()
        data["passed"] = bool(review["accepted"])
        data["metrics"] = {
            **dict(evaluation.metrics),
            "judge_model_id": str(
                state.task_manifest.metadata.get("review_model_id")
                or "dsh-generation-judge"
            ),
            "judge_accepted": bool(review["accepted"]),
            "judge_guidance": rationale.strip(),
            "judge_flags": list(flags),
            "judge_parameter_override": {},
            "judge_result_digest": digest(review),
            **review_result_metrics(review_policy, review),
        }
        judged = EvaluationBundle(
            artifact=artifact,
            evaluation=Evaluation.from_dict(data),
        )
    except Exception as exc:
        dsh_error = dsh_native_runtime_error_in_chain(exc)
        if dsh_error is not None and dsh_native_runtime_retryable(dsh_error):
            # Keep the started stage resumable.  Reusing a ``failed`` stage
            # event with attempt=1 would collide with the next retry's
            # idempotent ``started`` event and strand the candidate.
            raise
        _director_mutation(
            services,
            "record_judgment",
            _unavailable_judgment(state, evaluation, exc)
        )
        _record_stage(
            services,
            evaluation.run_id,
            candidate.generation,
            "judge",
            "failed",
            proposal_id=proposal.proposal_id,
            candidate_id=candidate.candidate_id,
            public_error=public_exception_summary(exc),
        )
        return
    _director_mutation(
        services,
        "record_judgment",
        _completed_judgment(evaluation, judged.evaluation)
    )
    _record_stage(
        services,
        evaluation.run_id,
        candidate.generation,
        "judge",
        "completed",
        proposal_id=proposal.proposal_id,
        candidate_id=candidate.candidate_id,
    )


def _algorithm_failure_code(exc: BaseException, fallback: str) -> str:
    code = getattr(exc, "code", None)
    return safe_error_code(code, fallback) or fallback


def _smoke_failure_feedback(exc: BaseException, attempt: int) -> dict[str, Any]:
    retryable = bool(getattr(exc, "retryable", False)) or isinstance(
        exc,
        (ConnectionError, TimeoutError),
    )
    result: dict[str, Any] = {
        "attempt": attempt,
        "failure_code": _algorithm_failure_code(exc, "algorithm_smoke_failed"),
        "retryable": retryable,
        "exception_type": type(exc).__name__,
        "public_error": public_exception_summary(exc),
    }
    details = getattr(exc, "evidence", None)
    if isinstance(details, Mapping):
        result["details"] = dict(details)
    canonical_json(result)
    return result


def _validate_smoke_evidence(
    value: Any,
    spec: AlgorithmSpec,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AlgorithmSmokeError(
            "smoke_invalid_evidence",
            "algorithm smoke runner must return an evidence object",
        )
    result = dict(value)
    canonical_json(result)
    if len(canonical_json(result)) > 24_000:
        raise AlgorithmSmokeError(
            "smoke_invalid_evidence",
            "algorithm smoke evidence exceeds the bounded contract",
        )
    stored_smoke_digest = result.pop("smoke_digest", None)
    expected_smoke_digest = digest(result)
    result["smoke_digest"] = stored_smoke_digest
    if stored_smoke_digest != expected_smoke_digest:
        raise AlgorithmSmokeError(
            "smoke_digest_mismatch",
            "algorithm smoke evidence digest does not match its payload",
            evidence={
                "expected_smoke_digest": expected_smoke_digest,
                "observed_smoke_digest": stored_smoke_digest,
            },
        )
    if result.get("status") not in {"passed", "compatibility_skipped"}:
        raise AlgorithmSmokeError(
            "smoke_not_passed",
            "algorithm smoke runner did not report a passing status",
        )
    if result.get("source_partition") != "training_fit":
        raise AlgorithmSmokeError(
            "smoke_forbidden_partition",
            "algorithm smoke execution must use training_fit only",
        )
    if result.get("restricted_partition_access") is not False:
        raise AlgorithmSmokeError(
            "smoke_forbidden_partition",
            "algorithm smoke evidence did not prove the restricted partition boundary",
        )
    if result.get("status") == "passed":
        algorithm_ir = spec.algorithm_ir
        expected_ir_digest = (
            algorithm_ir.get("ir_digest")
            if isinstance(algorithm_ir, Mapping)
            else None
        )
        if result.get("algorithm_ir_digest") != expected_ir_digest:
            raise AlgorithmSmokeError(
                "smoke_ir_mismatch",
                "algorithm smoke evidence does not match the compiled IR",
            )
    return result


def _run_algorithm_smoke(
    services: GenerationRuntime,
    spec: AlgorithmSpec,
    task: Any,
    proposal: Any,
    *,
    attempt: int,
    failure_feedback: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    runner = getattr(services, "algorithm_smoke_runner", None)
    if callable(runner):
        evidence = runner(
            spec,
            task,
            proposal,
            attempt=attempt,
            failure_feedback=failure_feedback,
        )
    else:
        evaluator = services.evaluators
        datasets = getattr(evaluator, "datasets", None)
        if datasets is None:
            datasets = getattr(services, "datasets", None)
        evidence = smoke_test_algorithm_spec(
            spec,
            task,
            datasets,
            attempt=attempt,
            failure_feedback=failure_feedback,
        )
    return _validate_smoke_evidence(evidence, spec)


def _ensure_candidate_algorithm_ready(
    services: GenerationRuntime,
    state: Any,
    proposal: Any,
    candidate: Any,
) -> bool:
    """Compile and debug only a host-registered pipeline before training."""

    attempts = state.algorithm_attempts_for(candidate.candidate_id)
    if any(item.phase == "debug" and item.status == "passed" for item in attempts):
        return True
    if any(item.phase == "compile" and item.status == "failed" for item in attempts):
        return False

    compiled = state.compiled_algorithm_for(candidate.candidate_id)
    spec: AlgorithmSpec | None = None
    if compiled is not None:
        spec = AlgorithmSpec.from_dict(compiled)
    else:
        try:
            spec = compile_algorithm_spec(
                state.task_manifest,
                proposal,
                state.knowledge_for(candidate.generation),
            )
        except Exception as exc:  # noqa: BLE001 - isolate candidate compilation
            _director_mutation(
                services,
                "record_algorithm_attempt",
                AlgorithmAttempt(
                    run_id=candidate.run_id,
                    generation=candidate.generation,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    phase="compile",
                    attempt=1,
                    status="failed",
                    evidence={
                        "registered_adapters_only": True,
                        "exception_type": type(exc).__name__,
                    },
                    failure_code=_algorithm_failure_code(
                        exc, "algorithm_compile_failed"
                    ),
                    public_error=public_exception_summary(exc),
                )
            )
            _director_mutation(
                services,
                "fail_candidate",
                candidate.run_id,
                candidate.candidate_id,
                f"候选算法编译失败：{public_exception_summary(exc)}",
            )
            return False
        _director_mutation(
            services,
            "record_algorithm_attempt",
            AlgorithmAttempt(
                run_id=candidate.run_id,
                generation=candidate.generation,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
                phase="compile",
                attempt=1,
                status="passed",
                algorithm_spec_digest=spec.spec_digest,
                algorithm_spec=spec.to_dict(),
                evidence={
                    "registered_adapters_only": True,
                    "tool_count": len(spec.tool_ids),
                    "knowledge_mapping_count": sum(
                        item.get("decision") == "adopted"
                        for item in spec.knowledge_mappings
                    ),
                    "knowledge_not_selected_count": sum(
                        item.get("decision") != "adopted"
                        for item in spec.knowledge_mappings
                    ),
                },
            )
        )

    debug_attempts = [item for item in attempts if item.phase == "debug"]
    failure_feedback = tuple(
        dict(item.evidence["failure_feedback"])
        for item in debug_attempts
        if item.status == "failed"
        and isinstance(item.evidence.get("failure_feedback"), Mapping)
    )
    next_attempt = max((item.attempt for item in debug_attempts), default=0) + 1
    if next_attempt > _ALGORITHM_SMOKE_MAX_ATTEMPTS:
        _director_mutation(
            services,
            "fail_candidate",
            candidate.run_id,
            candidate.candidate_id,
            "候选算法 training_fit smoke 重试预算已耗尽。",
        )
        return False

    for attempt_number in range(
        next_attempt,
        _ALGORITHM_SMOKE_MAX_ATTEMPTS + 1,
    ):
        try:
            debug_evidence = debug_algorithm_spec(
                spec,
                state.task_manifest,
                proposal,
                state.knowledge_for(candidate.generation),
            )
        except Exception as exc:  # noqa: BLE001 - isolate static debug validation
            _director_mutation(
                services,
                "record_algorithm_attempt",
                AlgorithmAttempt(
                    run_id=candidate.run_id,
                    generation=candidate.generation,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    phase="debug",
                    attempt=attempt_number,
                    status="failed",
                    algorithm_spec_digest=spec.spec_digest,
                    evidence={
                        "stage": "static_debug",
                        "registered_adapters_only": True,
                        "exception_type": type(exc).__name__,
                    },
                    failure_code=_algorithm_failure_code(
                        exc, "algorithm_debug_failed"
                    ),
                    public_error=public_exception_summary(exc),
                )
            )
            _director_mutation(
                services,
                "fail_candidate",
                candidate.run_id,
                candidate.candidate_id,
                f"候选算法静态调试失败：{public_exception_summary(exc)}",
            )
            return False

        try:
            smoke_evidence = _run_algorithm_smoke(
                services,
                spec,
                state.task_manifest,
                proposal,
                attempt=attempt_number,
                failure_feedback=failure_feedback,
            )
        except Exception as exc:  # noqa: BLE001 - isolate registered smoke tools
            feedback = _smoke_failure_feedback(exc, attempt_number)
            _director_mutation(
                services,
                "record_algorithm_attempt",
                AlgorithmAttempt(
                    run_id=candidate.run_id,
                    generation=candidate.generation,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    phase="debug",
                    attempt=attempt_number,
                    status="failed",
                    algorithm_spec_digest=spec.spec_digest,
                    evidence={
                        "stage": "training_fit_smoke",
                        "registered_operators_only": True,
                        "algorithm_ir_digest": (
                            spec.algorithm_ir.get("ir_digest")
                            if isinstance(spec.algorithm_ir, Mapping)
                            else None
                        ),
                        "failure_feedback": feedback,
                    },
                    failure_code=str(feedback["failure_code"]),
                    public_error=str(feedback["public_error"]),
                )
            )
            failure_feedback = (*failure_feedback, feedback)
            if (
                not bool(feedback["retryable"])
                or attempt_number >= _ALGORITHM_SMOKE_MAX_ATTEMPTS
            ):
                _director_mutation(
                    services,
                    "fail_candidate",
                    candidate.run_id,
                    candidate.candidate_id,
                    "候选算法 training_fit smoke 失败："
                    + str(feedback["public_error"]),
                )
                return False
            continue

        _director_mutation(
            services,
            "record_algorithm_attempt",
            AlgorithmAttempt(
                run_id=candidate.run_id,
                generation=candidate.generation,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
                phase="debug",
                attempt=attempt_number,
                status="passed",
                algorithm_spec_digest=spec.spec_digest,
                evidence={
                    **debug_evidence,
                    "stage": "training_fit_smoke",
                    "smoke": smoke_evidence,
                    "prior_smoke_failures": [
                        dict(item) for item in failure_feedback
                    ],
                },
            )
        )
        return True
    return False




def _has_advanced_to_current_generation(state: Any) -> bool:
    """Return whether the event stream proves the current generation was closed.

    Candidate count alone is not enough to infer budget exhaustion: a process can
    stop after spawning the last candidate but before training and evaluation.
    ``GenerationAdvanced`` is emitted only after the batch decision barrier, so
    it is the durable evidence needed to recover a lost finalization write.
    """

    generation = int(state.run.generation)
    return generation > 0 and any(
        event.kind == "GenerationAdvanced"
        and int(event.payload.get("generation", -1)) == generation
        for event in state.events
    )


def _generation_all_duplicates(state: Any, generation: int) -> bool:
    """Return whether a complete generation only rediscovered prior work."""

    candidates = tuple(
        candidate
        for candidate in state.candidates
        if candidate.generation == generation
        and _is_search_candidate(candidate)
    )
    if not candidates:
        return False
    batch = state.batch_for(generation)
    if batch is not None and len(candidates) != batch.batch_size:
        return False
    return all(
        candidate.status is CandidateStatus.DUPLICATE for candidate in candidates
    )


def _generation_decision_finalized(state: Any, generation: int) -> bool:
    """Return whether the duplicate batch crossed its durable decision barrier."""

    analysis = state.analysis_for(generation)
    if analysis is None:
        return False
    return any(
        event.kind == "GenerationChampionSelected"
        and int(event.payload.get("generation", -1)) == generation
        and event.payload.get("analysis_digest") == analysis.analysis_digest
        for event in state.events
    )


def _complete_converged_run(services: GenerationRuntime, run_id: str, state: Any) -> Any:
    _quiesce_native_terminal(services, state, "converged")
    services.director.complete_run(
        run_id,
        termination_reason="search_space_converged_all_candidates_duplicate",
        outcome=(
            "completed_with_search_retained_candidate"
            if state.run.best_candidate_id is not None
            else "completed_without_acceptable_candidate"
        ),
    )
    return services.director.state(run_id)


def complete_if_budget_exhausted(
    services: GenerationRuntime, run_id: str, state: Any | None = None
) -> Any:
    """Idempotently finish a run whose final budget event was already written.

    A crash can occur after a generation decision commits and before
    ``RunCompleted`` commits. On retry, starting another batch would either
    reject the exhausted budget or repeat finalized work. This helper
    reconstructs the same termination decision from the durable projection
    and writes the missing terminal event.
    """

    state = state or services.director.state(run_id)
    if state.run.status is not RunStatus.RUNNING:
        return state

    # Duplicate-only convergence has no new scientific evaluation, so it is
    # terminal without consuming an epoch or writing GenerationAdvanced. The
    # finalized decision event prevents a partially analyzed batch from being
    # mistaken for completed convergence during recovery.
    current_generation = int(state.run.generation)
    if (
        _generation_all_duplicates(state, current_generation)
        and _generation_decision_finalized(state, current_generation)
    ):
        return _complete_converged_run(services, run_id, state)

    # Preserve recovery for ledgers written by older releases, which advanced
    # an all-duplicate generation before attempting RunCompleted.
    previous_generation = int(state.run.generation) - 1
    if (
        _has_advanced_to_current_generation(state)
        and previous_generation >= 0
        and _generation_all_duplicates(state, previous_generation)
    ):
        return _complete_converged_run(services, run_id, state)

    max_generations = max(1, int(state.task_manifest.max_generations))
    generation_exhausted = state.run.generation >= max_generations
    current_generation_has_candidates = any(
        candidate.generation == int(state.run.generation)
        and _is_search_candidate(candidate)
        for candidate in state.candidates
    )
    search_candidate_count = sum(
        _is_search_candidate(candidate) for candidate in state.candidates
    )
    candidate_exhausted = (
        search_candidate_count >= int(state.task_manifest.max_candidates)
        and _has_advanced_to_current_generation(state)
        # A retry can re-enter this preflight after the final candidate slot
        # has been spawned but before its evaluation and generation decision
        # have committed.  The occupied slot is not a completed budget until
        # GenerationAdvanced moves past that candidate's generation.
        and not current_generation_has_candidates
    )
    if not generation_exhausted and not candidate_exhausted:
        return state

    reasons: list[str] = []
    if candidate_exhausted:
        reasons.append("candidate_budget_exhausted")
    if generation_exhausted:
        reasons.append("generation_budget_exhausted")
    diagnostic_smoke = (
        is_strict_origin_protocol(
            state.task_manifest.metadata.get("sample_agent_protocol")
        )
        and state.task_manifest.metadata.get("sample_budget_class")
        == "diagnostic_smoke"
    )
    _quiesce_native_terminal(
        services,
        state,
        "diagnostic_smoke_completed" if diagnostic_smoke else "budget_exhausted",
    )
    services.director.complete_run(
        run_id,
        termination_reason=(
            "diagnostic_smoke_completed_no_promotion"
            if diagnostic_smoke
            else "+".join(reasons) or "budget_exhausted"
        ),
        outcome=(
            "diagnostic_smoke_completed"
            if diagnostic_smoke
            else "completed_with_search_retained_candidate"
            if state.run.best_candidate_id is not None
            else "budget_exhausted_without_acceptable_candidate"
        ),
    )
    return services.director.state(run_id)






def _evaluate_generation_candidates(
    services: GenerationRuntime,
    run_id: str,
    candidates: Any,
) -> None:
    """Evaluate one frozen sibling cohort with bounded candidate parallelism."""

    state = services.director.state(run_id)
    raw_concurrency = state.task_manifest.metadata.get("candidate_concurrency", 1)
    # Manifests created before candidate-level parallelism may carry an
    # explicit JSON null.  Preserve their historical serial execution rather
    # than treating the compatibility placeholder as a malformed new value.
    if raw_concurrency is None:
        raw_concurrency = 1
    if (
        isinstance(raw_concurrency, bool)
        or not isinstance(raw_concurrency, int)
        or not 1 <= raw_concurrency <= _MAX_CANDIDATE_CONCURRENCY
    ):
        raise ValueError(
            "candidate_concurrency must be an integer between 1 and "
            f"{_MAX_CANDIDATE_CONCURRENCY}"
        )
    frozen_candidates = tuple(candidates)
    if _two_stage_screening_enabled(state, frozen_candidates):
        from .formal_trajectory import execute_formal_trajectory

        frozen_candidates = _prepare_formal_finalists(
            services,
            run_id,
            frozen_candidates,
            max_concurrency=raw_concurrency,
        )
        if not frozen_candidates:
            return
        run_candidate_evaluations(
            tuple(
                CandidateEvaluationTask(
                    slot_index=int(candidate.slot_index),
                    candidate_id=str(candidate.candidate_id),
                )
                for candidate in frozen_candidates
            ),
            max_concurrency=min(raw_concurrency, len(frozen_candidates)),
            evaluate=lambda candidate_id: execute_formal_trajectory(
                services, run_id, candidate_id
            ),
            admission_open=lambda: _run_admission_open(
                services.director, run_id
            ),
        )
        return
    raise RuntimeError("joint_plugin_epoch@2 requires the frozen two-stage schedule")


def _holdout_replay_inputs(
    state: Any,
    candidate: Candidate,
    generation: int,
    revision_id: str,
    task: TaskManifest,
) -> tuple[Candidate, Proposal, AlgorithmSpec]:
    """Rebind a historical incumbent to the current holdout generation."""

    source = candidate
    if source.generation != generation:
        candidate = replace(
            source,
            generation=generation,
            status=CandidateStatus.SPAWNED,
            evaluation_id=None,
            promotion_id=None,
        )
    from .formal_trajectory import _revision_evaluation_inputs

    _revision, revised_proposal, spec = _revision_evaluation_inputs(
        state, candidate, revision_id, task
    )
    if source.generation != generation:
        # A historical control is measured as frozen behavior, not as a new
        # mutation against its former parent under this generation's retry floor.
        control_metadata = dict(revised_proposal.metadata)
        for field in ("effect_contract", "effect_resolution", "mutation_effect_resolution", "mutation_operations"):
            control_metadata.pop(field, None)
        revised_proposal = replace(revised_proposal, metadata=control_metadata)
    return candidate, revised_proposal, spec


def _execute_adaptive_holdout_arm(
    services: GenerationRuntime,
    run_id: str,
    generation: int,
    arm: HoldoutArm,
    binding: Mapping[str, str],
    cohort: Any,
) -> HoldoutEvaluation:
    state = services.director.state(run_id)
    existing = state.holdout_evaluation_for(generation, arm)
    scope = EvaluationScope(
        run_id=run_id,
        generation=generation,
        candidate_id=binding["candidate_id"],
        candidate_revision_id=binding["candidate_revision_id"],
        phase=EvaluationPhase.HOLDOUT,
        cohort_digest=cohort.cohort_digest,
        origin_count=cohort.origin_count,
        holdout_arm=arm,
    )
    existing_artifact = state.artifact_for(binding["candidate_id"])
    existing_canonical = state.evaluation_for(binding["candidate_id"])
    canonical_complete = _canonical_holdout_outcome_complete(
        existing_artifact,
        existing_canonical,
        scope,
    )
    if existing is not None and (
        arm is HoldoutArm.INCUMBENT or canonical_complete
    ):
        return existing
    if (
        existing is None
        and arm in {HoldoutArm.FINALIST_1, HoldoutArm.FINALIST_2}
        and canonical_complete
    ):
        assert existing_canonical is not None
        recovered = _holdout_from_canonical_evaluation(existing_canonical, scope)
        return _director_mutation(
            services,
            "record_holdout_evaluation",
            run_id,
            recovered,
        )

    started_event_id = (
        f"{run_id}:generation:{generation}:holdout:{arm.value}:started"
    )
    if not any(event.event_id == started_event_id for event in state.events):
        services.ledger.append(
            run_id,
            "HoldoutArmStarted",
            {
                "schema_version": "ecologyrsi-dsh.holdout-arm-started/1",
                "generation": generation,
                "holdout_arm": arm.value,
                "candidate_id": scope.candidate_id,
                "candidate_revision_id": scope.candidate_revision_id,
                "cohort_digest": scope.cohort_digest,
                "origin_count": scope.origin_count,
            },
            event_id=started_event_id,
            expected_run_seq=state.events[-1].seq,
        )
        state = services.director.state(run_id)

    source_candidate = state.candidate(binding["candidate_id"])
    task = _phase_task_manifest(
        state.task_manifest,
        generation,
        "holdout",
    )
    candidate, proposal, spec = _holdout_replay_inputs(
        state,
        source_candidate,
        generation,
        binding["candidate_revision_id"],
        task,
    )

    callbacks = EvaluationSession(
        services,
        run_id=run_id,
        generation=generation,
        proposal_id=candidate.proposal_id,
        candidate_id=candidate.candidate_id,
        scope=scope,
    )

    bundle = services.evaluators.evaluate_scientific(
        task,
        candidate,
        proposal,
        scope=scope,
        cohort=cohort,
        algorithm_spec=spec,
        **callbacks.evaluation_kwargs(),
    )
    metrics = dict(bundle.evaluation.metrics)
    from ..evaluators.execution_validity import require_valid_execution
    require_valid_execution(metrics, cohort.origin_count, phase="holdout")
    schedule = OptimizationSchedule.from_dict(state.task_manifest.metadata["optimization_schedule"])
    pass
    summary = metrics.get("sample_execution")
    if not isinstance(summary, Mapping) or int(summary.get("attempted_origin_samples", 0)) < cohort.origin_count:
        raise RuntimeError(f"holdout did not complete the frozen {cohort.origin_count}-origin cohort")
    evaluation = existing or HoldoutEvaluation(
        evaluation_id=f"holdout-evaluation:{generation}:{arm.value}",
        scope=scope,
        score=bundle.evaluation.score,
        passed=bundle.evaluation.passed,
        metrics=metrics,
        # The evaluator identity is part of the causal comparison contract and
        # must remain identical across incumbent and both finalist arms.  The
        # arm/scope is already carried by ``EvaluationScope`` and the stable
        # evaluation id; including it in this digest would make every arm look
        # like it used a different evaluator and block promotion.
        evaluator_digest=str(bundle.evaluation.evaluator_digest),
    )

    # Finalist holdout evidence is also the canonical candidate outcome used by
    # the existing parent/search pipeline.  The incumbent replay arm is never
    # written as a new candidate evaluation.
    if arm in {HoldoutArm.FINALIST_1, HoldoutArm.FINALIST_2}:
        evaluated_artifact = replace(
            bundle.artifact,
            candidate_revision_id=binding["candidate_revision_id"],
            evaluation_scope_digest=scope.scope_key,
        )
        if existing_artifact is not None:
            if existing_artifact.digest != evaluated_artifact.digest:
                raise RuntimeError(
                    "persisted finalist holdout artifact differs from recovered evaluation"
                )
            artifact = existing_artifact
        else:
            artifact = _director_mutation(
                services, "record_artifact", evaluated_artifact
            )
        canonical = replace(
            bundle.evaluation,
            score=evaluation.score,
            passed=evaluation.passed,
            # Recovered HoldoutEvaluation metrics are deeply frozen with
            # MappingProxyType.  A shallow dict() leaves nested proxies in the
            # new Evaluation and fails the JSON contract during resume.
            metrics=deep_thaw_json(evaluation.metrics),
            evaluator_digest=evaluation.evaluator_digest,
            candidate_revision_id=binding["candidate_revision_id"],
            evaluation_scope=scope.to_dict(),
            artifact_digest=artifact.digest,
        )
        if bundle.sample_results is None or callbacks.revision is None:
            raise RuntimeError(
                "finalist holdout is missing its durable sample-result checkpoint"
            )
        _director_mutation(
            services,
            "record_evaluation",
            canonical,
            sample_results=sample_results_event_payload(
                canonical,
                bundle.sample_results,
                revision=callbacks.revision,
            ),
        )
    if existing is None:
        _director_mutation(
            services,
            "record_holdout_evaluation",
            run_id,
            evaluation,
            sample_results=(
                callbacks.completion_payload(
                    evaluation.evaluation_id,
                    bundle.sample_results,
                )
                if arm is HoldoutArm.INCUMBENT
                else None
            ),
        )
    return evaluation


def _execute_joint_holdout_recheck(services, run_id, original, cohort):
    """One checkpointed second inference; original results remain immutable."""
    state = services.director.state(run_id)
    scope = replace(original.scope, inference_replica=1)
    existing = state.holdout_recheck_for(scope.generation, scope.holdout_arm)
    if existing is not None:
        return existing
    if not _run_admission_open(services.director, run_id):
        raise RuntimeError("joint holdout recheck interrupted by run control")
    event_id = f"{run_id}:generation:{scope.generation}:holdout:{scope.holdout_arm.value}:recheck-started"
    if not any(event.event_id == event_id for event in state.events):
        services.ledger.append(run_id, "HoldoutRecheckStarted", {
            "schema_version": "ecologyrsi-dsh.holdout-arm-started/1",
            "generation": scope.generation, "holdout_arm": scope.holdout_arm.value,
            "candidate_id": scope.candidate_id, "candidate_revision_id": scope.candidate_revision_id,
            "cohort_digest": scope.cohort_digest, "origin_count": scope.origin_count}, event_id=event_id)
    task = _phase_task_manifest(state.task_manifest, scope.generation, "holdout")
    candidate, proposal, spec = _holdout_replay_inputs(state,
        state.candidate(scope.candidate_id), scope.generation, scope.candidate_revision_id, task)
    callbacks = EvaluationSession(services, run_id=run_id, generation=scope.generation,
        proposal_id=candidate.proposal_id, candidate_id=candidate.candidate_id, scope=scope)
    bundle = services.evaluators.evaluate_scientific(task, candidate, proposal,
        scope=scope, cohort=cohort, algorithm_spec=spec, **callbacks.evaluation_kwargs())
    from ..evaluators.execution_validity import require_valid_execution
    require_valid_execution(bundle.evaluation.metrics, cohort.origin_count, phase="holdout_recheck")
    if bundle.sample_results is None:
        raise RuntimeError("joint recheck is missing its durable sample-result checkpoint")
    evaluation = HoldoutEvaluation(
        evaluation_id=f"holdout-recheck:{scope.generation}:{scope.holdout_arm.value}",
        scope=scope, score=bundle.evaluation.score, passed=bundle.evaluation.passed,
        metrics=deep_thaw_json(bundle.evaluation.metrics), evaluator_digest=str(bundle.evaluation.evaluator_digest))
    return _director_mutation(services, "record_holdout_evaluation", run_id, evaluation,
        sample_results=callbacks.completion_payload(evaluation.evaluation_id, bundle.sample_results))


def _canonical_holdout_outcome_complete(
    artifact: Any,
    evaluation: Evaluation | None,
    scope: EvaluationScope,
) -> bool:
    """Require the exact frozen holdout binding, never any candidate outcome."""

    return bool(
        artifact is not None
        and evaluation is not None
        and artifact.candidate_id == scope.candidate_id
        and artifact.candidate_revision_id == scope.candidate_revision_id
        and artifact.evaluation_scope_digest == scope.scope_key
        and evaluation.candidate_id == scope.candidate_id
        and evaluation.candidate_revision_id == scope.candidate_revision_id
        and evaluation.evaluation_scope == scope.to_dict()
        and evaluation.artifact_digest == artifact.digest
    )


def _holdout_from_canonical_evaluation(
    evaluation: Evaluation,
    scope: EvaluationScope,
) -> HoldoutEvaluation:
    """Seal a missing holdout event from its already durable canonical result."""

    return HoldoutEvaluation(
        evaluation_id=(
            f"holdout-evaluation:{scope.generation}:{scope.holdout_arm.value}"
        ),
        scope=scope,
        score=evaluation.score,
        passed=evaluation.passed,
        metrics=dict(evaluation.metrics),
        evaluator_digest=evaluation.evaluator_digest,
        created_at=evaluation.created_at,
    )


def _bounded_operation_categories(operations: Any) -> list[str]:
    """Return a small, aggregate-only description of a mutation bundle."""

    if isinstance(operations, (str, bytes)) or not isinstance(
        operations, Sequence
    ):
        return []
    categories: list[str] = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        category = str(operation.get("op") or "unknown").strip()[:120]
        if category and category not in categories:
            categories.append(category)
        if len(categories) >= 5:
            break
    return categories


def _bounded_operation_targets(operations: Any) -> list[str]:
    """Return registered target identities without values or free-form prose."""

    if isinstance(operations, (str, bytes)) or not isinstance(
        operations, Sequence
    ):
        return []
    targets: list[str] = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        role = str(operation.get("role") or "").strip()[:80]
        target = next(
            (
                str(operation.get(name) or "").strip()[:120]
                for name in (
                    "name",
                    "predictor_id",
                    "program_id",
                    "instruction_template_id",
                    "workflow_template_id",
                )
                if str(operation.get(name) or "").strip()
            ),
            role,
        )
        if role and target and role != target:
            target = f"{role}:{target}"[:200]
        if target and target not in targets:
            targets.append(target)
        if len(targets) >= 5:
            break
    return targets


def _outer_mutation_evidence(state: Any, candidate: Candidate) -> dict[str, Any]:
    """Describe the once-per-generation mutation separately from local edits."""

    metadata = state.proposal(candidate.proposal_id).metadata
    operations = metadata.get("mutation_operations")
    operation_count = (
        len(operations)
        if isinstance(operations, Sequence)
        and not isinstance(operations, (str, bytes))
        else 0
    )
    return {
        "kind": "outer_generation_mutation",
        "direction_id": metadata.get("candidate_direction_id"),
        "direction_digest": metadata.get("candidate_direction_digest"),
        "source_behavior_digest": metadata.get("behavior_digest"),
        "operation_count": min(operation_count, 5),
        "operation_categories": _bounded_operation_categories(operations),
    }


def _adaptive_gate_failures(gate: Mapping[str, Any]) -> list[str]:
    """Explain a finalist gate from the canonical comparison artifact."""

    failures: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()[:180]
        if (
            text
            and text not in failures
            and len(failures) < _ADAPTIVE_FAILURE_LIMIT
        ):
            failures.append(text)

    positive_search_gate = "search_eligible" in gate
    raw_failures = (
        gate.get("search_failures")
        if positive_search_gate
        else gate.get("failures")
    )
    if isinstance(raw_failures, Sequence) and not isinstance(
        raw_failures, (str, bytes)
    ):
        for item in raw_failures:
            add(item)
    if not positive_search_gate and gate.get("passed") is not True:
        add("scientific_gate_failed")
    constraint_violations = gate.get("constraint_violations")
    if (
        isinstance(constraint_violations, (int, float))
        and not isinstance(constraint_violations, bool)
        and constraint_violations > 0
    ):
        add("constraint_violations")
    if gate.get("complete_objective_grid") is False:
        add("objective_grid_incomplete")
    if gate.get("coverage_pass") is False:
        add("coverage_failed")
    if not positive_search_gate and gate.get("no_cell_regression") is False:
        add("cell_regression")
    assessment = gate.get("promotion_assessment")
    if not positive_search_gate and isinstance(assessment, Mapping) and assessment.get(
        "primary_selection_gate"
    ) is not True:
        add(assessment.get("status") or "primary_selection_gate_failed")
    if gate.get("eligible") is not True and not failures:
        add("comparison_gate_ineligible")
    return failures


def _bounded_comparison_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Project the comparison gate without leaking evaluator sample records."""

    cell_deltas = gate.get("cell_deltas")
    bounded_cells = (
        {
            str(key)[:160]: value
            for key, value in sorted(
                cell_deltas.items(), key=lambda item: str(item[0])
            )[:_ADAPTIVE_CELL_LIMIT]
        }
        if isinstance(cell_deltas, Mapping)
        else {}
    )
    assessment = gate.get("promotion_assessment")
    bounded_assessment = (
        {
            name: assessment.get(name)
            for name in (
                "evidence_class",
                "status",
                "paired_block_count",
                "valid_three_day_start_count",
                "paired_block_ids_digest",
                "primary_delta",
                "selection_stability_floor",
                "primary_selection_gate",
            )
        }
        if isinstance(assessment, Mapping)
        else {}
    )
    return deep_thaw_json(
        {
            "eligible": gate.get("eligible") is True,
            "search_eligible": gate.get("search_eligible"),
            "certification_eligible": gate.get("certification_eligible"),
            "scientific_pass": gate.get("passed") is True,
            "constraint_violations": gate.get("constraint_violations"),
            "overall_coverage": gate.get("overall_coverage"),
            "coverage_pass": gate.get("coverage_pass"),
            "complete_objective_grid": gate.get("complete_objective_grid"),
            "no_cell_regression": gate.get("no_cell_regression"),
            "worst_cell_delta": gate.get("worst_cell_delta"),
            "cell_deltas": bounded_cells,
            "stability_lower_bound": gate.get("stability_lower_bound"),
            "strict_agent_chain_pass": gate.get("strict_agent_chain_pass"),
            "promotion_assessment": bounded_assessment,
            "failures": _adaptive_gate_failures(gate),
            "search_failures": list(gate.get("search_failures", ()))[:8],
            "certification_failures": list(
                gate.get("certification_failures", ())
            )[:8],
        }
    )


def _local_edit_trajectory_evidence(
    state: Any,
    candidate_id: str,
    final_revision_id: str,
) -> dict[str, Any]:
    """Build bounded ten-batch lineage evidence for one completed finalist."""

    trajectory = state.trajectory_for(candidate_id)
    if trajectory is None:
        raise RuntimeError("adaptive finalist is missing its formal trajectory")
    if trajectory.final_revision_id != final_revision_id:
        raise RuntimeError(
            "adaptive holdout revision does not match the completed trajectory"
        )
    formal_batches = sorted(
        (
            item
            for item in getattr(state, "formal_batches", ())
            if item.candidate_id == candidate_id
        ),
        key=lambda item: item.batch_index,
    )
    if (
        len(formal_batches) != trajectory.batch_count
        or [item.batch_index for item in formal_batches]
        != list(range(trajectory.batch_count))
    ):
        raise RuntimeError("adaptive finalist batch history is incomplete")
    schedule_value = getattr(state.task_manifest, "metadata", {}).get(
        "optimization_schedule"
    )
    paired_mode = bool(
        isinstance(schedule_value, Mapping)
        and OptimizationSchedule.from_dict(schedule_value).local_evaluation_mode
        == PAIRED_LOCAL_EVALUATION_MODE
    )
    proposal_by_batch = {
        int(item["batch_index"]): item
        for item in getattr(state, "local_edit_proposals", ())
        if item.get("candidate_id") == candidate_id
        and isinstance(item.get("batch_index"), int)
        and not isinstance(item.get("batch_index"), bool)
    }
    outcome_by_batch = {
        int(item["batch_index"]): item
        for item in getattr(state, "local_edit_outcomes", ())
        if item.get("candidate_id") == candidate_id
        and isinstance(item.get("batch_index"), int)
        and not isinstance(item.get("batch_index"), bool)
    }
    activation_by_batch = {
        item.batch_index: item
        for item in getattr(state, "trajectory_revision_activations", ())
        if item.candidate_id == candidate_id
    }
    outcome_counts: dict[str, int] = {}
    operation_category_counts: dict[str, int] = {}
    local_rows: dict[int, dict[str, Any]] = {}
    local_edit_count = (
        trajectory.batch_count - 1 if paired_mode else trajectory.batch_count
    )
    for batch_index in range(local_edit_count):
        proposal = proposal_by_batch.get(batch_index)
        outcome = outcome_by_batch.get(batch_index)
        activation = activation_by_batch.get(batch_index)
        if proposal is None or outcome is None or activation is None:
            raise RuntimeError("adaptive finalist local-edit history is incomplete")
        detail = proposal.get("proposal", proposal)
        operations = (
            detail.get("operations", ()) if isinstance(detail, Mapping) else ()
        )
        categories = _bounded_operation_categories(operations)
        category = "|".join(categories) if categories else "none"
        targets = _bounded_operation_targets(operations)
        target = "|".join(targets) if targets else "none"
        outcome_name = str(outcome.get("outcome") or "unknown")[:120]
        outcome_counts[outcome_name] = outcome_counts.get(outcome_name, 0) + 1
        for item in categories:
            operation_category_counts[item] = (
                operation_category_counts.get(item, 0) + 1
            )
        reason = getattr(activation.reason, "value", activation.reason)
        local_rows[batch_index] = {
            "batch_index": batch_index,
            "batch_number": batch_index + 1,
            "decision": (
                detail.get("decision") if isinstance(detail, Mapping) else None
            ),
            # Scalars keep the aggregate reflection envelope within its strict
            # depth bound and intentionally omit mutation values.
            "operation_category": category,
            "operation_targets": target,
            "operation_count": min(
                len(operations)
                if isinstance(operations, Sequence)
                and not isinstance(operations, (str, bytes))
                else 0,
                5,
            ),
            "outcome": outcome_name,
            "outcome_reason": str(outcome.get("reason") or "")[:240] or None,
            "mutation_parent_revision_id": activation.from_revision_id,
            "generated_challenger_revision_id": activation.to_revision_id,
            "advance_reason": str(reason)[:120],
        }

    if paired_mode:
        if any(
            batch_index in proposal_by_batch
            or batch_index in outcome_by_batch
            or batch_index in activation_by_batch
            for batch_index in range(local_edit_count, trajectory.batch_count)
        ):
            raise RuntimeError("paired finalist final batch cannot contain a local edit")
        comparisons = sorted(
            (
                item
                for item in getattr(state, "formal_batch_comparisons", ())
                if item.candidate_id == candidate_id
            ),
            key=lambda item: item.batch_index,
        )
        if (
            len(comparisons) != trajectory.batch_count
            or [item.batch_index for item in comparisons]
            != list(range(trajectory.batch_count))
            or comparisons[-1].champion_after_revision_id != final_revision_id
        ):
            raise RuntimeError("adaptive finalist comparison history is incomplete")
        batch_rows: list[dict[str, Any]] = []
        revision_chain = [trajectory.initial_revision_id]
        for comparison in comparisons:
            prior_local = local_rows.get(comparison.batch_index - 1)
            next_local = local_rows.get(comparison.batch_index)
            distinct_challenger = (
                comparison.challenger_revision_id
                != comparison.champion_before_revision_id
            )
            prior_challenger_edit = (
                prior_local
                if prior_local is not None
                and prior_local["outcome"] == "applied"
                and prior_local["generated_challenger_revision_id"]
                == comparison.challenger_revision_id
                else None
            )
            if distinct_challenger and prior_challenger_edit is None:
                raise RuntimeError(
                    "adaptive finalist challenger operation history is incomplete"
                )
            if (
                next_local is not None
                and next_local["mutation_parent_revision_id"]
                != comparison.champion_after_revision_id
            ):
                raise RuntimeError(
                    "adaptive finalist mutation parent is not the durable champion"
                )
            decision = getattr(comparison.decision, "value", comparison.decision)
            challenger_was_rejected = bool(
                decision
                == FormalBatchComparisonDecision.CHAMPION_RETAINED.value
                and distinct_challenger
            )
            batch_rows.append(
                {
                    "batch_index": comparison.batch_index,
                    "batch_number": comparison.batch_index + 1,
                    "decision": str(decision)[:120],
                    "reason": str(comparison.reason)[:160],
                    "score_delta": comparison.score_delta,
                    "minimum_score_delta": comparison.minimum_score_delta,
                    "safety_gate_passed": comparison.safety_gate_passed,
                    "cell_regression_gate_passed": (
                        comparison.cell_regression_gate_passed
                    ),
                    "champion_before_revision_id": (
                        comparison.champion_before_revision_id
                    ),
                    "challenger_revision_id": comparison.challenger_revision_id,
                    "champion_after_revision_id": (
                        comparison.champion_after_revision_id
                    ),
                    "challenger_operation_category": (
                        prior_challenger_edit["operation_category"]
                        if prior_challenger_edit is not None
                        else "none"
                    ),
                    "challenger_operation_targets": (
                        prior_challenger_edit["operation_targets"]
                        if prior_challenger_edit is not None
                        else "none"
                    ),
                    "rejected_operation_category": (
                        prior_challenger_edit["operation_category"]
                        if challenger_was_rejected
                        and prior_challenger_edit is not None
                        else None
                    ),
                    "rejected_operation_targets": (
                        prior_challenger_edit["operation_targets"]
                        if challenger_was_rejected
                        and prior_challenger_edit is not None
                        else None
                    ),
                    "next_mutation_parent_revision_id": (
                        next_local["mutation_parent_revision_id"]
                        if next_local is not None
                        else None
                    ),
                    "next_challenger_revision_id": (
                        next_local["generated_challenger_revision_id"]
                        if next_local is not None
                        else None
                    ),
                }
            )
            revision_chain.append(comparison.champion_after_revision_id)
        included_comparisons = batch_rows[-_ADAPTIVE_REFLECTION_BATCH_LIMIT:]
        included_local_edits = [
            local_rows[index]
            for index in sorted(local_rows)[-_ADAPTIVE_REFLECTION_BATCH_LIMIT:]
        ]
        return {
            "kind": "batch_local_edits",
            "local_evaluation_mode": PAIRED_LOCAL_EVALUATION_MODE,
            "batch_count": trajectory.batch_count,
            "local_edit_decision_count": local_edit_count,
            "included_batch_count": len(included_local_edits),
            "truncated_batch_count": max(
                0, local_edit_count - len(included_local_edits)
            ),
            "comparison_count": len(batch_rows),
            "included_comparison_count": len(included_comparisons),
            "truncated_comparison_count": max(
                0, len(batch_rows) - len(included_comparisons)
            ),
            "initial_revision_id": trajectory.initial_revision_id,
            "final_revision_id": final_revision_id,
            "revision_chain": revision_chain[
                -(_ADAPTIVE_REFLECTION_BATCH_LIMIT + 1) :
            ],
            "outcome_counts": outcome_counts,
            "operation_category_counts": operation_category_counts,
            "batches": included_local_edits,
            "recent_comparisons": included_comparisons,
        }

    batch_rows = []
    revision_chain = [trajectory.initial_revision_id]
    formal_by_index = {item.batch_index: item for item in formal_batches}
    for batch_index in range(trajectory.batch_count):
        local = local_rows[batch_index]
        batch_rows.append(
            {
                "batch_index": local["batch_index"],
                "batch_number": local["batch_number"],
                "evaluated_revision_id": formal_by_index[batch_index].revision_id,
                "decision": local["decision"],
                "operation_category": local["operation_category"],
                "operation_count": local["operation_count"],
                "outcome": local["outcome"],
                "outcome_reason": local["outcome_reason"],
                "from_revision_id": local["mutation_parent_revision_id"],
                "active_revision_id": local["generated_challenger_revision_id"],
                "advance_reason": local["advance_reason"],
            }
        )
        revision_chain.append(local["generated_challenger_revision_id"])
    included = batch_rows[:_ADAPTIVE_REFLECTION_BATCH_LIMIT]
    return {
        "kind": "batch_local_edits",
        "batch_count": trajectory.batch_count,
        "included_batch_count": len(included),
        "truncated_batch_count": max(0, len(batch_rows) - len(included)),
        "initial_revision_id": trajectory.initial_revision_id,
        "final_revision_id": final_revision_id,
        "revision_chain": revision_chain[: _ADAPTIVE_REFLECTION_BATCH_LIMIT + 1],
        "outcome_counts": outcome_counts,
        "operation_category_counts": operation_category_counts,
        "batches": included,
    }


def _build_adaptive_analysis(
    state: Any,
    generation: int,
    comparison: GenerationComparison,
    finalists: tuple[Candidate, ...],
    incumbent_candidate_id: str,
) -> Any:
    """Create one gate-derived analysis with bounded adaptive lineage evidence."""

    formal_selection_getter = getattr(state, "formal_selection_for", None)
    formal_selection = (
        formal_selection_getter(generation)
        if callable(formal_selection_getter)
        else None
    )
    formal_payload = getattr(formal_selection, "payload", {})
    if not isinstance(formal_payload, Mapping):
        formal_payload = {}
    exploration_only = formal_payload.get("exploration_only") is True
    raw_consecutive = formal_payload.get("consecutive_exploration_generations", 0)
    consecutive_exploration = (
        raw_consecutive
        if isinstance(raw_consecutive, int) and not isinstance(raw_consecutive, bool)
        else 0
    )
    force_replan = exploration_only and consecutive_exploration >= 2

    finalist_evaluations = {
        item.scope.candidate_id: item
        for item in comparison.holdout_evaluations
        if item.scope.holdout_arm in {HoldoutArm.FINALIST_1, HoldoutArm.FINALIST_2}
    }
    incumbent_evaluation = next(
        item
        for item in comparison.holdout_evaluations
        if item.scope.holdout_arm is HoldoutArm.INCUMBENT
    )
    gate_results = comparison.gate_results
    positive_delta_search = (
        gate_results.get("selection_policy") == "positive_delta_search@1"
    )
    arms = gate_results.get("arms")
    if not isinstance(arms, Mapping):
        raise RuntimeError("adaptive comparison is missing arm gate results")
    selected_arm = gate_results.get("selected_arm")
    if selected_arm not in {arm.value for arm in HoldoutArm}:
        raise RuntimeError("adaptive comparison selected arm is invalid")
    selected_evaluation = next(
        item
        for item in comparison.holdout_evaluations
        if item.scope.holdout_arm.value == selected_arm
    )
    if (
        comparison.selected_candidate_id != selected_evaluation.scope.candidate_id
        or comparison.selected_revision_id
        != selected_evaluation.scope.candidate_revision_id
    ):
        raise RuntimeError("adaptive comparison selection binding is inconsistent")
    selected = (
        comparison.selected_candidate_id
        if selected_arm
        in {HoldoutArm.FINALIST_1.value, HoldoutArm.FINALIST_2.value}
        else None
    )
    certification_selected_arm = gate_results.get("certification_selected_arm")
    certification_selected = next(
        (
            item.scope.candidate_id
            for item in comparison.holdout_evaluations
            if item.scope.holdout_arm is not None
            and item.scope.holdout_arm.value == certification_selected_arm
            and item.scope.holdout_arm
            in {HoldoutArm.FINALIST_1, HoldoutArm.FINALIST_2}
        ),
        None,
    )
    finalist_gate_by_candidate: dict[str, Mapping[str, Any]] = {}
    for candidate_id, evaluation in finalist_evaluations.items():
        gate = arms.get(evaluation.scope.holdout_arm.value)
        if not isinstance(gate, Mapping):
            raise RuntimeError("adaptive finalist is missing its comparison gate")
        finalist_gate_by_candidate[candidate_id] = gate
    eligible_finalist_ids = [
        candidate_id
        for candidate_id, gate in finalist_gate_by_candidate.items()
        if gate.get("eligible") is True
    ]
    if selected is not None:
        if selected not in eligible_finalist_ids:
            raise RuntimeError("adaptive comparison selected an ineligible finalist")
    elif eligible_finalist_ids:
        raise RuntimeError(
            "adaptive comparison retained incumbent despite eligible finalist"
        )
    rank_by_candidate: dict[str, int] = {}
    if selected is not None:
        rank_by_candidate[selected] = 1
        for candidate_id in sorted(
            (item for item in eligible_finalist_ids if item != selected),
            key=lambda item: (
                next(
                    candidate.slot_index
                    for candidate in finalists
                    if candidate.candidate_id == item
                ),
                item,
            ),
        ):
            rank_by_candidate[candidate_id] = len(rank_by_candidate) + 1
    ranking: list[dict[str, Any]] = []
    screening = _screening_records(state, generation)
    generation_candidates = sorted(
        (
            item
            for item in state.candidates
            if item.generation == generation and _is_search_candidate(item)
        ),
        key=lambda item: (item.slot_index, item.candidate_id),
    )
    for candidate in generation_candidates:
        holdout = finalist_evaluations.get(candidate.candidate_id)
        if holdout is not None:
            arm = holdout.scope.holdout_arm
            if arm is None:  # pragma: no cover - enforced by EvaluationScope
                raise RuntimeError("adaptive finalist holdout arm is missing")
            gate = finalist_gate_by_candidate[candidate.candidate_id]
            eligible = gate.get("eligible") is True
            certification_eligible = gate.get("certification_eligible") is True
            is_selected = candidate.candidate_id == selected
            failures = _adaptive_gate_failures(gate)
            if is_selected:
                selection_status = "selected"
                selection_reason = (
                    "next_round_search_version"
                    if positive_delta_search
                    else "generation_holdout_winner"
                )
                classification = "selected"
            elif eligible:
                selection_status = "eligible_not_selected"
                selection_reason = "lower_deterministic_holdout_rank"
                classification = "eligible_not_selected"
            else:
                selection_status = "holdout_gate_failed"
                selection_reason = (
                    "holdout_gate_failed:" + ",".join(failures[:4])
                )[:240]
                classification = "holdout_gate_failed"
            revision = state.revision(holdout.scope.candidate_revision_id)
            if revision.candidate_id != candidate.candidate_id:
                raise RuntimeError(
                    "adaptive finalist revision belongs to another candidate"
                )
            assessment = gate.get("promotion_assessment")
            ranking.append(
                {
                    "rank": rank_by_candidate.get(candidate.candidate_id),
                    "candidate_id": candidate.candidate_id,
                    "slot_index": candidate.slot_index,
                    "score": holdout.score,
                    "eligible": eligible,
                    "search_eligible": gate.get("search_eligible", eligible) is True,
                    "certification_eligible": certification_eligible,
                    "scientific_pass": gate.get("passed") is True,
                    "constraint_violations": int(
                        gate.get("constraint_violations") or 0
                    ),
                    "classification": classification,
                    "primary_selection_gate": bool(
                        isinstance(assessment, Mapping)
                        and assessment.get("primary_selection_gate") is True
                    ),
                    "selection_status": selection_status,
                    "selection_reason": selection_reason,
                    "judge_available": gate.get("judge_available") is True,
                    "judge_accepted": gate.get("judge_accepted") is True,
                    "holdout_arm": arm.value,
                    "delta_to_incumbent": holdout.score - incumbent_evaluation.score,
                    "failure_reasons": failures,
                    "final_revision_id": revision.revision_id,
                    "final_revision_digest": revision.revision_digest,
                    "final_genome_digest": revision.genome_digest,
                    "final_behavior_digest": revision.behavior_digest,
                    "comparison_gate": _bounded_comparison_gate(gate),
                    "outer_mutation_evidence": _outer_mutation_evidence(
                        state, candidate
                    ),
                    "local_edit_evidence": _local_edit_trajectory_evidence(
                        state,
                        candidate.candidate_id,
                        revision.revision_id,
                    ),
                }
            )
        else:
            record = screening.get(candidate.candidate_id, {})
            screening_passed = record.get("passed") is True
            screening_constraints = int(record.get("constraint_violations", 0))
            screening_gate_passed = bool(
                screening_passed and screening_constraints == 0
            )
            screening_failures = []
            if not screening_passed:
                screening_failures.append("screening_scientific_gate_failed")
            if screening_constraints > 0:
                screening_failures.append("screening_constraint_violations")
            ranking.append(
                {
                    "rank": None,
                    "candidate_id": candidate.candidate_id,
                    "slot_index": candidate.slot_index,
                    "score": record.get("score"),
                    "eligible": False,
                    "scientific_pass": screening_passed,
                    "constraint_violations": screening_constraints,
                    "classification": (
                        "screened_out_lower_rank"
                        if screening_gate_passed
                        else "screening_gate_failed"
                    ),
                    "primary_selection_gate": False,
                    "selection_status": "screened_out",
                    "selection_reason": (
                        "not_selected_by_screening_top_k"
                        if screening_gate_passed
                        else (
                            "screening_gate_failed:"
                            + ",".join(screening_failures)
                        )[:240]
                    ),
                    "judge_available": False,
                    "judge_accepted": False,
                    "failure_reasons": (
                        ["screened_out_before_adaptive_epoch"]
                        if screening_gate_passed
                        else screening_failures
                    ),
                    "outer_mutation_evidence": _outer_mutation_evidence(
                        state, candidate
                    ),
                    "local_edit_evidence": {
                        "kind": "not_run_screened_out",
                        "batch_count": 0,
                    },
                }
            )
    ranking.sort(
        key=lambda row: (
            row["rank"] is None,
            row["rank"] if row["rank"] is not None else 0,
            row["slot_index"],
            row["candidate_id"],
        )
    )
    # Replanning the research direction and advancing the next search parent
    # are independent decisions in runtime v3.  A positive-delta finalist may
    # become the parent while two consecutive exploration-only generations
    # still require new queries, hypotheses, and focus areas.
    force_replan = bool(force_replan)
    outcome = (
        "promoted"
        if certification_selected is not None
        else "search_version_advanced"
        if positive_delta_search and selected is not None
        else "exploration_only"
        if exploration_only
        else "no_improvement"
    )
    return GenerationAnalysis(
        run_id=state.run.run_id,
        generation=generation,
        candidate_count=len(generation_candidates),
        eligible_count=len(eligible_finalist_ids),
        outcome=outcome,
        selected_candidate_id=selected,
        champion_candidate_id=(
            certification_selected if positive_delta_search else selected
        ),
        incumbent_before_candidate_id=incumbent_candidate_id,
        incumbent_after_candidate_id=(
            certification_selected or incumbent_candidate_id
            if positive_delta_search
            else selected or incumbent_candidate_id
        ),
        search_parent_candidate_id=selected or incumbent_candidate_id,
        ranking=tuple(ranking),
        next_search_direction=(
            "连续探索轮未通过初筛，强制重新规划搜索方向与假设"
            if force_replan
            else "保留严格认证版本，并把本轮正增益版本作为下一轮搜索起点"
            if positive_delta_search and selected
            else "保留全局 incumbent，并把本轮结果仅作为探索证据"
            if exploration_only
            else "围绕留出比较选中候选的最终 revision 继续有界局部搜索"
            if selected
            else (
                "保留 incumbent 的最终 revision，针对 finalist "
                "的留出门禁失败开展下一轮有界搜索"
            ),
        ),
        next_generation_focus=(
            "重新生成候选方向，避免继续沿用连续失败的局部搜索路径"
            if force_replan
            else "围绕正增益搜索版本继续优化，并单独修复认证风险"
            if positive_delta_search and selected
            else "修复初筛失败原因后再寻求全局晋升"
            if exploration_only
            else (
                f"围绕本轮 {len(finalist_evaluations)} 个 finalist 的留出比较结果，"
                "继续批次局部更新"
            )
            if selected
            else (
                f"修复本轮 {len(finalist_evaluations)} 个 finalist 的留出门禁失败"
            )
        ),
        selection_reason=(
            (
                f"本代虽无候选通过初筛，候选 {selected} 在同一冻结 "
                "holdout 上取得正增益，成为下一轮搜索版本；"
                "严格认证版本不变，且下一轮必须重新规划搜索方向。"
            )
            if positive_delta_search and exploration_only and selected
            else (
                f"候选 {selected} 在同一冻结 holdout 上取得正增益，"
                "成为下一轮搜索版本；严格认证状态单独保留。"
            )
            if positive_delta_search and selected
            else "本代无候选通过初筛，仅保留探索证据；全局 incumbent 不变。"
            if exploration_only
            else (
                f"候选 {selected} 在本轮冻结的 "
                f"{len(comparison.holdout_evaluations)} 臂留出比较中胜出。"
            )
            if selected
            # The finalist count is scheduled, not fixed: `quick_adaptive_epoch@1`
            # runs a single finalist, so a hardcoded "两个 finalist" misreported
            # every failed quick generation. Report what this epoch actually ran.
            else (
                f"本轮 {len(finalist_evaluations)} 个 finalist "
                "均未通过留出集门禁，保留 incumbent。"
            )
        ),
        insufficient_evidence=False,
        replan_required=force_replan,
        consecutive_exploration_generations=(
            consecutive_exploration if exploration_only else 0
        ),
    )




def _trajectory_holdout_revision_id(
    state: Any,
    trajectory: Any,
    schedule: OptimizationSchedule,
) -> str:
    """Resolve a finalist holdout revision from completed durable evidence."""

    final_revision_id = trajectory.final_revision_id
    if (
        trajectory.status is not TrajectoryStatus.COMPLETED
        or not isinstance(final_revision_id, str)
        or not final_revision_id
    ):
        raise ValueError("holdout requires a completed trajectory final revision")
    final_comparison = state.batch_comparison_for(
        trajectory.candidate_id,
        trajectory.batch_count - 1,
    )
    if (
        final_comparison is None
        or final_revision_id
        != final_comparison.champion_after_revision_id
    ):
        raise ValueError(
            "paired trajectory final revision is not the durable champion"
        )
    return final_revision_id


def _ensure_finalist_reviews(services, run_id, evaluations):
    """Finish durable independent reviews before deriving any new final decision."""
    from ..core.finalist_review import finalist_review_evidence, requires_finalist_review

    state = services.director.state(run_id)
    if not requires_finalist_review(state.task_manifest.metadata):
        return None
    incumbent = next((item for item in evaluations if item.scope.holdout_arm is HoldoutArm.INCUMBENT), None)
    if incumbent is None:
        raise RuntimeError("finalist review requires the sealed same-cohort incumbent")
    training_effects = finalist_training_effect_evidence(evaluations,
        screening_events=getattr(state, "candidate_screening_events", ()),
        batch_evaluations=getattr(state, "formal_batch_evaluations", ()))
    for holdout in evaluations:
        if holdout.scope.holdout_arm is HoldoutArm.INCUMBENT:
            continue
        state = services.director.state(run_id)
        if state.run.status is not RunStatus.RUNNING:
            raise RuntimeError("finalist review interrupted by run control")
        candidate = state.candidate(holdout.scope.candidate_id)
        canonical = state.evaluation_for(candidate.candidate_id)
        artifact = state.artifact_for(candidate.candidate_id)
        if canonical is None or artifact is None:
            raise RuntimeError("finalist review requires its sealed scientific artifact")
        if canonical.metrics.get("judge_status") not in {"completed", "unavailable"}:
            _apply_candidate_judge(services, state, state.proposal(candidate.proposal_id),
                                   artifact, canonical, incumbent=incumbent,
                                   training_effect_evidence=training_effects)
    state = services.director.state(run_id)
    return finalist_review_evidence(evaluations,
        {item.candidate_id: item for item in state.evaluations},
        state.task_manifest.metadata["review_model_id"])


def _finalize_adaptive_generation(services: GenerationRuntime, run_id: str, batch: Any) -> Any:
    state = services.director.state(run_id)
    generation = batch.generation
    schedule = OptimizationSchedule.from_dict(
        state.task_manifest.metadata["optimization_schedule"]
    )
    formal = state.formal_selection_for(generation)
    if formal is None:
        raise RuntimeError("adaptive generation is missing Top-2 formal selection")
    finalist_ids = tuple(formal.payload["selected_candidate_ids"])
    finalists = tuple(state.candidate(item) for item in finalist_ids)
    trajectories = tuple(state.trajectory_for(item) for item in finalist_ids)
    if any(item is None or item.status is not TrajectoryStatus.COMPLETED for item in trajectories):
        return None
    finalist_revision_ids = tuple(
        _trajectory_holdout_revision_id(state, item, schedule)
        for item in trajectories
        if item is not None
    )
    cohorts = state.generation_cohort_for(generation)
    if cohorts is None:
        raise RuntimeError("adaptive generation is missing frozen holdout cohort")

    incumbent_revision_id = (
        state.effective_revision_for(generation - 1) if generation > 0 else None
    )
    incumbent_id = (
        state.revision(incumbent_revision_id).candidate_id
        if incumbent_revision_id is not None
        else state.run.best_candidate_id
    )
    if generation == 0 and incumbent_revision_id is None:
        seed_control = next(
            (
                item
                for item in state.candidates
                if getattr(item, "role", CandidateRole.SEARCH)
                is CandidateRole.INCUMBENT_CONTROL
            ),
            None,
        )
        if seed_control is not None:
            seed_revision = state.initial_revision_for(seed_control.candidate_id)
            if seed_revision is None:
                raise RuntimeError("adaptive seed incumbent is missing R0")
            incumbent_id = seed_control.candidate_id
            incumbent_revision_id = seed_revision.revision_id
        elif uses_global_incumbent_protocol(state.task_manifest):
            raise RuntimeError(
                "runtime-v2 generation zero requires the materialized seed incumbent"
            )
    if incumbent_id is None or incumbent_revision_id is None:
        # Legacy prequential runs predate the explicit seed control. Preserve
        # their historical audit/recovery semantics without silently applying
        # this fallback to new paired runs.
        fallback = next(
            (
                item
                for item in state.candidates
                if item.generation == generation
                and _is_search_candidate(item)
                and item.candidate_id not in finalist_ids
            ),
            None,
        )
        if fallback is None:
            raise RuntimeError("adaptive holdout requires a distinct incumbent arm")
        incumbent_id = fallback.candidate_id
        incumbent_revision = state.initial_revision_for(incumbent_id)
        if incumbent_revision is None:
            raise RuntimeError("adaptive incumbent is missing R0")
        incumbent_revision_id = incumbent_revision.revision_id
    bindings = {
        HoldoutArm.FINALIST_1.value: {
            "candidate_id": finalist_ids[0],
            "candidate_revision_id": finalist_revision_ids[0],
        },
        HoldoutArm.INCUMBENT.value: {
            "candidate_id": incumbent_id,
            "candidate_revision_id": incumbent_revision_id,
        },
    }
    pass
    holdout = _director_mutation(
        services,
        "freeze_generation_holdout",
        run_id,
        generation,
        cohorts.holdout.cohort_digest,
        bindings,
    )
    state = services.director.state(run_id)
    for arm in map(HoldoutArm, holdout.arm_bindings):
        _execute_adaptive_holdout_arm(
            services,
            run_id,
            generation,
            arm,
            holdout.arm_bindings[arm.value],
            cohorts.holdout,
        )
    state = services.director.state(run_id)
    evaluations = tuple(
        state.holdout_evaluation_for(generation, arm)
        for arm in map(HoldoutArm, holdout.arm_bindings)
    )
    if any(item is None for item in evaluations):
        return None
    comparison = state.comparison_for(generation)
    if comparison is None:
        reviews = _ensure_finalist_reviews(services, run_id, evaluations)
        state = services.director.state(run_id)
        comparison_inputs = dict(
            quick_experiment=schedule.quick,
            experiment_protocol=schedule.protocol,
            training_effect_evidence=finalist_training_effect_evidence(evaluations,
                screening_events=getattr(state, "candidate_screening_events", ()),
                batch_evaluations=getattr(state, "formal_batch_evaluations", ())),
            finalist_reviews=reviews,
            run_id=run_id,
            generation=generation,
            cohort_digest=holdout.cohort_digest,
            holdout_evaluations=tuple(item for item in evaluations if item is not None),
            incumbent_candidate_id=incumbent_id,
            fitness_profile=FitnessProfile.from_task(state.task_manifest),
            challenger_promotion_allowed=(
                formal.payload.get("exploration_only") is not True
            ),
            positive_delta_search=uses_positive_delta_search_protocol(
                state.task_manifest
            ),
            require_paired_strict_chain=guarded_search(state.task_manifest.metadata),
        )
        comparison = build_generation_comparison(**comparison_inputs)
        recheck = comparison.gate_results["arms"]["finalist_1"]["joint_recheck"]
        if recheck["required"]:
            repeats = tuple(_execute_joint_holdout_recheck(services, run_id, item, cohorts.holdout)
                            for item in evaluations)
            comparison = build_generation_comparison(**comparison_inputs, holdout_rechecks=repeats)
        _director_mutation(services, "record_generation_comparison", run_id, comparison)
    _director_mutation(
        services,
        "select_generation_champion",
        run_id,
        generation,
        comparison.selected_revision_id,
        comparison.comparison_digest,
    )
    state = services.director.state(run_id)
    analysis = state.analysis_for(generation)
    if analysis is None:
        analysis = _build_adaptive_analysis(
            state, generation, comparison, finalists, incumbent_id
        )
        services.ledger.append(
            run_id,
            "GenerationAnalyzed",
            {"analysis": analysis.to_dict()},
            event_id=f"{run_id}:generation:{generation}:analyzed",
        )
    state = services.director.state(run_id)
    for finalist in finalists:
        if state.promotion_for(finalist.candidate_id) is not None:
            continue
        approved = finalist.candidate_id == comparison.selected_candidate_id
        reason = analysis.selection_reason
        # ``record_evaluation`` above makes the finalists eligible for the
        # existing DSH identity/artifact promotion fence.
        _director_mutation(
            services,
            "decide_promotion",
            Promotion(
                promotion_id=f"promotion:{run_id}:{generation}:{finalist.candidate_id}",
                run_id=run_id,
                candidate_id=finalist.candidate_id,
                decision=(PromotionDecision.APPROVED if approved else PromotionDecision.REJECTED),
                reason=reason,
            ),
        )
    if state.task_manifest.metadata.get("autonomous_research_protocol"):
        from ..evolution.batches import _ensure_generation_reflection

        _ensure_generation_reflection(
            services.director,
            services.director.state(run_id),
            batch,
            analysis,
        )
    return analysis


def execute_generation(services: GenerationRuntime, run_id: str) -> Any:
    """Complete exactly one generation including its unified decision barrier."""

    state = complete_if_budget_exhausted(services, run_id)
    if state.run.status is not RunStatus.RUNNING:
        return state
    batch = start_generation_batch(services.director, run_id)
    try:
        spawned = _spawn_generation_candidates(services, run_id, batch)
        if not spawned:
            return services.director.state(run_id)
    except Exception as exc:
        # Control commands are intentionally allowed while a remote proposal is
        # waiting.  If pause/cancel wins that race, preserve the partially
        # written batch for resume instead of converting the control transition
        # into a terminal execution failure.
        latest = services.director.state(run_id)
        if latest.run.status is not RunStatus.RUNNING:
            return latest
        dsh_error = dsh_native_runtime_error_in_chain(exc)
        if dsh_error is not None and dsh_native_runtime_retryable(dsh_error):
            _record_stage(
                services,
                run_id,
                batch.generation,
                "proposal",
                "failed",
                public_error=public_exception_summary(exc),
            )
            raise
        _record_stage(
            services,
            run_id,
            batch.generation,
            "proposal",
            "failed",
            public_error=public_exception_summary(exc),
            event_id=f"{run_id}:stage:{batch.generation}:batch-generation:failed",
        )
        state = services.director.state(run_id)
        _quiesce_native_terminal(services, state, "proposal_failed")
        _director_mutation(
            services,
            "fail_run",
            run_id, f"候选批次生成失败：{public_exception_summary(exc)}"
        )
        return services.director.state(run_id)

    state = services.director.state(run_id)
    current = sorted(
        (
            item
            for item in state.candidates
            if item.generation == batch.generation and _is_search_candidate(item)
        ),
        key=lambda item: item.slot_index,
    )
    _freeze_adaptive_generation_inputs(services, run_id, batch.generation)
    _evaluate_generation_candidates(services, run_id, current)
    latest = services.director.state(run_id)
    if latest.run.status is not RunStatus.RUNNING:
        return latest

    # Adaptive Top-2 generations have their own final barrier: both finalist
    # lanes must finish their frozen formal schedule and expose a durable final
    # champion, then all three arms are scored on one fresh holdout cohort
    # before any global promotion or generation advance.
    if _two_stage_screening_enabled(latest, current):
        analysis = _finalize_adaptive_generation(services, run_id, batch)
        latest = services.director.state(run_id)
        if latest.run.status is not RunStatus.RUNNING or analysis is None:
            return latest
        services.director.advance_generation(run_id)
        return complete_if_budget_exhausted(services, run_id)

    raise RuntimeError("joint_plugin_epoch@2 requires the frozen two-stage schedule")


__all__ = [
    "_ensure_candidate_algorithm_ready",
    "complete_if_budget_exhausted",
    "execute_generation",
]
