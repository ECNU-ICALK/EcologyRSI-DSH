"""The minimal trusted evolution state machine."""

from __future__ import annotations

import math
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any
from uuid import uuid4

from ..evolution.analysis import (
    GenerationBatch,
    evaluation_cohort_comparison,
    evaluation_cohort_digest,
    sample_update_windows_enabled,
)
from ..evolution.execution_plan import derive_execution_plan
from ..evolution.genome import (
    FrozenRunInitialization,
    SeedGenomeTemplate,
    materialize_seed_genome,
)
from ..evolution.workflow_ir import (
    DEFAULT_COMPILER_SEMANTIC_DIGEST,
    CompilationInstanceContext,
    bind_phenotype_instance,
    compile_plugin_behavior,
)
from ..evolution.promotion import assess_promotion_improvement
from ..evolution.strategies import (
    DSHAdapter,
    FakeDSHAdapter,
    apply_bounded_interventions,
)
from ..knowledge.algorithms import (
    AlgorithmAttempt,
    PredictorAdoption,
    resolve_predictor_adoption,
)
from ..knowledge.research_iteration import ResearchIteration
from ..knowledge.program_registry import current_program_registry
from .ledger import ConcurrentRunMutationError, Event, EventLedger
from .models import (
    Candidate,
    CandidateStatus,
    Evaluation,
    ExpertConsultation,
    ExpertConsultationAnswer,
    HumanIntervention,
    InterventionKind,
    ModelArtifact,
    Promotion,
    PromotionDecision,
    Proposal,
    Run,
    RunStatus,
    TaskManifest,
    canonical_json,
    digest,
)
from .sample_results import (
    MAX_SAMPLE_RESULTS_RECORDS,
    decode_sample_result_batch,
    decode_sample_results,
    sample_results_cohort_digest,
    sample_results_event_id,
)
from .exposure_registry import (
    FormalStageToken,
    ScientificExposureRegistry,
    raw_holdout_exposure_key,
)
from .state import (
    DSH_NATIVE_EVOLUTION_PROTOCOL,
    RunState,
    is_dsh_native_protocol,
    persisted_genome_from_proposal,
    project_run_state,
    validate_evaluation_progress_payload,
    validate_evolution_stage_payload,
    validate_model_usage_payload,
)

_AGGREGATE_EVALUATION_METRICS = frozenset(
    {
        "baseline_normalized_rmse",
        "causal_interpretation",
        "constraint_violations",
        "improvement",
        "mae",
        "missing_or_nonfinite_rows",
        "n",
        "non_negative_state",
        "normalized_rmse",
        "rmse",
        "scientific_pass",
        "score",
        "skill_score",
        "water_balance_error",
    }
)

_SAMPLE_CHECKPOINT_SCHEMA_VERSION = "ecologyrsi-dsh.sample-checkpoint/1"
_SAMPLE_RESULTS_START_SCHEMA_VERSION = (
    "ecologyrsi-dsh.evaluation-sample-results-start/2"
)
_SAMPLE_RESULTS_RESUME_SCHEMA_VERSION = (
    "ecologyrsi-dsh.evaluation-sample-results-resume/1"
)
_TARGET_EVALUATION_METRICS = frozenset(
    {
        "baseline_mae",
        "baseline_normalized_rmse",
        "baseline_rmse",
        "constraint_violations",
        "eligible_rows",
        "mae",
        "missing_or_nonfinite_rows",
        "n",
        "normalized_rmse",
        "rmse",
        "target",
        "unit",
    }
)


def _frozen_runtime_binding(state: RunState) -> dict[str, Any] | None:
    """Recover the first host-validated autonomous binding from the ledger."""

    for proposal in sorted(
        state.proposals,
        key=lambda item: (item.generation, item.created_at, item.proposal_id),
    ):
        plan = proposal.metadata.get("plan")
        adoption = proposal.metadata.get("prediction_model_adoption")
        if not isinstance(plan, Mapping) or not isinstance(adoption, Mapping):
            continue
        return {
            "source_proposal_id": proposal.proposal_id,
            "source_proposal_digest": proposal.digest,
            "plan": dict(plan),
            "prediction_model_adoption": dict(adoption),
        }
    return None


def _task_for_proposal_predictor(
    task: TaskManifest,
    proposal: Proposal,
) -> TaskManifest:
    raw_plan = proposal.metadata.get("plan")
    raw_adoption = proposal.metadata.get("prediction_model_adoption")
    genome = persisted_genome_from_proposal(proposal)
    if genome is not None:
        if raw_plan is not None or raw_adoption is not None:
            raise ValueError("DSH-native proposal cannot mix genome and legacy predictor adoption")
        predictor_ref = genome.scientific_program["predictor_ref"]
        predictor_id = str(predictor_ref["id"])
        expected_ref = current_program_registry().program_ref(
            "predictors", predictor_id
        )
        if dict(predictor_ref) != expected_ref:
            raise ValueError("proposal genome predictor is outside the frozen Host registry")
        data = task.to_dict()
        data["metadata"] = {
            **dict(task.metadata),
            "prediction_model_id": predictor_id,
            "prediction_model_digest": expected_ref["catalog_digest"],
        }
        return TaskManifest.from_dict(data)
    if raw_plan is None and raw_adoption is None:
        return task
    if not isinstance(raw_plan, Mapping) or not isinstance(raw_adoption, Mapping):
        raise TypeError("proposal predictor adoption is incomplete")
    adoption = PredictorAdoption.from_dict(raw_adoption)
    expected = resolve_predictor_adoption(task, raw_plan)
    if adoption.to_dict() != expected.to_dict():
        raise ValueError("proposal predictor adoption does not match the host catalog")
    data = task.to_dict()
    data["metadata"] = {
        **dict(task.metadata),
        "prediction_model_id": adoption.adopted_id,
        "prediction_model_digest": adoption.adopted_digest,
    }
    return TaskManifest.from_dict(data)


_INCUMBENT_SCORE_TOLERANCE = 1e-12
_STATE_TRANSITION_RETRY_LIMIT = 16


def _bounded_summary_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:4000]


def _numeric_summary(value: Any, name: str) -> dict[str, int | float | bool]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if len(value) > 128:
        raise ValueError(f"{name} contains too many fields")
    result: dict[str, int | float | bool] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} contains an invalid field name")
        if isinstance(item, bool) or (
            isinstance(item, (int, float)) and math.isfinite(float(item))
        ):
            result[key] = item
    return result


def _evaluation_metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _AGGREGATE_EVALUATION_METRICS:
        value = metrics.get(key)
        if isinstance(value, bool) or (
            isinstance(value, (int, float)) and math.isfinite(float(value))
        ):
            result[key] = value
    raw_targets = metrics.get("targets")
    targets: list[dict[str, Any]] = []
    if isinstance(raw_targets, list):
        for raw_target in raw_targets[:32]:
            if not isinstance(raw_target, Mapping):
                continue
            target: dict[str, Any] = {}
            for key in _TARGET_EVALUATION_METRICS:
                value = raw_target.get(key)
                if isinstance(value, bool) or (
                    isinstance(value, (int, float)) and math.isfinite(float(value))
                ):
                    target[key] = value
                elif key in {"target", "unit"} and isinstance(value, str):
                    target[key] = value[:200]
            if target:
                targets.append(target)
    if targets:
        result["targets"] = targets
    return result


class EvolutionDirector:
    """Coordinate one small evolution run and persist every transition.

    The director is intentionally synchronous.  A UI can call these methods
    from a background job today; a queue or remote executor can be introduced
    behind the same methods later without changing the event contract.
    """

    def __init__(self, ledger: EventLedger, dsh: DSHAdapter | None = None) -> None:
        self.ledger = ledger
        self.dsh: DSHAdapter = dsh or FakeDSHAdapter()

    def _append_run_transition(
        self,
        run_id: str,
        kind: str,
        payload_factory: Any,
        *allowed: RunStatus,
    ) -> RunState:
        """Append a lifecycle event only if its replayed state is still current."""

        last_conflict: ConcurrentRunMutationError | None = None
        for _ in range(_STATE_TRANSITION_RETRY_LIMIT):
            state = self.state(run_id)
            self._require_status(state.run, *allowed)
            payload = payload_factory(state)
            try:
                self.ledger.append(
                    run_id,
                    kind,
                    payload,
                    expected_run_seq=state.events[-1].seq,
                )
            except ConcurrentRunMutationError as exc:
                last_conflict = exc
                continue
            return state
        if last_conflict is not None:
            raise last_conflict
        raise RuntimeError(f"run {run_id} transition could not be persisted")

    def prepare_run_creation(
        self, task_manifest: TaskManifest, *, run_id: str | None = None
    ) -> tuple[Run, dict[str, Any]]:
        """Purely prepare RunCreated, including a frozen expected seed when needed."""

        if not isinstance(task_manifest, TaskManifest):
            raise TypeError("task_manifest must be a TaskManifest")
        run_id = run_id or f"run:{uuid4()}"
        run = Run(
            run_id=run_id,
            task_id=task_manifest.task_id,
            task_manifest_digest=task_manifest.digest,
        )
        payload: dict[str, Any] = {
            "run": run.to_dict(),
            "task_manifest": task_manifest.to_dict(),
        }
        if is_dsh_native_protocol(task_manifest):
            metadata = task_manifest.metadata
            registry = current_program_registry()
            template_id = str(
                metadata.get("seed_genome_template_id") or ""
            ).strip()
            if not template_id:
                raise ValueError("DSH-native run requires seed_genome_template_id")
            template = registry.seed_template(template_id)
            initialization = FrozenRunInitialization(
                run_id=run_id,
                task_manifest_digest=task_manifest.digest,
                dataset_snapshot_set_digest=metadata.get(
                    "dataset_snapshot_set_digest"
                ),
                split_manifest_digest=metadata.get("split_manifest_digest"),
                data_protocol_digest=metadata.get("data_protocol_digest"),
                stage_policy_digest=metadata.get("stage_policy_digest"),
                evaluator_digest=metadata.get("evaluator_digest"),
                fitness_profile_digest=metadata.get("fitness_profile_digest"),
                security_kernel_digest=metadata.get("security_kernel_digest"),
                selection_reviewer_program_digest=metadata.get(
                    "selection_reviewer_program_digest"
                ),
                protocol=DSH_NATIVE_EVOLUTION_PROTOCOL,
                required_capability_digest=metadata.get(
                    "required_capability_digest"
                ),
                resolved_policy_route_digest=metadata.get(
                    "resolved_policy_route_digest"
                ),
                resolved_review_route_digest=metadata.get(
                    "resolved_review_route_digest"
                ),
                registry_catalog_digest=registry.catalog_digest,
                compiler_digest=str(
                    metadata.get("compiler_semantic_digest")
                    or DEFAULT_COMPILER_SEMANTIC_DIGEST
                ),
            )
            seed = materialize_seed_genome(template, initialization)
            payload["genome_initialization"] = {
                "schema_version": "ecologyrsi-dsh.run-genome-initialization/1",
                "materializer_version": "seed-genome-materializer@1",
                "seed_template_canonical_json": canonical_json(template.to_dict()),
                "seed_template_digest": template.template_digest,
                "materialization_input": initialization.to_dict(),
                "expected_seed_canonical_json": canonical_json(seed.to_dict()),
                "expected_seed_genome_digest": seed.genome_digest,
            }
        return run, payload

    def create_run(self, task_manifest: TaskManifest, *, run_id: str | None = None) -> Run:
        run_id = run_id or f"run:{uuid4()}"
        if self.ledger.events(run_id):
            raise ValueError(f"run already exists: {run_id}")
        run, payload = self.prepare_run_creation(task_manifest, run_id=run_id)
        self.ledger.append(
            run_id,
            "RunCreated",
            payload,
            event_id=f"{run_id}:created",
        )
        if is_dsh_native_protocol(task_manifest):
            self.recover_run_initialization(run_id)
        return run

    def recover_run_initialization(self, run_id: str):
        """Idempotently finish seed materialization using only RunCreated payload."""

        events = self.ledger.events(run_id)
        if not events or events[0].kind != "RunCreated":
            raise KeyError(f"unknown run: {run_id}")
        created = events[0]
        task = TaskManifest.from_dict(created.payload["task_manifest"])
        if not is_dsh_native_protocol(task):
            raise ValueError("historical run has no DSH-native initialization")
        existing = next(
            (event for event in events if event.kind == "RunSeedGenomeMaterialized"),
            None,
        )
        if existing is not None:
            return self.state(run_id).materialized_seed_genome()
        initialization = created.payload.get("genome_initialization")
        if not isinstance(initialization, Mapping):
            raise ValueError("RunCreated is missing genome initialization")
        canonical = initialization.get("expected_seed_canonical_json")
        if not isinstance(canonical, str):
            raise ValueError("RunCreated expected seed canonical JSON is invalid")
        from ..evolution.genome import EcologyEvolutionPluginGenome

        seed = EcologyEvolutionPluginGenome.from_dict(json.loads(canonical))
        if canonical_json(seed.to_dict()) != canonical:
            raise ValueError("RunCreated expected seed JSON is not canonical")
        if initialization.get("expected_seed_genome_digest") != seed.genome_digest:
            raise ValueError("RunCreated expected seed digest mismatch")
        self.ledger.append(
            run_id,
            "RunSeedGenomeMaterialized",
            {
                "schema_version": "ecologyrsi-dsh.run-seed-genome-materialized/1",
                "materializer_version": initialization["materializer_version"],
                "genome_canonical_json": canonical,
                "genome_digest": seed.genome_digest,
            },
            event_id=f"{run_id}:seed-genome-materialized",
            expected_run_seq=events[-1].seq,
        )
        return self.state(run_id).materialized_seed_genome()

    def start_evolution(self, task_manifest: TaskManifest, *, run_id: str | None = None) -> RunState:
        """Create and start a run in one call, suitable for a plugin action."""

        run = self.create_run(task_manifest, run_id=run_id)
        self.start_run(run.run_id)
        return self.state(run.run_id)

    def start_run(self, run_id: str) -> Run:
        opened_session_id: str | None = None

        def payload(state: RunState) -> dict[str, Any]:
            nonlocal opened_session_id
            if is_dsh_native_protocol(state.task_manifest):
                try:
                    state.materialized_seed_genome()
                except RuntimeError as exc:
                    raise RuntimeError(
                        "DSH-native run seed initialization is incomplete"
                    ) from exc
            session_id = state.run.session_id or opened_session_id
            if session_id is None:
                session_id = self.dsh.open_session(state.run, state.task_manifest)
                opened_session_id = session_id
            return {"session_id": session_id}

        self._append_run_transition(
            run_id,
            "RunStarted",
            payload,
            RunStatus.CREATED,
            RunStatus.PAUSED,
        )
        return self.state(run_id).run

    def pause_run(
        self,
        run_id: str,
        *,
        reason: str | None = None,
        code: str | None = None,
    ) -> Run:
        payload: dict[str, Any] = {}
        if reason is not None:
            normalized_reason = str(reason).strip()
            if not normalized_reason:
                raise ValueError("pause reason must be non-empty")
            payload["reason"] = normalized_reason[:500]
        if code is not None:
            normalized_code = str(code).strip()
            if (
                not normalized_code
                or len(normalized_code) > 80
                or not normalized_code.replace("_", "").isalnum()
            ):
                raise ValueError("pause code must be a bounded machine code")
            payload["code"] = normalized_code
        self._append_run_transition(
            run_id, "RunPaused", lambda _state: dict(payload), RunStatus.RUNNING
        )
        return self.state(run_id).run

    def resume_run(self, run_id: str) -> Run:
        self._append_run_transition(
            run_id, "RunResumed", lambda _state: {}, RunStatus.PAUSED
        )
        return self.state(run_id).run

    def cancel_run(self, run_id: str, reason: str = "cancelled by user") -> Run:
        state = self._append_run_transition(
            run_id,
            "RunCancelled",
            lambda _state: {"reason": str(reason)},
            RunStatus.CREATED,
            RunStatus.RUNNING,
            RunStatus.PAUSED,
        )
        self._close_session(state.run)
        return self.state(run_id).run

    def fail_run(self, run_id: str, reason: str) -> Run:
        if not str(reason).strip():
            raise ValueError("reason must be non-empty")
        state = self._append_run_transition(
            run_id,
            "RunFailed",
            lambda _state: {"reason": str(reason)},
            RunStatus.CREATED,
            RunStatus.RUNNING,
            RunStatus.PAUSED,
        )
        self._close_session(state.run)
        return self.state(run_id).run

    def request_proposal(
        self,
        run_id: str,
        *,
        parent_candidate_id: str | None = None,
        generation_batch: GenerationBatch | None = None,
        slot_index: int = 0,
        consume_interventions: bool = True,
    ) -> Proposal:
        state = self.state(run_id)
        self._require_status(state.run, RunStatus.RUNNING)
        self._require_generation_budget(state)
        if state.run.session_id is None:
            raise RuntimeError("run has no DSH session")
        if generation_batch is not None:
            if (
                generation_batch.run_id != run_id
                or generation_batch.generation != state.run.generation
            ):
                raise ValueError("generation batch is outside the current run scope")
            if not 0 <= slot_index < generation_batch.batch_size:
                raise ValueError("slot_index is outside the generation batch")
            frozen_ids = set(generation_batch.intervention_ids)
            pending_interventions = tuple(
                item for item in state.interventions if item.intervention_id in frozen_ids
            )
            if len(pending_interventions) != len(frozen_ids):
                raise RuntimeError("generation batch references missing interventions")
            parent_candidate_id = generation_batch.parent_candidate_id
        else:
            pending_interventions = state.pending_interventions
        selected_parents = [
            item.target_candidate_id
            for item in pending_interventions
            if item.kind is InterventionKind.PARENT_SELECTION
        ]
        if selected_parents:
            parent_candidate_id = selected_parents[-1]
        parent_context = None
        if parent_candidate_id is not None:
            parent_context = self._completed_parent_context(
                state, parent_candidate_id
            )
        previous_analysis = None
        knowledge_context = None
        research_iteration_context = None
        if generation_batch is not None and generation_batch.generation > 0:
            previous_analysis = state.analysis_for(generation_batch.generation - 1)
            actual_digest = (
                previous_analysis.analysis_digest if previous_analysis is not None else None
            )
            if actual_digest != generation_batch.previous_analysis_digest:
                raise RuntimeError("generation batch previous analysis changed")
        elif generation_batch is None and state.run.generation > 0:
            previous_analysis = state.analysis_for(state.run.generation - 1)
        if generation_batch is not None and generation_batch.knowledge_snapshot_digest:
            snapshot = state.knowledge_for(generation_batch.generation)
            if (
                snapshot is None
                or snapshot.snapshot_digest
                != generation_batch.knowledge_snapshot_digest
            ):
                raise RuntimeError("generation batch knowledge snapshot changed")
            knowledge_context = snapshot.proposal_context()
        if generation_batch is not None:
            research_iteration = state.research_iteration_for(
                generation_batch.generation
            )
            if research_iteration is not None:
                if (
                    research_iteration.knowledge_snapshot_digest
                    != generation_batch.knowledge_snapshot_digest
                ):
                    raise RuntimeError(
                        "generation batch research iteration changed knowledge snapshot"
                    )
                research_iteration_context = research_iteration.to_dict()
        existing_ids = {item.proposal_id for item in state.proposals}
        frozen_runtime_binding = _frozen_runtime_binding(state)
        intervention_overrides: dict[str, Any] = {}
        guidance_notes: list[str] = []
        constraint_notes: list[str] = []
        for intervention in pending_interventions:
            if intervention.kind is InterventionKind.GUIDANCE:
                guidance_notes.append(intervention.message)
            elif intervention.kind is InterventionKind.CONSTRAINT:
                constraint_notes.append(intervention.message)
            elif intervention.kind is InterventionKind.PARAMETER_OVERRIDE:
                intervention_overrides.update(intervention.parameter_overrides)
        adapter_interventions = None
        if guidance_notes or constraint_notes or intervention_overrides:
            adapter_interventions = {"host_applies_interventions": True}
            if guidance_notes:
                adapter_interventions["guidance"] = " | ".join(guidance_notes)
            if constraint_notes:
                adapter_interventions["constraints"] = list(constraint_notes)
            if intervention_overrides:
                adapter_interventions["parameter_override"] = intervention_overrides
        proposal = None
        # An adapter may keep only in-memory counters.  After replaying a
        # SQLite run in a fresh process, retry until its next proposal ID is
        # not already present in the append-only stream.
        for _ in range(64):
            candidate = self.dsh.propose(
                state.run,
                state.task_manifest,
                state.run.session_id,
                parent_candidate_id=parent_candidate_id,
                parent_context=parent_context,
                interventions=adapter_interventions,
                batch_context=(
                    {
                        "generation": generation_batch.generation,
                        "slot_index": slot_index,
                        "batch_size": generation_batch.batch_size,
                        "round_parent_candidate_id": generation_batch.parent_candidate_id,
                        "previous_generation_analysis": (
                            previous_analysis.to_dict()
                            if previous_analysis is not None
                            else None
                        ),
                        "knowledge_snapshot": knowledge_context,
                        "knowledge_snapshot_digest": (
                            generation_batch.knowledge_snapshot_digest
                        ),
                        "research_iteration": research_iteration_context,
                        "frozen_runtime_binding": frozen_runtime_binding,
                        "context_digest": generation_batch.context_digest,
                        "parent_genome_digest": generation_batch.parent_genome_digest,
                        "parent_genome_canonical_json": (
                            generation_batch.parent_genome_canonical_json
                        ),
                        "stage_context_digests": dict(
                            generation_batch.stage_context_digests or {}
                        ),
                        "run_state_revision": state.events[-1].seq,
                        "stage_attempt": 1,
                        "ledger_expected_revision": self.ledger.latest_seq(),
                    }
                    if generation_batch is not None
                    else None
                ),
            )
            if not isinstance(candidate, Proposal):
                raise TypeError("DSH adapter must return a Proposal")
            # A remote strategy call may outlive a pause or cancellation.  Do
            # not append its result after the run stopped or moved generation.
            current = self.state(run_id)
            self._require_status(current.run, RunStatus.RUNNING)
            if current.run.generation != state.run.generation:
                raise RuntimeError("run generation changed while requesting a proposal")
            if candidate.proposal_id not in existing_ids:
                proposal = candidate
                break
        if proposal is None:
            raise RuntimeError("DSH adapter returned duplicate proposal IDs")
        if not isinstance(proposal, Proposal):  # pragma: no cover - defensive
            raise TypeError("DSH adapter must return a Proposal")
        if proposal.run_id != run_id or proposal.generation != state.run.generation:
            raise ValueError("DSH proposal is outside the current run scope")
        # The strategy may suggest parameters, but only the host can derive the
        # executable retry/repair plan from aggregate prior-generation evidence.
        # Overwrite any model-supplied value and cover the frozen plan with the
        # proposal digest before compilation.
        execution_plan = derive_execution_plan(previous_analysis)
        proposal = replace(
            proposal,
            metadata={
                **dict(proposal.metadata),
                "derived_execution_plan": execution_plan.to_dict(),
            },
        )
        effective_task = _task_for_proposal_predictor(state.task_manifest, proposal)
        bounded_changes, application_receipts = apply_bounded_interventions(
            effective_task,
            proposal.changes,
            [item.to_dict() for item in pending_interventions],
            selected_parent_candidate_id=parent_candidate_id,
        )
        if application_receipts:
            rationale = proposal.rationale.strip()
            audit_notes = []
            for intervention, receipt in zip(
                pending_interventions, application_receipts
            ):
                status = receipt["application_status"]
                status_text = {
                    "enforced": "已强制执行",
                    "applied": "已应用",
                    "recorded": "仅记录、未执行",
                }[status]
                audit_notes.append(
                    f"人工干预{status_text}（{intervention.message}）："
                    f"{receipt['reason']}。"
                )
            proposal = replace(
                proposal,
                changes=bounded_changes,
                rationale="\n".join(
                    item for item in (rationale, "\n".join(audit_notes)) if item
                ),
            )
        elif dict(proposal.changes) != bounded_changes:
            proposal = replace(proposal, changes=bounded_changes)
        durable_receipts = [
            {
                **receipt,
                "intervention_id": intervention.intervention_id,
                "proposal_id": proposal.proposal_id,
            }
            for intervention, receipt in zip(
                pending_interventions, application_receipts
            )
        ]
        self.ledger.append(
            run_id,
            "ProposalSubmitted",
            {
                "proposal": proposal.to_dict(),
                "intervention_receipts": durable_receipts,
            },
        )
        for intervention, receipt in zip(pending_interventions, application_receipts):
            if not consume_interventions:
                continue
            self.ledger.append(
                run_id,
                "HumanInterventionApplied",
                {
                    "intervention_id": intervention.intervention_id,
                    "proposal_id": proposal.proposal_id,
                    **receipt,
                },
                event_id=(
                    f"{run_id}:intervention:{intervention.intervention_id}:"
                    f"applied:{proposal.proposal_id}"
                ),
            )
        return proposal

    def record_research_iteration(
        self,
        iteration: ResearchIteration,
        *,
        expert_consultation: ExpertConsultation | None = None,
    ) -> ResearchIteration:
        """Persist the one immutable research result allowed for a generation."""

        if not isinstance(iteration, ResearchIteration):
            raise TypeError("iteration must be a ResearchIteration")
        state = self.state(iteration.run_id)
        self._require_status(state.run, RunStatus.RUNNING)
        if iteration.generation != state.run.generation:
            raise ValueError("research iteration is outside the current generation")
        snapshot = state.knowledge_for(iteration.generation)
        if (
            snapshot is None
            or snapshot.snapshot_digest != iteration.knowledge_snapshot_digest
        ):
            raise ValueError("research iteration knowledge snapshot does not match")
        previous = (
            state.analysis_for(iteration.generation - 1)
            if iteration.generation > 0
            else None
        )
        expected_analysis_digest = previous.analysis_digest if previous else None
        if iteration.source_analysis_digest != expected_analysis_digest:
            raise ValueError("research iteration previous analysis does not match")
        assessment = (
            state.knowledge_assessment_for(iteration.generation - 1)
            if iteration.generation > 0
            else None
        )
        expected_assessment_digest = (
            assessment.assessment_digest if assessment is not None else None
        )
        if iteration.source_assessment_digest != expected_assessment_digest:
            raise ValueError("research iteration previous assessment does not match")
        expected_next_action = assessment.next_action if assessment is not None else None
        if iteration.previous_next_action != expected_next_action:
            raise ValueError("research iteration previous next_action does not match")
        expected_adoption = resolve_predictor_adoption(
            state.task_manifest,
            iteration.plan,
        )
        if (
            dict(iteration.prediction_model_adoption)
            != expected_adoption.to_dict()
        ):
            raise ValueError(
                "research iteration predictor adoption is not host resolved"
            )
        existing = state.research_iteration_for(iteration.generation)
        if existing is not None:
            if existing.to_dict() != iteration.to_dict():
                raise ValueError("generation already has a different research iteration")
            return existing
        pending_ids = {item.consultation_id for item in state.expert_consultations}
        if any(
            consultation_id not in pending_ids
            for consultation_id in iteration.pending_consultation_ids
        ):
            raise ValueError("research iteration references an unknown consultation")
        available_answers = {
            item.answer_id: item
            for item in state.available_expert_answers(iteration.generation)
        }
        if any(
            answer_id not in available_answers
            for answer_id in iteration.expert_answer_ids
        ):
            raise ValueError("research iteration references an unavailable expert answer")
        if expert_consultation is not None:
            if not isinstance(expert_consultation, ExpertConsultation):
                raise TypeError("expert_consultation must be an ExpertConsultation")
            if (
                expert_consultation.run_id != iteration.run_id
                or expert_consultation.generation != iteration.generation
            ):
                raise ValueError("expert consultation is outside the research iteration")
            if (
                iteration.model_id is not None
                and expert_consultation.requested_by_model_id != iteration.model_id
            ):
                raise ValueError("expert consultation model does not match research iteration")
            if expert_consultation.candidate_id is not None:
                candidate = state.candidate(expert_consultation.candidate_id)
                if candidate.generation != iteration.generation:
                    raise ValueError("expert consultation candidate is outside the generation")
            existing_consultation = next(
                (
                    item
                    for item in state.expert_consultations
                    if item.consultation_id == expert_consultation.consultation_id
                ),
                None,
            )
            if (
                existing_consultation is not None
                and existing_consultation.to_dict() != expert_consultation.to_dict()
            ):
                raise ValueError("consultation_id belongs to another consultation")

        entries: list[tuple[str, Mapping[str, Any], str | None]] = [
            (
                "GenerationResearchIterated",
                {"research_iteration": iteration.to_dict()},
                f"{iteration.run_id}:generation:{iteration.generation}:research-iterated",
            )
        ]
        for answer_id in iteration.expert_answer_ids:
            answer = available_answers[answer_id]
            entries.append(
                (
                    "ExpertConsultationApplied",
                    {
                        "consultation_id": answer.consultation_id,
                        "answer_id": answer.answer_id,
                        "generation": iteration.generation,
                        "research_iteration_digest": iteration.iteration_digest,
                    },
                    (
                        f"{iteration.run_id}:expert-consultation:"
                        f"{answer.consultation_id}:applied:{iteration.generation}"
                    ),
                )
            )
        if expert_consultation is not None:
            entries.append(
                (
                    "ExpertConsultationRequested",
                    {"consultation": expert_consultation.to_dict()},
                    (
                        f"{iteration.run_id}:expert-consultation:"
                        f"{expert_consultation.consultation_id}:requested"
                    ),
                )
            )
        self.ledger.append_many(
            iteration.run_id,
            entries,
            expected_run_seq=state.events[-1].seq,
        )
        recorded = self.state(iteration.run_id).research_iteration_for(
            iteration.generation
        )
        if recorded is None:  # pragma: no cover - append/replay invariant
            raise RuntimeError("research iteration was not recorded")
        return recorded

    def record_expert_consultation(
        self, consultation: ExpertConsultation
    ) -> ExpertConsultation:
        """Record one model-authored question without changing run status."""

        if not isinstance(consultation, ExpertConsultation):
            raise TypeError("consultation must be an ExpertConsultation")
        state = self.state(consultation.run_id)
        self._require_status(state.run, RunStatus.RUNNING)
        if consultation.generation != state.run.generation:
            raise ValueError("expert consultation generation does not match the run")
        existing = next(
            (
                item
                for item in state.expert_consultations
                if item.consultation_id == consultation.consultation_id
            ),
            None,
        )
        if existing is not None:
            if existing.to_dict() != consultation.to_dict():
                raise ValueError("consultation_id belongs to another consultation")
            return existing
        if consultation.candidate_id is not None:
            candidate = state.candidate(consultation.candidate_id)
            if candidate.generation != consultation.generation:
                raise ValueError("expert consultation candidate is outside the generation")
        self.ledger.append(
            consultation.run_id,
            "ExpertConsultationRequested",
            {"consultation": consultation.to_dict()},
            event_id=(
                f"{consultation.run_id}:expert-consultation:"
                f"{consultation.consultation_id}:requested"
            ),
            expected_run_seq=state.events[-1].seq,
        )
        return self.state(consultation.run_id).consultation(
            consultation.consultation_id
        )

    def answer_expert_consultation(
        self, answer: ExpertConsultationAnswer
    ) -> ExpertConsultationAnswer:
        """Append one asynchronous answer; it never pauses or resumes the run."""

        if not isinstance(answer, ExpertConsultationAnswer):
            raise TypeError("answer must be an ExpertConsultationAnswer")
        state = self.state(answer.run_id)
        consultation = state.consultation(answer.consultation_id)
        existing = state.answer_for_consultation(answer.consultation_id)
        if existing is not None:
            if existing.to_dict() != answer.to_dict():
                raise ValueError("expert consultation already has a different answer")
            return existing
        if any(
            item.answer_id == answer.answer_id
            for item in state.expert_consultation_answers
        ):
            raise ValueError("answer_id belongs to another expert consultation")
        if (
            answer.selected_option is not None
            and answer.selected_option not in consultation.options
        ):
            raise ValueError("selected_option is not one of the consultation options")
        if state.run.status in {
            RunStatus.COMPLETED,
            RunStatus.CANCELLED,
            RunStatus.FAILED,
        }:
            if answer.effective_generation is not None:
                raise ValueError("a terminal-run answer cannot have an effective generation")
        else:
            expected_generation = max(
                consultation.generation + 1,
                state.run.generation + 1,
            )
            if answer.effective_generation != expected_generation:
                raise ValueError(
                    "expert answer effective generation does not match the next open round"
                )
        if answer.applied_generation is not None:
            raise ValueError("a new expert answer cannot already be applied")
        self.ledger.append(
            answer.run_id,
            "ExpertConsultationAnswered",
            {"answer": answer.to_dict()},
            event_id=(
                f"{answer.run_id}:expert-consultation:"
                f"{answer.consultation_id}:answered"
            ),
            expected_run_seq=state.events[-1].seq,
        )
        recorded = self.state(answer.run_id).answer_for_consultation(
            answer.consultation_id
        )
        if recorded is None:  # pragma: no cover - append/replay invariant
            raise RuntimeError("expert consultation answer was not recorded")
        return recorded

    def submit_proposal(self, proposal: Proposal) -> Proposal:
        """Persist a plugin-created proposal after the same scope checks."""

        if not isinstance(proposal, Proposal):
            raise TypeError("proposal must be a Proposal")
        state = self.state(proposal.run_id)
        self._require_status(state.run, RunStatus.RUNNING)
        self._require_generation_budget(state)
        if proposal.generation != state.run.generation:
            raise ValueError("proposal generation does not match the run")
        if is_dsh_native_protocol(state.task_manifest):
            genome = persisted_genome_from_proposal(proposal)
            if genome is None:
                raise ValueError("DSH-native proposal requires a persisted genome")
            lineage = dict(genome.lineage)
            if (
                lineage["generation"] != proposal.generation
                or lineage["parent_candidate_id"] != proposal.parent_candidate_id
            ):
                raise ValueError("proposal scope does not match genome lineage")
            if dict(genome.scientific_program["parameter_overrides"]) != dict(
                proposal.changes
            ):
                raise ValueError("proposal changes do not match its genome")
            if proposal.generation == 0:
                seed = state.materialized_seed_genome()
                if lineage["parent_genome_digest"] != seed.genome_digest:
                    raise ValueError("first-generation proposal parent genome mismatch")
            elif proposal.parent_candidate_id is None:
                raise ValueError("later DSH-native proposal requires a parent candidate")
        existing = next(
            (item for item in state.proposals if item.proposal_id == proposal.proposal_id),
            None,
        )
        if existing is not None:
            if existing.to_dict() != proposal.to_dict():
                raise ValueError("proposal_id already belongs to a different proposal")
            return existing
        if proposal.parent_candidate_id is not None:
            parent = state.candidate(proposal.parent_candidate_id)
            if parent.run_id != proposal.run_id:
                raise ValueError("parent candidate belongs to another run")
        self.ledger.append(proposal.run_id, "ProposalSubmitted", {"proposal": proposal.to_dict()})
        return proposal

    def spawn_candidate(
        self,
        run_id: str,
        proposal: Proposal | str,
        *,
        candidate_id: str | None = None,
        slot_index: int = 0,
    ) -> Candidate:
        state = self.state(run_id)
        self._require_status(state.run, RunStatus.RUNNING)
        self._require_generation_budget(state)
        if isinstance(proposal, str):
            proposal_obj = state.proposal(proposal)
        else:
            if not isinstance(proposal, Proposal):
                raise TypeError("proposal must be a Proposal or proposal ID")
            try:
                proposal_obj = state.proposal(proposal.proposal_id)
            except KeyError as exc:
                raise ValueError("proposal must be submitted before spawning") from exc
            if proposal_obj.to_dict() != proposal.to_dict():
                raise ValueError("proposal does not match the submitted proposal")
        if proposal_obj.run_id != run_id:
            raise ValueError("proposal belongs to another run")
        if proposal_obj.generation != state.run.generation:
            raise ValueError("proposal generation does not match the run")
        if len(state.candidates) >= state.task_manifest.max_candidates:
            raise RuntimeError("run candidate budget exhausted")
        candidate_id = candidate_id or f"candidate:{uuid4()}"
        if any(item.candidate_id == candidate_id for item in state.candidates):
            raise ValueError("candidate_id already exists")
        candidate = Candidate(
            candidate_id=candidate_id,
            run_id=run_id,
            proposal_id=proposal_obj.proposal_id,
            generation=proposal_obj.generation,
            slot_index=slot_index,
        )
        payload: dict[str, Any] = {"candidate": candidate.to_dict()}
        if is_dsh_native_protocol(state.task_manifest):
            genome = persisted_genome_from_proposal(proposal_obj)
            if genome is None:
                raise ValueError("candidate proposal is missing its persisted genome")
            lineage = dict(genome.lineage)
            if (
                lineage["generation"] != candidate.generation
                or lineage["slot_index"] != candidate.slot_index
            ):
                raise ValueError("candidate coordinates do not match genome lineage")
            registry = current_program_registry()
            compiled = compile_plugin_behavior(
                genome,
                state.task_manifest,
                state.knowledge_for(candidate.generation),
                registry,
                compiler_semantic_digest=str(
                    state.task_manifest.metadata.get("compiler_semantic_digest")
                    or DEFAULT_COMPILER_SEMANTIC_DIGEST
                ),
            )
            metadata = state.task_manifest.metadata
            instance_context = CompilationInstanceContext(
                run_id=run_id,
                proposal_id=proposal_obj.proposal_id,
                candidate_id=candidate.candidate_id,
                generation=candidate.generation,
                slot_index=candidate.slot_index,
                task_manifest_digest=state.task_manifest.digest,
                dataset_snapshot_set_digest=metadata.get(
                    "dataset_snapshot_set_digest"
                ),
                split_manifest_digest=metadata.get("split_manifest_digest"),
                data_protocol_digest=metadata.get("data_protocol_digest"),
                stage_policy_digest=metadata.get("stage_policy_digest"),
                evaluator_digest=metadata.get("evaluator_digest"),
                evaluation_cohort_digest=metadata.get(
                    "evaluation_cohort_digest"
                ),
                required_capability_digest=metadata.get(
                    "required_capability_digest"
                ),
                resolved_policy_route_config_digest=metadata.get(
                    "resolved_policy_route_config_digest"
                ),
                resolved_review_route_config_digest=metadata.get(
                    "resolved_review_route_config_digest"
                ),
                preset_content_digest=metadata.get("preset_content_digest"),
                standing_tool_surface_digest=metadata.get(
                    "standing_tool_surface_digest"
                ),
                security_kernel_digest=metadata.get("security_kernel_digest"),
            )
            bound = bind_phenotype_instance(compiled, instance_context)
            payload["identity_binding"] = {
                "execution_protocol": DSH_NATIVE_EVOLUTION_PROTOCOL,
                "genome_digest": genome.genome_digest,
                "behavior_digest": genome.behavior_digest,
                "compiled_behavior_digest": compiled.compiled_behavior_digest,
                "phenotype_instance_digest": bound.phenotype_instance_digest,
                "compiler_semantic_digest": compiled.compiler_semantic_digest,
                "registry_catalog_digest": compiled.registry_catalog_digest,
                "security_semantic_digest": compiled.security_semantic_digest,
                "runtime_execution_digest": bound.runtime_execution_digest,
                "evaluation_cohort_digest": instance_context.evaluation_cohort_digest,
            }
        self.ledger.append(run_id, "CandidateSpawned", payload)
        return candidate

    def propose_and_spawn(self, run_id: str, *, parent_candidate_id: str | None = None) -> Candidate:
        proposal = self.request_proposal(run_id, parent_candidate_id=parent_candidate_id)
        return self.spawn_candidate(run_id, proposal)

    def record_artifact(self, artifact: ModelArtifact) -> ModelArtifact:
        if not isinstance(artifact, ModelArtifact):
            raise TypeError("artifact must be a ModelArtifact")
        state = self.state(artifact.run_id)
        self._require_status(state.run, RunStatus.RUNNING, RunStatus.PAUSED)
        candidate = state.candidate(artifact.candidate_id)
        if candidate.run_id != artifact.run_id:
            raise ValueError("candidate belongs to another run")
        same_id = next(
            (item for item in state.artifacts if item.artifact_id == artifact.artifact_id),
            None,
        )
        if same_id is not None:
            if same_id.to_dict() != artifact.to_dict():
                raise ValueError("artifact_id already belongs to a different artifact")
            return same_id
        existing = state.artifact_for(candidate.candidate_id)
        if existing is not None:
            if existing.to_dict() != artifact.to_dict():
                raise ValueError("candidate already has a different artifact")
            return existing
        artifact_payload: dict[str, Any] = {"artifact": artifact.to_dict()}
        if is_dsh_native_protocol(state.task_manifest):
            binding = state.candidate_identity_binding(candidate.candidate_id)
            if binding is None:
                raise ValueError("artifact candidate has no identity binding")
            artifact_payload["identity_binding"] = dict(binding)
        self.ledger.append(
            artifact.run_id,
            "ArtifactRecorded",
            artifact_payload,
        )
        return artifact

    def record_intervention(self, intervention: HumanIntervention) -> HumanIntervention:
        if not isinstance(intervention, HumanIntervention):
            raise TypeError("intervention must be a HumanIntervention")
        state = self.state(intervention.run_id)
        self._require_status(state.run, RunStatus.PAUSED)
        existing = next(
            (
                item
                for item in state.interventions
                if item.intervention_id == intervention.intervention_id
            ),
            None,
        )
        if existing is not None:
            if existing.to_dict() != intervention.to_dict():
                raise ValueError(
                    "intervention_id already belongs to a different intervention"
                )
            return existing
        if intervention.kind is InterventionKind.PARENT_SELECTION:
            if intervention.target_candidate_id is None:
                raise ValueError("parent selection requires target_candidate_id")
            self._completed_parent_context(state, intervention.target_candidate_id)
        elif intervention.target_candidate_id is not None:
            state.candidate(intervention.target_candidate_id)
        self.ledger.append(
            intervention.run_id,
            "HumanInterventionRecorded",
            {"intervention": intervention.to_dict()},
            event_id=f"{intervention.run_id}:intervention:{intervention.intervention_id}",
        )
        return intervention

    @staticmethod
    def _completed_parent_context(
        state: RunState,
        candidate_id: str,
    ) -> dict[str, Any]:
        """Build a replayable, aggregate-only context for one completed parent."""

        candidate = state.candidate(candidate_id)
        if candidate.run_id != state.run.run_id:
            raise ValueError("parent candidate belongs to another run")
        if candidate.status not in (
            CandidateStatus.PROMOTED,
            CandidateStatus.REJECTED,
        ):
            raise ValueError("parent candidate must have a completed promotion decision")
        evaluation = state.evaluation_for(candidate_id)
        if evaluation is None:
            raise ValueError("parent candidate must have an evaluation")
        proposal = state.proposal(candidate.proposal_id)
        artifact = state.artifact_for(candidate_id)
        metrics = dict(evaluation.metrics)
        evaluation_summary: dict[str, Any] = {
            "evaluation_id": evaluation.evaluation_id,
            "score": evaluation.score,
            "passed": evaluation.passed,
            "partition": evaluation.partition,
            "evaluator_digest": evaluation.evaluator_digest,
            "metrics": _evaluation_metric_summary(metrics),
        }
        judge_override = _numeric_summary(
            metrics.get("judge_parameter_override", {}),
            "judge_parameter_override",
        )
        judge_summary = {
            "model_id": metrics.get("judge_model_id"),
            "accepted": metrics.get("judge_accepted"),
            "guidance": _bounded_summary_text(metrics.get("judge_guidance")),
            "parameter_override": judge_override,
        }
        artifact_summary = None
        if artifact is not None:
            artifact_summary = {
                "artifact_id": artifact.artifact_id,
                "artifact_digest": artifact.digest,
                "model_id": artifact.model_id,
                "dataset_digest": artifact.dataset_digest,
                "training_partition": artifact.training_partition,
                "training_rows": artifact.training_rows,
                "learned_parameters": _numeric_summary(
                    artifact.learned_parameters,
                    "artifact.learned_parameters",
                ),
                "metrics": _numeric_summary(
                    artifact.metrics,
                    "artifact.metrics",
                ),
            }
        return {
            "candidate_id": candidate.candidate_id,
            "status": candidate.status.value,
            "proposal_id": proposal.proposal_id,
            "proposal_parameters": dict(proposal.changes),
            "artifact": artifact_summary,
            "evaluation": evaluation_summary,
            "judge": judge_summary,
        }

    def fail_candidate(self, run_id: str, candidate_id: str, reason: str) -> Candidate:
        state = self.state(run_id)
        self._require_status(state.run, RunStatus.RUNNING, RunStatus.PAUSED)
        candidate = state.candidate(candidate_id)
        if candidate.status is not CandidateStatus.SPAWNED:
            raise ValueError("only an unevaluated candidate can fail")
        if not str(reason).strip():
            raise ValueError("reason must be non-empty")
        self.ledger.append(
            run_id,
            "CandidateFailed",
            {"candidate_id": candidate_id, "reason": str(reason).strip()},
        )
        return self.state(run_id).candidate(candidate_id)

    def mark_candidate_duplicate(
        self,
        run_id: str,
        candidate_id: str,
        duplicate_of_candidate_id: str,
    ) -> Candidate:
        state = self.state(run_id)
        self._require_status(state.run, RunStatus.RUNNING, RunStatus.PAUSED)
        candidate = state.candidate(candidate_id)
        duplicate_of = state.candidate(duplicate_of_candidate_id)
        if candidate.status is not CandidateStatus.SPAWNED:
            raise ValueError("only a new candidate can be marked duplicate")
        if candidate.candidate_id == duplicate_of.candidate_id:
            raise ValueError("candidate cannot be a duplicate of itself")
        if duplicate_of.status is CandidateStatus.DUPLICATE:
            raise ValueError("duplicate_of_candidate_id must identify an original candidate")
        if duplicate_of.generation > candidate.generation:
            raise ValueError("duplicate target cannot come from a later generation")
        self.ledger.append(
            run_id,
            "CandidateMarkedDuplicate",
            {
                "candidate_id": candidate_id,
                "duplicate_of_candidate_id": duplicate_of_candidate_id,
            },
        )
        return self.state(run_id).candidate(candidate_id)

    def record_evaluation(
        self,
        evaluation: Evaluation,
        *,
        sample_results: Mapping[str, Any] | None = None,
    ) -> Evaluation:
        state = self.state(evaluation.run_id)
        self._require_status(state.run, RunStatus.RUNNING, RunStatus.PAUSED)
        candidate = state.candidate(evaluation.candidate_id)
        if candidate.run_id != evaluation.run_id:
            raise ValueError("candidate belongs to another run")
        same_id = next(
            (item for item in state.evaluations if item.evaluation_id == evaluation.evaluation_id),
            None,
        )
        if same_id is not None and same_id.candidate_id != evaluation.candidate_id:
            raise ValueError("evaluation_id already belongs to another candidate")
        existing = state.evaluation_for(candidate.candidate_id)
        if existing is not None:
            if existing.to_dict() != evaluation.to_dict():
                raise ValueError("candidate already has a different evaluation")
            if sample_results is not None:
                prior_result_event = next(
                    (
                        event
                        for event in state.events
                        if event.kind == "EvaluationSampleResultsRecorded"
                        and event.event_id == sample_results_event_id(evaluation)
                    ),
                    None,
                )
                if prior_result_event is not None:
                    if prior_result_event.payload != dict(sample_results):
                        raise ValueError(
                            "evaluation already has different sample results"
                        )
                    return existing
                self._validate_sample_results_completion(
                    state, evaluation, sample_results
                )
                self.ledger.append(
                    evaluation.run_id,
                    "EvaluationSampleResultsRecorded",
                    sample_results,
                    event_id=sample_results_event_id(evaluation),
                )
            return existing
        artifact = state.artifact_for(candidate.candidate_id)
        if evaluation.artifact_digest is not None and (
            artifact is None or artifact.digest != evaluation.artifact_digest
        ):
            raise ValueError("evaluation artifact_digest does not match candidate artifact")
        if candidate.status in (CandidateStatus.PROMOTED, CandidateStatus.REJECTED):
            raise ValueError("candidate already has a promotion decision")
        evaluation_payload: dict[str, Any] = {"evaluation": evaluation.to_dict()}
        if is_dsh_native_protocol(state.task_manifest):
            binding = state.candidate_identity_binding(candidate.candidate_id)
            if binding is None:
                raise ValueError("evaluation candidate has no identity binding")
            if artifact is None or evaluation.artifact_digest != artifact.digest:
                raise ValueError("DSH-native evaluation requires the exact artifact digest")
            evaluation_payload.update(
                {
                    "identity_binding": dict(binding),
                    "artifact_digest": artifact.digest,
                    "evaluation_digest": digest(evaluation.to_dict()),
                }
            )
        evaluation_entry = (
            "EvaluationRecorded",
            evaluation_payload,
            "candidate-evaluation:"
            + digest(
                {
                    "run_id": evaluation.run_id,
                    "candidate_id": evaluation.candidate_id,
                }
            ),
        )
        if sample_results is None:
            if self._candidate_has_unfinished_sample_results(
                state, evaluation.candidate_id
            ):
                raise ValueError(
                    "evaluation with an active sample result revision must be sealed atomically"
                )
            self.ledger.append(
                evaluation.run_id,
                evaluation_entry[0],
                evaluation_entry[1],
                event_id=evaluation_entry[2],
            )
        else:
            self._validate_sample_results_completion(
                state, evaluation, sample_results
            )
            self.ledger.append_many(
                evaluation.run_id,
                (
                    evaluation_entry,
                    (
                        "EvaluationSampleResultsRecorded",
                        sample_results,
                        sample_results_event_id(evaluation),
                    ),
                ),
            )
        return evaluation

    def _validate_sample_results_completion(
        self,
        state: RunState,
        evaluation: Evaluation,
        payload: Mapping[str, Any],
    ) -> None:
        if payload.get("candidate_id") != evaluation.candidate_id:
            raise ValueError("sample results belong to another candidate")
        if payload.get("run_id") != evaluation.run_id:
            raise ValueError("sample results belong to another run")
        if payload.get("evaluation_id") != evaluation.evaluation_id:
            raise ValueError("sample results belong to another evaluation")
        completed_rows = decode_sample_results(payload)
        revision = str(payload.get("revision") or "")
        start, batch_events, completed = self._sample_result_revision_events(
            state,
            evaluation.candidate_id,
            revision,
        )
        if completed is not None:
            raise ValueError("sample result revision is already completed")
        batch_rows = [
            dict(row)
            for event in batch_events
            for row in decode_sample_result_batch(event.payload)
        ]
        if len(batch_rows) != len(completed_rows):
            raise ValueError("sample result batches do not match completed row count")
        if sample_results_cohort_digest(batch_rows) != payload.get("cohort_digest"):
            raise ValueError("sample result batches do not match completion digest")
        checkpoint = start.payload.get("checkpoint")
        if (
            start.payload.get("schema_version")
            == _SAMPLE_RESULTS_START_SCHEMA_VERSION
            and isinstance(checkpoint, Mapping)
        ):
            expected_count = int(checkpoint["sample_count"])
            if expected_count != len(completed_rows):
                raise ValueError(
                    "sample result completion does not cover the checkpoint cohort"
                )
            planner_progress = [
                event
                for event in state.events
                if event.seq > start.seq
                and event.kind == "EvaluationProgressRecorded"
                and event.payload.get("schema_version")
                == "ecologyrsi-dsh.evaluation-progress/3"
                and event.payload.get("candidate_id") == evaluation.candidate_id
                and event.payload.get("revision") == revision
                and event.payload.get("role") == "planner"
            ]
            if (
                planner_progress
                and int(planner_progress[-1].payload["total_samples"])
                != expected_count
            ):
                raise ValueError(
                    "sample result progress total does not match the checkpoint cohort"
                )
            return

        # Legacy revisions declared their cohort only through planner progress.
        planner_progress = [
            event
            for event in state.events
            if event.seq > start.seq
            and event.kind == "EvaluationProgressRecorded"
            and event.payload.get("candidate_id") == evaluation.candidate_id
            and event.payload.get("role") == "planner"
        ]
        if planner_progress:
            expected_count = int(planner_progress[-1].payload["total_samples"])
            if expected_count != len(completed_rows):
                raise ValueError(
                    "sample result completion does not cover the declared evaluation cohort"
                )

    @staticmethod
    def _candidate_has_unfinished_sample_results(
        state: RunState,
        candidate_id: str,
    ) -> bool:
        starts = [
            event
            for event in state.events
            if event.kind == "EvaluationSampleResultsStarted"
            and event.payload.get("candidate_id") == candidate_id
        ]
        if not starts:
            return False
        latest = starts[-1]
        revision = latest.payload.get("revision")
        return not any(
            event.kind == "EvaluationSampleResultsRecorded"
            and event.payload.get("candidate_id") == candidate_id
            and event.payload.get("revision") == revision
            for event in state.events
        )

    @staticmethod
    def _sample_result_revision_events(
        state: RunState,
        candidate_id: str,
        revision: str,
    ) -> tuple[Event, tuple[Event, ...], Event | None]:
        """Resolve one ledger-backed revision and fence every superseded writer."""

        candidate = state.candidate(candidate_id)
        starts = [
            event
            for event in state.events
            if event.kind == "EvaluationSampleResultsStarted"
            and event.payload.get("candidate_id") == candidate_id
        ]
        matching_starts = [
            event for event in starts if event.payload.get("revision") == revision
        ]
        if not matching_starts:
            raise ValueError("sample result revision is missing its start event")
        if len(matching_starts) != 1:
            raise ValueError("sample result revision has multiple start events")
        start = matching_starts[0]
        if starts[-1].event_id != start.event_id:
            raise ValueError("sample result revision has been superseded")
        if (
            start.payload.get("run_id") != state.run.run_id
            or start.payload.get("generation") != candidate.generation
            or start.payload.get("proposal_id") != candidate.proposal_id
        ):
            raise ValueError("sample result revision start scope is invalid")
        completed_events = [
            event
            for event in state.events
            if event.kind == "EvaluationSampleResultsRecorded"
            and event.payload.get("candidate_id") == candidate_id
            and event.payload.get("revision") == revision
        ]
        if len(completed_events) > 1:
            raise ValueError("sample result revision has multiple completion events")
        batch_events = tuple(
            sorted(
                (
                    event
                    for event in state.events
                    if event.kind == "EvaluationSampleResultBatchRecorded"
                    and event.payload.get("candidate_id") == candidate_id
                    and event.payload.get("revision") == revision
                ),
                key=lambda event: (int(event.payload.get("batch_index", 0)), event.seq),
            )
        )
        indices = [int(event.payload.get("batch_index", 0)) for event in batch_events]
        if indices != list(range(1, len(batch_events) + 1)):
            raise ValueError("sample result batch indices are not contiguous")
        if any(event.payload.get("run_id") != state.run.run_id for event in batch_events):
            raise ValueError("sample result batch belongs to another run")
        return start, batch_events, completed_events[0] if completed_events else None

    def record_judgment(self, evaluation: Evaluation) -> Evaluation:
        """Attach a judge result without discarding the scientific evaluation.

        The scientific score is persisted first.  A separate event then adds
        governance fields and may only change the combined pass decision and
        metrics.  This keeps completed evaluation evidence replayable when a
        remote judge is unavailable or returns an invalid response.
        """

        state = self.state(evaluation.run_id)
        self._require_status(state.run, RunStatus.RUNNING, RunStatus.PAUSED)
        candidate = state.candidate(evaluation.candidate_id)
        existing = state.evaluation_for(candidate.candidate_id)
        if existing is None:
            raise ValueError("judgment requires a recorded scientific evaluation")
        if candidate.status in (CandidateStatus.PROMOTED, CandidateStatus.REJECTED):
            raise ValueError("candidate already has a promotion decision")
        immutable_fields = (
            "evaluation_id",
            "run_id",
            "candidate_id",
            "score",
            "partition",
            "evaluator_digest",
            "artifact_digest",
            "created_at",
        )
        changed = [
            name
            for name in immutable_fields
            if getattr(existing, name) != getattr(evaluation, name)
        ]
        if changed:
            raise ValueError(
                "judgment cannot change scientific evaluation fields: "
                + ", ".join(changed)
            )
        if existing.to_dict() == evaluation.to_dict():
            return existing
        prior_status = existing.metrics.get("judge_status")
        if prior_status == "completed":
            if existing.to_dict() != evaluation.to_dict():
                raise ValueError("candidate already has a different completed judgment")
            return existing
        judgment_payload: dict[str, Any] = {"evaluation": evaluation.to_dict()}
        if is_dsh_native_protocol(state.task_manifest):
            binding = state.candidate_identity_binding(candidate.candidate_id)
            if binding is None:
                raise ValueError("judgment candidate has no identity binding")
            artifact = state.artifact_for(candidate.candidate_id)
            if artifact is None or evaluation.artifact_digest != artifact.digest:
                raise ValueError("DSH-native judgment requires the exact artifact digest")
            judgment_payload.update(
                {
                    "identity_binding": dict(binding),
                    "artifact_digest": artifact.digest,
                    "evaluation_digest": digest(evaluation.to_dict()),
                }
            )
        self.ledger.append(
            evaluation.run_id,
            "EvaluationJudged",
            judgment_payload,
        )
        return evaluation

    def decide_promotion(self, promotion: Promotion) -> Promotion:
        state = self.state(promotion.run_id)
        self._require_status(state.run, RunStatus.RUNNING, RunStatus.PAUSED)
        candidate = state.candidate(promotion.candidate_id)
        same_id = next(
            (item for item in state.promotions if item.promotion_id == promotion.promotion_id),
            None,
        )
        if same_id is not None and same_id.candidate_id != promotion.candidate_id:
            raise ValueError("promotion_id already belongs to another candidate")
        evaluation = state.evaluation_for(candidate.candidate_id)
        if evaluation is None:
            raise ValueError("promotion requires an evaluation")
        if promotion.run_id != candidate.run_id:
            raise ValueError("promotion belongs to another run")
        existing = state.promotion_for(candidate.candidate_id)
        if existing is not None:
            if existing.to_dict() != promotion.to_dict():
                raise ValueError("candidate already has a different promotion")
            return existing
        if promotion.decision is PromotionDecision.APPROVED:
            if not evaluation.passed:
                raise ValueError("approved promotion requires a passing evaluation")
            incumbent = self._approved_incumbent(state)
            if sample_update_windows_enabled(state.task_manifest):
                cohort_digest = evaluation_cohort_digest(evaluation)
                if cohort_digest is None:
                    raise ValueError(
                        "approved bounded-window promotion requires a verifiable "
                        "evaluation cohort"
                    )
                for sibling in state.candidates:
                    if sibling.generation != candidate.generation:
                        continue
                    sibling_evaluation = state.evaluation_for(sibling.candidate_id)
                    if sibling_evaluation is None:
                        continue
                    sibling_digest = evaluation_cohort_digest(sibling_evaluation)
                    if sibling_digest is None:
                        raise ValueError(
                            "approved bounded-window promotion requires every evaluated "
                            "generation sibling to have a verifiable evaluation cohort"
                        )
                    if sibling_digest != cohort_digest:
                        raise ValueError(
                            "approved bounded-window promotion requires all evaluated "
                            "generation siblings to use the same evaluation cohort"
                        )
            cohort_comparison = (
                evaluation_cohort_comparison(
                    state.task_manifest,
                    evaluation,
                    incumbent[1],
                )
                if incumbent is not None
                else "no_incumbent"
            )
            if cohort_comparison == "unverifiable":
                raise ValueError(
                    "approved bounded-window promotion requires comparable cohort evidence"
                )
            if incumbent is not None and cohort_comparison == "different_cohort":
                raise ValueError(
                    "approved bounded-window promotion requires the same evaluation "
                    "cohort as the incumbent; a different-cohort winner may only be "
                    "used as a search parent"
                )
            if (
                incumbent is not None
                and cohort_comparison
                in {"legacy_full_cohort", "same_cohort"}
            ):
                assessment = assess_promotion_improvement(
                    evaluation,
                    incumbent[1],
                    execution_protocol=state.task_manifest.metadata.get(
                        "execution_protocol"
                    ),
                )
                if not assessment["comparable"]:
                    raise ValueError(
                        "approved promotion requires a compatible scoring contract"
                    )
                if not assessment["improved"]:
                    raise ValueError(
                        "approved promotion score "
                        f"{evaluation.score:.12g} must exceed incumbent "
                        f"{incumbent[0].candidate_id} score "
                        f"{incumbent[1].score:.12g} by more than "
                        f"{assessment['minimum_score_delta']:.12g} and satisfy "
                        "the configured paired-block confidence policy"
                    )
        if candidate.status in (CandidateStatus.PROMOTED, CandidateStatus.REJECTED):
            raise ValueError("candidate already has a promotion decision")
        promotion_payload: dict[str, Any] = {"promotion": promotion.to_dict()}
        if is_dsh_native_protocol(state.task_manifest):
            binding = state.candidate_identity_binding(candidate.candidate_id)
            artifact = state.artifact_for(candidate.candidate_id)
            if binding is None:
                raise ValueError("promotion candidate has no identity binding")
            if artifact is None or evaluation.artifact_digest != artifact.digest:
                raise ValueError("DSH-native promotion requires the exact artifact digest")
            promotion_payload.update(
                {
                    "identity_binding": dict(binding),
                    "evaluation_id": evaluation.evaluation_id,
                    "evaluation_digest": digest(evaluation.to_dict()),
                    "artifact_digest": artifact.digest,
                }
            )
        self.ledger.append(
            promotion.run_id,
            "PromotionDecided",
            promotion_payload,
        )
        return promotion

    def advance_generation(self, run_id: str) -> Run:
        last_conflict: ConcurrentRunMutationError | None = None
        for _ in range(_STATE_TRANSITION_RETRY_LIMIT):
            state = self.state(run_id)
            self._require_status(state.run, RunStatus.RUNNING)
            self._require_generation_budget(state)
            current = [
                item
                for item in state.candidates
                if item.generation == state.run.generation
            ]
            incomplete = [
                item.candidate_id
                for item in current
                if item.status
                not in (
                    CandidateStatus.PROMOTED,
                    CandidateStatus.REJECTED,
                    CandidateStatus.FAILED,
                    CandidateStatus.DUPLICATE,
                )
            ]
            if incomplete:
                raise RuntimeError(
                    "cannot advance generation with unevaluated candidates: "
                    + ", ".join(incomplete)
                )
            batch = state.batch_for(state.run.generation)
            if batch is not None:
                if len(current) != batch.batch_size:
                    raise RuntimeError("cannot advance an incomplete generation batch")
                if state.analysis_for(state.run.generation) is None:
                    raise RuntimeError("cannot advance generation before batch analysis")
                undecided = [
                    item.candidate_id
                    for item in current
                    if state.evaluation_for(item.candidate_id) is not None
                    and state.promotion_for(item.candidate_id) is None
                ]
                if undecided:
                    raise RuntimeError(
                        "cannot advance generation before unified decisions: "
                        + ", ".join(undecided)
                    )
            try:
                self.ledger.append(
                    run_id,
                    "GenerationAdvanced",
                    {"generation": state.run.generation + 1},
                    expected_run_seq=state.events[-1].seq,
                )
            except ConcurrentRunMutationError as exc:
                last_conflict = exc
                continue
            return self.state(run_id).run
        if last_conflict is not None:
            raise last_conflict
        raise RuntimeError(f"run {run_id} generation could not be advanced")

    def complete_run(
        self,
        run_id: str,
        *,
        termination_reason: str = "manual_completion",
        outcome: str | None = None,
    ) -> Run:
        if not str(termination_reason).strip():
            raise ValueError("termination_reason must be non-empty")

        def payload(state: RunState) -> dict[str, Any]:
            resolved_outcome = str(
                outcome
                or (
                    "completed_with_search_retained_candidate"
                    if state.run.best_candidate_id is not None
                    else "completed_without_acceptable_candidate"
                )
            ).strip()
            if resolved_outcome.casefold() == "accepted":
                # Normalize the legacy caller value at the write boundary so new
                # events cannot confuse search retention with formal validation.
                resolved_outcome = "completed_with_search_retained_candidate"
            if not resolved_outcome:
                raise ValueError("outcome must be non-empty")
            return {
                "outcome": resolved_outcome,
                "termination_reason": str(termination_reason).strip(),
                "accepted_candidate_id": state.run.best_candidate_id,
                "selection_scope": "iterative_training_feedback_only",
                "formal_validation_status": "not_run",
            }

        state = self._append_run_transition(
            run_id,
            "RunCompleted",
            payload,
            RunStatus.RUNNING,
            RunStatus.PAUSED,
        )
        self._close_session(state.run)
        return self.state(run_id).run

    def record_evolution_stage(
        self,
        run_id: str,
        *,
        generation: int,
        stage: str,
        status: str,
        attempt: int = 1,
        proposal_id: str | None = None,
        candidate_id: str | None = None,
        public_error: str | None = None,
        event_id: str | None = None,
    ) -> Event:
        """Append one idempotent, audit-only evolution-stage observation."""

        payload = {
            "generation": generation,
            "proposal_id": proposal_id,
            "candidate_id": candidate_id,
            "stage": stage,
            "status": status,
            "attempt": attempt,
            "public_error": public_error,
        }
        validate_evolution_stage_payload(payload)
        resolved_event_id = event_id or (
            f"{run_id}:stage:{generation}:"
            f"{candidate_id or proposal_id or 'batch'}:{stage}:{attempt}:{status}"
        )
        last_conflict: ConcurrentRunMutationError | None = None
        for _ in range(_STATE_TRANSITION_RETRY_LIMIT):
            state = self.state(run_id)
            self._require_status(state.run, RunStatus.RUNNING)
            try:
                return self.ledger.append(
                    run_id,
                    "EvolutionStageRecorded",
                    payload,
                    event_id=resolved_event_id,
                    expected_run_seq=state.events[-1].seq,
                )
            except ConcurrentRunMutationError as exc:
                last_conflict = exc
                continue
        if last_conflict is not None:
            raise last_conflict
        raise RuntimeError(f"run {run_id} stage observation could not be persisted")

    def record_evaluation_progress(
        self,
        run_id: str,
        *,
        generation: int,
        proposal_id: str,
        candidate_id: str,
        progress: Mapping[str, Any],
        revision: str | None = None,
    ) -> Event:
        """Persist one aggregate microbatch heartbeat without sample content."""

        state = self.state(run_id)
        self._require_status(state.run, RunStatus.RUNNING, RunStatus.PAUSED)
        candidate = state.candidate(candidate_id)
        if (
            candidate.generation != generation
            or candidate.proposal_id != proposal_id
        ):
            raise ValueError("evaluation progress does not match its candidate scope")
        if revision is not None:
            self._require_open_model_usage_checkpoint(state, candidate_id, revision)
        schema_version = (
            "ecologyrsi-dsh.evaluation-progress/3"
            if revision is not None
            else "ecologyrsi-dsh.evaluation-progress/2"
        )
        payload = {
            "schema_version": schema_version,
            "generation": generation,
            "proposal_id": proposal_id,
            "candidate_id": candidate_id,
            "role": progress.get("role"),
            "model_id": progress.get("model_id"),
            "batch_index": progress.get("batch_index"),
            "batch_count": progress.get("batch_count"),
            "batch_size": progress.get("batch_size"),
            "completed_samples": progress.get("completed_samples"),
            "total_samples": progress.get("total_samples"),
            "succeeded_samples": progress.get("succeeded_samples"),
            "failed_samples": progress.get("failed_samples"),
            "gateway_request_count": progress.get(
                "gateway_request_count", progress.get("batch_index")
            ),
            "adaptive_split_trigger_count": progress.get(
                "adaptive_split_trigger_count", 0
            ),
            "adaptive_split_count": progress.get("adaptive_split_count", 0),
            "adaptive_split_max_depth": progress.get(
                "adaptive_split_max_depth", 0
            ),
            "adaptive_split_recovered_samples": progress.get(
                "adaptive_split_recovered_samples", 0
            ),
            "adaptive_split_failed_samples": progress.get(
                "adaptive_split_failed_samples", 0
            ),
        }
        if revision is not None:
            payload.update(
                {
                    "revision": revision,
                    "progress_id": progress.get("progress_id"),
                    "progress_kind": progress.get("progress_kind"),
                    "in_flight_batches": progress.get("in_flight_batches"),
                    "queued_batches": progress.get("queued_batches"),
                }
            )
        validate_evaluation_progress_payload(payload)
        progress_identity = (
            f"{revision}:{payload['role']}:{payload['progress_id']}:"
            f"{payload['progress_kind']}"
            if revision is not None
            else (
                f"{payload['role']}:{payload['batch_index']}:"
                f"{payload['completed_samples']}:{payload['succeeded_samples']}:"
                f"{payload['failed_samples']}:{payload['gateway_request_count']}:"
                f"{payload['adaptive_split_count']}:"
                f"{payload['adaptive_split_recovered_samples']}:"
                f"{payload['adaptive_split_failed_samples']}"
            )
        )
        event = self.ledger.append(
            run_id,
            "EvaluationProgressRecorded",
            payload,
            event_id=f"{run_id}:evaluation-progress:{candidate_id}:{progress_identity}",
        )
        return event

    def record_model_usage(
        self,
        run_id: str,
        *,
        generation: int,
        candidate_id: str,
        role: str,
        model_id: str,
        usage: Mapping[str, Any],
        gateway_request_count: int,
        revision: str,
        usage_index: int,
        event_id: str | None = None,
    ) -> Event:
        """Append one idempotent, call-level public model usage delta."""

        if not isinstance(usage, Mapping):
            raise TypeError("model usage must be a mapping")
        expected_usage_fields = {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }
        if set(usage) != expected_usage_fields:
            raise ValueError("model usage must contain exactly the token counters")
        state = self.state(run_id)
        self._require_status(state.run, RunStatus.RUNNING, RunStatus.PAUSED)
        candidate = state.candidate(candidate_id)
        if candidate.generation != generation:
            raise ValueError("model usage generation does not match its candidate")
        payload = {
            "schema_version": "ecologyrsi-dsh.model-usage/1",
            "generation": generation,
            "candidate_id": candidate_id,
            "role": role,
            "model_id": model_id,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "gateway_request_count": gateway_request_count,
            "revision": revision,
            "usage_index": usage_index,
        }
        validate_model_usage_payload(payload)
        resolved_event_id = event_id or (
            f"{run_id}:model-usage:{candidate_id}:{generation}:{role}:"
            f"{model_id}:{revision}:{usage_index}:{gateway_request_count}"
        )
        return self.ledger.append(
            run_id,
            "ModelUsageRecorded",
            payload,
            event_id=resolved_event_id,
        )

    def record_model_usage_batch(
        self,
        run_id: str,
        *,
        generation: int,
        candidate_id: str,
        revision: str,
        receipts: Sequence[Mapping[str, Any]],
    ) -> tuple[Event, ...]:
        """Atomically record one public receipt per physical gateway call.

        V2 receipts are accepted only while the exact evaluation checkpoint is
        open. A physical call admitted before a terminal run transition may
        report usage afterwards, so terminal status does not discard billing.
        Candidate, revision, and call-id fences still prevent a delayed callback
        from being attached to a later candidate or superseding cohort.
        """

        if not isinstance(receipts, Sequence) or isinstance(
            receipts, (str, bytes, bytearray)
        ):
            raise TypeError("model usage receipts must be a sequence")
        if not receipts:
            return ()
        state = self.state(run_id)
        self._require_status(
            state.run,
            RunStatus.RUNNING,
            RunStatus.PAUSED,
            RunStatus.COMPLETED,
            RunStatus.CANCELLED,
            RunStatus.FAILED,
        )
        candidate = state.candidate(candidate_id)
        if candidate.generation != generation:
            raise ValueError("model usage generation does not match its candidate")
        if candidate.status is not CandidateStatus.SPAWNED:
            raise ValueError("model usage candidate is no longer being evaluated")
        self._require_open_model_usage_checkpoint(state, candidate_id, revision)

        existing_by_call_id: dict[str, Event] = {}
        next_usage_index = 0
        for event in state.events:
            if (
                event.kind != "ModelUsageRecorded"
                or event.payload.get("schema_version")
                != "ecologyrsi-dsh.model-usage/2"
                or event.payload.get("candidate_id") != candidate_id
                or event.payload.get("revision") != revision
            ):
                continue
            call_id = event.payload.get("call_id")
            if isinstance(call_id, str):
                existing_by_call_id[call_id] = event
            prior_index = event.payload.get("usage_index")
            if isinstance(prior_index, int) and not isinstance(prior_index, bool):
                next_usage_index = max(next_usage_index, prior_index + 1)

        expected_receipt_fields = {
            "call_id",
            "logical_call_digest",
            "role",
            "model_id",
            "outcome",
            "usage_reported",
            "http_attempts",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }
        entries: list[tuple[str, Mapping[str, Any], str]] = []
        seen_call_ids: set[str] = set()
        for position, receipt in enumerate(receipts):
            if not isinstance(receipt, Mapping):
                raise TypeError(f"model usage receipts[{position}] must be an object")
            if set(receipt) != expected_receipt_fields:
                raise ValueError("model usage receipt fields are invalid")
            call_id = receipt.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError("model usage receipt call_id is invalid")
            if call_id in seen_call_ids:
                raise ValueError("model usage batch contains duplicate call_id")
            seen_call_ids.add(call_id)
            existing = existing_by_call_id.get(call_id)
            usage_index = (
                existing.payload["usage_index"]
                if existing is not None
                else next_usage_index
            )
            if existing is None:
                next_usage_index += 1
            payload = {
                "schema_version": "ecologyrsi-dsh.model-usage/2",
                "generation": generation,
                "candidate_id": candidate_id,
                "role": receipt.get("role"),
                "model_id": receipt.get("model_id"),
                "call_id": call_id,
                "logical_call_digest": receipt.get("logical_call_digest"),
                "outcome": receipt.get("outcome"),
                "usage_reported": receipt.get("usage_reported"),
                "http_attempts": receipt.get("http_attempts"),
                "prompt_tokens": receipt.get("prompt_tokens"),
                "completion_tokens": receipt.get("completion_tokens"),
                "total_tokens": receipt.get("total_tokens"),
                "revision": revision,
                "usage_index": usage_index,
            }
            validate_model_usage_payload(payload)
            entries.append(
                (
                    "ModelUsageRecorded",
                    payload,
                    f"{run_id}:model-usage:{candidate_id}:{revision}:{call_id}",
                )
            )
        return self.ledger.append_many(run_id, entries)

    def _require_open_model_usage_checkpoint(
        self,
        state: RunState,
        candidate_id: str,
        revision: str,
    ) -> None:
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("model usage revision must be non-empty text")
        starts = [
            event
            for event in state.events
            if event.kind == "EvaluationSampleResultsStarted"
            and event.payload.get("candidate_id") == candidate_id
        ]
        if not starts or starts[-1].payload.get("revision") != revision:
            raise ValueError("model usage revision is not the current checkpoint")
        start, _batches, completed = self._sample_result_revision_events(
            state, candidate_id, revision
        )
        if (
            start.payload.get("schema_version") != _SAMPLE_RESULTS_START_SCHEMA_VERSION
            or not isinstance(start.payload.get("checkpoint"), Mapping)
        ):
            raise ValueError("model usage requires a v2 sample checkpoint")
        if completed is not None:
            raise ValueError("model usage checkpoint is already completed")

    def start_evaluation_sample_results(
        self,
        run_id: str,
        *,
        generation: int,
        proposal_id: str,
        candidate_id: str,
        revision: str,
        checkpoint: Mapping[str, Any] | None = None,
        supersedes_revision: str | None = None,
        resume_disposition: str | None = None,
    ) -> Event:
        """Start a new private result revision before sample API work begins."""

        state = self.state(run_id)
        self._require_status(state.run, RunStatus.RUNNING, RunStatus.PAUSED)
        candidate = state.candidate(candidate_id)
        if (
            candidate.status is not CandidateStatus.SPAWNED
            or candidate.generation != generation
            or candidate.proposal_id != proposal_id
        ):
            raise ValueError("sample result revision does not match its candidate")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("sample result revision must be non-empty text")
        checkpoint_data = (
            self._validated_sample_checkpoint(checkpoint)
            if checkpoint is not None
            else None
        )
        if supersedes_revision is not None and (
            not isinstance(supersedes_revision, str)
            or not supersedes_revision.strip()
        ):
            raise ValueError("supersedes_revision must be non-empty text")
        if resume_disposition is not None and (
            not isinstance(resume_disposition, str)
            or not resume_disposition.strip()
        ):
            raise ValueError("resume_disposition must be non-empty text")
        payload = {
            "schema_version": (
                _SAMPLE_RESULTS_START_SCHEMA_VERSION
                if checkpoint_data is not None
                else "ecologyrsi-dsh.evaluation-sample-results-start/1"
            ),
            "run_id": run_id,
            "generation": generation,
            "proposal_id": proposal_id,
            "candidate_id": candidate_id,
            "revision": revision.strip(),
        }
        if checkpoint_data is not None:
            payload["checkpoint"] = checkpoint_data
            payload["supersedes_revision"] = (
                supersedes_revision.strip()
                if supersedes_revision is not None
                else None
            )
            payload["resume_disposition"] = (
                resume_disposition.strip()[:120]
                if resume_disposition is not None
                else "new_checkpoint"
            )
        event = self.ledger.append(
            run_id,
            "EvaluationSampleResultsStarted",
            payload,
            event_id=f"{run_id}:sample-results-start:{candidate_id}:{revision.strip()}",
        )
        return event

    def prepare_evaluation_sample_checkpoint(
        self,
        run_id: str,
        *,
        generation: int,
        proposal_id: str,
        candidate_id: str,
        checkpoint: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Resume the latest matching revision or append a fenced replacement.

        The caller derives the complete cohort and execution-context digests at
        the executor boundary. Only an unfinished revision carrying those exact
        digests is reusable. Legacy or mismatched revisions remain in the
        append-only ledger but never contribute rows to the new evaluation.
        """

        checkpoint_data = self._validated_sample_checkpoint(checkpoint)
        state = self.state(run_id)
        self._require_status(state.run, RunStatus.RUNNING, RunStatus.PAUSED)
        candidate = state.candidate(candidate_id)
        if (
            candidate.status is not CandidateStatus.SPAWNED
            or candidate.generation != generation
            or candidate.proposal_id != proposal_id
        ):
            raise ValueError("sample checkpoint does not match its candidate")

        starts = [
            event
            for event in state.events
            if event.kind == "EvaluationSampleResultsStarted"
            and event.payload.get("candidate_id") == candidate_id
        ]
        latest = starts[-1] if starts else None
        rejection = "no_previous_revision"
        if latest is not None:
            revision = str(latest.payload.get("revision") or "")
            latest_checkpoint = latest.payload.get("checkpoint")
            if (
                latest.payload.get("schema_version")
                == _SAMPLE_RESULTS_START_SCHEMA_VERSION
                and isinstance(latest_checkpoint, Mapping)
                and dict(latest_checkpoint) == checkpoint_data
            ):
                _start, batch_events, completed = self._sample_result_revision_events(
                    state, candidate_id, revision
                )
                if completed is None:
                    rows = tuple(
                        dict(row)
                        for event in batch_events
                        for row in decode_sample_result_batch(event.payload)
                    )
                    if len(rows) > int(checkpoint_data["sample_count"]):
                        raise ValueError(
                            "sample checkpoint contains more rows than its cohort"
                        )
                    resume_attempt = 1 + sum(
                        event.kind == "EvaluationSampleResultsResumed"
                        and event.payload.get("candidate_id") == candidate_id
                        and event.payload.get("revision") == revision
                        for event in state.events
                    )
                    progress = next(
                        (
                            dict(event.payload)
                            for event in reversed(state.events)
                            if event.seq > latest.seq
                            and event.kind == "EvaluationProgressRecorded"
                            and event.payload.get("candidate_id") == candidate_id
                            and event.payload.get("role") == "planner"
                        ),
                        None,
                    )
                    durable_gateway_request_count = sum(
                        int(event.payload["http_attempts"])
                        for event in state.events
                        if event.kind == "ModelUsageRecorded"
                        and event.payload.get("schema_version")
                        == "ecologyrsi-dsh.model-usage/2"
                        and event.payload.get("candidate_id") == candidate_id
                        and event.payload.get("revision") == revision
                    )
                    if durable_gateway_request_count:
                        if progress is None:
                            progress = {
                                "gateway_request_count": (
                                    durable_gateway_request_count
                                )
                            }
                        else:
                            prior_gateway_request_count = progress.get(
                                "gateway_request_count", 0
                            )
                            if (
                                isinstance(prior_gateway_request_count, bool)
                                or not isinstance(prior_gateway_request_count, int)
                                or prior_gateway_request_count < 0
                            ):
                                prior_gateway_request_count = 0
                            progress["gateway_request_count"] = max(
                                prior_gateway_request_count,
                                durable_gateway_request_count,
                            )
                    self.ledger.append(
                        run_id,
                        "EvaluationSampleResultsResumed",
                        {
                            "schema_version": _SAMPLE_RESULTS_RESUME_SCHEMA_VERSION,
                            "run_id": run_id,
                            "generation": generation,
                            "proposal_id": proposal_id,
                            "candidate_id": candidate_id,
                            "revision": revision,
                            "resume_attempt": resume_attempt,
                            "record_count": len(rows),
                            "next_batch_index": len(batch_events) + 1,
                            "checkpoint": checkpoint_data,
                        },
                        event_id=(
                            f"{run_id}:sample-results-resume:{candidate_id}:"
                            f"{revision}:{resume_attempt}"
                        ),
                    )
                    return {
                        "revision": revision,
                        "rows": rows,
                        "next_batch_index": len(batch_events) + 1,
                        "progress": progress,
                        "resumed": True,
                    }
                rejection = "latest_revision_completed"
            elif latest.payload.get("schema_version") != _SAMPLE_RESULTS_START_SCHEMA_VERSION:
                rejection = "legacy_revision_without_checkpoint"
            else:
                rejection = "checkpoint_digest_mismatch"

        revision = str(uuid4())
        self.start_evaluation_sample_results(
            run_id,
            generation=generation,
            proposal_id=proposal_id,
            candidate_id=candidate_id,
            revision=revision,
            checkpoint=checkpoint_data,
            supersedes_revision=(
                str(latest.payload.get("revision")) if latest is not None else None
            ),
            resume_disposition=rejection,
        )
        return {
            "revision": revision,
            "rows": (),
            "next_batch_index": 1,
            "progress": None,
            "resumed": False,
        }

    @staticmethod
    def _validated_sample_checkpoint(
        checkpoint: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(checkpoint, Mapping):
            raise TypeError("sample checkpoint must be an object")
        allowed = {
            "schema_version",
            "cohort_digest",
            "execution_context_digest",
            "sample_count",
        }
        unknown = set(checkpoint) - allowed
        if unknown:
            raise ValueError(
                "sample checkpoint contains unsupported fields: "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        if checkpoint.get("schema_version") != _SAMPLE_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported sample checkpoint schema version")
        projected = {"schema_version": _SAMPLE_CHECKPOINT_SCHEMA_VERSION}
        for name in ("cohort_digest", "execution_context_digest"):
            value = checkpoint.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"sample checkpoint {name} must be a SHA-256 digest")
            projected[name] = value
        sample_count = checkpoint.get("sample_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or not 1 <= sample_count <= MAX_SAMPLE_RESULTS_RECORDS
        ):
            raise ValueError("sample checkpoint sample_count is outside the record limit")
        projected["sample_count"] = sample_count
        return projected

    def record_evaluation_sample_result_batch(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> Event:
        """Append one bounded batch already finalized by the host executor."""

        payload_dict = dict(payload)
        new_rows = decode_sample_result_batch(payload_dict)
        if payload_dict.get("run_id") != run_id:
            raise ValueError("sample result batch belongs to another run")
        candidate_id = str(payload_dict["candidate_id"])
        revision = str(payload_dict["revision"])
        batch_index = int(payload_dict["batch_index"])
        state = self.state(run_id)
        self._require_status(state.run, RunStatus.RUNNING, RunStatus.PAUSED)
        candidate = state.candidate(candidate_id)
        if candidate.status is not CandidateStatus.SPAWNED:
            raise ValueError("sample result candidate is no longer being evaluated")
        _start, batch_events, completed = self._sample_result_revision_events(
            state, candidate_id, revision
        )
        if completed is not None:
            raise ValueError("sample result revision is already completed")
        event_id = (
            f"{run_id}:sample-results-batch:{candidate_id}:"
            f"{revision}:{batch_index}"
        )
        existing = next(
            (
                event
                for event in batch_events
                if int(event.payload.get("batch_index", 0)) == batch_index
            ),
            None,
        )
        if existing is not None:
            return self.ledger.append(
                run_id,
                "EvaluationSampleResultBatchRecorded",
                payload_dict,
                event_id=event_id,
            )
        if batch_index != len(batch_events) + 1:
            raise ValueError("sample result batch_index must increase contiguously")
        prior_count = sum(int(event.payload.get("record_count", 0)) for event in batch_events)
        if prior_count + len(new_rows) > MAX_SAMPLE_RESULTS_RECORDS:
            raise ValueError("sample result revision exceeds the record limit")
        prior_sample_ids = {
            str(sample_id)
            for event in batch_events
            for sample_id in event.payload.get("sample_ids", [])
        }
        prior_sample_indices = {
            int(sample_index)
            for event in batch_events
            for sample_index in event.payload.get("sample_indices", [])
        }
        new_sample_ids = {str(row["sample_id"]) for row in new_rows}
        new_sample_indices = {int(row["sample_index"]) for row in new_rows}
        if prior_sample_ids.intersection(new_sample_ids):
            raise ValueError("sample result revision contains a duplicate sample_id")
        if prior_sample_indices.intersection(new_sample_indices):
            raise ValueError("sample result revision contains a duplicate sample_index")
        event = self.ledger.append(
            run_id,
            "EvaluationSampleResultBatchRecorded",
            payload_dict,
            event_id=event_id,
        )
        return event

    def record_algorithm_attempt(self, attempt: AlgorithmAttempt) -> AlgorithmAttempt:
        """Persist one candidate compile/debug result before model training."""

        if not isinstance(attempt, AlgorithmAttempt):
            raise TypeError("attempt must be an AlgorithmAttempt")
        state = self.state(attempt.run_id)
        self._require_status(state.run, RunStatus.RUNNING, RunStatus.PAUSED)
        candidate = state.candidate(attempt.candidate_id)
        if (
            candidate.proposal_id != attempt.proposal_id
            or candidate.generation != attempt.generation
        ):
            raise ValueError("algorithm attempt does not match its candidate scope")
        existing = next(
            (
                item
                for item in state.algorithm_attempts_for(attempt.candidate_id)
                if item.phase == attempt.phase and item.attempt == attempt.attempt
            ),
            None,
        )
        if existing is not None:
            expected = attempt.to_dict()
            actual = existing.to_dict()
            expected.pop("created_at", None)
            actual.pop("created_at", None)
            if expected != actual:
                raise ValueError("algorithm attempt identity was reused")
            return existing
        attempt_payload: dict[str, Any] = {"algorithm_attempt": attempt.to_dict()}
        if is_dsh_native_protocol(state.task_manifest):
            binding = state.candidate_identity_binding(candidate.candidate_id)
            if binding is None:
                raise ValueError("algorithm attempt candidate has no identity binding")
            attempt_payload["identity_binding"] = dict(binding)
        self.ledger.append(
            attempt.run_id,
            "AlgorithmAttemptRecorded",
            attempt_payload,
            event_id=(
                f"{attempt.run_id}:algorithm:{attempt.candidate_id}:"
                f"{attempt.phase}:{attempt.attempt}"
            ),
        )
        return next(
            item
            for item in self.state(attempt.run_id).algorithm_attempts_for(
                attempt.candidate_id
            )
            if item.phase == attempt.phase and item.attempt == attempt.attempt
        )

    def reserve_formal_stage(
        self,
        run_id: str,
        *,
        stage: str,
        candidate_id: str,
        objective_family_digest: str,
        analysis_plan_digest: str,
        partition_digest: str,
        idempotency_key: str,
    ) -> FormalStageToken:
        """Lock one candidate/artifact/plan before any formal metric is read."""

        state = self.state(run_id)
        if not is_dsh_native_protocol(state.task_manifest):
            raise ValueError("formal stage tokens require the DSH-native protocol")
        candidate = state.candidate(candidate_id)
        if stage == "validation":
            if state.run.selection_incumbent_id != candidate_id:
                raise ValueError("validation requires the locked selection incumbent")
        elif stage == "final_test":
            if state.run.validated_candidate_id != candidate_id:
                raise ValueError("final-test requires the validated candidate")
        else:
            raise ValueError("formal stage must be validation or final_test")
        artifact = state.artifact_for(candidate_id)
        if artifact is None:
            raise ValueError("formal stage requires a frozen artifact")
        genome = state.persisted_genome_for(candidate_id)
        metadata = state.task_manifest.metadata
        dataset_digest = str(metadata.get("dataset_digest") or "")
        split_digest = str(metadata.get("split_manifest_digest") or "")
        episode_id = str(metadata.get("episode_id") or "")
        raw_key = raw_holdout_exposure_key(
            dataset_digest=dataset_digest,
            split_manifest_digest=split_digest,
            episode_id=episode_id,
            stage=stage,
            stage_partition_digest=partition_digest,
        )
        token = ScientificExposureRegistry(self.ledger).reserve_formal_stage(
            raw_holdout_key=raw_key,
            objective_family_digest=objective_family_digest,
            plan_digest=analysis_plan_digest,
            idempotency_key=idempotency_key,
            run_id=run_id,
            stage=stage,
            candidate_id=candidate_id,
            artifact_digest=artifact.digest,
            genome_digest=genome.genome_digest,
            partition_digest=partition_digest,
        )
        self.ledger.append(
            run_id,
            "FormalStageFrozen",
            {
                "stage": stage,
                "candidate_id": candidate_id,
                "artifact_digest": artifact.digest,
                "genome_digest": genome.genome_digest,
                "analysis_plan_digest": analysis_plan_digest,
                "objective_family_digest": objective_family_digest,
                "partition_digest": partition_digest,
                "holdout_exposure_key": raw_key,
                "token_digest": token.token_digest,
            },
            event_id=f"{run_id}:formal:{stage}:frozen",
        )
        return token

    def execute_formal_stage(
        self,
        run_id: str,
        token: FormalStageToken,
        evaluator: Any,
    ) -> Mapping[str, Any]:
        """Open before evaluation, persist aggregate result, and always seal."""

        if token.run_id != run_id:
            raise ValueError("formal token belongs to another run")
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        registry = ScientificExposureRegistry(self.ledger)
        outcome = "failed"
        try:
            result = registry.with_formal_stage(token, evaluator)
            if not isinstance(result, Mapping):
                raise TypeError("formal evaluator must return an aggregate object")
            result_dict = dict(result)
            if any(name in result_dict for name in ("rows", "samples", "raw")):
                raise ValueError("formal evaluator cannot return raw rows")
            outcome = str(result_dict.get("outcome") or "")
            if outcome not in {"passed", "failed", "inconclusive"}:
                raise ValueError("formal evaluator outcome is invalid")
            self.ledger.append(
                run_id,
                "FormalStageCompleted",
                {
                    "stage": token.stage,
                    "candidate_id": token.candidate_id,
                    "token_digest": token.token_digest,
                    "outcome": outcome,
                    "assessment_digest": digest(result_dict),
                    "assessment": result_dict,
                },
                event_id=f"{run_id}:formal:{token.stage}:completed",
            )
            return result_dict
        finally:
            exposure = registry.formal_exposure(token.holdout_exposure_key)
            if exposure is not None and exposure.get("state") != "sealed":
                registry.seal_formal_stage(token, outcome=outcome)
            self.ledger.append(
                run_id,
                "FormalStageSealed",
                {
                    "stage": token.stage,
                    "candidate_id": token.candidate_id,
                    "token_digest": token.token_digest,
                    "outcome": outcome,
                },
                event_id=f"{run_id}:formal:{token.stage}:sealed",
            )

    def state(self, run_id: str) -> RunState:
        events = self.ledger.events(run_id)
        if not events:
            raise KeyError(f"unknown run: {run_id}")
        return project_run_state(events)

    # Explicit alias for callers that want to emphasize event replay.
    replay = state

    def run(self, run_id: str) -> Run:
        return self.state(run_id).run

    def evaluate_and_decide(
        self,
        evaluation: Evaluation,
        *,
        reason: str | None = None,
        promotion_id: str | None = None,
    ) -> Promotion:
        """Convenience for a local evaluator's single-candidate loop."""

        self.record_evaluation(evaluation)
        state = self.state(evaluation.run_id)
        incumbent = self._approved_incumbent(state)
        current_cohort_verifiable = bool(
            not sample_update_windows_enabled(state.task_manifest)
            or evaluation_cohort_digest(evaluation) is not None
        )
        cohort_comparison = (
            evaluation_cohort_comparison(
                state.task_manifest,
                evaluation,
                incumbent[1],
            )
            if incumbent is not None
            else "no_incumbent"
        )
        promotion_assessment = (
            assess_promotion_improvement(evaluation, incumbent[1])
            if incumbent is not None
            and cohort_comparison in {"legacy_full_cohort", "same_cohort"}
            else None
        )
        improves_incumbent = (
            current_cohort_verifiable
            and (
                incumbent is None
                or (
                    cohort_comparison in {"legacy_full_cohort", "same_cohort"}
                    and promotion_assessment is not None
                    and bool(promotion_assessment["improved"])
                )
            )
        )
        approved = evaluation.passed and improves_incumbent
        if not evaluation.passed:
            default_reason = "候选未通过固定科学评测门槛。"
        elif not current_cohort_verifiable or cohort_comparison == "unverifiable":
            default_reason = "候选缺少可验证的固定样本窗口摘要，未作正式晋升。"
        elif incumbent is None:
            default_reason = "首个通过科学评测的候选，建立当前最优基线。"
        elif cohort_comparison == "different_cohort":
            default_reason = (
                "候选通过本轮固定样本窗口门禁，但评测窗口与当前最优方案不同；"
                "未比较跨窗口原始分数，也未作正式晋升。该结果仅可作为后续搜索证据。"
            )
        elif promotion_assessment is not None and not promotion_assessment[
            "comparable"
        ]:
            default_reason = (
                "候选与当前最优方案的评分合同、基线摘要或评测证据不兼容，"
                "未作正式晋升。"
            )
        elif approved:
            if (
                promotion_assessment is not None
                and promotion_assessment.get("confidence_interval_95") is not None
            ):
                interval = promotion_assessment["confidence_interval_95"]
                default_reason = (
                    f"候选得分 {evaluation.score:.12g} 高于当前最优候选 "
                    f"{incumbent[0].candidate_id} 的 {incumbent[1].score:.12g}，"
                    f"且 24 小时配对区块差异的 95% 置信区间下界 "
                    f"{float(interval[0]):.6g} 大于 0。"
                )
            else:
                default_reason = (
                    f"候选得分 {evaluation.score:.12g} 严格高于当前最优候选 "
                    f"{incumbent[0].candidate_id} 的 {incumbent[1].score:.12g}。"
                )
        else:
            required_delta = (
                float(promotion_assessment["minimum_score_delta"])
                if promotion_assessment is not None
                else _INCUMBENT_SCORE_TOLERANCE
            )
            confidence_note = (
                "；配对区块置信区间未完全高于 0"
                if promotion_assessment is not None
                and promotion_assessment.get("reason_code")
                == "confidence_interval_crosses_zero"
                else ""
            )
            default_reason = (
                f"候选虽通过科学评测，但得分 {evaluation.score:.12g} 未严格高于"
                f"当前最优候选 {incumbent[0].candidate_id} 的 "
                f"{incumbent[1].score:.12g}（要求提升超过 "
                f"{required_delta:.12g}{confidence_note}）。"
            )
        promotion = Promotion(
            promotion_id=promotion_id or f"promotion:{uuid4()}",
            run_id=evaluation.run_id,
            candidate_id=evaluation.candidate_id,
            decision=(
                PromotionDecision.APPROVED
                if approved
                else PromotionDecision.REJECTED
            ),
            reason=reason or default_reason,
        )
        return self.decide_promotion(promotion)

    @staticmethod
    def _approved_incumbent(
        state: RunState,
    ) -> tuple[Candidate, Evaluation] | None:
        candidate_id = state.run.best_candidate_id
        if candidate_id is None:
            return None
        candidate = state.candidate(candidate_id)
        evaluation = state.evaluation_for(candidate_id)
        if (
            candidate.status is not CandidateStatus.PROMOTED
            or evaluation is None
        ):
            raise RuntimeError("approved incumbent is missing its promotion evidence")
        return candidate, evaluation

    def _close_session(self, run: Run) -> None:
        if run.session_id is not None:
            self.dsh.close_session(run.session_id)

    @staticmethod
    def _require_status(run: Run, *allowed: RunStatus) -> None:
        if run.status not in allowed:
            names = ", ".join(item.value for item in allowed)
            raise RuntimeError(f"run {run.run_id} is {run.status.value}; expected {names}")

    @staticmethod
    def _require_generation_budget(state: RunState) -> None:
        if state.run.generation >= state.task_manifest.max_generations:
            raise RuntimeError("run generation budget exhausted")
