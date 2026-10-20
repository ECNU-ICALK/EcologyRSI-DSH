"""Scientific evaluation: fit, feedback execution, scoring and artifact delivery.

Only training_fit and training_feedback enter this adaptive pipeline.
Development and external holdout observations stay outside its data boundary.
"""
from __future__ import annotations
from ..core.prediction_policy import RUNTIME_EVALUATOR_ID

from ..evolution.schedule import ADAPTIVE_PROTOCOLS
from pathlib import Path
from ..evolution.agent_policy import summarize_agent_tools, rebind_agent_policy
from ..evolution.effect_contracts import observe_edit_effects, verify_runtime_effects
from ..core.agent_prediction import successful_agent_provenance_passes
from ..core.prediction_policy import prediction_usage
import math
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Any
from ..execution.sample_admission import (
    HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK,
    MAX_SAMPLE_CONCURRENCY,
)
from ..execution.batch import NativeBatchBackend
from ..science.context import EvaluationContext
from ..core.models import Candidate, Evaluation, ModelArtifact, Proposal, TaskManifest, digest
from ..core.trajectory import EvaluationScope
from ..core.protocols import is_strict_origin_protocol
from ..core.sample_results import SAMPLE_REWARD_DEFINITION, build_sample_results
from ..data.registry import DatasetRegistry, DatasetSeries
from ..data.adapters import DatasetAdapter, dataset_adapter
from ..core.immutable import thaw_json
from ..evolution.execution_plan import DerivedExecutionPlan, derive_execution_plan
from ..evolution.promotion import V2_MINIMUM_SCORE_DELTA, build_promotion_block_evidence
from ..knowledge.algorithms import (
    EVOLUTION_ALLOWED_PARTITIONS,
    AlgorithmSpec,
)
from .dsh_sample_adapter import DshSampleCollaborationAdapter
from .agent_model_tools import AgentModelTools
from .noninferiority import (
    CellNoninferiorityAssessment,
    assess_cell_noninferiority,
    assess_relaxation_debt,
)
from .epoch_cohorts import PlannedCohort
from .shared_sample_context import sibling_stage_context_digest
from .baselines import apply_baseline_profile, fit_baseline_profile
from .greenhouse_prediction import (
    BASELINE_ALIGNED_RIDGE_MODEL_ID,
    BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID,
    BaselineAlignedRidgeConfig,
    BaselineAlignedCellwiseRidgeConfig,
    EXOGENOUS_RIDGE_MODEL_ID,
    HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    MAX_EXOGENOUS_RIDGE_HISTORY_STEPS,
    RECIPE_RIDGE_MODEL_ID,
    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    ExogenousRidgeConfig,
    HorizonTargetwiseExogenousRidgeConfig,
    RecipeRidgeConfig,
    RidgeConfig,
    TargetwiseExogenousRidgeConfig,
    origin_history_alignment_hours,
    predict_fitted_exogenous_ridge,
)
from .cell_scoring import score_cell, execution_for_cell
from .metrics import (
    MAX_ROLLING_WINDOW_HOURS,
    NORMALIZATION_SCALE_METHOD,
    _exact_time_eligible_rows,
    _normalization_scale,
    _rmse,
)
from .objectives import aggregate_greenhouse_objective, skill_score
from .sample_execution import (
    CollaborativeSampleExecutor,
    SampleExecutionPolicy,
    SamplePredictionRequest,
    _sample_id,
    bounded_sample_execution_records,
    encode_sample_execution_trace,
    summarize_tool_performance,
)
from .catalog import (
    GREENHOUSE_NO_REGRESSION_TOLERANCE,
    GREENHOUSE_OBJECTIVE_AGGREGATION_VERSION,
    GREENHOUSE_OBJECTIVE_MISSING_PENALTY,
    GREENHOUSE_OBJECTIVE_PROFILE_ID,
    GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS,
    _greenhouse_scoring_contract,
    _noninferiority_policy,
)

GREENHOUSE_PRIMARY_FITNESS_DEFINITION = "weighted_symmetric_rmse_skill@1"


_PREVIEW_ROWS_PER_TARGET = 16


_PREVIEW_ROWS_TOTAL = 48


_FEEDBACK_UPDATE_COHORT_SCHEMA_VERSION = (
    "ecologyrsi-dsh.feedback-update-cohort/1"
)


def _aggregate_greenhouse_objective(
    task_results: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    *,
    target_weights: Mapping[str, float] | None = None,
    missing_skill_penalty: float = GREENHOUSE_OBJECTIVE_MISSING_PENALTY,
    missing_reward_penalty: float = GREENHOUSE_OBJECTIVE_MISSING_PENALTY,
) -> dict[str, Any]:
    """Compatibility wrapper around the pure, versioned aggregator."""

    return aggregate_greenhouse_objective(
        task_results,
        horizons,
        target_weights=target_weights if target_weights is not None else GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS,
        missing_skill_penalty=missing_skill_penalty,
        missing_reward_penalty=missing_reward_penalty,
    )


def _feedback_sample_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the label-free identity used to freeze an update cohort."""

    target = str(row.get("target") or "").strip()
    horizon = row.get("horizon_hours")
    if not target:
        raise ValueError("feedback sample target must be non-empty")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("feedback sample horizon_hours must be a positive integer")
    return {
        "partition": str(row.get("partition") or "training_feedback"),
        "target": target,
        "horizon_hours": horizon,
        "origin_timestamp": row.get("origin_timestamp"),
        "target_timestamp": row.get("timestamp", row.get("target_timestamp")),
    }


def _feedback_timestamp_sort_key(value: Any) -> tuple[int, float, str]:
    if (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ):
        return (0, float(value), "")
    return (1, 0.0, str(value))






def _select_planned_evaluation_cohort(
    rows: Sequence[Mapping[str, Any]],
    cohort: PlannedCohort,
    *,
    expected_prediction_cells_per_origin: int = 9,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize one frozen origin cohort without inspecting label values.

    The cohort planner owns origin identity and order. The evaluator is only
    allowed to attach the complete target/horizon prediction vector already
    present in its causal evaluation population; it may not substitute or
    truncate a frozen occurrence. When the planner has explicitly wrapped the
    source population, each occurrence is materialized with a fresh sample
    identity while retaining the original causal timestamp.
    """

    if not isinstance(cohort, PlannedCohort):
        raise TypeError("cohort must be a PlannedCohort")
    if (
        isinstance(expected_prediction_cells_per_origin, bool)
        or not isinstance(expected_prediction_cells_per_origin, int)
        or expected_prediction_cells_per_origin < 1
    ):
        raise ValueError("expected prediction cells per origin must be positive")
    grouped: dict[Any, dict[tuple[str, int], dict[str, Any]]] = {}
    task_keys: set[tuple[str, int]] = set()
    for raw_row in rows:
        row = dict(raw_row)
        identity = _feedback_sample_identity(row)
        origin_timestamp = identity["origin_timestamp"]
        task_key = (str(identity["target"]), int(identity["horizon_hours"]))
        task_keys.add(task_key)
        origin_rows = grouped.setdefault(origin_timestamp, {})
        if task_key in origin_rows:
            raise ValueError("evaluation origin contains duplicate prediction cells")
        origin_rows[task_key] = row
    if len(task_keys) != expected_prediction_cells_per_origin:
        raise ValueError(
            "evaluation population does not expose the complete prediction vector"
        )
    ordered_tasks = tuple(sorted(task_keys))
    population_identities = sorted(
        (_feedback_sample_identity(row) for row in rows),
        key=lambda identity: (
            str(identity["partition"]),
            _feedback_timestamp_sort_key(identity["origin_timestamp"]),
            str(identity["target"]),
            int(identity["horizon_hours"]),
            _feedback_timestamp_sort_key(identity["target_timestamp"]),
        ),
    )
    population_count = len(population_identities)
    population_origin_count = len(grouped)
    population_digest = digest(
        {
            "schema_version": "ecologyrsi-dsh.frozen-evaluation-population/1",
            "selection_policy": "label_free_complete_origin_population@1",
            "rows": population_identities,
        }
    )
    source_origin_positions = {
        timestamp: position
        for position, timestamp in enumerate(
            sorted(grouped, key=_feedback_timestamp_sort_key)
        )
    }
    selected: list[dict[str, Any]] = []
    for origin_position, origin in enumerate(cohort.origins):
        origin_rows = grouped.get(origin.origin_timestamp)
        if origin_rows is None or set(origin_rows) != task_keys:
            raise ValueError(
                "frozen evaluation origin does not expose a complete prediction vector"
            )
        # Materialize a fresh row for every planned occurrence. This is
        # essential when the cohort planner wraps: source timestamps may be
        # reused, but sample identities and strict origin bundles must remain
        # unique within this evaluation scope.
        for task_offset, task_key in enumerate(ordered_tasks):
            row = dict(origin_rows[task_key])
            row["sample_index"] = (
                origin_position * len(ordered_tasks) + task_offset + 1
            )
            row["cohort_origin_occurrence"] = origin.reuse_index
            selected.append(row)
    first_origin = cohort.origins[0]
    first_cycle = int(first_origin.reuse_index)
    source_window_offset = source_origin_positions[first_origin.origin_timestamp]
    selected_count = len(selected)
    evidence = {
        "schema_version": "ecologyrsi-dsh.frozen-evaluation-cohort/1",
        "selection_policy": "planner_frozen_complete_origins@1",
        "cohort_role": cohort.role,
        "cohort_digest": cohort.cohort_digest,
        # Public update sizes are complete forecast origins. Prediction-cell
        # counts remain explicit so one 3-target x 3-horizon forecast is not
        # presented as nine independent samples.
        "samples_per_update": cohort.origin_count,
        "prediction_cell_budget": selected_count,
        "selected_origin_count": cohort.origin_count,
        "prediction_cells_per_origin": len(ordered_tasks),
        "population_origin_count": population_origin_count,
        "population_count": population_count,
        "population_digest": population_digest,
        "selected_count": selected_count,
        "deferred_count": max(0, population_count - selected_count),
        "window_offset": source_window_offset,
        "window_cycle": first_cycle,
        "window_wraps": any(
            origin.reuse_index != first_cycle for origin in cohort.origins
        ),
        "tasks": [
            {
                "target": target,
                "horizon_hours": horizon,
                "population_count": len(grouped),
                "selected_count": cohort.origin_count,
            }
            for target, horizon in ordered_tasks
        ],
    }
    return selected, evidence




def _resolve_origin_history_alignment(
    task: TaskManifest, config: RidgeConfig
) -> int:
    """How much complete target history every candidate's cohort is aligned to.

    This is a *run-level* constant, not a searchable parameter: the feedback
    cohort is intersected down to the origins obtainable at this depth, so an
    incumbent reading 12 lags and a challenger reading 24 are scored on one
    byte-identical origin set instead of the challenger silently getting an
    easier one. Manifests without the key keep the historical 12-hour
    alignment, so old runs stay replayable; a run that wants recipes must
    freeze the wider alignment explicitly rather than have it widened for it.

    Because it is the depth the cohort *guarantees*, it is also the depth the
    optional numerical tools may advertise: see
    ``AgentModelTools.catalog``, which withholds the recipe tool rather than
    offer a seed this alignment could not serve at every origin. Parsing lives
    in ``origin_history_alignment_hours`` so the value-blind cohort planner
    reads the manifest exactly the way this evaluator does; only the
    predictor-compatibility check below is specific to a bound candidate.
    """
    alignment = origin_history_alignment_hours(task.metadata)
    if config.history_steps > alignment:
        raise ValueError(
            "predictor requires "
            f"{config.history_steps} hours of origin history but this task is "
            f"aligned to {alignment}; freeze "
            "origin_history_alignment_hours before binding this predictor"
        )
    return alignment


def _select_task_evaluation_cohort(
    task: TaskManifest,
    candidate: Candidate,
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_digest: str,
    split_manifest_digest: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int | None]:
    raw_scope = task.metadata.get("_evaluation_scope")
    raw_cohort = task.metadata.get("_planned_evaluation_cohort")
    if not isinstance(raw_scope, Mapping) or not isinstance(
        raw_cohort, Mapping
    ):
        raise ValueError("scoped evaluation requires scope and planned cohort")
    scope = EvaluationScope.from_dict(raw_scope)
    cohort = PlannedCohort.from_dict(raw_cohort)
    if (
        scope.run_id != candidate.run_id
        or scope.generation != candidate.generation
        or scope.candidate_id != candidate.candidate_id
        or scope.cohort_digest != cohort.cohort_digest
        or scope.origin_count != cohort.origin_count
    ):
        raise ValueError("planned cohort does not match EvaluationScope")
    cells_per_origin = task.metadata.get("prediction_cells_per_origin")
    selected, evidence = _select_planned_evaluation_cohort(
        rows,
        cohort,
        expected_prediction_cells_per_origin=int(cells_per_origin),
    )
    return selected, evidence, cohort.origin_count


def _bind_bundle_to_scope(
    bundle: "EvaluationBundle", scope: EvaluationScope | None
) -> "EvaluationBundle":
    if scope is None:
        return bundle
    artifact_data = bundle.artifact.to_dict()
    artifact_data.pop("artifact_digest", None)
    artifact_metrics = dict(bundle.artifact.metrics)
    artifact_metrics.update(
        {
            "candidate_revision_id": scope.candidate_revision_id,
            "execution_scope_digest": scope.scope_key,
            "evaluation_phase": scope.phase.value,
        }
    )
    artifact_data.update(
        {
            "candidate_revision_id": scope.candidate_revision_id,
            "evaluation_scope_digest": scope.scope_key,
            "metrics": artifact_metrics,
        }
    )
    artifact = ModelArtifact.from_dict(artifact_data)
    evaluation_data = bundle.evaluation.to_dict()
    metrics = dict(bundle.evaluation.metrics)
    sample_summary = metrics.get("sample_execution")
    prediction_cell_count = (
        int(sample_summary.get("attempted_examples", 0))
        if isinstance(sample_summary, Mapping)
        else len(bundle.sample_results or ())
    )
    metrics.update(
        {
            "evaluation_origin_count": scope.origin_count,
            "prediction_cell_count": prediction_cell_count,
            "feedback_update_cohort_digest": scope.cohort_digest,
            "execution_scope_digest": scope.scope_key,
            "evaluation_phase": scope.phase.value,
        }
    )
    evaluation_data.update(
        {
            "candidate_revision_id": scope.candidate_revision_id,
            "evaluation_scope": scope.to_dict(),
            "artifact_digest": artifact.digest,
            "metrics": metrics,
        }
    )
    return EvaluationBundle(
        artifact=artifact,
        evaluation=Evaluation.from_dict(evaluation_data),
        sample_results=bundle.sample_results,
    )


def _cohort_task_counts(
    evidence: Mapping[str, Any] | None,
    field: str,
) -> dict[tuple[str, int], int]:
    if evidence is None:
        return {}
    result: dict[tuple[str, int], int] = {}
    raw_tasks = evidence.get("tasks", ())
    if not isinstance(raw_tasks, list):
        return result
    for item in raw_tasks:
        if not isinstance(item, Mapping):
            continue
        target = item.get("target")
        horizon = item.get("horizon_hours")
        value = item.get(field)
        if (
            isinstance(target, str)
            and target
            and isinstance(horizon, int)
            and not isinstance(horizon, bool)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        ):
            result[(target, horizon)] = value
    return result


def _evaluation_index_digest(
    series: DatasetSeries,
    rows: list[dict[str, Any]],
) -> str:
    """Bind a score cohort to its frozen dataset and split snapshots."""

    return digest(
        {
            "dataset_digest": series.digest,
            "split_manifest_digest_sha256": series.split_manifest_digest_sha256,
            "partition": "training_feedback",
            "rows": rows,
        }
    )


def _baseline_metrics_digest(
    evaluation_index_digest: str,
    target_results: list[dict[str, Any]],
) -> str:
    return digest(
        {
            "evaluation_index_digest": evaluation_index_digest,
            "targets": [
                {
                    key: item[key]
                    for key in (
                        "target",
                        "horizon_hours",
                        "n",
                        "baseline_mae",
                        "baseline_rmse",
                        "baseline_normalized_rmse",
                    )
                }
                for item in target_results
            ],
        }
    )


def _apply_scoring_baseline(
    series: DatasetSeries,
    rows: Sequence[Mapping[str, Any]],
    baseline_profile: Mapping[str, Any],
    normalization_scales: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Apply the fit-selected comparator after model prediction is complete."""

    result = apply_baseline_profile(series, rows, baseline_profile)
    for index, row in enumerate(result):
        scale = normalization_scales.get(str(row.get("target")))
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0
        ):
            raise ValueError(f"scoring row {index} has no valid normalization scale")
        row["normalization_scale"] = float(scale)
    return result


def _complete_scoring_rows(
    expected_rows: Sequence[Mapping[str, Any]],
    returned_rows: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    target_bounds: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Close holes in an adapter result vector before durable checkpointing.

    Remote sample adapters may report a failed origin in their counters while
    omitting the corresponding prediction cells from the response.  The
    cohort is still complete: synthesize an explicit, non-scoring failure row
    for every omitted cell so the persisted archive can satisfy the frozen
    checkpoint cardinality and the coverage gate can account for the loss.
    """

    expected_by_id: dict[str, dict[str, Any]] = {}
    expected_order: list[str] = []
    for index, source in enumerate(expected_rows, start=1):
        row = dict(source)
        # Preserve the frozen cohort's original sample index.  Selection and
        # holdout cohorts are often sparse slices of the source dataset (for
        # example 19, 20, ...), and changing those indices would change the
        # deterministic sample_id of every already-persisted result.  Only
        # synthetic/unit-test rows without an index use the executor ordinal.
        sample_index = row.get("sample_index")
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index < 1
        ):
            row["sample_index"] = index
        target = str(row.get("target") or "").strip()
        raw_horizon = row.get("horizon_hours")
        if not target or isinstance(raw_horizon, bool) or not isinstance(raw_horizon, int):
            raise ValueError("evaluation cohort contains an invalid sample identity")
        sample_id = _sample_id(
            row,
            context,
            target=target,
            horizon=raw_horizon,
        )
        row["sample_id"] = sample_id
        if sample_id in expected_by_id:
            raise ValueError("evaluation cohort contains duplicate sample identities")
        expected_by_id[sample_id] = row
        expected_order.append(sample_id)

    returned_by_id: dict[str, dict[str, Any]] = {}
    for source in returned_rows:
        row = dict(source)
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError("sample executor returned a row without sample_id")
        if sample_id not in expected_by_id:
            raise ValueError("sample executor returned a row outside the frozen cohort")
        if sample_id in returned_by_id:
            raise ValueError("sample executor returned duplicate sample results")
        returned_by_id[sample_id] = row

    completed: list[dict[str, Any]] = []
    for sample_id in expected_order:
        row = returned_by_id.get(sample_id)
        if row is not None:
            completed.append(row)
            continue
        source = expected_by_id[sample_id]
        target = str(source["target"])
        bounds = target_bounds.get(target, {})
        baseline = _finite_registry_number(source.get("baseline"))
        if baseline is None:
            minimum = _finite_registry_number(bounds.get("minimum"))
            maximum = _finite_registry_number(bounds.get("maximum"))
            if minimum is None or maximum is None:
                raise ValueError("missing sample result has no finite fallback bounds")
            baseline = min(maximum, max(minimum, (minimum + maximum) / 2.0))
        completed.append(
            {
                **source,
                "predicted": baseline,
                "sample_execution_status": "failed",
                "sample_execution_attempts": 0,
                "sample_execution_retry_count": 0,
                "sample_execution_failure": {
                    "class": "missing_sample_result",
                    "retryable": True,
                    "error_type": "IncompleteSampleExecutorResponse",
                },
                "scoring_fallback": "missing_sample_result_penalty",
                "scoring_fallback_source": "persistence_baseline",
            }
        )
    return completed


def _per_cell_noninferiority(
    promotion_block_evidence: Mapping[str, Any],
    cell_metrics: Sequence[MutableMapping[str, Any]],
    adapter: DatasetAdapter | None,
    *,
    effective_degrees_of_freedom: int | None = None,
) -> CellNoninferiorityAssessment:
    """The one per-cell regression check, shared by both evaluation paths.

    Previously this was two byte-identical comprehensions a thousand lines apart
    -- a standing invitation for the single-horizon and multi-horizon paths to
    drift into judging candidates by different rules. There is now one call site
    per path and one implementation.

    It also stamps ``paired_block_count`` onto each scored cell row. That field
    is read by ``build_formal_fitness_assessment`` and until now had no producer
    anywhere in ``src/`` -- the formal tests passed only because a fixture filled
    it in by hand. The count is a by-product of the interval this function
    already computes, so writing it here costs nothing and removes one place
    where a test could pass on data production never emits.
    """

    assessment = assess_cell_noninferiority(
        promotion_block_evidence,
        cell_metrics=cell_metrics,
        policy=_noninferiority_policy(adapter),
        legacy_tolerance=(
            adapter.no_regression_tolerance
            if adapter is not None
            else GREENHOUSE_NO_REGRESSION_TOLERANCE
        ),
        effective_degrees_of_freedom=effective_degrees_of_freedom,
    )
    counts = {
        (str(cell["target"]), int(cell["horizon_hours"])): int(
            cell["paired_block_count"]
        )
        for cell in assessment.evidence["cells"]
    }
    for row in cell_metrics:
        key = (str(row.get("target")), row.get("horizon_hours"))
        if key in counts:
            row["paired_block_count"] = counts[key]
    return assessment


def _relaxation_debt(
    assessment: CellNoninferiorityAssessment,
    objective_score: Any,
    adapter: DatasetAdapter | None,
) -> dict[str, Any]:
    """Escalate the overall skill floor for a candidate that used the relaxation.

    The escalated floor is the same practical delta the selection layer already
    enforces, so this does not invent a threshold -- it reuses the one the
    project already treats as "a real gain" and applies it where the evaluator
    previously accepted 1e-9.
    """

    return assess_relaxation_debt(
        assessment,
        objective_score=objective_score,
        escalated_minimum_skill=(
            adapter.selection_minimum_score_delta
            if adapter is not None
            else V2_MINIMUM_SCORE_DELTA
        ),
    )






def _finite_registry_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _candidate_sample_max_attempts(proposal: Proposal | None) -> int | None:
    """Read one candidate's own requested retry depth, or ``None`` for silence.

    The value comes from the compiled ``candidate_agent_profile`` carried on the
    proposal, so rescoring an archived candidate reproduces the depth it ran at
    instead of whatever the current registry defaults to. Profiles archived
    before ``workflow_parameters`` existed simply have no opinion, and the host
    floor stands alone -- which is exactly the behaviour they were scored with.
    The value's range is enforced twice over: by the registry contract when the
    mutation was applied, and by ``SampleExecutionPolicy`` when it is used.
    """

    if proposal is None:
        return None
    profile = proposal.metadata.get("candidate_agent_profile")
    if not isinstance(profile, Mapping):
        return None
    parameters = profile.get("workflow_parameters")
    if not isinstance(parameters, Mapping):
        return None
    value = parameters.get("max_attempts")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("candidate workflow max_attempts must be an integer")
    return value


def _candidate_remote_critic_policy(
    task: TaskManifest, proposal: Proposal | None
) -> Mapping[str, Any] | None:
    """Overlay one candidate's escalation threshold on the run's critic policy.

    The run-level policy is frozen on the TaskManifest, which is per run and not
    per candidate, so the candidate's own threshold has to be applied here -- at
    the point where the adapter is constructed and the proposal is still in
    scope. What the candidate owns is exactly one number: how eagerly its planner
    hands a low-confidence decision to the remote critic. Everything else about
    the policy stays the host's.

    Three cases return the run policy untouched, and each is a boundary rather
    than a convenience:

    * ``None`` means "always escalate" downstream
      (``dsh_sample_adapter._require_remote_critic``). Synthesising a policy here
      would let a candidate switch independent review off, which is a scoring
      guarantee and not an evolvable knob.
    * ``always@1`` carries no threshold at all -- ``_normalized_remote_critic_policy``
      rejects the key -- so there is nothing to overlay.
    * a profile with no ``confidence_threshold`` (archived before this parameter
      had a consumer) keeps the threshold it was scored with.
    """

    policy = task.metadata.get("sample_remote_critic_policy")
    if not isinstance(policy, Mapping) or "min_planner_confidence" not in policy:
        return policy
    if proposal is None:
        return policy
    profile = proposal.metadata.get("candidate_agent_profile")
    if not isinstance(profile, Mapping):
        return policy
    parameters = profile.get("instruction_parameters")
    if not isinstance(parameters, Mapping):
        return policy
    threshold = parameters.get("confidence_threshold")
    if threshold is None:
        return policy
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("candidate confidence_threshold must be a number")
    # Left unclamped on purpose: the registry contract bounds it when the
    # mutation is applied, and `_normalized_remote_critic_policy` bounds it again
    # here. Silently clamping would hide a contract breach behind a legal-looking
    # run.
    return {**dict(policy), "min_planner_confidence": float(threshold)}


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    artifact: ModelArtifact
    evaluation: Evaluation
    # ``None`` means an evaluator does not implement the private full-result
    # contract. An empty tuple is a completed, supported evaluation with no
    # scoreable rows.
    sample_results: tuple[dict[str, Any], ...] | None = None



class EvaluationPipeline:
    """Execution services composed with the registered model contracts."""

    def __init__(
        self,
        datasets: DatasetRegistry,
        sample_executor: CollaborativeSampleExecutor | None = None,
        *,
        dsh_runtime_provider: Callable[[], Any] | None = None,
        dsh_revision_provider: Callable[[str], Mapping[str, int]] | None = None,
        dsh_identity_provider: Callable[..., Mapping[str, str] | None]
        | None = None,
        dsh_prediction_tool_binder: Callable[..., Any] | None = None,
        origin_admission_provider: Callable[[str, int], Any] | None = None,
        origin_admission_snapshot_provider: Callable[
            [str], Mapping[str, int]
        ]
        | None = None,
    ) -> None:
        self.datasets = datasets
        self._sample_executor_injected = sample_executor is not None
        self.sample_executor = sample_executor
        self.dsh_runtime_provider = dsh_runtime_provider
        self.dsh_revision_provider = dsh_revision_provider
        self.dsh_identity_provider = dsh_identity_provider
        self.dsh_prediction_tool_binder = dsh_prediction_tool_binder
        self.origin_admission_provider = origin_admission_provider
        self.origin_admission_snapshot_provider = (
            origin_admission_snapshot_provider
        )


    def _sample_executor_for_task(
        self,
        task: TaskManifest,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        on_sample_control: Callable[[], str] | None = None,
        forecast_tool: Callable[[SamplePredictionRequest], float] | None = None,
        forecast_bundle_tool: Callable[
            [Sequence[SamplePredictionRequest]], Mapping[str, Any]
        ]
        | None = None,
        agent_model_tools: AgentModelTools | None = None,
        run_id: str | None = None,
        candidate_id: str | None = None,
        proposal: Proposal | None = None,
    ) -> CollaborativeSampleExecutor:
        """Resolve the frozen sample runtime for the current task.

        ``proposal`` carries the compiled candidate agent profile. It is optional
        because the injected and host-feedback runtimes never consult it, but the
        model-backed adapters below do: the candidate's escalation threshold is
        per candidate while the TaskManifest is per run.
        """

        if self._sample_executor_injected:
            return self.sample_executor
        mode = str(
            task.metadata.get("sample_agent_mode")
            or "host_feedback_state_machine"
        )
        if mode != "dsh_native_agent":
            raise ValueError("Only dsh_native_agent sample execution is supported")
        if (
            not run_id
            or not candidate_id
            or self.dsh_runtime_provider is None
            or self.dsh_revision_provider is None
            or self.dsh_identity_provider is None
        ):
            raise ValueError("DSH-native sample runtime binding is incomplete")
        identity = self._dsh_execution_identity(task, run_id, candidate_id, proposal)
        strategy_model_id = str(task.metadata.get("strategy_model_id") or "").strip()
        review_model_id = str(task.metadata.get("review_model_id") or "").strip()
        if not strategy_model_id or not review_model_id:
            raise ValueError("DSH-native sample roles require frozen model routes")
        if (
            not is_strict_origin_protocol(
                task.metadata.get("sample_agent_protocol")
            )
        ):
            raise ValueError(
                "DSH-native sample execution requires the strict origin-bundle protocol"
            )
        if forecast_bundle_tool is None:
            raise ValueError(
                "DSH-native sample execution requires a registered vector prediction tool"
            )
        if self.dsh_prediction_tool_binder is None:
            raise ValueError(
                "DSH-native sample execution requires an agent prediction-tool binder"
            )
        raw_batch_size = task.metadata.get("sample_agent_batch_size", 128)
        raw_concurrency = task.metadata.get(
            "sample_concurrency",
            HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK,
        )
        if (
            isinstance(raw_batch_size, bool)
            or not isinstance(raw_batch_size, int)
            or not 1 <= raw_batch_size <= 128
        ):
            raise ValueError("DSH sample batch size must be between 1 and 128")
        if (
            isinstance(raw_concurrency, bool)
            or not isinstance(raw_concurrency, int)
            or not 1 <= raw_concurrency <= MAX_SAMPLE_CONCURRENCY
        ):
            raise ValueError(
                "DSH sample concurrency must be between 1 and "
                f"{MAX_SAMPLE_CONCURRENCY}"
            )
        adapter = DshSampleCollaborationAdapter(
            run_id=run_id,
            runtime_provider=self.dsh_runtime_provider,
            revision_provider=self.dsh_revision_provider,
            identity_digests={
                "genome_digest": str(identity["genome_digest"]),
                "compiled_behavior_digest": str(identity["compiled_behavior_digest"]),
                "phenotype_instance_digest": str(identity["phenotype_instance_digest"]),
            },
            strategy_model_id=strategy_model_id,
            review_model_id=review_model_id,
            forecast_bundle_tool=forecast_bundle_tool,
            prediction_tool_catalog=agent_model_tools.catalog() if agent_model_tools else (),
            prediction_tool_executor=agent_model_tools.execute if agent_model_tools else None,
            prediction_tool_binder=self.dsh_prediction_tool_binder,
            microbatch_size=min(
                128,
                max(
                    int(raw_batch_size),
                    int(task.metadata.get("prediction_cells_per_origin", 1)),
                ),
            ),
            sample_concurrency=int(raw_concurrency),
            progress_callback=progress_callback,
            admission_snapshot_provider=(
                (
                    lambda: self.origin_admission_snapshot_provider(run_id)
                )
                if self.origin_admission_snapshot_provider is not None
                else None
            ),
            run_control_callback=on_sample_control,
            remote_critic_policy=_candidate_remote_critic_policy(task, proposal),
            sample_reflection_policy=task.metadata.get(
                "sample_reflection_policy", "candidate_aggregate_post_score@1"
            ),
            sample_planner_prompt_profile=task.metadata.get(
                "sample_planner_prompt_profile"
            ),
            operation_max_tokens=task.metadata.get(
                "sample_operation_max_tokens"
            ),
        )
        return CollaborativeSampleExecutor(
            adapter,
            origin_admission=(
                (
                    lambda: self.origin_admission_provider(
                        run_id, int(raw_concurrency)
                    )
                )
                if self.origin_admission_provider is not None
                else None
            ),
        )


    def _dsh_execution_identity(self, task, run_id, candidate_id, proposal):
        scope = task.metadata.get("_evaluation_scope")
        if isinstance(scope, Mapping):
            if scope.get("run_id") != run_id or scope.get("candidate_id") != candidate_id:
                raise ValueError("DSH-native evaluation scope identity mismatch")
            identity = self.dsh_identity_provider(run_id, candidate_id, scope["candidate_revision_id"])
        else:
            identity = self.dsh_identity_provider(run_id, candidate_id)
        if not isinstance(identity, Mapping):
            raise ValueError("DSH-native candidate identity is unavailable")
        expected = proposal.metadata.get("genome_digest") if proposal is not None else None
        if expected is not None and identity.get("genome_digest") != expected:
            raise ValueError("DSH-native execution identity does not match evaluated genome")
        return {key: str(identity[key]) for key in (
            "genome_digest", "compiled_behavior_digest", "phenotype_instance_digest")}

    def _dsh_sample_stage_context(
        self,
        task: TaskManifest,
        candidate: Candidate,
        proposal: Proposal,
    ) -> dict[str, Any]:
        if task.metadata.get("execution_protocol") != "dsh_native_plugin_evolution@1":
            return {}
        names = (
            "dataset_snapshot_set_digest",
            "data_protocol_digest",
            "stage_policy_digest",
            "evaluation_cohort_digest",
            "fitness_profile_digest",
            "compiler_semantic_digest",
            "security_kernel_digest",
        )
        contracts = {
            name: str(task.metadata[name])
            for name in names
            if task.metadata.get(name) is not None
        }
        genome_digest = proposal.metadata.get("genome_digest")
        if not isinstance(genome_digest, str):
            raise ValueError("DSH sample execution requires candidate genome identity")
        agent_profile = proposal.metadata.get("candidate_agent_profile")
        if (
            not isinstance(agent_profile, Mapping)
            or agent_profile.get("schema_version")
            != "ecologyrsi-dsh.candidate-agent-profile/1"
            or agent_profile.get("role") != "sample-planner"
            or not isinstance(agent_profile.get("skill_name"), str)
            or not agent_profile["skill_name"]
        ):
            raise ValueError(
                "DSH sample execution requires a compiled candidate agent profile"
            )
        return {
            "stage_context_digest": sibling_stage_context_digest(
                task_manifest_digest=task.digest,
                generation=candidate.generation,
                frozen_contract_digests=contracts,
            ),
            "candidate_genome_digest": genome_digest,
            "candidate_agent_profile": dict(agent_profile),
            # Included in checkpoint and wave digests. An old misbound partial
            # revision is superseded, never resumed or relabelled as the child.
            "candidate_execution_identity": self._dsh_execution_identity(
                task, candidate.run_id, candidate.candidate_id, proposal
            ),
        }


    @staticmethod
    def _evaluation_scope_context(task: TaskManifest) -> dict[str, Any]:
        raw = task.metadata.get("_evaluation_scope")
        if raw is None:
            return {}
        scope = EvaluationScope.from_dict(raw)
        return {"evaluation_scope": scope.to_dict()}

    @staticmethod
    def _observed_edit_effects(task, proposal, records):
        raw_scope = task.metadata.get("_evaluation_scope")
        phase = raw_scope.get("phase", "training_feedback") if isinstance(raw_scope, Mapping) else "training_feedback"
        return observe_edit_effects(
            proposal.metadata, records,
            scope_digest=(EvaluationScope.from_dict(raw_scope).scope_key if isinstance(raw_scope, Mapping)
                          else digest({"task": task.digest, "proposal": proposal.proposal_id})),
            phase=phase, parameters=proposal.changes,
        )




    def evaluate_scientific(
        self,
        task: TaskManifest,
        candidate: Candidate,
        proposal: Proposal,
        *,
        scope: EvaluationScope | Mapping[str, Any] | None = None,
        cohort: PlannedCohort | Mapping[str, Any] | None = None,
        on_training_complete: Callable[[], None] | None = None,
        on_evaluation_progress: Callable[[Mapping[str, Any]], None] | None = None,
        on_sample_results: Callable[
            [Sequence[Mapping[str, Any]]], None
        ]
        | None = None,
        on_sample_checkpoint: Callable[
            [Mapping[str, Any]], Mapping[str, Any]
        ]
        | None = None,
        on_sample_control: Callable[[], str] | None = None,
        algorithm_spec: AlgorithmSpec | Mapping[str, Any] | None = None,
    ) -> EvaluationBundle:
        resolved_scope = (
            scope
            if isinstance(scope, EvaluationScope)
            else EvaluationScope.from_dict(scope)
            if isinstance(scope, Mapping)
            else None
        )
        resolved_cohort = (
            cohort
            if isinstance(cohort, PlannedCohort)
            else PlannedCohort.from_dict(cohort)
            if isinstance(cohort, Mapping)
            else None
        )
        adaptive = task.metadata.get("optimization_protocol") in ADAPTIVE_PROTOCOLS
        if adaptive and (resolved_scope is None or resolved_cohort is None):
            raise ValueError(
                "adaptive scientific evaluation requires scope and frozen cohort"
            )
        if (resolved_scope is None) != (resolved_cohort is None):
            raise ValueError("scope and frozen cohort must be supplied together")
        if resolved_scope is not None and resolved_cohort is not None:
            if (
                resolved_scope.run_id != candidate.run_id
                or resolved_scope.generation != candidate.generation
                or resolved_scope.candidate_id != candidate.candidate_id
                or resolved_scope.cohort_digest != resolved_cohort.cohort_digest
                or resolved_scope.origin_count != resolved_cohort.origin_count
            ):
                raise ValueError("EvaluationScope does not match candidate/cohort")
            task = replace(
                task,
                metadata={
                    **dict(task.metadata),
                    "_evaluation_scope": resolved_scope.to_dict(),
                    "_planned_evaluation_cohort": resolved_cohort.to_dict(),
                },
            )
        dataset_id = task.dataset
        if dataset_id is None:
            raise ValueError("task has no visible dataset")
        evaluator_id = str(
            task.metadata.get("evaluator_id") or self.default_evaluator(dataset_id)
        )
        resolved_algorithm_spec = (
            algorithm_spec
            if isinstance(algorithm_spec, AlgorithmSpec)
            else AlgorithmSpec.from_dict(algorithm_spec)
            if isinstance(algorithm_spec, Mapping)
            else None
        )
        predictor_model_id = str(
            resolved_algorithm_spec.adapter_id
            if resolved_algorithm_spec is not None
            else task.metadata.get("prediction_model_id")
            or self.default_predictor(dataset_id)
        )
        self.validate_binding(dataset_id, evaluator_id, predictor_model_id)
        adapter = dataset_adapter(dataset_id)
        if adapter is not None and task.metadata.get("dataset_task") is not None:
            if thaw_json(task.metadata["dataset_task"]) != adapter.contract():
                raise ValueError("frozen dataset task does not match its registered adapter")
        series = self._validated_series(task, dataset_id)
        if adapter is not None:
            adapter.validate_series(series)
        if resolved_algorithm_spec is not None:
            if resolved_algorithm_spec.evaluator_id != evaluator_id:
                raise ValueError(
                    "algorithm spec evaluator does not match the frozen task"
                )
            self._validate_algorithm_data_boundary(
                resolved_algorithm_spec,
                series,
            )
        execution_plan = self._resolve_execution_plan(
            candidate,
            proposal,
            resolved_algorithm_spec,
        )
        effect_resolution = proposal.metadata.get("effect_resolution", proposal.metadata.get("mutation_effect_resolution"))
        if isinstance(effect_resolution, Mapping):
            base_policy = SampleExecutionPolicy.from_mapping(task.metadata.get("sample_execution_policy"))
            runtime_resolution = verify_runtime_effects(effect_resolution, {
                "sample_max_attempts_floor": max(base_policy.max_attempts, execution_plan.sample_max_attempts),
                "remote_critic_policy": task.metadata.get("sample_remote_critic_policy"),
            })
            proposal = replace(proposal, metadata={**dict(proposal.metadata), "effect_resolution": runtime_resolution})
        if not (predictor_model_id in {BASELINE_ALIGNED_RIDGE_MODEL_ID, BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID, EXOGENOUS_RIDGE_MODEL_ID, TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID, HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID, RECIPE_RIDGE_MODEL_ID}):
            raise ValueError("Unsupported registered numerical predictor")
        horizons = tuple(next(item for item in self.catalog(dataset_id)
                              if item["id"] == evaluator_id)["horizons_hours"])
        bundle = self._evaluate_greenhouse_ridge(
            task,
            candidate,
            proposal,
            series,
            horizons=horizons,
            on_training_complete=on_training_complete,
            on_evaluation_progress=on_evaluation_progress,
            on_sample_results=on_sample_results,
            on_sample_checkpoint=on_sample_checkpoint,
            on_sample_control=on_sample_control,
            execution_plan=execution_plan,
            predictor_model_id=predictor_model_id,
        )
        return _bind_bundle_to_scope(bundle, resolved_scope)


    @staticmethod
    def _validate_algorithm_data_boundary(
        spec: AlgorithmSpec,
        series: DatasetSeries,
    ) -> None:
        if spec.dataset_digest is None or spec.split_manifest_digest is None:
            raise ValueError("algorithm spec is missing its frozen data boundary")
        if spec.dataset_digest != series.digest:
            raise ValueError("algorithm spec dataset snapshot does not match evaluation")
        if spec.split_manifest_digest != series.split_manifest_digest_sha256:
            raise ValueError("algorithm spec split snapshot does not match evaluation")
        if spec.allowed_partitions != EVOLUTION_ALLOWED_PARTITIONS:
            raise ValueError("algorithm spec requested a forbidden data partition")


    @staticmethod
    def _resolve_execution_plan(
        candidate: Candidate,
        proposal: Proposal,
        algorithm_spec: AlgorithmSpec | Mapping[str, Any] | None,
    ) -> DerivedExecutionPlan:
        raw_proposal_plan = proposal.metadata.get("derived_execution_plan")
        proposal_plan = (
            DerivedExecutionPlan.from_dict(raw_proposal_plan)
            if isinstance(raw_proposal_plan, Mapping)
            else derive_execution_plan(None)
        )
        if algorithm_spec is None:
            return proposal_plan
        spec = (
            algorithm_spec
            if isinstance(algorithm_spec, AlgorithmSpec)
            else AlgorithmSpec.from_dict(algorithm_spec)
        )
        if (
            spec.run_id != candidate.run_id
            or spec.generation != candidate.generation
            or spec.proposal_id != proposal.proposal_id
            or dict(spec.parameters) != dict(proposal.changes)
        ):
            raise ValueError("algorithm spec does not match the evaluation candidate")
        if spec.derived_execution_plan is None:
            if raw_proposal_plan is not None:
                raise ValueError("algorithm spec omitted the proposal execution plan")
            return proposal_plan
        spec_plan = DerivedExecutionPlan.from_dict(spec.derived_execution_plan)
        if spec_plan.to_dict() != proposal_plan.to_dict():
            raise ValueError("algorithm spec execution plan does not match the proposal")
        return spec_plan


    @staticmethod
    def _sample_execution_policy(
        task: TaskManifest,
        execution_plan: DerivedExecutionPlan,
        proposal: Proposal | None = None,
    ) -> SampleExecutionPolicy:
        base = SampleExecutionPolicy.from_mapping(
            task.metadata.get("sample_execution_policy")
        )
        adapter = dataset_adapter(task.dataset)
        coverage_floor = adapter.minimum_coverage if adapter else base.minimum_coverage
        # Two different things are being combined here, and keeping them named
        # apart is the whole point. The derived plan's depth is a *reliability
        # floor*: it exists so a generation that just saw transient failures
        # retries harder, and the host owns it. The candidate's value is *this
        # generation's requested retry depth* -- the one workflow parameter the
        # genome can move that anything reads. The floor may raise the request
        # but never lowers it, so `max()` is the synthesis, written explicitly
        # rather than left to a chain of `max()` calls where a candidate's
        # intent could be swallowed without a trace.
        transient_floor = max(base.max_attempts, execution_plan.sample_max_attempts)
        requested_attempts = _candidate_sample_max_attempts(proposal)
        # Prior-generation evidence may increase resilience, but it may never
        # weaken the immutable coverage thresholds chosen by the host.
        return SampleExecutionPolicy(
            max_attempts=(
                transient_floor
                if requested_attempts is None
                else max(requested_attempts, transient_floor)
            ),
            plan_max_attempts=max(
                base.plan_max_attempts, execution_plan.plan_max_attempts
            ),
            minimum_coverage=max(base.minimum_coverage, coverage_floor),
            minimum_task_coverage=max(base.minimum_task_coverage, coverage_floor),
            retry_backoff_seconds=max(
                base.retry_backoff_seconds,
                execution_plan.retry_backoff_seconds,
            ),
        )


    def _validated_series(
        self,
        task: TaskManifest,
        dataset_id: str,
    ) -> DatasetSeries:
        metadata = task.metadata
        dataset_digest = metadata.get("dataset_digest")
        split_digest = metadata.get("split_manifest_digest")
        if not isinstance(dataset_digest, str) or not dataset_digest.strip():
            raise ValueError("任务清单缺少服务端冻结的数据集快照校验值")
        if not isinstance(split_digest, str) or not split_digest.strip():
            raise ValueError("任务清单缺少服务端冻结的时间分区快照校验值")
        episode_id = metadata.get("episode_id")
        series = self.datasets.selection_view(
            dataset_id,
            str(episode_id) if episode_id is not None else None,
            expected_dataset_digest=dataset_digest,
            expected_split_manifest_digest=split_digest,
            expected_data_protocol_digest=metadata.get("data_protocol_digest"),
            target_names=dataset_adapter(dataset_id).target_names,
            horizons=dataset_adapter(dataset_id).horizons_hours,
            history_steps=int(metadata.get("history_steps", 3)),
        )
        resolved_digest = getattr(series, "digest", getattr(series, "dataset_digest", None))
        if resolved_digest != dataset_digest:
            raise ValueError("数据集快照发生漂移，候选训练已拒绝执行")
        if series.split_manifest_digest_sha256 != split_digest:
            raise ValueError("时间分区快照发生漂移，候选评测已拒绝执行")
        return series




    def _execute_feedback(
        self, task, candidate, proposal, series, rows, *, context, target_bounds,
        policy, execution_plan, feedback_update_cohort, samples_per_update,
        finalize_rows, on_sample_results, on_sample_checkpoint, **adapter_options,
    ):
        """Own feedback execution and publish each settled or synthesized row once."""
        def publish(result_rows):
            on_sample_results(finalize_rows(result_rows))

        metadata = task.metadata
        context = {
            "run_id": candidate.run_id,
            "candidate_id": candidate.candidate_id,
            **self._evaluation_scope_context(task),
            **self._dsh_sample_stage_context(task, candidate, proposal),
            "dataset_digest": series.digest,
            "split_manifest_digest_sha256": series.split_manifest_digest_sha256,
            "partition": "training_feedback",
            "proposal_plan": dict(proposal.metadata),
            "derived_execution_plan": execution_plan.to_dict(),
            "sample_agent_mode": metadata.get("sample_agent_mode", "host_feedback_state_machine"),
            "sample_agent_batch_size": metadata.get("sample_agent_batch_size", 128),
            "sample_concurrency": metadata.get("sample_concurrency", HISTORICAL_SAMPLE_CONCURRENCY_FALLBACK),
            "candidate_concurrency": metadata.get("candidate_concurrency", 1),
            "strategy_model_id": metadata.get("strategy_model_id"),
            "review_model_id": metadata.get("review_model_id"),
            "tool_experience": proposal.metadata.get("tool_experience", []),
            "agent_policy": proposal.metadata.get("agent_policy"),
            **context,
        }
        if feedback_update_cohort is not None:
            context["feedback_update_cohort"] = feedback_update_cohort
            if metadata.get("_evaluation_scope") is None:
                context["samples_per_update"] = samples_per_update
        if metadata.get("sample_planner_prompt_profile") is not None:
            context["sample_planner_prompt_profile"] = metadata["sample_planner_prompt_profile"]
        batch = self._sample_executor_for_task(
            task, run_id=candidate.run_id, candidate_id=candidate.candidate_id,
            proposal=proposal, **adapter_options,
        ).execute(
            rows, context=context, target_bounds=target_bounds,
            algorithm_id=context["algorithm_id"], algorithm_version=context["algorithm_version"],
            policy=policy, result_callback=publish if on_sample_results is not None else None,
            checkpoint_callback=on_sample_checkpoint,
        )
        completed = _complete_scoring_rows(
            rows, batch.scoring_rows,
            context={"candidate_id": candidate.candidate_id, "dataset_digest": series.digest},
            target_bounds=target_bounds,
        )
        if on_sample_results is not None:
            returned_ids = {str(row["sample_id"]) for row in batch.scoring_rows
                            if isinstance(row.get("sample_id"), str)}
            missing = [row for row in completed if str(row.get("sample_id")) not in returned_ids]
            if missing:
                publish(missing)
        return batch, finalize_rows(completed)




    def _evaluate_greenhouse_ridge(
        self,
        task: TaskManifest,
        candidate: Candidate,
        proposal: Proposal,
        series: DatasetSeries,
        *,
        horizons: tuple[int, ...],
        on_training_complete: Callable[[], None] | None = None,
        on_evaluation_progress: Callable[[Mapping[str, Any]], None] | None = None,
        on_sample_results: Callable[
            [Sequence[Mapping[str, Any]]], None
        ]
        | None = None,
        on_sample_checkpoint: Callable[
            [Mapping[str, Any]], Mapping[str, Any]
        ]
        | None = None,
        on_sample_control: Callable[[], str] | None = None,
        execution_plan: DerivedExecutionPlan | None = None,
        predictor_model_id: str = EXOGENOUS_RIDGE_MODEL_ID,
        frozen_artifact: ModelArtifact | None = None,
    ) -> EvaluationBundle:
        adapter = dataset_adapter(series.dataset_id)
        targets = adapter.target_bounds
        target_weights = adapter.target_weights
        execution_plan = execution_plan or self._resolve_execution_plan(
            candidate, proposal, None
        )
        parameters = (
            BaselineAlignedCellwiseRidgeConfig.from_mapping(proposal.changes)
            if predictor_model_id == BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID
            else BaselineAlignedRidgeConfig.from_mapping(proposal.changes)
            if predictor_model_id == BASELINE_ALIGNED_RIDGE_MODEL_ID
            else RecipeRidgeConfig.from_mapping(proposal.changes)
            if predictor_model_id == RECIPE_RIDGE_MODEL_ID
            else HorizonTargetwiseExogenousRidgeConfig.from_mapping(proposal.changes)
            if predictor_model_id == HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
            else TargetwiseExogenousRidgeConfig.from_mapping(proposal.changes)
            if predictor_model_id == TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
            else ExogenousRidgeConfig.from_mapping(proposal.changes)
        )
        history_alignment = _resolve_origin_history_alignment(task, parameters)
        defer_feedback_prediction = (
            not self._sample_executor_injected
            and task.metadata.get("sample_agent_mode")
            == "dsh_native_agent"
        )
        prediction = NativeBatchBackend().run(
            series,
            targets=tuple(item[0] for item in targets),
            horizons=horizons,
            config=parameters,
            evaluation_history_steps=history_alignment,
            defer_prediction_partitions=(
                ("training_feedback",) if defer_feedback_prediction else ()
            ),
            on_fit_complete=on_training_complete,
            control=on_sample_control,
        )
        if frozen_artifact is not None:
            from .greenhouse_prediction import fitted_model_identity
            # Formal labels are scored by the Host only. Do not send them
            # back to a post-score learning/reflection Agent.
            task = replace(task, metadata={**dict(task.metadata),
                "sample_reflection_policy": "disabled_for_independent_evaluation@1"})
            # Reconstruct numerical tools only from the original fit partition.
            # No Agent executes if coefficients or parameters drifted from the
            # selected artifact. Formal labels never participate in fitting.
            if (frozen_artifact.model_id != predictor_model_id
                    or thaw_json(frozen_artifact.parameters) != parameters.to_dict()
                    or fitted_model_identity(thaw_json(frozen_artifact.learned_parameters.get("models", ())))
                    != fitted_model_identity(prediction["models"])):
                raise ValueError("frozen candidate fit differs from independent evaluation tools")
        agent_model_tools = AgentModelTools(
            series, targets=tuple(item[0] for item in targets), horizons=horizons,
            control=on_sample_control,
            cache_dir=Path.home() / ".cache" / "ecologyrsi-dsh" / "agent-fits-v2",
            default_fit=prediction, default_config=parameters,
            # The same depth the cohort was aligned to, so what the tool
            # advertises is what the frozen origin set can serve.
            origin_history_alignment=history_alignment,
        ) if task.metadata.get("sample_agent_mode") == "dsh_native_agent" else None
        if frozen_artifact is not None and agent_model_tools is not None:
            policy = rebind_agent_policy(proposal.metadata.get("agent_policy"),
                genome_digest=proposal.metadata.get("genome_digest", "0" * 64),
                profile=proposal.metadata.get("candidate_agent_profile", {}), parameters=parameters.to_dict())
            if (frozen_artifact.learned_parameters.get("training_data_digest") != agent_model_tools.training_digest
                    or thaw_json(frozen_artifact.learned_parameters.get("agent_policy")) != policy):
                raise ValueError("frozen Agent policy or training data differs from independent evaluation")
        generated_rows = prediction["prediction_rows"]
        if agent_model_tools is not None:
            # Every Agent sees the same current observations regardless of its
            # candidate's optional numerical predictor or residual scale.
            for row in generated_rows:
                origin = row["origin_index"]
                row["label_free_context"]["available_observations"] = {
                    name: float(values[origin]) for name, values in series.values.items()
                    if values[origin] is not None and math.isfinite(float(values[origin]))
                }
        models = prediction["models"]
        fit_range = series.partitions["training_fit"]
        feedback_range = series.partitions["training_feedback"]
        target_metadata = {
            name: {"unit": unit, "minimum": minimum, "maximum": maximum}
            for name, unit, minimum, maximum in targets
        }
        algorithm_name, separator, algorithm_revision = (
            predictor_model_id.rpartition("@")
        )
        if not separator:
            algorithm_name = predictor_model_id
            algorithm_revision = "unversioned"
        sample_policy = self._sample_execution_policy(task, execution_plan, proposal)
        generated_feedback_rows = [
            {
                **row,
                "unit": target_metadata.get(str(row.get("target")), {}).get(
                    "unit", "unknown"
                ),
            }
            for row in generated_rows
            if row["partition"] == "training_feedback"
        ]
        generated_feedback_rows, feedback_update_cohort, samples_per_update = (
            _select_task_evaluation_cohort(
                task,
                candidate,
                generated_feedback_rows,
                dataset_digest=series.digest,
                split_manifest_digest=series.split_manifest_digest_sha256,
            )
        )
        generated_feedback_rows = [
            {
                **row,
                "sample_index": (
                    row.get("sample_index")
                    if isinstance(row.get("sample_index"), int)
                    and not isinstance(row.get("sample_index"), bool)
                    and row.get("sample_index") >= 1
                    else index
                ),
            }
            for index, row in enumerate(generated_feedback_rows, start=1)
        ]
        selected_task_counts = _cohort_task_counts(
            feedback_update_cohort, "selected_count"
        )
        available_task_counts = _cohort_task_counts(
            feedback_update_cohort, "population_count"
        )
        baseline_profile = fit_baseline_profile(
            series,
            targets=tuple(item[0] for item in targets),
            horizons=horizons,
        )
        baseline_scales = {
            target_name: _normalization_scale(
                tuple(
                    series.values[target_name][
                        fit_range.start : fit_range.end
                    ]
                )
            )[0]
            for target_name, _unit, _minimum, _maximum in targets
        }

        measurement_context = EvaluationContext.native(
            series, generated_feedback_rows,
            units={name: metadata['unit'] for name, metadata in target_metadata.items()},
            baseline_profile=baseline_profile, scales=baseline_scales,
            scoring_contract=_greenhouse_scoring_contract(adapter),
        )

        def finalize_scoring_rows(
            result_rows: Sequence[Mapping[str, Any]],
        ) -> list[dict[str, Any]]:
            return _apply_scoring_baseline(
                series,
                result_rows,
                baseline_profile,
                baseline_scales,
            )


        def candidate_forecast_tool(request: SamplePredictionRequest) -> float:
            return predict_fitted_exogenous_ridge(
                target=request.target,
                horizon_hours=request.horizon_hours,
                baseline=request.baseline,
                label_free_context=request.label_free_context,
                models=models,
                config=parameters,
            )

        def candidate_forecast_bundle_tool(
            requests: Sequence[SamplePredictionRequest],
        ) -> Mapping[str, Any]:
            """Execute the registered ridge once for one multi-output origin."""

            if not requests:
                raise ValueError("ridge origin bundle must not be empty")
            origin = requests[0].origin_timestamp
            if any(request.origin_timestamp != origin for request in requests):
                raise ValueError("ridge origin bundle mixes forecast origins")
            return {
                request.sample_id: {
                    "predicted": candidate_forecast_tool(request),
                    "metadata": {
                        "source_model_id": predictor_model_id,
                        "prediction_unit": (
                            "forecast_origin_with_target_horizon_vector"
                        ),
                        "origin_timestamp": origin,
                    },
                }
                for request in requests
            }


        sample_batch, scored_feedback_rows = self._execute_feedback(
            task, candidate, proposal, series, generated_feedback_rows,
            context={
                "algorithm_id": algorithm_name,
                "algorithm_version": algorithm_revision,
                "evaluator_id": str(task.metadata.get("evaluator_id") or RUNTIME_EVALUATOR_ID),
                "horizons_hours": list(horizons),
                "candidate_parameters": parameters.to_dict(),
                "algorithm_artifact_digest": digest({
                    "prediction_model_id": predictor_model_id,
                    "parameters": parameters.to_dict(), "models": models,
                    "dataset_digest": series.digest,
                }),
            },
            target_bounds=target_metadata,
            policy=sample_policy, execution_plan=execution_plan,
            feedback_update_cohort=feedback_update_cohort, samples_per_update=samples_per_update,
            finalize_rows=finalize_scoring_rows, on_sample_results=on_sample_results,
            on_sample_checkpoint=on_sample_checkpoint,
            progress_callback=on_evaluation_progress,
            on_sample_control=on_sample_control,
            agent_model_tools=agent_model_tools,
            forecast_tool=candidate_forecast_tool if defer_feedback_prediction else None,
            forecast_bundle_tool=candidate_forecast_bundle_tool if defer_feedback_prediction else None,
        )
        measurement_context.validate_rows(scored_feedback_rows, allow_missing=True)
        promotion_block_evidence = build_promotion_block_evidence(
            scored_feedback_rows,
            horizons=horizons,
            target_weights=target_weights,
            dataset_digest=series.digest,
            split_manifest_digest_sha256=series.split_manifest_digest_sha256,
            dataset_task=adapter.contract(),
        )
        rows = [
            row for row in generated_rows if row["partition"] == "training_fit"
        ] + scored_feedback_rows
        sample_records = list(sample_batch.records)
        task_results: list[dict[str, Any]] = []
        training_metrics: dict[str, Any] = {}
        preview_rows: list[dict[str, Any]] = []
        normalization_scales: dict[str, dict[str, Any]] = {}
        total_fit_eligible_rows = 0
        total_fit_rows = 0
        total_fit_missing_rows = 0
        evaluation_index_rows: list[dict[str, Any]] = []
        preview_per_task = max(
            1, _PREVIEW_ROWS_TOTAL // (len(targets) * len(horizons))
        )

        for target_name, _unit, _minimum, _maximum in targets:
            scale, scale_method = _normalization_scale(
                tuple(series.values[target_name][fit_range.start : fit_range.end])
            )
            normalization_scales[target_name] = {
                "scale": scale,
                "method": scale_method,
                "source_partition": "training_fit",
                "n": sum(
                    value is not None
                    and not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in series.values[target_name][
                        fit_range.start : fit_range.end
                    ]
                ),
            }
            for horizon in horizons:
                fit_rows = [
                    row
                    for row in rows
                    if row["partition"] == "training_fit"
                    and row["target"] == target_name
                    and row["horizon_hours"] == horizon
                ]
                feedback_rows_all = [
                    row
                    for row in rows
                    if row["partition"] == "training_feedback"
                    and row["target"] == target_name
                    and row["horizon_hours"] == horizon
                ]
                if not fit_rows:
                    raise ValueError(
                        f"training_fit has no usable rows for {target_name} at {horizon}h"
                    )
                fit_errors = [
                    float(row["predicted"]) - float(row["observed"]) for row in fit_rows
                ]
                fit_eligible_rows = _exact_time_eligible_rows(
                    series,
                    fit_range.start,
                    fit_range.end,
                    horizon,
                    int(parameters.history_steps),
                )
                fit_missing_rows = max(0, fit_eligible_rows - len(fit_rows))
                total_fit_eligible_rows += fit_eligible_rows
                total_fit_rows += len(fit_rows)
                total_fit_missing_rows += fit_missing_rows
                key = f"{target_name}_{horizon}h"
                training_metrics[f"{key}_rmse"] = _rmse(fit_errors)
                training_metrics[f"{key}_n"] = len(fit_rows)
                training_metrics[f"{key}_missing_or_nonfinite_rows"] = (
                    fit_missing_rows
                )
                eligible_rows = _exact_time_eligible_rows(
                    series,
                    feedback_range.start,
                    feedback_range.end,
                    horizon,
                    MAX_EXOGENOUS_RIDGE_HISTORY_STEPS,
                )
                partition_eligible_rows = eligible_rows
                available_rows = available_task_counts.get(
                    (target_name, horizon), eligible_rows
                )
                if feedback_update_cohort is not None:
                    eligible_rows = selected_task_counts.get(
                        (target_name, horizon), 0
                    )
                metadata = target_metadata[target_name]
                cell, feedback_rows = score_cell(
                    feedback_rows_all, target=target_name, horizon=horizon,
                    bounds=metadata, scale=scale, eligible_rows=eligible_rows,
                    partition_eligible_rows=partition_eligible_rows, available_rows=available_rows,
                    execution=execution_for_cell(sample_batch.summary, target_name, horizon),
                    minimum_coverage=sample_policy.minimum_task_coverage,
                    raw_diagnostics=True,
                )
                task_results.append(cell)
                evaluation_index_rows.extend(
                    {"target": target_name, "horizon_hours": horizon,
                     "origin_timestamp": row["origin_timestamp"],
                     "target_timestamp": row["timestamp"]}
                    for row in feedback_rows_all
                )
                for row in feedback_rows[:preview_per_task]:
                    preview_rows.append(
                        {
                            "origin_timestamp": row["origin_timestamp"],
                            "target_timestamp": row["timestamp"],
                            "timestamp": row["timestamp"],
                            "horizon_hours": horizon,
                            "observed": row["observed"],
                            "predicted": row["predicted"],
                            "baseline": row["baseline"],
                            "target": target_name,
                            "unit": metadata["unit"],
                            "sample_execution_status": row.get(
                                "sample_execution_status", "succeeded"
                            ),
                            "scoring_fallback": row.get("scoring_fallback"),
                        }
                    )

        scored_cells = [item for item in task_results if item["n"]]
        normalized_candidate = [item["normalized_rmse"] for item in scored_cells]
        normalized_baseline = [item["baseline_normalized_rmse"] for item in scored_cells]
        normalized_mean_rewards = [item["normalized_mean_reward"] for item in scored_cells]
        raw_normalized_mean_rewards = [item["raw_normalized_mean_reward"] for item in scored_cells]
        raw_mae = [item["mae"] for item in scored_cells]
        raw_rmse = [item["rmse"] for item in scored_cells]
        total_rows = sum(item["n"] for item in task_results)
        total_missing_rows = sum(item["missing_or_nonfinite_rows"] for item in task_results)
        constraint_violations = sum(item["constraint_violations"] for item in task_results)
        incomplete_prediction_tasks = len(task_results) - len(scored_cells)
        failed_cell_count = sum(item["failed_rows"] for item in task_results)
        raw_out_of_range_count = sum(item["raw_out_of_range_predictions"] for item in task_results)
        prediction_clipped_count = sum(item["prediction_clipped_count"] for item in task_results)

        objective_aggregate = _aggregate_greenhouse_objective(
            task_results,
            horizons,
            target_weights=target_weights,
        )
        if normalized_candidate:
            overall_nrmse: float | None = fmean(normalized_candidate)
            overall_baseline_nrmse: float | None = fmean(normalized_baseline)
            overall_skill = skill_score(
                overall_nrmse, overall_baseline_nrmse
            )
        else:
            overall_nrmse = None
            overall_baseline_nrmse = None
            overall_skill = -1.0
        cell_noninferiority = _per_cell_noninferiority(
            promotion_block_evidence, task_results, adapter
        )
        per_task_no_regression = cell_noninferiority.passed
        relaxation_debt = _relaxation_debt(
            cell_noninferiority, objective_aggregate["weighted_skill_score"], adapter
        )
        total_eligible_rows = total_rows + total_missing_rows
        sample_execution_coverage = (
            int(sample_batch.summary["succeeded_examples"]) / total_eligible_rows
            if total_eligible_rows
            else 0.0
        )
        per_task_coverage_pass = all(
            bool(item["sample_execution_coverage_pass"])
            for item in task_results
        )
        sample_execution_coverage_pass = (
            total_eligible_rows > 0
            and sample_execution_coverage >= sample_policy.minimum_coverage
            and per_task_coverage_pass
        )
        scientific_pass = (
            objective_aggregate["weighted_skill_score"]
            > adapter.minimum_skill
            and per_task_no_regression
            and relaxation_debt["passed"]
            and constraint_violations <= adapter.maximum_constraint_violations
            and sample_execution_coverage_pass
            and successful_agent_provenance_passes(sample_batch.summary)
        )
        horizon_results = []
        for horizon in horizons:
            selected = [
                item for item in task_results if item["horizon_hours"] == horizon
            ]
            scored = [item for item in selected if item["normalized_rmse"] is not None]
            horizon_nrmse = (
                fmean(item["normalized_rmse"] for item in scored)
                if scored
                else None
            )
            horizon_baseline = (
                fmean(item["baseline_normalized_rmse"] for item in scored)
                if scored
                else None
            )
            horizon_normalized_reward = (
                fmean(item["normalized_mean_reward"] for item in scored)
                if scored
                else None
            )
            horizon_eligible = sum(item["eligible_rows"] for item in selected)
            horizon_used = sum(item["n"] for item in selected)
            horizon_results.append(
                {
                    "horizon_hours": horizon,
                    "normalized_rmse": horizon_nrmse,
                    "baseline_normalized_rmse": horizon_baseline,
                    "skill_score": (
                        skill_score(horizon_nrmse, horizon_baseline)
                        if horizon_nrmse is not None and horizon_baseline is not None
                        else -1.0
                    ),
                    "normalized_mean_reward": horizon_normalized_reward,
                    "negative_reward_task_count": sum(
                        item.get("normalized_mean_reward") is not None
                        and float(item["normalized_mean_reward"]) < 0.0
                        for item in selected
                    ),
                    "n": horizon_used,
                    "constraint_violations": sum(
                        item["constraint_violations"] for item in selected
                    ),
                    "sample_execution_coverage": (
                        horizon_used / horizon_eligible if horizon_eligible else 0.0
                    ),
                    "sample_execution_coverage_pass": all(
                        item["sample_execution_coverage_pass"] for item in selected
                    ),
                }
            )

        sample_execution_summary = dict(sample_batch.summary)
        effect_receipt = self._observed_edit_effects(task, proposal, sample_records)
        if effect_receipt is not None:
            sample_execution_summary["mutation_effect_receipt"] = effect_receipt
        sample_execution_summary["tool_performance"] = summarize_tool_performance(
            sample_records,
            scored_feedback_rows,
        )
        sample_execution_summary["agent_tool_performance"] = summarize_agent_tools(
            sample_records, scored_feedback_rows,
            source_phase=(task.metadata.get("_evaluation_scope") or {}).get("phase", "training_feedback"),
        )
        from ..evolution.diversity import enabled as diversity_enabled, summarize_behavior
        if diversity_enabled(task.metadata):
            sample_execution_summary["observed_behavior"] = summarize_behavior(sample_records, scored_feedback_rows)
        from ..core.prediction_policy import CONTRIBUTION_POLICIES
        contribution_policy = task.metadata.get("behavior_diagnostics_policy")
        if contribution_policy in CONTRIBUTION_POLICIES:
            from .prediction_contribution import summarize_contribution
            scope = task.metadata.get("_evaluation_scope") or {}
            sample_execution_summary["prediction_contribution"] = summarize_contribution(
                sample_records, scored_feedback_rows, cohort_digest=scope.get("cohort_digest"), source_phase=scope.get("phase"),
                review_policy=_candidate_remote_critic_policy(task, proposal), policy=contribution_policy)
        sample_execution_summary.update(
            {
                "result_vector_expected_examples": len(generated_feedback_rows),
                "result_vector_returned_examples": len(sample_batch.scoring_rows),
                "result_vector_missing_examples": max(
                    0, len(generated_feedback_rows) - len(sample_batch.scoring_rows)
                ),
                "adapter_attempt_coverage": sample_batch.summary["coverage"],
                "eligible_examples": total_eligible_rows,
                "succeeded_examples": int(
                    sample_batch.summary["succeeded_examples"]
                ),
                "scored_examples": total_rows,
                "failed_examples": int(
                    sample_batch.summary["failed_examples"]
                ),
                "skipped_examples": total_missing_rows,
                "input_unavailable_examples": max(
                    0, total_eligible_rows - len(sample_records)
                ),
                "coverage": sample_execution_coverage,
                "coverage_pass": sample_execution_coverage_pass,
                "per_task_coverage_pass": per_task_coverage_pass,
                "incomplete_prediction_tasks": incomplete_prediction_tasks,
                **(
                    {
                        "samples_per_update": samples_per_update,
                        "feedback_population_examples": int(
                            feedback_update_cohort["population_count"]
                        ),
                        "feedback_selected_examples": int(
                            feedback_update_cohort["selected_count"]
                        ),
                        "feedback_deferred_examples": int(
                            feedback_update_cohort["deferred_count"]
                        ),
                        "feedback_update_cohort_digest": feedback_update_cohort[
                            "cohort_digest"
                        ],
                    }
                    if feedback_update_cohort is not None
                    else {}
                ),
            }
        )

        evaluator_id = str(task.metadata.get("evaluator_id") or RUNTIME_EVALUATOR_ID)
        evaluator_digest = str(
            task.metadata.get("evaluator_digest")
            or self.evaluator_configuration_digest(evaluator_id, series.dataset_id)
        )
        evaluation_index_digest = _evaluation_index_digest(
            series, evaluation_index_rows
        )
        evaluation_metrics = {
            "measurement_context": measurement_context.summary(),
            "numerical_backend": NativeBatchBackend.version,
            "dataset_task": adapter.contract(),
                "objective_profile": GREENHOUSE_OBJECTIVE_PROFILE_ID,
            "objective_aggregation_version": (
                GREENHOUSE_OBJECTIVE_AGGREGATION_VERSION
            ),
            "objective_target_weights": dict(target_weights),
            "objective_horizons": list(horizons),
            # Explicit objective fields are additive. ``skill_score`` and
            # ``mean_normalized_reward`` retain their historical semantics.
            "objective_score": objective_aggregate["weighted_skill_score"],
            "primary_fitness": objective_aggregate["weighted_skill_score"],
            "primary_fitness_definition": GREENHOUSE_PRIMARY_FITNESS_DEFINITION,
            "overall_reward": objective_aggregate[
                "weighted_normalized_mean_reward"
            ],
            "auxiliary_mae_reward": objective_aggregate[
                "weighted_normalized_mean_reward"
            ],
            "auxiliary_mae_reward_definition": SAMPLE_REWARD_DEFINITION,
            **objective_aggregate,
            "normalization_scale_method": NORMALIZATION_SCALE_METHOD,
            "normalization_scales": normalization_scales,
            "baseline_profile": baseline_profile,
            "baseline_profile_digest": baseline_profile["digest"],
            "baseline_selection_partition": "training_fit",
            "promotion_block_evidence": promotion_block_evidence,
            "cell_noninferiority_evidence": cell_noninferiority.evidence,
            "gate_relaxation_audit": cell_noninferiority.audit,
            "noninferiority_relaxation_debt": relaxation_debt,
            "mean_target_mae_unscaled": fmean(raw_mae) if raw_mae else None,
            "mean_target_rmse_unscaled": fmean(raw_rmse) if raw_rmse else None,
            "raw_units_comparable_across_targets": False,
            "normalized_rmse": overall_nrmse,
            "baseline_normalized_rmse": overall_baseline_nrmse,
            "skill_score": overall_skill,
            "reward_definition": SAMPLE_REWARD_DEFINITION,
            "positive_reward_is_better": True,
            "mean_normalized_reward": (
                fmean(normalized_mean_rewards)
                if normalized_mean_rewards
                else None
            ),
            "mean_raw_normalized_reward": (
                fmean(raw_normalized_mean_rewards)
                if raw_normalized_mean_rewards
                else None
            ),
            "improvement": (
                overall_baseline_nrmse - overall_nrmse
                if overall_nrmse is not None
                and overall_baseline_nrmse is not None
                else None
            ),
            "n": total_rows,
            "evaluation_rows": total_rows,
            "execution_mode": "closed_form_ridge",
            "fit_method": "closed_form_ridge",
            "evaluation_partition_rows": feedback_range.size,
            "evaluation_eligible_examples": total_rows + total_missing_rows,
            "evaluation_used_examples": total_rows,
            "evaluation_skipped_examples": total_missing_rows,
            **(
                {
                    "evaluation_partition_eligible_examples": sum(
                        int(item["partition_eligible_rows"])
                        for item in task_results
                    ),
                    "evaluation_available_examples": int(
                        feedback_update_cohort["population_count"]
                    ),
                    "evaluation_selected_examples": int(
                        feedback_update_cohort["selected_count"]
                    ),
                    "evaluation_deferred_examples": int(
                        feedback_update_cohort["deferred_count"]
                    ),
                    "samples_per_update": samples_per_update,
                    "feedback_update_cohort_digest": feedback_update_cohort[
                        "cohort_digest"
                    ],
                    "feedback_update_window_offset": feedback_update_cohort[
                        "window_offset"
                    ],
                    "feedback_update_population_count": feedback_update_cohort[
                        "population_count"
                    ],
                }
                if feedback_update_cohort is not None
                else {}
            ),
            "missing_or_nonfinite_rows": total_missing_rows,
            "sample_execution": sample_execution_summary,
            "sample_execution_records": bounded_sample_execution_records(sample_records),
            "sample_execution_trace_archive": encode_sample_execution_trace(
                sample_records
            ),
            "sample_execution_trace_digest": sample_batch.summary["trace_digest"],
            "sample_execution_trace_record_count": len(sample_records),
            "sample_execution_coverage": sample_execution_coverage,
            "sample_execution_coverage_pass": sample_execution_coverage_pass,
            "sample_execution_failed_examples": int(
                sample_batch.summary["failed_examples"]
            ),
            "evaluation_scoring_fallback_examples": int(
                sample_batch.summary["scoring_fallback_examples"]
            ),
            "constraint_violations": constraint_violations,
            "failed_cell_count": failed_cell_count,
            "raw_out_of_range_count": raw_out_of_range_count,
            "prediction_clipped_count": prediction_clipped_count,
            "per_target_no_regression": per_task_no_regression,
            "scientific_pass": scientific_pass,
            "targets": task_results,
            "horizons": horizon_results,
            "prediction_preview": preview_rows[:_PREVIEW_ROWS_TOTAL],
            "evaluation_cohort": {
                "partition": "training_feedback",
                "history_source_partitions": ["training_feedback"],
                **({
                    "baseline_history_source_partitions": ["training_fit", "training_feedback"],
                    "baseline_reference_cutoff": "forecast_origin",
                } if predictor_model_id in (BASELINE_ALIGNED_RIDGE_MODEL_ID, BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID) else {}),
                "minimum_history_steps": MAX_EXOGENOUS_RIDGE_HISTORY_STEPS,
                "embargo_rows_visible": False,
                **(
                    {"update_window": feedback_update_cohort}
                    if feedback_update_cohort is not None
                    else {}
                ),
            },
            "evaluation_index_digest": evaluation_index_digest,
            "baseline_metrics_digest": _baseline_metrics_digest(
                evaluation_index_digest, task_results
            ),
            "dataset_digest": series.digest,
            "split_manifest_digest_sha256": series.split_manifest_digest_sha256,
            "prediction_model_id": predictor_model_id,
            "prediction_usage": ({"mode": "agent_per_origin", "default_model_id": predictor_model_id,
                                  "numerical_prediction_owner": "sample_agent"}
                                 if agent_model_tools is not None else prediction_usage(predictor_model_id, parameters.to_dict())),
            "evaluation_scope": "visible/training_feedback/historical_replay",
            "causal_interpretation": False,
        }
        if agent_model_tools is not None:
            evaluation_metrics["prediction_owner"] = "sample_agent"
            evaluation_metrics["agent_inference_stability_status"] = "requires_independent_holdout_replicas"
        baseline_only = all(item["status"] == "baseline_only" for item in models)
        fit_passes = int(not baseline_only)
        if agent_model_tools is not None:
            training_metrics.update({
                "prediction_role": "optional_candidate_model_tool",
                "final_prediction_owner": "sample_agent",
                "standalone_artifact_reproduces_agent_predictions": False,
            })
        training_metrics.update(
            {
                "model_task_count": len(models),
                "solver_fallback_count": sum(
                    item["status"] not in {"fitted", "baseline_only"} for item in models
                ),
                "selected_exogenous_feature_count": len(
                    {
                        name
                        for item in models
                        for name in item["selected_exogenous_features"]
                    }
                ),
                "execution_mode": "baseline_only" if baseline_only else "closed_form_ridge",
                "fit_method": "training_fit_baseline_selection" if baseline_only else "closed_form_ridge",
                "residual_models_fitted": sum(item["status"] == "fitted" for item in models),
                "baseline_only_task_count": sum(item["status"] == "baseline_only" for item in models),
                "fit_passes_requested": fit_passes,
                "fit_passes_completed": fit_passes,
                "iterative_epoch_training": False,
                "training_rows": fit_range.size,
                "training_partition_rows": fit_range.size,
                "training_eligible_examples": total_fit_eligible_rows,
                "training_used_examples": total_fit_rows,
                "training_skipped_examples": total_fit_missing_rows,
            }
        )
        delivery_policy = rebind_agent_policy(proposal.metadata.get("agent_policy"),
            genome_digest=proposal.metadata.get("genome_digest", "0" * 64),
            profile=proposal.metadata.get("candidate_agent_profile", {}), parameters=parameters.to_dict()) if agent_model_tools is not None else None
        artifact = ModelArtifact(
            artifact_id=f"artifact:{candidate.candidate_id}",
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            model_id=predictor_model_id,
            dataset_digest=series.digest,
            training_partition="training_fit",
            training_rows=fit_range.size,
            parameters=parameters.to_dict(),
            learned_parameters={
                "feature_policy": prediction["feature_policy"],
                "models": models,
                **({"agent_policy": delivery_policy, "optional_tool_catalog": agent_model_tools.catalog(),
                    "training_data_digest": agent_model_tools.training_digest,
                    # The alignment the catalog above was narrowed to. A
                    # redeployed policy has to advertise the same servable
                    # depth, or its tool contract check fails against a
                    # grammar it never saw.
                    "origin_history_alignment_hours": history_alignment,
                    "runtime_contract": {key: task.metadata.get(key) for key in ("strategy_model_id", "review_model_id", "preset_content_digest", "standing_tool_surface_digest", "resolved_policy_route_config_digest", "resolved_review_route_config_digest")}}
                   if agent_model_tools is not None else {}),
                **({"baseline_profile": prediction["baseline_profile"]}
                   if predictor_model_id in (BASELINE_ALIGNED_RIDGE_MODEL_ID, BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID) else {}),
            },
            metrics=training_metrics,
        )
        evaluation = Evaluation(
            evaluation_id=f"evaluation:{candidate.candidate_id}:training_feedback",
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            # Promotion/ranking must use the same explicit target/horizon
            # weighting exposed in ``objective_score``.  The unweighted
            # RMSE skill remains available as ``metrics.skill_score`` for the
            # scientific gate and backwards-compatible diagnostics.
            score=objective_aggregate["weighted_skill_score"],
            passed=scientific_pass,
            metrics=evaluation_metrics,
            partition="training_feedback",
            evaluator_digest=evaluator_digest,
            artifact_digest=artifact.digest,
        )
        return EvaluationBundle(
            artifact=artifact,
            evaluation=evaluation,
            sample_results=build_sample_results(
                candidate.candidate_id, scored_feedback_rows
            ),
        )
