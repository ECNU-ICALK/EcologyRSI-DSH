"""Small evaluator registry for toy and observed greenhouse time series.

The greenhouse evaluator fits only on ``training_fit`` and scores only on the
later ``training_feedback`` range.  Development, gate, and external holdout
rows never enter this local adaptive loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from ..core.models import (
    Candidate,
    Evaluation,
    ModelArtifact,
    Proposal,
    TaskManifest,
    digest,
)
from ..core.sample_results import SAMPLE_REWARD_DEFINITION, build_sample_results
from ..data.registry import DatasetRegistry, DatasetSeries
from ..data.toy import ToyCropSoilWater
from ..evolution.execution_plan import DerivedExecutionPlan, derive_execution_plan
from ..evolution.promotion import (
    PROMOTION_BLOCK_EVIDENCE_VERSION,
    PROMOTION_BLOCK_HOURS,
    PROMOTION_BOOTSTRAP_RESAMPLES,
    PROMOTION_CONFIDENCE_LEVEL,
    PROMOTION_CONFIDENCE_METHOD,
    PROMOTION_MAXIMUM_BLOCKS,
    PROMOTION_MINIMUM_PAIRED_BLOCKS,
    PROMOTION_POLICY_VERSION,
    V2_MINIMUM_SCORE_DELTA,
    build_promotion_block_evidence,
)
from ..integrations.model_bindings import RULE_JUDGE_ID
from ..integrations.model_gateway import ModelGateway
from ..knowledge.algorithms import (
    EVOLUTION_ALLOWED_PARTITIONS,
    AlgorithmSpec,
    PredictorAdoption,
)
from .gateway_sample_adapter import (
    GatewaySampleCollaborationAdapter,
    GatewaySampleTool,
)
from .dsh_sample_adapter import DshSampleCollaborationAdapter
from .shared_sample_context import sibling_stage_context_digest
from .baselines import (
    BASELINE_PROFILE_VERSION,
    BASELINE_SELECTION_TOLERANCE,
    apply_baseline_profile,
    fit_baseline_profile,
)
from .greenhouse_prediction import (
    EXOGENOUS_RIDGE_MODEL_ID,
    HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    MAX_EXOGENOUS_RIDGE_HISTORY_STEPS,
    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    ExogenousRidgeConfig,
    HorizonTargetwiseExogenousRidgeConfig,
    TargetwiseExogenousRidgeConfig,
    fit_predict_exogenous_ridge,
    predict_fitted_exogenous_ridge,
)
from .metrics import (
    MAX_ROLLING_WINDOW_HOURS,
    NORMALIZATION_SCALE_METHOD,
    _exact_time_eligible_rows,
    _clip_normalized_objective,
    _fit_bias,
    _greenhouse_parameters,
    _judge_metrics,
    _mae,
    _normalized_absolute_error_reward,
    _normalization_scale,
    _rmse,
    _rolling_eligible_rows,
    _skill_score,
    _standard_deviation,
    artifact_set_digest,
)
from .objectives import (
    DEFAULT_TARGET_WEIGHTS,
    OBJECTIVE_AGGREGATION_VERSION,
    OBJECTIVE_COMPONENT_BOUND,
    OBJECTIVE_MISSING_PENALTY,
    aggregate_greenhouse_objective,
)
from .sample_execution import (
    DEFAULT_SAMPLE_EXECUTION_MIN_COVERAGE,
    CollaborativeSampleExecutor,
    RegisteredToolCollaborationAdapter,
    SampleExecutionPolicy,
    SamplePredictionRequest,
    bounded_sample_execution_records,
    encode_sample_execution_trace,
)

TOY_DATASET_ID = "generated-toy-series@1"
GREENHOUSE_EVALUATOR_ID = "greenhouse_time_forward@1"
GREENHOUSE_MULTIHORIZON_EVALUATOR_ID = "greenhouse_multihorizon_time_forward@1"
GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID = (
    "greenhouse_multihorizon_time_forward@2"
)
TOY_EVALUATOR_ID = "toy_time_forward@1"
TOY_PREDICTOR_MODEL_ID = "toy-rolling-water@1"
GREENHOUSE_ROLLING_PREDICTOR_ID = "greenhouse-rolling-residual@1"
GREENHOUSE_OBJECTIVE_PROFILE_ID = "greenhouse_equal_weight_skill@1"
# The profile keeps RMSE skill as the canonical selection objective.  Reward is
# aggregated alongside it as a normalized, auditable learning signal; both
# components are versioned so historical evaluations remain interpretable.
GREENHOUSE_OBJECTIVE_AGGREGATION_VERSION = OBJECTIVE_AGGREGATION_VERSION
GREENHOUSE_OBJECTIVE_MISSING_PENALTY = OBJECTIVE_MISSING_PENALTY
GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS = dict(DEFAULT_TARGET_WEIGHTS)
GREENHOUSE_MIN_SKILL_EXCLUSIVE = 1e-9
GREENHOUSE_NO_REGRESSION_TOLERANCE = 1e-12
GREENHOUSE_MAX_CONSTRAINT_VIOLATIONS = 0

_TARGETS = (
    ("air_temperature", "degC", -10.0, 60.0),
    ("relative_humidity", "percent", 0.0, 100.0),
    ("co2_concentration", "ppm", 0.0, 5000.0),
)
_PREVIEW_ROWS_PER_TARGET = 16
_PREVIEW_ROWS_TOTAL = 48
_FEEDBACK_UPDATE_COHORT_SCHEMA_VERSION = (
    "ecologyrsi-dsh.feedback-update-cohort/1"
)


def _aggregate_greenhouse_objective(
    task_results: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    *,
    missing_skill_penalty: float = GREENHOUSE_OBJECTIVE_MISSING_PENALTY,
    missing_reward_penalty: float = GREENHOUSE_OBJECTIVE_MISSING_PENALTY,
) -> dict[str, Any]:
    """Compatibility wrapper around the pure, versioned aggregator."""

    return aggregate_greenhouse_objective(
        task_results,
        horizons,
        target_weights=GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS,
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


def _select_feedback_update_cohort(
    rows: Sequence[Mapping[str, Any]],
    *,
    generation: int,
    samples_per_update: int,
    dataset_digest: str,
    split_manifest_digest: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one deterministic, task-balanced, rotating feedback window.

    The ordering and digest deliberately exclude observations, predictions,
    candidate ids, and candidate parameters. Thus sibling candidates in one
    generation receive exactly the same target/horizon/timestamp identities.
    A contiguous window over the interleaved population advances by
    ``samples_per_update`` each generation and wraps only after reaching the
    end of the frozen population.
    """

    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    if (
        isinstance(samples_per_update, bool)
        or not isinstance(samples_per_update, int)
        or samples_per_update < 1
    ):
        raise ValueError("samples_per_update must be a positive integer")

    grouped: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    seen_identities: set[str] = set()
    for raw_row in rows:
        row = dict(raw_row)
        identity = _feedback_sample_identity(row)
        identity_digest = digest(identity)
        if identity_digest in seen_identities:
            raise ValueError("feedback update population contains duplicate sample identities")
        seen_identities.add(identity_digest)
        key = (str(identity["target"]), int(identity["horizon_hours"]))
        grouped.setdefault(key, []).append((row, identity))

    for values in grouped.values():
        values.sort(
            key=lambda item: (
                _feedback_timestamp_sort_key(item[1]["target_timestamp"]),
                _feedback_timestamp_sort_key(item[1]["origin_timestamp"]),
            )
        )

    task_keys = sorted(grouped)
    population: list[tuple[dict[str, Any], dict[str, Any]]] = []
    max_task_size = max((len(grouped[key]) for key in task_keys), default=0)
    for task_offset in range(max_task_size):
        for key in task_keys:
            task_rows = grouped[key]
            if task_offset < len(task_rows):
                population.append(task_rows[task_offset])

    population_count = len(population)
    selected_count = min(samples_per_update, population_count)
    if population_count == 0:
        window_offset = 0
        window_cycle = 0
        selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    elif samples_per_update >= population_count:
        # A limit at or above the population means a stable full cohort. Do
        # not reorder it between generations, which also makes restart replay
        # and comparisons with full-partition runs straightforward.
        window_offset = 0
        window_cycle = generation
        selected = list(population)
    else:
        absolute_offset = generation * samples_per_update
        window_offset = absolute_offset % population_count
        window_cycle = absolute_offset // population_count
        selected = [
            population[(window_offset + index) % population_count]
            for index in range(selected_count)
        ]

    available_by_task = {
        key: len(grouped[key]) for key in task_keys
    }
    selected_by_task = {key: 0 for key in task_keys}
    for _row, identity in selected:
        key = (str(identity["target"]), int(identity["horizon_hours"]))
        selected_by_task[key] += 1

    selected_identities = [identity for _row, identity in selected]
    population_digest = digest(
        {
            "schema_version": _FEEDBACK_UPDATE_COHORT_SCHEMA_VERSION,
            "selection_policy": "target_horizon_interleaved_rotating_window@1",
            "dataset_digest": dataset_digest,
            "split_manifest_digest": split_manifest_digest,
            "rows": [identity for _row, identity in population],
        }
    )
    cohort_digest = digest(
        {
            "schema_version": _FEEDBACK_UPDATE_COHORT_SCHEMA_VERSION,
            "dataset_digest": dataset_digest,
            "split_manifest_digest": split_manifest_digest,
            "samples_per_update": samples_per_update,
            "population_count": population_count,
            "window_offset": window_offset,
            "rows": selected_identities,
        }
    )
    evidence = {
        "schema_version": _FEEDBACK_UPDATE_COHORT_SCHEMA_VERSION,
        "selection_policy": "target_horizon_interleaved_rotating_window@1",
        "generation": generation,
        "samples_per_update": samples_per_update,
        "population_count": population_count,
        "population_digest": population_digest,
        "selected_count": selected_count,
        "deferred_count": max(0, population_count - selected_count),
        "window_offset": window_offset,
        "window_cycle": window_cycle,
        "window_wraps": (
            population_count > 0
            and window_offset + selected_count > population_count
        ),
        "cohort_digest": cohort_digest,
        "tasks": [
            {
                "target": target,
                "horizon_hours": horizon,
                "population_count": available_by_task[(target, horizon)],
                "selected_count": selected_by_task[(target, horizon)],
            }
            for target, horizon in task_keys
        ],
    }
    return [row for row, _identity in selected], evidence


def _feedback_update_limit(task: TaskManifest) -> int | None:
    raw = task.metadata.get("samples_per_update")
    if raw is None:
        # Existing manifests retain their historical full-partition ordering
        # and checkpoint identity. Newly bound real autonomous runs always
        # freeze an explicit value.
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError("samples_per_update must be a positive integer")
    return raw


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


def _greenhouse_hard_gates() -> list[dict[str, Any]]:
    """Describe the exact evaluator checks in a machine-readable form."""

    return [
        {
            "id": "positive_overall_skill",
            "scope": "overall",
            "metric": "objective_score",
            "operator": ">",
            "threshold": GREENHOUSE_MIN_SKILL_EXCLUSIVE,
        },
        {
            "id": "all_targets_no_regression",
            "scope": "per_target",
            "reduction": "all",
            "metric": "normalized_rmse",
            "operator": "<=",
            "reference_metric": "baseline_normalized_rmse",
            "tolerance": GREENHOUSE_NO_REGRESSION_TOLERANCE,
        },
        {
            "id": "no_constraint_violations",
            "scope": "aggregate",
            "metric": "constraint_violations",
            "operator": "<=",
            "threshold": GREENHOUSE_MAX_CONSTRAINT_VIOLATIONS,
        },
        {
            "id": "minimum_sample_execution_coverage",
            "scope": "overall_and_per_prediction_task",
            "metric": "sample_execution_coverage",
            "operator": ">=",
            "threshold": DEFAULT_SAMPLE_EXECUTION_MIN_COVERAGE,
        },
    ]


def _greenhouse_scoring_contract() -> dict[str, Any]:
    """Return every constant that can change scoring or promotion semantics."""

    return {
        "objective_aggregation_version": GREENHOUSE_OBJECTIVE_AGGREGATION_VERSION,
        "baseline_profile_version": BASELINE_PROFILE_VERSION,
        "reward_definition": SAMPLE_REWARD_DEFINITION,
        "target_weights": dict(GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS),
        "horizon_weighting": "equal",
        "missing_task_penalty": GREENHOUSE_OBJECTIVE_MISSING_PENALTY,
        "objective_component_bound": OBJECTIVE_COMPONENT_BOUND,
        "normalization_scale_method": NORMALIZATION_SCALE_METHOD,
        "baseline_selection_tolerance": BASELINE_SELECTION_TOLERANCE,
        "minimum_practical_score_delta": V2_MINIMUM_SCORE_DELTA,
        "promotion_policy_version": PROMOTION_POLICY_VERSION,
        "promotion_evidence_schema_version": PROMOTION_BLOCK_EVIDENCE_VERSION,
        "confidence_method": PROMOTION_CONFIDENCE_METHOD,
        "confidence_level": PROMOTION_CONFIDENCE_LEVEL,
        "minimum_paired_blocks": PROMOTION_MINIMUM_PAIRED_BLOCKS,
        "block_hours": PROMOTION_BLOCK_HOURS,
        "bootstrap_resamples": PROMOTION_BOOTSTRAP_RESAMPLES,
        "maximum_blocks": PROMOTION_MAXIMUM_BLOCKS,
        "hard_gates": _greenhouse_hard_gates(),
    }


def _rolling_feedback_rows(
    series: DatasetSeries,
    *,
    target: str,
    unit: str,
    values: Sequence[float | None],
    fit_start: int,
    feedback_start: int,
    feedback_end: int,
    visible_history_index: Mapping[int, int],
    parameters: Mapping[str, Any],
    fitted_bias: float,
    defer_prediction: bool,
) -> list[dict[str, Any]]:
    """Prepare one-hour rolling samples, optionally without executing the model."""

    window = int(parameters["window"])
    blend = float(parameters["blend"])
    first_timestamp = series.timestamps[fit_start]
    rows: list[dict[str, Any]] = []
    for label_index in range(feedback_start, feedback_end):
        target_timestamp = series.timestamps[label_index]
        if target_timestamp - first_timestamp < max(
            window, MAX_ROLLING_WINDOW_HOURS
        ):
            continue
        origin_index = visible_history_index.get(target_timestamp - 1)
        observed = _finite_registry_number(values[label_index])
        baseline = (
            _finite_registry_number(values[origin_index])
            if origin_index is not None
            and fit_start <= origin_index < label_index
            else None
        )
        history_points = [
            (series.timestamps[history_index], number)
            for lag in range(1, window + 1)
            if (history_index := visible_history_index.get(target_timestamp - lag))
            is not None
            and fit_start <= history_index < label_index
            if (number := _finite_registry_number(values[history_index])) is not None
        ]
        if observed is None or baseline is None or not history_points:
            continue
        history_timestamps = [timestamp for timestamp, _number in history_points]
        history_window = [number for _timestamp, number in history_points]
        origin_timestamp = series.timestamps[origin_index]
        row: dict[str, Any] = {
            "partition": "training_feedback",
            "timestamp": target_timestamp,
            "origin_timestamp": origin_timestamp,
            "target_timestamp": target_timestamp,
            "horizon_hours": 1,
            "observed": observed,
            "baseline": baseline,
            "target": target,
            "unit": unit,
            "label_free_context": {
                "schema_version": "ecologyrsi-dsh.label-free-sample-context/1",
                "history_window": history_window,
                "feature_snapshot": [],
                "causal_provenance": {
                    "schema_version": (
                        "ecologyrsi-dsh.causal-sample-provenance/1"
                    ),
                    "origin_cutoff_timestamp": origin_timestamp,
                    "latest_context_timestamp": max(history_timestamps),
                    "history_timestamps": history_timestamps,
                },
                "predictor_state": {
                    "window": window,
                    "blend": blend,
                    "bias_scale": float(parameters["bias_scale"]),
                    "fitted_bias": float(fitted_bias),
                },
            },
        }
        if not defer_prediction:
            row["predicted"] = (
                blend * baseline
                + (1.0 - blend) * fmean(history_window)
                + float(fitted_bias)
            )
        rows.append(row)
    return rows


def _rolling_tool_prediction(request: SamplePredictionRequest) -> float:
    context = request.label_free_context
    history = context.get("history_window")
    state = context.get("predictor_state")
    if not isinstance(history, (list, tuple)) or not history:
        raise ValueError("rolling tool requires a non-empty history window")
    if not isinstance(state, Mapping):
        raise ValueError("rolling tool requires fitted predictor state")
    values = [_finite_registry_number(item) for item in history]
    if any(item is None for item in values):
        raise ValueError("rolling tool history must contain finite numbers")
    blend = _finite_registry_number(state.get("blend"))
    fitted_bias = _finite_registry_number(state.get("fitted_bias"))
    if blend is None or not 0 <= blend <= 1 or fitted_bias is None:
        raise ValueError("rolling tool predictor state is invalid")
    predicted = (
        blend * request.baseline
        + (1.0 - blend) * fmean(float(item) for item in values)
        + fitted_bias
    )
    if not math.isfinite(predicted):
        raise ArithmeticError("rolling tool prediction is not finite")
    return predicted


def _finite_registry_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    artifact: ModelArtifact
    evaluation: Evaluation
    # ``None`` means an evaluator does not implement the private full-result
    # contract. An empty tuple is a completed, supported evaluation with no
    # scoreable rows.
    sample_results: tuple[dict[str, Any], ...] | None = None


class EvaluatorRegistry:
    """Resolve a frozen evaluator and optional independent model judge."""

    def __init__(
        self,
        datasets: DatasetRegistry,
        model_gateway: ModelGateway | None = None,
        sample_executor: CollaborativeSampleExecutor | None = None,
        *,
        dsh_runtime_provider: Callable[[], Any] | None = None,
        dsh_revision_provider: Callable[[str], Mapping[str, int]] | None = None,
        dsh_identity_provider: Callable[[str, str], Mapping[str, str] | None]
        | None = None,
    ) -> None:
        self.datasets = datasets
        self.model_gateway = model_gateway or ModelGateway.from_env()
        self._sample_executor_injected = sample_executor is not None
        self.sample_executor = sample_executor or CollaborativeSampleExecutor()
        self.dsh_runtime_provider = dsh_runtime_provider
        self.dsh_revision_provider = dsh_revision_provider
        self.dsh_identity_provider = dsh_identity_provider

    def _sample_executor_for_task(
        self,
        task: TaskManifest,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        model_usage_callback: Callable[[Sequence[Mapping[str, Any]]], None]
        | None = None,
        on_sample_control: Callable[[], str] | None = None,
        forecast_tool: Callable[[SamplePredictionRequest], float] | None = None,
        tools: Sequence[GatewaySampleTool] = (),
        run_id: str | None = None,
        candidate_id: str | None = None,
    ) -> CollaborativeSampleExecutor:
        """Resolve the frozen per-sample runtime without changing old runs."""

        if self._sample_executor_injected:
            return self.sample_executor
        mode = str(
            task.metadata.get("sample_agent_mode")
            or "host_feedback_state_machine"
        )
        if mode == "host_feedback_state_machine":
            return (
                CollaborativeSampleExecutor(
                    RegisteredToolCollaborationAdapter(
                        forecast_tool=forecast_tool
                    )
                )
                if forecast_tool is not None
                else self.sample_executor
            )
        if mode == "dsh_native_workflow":
            if (
                not run_id
                or not candidate_id
                or self.dsh_runtime_provider is None
                or self.dsh_revision_provider is None
                or self.dsh_identity_provider is None
            ):
                raise ValueError("DSH-native sample runtime binding is incomplete")
            identity = self.dsh_identity_provider(run_id, candidate_id)
            if not isinstance(identity, Mapping):
                raise ValueError("DSH-native candidate identity is unavailable")
            strategy_model_id = str(task.metadata.get("strategy_model_id") or "").strip()
            review_model_id = str(task.metadata.get("review_model_id") or "").strip()
            if not strategy_model_id or not review_model_id:
                raise ValueError("DSH-native sample roles require frozen model routes")
            raw_batch_size = task.metadata.get("sample_agent_batch_size", 128)
            raw_concurrency = task.metadata.get("sample_concurrency", 4)
            if (
                isinstance(raw_batch_size, bool)
                or not isinstance(raw_batch_size, int)
                or not 1 <= raw_batch_size <= 128
            ):
                raise ValueError("DSH sample batch size must be between 1 and 128")
            if (
                isinstance(raw_concurrency, bool)
                or not isinstance(raw_concurrency, int)
                or not 1 <= raw_concurrency <= 8
            ):
                raise ValueError("DSH sample concurrency must be between 1 and 8")
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
                forecast_tool=forecast_tool,
                tools=tools,
                microbatch_size=int(raw_batch_size),
                sample_concurrency=int(raw_concurrency),
                progress_callback=progress_callback,
                run_control_callback=on_sample_control,
                remote_critic_policy=task.metadata.get("sample_remote_critic_policy"),
                sample_planner_prompt_profile=task.metadata.get(
                    "sample_planner_prompt_profile"
                ),
            )
            return CollaborativeSampleExecutor(adapter)
        if mode != "gateway_microbatch":
            raise ValueError(f"unsupported sample_agent_mode: {mode}")

        strategy_model_id = str(
            task.metadata.get("strategy_model_id") or ""
        ).strip()
        if not strategy_model_id:
            raise ValueError(
                "gateway_microbatch sample execution requires strategy_model_id"
            )
        raw_batch_size = task.metadata.get("sample_agent_batch_size", 128)
        if (
            isinstance(raw_batch_size, bool)
            or not isinstance(raw_batch_size, int)
            or not 1 <= raw_batch_size <= 128
        ):
            raise ValueError(
                "sample_agent_batch_size must be an integer between 1 and 128"
            )
        raw_concurrency = task.metadata.get("sample_concurrency", 4)
        if (
            isinstance(raw_concurrency, bool)
            or not isinstance(raw_concurrency, int)
            or not 1 <= raw_concurrency <= 8
        ):
            raise ValueError("sample_concurrency must be an integer between 1 and 8")
        review_model_id = str(
            task.metadata.get("review_model_id") or ""
        ).strip()
        if not review_model_id:
            raise ValueError(
                "gateway_microbatch sample execution requires review_model_id"
            )
        adapter = GatewaySampleCollaborationAdapter(
            self.model_gateway,
            strategy_model_id=strategy_model_id,
            review_model_id=review_model_id,
            # The remote critic is microbatched and label-free. The host
            # physical critic remains authoritative after both model roles.
            remote_review_enabled=True,
            forecast_tool=forecast_tool,
            tools=tools,
            microbatch_size=raw_batch_size,
            sample_concurrency=raw_concurrency,
            progress_callback=progress_callback,
            model_usage_callback=model_usage_callback,
            run_control_callback=on_sample_control,
            operation_max_tokens=task.metadata.get("sample_operation_max_tokens"),
            remote_critic_policy=task.metadata.get("sample_remote_critic_policy"),
            sample_planner_prompt_profile=task.metadata.get(
                "sample_planner_prompt_profile"
            ),
            sample_truncation_retry_policy=task.metadata.get(
                "sample_truncation_retry_policy"
            ),
            token_limit=task.budget.get("token_limit", 0),
            token_reservation_per_wave=task.budget.get(
                "token_reservation_per_wave", 0
            ),
        )
        return CollaborativeSampleExecutor(adapter)

    @staticmethod
    def _dsh_sample_stage_context(
        task: TaskManifest,
        candidate: Candidate,
        proposal: Proposal,
    ) -> dict[str, str]:
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
        return {
            "stage_context_digest": sibling_stage_context_digest(
                task_manifest_digest=task.digest,
                generation=candidate.generation,
                frozen_contract_digests=contracts,
            ),
            "candidate_genome_digest": genome_digest,
        }

    def catalog(self) -> list[dict[str, Any]]:
        items = [
            {
                "id": TOY_EVALUATOR_ID,
                "label": "合成水分时间前向评测",
                "description": "仅用于工程演示的固定验证分区评测。",
                "dataset_ids": [TOY_DATASET_ID],
                "evaluation_partition": "validation",
                "scientific_scope": "prediction_demo_non_causal",
                "prediction_model_ids": [TOY_PREDICTOR_MODEL_ID],
                "horizons_hours": [1],
                "objective_profile": "toy_validation_skill@1",
                "implementation": "toy-forward-split/3",
            },
            {
                "id": GREENHOUSE_EVALUATOR_ID,
                "label": "温室环境时间前向评测",
                "description": "在固定训练反馈 cohort 中逐样本协作预测，允许受控失败并与仅由训练拟合分区选择的强基线比较。",
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "evaluation_partition": "training_feedback",
                "scientific_scope": "historical_replay_prediction_non_causal",
                "prediction_model_ids": [
                    GREENHOUSE_ROLLING_PREDICTOR_ID,
                    EXOGENOUS_RIDGE_MODEL_ID,
                    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                ],
                "horizons_hours": [1],
                "objective_profile": GREENHOUSE_OBJECTIVE_PROFILE_ID,
                "implementation": "greenhouse-one-hour-forward/7",
            },
            {
                "id": GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
                "label": "温室环境多时距时间前向评测",
                "description": "仅在训练拟合分区学习，并在固定训练反馈 cohort 中逐样本协作评测 1、6、24 小时预测。",
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "evaluation_partition": "training_feedback",
                "scientific_scope": "historical_replay_prediction_non_causal",
                "prediction_model_ids": [
                    EXOGENOUS_RIDGE_MODEL_ID,
                    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                ],
                "horizons_hours": [1, 6, 24],
                "objective_profile": GREENHOUSE_OBJECTIVE_PROFILE_ID,
                "implementation": "greenhouse-multihorizon-forward/6",
            },
            {
                "id": GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID,
                "label": "温室环境多时距时间前向评测 v2",
                "description": (
                    "仅在训练拟合分区学习，并在固定训练反馈 cohort 中逐样本协作"
                    "评测 1、6、24 小时预测；支持目标与时距独立残差修正。"
                ),
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "evaluation_partition": "training_feedback",
                "scientific_scope": "historical_replay_prediction_non_causal",
                "prediction_model_ids": [
                    EXOGENOUS_RIDGE_MODEL_ID,
                    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                    HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                ],
                "horizons_hours": [1, 6, 24],
                "objective_profile": GREENHOUSE_OBJECTIVE_PROFILE_ID,
                "implementation": "greenhouse-multihorizon-forward/7",
            },
        ]
        for item in items:
            target_count = (
                len(GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS)
                if item.get("objective_profile")
                == GREENHOUSE_OBJECTIVE_PROFILE_ID
                else 1
            )
            item["prediction_task_count"] = max(
                1,
                target_count * len(item.get("horizons_hours", ())),
            )
            item["minimum_samples_per_update"] = item["prediction_task_count"]
            digest_payload = {
                "evaluator_id": item["id"],
                "implementation": item.pop("implementation"),
                "evaluation_partition": item["evaluation_partition"],
                "prediction_model_ids": item["prediction_model_ids"],
                "horizons_hours": item["horizons_hours"],
            }
            if item.get("objective_profile") == GREENHOUSE_OBJECTIVE_PROFILE_ID:
                digest_payload["scoring_contract"] = _greenhouse_scoring_contract()
            item["configuration_digest"] = digest(digest_payload)
        return items

    def predictor_catalog(self) -> list[dict[str, Any]]:
        items = [
            {
                "id": TOY_PREDICTOR_MODEL_ID,
                "label": "合成水分滚动预测模型",
                "description": "仅用于合成作物—土壤—水分工程演示。",
                "dataset_ids": [TOY_DATASET_ID],
                "parameter_names": ["alpha", "window", "water_threshold"],
                "scientific_scope": "prediction_demo_non_causal",
                "implementation": "toy-rolling-water/1",
            },
            {
                "id": GREENHOUSE_ROLLING_PREDICTOR_ID,
                "label": "温室持续性偏差预测模型",
                "description": "使用目标变量历史滚动值和训练拟合分区偏差进行 1 小时预测。",
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "parameter_names": ["blend", "window", "bias_scale"],
                "scientific_scope": "historical_replay_prediction_non_causal",
                "implementation": "greenhouse-rolling-residual/2",
            },
            {
                "id": EXOGENOUS_RIDGE_MODEL_ID,
                "label": "温室外生变量岭回归残差模型",
                "description": "融合温室外气象、设定值、动作和根区观测，预测相对持续性基线的残差。",
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "parameter_names": ["history_steps", "ridge_alpha", "residual_scale"],
                "scientific_scope": "historical_replay_prediction_non_causal",
                "implementation": "greenhouse-exogenous-ridge/1",
            },
            {
                "id": TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                "label": "温室分目标岭回归残差模型",
                "description": (
                    "复用登记的外生变量岭回归，但分别缩放温度、湿度和 CO2 "
                    "残差；任一缩放为 0 时使用持续性预测。"
                ),
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "parameter_names": [
                    "history_steps",
                    "ridge_alpha",
                    "air_temperature_residual_scale",
                    "relative_humidity_residual_scale",
                    "co2_concentration_residual_scale",
                ],
                "scientific_scope": "historical_replay_prediction_non_causal",
                "implementation": "greenhouse-targetwise-ridge/1",
            },
            {
                "id": HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                "label": "温室分目标分时距岭回归残差模型",
                "description": (
                    "在 1、6、24 小时时距分别缩放温度、湿度和 CO2 残差；"
                    "任一目标时距的缩放为 0 时仅该单元使用持续性预测。"
                ),
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "parameter_names": [
                    "history_steps",
                    "ridge_alpha",
                    "air_temperature_1h_residual_scale",
                    "air_temperature_6h_residual_scale",
                    "air_temperature_24h_residual_scale",
                    "relative_humidity_1h_residual_scale",
                    "relative_humidity_6h_residual_scale",
                    "relative_humidity_24h_residual_scale",
                    "co2_concentration_1h_residual_scale",
                    "co2_concentration_6h_residual_scale",
                    "co2_concentration_24h_residual_scale",
                ],
                "scientific_scope": "historical_replay_prediction_non_causal",
                "implementation": "greenhouse-horizon-targetwise-ridge/1",
            },
        ]
        for item in items:
            item["configuration_digest"] = digest(
                {
                    "prediction_model_id": item["id"],
                    "implementation": item.pop("implementation"),
                    "parameter_names": item["parameter_names"],
                    "causal_interpretation": False,
                }
            )
        return items

    @staticmethod
    def judge_catalog(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        built_in = {
            "id": RULE_JUDGE_ID,
            "label": "内置规则评审",
            "model": "host-rule-gate",
            "available": True,
            "authenticated": True,
            "role": "judge",
        }
        remote = []
        for model in models:
            item = dict(model)
            item["id"] = item.get("id", item.get("model_id"))
            item.setdefault("label", item.get("model", item["id"]))
            item.setdefault("role", "policy_or_judge")
            remote.append(item)
        return [built_in, *remote]

    def default_evaluator(self, dataset_id: str) -> str:
        return (
            TOY_EVALUATOR_ID
            if dataset_id == TOY_DATASET_ID
            else GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID
        )

    def default_predictor(self, dataset_id: str) -> str:
        return (
            TOY_PREDICTOR_MODEL_ID
            if dataset_id == TOY_DATASET_ID
            else EXOGENOUS_RIDGE_MODEL_ID
        )

    def minimum_samples_per_update(self, evaluator_id: str) -> int:
        """Return the smallest cohort that can cover every scoring task once."""

        for item in self.catalog():
            if item["id"] == evaluator_id:
                return int(item["minimum_samples_per_update"])
        raise ValueError(f"unknown evaluator_id: {evaluator_id}")

    def evaluator_configuration_digest(self, evaluator_id: str) -> str:
        for item in self.catalog():
            if item["id"] == evaluator_id:
                return str(item["configuration_digest"])
        raise ValueError(f"unknown evaluator_id: {evaluator_id}")

    def objective_profile(self, evaluator_id: str) -> dict[str, Any]:
        """Return the frozen optimization objective for UI and audit output."""

        for item in self.catalog():
            if item["id"] != evaluator_id:
                continue
            if item.get("objective_profile") == GREENHOUSE_OBJECTIVE_PROFILE_ID:
                return {
                    "id": GREENHOUSE_OBJECTIVE_PROFILE_ID,
                    **_greenhouse_scoring_contract(),
                }
            return {
                "id": str(item.get("objective_profile") or "unspecified"),
                "target_weights": None,
                "horizon_weighting": "evaluator_defined",
                "hard_gates": [
                    {
                        "id": "evaluator_passed",
                        "scope": "evaluation",
                        "metric": "passed",
                        "operator": "==",
                        "threshold": True,
                    }
                ],
            }
        raise ValueError(f"unknown evaluator_id: {evaluator_id}")

    def predictor_configuration_digest(self, predictor_model_id: str) -> str:
        for item in self.predictor_catalog():
            if item["id"] == predictor_model_id:
                return str(item["configuration_digest"])
        raise ValueError(f"unknown prediction_model_id: {predictor_model_id}")

    def validate_binding(
        self,
        dataset_id: str,
        evaluator_id: str,
        predictor_model_id: str | None = None,
    ) -> None:
        predictor_model_id = predictor_model_id or self.default_predictor(dataset_id)
        for item in self.catalog():
            if item["id"] == evaluator_id:
                if dataset_id not in item["dataset_ids"]:
                    raise ValueError("evaluator_id is incompatible with dataset_id")
                if predictor_model_id not in item["prediction_model_ids"]:
                    raise ValueError(
                        "prediction_model_id is incompatible with evaluator_id"
                    )
                predictor = next(
                    (
                        model
                        for model in self.predictor_catalog()
                        if model["id"] == predictor_model_id
                    ),
                    None,
                )
                if predictor is None:
                    raise ValueError(
                        f"unknown prediction_model_id: {predictor_model_id}"
                    )
                if dataset_id not in predictor["dataset_ids"]:
                    raise ValueError(
                        "prediction_model_id is incompatible with dataset_id"
                    )
                return
        raise ValueError(f"unknown evaluator_id: {evaluator_id}")

    @staticmethod
    def validate_parameter_overrides(
        task: TaskManifest, overrides: Mapping[str, Any]
    ) -> None:
        if not isinstance(overrides, Mapping):
            raise TypeError("parameter_overrides must be an object")
        greenhouse = "greenhouse" in task.domain_pack.lower()
        predictor_model_id = str(
            task.metadata.get("prediction_model_id")
            or (
                EXOGENOUS_RIDGE_MODEL_ID
                if greenhouse
                else TOY_PREDICTOR_MODEL_ID
            )
        )
        schemas = (
            {
                "history_steps": (int, 1, 12),
                "ridge_alpha": (float, 0.0001, 1.0),
                "air_temperature_1h_residual_scale": (float, 0.0, 1.0),
                "air_temperature_6h_residual_scale": (float, 0.0, 1.0),
                "air_temperature_24h_residual_scale": (float, 0.0, 1.0),
                "relative_humidity_1h_residual_scale": (float, 0.0, 1.0),
                "relative_humidity_6h_residual_scale": (float, 0.0, 1.0),
                "relative_humidity_24h_residual_scale": (float, 0.0, 1.0),
                "co2_concentration_1h_residual_scale": (float, 0.0, 1.0),
                "co2_concentration_6h_residual_scale": (float, 0.0, 1.0),
                "co2_concentration_24h_residual_scale": (float, 0.0, 1.0),
            }
            if predictor_model_id == HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
            else
            {
                "history_steps": (int, 1, 12),
                "ridge_alpha": (float, 0.0001, 1.0),
                "air_temperature_residual_scale": (float, 0.0, 1.0),
                "relative_humidity_residual_scale": (float, 0.0, 1.0),
                "co2_concentration_residual_scale": (float, 0.0, 1.0),
            }
            if predictor_model_id == TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
            else
            {
                "history_steps": (int, 1, 12),
                "ridge_alpha": (float, 0.0001, 1.0),
                "residual_scale": (float, 0.0, 1.0),
            }
            if predictor_model_id == EXOGENOUS_RIDGE_MODEL_ID
            else {
                "blend": (float, 0.0, 1.0),
                "window": (int, 1, 48),
                "bias_scale": (float, 0.0, 2.0),
            }
            if greenhouse
            else {
                "alpha": (float, 0.05, 0.95),
                "window": (int, 1, 30),
                "water_threshold": (float, 0.05, 0.85),
            }
        )
        unknown = set(overrides) - set(schemas)
        if unknown:
            raise ValueError(
                "parameter_overrides contains unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        for name, value in overrides.items():
            expected, minimum, maximum = schemas[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"parameter_overrides.{name} must be numeric")
            if expected is int and not isinstance(value, int):
                raise ValueError(f"parameter_overrides.{name} must be an integer")
            if (
                not math.isfinite(float(value))
                or not minimum <= float(value) <= maximum
            ):
                raise ValueError(
                    f"parameter_overrides.{name} is outside the allowed range"
                )

    def evaluate(
        self,
        task: TaskManifest,
        candidate: Candidate,
        proposal: Proposal,
    ) -> EvaluationBundle:
        bundle = self.evaluate_scientific(task, candidate, proposal)
        return self.apply_judge(task, proposal, bundle)

    def evaluate_scientific(
        self,
        task: TaskManifest,
        candidate: Candidate,
        proposal: Proposal,
        *,
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
        on_model_usage: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
        on_sample_control: Callable[[], str] | None = None,
        algorithm_spec: AlgorithmSpec | Mapping[str, Any] | None = None,
    ) -> EvaluationBundle:
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
        series = self._validated_series(task, dataset_id)
        if resolved_algorithm_spec is not None:
            if resolved_algorithm_spec.evaluator_id != evaluator_id:
                raise ValueError(
                    "algorithm spec evaluator does not match the frozen task"
                )
            self._validate_algorithm_data_boundary(
                resolved_algorithm_spec,
                series,
            )
            if resolved_algorithm_spec.predictor_adoption is not None:
                adoption = PredictorAdoption.from_dict(
                    resolved_algorithm_spec.predictor_adoption
                )
                if (
                    adoption.adopted_id != predictor_model_id
                    or adoption.adopted_digest
                    != self.predictor_configuration_digest(predictor_model_id)
                ):
                    raise ValueError(
                        "algorithm predictor adoption does not match the registered runtime"
                    )
        execution_plan = self._resolve_execution_plan(
            candidate,
            proposal,
            resolved_algorithm_spec,
        )
        if evaluator_id == TOY_EVALUATOR_ID:
            return self._evaluate_toy(
                task,
                candidate,
                proposal,
                on_training_complete=on_training_complete,
                on_evaluation_progress=on_evaluation_progress,
                on_sample_results=on_sample_results,
                on_sample_checkpoint=on_sample_checkpoint,
                on_model_usage=on_model_usage,
                on_sample_control=on_sample_control,
                execution_plan=execution_plan,
            )
        if predictor_model_id in {
            EXOGENOUS_RIDGE_MODEL_ID,
            TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
            HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
        }:
            horizons = (
                (1, 6, 24)
                if evaluator_id
                in {
                    GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
                    GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID,
                }
                else (1,)
            )
            return self._evaluate_greenhouse_ridge(
                task,
                candidate,
                proposal,
                series,
                horizons=horizons,
                on_training_complete=on_training_complete,
                on_evaluation_progress=on_evaluation_progress,
                on_sample_results=on_sample_results,
                on_sample_checkpoint=on_sample_checkpoint,
                on_model_usage=on_model_usage,
                on_sample_control=on_sample_control,
                execution_plan=execution_plan,
                predictor_model_id=predictor_model_id,
            )
        else:
            return self._evaluate_greenhouse(
                task,
                candidate,
                proposal,
                series,
                on_training_complete=on_training_complete,
                on_evaluation_progress=on_evaluation_progress,
                on_sample_results=on_sample_results,
                on_sample_checkpoint=on_sample_checkpoint,
                on_model_usage=on_model_usage,
                on_sample_control=on_sample_control,
                execution_plan=execution_plan,
            )

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
    ) -> SampleExecutionPolicy:
        base = SampleExecutionPolicy.from_mapping(
            task.metadata.get("sample_execution_policy")
        )
        # Prior-generation evidence may increase resilience, but it may never
        # weaken the immutable coverage thresholds chosen by the host.
        return SampleExecutionPolicy(
            max_attempts=max(base.max_attempts, execution_plan.sample_max_attempts),
            plan_max_attempts=max(
                base.plan_max_attempts, execution_plan.plan_max_attempts
            ),
            minimum_coverage=base.minimum_coverage,
            minimum_task_coverage=base.minimum_task_coverage,
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
        if metadata.get("execution_protocol") == "dsh_native_plugin_evolution@1":
            series = self.datasets.selection_view(
                dataset_id,
                str(episode_id) if episode_id is not None else None,
                expected_dataset_digest=dataset_digest,
                expected_split_manifest_digest=split_digest,
                expected_data_protocol_digest=metadata.get("data_protocol_digest"),
                target_names=tuple(
                    metadata.get(
                        "objective_targets",
                        (
                            "air_temperature",
                            "relative_humidity",
                            "co2_concentration",
                        ),
                    )
                ),
                horizons=tuple(metadata.get("objective_horizons", (1, 6, 24))),
                history_steps=int(metadata.get("history_steps", 3)),
            )
        else:
            series = self.datasets.series(
                dataset_id,
                str(episode_id) if episode_id is not None else None,
                expected_dataset_digest=dataset_digest,
                expected_split_manifest_digest=split_digest,
            )
        resolved_digest = getattr(series, "digest", getattr(series, "dataset_digest", None))
        if resolved_digest != dataset_digest:
            raise ValueError("数据集快照发生漂移，候选训练已拒绝执行")
        if series.split_manifest_digest_sha256 != split_digest:
            raise ValueError("时间分区快照发生漂移，候选评测已拒绝执行")
        return series

    def _evaluate_toy(
        self,
        task: TaskManifest,
        candidate: Candidate,
        proposal: Proposal,
        *,
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
        on_model_usage: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
        on_sample_control: Callable[[], str] | None = None,
        execution_plan: DerivedExecutionPlan | None = None,
    ) -> EvaluationBundle:
        execution_plan = execution_plan or self._resolve_execution_plan(
            candidate, proposal, None
        )
        toy = ToyCropSoilWater(seed=int(task.metadata.get("dataset_seed", 0)))
        training_metrics = toy.score(proposal.changes, "train")
        training_partition_rows = len(toy.splits["train"])
        training_used_examples = int(training_metrics["n"])
        artifact = ModelArtifact(
            artifact_id=f"artifact:{candidate.candidate_id}",
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            model_id="toy-rolling-water@1",
            dataset_digest=toy.dataset_digest,
            training_partition="train",
            training_rows=training_partition_rows,
            parameters=dict(proposal.changes),
            learned_parameters={},
            metrics={
                "training_mae": training_metrics["mae"],
                "training_rmse": training_metrics["rmse"],
                "execution_mode": "registered_lightweight",
                "fit_method": "toy_score",
                "fit_passes_requested": 1,
                "fit_passes_completed": 1,
                "iterative_epoch_training": False,
                # Compatibility aliases for projections created before fit
                # passes and iterative epochs were separated explicitly.
                "epochs_requested": 1,
                "epochs_completed": 1,
                "training_rows": training_partition_rows,
                "training_partition_rows": training_partition_rows,
                "training_eligible_examples": training_partition_rows,
                "training_used_examples": training_used_examples,
                "training_skipped_examples": max(
                    0, training_partition_rows - training_used_examples
                ),
            },
        )
        if on_training_complete is not None:
            on_training_complete()
        original = toy.evaluate_candidate(
            candidate.run_id,
            candidate,
            proposal,
            split="validation",
            evaluator_digest=str(
                task.metadata.get("evaluator_digest")
                or self.evaluator_configuration_digest(TOY_EVALUATOR_ID)
            ),
        )
        evaluation_data = original.to_dict()
        evaluation_metrics = dict(evaluation_data["metrics"])
        evaluation_partition_rows = len(toy.splits["validation"])
        sample_policy = self._sample_execution_policy(task, execution_plan)
        raw_prediction_rows = [
            {**row, "partition": "validation"}
            for row in evaluation_metrics.get("prediction_preview", [])
        ]
        sample_batch = self._sample_executor_for_task(
            task,
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            progress_callback=on_evaluation_progress,
            model_usage_callback=on_model_usage,
            on_sample_control=on_sample_control,
        ).execute(
            raw_prediction_rows,
            context={
                "run_id": candidate.run_id,
                "candidate_id": candidate.candidate_id,
                **self._dsh_sample_stage_context(task, candidate, proposal),
                "dataset_digest": toy.dataset_digest,
                "partition": "validation",
                "algorithm_id": "toy-rolling-water",
                "algorithm_version": "1",
                "evaluator_id": TOY_EVALUATOR_ID,
                "horizons_hours": [24],
                "candidate_parameters": dict(proposal.changes),
                "proposal_plan": dict(proposal.metadata),
                "derived_execution_plan": execution_plan.to_dict(),
                **(
                    {
                        "sample_planner_prompt_profile": task.metadata[
                            "sample_planner_prompt_profile"
                        ]
                    }
                    if task.metadata.get("sample_planner_prompt_profile") is not None
                    else {}
                ),
            },
            target_bounds={
                "soil_water": {
                    "unit": "fraction",
                    "minimum": 0.0,
                    "maximum": 1.0,
                }
            },
            algorithm_id="toy-rolling-water",
            algorithm_version="1",
            policy=sample_policy,
            result_callback=on_sample_results,
            checkpoint_callback=on_sample_checkpoint,
        )
        scoring_rows = list(sample_batch.scoring_rows)
        errors = [
            float(row["predicted"]) - float(row["observed"])
            for row in scoring_rows
        ]
        successful_examples = int(sample_batch.summary["succeeded_examples"])
        evaluation_used_examples = len(scoring_rows)
        evaluation_skipped_examples = max(
            0, evaluation_partition_rows - evaluation_used_examples
        )
        sample_coverage = (
            successful_examples / evaluation_partition_rows
            if evaluation_partition_rows
            else 0.0
        )
        sample_coverage_pass = (
            evaluation_partition_rows > 0
            and sample_coverage >= sample_policy.minimum_coverage
            and sample_coverage >= sample_policy.minimum_task_coverage
        )
        if errors:
            evaluation_mae = _mae(errors)
            evaluation_rmse = _rmse(errors)
            evaluation_score = max(0.0, 1.0 - evaluation_rmse)
        else:
            evaluation_mae = 1.0
            evaluation_rmse = 1.0
            evaluation_score = 0.0
        constraint_violations = sum(
            not 0.0 <= float(row["predicted"]) <= 1.0
            for row in scoring_rows
        )
        scientific_pass = (
            evaluation_rmse <= 0.12
            and float(evaluation_metrics.get("water_balance_error", 1.0)) <= 0.25
            and constraint_violations == 0
            and sample_coverage_pass
        )
        sample_execution_summary = dict(sample_batch.summary)
        sample_execution_summary.update(
            {
                "adapter_attempt_coverage": sample_batch.summary["coverage"],
                "eligible_examples": evaluation_partition_rows,
                "succeeded_examples": successful_examples,
                "scored_examples": evaluation_used_examples,
                "skipped_examples": evaluation_skipped_examples,
                "input_unavailable_examples": max(
                    0, evaluation_partition_rows - len(sample_batch.records)
                ),
                "coverage": sample_coverage,
                "coverage_pass": sample_coverage_pass,
                "per_task_coverage_pass": sample_coverage_pass,
                "incomplete_prediction_tasks": int(not scoring_rows),
            }
        )
        evaluation_metrics.update(
            {
                "score": evaluation_score,
                "mae": evaluation_mae,
                "rmse": evaluation_rmse,
                "n": evaluation_used_examples,
                "non_negative_state": 1.0 if constraint_violations == 0 else 0.0,
                "constraint_violations": constraint_violations,
                "scientific_pass": scientific_pass,
                "execution_mode": "registered_lightweight",
                "fit_method": "toy_score",
                "evaluation_partition_rows": evaluation_partition_rows,
                "evaluation_eligible_examples": evaluation_partition_rows,
                "evaluation_used_examples": evaluation_used_examples,
                "evaluation_skipped_examples": evaluation_skipped_examples,
                "sample_execution": sample_execution_summary,
                "sample_execution_records": bounded_sample_execution_records(
                    sample_batch.records
                ),
                "sample_execution_trace_archive": encode_sample_execution_trace(
                    sample_batch.records
                ),
                "sample_execution_trace_digest": sample_batch.summary["trace_digest"],
                "sample_execution_trace_record_count": len(sample_batch.records),
                "sample_execution_coverage": sample_coverage,
                "sample_execution_coverage_pass": sample_coverage_pass,
                "sample_execution_failed_examples": int(
                    sample_batch.summary["failed_examples"]
                ),
                "evaluation_scoring_fallback_examples": int(
                    sample_batch.summary["scoring_fallback_examples"]
                ),
                "prediction_preview": scoring_rows[:_PREVIEW_ROWS_TOTAL],
            }
        )
        evaluation = Evaluation(
            **{
                **evaluation_data,
                "score": evaluation_score,
                "passed": scientific_pass,
                "metrics": evaluation_metrics,
                "artifact_digest": artifact.digest,
            }
        )
        return EvaluationBundle(
            artifact=artifact,
            evaluation=evaluation,
            sample_results=build_sample_results(
                candidate.candidate_id, scoring_rows
            ),
        )

    def _evaluate_greenhouse(
        self,
        task: TaskManifest,
        candidate: Candidate,
        proposal: Proposal,
        series: DatasetSeries,
        *,
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
        on_model_usage: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
        on_sample_control: Callable[[], str] | None = None,
        execution_plan: DerivedExecutionPlan | None = None,
    ) -> EvaluationBundle:
        execution_plan = execution_plan or self._resolve_execution_plan(
            candidate, proposal, None
        )
        dataset_id = task.dataset
        assert dataset_id is not None
        parameters = _greenhouse_parameters(proposal.changes)
        fit_range = series.partitions["training_fit"]
        feedback_range = series.partitions["training_feedback"]
        learned: dict[str, float] = {}
        training_metrics: dict[str, float] = {}
        evaluation_metrics: dict[str, Any] = {}
        target_results: list[dict[str, Any]] = []
        preview_rows: list[dict[str, Any]] = []
        constraint_violations = 0

        normalized_candidate: list[float] = []
        normalized_baseline: list[float] = []
        normalized_mean_rewards: list[float] = []
        raw_normalized_mean_rewards: list[float] = []
        normalization_scales: dict[str, dict[str, Any]] = {}
        raw_mae: list[float] = []
        raw_rmse: list[float] = []
        total_rows = 0
        total_missing_rows = 0
        total_fit_eligible_rows = 0
        total_fit_rows = 0
        total_fit_missing_rows = 0
        evaluation_index_rows: list[dict[str, Any]] = []

        # Only the two visible ranges enter this lookup.  This permits causal
        # pre-history while keeping any embargo rows between them inaccessible.
        visible_history_index = {
            series.timestamps[index]: index
            for selected in (fit_range, feedback_range)
            for index in range(selected.start, selected.end)
        }

        fitted_targets: list[
            tuple[str, str, float, float, tuple[float | None, ...], float]
        ] = []
        for target_name, unit, minimum, maximum in _TARGETS:
            try:
                values = tuple(series.values[target_name])
            except KeyError:
                raise ValueError(
                    f"dataset is missing required evaluation target: {target_name}"
                ) from None
            fitted_bias, fit_errors, fit_indices = _fit_bias(
                values,
                fit_range.start,
                fit_range.end,
                parameters,
                timestamps=tuple(series.timestamps),
                timestamp_index={
                    timestamp: position
                    for position, timestamp in enumerate(
                        series.timestamps[fit_range.start : fit_range.end],
                        fit_range.start,
                    )
                },
            )
            fit_eligible_rows = _rolling_eligible_rows(
                tuple(series.timestamps),
                fit_range.start,
                fit_range.end,
                int(parameters["window"]),
            )
            fit_missing_rows = fit_eligible_rows - len(fit_indices)
            total_fit_eligible_rows += fit_eligible_rows
            total_fit_rows += len(fit_indices)
            total_fit_missing_rows += fit_missing_rows
            learned[target_name + "_bias"] = fitted_bias
            training_metrics[target_name + "_rmse"] = _rmse(fit_errors)
            training_metrics[target_name + "_n"] = len(fit_indices)
            training_metrics[target_name + "_missing_or_nonfinite_rows"] = (
                fit_missing_rows
            )
            fitted_targets.append(
                (target_name, unit, minimum, maximum, values, fitted_bias)
            )

        if on_training_complete is not None:
            on_training_complete()

        defer_feedback_prediction = (
            not self._sample_executor_injected
            and task.metadata.get("sample_agent_mode") == "gateway_microbatch"
        )
        generated_feedback_rows: list[dict[str, Any]] = []
        target_contexts: list[dict[str, Any]] = []
        for target_name, unit, minimum, maximum, values, fitted_bias in fitted_targets:
            feedback_rows = _rolling_feedback_rows(
                series,
                target=target_name,
                unit=unit,
                values=values,
                fit_start=fit_range.start,
                feedback_start=feedback_range.start,
                feedback_end=feedback_range.end,
                visible_history_index=visible_history_index,
                parameters=parameters,
                fitted_bias=fitted_bias,
                defer_prediction=defer_feedback_prediction,
            )
            if not feedback_rows:
                raise ValueError(
                    f"training_feedback has no usable rows for {target_name}"
                )
            feedback_eligible_rows = _rolling_eligible_rows(
                tuple(series.timestamps),
                feedback_range.start,
                feedback_range.end,
                int(parameters["window"]),
                history_start=fit_range.start,
                timestamp_index=visible_history_index,
                minimum_history_hours=MAX_ROLLING_WINDOW_HOURS,
            )
            target_contexts.append(
                {
                    "target": target_name,
                    "unit": unit,
                    "minimum": minimum,
                    "maximum": maximum,
                    "scale": _normalization_scale(
                        values[fit_range.start : fit_range.end]
                    )[0],
                    "scale_method": NORMALIZATION_SCALE_METHOD,
                    "eligible_rows": feedback_eligible_rows,
                }
            )
            generated_feedback_rows.extend(feedback_rows)

        feedback_update_cohort: dict[str, Any] | None = None
        samples_per_update = _feedback_update_limit(task)
        if samples_per_update is not None:
            generated_feedback_rows, feedback_update_cohort = (
                _select_feedback_update_cohort(
                    generated_feedback_rows,
                    generation=candidate.generation,
                    samples_per_update=samples_per_update,
                    dataset_digest=series.digest,
                    split_manifest_digest=series.split_manifest_digest_sha256,
                )
            )
            selected_task_counts = _cohort_task_counts(
                feedback_update_cohort, "selected_count"
            )
            available_task_counts = _cohort_task_counts(
                feedback_update_cohort, "population_count"
            )
            for target_context in target_contexts:
                key = (str(target_context["target"]), 1)
                target_context["partition_eligible_rows"] = target_context[
                    "eligible_rows"
                ]
                target_context["available_rows"] = available_task_counts.get(key, 0)
                target_context["eligible_rows"] = selected_task_counts.get(key, 0)

        baseline_profile = fit_baseline_profile(
            series,
            targets=tuple(item[0] for item in _TARGETS),
            horizons=(1,),
        )
        baseline_scales = {
            str(item["target"]): float(item["scale"])
            for item in target_contexts
        }

        def finalize_scoring_rows(
            result_rows: Sequence[Mapping[str, Any]],
        ) -> list[dict[str, Any]]:
            return _apply_scoring_baseline(
                series,
                result_rows,
                baseline_profile,
                baseline_scales,
            )

        def publish_scoring_rows(
            result_rows: Sequence[Mapping[str, Any]],
        ) -> None:
            if on_sample_results is not None:
                on_sample_results(finalize_scoring_rows(result_rows))

        algorithm_name, separator, algorithm_revision = (
            GREENHOUSE_ROLLING_PREDICTOR_ID.rpartition("@")
        )
        if not separator:
            algorithm_name = GREENHOUSE_ROLLING_PREDICTOR_ID
            algorithm_revision = "unversioned"
        sample_policy = self._sample_execution_policy(task, execution_plan)
        sample_batch = self._sample_executor_for_task(
            task,
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            progress_callback=on_evaluation_progress,
            model_usage_callback=on_model_usage,
            on_sample_control=on_sample_control,
            forecast_tool=(
                _rolling_tool_prediction if defer_feedback_prediction else None
            ),
        ).execute(
            generated_feedback_rows,
            context={
                "run_id": candidate.run_id,
                "candidate_id": candidate.candidate_id,
                **self._dsh_sample_stage_context(task, candidate, proposal),
                "dataset_digest": series.digest,
                "split_manifest_digest_sha256": series.split_manifest_digest_sha256,
                "partition": "training_feedback",
                "algorithm_id": algorithm_name,
                "algorithm_version": algorithm_revision,
                "evaluator_id": GREENHOUSE_EVALUATOR_ID,
                "horizons_hours": [1],
                "candidate_parameters": dict(parameters),
                "proposal_plan": dict(proposal.metadata),
                "derived_execution_plan": execution_plan.to_dict(),
                "sample_agent_mode": task.metadata.get(
                    "sample_agent_mode", "host_feedback_state_machine"
                ),
                "sample_agent_batch_size": task.metadata.get(
                    "sample_agent_batch_size", 128
                ),
                "sample_concurrency": task.metadata.get(
                    "sample_concurrency", 4
                ),
                **(
                    {
                        "samples_per_update": samples_per_update,
                        "feedback_update_cohort": feedback_update_cohort,
                    }
                    if feedback_update_cohort is not None
                    else {}
                ),
                **(
                    {
                        "sample_planner_prompt_profile": task.metadata[
                            "sample_planner_prompt_profile"
                        ]
                    }
                    if task.metadata.get("sample_planner_prompt_profile") is not None
                    else {}
                ),
                "strategy_model_id": task.metadata.get("strategy_model_id"),
                "review_model_id": task.metadata.get("review_model_id"),
                "tool_experience": proposal.metadata.get("tool_experience", []),
                "algorithm_artifact_digest": digest(
                    {
                        "prediction_model_id": GREENHOUSE_ROLLING_PREDICTOR_ID,
                        "parameters": dict(parameters),
                        "learned": learned,
                        "dataset_digest": series.digest,
                    }
                ),
            },
            target_bounds={
                name: {"unit": unit, "minimum": minimum, "maximum": maximum}
                for name, unit, minimum, maximum in _TARGETS
            },
            algorithm_id=algorithm_name,
            algorithm_version=algorithm_revision,
            policy=sample_policy,
            result_callback=(
                publish_scoring_rows if on_sample_results is not None else None
            ),
            checkpoint_callback=on_sample_checkpoint,
        )
        scoring_rows = finalize_scoring_rows(sample_batch.scoring_rows)
        promotion_block_evidence = build_promotion_block_evidence(
            scoring_rows,
            horizons=(1,),
            target_weights=GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS,
            dataset_digest=series.digest,
            split_manifest_digest_sha256=series.split_manifest_digest_sha256,
        )
        sample_records = list(sample_batch.records)
        incomplete_prediction_tasks = 0

        for target_context in target_contexts:
            target_name = target_context["target"]
            unit = target_context["unit"]
            minimum = target_context["minimum"]
            maximum = target_context["maximum"]
            scale = target_context["scale"]
            feedback_eligible_rows = target_context["eligible_rows"]
            partition_eligible_rows = target_context.get(
                "partition_eligible_rows", feedback_eligible_rows
            )
            available_rows = target_context.get(
                "available_rows", feedback_eligible_rows
            )
            target_rows_all = [
                row for row in scoring_rows if row["target"] == target_name
            ]
            target_rows = [
                row
                for row in target_rows_all
                if str(row.get("sample_execution_status", "succeeded"))
                .strip()
                .casefold()
                != "failed"
                and not row.get("scoring_fallback")
            ]
            failed_rows = [
                row for row in target_rows_all if row not in target_rows
            ]
            normalization_scales[target_name] = {
                "scale": scale,
                "method": target_context.get(
                    "scale_method", NORMALIZATION_SCALE_METHOD
                ),
                "source_partition": "training_fit",
            }
            target_execution = next(
                (
                    item
                    for item in sample_batch.summary["tasks"]
                    if item["target"] == target_name
                    and item["horizon_hours"] == 1
                ),
                {
                    "attempted_examples": 0,
                    "succeeded_examples": 0,
                    "failed_examples": 0,
                },
            )
            candidate_errors = [
                float(row["predicted"]) - float(row["observed"])
                for row in target_rows
            ]
            baseline_errors = [
                float(row["baseline"]) - float(row["observed"])
                for row in target_rows
            ]
            feedback_missing_rows = max(
                0, feedback_eligible_rows - len(target_rows)
            )
            target_coverage = (
                int(target_execution["succeeded_examples"]) / feedback_eligible_rows
                if feedback_eligible_rows
                else 0.0
            )
            target_coverage_pass = (
                feedback_eligible_rows > 0
                and target_coverage >= sample_policy.minimum_task_coverage
            )
            total_missing_rows += feedback_missing_rows
            evaluation_index_rows.extend(
                {
                    "target": target_name,
                    "horizon_hours": 1,
                    "origin_timestamp": row["origin_timestamp"],
                    "target_timestamp": row["target_timestamp"],
                }
                for row in target_rows_all
            )
            if not target_rows:
                incomplete_prediction_tasks += 1
                target_results.append(
                    {
                        "target": target_name,
                        "unit": unit,
                        "horizon_hours": 1,
                        "n": 0,
                        "eligible_rows": feedback_eligible_rows,
                        "partition_eligible_rows": partition_eligible_rows,
                        "available_rows": available_rows,
                        "deferred_rows": max(0, available_rows - feedback_eligible_rows),
                        "missing_or_nonfinite_rows": feedback_missing_rows,
                        "failed_rows": len(failed_rows),
                        "normalization_scale": scale,
                        "mae": None,
                        "rmse": None,
                        "baseline_mae": None,
                        "baseline_rmse": None,
                        "normalized_rmse": None,
                        "baseline_normalized_rmse": None,
                        "skill_score": -1.0,
                        "mean_reward": None,
                        "normalized_mean_reward": None,
                        "raw_normalized_mean_reward": None,
                        "constraint_violations": 0,
                        "sample_execution_attempted": int(
                            target_execution["attempted_examples"]
                        ),
                        "sample_execution_failed": int(
                            target_execution["failed_examples"]
                        ),
                        "sample_execution_coverage": target_coverage,
                        "sample_execution_coverage_pass": False,
                        "objective_quality": 0.0,
                    }
                )
                continue
            candidate_rmse = _rmse(candidate_errors)
            baseline_rmse = _rmse(baseline_errors)
            candidate_mae = _mae(candidate_errors)
            baseline_mae = _mae(baseline_errors)
            normalized_candidate.append(candidate_rmse / scale)
            normalized_baseline.append(baseline_rmse / scale)
            sample_rewards = [
                abs(baseline_error) - abs(candidate_error)
                for baseline_error, candidate_error in zip(
                    baseline_errors, candidate_errors
                )
            ]
            mean_reward = fmean(sample_rewards)
            raw_normalized_mean_reward, normalized_mean_reward = (
                _normalized_absolute_error_reward(
                    baseline_errors, candidate_errors, scale
                )
            )
            task_skill = _skill_score(
                candidate_rmse / scale, baseline_rmse / scale
            )
            normalized_mean_rewards.append(normalized_mean_reward)
            raw_normalized_mean_rewards.append(raw_normalized_mean_reward)
            raw_mae.append(candidate_mae)
            raw_rmse.append(candidate_rmse)
            total_rows += len(candidate_errors)
            invalid = sum(
                not minimum <= float(row["predicted"]) <= maximum
                for row in target_rows_all
            )
            constraint_violations += invalid
            target_results.append(
                {
                    "target": target_name,
                    "unit": unit,
                    "horizon_hours": 1,
                    "n": len(candidate_errors),
                    "eligible_rows": feedback_eligible_rows,
                    "partition_eligible_rows": partition_eligible_rows,
                    "available_rows": available_rows,
                    "deferred_rows": max(0, available_rows - feedback_eligible_rows),
                    "missing_or_nonfinite_rows": feedback_missing_rows,
                    "failed_rows": len(failed_rows),
                    "normalization_scale": scale,
                    "mae": candidate_mae,
                    "rmse": candidate_rmse,
                    "baseline_mae": baseline_mae,
                    "baseline_rmse": baseline_rmse,
                    "normalized_rmse": candidate_rmse / scale,
                    "baseline_normalized_rmse": baseline_rmse / scale,
                    "skill_score": task_skill,
                    "raw_skill_score": (
                        1.0 - (candidate_rmse / scale) / (baseline_rmse / scale)
                        if baseline_rmse / scale > 1e-12
                        else (0.0 if candidate_rmse <= 1e-12 else -1.0)
                    ),
                    "mean_reward": mean_reward,
                    "normalized_mean_reward": normalized_mean_reward,
                    "raw_normalized_mean_reward": raw_normalized_mean_reward,
                    "negative_reward_fraction": (
                        sum(item < 0.0 for item in sample_rewards)
                        / len(sample_rewards)
                    ),
                    "constraint_violations": invalid,
                    "sample_execution_attempted": int(
                        target_execution["attempted_examples"]
                    ),
                    "sample_execution_failed": int(
                        target_execution["failed_examples"]
                    ),
                    "sample_execution_coverage": target_coverage,
                    "sample_execution_coverage_pass": target_coverage_pass,
                    "objective_quality": target_coverage,
                }
            )
            preview_rows.extend(
                target_rows[
                    : min(
                        _PREVIEW_ROWS_PER_TARGET,
                        _PREVIEW_ROWS_TOTAL - len(preview_rows),
                    )
                ]
            )

        if normalized_candidate:
            candidate_nrmse: float | None = fmean(normalized_candidate)
            baseline_nrmse: float | None = fmean(normalized_baseline)
        else:
            candidate_nrmse = None
            baseline_nrmse = None
        unweighted_skill_score = (
            _skill_score(candidate_nrmse, baseline_nrmse)
            if candidate_nrmse is not None and baseline_nrmse is not None
            else -1.0
        )
        objective_aggregate = _aggregate_greenhouse_objective(
            target_results, (1,)
        )
        objective_score = objective_aggregate["weighted_skill_score"]
        per_target_no_regression = all(
            item["normalized_rmse"] is not None
            and item["baseline_normalized_rmse"] is not None
            and item["normalized_rmse"]
            <= item["baseline_normalized_rmse"] + GREENHOUSE_NO_REGRESSION_TOLERANCE
            for item in target_results
        )
        total_eligible_rows = total_rows + total_missing_rows
        sample_execution_coverage = (
            int(sample_batch.summary["succeeded_examples"]) / total_eligible_rows
            if total_eligible_rows
            else 0.0
        )
        per_task_coverage_pass = all(
            bool(item["sample_execution_coverage_pass"])
            for item in target_results
        )
        sample_execution_coverage_pass = (
            total_eligible_rows > 0
            and sample_execution_coverage >= sample_policy.minimum_coverage
            and per_task_coverage_pass
        )
        scientific_pass = (
            objective_score > GREENHOUSE_MIN_SKILL_EXCLUSIVE
            and per_target_no_regression
            and constraint_violations <= GREENHOUSE_MAX_CONSTRAINT_VIOLATIONS
            and sample_execution_coverage_pass
        )
        sample_execution_summary = dict(sample_batch.summary)
        sample_execution_summary.update(
            {
                "adapter_attempt_coverage": sample_batch.summary["coverage"],
                "eligible_examples": total_eligible_rows,
                "succeeded_examples": int(
                    sample_batch.summary["succeeded_examples"]
                ),
                "scored_examples": total_rows,
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
        evaluation_index_digest = _evaluation_index_digest(
            series, evaluation_index_rows
        )
        evaluation_metrics.update(
            {
                "objective_profile": GREENHOUSE_OBJECTIVE_PROFILE_ID,
                "objective_aggregation_version": (
                    GREENHOUSE_OBJECTIVE_AGGREGATION_VERSION
                ),
                "objective_target_weights": dict(GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS),
                "objective_horizons": [1],
                "objective_horizon_weighting": "equal",
                "objective_score": objective_score,
                "overall_reward": objective_aggregate[
                    "weighted_normalized_mean_reward"
                ],
                **objective_aggregate,
                "normalization_scale_method": NORMALIZATION_SCALE_METHOD,
                "normalization_scales": normalization_scales,
                "baseline_profile": baseline_profile,
                "baseline_profile_digest": baseline_profile["digest"],
                "baseline_selection_partition": "training_fit",
                "reward_definition": SAMPLE_REWARD_DEFINITION,
                "positive_reward_is_better": True,
                "promotion_block_evidence": promotion_block_evidence,
                "mean_target_mae_unscaled": fmean(raw_mae) if raw_mae else None,
                "mean_target_rmse_unscaled": fmean(raw_rmse) if raw_rmse else None,
                "raw_units_comparable_across_targets": False,
                "normalized_rmse": candidate_nrmse,
                "baseline_normalized_rmse": baseline_nrmse,
                "skill_score": objective_score,
                "unweighted_skill_score": unweighted_skill_score,
                "improvement": (
                    baseline_nrmse - candidate_nrmse
                    if baseline_nrmse is not None and candidate_nrmse is not None
                    else None
                ),
                "n": total_rows,
                "evaluation_rows": total_rows,
                "execution_mode": "registered_lightweight",
                "fit_method": "bias_fit",
                "evaluation_partition_rows": feedback_range.size,
                "evaluation_eligible_examples": total_rows + total_missing_rows,
                "evaluation_used_examples": total_rows,
                "evaluation_skipped_examples": total_missing_rows,
                **(
                    {
                        "evaluation_partition_eligible_examples": sum(
                            int(item["partition_eligible_rows"])
                            for item in target_results
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
                "sample_execution_records": bounded_sample_execution_records(
                    sample_records
                ),
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
                "per_target_no_regression": per_target_no_regression,
                "scientific_pass": scientific_pass,
                "targets": target_results,
                "horizons": [
                    {
                        "horizon_hours": 1,
                        "normalized_rmse": candidate_nrmse,
                        "baseline_normalized_rmse": baseline_nrmse,
                        "skill_score": objective_score,
                        "n": total_rows,
                        "sample_execution_coverage": sample_execution_coverage,
                        "sample_execution_coverage_pass": (
                            sample_execution_coverage_pass
                        ),
                    }
                ],
                "prediction_preview": preview_rows,
                "evaluation_cohort": {
                    "partition": "training_feedback",
                    "history_source_partitions": [
                        "training_fit",
                        "training_feedback",
                    ],
                    "minimum_history_hours": MAX_ROLLING_WINDOW_HOURS,
                    "embargo_rows_visible": False,
                    **(
                        {"update_window": feedback_update_cohort}
                        if feedback_update_cohort is not None
                        else {}
                    ),
                },
                "evaluation_index_digest": evaluation_index_digest,
                "baseline_metrics_digest": _baseline_metrics_digest(
                    evaluation_index_digest, target_results
                ),
                "dataset_digest": series.digest,
                "split_manifest_digest_sha256": series.split_manifest_digest_sha256,
                "evaluation_scope": "visible/training_feedback/historical_replay",
                "causal_interpretation": False,
            }
        )
        artifact = ModelArtifact(
            artifact_id=f"artifact:{candidate.candidate_id}",
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            model_id=GREENHOUSE_ROLLING_PREDICTOR_ID,
            dataset_digest=series.digest,
            training_partition="training_fit",
            training_rows=fit_range.size,
            parameters=parameters,
            learned_parameters=learned,
            metrics={
                **training_metrics,
                # The registered residual predictor performs one scalar bias
                # fit over the frozen training range, then a rolling score.
                # It is deliberately lightweight rather than epoch-based
                # neural-network training.
                "execution_mode": "registered_lightweight",
                "fit_method": "bias_fit",
                "fit_passes_requested": 1,
                "fit_passes_completed": 1,
                "iterative_epoch_training": False,
                # Compatibility aliases; this predictor is not epoch trained.
                "epochs_requested": 1,
                "epochs_completed": 1,
                "training_rows": fit_range.size,
                "training_partition_rows": fit_range.size,
                "training_eligible_examples": total_fit_eligible_rows,
                "training_used_examples": total_fit_rows,
                "training_skipped_examples": total_fit_missing_rows,
            },
        )
        evaluation = Evaluation(
            evaluation_id=f"evaluation:{candidate.candidate_id}:training_feedback",
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            score=objective_score,
            passed=scientific_pass,
            metrics=evaluation_metrics,
            partition="training_feedback",
            evaluator_digest=str(
                task.metadata.get("evaluator_digest")
                or self.evaluator_configuration_digest(GREENHOUSE_EVALUATOR_ID)
            ),
            artifact_digest=artifact.digest,
        )
        return EvaluationBundle(
            artifact=artifact,
            evaluation=evaluation,
            sample_results=build_sample_results(
                candidate.candidate_id, scoring_rows
            ),
        )

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
        on_model_usage: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
        on_sample_control: Callable[[], str] | None = None,
        execution_plan: DerivedExecutionPlan | None = None,
        predictor_model_id: str = EXOGENOUS_RIDGE_MODEL_ID,
    ) -> EvaluationBundle:
        execution_plan = execution_plan or self._resolve_execution_plan(
            candidate, proposal, None
        )
        parameters = (
            HorizonTargetwiseExogenousRidgeConfig.from_mapping(proposal.changes)
            if predictor_model_id == HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
            else TargetwiseExogenousRidgeConfig.from_mapping(proposal.changes)
            if predictor_model_id == TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
            else ExogenousRidgeConfig.from_mapping(proposal.changes)
        )
        defer_feedback_prediction = (
            not self._sample_executor_injected
            and task.metadata.get("sample_agent_mode") == "gateway_microbatch"
        )
        prediction = fit_predict_exogenous_ridge(
            series,
            targets=tuple(item[0] for item in _TARGETS),
            horizons=horizons,
            config=parameters,
            evaluation_history_steps=MAX_EXOGENOUS_RIDGE_HISTORY_STEPS,
            defer_prediction_partitions=(
                ("training_feedback",) if defer_feedback_prediction else ()
            ),
            on_fit_complete=on_training_complete,
        )
        generated_rows = prediction["prediction_rows"]
        models = prediction["models"]
        fit_range = series.partitions["training_fit"]
        feedback_range = series.partitions["training_feedback"]
        target_metadata = {
            name: {"unit": unit, "minimum": minimum, "maximum": maximum}
            for name, unit, minimum, maximum in _TARGETS
        }
        algorithm_name, separator, algorithm_revision = (
            predictor_model_id.rpartition("@")
        )
        if not separator:
            algorithm_name = predictor_model_id
            algorithm_revision = "unversioned"
        sample_policy = self._sample_execution_policy(task, execution_plan)
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
        feedback_update_cohort: dict[str, Any] | None = None
        samples_per_update = _feedback_update_limit(task)
        if samples_per_update is not None:
            generated_feedback_rows, feedback_update_cohort = (
                _select_feedback_update_cohort(
                    generated_feedback_rows,
                    generation=candidate.generation,
                    samples_per_update=samples_per_update,
                    dataset_digest=series.digest,
                    split_manifest_digest=series.split_manifest_digest_sha256,
                )
            )
        selected_task_counts = _cohort_task_counts(
            feedback_update_cohort, "selected_count"
        )
        available_task_counts = _cohort_task_counts(
            feedback_update_cohort, "population_count"
        )
        baseline_profile = fit_baseline_profile(
            series,
            targets=tuple(item[0] for item in _TARGETS),
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
            for target_name, _unit, _minimum, _maximum in _TARGETS
        }

        def finalize_scoring_rows(
            result_rows: Sequence[Mapping[str, Any]],
        ) -> list[dict[str, Any]]:
            return _apply_scoring_baseline(
                series,
                result_rows,
                baseline_profile,
                baseline_scales,
            )

        def publish_scoring_rows(
            result_rows: Sequence[Mapping[str, Any]],
        ) -> None:
            if on_sample_results is not None:
                on_sample_results(finalize_scoring_rows(result_rows))

        def candidate_forecast_tool(request: SamplePredictionRequest) -> float:
            return predict_fitted_exogenous_ridge(
                target=request.target,
                horizon_hours=request.horizon_hours,
                baseline=request.baseline,
                label_free_context=request.label_free_context,
                models=models,
                config=parameters,
            )

        def conservative_ridge_tool(
            request: SamplePredictionRequest,
            execution_context: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            del execution_context
            candidate_prediction = candidate_forecast_tool(request)
            return {
                "predicted": request.baseline
                + 0.5 * (candidate_prediction - request.baseline),
                "metadata": {
                    "source_model_id": predictor_model_id,
                    "residual_multiplier": 0.5,
                    "execution_mode": "selected_host_tool_on_demand",
                },
            }

        alternate_tools = (
            (
                GatewaySampleTool(
                    tool_id="greenhouse-ridge-conservative@1",
                    version="1",
                    handler=conservative_ridge_tool,
                    purpose=(
                        "conservative_half_residual_ridge_prediction_on_demand"
                    ),
                ),
            )
            if defer_feedback_prediction
            else ()
        )
        sample_batch = self._sample_executor_for_task(
            task,
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            progress_callback=on_evaluation_progress,
            model_usage_callback=on_model_usage,
            on_sample_control=on_sample_control,
            forecast_tool=(
                candidate_forecast_tool if defer_feedback_prediction else None
            ),
            tools=alternate_tools,
        ).execute(
            generated_feedback_rows,
            context={
                "run_id": candidate.run_id,
                "candidate_id": candidate.candidate_id,
                **self._dsh_sample_stage_context(task, candidate, proposal),
                "dataset_digest": series.digest,
                "split_manifest_digest_sha256": series.split_manifest_digest_sha256,
                "partition": "training_feedback",
                "algorithm_id": algorithm_name,
                "algorithm_version": algorithm_revision,
                "evaluator_id": str(
                    task.metadata.get("evaluator_id")
                    or GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID
                ),
                "horizons_hours": list(horizons),
                "candidate_parameters": parameters.to_dict(),
                "proposal_plan": dict(proposal.metadata),
                "derived_execution_plan": execution_plan.to_dict(),
                "sample_agent_mode": task.metadata.get(
                    "sample_agent_mode", "host_feedback_state_machine"
                ),
                "sample_agent_batch_size": task.metadata.get(
                    "sample_agent_batch_size", 128
                ),
                "sample_concurrency": task.metadata.get(
                    "sample_concurrency", 4
                ),
                **(
                    {
                        "samples_per_update": samples_per_update,
                        "feedback_update_cohort": feedback_update_cohort,
                    }
                    if feedback_update_cohort is not None
                    else {}
                ),
                **(
                    {
                        "sample_planner_prompt_profile": task.metadata[
                            "sample_planner_prompt_profile"
                        ]
                    }
                    if task.metadata.get("sample_planner_prompt_profile") is not None
                    else {}
                ),
                "strategy_model_id": task.metadata.get("strategy_model_id"),
                "review_model_id": task.metadata.get("review_model_id"),
                "tool_experience": proposal.metadata.get("tool_experience", []),
                "algorithm_artifact_digest": digest(
                    {
                        "prediction_model_id": predictor_model_id,
                        "parameters": parameters.to_dict(),
                        "models": models,
                        "dataset_digest": series.digest,
                    }
                ),
            },
            target_bounds=target_metadata,
            algorithm_id=algorithm_name,
            algorithm_version=algorithm_revision,
            policy=sample_policy,
            result_callback=(
                publish_scoring_rows if on_sample_results is not None else None
            ),
            checkpoint_callback=on_sample_checkpoint,
        )
        # Training rows stay local to the registered fit. Every feedback row
        # is replaced by its independently adjudicated result; a failed tool
        # call first uses the model's declared persistence fallback. Scoring
        # then applies the fit-selected comparator and prevents a failed call
        # from receiving a positive reward.
        scored_feedback_rows = finalize_scoring_rows(sample_batch.scoring_rows)
        promotion_block_evidence = build_promotion_block_evidence(
            scored_feedback_rows,
            horizons=horizons,
            target_weights=GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS,
            dataset_digest=series.digest,
            split_manifest_digest_sha256=series.split_manifest_digest_sha256,
        )
        rows = [
            row for row in generated_rows if row["partition"] == "training_fit"
        ] + scored_feedback_rows
        sample_records = list(sample_batch.records)
        task_results: list[dict[str, Any]] = []
        training_metrics: dict[str, Any] = {}
        preview_rows: list[dict[str, Any]] = []
        normalized_candidate: list[float] = []
        normalized_baseline: list[float] = []
        normalized_mean_rewards: list[float] = []
        raw_normalized_mean_rewards: list[float] = []
        normalization_scales: dict[str, dict[str, Any]] = {}
        raw_mae: list[float] = []
        raw_rmse: list[float] = []
        total_rows = 0
        total_missing_rows = 0
        total_fit_eligible_rows = 0
        total_fit_rows = 0
        total_fit_missing_rows = 0
        constraint_violations = 0
        incomplete_prediction_tasks = 0
        evaluation_index_rows: list[dict[str, Any]] = []
        preview_per_task = max(
            1, _PREVIEW_ROWS_TOTAL // (len(_TARGETS) * len(horizons))
        )

        for target_name, _unit, _minimum, _maximum in _TARGETS:
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
                # Failed gateway samples remain in the durable sample archive
                # for inspection, but their fallback prediction is not model
                # evidence and must not enter RMSE/MAE/reward aggregation.
                feedback_rows = [
                    row
                    for row in feedback_rows_all
                    if str(row.get("sample_execution_status", "succeeded"))
                    .strip()
                    .casefold()
                    != "failed"
                    and not row.get("scoring_fallback")
                ]
                failed_feedback_rows = [
                    row for row in feedback_rows_all if row not in feedback_rows
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
                missing_rows = max(0, eligible_rows - len(feedback_rows))
                metadata = target_metadata[target_name]
                task_execution = next(
                    (
                        item
                        for item in sample_batch.summary["tasks"]
                        if item["target"] == target_name
                        and item["horizon_hours"] == horizon
                    ),
                    {
                        "attempted_examples": 0,
                        "succeeded_examples": 0,
                        "failed_examples": 0,
                    },
                )
                task_execution_failed = int(task_execution["failed_examples"])
                task_coverage = (
                    int(task_execution["succeeded_examples"]) / eligible_rows
                    if eligible_rows
                    else 0.0
                )
                task_coverage_pass = (
                    eligible_rows > 0
                    and task_coverage >= sample_policy.minimum_task_coverage
                )
                total_missing_rows += missing_rows
                evaluation_index_rows.extend(
                    {
                        "target": target_name,
                        "horizon_hours": horizon,
                        "origin_timestamp": row["origin_timestamp"],
                        "target_timestamp": row["timestamp"],
                    }
                    for row in feedback_rows_all
                )

                # A depleted task is a failed scientific gate, not a candidate
                # execution exception.  This preserves all other sample
                # evidence and lets the next generation learn from the failure.
                if not feedback_rows:
                    incomplete_prediction_tasks += 1
                    task_results.append(
                        {
                            "target": target_name,
                            "unit": metadata["unit"],
                            "horizon_hours": horizon,
                            "n": 0,
                            "eligible_rows": eligible_rows,
                            "partition_eligible_rows": partition_eligible_rows,
                            "available_rows": available_rows,
                            "deferred_rows": max(0, available_rows - eligible_rows),
                            "missing_or_nonfinite_rows": missing_rows,
                            "failed_rows": len(failed_feedback_rows),
                            "normalization_scale": scale,
                            "mae": None,
                            "rmse": None,
                            "baseline_mae": None,
                            "baseline_rmse": None,
                            "normalized_rmse": None,
                            "baseline_normalized_rmse": None,
                            "skill_score": -1.0,
                            "mean_reward": None,
                            "normalized_mean_reward": None,
                            "negative_reward_fraction": None,
                            "constraint_violations": 0,
                            "sample_execution_attempted": int(
                                task_execution["attempted_examples"]
                            ),
                            "sample_execution_failed": task_execution_failed,
                            "sample_execution_coverage": task_coverage,
                            "sample_execution_coverage_pass": False,
                            "objective_quality": 0.0,
                        }
                    )
                    continue
                candidate_errors = [
                    float(row["predicted"]) - float(row["observed"])
                    for row in feedback_rows
                ]
                baseline_errors = [
                    float(row["baseline"]) - float(row["observed"])
                    for row in feedback_rows
                ]
                candidate_mae = _mae(candidate_errors)
                candidate_rmse = _rmse(candidate_errors)
                baseline_mae = _mae(baseline_errors)
                baseline_rmse = _rmse(baseline_errors)
                candidate_nrmse = candidate_rmse / scale
                baseline_nrmse = baseline_rmse / scale
                # This is the exact aggregate of the durable per-sample reward:
                # abs(baseline - observed) - abs(predicted - observed).
                sample_rewards = [
                    abs(baseline_error) - abs(candidate_error)
                    for baseline_error, candidate_error in zip(
                        baseline_errors, candidate_errors
                    )
                ]
                mean_reward = fmean(sample_rewards)
                raw_normalized_mean_reward, normalized_mean_reward = (
                    _normalized_absolute_error_reward(
                        baseline_errors, candidate_errors, scale
                    )
                )
                skill = _skill_score(candidate_nrmse, baseline_nrmse)
                invalid = sum(
                    1
                    for row in feedback_rows
                    if float(row["predicted"]) < metadata["minimum"]
                    or float(row["predicted"]) > metadata["maximum"]
                )
                task_results.append(
                    {
                        "target": target_name,
                        "unit": metadata["unit"],
                        "horizon_hours": horizon,
                        "n": len(feedback_rows),
                        "eligible_rows": eligible_rows,
                        "partition_eligible_rows": partition_eligible_rows,
                        "available_rows": available_rows,
                        "deferred_rows": max(0, available_rows - eligible_rows),
                        "missing_or_nonfinite_rows": missing_rows,
                        "failed_rows": len(failed_feedback_rows),
                        "normalization_scale": scale,
                        "mae": candidate_mae,
                        "rmse": candidate_rmse,
                        "baseline_mae": baseline_mae,
                        "baseline_rmse": baseline_rmse,
                        "normalized_rmse": candidate_nrmse,
                        "baseline_normalized_rmse": baseline_nrmse,
                        "skill_score": skill,
                        "raw_skill_score": (
                            1.0 - candidate_nrmse / baseline_nrmse
                            if baseline_nrmse > 1e-12
                            else (0.0 if candidate_nrmse <= 1e-12 else -1.0)
                        ),
                        "mean_reward": mean_reward,
                        "normalized_mean_reward": normalized_mean_reward,
                        "raw_normalized_mean_reward": raw_normalized_mean_reward,
                        "negative_reward_fraction": (
                            sum(item < 0.0 for item in sample_rewards)
                            / len(sample_rewards)
                        ),
                        "constraint_violations": invalid,
                        "sample_execution_attempted": int(
                            task_execution["attempted_examples"]
                        ),
                        "sample_execution_failed": task_execution_failed,
                        "sample_execution_coverage": task_coverage,
                        "sample_execution_coverage_pass": task_coverage_pass,
                        "objective_quality": task_coverage,
                    }
                )
                normalized_candidate.append(candidate_nrmse)
                normalized_baseline.append(baseline_nrmse)
                normalized_mean_rewards.append(normalized_mean_reward)
                raw_normalized_mean_rewards.append(raw_normalized_mean_reward)
                raw_mae.append(candidate_mae)
                raw_rmse.append(candidate_rmse)
                total_rows += len(feedback_rows)
                constraint_violations += invalid
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

        objective_aggregate = _aggregate_greenhouse_objective(
            task_results,
            horizons,
        )
        if normalized_candidate:
            overall_nrmse: float | None = fmean(normalized_candidate)
            overall_baseline_nrmse: float | None = fmean(normalized_baseline)
            overall_skill = _skill_score(
                overall_nrmse, overall_baseline_nrmse
            )
        else:
            overall_nrmse = None
            overall_baseline_nrmse = None
            overall_skill = -1.0
        per_task_no_regression = all(
            item["normalized_rmse"] is not None
            and item["baseline_normalized_rmse"] is not None
            and item["normalized_rmse"]
            <= item["baseline_normalized_rmse"] + GREENHOUSE_NO_REGRESSION_TOLERANCE
            for item in task_results
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
            > GREENHOUSE_MIN_SKILL_EXCLUSIVE
            and per_task_no_regression
            and constraint_violations <= GREENHOUSE_MAX_CONSTRAINT_VIOLATIONS
            and sample_execution_coverage_pass
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
                        _skill_score(horizon_nrmse, horizon_baseline)
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
        sample_execution_summary.update(
            {
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

        evaluator_id = str(task.metadata.get("evaluator_id") or GREENHOUSE_EVALUATOR_ID)
        evaluator_digest = str(
            task.metadata.get("evaluator_digest")
            or self.evaluator_configuration_digest(evaluator_id)
        )
        evaluation_index_digest = _evaluation_index_digest(
            series, evaluation_index_rows
        )
        evaluation_metrics = {
            "objective_profile": GREENHOUSE_OBJECTIVE_PROFILE_ID,
            "objective_aggregation_version": (
                GREENHOUSE_OBJECTIVE_AGGREGATION_VERSION
            ),
            "objective_target_weights": dict(GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS),
            "objective_horizons": list(horizons),
            # Explicit objective fields are additive. ``skill_score`` and
            # ``mean_normalized_reward`` retain their historical semantics.
            "objective_score": objective_aggregate["weighted_skill_score"],
            "overall_reward": objective_aggregate[
                "weighted_normalized_mean_reward"
            ],
            **objective_aggregate,
            "normalization_scale_method": NORMALIZATION_SCALE_METHOD,
            "normalization_scales": normalization_scales,
            "baseline_profile": baseline_profile,
            "baseline_profile_digest": baseline_profile["digest"],
            "baseline_selection_partition": "training_fit",
            "promotion_block_evidence": promotion_block_evidence,
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
            "per_target_no_regression": per_task_no_regression,
            "scientific_pass": scientific_pass,
            "targets": task_results,
            "horizons": horizon_results,
            "prediction_preview": preview_rows[:_PREVIEW_ROWS_TOTAL],
            "evaluation_cohort": {
                "partition": "training_feedback",
                "history_source_partitions": ["training_feedback"],
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
            "evaluation_scope": "visible/training_feedback/historical_replay",
            "causal_interpretation": False,
        }
        training_metrics.update(
            {
                "model_task_count": len(models),
                "solver_fallback_count": sum(
                    item["status"] != "fitted" for item in models
                ),
                "selected_exogenous_feature_count": len(
                    {
                        name
                        for item in models
                        for name in item["selected_exogenous_features"]
                    }
                ),
                "execution_mode": "closed_form_ridge",
                "fit_method": "closed_form_ridge",
                "fit_passes_requested": 1,
                "fit_passes_completed": 1,
                "iterative_epoch_training": False,
                # Compatibility aliases; ridge is solved in one closed-form fit.
                "epochs_requested": 1,
                "epochs_completed": 1,
                "training_rows": fit_range.size,
                "training_partition_rows": fit_range.size,
                "training_eligible_examples": total_fit_eligible_rows,
                "training_used_examples": total_fit_rows,
                "training_skipped_examples": total_fit_missing_rows,
            }
        )
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

    def apply_judge(
        self,
        task: TaskManifest,
        proposal: Proposal,
        bundle: EvaluationBundle,
    ) -> EvaluationBundle:
        judge_model_id = str(task.metadata.get("judge_model_id") or RULE_JUDGE_ID)
        metrics = dict(bundle.evaluation.metrics)
        scientific_pass = bool(metrics.get("scientific_pass", bundle.evaluation.passed))
        judge_parameter_override: dict[str, Any] = {}
        if judge_model_id == RULE_JUDGE_ID:
            judge_accepted = scientific_pass
            guidance = (
                "候选通过固定科学门槛。"
                if judge_accepted
                else "候选未达到固定科学门槛。"
            )
        else:
            result = self.model_gateway.judge(
                judge_model_id,
                {
                    "title": proposal.title,
                    "changes": dict(proposal.changes),
                    "rationale": proposal.rationale,
                },
                _judge_metrics(metrics),
            )
            judge_accepted = bool(result["accepted"])
            guidance = str(result.get("guidance") or "独立模型未提供补充意见。")
            raw_override = result.get("parameter_override", {})
            if raw_override:
                self.validate_parameter_overrides(task, raw_override)
                judge_parameter_override = dict(raw_override)
        passed = scientific_pass and judge_accepted
        metrics.update(
            {
                "scientific_pass": scientific_pass,
                "judge_model_id": judge_model_id,
                "judge_accepted": judge_accepted,
                "judge_guidance": guidance,
                "judge_parameter_override": judge_parameter_override,
            }
        )
        evaluation = Evaluation(
            evaluation_id=bundle.evaluation.evaluation_id,
            run_id=bundle.evaluation.run_id,
            candidate_id=bundle.evaluation.candidate_id,
            score=bundle.evaluation.score,
            passed=passed,
            metrics=metrics,
            partition=bundle.evaluation.partition,
            evaluator_digest=bundle.evaluation.evaluator_digest,
            artifact_digest=bundle.evaluation.artifact_digest,
            created_at=bundle.evaluation.created_at,
        )
        return EvaluationBundle(
            artifact=bundle.artifact,
            evaluation=evaluation,
            sample_results=bundle.sample_results,
        )


__all__ = [
    "DEFAULT_SAMPLE_EXECUTION_MIN_COVERAGE",
    "EXOGENOUS_RIDGE_MODEL_ID",
    "HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID",
    "TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID",
    "GREENHOUSE_EVALUATOR_ID",
    "GREENHOUSE_MULTIHORIZON_EVALUATOR_ID",
    "GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID",
    "GREENHOUSE_ROLLING_PREDICTOR_ID",
    "RULE_JUDGE_ID",
    "TOY_EVALUATOR_ID",
    "TOY_PREDICTOR_MODEL_ID",
    "CollaborativeSampleExecutor",
    "EvaluationBundle",
    "EvaluatorRegistry",
    "SampleExecutionPolicy",
    "artifact_set_digest",
]
