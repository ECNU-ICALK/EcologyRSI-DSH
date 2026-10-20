"""Training-fit-only smoke execution for restricted algorithm IRs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..core.models import TaskManifest, digest
from ..data.contracts import DatasetSeries, SelectionDatasetView
from ..data.splits import IndexRange
from ..evaluators.greenhouse_prediction import (
    BASELINE_ALIGNED_RIDGE_MODEL_ID,
    BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID,
    BaselineAlignedRidgeConfig,
    BaselineAlignedCellwiseRidgeConfig,
    ExogenousRidgeConfig,
    HorizonTargetwiseExogenousRidgeConfig,
    TargetwiseExogenousRidgeConfig,
    fit_predict_exogenous_ridge,
)
from .algorithm_ir import AlgorithmIR
from .algorithms import AlgorithmSpec

ALGORITHM_SMOKE_VERSION = "ecologyrsi-dsh.algorithm-smoke/1"

_RIDGE_PREDICTOR_ID = "greenhouse-exogenous-ridge@1"
_TARGETWISE_RIDGE_PREDICTOR_ID = "greenhouse-targetwise-ridge@1"
_HORIZON_TARGETWISE_RIDGE_PREDICTOR_ID = (
    "greenhouse-horizon-targetwise-ridge@1"
)
_TARGETS = (
    ("air_temperature", -10.0, 60.0),
    ("relative_humidity", 0.0, 100.0),
    ("co2_concentration", 0.0, 5000.0),
)


class AlgorithmSmokeError(RuntimeError):
    """A registered algorithm could not pass its bounded smoke execution."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)[:100]
        self.retryable = bool(retryable)
        self.evidence = dict(evidence or {})


def _finalize_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["smoke_digest"] = digest(result)
    return result


def _retry_context(
    attempt: int,
    failure_feedback: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise AlgorithmSmokeError(
            "smoke_retry_feedback_invalid",
            "smoke attempt must be a positive integer",
        )
    if len(failure_feedback) != attempt - 1:
        raise AlgorithmSmokeError(
            "smoke_retry_feedback_invalid",
            "smoke retry must carry one persisted failure record per prior attempt",
        )
    normalized: list[dict[str, Any]] = []
    for expected_attempt, raw in enumerate(failure_feedback, start=1):
        if not isinstance(raw, Mapping):
            raise AlgorithmSmokeError(
                "smoke_retry_feedback_invalid",
                "smoke retry feedback must contain only objects",
            )
        if raw.get("attempt") != expected_attempt or raw.get("retryable") is not True:
            raise AlgorithmSmokeError(
                "smoke_retry_feedback_invalid",
                "only sequential transient smoke failures may retry the same algorithm IR",
            )
        normalized.append(dict(raw))
    return {
        "attempt": attempt,
        "prior_failure_count": len(normalized),
        "prior_failure_digest": digest(normalized) if normalized else None,
        "retry_mode": (
            "initial_execution"
            if attempt == 1
            else "same_immutable_ir_after_transient_runtime_failure"
        ),
        "algorithm_revision_policy": "new_ir_requires_new_proposal",
    }


def _attach_retry_context(
    evidence: Mapping[str, Any],
    *,
    attempt: int,
    failure_feedback: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    result = dict(evidence)
    result.pop("smoke_digest", None)
    result["retry_context"] = _retry_context(attempt, failure_feedback)
    return _finalize_evidence(result)


def _operator_trace(algorithm_ir: AlgorithmIR) -> list[dict[str, str]]:
    return [
        {
            "operator_id": str(item["operator_id"]),
            "stage": str(item["stage"]),
            "status": "completed",
        }
        for item in algorithm_ir.operators
    ]






def _load_greenhouse_series(
    task: TaskManifest,
    datasets: Any,
) -> DatasetSeries:
    dataset_id = task.dataset
    if dataset_id is None or not callable(getattr(datasets, "selection_view", None)):
        raise AlgorithmSmokeError(
            "smoke_dataset_unavailable",
            "greenhouse smoke execution requires the registered dataset runtime",
        )
    try:
        series = datasets.selection_view(
            dataset_id,
            str(task.metadata.get("episode_id"))
            if task.metadata.get("episode_id") is not None
            else None,
            expected_dataset_digest=str(task.metadata["dataset_digest"]),
            expected_split_manifest_digest=str(
                task.metadata["split_manifest_digest"]
            ),
        )
    except (OSError, TimeoutError, ConnectionError) as exc:
        raise AlgorithmSmokeError(
            "smoke_dataset_transient",
            f"greenhouse smoke dataset is temporarily unavailable: {type(exc).__name__}",
            retryable=True,
        ) from exc
    if not isinstance(series, SelectionDatasetView):
        raise AlgorithmSmokeError(
            "smoke_dataset_contract",
            "registered dataset runtime did not return a SelectionDatasetView",
        )
    if (
        series.digest != task.metadata["dataset_digest"]
        or series.split_manifest_digest_sha256
        != task.metadata["split_manifest_digest"]
    ):
        raise AlgorithmSmokeError(
            "smoke_dataset_snapshot_mismatch",
            "greenhouse smoke dataset does not match the frozen task snapshot",
        )
    return series




def _smoke_split(
    series: DatasetSeries,
    *,
    history_steps: int,
    maximum_horizon: int,
) -> DatasetSeries:
    source = series.partitions["training_fit"]
    minimum_width = history_steps + maximum_horizon + 2
    if source.size < minimum_width * 2 + 1:
        raise AlgorithmSmokeError(
            "smoke_insufficient_training_rows",
            "training_fit is too small for a causal ridge smoke split",
            evidence={
                "available_rows": source.size,
                "minimum_rows": minimum_width * 2 + 1,
            },
        )
    feedback_start = source.end - minimum_width
    fit_end = feedback_start - 1
    return DatasetSeries(
        schema=series.schema,
        dataset_id=series.dataset_id,
        domain_id=series.domain_id,
        episode_id=series.episode_id,
        digest=series.digest,
        timestamps=tuple(series.timestamps[: source.end]),
        values={name: tuple(values[: source.end]) for name, values in series.values.items()},
        partitions={
            "training_fit": IndexRange(source.start, fit_end),
            "training_feedback": IndexRange(feedback_start, source.end),
            "development": IndexRange(source.end, source.end),
        },
        features=dict(series.features),
        split_manifest_digest_sha256=series.split_manifest_digest_sha256,
    )


def _ridge_smoke(
    spec: AlgorithmSpec,
    series: DatasetSeries,
    algorithm_ir: AlgorithmIR,
) -> dict[str, Any]:
    config = (
        BaselineAlignedCellwiseRidgeConfig.from_mapping(spec.parameters)
        if spec.adapter_id == BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID
        else BaselineAlignedRidgeConfig.from_mapping(spec.parameters)
        if spec.adapter_id == BASELINE_ALIGNED_RIDGE_MODEL_ID
        else HorizonTargetwiseExogenousRidgeConfig.from_mapping(spec.parameters)
        if spec.adapter_id == _HORIZON_TARGETWISE_RIDGE_PREDICTOR_ID
        else TargetwiseExogenousRidgeConfig.from_mapping(spec.parameters)
        if spec.adapter_id == _TARGETWISE_RIDGE_PREDICTOR_ID
        else ExogenousRidgeConfig.from_mapping(spec.parameters)
    )
    horizons = (1, 6, 24)
    smoke_series = _smoke_split(
        series,
        history_steps=config.history_steps,
        maximum_horizon=max(horizons),
    )
    result = fit_predict_exogenous_ridge(
        smoke_series,
        targets=tuple(item[0] for item in _TARGETS),
        horizons=horizons,
        config=config,
    )
    rows = list(result["prediction_rows"])
    if not rows:
        raise AlgorithmSmokeError(
            "smoke_no_predictions",
            "ridge smoke execution produced no predictions",
        )
    bounds = {name: (minimum, maximum) for name, minimum, maximum in _TARGETS}
    violations = 0
    for row in rows:
        predicted = row.get("predicted")
        if (
            isinstance(predicted, bool)
            or not isinstance(predicted, (int, float))
            or not math.isfinite(float(predicted))
        ):
            raise AlgorithmSmokeError(
                "smoke_nonfinite_output",
                "ridge smoke execution returned a non-finite prediction",
            )
        minimum, maximum = bounds[str(row["target"])]
        violations += not minimum <= float(predicted) <= maximum
    if violations:
        raise AlgorithmSmokeError(
            "smoke_constraint_violation",
            "ridge smoke execution violated registered physical bounds",
            evidence={"constraint_violations": violations},
        )
    return _finalize_evidence(
        {
            "schema_version": ALGORITHM_SMOKE_VERSION,
            "status": "passed",
            "dataset_kind": "observed",
            "dataset_digest": series.digest,
            "source_partition": "training_fit",
            "smoke_fit_range": smoke_series.partitions["training_fit"].to_dict(),
            "smoke_feedback_range": smoke_series.partitions[
                "training_feedback"
            ].to_dict(),
            "restricted_partition_access": False,
            "predictor_id": spec.adapter_id,
            "algorithm_ir_digest": algorithm_ir.ir_digest,
            "rows_examined": series.partitions["training_fit"].size,
            "usable_predictions": len(rows),
            "finite_outputs": True,
            "constraint_violations": 0,
            "operator_trace": _operator_trace(algorithm_ir),
        }
    )


def smoke_test_algorithm_spec(
    spec: AlgorithmSpec,
    task: TaskManifest,
    datasets: Any = None,
    *,
    attempt: int = 1,
    failure_feedback: tuple[Mapping[str, Any], ...] = (),
) -> dict[str, Any]:
    """Execute one registered IR against synthetic or observed training data."""

    if spec.algorithm_ir is None:
        raise AlgorithmSmokeError(
            "smoke_missing_algorithm_ir",
            "compiled candidate has no restricted algorithm IR",
        )
    algorithm_ir = AlgorithmIR.from_dict(spec.algorithm_ir)
    if not isinstance(task.metadata.get("dataset_digest"), str) or not isinstance(
        task.metadata.get("split_manifest_digest"), str
    ):
        raise AlgorithmSmokeError("smoke_missing_snapshot", "task requires frozen dataset and split identities")
    series = _load_greenhouse_series(task, datasets)
    if spec.adapter_id not in {
        BASELINE_ALIGNED_RIDGE_MODEL_ID, BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID,
        _RIDGE_PREDICTOR_ID, _TARGETWISE_RIDGE_PREDICTOR_ID, _HORIZON_TARGETWISE_RIDGE_PREDICTOR_ID,
    }:
        raise AlgorithmSmokeError("smoke_unregistered_predictor", "no registered smoke runner for predictor")
    evidence = _ridge_smoke(spec, series, algorithm_ir)
    return _attach_retry_context(
        evidence,
        attempt=attempt,
        failure_feedback=failure_feedback,
    )


__all__ = [
    "ALGORITHM_SMOKE_VERSION",
    "AlgorithmSmokeError",
    "smoke_test_algorithm_spec",
]
