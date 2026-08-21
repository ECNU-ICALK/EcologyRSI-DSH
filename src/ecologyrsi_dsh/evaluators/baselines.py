"""Leakage-safe fit selection and application of greenhouse baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from ..core.models import digest
from ..data.contracts import DatasetSeries


BASELINE_PROFILE_VERSION = "fit_selected_persistence_or_seasonal_24h@1"
BASELINE_SELECTION_TOLERANCE = 1e-12


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _rmse(errors: Sequence[float]) -> float:
    if not errors:
        raise ValueError("baseline comparison requires at least one error")
    return math.sqrt(sum(value * value for value in errors) / len(errors))


def _validated_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    result = tuple(horizons)
    if not result:
        raise ValueError("horizons must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in result
    ):
        raise ValueError("horizons must contain positive integers")
    if len(set(result)) != len(result):
        raise ValueError("horizons must not contain duplicates")
    return result


def fit_baseline_profile(
    series: DatasetSeries,
    *,
    targets: Sequence[str],
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Select one comparator per target/horizon using ``training_fit`` only.

    Persistence predicts the label from the forecast origin.  The seasonal
    comparator predicts it from 24 hours before the label and is eligible only
    when that reference is available no later than the origin.  Both models
    are compared on the same fit rows; exact ties retain persistence.
    """

    fit_range = series.partitions.get("training_fit")
    if fit_range is None:
        raise ValueError("series is missing training_fit")
    resolved_targets = tuple(str(target).strip() for target in targets)
    if not resolved_targets or any(not target for target in resolved_targets):
        raise ValueError("targets must contain non-empty names")
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("targets must not contain duplicates")
    resolved_horizons = _validated_horizons(horizons)
    timestamp_index = {
        series.timestamps[index]: index
        for index in range(fit_range.start, fit_range.end)
    }
    cells: list[dict[str, Any]] = []
    for target in resolved_targets:
        if target not in series.values:
            raise ValueError(f"series is missing baseline target {target}")
        values = series.values[target]
        for horizon in resolved_horizons:
            persistence_errors: list[float] = []
            seasonal_errors: list[float] = []
            seasonal_causal = horizon <= 24
            if seasonal_causal:
                for label_index in range(fit_range.start, fit_range.end):
                    label_timestamp = series.timestamps[label_index]
                    origin_index = timestamp_index.get(label_timestamp - horizon)
                    seasonal_index = timestamp_index.get(label_timestamp - 24)
                    observed = _finite(values[label_index])
                    persistence = (
                        _finite(values[origin_index])
                        if origin_index is not None
                        else None
                    )
                    seasonal = (
                        _finite(values[seasonal_index])
                        if seasonal_index is not None
                        else None
                    )
                    if observed is None or persistence is None or seasonal is None:
                        continue
                    persistence_errors.append(persistence - observed)
                    seasonal_errors.append(seasonal - observed)
            comparison_n = len(persistence_errors)
            persistence_rmse = (
                _rmse(persistence_errors) if persistence_errors else None
            )
            seasonal_rmse = _rmse(seasonal_errors) if seasonal_errors else None
            seasonal_wins = bool(
                persistence_rmse is not None
                and seasonal_rmse is not None
                and seasonal_rmse
                < persistence_rmse - BASELINE_SELECTION_TOLERANCE
            )
            cells.append(
                {
                    "target": target,
                    "horizon_hours": horizon,
                    "baseline_id": (
                        "seasonal_24h" if seasonal_wins else "persistence"
                    ),
                    "comparison_n": comparison_n,
                    "persistence_rmse": persistence_rmse,
                    "seasonal_24h_rmse": seasonal_rmse,
                    "seasonal_status": (
                        "eligible"
                        if seasonal_causal and comparison_n > 0
                        else "insufficient_fit_rows"
                        if seasonal_causal
                        else "not_causal"
                    ),
                }
            )
    body = {
        "schema_version": BASELINE_PROFILE_VERSION,
        "selection_partition": "training_fit",
        "selection_rule": "lowest_paired_fit_rmse_tie_persistence",
        "dataset_digest": series.digest,
        "split_manifest_digest_sha256": series.split_manifest_digest_sha256,
        "cells": cells,
    }
    return {**body, "digest": digest(body)}


def apply_baseline_profile(
    series: DatasetSeries,
    rows: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return scoring rows with the selected comparator and provenance."""

    profile_digest = profile.get("digest")
    if not isinstance(profile_digest, str) or not profile_digest:
        raise ValueError("baseline profile digest is missing")
    profile_body = {
        key: value for key, value in profile.items() if key != "digest"
    }
    if profile_digest != digest(profile_body):
        raise ValueError("baseline profile digest does not match its contents")
    if profile.get("schema_version") != BASELINE_PROFILE_VERSION:
        raise ValueError("baseline profile schema version is unsupported")
    if profile.get("dataset_digest") != series.digest:
        raise ValueError("baseline profile dataset digest does not match the series")
    if (
        profile.get("split_manifest_digest_sha256")
        != series.split_manifest_digest_sha256
    ):
        raise ValueError("baseline profile split manifest does not match the series")
    raw_cells = profile.get("cells")
    if not isinstance(raw_cells, (list, tuple)):
        raise ValueError("baseline profile cells are missing")
    selected: dict[tuple[str, int], str] = {}
    for cell in raw_cells:
        if not isinstance(cell, Mapping):
            raise ValueError("baseline profile cell must be a mapping")
        target = cell.get("target")
        horizon = cell.get("horizon_hours")
        baseline_id = cell.get("baseline_id")
        if (
            not isinstance(target, str)
            or isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or baseline_id not in {"persistence", "seasonal_24h"}
        ):
            raise ValueError("baseline profile cell is invalid")
        key = (target, horizon)
        if key in selected:
            raise ValueError("baseline profile contains duplicate cells")
        selected[key] = str(baseline_id)
    targets = tuple(dict.fromkeys(target for target, _horizon in selected))
    horizons = tuple(dict.fromkeys(horizon for _target, horizon in selected))
    expected_profile = fit_baseline_profile(
        series,
        targets=targets,
        horizons=horizons,
    )
    if dict(profile) != expected_profile:
        raise ValueError("baseline profile does not match the canonical fit selection")

    visible_indices: set[int] = set()
    for name in ("training_fit", "training_feedback"):
        partition = series.partitions.get(name)
        if partition is not None:
            visible_indices.update(range(partition.start, partition.end))
    timestamp_index = {
        series.timestamps[index]: index for index in sorted(visible_indices)
    }
    result: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        target = source.get("target")
        horizon = source.get("horizon_hours")
        if not isinstance(target, str) or isinstance(horizon, bool) or not isinstance(
            horizon, int
        ):
            raise ValueError(f"rows[{index}] has invalid baseline task identity")
        requested = selected.get((target, horizon))
        if requested is None:
            raise ValueError(f"rows[{index}] is not covered by the baseline profile")
        model_reference = _finite(source.get("baseline"))
        if model_reference is None:
            raise ValueError(f"rows[{index}] has no finite model reference baseline")
        resolved = model_reference
        resolved_id = "persistence"
        fallback: str | None = None
        if requested == "seasonal_24h":
            target_timestamp = source.get("target_timestamp", source.get("timestamp"))
            origin_timestamp = source.get("origin_timestamp")
            seasonal_index = (
                timestamp_index.get(target_timestamp - 24)
                if isinstance(target_timestamp, (int, float))
                and not isinstance(target_timestamp, bool)
                else None
            )
            causal = bool(
                seasonal_index is not None
                and isinstance(origin_timestamp, (int, float))
                and not isinstance(origin_timestamp, bool)
                and series.timestamps[seasonal_index] <= origin_timestamp
            )
            seasonal = (
                _finite(series.values[target][seasonal_index])
                if causal and seasonal_index is not None
                else None
            )
            if seasonal is None:
                fallback = "seasonal_reference_missing"
            else:
                resolved = seasonal
                resolved_id = "seasonal_24h"
        row = dict(source)
        row.update(
            {
                "model_reference_baseline": model_reference,
                "baseline": resolved,
                "baseline_id": resolved_id,
                "baseline_profile_digest": profile_digest,
            }
        )
        failed = (
            str(source.get("sample_execution_status", source.get("status", "")))
            .strip()
            .casefold()
            == "failed"
            or bool(source.get("scoring_fallback"))
        )
        if failed:
            observed = _finite(source.get("observed"))
            predicted = _finite(source.get("predicted"))
            if (
                observed is not None
                and predicted is not None
                and abs(predicted - observed) < abs(resolved - observed)
            ):
                row["predicted"] = resolved
            row["failed_reward_policy"] = "nonpositive"
        if fallback is not None:
            row["baseline_fallback"] = fallback
        result.append(row)
    return result


__all__ = [
    "BASELINE_PROFILE_VERSION",
    "apply_baseline_profile",
    "fit_baseline_profile",
]
