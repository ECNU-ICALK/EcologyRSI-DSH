"""Pure, versioned reward and objective calculations for greenhouse scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from statistics import fmean
from typing import Any


OBJECTIVE_COMPONENT_BOUND = 1.0
OBJECTIVE_MISSING_PENALTY = -1.0
OBJECTIVE_AGGREGATION_VERSION = "weighted_task_skill_reward@2"
DEFAULT_TARGET_WEIGHTS = {
    "air_temperature": 1 / 3,
    "relative_humidity": 1 / 3,
    "co2_concentration": 1 / 3,
}


def clip_normalized_objective(
    value: float,
    *,
    bound: float = OBJECTIVE_COMPONENT_BOUND,
) -> float:
    """Bound one finite normalized objective component."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("normalized objective must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError("normalized objective must be finite")
    if (
        isinstance(bound, bool)
        or not isinstance(bound, (int, float))
        or not math.isfinite(float(bound))
        or float(bound) <= 0
    ):
        raise ValueError("normalized objective bound must be positive")
    return max(-float(bound), min(float(bound), float(value)))


def skill_score(candidate_nrmse: float, baseline_nrmse: float) -> float:
    """Return bounded relative RMSE improvement over a baseline."""

    for name, value in (
        ("candidate_nrmse", candidate_nrmse),
        ("baseline_nrmse", baseline_nrmse),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")
    candidate = float(candidate_nrmse)
    baseline = float(baseline_nrmse)
    if baseline > 1e-12:
        value = 1.0 - candidate / baseline
    else:
        value = 0.0 if candidate <= 1e-12 else -1.0
    return clip_normalized_objective(value)


def normalized_absolute_error_reward(
    baseline_errors: Sequence[float],
    candidate_errors: Sequence[float],
    scale: float,
) -> tuple[float, float]:
    """Return raw and per-sample-bounded mean absolute-error improvement."""

    if len(baseline_errors) != len(candidate_errors) or not baseline_errors:
        raise ValueError("baseline and candidate errors must be non-empty and aligned")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0
    ):
        raise ValueError("normalization scale must be positive and finite")
    values: list[float] = []
    for index, (baseline, candidate) in enumerate(
        zip(baseline_errors, candidate_errors)
    ):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (baseline, candidate)
        ):
            raise ValueError(f"error pair {index} must be finite")
        values.append(
            (abs(float(baseline)) - abs(float(candidate))) / float(scale)
        )
    return fmean(values), fmean(clip_normalized_objective(value) for value in values)


def _validated_target_weights(
    target_weights: Mapping[str, float],
) -> dict[str, float]:
    expected = set(DEFAULT_TARGET_WEIGHTS)
    if set(target_weights) != expected:
        missing = expected - set(target_weights)
        unknown = set(target_weights) - expected
        detail = []
        if missing:
            detail.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            detail.append("unknown " + ", ".join(sorted(unknown)))
        raise ValueError("objective weights are invalid: " + "; ".join(detail))
    result: dict[str, float] = {}
    for target, value in target_weights.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"objective weights are invalid for {target}")
        result[target] = float(value)
    if sum(result.values()) <= 0:
        raise ValueError("objective weights must sum to a positive value")
    return result


def _quality(row: Mapping[str, Any]) -> float:
    raw = row.get("objective_quality")
    if (
        isinstance(raw, (int, float))
        and not isinstance(raw, bool)
        and math.isfinite(float(raw))
    ):
        return max(0.0, min(1.0, float(raw)))
    n = row.get("n")
    eligible = row.get("eligible_rows")
    if (
        isinstance(n, int)
        and not isinstance(n, bool)
        and n >= 0
        and isinstance(eligible, int)
        and not isinstance(eligible, bool)
        and eligible > 0
    ):
        return max(0.0, min(1.0, n / eligible))
    return 1.0


def aggregate_greenhouse_objective(
    task_results: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    *,
    target_weights: Mapping[str, float],
    missing_skill_penalty: float = OBJECTIVE_MISSING_PENALTY,
    missing_reward_penalty: float = OBJECTIVE_MISSING_PENALTY,
) -> dict[str, Any]:
    """Aggregate a complete target/horizon grid with explicit stable weights."""

    if not horizons:
        raise ValueError("horizons must not be empty")
    if any(
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 1
        for horizon in horizons
    ):
        raise ValueError("horizons must contain positive integers")
    resolved_horizons = tuple(horizons)
    if len(set(resolved_horizons)) != len(resolved_horizons):
        raise ValueError("horizons contain duplicate values")
    weights = _validated_target_weights(target_weights)
    skill_penalty = clip_normalized_objective(missing_skill_penalty)
    reward_penalty = clip_normalized_objective(missing_reward_penalty)

    rows_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for index, row in enumerate(task_results):
        if not isinstance(row, Mapping):
            raise TypeError(f"task_results[{index}] must be a mapping")
        target = row.get("target")
        horizon = row.get("horizon_hours")
        if target not in weights:
            raise ValueError(f"task_results[{index}] has unknown target {target!r}")
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon not in resolved_horizons
        ):
            raise ValueError(f"task_results[{index}] has unknown horizon")
        key = (str(target), horizon)
        if key in rows_by_key:
            raise ValueError(f"task_results contains duplicate cell {key!r}")
        rows_by_key[key] = row

    horizon_weight = 1.0 / len(resolved_horizons)
    total_weight = 0.0
    weighted_skill = 0.0
    weighted_reward = 0.0
    observed_weight = 0.0
    effective_weight = 0.0
    missing_count = 0
    quality_sum = 0.0

    for target, target_weight in weights.items():
        for horizon in resolved_horizons:
            weight = target_weight * horizon_weight
            total_weight += weight
            row = rows_by_key.get((target, horizon))
            n = row.get("n") if row is not None else 0
            usable = bool(
                row is not None
                and isinstance(n, int)
                and not isinstance(n, bool)
                and n > 0
                and isinstance(row.get("skill_score"), (int, float))
                and not isinstance(row.get("skill_score"), bool)
                and math.isfinite(float(row["skill_score"]))
                and isinstance(row.get("normalized_mean_reward"), (int, float))
                and not isinstance(row.get("normalized_mean_reward"), bool)
                and math.isfinite(float(row["normalized_mean_reward"]))
            )
            if usable and row is not None:
                quality = _quality(row)
                skill = clip_normalized_objective(float(row["skill_score"]))
                reward = clip_normalized_objective(
                    float(row["normalized_mean_reward"])
                )
                observed_weight += weight
                effective_weight += weight * quality
                quality_sum += weight * quality
                skill = quality * skill + (1.0 - quality) * skill_penalty
                reward = quality * reward + (1.0 - quality) * reward_penalty
            else:
                skill = skill_penalty
                reward = reward_penalty
                missing_count += 1
            weighted_skill += weight * skill
            weighted_reward += weight * reward

    if total_weight <= 0:
        raise ValueError("objective weights must sum to a positive value")
    return {
        "weighted_skill_score": weighted_skill / total_weight,
        "weighted_normalized_mean_reward": weighted_reward / total_weight,
        "objective_aggregation_version": OBJECTIVE_AGGREGATION_VERSION,
        "objective_task_count": len(weights) * len(resolved_horizons),
        "objective_missing_task_count": missing_count,
        "objective_weight_coverage": observed_weight / total_weight,
        "objective_effective_weight_coverage": effective_weight / total_weight,
        "objective_quality": quality_sum / total_weight,
        "objective_horizon_weighting": "equal",
    }


__all__ = [
    "DEFAULT_TARGET_WEIGHTS",
    "OBJECTIVE_AGGREGATION_VERSION",
    "OBJECTIVE_COMPONENT_BOUND",
    "OBJECTIVE_MISSING_PENALTY",
    "aggregate_greenhouse_objective",
    "clip_normalized_objective",
    "normalized_absolute_error_reward",
    "skill_score",
]
