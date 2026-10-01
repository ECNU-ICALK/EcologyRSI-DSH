"""Score a target/horizon cell without treating failed calls as predictions."""
from __future__ import annotations

import math
from statistics import fmean

from .objectives import normalized_absolute_error_reward, skill_score
from .metrics import _mae, _rmse


def successful_rows(rows):
    return [row for row in rows
            if str(row.get("sample_execution_status", "succeeded")).strip().casefold() != "failed"
            and not row.get("scoring_fallback")]


def score_cell(rows, *, target, horizon, bounds, scale, eligible_rows,
               partition_eligible_rows, available_rows, execution, minimum_coverage,
               raw_diagnostics=False):
    """Return public cell metrics and the rows admitted to scientific scoring.

    Rolling forecasts count physical violations over their whole returned vector.
    Ridge forecasts additionally report raw/clipped diagnostics, and score physical
    violations only on successful predictions. Neither admits fallback penalties
    into errors, rewards, or the model evidence count.
    """
    admitted = successful_rows(rows)
    coverage = int(execution["succeeded_examples"]) / eligible_rows if eligible_rows else 0.0
    result = {
        "target": target, "unit": bounds["unit"], "horizon_hours": horizon,
        "n": len(admitted), "eligible_rows": eligible_rows,
        "partition_eligible_rows": partition_eligible_rows,
        "available_rows": available_rows,
        "deferred_rows": max(0, available_rows - eligible_rows),
        "missing_or_nonfinite_rows": max(0, eligible_rows - len(admitted)),
        "failed_rows": len(rows) - len(admitted), "normalization_scale": scale,
        "mae": None, "bias": None, "rmse": None,
        "baseline_mae": None, "baseline_rmse": None,
        "normalized_rmse": None, "baseline_normalized_rmse": None,
        "skill_score": -1.0, "mean_reward": None, "normalized_mean_reward": None,
        "constraint_violations": 0,
        "sample_execution_attempted": int(execution["attempted_examples"]),
        "sample_execution_failed": int(execution["failed_examples"]),
        "sample_execution_coverage": coverage, "sample_execution_coverage_pass": False,
        "objective_quality": 0.0,
    }
    if raw_diagnostics:
        raw_values = (row.get("raw_predicted") for row in rows)
        result.update(
            raw_out_of_range_predictions=sum(
                not bounds["minimum"] <= value <= bounds["maximum"]
                for value in raw_values if isinstance(value, (int, float))
                and not isinstance(value, bool) and math.isfinite(value)
            ),
            prediction_clipped_count=sum(row.get("prediction_clipped") is True for row in rows),
            negative_reward_fraction=None,
        )
    else:
        result["raw_normalized_mean_reward"] = None
    if not admitted:
        return result, admitted

    errors = [float(row["predicted"]) - float(row["observed"]) for row in admitted]
    baseline_errors = [float(row["baseline"]) - float(row["observed"]) for row in admitted]
    rmse, baseline_rmse = _rmse(errors), _rmse(baseline_errors)
    nrmse, baseline_nrmse = rmse / scale, baseline_rmse / scale
    rewards = [abs(baseline) - abs(error) for baseline, error in zip(baseline_errors, errors)]
    raw_reward, reward = normalized_absolute_error_reward(baseline_errors, errors, scale)
    zero_error = nrmse <= 1e-12 if raw_diagnostics else rmse <= 1e-12
    result.update(
        mae=_mae(errors), bias=fmean(errors), rmse=rmse,
        baseline_mae=_mae(baseline_errors), baseline_rmse=baseline_rmse,
        normalized_rmse=nrmse, baseline_normalized_rmse=baseline_nrmse,
        skill_score=skill_score(nrmse, baseline_nrmse),
        raw_skill_score=1.0 - nrmse / baseline_nrmse if baseline_nrmse > 1e-12 else (0.0 if zero_error else -1.0),
        mean_reward=fmean(rewards), normalized_mean_reward=reward,
        raw_normalized_mean_reward=raw_reward,
        negative_reward_fraction=sum(value < 0.0 for value in rewards) / len(rewards),
        constraint_violations=sum(
            not bounds["minimum"] <= float(row["predicted"]) <= bounds["maximum"]
            for row in (admitted if raw_diagnostics else rows)
        ),
        sample_execution_coverage_pass=eligible_rows > 0 and coverage >= minimum_coverage,
        objective_quality=coverage,
    )
    return result, admitted


def execution_for_cell(summary, target, horizon):
    return next((item for item in summary["tasks"]
                 if item["target"] == target and item["horizon_hours"] == horizon),
                {"attempted_examples": 0, "succeeded_examples": 0, "failed_examples": 0})
