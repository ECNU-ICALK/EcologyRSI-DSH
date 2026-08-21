"""Version-aware, statistically guarded promotion policy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import random
from typing import Any

from ..core.models import digest
from ..evaluators.objectives import (
    DEFAULT_TARGET_WEIGHTS,
    OBJECTIVE_AGGREGATION_VERSION,
    aggregate_greenhouse_objective,
    skill_score,
)


PROMOTION_POLICY_VERSION = "practical_delta_paired_block_bootstrap@1"
PROMOTION_BLOCK_EVIDENCE_VERSION = "paired_24h_objective_sufficient_statistics@1"
PROMOTION_CONFIDENCE_METHOD = "paired_moving_block_bootstrap@1"
PROMOTION_BLOCK_HOURS = 24
PROMOTION_MAXIMUM_BLOCKS = 128
LEGACY_MINIMUM_SCORE_DELTA = 1e-12
V2_MINIMUM_SCORE_DELTA = 0.005
PROMOTION_BOOTSTRAP_RESAMPLES = 1_000
PROMOTION_CONFIDENCE_LEVEL = 0.95
PROMOTION_MINIMUM_PAIRED_BLOCKS = 4
_COMMON_CONTRACT_FIELDS = (
    "objective_aggregation_version",
    "baseline_profile_digest",
    "evaluation_index_digest",
    "dataset_digest",
    "split_manifest_digest_sha256",
)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validated_grid(
    horizons: Sequence[int], target_weights: Mapping[str, float]
) -> tuple[tuple[int, ...], dict[str, float]]:
    resolved_horizons = tuple(horizons)
    if (
        not resolved_horizons
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in resolved_horizons
        )
        or len(set(resolved_horizons)) != len(resolved_horizons)
    ):
        raise ValueError("promotion horizons must be unique positive integers")
    weights = {
        str(name): _finite(value, f"target weight {name}")
        for name, value in target_weights.items()
    }
    if not weights or any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("promotion target weights must be non-negative and non-zero")
    if set(weights) != set(DEFAULT_TARGET_WEIGHTS) or any(
        not math.isclose(
            weights[target],
            DEFAULT_TARGET_WEIGHTS[target],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for target in DEFAULT_TARGET_WEIGHTS
    ):
        raise ValueError("promotion target weights do not match the frozen objective")
    return resolved_horizons, weights


def build_promotion_block_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizons: Sequence[int],
    target_weights: Mapping[str, float],
    dataset_digest: str,
    split_manifest_digest_sha256: str,
) -> dict[str, Any]:
    """Build bounded 24-hour origin blocks of objective sufficient statistics."""

    resolved_horizons, weights = _validated_grid(horizons, target_weights)
    cell_keys = tuple(
        (target, horizon) for target in weights for horizon in resolved_horizons
    )
    groups: dict[int, dict[tuple[str, int], dict[str, float | int]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "eligible": 0,
                "succeeded": 0,
                "candidate_squared_error_sum": 0.0,
                "baseline_squared_error_sum": 0.0,
                "normalized_reward_sum": 0.0,
            }
        )
    )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"rows[{index}] must be a mapping")
        target = str(row.get("target"))
        horizon = row.get("horizon_hours")
        origin = row.get("origin_timestamp")
        if target not in weights or horizon not in resolved_horizons:
            raise ValueError(f"rows[{index}] has an unknown prediction task")
        if (
            isinstance(origin, bool)
            or not isinstance(origin, (int, float))
            or not math.isfinite(float(origin))
        ):
            raise ValueError(f"rows[{index}] has no numeric origin timestamp")
        cell = groups[math.floor(float(origin) / PROMOTION_BLOCK_HOURS)][
            (target, int(horizon))
        ]
        cell["eligible"] = int(cell["eligible"]) + 1
        if (
            str(row.get("sample_execution_status", "succeeded")).strip().casefold()
            != "succeeded"
            or bool(row.get("scoring_fallback"))
        ):
            continue
        observed = _finite(row.get("observed"), f"rows[{index}].observed")
        predicted = _finite(row.get("predicted"), f"rows[{index}].predicted")
        baseline = _finite(row.get("baseline"), f"rows[{index}].baseline")
        scale = _finite(row.get("normalization_scale"), f"rows[{index}].normalization_scale")
        if scale <= 0:
            raise ValueError(f"rows[{index}].normalization_scale must be positive")
        candidate_error = (predicted - observed) / scale
        baseline_error = (baseline - observed) / scale
        cell["succeeded"] = int(cell["succeeded"]) + 1
        cell["candidate_squared_error_sum"] = float(
            cell["candidate_squared_error_sum"]
        ) + candidate_error**2
        cell["baseline_squared_error_sum"] = float(
            cell["baseline_squared_error_sum"]
        ) + baseline_error**2
        cell["normalized_reward_sum"] = float(cell["normalized_reward_sum"]) + max(
            -1.0, min(1.0, abs(baseline_error) - abs(candidate_error))
        )

    blocks: list[dict[str, Any]] = []
    for block_index, block_cells in sorted(groups.items()):
        cells = []
        for target, horizon in cell_keys:
            stats = block_cells[(target, horizon)]
            cells.append(
                {
                    "target": target,
                    "horizon_hours": horizon,
                    "eligible": int(stats["eligible"]),
                    "succeeded": int(stats["succeeded"]),
                    "candidate_squared_error_sum": float(stats["candidate_squared_error_sum"]),
                    "baseline_squared_error_sum": float(stats["baseline_squared_error_sum"]),
                    "normalized_reward_sum": float(stats["normalized_reward_sum"]),
                }
            )
        blocks.append(
            {
                "block_id": digest(
                    {
                        "dataset_digest": dataset_digest,
                        "split_manifest_digest_sha256": split_manifest_digest_sha256,
                        "block_hours": PROMOTION_BLOCK_HOURS,
                        "origin_block_index": block_index,
                    }
                ),
                "origin_block_index": block_index,
                "cells": cells,
            }
        )
    body = {
        "schema_version": PROMOTION_BLOCK_EVIDENCE_VERSION,
        "block_hours": PROMOTION_BLOCK_HOURS,
        "maximum_blocks": PROMOTION_MAXIMUM_BLOCKS,
        "objective_aggregation_version": OBJECTIVE_AGGREGATION_VERSION,
        "score_definition": "coverage_penalized_weighted_rmse_skill@2",
        "target_weights": weights,
        "horizons": list(resolved_horizons),
        "block_count": len(blocks),
        "blocks": blocks,
    }
    return {**body, "evidence_digest": digest(body)}


def _contract_value(evaluation: Any, name: str) -> Any:
    metrics = getattr(evaluation, "metrics", None)
    return metrics.get(name) if isinstance(metrics, Mapping) else None


def _sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _common_contract_matches(
    evaluation: Any,
    incumbent: Any,
    *,
    require_complete: bool = False,
) -> bool:
    current_digest = getattr(evaluation, "evaluator_digest", None)
    matches = bool(
        isinstance(current_digest, str)
        and current_digest
        and current_digest == getattr(incumbent, "evaluator_digest", None)
        and all(
            _contract_value(evaluation, name) == _contract_value(incumbent, name)
            for name in _COMMON_CONTRACT_FIELDS
        )
    )
    if not matches or not require_complete:
        return matches
    return _sha256(current_digest) and all(
        _sha256(_contract_value(evaluation, name))
        for name in (
            "baseline_profile_digest",
            "evaluation_index_digest",
            "dataset_digest",
            "split_manifest_digest_sha256",
        )
    )


def _validated_evidence(evaluation: Any) -> dict[str, Any] | None:
    raw = _contract_value(evaluation, "promotion_block_evidence")
    if not isinstance(raw, Mapping):
        return None
    body = {key: value for key, value in raw.items() if key != "evidence_digest"}
    if (
        raw.get("schema_version") != PROMOTION_BLOCK_EVIDENCE_VERSION
        or raw.get("block_hours") != PROMOTION_BLOCK_HOURS
        or raw.get("maximum_blocks") != PROMOTION_MAXIMUM_BLOCKS
        or raw.get("objective_aggregation_version") != OBJECTIVE_AGGREGATION_VERSION
        or raw.get("score_definition") != "coverage_penalized_weighted_rmse_skill@2"
        or raw.get("evidence_digest") != digest(body)
    ):
        return None
    try:
        horizons, weights = _validated_grid(
            raw.get("horizons", ()), raw.get("target_weights", {})
        )
    except (TypeError, ValueError):
        return None
    expected_cells = {(target, horizon) for target in weights for horizon in horizons}
    raw_blocks = raw.get("blocks")
    if (
        not isinstance(raw_blocks, (list, tuple))
        or not raw_blocks
        or raw.get("block_count") != len(raw_blocks)
    ):
        return None
    blocks: dict[str, dict[tuple[str, int], dict[str, float | int]]] = {}
    for block in raw_blocks:
        if not isinstance(block, Mapping):
            return None
        block_id = block.get("block_id")
        cells = block.get("cells")
        if not _sha256(block_id) or block_id in blocks:
            return None
        if not isinstance(cells, (list, tuple)):
            return None
        parsed: dict[tuple[str, int], dict[str, float | int]] = {}
        for cell in cells:
            if not isinstance(cell, Mapping):
                return None
            key = (str(cell.get("target")), cell.get("horizon_hours"))
            eligible = cell.get("eligible")
            succeeded = cell.get("succeeded")
            if (
                key not in expected_cells
                or key in parsed
                or isinstance(eligible, bool)
                or not isinstance(eligible, int)
                or isinstance(succeeded, bool)
                or not isinstance(succeeded, int)
                or eligible < 0
                or succeeded < 0
                or succeeded > eligible
            ):
                return None
            try:
                candidate_sq = _finite(cell.get("candidate_squared_error_sum"), "candidate squared error")
                baseline_sq = _finite(cell.get("baseline_squared_error_sum"), "baseline squared error")
                reward_sum = _finite(cell.get("normalized_reward_sum"), "normalized reward sum")
            except (TypeError, ValueError):
                return None
            if candidate_sq < 0 or baseline_sq < 0:
                return None
            parsed[key] = {
                "eligible": eligible,
                "succeeded": succeeded,
                "candidate_squared_error_sum": candidate_sq,
                "baseline_squared_error_sum": baseline_sq,
                "normalized_reward_sum": reward_sum,
            }
        if set(parsed) != expected_cells:
            return None
        blocks[block_id] = parsed
    config = {
        "schema_version": raw["schema_version"],
        "block_hours": raw["block_hours"],
        "maximum_blocks": raw["maximum_blocks"],
        "objective_aggregation_version": raw["objective_aggregation_version"],
        "score_definition": raw["score_definition"],
        "target_weights": weights,
        "horizons": list(horizons),
    }
    return {
        "config_digest": digest(config),
        "horizons": horizons,
        "weights": weights,
        "blocks": blocks,
    }


def _evidence_matches_evaluation(
    evidence: Mapping[str, Any], evaluation: Any
) -> bool:
    raw_weights = _contract_value(evaluation, "objective_target_weights")
    raw_horizons = _contract_value(evaluation, "objective_horizons")
    if not isinstance(raw_weights, Mapping) or not isinstance(
        raw_horizons, (list, tuple)
    ):
        return False
    try:
        horizons, weights = _validated_grid(raw_horizons, raw_weights)
    except (TypeError, ValueError):
        return False
    return bool(
        tuple(evidence["horizons"]) == horizons
        and evidence["weights"] == weights
    )


def _resampled_objective(
    evidence: Mapping[str, Any], sampled_ids: Sequence[str]
) -> float:
    blocks = evidence["blocks"]
    horizons = evidence["horizons"]
    weights = evidence["weights"]
    task_results = []
    for target in weights:
        for horizon in horizons:
            key = (target, horizon)
            selected = [blocks[block_id][key] for block_id in sampled_ids]
            eligible = sum(int(cell["eligible"]) for cell in selected)
            succeeded = sum(int(cell["succeeded"]) for cell in selected)
            if succeeded:
                candidate_nrmse = math.sqrt(
                    sum(float(cell["candidate_squared_error_sum"]) for cell in selected)
                    / succeeded
                )
                baseline_nrmse = math.sqrt(
                    sum(float(cell["baseline_squared_error_sum"]) for cell in selected)
                    / succeeded
                )
                task_skill = skill_score(candidate_nrmse, baseline_nrmse)
                normalized_reward = sum(
                    float(cell["normalized_reward_sum"]) for cell in selected
                ) / succeeded
            else:
                task_skill = -1.0
                normalized_reward = -1.0
            task_results.append(
                {
                    "target": target,
                    "horizon_hours": horizon,
                    "n": succeeded,
                    "eligible_rows": eligible,
                    "skill_score": task_skill,
                    "normalized_mean_reward": normalized_reward,
                    "objective_quality": succeeded / eligible if eligible else 0.0,
                }
            )
    return float(
        aggregate_greenhouse_objective(
            task_results, horizons, target_weights=weights
        )["weighted_skill_score"]
    )


def _paired_bootstrap_interval(
    current: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    block_ids: Sequence[str],
    *,
    seed_material: Any,
) -> tuple[float, float]:
    randomizer = random.Random(int(digest(seed_material)[:16], 16))
    count = len(block_ids)
    deltas = []
    for _ in range(PROMOTION_BOOTSTRAP_RESAMPLES):
        sampled = [block_ids[randomizer.randrange(count)] for _ in range(count)]
        deltas.append(
            _resampled_objective(current, sampled)
            - _resampled_objective(incumbent, sampled)
        )
    deltas.sort()
    return (
        deltas[int(0.025 * (len(deltas) - 1))],
        deltas[int(0.975 * (len(deltas) - 1))],
    )


def _incomparable(score_delta: float, reason_code: str) -> dict[str, Any]:
    return {
        "policy_version": PROMOTION_POLICY_VERSION,
        "comparable": False,
        "score_delta": score_delta,
        "minimum_score_delta": V2_MINIMUM_SCORE_DELTA,
        "paired_block_count": 0,
        "bootstrap_resamples": 0,
        "confidence_level": PROMOTION_CONFIDENCE_LEVEL,
        "confidence_method": PROMOTION_CONFIDENCE_METHOD,
        "confidence_status": "not_evaluated",
        "confidence_interval_95": None,
        "improved": False,
        "reason_code": reason_code,
    }


def _assess_promotion_improvement_legacy(
    evaluation: Any, incumbent_evaluation: Any
) -> dict[str, Any]:
    """Assess practical and paired-block statistical improvement."""

    score_delta = _finite(getattr(evaluation, "score", None), "evaluation score") - _finite(
        getattr(incumbent_evaluation, "score", None), "incumbent score"
    )
    current_version = _contract_value(evaluation, "objective_aggregation_version")
    incumbent_version = _contract_value(incumbent_evaluation, "objective_aggregation_version")
    current_v2 = current_version == OBJECTIVE_AGGREGATION_VERSION
    incumbent_v2 = incumbent_version == OBJECTIVE_AGGREGATION_VERSION
    if not current_v2 and not incumbent_v2:
        if not _common_contract_matches(evaluation, incumbent_evaluation):
            return _incomparable(score_delta, "incompatible_scoring_contract")
        improved = score_delta > LEGACY_MINIMUM_SCORE_DELTA
        return {
            "policy_version": "strict_score_delta@1",
            "comparable": True,
            "score_delta": score_delta,
            "minimum_score_delta": LEGACY_MINIMUM_SCORE_DELTA,
            "paired_block_count": 0,
            "confidence_interval_95": None,
            "improved": improved,
            "reason_code": "improved" if improved else "below_legacy_delta",
        }
    if not current_v2 or not incumbent_v2 or not _common_contract_matches(
        evaluation, incumbent_evaluation, require_complete=True
    ):
        return _incomparable(score_delta, "incompatible_scoring_contract")
    current = _validated_evidence(evaluation)
    incumbent = _validated_evidence(incumbent_evaluation)
    if current is None or incumbent is None:
        return _incomparable(score_delta, "invalid_block_evidence")
    if not _evidence_matches_evaluation(
        current, evaluation
    ) or not _evidence_matches_evaluation(incumbent, incumbent_evaluation):
        return _incomparable(score_delta, "incompatible_block_configuration")
    if current["config_digest"] != incumbent["config_digest"]:
        return _incomparable(score_delta, "incompatible_block_configuration")
    block_ids = sorted(current["blocks"])
    if set(block_ids) != set(incumbent["blocks"]):
        return _incomparable(score_delta, "mismatched_block_identities")

    point_pass = score_delta > V2_MINIMUM_SCORE_DELTA
    interval: tuple[float, float] | None = None
    confidence_pass = True
    confidence_status = "insufficient_blocks"
    if len(block_ids) >= PROMOTION_MINIMUM_PAIRED_BLOCKS:
        try:
            interval = _paired_bootstrap_interval(
                current,
                incumbent,
                block_ids,
                seed_material={
                    "evaluation_index_digest": _contract_value(
                        evaluation, "evaluation_index_digest"
                    ),
                    "candidate_evaluation": digest(
                        {
                            "evaluation_id": getattr(evaluation, "evaluation_id", None),
                            "candidate_id": getattr(evaluation, "candidate_id", None),
                            "score": getattr(evaluation, "score", None),
                            "evaluator_digest": getattr(evaluation, "evaluator_digest", None),
                            "evidence_digest": _contract_value(
                                evaluation, "promotion_block_evidence"
                            ).get("evidence_digest"),
                        }
                    ),
                    "incumbent_evaluation": digest(
                        {
                            "evaluation_id": getattr(
                                incumbent_evaluation, "evaluation_id", None
                            ),
                            "candidate_id": getattr(
                                incumbent_evaluation, "candidate_id", None
                            ),
                            "score": getattr(incumbent_evaluation, "score", None),
                            "evaluator_digest": getattr(
                                incumbent_evaluation, "evaluator_digest", None
                            ),
                            "evidence_digest": _contract_value(
                                incumbent_evaluation, "promotion_block_evidence"
                            ).get("evidence_digest"),
                        }
                    ),
                    "block_ids": block_ids,
                },
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return _incomparable(score_delta, "invalid_block_evidence")
        confidence_pass = interval[0] > 0.0
        confidence_status = "passed" if confidence_pass else "crosses_zero"
    improved = point_pass and confidence_pass
    reason_code = (
        "below_practical_delta"
        if not point_pass
        else "confidence_interval_crosses_zero"
        if not confidence_pass
        else "improved"
    )
    return {
        "policy_version": PROMOTION_POLICY_VERSION,
        "comparable": True,
        "score_delta": score_delta,
        "minimum_score_delta": V2_MINIMUM_SCORE_DELTA,
        "paired_block_count": len(block_ids),
        "bootstrap_resamples": PROMOTION_BOOTSTRAP_RESAMPLES if interval else 0,
        "confidence_level": PROMOTION_CONFIDENCE_LEVEL,
        "confidence_method": PROMOTION_CONFIDENCE_METHOD,
        "confidence_status": confidence_status,
        "confidence_interval_95": list(interval) if interval else None,
        "improved": improved,
        "reason_code": reason_code,
    }


def assess_promotion_improvement(
    evaluation: Any,
    incumbent_evaluation: Any,
    *,
    execution_protocol: str | None = None,
) -> dict[str, Any]:
    """Compatibility projection; adaptive DSH selection is never confirmatory."""

    result = _assess_promotion_improvement_legacy(
        evaluation, incumbent_evaluation
    )
    if execution_protocol != "dsh_native_plugin_evolution@1":
        return result
    projected = {
        **result,
        "evidence_class": "exploratory_adaptive_data",
        "selection_only": True,
        "validated": False,
        "confirmed": False,
        "formal_stage": None,
    }
    if (
        projected.get("comparable")
        and int(projected.get("paired_block_count") or 0) < 8
    ):
        projected.update(
            {
                "improved": False,
                "reason_code": "insufficient_evidence",
                "confidence_status": "display_only_insufficient_evidence",
            }
        )
    elif "confidence_status" in projected:
        projected["confidence_status"] = (
            "display_only_" + str(projected["confidence_status"])
        )
    return projected


__all__ = [
    "LEGACY_MINIMUM_SCORE_DELTA",
    "PROMOTION_BLOCK_EVIDENCE_VERSION",
    "PROMOTION_BLOCK_HOURS",
    "PROMOTION_BOOTSTRAP_RESAMPLES",
    "PROMOTION_CONFIDENCE_LEVEL",
    "PROMOTION_CONFIDENCE_METHOD",
    "PROMOTION_MAXIMUM_BLOCKS",
    "PROMOTION_POLICY_VERSION",
    "V2_MINIMUM_SCORE_DELTA",
    "assess_promotion_improvement",
    "build_promotion_block_evidence",
]
