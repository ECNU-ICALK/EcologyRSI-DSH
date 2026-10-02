"""Produce independent point-certification statistics from scored day blocks.

No labels are reread and no intervals are invented. Predictive-interval claims
remain inconclusive until real frozen UQ evidence is available.
"""
from dataclasses import replace
import math
from typing import Any

from ..core.agent_prediction import successful_agent_provenance_passes
from ..core.immutable import thaw_json
from ..core.models import digest
from ..evolution.promotion import _evidence_matches_evaluation, _resampled_objective, _validated_evidence
from .fitness import FitnessProfile, _legal_starts, build_formal_fitness_assessment, build_moving_block_resample_indices
from .objectives import skill_score


def certification_policy(profile: FitnessProfile) -> dict[str, Any]:
    """Freeze the complete rule before the formal partition is opened."""
    body = {
        "schema_version": "ecologyrsi-dsh.independent-certification-policy/1",
        "claim_scope": "point_and_interval" if profile.require_predictive_intervals else "point_prediction",
        "fitness_profile_digest": profile.profile_digest,
        "minimum_paired_blocks": 14, "minimum_contiguous_starts": 10,
        "minimum_cell_samples": 80, "minimum_coverage": max(.95, profile.selection_minimum_coverage),
        "minimum_score_delta": profile.selection_minimum_score_delta,
        "moving_block_days": 3, "bootstrap_resamples": 1000, "confidence_level": .95,
        "comparison_reference": "training_fit_selected_frozen_baseline",
        "both_independent_inference_replicas_must_pass": True,
        "replicas_increase_independent_time_blocks": False,
    }
    return {**body, "policy_digest": digest(body)}


def produce_formal_point_metrics(evaluation: Any, *, policy: dict[str, Any],
                                 baseline_profile_digest: str) -> dict[str, Any]:
    """Reconstruct the point score and its paired moving-day-block interval."""
    metrics = thaw_json(evaluation.metrics)
    evidence = _validated_evidence(evaluation)
    if evidence is None or not _evidence_matches_evaluation(evidence, evaluation):
        raise ValueError("formal_scoring_evidence_invalid")
    if not baseline_profile_digest or metrics.get("baseline_profile_digest") != baseline_profile_digest:
        raise ValueError("formal_frozen_baseline_binding_mismatch")
    raw = metrics["promotion_block_evidence"]
    indices = [block.get("origin_block_index") for block in raw["blocks"]]
    if (any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("formal_day_block_identities_invalid")
    block_ids = {block["origin_block_index"]: block["block_id"] for block in raw["blocks"]}
    indices = tuple(sorted(indices))
    point = _resampled_objective(evidence, [block_ids[index] for index in indices])
    if not math.isclose(point, evaluation.score, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("formal_score_does_not_match_scored_blocks")
    starts = _legal_starts(indices, policy["moving_block_days"])
    draws = build_moving_block_resample_indices(
        indices, resamples=policy["bootstrap_resamples"],
        moving_block_days=policy["moving_block_days"],
        seed_material={"policy_digest": policy["policy_digest"],
                       "evidence_digest": raw["evidence_digest"],
                       "baseline_profile_digest": baseline_profile_digest},
    )
    scores = sorted(_resampled_objective(evidence, [block_ids[index] for index in draw]) for draw in draws)
    tail = (1 - policy["confidence_level"]) / 2
    lower = scores[int(tail * (len(scores) - 1))] if scores else None
    upper = scores[int((1 - tail) * (len(scores) - 1))] if scores else None
    # Cross-check every scored cell against the sufficient statistics. A larger
    # fixture count, an empty placeholder day, or a duplicate row buys no power.
    expected = {(target, horizon) for target in evidence["weights"] for horizon in evidence["horizons"]}
    rows = metrics.get("targets", ())
    keys = [(row.get("target"), row.get("horizon_hours")) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("formal_objective_grid_mismatch")
    for row, key in zip(rows, keys):
        cells = [evidence["blocks"][block_id][key] for block_id in block_ids.values()]
        succeeded = sum(cell["succeeded"] for cell in cells)
        eligible = sum(cell["eligible"] for cell in cells)
        if type(row.get("n")) is not int or row["n"] != succeeded:
            raise ValueError("formal_cell_count_mismatch")
        if succeeded:
            candidate_rmse = math.sqrt(sum(cell["candidate_squared_error_sum"] for cell in cells) / succeeded)
            baseline_rmse = math.sqrt(sum(cell["baseline_squared_error_sum"] for cell in cells) / succeeded)
            expected_skill = skill_score(candidate_rmse, baseline_rmse)
            if type(row.get("skill_score")) not in (int, float) or not math.isclose(row["skill_score"], expected_skill, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("formal_cell_skill_mismatch")
        row["sample_execution_coverage"] = succeeded / eligible if eligible else 0.
        row["paired_block_count"] = sum(cell["succeeded"] > 0 for cell in cells)
    metrics.update({"formal_score": point, "formal_score_lcb": lower,
                    "formal_score_ucb": upper,
                    "formal_valid_three_day_start_count": len(starts),
                    "formal_paired_block_count": len(indices),
                    "formal_point_evidence": {
                        "source_evidence_digest": raw["evidence_digest"],
                        "frozen_baseline_profile_digest": baseline_profile_digest,
                        "resample_family_digest": digest(draws),
                        "policy_digest": policy["policy_digest"],
                    }})
    return metrics


def assess_independent_replica(evaluation: Any, profile: FitnessProfile, *,
                               policy: dict[str, Any], baseline_profile_digest: str,
                               frozen_baseline_uq_artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    """Keep current scientific guards and add the frozen formal evidence gate."""
    if policy != certification_policy(profile):
        raise ValueError("independent certification policy changed after freezing")
    try:
        metrics = produce_formal_point_metrics(evaluation, policy=policy,
                                               baseline_profile_digest=baseline_profile_digest)
    except (TypeError, ValueError, KeyError) as exc:
        return {"outcome": "inconclusive", "formal_confirmation": False,
                "certification_scope": policy["claim_scope"], "failures": [str(exc)],
                "policy_digest": policy["policy_digest"]}
    assessment = build_formal_fitness_assessment(
        replace(evaluation, metrics=metrics), frozen_baseline_uq_artifact or {}, profile,
        claim_scope=policy["claim_scope"],
    ).to_dict()
    failures = list(assessment["failures"])
    if evaluation.passed is not True or metrics.get("scientific_pass") is not True:
        failures.append("existing_scientific_gate_failed")
    if not successful_agent_provenance_passes(metrics.get("sample_execution")):
        failures.append("strict_agent_chain_failed")
    if metrics["formal_score"] <= policy["minimum_score_delta"]:
        failures.append("below_practical_score_delta")
    coverage = metrics.get("sample_execution_coverage")
    if type(coverage) not in (int, float) or not math.isfinite(coverage) or coverage < policy["minimum_coverage"]:
        failures.append("formal_execution_coverage_insufficient")
    sufficient = (metrics["formal_paired_block_count"] >= policy["minimum_paired_blocks"]
                  and metrics["formal_valid_three_day_start_count"] >= policy["minimum_contiguous_starts"]
                  and all(row["n"] >= policy["minimum_cell_samples"] and row["paired_block_count"] >= policy["minimum_paired_blocks"]
                          for row in metrics["targets"]))
    inconclusive = not sufficient or assessment["status"] == "inconclusive" or "formal_execution_coverage_insufficient" in failures
    outcome = "inconclusive" if inconclusive else "failed" if failures else "passed"
    return {**assessment, "outcome": outcome, "formal_confirmation": outcome == "passed",
            "status": "rejected" if outcome == "failed" else outcome,
            "failures": list(dict.fromkeys(failures)), "policy_digest": policy["policy_digest"],
            "paired_block_count": metrics["formal_paired_block_count"],
            "valid_three_day_start_count": metrics["formal_valid_three_day_start_count"],
            "formal_score_ucb": metrics["formal_score_ucb"],
            "point_evidence": metrics["formal_point_evidence"]}
