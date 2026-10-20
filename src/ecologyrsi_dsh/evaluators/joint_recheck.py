"""Predeclared, bounded confirmation for the unified joint search protocol.

Time blocks and inference replicas answer different questions. Neither grants
independent certification; repeats never increase the observation count.
"""
from dataclasses import replace

from ..core.trajectory import HoldoutArm
from ..evolution.promotion import _validated_evidence, _resampled_objective

POLICY = "joint-conditional-paired-recheck@1"
MINIMUM_TIME_BLOCKS = 3
MARGINAL_GAIN_MULTIPLIER = 2


def temporal_gate(candidate, incumbent, profile):
    left, right = _validated_evidence(candidate), _validated_evidence(incumbent)
    if (left is None or right is None or left["config_digest"] != right["config_digest"]
            or set(left["blocks"]) != set(right["blocks"])):
        return {"passed": False, "reason": "temporal_evidence_incomplete", "block_count": 0}
    ids = sorted(left["blocks"])
    if len(ids) < MINIMUM_TIME_BLOCKS:
        return {"passed": False, "reason": "additional_time_blocks_required", "block_count": len(ids)}
    # Every cell in every compared block must actually have been observed.
    if any(cell["eligible"] <= 0 or cell["succeeded"] / cell["eligible"] < profile.selection_minimum_coverage
           for evidence in (left, right) for cells in evidence["blocks"].values() for cell in cells.values()):
        return {"passed": False, "reason": "temporal_coverage_failed", "block_count": len(ids)}
    deltas = [_resampled_objective(left, [key]) - _resampled_objective(right, [key]) for key in ids]
    passed = (sum(value > 0 for value in deltas) > len(deltas) / 2
              and min(deltas) >= -profile.selection_cell_regression_tolerance)
    return {"passed": passed, "reason": "passed" if passed else "gain_not_consistent_across_time_blocks",
            "block_count": len(ids), "block_deltas": deltas,
            "independent_certification": False}


def assess_recheck(candidate, incumbent, profile, *, base_eligible, repeats=()):
    """A borderline *passing* result may be confirmed once, never rescued.

    Failed coverage/constraints or inadequate temporal support retain the
    incumbent. Only small gains or mixed (within-tolerance) day gains trigger
    the one repeat. Both original and repeated comparisons must pass.
    """
    temporal = temporal_gate(candidate, incumbent, profile)
    delta = candidate.score - incumbent.score
    required = bool(base_eligible and temporal["passed"] and (
        delta <= MARGINAL_GAIN_MULTIPLIER * profile.selection_minimum_score_delta
        or any(value <= 0 for value in temporal.get("block_deltas", ()))))
    result = {"policy": POLICY, "required": required, "temporal": temporal,
              "maximum_rechecks": 1, "independent_observation_count_multiplier": 1,
              "passed": bool(base_eligible and temporal["passed"] and not required),
              "status": "pending" if required else "not_required" if base_eligible and temporal["passed"] else "retained"}
    if not repeats:
        return result
    if not required:
        raise ValueError("joint recheck was not triggered by the frozen first comparison")
    by_arm = {item.scope.holdout_arm: item for item in repeats}
    if len(repeats) != 2 or set(by_arm) != {HoldoutArm.FINALIST_1, HoldoutArm.INCUMBENT}:
        raise ValueError("joint recheck requires exactly two arms")
    for original in (candidate, incumbent):
        repeated = by_arm[original.scope.holdout_arm]
        if (repeated.scope != replace(original.scope, inference_replica=1)
                or repeated.evaluator_digest != original.evaluator_digest):
            raise ValueError("joint recheck differs from frozen cohort, revision or evaluator")
    from .generation_comparison import build_generation_comparison
    repeated = build_generation_comparison(run_id=candidate.scope.run_id,
        generation=candidate.scope.generation, cohort_digest=candidate.scope.cohort_digest,
        holdout_evaluations=tuple(repeats), fitness_profile=profile,
        quick_experiment=True, experiment_protocol="joint_plugin_epoch@1")
    repeat_gate = repeated.gate_results["arms"][HoldoutArm.FINALIST_1.value]
    repeat_temporal = temporal_gate(by_arm[HoldoutArm.FINALIST_1], by_arm[HoldoutArm.INCUMBENT], profile)
    passed = bool(repeat_gate["search_eligible"] and repeat_temporal["passed"])
    result.update(passed=passed, status="confirmed" if passed else "not_confirmed",
        repeated_delta=by_arm[HoldoutArm.FINALIST_1].score - by_arm[HoldoutArm.INCUMBENT].score,
        repeated_failures=list(repeat_gate["search_failures"]), repeated_temporal=repeat_temporal)
    return result
