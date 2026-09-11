"""Shared new-run parameter contract for API validation and browser controls."""

from copy import deepcopy
from typing import Any

from ..execution.sample_admission import DEFAULT_SAMPLE_CONCURRENCY, MAX_SAMPLE_CONCURRENCY


PARAMETER_RULES = {
    "rounds": {"default": 5, "minimum": 1, "maximum": 50},
    "candidates_per_generation": {"default": 4, "minimum": 4, "maximum": 4},
    "max_candidates": {"default": 20, "minimum": 4, "maximum": 256},
    "formal_origin_count": {"default": 100, "minimum": 2},
    "local_batch_origin_count": {"default": 10, "minimum": 2},
    "max_local_edits_per_batch": {"default": 2, "minimum": 0, "maximum": 5},
    "selection_holdout_origin_count": {"default": 50, "minimum": 40},
    "candidate_concurrency": {"default": 4, "minimum": 1, "maximum": 8},
    "sample_concurrency": {"default": DEFAULT_SAMPLE_CONCURRENCY, "minimum": 1,
                           "maximum": MAX_SAMPLE_CONCURRENCY},
}


def run_parameter_contract(profile: Any = None) -> dict[str, Any]:
    parameters = deepcopy(PARAMETER_RULES)
    result = {"schema_version": "ecologyrsi-dsh.run-parameters/1", "parameters": parameters,
              "local_comparison_mode": "exploratory_paired_point_comparison",
              "constraints": ["formal_origins_divisible_by_batch", "candidate_budget_covers_epochs",
                              "target_time_purged", "epoch_temporal_evidence_required"]}
    if profile is not None:
        rule = parameters["selection_holdout_origin_count"]
        rule["minimum"] = max(rule["minimum"], profile.selection_minimum_cell_samples)
        rule["default"] = max(rule["default"], rule["minimum"])
        cells = profile.prediction_cell_count
        parameters["sample_agent_batch_size"] = {"default": cells, "minimum": cells, "maximum": cells}
        result["selection_evidence"] = {
            "minimum_cell_samples": rule["minimum"],
            "minimum_day_blocks": profile.selection_minimum_paired_blocks,
            "moving_block_days": profile.moving_block_days,
            "minimum_contiguous_starts": profile.selection_minimum_valid_three_day_starts,
            "minimum_coverage": profile.selection_minimum_coverage,
            "complete_paired_execution_required": True,
        }
    return result


def validate_run_parameter(name: str, value: Any, *, profile: Any = None) -> int:
    rule = run_parameter_contract(profile)["parameters"][name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < rule["minimum"] or ("maximum" in rule and value > rule["maximum"]):
        boundary = f"{rule['minimum']}..{rule['maximum']}" if "maximum" in rule else f">= {rule['minimum']}"
        raise ValueError(f"{name} must be {boundary}")
    return value
