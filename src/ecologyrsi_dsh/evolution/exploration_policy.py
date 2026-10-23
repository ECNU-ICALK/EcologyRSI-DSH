"""Opt-in search semantics; never add defaults while replaying an old run."""
from collections.abc import Mapping

POLICY_KEY = "candidate_exploration_policy"
POLICY = "mechanism_evidence_search@1"


def enabled(metadata: Mapping) -> bool:
    value = metadata.get(POLICY_KEY)
    if value not in (None, POLICY):
        raise ValueError("unsupported candidate_exploration_policy")
    return value == POLICY


def contract(metadata: Mapping) -> dict:
    if not enabled(metadata):
        return {}
    return {
        "policy": POLICY,
        "minimum_distinct_mechanisms": 3,
        "free_slots": 1,
        "prediction_distance": "mean_absolute_delta_per_training_fit_scale@1",
        "near_output_threshold": 1e-6,
        "comparison_use": "training_diagnostic_not_selection_or_scientific_gain",
        "archive_max_entries": 6,
        "archive_max_bytes": 6000,
        "additional_prediction_calls": 0,
    }
