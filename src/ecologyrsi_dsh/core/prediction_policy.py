"""Runtime-owned predictor choices, separate from frozen measurement rules."""
from collections.abc import Mapping

RUNTIME_PREDICTION_POLICY = "model_during_run@1"
RUNTIME_EVALUATOR_ID = "greenhouse_multihorizon_time_forward@4"
CELLWISE_EVALUATOR_ID = "greenhouse_multihorizon_time_forward@5"
# Same cellwise scoring as @5, plus the research-derived expert predictor. A
# separate id keeps @5's frozen configuration digest (and every run bound to
# it) byte-identical.
EXPERT_CELLWISE_EVALUATOR_ID = "greenhouse_multihorizon_time_forward@6"
CELLWISE_EVALUATOR_IDS = frozenset((CELLWISE_EVALUATOR_ID, EXPERT_CELLWISE_EVALUATOR_ID))
DIURNAL_PROFILE_PREDICTOR_ID = "greenhouse-diurnal-profile-ensemble@1"
# Opt-in seed for native autonomous greenhouse runs: start the evolution from
# the expert predictor instead of the no-residual baseline anchor.
SEED_PRIOR_POLICY_KEY = "seed_prior_policy"
EXPERT_SEED_PRIOR_POLICY = "expert_diurnal_profile@1"
TARGETED_SEARCH_POLICY = "cellwise_training_evidence@1"
LEGACY_CONTRIBUTION_POLICY = "prediction_contribution@1"
PREVIOUS_CONTRIBUTION_POLICY = "prediction_contribution@2"
CONTRIBUTION_POLICY = "prediction_contribution@3"
CONTRIBUTION_POLICIES = frozenset((LEGACY_CONTRIBUTION_POLICY, PREVIOUS_CONTRIBUTION_POLICY, CONTRIBUTION_POLICY))
BASELINE_REFERENCE_PREDICTOR_ID = "greenhouse-baseline-aligned-ridge@1"


def prediction_usage(predictor_id: str, parameters: Mapping) -> dict:
    if predictor_id in ("greenhouse-baseline-aligned-ridge@2", DIURNAL_PROFILE_PREDICTOR_ID):
        active = [f"{t}@{h}h" for t in ("air_temperature", "relative_humidity", "co2_concentration")
                  for h in (1, 6, 24) if parameters.get(f"{t}_{h}h_residual_scale", 0) != 0]
        return {"mode": "residual_model" if active else "baseline_only", "predictor_id": predictor_id,
                "active_residual_cells": active, "baseline_only_cell_count": 9-len(active)}
    if predictor_id == BASELINE_REFERENCE_PREDICTOR_ID:
        horizons = [h for h in (1, 6, 24) if parameters.get(f"residual_scale_{h}h", 0) != 0]
        return {"mode": "residual_model" if horizons else "baseline_only",
                "predictor_id": predictor_id, "model_horizons": horizons,
                "baseline_only_horizons": [h for h in (1, 6, 24) if h not in horizons]}
    return {"mode": "registered_model", "predictor_id": predictor_id}
