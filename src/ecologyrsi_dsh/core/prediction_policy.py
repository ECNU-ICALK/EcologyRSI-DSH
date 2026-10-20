"""Runtime-owned predictor choices, separate from frozen measurement rules."""
from collections.abc import Mapping

RUNTIME_PREDICTION_POLICY = "model_during_run@1"
RUNTIME_EVALUATOR_ID = "greenhouse_multihorizon_time_forward@4"
CELLWISE_EVALUATOR_ID = "greenhouse_multihorizon_time_forward@5"
TARGETED_SEARCH_POLICY = "cellwise_training_evidence@1"
LEGACY_CONTRIBUTION_POLICY = "prediction_contribution@1"
CONTRIBUTION_POLICY = "prediction_contribution@2"
CONTRIBUTION_POLICIES = frozenset((LEGACY_CONTRIBUTION_POLICY, CONTRIBUTION_POLICY))
BASELINE_REFERENCE_PREDICTOR_ID = "greenhouse-baseline-aligned-ridge@1"


def prediction_usage(predictor_id: str, parameters: Mapping) -> dict:
    if predictor_id == "greenhouse-baseline-aligned-ridge@2":
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
