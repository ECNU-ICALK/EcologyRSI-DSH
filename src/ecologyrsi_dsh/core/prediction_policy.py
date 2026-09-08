"""Runtime-owned predictor choices, separate from frozen measurement rules."""
from collections.abc import Mapping

RUNTIME_PREDICTION_POLICY = "model_during_run@1"
RUNTIME_EVALUATOR_ID = "greenhouse_multihorizon_time_forward@4"
BASELINE_REFERENCE_PREDICTOR_ID = "greenhouse-baseline-aligned-ridge@1"


def prediction_usage(predictor_id: str, parameters: Mapping) -> dict:
    if predictor_id == BASELINE_REFERENCE_PREDICTOR_ID:
        horizons = [h for h in (1, 6, 24) if parameters.get(f"residual_scale_{h}h", 0) != 0]
        return {"mode": "residual_model" if horizons else "baseline_only",
                "predictor_id": predictor_id, "model_horizons": horizons,
                "baseline_only_horizons": [h for h in (1, 6, 24) if h not in horizons]}
    return {"mode": "registered_model", "predictor_id": predictor_id}
