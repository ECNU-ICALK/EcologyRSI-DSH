"""Default-tool sensitivity is advisory; the Agent owns final predictions."""
from typing import Any
from ..evaluators.greenhouse_prediction import BASELINE_ALIGNED_RIDGE_MODEL_ID
from .genome import EcologyEvolutionPluginGenome


def parameter_activity_contract(genome: EcologyEvolutionPluginGenome) -> dict[str, Any]:
    scientific = genome.scientific_program
    parameters = scientific["parameter_overrides"]
    zero = (scientific["predictor_ref"]["id"] == BASELINE_ALIGNED_RIDGE_MODEL_ID
            and all(parameters.get(f"residual_scale_{h}h") == 0.0 for h in (1, 6, 24)))
    return {
        "schema_version": "ecologyrsi-dsh.default-tool-parameter-activity/2",
        "scope": "optional_candidate_model_only",
        "zero_effect_default_tool_parameters": ["history_steps", "ridge_alpha"] if zero else [],
        "agent_policy_parameters_pruned": False,
        "reason": (
            "Zero residual scales neutralize history_steps and ridge_alpha only in the default tool. "
            "The Agent can use other tools, override their parameters, and adjust final predictions. "
            "Policy effects must be measured on the frozen evaluation cohort."
        ),
    }
