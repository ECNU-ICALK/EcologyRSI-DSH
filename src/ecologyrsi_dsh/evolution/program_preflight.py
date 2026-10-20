"""Value-blind strategy-tool input checks before a candidate can be scored.

This checks necessary execution conditions, not prediction quality or every
possible arithmetic/runtime failure. No observations, labels, fitted models or
LLM calls enter the check. Historical manifests opt out by omission.
"""
from ..core.models import digest
from ..core.immutable import thaw_json

POLICY_KEY = "candidate_executability_policy"
POLICY = "strategy_tool_input_contract@1"


def enabled(metadata):
    value = metadata.get(POLICY_KEY)
    if value not in (None, POLICY):
        raise ValueError("unsupported candidate_executability_policy")
    return value == POLICY


class CandidateProgramPreflightError(ValueError):
    pass


def contract(metadata):
    if not enabled(metadata):
        return None
    return {"policy": POLICY, "label_access": False,
            "checks": ["declared_call_cost", "causal_history_offsets", "model_history_capacity"],
            "instruction": (
                "Every declared strategy call executes before its expression, including unused branches. "
                "Causal tools see the candidate predictor's history offsets, not the larger cohort alignment. "
                "Seasonal period P needs offset h%P-P (or zero when divisible) for every horizon. "
                "Choose applicable calls or revise the program within the existing proposal budget; "
                "never extend history, skip cells, use future labels or substitute a fallback."
            )}


def validate_program_inputs(task, *, predictor_id, parameters, role_profiles):
    """Return a digest-bound compile receipt, or reject before native evaluation."""
    if not enabled(task.metadata):
        return None
    # Local revisions carry immutable nested mappings; normalize only this
    # validation copy, exactly as the compiled/program persistence boundaries do.
    parameters = thaw_json(parameters)
    from ..data.adapters import dataset_adapter
    from ..evaluators.agent_model_tools import _CONFIGS, _defaults
    from ..evaluators.feature_recipe import compile_feature_plan
    from ..evaluators.greenhouse_prediction import RecipeRidgeConfig
    from ..evaluators.strategy_program import validate_strategy_program, _causal_parameters
    from ..core.agent_prediction import PREDICTION_TOOL_CALL_BUDGET

    programs = [p["strategy_program"] for p in role_profiles
                if p.get("role") == "sample-planner" and p.get("strategy_program") is not None]
    body = {"schema_version": "ecologyrsi-dsh.candidate-executability/1", "policy": POLICY,
            "scope": "necessary_tool_input_conditions_only", "label_access": False,
            "predictor_id": predictor_id, "parameters_digest": digest(parameters),
            "status": "not_applicable", "programs": []}
    if not programs:
        return {**body, "receipt_digest": digest(body)}
    if predictor_id not in _CONFIGS:
        raise CandidateProgramPreflightError("strategy preflight has no history contract for this predictor")
    config = _CONFIGS[predictor_id].from_mapping(parameters)
    adapter = dataset_adapter(task.dataset)
    alignment = task.metadata.get("origin_history_alignment_hours")
    if type(alignment) is not int or alignment < 1:
        raise CandidateProgramPreflightError("strategy preflight requires frozen origin history alignment")

    def offsets(cfg, target, horizon):
        if isinstance(cfg, RecipeRidgeConfig):
            return compile_feature_plan(cfg.recipe, target=target, horizon_hours=horizon,
                                        origin_history_alignment=alignment).required_timestamp_offsets
        return tuple(-lag for lag in range(cfg.history_steps))

    for raw in programs:
        program = validate_strategy_program(raw)
        cost = len(program["calls"]) + 1
        if cost > PREDICTION_TOOL_CALL_BUDGET:
            raise CandidateProgramPreflightError("strategy declared calls exceed the frozen tool budget")
        cells = []
        for target in adapter.target_names:
            for horizon in adapter.horizons_hours:
                available = set(offsets(config, target, horizon))
                if not available or max(available) > 0 or -min(available) >= alignment:
                    raise CandidateProgramPreflightError("candidate history exceeds frozen causal alignment")
                checked = []
                for call in program["calls"]:
                    tool, params = call["tool_id"], call["parameters"]
                    required = []
                    if tool.startswith("causal-"):
                        params = _causal_parameters(tool, params)
                        if tool == "causal-seasonal@1":
                            phase = horizon % params["period"]
                            required = [phase - params["period"] if phase else 0]
                        else:
                            if len(available) < params["window"]:
                                raise CandidateProgramPreflightError(
                                    f"{tool} needs {params['window']} history values but {target}@{horizon}h "
                                    f"provides {len(available)}; revise the declared call")
                            required = sorted(available, reverse=True)[:params["window"]]
                    elif tool == "persistence":
                        required = [0]
                    elif tool in _CONFIGS:
                        # Model tools read their own causal series, unlike the
                        # causal-* tools that read the candidate's prompt window.
                        defaults = (config.to_dict() if tool == predictor_id
                                    else _defaults(_CONFIGS[tool], adapter.horizons_hours))
                        # A recipe tool is advertised only if its default read
                        # plan fits the frozen origin alignment.
                        if _CONFIGS[tool] is RecipeRidgeConfig:
                            offsets(RecipeRidgeConfig.from_mapping(defaults), target, horizon)
                        cfg = _CONFIGS[tool].from_mapping({**defaults, **dict(params)})
                        model_offsets = offsets(cfg, target, horizon)
                        if -min(model_offsets) >= alignment:
                            raise CandidateProgramPreflightError(
                                f"{tool} requires history beyond the frozen alignment")
                    missing = sorted(set(required) - available)
                    if missing:
                        raise CandidateProgramPreflightError(
                            f"{tool} cannot execute for {target}@{horizon}h: missing causal offsets {missing}; "
                            "revise the declared call, not the data boundary")
                    checked.append({"tool_id": tool, "required_prompt_offsets": required})
                cells.append({"target": target, "horizon_hours": horizon,
                              "available_prompt_offsets": sorted(available), "calls": checked})
        body["programs"].append({"program_digest": digest(program), "charged_calls": cost,
                                 "checked_cell_count": len(cells), "input_contract_digest": digest(cells)})
    body["status"] = "input_contract_checked"
    return {**body, "receipt_digest": digest(body)}
