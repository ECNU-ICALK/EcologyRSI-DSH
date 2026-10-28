"""Run-frozen genome-owned prediction: the candidate's own output is the forecast.

Without this policy each sample Agent freely chose which tools to call and how
to average them, separately for every origin. On the sealed 2026-10 cucumber
runs that freedom cost about 0.03-0.08 weighted skill against the best single
registered tool on the same cohorts, and an unchanged genome re-run on the same
ten origins drifted by about 0.02 -- more than the gains being searched for. A
genome edit could therefore neither be measured nor attributed.

Under ``candidate_genome_copy@1`` the sample Agent still runs the strict
remote chain, but its only capability is the candidate's own executable tool
(``candidate-program`` when the genome authored a strategy program, otherwise
``candidate-model``), and its only admitted method is an exact ``model`` copy
validated by the frozen evidence formula. The genome alone then determines
every number: cells an edit does not touch stay bit-identical, an A/A repeat
measures zero drift, and the Host gates compare the edit itself.

Opt-in at creation only. Absent metadata keeps every archived run's semantics.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

POLICY_KEY = "prediction_ownership_policy"
POLICY = "candidate_genome_copy@1"

# Axes whose edits change the candidate's executable numerical output.
# ``registered_predictor`` is deliberately excluded: every alternative predictor
# admitted by the cellwise evaluator installs registry defaults that are either
# baseline-only (identical numbers) or a uniform 0.5-0.8 residual that the
# sealed evidence shows is worse in most cells, so the slot would be spent on a
# known non-improvement. Instruction, Skill, directive, critic-threshold and
# workflow edits cannot change an exact copy and are therefore withheld.
PREDICTION_AXES = frozenset({"scientific_parameter", "feature_recipe", "strategy_program"})
PREDICTION_OPERATIONS = frozenset({
    "set_bounded_parameter", "author_feature_recipe", "author_strategy_program",
})
CANDIDATE_MODEL_TOOL = "candidate-model"
CANDIDATE_PROGRAM_TOOL = "candidate-program"
SAMPLE_COPY_METHODS = ("model",)


def enabled(metadata: Mapping[str, Any] | None) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    value = metadata.get(POLICY_KEY)
    if value not in (None, POLICY):
        raise ValueError(f"{POLICY_KEY} must be {POLICY}")
    return value == POLICY


def validate_policy(value: Any) -> Any:
    if value not in (None, POLICY):
        raise ValueError(f"{POLICY_KEY} must be {POLICY}")
    return value


def candidate_tool_id(profile: Mapping[str, Any] | None) -> str:
    """The one capability whose output this candidate's predictions must copy."""
    if isinstance(profile, Mapping) and profile.get("strategy_program"):
        return CANDIDATE_PROGRAM_TOOL
    return CANDIDATE_MODEL_TOOL


def restrict_targets(targets: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    return {axis: tuple(values) for axis, values in targets.items() if axis in PREDICTION_AXES}


def non_prediction_operations(operations: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(item.get("op")) for item in operations
                   if not isinstance(item, Mapping) or item.get("op") not in PREDICTION_OPERATIONS})


def proposal_contract() -> dict[str, Any]:
    """Model-facing semantics shared by research synthesis and proposal stages."""
    return {
        "policy": POLICY,
        "prediction_owner": "candidate_genome",
        "sample_agent_role": (
            "Each sample Agent calls only this candidate's own tool once per origin and copies its "
            "exact output with method=model. It cannot select other tools, blend, adjust or "
            "predict directly, so prompts, Skills, directives, critic thresholds and workflow "
            "settings cannot change any forecast and are not offered."
        ),
        "candidate_tool": (
            "candidate-program when the genome has a strategy_program, otherwise candidate-model "
            "(the bound registered predictor with its nine per-cell residual scales: "
            "greenhouse-baseline-aligned-ridge@2 for the baseline-anchor seed, "
            "greenhouse-diurnal-profile-ensemble@1 for the expert-prior seed)."
        ),
        "allowed_operations": sorted(PREDICTION_OPERATIONS),
        "cell_isolation": (
            "Predictions are deterministic given the genome. A cell whose executable value an edit "
            "does not touch is bit-identical to the parent, so per-cell gates only test the cells "
            "the edit changes. Each changed cell must not regress; the weighted 9-cell gain must "
            "exceed the frozen practical delta."
        ),
        "scientific_parameter_reach": (
            "A residual-scale parameter changes forecasts only where candidate-model is used: "
            "directly when no strategy_program exists, or in cells whose program expression "
            "actually evaluates a candidate-model call. Otherwise the edit is inert and its "
            "hard effect check fails."
        ),
        "strategy_program_use": (
            "A strategy program may call up to three registered tools and combine them with one "
            "bounded expression of tool outputs and horizon. greenhouse-recipe-ridge@1 accepts "
            "parameters.feature_recipe; its per_cell map {'<target>@<h>h': {'residual_scale': s}} "
            "overrides per_horizon, and a cell absent from both uses no residual (the fit-selected "
            "baseline). Copy the default recipe from fit_partition_residual_panel when changing scales."
        ),
        "evidence": (
            "fit_partition_residual_panel reports training_fit-only forward-fold skill for both "
            "residual families at several scales. It is advisory search evidence, not a selection "
            "result; model_selection results decide."
        ),
    }


__all__ = [
    "CANDIDATE_MODEL_TOOL", "CANDIDATE_PROGRAM_TOOL", "POLICY", "POLICY_KEY", "PREDICTION_AXES",
    "PREDICTION_OPERATIONS", "SAMPLE_COPY_METHODS", "candidate_tool_id", "enabled",
    "non_prediction_operations", "proposal_contract", "restrict_targets", "validate_policy",
]
