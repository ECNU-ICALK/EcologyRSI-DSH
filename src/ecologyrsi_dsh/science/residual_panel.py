"""Label-safe training_fit forward-fold panel of per-cell residual scales.

Scope: only ``training_fit`` rows and labels, the partition already used to fit
every registered model and to select the scoring baseline. Calibration,
model-selection, validation and final-test labels are never read; no model is
called and no run is mutated. The panel is advisory search evidence for the
research and proposal stages, never selection or certification evidence.

Both panel families anchor on the fit-selected baseline, so a residual scale s
predicts ``baseline + s * (prediction(1) - baseline)``; one fit per family and
fold gives the whole scale curve.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from ..core.models import digest
from ..evaluators import greenhouse_prediction as prediction
from ..evaluators.baselines import apply_baseline_profile, fit_baseline_profile
from ..evaluators.objectives import skill_score
from .diagnostics import HORIZONS, TARGETS, _fold_series, training_fit_only

SCHEMA = "ecologyrsi-dsh.fit-partition-residual-panel/1"
METADATA_KEY = "fit_partition_residual_panel"
SCALES = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
_EVALUATION_HISTORY = prediction.COHORT_HISTORY_HOURS


def _families(candidate_model_id: str | None = None) -> tuple[dict[str, Any], ...]:
    if candidate_model_id == prediction.DIURNAL_PROFILE_MODEL_ID:
        # Expert-prior runs: candidate-model is the bound diurnal-profile
        # ensemble, so its scale curve is the one a scale edit actually moves.
        aligned = prediction.DiurnalProfileEnsembleConfig.defaults()
        candidate_tool = prediction.DIURNAL_PROFILE_MODEL_ID
    else:
        aligned = prediction.BaselineAlignedCellwiseRidgeConfig.from_mapping({
            "history_steps": 6, "ridge_alpha": 0.1,
            **{name: 1.0 for name in prediction.BaselineAlignedCellwiseRidgeConfig._SCALE_FIELDS.values()},
        })
        candidate_tool = prediction.BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID
    recipe_defaults = prediction.seed_recipe_parameters(horizons=HORIZONS)
    full = {**recipe_defaults["feature_recipe"],
            "per_horizon": {str(h): {"residual_scale": 1.0} for h in HORIZONS}}
    full.pop("per_cell", None)
    recipe = prediction.RecipeRidgeConfig.from_mapping({"feature_recipe": full})
    return (
        {"family": "candidate-model", "tool_id": candidate_tool,
         "how_to_apply": ("set_bounded_parameter <target>_<h>h_residual_scale on the bound candidate "
                          "model (one-operation step limit applies)"),
         "config": aligned, "default_parameters": None},
        {"family": "strategy-program:greenhouse-recipe-ridge@1", "tool_id": prediction.RECIPE_RIDGE_MODEL_ID,
         "how_to_apply": ("author_strategy_program calling greenhouse-recipe-ridge@1 with "
                          "parameters.feature_recipe = default_parameters.feature_recipe plus a per_cell "
                          "map of residual scales; the expression copies that call"),
         "config": recipe, "default_parameters": recipe_defaults},
    )


def fit_partition_residual_panel(source: Any, *, folds: int = 3, initial_fit_fraction: float = 0.4,
                                 purge_hours: int = 24, candidate_model_id: str | None = None) -> dict[str, Any]:
    cropped = training_fit_only(source)
    initial_end = int(len(cropped.timestamps) * initial_fit_fraction)
    if initial_end < 2 * _EVALUATION_HISTORY:
        raise ValueError("training_fit is too short for the residual panel")
    boundaries = [initial_end + (len(cropped.timestamps) - initial_end) * i // folds
                  for i in range(folds + 1)]
    families = _families(candidate_model_id)
    # sums[family][cell] = [candidate_se_by_scale..., baseline_se, n]
    sums = {item["family"]: {(t, h): [[0.0] * len(SCALES), 0.0, 0] for t in TARGETS for h in HORIZONS}
            for item in families}
    origin_count = 0
    for fold in range(folds):
        series = _fold_series(cropped, boundaries[fold], boundaries[fold + 1], purge_hours)
        profile = fit_baseline_profile(series, targets=TARGETS, horizons=HORIZONS)
        tables = {}
        for item in families:
            fitted = prediction.fit_predict_exogenous_ridge(
                series, targets=TARGETS, horizons=HORIZONS, config=item["config"],
                evaluation_history_steps=_EVALUATION_HISTORY, prediction_partitions=("training_feedback",))
            rows = [row for row in fitted["prediction_rows"] if row["partition"] == "training_feedback"]
            rows = apply_baseline_profile(series, rows, profile)
            table = {}
            for row in rows:
                values = (row.get("observed"), row.get("baseline"), row.get("predicted"))
                if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                       for v in values):
                    table[(row["origin_timestamp"], row["target"], row["horizon_hours"])] = tuple(map(float, values))
            tables[item["family"]] = table
        complete = None
        for table in tables.values():
            origins = {key[0] for key in table}
            full = {o for o in origins if all((o, t, h) in table for t in TARGETS for h in HORIZONS)}
            complete = full if complete is None else complete & full
        complete = complete or set()
        origin_count += len(complete)
        for family, table in tables.items():
            for (origin, target, horizon), (observed, baseline, predicted) in table.items():
                if origin not in complete:
                    continue
                cell = sums[family][(target, horizon)]
                for index, scale in enumerate(SCALES):
                    cell[0][index] += (baseline + scale * (predicted - baseline) - observed) ** 2
                cell[1] += (baseline - observed) ** 2
                cell[2] += 1
    if origin_count == 0:
        raise ValueError("residual panel has no complete nine-cell origins")
    report_families = []
    for item in families:
        cells = {}
        for (target, horizon), (candidate_se, baseline_se, n) in sums[item["family"]].items():
            skills = [round(skill_score(math.sqrt(se / n), math.sqrt(baseline_se / n)), 4)
                      for se in candidate_se]
            best = max(range(len(SCALES)), key=lambda i: skills[i])
            cells[f"{target}@{horizon}h"] = {"skill_by_scale": skills,
                                             "best_scale": SCALES[best] if skills[best] > 0 else 0.0}
        report_families.append({
            "family": item["family"], "tool_id": item["tool_id"], "how_to_apply": item["how_to_apply"],
            **({"default_parameters": item["default_parameters"]} if item["default_parameters"] else {}),
            "cells": cells,
        })
    body = {
        "schema_version": SCHEMA,
        "label_scope": "training_fit_only_forward_folds",
        "not_selection_evidence": True,
        "folds": folds, "initial_fit_fraction": initial_fit_fraction, "purge_hours": purge_hours,
        "fit_rows": len(cropped.timestamps), "evaluated_origins": origin_count,
        # Only expert-prior panels name their candidate family, so a baseline
        # run's panel body (and digest) stays byte-identical to 0.10.32.
        **({"candidate_model_id": candidate_model_id}
           if candidate_model_id == prediction.DIURNAL_PROFILE_MODEL_ID else {}),
        "scales": list(SCALES),
        "skill_definition": "weighted_symmetric_rmse_skill against the fold's fit-selected baseline; 0 = baseline",
        "families": report_families,
        "interpretation": (
            "Positive skill means that residual scale beat the fit-selected baseline on later training_fit "
            "hours. Scale 0 reproduces the baseline exactly. Skill can differ in the model_selection period; "
            "prefer cells with a clear positive curve and keep scales conservative where it peaks low."
        ),
    }
    return {**body, "panel_digest": digest(body)}


def validated_panel(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA:
        return None
    body = {key: item for key, item in value.items() if key != "panel_digest"}
    from ..core.immutable import thaw_json
    body = thaw_json(body)
    if value.get("panel_digest") != digest(body):
        raise ValueError("fit-partition residual panel digest mismatch")
    return {**body, "panel_digest": value["panel_digest"]}


__all__ = ["METADATA_KEY", "SCALES", "SCHEMA", "fit_partition_residual_panel", "validated_panel"]
