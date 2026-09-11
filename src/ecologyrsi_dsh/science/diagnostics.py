"""Bounded, local diagnostics physically confined to the training-fit partition.

These experiments never publish an evaluation, mutate a run, or call an LLM.
The numerical execution comparison is a host-only shadow benchmark, not a
measurement of the value or performance of a real native agent session.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from statistics import fmean
from time import perf_counter
from typing import Any

from ..core.models import digest
from ..data.contracts import DatasetSeries, SelectionDatasetView
from ..data.splits import IndexRange
from ..evaluators.baselines import apply_baseline_profile, fit_baseline_profile
from ..evaluators import greenhouse_prediction as prediction
from ..evaluators.objectives import skill_score


TARGETS = ("air_temperature", "relative_humidity", "co2_concentration")
HORIZONS = (1, 6, 24)
SCHEMA = "ecologyrsi-dsh.training-fit-diagnostic/1"
_BASELINES = ("persistence", "seasonal_24h", "fit_selected_baseline")


@dataclass(frozen=True)
class DiagnosticCandidate:
    name: str
    model_id: str
    parameters: Mapping[str, Any]

    def config(self) -> Any:
        classes = {
            prediction.EXOGENOUS_RIDGE_MODEL_ID: prediction.ExogenousRidgeConfig,
            prediction.TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID: prediction.TargetwiseExogenousRidgeConfig,
            prediction.HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID: prediction.HorizonTargetwiseExogenousRidgeConfig,
        }
        aligned = getattr(prediction, "BaselineAlignedRidgeConfig", None)
        if aligned is not None:
            classes["greenhouse-baseline-aligned-ridge@1"] = aligned
        recipe = getattr(prediction, "RecipeRidgeConfig", None)
        if recipe is not None:
            classes[prediction.RECIPE_RIDGE_MODEL_ID] = recipe
        if self.model_id not in classes:
            raise ValueError("diagnostic model is not a supported local ridge")
        return classes[self.model_id].from_mapping(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "model_id": self.model_id,
                "parameters": self.config().to_dict()}


def default_candidates() -> tuple[DiagnosticCandidate, ...]:
    """Small explicit presets; names do not assert a historical run identity."""
    base = {"history_steps": 6, "ridge_alpha": .1, "residual_scale": .5}
    model = prediction.EXOGENOUS_RIDGE_MODEL_ID
    scalar = tuple(DiagnosticCandidate(name, model, parameters) for name, parameters in (
        ("seed", base),
        ("champion", {**base, "ridge_alpha": .3}),
        ("scale_0", {**base, "residual_scale": 0.0}),
        ("scale_0.25", {**base, "residual_scale": .25}),
        ("scale_1", {**base, "residual_scale": 1.0}),
        ("alpha_1", {**base, "ridge_alpha": 1.0}),
    ))
    # The recipe seed is the cheap, model-free answer to "does a winner exist in
    # the search space at all". A fixed-window candidate cannot read the cell
    # the selected seasonal_24h baseline reads at h=6, so its skill there is
    # negative by construction; the recipe seed can. Running both here means
    # the question is settled deterministically before a run is paid for.
    return scalar + (
        DiagnosticCandidate(
            "recipe_seed",
            prediction.RECIPE_RIDGE_MODEL_ID,
            prediction.seed_recipe_parameters(
                horizons=HORIZONS,
                exogenous_columns=prediction.GREENHOUSE_SEED_EXOGENOUS_COLUMNS,
            ),
        ),
    )


def candidates_from_payload(payload: Any) -> tuple[DiagnosticCandidate, ...]:
    """Load explicit local candidates without executable configuration imports."""
    if not isinstance(payload, list) or not 1 <= len(payload) <= 24:
        raise ValueError("candidate JSON must be a list of one to 24 candidates")
    result = []
    for item in payload:
        if not isinstance(item, Mapping) or set(item) != {"name", "model_id", "parameters"}:
            raise ValueError("candidate requires exactly name, model_id and parameters")
        candidate = DiagnosticCandidate(**item)
        candidate.config()
        result.append(candidate)
    return tuple(result)


def training_fit_only(source: DatasetSeries | SelectionDatasetView) -> DatasetSeries:
    """Copy only declared fit values before fitting, feature scans or scoring."""
    selected = source.partitions.get("training_fit")
    if selected is None or not 0 <= selected.start < selected.end <= len(source.timestamps):
        raise ValueError("a non-empty bounded training_fit partition is required")
    timestamps = tuple(source.timestamps[selected.start:selected.end])
    values = {name: tuple(column[selected.start:selected.end])
              for name, column in source.values.items()}
    if any(len(column) != len(timestamps) for column in values.values()):
        raise ValueError("fit feature columns must align with fit timestamps")
    if any(type(value) is not int for value in timestamps) or any(
        left >= right for left, right in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("fit timestamps must be strictly increasing integer hours")
    identity = {"schema": SCHEMA, "dataset_id": source.dataset_id,
                "episode_id": source.episode_id, "timestamps": timestamps,
                "values": {name: [value if _finite(value) else None for value in column]
                           for name, column in values.items()}}
    for name in TARGETS:
        if name not in values:
            raise ValueError(f"missing diagnostic target: {name}")
    return DatasetSeries(
        schema=source.schema, dataset_id=source.dataset_id, domain_id=source.domain_id,
        episode_id=source.episode_id, digest=digest(identity), timestamps=timestamps,
        values=values, features=dict(source.features),
        partitions={"training_fit": IndexRange(0, len(timestamps)),
                    "training_feedback": IndexRange(len(timestamps), len(timestamps))},
        split_manifest_digest_sha256=digest({"scope": "training_fit_only",
                                            "timestamps": timestamps}),
    )


def _finite(value: Any) -> bool:
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _fold_series(source: DatasetSeries, fit_end: int, end: int, purge_hours: int) -> DatasetSeries:
    threshold = source.timestamps[fit_end - 1] + purge_hours + 1
    start = next((i for i in range(fit_end, end) if source.timestamps[i] >= threshold), end)
    if start == end:
        raise ValueError("fold has no observations after the elapsed-time purge")
    partitions = {"training_fit": IndexRange(0, fit_end),
                  "training_feedback": IndexRange(start, end)}
    return DatasetSeries(
        schema=source.schema, dataset_id=source.dataset_id, domain_id=source.domain_id,
        episode_id=source.episode_id,
        digest=digest({"dataset_id": source.dataset_id, "episode_id": source.episode_id,
                       "fit_timestamps": source.timestamps[:fit_end],
                       "fit_values": {name: [value if _finite(value) else None for value in values[:fit_end]]
                                      for name, values in source.values.items()}}),
        timestamps=source.timestamps[:end],
        values={name: values[:end] for name, values in source.values.items()},
        partitions=partitions, features=source.features,
        split_manifest_digest_sha256=digest({"source_fit": source.split_manifest_digest_sha256,
                                            "fold_partitions": {k: v.to_dict() for k, v in partitions.items()}}),
    )


def _case(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return (row["origin_timestamp"], row["target"], row["horizon_hours"])


def _fitted_identity(models: Sequence[Mapping[str, Any]]) -> str:
    from ..evaluators.greenhouse_prediction import fitted_model_identity
    return fitted_model_identity(models)


def _metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells = []
    for target in TARGETS:
        for horizon in HORIZONS:
            selected = [r for r in rows if r["target"] == target and r["horizon_hours"] == horizon]
            if not selected:
                raise ValueError("diagnostic comparison requires complete nine-cell origins")
            errors = [float(r["predicted"]) - r["observed"] for r in selected]
            base_errors = [float(r["baseline"]) - r["observed"] for r in selected]
            rmse = math.sqrt(fmean(e * e for e in errors))
            baseline_rmse = math.sqrt(fmean(e * e for e in base_errors))
            cells.append({"target": target, "horizon_hours": horizon, "n": len(selected),
                          "rmse": rmse, "mae": fmean(abs(e) for e in errors),
                          "bias": fmean(errors), "baseline_rmse": baseline_rmse,
                          "skill_score": skill_score(rmse, baseline_rmse),
                          "relative_rmse_gain": 1 - rmse / baseline_rmse if baseline_rmse else None,
                          "fallback_count": sum(bool(r.get("used_zero_residual_fallback")) for r in selected)})
    return cells


def _scoring_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return digest([{"case": _case(r), "prediction": r["predicted"], "baseline": r["baseline"]}
                   for r in sorted(rows, key=_case)])


def compare_host_execution(rows: Sequence[Mapping[str, Any]], models: Sequence[Mapping[str, Any]],
                           config: Any, *, predictor: Callable[..., float] | None = None) -> tuple[list[dict], dict]:
    """Execute the same fitted snapshot flat and per origin; return bounded audit.

    Raw rows stay internal to diagnostics. Callers must not publish the first
    return value; the second value has only hashes, counts and wall-clock time.
    """
    predict = predictor or prediction.predict_fitted_exogenous_ridge
    frozen_rows = tuple(sorted(rows, key=_case))
    def invoke(row):
        return float(predict(target=row["target"], horizon_hours=row["horizon_hours"],
                             baseline=row["baseline"], label_free_context=row["label_free_context"],
                             models=models, config=config))
    started = perf_counter()
    flat = [invoke(row) for row in frozen_rows]
    batch_seconds = perf_counter() - started
    started = perf_counter()
    by_origin = {}
    for row in frozen_rows:
        by_origin.setdefault(row["origin_timestamp"], []).append(row)
    sequential = {_case(row): invoke(row) for origin in sorted(by_origin) for row in by_origin[origin]}
    origin_seconds = perf_counter() - started
    returned = [{**row, "predicted": value} for row, value in zip(frozen_rows, flat, strict=True)]
    other = [{**row, "predicted": sequential[_case(row)]} for row in frozen_rows]
    delta = max((abs(left["predicted"] - right["predicted"]) for left, right in zip(returned, other)), default=0.0)
    audit = {"mode": "host_only_same_snapshot_flat_vs_origin_grouped",
             "real_native_llm_executed": False, "remote_requests": 0,
             "prediction_count": len(rows), "origin_count": len(by_origin),
             "fitted_snapshot_digest": _fitted_identity(models),
             "input_identity_digest": digest([{"case": _case(r), "context": r["label_free_context"]} for r in frozen_rows]),
             "prediction_identity_equal": _scoring_digest(returned) == _scoring_digest(other),
             "prediction_max_absolute_difference": delta,
             "mask_equal": all(_finite(r["predicted"]) == _finite(o["predicted"]) for r, o in zip(returned, other)),
             "case_identity_equal": [_case(r) for r in returned] == [_case(r) for r in other],
             "baseline_equal": [r["baseline"] for r in returned] == [r["baseline"] for r in other],
             "flat_host_seconds": batch_seconds, "origin_grouped_host_seconds": origin_seconds}
    return returned, audit


def run_training_fit_diagnostics(source: DatasetSeries | SelectionDatasetView, *,
                          candidates: Sequence[DiagnosticCandidate] | None = None,
                          folds: int = 3, initial_fit_fraction: float = .4,
                          purge_hours: int = 24,
                          fit_runner: Callable[..., dict] | None = None,
                          predictor: Callable[..., float] | None = None) -> dict[str, Any]:
    """Run independent expanding folds, comparing all arms on common origins."""
    cropped = training_fit_only(source)
    if type(folds) is not int or not 3 <= folds <= 10:
        raise ValueError("folds must be between 3 and 10")
    if not _finite(initial_fit_fraction) or not .2 <= initial_fit_fraction <= .7:
        raise ValueError("initial_fit_fraction must be between .2 and .7")
    if type(purge_hours) is not int or not 24 <= purge_hours <= 168:
        raise ValueError("purge_hours must be between 24 and 168")
    specs = tuple(default_candidates() if candidates is None else candidates)
    names = [s.name for s in specs]
    if not 1 <= len(specs) <= 24 or len(set(names)) != len(names) or any(
        not isinstance(name, str) or not name or len(name) > 80 or name in _BASELINES for name in names
    ):
        raise ValueError("one to 24 uniquely named diagnostic candidates are required")
    configs = [spec.config() for spec in specs]
    common_history = max(config.history_steps for config in configs)
    run = fit_runner or prediction.fit_predict_exogenous_ridge
    initial_end = int(len(cropped.timestamps) * initial_fit_fraction)
    if initial_end < 48:
        raise ValueError("training_fit is too short for initial causal fitting")
    boundaries = [initial_end + (len(cropped.timestamps) - initial_end) * i // folds for i in range(folds + 1)]
    results = []
    for fold in range(folds):
        series = _fold_series(cropped, boundaries[fold], boundaries[fold + 1], purge_hours)
        profile = fit_baseline_profile(series, targets=TARGETS, horizons=HORIZONS)
        indexed, artifacts = {}, {}
        for spec, config in zip(specs, configs, strict=True):
            started = perf_counter()
            fitted = run(series, targets=TARGETS, horizons=HORIZONS, config=config,
                         evaluation_history_steps=common_history,
                         defer_prediction_partitions=("training_feedback",))
            fit_seconds = perf_counter() - started
            rows = [r for r in fitted["prediction_rows"] if r["partition"] == "training_feedback"]
            scored, audit = compare_host_execution(rows, fitted["models"], config, predictor=predictor)
            if not audit["prediction_identity_equal"] or not audit["mask_equal"]:
                raise ValueError("host execution changed the frozen numerical result")
            scored = apply_baseline_profile(series, scored, profile)
            indexed[spec.name] = {_case(row): row for row in scored}
            artifacts[spec.name] = {"fit_seconds": fit_seconds, "execution_comparison": audit,
                                    "fitted_models": len(fitted["models"]),
                                    "fit_rows_per_cell": [{"target": m["target"], "horizon_hours": m["horizon_hours"],
                                                           "n": m["training_rows"], "status": m["status"]}
                                                          for m in fitted["models"]]}
        origin_sets = [{key[0] for key in rows if key[1:] == (target, horizon)}
                       for rows in indexed.values() for target in TARGETS for horizon in HORIZONS]
        origins = sorted(set.intersection(*origin_sets))
        if not origins:
            raise ValueError(f"fold {fold + 1} has no common complete nine-cell origins; enlarge fit capacity or reduce folds/history")
        cases = [(o, t, h) for o in origins for t in TARGETS for h in HORIZONS]
        reference = indexed[specs[0].name]
        time_index = {series.timestamps[i]: i for p in series.partitions.values() for i in range(p.start, p.end)}
        baseline_rows = {name: [] for name in _BASELINES}
        seasonal_fallback = 0
        for key in cases:
            row = reference[key]
            persistence = float(series.values[row["target"]][time_index[row["origin_timestamp"]]])
            seasonal_index = time_index.get(row["timestamp"] - 24)
            seasonal_value = series.values[row["target"]][seasonal_index] if seasonal_index is not None else None
            if not _finite(seasonal_value):
                seasonal_value = persistence
                seasonal_fallback += 1
            for name, value in (("persistence", persistence), ("seasonal_24h", seasonal_value),
                                ("fit_selected_baseline", row["baseline"])):
                baseline_rows[name].append({**row, "predicted": value, "used_zero_residual_fallback": False})
        arms = []
        for name, rows in [(name, [values[key] for key in cases]) for name, values in indexed.items()] + list(baseline_rows.items()):
            metrics = _metrics(rows)
            arm = {"name": name, "cells": metrics, "mean_skill_score": fmean(m["skill_score"] for m in metrics),
                   "scoring_digest": _scoring_digest(rows)}
            if name in artifacts:
                arm.update(artifacts[name])
                arm["execution_comparison"]["scoring_equal"] = True
                arm["execution_comparison"]["scoring_basis"] = "identical predictions, masks, rows and frozen fit-selected baseline"
            arms.append(arm)
        results.append({"fold": fold + 1, "fit_end_exclusive": boundaries[fold],
                        "evaluation_range": series.partitions["training_feedback"].to_dict(),
                        "last_fit_label_time": series.timestamps[boundaries[fold] - 1],
                        "first_feedback_time": series.timestamps[series.partitions["training_feedback"].start],
                        "common_origin_count": len(origins), "common_scoring_cells": len(cases),
                        "origin_day_blocks": len({origin // 24 for origin in origins}),
                        "common_origin_digest": digest(origins), "case_identity_digest": digest(cases),
                        "first_origin": origins[0], "last_origin": origins[-1],
                        "baseline_profile_digest": profile["digest"],
                        "baseline_selection": [{"target": c["target"], "horizon_hours": c["horizon_hours"],
                                                "baseline_id": c["baseline_id"], "comparison_n": c["comparison_n"]}
                                               for c in profile["cells"]],
                        "seasonal_reference_missing_fallback_cells": seasonal_fallback,
                        "arms": arms})
    return {"schema_version": SCHEMA, "scope": "training_fit_internal_forward_folds",
            "not_independent_validation": True, "not_official_run_evaluation": True,
            "remote_requests": 0, "ledger_mutations": 0,
            "execution_value_claim": "host numerical equivalence only; real native LLM value has not been measured",
            "sealed_native_comparison": {"status": "not_measured",
                                          "reason": "internal diagnostic folds do not share the historical native fit snapshot or frozen evaluation identity"},
            "source": {"dataset_id": cropped.dataset_id, "episode_id": cropped.episode_id,
                       "training_fit_digest": cropped.digest, "fit_rows": len(cropped.timestamps),
                       "first_timestamp": cropped.timestamps[0], "last_timestamp": cropped.timestamps[-1],
                       "gap_count": sum(b - a != 1 for a, b in zip(cropped.timestamps, cropped.timestamps[1:]))},
            "configuration": {"folds": folds, "initial_fit_fraction": initial_fit_fraction,
                              "purge_hours": purge_hours, "common_history_steps": common_history,
                              "candidates": [spec.to_dict() for spec in specs],
                              "baseline_missing_policy": "seasonal reference missing: persistence fallback",
                              "selection_policy": "complete common nine-cell origins; no random splitting"},
            "folds": results}
