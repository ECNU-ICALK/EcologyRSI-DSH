"""Generate the ecogym parity fixtures from the v0.10.33 implementation.

This script is provenance, not a test. It imports the retired
``ecologyrsi_dsh`` package, so run it from a checkout of tag ``v0.10.33``
(the package is deleted later on the ``ecogym`` branch)::

    git worktree add /tmp/ecologyrsi-v0.10.33 v0.10.33
    PYTHONPATH=/tmp/ecologyrsi-v0.10.33/src python tests/ecogym/golden/make_golden.py

It reads only development partitions (calibration_fit, calibration_uq and
model_selection). Validation and final_test contribute their boundaries, never
their values.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

from ecologyrsi_dsh.data.adapters import dataset_adapter
from ecologyrsi_dsh.data.registry import DatasetRegistry
from ecologyrsi_dsh.evaluators.agent_model_tools import _CONFIGS, _defaults
from ecologyrsi_dsh.evaluators.baselines import fit_baseline_profile
from ecologyrsi_dsh.evaluators.metrics import _normalization_scale
from ecologyrsi_dsh.evaluators.objectives import skill_score
from ecologyrsi_dsh.evaluators.pipeline import _apply_scoring_baseline

from ecologyrsi_dsh.evaluators import greenhouse_prediction as gp

OUT = Path(__file__).resolve().parent
DATASET = "agc_cucumber_2018"
EPISODE = "agc_cucumber_2018:AiCU"
HISTORY = 48
SAMPLE_EVERY = 57


def sha(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def column_digest(values) -> str:
    """Digest of one column as JSON floats (None for missing)."""

    return sha([None if value is None else float(value) for value in values])


def dataset_section(registry: DatasetRegistry) -> dict:
    descriptor = registry._descriptor(DATASET)
    canonical = registry._load_series(descriptor)
    manifest = registry._split_manifest(descriptor, canonical)
    episodes = []
    for episode in canonical.episodes:
        split = manifest.split_for(episode.episode_id)
        episodes.append(
            {
                "episode_id": episode.episode_id,
                "role": split.role,
                "row_count": episode.row_count,
                "first_timestamp": episode.timestamps[0],
                "last_timestamp": episode.timestamps[-1],
            }
        )
    aicu = next(item for item in canonical.episodes if item.episode_id == EPISODE)
    timestamps = list(aicu.timestamps)
    gaps = [
        {"after_row": index, "from": left, "to": right}
        for index, (left, right) in enumerate(pairwise(timestamps))
        if right - left != 1
    ]
    columns = {}
    for name, values in sorted(aicu.values.items()):
        finite = [float(value) for value in values if value is not None]
        columns[name] = {
            "finite": len(finite),
            "fsum": math.fsum(finite),
            "digest": column_digest(values),
        }
    return {
        "episodes": episodes,
        "aicu": {
            "episode_id": EPISODE,
            "row_count": len(timestamps),
            "first_timestamp": timestamps[0],
            "last_timestamp": timestamps[-1],
            "timestamps_digest": sha(timestamps),
            "gaps": gaps,
            "content_sha256": aicu.content_sha256,
            "columns": columns,
        },
    }


def protocol_section(registry: DatasetRegistry) -> dict:
    protocol = registry.data_protocol(DATASET, EPISODE)
    stages = {}
    for name in (
        "calibration_fit",
        "calibration_uq",
        "model_selection",
        "validation",
        "final_test",
    ):
        selected = protocol.range_for(name)
        start, end = protocol.partition_timestamp_bounds[name]
        stages[name] = {
            "row_start": selected.start,
            "row_end": selected.end,
            "rows": selected.size,
            "time_start": start,
            "time_end": end,
        }
    return {"stages": stages}


def selection_view():
    adapter = dataset_adapter(DATASET)
    view = DatasetRegistry().selection_view(
        DATASET,
        EPISODE,
        target_names=adapter.target_names,
        horizons=adapter.horizons_hours,
        history_steps=3,
    )
    return adapter, view


def cohort_origins(series, partition, targets, horizons) -> list[int]:
    """Origins whose every cell has a 48-step history and a label in ``partition``."""

    selected = series.partitions[partition]
    common = None
    for target in targets:
        for horizon in horizons:
            samples = gp._base_samples(series, target, selected, horizon, HISTORY)
            stamps = {series.timestamps[sample.origin_index] for sample in samples}
            common = stamps if common is None else common & stamps
    return sorted(common or ())


def recipe_seed(horizons) -> dict:
    values = _defaults(_CONFIGS["greenhouse-recipe-ridge@1"], horizons)
    recipe = json.loads(json.dumps(values["feature_recipe"]))
    recipe["per_horizon"] = {str(h): {"residual_scale": 0.5} for h in horizons}
    recipe.pop("per_cell", None)
    return recipe


def evaluate(series, adapter, config, targets, horizons, partition_name):
    """Fit on calibration_fit, predict ``partition_name``; return per-cell evidence."""

    view = replace(
        series,
        partitions={**series.partitions, "training_feedback": series.partitions[partition_name]},
    )
    result = gp.fit_predict_exogenous_ridge(
        view,
        targets=targets,
        horizons=horizons,
        config=config,
        evaluation_history_steps=HISTORY,
        prediction_partitions=("training_feedback",),
    )
    rows = [row for row in result["prediction_rows"] if row["partition"] == "training_feedback"]
    profile = fit_baseline_profile(view, targets=targets, horizons=horizons)
    fit = view.partitions["training_fit"]
    scales = {
        target: _normalization_scale(tuple(view.values[target][fit.start : fit.end]))[0]
        for target in targets
    }
    rows = _apply_scoring_baseline(view, rows, profile, scales)
    table = {}
    for row in rows:
        values = (row.get("observed"), row.get("baseline"), row.get("predicted"))
        if all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            table[(row["origin_timestamp"], row["target"], row["horizon_hours"])] = tuple(
                float(value) for value in values
            )
    counts: dict[int, int] = {}
    for origin, _target, _horizon in table:
        counts[origin] = counts.get(origin, 0) + 1
    complete = sorted(origin for origin, count in counts.items() if count == len(targets) * len(horizons))
    keep = set(complete)
    cells = {}
    skills = []
    for target in targets:
        for horizon in horizons:
            items = sorted(
                (origin, values)
                for (origin, cell_target, cell_horizon), values in table.items()
                if origin in keep and cell_target == target and cell_horizon == horizon
            )
            observed = [values[0] for _, values in items]
            baseline = [values[1] for _, values in items]
            predicted = [values[2] for _, values in items]
            n = len(items)
            model_sse = math.fsum((p - o) ** 2 for p, o in zip(predicted, observed, strict=True))
            base_sse = math.fsum((b - o) ** 2 for b, o in zip(baseline, observed, strict=True))
            model_rmse = math.sqrt(model_sse / n)
            base_rmse = math.sqrt(base_sse / n)
            skill = skill_score(model_rmse, base_rmse)
            skills.append(skill)
            cells[f"{target}@{horizon}"] = {
                "n": n,
                "baseline_rmse": base_rmse,
                "model_rmse": model_rmse,
                "skill": skill,
                "sum_predicted": math.fsum(predicted),
                "sum_baseline": math.fsum(baseline),
                "sum_observed": math.fsum(observed),
                "samples": [
                    {"origin": origin, "observed": values[0], "baseline": values[1], "predicted": values[2]}
                    for origin, values in items[::SAMPLE_EVERY]
                ],
            }
    models = []
    for model in result["models"]:
        entry = {
            "cell": f"{model['target']}@{model['horizon_hours']}",
            "status": model["status"],
            "training_rows": model["training_rows"],
            "feature_names": model["feature_names"],
            "intercept": model["intercept"],
            "coefficients": model["coefficients"],
        }
        audit = model.get("diurnal_profile_ensemble")
        if audit:
            entry["constrained_member"] = audit["constrained_member"]
            entry["anchor_feature"] = audit["anchor_feature"]
        models.append(entry)
    return {
        "origins": len(complete),
        "origins_digest": sha(complete),
        "overall_skill": sum(skills) / len(skills),
        "cells": cells,
        "models": models,
        "normalization_scales": scales,
    }


def data_only(dataset_id: str, episode_id: str) -> None:
    """Loader parity for a second dataset: episodes and every column of one episode."""

    registry = DatasetRegistry()
    descriptor = registry._descriptor(dataset_id)
    canonical = registry._load_series(descriptor)
    episode = next(item for item in canonical.episodes if item.episode_id == episode_id)
    golden = {
        "schema": f"ecogym.golden.{dataset_id}/1",
        "source": "ecologyrsi-dsh v0.10.33",
        "episodes": [
            {"episode_id": item.episode_id, "row_count": item.row_count,
             "first_timestamp": item.timestamps[0], "last_timestamp": item.timestamps[-1]}
            for item in canonical.episodes
        ],
        "episode": {
            "episode_id": episode_id,
            "timestamps_digest": sha(list(episode.timestamps)),
            "columns": {
                name: {"finite": sum(value is not None for value in values), "digest": column_digest(values)}
                for name, values in sorted(episode.values.items())
            },
        },
    }
    path = OUT / f"{dataset_id}.json"
    path.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)


def main() -> None:
    registry = DatasetRegistry()
    adapter, view = selection_view()
    targets = tuple(item[0] for item in adapter.target_bounds)
    horizons = tuple(adapter.horizons_hours)
    profile = fit_baseline_profile(view, targets=targets, horizons=horizons)
    recipe = recipe_seed(horizons)
    configs = {
        "expert": gp.DiurnalProfileEnsembleConfig.defaults(),
        "recipe_seed": gp.RecipeRidgeConfig.from_mapping({"feature_recipe": recipe}),
    }
    golden = {
        "schema": "ecogym.golden.agc_cucumber_2018/1",
        "source": "ecologyrsi-dsh v0.10.33",
        "dataset": dataset_section(registry),
        "protocol": protocol_section(registry),
        "baselines": [
            {
                "cell": f"{cell['target']}@{cell['horizon_hours']}",
                "kind": cell["baseline_id"],
                "comparison_n": cell["comparison_n"],
                "persistence_rmse": cell["persistence_rmse"],
                "seasonal_rmse": cell["seasonal_24h_rmse"],
            }
            for cell in profile["cells"]
        ],
        "origins": {
            name: {
                "count": len(stamps),
                "digest": sha(stamps),
                "first": stamps[0],
                "last": stamps[-1],
            }
            for name, stamps in (
                ("calibration_fit", cohort_origins(view, "training_fit", targets, horizons)),
                ("calibration_uq", cohort_origins(view, "calibration_uq", targets, horizons)),
                ("model_selection", cohort_origins(view, "training_feedback", targets, horizons)),
            )
        },
        "recipe_seed": recipe,
        "results": {
            label: {
                period: evaluate(view, adapter, config, targets, horizons, partition)
                for period, partition in (
                    ("model_selection", "training_feedback"),
                    ("calibration_uq", "calibration_uq"),
                )
            }
            for label, config in configs.items()
        },
    }
    path = OUT / "agc_cucumber_2018.json"
    path.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for label, periods in golden["results"].items():
        for period, body in periods.items():
            print(f"{label:12s} {period:16s} origins={body['origins']} overall={body['overall_skill']:+.4f}")
    print("origins", {name: item["count"] for name, item in golden["origins"].items()})
    print("wrote", path)


if __name__ == "__main__":
    main()
    data_only("agc_tomato_2019", "agc_tomato_2019:AICU")
