#!/usr/bin/env python3
"""Read-only replay of sealed native predictions with their original fitted model.

No refit, provider calls, evaluation events, or unopened validation/test labels.
The full Planner prompt is not archived; its wave identity is preserved, not
reconstructed. Agent final predictions and tool receipts are verified. Optional candidate-model
outputs can be independently rebuilt from their original fitted artifact.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sqlite3
from time import perf_counter
from typing import Any
from types import SimpleNamespace

from ecologyrsi_dsh.core.artifact_identity import validate_artifact_revision_binding
from ecologyrsi_dsh.core.models import ModelArtifact, digest
from ecologyrsi_dsh.core.sample_results import decode_sample_results
from ecologyrsi_dsh.core.trajectory import CandidateRevision, EvaluationScope
from ecologyrsi_dsh.data.registry import DatasetRegistry
from ecologyrsi_dsh.evaluators import greenhouse_prediction as ridge
from ecologyrsi_dsh.evaluators.epoch_cohorts import GenerationCohorts, RunAdaptationCohort
from ecologyrsi_dsh.evaluators.objectives import aggregate_greenhouse_objective, skill_score
from ecologyrsi_dsh.evolution.genome import EcologyEvolutionPluginGenome
from ecologyrsi_dsh.core.agent_prediction import validate_tool_event, validate_prediction_receipt


SCHEMA = "ecologyrsi-dsh.sealed-native-host-shadow/2"
CONFIGS = {
    ridge.EXOGENOUS_RIDGE_MODEL_ID: ridge.ExogenousRidgeConfig,
    ridge.TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID: ridge.TargetwiseExogenousRidgeConfig,
    ridge.HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID: ridge.HorizonTargetwiseExogenousRidgeConfig,
    ridge.BASELINE_ALIGNED_RIDGE_MODEL_ID: ridge.BaselineAlignedRidgeConfig,
}


def finite(value: Any) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def causal_context(source: Any, row: dict, model: dict, config: Any) -> tuple[float, dict]:
    """Read only origin-known predictors; imputation statistics come from the artifact."""
    origin, target, horizon = row["origin_timestamp"], row["target"], row["horizon_hours"]
    selected = source.partitions["training_feedback"]
    indices = {source.timestamps[index]: index for index in range(selected.start, selected.end)
               if source.timestamps[index] <= origin}
    if origin not in indices:
        raise ValueError("sealed origin is outside the visible causal source")
    history_times = [origin - lag for lag in range(config.history_steps)]
    history = []
    for timestamp in history_times:
        index = indices.get(timestamp)
        if index is None or not finite(source.values[target][index]):
            raise ValueError("sealed forecast history cannot be reconstructed causally")
        history.append(float(source.values[target][index]))
    baseline = history[0]
    if row["model_reference_baseline"] != baseline and model.get("prediction_model_id") != ridge.BASELINE_ALIGNED_RIDGE_MODEL_ID:
        raise ValueError("sealed model reference differs from the causal source baseline")
    snapshot = []
    for statistic in model["feature_statistics"]:
        name = statistic["name"]
        lag = re.fullmatch(rf"target:{re.escape(target)}:lag_(\d+)h", name)
        if lag:
            value = history[int(lag[1])]
        elif name.startswith("exogenous:"):
            feature = name.removeprefix("exogenous:")
            specification = source.features.get(feature)
            role = getattr(specification, "role", None)
            if role not in ridge._ALLOWED_EXOGENOUS_ROLES:
                raise ValueError("artifact exogenous feature is not in the causal registered policy")
            maximum_age = 6 if role in ridge._SHORT_FORWARD_FILL_ROLES else 168
            value = statistic["median"]
            for index in range(indices[origin], selected.start - 1, -1):
                if origin - source.timestamps[index] > maximum_age:
                    break
                candidate = source.values[feature][index]
                if finite(candidate):
                    value = float(candidate)
                    break
        elif name == "time:hour_sin":
            value = math.sin(2 * math.pi * (origin % 24) / 24)
        elif name == "time:hour_cos":
            value = math.cos(2 * math.pi * (origin % 24) / 24)
        else:
            raise ValueError("unsupported stored feature schema; an explicit replay adapter is required")
        if not finite(value):
            raise ValueError("non-finite causal feature")
        snapshot.append({"name": name, "value": value})
    context = {"feature_snapshot": snapshot, "history_window": history,
               "causal_provenance": {"origin_cutoff_timestamp": origin,
                                     "latest_context_timestamp": origin,
                                     "history_timestamps": history_times}}
    if isinstance(config, ridge.BaselineAlignedRidgeConfig):
        requested = model["requested_baseline_id"]
        source_time, resolved, fallback = origin, "persistence", None
        if requested == "seasonal_24h":
            seasonal_time = origin + horizon - 24
            seasonal_index = indices.get(seasonal_time)
            # The original aligned reference can also use visible training-fit history.
            if seasonal_index is None:
                fit = source.partitions["training_fit"]
                seasonal_index = next((i for i in range(fit.start, fit.end)
                                       if source.timestamps[i] == seasonal_time and seasonal_time <= origin), None)
            seasonal = source.values[target][seasonal_index] if seasonal_index is not None else None
            if finite(seasonal):
                baseline, source_time, resolved = float(seasonal), seasonal_time, "seasonal_24h"
            else:
                fallback = "seasonal_reference_missing"
        context["baseline_reference"] = {
            "schema_version": "ecologyrsi-dsh.ridge-baseline-reference/1",
            "target": target, "horizon_hours": horizon, "requested_baseline_id": requested,
            "baseline_id": resolved, "baseline_profile_digest": model["baseline_profile_digest"],
            "selection_partition": "training_fit", "value": baseline, "source_timestamp": source_time,
            "origin_timestamp": origin, "fallback_reason": fallback,
        }
        if baseline != row["model_reference_baseline"]:
            raise ValueError("aligned baseline does not reproduce its sealed model reference")
    return baseline, context


def validate_artifact(raw: dict) -> ModelArtifact:
    artifact = ModelArtifact.from_dict(raw)
    if artifact.digest != raw.get("artifact_digest") or artifact.training_partition != "training_fit":
        raise ValueError("fitted artifact identity or training partition is invalid")
    if artifact.model_id not in CONFIGS:
        raise ValueError("unsupported fitted replay model")
    models = artifact.learned_parameters.get("models", ())
    if len(models) != 9 or len({(model["target"], model["horizon_hours"]) for model in models}) != 9:
        raise ValueError("fitted artifact lacks the complete nine-cell model grid")
    for model in models:
        if model.get("fit_digest_sha256") != digest({key: value for key, value in model.items() if key != "fit_digest_sha256"}):
            raise ValueError("stored fitted model digest does not match")
    return artifact


def replay_vector(*, rows: list[dict], prepared: dict, artifact: ModelArtifact, event: dict) -> tuple[float, float]:
    """Measure only numerical calls, then verify the complete native tool receipt."""
    config = CONFIGS[artifact.model_id].from_mapping(artifact.parameters)
    models = artifact.learned_parameters["models"]
    outputs = {}
    start = perf_counter()
    for row in rows:
        baseline, context = prepared[row["sample_id"]]
        predicted = ridge.predict_fitted_exogenous_ridge(
            target=row["target"], horizon_hours=row["horizon_hours"], baseline=baseline,
            label_free_context=context, models=models, config=config)
        outputs[row["sample_id"]] = {"predicted": predicted, "metadata": {
            "source_model_id": artifact.model_id,
            "prediction_unit": "forecast_origin_with_target_horizon_vector",
            "origin_timestamp": row["origin_timestamp"],
        }}
    seconds = perf_counter() - start
    payload = event["payload"]
    validate_tool_event(payload)
    if payload["tool_id"] != "candidate-model":
        raise ValueError("default-model replay requires a candidate-model tool receipt")
    reproduced = {"tool_id": "candidate-model", "call_id": payload["call_id"],
                  "wave_digest": payload["wave_digest"], "status": "completed",
                  "outputs": [{"sample_id": sample_id, **outputs[sample_id]} for sample_id in payload["sample_ids"]]}
    # Wall time is recorded operational evidence, not a deterministic model output.
    # Verify its receipt, then preserve it while rebuilding only the numerical cells.
    elapsed_ms = payload["result"]["elapsed_ms"]
    if not finite(elapsed_ms) or elapsed_ms < 0:
        raise ValueError("recorded tool duration is invalid")
    reproduced["elapsed_ms"] = elapsed_ms
    if digest(reproduced) != payload["output_digest"]:
        raise ValueError("recorded numerical tool result no longer reproduces")
    # Final Agent predictions may differ: this utility verifies numerical tool
    # evidence only. The structured prediction receipt attests the final vector.
    recorded = {row["sample_id"]: row["predicted"] for row in payload["result"]["outputs"]}
    difference = max(abs(outputs[key]["predicted"] - recorded[key]) for key in outputs)
    return difference, seconds


def validate_revision_binding(artifact: ModelArtifact, raw_revision: dict, scope: EvaluationScope, *,
                              run_id: str, event_payload: dict, original_proposal_binding: dict) -> None:
    revision = CandidateRevision.from_dict(raw_revision)
    genome = EcologyEvolutionPluginGenome.from_dict(raw_revision["genome"])
    if (artifact.run_id != run_id or revision.run_id != run_id or scope.run_id != run_id
            or artifact.candidate_id != revision.candidate_id or scope.candidate_id != revision.candidate_id
            or artifact.candidate_revision_id != revision.revision_id or scope.candidate_revision_id != revision.revision_id
            or artifact.evaluation_scope_digest != scope.scope_key or revision.generation != scope.generation
            or revision.genome_digest != genome.genome_digest or revision.behavior_digest != genome.behavior_digest):
        raise ValueError("artifact, revision, native binding and evaluation scope do not identify the same candidate")
    if dict(genome.scientific_program["parameter_overrides"]) != dict(artifact.parameters) or genome.scientific_program["predictor_ref"]["id"] != artifact.model_id:
        raise ValueError("artifact does not match its frozen candidate revision parameters")
    schema = event_payload.get("schema_version")
    if schema == "ecologyrsi-dsh.artifact-recorded/2":
        if not original_proposal_binding or event_payload.get("proposal_identity_binding") != original_proposal_binding:
            raise ValueError("artifact proposal binding differs from its original CandidateSpawned identity")
        validate_artifact_revision_binding(event_payload.get("artifact_revision_binding"),
                                           artifact=artifact, revision=revision, scope=scope)
    elif schema in (None, "ecologyrsi-dsh.artifact-recorded/1"):
        binding = event_payload.get("identity_binding", {})
        # A legacy R0 envelope cannot attest an R1 artifact even if the numerical
        # model reproduces. Historical events are evidence, never repaired here.
        if binding != original_proposal_binding or binding.get("genome_digest") != genome.genome_digest:
            raise ValueError("artifact, revision, native binding and evaluation scope do not identify the same candidate")
    else:
        raise ValueError("unsupported artifact event envelope; an explicit replay adapter is required")


def validate_scored_cohort(source: Any, rows: list[dict], cohort: Any, scope: EvaluationScope) -> None:
    if (cohort.cohort_digest != scope.cohort_digest or len(cohort.origins) != scope.origin_count
            or any(origin.dataset_id != source.dataset_id or origin.episode_id != source.episode_id
                   or origin.origin_index < 0 or origin.origin_index >= len(source.timestamps)
                   or source.timestamps[origin.origin_index] != origin.origin_timestamp for origin in cohort.origins)):
        raise ValueError("frozen cohort differs from its scope or registered source episode")
    expected = {(origin.origin_timestamp, target, horizon) for origin in cohort.origins
                for target in ("air_temperature", "relative_humidity", "co2_concentration") for horizon in (1, 6, 24)}
    if (len(rows) != len(expected) or any(row["candidate_id"] != scope.candidate_id for row in rows)
            or {(row["origin_timestamp"], row["target"], row["horizon_hours"]) for row in rows} != expected):
        raise ValueError("sealed rows differ from their candidate or frozen cohort identities")


def native_usage(events: list[dict], keys: set[str]) -> dict:
    latest, launches, completions, reservations = {}, [], [], set()
    for event in events:
        payload = event["payload"]
        if event["kind"] == "DshSessionUsageRecorded" and payload["identity"]["idempotency_key"] in keys:
            latest[payload["identity"]["session_id"]] = payload
        if event["kind"] == "DshChildLaunchReserved" and payload["launch"].get("idempotency_key") in keys:
            launches.append(event["created_at"])
            reservations.add(payload["launch"]["reservation_id"])
        if event["kind"] == "DshStructuredResultAccepted" and payload["identity"].get("idempotency_key") in keys:
            completions.append(event["created_at"])
    totals, complete = defaultdict(int), True
    for payload in latest.values():
        usage = payload["session_metrics"].get("provider_usage", {})
        measured = usage.get("totals", {})
        required = {"total_tokens", "uncached_input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"}
        if (not payload.get("usage_complete") or not usage.get("available")
                or not isinstance(measured, dict) or not required.issubset(measured)
                or any(not finite(value) or value < 0 for value in measured.values())):
            complete = False
        if isinstance(measured, dict) and usage.get("available"):
            for key, value in measured.items():
                if finite(value) and value >= 0:
                    totals[key] += value
    elapsed = (datetime.fromisoformat(max(completions)) - datetime.fromisoformat(min(launches))).total_seconds() if launches and completions else None
    missing = reservations - {payload["identity"]["child_reservation_id"] for payload in latest.values()}
    return {"sessions_with_usage": len(latest), "usage_complete": complete and bool(latest) and not missing,
            "launches_without_usage": len(missing),
            "reported_tokens": dict(totals) if totals else None,
            "launches": len(launches), "accepted_results": len(completions),
            "launch_to_last_acceptance_window_seconds": elapsed,
            "timing_scope": "shared_live_wall_window_including_queue_pauses_and_overlap_not_model_compute_time"}


def verify_scoring_inputs(source: Any, rows: list[dict], evaluation: dict) -> dict:
    """Recheck already-sealed scoring labels against origin-known baselines."""
    metrics = evaluation["metrics"]
    profile = metrics["baseline_profile"]
    if profile["digest"] != digest({key: value for key, value in profile.items() if key != "digest"}):
        raise ValueError("sealed baseline profile digest is invalid")
    visible = {source.timestamps[index]: index for partition in ("training_fit", "training_feedback")
               for index in range(source.partitions[partition].start, source.partitions[partition].end)}
    groups = defaultdict(list)
    for row in rows:
        origin, horizon, target = row["origin_timestamp"], row["horizon_hours"], row["target"]
        if row["target_timestamp"] != origin + horizon or row["baseline_profile_digest"] != profile["digest"]:
            raise ValueError("sealed scoring time or baseline identity is invalid")
        baseline_time = origin if row["baseline_id"] == "persistence" else origin + horizon - 24
        if baseline_time > origin or baseline_time not in visible or row["baseline"] != source.values[target][visible[baseline_time]]:
            raise ValueError("sealed scoring baseline differs from the origin-known source")
        if row["normalization_scale"] != metrics["normalization_scales"][target]["scale"]:
            raise ValueError("sealed scoring normalization differs from its evaluation")
        groups[target, horizon].append(row)
    target_metrics = {(item["target"], item["horizon_hours"]): item for item in metrics["targets"]}
    recomputed = []
    for (target, horizon), selected in groups.items():
        count = len(selected)
        candidate_rmse = math.sqrt(sum((row["predicted"] - row["observed"]) ** 2 for row in selected) / count)
        baseline_rmse = math.sqrt(sum((row["baseline"] - row["observed"]) ** 2 for row in selected) / count)
        original = target_metrics[target, horizon]
        if count != original["n"] or not math.isclose(candidate_rmse, original["rmse"], rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(baseline_rmse, original["baseline_rmse"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("sealed per-cell score does not reproduce from its archived rows")
        recomputed.append({"target": target, "horizon_hours": horizon, "n": count, "eligible_rows": count,
                           "skill_score": skill_score(candidate_rmse, baseline_rmse), "objective_quality": 1.,
                           "normalized_mean_reward": sum(row["normalized_reward"] for row in selected) / count})
    score = aggregate_greenhouse_objective(recomputed, metrics["objective_horizons"],
                                          target_weights=metrics["objective_target_weights"])["weighted_skill_score"]
    if not math.isclose(score, evaluation["score"], rel_tol=0., abs_tol=1e-12):
        raise ValueError("sealed aggregate score does not reproduce")
    return {"sealed_score": evaluation["score"], "recomputed_score": score,
            "absolute_score_difference": abs(score - evaluation["score"]),
            "baseline_and_normalization_inputs_verified": True,
            "observations_source": "already_sealed_scored_rows_only"}


def replay_run(database: Path, run_id: str, data_root: Path, candidate_id: str | None = None) -> dict:
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        events = [{"seq": seq, "event_id": eid, "kind": kind, "created_at": created, "payload": json.loads(payload)}
                  for seq, eid, kind, created, payload in connection.execute(
                      "SELECT seq,event_id,kind,created_at,payload_json FROM evolution_events WHERE run_id=? ORDER BY seq", (run_id,))]
    created = next(event for event in events if event["kind"] == "RunCreated")["payload"]["task_manifest"]
    metadata = created["metadata"]
    if metadata.get("execution_protocol") != "dsh_native_plugin_evolution@1":
        raise ValueError("replay requires an actual native execution run")
    source = DatasetRegistry(data_root=data_root).selection_view(
        created["visible_datasets"][0], metadata["episode_id"], expected_dataset_digest=metadata["dataset_digest"],
        expected_split_manifest_digest=metadata["split_manifest_digest"], expected_data_protocol_digest=metadata["data_protocol_digest"])
    results, used_keys = [], set()
    for event in events:
        if event["kind"] != "ArtifactRecorded":
            continue
        artifact = validate_artifact(event["payload"]["artifact"])
        if candidate_id is not None and artifact.candidate_id != candidate_id:
            continue
        if artifact.dataset_digest != source.digest:
            raise ValueError("fitted artifact and source dataset differ")
        revision = next(e["payload"]["revision"] for e in events if e["kind"] == "CandidateRevisionCreated"
                        and e["payload"]["revision"]["revision_id"] == artifact.candidate_revision_id)
        evaluation = next(e["payload"]["evaluation"] for e in events if e["kind"] in {"HoldoutEvaluationRecorded", "FormalBatchEvaluated"}
                          and e["payload"]["evaluation"]["scope"]["candidate_id"] == artifact.candidate_id
                          and EvaluationScope.from_dict(e["payload"]["evaluation"]["scope"]).scope_key == artifact.evaluation_scope_digest)
        scope = EvaluationScope.from_dict(evaluation["scope"])
        spawned = next(e["payload"] for e in events if e["kind"] == "CandidateSpawned"
                       and e["payload"]["candidate"]["candidate_id"] == artifact.candidate_id)
        validate_revision_binding(artifact, revision, scope, run_id=run_id,
                                  event_payload=event["payload"], original_proposal_binding=spawned["identity_binding"])
        if (evaluation["metrics"]["dataset_digest"] != source.digest
                or evaluation["metrics"]["split_manifest_digest_sha256"] != source.split_manifest_digest_sha256
                or evaluation["evaluator_digest"] != metadata["evaluator_digest"]):
            raise ValueError("sealed evaluation uses a different frozen data or scoring configuration")
        if scope.phase.value not in {"holdout", "formal_batch", "screening"}:
            raise ValueError("unopened validation or final test cannot be replayed")
        started = next(e for e in events if e["kind"] == "EvaluationSampleResultsStarted"
                       and e["payload"]["checkpoint"].get("execution_scope_digest") == scope.scope_key)
        sealed = next(e for e in events if e["kind"] == "EvaluationSampleResultsRecorded"
                      and e["payload"]["revision"] == started["payload"]["revision"])
        rows = list(decode_sample_results(sealed["payload"]))
        checkpoint = started["payload"]["checkpoint"]
        if (len(rows) != scope.origin_count * 9 or checkpoint["cohort_digest"] != scope.cohort_digest
                or checkpoint["candidate_revision_id"] != artifact.candidate_revision_id
                or sealed["payload"]["candidate_id"] != artifact.candidate_id
                or sealed["payload"]["run_id"] != run_id
                or started["payload"]["candidate_id"] != artifact.candidate_id):
            raise ValueError("sealed results do not cover their frozen checkpoint")
        cohorts = []
        for recorded in events:
            if recorded["kind"] == "GenerationCohortsFrozen":
                plan = GenerationCohorts.from_dict(recorded["payload"]["generation_cohorts"])
                cohorts.extend((plan.screening, plan.holdout))
            elif recorded["kind"] == "RunAdaptationCohortFrozen":
                plan = RunAdaptationCohort.from_dict(recorded["payload"]["adaptation"])
                cohorts.extend(batch.cohort for batch in plan.batches)
        cohort = next(item for item in cohorts if item.cohort_digest == scope.cohort_digest)
        validate_scored_cohort(source, rows, cohort, scope)
        if any(row["status"] != "succeeded" or row["scoring_fallback"] or not row["sample_agent_chain"]["complete"] for row in rows):
            raise ValueError("this replay requires complete successful native chains")
        scoring_check = verify_scoring_inputs(source, rows, evaluation)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["origin_timestamp"]].append(row)
        models = {(m["target"], m["horizon_hours"]): dict(m) for m in artifact.learned_parameters["models"]}
        config = CONFIGS[artifact.model_id].from_mapping(artifact.parameters)
        start = perf_counter()
        prepared = {row["sample_id"]: causal_context(source, row, models[row["target"], row["horizon_hours"]], config) for row in rows}
        construction_seconds = perf_counter() - start
        numerical_seconds, max_difference, receipts = 0., 0., []
        reproduced_model_calls = 0
        event_lookup = {e["event_id"]: SimpleNamespace(**e) for e in events}
        for vector in grouped.values():
            ids = {row["sample_id"] for row in vector}
            accepted = [e for e in events if e["kind"] == "DshStructuredResultAccepted"
                        and started["seq"] < e["seq"] < sealed["seq"]
                        and set(e["payload"].get("required_tool_receipt", {}).get("sample_ids", [])) == ids]
            if len(vector) != 9 or not accepted:
                raise ValueError("sealed origin has no complete Agent prediction receipt")
            final_event = accepted[-1]
            final = final_event["payload"]
            validate_prediction_receipt(final["structured"], final["required_tool_receipt"],
                                        event_lookup=event_lookup.get, identity=final["identity"],
                                        before_seq=final_event["seq"])
            predicted = {row["sample_id"]: row["predicted"] for row in final["structured"]["decisions"]}
            if any(row["predicted"] != predicted[row["sample_id"]] for row in vector):
                raise ValueError("sealed prediction differs from the Agent final submission")
            used_keys.add(final["identity"]["idempotency_key"])
            for call in final["required_tool_receipt"]["calls"]:
                tool_event = event_lookup[call["event_id"]]
                receipts.append(tool_event.payload["output_digest"])
                if (tool_event.payload["tool_id"] == "candidate-model"
                        and tool_event.payload["result"]["status"] == "completed"):
                    error, seconds = replay_vector(rows=vector, prepared=prepared, artifact=artifact,
                                                   event=vars(tool_event))
                    max_difference = max(max_difference, error)
                    numerical_seconds += seconds
                    reproduced_model_calls += 1
        results.append({"candidate_id": artifact.candidate_id, "candidate_revision_id": artifact.candidate_revision_id,
                        "artifact_digest": artifact.digest, "parameters_digest": digest(artifact.parameters),
                        "candidate_revision_and_artifact_identity_verified": True,
                        "fitted_models_digest": digest(artifact.learned_parameters["models"]),
                        "evaluation_scope": scope.to_dict(), "sealed_results_digest": sealed["payload"]["result_digest"],
                        "reconstructed_causal_inputs_digest": digest(prepared), "native_tool_outputs_digest": digest(receipts),
                        "origins": len(grouped), "prediction_cells": len(rows), "verified_tool_receipts": len(receipts),
                        "reproduced_candidate_model_calls": reproduced_model_calls,
                        "maximum_absolute_candidate_model_difference": max_difference if reproduced_model_calls else None,
                        "context_construction_seconds": construction_seconds, "host_numerical_seconds": numerical_seconds,
                        "host_timing_scope": "prediction_calls_and_output_assembly_excluding_context_construction_receipt_verification_and_loading",
                        "scoring_check": scoring_check,
                        "status": "agent_predictions_and_scoring_verified"})
    if not results:
        raise ValueError("no complete fitted artifact exists for the selected candidate")
    return {"schema_version": SCHEMA, "run_id": run_id, "dataset_digest": source.digest,
            "source_event_digest": digest([{key: e[key] for key in ("seq", "event_id", "kind", "payload")} for e in events]),
            "candidates": results, "native_execution": native_usage(events, used_keys),
            "refits": 0, "remote_calls": 0, "ledger_writes": 0,
            "full_planner_input_wave_reconstructed": False,
            "limits": ["Planner full prompt/context was not archived; original wave digest is preserved, not recomputed.",
                       "Agent final values are verified against persisted submissions. Only candidate-model tool calls are numerically reproduced; other model evidence is verified by its persisted receipts.",
                       "Host pure numerical time cannot be divided into the shared native wall window to claim a speedup or token saving.",
                       "Only already scored adaptive-search rows are compared; this is not independent validation or new promotion evidence."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = replay_run(arguments.database, arguments.run_id, arguments.data_root, arguments.candidate_id)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(arguments.output), "candidates": len(result["candidates"]),
                      "prediction_cells": sum(item["prediction_cells"] for item in result["candidates"]),
                      "status": "verified"}))


if __name__ == "__main__":
    main()
