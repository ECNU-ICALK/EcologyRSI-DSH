"""A preregistered A/A diagnostic, separate from candidate selection scores."""
from collections.abc import Mapping
import base64
import json
import math
import zlib

from ..core.immutable import thaw_json
from ..core.models import CandidateRole, digest
from ..core.trajectory import EvaluationScope, EvaluationPhase

POLICY_KEY = "inference_repeatability_policy"
POLICY = "screening_incumbent_repeat@1"
EVENT = "ScreeningRepeatabilityRecorded"


def enabled(metadata):
    value = metadata.get(POLICY_KEY)
    if value not in (None, POLICY):
        raise ValueError("unsupported inference repeatability policy")
    return value == POLICY


def control_binding(state, generation):
    planned = state.generation_cohort_for(generation)
    if planned is None or not planned.revision_bindings or len(planned.revision_bindings) != 5:
        raise ValueError("repeatability requires a frozen five-arm race")
    current = {c.candidate_id for c in state.candidates
               if c.generation == generation and c.role is CandidateRole.SEARCH}
    controls = [(cid, rid) for cid, rid in planned.revision_bindings.items() if cid not in current]
    if len(controls) != 1:
        raise ValueError("repeatability requires exactly one incumbent control")
    return controls[0]


def sealed_reference(state, generation):
    cid, rid = control_binding(state, generation)
    planned = state.generation_cohort_for(generation)
    for candidate_id, revision_id in planned.revision_bindings.items():
        event = state.screening_for(generation, candidate_id)
        if (event is None or event.payload.get("candidate_revision_id") != revision_id
                or event.payload.get("cohort_digest") != planned.screening.cohort_digest
                or event.payload.get("origin_count") != planned.screening.origin_count):
            raise ValueError("repeatability requires all five original screening arms sealed")
    return cid, rid, state.screening_for(generation, cid)


def _predictions(metrics):
    archive = metrics.get("sample_execution_trace_archive")
    if not isinstance(archive, Mapping) or archive.get("encoding") != "zlib+base64+canonical-json":
        raise ValueError("repeatability requires sealed prediction traces")
    records = json.loads(zlib.decompress(base64.b64decode(archive["payload"], validate=True)))
    if not isinstance(records, list) or not records:
        raise ValueError("repeatability prediction traces must be nonempty")
    if digest(records) != archive.get("trace_digest") or len(records) != archive.get("record_count"):
        raise ValueError("repeatability trace integrity mismatch")
    result = {}
    for row in records:
        value = row.get("predicted")
        if row.get("status") != "succeeded" or type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("repeatability requires complete successful predictions")
        if row["sample_id"] in result:
            raise ValueError("repeatability trace has duplicate samples")
        result[row["sample_id"]] = value
    return result


def summarize(reference, metrics, score, scope):
    """One repeat estimates realized drift; it is not a confidence interval."""
    base = reference["metrics"]
    for key in ("dataset_digest", "split_manifest_digest_sha256", "evaluation_index_digest",
                "baseline_profile_digest", "baseline_metrics_digest", "feedback_update_cohort_digest",
                "screening_evaluator_digest", "prediction_model_id", "objective_aggregation_version",
                "objective_target_weights", "objective_horizons", "comparison_retrieval_snapshot_digest",
                "prediction_formula_policy"):
        if base.get(key) != metrics.get(key):
            raise ValueError("repeatability comparison inputs differ: " + key)
    left, right = _predictions(base), _predictions(metrics)
    if left.keys() != right.keys():
        raise ValueError("repeatability samples differ")
    def cells(value):
        rows = value["targets"]
        if not rows or any(type(r.get("skill_score")) not in (int, float)
                           or not math.isfinite(r["skill_score"]) for r in rows):
            raise ValueError("repeatability target cells must have finite skills")
        result = {(r["target"], r["horizon_hours"]): float(r["skill_score"]) for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate repeatability target cells")
        return result
    old, new = cells(base), cells(metrics)
    if old.keys() != new.keys():
        raise ValueError("repeatability target grid differs")
    return {
        "kind": "same_genome_same_cohort_independent_inference",
        "reference_score": reference["score"], "repeat_score": score,
        "score_delta": score - reference["score"],
        "absolute_score_drift": abs(score - reference["score"]),
        "paired_origin_count": scope.origin_count,
        "paired_prediction_count": len(left),
        "changed_prediction_count": sum(not math.isclose(left[k], right[k], rel_tol=0, abs_tol=1e-9) for k in left),
        "cells": [{"target": t, "horizon_hours": h, "reference_skill": old[t,h],
                   "repeat_skill": new[t,h], "skill_delta": new[t,h]-old[t,h]} for t,h in sorted(old)],
        "independent_observation_count_multiplier": 1,
        "selection_score_replaced": False,
        "interpretation": "One A/A pair measures realized Agent drift, not a confidence bound or a causal attribution; do not select the better repeat.",
    }


def build_payload(state, scope, metrics, score):
    if not enabled(state.task_manifest.metadata):
        raise ValueError("repeatability policy is not frozen")
    cid, rid, reference = sealed_reference(state, scope.generation)
    if (scope.run_id != state.run.run_id or scope.phase is not EvaluationPhase.SCREENING
            or scope.inference_replica != 1 or scope.candidate_id != cid or scope.candidate_revision_id != rid):
        raise ValueError("repeatability scope must repeat the sealed incumbent")
    if (scope.cohort_digest != reference.payload["cohort_digest"]
            or scope.origin_count != reference.payload["origin_count"]):
        raise ValueError("repeatability cohort mismatch")
    metrics = thaw_json(metrics)
    if type(score) not in (int, float) or not math.isfinite(score):
        raise ValueError("invalid repeatability score")
    for value, actual_score in ((metrics, float(score)), (reference.payload["metrics"], reference.payload["score"])):
        objective = value.get("objective_score")
        if type(objective) not in (int, float) or not math.isfinite(objective) or not math.isclose(
                objective, actual_score, rel_tol=0, abs_tol=1e-12):
            raise ValueError("repeatability score must match its sealed objective")
        expected = scope.origin_count * state.task_manifest.metadata["prediction_cells_per_origin"]
        if len(_predictions(value)) != expected or reference.payload["prediction_cell_count"] != expected:
            raise ValueError("repeatability requires every frozen prediction cell")
        from .execution_validity import require_valid_execution
        require_valid_execution(value, scope.origin_count, phase="screening_repeatability",
                                metadata=state.task_manifest.metadata)
    body = {"schema_version": "ecologyrsi-dsh.screening-repeatability/1",
            "generation": scope.generation, "candidate_id": cid, "scope": scope.to_dict(),
            "reference_record_digest": reference.payload["record_digest"],
            "score": float(score), "metrics": metrics,
            "summary": summarize(reference.payload, metrics, float(score), scope)}
    return {**body, "evidence_digest": digest(body)}


def validate_payload(state, value):
    expected = build_payload(state, EvaluationScope.from_dict(value["scope"]), value["metrics"], value["score"])
    if thaw_json(value) != expected:
        raise ValueError("repeatability evidence does not match sealed sources")


def summary_for(state, generation):
    if not enabled(state.task_manifest.metadata):
        return None
    found = [e.payload["summary"] for e in state.events
             if e.kind == EVENT and e.payload["generation"] == generation]
    return thaw_json(found[0]) if found else None
