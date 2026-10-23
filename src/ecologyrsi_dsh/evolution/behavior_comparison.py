"""Reproducible, label-free diagnostics on already executed training cohorts.

Configuration hashes, execution differences and predictive quality are distinct.
Nothing in this module changes scores, selection, or the inference budget.
"""
import base64
import json
import math
import zlib
from collections.abc import Mapping
from itertools import combinations

from ..core.immutable import thaw_json
from ..core.models import canonical_json, digest
from ..core.screening import screening_cohort_digest
from . import exploration_policy

SCHEMA = "ecologyrsi-dsh.execution-behavior/1"
COMPARISON_SCHEMA = "ecologyrsi-dsh.generation-behavior-comparison/1"
MAX_BYTES = 1_000_000
MAX_CELLS = 10_000


def _number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def build_execution_behavior(task, records, rows, normalization_scales):
    if not exploration_policy.enabled(task.metadata):
        return None
    scope = task.metadata.get("_evaluation_scope") or {}
    if scope.get("phase") not in {"screening", "formal_batch", "training_feedback"}:
        return None
    from ..evaluators.epoch_cohorts import PlannedCohort
    from ..data.adapters import dataset_adapter
    from ..core.sample_results import _sample_agent_chain_attestation
    raw_cohort = task.metadata.get("_planned_evaluation_cohort")
    if not isinstance(raw_cohort, Mapping):
        return None
    cohort = PlannedCohort.from_dict(raw_cohort)
    if scope.get("cohort_digest") != cohort.cohort_digest:
        raise ValueError("execution behavior cohort mismatch")
    adapter = dataset_adapter(task.dataset)
    origins = {(o.origin_timestamp, o.reuse_index): o.origin_id for o in cohort.origins}
    expected = {(o.origin_id, o.reuse_index, target, horizon)
                for o in cohort.origins for target in adapter.target_names for horizon in adapter.horizons_hours}
    scales = {name: info["scale"] for name, info in normalization_scales.items()
              if isinstance(info, Mapping) and info.get("source_partition") == "training_fit"
              and _number(info.get("scale")) is not None and info["scale"] > 0}
    by_id = {r.get("sample_id"): r for r in records}
    vectors, seen = [], set()
    complete = set(scales) == set(adapter.target_names)
    for row in rows:
        source = origins.get((row.get("origin_timestamp"), row.get("cohort_origin_occurrence", 0)))
        key = (source, row.get("cohort_origin_occurrence", 0), row.get("target"), row.get("horizon_hours"))
        if key not in expected or key in seen:
            complete = False
            continue
        seen.add(key)
        record = by_id.get(row.get("sample_id"), {})
        accepted = next((a for a in reversed(record.get("attempt_trace", ()))
                         if a.get("outcome") == "accepted"), None)
        # Completed checkpoints retain a validated chain attestation and the
        # accepted method/tool summary, but not the full attempt trace. Use
        # that same durable projection before and after a restart.
        chain = _sample_agent_chain_attestation(row.get("sample_agent_chain"), sample_id=row.get("sample_id"))
        value = _number(row.get("predicted"))
        valid = ((accepted is not None or (chain and chain["complete"]))
                 and row.get("sample_execution_status") == "succeeded"
                 and not row.get("scoring_fallback") and value is not None)
        if not valid:
            complete = False
            continue
        info = row.get("agent_prediction") or {}
        used = [e for e in info.get("tools", ()) if e.get("status") == "completed" and e.get("used_as_evidence") is True]
        tool_values = [t.get("tool_predicted") for t in info.get("tools", ())
                       if t.get("tool_id") == "candidate-model" and t.get("status") == "completed"
                       and t.get("used_as_evidence") is True and _number(t.get("tool_predicted")) is not None]
        vectors.append({
            "key": list(key), "predicted": value,
            "tool_predicted": _number(tool_values[-1]) if tool_values else None,
            "actions": {
                "method": info.get("method"),
                "tool_path": [str(e.get("tool_id")) for e in used],
            },
        })
    vectors.sort(key=lambda item: tuple(item["key"]))
    body = {
        "schema_version": SCHEMA, "policy": exploration_policy.POLICY,
        "candidate_id": scope.get("candidate_id"), "revision_id": scope.get("candidate_revision_id"),
        "source_phase": scope.get("phase"), "cohort_digest": cohort.cohort_digest,
        "dataset_digest": task.metadata.get("dataset_digest"),
        "split_manifest_digest": task.metadata.get("split_manifest_digest"),
        "evaluator_digest": task.metadata.get("evaluator_digest"),
        "runtime_binding_digest": digest({k: thaw_json(task.metadata.get(k)) for k in (
            "sample_agent_protocol", "sample_planner_model_id", "sample_critic_model_id",
            "inference_policy", "sample_remote_critic_policy", "resolved_policy_route_config_digest",
            "resolved_review_route_config_digest", "preset_content_digest")}),
        "normalization_scales": scales, "expected_cell_count": len(expected),
        "action_projection": "accepted_method_and_cited_tool_sequence",
        "successful_cell_count": len(vectors),
        "status": "measured" if complete and seen == expected and len(vectors) == len(expected) else "unverified",
        "vectors": vectors,
    }
    return pack_behavior(body)


def pack_behavior(body):
    raw = canonical_json(body).encode()
    if len(raw) > MAX_BYTES or len(body.get("vectors", ())) > MAX_CELLS:
        raise ValueError("execution behavior exceeds bounded archive capacity")
    return {
        "schema_version": SCHEMA, "encoding": "zlib+base64",
        "projection_digest": digest(body), "uncompressed_bytes": len(raw),
        "status": body["status"], "expected_cell_count": body["expected_cell_count"],
        "successful_cell_count": body["successful_cell_count"],
        "payload": base64.b64encode(zlib.compress(raw)).decode(),
    }


def unpack_behavior(archive):
    if not isinstance(archive, Mapping) or archive.get("schema_version") != SCHEMA or archive.get("encoding") != "zlib+base64":
        raise ValueError("invalid execution behavior archive")
    if type(archive.get("uncompressed_bytes")) is not int or not 0 < archive["uncompressed_bytes"] <= MAX_BYTES:
        raise ValueError("invalid execution behavior archive size")
    payload = archive.get("payload")
    if not isinstance(payload, str) or len(payload) > 2 * MAX_BYTES:
        raise ValueError("invalid execution behavior payload size")
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(payload, validate=True), MAX_BYTES + 1)
        if not decoder.eof or decoder.unused_data or len(raw) != archive["uncompressed_bytes"]:
            raise ValueError("invalid execution behavior compressed length")
        body = json.loads(raw)
    except (TypeError, zlib.error, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid execution behavior encoding") from exc
    if (digest(body) != archive["projection_digest"] or body.get("schema_version") != SCHEMA
            or any(body.get(k) != archive.get(k) for k in ("status", "expected_cell_count", "successful_cell_count"))
            or len(body.get("vectors", ())) > MAX_CELLS):
        raise ValueError("execution behavior identity mismatch")
    return body


def _compare(left, right, threshold):
    for key in ("policy", "cohort_digest", "dataset_digest", "split_manifest_digest",
                "evaluator_digest", "runtime_binding_digest", "normalization_scales", "expected_cell_count"):
        if left.get(key) != right.get(key):
            return {"status": "unverified", "reason": f"{key}_mismatch"}
    if left["status"] != "measured" or right["status"] != "measured":
        return {"status": "unverified", "reason": "incomplete_execution"}
    before = {tuple(v["key"]): v for v in left["vectors"]}
    after = {tuple(v["key"]): v for v in right["vectors"]}
    if not before or before.keys() != after.keys() or len(before) != left["expected_cell_count"]:
        return {"status": "unverified", "reason": "origin_coverage_mismatch"}
    groups, action_changes = {}, 0
    for key, a in before.items():
        b = after[key]
        cell = f"{key[2]}@{key[3]}h"
        scale = left["normalization_scales"].get(key[2])
        if _number(scale) is None or scale <= 0 or any(_number(v["predicted"]) is None for v in (a, b)):
            raise ValueError("invalid execution behavior numeric evidence")
        group = groups.setdefault(cell, {"output": [], "tool": []})
        group["output"].append(abs(a["predicted"] - b["predicted"]) / scale)
        if a["tool_predicted"] is not None and b["tool_predicted"] is not None:
            group["tool"].append(abs(a["tool_predicted"] - b["tool_predicted"]) / scale)
        action_changes += a["actions"] != b["actions"]
    distances = {cell: {"prediction_distance": sum(g["output"]) / len(g["output"]),
                        "tool_distance": sum(g["tool"]) / len(g["tool"]) if len(g["tool"]) == len(g["output"]) else None}
                 for cell, g in sorted(groups.items())}
    return {
        "status": "measured", "distances": distances, "compared_cells": len(before),
        "near_outputs": all(v["prediction_distance"] <= threshold for v in distances.values()),
        "changed_action_cells": action_changes,
        "qualification": "same_cohort_observation_not_causal_attribution_or_scientific_gain",
    }


def compare_generation_behaviors(planned_cohort, screening_records, *, metadata):
    if not exploration_policy.enabled(metadata):
        raise ValueError("behavior comparison policy is not enabled")
    records = list(screening_records.values()) if isinstance(screening_records, Mapping) else list(screening_records)
    expected = dict(planned_cohort.revision_bindings)
    if len(records) != len(expected) or {r["candidate_id"] for r in records} != set(expected):
        raise ValueError("behavior comparison requires every frozen screening arm")
    profiles, members = {}, []
    generations = {r["generation"] for r in records}
    if generations != {planned_cohort.generation}:
        raise ValueError("behavior comparison generation mismatch")
    for record in sorted(records, key=lambda r: r["candidate_id"]):
        cid = record["candidate_id"]
        if record.get("candidate_revision_id") != expected[cid] or record["cohort_digest"] != planned_cohort.screening.cohort_digest:
            raise ValueError("behavior comparison revision or cohort mismatch")
        archive = record.get("metrics", {}).get("sample_execution", {}).get("execution_behavior")
        profile = unpack_behavior(archive) if archive is not None else None
        if profile is not None and (profile["candidate_id"] != cid or profile["revision_id"] != expected[cid]
                                   or profile["cohort_digest"] != record["cohort_digest"]
                                   or profile["source_phase"] != "screening"):
            raise ValueError("behavior projection belongs to another screening arm")
        profiles[cid] = profile
        members.append({"candidate_id": cid, "revision_id": expected[cid],
                        "screening_record_digest": record["record_digest"],
                        "projection_digest": archive["projection_digest"] if archive else None})
    pairs = []
    threshold = exploration_policy.contract(metadata)["near_output_threshold"]
    for left, right in combinations(sorted(profiles), 2):
        result = (_compare(profiles[left], profiles[right], threshold) if profiles[left] and profiles[right]
                  else {"status": "unverified", "reason": "behavior_not_recorded"})
        pairs.append({"left": left, "right": right, **result})
    # Complete-link grouping prevents A~B and B~C from falsely implying A~C.
    close = {frozenset((p["left"], p["right"])) for p in pairs if p.get("near_outputs") is True}
    groups = []
    for cid in sorted(profiles):
        group = next((g for g in groups if all(frozenset((cid, other)) in close for other in g)), None)
        if group is None:
            groups.append([cid])
        else:
            group.append(cid)
    body = {"schema_version": COMPARISON_SCHEMA, "policy": exploration_policy.POLICY,
            "generation": planned_cohort.generation, "cohort_digest": planned_cohort.screening.cohort_digest,
            "screening_digest": screening_cohort_digest(records), "members": members, "pairs": pairs,
            "groups": groups, "near_output_threshold": threshold,
            "selection_unchanged": True, "scientific_gain_established": False}
    return {**body, "comparison_digest": digest(body)}
