"""Training-only, origin-weighted diagnostics of candidate tools and Agent outputs.

These are paired observations from one inference, not causal effects. Successful
retries cannot inflate the population, and absent/ambiguous receipts stay unknown.
"""
from collections import defaultdict
from collections.abc import Mapping
from math import isclose, isfinite, sqrt
from statistics import fmean

from ..core.immutable import thaw_json
from ..core.models import digest

POLICY_KEY = "candidate_attribution_policy"
POLICY = "sealed_unique_origin_attribution@1"
SCHEMA = "ecologyrsi-dsh.candidate-agent-attribution/1"
METRIC_KEY = "candidate_agent_attribution"
_ALLOWED_PHASES = frozenset({"screening", "formal_batch"})


def enabled(metadata):
    policy = metadata.get(POLICY_KEY)
    if policy not in (None, POLICY):
        raise ValueError("unsupported candidate_attribution_policy")
    return policy == POLICY


def _number(value):
    return float(value) if type(value) in (int, float) and isfinite(value) else None


def _candidate_output(row, record, identity):
    if not isinstance(identity, Mapping) or not identity.get("parameters_digest"):
        return None, False
    accepted = [a for a in record.get("attempt_trace", ())
                if isinstance(a, Mapping) and a.get("outcome") == "accepted"]
    if not accepted or _number(record.get("predicted")) != _number(row.get("predicted")):
        return None, False
    receipts = accepted[-1].get("model_evidence", ())
    info = row.get("agent_prediction", {})
    values, cited = [], False
    for tool in info.get("tools", ()) if isinstance(info, Mapping) else ():
        if not isinstance(tool, Mapping) or tool.get("status") != "completed":
            continue
        value = _number(tool.get("tool_predicted"))
        if value is None or not tool.get("call_id"):
            continue
        for receipt in receipts:
            if not (isinstance(receipt, Mapping) and receipt.get("status") == "completed"
                    and receipt.get("dsh_tool_event_id")
                    and receipt.get("call_id") == tool["call_id"]
                    and receipt.get("tool_id") == tool.get("tool_id")
                    and _number(receipt.get("tool_predicted")) == value):
                continue
            alias = receipt.get("tool_id") == "candidate-model" and not receipt.get("parameters")
            explicit = (receipt.get("tool_id") == identity.get("predictor_id")
                        and receipt.get("effective_parameters_digest") == identity["parameters_digest"])
            if alias or explicit:
                values.append(value)
                cited = cited or tool.get("used_as_evidence") is True
    if not values or any(not isclose(v, values[0], rel_tol=0, abs_tol=1e-10) for v in values):
        return None, False
    return values[0], cited


def summarize_candidate_attribution(records, scoring_rows, *, metadata, source_phase,
                                    candidate_identity, cohort_digest=None):
    """Return all declared cells, using only completed adaptation predictions.

    The caller persists this with the sealed evaluation. A screening summary is
    permitted for local diagnosis, but next-generation delivery must select a
    sealed formal adaptation batch, never training holdout or independent data.
    """
    if (not enabled(metadata) or source_phase not in _ALLOWED_PHASES
            or metadata.get("evaluation_partition", "training_feedback") != "training_feedback"):
        return None
    task = metadata.get("dataset_task", {})
    targets = [v["name"] for v in task.get("targets", ())]
    horizons = list(task.get("horizons_hours", ()))
    grid = sorted({(t, h) for t in targets for h in horizons})
    if not grid or len(grid) > 32 or any(not isinstance(t, str) or type(h) is not int or h <= 0 for t, h in grid):
        raise ValueError("candidate attribution requires a bounded frozen task grid")
    by_id = defaultdict(list)
    for record in records:
        if isinstance(record, Mapping):
            by_id[record.get("sample_id")].append(record)
    groups = {key: defaultdict(list) for key in grid}
    for row in scoring_rows:
        if row.get("partition") != "training_feedback":
            continue
        key = (row.get("target"), row.get("horizon_hours"))
        origin = row.get("origin_timestamp")
        if key not in groups or type(origin) is not int:
            continue
        observed, final = _number(row.get("observed")), _number(row.get("predicted"))
        successful = (row.get("sample_execution_status", row.get("status", "succeeded")) == "succeeded"
                      and not row.get("scoring_fallback") and observed is not None and final is not None)
        raw, cited = None, False
        records_for_sample = by_id.get(row.get("sample_id"), ())
        # Identical duplicate archives do not create observations. Conflicting
        # traces are unverified instead of selecting a convenient prediction.
        unique_records = {digest(thaw_json(r)): r for r in records_for_sample}
        if successful and len(unique_records) == 1:
            raw, cited = _candidate_output(row, next(iter(unique_records.values())), candidate_identity)
        groups[key][origin].append((observed, final, raw, cited, successful))
    cells = []
    for (target, horizon), origins in groups.items():
        successful, paired, blocks = [], [], defaultdict(list)
        ambiguous = 0
        for origin, duplicates in origins.items():
            distinct = set(duplicates)
            if len(distinct) != 1:
                ambiguous += 1
                continue
            observed, final, raw, cited, ok = next(iter(distinct))
            if not ok:
                continue
            successful.append(abs(final-observed))
            if raw is not None:
                item = (abs(raw-observed), abs(final-observed), abs(final-raw), cited)
                paired.append(item)
                blocks[origin // 24].append(item)
        n = len(paired)
        gains = [r-f for r, f, _, _ in paired]
        block_gains = [(fmean(r-f for r, f, _, _ in values), len(values)) for values in blocks.values()]
        worst = min(block_gains) if block_gains else (None, 0)
        cells.append({"target": target, "horizon_hours": horizon,
            "distinct_origins": len(origins), "successful_origins": len(successful),
            "paired_origins": n, "missing_candidate_origins": len(successful)-n,
            "ambiguous_origins": ambiguous,
            "candidate_cited_origins": sum(c for _, _, _, c in paired),
            "candidate_raw_mae": fmean(r for r, _, _, _ in paired) if n else None,
            "candidate_raw_rmse": sqrt(fmean(r*r for r, _, _, _ in paired)) if n else None,
            "final_mae": fmean(f for _, f, _, _ in paired) if n else None,
            "final_rmse": sqrt(fmean(f*f for _, f, _, _ in paired)) if n else None,
            "all_successful_final_mae": fmean(successful) if successful else None,
            "all_successful_final_rmse": sqrt(fmean(v*v for v in successful)) if successful else None,
            "adjustment_mae_gain": fmean(gains) if n else None,
            "adjustment_mean_absolute_delta": fmean(d for _, _, d, _ in paired) if n else None,
            "adjustment_harmed_origins": sum(g < -1e-10 for g in gains),
            "adjustment_helped_origins": sum(g > 1e-10 for g in gains),
            "adjustment_tied_origins": sum(abs(g) <= 1e-10 for g in gains),
            "paired_24h_blocks": len(blocks),
            "worst_24h_adjustment_mae_gain": worst[0],
            "worst_24h_paired_origins": worst[1]})
    body = {"schema_version": SCHEMA, "policy": POLICY, "source_phase": source_phase,
        "source_partition": "training_feedback", "cohort_digest": cohort_digest,
        "candidate_model_identity": thaw_json(candidate_identity) if isinstance(candidate_identity, Mapping) else None,
        "observation_unit": "distinct_origin_per_target_horizon; retries_are_not_samples",
        "interpretation": "observational_not_causal; positive_adjustment_mae_gain_means_final_error_lower; paired_metrics_use_same_origins; time_blocks_not_independent_certification",
        "scientific_gain_established": False, "cells": cells}
    return {**body, "summary_digest": digest(body)}


def validated_summary(value, *, phases=_ALLOWED_PHASES):
    """Validate a persisted aggregate without reading its raw scoring labels."""
    if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA:
        return None
    body = thaw_json({k: v for k, v in value.items() if k != "summary_digest"})
    if (value.get("policy") != POLICY or value.get("source_partition") != "training_feedback"
            or value.get("source_phase") not in phases or value.get("summary_digest") != digest(body)):
        return None
    cells = body.get("cells")
    if not isinstance(cells, list) or not 1 <= len(cells) <= 32:
        return None
    if len({(r.get("target"), r.get("horizon_hours")) for r in cells}) != len(cells):
        return None
    return {**body, "summary_digest": value["summary_digest"]}
