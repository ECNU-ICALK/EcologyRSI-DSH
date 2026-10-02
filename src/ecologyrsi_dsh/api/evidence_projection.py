"""Public evidence semantics; missing measurements are never zero successes."""
from collections import Counter
from collections.abc import Mapping
import math

from ..evolution.diagnosis import failure_category
from ..core.models import digest


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _value(value):
    return getattr(value, "value", value)


def candidate_score_evidence(state, candidate, evaluation):
    """Use recorded comparisons, never subtract unrelated candidate windows."""
    result = {
        "score_vs_baseline": _number(getattr(evaluation, "score", None)),
        "baseline_reference": getattr(evaluation, "metrics", {}).get("baseline_profile_digest"),
        "delta_vs_parent": None, "parent_comparison_id": None,
        "delta_vs_incumbent": None, "incumbent_comparison_id": None,
        "cohort_digest": None, "selected_gain": None,
        "interpretation": "baseline_skill_is_not_revision_gain",
    }
    comparisons = [item for item in getattr(state, "formal_batch_comparisons", ())
                   if item.candidate_id == candidate.candidate_id
                   and getattr(item, "challenger_revision_id", None)
                   and _value(item.decision) != "initial_champion"]
    if comparisons:
        item = max(comparisons, key=lambda row: row.batch_index)
        result.update(delta_vs_parent=_number(item.score_delta),
                      parent_comparison_id=item.comparison_id,
                      parent_cohort_digest=item.cohort_digest,
                      parent_reference_revision_id=item.champion_before_revision_id,
                      parent_compared_revision_id=item.challenger_revision_id)
    for comparison in reversed(getattr(state, "generation_comparisons", ())):
        if comparison.generation != candidate.generation:
            continue
        arms = {str(_value(item.scope.holdout_arm)): item for item in comparison.holdout_evaluations}
        incumbent = arms.get("incumbent")
        challenger = next((item for item in arms.values()
                           if item.scope.candidate_id == candidate.candidate_id), None)
        if (incumbent is None or challenger is None
                or not incumbent.scope.cohort_digest
                or incumbent.scope.cohort_digest != challenger.scope.cohort_digest
                or incumbent.evaluator_digest != challenger.evaluator_digest):
            continue
        current, reference = _number(challenger.score), _number(incumbent.score)
        result.update(score_vs_baseline=current,
                      delta_vs_incumbent=current-reference if current is not None and reference is not None else None,
                      incumbent_comparison_id=comparison.comparison_id,
                      incumbent_reference_revision_id=incumbent.scope.candidate_revision_id,
                      incumbent_compared_revision_id=challenger.scope.candidate_revision_id,
                      cohort_digest=challenger.scope.cohort_digest,
                      selected_gain=_number(comparison.gate_results.get("delta_to_incumbent")))
        break
    return result


def _ratio(numerator, denominator, unit, *, unavailable=None):
    return {"numerator": numerator, "denominator": denominator, "unit": unit,
            "rate": numerator / denominator if numerator is not None and denominator else None,
            "status": unavailable or ("measured" if denominator else "not_started")}


def _identified(value, schema, identity):
    return (isinstance(value, Mapping) and value.get("schema_version") == schema
            and value.get(identity) == digest({key: item for key, item in value.items() if key != identity}))


def execution_effect_observations(state, candidates):
    """Bounded aggregates of Host records; never return trace or prompt bodies."""
    registered, resolved = set(), set()
    proposal_for = getattr(state, "proposal", None)
    for candidate in candidates:
        key = "candidate:" + candidate.candidate_id
        registered.add(key)
        if not callable(proposal_for) or not isinstance(getattr(candidate, "proposal_id", None), str):
            continue
        resolution = proposal_for(candidate.proposal_id).metadata.get("effect_resolution")
        if _identified(resolution, "ecologyrsi-dsh.mutation-effect-resolution/1", "resolution_digest"):
            resolved.add(key)
    for item in getattr(state, "local_edit_outcomes", ()):
        if item.get("outcome") != "applied" or not item.get("active_revision_id"):
            continue
        key = "revision:" + item["active_revision_id"]
        registered.add(key)
        if (_identified(item.get("effect_resolution"), "ecologyrsi-dsh.mutation-effect-resolution/1", "resolution_digest")
                and _identified(item.get("effect_contract"), "ecologyrsi-dsh.edit-effect-contract/1", "contract_id")):
            resolved.add(key)
    metrics = [item.metrics for item in getattr(state, "formal_batch_evaluations", ())]
    metrics.extend(event.payload.get("metrics", {}) for event in getattr(state, "events", ())
                   if event.kind == "CandidateScreeningRecorded")
    receipts = {}
    for row in metrics:
        sample = row.get("sample_execution", {}) if isinstance(row, Mapping) else {}
        receipt = sample.get("mutation_effect_receipt") if isinstance(sample, Mapping) else None
        if (_identified(receipt, "ecologyrsi-dsh.edit-effect-receipt/1", "receipt_digest")
                and receipt.get("status") in {"passed", "failed", "inconclusive", "not_exercised"}):
            receipts[receipt["receipt_digest"]] = receipt["status"]
    counts = Counter(receipts.values())
    structural = _ratio(len(resolved), len(registered), "registered_mutation") if resolved else _ratio(
        None, None, "registered_mutation", unavailable="host_effect_resolution_not_recorded")
    structural["interpretation"] = "coverage_of_registered_mutations_not_all_generated_attempts"
    observed = {name: counts[name] for name in ("passed", "failed", "inconclusive", "not_exercised")}
    observed.update(recorded=len(receipts), unit="distinct_training_effect_receipt",
                    controlled_behavior_change_verified=False,
                    interpretation="execution_observation_is_not_paired_behavior_change_or_scientific_gain")
    return structural, observed


def evidence_funnel(state):
    """Separate registration, execution, selection and independent certification.

    Registered proposals do not reveal how many invalid outputs the model tried.
    Formal search gates are not independent final certification events.
    """
    candidates = [item for item in getattr(state, "candidates", ())
                  if _value(getattr(item, "role", "search")) != "incumbent_control"]
    ids = {item.candidate_id for item in candidates}
    events = getattr(state, "events", ())
    trained = {item.scope.candidate_id for item in getattr(state, "formal_batch_evaluations", ())} & ids
    screened = {event.payload.get("candidate_id") for event in events
                if event.kind == "CandidateScreeningRecorded"} & ids
    omitted = {event.payload.get("candidate_id") for event in events
               if event.kind == "CandidateScreenedOut"
               and event.payload.get("reason") == "outside_preregistered_quick_trajectory"} & ids
    local = [item for item in getattr(state, "formal_batch_comparisons", ())
             if getattr(item, "challenger_revision_id", None)
             and _value(item.decision) != "initial_champion"]
    started_pairs = {(item.candidate_id, item.batch_index) for item in local}
    schedule = state.task_manifest.metadata.get("optimization_schedule")
    if isinstance(schedule, Mapping) and schedule.get("local_evaluation_mode") == "paired_champion_challenger":
        started_pairs.update((item.candidate_id, item.batch_index) for item in getattr(state, "formal_batches", ()) if item.batch_index > 0)
    local_complete = [item for item in local if _number(item.score_delta) is not None
                      and item.reason not in {"incompatible_comparison_contract", "champion_evaluation_incomplete",
                                              "challenger_evaluation_incomplete", "paired_scoring_evidence_incomplete",
                                              "paired_strict_agent_chain_failed"}]
    generations = getattr(state, "generation_comparisons", ())
    paired, adopted = [], set()
    failures = Counter()
    for comparison in generations:
        gates = comparison.gate_results
        for arm, gate in gates.get("arms", {}).items():
            if arm == "incumbent":
                continue
            codes = set(gate.get("search_failures", ())) | set(gate.get("certification_failures", ()))
            for code in codes:
                failures[failure_category(code)] += 1
            if gate.get("paired_scoring_evidence_complete") is True and gate.get("strict_agent_chain_pass") is True:
                paired.append((comparison, arm))
        getter = getattr(state, "effective_revision_for", None)
        committed = callable(getter) and getter(comparison.generation) == comparison.selected_revision_id
        if committed and gates.get("selected_arm") in {"finalist_1", "finalist_2"}:
            adopted.add(comparison.generation)
    for event in events:
        if event.kind == "DshChildExecutionFailed":
            failures["operational"] += 1
        elif event.kind == "LocalEditDecided" and event.payload.get("outcome") == "rejected":
            failures["invalid_change"] += 1
    for item in local:
        if _value(item.decision) != "challenger_promoted":
            category = failure_category(item.reason)
            if category != "unclassified":
                failures[category] += 1
    outcomes = getattr(state, "local_edit_outcomes", ())
    applied = sum(item.get("outcome") == "applied" for item in outcomes)
    certification = {event.payload.get("token_digest", event.payload.get("candidate_id")): event.payload
                     for event in events if event.kind == "FormalStageCompleted" and event.payload.get("stage") == "validation"}
    certification_valid = [item for item in certification.values() if item.get("outcome") in {"passed", "failed"}]
    structural, effects = execution_effect_observations(state, candidates)
    return {
        "schema_version": "ecologyrsi-dsh.evidence-funnel/1",
        "protocol": state.task_manifest.metadata.get("optimization_protocol"),
        "registered_candidates": len(ids), "budget_unallocated_candidates": len(omitted),
        "applied_edits": applied, "applied_is_verified_gain": False,
        "generated_attempts": None, "execution_effect_observations": effects,
        "stages": {
            "structural_validity": structural,
            "behavior_change": _ratio(None, None, "revision", unavailable="controlled_behavior_comparison_not_recorded"),
            "screening_coverage": _ratio(len(screened), len(ids), "candidate"),
            "training_coverage": _ratio(len(trained), len(ids), "candidate"),
            "local_pair_completion": _ratio(len(local_complete), len(started_pairs), "comparison"),
            "local_positive_gain": _ratio(sum(item.score_delta > 0 for item in local_complete), len(local_complete), "comparison"),
            "local_acceptance": _ratio(sum(_value(item.decision) == "challenger_promoted" for item in local_complete), len(local_complete), "comparison"),
            "search_adoption": _ratio(len(adopted & {item.generation for item, _ in paired}), len({item.generation for item, _ in paired}), "generation"),
            "independent_certification": _ratio(sum(item["outcome"] == "passed" for item in certification_valid), len(certification_valid), "candidate") if certification else _ratio(None, None, "candidate", unavailable="independent_certification_not_recorded"),
        },
        "certification_inconclusive": sum(item.get("outcome") == "inconclusive" for item in certification.values()),
        "failure_observations": {key: failures[key] for key in (
            "operational", "invalid_change", "insufficient_evidence", "performance", "unclassified")},
        "failure_count_unit": "recorded_reason_observations_not_unique_candidates",
        "aggregation_scope": "this_run_and_frozen_protocol_only",
    }
