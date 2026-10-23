"""Bounded donors from executed training branches, never transferred scores."""
from collections import Counter
from collections.abc import Mapping

from ..core.immutable import thaw_json
from ..core.models import canonical_json, digest
from . import exploration_policy
from .effect_contracts import hard_effect_failure


def build_branch_archive(state, generation):
    if not exploration_policy.enabled(state.task_manifest.metadata):
        return {}
    from .genome import EcologyEvolutionPluginGenome
    events = sorted((e for e in state.generation_behavior_events if e.payload["generation"] < generation),
                    key=lambda e: e.payload["generation"], reverse=True)
    pools, seen = {}, set()
    for event in events:
        comparison = event.payload
        records = {e.payload["candidate_id"]: e.payload for e in state.candidate_screening_events
                   if e.payload["generation"] == comparison["generation"]}
        grouped = {cid: tuple(group) for group in comparison["groups"] for cid in group}
        local_seen = set()
        # Scores order candidates only within this exact frozen cohort.
        for member in sorted(comparison["members"], key=lambda m: (
                -records[m["candidate_id"]]["score"], m["candidate_id"])):
            candidate = state.candidate(member["candidate_id"])
            if getattr(candidate.role, "value", candidate.role) != "search":
                continue
            record = records[candidate.candidate_id]
            sample = record.get("metrics", {}).get("sample_execution", {})
            if (record.get("constraint_violations") != 0 or hard_effect_failure(record.get("metrics", {}))
                    or sample.get("execution_behavior", {}).get("status") != "measured"):
                continue
            proposal = state.proposal(candidate.proposal_id)
            realization = proposal.metadata.get("mechanism_realization", {})
            tags = list(realization.get("mechanism_ids", ()))
            if not tags:
                continue
            primary = proposal.metadata["candidate_direction"]["mechanism_hypothesis"]["mechanism_id"]
            identity = (primary, grouped[candidate.candidate_id])
            if identity in local_seen:
                continue
            local_seen.add(identity)
            revision = state.revision(member["revision_id"])
            genome = EcologyEvolutionPluginGenome.from_dict(thaw_json(revision.genome)).to_dict()
            execution = genome["agent_program"]["candidate_execution_program"]
            planner = next(p for p in execution["role_profiles"]
                           if p["role"] == "sample-planner")
            components = {key: planner[key] for key in ("authored_directive", "skill_program", "strategy_program")
                          if key in planner}
            components["instruction_template_id"] = planner["instruction_template_ref"]["id"]
            components["instruction_parameters"] = planner["instruction_parameters"]
            components["workflow_overrides"] = execution["workflow_overrides"]
            components["scientific_program"] = genome["scientific_program"]
            identity = digest(components)
            if (primary, identity) in seen:
                continue
            seen.add((primary, identity))
            entry = {
                "branch_id": digest({"mechanism": primary, "components": identity}),
                "family": primary, "mechanism_tags": tags,
                "candidate_id": candidate.candidate_id, "revision_id": revision.revision_id,
                "generation": comparison["generation"], "component_digest": identity,
                "components": components, "cohort_digest": comparison["cohort_digest"],
                "behavior_comparison_ref": comparison["comparison_digest"],
                "training_score": record["score"],
                "qualification": "screening_only_not_independent_validation",
                "reuse_requires_new_child_and_fresh_evaluation": True,
            }
            pools.setdefault(primary, []).append(entry)
    # Round robin across mechanisms; a prolific joint family cannot fill every slot.
    entries, counts = [], Counter()
    limit = exploration_policy.contract(state.task_manifest.metadata)
    for offset in range(2):
        for mechanism in sorted(pools):
            if len(entries) >= limit["archive_max_entries"] or len(pools[mechanism]) <= offset:
                continue
            entry = pools[mechanism][offset]
            if len(canonical_json([*entries, entry]).encode()) <= limit["archive_max_bytes"]:
                entries.append(entry)
                counts[mechanism] += 1
    body = {
        "schema_version": "ecologyrsi-dsh.exploration-archive/3",
        "entries": entries, "mechanism_counts": dict(counts),
        "source": "prior_generation_executed_training_screening_only",
        "selection": "mechanism_round_robin_no_cross_cohort_score_ranking",
        "reuse": "Use a compatible archived component in a new bounded mutation of the frozen current parent; "
                "the free slot may recombine donors. Re-evaluate every new child; scores never transfer.",
    }
    return {**body, "archive_digest": digest(body)}


def comparison_summary(state):
    events = getattr(state, "generation_behavior_events", ())
    if not events:
        return {"status": "not_recorded", "completed_generations": 0}
    event = max(events, key=lambda e: e.payload["generation"])
    p = event.payload
    measured = [pair for pair in p["pairs"] if pair["status"] == "measured"]
    return {"status": "measured" if len(measured) == len(p["pairs"]) else "unverified",
            "generation": p["generation"], "completed_generations": len(events),
            "measured_pairs": len(measured), "expected_pairs": len(p["pairs"]),
            "near_output_pairs": sum(pair.get("near_outputs") is True for pair in measured),
            "output_group_count": len(p["groups"]) if len(measured) == len(p["pairs"]) else None,
            "comparison_digest": p["comparison_digest"], "cohort_digest": p["cohort_digest"],
            "scientific_gain_established": False}


def candidate_mechanism_evidence(state, candidate, proposal, *, summary_only=False):
    receipt = proposal.metadata.get("mechanism_realization")
    if not isinstance(receipt, Mapping):
        return None
    screening = state.screening_for(candidate.generation, candidate.candidate_id)
    sample = (screening.payload.get("metrics", {}).get("sample_execution", {}) if screening else {})
    event = next((e for e in state.generation_behavior_events if e.payload["generation"] == candidate.generation), None)
    pairs = ([thaw_json(p) for p in event.payload["pairs"]
              if candidate.candidate_id in (p["left"], p["right"])] if event else [])
    member = next((m for m in event.payload["members"] if m["candidate_id"] == candidate.candidate_id), None) if event else None
    result = {"mechanism_ids": list(receipt["mechanism_ids"]),
              "revision_id": member["revision_id"] if member else None,
              "scope": "initial_candidate_screening",
              "execution_status": sample.get("mutation_effect_receipt", {}).get("status", "not_recorded"),
              "comparison_status": "measured" if pairs and all(p["status"] == "measured" for p in pairs) else "unverified",
              "scientific_gain_established": False}
    if not summary_only:
        effect = sample.get("mutation_effect_receipt") or {}
        result.update(hypothesis=thaw_json(proposal.metadata["candidate_direction"]["mechanism_hypothesis"]),
                      effect={"status": effect.get("status", "not_recorded"),
                              "receipt_digest": effect.get("receipt_digest"),
                              "checks": [{k: thaw_json(c[k]) for k in ("predicate_id", "status", "eligible", "matched") if k in c}
                                         for c in effect.get("checks", ())]},
                      comparisons=pairs, comparison_digest=event.payload["comparison_digest"] if event else None)
    return result
