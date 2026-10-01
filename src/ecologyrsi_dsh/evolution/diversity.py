"""Frozen family exploration and descriptive execution diversity.

No scores enter preregistration. New policies opt in through the run manifest;
old manifests keep their original first-slot selection and replay semantics.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from ..core.models import canonical_json, digest

DIVERSITY_POLICY = "executed_family_rotation@1"
FAMILY_AXES = {
    "prompt": ("instruction_directive", "instruction_profile"),
    "skill": ("skill_program",),
    "scientific": ("scientific_parameter", "registered_predictor", "feature_recipe"),
    "execution": ("workflow_parameter", "instruction_parameter"),
}


def enabled(metadata: Mapping) -> bool:
    policy = metadata.get("evolution_diversity_policy")
    if policy not in (None, DIVERSITY_POLICY):
        raise ValueError("unsupported evolution_diversity_policy")
    return policy == DIVERSITY_POLICY


def family_for_axis(axis: str) -> str:
    return next((family for family, axes in FAMILY_AXES.items() if axis in axes), "unknown")


def direction_schedule(metadata: Mapping, targets: Mapping, generation: int, count: int) -> dict:
    if not enabled(metadata):
        return {}
    families = [name for name, axes in FAMILY_AXES.items() if any(targets.get(axis) for axis in axes)]
    if not families:
        raise ValueError("diversity policy has no executable mutation families")
    return {"policy": DIVERSITY_POLICY,
            "required_family_by_slot": [families[(generation + slot) % len(families)] for slot in range(count)],
            "axes_by_family": {family: [axis for axis in FAMILY_AXES[family] if targets.get(axis)] for family in families},
            "evidence_rule": "Exploration is a testable hypothesis, not a claim of an observed weakness.",
            "selection": "preregister rotated family before evaluation; no additional prediction calls"}


def validate_direction_schedule(directions: Sequence[Mapping], schedule: Mapping) -> None:
    for slot, family in enumerate(schedule.get("required_family_by_slot", ())):
        if slot >= len(directions) or family_for_axis(directions[slot].get("mutation_axis")) != family:
            raise ValueError(f"candidate_directions[{slot}] must explore the {family} family")


def candidate_family(proposal) -> str:
    direction = proposal.metadata.get("candidate_direction")
    return family_for_axis(direction.get("mutation_axis")) if isinstance(direction, Mapping) else "unknown"


def preregister_candidate(candidates, proposal_for, metadata: Mapping, generation: int):
    ordered = sorted(candidates, key=lambda item: item.slot_index)
    if not ordered:
        raise ValueError("cannot preregister an empty candidate set")
    if not enabled(metadata):
        return ordered[0]
    frozen_schedule = proposal_for(ordered[0].proposal_id).metadata.get("diversity_schedule")
    if isinstance(frozen_schedule, Mapping) and frozen_schedule.get("required_family_by_slot"):
        family = frozen_schedule["required_family_by_slot"][0]
        matches = [item for item in ordered if candidate_family(proposal_for(item.proposal_id)) == family]
        if not matches:
            raise ValueError("preregistered diversity family has no candidate")
        return matches[0]
    # The synthesis contract rotates the family into slot zero. Selection also
    # handles imported/manual proposals deterministically without using labels.
    available = {candidate_family(proposal_for(item.proposal_id)) for item in ordered}
    families = [family for family in FAMILY_AXES if family in available]
    if not families:
        return ordered[0]
    family = families[generation % len(families)]
    return next(item for item in ordered if candidate_family(proposal_for(item.proposal_id)) == family)


def summarize_behavior(records: Sequence[Mapping], scoring_rows: Sequence[Mapping]) -> dict:
    """A descriptive signature of actual actions, excluding text and genome IDs.

    Counts are per scored cell (one origin can have several cells), not billable
    calls. Signatures across different cohorts are not novelty or quality tests.
    """
    methods, tool_paths, skill_paths = Counter(), Counter(), Counter()
    vector = []
    for row in scoring_rows:
        prediction = row.get("agent_prediction")
        if isinstance(prediction, Mapping):
            methods[str(prediction.get("method", "unknown"))] += 1
        value = row.get("predicted")
        if type(value) in (int, float) and math.isfinite(value):
            vector.append((str(row.get("sample_id", "")), value))
    for record in records:
        for attempt in record.get("attempt_trace", ()):
            if not isinstance(attempt, Mapping):
                continue
            path = [str(tool.get("tool_id")) for tool in attempt.get("model_evidence", ()) if isinstance(tool, Mapping)]
            if path:
                tool_paths[">".join(path)] += 1
            for tool in attempt.get("skill_evidence", ()):
                if isinstance(tool, Mapping):
                    skill_paths[str(tool.get("behavior_signature"))] += 1
    body = {"method_counts": dict(sorted(methods.items())), "tool_path_counts": dict(sorted(tool_paths.items())),
            "skill_path_counts": dict(sorted(skill_paths.items()))}
    return {"schema_version": "ecologyrsi-dsh.observed-behavior/1", **body,
            "signature": digest(body), "prediction_vector_digest": digest(sorted(vector)),
            "scored_cells": len(scoring_rows), "trace_cells": len(records),
            "comparison_scope": "same_cohort_only", "novelty_verified": False}


def exploration_archive(state, generation: int) -> dict:
    """Keep measured training components; never rank different windows' scores.

    Reuse means authoring a new child of the current parent and evaluating it.
    Unexecuted proposals and holdout evaluations cannot enter this archive.
    """
    if not enabled(getattr(getattr(state, "task_manifest", None), "metadata", {})):
        return {}
    from .genome import EcologyEvolutionPluginGenome
    entries, seen, counts = [], set(), Counter()
    batches = sorted((item for item in getattr(state, "formal_batch_evaluations", ())
                      if item.scope.generation < generation and item.scope.phase.value == "formal_batch"),
                     key=lambda item: (item.scope.generation, item.scope.batch_index, item.scope.candidate_id), reverse=True)
    for batch in batches:
        candidate_id = batch.scope.candidate_id
        if candidate_id in seen:
            continue
        metrics = batch.metrics
        summary = metrics.get("sample_execution", {})
        behavior = summary.get("observed_behavior", {})
        if (metrics.get("constraint_violations") != 0 or summary.get("coverage_pass") is not True
                or not behavior.get("method_counts") or not math.isfinite(batch.score)):
            continue
        # A later failed batch must not erase an earlier measured component.
        seen.add(candidate_id)
        candidate = state.candidate(candidate_id)
        family = candidate_family(state.proposal(candidate.proposal_id))
        if family == "unknown" or counts[family] >= 2:
            continue
        revision = state.revision(batch.scope.candidate_revision_id)
        genome = EcologyEvolutionPluginGenome.from_dict(dict(revision.genome)).to_dict()
        planner = next(p for p in genome["agent_program"]["candidate_execution_program"]["role_profiles"] if p["role"] == "sample-planner")
        components = {key: planner[key] for key in ("authored_directive", "skill_program") if key in planner}
        components["instruction_template_id"] = planner["instruction_template_ref"]["id"]
        if family == "scientific":
            components["scientific_program"] = genome["scientific_program"]
        if family == "execution":
            components["workflow_overrides"] = genome["agent_program"]["candidate_execution_program"]["workflow_overrides"]
            components["instruction_parameters"] = planner["instruction_parameters"]
        identity = digest(components)
        if any(entry["component_digest"] == identity and entry["family"] == family for entry in entries):
            continue
        entry = {"candidate_id": candidate_id, "revision_id": revision.revision_id,
                 "generation": batch.scope.generation, "family": family,
                 "component_digest": identity, "components": components,
                 "training_scope_digest": batch.scope.scope_key,
                 "training_score": batch.score, "behavior_signature": behavior["signature"],
                 "qualification": "executable_training_example_not_certified"}
        if len(canonical_json([*entries, entry]).encode("utf-8")) > 6000:
            continue
        entries.append(entry)
        counts[family] += 1
    return {"schema_version": "ecologyrsi-dsh.exploration-archive/1", "entries": entries,
            "reuse": "Revise or recombine a component in a new candidate; re-evaluate before adoption.",
            "selection": "latest_distinct_per_family_no_cross_cohort_score_ranking",
            "source": "prior_generation_formal_training_only", "max_entries_per_family": 2}


def diversity_report(state) -> dict:
    """Public coverage counts; an untrained proposal is never an experiment."""
    proposed, trained, completed = Counter(), Counter(), Counter()
    trained_ids = {item.scope.candidate_id for item in getattr(state, "formal_batch_evaluations", ())}
    observed = set()
    for candidate in state.candidates:
        if getattr(candidate.role, "value", candidate.role) != "search":
            continue
        family = candidate_family(state.proposal(candidate.proposal_id))
        proposed[family] += 1
        if candidate.candidate_id in trained_ids:
            trained[family] += 1
            if state.evaluation_for(candidate.candidate_id) is not None:
                completed[family] += 1
    for batch in getattr(state, "formal_batch_evaluations", ()):
        behavior = batch.metrics.get("sample_execution", {}).get("observed_behavior", {})
        if behavior.get("signature"):
            observed.add(behavior["signature"])
    return {"policy": state.task_manifest.metadata.get("evolution_diversity_policy"),
            "proposed_by_family": dict(proposed), "trained_by_family": dict(trained),
            "completed_by_family": dict(completed), "observed_signature_count": len(observed),
            "signature_count_is_descriptive_not_novelty_proof": True,
            "prompt_skill_training_fraction": (sum(v for k, v in trained.items() if k in {"prompt", "skill"}) / sum(trained.values())) if trained else None,
            "archive": exploration_archive(state, state.run.generation + 1)}
