"""Host-owned mechanism portfolios and bounded, non-executable hypotheses."""
from collections.abc import Mapping

from ..core.immutable import freeze_json, thaw_json
from ..core.models import canonical_json, digest
from . import exploration_policy

HYPOTHESIS_SCHEMA = "ecologyrsi-dsh.mechanism-hypothesis/1"
AXIS_MECHANISMS = {
    "scientific_parameter": "parameter_adjustment",
    "registered_predictor": "prediction_structure",
    "feature_recipe": "feature_structure",
    "instruction_directive": "agent_decision",
    "instruction_profile": "agent_decision",
    "skill_program": "agent_decision",
    "strategy_program": "agent_decision",
    "instruction_parameter": "execution_control",
    "workflow_parameter": "execution_control",
    "workflow_template": "execution_control",
    "instruction_tool_policy": "execution_control",
    "component_reset": "agent_decision",
    "feature_policy": "feature_structure",
    "fit_policy": "prediction_structure",
    "uncertainty_policy": "prediction_structure",
}
OBSERVATIONS = ("paired_error_decreases", "tool_output_changes", "execution_branch_exercised")
FALSIFICATIONS = ("no_paired_error_improvement", "no_tool_output_change", "branch_not_exercised")


def validate_hypothesis(value, *, catalog=None, axis=None):
    fields = {"schema_version", "mechanism_id", "input_refs", "target_cells",
              "expected_observation_id", "falsification_rule_id"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("mechanism_hypothesis fields do not match the contract")
    body = thaw_json(value)
    if body["schema_version"] != HYPOTHESIS_SCHEMA:
        raise ValueError("unsupported mechanism hypothesis schema")
    if body["mechanism_id"] not in set(AXIS_MECHANISMS.values()):
        raise ValueError("unknown hypothesis mechanism")
    if axis is not None and body["mechanism_id"] != AXIS_MECHANISMS.get(axis):
        raise ValueError("mechanism hypothesis does not match the mutation axis")
    if (body["expected_observation_id"], body["falsification_rule_id"]) not in tuple(zip(OBSERVATIONS, FALSIFICATIONS)):
        raise ValueError("hypothesis observation and falsification must be a registered pair")
    for key, maximum in (("input_refs", 8), ("target_cells", 16)):
        values = body[key]
        if (not isinstance(values, list) or not 1 <= len(values) <= maximum
                or any(not isinstance(v, str) or not v or len(v) > 160 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"invalid mechanism hypothesis {key}")
        if catalog is not None and set(values) - set(catalog[key]):
            raise ValueError(f"mechanism hypothesis names unavailable {key}")
    if len(canonical_json(body).encode()) > 2048:
        raise ValueError("mechanism hypothesis exceeds 2048 bytes")
    return freeze_json(body)


def build_candidate_portfolio(*, task, parent, generation, count, targets):
    if not exploration_policy.enabled(task.metadata):
        return {}
    from ..data.adapters import dataset_adapter
    adapter = dataset_adapter(task.dataset)
    axes = {axis: sorted(set(values)) for axis, values in sorted(targets.items())
            if values and axis in AXIS_MECHANISMS}
    mechanisms = sorted({AXIS_MECHANISMS[axis] for axis in axes})
    if not mechanisms:
        raise ValueError("exploration portfolio has no executable mechanisms")
    catalog = {
        "axes": axes,
        "mechanism_by_axis": {axis: AXIS_MECHANISMS[axis] for axis in axes},
        "input_refs": [f"history:{name}" for name in adapter.target_names],
        "target_cells": [f"{name}@{horizon}h" for name in adapter.target_names for horizon in adapter.horizons_hours],
        "expected_observation_ids": list(OBSERVATIONS),
        "falsification_rule_ids": list(FALSIFICATIONS),
    }
    minimum = min(3, len(mechanisms), count)
    rotated = mechanisms[generation % len(mechanisms):] + mechanisms[:generation % len(mechanisms)]
    body = {
        "schema_version": "ecologyrsi-dsh.candidate-portfolio/1",
        "policy": exploration_policy.POLICY,
        "generation": generation,
        "parent_genome_digest": parent.genome_digest,
        "mechanism_catalog": catalog,
        "mechanism_catalog_digest": digest(catalog),
        "minimum_distinct_mechanisms": minimum,
        "availability_reason": "full_coverage" if minimum >= 3 else "fewer_executable_mechanisms",
        "slot_requirements": [
            {"slot_index": slot, "allowed_mechanism_ids": [rotated[slot]] if slot < minimum else mechanisms}
            for slot in range(count)
        ],
    }
    return {**body, "portfolio_digest": digest(body)}


def validate_direction_portfolio(directions, portfolio):
    if not portfolio:
        return
    body = {k: thaw_json(v) for k, v in portfolio.items() if k != "portfolio_digest"}
    if portfolio.get("portfolio_digest") != digest(body):
        raise ValueError("candidate portfolio digest mismatch")
    if len(directions) != len(portfolio["slot_requirements"]):
        raise ValueError("candidate portfolio slot count mismatch")
    catalog = portfolio["mechanism_catalog"]
    actual = set()
    for slot, raw in enumerate(directions):
        direction = raw.to_dict() if hasattr(raw, "to_dict") else raw
        axis = direction["mutation_axis"]
        if direction["mutation_target"] not in catalog["axes"].get(axis, ()):
            raise ValueError("direction is outside the frozen mechanism catalog")
        hypothesis = validate_hypothesis(direction.get("mechanism_hypothesis"), catalog=catalog, axis=axis)
        mechanism = hypothesis["mechanism_id"]
        if mechanism not in portfolio["slot_requirements"][slot]["allowed_mechanism_ids"]:
            raise ValueError(f"candidate slot {slot} must use its assigned mechanism")
        actual.add(mechanism)
    if len(actual) < portfolio["minimum_distinct_mechanisms"]:
        raise ValueError("candidate portfolio lacks distinct mechanisms")


def realization_receipt(*, parent, child, operations, direction, portfolio, slot):
    """Classify compiled changes; source differences alone never prove novelty."""
    from .mutation_specs import mutation_spec
    from .effect_contracts import resolve_mutation_effects
    from ..knowledge.program_registry import current_program_registry

    hypothesis = validate_hypothesis(direction.get("mechanism_hypothesis"),
                                    catalog=portfolio["mechanism_catalog"], axis=direction["mutation_axis"])
    if not operations or mutation_spec(operations[0]["op"]).coordinates(operations[0])[:2] != (
            direction["mutation_axis"], direction["mutation_target"]):
        raise ValueError("primary mutation does not implement its mechanism hypothesis")
    if operations[0].get("role", "sample-planner") != "sample-planner":
        raise ValueError("primary mechanism must change the assigned sample-planner role")
    resolution = resolve_mutation_effects(parent, child, operations, current_program_registry())
    primary_change = resolution["operations"][0]
    move = direction["mutation_direction"]
    directions = mutation_spec(operations[0]["op"]).directions
    if move not in directions:
        raise ValueError("mechanism hypothesis direction is not supported")
    before, after = primary_change["before_compiled_value"], primary_change["after_compiled_value"]
    if "increase" in directions and (
            (move == "increase" and after <= before) or (move == "decrease" and after >= before)):
        raise ValueError("mechanism proposal contradicts its numeric hypothesis direction")
    if "author" in directions and (move == "author") == (before is not None):
        raise ValueError("mechanism hypothesis author/revise differs from its parent")
    mechanisms = sorted({AXIS_MECHANISMS[row["axis"]] for row in resolution["operations"]})
    primary = hypothesis["mechanism_id"]
    if primary not in portfolio["slot_requirements"][slot]["allowed_mechanism_ids"] or primary not in mechanisms:
        raise ValueError("realized mutation does not match its portfolio slot")
    body = {
        "schema_version": "ecologyrsi-dsh.mechanism-realization/1",
        "parent_genome_digest": parent.genome_digest, "child_genome_digest": child.genome_digest,
        "portfolio_digest": portfolio["portfolio_digest"], "hypothesis_digest": digest(thaw_json(hypothesis)),
        "slot_index": slot, "mechanism_ids": mechanisms,
        "qualification": "compiled_change_not_observed_behavior_or_scientific_gain",
    }
    return {**body, "receipt_digest": digest(body)}


def validate_proposal_exploration(proposal, *, task, parent, plan, generation, count):
    """Shared proposal admission and replay guard."""
    if not exploration_policy.enabled(task.metadata) or proposal.metadata.get("candidate_role") == "incumbent_control":
        return
    from .strategies import _registered_mutation_targets, _genome_parameter_boundary
    from ..core.state import persisted_genome_from_proposal
    from .genome import GenomeMutationContextV1, apply_genome_mutation, MUTATION_SCHEMA_VERSION
    from ..knowledge.program_registry import current_program_registry
    portfolio = build_candidate_portfolio(task=task, parent=parent, generation=generation, count=count,
                                          targets=_registered_mutation_targets(task, parent))
    if canonical_json(plan.get("candidate_portfolio")) != canonical_json(portfolio):
        raise ValueError("proposal portfolio differs from frozen research plan")
    directions = plan.get("candidate_directions", ())
    validate_direction_portfolio(directions, portfolio)
    child = persisted_genome_from_proposal(proposal)
    raw_context = thaw_json(proposal.metadata.get("mutation_context", {}))
    if raw_context.pop("schema_version", None) != "ecologyrsi-dsh.genome-mutation-context/1":
        raise ValueError("mechanism proposal requires its original mutation context")
    context = GenomeMutationContextV1(**raw_context)
    if (context.run_id != proposal.run_id or context.generation != generation
            or context.parent_candidate_id != proposal.parent_candidate_id):
        raise ValueError("mechanism mutation context is outside the proposal")
    _, schemas = _genome_parameter_boundary(task, parent)
    rebuilt = apply_genome_mutation(parent, {
        "schema_version": MUTATION_SCHEMA_VERSION,
        "operations": thaw_json(proposal.metadata.get("mutation_operations")),
    }, context, current_program_registry(), parameter_schemas=schemas)
    if rebuilt.genome_digest != child.genome_digest:
        raise ValueError("mechanism proposal genome differs from its bounded mutation")
    slot = child.lineage["slot_index"]
    if type(slot) is not int or not 0 <= slot < count:
        raise ValueError("proposal mechanism slot is invalid")
    if canonical_json(proposal.metadata.get("candidate_direction")) != canonical_json(directions[slot]):
        raise ValueError("proposal mechanism direction differs from frozen slot")
    receipt = realization_receipt(parent=parent, child=child, operations=proposal.metadata.get("mutation_operations"),
                                  direction=directions[slot], portfolio=portfolio, slot=slot)
    if canonical_json(proposal.metadata.get("mechanism_realization")) != canonical_json(receipt):
        raise ValueError("proposal mechanism realization receipt mismatch")


def validate_state_proposal(state, proposal):
    """Accept RunState or its reducer, using only already frozen generation data."""
    task = getattr(state, "task_manifest", None) or state.task
    if not exploration_policy.enabled(task.metadata) or proposal.metadata.get("candidate_role") == "incumbent_control":
        return
    import json
    from .genome import EcologyEvolutionPluginGenome
    batches, iterations = state.generation_batches, state.research_iterations
    batch = batches.get(proposal.generation) if isinstance(batches, Mapping) else next(
        (v for v in batches if v.generation == proposal.generation), None)
    iteration = iterations.get(proposal.generation) if isinstance(iterations, Mapping) else next(
        (v for v in iterations if v.generation == proposal.generation), None)
    if batch is None or iteration is None or not batch.parent_genome_canonical_json:
        raise ValueError("mechanism proposal requires a frozen generation and research plan")
    parent = EcologyEvolutionPluginGenome.from_dict(json.loads(batch.parent_genome_canonical_json))
    validate_proposal_exploration(proposal, task=task, parent=parent, plan=iteration.plan,
                                 generation=proposal.generation, count=batch.batch_size)
