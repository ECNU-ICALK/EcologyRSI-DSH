"""Host-resolved mutation effects and bounded checks, independent of scoring.

These contracts establish what changed and whether it was exercised. They never
replace scientific comparison, lower a gate, or accept model-authored validators.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from ..core.models import canonical_json, digest
from ..core.immutable import thaw_json
from .mutation_specs import mutation_spec

EFFECT_CONTRACT_SCHEMA = "ecologyrsi-dsh.edit-effect-contract/1"
TRAINING_EVIDENCE_PHASES = frozenset({"screening", "formal_batch", "training_feedback"})
_PREDICATES = {
    "candidate_model_evidence": "hard",
    "guidance_behavior_changed": "soft",
    "skill_triggered": "soft",
    "execution_branch_exercised": "hard",
}


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _texts(values: Any, name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be an array")
    if any(not isinstance(v, str) or not v.strip() for v in values) or len(set(values)) != len(values):
        raise ValueError(f"{name} requires unique non-empty text")
    return list(values)


def validate_runtime_constraints(value: Mapping | None) -> dict:
    """Only Host-owned policy inputs needed to resolve effective parameters."""
    if value is None:
        return {}
    if not isinstance(value, Mapping) or set(value) - {"sample_max_attempts_floor", "remote_critic_policy"}:
        raise ValueError("unsupported effect runtime constraints")
    result = thaw_json(value)
    floor = result.get("sample_max_attempts_floor")
    if floor is not None and (type(floor) is not int or not 1 <= floor <= 8):
        raise ValueError("sample_max_attempts_floor must be between 1 and 8")
    if "remote_critic_policy" in result:
        from ..evaluators.sample_contracts import _normalized_remote_critic_policy
        result["remote_critic_policy"] = _normalized_remote_critic_policy(result["remote_critic_policy"])
    return result


def mutation_effect_cells(operations, available_cells) -> tuple[str, ...]:
    """Host impact domain; model-authored expected cells cannot narrow it."""
    effects = set()
    for operation in operations:
        name = str(operation.get("name", ""))
        if operation.get("op") != "set_bounded_parameter" or "residual_scale" not in name:
            effects.update(available_cells)
            continue
        for cell in available_cells:
            target, horizon = cell.rsplit("@", 1)
            if name in {"residual_scale", f"residual_scale_{horizon}",
                        f"{target}_residual_scale", f"{target}_{horizon}_residual_scale"}:
                effects.add(cell)
    return tuple(sorted(effects))


def hard_effect_failure(metrics: Mapping) -> str | None:
    """A measured hard contradiction blocks selection, never changes scores.

    Missing/soft/inconclusive/unexercised checks carry no claim of success and
    remain distinguishable from a failed deterministic behavior constraint.
    """
    sample = metrics.get("sample_execution", {})
    receipt = sample.get("mutation_effect_receipt") if isinstance(sample, Mapping) else None
    if not isinstance(receipt, Mapping):
        return None
    receipt = thaw_json(receipt)
    if (receipt.get("schema_version") != "ecologyrsi-dsh.edit-effect-receipt/1"
            or receipt.get("receipt_digest") != digest({k: v for k, v in receipt.items() if k != "receipt_digest"})):
        raise ValueError("mutation effect receipt identity mismatch")
    checks = receipt.get("checks", ())
    if not isinstance(checks, (list, tuple)):
        raise ValueError("mutation effect receipt checks are invalid")
    if isinstance(checks, (list, tuple)) and any(
        isinstance(row, Mapping) and row.get("semantics") == "hard" and row.get("status") == "failed"
        for row in checks
    ):
        return "hard_mutation_effect_failed"
    return None


def resolve_mutation_effects(parent, child, operations, registry, *, runtime_constraints=None) -> dict:
    """Resolve requested values against the same compiled profiles as execution.

    Runtime floors must be supplied by the caller that owns the execution plan.
    With no runtime constraints this is explicitly a compiled-profile check.
    """
    from .workflow_ir import EXECUTABLE_WORKFLOW_PARAMETERS, resolve_candidate_agent_profile

    runtime = validate_runtime_constraints(runtime_constraints)
    before, after = parent.to_dict(), child.to_dict()
    profiles = {}

    def profile(genome, role):
        key = (genome.genome_digest, role)
        if key not in profiles:
            profiles[key] = resolve_candidate_agent_profile(genome, registry, role=role)
        return profiles[key]

    def effective(genome, source, operation, spec, constraints):
        scientific = source["scientific_program"]
        op = operation["op"]
        name = operation.get("name")
        if op == "set_bounded_parameter":
            return scientific["parameter_overrides"].get(name), "candidate_parameter"
        if op == "select_registered_pipeline":
            return {k: scientific[k] for k in ("predictor_ref", "parameter_overrides")}, "registered_pipeline"
        if op == "author_feature_recipe":
            return scientific.get("feature_recipe"), "registered_feature_recipe"
        if spec.axis in {"feature_policy", "fit_policy", "uncertainty_policy"}:
            return scientific[spec.axis + "_ref"], "registered_policy"
        role = operation.get("role", "sample-planner")
        resolved = profile(genome, role)
        if op in {"set_bounded_workflow_parameter", "select_registered_workflow_template"}:
            values = resolved["workflow_parameters"]
            if op == "select_registered_workflow_template":
                return {key: values[key] for key in EXECUTABLE_WORKFLOW_PARAMETERS}, "compiled_workflow"
            if name not in EXECUTABLE_WORKFLOW_PARAMETERS:
                raise ValueError("effective_noop: workflow parameter has no execution consumer")
            requested = values[name]
            floor = constraints.get("sample_max_attempts_floor")
            return (max(requested, floor), "host_floor") if floor is not None else (requested, "compiled_workflow")
        if op == "set_instruction_parameter":
            if name == "confidence_threshold" and "remote_critic_policy" in constraints:
                policy = constraints["remote_critic_policy"]
                if policy is None or policy["version"] == "always@1":
                    return "always", "host_critic_policy"
            return resolved["instruction_parameters"].get(name), "compiled_instruction"
        if op == "select_instruction_template":
            # A template shadowed by an authored directive is not a prompt edit.
            return {key: resolved[key] for key in ("instruction_directive", "instruction_parameters")}, "compiled_instruction"
        if op == "narrow_role_tool_policy":
            return sorted(resolved["enabled_tool_ids"]), "compiled_tool_policy"
        component = (operation["component"] if op == "reset_role_component" else
                     "skill_program" if op == "author_skill_program" else "authored_directive")
        return resolved.get(component), "compiled_role_component"

    rows = []
    for operation in operations:
        spec = mutation_spec(operation.get("op"))
        axis, target, path = spec.coordinates(operation)
        old, _ = effective(parent, before, operation, spec, runtime)
        new, source = effective(child, after, operation, spec, runtime)
        if canonical_json(old) == canonical_json(new):
            raise ValueError(f"effective_noop: {path} does not change its executable value")
        requested = {key: thaw_json(value) for key, value in operation.items() if key != "op"}
        rows.append({"operation": operation["op"], "axis": axis, "target": target, "path": path,
                     "requested": requested, "requested_value": operation.get("value", requested),
                     "before_compiled_value": effective(parent, before, operation, spec, {})[0],
                     "after_compiled_value": effective(child, after, operation, spec, {})[0],
                     "before_effective_value": old, "after_effective_value": new,
                     "effective_source": source, "effect_kind": spec.effect_kind})
    if not rows:
        raise ValueError("effect resolution requires a mutation")
    body = {"schema_version": "ecologyrsi-dsh.mutation-effect-resolution/1",
            "parent_genome_digest": parent.genome_digest, "child_genome_digest": child.genome_digest,
            "operation_digest": digest(list(operations)), "runtime_constraints": runtime,
            "resolution_scope": "runtime_policy" if runtime else "compiled_profile", "operations": rows}
    return {**body, "resolution_digest": digest(thaw_json(body))}


def _effect_checks(resolution: Mapping) -> list[dict]:
    checks = {}
    for row in resolution["operations"]:
        predicate = ("candidate_model_evidence" if row["effect_kind"] == "prediction" else
                     "execution_branch_exercised" if row["effect_kind"] == "execution" else
                     "guidance_behavior_changed")
        checks[predicate] = {"predicate_id": predicate, "semantics": _PREDICATES[predicate]}
        if row["operation"] == "author_skill_program":
            checks["skill_triggered"] = {"predicate_id": "skill_triggered", "semantics": "soft"}
    return list(checks.values())


def build_edit_effect_contract(*, parent_revision_id: str, evidence_scope_digest: str,
                              evidence_refs, affected_cells, resolution: Mapping,
                              source_phase="formal_batch", hypothesis_ref=None,
                              evaluation_policy_ref="frozen_host_scientific_gates") -> dict:
    """Freeze a Host-built contract without adding model-authored score gates."""
    if source_phase not in TRAINING_EVIDENCE_PHASES:
        raise ValueError("mutation effects require completed training evidence")
    if not isinstance(parent_revision_id, str) or not parent_revision_id.strip():
        raise ValueError("effect contract requires its parent revision")
    _sha(evidence_scope_digest, "evidence_scope_digest")
    refs, cells = _texts(evidence_refs, "evidence_refs"), _texts(affected_cells, "affected_cells")
    if not refs or not cells:
        raise ValueError("effect contract requires evidence and affected cells")
    body = {key: value for key, value in resolution.items() if key != "resolution_digest"}
    if resolution.get("resolution_digest") != digest(thaw_json(body)):
        raise ValueError("effect resolution digest mismatch")
    contract = {"schema_version": EFFECT_CONTRACT_SCHEMA, "parent_revision_id": parent_revision_id,
                "parent_genome_digest": resolution["parent_genome_digest"],
                "child_genome_digest": resolution["child_genome_digest"],
                "operation_digest": resolution["operation_digest"], "resolution_digest": resolution["resolution_digest"],
                "evidence_scope_digest": evidence_scope_digest, "source_phase": source_phase,
                "evidence_refs": refs, "affected_cells": cells, "hypothesis_ref": hypothesis_ref,
                "evaluation_policy_ref": evaluation_policy_ref, "effect_checks": _effect_checks(resolution),
                "qualification": "behavior_effect_only_not_scientific_improvement"}
    return {**contract, "contract_id": digest(contract)}


def _checked_observations(checks, observations):
    expected = {item["predicate_id"] for item in checks}
    if not isinstance(observations, Mapping) or set(observations) - expected:
        raise ValueError("unknown effect observation")
    results = []
    for check in checks:
        predicate = check["predicate_id"]
        if predicate not in _PREDICATES or check.get("semantics") != _PREDICATES[predicate]:
            raise ValueError("effect predicate is not registered")
        observation = observations.get(predicate)
        if observation is None:
            results.append({**check, "status": "inconclusive", "evidence_refs": []})
            continue
        if not isinstance(observation, Mapping) or set(observation) != {"eligible", "matched", "evidence_refs"}:
            raise ValueError("effect observation fields are invalid")
        eligible, matched = observation["eligible"], observation["matched"]
        if type(eligible) is not int or type(matched) is not int or not 0 <= matched <= eligible:
            raise ValueError("effect observation counts are invalid")
        refs = _texts(observation["evidence_refs"], "effect evidence_refs")
        if eligible and not refs:
            raise ValueError("effect observation requires receipt references")
        satisfied = matched == eligible if predicate == "candidate_model_evidence" else matched > 0
        status = ("not_exercised" if eligible == 0 else "passed" if satisfied else
                  "failed" if check["semantics"] == "hard" else "inconclusive")
        results.append({**check, **dict(observation), "status": status})
    statuses = {row["status"] for row in results}
    status = next((item for item in ("failed", "inconclusive", "not_exercised") if item in statuses), "passed")
    return results, status


def check_edit_effect_contract(contract: Mapping, *, scope_digest: str, child_genome_digest: str,
                               observations: Mapping, phase="formal_batch") -> dict:
    """Check bounded Host aggregates of existing receipts, never model claims.

    Each observation contains eligible/matched counts and receipt references.
    A branch that never occurred is not exercised, not a successful change.
    The caller resolves references against the frozen evaluation's receipts.
    """
    body = {key: value for key, value in contract.items() if key != "contract_id"}
    if contract.get("schema_version") != EFFECT_CONTRACT_SCHEMA or contract.get("contract_id") != digest(thaw_json(body)):
        raise ValueError("effect contract identity mismatch")
    if phase not in TRAINING_EVIDENCE_PHASES or contract["source_phase"] not in TRAINING_EVIDENCE_PHASES:
        raise ValueError("effect checks must not consume independent evaluation")
    _sha(scope_digest, "scope_digest")
    if child_genome_digest != contract["child_genome_digest"]:
        raise ValueError("effect receipt belongs to another child")
    results, status = _checked_observations(contract["effect_checks"], observations)
    receipt = {"schema_version": "ecologyrsi-dsh.edit-effect-receipt/1", "contract_id": contract["contract_id"],
               "scope_digest": scope_digest, "child_genome_digest": child_genome_digest, "status": status,
               "checks": results, "qualification": "behavior_effect_only_not_scientific_improvement"}
    return {**receipt, "receipt_digest": digest(receipt)}


def verify_runtime_effects(resolution: Mapping, runtime_constraints: Mapping) -> dict:
    """Recheck the actual Host floor before inference, including outer mutations."""
    body = {key: value for key, value in resolution.items() if key != "resolution_digest"}
    if resolution.get("resolution_digest") != digest(thaw_json(body)):
        raise ValueError("effect resolution digest mismatch")
    runtime = validate_runtime_constraints(runtime_constraints)
    rows = thaw_json(resolution["operations"])
    for row in rows:
        old, new = row["before_compiled_value"], row["after_compiled_value"]
        if row["axis"] == "workflow_parameter" and row["target"] == "max_attempts":
            floor = runtime.get("sample_max_attempts_floor", 1)
            old, new = max(old, floor), max(new, floor)
        if row["axis"] == "instruction_parameter" and "remote_critic_policy" in runtime:
            policy = runtime["remote_critic_policy"]
            if policy is None or policy["version"] == "always@1":
                old = new = "always"
        if canonical_json(old) == canonical_json(new):
            raise ValueError(f"effective_noop: {row['path']} is masked by the active Host policy")
        row.update(before_effective_value=old, after_effective_value=new)
    body.update(operations=rows, runtime_constraints=runtime, resolution_scope="runtime_policy")
    return {**body, "resolution_digest": digest(thaw_json(body))}


def observe_edit_effects(metadata: Mapping, records: Sequence[Mapping], *, scope_digest: str,
                        phase: str, parameters: Mapping) -> dict | None:
    """Measure execution from complete Host traces; no labels or score are read.

    Soft instruction compliance stays inconclusive without a paired behavioral
    reference. A delivered prompt or a generated skill receipt is not success.
    """
    if phase not in TRAINING_EVIDENCE_PHASES:
        return None
    operations = metadata.get("mutation_operations", ())
    contract = metadata.get("effect_contract")
    resolution = metadata.get("effect_resolution", metadata.get("mutation_effect_resolution"))
    if not operations and not isinstance(contract, Mapping):
        return None
    genome_json = metadata.get("evolution_genome_canonical_json")
    scientific = json.loads(genome_json)["scientific_program"] if isinstance(genome_json, str) else {}
    expected_predictor = scientific.get("predictor_ref", {}).get("id")
    expected_parameters = digest(thaw_json(
        {"feature_recipe": scientific["feature_recipe"]} if "feature_recipe" in scientific else parameters
    ))
    affected_cells = set(contract.get("affected_cells", ())) if isinstance(contract, Mapping) else set()
    if not affected_cells:
        cells = {f"{record['target']}@{record['horizon_hours']}h" for record in records
                 if "target" in record and "horizon_hours" in record}
        affected_cells = set(mutation_effect_cells(operations, cells))

    model_refs, skill_refs, branch_refs = set(), set(), set()
    eligible, model_matches, skill_eligible, triggered, branch_eligible, branches = 0, 0, 0, 0, 0, 0
    trace_missing = False
    formula_observed = formula_matched = 0
    for record in records:
        if affected_cells and "target" in record and "horizon_hours" in record:
            if f"{record['target']}@{record['horizon_hours']}h" not in affected_cells:
                continue
        attempts = record.get("attempt_trace", ())
        if isinstance(resolution, Mapping):
            for row in resolution["operations"]:
                if row["axis"] != "workflow_parameter" or row["target"] != "max_attempts":
                    continue
                old, new = row["before_effective_value"], row["after_effective_value"]
                depth = max((a.get("attempt", 0) for a in attempts), default=0)
                exercised = (depth > old if new > old else
                             depth == new and bool(attempts) and attempts[-1].get("outcome") != "accepted")
                if exercised:
                    branch_eligible += 1
                    branches += 1
                    branch_refs.add("trace:" + digest(attempts))
        accepted = next((a for a in reversed(attempts) if a.get("outcome") == "accepted"), None)
        if accepted is None:
            trace_missing = True
            continue
        eligible += 1
        matched = False
        for tool in accepted.get("model_evidence", ()):
            ref = tool.get("dsh_tool_event_id")
            if ref:
                model_refs.add(ref)
            if (tool.get("status") == "completed" and tool.get("used_as_evidence") is True
                    and (tool.get("tool_id") == "candidate-model"
                         or (expected_predictor is not None and tool.get("tool_id") == expected_predictor
                             and tool.get("effective_parameters_digest") == expected_parameters))):
                matched = True
        model_matches += int(matched)
        # A final-prediction digest also proves an observed direct/no-tool case.
        formula_status = accepted.get("selected_tool", {}).get("formula_status")
        formula_observed += int(formula_status in {"passed", "not_exercised"})
        formula_matched += int(formula_status == "passed")
        final_ref = accepted.get("selected_tool", {}).get("output_digest")
        if final_ref:
            model_refs.add("prediction:" + final_ref)
        skills = accepted.get("skill_evidence", ())
        for skill in skills:
            if skill.get("trigger_status") not in {"triggered", "not_triggered"}:
                continue
            skill_eligible += 1
            triggered += int(skill["trigger_status"] == "triggered")
            skill_refs.add("skill:" + skill["output_digest"])
    observations = {
        "candidate_model_evidence": {"eligible": eligible, "matched": model_matches, "evidence_refs": sorted(model_refs)},
        "skill_triggered": {"eligible": skill_eligible, "matched": triggered, "evidence_refs": sorted(skill_refs)},
        "execution_branch_exercised": {"eligible": branch_eligible, "matched": branches, "evidence_refs": sorted(branch_refs)},
    }
    if eligible and not model_refs:
        observations.pop("candidate_model_evidence")
        trace_missing = True
    # A skill with no active step is unexercised, not a failed guidance test.
    if skill_eligible and not triggered:
        observations["skill_triggered"]["eligible"] = 0
    if isinstance(contract, Mapping):
        expected = {item["predicate_id"] for item in contract["effect_checks"]}
        result = check_edit_effect_contract(
            contract, scope_digest=scope_digest, child_genome_digest=metadata["genome_digest"],
            observations={key: value for key, value in observations.items() if key in expected}, phase=phase,
        )
    else:
        # An outer proposal already binds immutable parent/child effects, but
        # has no local-parent revision contract. Measure the same registered
        # checks directly without fabricating such a contract or source scope.
        checks = _effect_checks(resolution) if isinstance(resolution, Mapping) else []
        expected = {item["predicate_id"] for item in checks}
        checked, status = _checked_observations(checks, {k: v for k, v in observations.items() if k in expected})
        result = {"schema_version": "ecologyrsi-dsh.edit-effect-receipt/1", "contract_id": None,
                  "scope_digest": scope_digest, "child_genome_digest": metadata.get("genome_digest"),
                  "status": status if checks else "inconclusive", "checks": checked,
                  "qualification": "behavior_effect_only_not_scientific_improvement"}
    if trace_missing:
        result["status"] = "inconclusive"
        result["reason"] = "incomplete_effect_trace"
    if formula_observed:
        result["mean_formula"] = {"policy": "mean-referenced-tools@1", "observed": formula_observed,
                                  "checked": formula_matched,
                                  "status": "passed" if formula_matched else "not_exercised"}
    if isinstance(resolution, Mapping):
        result["resolution_digest"] = resolution["resolution_digest"]
    result.pop("receipt_digest", None)
    return {**result, "receipt_digest": digest(result)}
