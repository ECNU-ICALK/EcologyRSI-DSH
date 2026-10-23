"""Compact, evidence-preserving inputs for newly frozen research policies."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ..core.model_execution_policy import research_execution_policy
from ..core.models import canonical_json, digest
from ..knowledge.autonomous_cycle import GenerationReflection
from ..knowledge.research_iteration import RESEARCH_PLAN_MAX_CHARS, RESEARCH_HOST_EXTENSIONS_MAX_CHARS, ResearchPlanSizeError
from .genome import EcologyEvolutionPluginGenome, TRUST_REGION_MAX_NORMALIZED_STEP, parameter_trust_region_neighborhood


CONCISE_REPORT_LIMITS = {
    "summary_max_chars": 1200,
    "max_evidence_items": 8,
    "finding_max_chars": 400,
    "relevance_max_chars": 240,
    "max_evidence_refs_per_direction": 4,
    "direction_max_chars": {
        "title": 120, "hypothesis": 400, "target_weakness": 240,
        "capability_focus": 160, "expected_tradeoff": 240, "success_criterion": 240,
    },
}


def research_plan_size_feedback(error: ResearchPlanSizeError, output: Mapping[str, Any]) -> dict[str, Any]:
    """Name the space left for model values after Host assembly, without editing them.

    The measured difference includes normalized direction receipts, JSON keys,
    search provenance and diagnostics. Host extensions have a separate bound.
    Reserve a small margin because
    subsequent normalization/diagnostics can change when the model rewrites text.
    Counts are Unicode characters, matching canonical_json, not bytes or tokens.
    """
    fields = {name: len(canonical_json(output[name]))
              for name in ("summary", "evidence", "candidate_directions")}
    authored_chars = sum(fields.values())
    overhead = max(0, error.actual_chars - authored_chars)
    available = max(0, RESEARCH_PLAN_MAX_CHARS - overhead)
    return {
        "maximum_plan_chars": RESEARCH_PLAN_MAX_CHARS,
        "maximum_host_extension_chars": RESEARCH_HOST_EXTENSIONS_MAX_CHARS,
        "maximum_assembled_plan_chars": RESEARCH_PLAN_MAX_CHARS + RESEARCH_HOST_EXTENSIONS_MAX_CHARS,
        "assembled_plan_chars": error.assembled_chars,
        "checked_plan_chars": error.actual_chars,
        "host_extension_chars": error.host_extension_chars,
        "excess_plan_chars": error.actual_chars - RESEARCH_PLAN_MAX_CHARS,
        "model_value_chars": fields,
        "observed_host_and_normalization_chars": overhead,
        "available_model_value_chars": available,
        "recommended_model_value_chars": max(0, available - 512),
        "measurement": "sum_of_canonical_JSON_character_counts_of_summary_evidence_candidate_directions_values",
        "instruction": (
            "The 24000-character plan ceiling includes Host metadata you cannot edit, "
            "except the separately bounded exploration archive and mechanism catalog. "
            "Keep the combined JSON values of summary, evidence and candidate_directions "
            "within recommended_model_value_chars. Shorten repeated prose and evidence "
            "summaries across the whole response; retain all required candidate slots, "
            "exact identifiers, mechanism hypotheses and falsifiable claims. Do not copy "
            "Host archives, catalogs or the parent plan into your output. Return the "
            "complete corrected object; the Host will validate the assembled plan again."
        ),
    }


def research_reflection_reference(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Bind the saved reflection without copying it into every durable plan.

    The researcher still receives the full reflection as a separate input.
    Its original ledger record retains all lessons, directions and outcomes;
    downstream proposals use the newly synthesized directions in this plan.
    """
    if value is None:
        return None
    reflection = GenerationReflection.from_dict(value)
    return {
        "schema_version": "ecologyrsi-dsh.research-reflection-reference/1",
        "run_id": reflection.run_id,
        "generation": reflection.generation,
        "analysis_digest": reflection.analysis_digest,
        "reflection_digest": reflection.reflection_digest,
        "source_payload_digest": digest(reflection.to_dict()),
    }


def _deduplicate_knowledge(knowledge: Mapping[str, Any], search_plan: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(knowledge))
    # Preserve every complete evidence card: its digest remains the digest of
    # the full frozen evidence identity, including metadata-only limitations.
    catalog = {card["knowledge_id"]: card for card in result.get("evidence_catalog", ())}
    for field in ("adopted_knowledge", "research_only_knowledge"):
        if field not in result:
            continue
        projected = []
        for item in result[field]:
            card = catalog.get(item.get("knowledge_id"), {})
            projected.append({key: value for key, value in item.items()
                              if key == "knowledge_id" or key not in card or card[key] != value})
        if projected:
            result[field] = projected
        else:
            result.pop(field)
    if result.get("query_terms") == search_plan.get("search_queries"):
        result.pop("query_terms", None)
        result["query_terms_source"] = "generation_search_plan.search_queries"
    return result


def compact_research_context(context: Mapping[str, Any], *, parent: EcologyEvolutionPluginGenome,
                             parameter_schemas: Mapping[str, Mapping[str, Any]],
                             policy: Mapping[str, Any]) -> dict[str, Any]:
    """Project repeated provenance; never truncate scientific or evidence prose.

    The original validated genome, snapshot and search plan stay in the ledger.
    Their identities and a source-context digest bind this model-facing view.
    This format is never retroactively applied to an unversioned historical run.
    """
    validated_policy = research_execution_policy({"research_execution_policy": policy})
    result = deepcopy(dict(context))
    result["schema_version"] = "ecologyrsi-dsh.research-synthesis-context/2"
    result["source_context_digest"] = digest(context)
    result["research_execution_policy"] = validated_policy
    result.pop("parent_genome", None)
    raw_parent = parent.to_dict()
    result["parent_program"] = {
        "genome_digest": parent.genome_digest,
        "behavior_digest": parent.behavior_digest,
        "scientific_program": raw_parent["scientific_program"],
        "agent_program": raw_parent["agent_program"],
    }
    search_plan = result["generation_search_plan"]
    result["knowledge_snapshot"] = _deduplicate_knowledge(result["knowledge_snapshot"], search_plan)
    # These fields identify the saved plan but add no experiment content. Keep
    # its digest, queries, focus areas, rationale and source evidence identities.
    for field in ("created_at", "model_id", "schema_version", "run_id", "generation"):
        search_plan.pop(field, None)
    forecast = result["forecast_objective"]
    expected_cells = [{"target": target, "horizon_hours": horizon}
                      for target in forecast.get("expected_targets", ())
                      for horizon in forecast.get("expected_horizons_hours", ())]
    if forecast.get("target_horizon_cells") == expected_cells:
        forecast.pop("target_horizon_cells")
        forecast["matrix_definition"] = "Cartesian product of expected_targets and expected_horizons_hours"
    experience = result.get("cross_generation_experience")
    if isinstance(experience, dict):
        experience.pop("capacity", None)  # Projection storage limits, not evidence gates.
    contract = result["synthesis_contract"]
    semantics = contract["predictor_semantics"]
    if semantics.get("scientific_parameter_activity") == contract.get("scientific_parameter_activity"):
        semantics.pop("scientific_parameter_activity", None)
    # Parameters and exact legal one-operation intervals appear together once.
    current_parameters = raw_parent["scientific_program"]["parameter_overrides"]
    rows, extra_schema_fields = {}, {}
    for name, schema in parameter_schemas.items():
        step = parameter_trust_region_neighborhood(name=name, previous=current_parameters[name], contract=schema)
        rows[name] = [schema["type"], schema["minimum"], schema["maximum"], step["minimum"],
                      step["maximum"], step["scale"], True]
        extra = {key: deepcopy(value) for key, value in schema.items() if key not in {"type", "minimum", "maximum"}}
        if extra:
            extra_schema_fields[name] = extra
    contract["scientific_parameter_boundaries"] = {
        "columns": ["type", "registered_minimum", "registered_maximum", "one_step_minimum",
                    "one_step_maximum", "coordinate_scale", "agent_policy_mutation_allowed"],
        "rows_by_parameter": rows,
        "maximum_normalized_step": TRUST_REGION_MAX_NORMALIZED_STEP,
        "current_values_source": "parent_program.scientific_program.parameter_overrides",
        **({"extra_schema_fields": extra_schema_fields} if extra_schema_fields else {}),
    }
    contract["direction_diversity"] = (
        "When diversity_schedule is present, every slot must use its assigned family. "
        "An exploratory hypothesis need not assert an observed weakness. "
        "Distinct directions do not require distinct axes or targets. A parameter target may recur "
        "with a different falsifiable hypothesis; the candidate proposer must choose a distinct legal "
        "behavior under sibling avoidance. Prose is audit-only; do not invent "
        "an instruction-profile weakness merely to fill a slot."
    )
    if contract.get("candidate_portfolio"):
        contract["direction_diversity"] = (
            "Follow candidate_portfolio.slot_requirements. Each direction requires a bounded "
            "mechanism_hypothesis from the Host catalog. Distinct prose, prompt or skill names "
            "do not count as distinct mechanisms. Related joint operations remain allowed. "
            "Mechanism diversity is a search constraint, not evidence of scientific improvement."
        )
    evidence_contract = contract.get("evaluation_evidence_contract", {})
    if evidence_contract.get("selection_evidence_eligible") is True:
        evidence_contract["success_criterion_scope"] = (
            "All directions, including instruction profiles, assess final Agent predictions on the "
            "complete frozen forecast matrix, alongside protocol reliability and resource cost. "
            "Host gates alone determine selection and promotion."
        )
    result["synthesis_report_contract"] = {
        "schema_version": "ecologyrsi-dsh.research-synthesis-report-contract/2",
        "format": "concise@2",
        "length_limits_are_advisory": True,
        "durable_plan_maximum_chars": RESEARCH_PLAN_MAX_CHARS,
        "durable_host_extensions_maximum_chars": RESEARCH_HOST_EXTENSIONS_MAX_CHARS,
        "durable_plan_limit_includes_host_metadata": True,
        "separately_bounded_host_fields": ["exploration_archive", "candidate_portfolio"],
        **deepcopy(CONCISE_REPORT_LIMITS),
        "instruction": (
            "Return the required JSON directly: one brief summary, only evidence needed for the "
            "directions, and short falsifiable direction fields. Evidence may be empty when no source "
            "is cited; do not restate the full catalog, input constraints, or search rationale."
        ),
    }
    for field in ("parent_plan", "previous_generation_analysis", "previous_generation_reflection",
                  "previous_knowledge_assessment", "host_validation_feedback"):
        if result.get(field) is None or result.get(field) == {}:
            result.pop(field, None)
    return result


def research_report_size_diagnostics(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Report verbosity without rejecting otherwise valid scientific content.

    The scientific schema still enforces absolute size limits. Concise targets
    control presentation and context cost; crossing one must not restart an
    expensive research stage or discard a valid, reproducible hypothesis.
    """
    limits = CONCISE_REPORT_LIMITS
    findings = []
    def measure(path: str, item: Any, target: int) -> None:
        if len(item) > target:
            findings.append({"field": path, "actual": len(item), "recommended_maximum": target})
    measure("summary", value["summary"], limits["summary_max_chars"])
    measure("evidence", value["evidence"], limits["max_evidence_items"])
    for index, item in enumerate(value["evidence"]):
        for field in ("finding", "relevance"):
            measure(f"evidence[{index}].{field}", item[field], limits[f"{field}_max_chars"])
    for index, direction in enumerate(value["candidate_directions"]):
        measure(f"candidate_directions[{index}].evidence_refs", direction["evidence_refs"], limits["max_evidence_refs_per_direction"])
        if "mechanism_hypothesis" in direction:
            from ..core.models import canonical_json
            measure(f"candidate_directions[{index}].mechanism_hypothesis",
                    canonical_json(direction["mechanism_hypothesis"]), 800)
        for field, maximum in limits["direction_max_chars"].items():
            measure(f"candidate_directions[{index}].{field}", direction[field], maximum)
    return findings
