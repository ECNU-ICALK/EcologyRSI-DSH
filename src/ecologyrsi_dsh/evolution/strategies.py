"""Small DSH boundary used by the local evolution mode.

The real DSH integration can implement the same protocol later.  The default
adapter only emits structured parameter proposals; it never executes arbitrary
code or receives evaluator internals.
"""

from __future__ import annotations

from .diversity import enabled as diversity_enabled, direction_schedule, validate_direction_schedule
from .effect_contracts import resolve_mutation_effects
from .execution_plan import DerivedExecutionPlan, derive_execution_plan
from .schedule import EVIDENCE_GUIDED_SCHEDULE_SCHEMA
from ..evaluators.sample_execution import SampleExecutionPolicy
from ..evaluators.skill_program import SKILL_POLICY_ID, skill_grammar, skill_preflight_programs

from ..core.prediction_policy import (
    BASELINE_REFERENCE_PREDICTOR_ID, RUNTIME_PREDICTION_POLICY, prediction_usage,
)

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, ClassVar, Protocol, runtime_checkable

from ..core.errors import DshNativeRuntimeUnavailableError, DshResearchOutputBudgetError
from .research_recovery import validate_research_checkpoint
from ..core.model_execution_policy import research_execution_policy
from ..core.models import Proposal, Run, TaskManifest, canonical_json, digest
from ..core.protocols import is_strict_origin_protocol
from ..core.redaction import public_error_summary, public_exception_summary
from ..evaluators.authored_directive import AUTHORED_DIRECTIVE_POLICY_ID
from ..evaluators.greenhouse_prediction import (
    BASELINE_ALIGNED_RIDGE_MODEL_ID,
    BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID,
    EXOGENOUS_RIDGE_MODEL_ID,
    HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    RECIPE_RIDGE_MODEL_ID,
    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
)
from ..integrations.dsh_native_runtime import DSH_NATIVE_EXECUTION_PROTOCOL
from ..integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ..integrations.model_directory import ModelDirectory
from ..knowledge.algorithms import (
    registered_predictor_evaluator_ids,
)
from ..knowledge.autonomous_cycle import (
    AUTONOMOUS_RESEARCH_PROTOCOL,
    CANDIDATE_MUTATION_AXES,
    CandidateDirection,
    GENERATION_REFLECTION_SCHEMA_VERSION,
    # The axis catalog lives in `autonomous_cycle` because that module validates
    # a proposed direction against it and cannot import from here -- this file
    # already imports from there. Re-exported below so both callers can keep
    # reading it off `strategies`, which is where the local-edit side and its
    # tests have always looked for it.
    MUTATION_AXIS_EFFECTS as MUTATION_AXIS_EFFECTS,
    MUTATION_DIRECTIONS_BY_AXIS,
    MUTATION_OPERATION_BY_AXIS,
    SEARCH_PLAN_SCHEMA_VERSION,
    GenerationReflection,
    GenerationSearchPlan,
    mutation_axis_contract,
    normalize_candidate_directions,
    normalize_search_queries,
    reject_executable_fields,
    UnknownEvidenceRefsError,
    validate_research_synthesis,
)
from ..knowledge.models import validate_knowledge_context
from ..knowledge.program_registry import current_program_registry
from ..knowledge.research_iteration import validate_research_plan
from .context import (
    is_sensitive_context_field,
    parameter_semantics,
    safe_aggregate_feedback,
)
from .context import (
    batch_context as validate_batch_context,
)
from .genome import (
    TRUST_REGION_MAX_NORMALIZED_STEP,
    EcologyEvolutionPluginGenome,
    GenomeMutationContextV1,
    apply_genome_mutation,
    mutation_policy_contract,
    parameter_trust_region_neighborhood,
)
from .agent_policy import build_agent_policy
from . import joint_policy, novelty, program_preflight
from ..evaluators.strategy_program import strategy_program_grammar, seed_strategy_program
from .parameter_activity import (
    parameter_activity_contract,
)
from .research_context import compact_research_context, research_report_size_diagnostics, research_reflection_reference
from .workflow_ir import (
    DEFAULT_COMPILER_SEMANTIC_DIGEST,
    EXECUTABLE_WORKFLOW_PARAMETERS,
    compile_plugin_behavior,
    resolve_candidate_agent_profile,
)

_HISTORICAL_PARAMETER_GUARDRAILS_FIELD = "historical_parameter_guardrails"
_HISTORICAL_PARAMETER_GUARDRAIL_POLICY = (
    "preserve_verified_target_horizon_parameters_without_new_aggregate_evidence"
)
_HISTORICAL_PARAMETER_GUARDRAIL_MIN_COHORTS = 2
_HISTORICAL_PARAMETER_GUARDRAIL_MIN_CELL_N_PER_COHORT = 20
_HISTORICAL_PARAMETER_GUARDRAIL_MIN_TOTAL_CELL_N = 40
_HISTORICAL_PARAMETER_GUARDRAIL_COHORT_SCHEMA = (
    "ecologyrsi-dsh.feedback-update-cohort/1"
)
_HISTORICAL_PARAMETER_GUARDRAIL_SELECTION_POLICY = (
    "target_horizon_interleaved_rotating_window@1"
)
_NATIVE_AVOID_BEHAVIOR_LIMIT = 8
_NATIVE_MAX_PROPOSAL_ATTEMPTS = 4
_NATIVE_MAX_SEMANTIC_CONTRACT_ATTEMPTS = 2
# A `max-tokens` terminal is not a repairable output; it is a turn that never
# produced one. The Host owns the only correction that can change the outcome,
# so it is carried through the same bounded repair loop as a semantic rejection.
_OUTPUT_BUDGET_ERROR_CODE = "structured_child_output_budget_exhausted"
_OUTPUT_BUDGET_REJECTION_CODE = "output_budget_exhausted_before_structured_output"
_DEFAULT_SYNTHESIS_MAX_OUTPUT_TOKENS = 8192
_FORECAST_CONTEXT_MAX_TARGETS = 32
_FORECAST_CONTEXT_MAX_HORIZONS = 32
_FORECAST_CONTEXT_MAX_CELLS = 256


def _predictor_semantics(parent: EcologyEvolutionPluginGenome) -> dict[str, Any]:
    """Describe the actual parent; scale controls are not fitted coefficients."""
    program = parent.scientific_program
    predictor_id = program["predictor_ref"]["id"]
    result: dict[str, Any] = {
        "sample_inference": {
            "numerical_prediction_owner": "sample_agent",
            "predictor_scope": "optional candidate default; each sample Agent can select other capabilities or predict directly",
            "evolvable_agent_policy": "candidate-selected instruction Skill and its parameters guide analysis, model use and final prediction",
            "tool_budget_per_origin_attempt": "follow the current frozen prediction_contract.max_prediction_tool_calls and Host remaining_calls; failed calls also consume budget",
            "model_parameters": "Agent may override supported tool parameters within the catalog bounds; only training_fit is used for fitting",
            "scoring_scope": "final Agent vector, not the first numerical tool result",
        },
        "active_predictor_id": predictor_id,
        "numerical_fields_scope": "optional default tool only; never infer final Agent behavior from its residual scales or model family",
        "feature_availability": "only observations available at forecast origin; no future measured forcing",
        "pipeline_selection": "selecting a registered pipeline installs its catalog defaults; it does not automatically tune scale controls",
    }
    scales = {
        BASELINE_ALIGNED_RIDGE_MODEL_ID: "one fixed scale per horizon, shared across targets",
        BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID: "one independent fit-selected-baseline residual scale per target and horizon",
        EXOGENOUS_RIDGE_MODEL_ID: "shared across targets and horizons",
        TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID: "one scale per target, shared across horizons",
        HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID: "one scale per target and horizon",
    }
    if predictor_id in scales:
        parameters = program["parameter_overrides"]
        result.update(
            fit_scope="independent ridge coefficients for every target and horizon in the registered ridge variants",
            shared_hyperparameters=["history_steps", "ridge_alpha"],
            residual_scale_scope=scales[predictor_id],
            residual_scale_values={k: v for k, v in parameters.items()
                                   if k == "residual_scale" or k.endswith("_residual_scale") or k.startswith("residual_scale_")},
            fit_updates_residual_scales=False,
            residual_scale_authority="genome controls the default tool parameters; each sample Agent can override supported call parameters. Fitting ridge coefficients never optimizes these scales",
            variant_fit_parameter_count="at identical history and feature settings, changing residual scale layout does not add fitted ridge coefficients",
            numerical_formula="persistence + genome residual scale * independently fitted ridge residual",
        )
    if predictor_id in (BASELINE_ALIGNED_RIDGE_MODEL_ID, BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID):
        result["numerical_formula"] = "training-fit-selected persistence/seasonal baseline + genome residual scale * ridge residual fitted against that same baseline"
        result["prediction_usage"] = prediction_usage(predictor_id, program["parameter_overrides"])
        result["fit_scope"] = "independent ridge coefficients only for target-horizon cells with a nonzero residual scale; zero-scale cells use the selected baseline and skip residual fitting"
        activity = parameter_activity_contract(parent)
        if activity:
            result["scientific_parameter_activity"] = activity
    return result


def _model_search_cycle_enabled(task: TaskManifest) -> bool:
    return (
        task.metadata.get("execution_protocol") == DSH_NATIVE_EXECUTION_PROTOCOL
        and task.metadata.get("autonomous_research_protocol")
        == AUTONOMOUS_RESEARCH_PROTOCOL
    )


def _forecast_objective_context(task: TaskManifest) -> dict[str, Any]:
    """Expose the complete Host-owned forecast matrix to research roles."""

    fitness_profile = task.metadata.get("fitness_profile")
    if not isinstance(fitness_profile, Mapping):
        fitness_profile = {}

    targets: list[str] = []
    raw_targets = fitness_profile.get("expected_targets")
    if isinstance(raw_targets, Sequence) and not isinstance(
        raw_targets, (str, bytes)
    ):
        for value in raw_targets:
            target = str(value).strip() if isinstance(value, str) else ""
            if target and target not in targets:
                targets.append(target)
            if len(targets) >= _FORECAST_CONTEXT_MAX_TARGETS:
                break

    horizons: list[int | float] = []
    raw_horizons = fitness_profile.get("expected_horizons")
    if isinstance(raw_horizons, Sequence) and not isinstance(
        raw_horizons, (str, bytes)
    ):
        for value in raw_horizons:
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
            ):
                horizon: int | float = (
                    int(value) if float(value).is_integer() else float(value)
                )
                if horizon not in horizons:
                    horizons.append(horizon)
            if len(horizons) >= _FORECAST_CONTEXT_MAX_HORIZONS:
                break

    cells = [
        {"target": target, "horizon_hours": horizon}
        for target in targets
        for horizon in horizons
    ][:_FORECAST_CONTEXT_MAX_CELLS]
    configured_cell_count = task.metadata.get("prediction_cells_per_origin")
    if (
        isinstance(configured_cell_count, bool)
        or not isinstance(configured_cell_count, int)
        or configured_cell_count < 1
    ):
        configured_cell_count = len(cells)

    objective_profile = task.metadata.get("objective_profile")
    raw_weights = (
        objective_profile.get("target_weights")
        if isinstance(objective_profile, Mapping)
        else None
    )
    target_weights: dict[str, float] = {}
    if isinstance(raw_weights, Mapping):
        for target in targets:
            value = raw_weights.get(target)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0
            ):
                target_weights[target] = float(value)

    result: dict[str, Any] = {
        "schema_version": "ecologyrsi-dsh.forecast-objective-context/1",
        "dataset_task": task.to_dict()["metadata"].get("dataset_task"),
        "dataset_ids": list(task.visible_datasets),
        "expected_targets": targets,
        "expected_horizons_hours": horizons,
        "target_horizon_cells": cells,
        "matrix_cell_count": len(cells),
        "prediction_cells_per_origin": configured_cell_count,
        "complete_matrix_required": bool(cells),
        "joint_prediction_call_per_origin": is_strict_origin_protocol(
            task.metadata.get("sample_agent_protocol")
        ),
    }
    dataset_display_name = task.metadata.get("dataset_display_name")
    if isinstance(dataset_display_name, str) and dataset_display_name.strip():
        result["dataset_display_name"] = dataset_display_name.strip()
    if target_weights:
        result["target_weights"] = target_weights
    return result


def _frozen_evidence_refs(
    knowledge: Mapping[str, Any] | None,
) -> set[str]:
    """Return every Host-issued identifier that may be cited by a model."""

    references: set[str] = set()
    if not isinstance(knowledge, Mapping):
        return references
    for item in knowledge.get("evidence_catalog", []):
        if not isinstance(item, Mapping):
            continue
        for name in ("knowledge_id", "evidence_digest", "capability_id"):
            reference = item.get(name)
            if isinstance(reference, str) and reference.strip():
                references.add(reference.strip())
        capability_ids = item.get("capability_ids")
        if isinstance(capability_ids, Sequence) and not isinstance(
            capability_ids, (str, bytes)
        ):
            references.update(
                str(reference).strip()
                for reference in capability_ids
                if isinstance(reference, str) and reference.strip()
            )
    return references


def _semantic_contract_stage_attempt(base_attempt: int, attempt: int) -> int:
    """Allocate a unique one-shot admission fence for an internal repair."""

    return (
        (max(1, int(base_attempt)) - 1) * _NATIVE_MAX_SEMANTIC_CONTRACT_ATTEMPTS
        + attempt
    )


def _host_validation_feedback(
    exc: BaseException,
    *,
    rejected_output: Mapping[str, Any],
    attempt: int,
) -> dict[str, Any]:
    """Build bounded, non-secret feedback for one fresh model correction."""

    detail = public_error_summary(str(exc), limit=500)
    feedback: dict[str, Any] = {
        "schema_version": "ecologyrsi-dsh.host-validation-feedback/1",
        "rejection_code": "semantic_contract_validation_failed",
        "validation_detail": detail or public_exception_summary(exc),
        "rejected_output_digest": digest(rejected_output),
        "rejected_attempt": attempt,
        "required_action": (
            "Return a fresh complete object that corrects the stated Host "
            "validation failure; do not repeat the rejected output."
        ),
    }
    # A rejected identifier is the model's own output and already sits in this
    # transcript, so echoing a bounded copy adds no exposure while turning an
    # unactionable rule restatement into a named correction. The free-text
    # summary above cannot carry them: `public_error_summary` redacts long
    # alphanumeric runs, which is exactly the shape of an evidence digest.
    rejected_refs = tuple(getattr(exc, "rejected_refs", ()) or ())[:8]
    if rejected_refs:
        feedback["rejected_evidence_refs"] = [
            str(reference)[:160] for reference in rejected_refs
        ]
        feedback["required_action"] = (
            "Return a fresh complete object. Every evidence_ref must be copied "
            "exactly from allowed_evidence_refs; drop or replace each entry in "
            "rejected_evidence_refs and do not repeat the rejected output."
        )
    return feedback


def _output_budget_feedback(*, attempt: int, max_output_tokens: int | None = None) -> dict[str, Any]:
    """Turn one budget-exhausted turn into a correctable Host rejection.

    A `max-tokens` terminal means the child spent its whole single-call output
    budget reasoning and never called `structured_output`, so there is no
    rejected output to digest and nothing the model can learn from the failure
    alone.  Identifying the exhausted budget and the submission ordering changes the
    request, which is what `retry_identical_exhausted_request: False` requires:
    the frozen policy forbids replaying the same request, not repairing it.
    """

    budget_label = f"{max_output_tokens}-token " if max_output_tokens is not None else ""
    return {
        "schema_version": "ecologyrsi-dsh.host-validation-feedback/1",
        "rejection_code": _OUTPUT_BUDGET_REJECTION_CODE,
        "validation_detail": (
            f"The previous attempt spent its entire {budget_label}"
            "single-response output budget on reasoning and ended without "
            "calling structured_output, so it produced no result."
        ),
        "rejected_attempt": attempt,
        **({"output_token_budget": max_output_tokens} if max_output_tokens is not None else {}),
        "required_action": (
            "Call structured_output first with one complete object, then stop. "
            "Keep every field at its concise target and do not draft or "
            "restate the analysis in prose before that call."
        ),
    }


def _native_evolution_reflection_from_experience(
    experience: Mapping[str, Any] | None,
    *,
    current_run_id: str,
) -> dict[str, Any]:
    """Project candidate-specific failures as scoped research advice.

    A previous cohort is not the next experiment's comparison context. Past
    scientific failures cannot grant a permanent veto over a valid program.
    """

    safe_experience = safe_aggregate_feedback(
        experience,
        name="DSH-native cross-generation experience",
    )
    if safe_experience is None:
        safe_experience = {}
    avoid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scope, field_name in (
        ("current_run", "generations"),
        ("historical_run", "historical_generations"),
    ):
        rows = safe_experience.get(field_name)
        if not isinstance(rows, list):
            continue
        for row in reversed(rows):
            if not isinstance(row, Mapping):
                continue
            if row.get("improved") is True or row.get("outcome") == "promoted":
                continue
            gate_result = row.get("gate_result")
            if (
                isinstance(gate_result, Mapping)
                and gate_result.get("insufficient_evidence") is True
            ):
                # Diagnostic cohorts still inform the next hypothesis, but
                # they cannot justify a durable exact-behavior blacklist.
                continue
            modifications = row.get("modifications")
            if not isinstance(modifications, Mapping):
                continue
            raw_behaviors = modifications.get("candidate_behaviors")
            if not isinstance(raw_behaviors, list):
                continue
            failures = row.get("common_failures")
            reason = (
                str(failures[0])
                if isinstance(failures, list) and failures
                else str(row.get("outcome") or "not_improved")
            )
            source_run_id = row.get("source_run_id")
            if not isinstance(source_run_id, str) or not source_run_id.strip():
                source_run_id = current_run_id
            source_generation = row.get("source_generation", row.get("generation"))
            for behavior in raw_behaviors:
                if not isinstance(behavior, Mapping):
                    continue
                if behavior.get("classification") != "scientific_gate_failed":
                    continue
                behavior_digest = behavior.get("behavior_digest")
                if (
                    not isinstance(behavior_digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", behavior_digest)
                    or behavior_digest in seen
                ):
                    continue
                seen.add(behavior_digest)
                item: dict[str, Any] = {
                    "behavior_digest": behavior_digest,
                    "reason": str(behavior.get("classification") or reason),
                    "source_scope": scope,
                    "source_run_id": source_run_id,
                    "source_generation": source_generation,
                }
                predictor_id = behavior.get("prediction_model_id")
                if isinstance(predictor_id, str) and predictor_id.strip():
                    item["prediction_model_id"] = predictor_id.strip()
                parameters_digest = behavior.get("parameters_digest")
                if isinstance(parameters_digest, str) and re.fullmatch(
                    r"[0-9a-f]{64}", parameters_digest
                ):
                    item["parameters_digest"] = parameters_digest
                avoid.append(item)
                if len(avoid) >= _NATIVE_AVOID_BEHAVIOR_LIMIT:
                    break
            if len(avoid) >= _NATIVE_AVOID_BEHAVIOR_LIMIT:
                break
        if len(avoid) >= _NATIVE_AVOID_BEHAVIOR_LIMIT:
            break
    active_unresolved = safe_experience.get("active_unresolved")
    return {
        "schema_version": "ecologyrsi-dsh.evolution-reflection/3",
        "avoid_behaviors": [],
        "review_behaviors": avoid,
        "active_unresolved": (
            list(active_unresolved[:8])
            if isinstance(active_unresolved, list)
            else []
        ),
        "policy": {
            "exact_failed_behavior_replay": "review_evidence_and_applicability",
            "host_enforced": False,
            "maximum_avoid_behaviors": _NATIVE_AVOID_BEHAVIOR_LIMIT,
        },
    }


def _historical_parameter_guardrails_from_experience(
    experience: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(experience, Mapping):
        return None
    raw = experience.get(_HISTORICAL_PARAMETER_GUARDRAILS_FIELD)
    if not isinstance(raw, Mapping):
        return None
    return safe_aggregate_feedback(
        raw,
        name="historical parameter guardrails",
    )


def _host_guardrail_plan(
    plan: Mapping[str, Any],
    guardrails: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(plan)
    result.pop(_HISTORICAL_PARAMETER_GUARDRAILS_FIELD, None)
    if guardrails is not None:
        result[_HISTORICAL_PARAMETER_GUARDRAILS_FIELD] = deepcopy(dict(guardrails))
    return result








@runtime_checkable
class DSHAdapter(Protocol):
    """The only interface the trusted director needs from DSH."""

    def open_session(self, run: Run, task: TaskManifest) -> str:
        """Open a short-lived DSH conversation for one run."""

    def propose(
        self,
        run: Run,
        task: TaskManifest,
        session_id: str,
        *,
        parent_candidate_id: str | None = None,
        parent_context: Mapping[str, Any] | None = None,
        batch_context: Mapping[str, Any] | None = None,
    ) -> Proposal:
        """Return one structured proposal for the current generation."""

    def close_session(self, session_id: str) -> None:
        """Release adapter-side session state."""






def _registered_scalar_axis_targets(
    genome: EcologyEvolutionPluginGenome, axis: str
) -> tuple[str, ...]:
    """The legal targets on one registry-bounded scalar axis.

    Shared by the generation-level proposer and the batch-local editor so the two
    can never advertise a different set of knobs for the same axis. Both filters
    below are deliberate:

    * ``workflow_parameter`` is narrowed to the parameters a Host component
      actually reads. The template also declares ``max_concurrent`` and
      ``wave_size``, which nothing consumes; advertising them spends a
      proposal's single operation on an edit no measurement can attribute.
    * ``instruction_parameter`` is read off the sample-planner profile, which is
      the role both mutation paths pin. ``sample-repair@1`` declares no
      parameters, so this is the whole reachable set rather than a restriction.
    """

    execution = genome.agent_program["candidate_execution_program"]
    registry = current_program_registry()
    if axis == "workflow_parameter":
        template = registry.program(
            "workflow_templates", str(execution["workflow_template_ref"]["id"])
        )
        return tuple(
            sorted(
                name
                for name in template["parameters"]
                if name in EXECUTABLE_WORKFLOW_PARAMETERS
            )
        )
    if axis == "instruction_parameter":
        profile = next(
            (
                item
                for item in execution["role_profiles"]
                if item["role"] == "sample-planner"
            ),
            None,
        )
        if profile is None:
            return ()
        template = registry.program(
            "instruction_templates", str(profile["instruction_template_ref"]["id"])
        )
        return tuple(sorted(template["parameters"]))
    raise ValueError(f"unsupported registered scalar mutation axis: {axis}")


def _registered_scalar_axis_contract(
    parent: EcologyEvolutionPluginGenome, axis: str, target: str
) -> tuple[Mapping[str, Any] | None, Any]:
    """The registered contract and the parent's effective value for one axis.

    Covers the two scalar axes whose bounds live in the program registry rather
    than in the frozen parameter boundary. The effective value falls back to the
    registered default when the genome carries no override, and that fallback is
    load-bearing rather than defensive: ``select_instruction_template`` clears a
    role profile's ``instruction_parameters`` outright, so right after a template
    switch the default *is* the value the run would use.

    Returns ``(None, None)`` when the axis or the target is not something this
    parent can express, which every caller reads as "no legal move here".
    """

    execution = parent.to_dict()["agent_program"]["candidate_execution_program"]
    registry = current_program_registry()
    if axis == "workflow_parameter":
        template = registry.program(
            "workflow_templates", str(execution["workflow_template_ref"]["id"])
        )
        overrides = execution["workflow_overrides"]
    elif axis == "instruction_parameter":
        profile = next(
            (
                item
                for item in execution["role_profiles"]
                if item["role"] == "sample-planner"
            ),
            None,
        )
        if profile is None:
            return None, None
        template = registry.program(
            "instruction_templates", str(profile["instruction_template_ref"]["id"])
        )
        overrides = profile["instruction_parameters"]
    else:
        return None, None
    contract = template["parameters"].get(target)
    if not isinstance(contract, Mapping):
        return None, None
    return contract, overrides.get(target, contract["default"])


def _scalar_axis_preflight_schema(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a registry parameter contract into a trust-region schema.

    The registry declares integrality as ``integer: bool``; the witness generator
    reads ``type == "integer"``. Doing the translation once means the registered
    axes get exactly the witnesses the scientific axis gets, including the
    integer enumeration that makes a bounded ``max_attempts`` step checkable.
    """

    return {
        "minimum": contract["minimum"],
        "maximum": contract["maximum"],
        "type": "integer" if contract.get("integer") else "number",
    }


def _assigned_direction_baseline(
    parent: EcologyEvolutionPluginGenome, axis: str, target: str
) -> Any:
    """The value an ``increase``/``decrease`` direction has to move away from.

    ``scientific_parameter`` deliberately does not get the registered-default
    fallback the other scalar axes get: its overrides are fully materialized by
    the seed genome, so a missing one means something is wrong with the parent
    rather than that a default applies.
    """

    if axis == "scientific_parameter":
        return parent.to_dict()["scientific_program"]["parameter_overrides"].get(
            target
        )
    return _registered_scalar_axis_contract(parent, axis, target)[1]


def _has_authored_directive(
    parent: EcologyEvolutionPluginGenome, role: str
) -> bool:
    """Whether the parent already carries an authored directive for ``role``.

    The field is optional and omitted when absent, so its presence is exactly
    the question ``author`` versus ``revise`` asks.
    """

    profiles = parent.to_dict()["agent_program"]["candidate_execution_program"][
        "role_profiles"
    ]
    return any(
        profile.get("role") == role and "authored_directive" in profile
        for profile in profiles
    )


class StrategyRouterDSHAdapter:
    """Route a task to bounded parameter proposal strategies.

    DSH owns the model calls; the Host validates registered mutation targets,
    numerical boundaries and the immutable scientific contract.
    """

    SUPPORTED_STRATEGIES = ("autonomous_model@1",)
    _STRATEGY_DESCRIPTORS: ClassVar[dict[str, dict[str, Any]]] = {
        "autonomous_model@1": {
            "label": "模型自主调研与进化",
            "description": (
                "由策略模型检索公开知识、组建模型团队并提出预测模型与搜索策略；"
                "宿主只执行已登记且有界的参数方案。"
            ),
            "requires_authenticated_model": True,
            "implementation": "model-search-reflect-candidate-directions/20",
        },
    }

    _GREENHOUSE_RIDGE_SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "history_steps": {"type": "integer", "minimum": 1, "maximum": 12},
        "ridge_alpha": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
        "residual_scale": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    _GREENHOUSE_ALIGNED_RIDGE_SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "history_steps": {"type": "integer", "minimum": 1, "maximum": 12},
        "ridge_alpha": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
        **{f"residual_scale_{horizon}h": {"type": "number", "minimum": 0.0, "maximum": 1.0} for horizon in (1, 6, 24)},
    }
    _GREENHOUSE_TARGETWISE_RIDGE_SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "history_steps": {"type": "integer", "minimum": 1, "maximum": 12},
        "ridge_alpha": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
        "air_temperature_residual_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "relative_humidity_residual_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "co2_concentration_residual_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    }
    _GREENHOUSE_HORIZON_TARGETWISE_RIDGE_SCHEMAS: ClassVar[
        dict[str, dict[str, Any]]
    ] = {
        "history_steps": {"type": "integer", "minimum": 1, "maximum": 12},
        "ridge_alpha": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
        "air_temperature_1h_residual_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "air_temperature_6h_residual_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "air_temperature_24h_residual_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "relative_humidity_1h_residual_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "relative_humidity_6h_residual_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "relative_humidity_24h_residual_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "co2_concentration_1h_residual_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "co2_concentration_6h_residual_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "co2_concentration_24h_residual_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    }
    # Every tunable of the recipe predictor lives inside its declarative
    # feature_recipe, and evaluators.registry.validate_parameter_overrides
    # refuses scalar overrides for it outright. So this boundary is empty on
    # purpose: it advertises no knob the runtime cannot read, which is also
    # what keeps _genome_parameter_boundary from offering a recipe genome a
    # scientific_parameter target that no evaluator would accept. The single
    # empty sweep row keeps the schedule arithmetic at _schedule_index well
    # defined -- without it a recipe run divided by zero before proposing.
    _GREENHOUSE_RECIPE_RIDGE_SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {}

    def __init__(
        self,
        gateway: ModelDirectory | None = None,
        *,
        max_proposals: int = 256,
        native_runtime_provider: Callable[[], Any] | None = None,
        candidate_novelty_provider: Callable[[str, int], Mapping[str, Any]] | None = None,
    ) -> None:
        if isinstance(max_proposals, bool) or not isinstance(max_proposals, int) or max_proposals < 1:
            raise ValueError("max_proposals must be a positive integer")
        self.gateway = gateway or ModelDirectory.from_env()
        self.max_proposals = min(max_proposals, 256)
        self._session_counts: dict[tuple[str, int], int] = {}
        self._native_runtime_provider = native_runtime_provider
        self._candidate_novelty_provider = candidate_novelty_provider

    def _candidate_novelty_archive(self, run, task, reference):
        if not novelty.enabled(task.metadata):
            return None
        if self._candidate_novelty_provider is None:
            raise ValueError("candidate novelty archive provider is not configured")
        archive = novelty.validate_archive(
            self._candidate_novelty_provider(run.run_id, run.generation), run=run, task=task,
        )
        if reference != novelty.archive_reference(archive):
            raise ValueError("candidate novelty archive differs from the frozen research reference")
        return archive

    def _native_runtime(self) -> DshStructuredRoleRuntime:
        client = self._native_runtime_provider() if self._native_runtime_provider else None
        if client is None:
            raise RuntimeError("DSH-native Agent runtime is not configured")
        if isinstance(client, DshStructuredRoleRuntime):
            return client
        return DshStructuredRoleRuntime(client)

    @staticmethod
    def _native_stage_identity(
        parent: EcologyEvolutionPluginGenome,
        task: TaskManifest,
        run: Run,
        stage: str,
        context: Mapping[str, Any],
    ) -> dict[str, str]:
        compiled = compile_plugin_behavior(
            parent,
            task,
            None,
            current_program_registry(),
            compiler_semantic_digest=str(
                task.metadata.get("compiler_semantic_digest")
                or DEFAULT_COMPILER_SEMANTIC_DIGEST
            ),
        )
        return {
            "genome_digest": parent.genome_digest,
            "compiled_behavior_digest": compiled.compiled_behavior_digest,
            "phenotype_instance_digest": digest(
                {
                    "schema_version": "ecologyrsi-dsh.role-stage-instance/1",
                    "run_id": run.run_id,
                    "generation": run.generation,
                    "stage": stage,
                    "source_genome_digest": parent.genome_digest,
                    "compiled_behavior_digest": compiled.compiled_behavior_digest,
                    "context_digest": digest(context),
                }
            ),
        }

    @classmethod
    def configuration_digest(cls, strategy_id: str) -> str:
        # ``implementation`` is a durable replay boundary. Any proposal,
        # parent-selection, or feedback behavior change must bump its version.
        try:
            descriptor = cls._STRATEGY_DESCRIPTORS[strategy_id]
        except KeyError:
            raise ValueError(f"unknown strategy_id: {strategy_id}") from None
        return digest(
            {
                "strategy_id": strategy_id,
                "implementation": descriptor["implementation"],
                "host_parameter_boundary": "prediction-model-specific/1",
                "mutation_policy": mutation_policy_contract(),
            }
        )

    @classmethod
    def catalog(cls) -> list[dict[str, Any]]:
        return [
            {
                "id": strategy_id,
                **{
                    key: value
                    for key, value in cls._STRATEGY_DESCRIPTORS[strategy_id].items()
                    if key != "implementation"
                },
                "configuration_digest": cls.configuration_digest(strategy_id),
            }
            for strategy_id in cls.SUPPORTED_STRATEGIES
        ]

    @classmethod
    def parameter_schemas_for_task(
        cls,
        task: TaskManifest,
        current_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return an isolated copy of the active predictor parameter boundary."""

        _boundary, schemas = _task_parameter_boundary(task, current_plan)
        return deepcopy(dict(schemas))

    @classmethod
    def parameter_semantics_for_task(
        cls,
        task: TaskManifest,
        current_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        """Return host-authored meanings for the active predictor parameters."""

        boundary, schemas = _task_parameter_boundary(task, current_plan)
        return parameter_semantics(boundary, schemas)

    @property
    def connection_catalog(self) -> list[dict[str, Any]]:
        """Expose redacted policy-model connection state for diagnostics."""

        return self.gateway.catalog()

    def open_session(self, run: Run, task: TaskManifest) -> str:
        if task.metadata.get("execution_protocol") != DSH_NATIVE_EXECUTION_PROTOCOL:
            raise ValueError("only the native execution protocol is supported")
        session_id = f"dsh-native:{run.run_id}"
        self._session_counts.setdefault((session_id, run.generation), 0)
        return session_id

    def plan_generation_search(
        self,
        *,
        run: Run,
        task: TaskManifest,
        parent_genome: Mapping[str, Any],
        previous_generation_analysis: Mapping[str, Any] | None,
        previous_generation_reflection: Mapping[str, Any] | None,
        current_plan: Mapping[str, Any],
        host_query_hints: Sequence[str],
        candidate_count: int,
        run_state_revision: int,
        stage_attempt: int,
        ledger_expected_revision: int,
    ) -> GenerationSearchPlan:
        """Ask the strategy model what to retrieve before the Host goes online."""

        if not _model_search_cycle_enabled(task):
            raise ValueError("model-directed generation search is not enabled")
        if isinstance(candidate_count, bool) or not 1 <= candidate_count <= 8:
            raise ValueError("candidate_count must be between 1 and 8")
        parent = EcologyEvolutionPluginGenome.from_dict(parent_genome)
        previous_analysis = safe_aggregate_feedback(
            previous_generation_analysis,
            name="search-plan previous generation analysis",
        )
        previous_reflection = safe_aggregate_feedback(
            previous_generation_reflection,
            name="search-plan previous generation reflection",
        )
        base_context = {
            "schema_version": "ecologyrsi-dsh.search-planning-context/1",
            "generation": run.generation,
            "objective": task.objective,
            "forecast_objective": _forecast_objective_context(task),
            "research_domain": task.metadata.get(
                "research_domain", task.domain_pack
            ),
            "candidate_count": candidate_count,
            "parent_genome": parent.to_dict(),
            **_inherited_strategy_context(parent, task),
            "previous_generation_analysis": previous_analysis,
            "previous_generation_reflection": previous_reflection,
            "current_research_plan": safe_aggregate_feedback(
                current_plan,
                name="search-plan current research plan",
            ),
            "host_query_hints": list(normalize_search_queries(host_query_hints)),
            "retrieval_contract": {
                "provider": "OpenAlex metadata and built-in verified catalog",
                "maximum_queries": 6,
                "maximum_query_characters": 180,
                "maximum_focus_areas": 8,
                "maximum_focus_area_characters": 240,
                "external_code_execution": False,
                "full_text_fetch": False,
                "query_is_advisory_until_host_validation": True,
            },
        }
        model_id = str(
            task.metadata.get("strategy_model_id")
            or task.metadata.get("policy_model_id")
            or ""
        ).strip() or None
        feedback: dict[str, Any] | None = None
        for contract_attempt in range(
            1,
            _NATIVE_MAX_SEMANTIC_CONTRACT_ATTEMPTS + 1,
        ):
            context = {
                **deepcopy(base_context),
                "host_validation_feedback": deepcopy(feedback),
            }
            dsh_stage_attempt = _semantic_contract_stage_attempt(
                stage_attempt,
                contract_attempt,
            )
            structured = self._native_runtime().run(
                run_id=run.run_id,
                stage="generation.search-plan",
                role="researcher",
                context=context,
                output_schema_id="ecology-research-search-plan@1",
                run_state_revision=run_state_revision,
                stage_attempt=dsh_stage_attempt,
                ledger_expected_revision=ledger_expected_revision,
                idempotency_key=(
                    f"{run.run_id}:generation:{run.generation}:search-plan:"
                    f"attempt:{dsh_stage_attempt}"
                ),
                identity_digests=self._native_stage_identity(
                    parent, task, run, "generation.search-plan", context
                ),
            )
            try:
                reject_executable_fields(structured)
                if set(structured) != {
                    "schema_version",
                    "search_queries",
                    "focus_areas",
                    "rationale",
                }:
                    raise ValueError(
                        "DSH search planner returned unsupported fields"
                    )
                if structured.get("schema_version") != SEARCH_PLAN_SCHEMA_VERSION:
                    raise ValueError(
                        "DSH search planner returned an unsupported schema"
                    )
                return GenerationSearchPlan(
                    run_id=run.run_id,
                    generation=run.generation,
                    search_queries=normalize_search_queries(
                        structured.get("search_queries")
                    ),
                    focus_areas=tuple(structured.get("focus_areas") or ()),
                    rationale=structured.get("rationale"),
                    source_analysis_digest=(
                        previous_analysis.get("analysis_digest")
                        if isinstance(previous_analysis, Mapping)
                        else None
                    ),
                    source_reflection_digest=(
                        previous_reflection.get("reflection_digest")
                        if isinstance(previous_reflection, Mapping)
                        else None
                    ),
                    model_id=model_id,
                )
            except (TypeError, ValueError) as exc:
                if contract_attempt >= _NATIVE_MAX_SEMANTIC_CONTRACT_ATTEMPTS:
                    raise
                feedback = _host_validation_feedback(
                    exc,
                    rejected_output=structured,
                    attempt=contract_attempt,
                )
        raise RuntimeError("generation search-plan repair loop did not terminate")

    def reflect_generation(
        self,
        *,
        run: Run,
        task: TaskManifest,
        parent_genome: Mapping[str, Any],
        generation_analysis: Mapping[str, Any],
        knowledge_snapshot: Mapping[str, Any] | None,
        research_iteration: Mapping[str, Any] | None,
        candidate_outcomes: Sequence[Mapping[str, Any]],
        direction_count: int,
        run_state_revision: int,
        stage_attempt: int,
        ledger_expected_revision: int,
    ) -> GenerationReflection:
        """Run one aggregate-only batch reflection after all candidates settle."""

        if not _model_search_cycle_enabled(task):
            raise ValueError("batch-level model reflection is not enabled")
        if isinstance(direction_count, bool) or not 2 <= direction_count <= 8:
            raise ValueError("direction_count must be between 2 and 8")
        parent = EcologyEvolutionPluginGenome.from_dict(parent_genome)
        analysis = safe_aggregate_feedback(
            generation_analysis,
            name="generation reflection analysis",
        )
        if analysis is None:
            raise ValueError("generation reflection requires aggregate analysis")
        knowledge = validate_knowledge_context(knowledge_snapshot)
        safe_research_iteration = safe_aggregate_feedback(
            research_iteration,
            name="generation reflection research iteration",
        )
        mutation_catalog = _mutation_contract_catalog(task, parent)
        allowed_mutation_targets = mutation_catalog["allowed_mutation_targets"]
        allowed_evidence_refs = _frozen_evidence_refs(knowledge)
        base_context = {
            "schema_version": "ecologyrsi-dsh.generation-reflection-context/1",
            "generation": run.generation,
            "objective": task.objective,
            "forecast_objective": _forecast_objective_context(task),
            "direction_count": direction_count,
            "generation_analysis": analysis,
            "predictor_semantics_scope": "generation_parent",
            "predictor_semantics": _predictor_semantics(parent),
            "candidate_outcomes": [
                safe_aggregate_feedback(
                    item,
                    name="generation reflection candidate outcome",
                )
                for item in candidate_outcomes
            ],
            "knowledge_snapshot": knowledge,
            "research_iteration": safe_research_iteration,
            "host_boundary": {
                "aggregate_results_only": True,
                "reflection_is_advisory": True,
                "mechanistic_explanations_are_hypotheses": True,
                "host_predictor_semantics_are_facts": True,
                "next_generation_research_synthesis_preflight_required": True,
                "candidate_directions_must_be_distinct": True,
                **deepcopy(mutation_catalog),
                "cite_only_frozen_evidence": True,
                # The Host rejects any evidence_ref outside this set, so the set
                # itself has to be in the prompt: reflection used to see only the
                # rule plus `allowed_mutation_targets`, and both attempts of
                # run:e4332050-18c1-4562-8f03-3c4c8ee3a8bf cited a mutation
                # target as an evidence_ref -- the one identifier list they could
                # actually see. Everything here is already frozen knowledge.
                "allowed_evidence_refs": sorted(allowed_evidence_refs),
                "mutation_targets_are_not_evidence_refs": True,
                "registered_capabilities_only": True,
                "model_generated_code_execution": False,
                "selection_and_stopping_are_host_controlled": True,
            },
        }
        model_id = str(
            task.metadata.get("review_model_id")
            or task.metadata.get("judge_model_id")
            or ""
        ).strip() or None
        feedback: dict[str, Any] | None = None
        for contract_attempt in range(
            1,
            _NATIVE_MAX_SEMANTIC_CONTRACT_ATTEMPTS + 1,
        ):
            context = {
                **deepcopy(base_context),
                "host_validation_feedback": deepcopy(feedback),
            }
            dsh_stage_attempt = _semantic_contract_stage_attempt(
                stage_attempt,
                contract_attempt,
            )
            structured = self._native_runtime().run(
                run_id=run.run_id,
                stage="generation.reflect",
                role="generation-judge",
                context=context,
                output_schema_id="ecology-generation-reflection@1",
                run_state_revision=run_state_revision,
                stage_attempt=dsh_stage_attempt,
                ledger_expected_revision=ledger_expected_revision,
                idempotency_key=(
                    f"{run.run_id}:generation:{run.generation}:reflection:"
                    f"attempt:{dsh_stage_attempt}"
                ),
                identity_digests=self._native_stage_identity(
                    parent, task, run, "generation.reflect", context
                ),
            )
            try:
                reject_executable_fields(structured)
                if set(structured) != {
                    "schema_version",
                    "summary",
                    "lessons",
                    "recommended_search_queries",
                    "candidate_directions",
                    "stop_recommendation",
                }:
                    raise ValueError(
                        "DSH generation reflector returned unsupported fields"
                    )
                if (
                    structured.get("schema_version")
                    != GENERATION_REFLECTION_SCHEMA_VERSION
                ):
                    raise ValueError(
                        "DSH generation reflector returned an unsupported schema"
                    )
                directions = normalize_candidate_directions(
                    structured.get("candidate_directions"),
                    exact_items=direction_count,
                    minimum_items=2,
                    allowed_mutation_targets=allowed_mutation_targets,
                )
                for direction in directions:
                    unknown = tuple(
                        reference
                        for reference in direction.evidence_refs
                        if reference not in allowed_evidence_refs
                    )
                    if unknown:
                        raise UnknownEvidenceRefsError(
                            "generation reflection cited evidence outside the "
                            "frozen snapshot",
                            rejected_refs=unknown,
                        )
                return GenerationReflection(
                    run_id=run.run_id,
                    generation=run.generation,
                    analysis_digest=str(analysis.get("analysis_digest") or ""),
                    summary=structured.get("summary"),
                    lessons=tuple(structured.get("lessons") or ()),
                    recommended_search_queries=normalize_search_queries(
                        structured.get("recommended_search_queries")
                    ),
                    candidate_directions=directions,
                    stop_recommendation=structured.get("stop_recommendation"),
                    model_id=model_id,
                )
            except (TypeError, ValueError) as exc:
                if contract_attempt >= _NATIVE_MAX_SEMANTIC_CONTRACT_ATTEMPTS:
                    raise
                feedback = _host_validation_feedback(
                    exc,
                    rejected_output=structured,
                    attempt=contract_attempt,
                )
        raise RuntimeError("generation reflection repair loop did not terminate")

    def research_plan(
        self,
        model_id: str,
        *,
        run: Run,
        task: TaskManifest,
        parameter_schemas: Mapping[str, Mapping[str, Any]] | None = None,
        previous_generation_analysis: Mapping[str, Any] | None = None,
        knowledge_snapshot: Mapping[str, Any] | None = None,
        previous_knowledge_assessment: Mapping[str, Any] | None = None,
        current_plan: Mapping[str, Any] | None = None,
        cross_generation_experience: Mapping[str, Any] | None = None,
        parent_genome: Mapping[str, Any] | None = None,
        generation_search_plan: Mapping[str, Any] | None = None,
        previous_generation_reflection: Mapping[str, Any] | None = None,
        candidate_count: int | None = None,
        run_state_revision: int = 0,
        stage_attempt: int = 1,
        ledger_expected_revision: int = 0,
        research_recovery: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if task.metadata.get("execution_protocol") != DSH_NATIVE_EXECUTION_PROTOCOL:
            raise ValueError("Only DSH-native research is supported")
        if not isinstance(parent_genome, Mapping):
            raise ValueError("DSH-native research requires the frozen parent genome")
        parent = EcologyEvolutionPluginGenome.from_dict(parent_genome)
        novelty_archive = self._candidate_novelty_archive(
            run, task, (cross_generation_experience or {}).get(novelty.ARCHIVE_KEY),
        )
        context = {
            "generation": run.generation,
            "objective": task.objective,
            "forecast_objective": _forecast_objective_context(task),
            "parent_genome": parent.to_dict(),
        **_inherited_strategy_context(parent, task),
            "parent_plan": dict(current_plan or {}),
            "previous_generation_analysis": previous_generation_analysis,
            "knowledge_snapshot": knowledge_snapshot,
            "previous_knowledge_assessment": previous_knowledge_assessment,
            "cross_generation_experience": cross_generation_experience,
            **({novelty.ARCHIVE_KEY: novelty.context_view(novelty_archive)} if novelty_archive is not None else {}),
        }
        active_search_cycle = _model_search_cycle_enabled(task)
        if not active_search_cycle:
            raise ValueError("The model-directed research cycle is required")
        if (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or not 1 <= candidate_count <= 8
        ):
            raise ValueError(
                "model-directed research requires candidate_count between 1 and 8"
            )
        if not isinstance(generation_search_plan, Mapping):
            raise ValueError(
                "model-directed research requires a frozen generation search plan"
            )
        frozen_search_plan = GenerationSearchPlan.from_dict(
            generation_search_plan
        )
        if (
            frozen_search_plan.run_id != run.run_id
            or frozen_search_plan.generation != run.generation
        ):
            raise ValueError("generation search plan is outside the run scope")
        knowledge = validate_knowledge_context(knowledge_snapshot)
        if knowledge is None:
            raise ValueError("research synthesis requires frozen knowledge")
        mutation_catalog = _mutation_contract_catalog(task, parent)
        allowed_mutation_targets = mutation_catalog[
            "allowed_mutation_targets"
        ]
        family_schedule = direction_schedule(task.metadata, allowed_mutation_targets, run.generation, candidate_count)
        allowed_evidence_refs = _frozen_evidence_refs(knowledge)
        evolution_reflection = _native_evolution_reflection_from_experience(
            cross_generation_experience,
            current_run_id=run.run_id,
        )
        context.update(
            {
                "knowledge_snapshot": knowledge,
                "generation_search_plan": frozen_search_plan.to_dict(),
                "previous_generation_reflection": safe_aggregate_feedback(
                    previous_generation_reflection,
                    name="research previous generation reflection",
                ),
                "required_candidate_direction_count": candidate_count,
                "synthesis_contract": {
                    "one_distinct_direction_per_candidate_slot": True,
                    "exactly_one_operation_per_direction": not joint_policy.enabled(task.metadata),
                    "one_operation_must_fully_test_the_hypothesis": not joint_policy.enabled(task.metadata),
                    "maximum_related_operations": mutation_policy_contract(task.metadata)["candidate_maximum_operations"],
                    **deepcopy(mutation_catalog),
                    **({"diversity_schedule": family_schedule} if family_schedule else {}),
                    "prior_failure_advice": deepcopy(
                        evolution_reflection["review_behaviors"]
                    ),
                    "diagnostic_insufficient_evidence_is_advisory": True,
                    "cite_only_frozen_evidence": True,
                    # Same reason as the reflection boundary: the Host
                    # validates every evidence_ref against this set, so
                    # the model has to be able to read it.
                    "allowed_evidence_refs": sorted(allowed_evidence_refs),
                    "mutation_targets_are_not_evidence_refs": True,
                    "registered_capabilities_only": True,
                    "model_generated_code_execution": False,
                    "predictor_semantics": _predictor_semantics(parent),
                },
            }
        )
        stage = "generation.research-synthesis"
        output_schema_id = "ecology-research-synthesis@1"
        reflection_reference = research_reflection_reference(
            previous_generation_reflection
        )
        historical_guardrails = _historical_parameter_guardrails_from_experience(
            cross_generation_experience
        )
        execution_policy = research_execution_policy(task.metadata)
        if execution_policy is not None:
            _boundary, synthesis_parameter_schemas = _genome_parameter_boundary(task, parent)
            context = compact_research_context(
                context, parent=parent, parameter_schemas=synthesis_parameter_schemas,
                policy=execution_policy,
            )
        feedback: dict[str, Any] | None = None
        rejected_synthesis = validate_research_checkpoint(
            research_recovery, run=run, parent=parent,
            search=frozen_search_plan, knowledge=knowledge,
        )
        semantic_feedback = None
        semantic_failures = 0
        output_failures = 0
        synthesis_max_output_tokens = (
            execution_policy["synthesis_max_output_tokens"]
            if execution_policy is not None
            else _DEFAULT_SYNTHESIS_MAX_OUTPUT_TOKENS
        )
        # Local checkpoint validation costs no model attempt. New
        # submissions and output exhaustion have separate bounded
        # budgets: an incomplete response cannot consume the sole
        # opportunity to correct a later, real structured submission.
        first_attempt = 0 if rejected_synthesis is not None else 1
        for contract_attempt in range(first_attempt, 5):
            stage_context = {
                **deepcopy(context),
                "host_validation_feedback": deepcopy(feedback),
                **({"rejected_research_synthesis": deepcopy(rejected_synthesis),
                    "semantic_validation_feedback": deepcopy(semantic_feedback),
                    "repair_instruction": (
                        "Correct the stated validation failure using the actual frozen parent. "
                        "Preserve unaffected valid directions and evidence; submit a complete "
                        "concise object through structured_output, not a patch or plain text."
                    )} if semantic_feedback is not None else {}),
            }
            # Separate namespace from the historical two-slot loop;
            # four slots per activation also prevents resume collisions.
            dsh_stage_attempt = (max(1, int(stage_attempt)) - 1) * 4 + contract_attempt
            try:
                structured = (
                    deepcopy(rejected_synthesis) if contract_attempt == 0
                    else self._native_runtime().run(
                        run_id=run.run_id,
                        stage=stage,
                        role="researcher",
                        context=stage_context,
                        output_schema_id=output_schema_id,
                        run_state_revision=run_state_revision,
                        stage_attempt=dsh_stage_attempt,
                        ledger_expected_revision=ledger_expected_revision,
                        idempotency_key=(
                            f"{run.run_id}:generation:{run.generation}:{stage}:"
                            f"repair-v2:attempt:{dsh_stage_attempt}"
                        ),
                        identity_digests=self._native_stage_identity(
                            parent, task, run, stage, stage_context
                        ),
                        max_tokens=synthesis_max_output_tokens,
                    )
                )
            except DshNativeRuntimeUnavailableError as exc:
                if exc.error_code != _OUTPUT_BUDGET_ERROR_CODE:
                    raise
                output_failures += 1
                if output_failures >= 2:
                    raise DshResearchOutputBudgetError() from exc
                feedback = _output_budget_feedback(
                    attempt=contract_attempt,
                    max_output_tokens=synthesis_max_output_tokens,
                )
                continue
            try:
                reject_executable_fields(structured)
                normalized = validate_research_synthesis(
                    structured,
                    candidate_count=candidate_count,
                    allowed_evidence_refs=allowed_evidence_refs,
                    allowed_mutation_targets=allowed_mutation_targets,
                )
                validate_direction_schedule(normalized["candidate_directions"], family_schedule)
                direction_preflight = _validate_candidate_direction_realizability(
                    normalized["candidate_directions"],
                    run=run,
                    task=task,
                    parent=parent,
                    avoid_behaviors=evolution_reflection[
                        "avoid_behaviors"
                    ],
                    prior_generation_behaviors=(novelty_archive["entries"]
                        if novelty_archive is not None and not joint_policy.enabled(task.metadata) else ()),
                )
            except (TypeError, ValueError) as exc:
                semantic_failures += int(contract_attempt != 0)
                if semantic_failures >= _NATIVE_MAX_SEMANTIC_CONTRACT_ATTEMPTS:
                    raise
                feedback = _host_validation_feedback(
                    exc,
                    rejected_output=structured,
                    attempt=contract_attempt,
                )
                semantic_feedback = feedback
                rejected_synthesis = deepcopy(structured)
                continue
            plan = dict(current_plan or {})
            plan.update(
                {
                    "status": "model_generated",
                    "autonomous_research_protocol": (
                        AUTONOMOUS_RESEARCH_PROTOCOL
                    ),
                    "generation_search_plan": frozen_search_plan.to_dict(),
                    "dsh_research_summary": normalized["summary"],
                    "dsh_research_evidence": normalized["evidence"],
                    "candidate_directions": normalized[
                        "candidate_directions"
                    ],
                    "candidate_direction_preflight": direction_preflight,
                    **({novelty.ARCHIVE_KEY: novelty.archive_reference(novelty_archive)}
                       if novelty_archive is not None else {}),
                    **({"exploration_archive": cross_generation_experience["exploration_archive"]}
                       if isinstance(cross_generation_experience, Mapping) and "exploration_archive" in cross_generation_experience else {}),
                    **({"diversity_schedule": family_schedule} if family_schedule else {}),
                    "report_size_advisories": research_report_size_diagnostics(normalized) if execution_policy is not None else [],
                    "previous_generation_reflection": (
                        reflection_reference
                    ),
                    "dsh_evolution_reflection": evolution_reflection,
                }
            )
            # The final storage contract must participate in the same
            # bounded repair loop as the output schema, rather than
            # first rejecting an accepted result during ledger commit.
            plan = _host_guardrail_plan(plan, historical_guardrails)
            try:
                plan = validate_research_plan(plan)
            except (TypeError, ValueError) as exc:
                semantic_failures += int(contract_attempt != 0)
                if semantic_failures >= _NATIVE_MAX_SEMANTIC_CONTRACT_ATTEMPTS:
                    raise
                feedback = _host_validation_feedback(
                    exc, rejected_output=structured, attempt=contract_attempt,
                )
                semantic_feedback = feedback
                rejected_synthesis = deepcopy(structured)
                continue
            return plan
        raise RuntimeError("research synthesis repair loop did not terminate")

    def research_iteration(
        self,
        *,
        run: Run,
        task: TaskManifest,
        previous_generation_analysis: Mapping[str, Any] | None,
        knowledge_snapshot: Mapping[str, Any],
        previous_knowledge_assessment: Mapping[str, Any] | None,
        current_plan: Mapping[str, Any],
        cross_generation_experience: Mapping[str, Any] | None = None,
        parent_genome: Mapping[str, Any] | None = None,
        generation_search_plan: Mapping[str, Any] | None = None,
        previous_generation_reflection: Mapping[str, Any] | None = None,
        candidate_count: int | None = None,
        run_state_revision: int = 0,
        stage_attempt: int = 1,
        ledger_expected_revision: int = 0,
        research_recovery: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request one bounded advisory plan for the current generation."""

        metadata = task.metadata
        historical_parameter_guardrails = (
            _historical_parameter_guardrails_from_experience(
                cross_generation_experience
            )
        )
        model_id = _optional_text(
            metadata.get("strategy_model_id") or metadata.get("policy_model_id"),
            "strategy_model_id",
        )
        if model_id is None:
            raise ValueError("autonomous research iteration requires a strategy model")
        plan = self.research_plan(
            model_id,
            run=run,
            task=task,
            parameter_schemas=self.parameter_schemas_for_task(
                task,
                current_plan,
            ),
            previous_generation_analysis=previous_generation_analysis,
            knowledge_snapshot=knowledge_snapshot,
            previous_knowledge_assessment=previous_knowledge_assessment,
            current_plan=current_plan,
            cross_generation_experience=cross_generation_experience,
            parent_genome=parent_genome,
            generation_search_plan=generation_search_plan,
            previous_generation_reflection=previous_generation_reflection,
            candidate_count=candidate_count,
            run_state_revision=run_state_revision,
            stage_attempt=stage_attempt,
            ledger_expected_revision=ledger_expected_revision,
            research_recovery=research_recovery,
        )
        plan_status = str(plan.get("status") or "").casefold()
        status = (
            plan_status
            if plan_status in {"host_fallback", "unavailable"}
            else "model_generated"
        )
        merged_plan = _host_guardrail_plan(
            plan,
            historical_parameter_guardrails,
        )
        result = {
            "status": status,
            "model_id": model_id,
            "plan": merged_plan,
        }
        return result

    def propose(
        self,
        run: Run,
        task: TaskManifest,
        session_id: str,
        *,
        parent_candidate_id: str | None = None,
        parent_context: Mapping[str, Any] | None = None,
        batch_context: Mapping[str, Any] | None = None,
    ) -> Proposal:
        if task.metadata.get("execution_protocol") != DSH_NATIVE_EXECUTION_PROTOCOL:
            raise ValueError("Only DSH-native execution is supported")
        return self._propose_native(
            run, task, session_id, parent_candidate_id=parent_candidate_id,
            batch_context=batch_context,
        )

    def _propose_native(
        self,
        run: Run,
        task: TaskManifest,
        session_id: str,
        *,
        parent_candidate_id: str | None,
        batch_context: Mapping[str, Any] | None,
    ) -> Proposal:
        if session_id != f"dsh-native:{run.run_id}":
            raise RuntimeError("DSH-native strategy session is not open")
        batch = validate_batch_context(batch_context, run)
        if batch is None:
            raise ValueError("DSH-native proposal requires a frozen GenerationBatch")
        canonical_parent = batch.get("parent_genome_canonical_json")
        if not isinstance(canonical_parent, str):
            raise ValueError("DSH-native proposal is missing its frozen parent genome")
        parent = EcologyEvolutionPluginGenome.from_dict(json.loads(canonical_parent))
        if canonical_json(parent.to_dict()) != canonical_parent:
            raise ValueError("DSH-native parent genome is not canonical")
        if batch.get("parent_genome_digest") != parent.genome_digest:
            raise ValueError("DSH-native parent genome digest does not match the batch")
        slot_index = int(batch["slot_index"])
        stage_digests = batch.get("stage_context_digests")
        if not isinstance(stage_digests, Mapping):
            raise ValueError("DSH-native proposal stage context is incomplete")
        previous_analysis = safe_aggregate_feedback(
            batch.get("previous_generation_analysis"),
            name="DSH-native previous generation analysis",
        )
        raw_execution_plan = batch.get("derived_execution_plan")
        execution_plan = (DerivedExecutionPlan.from_dict(raw_execution_plan)
                          if isinstance(raw_execution_plan, Mapping) else derive_execution_plan(None))
        base_policy = SampleExecutionPolicy.from_mapping(task.metadata.get("sample_execution_policy"))
        runtime_constraints = {
            "sample_max_attempts_floor": max(base_policy.max_attempts, execution_plan.sample_max_attempts),
            "remote_critic_policy": task.metadata.get("sample_remote_critic_policy"),
        }
        research_iteration = safe_aggregate_feedback(
            batch.get("research_iteration"),
            name="DSH-native research iteration",
        )
        research_plan = (
            research_iteration.get("plan")
            if isinstance(research_iteration, Mapping)
            and isinstance(research_iteration.get("plan"), Mapping)
            else {}
        )
        novelty_archive = self._candidate_novelty_archive(
            run, task, research_plan.get(novelty.ARCHIVE_KEY),
        )
        assigned_direction = None
        family_schedule = {}
        if _model_search_cycle_enabled(task):
            allowed_mutation_targets = _registered_mutation_targets(task, parent)
            family_schedule = direction_schedule(task.metadata, allowed_mutation_targets, run.generation, int(batch["batch_size"]))
            raw_directions = research_plan.get("candidate_directions")
            directions = normalize_candidate_directions(
                raw_directions,
                exact_items=int(batch["batch_size"]),
                allowed_mutation_targets=allowed_mutation_targets,
            )
            assigned_direction = directions[slot_index]
            validate_direction_schedule([direction.to_dict() for direction in directions], family_schedule)
        assigned_direction_execution = (
            _candidate_direction_execution_view(assigned_direction)
            if assigned_direction is not None
            else None
        )
        slot_seed = int(
            digest(
                {
                    "run_id": run.run_id,
                    "generation": run.generation,
                    "slot_index": slot_index,
                    "task_seed": task.seed,
                }
            )[:16],
            16,
        )
        mutation_policy = mutation_policy_contract(task.metadata)
        mutation_budget_digest = digest(
            {
                "policy": mutation_policy["mutation_operator_id"],
                "generation": run.generation,
                "slot_index": slot_index,
                "maximum_operations": mutation_policy["candidate_maximum_operations"],
                "maximum_normalized_parameter_step": (
                    TRUST_REGION_MAX_NORMALIZED_STEP
                ),
                "candidate_direction_digest": (
                    assigned_direction.direction_digest
                    if assigned_direction is not None
                    else None
                ),
            }
        )
        context = GenomeMutationContextV1(
            run_id=run.run_id,
            generation=run.generation,
            slot_index=slot_index,
            slot_seed=slot_seed,
            parent_candidate_id=parent_candidate_id,
            parent_genome_digest=parent.genome_digest,
            generation_batch_digest=str(batch["context_digest"]),
            research_iteration_digest=str(stage_digests["research_iteration_digest"]),
            knowledge_snapshot_digest=str(stage_digests["knowledge_snapshot_digest"]),
            mutation_budget_digest=mutation_budget_digest,
            mutation_operator_id=mutation_policy["mutation_operator_id"],
        )
        research_directives = research_plan.get("dsh_evolution_reflection")
        if not isinstance(research_directives, Mapping):
            research_directives = {}
        avoid_behaviors = research_directives.get("review_behaviors", research_directives.get("avoid_behaviors"))
        if not isinstance(avoid_behaviors, list):
            avoid_behaviors = []
        sibling_candidate_behaviors = batch.get("sibling_candidate_behaviors")
        if not isinstance(sibling_candidate_behaviors, list):
            sibling_candidate_behaviors = []
        _boundary, active_parameter_schemas = _genome_parameter_boundary(
            task,
            parent,
        )
        mutation_catalog = _mutation_contract_catalog(task, parent)
        mutation_contract = {
            "schema_version": "ecologyrsi-dsh.genome-mutation-contract/1",
            **({"candidate_executability": program_preflight.contract(task.metadata)}
               if program_preflight.enabled(task.metadata) else {}),
            **({novelty.ARCHIVE_KEY: novelty.context_view(novelty_archive)}
               if novelty_archive is not None else {}),
            **({"exploration_archive": research_plan["exploration_archive"]}
               if diversity_enabled(task.metadata) and "exploration_archive" in research_plan else {}),
            "mutation_operator_id": mutation_policy["mutation_operator_id"],
            "maximum_operations": mutation_policy["candidate_maximum_operations"],
            "maximum_normalized_parameter_step": (
                TRUST_REGION_MAX_NORMALIZED_STEP
            ),
            "parameter_step_scale": (
                "log_for_positive_multi_order_ranges_otherwise_linear"
            ),
            "active_predictor_id": parent.scientific_program["predictor_ref"]["id"],
            "bounded_parameters": deepcopy(dict(active_parameter_schemas)),
            "one_step_parameter_intervals": {
                name: parameter_trust_region_neighborhood(
                    name=name,
                    previous=parent.scientific_program["parameter_overrides"][name],
                    contract=contract,
                )
                for name, contract in active_parameter_schemas.items()
            },
            **deepcopy(mutation_catalog),
            "policy": ("bounded_related_joint_edits" if joint_policy.enabled(task.metadata) else "single_axis_reject_out_of_bounds_without_clamping"),
            "runtime_constraints": runtime_constraints,
        }
        explicit_sibling_feedback = research_execution_policy(task.metadata) is not None
        if explicit_sibling_feedback and assigned_direction is not None and assigned_direction.mutation_axis == "scientific_parameter":
            exclusions = []
            target = assigned_direction.mutation_target
            for sibling in sibling_candidate_behaviors:
                if not isinstance(sibling, Mapping) or sibling.get("prediction_model_id") != parent.scientific_program["predictor_ref"]["id"]:
                    continue
                parameters = sibling.get("parameters")
                if not isinstance(parameters, Mapping) or target not in parameters:
                    continue
                try:
                    # A numerical coincidence in another Agent program is not
                    # an exclusion. Rebuild this single operation under the
                    # actual parent and require the exact sibling behavior.
                    witness = apply_genome_mutation(parent, {
                        "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                        "operations": [{"op": "set_bounded_parameter", "name": target, "value": parameters[target]}],
                    }, context, current_program_registry(), parameter_schemas=active_parameter_schemas)
                except (TypeError, ValueError):
                    continue
                if sibling.get("behavior_digest") != witness.behavior_digest:
                    continue
                exclusions.append({"slot_index": sibling["slot_index"], "parameter": target,
                                   "value": parameters[target], "behavior_digest": witness.behavior_digest})
            mutation_contract["sibling_parameter_exclusions"] = exclusions
            mutation_contract["sibling_parameter_exclusion_policy"] = (
                "The listed values exactly reproduce a frozen sibling under this same parent program. "
                "Choose another value inside the one-step interval and assigned direction; "
                "do not change a second parameter or use a different axis."
            )
        analysis_feedback = (
            previous_analysis if isinstance(previous_analysis, Mapping) else {}
        )
        common_failures = analysis_feedback.get("common_failures")
        if not isinstance(common_failures, list):
            common_failures = []
        base_reflection = {
            "schema_version": "ecologyrsi-dsh.evolution-execution-signals/1",
            "previous_generation_signals": {
                "has_analysis": bool(analysis_feedback),
                "no_eligible_candidate": (
                    analysis_feedback.get("outcome") == "no_eligible_candidate"
                ),
                "scientific_gate_failed": (
                    "scientific_gate_failed" in common_failures
                ),
                "execution_failed": "execution_failed" in common_failures,
            },
            "research_signals": {
                "has_research_summary": bool(
                    research_plan.get("dsh_research_summary")
                ),
                "has_research_evidence": bool(
                    research_plan.get("dsh_research_evidence")
                ),
                "has_previous_action": bool(
                    research_iteration.get("previous_next_action")
                    if isinstance(research_iteration, Mapping)
                    else None
                ),
                "has_previous_generation_reflection": bool(
                    research_plan.get("previous_generation_reflection")
                ),
            },
            "assigned_candidate_direction": assigned_direction_execution,
            "prior_failure_behavior_digests": [
                str(item["behavior_digest"])
                for item in avoid_behaviors
                if isinstance(item, Mapping)
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(item.get("behavior_digest") or "")
                )
            ],
            "prior_failures_are_advisory": True,
            "sibling_behavior_digests": [
                str(item["behavior_digest"])
                for item in sibling_candidate_behaviors
                if isinstance(item, Mapping)
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(item.get("behavior_digest") or "")
                )
            ],
        }
        proposal_research_iteration = {
            "schema_version": "ecologyrsi-dsh.research-iteration-execution/1",
            "source_digest": str(
                stage_digests["research_iteration_digest"]
            ),
            "has_research_plan": bool(research_plan),
        }
        host_rejections: list[dict[str, Any]] = []
        child: EcologyEvolutionPluginGenome | None = None
        base_stage_attempt = max(1, int(batch.get("stage_attempt", 1)))
        for proposal_attempt in range(1, _NATIVE_MAX_PROPOSAL_ATTEMPTS + 1):
            # A host rejection asks the DSH proposer for a genuinely new child
            # launch.  Admission fences are intentionally one-shot, so encode
            # the batch attempt, sibling slot, and internal retry into a unique
            # stage attempt instead of trying to reopen the prior fence.
            dsh_stage_attempt = (
                (
                    (base_stage_attempt - 1) * int(batch["batch_size"])
                    + slot_index
                )
                * _NATIVE_MAX_PROPOSAL_ATTEMPTS
                + proposal_attempt
            )
            stage_context = {
                "parent_genome": parent.to_dict(),
            **_inherited_strategy_context(parent, task),
                "mutation_context": context.to_dict(),
                "mutation_contract": deepcopy(mutation_contract),
                "assigned_candidate_direction": assigned_direction_execution,
                "generation_context_digest": batch["context_digest"],
                "research_iteration": deepcopy(proposal_research_iteration),
                "diagnostic_report": (
                    research_iteration.get("diagnostic_report")
                    if isinstance(research_iteration, Mapping) else None
                ),
                "evolution_reflection": {
                    **deepcopy(base_reflection),
                    "host_rejections": deepcopy(host_rejections),
                    "proposal_attempt": proposal_attempt,
                    "maximum_proposal_attempts": _NATIVE_MAX_PROPOSAL_ATTEMPTS,
                },
            }
            mutation = self._native_runtime().run(
                run_id=run.run_id,
                stage="candidate.propose",
                role="candidate-proposer",
                context=stage_context,
                output_schema_id="ecology-genome-mutation@1",
                run_state_revision=int(batch.get("run_state_revision", run.generation)),
                stage_attempt=dsh_stage_attempt,
                ledger_expected_revision=int(batch.get("ledger_expected_revision", 0)),
                idempotency_key=(
                    f"{run.run_id}:generation:{run.generation}:slot:{slot_index}:"
                    f"proposal:attempt:{proposal_attempt}"
                ),
                identity_digests=self._native_stage_identity(
                    parent, task, run, "candidate.propose", stage_context
                ),
            )
            try:
                proposed_child = apply_genome_mutation(
                    parent,
                    mutation,
                    context,
                    current_program_registry(),
                    parameter_schemas=active_parameter_schemas,
                    runtime_constraints=runtime_constraints,
                )
            except (TypeError, ValueError) as exc:
                message = str(exc)
                rejection_code = (
                    "mutation_has_no_effect"
                    if "does not change" in message
                    or "effective_noop" in message
                    or "same behavior as its parent" in message
                    else "mutation_validation_failed"
                )
                host_rejections.append(
                    {
                        "reason": "invalid_mutation_contract",
                        "rejection_code": rejection_code,
                        # The exception originates exclusively in the bounded
                        # Host mutation validator.  Returning its compact rule
                        # violation gives the next DSH child enough information
                        # to repair the proposal instead of blindly replaying it.
                        "validation_detail": message[:500],
                        "rejected_operations": deepcopy(mutation["operations"]),
                    }
                )
                continue
            try:
                compile_plugin_behavior(
                    proposed_child,
                    task,
                    None,
                    current_program_registry(),
                    compiler_semantic_digest=str(
                        task.metadata.get("compiler_semantic_digest")
                        or DEFAULT_COMPILER_SEMANTIC_DIGEST
                    ),
                )
            except (TypeError, ValueError) as exc:
                message = str(exc)
                host_rejections.append(
                    {
                        "reason": "compiled_behavior_validation_failed",
                        "rejection_code": (
                            "candidate_program_not_executable"
                            if isinstance(exc, program_preflight.CandidateProgramPreflightError)
                            else "predictor_evaluator_incompatible"
                            if "predictor and evaluator" in message
                            else "candidate_compile_validation_failed"
                        ),
                        "validation_detail": message[:500],
                        "rejected_operations": deepcopy(mutation["operations"]),
                    }
                )
                continue
            if assigned_direction is not None:
                accepted_operation = mutation["operations"][0]
                # Only the *identifying* fields per axis; the operation name comes
                # from the shared catalog so this check cannot drift from the one
                # the proposer was shown. An axis opened in
                # `CANDIDATE_MUTATION_AXES` without an entry here would otherwise
                # raise `KeyError` mid-batch and fail the whole run, so a missing
                # entry is rejected as one bad proposal instead.
                expected_fields_by_axis = {
                    "scientific_parameter": {
                        "name": assigned_direction.mutation_target,
                    },
                    "registered_predictor": {
                        "predictor_id": assigned_direction.mutation_target,
                    },
                    "instruction_profile": {
                        "role": "sample-planner",
                        "instruction_template_id": (
                            assigned_direction.mutation_target
                        ),
                    },
                    "instruction_parameter": {
                        "role": "sample-planner",
                        "name": assigned_direction.mutation_target,
                    },
                    "workflow_parameter": {
                        "name": assigned_direction.mutation_target,
                    },
                    # The target is the grammar, which the operation does not
                    # carry -- the operation carries the authored clauses. So
                    # the only identifying field is the role; whether the text
                    # is legal under that grammar is settled by the validator,
                    # and whether it is `author` or `revise` by the check just
                    # below, which needs the parent to answer.
                    "instruction_directive": {"role": "sample-planner"},
                    "skill_program": {"role": "sample-planner"},
                    "strategy_program": {"role": "sample-planner"},
                    "feature_recipe": {},
                }
                if assigned_direction.mutation_axis not in expected_fields_by_axis:
                    host_rejections.append(
                        {
                            "reason": "assigned_direction_unverifiable",
                            "rejection_code": "mutation_axis_has_no_direction_check",
                            "validation_detail": (
                                "the Host cannot verify that an operation "
                                "implements a direction on axis "
                                f"{assigned_direction.mutation_axis}"
                            )[:500],
                            "rejected_operations": deepcopy(
                                mutation["operations"]
                            ),
                        }
                    )
                    continue
                expected_operation = {
                    "op": MUTATION_OPERATION_BY_AXIS[
                        assigned_direction.mutation_axis
                    ],
                    **expected_fields_by_axis[assigned_direction.mutation_axis],
                }
                direction_mismatch = any(
                    accepted_operation.get(name) != value
                    for name, value in expected_operation.items()
                )
                if not direction_mismatch and "increase" in (
                    MUTATION_DIRECTIONS_BY_AXIS[assigned_direction.mutation_axis]
                ):
                    # Every scalar axis has to be checked for more than "it named
                    # the right knob": a direction is a claim about which way the
                    # number moves, and an accepted operation that moves it the
                    # other way implements a different experiment than the one
                    # this candidate was assigned.
                    previous_value = _assigned_direction_baseline(
                        parent,
                        assigned_direction.mutation_axis,
                        assigned_direction.mutation_target,
                    )
                    proposed_value = accepted_operation.get("value")
                    mutation_direction = assigned_direction.mutation_direction
                    direction_mismatch = (
                        isinstance(previous_value, bool)
                        or not isinstance(previous_value, (int, float))
                        or isinstance(proposed_value, bool)
                        or not isinstance(proposed_value, (int, float))
                        or mutation_direction == "increase"
                        and float(proposed_value) <= float(previous_value)
                        or mutation_direction == "decrease"
                        and float(proposed_value) >= float(previous_value)
                        or mutation_direction not in {"increase", "decrease"}
                    )
                if (
                    not direction_mismatch
                    and assigned_direction.mutation_axis == "instruction_directive"
                ):
                    # `author` and `revise` are claims about the parent, not
                    # about a number: writing the first directive and rewriting
                    # an existing one are different experiments, and only the
                    # parent can say which one this is.
                    had_directive = _has_authored_directive(parent, "sample-planner")
                    direction_mismatch = (
                        assigned_direction.mutation_direction == "author"
                        and had_directive
                        or assigned_direction.mutation_direction == "revise"
                        and not had_directive
                        or assigned_direction.mutation_direction
                        not in {"author", "revise"}
                    )
                if not direction_mismatch and assigned_direction.mutation_axis in {"skill_program", "feature_recipe", "strategy_program"}:
                    current = (resolve_candidate_agent_profile(parent, current_program_registry()).get(assigned_direction.mutation_axis)
                               if assigned_direction.mutation_axis in {"skill_program", "strategy_program"}
                               else parent.scientific_program.get("feature_recipe"))
                    direction_mismatch = (assigned_direction.mutation_direction == "author") == (current is not None)
                if direction_mismatch:
                    host_rejections.append(
                        {
                            "reason": "assigned_direction_mismatch",
                            "rejection_code": "mutation_does_not_implement_direction",
                            "validation_detail": (
                                "the accepted operation must exactly implement "
                                f"{assigned_direction.mutation_axis}:"
                                f"{assigned_direction.mutation_target}:"
                                f"{assigned_direction.mutation_direction} as "
                                f"{expected_operation}"
                            )[:500],
                            "rejected_operations": deepcopy(
                                mutation["operations"]
                            ),
                        }
                    )
                    continue
            scientific = proposed_child.to_dict()["scientific_program"]
            prior_repeated = next((item for item in novelty_archive["entries"]
                                   if item["behavior_digest"] == proposed_child.behavior_digest), None) if novelty_archive is not None else None
            if prior_repeated is not None:
                host_rejections.append({
                    "reason": "prior_generation_behavior_duplicate",
                    "rejection_code": "candidate_behavior_already_proposed",
                    "prior_candidate_id": prior_repeated["candidate_id"],
                    "prior_revision_id": prior_repeated["revision_id"],
                    "prior_generation": prior_repeated["generation"],
                    "prior_behavior_digest": prior_repeated["behavior_digest"],
                    "rejected_operations": deepcopy(mutation["operations"]),
                    "validation_detail": (
                        "The complete executable behavior was already proposed in this run. "
                        "Keep the frozen primary direction and bounds; choose a different legal value "
                        "or, only when permitted by maximum_operations, a related additional edit. "
                        "A changed explanation or genome identity is not a new behavior."
                    ),
                })
                continue
            sibling_repeated = next(
                (
                    item
                    for item in sibling_candidate_behaviors
                    if isinstance(item, Mapping)
                    and (
                        (
                            isinstance(item.get("behavior_digest"), str)
                            and item["behavior_digest"]
                            == proposed_child.behavior_digest
                        )
                        or (
                            not isinstance(item.get("behavior_digest"), str)
                            and item.get("prediction_model_id")
                            == scientific["predictor_ref"]["id"]
                            and isinstance(item.get("parameters"), Mapping)
                            and canonical_json(item["parameters"])
                            == canonical_json(scientific["parameter_overrides"])
                        )
                    )
                ),
                None,
            )
            if sibling_repeated is None:
                child = proposed_child
                break
            if sibling_repeated is not None:
                host_rejections.append(
                    {
                        "reason": "sibling_behavior_duplicate",
                        "prediction_model_id": scientific["predictor_ref"]["id"],
                        "parameters_digest": digest(scientific["parameter_overrides"]),
                        "sibling_slot_index": sibling_repeated.get("slot_index"),
                        "sibling_behavior_digest": sibling_repeated.get(
                            "behavior_digest"
                        ),
                        **({
                            "rejected_operations": deepcopy(mutation["operations"]),
                            "validation_detail": (
                                "This exact operation repeats a frozen sibling behavior. Choose a different "
                                "legal value for the same assigned parameter and direction; keep one operation."
                            ),
                        } if explicit_sibling_feedback else {}),
                    }
                )
        if child is None:
            if host_rejections and host_rejections[-1]["reason"] == "prior_generation_behavior_duplicate":
                raise ValueError("DSH proposal repeats a prior-generation behavior after bounded retry")
            if host_rejections and host_rejections[-1]["reason"] == "sibling_behavior_duplicate":
                raise ValueError(
                    "DSH proposal repeats an existing sibling behavior after bounded retry"
                )
            if (
                host_rejections
                and host_rejections[-1]["reason"]
                == "compiled_behavior_validation_failed"
            ):
                raise ValueError(
                    "DSH proposal does not compile after bounded retry"
                )
            if host_rejections and host_rejections[-1]["reason"] in {
                "assigned_direction_mismatch",
                "invalid_mutation_contract",
            }:
                raise ValueError(
                    "DSH proposal violates the mutation contract after bounded retry"
                )
            raise ValueError(
                "DSH proposal failed host validation after bounded retry"
            )
        key = (session_id, run.generation)
        index = self._session_counts.setdefault(key, 0)
        self._session_counts[key] = index + 1
        candidate_agent_profile = resolve_candidate_agent_profile(
            child,
            current_program_registry(),
        )
        agent_policy = build_agent_policy(
            genome_digest=child.genome_digest, profile=candidate_agent_profile,
            parameters=child.scientific_program["parameter_overrides"],
            previous_analysis=previous_analysis, generation=run.generation,
            joint=joint_policy.enabled(task.metadata),
            prediction_formula_policy=("mean-referenced-tools@1"
                if task.metadata.get("optimization_schedule", {}).get("schema_version") in {EVIDENCE_GUIDED_SCHEDULE_SCHEMA, joint_policy.SCHEDULE_SCHEMA}
                else None),
        )
        return Proposal(
            proposal_id=(
                f"proposal:{run.run_id}:{run.generation}:{slot_index + 1}:"
                f"{child.genome_digest[:12]}"
            ),
            run_id=run.run_id,
            generation=run.generation,
            title=f"DSH plugin genome mutation {slot_index + 1}",
            changes=dict(child.scientific_program["parameter_overrides"]),
            parent_candidate_id=parent_candidate_id,
            rationale=(
                "DSH structured GenomeMutation materialized as an isolated candidate "
                "by the Host registry; this is not a global-champion promotion."
            ),
            metadata={
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "proposal_source": "dsh_native_agent",
                **({"diversity_schedule": family_schedule} if family_schedule else {}),
                "agent_policy": agent_policy,
                "tool_experience": agent_policy["experience"]["rows"],
                "remote_strategy_called": True,
                "remote_strategy_succeeded": True,
                "evolution_genome_canonical_json": canonical_json(child.to_dict()),
                "genome_digest": child.genome_digest,
                "behavior_digest": child.behavior_digest,
                "mutation_context": context.to_dict(),
                "mutation_digest": child.lineage["mutation_digest"],
                "mutation_operations": deepcopy(mutation["operations"]),
                "effect_resolution": resolve_mutation_effects(
                    parent, child, mutation["operations"], current_program_registry(),
                    runtime_constraints=runtime_constraints,
                ),
                "candidate_agent_profile": candidate_agent_profile,
                "candidate_direction": (
                    assigned_direction.to_dict()
                    if assigned_direction is not None
                    else None
                ),
                "candidate_direction_id": (
                    assigned_direction.direction_id
                    if assigned_direction is not None
                    else None
                ),
                "candidate_direction_digest": (
                    assigned_direction.direction_digest
                    if assigned_direction is not None
                    else None
                ),
            },
        )



    def close_session(self, session_id: str) -> None:
        for key in tuple(self._session_counts):
            if key[0] == session_id:
                self._session_counts.pop(key, None)




def _domain_name(value: Any) -> str:
    raw = _optional_text(value, "domain")
    if raw is None:
        return "toy"
    normalized = raw.lower().replace("_", "-")
    if "greenhouse" in normalized or "温室" in normalized:
        return "greenhouse"
    if "toy" in normalized or "crop-soil-water" in normalized or "土壤水" in normalized:
        return "toy"
    raise ValueError(f"unsupported task domain: {raw}")




def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()






def _finite_number(value: Any, *, default: float | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default




def _task_parameter_boundary(
    task: TaskManifest,
    current_plan: Mapping[str, Any] | None = None,
) -> tuple[str, Mapping[str, Mapping[str, Any]]]:
    metadata = dict(task.metadata)
    domain = _domain_name(metadata.get("domain", task.domain_pack))
    boundary, schemas, _sweep = _task_parameter_space(metadata, domain)
    return boundary, schemas


def _genome_parameter_boundary(
    task: TaskManifest,
    genome: EcologyEvolutionPluginGenome,
) -> tuple[str, Mapping[str, Mapping[str, Any]]]:
    """Resolve the mutation space from the frozen parent, never stale UI metadata."""

    predictor_id = str(genome.scientific_program["predictor_ref"]["id"])
    metadata = {**dict(task.metadata), "prediction_model_id": predictor_id}
    domain = _domain_name(metadata.get("domain", task.domain_pack))
    boundary, schemas, _sweep = _task_parameter_space(metadata, domain)
    return boundary, schemas


def _registered_mutation_targets(
    task: TaskManifest,
    genome: EcologyEvolutionPluginGenome,
) -> dict[str, tuple[str, ...]]:
    """Build the exact Host-owned target set exposed to research agents."""

    _boundary, parameter_schemas = _genome_parameter_boundary(task, genome)
    registry = current_program_registry()
    current_predictor = str(genome.scientific_program["predictor_ref"]["id"])
    runtime_catalog = task.metadata.get("runtime_component_catalog")
    raw_predictors = (
        runtime_catalog.get("prediction_models")
        if isinstance(runtime_catalog, Mapping)
        else None
    )
    predictor_ids: list[str] = []
    if isinstance(raw_predictors, list):
        evaluator_id = str(task.metadata.get("evaluator_id") or "")
        for item in raw_predictors:
            predictor_id = (
                str(item.get("id") or "").strip()
                if isinstance(item, Mapping)
                else ""
            )
            baseline_reset = (predictor_id == current_predictor == BASELINE_REFERENCE_PREDICTOR_ID
                              and prediction_usage(current_predictor, genome.scientific_program["parameter_overrides"])["mode"] != "baseline_only")
            if not predictor_id or (predictor_id == current_predictor and not baseline_reset):
                continue
            try:
                registry.program("predictors", predictor_id)
            except ValueError:
                continue
            if evaluator_id not in registered_predictor_evaluator_ids(predictor_id):
                continue
            predictor_ids.append(predictor_id)

    current_profile = resolve_candidate_agent_profile(genome, registry)
    recipe_policy_id = registry.program(
        "predictors", current_predictor
    ).get("feature_policy_id")
    instruction_ids: list[str] = []
    for instruction_id in registry.program_ids("instruction_templates"):
        instruction = registry.program("instruction_templates", instruction_id)
        if (
            instruction.get("role") == "sample-planner"
            and instruction_id != current_profile["instruction_template_id"]
            and not (diversity_enabled(task.metadata) and "authored_directive" in current_profile)
        ):
            instruction_ids.append(instruction_id)

    return {
        "scientific_parameter": tuple(sorted(
            set(parameter_schemas)
        )),
        # Present only when the bound predictor declares a feature policy. On a
        # scalar-tunable predictor there is nothing to author, and advertising
        # the axis would invite an edit the compiler is about to refuse.
        **(
            {"feature_recipe": (recipe_policy_id,)}
            if recipe_policy_id is not None
            else {}
        ),
        "registered_predictor": tuple(sorted(set(predictor_ids))),
        "instruction_profile": tuple(sorted(instruction_ids)),
        # The two axes whose bounds come from the program registry rather than
        # the frozen parameter boundary. They are advertised here, and not only
        # to the batch-local editor, because a generation-level proposal is where
        # a real change of retry depth or escalation rate belongs: both are
        # properties of how the candidate runs for a whole generation, and both
        # now reach the runtime, so a direction on either is something a score
        # can be attributed to.
        "workflow_parameter": _registered_scalar_axis_targets(
            genome, "workflow_parameter"
        ),
        "instruction_parameter": _registered_scalar_axis_targets(
            genome, "instruction_parameter"
        ),
        # Always available, and deliberately a single target: the target names
        # the grammar an authored directive is written under, not the text. A
        # candidate that already authored one revises it; one that has not
        # writes its first. Unlike `instruction_profile`, this axis does not
        # run out of unused options.
        "instruction_directive": (AUTHORED_DIRECTIVE_POLICY_ID,),
        "skill_program": (SKILL_POLICY_ID,),
        **({"strategy_program": (joint_policy.PROGRAM_POLICY,)} if joint_policy.enabled(task.metadata) else {}),
    }


def _mutation_contract_catalog(
    task: TaskManifest,
    genome: EcologyEvolutionPluginGenome,
) -> dict[str, Any]:
    targets = _registered_mutation_targets(task, genome)
    registry = current_program_registry()
    sample_budget_class = task.metadata.get("sample_budget_class")
    if sample_budget_class not in {"diagnostic_smoke", "selection_eligible"}:
        sample_budget_class = None
    samples_per_update = task.metadata.get("samples_per_update")
    if isinstance(samples_per_update, bool) or not isinstance(
        samples_per_update, int
    ):
        samples_per_update = None
    prediction_cells_per_origin = task.metadata.get("prediction_cells_per_origin")
    if isinstance(prediction_cells_per_origin, bool) or not isinstance(
        prediction_cells_per_origin, int
    ):
        prediction_cells_per_origin = None
    origin_budget = (
        samples_per_update // prediction_cells_per_origin
        if samples_per_update is not None
        and prediction_cells_per_origin is not None
        and prediction_cells_per_origin > 0
        else None
    )
    effective_cell_budget = (
        origin_budget * prediction_cells_per_origin
        if origin_budget is not None and prediction_cells_per_origin is not None
        else None
    )
    unused_cell_budget = (
        samples_per_update - effective_cell_budget
        if samples_per_update is not None and effective_cell_budget is not None
        else None
    )
    diagnostic_only = sample_budget_class == "diagnostic_smoke"
    selection_eligible = sample_budget_class == "selection_eligible"
    axis_contract = mutation_axis_contract(targets)
    current_profile = resolve_candidate_agent_profile(genome, registry)
    authoring_state = {
        axis: {
            "present": program is not None,
            "program_digest": digest(program) if program is not None else None,
            "allowed_direction": "revise" if program is not None else "author",
        }
        for axis, program in {
            "skill_program": current_profile.get("skill_program"),
            "strategy_program": current_profile.get("strategy_program"),
            "instruction_directive": current_profile.get("authored_directive"),
            "feature_recipe": genome.to_dict()["scientific_program"].get("feature_recipe"),
        }.items() if targets.get(axis)
    }
    for axis, entry in authoring_state.items():
        axis_contract["mutation_directions_by_axis"][axis] = [entry["allowed_direction"]]
    return {
        "parent_authoring_state": {
            "genome_digest": genome.genome_digest,
            "axes": authoring_state,
            "instruction": (
                "Only this frozen parent is inherited. Programs from rejected siblings or "
                "historical proposals are not present in it. Author absent programs; revise "
                "only existing programs. The Host rejects, and never rewrites, a mismatched direction."
            ),
        },
        "scientific_parameter_activity": parameter_activity_contract(genome),
        "allowed_mutation_targets": {
            axis: list(values) for axis, values in targets.items()
        },
        "operation_by_axis": axis_contract["operation_by_axis"],
        "mutation_directions_by_axis": axis_contract["mutation_directions_by_axis"],
        **({"prediction_selection": {
            "owner": "sample_agent", "default_tool_owner": "research_model", "current_usage": prediction_usage(
                str(genome.scientific_program["predictor_ref"]["id"]), genome.scientific_program["parameter_overrides"]),
            "baseline_only_choice": BASELINE_REFERENCE_PREDICTOR_ID,
            "instructions": "Research evolves the optional default model and Agent policy. Each sample Agent decides whether to use that model, invoke other registered tools with bounded parameters, combine evidence or predict directly. Evaluate the final Agent numbers; evaluator, data and gates remain fixed.",
            "predictor_options": [
                {"id": item["id"], "label": item.get("label"), "description": item.get("description"),
                 "parameter_schemas": item.get("parameter_schemas", {})}
                for item in task.metadata.get("runtime_component_catalog", {}).get("prediction_models", [])
            ],
        }} if task.metadata.get("prediction_selection_policy") == RUNTIME_PREDICTION_POLICY else {}),
        "mutation_axis_effects": axis_contract["mutation_axis_effects"],
        "evaluation_evidence_contract": {
            "schema_version": "ecologyrsi-dsh.evaluation-evidence-contract/1",
            "sample_budget_class": sample_budget_class,
            "samples_per_update": samples_per_update,
            "prediction_cells_per_origin": prediction_cells_per_origin,
            "origin_budget_per_candidate": origin_budget,
            "effective_prediction_cell_budget": effective_cell_budget,
            "unused_prediction_cell_budget": unused_cell_budget,
            "diagnostic_only": diagnostic_only,
            "selection_evidence_eligible": selection_eligible,
            "promotion_possible_from_this_run": selection_eligible,
            "required_terminal_outcome": (
                "diagnostic_smoke_completed_no_promotion"
                if diagnostic_only
                else "host_gate_result"
                if selection_eligible
                else "host_controlled"
            ),
            "success_criterion_scope": (
                "Compare diagnostic scores, target-horizon behavior, and agent "
                "reliability; do not claim eligibility, gate passage, or "
                "promotion from this run."
                if diagnostic_only
                else "State measurable full-matrix improvement; Host gates "
                "alone determine selection and promotion."
                if selection_eligible
                else "Use only observed aggregate outcomes; Host gates alone "
                "determine selection and promotion."
            ),
        },
        "host_owned_immutable_controls": [
            "evaluation cohort membership and origin selection",
            "samples_per_update and prediction cells per origin",
            "evaluator, evidence thresholds, and promotion gates",
            "registered prediction-tool numerical output",
        ],
        "instruction_profiles": [
            {
                "instruction_template_id": instruction_id,
                "skill_name": registry.program(
                    "instruction_templates", instruction_id
                )["skill_name"],
            }
            for instruction_id in targets["instruction_profile"]
        ],
        # The clause whitelist an authored directive has to satisfy. Published
        # alongside the template list because the two are alternatives for the
        # same slot: a proposer choosing between selecting and authoring needs
        # to see both, and one that authors without the grammar can only guess
        # at the enumerations its text will be rejected against.
        "directive_grammar": _directive_authoring_contract(genome),
        **({"strategy_program_grammar": strategy_program_grammar()} if joint_policy.enabled(task.metadata) else {}),
        **({"skill_grammar": skill_grammar(), "current_skill_program": resolve_candidate_agent_profile(genome, registry).get("skill_program")}
           if targets.get("skill_program") else {}),
    }


def _directive_authoring_contract(
    genome: EcologyEvolutionPluginGenome,
) -> dict[str, Any]:
    """The authoring grammar, narrowed to what this candidate may actually write.

    The registered grammar is the global whitelist; the tools a given candidate
    may plan come from its own profile, which may have been narrowed below the
    base policy. Publishing the intersection means a proposal that follows the
    contract cannot be rejected by the compiler for naming a withheld tool.
    """

    registry = current_program_registry()
    policy = registry.program("directive_policies", AUTHORED_DIRECTIVE_POLICY_ID)
    profile = next(
        (
            item
            for item in genome.agent_program["candidate_execution_program"][
                "role_profiles"
            ]
            if item.get("role") == "sample-planner"
        ),
        None,
    )
    enabled_tools = list((profile or {}).get("enabled_tool_ids") or ())
    return {
        **deepcopy(dict(policy["grammar"])),
        "policy_id": AUTHORED_DIRECTIVE_POLICY_ID,
        "policy_digest": registry.program_ref(
            "directive_policies", AUTHORED_DIRECTIVE_POLICY_ID
        )["catalog_digest"],
        "effective_parameters": {
            name: contract["default"]
            for name, contract in policy["parameters"].items()
        },
        "tool_plan.allowed_tool_ids": enabled_tools,
        "current_directive_present": _has_authored_directive(
            genome, "sample-planner"
        ),
    }


def _inherited_strategy_context(parent, task):
    if not joint_policy.enabled(task.metadata):
        return {}
    program = resolve_candidate_agent_profile(parent, current_program_registry()).get("strategy_program")
    if not program:
        return {}
    return {"inherited_search_strategy": {"program_digest": digest(program),
        **{key: program[key] for key in ("research_prompt", "proposal_prompt", "memory")},
        "qualification": "candidate_authored_advice_not_verified_fact_or_evaluation_rules"}}


def _candidate_direction_execution_view(
    direction: CandidateDirection,
) -> dict[str, Any]:
    """Expose only Host-validated mutation coordinates to the proposer.

    Research prose remains persisted on the direction for audit and display,
    but it is not executable candidate-proposer context.  In particular, a
    historical exact value plus a cross-field pronoun cannot become a second
    source of mutation values.
    """

    return {
        "schema_version": "ecologyrsi-dsh.candidate-direction-execution/1",
        "direction_digest": direction.direction_digest,
        "mutation_axis": direction.mutation_axis,
        "mutation_target": direction.mutation_target,
        "mutation_direction": direction.mutation_direction,
        "evidence_refs": list(direction.evidence_refs),
        "research_prose": "audit_only_not_executable",
    }


def _parameter_preflight_values(
    schema: Mapping[str, Any],
    current_value: Any,
    mutation_direction: str,
    *,
    minimum_witness_count: int = 0,
) -> tuple[int | float, ...]:
    """Return deterministic witnesses within the bounded trust region."""

    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, (int, float))
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(minimum))
        or not math.isfinite(float(maximum))
        or float(maximum) <= float(minimum)
        or isinstance(current_value, bool)
        or not isinstance(current_value, (int, float))
        or not math.isfinite(float(current_value))
    ):
        return ()
    if schema.get("type") == "integer":
        values = range(math.ceil(float(minimum)), math.floor(float(maximum)) + 1)
        return tuple(
            sorted(
                (
                    value
                    for value in values
                    if value != current_value
                    and (
                        mutation_direction == "increase" and value > current_value
                        or mutation_direction == "decrease" and value < current_value
                    )
                ),
                key=lambda value: (abs(float(value) - float(current_value)), value),
            )
        )

    lower = float(minimum)
    upper = float(maximum)
    current = float(current_value)
    if not lower <= current <= upper:
        return ()
    witness_count = max(_NATIVE_AVOID_BEHAVIOR_LIMIT + 8, minimum_witness_count)
    values: list[float] = []
    if lower > 0.0 and upper / lower >= 100.0 and current > 0.0:
        lower_coordinate = math.log(lower)
        upper_coordinate = math.log(upper)
        current_coordinate = math.log(current)
        to_value = math.exp
    else:
        lower_coordinate = lower
        upper_coordinate = upper
        current_coordinate = current
        to_value = float
    trust_extent = (
        (upper_coordinate - lower_coordinate) * TRUST_REGION_MAX_NORMALIZED_STEP
    )
    decrease_extent = min(
        trust_extent,
        current_coordinate - lower_coordinate,
    )
    increase_extent = min(
        trust_extent,
        upper_coordinate - current_coordinate,
    )
    for index in range(1, witness_count + 1):
        fraction = index / witness_count
        if mutation_direction == "decrease" and decrease_extent > 0.0:
            values.append(
                to_value(current_coordinate - decrease_extent * fraction)
            )
        if mutation_direction == "increase" and increase_extent > 0.0:
            values.append(
                to_value(current_coordinate + increase_extent * fraction)
            )
    result: list[float] = []
    for value in values:
        bounded = min(upper, max(lower, value))
        if bounded == current:
            continue
        if bounded in result:
            continue
        result.append(bounded)
    return tuple(result)


def _directive_preflight_operations(
    parent: EcologyEvolutionPluginGenome, mutation_direction: str
) -> tuple[dict[str, Any], ...]:
    """Witness directives proving the authoring axis is realizable.

    The counterpart to ``_parameter_preflight_values`` for an axis with no
    numbers in it. These are Host-built placeholders whose only job is to show
    a legal one-operation implementation exists; the proposer writes its own
    clauses, and the validator -- not this function -- decides whether they are
    legal. Every witness differs from the parent's directive, because a
    mutation equal to its parent is refused.
    """

    from ..evaluators.authored_directive import (
        ALLOWED_DIRECTIVE_ANCHORS,
        BLEND_RULE_METHODS,
    )

    if mutation_direction not in {"author", "revise"}:
        return ()
    had_directive = _has_authored_directive(parent, "sample-planner")
    # `author` writes the first directive and `revise` rewrites an existing
    # one, so a direction that disagrees with the parent has no
    # implementation at all -- which is what the preflight exists to catch.
    if (mutation_direction == "author") == had_directive:
        return ()
    profile = next(
        (
            item
            for item in parent.to_dict()["agent_program"][
                "candidate_execution_program"
            ]["role_profiles"]
            if item.get("role") == "sample-planner"
        ),
        None,
    )
    if profile is None:
        return ()
    tool_ids = list(profile.get("enabled_tool_ids") or ())
    if not tool_ids:
        return ()
    current = profile.get("authored_directive") or {}
    witnesses: list[dict[str, Any]] = []
    for anchor in sorted(ALLOWED_DIRECTIVE_ANCHORS):
        for blend_rule in sorted(BLEND_RULE_METHODS):
            directive = {
                "anchor": anchor,
                "blend_rule": blend_rule,
                "tool_plan": [
                    {"tool_id": tool_ids[0], "purpose": "candidate_model_baseline"}
                ],
                "rationale": (
                    "Host preflight witness: establish the candidate's own model "
                    "baseline before departing from it."
                ),
            }
            if (
                current.get("anchor") == anchor
                and current.get("blend_rule") == blend_rule
            ):
                continue
            witnesses.append(
                {
                    "op": "author_role_directive",
                    "role": "sample-planner",
                    "authored_directive": directive,
                }
            )
    return tuple(witnesses)


def _validate_candidate_direction_realizability(
    raw_directions: Sequence[Mapping[str, Any]],
    *,
    run: Run,
    task: TaskManifest,
    parent: EcologyEvolutionPluginGenome,
    avoid_behaviors: Sequence[Mapping[str, Any]],
    prior_generation_behaviors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Prove each research direction has a legal one-operation implementation.

    This is a validation boundary, not a proposal fallback.  The Host never
    invents a replacement direction: an impossible model response is returned
    to the Researcher through the existing semantic repair loop.
    """

    directions = tuple(CandidateDirection.from_dict(item) for item in raw_directions)
    hard_avoid_digests = {
        str(item.get("behavior_digest"))
        for item in avoid_behaviors
        if isinstance(item, Mapping)
        and isinstance(item.get("behavior_digest"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(item.get("behavior_digest")))
    }
    prior_behavior_digests = {
        item["behavior_digest"] for item in prior_generation_behaviors
    }
    registry = current_program_registry()
    _boundary, parameter_schemas = _genome_parameter_boundary(task, parent)
    parent_parameters = parent.to_dict()["scientific_program"][
        "parameter_overrides"
    ]
    occupied_behavior_digests: set[str] = set()
    checks: list[dict[str, Any]] = []
    # A complete run archive can outgrow the legacy advisory window. Do not
    # mistake exhaustion of a small sample of continuous values for no space.
    witness_count = (len(hard_avoid_digests | prior_behavior_digests) + len(directions) + 8
                     if novelty.enabled(task.metadata) else 0)

    for index, direction in enumerate(directions):
        # Authored programs/directives have an open composition space. A small
        # set of Host witnesses cannot prove that history exhausted that space;
        # the proposer checks its actual complete behavior against all history.
        direction_avoid_digests = hard_avoid_digests | (
            prior_behavior_digests if direction.mutation_axis in {
                "scientific_parameter", "registered_predictor", "instruction_profile",
                "instruction_parameter", "workflow_parameter",
            } else set()
        )
        if direction.mutation_axis == "scientific_parameter":
            schema = parameter_schemas.get(direction.mutation_target)
            if not isinstance(schema, Mapping):
                raise ValueError(
                    f"candidate_directions[{index}] has no frozen parameter schema"
                )
            values = _parameter_preflight_values(
                schema,
                parent_parameters.get(direction.mutation_target),
                str(direction.mutation_direction),
                minimum_witness_count=witness_count,
            )
            operations = tuple(
                {
                    "op": "set_bounded_parameter",
                    "name": direction.mutation_target,
                    "value": value,
                }
                for value in values
            )
        elif direction.mutation_axis == "registered_predictor":
            operations = (
                {
                    "op": "select_registered_pipeline",
                    "predictor_id": direction.mutation_target,
                },
            )
        elif direction.mutation_axis == "instruction_profile":
            operations = (
                {
                    "op": "select_instruction_template",
                    "role": "sample-planner",
                    "instruction_template_id": direction.mutation_target,
                },
            )
        elif direction.mutation_axis in {
            "instruction_parameter",
            "workflow_parameter",
        }:
            # Spelled out per axis rather than left to a catch-all `else`: the
            # witnesses have to come from the *registered* contract for this axis,
            # and an axis that fell through to someone else's operation would fail
            # preflight with a message naming the wrong knob.
            contract, current_value = _registered_scalar_axis_contract(
                parent, direction.mutation_axis, direction.mutation_target
            )
            if contract is None:
                raise ValueError(
                    f"candidate_directions[{index}] names no registered "
                    f"{direction.mutation_axis} target"
                )
            operations = tuple(
                {
                    "op": MUTATION_OPERATION_BY_AXIS[direction.mutation_axis],
                    **(
                        {"role": "sample-planner"}
                        if direction.mutation_axis == "instruction_parameter"
                        else {}
                    ),
                    "name": direction.mutation_target,
                    "value": value,
                }
                for value in _parameter_preflight_values(
                    _scalar_axis_preflight_schema(contract),
                    current_value,
                    str(direction.mutation_direction),
                    minimum_witness_count=witness_count,
                )
            )
        elif direction.mutation_axis == "strategy_program":
            current = resolve_candidate_agent_profile(parent, registry).get("strategy_program")
            if (direction.mutation_direction == "author") == (current is not None):
                raise ValueError(
                    f"candidate_directions[{index}] ({direction.direction_id}): "
                    "strategy direction must author absent or revise existing program; "
                    f"frozen parent requires {'revise' if current is not None else 'author'}"
                )
            program = deepcopy(current) if current else seed_strategy_program()
            program["planner_prompt"] = "Check causal changes and use the computed strategy as measured evidence."
            if program == current:
                program["planner_prompt"] = "Compare actual tool evidence by horizon before submitting."
            operations = ({"op": "author_strategy_program", "role": "sample-planner", "strategy_program": program},)
        elif direction.mutation_axis == "skill_program":
            current = resolve_candidate_agent_profile(parent, registry).get("skill_program")
            if (direction.mutation_direction == "author") == (current is not None):
                raise ValueError(
                    f"candidate_directions[{index}] ({direction.direction_id}): "
                    "skill direction must author an absent program or revise an existing one; "
                    f"frozen parent requires {'revise' if current is not None else 'author'}"
                )
            operations = tuple({"op": "author_skill_program", "role": "sample-planner", "skill_program": program}
                               for program in skill_preflight_programs() if program != current)
        elif direction.mutation_axis == "feature_recipe":
            current = parent.to_dict()["scientific_program"].get("feature_recipe")
            if (direction.mutation_direction == "author") == (current is not None):
                raise ValueError(
                    f"candidate_directions[{index}] ({direction.direction_id}): "
                    "feature direction must author an absent recipe or revise an existing one; "
                    f"frozen parent requires {'revise' if current is not None else 'author'}"
                )
            from ..evaluators.greenhouse_prediction import seed_feature_recipe
            recipe = deepcopy(current) if current else seed_feature_recipe(horizons=task.metadata.get("fitness_profile", {}).get("expected_horizons", (1, 6, 24)))
            operations_list = []
            for lag in (4, 5, 6):
                alternative = deepcopy(recipe)
                alternative["features"][0] = {"op": "target_lag", "k": lag}
                operations_list.append({"op": "author_feature_recipe", "feature_recipe": alternative})
            operations = tuple(operations_list)
        elif direction.mutation_axis == "instruction_directive":
            # Not a scalar move, so there are no numeric witnesses: the
            # witnesses are whole directives, and the preflight proves the axis
            # is reachable at all by building ones that differ from the parent.
            operations = _directive_preflight_operations(
                parent, str(direction.mutation_direction)
            )
            if not operations:
                had = _has_authored_directive(parent, "sample-planner")
                reason = (
                    "this candidate already authored a directive, so it can "
                    "only be revised"
                    if direction.mutation_direction == "author" and had
                    else "this candidate has no directive to revise"
                    if direction.mutation_direction == "revise" and not had
                    else "the sample-planner has no enabled tool a directive "
                    "could plan"
                )
                raise ValueError(
                    f"candidate_directions[{index}] cannot be realized: {reason}"
                )
        else:
            # `CandidateDirection.from_dict` already rejects axes outside
            # `CANDIDATE_MUTATION_AXES`, so reaching here means that set grew
            # without a preflight implementation. Say that, instead of proving an
            # unrelated operation legal.
            raise ValueError(
                f"candidate_directions[{index}] axis "
                f"{direction.mutation_axis} has no preflight implementation"
            )

        preflight_context = GenomeMutationContextV1(
            run_id=run.run_id,
            generation=run.generation,
            slot_index=index,
            slot_seed=index,
            parent_candidate_id=(
                None if run.generation == 0 else "candidate:direction-preflight-parent"
            ),
            parent_genome_digest=parent.genome_digest,
            generation_batch_digest=digest(
                {"kind": "direction-preflight", "run_id": run.run_id}
            ),
            research_iteration_digest=digest(
                {"kind": "direction-preflight-research", "generation": run.generation}
            ),
            knowledge_snapshot_digest=digest(
                {"kind": "direction-preflight-knowledge", "generation": run.generation}
            ),
            mutation_budget_digest=digest(
                {
                    "kind": "direction-preflight-budget",
                    "direction_digest": direction.direction_digest,
                }
            ),
            mutation_operator_id=mutation_policy_contract(task.metadata)["mutation_operator_id"],
        )
        valid_child: EcologyEvolutionPluginGenome | None = None
        last_error: BaseException | None = None
        hard_avoid_conflicts = 0
        sibling_conflicts = 0
        for operation in operations:
            mutation = {
                "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                "operations": [operation],
            }
            try:
                child = apply_genome_mutation(
                    parent,
                    mutation,
                    preflight_context,
                    registry,
                    parameter_schemas=parameter_schemas,
                )
                compile_plugin_behavior(
                    child,
                    task,
                    None,
                    registry,
                    compiler_semantic_digest=str(
                        task.metadata.get("compiler_semantic_digest")
                        or DEFAULT_COMPILER_SEMANTIC_DIGEST
                    ),
                )
            except (TypeError, ValueError) as exc:
                last_error = exc
                continue
            if child.behavior_digest in direction_avoid_digests:
                hard_avoid_conflicts += 1
                continue
            if child.behavior_digest in occupied_behavior_digests:
                sibling_conflicts += 1
                continue
            valid_child = child
            break

        if valid_child is None:
            if hard_avoid_conflicts and sibling_conflicts:
                detail = (
                    "every legal witness is blocked by a hard-avoided or "
                    "already allocated sibling behavior"
                )
            elif hard_avoid_conflicts:
                detail = "every legal witness replays a hard-avoided behavior"
            elif sibling_conflicts:
                detail = "every legal witness duplicates another candidate direction"
            elif last_error is not None:
                detail = public_error_summary(str(last_error), limit=300)
            else:
                detail = "no changed value exists inside the frozen trust region"
            raise ValueError(
                f"candidate_directions[{index}] is not realizable as one legal "
                f"operation: {detail}"
            )
        occupied_behavior_digests.add(valid_child.behavior_digest)
        check = {
            "direction_id": direction.direction_id,
            "mutation_axis": direction.mutation_axis,
            "mutation_target": direction.mutation_target,
            "mutation_direction": direction.mutation_direction,
            "operation": operations[0]["op"] if operations else None,
            "status": "single_operation_feasible",
            "witness_behavior_digest": valid_child.behavior_digest,
        }
        if direction.mutation_axis != "scientific_parameter":
            check["deterministic_behavior_digest"] = valid_child.behavior_digest
        checks.append(check)

    return {
        "schema_version": "ecologyrsi-dsh.candidate-direction-preflight/1",
        "parent_behavior_digest": parent.behavior_digest,
        "hard_avoid_behavior_count": len(hard_avoid_digests),
        "checks": checks,
    }








def _task_parameter_space(metadata: Mapping[str, Any], domain: str):
    if domain != "greenhouse":
        raise ValueError("Only registered greenhouse prediction is supported")
    predictor_id = str(metadata.get("prediction_model_id") or BASELINE_ALIGNED_RIDGE_MODEL_ID)
    families = {
        BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID: ("greenhouse_baseline_aligned_cellwise_ridge", "_GREENHOUSE_HORIZON_TARGETWISE_RIDGE_SCHEMAS"),
        BASELINE_ALIGNED_RIDGE_MODEL_ID: ("greenhouse_baseline_aligned_ridge", "_GREENHOUSE_ALIGNED_RIDGE_SCHEMAS"),
        EXOGENOUS_RIDGE_MODEL_ID: ("greenhouse_ridge", "_GREENHOUSE_RIDGE_SCHEMAS"),
        TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID: ("greenhouse_targetwise_ridge", "_GREENHOUSE_TARGETWISE_RIDGE_SCHEMAS"),
        HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID: ("greenhouse_horizon_targetwise_ridge", "_GREENHOUSE_HORIZON_TARGETWISE_RIDGE_SCHEMAS"),
        RECIPE_RIDGE_MODEL_ID: ("greenhouse_recipe_ridge", "_GREENHOUSE_RECIPE_RIDGE_SCHEMAS"),
    }
    if predictor_id not in families:
        raise ValueError(f"unsupported prediction_model_id: {predictor_id}")
    boundary, schemas = families[predictor_id]
    return boundary, getattr(StrategyRouterDSHAdapter, schemas), ()
