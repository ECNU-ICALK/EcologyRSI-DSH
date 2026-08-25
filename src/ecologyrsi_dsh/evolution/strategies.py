"""Small DSH boundary used by the local evolution mode.

The real DSH integration can implement the same protocol later.  The default
adapter only emits structured parameter proposals; it never executes arbitrary
code or receives evaluator internals.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, ClassVar, Protocol, runtime_checkable

from ..core.models import Proposal, Run, TaskManifest, canonical_json, digest
from ..core.redaction import public_error_summary, public_exception_summary
from ..evaluators.greenhouse_prediction import (
    EXOGENOUS_RIDGE_MODEL_ID,
    HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
)
from ..integrations.dsh_native_runtime import DSH_NATIVE_EXECUTION_PROTOCOL
from ..integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ..integrations.model_gateway import GatewayResponseError, ModelGateway
from ..knowledge.algorithm_ir import registered_algorithm_blueprint_catalog
from ..knowledge.algorithms import (
    PredictorAdoption,
    registered_predictor_evaluator_ids,
    resolve_predictor_adoption,
)
from ..knowledge.autonomous_cycle import (
    AUTONOMOUS_RESEARCH_PROTOCOL,
    CandidateDirection,
    GENERATION_REFLECTION_SCHEMA_VERSION,
    SEARCH_PLAN_SCHEMA_VERSION,
    GenerationReflection,
    GenerationSearchPlan,
    normalize_candidate_directions,
    normalize_search_queries,
    reject_executable_fields,
    validate_research_synthesis,
)
from ..knowledge.mapping import knowledge_focus_parameter
from ..knowledge.models import validate_knowledge_context
from ..knowledge.program_registry import current_program_registry
from ..knowledge.research_iteration import ResearchIteration
from .context import (
    analysis_focus_parameter,
    is_sensitive_context_field,
    parameter_semantics,
    safe_aggregate_feedback,
)
from .context import (
    batch_context as validate_batch_context,
)
from .genome import (
    TRUST_REGION_MAX_NORMALIZED_STEP,
    TRUST_REGION_MUTATION_OPERATOR_ID,
    EcologyEvolutionPluginGenome,
    GenomeMutationContextV1,
    apply_genome_mutation,
)
from .workflow_ir import (
    DEFAULT_COMPILER_SEMANTIC_DIGEST,
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
_FORECAST_CONTEXT_MAX_TARGETS = 32
_FORECAST_CONTEXT_MAX_HORIZONS = 32
_FORECAST_CONTEXT_MAX_CELLS = 256


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
        "dataset_ids": list(task.visible_datasets),
        "expected_targets": targets,
        "expected_horizons_hours": horizons,
        "target_horizon_cells": cells,
        "matrix_cell_count": len(cells),
        "prediction_cells_per_origin": configured_cell_count,
        "complete_matrix_required": bool(cells),
        "joint_prediction_call_per_origin": (
            task.metadata.get("sample_agent_protocol")
            == "dsh-strict-origin-bundle@3"
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
    return {
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


def _native_evolution_reflection_from_experience(
    experience: Mapping[str, Any] | None,
    *,
    current_run_id: str,
) -> dict[str, Any]:
    """Project failed effective behaviors into a compact Host-owned directive."""

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
        "schema_version": "ecologyrsi-dsh.evolution-reflection/2",
        "avoid_behaviors": avoid,
        "active_unresolved": (
            list(active_unresolved[:8])
            if isinstance(active_unresolved, list)
            else []
        ),
        "policy": {
            "exact_failed_behavior_replay": "reject_and_retry_once",
            "host_enforced": True,
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


def _guardrail_cohort_window_segments(
    window: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    population_count = int(window["population_count"])
    selected_count = int(window["selected_count"])
    window_offset = int(window["window_offset"])
    end = window_offset + selected_count
    if end <= population_count:
        return ((window_offset, end),)
    return ((window_offset, population_count), (0, end - population_count))


def _guardrail_cohort_windows_are_pairwise_non_overlapping(
    cohorts: list[Mapping[str, Any]],
) -> bool:
    if not cohorts or len(
        {
            (str(item["population_digest"]), int(item["population_count"]))
            for item in cohorts
        }
    ) != 1:
        return False
    for index, left in enumerate(cohorts):
        left_segments = _guardrail_cohort_window_segments(left)
        for right in cohorts[index + 1 :]:
            if any(
                max(left_start, right_start) < min(left_end, right_end)
                for left_start, left_end in left_segments
                for right_start, right_end in (
                    _guardrail_cohort_window_segments(right)
                )
            ):
                return False
    return True


def _protected_historical_parameter_values(
    plan: Mapping[str, Any],
    schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, int | float]:
    """Resolve only statistically verified, unambiguous host guardrails."""

    raw = plan.get(_HISTORICAL_PARAMETER_GUARDRAILS_FIELD)
    if not isinstance(raw, Mapping) or raw.get("policy") != (
        _HISTORICAL_PARAMETER_GUARDRAIL_POLICY
    ):
        return {}
    requirements = raw.get("evidence_requirements")
    expected_requirements = {
        "minimum_independent_cohort_count": (
            _HISTORICAL_PARAMETER_GUARDRAIL_MIN_COHORTS
        ),
        "minimum_cell_sample_count_per_cohort": (
            _HISTORICAL_PARAMETER_GUARDRAIL_MIN_CELL_N_PER_COHORT
        ),
        "minimum_total_cell_sample_count": (
            _HISTORICAL_PARAMETER_GUARDRAIL_MIN_TOTAL_CELL_N
        ),
        "same_cohort_generations_count_once": True,
        "requires_pairwise_non_overlapping_cohorts": True,
        "requires_nonnegative_skill_in_every_observation": True,
        "requires_zero_constraint_violations_in_every_observation": True,
    }
    if not isinstance(requirements, Mapping) or dict(requirements) != (
        expected_requirements
    ):
        return {}
    evidence_rows = raw.get("protected_parameter_evidence")
    if not isinstance(evidence_rows, (list, tuple)) or len(evidence_rows) > 16:
        return {}

    protected: dict[str, int | float] = {}
    rejected: set[str] = set()
    for row in evidence_rows:
        if not isinstance(row, Mapping):
            continue
        parameter = row.get("parameter")
        if not isinstance(parameter, str) or parameter not in schemas:
            continue
        if row.get("policy") != _HISTORICAL_PARAMETER_GUARDRAIL_POLICY:
            rejected.add(parameter)
            continue
        raw_skill = row.get("skill_score")
        cohorts = row.get("cohort_evidence")
        if (
            isinstance(raw_skill, bool)
            or not isinstance(raw_skill, (int, float))
            or not math.isfinite(float(raw_skill))
            or float(raw_skill) < 0
            or not isinstance(cohorts, (list, tuple))
        ):
            rejected.add(parameter)
            continue
        cohort_digests: set[str] = set()
        total_cell_sample_count = 0
        cohort_evidence_valid = True
        validated_cohorts: list[Mapping[str, Any]] = []
        for cohort in cohorts:
            if not isinstance(cohort, Mapping):
                cohort_evidence_valid = False
                break
            cohort_digest = cohort.get("evaluation_cohort_digest")
            feedback_cohort_digest = cohort.get(
                "feedback_update_cohort_digest"
            )
            cell_sample_count = cohort.get("cell_sample_count")
            minimum_skill_score = cohort.get("minimum_skill_score")
            population_count = cohort.get("population_count")
            population_digest = cohort.get("population_digest")
            selected_count = cohort.get("selected_count")
            window_offset = cohort.get("window_offset")
            if (
                cohort.get("schema_version")
                != _HISTORICAL_PARAMETER_GUARDRAIL_COHORT_SCHEMA
                or cohort.get("selection_policy")
                != _HISTORICAL_PARAMETER_GUARDRAIL_SELECTION_POLICY
                or not isinstance(cohort_digest, str)
                or len(cohort_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in cohort_digest
                )
                or cohort_digest in cohort_digests
                or not isinstance(feedback_cohort_digest, str)
                or len(feedback_cohort_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in feedback_cohort_digest
                )
                or isinstance(cell_sample_count, bool)
                or not isinstance(cell_sample_count, int)
                or cell_sample_count
                < _HISTORICAL_PARAMETER_GUARDRAIL_MIN_CELL_N_PER_COHORT
                or isinstance(population_count, bool)
                or not isinstance(population_count, int)
                or population_count <= 0
                or not isinstance(population_digest, str)
                or len(population_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in population_digest
                )
                or isinstance(selected_count, bool)
                or not isinstance(selected_count, int)
                or selected_count <= 0
                or selected_count > population_count
                or cell_sample_count > selected_count
                or isinstance(window_offset, bool)
                or not isinstance(window_offset, int)
                or window_offset < 0
                or window_offset >= population_count
                or isinstance(minimum_skill_score, bool)
                or not isinstance(minimum_skill_score, (int, float))
                or not math.isfinite(float(minimum_skill_score))
                or float(minimum_skill_score) < 0
            ):
                cohort_evidence_valid = False
                break
            cohort_digests.add(cohort_digest)
            total_cell_sample_count += cell_sample_count
            validated_cohorts.append(cohort)
        if (
            not cohort_evidence_valid
            or len(cohort_digests) < _HISTORICAL_PARAMETER_GUARDRAIL_MIN_COHORTS
            or total_cell_sample_count
            < _HISTORICAL_PARAMETER_GUARDRAIL_MIN_TOTAL_CELL_N
            or row.get("independent_cohort_count") != len(cohort_digests)
            or row.get("total_cell_sample_count") != total_cell_sample_count
            or not _guardrail_cohort_windows_are_pairwise_non_overlapping(
                validated_cohorts
            )
        ):
            rejected.add(parameter)
            continue
        try:
            bounded = _bounded_parameters(
                {parameter: row.get("value")},
                schemas,
                partial=True,
                source="historical_parameter_guardrails",
            )[parameter]
        except (TypeError, ValueError):
            rejected.add(parameter)
            continue
        previous = protected.get(parameter)
        if previous is not None and previous != bounded:
            rejected.add(parameter)
            continue
        protected[parameter] = bounded
    return {
        parameter: value
        for parameter, value in sorted(protected.items())
        if parameter not in rejected
    }


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
        interventions: Mapping[str, Any] | None = None,
        batch_context: Mapping[str, Any] | None = None,
    ) -> Proposal:
        """Return one structured proposal for the current generation."""

    def close_session(self, session_id: str) -> None:
        """Release adapter-side session state."""


class FakeDSHAdapter:
    """Deterministic local adapter for tests and the first demo.

    It emits a small set of crop-soil-water parameter candidates.  The values
    are ordinary JSON, so a future DSH plugin can display or edit them without
    introducing a code sandbox into the core.
    """

    _PARAMETERS = (
        {"alpha": 0.20, "window": 3, "water_threshold": 0.35},
        {"alpha": 0.35, "window": 5, "water_threshold": 0.40},
        {"alpha": 0.50, "window": 7, "water_threshold": 0.45},
        {"alpha": 0.65, "window": 9, "water_threshold": 0.50},
    )

    def __init__(self, *, max_proposals: int = 256) -> None:
        if isinstance(max_proposals, bool) or not isinstance(max_proposals, int) or max_proposals < 1:
            raise ValueError("max_proposals must be a positive integer")
        self.max_proposals = min(max_proposals, 256)
        self._session_counts: dict[tuple[str, int], int] = {}

    def open_session(self, run: Run, task: TaskManifest) -> str:
        session_id = f"fake-dsh:{run.run_id}"
        self._session_counts.setdefault((session_id, run.generation), 0)
        return session_id

    def propose(
        self,
        run: Run,
        task: TaskManifest,
        session_id: str,
        *,
        parent_candidate_id: str | None = None,
        parent_context: Mapping[str, Any] | None = None,
        interventions: Mapping[str, Any] | None = None,
        batch_context: Mapping[str, Any] | None = None,
    ) -> Proposal:
        key = (session_id, run.generation)
        if not any(item[0] == session_id for item in self._session_counts):
            if session_id != f"fake-dsh:{run.run_id}":
                raise RuntimeError("DSH session is not open")
            self._session_counts[key] = 0
        # A local server can be restarted while the event ledger still says
        # that the run is active.  Rehydrate the generation counter lazily for
        # this deterministic adapter; a real DSH adapter would restore its
        # session through the gateway instead.
        self._session_counts.setdefault(key, 0)
        index = self._session_counts[key]
        if index >= self.max_proposals:
            raise StopIteration("fake adapter proposal budget exhausted")
        self._session_counts[key] = index + 1
        # Generation participates in the deterministic schedule so advancing
        # the run actually explores a new parameter set.  The per-generation
        # counter remains local, which also lets a restarted process replay
        # the next unseen proposal ID safely.
        parameter_index = run.generation + index
        base = self._PARAMETERS[parameter_index % len(self._PARAMETERS)]
        cycle = parameter_index // len(self._PARAMETERS)
        # Keep proposals structured and bounded even when the UI requests a
        # larger local budget than the four hand-written seed configurations.
        params = {
            "alpha": round(min(0.95, base["alpha"] + 0.03 * cycle), 4),
            "window": min(30, base["window"] + 2 * cycle),
            "water_threshold": round(min(0.85, base["water_threshold"] + 0.01 * cycle), 4),
        }
        return Proposal(
            proposal_id=f"proposal:{run.run_id}:{run.generation}:{index + 1}",
            run_id=run.run_id,
            generation=run.generation,
            title=f"toy parameter set {index + 1}",
            changes=params,
            parent_candidate_id=parent_candidate_id,
            rationale="deterministic FakeDSH proposal",
        )

    def close_session(self, session_id: str) -> None:
        for key in tuple(self._session_counts):
            if key[0] == session_id:
                self._session_counts.pop(key, None)


class StrategyRouterDSHAdapter:
    """Route a task to bounded parameter proposal strategies.

    The authenticated strategy delegates reasoning to ``ModelGateway`` but
    keeps parameter names and ranges under host control.  No strategy accepts
    source code or executable content.
    """

    SUPPORTED_STRATEGIES = (
        "parameter_sweep@1",
        "adaptive_local@1",
        "dsh_authenticated@1",
        "autonomous_model@1",
    )
    _STRATEGY_DESCRIPTORS: ClassVar[dict[str, dict[str, Any]]] = {
        "parameter_sweep@1": {
            "label": "有界参数扫描",
            "description": "继承已完成父候选参数，每轮只扫描一个有界维度。",
            "requires_authenticated_model": False,
            "implementation": "bounded-parent-sweep/6",
        },
        "adaptive_local@1": {
            "label": "局部自适应搜索",
            "description": "根据父候选反馈确定调整方向与步长。",
            "requires_authenticated_model": False,
            "implementation": "bounded-feedback-local-search/6",
        },
        "dsh_authenticated@1": {
            "label": "DSH 模型提案",
            "description": "由已安全配置的服务端模型提出参数，宿主继续执行范围校验。",
            "requires_authenticated_model": True,
            "implementation": "authenticated-structured-proposal/7",
        },
        "autonomous_model@1": {
            "label": "模型自主调研与进化",
            "description": (
                "由策略模型检索公开知识、组建模型团队并提出预测模型与搜索策略；"
                "宿主只执行已登记且有界的参数方案。"
            ),
            "requires_authenticated_model": True,
            "implementation": "model-search-reflect-candidate-directions/19",
        },
    }

    _TOY_SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95},
        "window": {"type": "integer", "minimum": 1, "maximum": 30},
        "water_threshold": {"type": "number", "minimum": 0.05, "maximum": 0.85},
    }
    _GREENHOUSE_SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "blend": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "window": {"type": "integer", "minimum": 1, "maximum": 48},
        "bias_scale": {"type": "number", "minimum": 0.0, "maximum": 2.0},
    }
    _GREENHOUSE_RIDGE_SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "history_steps": {"type": "integer", "minimum": 1, "maximum": 12},
        "ridge_alpha": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
        "residual_scale": {"type": "number", "minimum": 0.0, "maximum": 1.0},
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
    _TOY_SWEEP = (
        {"alpha": 0.20, "window": 3, "water_threshold": 0.35},
        {"alpha": 0.35, "window": 5, "water_threshold": 0.40},
        {"alpha": 0.50, "window": 7, "water_threshold": 0.45},
        {"alpha": 0.65, "window": 9, "water_threshold": 0.50},
    )
    _GREENHOUSE_SWEEP = (
        {"blend": 0.20, "window": 3, "bias_scale": 0.50},
        {"blend": 0.40, "window": 6, "bias_scale": 0.80},
        {"blend": 0.60, "window": 12, "bias_scale": 1.00},
        {"blend": 0.80, "window": 18, "bias_scale": 1.20},
    )
    _GREENHOUSE_INITIAL_DESIGN = (
        # Persistence-equivalent diagnostic anchor. The window is immaterial
        # to the formula at blend=1, but remains explicit and in range.
        {"blend": 1.00, "window": 24, "bias_scale": 0.00},
        {"blend": 0.93, "window": 25, "bias_scale": 0.00},
        {"blend": 0.87, "window": 24, "bias_scale": 0.00},
        {"blend": 0.87, "window": 48, "bias_scale": 0.00},
    )
    _GREENHOUSE_RIDGE_SWEEP = (
        {"history_steps": 3, "ridge_alpha": 0.01, "residual_scale": 0.50},
        {"history_steps": 6, "ridge_alpha": 0.05, "residual_scale": 0.75},
        {"history_steps": 9, "ridge_alpha": 0.10, "residual_scale": 1.00},
        {"history_steps": 12, "ridge_alpha": 0.50, "residual_scale": 1.00},
    )
    _GREENHOUSE_TARGETWISE_RIDGE_SWEEP = (
        {
            "history_steps": 3,
            "ridge_alpha": 0.01,
            "air_temperature_residual_scale": 0.50,
            "relative_humidity_residual_scale": 0.50,
            "co2_concentration_residual_scale": 0.00,
        },
        {
            "history_steps": 6,
            "ridge_alpha": 0.05,
            "air_temperature_residual_scale": 0.75,
            "relative_humidity_residual_scale": 0.75,
            "co2_concentration_residual_scale": 0.00,
        },
        {
            "history_steps": 9,
            "ridge_alpha": 0.10,
            "air_temperature_residual_scale": 1.00,
            "relative_humidity_residual_scale": 1.00,
            "co2_concentration_residual_scale": 0.25,
        },
        {
            "history_steps": 12,
            "ridge_alpha": 0.50,
            "air_temperature_residual_scale": 1.00,
            "relative_humidity_residual_scale": 1.00,
            "co2_concentration_residual_scale": 0.50,
        },
    )
    _GREENHOUSE_HORIZON_TARGETWISE_RIDGE_SWEEP = (
        {
            "history_steps": 3,
            "ridge_alpha": 0.01,
            "air_temperature_1h_residual_scale": 0.50,
            "air_temperature_6h_residual_scale": 0.50,
            "air_temperature_24h_residual_scale": 0.50,
            "relative_humidity_1h_residual_scale": 0.50,
            "relative_humidity_6h_residual_scale": 0.50,
            "relative_humidity_24h_residual_scale": 0.50,
            "co2_concentration_1h_residual_scale": 0.00,
            "co2_concentration_6h_residual_scale": 0.50,
            "co2_concentration_24h_residual_scale": 0.50,
        },
        {
            "history_steps": 6,
            "ridge_alpha": 0.05,
            "air_temperature_1h_residual_scale": 0.75,
            "air_temperature_6h_residual_scale": 0.75,
            "air_temperature_24h_residual_scale": 0.75,
            "relative_humidity_1h_residual_scale": 0.75,
            "relative_humidity_6h_residual_scale": 0.75,
            "relative_humidity_24h_residual_scale": 0.75,
            "co2_concentration_1h_residual_scale": 0.00,
            "co2_concentration_6h_residual_scale": 0.75,
            "co2_concentration_24h_residual_scale": 0.75,
        },
        {
            "history_steps": 9,
            "ridge_alpha": 0.10,
            "air_temperature_1h_residual_scale": 1.00,
            "air_temperature_6h_residual_scale": 1.00,
            "air_temperature_24h_residual_scale": 1.00,
            "relative_humidity_1h_residual_scale": 1.00,
            "relative_humidity_6h_residual_scale": 1.00,
            "relative_humidity_24h_residual_scale": 1.00,
            "co2_concentration_1h_residual_scale": 0.00,
            "co2_concentration_6h_residual_scale": 1.00,
            "co2_concentration_24h_residual_scale": 1.00,
        },
        {
            "history_steps": 12,
            "ridge_alpha": 0.50,
            "air_temperature_1h_residual_scale": 1.00,
            "air_temperature_6h_residual_scale": 1.00,
            "air_temperature_24h_residual_scale": 1.00,
            "relative_humidity_1h_residual_scale": 1.00,
            "relative_humidity_6h_residual_scale": 1.00,
            "relative_humidity_24h_residual_scale": 1.00,
            "co2_concentration_1h_residual_scale": 0.25,
            "co2_concentration_6h_residual_scale": 1.00,
            "co2_concentration_24h_residual_scale": 1.00,
        },
    )

    def __init__(
        self,
        gateway: ModelGateway | None = None,
        *,
        max_proposals: int = 256,
        native_runtime_provider: Callable[[], Any] | None = None,
    ) -> None:
        if isinstance(max_proposals, bool) or not isinstance(max_proposals, int) or max_proposals < 1:
            raise ValueError("max_proposals must be a positive integer")
        self.gateway = gateway or ModelGateway.from_env()
        self.max_proposals = min(max_proposals, 256)
        self._session_counts: dict[tuple[str, int], int] = {}
        self._session_plans: dict[str, dict[str, Any]] = {}
        self._resolved_plan_sessions: set[str] = set()
        self._native_runtime_provider = native_runtime_provider

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
        if task.metadata.get("execution_protocol") == DSH_NATIVE_EXECUTION_PROTOCOL:
            session_id = f"dsh-native:{run.run_id}"
            self._session_counts.setdefault((session_id, run.generation), 0)
            return session_id
        session_id = f"strategy-dsh:{run.run_id}"
        self._session_counts.setdefault((session_id, run.generation), 0)
        frozen_plan = task.metadata.get("autonomous_plan")
        if isinstance(frozen_plan, Mapping):
            # The creation-time plan is part of the immutable manifest.  Seed
            # the in-memory session from it so the first proposal reuses the
            # exact plan instead of issuing a second network request.
            self._session_plans.setdefault(session_id, dict(frozen_plan))
            if frozen_plan:
                self._resolved_plan_sessions.add(session_id)
        else:
            self._session_plans.setdefault(session_id, {})
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
                "next_generation_research_synthesis_preflight_required": True,
                "candidate_directions_must_be_distinct": True,
                **deepcopy(mutation_catalog),
                "cite_only_frozen_evidence": True,
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
                for index, direction in enumerate(directions):
                    _validate_direction_claim_scope(direction, task=task)
                    _reject_direction_explicit_parameter_assignments(
                        direction,
                        parent=parent,
                        index=index,
                    )
                    if any(
                        reference not in allowed_evidence_refs
                        for reference in direction.evidence_refs
                    ):
                        raise ValueError(
                            "generation reflection cited evidence outside the "
                            "frozen snapshot"
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
        expert_collaboration: Mapping[str, Any] | None = None,
        parent_genome: Mapping[str, Any] | None = None,
        generation_search_plan: Mapping[str, Any] | None = None,
        previous_generation_reflection: Mapping[str, Any] | None = None,
        candidate_count: int | None = None,
        run_state_revision: int = 0,
        stage_attempt: int = 1,
        ledger_expected_revision: int = 0,
    ) -> dict[str, Any]:
        """Resolve one bounded research plan for an autonomous run.

        ``ModelGateway`` provides the strict JSON contract.  Lightweight test
        gateways from older integrations may not expose ``research_plan``;
        in that case we return an explicit host-fallback plan rather than
        failing the entire compatibility path.
        """

        if task.metadata.get("execution_protocol") == DSH_NATIVE_EXECUTION_PROTOCOL:
            if not isinstance(parent_genome, Mapping):
                raise ValueError("DSH-native research requires the frozen parent genome")
            parent = EcologyEvolutionPluginGenome.from_dict(parent_genome)
            context = {
                "generation": run.generation,
                "objective": task.objective,
                "forecast_objective": _forecast_objective_context(task),
                "parent_genome": parent.to_dict(),
                "parent_plan": dict(current_plan or {}),
                "previous_generation_analysis": previous_generation_analysis,
                "knowledge_snapshot": knowledge_snapshot,
                "previous_knowledge_assessment": previous_knowledge_assessment,
                "cross_generation_experience": cross_generation_experience,
                "expert_collaboration": expert_collaboration,
            }
            active_search_cycle = _model_search_cycle_enabled(task)
            if active_search_cycle:
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
                            "exactly_one_operation_per_direction": True,
                            "one_operation_must_fully_test_the_hypothesis": True,
                            **deepcopy(mutation_catalog),
                            "hard_avoid_behaviors": deepcopy(
                                evolution_reflection["avoid_behaviors"]
                            ),
                            "diagnostic_insufficient_evidence_is_advisory": True,
                            "cite_only_frozen_evidence": True,
                            "registered_capabilities_only": True,
                            "model_generated_code_execution": False,
                        },
                    }
                )
            stage = (
                "generation.research-synthesis"
                if active_search_cycle
                else "generation.research"
            )
            output_schema_id = (
                "ecology-research-synthesis@1"
                if active_search_cycle
                else "ecology-research-result@1"
            )
            if active_search_cycle:
                feedback: dict[str, Any] | None = None
                for contract_attempt in range(
                    1,
                    _NATIVE_MAX_SEMANTIC_CONTRACT_ATTEMPTS + 1,
                ):
                    stage_context = {
                        **deepcopy(context),
                        "host_validation_feedback": deepcopy(feedback),
                    }
                    dsh_stage_attempt = _semantic_contract_stage_attempt(
                        stage_attempt,
                        contract_attempt,
                    )
                    structured = self._native_runtime().run(
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
                            f"attempt:{dsh_stage_attempt}"
                        ),
                        identity_digests=self._native_stage_identity(
                            parent, task, run, stage, stage_context
                        ),
                    )
                    try:
                        reject_executable_fields(structured)
                        normalized = validate_research_synthesis(
                            structured,
                            candidate_count=candidate_count,
                            allowed_evidence_refs=allowed_evidence_refs,
                            allowed_mutation_targets=allowed_mutation_targets,
                        )
                        direction_preflight = _validate_candidate_direction_realizability(
                            normalized["candidate_directions"],
                            run=run,
                            task=task,
                            parent=parent,
                            avoid_behaviors=evolution_reflection[
                                "avoid_behaviors"
                            ],
                        )
                    except (TypeError, ValueError) as exc:
                        if (
                            contract_attempt
                            >= _NATIVE_MAX_SEMANTIC_CONTRACT_ATTEMPTS
                        ):
                            raise
                        feedback = _host_validation_feedback(
                            exc,
                            rejected_output=structured,
                            attempt=contract_attempt,
                        )
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
                            "previous_generation_reflection": (
                                safe_aggregate_feedback(
                                    previous_generation_reflection,
                                    name=(
                                        "research previous generation "
                                        "reflection plan"
                                    ),
                                )
                            ),
                            "dsh_evolution_reflection": evolution_reflection,
                        }
                    )
                    return plan
                raise RuntimeError("research synthesis repair loop did not terminate")
            structured = self._native_runtime().run(
                run_id=run.run_id,
                stage=stage,
                role="researcher",
                context=context,
                output_schema_id=output_schema_id,
                run_state_revision=run_state_revision,
                stage_attempt=stage_attempt,
                ledger_expected_revision=ledger_expected_revision,
                idempotency_key=(
                    f"{run.run_id}:generation:{run.generation}:{stage}:"
                    f"attempt:{stage_attempt}"
                ),
                identity_digests=self._native_stage_identity(
                    parent, task, run, stage, context
                ),
            )
            if structured.get("schema_version") != "ecology-research-result@1":
                raise ValueError("DSH researcher returned an unsupported schema")
            plan = dict(current_plan or {})
            plan.update(
                {
                    "status": "model_generated",
                    "dsh_research_summary": structured.get("summary"),
                    "dsh_research_evidence": list(structured.get("evidence") or []),
                    "dsh_evolution_reflection": (
                        _native_evolution_reflection_from_experience(
                            cross_generation_experience,
                            current_run_id=run.run_id,
                        )
                    ),
                }
            )
            return plan

        boundary, registered_schemas = _task_parameter_boundary(task, current_plan)
        active_schemas = parameter_schemas or registered_schemas
        objective_profile = task.metadata.get("objective_profile")
        hard_gates = task.metadata.get("hard_gates")
        if isinstance(objective_profile, Mapping):
            hard_gates = objective_profile.get("hard_gates", hard_gates)
        runtime_catalog = task.metadata.get("runtime_component_catalog")
        raw_predictors = (
            runtime_catalog.get("prediction_models")
            if isinstance(runtime_catalog, Mapping)
            else None
        )
        compatible_predictor_ids = [
            str(item.get("id"))
            for item in raw_predictors
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        ] if isinstance(raw_predictors, list) else []
        if not compatible_predictor_ids:
            compatible_predictor_ids = [
                str(task.metadata.get("prediction_model_id") or "").strip()
            ]
        blueprint_catalog = []
        for predictor_id in compatible_predictor_ids:
            if not predictor_id:
                continue
            try:
                blueprint_catalog.extend(
                    registered_algorithm_blueprint_catalog((predictor_id,))
                )
            except ValueError:
                continue
        historical_parameter_guardrails = (
            _historical_parameter_guardrails_from_experience(
                cross_generation_experience
            )
        )
        context = {
            "run_id": run.run_id,
            "generation": run.generation,
            "objective": task.objective,
            "domain": _domain_name(task.metadata.get("domain", task.domain_pack)),
            "domain_pack": task.domain_pack,
            "research_domain": task.metadata.get(
                "research_domain", task.domain_pack
            ),
            "model_workflow": task.metadata.get(
                "model_workflow", "research_compile_evolve@1"
            ),
            "visible_datasets": list(task.visible_datasets),
            "seed": task.seed,
            "knowledge_online_enabled": bool(
                task.metadata.get("knowledge_online_enabled", True)
            ),
            "allowed_parameter_schemas": dict(active_schemas),
            "parameter_semantics": parameter_semantics(boundary, active_schemas),
            "runtime_component_catalog": runtime_catalog,
            "algorithm_blueprint_catalog": blueprint_catalog,
            "objective_profile": objective_profile,
            "hard_gates": hard_gates,
            "cross_generation_experience": safe_aggregate_feedback(
                cross_generation_experience,
                name="research cross_generation_experience",
            ),
            _HISTORICAL_PARAMETER_GUARDRAILS_FIELD: (
                historical_parameter_guardrails
            ),
            "expert_collaboration": safe_aggregate_feedback(
                expert_collaboration,
                name="research expert_collaboration",
            ),
            "host_boundary": {
                "no_code_execution": True,
                "only_registered_predictors_can_run": True,
                "only_registered_operator_graphs_can_run": True,
                "scientific_gate_is_host_controlled": True,
                "expert_answers_are_advisory": True,
                "expert_answers_cannot_expand_data_or_tool_permissions": True,
            },
        }
        if any(
            item is not None
            for item in (
                previous_generation_analysis,
                knowledge_snapshot,
                previous_knowledge_assessment,
                current_plan,
                cross_generation_experience,
                expert_collaboration,
            )
        ):
            context.update(
                {
                    "previous_generation_analysis": safe_aggregate_feedback(
                        previous_generation_analysis,
                        name="research previous_generation_analysis",
                    ),
                    "knowledge_snapshot": validate_knowledge_context(
                        knowledge_snapshot
                    ),
                    "previous_knowledge_assessment": safe_aggregate_feedback(
                        previous_knowledge_assessment,
                        name="research previous_knowledge_assessment",
                    ),
                    "current_research_plan": safe_aggregate_feedback(
                        current_plan,
                        name="research current_plan",
                    ),
                    "iteration_boundary": {
                        "one_frozen_result_per_generation": True,
                        "registered_capability_requests_only": True,
                        "model_generated_code_execution": False,
                    },
                }
            )
        planner = getattr(self.gateway, "research_plan", None)
        if callable(planner):
            plan = planner(model_id, context)
            if not isinstance(plan, Mapping):
                raise TypeError("model research plan must be an object")
            return dict(plan)
        return {
            "status": "host_fallback",
            "team": {
                "id": "host-validated-team@1",
                "name": "宿主验证团队",
                "roles": ["预测建模", "科学评测", "进化搜索"],
                "rationale": "兼容网关未提供研究计划接口，使用宿主已登记组件。",
            },
            "strategy": {
                "id": "autonomous_model@1",
                "name": "模型自主调研与进化",
                "rationale": "模型计划接口不可用时保留有界搜索。",
            },
            "research": [],
        }

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
        expert_collaboration: Mapping[str, Any] | None = None,
        parent_genome: Mapping[str, Any] | None = None,
        generation_search_plan: Mapping[str, Any] | None = None,
        previous_generation_reflection: Mapping[str, Any] | None = None,
        candidate_count: int | None = None,
        run_state_revision: int = 0,
        stage_attempt: int = 1,
        ledger_expected_revision: int = 0,
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
        try:
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
                expert_collaboration=expert_collaboration,
                parent_genome=parent_genome,
                generation_search_plan=generation_search_plan,
                previous_generation_reflection=previous_generation_reflection,
                candidate_count=candidate_count,
                run_state_revision=run_state_revision,
                stage_attempt=stage_attempt,
                ledger_expected_revision=ledger_expected_revision,
            )
        except GatewayResponseError as exc:
            if metadata.get("remote_fallback_policy") == "fail_run":
                raise
            fallback_plan = _host_guardrail_plan(
                {
                    **dict(current_plan),
                    "status": "unavailable",
                    "fallback_diagnostics": {
                        "stage": "research_iteration",
                        "reason": "gateway_error",
                        "error_type": type(exc).__name__,
                        "retryable": exc.retryable,
                        "algorithm_synthesis_status": (
                            "not_refreshed_due_gateway_error"
                        ),
                    },
                },
                historical_parameter_guardrails,
            )
            return {
                "status": "unavailable",
                "model_id": model_id,
                "plan": fallback_plan,
            }
        consultation = plan.pop("expert_consultation", None)
        plan_status = str(plan.get("status") or "").casefold()
        status = (
            plan_status
            if plan_status in {"host_fallback", "unavailable"}
            else "model_generated"
        )
        merged_plan = _host_guardrail_plan(
            _merge_research_plan(current_plan, plan),
            historical_parameter_guardrails,
        )
        result = {
            "status": status,
            "model_id": model_id,
            "plan": merged_plan,
        }
        if isinstance(consultation, Mapping):
            result["expert_consultation"] = dict(consultation)
        return result

    def propose(
        self,
        run: Run,
        task: TaskManifest,
        session_id: str,
        *,
        parent_candidate_id: str | None = None,
        parent_context: Mapping[str, Any] | None = None,
        interventions: Mapping[str, Any] | None = None,
        batch_context: Mapping[str, Any] | None = None,
    ) -> Proposal:
        if task.metadata.get("execution_protocol") == DSH_NATIVE_EXECUTION_PROTOCOL:
            return self._propose_native(
                run,
                task,
                session_id,
                parent_candidate_id=parent_candidate_id,
                batch_context=batch_context,
            )
        expected_session = f"strategy-dsh:{run.run_id}"
        if session_id != expected_session:
            raise RuntimeError("DSH session is not open")
        key = (session_id, run.generation)
        self._session_counts.setdefault(key, 0)
        index = self._session_counts[key]
        if index >= self.max_proposals:
            raise StopIteration("strategy adapter proposal budget exhausted")

        metadata = dict(task.metadata)
        autonomous_mode = bool(metadata.get("autonomous_mode", False))
        strategy_id = _optional_text(metadata.get("strategy_id"), "strategy_id")
        if autonomous_mode and strategy_id in (None, "parameter_sweep@1"):
            strategy_id = "autonomous_model@1"
        strategy_id = strategy_id or "parameter_sweep@1"
        if strategy_id not in self.SUPPORTED_STRATEGIES:
            raise ValueError(f"unsupported strategy_id: {strategy_id}")
        domain = _domain_name(metadata.get("domain", task.domain_pack))
        batch = validate_batch_context(batch_context, run)
        slot_index = int(batch["slot_index"]) if batch is not None else index
        batch_size = int(batch["batch_size"]) if batch is not None else 1
        previous_analysis = (
            batch.get("previous_generation_analysis") if batch is not None else None
        )
        knowledge_context = batch.get("knowledge_snapshot") if batch is not None else None
        research_iteration_context = (
            batch.get("research_iteration") if batch is not None else None
        )
        frozen_runtime_binding = (
            batch.get("frozen_runtime_binding") if batch is not None else None
        )
        _initial_boundary, initial_schemas, _initial_sweep = _task_parameter_space(
            metadata, domain
        )
        autonomous_plan: dict[str, Any] = {}
        predictor_adoption: PredictorAdoption | None = None
        planning_gateway_error: GatewayResponseError | None = None
        model_id: str | None = None
        if strategy_id in {"dsh_authenticated@1", "autonomous_model@1"}:
            model_id = _optional_text(
                metadata.get("strategy_model_id") or metadata.get("policy_model_id"),
                "strategy_model_id",
            )
            if model_id is None:
                raise ValueError(
                    f"{strategy_id} requires task.metadata.strategy_model_id"
                )
        if strategy_id == "autonomous_model@1":
            if isinstance(research_iteration_context, Mapping):
                research_iteration = ResearchIteration.from_dict(
                    research_iteration_context
                )
                autonomous_plan = dict(research_iteration.plan)
                predictor_adoption = PredictorAdoption.from_dict(
                    research_iteration.prediction_model_adoption
                )
                expected_adoption = resolve_predictor_adoption(
                    task,
                    autonomous_plan,
                )
                if predictor_adoption.to_dict() != expected_adoption.to_dict():
                    raise ValueError(
                        "research iteration predictor adoption does not match "
                        "the host catalog"
                    )
                self._session_plans[session_id] = dict(autonomous_plan)
                self._resolved_plan_sessions.add(session_id)
            elif isinstance(frozen_runtime_binding, Mapping):
                raw_plan = frozen_runtime_binding.get("plan")
                raw_adoption = frozen_runtime_binding.get(
                    "prediction_model_adoption"
                )
                if not isinstance(raw_plan, Mapping) or not isinstance(
                    raw_adoption, Mapping
                ):
                    raise TypeError("frozen runtime binding is incomplete")
                autonomous_plan = dict(raw_plan)
                predictor_adoption = PredictorAdoption.from_dict(raw_adoption)
                expected_adoption = resolve_predictor_adoption(task, autonomous_plan)
                if predictor_adoption.to_dict() != expected_adoption.to_dict():
                    raise ValueError(
                        "frozen runtime predictor adoption does not match the task catalog"
                    )
                self._session_plans[session_id] = dict(autonomous_plan)
                self._resolved_plan_sessions.add(session_id)
            elif session_id in self._resolved_plan_sessions:
                autonomous_plan = dict(self._session_plans.get(session_id, {}))
                predictor_adoption = resolve_predictor_adoption(task, autonomous_plan)
            else:
                frozen_plan = metadata.get("autonomous_plan")
                if isinstance(frozen_plan, Mapping) and frozen_plan:
                    autonomous_plan = dict(frozen_plan)
                    self._session_plans[session_id] = dict(autonomous_plan)
                    self._resolved_plan_sessions.add(session_id)
                else:
                    assert model_id is not None
                    try:
                        autonomous_plan = self.research_plan(
                            model_id,
                            run=run,
                            task=task,
                            parameter_schemas=initial_schemas,
                        )
                        autonomous_plan.pop("expert_consultation", None)
                        autonomous_plan = _host_guardrail_plan(
                            autonomous_plan,
                            None,
                        )
                    except GatewayResponseError as exc:
                        if metadata.get("remote_fallback_policy") == "fail_run":
                            raise
                        planning_gateway_error = exc
                        autonomous_plan = {}
                    self._session_plans[session_id] = dict(autonomous_plan)
                    self._resolved_plan_sessions.add(session_id)
                predictor_adoption = resolve_predictor_adoption(task, autonomous_plan)
            metadata["prediction_model_id"] = predictor_adoption.adopted_id
            metadata["prediction_model_digest"] = predictor_adoption.adopted_digest

        boundary, schemas, sweep = _task_parameter_space(metadata, domain)
        semantics = parameter_semantics(boundary, schemas)
        autonomous_search_name = _autonomous_search_name(autonomous_plan)
        single_parameter_sweep = (
            autonomous_search_name == "bounded_single_parameter_sweep"
        )
        schedule_index = run.generation * batch_size + slot_index
        base = dict(sweep[schedule_index % len(sweep)])
        model_response: dict[str, Any] = {}
        search_design_audit: dict[str, Any] | None = None
        host_fallback: dict[str, Any] | None = None
        remote_strategy_called = False
        remote_strategy_succeeded = False
        parent = _parent_context(parent_context, schemas)
        if parent is not None:
            if parent_candidate_id is None:
                raise ValueError("parent_context requires parent_candidate_id")
            if parent["candidate_id"] != parent_candidate_id:
                raise ValueError(
                    "parent_context candidate_id does not match parent_candidate_id"
                )
        parent_parameter_space_compatible = bool(
            parent is not None and parent["parameter_space_compatible"]
        )
        parent_parameters = (
            dict(parent["proposal_parameters"])
            if parent is not None and parent_parameter_space_compatible
            else None
        )
        judge_override: dict[str, int | float] = {}
        if parent is not None:
            judge = parent.get("judge")
            if (
                parent_parameter_space_compatible
                and isinstance(judge, Mapping)
                and judge.get("parameter_override")
            ):
                judge_override = _bounded_parameters(
                    judge["parameter_override"],
                    schemas,
                    partial=True,
                    source="parent judge parameter_override",
                )
        guided_parent_parameters = (
            dict(parent_parameters) if parent_parameters is not None else None
        )
        if guided_parent_parameters is not None and judge_override:
            guided_parent_parameters.update(judge_override)
        if (
            single_parameter_sweep
            and batch is not None
            and guided_parent_parameters is None
        ):
            # A single-parameter sibling comparison needs one common reference
            # point.  Slot-specific seeds would confound every observed score
            # difference even when each remote response describes itself as a
            # one-dimensional sweep.
            base = dict(sweep[run.generation % len(sweep)])
        search_metadata = metadata
        if parent is not None and not parent_parameter_space_compatible:
            search_metadata = dict(metadata)
            search_metadata.pop("incumbent_parameters", None)
            search_metadata.pop("current_parameters", None)
        search_design_role = "exploratory_candidate"
        diagnostic_anchor = False
        conservative_recovery_seed = False
        host_reserved_seed = False
        if (
            boundary == "greenhouse"
            and run.generation == 0
            and parent_parameters is None
            and batch is not None
            and batch_size >= 2
            and not single_parameter_sweep
        ):
            base = dict(
                self._GREENHOUSE_INITIAL_DESIGN[
                    slot_index % len(self._GREENHOUSE_INITIAL_DESIGN)
                ]
            )
            diagnostic_anchor = slot_index == 0
            # A two-candidate round must still exercise the selected strategy
            # model.  Keep both host seeds only when a third exploratory slot
            # exists; K=1 is entirely remote, K=2 is one host + one remote.
            conservative_recovery_seed = batch_size >= 3 and slot_index == 1
            host_reserved_seed = diagnostic_anchor or conservative_recovery_seed
            if diagnostic_anchor:
                search_design_role = "persistence_diagnostic_anchor"
            elif conservative_recovery_seed:
                search_design_role = "conservative_recovery_seed"
        human_guidance: str | None = None
        human_constraints: list[str] = []
        human_override: dict[str, int | float] = {}
        host_applies_interventions = False

        if interventions is not None:
            if not isinstance(interventions, Mapping):
                raise TypeError("interventions must be a mapping")
            unknown = set(interventions) - {
                "accepted",
                "guidance",
                "constraints",
                "parameter_override",
                "host_applies_interventions",
            }
            if unknown:
                raise ValueError(
                    f"unsupported intervention fields: {', '.join(sorted(unknown))}"
                )
            if "guidance" in interventions:
                human_guidance = _optional_text(
                    interventions["guidance"], "interventions.guidance"
                )
            if "constraints" in interventions:
                raw_constraints = interventions["constraints"]
                if not isinstance(raw_constraints, (list, tuple)):
                    raise TypeError("interventions.constraints must be an array")
                if len(raw_constraints) > 32:
                    raise ValueError("interventions.constraints contains too many items")
                for item in raw_constraints:
                    constraint = _optional_text(
                        item, "interventions.constraints item"
                    )
                    if constraint is None:
                        raise ValueError(
                            "interventions.constraints item must be a non-empty string"
                        )
                    human_constraints.append(constraint)
            if "parameter_override" in interventions:
                human_override = _bounded_parameters(
                    interventions["parameter_override"],
                    schemas,
                    partial=True,
                    source="parameter_override",
                )
            if "host_applies_interventions" in interventions:
                if not isinstance(interventions["host_applies_interventions"], bool):
                    raise TypeError(
                        "interventions.host_applies_interventions must be a bool"
                    )
                host_applies_interventions = interventions[
                    "host_applies_interventions"
                ]

        if host_reserved_seed:
            # These host-defined points make the baseline and a conservative
            # recovery neighborhood observable even if a remote policy keeps
            # proposing the same region.
            pass
        elif (
            strategy_id == "parameter_sweep@1"
            and guided_parent_parameters is not None
        ):
            base = self._sweep_around_parent(
                guided_parent_parameters,
                schemas,
                schedule_index,
            )
        elif strategy_id == "adaptive_local@1":
            base = self._adaptive_parameters(
                search_metadata,
                base,
                schemas,
                schedule_index,
                parent_parameters=guided_parent_parameters,
                evaluation=(parent.get("evaluation") if parent is not None else None),
                generation_analysis=previous_analysis,
                knowledge_context=knowledge_context,
                slot_index=slot_index,
                batch_size=batch_size,
            )
        elif strategy_id in {"dsh_authenticated@1", "autonomous_model@1"}:
            assert model_id is not None
            if guided_parent_parameters is not None:
                base = dict(guided_parent_parameters)
            human_input: dict[str, Any] = {}
            if human_guidance is not None:
                human_input["guidance"] = human_guidance
            if human_constraints:
                human_input["constraints"] = list(human_constraints)
            if human_override:
                human_input["parameter_override"] = dict(human_override)
            remote_strategy_called = True
            remote_operation = "proposal"
            remote_error: GatewayResponseError | None = planning_gateway_error
            if remote_error is None:
                try:
                    model_response = self.gateway.propose(
                        model_id,
                        {
                            "run_id": run.run_id,
                            "generation": run.generation,
                            "proposal_index": slot_index + 1,
                            "slot_index": slot_index,
                            "batch_size": batch_size,
                            "objective": task.objective,
                            "domain": domain,
                            "domain_pack": task.domain_pack,
                            "prediction_model_id": metadata.get(
                                "prediction_model_id"
                            ),
                            "prediction_model_adoption": (
                                predictor_adoption.to_dict()
                                if predictor_adoption is not None
                                else None
                            ),
                            "model_selection_policy": metadata.get(
                                "model_selection_policy",
                                "model_research_and_runtime_compile@1"
                                if strategy_id == "autonomous_model@1"
                                else None,
                            ),
                            "autonomous_plan": autonomous_plan or None,
                            _HISTORICAL_PARAMETER_GUARDRAILS_FIELD: (
                                safe_aggregate_feedback(
                                    autonomous_plan.get(
                                        _HISTORICAL_PARAMETER_GUARDRAILS_FIELD
                                    ),
                                    name=(
                                        "autonomous proposal historical parameter "
                                        "guardrails"
                                    ),
                                )
                                if isinstance(
                                    autonomous_plan.get(
                                        _HISTORICAL_PARAMETER_GUARDRAILS_FIELD
                                    ),
                                    Mapping,
                                )
                                else None
                            ),
                            "visible_datasets": list(task.visible_datasets),
                            "parent_candidate_id": parent_candidate_id,
                            "seed": task.seed,
                            "parent": parent,
                            "previous_generation_analysis": previous_analysis,
                            "knowledge_snapshot": knowledge_context,
                            "research_iteration": research_iteration_context,
                            "parameter_semantics": semantics,
                            "search_design": {
                                "role": search_design_role,
                                "host_seed_parameters": dict(base),
                                "sibling_slots_must_be_distinct": batch_size > 1,
                                "requested_search_policy": autonomous_search_name,
                                "max_parameter_changes_from_shared_reference": (
                                    1 if single_parameter_sweep else None
                                ),
                            },
                            "human_input": human_input,
                            "host_boundary": {
                                "parameter_ranges_are_fixed": True,
                                "scientific_gate_is_not_mutable": True,
                                "model_generated_code_execution": False,
                            },
                        },
                        schemas,
                    )
                except GatewayResponseError as exc:
                    remote_error = exc
                else:
                    remote_strategy_succeeded = True
                    if single_parameter_sweep:
                        base, search_design_audit = _project_single_parameter_change(
                            base,
                            model_response["parameters"],
                            schemas,
                            slot_index=slot_index,
                        )
                    else:
                        base.update(model_response["parameters"])
            if remote_error is not None:
                exc = remote_error
                remote_operation = (
                    "research_plan"
                    if planning_gateway_error is not None
                    else "proposal"
                )
                if (
                    metadata.get("remote_fallback_policy") == "fail_run"
                    and exc.retryable
                ):
                    # Let the autonomous worker replay the idempotent
                    # generation after request-local retries are exhausted.
                    # A transient queue timeout is not a valid reason to switch
                    # algorithms or declare the model API invalid.
                    raise exc
                fallback_source = "host_seed_parameters"
                if guided_parent_parameters is not None:
                    base = self._sweep_around_parent(
                        guided_parent_parameters,
                        schemas,
                        schedule_index,
                    )
                    fallback_source = "bounded_parent_sweep"
                host_fallback = {
                    "applied": True,
                    "reason": (
                        "remote_strategy_gateway_error"
                        if remote_operation == "proposal"
                        else "remote_research_plan_gateway_error"
                    ),
                    "operation": remote_operation,
                    "error_type": type(exc).__name__,
                    "public_error": public_exception_summary(exc),
                    "parameter_source": fallback_source,
                    "strategy_id": strategy_id,
                    "strategy_model_id": model_id,
                    "generation": run.generation,
                    "slot_index": slot_index,
                    "batch_size": batch_size,
                }

        if human_override:
            base.update(human_override)

        parameters = _bounded_parameters(base, schemas, partial=False, source="proposal")
        effective_task = _task_with_predictor_adoption(task, predictor_adoption)
        if interventions is not None and not host_applies_interventions:
            local_controls: list[dict[str, Any]] = []
            if human_guidance is not None:
                local_controls.append(
                    {
                        "intervention_id": "local-guidance",
                        "kind": "guidance",
                        "message": human_guidance,
                    }
                )
            for offset, constraint in enumerate(human_constraints, start=1):
                local_controls.append(
                    {
                        "intervention_id": f"local-constraint-{offset}",
                        "kind": "constraint",
                        "message": constraint,
                    }
                )
            if human_override:
                local_controls.append(
                    {
                        "intervention_id": "local-parameter-override",
                        "kind": "parameter_override",
                        "message": "direct adapter parameter override",
                        "parameter_overrides": human_override,
                    }
                )
            parameters, _ = apply_bounded_interventions(
                effective_task,
                parameters,
                local_controls,
            )
        protected_parameter_values = (
            _protected_historical_parameter_values(autonomous_plan, schemas)
            if strategy_id == "autonomous_model@1"
            else {}
        )
        guardrail_overridden_parameters = sorted(
            name
            for name, value in protected_parameter_values.items()
            if parameters.get(name) != value
        )
        if protected_parameter_values:
            parameters = _bounded_parameters(
                {**parameters, **protected_parameter_values},
                schemas,
                partial=False,
                source="host-enforced historical parameter guardrails",
            )
        self._session_counts[key] = index + 1
        domain_title = (
            "温室环境外生变量预测"
            if boundary
            in {
                "greenhouse_ridge",
                "greenhouse_targetwise_ridge",
                "greenhouse_horizon_targetwise_ridge",
            }
            else "温室环境"
            if domain == "greenhouse"
            else "作物土壤水分"
        )
        strategy_title = {
            "parameter_sweep@1": "参数扫描",
            "adaptive_local@1": "局部自适应",
            "dsh_authenticated@1": "认证模型",
            "autonomous_model@1": "模型自主调研",
        }[strategy_id]
        rationale = (
            f"使用{strategy_title}策略生成{domain_title}本轮槽位 "
            f"{slot_index + 1}/{batch_size} 的有界参数。"
        )
        if diagnostic_anchor:
            rationale += (
                " 该槽位是宿主固定的持续性诊断锚点：blend=1 仅使用最新观测，"
                "bias_scale=0 禁用训练偏差修正；锚点只用于校准搜索，不代表正式通过。"
            )
        elif conservative_recovery_seed:
            rationale += (
                " 该槽位是宿主固定的保守恢复种子：blend=0.93、window=25、"
                "bias_scale=0，用于确保首轮覆盖持续性基线附近的可行邻域；"
                "仍须通过全部科学门禁和独立评审。"
            )
        if parent is not None and parent_parameter_space_compatible:
            evaluation = parent["evaluation"]
            rationale += (
                f" 以已完成父候选 {parent['candidate_id']} 的参数为起点"
                f"（聚合得分 {float(evaluation['score']):.6g}）。"
            )
        elif parent is not None:
            rationale += (
                f" 保留父候选 {parent['candidate_id']} 的聚合评测反馈；"
                "因登记预测器参数空间已切换，本轮从新预测器的宿主种子开始，"
                "不复用旧参数或旧评审覆盖。"
            )
        if model_response.get("rationale"):
            rationale += f" 模型说明：{model_response['rationale']}"
        if (
            search_design_audit is not None
            and search_design_audit["host_projection_applied"]
        ):
            rationale += (
                " 宿主已将模型的多参数输出投影为相对同轮共同参考点的单参数变化："
                f"{search_design_audit['adopted_parameter']}，避免同轮比较混杂。"
            )
        if host_fallback is not None:
            rationale += (
                " 远程策略提案暂不可用；本槽位已改用宿主有界回退参数，"
                "仍需通过相同的训练、科学门禁和独立评审。"
            )
        if judge_override:
            rationale += (
                " 上一轮评审参数建议已经宿主边界校验，并作为本轮策略起点："
                + "、".join(sorted(judge_override))
                + "。"
            )
        if human_guidance:
            rationale += f" 人工调整指引已进入候选生成上下文：{human_guidance}"
        if model_response.get("guidance"):
            rationale += f" 模型后续建议：{model_response['guidance']}"
        if host_reserved_seed:
            proposal_source = "host_reserved_seed"
        elif host_fallback is not None:
            proposal_source = "host_fallback"
        elif remote_strategy_succeeded:
            proposal_source = "remote_model"
        else:
            proposal_source = "host_strategy"
        proposal_metadata: dict[str, Any] = {
            "proposal_source": proposal_source,
            "remote_strategy_called": remote_strategy_called,
            "remote_strategy_succeeded": remote_strategy_succeeded,
        }
        if search_design_audit is not None:
            proposal_metadata["search_design_audit"] = search_design_audit
        tool_experience = _previous_tool_experience(previous_analysis)
        if tool_experience:
            proposal_metadata["tool_experience"] = tool_experience
        if host_fallback is not None:
            proposal_metadata["host_fallback"] = host_fallback
        if protected_parameter_values:
            proposal_metadata["historical_parameter_guardrail_enforcement"] = {
                "policy": _HISTORICAL_PARAMETER_GUARDRAIL_POLICY,
                "protected_parameter_names": sorted(protected_parameter_values),
                "overridden_parameter_names": guardrail_overridden_parameters,
            }
        if strategy_id == "autonomous_model@1":
            if predictor_adoption is None:
                raise RuntimeError("autonomous proposal is missing predictor adoption")
            proposal_metadata.update(
                {
                    "autonomous_mode": True,
                    "strategy_model_id": _optional_text(
                        metadata.get("strategy_model_id")
                        or metadata.get("policy_model_id"),
                        "strategy_model_id",
                    ),
                    "plan": autonomous_plan,
                    "prediction_model_adoption": predictor_adoption.to_dict(),
                }
            )
            if isinstance(research_iteration_context, Mapping):
                proposal_metadata["research_iteration_digest"] = str(
                    research_iteration_context["iteration_digest"]
                )
                proposal_metadata["research_iteration_status"] = str(
                    research_iteration_context["status"]
                )
            selected_predictor = (
                autonomous_plan.get("prediction_model", {}).get("id")
                if isinstance(autonomous_plan.get("prediction_model"), Mapping)
                else autonomous_plan.get("algorithm_blueprint", {}).get(
                    "pipeline_id"
                )
                if isinstance(
                    autonomous_plan.get("algorithm_blueprint"), Mapping
                )
                else None
            )
            selected_strategy = (
                autonomous_plan.get("strategy", {}).get("name")
                if isinstance(autonomous_plan.get("strategy"), Mapping)
                else None
            )
            if selected_predictor:
                rationale += f" 模型调研建议预测模型：{str(selected_predictor)[:200]}。"
            if predictor_adoption.status == "adopted":
                rationale += (
                    " 宿主已采用兼容的登记预测器 "
                    f"{predictor_adoption.adopted_id}，并切换其参数边界。"
                )
            elif predictor_adoption.status == "research_only":
                rationale += (
                    " 模型建议未通过宿主登记与兼容性边界，继续使用默认预测器 "
                    f"{predictor_adoption.default_id}。"
                )
            if selected_strategy:
                rationale += f" 模型调研策略：{str(selected_strategy)[:200]}。"
        if human_constraints:
            rationale += (
                " 人工参数约束已进入候选生成上下文，未修改宿主科学门禁："
                + " | ".join(human_constraints)
            )
        if human_override:
            rationale += " 人工参数覆盖已按宿主边界应用。"
        adopted_knowledge = (
            knowledge_context.get("adopted_knowledge", [])
            if isinstance(knowledge_context, Mapping)
            else []
        )
        adopted_titles = [
            str(item.get("title"))
            for item in adopted_knowledge
            if isinstance(item, Mapping) and item.get("title")
        ]
        if adopted_titles:
            rationale += (
                " 本轮采用已映射到宿主能力的知识依据："
                + "、".join(adopted_titles[:4])
                + "；知识只影响有界候选方向，不修改数据切分和科学门禁。"
            )
        return Proposal(
            proposal_id=f"proposal:{run.run_id}:{run.generation}:{index + 1}",
            run_id=run.run_id,
            generation=run.generation,
            title=f"{domain_title}{strategy_title}方案 {index + 1}",
            changes=parameters,
            parent_candidate_id=parent_candidate_id,
            rationale=rationale,
            metadata=proposal_metadata,
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
        assigned_direction = None
        if _model_search_cycle_enabled(task):
            allowed_mutation_targets = _registered_mutation_targets(task, parent)
            directions = normalize_candidate_directions(
                research_plan.get("candidate_directions"),
                exact_items=int(batch["batch_size"]),
                allowed_mutation_targets=allowed_mutation_targets,
            )
            assigned_direction = directions[slot_index]
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
        mutation_budget_digest = digest(
            {
                "policy": TRUST_REGION_MUTATION_OPERATOR_ID,
                "generation": run.generation,
                "slot_index": slot_index,
                "maximum_operations": 1,
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
            mutation_operator_id=TRUST_REGION_MUTATION_OPERATOR_ID,
        )
        research_directives = research_plan.get("dsh_evolution_reflection")
        if not isinstance(research_directives, Mapping):
            research_directives = {}
        avoid_behaviors = research_directives.get("avoid_behaviors")
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
            "mutation_operator_id": TRUST_REGION_MUTATION_OPERATOR_ID,
            "maximum_operations": 1,
            "maximum_normalized_parameter_step": (
                TRUST_REGION_MAX_NORMALIZED_STEP
            ),
            "parameter_step_scale": (
                "log_for_positive_multi_order_ranges_otherwise_linear"
            ),
            "active_predictor_id": parent.scientific_program["predictor_ref"]["id"],
            "bounded_parameters": deepcopy(dict(active_parameter_schemas)),
            **deepcopy(mutation_catalog),
            "policy": "single_axis_reject_out_of_bounds_without_clamping",
        }
        base_reflection = {
            "schema_version": "ecologyrsi-dsh.evolution-reflection/2",
            "previous_generation_analysis": previous_analysis,
            "research_summary": research_plan.get("dsh_research_summary"),
            "research_evidence": research_plan.get(
                "dsh_research_evidence",
                [],
            ),
            "previous_next_action": (
                research_iteration.get("previous_next_action")
                if isinstance(research_iteration, Mapping)
                else None
            ),
            "historical_provenance": (
                research_iteration.get("historical_provenance")
                if isinstance(research_iteration, Mapping)
                else None
            ),
            "research_directives": dict(research_directives),
            "previous_generation_reflection": research_plan.get(
                "previous_generation_reflection"
            ),
            "assigned_candidate_direction": (
                assigned_direction.to_dict()
                if assigned_direction is not None
                else None
            ),
            "avoid_behaviors": list(avoid_behaviors),
            "sibling_candidate_behaviors": deepcopy(sibling_candidate_behaviors),
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
                "mutation_context": context.to_dict(),
                "mutation_contract": deepcopy(mutation_contract),
                "assigned_candidate_direction": (
                    assigned_direction.to_dict()
                    if assigned_direction is not None
                    else None
                ),
                "generation_context_digest": batch["context_digest"],
                "research_iteration": research_iteration,
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
                )
            except (TypeError, ValueError) as exc:
                message = str(exc)
                rejection_code = (
                    "mutation_has_no_effect"
                    if "does not change" in message
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
                            "predictor_evaluator_incompatible"
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
                expected_by_axis = {
                    "scientific_parameter": {
                        "op": "set_bounded_parameter",
                        "name": assigned_direction.mutation_target,
                    },
                    "registered_predictor": {
                        "op": "select_registered_pipeline",
                        "predictor_id": assigned_direction.mutation_target,
                    },
                    "instruction_profile": {
                        "op": "select_instruction_template",
                        "role": "sample-planner",
                        "instruction_template_id": (
                            assigned_direction.mutation_target
                        ),
                    },
                }
                expected_operation = expected_by_axis[
                    assigned_direction.mutation_axis
                ]
                direction_mismatch = any(
                    accepted_operation.get(name) != value
                    for name, value in expected_operation.items()
                )
                if (
                    not direction_mismatch
                    and assigned_direction.mutation_axis
                    == "scientific_parameter"
                ):
                    parameter_name = assigned_direction.mutation_target
                    previous_value = parent.to_dict()["scientific_program"][
                        "parameter_overrides"
                    ].get(parameter_name)
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
            repeated = next(
                (
                    item
                    for item in avoid_behaviors
                    if isinstance(item, Mapping)
                    and item.get("behavior_digest") == proposed_child.behavior_digest
                ),
                None,
            )
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
            if repeated is None and sibling_repeated is None:
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
                    }
                )
            else:
                host_rejections.append(
                    {
                        "reason": "exact_failed_behavior_replay",
                        "prediction_model_id": scientific["predictor_ref"]["id"],
                        "parameters_digest": digest(scientific["parameter_overrides"]),
                        "source_reason": repeated.get("reason"),
                    }
                )
        if child is None:
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
                "DSH proposal repeats a previously failed behavior after bounded retry"
            )
        key = (session_id, run.generation)
        index = self._session_counts.setdefault(key, 0)
        self._session_counts[key] = index + 1
        candidate_agent_profile = resolve_candidate_agent_profile(
            child,
            current_program_registry(),
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
            rationale="DSH structured GenomeMutation applied by the Host registry.",
            metadata={
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "proposal_source": "dsh_native_agent",
                "remote_strategy_called": True,
                "remote_strategy_succeeded": True,
                "evolution_genome_canonical_json": canonical_json(child.to_dict()),
                "genome_digest": child.genome_digest,
                "behavior_digest": child.behavior_digest,
                "mutation_context": context.to_dict(),
                "mutation_digest": child.lineage["mutation_digest"],
                "mutation_operations": deepcopy(mutation["operations"]),
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

    @staticmethod
    def _sweep_around_parent(
        parent_parameters: Mapping[str, Any],
        schemas: Mapping[str, Mapping[str, Any]],
        schedule_index: int,
    ) -> dict[str, Any]:
        """Vary one bounded parameter while preserving the rest of the parent."""

        base = _bounded_parameters(
            parent_parameters,
            schemas,
            partial=False,
            source="parent_context.proposal_parameters",
        )
        names = tuple(schemas)
        phase = schedule_index % (2 * len(names))
        name = names[phase // 2]
        direction = 1 if phase % 2 == 0 else -1
        schema = schemas[name]
        span = float(schema["maximum"]) - float(schema["minimum"])
        step = max(1, round(span * 0.1)) if schema["type"] == "integer" else span * 0.1
        original = float(base[name])
        candidate = original + direction * step
        candidate = min(
            float(schema["maximum"]),
            max(float(schema["minimum"]), candidate),
        )
        if math.isclose(candidate, original, rel_tol=0.0, abs_tol=1e-12):
            # At a boundary, simply reflecting the blocked move by one step
            # aliases the sibling already exploring that direction.  Move two
            # steps inward so opposite batch slots remain distinct whenever
            # the bounded space has room.
            candidate = original - direction * step * 2
            candidate = min(
                float(schema["maximum"]),
                max(float(schema["minimum"]), candidate),
            )
        base[name] = (
            round(candidate)
            if schema["type"] == "integer"
            else round(candidate, 4)
        )
        return base

    @staticmethod
    def _adaptive_parameters(
        metadata: Mapping[str, Any],
        fallback: Mapping[str, Any],
        schemas: Mapping[str, Mapping[str, Any]],
        schedule_index: int,
        *,
        parent_parameters: Mapping[str, Any] | None = None,
        evaluation: Mapping[str, Any] | None = None,
        generation_analysis: Mapping[str, Any] | None = None,
        knowledge_context: Mapping[str, Any] | None = None,
        slot_index: int = 0,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        current = parent_parameters
        if current is None:
            current = metadata.get(
                "incumbent_parameters", metadata.get("current_parameters")
            )
        if current is None:
            base = dict(fallback)
        else:
            base = _bounded_parameters(current, schemas, partial=False, source="incumbent_parameters")
        centered_slot = slot_index - (batch_size - 1) / 2
        # A positive parent signal keeps the established local-search
        # convention: the central/single slot moves downward first, while a
        # negative signal reverses that direction. Sibling slots still probe
        # both sides of the same frozen parent.
        direction = -1 if centered_slot >= 0 else 1
        magnitude = max(0.35, abs(centered_slot) + 0.5)
        if evaluation is not None:
            score = _finite_number(evaluation.get("score"), default=0.0)
            metrics = evaluation.get("metrics")
            signal = score
            if isinstance(metrics, Mapping):
                signal = _finite_number(metrics.get("improvement"), default=score)
            if signal < 0:
                direction *= -1
            if bool(evaluation.get("passed", False)):
                magnitude *= max(0.35, 0.75 - min(max(score, 0.0), 1.0) * 0.25)
            else:
                magnitude *= 1.0 + min(abs(score), 2.0) * 0.25
        focus_name = analysis_focus_parameter(
            generation_analysis, schemas
        ) or knowledge_focus_parameter(knowledge_context, schemas)
        ordered_names = [focus_name] if focus_name is not None else list(schemas)
        if focus_name is None and len(ordered_names) > 1:
            offset = schedule_index % len(ordered_names)
            ordered_names = ordered_names[offset:] + ordered_names[:offset]
        result = dict(base)
        for name in ordered_names:
            schema = schemas[name]
            span = float(schema["maximum"]) - float(schema["minimum"])
            step = max(1, round(span * 0.08 * magnitude)) if schema["type"] == "integer" else span * 0.08 * magnitude
            candidate = float(base[name]) + direction * step
            candidate = min(float(schema["maximum"]), max(float(schema["minimum"]), candidate))
            result[name] = round(candidate) if schema["type"] == "integer" else round(candidate, 4)
            direction *= -1
        return result

    def close_session(self, session_id: str) -> None:
        for key in tuple(self._session_counts):
            if key[0] == session_id:
                self._session_counts.pop(key, None)
        self._session_plans.pop(session_id, None)
        self._resolved_plan_sessions.discard(session_id)


def _task_with_predictor_adoption(
    task: TaskManifest,
    adoption: PredictorAdoption | None,
) -> TaskManifest:
    if adoption is None:
        return task
    data = task.to_dict()
    data["metadata"] = {
        **dict(task.metadata),
        "prediction_model_id": adoption.adopted_id,
        "prediction_model_digest": adoption.adopted_digest,
    }
    return TaskManifest.from_dict(data)


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


def _previous_tool_experience(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    ranking = value.get("ranking")
    if not isinstance(ranking, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    for candidate in ranking[:8]:
        if not isinstance(candidate, Mapping):
            continue
        rows = candidate.get("tool_performance")
        if not isinstance(rows, (list, tuple)):
            continue
        candidate_id = candidate.get("candidate_id")
        for raw in rows[:32]:
            if not isinstance(raw, Mapping):
                continue
            row = {
                name: raw[name]
                for name in (
                    "tool_id",
                    "version",
                    "target",
                    "horizon_hours",
                    "selected",
                    "failed",
                    "critic_accept",
                    "critic_repair",
                    "final_accept",
                    "n",
                    "rmse_improvement",
                    "skill_score",
                )
                if name in raw
            }
            if isinstance(candidate_id, str) and candidate_id.strip():
                row["candidate_id"] = candidate_id.strip()[:200]
            if row.get("tool_id"):
                result.append(row)
            if len(result) >= 32:
                return result
    return result


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _parent_context(
    value: Mapping[str, Any] | None,
    schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Validate and redact the aggregate parent evidence sent to a strategy."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("parent_context must be a mapping")
    allowed = {
        "candidate_id",
        "status",
        "proposal_id",
        "proposal_parameters",
        "artifact",
        "evaluation",
        "judge",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            "parent_context contains unsupported fields: "
            + ", ".join(sorted(unknown))
        )
    candidate_id = _optional_text(value.get("candidate_id"), "parent_context.candidate_id")
    proposal_id = _optional_text(value.get("proposal_id"), "parent_context.proposal_id")
    status = _optional_text(value.get("status"), "parent_context.status")
    if candidate_id is None or proposal_id is None or status is None:
        raise ValueError("parent_context is missing its candidate identity")
    if status not in {"promoted", "rejected"}:
        raise ValueError("parent_context must describe a completed candidate")
    raw_parameters = value.get("proposal_parameters")
    if not isinstance(raw_parameters, Mapping):
        raise TypeError("parent_context.proposal_parameters must be a mapping")
    registered_spaces = (
        ("toy", StrategyRouterDSHAdapter._TOY_SCHEMAS),
        ("greenhouse", StrategyRouterDSHAdapter._GREENHOUSE_SCHEMAS),
        ("greenhouse_ridge", StrategyRouterDSHAdapter._GREENHOUSE_RIDGE_SCHEMAS),
        (
            "greenhouse_targetwise_ridge",
            StrategyRouterDSHAdapter._GREENHOUSE_TARGETWISE_RIDGE_SCHEMAS,
        ),
        (
            "greenhouse_horizon_targetwise_ridge",
            StrategyRouterDSHAdapter._GREENHOUSE_HORIZON_TARGETWISE_RIDGE_SCHEMAS,
        ),
    )
    matched_space = next(
        (
            (name, registered_schemas)
            for name, registered_schemas in registered_spaces
            if set(raw_parameters) == set(registered_schemas)
        ),
        None,
    )
    if matched_space is None:
        raise ValueError(
            "parent_context.proposal_parameters do not match a registered predictor contract"
        )
    parent_parameter_space, parent_schemas = matched_space
    parameters = _bounded_parameters(
        raw_parameters,
        parent_schemas,
        partial=False,
        source="parent_context.proposal_parameters",
    )
    parameter_space_compatible = set(parent_schemas) == set(schemas)
    evaluation = value.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise TypeError("parent_context.evaluation must be a mapping")
    score = _finite_number(evaluation.get("score"), default=None)
    if score is None or not isinstance(evaluation.get("passed"), bool):
        raise ValueError("parent_context.evaluation requires a finite score and bool passed")

    result = {
        "candidate_id": candidate_id,
        "status": status,
        "proposal_id": proposal_id,
        "proposal_parameters": parameters,
        "parameter_space": parent_parameter_space,
        "parameter_space_compatible": parameter_space_compatible,
        "artifact": _redacted_context_value(
            value.get("artifact"), "parent_context.artifact"
        ),
        "evaluation": _redacted_context_value(
            dict(evaluation), "parent_context.evaluation"
        ),
        "judge": _redacted_context_value(
            value.get("judge"), "parent_context.judge"
        ),
    }
    judge = result.get("judge")
    if isinstance(judge, Mapping) and judge.get("parameter_override"):
        judge_copy = dict(judge)
        judge_copy["parameter_override"] = _bounded_parameters(
            judge["parameter_override"],
            parent_schemas,
            partial=True,
            source="parent_context.judge.parameter_override",
        )
        result["judge"] = judge_copy
    return result


def _redacted_context_value(value: Any, name: str, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > 4000:
            raise ValueError(f"{name} contains an overlong string")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return value
    if depth >= 6:
        raise ValueError(f"{name} is nested too deeply")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} contains an invalid field name")
            if is_sensitive_context_field(key):
                continue
            result[key] = _redacted_context_value(
                item, f"{name}.{key}", depth=depth + 1
            )
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 128:
            raise ValueError(f"{name} contains too many items")
        return [
            _redacted_context_value(item, f"{name} item", depth=depth + 1)
            for item in value
        ]
    raise TypeError(f"{name} must contain only JSON-compatible summary values")


def _finite_number(value: Any, *, default: float | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _bounded_parameters(
    value: Any,
    schemas: Mapping[str, Mapping[str, Any]],
    *,
    partial: bool,
    source: str,
) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} must be a mapping")
    unknown = set(value) - set(schemas)
    if unknown:
        raise ValueError(f"{source} contains unsupported parameters: {', '.join(sorted(unknown))}")
    if not partial and set(value) != set(schemas):
        missing = set(schemas) - set(value)
        raise ValueError(f"{source} is missing parameters: {', '.join(sorted(missing))}")
    result: dict[str, int | float] = {}
    for name, raw in value.items():
        schema = schemas[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ValueError(f"{source}.{name} must be a finite number")
        if schema["type"] == "integer" and not isinstance(raw, int):
            raise ValueError(f"{source}.{name} must be an integer")
        if float(raw) < float(schema["minimum"]) or float(raw) > float(schema["maximum"]):
            raise ValueError(f"{source}.{name} is outside the allowed range")
        result[name] = int(raw) if schema["type"] == "integer" else float(raw)
    return result


_PARAMETER_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "toy": {
        "alpha": ("alpha", "平滑权重", "平滑系数"),
        "window": ("window", "时间窗口", "历史窗口"),
        "water_threshold": (
            "water_threshold",
            "water threshold",
            "土壤水分阈值",
            "水分阈值",
        ),
    },
    "greenhouse": {
        "blend": ("blend", "混合权重", "融合权重"),
        "window": ("window", "时间窗口", "历史窗口"),
        "bias_scale": (
            "bias_scale",
            "bias scale",
            "偏差缩放系数",
            "偏差缩放",
        ),
    },
    "greenhouse_ridge": {
        "history_steps": (
            "history_steps",
            "history steps",
            "历史步数",
            "滞后步数",
        ),
        "ridge_alpha": (
            "ridge_alpha",
            "ridge alpha",
            "岭回归强度",
            "正则化强度",
        ),
        "residual_scale": (
            "residual_scale",
            "residual scale",
            "残差缩放系数",
            "残差缩放",
        ),
    },
    "greenhouse_targetwise_ridge": {
        "history_steps": (
            "history_steps",
            "history steps",
            "历史步数",
            "滞后步数",
        ),
        "ridge_alpha": (
            "ridge_alpha",
            "ridge alpha",
            "岭回归强度",
            "正则化强度",
        ),
        "air_temperature_residual_scale": (
            "air_temperature_residual_scale",
            "temperature residual scale",
            "温度残差缩放",
        ),
        "relative_humidity_residual_scale": (
            "relative_humidity_residual_scale",
            "humidity residual scale",
            "湿度残差缩放",
        ),
        "co2_concentration_residual_scale": (
            "co2_concentration_residual_scale",
            "co2 residual scale",
            "二氧化碳残差缩放",
        ),
    },
}
_GUIDANCE_STEPS: dict[str, int | float] = {
    "alpha": 0.1,
    "blend": 0.1,
    "bias_scale": 0.1,
    "water_threshold": 0.05,
    "window": 1,
    "history_steps": 1,
    "ridge_alpha": 0.05,
    "residual_scale": 0.1,
    "air_temperature_residual_scale": 0.1,
    "relative_humidity_residual_scale": 0.1,
    "co2_concentration_residual_scale": 0.1,
}
_GUIDANCE_DIRECTIONS: dict[str, tuple[str, ...]] = {
    "decrease": ("缩短", "降低", "减小", "下调", "减少", "decrease", "shorten", "lower"),
    "increase": ("延长", "提高", "增大", "上调", "增加", "increase", "extend", "raise"),
}
_NEGATED_GUIDANCE = re.compile(
    r"(?:不要|不得|禁止|不应|无需)(?:[^，。；,;]{0,12})$", re.IGNORECASE
)
_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_ENGLISH_NUMBER_WORD_PATTERN = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|half)(?:[-\s]+(?:zero|one|two|three|four|five|"
    r"six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    r"sixty|seventy|eighty|ninety|hundred|thousand|half))*\b"
)
_CHINESE_CARDINAL_PATTERN = (
    r"[零〇○一二两三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億]+"
)
_CHINESE_NUMBER_PATTERN = (
    rf"(?:[正负]?(?:{_CHINESE_CARDINAL_PATTERN}"
    rf"(?:点[零〇○一二两三四五六七八九壹贰叁肆伍陆柒捌玖]+)?|半)"
    rf"|(?:百分之|千分之){_CHINESE_CARDINAL_PATTERN}"
    rf"|{_CHINESE_CARDINAL_PATTERN}分之{_CHINESE_CARDINAL_PATTERN})"
)
_EXACT_PARAMETER_VALUE_PATTERN = (
    rf"(?:{_NUMBER_PATTERN}(?:[eE][-+]?\d+)?|"
    rf"{_ENGLISH_NUMBER_WORD_PATTERN}|{_CHINESE_NUMBER_PATTERN})"
)


def _task_parameter_boundary(
    task: TaskManifest,
    current_plan: Mapping[str, Any] | None = None,
) -> tuple[str, Mapping[str, Mapping[str, Any]]]:
    metadata = dict(task.metadata)
    if current_plan is not None:
        metadata["prediction_model_id"] = resolve_predictor_adoption(
            task,
            current_plan,
        ).adopted_id
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
            if not predictor_id or predictor_id == current_predictor:
                continue
            try:
                registry.program("predictors", predictor_id)
            except ValueError:
                continue
            if evaluator_id not in registered_predictor_evaluator_ids(predictor_id):
                continue
            predictor_ids.append(predictor_id)

    current_profile = resolve_candidate_agent_profile(genome, registry)
    instruction_ids: list[str] = []
    for instruction_id in registry.program_ids("instruction_templates"):
        instruction = registry.program("instruction_templates", instruction_id)
        if (
            instruction.get("role") == "sample-planner"
            and instruction_id != current_profile["instruction_template_id"]
        ):
            instruction_ids.append(instruction_id)

    return {
        "scientific_parameter": tuple(sorted(parameter_schemas)),
        "registered_predictor": tuple(sorted(set(predictor_ids))),
        "instruction_profile": tuple(sorted(instruction_ids)),
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
    return {
        "allowed_mutation_targets": {
            axis: list(values) for axis, values in targets.items()
        },
        "operation_by_axis": {
            "scientific_parameter": "set_bounded_parameter",
            "registered_predictor": "select_registered_pipeline",
            "instruction_profile": "select_instruction_template",
        },
        "mutation_directions_by_axis": {
            "scientific_parameter": ["increase", "decrease"],
            "registered_predictor": ["select"],
            "instruction_profile": ["select"],
        },
        "mutation_axis_effects": {
            "scientific_parameter": (
                "Changes exactly one registered predictor parameter within its "
                "Host trust region and in the declared increase/decrease "
                "direction; it cannot change the predictor, Planner "
                "instruction, sample cohort, budget, evaluator, or gate."
            ),
            "registered_predictor": (
                "Switches only to one registered predictor using that "
                "predictor's registered defaults; parameter tuning requires a "
                "later generation."
            ),
            "instruction_profile": (
                "Changes only Planner reasoning, confidence, reason codes, and "
                "protocol reliability for the same Host-selected complete "
                "origin vector. It cannot select origins, increase sample "
                "counts, change samples_per_update, alter prediction-tool "
                "values, improve numerical forecast metrics when execution is "
                "already valid, or satisfy statistical evidence gates."
            ),
        },
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
    }


_DIAGNOSTIC_SELECTION_CLAIM_RE = re.compile(
    r"(?:\beligib(?:le|ility)\b|\bpromot(?:e|ion|ed)\b|"
    r"\b(?:selected|advance)\s+(?:for|to)\s+(?:the\s+)?next\s+generation\b|"
    r"\b(?:scientific|selection|statistical|evidence)\s+(?:gate|threshold)\b|"
    r"\b(?:pass|meet|satisfy|exceed)\w*\b[^.;。；]{0,40}"
    r"\b(?:gate|threshold|eligibility)\b|"
    r"\bsufficient\s+evidence\b|\bminimum\s+sample(?:s|\s+count)?\b|"
    r"晋级|入选下一代|进入下一代|门禁|证据充足|样本量达标|"
    r"(?:通过|满足|超过)[^，。；]{0,16}(?:门槛|门禁|阈值))",
    re.IGNORECASE,
)
_INSTRUCTION_SAMPLING_CLAIM_RE = re.compile(
    r"(?:\b(?:increase|decrease|change|raise|lower|expand|reduce|allocate|"
    r"reallocate|choose|select|stratif\w*|concentrat\w*|resampl\w*|"
    r"oversampl\w*|undersampl\w*)\b[^.;。；]{0,48}\b(?:sample|origin|cohort)s?\b|"
    r"\b(?:sample|origin|cohort)s?\b[^.;。；]{0,48}\b(?:increase|decrease|"
    r"change|raise|lower|expand|reduce|allocate|reallocate|choose|select|"
    r"stratif\w*|concentrat\w*|resampl\w*|oversampl\w*|undersampl\w*)\b|"
    r"(?:增加|减少|改变|调整|选择|筛选|分层|重采样|过采样|欠采样|重新分配)"
    r"[^，。；]{0,24}(?:样本|预测起点|起点数量|队列|群组)|"
    r"(?:样本|预测起点|起点数量|队列|群组)[^，。；]{0,24}"
    r"(?:增加|减少|改变|调整|选择|筛选|分层|重采样|过采样|欠采样|重新分配))",
    re.IGNORECASE,
)
_INSTRUCTION_NUMERICAL_CLAIM_RE = re.compile(
    r"(?:\b(?:improv\w*|reduc\w*|lower\w*|decreas\w*|rais\w*|increas\w*|"
    r"boost\w*|optimi[sz]\w*|minimi[sz]\w*|maximi[sz]\w*|change\w*|"
    r"alter\w*)\b[^.;。；]{0,48}\b(?:rmse|mae|mse|r\s*(?:\^?2|²)|"
    r"accuracy|bias|normalized\s+reward|predictive\s+skill|forecast\s+"
    r"(?:score|error|accuracy))\b|"
    r"\b(?:rmse|mae|mse|r\s*(?:\^?2|²)|accuracy|bias|normalized\s+reward|"
    r"predictive\s+skill|forecast\s+(?:score|error|accuracy))\b"
    r"[^.;。；]{0,48}\b(?:improv\w*|reduc\w*|lower\w*|decreas\w*|"
    r"rais\w*|increas\w*|boost\w*|optimi[sz]\w*|minimi[sz]\w*|"
    r"maximi[sz]\w*|change\w*|alter\w*)\b|"
    r"(?:降低|减少|改善|提高|优化|改变|调整)[^，。；]{0,24}"
    r"(?:均方误差|预测误差|预测精度|预测得分|技能得分|偏差|RMSE|MAE|MSE|R²)|"
    r"(?:均方误差|预测误差|预测精度|预测得分|技能得分|偏差|RMSE|MAE|MSE|R²)"
    r"[^，。；]{0,24}(?:降低|减少|改善|提高|优化|改变|调整))",
    re.IGNORECASE,
)
_CLAIM_NEGATION_BEFORE_RE = re.compile(
    r"(?:\b(?:no(?!\s+only\b)|not(?!\s+only\b)|never|without|cannot|can't|does\s+not|doesn't|"
    r"must\s+not|mustn't|will\s+not|won't|avoid)\b|"
    r"不(?!仅|但|只是)|无|未|不会|不能|不得|禁止|避免|保持|维持)[^.;。；!?！？]{0,48}$",
    re.IGNORECASE,
)
_CLAIM_UNCHANGED_AFTER_RE = re.compile(
    r"^[^.;。；!?！？]{0,32}(?:\b(?:unchanged|constant|fixed|preserved)\b|"
    r"不变|不改变|保持不变|维持不变)",
    re.IGNORECASE,
)
_CLAIM_HARD_BOUNDARY_RE = re.compile(
    r"[.;。；!?！？]+|"
    r"\b(?:but|then|yet|however|nevertheless|instead|while)\b|"
    r"但是|然而|然后|不过|却",
    re.IGNORECASE,
)
_CLAIM_SCOPE_RESTART_RE = re.compile(
    r"[,，:：]|\band\b|并且|而且",
    re.IGNORECASE,
)
_CLAIM_NEGATED_ASSERTION_BEFORE_RE = re.compile(
    r"(?:\b(?:do|does|did|will|would|should|must|can|could)\s+not\s+"
    r"(?:claim|assert|imply)\b|\bwithout\s+(?:claiming|asserting|implying)\b|"
    r"(?:不|未|不会|不能|不得)\s*(?:声称|宣称|断言|暗示))"
    r"[^.;。；!?！？]*$",
    re.IGNORECASE,
)
_CLAIM_NEGATED_ENUMERATION_ITEM_RE = re.compile(
    r"^(?:(?:and|or)\s+)?(?:eligib\w*|promot\w*|gate\b|threshold\b|"
    r"(?:scientific|selection|statistical|evidence)[-\s]+(?:gate|threshold)\b|"
    r"(?:门禁|门槛|阈值|晋级|入选|证据充足|样本量达标))",
    re.IGNORECASE,
)


def _claim_scopes(text: str) -> tuple[str, ...]:
    """Return full sentences plus suffixes that restart negation scope.

    Hard discourse boundaries cannot join a predicate to an argument. Within
    each resulting clause, retaining the full text lets a predicate still
    reach a comma-separated modifier (``choose recent, high-error origins``).
    Rechecking suffixes after soft boundaries prevents an earlier ``no`` or
    ``without`` from negating a later independent predicate.
    """

    scopes: list[str] = []
    for clause in _CLAIM_HARD_BOUNDARY_RE.split(text):
        clause = clause.strip()
        if not clause:
            continue
        scopes.append(clause)
        for boundary in _CLAIM_SCOPE_RESTART_RE.finditer(clause):
            suffix = clause[boundary.end() :].strip()
            if not suffix:
                continue
            prefix = clause[: boundary.start()]
            if (
                _CLAIM_NEGATED_ASSERTION_BEFORE_RE.search(prefix)
                and _CLAIM_NEGATED_ENUMERATION_ITEM_RE.match(suffix)
            ):
                continue
            scopes.append(suffix)
    return tuple(scopes)


def _contains_unnegated_claim(text: str, pattern: re.Pattern[str]) -> bool:
    """Return true only for an affirmative prohibited claim.

    The model is encouraged to state invariants explicitly. A raw keyword
    blacklist therefore rejects correct phrases such as ``no RMSE change``.
    Keep the bounded lexical guard, but ignore a match with an adjacent
    negation or unchanged qualifier.
    """

    for scope in _claim_scopes(text):
        for match in pattern.finditer(scope):
            full_before = scope[: match.start()]
            before = full_before[max(0, len(full_before) - 64) :]
            after = scope[match.end() : min(len(scope), match.end() + 40)]
            if boundary := _CLAIM_SCOPE_RESTART_RE.search(after):
                after = after[: boundary.start()]
            if _CLAIM_NEGATED_ASSERTION_BEFORE_RE.search(full_before):
                continue
            if _CLAIM_NEGATION_BEFORE_RE.search(before):
                continue
            if _CLAIM_NEGATION_BEFORE_RE.search(match.group()):
                continue
            if _CLAIM_UNCHANGED_AFTER_RE.search(match.group()):
                continue
            if _CLAIM_UNCHANGED_AFTER_RE.search(after):
                continue
            return True
    return False


def _validate_direction_claim_scope(
    direction: CandidateDirection,
    *,
    task: TaskManifest,
) -> None:
    """Reject outcome claims that the declared mutation cannot causally affect."""

    claim_fields = (
        direction.title,
        direction.hypothesis,
        direction.target_weakness,
        direction.capability_focus,
        direction.expected_tradeoff,
        direction.success_criterion,
    )
    if (
        task.metadata.get("sample_budget_class") == "diagnostic_smoke"
        and any(
            _contains_unnegated_claim(field, _DIAGNOSTIC_SELECTION_CLAIM_RE)
            for field in claim_fields
        )
    ):
        raise ValueError(
            f"candidate direction {direction.direction_id} uses a selection or "
            "promotion success criterion in a diagnostic-only run"
        )
    if direction.mutation_axis != "instruction_profile":
        return
    if any(
        _contains_unnegated_claim(field, _INSTRUCTION_SAMPLING_CLAIM_RE)
        or _contains_unnegated_claim(field, _INSTRUCTION_NUMERICAL_CLAIM_RE)
        for field in claim_fields
    ):
        raise ValueError(
            f"candidate direction {direction.direction_id} assigns an "
            "instruction profile an effect owned by the Host or registered "
            "prediction tool"
        )


def _direction_explicit_parameter_assignments(
    direction: CandidateDirection,
    *,
    parent: EcologyEvolutionPluginGenome,
) -> tuple[str, ...]:
    """Find exact parameter settings hidden in free-form direction prose.

    A direction declares an axis, target, and increase/decrease intent.  The
    candidate proposer owns the bounded value selection.  Accepting an exact
    value in prose would create a second, unchecked source of truth even when
    it names the declared target itself.
    """

    registry = current_program_registry()
    # An exact value for any registered predictor parameter would add a second
    # operation to the direction, even when that parameter is not part of the
    # current predictor.  Inspect the complete bounded registry rather than
    # allowing cross-predictor assignments to evade this guard.
    predictor_ids = set(registry.program_ids("predictors"))
    predictor_ids.add(str(parent.scientific_program["predictor_ref"]["id"]))
    if direction.mutation_axis == "registered_predictor":
        predictor_ids.add(direction.mutation_target)
    parameter_names: set[str] = set()
    for predictor_id in predictor_ids:
        try:
            predictor = registry.program("predictors", predictor_id)
        except ValueError:
            continue
        parameters = predictor.get("parameters")
        if isinstance(parameters, Mapping):
            parameter_names.update(str(name) for name in parameters)

    claim_fields = (
        direction.title,
        direction.hypothesis,
        direction.target_weakness,
        direction.capability_focus,
        direction.expected_tradeoff,
        direction.success_criterion,
    )
    assignments: list[str] = []
    number = _EXACT_PARAMETER_VALUE_PATTERN
    for name in sorted(parameter_names):
        aliases = {name, name.replace("_", " ")}
        for catalog in _PARAMETER_ALIASES.values():
            aliases.update(catalog.get(name, ()))
        alias_pattern = "(?:" + "|".join(
            r"\s+".join(re.escape(part) for part in alias.split())
            for alias in sorted(aliases, key=lambda item: (-len(item), item))
        ) + ")"
        bounded_alias = rf"(?<![A-Za-z0-9_]){alias_pattern}(?![A-Za-z0-9_])"
        assignment_after = re.compile(
            bounded_alias
            + (
                rf"(?:\s*(?:=|:=|:)\s*{number}"
                rf"|\s*(?:->|=>|→|⇒)\s*{number}"
                rf"|\s+(?:is|becomes?|equals?|to|at|of|by)\s+(?:the\s+)?{number}"
                rf"|\s+(?:should|must|will)\s+(?:be|equal|become)"
                rf"\s+(?:to\s+)?{number}"
                rf"|\s+(?:set|fixed|changed|adjusted|raised|lowered|"
                rf"increased|decreased|reduced)\s+(?:to|at|by)\s+{number}"
                rf"|\s+from\s+{number}\s+to\s+{number}"
                rf"|\s*(?:设为|设成|设置为|设置成|改为|改成|"
                rf"变为|变成|取|调(?:整)?至|调(?:整)?为|"
                rf"升至|增(?:加)?到|提高到|降至|降到|"
                rf"降低到|减至|减少到|为)\s*{number})"
            ),
            re.IGNORECASE,
        )
        assignment_before = re.compile(
            (
                rf"(?:\b(?:use|set|assign|choose|keep|fix|make)\b\s+(?:the\s+)?"
                rf"(?:{number}\s+(?:(?:for|as)\s+)?{bounded_alias}"
                rf"|{bounded_alias}\s*(?:(?:to|at|as)\s+|(?:=|:=|:|->|=>|→|⇒)\s*)?{number})"
                rf"|(?:使用|采用|选用|设置|设定|取)\s*{number}"
                rf"\s*(?:个|步)?\s*{bounded_alias}"
                rf"|(?:将|把)?\s*(?:使用|采用|选用|设置|设定|"
                rf"调整|改变|修改|令|使)\s*{bounded_alias}"
                rf"\s*(?:为|成|至|到|=|:=|:|：|->|=>|→|⇒)?\s*{number})"
            ),
            re.IGNORECASE,
        )
        if any(
            assignment_after.search(field) or assignment_before.search(field)
            for field in claim_fields
        ):
            assignments.append(name)
    return tuple(assignments)


def _reject_direction_explicit_parameter_assignments(
    direction: CandidateDirection,
    *,
    parent: EcologyEvolutionPluginGenome,
    index: int,
) -> None:
    explicit_parameters = _direction_explicit_parameter_assignments(
        direction,
        parent=parent,
    )
    if explicit_parameters:
        raise ValueError(
            f"candidate_directions[{index}] contains explicit parameter "
            f"assignment(s) for {', '.join(explicit_parameters)}; declare "
            "only mutation_target and mutation_direction so the bounded "
            "candidate proposer remains the sole value authority"
        )


def _parameter_preflight_values(
    schema: Mapping[str, Any],
    current_value: Any,
    mutation_direction: str,
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
    witness_count = _NATIVE_AVOID_BEHAVIOR_LIMIT + 8
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


def _validate_candidate_direction_realizability(
    raw_directions: Sequence[Mapping[str, Any]],
    *,
    run: Run,
    task: TaskManifest,
    parent: EcologyEvolutionPluginGenome,
    avoid_behaviors: Sequence[Mapping[str, Any]],
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
    registry = current_program_registry()
    _boundary, parameter_schemas = _genome_parameter_boundary(task, parent)
    parent_parameters = parent.to_dict()["scientific_program"][
        "parameter_overrides"
    ]
    occupied_behavior_digests: set[str] = set()
    checks: list[dict[str, Any]] = []

    for index, direction in enumerate(directions):
        _validate_direction_claim_scope(direction, task=task)
        _reject_direction_explicit_parameter_assignments(
            direction,
            parent=parent,
            index=index,
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
        else:
            operations = (
                {
                    "op": "select_instruction_template",
                    "role": "sample-planner",
                    "instruction_template_id": direction.mutation_target,
                },
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
            mutation_operator_id=TRUST_REGION_MUTATION_OPERATOR_ID,
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
            if child.behavior_digest in hard_avoid_digests:
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


def _research_plan_predictor_id(plan: Mapping[str, Any]) -> str | None:
    blueprint = plan.get("algorithm_blueprint")
    if isinstance(blueprint, Mapping):
        pipeline_id = blueprint.get("pipeline_id")
        if isinstance(pipeline_id, str) and pipeline_id.strip():
            return pipeline_id.strip()
    prediction_model = plan.get("prediction_model")
    if isinstance(prediction_model, Mapping):
        predictor_id = prediction_model.get("id")
        if isinstance(predictor_id, str) and predictor_id.strip():
            return predictor_id.strip()
    return None


def _merge_research_plan(
    current_plan: Mapping[str, Any],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the active algorithm when an iteration only updates guidance."""

    current = dict(current_plan)
    result = dict(update)
    current.pop("expert_consultation", None)
    result.pop("expert_consultation", None)
    current_predictor = _research_plan_predictor_id(current)
    requested_predictor = _research_plan_predictor_id(result)
    explicitly_requested = any(
        field in result for field in ("prediction_model", "algorithm_blueprint")
    )

    if not explicitly_requested or requested_predictor == current_predictor:
        inherited_fields = ["prediction_model"]
        if "algorithm_synthesis_degradation" not in result:
            inherited_fields.extend(
                ["algorithm_blueprint", "algorithm_synthesis"]
            )
        for field in inherited_fields:
            if field not in result and field in current:
                result[field] = deepcopy(current[field])
    return result


def _task_parameter_space(
    metadata: Mapping[str, Any],
    domain: str,
) -> tuple[
    str,
    Mapping[str, Mapping[str, Any]],
    tuple[Mapping[str, int | float], ...],
]:
    if domain != "greenhouse":
        return (
            "toy",
            StrategyRouterDSHAdapter._TOY_SCHEMAS,
            StrategyRouterDSHAdapter._TOY_SWEEP,
        )
    predictor_id = str(
        metadata.get("prediction_model_id") or "greenhouse-rolling-residual@1"
    )
    if predictor_id == EXOGENOUS_RIDGE_MODEL_ID:
        return (
            "greenhouse_ridge",
            StrategyRouterDSHAdapter._GREENHOUSE_RIDGE_SCHEMAS,
            StrategyRouterDSHAdapter._GREENHOUSE_RIDGE_SWEEP,
        )
    if predictor_id == TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID:
        return (
            "greenhouse_targetwise_ridge",
            StrategyRouterDSHAdapter._GREENHOUSE_TARGETWISE_RIDGE_SCHEMAS,
            StrategyRouterDSHAdapter._GREENHOUSE_TARGETWISE_RIDGE_SWEEP,
        )
    if predictor_id == HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID:
        return (
            "greenhouse_horizon_targetwise_ridge",
            StrategyRouterDSHAdapter._GREENHOUSE_HORIZON_TARGETWISE_RIDGE_SCHEMAS,
            StrategyRouterDSHAdapter._GREENHOUSE_HORIZON_TARGETWISE_RIDGE_SWEEP,
        )
    if predictor_id == "greenhouse-rolling-residual@1":
        return (
            "greenhouse",
            StrategyRouterDSHAdapter._GREENHOUSE_SCHEMAS,
            StrategyRouterDSHAdapter._GREENHOUSE_SWEEP,
        )
    raise ValueError(f"unsupported prediction_model_id: {predictor_id}")


def _autonomous_search_name(plan: Mapping[str, Any]) -> str | None:
    strategy = plan.get("strategy")
    if not isinstance(strategy, Mapping):
        return None
    value = strategy.get("name", strategy.get("id"))
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:200]


def _project_single_parameter_change(
    seed: Mapping[str, int | float],
    proposed: Mapping[str, Any],
    schemas: Mapping[str, Mapping[str, Any]],
    *,
    slot_index: int,
) -> tuple[dict[str, int | float], dict[str, Any]]:
    """Enforce one identifiable parameter change from a shared sibling seed."""

    reference = _bounded_parameters(
        seed,
        schemas,
        partial=False,
        source="single_parameter_sweep.seed",
    )
    requested = _bounded_parameters(
        proposed,
        schemas,
        partial=True,
        source="single_parameter_sweep.remote_parameters",
    )
    changed = [
        name
        for name in schemas
        if name in requested
        and not math.isclose(
            float(requested[name]),
            float(reference[name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    source = "remote_parameter"
    if changed:
        preferred = tuple(schemas)[slot_index % len(schemas)]
        adopted = preferred if preferred in changed else changed[0]
        result = dict(reference)
        result[adopted] = requested[adopted]
    else:
        result = StrategyRouterDSHAdapter._sweep_around_parent(
            reference,
            schemas,
            slot_index,
        )
        adopted = next(
            name
            for name in schemas
            if not math.isclose(
                float(result[name]),
                float(reference[name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        source = "host_bounded_fallback"
    return result, {
        "policy": "bounded_single_parameter_sweep@1",
        "shared_reference_parameters": dict(reference),
        "remote_changed_parameters": changed,
        "adopted_parameter": adopted,
        "parameter_source": source,
        "host_projection_applied": len(changed) != 1,
    }


from .interventions import apply_bounded_interventions
