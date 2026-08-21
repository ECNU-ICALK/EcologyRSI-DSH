"""Deterministic batch contracts, ranking, and public feedback analysis."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean, median
from typing import Any

from ..core.models import CandidateStatus, JsonObject, canonical_json, digest, utc_now
from .promotion import assess_promotion_improvement
from ..evaluators.fitness import (
    EXPLORATORY_EVIDENCE_CLASS,
    FitnessProfile,
    assess_generation_selection,
    build_fitness_assessment,
)

_IMPROVEMENT_TOLERANCE = 1e-12
_EXPERIENCE_MAX_GENERATIONS = 6
_EXPERIENCE_SCAN_GENERATIONS = 24
_EXPERIENCE_MAX_ACTIVE_ISSUES = 16
_EXPERIENCE_MAX_ARCHIVED_ISSUES = 16
_EXPERIENCE_MAX_WEAKNESSES = 12
_METRIC_WEAKNESS_MAX_GROUPS = 32
_CURRENT_RUN_EXPERIENCE_MAX_BYTES = 16 * 1024
_HISTORICAL_EXPERIENCE_MAX_BYTES = 8 * 1024
_HISTORICAL_EXPERIENCE_SCAN_RUNS = 24
_HISTORICAL_EXPERIENCE_MAX_RUNS = 8
_HISTORICAL_EXPERIENCE_MAX_GENERATIONS = 12
_HISTORICAL_PARAMETER_GUARDRAIL_MAX_ITEMS = 16
_HISTORICAL_PARAMETER_GUARDRAIL_MIN_COHORTS = 2
_HISTORICAL_PARAMETER_GUARDRAIL_MIN_CELL_N_PER_COHORT = 20
_HISTORICAL_PARAMETER_GUARDRAIL_MIN_TOTAL_CELL_N = (
    _HISTORICAL_PARAMETER_GUARDRAIL_MIN_COHORTS
    * _HISTORICAL_PARAMETER_GUARDRAIL_MIN_CELL_N_PER_COHORT
)
_HISTORICAL_PARAMETER_GUARDRAIL_POLICY = (
    "preserve_verified_target_horizon_parameters_without_new_aggregate_evidence"
)
CROSS_GENERATION_EXPERIENCE_MAX_BYTES = (
    _CURRENT_RUN_EXPERIENCE_MAX_BYTES + _HISTORICAL_EXPERIENCE_MAX_BYTES
)
_EXPERIENCE_PLAN_FIELDS = (
    "team",
    "prediction_model",
    "strategy",
    "research",
    "algorithm_blueprint",
    "algorithm_synthesis",
    "implementation_notes",
    "confidence",
)
_EXPERIENCE_BLOCKED_PARAMETER_NAMES = frozenset(
    {
        "baseline",
        "labels",
        "observations",
        "observed",
        "predicted",
        "prediction_records",
        "raw",
        "rows",
        "samples",
        "sample_execution_records",
        "timestamps",
    }
)
_TARGET_LABELS = {
    "air_temperature": "室内气温",
    "relative_humidity": "室内相对湿度",
    "co2_concentration": "室内二氧化碳浓度",
    "soil_moisture": "土壤含水量",
    "soil_water": "土壤含水量",
}
_ALGORITHM_DETAIL_TEXT_FIELDS = frozenset(
    {
        "algorithm_ir_digest",
        "predictor_id",
        "smoke_digest",
        "source_partition",
    }
)
_ALGORITHM_DETAIL_BOOLEAN_FIELDS = frozenset(
    {
        "registered_adapters_only",
        "registered_operators_only",
        "restricted_partition_access",
    }
)
_ALGORITHM_DETAIL_INTEGER_FIELDS = frozenset(
    {
        "knowledge_mapping_count",
        "knowledge_not_selected_count",
        "tool_count",
    }
)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def sample_update_windows_enabled(task: Any) -> bool:
    """Return whether a frozen task evaluates one bounded feedback window."""

    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    value = metadata.get("samples_per_update")
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def evaluation_cohort_digest(evaluation: Any) -> str | None:
    """Resolve the candidate-independent score cohort identity.

    ``evaluation_index_digest`` binds the rows that actually contributed to the
    score. The explicit pre-execution feedback digest remains a compatibility
    fallback when the scored-index digest is unavailable.
    """

    metrics = getattr(evaluation, "metrics", None)
    if not isinstance(metrics, Mapping):
        return None
    for name in ("evaluation_index_digest", "feedback_update_cohort_digest"):
        value = metrics.get(name)
        if (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            return value
    return None


def _evaluation_cohort_window_summary(
    metrics: Mapping[str, Any],
    *,
    cohort_digest: str,
) -> dict[str, Any] | None:
    raw_cohort = metrics.get("evaluation_cohort")
    raw_window = (
        raw_cohort.get("update_window")
        if isinstance(raw_cohort, Mapping)
        else None
    )
    if not isinstance(raw_window, Mapping):
        return None
    schema_version = raw_window.get("schema_version")
    selection_policy = raw_window.get("selection_policy")
    population_count = raw_window.get("population_count")
    population_digest = raw_window.get("population_digest")
    selected_count = raw_window.get("selected_count")
    window_offset = raw_window.get("window_offset")
    feedback_cohort_digest = metrics.get("feedback_update_cohort_digest")
    if (
        schema_version != "ecologyrsi-dsh.feedback-update-cohort/1"
        or selection_policy
        != "target_horizon_interleaved_rotating_window@1"
        or not isinstance(feedback_cohort_digest, str)
        or len(feedback_cohort_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in feedback_cohort_digest
        )
        or raw_window.get("cohort_digest") != feedback_cohort_digest
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
        or isinstance(window_offset, bool)
        or not isinstance(window_offset, int)
        or window_offset < 0
        or window_offset >= population_count
    ):
        return None
    return {
        "evaluation_cohort_digest": cohort_digest,
        "feedback_update_cohort_digest": feedback_cohort_digest,
        "schema_version": schema_version,
        "selection_policy": selection_policy,
        "population_count": population_count,
        "population_digest": population_digest,
        "selected_count": selected_count,
        "window_offset": window_offset,
    }


def evaluation_cohort_comparison(
    task: Any,
    evaluation: Any,
    incumbent_evaluation: Any,
) -> str:
    """Classify whether two scores may be compared directly.

    Full-cohort legacy runs retain their historical score semantics. Bounded
    update runs require explicit cohort evidence: equal digests permit a paired
    score comparison, while unequal digests identify distinct, non-comparable
    windows. Digest inequality alone does not establish sample independence.
    """

    if not sample_update_windows_enabled(task):
        return "legacy_full_cohort"
    current_digest = evaluation_cohort_digest(evaluation)
    incumbent_digest = evaluation_cohort_digest(incumbent_evaluation)
    if current_digest is None or incumbent_digest is None:
        return "unverifiable"
    if current_digest == incumbent_digest:
        return "same_cohort"
    return "different_cohort"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _json_rows(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an array")
    rows = tuple(dict(item) for item in value)
    canonical_json(rows)
    return rows


def _text_rows(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an array")
    return tuple(_text(item, f"{name} item") for item in value)


@dataclass(frozen=True, slots=True)
class GenerationBatch:
    """Frozen context shared by every sibling candidate in one generation."""

    run_id: str
    generation: int
    batch_size: int
    task_manifest_digest: str
    parent_candidate_id: str | None = None
    parent_genome_digest: str | None = None
    parent_genome_canonical_json: str | None = None
    stage_context_digests: Mapping[str, str] | None = None
    previous_analysis_digest: str | None = None
    knowledge_snapshot_digest: str | None = None
    intervention_ids: tuple[str, ...] = ()
    context_digest: str = ""
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "task_manifest_digest",
            _text(self.task_manifest_digest, "task_manifest_digest"),
        )
        object.__setattr__(self, "generation", _integer(self.generation, "generation"))
        size = _integer(self.batch_size, "batch_size", 1)
        if size > 8:
            raise ValueError("batch_size must be <= 8")
        object.__setattr__(self, "batch_size", size)
        object.__setattr__(
            self,
            "parent_candidate_id",
            _optional_text(self.parent_candidate_id, "parent_candidate_id"),
        )
        new_parent_fields = (
            self.parent_genome_digest,
            self.parent_genome_canonical_json,
            self.stage_context_digests,
        )
        if any(item is not None for item in new_parent_fields):
            if any(item is None for item in new_parent_fields):
                raise ValueError("generation batch parent genome binding is incomplete")
            parent_digest = _text(self.parent_genome_digest, "parent_genome_digest")
            parent_canonical = _text(
                self.parent_genome_canonical_json,
                "parent_genome_canonical_json",
            )
            if len(parent_digest) != 64 or any(
                character not in "0123456789abcdef" for character in parent_digest
            ):
                raise ValueError("parent_genome_digest must be a SHA-256 digest")
            try:
                from .genome import EcologyEvolutionPluginGenome

                parent = EcologyEvolutionPluginGenome.from_dict(json.loads(parent_canonical))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("parent_genome_canonical_json is invalid") from exc
            if canonical_json(parent.to_dict()) != parent_canonical:
                raise ValueError("parent genome JSON must be canonical")
            if parent.genome_digest != parent_digest:
                raise ValueError("parent genome digest mismatch")
            if not isinstance(self.stage_context_digests, Mapping):
                raise TypeError("stage_context_digests must be an object")
            stage_digests = dict(self.stage_context_digests)
            if not stage_digests:
                raise ValueError("stage_context_digests must not be empty")
            for name, value in stage_digests.items():
                _text(name, "stage context name")
                item = _text(value, f"stage_context_digests.{name}")
                if len(item) != 64 or any(
                    character not in "0123456789abcdef" for character in item
                ):
                    raise ValueError(
                        f"stage_context_digests.{name} must be a SHA-256 digest"
                    )
            object.__setattr__(self, "parent_genome_digest", parent_digest)
            object.__setattr__(self, "parent_genome_canonical_json", parent_canonical)
            object.__setattr__(
                self,
                "stage_context_digests",
                {name: stage_digests[name] for name in sorted(stage_digests)},
            )
        object.__setattr__(
            self,
            "previous_analysis_digest",
            _optional_text(self.previous_analysis_digest, "previous_analysis_digest"),
        )
        object.__setattr__(
            self,
            "knowledge_snapshot_digest",
            _optional_text(self.knowledge_snapshot_digest, "knowledge_snapshot_digest"),
        )
        object.__setattr__(
            self,
            "intervention_ids",
            tuple(_text(item, "intervention_id") for item in self.intervention_ids),
        )
        expected = digest(self.identity_dict())
        if self.context_digest and self.context_digest != expected:
            raise ValueError("generation batch context_digest mismatch")
        object.__setattr__(self, "context_digest", expected)
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at"))

    def identity_dict(self) -> JsonObject:
        result: JsonObject = {
            "run_id": self.run_id,
            "generation": self.generation,
            "batch_size": self.batch_size,
            "task_manifest_digest": self.task_manifest_digest,
            "parent_candidate_id": self.parent_candidate_id,
            "previous_analysis_digest": self.previous_analysis_digest,
            "intervention_ids": list(self.intervention_ids),
        }
        # Omit this field for legacy batches so their recorded digest remains valid.
        if self.knowledge_snapshot_digest is not None:
            result["knowledge_snapshot_digest"] = self.knowledge_snapshot_digest
        # New DSH-native batches freeze their complete inheritable parent and
        # every host-owned stage identity. Historical batches omit all fields.
        if self.parent_genome_digest is not None:
            result.update(
                {
                    "parent_genome_digest": self.parent_genome_digest,
                    "parent_genome_canonical_json": self.parent_genome_canonical_json,
                    "stage_context_digests": dict(self.stage_context_digests or {}),
                }
            )
        return result

    def to_dict(self) -> JsonObject:
        return {
            **self.identity_dict(),
            "context_digest": self.context_digest,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GenerationBatch:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class GenerationAnalysis:
    """Aggregate-only analysis used for selection and the next generation."""

    run_id: str
    generation: int
    candidate_count: int
    eligible_count: int
    outcome: str
    selected_candidate_id: str | None = None
    champion_candidate_id: str | None = None
    incumbent_before_candidate_id: str | None = None
    incumbent_after_candidate_id: str | None = None
    search_parent_candidate_id: str | None = None
    ranking: tuple[Mapping[str, Any], ...] = ()
    common_failures: tuple[str, ...] = ()
    target_weaknesses: tuple[Mapping[str, Any], ...] = ()
    horizon_weaknesses: tuple[Mapping[str, Any], ...] = ()
    constraint_failures: tuple[str, ...] = ()
    judge_disagreements: tuple[Mapping[str, Any], ...] = ()
    parameter_effects: tuple[Mapping[str, Any], ...] = ()
    next_search_direction: tuple[str, ...] = ()
    next_generation_focus: str = ""
    selection_reason: str = ""
    insufficient_evidence: bool = False
    algorithm_failures: tuple[Mapping[str, Any], ...] = ()
    sample_failures: tuple[Mapping[str, Any], ...] = ()
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(self, "generation", _integer(self.generation, "generation"))
        object.__setattr__(
            self, "candidate_count", _integer(self.candidate_count, "candidate_count")
        )
        eligible = _integer(self.eligible_count, "eligible_count")
        if eligible > self.candidate_count:
            raise ValueError("eligible_count cannot exceed candidate_count")
        object.__setattr__(self, "eligible_count", eligible)
        object.__setattr__(self, "outcome", _text(self.outcome, "outcome"))
        for name in (
            "selected_candidate_id",
            "champion_candidate_id",
            "incumbent_before_candidate_id",
            "incumbent_after_candidate_id",
            "search_parent_candidate_id",
        ):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))
        for name in (
            "ranking",
            "target_weaknesses",
            "horizon_weaknesses",
            "judge_disagreements",
            "parameter_effects",
            "algorithm_failures",
            "sample_failures",
        ):
            object.__setattr__(self, name, _json_rows(getattr(self, name), name))
        for name in ("common_failures", "constraint_failures", "next_search_direction"):
            object.__setattr__(self, name, _text_rows(getattr(self, name), name))
        if not isinstance(self.insufficient_evidence, bool):
            raise TypeError("insufficient_evidence must be a bool")
        if not isinstance(self.next_generation_focus, str):
            raise TypeError("next_generation_focus must be a string")
        if not isinstance(self.selection_reason, str):
            raise TypeError("selection_reason must be a string")
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at"))

    def identity_dict(self) -> JsonObject:
        result: JsonObject = {
            "schema_version": "ecologyrsi-dsh.generation-analysis/1",
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_count": self.candidate_count,
            "eligible_count": self.eligible_count,
            "outcome": self.outcome,
            "selected_candidate_id": self.selected_candidate_id,
            "champion_candidate_id": self.champion_candidate_id,
            "incumbent_before_candidate_id": self.incumbent_before_candidate_id,
            "incumbent_after_candidate_id": self.incumbent_after_candidate_id,
            "ranking": [dict(item) for item in self.ranking],
            "common_failures": list(self.common_failures),
            "target_weaknesses": [dict(item) for item in self.target_weaknesses],
            "horizon_weaknesses": [dict(item) for item in self.horizon_weaknesses],
            "constraint_failures": list(self.constraint_failures),
            "judge_disagreements": [dict(item) for item in self.judge_disagreements],
            "parameter_effects": [dict(item) for item in self.parameter_effects],
            "next_search_direction": list(self.next_search_direction),
            "next_generation_focus": self.next_generation_focus,
            "selection_reason": self.selection_reason,
            "insufficient_evidence": self.insufficient_evidence,
        }
        # Legacy analysis events predate the distinction between the formally
        # promoted incumbent and the candidate used to continue the search.
        # Omitting a missing value preserves their recorded digest on replay.
        if self.search_parent_candidate_id is not None:
            result["search_parent_candidate_id"] = self.search_parent_candidate_id
        if self.algorithm_failures:
            result["algorithm_failures"] = [
                dict(item) for item in self.algorithm_failures
            ]
        if self.sample_failures:
            result["sample_failures"] = [dict(item) for item in self.sample_failures]
        return result

    @property
    def analysis_digest(self) -> str:
        return digest(self.identity_dict())

    def to_dict(self) -> JsonObject:
        return {
            **self.identity_dict(),
            "analysis_digest": self.analysis_digest,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GenerationAnalysis:
        data = dict(value)
        stored = data.pop("analysis_digest", None)
        data.pop("schema_version", None)
        item = cls(**data)
        if stored is not None and stored != item.analysis_digest:
            raise ValueError("generation analysis digest mismatch")
        return item


def _finite(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _skill(metrics: Mapping[str, Any]) -> float:
    direct = _finite(metrics.get("skill_score"), math.nan)
    if math.isfinite(direct):
        return direct
    value = _finite(metrics.get("normalized_rmse"), math.inf)
    baseline = _finite(metrics.get("baseline_normalized_rmse"), 0.0)
    if baseline > _IMPROVEMENT_TOLERANCE and math.isfinite(value):
        return 1.0 - value / baseline
    return -math.inf


def _worst_skill(metrics: Mapping[str, Any]) -> float:
    rows: list[float] = []
    for field_name in ("targets", "horizons"):
        raw = metrics.get(field_name)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, Mapping):
                value = _skill(item)
                if math.isfinite(value):
                    rows.append(value)
    return min(rows) if rows else _skill(metrics)


def _parameter_summary(value: Mapping[str, Any]) -> dict[str, int | float]:
    """Return only bounded-size finite numeric proposal parameters."""

    result: dict[str, int | float] = {}
    for name in sorted(value)[:32]:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        number = float(raw)
        if not math.isfinite(number):
            continue
        result[name] = raw
    return result


def _target_skill_summary(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose aggregate target skills without observations or predictions."""

    raw_rows = metrics.get("targets")
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw in raw_rows[:32]:
        if not isinstance(raw, Mapping):
            continue
        skill = _skill(raw)
        if not math.isfinite(skill):
            continue
        row: dict[str, Any] = {"skill_score": skill}
        for name in ("target", "unit"):
            value = raw.get(name)
            if isinstance(value, str) and value.strip():
                row[name] = value.strip()[:200]
        horizon = raw.get("horizon_hours")
        if (
            not isinstance(horizon, bool)
            and isinstance(horizon, (int, float))
            and math.isfinite(float(horizon))
        ):
            row["horizon_hours"] = horizon
        cell_sample_count = raw.get("n")
        if (
            isinstance(cell_sample_count, int)
            and not isinstance(cell_sample_count, bool)
            and cell_sample_count > 0
        ):
            row["n"] = cell_sample_count
        for name in (
            "mean_reward",
            "normalized_mean_reward",
            "negative_reward_fraction",
        ):
            value = _finite(raw.get(name), math.nan)
            if math.isfinite(value):
                row[name] = value
        rows.append(row)
    return rows


def _parameter_distance(state: Any, candidate: Any, parent_id: str | None) -> float:
    if parent_id is None:
        return 0.0
    try:
        parent = state.candidate(parent_id)
        parent_proposal = state.proposal(parent.proposal_id)
        proposal = state.proposal(candidate.proposal_id)
    except (KeyError, ValueError):
        return math.inf
    distance = 0.0
    for name, raw in proposal.changes.items():
        before = parent_proposal.changes.get(name)
        if isinstance(raw, bool) or isinstance(before, bool):
            continue
        if isinstance(raw, (int, float)) and isinstance(before, (int, float)):
            distance += abs(float(raw) - float(before)) / max(abs(float(before)), 1.0)
    return distance


def _candidate_row(state: Any, candidate: Any, parent_id: str | None) -> dict[str, Any]:
    evaluation = state.evaluation_for(candidate.candidate_id)
    metrics = dict(evaluation.metrics) if evaluation is not None else {}
    proposal = state.proposal(candidate.proposal_id)
    execution_success = evaluation is not None
    scientific_pass = (
        bool(metrics.get("scientific_pass", evaluation.passed))
        if evaluation
        else False
    )
    raw_judge_accepted = metrics.get("judge_accepted")
    judge_status = metrics.get("judge_status")
    judge_available = bool(
        evaluation is not None
        and (
            judge_status == "completed"
            or (judge_status is None and isinstance(raw_judge_accepted, bool))
        )
        and isinstance(raw_judge_accepted, bool)
    )
    judge_accepted = bool(raw_judge_accepted) if judge_available else False
    eligible = execution_success and scientific_pass and judge_available and judge_accepted
    violations = int(max(0, _finite(metrics.get("constraint_violations"), 0.0)))
    worst_skill = _worst_skill(metrics) if evaluation is not None else -math.inf
    if evaluation is not None and not math.isfinite(worst_skill):
        worst_skill = evaluation.score
    failed_algorithm_attempt = next(
        (
            item
            for item in reversed(state.algorithm_attempts_for(candidate.candidate_id))
            if item.status == "failed"
        ),
        None,
    )
    classification = (
        "duplicate"
        if candidate.status is CandidateStatus.DUPLICATE
        else f"algorithm_{failed_algorithm_attempt.phase}_failed"
        if failed_algorithm_attempt is not None
        else "execution_failed"
        if candidate.status is CandidateStatus.FAILED or evaluation is None
        else "scientific_gate_failed"
        if not scientific_pass
        else "judge_unavailable"
        if not judge_available
        else "judge_rejected"
        if not judge_accepted
        else "eligible"
    )
    row = {
        "candidate_id": candidate.candidate_id,
        "slot_index": candidate.slot_index,
        "status": candidate.status.value,
        "score": evaluation.score if evaluation is not None else None,
        "scientific_pass": scientific_pass,
        "judge_status": "completed" if judge_available else "unavailable",
        "judge_available": judge_available,
        "judge_accepted": judge_accepted,
        "eligible": eligible,
        "constraint_violations": violations,
        "worst_skill_score": (
            worst_skill if evaluation is not None else None
        ),
        "parameter_distance": _parameter_distance(state, candidate, parent_id),
        "parameters": _parameter_summary(proposal.changes),
        "target_skill_scores": _target_skill_summary(metrics),
        "classification": classification,
    }
    cohort_digest = evaluation_cohort_digest(evaluation)
    if cohort_digest is not None:
        row["evaluation_cohort_digest"] = cohort_digest
        cohort_window = _evaluation_cohort_window_summary(
            metrics,
            cohort_digest=cohort_digest,
        )
        if cohort_window is not None:
            row["evaluation_cohort_window"] = cohort_window
    sample_summary = metrics.get("sample_execution")
    if isinstance(sample_summary, Mapping):
        tool_performance = _bounded_tool_performance(
            sample_summary.get("tool_performance")
        )
        if tool_performance:
            row["tool_performance"] = tool_performance
    return row


def _algorithm_failure_summary(
    state: Any, candidates: list[Any]
) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str, str, bool, str, str], list[str]] = defaultdict(list)
    for candidate in candidates:
        for attempt in state.algorithm_attempts_for(candidate.candidate_id):
            if attempt.status != "failed":
                continue
            stage, retryable, error_type, details = _algorithm_failure_metadata(attempt)
            grouped[
                (
                    attempt.phase,
                    stage,
                    str(attempt.failure_code or "unknown")[:100],
                    retryable,
                    error_type,
                    canonical_json(details),
                )
            ].append(candidate.candidate_id)
    rows = [
        {
            "phase": phase,
            "stage": stage,
            "failure_code": failure_code,
            "retryable": retryable,
            "error_type": error_type,
            "attempt_count": len(candidate_ids),
            # Compatibility alias retained for established analysis clients.
            "count": len(candidate_ids),
            "candidate_ids": candidate_ids[:8],
            "details": json_details,
        }
        for (
            phase,
            stage,
            failure_code,
            retryable,
            error_type,
            encoded_details,
        ), candidate_ids in sorted(grouped.items())
        for json_details in (json.loads(encoded_details),)
    ]
    return tuple(rows[:16])


def _algorithm_failure_metadata(attempt: Any) -> tuple[str, bool, str, dict[str, Any]]:
    """Project an attempt's evidence onto a bounded, non-error-text schema."""

    evidence = attempt.evidence if isinstance(attempt.evidence, Mapping) else {}
    feedback = evidence.get("failure_feedback")
    feedback = feedback if isinstance(feedback, Mapping) else {}
    stage_value = evidence.get("stage")
    stage = (
        stage_value.strip()[:120]
        if isinstance(stage_value, str) and stage_value.strip()
        else str(attempt.phase)[:120]
    )
    retryable = feedback.get("retryable")
    retryable = bool(retryable) if isinstance(retryable, bool) else False
    error_value = feedback.get("exception_type", evidence.get("exception_type"))
    error_type = (
        error_value.strip()[:120]
        if isinstance(error_value, str) and error_value.strip()
        else "UnknownError"
    )
    details = _algorithm_failure_details(evidence, feedback)
    return stage, retryable, error_type, details


def _algorithm_failure_details(
    evidence: Mapping[str, Any], feedback: Mapping[str, Any]
) -> dict[str, Any]:
    """Retain only fixed identifiers and finite counters from debug evidence."""

    sources = (evidence, feedback.get("details"))
    details: dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for name in _ALGORITHM_DETAIL_TEXT_FIELDS:
            value = source.get(name)
            if isinstance(value, str) and value.strip():
                details[name] = value.strip()[:200]
        for name in _ALGORITHM_DETAIL_BOOLEAN_FIELDS:
            value = source.get(name)
            if isinstance(value, bool):
                details[name] = value
        for name in _ALGORITHM_DETAIL_INTEGER_FIELDS:
            value = source.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                details[name] = min(value, 1_000_000_000)
    return dict(sorted(details.items()))


def _sample_failure_summary(
    state: Any, candidates: list[Any]
) -> tuple[dict[str, Any], ...]:
    """Expose counts only; raw samples and predictions never enter strategy context."""

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        evaluation = state.evaluation_for(candidate.candidate_id)
        if evaluation is None:
            continue
        summary = evaluation.metrics.get("sample_execution")
        if not isinstance(summary, Mapping):
            continue
        row: dict[str, Any] = {"candidate_id": candidate.candidate_id}
        count_fields = {
            "eligible": "eligible_examples",
            "attempted": "attempted_examples",
            "succeeded": "succeeded_examples",
            "failed": "failed_examples",
            "skipped": "skipped_examples",
            "retry_count": "retry_count",
            "repair_count": "repair_count",
            "exploration_failures": "exploration_failures",
            "recovered_examples": "recovered_examples",
            "input_failures": "input_failures",
        }
        for name, source_name in count_fields.items():
            value = summary.get(source_name, summary.get(name))
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                row[name] = value
        for name in ("coverage", "minimum_coverage"):
            value = summary.get(name)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                row[name] = float(value)
        coverage_pass = summary.get("coverage_pass")
        if isinstance(coverage_pass, bool):
            row["coverage_pass"] = coverage_pass
        raw_counts = summary.get("failure_counts")
        if isinstance(raw_counts, Mapping):
            row["failure_counts"] = {
                str(name)[:100]: int(value)
                for name, value in list(raw_counts.items())[:32]
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            }
        for source_name in (
            "critic_outcome_counts",
            "reason_code_counts",
            "recovered_by_failure_class",
        ):
            counts = _bounded_count_mapping(summary.get(source_name))
            if counts:
                row[source_name] = counts
        repair_outcomes = _bounded_repair_tool_outcomes(
            summary.get("repair_tool_outcomes")
        )
        if repair_outcomes:
            row["repair_tool_outcomes"] = repair_outcomes
        tool_performance = _bounded_tool_performance(
            summary.get("tool_performance")
        )
        if tool_performance:
            row["tool_performance"] = tool_performance
        if (
            int(row.get("failed", 0)) > 0
            or int(row.get("skipped", 0)) > 0
            or int(row.get("repair_count", 0)) > 0
            or row.get("coverage_pass") is False
            or bool(tool_performance)
        ):
            rows.append(row)
    return tuple(rows[:16])


def _bounded_count_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name).strip()[:160]: int(count)
        for name, count in sorted(value.items(), key=lambda item: str(item[0]))[:32]
        if isinstance(name, str)
        and name.strip()
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
    }


def _bounded_repair_tool_outcomes(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping):
        return {}
    allowed_tools = {
        "bounded-projection-repair",
        "bounded-persistence-fallback",
    }
    allowed_statuses = {"completed", "failed", "rejected"}
    result: dict[str, dict[str, int]] = {}
    for tool_id in sorted(allowed_tools):
        outcomes = value.get(tool_id)
        if not isinstance(outcomes, Mapping):
            continue
        projected = {
            status: int(count)
            for status, count in sorted(outcomes.items())
            if status in allowed_statuses
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
        }
        if projected:
            result[tool_id] = projected
    return result


def _bounded_tool_performance(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    text_fields = {"tool_id", "version", "target"}
    count_fields = {
        "horizon_hours",
        "selected",
        "completed",
        "failed",
        "rejected",
        "critic_accept",
        "critic_repair",
        "critic_failed",
        "final_accept",
        "recovered",
        "n",
    }
    metric_fields = {
        "mae",
        "rmse",
        "baseline_mae",
        "baseline_rmse",
        "rmse_improvement",
        "skill_score",
    }
    result: list[dict[str, Any]] = []
    for raw in value[:64]:
        if not isinstance(raw, Mapping):
            continue
        row: dict[str, Any] = {}
        for name in text_fields:
            item = raw.get(name)
            if isinstance(item, str) and item.strip():
                row[name] = item.strip()[:160]
        for name in count_fields:
            item = raw.get(name)
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                row[name] = min(item, 1_000_000_000)
        for name in metric_fields:
            item = raw.get(name)
            if (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
            ):
                row[name] = float(item)
        if all(name in row for name in ("tool_id", "version", "target", "horizon_hours")):
            result.append(row)
    return result


def _experience_text(value: Any, *, maximum: int = 160) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:maximum]


def _experience_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def _experience_parameter_set(value: Any) -> dict[str, int | float | bool]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int | float | bool] = {}
    for raw_name, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
        if len(result) >= 12:
            break
        name = _experience_text(raw_name, maximum=100)
        if name is None or name.casefold() in _EXPERIENCE_BLOCKED_PARAMETER_NAMES:
            continue
        if isinstance(raw_value, bool):
            result[name] = raw_value
        else:
            number = _experience_number(raw_value)
            if number is not None:
                result[name] = number
    return result


def _experience_plan_sources(
    state: Any,
    generation: int,
) -> tuple[dict[int, Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    plans: dict[int, Mapping[str, Any]] = {}
    adoptions: dict[int, Mapping[str, Any]] = {}
    for iteration in sorted(
        getattr(state, "research_iterations", ()),
        key=lambda item: item.generation,
    ):
        if iteration.generation >= generation:
            continue
        if isinstance(iteration.plan, Mapping):
            plans[iteration.generation] = iteration.plan
        if isinstance(iteration.prediction_model_adoption, Mapping):
            adoptions[iteration.generation] = iteration.prediction_model_adoption
    proposals = sorted(
        getattr(state, "proposals", ()),
        key=lambda item: (item.generation, item.created_at, item.proposal_id),
    )
    for proposal in proposals:
        if proposal.generation >= generation or proposal.generation in plans:
            continue
        raw_plan = proposal.metadata.get("plan")
        if isinstance(raw_plan, Mapping):
            plans[proposal.generation] = raw_plan
        raw_adoption = proposal.metadata.get("prediction_model_adoption")
        if isinstance(raw_adoption, Mapping):
            adoptions[proposal.generation] = raw_adoption
    return plans, adoptions


def _experience_plan_changes(
    plan: Mapping[str, Any] | None,
    previous_plan: Mapping[str, Any] | None,
) -> list[str]:
    if plan is None:
        return []
    previous = previous_plan or {}
    changed = []
    for field_name in _EXPERIENCE_PLAN_FIELDS:
        if field_name not in plan and field_name not in previous:
            continue
        if canonical_json(plan.get(field_name)) != canonical_json(
            previous.get(field_name)
        ):
            changed.append(field_name)
    return changed


def _experience_parameter_sets(
    state: Any,
    analysis: GenerationAnalysis,
) -> list[dict[str, int | float | bool]]:
    raw_sets: list[Any] = [row.get("parameters") for row in analysis.ranking]
    if not any(isinstance(item, Mapping) for item in raw_sets):
        raw_sets.extend(
            proposal.changes
            for proposal in sorted(
                getattr(state, "proposals", ()),
                key=lambda item: (item.created_at, item.proposal_id),
            )
            if proposal.generation == analysis.generation
        )
    result: list[dict[str, int | float | bool]] = []
    seen: set[str] = set()
    for raw in raw_sets:
        projected = _experience_parameter_set(raw)
        if not projected:
            continue
        encoded = canonical_json(projected)
        if encoded in seen:
            continue
        seen.add(encoded)
        result.append(projected)
        if len(result) >= 3:
            break
    return result


def _experience_modifications(
    state: Any,
    analysis: GenerationAnalysis,
    plans: Mapping[int, Mapping[str, Any]],
    adoptions: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    generation = analysis.generation
    plan = plans.get(generation)
    previous_generations = sorted(item for item in plans if item < generation)
    previous_plan = plans.get(previous_generations[-1]) if previous_generations else None
    adoption = adoptions.get(generation, {})
    previous_adoption = (
        adoptions.get(previous_generations[-1], {}) if previous_generations else {}
    )
    blueprint = plan.get("algorithm_blueprint") if isinstance(plan, Mapping) else None
    synthesis = plan.get("algorithm_synthesis") if isinstance(plan, Mapping) else None
    previous_blueprint = (
        previous_plan.get("algorithm_blueprint")
        if isinstance(previous_plan, Mapping)
        else None
    )
    previous_synthesis = (
        previous_plan.get("algorithm_synthesis")
        if isinstance(previous_plan, Mapping)
        else None
    )
    parameter_sets = _experience_parameter_sets(state, analysis)
    parameter_names = sorted(
        {name for parameter_set in parameter_sets for name in parameter_set}
    )[:16]
    requested_id = _experience_text(adoption.get("requested_id"), maximum=200)
    adopted_id = _experience_text(adoption.get("adopted_id"), maximum=200)
    if adopted_id is None and isinstance(plan, Mapping):
        prediction_model = plan.get("prediction_model")
        if isinstance(prediction_model, Mapping):
            adopted_id = _experience_text(prediction_model.get("id"), maximum=200)
    previous_adopted_id = _experience_text(
        previous_adoption.get("adopted_id"), maximum=200
    )
    result: dict[str, Any] = {
        "research_plan_changed_fields": _experience_plan_changes(
            plan, previous_plan
        ),
        "parameter_names_modified": parameter_names,
        "candidate_parameter_sets": parameter_sets,
        "algorithm_changed": bool(
            previous_adopted_id
            and adopted_id
            and previous_adopted_id != adopted_id
        ),
    }
    for name, value in (
        ("requested_predictor_id", requested_id),
        ("adopted_predictor_id", adopted_id),
        ("adoption_status", _experience_text(adoption.get("status"), maximum=100)),
        ("adoption_reason", _experience_text(adoption.get("reason"), maximum=200)),
    ):
        if value is not None:
            result[name] = value
    if isinstance(blueprint, Mapping):
        pipeline_id = _experience_text(blueprint.get("pipeline_id"), maximum=200)
        if pipeline_id is not None:
            result["algorithm_blueprint_pipeline_id"] = pipeline_id
        result["algorithm_blueprint_status"] = (
            "inherited"
            if isinstance(previous_blueprint, Mapping)
            and canonical_json(blueprint) == canonical_json(previous_blueprint)
            else "changed"
        )
    else:
        result["algorithm_blueprint_status"] = "none"
    if isinstance(synthesis, Mapping):
        result["algorithm_synthesis_digest"] = digest(dict(synthesis))
        pipeline_id = _experience_text(synthesis.get("pipeline_id"), maximum=200)
        if pipeline_id is not None:
            result["algorithm_synthesis_pipeline_id"] = pipeline_id
        raw_focus = synthesis.get("parameter_focus")
        if isinstance(raw_focus, (list, tuple)):
            result["algorithm_synthesis_parameter_focus"] = [
                value
                for item in raw_focus[:16]
                if (value := _experience_text(item, maximum=100)) is not None
                and value.casefold() not in _EXPERIENCE_BLOCKED_PARAMETER_NAMES
            ]
        raw_refs = synthesis.get("evidence_refs")
        if isinstance(raw_refs, (list, tuple)):
            result["algorithm_synthesis_evidence_ref_count"] = min(
                len(raw_refs), 16
            )
        result["algorithm_synthesis_status"] = (
            "inherited"
            if isinstance(previous_synthesis, Mapping)
            and canonical_json(synthesis) == canonical_json(previous_synthesis)
            else "changed"
        )
    else:
        result["algorithm_synthesis_status"] = "none"
    return result


def _experience_algorithm_failures(
    analysis: GenerationAnalysis,
) -> list[dict[str, Any]]:
    result = []
    for row in analysis.algorithm_failures[:8]:
        failure_code = _experience_text(row.get("failure_code"), maximum=120)
        if failure_code is None:
            continue
        item: dict[str, Any] = {"failure_code": failure_code}
        for name in ("phase", "stage", "error_type"):
            value = _experience_text(row.get(name), maximum=120)
            if value is not None:
                item[name] = value
        retryable = row.get("retryable")
        if isinstance(retryable, bool):
            item["retryable"] = retryable
        count = _experience_number(row.get("attempt_count", row.get("count")))
        if count is not None and count >= 0:
            item["attempt_count"] = int(count)
        result.append(item)
    return result


def _experience_sample_failure_counts(
    analysis: GenerationAnalysis,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in analysis.sample_failures[:16]:
        raw_counts = row.get("failure_counts")
        found = False
        if isinstance(raw_counts, Mapping):
            for raw_name, raw_count in sorted(
                raw_counts.items(), key=lambda item: str(item[0])
            )[:16]:
                name = _experience_text(raw_name, maximum=120)
                if (
                    name is not None
                    and isinstance(raw_count, int)
                    and not isinstance(raw_count, bool)
                    and raw_count > 0
                ):
                    counts[name] += raw_count
                    found = True
        failed = row.get("failed")
        if (
            not found
            and isinstance(failed, int)
            and not isinstance(failed, bool)
            and failed > 0
        ):
            counts["unclassified_sample_failure"] += failed
    return dict(sorted(counts.items())[:16])


def _experience_weaknesses(
    rows: tuple[Mapping[str, Any], ...],
    *,
    kind: str,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        score = _experience_number(row.get("median_skill_score"))
        reward = _experience_number(
            row.get("median_normalized_mean_reward")
        )
        weak_skill = score is not None and float(score) < 0.0
        weak_reward = reward is not None and float(reward) < 0.0
        if not weak_skill and not weak_reward:
            continue
        item: dict[str, Any] = {
            "weakness_basis": (
                "skill_and_mean_reward"
                if weak_skill and weak_reward
                else "skill"
                if weak_skill
                else "mean_reward"
            )
        }
        if score is not None:
            item["median_skill_score"] = score
        if reward is not None:
            item["median_normalized_mean_reward"] = reward
        if kind == "target":
            target = _experience_text(row.get("target"), maximum=120)
            if target is None:
                continue
            item["target"] = target
            unit = _experience_text(row.get("unit"), maximum=80)
            if unit is not None:
                item["unit"] = unit
        horizon = _experience_number(row.get("horizon_hours"))
        if kind == "horizon" and horizon is None:
            continue
        if horizon is not None:
            item["horizon_hours"] = horizon
        evidence_count = _experience_number(row.get("evidence_count"))
        if evidence_count is not None and evidence_count >= 0:
            item["evidence_count"] = int(evidence_count)
        reward_evidence_count = _experience_number(
            row.get("reward_evidence_count")
        )
        if reward_evidence_count is not None and reward_evidence_count >= 0:
            item["reward_evidence_count"] = int(reward_evidence_count)
        result.append(item)
        if len(result) >= _EXPERIENCE_MAX_WEAKNESSES:
            break
    return result


def _experience_repair_effectiveness(
    analysis: GenerationAnalysis,
) -> dict[str, Any]:
    attempted = 0
    completed = 0
    failed = 0
    reported_repairs = 0
    recovered = 0
    for row in analysis.sample_failures[:16]:
        for field_name, accumulator in (
            ("repair_count", "reported"),
            ("recovered_examples", "recovered"),
        ):
            value = row.get(field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                continue
            if accumulator == "reported":
                reported_repairs += value
            else:
                recovered += value
        outcomes = row.get("repair_tool_outcomes")
        if not isinstance(outcomes, Mapping):
            continue
        for tool_outcomes in list(outcomes.values())[:4]:
            if not isinstance(tool_outcomes, Mapping):
                continue
            for status in ("completed", "failed", "rejected"):
                count = tool_outcomes.get(status)
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    continue
                attempted += count
                if status == "completed":
                    completed += count
                else:
                    failed += count
    result: dict[str, Any] = {
        "attempted": attempted,
        "completed": completed,
        "failed_or_rejected": failed,
        "reported_repairs": reported_repairs,
        "recovered_examples": recovered,
        "effective": bool(recovered > 0 or completed > 0),
    }
    if attempted:
        result["completion_rate"] = completed / attempted
    return result


def _experience_batch_highest_observed_score(
    analysis: GenerationAnalysis,
) -> float | None:
    scores = [
        float(score)
        for row in analysis.ranking
        if (score := _experience_number(row.get("score"))) is not None
    ]
    return max(scores) if scores else None


def _experience_comparison_candidate(
    analysis: GenerationAnalysis,
) -> tuple[str, str, float] | None:
    """Return the formal candidate whose score may represent this generation."""

    if analysis.champion_candidate_id is not None:
        candidate_id = analysis.champion_candidate_id
        role = "champion"
    elif analysis.selected_candidate_id is not None:
        candidate_id = analysis.selected_candidate_id
        role = "selected"
    else:
        return None
    for row in analysis.ranking:
        if row.get("candidate_id") != candidate_id:
            continue
        score = _experience_number(row.get("score"))
        return (candidate_id, role, float(score)) if score is not None else None
    return None


def _experience_generation_cohort_digest(
    analysis: GenerationAnalysis,
) -> str | None:
    """Return the one verifiable score cohort represented by a generation."""

    cohort_digests: set[str] = set()
    for row in analysis.ranking:
        if _experience_number(row.get("score")) is None:
            continue
        value = row.get("evaluation_cohort_digest")
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            return None
        cohort_digests.add(value)
    return next(iter(cohort_digests)) if len(cohort_digests) == 1 else None


def _experience_score_comparison(
    state: Any,
    analysis: GenerationAnalysis,
    previous_analysis: GenerationAnalysis | None,
) -> str:
    """Describe whether adjacent generation scores support an improvement claim."""

    if previous_analysis is None:
        return "no_previous_generation"
    if not sample_update_windows_enabled(getattr(state, "task_manifest", None)):
        return "legacy_full_cohort"
    cohort_digest = _experience_generation_cohort_digest(analysis)
    previous_cohort_digest = _experience_generation_cohort_digest(previous_analysis)
    if cohort_digest is None or previous_cohort_digest is None:
        return "cohort_digest_unavailable"
    if cohort_digest == previous_cohort_digest:
        return "same_cohort"
    return "different_cohort_not_compared"


def _experience_synthesis_effect(
    state: Any,
    analysis: GenerationAnalysis,
    modifications: Mapping[str, Any],
    *,
    score_comparison: str,
    comparison_candidate: tuple[str, str, float] | None,
    score_delta: float | None,
    observed_improvement: bool,
) -> dict[str, Any] | None:
    synthesis_digest = _experience_text(
        modifications.get("algorithm_synthesis_digest"), maximum=200
    )
    if synthesis_digest is None:
        return None
    proposal_ids = set()
    for proposal in getattr(state, "proposals", ()):
        if proposal.generation != analysis.generation:
            continue
        plan = proposal.metadata.get("plan")
        synthesis = (
            plan.get("algorithm_synthesis") if isinstance(plan, Mapping) else None
        )
        if isinstance(synthesis, Mapping) and digest(dict(synthesis)) == synthesis_digest:
            proposal_ids.add(proposal.proposal_id)
    attempts = [
        attempt
        for attempt in getattr(state, "algorithm_attempts", ())
        if attempt.generation == analysis.generation
        and (not proposal_ids or attempt.proposal_id in proposal_ids)
    ]

    def lowered_with_synthesis(attempt: Any) -> bool:
        spec = attempt.algorithm_spec
        if not isinstance(spec, Mapping):
            return False
        algorithm_ir = spec.get("algorithm_ir")
        return isinstance(algorithm_ir, Mapping) and (
            algorithm_ir.get("source_synthesis_digest") == synthesis_digest
        )

    lowered_candidates = {
        attempt.candidate_id for attempt in attempts if lowered_with_synthesis(attempt)
    }
    compile_passed = {
        attempt.candidate_id
        for attempt in attempts
        if attempt.phase == "compile"
        and attempt.status == "passed"
        and lowered_with_synthesis(attempt)
    }
    debug_passed = {
        attempt.candidate_id
        for attempt in attempts
        if attempt.phase == "debug"
        and attempt.status == "passed"
        and lowered_with_synthesis(attempt)
    }
    evaluated_count = sum(row.get("score") is not None for row in analysis.ranking)
    batch_highest_observed_score = _experience_batch_highest_observed_score(analysis)
    selected = bool(
        analysis.outcome == "promoted"
        and analysis.champion_candidate_id is not None
    )
    selection_interpretation = (
        "current_cohort_batch_champion"
        if selected and score_comparison == "different_cohort_not_compared"
        else "current_cohort_champion_unverifiable"
        if selected and score_comparison == "cohort_digest_unavailable"
        else "initial_champion"
        if selected and score_comparison == "no_previous_generation"
        else "comparable_score_improvement"
        if observed_improvement
        else "selected_without_comparable_score_baseline"
        if selected
        and score_comparison in {"legacy_full_cohort", "same_cohort"}
        and score_delta is None
        else "selected_without_observed_score_improvement"
        if selected and score_comparison in {"legacy_full_cohort", "same_cohort"}
        else "not_selected"
    )
    result: dict[str, Any] = {
        "synthesis_digest": synthesis_digest,
        "association_scope": "same_generation_frozen_plan",
        "causal_attribution": False,
        "score_comparison": score_comparison,
        "selection_interpretation": selection_interpretation,
        "lowered_candidate_count": len(lowered_candidates),
        "compile_passed_candidate_count": len(compile_passed),
        "debug_passed_candidate_count": len(debug_passed),
        "failed_algorithm_attempt_count": sum(
            attempt.status == "failed" for attempt in attempts
        ),
        "evaluated_candidate_count": evaluated_count,
        "eligible_candidate_count": analysis.eligible_count,
        "outcome": _experience_text(analysis.outcome, maximum=100) or "unknown",
        "observed_improvement": observed_improvement,
    }
    if comparison_candidate is not None:
        candidate_id, role, comparison_score = comparison_candidate
        result.update(
            {
                "comparison_candidate_id": candidate_id,
                "comparison_candidate_role": role,
                "comparison_score": comparison_score,
                # Compatibility alias: this is now explicitly the formal
                # comparison score, never an unselected batch maximum.
                "best_score": comparison_score,
            }
        )
    if batch_highest_observed_score is not None:
        result["batch_highest_observed_score"] = batch_highest_observed_score
    if score_delta is not None:
        result["best_score_delta_vs_previous_generation"] = score_delta
    return result


def _experience_generation_summary(
    state: Any,
    analysis: GenerationAnalysis,
    previous_analysis: GenerationAnalysis | None,
    plans: Mapping[int, Mapping[str, Any]],
    adoptions: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    comparison_candidate = _experience_comparison_candidate(analysis)
    previous_comparison_candidate = (
        _experience_comparison_candidate(previous_analysis)
        if previous_analysis is not None
        else None
    )
    batch_highest_observed_score = _experience_batch_highest_observed_score(analysis)
    score_comparison = _experience_score_comparison(
        state, analysis, previous_analysis
    )
    selected = bool(
        analysis.outcome == "promoted"
        and analysis.champion_candidate_id is not None
    )
    score_delta = (
        comparison_candidate[2] - previous_comparison_candidate[2]
        if comparison_candidate is not None
        and previous_comparison_candidate is not None
        and score_comparison in {"legacy_full_cohort", "same_cohort"}
        else None
    )
    observed_improvement = bool(
        selected
        and score_delta is not None
        and score_delta > _IMPROVEMENT_TOLERANCE
    )
    modifications = _experience_modifications(state, analysis, plans, adoptions)
    result: dict[str, Any] = {
        "generation": analysis.generation,
        "score_comparison": score_comparison,
        "modifications": modifications,
        "algorithm_failures": _experience_algorithm_failures(analysis),
        "sample_failure_counts": _experience_sample_failure_counts(analysis),
        "weak_targets": _experience_weaknesses(
            analysis.target_weaknesses, kind="target"
        ),
        "weak_horizons": _experience_weaknesses(
            analysis.horizon_weaknesses, kind="horizon"
        ),
        "repair_effectiveness": _experience_repair_effectiveness(analysis),
        "tool_performance": _experience_tool_performance(analysis),
        "outcome": _experience_text(analysis.outcome, maximum=100) or "unknown",
        "improved": observed_improvement,
        "common_failures": [
            value
            for item in analysis.common_failures[:6]
            if (value := _experience_text(item, maximum=120)) is not None
        ],
    }
    if comparison_candidate is not None:
        candidate_id, role, comparison_score = comparison_candidate
        result.update(
            {
                "comparison_candidate_id": candidate_id,
                "comparison_candidate_role": role,
                "comparison_score": comparison_score,
                "best_score": comparison_score,
            }
        )
    if batch_highest_observed_score is not None:
        result["batch_highest_observed_score"] = batch_highest_observed_score
    cohort_digest = _experience_generation_cohort_digest(analysis)
    if cohort_digest is not None:
        result["evaluation_cohort_digest"] = cohort_digest
    if score_delta is not None:
        result["best_score_delta_vs_previous_generation"] = score_delta
    synthesis_effect = _experience_synthesis_effect(
        state,
        analysis,
        modifications,
        score_comparison=score_comparison,
        comparison_candidate=comparison_candidate,
        score_delta=score_delta,
        observed_improvement=observed_improvement,
    )
    if synthesis_effect is not None:
        result["algorithm_synthesis_effect"] = synthesis_effect
    return result


def _experience_tool_performance(
    analysis: GenerationAnalysis,
) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    count_fields = (
        "selected",
        "completed",
        "failed",
        "rejected",
        "critic_accept",
        "critic_repair",
        "critic_failed",
        "final_accept",
        "recovered",
        "n",
    )
    for candidate in analysis.ranking:
        rows = candidate.get("tool_performance")
        if not isinstance(rows, (list, tuple)):
            continue
        for raw in rows[:64]:
            if not isinstance(raw, Mapping):
                continue
            tool_id = _experience_text(raw.get("tool_id"), maximum=160)
            version = _experience_text(raw.get("version"), maximum=80)
            target = _experience_text(raw.get("target"), maximum=120)
            horizon = raw.get("horizon_hours")
            if (
                tool_id is None
                or version is None
                or target is None
                or isinstance(horizon, bool)
                or not isinstance(horizon, int)
                or horizon < 0
            ):
                continue
            key = (tool_id, version, target, horizon)
            aggregate = aggregates.setdefault(
                key,
                {
                    "tool_id": tool_id,
                    "version": version,
                    "target": target,
                    "horizon_hours": horizon,
                    **{name: 0 for name in count_fields},
                    "_absolute_error": 0.0,
                    "_squared_error": 0.0,
                    "_baseline_absolute_error": 0.0,
                    "_baseline_squared_error": 0.0,
                },
            )
            for name in count_fields:
                value = raw.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    aggregate[name] += value
            n = raw.get("n")
            if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
                continue
            for source, destination, squared in (
                ("mae", "_absolute_error", False),
                ("rmse", "_squared_error", True),
                ("baseline_mae", "_baseline_absolute_error", False),
                ("baseline_rmse", "_baseline_squared_error", True),
            ):
                value = _experience_number(raw.get(source))
                if value is None:
                    continue
                aggregate[destination] += (
                    float(value) * float(value) * n
                    if squared
                    else float(value) * n
                )

    result: list[dict[str, Any]] = []
    for aggregate in sorted(
        aggregates.values(),
        key=lambda item: (
            -int(item["final_accept"]),
            str(item["tool_id"]),
            str(item["target"]),
            int(item["horizon_hours"]),
        ),
    )[:32]:
        public = {
            name: value
            for name, value in aggregate.items()
            if not name.startswith("_")
        }
        n = int(aggregate["n"])
        if n:
            mae = float(aggregate["_absolute_error"]) / n
            rmse = math.sqrt(float(aggregate["_squared_error"]) / n)
            baseline_mae = float(aggregate["_baseline_absolute_error"]) / n
            baseline_rmse = math.sqrt(
                float(aggregate["_baseline_squared_error"]) / n
            )
            public.update(
                {
                    "mae": mae,
                    "rmse": rmse,
                    "baseline_mae": baseline_mae,
                    "baseline_rmse": baseline_rmse,
                    "rmse_improvement": baseline_rmse - rmse,
                    "skill_score": (
                        1.0 - rmse / baseline_rmse
                        if baseline_rmse > 1e-15
                        else 0.0
                    ),
                }
            )
        result.append(public)
    return result


def _experience_target_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    target = _experience_text(row.get("target"), maximum=120)
    if target is not None:
        result["target"] = target
    horizon = _experience_number(row.get("horizon_hours"))
    if horizon is not None:
        result["horizon_hours"] = horizon
    return result


def _experience_has_resolution_evidence(
    analyses: list[GenerationAnalysis],
    *,
    kind: str,
    identity: Mapping[str, Any],
    after_generation: int,
) -> int | None:
    for analysis in analyses:
        if analysis.generation <= after_generation:
            continue
        if kind == "algorithm_failure" and analysis.candidate_count > 0:
            return analysis.generation
        if kind == "sample_failure" and any(
            row.get("score") is not None for row in analysis.ranking
        ):
            return analysis.generation
        if kind == "weak_target" and any(
            _experience_target_identity(row) == dict(identity)
            for row in analysis.target_weaknesses
        ):
            return analysis.generation
        if kind == "weak_horizon" and any(
            _experience_number(row.get("horizon_hours"))
            == identity.get("horizon_hours")
            for row in analysis.horizon_weaknesses
        ):
            return analysis.generation
    return None


def _experience_issue_records(
    analyses: list[GenerationAnalysis],
    summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: dict[tuple[str, str], dict[str, Any]] = {}

    def record(
        *,
        kind: str,
        identity: Mapping[str, Any],
        generation: int,
        observation: Mapping[str, Any],
    ) -> None:
        encoded_identity = canonical_json(dict(identity))
        key = (kind, encoded_identity)
        existing = issues.get(key)
        if existing is None:
            existing = {
                "issue_id": digest({"kind": kind, "identity": dict(identity)})[:16],
                "kind": kind,
                "identity": dict(identity),
                "first_seen_generation": generation,
                "last_seen_generation": generation,
                "occurrence_generations": [generation],
                "latest_observation": dict(observation),
            }
            issues[key] = existing
            return
        existing["last_seen_generation"] = generation
        occurrences = list(existing["occurrence_generations"])
        if generation not in occurrences:
            occurrences.append(generation)
        existing["occurrence_generations"] = occurrences[-6:]
        existing["latest_observation"] = dict(observation)

    for summary in summaries:
        generation = int(summary["generation"])
        for row in summary["algorithm_failures"]:
            identity = {
                name: row[name]
                for name in ("phase", "stage", "failure_code")
                if name in row
            }
            record(
                kind="algorithm_failure",
                identity=identity,
                generation=generation,
                observation=row,
            )
        for failure_code, count in summary["sample_failure_counts"].items():
            record(
                kind="sample_failure",
                identity={"failure_code": failure_code},
                generation=generation,
                observation={"count": count},
            )
        for row in summary["weak_targets"]:
            record(
                kind="weak_target",
                identity=_experience_target_identity(row),
                generation=generation,
                observation=row,
            )
        for row in summary["weak_horizons"]:
            record(
                kind="weak_horizon",
                identity={"horizon_hours": row.get("horizon_hours")},
                generation=generation,
                observation=row,
            )

    active: list[dict[str, Any]] = []
    archived: list[dict[str, Any]] = []
    for item in issues.values():
        resolved_generation = _experience_has_resolution_evidence(
            analyses,
            kind=str(item["kind"]),
            identity=item["identity"],
            after_generation=int(item["last_seen_generation"]),
        )
        if resolved_generation is None:
            item["status"] = "active_unresolved"
            active.append(item)
        else:
            item["status"] = "resolved_archived"
            item["resolved_generation"] = resolved_generation
            archived.append(item)
    active.sort(
        key=lambda item: (
            -int(item["last_seen_generation"]),
            str(item["kind"]),
            str(item["issue_id"]),
        )
    )
    archived.sort(
        key=lambda item: (
            -int(item["resolved_generation"]),
            -int(item["last_seen_generation"]),
            str(item["kind"]),
            str(item["issue_id"]),
        )
    )
    return active, archived


def _history_task_identity(state: Any) -> dict[str, str] | None:
    task = getattr(state, "task_manifest", None)
    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    required = {
        "dataset_digest": metadata.get("dataset_digest"),
        "split_manifest_digest": metadata.get("split_manifest_digest"),
        "evaluation_partition": metadata.get("evaluation_partition"),
        "scientific_scope": metadata.get("scientific_scope"),
        "evaluator_id": metadata.get("evaluator_id"),
        "evaluator_digest": metadata.get("evaluator_digest"),
    }
    if any(
        not isinstance(value, str) or not value.strip()
        for value in required.values()
    ):
        return None
    return {name: str(value).strip() for name, value in required.items()}


def _history_evaluator_series(evaluator_id: str) -> str:
    return evaluator_id.rsplit("@", 1)[0].casefold()


def _history_is_toy_state(state: Any, identity: Mapping[str, str]) -> bool:
    task = getattr(state, "task_manifest", None)
    metadata = getattr(task, "metadata", {})
    domain = str(metadata.get("domain") or "").strip().casefold()
    domain_pack = str(getattr(task, "domain_pack", "")).strip().casefold()
    evaluator_id = identity["evaluator_id"].casefold()
    datasets = tuple(
        str(item).strip().casefold()
        for item in getattr(task, "visible_datasets", ())
    )
    execution_mode = str(
        metadata.get("execution_mode")
        or metadata.get("fit_method")
        or ""
    ).casefold()
    return bool(
        domain == "toy"
        or "@toy" in domain_pack
        or evaluator_id.startswith("toy_")
        or any("generated-toy" in item for item in datasets)
        or execution_mode == "toy_score"
    )


def _history_compatibility_scope(
    current: Mapping[str, str],
    historical: Mapping[str, str],
) -> str | None:
    for field_name in (
        "dataset_digest",
        "split_manifest_digest",
        "evaluation_partition",
        "scientific_scope",
    ):
        if current[field_name] != historical[field_name]:
            return None
    # A new run has not evaluated its first fixed update cohort yet. Even an
    # identical evaluator therefore supplies directional, non-causal lessons,
    # never a directly comparable score or improvement delta.
    if _history_evaluator_series(current["evaluator_id"]) == (
        _history_evaluator_series(historical["evaluator_id"])
    ):
        return "directional_only"
    return None


def _history_analysis_events(
    state: Any,
    *,
    history_cutoff_seq: int | None,
) -> dict[int, int]:
    analyses = {
        item.generation: item
        for item in getattr(state, "generation_analyses", ())
    }
    result: dict[int, int] = {}
    for event in getattr(state, "events", ()):
        if getattr(event, "kind", None) != "GenerationAnalyzed":
            continue
        seq = getattr(event, "seq", None)
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            continue
        if history_cutoff_seq is not None and seq >= history_cutoff_seq:
            continue
        payload = getattr(event, "payload", None)
        raw_analysis = payload.get("analysis") if isinstance(payload, Mapping) else None
        if not isinstance(raw_analysis, Mapping):
            continue
        raw_generation = raw_analysis.get("generation")
        if (
            isinstance(raw_generation, bool)
            or not isinstance(raw_generation, int)
            or raw_generation < 0
        ):
            continue
        analysis = analyses.get(raw_generation)
        if analysis is None:
            continue
        stored_digest = raw_analysis.get("analysis_digest")
        if (
            stored_digest is not None
            and stored_digest != analysis.analysis_digest
        ):
            continue
        result[raw_generation] = max(seq, result.get(raw_generation, -1))
    return result


def _history_without_scores(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for raw_name, item in value.items():
            name = str(raw_name)
            normalized = name.casefold()
            if (
                "score" in normalized
                or "delta" in normalized
                or normalized in {"improved", "observed_improvement"}
            ):
                continue
            result[name] = _history_without_scores(item)
        return result
    if isinstance(value, list):
        return [_history_without_scores(item) for item in value]
    return value


def _history_generation_projection(
    summary: Mapping[str, Any],
    analysis: GenerationAnalysis,
    *,
    source_run_id: str,
    compatibility_scope: str,
    history_cutoff_seq: int | None,
) -> dict[str, Any]:
    allowed_fields = (
        "score_comparison",
        "modifications",
        "algorithm_failures",
        "sample_failure_counts",
        "weak_targets",
        "weak_horizons",
        "repair_effectiveness",
        "tool_performance",
        "outcome",
        "improved",
        "common_failures",
        "comparison_candidate_role",
        "comparison_score",
        "best_score",
        "batch_highest_observed_score",
        "evaluation_cohort_digest",
        "best_score_delta_vs_previous_generation",
        "algorithm_synthesis_effect",
    )
    projected = {
        name: json.loads(canonical_json(summary[name]))
        for name in allowed_fields
        if name in summary
    }
    synthesis_effect = projected.get("algorithm_synthesis_effect")
    if isinstance(synthesis_effect, dict):
        synthesis_effect.pop("comparison_candidate_id", None)
    modifications = projected.get("modifications")
    if isinstance(modifications, dict):
        modifications.pop("adoption_reason", None)
    projected["gate_result"] = {
        "candidate_count": analysis.candidate_count,
        "eligible_count": analysis.eligible_count,
        "outcome": analysis.outcome,
        "insufficient_evidence": analysis.insufficient_evidence,
        "constraint_failure_count": len(analysis.constraint_failures),
        "judge_disagreement_count": len(analysis.judge_disagreements),
    }
    if compatibility_scope == "directional_only":
        projected = _history_without_scores(projected)
    list_limits = {
        "algorithm_failures": 3,
        "common_failures": 3,
        "tool_performance": 4,
        "weak_horizons": 4,
        "weak_targets": 4,
    }
    for field_name, limit in list_limits.items():
        rows = projected.get(field_name)
        if isinstance(rows, list):
            projected[field_name] = rows[:limit]
    failure_counts = projected.get("sample_failure_counts")
    if isinstance(failure_counts, dict):
        projected["sample_failure_counts"] = dict(
            sorted(failure_counts.items(), key=lambda item: str(item[0]))[:8]
        )
    modifications = projected.get("modifications")
    if isinstance(modifications, dict):
        for field_name, limit in (
            ("research_plan_changed_fields", 6),
            ("parameter_names_modified", 12),
            ("candidate_parameter_sets", 2),
            ("algorithm_synthesis_parameter_focus", 8),
        ):
            rows = modifications.get(field_name)
            if isinstance(rows, list):
                modifications[field_name] = rows[:limit]
    result = {
        "source_run_id": source_run_id,
        "source_generation": analysis.generation,
        "source_analysis_digest": analysis.analysis_digest,
        "compatibility_scope": compatibility_scope,
        **projected,
    }
    if history_cutoff_seq is not None:
        result["history_cutoff_seq"] = history_cutoff_seq
    return result


def _historical_parameter_guardrail_evidence(
    analysis: GenerationAnalysis,
    *,
    source_run_id: str,
) -> list[dict[str, Any]]:
    """Project aggregate cells for conservative cross-cohort verification."""

    result: list[dict[str, Any]] = []
    for candidate in analysis.ranking[:8]:
        violations = _experience_number(candidate.get("constraint_violations"))
        raw_cohort_digest = candidate.get("evaluation_cohort_digest")
        raw_cohort_window = candidate.get("evaluation_cohort_window")
        if (
            violations is None
            or not isinstance(raw_cohort_digest, str)
            or len(raw_cohort_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in raw_cohort_digest
            )
            or not isinstance(raw_cohort_window, Mapping)
            or raw_cohort_window.get("schema_version")
            != "ecologyrsi-dsh.feedback-update-cohort/1"
            or raw_cohort_window.get("selection_policy")
            != "target_horizon_interleaved_rotating_window@1"
            or raw_cohort_window.get("evaluation_cohort_digest")
            != raw_cohort_digest
        ):
            continue
        feedback_cohort_digest = raw_cohort_window.get(
            "feedback_update_cohort_digest"
        )
        population_count = raw_cohort_window.get("population_count")
        population_digest = raw_cohort_window.get("population_digest")
        selected_count = raw_cohort_window.get("selected_count")
        window_offset = raw_cohort_window.get("window_offset")
        if (
            isinstance(population_count, bool)
            or not isinstance(population_count, int)
            or population_count <= 0
            or not isinstance(population_digest, str)
            or len(population_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in population_digest
            )
            or not isinstance(feedback_cohort_digest, str)
            or len(feedback_cohort_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in feedback_cohort_digest
            )
            or isinstance(selected_count, bool)
            or not isinstance(selected_count, int)
            or selected_count <= 0
            or selected_count > population_count
            or isinstance(window_offset, bool)
            or not isinstance(window_offset, int)
            or window_offset < 0
            or window_offset >= population_count
        ):
            continue
        parameters = candidate.get("parameters")
        target_skills = candidate.get("target_skill_scores")
        if not isinstance(parameters, Mapping) or not isinstance(
            target_skills, (list, tuple)
        ):
            continue
        for row in target_skills[:32]:
            if not isinstance(row, Mapping):
                continue
            target = _experience_text(row.get("target"), maximum=120)
            raw_horizon = row.get("horizon_hours")
            skill_score = _experience_number(row.get("skill_score"))
            cell_sample_count = row.get("n")
            if (
                target is None
                or target != target.casefold()
                or any(
                    character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                    for character in target
                )
                or isinstance(raw_horizon, bool)
                or not isinstance(raw_horizon, (int, float))
                or not math.isfinite(float(raw_horizon))
                or float(raw_horizon) <= 0
                or not float(raw_horizon).is_integer()
                or skill_score is None
                or isinstance(cell_sample_count, bool)
                or not isinstance(cell_sample_count, int)
                or cell_sample_count <= 0
                or cell_sample_count > selected_count
            ):
                continue
            horizon = int(raw_horizon)
            parameter = f"{target}_{horizon}h_residual_scale"
            value = _experience_number(parameters.get(parameter))
            if value is None:
                continue
            result.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "target": target,
                    "horizon_hours": horizon,
                    "skill_score": skill_score,
                    "cell_sample_count": cell_sample_count,
                    "constraint_violations": violations,
                    "evaluation_cohort_digest": raw_cohort_digest,
                    "cohort_window": {
                        "schema_version": raw_cohort_window["schema_version"],
                        "selection_policy": raw_cohort_window["selection_policy"],
                        "feedback_update_cohort_digest": (
                            feedback_cohort_digest
                        ),
                        "population_count": population_count,
                        "population_digest": population_digest,
                        "selected_count": selected_count,
                        "window_offset": window_offset,
                    },
                    "source_run_id": source_run_id,
                    "source_generation": analysis.generation,
                }
            )
    return result


def _cohort_window_segments(window: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    population_count = int(window["population_count"])
    selected_count = int(window["selected_count"])
    window_offset = int(window["window_offset"])
    end = window_offset + selected_count
    if end <= population_count:
        return ((window_offset, end),)
    return ((window_offset, population_count), (0, end - population_count))


def _cohort_windows_are_pairwise_non_overlapping(
    cohorts: Sequence[Mapping[str, Any]],
) -> bool:
    if not cohorts:
        return False
    population_identities = {
        (str(item["population_digest"]), int(item["population_count"]))
        for item in cohorts
    }
    if len(population_identities) != 1:
        return False
    for index, left in enumerate(cohorts):
        left_segments = _cohort_window_segments(left)
        for right in cohorts[index + 1 :]:
            if any(
                max(left_start, right_start) < min(left_end, right_end)
                for left_start, left_end in left_segments
                for right_start, right_end in _cohort_window_segments(right)
            ):
                return False
    return True


def _historical_experience_payload(
    state: Any,
    historical_states: Sequence[Any],
    *,
    history_cutoff_seq: int | None,
) -> dict[str, Any]:
    current_identity = _history_task_identity(state)
    current_run_id = str(getattr(getattr(state, "run", None), "run_id", ""))
    candidate_states: list[tuple[int, str, Any]] = []
    if current_identity is not None and not _history_is_toy_state(
        state, current_identity
    ):
        for historical_state in historical_states:
            historical_run = getattr(historical_state, "run", None)
            source_run_id = str(getattr(historical_run, "run_id", ""))
            events = tuple(getattr(historical_state, "events", ()))
            created_seq = (
                getattr(events[0], "seq", None)
                if events and getattr(events[0], "kind", None) == "RunCreated"
                else None
            )
            if (
                not source_run_id
                or source_run_id == current_run_id
                or isinstance(created_seq, bool)
                or not isinstance(created_seq, int)
                or (
                    history_cutoff_seq is not None
                    and created_seq >= history_cutoff_seq
                )
            ):
                continue
            candidate_states.append((created_seq, source_run_id, historical_state))

    candidate_states.sort(key=lambda item: (-item[0], item[1]))
    scanned_candidates = candidate_states[:_HISTORICAL_EXPERIENCE_SCAN_RUNS]
    scanned_states: list[tuple[int, str, Any, str, bool]] = []
    for created_seq, source_run_id, historical_state in scanned_candidates:
        historical_identity = _history_task_identity(historical_state)
        if historical_identity is None or _history_is_toy_state(
            historical_state, historical_identity
        ):
            continue
        scope = _history_compatibility_scope(current_identity, historical_identity)
        if scope is None:
            continue
        guardrail_compatible = bool(
            current_identity["evaluator_id"] == historical_identity["evaluator_id"]
            and current_identity["evaluator_digest"]
            == historical_identity["evaluator_digest"]
        )
        scanned_states.append(
            (
                created_seq,
                source_run_id,
                historical_state,
                scope,
                guardrail_compatible,
            )
        )
    records: list[
        tuple[int, str, int, dict[str, Any], list[dict[str, Any]]]
    ] = []
    for (
        _created_seq,
        source_run_id,
        historical_state,
        scope,
        guardrail_compatible,
    ) in scanned_states:
        event_seqs = _history_analysis_events(
            historical_state,
            history_cutoff_seq=history_cutoff_seq,
        )
        analyses = sorted(
            (
                item
                for item in getattr(historical_state, "generation_analyses", ())
                if item.generation in event_seqs
            ),
            key=lambda item: item.generation,
        )
        if not analyses:
            continue
        plans, adoptions = _experience_plan_sources(
            historical_state, analyses[-1].generation + 1
        )
        previous_analysis: GenerationAnalysis | None = None
        for analysis in analyses:
            summary = _experience_generation_summary(
                historical_state,
                analysis,
                previous_analysis,
                plans,
                adoptions,
            )
            records.append(
                (
                    event_seqs[analysis.generation],
                    source_run_id,
                    analysis.generation,
                    _history_generation_projection(
                        summary,
                        analysis,
                        source_run_id=source_run_id,
                        compatibility_scope=scope,
                        history_cutoff_seq=history_cutoff_seq,
                    ),
                    (
                        _historical_parameter_guardrail_evidence(
                            analysis,
                            source_run_id=source_run_id,
                        )
                        if guardrail_compatible
                        else []
                    ),
                )
            )
            previous_analysis = analysis

    records.sort(key=lambda item: (-item[0], item[1], item[2]))
    records_by_run: dict[
        str,
        list[tuple[int, str, int, dict[str, Any], list[dict[str, Any]]]],
    ] = defaultdict(list)
    for record in records:
        records_by_run[record[1]].append(record)
    run_order = sorted(
        records_by_run,
        key=lambda run_id: (-records_by_run[run_id][0][0], run_id),
    )[:_HISTORICAL_EXPERIENCE_MAX_RUNS]
    selected_records: list[
        tuple[int, str, int, dict[str, Any], list[dict[str, Any]]]
    ] = []
    generation_depth = 0
    while len(selected_records) < _HISTORICAL_EXPERIENCE_MAX_GENERATIONS:
        added = False
        for run_id in run_order:
            run_records = records_by_run[run_id]
            if generation_depth >= len(run_records):
                continue
            selected_records.append(run_records[generation_depth])
            added = True
            if len(selected_records) >= _HISTORICAL_EXPERIENCE_MAX_GENERATIONS:
                break
        if not added:
            break
        generation_depth += 1
    selected = [record[3] for record in selected_records]

    available_count = len(records)
    omitted_detail_count = 0

    def historical_parameter_guardrails() -> dict[str, Any]:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for record in selected_records:
            for evidence in record[4]:
                parameter = str(evidence["parameter"])
                value_key = canonical_json(evidence["value"])
                group = grouped.setdefault(
                    (parameter, value_key),
                    {
                        "representative": evidence,
                        "cohorts": {},
                        "disqualified": False,
                    },
                )
                skill_score = float(evidence["skill_score"])
                constraint_violations = float(evidence["constraint_violations"])
                if skill_score < 0 or constraint_violations != 0:
                    group["disqualified"] = True
                cohort_digest = str(evidence["evaluation_cohort_digest"])
                cohort = group["cohorts"].get(cohort_digest)
                if cohort is None:
                    cohort_window = evidence["cohort_window"]
                    group["cohorts"][cohort_digest] = {
                        "evaluation_cohort_digest": cohort_digest,
                        "cell_sample_count": int(evidence["cell_sample_count"]),
                        "minimum_skill_score": skill_score,
                        "schema_version": cohort_window["schema_version"],
                        "selection_policy": cohort_window["selection_policy"],
                        "feedback_update_cohort_digest": cohort_window[
                            "feedback_update_cohort_digest"
                        ],
                        "population_count": cohort_window["population_count"],
                        "population_digest": cohort_window["population_digest"],
                        "selected_count": cohort_window["selected_count"],
                        "window_offset": cohort_window["window_offset"],
                        "source_run_id": str(evidence["source_run_id"]),
                        "source_generation": int(evidence["source_generation"]),
                    }
                    continue
                # A cohort digest binds one scored row set. Repeated generations
                # on that same cohort are correlated evidence, so count the cell
                # once and retain the most conservative observed statistics.
                cohort["cell_sample_count"] = min(
                    int(cohort["cell_sample_count"]),
                    int(evidence["cell_sample_count"]),
                )
                cohort["minimum_skill_score"] = min(
                    float(cohort["minimum_skill_score"]),
                    skill_score,
                )
                cohort_window = evidence["cohort_window"]
                if any(
                    cohort[name] != cohort_window[name]
                    for name in (
                        "schema_version",
                        "selection_policy",
                        "feedback_update_cohort_digest",
                        "population_count",
                        "population_digest",
                        "selected_count",
                        "window_offset",
                    )
                ):
                    group["disqualified"] = True

        qualified_by_parameter: dict[str, list[dict[str, Any]]] = defaultdict(list)
        parameter_order: list[str] = []
        for (parameter, _value_key), group in grouped.items():
            if bool(group["disqualified"]):
                continue
            cohorts = sorted(
                (dict(item) for item in group["cohorts"].values()),
                key=lambda item: str(item["evaluation_cohort_digest"]),
            )
            total_cell_sample_count = sum(
                int(item["cell_sample_count"]) for item in cohorts
            )
            if (
                len(cohorts) < _HISTORICAL_PARAMETER_GUARDRAIL_MIN_COHORTS
                or any(
                    int(item["cell_sample_count"])
                    < _HISTORICAL_PARAMETER_GUARDRAIL_MIN_CELL_N_PER_COHORT
                    for item in cohorts
                )
                or total_cell_sample_count
                < _HISTORICAL_PARAMETER_GUARDRAIL_MIN_TOTAL_CELL_N
                or not _cohort_windows_are_pairwise_non_overlapping(cohorts)
            ):
                continue
            representative = group["representative"]
            if parameter not in qualified_by_parameter:
                parameter_order.append(parameter)
            qualified_by_parameter[parameter].append(
                {
                    "parameter": parameter,
                    "value": representative["value"],
                    "target": representative["target"],
                    "horizon_hours": representative["horizon_hours"],
                    "skill_score": min(
                        float(item["minimum_skill_score"])
                        for item in cohorts
                    ),
                    "source_run_id": representative["source_run_id"],
                    "source_generation": representative["source_generation"],
                    "independent_cohort_count": len(cohorts),
                    "total_cell_sample_count": total_cell_sample_count,
                    "cohort_evidence": cohorts,
                    "policy": _HISTORICAL_PARAMETER_GUARDRAIL_POLICY,
                }
            )

        protected: list[dict[str, Any]] = []
        for parameter in parameter_order:
            qualified_values = qualified_by_parameter[parameter]
            # Two independently supported values for one parameter are
            # contradictory evidence. A hard guardrail must fail closed instead
            # of selecting whichever value happened to be visited first.
            if len(qualified_values) != 1:
                continue
            protected.append(qualified_values[0])
            if len(protected) >= _HISTORICAL_PARAMETER_GUARDRAIL_MAX_ITEMS:
                break
        return {
            "policy": _HISTORICAL_PARAMETER_GUARDRAIL_POLICY,
            "evidence_requirements": {
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
            },
            "protected_parameter_evidence": protected,
        }

    def envelope() -> dict[str, Any]:
        guardrails = historical_parameter_guardrails()
        protected = guardrails["protected_parameter_evidence"]
        source_material: Any = selected
        if protected:
            source_material = {
                "historical_generations": selected,
                "historical_parameter_guardrails": guardrails,
            }
        provenance: dict[str, Any] = {
            "source_digest": digest(source_material),
            "scanned_run_count": len(scanned_candidates),
            "compatible_source_run_count": len(
                {item[1] for item in records}
            ),
            "included_source_run_count": len(
                {str(item["source_run_id"]) for item in selected}
            ),
            "available_generation_count": available_count,
            "included_generation_count": len(selected),
            "omitted_generation_summaries": max(
                0, available_count - len(selected)
            ),
            "omitted_detail_count": omitted_detail_count,
            "max_scanned_runs": _HISTORICAL_EXPERIENCE_SCAN_RUNS,
            "max_source_runs": _HISTORICAL_EXPERIENCE_MAX_RUNS,
            "max_generation_summaries": _HISTORICAL_EXPERIENCE_MAX_GENERATIONS,
            "max_serialized_bytes": _HISTORICAL_EXPERIENCE_MAX_BYTES,
        }
        if history_cutoff_seq is not None and records:
            provenance["history_cutoff_seq"] = history_cutoff_seq
        return {
            "historical_generations": selected,
            "historical_parameter_guardrails": guardrails,
            "historical_provenance": provenance,
        }

    def trim_one_detail() -> bool:
        nonlocal omitted_detail_count
        for source in reversed(selected):
            for field_name in (
                "tool_performance",
                "common_failures",
                "algorithm_failures",
                "weak_horizons",
                "weak_targets",
            ):
                rows = source.get(field_name)
                if isinstance(rows, list) and rows:
                    rows.pop()
                    omitted_detail_count += 1
                    return True
            failure_counts = source.get("sample_failure_counts")
            if isinstance(failure_counts, dict) and failure_counts:
                failure_counts.pop(sorted(failure_counts)[-1])
                omitted_detail_count += 1
                return True
            modifications = source.get("modifications")
            parameter_sets = (
                modifications.get("candidate_parameter_sets")
                if isinstance(modifications, dict)
                else None
            )
            if isinstance(parameter_sets, list) and len(parameter_sets) > 1:
                parameter_sets.pop()
                omitted_detail_count += 1
                return True
            if source.pop("algorithm_synthesis_effect", None) is not None:
                omitted_detail_count += 1
                return True
        return False

    while len(canonical_json(envelope()).encode("utf-8")) > (
        _HISTORICAL_EXPERIENCE_MAX_BYTES
    ):
        if trim_one_detail():
            continue
        if len(selected) > 1:
            selected.pop()
            selected_records.pop()
            continue
        if not selected:
            break
        source = selected[0]
        modifications = source.get("modifications")
        parameter_sets = (
            modifications.get("candidate_parameter_sets")
            if isinstance(modifications, dict)
            else None
        )
        if isinstance(parameter_sets, list) and parameter_sets:
            parameter_sets.pop()
            omitted_detail_count += 1
            continue
        if source.pop("algorithm_synthesis_effect", None) is not None:
            omitted_detail_count += 1
            continue
        selected.pop()
        selected_records.pop()
    return envelope()


def _fit_cross_generation_experience(result: dict[str, Any]) -> dict[str, Any]:
    capacity = result["capacity"]
    while len(canonical_json(result).encode("utf-8")) > (
        _CURRENT_RUN_EXPERIENCE_MAX_BYTES
    ):
        if result["resolved_archived"]:
            result["resolved_archived"].pop()
            capacity["omitted_archived_issues"] += 1
            continue
        if len(result["generations"]) > 2:
            result["generations"].pop(0)
            capacity["omitted_generation_summaries"] += 1
            result["window"]["history_truncated"] = True
            result["window"]["included_generations"] = [
                item["generation"] for item in result["generations"]
            ]
            continue
        if result["active_unresolved"]:
            result["active_unresolved"].pop()
            capacity["omitted_active_issues"] += 1
            continue
        removed_detail = False
        for generation in result["generations"]:
            modifications = generation.get("modifications")
            if isinstance(modifications, dict) and modifications.get(
                "candidate_parameter_sets"
            ):
                modifications["candidate_parameter_sets"] = []
                removed_detail = True
        if removed_detail:
            continue
        raise RuntimeError("cross-generation experience cannot fit its byte budget")
    return result


def build_cross_generation_experience(
    state: Any,
    generation: int,
    *,
    historical_states: Sequence[Any] = (),
    history_cutoff_seq: int | None = None,
) -> JsonObject:
    """Derive bounded, aggregate-only lessons from the replayed event state."""

    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    if (
        history_cutoff_seq is not None
        and (
            isinstance(history_cutoff_seq, bool)
            or not isinstance(history_cutoff_seq, int)
            or history_cutoff_seq < 0
        )
    ):
        raise ValueError("history_cutoff_seq must be a non-negative integer or null")
    if isinstance(historical_states, (str, bytes)) or not isinstance(
        historical_states, Sequence
    ):
        raise TypeError("historical_states must be a sequence")
    available = sorted(
        (
            item
            for item in getattr(state, "generation_analyses", ())
            if item.generation < generation
        ),
        key=lambda item: item.generation,
    )
    scanned = available[-_EXPERIENCE_SCAN_GENERATIONS:]
    plans, adoptions = _experience_plan_sources(state, generation)
    summaries: list[dict[str, Any]] = []
    previous_analysis: GenerationAnalysis | None = None
    for analysis in scanned:
        summaries.append(
            _experience_generation_summary(
                state,
                analysis,
                previous_analysis,
                plans,
                adoptions,
            )
        )
        previous_analysis = analysis
    active, archived = _experience_issue_records(scanned, summaries)
    visible_summaries = summaries[-_EXPERIENCE_MAX_GENERATIONS:]
    selected_active = active[:_EXPERIENCE_MAX_ACTIVE_ISSUES]
    selected_archived = archived[:_EXPERIENCE_MAX_ARCHIVED_ISSUES]
    result: dict[str, Any] = {
        "schema_version": "ecologyrsi-dsh.cross-generation-experience/2",
        "source": "event_ledger_projection",
        "through_generation": available[-1].generation if available else None,
        "window": {
            "requested_generation": generation,
            "available_generation_count": len(available),
            "scanned_generation_count": len(scanned),
            "included_generations": [
                item["generation"] for item in visible_summaries
            ],
            "history_truncated": len(available) > len(visible_summaries),
        },
        "generations": visible_summaries,
        "active_unresolved": selected_active,
        "resolved_archived": selected_archived,
        "capacity": {
            "max_generation_summaries": _EXPERIENCE_MAX_GENERATIONS,
            "max_scanned_generations": _EXPERIENCE_SCAN_GENERATIONS,
            "max_active_issues": _EXPERIENCE_MAX_ACTIVE_ISSUES,
            "max_archived_issues": _EXPERIENCE_MAX_ARCHIVED_ISSUES,
            "max_current_run_serialized_bytes": (
                _CURRENT_RUN_EXPERIENCE_MAX_BYTES
            ),
            "max_historical_serialized_bytes": (
                _HISTORICAL_EXPERIENCE_MAX_BYTES
            ),
            "max_serialized_bytes": CROSS_GENERATION_EXPERIENCE_MAX_BYTES,
            "omitted_generation_summaries": max(
                0, len(summaries) - len(visible_summaries)
            ),
            "omitted_active_issues": max(0, len(active) - len(selected_active)),
            "omitted_archived_issues": max(
                0, len(archived) - len(selected_archived)
            ),
        },
        "contains_raw_samples": False,
    }
    result = _fit_cross_generation_experience(result)
    result.update(
        _historical_experience_payload(
            state,
            historical_states,
            history_cutoff_seq=history_cutoff_seq,
        )
    )
    if len(canonical_json(result).encode("utf-8")) > (
        CROSS_GENERATION_EXPERIENCE_MAX_BYTES
    ):
        raise RuntimeError("cross-generation experience exceeds its byte budget")
    return result


def _rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    if row.get("evidence_class") == EXPLORATORY_EVIDENCE_CLASS:
        floor = row.get("primary_selection_stability_floor")
        interval = row.get("interval_score")
        return (
            bool(row.get("validity_pass")),
            bool(row.get("primary_selection_gate")),
            _finite(floor, -math.inf),
            _finite(row.get("robustness_min_cell_delta"), -math.inf),
            bool(row.get("uq_pass_or_not_required")),
            -_finite(interval, 0.0),
            _finite(row.get("efficiency_score"), -math.inf),
            -_finite(row.get("complexity"), math.inf),
            -int(row.get("slot_index", 0)),
            str(row.get("candidate_id") or ""),
        )
    score = _finite(row.get("score"), -math.inf)
    worst = _finite(row.get("worst_skill_score"), -math.inf)
    scientific_pass = bool(row.get("scientific_pass"))
    judge_accepted = bool(row.get("judge_accepted"))
    return (
        bool(row.get("score") is not None),
        scientific_pass,
        -int(row.get("constraint_violations", 0)),
        score,
        worst,
        # Independent review remains a formal eligibility filter below, but it
        # cannot override objective evidence when choosing a search parent.
        judge_accepted,
        -_finite(row.get("parameter_distance"), math.inf),
        -int(row.get("slot_index", 0)),
    )


def _metric_weaknesses(
    state: Any,
    candidates: list[Any],
    field_name: str,
    identity_fields: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    groups: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: {"skills": [], "normalized_mean_rewards": []}
    )
    for candidate in candidates:
        evaluation = state.evaluation_for(candidate.candidate_id)
        if evaluation is None:
            continue
        raw_rows = evaluation.metrics.get(field_name)
        if not isinstance(raw_rows, list):
            continue
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                continue
            score = _skill(raw)
            reward = _finite(raw.get("normalized_mean_reward"), math.nan)
            if not math.isfinite(score) and not math.isfinite(reward):
                continue
            identity = tuple(raw.get(name) for name in identity_fields)
            if math.isfinite(score):
                groups[identity]["skills"].append(score)
            if math.isfinite(reward):
                groups[identity]["normalized_mean_rewards"].append(reward)
    rows = []
    for identity, aggregates in groups.items():
        scores = aggregates["skills"]
        rewards = aggregates["normalized_mean_rewards"]
        row = {name: value for name, value in zip(identity_fields, identity)}
        if scores:
            row.update(
                {
                    "median_skill_score": median(scores),
                    "mean_skill_score": fmean(scores),
                    "evidence_count": len(scores),
                }
            )
        if rewards:
            row.update(
                {
                    "median_normalized_mean_reward": median(rewards),
                    "mean_normalized_mean_reward": fmean(rewards),
                    "reward_evidence_count": len(rewards),
                }
            )
        rows.append(row)
    rows.sort(
        key=lambda item: (
            min(
                _finite(item.get("median_skill_score"), math.inf),
                _finite(
                    item.get("median_normalized_mean_reward"), math.inf
                ),
            ),
            *(str(item.get(name)) for name in identity_fields),
        )
    )
    return tuple(rows[:_METRIC_WEAKNESS_MAX_GROUPS])


def _historical_scored_configurations(
    state: Any,
    generation: int,
) -> list[tuple[Any, float]]:
    """Accumulate one score per unique parameter set through this generation."""

    required_cohort_digest: str | None = None
    if sample_update_windows_enabled(state.task_manifest):
        current_digests = {
            digest_value
            for candidate in state.candidates
            if candidate.generation == generation
            if (evaluation := state.evaluation_for(candidate.candidate_id)) is not None
            if (digest_value := evaluation_cohort_digest(evaluation)) is not None
        }
        if len(current_digests) != 1:
            return []
        required_cohort_digest = next(iter(current_digests))

    grouped: dict[str, tuple[Any, list[float]]] = {}
    candidates = sorted(
        (item for item in state.candidates if item.generation <= generation),
        key=lambda item: (item.generation, item.slot_index, item.candidate_id),
    )
    for candidate in candidates:
        evaluation = state.evaluation_for(candidate.candidate_id)
        if evaluation is None:
            continue
        if (
            required_cohort_digest is not None
            and evaluation_cohort_digest(evaluation) != required_cohort_digest
        ):
            continue
        proposal = state.proposal(candidate.proposal_id)
        key = canonical_json(_parameter_summary(proposal.changes))
        if key not in grouped:
            grouped[key] = (proposal, [])
        grouped[key][1].append(float(evaluation.score))
    return [
        (proposal, fmean(scores))
        for proposal, scores in grouped.values()
        if scores
    ]


def _parameter_effects(state: Any, generation: int) -> tuple[dict[str, Any], ...]:
    scored = _historical_scored_configurations(state, generation)
    if len(scored) < 3:
        return ()
    names = sorted({name for proposal, _ in scored for name in proposal.changes})
    effects = []
    for name in names:
        pairs = [
            (float(proposal.changes[name]), float(score))
            for proposal, score in scored
            if isinstance(proposal.changes.get(name), (int, float))
            and not isinstance(proposal.changes.get(name), bool)
        ]
        if len(pairs) < 3 or len({item[0] for item in pairs}) < 2:
            continue
        xs = [item[0] for item in pairs]
        ys = [item[1] for item in pairs]
        x_mean, y_mean = fmean(xs), fmean(ys)
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
        denominator = math.sqrt(
            sum((x - x_mean) ** 2 for x in xs)
            * sum((y - y_mean) ** 2 for y in ys)
        )
        association = numerator / denominator if denominator > 1e-15 else 0.0
        effects.append(
            {
                "parameter": name,
                "association_with_score": association,
                "direction": (
                    "positive" if association > 0.2 else "negative" if association < -0.2 else "weak"
                ),
                "evidence_count": len(pairs),
                "unique_configuration_count": len(scored),
                "generation_end": generation,
                "interpretation": (
                    "same_cohort_observational_association_non_causal"
                    if sample_update_windows_enabled(state.task_manifest)
                    else "cross_generation_observational_association_non_causal"
                ),
            }
        )
    effects.sort(key=lambda item: (-abs(item["association_with_score"]), item["parameter"]))
    return tuple(effects)


def _previous_incumbent(state: Any, generation: int) -> tuple[Any, Any] | None:
    candidate_id = state.run.best_candidate_id
    if candidate_id is None:
        return None
    candidate = state.candidate(candidate_id)
    evaluation = state.evaluation_for(candidate_id)
    if (
        candidate.generation >= generation
        or candidate.status is not CandidateStatus.PROMOTED
        or evaluation is None
    ):
        raise RuntimeError("run incumbent is missing its prior promotion evidence")
    return candidate, evaluation


def build_generation_analysis(state: Any, batch: GenerationBatch) -> GenerationAnalysis:
    """Analyze one terminal batch without using raw rows or completion times."""

    candidates = sorted(
        (item for item in state.candidates if item.generation == batch.generation),
        key=lambda item: item.slot_index,
    )
    if len(candidates) != batch.batch_size:
        raise RuntimeError("generation batch is not fully generated")
    terminal = {
        CandidateStatus.EVALUATED,
        CandidateStatus.PROMOTED,
        CandidateStatus.REJECTED,
        CandidateStatus.FAILED,
        CandidateStatus.DUPLICATE,
    }
    if any(item.status not in terminal for item in candidates):
        raise RuntimeError("generation batch still has unfinished candidates")

    incumbent = _previous_incumbent(state, batch.generation)
    rows = [_candidate_row(state, item, batch.parent_candidate_id) for item in candidates]
    if (
        state.task_manifest.metadata.get("execution_protocol")
        == "dsh_native_plugin_evolution@1"
    ):
        profile = FitnessProfile.from_task(state.task_manifest)
        selection_by_candidate: dict[str, Any] = {}
        evaluated = [
            evaluation
            for candidate in candidates
            if (evaluation := state.evaluation_for(candidate.candidate_id)) is not None
        ]
        if incumbent is not None and evaluated:
            selection_by_candidate = {
                item.candidate_id: item
                for item in assess_generation_selection(
                    tuple(evaluated), incumbent[1], profile
                )
            }
        for row in rows:
            evaluation = state.evaluation_for(str(row["candidate_id"]))
            if evaluation is None:
                continue
            runtime = evaluation.metrics.get("runtime_metrics", {})
            assessment = build_fitness_assessment(
                evaluation,
                incumbent[1] if incumbent is not None else None,
                runtime if isinstance(runtime, Mapping) else {},
                profile,
            )
            row.update(assessment.to_dict())
            row["fitness_profile_digest"] = profile.profile_digest
            selection = selection_by_candidate.get(assessment.candidate_id)
            if selection is not None:
                row.update(
                    {
                        "selection_status": selection.status,
                        "primary_selection_stability_floor": (
                            selection.selection_stability_floor
                        ),
                        "primary_selection_gate": bool(
                            assessment.primary_selection_gate
                            and assessment.robustness_pass
                            and selection.primary_selection_gate
                        ),
                        "paired_block_count": selection.paired_block_count,
                        "valid_three_day_start_count": (
                            selection.valid_three_day_start_count
                        ),
                        "paired_block_ids_digest": (
                            selection.paired_block_ids_digest
                        ),
                        "formal_confirmation": False,
                    }
                )
            else:
                row["primary_selection_gate"] = bool(
                    assessment.primary_selection_gate
                    and assessment.robustness_pass
                )
    rows.sort(key=_rank_key, reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    eligible = [row for row in rows if row["eligible"]]
    selected_id = eligible[0]["candidate_id"] if eligible else None
    incumbent_id = incumbent[0].candidate_id if incumbent else None
    selected_evaluation = state.evaluation_for(selected_id) if selected_id else None
    cohort_comparison = (
        evaluation_cohort_comparison(
            state.task_manifest,
            selected_evaluation,
            incumbent[1],
        )
        if selected_evaluation is not None and incumbent is not None
        else "no_incumbent"
    )
    current_cohort_verifiable = bool(
        not sample_update_windows_enabled(state.task_manifest)
        or (
            selected_evaluation is not None
            and evaluation_cohort_digest(selected_evaluation) is not None
        )
    )
    promotion_assessment = (
        assess_promotion_improvement(
            selected_evaluation,
            incumbent[1],
            execution_protocol=state.task_manifest.metadata.get(
                "execution_protocol"
            ),
        )
        if selected_evaluation is not None
        and incumbent is not None
        and cohort_comparison in {"legacy_full_cohort", "same_cohort"}
        else None
    )
    if (
        state.task_manifest.metadata.get("execution_protocol")
        == "dsh_native_plugin_evolution@1"
    ):
        selected_row = next(
            (row for row in rows if row["candidate_id"] == selected_id), None
        )
        improved = bool(
            selected_evaluation is not None
            and current_cohort_verifiable
            and selected_row is not None
            and selected_row.get("primary_selection_gate")
        )
    else:
        improved = bool(
            selected_evaluation is not None
            and current_cohort_verifiable
            and (
                incumbent is None
                or (
                    cohort_comparison in {"legacy_full_cohort", "same_cohort"}
                    and promotion_assessment is not None
                    and promotion_assessment["improved"]
                )
            )
        )
    champion_id = selected_id if improved else None
    outcome = (
        "promoted"
        if champion_id is not None
        else "no_improvement"
        if selected_id is not None
        else "no_eligible_candidate"
    )
    after_id = champion_id or incumbent_id
    search_rows = [
        row
        for row in rows
        if row["score"] is not None and int(row["constraint_violations"]) == 0
    ]
    if not search_rows:
        search_rows = [row for row in rows if row["score"] is not None]
    search_parent_id = (
        str(search_rows[0]["candidate_id"]) if search_rows else None
    )
    for row in rows:
        row["search_parent"] = row["candidate_id"] == search_parent_id
        if row["candidate_id"] == champion_id:
            row["selection_reason"] = (
                "cohort_changed_batch_champion"
                if cohort_comparison == "different_cohort"
                else "generation_champion_strictly_improved_incumbent"
            )
        elif row["candidate_id"] == selected_id:
            row["selection_reason"] = (
                "cohort_digest_unavailable"
                if not current_cohort_verifiable
                or cohort_comparison == "unverifiable"
                else "cohort_changed_search_parent_only"
                if cohort_comparison == "different_cohort"
                else "generation_best_did_not_improve_incumbent"
            )
        elif row["eligible"]:
            row["selection_reason"] = "lower_stable_rank_than_generation_best"
        else:
            row["selection_reason"] = row["classification"]

    failures = Counter(row["classification"] for row in rows if not row["eligible"])
    threshold = 1 if len(rows) == 1 else max(2, math.ceil(len(rows) / 2))
    common_failures = tuple(
        name for name, count in sorted(failures.items()) if count >= threshold
    )
    constraint_ids = tuple(
        str(row["candidate_id"])
        for row in rows
        if int(row["constraint_violations"]) > 0
    )
    disagreements = tuple(
        {
            "candidate_id": row["candidate_id"],
            "scientific_pass": row["scientific_pass"],
            "judge_accepted": row["judge_accepted"],
        }
        for row in rows
        if row["score"] is not None
        and bool(row["judge_available"])
        and bool(row["scientific_pass"]) != bool(row["judge_accepted"])
    )
    target_weaknesses = _metric_weaknesses(
        state, candidates, "targets", ("target", "horizon_hours", "unit")
    )
    horizon_weaknesses = _metric_weaknesses(
        state, candidates, "horizons", ("horizon_hours",)
    )
    historical_scored = _historical_scored_configurations(state, batch.generation)
    effects = _parameter_effects(state, batch.generation)
    algorithm_failures = _algorithm_failure_summary(state, candidates)
    sample_failures = _sample_failure_summary(state, candidates)
    insufficient = len(historical_scored) < 3 or not effects

    directions = []
    if constraint_ids:
        directions.append("收缩引发物理范围违规的参数和搜索步长")
    if target_weaknesses:
        weakest = target_weaknesses[0]
        target = str(weakest.get("target") or "")
        label = _TARGET_LABELS.get(target, target or "预测目标")
        horizon = weakest.get("horizon_hours")
        directions.append(
            f"优先改善 {label}" + (f" 的 {horizon} 小时时距" if horizon is not None else "")
        )
    if disagreements:
        directions.append("保留科学指标，单独处理独立评审提出的治理问题")
    if algorithm_failures:
        directions.append("根据编译或调试失败码更换已登记算法配置并重新预检")
    if any(
        int(row.get("failed", 0)) > 0
        or int(row.get("skipped", 0)) > 0
        or row.get("coverage_pass") is False
        for row in sample_failures
    ):
        directions.append("根据样本失败计数调整工具调用与重试策略，同时保持覆盖率门槛")
    if any(row.get("tool_performance") for row in sample_failures):
        directions.append("根据工具采用率和分目标误差调整下一轮模型工具与路由偏好")
    if outcome == "no_improvement":
        directions.append("保持当前最优方案，仅扩大一个可识别参数方向")
    elif outcome == "no_eligible_candidate":
        if search_parent_id is not None:
            directions.append("从本轮最佳安全评测候选继续搜索，但不将其视为正式通过")
        else:
            directions.append("回到当前最优方案附近，采用更保守的恢复搜索")
    else:
        directions.append("围绕新冠军缩小步长并继续时间前向评测")
    if insufficient:
        directions.append("参数效果证据不足，不推断因果影响")
    directions = list(dict.fromkeys(directions))

    if selected_id is not None and cohort_comparison == "different_cohort":
        reason = (
            f"候选 {selected_id} 是本轮固定样本窗口内排名第一的合格方案；"
            "本轮与历史最优方案使用不同评测窗口，因此未比较跨窗口原始分数，"
            "不改变正式最优方案，仅作为下一轮搜索父方案。"
        )
    elif champion_id is not None and (
        state.task_manifest.metadata.get("execution_protocol")
        == "dsh_native_plugin_evolution@1"
    ):
        reason = (
            f"候选 {champion_id} 在冻结的自适应选择数据上通过实用差异、"
            "单元稳健性与同轮 max-T 稳定性门禁；该结果仅用于选择与后续搜索，"
            "证据类别为 exploratory_adaptive_data，不构成正式确认。"
        )
    elif champion_id is not None:
        reason = (
            f"候选 {champion_id} 同轮排名第一且通过实用差异与"
            "配对区块置信度晋级门槛。"
        )
    elif selected_id is not None and (
        not current_cohort_verifiable or cohort_comparison == "unverifiable"
    ):
        reason = (
            f"同轮最佳候选 {selected_id} 缺少可验证的固定样本窗口摘要；"
            "为避免错误比较跨窗口分数，未作正式晋升。"
        )
    elif selected_id is not None and incumbent is not None and selected_evaluation is not None:
        policy_reason = (
            str(promotion_assessment.get("reason_code"))
            if promotion_assessment is not None
            else "score_not_improved"
        )
        reason = (
            f"同轮最佳候选 {selected_id} 得分 {selected_evaluation.score:.12g}，"
            f"未通过当前最优方案 {incumbent_id} 的晋级门槛"
            f"（{policy_reason}）。"
        )
    else:
        reason = "本轮没有同时通过科学门禁和独立评审的候选，保留当前正式最优方案。"
        if search_parent_id is not None:
            reason += f" 后续搜索从已完成评测候选 {search_parent_id} 继续。"
    focus = directions[0] if directions else "保持当前最优方案并继续有界搜索"
    return GenerationAnalysis(
        run_id=state.run.run_id,
        generation=batch.generation,
        candidate_count=len(candidates),
        eligible_count=len(eligible),
        outcome=outcome,
        selected_candidate_id=selected_id,
        champion_candidate_id=champion_id,
        incumbent_before_candidate_id=incumbent_id,
        incumbent_after_candidate_id=after_id,
        search_parent_candidate_id=search_parent_id,
        ranking=tuple(rows),
        common_failures=common_failures,
        target_weaknesses=target_weaknesses,
        horizon_weaknesses=horizon_weaknesses,
        constraint_failures=constraint_ids,
        judge_disagreements=disagreements,
        parameter_effects=effects,
        next_search_direction=tuple(directions),
        next_generation_focus=focus,
        selection_reason=reason,
        insufficient_evidence=insufficient,
        algorithm_failures=algorithm_failures,
        sample_failures=sample_failures,
    )


__all__ = [
    "CROSS_GENERATION_EXPERIENCE_MAX_BYTES",
    "GenerationAnalysis",
    "GenerationBatch",
    "build_cross_generation_experience",
    "build_generation_analysis",
    "evaluation_cohort_comparison",
    "evaluation_cohort_digest",
    "sample_update_windows_enabled",
]
