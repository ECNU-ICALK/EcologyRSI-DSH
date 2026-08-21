"""Host-owned hierarchical fitness and exploratory selection statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
import random
from typing import Any

from ..core.models import TaskManifest, digest


FITNESS_PROFILE_VERSION = "fitness_profile@1"
EXPLORATORY_EVIDENCE_CLASS = "exploratory_adaptive_data"
FORMAL_EVIDENCE_CLASS = "formal_locked_holdout"
DEFAULT_TARGETS = (
    "air_temperature",
    "relative_humidity",
    "co2_concentration",
)
DEFAULT_HORIZONS = (1, 6, 24)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class FitnessProfile:
    schema_version: str = FITNESS_PROFILE_VERSION
    expected_targets: tuple[str, ...] = DEFAULT_TARGETS
    expected_horizons: tuple[int, ...] = DEFAULT_HORIZONS
    selection_minimum_score_delta: float = 0.005
    selection_minimum_coverage: float = 0.90
    selection_minimum_cell_samples: int = 40
    selection_minimum_paired_blocks: int = 8
    selection_minimum_valid_three_day_starts: int = 4
    moving_block_days: int = 3
    exploratory_resamples: int = 10_000
    exploratory_quantile: float = 0.95
    require_predictive_intervals: bool = False
    latency_reference_ms: float = 10_000.0
    evidence_class: str = EXPLORATORY_EVIDENCE_CLASS

    def __post_init__(self) -> None:
        if self.schema_version != FITNESS_PROFILE_VERSION:
            raise ValueError("unsupported fitness profile")
        if self.evidence_class != EXPLORATORY_EVIDENCE_CLASS:
            raise ValueError("selection evidence class is immutable")
        if not self.expected_targets or not self.expected_horizons:
            raise ValueError("fitness profile objective grid cannot be empty")
        if len(set(self.expected_targets)) != len(self.expected_targets):
            raise ValueError("fitness targets must be unique")
        if len(set(self.expected_horizons)) != len(self.expected_horizons):
            raise ValueError("fitness horizons must be unique")
        if any(not isinstance(item, str) or not item for item in self.expected_targets):
            raise ValueError("fitness targets must be non-empty text")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in self.expected_horizons
        ):
            raise ValueError("fitness horizons must be positive integers")
        for name in (
            "selection_minimum_cell_samples",
            "selection_minimum_paired_blocks",
            "selection_minimum_valid_three_day_starts",
            "moving_block_days",
            "exploratory_resamples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "selection_minimum_score_delta",
            "selection_minimum_coverage",
            "exploratory_quantile",
            "latency_reference_ms",
        ):
            _finite(getattr(self, name), name)
        if not 0 < self.selection_minimum_coverage <= 1:
            raise ValueError("selection coverage must be in (0, 1]")
        if not 0 < self.exploratory_quantile < 1:
            raise ValueError("exploratory quantile must be in (0, 1)")

    @classmethod
    def from_task(cls, task: TaskManifest) -> "FitnessProfile":
        if not isinstance(task, TaskManifest):
            raise TypeError("task must be a TaskManifest")
        raw = task.metadata.get("fitness_profile")
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError("fitness_profile must be an object")
        return cls(**dict(raw))

    def with_overrides(self, **changes: Any) -> "FitnessProfile":
        """Test/configuration helper that still revalidates the whole profile."""

        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "expected_targets": list(self.expected_targets),
            "expected_horizons": list(self.expected_horizons),
            "selection_minimum_score_delta": self.selection_minimum_score_delta,
            "selection_minimum_coverage": self.selection_minimum_coverage,
            "selection_minimum_cell_samples": self.selection_minimum_cell_samples,
            "selection_minimum_paired_blocks": self.selection_minimum_paired_blocks,
            "selection_minimum_valid_three_day_starts": self.selection_minimum_valid_three_day_starts,
            "moving_block_days": self.moving_block_days,
            "exploratory_resamples": self.exploratory_resamples,
            "exploratory_quantile": self.exploratory_quantile,
            "require_predictive_intervals": self.require_predictive_intervals,
            "latency_reference_ms": self.latency_reference_ms,
            "evidence_class": self.evidence_class,
        }

    @property
    def profile_digest(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class FitnessAssessment:
    candidate_id: str
    evidence_class: str
    validity_pass: bool
    validity_failures: tuple[str, ...]
    primary_score: float
    primary_delta: float
    primary_selection_gate: bool
    primary_selection_stability_floor: float | None
    robustness_pass: bool
    robustness_min_cell_delta: float
    robustness_lower_quartile_cell_delta: float
    uq_pass_or_not_required: bool
    interval_score: float | None
    execution_policy_score: float
    efficiency_score: float
    complexity: float
    slot_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class FormalFitnessAssessment:
    candidate_id: str
    evidence_class: str
    status: str
    formal_confirmation: bool
    point_pass: bool
    uq_pass: bool
    failures: tuple[str, ...]
    formal_score: float | None
    formal_score_lcb: float | None
    paired_interval_score_delta_ucb: float | None
    frozen_baseline_uq_artifact_digest: str | None

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _cell_map(evaluation: Any) -> dict[tuple[str, int], Mapping[str, Any]]:
    metrics = getattr(evaluation, "metrics", {})
    rows = metrics.get("targets") if isinstance(metrics, Mapping) else None
    if not isinstance(rows, (list, tuple)):
        return {}
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        target = row.get("target")
        horizon = row.get("horizon_hours")
        if isinstance(target, str) and isinstance(horizon, int) and not isinstance(horizon, bool):
            result[(target, horizon)] = row
    return result


def _cell_skill(row: Mapping[str, Any]) -> float:
    for name in ("skill_score", "score"):
        value = row.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return _finite(value, name)
    raise ValueError("cell skill is missing")


def _execution_policy_score(metrics: Mapping[str, Any]) -> float:
    """Score agent execution separately from registered predictor skill."""

    summary = metrics.get("sample_execution")
    if not isinstance(summary, Mapping):
        return 0.0

    def count(name: str) -> int:
        value = summary.get(name)
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else 0
        )

    eligible = max(
        count("eligible_examples"),
        count("attempted_examples"),
        count("succeeded_examples") + count("failed_examples"),
    )
    if eligible < 1:
        return 0.0
    failure_rate = min(1.0, count("failed_examples") / eligible)
    retry_rate = min(1.0, count("retry_count") / eligible)
    repair_rate = min(1.0, count("repair_count") / eligible)
    raw_critic = summary.get("critic_outcome_counts")
    critic_counts = raw_critic if isinstance(raw_critic, Mapping) else {}
    critic_total = sum(
        int(value)
        for value in critic_counts.values()
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    )
    rejected = sum(
        int(critic_counts.get(name, 0))
        for name in ("rejected", "other")
        if isinstance(critic_counts.get(name, 0), int)
        and not isinstance(critic_counts.get(name, 0), bool)
        and int(critic_counts.get(name, 0)) >= 0
    )
    critic_rejection_rate = rejected / critic_total if critic_total else 0.0
    penalty = (
        0.50 * failure_rate
        + 0.20 * retry_rate
        + 0.20 * repair_rate
        + 0.10 * critic_rejection_rate
    )
    return max(0.0, min(1.0, 1.0 - penalty))


def build_fitness_assessment(
    evaluation: Any,
    incumbent: Any | None,
    runtime_metrics: Mapping[str, Any],
    profile: FitnessProfile,
) -> FitnessAssessment:
    """Build lexicographic validity/skill/robustness/UQ/efficiency layers."""

    if not isinstance(profile, FitnessProfile):
        raise TypeError("profile must be a FitnessProfile")
    metrics = getattr(evaluation, "metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    failures: list[str] = []
    try:
        score = _finite(getattr(evaluation, "score", None), "evaluation score")
    except (TypeError, ValueError):
        score = -math.inf
        failures.append("non_finite_primary_score")
    if not bool(metrics.get("scientific_pass", getattr(evaluation, "passed", False))):
        failures.append("scientific_gate_failed")
    violations = metrics.get("constraint_violations", 0)
    if not isinstance(violations, (int, float)) or isinstance(violations, bool) or violations != 0:
        failures.append("physical_constraint_violation")
    overall_coverage = metrics.get(
        "objective_weight_coverage", metrics.get("sample_execution_coverage")
    )
    if (
        not isinstance(overall_coverage, (int, float))
        or isinstance(overall_coverage, bool)
        or not math.isfinite(float(overall_coverage))
        or float(overall_coverage) < profile.selection_minimum_coverage
    ):
        failures.append("overall_coverage_insufficient")
    expected_grid = {
        (target, horizon)
        for target in profile.expected_targets
        for horizon in profile.expected_horizons
    }
    cells = _cell_map(evaluation)
    if set(cells) != expected_grid:
        failures.append("objective_grid_incomplete")
    incumbent_cells = _cell_map(incumbent) if incumbent is not None else {}
    deltas: list[float] = []
    for key in sorted(expected_grid):
        row = cells.get(key)
        if row is None:
            continue
        coverage = row.get("sample_execution_coverage", row.get("coverage"))
        count = row.get("n", row.get("succeeded"))
        blocks = row.get("paired_block_count", profile.selection_minimum_paired_blocks)
        if (
            not isinstance(coverage, (int, float))
            or isinstance(coverage, bool)
            or not math.isfinite(float(coverage))
            or float(coverage) < profile.selection_minimum_coverage
        ):
            failures.append(f"cell_coverage_insufficient:{key[0]}:{key[1]}")
        if isinstance(count, bool) or not isinstance(count, int) or count < profile.selection_minimum_cell_samples:
            failures.append(f"cell_samples_insufficient:{key[0]}:{key[1]}")
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < profile.selection_minimum_paired_blocks:
            failures.append(f"cell_blocks_insufficient:{key[0]}:{key[1]}")
        try:
            current_skill = _cell_skill(row)
            prior_skill = _cell_skill(incumbent_cells[key]) if key in incumbent_cells else 0.0
            deltas.append(current_skill - prior_skill)
        except (KeyError, TypeError, ValueError):
            failures.append(f"cell_score_non_finite:{key[0]}:{key[1]}")
    if metrics.get("label_isolation_pass", True) is not True:
        failures.append("label_isolation_failed")
    if metrics.get("independent_reviewer_pass", True) is not True:
        failures.append("independent_reviewer_failed")
    primary_delta = (
        score - _finite(getattr(incumbent, "score", 0.0), "incumbent score")
        if incumbent is not None and math.isfinite(score)
        else score
    )
    ordered = sorted(deltas)
    minimum = ordered[0] if ordered else -math.inf
    lower_quartile = (
        ordered[max(0, math.ceil(0.25 * len(ordered)) - 1)]
        if ordered
        else -math.inf
    )
    robustness_pass = bool(ordered) and minimum >= 0.0
    uq_pass = (
        not profile.require_predictive_intervals
        or metrics.get("uq_pass") is True
    )
    interval_value = metrics.get("normalized_interval_score")
    interval_score = (
        _finite(interval_value, "normalized_interval_score")
        if isinstance(interval_value, (int, float)) and not isinstance(interval_value, bool)
        else None
    )
    execution_policy_score = _execution_policy_score(metrics)
    latency = runtime_metrics.get("latency_ms") if isinstance(runtime_metrics, Mapping) else None
    efficiency = (
        -math.log1p(max(0.0, _finite(latency, "latency_ms")) / profile.latency_reference_ms)
        if isinstance(latency, (int, float)) and not isinstance(latency, bool)
        else 0.0
    )
    complexity_value = metrics.get("behavior_complexity", 0.0)
    complexity = (
        _finite(complexity_value, "behavior_complexity")
        if isinstance(complexity_value, (int, float)) and not isinstance(complexity_value, bool)
        else math.inf
    )
    validity = not failures
    return FitnessAssessment(
        candidate_id=str(getattr(evaluation, "candidate_id", "")),
        evidence_class=EXPLORATORY_EVIDENCE_CLASS,
        validity_pass=validity,
        validity_failures=tuple(dict.fromkeys(failures)),
        primary_score=score,
        primary_delta=primary_delta,
        primary_selection_gate=(
            validity and primary_delta > profile.selection_minimum_score_delta
        ),
        primary_selection_stability_floor=None,
        robustness_pass=robustness_pass,
        robustness_min_cell_delta=minimum,
        robustness_lower_quartile_cell_delta=lower_quartile,
        uq_pass_or_not_required=uq_pass,
        interval_score=interval_score,
        execution_policy_score=execution_policy_score,
        efficiency_score=efficiency,
        complexity=complexity,
        slot_index=int(metrics.get("slot_index", 0) or 0),
    )


def fitness_ranking_key(assessment: FitnessAssessment) -> tuple[Any, ...]:
    """Lower layers can never compensate for failure in a higher layer."""

    stability = assessment.primary_selection_stability_floor
    return (
        assessment.validity_pass,
        assessment.primary_selection_gate,
        stability if stability is not None else -math.inf,
        assessment.robustness_min_cell_delta,
        assessment.uq_pass_or_not_required,
        -assessment.interval_score if assessment.interval_score is not None else 0.0,
        assessment.execution_policy_score,
        assessment.efficiency_score,
        -assessment.complexity,
        -assessment.slot_index,
        assessment.candidate_id,
    )


def build_formal_fitness_assessment(
    evaluation: Any,
    frozen_baseline_uq_artifact: Mapping[str, Any],
    profile: FitnessProfile,
) -> FormalFitnessAssessment:
    """Apply non-compensatory formal point and interval gates.

    The comparator is the explicitly supplied calibration-fit-selected frozen
    baseline artifact.  The adaptive selection incumbent is intentionally not
    accepted by this interface.
    """

    if not isinstance(profile, FitnessProfile):
        raise TypeError("profile must be a FitnessProfile")
    if not isinstance(frozen_baseline_uq_artifact, Mapping):
        raise TypeError("frozen baseline UQ artifact must be an object")
    metrics = getattr(evaluation, "metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    candidate_id = str(getattr(evaluation, "candidate_id", ""))
    baseline_digest = frozen_baseline_uq_artifact.get("artifact_digest")
    required = (
        "formal_score",
        "formal_score_lcb",
        "formal_valid_three_day_start_count",
        "formal_baseline_uq_artifact_digest",
        "paired_interval_score_delta_ucb",
    )
    missing = [name for name in required if metrics.get(name) is None]
    if missing:
        return FormalFitnessAssessment(
            candidate_id=candidate_id,
            evidence_class=FORMAL_EVIDENCE_CLASS,
            status="inconclusive",
            formal_confirmation=False,
            point_pass=False,
            uq_pass=False,
            failures=tuple(f"formal_evidence_missing:{name}" for name in missing),
            formal_score=None,
            formal_score_lcb=None,
            paired_interval_score_delta_ucb=None,
            frozen_baseline_uq_artifact_digest=(
                str(baseline_digest) if isinstance(baseline_digest, str) else None
            ),
        )

    failures: list[str] = []
    formal_score = _finite(metrics["formal_score"], "formal_score")
    formal_score_lcb = _finite(metrics["formal_score_lcb"], "formal_score_lcb")
    interval_delta_ucb = _finite(
        metrics["paired_interval_score_delta_ucb"],
        "paired_interval_score_delta_ucb",
    )
    if formal_score <= 0.0:
        failures.append("formal_score_nonpositive")
    if formal_score_lcb <= 0.0:
        failures.append("formal_score_lcb_nonpositive")
    starts = metrics["formal_valid_three_day_start_count"]
    if isinstance(starts, bool) or not isinstance(starts, int) or starts < 10:
        failures.append("formal_continuous_starts_insufficient")
    overall_coverage = metrics.get("objective_weight_coverage")
    if (
        isinstance(overall_coverage, bool)
        or not isinstance(overall_coverage, (int, float))
        or not math.isfinite(float(overall_coverage))
        or float(overall_coverage) < 0.95
    ):
        failures.append("formal_overall_coverage_insufficient")
    if (
        frozen_baseline_uq_artifact.get("policy_id")
        != "cellwise_time_block_calibrated_residual@1"
        or frozen_baseline_uq_artifact.get("alpha") != 0.1
        or not isinstance(baseline_digest, str)
        or metrics["formal_baseline_uq_artifact_digest"] != baseline_digest
    ):
        failures.append("formal_baseline_uq_binding_mismatch")

    expected_grid = {
        (target, horizon)
        for target in profile.expected_targets
        for horizon in profile.expected_horizons
    }
    cells = _cell_map(evaluation)
    if set(cells) != expected_grid:
        failures.append("formal_objective_grid_incomplete")
    for target, horizon in sorted(expected_grid):
        cell = cells.get((target, horizon))
        if cell is None:
            continue
        suffix = f":{target}:{horizon}"
        count = cell.get("n")
        blocks = cell.get("paired_block_count")
        coverage = cell.get("sample_execution_coverage", cell.get("coverage"))
        interval_lcb = cell.get("interval_coverage_lcb")
        try:
            skill = _cell_skill(cell)
        except (TypeError, ValueError):
            failures.append("formal_cell_skill_missing" + suffix)
        else:
            if skill < 0.0:
                failures.append("formal_cell_skill_negative" + suffix)
        if isinstance(count, bool) or not isinstance(count, int) or count < 80:
            failures.append("formal_cell_samples_insufficient" + suffix)
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 14:
            failures.append("formal_cell_blocks_insufficient" + suffix)
        if (
            isinstance(coverage, bool)
            or not isinstance(coverage, (int, float))
            or not math.isfinite(float(coverage))
            or float(coverage) < 0.95
        ):
            failures.append("formal_cell_coverage_insufficient" + suffix)
        if (
            isinstance(interval_lcb, bool)
            or not isinstance(interval_lcb, (int, float))
            or not math.isfinite(float(interval_lcb))
            or float(interval_lcb) < 0.85
        ):
            failures.append("formal_interval_coverage_lcb_insufficient" + suffix)
    if interval_delta_ucb > 0.05:
        failures.append("formal_interval_score_noninferiority_failed")

    point_failures = tuple(
        item
        for item in failures
        if not item.startswith("formal_interval_")
        and item != "formal_baseline_uq_binding_mismatch"
    )
    uq_failures = tuple(
        item
        for item in failures
        if item.startswith("formal_interval_")
        or item == "formal_baseline_uq_binding_mismatch"
    )
    point_pass = not point_failures
    uq_pass = not uq_failures
    passed = point_pass and uq_pass and not failures
    return FormalFitnessAssessment(
        candidate_id=candidate_id,
        evidence_class=FORMAL_EVIDENCE_CLASS,
        status="passed" if passed else "rejected",
        formal_confirmation=passed,
        point_pass=point_pass,
        uq_pass=uq_pass,
        failures=tuple(dict.fromkeys(failures)),
        formal_score=formal_score,
        formal_score_lcb=formal_score_lcb,
        paired_interval_score_delta_ucb=interval_delta_ucb,
        frozen_baseline_uq_artifact_digest=str(baseline_digest),
    )


def _legal_starts(block_indices: Sequence[int], days: int) -> tuple[int, ...]:
    return tuple(
        offset
        for offset in range(0, len(block_indices) - days + 1)
        if tuple(block_indices[offset : offset + days])
        == tuple(range(block_indices[offset], block_indices[offset] + days))
    )


def build_moving_block_resample_indices(
    block_indices: Sequence[int],
    *,
    resamples: int,
    seed_material: Any,
    moving_block_days: int = 3,
) -> tuple[tuple[int, ...], ...]:
    """Draw non-circular contiguous calendar blocks and truncate to cohort size."""

    indices = tuple(block_indices)
    if len(indices) != len(set(indices)) or tuple(sorted(indices)) != indices:
        raise ValueError("block indices must be unique and increasing")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be positive")
    starts = _legal_starts(indices, moving_block_days)
    if not starts:
        return ()
    randomizer = random.Random(int(digest(seed_material)[:16], 16))
    result = []
    for _ in range(resamples):
        selected: list[int] = []
        while len(selected) < len(indices):
            start = starts[randomizer.randrange(len(starts))]
            selected.extend(indices[start : start + moving_block_days])
        result.append(tuple(selected[: len(indices)]))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SelectionAssessment:
    candidate_id: str
    evidence_class: str
    status: str
    paired_block_count: int
    valid_three_day_start_count: int
    paired_block_ids_digest: str | None
    primary_delta: float | None
    selection_stability_floor: float | None
    primary_selection_gate: bool
    formal_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def assess_generation_selection(
    candidates: Sequence[Any],
    incumbent: Any,
    profile: FitnessProfile,
) -> tuple[SelectionAssessment, ...]:
    """Apply one shared centered max-T resample family to predeclared siblings."""

    from ..evolution.promotion import _resampled_objective, _validated_evidence

    incumbent_evidence = _validated_evidence(incumbent)
    parsed: list[tuple[Any, Mapping[str, Any], dict[int, str]]] = []
    if incumbent_evidence is not None:
        incumbent_raw = incumbent.metrics.get("promotion_block_evidence", {})
        incumbent_map = {
            int(block["origin_block_index"]): str(block["block_id"])
            for block in incumbent_raw.get("blocks", ())
            if isinstance(block, Mapping)
            and isinstance(block.get("origin_block_index"), int)
            and not isinstance(block.get("origin_block_index"), bool)
        }
    else:
        incumbent_map = {}
    failures: dict[str, SelectionAssessment] = {}
    for evaluation in candidates:
        candidate_id = str(getattr(evaluation, "candidate_id", ""))
        evidence = _validated_evidence(evaluation)
        raw = getattr(evaluation, "metrics", {}).get("promotion_block_evidence", {})
        mapping = {
            int(block["origin_block_index"]): str(block["block_id"])
            for block in raw.get("blocks", ())
            if isinstance(block, Mapping)
            and isinstance(block.get("origin_block_index"), int)
            and not isinstance(block.get("origin_block_index"), bool)
        } if isinstance(raw, Mapping) else {}
        if (
            evidence is None
            or incumbent_evidence is None
            or mapping != incumbent_map
            or set(evidence["blocks"]) != set(incumbent_evidence["blocks"])
        ):
            failures[candidate_id] = SelectionAssessment(
                candidate_id, EXPLORATORY_EVIDENCE_CLASS, "incompatible_evidence",
                0, 0, None, None, None, False,
            )
        else:
            parsed.append((evaluation, evidence, mapping))
    indices = tuple(sorted(incumbent_map))
    starts = _legal_starts(indices, profile.moving_block_days)
    block_digest = digest(
        {"indices": list(indices), "block_ids": [incumbent_map[index] for index in indices]}
    ) if indices else None
    if (
        len(indices) < profile.selection_minimum_paired_blocks
        or len(starts) < profile.selection_minimum_valid_three_day_starts
    ):
        for evaluation, _evidence, _mapping in parsed:
            candidate_id = str(evaluation.candidate_id)
            failures[candidate_id] = SelectionAssessment(
                candidate_id, EXPLORATORY_EVIDENCE_CLASS, "insufficient_evidence",
                len(indices), len(starts), block_digest, None, None, False,
            )
        return tuple(failures[str(item.candidate_id)] for item in candidates)
    draws = build_moving_block_resample_indices(
        indices,
        resamples=profile.exploratory_resamples,
        seed_material={
            "profile_digest": profile.profile_digest,
            "block_ids_digest": block_digest,
            "candidate_ids": sorted(str(item[0].candidate_id) for item in parsed),
        },
        moving_block_days=profile.moving_block_days,
    )
    all_ids = [incumbent_map[index] for index in indices]
    point = {
        str(evaluation.candidate_id): (
            _resampled_objective(evidence, all_ids)
            - _resampled_objective(incumbent_evidence, all_ids)
        )
        for evaluation, evidence, _mapping in parsed
    }
    maxima: list[float] = []
    for draw in draws:
        ids = [incumbent_map[index] for index in draw]
        centered = []
        for evaluation, evidence, _mapping in parsed:
            candidate_id = str(evaluation.candidate_id)
            delta_star = (
                _resampled_objective(evidence, ids)
                - _resampled_objective(incumbent_evidence, ids)
            )
            centered.append(delta_star - point[candidate_id])
        maxima.append(max(centered))
    maxima.sort()
    q_index = max(0, math.ceil(profile.exploratory_quantile * len(maxima)) - 1)
    q = maxima[q_index]
    results = dict(failures)
    for evaluation, _evidence, _mapping in parsed:
        candidate_id = str(evaluation.candidate_id)
        delta = point[candidate_id]
        floor = delta - q
        passed = delta > profile.selection_minimum_score_delta and floor > 0.0
        results[candidate_id] = SelectionAssessment(
            candidate_id=candidate_id,
            evidence_class=EXPLORATORY_EVIDENCE_CLASS,
            status="selection_only" if passed else "unstable_or_below_delta",
            paired_block_count=len(indices),
            valid_three_day_start_count=len(starts),
            paired_block_ids_digest=block_digest,
            primary_delta=delta,
            selection_stability_floor=floor,
            primary_selection_gate=passed,
        )
    return tuple(results[str(item.candidate_id)] for item in candidates)


__all__ = [
    "EXPLORATORY_EVIDENCE_CLASS",
    "FORMAL_EVIDENCE_CLASS",
    "FITNESS_PROFILE_VERSION",
    "FitnessAssessment",
    "FitnessProfile",
    "FormalFitnessAssessment",
    "SelectionAssessment",
    "assess_generation_selection",
    "build_fitness_assessment",
    "build_formal_fitness_assessment",
    "build_moving_block_resample_indices",
    "fitness_ranking_key",
]
