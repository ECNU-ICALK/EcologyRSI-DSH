"""Value-blind, causal cohort planning for adaptive finalist epochs."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, Sequence

from ..core.models import digest
from ..core.trajectory import OriginOccurrence
from ..data.splits import IndexRange
from ..evolution.schedule import OptimizationSchedule
from .greenhouse_prediction import MAX_EXOGENOUS_RIDGE_HISTORY_STEPS


COHORT_PLANNER_SCHEMA = "ecologyrsi-dsh.epoch-cohort-planner/1"
RUN_ADAPTATION_COHORT_SCHEMA = "ecologyrsi-dsh.run-adaptation-cohort/1"
GENERATION_COHORTS_SCHEMA = "ecologyrsi-dsh.generation-cohorts/1"
CAPACITY_REPORT_SCHEMA = "ecologyrsi-dsh.epoch-capacity-report/1"
COHORT_REUSE_POLICY = "cycle_after_exhaustion@1"
DEFAULT_HORIZONS = (1, 6, 24)
# Cohorts are shared by every candidate in a generation, including candidates
# that request the predictor's largest legal history window.  Plan against that
# common execution envelope so a frozen origin is materializable regardless of
# the candidate selected for the formal trajectory.
DEFAULT_HISTORY_STEPS = MAX_EXOGENOUS_RIDGE_HISTORY_STEPS
DEFAULT_SCORING_CELLS_PER_ORIGIN = 9


class DatasetIdentityView(Protocol):
    dataset_id: str
    episode_id: str
    timestamps: Sequence[int]
    partitions: Mapping[str, IndexRange]


class CohortCapacityError(ValueError):
    """Raised when no eligible causal origin exists for a requested cohort."""

    def __init__(
        self,
        *,
        required: int,
        available: int,
        max_generations: int,
    ) -> None:
        self.required = required
        self.available = available
        self.max_generations = max_generations
        super().__init__(
            "insufficient causal cohort capacity: "
            f"required={required}, available={available}, "
            f"max_generations={max_generations}"
        )


def _strict_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _selection_partition(dataset: DatasetIdentityView) -> IndexRange:
    for name in ("model_selection", "training_feedback", "validation"):
        selected = dataset.partitions.get(name)
        if isinstance(selected, IndexRange):
            return selected
    raise ValueError("dataset does not expose a model-selection partition")


def _dataset_identity(dataset: DatasetIdentityView) -> tuple[str, str, tuple[int, ...]]:
    dataset_id = str(getattr(dataset, "dataset_id", "")).strip()
    episode_id = str(getattr(dataset, "episode_id", "")).strip()
    if not dataset_id or not episode_id:
        raise ValueError("dataset_id and episode_id must be non-empty")
    raw_timestamps = getattr(dataset, "timestamps", None)
    if not isinstance(raw_timestamps, (list, tuple)):
        raise TypeError("dataset timestamps must be a sequence")
    timestamps = tuple(raw_timestamps)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in timestamps):
        raise ValueError("dataset timestamps must be integer hours")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("dataset timestamps must increase strictly")
    selected = _selection_partition(dataset)
    if selected.end > len(timestamps):
        raise ValueError("model-selection partition exceeds dataset timestamps")
    return dataset_id, episode_id, timestamps


@dataclass(frozen=True, slots=True)
class PlannedOrigin:
    origin_id: str
    dataset_id: str
    episode_id: str
    origin_index: int
    origin_timestamp: int
    maximum_target_timestamp: int
    maturity_digest: str
    # A deterministic occurrence number distinguishes a reused source origin
    # from its first pass. It is zero for the initial pass, then increments
    # each time the planner wraps around the eligible-origin population.
    reuse_index: int = 0

    def __post_init__(self) -> None:
        for name in ("origin_id", "maturity_digest"):
            _sha256(getattr(self, name), name)
        _strict_integer(self.origin_index, "origin_index")
        if isinstance(self.origin_timestamp, bool) or not isinstance(
            self.origin_timestamp, int
        ):
            raise ValueError("origin_timestamp must be an integer")
        if (
            isinstance(self.maximum_target_timestamp, bool)
            or not isinstance(self.maximum_target_timestamp, int)
            or self.maximum_target_timestamp <= self.origin_timestamp
        ):
            raise ValueError("maximum target timestamp must follow origin")
        _strict_integer(self.reuse_index, "reuse_index", minimum=0)

    def identity_dict(self) -> dict[str, Any]:
        return {
            "origin_id": self.origin_id,
            "dataset_id": self.dataset_id,
            "episode_id": self.episode_id,
            "origin_index": self.origin_index,
            "origin_timestamp": self.origin_timestamp,
            "maximum_target_timestamp": self.maximum_target_timestamp,
            "maturity_digest": self.maturity_digest,
            "reuse_index": self.reuse_index,
        }

    def occurrence(
        self,
        *,
        cohort_role: str,
        generation: int,
        candidate_id: str,
        revision_id: str,
    ) -> OriginOccurrence:
        """Bind this planned source to one executable run occurrence."""

        return OriginOccurrence.from_source(
            source_origin_id=self.origin_id,
            cycle_index=self.reuse_index,
            origin_timestamp=self.origin_timestamp,
            maturity_digest=self.maturity_digest,
            cohort_role=cohort_role,
            generation=generation,
            candidate_id=candidate_id,
            revision_id=revision_id,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlannedOrigin":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class PlannedCohort:
    role: str
    origins: tuple[PlannedOrigin, ...]
    maximum_horizon: int
    shared_candidate_count: int = 0
    shared_arm_count: int = 0

    def __post_init__(self) -> None:
        role = str(self.role).strip()
        if not role:
            raise ValueError("cohort role must be non-empty")
        object.__setattr__(self, "role", role)
        raw = self.origins
        if not isinstance(raw, (list, tuple)):
            raise TypeError("cohort origins must be a sequence")
        origins = tuple(
            PlannedOrigin.from_dict(item) if isinstance(item, Mapping) else item
            for item in raw
        )
        if not origins or not all(isinstance(item, PlannedOrigin) for item in origins):
            raise ValueError("cohort requires planned origins")
        occurrence_keys = {
            (item.origin_id, item.reuse_index) for item in origins
        }
        if len(occurrence_keys) != len(origins):
            raise ValueError("cohort origin occurrences must be unique")
        if any(
            (right.reuse_index, right.origin_timestamp)
            <= (left.reuse_index, left.origin_timestamp)
            for left, right in zip(origins, origins[1:])
        ):
            raise ValueError("cohort origins must be ordered causally")
        object.__setattr__(self, "origins", origins)
        _strict_integer(self.maximum_horizon, "maximum_horizon", minimum=1)
        _strict_integer(
            self.shared_candidate_count, "shared_candidate_count", minimum=0
        )
        _strict_integer(self.shared_arm_count, "shared_arm_count", minimum=0)

    @property
    def origin_ids(self) -> tuple[str, ...]:
        return tuple(item.origin_id for item in self.origins)

    @property
    def origin_occurrence_keys(self) -> tuple[tuple[str, int], ...]:
        """Stable identity for one planned source-origin occurrence."""

        return tuple((item.origin_id, item.reuse_index) for item in self.origins)

    @property
    def origin_count(self) -> int:
        return len(self.origins)

    def identity_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "origin_count": self.origin_count,
            "origins": [item.identity_dict() for item in self.origins],
            "maximum_horizon": self.maximum_horizon,
            "shared_candidate_count": self.shared_candidate_count,
            "shared_arm_count": self.shared_arm_count,
        }

    @property
    def cohort_digest(self) -> str:
        return digest(self.identity_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_dict(), "cohort_digest": self.cohort_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlannedCohort":
        data = dict(value)
        supplied = data.pop("cohort_digest", None)
        data.pop("origin_count", None)
        result = cls(**data)
        if supplied is not None and supplied != result.cohort_digest:
            raise ValueError("cohort_digest does not match cohort identity")
        return result


@dataclass(frozen=True, slots=True)
class PlannedBatch:
    batch_index: int
    cohort: PlannedCohort

    def __post_init__(self) -> None:
        _strict_integer(self.batch_index, "batch_index")
        if isinstance(self.cohort, Mapping):
            object.__setattr__(self, "cohort", PlannedCohort.from_dict(self.cohort))
        if not isinstance(self.cohort, PlannedCohort):
            raise TypeError("batch cohort must be PlannedCohort")
        if self.cohort.role != "adaptation_batch":
            raise ValueError("planned batch must use adaptation_batch role")

    @property
    def origin_ids(self) -> tuple[str, ...]:
        return self.cohort.origin_ids

    @property
    def origin_occurrence_keys(self) -> tuple[tuple[str, int], ...]:
        return self.cohort.origin_occurrence_keys

    @property
    def origin_count(self) -> int:
        return self.cohort.origin_count

    @property
    def batch_digest(self) -> str:
        return digest(self.identity_dict())

    def identity_dict(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "cohort": self.cohort.identity_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_dict(), "batch_digest": self.batch_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlannedBatch":
        data = dict(value)
        supplied = data.pop("batch_digest", None)
        result = cls(**data)
        if supplied is not None and supplied != result.batch_digest:
            raise ValueError("batch_digest does not match batch identity")
        return result


@dataclass(frozen=True, slots=True)
class RunAdaptationCohort:
    dataset_id: str
    episode_id: str
    seed: int
    cohort: PlannedCohort
    batches: tuple[PlannedBatch, ...]
    schema_version: str = RUN_ADAPTATION_COHORT_SCHEMA

    def __post_init__(self) -> None:
        _strict_integer(self.seed, "seed")
        if isinstance(self.cohort, Mapping):
            object.__setattr__(self, "cohort", PlannedCohort.from_dict(self.cohort))
        raw_batches = self.batches
        if not isinstance(raw_batches, (list, tuple)):
            raise TypeError("adaptation batches must be a sequence")
        batches = tuple(
            PlannedBatch.from_dict(item) if isinstance(item, Mapping) else item
            for item in raw_batches
        )
        if [item.batch_index for item in batches] != list(range(len(batches))):
            raise ValueError("adaptation batch indices must be contiguous")
        flattened = tuple(
            occurrence
            for batch in batches
            for occurrence in batch.origin_occurrence_keys
        )
        if flattened != self.cohort.origin_occurrence_keys:
            raise ValueError("adaptation batches must exactly partition the cohort")
        object.__setattr__(self, "batches", batches)

    @property
    def origins(self) -> tuple[PlannedOrigin, ...]:
        return self.cohort.origins

    @property
    def origin_ids(self) -> tuple[str, ...]:
        return self.cohort.origin_ids

    @property
    def origin_occurrence_keys(self) -> tuple[tuple[str, int], ...]:
        return self.cohort.origin_occurrence_keys

    @property
    def origin_count(self) -> int:
        return self.cohort.origin_count

    @property
    def batch_digests(self) -> tuple[str, ...]:
        return tuple(item.batch_digest for item in self.batches)

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "planner_schema": COHORT_PLANNER_SCHEMA,
            "dataset_id": self.dataset_id,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "cohort": self.cohort.identity_dict(),
            "batches": [item.identity_dict() for item in self.batches],
        }

    @property
    def adaptation_digest(self) -> str:
        return digest(self.identity_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_dict(), "adaptation_digest": self.adaptation_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunAdaptationCohort":
        data = dict(value)
        supplied = data.pop("adaptation_digest", None)
        data.pop("planner_schema", None)
        result = cls(**data)
        if supplied is not None and supplied != result.adaptation_digest:
            raise ValueError("adaptation_digest does not match cohort identity")
        return result


@dataclass(frozen=True, slots=True)
class GenerationCohorts:
    dataset_id: str
    episode_id: str
    generation: int
    seed: int
    adaptation_digest: str
    adaptation_batch_digests: tuple[str, ...]
    screening: PlannedCohort
    holdout: PlannedCohort
    schema_version: str = GENERATION_COHORTS_SCHEMA

    def __post_init__(self) -> None:
        _strict_integer(self.generation, "generation")
        _strict_integer(self.seed, "seed")
        _sha256(self.adaptation_digest, "adaptation_digest")
        object.__setattr__(
            self,
            "adaptation_batch_digests",
            tuple(
                _sha256(item, "adaptation_batch_digest")
                for item in self.adaptation_batch_digests
            ),
        )
        for name in ("screening", "holdout"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = PlannedCohort.from_dict(value)
                object.__setattr__(self, name, value)
            if not isinstance(value, PlannedCohort) or value.role != name:
                raise ValueError(f"generation {name} cohort role is invalid")
        if set(self.screening.origin_occurrence_keys) & set(
            self.holdout.origin_occurrence_keys
        ):
            raise ValueError("generation screening and holdout must be disjoint")

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "planner_schema": COHORT_PLANNER_SCHEMA,
            "dataset_id": self.dataset_id,
            "episode_id": self.episode_id,
            "generation": self.generation,
            "seed": self.seed,
            "adaptation_digest": self.adaptation_digest,
            "adaptation_batch_digests": list(self.adaptation_batch_digests),
            "screening": self.screening.identity_dict(),
            "holdout": self.holdout.identity_dict(),
        }

    @property
    def generation_cohorts_digest(self) -> str:
        return digest(self.identity_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_dict(),
            "generation_cohorts_digest": self.generation_cohorts_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationCohorts":
        data = dict(value)
        supplied = data.pop("generation_cohorts_digest", None)
        data.pop("planner_schema", None)
        result = cls(**data)
        if supplied is not None and supplied != result.generation_cohorts_digest:
            raise ValueError("generation cohort digest does not match identity")
        return result


@dataclass(frozen=True, slots=True)
class CohortCapacityReport:
    dataset_id: str
    episode_id: str
    planned_generations: int
    available_partition_origins: int
    available_eligible_origins: int
    required_unique_origins: int
    max_feasible_generations: int
    sufficient: bool
    maturity_gaps: Mapping[str, int]
    candidate_origin_executions_per_generation: int
    scoring_cells_per_generation: int
    candidate_origin_executions_for_run: int
    scoring_cells_for_run: int
    schedule_digest: str
    seed: int
    cohort_reuse_policy: str = COHORT_REUSE_POLICY
    reused_origin_occurrences: int = 0
    schema_version: str = CAPACITY_REPORT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "planned_generations",
            "available_partition_origins",
            "available_eligible_origins",
            "required_unique_origins",
            "max_feasible_generations",
            "candidate_origin_executions_per_generation",
            "scoring_cells_per_generation",
            "candidate_origin_executions_for_run",
            "scoring_cells_for_run",
            "seed",
            "reused_origin_occurrences",
        ):
            _strict_integer(getattr(self, name), name)
        if self.cohort_reuse_policy != COHORT_REUSE_POLICY:
            raise ValueError(
                f"cohort_reuse_policy must be {COHORT_REUSE_POLICY!r}"
            )
        if not isinstance(self.sufficient, bool):
            raise TypeError("sufficient must be a bool")
        _sha256(self.schedule_digest, "schedule_digest")
        object.__setattr__(
            self,
            "maturity_gaps",
            {str(key): int(value) for key, value in self.maturity_gaps.items()},
        )

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "planner_schema": COHORT_PLANNER_SCHEMA,
            "dataset_id": self.dataset_id,
            "episode_id": self.episode_id,
            "planned_generations": self.planned_generations,
            "available_partition_origins": self.available_partition_origins,
            "available_eligible_origins": self.available_eligible_origins,
            "required_unique_origins": self.required_unique_origins,
            "max_feasible_generations": self.max_feasible_generations,
            "sufficient": self.sufficient,
            "maturity_gaps": dict(self.maturity_gaps),
            "candidate_origin_executions_per_generation": (
                self.candidate_origin_executions_per_generation
            ),
            "scoring_cells_per_generation": self.scoring_cells_per_generation,
            "candidate_origin_executions_for_run": (
                self.candidate_origin_executions_for_run
            ),
            "scoring_cells_for_run": self.scoring_cells_for_run,
            "schedule_digest": self.schedule_digest,
            "seed": self.seed,
            "cohort_reuse_policy": self.cohort_reuse_policy,
            "reused_origin_occurrences": self.reused_origin_occurrences,
        }

    @property
    def planner_digest(self) -> str:
        return digest(self.identity_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {**self.identity_dict(), "planner_digest": self.planner_digest}
        # Explicit occurrence/source terminology prevents cyclic reuse from
        # being mistaken for additional independent observations.  Keep the
        # legacy keys in the frozen identity for existing ledgers, while all
        # new UI/API consumers can use these unambiguous fields.
        payload.update(
            {
                "planned_origin_occurrences": self.required_unique_origins,
                "available_source_origins": self.available_eligible_origins,
                "effective_source_count": min(
                    self.required_unique_origins,
                    self.available_eligible_origins,
                ),
                "reuse_fraction": (
                    self.reused_origin_occurrences / self.required_unique_origins
                    if self.required_unique_origins
                    else 0.0
                ),
                # A cycle is a complete pass over the eligible source pool.
                # Keeping this explicit makes reuse auditable without
                # treating repeated occurrences as independent origins.
                "reused_occurrence_count": self.reused_origin_occurrences,
                "cycle_count": (
                    math.ceil(
                        self.required_unique_origins
                        / self.available_eligible_origins
                    )
                    if self.available_eligible_origins
                    else 0
                ),
            }
        )
        return payload


def _eligible_origins(
    dataset: DatasetIdentityView,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    history_steps: int = DEFAULT_HISTORY_STEPS,
) -> tuple[tuple[PlannedOrigin, ...], dict[str, int]]:
    dataset_id, episode_id, timestamps = _dataset_identity(dataset)
    selected = _selection_partition(dataset)
    maximum_horizon = max(horizons)
    by_timestamp = {timestamp: index for index, timestamp in enumerate(timestamps)}
    origins: list[PlannedOrigin] = []
    missing_history = 0
    missing_horizon = 0
    timestamp_gap_origins = 0
    for index in range(selected.start, selected.end):
        origin_timestamp = timestamps[index]
        # ``_base_samples`` treats history_steps as the number of observations
        # including the forecast origin (lags 0..N-1) and forbids history from
        # crossing the evaluation partition.  Keep the value-blind planner's
        # timestamp contract identical to that evaluator contract.
        history_indices = tuple(
            by_timestamp.get(origin_timestamp - lag)
            for lag in range(history_steps)
        )
        target_indices = tuple(
            by_timestamp.get(origin_timestamp + horizon) for horizon in horizons
        )
        history_valid = all(
            item is not None and selected.start <= item <= index
            for item in history_indices
        )
        horizon_valid = all(
            item is not None and selected.start <= item < selected.end
            for item in target_indices
        )
        if not history_valid:
            missing_history += 1
        if not horizon_valid:
            missing_horizon += 1
        if not history_valid or not horizon_valid:
            if (
                selected.start + history_steps - 1
                <= index
                < selected.end - maximum_horizon
            ):
                timestamp_gap_origins += 1
            continue
        maturity_identity = {
            "dataset_id": dataset_id,
            "episode_id": episode_id,
            "origin_index": index,
            "origin_timestamp": origin_timestamp,
            "history_timestamps": [
                origin_timestamp - lag for lag in range(history_steps)
            ],
            "target_timestamps": [origin_timestamp + item for item in horizons],
        }
        maturity_digest = digest(maturity_identity)
        origin_id = digest(
            {
                "dataset_id": dataset_id,
                "episode_id": episode_id,
                "origin_index": index,
                "origin_timestamp": origin_timestamp,
                "maturity_digest": maturity_digest,
            }
        )
        origins.append(
            PlannedOrigin(
                origin_id=origin_id,
                dataset_id=dataset_id,
                episode_id=episode_id,
                origin_index=index,
                origin_timestamp=origin_timestamp,
                maximum_target_timestamp=origin_timestamp + maximum_horizon,
                maturity_digest=maturity_digest,
            )
        )
    gaps = {
        "history_ineligible_origins": missing_history,
        "horizon_ineligible_origins": missing_horizon,
        "timestamp_gap_origins": timestamp_gap_origins,
        "total_ineligible_origins": selected.size - len(origins),
    }
    return tuple(origins), gaps


def _capacity_error(
    *, required: int, available: int, schedule: OptimizationSchedule
) -> CohortCapacityError:
    remaining = max(0, available - schedule.formal_origin_count_per_finalist)
    per_generation = (
        schedule.screening_origin_count + schedule.selection_holdout_origin_count
    )
    return CohortCapacityError(
        required=required,
        available=available,
        max_generations=remaining // per_generation,
    )


def _cycled_origins(
    eligible: Sequence[PlannedOrigin], *, start: int, count: int
) -> tuple[PlannedOrigin, ...]:
    """Return a deterministic contiguous window, wrapping after exhaustion.

    The source origin identity remains unchanged when it is reused, while the
    occurrence index records the cycle. This keeps every repeated forecast
    vector independently addressable in checkpoints and audit events.
    """

    if not eligible:
        raise CohortCapacityError(
            required=max(1, count), available=0, max_generations=0
        )
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("origin window start must be a non-negative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("origin window count must be a positive integer")
    population = len(eligible)
    selected: list[PlannedOrigin] = []
    for offset in range(count):
        absolute = start + offset
        source = eligible[absolute % population]
        cycle = absolute // population
        selected.append(
            source
            if cycle == source.reuse_index
            else replace(source, reuse_index=cycle)
        )
    return tuple(selected)


def plan_run_adaptation_cohort(
    dataset: DatasetIdentityView,
    *,
    schedule: OptimizationSchedule,
    seed: int,
) -> RunAdaptationCohort:
    if not isinstance(schedule, OptimizationSchedule):
        raise TypeError("schedule must be OptimizationSchedule")
    _strict_integer(seed, "seed")
    eligible, _gaps = _eligible_origins(dataset)
    required = schedule.formal_origin_count_per_finalist
    selected = _cycled_origins(eligible, start=0, count=required)
    cohort = PlannedCohort(
        role="adaptation",
        origins=selected,
        maximum_horizon=max(DEFAULT_HORIZONS),
        shared_candidate_count=schedule.finalist_count,
    )
    batches = tuple(
        PlannedBatch(
            batch_index=index,
            cohort=PlannedCohort(
                role="adaptation_batch",
                origins=selected[
                    index * schedule.local_batch_origin_count :
                    (index + 1) * schedule.local_batch_origin_count
                ],
                maximum_horizon=max(DEFAULT_HORIZONS),
                shared_candidate_count=schedule.finalist_count,
            ),
        )
        for index in range(schedule.batch_count)
    )
    return RunAdaptationCohort(
        dataset_id=cohort.origins[0].dataset_id,
        episode_id=cohort.origins[0].episode_id,
        seed=seed,
        cohort=cohort,
        batches=batches,
    )


def plan_generation_selection_cohorts(
    dataset: DatasetIdentityView,
    *,
    schedule: OptimizationSchedule,
    generation: int,
    adaptation: RunAdaptationCohort,
    seed: int,
) -> GenerationCohorts:
    if not isinstance(schedule, OptimizationSchedule):
        raise TypeError("schedule must be OptimizationSchedule")
    if not isinstance(adaptation, RunAdaptationCohort):
        raise TypeError("adaptation must be RunAdaptationCohort")
    _strict_integer(generation, "generation")
    _strict_integer(seed, "seed")
    eligible, _gaps = _eligible_origins(dataset)
    if not eligible:
        raise _capacity_error(required=1, available=0, schedule=schedule)
    expected_adaptation = _cycled_origins(
        eligible, start=0, count=adaptation.origin_count
    )
    if (
        adaptation.dataset_id != eligible[0].dataset_id
        or adaptation.episode_id != eligible[0].episode_id
        or adaptation.origin_occurrence_keys
        != tuple((item.origin_id, item.reuse_index) for item in expected_adaptation)
    ):
        raise ValueError("adaptation cohort does not match dataset identity plan")
    per_generation = (
        schedule.screening_origin_count + schedule.selection_holdout_origin_count
    )
    start = adaptation.origin_count + generation * per_generation
    selected = _cycled_origins(eligible, start=start, count=per_generation)
    screening_end = schedule.screening_origin_count
    screening = PlannedCohort(
        role="screening",
        origins=selected[:screening_end],
        maximum_horizon=max(DEFAULT_HORIZONS),
        shared_candidate_count=4,
    )
    holdout = PlannedCohort(
        role="holdout",
        origins=selected[screening_end:],
        maximum_horizon=max(DEFAULT_HORIZONS),
        shared_arm_count=3,
    )
    return GenerationCohorts(
        dataset_id=adaptation.dataset_id,
        episode_id=adaptation.episode_id,
        generation=generation,
        seed=seed,
        adaptation_digest=adaptation.adaptation_digest,
        adaptation_batch_digests=adaptation.batch_digests,
        screening=screening,
        holdout=holdout,
    )


def estimate_epoch_capacity(
    dataset: DatasetIdentityView,
    *,
    schedule: OptimizationSchedule,
    planned_generations: int,
    seed: int,
    scoring_cells_per_origin: int = DEFAULT_SCORING_CELLS_PER_ORIGIN,
) -> CohortCapacityReport:
    if not isinstance(schedule, OptimizationSchedule):
        raise TypeError("schedule must be OptimizationSchedule")
    _strict_integer(planned_generations, "planned_generations", minimum=1)
    _strict_integer(seed, "seed")
    _strict_integer(
        scoring_cells_per_origin, "scoring_cells_per_origin", minimum=1
    )
    dataset_id, episode_id, _timestamps = _dataset_identity(dataset)
    eligible, gaps = _eligible_origins(dataset)
    selected = _selection_partition(dataset)
    required = schedule.required_unique_origins(planned_generations)
    # A non-empty eligible population can service any requested number of
    # rounds. The planner consumes each source origin once, then wraps with a
    # deterministic reuse index. Keep the requested budget as the executable
    # maximum and expose the reuse count for operator/audit visibility.
    max_generations = planned_generations if eligible else 0
    candidate_origins = schedule.generation_execution_budget(
        cells_per_origin=scoring_cells_per_origin
    )["total_candidate_origins"]
    return CohortCapacityReport(
        dataset_id=dataset_id,
        episode_id=episode_id,
        planned_generations=planned_generations,
        available_partition_origins=selected.size,
        available_eligible_origins=len(eligible),
        required_unique_origins=required,
        max_feasible_generations=max_generations,
        sufficient=bool(eligible),
        maturity_gaps=gaps,
        candidate_origin_executions_per_generation=candidate_origins,
        scoring_cells_per_generation=(
            candidate_origins * scoring_cells_per_origin
        ),
        candidate_origin_executions_for_run=(
            candidate_origins * planned_generations
        ),
        scoring_cells_for_run=(
            candidate_origins * planned_generations * scoring_cells_per_origin
        ),
        schedule_digest=digest(schedule.to_dict()),
        seed=seed,
        cohort_reuse_policy=COHORT_REUSE_POLICY,
        reused_origin_occurrences=max(0, required - len(eligible)),
    )


__all__ = [
    "CAPACITY_REPORT_SCHEMA",
    "COHORT_REUSE_POLICY",
    "COHORT_PLANNER_SCHEMA",
    "CohortCapacityError",
    "CohortCapacityReport",
    "GenerationCohorts",
    "PlannedBatch",
    "PlannedCohort",
    "PlannedOrigin",
    "RunAdaptationCohort",
    "estimate_epoch_capacity",
    "plan_generation_selection_cohorts",
    "plan_run_adaptation_cohort",
]
