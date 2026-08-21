"""Immutable time-forward dataset splits for local ecological datasets."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Protocol, Sequence


class EpisodeLike(Protocol):
    episode_id: str
    timestamps: Sequence[int]
    content_sha256: str


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class IndexRange:
    """A half-open row range within one canonical episode."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int) or self.start < 0:
            raise ValueError("split start must be a non-negative integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int) or self.end < self.start:
            raise ValueError("split end must be an integer not smaller than start")

    @property
    def size(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end, "size": self.size}


@dataclass(frozen=True, slots=True)
class EpisodeSplit:
    dataset_id: str
    episode_id: str
    role: str
    row_count: int
    training_fit: IndexRange
    training_feedback: IndexRange
    development: IndexRange
    gate: IndexRange
    content_sha256: str

    def range_for(self, partition: str) -> IndexRange:
        if partition == "training_fit":
            return self.training_fit
        if partition == "training_feedback":
            return self.training_feedback
        if partition == "development":
            return self.development
        if partition == "gate":
            return self.gate
        raise KeyError(f"unknown partition: {partition}")

    def to_dict(self, *, include_restricted: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "episode_id": self.episode_id,
            "role": self.role,
            "row_count": self.row_count,
            "training_fit": self.training_fit.to_dict(),
            "training_feedback": self.training_feedback.to_dict(),
            "development": self.development.to_dict(),
            "content_sha256": self.content_sha256,
        }
        if include_restricted:
            value["gate"] = self.gate.to_dict()
        return value


@dataclass(frozen=True, slots=True)
class SplitManifest:
    schema_version: str
    split_policy_version: str
    dataset_id: str
    dataset_digest_sha256: str
    split_manifest_digest_sha256: str
    train_fraction: float
    training_feedback_fraction: float
    development_fraction: float
    training_feedback_embargo_hours: int
    embargo_hours: int
    episode_records: tuple[EpisodeSplit, ...]

    def split_for(self, episode_id: str) -> EpisodeSplit:
        for record in self.episode_records:
            if record.episode_id == episode_id:
                return record
        raise KeyError(f"episode is absent from split manifest: {episode_id}")

    def to_dict(self, *, include_restricted: bool = False) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_policy_version": self.split_policy_version,
            "dataset_id": self.dataset_id,
            "dataset_digest_sha256": self.dataset_digest_sha256,
            "split_manifest_digest_sha256": self.split_manifest_digest_sha256,
            "train_fraction": self.train_fraction,
            "training_feedback_fraction": self.training_feedback_fraction,
            "development_fraction": self.development_fraction,
            "training_feedback_embargo_hours": self.training_feedback_embargo_hours,
            "embargo_hours": self.embargo_hours,
            "episode_records": [
                item.to_dict(include_restricted=include_restricted)
                for item in self.episode_records
            ],
        }


FOUR_STAGE_PROTOCOL_VERSION = "time-forward-four-stage@2"
FOUR_STAGE_SCHEMA_VERSION = "ecologyrsi-dsh.data-protocol/2"
SOURCE_TIMEZONE = "unspecified-naive-local"
CALENDAR_ENCODING = "excel-serial-hour-fixed-24h@1"


@dataclass(frozen=True, slots=True)
class FourStageDataProtocol:
    """Frozen stage ranges; every range is half-open by target timestamp."""

    dataset_id: str
    episode_id: str
    dataset_digest: str
    split_manifest_digest: str
    content_sha256: str
    calibration_fit: IndexRange
    fit_to_uq_embargo: IndexRange
    calibration_uq: IndexRange
    uq_to_selection_embargo: IndexRange
    model_selection: IndexRange
    validation: IndexRange
    final_test: IndexRange
    partition_digests: dict[str, str]
    partition_timestamp_bounds: dict[str, tuple[int, int]]
    objective_grid: dict[str, Any]
    protocol_digest: str
    schema_version: str = FOUR_STAGE_SCHEMA_VERSION
    protocol_version: str = FOUR_STAGE_PROTOCOL_VERSION
    source_timezone: str = SOURCE_TIMEZONE
    calendar_encoding: str = CALENDAR_ENCODING

    def range_for(self, stage: str) -> IndexRange:
        value = getattr(self, str(stage), None)
        if not isinstance(value, IndexRange):
            raise KeyError(f"unknown data protocol stage: {stage}")
        return value

    def contains_target(self, stage: str, target_timestamp: int) -> bool:
        if isinstance(target_timestamp, bool) or not isinstance(target_timestamp, int):
            raise TypeError("target_timestamp must be an integer")
        partition = self.range_for(stage)
        del partition
        try:
            start_timestamp, end_timestamp = self.partition_timestamp_bounds[stage]
        except KeyError:
            raise KeyError(f"unknown data protocol stage: {stage}") from None
        return start_timestamp <= target_timestamp < end_timestamp

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "source_timezone": self.source_timezone,
            "calendar_encoding": self.calendar_encoding,
            "dataset_id": self.dataset_id,
            "episode_id": self.episode_id,
            "dataset_digest": self.dataset_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "content_sha256": self.content_sha256,
            "partitions": {
                name: self.range_for(name).to_dict()
                for name in (
                    "calibration_fit",
                    "fit_to_uq_embargo",
                    "calibration_uq",
                    "uq_to_selection_embargo",
                    "model_selection",
                    "validation",
                    "final_test",
                )
            },
            "partition_digests": dict(self.partition_digests),
            "partition_timestamp_bounds": {
                name: {"start": bounds[0], "end": bounds[1]}
                for name, bounds in self.partition_timestamp_bounds.items()
            },
            "objective_grid": dict(self.objective_grid),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_dict(), "protocol_digest": self.protocol_digest}


def _partition_digest(
    name: str,
    partition: IndexRange,
    timestamps: Sequence[int],
    *,
    dataset_digest: str,
    split_manifest_digest: str,
) -> str:
    return _digest(
        {
            "name": name,
            "range": partition.to_dict(),
            "first_target_timestamp": (
                timestamps[partition.start] if partition.size else None
            ),
            "last_target_timestamp": (
                timestamps[partition.end - 1] if partition.size else None
            ),
            "dataset_digest": dataset_digest,
            "split_manifest_digest": split_manifest_digest,
        }
    )


def build_four_stage_data_protocol(
    dataset_id: str,
    episode: EpisodeLike,
    legacy_split: EpisodeSplit,
    *,
    dataset_digest: str,
    split_manifest_digest: str,
    target_names: Sequence[str],
    horizons: Sequence[int],
    history_steps: int,
) -> FourStageDataProtocol:
    """Derive the literal @2 stage protocol without optimizing its cut points."""

    if legacy_split.role != "optimization":
        raise PermissionError("external episodes cannot become evolution episodes")
    timestamps = tuple(int(item) for item in episode.timestamps)
    if len(timestamps) != legacy_split.row_count:
        raise ValueError("episode and split row counts differ")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("episode timestamps must increase strictly")
    if not target_names or not horizons:
        raise ValueError("data protocol objective grid cannot be empty")
    if isinstance(history_steps, bool) or not isinstance(history_steps, int) or history_steps < 1:
        raise ValueError("history_steps must be positive")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in horizons):
        raise ValueError("horizons must be positive integers")
    a, b = legacy_split.training_fit.start, legacy_split.training_fit.end
    # Integer arithmetic implements floor(70/100*n) without binary-float drift.
    raw_cut = a + ((b - a) * 70) // 100
    if not a < raw_cut < b:
        raise ValueError("calibration split is insufficient")
    uq_start = bisect_left(timestamps, timestamps[raw_cut - 1] + 24)
    if uq_start >= b:
        raise ValueError("calibration-UQ is insufficient after the 24h embargo")
    selection_boundary = bisect_left(timestamps, timestamps[b - 1] + 24)
    selection_start = max(legacy_split.training_feedback.start, selection_boundary)
    if selection_start >= legacy_split.training_feedback.end:
        raise ValueError("model-selection is insufficient after the 24h embargo")
    ranges = {
        "calibration_fit": IndexRange(a, raw_cut),
        "fit_to_uq_embargo": IndexRange(raw_cut, uq_start),
        "calibration_uq": IndexRange(uq_start, b),
        "uq_to_selection_embargo": IndexRange(b, selection_start),
        "model_selection": IndexRange(
            selection_start, legacy_split.training_feedback.end
        ),
        "validation": legacy_split.development,
        "final_test": legacy_split.gate,
    }
    maximum_horizon = max(horizons)
    fit_eligible = max(0, ranges["calibration_fit"].size - history_steps - maximum_horizon + 1)
    uq_eligible = ranges["calibration_uq"].size
    fit_blocks = len(
        {
            timestamps[index] // 24
            for index in range(
                min(ranges["calibration_fit"].end, ranges["calibration_fit"].start + history_steps + maximum_horizon),
                ranges["calibration_fit"].end,
            )
        }
    )
    uq_blocks = len(
        {
            timestamps[index] // 24
            for index in range(ranges["calibration_uq"].start, ranges["calibration_uq"].end)
        }
    )
    if fit_eligible < 80 or fit_blocks < 14:
        raise ValueError("calibration-fit evidence is insufficient")
    if uq_eligible < 40 or uq_blocks < 8:
        raise ValueError("calibration-UQ evidence is insufficient")
    partition_digests = {
        name: _partition_digest(
            name,
            partition,
            timestamps,
            dataset_digest=dataset_digest,
            split_manifest_digest=split_manifest_digest,
        )
        for name, partition in ranges.items()
    }
    partition_timestamp_bounds = {
        name: (
            timestamps[partition.start],
            (
                timestamps[partition.end]
                if partition.end < len(timestamps)
                else timestamps[-1] + 1
            ),
        )
        for name, partition in ranges.items()
    }
    objective_grid = {
        "targets": list(target_names),
        "horizons": list(horizons),
        "history_steps": history_steps,
        "calibration_fit_eligible_per_cell": fit_eligible,
        "calibration_fit_day_blocks_per_cell": fit_blocks,
        "calibration_uq_eligible_per_cell": uq_eligible,
        "calibration_uq_day_blocks_per_cell": uq_blocks,
    }
    identity = {
        "schema_version": FOUR_STAGE_SCHEMA_VERSION,
        "protocol_version": FOUR_STAGE_PROTOCOL_VERSION,
        "source_timezone": SOURCE_TIMEZONE,
        "calendar_encoding": CALENDAR_ENCODING,
        "dataset_id": dataset_id,
        "episode_id": episode.episode_id,
        "dataset_digest": dataset_digest,
        "split_manifest_digest": split_manifest_digest,
        "content_sha256": episode.content_sha256,
        "partitions": {name: item.to_dict() for name, item in ranges.items()},
        "partition_digests": partition_digests,
        "partition_timestamp_bounds": {
            name: {"start": bounds[0], "end": bounds[1]}
            for name, bounds in partition_timestamp_bounds.items()
        },
        "objective_grid": objective_grid,
    }
    return FourStageDataProtocol(
        dataset_id=dataset_id,
        episode_id=episode.episode_id,
        dataset_digest=dataset_digest,
        split_manifest_digest=split_manifest_digest,
        content_sha256=episode.content_sha256,
        calibration_fit=ranges["calibration_fit"],
        fit_to_uq_embargo=ranges["fit_to_uq_embargo"],
        calibration_uq=ranges["calibration_uq"],
        uq_to_selection_embargo=ranges["uq_to_selection_embargo"],
        model_selection=ranges["model_selection"],
        validation=ranges["validation"],
        final_test=ranges["final_test"],
        partition_digests=partition_digests,
        partition_timestamp_bounds=partition_timestamp_bounds,
        objective_grid=objective_grid,
        protocol_digest=_digest(identity),
    )


def build_split_manifest(
    dataset_id: str,
    episodes: Iterable[EpisodeLike],
    *,
    train_fraction: float = 0.6,
    training_feedback_fraction: float = 0.5,
    development_fraction: float = 0.2,
    training_feedback_embargo_hours: int = 1,
    embargo_hours: int = 24,
    external_episode_patterns: tuple[str, ...] = ("reference",),
    split_policy_version: str = "time-forward-embargo/1",
) -> SplitManifest:
    """Build deterministic fit/feedback/development/gate row ranges.

    Reference episodes remain present in the immutable manifest but carry the
    ``external_holdout`` role. Callers must not expose their rows through an
    adaptive or sample-browsing surface.
    """

    items = tuple(episodes)
    if not items:
        raise ValueError("cannot split an empty dataset")
    if not dataset_id or not isinstance(dataset_id, str):
        raise ValueError("dataset_id must be a non-empty string")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if not 0 < training_feedback_fraction < 1:
        raise ValueError("training_feedback_fraction must be between zero and one")
    if not 0 < development_fraction < 1 or train_fraction + development_fraction >= 1:
        raise ValueError("train and development fractions must leave a gate tail")
    for name, value in (
        ("training_feedback_embargo_hours", training_feedback_embargo_hours),
        ("embargo_hours", embargo_hours),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    normalized_patterns = tuple(item.casefold() for item in external_episode_patterns if item)
    records: list[EpisodeSplit] = []
    seen: set[str] = set()
    for episode in sorted(items, key=lambda item: item.episode_id):
        if episode.episode_id in seen:
            raise ValueError(f"duplicate episode_id: {episode.episode_id}")
        seen.add(episode.episode_id)
        timestamps = tuple(int(item) for item in episode.timestamps)
        if not timestamps:
            raise ValueError(f"episode is empty: {episode.episode_id}")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError(f"episode timestamps must increase strictly: {episode.episode_id}")

        row_count = len(timestamps)
        train_end = int(row_count * train_fraction)
        development_end = int(row_count * (train_fraction + development_fraction))
        training_fit_end = int(train_end * (1 - training_feedback_fraction))
        if train_end >= 2:
            training_fit_end = min(max(training_fit_end, 1), train_end - 1)
        else:
            training_fit_end = max(0, min(training_fit_end, train_end))
        feedback_start = bisect_left(
            timestamps,
            timestamps[training_fit_end] + training_feedback_embargo_hours,
        )
        development_start = bisect_left(
            timestamps,
            timestamps[train_end] + embargo_hours,
        )
        gate_start = bisect_left(
            timestamps,
            timestamps[development_end] + embargo_hours,
        )
        role = (
            "external_holdout"
            if any(pattern in episode.episode_id.casefold() for pattern in normalized_patterns)
            else "optimization"
        )
        records.append(
            EpisodeSplit(
                dataset_id=dataset_id,
                episode_id=episode.episode_id,
                role=role,
                row_count=row_count,
                training_fit=IndexRange(0, training_fit_end),
                training_feedback=IndexRange(min(feedback_start, train_end), train_end),
                development=IndexRange(min(development_start, development_end), development_end),
                gate=IndexRange(min(gate_start, row_count), row_count),
                content_sha256=episode.content_sha256,
            )
        )

    dataset_digest = _digest(
        [{"episode_id": item.episode_id, "content_sha256": item.content_sha256} for item in records]
    )
    identity = {
        "schema_version": "ecologyrsi-dsh.split-manifest/1",
        "split_policy_version": split_policy_version,
        "dataset_id": dataset_id,
        "dataset_digest_sha256": dataset_digest,
        "train_fraction": train_fraction,
        "training_feedback_fraction": training_feedback_fraction,
        "development_fraction": development_fraction,
        "training_feedback_embargo_hours": training_feedback_embargo_hours,
        "embargo_hours": embargo_hours,
        "episode_records": [item.to_dict(include_restricted=True) for item in records],
    }
    return SplitManifest(
        schema_version=str(identity["schema_version"]),
        split_policy_version=split_policy_version,
        dataset_id=dataset_id,
        dataset_digest_sha256=dataset_digest,
        split_manifest_digest_sha256=_digest(identity),
        train_fraction=train_fraction,
        training_feedback_fraction=training_feedback_fraction,
        development_fraction=development_fraction,
        training_feedback_embargo_hours=training_feedback_embargo_hours,
        embargo_hours=embargo_hours,
        episode_records=tuple(records),
    )


__all__ = [
    "EpisodeSplit",
    "FourStageDataProtocol",
    "IndexRange",
    "SplitManifest",
    "build_four_stage_data_protocol",
    "build_split_manifest",
]
