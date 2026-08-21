"""Small dataset catalogue, registry, readiness, and safe sample surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .greenhouse import FeatureSpec
from .splits import FourStageDataProtocol, IndexRange
from ..core.models import digest


_VISIBLE_PARTITIONS = ("training_fit", "training_feedback")
_RESTRICTED_PARTITIONS = frozenset(
    {"development", "gate", "external", "external_holdout", "hidden", "test", "final"}
)
_TOY_DATASET_ID = "generated-toy-series@1"
_PREPARABLE_DATASET_IDS = frozenset({"agc_cucumber_2018", "agc_tomato_2019"})


def _text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"dataset descriptor {name} must be a list of strings")
    return tuple(item.strip() for item in value)


def _mapping_sequence(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"dataset descriptor {name} must be a list of objects")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    dataset_id: str
    display_name_zh: str
    domain_id: str
    adapter_id: str
    title: str
    subject: str
    edition: str
    version: str
    publisher: str
    doi: str
    landing_page: str
    license: str
    modalities: tuple[str, ...]
    runnable: bool
    required_globs: tuple[str, ...]
    source_files: tuple[Mapping[str, Any], ...]
    notes_zh: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetDescriptor":
        required = {
            "dataset_id",
            "display_name_zh",
            "domain_id",
            "adapter_id",
            "title",
            "subject",
            "edition",
            "version",
            "publisher",
            "doi",
            "landing_page",
            "license",
            "modalities",
            "runnable",
            "required_globs",
            "source_files",
            "notes_zh",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"dataset descriptor is missing fields: {', '.join(missing)}")
        text_fields = required.difference(
            {"modalities", "runnable", "required_globs", "source_files", "notes_zh"}
        )
        for name in text_fields:
            if not isinstance(value[name], str) or not str(value[name]).strip():
                raise ValueError(f"dataset descriptor {name} must be a non-empty string")
        if not isinstance(value["runnable"], bool):
            raise TypeError("dataset descriptor runnable must be a bool")
        return cls(
            **{name: str(value[name]).strip() for name in text_fields},
            modalities=_text_tuple(value["modalities"], "modalities"),
            runnable=value["runnable"],
            required_globs=_text_tuple(value["required_globs"], "required_globs"),
            source_files=tuple(dict(item) for item in _mapping_sequence(value["source_files"], "source_files")),
            notes_zh=_text_tuple(value["notes_zh"], "notes_zh"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "display_name_zh": self.display_name_zh,
            "domain_id": self.domain_id,
            "adapter_id": self.adapter_id,
            "title": self.title,
            "subject": self.subject,
            "edition": self.edition,
            "version": self.version,
            "publisher": self.publisher,
            "doi": self.doi,
            "landing_page": self.landing_page,
            "license": self.license,
            "modalities": list(self.modalities),
            "runnable": self.runnable,
            "required_globs": list(self.required_globs),
            "source_files": [dict(item) for item in self.source_files],
            "notes_zh": list(self.notes_zh),
        }


@dataclass(frozen=True, slots=True)
class DatasetSeries:
    """One non-restricted episode prepared for evaluator integration."""

    schema: str
    dataset_id: str
    domain_id: str
    episode_id: str
    digest: str
    timestamps: tuple[int, ...]
    values: Mapping[str, tuple[float | None, ...]]
    partitions: Mapping[str, IndexRange]
    features: Mapping[str, FeatureSpec]
    split_manifest_digest_sha256: str
    evaluation_partition: str = "training_feedback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "domain_id": self.domain_id,
            "episode_id": self.episode_id,
            "digest": self.digest,
            "timestamps": list(self.timestamps),
            "values": {name: list(items) for name, items in self.values.items()},
            "partitions": {name: item.to_dict() for name, item in self.partitions.items()},
            "features": {name: item.to_dict() for name, item in self.features.items()},
            "split_manifest_digest_sha256": self.split_manifest_digest_sha256,
            "evaluation_partition": self.evaluation_partition,
        }


@dataclass(frozen=True, slots=True)
class SelectionDatasetView:
    """Typed adaptive view with no validation, gate, or external rows."""

    schema: str
    dataset_id: str
    domain_id: str
    episode_id: str
    dataset_digest: str
    split_manifest_digest_sha256: str
    data_protocol_digest: str
    timestamps: tuple[int, ...]
    values: Mapping[str, tuple[float | None, ...]]
    partitions: Mapping[str, IndexRange]
    features: Mapping[str, FeatureSpec]
    selection_view_digest: str
    evaluation_partition: str = "training_feedback"

    @property
    def digest(self) -> str:
        return self.dataset_digest

    @classmethod
    def from_series(
        cls,
        series: DatasetSeries,
        protocol: FourStageDataProtocol,
    ) -> "SelectionDatasetView":
        if series.dataset_id != protocol.dataset_id or series.episode_id != protocol.episode_id:
            raise ValueError("selection view protocol belongs to another episode")
        if series.split_manifest_digest_sha256 != protocol.split_manifest_digest:
            raise ValueError("selection view split binding mismatch")
        visible_end = protocol.model_selection.end
        if len(series.timestamps) < visible_end:
            raise ValueError("dataset series does not contain the selection range")
        timestamps = tuple(series.timestamps[:visible_end])
        values = {
            name: tuple(items[:visible_end]) for name, items in series.values.items()
        }
        partitions = {
            "calibration_fit": protocol.calibration_fit,
            "calibration_uq": protocol.calibration_uq,
            "model_selection": protocol.model_selection,
            # Compatibility aliases are constrained to the new stage ranges.
            "training_fit": protocol.calibration_fit,
            "training_feedback": protocol.model_selection,
        }
        identity = {
            "schema": "ecologyrsi-dsh.selection-dataset-view/1",
            "dataset_id": series.dataset_id,
            "domain_id": series.domain_id,
            "episode_id": series.episode_id,
            "dataset_digest": series.digest,
            "split_manifest_digest_sha256": series.split_manifest_digest_sha256,
            "data_protocol_digest": protocol.protocol_digest,
            "timestamps": list(timestamps),
            "values": {name: list(items) for name, items in values.items()},
            "partitions": {name: item.to_dict() for name, item in partitions.items()},
        }
        return cls(
            schema=str(identity["schema"]),
            dataset_id=series.dataset_id,
            domain_id=series.domain_id,
            episode_id=series.episode_id,
            dataset_digest=series.digest,
            split_manifest_digest_sha256=series.split_manifest_digest_sha256,
            data_protocol_digest=protocol.protocol_digest,
            timestamps=timestamps,
            values=values,
            partitions=partitions,
            features=series.features,
            selection_view_digest=digest(identity),
        )


__all__ = ["DatasetDescriptor", "DatasetSeries", "SelectionDatasetView"]
