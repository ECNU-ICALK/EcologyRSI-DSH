"""JSON configuration loaders for the dependency-free local runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ..core.models import TaskManifest
from ..data.toy import ToyCropSoilWater


_TOY_DOMAIN_IDS = {"crop-soil-water@toy", "crop_soil_water", "crop-soil-water"}
_TOY_DATASET_ALIASES = {
    "generated-toy-series@1": "generated-toy-series@1",
    # Kept only for manifests written by the first draft of v0.1.
    "toy-dataset@1": "generated-toy-series@1",
}


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def bind_toy_dataset(manifest: TaskManifest, *, required: bool = False) -> TaskManifest:
    """Bind and verify the deterministic toy dataset digest."""

    if manifest.domain_pack not in _TOY_DOMAIN_IDS:
        if required:
            raise ValueError(
                "local runtime only supports the crop-soil-water toy domain"
            )
        return manifest
    if len(manifest.visible_datasets) != 1:
        raise ValueError(
            "local crop-soil-water runtime requires exactly one visible dataset"
        )
    dataset_id = manifest.visible_datasets[0]
    canonical_dataset = _TOY_DATASET_ALIASES.get(dataset_id)
    if canonical_dataset is None:
        raise ValueError(
            "local runtime only supports dataset generated-toy-series@1"
        )
    dataset_seed = 0
    expected = ToyCropSoilWater(seed=dataset_seed).dataset_digest
    metadata = dict(manifest.metadata)
    existing = metadata.get("dataset_digest")
    if existing is not None and existing != expected:
        raise ValueError("manifest dataset_digest does not match the fixed seed-0 snapshot")
    metadata["dataset_digest"] = expected
    metadata["dataset_seed"] = dataset_seed
    metadata["episode_id"] = "generated-toy-series@1:seed-0"
    metadata["domain"] = "toy"
    metadata["scientific_scope"] = "prediction_demo_non_causal"
    metadata.setdefault("evaluation_partition", "validation")
    data = manifest.to_dict()
    data["visible_datasets"] = [canonical_dataset]
    data["metadata"] = metadata
    return TaskManifest.from_dict(data)


def load_task_manifest(path: str | Path) -> TaskManifest:
    # The local executable has exactly one evaluator; rejecting another domain
    # here prevents a manifest label from being silently scored by the toy.
    return bind_toy_dataset(TaskManifest.from_dict(load_json_object(path)), required=True)


@dataclass(frozen=True, slots=True)
class LocalConfig:
    db: str = "ecologyrsi-dsh.sqlite3"
    host: str = "127.0.0.1"
    port: int = 8765
    manifest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.db, str) or not self.db.strip():
            raise ValueError("config.db must be a non-empty string")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("config.host must be a non-empty string")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("config.port must be an integer between 1 and 65535")
        if self.manifest is not None and (not isinstance(self.manifest, str) or not self.manifest.strip()):
            raise ValueError("config.manifest must be a non-empty string or null")


def load_local_config(path: str | Path) -> LocalConfig:
    source = Path(path).expanduser().resolve()
    raw = load_json_object(source)
    unknown = set(raw) - {"db", "host", "port", "manifest"}
    if unknown:
        raise ValueError(f"unknown local config fields: {', '.join(sorted(unknown))}")
    data = dict(raw)
    for field_name in ("db", "manifest"):
        value = data.get(field_name)
        if value is not None:
            resolved = Path(str(value)).expanduser()
            if not resolved.is_absolute():
                resolved = source.parent / resolved
            data[field_name] = str(resolved.resolve())
    return LocalConfig(**data)
