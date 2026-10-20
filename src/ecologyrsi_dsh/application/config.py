"""JSON configuration loaders for the dependency-free local runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ..core.models import TaskManifest


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value




def load_task_manifest(path: str | Path) -> TaskManifest:
    task = TaskManifest.from_dict(load_json_object(path))
    if (task.metadata.get("execution_protocol") != "dsh_native_plugin_evolution@1"
            or task.metadata.get("optimization_protocol") != "joint_plugin_epoch@2"):
        raise ValueError("only the current native joint manifest is supported")
    return task


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
