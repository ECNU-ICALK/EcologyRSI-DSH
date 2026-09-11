"""Load versioned, data-only dataset definitions from the shipped catalog."""
from __future__ import annotations

import json
from pathlib import Path
import sysconfig
from typing import Any


def default_catalog_path() -> Path:
    candidates = (
        Path(__file__).resolve().parents[3] / "datasets" / "autonomous_greenhouse.json",
        Path.cwd() / "datasets" / "autonomous_greenhouse.json",
        Path(sysconfig.get_path("data")) / "share/ecologyrsi-dsh/datasets/autonomous_greenhouse.json",
    )
    return next((p.resolve() for p in candidates if p.is_file()), candidates[0].resolve())


def load_dataset_definitions(catalog_path: Path | None = None) -> dict[str, dict[str, Any]]:
    catalog_path = catalog_path or default_catalog_path()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    definitions = {}
    for entry in catalog["datasets"]:
        name = entry.get("adapter_definition")
        if not name:
            if entry.get("runnable"):
                raise ValueError(f"runnable dataset needs adapter_definition: {entry['dataset_id']}")
            continue
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".json"):
            raise ValueError("adapter_definition must be a JSON filename in the catalog directory")
        path = (catalog_path.parent / name).resolve()
        if path.parent != catalog_path.parent.resolve():
            raise ValueError("adapter_definition must stay in the catalog directory")
        definition = json.loads(path.read_text(encoding="utf-8"))
        _validate_definition(definition)
        task = definition["prediction_task"]
        if any(task[key] != entry[key] for key in ("dataset_id", "domain_id", "adapter_id")):
            raise ValueError("dataset definition identity differs from catalog")
        if entry["dataset_id"] in definitions:
            raise ValueError("duplicate dataset definition")
        definitions[entry["dataset_id"]] = definition
    return definitions


def _validate_definition(definition: dict[str, Any]) -> None:
    if definition.get("schema_version") != "ecologyrsi-dsh.dataset-definition/1":
        raise ValueError("unsupported dataset definition schema")
    task = definition["prediction_task"]
    if any(definition[key] != task[key] for key in ("dataset_id", "domain_id")):
        raise ValueError("dataset definition identity differs from prediction task")
    features = definition["features"]
    names = [f["name"] for f in features]
    if not names or len(names) != len(set(names)):
        raise ValueError("dataset features must be nonempty and unique")
    for feature in features:
        if any(not isinstance(feature[k], str) or not feature[k] for k in ("name", "unit", "role", "display_name_zh")):
            raise ValueError("dataset features require names, units and roles")
    tables = definition["tables"]
    if not tables or tables[0]["filename"] != task["climate_filename"]:
        raise ValueError("first dataset table must be the primary climate table")
    allowed = {"filename", "features", "timestamp_aliases", "source_interval", "availability_delay_hours", "scope", "allow_suffix", "week_year"}
    for table in tables:
        filename = Path(table["filename"])
        if filename.is_absolute() or ".." in filename.parts or set(table) - allowed:
            raise ValueError("invalid dataset table definition")
        if table.get("scope", "team") not in ("team", "dataset"):
            raise ValueError("invalid dataset table scope")
        if table["source_interval"] not in ("5_minutes", "daily", "weekly", "biweekly", "harvest_event"):
            raise ValueError("unsupported source interval")
        delay = table["availability_delay_hours"]
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
            raise ValueError("invalid observation availability delay")
        if not table["timestamp_aliases"] or not set(table["features"]) <= set(names):
            raise ValueError("dataset table references unknown features or no timestamp")
    by_name = {f["name"]: f for f in features}
    for target in task["targets"]:
        if target["name"] not in by_name or target["unit"] != by_name[target["name"]]["unit"]:
            raise ValueError("prediction target missing or unit mismatch in definition")
        sources = [t for t in tables if target["name"] in t["features"]]
        if not sources or any(t["source_interval"] != "5_minutes" or t["availability_delay_hours"] for t in sources):
            raise ValueError("hourly evaluator requires high-frequency prediction labels")


# Definitions are fixed for the lifetime of a worker. Restarting loads edits;
# the complete definition digest prevents resuming a run under changed rules.
DATASET_DEFINITIONS = load_dataset_definitions()


def definition_for_domain(domain_id: str) -> dict[str, Any]:
    for definition in DATASET_DEFINITIONS.values():
        if definition["domain_id"] == domain_id:
            return definition
    raise ValueError(f"unsupported greenhouse domain: {domain_id}")
