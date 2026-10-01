"""Current role identities, shared with the delivered native plugin."""
from __future__ import annotations

import json
from pathlib import Path
import sysconfig

_RELATIVE = Path("integrations/dsh_ecology_plugin/presets/preset-manifest.json")


def _load_manifest():
    roots = (Path(__file__).resolve().parents[3],
             Path(sysconfig.get_path("data")) / "share/ecologyrsi-dsh")
    path = next((root / _RELATIVE for root in roots if (root / _RELATIVE).is_file()), None)
    if path is None:
        raise RuntimeError("the native role contract is missing from the installation")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "ecologyrsi-dsh.preset-manifest/1":
        raise ValueError("unsupported native role contract")
    return manifest


MANIFEST = _load_manifest()
ROLES = {item["role"]: item for item in MANIFEST["presets"]}
STAGES = MANIFEST["stages"]
PRESET_IDS = tuple(item["preset_id"] for item in MANIFEST["presets"])
CANARY_ROLES = tuple(
    (
        "review_model_id" if ROLES[item["role"]]["model_route"] == "review" else "strategy_model_id",
        f"resolved_{ROLES[item['role']]['model_route']}_route_config_digest",
        item["role"], ROLES[item["role"]]["preset_id"], stage, item["schema"],
    )
    for stage in MANIFEST["canary_stages"] for item in (STAGES[stage],)
)
