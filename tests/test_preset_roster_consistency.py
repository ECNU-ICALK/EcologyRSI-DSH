"""Every DSH preset roster in the tree agrees with the one preset manifest."""
from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PRESET_ROOT = ROOT / "integrations" / "dsh_ecology_plugin" / "presets"
MANIFEST_PATH = PRESET_ROOT / "preset-manifest.json"

# Mirrors the managed-preset patterns already enforced by scripts/verify_artifacts.py
# and scripts/install_dsh_ecology_runtime.mjs, so a retired role still gets noticed.
MANAGED_PRESET_ID = re.compile(
    r"ecology-(?:coordinator|researcher|candidate-proposer|sample-planner"
    r"|sample-critic|generation-judge|local-editor)-v[0-9]+"
)
QUOTED = re.compile(r"[\"']([^\"']+)[\"']")


def _mentioned_ids(relative_path: str) -> frozenset[str]:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    return frozenset(MANAGED_PRESET_ID.findall(text))


class PresetRosterConsistencyTests(unittest.TestCase):
    """Contract consumers and shipped resources agree with the shared manifest."""

    def setUp(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.roster = tuple(entry["preset_id"] for entry in manifest["presets"])
        self.assertEqual(len(self.roster), len(set(self.roster)), "manifest repeats a preset id")
        for preset_id in self.roster:
            self.assertRegex(preset_id, MANAGED_PRESET_ID)

    def test_installed_preset_directories_match_the_manifest(self):
        # A rename that leaves the retired directory behind (v8 next to v9) ships two
        # rosters in one archive; the manifest and the tree must agree exactly.
        directories = sorted(entry.name for entry in PRESET_ROOT.iterdir() if entry.is_dir())
        self.assertEqual(directories, sorted(self.roster))
        for preset_id in self.roster:
            self.assertTrue((PRESET_ROOT / preset_id / "preset.yml").is_file(), preset_id)
            self.assertTrue((PRESET_ROOT / preset_id / "agent.cordis.yml").is_file(), preset_id)

    def test_python_consumers_read_the_shared_contract(self):
        from ecologyrsi_dsh.api.handler import _DSH_NATIVE_PRESET_IDS
        from ecologyrsi_dsh.integrations.role_contracts import PRESET_IDS
        self.assertEqual(PRESET_IDS, self.roster)
        self.assertEqual(_DSH_NATIVE_PRESET_IDS, PRESET_IDS)

    def test_delivery_manifests_mention_exactly_the_current_roster(self):
        # These two must stay literal: pyproject data-files map install targets and
        # verify_delivery.sh checks a fixed file list. Assert the set, not the order.
        for relative_path in ("pyproject.toml", "scripts/verify_delivery.sh"):
            with self.subTest(path=relative_path):
                self.assertEqual(_mentioned_ids(relative_path), frozenset(self.roster))

    def test_new_run_seed_bindings_use_current_presets(self):
        from ecologyrsi_dsh.api.handler import _DSH_NATIVE_SEED_TEMPLATE_BY_PREDICTOR
        from ecologyrsi_dsh.knowledge.program_registry import current_program_registry

        registry = current_program_registry()
        for template_id in _DSH_NATIVE_SEED_TEMPLATE_BY_PREDICTOR.values():
            with self.subTest(template=template_id):
                self.assertTrue(template_id.endswith("@2"))
                template = registry.seed_template(template_id).to_dict()
                profiles = template["agent_program"]["candidate_execution_program"]["role_profiles"]
                self.assertTrue(profiles)
                self.assertLessEqual({item["preset_id"] for item in profiles}, set(self.roster))
        # The sole historical reference belongs to the byte-preserved @1 seeds.
        # Active bindings above must never select it.
        mentioned = _mentioned_ids("src/ecologyrsi_dsh/knowledge/program_registry.py")
        self.assertEqual(mentioned - set(self.roster), {"ecology-sample-planner-v11"})

    def test_node_installer_stages_and_python_canaries_share_the_contract(self):
        import subprocess
        from ecologyrsi_dsh.integrations.role_contracts import CANARY_ROLES, STAGES
        result = subprocess.run(["node", "--input-type=module", "-e", """
            import { PRESET_IDS } from './scripts/install_dsh_ecology_runtime.mjs';
            import { CANARY_PRESETS, STAGE_CONTRACTS } from './integrations/dsh_ecology_plugin/lib/runtime/contracts.js';
            console.log(JSON.stringify({presets: PRESET_IDS, canaries: CANARY_PRESETS, stages: STAGE_CONTRACTS}));
        """], cwd=ROOT, check=True, text=True, capture_output=True)
        node = json.loads(result.stdout)
        self.assertEqual(node["presets"], list(self.roster))
        self.assertEqual(node["stages"], STAGES)
        self.assertEqual(node["canaries"], {row[4]: row[3] for row in CANARY_ROLES})
        for stage in STAGES.values():
            schema = json.loads((PRESET_ROOT.parent / "schemas" / (stage["file"] + ".schema.json")).read_text())
            self.assertEqual(schema["$id"], stage["schema"])

    def test_mutation_schemas_match_the_host_operation_catalog(self):
        from ecologyrsi_dsh.evolution.mutation_specs import MUTATION_SPECS
        for name in ("genome-mutation", "local-edit"):
            schema = json.loads((PRESET_ROOT.parent / "schemas" / (name + ".schema.json")).read_text())
            declared = {}
            for variant in schema["properties"]["operations"]["items"]["oneOf"]:
                op = variant["properties"]["op"]
                for name in op.get("enum", [op.get("const")]):
                    declared[name] = set(variant["required"]) - {"op"}
            self.assertEqual(declared, {op: set(spec.fields.split()) for op, spec in MUTATION_SPECS.items()})


if __name__ == "__main__":
    unittest.main()
