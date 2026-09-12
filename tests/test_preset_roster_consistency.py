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


def _block(relative_path: str, opener: str, closer: str) -> str:
    """Return the literal body a roster is declared in, without evaluating the file."""
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    start = source.index(opener) + len(opener)
    return source[start : source.index(closer, start)]


def _ordered_ids(relative_path: str, opener: str, closer: str) -> tuple[str, ...]:
    return tuple(
        value
        for value in QUOTED.findall(_block(relative_path, opener, closer))
        if MANAGED_PRESET_ID.fullmatch(value)
    )


def _mentioned_ids(relative_path: str) -> frozenset[str]:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    return frozenset(MANAGED_PRESET_ID.findall(text))


class PresetRosterConsistencyTests(unittest.TestCase):
    """The manifest is the single source of truth; nine other sites restate it."""

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

    def test_ordered_rosters_match_the_manifest_order(self):
        # api/handler.py projects native capabilities in roster order, and the installer
        # verifies presets in roster order, so order is part of the contract.
        cases = {
            "src/ecologyrsi_dsh/api/handler.py": (
                "_DSH_NATIVE_PRESET_IDS = (", ")"),
            "scripts/verify_artifacts.py": (
                "CURRENT_DSH_PRESET_IDS = frozenset(", ")"),
            "scripts/install_dsh_ecology_runtime.mjs": (
                "PRESET_IDS = Object.freeze([", "]"),
        }
        for relative_path, (opener, closer) in cases.items():
            with self.subTest(path=relative_path):
                self.assertEqual(_ordered_ids(relative_path, opener, closer), self.roster)

    def test_delivery_manifests_mention_exactly_the_current_roster(self):
        # These two must stay literal: pyproject data-files map install targets and
        # verify_delivery.sh checks a fixed file list. Assert the set, not the order.
        for relative_path in ("pyproject.toml", "scripts/verify_delivery.sh"):
            with self.subTest(path=relative_path):
                self.assertEqual(_mentioned_ids(relative_path), frozenset(self.roster))

    def test_role_scoped_rosters_are_subsets_of_the_manifest(self):
        # The canaries and the program registry only bind the roles they execute.
        for relative_path in (
            "src/ecologyrsi_dsh/integrations/model_canary.py",
            "integrations/dsh_ecology_plugin/lib/runtime/model-canary.js",
            "src/ecologyrsi_dsh/knowledge/program_registry.py",
        ):
            with self.subTest(path=relative_path):
                mentioned = _mentioned_ids(relative_path)
                self.assertTrue(mentioned, "no managed preset id found; the parser drifted")
                self.assertLessEqual(mentioned, frozenset(self.roster))

    def test_both_model_canaries_bind_the_same_preset_per_stage(self):
        # The Python receipt builder and the Node verifier are hand-aligned; if they
        # disagree the canary rejects a legitimately frozen identity at run time.
        python_block = _block(
            "src/ecologyrsi_dsh/integrations/model_canary.py", "_ROLES = (", "\n)")
        python_stages = {
            stage: preset
            for _route, _digest, _role, preset, stage, _schema in re.findall(
                r"\(" + ", ".join([r'"([^"]+)"'] * 6) + r"\)", python_block)
        }
        node_block = _block(
            "integrations/dsh_ecology_plugin/lib/runtime/model-canary.js",
            "const PRESETS = Object.freeze({", "})")
        node_stages = dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', node_block))
        self.assertEqual(python_stages, node_stages)
        self.assertTrue(python_stages, "no canary stage bindings found; the parser drifted")
        self.assertLessEqual(frozenset(python_stages.values()), frozenset(self.roster))


if __name__ == "__main__":
    unittest.main()
