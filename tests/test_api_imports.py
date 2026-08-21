from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


class APIImportIsolationTests(unittest.TestCase):
    def _fresh_python(
        self, source: str, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(SOURCE_ROOT), python_path) if item
        )
        return subprocess.run(
            # The suite supplies both source and dependency roots through
            # PYTHONPATH.  Avoid re-processing the editable-install .pth here:
            # Python 3.11 may decode a non-ASCII workspace path as ASCII before
            # the child reaches the import-isolation assertion.
            [sys.executable, "-S", "-B", "-c", source, *arguments],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_api_modules_import_independently(self) -> None:
        modules = (
            "ecologyrsi_dsh.api.events",
            "ecologyrsi_dsh.api.execution",
            "ecologyrsi_dsh.api.catalog",
            "ecologyrsi_dsh.api.projection",
            "ecologyrsi_dsh.api.transport",
            "ecologyrsi_dsh.api.handler",
        )
        for module in modules:
            with self.subTest(module=module):
                result = self._fresh_python(
                    "from importlib import import_module; import_module(__import__('sys').argv[1])",
                    module,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_server_compatibility_exports_remain_importable(self) -> None:
        result = self._fresh_python(
            """
from ecologyrsi_dsh.server import (
    AUTO_ADVANCE_CONTINUOUS,
    PLUGIN_MANIFEST,
    EvolutionHTTPServer,
    EvolutionRequestHandler,
    _assert_http_scope,
    _assert_manifest_http_scope,
    _auto_advance_steps,
    _budget_value,
    _candidate_projection,
    _derived_seed,
    _evaluation_partition,
    _event_type,
    _expected_partition,
    _intervention_projection,
    _is_loopback_host,
    _max_generations,
    _parse_steps,
    _PLUGIN_FILES,
    _plugin_root,
    _projection_json,
    _public_intervention_receipt,
    _request_integer,
    _state_payload,
    serve,
)
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
