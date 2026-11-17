import unittest
from pathlib import Path
from unittest.mock import Mock

from ecogym.episodes.classical import classical_reference_tool
from ecogym.episodes.gateway import PREFIX, Gateway
from ecogym.episodes.mcp import ToolError
from tests.ecogym.helpers import synthetic_bench


class ClassicalReferenceTests(unittest.TestCase):
    def test_reference_calls_score_only_local_origins(self):
        bench = synthetic_bench()
        check_open = Mock()
        tool = classical_reference_tool((bench,), check_open=check_open)
        catalog = tool.handler()
        self.assertIn("seasonal_naive", catalog["methods"][bench.task.id])
        for name in ("selected_naive", "persistence", "seasonal_naive"):
            report = tool.handler(name=name)
            row = report["tasks"][bench.task.id]
            self.assertEqual(report["stage"], "calibration_uq")
            self.assertEqual(row["origins"], bench.stage("calibration_uq").size)
            self.assertEqual(set(row["cells"]), {cell.key for cell in bench.task.cells})
            if name == "selected_naive":
                self.assertAlmostEqual(row["score"], 0)
        with self.assertRaises(ToolError):
            tool.handler(name="validation")
        check_open.assert_called()

    def test_disabled_search_removes_gateway_capability(self):
        gateway = Gateway(key="test", token="test", log_path=Path("unused"), chat=PREFIX)
        self.assertTrue(gateway.permits("POST", PREFIX, "/anthropic/v1/messages"))
        gateway.search_enabled = False
        self.assertFalse(gateway.permits("POST", PREFIX, "/anthropic/v1/messages"))
        self.assertTrue(gateway.permits("POST", PREFIX, "/chat/completions"))
