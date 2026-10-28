"""CLI smoke tests on the AGC task (development stages only; nothing is sealed)."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ecogym.cli import main

from .helpers import TASK_ID, golden, requires_agc


def run_cli(*argv: str) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(list(argv))
    return code, buffer.getvalue()


@requires_agc
class CliTests(unittest.TestCase):
    def test_list_and_describe(self):
        code, out = run_cli("list", "--json")
        self.assertEqual(code, 0)
        body = json.loads(out)
        self.assertIn(TASK_ID, [task["id"] for task in body["tasks"]])
        self.assertIn("greenhouse.diurnal_profile@1", [solver["ref"] for solver in body["solvers"]])
        code, out = run_cli("describe", TASK_ID, "--json")
        self.assertEqual(json.loads(out)["stages"]["model_selection"]["eligible_origins"], 683)
        self.assertIsNone(json.loads(out)["stages"]["validation"]["eligible_origins"])

    def test_evaluate_matches_the_golden_score(self):
        code, out = run_cli("evaluate", TASK_ID, "--preset", "expert", "--json")
        self.assertEqual(code, 0)
        expected = golden()["results"]["expert"]["model_selection"]["overall_skill"]
        self.assertAlmostEqual(json.loads(out)["score"], expected, delta=1e-9)

    def test_run_replay_report_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = str(Path(directory) / "ledger.sqlite3")
            code, out = run_cli("run", TASK_ID, "--preset", "recipe_seed", "--proposer", "random",
                                "--generations", "2", "--ledger", ledger, "--json")
            self.assertEqual(code, 0)
            run_id = json.loads(out)["run_id"]
            code, out = run_cli("replay", run_id, "--ledger", ledger)
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(out)["identical"])
            page = Path(directory) / "report.html"
            self.assertEqual(run_cli("report", run_id, "--ledger", ledger, "--out", str(page))[0], 0)
            self.assertIn("Generation 1", page.read_text(encoding="utf-8"))
            self.assertEqual(run_cli("verify", "--ledger", ledger)[0], 0)
            code, out = run_cli("runs", "--ledger", ledger)
            self.assertIn(run_id, out)

    def test_unknown_task_fails_cleanly(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_cli("describe", "nope@1")[0], 2)


if __name__ == "__main__":
    unittest.main()
