import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ecogym.evolution.evidence import related_runs
from ecogym.evolution.loop import Evolution, EvolveConfig
from ecogym.kernel.ledger import Ledger


class EvolutionMethodTests(unittest.TestCase):
    def test_legacy_runs_keep_the_previous_research_mode(self):
        self.assertEqual(EvolveConfig(suite="demo").method_research, "autonomous")
        self.assertEqual(EvolveConfig.from_dict({"suite": "demo"}).method_research, "internal")
        config = EvolveConfig(suite="demo", method_research="supplied", method_registry="methods")
        self.assertTrue(Path(config.method_registry).is_absolute())
        self.assertEqual(EvolveConfig.from_dict(config.to_dict()), config)
        with self.assertRaisesRegex(ValueError, "host method registry"):
            EvolveConfig(suite="demo", method_research="supplied")

    def test_supplied_evidence_does_not_enter_autonomous_history(self):
        with tempfile.TemporaryDirectory() as directory, Ledger(Path(directory) / "ledger.sqlite3") as ledger:
            base = {"suite": {"train": ["task"]}, "seed": {"digest": "seed"}}
            ledger.append("evo:old", "evolve_started", base)
            ledger.append("evo:supplied", "evolve_started", {
                **base, "config": {"method_research": "supplied"}, "method_registry_digest": "registry"})
            ledger.append("evo:auto", "evolve_started", {
                **base, "config": {"method_research": "autonomous"}, "method_registry_digest": "registry"})
            view = {**base, "run_id": "evo:current", "config": {"method_research": "autonomous"},
                    "method_registry_digest": "registry"}
            self.assertEqual({run["run_id"] for run in related_runs(ledger, view)}, {"evo:old", "evo:auto"})

    def test_public_reference_uses_only_complete_probes_and_compares_same_stage(self):
        task = SimpleNamespace(id="demo-task")
        suite = SimpleNamespace(id="demo", train=[task.id], tasks=lambda _: [task],
                                gates=SimpleNamespace(test="t"))
        with patch("ecogym.evolution.loop.suite_spec", return_value=suite):
            evolution = Evolution(EvolveConfig(suite="demo", repeats=1), Path("unused.sqlite3"))
        evolution.benches[task.id] = Mock()
        evolution.method_registry = Mock()
        evolution.method_registry.judge.return_value = (
            {"status": "ok", "coverage": 1, "score": 0.4, "judge_protocol": "causal_stream/2"}, None)
        probes = [
            {"status": "evaluated", "method": {"digest": "broken"}, "result": {"status": "failed"}},
            {"status": "evaluated", "method": {"digest": "measured"}, "result": {"status": "ok", "mean": 0.6}},
        ]
        entries = [{"task": task.id, "role": "incumbent", "status": "finished", "score": 0.1, "predictions": "a"},
                   {"task": task.id, "role": "candidate", "candidate": "a", "status": "finished",
                    "score": 0.55, "predictions": "b"}]
        with patch("ecogym.evolution.loop.discover_runtime", return_value=Mock()):
            result = evolution._public_reference({"probes": probes}, entries, "a")
            self.assertAlmostEqual(result["selected_minus_external"], 0.15)
            self.assertIsNone(evolution._public_reference({"probes": probes}, [], "a")["selected_minus_external"])
        self.assertEqual(evolution.method_registry.judge.call_args.args[0], "measured")
        self.assertEqual(evolution.method_registry.judge.call_args.args[2], "model_selection")
