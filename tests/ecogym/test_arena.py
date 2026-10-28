from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ecogym.arena import Arena, Edit, RunConfig, SweepProposer, apply_edit, check_edit, fold, replay, seal
from ecogym.arena.proposers import ProposalContext, RandomProposer
from ecogym.arena.sealed import exposure_key
from ecogym.kernel.ledger import Ledger

from .helpers import synthetic_bench, synthetic_task
from .test_solvers import RECIPE

WEAK = {"solver": "recipe_ridge@1", "params": {**RECIPE["params"], "scale": 0.3}}


class EditTests(unittest.TestCase):
    def setUp(self):
        self.bench = synthetic_bench()
        self.genome = self.bench.genome(WEAK)

    def test_trust_region(self):
        small = apply_edit(self.genome, Edit({"scale.y@1": 0.4}), self.bench.task)
        self.assertEqual(check_edit(self.genome, small, max_step=0.15, max_changes=3), [])
        large = apply_edit(self.genome, Edit({"scale.y@1": 0.9}), self.bench.task)
        self.assertTrue(check_edit(self.genome, large, max_step=0.15, max_changes=3))
        # alpha spans four decades, so it moves on a log scale: 0.1 -> 0.2 is 0.075 of the range
        alpha = apply_edit(self.genome, Edit({"alpha": 0.2}), self.bench.task)
        self.assertEqual(check_edit(self.genome, alpha, max_step=0.15, max_changes=3), [])
        many = apply_edit(self.genome, Edit({"scale.y@1": 0.35, "scale.y@6": 0.35, "alpha": 0.12}), self.bench.task)
        self.assertTrue(check_edit(self.genome, many, max_step=0.15, max_changes=2))
        self.assertTrue(check_edit(self.genome, self.genome, max_step=0.15, max_changes=3))

    def test_edits_are_validated_by_the_schema(self):
        with self.assertRaises(ValueError):
            apply_edit(self.genome, Edit({"scale.nope@1": 0.4}), self.bench.task)
        with self.assertRaises(ValueError):
            apply_edit(self.genome, Edit({"terms": [{"op": "target_lag", "k": -3}]}), self.bench.task)

    def test_proposers_stay_inside_the_trust_region_and_are_deterministic(self):
        context = ProposalContext(self.bench.task, self.genome, 0, 7, 0.15, 3)
        for proposer in (SweepProposer(), RandomProposer()):
            edits = proposer.propose(context, 4)
            self.assertEqual(len(edits), 4)
            for edit in edits:
                child = apply_edit(self.genome, edit, self.bench.task)
                self.assertEqual(check_edit(self.genome, child, max_step=0.15, max_changes=3), [])
        first = [edit.changes for edit in RandomProposer().propose(context, 4)]
        self.assertEqual(first, [edit.changes for edit in RandomProposer().propose(context, 4)])


class ArenaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.directory.name) / "ledger.sqlite3")
        self.bench = synthetic_bench()

    def tearDown(self):
        self.ledger.close()
        self.directory.cleanup()

    def run_arena(self, proposer=None, **config):
        arena = Arena(self.bench, self.ledger, RunConfig(**{"generations": 3, **config}), proposer or SweepProposer())
        return arena.run(self.bench.genome(WEAK), label="test")

    def test_run_is_recorded_and_replays_identically(self):
        result = self.run_arena()
        events = self.ledger.events(result["run_id"])
        kinds = [event.kind for event in events]
        self.assertEqual(kinds[0], "run_started")
        self.assertEqual(kinds[-1], "run_finished")
        self.assertEqual(kinds.count("generation_started"), 3)
        self.assertEqual(self.ledger.verify(), len(events))
        view = fold(events)
        self.assertEqual(view["status"], "finished")
        self.assertEqual(len(view["generations"]), 3)
        self.assertTrue(replay(self.ledger, result["run_id"], self.bench)["identical"])

    def test_promotions_pass_every_gate(self):
        result = self.run_arena(RandomProposer(), generations=4, seed=3)
        view = fold(self.ledger.events(result["run_id"]))
        promoted = [item for item in view["generations"] if item["promoted"]]
        self.assertGreaterEqual(result["promotions"], 1)
        for generation in promoted:
            decision = generation["comparison"]["decision"]
            self.assertGreater(decision["delta"], self.bench.task.gates.min_delta)
            self.assertGreater(decision["interval"][0], 0)
            self.assertTrue(all(cell["passed"] for cell in decision["cells"]))
        self.assertGreater(result["incumbent_score"], result["seed_score"])

    def test_rejected_edits_are_logged_not_scored(self):
        class Wild:
            name = "wild"

            def options(self):
                return {}

            def propose(self, context, k):
                return [Edit({"scale.y@1": 1.0}, "too far"), Edit({"alpha": 0.1}, "no change")]

            def observe(self, report):
                return None

        result = self.run_arena(Wild(), generations=1)
        view = fold(self.ledger.events(result["run_id"]))
        self.assertEqual(len(view["generations"][0]["rejected"]), 2)
        self.assertIsNone(view["generations"][0]["comparison"])
        self.assertEqual(result["promotions"], 0)


class SealedTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.directory.name) / "ledger.sqlite3")
        self.bench = synthetic_bench()
        self.genome = self.bench.genome(RECIPE)

    def tearDown(self):
        self.ledger.close()
        self.directory.cleanup()

    def test_each_method_sees_each_sealed_stage_once_and_in_order(self):
        with self.assertRaises(PermissionError):
            seal(self.bench, self.ledger, self.genome, "final_test", run_id="run:t")
        report = seal(self.bench, self.ledger, self.genome, "validation", run_id="run:t")
        self.assertGreater(report["origins"], 0)
        with self.assertRaises(PermissionError):
            seal(self.bench, self.ledger, self.genome, "validation", run_id="run:t")
        seal(self.bench, self.ledger, self.genome, "final_test", run_id="run:t")
        self.assertEqual(len(self.ledger.events("run:t", "sealed_evaluated")), 2)
        with self.assertRaises(ValueError):
            seal(self.bench, self.ledger, self.genome, "model_selection", run_id="run:t")

    def test_exposure_ignores_gate_settings(self):
        task = synthetic_task()
        looser = synthetic_bench(replace(task, gates=replace(task.gates, min_delta=0.0)))
        self.assertEqual(exposure_key(self.bench, self.genome, "validation"),
                         exposure_key(looser, looser.genome(RECIPE), "validation"))


if __name__ == "__main__":
    unittest.main()
