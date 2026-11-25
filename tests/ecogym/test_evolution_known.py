"""Which earlier versions a run treats as already compared (never run again)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ecogym.evolution.evolver import comparison_conditions, known_versions


def run(run_id, behavior, *, model="pjlab/deepseek-v4.1-flash", minutes=8.0, tokens=3_000_000,
        policy="best_local/1", train=("t1", "t2")):
    return {"run_id": run_id, "seed": {"behavior": "seed"}, "suite": {"train": list(train)},
            "config": {"model": model, "minutes": minutes, "tokens": tokens, "local_selection": policy},
            "rounds": [{"generation": 1, "finished": True, "candidates": [
                {"name": "a", "behavior": behavior, "decision": {"reason": "below_practical_delta"}}]}]}


class KnownVersionTests(unittest.TestCase):
    def known(self, view, prior):
        with patch("ecogym.evolution.evolver.behavior_digest", return_value="incumbent"):
            return known_versions(view, object(), prior)

    def test_a_version_compared_under_other_conditions_may_run_again(self):
        """2026-11-18 review: a candidate compared with one model and budget blocked a test with another."""

        current = run("evo:new", "own", model="token-plan/Sage-S2.1-Flash", minutes=20.0)
        prior = [run("evo:same", "same", model="token-plan/Sage-S2.1-Flash", minutes=20.0),
                 run("evo:other-model", "other-model", minutes=20.0),
                 run("evo:other-budget", "other-budget", model="token-plan/Sage-S2.1-Flash", minutes=8.0),
                 run("evo:other-policy", "other-policy", model="token-plan/Sage-S2.1-Flash", minutes=20.0,
                     policy="single_window/1"),
                 run("evo:fewer-tasks", "fewer-tasks", model="token-plan/Sage-S2.1-Flash", minutes=20.0,
                     train=("t1",))]
        known = self.known(current, prior)
        self.assertEqual(set(known), {"seed", "same", "own", "incumbent"})

    def test_conditions_name_one_model_however_it_is_written(self):
        self.assertEqual(comparison_conditions({"model": "deepseek-v4-flash", "minutes": 8, "tokens": 10}),
                         comparison_conditions({"model": "deepseek-official/deepseek-v4-flash", "minutes": 8.0,
                                                "tokens": 10, "local_selection": "single_window/1"}))
        self.assertNotEqual(comparison_conditions({"model": None, "minutes": 8, "tokens": 10}),
                            comparison_conditions({"model": "deepseek-v4-flash", "minutes": 8, "tokens": 10}))


if __name__ == "__main__":
    unittest.main()
