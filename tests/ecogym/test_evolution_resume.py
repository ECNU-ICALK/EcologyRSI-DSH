from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ecogym.episodes.agent import Cancelled
from ecogym.evolution.evidence import archive
from ecogym.evolution.loop import Evolution, EvolveConfig
from ecogym.kernel.ledger import Ledger

from .helpers import synthetic_bench


class EvolutionResumeTests(unittest.TestCase):
    def test_interrupted_round_preserves_candidates_and_recovers_only_matching_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite3"
            bench = synthetic_bench()
            task = bench.task.id
            suite = SimpleNamespace(id="synthetic", train=[task], digest="suite",
                                    gates=SimpleNamespace(test="t", confidence=0.95),
                                    tasks=lambda _: [bench.task])
            config = EvolveConfig(suite="synthetic", repeats=2, concurrency=1)
            harness = SimpleNamespace(digest="seed")
            with patch("ecogym.evolution.loop.suite_spec", return_value=suite):
                evolution = Evolution(config, path)
            evolution.benches[task] = bench
            with Ledger(path) as ledger:
                ledger.append("evo:run", "evolve_started", {"seed": {"digest": "seed"}})
                ledger.append("evo:run", "round_started", {"generation": 0, "incumbent": "seed"})
                ledger.append("evo:run", "round_finished", {"incumbent": "seed", "outcome": "baseline"})

            candidate = SimpleNamespace(digest="candidate")
            jobs = [{"name": "a", "digest": "candidate", "harness": candidate}]

            def record(ep, plan_id, replicate, attempt, *, artifact=True):
                with Ledger(path) as ledger:
                    ledger.append(ep, "episode_started", {
                        "task_id": task, "harness": {"digest": "candidate"},
                        "context": {"evolution": "evo:run", "generation": 1, "round_plan": plan_id,
                                    "role": "candidate", "candidate": "a", "replicate": replicate, "attempt": attempt}})
                    if artifact:
                        ledger.append(ep, "episode_scored", {"score": 0.5, "predictions": "pred.npz",
                                                              "predictions_digest": "sha"})
                    ledger.append(ep, "episode_finished", {"status": "finished" if artifact else "interrupted",
                                                            "trajectory": {"dir": directory}})

            def interrupt(run_id, generation, *args):
                plan_id = args[-1]
                record("ep:ready", plan_id, 0, 0)
                record("ep:interrupted", plan_id, 1, 0, artifact=False)
                record("ep:corrupt", plan_id, 1, 1)
                record("ep:other-plan", "other", 1, 5)
                raise Cancelled("stop")

            with patch("ecogym.evolution.loop.resolve", side_effect={"seed": harness, "candidate": candidate}.get), \
                    patch.object(evolution, "_measurement", return_value="measurement"), \
                    patch.object(evolution, "_propose", return_value=jobs) as propose:
                view = evolution.view("evo:run")
                with patch.object(evolution, "_episodes", side_effect=interrupt), self.assertRaises(Cancelled):
                    evolution._round("evo:run", view, evolution.state(view))
                view = evolution.view("evo:run")
                plan_id = view["rounds"][-1]["plan"]["id"]
                with patch.object(evolution, "_measurement", return_value="changed"), \
                        self.assertRaisesRegex(ValueError, "environment changed"):
                    evolution._round("evo:run", view, evolution.state(view))

                def failed(run_id, generation, harness, role, name, task, replicate, *args):
                    return {"status": "error", "task": task.id, "role": role, "candidate": name}

                def load(entry):
                    if entry["episode"] == "ep:corrupt":
                        raise ValueError("bad digest")
                    return object()

                with patch.object(evolution, "_evaluation", side_effect=load), \
                        patch.object(evolution, "_episode", side_effect=failed) as episode:
                    evolution._round("evo:run", view, evolution.state(view))
                self.assertEqual(propose.call_count, 1)
                calls = episode.call_args_list
                self.assertEqual(len(calls), 3)
                self.assertEqual([(call.args[3], call.args[6], call.args[-2], call.args[-1]) for call in calls],
                                 [("incumbent", 0, plan_id, 0), ("incumbent", 1, plan_id, 0),
                                  ("candidate", 1, plan_id, 2)])
            view = evolution.view("evo:run")
            self.assertEqual(len(view["rounds"]), 2)
            self.assertEqual(view["rounds"][-1]["episodes"][1]["episode"], "ep:ready")
            self.assertEqual(view["rounds"][-1]["candidates"][0]["missing"], [task])

    def test_rejected_candidate_with_no_digest_is_rendered(self):
        rendered = archive({"rounds": [{"generation": 1, "incumbent": "seed", "notes": [], "candidates": [
            {"name": "a", "digest": None, "status": "invalid"}]}]})
        self.assertIn("invalid", rendered)
