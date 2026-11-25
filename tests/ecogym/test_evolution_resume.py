from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ecogym.episodes.agent import Cancelled
from ecogym.evolution.evidence import archive
from ecogym.evolution.loop import Evolution, EvolveConfig, measured_sources
from ecogym.kernel.ledger import Ledger

from .helpers import synthetic_bench


class EvolutionResumeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.path = Path(self.directory) / "ledger.sqlite3"
        bench = synthetic_bench()
        self.task = bench.task.id
        suite = SimpleNamespace(id="synthetic", train=[self.task], digest="suite",
                                gates=SimpleNamespace(test="t", confidence=0.95),
                                tasks=lambda _: [bench.task])
        config = EvolveConfig(suite="synthetic", repeats=2, concurrency=1)
        with patch("ecogym.evolution.loop.suite_spec", return_value=suite):
            self.evolution = Evolution(config, self.path)
        self.evolution.benches[self.task] = bench
        with Ledger(self.path) as ledger:
            ledger.append("evo:run", "evolve_started", {"seed": {"digest": "seed"}})
            ledger.append("evo:run", "round_started", {"generation": 0, "incumbent": "seed"})
            ledger.append("evo:run", "round_finished", {"incumbent": "seed", "outcome": "baseline"})
        self.harnesses = {"seed": SimpleNamespace(digest="seed"), "candidate": SimpleNamespace(digest="candidate")}
        self.jobs = [{"name": "a", "digest": "candidate", "harness": self.harnesses["candidate"]}]

    def record(self, ep, plan_id, replicate, attempt, *, artifact=True):
        with Ledger(self.path) as ledger:
            ledger.append(ep, "episode_started", {
                "task_id": self.task, "harness": {"digest": "candidate"},
                "context": {"evolution": "evo:run", "generation": 1, "round_plan": plan_id,
                            "role": "candidate", "candidate": "a", "replicate": replicate, "attempt": attempt}})
            if artifact:
                ledger.append(ep, "episode_scored", {"score": 0.5, "predictions": "pred.npz",
                                                      "predictions_digest": "sha"})
            ledger.append(ep, "episode_finished", {"status": "finished" if artifact else "interrupted",
                                                    "trajectory": {"dir": self.directory}})

    def interrupt(self, run_id, generation, *args):
        """The first attempt at round 1: a few episodes, then the run stops."""

        plan_id = args[-1]
        self.record("ep:ready", plan_id, 0, 0)
        self.record("ep:interrupted", plan_id, 1, 0, artifact=False)
        self.record("ep:corrupt", plan_id, 1, 1)
        self.record("ep:other-plan", "other", 1, 5)
        raise Cancelled("stop")

    @staticmethod
    def load(entry):
        if entry["episode"] == "ep:corrupt":
            raise ValueError("bad digest")
        return object()

    def test_interrupted_round_preserves_candidates_and_recovers_only_matching_slots(self):
        evolution = self.evolution
        with patch("ecogym.evolution.loop.resolve", side_effect=self.harnesses.get), \
                patch.object(evolution, "_measurement", return_value="measurement"), \
                patch.object(evolution, "_propose", return_value=self.jobs) as propose:
            view = evolution.view("evo:run")
            with patch.object(evolution, "_episodes", side_effect=self.interrupt), self.assertRaises(Cancelled):
                evolution._round("evo:run", view, evolution.state(view))
            view = evolution.view("evo:run")
            plan_id = view["rounds"][-1]["plan"]["id"]

            def failed(run_id, generation, harness, role, name, task, replicate, *args):
                return {"status": "error", "task": task.id, "role": role, "candidate": name}

            with patch.object(evolution, "_evaluation", side_effect=self.load), \
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
        self.assertEqual(view["rounds"][-1]["plan"]["id"], plan_id)
        self.assertEqual(view["rounds"][-1]["episodes"][1]["episode"], "ep:ready")
        self.assertEqual(view["rounds"][-1]["candidates"][0]["missing"], [self.task])

    def test_changed_environment_replans_the_same_candidates_and_reuses_no_episode(self):
        """An upgrade or a new runtime between the interruption and the resume: the round goes on with the same
        candidates and guidance, but every slot runs again under a new plan instead of the run being stuck."""

        evolution = self.evolution
        seen = []

        def rerun(run_id, generation, jobs, snapshot, plan_id):
            recovered, _ = evolution._recover_episodes(run_id, generation, plan_id, snapshot)
            seen.append((plan_id, [role for _, role, _ in jobs], recovered))
            raise Cancelled("stop")

        with patch("ecogym.evolution.loop.resolve", side_effect=self.harnesses.get), \
                patch.object(evolution, "_evaluation", side_effect=self.load), \
                patch.object(evolution, "_propose", return_value=self.jobs) as propose:
            with patch.object(evolution, "_measurement", return_value="measurement"):
                view = evolution.view("evo:run")
                with patch.object(evolution, "_episodes", side_effect=self.interrupt), self.assertRaises(Cancelled):
                    evolution._round("evo:run", view, evolution.state(view))
            first = evolution.view("evo:run")["rounds"][-1]["plan"]
            snapshot = SimpleNamespace(guidance=first["guidance"])
            reusable, _ = evolution._recover_episodes("evo:run", 1, first["id"], snapshot)
            self.assertEqual([entry["episode"] for entry in reusable.values()], ["ep:ready"])

            with patch.object(evolution, "_measurement", return_value="changed"), \
                    patch.object(evolution, "_episodes", side_effect=rerun):
                for _ in range(2):  # the second resume keeps the new plan: its environment has not changed again
                    view = evolution.view("evo:run")
                    with self.assertRaises(Cancelled):
                        evolution._round("evo:run", view, evolution.state(view))
        self.assertEqual(propose.call_count, 1)  # the candidates are not proposed again
        view = evolution.view("evo:run")
        self.assertEqual(len(view["rounds"]), 2)  # the same round 1, resumed
        plan = view["rounds"][-1]["plan"]
        self.assertEqual((plan["replaces"], plan["reason"], plan["measurement"]),
                         (first["id"], "environment_changed", "changed"))
        self.assertNotEqual(plan["id"], first["id"])
        for key in ("schema", "generation", "incumbent", "expert_until", "guidance", "candidates"):
            self.assertEqual(plan[key], first[key], key)
        self.assertEqual(seen, [(plan["id"], ["incumbent", "candidate"], {})] * 2)
        with Ledger(self.path) as ledger:
            plans = [event.body["id"] for event in ledger.events("evo:run") if event.kind == "round_planned"]
        self.assertEqual(plans, [first["id"], plan["id"]])

    def test_the_measurement_covers_what_agents_read_and_the_task_code(self):
        """A domain pack's task background, knowledge and expert, and the task and ledger code, are part of what a
        round measured; until 0.4.2 only some layers' *.py files were."""

        import ecogym

        package = Path(ecogym.__file__).resolve().parent
        names = {str(path.relative_to(package)) for path in measured_sources(package)}
        for name in ("evolution/loop.py", "evolution/evolver.md", "episodes/predictor_runner.py", "task/spec.py",
                     "kernel/canonical.py", "domains/greenhouse/knowledge.md", "domains/greenhouse/expert.py",
                     "domains/greenhouse/tasks/cucumber_climate.toml", "harness/seed/instructions.md"):
            self.assertIn(name, names)
        self.assertFalse(any("__pycache__" in name or name.endswith(".pyc") or "/." in f"/{name}" for name in names))

    def test_a_plan_for_another_incumbent_is_refused(self):
        evolution = self.evolution
        with Ledger(self.path) as ledger:
            ledger.append("evo:run", "round_started", {"generation": 1, "incumbent": "seed"})
            ledger.append("evo:run", "round_planned", {
                "schema": "ecogym.round-plan/1", "generation": 1, "incumbent": "other", "measurement": "measurement",
                "expert_until": 0, "guidance": {self.task: ""}, "candidates": [], "id": "plan"})
        with patch("ecogym.evolution.loop.resolve", side_effect=self.harnesses.get), \
                patch.object(evolution, "_measurement", return_value="measurement"):
            view = evolution.view("evo:run")
            with self.assertRaisesRegex(ValueError, "another incumbent"):
                evolution._round("evo:run", view, evolution.state(view))

    def test_rejected_candidate_with_no_digest_is_rendered(self):
        rendered = archive({"rounds": [{"generation": 1, "incumbent": "seed", "notes": [], "candidates": [
            {"name": "a", "digest": None, "status": "invalid"}]}]})
        self.assertIn("invalid", rendered)
