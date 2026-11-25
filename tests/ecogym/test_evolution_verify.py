"""Verification plans and exposure recovery with a scratch ledger and synthetic results."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from ecogym.evolution import verify
from ecogym.evolution.loop import EvolveConfig
from ecogym.kernel.ledger import Ledger
from ecogym.task.suite import SuiteGates


class VerificationTests(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.ledger = self.root / "ledger.sqlite3"
        self.run = "evo:demo"
        self.config = EvolveConfig(suite="demo", model="deepseek-v4-flash", minutes=8, tokens=3000)
        self.train = [SimpleNamespace(id="train")]
        self.holdout = []
        self.suite = SimpleNamespace(id="demo", digest="suite", gates=SuiteGates(),
                                     tasks=lambda side: self.train if side == "train" else self.holdout)
        for name, value in (("ecogym_home", lambda: self.root), ("scratch_root", lambda: self.root),
                            ("suite_spec", lambda _: self.suite)):
            mock = patch.object(verify, name, value)
            mock.start()
            self.addCleanup(mock.stop)

    def append(self, kind, body, *, run=None):
        with Ledger(self.ledger) as db:
            db.append(run or self.run, kind, body)

    def start(self, *, final="new", episodes=()):
        self.append("evolve_started", {"seed": {"digest": "seed"}, "suite": {"digest": "suite"},
                                      "config": self.config.to_dict()})
        if episodes:
            self.append("round_started", {"generation": 0, "incumbent": "seed"})
            self.append("episodes_finished", {"episodes": episodes})
            self.append("round_finished", {"incumbent": "seed"})
        self.append("evolve_finished", {"final": final})

    def episode(self, episode, harness, *, task="train", role="incumbent", minutes=8, tokens=3000,
                protocol=verify.JUDGE_PROTOCOL):
        (self.root / "test.npz").touch()
        self.append("episode_started", {"task_id": task, "harness": {"digest": harness},
                                       "model": "deepseek-v4-flash", "minutes": minutes, "tokens": tokens,
                                       "context": {"role": role}}, run=episode)
        self.append("episode_scored", {"digest": episode + "-program", "score": 0.2, "interval": None,
                                      "predictions": "test.npz", "judge_protocol": protocol}, run=episode)
        self.append("episode_finished", {"status": "finished", "trajectory": {"dir": str(self.root)}},
                    run=episode)
        return {"episode": episode, "task": task, "harness": harness, "status": "finished",
                "predictions": str(self.root / "test.npz")}

    def test_empty_plan_records_both_versions_as_incomplete(self):
        self.start()
        body = verify.verify_sealed(self.ledger, self.run, "validation")
        self.assertFalse(body["complete"])
        self.assertIsNone(body["decision"])
        self.assertEqual(body["versions"], {"seed": {"digest": "seed", "tasks": {}},
                                            "final": {"digest": "new", "tasks": {}}})
        self.assertEqual({(row["version"], row["task"]) for row in body["missing"]},
                         {("seed", "train"), ("final", "train")})
        verify.verify_sealed(self.ledger, self.run, "validation")
        with Ledger(self.ledger) as db:
            self.assertEqual(len(db.events(self.run, "sealed_verified")), 1)

    def test_final_test_requires_validation_on_entire_suite(self):
        self.holdout = [SimpleNamespace(id="holdout")]
        self.start(final="seed")
        self.append("sealed_verified", {"stage": "validation", "versions": {"seed": {"tasks": {}}}})
        with (patch.object(verify, "_seal_locked") as seal,
              self.assertRaisesRegex(PermissionError, "full validation plan")):
            verify.verify_sealed(self.ledger, self.run, "final_test")
        seal.assert_not_called()

    def test_holdout_replaces_only_the_wrong_budget_legacy_episode(self):
        self.config = EvolveConfig.from_dict({"suite": "demo", "model": "deepseek-v4-flash",
                                              "minutes": 8, "tokens": 3000})
        self.holdout = [SimpleNamespace(id="holdout")]
        self.start()
        old = self.episode("ep:old", "seed", task="holdout", role="holdout", minutes=4)
        final = self.episode("ep:final", "new", task="holdout", role="holdout")
        self.append("holdout_verified", {"tasks": ["holdout"],
                                        "versions": {"seed": {"digest": "seed", "episodes": {"holdout": old}},
                                                     "final": {"digest": "new", "episodes": {"holdout": final}}}})

        def run(config, *args, **kwargs):
            self.assertEqual(config.harness, "seed")
            entry = self.episode("ep:correct", "seed", task="holdout", role="holdout")
            return {**entry, "score": 0.2, "scored": {"interval": None, "judge_protocol": verify.JUDGE_PROTOCOL}}

        def open_bench(task, benches):
            benches[task] = object()

        with (patch.object(verify, "_open", side_effect=open_bench),
              patch.object(verify, "exposure_key", side_effect=lambda _, harness, *a, **kw: f"exposure-{harness}"),
              patch.object(verify, "run_episode", side_effect=run) as runs,
              patch.object(verify, "load_predictions"),
              patch.object(verify, "_compare", return_value=None)):
            body = verify.verify_holdout(self.ledger, self.run)
            repeated = verify.verify_holdout(self.ledger, self.run)
        self.assertEqual(runs.call_count, 1)
        self.assertTrue(body["complete"])
        self.assertEqual(repeated, body | {"at": repeated["at"]})
        self.assertEqual(body["versions"]["seed"]["episodes"]["holdout"]["episode"], "ep:correct")
        self.assertEqual(body["versions"]["final"]["episodes"]["holdout"]["episode"], "ep:final")

    def test_run_summary_marks_every_incomplete_verification(self):
        """The run list read "已验证" for a holdout verification with missing tasks (2026-11-18 review)."""

        self.start()
        self.append("holdout_verified", {"tasks": ["holdout"], "versions": {}, "means": {}, "decision": None,
                                        "missing": [{"version": "seed", "task": "holdout",
                                                     "reason": "unscored_episode"}], "complete": False})
        self.append("sealed_verified", {"stage": "validation", "versions": {}, "means": {}, "decision": None,
                                       "missing": [], "complete": True})
        with Ledger(self.ledger) as db:
            verified = verify.fold(db.events(self.run))
        from ecogym.evolution.fold import summary

        self.assertEqual(summary(verified)["verified"]["incomplete"], ["holdout"])

    def test_holdout_does_not_reuse_a_different_selection_policy(self):
        expected = verify.holdout_conditions(self.config)
        archived = {"model": self.config.model, "minutes": self.config.minutes, "tokens": self.config.tokens}
        self.assertFalse(verify._ran_under(verify.conditions_of(archived), expected))
        archived["local_selection"] = self.config.local_selection
        self.assertTrue(verify._ran_under(verify.conditions_of(archived), expected))

    def test_final_test_uses_the_programs_in_complete_validation(self):
        episodes = [self.episode(episode, "seed") for episode in ("ep:c", "ep:d")]
        self.start(final="seed", episodes=episodes)
        for episode in ("ep:a", "ep:b"):
            self.episode(episode, "seed")
        self.append("sealed_task", {"stage": "validation", "version": "parent", "harness": "seed", "task": "train",
                                    "exposure_key": "first-look", "judge_protocol": verify.JUDGE_PROTOCOL,
                                    "plan": {"episodes": ["ep:a", "ep:b"]}})

        def seal(_path, _run, stage, looks, **kwargs):
            look, = looks
            self.assertEqual(look.episodes, ("ep:a", "ep:b"))
            artifact = self.root / f"{stage}.npz"
            artifact.write_text("synthetic")
            result = {"plan": look.plan(), "conditions": look.conditions, "judge_protocol": verify.JUDGE_PROTOCOL,
                      "programs": [{"episode": episode, "predictions": str(artifact),
                                    "predictions_digest": verify.file_digest(artifact)} for episode in look.episodes]}
            return {"seed": {"train": result}}, {}, []

        with (patch.object(verify, "_seal_locked", side_effect=seal) as work, patch.object(verify, "_key_bench"),
              patch.object(verify, "exposure_key", return_value="first-look")):
            validation = verify.verify_sealed(self.ledger, self.run, "validation")
            final_test = verify.verify_sealed(self.ledger, self.run, "final_test")
            (self.root / "validation.npz").write_text("changed")
            with self.assertRaisesRegex(PermissionError, "full validation plan"):
                verify.verify_sealed(self.ledger, self.run, "final_test")
        self.assertTrue(validation["complete"])
        self.assertTrue(final_test["complete"])
        self.assertEqual(final_test["scope"], "fixed_programs")
        self.assertEqual(validation["plan"], final_test["plan"])
        self.assertEqual(work.call_count, 2)

    def test_frozen_look_resumes_only_missing_program_and_rejects_another_plan(self):
        look = verify.Look("seed", "seed", "train", ("ep:a", "ep:b"), {"model": "example"},
                           {"ep:a": "sha-a", "ep:b": "sha-b"})
        available = {"ep:a"}
        evaluated = []

        def extract(_ledger, pending, scratch):
            return ([(episode, scratch / episode) for episode in pending.episodes if episode in available],
                    [episode for episode in pending.episodes if episode not in available])

        def judge(_bench, program, *args, **kwargs):
            evaluated.append(program.name)
            return {"status": "ok", "score": 0.2, "interval": None, "coverage": 1, "origins": 1, "cells": []}, object()

        def run(planned):
            return verify._seal(self.ledger, self.run, "validation", [planned], limits=None, store=self.root,
                                filename=lambda _, index: f"{index}.npz")

        bench = SimpleNamespace(stage=lambda *args, **kwargs: None)
        with (patch.object(verify, "_open", return_value=bench),
              patch.object(verify, "exposure_key", return_value="same-exposure"),
              patch.object(verify, "discover_runtime", return_value=None),
              patch.object(verify, "_extract", side_effect=extract),
              patch.object(verify, "judge_program", side_effect=judge),
              patch.object(verify, "save_predictions", side_effect=lambda path, *_: path.write_text("synthetic")),
              patch.object(verify, "_evaluations", return_value=(object(),))):
            partial, _, missing = run(look)
            self.assertFalse(partial["seed"]["train"]["complete"])
            self.assertEqual(missing[0]["episodes"], ["ep:b"])
            with Ledger(self.ledger) as db:
                self.assertEqual(verify.gaps(db, {"versions": {"seed": {"tasks": partial["seed"]}}}, [look]), [look])
            available.add("ep:b")
            full, _, missing = run(look)
            self.assertTrue(full["seed"]["train"]["complete"])
            self.assertEqual(missing, [])
            self.assertEqual(verify.program_episodes(full["seed"]["train"]), ["ep:a", "ep:b"])
            changed = verify.Look("seed", "seed", "train", ("ep:b",), look.conditions, {"ep:b": "sha-b"})
            results, _, missing = run(changed)
            self.assertEqual(results, {"seed": {}})
            self.assertEqual(missing[0]["reason"], "prior_exposure_differs")
        self.assertEqual(evaluated, ["ep:a", "ep:b"])

    def test_legacy_judge_exposure_does_not_reopen_under_the_new_protocol(self):
        look = verify.Look("seed", "seed", "train", ("ep:a",), {"model": "example"})
        self.append("sealed_task", {"stage": "validation", "version": "seed", "harness": "seed", "task": "train",
                                    "exposure_key": "same-exposure", "conditions": look.conditions,
                                    "episode": "ep:a"})
        with (patch.object(verify, "_open"), patch.object(verify, "exposure_key", return_value="same-exposure"),
              patch.object(verify, "judge_program") as judge):
            results, _, missing = verify._seal(self.ledger, self.run, "validation", [look], limits=None,
                                               store=self.root, filename=lambda *_: "0.npz")
        judge.assert_not_called()
        self.assertEqual(results, {"seed": {}})
        self.assertEqual(missing[0]["reason"], "prior_judge_protocol_differs")

    def earlier_look(self, harness, episodes, *, key):
        """A complete validation look another run took of ``harness`` with ``episodes``."""

        artifact = self.root / f"earlier-{harness}.npz"
        artifact.write_text("synthetic")
        with Ledger(self.ledger) as db:
            (look,), _ = verify._looks(db, [("final", harness, "train", episodes)])
        self.append("sealed_task", {
            "stage": "validation", "version": "final", "harness": harness, "task": "train", "exposure_key": key,
            "conditions": look.conditions, "judge_protocol": verify.JUDGE_PROTOCOL, "plan": look.plan(),
            "episodes": list(episodes), "complete": True,
            "programs": [{"episode": episode, "predictions": str(artifact),
                          "predictions_digest": verify.file_digest(artifact)} for episode in episodes]},
            run="evo:earlier")

    def test_a_version_an_earlier_run_judged_keeps_that_look(self):
        """A run that starts from an earlier run's final version plans that version's recorded look: planning its
        own programs would find the exposure taken (prior_exposure_differs) and never complete."""

        self.start(final="seed", episodes=[self.episode(episode, "seed") for episode in ("ep:c", "ep:d")])
        for episode in ("ep:a", "ep:b"):
            self.episode(episode, "seed")
        self.earlier_look("seed", ["ep:a", "ep:b"], key="same-exposure")
        with (patch.object(verify, "_key_bench"), patch.object(verify, "_open"),
              patch.object(verify, "exposure_key", return_value="same-exposure"),
              patch.object(verify, "judge_program") as judge,
              patch.object(verify, "_evaluations", return_value=(object(), object()))):
            with Ledger(self.ledger) as db:
                looks, missing = verify.sealed_plan(db, verify.fold(db.events(self.run)), self.suite, "validation")
            body = verify.verify_sealed(self.ledger, self.run, "validation")
        judge.assert_not_called()
        self.assertEqual([look.episodes for look in looks], [("ep:a", "ep:b")])
        self.assertEqual(missing, [])
        self.assertTrue(body["complete"])
        result = body["versions"]["seed"]["tasks"]["train"]
        self.assertEqual((result["reused"], result["source_run"]), (True, "evo:earlier"))

    def test_too_few_programs_open_no_look_at_the_sealed_stage(self):
        """2026-11-18 review: a version with one program where the run asks for two used to be judged anyway,
        spending its only look at the stage on whichever replicate scored."""

        self.start(final="seed", episodes=[self.episode("ep:c", "seed")])
        with (patch.object(verify, "_key_bench"), patch.object(verify, "_open"),
              patch.object(verify, "exposure_key", return_value="exposure"),
              patch.object(verify, "_extract", return_value=([("ep:c", self.root)], [])) as extract,
              patch.object(verify, "judge_program") as judge):
            body = verify.verify_sealed(self.ledger, self.run, "validation")
        extract.assert_not_called()
        judge.assert_not_called()
        self.assertFalse(body["complete"])
        self.assertIsNone(body["decision"])
        self.assertEqual([(item["reason"], item["episodes"], item["expected"]) for item in body["missing"]],
                         [("missing_replicates", ["ep:c"], 2)])
        with Ledger(self.ledger) as db:
            self.assertEqual(db.exposures(), [])
            self.assertEqual(db.events(self.run, "sealed_task"), [])

    def test_an_earlier_single_program_look_cannot_complete_a_two_replicate_plan(self):
        self.start(final="seed", episodes=[self.episode(episode, "seed") for episode in ("ep:c", "ep:d")])
        self.episode("ep:a", "seed")
        self.earlier_look("seed", ["ep:a"], key="same-exposure")
        with (patch.object(verify, "_key_bench"), patch.object(verify, "_open"),
              patch.object(verify, "exposure_key", return_value="same-exposure"),
              patch.object(verify, "judge_program") as judge,
              patch.object(verify, "_evaluations", return_value=(object(),))):
            body = verify.verify_sealed(self.ledger, self.run, "validation")
        judge.assert_not_called()
        self.assertFalse(body["complete"])
        self.assertIsNone(body["decision"])
        item, = body["missing"]
        self.assertEqual((item["reason"], item["episodes"], item["expected"], item["recorded_before"]),
                         ("missing_replicates", ["ep:a"], 2, True))
        shown = body["versions"]["seed"]["tasks"]["train"]  # the earlier result stays visible, as evidence only
        self.assertEqual((shown["source_run"], verify.program_episodes(shown)), ("evo:earlier", ["ep:a"]))

    def test_adoption_check_keeps_the_replaced_versions_earlier_look(self):
        self.append("evolve_started", {"seed": {"digest": "seed"}, "suite": {"digest": "suite"},
                                      "config": self.config.to_dict()})
        self.append("round_started", {"generation": 0, "incumbent": "seed"})
        self.append("episodes_finished", {"episodes": [self.episode(ep, "seed") for ep in ("ep:c", "ep:d")]})
        self.append("round_finished", {"incumbent": "seed"})
        self.append("round_started", {"generation": 1, "incumbent": "seed"})
        self.append("candidate_proposed", {"generation": 1, "name": "a", "digest": "new"})
        self.append("episodes_finished", {"episodes": [
            *({**self.episode(ep, "new", role="candidate"), "role": "candidate", "candidate": "a"}
              for ep in ("ep:e", "ep:f")),
            *(self.episode(ep, "seed") for ep in ("ep:g", "ep:h"))]})
        self.append("harness_adopted", {"generation": 1, "name": "a", "digest": "new", "delta": 0.01})
        self.append("round_finished", {"generation": 1, "incumbent": "new", "adopted": "a"})
        for episode in ("ep:a", "ep:b"):
            self.episode(episode, "seed")
        self.earlier_look("seed", ["ep:a", "ep:b"], key="exposure-seed")

        def judge(_bench, program, *args, **kwargs):
            return {"status": "ok", "score": 0.3, "interval": None, "coverage": 1, "origins": 1, "cells": []}, object()

        with (patch.object(verify, "_key_bench"), patch.object(verify, "_open"),
              patch.object(verify, "exposure_key", side_effect=lambda _, harness, *a, **kw: f"exposure-{harness}"),
              patch.object(verify, "discover_runtime", return_value=None),
              patch.object(verify, "_extract", side_effect=lambda _l, look, scratch: (
                  [(episode, scratch / episode) for episode in look.episodes], [])),
              patch.object(verify, "judge_program", side_effect=judge) as judged,
              patch.object(verify, "save_predictions", side_effect=lambda path, *_: path.write_text("synthetic")),
              patch.object(verify, "_evaluations", return_value=(object(), object())),
              patch.object(verify, "_compare", return_value=None)):
            body, = verify.check_adoptions(self.ledger, self.run)
        self.assertNotIn("error", body)
        self.assertEqual(body["missing"], [])
        self.assertEqual(judged.call_count, 2)  # the adopted version's two programs; the replaced one is reused
        self.assertEqual(body["versions"]["parent"]["tasks"]["train"]["source_run"], "evo:earlier")
        self.assertEqual(verify.program_episodes(body["versions"]["parent"]["tasks"]["train"]), ["ep:a", "ep:b"])
        self.assertEqual(verify.program_episodes(body["versions"]["adopted"]["tasks"]["train"]), ["ep:e", "ep:f"])


if __name__ == "__main__":
    unittest.main()
