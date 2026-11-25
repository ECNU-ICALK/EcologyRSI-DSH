"""Matched development diagnostics for fixed templates and episode submissions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from ecogym.episodes.agent import Cancelled
from ecogym.episodes.judge import JUDGE_PROTOCOL, file_digest, save_predictions
from ecogym.episodes.tools import BEST_LOCAL_SELECTION, ROLLING_SELECTION, SINGLE_WINDOW_SELECTION, snapshot
from ecogym.episodes.workspace import LOCAL_STAGE, SCORED_STAGE
from ecogym.evaluation.harness import Evaluation
from ecogym.evolution.evidence import overview
from ecogym.evolution.loop import Evolution, EvolveConfig
from ecogym.kernel.ledger import Ledger
from ecogym.solvers.base import Genome

from .helpers import synthetic_bench


class TemplateAttributionTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.path = self.root / "ledger.sqlite3"
        self.bench = synthetic_bench()
        self.task = self.bench.task.id
        suite = SimpleNamespace(train=[self.task], gates=SimpleNamespace(test="t", confidence=0.95),
                                tasks=lambda _: [self.bench.task])
        with patch("ecogym.evolution.loop.suite_spec", return_value=suite):
            self.evolution = Evolution(EvolveConfig(suite="synthetic", repeats=2), self.path)
        self.evolution.benches[self.task] = self.bench
        for name, value in (("scratch_root", self.root), ("discover_runtime", Mock())):
            override = patch(f"ecogym.evolution.loop.{name}", return_value=value)
            override.start()
            self.addCleanup(override.stop)

    def harness(self, name, text):
        root = self.root / name
        (root / "template").mkdir(parents=True)
        (root / "template/predictor.py").write_text(text, encoding="utf-8")
        target = root / "snapshot"
        target.mkdir()
        return SimpleNamespace(root=root, digest=name, program=snapshot(root / "template", target))

    def evaluation(self, improvement, stage=SCORED_STAGE):
        data = self.bench.stage(stage)
        predicted = data.baseline + improvement * (data.observed - data.baseline)
        return Evaluation(Genome("test@1"), data, predicted, np.zeros(len(data.cells)))

    def entry(self, name, replicate, improvement, *, program=None, stage=SCORED_STAGE):
        label = f"{name or 'inc'}-{replicate}-{stage}"
        evaluation = self.evaluation(improvement, stage)
        path = save_predictions(self.root / f"{label}.npz", self.bench, evaluation)
        return {"task": self.task, "role": "candidate" if name else "incumbent", "candidate": name,
                "episode": label, "status": "finished", "replicate": replicate, "judge_protocol": JUDGE_PROTOCOL,
                "predictions": str(path), "predictions_digest": file_digest(path), "score_stage": SCORED_STAGE,
                "program_digest": program, "scored_start": bool(program), "score_status": "ok",
                "score_limits_match": True}

    def test_replays_only_absent_template_and_decomposes_attribution(self):
        inc, candidate = self.harness("seed", "# seed"), self.harness("candidate", "# different")
        entries = [self.entry(None, 0, 0.0, program=inc.program), self.entry(None, 1, 0.2),
                   self.entry("a", 0, 0.7), self.entry("a", 1, 0.5)]
        jobs = [(inc, "incumbent", None), (candidate, "candidate", "a")]
        evaluation = self.evaluation(0.3)
        report = {"status": "ok", "coverage": 1.0, "origins": evaluation.stage.size, "score": evaluation.score(),
                  "seconds": 0.1}
        with patch("ecogym.evolution.loop.judge_program", return_value=(report, evaluation)) as judge:
            result = self.evolution._template_attribution(jobs, entries, "measurement", [])
        judge.assert_called_once()
        self.assertEqual(judge.call_args.args[2], SCORED_STAGE)
        self.assertTrue(judge.call_args.args[1].name == "a")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["matched_arm_tasks"], 2)
        self.assertEqual(result["arms"]["incumbent"]["tasks"][self.task]["template"]["source"], "episode_start")
        mean = result["comparisons"]["a"]["mean_on_matched_tasks"]
        self.assertAlmostEqual(mean["template_gain"], 0.3)
        self.assertAlmostEqual(mean["episode_over_template_gain"], 0.2)
        self.assertAlmostEqual(mean["episode_gain"], mean["template_gain"] + mean["episode_over_template_gain"])
        horizons = result["comparisons"]["a"]["tasks"][self.task]["horizons"]
        self.assertTrue(horizons)
        for scores in horizons.values():
            self.assertAlmostEqual(scores["episode_gain"],
                                   scores["template_gain"] + scores["episode_over_template_gain"])

        prior = [{"generation": 1, "template_attribution": result}]
        with patch("ecogym.evolution.loop.judge_program") as judge:
            reused = self.evolution._template_attribution(jobs, entries, "measurement", prior)
        judge.assert_not_called()
        candidate_result = reused["arms"]["a"]["tasks"][self.task]["template"]
        self.assertEqual((candidate_result["source"], candidate_result["source_generation"]), ("prior_round", 1))

        # the evolver's overview of the next round renders the complete diagnostic
        run = "evo:render"
        with Ledger(self.path) as ledger:
            ledger.append(run, "evolve_started", {"seed": {"digest": "seed"}})
            ledger.append(run, "round_started", {"generation": 1, "incumbent": "seed"})
            ledger.append(run, "round_planned", {"id": "plan", "measurement": "measurement"})
            ledger.append(run, "round_finished", {"incumbent": "seed"})
            ledger.append(run, "template_attribution_checked", {"generation": 1, "round_plan": "plan", **result})
        text = overview(self.evolution.view(run), {}, [self.task], {})
        self.assertIn("匹配 2/2 个版本与任务组合", text)
        self.assertIn("模板分差 +0.3000", text)
        self.assertIn("相对各自模板的回合增益差 +0.2000", text)

    def test_conflicting_starts_are_flagged_and_arms_share_identical_template_replay(self):
        inc, candidate = self.harness("seed", "# same"), self.harness("candidate", "# same")
        jobs = [(inc, "incumbent", None), (candidate, "candidate", "a")]
        entries = [self.entry(None, 0, 0.1, program=inc.program), self.entry(None, 1, 0.2, program=inc.program),
                   self.entry("a", 0, 0.1), self.entry("a", 1, 0.2)]
        with patch("ecogym.evolution.loop.judge_program") as judge:
            result = self.evolution._template_attribution(jobs, entries, "measurement", [])
        judge.assert_not_called()
        for label in ("incumbent", "a"):
            result_task = result["arms"][label]["tasks"][self.task]
            self.assertEqual(result_task["template"]["status"], "drift")
            self.assertIsNone(result_task["episode_over_template"])
        self.assertEqual(result["comparisons"]["a"]["matched_tasks"], 0)

        entries = [self.entry(None, 0, 0.1, stage=LOCAL_STAGE), self.entry("a", 0, 0.1)]
        with patch("ecogym.evolution.loop.judge_program", return_value=(
                {"status": "timeout", "coverage": 0.5, "origins": 100, "seconds": 7}, Mock())) as judge:
            incomplete = self.evolution._template_attribution(jobs, entries, "measurement", [])
        judge.assert_called_once()
        task = incomplete["arms"]["incumbent"]["tasks"][self.task]
        self.assertEqual((task["episodes_unreadable"], task["episode_count"], task["template"]["coverage"]),
                         (1, 0, 0.5))
        self.assertIsNone(incomplete["comparisons"]["a"]["mean_on_matched_tasks"]["episode_gain"])

    def test_diagnostic_failure_or_stop_cannot_undo_committed_round(self):
        run, plan = "evo:test", {"id": "plan", "measurement": "measurement"}
        with Ledger(self.path) as ledger:
            ledger.append(run, "evolve_started", {"seed": {"digest": "seed"}})
            ledger.append(run, "round_started", {"generation": 0, "incumbent": "seed"})
            ledger.append(run, "round_planned", plan)
            ledger.append(run, "round_finished", {"incumbent": "seed"})
        view = self.evolution.view(run)
        with (patch.object(self.evolution, "_template_attribution", side_effect=Cancelled("stop")),
              self.assertRaises(Cancelled)):
            self.evolution._record_template_attribution(run, 0, plan, [], [], view)
        self.assertTrue(self.evolution.view(run)["rounds"][0]["finished"])
        with patch.object(self.evolution, "_template_attribution", side_effect=OSError("missing runtime")):
            self.evolution._record_template_attribution(run, 0, {**plan, "id": "wrong"}, [], [], view)
            self.evolution._record_template_attribution(run, 0, plan, [], [], view)
        folded = self.evolution.view(run)
        self.assertEqual(folded["rounds"][0]["template_attribution"]["round_plan"], "plan")
        self.assertEqual(folded["rounds"][0]["template_attribution"]["status"], "error")
        self.assertIn("诊断未能完成", overview(folded, {}, [self.task], {}))

    def test_overview_shows_the_baseline_rounds_counterexamples_and_the_runs_own_rule(self):
        """The baseline round has no comparison; its diagnostic used to read only "matched 1/2" and the rule text
        said "the latest submission" whatever the run's policy (2026-11-18 review)."""

        attribution = {"status": "partial", "matched_arm_tasks": 1, "expected_arm_tasks": 2, "expected_episodes": 2,
                       "comparisons": {}, "arms": {"incumbent": {"harness": "seed", "tasks": {
                           "t1": {"episode_count": 2, "scored_start_count": 1, "episode_over_template": -0.01,
                                  "template": {"status": "ok", "score": 0.25}},
                           "t2": {"episode_count": 1, "scored_start_count": 1, "episode_over_template": None,
                                  "template": {"status": "ok", "score": 0.17}}}}}}

        def entry(episode, task, replicate, status="finished"):
            return {"episode": episode, "task": task, "role": "incumbent", "candidate": None,
                    "replicate": replicate, "status": status, "harness": "seed"}

        start = {"local": {"score": 0.20}}
        records = {
            "ep:1": {"start": start, "scored": {"n": 3, "start": False, "score": 0.23, "cells": []},
                     "submissions": [{"n": 3, "local": {"score": 0.222}, "local_selection": {"delta": 0.022031}}]},
            "ep:2": {"start": start, "scored": {"n": None, "start": True, "score": 0.25, "cells": []},
                     "submissions": []},
            "ep:3": {"start": start, "scored": {"n": None, "start": True, "score": 0.17, "cells": []},
                     "submissions": []},
        }
        rounds = [{"generation": 0, "finished": True, "template_attribution": attribution,
                   "episodes": [entry("ep:1", "t1", 0), entry("ep:2", "t1", 1), entry("ep:3", "t2", 0),
                                entry("ep:4", "t2", 1, "infra_failed")]}]
        text = overview({"incumbent": "seed", "config": {"local_selection": BEST_LOCAL_SELECTION}, "rounds": rounds},
                        records, ["t1", "t2"], {})
        self.assertIn(f"（`{BEST_LOCAL_SELECTION}`）", text)
        self.assertIn("本地总分最高者用于评分", text)
        self.assertNotIn("最后一次被接受", text)
        self.assertIn("有成绩的重复回合 3/4", text)
        self.assertIn("| 在任 | `t1` | +0.2500 | -0.0100 | 2/2 | 1 |", text)
        self.assertIn("| 在任 | `t2` | +0.1700 | – | 1/2 | 1 |", text)
        self.assertIn("提交替换了模板的回合 1 个：隐藏期上比模板好 0 个、差 1 个", text)
        self.assertIn("在任 `t1` 重复 0（`ep:1`）：第 3 次提交，本地 +0.0220 → 隐藏期 -0.0200", text)
        legacy = overview({"incumbent": "seed", "config": {}, "rounds": rounds}, records, ["t1", "t2"], {})
        self.assertIn(f"（`{SINGLE_WINDOW_SELECTION}`）", legacy)
        self.assertIn("最后一次被接受", legacy)

    def test_local_selection_defaults_for_new_and_legacy_rounds(self):
        self.assertEqual(self.evolution.config.local_selection, BEST_LOCAL_SELECTION)
        self.assertEqual(EvolveConfig.from_dict({"suite": "legacy"}).local_selection, SINGLE_WINDOW_SELECTION)
        self.assertEqual(EvolveConfig(suite="synthetic", local_selection=ROLLING_SELECTION).local_selection,
                         ROLLING_SELECTION)
        with self.assertRaisesRegex(ValueError, "local_selection"):
            EvolveConfig(suite="synthetic", local_selection="unsupported")
