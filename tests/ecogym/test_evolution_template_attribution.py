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

    def test_local_selection_defaults_for_new_and_legacy_rounds(self):
        self.assertEqual(self.evolution.config.local_selection, BEST_LOCAL_SELECTION)
        self.assertEqual(EvolveConfig.from_dict({"suite": "legacy"}).local_selection, SINGLE_WINDOW_SELECTION)
        self.assertEqual(EvolveConfig(suite="synthetic", local_selection=ROLLING_SELECTION).local_selection,
                         ROLLING_SELECTION)
        with self.assertRaisesRegex(ValueError, "local_selection"):
            EvolveConfig(suite="synthetic", local_selection="unsupported")
