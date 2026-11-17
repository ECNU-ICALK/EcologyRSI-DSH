from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ecogym.episodes.judge import Limits, Run, report_of
from ecogym.episodes.tools import (
    BEST_LOCAL_SELECTION,
    ROLLING_SELECTION,
    SINGLE_WINDOW_SELECTION,
    EpisodeTools,
    rolling_selection,
)
from ecogym.episodes.workspace import LOCAL_STAGE
from ecogym.evaluation.harness import Evaluation
from ecogym.kernel import clock
from ecogym.solvers import Genome

from .helpers import synthetic_bench, synthetic_frame


class LocalSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bench = synthetic_bench(frame=synthetic_frame(days=120))
        data = self.bench.stage(LOCAL_STAGE)
        self.stage = replace(data, observed=data.baseline + 1)
        self.anchor = self.evaluation(np.ones(self.stage.size))
        self.windows = [np.isin(self.stage.blocks, blocks)
                        for blocks in np.array_split(np.unique(self.stage.blocks), 3)]

    def evaluation(self, errors: np.ndarray) -> Evaluation:
        return Evaluation(Genome("test"), self.stage, self.stage.observed - errors,
                          np.zeros(len(self.stage.cells), dtype=int))

    def window_scores(self, gains: tuple[float, float, float]) -> Evaluation:
        errors = np.ones(self.stage.size)
        for mask, gain in zip(self.windows, gains, strict=True):
            errors[mask] = 1 - gain if gain >= 0 else 1 / (1 + gain)
        return self.evaluation(errors)

    def test_two_wins_required_and_one_large_window_loss_blocks_replacement(self) -> None:
        gained = rolling_selection(self.window_scores((0.10, 0.06, -0.01)), self.anchor)
        self.assertTrue(gained["passed"])
        self.assertEqual(len(gained["windows"]), 3)
        self.assertEqual(sum(window["origins"] for window in gained["windows"]), self.stage.size)
        self.assertGreaterEqual(min(window["blocks"] for window in gained["windows"]), 2)
        for a, b in zip(gained["windows"], gained["windows"][1:], strict=False):
            self.assertLess(a["last_origin"], b["first_origin"])
        for gains, reason in (
            ((0.01, 0.01, 0.01), "overall_gain_below_margin"),
            ((0.20, 0.20, -0.08), "window_loss_above_cap"),
            ((0.40, 0.0, 0.0), "insufficient_window_wins"),
        ):
            with self.subTest(gains=gains):
                result = rolling_selection(self.window_scores(gains), self.anchor)
                self.assertFalse(result["passed"])
                self.assertEqual(result["reason"], reason)

    def test_fewer_than_six_eligible_blocks_cannot_replace_a_healthy_start(self) -> None:
        bench = synthetic_bench()
        stage = bench.stage(LOCAL_STAGE)
        start = Evaluation(Genome("start"), stage, stage.baseline, np.zeros(len(stage.cells), dtype=int))
        candidate = replace(start, predicted=stage.observed)
        result = rolling_selection(candidate, start)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "insufficient_blocks")
        tools = EpisodeTools(bench=bench, root=Path("."), runtime=None, limits=Limits(), deadline=0,
                             usage=dict, event=lambda *_: None, start={"healthy": True, "evaluation": start})
        report = self.report(candidate)
        accepted, default = tools._selection(report, candidate, report)
        self.assertTrue(accepted)
        self.assertTrue(default["passed"])
        self.assertEqual(default["policy"], BEST_LOCAL_SELECTION)
        self.assertEqual(default["window_reason"], "insufficient_blocks")

    def report(self, evaluation: Evaluation) -> dict:
        return report_of(self.bench, evaluation, Run(np.empty(0), "ok", [], 0.1, 0.2, ""))

    def test_default_gates_on_exact_total_and_preserves_failed_window_diagnostic(self) -> None:
        tools = EpisodeTools(bench=self.bench, root=Path("."), runtime=None, limits=Limits(), deadline=0,
                             usage=dict, event=lambda *_: None, start={"healthy": True, "evaluation": self.anchor})
        candidate = self.window_scores((0.2, 0.2, -0.08))
        report = self.report(candidate)
        accepted, result = tools._selection(report, candidate, report)
        self.assertTrue(accepted)
        self.assertTrue(result["passed"])
        self.assertFalse(result["window_passed"])
        self.assertEqual(result["window_reason"], "window_loss_above_cap")
        self.assertEqual(len(result["windows"]), 3)

        candidate = self.window_scores((0.01999, 0.01999, 0.01999))
        report = self.report(candidate)
        self.assertEqual(round(report["score"], 4), 0.02)
        self.assertFalse(tools._selection(report, candidate, report)[1]["passed"])

    def test_healthy_submission_can_recover_invalid_start_without_gain_requirement(self) -> None:
        tools = EpisodeTools(bench=self.bench, root=Path("."), runtime=None, limits=Limits(), deadline=0,
                             usage=dict, event=lambda *_: None, start={"healthy": False})
        candidate = self.window_scores((0.0, 0.0, 0.0))
        report = self.report(candidate)
        accepted, result = tools._selection(report, candidate, report)
        self.assertTrue(accepted)
        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "recover_invalid_start")

    def test_rolling_keeps_best_passing_submission_and_legacy_keeps_latest(self) -> None:
        strong = self.window_scores((0.12, 0.12, 0.12))
        weak = self.window_scores((0.05, 0.05, 0.05))
        missing = replace(strong, predicted=strong.predicted.copy())
        missing.predicted[0, 0] = np.nan
        for policy, expected in ((BEST_LOCAL_SELECTION, 2), (ROLLING_SELECTION, 2), (SINGLE_WINDOW_SELECTION, 4)):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                program = root / "work" / "submission"
                program.mkdir(parents=True)
                (program / "predictor.py").write_text("# snapshot fixture\n", encoding="utf-8")
                events = []
                start = {"local": {"score": 0.0}, "healthy": True, "evaluation": self.anchor}
                tools = EpisodeTools(bench=self.bench, root=root, runtime=None, limits=Limits(),
                                     deadline=clock.now() + 60, usage=dict,
                                     event=lambda kind, body, events=events: events.append((kind, body)), start=start,
                                     selection_policy=policy)
                checks = [strong, strong, weak, weak, missing]
                with patch("ecogym.episodes.tools.judge_program",
                           side_effect=[(self.report(x), x) for x in checks]) as judge:
                    evaluation = tools.evaluate_local()
                    self.assertTrue(evaluation["local_selection"]["passed"])
                    for _ in range(3):
                        self.assertTrue(tools.submit()["replaces_start"])
                    if policy != SINGLE_WINDOW_SELECTION:
                        self.assertFalse(tools.submit()["accepted"])
                        self.assertEqual(judge.call_count, 5)
                    else:
                        self.assertEqual(judge.call_count, 4)
                    self.assertTrue(all(call.args[2] == LOCAL_STAGE for call in judge.call_args_list))
                self.assertEqual(tools.final()["n"], expected)
                self.assertEqual(tools.task_info()["scored_now"], expected)
                self.assertEqual(events[0][1]["local_selection"], evaluation["local_selection"])
