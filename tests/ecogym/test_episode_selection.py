from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ecogym.episodes import sandbox as sbx
from ecogym.episodes.judge import PRE_LOCAL_STAGE, Limits, Run, calibration_pre_stage, report_of
from ecogym.episodes.tools import (
    BEST_LOCAL_SELECTION,
    DEFAULT_LOCAL_SELECTION,
    ROLLING_SELECTION,
    SINGLE_WINDOW_SELECTION,
    WALK_FORWARD_SELECTION,
    EpisodeTools,
    rolling_selection,
    walk_forward_selection,
)
from ecogym.episodes.workspace import LOCAL_STAGE, task_brief
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
                             usage=dict, event=lambda *_: None, start={"healthy": True, "evaluation": start},
                             selection_policy=BEST_LOCAL_SELECTION)
        report = self.report(candidate)
        accepted, default = tools._selection(report, candidate, report)
        self.assertTrue(accepted)
        self.assertTrue(default["passed"])
        self.assertEqual(default["policy"], BEST_LOCAL_SELECTION)
        self.assertEqual(default["window_reason"], "insufficient_blocks")

        short_bench = synthetic_bench(frame=synthetic_frame(days=32))
        short_stage = short_bench.stage(LOCAL_STAGE)
        short_start = Evaluation(Genome("start"), short_stage, short_stage.baseline,
                                 np.zeros(len(short_stage.cells), dtype=int))
        short_cand = replace(short_start, predicted=short_stage.observed)
        short_report = report_of(short_bench, short_cand, Run(np.empty(0), "ok", [], 0.1, 0.2, ""))
        wf_tools = EpisodeTools(bench=short_bench, root=Path("."), runtime=None, limits=Limits(), deadline=0,
                                usage=dict, event=lambda *_: None, start={"healthy": True, "evaluation": short_start})
        self.assertEqual(wf_tools.selection_policy, DEFAULT_LOCAL_SELECTION)
        self.assertIsNone(calibration_pre_stage(short_bench))
        _, wf_short = wf_tools._selection(short_report, short_cand, short_report)
        self.assertTrue(wf_short["passed"])
        self.assertEqual(wf_short["policy"], WALK_FORWARD_SELECTION)
        self.assertEqual(wf_short["reason"], "passed_single_fold")

    def report(self, evaluation: Evaluation) -> dict:
        return report_of(self.bench, evaluation, Run(np.empty(0), "ok", [], 0.1, 0.2, ""))

    def test_default_gates_on_exact_total_and_preserves_failed_window_diagnostic(self) -> None:
        tools = EpisodeTools(bench=self.bench, root=Path("."), runtime=None, limits=Limits(), deadline=0,
                             usage=dict, event=lambda *_: None, start={"healthy": True, "evaluation": self.anchor},
                             selection_policy=BEST_LOCAL_SELECTION)
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

    def test_a_small_local_gain_hears_the_margin_and_the_short_window_caution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = root / "work" / "submission"
            program.mkdir(parents=True)
            (program / "predictor.py").write_text("# snapshot fixture\n", encoding="utf-8")
            start = {"local": {"score": 0.0}, "healthy": True, "evaluation": self.anchor}
            tools = EpisodeTools(bench=self.bench, root=root, runtime=None, limits=Limits(),
                                 deadline=clock.now() + 60, usage=dict, event=lambda *_: None, start=start,
                                 selection_policy=BEST_LOCAL_SELECTION)
            small, worse = self.window_scores((0.01, 0.01, 0.01)), self.window_scores((-0.01, -0.01, -0.01))
            with patch("ecogym.episodes.tools.judge_program",
                       side_effect=[(self.report(x), x) for x in (small, worse)]):
                gained, lost = tools.submit(), tools.submit()
        for result in (gained, lost):
            self.assertTrue(result["accepted"])
            self.assertFalse(result["replaces_start"])
            self.assertIn("替换至少要高 0.02", result["verdict"])
            self.assertIn("现在评分用起始程序", result["verdict"])
        self.assertIn("短得多", gained["verdict"])
        self.assertNotIn("短得多", lost["verdict"])

    def test_task_brief_keeps_the_short_local_window_caution(self) -> None:
        brief = task_brief(self.bench, {"scored_days": 10}, minutes=8, tokens=1000, deadline=clock.now() + 60,
                           limits=Limits(), start_local=0.05, selection_policy=BEST_LOCAL_SELECTION)
        self.assertIn("本地总分为 +0.0500", brief)
        self.assertIn("本地只高一点的改动常常不能在隐藏期上成立", brief)
        self.assertIn("`classical_reference`", brief)

    def test_walk_forward_blocks_pre_fold_regression_and_ranks_by_two_fold_score(self) -> None:
        pre_built = calibration_pre_stage(self.bench)
        self.assertIsNotNone(pre_built)
        _, pre_data = pre_built
        pre_stage = replace(pre_data, observed=pre_data.baseline + 1)
        start_pre = Evaluation(Genome("start-pre"), pre_stage, pre_stage.observed - np.ones(pre_stage.size),
                               np.zeros(len(pre_stage.cells), dtype=int))
        good_pre = Evaluation(Genome("good-pre"), pre_stage, pre_stage.observed - np.full(pre_stage.size, 0.90),
                              np.zeros(len(pre_stage.cells), dtype=int))
        bad_pre = Evaluation(Genome("bad-pre"), pre_stage, pre_stage.observed - np.full(pre_stage.size, 1.05),
                             np.zeros(len(pre_stage.cells), dtype=int))

        uq_cand = self.window_scores((0.04, 0.04, 0.04))
        blocked = walk_forward_selection(uq_cand, self.anchor, candidate_pre=bad_pre, start_pre=start_pre)
        self.assertFalse(blocked["passed"])
        self.assertEqual(blocked["reason"], "pre_fold_regression")

        passed = walk_forward_selection(uq_cand, self.anchor, candidate_pre=good_pre, start_pre=start_pre)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["reason"], "passed")
        self.assertGreater(passed["walk_forward_delta"], 0.05)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sub_dir = root / "work" / "submission"
            sub_dir.mkdir(parents=True)
            start_prog = root / "submissions" / "start" / "program"
            start_prog.mkdir(parents=True)
            (start_prog / "predictor.py").write_text("# start\n", encoding="utf-8")
            start = {"digest": "start_dig", "program": str(start_prog), "local": {"score": 0.0},
                     "healthy": True, "evaluation": self.anchor, "pre_evaluation": start_pre}
            tools = EpisodeTools(bench=self.bench, root=root, runtime=None, limits=Limits(),
                                 deadline=clock.now() + 60, usage=dict, event=lambda *_: None,
                                 start=start, selection_policy=WALK_FORWARD_SELECTION, reuse_local=True)

            # Candidate 1: high uq (+0.08), negative pre (-0.048) -> blocked!
            (sub_dir / "predictor.py").write_text("# overfit uq\n", encoding="utf-8")
            overfit_uq = self.window_scores((0.08, 0.08, 0.08))
            with patch("ecogym.episodes.tools.judge_program",
                       side_effect=[(self.report(overfit_uq), overfit_uq), (self.report(bad_pre), bad_pre)]) as judge:
                sub1 = tools.submit()
                self.assertFalse(sub1["replaces_start"])
                self.assertEqual(sub1["local_selection"]["reason"], "pre_fold_regression")
                self.assertIn("更早拟合折前滚检验", sub1["verdict"])
                self.assertEqual(judge.call_count, 2)

            # Candidate 2: moderate uq (+0.04), strong pre (+0.10) -> passes, AND evaluate_local + submit reuses cache!
            (sub_dir / "predictor.py").write_text("# balanced model\n", encoding="utf-8")
            with patch("ecogym.episodes.tools.judge_program",
                       side_effect=[(self.report(uq_cand), uq_cand), (self.report(good_pre), good_pre)]) as judge:
                ev2 = tools.evaluate_local()
                self.assertTrue(ev2["local_selection"]["passed"])
                sub2 = tools.submit()
                self.assertTrue(sub2["replaces_start"])
                self.assertTrue(sub2["local_selection"].get("reused_local"))
                self.assertEqual(judge.call_count, 2)  # NOT 4: submit reused both LOCAL_STAGE and PRE_LOCAL_STAGE!
                self.assertEqual([c.args[2] for c in judge.call_args_list], [LOCAL_STAGE, PRE_LOCAL_STAGE])
            self.assertEqual(tools.final()["n"], sub2["n"])

    def test_seed_backtest_excludes_local_tail_and_streams_every_step(self) -> None:
        try:
            runtime = sbx.discover_runtime()
        except (sbx.SandboxUnavailable, PermissionError):
            self.skipTest("agent runtime not available")
        bt_path = Path(__file__).resolve().parents[2] / "src/ecogym/harness/seed/template/ecotk/backtest.py"
        script = f"""
import importlib.util, json
import numpy as np, pandas as pd

spec_bt = importlib.util.spec_from_file_location("seed_backtest", {str(bt_path)!r})
bt = importlib.util.module_from_spec(spec_bt)
spec_bt.loader.exec_module(bt)

idx = pd.date_range("2025-01-01", periods=24 * 45, freq="1h")
vals = np.sin(np.arange(len(idx)) * 2 * np.pi / 24.0)
vals[24 * 25 + 3] = np.nan
df = pd.DataFrame({{"y": vals}}, index=idx)
spec = {{"targets": ["y"], "horizons": [1, 6], "step": "1h", "season_steps": 24,
         "history_steps": 4, "baselines": {{"y@1": "persistence", "y@6": "seasonal"}}}}
spans_fit = bt.windows(df, count=3, days=5.0, step="1h", fit_only=True)
spans_all = bt.windows(df, count=3, days=5.0, step="1h", fit_only=False)
seen = []
class StatefulPredictor:
    def __init__(self, s):
        self.s = s
    def fit(self, train_df):
        pass
    def forecast(self, hist_df):
        seen.append(len(hist_df))
        last_finite = hist_df["y"].dropna().iloc[-1]
        return {{"y": [float(last_finite), float(last_finite)]}}
res = bt.run(StatefulPredictor, spec, df, idx[24 * 25], idx[24 * 27])
print(json.dumps({{"fit_before_all": str(spans_fit[-1][1]) < str(spans_all[-1][1]),
                   "calls": len(seen), "score": res["score"], "horizons": res["horizons"]}}))
"""
        proc = subprocess.run([str(runtime.python), "-c", script], check=True, capture_output=True, text=True)
        payload = json.loads(proc.stdout.strip())
        self.assertTrue(payload["fit_before_all"])
        self.assertEqual(payload["calls"], 48)
        self.assertIsNotNone(payload["score"])
        self.assertIn("1", payload["horizons"])
        self.assertIn("6", payload["horizons"])
