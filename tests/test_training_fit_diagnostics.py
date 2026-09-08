from __future__ import annotations

from dataclasses import replace
import json
import math
import unittest

from ecologyrsi_dsh.data.contracts import DatasetSeries
from ecologyrsi_dsh.data.greenhouse import FeatureSpec
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.greenhouse_prediction import (
    EXOGENOUS_RIDGE_MODEL_ID, fit_predict_exogenous_ridge,
)
from ecologyrsi_dsh.science.diagnostics import (
    DiagnosticCandidate, HORIZONS, TARGETS, candidates_from_payload,
    compare_host_execution, run_training_fit_diagnostics, training_fit_only,
)


def series(*, irregular=False, missing=False):
    timestamps = tuple(i for i in range(900) if not irregular or i not in {80, 242, 410, 623})
    values = {}
    for name, center, amplitude in zip(TARGETS, (20, 70, 700), (4, 8, 80)):
        values[name] = tuple(center + amplitude * math.sin(t * 2 * math.pi / 24) + .05 * math.cos(t / 9)
                             if not missing or t not in {190, 445, 682} else None for t in timestamps)
    values["ventilation"] = tuple(math.cos(t / 8) for t in timestamps)
    features = {name: FeatureSpec(name, name, "action" if name == "ventilation" else "environment", "unit")
                for name in values}
    return DatasetSeries(schema="ecologyrsi-dsh.dataset-series/1", dataset_id="agc_cucumber_2018",
                         domain_id="greenhouse_cucumber_2018", episode_id="diagnostic-test", digest="a" * 64,
                         timestamps=timestamps, values=values, features=features,
                         partitions={"training_fit": IndexRange(0, 720), "training_feedback": IndexRange(720, len(timestamps))},
                         split_manifest_digest_sha256="b" * 64)


def specs():
    return (DiagnosticCandidate("seed", EXOGENOUS_RIDGE_MODEL_ID,
                                {"history_steps": 3, "ridge_alpha": .1, "residual_scale": 1}),
            DiagnosticCandidate("champion", EXOGENOUS_RIDGE_MODEL_ID,
                                {"history_steps": 6, "ridge_alpha": .3, "residual_scale": 1}))


def without_timing(report):
    value = json.loads(json.dumps(report))
    for fold in value["folds"]:
        for arm in fold["arms"]:
            arm.pop("fit_seconds", None)
            comparison = arm.get("execution_comparison", {})
            comparison.pop("flat_host_seconds", None)
            comparison.pop("origin_grouped_host_seconds", None)
    return value


class TrainingFitDiagnosticsTests(unittest.TestCase):
    def test_non_fit_values_are_physically_excluded_before_any_runner(self):
        source = series()
        changed = replace(source, values={name: column[:720] + (999999.0,) * (len(column) - 720)
                                          for name, column in source.values.items()})
        seen = []
        def run(view, **kwargs):
            seen.append(view)
            self.assertLessEqual(len(view.timestamps), 720)
            self.assertNotIn(999999.0, view.values["air_temperature"])
            return fit_predict_exogenous_ridge(view, **kwargs)
        first = run_training_fit_diagnostics(source, candidates=specs(), fit_runner=run)
        second = run_training_fit_diagnostics(changed, candidates=specs(), fit_runner=run)
        self.assertEqual(without_timing(first), without_timing(second))
        self.assertEqual(len(seen), 12)
        self.assertEqual(set(training_fit_only(source).partitions), {"training_fit", "training_feedback"})

    def test_forward_folds_use_elapsed_purge_and_complete_shared_nine_cells(self):
        report = run_training_fit_diagnostics(series(irregular=True, missing=True), candidates=specs())
        self.assertGreater(report["source"]["gap_count"], 0)
        self.assertEqual(len(report["folds"]), 3)
        for fold in report["folds"]:
            self.assertGreater(fold["first_feedback_time"] - fold["last_fit_label_time"], 24)
            self.assertEqual(fold["common_scoring_cells"], 9 * fold["common_origin_count"])
            self.assertEqual(len(fold["baseline_selection"]), 9)
            for arm in fold["arms"]:
                self.assertEqual(len(arm["cells"]), 9)
                self.assertTrue(all(cell["n"] == fold["common_origin_count"] for cell in arm["cells"]))
                if "execution_comparison" in arm:
                    comparison = arm["execution_comparison"]
                    self.assertFalse(comparison["real_native_llm_executed"])
                    for key in ("prediction_identity_equal", "mask_equal", "case_identity_equal", "baseline_equal", "scoring_equal"):
                        self.assertTrue(comparison[key])
            selected = next(arm for arm in fold["arms"] if arm["name"] == "fit_selected_baseline")
            self.assertEqual(selected["mean_skill_score"], 0.0)

    def test_future_labels_do_not_change_fit_or_predictions_at_earlier_origins(self):
        source = series()
        changed_values = dict(source.values)
        changed_values["air_temperature"] = tuple(value + 100 if t >= 380 else value
                                                   for t, value in zip(source.timestamps, source.values["air_temperature"]))
        changed = replace(source, values=changed_values)
        captured = []
        def run(view, **kwargs):
            fitted = fit_predict_exogenous_ridge(view, **kwargs)
            config = kwargs["config"]
            rows = [r for r in fitted["prediction_rows"] if r["partition"] == "training_feedback"]
            outputs, audit = compare_host_execution(rows, fitted["models"], config)
            captured.append((fitted, outputs, audit))
            return fitted
        run_training_fit_diagnostics(source, candidates=specs()[:1], fit_runner=run)
        run_training_fit_diagnostics(changed, candidates=specs()[:1], fit_runner=run)
        first, second = captured[0], captured[3]
        self.assertEqual(first[2]["fitted_snapshot_digest"], second[2]["fitted_snapshot_digest"])
        self.assertEqual([m["coefficients"] for m in first[0]["models"]],
                         [m["coefficients"] for m in second[0]["models"]])
        def earlier(rows):
            return {(r["origin_timestamp"], r["target"], r["horizon_hours"]): r["predicted"]
                    for r in rows if r["origin_timestamp"] < 380}
        self.assertTrue(earlier(first[1]))
        self.assertEqual(earlier(first[1]), earlier(second[1]))

    def test_bad_capacity_and_configuration_fail_closed(self):
        source = series()
        for kwargs in ({"folds": 2}, {"folds": 11}, {"purge_hours": 23},
                       {"candidates": []}, {"initial_fit_fraction": .9}):
            with self.assertRaises(ValueError):
                run_training_fit_diagnostics(source, **kwargs)
        with self.assertRaises(ValueError):
            run_training_fit_diagnostics(replace(source, partitions={"training_fit": IndexRange(0, 100)}), candidates=specs())
        with self.assertRaises(ValueError):
            candidates_from_payload([{"name": "unsafe", "model_id": "unknown", "parameters": {}}])
        with self.assertRaises(ValueError):
            candidates_from_payload([{"name": "bad", "model_id": EXOGENOUS_RIDGE_MODEL_ID,
                                      "parameters": {"history_steps": 3, "ridge_alpha": 2, "residual_scale": 1}}])

    def test_report_is_aggregate_only_and_candidate_specs_round_trip(self):
        candidates = candidates_from_payload([spec.to_dict() for spec in specs()])
        report = run_training_fit_diagnostics(series(), candidates=candidates)
        forbidden = {"observed", "prediction_rows", "feature_snapshot", "history_window", "coefficients", "label_free_context"}
        def scan(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for child in value.values(): scan(child)
            elif isinstance(value, list):
                for child in value: scan(child)
        scan(report)
        self.assertLess(len(json.dumps(report, allow_nan=False)), 150000)
        self.assertEqual(report["remote_requests"], 0)
        self.assertEqual(report["ledger_mutations"], 0)

    def test_baseline_aligned_zero_scale_matches_fold_fit_selected_baseline(self):
        aligned = DiagnosticCandidate("aligned_zero", "greenhouse-baseline-aligned-ridge@1",
                                      {"history_steps": 6, "ridge_alpha": .3, "residual_scale": 0})
        report = run_training_fit_diagnostics(series(), candidates=(*specs(), aligned))
        for fold in report["folds"]:
            actual = next(arm for arm in fold["arms"] if arm["name"] == "aligned_zero")
            reference = next(arm for arm in fold["arms"] if arm["name"] == "fit_selected_baseline")
            self.assertEqual(actual["scoring_digest"], reference["scoring_digest"])
            self.assertEqual(actual["mean_skill_score"], 0)
            self.assertTrue(actual["execution_comparison"]["prediction_identity_equal"])

    def test_single_horizon_zero_ablation_preserves_other_horizons(self):
        common = {"history_steps": 6, "ridge_alpha": .3,
                  "residual_scale_1h": .25, "residual_scale_6h": .25, "residual_scale_24h": .25}
        candidates = (DiagnosticCandidate("aligned", "greenhouse-baseline-aligned-ridge@1", common),
                      DiagnosticCandidate("six_hour_zero", "greenhouse-baseline-aligned-ridge@1",
                                          {**common, "residual_scale_6h": 0}))
        report = run_training_fit_diagnostics(series(), candidates=candidates)
        for fold in report["folds"]:
            first, second = fold["arms"][:2]
            for left, right in zip(first["cells"], second["cells"], strict=True):
                if left["horizon_hours"] == 6:
                    self.assertEqual(right["rmse"], right["baseline_rmse"])
                    self.assertEqual(right["skill_score"], 0)
                else:
                    self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
