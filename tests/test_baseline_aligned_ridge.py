from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import unittest

from ecologyrsi_dsh.core.models import Candidate, Proposal, TaskManifest
from ecologyrsi_dsh.core.sample_results import build_sample_results
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.baselines import apply_baseline_profile
from ecologyrsi_dsh.evaluators.greenhouse_prediction import (
    BASELINE_ALIGNED_RIDGE_MODEL_ID,
    BaselineAlignedRidgeConfig,
    ExogenousRidgeConfig,
    fit_predict_exogenous_ridge,
    predict_fitted_exogenous_ridge,
)
from ecologyrsi_dsh.evaluators.registry import (
    EvaluatorRegistry,
    GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID,
    GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID,
)
from ecologyrsi_dsh.knowledge.algorithms import compile_algorithm_spec
from ecologyrsi_dsh.knowledge.algorithm_smoke import smoke_test_algorithm_spec
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_evaluation import (
    _DatasetStub, _DshOriginRuntimeStub, _dsh_prediction_tool_binder,
)
from tests.test_greenhouse_prediction import _series


TARGETS = ("air_temperature", "relative_humidity", "co2_concentration")


def periodic_series():
    timestamps = tuple(range(300))
    values = {
        name: tuple(center + amplitude * math.sin(2 * math.pi * t / 24)
                    for t in timestamps)
        for name, center, amplitude in zip(TARGETS, (22, 70, 700), (5, 15, 200))
    }
    return _series(timestamps, values, dict.fromkeys(TARGETS, "environment"),
                   fit=(0, 168), feedback=(168, 280))


def run(series, scale=.5, *, deferred=False, config=None):
    return fit_predict_exogenous_ridge(
        series, targets=TARGETS, horizons=(1, 6, 24),
        config=config or BaselineAlignedRidgeConfig(6, .1, scale),
        evaluation_history_steps=12,
        defer_prediction_partitions=("training_feedback",) if deferred else (),
    )


class BaselineAlignedRidgeTests(unittest.TestCase):
    def test_horizon_scale_zero_preserves_baseline_without_disabling_other_horizons(self):
        source = periodic_series()
        shared = run(source)
        config = BaselineAlignedRidgeConfig(6, .1, .5, residual_scale_6h=0.0)
        split = run(source, config=config)
        self.assertEqual(BaselineAlignedRidgeConfig.from_mapping(config.to_dict()).to_dict(), config.to_dict())
        for original, candidate in zip(shared["prediction_rows"], split["prediction_rows"]):
            if candidate["horizon_hours"] == 6:
                self.assertEqual(candidate["predicted"], candidate["baseline"])
            else:
                self.assertEqual(original["predicted"], candidate["predicted"])
        with self.assertRaises(ValueError):
            BaselineAlignedRidgeConfig.from_mapping({**config.to_dict(), "residual_scale_6h": 1.1})

    def test_zero_residual_equals_frozen_scoring_baseline_for_nine_cells(self):
        series = periodic_series()
        result = run(series, 0)
        self.assertEqual(result["model_id"], BASELINE_ALIGNED_RIDGE_MODEL_ID)
        self.assertEqual(len(result["models"]), 9)
        policy = result["baseline_profile"]
        self.assertEqual(policy["selection_partition"], "training_fit")
        self.assertTrue(all(cell["baseline_id"] == "seasonal_24h"
                            for cell in policy["cells"] if cell["horizon_hours"] == 6))
        # Reconstruct persistence source rows, then independently invoke the
        # scoring baseline implementation. This guards predictor/scorer drift.
        persistence_rows = [
            {**row, "baseline": row["label_free_context"]["history_window"][0]}
            for row in result["prediction_rows"]
        ]
        scoring_rows = apply_baseline_profile(series, persistence_rows, policy)
        for prediction, scored in zip(result["prediction_rows"], scoring_rows):
            self.assertEqual(prediction["predicted"], scored["baseline"])
            self.assertEqual(prediction["baseline"], scored["baseline"])
            reference = prediction["baseline_reference"]
            self.assertLessEqual(reference["source_timestamp"], prediction["origin_timestamp"])
        seasonal = next(row for row in result["prediction_rows"]
                        if row["partition"] == "training_feedback" and row["horizon_hours"] == 6)
        self.assertFalse(seasonal["used_target_persistence"])

    def test_deferred_replay_matches_eager_and_rejects_reference_or_config_drift(self):
        series = periodic_series()
        eager = run(series)
        deferred = run(series, deferred=True)
        expected = {(r["target"], r["horizon_hours"], r["origin_timestamp"]): r
                    for r in eager["prediction_rows"]}
        # JSON round-tripping is exercised by the actual durable public shape.
        import json
        deferred = json.loads(json.dumps(deferred))
        config = BaselineAlignedRidgeConfig.from_mapping(deferred["parameters"])
        for row in deferred["prediction_rows"]:
            if row["partition"] != "training_feedback":
                continue
            self.assertNotIn("predicted", row)
            actual = predict_fitted_exogenous_ridge(
                target=row["target"], horizon_hours=row["horizon_hours"],
                baseline=row["baseline"], label_free_context=row["label_free_context"],
                models=deferred["models"], config=config,
            )
            self.assertEqual(actual, expected[row["target"], row["horizon_hours"],
                                              row["origin_timestamp"]]["predicted"])
        row = next(r for r in deferred["prediction_rows"]
                   if r["partition"] == "training_feedback" and r["horizon_hours"] == 6)
        for field, value in (("source_timestamp", row["origin_timestamp"] + 1),
                             ("baseline_profile_digest", "other"),
                             ("baseline_id", "persistence")):
            context = deepcopy(row["label_free_context"])
            context["baseline_reference"][field] = value
            with self.assertRaisesRegex(ValueError, "aligned ridge"):
                predict_fitted_exogenous_ridge(
                    target=row["target"], horizon_hours=6, baseline=row["baseline"],
                    label_free_context=context, models=deferred["models"], config=config)
        with self.assertRaisesRegex(ValueError, "identity"):
            predict_fitted_exogenous_ridge(
                target=row["target"], horizon_hours=6, baseline=row["baseline"],
                label_free_context=row["label_free_context"], models=deferred["models"],
                config=replace(config, ridge_alpha=.3))

    def test_fit_and_origin_inputs_are_independent_of_later_labels(self):
        source = periodic_series()
        original = run(source)
        modified_values = {name: tuple(value if i < 168 else value + 13
                                       for i, value in enumerate(values))
                           for name, values in source.values.items()}
        changed = run(replace(source, values=modified_values))
        self.assertEqual(original["models"], changed["models"])
        self.assertEqual(original["baseline_profile"], changed["baseline_profile"])
        cutoff = 200
        future_values = {name: tuple(value if i <= cutoff else value + 17
                                     for i, value in enumerate(values))
                         for name, values in source.values.items()}
        changed = run(replace(source, values=future_values))
        for result in (original, changed):
            self.assertEqual(len([r for r in result["prediction_rows"]
                                  if r["origin_timestamp"] == cutoff]), 9)
        for before, after in zip(original["prediction_rows"], changed["prediction_rows"]):
            if before["origin_timestamp"] == cutoff:
                self.assertEqual(before["predicted"], after["predicted"])
                self.assertEqual(before["label_free_context"], after["label_free_context"])
        tail_values = {name: tuple(value if i < 280 else 999999
                                   for i, value in enumerate(values))
                       for name, values in source.values.items()}
        self.assertEqual(original, run(replace(source, values=tail_values)))

    def test_missing_seasonal_reference_falls_back_causally_and_is_audited(self):
        source = periodic_series()
        values = {name: list(items) for name, items in source.values.items()}
        # Origin 179 is eligible with the shared 12-step history. Its 6h
        # seasonal reference is t=161, before the feedback partition.
        values["air_temperature"][161] = None
        source = replace(source, values={k: tuple(v) for k, v in values.items()})
        result = run(source, 0)
        row = next(r for r in result["prediction_rows"]
                   if r["target"] == "air_temperature" and r["horizon_hours"] == 6
                   and r["origin_timestamp"] == 179)
        reference = row["baseline_reference"]
        self.assertEqual(reference["requested_baseline_id"], "seasonal_24h")
        self.assertEqual(reference["baseline_id"], "persistence")
        self.assertEqual(reference["fallback_reason"], "seasonal_reference_missing")
        self.assertEqual(reference["source_timestamp"], 179)
        self.assertEqual(row["predicted"], source.values["air_temperature"][179])
        self.assertTrue(row["used_target_persistence"])
        # A finite value in a non-visible gap must also remain unavailable.
        gap_source = replace(periodic_series(), partitions={
            "training_fit": IndexRange(0, 155), "training_feedback": IndexRange(168, 280),
            "development": IndexRange(155, 168),
        })
        gap_result = run(gap_source, 0)
        gap = next(r for r in gap_result["prediction_rows"] if r["target"] == "air_temperature"
                   and r["horizon_hours"] == 6 and r["origin_timestamp"] == 179)
        self.assertEqual(gap["baseline_reference"]["fallback_reason"], "seasonal_reference_missing")

    def test_legacy_predictions_match_where_the_selected_baseline_is_persistence(self):
        series = periodic_series()
        legacy = run(series, config=ExogenousRidgeConfig(6, .1, .5))
        aligned = run(series)
        selected = {(c["target"], c["horizon_hours"]): c["baseline_id"]
                    for c in aligned["baseline_profile"]["cells"]}
        for old, new in zip(legacy["prediction_rows"], aligned["prediction_rows"]):
            self.assertNotIn("baseline_reference", old)
            if selected[old["target"], old["horizon_hours"]] == "persistence":
                self.assertEqual(old["predicted"], new["predicted"])

    def test_registered_seed_compiles_smokes_and_scores_nine_cells(self):
        series = periodic_series()
        params = BaselineAlignedRidgeConfig(6, .1, 0.0).to_dict()
        task = TaskManifest(
            task_id="aligned", objective="test aligned forecasting",
            domain_pack="greenhouse_environment@1", visible_datasets=(series.dataset_id,),
            budget={"max_candidates": 1}, metadata={
                "episode_id": series.episode_id, "prediction_model_id": BASELINE_ALIGNED_RIDGE_MODEL_ID,
                "evaluator_id": GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID,
                "dataset_digest": series.digest, "split_manifest_digest": series.split_manifest_digest_sha256,
            })
        proposal = Proposal(proposal_id="proposal:aligned", run_id="run:aligned",
                            generation=0, title="aligned", changes=params)
        candidate = Candidate(candidate_id="candidate:aligned", run_id=proposal.run_id,
                              proposal_id=proposal.proposal_id, generation=0)
        registry = EvaluatorRegistry(_DatasetStub(series))
        compiled = compile_algorithm_spec(task, proposal, None)
        self.assertEqual(compiled.adapter_id, BASELINE_ALIGNED_RIDGE_MODEL_ID)
        self.assertIn("host.fit.partition-selected-baseline@1",
                      [item["operator_id"] for item in compiled.algorithm_ir["operators"]])
        smoke = smoke_test_algorithm_spec(compiled, task, datasets=_DatasetStub(series))
        self.assertEqual(smoke["status"], "passed")
        self.assertEqual(smoke["source_partition"], "training_fit")
        bundle = registry.evaluate_scientific(task, candidate, proposal)
        self.assertEqual(bundle.artifact.model_id, BASELINE_ALIGNED_RIDGE_MODEL_ID)
        self.assertEqual(bundle.artifact.parameters, params)
        self.assertEqual(len(bundle.evaluation.metrics["targets"]), 9)
        self.assertEqual(bundle.evaluation.score, 0.0)
        self.assertTrue(all(r["rmse"] == r["baseline_rmse"]
                            for r in bundle.evaluation.metrics["targets"]))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            registry.validate_binding(series.dataset_id, GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID,
                                      BASELINE_ALIGNED_RIDGE_MODEL_ID)
        template = current_program_registry().seed_template("greenhouse-baseline-aligned-default@1")
        self.assertEqual(template.to_dict()["scientific_program"]["predictor_ref"]["id"],
                         BASELINE_ALIGNED_RIDGE_MODEL_ID)

    def test_native_agent_path_executes_aligned_fitted_tool_with_complete_origin_vector(self):
        series = periodic_series()
        runtime = _DshOriginRuntimeStub()
        task = TaskManifest(
            task_id="aligned-native", objective="test native aligned forecasting",
            domain_pack="greenhouse_environment@1", visible_datasets=(series.dataset_id,),
            budget={"max_candidates": 1}, metadata={
                "episode_id": series.episode_id,
                "prediction_model_id": BASELINE_ALIGNED_RIDGE_MODEL_ID,
                "evaluator_id": GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID,
                "dataset_digest": series.digest,
                "split_manifest_digest": series.split_manifest_digest_sha256,
                "sample_agent_mode": "dsh_native_agent",
                "sample_agent_protocol": "dsh-strict-origin-bundle@3",
                "sample_agent_batch_size": 9, "sample_concurrency": 1,
                "prediction_cells_per_origin": 9, "samples_per_update": 9,
                "strategy_model_id": "dsh/strategy", "review_model_id": "dsh/review",
            })
        proposal = Proposal(proposal_id="proposal:aligned-native", run_id="run:aligned-native",
                            generation=0, title="aligned native",
                            changes=BaselineAlignedRidgeConfig(6, .1, 0.0).to_dict())
        candidate = Candidate(candidate_id="candidate:aligned-native", run_id=proposal.run_id,
                              proposal_id=proposal.proposal_id, generation=0)
        registry = EvaluatorRegistry(
            _DatasetStub(series), object(), dsh_runtime_provider=lambda: runtime,
            dsh_revision_provider=lambda _: {"run_state_revision": 1, "ledger_expected_revision": 1},
            dsh_identity_provider=lambda *_: {"genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64, "phenotype_instance_digest": "c" * 64},
            dsh_prediction_tool_binder=_dsh_prediction_tool_binder,
        )
        checkpoints = []
        durable_rows = {}

        def capture_checkpoint(checkpoint):
            checkpoints.append(deepcopy(checkpoint))
            return {}

        def capture_rows(rows):
            durable_rows.update({row["sample_id"]: deepcopy(row)
                                 for row in build_sample_results(candidate.candidate_id, rows)})

        bundle = registry.evaluate_scientific(
            task, candidate, proposal,
            on_sample_checkpoint=capture_checkpoint, on_sample_results=capture_rows,
        )
        summary = bundle.evaluation.metrics["sample_execution"]
        self.assertEqual(summary["prediction_cell_count"], 9)
        self.assertEqual(summary["dsh_agent_prediction_tool_invocations"], 1)
        self.assertEqual(summary["failed_examples"], 0)
        self.assertEqual(bundle.evaluation.score, 0.0)
        self.assertEqual([r["stage"] for r in runtime.requests],
                         ["sample.plan", "sample.critic", "sample.reflect"])
        self.assertEqual(len(durable_rows), 9)
        runtime.requests.clear()

        def resume_checkpoint(checkpoint):
            self.assertEqual(checkpoint, checkpoints[0])
            return {"rows": list(durable_rows.values())}

        replay = registry.evaluate_scientific(
            task, candidate, proposal, on_sample_checkpoint=resume_checkpoint,
        )
        self.assertEqual(runtime.requests, [])
        self.assertEqual(replay.evaluation.score, bundle.evaluation.score)
        self.assertEqual(replay.artifact.learned_parameters, bundle.artifact.learned_parameters)


if __name__ == "__main__":
    unittest.main()
