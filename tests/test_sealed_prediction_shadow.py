"""Sealed native replay must reproduce original models, inputs and receipts."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

from scripts.shadow_replay_sealed_predictions import (
    causal_context, native_usage, replay_vector, validate_artifact, validate_revision_binding, validate_scored_cohort,
)
from ecologyrsi_dsh.core.artifact_identity import build_artifact_revision_binding
from ecologyrsi_dsh.core.models import ModelArtifact, digest
from ecologyrsi_dsh.core.trajectory import CandidateRevision, EvaluationScope
from ecologyrsi_dsh.evaluators import greenhouse_prediction as ridge
from ecologyrsi_dsh.integrations.dsh_tools import DshPredictionToolBinding
from tests.test_baseline_aligned_ridge import periodic_series
from tests.test_evolution_genome import _initialization, _mutation_context
from ecologyrsi_dsh.evolution.genome import apply_genome_mutation, materialize_seed_genome
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry


class CausalColumn:
    def __init__(self, values, latest):
        self.values, self.latest = values, latest

    def __getitem__(self, index):
        if not isinstance(index, int) or index > self.latest:
            raise AssertionError("replay attempted to read a future target or exogenous value")
        return self.values[index]


class SealedPredictionShadowTests(unittest.TestCase):
    def fixture(self, aligned=False, alpha=.1):
        series = periodic_series()
        config = ridge.BaselineAlignedRidgeConfig(6, alpha, .25) if aligned else ridge.ExogenousRidgeConfig(6, alpha, .5)
        result = ridge.fit_predict_exogenous_ridge(
            series, targets=("air_temperature", "relative_humidity", "co2_concentration"),
            horizons=(1, 6, 24), config=config, evaluation_history_steps=12,
            defer_prediction_partitions=("training_feedback",),
        )
        models = result["models"]
        by_cell = {(m["target"], m["horizon_hours"]): m for m in models}
        rows = [row for row in result["prediction_rows"] if row["partition"] == "training_feedback"
                and row["origin_timestamp"] == 205]
        self.assertEqual(len(rows), 9)
        for index, row in enumerate(rows):
            row["sample_id"] = f"sample:{index}"
            row["model_reference_baseline"] = row["baseline"]
            row["predicted"] = ridge.predict_fitted_exogenous_ridge(
                target=row["target"], horizon_hours=row["horizon_hours"], baseline=row["baseline"],
                label_free_context=row["label_free_context"], models=models, config=config)
        original = ModelArtifact(
            artifact_id="artifact:shadow", run_id="run:shadow", candidate_id="candidate:shadow",
            candidate_revision_id="revision:shadow", evaluation_scope_digest="a" * 64,
            model_id=result["model_id"], dataset_digest=series.digest,
            training_partition="training_fit", training_rows=150,
            parameters=result["parameters"], learned_parameters={"models": models},
        )
        artifact = validate_artifact(original.to_dict())
        raw = {row["sample_id"]: {"predicted": row["predicted"], "metadata": {
            "source_model_id": result["model_id"],
            "prediction_unit": "forecast_origin_with_target_horizon_vector", "origin_timestamp": 205}}
               for row in rows}
        binding = DshPredictionToolBinding(
            run_id="run:shadow", stage_attempt=1, idempotency_key="sealed-origin", wave_digest="b"*64,
            catalog=[{"tool_id": "candidate-model", "version": "1"}], sample_ids=tuple(raw), executor=lambda *_: raw)
        events = []
        def persist(payload):
            events.append({"seq": 2, "event_id": "tool:sealed", "payload": payload})
            return SimpleNamespace(event_id="tool:sealed")
        binding.execute({"tool_id": "candidate-model", "call_id": "one", "parameters": {}, "wave_digest": "b"*64},
                        session_id="agent", persist=persist)
        event = events[0]
        return series, config, by_cell, rows, artifact, event

    def test_original_fitted_predictions_and_receipts_reproduce_without_any_future_value(self):
        for aligned in (False, True):
            with self.subTest(aligned=aligned):
                series, config, models, rows, artifact, event = self.fixture(aligned)
                cutoff = series.timestamps.index(205)
                source = SimpleNamespace(
                    timestamps=series.timestamps, partitions=series.partitions, features=series.features,
                    values={name: CausalColumn(values, cutoff) for name, values in series.values.items()},
                )
                prepared = {row["sample_id"]: causal_context(source, row, models[row["target"], row["horizon_hours"]], config)
                            for row in rows}
                for row in rows:
                    self.assertEqual(prepared[row["sample_id"]][1]["feature_snapshot"], row["label_free_context"]["feature_snapshot"])
                difference, seconds = replay_vector(rows=rows, prepared=prepared, artifact=artifact, event=event)
                self.assertEqual(difference, 0.)
                self.assertGreaterEqual(seconds, 0.)

    def test_different_fitted_model_is_not_accepted_even_if_artifact_digest_is_recomputed(self):
        _, _, _, _, artifact, _ = self.fixture()
        learned = artifact.to_dict()["learned_parameters"]
        learned["models"][0]["intercept"] += .01
        altered = replace(artifact, learned_parameters=learned)
        with self.assertRaisesRegex(ValueError, "fitted model digest"):
            validate_artifact(altered.to_dict())

    def test_wrong_tool_receipt_is_rejected_but_agent_can_change_final_prediction(self):
        source, config, models, rows, artifact, event = self.fixture()
        prepared = {row["sample_id"]: causal_context(source, row, models[row["target"], row["horizon_hours"]], config)
                    for row in rows}
        wrong_receipt = deepcopy(event)
        wrong_receipt["payload"]["output_digest"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch|no longer reproduces"):
            replay_vector(rows=rows, prepared=prepared, artifact=artifact, event=wrong_receipt)
        changed_rows = deepcopy(rows)
        changed_rows[0]["predicted"] += .01
        difference, _ = replay_vector(rows=changed_rows, prepared=prepared, artifact=artifact, event=event)
        self.assertEqual(difference, 0.0)  # Agent final values are independent of numerical tool evidence.

    def test_no_native_usage_is_unavailable_instead_of_zero(self):
        report = native_usage([], {"unknown-stage"})
        self.assertIsNone(report["reported_tokens"])
        self.assertFalse(report["usage_complete"])
        self.assertIsNone(report["launch_to_last_acceptance_window_seconds"])

    def test_wrong_revision_or_stale_outer_artifact_genome_binding_is_rejected(self):
        _, _, _, _, artifact, _ = self.fixture()
        registry = current_program_registry()
        genome = materialize_seed_genome(registry.seed_template("greenhouse-exogenous-default@1"), _initialization())
        revision = CandidateRevision(revision_id="revision:shadow", run_id=artifact.run_id, generation=0,
            candidate_id=artifact.candidate_id, genome=genome.to_dict(), genome_digest=genome.genome_digest,
            behavior_digest=genome.behavior_digest, mutation_digest="c" * 64)
        scope = EvaluationScope(run_id=artifact.run_id, generation=0, candidate_id=artifact.candidate_id,
            candidate_revision_id=revision.revision_id, phase="holdout", cohort_digest="e" * 64,
            origin_count=1, holdout_arm="finalist_1")
        artifact = replace(artifact, evaluation_scope_digest=scope.scope_key)
        original = {"genome_digest": genome.genome_digest}
        validate_revision_binding(artifact, revision.to_dict(), scope, run_id=artifact.run_id,
                                  event_payload={"identity_binding": original}, original_proposal_binding=original)
        for wrong_scope, wrong_binding in (
            (replace(scope, candidate_revision_id="revision:wrong"), {"genome_digest": genome.genome_digest}),
            (scope, {"genome_digest": "f" * 64}),
        ):
            with self.assertRaisesRegex(ValueError, "do not identify the same candidate"):
                validate_revision_binding(artifact, revision.to_dict(), wrong_scope, run_id=artifact.run_id,
                                          event_payload={"identity_binding": wrong_binding}, original_proposal_binding=original)

    def test_v2_effective_revision_is_verified_separately_from_original_proposal(self):
        _, _, _, _, artifact, _ = self.fixture(alpha=.2)
        registry = current_program_registry()
        parent = materialize_seed_genome(registry.seed_template("greenhouse-exogenous-default@1"), _initialization())
        child = apply_genome_mutation(parent, {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [{"op": "set_bounded_parameter", "name": "ridge_alpha", "value": .2}],
        }, _mutation_context(parent), registry)
        revision = CandidateRevision(revision_id="revision:shadow:r1", run_id=artifact.run_id, generation=0,
            candidate_id=artifact.candidate_id, genome=child.to_dict(), genome_digest=child.genome_digest,
            behavior_digest=child.behavior_digest, mutation_digest="c" * 64,
            parent_revision_id="revision:shadow:r0", source_batch_index=0)
        scope = EvaluationScope(run_id=artifact.run_id, generation=0, candidate_id=artifact.candidate_id,
            candidate_revision_id=revision.revision_id, phase="holdout", cohort_digest="e" * 64,
            origin_count=1, holdout_arm="finalist_1")
        artifact = replace(artifact, candidate_revision_id=revision.revision_id, evaluation_scope_digest=scope.scope_key)
        proposal = {"genome_digest": parent.genome_digest, "runtime_execution_digest": "a" * 64}
        envelope = {"schema_version": "ecologyrsi-dsh.artifact-recorded/2", "artifact": artifact.to_dict(),
                    "proposal_identity_binding": proposal,
                    "artifact_revision_binding": build_artifact_revision_binding(artifact, revision, scope)}
        self.assertNotEqual(parent.genome_digest, revision.genome_digest)
        validate_revision_binding(artifact, revision.to_dict(), scope, run_id=artifact.run_id,
                                  event_payload=envelope, original_proposal_binding=proposal)
        altered = deepcopy(envelope)
        altered["artifact_revision_binding"]["genome_digest"] = parent.genome_digest
        wrong_proposal = deepcopy(envelope)
        wrong_proposal["proposal_identity_binding"]["runtime_execution_digest"] = "f" * 64
        for payload in (altered, wrong_proposal, {"identity_binding": proposal},
                        {**envelope, "schema_version": "ecologyrsi-dsh.artifact-recorded/999"}):
            with self.subTest(schema=payload.get("schema_version")):
                with self.assertRaises(ValueError):
                    validate_revision_binding(artifact, revision.to_dict(), scope, run_id=artifact.run_id,
                                              event_payload=payload, original_proposal_binding=proposal)

    def test_scored_cohort_requires_the_same_episode_source_and_candidate(self):
        scope = EvaluationScope(run_id="run:shadow", generation=0, candidate_id="candidate:shadow",
            candidate_revision_id="revision:shadow", phase="holdout", cohort_digest="e" * 64,
            origin_count=1, holdout_arm="finalist_1")
        source = SimpleNamespace(dataset_id="dataset", episode_id="episode", timestamps=(205,))
        origin = SimpleNamespace(dataset_id="dataset", episode_id="episode", origin_index=0, origin_timestamp=205)
        cohort = SimpleNamespace(cohort_digest=scope.cohort_digest, origins=[origin])
        rows = [{"candidate_id": scope.candidate_id, "origin_timestamp": 205, "target": target, "horizon_hours": horizon}
                for target in ("air_temperature", "relative_humidity", "co2_concentration") for horizon in (1, 6, 24)]
        validate_scored_cohort(source, rows, cohort, scope)
        for bad_source in (SimpleNamespace(dataset_id="dataset", episode_id="other", timestamps=(205,)),
                           SimpleNamespace(dataset_id="dataset", episode_id="episode", timestamps=(206,))):
            with self.assertRaisesRegex(ValueError, "registered source episode"):
                validate_scored_cohort(bad_source, rows, cohort, scope)
        wrong_rows = deepcopy(rows)
        wrong_rows[0]["candidate_id"] = "candidate:other"
        with self.assertRaisesRegex(ValueError, "candidate or frozen cohort"):
            validate_scored_cohort(source, wrong_rows, cohort, scope)

    def test_usage_missing_one_launched_child_is_incomplete_and_snapshots_are_not_summed(self):
        time = "2026-09-07T00:00:00+00:00"
        events = [{"kind": "DshChildLaunchReserved", "created_at": time,
                   "payload": {"launch": {"idempotency_key": "origin", "reservation_id": reservation}}}
                  for reservation in ("child:1", "child:2")]
        for total in (10, 20):
            events.append({"kind": "DshSessionUsageRecorded", "created_at": time,
                           "payload": {"identity": {"session_id": "session:1", "child_reservation_id": "child:1",
                                                     "idempotency_key": "origin"},
                                       "session_metrics": {"provider_usage": {"available": True,
                                                                              "totals": {"total_tokens": total}}},
                                       "usage_complete": True}})
        report = native_usage(events, {"origin"})
        self.assertFalse(report["usage_complete"])
        self.assertEqual(report["launches_without_usage"], 1)
        self.assertEqual(report["reported_tokens"]["total_tokens"], 20)
