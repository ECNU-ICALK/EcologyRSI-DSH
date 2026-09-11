from dataclasses import replace
import unittest

from ecologyrsi_dsh.core.models import TaskManifest
from ecologyrsi_dsh.core.trajectory import EvaluationPhase, EvaluationScope
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.epoch_cohorts import PlannedCohort, _eligible_origins
from ecologyrsi_dsh.evaluators.greenhouse_prediction import fitted_model_identity
from ecologyrsi_dsh.evaluators.registry import EvaluatorRegistry, EXOGENOUS_RIDGE_MODEL_ID, GREENHOUSE_MULTIHORIZON_EVALUATOR_ID
from tests.test_evaluation import (
    _cohort_series, _DatasetStub, _task, _candidate_and_proposal,
    _DshOriginRuntimeStub, _dsh_prediction_tool_binder,
)


class FrozenAgentEvaluationTests(unittest.TestCase):
    def test_later_partition_uses_same_fit_and_agent_and_rejects_changed_artifact(self):
        series = _cohort_series()
        runtime = _DshOriginRuntimeStub()
        data = _task().to_dict()
        data["metadata"].update(prediction_model_id=EXOGENOUS_RIDGE_MODEL_ID,
            evaluator_id=GREENHOUSE_MULTIHORIZON_EVALUATOR_ID, sample_agent_mode="dsh_native_agent",
            sample_agent_protocol="dsh-strict-origin-bundle@3", sample_agent_batch_size=9,
            sample_concurrency=1, prediction_cells_per_origin=9, samples_per_update=9,
            strategy_model_id="dsh/strategy", review_model_id="dsh/review")
        task = TaskManifest.from_dict(data)
        candidate, proposal = _candidate_and_proposal()
        proposal = replace(proposal, changes={"history_steps": 3, "ridge_alpha": 0.1, "residual_scale": 1.0})
        registry = EvaluatorRegistry(_DatasetStub(series), object(),
            dsh_runtime_provider=lambda: runtime,
            dsh_revision_provider=lambda _: {"run_state_revision": 1, "ledger_expected_revision": 1},
            dsh_identity_provider=lambda *_: {"genome_digest": "a" * 64, "compiled_behavior_digest": "b" * 64,
                                             "phenotype_instance_digest": "c" * 64},
            dsh_prediction_tool_binder=_dsh_prediction_tool_binder)
        trained = registry.evaluate_scientific(task, candidate, proposal)
        later = replace(series, partitions={"training_fit": series.partitions["training_fit"],
                                           "training_feedback": IndexRange(160, 220)},
                        evaluation_partition="validation")
        origins, _ = _eligible_origins(later)
        cohort = PlannedCohort("validation", origins[:2], 24)
        scope = EvaluationScope(candidate.run_id, candidate.generation, candidate.candidate_id,
            "revision:frozen", EvaluationPhase.VALIDATION, cohort.cohort_digest, cohort.origin_count)
        task = replace(task, metadata={**dict(task.metadata), "_evaluation_scope": scope.to_dict(),
                                       "_planned_evaluation_cohort": cohort.to_dict()})
        runtime.requests.clear()
        verified = registry._evaluate_greenhouse_ridge(task, candidate, proposal, later,
            horizons=(1, 6, 24), frozen_artifact=trained.artifact)
        self.assertEqual(fitted_model_identity(verified.artifact.learned_parameters["models"]),
                         fitted_model_identity(trained.artifact.learned_parameters["models"]))
        for key in ("agent_policy", "training_data_digest"):
            self.assertEqual(verified.artifact.learned_parameters[key], trained.artifact.learned_parameters[key])
        self.assertEqual(verified.evaluation.metrics["prediction_owner"], "sample_agent")
        self.assertEqual(len([r for r in runtime.requests if r["stage"] == "sample.plan"]), 2)
        self.assertFalse(any(r["stage"] == "sample.reflect" for r in runtime.requests))
        for request in runtime.requests:
            context = request["request"]["context"]
            for sample in context.get("samples", []):
                self.assertNotIn("actual", sample)
                self.assertNotIn("label", sample)
        runtime.requests.clear()
        bad = replace(trained.artifact, learned_parameters={**dict(trained.artifact.learned_parameters), "models": []})
        with self.assertRaisesRegex(ValueError, "frozen candidate fit"):
            registry._evaluate_greenhouse_ridge(task, candidate, proposal, later,
                horizons=(1, 6, 24), frozen_artifact=bad)
        self.assertFalse(runtime.requests)
