"""Scientific contracts exercised through the current native, frozen-cohort path."""
from __future__ import annotations
import math
import unittest
from unittest.mock import patch
from dataclasses import replace
from ecologyrsi_dsh.core.models import Candidate, Proposal, TaskManifest, digest
from ecologyrsi_dsh.core.prediction_policy import RUNTIME_EVALUATOR_ID, CELLWISE_EVALUATOR_ID
from ecologyrsi_dsh.core.trajectory import EvaluationScope, EvaluationPhase
from ecologyrsi_dsh.data.contracts import DatasetSeries, SelectionDatasetView
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.data.greenhouse import feature_specs
from ecologyrsi_dsh.evaluators.registry import EvaluatorRegistry
from ecologyrsi_dsh.evaluators.greenhouse_prediction import EXOGENOUS_RIDGE_MODEL_ID
from ecologyrsi_dsh.evaluators.epoch_cohorts import PlannedCohort, _eligible_origins
from ecologyrsi_dsh.evaluators.pipeline import _aggregate_greenhouse_objective, _complete_scoring_rows
from ecologyrsi_dsh.evaluators.sample_execution import _sample_id
from tests.test_dsh_sample_execution import _SampleRuntime as _DshOriginRuntimeStub
from tests.agent_prediction_fixtures import agent_binding as _dsh_prediction_tool_binder
DATASET_ID = "agc_cucumber_2018"
SPLIT_DIGEST = "f" * 64

def _selection_view(series):
    if isinstance(series, SelectionDatasetView):
        return series
    end=series.partitions["training_feedback"].end
    return SelectionDatasetView(schema="ecologyrsi-dsh.selection-dataset-view/1",
        dataset_id=series.dataset_id,domain_id=series.domain_id,episode_id=series.episode_id,
        dataset_digest=series.digest,split_manifest_digest_sha256=series.split_manifest_digest_sha256,
        data_protocol_digest="f"*64,timestamps=series.timestamps[:end],
        values={k:v[:end] for k,v in series.values.items()},
        partitions={"calibration_fit":series.partitions["training_fit"],
                    "model_selection":series.partitions["training_feedback"],
                    "training_fit":series.partitions["training_fit"],
                    "training_feedback":series.partitions["training_feedback"]},
        features=series.features,selection_view_digest=digest({"source":series.digest,"end":end}))

class _DatasetStub:
    def __init__(self, series):self._series=_selection_view(series)
    def selection_view(self,dataset_id,episode_id=None,*,expected_dataset_digest=None,
                       expected_split_manifest_digest=None,expected_data_protocol_digest=None,**kwargs):
        series=self._series
        if dataset_id != series.dataset_id or episode_id not in (None,series.episode_id):raise KeyError(dataset_id)
        if expected_dataset_digest not in (None,series.digest):raise ValueError("数据集快照发生漂移")
        if expected_split_manifest_digest not in (None,series.split_manifest_digest_sha256):raise ValueError("时间分区快照发生漂移")
        if expected_data_protocol_digest not in (None,series.data_protocol_digest):raise ValueError("数据协议发生漂移")
        return series

def _raw_series(
    *, tail_marker: float = 0.0, feedback_missing: bool = False
) -> DatasetSeries:
    timestamps = tuple(range(100))
    temperature: list[float | None] = [
        21.0 + 0.03 * index + 0.2 * math.sin(index / 4) for index in timestamps
    ]
    humidity: list[float | None] = [
        70.0 + 2.0 * math.sin(index / 5) for index in timestamps
    ]
    co2: list[float | None] = [
        720.0 + 15.0 * math.sin(index / 3) for index in timestamps
    ]
    temperature[5] = None
    temperature[35] = None
    temperature[40] = float("nan")
    if feedback_missing:
        for index in range(31, 60):
            temperature[index] = None
    # The fit/feedback evaluator must not read the embargo row or any later
    # development/gate-tail values.  Variants deliberately disagree there.
    temperature[30] = tail_marker
    for values in (temperature, humidity, co2):
        for index in range(60, 100):
            values[index] = tail_marker
    return DatasetSeries(
        schema="ecologyrsi-dsh.dataset-series/1",
        dataset_id=DATASET_ID,
        domain_id="greenhouse_cucumber_2018",
        episode_id=f"{DATASET_ID}:TeamA",
        digest="d" * 64,
        timestamps=timestamps,
        values={
            "air_temperature": tuple(temperature),
            "relative_humidity": tuple(humidity),
            "co2_concentration": tuple(co2),
        },
        partitions={
            "training_fit": IndexRange(0, 30),
            "training_feedback": IndexRange(31, 60),
            "development": IndexRange(80, 80),
        },
        features=feature_specs("greenhouse_cucumber_2018"),
        split_manifest_digest_sha256=SPLIT_DIGEST,
    )

def _raw_cohort_series() -> DatasetSeries:
    timestamps = tuple(range(220))
    temperature: list[float | None] = [
        22.0 + 0.015 * index + 0.3 * math.sin(index / 7) for index in timestamps
    ]
    humidity: list[float | None] = [
        68.0 + 1.5 * math.sin(index / 9) for index in timestamps
    ]
    co2: list[float | None] = [
        710.0 + 12.0 * math.sin(index / 5) for index in timestamps
    ]
    temperature[140] = None
    return DatasetSeries(
        schema="ecologyrsi-dsh.dataset-series/1",
        dataset_id=DATASET_ID,
        domain_id="greenhouse_cucumber_2018",
        episode_id=f"{DATASET_ID}:TeamA",
        digest="d" * 64,
        timestamps=timestamps,
        values={
            "air_temperature": tuple(temperature),
            "relative_humidity": tuple(humidity),
            "co2_concentration": tuple(co2),
        },
        partitions={
            "training_fit": IndexRange(0, 100),
            # Index 100 is an embargo row and must not enter model history.
            "training_feedback": IndexRange(101, 200),
            "development": IndexRange(200, 220),
        },
        features=feature_specs("greenhouse_cucumber_2018"),
        split_manifest_digest_sha256=SPLIT_DIGEST,
    )

def _nested_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _nested_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_values(item)
    else:
        yield value

def _series(**kwargs):return _selection_view(_raw_series(**kwargs))
def _cohort_series():return _selection_view(_raw_cohort_series())

def _task():
    from tests.native_fixtures import native_manifest
    return native_manifest(task_id="greenhouse-evaluation-test",metadata={
        "prediction_model_id":EXOGENOUS_RIDGE_MODEL_ID,"episode_id":f"{DATASET_ID}:TeamA",
        "dataset_digest":"d"*64,"split_manifest_digest":SPLIT_DIGEST,
        "data_protocol_digest":"f"*64,"sample_reflection_policy":"candidate_aggregate_post_score@1",
        "sample_remote_critic_policy":{"version":"uncertain_or_failure@1","min_planner_confidence":.5}})

def _profile_metadata():
    from tests.test_evolution_genome import _seed_genome
    from ecologyrsi_dsh.evolution.workflow_ir import resolve_candidate_agent_profile
    from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
    return {"genome_digest":"a"*64,"candidate_agent_profile":resolve_candidate_agent_profile(_seed_genome(),current_program_registry())}

def _candidate_and_proposal():
    proposal=Proposal(proposal_id="proposal:evaluation:1",run_id="run:evaluation",generation=0,
        title="Native scientific fixture",changes={"history_steps":3,"ridge_alpha":.1,"residual_scale":1.}, metadata=_profile_metadata())
    return Candidate(candidate_id="candidate:evaluation:1",run_id=proposal.run_id,
        proposal_id=proposal.proposal_id,generation=0),proposal

def _native_registry(series,runtime=None):
    runtime=runtime or _DshOriginRuntimeStub()
    return EvaluatorRegistry(_DatasetStub(series),dsh_runtime_provider=lambda:runtime,
        dsh_revision_provider=lambda _: {"run_state_revision":1,"ledger_expected_revision":1},
        dsh_identity_provider=lambda *_:{"genome_digest":"a"*64,"compiled_behavior_digest":"b"*64,
                                         "phenotype_instance_digest":"c"*64},
        dsh_prediction_tool_binder=_dsh_prediction_tool_binder)

def _scope_cohort(series,candidate,count=1):
    origins,_=_eligible_origins(_selection_view(series),history_steps=12)
    cohort=PlannedCohort("screening",origins[:count],24,shared_candidate_count=5)
    scope=EvaluationScope(candidate.run_id,candidate.generation,candidate.candidate_id,
        "revision:fixture",EvaluationPhase.SCREENING,cohort.cohort_digest,cohort.origin_count)
    return {"scope":scope,"cohort":cohort}


class GreenhouseEvaluationTests(unittest.TestCase):
    def test_checkpoint_completion_preserves_sparse_frozen_sample_indices(self):
        expected = [
            {
                "sample_index": 19,
                "target": "air_temperature",
                "horizon_hours": 1,
                "origin_timestamp": 100,
                "timestamp": 101,
                "baseline": 20.0,
            },
            {
                "sample_index": 20,
                "target": "air_temperature",
                "horizon_hours": 1,
                "origin_timestamp": 101,
                "timestamp": 102,
                "baseline": 20.0,
            },
        ]
        context = {"candidate_id": "candidate:sparse", "dataset_digest": "d" * 64}
        returned = dict(expected[1])
        returned["sample_id"] = _sample_id(
            returned,
            context,
            target="air_temperature",
            horizon=1,
        )
        returned["predicted"] = 20.5
        returned["sample_execution_status"] = "succeeded"
        completed = _complete_scoring_rows(
            expected,
            [returned],
            context=context,
            target_bounds={
                "air_temperature": {
                    "minimum": -40.0,
                    "maximum": 80.0,
                }
            },
        )
        self.assertEqual([row["sample_index"] for row in completed], [19, 20])
        self.assertEqual(
            completed[1]["sample_id"],
            returned["sample_id"],
        )
        self.assertEqual(completed[0]["sample_execution_failure"]["class"], "missing_sample_result")

    def test_weighted_objective_includes_non_equal_task_counts_and_missing_cells(self):
        rows = [
            {
                "target": "air_temperature",
                "horizon_hours": 1,
                "n": 10,
                "skill_score": 0.2,
                "normalized_mean_reward": 0.1,
            },
            {
                "target": "air_temperature",
                "horizon_hours": 6,
                "n": 10,
                "skill_score": 0.4,
                "normalized_mean_reward": 0.3,
            },
            {
                "target": "relative_humidity",
                "horizon_hours": 1,
                "n": 1,
                "skill_score": 0.6,
                "normalized_mean_reward": 0.5,
            },
            {
                "target": "relative_humidity",
                "horizon_hours": 6,
                "n": 1,
                "skill_score": 0.8,
                "normalized_mean_reward": 0.7,
            },
            {
                "target": "co2_concentration",
                "horizon_hours": 1,
                "n": 1,
                "skill_score": 0.0,
                "normalized_mean_reward": 0.0,
            },
        ]
        aggregate = _aggregate_greenhouse_objective(rows, (1, 6))
        self.assertAlmostEqual(aggregate["weighted_skill_score"], 1.0 / 6.0)
        self.assertAlmostEqual(
            aggregate["weighted_normalized_mean_reward"], 0.6 / 6.0
        )
        self.assertEqual(aggregate["objective_task_count"], 6)
        self.assertEqual(aggregate["objective_missing_task_count"], 1)
        self.assertAlmostEqual(aggregate["objective_weight_coverage"], 5.0 / 6.0)

    def test_weighted_objective_honors_target_weights(self):
        rows = [
            {
                "target": "air_temperature",
                "horizon_hours": 1,
                "n": 1,
                "skill_score": 1.0,
                "normalized_mean_reward": 1.0,
            },
            {
                "target": "relative_humidity",
                "horizon_hours": 1,
                "n": 1,
                "skill_score": 0.0,
                "normalized_mean_reward": 0.0,
            },
            {
                "target": "co2_concentration",
                "horizon_hours": 1,
                "n": 1,
                "skill_score": 0.0,
                "normalized_mean_reward": 0.0,
            },
        ]
        with patch.dict(
            "ecologyrsi_dsh.evaluators.catalog.GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS",
            {
                "air_temperature": 0.6,
                "relative_humidity": 0.3,
                "co2_concentration": 0.1,
            },
            clear=True,
        ):
            aggregate = _aggregate_greenhouse_objective(rows, (1,))
        self.assertAlmostEqual(aggregate["weighted_skill_score"], 0.6)
        self.assertAlmostEqual(aggregate["weighted_normalized_mean_reward"], 0.6)

    def test_weighted_objective_rejects_invalid_weights_and_horizons(self):
        with patch.dict(
            "ecologyrsi_dsh.evaluators.catalog.GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS",
            {"air_temperature": -1.0},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                _aggregate_greenhouse_objective([], (1,))
        with self.assertRaises(ValueError):
            _aggregate_greenhouse_objective([], (1, 1))

    def test_split_manifest_digest_uses_explicit_sha256_field(self) -> None:
        series=_cohort_series();candidate,proposal=_candidate_and_proposal()
        metrics = _native_registry(series).evaluate_scientific(_task(),candidate,proposal,**_scope_cohort(series,candidate)).evaluation.metrics
        self.assertEqual(metrics["split_manifest_digest_sha256"], SPLIT_DIGEST)
        self.assertNotIn("split_manifest_digest", metrics)

    def test_native_ridge_executes_one_tool_vector_per_frozen_origin(self):
        series=_cohort_series();runtime=_DshOriginRuntimeStub();registry=_native_registry(series,runtime)
        candidate,proposal=_candidate_and_proposal()
        bundle=registry.evaluate_scientific(_task(),candidate,proposal,**_scope_cohort(series,candidate,2))
        self.assertEqual([r["stage"] for r in runtime.requests],["sample.plan"]*2)
        self.assertEqual(len(bundle.sample_results),18)
        summary=bundle.evaluation.metrics["sample_execution"]
        self.assertEqual(summary["attempted_origin_samples"],2)
        self.assertEqual(summary["dsh_agent_prediction_tool_invocations"],2)
        self.assertTrue(summary["strict_agent_chain_pass"])
        self.assertTrue(all(len(r["request"]["context"]["samples"])==9 for r in runtime.requests))

    def test_current_catalog_exact_bindings_and_scientific_gates(self):
        registry=_native_registry(_cohort_series())
        catalog={x["id"]:x for x in registry.catalog()}
        self.assertEqual(set(catalog),{RUNTIME_EVALUATOR_ID,CELLWISE_EVALUATOR_ID})
        self.assertEqual(len(registry.predictor_catalog()),6)
        for evaluator in catalog.values():
            self.assertEqual(evaluator["prediction_task_count"],9)
            self.assertEqual(evaluator["fitness_profile_digest"],digest(evaluator["fitness_profile"]))
            for predictor in registry.predictor_catalog():
                registry.validate_binding(DATASET_ID,evaluator["id"],predictor["id"])
        for old in ("toy_time_forward@1","greenhouse_multihorizon_time_forward@2"):
            with self.assertRaises(ValueError):registry.objective_profile(old)
        profile=registry.objective_profile(RUNTIME_EVALUATOR_ID)
        self.assertEqual(profile["confidence_method"],"paired_moving_block_bootstrap@1")
        self.assertEqual(profile["bootstrap_resamples"],1000)
        self.assertEqual(profile["minimum_paired_blocks"],8)

    def test_snapshot_and_scope_mismatch_fail_before_native_calls(self):
        series=_cohort_series();runtime=_DshOriginRuntimeStub();registry=_native_registry(series,runtime)
        candidate,proposal=_candidate_and_proposal();task=_task();kwargs=_scope_cohort(series,candidate)
        with self.assertRaisesRegex(ValueError,"scope and frozen cohort"):
            registry.evaluate_scientific(task,candidate,proposal)
        with self.assertRaises(ValueError):
            registry.evaluate_scientific(replace(task,metadata={**task.metadata,"dataset_digest":"x"*64}),candidate,proposal,**kwargs)
        with self.assertRaises(ValueError):
            registry.evaluate_scientific(task,candidate,proposal,scope=replace(kwargs["scope"],candidate_id="other"),cohort=kwargs["cohort"])
        self.assertEqual(runtime.requests,[])

    def test_restricted_tail_is_unavailable_and_cannot_change_fit_or_score(self):
        base=_raw_cohort_series();changed=replace(base,values={k:tuple(v if i<200 else 1e9 for i,v in enumerate(values)) for k,values in base.values.items()})
        candidate,proposal=_candidate_and_proposal();bundles=[]
        for raw in (base,changed):
            series=_selection_view(raw);self.assertEqual(len(series.timestamps),200)
            self.assertNotIn("development",series.partitions)
            bundles.append(_native_registry(series).evaluate_scientific(_task(),candidate,proposal,**_scope_cohort(series,candidate)))
        self.assertEqual(bundles[0].artifact.learned_parameters,bundles[1].artifact.learned_parameters)
        self.assertEqual(bundles[0].evaluation.score,bundles[1].evaluation.score)
