"""Independent evaluation opens frozen data once and never updates the search."""
from dataclasses import replace
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from ecologyrsi_dsh.application.independent_evaluation import IndependentEvaluationService
from ecologyrsi_dsh.core.exposure_registry import ScientificExposureRegistry, raw_holdout_exposure_key
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import CandidateStatus, RunStatus, digest
from ecologyrsi_dsh.data.greenhouse import CanonicalEpisode, CanonicalSeries, feature_specs
from ecologyrsi_dsh.data.registry import DatasetRegistry
from tests import test_artifact_revision_identity as identity_fixture


class IndependentJobTests(unittest.TestCase):
    def setUp(self):
        self.fixture = f = identity_fixture.ArtifactRevisionIdentityTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        f._record()
        f.reducer.candidates[f.candidate.candidate_id] = replace(f.reducer.candidates[f.candidate.candidate_id], status=CandidateStatus.PROMOTED)
        f.reducer.run = replace(f.reducer.run, status=RunStatus.COMPLETED,
                                selection_incumbent_id=f.candidate.candidate_id)
        f.reducer.task = replace(f.reducer.task, visible_datasets=("agc_cucumber_2018",),
            metadata={**f.reducer.task.metadata, "episode_id": "episode:identity",
                      "data_protocol_digest": "a" * 64, "sample_agent_mode": "dsh_native_agent"})
        self.state_patch = patch.object(f.director, "state", side_effect=f._state)
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)
        protocol = SimpleNamespace(protocol_digest="a" * 64,
                                   partition_digests={stage: digest(stage) for stage in ("validation", "final_test")})
        self.server = SimpleNamespace(director=f.director, ledger=f.ledger,
            datasets=SimpleNamespace(data_protocol=Mock(return_value=protocol)),
            dsh_tools=Mock(), sample_admission=Mock(),
            validate_frozen_runtime_bindings=Mock())
        self.service = IndependentEvaluationService(self.server)
        self.addCleanup(self.service.close)

    def test_single_job_validation_then_test_without_search_updates(self):
        f = self.fixture
        before = f.director.state(f.run_id)
        entered, release = threading.Event(), threading.Event()
        def evaluate(run_id, token, plan):
            self.assertEqual(ScientificExposureRegistry(f.ledger).formal_exposure(token.holdout_exposure_key)["state"], "opened")
            entered.set()
            self.assertTrue(release.wait(3))
            return {"outcome": "passed", "origin_count": 10, "replicas": [], "candidate_updated": False}
        with patch.object(self.service, "_evaluate", side_effect=evaluate) as call:
            with self.assertRaisesRegex(ValueError, "先通过独立验证"):
                self.service.start(f.run_id, "final_test")
            self.service.start(f.run_id, "validation")
            self.assertTrue(entered.wait(3))
            report = self.service.start(f.run_id, "validation")
            self.assertEqual(report["stages"][0]["status"], "running")
            release.set()
            self.service.workers[0].join(3)
            report = self.service.start(f.run_id, "validation")
            self.assertEqual(report["stages"][0]["outcome"], "passed")
            self.assertTrue(report["stages"][1]["available"])
            self.service.start(f.run_id, "final_test")
            self.service.workers[0].join(3)
            self.assertEqual(call.call_count, 2)
        after = f.director.state(f.run_id)
        self.assertEqual(after.run.final_test_candidate_id, f.candidate.candidate_id)
        self.assertEqual(self.server.dsh_tools.open_run_admissions.call_count, 2)
        self.assertEqual(self.server.dsh_tools.close_run_admissions.call_count, 2)
        self.assertEqual(before.candidates, after.candidates)
        self.assertEqual(before.evaluations, after.evaluations)
        self.assertEqual(before.artifacts, after.artifacts)
        self.assertTrue(all(e.kind.startswith("FormalStage") for e in after.events[len(before.events):]))

    def test_interrupted_opened_stage_is_sealed_without_reexecution(self):
        f = self.fixture
        token = f.director.reserve_formal_stage(f.run_id, stage="validation",
            candidate_id=f.candidate.candidate_id, objective_family_digest="a" * 64,
            analysis_plan_digest="b" * 64, partition_digest=digest("validation"),
            idempotency_key=f"{f.run_id}:independent:validation")
        registry = ScientificExposureRegistry(f.ledger)
        registry.open_formal_stage(token)
        self.service.recover_interrupted()
        self.service.recover_interrupted()
        self.assertEqual(registry.formal_exposure(token.holdout_exposure_key)["state"], "sealed")
        with patch.object(self.service, "_evaluate") as evaluate:
            report = self.service.start(f.run_id, "validation")
            self.assertEqual(report["stages"][0]["outcome"], "inconclusive")
            self.assertFalse(report["stages"][1]["available"])
            evaluate.assert_not_called()

    def test_running_search_cannot_open_evaluation_data(self):
        f = self.fixture
        f.reducer.run = replace(f.reducer.run, status=RunStatus.RUNNING)
        with self.assertRaisesRegex(ValueError, "先完成进化训练"):
            self.service.start(f.run_id, "validation")
        self.assertFalse(ScientificExposureRegistry(f.ledger).formal_stage_tokens())


class FormalDatasetViewTests(unittest.TestCase):
    def test_open_token_required_and_view_contains_only_reserved_partition(self):
        dataset_id = "agc_cucumber_2018"
        episode = CanonicalEpisode(dataset_id, "greenhouse_cucumber_2018", dataset_id + ":TeamA",
            tuple(range(3000)), {"air_temperature": tuple(float(t) for t in range(3000))},
            feature_specs("greenhouse_cucumber_2018"), (), "a" * 64)
        canonical = CanonicalSeries(dataset_id, episode.domain_id, (episode,))
        datasets = DatasetRegistry()
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        exposures = ScientificExposureRegistry(ledger)
        with patch.object(datasets, "_load_series", return_value=canonical):
            protocol = datasets.data_protocol(dataset_id)
            manifest = datasets._split_manifest(datasets._descriptor(dataset_id), canonical)
            for stage in ("validation", "final_test"):
                key = raw_holdout_exposure_key(dataset_digest=episode.content_sha256,
                    split_manifest_digest=manifest.split_manifest_digest_sha256, episode_id=episode.episode_id,
                    stage=stage, stage_partition_digest=protocol.partition_digests[stage])
                token = exposures.reserve_formal_stage(raw_holdout_key=key,
                    objective_family_digest="b" * 64, plan_digest="c" * 64, idempotency_key=stage,
                    run_id="run:view", stage=stage, candidate_id="candidate:view", artifact_digest="d" * 64,
                    genome_digest="e" * 64, partition_digest=protocol.partition_digests[stage])
                with self.assertRaises(PermissionError):
                    datasets.formal_view(dataset_id, token, exposures)
                with self.assertRaises(PermissionError):
                    datasets.sample(dataset_id, partition=stage)
                exposures.open_formal_stage(token)
                view = datasets.formal_view(dataset_id, token, exposures)
                self.assertEqual(view.partitions["training_fit"], protocol.calibration_fit)
                self.assertEqual(view.partitions["training_feedback"], protocol.range_for(stage))
                self.assertEqual(len(view.timestamps), protocol.range_for(stage).end)
                self.assertEqual(view.evaluation_partition, stage)
                exposures.seal_formal_stage(token, outcome="inconclusive")
                with self.assertRaises(PermissionError):
                    datasets.formal_view(dataset_id, token, exposures)
