"""Independent evaluation opens frozen data once and never updates the search."""
from dataclasses import replace
from types import MethodType, SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from ecologyrsi_dsh.application.independent_evaluation import IndependentEvaluationService
from ecologyrsi_dsh.api.handler import EvolutionHTTPServer
from ecologyrsi_dsh.core.exposure_registry import ScientificExposureRegistry, raw_holdout_exposure_key
from ecologyrsi_dsh.core.ledger import ConcurrentRunMutationError, EventLedger
from ecologyrsi_dsh.core.models import CandidateStatus, RunStatus, digest
from ecologyrsi_dsh.data.greenhouse import CanonicalEpisode, CanonicalSeries, feature_specs
from ecologyrsi_dsh.data.registry import DatasetRegistry
from ecologyrsi_dsh.evaluators.fitness import FitnessProfile
from ecologyrsi_dsh.evaluators.formal_evidence import certification_policy
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
                      "data_protocol_digest": "a" * 64, "sample_agent_mode": "dsh_native_agent",
                      "fitness_profile": FitnessProfile().to_dict(), "fitness_profile_digest": FitnessProfile().profile_digest})
        self.state_patch = patch.object(f.director, "state", side_effect=f._state)
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)
        protocol = SimpleNamespace(protocol_digest="a" * 64,
                                   partition_digests={stage: digest(stage) for stage in ("validation", "final_test")})
        self.server = SimpleNamespace(director=f.director, ledger=f.ledger,
            datasets=SimpleNamespace(data_protocol=Mock(return_value=protocol)),
            dsh_tools=Mock(), sample_admission=Mock(),
            validate_frozen_runtime_bindings=Mock())
        self.server._generation_locks_guard = threading.Lock()
        self.server._generation_locks = {}
        for name in ("reserve_run_work", "generation_lock", "try_acquire_generation_purge_lease", "release_generation_purge_lease"):
            setattr(self.server, name, MethodType(getattr(EvolutionHTTPServer, name), self.server))
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
            # The worker has no admitted sample at this barrier, but still owns
            # the run until evaluation and admission cleanup have both finished.
            self.assertIsNone(self.server.try_acquire_generation_purge_lease(f.run_id))
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
        lease = self.server.try_acquire_generation_purge_lease(f.run_id)
        self.assertIsNotNone(lease)
        self.server.release_generation_purge_lease(lease, purged=False)
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

    def test_worker_start_failure_seals_partition_and_releases_reservation(self):
        with patch("threading.Thread.start", side_effect=RuntimeError("start failed")):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                self.service.start(self.fixture.run_id, "validation")
        self.assertEqual(self.service.workers, [])
        self.assertEqual(self.service.status(self.fixture.run_id)["stages"][0]["outcome"], "inconclusive")
        lease = self.server.try_acquire_generation_purge_lease(self.fixture.run_id)
        self.assertIsNotNone(lease)
        self.server.release_generation_purge_lease(lease, purged=False)

    def _late_result_after_purge(self, recreate):
        f = self.fixture
        original = f.ledger.events_after(f.run_id, limit=1)[0]
        token = f.director.reserve_formal_stage(f.run_id, stage="validation",
            candidate_id=f.candidate.candidate_id, objective_family_digest="a" * 64,
            analysis_plan_digest="b" * 64, partition_digest=digest("validation"),
            idempotency_key="late-result")
        def evaluate():
            f.ledger.archive_run(f.run_id)
            f.ledger.purge_run(f.run_id, confirmation=f.run_id, terminal_status="completed")
            if recreate:
                f.ledger.append(f.run_id, "RunCreated", original.payload)
            return {"outcome": "passed"}
        with self.assertRaises(ConcurrentRunMutationError):
            f.director.execute_formal_stage(f.run_id, token, evaluate)
        self.assertEqual([e.kind for e in f.ledger.events_after(f.run_id)], ["RunCreated"] if recreate else [])

    def test_direct_late_result_cannot_resurrect_purged_run(self):
        self._late_result_after_purge(False)

    def test_direct_late_result_cannot_attach_to_recreated_run(self):
        self._late_result_after_purge(True)


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


class IndependentCertificationIntegrationTests(unittest.TestCase):
    def test_service_requires_real_certificate_from_each_replica(self):
        from ecologyrsi_dsh.evaluators.epoch_cohorts import PlannedOrigin
        from ecologyrsi_dsh.core.models import Proposal, TaskManifest
        from tests.test_formal_evidence import scored_evaluation

        profile = FitnessProfile()
        task = TaskManifest("independent", "frozen point certification", "greenhouse", ("agc_cucumber_2018",),
                            metadata={"fitness_profile": profile.to_dict(), "fitness_profile_digest": profile.profile_digest})
        candidate = SimpleNamespace(candidate_id="candidate:fixed", generation=0)
        artifact = SimpleNamespace(digest="a" * 64, candidate_revision_id="revision:fixed", model_id="model:fixed",
                                   learned_parameters={"agent_policy": {"experience": {"rows": []}}})
        state = SimpleNamespace(task_manifest=task, candidate=lambda _: candidate, artifact_for=lambda _: artifact)
        origins = tuple(PlannedOrigin(digest(i), task.dataset, "episode", i, (i // 6) * 24 + i % 6,
                                      (i // 6) * 24 + i % 6 + 24, "d" * 64) for i in range(84))
        proposal = Proposal("proposal:fixed", "run:fixed", 0, "frozen proposal", metadata={"genome_digest": "b" * 64})
        token = SimpleNamespace(stage="validation", candidate_id=candidate.candidate_id,
                                artifact_digest=artifact.digest, genome_digest="b" * 64)
        plan = {"inference_replicas": 2, "certification_policy": certification_policy(profile),
                "baseline_profile_digest": "b" * 64}
        good = scored_evaluation()
        # Both old evaluation.passed values are True; only computed formal
        # statistics identify the second independent inference as a zero gain.
        bad = scored_evaluation([1.] * 14)
        evaluations = [replace(e, metrics={**dict(e.metrics), "prediction_owner": "sample_agent"})
                       for e in (good, bad)]
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        server = SimpleNamespace(director=SimpleNamespace(state=lambda _: state), ledger=ledger,
            datasets=SimpleNamespace(formal_view=Mock(return_value=object())),
            evaluators=SimpleNamespace(_resolve_execution_plan=Mock(return_value=None),
                _evaluate_greenhouse_ridge=Mock(side_effect=[SimpleNamespace(evaluation=e) for e in evaluations])))
        service = IndependentEvaluationService(server)
        self.addCleanup(service.close)
        with patch("ecologyrsi_dsh.application.formal_trajectory._revision_evaluation_inputs", return_value=(None, proposal, None)), \
             patch("ecologyrsi_dsh.application.independent_evaluation._eligible_origins", return_value=(origins, ())):
            result = service._evaluate("run:fixed", token, plan)
        self.assertEqual(result["outcome"], "failed")
        self.assertFalse(result["formal_confirmation"])
        self.assertTrue(all(replica["passed"] for replica in result["replicas"]))
        self.assertEqual([replica["certification"]["outcome"] for replica in result["replicas"]], ["passed", "failed"])
        self.assertFalse(result["feedback_to_evolution"])
