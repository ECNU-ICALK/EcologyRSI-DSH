"""Independent evaluation opens frozen data once and never updates the search."""
from dataclasses import replace
from contextlib import nullcontext
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


class FormalRuntimeFixture:
    """In-memory native lifecycle; never invokes a model or reads real data."""

    def __init__(self, status="cancelled"):
        self.status = status
        self.execution_id = None
        self.prepared = []
        self.closed = []

    def prepare_formal_run(self, request):
        if self.status not in {None, "cancelled"}:
            raise ValueError("native runtime is active")
        self.prepared.append(request)
        self.execution_id = request["formal_execution"]["execution_id"]
        self.status = "running"
        return self._receipt(request, "running")

    def close_formal_run(self, request):
        if self.execution_id != request["formal_execution_id"]:
            raise ValueError("formal execution identity mismatch")
        self.closed.append(request)
        self.status = "cancelled"
        return self._receipt(request, "cancelled")

    def _receipt(self, request, status):
        return {name: request[name] for name in (
            "run_id", "run_state_revision", "stage_attempt",
            "ledger_expected_revision", "idempotency_key")} | {
                "accepted": True, "status": status,
                "formal_execution_id": self.execution_id}

    def execute_sample(self):
        if self.status != "running" or not self.execution_id:
            raise RuntimeError("native role-host is unavailable after training completion")
        return {"outcome": "passed", "origin_count": 10,
                "replicas": [], "candidate_updated": False}


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
            dsh_native_runtime=FormalRuntimeFixture(),
            validate_frozen_runtime_bindings=Mock())
        self.server.dsh_tools.bind_formal_retrieval.side_effect = lambda *_: nullcontext()
        self.server._generation_locks_guard = threading.Lock()
        self.server._generation_locks = {}
        for name in ("reserve_run_work", "generation_lock", "try_acquire_generation_purge_lease", "release_generation_purge_lease"):
            setattr(self.server, name, MethodType(getattr(EvolutionHTTPServer, name), self.server))
        self.service = IndependentEvaluationService(self.server)
        self.addCleanup(self.service.close)

    def test_completed_training_gets_fresh_formal_runtime_before_data_open(self):
        f = self.fixture
        native = self.server.dsh_native_runtime
        def validate(*_args, **_kwargs):
            self.assertEqual(native.status, "running")
            self.assertFalse(ScientificExposureRegistry(f.ledger).formal_stage_tokens())
            self.assertFalse(f.ledger.events_by_kind(f.run_id, "FormalRetrievalSnapshotFrozen"))
        self.server.validate_frozen_runtime_bindings.side_effect = validate
        with patch.object(self.service, "_evaluate", side_effect=lambda *_: native.execute_sample()):
            self.service.start(f.run_id, "validation")
            self.service.workers[0].join(3)
        self.assertEqual(self.service.status(f.run_id)["stages"][0]["outcome"], "passed")
        self.assertEqual(f.director.state(f.run_id).run.status, RunStatus.COMPLETED)
        self.assertEqual(len(native.prepared), 1)
        self.assertEqual(len(native.closed), 1)
        self.assertEqual(native.status, "cancelled")
        self.assertEqual(native.closed[0]["formal_execution_id"],
                         native.prepared[0]["formal_execution"]["execution_id"])

    def test_missing_native_registry_can_prepare_a_fresh_formal_generation(self):
        native = self.server.dsh_native_runtime = FormalRuntimeFixture(None)
        with patch.object(self.service, "_evaluate", side_effect=lambda *_: native.execute_sample()):
            self.service.start(self.fixture.run_id, "validation")
            self.service.workers[0].join(3)
        self.assertEqual(self.service.status(self.fixture.run_id)["stages"][0]["outcome"], "passed")
        self.assertEqual(len(native.prepared), 1)
        self.assertEqual(len(native.closed), 1)

    def test_native_active_or_paused_rejection_never_freezes_formal_data(self):
        f = self.fixture
        for status in ("running", "paused", "cancelling"):
            with self.subTest(status=status):
                native = self.server.dsh_native_runtime = FormalRuntimeFixture(status)
                with self.assertLogs("ecologyrsi_dsh.application.independent_evaluation", level="ERROR"):
                    with self.assertRaisesRegex(ValueError, "native runtime is active"):
                        self.service.start(f.run_id, "validation")
                self.assertEqual(native.status, status)
                self.assertFalse(native.closed)
                self.assertFalse(f.ledger.events_by_kind(f.run_id, "FormalRetrievalSnapshotFrozen"))
                self.assertFalse(ScientificExposureRegistry(f.ledger).formal_stage_tokens())

    def test_frozen_binding_failure_closes_prepared_runtime_without_exposure(self):
        f, native = self.fixture, self.server.dsh_native_runtime
        self.server.validate_frozen_runtime_bindings.side_effect = ValueError("frozen binding drift")
        with self.assertRaisesRegex(ValueError, "frozen binding drift"):
            self.service.start(f.run_id, "validation")
        self.assertEqual(native.status, "cancelled")
        self.assertEqual(len(native.closed), 1)
        self.assertFalse(f.ledger.events_by_kind(f.run_id, "FormalRetrievalSnapshotFrozen"))
        self.assertFalse(ScientificExposureRegistry(f.ledger).formal_stage_tokens())

    def test_prefreeze_retry_has_fresh_execution_id_with_identical_scientific_identity(self):
        f, native = self.fixture, self.server.dsh_native_runtime
        self.server.validate_frozen_runtime_bindings.side_effect = ValueError("temporary readiness failure")
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "temporary readiness failure"):
                self.service.start(f.run_id, "validation")
        first, second = [request["formal_execution"] for request in native.prepared]
        self.assertNotEqual(first["execution_id"], second["execution_id"])
        self.assertEqual({key: value for key, value in first.items() if key != "execution_id"},
                         {key: value for key, value in second.items() if key != "execution_id"})
        self.assertFalse(f.ledger.events_by_kind(f.run_id, "FormalRetrievalSnapshotFrozen"))
        self.assertFalse(ScientificExposureRegistry(f.ledger).formal_stage_tokens())

    def test_delayed_old_cleanup_cannot_close_new_prefreeze_retry(self):
        f, native = self.fixture, self.server.dsh_native_runtime
        self.server.validate_frozen_runtime_bindings.side_effect = ValueError("temporary readiness failure")
        with self.assertRaisesRegex(ValueError, "temporary readiness failure"):
            self.service.start(f.run_id, "validation")
        old_close = dict(native.closed[0])
        self.server.validate_frozen_runtime_bindings.side_effect = None
        entered, release = threading.Event(), threading.Event()
        def evaluate(*_args):
            entered.set()
            self.assertTrue(release.wait(3))
            return native.execute_sample()
        with patch.object(self.service, "_evaluate", side_effect=evaluate):
            self.service.start(f.run_id, "validation")
            try:
                self.assertTrue(entered.wait(3))
                with self.assertRaisesRegex(ValueError, "formal execution identity mismatch"):
                    native.close_formal_run(old_close)
                self.assertEqual(native.status, "running")
            finally:
                release.set()
                self.service.workers[0].join(3)
        self.assertEqual(self.service.status(f.run_id)["stages"][0]["outcome"], "passed")
        self.assertEqual(len(native.closed), 2)

    def test_snapshot_persistence_failure_closes_prepared_runtime_without_exposure(self):
        f, native = self.fixture, self.server.dsh_native_runtime
        original_append = f.ledger.append
        def fail_snapshot(run_id, kind, *args, **kwargs):
            if kind == "FormalRetrievalSnapshotFrozen":
                raise RuntimeError("snapshot persistence failed")
            return original_append(run_id, kind, *args, **kwargs)
        with patch.object(f.ledger, "append", side_effect=fail_snapshot):
            with self.assertRaisesRegex(RuntimeError, "snapshot persistence failed"):
                self.service.start(f.run_id, "validation")
        self.assertEqual(native.status, "cancelled")
        self.assertEqual(len(native.closed), 1)
        self.assertFalse(ScientificExposureRegistry(f.ledger).formal_stage_tokens())

    def test_formal_reservation_failure_closes_only_prepared_generation(self):
        f, native = self.fixture, self.server.dsh_native_runtime
        with patch.object(f.director, "reserve_formal_stage", side_effect=RuntimeError("reserve failed")):
            with self.assertRaisesRegex(RuntimeError, "reserve failed"):
                self.service.start(f.run_id, "validation")
        self.assertEqual(native.status, "cancelled")
        self.assertEqual(len(native.closed), 1)
        self.assertFalse(ScientificExposureRegistry(f.ledger).formal_stage_tokens())
        self.server.dsh_tools.open_run_admissions.assert_not_called()

    def test_worker_failure_closes_formal_runtime_and_never_reopens_stage(self):
        f, native = self.fixture, self.server.dsh_native_runtime
        with patch.object(self.service, "_evaluate", side_effect=RuntimeError("sample failed")):
            with self.assertLogs("ecologyrsi_dsh.application.independent_evaluation", level="ERROR"):
                self.service.start(f.run_id, "validation")
                self.service.workers[0].join(3)
        self.assertEqual(native.status, "cancelled")
        self.assertEqual(len(native.closed), 1)
        self.assertEqual(self.service.status(f.run_id)["stages"][0]["status"], "sealed")
        self.service.start(f.run_id, "validation")
        self.assertEqual(len(native.prepared), 1)

    def test_lost_prepare_response_is_closed_by_execution_identity_without_exposure(self):
        f, native = self.fixture, self.server.dsh_native_runtime
        prepare = native.prepare_formal_run
        def lost_response(request):
            prepare(request)
            raise RuntimeError("prepare response lost")
        with patch.object(native, "prepare_formal_run", side_effect=lost_response):
            with self.assertRaisesRegex(RuntimeError, "prepare response lost"):
                self.service.start(f.run_id, "validation")
        self.assertEqual(native.status, "cancelled")
        self.assertEqual(len(native.closed), 1)
        self.assertFalse(f.ledger.events_by_kind(f.run_id, "FormalRetrievalSnapshotFrozen"))
        self.assertFalse(ScientificExposureRegistry(f.ledger).formal_stage_tokens())

    def test_concurrent_fatal_formal_samples_close_the_same_generation_once(self):
        f, native = self.fixture, self.server.dsh_native_runtime
        self.assertFalse(self.service.abort_native_evaluation(f.run_id))
        handled = []
        def evaluate(*_args):
            threads = [threading.Thread(target=lambda: handled.append(
                self.service.abort_native_evaluation(f.run_id))) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(3)
                self.assertFalse(thread.is_alive())
            raise RuntimeError("sample failed")
        with patch.object(self.service, "_evaluate", side_effect=evaluate):
            with self.assertLogs("ecologyrsi_dsh.application.independent_evaluation", level="ERROR"):
                self.service.start(f.run_id, "validation")
                self.service.workers[0].join(3)
        self.assertEqual(handled, [True, True])
        self.assertEqual(len(native.closed), 1)
        self.assertFalse(self.service.abort_native_evaluation(f.run_id))

    def test_fatal_cleanup_lost_response_retries_exact_close_identity_in_worker(self):
        f, native = self.fixture, self.server.dsh_native_runtime
        original_close = native.close_formal_run
        calls = []
        def close(request):
            calls.append(dict(request))
            response = original_close(request)
            if len(calls) == 1:
                # A concurrent unrelated run advances the global ledger cursor.
                f.ledger.append("run:unrelated", "Diagnostic", {"sample": "synthetic"})
                raise RuntimeError("close response lost")
            return response
        def evaluate(*_args):
            self.assertTrue(self.service.abort_native_evaluation(f.run_id))
            raise RuntimeError("sample failed")
        with patch.object(native, "close_formal_run", side_effect=close):
            with patch.object(self.service, "_evaluate", side_effect=evaluate):
                with self.assertLogs("ecologyrsi_dsh.application.independent_evaluation", level="ERROR"):
                    self.service.start(f.run_id, "validation")
                    self.service.workers[0].join(3)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(native.status, "cancelled")

    def test_single_job_validation_then_test_without_search_updates(self):
        f = self.fixture
        before = f.director.state(f.run_id)
        entered, release = threading.Event(), threading.Event()
        def evaluate(run_id, token, plan):
            self.assertEqual(ScientificExposureRegistry(f.ledger).formal_exposure(token.holdout_exposure_key)["state"], "opened")
            self.assertFalse(plan["formal_retrieval"]["online_enabled"])
            self.assertEqual(plan["formal_retrieval"]["mode"], "frozen_snapshot")
            self.assertEqual(self.service._retrieval_snapshot(run_id)["snapshot_digest"],
                             plan["formal_retrieval"]["snapshot_digest"])
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
        self.assertTrue(all(e.kind.startswith("FormalStage") or e.kind == "FormalRetrievalSnapshotFrozen"
                            for e in after.events[len(before.events):]))
        snapshots = [e for e in after.events if e.kind == "FormalRetrievalSnapshotFrozen"]
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(self.server.dsh_tools.bind_formal_retrieval.call_count, 2)
        self.assertEqual(self.server.dsh_tools.bind_formal_retrieval.call_args_list[0],
                         self.server.dsh_tools.bind_formal_retrieval.call_args_list[1])
        snapshot_index = next(i for i, e in enumerate(after.events) if e.kind == "FormalRetrievalSnapshotFrozen")
        first_stage_index = next(i for i, e in enumerate(after.events) if e.kind == "FormalStageFrozen")
        self.assertLess(snapshot_index, first_stage_index)

    def test_snapshot_reused_even_when_packaged_catalog_changes(self):
        run_id = self.fixture.run_id
        original = self.service._retrieval_snapshot(run_id, create=True)
        with patch("ecologyrsi_dsh.application.independent_evaluation.build_formal_retrieval_snapshot",
                   side_effect=AssertionError("must use frozen contents")):
            self.assertEqual(self.service._retrieval_snapshot(run_id, create=True), original)

    def test_snapshot_failure_does_not_reserve_formal_data(self):
        with patch("ecologyrsi_dsh.application.independent_evaluation.build_formal_retrieval_snapshot",
                   side_effect=ValueError("snapshot invalid")):
            with self.assertRaisesRegex(ValueError, "snapshot invalid"):
                self.service.start(self.fixture.run_id, "validation")
        self.assertFalse(ScientificExposureRegistry(self.fixture.ledger).formal_stage_tokens())
        self.server.dsh_tools.open_run_admissions.assert_not_called()

    def test_binding_failure_never_calls_evaluator_and_seals_stage(self):
        self.server.dsh_tools.bind_formal_retrieval.side_effect = ValueError("binding invalid")
        with patch.object(self.service, "_evaluate") as evaluate:
            self.service.start(self.fixture.run_id, "validation")
            self.service.workers[0].join(3)
            evaluate.assert_not_called()
        report = self.service.status(self.fixture.run_id)
        self.assertFalse(report["stages"][1]["available"])
        self.assertEqual(report["stages"][0]["status"], "sealed")
        self.server.dsh_tools.open_run_admissions.assert_not_called()

    def test_invalid_replica_count_or_changed_plan_is_rejected(self):
        plan = {"stage": "validation", "inference_replicas": 2, "frozen": True}
        token = SimpleNamespace(stage="validation", analysis_plan_digest=digest(plan))
        self.service._validate_plan(token, plan)
        for count in (0, 1, 3, True, 2.0):
            changed = {**plan, "inference_replicas": count}
            token.analysis_plan_digest = digest(changed)
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "分析计划"):
                self.service._validate_plan(token, changed)
        token.analysis_plan_digest = digest(plan)
        with self.assertRaisesRegex(ValueError, "分析计划"):
            self.service._validate_plan(token, {**plan, "frozen": False})

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
        plan = {"stage": "validation", "inference_replicas": 2, "certification_policy": certification_policy(profile),
                "baseline_profile_digest": "b" * 64}
        token.analysis_plan_digest = digest(plan)
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
