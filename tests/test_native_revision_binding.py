"""Native child identities must follow immutable revisions, including replay."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import unittest
from unittest.mock import Mock

from tests import test_artifact_revision_identity as native_fixture
from ecologyrsi_dsh.application.formal_trajectory import _revision_evaluation_inputs
from ecologyrsi_dsh.application.runtime_bindings import (
    ValidatedCandidateIdentityCache, candidate_revision_identity_binding,
)
from ecologyrsi_dsh.evaluators.pipeline import EvaluationPipeline
from ecologyrsi_dsh.evaluators.dsh_sample_adapter import _DshSampleDecisionClient
from ecologyrsi_dsh.evaluators.sample_execution import SampleExecutionContractError


class NativeRevisionBindingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = native_fixture.ArtifactRevisionIdentityTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.state = self.fixture.reducer.snapshot()
        self.candidate = self.fixture.candidate
        self.run_id = self.fixture.run_id
        self.provider = Mock(return_value=self.state)
        self.cache = ValidatedCandidateIdentityCache(self.fixture.ledger, self.provider)

    def test_real_child_compilation_does_not_overwrite_r0_and_survives_cache_restart(self):
        original = self.state.candidate_identity_binding(self.candidate.candidate_id)
        r0 = self.cache.get(self.run_id, self.candidate.candidate_id, self.fixture.r0.revision_id)
        r1 = self.cache.get(self.run_id, self.candidate.candidate_id, self.fixture.r1.revision_id)
        self.assertEqual(r0, original)
        self.assertEqual(r1["genome_digest"], self.fixture.r1.genome_digest)
        for field in ("compiled_behavior_digest", "phenotype_instance_digest", "runtime_execution_digest"):
            self.assertNotEqual(r1[field], r0[field])
        self.assertEqual(self.cache.get(self.run_id, self.candidate.candidate_id), original)
        self.assertEqual(self.cache.get(self.run_id, self.candidate.candidate_id, self.fixture.r0.revision_id), original)
        self.assertTrue(self.cache.forget(self.run_id))
        self.assertEqual(self.cache.get(self.run_id, self.candidate.candidate_id, self.fixture.r1.revision_id), r1)
        self.assertEqual(self.state.candidate_identity_binding(self.candidate.candidate_id), original)

    def test_concurrent_revision_workers_compile_once_and_return_defensive_copies(self):
        with ThreadPoolExecutor(max_workers=16) as pool:
            values = list(pool.map(lambda _: self.cache.get(
                self.run_id, self.candidate.candidate_id, self.fixture.r1.revision_id), range(64)))
        self.assertEqual(self.provider.call_count, 1)
        values[0]["genome_digest"] = "0" * 64
        self.assertEqual(values[1]["genome_digest"], self.fixture.r1.genome_digest)
        self.cache.clear()
        self.cache.get(self.run_id, self.candidate.candidate_id, self.fixture.r1.revision_id)
        self.assertEqual(self.provider.call_count, 2)

    def test_foreign_revision_is_rejected(self):
        wrong = replace(self.fixture.r1, candidate_id="candidate:other",
                        genome=self.fixture.r1_genome.to_dict())
        state = replace(self.state, candidate_revisions=(self.fixture.r0, wrong))
        with self.assertRaisesRegex(ValueError, "another candidate"):
            candidate_revision_identity_binding(state, self.candidate.candidate_id, wrong.revision_id)

    def test_scoped_evaluation_binds_child_and_rejects_old_candidate_provider(self):
        task = replace(self.state.task_manifest, metadata={**self.state.task_manifest.metadata,
            "_evaluation_scope": self.fixture.scope.to_dict()})
        _, proposal, _ = _revision_evaluation_inputs(
            self.state, self.candidate, self.fixture.r1.revision_id, task)
        pipeline = EvaluationPipeline(None, dsh_identity_provider=self.cache.get)
        context = pipeline._dsh_sample_stage_context(task, self.candidate, proposal)
        identity = context["candidate_execution_identity"]
        self.assertEqual(identity["genome_digest"], self.fixture.r1.genome_digest)
        self.assertEqual(context["candidate_genome_digest"], identity["genome_digest"])
        self.assertEqual(identity, pipeline._dsh_execution_identity(
            task, self.run_id, self.candidate.candidate_id, proposal))
        pipeline.dsh_identity_provider = lambda *_: self.state.candidate_identity_binding(self.candidate.candidate_id)
        with self.assertRaisesRegex(ValueError, "does not match evaluated genome"):
            pipeline._dsh_sample_stage_context(task, self.candidate, proposal)

    def test_misbound_wave_is_rejected_before_runtime_or_provider_call(self):
        runtime, revisions = Mock(), Mock()
        client = _DshSampleDecisionClient(run_id=self.run_id, runtime_provider=runtime,
            revision_provider=revisions, identity_digests={"genome_digest": "a" * 64})
        for context in ({"candidate_genome_digest": "b" * 64},
                        {"candidate_execution_identity": {"genome_digest": "b" * 64}}):
            with self.assertRaisesRegex(SampleExecutionContractError, "identity mismatch"):
                client.sample_decide("model", role="planner", samples=[], context=context,
                                     available_tools=[], max_tokens=16384)
        runtime.assert_not_called()
        revisions.assert_not_called()

    def test_final_native_planner_and_critic_waves_cannot_replay_other_revision(self):
        from tests.test_dsh_sample_execution import (
            _SampleRuntime, _request, _constant_forecast_bundle, _fake_agent_prediction_binder,
        )
        from ecologyrsi_dsh.evaluators.dsh_sample_adapter import DshSampleCollaborationAdapter
        request_sets = []
        for revision in (self.fixture.r0, self.fixture.r1):
            binding = self.cache.get(self.run_id, self.candidate.candidate_id, revision.revision_id)
            identity = {key: binding[key] for key in (
                "genome_digest", "compiled_behavior_digest", "phenotype_instance_digest")}
            runtime = _SampleRuntime()
            adapter = DshSampleCollaborationAdapter(run_id=self.run_id,
                runtime_provider=lambda: runtime,
                revision_provider=lambda _: {"run_state_revision": 7, "ledger_expected_revision": 11},
                identity_digests=identity, strategy_model_id="dsh/strategy", review_model_id="dsh/review",
                forecast_bundle_tool=_constant_forecast_bundle(21.5),
                prediction_tool_binder=_fake_agent_prediction_binder)
            # Deliberately identical visible content: the final transport must
            # still separate identities, even if compaction drops context keys.
            plan = adapter.plan_batch({"run_id": self.run_id, "candidate_id": self.candidate.candidate_id})
            outcome = adapter.predict_samples((_request("sample-1"),), (plan,), attempts=(1,))[0]
            self.assertIsNone(outcome.error)
            self.assertEqual([r["stage"] for r in runtime.requests], ["sample.plan", "sample.critic"])
            for request in runtime.requests:
                wire = request["request"]
                self.assertEqual(wire["context"]["context"]["candidate_execution_identity"], identity)
                self.assertEqual(wire["identity_digests"], identity)
            request_sets.append(runtime.requests)
        for old, new in zip(*request_sets):
            self.assertNotEqual(old["idempotency_key"], new["idempotency_key"])
