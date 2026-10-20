"""Current native write/replay boundaries; scientific selection uses joint receipts."""
from dataclasses import replace
import unittest

from ecologyrsi_dsh.core.models import Promotion, PromotionDecision
from ecologyrsi_dsh.core.state import project_run_state
from tests import test_genome_replay as fixture


class DirectorInvariantTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.GenomeReplayTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.director, self.ledger = self.fixture.director, self.fixture.ledger
        self.run_id = self.fixture._start_new("run:invariants")

    def test_spawn_requires_submitted_proposal(self):
        proposal = self.fixture._proposal(self.run_id)
        before = self.ledger.count()
        with self.assertRaisesRegex(ValueError, "submitted"):
            self.director.spawn_candidate(self.run_id, proposal)
        self.assertEqual(self.ledger.count(), before)

    def test_proposal_and_spawn_retries_preserve_exact_identity(self):
        proposal = self.fixture._proposal(self.run_id)
        self.director.submit_proposal(proposal)
        self.director.submit_proposal(proposal)
        candidate = self.director.spawn_candidate(self.run_id, proposal, candidate_id="candidate:a", slot_index=0)
        with self.assertRaises(ValueError):
            self.director.spawn_candidate(self.run_id, proposal, candidate_id="candidate:a", slot_index=0)
        before = self.ledger.count()
        with self.assertRaises(ValueError):
            self.director.submit_proposal(replace(proposal, title="different"))
        self.assertEqual(self.ledger.count(), before)
        self.assertEqual(project_run_state(self.ledger.events(self.run_id)).candidate(candidate.candidate_id), candidate)

    def test_advance_requires_complete_joint_decision(self):
        self.fixture._spawn(self.run_id)
        before = self.ledger.count()
        with self.assertRaises(RuntimeError):
            self.director.advance_generation(self.run_id)
        self.assertEqual(self.ledger.count(), before)

    def test_late_proposal_after_pause_cannot_commit(self):
        proposal = self.fixture._proposal(self.run_id)
        self.director.pause_run(self.run_id)
        before = self.ledger.count()
        with self.assertRaises(RuntimeError):
            self.director.submit_proposal(proposal)
        self.assertEqual(self.ledger.count(), before)

    def test_paused_run_rejects_late_artifact_and_scientific_evaluation(self):
        candidate = self.fixture._spawn(self.run_id)
        artifact, evaluation = self.fixture._record_artifact_and_evaluation(self.run_id, candidate.candidate_id)
        self.director.pause_run(self.run_id)
        before = self.ledger.count()
        for value, write in ((artifact, self.director.record_artifact), (evaluation, self.director.record_evaluation)):
            with self.subTest(write=write.__name__), self.assertRaises(RuntimeError):
                write(value)
        self.assertEqual(self.ledger.count(), before)

    def test_artifact_and_evaluation_retries_are_exact_and_candidate_bound(self):
        candidate = self.fixture._spawn(self.run_id)
        artifact, evaluation = self.fixture._record_artifact_and_evaluation(self.run_id, candidate.candidate_id)
        before = self.ledger.count()
        self.assertEqual(self.director.record_artifact(artifact), artifact)
        self.assertEqual(self.director.record_evaluation(evaluation), evaluation)
        self.assertEqual(self.ledger.count(), before)
        with self.assertRaises(ValueError):
            self.director.record_artifact(replace(artifact, training_rows=artifact.training_rows + 1))
        with self.assertRaises(ValueError):
            self.director.record_evaluation(replace(evaluation, score=evaluation.score + .1))
        self.assertEqual(self.ledger.count(), before)

    def test_promotion_cannot_be_invented_from_an_individual_score(self):
        candidate = self.fixture._spawn(self.run_id)
        self.fixture._record_artifact_and_evaluation(self.run_id, candidate.candidate_id)
        before = self.ledger.count()
        for decision in PromotionDecision:
            with self.subTest(decision=decision), self.assertRaisesRegex(ValueError, "comparison"):
                self.director.decide_promotion(Promotion("promotion:forged", self.run_id,
                    candidate.candidate_id, decision, "no joint receipt"))
        self.assertEqual(self.ledger.count(), before)

    def test_evaluation_id_cannot_be_reused_across_candidates(self):
        first = self.fixture._spawn(self.run_id, slot_index=0)
        second = self.fixture._spawn(self.run_id, slot_index=1)
        artifact, evaluation = self.fixture._record_artifact_and_evaluation(self.run_id, first.candidate_id)
        other = replace(artifact, artifact_id="artifact:second", candidate_id=second.candidate_id)
        self.director.record_artifact(other)
        before = self.ledger.count()
        with self.assertRaisesRegex(ValueError, "evaluation_id"):
            self.director.record_evaluation(replace(evaluation, candidate_id=second.candidate_id, artifact_digest=other.digest))
        self.assertEqual(self.ledger.count(), before)

    def test_no_legacy_usage_event_can_enter_current_replay(self):
        self.ledger.append(self.run_id, "ModelUsageRecorded", {"total_tokens": 1})
        with self.assertRaisesRegex(ValueError, "unknown event"):
            project_run_state(self.ledger.events(self.run_id))

    def test_generation_slot_budget_is_enforced_before_append(self):
        for slot in range(4):
            self.fixture._spawn(self.run_id, slot_index=slot)
        before = self.ledger.count()
        with self.assertRaises((RuntimeError, ValueError)):
            self.fixture._spawn(self.run_id, slot_index=4)
        self.assertEqual(self.ledger.count(), before)
