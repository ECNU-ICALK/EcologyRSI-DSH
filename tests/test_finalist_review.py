from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ecologyrsi_dsh.application.generation_execution import _ensure_finalist_reviews
from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.finalist_review import (
    FINALIST_REVIEW_QUALIFICATION, finalist_review_evidence,
)
from ecologyrsi_dsh.core.models import Evaluation, RunStatus, TaskManifest
from ecologyrsi_dsh.core.state import validate_generation_comparison_binding
from ecologyrsi_dsh.core.trajectory import GenerationHoldout, HoldoutArm
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from tests.test_paired_execution_qualification import GUARD_METADATA, evaluation


MODEL = "provider/reviewer"


class FinalistReviewTests(unittest.TestCase):
    def setUp(self):
        self.items = tuple(evaluation(arm, skill=.1 if arm is HoldoutArm.INCUMBENT else .2)
                           for arm in HoldoutArm)
        self.task = TaskManifest(task_id="review", objective="review gate", domain_pack="greenhouse_environment@1",
                                 metadata={**GUARD_METADATA, "review_model_id": MODEL})
        self.judgments = {}
        for item in self.items:
            if item.scope.holdout_arm is HoldoutArm.INCUMBENT:
                continue
            self.judgments[item.scope.candidate_id] = Evaluation(
                evaluation_id=item.evaluation_id, run_id=item.scope.run_id,
                candidate_id=item.scope.candidate_id, score=item.score, passed=True,
                metrics={**item.to_dict()["metrics"], "judge_status": "completed",
                         "judge_model_id": MODEL, "judge_accepted": True, "judge_result_digest": "a" * 64},
                artifact_digest="b" * 64, evaluator_digest=item.evaluator_digest,
                candidate_revision_id=item.scope.candidate_revision_id, evaluation_scope=item.scope.to_dict())

    def compare(self, reviews=True):
        return build_generation_comparison(run_id=self.items[0].scope.run_id, generation=0,
            cohort_digest=self.items[0].scope.cohort_digest, holdout_evaluations=self.items,
            require_paired_strict_chain=True, positive_delta_search=False,
            finalist_reviews=finalist_review_evidence(self.items, self.judgments, MODEL) if reviews else None)

    def validate(self, comparison, judgments=None):
        holdout = GenerationHoldout(holdout_id="holdout:review", run_id=comparison.run_id,
            generation=0, cohort_digest=comparison.cohort_digest, origin_count=192,
            arm_bindings={i.scope.holdout_arm.value: dict(candidate_id=i.scope.candidate_id,
                candidate_revision_id=i.scope.candidate_revision_id) for i in self.items})
        validate_generation_comparison_binding(self.task, comparison.run_id, holdout,
            {"exploration_only": False}, comparison,
            persisted_evaluations={i.scope.holdout_arm: i for i in self.items},
            persisted_judgments=self.judgments if judgments is None else judgments)

    def test_positive_scientific_results_cannot_bypass_rejected_or_unavailable_reviews(self):
        accepted = self.compare()
        self.assertEqual(accepted.gate_results["search_eligible_finalist_count"], 2)
        for cid, item in self.judgments.items():
            self.judgments[cid] = replace(item, passed=False, metrics={**item.metrics,
                "judge_accepted": False,
                "judge_status": "unavailable" if cid.endswith("2") else "completed"})
        rejected = self.compare()
        self.assertEqual(rejected.selected_candidate_id, "candidate:incumbent")
        self.assertEqual(rejected.gate_results["certification_eligible_finalist_count"], 0)
        self.assertEqual([i.score for i in accepted.holdout_evaluations],
                         [i.score for i in rejected.holdout_evaluations])
        self.validate(rejected)

    def test_review_evidence_binds_actual_revision_model_and_scientific_score(self):
        cid, item = next(iter(self.judgments.items()))
        for changed in (replace(item, score=item.score + .1),
                        replace(item, metrics={**item.metrics, "judge_model_id": "other/model"}),
                        replace(item, metrics={**item.metrics, "judge_status": "pending"})):
            with self.subTest(changed=changed.metrics.get("judge_status")):
                with self.assertRaises(ValueError):
                    finalist_review_evidence(self.items, {**self.judgments, cid: changed}, MODEL)
        altered = replace(self.items[0], metrics=self.items[0].to_dict()["metrics"],
                          scope=replace(self.items[0].scope, candidate_revision_id="revision:other"))
        with self.assertRaisesRegex(ValueError, "identity"):
            finalist_review_evidence((altered, *self.items[1:]), self.judgments, MODEL)

    def test_replay_rejects_forged_acceptance_and_missing_prior_judgment(self):
        comparison = self.compare()
        self.validate(comparison)
        with self.assertRaisesRegex(ValueError, "durable judgment"):
            self.validate(comparison, {})
        data = comparison.to_dict()["gate_results"]
        data["finalist_reviews"]["finalist_1"]["judge_accepted"] = False
        with self.assertRaisesRegex(ValueError, "durable independent judgments"):
            self.validate(replace(comparison, gate_results=data))

    def test_new_writer_requires_review_but_unmarked_history_replays(self):
        historical = self.compare(reviews=False)
        self.validate(historical, {})
        director = object.__new__(EvolutionDirector)
        director.ledger = Mock()
        state = SimpleNamespace(task_manifest=self.task, comparison_for=lambda _: None)
        director.state = lambda _: state
        with self.assertRaisesRegex(ValueError, "independent finalist review"):
            director.record_generation_comparison(historical.run_id, historical)
        state.comparison_for = lambda _: historical
        self.assertIs(director.record_generation_comparison(historical.run_id, historical), historical)
        director.ledger.append.assert_not_called()

    def test_final_barrier_invokes_only_missing_review_and_is_resumable(self):
        cid = "candidate:finalist_2"
        completed = self.judgments[cid]
        self.judgments[cid] = replace(completed, metrics={k: v for k, v in completed.metrics.items()
                                                       if not k.startswith("judge_")})
        state = SimpleNamespace(task_manifest=self.task, run=SimpleNamespace(status=RunStatus.RUNNING),
            candidate=lambda key: SimpleNamespace(candidate_id=key, proposal_id="proposal:" + key),
            proposal=lambda key: SimpleNamespace(proposal_id=key),
            evaluation_for=lambda key: self.judgments[key], artifact_for=lambda _: SimpleNamespace())
        state.evaluations = tuple(self.judgments.values())
        services = SimpleNamespace(director=SimpleNamespace(state=lambda _: state))
        def finish(*args):
            self.assertEqual(args[-1].candidate_id, cid)
            self.judgments[cid] = completed
            state.evaluations = tuple(self.judgments.values())
        with patch("ecologyrsi_dsh.application.generation_execution._apply_candidate_judge", side_effect=finish) as judge:
            first = _ensure_finalist_reviews(services, self.items[0].scope.run_id, self.items)
            second = _ensure_finalist_reviews(services, self.items[0].scope.run_id, self.items)
        self.assertEqual(first, second)
        judge.assert_called_once()


if __name__ == "__main__":
    unittest.main()
