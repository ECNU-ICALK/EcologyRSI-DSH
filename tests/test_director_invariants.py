from __future__ import annotations

import unittest

from ecologyrsi_dsh import (
    CandidateStatus,
    Evaluation,
    EventLedger,
    EvolutionDirector,
    FakeDSHAdapter,
    Promotion,
    PromotionDecision,
    Proposal,
    TaskManifest,
)


def task(max_candidates: int = 3) -> TaskManifest:
    return TaskManifest(
        task_id="invariant-task",
        objective="predict soil water",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={"max_candidates": max_candidates},
        seed=3,
    )


class DirectorInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.director = EvolutionDirector(self.ledger, FakeDSHAdapter(max_proposals=8))
        self.director.start_evolution(task(), run_id="run:invariants")

    def tearDown(self) -> None:
        self.ledger.close()

    def test_spawn_requires_submitted_proposal_and_advance_requires_decision(self) -> None:
        state = self.director.state("run:invariants")
        proposal = Proposal(
            proposal_id="proposal:unsubmitted",
            run_id=state.run.run_id,
            generation=state.run.generation,
            title="unsubmitted",
            changes={"alpha": 0.2, "window": 3, "water_threshold": 0.35},
        )
        with self.assertRaisesRegex(ValueError, "submitted"):
            self.director.spawn_candidate(state.run.run_id, proposal)

        candidate = self.director.propose_and_spawn(state.run.run_id)
        with self.assertRaisesRegex(RuntimeError, "unevaluated"):
            self.director.advance_generation(state.run.run_id)
        evaluation = Evaluation(
            evaluation_id="evaluation:invariant",
            run_id=state.run.run_id,
            candidate_id=candidate.candidate_id,
            score=0.8,
            passed=True,
        )
        self.director.evaluate_and_decide(evaluation)
        self.director.advance_generation(state.run.run_id)

    def test_promotion_cannot_override_evaluation_and_proposals_are_idempotent(self) -> None:
        run_id = "run:invariants"
        state = self.director.state(run_id)
        proposal = Proposal(
            proposal_id="proposal:manual",
            run_id=run_id,
            generation=state.run.generation,
            title="manual",
            changes={"alpha": 0.2},
        )
        self.assertEqual(self.director.submit_proposal(proposal), proposal)
        self.assertEqual(self.director.submit_proposal(proposal), proposal)
        candidate = self.director.spawn_candidate(run_id, proposal.proposal_id)
        evaluation = Evaluation(
            evaluation_id="evaluation:manual",
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            score=0.2,
            passed=False,
        )
        self.director.record_evaluation(evaluation)
        with self.assertRaisesRegex(ValueError, "passing"):
            self.director.decide_promotion(
                Promotion(
                    promotion_id="promotion:bad",
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    decision=PromotionDecision.APPROVED,
                    reason="inconsistent",
                )
            )
        self.director.decide_promotion(
            Promotion(
                promotion_id="promotion:good",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                decision=PromotionDecision.REJECTED,
                reason="visible metric failed",
            )
        )

    def test_model_usage_is_candidate_scoped_validated_and_idempotent(self) -> None:
        run_id = "run:invariants"
        candidate = self.director.propose_and_spawn(run_id)
        usage = {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
        }
        first = self.director.record_model_usage(
            run_id,
            generation=0,
            candidate_id=candidate.candidate_id,
            role="planner",
            model_id="newapi/glm-5.2",
            usage=usage,
            gateway_request_count=2,
            revision="revision:usage",
            usage_index=0,
        )
        duplicate = self.director.record_model_usage(
            run_id,
            generation=0,
            candidate_id=candidate.candidate_id,
            role="planner",
            model_id="newapi/glm-5.2",
            usage=usage,
            gateway_request_count=2,
            revision="revision:usage",
            usage_index=0,
        )
        self.assertEqual(first.seq, duplicate.seq)
        self.assertEqual(
            [event.kind for event in self.director.replay(run_id).events].count(
                "ModelUsageRecorded"
            ),
            1,
        )
        event_count = self.ledger.count(run_id)
        with self.assertRaisesRegex(ValueError, "gateway_request_count"):
            self.director.record_model_usage(
                run_id,
                generation=0,
                candidate_id=candidate.candidate_id,
                role="planner",
                model_id="newapi/glm-5.2",
                usage=usage,
                gateway_request_count=0,
                revision="revision:usage",
                usage_index=1,
            )
        self.assertEqual(self.ledger.count(run_id), event_count)

    def test_model_usage_batch_is_checkpoint_scoped_and_idempotent(self) -> None:
        run_id = "run:invariants"
        candidate = self.director.propose_and_spawn(run_id)
        revision = "revision:checkpoint-usage"
        self.director.start_evaluation_sample_results(
            run_id,
            generation=candidate.generation,
            proposal_id=candidate.proposal_id,
            candidate_id=candidate.candidate_id,
            revision=revision,
            checkpoint={
                "schema_version": "ecologyrsi-dsh.sample-checkpoint/1",
                "cohort_digest": "a" * 64,
                "execution_context_digest": "b" * 64,
                "sample_count": 2,
            },
        )
        reported = {
            "call_id": "call:planner:1",
            "logical_call_digest": "c" * 64,
            "role": "planner",
            "model_id": "newapi/glm-5.2",
            "outcome": "succeeded",
            "usage_reported": True,
            "http_attempts": 2,
            "prompt_tokens": 17,
            "completion_tokens": 3,
            "total_tokens": 20,
        }
        missing = {
            "call_id": "call:critic:1",
            "logical_call_digest": "d" * 64,
            "role": "critic",
            "model_id": "newapi/deepseek-flash",
            "outcome": "failed",
            "usage_reported": False,
            "http_attempts": 1,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        first = self.director.record_model_usage_batch(
            run_id,
            generation=0,
            candidate_id=candidate.candidate_id,
            revision=revision,
            receipts=(reported, missing),
        )
        duplicate = self.director.record_model_usage_batch(
            run_id,
            generation=0,
            candidate_id=candidate.candidate_id,
            revision=revision,
            receipts=(reported, missing),
        )
        self.assertEqual([event.seq for event in first], [event.seq for event in duplicate])
        self.assertEqual([event.payload["usage_index"] for event in first], [0, 1])
        with self.assertRaisesRegex(ValueError, "current checkpoint"):
            self.director.record_model_usage_batch(
                run_id,
                generation=0,
                candidate_id=candidate.candidate_id,
                revision="revision:other",
                receipts=(reported,),
            )

    def test_judge_failure_is_recorded_without_losing_scientific_evidence(self) -> None:
        run_id = "run:invariants"
        candidate = self.director.propose_and_spawn(run_id)
        scientific = Evaluation(
            evaluation_id="evaluation:judge-boundary",
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            score=0.27,
            passed=True,
            metrics={"scientific_pass": True, "skill_score": 0.27},
            partition="training_feedback",
            evaluator_digest="evaluator:fixed-cohort",
        )
        self.director.record_evaluation(scientific)
        judged_data = scientific.to_dict()
        judged_data.update(
            {
                "passed": False,
                "metrics": {
                    **dict(scientific.metrics),
                    "judge_status": "unavailable",
                    "judge_accepted": False,
                    "judge_error_type": "TimeoutError",
                },
            }
        )
        judged = Evaluation.from_dict(judged_data)
        self.assertEqual(self.director.record_judgment(judged), judged)

        state = self.director.state(run_id)
        persisted = state.evaluation_for(candidate.candidate_id)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.score, scientific.score)
        self.assertTrue(persisted.metrics["scientific_pass"])
        self.assertEqual(persisted.metrics["judge_status"], "unavailable")
        self.assertFalse(persisted.passed)
        self.assertEqual(
            [
                event.kind
                for event in state.events
                if event.kind in {"EvaluationRecorded", "EvaluationJudged"}
            ],
            ["EvaluationRecorded", "EvaluationJudged"],
        )

    def test_duplicate_may_reference_an_original_candidate_from_prior_generation(self) -> None:
        run_id = "run:cross-generation-duplicate"
        self.director.start_evolution(task(max_candidates=2), run_id=run_id)
        first = self.director.propose_and_spawn(run_id)
        self.director.evaluate_and_decide(
            Evaluation(
                evaluation_id="evaluation:cross-generation-original",
                run_id=run_id,
                candidate_id=first.candidate_id,
                score=0.1,
                passed=False,
            )
        )
        self.director.advance_generation(run_id)
        second = self.director.propose_and_spawn(run_id)
        marked = self.director.mark_candidate_duplicate(
            run_id, second.candidate_id, first.candidate_id
        )
        self.assertIs(marked.status, CandidateStatus.DUPLICATE)

    def test_passing_candidates_must_strictly_improve_the_incumbent(self) -> None:
        run_id = "run:strict-incumbent"
        self.director.start_evolution(task(max_candidates=4), run_id=run_id)

        def evaluate(score: float, label: str):
            candidate = self.director.propose_and_spawn(run_id)
            promotion = self.director.evaluate_and_decide(
                Evaluation(
                    evaluation_id=f"evaluation:{label}",
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    score=score,
                    passed=True,
                ),
                promotion_id=f"promotion:{label}",
            )
            return candidate, promotion

        first, first_promotion = evaluate(0.8, "incumbent")
        self.assertIs(first_promotion.decision, PromotionDecision.APPROVED)
        self.assertIn("建立当前最优基线", first_promotion.reason)
        self.assertEqual(self.director.state(run_id).run.best_candidate_id, first.candidate_id)

        self.director.advance_generation(run_id)
        lower, lower_promotion = evaluate(0.7, "lower")
        self.assertIs(lower_promotion.decision, PromotionDecision.REJECTED)
        self.assertIn("未严格高于", lower_promotion.reason)
        self.assertIs(
            self.director.state(run_id).candidate(lower.candidate_id).status,
            CandidateStatus.REJECTED,
        )
        self.assertEqual(self.director.state(run_id).run.best_candidate_id, first.candidate_id)

        self.director.advance_generation(run_id)
        _equal, equal_promotion = evaluate(0.8 + 0.5e-12, "equal-within-tolerance")
        self.assertIs(equal_promotion.decision, PromotionDecision.REJECTED)
        self.assertIn("1e-12", equal_promotion.reason)
        self.assertEqual(self.director.state(run_id).run.best_candidate_id, first.candidate_id)

        self.director.advance_generation(run_id)
        higher, higher_promotion = evaluate(0.81, "higher")
        self.assertIs(higher_promotion.decision, PromotionDecision.APPROVED)
        self.assertIn("严格高于", higher_promotion.reason)
        self.assertEqual(
            self.director.state(run_id).run.best_candidate_id,
            higher.candidate_id,
        )
        self.assertEqual(self.director.decide_promotion(first_promotion), first_promotion)

    def test_windowed_candidates_compare_scores_only_within_the_same_cohort(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=8))
        run_id = "run:windowed-incumbent"
        windowed_task = TaskManifest(
            task_id="windowed-invariant-task",
            objective="bounded feedback updates",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={"max_candidates": 3},
            metadata={"samples_per_update": 500},
        )
        try:
            director.start_evolution(windowed_task, run_id=run_id)

            def evaluate(score: float, label: str, cohort_digest: str):
                candidate = director.propose_and_spawn(run_id)
                promotion = director.evaluate_and_decide(
                    Evaluation(
                        evaluation_id=f"evaluation:{label}",
                        run_id=run_id,
                        candidate_id=candidate.candidate_id,
                        score=score,
                        passed=True,
                        metrics={"evaluation_index_digest": cohort_digest},
                    ),
                    promotion_id=f"promotion:{label}",
                )
                return candidate, promotion

            first, first_promotion = evaluate(0.8, "window-a", "a" * 64)
            self.assertIs(first_promotion.decision, PromotionDecision.APPROVED)
            director.advance_generation(run_id)

            changed, changed_promotion = evaluate(0.2, "window-b", "b" * 64)
            self.assertIs(changed_promotion.decision, PromotionDecision.REJECTED)
            self.assertIn("未作正式晋升", changed_promotion.reason)
            self.assertEqual(director.state(run_id).run.best_candidate_id, first.candidate_id)
            director.advance_generation(run_id)

            same, same_promotion = evaluate(0.1, "window-b-lower", "b" * 64)
            self.assertIs(same_promotion.decision, PromotionDecision.REJECTED)
            self.assertIn("未作正式晋升", same_promotion.reason)
            self.assertEqual(director.state(run_id).run.best_candidate_id, first.candidate_id)
            self.assertNotEqual(first.candidate_id, same.candidate_id)
        finally:
            ledger.close()

    def test_windowed_approval_requires_a_verifiable_cohort_digest(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=2))
        run_id = "run:windowed-missing-cohort"
        try:
            director.start_evolution(
                TaskManifest(
                    task_id="windowed-missing-cohort",
                    objective="bounded feedback updates",
                    domain_pack="crop-soil-water@toy",
                    visible_datasets=("generated-toy-series@1",),
                    budget={"max_candidates": 1},
                    metadata={"samples_per_update": 500},
                ),
                run_id=run_id,
            )
            candidate = director.propose_and_spawn(run_id)
            director.record_evaluation(
                Evaluation(
                    evaluation_id="evaluation:missing-window-cohort",
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    score=0.9,
                    passed=True,
                )
            )
            with self.assertRaisesRegex(ValueError, "verifiable evaluation cohort"):
                director.decide_promotion(
                    Promotion(
                        promotion_id="promotion:missing-window-cohort",
                        run_id=run_id,
                        candidate_id=candidate.candidate_id,
                        decision=PromotionDecision.APPROVED,
                        reason="missing cohort must fail closed",
                    )
                )
        finally:
            ledger.close()

    def test_direct_windowed_approval_rejects_a_different_incumbent_cohort(
        self,
    ) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=2))
        run_id = "run:windowed-incumbent-cohort"
        try:
            director.start_evolution(
                TaskManifest(
                    task_id="windowed-incumbent-cohort",
                    objective="bounded feedback updates",
                    domain_pack="crop-soil-water@toy",
                    visible_datasets=("generated-toy-series@1",),
                    budget={"max_candidates": 2},
                    metadata={"samples_per_update": 500},
                ),
                run_id=run_id,
            )
            first = director.propose_and_spawn(run_id)
            director.evaluate_and_decide(
                Evaluation(
                    evaluation_id="evaluation:windowed-incumbent:first",
                    run_id=run_id,
                    candidate_id=first.candidate_id,
                    score=0.7,
                    passed=True,
                    metrics={"evaluation_index_digest": "a" * 64},
                )
            )
            director.advance_generation(run_id)
            second = director.propose_and_spawn(run_id)
            director.record_evaluation(
                Evaluation(
                    evaluation_id="evaluation:windowed-incumbent:second",
                    run_id=run_id,
                    candidate_id=second.candidate_id,
                    score=0.9,
                    passed=True,
                    metrics={"evaluation_index_digest": "b" * 64},
                )
            )

            event_count = ledger.count(run_id)
            with self.assertRaisesRegex(ValueError, "same evaluation cohort"):
                director.decide_promotion(
                    Promotion(
                        promotion_id="promotion:windowed-incumbent:second",
                        run_id=run_id,
                        candidate_id=second.candidate_id,
                        decision=PromotionDecision.APPROVED,
                        reason="must compare the formal incumbent on the same cohort",
                    )
                )

            self.assertEqual(ledger.count(run_id), event_count)
            self.assertEqual(director.state(run_id).run.best_candidate_id, first.candidate_id)
        finally:
            ledger.close()

    def test_direct_windowed_approval_rejects_mismatched_evaluated_sibling(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=2))
        run_id = "run:windowed-sibling-cohort"
        try:
            director.start_evolution(
                TaskManifest(
                    task_id="windowed-sibling-cohort",
                    objective="bounded feedback updates",
                    domain_pack="crop-soil-water@toy",
                    visible_datasets=("generated-toy-series@1",),
                    budget={"max_candidates": 2},
                    metadata={"samples_per_update": 500},
                ),
                run_id=run_id,
            )
            first = director.propose_and_spawn(run_id)
            second = director.propose_and_spawn(run_id)
            director.record_evaluation(
                Evaluation(
                    evaluation_id="evaluation:windowed-sibling:first",
                    run_id=run_id,
                    candidate_id=first.candidate_id,
                    score=0.7,
                    passed=True,
                    metrics={"evaluation_index_digest": "a" * 64},
                )
            )

            # Incremental execution may approve an evaluated candidate without
            # waiting for siblings that have not produced any evidence yet.
            first_promotion = director.decide_promotion(
                Promotion(
                    promotion_id="promotion:windowed-sibling:first",
                    run_id=run_id,
                    candidate_id=first.candidate_id,
                    decision=PromotionDecision.APPROVED,
                    reason="first completed sibling",
                )
            )
            self.assertIs(first_promotion.decision, PromotionDecision.APPROVED)

            director.record_evaluation(
                Evaluation(
                    evaluation_id="evaluation:windowed-sibling:second",
                    run_id=run_id,
                    candidate_id=second.candidate_id,
                    score=0.9,
                    passed=True,
                    metrics={"evaluation_index_digest": "b" * 64},
                )
            )
            event_count = ledger.count(run_id)
            with self.assertRaisesRegex(
                ValueError, "siblings to use the same evaluation cohort"
            ):
                director.decide_promotion(
                    Promotion(
                        promotion_id="promotion:windowed-sibling:second",
                        run_id=run_id,
                        candidate_id=second.candidate_id,
                        decision=PromotionDecision.APPROVED,
                        reason="must not bypass sibling cohort validation",
                    )
                )
            self.assertEqual(ledger.count(run_id), event_count)
            self.assertIs(
                director.state(run_id).candidate(second.candidate_id).status,
                CandidateStatus.EVALUATED,
            )
        finally:
            ledger.close()

    def test_manual_approval_of_non_improving_passing_candidate_fails_closed(self) -> None:
        run_id = "run:manual-incumbent"
        self.director.start_evolution(task(max_candidates=2), run_id=run_id)
        first = self.director.propose_and_spawn(run_id)
        self.director.evaluate_and_decide(
            Evaluation(
                evaluation_id="evaluation:manual-incumbent",
                run_id=run_id,
                candidate_id=first.candidate_id,
                score=0.8,
                passed=True,
            )
        )
        self.director.advance_generation(run_id)
        second = self.director.propose_and_spawn(run_id)
        self.director.record_evaluation(
            Evaluation(
                evaluation_id="evaluation:manual-non-improvement",
                run_id=run_id,
                candidate_id=second.candidate_id,
                score=0.7,
                passed=True,
            )
        )
        event_count = self.ledger.count(run_id)
        with self.assertRaisesRegex(ValueError, "must exceed incumbent"):
            self.director.decide_promotion(
                Promotion(
                    promotion_id="promotion:manual-invalid-approval",
                    run_id=run_id,
                    candidate_id=second.candidate_id,
                    decision=PromotionDecision.APPROVED,
                    reason="人工强制批准较低分候选。",
                )
            )
        self.assertEqual(self.ledger.count(run_id), event_count)

        rejected = Promotion(
            promotion_id="promotion:manual-valid-rejection",
            run_id=run_id,
            candidate_id=second.candidate_id,
            decision=PromotionDecision.REJECTED,
            reason="候选通过门槛但未改善当前最优结果。",
        )
        self.assertEqual(self.director.decide_promotion(rejected), rejected)
        self.assertEqual(self.director.decide_promotion(rejected), rejected)

    def test_evaluation_id_cannot_be_reused_across_candidates(self) -> None:
        run_id = "run:invariants"
        first = self.director.propose_and_spawn(run_id)
        first_evaluation = Evaluation(
            evaluation_id="evaluation:shared",
            run_id=run_id,
            candidate_id=first.candidate_id,
            score=0.8,
            passed=True,
        )
        self.assertEqual(self.director.record_evaluation(first_evaluation), first_evaluation)
        self.assertEqual(self.director.record_evaluation(first_evaluation), first_evaluation)
        self.director.decide_promotion(
            Promotion(
                promotion_id="promotion:first",
                run_id=run_id,
                candidate_id=first.candidate_id,
                decision=PromotionDecision.APPROVED,
                reason="accepted",
            )
        )
        self.director.advance_generation(run_id)
        second = self.director.propose_and_spawn(run_id)
        event_count = self.ledger.count(run_id)
        with self.assertRaisesRegex(ValueError, "evaluation_id"):
            self.director.record_evaluation(
                Evaluation(
                    evaluation_id="evaluation:shared",
                    run_id=run_id,
                    candidate_id=second.candidate_id,
                    score=0.7,
                    passed=True,
                )
            )
        self.assertEqual(self.ledger.count(run_id), event_count)
        self.assertIsNone(self.director.state(run_id).evaluation_for(second.candidate_id))

    def test_promotion_id_cannot_be_reused_across_candidates(self) -> None:
        run_id = "run:invariants"
        first = self.director.propose_and_spawn(run_id)
        self.director.evaluate_and_decide(
            Evaluation(
                evaluation_id="evaluation:first-promotion",
                run_id=run_id,
                candidate_id=first.candidate_id,
                score=0.8,
                passed=True,
            ),
            promotion_id="promotion:shared",
        )
        self.director.advance_generation(run_id)
        second = self.director.propose_and_spawn(run_id)
        self.director.record_evaluation(
            Evaluation(
                evaluation_id="evaluation:second-promotion",
                run_id=run_id,
                candidate_id=second.candidate_id,
                score=0.7,
                passed=True,
            )
        )
        event_count = self.ledger.count(run_id)
        with self.assertRaisesRegex(ValueError, "promotion_id"):
            self.director.decide_promotion(
                Promotion(
                    promotion_id="promotion:shared",
                    run_id=run_id,
                    candidate_id=second.candidate_id,
                    decision=PromotionDecision.APPROVED,
                    reason="accepted",
                )
            )
        self.assertEqual(self.ledger.count(run_id), event_count)
        self.assertIsNone(self.director.state(run_id).promotion_for(second.candidate_id))

    def test_fake_adapter_changes_parameters_across_generations(self) -> None:
        run_id = "run:invariants"
        first = self.director.propose_and_spawn(run_id)
        first_changes = self.director.state(run_id).proposal(first.proposal_id).changes
        self.director.evaluate_and_decide(
            Evaluation(
                evaluation_id="evaluation:first",
                run_id=run_id,
                candidate_id=first.candidate_id,
                score=0.8,
                passed=True,
            )
        )
        self.director.advance_generation(run_id)
        second = self.director.propose_and_spawn(run_id)
        second_changes = self.director.state(run_id).proposal(second.proposal_id).changes
        self.assertNotEqual(dict(first_changes), dict(second_changes))

    def test_generation_budget_is_enforced_by_core_director(self) -> None:
        limited_task = TaskManifest(
            task_id="limited-generation-task",
            objective="enforce generation budget",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={"max_candidates": 3, "max_generations": 1},
            seed=3,
        )
        run_id = "run:generation-budget"
        self.director.start_evolution(limited_task, run_id=run_id)
        candidate = self.director.propose_and_spawn(run_id)
        self.director.evaluate_and_decide(
            Evaluation(
                evaluation_id="evaluation:generation-budget",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                score=0.8,
                passed=True,
            )
        )
        self.director.advance_generation(run_id)
        with self.assertRaisesRegex(RuntimeError, "generation budget"):
            self.director.propose_and_spawn(run_id)
        with self.assertRaisesRegex(RuntimeError, "generation budget"):
            self.director.advance_generation(run_id)
        with self.assertRaisesRegex(RuntimeError, "generation budget"):
            self.director.submit_proposal(
                Proposal(
                    proposal_id="proposal:after-generation-budget",
                    run_id=run_id,
                    generation=1,
                    title="must be rejected",
                    changes={"alpha": 0.2},
                )
            )


if __name__ == "__main__":
    unittest.main()
