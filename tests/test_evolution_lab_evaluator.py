import unittest

from ecologyrsi_dsh.evolution_lab.evaluator import EvaluationReport, evaluate_candidate


def report(**overrides):
    base = dict(
        cohort_id="cohort-a",
        scientific_score=0.2,
        per_cell_scores={"temperature@1h": 0.2, "co2@24h": 0.2},
        physical_violations=0,
        skill_success_rate=1.0,
        tool_success_rate=1.0,
        latency_ms=100,
        cost_units=1.0,
        stable=True,
        sample_count=64,
    )
    base.update(overrides)
    return EvaluationReport(**base)


class EvaluatorTests(unittest.TestCase):
    def test_same_cohort_improvement_is_eligible(self):
        decision = evaluate_candidate(report(scientific_score=0.21), report(scientific_score=0.20), practical_delta=0.005)
        self.assertEqual(decision.status, "certification_eligible")
        self.assertGreater(decision.delta, 0.005)

    def test_negative_absolute_score_can_only_be_search_winner(self):
        decision = evaluate_candidate(report(scientific_score=-0.10), report(scientific_score=-0.12), practical_delta=0.005)
        self.assertEqual(decision.status, "search_winner")
        self.assertFalse(decision.certification_eligible)

    def test_different_cohort_or_cell_regression_is_rejected(self):
        self.assertEqual(
            evaluate_candidate(report(cohort_id="b", scientific_score=0.3), report(), practical_delta=0.005).status,
            "rejected",
        )
        self.assertEqual(
            evaluate_candidate(report(scientific_score=0.3, per_cell_scores={"temperature@1h": 0.1, "co2@24h": 0.3}), report(), practical_delta=0.005).status,
            "rejected",
        )


if __name__ == "__main__":
    unittest.main()
