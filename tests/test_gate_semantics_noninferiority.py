"""The per-cell gate must be a real gate, not a relabelled pass-through.

Every assertion here exists because it names a specific way a statistical
non-inferiority rule can be abused. A regression that is large but noisy must
still fail. A regression that is tiny and demonstrably inside the sampling noise
must be admitted, and must be *labelled* as such. Shrinking the cohort to widen
the interval must fail earlier, not pass more easily. And none of it may relax
the requirement that the candidate be better overall.
"""

from __future__ import annotations

import unittest

from ecologyrsi_dsh.evaluators.noninferiority import (
    ADMITTED_WITHIN_UNCERTAINTY,
    CELL_NONINFERIORITY_GATE_ID,
    RELAXATION_DEBT_GATE_ID,
    NoninferiorityPolicy,
    assess_cell_noninferiority,
    assess_relaxation_debt,
)
from ecologyrsi_dsh.evaluators.objectives import DEFAULT_TARGET_WEIGHTS
from ecologyrsi_dsh.evolution.promotion import build_promotion_block_evidence

HORIZON = 6
ORIGINS_PER_BLOCK = 8
BASELINE_ERROR = 1.0


def _rows(
    candidate_error_by_block: dict[str, list[float]],
) -> list[dict[str, object]]:
    """Synthesize scoring rows whose per-cell nRMSE is exactly prescribed.

    ``observed`` is zero and ``normalization_scale`` is one, so a row's
    normalized candidate error *is* the number handed in. With a constant
    magnitude inside a block, that block's nRMSE equals that magnitude, which
    makes ``d`` and its resampling spread directly controllable.
    """

    rows: list[dict[str, object]] = []
    for target, magnitudes in candidate_error_by_block.items():
        for block_index, magnitude in enumerate(magnitudes):
            for hour in range(ORIGINS_PER_BLOCK):
                rows.append(
                    {
                        "target": target,
                        "horizon_hours": HORIZON,
                        "origin_timestamp": block_index * 24 + hour,
                        "observed": 0.0,
                        "predicted": magnitude,
                        "baseline": BASELINE_ERROR,
                        "normalization_scale": 1.0,
                        "sample_execution_status": "succeeded",
                    }
                )
    return rows


def _evidence(candidate_error_by_block: dict[str, list[float]]):
    return build_promotion_block_evidence(
        _rows(candidate_error_by_block),
        horizons=(HORIZON,),
        target_weights=dict(DEFAULT_TARGET_WEIGHTS),
        dataset_digest="a" * 64,
        split_manifest_digest_sha256="b" * 64,
    )


def _cell_metrics(candidate_error_by_block: dict[str, list[float]]):
    """The evaluator's own per-cell rows, in the shape the two paths produce.

    Aggregate nRMSE here is the root-mean-square over blocks, matching what the
    block evidence reduces to, so the legacy-tolerance audit line compares the
    quantity the retired gate actually compared.
    """

    rows = []
    for target, magnitudes in candidate_error_by_block.items():
        mean_square = sum(value**2 for value in magnitudes) / len(magnitudes)
        rows.append(
            {
                "target": target,
                "horizon_hours": HORIZON,
                "normalized_rmse": mean_square**0.5,
                "baseline_normalized_rmse": BASELINE_ERROR,
            }
        )
    return rows


def _uniform(magnitudes: list[float]) -> dict[str, list[float]]:
    return {target: list(magnitudes) for target in DEFAULT_TARGET_WEIGHTS}


def _assess(candidate_error_by_block: dict[str, list[float]], **kwargs):
    policy = NoninferiorityPolicy(**kwargs)
    return assess_cell_noninferiority(
        _evidence(candidate_error_by_block),
        cell_metrics=_cell_metrics(candidate_error_by_block),
        policy=policy,
        legacy_tolerance=1e-12,
    )


# Twelve blocks whose candidate error hovers around the baseline: the mean is a
# hair worse, but block-to-block variation is far larger than that hair, so a
# paired interval on the difference straddles zero.
_NOISY_TIE = [0.86, 1.14, 0.9, 1.12, 0.88, 1.13, 0.91, 1.11, 0.87, 1.15, 0.89, 1.1]


class CellNoninferiorityGateTests(unittest.TestCase):
    def test_large_regression_fails_on_the_absolute_cap_not_the_interval(self) -> None:
        # A replay of last run's air_temperature@6h: candidate nRMSE 2.25x the
        # baseline. The point of this case is *which* clause refuses it. If only
        # the interval clause fired, the gate would be one unlucky cohort away
        # from admitting a 2.25x regression.
        assessment = _assess(_uniform([3.25] * 12))

        self.assertFalse(assessment.passed)
        self.assertTrue(assessment.evidence_sufficient)
        for cell in assessment.evidence["cells"]:
            self.assertAlmostEqual(cell["nrmse_base"], 1.0)
            self.assertAlmostEqual(cell["d"], 2.25)
            self.assertAlmostEqual(cell["hard_cap_value"], 0.03)
            self.assertEqual(cell["failure_reason"], "absolute_regression_cap_exceeded")
        self.assertEqual(assessment.audit["cells_failed"], 3)
        self.assertFalse(
            assessment.audit["would_pass_under_legacy_zero_tolerance"]
        )

    def test_regression_inside_sampling_noise_is_admitted_and_labelled(self) -> None:
        assessment = _assess(_uniform(_NOISY_TIE))

        self.assertTrue(assessment.passed)
        for cell in assessment.evidence["cells"]:
            # Worse on the point estimate, so the retired gate would have
            # refused it, yet the interval covers zero: an honest tie.
            self.assertGreater(cell["d"], 0.0)
            self.assertLessEqual(cell["d"], cell["hard_cap_value"])
            self.assertLess(cell["d_lcb"], 0.0)
            self.assertEqual(cell["admitted_by"], ADMITTED_WITHIN_UNCERTAINTY)
            self.assertFalse(cell["legacy_no_regression_pass"])
        audit = assessment.audit
        self.assertEqual(audit["cells_admitted_only_by_noninferiority"], 3)
        self.assertEqual(audit["cells_strictly_better"], 0)
        self.assertFalse(audit["would_pass_under_legacy_zero_tolerance"])
        # Having used the relaxation, the candidate now owes the stronger overall
        # evidence. The flag is the machine-readable form of that debt.
        self.assertTrue(audit["requires_stronger_overall_evidence"])

    def test_a_strictly_better_candidate_needs_no_relaxation(self) -> None:
        assessment = _assess(_uniform([0.8] * 12))

        self.assertTrue(assessment.passed)
        self.assertEqual(assessment.audit["cells_strictly_better"], 3)
        self.assertEqual(
            assessment.audit["cells_admitted_only_by_noninferiority"], 0
        )
        self.assertFalse(assessment.audit["requires_stronger_overall_evidence"])
        # The one-line reviewer question: true means the candidate simply got
        # better and the new gate changed nothing about this verdict.
        self.assertTrue(
            assessment.audit["would_pass_under_legacy_zero_tolerance"]
        )

    def test_shrinking_the_cohort_fails_earlier_instead_of_passing(self) -> None:
        # The same per-cell numbers as the admitted case, on three blocks instead
        # of twelve. Fewer blocks means a wider interval, so a gate that checked
        # the interval first would reward cohort shrinkage. It must fail at the
        # prerequisite instead.
        assessment = _assess(_uniform(_NOISY_TIE[:3]))

        self.assertFalse(assessment.passed)
        self.assertFalse(assessment.evidence_sufficient)
        self.assertEqual(assessment.evidence["paired_block_count"], 3)
        self.assertEqual(
            assessment.evidence["evidence_failure"], "insufficient_paired_evidence"
        )
        for cell in assessment.evidence["cells"]:
            self.assertEqual(cell["failure_reason"], "insufficient_paired_evidence")
            # No interval is published at all, so nothing downstream can mistake
            # a missing bound for a satisfied one.
            self.assertIsNone(cell["d_lcb"])
        self.assertIsNone(assessment.evidence["resample_family_digest"])

    def test_the_same_evidence_yields_the_same_bounds(self) -> None:
        first = _assess(_uniform(_NOISY_TIE))
        replay = _assess(_uniform(_NOISY_TIE))

        self.assertEqual(first.evidence, replay.evidence)
        self.assertEqual(
            first.evidence["block_ids_digest"], replay.evidence["block_ids_digest"]
        )
        self.assertEqual(
            first.evidence["resample_family_digest"],
            replay.evidence["resample_family_digest"],
        )

    def test_one_bad_cell_fails_the_gate_even_when_the_rest_improve(self) -> None:
        # The reduction is "all". Eight good cells never buy the ninth.
        targets = list(DEFAULT_TARGET_WEIGHTS)
        magnitudes = {target: [0.5] * 12 for target in targets}
        magnitudes[targets[0]] = [3.25] * 12
        assessment = _assess(magnitudes)

        self.assertFalse(assessment.passed)
        self.assertEqual(assessment.failed_cells, (f"{targets[0]}@{HORIZON}h",))
        self.assertEqual(assessment.audit["cells_strictly_better"], 2)

    def test_grid_disagreement_fails_closed(self) -> None:
        # Block evidence and scored rows describing different grids is a
        # measurement bug. It must never be a promotion.
        magnitudes = _uniform([0.5] * 12)
        assessment = assess_cell_noninferiority(
            _evidence(magnitudes),
            cell_metrics=_cell_metrics(magnitudes)[:1],
            policy=NoninferiorityPolicy(),
            legacy_tolerance=1e-12,
        )

        self.assertFalse(assessment.passed)
        self.assertFalse(assessment.evidence_sufficient)
        self.assertEqual(
            assessment.evidence["evidence_failure"], "promotion_block_grid_mismatch"
        )
        self.assertEqual(assessment.evidence["cells"], [])

    def test_the_gate_declares_its_own_semantics(self) -> None:
        assessment = _assess(_uniform([0.5] * 12))

        self.assertEqual(
            assessment.evidence["gate_semantics_id"], CELL_NONINFERIORITY_GATE_ID
        )
        self.assertEqual(assessment.evidence["confidence_level"], 0.95)
        self.assertEqual(assessment.evidence["hard_cap_ratio"], 0.03)
        self.assertEqual(assessment.evidence["valid_three_day_start_count"], 10)

    def test_relaxation_is_charged_for_but_only_when_it_was_used(self) -> None:
        """A tiny overall gain is enough alone, but not alongside a relaxation.

        This is the anti-"quietly lowered the bar" clause. The 1e-9 overall floor
        is trivially clearable, so a candidate that banked unprovable
        non-inferiority on some cell must instead clear the practical delta the
        selection layer already calls a real gain.
        """

        clean = _assess(_uniform([0.5] * 12))
        self.assertFalse(clean.audit["requires_stronger_overall_evidence"])
        dormant = assess_relaxation_debt(
            clean, objective_score=1e-8, escalated_minimum_skill=0.005
        )
        self.assertFalse(dormant["applies"])
        self.assertTrue(dormant["passed"])
        self.assertEqual(dormant["gate_semantics_id"], RELAXATION_DEBT_GATE_ID)

        relaxed = _assess(_uniform(_NOISY_TIE))
        self.assertTrue(relaxed.passed)
        self.assertTrue(relaxed.audit["requires_stronger_overall_evidence"])

        # Same trivial overall gain, now refused.
        binding = assess_relaxation_debt(
            relaxed, objective_score=1e-8, escalated_minimum_skill=0.005
        )
        self.assertTrue(binding["applies"])
        self.assertFalse(binding["passed"])
        self.assertEqual(binding["cells_admitted_only_by_noninferiority"], 3)

        # And paid off by a gain that actually clears the practical delta.
        paid = assess_relaxation_debt(
            relaxed, objective_score=0.02, escalated_minimum_skill=0.005
        )
        self.assertTrue(paid["applies"])
        self.assertTrue(paid["passed"])

        # A missing or non-finite score cannot satisfy a debt that applies.
        for bad in (None, float("nan")):
            self.assertFalse(
                assess_relaxation_debt(
                    relaxed, objective_score=bad, escalated_minimum_skill=0.005
                )["passed"]
            )


if __name__ == "__main__":
    unittest.main()
