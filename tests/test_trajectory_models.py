from __future__ import annotations

import unittest

from ecologyrsi_dsh.core.models import Evaluation, ModelArtifact, digest
from ecologyrsi_dsh.core.trajectory import (
    BatchEvaluation,
    CandidateRevision,
    EvaluationPhase,
    EvaluationScope,
    FormalBatch,
    GenerationComparison,
    GenerationHoldout,
    HoldoutArm,
    HoldoutEvaluation,
    RevisionStatus,
)


def _sha(label: str) -> str:
    return digest({"label": label})


def _revision(**patch: object) -> CandidateRevision:
    value = {
        "revision_id": "revision:child",
        "run_id": "run:1",
        "generation": 0,
        "candidate_id": "candidate:a",
        "parent_revision_id": "revision:parent",
        "source_batch_index": 3,
        "genome": {"parameters": {"ridge_alpha": 0.1}},
        "genome_digest": _sha("genome"),
        "behavior_digest": _sha("behavior"),
        "mutation_digest": _sha("mutation"),
        "status": RevisionStatus.ACTIVE,
    }
    value.update(patch)
    return CandidateRevision(**value)


def _scope(
    *,
    candidate_id: str = "candidate:a",
    revision_id: str = "revision:a",
    phase: EvaluationPhase = EvaluationPhase.HOLDOUT,
    cohort: str | None = None,
    batch_index: int | None = None,
    arm: HoldoutArm | None = HoldoutArm.FINALIST_1,
) -> EvaluationScope:
    return EvaluationScope(
        run_id="run:1",
        generation=0,
        candidate_id=candidate_id,
        candidate_revision_id=revision_id,
        phase=phase,
        cohort_digest=cohort or _sha("cohort"),
        origin_count=169 if phase is EvaluationPhase.HOLDOUT else 50,
        batch_index=batch_index,
        holdout_arm=arm,
    )


def _holdout_evaluation(
    arm: HoldoutArm,
    candidate_id: str,
    revision_id: str,
    *,
    cohort: str | None = None,
) -> HoldoutEvaluation:
    scope = _scope(
        candidate_id=candidate_id,
        revision_id=revision_id,
        arm=arm,
        cohort=cohort,
    )
    return HoldoutEvaluation(
        evaluation_id=f"evaluation:{arm.value}",
        scope=scope,
        score=0.5,
        passed=True,
        metrics={"rmse": 1.0},
        evaluator_digest=_sha("evaluator"),
    )


class TrajectoryModelTests(unittest.TestCase):
    def test_revision_identity_covers_parent_genome_and_source_batch(self) -> None:
        revision = _revision()

        self.assertEqual(revision.candidate_id, "candidate:a")
        self.assertEqual(revision.source_batch_index, 3)
        self.assertEqual(revision.revision_digest, digest(revision.identity_dict()))

    def test_revision_deep_copies_nested_genome(self) -> None:
        genome = {"parameters": {"ridge_alpha": 0.1}}
        revision = _revision(genome=genome)

        genome["parameters"]["ridge_alpha"] = 99.0

        self.assertEqual(revision.to_dict()["genome"]["parameters"]["ridge_alpha"], 0.1)
        with self.assertRaises(TypeError):
            revision.genome["parameters"]["ridge_alpha"] = 2.0

    def test_scope_rules_are_phase_specific_and_identity_complete(self) -> None:
        screening = _scope(
            phase=EvaluationPhase.SCREENING,
            arm=None,
            batch_index=None,
        )
        formal = _scope(
            phase=EvaluationPhase.FORMAL_BATCH,
            arm=None,
            batch_index=0,
        )

        self.assertNotEqual(screening.scope_key, formal.scope_key)
        with self.assertRaisesRegex(ValueError, "batch_index"):
            _scope(phase=EvaluationPhase.FORMAL_BATCH, arm=None, batch_index=None)
        with self.assertRaisesRegex(ValueError, "holdout_arm"):
            _scope(phase=EvaluationPhase.HOLDOUT, arm=None)
        with self.assertRaisesRegex(TypeError, "batch_index"):
            _scope(phase=EvaluationPhase.FORMAL_BATCH, arm=None, batch_index=True)

    def test_scope_rejects_non_sha_cohort_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            _scope(cohort="not-a-digest")

    def test_formal_batch_requires_matching_revision_candidate_and_index(self) -> None:
        raw = {
            "batch_id": "batch:a:0",
            "trajectory_id": "trajectory:a",
            "run_id": "run:1",
            "generation": 0,
            "candidate_id": "candidate:a",
            "revision_id": "revision:b",
            "revision_candidate_id": "candidate:b",
            "batch_index": 0,
            "batch_count": 10,
            "cohort_digest": _sha("batch"),
            "origin_count": 50,
        }

        with self.assertRaisesRegex(ValueError, "revision candidate"):
            FormalBatch.from_dict(raw)
        with self.assertRaisesRegex(ValueError, "batch_index"):
            FormalBatch.from_dict(
                {**raw, "revision_candidate_id": "candidate:a", "batch_index": 10}
            )

    def test_holdout_requires_exactly_two_finalists_and_one_incumbent(self) -> None:
        bindings = {
            HoldoutArm.FINALIST_1.value: {
                "candidate_id": "candidate:a",
                "candidate_revision_id": "revision:a",
            },
            HoldoutArm.FINALIST_2.value: {
                "candidate_id": "candidate:b",
                "candidate_revision_id": "revision:b",
            },
        }

        with self.assertRaisesRegex(ValueError, "three holdout arms"):
            GenerationHoldout(
                holdout_id="holdout:0",
                run_id="run:1",
                generation=0,
                cohort_digest=_sha("cohort"),
                origin_count=169,
                arm_bindings=bindings,
            )

    def test_holdout_rejects_repeated_role_binding(self) -> None:
        repeated = {
            arm.value: {
                "candidate_id": "candidate:a",
                "candidate_revision_id": "revision:a",
            }
            for arm in HoldoutArm
        }
        with self.assertRaisesRegex(ValueError, "unique candidate revisions"):
            GenerationHoldout(
                holdout_id="holdout:0",
                run_id="run:1",
                generation=0,
                cohort_digest=_sha("cohort"),
                origin_count=169,
                arm_bindings=repeated,
            )

    def test_comparison_rejects_mixed_cohorts(self) -> None:
        evaluations = (
            _holdout_evaluation(
                HoldoutArm.FINALIST_1, "candidate:a", "revision:a"
            ),
            _holdout_evaluation(
                HoldoutArm.FINALIST_2,
                "candidate:b",
                "revision:b",
                cohort=_sha("other"),
            ),
            _holdout_evaluation(
                HoldoutArm.INCUMBENT, "candidate:i", "revision:i"
            ),
        )

        with self.assertRaisesRegex(ValueError, "same cohort"):
            GenerationComparison(
                comparison_id="comparison:0",
                run_id="run:1",
                generation=0,
                cohort_digest=_sha("cohort"),
                holdout_evaluations=evaluations,
                selected_candidate_id="candidate:a",
                selected_revision_id="revision:a",
                gate_results={"practical_improvement": True},
            )

    def test_batch_evaluation_requires_formal_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "formal_batch"):
            BatchEvaluation(
                evaluation_id="evaluation:batch",
                scope=_scope(),
                score=0.1,
                passed=True,
                metrics={},
                evaluator_digest=_sha("evaluator"),
            )

    def test_artifact_and_evaluation_bind_revision_scope(self) -> None:
        scope = _scope(phase=EvaluationPhase.SCREENING, arm=None).to_dict()
        artifact = ModelArtifact(
            artifact_id="artifact:a",
            run_id="run:1",
            candidate_id="candidate:a",
            model_id="model:1",
            dataset_digest=_sha("dataset"),
            training_partition="training_fit",
            training_rows=10,
            candidate_revision_id="revision:a",
            evaluation_scope_digest=digest(scope),
        )
        evaluation = Evaluation(
            evaluation_id="evaluation:a",
            run_id="run:1",
            candidate_id="candidate:a",
            candidate_revision_id="revision:a",
            evaluation_scope=scope,
            score=0.1,
            passed=True,
        )

        self.assertEqual(artifact.candidate_revision_id, "revision:a")
        self.assertEqual(evaluation.evaluation_scope["phase"], "screening")
        self.assertEqual(evaluation.evaluation_scope_digest, digest(scope))


if __name__ == "__main__":
    unittest.main()
