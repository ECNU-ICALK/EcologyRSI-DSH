from __future__ import annotations

from dataclasses import dataclass
import unittest

from ecologyrsi_dsh import EventLedger, EvolutionDirector, FakeDSHAdapter, TaskManifest
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.trajectory import CandidateRevision, RevisionStatus
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.epoch_cohorts import (
    estimate_epoch_capacity,
    plan_generation_selection_cohorts,
    plan_run_adaptation_cohort,
)
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule


@dataclass(frozen=True)
class DatasetFixture:
    dataset_id: str
    episode_id: str
    timestamps: tuple[int, ...]
    partitions: dict[str, IndexRange]
    values: dict[str, tuple[float, ...]]


def dataset_fixture(
    count: int,
    *,
    changed_labels: bool = False,
    timestamp_gap_at: int | None = None,
) -> DatasetFixture:
    timestamps = list(range(count))
    if timestamp_gap_at is not None:
        for index in range(timestamp_gap_at, count):
            timestamps[index] += 1
    multiplier = -1.0 if changed_labels else 1.0
    return DatasetFixture(
        dataset_id="dataset:fixture",
        episode_id="episode:fixture",
        timestamps=tuple(timestamps),
        partitions={"model_selection": IndexRange(0, count)},
        values={"label": tuple(multiplier * index for index in range(count))},
    )


class EpochCohortPlanningTests(unittest.TestCase):
    def test_default_adaptation_cohort_has_ten_ordered_batches(self) -> None:
        schedule = OptimizationSchedule.default()
        adaptation = plan_run_adaptation_cohort(
            dataset_fixture(3200), schedule=schedule, seed=7
        )

        self.assertEqual(adaptation.origin_count, 500)
        self.assertEqual(len(adaptation.batches), 10)
        self.assertEqual(
            [batch.origin_count for batch in adaptation.batches], [50] * 10
        )
        self.assertEqual(
            len(
                {
                    origin
                    for batch in adaptation.batches
                    for origin in batch.origin_ids
                }
            ),
            500,
        )
        self.assertEqual(
            tuple(
                origin
                for batch in adaptation.batches
                for origin in batch.origin_ids
            ),
            adaptation.origin_ids,
        )
        self.assertEqual(
            [batch.batch_index for batch in adaptation.batches], list(range(10))
        )

    def test_generation_cohorts_are_disjoint_value_blind_and_not_reused(self) -> None:
        schedule = OptimizationSchedule.default()
        left_data = dataset_fixture(3200)
        right_data = dataset_fixture(3200, changed_labels=True)
        left_adaptation = plan_run_adaptation_cohort(
            left_data, schedule=schedule, seed=7
        )
        right_adaptation = plan_run_adaptation_cohort(
            right_data, schedule=schedule, seed=7
        )
        left = plan_generation_selection_cohorts(
            left_data,
            schedule=schedule,
            generation=2,
            adaptation=left_adaptation,
            seed=7,
        )
        right = plan_generation_selection_cohorts(
            right_data,
            schedule=schedule,
            generation=2,
            adaptation=right_adaptation,
            seed=7,
        )
        next_generation = plan_generation_selection_cohorts(
            left_data,
            schedule=schedule,
            generation=3,
            adaptation=left_adaptation,
            seed=7,
        )

        self.assertEqual(
            left_adaptation.identity_dict(), right_adaptation.identity_dict()
        )
        self.assertEqual(left.identity_dict(), right.identity_dict())
        self.assertTrue(
            set(left.screening.origin_ids).isdisjoint(left.holdout.origin_ids)
        )
        self.assertTrue(
            set(left.holdout.origin_ids).isdisjoint(left_adaptation.origin_ids)
        )
        self.assertTrue(
            set(left.screening.origin_ids).isdisjoint(left_adaptation.origin_ids)
        )
        self.assertTrue(
            set(left.screening.origin_ids).isdisjoint(
                next_generation.screening.origin_ids
            )
        )
        self.assertTrue(
            set(left.holdout.origin_ids).isdisjoint(
                next_generation.holdout.origin_ids
            )
        )
        self.assertEqual(left.screening.shared_candidate_count, 4)
        self.assertEqual(left.holdout.shared_arm_count, 3)
        self.assertEqual(
            left.adaptation_batch_digests, left_adaptation.batch_digests
        )

    def test_capacity_report_uses_schedule_and_reports_cyclic_reuse(self) -> None:
        schedule = OptimizationSchedule.default()
        dataset = dataset_fixture(3200)
        report = estimate_epoch_capacity(
            dataset, schedule=schedule, planned_generations=5, seed=7
        )

        self.assertEqual(report.required_unique_origins, 1665)
        self.assertEqual(report.candidate_origin_executions_per_generation, 1763)
        self.assertEqual(report.scoring_cells_per_generation, 15867)
        self.assertEqual(report.candidate_origin_executions_for_run, 8815)
        self.assertEqual(report.scoring_cells_for_run, 79335)
        self.assertGreaterEqual(report.available_eligible_origins, 1665)
        self.assertGreaterEqual(report.max_feasible_generations, 5)
        self.assertTrue(report.sufficient)

        too_small = dataset_fixture(1000)
        small_report = estimate_epoch_capacity(
            too_small, schedule=schedule, planned_generations=5, seed=7
        )
        self.assertTrue(small_report.sufficient)
        self.assertEqual(small_report.required_unique_origins, 1665)
        self.assertEqual(small_report.max_feasible_generations, 5)
        self.assertEqual(
            small_report.reused_origin_occurrences,
            small_report.required_unique_origins
            - small_report.available_eligible_origins,
        )
        self.assertEqual(
            small_report.cohort_reuse_policy, "cycle_after_exhaustion@1"
        )

        # A run with fewer eligible origins than the requested five epochs is
        # still executable. Origins are consumed in deterministic order and
        # then repeated with a distinct occurrence index, so each origin
        # occurrence remains auditable without weakening within-cohort pairing.
        small_data = dataset_fixture(773)
        adaptation = plan_run_adaptation_cohort(
            small_data, schedule=schedule, seed=7
        )
        generation = plan_generation_selection_cohorts(
            small_data,
            schedule=schedule,
            generation=1,
            adaptation=adaptation,
            seed=7,
        )
        self.assertEqual(adaptation.origin_count, 500)
        self.assertEqual(generation.screening.origin_count, 64)
        self.assertEqual(generation.holdout.origin_count, 169)
        self.assertTrue(
            set(adaptation.origin_occurrence_keys).isdisjoint(
                generation.screening.origin_occurrence_keys
            )
        )
        self.assertTrue(
            set(generation.screening.origin_occurrence_keys).isdisjoint(
                generation.holdout.origin_occurrence_keys
            )
        )
        self.assertTrue(
            set(adaptation.origin_ids)
            & set(generation.screening.origin_ids)
        )

    def test_non_default_schedule_and_causal_maturity_are_derived(self) -> None:
        schedule = OptimizationSchedule.from_dict(
            {
                **OptimizationSchedule.default().to_dict(),
                "formal_origin_count_per_finalist": 600,
                "local_batch_origin_count": 60,
                "selection_holdout_origin_count": 200,
            }
        )
        report = estimate_epoch_capacity(
            dataset_fixture(3200),
            schedule=schedule,
            planned_generations=3,
            seed=13,
        )
        self.assertEqual(report.required_unique_origins, 1392)
        self.assertEqual(report.candidate_origin_executions_per_generation, 2056)
        self.assertEqual(report.scoring_cells_per_generation, 18504)

        gapped = estimate_epoch_capacity(
            dataset_fixture(3200, timestamp_gap_at=1700),
            schedule=OptimizationSchedule.default(),
            planned_generations=5,
            seed=7,
        )
        self.assertGreater(gapped.maturity_gaps["timestamp_gap_origins"], 0)
        adaptation = plan_run_adaptation_cohort(
            dataset_fixture(3200),
            schedule=OptimizationSchedule.default(),
            seed=7,
        )
        self.assertTrue(
            all(
                origin.maximum_target_timestamp
                == origin.origin_timestamp + 24
                for origin in adaptation.origins
            )
        )


class EpochCohortEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.director = EvolutionDirector(
            self.ledger, FakeDSHAdapter(max_proposals=8)
        )
        self.schedule = OptimizationSchedule.default()
        self.dataset = dataset_fixture(3200)
        self.run_id = "run:epoch-cohorts"
        task = TaskManifest(
            task_id="epoch-cohorts",
            objective="freeze cohort identities",
            domain_pack="crop-soil-water@toy",
            visible_datasets=(self.dataset.dataset_id,),
            budget={
                "max_generations": 1,
                "candidates_per_generation": 4,
                "max_candidates": 4,
            },
            seed=7,
            metadata={
                "episode_id": self.dataset.episode_id,
                "optimization_protocol": "top2_adaptive_epoch@1",
                "optimization_schedule": self.schedule.to_dict(),
            },
        )
        self.director.start_evolution(task, run_id=self.run_id)
        self.candidates = tuple(
            self.director.propose_and_spawn(self.run_id) for _ in range(4)
        )
        for index, candidate in enumerate(self.candidates):
            self.director.create_candidate_revision(
                self.run_id,
                CandidateRevision(
                    revision_id=f"revision:cohort:{index}",
                    run_id=self.run_id,
                    generation=0,
                    candidate_id=candidate.candidate_id,
                    genome={"parameters": {"slot": index}},
                    genome_digest=digest({"genome": index}),
                    behavior_digest=digest({"behavior": index}),
                    mutation_digest=digest({"mutation": index}),
                    status=RevisionStatus.ACTIVE,
                ),
            )
        self.adaptation = plan_run_adaptation_cohort(
            self.dataset, schedule=self.schedule, seed=7
        )
        self.generation = plan_generation_selection_cohorts(
            self.dataset,
            schedule=self.schedule,
            generation=0,
            adaptation=self.adaptation,
            seed=7,
        )

    def tearDown(self) -> None:
        self.ledger.close()

    def test_freeze_events_replay_exact_cohorts_and_gate_screening_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "adaptation cohort"):
            self.director.freeze_generation_selection_cohorts(
                self.run_id, self.generation
            )

        frozen_adaptation = self.director.freeze_run_adaptation_cohort(
            self.run_id, self.adaptation
        )
        frozen_generation = self.director.freeze_generation_selection_cohorts(
            self.run_id, self.generation
        )
        state = self.director.state(self.run_id)
        self.assertEqual(
            state.run_adaptation_cohort.adaptation_digest,
            frozen_adaptation.adaptation_digest,
        )
        self.assertEqual(
            state.generation_cohort_for(0).generation_cohorts_digest,
            frozen_generation.generation_cohorts_digest,
        )

        with self.assertRaisesRegex(ValueError, "frozen screening cohort"):
            self.director.record_candidate_screening(
                self.run_id,
                candidate_id=self.candidates[0].candidate_id,
                generation=0,
                score=0.5,
                passed=True,
                constraint_violations=0,
                origin_count=64,
                prediction_cell_count=64,
                cohort_digest=digest({"wrong": "cohort"}),
            )
        event = self.director.record_candidate_screening(
            self.run_id,
            candidate_id=self.candidates[0].candidate_id,
            generation=0,
            score=0.5,
            passed=True,
            constraint_violations=0,
            origin_count=64,
            prediction_cell_count=64,
            cohort_digest=self.generation.screening.cohort_digest,
        )
        self.assertEqual(
            event.payload["cohort_digest"],
            self.generation.screening.cohort_digest,
        )


if __name__ == "__main__":
    unittest.main()
