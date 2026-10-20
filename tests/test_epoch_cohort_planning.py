"""Current joint cohorts remain value-blind, disjoint and causally mature."""
from dataclasses import dataclass, replace
import unittest
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.epoch_cohorts import (
    estimate_epoch_capacity, plan_generation_selection_cohorts,
    plan_run_adaptation_cohort, CohortCapacityError, RunAdaptationCohort, GenerationCohorts,
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
    def test_current_two_rounds_have_fresh_mature_disjoint_cohorts(self):
        dataset=dataset_fixture(2000);schedule=OptimizationSchedule.for_new_run()
        adaptation=plan_run_adaptation_cohort(dataset,schedule=schedule,seed=7)
        self.assertEqual([b.origin_count for b in adaptation.batches],[10,25])
        self.assertEqual(RunAdaptationCohort.from_dict(adaptation.to_dict()),adaptation)
        seen=set();last_maturity=None
        for generation in range(2):
            selection=plan_generation_selection_cohorts(dataset,schedule=schedule,generation=generation,adaptation=adaptation,seed=7)
            self.assertEqual(GenerationCohorts.from_dict(selection.to_dict()),selection)
            phases=(selection.screening,selection.adaptation.batches[1].cohort,selection.holdout)
            self.assertEqual([p.origin_count for p in phases],[10,25,50])
            for phase in phases:
                ids=set(phase.origin_ids);self.assertFalse(seen & ids);seen.update(ids)
                if last_maturity is not None:self.assertGreater(phase.origins[0].origin_timestamp,last_maturity)
                last_maturity=max(o.maximum_target_timestamp for o in phase.origins)
        self.assertEqual(len(seen),170)

    def test_labels_never_affect_identity_or_capacity(self):
        schedule=OptimizationSchedule.for_new_run();outputs=[]
        for changed in (False,True):
            dataset=dataset_fixture(2000,changed_labels=changed)
            adaptation=plan_run_adaptation_cohort(dataset,schedule=schedule,seed=7)
            outputs.append((adaptation,estimate_epoch_capacity(dataset,schedule=schedule,planned_generations=2,seed=7)))
        self.assertEqual(outputs[0],outputs[1])
        report=outputs[0][1]
        self.assertEqual(report.required_unique_origins,170)
        self.assertEqual(report.candidate_origin_executions_for_run,600)
        self.assertEqual(report.scoring_cells_for_run,5400)
        self.assertEqual(report.reused_origin_occurrences,0)

    def test_insufficient_capacity_and_timestamp_gaps_are_explicit(self):
        schedule=OptimizationSchedule.for_new_run()
        report=estimate_epoch_capacity(dataset_fixture(90),schedule=schedule,planned_generations=1,seed=0)
        self.assertFalse(report.sufficient)
        with self.assertRaises(CohortCapacityError):plan_run_adaptation_cohort(dataset_fixture(90),schedule=schedule,seed=0)
        gap=estimate_epoch_capacity(dataset_fixture(2000,timestamp_gap_at=300),schedule=schedule,planned_generations=2,seed=0)
        self.assertGreater(gap.maturity_gaps['timestamp_gap_origins'],0)

    def test_previous_schedules_and_changed_scientific_counts_are_rejected(self):
        current=OptimizationSchedule.for_new_run()
        for change in ({'schema_version':'ecologyrsi-dsh.top2-adaptive-epoch-schedule/2'},
                       {'local_evaluation_mode':'prequential'}, {'screening_origin_count':11},
                       {'selection_holdout_origin_count':49}, {'finalist_count':True}):
            with self.subTest(change=change),self.assertRaises(ValueError):replace(current,**change)
