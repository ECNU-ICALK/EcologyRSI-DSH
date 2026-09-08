"""Value-blind, target-time-purged cohorts for schedule v3.

Only timestamps select observations. Adaptation is shared training material;
each generation receives fresh screening and holdout observations. A later
decision cohort starts strictly after all earlier cohort labels have matured.
"""
from __future__ import annotations

import math

from .epoch_cohorts import (
    CohortCapacityError, CohortCapacityReport, GenerationCohorts, PlannedBatch,
    PlannedCohort, RunAdaptationCohort, ISOLATED_PLANNER_SCHEMA,
    ISOLATED_REUSE_POLICY, _dataset_horizons, _eligible_origins,
    _dataset_identity, _selection_partition, _strict_integer,
)
from ..core.models import digest


class _Cursor:
    def __init__(self, eligible):
        self.eligible = eligible
        self.index = 0
        self.purged = 0

    def take(self, count, *, cross_day=False):
        start = self.index
        # Small adaptation batches still span at least a complete daily cycle.
        stride = max(1, math.ceil(24 / (count - 1))) if cross_day else 1
        indices = range(start, start + count * stride, stride)
        if indices[-1] >= len(self.eligible):
            raise CohortCapacityError(required=indices[-1] + 1,
                                      available=len(self.eligible), max_generations=0)
        result = tuple(self.eligible[i] for i in indices)
        self.index = indices[-1] + 1
        while (self.index < len(self.eligible) and
               self.eligible[self.index].origin_timestamp <= result[-1].maximum_target_timestamp):
            self.index += 1
            self.purged += 1
        return result


def _adaptation(eligible, schedule, seed, horizons):
    _strict_integer(seed, "seed")
    cursor = _Cursor(eligible)
    batches = tuple(PlannedBatch(i, PlannedCohort(
        role="adaptation_batch",
        origins=cursor.take(schedule.local_batch_origin_count, cross_day=True),
        maximum_horizon=max(horizons),
        shared_candidate_count=schedule.finalist_count,
    )) for i in range(schedule.batch_count))
    origins = tuple(o for batch in batches for o in batch.cohort.origins)
    cohort = PlannedCohort("adaptation", origins, max(horizons),
                           shared_candidate_count=schedule.finalist_count)
    return RunAdaptationCohort(origins[0].dataset_id, origins[0].episode_id,
                              seed, cohort, batches,
                              planner_schema=ISOLATED_PLANNER_SCHEMA), cursor


def plan_adaptation(dataset, *, schedule, seed):
    eligible, _ = _eligible_origins(dataset)
    return _adaptation(eligible, schedule, seed, _dataset_horizons(dataset))[0]


def _selection(cursor, schedule, horizons):
    screening = PlannedCohort("screening", cursor.take(schedule.screening_origin_count),
                              max(horizons), shared_candidate_count=4)
    holdout = PlannedCohort("holdout", cursor.take(schedule.selection_holdout_origin_count),
                            max(horizons), shared_arm_count=3)
    return screening, holdout


def plan_selection(dataset, *, schedule, generation, adaptation, seed):
    _strict_integer(generation, "generation")
    eligible, _ = _eligible_origins(dataset)
    expected, cursor = _adaptation(eligible, schedule, seed, _dataset_horizons(dataset))
    if expected.adaptation_digest != adaptation.adaptation_digest:
        raise ValueError("adaptation cohort does not match isolated dataset plan")
    for _ in range(generation + 1):
        screening, holdout = _selection(cursor, schedule, _dataset_horizons(dataset))
    return GenerationCohorts(
        adaptation.dataset_id, adaptation.episode_id, generation, seed,
        adaptation.adaptation_digest, adaptation.batch_digests, screening, holdout,
        planner_schema=ISOLATED_PLANNER_SCHEMA,
    )


def estimate_capacity(dataset, *, schedule, planned_generations, seed, scoring_cells_per_origin):
    _strict_integer(planned_generations, "planned_generations", minimum=1)
    _strict_integer(seed, "seed")
    _strict_integer(scoring_cells_per_origin, "scoring_cells_per_origin", minimum=1)
    dataset_id, episode_id, _ = _dataset_identity(dataset)
    eligible, gaps = _eligible_origins(dataset)
    feasible, purged = 0, 0
    try:
        adaptation, cursor = _adaptation(eligible, schedule, seed, _dataset_horizons(dataset))
        gaps["adaptation_day_buckets"] = len({o.origin_timestamp // 24 for o in adaptation.origins})
        while True:
            _selection(cursor, schedule, _dataset_horizons(dataset))
            feasible += 1
            if feasible == planned_generations:
                purged = cursor.purged
    except CohortCapacityError:
        pass
    gaps["purged_origin_count"] = purged
    required = schedule.required_unique_origins(planned_generations)
    budget = schedule.generation_execution_budget(cells_per_origin=scoring_cells_per_origin)
    return CohortCapacityReport(
        dataset_id, episode_id, planned_generations, _selection_partition(dataset).size,
        len(eligible), required, feasible, feasible >= planned_generations, gaps,
        budget["total_candidate_origins"], budget["total_scoring_cells"],
        budget["total_candidate_origins"] * planned_generations,
        budget["total_scoring_cells"] * planned_generations, digest(schedule.to_dict()), seed,
        cohort_reuse_policy=ISOLATED_REUSE_POLICY, reused_origin_occurrences=0,
        planner_schema=ISOLATED_PLANNER_SCHEMA,
    )
