"""Fresh, value-blind three-stage cohorts for the evidence-guided pilot."""
from dataclasses import replace

from .epoch_cohorts import (
    CohortCapacityError, CohortCapacityReport, GenerationCohorts, PlannedBatch,
    PlannedCohort, RunAdaptationCohort, DEFAULT_HISTORY_STEPS, _dataset_horizons,
    _eligible_origins, _dataset_identity, _selection_partition, _strict_integer,
)
from .isolated_cohorts import _Cursor
from ..core.models import digest

PLANNER_SCHEMA = "ecologyrsi-dsh.evidence-guided-cohort-planner/1"


def _epoch(cursor, schedule, seed, horizons, generation):
    # The first batch is an explicit reuse of screening evidence for diagnosis.
    # It causes no additional inference; the next batch is the fresh paired test.
    race = cursor.take(schedule.screening_origin_count)
    local = cursor.take(schedule.local_batch_origin_count, span_hours=72)
    holdout = cursor.take(schedule.selection_holdout_origin_count)
    batches = tuple(PlannedBatch(i, PlannedCohort(
        "adaptation_batch", origins, max(horizons), shared_candidate_count=1,
    )) for i, origins in enumerate((race, local)))
    adaptation = RunAdaptationCohort(race[0].dataset_id, race[0].episode_id, seed,
        PlannedCohort("adaptation", (*race, *local), max(horizons), shared_candidate_count=1),
        batches, planner_schema=PLANNER_SCHEMA)
    return GenerationCohorts(adaptation.dataset_id, adaptation.episode_id, generation,
        seed, adaptation.adaptation_digest, adaptation.batch_digests,
        PlannedCohort("screening", race, max(horizons), shared_candidate_count=5),
        PlannedCohort("holdout", holdout, max(horizons), shared_arm_count=2),
        planner_schema=PLANNER_SCHEMA, adaptation=adaptation)


def plan_adaptation(dataset, *, schedule, seed, history_steps=DEFAULT_HISTORY_STEPS):
    _strict_integer(seed, "seed")
    eligible, _ = _eligible_origins(dataset, history_steps=history_steps)
    return _epoch(_Cursor(eligible), schedule, seed, _dataset_horizons(dataset), 0).adaptation


def plan_selection(dataset, *, schedule, generation, adaptation, seed,
                   history_steps=DEFAULT_HISTORY_STEPS):
    _strict_integer(generation, "generation")
    eligible, _ = _eligible_origins(dataset, history_steps=history_steps)
    cursor = _Cursor(eligible)
    for g in range(generation + 1):
        result = _epoch(cursor, schedule, seed, _dataset_horizons(dataset), g)
        if g == 0 and result.adaptation.adaptation_digest != adaptation.adaptation_digest:
            raise ValueError("race adaptation differs from the frozen initial plan")
    return result


def estimate_capacity(dataset, *, schedule, planned_generations, seed,
                      scoring_cells_per_origin, history_steps=DEFAULT_HISTORY_STEPS):
    _strict_integer(planned_generations, "planned_generations", minimum=1)
    _strict_integer(seed, "seed")
    dataset_id, episode_id, _ = _dataset_identity(dataset)
    eligible, gaps = _eligible_origins(dataset, history_steps=history_steps)
    cursor = _Cursor(eligible)
    feasible = purged = 0
    try:
        while True:
            _epoch(cursor, schedule, seed, _dataset_horizons(dataset), feasible)
            feasible += 1
            if feasible == planned_generations:
                purged = cursor.purged
    except CohortCapacityError:
        pass
    gaps["purged_origin_count"] = purged
    budget = schedule.generation_execution_budget(cells_per_origin=scoring_cells_per_origin)
    return CohortCapacityReport(dataset_id, episode_id, planned_generations,
        _selection_partition(dataset).size, len(eligible),
        schedule.required_unique_origins(planned_generations), feasible,
        feasible >= planned_generations, gaps,
        budget["total_candidate_origins"], budget["total_scoring_cells"],
        budget["total_candidate_origins"] * planned_generations,
        budget["total_scoring_cells"] * planned_generations, digest(schedule.to_dict()), seed,
        cohort_reuse_policy="fresh_training_race_and_confirmation@1",
        reused_origin_occurrences=0, planner_schema=PLANNER_SCHEMA,
        origin_history_alignment_hours=history_steps)
