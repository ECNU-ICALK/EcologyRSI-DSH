"""Fixed chronological training subsets reused across optimization epochs.

Within an epoch, adaptation, screening and comparison stay target-time purged.
Repeating an epoch does not create new independent evidence. Validation/test
are separate datasets and are never consulted by this planner.
"""
from dataclasses import replace

from . import isolated_cohorts
from .epoch_cohorts import TRAINING_REUSE_POLICY, _strict_integer

PLANNER_SCHEMA = "ecologyrsi-dsh.epoch-cohort-planner/3"


def plan_adaptation(dataset, *, schedule, seed):
    return replace(isolated_cohorts.plan_adaptation(dataset, schedule=schedule, seed=seed),
                   planner_schema=PLANNER_SCHEMA)


def plan_selection(dataset, *, schedule, generation, adaptation, seed):
    _strict_integer(generation, "generation")
    expected = plan_adaptation(dataset, schedule=schedule, seed=seed)
    if adaptation.adaptation_digest != expected.adaptation_digest:
        raise ValueError("adaptation does not match the fixed training plan")
    original = isolated_cohorts.plan_adaptation(dataset, schedule=schedule, seed=seed)
    first = isolated_cohorts.plan_selection(dataset, schedule=schedule, generation=0,
                                           adaptation=original, seed=seed)
    def occurrence(cohort):
        return replace(cohort, origins=tuple(replace(o, reuse_index=generation) for o in cohort.origins))
    return replace(first, generation=generation, adaptation_digest=adaptation.adaptation_digest,
                   adaptation_batch_digests=adaptation.batch_digests,
                   screening=occurrence(first.screening), holdout=occurrence(first.holdout),
                   planner_schema=PLANNER_SCHEMA)


def estimate_capacity(dataset, *, schedule, planned_generations, seed, scoring_cells_per_origin):
    _strict_integer(planned_generations, "planned_generations", minimum=1)
    first = isolated_cohorts.estimate_capacity(dataset, schedule=schedule, planned_generations=1,
                                              seed=seed, scoring_cells_per_origin=scoring_cells_per_origin)
    # Candidate/replica executions cost time, not additional independent rows.
    return replace(first, planned_generations=planned_generations,
                   max_feasible_generations=planned_generations if first.sufficient else 0,
                   candidate_origin_executions_for_run=first.candidate_origin_executions_per_generation * planned_generations,
                   scoring_cells_for_run=first.scoring_cells_per_generation * planned_generations,
                   reused_origin_occurrences=first.required_unique_origins * (planned_generations - 1),
                   cohort_reuse_policy=TRAINING_REUSE_POLICY, planner_schema=PLANNER_SCHEMA)
