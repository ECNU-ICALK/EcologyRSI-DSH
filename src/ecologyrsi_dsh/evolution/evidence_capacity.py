"""Value-blind evidence-capacity checks for newly guarded runs."""

from typing import Any

from ..core.models import digest
from ..core.search_policy import LOCAL_PAIRED_BLOCK_MINIMUM, SEARCH_GUARD_POLICY
from ..evaluators.epoch_cohorts import (
    PlannedCohort, plan_generation_selection_cohorts, plan_run_adaptation_cohort,
)
from .promotion import PROMOTION_BLOCK_HOURS, PROMOTION_MINIMUM_PAIRED_BLOCKS
from .schedule import OptimizationSchedule


def _cohort_evidence(cohort: PlannedCohort, *, minimum: int) -> dict[str, Any]:
    days = {origin.origin_timestamp // PROMOTION_BLOCK_HOURS for origin in cohort.origins}
    return {
        "cohort_digest": cohort.cohort_digest,
        "origin_count": cohort.origin_count,
        "day_block_count": len(days),
        "minimum_day_blocks": minimum,
        "sufficient": len(days) >= minimum,
    }


def guarded_cohort_evidence_capacity(
    dataset: Any,
    *,
    schedule: OptimizationSchedule,
    planned_generations: int,
    seed: int,
) -> dict[str, Any]:
    """Count actual frozen origin-day identities without looking at labels.

    This is a necessary admission check. It does not promise later sample
    success, noninferiority, a positive interval, or independent validation.
    The existing planner remains the authority for every cohort identity.
    """

    if isinstance(planned_generations, bool) or not isinstance(planned_generations, int) or planned_generations < 1:
        raise ValueError("planned_generations must be a positive integer")
    adaptation = plan_run_adaptation_cohort(dataset, schedule=schedule, seed=seed)
    formal = [
        {"batch_index": batch.batch_index,
         **_cohort_evidence(batch.cohort, minimum=LOCAL_PAIRED_BLOCK_MINIMUM)}
        for batch in adaptation.batches
    ]
    holdouts = []
    for generation in range(planned_generations):
        selection = plan_generation_selection_cohorts(
            dataset, schedule=schedule, generation=generation, adaptation=adaptation, seed=seed,
        )
        holdouts.append({"generation": generation,
                         **_cohort_evidence(selection.holdout, minimum=PROMOTION_MINIMUM_PAIRED_BLOCKS)})
    body = {
        "schema_version": "ecologyrsi-dsh.guarded-cohort-evidence-capacity/1",
        "search_guard_policy": SEARCH_GUARD_POLICY,
        "block_hours": PROMOTION_BLOCK_HOURS,
        "schedule_digest": digest(schedule.to_dict()),
        "adaptation_digest": adaptation.adaptation_digest,
        "formal_batches": formal,
        "selection_holdouts": holdouts,
        "sufficient": all(row["sufficient"] for row in (*formal, *holdouts)),
        "scope": "planned_origin_identity_capacity_only_not_scientific_acceptance",
    }
    return {**body, "report_digest": digest(body)}


def require_guarded_cohort_evidence_capacity(**kwargs: Any) -> dict[str, Any]:
    report = guarded_cohort_evidence_capacity(**kwargs)
    for batch in report["formal_batches"]:
        if not batch["sufficient"]:
            raise ValueError(
                "稳健搜索证据不足：局部批次 "
                f"{batch['batch_index'] + 1} 的实际 cohort 仅覆盖 {batch['day_block_count']} 个日块，"
                f"至少需要 {batch['minimum_day_blocks']} 个；请增大局部批次 origin 数，"
                "并将每位入围者的正式 origin 数调整为局部批次 origin 数的整数倍。"
            )
    for holdout in report["selection_holdouts"]:
        if not holdout["sufficient"]:
            raise ValueError(
                "稳健搜索证据不足：第 "
                f"{holdout['generation'] + 1} 轮选择留出 cohort 仅覆盖 {holdout['day_block_count']} 个日块，"
                f"至少需要 {holdout['minimum_day_blocks']} 个；请增加可用时间范围或选择留出 origin 数。"
            )
    return report
