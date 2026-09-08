"""Qualify complete scientific observations separately from availability scores.

Only the explicitly marked comparison contract uses these checks. They inspect
sealed aggregate evidence, never labels, predictions, or a shared-success mask.
"""
from collections.abc import Mapping
from typing import Any

from ..evaluators.objectives import DEFAULT_TARGET_WEIGHTS
from ..data.adapters import dataset_adapter
from ..core.immutable import thaw_json
from .promotion import _evidence_matches_evaluation, _validated_evidence


_GRID = {(target, horizon) for target in DEFAULT_TARGET_WEIGHTS for horizon in (1, 6, 24)}


def _count(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _complete_scoring_blocks(evaluation: Any) -> dict[str, tuple[int, int]] | None:
    """Return complete day identities/counts after cross-checking frozen totals."""
    scope = getattr(evaluation, "scope", None)
    origins = getattr(scope, "origin_count", None)
    if not isinstance(origins, int) or isinstance(origins, bool) or origins <= 0:
        return None
    metrics = getattr(evaluation, "metrics", None)
    if not isinstance(metrics, Mapping):
        return None
    contract = metrics.get("dataset_task")
    grid = _GRID
    if contract is not None:
        if not isinstance(contract, Mapping):
            return None
        try:
            adapter = dataset_adapter(contract.get("dataset_id"))
        except ValueError:
            return None
        if thaw_json(contract) != adapter.contract():
            return None
        grid = {(target, horizon) for target in adapter.target_names for horizon in adapter.horizons_hours}
    sample = metrics.get("sample_execution")
    if not isinstance(sample, Mapping):
        return None
    if not all(_count(sample.get(key), origins) for key in (
        "attempted_origin_samples", "succeeded_origin_samples", "complete_origin_agent_chains"
    )):
        return None
    if not all(_count(sample.get(key), 0) for key in (
        "failed_origin_samples", "failed_examples", "scoring_fallback_examples"
    )):
        return None
    if sample.get("coverage_pass") is not True or metrics.get("sample_execution_coverage_pass") is not True:
        return None
    # Native evaluator reports cell totals as well. Compact local evidence may
    # omit them; when present, they must agree with the independently summed grid.
    for key in ("eligible_examples", "attempted_examples", "succeeded_examples", "prediction_cell_count"):
        if key in sample and not _count(sample[key], origins * len(grid)):
            return None
    for key in ("input_failures", "skipped_examples"):
        if key in sample and not _count(sample[key], 0):
            return None
    evidence = _validated_evidence(evaluation)
    if evidence is None or not _evidence_matches_evaluation(evidence, evaluation):
        return None
    if {(target, horizon) for target in evidence["weights"] for horizon in evidence["horizons"]} != grid:
        return None
    totals = {cell: 0 for cell in grid}
    day_counts: dict[str, int] = {}
    for block_id, cells in evidence["blocks"].items():
        if set(cells) != grid:
            return None
        counts = {cell["eligible"] for cell in cells.values()}
        if len(counts) != 1 or next(iter(counts)) <= 0:
            return None
        if any(cell["succeeded"] != cell["eligible"] for cell in cells.values()):
            return None
        day_counts[block_id] = next(iter(counts))
        for key, cell in cells.items():
            totals[key] += cell["eligible"]
    if any(total != origins for total in totals.values()):
        return None
    days: dict[str, tuple[int, int]] = {}
    seen_indices: set[int] = set()
    for block in metrics["promotion_block_evidence"]["blocks"]:
        index = block.get("origin_block_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index in seen_indices:
            return None
        seen_indices.add(index)
        block_id = block["block_id"]
        days[block_id] = (index, day_counts[block_id])
    return days


def paired_scoring_evidence_complete(champion: Any, challenger: Any) -> bool:
    """Require both frozen cohorts to have every scientific cell successfully scored."""
    if champion.metrics.get("dataset_task") != challenger.metrics.get("dataset_task"):
        return False
    champion_days = _complete_scoring_blocks(champion)
    challenger_days = _complete_scoring_blocks(challenger)
    return champion_days is not None and champion_days == challenger_days


__all__ = ["paired_scoring_evidence_complete"]
