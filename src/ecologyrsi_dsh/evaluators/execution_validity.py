"""Execution coverage is a prerequisite for scientific stage progression."""

from collections.abc import Mapping
from ..core.errors import DshNativeRuntimeUnavailableError

EXECUTION_OUTCOME_POLICY_KEY = "execution_outcome_policy"
EXECUTION_OUTCOME_POLICY = "terminal_review_rejection@1"


def execution_outcome_policy_enabled(metadata: Mapping) -> bool:
    value = metadata.get(EXECUTION_OUTCOME_POLICY_KEY)
    if value not in (None, EXECUTION_OUTCOME_POLICY):
        raise ValueError("unsupported execution_outcome_policy")
    return value == EXECUTION_OUTCOME_POLICY


def complete_origin_coverage(summary: Mapping, metadata: Mapping) -> bool:
    """New runs cannot pass scientific gates with any failed origin."""
    if not execution_outcome_policy_enabled(metadata):
        return True
    attempted = summary.get("attempted_origin_samples")
    succeeded = summary.get("succeeded_origin_samples")
    return (type(attempted) is int and attempted > 0
            and type(succeeded) is int and succeeded == attempted)


def _terminal_review_rejection(summary: Mapping, origin_count: int) -> bool:
    # These are Host-normalized outcomes of completed critic decisions. An
    # unavailable provider, malformed output, or unknown failure is never a
    # scientific rejection. Counts must account for every origin and cell.
    names = ("attempted_origin_samples", "succeeded_origin_samples", "failed_origin_samples",
             "attempted_examples", "succeeded_examples", "failed_examples")
    if any(type(summary.get(name)) is not int for name in names):
        return False
    attempted, succeeded, failed, cells, successes, failures = (summary[name] for name in names)
    counts = summary.get("failure_counts")
    return (attempted == origin_count and 0 <= succeeded < attempted
            and succeeded + failed == attempted and 0 < failed <= failures
            and cells >= attempted and successes >= 0 and successes + failures == cells
            and isinstance(counts, Mapping) and bool(counts)
            and set(counts) <= {"critic_revise", "critic_uncertain"}
            and all(type(value) is int and value > 0 for value in counts.values())
            and sum(counts.values()) == failures)


def require_valid_execution(metrics: Mapping, origin_count: int, *, phase: str,
                            metadata: Mapping | None = None) -> None:
    summary = metrics.get("sample_execution")
    if not isinstance(summary, Mapping):
        raise DshNativeRuntimeUnavailableError(
            f"{phase} 缺少有效样本执行证据，已停止推进。",
            error_code="evaluation_execution_incomplete",
        )
    attempted = summary.get("attempted_origin_samples", 0)
    succeeded = summary.get("succeeded_origin_samples", 0)
    if attempted != origin_count or succeeded != origin_count:
        if (execution_outcome_policy_enabled(metadata or {})
                and _terminal_review_rejection(summary, origin_count)
                and metrics.get("scientific_pass") is False
                and metrics.get("sample_execution_coverage_pass") is False
                and summary.get("coverage_pass") is False):
            # A fully attempted, rejected candidate can be recorded and lose.
            # Its immutable failed cells must not be resampled until accepted.
            return
        raise DshNativeRuntimeUnavailableError(
            f"{phase} 有效样本 {succeeded}/{origin_count}，不能进入后续批次或科学选择。",
            error_code="evaluation_execution_incomplete",
        )
