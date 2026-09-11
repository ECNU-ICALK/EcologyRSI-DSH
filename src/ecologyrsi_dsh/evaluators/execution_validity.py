"""Execution coverage is a prerequisite for scientific stage progression."""

from collections.abc import Mapping
from ..core.errors import DshNativeRuntimeUnavailableError


def require_valid_execution(metrics: Mapping, origin_count: int, *, phase: str) -> None:
    summary = metrics.get("sample_execution")
    if not isinstance(summary, Mapping):
        raise DshNativeRuntimeUnavailableError(
            f"{phase} 缺少有效样本执行证据，已停止推进。",
            error_code="evaluation_execution_incomplete",
        )
    attempted = summary.get("attempted_origin_samples", 0)
    succeeded = summary.get("succeeded_origin_samples", 0)
    if attempted != origin_count or succeeded != origin_count:
        raise DshNativeRuntimeUnavailableError(
            f"{phase} 有效样本 {succeeded}/{origin_count}，不能进入后续批次或科学选择。",
            error_code="evaluation_execution_incomplete",
        )
