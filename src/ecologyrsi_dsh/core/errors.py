"""Host-owned exception types with stable public error codes."""

from __future__ import annotations

from collections.abc import Callable, Iterator


def walk_exception_graph(
    exc: BaseException,
    *,
    max_depth: int = 32,
    max_nodes: int = 256,
) -> Iterator[BaseException]:
    """Yield a bounded, cycle-safe exception graph in causal order.

    Exception groups are read through their ``exceptions`` attribute so this
    module remains importable on Python 3.10, where ``ExceptionGroup`` is not
    a builtin name.
    """

    if max_depth < 0 or max_nodes <= 0:
        return
    pending: list[tuple[BaseException, int]] = [(exc, 0)]
    seen: set[int] = set()
    visited = 0
    while pending and visited < max_nodes:
        current, depth = pending.pop()
        identity = id(current)
        if identity in seen or depth > max_depth:
            continue
        seen.add(identity)
        visited += 1
        yield current
        if depth == max_depth:
            continue
        related: list[BaseException] = []
        cause = getattr(current, "__cause__", None)
        if isinstance(cause, BaseException):
            related.append(cause)
        context = getattr(current, "__context__", None)
        if isinstance(context, BaseException):
            related.append(context)
        grouped = getattr(current, "exceptions", None)
        if isinstance(grouped, (tuple, list)):
            related.extend(item for item in grouped if isinstance(item, BaseException))
        remaining = max_nodes - visited
        pending.extend(
            (item, depth + 1) for item in reversed(related[:remaining])
        )


def find_exception(
    exc: BaseException,
    expected_type: type[BaseException],
    predicate: Callable[[BaseException], bool] | None = None,
) -> BaseException | None:
    """Return the first matching exception from a bounded exception graph."""

    for current in walk_exception_graph(exc):
        if isinstance(current, expected_type) and (
            predicate is None or predicate(current)
        ):
            return current
    return None


FROZEN_RUNTIME_BINDING_DRIFT_CODE = "frozen_runtime_binding_drift"
FROZEN_RUNTIME_BINDING_DRIFT_PUBLIC_MESSAGE = (
    "该运行的冻结算法或模型绑定与当前服务版本不一致。"
    "为保证可复现性，系统已停止继续执行；请使用当前配置新建进化运行。"
)

_PUBLIC_BINDING_LABELS = frozenset(
    {
        "数据集快照",
        "时间分区快照",
        "进化策略实现",
        "评测器实现",
        "预测模型实现",
        "候选生成模型配置",
        "独立评审模型配置",
    }
)


class FrozenRuntimeBindingDriftError(ValueError):
    """Reject replay when an immutable runtime binding changed after creation."""

    error_code = FROZEN_RUNTIME_BINDING_DRIFT_CODE

    def __init__(self, binding_label: str = "冻结运行时绑定") -> None:
        # Only host-owned labels may reach an HTTP response. In particular,
        # neither side of the digest comparison is retained in this message.
        label = (
            binding_label
            if binding_label in _PUBLIC_BINDING_LABELS
            else "冻结运行时绑定"
        )
        super().__init__(
            f"{label}发生漂移；为保证可复现性，旧运行已拒绝继续。"
            "请使用当前配置新建进化运行。"
        )


DSH_NATIVE_RUNTIME_UNAVAILABLE_CODE = "dsh_native_runtime_unavailable"


class DshNativeRuntimeUnavailableError(RuntimeError):
    """Fail a native run without falling back to direct model HTTP calls."""

    error_code = DSH_NATIVE_RUNTIME_UNAVAILABLE_CODE

    def __init__(
        self,
        message: str = "DSH 原生智能体运行时当前不可用。",
        *,
        error_code: str | None = None,
        status_code: int | None = None,
        failure_domain: str | None = None,
        provider_status: int | None = None,
        retry_after_ms: int | None = None,
    ) -> None:
        if error_code:
            self.error_code = error_code
        self.status_code = status_code
        self.failure_domain = failure_domain if failure_domain in {"provider", "execution_contract"} else None
        self.provider_status = provider_status if type(provider_status) is int and 400 <= provider_status <= 599 else None
        self.retry_after_seconds = retry_after_ms / 1000 if type(retry_after_ms) is int and 0 < retry_after_ms <= 3_600_000 else None
        super().__init__(message)


def dsh_native_runtime_retryable(
    exc: DshNativeRuntimeUnavailableError,
) -> bool:
    """Classify service outages separately from fail-closed DSH contracts."""

    if exc.error_code == "structured_child_tool_protocol_error":
        # A protocol error may be emitted by the DSH controller after a
        # transient provider/tool stream interruption.  Retry untyped or 503
        # envelopes; an explicit 422 is a deterministic contract rejection.
        # This keeps malformed model output fail-closed while preventing a
        # missing status field from pausing an otherwise recoverable run.
        if exc.status_code == 422:
            return False
        return exc.status_code is None or (
            exc.failure_domain == "provider" and exc.status_code >= 500
        )

    # Runtime HTTP 503 is also the envelope for completed, deterministic
    # failures. Their Host-owned codes take precedence over transport status:
    # replaying an unchanged output cap cannot repair it.
    if str(getattr(exc, "error_code", "") or "") in {
        "structured_child_output_budget_exhausted",
        "structured_child_execution_budget_exhausted",
        "structured_child_output_schema_invalid",
        "structured_result_missing",
        "dsh_native_runtime_contract_error",
        "evaluation_execution_incomplete",
    }:
        return False
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return status_code in {408, 425, 429} or 500 <= status_code <= 599
    return str(getattr(exc, "error_code", "") or "") in {
        DSH_NATIVE_RUNTIME_UNAVAILABLE_CODE,
        "dsh_native_runtime_not_ready",
        "dsh_native_runtime_transport_error",
    }


def dsh_native_runtime_error_in_chain(
    exc: BaseException,
    *,
    max_depth: int = 32,
) -> DshNativeRuntimeUnavailableError | None:
    """Find the bounded DSH error that owns retry classification."""

    first = None
    for candidate in walk_exception_graph(exc, max_depth=max_depth):
        if isinstance(candidate, DshNativeRuntimeUnavailableError):
            if dsh_native_runtime_evaluation_fatal(candidate):
                return candidate
            if first is None:
                first = candidate
    return first


def preferred_execution_failure(
    first: BaseException | None, later: BaseException,
) -> BaseException:
    """Preserve fatal execution faults when draining parallel workers.

    A sibling's transport error or cancellation can finish first. It must not
    hide the fault that closed admission and cancelled the native evaluator.
    Other failures keep their existing deterministic ordering.
    """
    if first is None:
        return later
    primary = dsh_native_runtime_error_in_chain(first)
    secondary = dsh_native_runtime_error_in_chain(later)
    if secondary is not None and dsh_native_runtime_evaluation_fatal(secondary):
        if primary is None or not dsh_native_runtime_evaluation_fatal(primary):
            return later
    return first


def dsh_native_runtime_evaluation_fatal(exc: DshNativeRuntimeUnavailableError) -> bool:
    """An irrecoverable origin makes a complete frozen cohort impossible.

    Stop admission immediately instead of spending on siblings/retries before
    discovering the same invalid coverage at the batch boundary.
    """
    if exc.error_code == "structured_child_tool_protocol_error":
        return not dsh_native_runtime_retryable(exc)
    return exc.error_code in {
        "structured_child_output_budget_exhausted",
        "structured_child_execution_budget_exhausted",
        "structured_child_output_schema_invalid",
        "structured_result_missing",
        "evaluation_execution_incomplete",
    }


class DshToolAdmissionClosedError(RuntimeError):
    error_code = "dsh_tool_admission_closed"
