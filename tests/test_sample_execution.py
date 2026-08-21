from __future__ import annotations

import base64
import json
import math
import threading
import time
import unittest
import zlib
from unittest.mock import patch

import ecologyrsi_dsh.evaluators.gateway_sample_adapter as gateway_sample_adapter_module
import ecologyrsi_dsh.evaluators.registry as evaluator_registry_module
import ecologyrsi_dsh.evaluators.sample_execution as sample_execution_module
from ecologyrsi_dsh.core.models import Candidate, Proposal, TaskManifest
from ecologyrsi_dsh.core.sample_results import build_sample_results
from ecologyrsi_dsh.core.state import validate_evaluation_progress_payload
from ecologyrsi_dsh.data.registry import DatasetSeries
from ecologyrsi_dsh.evaluators.gateway_sample_adapter import (
    GatewaySampleCollaborationAdapter,
    GatewaySampleTool,
    ModelTokenBudgetExhaustedError,
)
from ecologyrsi_dsh.evaluators.registry import (
    EXOGENOUS_RIDGE_MODEL_ID,
    GREENHOUSE_EVALUATOR_ID,
    GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
    GREENHOUSE_ROLLING_PREDICTOR_ID,
    TOY_DATASET_ID,
    TOY_EVALUATOR_ID,
    TOY_PREDICTOR_MODEL_ID,
    EvaluatorRegistry,
)
from ecologyrsi_dsh.evaluators.sample_execution import (
    COVERAGE_UNREACHABLE_NOT_EXECUTED_FAILURE,
    SAMPLE_EXECUTION_SCHEMA_VERSION,
    SAMPLE_EXECUTION_TRACE_ARCHIVE_VERSION,
    CollaborativeSampleExecutor,
    RegisteredToolCollaborationAdapter,
    SampleExecutionCancelledError,
    SampleExecutionControlError,
    SampleExecutionControlUnavailableError,
    SampleExecutionPausedError,
    SampleExecutionPolicy,
    SamplePredictionRequest,
    encode_sample_execution_trace,
)
from ecologyrsi_dsh.evaluators.shared_sample_context import (
    expand_origin_shared_routing_payload,
)
from ecologyrsi_dsh.evolution.analysis import GenerationAnalysis
from ecologyrsi_dsh.evolution.execution_plan import derive_execution_plan
from ecologyrsi_dsh.integrations.model_gateway import (
    GatewayConfigurationError,
    GatewayResponseError,
    ModelGateway,
)
from ecologyrsi_dsh.splits import IndexRange


class _DatasetStub:
    def __init__(self, series: DatasetSeries) -> None:
        self.value = series

    def series(self, dataset_id, episode_id=None, **kwargs):
        del episode_id, kwargs
        if dataset_id != self.value.dataset_id:
            raise KeyError(dataset_id)
        return self.value


class _FailureAdapter:
    adapter_id = "fake-collaboration"
    adapter_version = "1"

    def __init__(self, *, fail_sample: int | None = None) -> None:
        self.fail_sample = fail_sample
        self.plan_calls = 0
        self.sample_calls = 0

    def plan_batch(self, context):
        self.plan_calls += 1
        return {"plan_id": "fake@1", "algorithm": context["algorithm_id"]}

    def predict_sample(self, request, plan, *, attempt):
        del plan, attempt
        self.sample_calls += 1
        self.assert_label_hidden(request)
        if self.fail_sample == self.sample_calls:
            raise ArithmeticError("synthetic sample failure")
        return {
            "predicted": request.proposed_prediction,
            "agent_decisions": [
                {
                    "role": "forecast_agent",
                    "decision": "predict",
                    "status": "completed",
                }
            ],
            "tool_calls": [
                {"tool_id": "fake-tool", "version": "1", "status": "completed"}
            ],
        }

    @staticmethod
    def assert_label_hidden(request):
        assert not hasattr(request, "observed")
        assert "observed" not in request.to_dict()


class _ControlSignalBatchAdapter(_FailureAdapter):
    def predict_samples(self, requests, plans, *, attempts):
        del requests, plans, attempts
        raise SampleExecutionControlError("pause this run")


class _SampleDecisionGatewayFake:
    def __init__(
        self,
        *,
        planner_tool: str | None = None,
        critic_repair_tool_once: str | None = None,
        failure_critic_tool: str = "terminate",
    ) -> None:
        self.planner_tool = planner_tool
        self.critic_repair_tool_once = critic_repair_tool_once
        self.failure_critic_tool = failure_critic_tool
        self.calls: list[dict] = []

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        payload = {
            "model_id": model_id,
            "role": role,
            "samples": [dict(item) for item in samples],
            "context": dict(context),
            "available_tools": [dict(item) for item in available_tools],
        }
        self._assert_label_hidden(payload)
        self.calls.append(payload)
        decisions = []
        for item in samples:
            if role == "planner":
                next_tool = self.planner_tool or item["sample"]["algorithm_id"]
                reason_code = "initial_registered_route"
            elif role == "repair":
                assert item["failure_feedback"]
                next_tool = "bounded-persistence-fallback"
                reason_code = "repair_after_host_rejection"
            else:
                if "tool_failure" in item:
                    next_tool = self.failure_critic_tool
                    reason_code = (
                        "critic_terminates_failed_tool"
                        if next_tool == "terminate"
                        else "critic_selects_alternate_tool"
                    )
                else:
                    critic_call_count = sum(
                        call["role"] == "critic" for call in self.calls
                    )
                    if self.critic_repair_tool_once and critic_call_count == 1:
                        next_tool = self.critic_repair_tool_once
                        reason_code = "critic_requests_specific_repair"
                    else:
                        next_tool = "accept"
                        reason_code = "remote_review_accept"
            decisions.append(
                {
                    "sample_id": item["sample_id"],
                    "next_tool": next_tool,
                    "reason_code": reason_code,
                    "confidence": 0.9,
                }
            )
        return {"decisions": decisions}

    @classmethod
    def _assert_label_hidden(cls, value):
        if isinstance(value, dict):
            forbidden = {
                "actual",
                "actual_value",
                "ground_truth",
                "label",
                "labels",
                "observed",
                "observation",
                "target_value",
            }
            assert not forbidden.intersection(
                str(key).casefold().replace("-", "_") for key in value
            )
            for item in value.values():
                cls._assert_label_hidden(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                cls._assert_label_hidden(item)


class _AdaptiveSplitGatewayFake(_SampleDecisionGatewayFake):
    def __init__(
        self,
        *,
        maximum_successful_batch_size: int,
        incomplete_decisions: bool = False,
    ) -> None:
        super().__init__()
        self.maximum_successful_batch_size = maximum_successful_batch_size
        self.incomplete_decisions = incomplete_decisions

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        if len(samples) > self.maximum_successful_batch_size:
            payload = {
                "model_id": model_id,
                "role": role,
                "samples": [dict(item) for item in samples],
                "context": dict(context),
                "available_tools": [dict(item) for item in available_tools],
            }
            self._assert_label_hidden(payload)
            self.calls.append(payload)
            if self.incomplete_decisions:
                return {"decisions": []}
            raise GatewayResponseError(
                "model response content must contain one JSON object",
                error_code="output_truncated",
                split_eligible=True,
            )
        response = super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )
        for decision in response["decisions"]:
            decision["reason_code"] = f"route_{decision['sample_id'][-8:]}"
        response["decisions"].reverse()
        return response


class _TruncationRetryGatewayFake(_SampleDecisionGatewayFake):
    """Expose one explicit length failure before returning a valid wave."""

    def __init__(self) -> None:
        super().__init__(planner_tool="algorithm")
        self.diagnostic_limits: list[int | None] = []

    def sample_decide_with_diagnostics(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
        allow_format_retry,
        max_tokens=None,
    ):
        del allow_format_retry
        self.diagnostic_limits.append(max_tokens)
        if len(self.diagnostic_limits) == 1:
            raise GatewayResponseError(
                "model response ended before a complete final JSON object",
                error_code="output_truncated",
                split_eligible=True,
                finish_reason="length",
            )
        return (
            self.sample_decide(
                model_id,
                role=role,
                samples=samples,
                context=context,
                available_tools=available_tools,
            ),
            {"http_attempts": 1, "usage_reported": False},
        )


class _AlwaysFailingGatewayFake:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls: list[dict] = []

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        payload = {
            "model_id": model_id,
            "role": role,
            "samples": [dict(item) for item in samples],
            "context": dict(context),
            "available_tools": [dict(item) for item in available_tools],
        }
        _SampleDecisionGatewayFake._assert_label_hidden(payload)
        self.calls.append(payload)
        raise self.error


class _FailFirstGatewayFake(_SampleDecisionGatewayFake):
    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        if not self.calls:
            payload = {
                "model_id": model_id,
                "role": role,
                "samples": [dict(item) for item in samples],
                "context": dict(context),
                "available_tools": [dict(item) for item in available_tools],
            }
            self._assert_label_hidden(payload)
            self.calls.append(payload)
            raise GatewayResponseError(
                "sample decision contract rejected",
                retryable=False,
            )
        return super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )


class _FailPartitionGatewayFake(_SampleDecisionGatewayFake):
    """Fail one deterministic causal wave while sibling waves may overlap."""

    def __init__(self, partition: str) -> None:
        super().__init__()
        self.partition = partition

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        if any(
            item["sample"]["partition"] == self.partition for item in samples
        ):
            payload = {
                "model_id": model_id,
                "role": role,
                "samples": [dict(item) for item in samples],
                "context": dict(context),
                "available_tools": [dict(item) for item in available_tools],
            }
            self._assert_label_hidden(payload)
            self.calls.append(payload)
            raise GatewayResponseError(
                "sample decision contract rejected",
                retryable=False,
            )
        return super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )


class _FailFirstInvalidToolGatewayFake(_SampleDecisionGatewayFake):
    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        if not self.calls:
            if role != "planner":
                raise AssertionError("the first causal wave must use the planner")
            payload = {
                "model_id": model_id,
                "role": role,
                "samples": [dict(item) for item in samples],
                "context": dict(context),
                "available_tools": [dict(item) for item in available_tools],
            }
            self._assert_label_hidden(payload)
            self.calls.append(payload)
            raise GatewayResponseError(
                "sample decision next_tool is not registered",
                error_code="sample_decision_tool_invalid",
                split_eligible=True,
            )
        if role != "repair":
            raise AssertionError("invalid planner output must enter a repair wave")
        return super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )


class _FailFirstLocalDecisionContractGatewayFake(_SampleDecisionGatewayFake):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        response = super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )
        if not self.failed:
            self.failed = True
            response["decisions"][0]["confidence"] = 2.0
        return response


class _FailFirstCriticGatewayFake(_SampleDecisionGatewayFake):
    def __init__(self, error: BaseException, *, planner_tool: str | None = None) -> None:
        super().__init__(planner_tool=planner_tool)
        self.error = error
        self.failed_critic = False

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        if role == "critic" and not self.failed_critic:
            payload = {
                "model_id": model_id,
                "role": role,
                "samples": [dict(item) for item in samples],
                "context": dict(context),
                "available_tools": [dict(item) for item in available_tools],
            }
            self._assert_label_hidden(payload)
            self.calls.append(payload)
            self.failed_critic = True
            raise self.error
        return super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )


class _ConcurrencyTrackingGatewayFake(_SampleDecisionGatewayFake):
    """Track overlapping calls to guard the adapter's worker-pool contract."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._inflight = 0
        self.max_inflight = 0

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        with self._lock:
            self._inflight += 1
            self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            # Give sibling worker threads an opportunity to enter the gateway.
            time.sleep(0.01)
            return super().sample_decide(
                model_id,
                role=role,
                samples=samples,
                context=context,
                available_tools=available_tools,
            )
        finally:
            with self._lock:
                self._inflight -= 1


class _HeartbeatBlockingGatewayFake(_SampleDecisionGatewayFake):
    """Keep a gateway request in flight long enough to exercise heartbeats."""

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        time.sleep(0.03)
        return super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )


class _DiagnosticSampleDecisionGatewayFake(_SampleDecisionGatewayFake):
    """Return deterministic call-local token and HTTP diagnostics."""

    @staticmethod
    def _receipt(sample_count: int) -> dict[str, int | bool]:
        prompt_tokens = 10 * sample_count
        completion_tokens = 2 * sample_count
        return {
            "usage_reported": True,
            "http_attempts": 2,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def sample_decide_with_diagnostics(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
        allow_format_retry,
    ):
        del allow_format_retry
        response = super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )
        return response, self._receipt(len(samples))


class _BlockingDiagnosticCriticGatewayFake(_DiagnosticSampleDecisionGatewayFake):
    """Expose a planner receipt while the critic call remains in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.critic_started = threading.Event()
        self.release_critic = threading.Event()

    def sample_decide_with_diagnostics(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
        allow_format_retry,
    ):
        if role == "critic":
            self.critic_started.set()
            if not self.release_critic.wait(timeout=2):
                raise AssertionError("test did not release the critic call")
        return super().sample_decide_with_diagnostics(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
            allow_format_retry=allow_format_retry,
        )


class _DiagnosticAdaptiveSplitGatewayFake(_DiagnosticSampleDecisionGatewayFake):
    """Attach diagnostics to failed parents as well as successful children."""

    def __init__(self, *, maximum_successful_batch_size: int) -> None:
        super().__init__()
        self.maximum_successful_batch_size = maximum_successful_batch_size

    def sample_decide_with_diagnostics(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
        allow_format_retry,
    ):
        if len(samples) <= self.maximum_successful_batch_size:
            return super().sample_decide_with_diagnostics(
                model_id,
                role=role,
                samples=samples,
                context=context,
                available_tools=available_tools,
                allow_format_retry=allow_format_retry,
            )
        payload = {
            "model_id": model_id,
            "role": role,
            "samples": [dict(item) for item in samples],
            "context": dict(context),
            "available_tools": [dict(item) for item in available_tools],
        }
        self._assert_label_hidden(payload)
        self.calls.append(payload)
        error = GatewayResponseError(
            "model response content must contain one JSON object",
            error_code="output_truncated",
            split_eligible=True,
        )
        error.http_attempts = 2
        error.token_usage = self._receipt(len(samples))
        raise error


class _ConcurrentDiagnosticAdaptiveSplitGatewayFake(
    _DiagnosticAdaptiveSplitGatewayFake
):
    """Hold two failed roots until both per-call reservations are in flight."""

    def __init__(self) -> None:
        super().__init__(maximum_successful_batch_size=1)
        self._root_lock = threading.Lock()
        self._root_count = 0
        self.roots_started = threading.Event()

    def sample_decide_with_diagnostics(self, model_id, **kwargs):
        samples = kwargs["samples"]
        if len(samples) == 2:
            with self._root_lock:
                self._root_count += 1
                if self._root_count == 2:
                    self.roots_started.set()
            if not self.roots_started.wait(timeout=2):
                raise AssertionError("concurrent root calls did not both start")
        return super().sample_decide_with_diagnostics(model_id, **kwargs)


class _ConcurrentDiagnosticGatewayFake(_DiagnosticSampleDecisionGatewayFake):
    """Require two admitted calls to enter the gateway before either returns."""

    def __init__(self) -> None:
        super().__init__()
        self._admitted_calls = threading.Barrier(2)

    def sample_decide_with_diagnostics(self, model_id, **kwargs):
        try:
            self._admitted_calls.wait(timeout=2)
        except threading.BrokenBarrierError as exc:
            raise AssertionError(
                "two calls were not admitted concurrently"
            ) from exc
        return super().sample_decide_with_diagnostics(model_id, **kwargs)


class _LateDiagnosticReceiptGatewayFake(_DiagnosticSampleDecisionGatewayFake):
    """Return one receipt only after the coordinator callback has failed."""

    def __init__(self) -> None:
        super().__init__()
        self._calls_started = threading.Barrier(2)
        self.release_late_call = threading.Event()

    def sample_decide_with_diagnostics(self, model_id, **kwargs):
        sample_index = int(
            kwargs["samples"][0]["sample"]["partition"].rsplit(":", 1)[1]
        )
        try:
            self._calls_started.wait(timeout=2)
        except threading.BrokenBarrierError as exc:
            raise AssertionError("two calls did not enter the gateway") from exc
        if sample_index == 1:
            if not self.release_late_call.wait(timeout=2):
                raise AssertionError("coordinator callback did not release late call")
            time.sleep(0.05)
        return super().sample_decide_with_diagnostics(model_id, **kwargs)


class _SlidingWindowGatewayFake(_SampleDecisionGatewayFake):
    """Require the third schedule to start before the first can complete."""

    def __init__(self) -> None:
        super().__init__()
        self.first_started = threading.Event()
        self.third_started = threading.Event()
        self.started: list[int] = []
        self._started_lock = threading.Lock()

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        sample_index = int(samples[0]["sample"]["partition"].rsplit(":", 1)[1])
        with self._started_lock:
            self.started.append(sample_index)
        if sample_index == 0:
            self.first_started.set()
            if not self.third_started.wait(timeout=2):
                raise AssertionError("third schedule did not replenish the open slot")
        elif sample_index == 1:
            if not self.first_started.wait(timeout=2):
                raise AssertionError("first schedule did not enter the gateway")
        elif sample_index == 2:
            self.third_started.set()
        return super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )


class _CoverageDrainGatewayFake(_SampleDecisionGatewayFake):
    """Hold one initial request until coverage failure has been published."""

    def __init__(self) -> None:
        super().__init__()
        self.first_started = threading.Event()
        self.second_started = threading.Event()
        self.release_second = threading.Event()
        self.second_drained = False
        self.unexpected_submissions: list[int] = []

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        sample_index = int(samples[0]["sample"]["partition"].rsplit(":", 1)[1])
        if sample_index == 0:
            self.first_started.set()
            if not self.second_started.wait(timeout=2):
                raise AssertionError("second initial schedule did not enter the gateway")
            super().sample_decide(
                model_id,
                role=role,
                samples=samples,
                context=context,
                available_tools=available_tools,
            )
            raise GatewayResponseError("deterministic terminal failure")
        if sample_index == 1:
            if not self.first_started.wait(timeout=2):
                raise AssertionError("first initial schedule did not enter the gateway")
            self.second_started.set()
            if not self.release_second.wait(timeout=2):
                raise AssertionError("in-flight schedule was not drained")
            self.second_drained = True
        elif sample_index > 1:
            self.unexpected_submissions.append(sample_index)
            raise AssertionError("coverage stop submitted unscheduled work")
        return super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )


class _CoverageSpeculativeRefillGatewayFake(_SampleDecisionGatewayFake):
    """Keep a coverage-breaking failure pending while one sibling succeeds."""

    def __init__(self) -> None:
        super().__init__()
        self.release_failure = threading.Event()
        self.unexpected_submissions: list[int] = []

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        sample_index = int(samples[0]["sample"]["partition"].rsplit(":", 1)[1])
        if sample_index == 0:
            if not self.release_failure.wait(timeout=2):
                raise AssertionError("successful sibling progress was not published")
            super().sample_decide(
                model_id,
                role=role,
                samples=samples,
                context=context,
                available_tools=available_tools,
            )
            raise GatewayResponseError("deterministic terminal failure")
        if sample_index > 1:
            self.unexpected_submissions.append(sample_index)
        return super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )


class _CoverageMultipleSlotRefillGatewayFake(_SampleDecisionGatewayFake):
    """Expose several refill slots while one large failure chunk is pending."""

    def __init__(self) -> None:
        super().__init__()
        self.release_pending = threading.Event()
        self.premature_submissions: list[int] = []

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        first_index = int(
            samples[0]["sample"]["partition"].rsplit(":", 1)[1]
        )
        if first_index in {0, 12}:
            if not self.release_pending.wait(timeout=2):
                raise AssertionError("completed initial window did not publish progress")
        elif first_index >= 15 and not self.release_pending.is_set():
            self.premature_submissions.append(first_index)
        result = super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )
        if first_index == 0:
            raise GatewayResponseError("deterministic terminal chunk failure")
        return result


class _RetryableGatewayDrainFake(_SampleDecisionGatewayFake):
    """Fail one schedule only after a sibling is known to be in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.first_started = threading.Event()
        self.second_started = threading.Event()
        self.release_second = threading.Event()
        self.error = GatewayResponseError(
            "policy model request failed: retry budget exhausted",
            retryable=True,
            attempts=4,
        )

    def sample_decide(
        self,
        model_id,
        *,
        role,
        samples,
        context,
        available_tools,
    ):
        sample_index = int(samples[0]["sample"]["partition"].rsplit(":", 1)[1])
        if sample_index == 0:
            self.first_started.set()
            if not self.second_started.wait(timeout=2):
                raise AssertionError("sibling schedule did not enter the gateway")
            raise self.error
        if sample_index == 1:
            self.second_started.set()
            if not self.first_started.wait(timeout=2):
                raise AssertionError("failed schedule did not enter the gateway")
            if not self.release_second.wait(timeout=2):
                raise AssertionError("test did not release successful sibling")
        return super().sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )


class _GatewayHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return self.payload


class _AlwaysMalformedModelOpener:
    def __init__(self) -> None:
        self.calls = 0

    def open(self, request, timeout):
        del request, timeout
        self.calls += 1
        return _GatewayHTTPResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "not-json"},
                    }
                ]
            }
        )


class _ReflectingReasonModelOpener:
    def __init__(self) -> None:
        self.calls = 0

    def open(self, request, timeout):
        del timeout
        self.calls += 1
        authorization = request.get_header("Authorization")
        body = json.loads(request.data.decode("utf-8"))
        user_payload = json.loads(body["messages"][1]["content"])
        model_input = user_payload["input"]
        tool_id = model_input["allowed_tool_ids"][0]
        decisions = [
            {
                "sample_id": sample["sample_id"],
                "next_tool": tool_id,
                "reason_code": authorization,
                "confidence": 1.0,
            }
            for sample in model_input["samples"]
        ]
        return _GatewayHTTPResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"decisions": decisions}),
                        },
                    }
                ]
            }
        )


def _model_gateway(opener, *, token: str = "test-token") -> ModelGateway:
    catalog = json.dumps(
        [
            {
                "model_id": "strategy-model",
                "gateway_url": "http://127.0.0.1:1/v1",
                "model": "test-model",
                "api_key_env": "TEST_MODEL_TOKEN",
            }
        ]
    )
    gateway = ModelGateway.from_env(
        {
            "ECOLOGYRSI_DSH_MODELS_JSON": catalog,
            "TEST_MODEL_TOKEN": token,
            "ECOLOGYRSI_DSH_MODEL_MAX_ATTEMPTS": "8",
            "ECOLOGYRSI_DSH_MODEL_RETRY_BASE_SECONDS": "0",
            "ECOLOGYRSI_DSH_MODEL_RETRY_MAX_SECONDS": "0",
        },
        timeout=1,
    )
    gateway._opener = opener
    return gateway


def _rows() -> list[dict]:
    return [
        {
            "partition": f"training_feedback:{index}",
            "target": "x",
            "horizon_hours": 1,
            "origin_timestamp": 0,
            "timestamp": 1,
            "observed": float(index + 1),
            "predicted": float(index + 1),
            "baseline": float(index + 2),
            "label_free_context": _causal_context(0, [0]),
        }
        for index in range(3)
    ]


def _many_rows(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        row = dict(_rows()[0])
        row.update(
            {
                "partition": f"training_feedback:{index}",
                "observed": float(index + 1),
                "predicted": float(index + 1),
                "baseline": float(index + 2),
            }
        )
        rows.append(row)
    return rows


def _sample_requests(count: int) -> tuple[SamplePredictionRequest, ...]:
    return tuple(
        SamplePredictionRequest(
            sample_id=f"sample:{index}",
            candidate_id="candidate:test",
            dataset_digest="dataset:test",
            partition=f"training_feedback:{index}",
            target="x",
            unit="u",
            horizon_hours=1,
            origin_timestamp=0,
            target_timestamp=1,
            baseline=1.0,
            proposed_prediction=1.0,
            minimum=-100.0,
            maximum=100.0,
            algorithm_id="algorithm",
            algorithm_version="1",
            label_free_context=_causal_context(0, [0]),
        )
        for index in range(count)
    )


def _causal_context(
    origin: float, history_timestamps: list[int | float]
) -> dict:
    return {
        "history_window": [float(timestamp) for timestamp in history_timestamps],
        "causal_provenance": {
            "schema_version": "ecologyrsi-dsh.causal-sample-provenance/1",
            "origin_cutoff_timestamp": origin,
            "latest_context_timestamp": origin,
            "history_timestamps": list(history_timestamps),
        },
    }


def _adjacent_rows() -> list[dict]:
    rows = _rows()
    for index, row in enumerate(rows):
        row.update(
            {
                "partition": "training_feedback",
                "origin_timestamp": index,
                "timestamp": index + 1,
                "label_free_context": _causal_context(index, [index]),
            }
        )
    return rows


def _series() -> DatasetSeries:
    timestamps = tuple(range(100))
    return DatasetSeries(
        schema="ecologyrsi-dsh.dataset-series/1",
        dataset_id="agc_cucumber_2018",
        domain_id="greenhouse_cucumber_2018",
        episode_id="agc_cucumber_2018:TeamA",
        digest="d" * 64,
        timestamps=timestamps,
        values={
            "air_temperature": tuple(
                21.0 + 0.02 * i + 0.1 * math.sin(i / 4) for i in timestamps
            ),
            "relative_humidity": tuple(
                68.0 + 1.5 * math.sin(i / 5) for i in timestamps
            ),
            "co2_concentration": tuple(
                700.0 + 12.0 * math.sin(i / 3) for i in timestamps
            ),
        },
        partitions={
            "training_fit": IndexRange(0, 40),
            "training_feedback": IndexRange(41, 80),
            "development": IndexRange(90, 100),
        },
        features={},
        split_manifest_digest_sha256="s" * 64,
    )


def _candidate(changes: dict) -> tuple[Candidate, Proposal]:
    proposal = Proposal(
        proposal_id="proposal:sample-execution",
        run_id="run:sample-execution",
        generation=0,
        title="sample execution",
        changes=changes,
    )
    return (
        Candidate(
            candidate_id="candidate:sample-execution",
            run_id=proposal.run_id,
            proposal_id=proposal.proposal_id,
            generation=0,
        ),
        proposal,
    )


class SampleExecutionTests(unittest.TestCase):
    def test_durable_result_batches_can_outnumber_gateway_requests(self):
        rows = [
            {"target": "air_temperature", "horizon_hours": 1}
            for _ in range(9)
        ]
        resumed_rows = {
            f"sample-{index}": {
                "status": "succeeded",
                "target": "air_temperature",
                "horizon_hours": 1,
            }
            for index in range(9)
        }
        progress = {
            "batch_index": 5,
            "batch_count": 5,
            "progress_id": 10,
            "gateway_request_count": 4,
        }

        checkpoint = sample_execution_module._adapter_resume_checkpoint(
            rows,
            resumed_rows,
            progress,
        )

        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint["completed_samples"], 9)
        self.assertEqual(checkpoint["batch_index"], 5)
        self.assertEqual(checkpoint["gateway_request_count"], 4)
        normalized = gateway_sample_adapter_module._normalized_resume_checkpoint(
            checkpoint
        )
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["gateway_request_count"], 4)

        validate_evaluation_progress_payload(
            {
                "schema_version": "ecologyrsi-dsh.evaluation-progress/3",
                "generation": 1,
                "proposal_id": "proposal:test",
                "candidate_id": "candidate:test",
                "role": "planner",
                "model_id": "model:test",
                "batch_index": 5,
                "batch_count": 5,
                "batch_size": 1,
                "completed_samples": 9,
                "total_samples": 9,
                "succeeded_samples": 9,
                "failed_samples": 0,
                "gateway_request_count": 4,
                "adaptive_split_trigger_count": 0,
                "adaptive_split_count": 0,
                "adaptive_split_max_depth": 0,
                "adaptive_split_recovered_samples": 0,
                "adaptive_split_failed_samples": 0,
                "revision": "revision:test",
                "progress_id": 10,
                "progress_kind": "completed_batch",
                "in_flight_batches": 0,
                "queued_batches": 0,
            }
        )

    def test_tool_feedback_uses_the_second_execution_schema(self):
        self.assertEqual(
            SAMPLE_EXECUTION_SCHEMA_VERSION,
            "ecologyrsi-dsh.sample-execution/2",
        )
        self.assertEqual(
            SAMPLE_EXECUTION_TRACE_ARCHIVE_VERSION,
            "ecologyrsi-dsh.sample-execution-trace/2",
        )

    def execute(
        self,
        adapter,
        *,
        rows=None,
        max_attempts=1,
        minimum_coverage=0.6,
        minimum_task_coverage=None,
        result_callback=None,
    ):
        if minimum_task_coverage is None:
            minimum_task_coverage = minimum_coverage
        selected_rows = rows or _rows()
        target_bounds = {
            str(row["target"]): {
                "unit": "u",
                "minimum": -100.0,
                "maximum": 100.0,
            }
            for row in selected_rows
        }
        return CollaborativeSampleExecutor(adapter, sleep=lambda _: None).execute(
            selected_rows,
            context={
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            },
            target_bounds=target_bounds,
            algorithm_id="algorithm",
            algorithm_version="1",
            policy=SampleExecutionPolicy(
                max_attempts=max_attempts,
                plan_max_attempts=1,
                minimum_coverage=minimum_coverage,
                minimum_task_coverage=minimum_task_coverage,
            ),
            result_callback=result_callback,
        )

    def test_partial_failure_uses_fixed_cohort_fallback_and_compact_trace(self):
        adapter = _FailureAdapter(fail_sample=2)
        batch = self.execute(adapter)
        self.assertEqual(adapter.plan_calls, 1)
        self.assertEqual(adapter.sample_calls, 3)
        self.assertEqual(batch.summary["succeeded_examples"], 2)
        self.assertEqual(batch.summary["failed_examples"], 1)
        self.assertTrue(batch.summary["coverage_pass"])
        self.assertEqual(len(batch.scoring_rows), 3)
        self.assertEqual(batch.scoring_rows[1]["predicted"], _rows()[1]["baseline"])
        self.assertEqual(
            batch.scoring_rows[1]["scoring_fallback"],
            "failure_non_improvement_penalty",
        )
        success = next(item for item in batch.records if item["status"] == "succeeded")
        self.assertEqual(
            set(success),
            {
                "schema_version",
                "sample_id",
                "status",
                "attempts",
                "retry_count",
                "predicted",
                "batch_plan_digest",
                "action_digest",
            },
        )
        self.assertNotIn("agent_decisions", success)
        self.assertTrue(batch.summary["action_catalog"])

    def test_batched_run_control_signal_crosses_sample_failure_isolation(self):
        with self.assertRaisesRegex(SampleExecutionControlError, "pause this run"):
            self.execute(_ControlSignalBatchAdapter())

    def test_registered_forecast_tool_is_invoked_for_every_sample(self):
        called: list[str] = []

        def forecast_tool(request):
            called.append(request.sample_id)
            return request.proposed_prediction

        adapter = RegisteredToolCollaborationAdapter(forecast_tool=forecast_tool)
        batch = self.execute(adapter)

        self.assertEqual(called, [row["sample_id"] for row in batch.records])
        self.assertEqual(batch.summary["failed_examples"], 0)
        self.assertEqual(
            batch.summary["batch_plan"]["agent_runtime"],
            "host_feedback_state_machine",
        )
        self.assertFalse(batch.summary["remote_sample_agents"])

    def test_identical_rows_receive_distinct_stable_sample_ids(self):
        row = _rows()[0]
        batch = self.execute(
            RegisteredToolCollaborationAdapter(),
            rows=[dict(row), dict(row)],
        )

        self.assertEqual(batch.summary["succeeded_examples"], 2)
        self.assertEqual(len({record["sample_id"] for record in batch.records}), 2)

    def test_invalid_sample_request_is_local_failure_not_candidate_failure(self):
        rows = _rows()
        rows[1]["predicted"] = float("nan")

        batch = self.execute(RegisteredToolCollaborationAdapter(), rows=rows)

        self.assertEqual(batch.summary["succeeded_examples"], 2)
        self.assertEqual(batch.summary["failed_examples"], 1)
        self.assertEqual(batch.summary["input_failures"], 1)
        self.assertEqual(batch.summary["scoring_fallback_examples"], 1)
        self.assertEqual(len(batch.scoring_rows), 3)
        failed = next(item for item in batch.records if item["status"] == "failed")
        self.assertEqual(failed["attempts"], 0)
        self.assertEqual(failed["failure"]["class"], "invalid_sample_input")
        self.assertEqual(
            batch.scoring_rows[1]["scoring_fallback"],
            "invalid_input_physical_penalty",
        )

    def test_checkpoint_resume_calls_only_pending_samples_and_keeps_full_cohort(self):
        rows = _rows()
        first = self.execute(_FailureAdapter(), rows=rows)
        persisted = build_sample_results(
            "candidate:test", (first.scoring_rows[0],)
        )
        adapter = _FailureAdapter()
        descriptors = []
        published = []
        bounds = {"x": {"unit": "u", "minimum": -100.0, "maximum": 100.0}}
        batch = CollaborativeSampleExecutor(
            adapter, sleep=lambda _: None
        ).execute(
            rows,
            context={
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            },
            target_bounds=bounds,
            algorithm_id="algorithm",
            algorithm_version="1",
            policy=SampleExecutionPolicy(max_attempts=1),
            checkpoint_callback=lambda descriptor: (
                descriptors.append(dict(descriptor))
                or {"rows": persisted, "progress": None}
            ),
            result_callback=lambda values: published.extend(
                dict(value) for value in values
            ),
        )

        self.assertEqual(adapter.sample_calls, 2)
        self.assertEqual(descriptors[0]["sample_count"], 3)
        self.assertEqual(len(descriptors[0]["cohort_digest"]), 64)
        self.assertEqual(len(batch.scoring_rows), 3)
        self.assertEqual(batch.summary["checkpoint_resumed_examples"], 1)
        self.assertEqual(batch.summary["checkpoint_pending_examples"], 2)
        self.assertEqual(batch.summary["succeeded_examples"], 3)
        self.assertEqual(batch.summary["coverage"], 1.0)
        self.assertTrue(batch.records[0]["checkpoint_resumed"])
        published_ids = {row["sample_id"] for row in published}
        self.assertEqual(
            published_ids,
            {row["sample_id"] for row in batch.scoring_rows[1:]},
        )
        self.assertNotIn(persisted[0]["sample_id"], published_ids)

    def test_resumed_failure_stops_before_gateway_submission(self):
        rows = _many_rows(4)
        failed_seed = self.execute(
            _FailureAdapter(fail_sample=1), rows=rows, max_attempts=1
        )
        persisted_failure = build_sample_results(
            "candidate:test", (failed_seed.scoring_rows[0],)
        )
        gateway = _SampleDecisionGatewayFake()
        progress = []
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=1,
            progress_callback=lambda value: progress.append(dict(value)),
        )
        batch = CollaborativeSampleExecutor(
            adapter, sleep=lambda _: None
        ).execute(
            rows,
            context={
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            },
            target_bounds={
                "x": {"unit": "u", "minimum": -100.0, "maximum": 100.0}
            },
            algorithm_id="algorithm",
            algorithm_version="1",
            policy=SampleExecutionPolicy(
                max_attempts=1,
                minimum_coverage=0.8,
                minimum_task_coverage=0.8,
            ),
            checkpoint_callback=lambda _descriptor: {
                "rows": persisted_failure,
                "progress": None,
            },
        )

        self.assertEqual(gateway.calls, [])
        self.assertEqual(progress, [])
        self.assertEqual(batch.summary["checkpoint_resumed_examples"], 1)
        self.assertTrue(batch.summary["coverage_early_stopped"])
        self.assertEqual(batch.summary["succeeded_examples"], 0)
        self.assertEqual(batch.summary["failed_examples"], 4)
        self.assertEqual(len(batch.scoring_rows), 4)

    def test_failure_cannot_gain_score_by_removing_a_difficult_sample(self):
        complete = self.execute(_FailureAdapter()).scoring_rows
        partial = self.execute(_FailureAdapter(fail_sample=2)).scoring_rows

        def rmse(rows):
            errors = [row["predicted"] - row["observed"] for row in rows]
            return math.sqrt(sum(error * error for error in errors) / len(errors))

        self.assertGreaterEqual(rmse(partial), rmse(complete))

    def test_failure_cannot_replace_a_worse_proposal_with_a_better_baseline(self):
        rows = _rows()
        rows[1].update(
            {
                "predicted": rows[1]["observed"] + 20.0,
                "baseline": rows[1]["observed"] + 1.0,
            }
        )
        complete = self.execute(_FailureAdapter(), rows=rows).scoring_rows
        partial = self.execute(
            _FailureAdapter(fail_sample=2), rows=rows
        ).scoring_rows

        self.assertEqual(partial[1]["predicted"], rows[1]["predicted"])
        self.assertEqual(
            partial[1]["scoring_fallback_source"],
            "registered_algorithm_prediction",
        )
        self.assertEqual(
            sum((row["predicted"] - row["observed"]) ** 2 for row in partial),
            sum((row["predicted"] - row["observed"]) ** 2 for row in complete),
        )

    def test_default_critic_repairs_out_of_range_prediction(self):
        row = _rows()[0]
        row.update({"predicted": 999.0, "baseline": 3.0})
        batch = self.execute(
            RegisteredToolCollaborationAdapter(), rows=[row], max_attempts=3
        )
        self.assertEqual(batch.scoring_rows[0]["predicted"], 3.0)
        self.assertEqual(batch.summary["repair_count"], 1)
        self.assertEqual(batch.summary["retry_count"], 2)
        self.assertEqual(batch.summary["exploration_failures"], 2)
        self.assertEqual(batch.summary["recovered_examples"], 1)
        self.assertEqual(batch.records[0]["attempts"], 3)
        self.assertEqual(
            [item["failure_class"] for item in batch.records[0]["failure_history"]],
            ["constraint_rejected", "constraint_rejected"],
        )
        decisions = batch.summary["action_catalog"][0]["agent_decisions"]
        self.assertEqual(decisions[0]["role"], "forecast_agent")
        self.assertIn("failure_analyst", [item["role"] for item in decisions])
        self.assertIn("repair_agent", [item["role"] for item in decisions])
        self.assertEqual(decisions[-1]["role"], "host_adjudicator")
        tools = batch.summary["action_catalog"][0]["tool_calls"]
        self.assertEqual(
            [item["tool_id"] for item in tools],
            [
                "algorithm",
                "physical-range-check",
                "bounded-projection-repair",
                "physical-range-check",
                "bounded-persistence-fallback",
                "physical-range-check",
            ],
        )
        self.assertEqual(
            batch.summary["critic_outcome_counts"], {"accepted": 1, "rejected": 2}
        )
        self.assertEqual(
            batch.summary["repair_tool_outcomes"],
            {
                "bounded-persistence-fallback": {"completed": 1},
                "bounded-projection-repair": {"completed": 1},
            },
        )
        self.assertEqual(
            batch.summary["recovered_by_failure_class"],
            {"constraint_rejected": 1},
        )
        self.assertNotIn("observed", str(batch.summary))

    def test_previous_constraint_failure_changes_next_repair_route(self):
        analysis = GenerationAnalysis(
            run_id="run:sample-route",
            generation=0,
            candidate_count=1,
            eligible_count=0,
            outcome="no_eligible_candidate",
            sample_failures=(
                {
                    "attempted": 1,
                    "failed": 1,
                    "coverage_pass": False,
                    "failure_counts": {"constraint_rejected": 1},
                },
            ),
        )
        execution_plan = derive_execution_plan(analysis)
        row = _rows()[0]
        row.update({"predicted": 999.0, "baseline": 3.0})
        batch = CollaborativeSampleExecutor(sleep=lambda _: None).execute(
            [row],
            context={
                "candidate_id": "candidate:next-generation",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
                "derived_execution_plan": execution_plan.to_dict(),
            },
            target_bounds={
                "x": {"unit": "u", "minimum": -100.0, "maximum": 100.0}
            },
            algorithm_id="algorithm",
            algorithm_version="1",
            policy=SampleExecutionPolicy(max_attempts=3),
        )

        self.assertEqual(batch.records[0]["attempts"], 2)
        self.assertEqual(batch.summary["repair_count"], 1)
        self.assertEqual(
            batch.summary["batch_plan"]["derived_execution_plan"]["plan_digest"],
            execution_plan.plan_digest,
        )
        tools = batch.summary["action_catalog"][0]["tool_calls"]
        self.assertEqual(
            [item["tool_id"] for item in tools],
            [
                "algorithm",
                "physical-range-check",
                "bounded-persistence-fallback",
                "physical-range-check",
            ],
        )

    def test_gateway_adapter_microbatches_every_sample_and_repairs_sparse_failure(self):
        rows = _rows()
        for index, row in enumerate(rows):
            row["label_free_context"] = {
                **row["label_free_context"],
                "history_window_ref": f"window:{index}",
                "features": {"air_temperature": 20.0 + index},
            }
        rows[1].update({"predicted": 999.0, "baseline": 3.0})
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
        )

        batch = self.execute(adapter, rows=rows, max_attempts=2)

        planner_calls = [item for item in gateway.calls if item["role"] == "planner"]
        repair_calls = [item for item in gateway.calls if item["role"] == "repair"]
        critic_calls = [item for item in gateway.calls if item["role"] == "critic"]
        self.assertEqual(len(planner_calls), 1)
        self.assertEqual(len(planner_calls[0]["samples"]), 3)
        self.assertEqual(len(repair_calls), 1)
        self.assertEqual(len(repair_calls[0]["samples"]), 1)
        self.assertEqual(critic_calls, [])
        self.assertEqual(batch.summary["failed_examples"], 0)
        self.assertTrue(batch.summary["remote_sample_agents"])
        self.assertEqual(batch.summary["remote_roles"], ["planner", "repair"])
        self.assertIn("constraint_critic", batch.summary["host_roles"])
        repaired = next(item for item in batch.records if item["attempts"] == 2)
        self.assertEqual(repaired["predicted"], 3.0)
        self.assertEqual(
            [item["role"] for item in repaired["agent_trace"]],
            ["remote_planner_agent", "remote_repair_agent"],
        )
        self.assertTrue(
            all(item.get("input_digest") for item in repaired["tool_trace"])
        )
        self.assertTrue(
            all(item.get("output_digest") for item in repaired["tool_trace"])
        )
        self.assertIn("tool_trace_digest", repaired)
        self.assertEqual(
            batch.summary["reason_code_counts"],
            {
                "initial_registered_route": 3,
                "repair_after_host_rejection": 1,
            },
        )
        self.assertEqual(
            batch.summary["recovered_by_failure_class"],
            {"constraint_rejected": 1},
        )

    def test_origin_shared_prompt_routes_with_lossless_context_references(self):
        gateway = _SampleDecisionGatewayFake(planner_tool="algorithm")
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            sample_planner_prompt_profile={
                "version": "origin_shared_context@1"
            },
        )
        requests = _sample_requests(3)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
                "tool_experience": [{"tool_id": "algorithm", "final_accept": 2}],
            }
        )

        outcomes = adapter.predict_samples(
            requests,
            tuple(dict(plan) for _request in requests),
            attempts=(1,) * len(requests),
        )

        self.assertTrue(all(outcome.error is None for outcome in outcomes))
        self.assertEqual(adapter.adapter_version, "12-origin-shared-context")
        self.assertEqual(len(gateway.calls), 1)
        call = gateway.calls[0]
        self.assertEqual(
            call["context"]["sample_planner_prompt_profile"],
            {"version": "origin_shared_context@1"},
        )
        self.assertEqual(
            call["context"]["evolution_context"]["tool_experience"],
            [{"tool_id": "algorithm", "final_accept": 2}],
        )
        expanded = expand_origin_shared_routing_payload(
            call["samples"], call["context"]["shared_sample_contexts"]
        )
        for request in requests:
            self.assertEqual(
                expanded[request.sample_id]["label_free_context"],
                request.label_free_context,
            )
            self.assertEqual(
                expanded[request.sample_id]["baseline"], request.baseline
            )
        encoded = json.dumps(call, sort_keys=True)
        self.assertNotIn("candidate:test", encoded)
        self.assertNotIn("dataset:test", encoded)

    def test_origin_shared_profile_changes_token_budget_checkpoint_identity(self):
        adapter = GatewaySampleCollaborationAdapter(
            _SampleDecisionGatewayFake(planner_tool="algorithm"),
            strategy_model_id="strategy-model",
            sample_planner_prompt_profile={
                "version": "origin_shared_context@1"
            },
            model_usage_callback=lambda _receipts: None,
            token_limit=1_000,
            token_reservation_per_wave=100,
        )

        self.assertEqual(
            adapter.adapter_version,
            "12-origin-shared-context-token-call-budget",
        )

    def test_opt_in_truncation_retry_escalates_once_before_split(self):
        gateway = _TruncationRetryGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            operation_max_tokens={
                "sample.planner": 3072,
                "sample.repair": 3072,
                "sample.critic": 2048,
            },
            sample_truncation_retry_policy={
                "version": "escalate_once@1",
                "max_tokens": 8192,
            },
        )
        requests = _sample_requests(3)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with patch.object(
            gateway_sample_adapter_module, "ModelGateway", _TruncationRetryGatewayFake
        ):
            outcomes = adapter.predict_samples(
                requests,
                tuple(dict(plan) for _request in requests),
                attempts=(1,) * len(requests),
            )

        self.assertTrue(all(item.error is None for item in outcomes))
        self.assertEqual(len(gateway.diagnostic_limits), 2)
        self.assertEqual(gateway.diagnostic_limits, [3072, 8192])
        self.assertEqual(adapter.adapter_version, "9-truncation-retry")

    def test_truncation_retry_is_disabled_for_legacy_adapter(self):
        gateway = _TruncationRetryGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            operation_max_tokens={
                "sample.planner": 3072,
                "sample.repair": 3072,
                "sample.critic": 2048,
            },
        )
        requests = _sample_requests(3)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with patch.object(
            gateway_sample_adapter_module, "ModelGateway", _TruncationRetryGatewayFake
        ):
            outcomes = adapter.predict_samples(
                requests,
                tuple(dict(plan) for _request in requests),
                attempts=(1,) * len(requests),
            )

        # Legacy runs retain adaptive splitting, but they do not replay the
        # same wave with an escalated output limit.  The contract we need to
        # preserve is the latter; split children may still recover normally.
        self.assertTrue(all(item.error is None for item in outcomes))
        self.assertEqual(len(gateway.diagnostic_limits), 3)
        self.assertEqual(gateway.diagnostic_limits, [3072, 3072, 3072])
        self.assertEqual(adapter.adapter_version, "9")

    def test_origin_shared_profile_does_not_require_shared_context_for_critic(self):
        gateway = _SampleDecisionGatewayFake(planner_tool="algorithm")
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            sample_planner_prompt_profile={
                "version": "origin_shared_context@1"
            },
        )
        requests = _sample_requests(1)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        outcomes = adapter.predict_samples(requests, (plan,), attempts=(1,))

        self.assertIsNone(outcomes[0].error)
        self.assertEqual([call["role"] for call in gateway.calls], ["planner", "critic"])
        critic_context = gateway.calls[1]["context"]
        self.assertNotIn("sample_planner_prompt_profile", critic_context)
        self.assertNotIn("shared_sample_contexts", critic_context)

    def test_gateway_causal_waves_split_adjacent_origins_and_block_sentinel_leak(self):
        sentinel = 987654321.125
        rows = _adjacent_rows()[:2]
        rows[0]["observed"] = sentinel
        rows[1]["baseline"] = sentinel
        rows[1]["label_free_context"]["history_window"] = [sentinel]
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
        )

        batch = self.execute(adapter, rows=rows, max_attempts=1)

        planner_calls = [call for call in gateway.calls if call["role"] == "planner"]
        self.assertEqual([len(call["samples"]) for call in planner_calls], [1, 1])
        first_sample_id = batch.records[0]["sample_id"]
        for call in gateway.calls:
            encoded = json.dumps(call["samples"], sort_keys=True)
            self.assertFalse(first_sample_id in encoded and str(sentinel) in encoded)

    def test_gateway_causal_wave_batches_same_origin_across_targets_and_horizons(self):
        rows = _rows()[:2]
        for row in rows:
            row["origin_timestamp"] = 0.5
            row["label_free_context"] = _causal_context(0.5, [0.5])
        rows[0]["timestamp"] = 1.5
        rows[1].update(
            {
                "target": "y",
                "horizon_hours": 3,
                "timestamp": 3.5,
            }
        )
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
        )

        batch = self.execute(adapter, rows=rows, max_attempts=1)

        self.assertEqual(batch.summary["succeeded_examples"], 2)
        for role in ("planner", "critic"):
            calls = [call for call in gateway.calls if call["role"] == role]
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0]["samples"]), 2)

    def test_gateway_unverified_context_falls_back_to_singleton_requests(self):
        rows = _rows()
        for row in rows:
            row["label_free_context"].pop("causal_provenance")
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
        )

        self.execute(adapter, rows=rows, max_attempts=1)

        planner_calls = [call for call in gateway.calls if call["role"] == "planner"]
        self.assertEqual([len(call["samples"]) for call in planner_calls], [1, 1, 1])

    def test_gateway_rejects_explicit_future_history_without_remote_call(self):
        row = _rows()[0]
        row["label_free_context"] = _causal_context(0, [1])
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
        )

        batch = self.execute(adapter, rows=[row], max_attempts=1)

        self.assertEqual(gateway.calls, [])
        self.assertEqual(batch.summary["failed_examples"], 1)
        self.assertEqual(
            batch.records[0]["failure"]["class"], "invalid_causal_provenance"
        )

    def test_gateway_repair_and_critic_calls_inherit_causal_wave_boundary(self):
        rows = _adjacent_rows()[:2]
        for row in rows:
            row.update({"predicted": 999.0, "baseline": 3.0})
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
        )

        batch = self.execute(adapter, rows=rows, max_attempts=2)

        self.assertEqual(batch.summary["succeeded_examples"], 2)
        self.assertEqual({record["attempts"] for record in batch.records}, {2})
        for role in ("planner", "repair", "critic"):
            calls = [call for call in gateway.calls if call["role"] == role]
            self.assertTrue(calls)
            self.assertTrue(all(len(call["samples"]) == 1 for call in calls))

    def test_gateway_planner_never_receives_precomputed_prediction(self):
        forecast_calls: list[str] = []

        def forecast_tool(request):
            forecast_calls.append(request.sample_id)
            return request.proposed_prediction

        gateway = _SampleDecisionGatewayFake(
            planner_tool="bounded-persistence-fallback"
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            forecast_tool=forecast_tool,
        )

        batch = self.execute(adapter, rows=[_rows()[0]])

        planner_sample = gateway.calls[0]["samples"][0]["sample"]
        self.assertNotIn("proposed_prediction", planner_sample)
        self.assertNotIn("predicted", planner_sample)
        self.assertEqual(forecast_calls, [])
        self.assertEqual(batch.records[0]["predicted"], _rows()[0]["baseline"])

    def test_projection_repairs_the_previous_selected_tool_output(self):
        candidate_calls: list[str] = []

        def candidate_tool(request):
            candidate_calls.append(request.sample_id)
            return 10.0

        def extreme_tool(request, context):
            del request, context
            return 999.0

        gateway = _SampleDecisionGatewayFake(
            planner_tool="extreme-tool",
            critic_repair_tool_once="bounded-projection-repair",
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            forecast_tool=candidate_tool,
            tools=(
                GatewaySampleTool(
                    tool_id="extreme-tool",
                    version="1",
                    handler=extreme_tool,
                ),
            ),
        )

        batch = self.execute(adapter, rows=[_rows()[0]], max_attempts=2)

        self.assertEqual(candidate_calls, [])
        self.assertEqual(batch.records[0]["predicted"], 100.0)
        self.assertEqual(
            [item for item in gateway.calls if item["role"] == "repair"], []
        )
        self.assertEqual(
            [
                item["selected_tool"]["tool_id"]
                for item in batch.records[0]["attempt_trace"]
            ],
            ["extreme-tool", "bounded-projection-repair"],
        )
        performance = {
            item["tool_id"]: item for item in batch.summary["tool_performance"]
        }
        self.assertEqual(performance["extreme-tool"]["critic_repair"], 1)
        self.assertEqual(
            performance["bounded-projection-repair"]["final_accept"], 1
        )
        self.assertEqual(
            performance["bounded-projection-repair"]["recovered"], 1
        )

    def test_gateway_adapter_batches_one_repair_wave_for_multiple_failures(self):
        rows = _rows()
        for row in rows[:2]:
            row.update({"predicted": 999.0, "baseline": 3.0})
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
        )

        batch = self.execute(adapter, rows=rows, max_attempts=2)

        repair_calls = [item for item in gateway.calls if item["role"] == "repair"]
        self.assertEqual(len(repair_calls), 1)
        self.assertEqual(len(repair_calls[0]["samples"]), 2)
        self.assertEqual(batch.summary["failed_examples"], 0)
        self.assertEqual(batch.summary["recovered_examples"], 2)

    def test_gateway_adapter_batches_planner_and_critic_without_labels(self):
        rows = _rows()
        for index, row in enumerate(rows):
            row["label_free_context"] = {
                **row["label_free_context"],
                "history_window_ref": f"window:{index}",
                "features": {"air_temperature": 20.0 + index},
            }
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
        )

        batch = self.execute(adapter, rows=rows, max_attempts=1)

        planner_calls = [item for item in gateway.calls if item["role"] == "planner"]
        critic_calls = [item for item in gateway.calls if item["role"] == "critic"]
        self.assertEqual(len(planner_calls), 1)
        self.assertEqual(len(planner_calls[0]["samples"]), 3)
        self.assertEqual(len(critic_calls), 1)
        self.assertEqual(len(critic_calls[0]["samples"]), 3)
        self.assertEqual(batch.summary["failed_examples"], 0)
        first_review = critic_calls[0]["samples"][0]
        review_sample = first_review["sample"]
        self.assertEqual(review_sample["baseline"], rows[0]["baseline"])
        self.assertEqual(
            review_sample["origin_timestamp"], rows[0]["origin_timestamp"]
        )
        self.assertEqual(review_sample["target_timestamp"], rows[0]["timestamp"])
        self.assertEqual(
            review_sample["label_free_context"], rows[0]["label_free_context"]
        )
        self.assertEqual(first_review["predicted"], rows[0]["predicted"])
        self.assertEqual(first_review["selected_tool"]["tool_id"], "algorithm")
        for record in batch.records:
            roles = {item["role"] for item in record["agent_trace"]}
            self.assertIn("remote_planner_agent", roles)
            self.assertIn("remote_critic_agent", roles)
        for action in batch.summary["action_catalog"]:
            roles = {item["role"] for item in action["agent_decisions"]}
            self.assertIn("remote_critic_agent", roles)
            self.assertIn("constraint_critic", roles)
        self.assertNotIn("observed", str(gateway.calls))
        _SampleDecisionGatewayFake._assert_label_hidden(critic_calls)

    def test_uncertain_or_failure_policy_skips_critic_for_confident_samples(self):
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            remote_critic_policy={
                "version": "uncertain_or_failure@1",
                "min_planner_confidence": 0.9,
            },
        )

        batch = self.execute(adapter, rows=_rows(), max_attempts=1)

        self.assertEqual(batch.summary["succeeded_examples"], 3)
        self.assertEqual(
            [call["role"] for call in gateway.calls], ["planner"]
        )
        for record in batch.records:
            roles = {item["role"] for item in record["agent_trace"]}
            self.assertIn("remote_planner_agent", roles)
            self.assertNotIn("remote_critic_agent", roles)
        self.assertTrue(
            all(
                "constraint_critic"
                in {item["role"] for item in action["agent_decisions"]}
                for action in batch.summary["action_catalog"]
            )
        )

    def test_always_policy_reviews_every_confident_sample(self):
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            remote_critic_policy={"version": "always@1"},
        )

        batch = self.execute(adapter, rows=_rows(), max_attempts=1)

        self.assertEqual(batch.summary["succeeded_examples"], 3)
        self.assertEqual(
            [call["role"] for call in gateway.calls], ["planner", "critic"]
        )
        self.assertEqual(
            adapter.plan_batch({})["routing_policy"],
            (
                "remote_planner_then_host_tool_then_remote_critic_then_"
                "host_constraint_critic;remote_critic_reviews_every_sample;"
                "remote_critic_selects_registered_repair_tool_or_terminate"
            ),
        )
        for record in batch.records:
            roles = {item["role"] for item in record["agent_trace"]}
            self.assertIn("remote_planner_agent", roles)
            self.assertIn("remote_critic_agent", roles)

    def test_uncertain_or_failure_policy_uses_compact_critic_payload(self):
        rows = _rows()
        for row in rows:
            row["label_free_context"] = {
                **row["label_free_context"],
                "feature_summary": [1, 2, 3],
            }
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            remote_critic_policy={
                "version": "uncertain_or_failure@1",
                "min_planner_confidence": 0.91,
            },
        )

        batch = self.execute(adapter, rows=rows, max_attempts=1)

        self.assertEqual(batch.summary["succeeded_examples"], 3)
        critic_calls = [call for call in gateway.calls if call["role"] == "critic"]
        self.assertEqual(len(critic_calls), 1)
        self.assertEqual(
            critic_calls[0]["context"],
            {
                "role": "critic",
                "critic_prompt_profile": "uncertain_or_failure_compact@1",
                "decision_policy": (
                    "select_accept_or_one_allowed_recovery_tool_using_only_"
                    "the_supplied_trigger_and_bounded_prediction_context"
                ),
            },
        )
        compact_sample = critic_calls[0]["samples"][0]
        self.assertEqual(
            set(compact_sample),
            {
                "sample_id",
                "target",
                "unit",
                "horizon_hours",
                "baseline",
                "minimum",
                "maximum",
                "predicted",
                "selected_tool_id",
                "planner_confidence",
                "trigger",
                "allowed_next_tool_ids",
            },
        )
        self.assertEqual(compact_sample["trigger"], "planner_low_confidence")
        self.assertEqual(compact_sample["planner_confidence"], 0.9)
        self.assertNotIn("sample", compact_sample)
        self.assertNotIn("label_free_context", compact_sample)
        self.assertNotIn("dataset_digest", compact_sample)
        self.assertNotIn("origin_timestamp", compact_sample)

    def test_uncertain_or_failure_policy_uses_compact_tool_failure_payload(self):
        def failed_tool(request, execution_context):
            del request, execution_context
            raise ArithmeticError("synthetic tool failure")

        gateway = _SampleDecisionGatewayFake(planner_tool="failed-tool")
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            remote_critic_policy={
                "version": "uncertain_or_failure@1",
                "min_planner_confidence": 0.0,
            },
            tools=(
                GatewaySampleTool(
                    tool_id="failed-tool",
                    version="1",
                    handler=failed_tool,
                ),
            ),
        )

        batch = self.execute(adapter, rows=[_rows()[0]], max_attempts=1)

        self.assertEqual(batch.summary["failed_examples"], 1)
        critic_call = next(call for call in gateway.calls if call["role"] == "critic")
        compact_sample = critic_call["samples"][0]
        self.assertEqual(compact_sample["trigger"], "tool_failure")
        self.assertEqual(
            set(compact_sample["tool_failure"]),
            {"failure_class", "host_retryable", "error_type", "failed_tool_id"},
        )
        self.assertNotIn("sample", compact_sample)
        self.assertNotIn("failure_feedback", compact_sample)
        self.assertNotIn("available_recovery_tools", compact_sample)

    def test_uncertain_or_failure_policy_requires_explicit_valid_freeze(self):
        gateway = _SampleDecisionGatewayFake()
        invalid_policies = (
            {},
            {"version": "always@1", "min_planner_confidence": 0.7},
            {"version": "uncertain_or_failure@2", "min_planner_confidence": 0.7},
            {"version": "uncertain_or_failure@1", "min_planner_confidence": 1.1},
        )
        for policy in invalid_policies:
            with self.subTest(policy=policy):
                with self.assertRaises((TypeError, ValueError)):
                    GatewaySampleCollaborationAdapter(
                        gateway,
                        strategy_model_id="strategy-model",
                        review_model_id="review-model",
                        remote_review_enabled=True,
                        remote_critic_policy=policy,
                    )

        with self.assertRaises(ValueError):
            GatewaySampleCollaborationAdapter(
                gateway,
                strategy_model_id="strategy-model",
                remote_critic_policy={
                    "version": "uncertain_or_failure@1",
                    "min_planner_confidence": 0.7,
                },
            )

    def test_remote_critic_cannot_replace_a_valid_scientific_prediction(self):
        row = _rows()[0]
        row.update({"predicted": 10.0, "baseline": 2.0})
        gateway = _SampleDecisionGatewayFake(
            critic_repair_tool_once="bounded-projection-repair"
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
        )

        batch = self.execute(adapter, rows=[row], max_attempts=2)

        self.assertEqual(batch.summary["failed_examples"], 0)
        self.assertEqual(batch.records[0]["attempts"], 1)
        self.assertEqual(batch.records[0]["predicted"], 10.0)
        self.assertEqual(batch.records[0].get("failure_history", []), [])
        self.assertNotIn(
            "bounded-projection-repair",
            [item["tool_id"] for item in batch.records[0]["tool_trace"]],
        )
        self.assertEqual(
            [item for item in gateway.calls if item["role"] == "repair"], []
        )
        self.assertNotIn(
            "remote_repair_agent",
            [item["role"] for item in batch.records[0]["agent_trace"]],
        )
        self.assertEqual(
            batch.summary["critic_outcome_counts"],
            {"accepted": 1, "rejected": 1},
        )

    def test_gateway_adapter_rejects_outcome_fields_before_remote_call(self):
        row = _rows()[0]
        row["label_free_context"] = {
            "features": {"observed": row["observed"]}
        }
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
        )

        batch = self.execute(adapter, rows=[row])

        self.assertEqual(gateway.calls, [])
        self.assertEqual(batch.summary["input_failures"], 1)
        self.assertEqual(batch.summary["failed_examples"], 1)
        self.assertEqual(batch.records[0]["failure"]["class"], "invalid_sample_input")

    def test_gateway_adapter_rejects_folded_outcome_field_aliases_locally(self):
        for field_name in ("actualValue", "Ground.Truth"):
            with self.subTest(field_name=field_name):
                row = _rows()[0]
                row["label_free_context"] = {
                    "features": {"nested": {field_name: row["observed"]}}
                }
                gateway = _SampleDecisionGatewayFake()
                adapter = GatewaySampleCollaborationAdapter(
                    gateway,
                    strategy_model_id="strategy-model",
                )

                batch = self.execute(adapter, rows=[row])

                self.assertEqual(gateway.calls, [])
                self.assertEqual(batch.summary["input_failures"], 1)
                self.assertEqual(
                    batch.records[0]["failure"]["class"], "invalid_sample_input"
                )

    def test_gateway_tool_failure_is_isolated_and_keeps_digest_trace(self):
        rows = _rows()
        for index, row in enumerate(rows):
            row["label_free_context"] = {
                **row["label_free_context"],
                "features": {"force_timeout": index == 1}
            }

        def context_tool(request, execution_context):
            self.assertEqual(execution_context["attempt"], 1)
            if request.label_free_context["features"]["force_timeout"]:
                raise TimeoutError("synthetic queued tool timed out")
            return request.proposed_prediction

        gateway = _SampleDecisionGatewayFake(planner_tool="context-tool")
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            tools=(
                GatewaySampleTool(
                    tool_id="context-tool",
                    version="1",
                    handler=context_tool,
                ),
            ),
        )

        batch = self.execute(adapter, rows=rows, max_attempts=1)

        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(len(gateway.calls[0]["samples"]), 3)
        self.assertEqual(batch.summary["succeeded_examples"], 2)
        self.assertEqual(batch.summary["failed_examples"], 1)
        self.assertTrue(batch.summary["coverage_pass"])
        failed = next(item for item in batch.records if item["status"] == "failed")
        self.assertEqual(failed["failure"]["class"], "timeout")
        self.assertEqual(failed["tool_trace"][0]["status"], "failed")
        self.assertIn("input_digest", failed["tool_trace"][0])
        self.assertNotIn("output_digest", failed["tool_trace"][0])

    def test_gateway_batches_failed_tool_review_and_critic_drives_repair(self):
        rows = _rows()
        for index, row in enumerate(rows):
            row["label_free_context"] = {
                **row["label_free_context"],
                "features": {"force_numerical_failure": index < 2}
            }

        def context_tool(request, execution_context):
            self.assertEqual(execution_context["attempt"], 1)
            if request.label_free_context["features"]["force_numerical_failure"]:
                raise ArithmeticError("synthetic numerical failure")
            return request.proposed_prediction

        gateway = _SampleDecisionGatewayFake(
            planner_tool="context-tool",
            failure_critic_tool="bounded-persistence-fallback",
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            tools=(
                GatewaySampleTool(
                    tool_id="context-tool",
                    version="1",
                    handler=context_tool,
                ),
            ),
        )

        batch = self.execute(adapter, rows=rows, max_attempts=2)

        planner_calls = [item for item in gateway.calls if item["role"] == "planner"]
        repair_calls = [item for item in gateway.calls if item["role"] == "repair"]
        critic_calls = [item for item in gateway.calls if item["role"] == "critic"]
        failure_reviews = [
            item
            for item in critic_calls
            if "tool_failure" in item["samples"][0]
        ]
        self.assertEqual(len(planner_calls), 1)
        self.assertEqual(len(planner_calls[0]["samples"]), 3)
        self.assertEqual(len(failure_reviews), 1)
        self.assertEqual(len(failure_reviews[0]["samples"]), 2)
        self.assertEqual(repair_calls, [])
        self.assertEqual(batch.summary["succeeded_examples"], 3)
        self.assertEqual(batch.summary["failed_examples"], 0)
        self.assertEqual(batch.summary["recovered_examples"], 2)

        reviewed = failure_reviews[0]["samples"][0]["tool_failure"]
        self.assertEqual(reviewed["failure_class"], "numerical")
        self.assertFalse(reviewed["host_retryable"])
        self.assertEqual(reviewed["selected_tool"]["status"], "failed")
        self.assertNotIn(
            "context-tool",
            {item["tool_id"] for item in failure_reviews[0]["available_tools"]},
        )
        repaired = [item for item in batch.records if item["attempts"] == 2]
        self.assertEqual(len(repaired), 2)
        for record in repaired:
            self.assertEqual(record["failure_history"][0]["failure_class"], "numerical")
            self.assertEqual(
                record["failure_history"][0]["requested_tool_id"],
                "bounded-persistence-fallback",
            )
            self.assertEqual(
                len(record["failure_history"][0]["critic_response_digests"]),
                1,
            )
            self.assertEqual(
                len(record["failure_history"][0]["critic_response_digests"][0]),
                64,
            )
            self.assertEqual(record["tool_trace"][0]["tool_id"], "context-tool")
            self.assertEqual(record["tool_trace"][0]["status"], "failed")
            self.assertIn(
                "bounded-persistence-fallback",
                [item["tool_id"] for item in record["tool_trace"]],
            )
            roles = [item["role"] for item in record["agent_trace"]]
            self.assertIn("remote_critic_agent", roles)
            self.assertIn("critic_repair_router", roles)
        self.assertNotIn("observed", str(gateway.calls))
        _SampleDecisionGatewayFake._assert_label_hidden(gateway.calls)

    def test_gateway_failure_critic_can_terminate_without_repair_wave(self):
        row = _rows()[0]

        def failed_tool(request, execution_context):
            del request, execution_context
            raise ArithmeticError("synthetic terminal failure")

        gateway = _SampleDecisionGatewayFake(
            planner_tool="failed-tool",
            failure_critic_tool="terminate",
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            tools=(
                GatewaySampleTool(
                    tool_id="failed-tool",
                    version="1",
                    handler=failed_tool,
                ),
            ),
        )

        batch = self.execute(adapter, rows=[row], max_attempts=3)

        self.assertEqual(batch.summary["succeeded_examples"], 0)
        self.assertEqual(batch.summary["failed_examples"], 1)
        self.assertEqual(batch.records[0]["attempts"], 1)
        self.assertFalse(batch.records[0]["failure"]["retryable"])
        self.assertNotIn("requested_tool_id", batch.records[0]["failure_history"][0])
        self.assertEqual(
            len(batch.records[0]["failure_history"][0]["critic_response_digests"]),
            1,
        )
        self.assertEqual(
            [item["role"] for item in gateway.calls],
            ["planner", "critic"],
        )
        critic_call = gateway.calls[1]
        self.assertEqual(len(critic_call["samples"]), 1)
        self.assertIn(
            "terminate",
            {item["tool_id"] for item in critic_call["available_tools"]},
        )
        self.assertIn(
            "remote_critic_agent",
            [item["role"] for item in batch.records[0]["agent_trace"]],
        )
        self.assertNotIn("observed", str(gateway.calls))

    def test_success_critic_contract_failure_does_not_replace_prediction(self):
        gateway = _FailFirstCriticGatewayFake(
            GatewayResponseError(
                "sample decision response selects an unavailable tool",
                error_code="sample_decision_tool_invalid",
                split_eligible=True,
            )
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
        )

        batch = self.execute(adapter, rows=[_rows()[0]], max_attempts=2)

        self.assertEqual(batch.summary["succeeded_examples"], 1)
        self.assertEqual(batch.records[0]["attempts"], 1)
        self.assertEqual(batch.records[0].get("failure_history", []), [])
        self.assertEqual(
            [call["role"] for call in gateway.calls],
            ["planner", "critic"],
        )
        self.assertEqual(
            batch.summary["critic_outcome_counts"],
            {"accepted": 1, "rejected": 1},
        )

    def test_tool_failure_critic_timeout_preserves_failure_and_repairs(self):
        def failed_tool(request, execution_context):
            del request, execution_context
            raise ArithmeticError("synthetic numerical failure")

        gateway = _FailFirstCriticGatewayFake(
            TimeoutError("queued critic timed out"),
            planner_tool="failed-tool",
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            tools=(
                GatewaySampleTool(
                    tool_id="failed-tool",
                    version="1",
                    handler=failed_tool,
                ),
            ),
        )

        batch = self.execute(adapter, rows=[_rows()[0]], max_attempts=2)

        self.assertEqual(batch.summary["succeeded_examples"], 1)
        self.assertEqual(batch.records[0]["attempts"], 2)
        self.assertEqual(
            batch.records[0]["failure_history"][0]["failure_class"], "numerical"
        )
        self.assertEqual(
            [call["role"] for call in gateway.calls],
            ["planner", "critic", "repair", "critic"],
        )

    def test_retryable_success_critic_gateway_failure_is_raised_for_resume(self):
        gateway = _FailFirstCriticGatewayFake(
            GatewayResponseError(
                "queued critic request exhausted",
                retryable=True,
                attempts=4,
            )
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
        )

        with self.assertRaises(GatewayResponseError) as captured:
            self.execute(adapter, rows=[_rows()[0]], max_attempts=3)

        self.assertIs(captured.exception, gateway.error)
        self.assertEqual(
            [call["role"] for call in gateway.calls], ["planner", "critic"]
        )

    def test_retryable_tool_failure_critic_gateway_error_is_raised_for_resume(self):
        def failed_tool(request, execution_context):
            del request, execution_context
            raise ArithmeticError("synthetic numerical failure")

        gateway = _FailFirstCriticGatewayFake(
            GatewayResponseError(
                "queued critic request exhausted",
                retryable=True,
                attempts=4,
            ),
            planner_tool="failed-tool",
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            tools=(
                GatewaySampleTool(
                    tool_id="failed-tool",
                    version="1",
                    handler=failed_tool,
                ),
            ),
        )

        with self.assertRaises(GatewayResponseError) as captured:
            self.execute(adapter, rows=[_rows()[0]], max_attempts=3)

        self.assertIs(captured.exception, gateway.error)
        self.assertEqual(
            [call["role"] for call in gateway.calls], ["planner", "critic"]
        )

    def test_critic_authentication_failure_remains_terminal(self):
        gateway = _FailFirstCriticGatewayFake(
            GatewayResponseError("authentication rejected", status_code=401)
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
        )

        batch = self.execute(adapter, rows=[_rows()[0]], max_attempts=3)

        self.assertEqual(batch.summary["failed_examples"], 1)
        self.assertEqual(batch.records[0]["attempts"], 1)
        self.assertFalse(batch.records[0]["failure"]["retryable"])
        self.assertEqual(
            [call["role"] for call in gateway.calls], ["planner", "critic"]
        )

    def test_gateway_adapter_reports_aggregate_microbatch_progress(self):
        progress = []
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=2,
            progress_callback=lambda item: progress.append(dict(item)),
        )

        batch = self.execute(adapter, rows=_rows(), max_attempts=1)

        self.assertEqual(batch.summary["succeeded_examples"], 3)
        # Concurrent schedules can finish in one coordinator wakeup.  In that
        # case the adapter deliberately publishes one final durable aggregate
        # instead of an obsolete intermediate heartbeat.
        self.assertIn(len(progress), (1, 2))
        completed_samples = [item["completed_samples"] for item in progress]
        self.assertIn(completed_samples[0], (1, 2, 3))
        self.assertEqual(completed_samples[-1], 3)
        self.assertEqual(completed_samples, sorted(completed_samples))
        self.assertTrue(
            all(item["progress_kind"] == "completed_batch" for item in progress)
        )
        self.assertTrue(all(item["total_samples"] == 3 for item in progress))
        batch_indices = [item["batch_index"] for item in progress]
        self.assertEqual(batch_indices[-1], 2)
        self.assertEqual(batch_indices, sorted(batch_indices))
        self.assertTrue(all(item["batch_count"] == 2 for item in progress))
        self.assertEqual(sum(item["batch_size"] for item in progress), 3)
        self.assertEqual(progress[-1]["succeeded_samples"], 3)
        self.assertEqual(progress[-1]["failed_samples"], 0)
        self.assertNotIn("sample_id", str(progress))
        self.assertNotIn("observed", str(progress))

    def test_gateway_stops_when_fixed_task_coverage_is_unreachable(self):
        rows = []
        for horizon in range(1, 10):
            for _task_index in range(267):
                sample_index = len(rows)
                origin_timestamp = sample_index // 128
                rows.append(
                    {
                        "partition": f"training_feedback:{sample_index}",
                        "target": "x",
                        "horizon_hours": horizon,
                        "origin_timestamp": origin_timestamp,
                        "timestamp": origin_timestamp + horizon,
                        "observed": 0.0,
                        "predicted": 0.0,
                        "baseline": 1.0,
                        "label_free_context": _causal_context(
                            origin_timestamp, [origin_timestamp]
                        ),
                    }
                )
        gateway = _FailPartitionGatewayFake("training_feedback:0")
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=128,
        )

        batch = self.execute(
            adapter,
            rows=rows,
            max_attempts=3,
            minimum_coverage=0.8,
            minimum_task_coverage=0.8,
        )

        self.assertEqual(len(rows), 2403)
        self.assertEqual(
            [len(call["samples"]) for call in gateway.calls],
            [128, 128, 128, 128],
        )
        self.assertTrue(batch.summary["coverage_early_stopped"])
        self.assertEqual(batch.summary["early_stop_reason"], "coverage_unreachable")
        self.assertEqual(batch.summary["succeeded_examples"], 384)
        self.assertEqual(batch.summary["failed_examples"], 2403 - 384)
        self.assertFalse(batch.summary["coverage_pass"])
        self.assertEqual(len(batch.records), 2403)
        self.assertEqual(len(batch.scoring_rows), 2403)
        unexecuted = [
            record
            for record in batch.records
            if record.get("failure", {}).get("class")
            == COVERAGE_UNREACHABLE_NOT_EXECUTED_FAILURE
        ]
        self.assertEqual(len(unexecuted), 2403 - 512)
        self.assertTrue(all(record["attempts"] == 0 for record in unexecuted))
        self.assertEqual(
            sum(
                row["scoring_fallback"] == "failure_non_improvement_penalty"
                for row in batch.scoring_rows
            ),
            2403 - 384,
        )
        self.assertEqual(
            sum(row["scoring_fallback"] is None for row in batch.scoring_rows),
            384,
        )
        first_task = next(
            task
            for task in batch.summary["tasks"]
            if task["target"] == "x" and task["horizon_hours"] == 1
        )
        self.assertEqual(first_task["attempted_examples"], 267)
        self.assertEqual(first_task["succeeded_examples"], 139)
        self.assertEqual(first_task["failed_examples"], 128)
        self.assertEqual(first_task["coverage"], 139 / 267)

    def test_coverage_refill_waits_for_unresolved_breaking_failure(self):
        gateway = _CoverageSpeculativeRefillGatewayFake()
        progress = []

        def publish_progress(item):
            progress.append(dict(item))
            if item["completed_samples"] == 1:
                gateway.release_failure.set()

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=2,
            progress_callback=publish_progress,
        )

        batch = self.execute(
            adapter,
            rows=_many_rows(4),
            max_attempts=1,
            minimum_coverage=0.8,
            minimum_task_coverage=0.8,
        )

        self.assertTrue(progress)
        self.assertEqual(progress[0]["completed_samples"], 1)
        self.assertEqual(gateway.unexpected_submissions, [])
        self.assertEqual(
            [
                call["samples"][0]["sample"]["partition"]
                for call in gateway.calls
            ],
            ["training_feedback:1", "training_feedback:0"],
        )
        self.assertTrue(batch.summary["coverage_early_stopped"])
        self.assertEqual(batch.summary["succeeded_examples"], 1)
        self.assertEqual(batch.summary["failed_examples"], 3)

    def test_coverage_refill_preserves_safe_sliding_window(self):
        gateway = _SlidingWindowGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=2,
        )

        batch = self.execute(
            adapter,
            rows=_many_rows(10),
            max_attempts=1,
            minimum_coverage=0.8,
            minimum_task_coverage=0.8,
        )

        self.assertTrue(gateway.third_started.is_set())
        self.assertEqual(batch.summary["succeeded_examples"], 10)
        self.assertTrue(batch.summary["coverage_pass"])

    def test_coverage_refill_checks_each_large_chunk_across_open_slots(self):
        gateway = _CoverageMultipleSlotRefillGatewayFake()
        progress = []

        def publish_progress(item):
            progress.append(dict(item))
            if item["completed_samples"] >= 9:
                gateway.release_pending.set()

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=3,
            sample_concurrency=4,
            progress_callback=publish_progress,
        )

        batch = self.execute(
            adapter,
            rows=_many_rows(30),
            max_attempts=1,
            minimum_coverage=0.8,
            minimum_task_coverage=0.8,
        )

        self.assertTrue(gateway.release_pending.is_set())
        self.assertEqual(gateway.premature_submissions, [])
        self.assertEqual(batch.summary["succeeded_examples"], 27)
        self.assertEqual(batch.summary["failed_examples"], 3)
        self.assertTrue(batch.summary["coverage_pass"])
        self.assertFalse(batch.summary["coverage_early_stopped"])

    def test_gateway_continues_when_maximum_reachable_equals_threshold(self):
        rows = _many_rows(10)
        gateway = _FailFirstGatewayFake()
        persisted = []
        progress = []
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=2,
            progress_callback=lambda item: progress.append(dict(item)),
        )

        batch = self.execute(
            adapter,
            rows=rows,
            max_attempts=1,
            minimum_coverage=0.8,
            minimum_task_coverage=0.8,
            result_callback=lambda values: persisted.extend(
                dict(value) for value in values
            ),
        )

        self.assertEqual(len(gateway.calls), 5)
        self.assertFalse(batch.summary["coverage_early_stopped"])
        self.assertIsNone(batch.summary["early_stop_reason"])
        self.assertEqual(batch.summary["succeeded_examples"], 8)
        self.assertEqual(batch.summary["failed_examples"], 2)
        self.assertEqual(batch.summary["coverage"], 0.8)
        self.assertTrue(batch.summary["coverage_pass"])
        self.assertEqual(len(batch.records), 10)
        self.assertEqual(len(batch.scoring_rows), 10)
        durable = build_sample_results("candidate:test", persisted)
        scientific = build_sample_results("candidate:test", batch.scoring_rows)
        self.assertEqual(
            sorted(durable, key=lambda row: row["sample_id"]),
            sorted(scientific, key=lambda row: row["sample_id"]),
        )
        self.assertEqual(sum(row["status"] == "failed" for row in durable), 2)
        planner_progress = [item for item in progress if item["role"] == "planner"]
        self.assertEqual(planner_progress[-1]["completed_samples"], len(durable))

    def test_gateway_keeps_configured_concurrency_without_coverage_stop(self):
        gateway = _ConcurrencyTrackingGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=3,
        )
        context = {
            "candidate_id": "candidate:test",
            "dataset_digest": "dataset:test",
            "algorithm_id": "algorithm",
            "algorithm_version": "1",
        }
        plan = adapter.plan_batch(context)
        requests = tuple(
            SamplePredictionRequest(
                sample_id=f"sample:{index}",
                candidate_id="candidate:test",
                dataset_digest="dataset:test",
                partition=f"training_feedback:{index}",
                target="x",
                unit="u",
                horizon_hours=1,
                origin_timestamp=0,
                target_timestamp=1,
                baseline=1.0,
                proposed_prediction=1.0,
                minimum=-100.0,
                maximum=100.0,
                algorithm_id="algorithm",
                algorithm_version="1",
                label_free_context=_causal_context(0, [0]),
            )
            for index in range(6)
        )

        outcomes = adapter.predict_samples(
            requests,
            tuple(dict(plan) for _request in requests),
            attempts=(1,) * len(requests),
        )

        self.assertTrue(all(outcome.error is None for outcome in outcomes))
        self.assertGreaterEqual(gateway.max_inflight, 2)
        self.assertLessEqual(gateway.max_inflight, 3)

    def test_gateway_keeps_configured_concurrency_with_coverage_stop(self):
        gateway = _ConcurrencyTrackingGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=3,
        )

        batch = self.execute(
            adapter,
            rows=_many_rows(6),
            max_attempts=1,
            minimum_coverage=0.8,
            minimum_task_coverage=0.8,
        )

        self.assertEqual(batch.summary["succeeded_examples"], 6)
        self.assertTrue(batch.summary["coverage_pass"])
        self.assertGreaterEqual(gateway.max_inflight, 2)
        self.assertLessEqual(gateway.max_inflight, 3)

    def test_gateway_completed_progress_reflects_refilled_sliding_window(self):
        gateway = _SlidingWindowGatewayFake()
        progress = []
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=2,
            progress_callback=lambda item: progress.append(dict(item)),
        )
        requests = _sample_requests(3)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        outcomes = adapter.predict_samples(
            requests,
            tuple(dict(plan) for _request in requests),
            attempts=(1, 1, 1),
        )

        self.assertTrue(gateway.third_started.is_set())
        self.assertEqual(set(gateway.started), {0, 1, 2})
        self.assertTrue(all(outcome.error is None for outcome in outcomes))
        first_completion = next(
            item
            for item in progress
            if item["progress_kind"] == "completed_batch"
            and item["completed_samples"] == 1
        )
        self.assertEqual(first_completion["in_flight_batches"], 2)
        self.assertEqual(first_completion["queued_batches"], 0)

    def test_gateway_coverage_stop_drains_inflight_without_submitting_more(self):
        gateway = _CoverageDrainGatewayFake()
        progress = []

        def publish_progress(item):
            progress.append(dict(item))
            gateway.release_second.set()

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=2,
            progress_callback=publish_progress,
        )

        batch = self.execute(
            adapter,
            rows=_many_rows(4),
            max_attempts=1,
            minimum_coverage=0.8,
            minimum_task_coverage=0.8,
        )

        self.assertTrue(gateway.second_drained)
        self.assertEqual(gateway.unexpected_submissions, [])
        self.assertEqual(
            [
                call["samples"][0]["sample"]["partition"]
                for call in gateway.calls
            ],
            ["training_feedback:0", "training_feedback:1"],
        )
        self.assertEqual(len(progress), 2)
        self.assertEqual(progress[-1]["completed_samples"], 2)
        self.assertEqual(progress[-1]["succeeded_samples"], 1)
        self.assertEqual(progress[-1]["failed_samples"], 1)
        self.assertTrue(batch.summary["coverage_early_stopped"])
        self.assertEqual(batch.summary["succeeded_examples"], 1)
        self.assertEqual(batch.summary["failed_examples"], 3)
        self.assertEqual(
            sum(
                record.get("failure", {}).get("class")
                == COVERAGE_UNREACHABLE_NOT_EXECUTED_FAILURE
                for record in batch.records
            ),
            2,
        )

    def test_retryable_gateway_error_drains_and_persists_sibling_before_raising(self):
        gateway = _RetryableGatewayDrainFake()
        progress = []
        persisted = []
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=2,
            progress_callback=lambda item: progress.append(dict(item)),
        )
        release_timer = threading.Timer(0.03, gateway.release_second.set)
        release_timer.start()
        try:
            with self.assertRaises(GatewayResponseError) as captured:
                self.execute(
                    adapter,
                    rows=_many_rows(2),
                    max_attempts=1,
                    result_callback=lambda rows: persisted.extend(
                        dict(row) for row in rows
                    ),
                )
        finally:
            release_timer.cancel()

        self.assertIs(captured.exception, gateway.error)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0]["sample_execution_status"], "succeeded")
        self.assertEqual(
            [item["completed_samples"] for item in progress],
            [1],
        )

    def test_retryable_gateway_error_persists_partial_schedule_outcomes(self):
        gateway = _SampleDecisionGatewayFake()
        error = GatewayResponseError(
            "policy model request failed: retry budget exhausted",
            retryable=True,
            attempts=4,
        )
        progress: list[dict] = []
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=2,
            sample_concurrency=1,
            progress_callback=lambda item: progress.append(dict(item)),
        )
        published: list[str] = []

        def partial_route(requests, _plans, _attempts, indices, *, outcomes, **_kwargs):
            index = indices[0]
            outcomes[index] = sample_execution_module.SamplePredictionOutcome(
                sample_id=requests[index].sample_id,
                result={"predicted": requests[index].proposed_prediction},
            )
            raise error

        adapter.set_outcome_callback(
            lambda requests, _outcomes, _attempts: published.extend(
                request.sample_id for request in requests
            )
        )
        requests = _sample_requests(2)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )
        with patch.object(adapter, "_route_chunk", partial_route):
            with self.assertRaises(GatewayResponseError) as captured:
                adapter.predict_samples(
                    requests,
                    (dict(plan), dict(plan)),
                    attempts=(1, 1),
                )
        self.assertIs(captured.exception, error)
        self.assertEqual(published, ["sample:0"])
        self.assertEqual(
            [item["completed_samples"] for item in progress],
            [1],
        )

    def test_same_wakeup_retryable_error_does_not_refill_sliding_window(self):
        class ImmediateFuture:
            def __init__(self, value):
                self._value = value

            def add_done_callback(self, callback):
                callback(self)

            def done(self):
                return True

            def result(self):
                return self._value

        class ImmediateExecutor:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def __enter__(self):
                return self

            def __exit__(self, *args):
                del args

            def submit(self, function, *args):
                return ImmediateFuture(function(*args))

        class SameWakeupGateway(_SampleDecisionGatewayFake):
            def __init__(self):
                super().__init__()
                self.error = GatewayResponseError(
                    "policy model request failed: retry budget exhausted",
                    retryable=True,
                    attempts=4,
                )

            def sample_decide(self, model_id, **kwargs):
                response = super().sample_decide(model_id, **kwargs)
                partition = kwargs["samples"][0]["sample"]["partition"]
                if partition == "training_feedback:1":
                    raise self.error
                return response

        gateway = SameWakeupGateway()
        published_sample_ids = []
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=2,
        )
        adapter.set_outcome_callback(
            lambda requests, _outcomes, _attempts: published_sample_ids.extend(
                request.sample_id for request in requests
            )
        )
        requests = _sample_requests(3)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with patch.object(
            gateway_sample_adapter_module,
            "ThreadPoolExecutor",
            ImmediateExecutor,
        ):
            with self.assertRaises(GatewayResponseError) as captured:
                adapter.predict_samples(
                    requests,
                    tuple(dict(plan) for _request in requests),
                    attempts=(1, 1, 1),
                )

        self.assertIs(captured.exception, gateway.error)
        self.assertEqual(
            [
                call["samples"][0]["sample"]["partition"]
                for call in gateway.calls
            ],
            ["training_feedback:0", "training_feedback:1"],
        )
        self.assertEqual(published_sample_ids, ["sample:0"])

    def test_same_wakeup_coalesces_completed_progress_to_latest_durable_state(self):
        class ImmediateFuture:
            def __init__(self, value):
                self._value = value

            def add_done_callback(self, callback):
                callback(self)

            def done(self):
                return True

            def result(self):
                return self._value

        class ImmediateExecutor:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def __enter__(self):
                return self

            def __exit__(self, *args):
                del args

            def submit(self, function, *args):
                return ImmediateFuture(function(*args))

        events = []
        progress = []

        def publish_outcomes(requests, _outcomes, _attempts):
            sample_ids = [request.sample_id for request in requests]
            events.append(("results", sample_ids))
            return {sample_id: "succeeded" for sample_id in sample_ids}

        def publish_progress(item):
            copied = dict(item)
            progress.append(copied)
            events.append(("progress", copied["completed_samples"]))

        adapter = GatewaySampleCollaborationAdapter(
            _SampleDecisionGatewayFake(),
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=2,
            progress_callback=publish_progress,
        )
        adapter.set_outcome_callback(publish_outcomes)
        requests = _sample_requests(2)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with patch.object(
            gateway_sample_adapter_module,
            "ThreadPoolExecutor",
            ImmediateExecutor,
        ):
            outcomes = adapter.predict_samples(
                requests,
                tuple(dict(plan) for _request in requests),
                attempts=(1, 1),
            )

        self.assertTrue(all(outcome.error is None for outcome in outcomes))
        self.assertEqual(
            events,
            [
                ("results", ["sample:0"]),
                ("results", ["sample:1"]),
                ("progress", 2),
            ],
        )
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["progress_kind"], "completed_batch")
        self.assertEqual(progress[0]["completed_samples"], 2)
        self.assertEqual(progress[0]["succeeded_samples"], 2)
        self.assertEqual(progress[0]["total_samples"], 2)
        self.assertEqual(progress[0]["batch_index"], 2)
        self.assertEqual(progress[0]["batch_count"], 2)
        self.assertEqual(progress[0]["batch_size"], 2)

    def test_gateway_publishes_planner_and_critic_usage_receipts_per_call(self):
        receipt_batches = []
        gateway = _DiagnosticSampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            model_usage_callback=lambda receipts: receipt_batches.append(receipts),
        )

        batch = self.execute(adapter, rows=[_rows()[0]], max_attempts=1)

        self.assertEqual(batch.summary["succeeded_examples"], 1)
        self.assertEqual([len(receipts) for receipts in receipt_batches], [1, 1])
        receipts = [batch[0] for batch in receipt_batches]
        self.assertEqual(
            [receipt["role"] for receipt in receipts],
            ["planner", "critic"],
        )
        self.assertEqual(
            [receipt["model_id"] for receipt in receipts],
            ["strategy-model", "review-model"],
        )
        expected_fields = {
            "call_id",
            "logical_call_digest",
            "role",
            "model_id",
            "outcome",
            "usage_reported",
            "http_attempts",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }
        for receipt in receipts:
            self.assertEqual(set(receipt), expected_fields)
            self.assertRegex(receipt["call_id"], r"^[0-9a-f]{32}$")
            self.assertRegex(receipt["logical_call_digest"], r"^[0-9a-f]{64}$")
            self.assertEqual(receipt["outcome"], "succeeded")
            self.assertTrue(receipt["usage_reported"])
            self.assertEqual(receipt["http_attempts"], 2)
            self.assertEqual(
                receipt["total_tokens"],
                receipt["prompt_tokens"] + receipt["completion_tokens"],
            )

    def test_planner_usage_is_published_while_critic_is_blocked(self):
        receipt_batches = []
        planner_persisted = threading.Event()
        gateway = _BlockingDiagnosticCriticGatewayFake()

        def persist_usage(receipts):
            copied = tuple(dict(receipt) for receipt in receipts)
            receipt_batches.append(copied)
            if copied[0]["role"] == "planner":
                planner_persisted.set()

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            model_usage_callback=persist_usage,
        )
        requests = _sample_requests(1)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )
        predictions = []
        execution_errors = []

        def execute_prediction():
            try:
                predictions.extend(
                    adapter.predict_samples(requests, (plan,), attempts=(1,))
                )
            except BaseException as exc:  # noqa: BLE001 - surface thread failures
                execution_errors.append(exc)

        execution_thread = threading.Thread(target=execute_prediction)
        execution_thread.start()
        try:
            self.assertTrue(gateway.critic_started.wait(timeout=1))
            self.assertTrue(planner_persisted.wait(timeout=1))
            self.assertTrue(execution_thread.is_alive())
            self.assertEqual(
                [batch[0]["role"] for batch in receipt_batches],
                ["planner"],
            )
        finally:
            gateway.release_critic.set()
            execution_thread.join(timeout=2)

        self.assertFalse(execution_thread.is_alive())
        self.assertEqual(execution_errors, [])
        self.assertEqual(len(predictions), 1)
        self.assertIsNone(predictions[0].error)
        self.assertEqual([len(batch) for batch in receipt_batches], [1, 1])
        self.assertEqual(
            [batch[0]["role"] for batch in receipt_batches],
            ["planner", "critic"],
        )

    def test_concurrent_worker_usage_has_one_coordinator_writer(self):
        receipt_batches = []
        callback_threads = []
        coordinator_thread = threading.get_ident()

        def persist_usage(receipts):
            receipt_batches.append(tuple(dict(receipt) for receipt in receipts))
            callback_threads.append(threading.get_ident())

        adapter = GatewaySampleCollaborationAdapter(
            _DiagnosticSampleDecisionGatewayFake(),
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=3,
            model_usage_callback=persist_usage,
        )
        requests = _sample_requests(6)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        outcomes = adapter.predict_samples(
            requests,
            tuple(dict(plan) for _request in requests),
            attempts=(1,) * len(requests),
        )

        self.assertTrue(all(outcome.error is None for outcome in outcomes))
        self.assertEqual([len(batch) for batch in receipt_batches], [1] * 6)
        self.assertEqual(set(callback_threads), {coordinator_thread})
        self.assertEqual(
            len({batch[0]["call_id"] for batch in receipt_batches}),
            6,
        )

    def test_token_budget_reserves_inflight_waves_and_stops_refill(self):
        gateway = _DiagnosticSampleDecisionGatewayFake()
        receipt_batches = []
        published_sample_ids = []
        durable_tokens = 0

        def persist_usage(receipts):
            nonlocal durable_tokens
            copied = tuple(dict(receipt) for receipt in receipts)
            receipt_batches.append(copied)
            durable_tokens += sum(receipt["total_tokens"] for receipt in copied)
            return {
                "token_limit": 25,
                "tokens_used": durable_tokens,
                "missing_call_count": 0,
            }

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=4,
            model_usage_callback=persist_usage,
            token_limit=25,
            token_reservation_per_wave=12,
        )
        adapter.set_outcome_callback(
            lambda requests, _outcomes, _attempts: published_sample_ids.extend(
                request.sample_id for request in requests
            )
        )
        requests = _sample_requests(4)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(ModelTokenBudgetExhaustedError) as captured:
            adapter.predict_samples(
                requests,
                tuple(dict(plan) for _request in requests),
                attempts=(1,) * len(requests),
            )

        self.assertEqual(captured.exception.reason, "insufficient_remaining_budget")
        self.assertEqual(captured.exception.tokens_used, 24)
        self.assertEqual(captured.exception.reserved_tokens, 0)
        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(len(receipt_batches), 2)
        self.assertEqual(set(published_sample_ids), {"sample:0", "sample:1"})
        self.assertEqual(
            {
                sample["sample_id"]
                for call in gateway.calls
                for sample in call["samples"]
            },
            {"sample:0", "sample:1"},
        )

    def test_token_budget_conservatively_charges_unreported_usage(self):
        gateway = _SampleDecisionGatewayFake()
        receipt_batches = []
        durable_tokens = 0

        def persist_usage(receipts):
            nonlocal durable_tokens
            copied = tuple(dict(receipt) for receipt in receipts)
            receipt_batches.append(copied)
            # Production derives this durable snapshot from persisted v2
            # receipts, charging the frozen per-call cap when usage is absent.
            durable_tokens += 10 * len(copied)
            return {
                "token_limit": 25,
                "tokens_used": durable_tokens,
                "missing_call_count": 0,
            }

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=2,
            model_usage_callback=persist_usage,
            token_limit=25,
            token_reservation_per_wave=10,
        )
        requests = _sample_requests(3)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(ModelTokenBudgetExhaustedError) as captured:
            adapter.predict_samples(
                requests,
                tuple(dict(plan) for _request in requests),
                attempts=(1,) * len(requests),
            )

        self.assertEqual(captured.exception.reason, "insufficient_remaining_budget")
        self.assertEqual(captured.exception.missing_usage_call_count, 0)
        self.assertEqual(captured.exception.tokens_used, 20)
        self.assertEqual(len(receipt_batches), len(gateway.calls))
        self.assertEqual(len(gateway.calls), 2)

    def test_token_budget_resume_state_blocks_calls_before_submission(self):
        gateway = _DiagnosticSampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            model_usage_callback=lambda _receipts: None,
            token_limit=100,
            token_reservation_per_wave=10,
        )
        adapter.set_token_budget_state(
            {
                "token_limit": 100,
                "tokens_used": 95,
                "missing_call_count": 0,
            }
        )
        requests = _sample_requests(1)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(ModelTokenBudgetExhaustedError) as captured:
            adapter.predict_samples(requests, (plan,), attempts=(1,))

        self.assertEqual(captured.exception.reason, "insufficient_remaining_budget")
        self.assertEqual(gateway.calls, [])

    def test_token_budget_readmits_each_adaptive_split_call(self):
        gateway = _DiagnosticAdaptiveSplitGatewayFake(
            maximum_successful_batch_size=1
        )
        durable_tokens = 0
        receipts = []

        def persist_usage(batch):
            nonlocal durable_tokens
            receipts.extend(dict(item) for item in batch)
            durable_tokens += sum(int(item["total_tokens"]) for item in batch)
            return {
                "token_limit": 55,
                "tokens_used": durable_tokens,
                "missing_call_count": 0,
            }

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=4,
            minimum_split_batch_size=1,
            max_split_depth=2,
            model_usage_callback=persist_usage,
            token_limit=55,
            token_reservation_per_wave=50,
        )
        requests = _sample_requests(4)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(ModelTokenBudgetExhaustedError) as captured:
            adapter.predict_samples(requests, (plan,) * 4, attempts=(1,) * 4)

        self.assertEqual(captured.exception.reason, "insufficient_remaining_budget")
        self.assertEqual([len(call["samples"]) for call in gateway.calls], [4])
        self.assertEqual(len(receipts), 1)
        self.assertEqual(durable_tokens, 48)

    def test_token_budget_readmits_remote_critic_after_planner(self):
        gateway = _DiagnosticSampleDecisionGatewayFake()
        durable_tokens = 0

        def persist_usage(batch):
            nonlocal durable_tokens
            durable_tokens += sum(int(item["total_tokens"]) for item in batch)
            return {
                "token_limit": 20,
                "tokens_used": durable_tokens,
                "missing_call_count": 0,
            }

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            model_usage_callback=persist_usage,
            token_limit=20,
            token_reservation_per_wave=12,
        )
        request = _sample_requests(1)[0]
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(ModelTokenBudgetExhaustedError) as captured:
            adapter.predict_samples((request,), (plan,), attempts=(1,))

        self.assertEqual(captured.exception.reason, "insufficient_remaining_budget")
        self.assertEqual([call["role"] for call in gateway.calls], ["planner"])
        self.assertEqual(durable_tokens, 12)

    def test_concurrent_adaptive_splits_share_per_call_budget_gate(self):
        gateway = _ConcurrentDiagnosticAdaptiveSplitGatewayFake()
        durable_tokens = 0
        receipts = []
        lock = threading.Lock()

        def persist_usage(batch):
            nonlocal durable_tokens
            with lock:
                receipts.extend(dict(item) for item in batch)
                durable_tokens += sum(int(item["total_tokens"]) for item in batch)
                return {
                    "token_limit": 55,
                    "tokens_used": durable_tokens,
                    "missing_call_count": 0,
                }

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=2,
            sample_concurrency=2,
            minimum_split_batch_size=1,
            max_split_depth=2,
            model_usage_callback=persist_usage,
            token_limit=55,
            token_reservation_per_wave=25,
        )
        requests = _sample_requests(4)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(ModelTokenBudgetExhaustedError) as captured:
            adapter.predict_samples(requests, (plan,) * 4, attempts=(1,) * 4)

        self.assertEqual(captured.exception.reason, "insufficient_remaining_budget")
        self.assertEqual(sorted(len(call["samples"]) for call in gateway.calls), [2, 2])
        self.assertEqual(len(receipts), 2)
        self.assertEqual(durable_tokens, 48)

    def test_token_limit_without_wave_reservation_preserves_legacy_identity(self):
        gateway = _SampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            token_limit=1,
        )
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        outcomes = adapter.predict_samples(
            _sample_requests(1),
            (plan,),
            attempts=(1,),
        )

        self.assertEqual(adapter.adapter_version, "9")
        self.assertNotIn("token_budget_policy", plan)
        self.assertIsNone(outcomes[0].error)

    def test_gateway_reports_failed_parent_and_recursive_child_usage(self):
        receipt_batches = []
        gateway = _DiagnosticAdaptiveSplitGatewayFake(
            maximum_successful_batch_size=1
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=4,
            minimum_split_batch_size=1,
            max_split_depth=2,
            model_usage_callback=lambda receipts: receipt_batches.append(receipts),
        )

        batch = self.execute(adapter, rows=_many_rows(4), max_attempts=1)

        self.assertEqual(batch.summary["succeeded_examples"], 4)
        self.assertEqual(
            [len(call["samples"]) for call in gateway.calls],
            [4, 2, 1, 1, 2, 1, 1],
        )
        self.assertEqual(len(receipt_batches), 7)
        self.assertTrue(all(len(batch) == 1 for batch in receipt_batches))
        receipts = [batch[0] for batch in receipt_batches]
        self.assertEqual(len(receipts), 7)
        self.assertEqual(
            [receipt["outcome"] for receipt in receipts],
            [
                "failed",
                "failed",
                "succeeded",
                "succeeded",
                "failed",
                "succeeded",
                "succeeded",
            ],
        )
        self.assertEqual(
            [receipt["prompt_tokens"] for receipt in receipts],
            [40, 20, 10, 10, 20, 10, 10],
        )
        self.assertTrue(all(receipt["usage_reported"] for receipt in receipts))
        self.assertTrue(all(receipt["http_attempts"] == 2 for receipt in receipts))
        self.assertEqual(len({receipt["call_id"] for receipt in receipts}), 7)
        self.assertEqual(len({receipt["logical_call_digest"] for receipt in receipts}), 7)

    def test_gateway_usage_callback_exception_short_circuits_publication(self):
        expected_error = RuntimeError("usage ledger unavailable")
        receipt_batches = []
        progress = []
        outcomes_published = []

        def reject_receipts(receipts):
            receipt_batches.append(receipts)
            raise expected_error

        gateway = _DiagnosticSampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            model_usage_callback=reject_receipts,
            progress_callback=lambda item: progress.append(dict(item)),
        )
        adapter.set_outcome_callback(
            lambda *values: outcomes_published.append(values)
        )
        requests = _sample_requests(1)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(RuntimeError) as raised:
            adapter.predict_samples(requests, (plan,), attempts=(1,))

        self.assertIs(raised.exception, expected_error)
        self.assertEqual(len(receipt_batches), 1)
        self.assertEqual(outcomes_published, [])
        self.assertEqual(progress, [])

    def test_usage_callback_failure_blocks_waiting_gateway_call(self):
        expected_error = RuntimeError("usage ledger unavailable")
        gateway = _ConcurrentDiagnosticGatewayFake()
        receipt_batches = []

        def reject_first_receipt(receipts):
            receipt_batches.append(tuple(dict(item) for item in receipts))
            raise expected_error

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=4,
            model_usage_callback=reject_first_receipt,
            token_limit=20,
            token_reservation_per_wave=10,
        )
        requests = _sample_requests(4)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(RuntimeError) as raised:
            adapter.predict_samples(
                requests,
                tuple(dict(plan) for _request in requests),
                attempts=(1,) * len(requests),
            )

        self.assertIs(raised.exception, expected_error)
        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(len(receipt_batches), 1)

    def test_outcome_callback_failure_drains_late_usage_receipt(self):
        expected_error = RuntimeError("sample result ledger unavailable")
        gateway = _LateDiagnosticReceiptGatewayFake()
        receipt_batches = []
        durable_tokens = 0
        outcome_calls = []

        def persist_usage(receipts):
            nonlocal durable_tokens
            copied = tuple(dict(item) for item in receipts)
            receipt_batches.append(copied)
            durable_tokens += sum(int(item["total_tokens"]) for item in copied)
            return {
                "token_limit": 24,
                "tokens_used": durable_tokens,
                "missing_call_count": 0,
            }

        def reject_first_outcome(requests, outcomes, attempts):
            outcome_calls.append((requests, outcomes, attempts))
            gateway.release_late_call.set()
            raise expected_error

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=2,
            model_usage_callback=persist_usage,
            token_limit=24,
            token_reservation_per_wave=12,
        )
        adapter.set_outcome_callback(reject_first_outcome)
        requests = _sample_requests(2)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(RuntimeError) as raised:
            adapter.predict_samples(
                requests,
                tuple(dict(plan) for _request in requests),
                attempts=(1,) * len(requests),
            )

        self.assertIs(raised.exception, expected_error)
        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(len(receipt_batches), 2)
        self.assertEqual(len(outcome_calls), 1)

    def test_paused_control_before_submission_makes_no_gateway_call(self):
        gateway = _DiagnosticSampleDecisionGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            run_control_callback=lambda: "paused",
        )
        request = _sample_requests(1)[0]
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(SampleExecutionPausedError):
            adapter.predict_samples((request,), (plan,), attempts=(1,))

        self.assertEqual(gateway.calls, [])

    def test_pause_drain_persists_terminal_failure_before_completed_progress(self):
        control = {"status": "running"}
        entered_tools = threading.Barrier(2)
        persisted = []
        progress = []

        def forecast_tool(request):
            try:
                entered_tools.wait(timeout=2)
            except threading.BrokenBarrierError as exc:
                raise AssertionError("both sample tools were not admitted") from exc
            sample_index = int(request.partition.rsplit(":", 1)[1])
            if sample_index == 0:
                raise ArithmeticError("private terminal tool detail")
            control["status"] = "paused"
            return request.proposed_prediction

        adapter = GatewaySampleCollaborationAdapter(
            _SampleDecisionGatewayFake(),
            strategy_model_id="strategy-model",
            forecast_tool=forecast_tool,
            microbatch_size=1,
            sample_concurrency=2,
            run_control_callback=lambda: control["status"],
            progress_callback=lambda item: progress.append(dict(item)),
        )

        with self.assertRaises(SampleExecutionPausedError):
            self.execute(
                adapter,
                rows=_many_rows(2),
                max_attempts=3,
                result_callback=lambda rows: persisted.extend(
                    dict(row) for row in rows
                ),
            )

        durable = build_sample_results("candidate:test", persisted)
        self.assertEqual(len(durable), 2)
        failed = next(row for row in durable if row["status"] == "failed")
        self.assertEqual(failed["observed"], 1.0)
        self.assertEqual(failed["baseline"], 2.0)
        self.assertEqual(failed["predicted"], 2.0)
        self.assertEqual(failed["reward"], 0.0)
        self.assertEqual(failed["attempts"], 1)
        self.assertEqual(failed["failure_class"], "numerical")
        self.assertEqual(
            failed["failure_summary"]["tools"][0]["tool_id"], "algorithm"
        )
        self.assertNotIn(
            "private terminal tool detail",
            json.dumps(failed["failure_summary"]),
        )
        planner_drained = [
            item
            for item in progress
            if item["role"] == "planner" and item["progress_kind"] == "drained"
        ]
        self.assertTrue(planner_drained)
        self.assertEqual(planner_drained[-1]["completed_samples"], len(durable))
        self.assertEqual(planner_drained[-1]["succeeded_samples"], 1)
        self.assertEqual(planner_drained[-1]["failed_samples"], 1)

    def test_pause_keeps_retryable_first_failure_pending_for_resume(self):
        control = {"status": "running"}
        tool_calls = 0
        persisted = []
        progress = []

        def forecast_tool(request):
            nonlocal tool_calls
            tool_calls += 1
            if tool_calls == 1:
                control["status"] = "paused"
                raise TimeoutError("private transient tool detail")
            return request.proposed_prediction

        adapter = GatewaySampleCollaborationAdapter(
            _SampleDecisionGatewayFake(),
            strategy_model_id="strategy-model",
            forecast_tool=forecast_tool,
            microbatch_size=1,
            sample_concurrency=1,
            run_control_callback=lambda: control["status"],
            progress_callback=lambda item: progress.append(dict(item)),
        )
        rows = _many_rows(1)
        executor = CollaborativeSampleExecutor(adapter, sleep=lambda _: None)
        policy = SampleExecutionPolicy(max_attempts=3, retry_backoff_seconds=0)
        context = {
            "candidate_id": "candidate:test",
            "dataset_digest": "dataset:test",
            "algorithm_id": "algorithm",
            "algorithm_version": "1",
        }
        bounds = {"x": {"unit": "u", "minimum": -100.0, "maximum": 100.0}}

        with self.assertRaises(SampleExecutionPausedError):
            executor.execute(
                rows,
                context=context,
                target_bounds=bounds,
                algorithm_id="algorithm",
                algorithm_version="1",
                policy=policy,
                checkpoint_callback=lambda _checkpoint: {
                    "rows": (),
                    "progress": None,
                },
                result_callback=lambda values: persisted.extend(
                    dict(value) for value in values
                ),
            )

        first_drained = next(
            item
            for item in reversed(progress)
            if item["role"] == "planner" and item["progress_kind"] == "drained"
        )
        self.assertEqual(first_drained["completed_samples"], 0)
        self.assertEqual(persisted, [])

        control["status"] = "running"
        batch = executor.execute(
            rows,
            context=context,
            target_bounds=bounds,
            algorithm_id="algorithm",
            algorithm_version="1",
            policy=policy,
            checkpoint_callback=lambda _checkpoint: {
                "rows": (),
                "progress": first_drained,
            },
            result_callback=lambda values: persisted.extend(
                dict(value) for value in values
            ),
        )

        self.assertEqual(tool_calls, 2)
        self.assertEqual(batch.summary["succeeded_examples"], 1)
        durable = build_sample_results("candidate:test", persisted)
        self.assertEqual(len(durable), 1)
        self.assertEqual(durable[0]["status"], "succeeded")
        self.assertNotIn("failure_summary", durable[0])
        planner_progress = [item for item in progress if item["role"] == "planner"]
        self.assertEqual(planner_progress[-1]["completed_samples"], len(durable))

    def test_pause_drains_admitted_calls_and_latches_across_fast_resume(self):
        gateway = _ConcurrentDiagnosticGatewayFake()
        control = {"status": "running"}
        receipt_batches = []
        published_sample_ids = []
        progress = []

        def persist_usage(receipts):
            receipt_batches.append(tuple(dict(item) for item in receipts))
            control["status"] = "paused"

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=2,
            model_usage_callback=persist_usage,
            run_control_callback=lambda: control["status"],
            progress_callback=lambda item: progress.append(dict(item)),
        )
        def publish_outcomes(requests, _outcomes, _attempts):
            published_sample_ids.extend(request.sample_id for request in requests)
            control["status"] = "running"

        adapter.set_outcome_callback(publish_outcomes)
        requests = _sample_requests(4)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(SampleExecutionPausedError):
            adapter.predict_samples(
                requests,
                tuple(dict(plan) for _request in requests),
                attempts=(1,) * len(requests),
            )

        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(len(receipt_batches), 2)
        self.assertEqual(set(published_sample_ids), {"sample:0", "sample:1"})
        drained = [item for item in progress if item["progress_kind"] == "drained"]
        self.assertTrue(drained)
        self.assertEqual(drained[-1]["in_flight_batches"], 0)
        self.assertEqual(drained[-1]["queued_batches"], 0)

    def test_pause_after_planner_receipt_prevents_remote_critic(self):
        gateway = _DiagnosticSampleDecisionGatewayFake()
        control = {"status": "running"}
        receipt_batches = []

        def persist_usage(receipts):
            receipt_batches.append(tuple(dict(item) for item in receipts))
            control["status"] = "paused"

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            review_model_id="review-model",
            remote_review_enabled=True,
            model_usage_callback=persist_usage,
            run_control_callback=lambda: control["status"],
        )
        request = _sample_requests(1)[0]
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(SampleExecutionPausedError):
            adapter.predict_samples((request,), (plan,), attempts=(1,))

        self.assertEqual([call["role"] for call in gateway.calls], ["planner"])
        self.assertEqual(len(receipt_batches), 1)

    def test_pause_after_failed_parent_prevents_adaptive_split(self):
        gateway = _DiagnosticAdaptiveSplitGatewayFake(
            maximum_successful_batch_size=1
        )
        control = {"status": "running"}
        receipt_batches = []

        def persist_usage(receipts):
            receipt_batches.append(tuple(dict(item) for item in receipts))
            control["status"] = "paused"

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=4,
            minimum_split_batch_size=1,
            max_split_depth=2,
            model_usage_callback=persist_usage,
            run_control_callback=lambda: control["status"],
        )
        requests = _sample_requests(4)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(SampleExecutionPausedError):
            adapter.predict_samples(requests, (plan,) * 4, attempts=(1,) * 4)

        self.assertEqual([len(call["samples"]) for call in gateway.calls], [4])
        self.assertEqual(len(receipt_batches), 1)

    def test_cancel_drains_usage_without_publishing_outcome_or_progress(self):
        gateway = _ConcurrentDiagnosticGatewayFake()
        control = {"status": "running"}
        receipt_batches = []
        outcomes = []
        progress = []

        def persist_usage(receipts):
            receipt_batches.append(tuple(dict(item) for item in receipts))
            control["status"] = "cancelled"

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=2,
            model_usage_callback=persist_usage,
            run_control_callback=lambda: control["status"],
            progress_callback=lambda item: progress.append(dict(item)),
        )
        adapter.set_outcome_callback(lambda *values: outcomes.append(values))
        requests = _sample_requests(4)
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(SampleExecutionCancelledError):
            adapter.predict_samples(
                requests,
                tuple(dict(plan) for _request in requests),
                attempts=(1,) * len(requests),
            )

        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(len(receipt_batches), 2)
        self.assertEqual(outcomes, [])
        self.assertEqual(progress, [])

    def test_run_control_callback_failure_stops_before_submission(self):
        gateway = _DiagnosticSampleDecisionGatewayFake()

        def unavailable_control():
            raise RuntimeError("control projection unavailable")

        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            run_control_callback=unavailable_control,
        )
        request = _sample_requests(1)[0]
        plan = adapter.plan_batch(
            {
                "candidate_id": "candidate:test",
                "dataset_digest": "dataset:test",
                "algorithm_id": "algorithm",
                "algorithm_version": "1",
            }
        )

        with self.assertRaises(SampleExecutionControlUnavailableError):
            adapter.predict_samples((request,), (plan,), attempts=(1,))

        self.assertEqual(gateway.calls, [])

    def test_legacy_gateway_usage_is_explicitly_unreported(self):
        receipt_batches = []
        adapter = GatewaySampleCollaborationAdapter(
            _SampleDecisionGatewayFake(),
            strategy_model_id="strategy-model",
            model_usage_callback=lambda receipts: receipt_batches.append(receipts),
        )

        batch = self.execute(adapter, rows=[_rows()[0]], max_attempts=1)

        self.assertEqual(batch.summary["succeeded_examples"], 1)
        self.assertEqual(len(receipt_batches), 1)
        self.assertEqual(len(receipt_batches[0]), 1)
        receipt = receipt_batches[0][0]
        self.assertFalse(receipt["usage_reported"])
        self.assertEqual(receipt["http_attempts"], 1)
        self.assertEqual(receipt["prompt_tokens"], 0)
        self.assertEqual(receipt["completion_tokens"], 0)
        self.assertEqual(receipt["total_tokens"], 0)

    def test_gateway_adapter_splits_contract_failure_and_recovers_by_sample_id(self):
        progress = []
        gateway = _AdaptiveSplitGatewayFake(maximum_successful_batch_size=2)
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=8,
            minimum_split_batch_size=2,
            max_split_depth=3,
            progress_callback=lambda item: progress.append(dict(item)),
        )

        batch = self.execute(adapter, rows=_many_rows(8), max_attempts=1)

        self.assertEqual(
            [len(call["samples"]) for call in gateway.calls],
            [8, 4, 2, 2, 4, 2, 2],
        )
        self.assertEqual(batch.summary["succeeded_examples"], 8)
        for record in batch.records:
            remote_step = next(
                item
                for item in record["agent_trace"]
                if item["role"] == "remote_planner_agent"
            )
            self.assertEqual(remote_step["reason_code"], "remote_reason_invalid")
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["batch_size"], 8)
        self.assertEqual(progress[0]["completed_samples"], 8)
        self.assertEqual(progress[0]["succeeded_samples"], 8)
        self.assertEqual(progress[0]["failed_samples"], 0)
        self.assertEqual(progress[0]["gateway_request_count"], 7)
        self.assertEqual(progress[0]["adaptive_split_trigger_count"], 3)
        self.assertEqual(progress[0]["adaptive_split_count"], 3)
        self.assertEqual(progress[0]["adaptive_split_max_depth"], 2)
        self.assertEqual(progress[0]["adaptive_split_recovered_samples"], 8)
        self.assertEqual(progress[0]["adaptive_split_failed_samples"], 0)
        self.assertNotIn("sample_id", str(progress))
        self.assertNotIn("one JSON object", str(progress))

    def test_gateway_adapter_default_small_wave_stops_splitting_at_pairs(self):
        progress = []
        gateway = _AdaptiveSplitGatewayFake(maximum_successful_batch_size=1)
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=64,
            progress_callback=lambda item: progress.append(dict(item)),
        )

        batch = self.execute(adapter, rows=_rows(), max_attempts=1)

        self.assertEqual(
            [len(call["samples"]) for call in gateway.calls],
            [3, 1, 2],
        )
        self.assertEqual(batch.summary["succeeded_examples"], 1)
        self.assertEqual(batch.summary["failed_examples"], 2)
        self.assertEqual(progress[0]["adaptive_split_trigger_count"], 2)
        self.assertEqual(progress[0]["adaptive_split_count"], 1)
        self.assertEqual(progress[0]["adaptive_split_recovered_samples"], 1)
        self.assertEqual(progress[0]["adaptive_split_failed_samples"], 2)
        self.assertEqual(
            adapter.plan_batch({})["adaptive_split_policy"][
                "default_small_wave_floor"
            ],
            2,
        )

    def test_gateway_adapter_publishes_waiting_heartbeat_for_slow_gateway(self):
        progress = []
        adapter = GatewaySampleCollaborationAdapter(
            _HeartbeatBlockingGatewayFake(),
            strategy_model_id="strategy-model",
            microbatch_size=1,
            sample_concurrency=1,
            progress_callback=lambda item: progress.append(dict(item)),
        )

        with patch.object(
            gateway_sample_adapter_module,
            "_PROGRESS_HEARTBEAT_SECONDS",
            0.005,
        ):
            self.execute(adapter, rows=_many_rows(2), max_attempts=1)

        waiting = [
            item for item in progress if item["progress_kind"] == "waiting"
        ]
        self.assertTrue(waiting)
        heartbeat = waiting[0]
        self.assertEqual(heartbeat["batch_index"], 0)
        self.assertEqual(heartbeat["batch_size"], 0)
        self.assertEqual(heartbeat["completed_samples"], 0)
        self.assertGreater(heartbeat["in_flight_batches"], 0)
        self.assertGreaterEqual(heartbeat["queued_batches"], 1)

    def test_gateway_adapter_default_nine_sample_wave_stops_at_pairs(self):
        progress = []
        gateway = _AdaptiveSplitGatewayFake(maximum_successful_batch_size=1)
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=64,
            progress_callback=lambda item: progress.append(dict(item)),
        )

        batch = self.execute(adapter, rows=_many_rows(9), max_attempts=1)

        self.assertEqual(batch.summary["succeeded_examples"], 1)
        self.assertEqual(batch.summary["failed_examples"], 8)
        self.assertEqual(
            [len(call["samples"]) for call in gateway.calls],
            [9, 4, 2, 2, 5, 2, 3, 1, 2],
        )
        self.assertEqual(progress[0]["adaptive_split_trigger_count"], 8)
        self.assertEqual(progress[0]["adaptive_split_count"], 4)
        self.assertEqual(progress[0]["adaptive_split_max_depth"], 3)
        self.assertEqual(progress[0]["adaptive_split_recovered_samples"], 1)
        self.assertEqual(progress[0]["adaptive_split_failed_samples"], 8)
        self.assertEqual(
            adapter.plan_batch({})["adaptive_split_policy"][
                "default_fine_grained_wave_max_size"
            ],
            16,
        )

    def test_real_gateway_format_failure_has_a_bounded_split_request_count(self):
        progress = []
        opener = _AlwaysMalformedModelOpener()
        adapter = GatewaySampleCollaborationAdapter(
            _model_gateway(opener),
            strategy_model_id="strategy-model",
            microbatch_size=64,
            minimum_split_batch_size=8,
            max_split_depth=4,
            progress_callback=lambda item: progress.append(dict(item)),
        )

        batch = self.execute(adapter, rows=_many_rows(64), max_attempts=3)

        # The root receives one syntax retry. The fourteen descendants each
        # issue one request instead of inheriting the format retry budget. The
        # exhausted split tree is terminal and never receives a repair wave.
        self.assertEqual(opener.calls, 16)
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["role"], "planner")
        self.assertEqual(progress[0]["gateway_request_count"], opener.calls)
        self.assertEqual(progress[0]["adaptive_split_trigger_count"], 15)
        self.assertEqual(batch.summary["failed_examples"], 64)
        self.assertEqual({record["attempts"] for record in batch.records}, {1})
        self.assertEqual(
            {record["failure"]["retryable"] for record in batch.records},
            {False},
        )

    def test_real_gateway_preflight_failure_reports_zero_http_attempts(self):
        progress = []
        receipt_batches = []
        opener = _AlwaysMalformedModelOpener()
        adapter = GatewaySampleCollaborationAdapter(
            _model_gateway(opener),
            strategy_model_id="missing-strategy-model",
            microbatch_size=8,
            progress_callback=lambda item: progress.append(dict(item)),
            model_usage_callback=lambda receipts: receipt_batches.append(receipts),
        )

        batch = self.execute(adapter, rows=_many_rows(1), max_attempts=1)

        self.assertEqual(opener.calls, 0)
        self.assertEqual(receipt_batches, [])
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["gateway_request_count"], 0)
        self.assertEqual(batch.summary["failed_examples"], 1)

    def test_reflected_authorization_never_enters_sample_trace_or_summary(self):
        token = "sample-reflected-secret"
        opener = _ReflectingReasonModelOpener()
        adapter = GatewaySampleCollaborationAdapter(
            _model_gateway(opener, token=token),
            strategy_model_id="strategy-model",
            microbatch_size=64,
        )

        batch = self.execute(adapter, rows=_rows(), max_attempts=1)
        archive = encode_sample_execution_trace(batch.records)
        decoded_archive = zlib.decompress(
            base64.b64decode(archive["payload"])
        ).decode("utf-8")
        public = json.dumps(
            {"summary": batch.summary, "records": batch.records},
            ensure_ascii=False,
        )

        self.assertEqual(opener.calls, 1)
        self.assertEqual(batch.summary["failed_examples"], 0)
        self.assertEqual(
            batch.summary["reason_code_counts"], {"remote_reason_invalid": 3}
        )
        self.assertNotIn(token, public)
        self.assertNotIn(token, decoded_archive)
        self.assertNotIn("Bearer", public)
        self.assertNotIn("Bearer", decoded_archive)

    def test_gateway_adapter_does_not_split_exhausted_transport_failure(self):
        progress = []
        gateway = _AlwaysFailingGatewayFake(
            GatewayResponseError(
                "policy model request failed: retry budget exhausted",
                retryable=True,
                attempts=4,
            )
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=8,
            minimum_split_batch_size=1,
            max_split_depth=8,
            progress_callback=lambda item: progress.append(dict(item)),
        )

        with self.assertRaises(GatewayResponseError) as captured:
            self.execute(adapter, rows=_many_rows(8), max_attempts=3)

        self.assertIs(captured.exception, gateway.error)
        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual([call["role"] for call in gateway.calls], ["planner"])
        self.assertEqual(progress, [])

    def test_gateway_adapter_repairs_unsplittable_invalid_tool_wave(self):
        gateway = _FailFirstInvalidToolGatewayFake()
        progress = []
        persisted = []
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=3,
            minimum_split_batch_size=3,
            max_split_depth=3,
            progress_callback=lambda item: progress.append(dict(item)),
        )

        batch = self.execute(
            adapter,
            rows=_rows(),
            max_attempts=3,
            result_callback=lambda rows: persisted.extend(dict(row) for row in rows),
        )

        self.assertEqual(
            [call["role"] for call in gateway.calls], ["planner", "repair"]
        )
        self.assertEqual([len(call["samples"]) for call in gateway.calls], [3, 3])
        self.assertEqual(
            [item["sample_id"] for item in gateway.calls[0]["samples"]],
            [item["sample_id"] for item in gateway.calls[1]["samples"]],
        )
        for call, expected_attempt in zip(gateway.calls, (1, 2)):
            self.assertEqual(
                {item["attempt"] for item in call["samples"]},
                {expected_attempt},
            )
            self.assertEqual(
                {
                    item["sample"]["label_free_context"]["causal_provenance"][
                        "origin_cutoff_timestamp"
                    ]
                    for item in call["samples"]
                },
                {0},
            )
        self.assertTrue(
            all(item["failure_feedback"] for item in gateway.calls[1]["samples"])
        )
        self.assertEqual(batch.summary["succeeded_examples"], 3)
        self.assertEqual(batch.summary["failed_examples"], 0)
        self.assertFalse(batch.summary["coverage_early_stopped"])
        self.assertIsNone(batch.summary["early_stop_reason"])
        self.assertEqual({record["attempts"] for record in batch.records}, {2})
        self.assertEqual(
            {
                record["failure_history"][0]["failure_class"]
                for record in batch.records
            },
            {"remote_batch_invalid_output"},
        )
        durable = build_sample_results("candidate:test", persisted)
        self.assertEqual(len(durable), 3)
        self.assertTrue(all(row["status"] == "succeeded" for row in durable))
        planner_progress = [item for item in progress if item["role"] == "planner"]
        self.assertTrue(planner_progress)
        self.assertEqual(planner_progress[-1]["total_samples"], 3)
        self.assertEqual(planner_progress[-1]["completed_samples"], len(durable))
        self.assertEqual(planner_progress[-1]["succeeded_samples"], len(durable))

    def test_gateway_adapter_repairs_one_local_per_decision_contract_error(self):
        gateway = _FailFirstLocalDecisionContractGatewayFake()
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=3,
            minimum_split_batch_size=3,
        )

        batch = self.execute(adapter, rows=_rows(), max_attempts=3)

        self.assertEqual(
            [call["role"] for call in gateway.calls], ["planner", "repair"]
        )
        self.assertEqual(batch.summary["succeeded_examples"], 3)
        self.assertEqual({record["attempts"] for record in batch.records}, {2})

    def test_gateway_adapter_splits_local_decision_count_contract_failure(self):
        progress = []
        gateway = _AdaptiveSplitGatewayFake(
            maximum_successful_batch_size=2,
            incomplete_decisions=True,
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=8,
            minimum_split_batch_size=2,
            max_split_depth=3,
            progress_callback=lambda item: progress.append(dict(item)),
        )

        batch = self.execute(adapter, rows=_many_rows(8), max_attempts=1)

        self.assertEqual(
            [len(call["samples"]) for call in gateway.calls],
            [8, 4, 2, 2, 4, 2, 2],
        )
        self.assertEqual(batch.summary["succeeded_examples"], 8)
        self.assertEqual(progress[0]["adaptive_split_trigger_count"], 3)
        self.assertEqual(progress[0]["adaptive_split_recovered_samples"], 8)

    def test_gateway_adapter_stops_adaptive_split_at_maximum_depth(self):
        progress = []
        gateway = _AlwaysFailingGatewayFake(
            GatewayResponseError(
                "model response content must contain one JSON object",
                error_code="response_format_invalid",
                split_eligible=True,
            )
        )
        adapter = GatewaySampleCollaborationAdapter(
            gateway,
            strategy_model_id="strategy-model",
            microbatch_size=16,
            minimum_split_batch_size=1,
            max_split_depth=1,
            progress_callback=lambda item: progress.append(dict(item)),
        )

        batch = self.execute(adapter, rows=_many_rows(16), max_attempts=1)

        self.assertEqual([len(call["samples"]) for call in gateway.calls], [16, 8, 8])
        self.assertEqual(batch.summary["failed_examples"], 16)
        self.assertEqual(progress[0]["gateway_request_count"], 3)
        self.assertEqual(progress[0]["adaptive_split_trigger_count"], 3)
        self.assertEqual(progress[0]["adaptive_split_count"], 1)
        self.assertEqual(progress[0]["adaptive_split_max_depth"], 1)
        self.assertEqual(progress[0]["adaptive_split_recovered_samples"], 0)
        self.assertEqual(progress[0]["adaptive_split_failed_samples"], 16)

    def test_gateway_adapter_does_not_split_permanent_gateway_failures(self):
        permanent_failures = (
            GatewayResponseError("authentication rejected", status_code=401),
            GatewayResponseError("permission rejected", status_code=403),
            GatewayResponseError("route not found", status_code=404),
            GatewayConfigurationError("model role is not configured"),
            KeyError("unknown policy model route"),
        )
        for error in permanent_failures:
            with self.subTest(error_type=type(error).__name__, error=str(error)):
                progress = []
                gateway = _AlwaysFailingGatewayFake(error)
                adapter = GatewaySampleCollaborationAdapter(
                    gateway,
                    strategy_model_id="strategy-model",
                    microbatch_size=8,
                    minimum_split_batch_size=1,
                    max_split_depth=8,
                    progress_callback=lambda item, progress=progress: progress.append(
                        dict(item)
                    ),
                )

                batch = self.execute(adapter, rows=_many_rows(8), max_attempts=3)

                self.assertEqual(len(gateway.calls), 1)
                self.assertEqual(
                    [call["role"] for call in gateway.calls], ["planner"]
                )
                self.assertEqual(batch.summary["failed_examples"], 8)
                self.assertEqual(progress[0]["gateway_request_count"], 1)
                self.assertEqual(progress[0]["adaptive_split_trigger_count"], 0)
                self.assertEqual(progress[0]["adaptive_split_count"], 0)

    def test_registry_selects_gateway_executor_only_for_frozen_new_runs(self):
        gateway = _SampleDecisionGatewayFake()
        registry = EvaluatorRegistry(object(), model_gateway=gateway)
        remote_task = TaskManifest(
            task_id="remote-sample-runtime",
            objective="route real samples remotely",
            domain_pack="greenhouse_cucumber_2018",
            visible_datasets=("agc_cucumber_2018",),
            budget={
                "max_candidates": 1,
                "token_limit": 50_000,
                "token_reservation_per_wave": 8_000,
            },
            metadata={
                "sample_agent_mode": "gateway_microbatch",
                "sample_agent_batch_size": 64,
                "strategy_model_id": "strategy-model",
                "review_model_id": "review-model",
                "sample_operation_max_tokens": {
                    "sample.planner": 3072,
                    "sample.repair": 3072,
                    "sample.critic": 2048,
                },
                "sample_remote_critic_policy": {
                    "version": "uncertain_or_failure@1",
                    "min_planner_confidence": 0.7,
                },
            },
        )

        remote_executor = registry._sample_executor_for_task(remote_task)

        self.assertIsInstance(
            remote_executor.adapter, GatewaySampleCollaborationAdapter
        )
        self.assertEqual(remote_executor.adapter.strategy_model_id, "strategy-model")
        self.assertEqual(remote_executor.adapter.microbatch_size, 64)
        self.assertTrue(remote_executor.adapter.remote_review_enabled)
        self.assertEqual(remote_executor.adapter.review_model_id, "review-model")
        self.assertEqual(
            remote_executor.adapter.operation_max_tokens,
            {
                "sample.planner": 3072,
                "sample.repair": 3072,
                "sample.critic": 2048,
            },
        )
        self.assertEqual(
            remote_executor.adapter.remote_critic_policy,
            {
                "version": "uncertain_or_failure@1",
                "min_planner_confidence": 0.7,
            },
        )
        self.assertTrue(remote_executor.adapter.token_budget_enabled)
        self.assertEqual(remote_executor.adapter.token_limit, 50_000)
        self.assertEqual(remote_executor.adapter.token_reservation_per_wave, 8_000)
        self.assertEqual(
            remote_executor.adapter.adapter_version,
            "11-token-call-budget",
        )
        self.assertEqual(
            remote_executor.adapter.plan_batch({})["token_budget_policy"],
            {
                "mode": "hard_gateway_call_reservation",
                "reservation_scope": "logical_gateway_call_with_http_retries",
                "token_limit": 50_000,
                "token_reservation_per_wave": 8_000,
            },
        )

        legacy_task = TaskManifest(
            task_id="legacy-sample-runtime",
            objective="replay historical semantics",
            domain_pack="greenhouse_cucumber_2018",
            visible_datasets=("agc_cucumber_2018",),
        )
        self.assertIs(
            registry._sample_executor_for_task(legacy_task), registry.sample_executor
        )

        injected = CollaborativeSampleExecutor(sleep=lambda _: None)
        injected_registry = EvaluatorRegistry(
            object(), model_gateway=gateway, sample_executor=injected
        )
        self.assertIs(
            injected_registry._sample_executor_for_task(remote_task), injected
        )

    def test_remote_ridge_evaluation_executes_selected_tools_on_demand(self):
        series = _series()
        gateway = _SampleDecisionGatewayFake()
        task = TaskManifest(
            task_id="remote-ridge-tools",
            objective="route real ridge samples through selected host tools",
            domain_pack="greenhouse_cucumber_2018",
            visible_datasets=(series.dataset_id,),
            metadata={
                "evaluator_id": GREENHOUSE_EVALUATOR_ID,
                "prediction_model_id": EXOGENOUS_RIDGE_MODEL_ID,
                "episode_id": series.episode_id,
                "dataset_digest": series.digest,
                "split_manifest_digest": series.split_manifest_digest_sha256,
                "sample_agent_mode": "gateway_microbatch",
                "sample_agent_batch_size": 64,
                "strategy_model_id": "strategy-model",
                "review_model_id": "review-model",
            },
        )
        candidate, proposal = _candidate(
            {"history_steps": 3, "ridge_alpha": 0.1, "residual_scale": 0.5}
        )

        bundle = EvaluatorRegistry(
            _DatasetStub(series), model_gateway=gateway
        ).evaluate_scientific(task, candidate, proposal)

        planner_calls = [item for item in gateway.calls if item["role"] == "planner"]
        critic_calls = [item for item in gateway.calls if item["role"] == "critic"]
        self.assertTrue(planner_calls)
        self.assertTrue(critic_calls)
        self.assertTrue(
            all(
                "predicted" not in item["sample"]
                and "proposed_prediction" not in item["sample"]
                for call in planner_calls
                for item in call["samples"]
            )
        )
        available = {
            tool["tool_id"]
            for call in planner_calls
            for tool in call["available_tools"]
        }
        self.assertIn("greenhouse-exogenous-ridge", available)
        self.assertIn("greenhouse-ridge-conservative@1", available)
        self.assertIn("bounded-persistence-fallback", available)
        summary = bundle.evaluation.metrics["sample_execution"]
        self.assertEqual(summary["failed_examples"], 0)
        self.assertTrue(summary["tool_performance"])
        self.assertTrue(
            all(
                record.get("attempt_trace")
                for record in bundle.evaluation.metrics["sample_execution_records"]
            )
        )

    def test_remote_rolling_evaluation_only_executes_selected_tool(self):
        series = _series()
        gateway = _SampleDecisionGatewayFake(
            planner_tool="bounded-persistence-fallback"
        )
        task = TaskManifest(
            task_id="remote-rolling-tools",
            objective="route rolling samples through selected host tools",
            domain_pack="greenhouse_cucumber_2018",
            visible_datasets=(series.dataset_id,),
            metadata={
                "evaluator_id": GREENHOUSE_EVALUATOR_ID,
                "prediction_model_id": GREENHOUSE_ROLLING_PREDICTOR_ID,
                "episode_id": series.episode_id,
                "dataset_digest": series.digest,
                "split_manifest_digest": series.split_manifest_digest_sha256,
                "sample_agent_mode": "gateway_microbatch",
                "sample_agent_batch_size": 64,
                "strategy_model_id": "strategy-model",
                "review_model_id": "review-model",
            },
        )
        candidate, proposal = _candidate(
            {"blend": 0.5, "window": 3, "bias_scale": 0.5}
        )

        with patch(
            "ecologyrsi_dsh.evaluators.registry._rolling_tool_prediction",
            side_effect=AssertionError("unselected rolling tool executed"),
        ) as rolling_tool:
            bundle = EvaluatorRegistry(
                _DatasetStub(series), model_gateway=gateway
            ).evaluate_scientific(task, candidate, proposal)

        rolling_tool.assert_not_called()
        planner_calls = [item for item in gateway.calls if item["role"] == "planner"]
        self.assertTrue(planner_calls)
        self.assertTrue(
            all(
                "predicted" not in item["sample"]
                and "proposed_prediction" not in item["sample"]
                for call in planner_calls
                for item in call["samples"]
            )
        )
        records = bundle.evaluation.metrics["sample_execution_records"]
        self.assertTrue(records)
        self.assertTrue(
            all(
                record["attempt_trace"][0]["selected_tool"]["tool_id"]
                == "bounded-persistence-fallback"
                for record in records
            )
        )
        self.assertTrue(
            all(
                row["predicted"] == row["baseline"]
                for row in bundle.evaluation.metrics["prediction_preview"]
            )
        )

    def test_selected_remote_rolling_tool_matches_eager_host_prediction(self):
        series = _series()
        metadata = {
            "evaluator_id": GREENHOUSE_EVALUATOR_ID,
            "prediction_model_id": GREENHOUSE_ROLLING_PREDICTOR_ID,
            "episode_id": series.episode_id,
            "dataset_digest": series.digest,
            "split_manifest_digest": series.split_manifest_digest_sha256,
        }
        host_task = TaskManifest(
            task_id="host-rolling-reference",
            objective="preserve historical rolling predictions",
            domain_pack="greenhouse_cucumber_2018",
            visible_datasets=(series.dataset_id,),
            metadata=metadata,
        )
        remote_task = TaskManifest(
            task_id="remote-rolling-primary",
            objective="execute selected rolling tool on demand",
            domain_pack="greenhouse_cucumber_2018",
            visible_datasets=(series.dataset_id,),
            metadata={
                **metadata,
                "sample_agent_mode": "gateway_microbatch",
                "sample_agent_batch_size": 64,
                "strategy_model_id": "strategy-model",
                "review_model_id": "review-model",
            },
        )
        candidate, proposal = _candidate(
            {"blend": 0.5, "window": 3, "bias_scale": 0.5}
        )
        host_bundle = EvaluatorRegistry(
            _DatasetStub(series), model_gateway=object()
        ).evaluate_scientific(host_task, candidate, proposal)
        gateway = _SampleDecisionGatewayFake()

        with patch.object(
            evaluator_registry_module,
            "_rolling_tool_prediction",
            wraps=evaluator_registry_module._rolling_tool_prediction,
        ) as rolling_tool:
            remote_bundle = EvaluatorRegistry(
                _DatasetStub(series), model_gateway=gateway
            ).evaluate_scientific(remote_task, candidate, proposal)

        attempted = remote_bundle.evaluation.metrics["sample_execution"][
            "attempted_examples"
        ]
        self.assertEqual(rolling_tool.call_count, attempted)
        self.assertTrue(
            all(call.args[0].proposed_prediction is None for call in rolling_tool.call_args_list)
        )
        self.assertEqual(remote_bundle.evaluation.score, host_bundle.evaluation.score)
        self.assertEqual(
            remote_bundle.evaluation.metrics["targets"],
            host_bundle.evaluation.metrics["targets"],
        )
        preview_fields = (
            "origin_timestamp",
            "target_timestamp",
            "timestamp",
            "horizon_hours",
            "observed",
            "predicted",
            "baseline",
            "target",
            "unit",
        )
        self.assertEqual(
            [
                {name: row[name] for name in preview_fields}
                for row in remote_bundle.evaluation.metrics["prediction_preview"]
            ],
            [
                {name: row[name] for name in preview_fields}
                for row in host_bundle.evaluation.metrics["prediction_preview"]
            ],
        )
        for digest_field in (
            "evaluation_index_digest",
            "baseline_metrics_digest",
        ):
            self.assertEqual(
                remote_bundle.evaluation.metrics[digest_field],
                host_bundle.evaluation.metrics[digest_field],
            )
        self.assertTrue(any(call["role"] == "planner" for call in gateway.calls))
        self.assertTrue(any(call["role"] == "critic" for call in gateway.calls))
        self.assertTrue(
            all(
                attempt["selected_tool"].get("output_digest")
                for record in remote_bundle.evaluation.metrics[
                    "sample_execution_records"
                ]
                for attempt in record["attempt_trace"]
            )
        )

    def test_all_registered_evaluator_paths_emit_sample_evidence(self):
        executor = CollaborativeSampleExecutor(sleep=lambda _: None)

        toy_task = TaskManifest(
            task_id="toy-samples",
            objective="toy",
            domain_pack="toy",
            visible_datasets=(TOY_DATASET_ID,),
            metadata={
                "evaluator_id": TOY_EVALUATOR_ID,
                "prediction_model_id": TOY_PREDICTOR_MODEL_ID,
            },
        )
        toy_candidate, toy_proposal = _candidate(
            {"alpha": 0.4, "window": 5, "water_threshold": 0.4}
        )
        toy = EvaluatorRegistry(
            object(), model_gateway=object(), sample_executor=executor
        )._evaluate_toy(toy_task, toy_candidate, toy_proposal)
        self._assert_evidence(toy.evaluation.metrics)

        series = _series()
        for predictor_id, evaluator_id, changes in (
            (
                GREENHOUSE_ROLLING_PREDICTOR_ID,
                GREENHOUSE_EVALUATOR_ID,
                {"blend": 0.5, "window": 3, "bias_scale": 0.5},
            ),
            (
                EXOGENOUS_RIDGE_MODEL_ID,
                GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
                {"history_steps": 3, "ridge_alpha": 0.1, "residual_scale": 0.5},
            ),
        ):
            task = TaskManifest(
                task_id="greenhouse-samples-" + predictor_id,
                objective="greenhouse",
                domain_pack="greenhouse_cucumber_2018",
                visible_datasets=(series.dataset_id,),
                metadata={
                    "evaluator_id": evaluator_id,
                    "prediction_model_id": predictor_id,
                    "episode_id": series.episode_id,
                    "dataset_digest": series.digest,
                    "split_manifest_digest": series.split_manifest_digest_sha256,
                },
            )
            candidate, proposal = _candidate(changes)
            bundle = EvaluatorRegistry(
                _DatasetStub(series),
                model_gateway=object(),
                sample_executor=executor,
            ).evaluate_scientific(task, candidate, proposal)
            self._assert_evidence(bundle.evaluation.metrics)

    def _assert_evidence(self, metrics):
        summary = metrics["sample_execution"]
        records = metrics["sample_execution_records"]
        archive = metrics["sample_execution_trace_archive"]
        self.assertGreater(summary["eligible_examples"], 0)
        self.assertEqual(summary["failed_examples"], 0)
        self.assertTrue(summary["coverage_pass"])
        self.assertLessEqual(len(records), 64)
        self.assertTrue(all("action_digest" in item for item in records))
        restored = json.loads(
            zlib.decompress(base64.b64decode(archive["payload"])).decode("utf-8")
        )
        self.assertEqual(len(restored), summary["attempted_examples"])
        self.assertEqual(archive["record_count"], summary["attempted_examples"])
        self.assertEqual(archive["trace_digest"], summary["trace_digest"])
        self.assertEqual(
            metrics["sample_execution_trace_record_count"],
            summary["attempted_examples"],
        )


if __name__ == "__main__":
    unittest.main()
