"""Auditable, failure-tolerant execution for per-sample predictions.

The registered predictor may prepare one candidate-wide plan, but every
evaluation example crosses this boundary independently.  Adapters never
receive the observed label.  They return only a bounded operational trace:
the prediction, public agent decisions, and versioned tool calls.  Hidden
reasoning and raw prompts are deliberately outside this contract.
"""

from __future__ import annotations

import base64
import math
import socket
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.error import HTTPError, URLError

from ..core.errors import (
    DshNativeRuntimeUnavailableError,
    dsh_native_runtime_error_in_chain,
    dsh_native_runtime_retryable,
)
from ..core.models import canonical_json, digest
from ..core.protocols import (
    is_strict_origin_protocol,
    supports_concurrent_origins,
)
from ..core.redaction import safe_remote_reason_code
from ..evolution.execution_plan import DerivedExecutionPlan
from ..integrations.model_gateway import gateway_error_in_chain
from .objectives import skill_score

SAMPLE_EXECUTION_SCHEMA_VERSION = "ecologyrsi-dsh.sample-execution/2"
SAMPLE_EXECUTION_TRACE_ARCHIVE_VERSION = "ecologyrsi-dsh.sample-execution-trace/2"
DEFAULT_SAMPLE_EXECUTION_MIN_COVERAGE = 0.80
DEFAULT_SAMPLE_EXECUTION_MAX_ATTEMPTS = 3
COVERAGE_UNREACHABLE_TERMINAL_REASON = "coverage_unreachable"
COVERAGE_UNREACHABLE_NOT_EXECUTED_FAILURE = (
    "coverage_unreachable_not_executed"
)
_MAX_PLAN_BYTES = 16_384
_MAX_SAMPLE_CONTEXT_BYTES = 32_768
_MAX_PUBLIC_STEPS = 8
_MAX_PERSISTED_RECORD_PREVIEW = 64
_FORBIDDEN_SAMPLE_CONTEXT_KEYS = frozenset(
    {
        "actual",
        "actual_value",
        "ground_truth",
        "label",
        "labels",
        "observed",
        "observation",
        "predicted",
        "prediction",
        "proposed_prediction",
        "target_value",
    }
)
_FORBIDDEN_SAMPLE_CONTEXT_TOKENS = frozenset(
    "".join(character for character in name if character.isalnum())
    for name in _FORBIDDEN_SAMPLE_CONTEXT_KEYS
)
_SAMPLE_CHECKPOINT_SCHEMA_VERSION = "ecologyrsi-dsh.sample-checkpoint/1"
SAMPLE_AGENT_CHAIN_ATTESTATION_VERSION = (
    "ecologyrsi-dsh.sample-agent-chain-attestation/3"
)


class SampleExecutionContractError(ValueError):
    """An adapter returned an invalid public sample result."""


class SampleResultCallbackError(RuntimeError):
    """The host could not durably publish a finalized sample-result batch."""


class SampleExecutionControlError(RuntimeError):
    """A run-level control signal that must cross sample failure isolation."""


class SampleExecutionPausedError(SampleExecutionControlError):
    """The owning run was paused while sample execution was active."""

    run_status = "paused"
    retryable = False


class SampleExecutionCancelledError(SampleExecutionControlError):
    """The owning run was cancelled while sample execution was active."""

    run_status = "cancelled"
    retryable = False


class SampleExecutionControlUnavailableError(SampleExecutionControlError):
    """The run-control state could not be read safely."""

    run_status = "unavailable"
    retryable = False


class SampleRepairRequired(SampleExecutionContractError):
    """A public critic rejected an attempt and requested another tool path."""

    def __init__(
        self,
        message: str,
        *,
        agent_decisions: Sequence[Mapping[str, Any]],
        tool_calls: Sequence[Mapping[str, Any]],
        requested_tool_id: str | None = None,
        previous_prediction: float | None = None,
    ) -> None:
        super().__init__(message)
        self.agent_decisions = tuple(dict(item) for item in agent_decisions)
        self.tool_calls = tuple(dict(item) for item in tool_calls)
        if requested_tool_id is not None:
            if not isinstance(requested_tool_id, str) or not requested_tool_id.strip():
                raise ValueError("requested repair tool id must be non-empty text")
            requested_tool_id = requested_tool_id.strip()[:160]
        self.requested_tool_id = requested_tool_id
        self.previous_prediction = _optional_finite_prediction(
            previous_prediction, "previous_prediction"
        )


class SampleExecutionAttemptError(RuntimeError):
    """One sample attempt failed after producing bounded public evidence."""

    def __init__(
        self,
        message: str,
        *,
        failure_class: str,
        retryable: bool,
        error_type: str,
        agent_decisions: Sequence[Mapping[str, Any]] = (),
        tool_calls: Sequence[Mapping[str, Any]] = (),
        requested_tool_id: str | None = None,
        previous_prediction: float | None = None,
        attempted: bool = True,
    ) -> None:
        super().__init__(message)
        self.failure_class = str(failure_class)[:120]
        self.retryable = bool(retryable)
        self.error_type = str(error_type)[:120]
        self.agent_decisions = tuple(dict(item) for item in agent_decisions)
        self.tool_calls = tuple(dict(item) for item in tool_calls)
        if not isinstance(attempted, bool):
            raise TypeError("sample execution attempted must be a boolean")
        self.attempted = attempted
        if requested_tool_id is not None:
            if not isinstance(requested_tool_id, str) or not requested_tool_id.strip():
                raise ValueError("requested sample tool id must be non-empty text")
            requested_tool_id = requested_tool_id.strip()[:160]
        self.requested_tool_id = requested_tool_id
        self.previous_prediction = _optional_finite_prediction(
            previous_prediction, "previous_prediction"
        )


class SamplePredictionAdapter(Protocol):
    """Injectable collaboration boundary used by real or fake agents."""

    adapter_id: str
    adapter_version: str

    def plan_batch(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return one label-free plan that can be reused for all samples."""

    def predict_sample(
        self,
        request: SamplePredictionRequest,
        plan: Mapping[str, Any],
        *,
        attempt: int,
    ) -> Mapping[str, Any]:
        """Predict one sample without access to its observed target."""


@dataclass(frozen=True, slots=True)
class SamplePredictionOutcome:
    """One adapter result in a microbatch, isolated from sibling failures."""

    sample_id: str
    result: Mapping[str, Any] | None = None
    error: BaseException | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample outcome sample_id must be non-empty text")
        if (self.result is None) == (self.error is None):
            raise ValueError("sample outcome requires exactly one result or error")
        if self.result is not None:
            object.__setattr__(self, "result", dict(self.result))
        if self.terminal_reason is not None:
            if (
                not isinstance(self.terminal_reason, str)
                or not self.terminal_reason.strip()
            ):
                raise ValueError(
                    "sample outcome terminal_reason must be non-empty text"
                )
            object.__setattr__(
                self, "terminal_reason", self.terminal_reason.strip()[:120]
            )


class BatchedSamplePredictionAdapter(SamplePredictionAdapter, Protocol):
    """Optional adapter extension used to microbatch remote sample decisions."""

    def predict_samples(
        self,
        requests: Sequence[SamplePredictionRequest],
        plans: Sequence[Mapping[str, Any]],
        *,
        attempts: Sequence[int],
    ) -> Sequence[SamplePredictionOutcome]:
        """Return one isolated outcome for every request in input order."""


@dataclass(frozen=True, slots=True)
class SampleExecutionPolicy:
    max_attempts: int = DEFAULT_SAMPLE_EXECUTION_MAX_ATTEMPTS
    plan_max_attempts: int = DEFAULT_SAMPLE_EXECUTION_MAX_ATTEMPTS
    minimum_coverage: float = DEFAULT_SAMPLE_EXECUTION_MIN_COVERAGE
    minimum_task_coverage: float = DEFAULT_SAMPLE_EXECUTION_MIN_COVERAGE
    retry_backoff_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in ("max_attempts", "plan_max_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"sample execution {name} must be an integer")
            if not 1 <= value <= 8:
                raise ValueError(f"sample execution {name} must be between 1 and 8")
        for name in ("minimum_coverage", "minimum_task_coverage"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"sample execution {name} must be a number")
            if not math.isfinite(float(value)) or not 0 < float(value) <= 1:
                raise ValueError(f"sample execution {name} must be in (0, 1]")
            object.__setattr__(self, name, float(value))
        delay = self.retry_backoff_seconds
        if isinstance(delay, bool) or not isinstance(delay, (int, float)):
            raise TypeError("sample execution retry_backoff_seconds must be a number")
        if not math.isfinite(float(delay)) or not 0 <= float(delay) <= 60:
            raise ValueError(
                "sample execution retry_backoff_seconds must be between 0 and 60"
            )
        object.__setattr__(self, "retry_backoff_seconds", float(delay))

    @classmethod
    def from_mapping(cls, value: Any) -> SampleExecutionPolicy:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("sample_execution_policy must be an object")
        allowed = {
            "max_attempts",
            "plan_max_attempts",
            "minimum_coverage",
            "minimum_task_coverage",
            "retry_backoff_seconds",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "sample_execution_policy contains unsupported fields: "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        data = dict(value)
        if "plan_max_attempts" not in data and "max_attempts" in data:
            data["plan_max_attempts"] = data["max_attempts"]
        if "minimum_task_coverage" not in data and "minimum_coverage" in data:
            data["minimum_task_coverage"] = data["minimum_coverage"]
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "plan_max_attempts": self.plan_max_attempts,
            "minimum_coverage": self.minimum_coverage,
            "minimum_task_coverage": self.minimum_task_coverage,
            "retry_backoff_seconds": self.retry_backoff_seconds,
        }


@dataclass(frozen=True, slots=True)
class SamplePredictionRequest:
    """Label-free input exposed to a sample prediction adapter."""

    sample_id: str
    candidate_id: str
    dataset_digest: str
    partition: str
    target: str
    unit: str
    horizon_hours: int
    origin_timestamp: str | int | float
    target_timestamp: str | int | float
    baseline: float
    # Compatibility-only host reference for legacy/injected adapters. New
    # gateway runs leave this unset and it is never serialized to remote agents.
    proposed_prediction: float | None
    minimum: float
    maximum: float
    algorithm_id: str
    algorithm_version: str
    label_free_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "candidate_id",
            "dataset_digest",
            "partition",
            "target",
            "unit",
            "algorithm_id",
            "algorithm_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"sample request {name} must be non-empty text")
        if (
            isinstance(self.horizon_hours, bool)
            or not isinstance(self.horizon_hours, int)
            or self.horizon_hours < 1
        ):
            raise ValueError("sample request horizon_hours must be a positive integer")
        for name in ("baseline", "minimum", "maximum"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"sample request {name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"sample request {name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.proposed_prediction is not None:
            if isinstance(self.proposed_prediction, bool) or not isinstance(
                self.proposed_prediction, (int, float)
            ):
                raise TypeError("sample request proposed_prediction must be numeric or null")
            if not math.isfinite(float(self.proposed_prediction)):
                raise ValueError("sample request proposed_prediction must be finite")
            object.__setattr__(
                self, "proposed_prediction", float(self.proposed_prediction)
            )
        context = _label_free_mapping(
            self.label_free_context,
            "sample request label_free_context",
        )
        object.__setattr__(self, "label_free_context", context)

    def to_dict(self) -> dict[str, Any]:
        # There is intentionally no observed/label field in this contract.
        return {
            "sample_id": self.sample_id,
            "candidate_id": self.candidate_id,
            "dataset_digest": self.dataset_digest,
            "partition": self.partition,
            "target": self.target,
            "unit": self.unit,
            "horizon_hours": self.horizon_hours,
            "origin_timestamp": self.origin_timestamp,
            "target_timestamp": self.target_timestamp,
            "baseline": self.baseline,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "label_free_context": dict(self.label_free_context),
        }


def _legacy_registered_prediction(request: SamplePredictionRequest) -> float:
    if request.proposed_prediction is None:
        raise SampleExecutionContractError(
            "sample requires an explicit registered forecast tool"
        )
    return request.proposed_prediction


@dataclass(frozen=True, slots=True)
class SampleExecutionBatch:
    successful_rows: tuple[dict[str, Any], ...]
    scoring_rows: tuple[dict[str, Any], ...]
    records: tuple[dict[str, Any], ...]
    summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _StrictOriginBundle:
    """One atomic forecast origin containing all target/horizon cells."""

    origin_sample_id: str
    rows: tuple[dict[str, Any], ...]
    requests: tuple[SamplePredictionRequest, ...]
    observed: tuple[float, ...]


class _RegisteredSampleToolRuntime:
    """Callable host tools used by the public sample-agent state machine."""

    def __init__(
        self,
        forecast_tool: Callable[[SamplePredictionRequest], float] | None = None,
    ) -> None:
        self._forecast_tool = forecast_tool or (
            _legacy_registered_prediction
        )

    def forecast(self, request: SamplePredictionRequest) -> float:
        return _finite_float(
            self._forecast_tool(request),
            "registered forecast tool output",
        )

    @staticmethod
    def physical_range_check(
        request: SamplePredictionRequest, predicted: float
    ) -> bool:
        return request.minimum <= predicted <= request.maximum

    def bounded_projection(self, request: SamplePredictionRequest) -> float:
        proposed = self.forecast(request)
        return min(
            request.maximum,
            max(request.minimum, proposed),
        )

    @staticmethod
    def bounded_persistence(request: SamplePredictionRequest) -> float:
        return min(
            request.maximum,
            max(request.minimum, request.baseline),
        )


class _ForecastAgent:
    def __init__(self, tools: _RegisteredSampleToolRuntime) -> None:
        self.tools = tools

    def predict(
        self, request: SamplePredictionRequest
    ) -> tuple[float, dict[str, str], dict[str, str]]:
        predicted = self.tools.forecast(request)
        return (
            predicted,
            {
                "role": "forecast_agent",
                "decision": "use_registered_algorithm_prediction",
                "status": "completed",
            },
            {
                "tool_id": request.algorithm_id,
                "version": request.algorithm_version,
                "status": "completed",
            },
        )


class _ConstraintCriticAgent:
    def __init__(self, tools: _RegisteredSampleToolRuntime) -> None:
        self.tools = tools

    def review(
        self,
        request: SamplePredictionRequest,
        predicted: float,
        *,
        unresolved_projection: bool,
    ) -> tuple[bool, dict[str, str], dict[str, str]]:
        in_range = self.tools.physical_range_check(request, predicted)
        accepted = in_range and not unresolved_projection
        decision = (
            "reject_boundary_projection_as_unresolved"
            if unresolved_projection
            else "prediction_within_registered_range"
            if in_range
            else "reject_prediction_outside_registered_range"
        )
        return (
            accepted,
            {
                "role": "constraint_critic",
                "decision": decision,
                "status": "completed",
            },
            {
                "tool_id": "physical-range-check",
                "version": "1",
                "status": "completed" if accepted else "rejected",
            },
        )


class _RepairAgent:
    def __init__(self, tools: _RegisteredSampleToolRuntime) -> None:
        self.tools = tools

    def repair(
        self,
        request: SamplePredictionRequest,
        *,
        attempt: int,
        repair_tool: str | None = None,
    ) -> tuple[float, dict[str, str], dict[str, str], bool]:
        if repair_tool is None:
            repair_tool = (
                "bounded-projection-repair"
                if attempt == 2
                else "bounded-persistence-fallback"
            )
        if repair_tool not in {
            "bounded-projection-repair",
            "bounded-persistence-fallback",
        }:
            raise SampleExecutionContractError(
                "derived plan selected an unregistered repair tool"
            )
        use_projection = repair_tool == "bounded-projection-repair"
        if use_projection:
            predicted = self.tools.bounded_projection(request)
            decision = "project_prediction_to_registered_range"
            tool_id = "bounded-projection-repair"
        else:
            predicted = self.tools.bounded_persistence(request)
            decision = "replace_with_bounded_persistence_fallback"
            tool_id = "bounded-persistence-fallback"
        unresolved_projection = (
            use_projection
            and request.proposed_prediction is not None
            and not (
                request.minimum
                <= request.proposed_prediction
                <= request.maximum
            )
            and predicted in {request.minimum, request.maximum}
        )
        return (
            predicted,
            {
                "role": "repair_agent",
                "decision": decision,
                "status": "completed",
            },
            {"tool_id": tool_id, "version": "1", "status": "completed"},
            unresolved_projection,
        )


class RegisteredToolCollaborationAdapter:
    """Default host collaboration around a registered numerical predictor.

    The forecasting agent invokes the fitted algorithm, a critic checks the
    physical range, and the host adjudicator records the final decision.  A
    remote multi-agent implementation can be injected behind the same
    boundary without changing evaluator or persistence code.
    """

    adapter_id = "registered-tool-collaboration"
    adapter_version = "2"

    def __init__(
        self,
        *,
        forecast_tool: Callable[[SamplePredictionRequest], float] | None = None,
    ) -> None:
        self.tools = _RegisteredSampleToolRuntime(forecast_tool)
        self.forecast_agent = _ForecastAgent(self.tools)
        self.constraint_critic = _ConstraintCriticAgent(self.tools)
        self.repair_agent = _RepairAgent(self.tools)

    def plan_batch(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        raw_execution_plan = context.get("derived_execution_plan")
        execution_plan = (
            DerivedExecutionPlan.from_dict(raw_execution_plan).to_dict()
            if isinstance(raw_execution_plan, Mapping)
            else None
        )
        plan = {
            "plan_id": "registered-predict-critic-adjudicate@2",
            "roles": [
                "forecast_agent",
                "constraint_critic",
                "failure_analyst",
                "repair_agent",
                "host_adjudicator",
            ],
            "algorithm_id": context.get("algorithm_id"),
            "algorithm_version": context.get("algorithm_version"),
            "tools": [
                {
                    "tool_id": context.get("algorithm_id"),
                    "version": context.get("algorithm_version"),
                },
                {"tool_id": "physical-range-check", "version": "1"},
                {"tool_id": "bounded-projection-repair", "version": "1"},
                {"tool_id": "bounded-persistence-fallback", "version": "1"},
            ],
            "decisions": {
                "valid": "accept_registered_algorithm_prediction",
                "invalid": "classify_failure_then_select_another_registered_tool",
                "failed": "skip_after_retry_budget",
            },
            "workflow": {
                "entrypoint": "forecast_agent",
                "transitions": [
                    "forecast_agent -> constraint_critic",
                    "constraint_critic:accepted -> host_adjudicator",
                    "constraint_critic:rejected -> failure_analyst",
                    "failure_analyst:retryable -> repair_agent",
                    "repair_agent -> constraint_critic",
                    "failure_analyst:exhausted -> host_adjudicator",
                ],
                "routing": "critic_feedback_and_retry_budget",
            },
            "decision_policy": "feedback_driven_retry_repair_or_skip",
            "agent_runtime": "host_feedback_state_machine",
            "remote_sample_agents": False,
            "forecast_value_source": "registered_predictor_tool_output",
        }
        if execution_plan is not None:
            plan["derived_execution_plan"] = execution_plan
        return plan

    def predict_sample(
        self,
        request: SamplePredictionRequest,
        plan: Mapping[str, Any],
        *,
        attempt: int,
    ) -> Mapping[str, Any]:
        retry_feedback = plan.get("sample_retry_feedback")
        if attempt == 1:
            predicted, decision, tool_call = self.forecast_agent.predict(request)
            decisions = [decision]
            tools = [tool_call]
            unresolved_projection = False
        else:
            # The retry path is selected only after the previous public critic
            # feedback. Attempt two projects the algorithm result onto the
            # registered physical domain; later attempts use the more
            # conservative persistence tool.
            if not isinstance(retry_feedback, (list, tuple)) or not retry_feedback:
                raise SampleExecutionContractError(
                    "repair attempt requires bounded public failure feedback"
                )
            execution_plan = plan.get("derived_execution_plan")
            repair_sequence = (
                DerivedExecutionPlan.from_dict(execution_plan).repair_sequence
                if isinstance(execution_plan, Mapping)
                else (
                    "bounded-projection-repair",
                    "bounded-persistence-fallback",
                )
            )
            repair_index = min(attempt - 2, len(repair_sequence) - 1)
            (
                predicted,
                decision,
                tool_call,
                unresolved_projection,
            ) = self.repair_agent.repair(
                request,
                attempt=attempt,
                repair_tool=repair_sequence[repair_index],
            )
            decisions = [decision]
            tools = [tool_call]

        accepted, critic_decision, critic_tool_call = self.constraint_critic.review(
            request,
            predicted,
            unresolved_projection=unresolved_projection,
        )
        decisions.append(critic_decision)
        tools.append(critic_tool_call)
        if not accepted:
            raise SampleRepairRequired(
                (
                    "boundary projection requires a conservative repair"
                    if unresolved_projection
                    else "registered prediction is outside the physical range"
                ),
                agent_decisions=decisions,
                tool_calls=tools,
            )
        return {
            "predicted": predicted,
            "agent_decisions": decisions,
            "tool_calls": tools,
        }


class CollaborativeSampleExecutor:
    """Execute and audit one independent prediction decision per sample."""

    def __init__(
        self,
        adapter: SamplePredictionAdapter | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        origin_admission: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> None:
        if origin_admission is not None and not callable(origin_admission):
            raise TypeError("origin_admission must be callable")
        self.adapter = adapter or RegisteredToolCollaborationAdapter()
        self._sleep = sleep
        self.origin_admission = origin_admission

    def execute(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        context: Mapping[str, Any],
        target_bounds: Mapping[str, Mapping[str, Any]],
        algorithm_id: str,
        algorithm_version: str,
        policy: SampleExecutionPolicy | None = None,
        result_callback: Callable[
            [Sequence[Mapping[str, Any]]], None
        ]
        | None = None,
        checkpoint_callback: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        result_batch_size: int = 50,
    ) -> SampleExecutionBatch:
        policy = policy or SampleExecutionPolicy()
        if result_callback is not None and not callable(result_callback):
            raise TypeError("result_callback must be callable")
        if checkpoint_callback is not None and not callable(checkpoint_callback):
            raise TypeError("checkpoint_callback must be callable")
        if (
            isinstance(result_batch_size, bool)
            or not isinstance(result_batch_size, int)
            or not 1 <= result_batch_size <= 200
        ):
            raise ValueError("result_batch_size must be an integer between 1 and 200")
        indexed_rows: list[dict[str, Any]] = []
        seen_sample_indices: set[int] = set()
        for ordinal, raw_row in enumerate(rows, start=1):
            indexed = dict(raw_row)
            raw_index = indexed.get("sample_index", ordinal)
            if (
                isinstance(raw_index, bool)
                or not isinstance(raw_index, int)
                or raw_index < 1
            ):
                raise ValueError("sample_index must be a positive integer")
            if raw_index in seen_sample_indices:
                raise ValueError("sample_index must be unique within the evaluation cohort")
            indexed["sample_index"] = raw_index
            indexed_rows.append(indexed)
            seen_sample_indices.add(raw_index)
        rows = tuple(indexed_rows)
        context_data = dict(context)
        # The executor owns coverage gates. Batched adapters may use this
        # host-only policy to stop remote work once the fixed cohort can no
        # longer pass, but it is never part of a sample request.
        context_data["sample_execution_policy"] = policy.to_dict()
        canonical_json(context_data)
        plan, plan_attempts, plan_failure = self._prepare_plan(context_data, policy)
        plan_digest = digest(plan) if plan is not None else None
        sample_agent_protocol = (
            str(plan.get("sample_agent_protocol")) if plan is not None else ""
        )
        strict_origin_contract = is_strict_origin_protocol(sample_agent_protocol)
        strict_agent_contract = strict_origin_contract
        required_success_remote_roles: tuple[str, ...] = ()
        if strict_agent_contract:
            raw_required_roles = plan.get("required_success_remote_roles")
            if raw_required_roles is None:
                # Historical strict adapters predate the explicit role list.
                required_success_remote_roles = (
                    "planner",
                    "critic",
                    "reflector",
                )
            elif (
                not isinstance(raw_required_roles, (list, tuple))
                or not raw_required_roles
                or any(
                    role not in {"planner", "critic", "reflector"}
                    for role in raw_required_roles
                )
                or len(set(raw_required_roles)) != len(raw_required_roles)
                or "planner" not in raw_required_roles
            ):
                raise SampleExecutionContractError(
                    "strict origin required remote roles are invalid"
                )
            else:
                required_success_remote_roles = tuple(raw_required_roles)
        concurrent_origin_contract = supports_concurrent_origins(
            sample_agent_protocol
        )
        origin_bundles: tuple[_StrictOriginBundle, ...] = ()
        origin_bundle_by_sample_id: dict[str, _StrictOriginBundle] = {}
        if strict_origin_contract and rows:
            origin_bundles = _strict_origin_bundles(
                rows,
                context=context_data,
                target_bounds=target_bounds,
                algorithm_id=algorithm_id,
                algorithm_version=algorithm_version,
            )
            rows = tuple(row for bundle in origin_bundles for row in bundle.rows)
            origin_bundle_by_sample_id = {
                request.sample_id: bundle
                for bundle in origin_bundles
                for request in bundle.requests
            }
        checkpoint = {
            "schema_version": _SAMPLE_CHECKPOINT_SCHEMA_VERSION,
            "cohort_digest": digest(
                _checkpoint_digest_value({
                    "rows": rows,
                    "target_bounds": dict(target_bounds),
                })
            ),
            "execution_context_digest": digest(
                {
                    "schema_version": _SAMPLE_CHECKPOINT_SCHEMA_VERSION,
                    "context": context_data,
                    "target_bounds": dict(target_bounds),
                    "algorithm_id": algorithm_id,
                    "algorithm_version": algorithm_version,
                    "policy": policy.to_dict(),
                    "adapter": _adapter_identity(self.adapter),
                    "batch_plan_digest": plan_digest,
                }
            ),
            "sample_count": len(rows),
        }
        checkpoint_response: Mapping[str, Any] = {}
        if checkpoint_callback is not None and rows:
            raw_checkpoint_response = checkpoint_callback(checkpoint)
            if not isinstance(raw_checkpoint_response, Mapping):
                raise SampleExecutionContractError(
                    "sample checkpoint callback must return an object"
                )
            checkpoint_response = dict(raw_checkpoint_response)
        resumed_rows = _validated_checkpoint_rows(
            checkpoint_response.get("rows", ()),
            rows=rows,
            context=context_data,
            target_bounds=target_bounds,
            algorithm_id=algorithm_id,
            algorithm_version=algorithm_version,
        )
        if strict_origin_contract:
            for bundle in origin_bundles:
                resumed_count = sum(
                    request.sample_id in resumed_rows
                    for request in bundle.requests
                )
                if resumed_count not in {0, len(bundle.requests)}:
                    raise SampleExecutionContractError(
                        "strict origin checkpoint contains a partial prediction vector"
                    )
        published_sample_ids: set[str] = set(resumed_rows)
        pending_rows = tuple(
            row
            for row in rows
            if _sample_id(
                row,
                context_data,
                target=str(row.get("target") or "").strip(),
                horizon=(
                    int(row.get("horizon_hours"))
                    if isinstance(row.get("horizon_hours"), int)
                    and not isinstance(row.get("horizon_hours"), bool)
                    else 0
                ),
            )
            not in resumed_rows
        )
        resume_checkpoint = _adapter_resume_checkpoint(
            rows,
            resumed_rows,
            checkpoint_response.get("progress"),
            force_full_cohort=strict_agent_contract,
            origin_bundles=origin_bundles,
        )
        outcome_setter = getattr(self.adapter, "set_outcome_callback", None)
        resume_setter = getattr(self.adapter, "set_resume_checkpoint", None)
        token_budget_setter = getattr(self.adapter, "set_token_budget_state", None)
        raw_rows_by_sample_id: dict[str, dict[str, Any]] = {}
        # A strict sample is durable only after the post-score remote
        # reflection has completed.  The adapter-level callback fires as soon
        # as Planner -> tool -> Critic returns, so using it here would allow a
        # crash to resume a prediction that has never crossed Reflector.
        # Strict runs therefore publish only the finalized row below; an
        # interrupted sample is deliberately executed again.
        use_preflection_publication = bool(
            result_callback is not None
            and callable(outcome_setter)
            and not strict_agent_contract
        )
        if use_preflection_publication:
            for raw_row in pending_rows:
                projected = dict(raw_row)
                try:
                    request, _observed = _request_from_row(
                        projected,
                        context=context_data,
                        target_bounds=target_bounds,
                        algorithm_id=algorithm_id,
                        algorithm_version=algorithm_version,
                    )
                except (SampleExecutionContractError, TypeError, ValueError):
                    continue
                raw_rows_by_sample_id[request.sample_id] = projected

            def publish_gateway_outcomes(
                requests: Sequence[SamplePredictionRequest],
                outcomes: Sequence[SamplePredictionOutcome],
                attempts: Sequence[int],
            ) -> Mapping[str, str]:
                finalized: list[dict[str, Any]] = []
                finalized_statuses: dict[str, str] = {}
                for request, outcome, attempt in zip(requests, outcomes, attempts):
                    if request.sample_id in published_sample_ids:
                        continue
                    source = raw_rows_by_sample_id.get(request.sample_id)
                    if source is None:
                        continue
                    failure = _prefetched_outcome_failure(outcome, request)
                    if failure is not None:
                        category, retryable, error_type = classify_sample_failure(
                            failure
                        )
                        attempt_executed = not (
                            isinstance(failure, SampleExecutionAttemptError)
                            and not failure.attempted
                        )
                        terminal = (
                            outcome.terminal_reason is not None
                            or not attempt_executed
                            or not retryable
                            or int(attempt) >= policy.max_attempts
                        )
                        if not terminal:
                            continue
                        completed_attempts = (
                            int(attempt)
                            if attempt_executed
                            else max(0, int(attempt) - 1)
                        )
                        record = _failed_record(
                            request,
                            observed=_finite_float(
                                source.get("observed"), "sample observed"
                            ),
                            attempts=completed_attempts,
                            category=category,
                            retryable=retryable,
                            error_type=error_type,
                            plan_digest=plan_digest,
                            adapter=self.adapter,
                        )
                        record["failure_summary"] = _sample_failure_summary(
                            request,
                            self.adapter,
                            category=category,
                            attempts=completed_attempts,
                            failure=failure,
                        )
                        finalized.append(
                            _fallback_scoring_row(source, request, record)
                        )
                        finalized_statuses[request.sample_id] = "failed"
                        continue
                    assert outcome.result is not None
                    result = _host_critic_result(
                        _validated_result(outcome.result), request
                    )
                    finalized_row = dict(source)
                    finalized_row["predicted"] = float(result["predicted"])
                    finalized_row["sample_id"] = request.sample_id
                    finalized_row["sample_execution_status"] = "succeeded"
                    finalized_row["sample_execution_attempts"] = int(attempt)
                    finalized_row["sample_execution_retry_count"] = max(
                        0, int(attempt) - 1
                    )
                    finalized_row["scoring_fallback"] = None
                    finalized.append(finalized_row)
                    finalized_statuses[request.sample_id] = "succeeded"
                if not finalized:
                    return {}
                try:
                    result_callback(tuple(finalized))
                except Exception as exc:  # noqa: BLE001 - preserve persistence cause
                    raise SampleResultCallbackError(
                        "finalized sample-result batch could not be persisted"
                    ) from exc
                published_sample_ids.update(finalized_statuses)
                return finalized_statuses

        try:
            if use_preflection_publication:
                outcome_setter(publish_gateway_outcomes)
            if callable(resume_setter):
                resume_setter(resume_checkpoint)
            if callable(token_budget_setter):
                token_budget_setter(checkpoint_response.get("token_budget_state"))
            if strict_agent_contract:
                # A strict sample is one transactional semantic chain.  Do not
                # prefetch Planner/Critic outcomes for the whole cohort and
                # postpone every Reflector until the end: a pause would then
                # discard hours of completed remote work.  The ordinary loop
                # below executes and checkpoints each complete chain before
                # beginning the next sample.
                terminal_reason = None
                prefetched_attempt_outcomes = {}
            else:
                first_attempt_outcomes = self._prepare_first_attempt_batch(
                    pending_rows,
                    context=context_data,
                    target_bounds=target_bounds,
                    algorithm_id=algorithm_id,
                    algorithm_version=algorithm_version,
                    plan=plan,
                )
                terminal_reason = _batch_terminal_reason(
                    first_attempt_outcomes.values()
                )
                prefetched_attempt_outcomes = self._prepare_retry_attempt_waves(
                    pending_rows,
                    context=context_data,
                    target_bounds=target_bounds,
                    algorithm_id=algorithm_id,
                    algorithm_version=algorithm_version,
                    plan=plan,
                    policy=policy,
                    first_attempt_outcomes=first_attempt_outcomes,
                )
        finally:
            if use_preflection_publication:
                outcome_setter(None)
            if callable(resume_setter) and not strict_agent_contract:
                resume_setter(None)
            if callable(token_budget_setter) and not strict_agent_contract:
                token_budget_setter(None)

        successful_rows: list[dict[str, Any]] = []
        scoring_rows: list[dict[str, Any]] = []
        finalized_result_rows: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        action_catalog: dict[str, dict[str, Any]] = {}
        task_counts: dict[tuple[str, int], dict[str, Any]] = {}
        failure_counts: dict[str, int] = {}
        repair_count = 0
        recovered_examples = 0
        exploration_failures = 0
        input_failures = 0
        scoring_fallback_examples = 0
        total_retries = max(0, plan_attempts - 1) + sum(
            int(row.get("retry_count", 0)) for row in resumed_rows.values()
        )
        critic_outcome_counts: dict[str, int] = {}
        reason_code_counts: dict[str, int] = {}
        repair_tool_outcomes: dict[str, dict[str, int]] = {}
        prepared_origin_ids: set[str] = set()
        prepared_origin_reflections: dict[str, dict[str, Any]] = {}
        origin_publication_rows: dict[str, list[dict[str, Any]]] = {}
        origin_executor: ThreadPoolExecutor | None = None
        origin_futures: dict[
            str,
            Future[
                tuple[
                    dict[tuple[str, int], SamplePredictionOutcome],
                    dict[str, dict[str, Any]],
                    str | None,
                ]
            ],
        ] = {}
        remaining_origin_bundles: Any = iter(())

        def submit_next_origin_bundle() -> bool:
            if origin_executor is None:
                return False
            try:
                bundle = next(remaining_origin_bundles)
            except StopIteration:
                return False
            origin_futures[bundle.origin_sample_id] = origin_executor.submit(
                self._prepare_strict_origin_bundle,
                bundle,
                context=context_data,
                target_bounds=target_bounds,
                algorithm_id=algorithm_id,
                algorithm_version=algorithm_version,
                plan=plan,
                policy=policy,
            )
            return True

        if concurrent_origin_contract and origin_bundles:
            pending_origin_bundles = tuple(
                bundle
                for bundle in origin_bundles
                if not all(
                    request.sample_id in resumed_rows
                    for request in bundle.requests
                )
            )
            configured_concurrency = int(
                context_data.get(
                    "sample_concurrency",
                    getattr(self.adapter, "sample_concurrency", 1),
                )
            )
            candidate_concurrency = max(
                1, int(context_data.get("candidate_concurrency", 1))
            )
            local_worker_count = min(
                len(pending_origin_bundles),
                max(1, math.ceil(configured_concurrency / candidate_concurrency)),
            )
            if local_worker_count:
                origin_executor = ThreadPoolExecutor(
                    max_workers=local_worker_count,
                    thread_name_prefix="dsh-origin",
                )
                remaining_origin_bundles = iter(pending_origin_bundles)
                for _ in range(local_worker_count):
                    submit_next_origin_bundle()

        def append_scoring_row(finalized: Mapping[str, Any]) -> None:
            """Publish only rows finalized by host validation or fallback."""

            projected = dict(finalized)
            sample_id = projected.get("sample_id")
            if strict_origin_contract and isinstance(sample_id, str):
                origin_bundle = origin_bundle_by_sample_id.get(sample_id)
                if origin_bundle is not None:
                    projected["origin_sample_id"] = (
                        origin_bundle.origin_sample_id
                    )
            scoring_rows.append(projected)
            if isinstance(sample_id, str) and sample_id in published_sample_ids:
                return
            if strict_origin_contract:
                if not isinstance(sample_id, str):
                    raise SampleExecutionContractError(
                        "strict origin result requires sample_id"
                    )
                bundle = origin_bundle_by_sample_id.get(sample_id)
                if bundle is None:
                    raise SampleExecutionContractError(
                        "strict origin result is outside the frozen bundle cohort"
                    )
                bucket = origin_publication_rows.setdefault(
                    bundle.origin_sample_id, []
                )
                bucket.append(projected)
                if len(bucket) > len(bundle.requests):
                    raise SampleExecutionContractError(
                        "strict origin result contains duplicate prediction cells"
                    )
                if len(bucket) < len(bundle.requests):
                    return
                bucket_ids = {str(item.get("sample_id")) for item in bucket}
                expected_ids = {request.sample_id for request in bundle.requests}
                if bucket_ids != expected_ids:
                    raise SampleExecutionContractError(
                        "strict origin result does not cover its complete prediction vector"
                    )
                if result_callback is not None:
                    result_callback(tuple(bucket))
                published_sample_ids.update(expected_ids)
                origin_publication_rows.pop(bundle.origin_sample_id, None)
                progress_recorder = getattr(
                    self.adapter, "record_finalized_origin_progress", None
                )
                if callable(progress_recorder):
                    progress_recorder(
                        status=(
                            "succeeded"
                            if all(
                                item.get("sample_execution_status") == "succeeded"
                                for item in bucket
                            )
                            else "failed"
                        ),
                        prediction_cell_count=len(bucket),
                    )
                if (
                    concurrent_origin_contract
                    and bundle.origin_sample_id in prepared_origin_ids
                ):
                    submit_next_origin_bundle()
                return
            if result_callback is not None:
                finalized_result_rows.append(projected)
                publication_batch_size = (
                    1 if strict_agent_contract else result_batch_size
                )
                if len(finalized_result_rows) >= publication_batch_size:
                    result_callback(tuple(finalized_result_rows))
                    finalized_result_rows.clear()
            if strict_agent_contract:
                progress_recorder = getattr(
                    self.adapter, "record_finalized_sample_progress", None
                )
                if callable(progress_recorder):
                    progress_recorder(
                        status=str(
                            projected.get("sample_execution_status") or "failed"
                        )
                    )

        for raw_row in rows:
            row = dict(raw_row)
            try:
                request, observed = _request_from_row(
                    row,
                    context=context_data,
                    target_bounds=target_bounds,
                    algorithm_id=algorithm_id,
                    algorithm_version=algorithm_version,
                )
            except (SampleExecutionContractError, TypeError, ValueError) as exc:
                input_failures += 1
                target = str(row.get("target") or "unknown")[:160]
                raw_horizon = row.get("horizon_hours")
                horizon = (
                    raw_horizon
                    if isinstance(raw_horizon, int) and not isinstance(raw_horizon, bool)
                    else 0
                )
                task_key = (target, horizon)
                task_count = task_counts.setdefault(
                    task_key,
                    {
                        "target": target,
                        "horizon_hours": horizon,
                        "attempted_examples": 0,
                        "succeeded_examples": 0,
                        "failed_examples": 0,
                        "retry_count": 0,
                    },
                )
                task_count["attempted_examples"] += 1
                task_count["failed_examples"] += 1
                record = _invalid_input_record(
                    row,
                    context=context_data,
                    error_type=type(exc).__name__,
                    plan_digest=plan_digest,
                )
                resumed = resumed_rows.get(str(record["sample_id"]))
                if resumed is not None:
                    checkpoint_request, _checkpoint_observed = (
                        _checkpoint_request_for_invalid_row(
                            row,
                            context=context_data,
                            target_bounds=target_bounds,
                            algorithm_id=algorithm_id,
                            algorithm_version=algorithm_version,
                        )
                    )
                    retry_count = int(resumed["retry_count"])
                    failure_class = str(
                        resumed.get("failure_class") or "invalid_sample_input"
                    )[:120]
                    resumed_record = {
                        "schema_version": SAMPLE_EXECUTION_SCHEMA_VERSION,
                        "sample_id": checkpoint_request.sample_id,
                        "status": "failed",
                        "attempts": int(resumed["attempts"]),
                        "retry_count": retry_count,
                        "predicted": float(resumed["predicted"]),
                        "batch_plan_digest": plan_digest,
                        "checkpoint_resumed": True,
                        "failure": {
                            "class": failure_class,
                            "retryable": False,
                            "error_type": "CheckpointResumedFailure",
                        },
                    }
                    records.append(resumed_record)
                    append_scoring_row(
                        _checkpoint_scoring_row(
                            row, checkpoint_request, resumed
                        )
                    )
                    task_count["retry_count"] += retry_count
                    if resumed.get("scoring_fallback") is not None:
                        scoring_fallback_examples += 1
                    failure_counts[failure_class] = (
                        failure_counts.get(failure_class, 0) + 1
                    )
                    continue
                records.append(record)
                fallback = _invalid_input_scoring_row(
                    row,
                    record,
                    target_bounds=target_bounds,
                )
                if fallback is not None:
                    append_scoring_row(fallback)
                    scoring_fallback_examples += 1
                failure_counts["invalid_sample_input"] = (
                    failure_counts.get("invalid_sample_input", 0) + 1
                )
                continue

            task_key = (request.target, request.horizon_hours)
            task_count = task_counts.setdefault(
                task_key,
                {
                    "target": request.target,
                    "horizon_hours": request.horizon_hours,
                    "attempted_examples": 0,
                    "succeeded_examples": 0,
                    "failed_examples": 0,
                    "retry_count": 0,
                },
            )
            task_count["attempted_examples"] += 1
            resumed = resumed_rows.get(request.sample_id)
            if resumed is not None:
                scoring_row = _checkpoint_scoring_row(row, request, resumed)
                status = str(resumed["status"])
                attempts = int(resumed["attempts"])
                retry_count = int(resumed["retry_count"])
                record: dict[str, Any] = {
                    "schema_version": SAMPLE_EXECUTION_SCHEMA_VERSION,
                    "sample_id": request.sample_id,
                    "status": status,
                    "attempts": attempts,
                    "retry_count": retry_count,
                    "predicted": float(resumed["predicted"]),
                    "batch_plan_digest": plan_digest,
                    "checkpoint_resumed": True,
                }
                if isinstance(resumed.get("sample_reflection"), Mapping):
                    record["sample_reflection"] = dict(
                        resumed["sample_reflection"]
                    )
                if isinstance(resumed.get("sample_agent_chain"), Mapping):
                    record["sample_agent_chain"] = dict(
                        resumed["sample_agent_chain"]
                    )
                if status == "succeeded":
                    task_count["succeeded_examples"] += 1
                    successful_rows.append(scoring_row)
                    if retry_count:
                        recovered_examples += 1
                else:
                    failure_class = str(
                        resumed.get("failure_class")
                        or "checkpoint_resumed_failure"
                    )[:120]
                    record["failure"] = {
                        "class": failure_class,
                        "retryable": False,
                        "error_type": "CheckpointResumedFailure",
                    }
                    task_count["failed_examples"] += 1
                    failure_counts[failure_class] = (
                        failure_counts.get(failure_class, 0) + 1
                    )
                    if resumed.get("scoring_fallback") is not None:
                        scoring_fallback_examples += 1
                task_count["retry_count"] += retry_count
                records.append(record)
                append_scoring_row(scoring_row)
                continue
            if strict_origin_contract:
                bundle = origin_bundle_by_sample_id.get(request.sample_id)
                if bundle is None:
                    raise SampleExecutionContractError(
                        "strict origin sample is outside the frozen bundle cohort"
                    )
                if bundle.origin_sample_id not in prepared_origin_ids:
                    if concurrent_origin_contract:
                        future = origin_futures.pop(
                            bundle.origin_sample_id, None
                        )
                        if future is None:
                            raise SampleExecutionContractError(
                                "strict origin scheduler lost a frozen origin"
                            )
                        try:
                            (
                                origin_outcomes,
                                origin_reflections,
                                origin_terminal_reason,
                            ) = future.result()
                        except BaseException:
                            if origin_executor is not None:
                                origin_executor.shutdown(
                                    wait=True, cancel_futures=True
                                )
                                origin_executor = None
                            raise
                    else:
                        (
                            origin_outcomes,
                            origin_reflections,
                            origin_terminal_reason,
                        ) = self._prepare_strict_origin_bundle(
                            bundle,
                            context=context_data,
                            target_bounds=target_bounds,
                            algorithm_id=algorithm_id,
                            algorithm_version=algorithm_version,
                            plan=plan,
                            policy=policy,
                        )
                    prefetched_attempt_outcomes.update(origin_outcomes)
                    prepared_origin_reflections.update(origin_reflections)
                    prepared_origin_ids.add(bundle.origin_sample_id)
                    if origin_terminal_reason is not None:
                        terminal_reason = origin_terminal_reason
            if plan is None:
                category, retryable, error_type = plan_failure or (
                    "batch_plan_failure",
                    False,
                    "UnknownPlanFailure",
                )
                reflection, reflection_decision = _run_sample_reflection(
                    self.adapter,
                    request,
                    observed=observed,
                    predicted=None,
                    status="failed",
                    failure_class="batch_plan_" + category,
                    agent_decisions=(),
                    tool_calls=(),
                )
                record = _failed_record(
                    request,
                    observed=observed,
                    attempts=0,
                    category="batch_plan_" + category,
                    retryable=retryable,
                    error_type=error_type,
                    plan_digest=None,
                    adapter=self.adapter,
                )
                if reflection is not None:
                    record["sample_reflection"] = reflection
                if reflection_decision is not None:
                    _attach_execution_trace(record, [reflection_decision], [])
                if strict_agent_contract:
                    record["sample_agent_chain"] = _sample_agent_chain_attestation(
                        record,
                        required_remote_roles=required_success_remote_roles,
                    )
                records.append(record)
                fallback_row = _fallback_scoring_row(row, request, record)
                if reflection is not None:
                    fallback_row["sample_reflection"] = reflection
                if strict_agent_contract:
                    fallback_row["sample_agent_chain"] = record[
                        "sample_agent_chain"
                    ]
                append_scoring_row(fallback_row)
                scoring_fallback_examples += 1
                task_count["failed_examples"] += 1
                failure_counts[record["failure"]["class"]] = (
                    failure_counts.get(record["failure"]["class"], 0) + 1
                )
                continue

            result: dict[str, Any] | None = None
            final_failure: tuple[str, bool, str] | None = None
            final_exception: BaseException | None = None
            failure_history: list[dict[str, Any]] = []
            prior_decisions: list[dict[str, str]] = []
            prior_tools: list[dict[str, str]] = []
            attempt_trace: list[dict[str, Any]] = []
            attempts = 0
            sample_attempt_limit = (
                1 if terminal_reason is not None else policy.max_attempts
            )
            for attempt in range(1, sample_attempt_limit + 1):
                attempts = attempt
                try:
                    attempt_plan = dict(plan)
                    if failure_history:
                        attempt_plan["sample_retry_feedback"] = [
                            dict(item) for item in failure_history
                        ]
                    prefetched = prefetched_attempt_outcomes.get(
                        (request.sample_id, attempt)
                    )
                    used_prefetched = prefetched is not None
                    if prefetched is not None:
                        if prefetched.error is not None:
                            raise prefetched.error
                        assert prefetched.result is not None
                        raw_result = prefetched.result
                    else:
                        raw_result = self.adapter.predict_sample(
                            request,
                            attempt_plan,
                            attempt=attempt,
                        )
                    result = _host_critic_result(
                        _validated_result(raw_result), request
                    )
                    final_failure = None
                    final_exception = None
                    break
                except Exception as exc:  # noqa: BLE001 - isolate third-party sample tools
                    final_failure = classify_sample_failure(exc)
                    final_exception = exc
                    category, retryable, error_type = final_failure
                    attempt_executed = not (
                        isinstance(exc, SampleExecutionAttemptError)
                        and not exc.attempted
                    )
                    if not attempt_executed:
                        attempts = max(0, attempt - 1)
                    else:
                        exploration_failures += 1
                    feedback, exception_decisions, exception_tools = (
                        _failure_feedback_from_exception(
                            exc, attempt if attempt_executed else 0
                        )
                    )
                    attempt_entry = _attempt_trace_entry(
                        attempt if attempt_executed else 0,
                        exception_decisions,
                        exception_tools,
                        outcome=(
                            "repair_requested"
                            if feedback.get("requested_tool_id")
                            else "failed_retryable"
                            if retryable and attempt < sample_attempt_limit
                            else "terminated"
                        ),
                        requested_tool_id=feedback.get("requested_tool_id"),
                    )
                    if attempt_entry is not None:
                        attempt_trace.append(attempt_entry)
                    _record_feedback_steps(
                        exception_decisions,
                        exception_tools,
                        critic_outcome_counts=critic_outcome_counts,
                        reason_code_counts=reason_code_counts,
                        repair_tool_outcomes=repair_tool_outcomes,
                    )
                    failure_history.append(feedback)
                    prior_decisions.extend(exception_decisions)
                    prior_tools.extend(exception_tools)
                    prior_decisions.append(
                        {
                            "role": "failure_analyst",
                            "decision": (
                                f"retry_after_{category}"
                                if retryable and attempt < sample_attempt_limit
                                else f"stop_after_{category}"
                            ),
                            "status": "completed",
                        }
                    )
                    if not exception_tools:
                        prior_tools.append(
                            {
                                "tool_id": _adapter_identity(self.adapter)["id"],
                                "version": _adapter_identity(self.adapter)["version"],
                                "status": "failed",
                            }
                        )
                    if (
                        not attempt_executed
                        or not final_failure[1]
                        or attempt >= sample_attempt_limit
                    ):
                        break
                    total_retries += 1
                    if not used_prefetched:
                        self._backoff(policy, attempt)

            if result is None:
                category, retryable, error_type = final_failure or (
                    "unknown_failure",
                    False,
                    "UnknownSampleFailure",
                )
                reflection, reflection_decision = _run_sample_reflection(
                    self.adapter,
                    request,
                    observed=observed,
                    predicted=None,
                    status="failed",
                    failure_class=category,
                    agent_decisions=prior_decisions,
                    tool_calls=prior_tools,
                    prepared_reflection=prepared_origin_reflections.get(
                        request.sample_id
                    ),
                )
                if reflection_decision is not None:
                    prior_decisions.append(reflection_decision)
                record = _failed_record(
                    request,
                    observed=observed,
                    attempts=attempts,
                    category=category,
                    retryable=retryable,
                    error_type=error_type,
                    plan_digest=plan_digest,
                    adapter=self.adapter,
                )
                record["failure_summary"] = _sample_failure_summary(
                    request,
                    self.adapter,
                    category=category,
                    attempts=attempts,
                    failure=final_exception,
                )
                failure_action = _sample_action(
                    request,
                    self.adapter,
                    agent_decisions=[
                        *prior_decisions,
                        {
                            "role": "host_adjudicator",
                            "decision": "skip_sample_after_retry_budget",
                            "status": "completed",
                        },
                    ],
                    tool_calls=prior_tools,
                )
                action_digest = digest(failure_action)
                action_catalog.setdefault(
                    action_digest,
                    {"action_digest": action_digest, **failure_action},
                )
                record["action_digest"] = action_digest
                if failure_history:
                    record["failure_history"] = [
                        dict(item) for item in failure_history
                    ]
                _attach_execution_trace(record, prior_decisions, prior_tools)
                if attempt_trace:
                    record["attempt_trace"] = attempt_trace
                if reflection is not None:
                    record["sample_reflection"] = reflection
                if strict_agent_contract:
                    record["sample_agent_chain"] = _sample_agent_chain_attestation(
                        record,
                        required_remote_roles=required_success_remote_roles,
                    )
                records.append(record)
                fallback_row = _fallback_scoring_row(row, request, record)
                if reflection is not None:
                    fallback_row["sample_reflection"] = reflection
                if strict_agent_contract:
                    fallback_row["sample_agent_chain"] = record[
                        "sample_agent_chain"
                    ]
                append_scoring_row(fallback_row)
                scoring_fallback_examples += 1
                task_count["failed_examples"] += 1
                task_count["retry_count"] += max(0, attempts - 1)
                failure_counts[category] = failure_counts.get(category, 0) + 1
                continue

            predicted = float(result["predicted"])
            decisions = _bounded_public_steps(
                [*prior_decisions, *result["agent_decisions"]]
            )
            tools = _bounded_public_steps(
                [*prior_tools, *result["tool_calls"]]
            )
            reflection, reflection_decision = _run_sample_reflection(
                self.adapter,
                request,
                observed=observed,
                predicted=predicted,
                status="succeeded",
                failure_class=None,
                agent_decisions=decisions,
                tool_calls=tools,
                prepared_reflection=prepared_origin_reflections.get(
                    request.sample_id
                ),
            )
            if reflection_decision is not None:
                decisions.append(reflection_decision)
            _record_feedback_steps(
                result["agent_decisions"],
                result["tool_calls"],
                critic_outcome_counts=critic_outcome_counts,
                reason_code_counts=reason_code_counts,
                repair_tool_outcomes=repair_tool_outcomes,
            )
            attempt_entry = _attempt_trace_entry(
                attempts,
                result["agent_decisions"],
                result["tool_calls"],
                outcome="accepted",
            )
            if attempt_entry is not None:
                attempt_trace.append(attempt_entry)
            if failure_history:
                recovered_examples += 1
            repaired = any(
                item.get("role") == "repair_agent"
                or "repair" in str(item.get("decision", ""))
                or "fallback" in str(item.get("decision", ""))
                for item in decisions
            )
            if repaired:
                repair_count += 1
                task_count["repair_count"] = int(
                    task_count.get("repair_count", 0)
                ) + 1
            adjudicator_role = (
                "host_adjudicator"
                if all(item["role"] != "host_adjudicator" for item in decisions)
                else "host_evidence_verifier"
            )
            decisions.append(
                {
                    "role": adjudicator_role,
                    "decision": "accept_validated_sample_prediction",
                    "status": "completed",
                }
            )
            action = _sample_action(
                request,
                self.adapter,
                agent_decisions=decisions,
                tool_calls=tools,
            )
            action_digest = digest(action)
            action_catalog.setdefault(
                action_digest,
                {"action_digest": action_digest, **action},
            )
            record = {
                "schema_version": SAMPLE_EXECUTION_SCHEMA_VERSION,
                "sample_id": request.sample_id,
                "status": "succeeded",
                "attempts": attempts,
                "retry_count": attempts - 1,
                "predicted": predicted,
                "batch_plan_digest": plan_digest,
                "action_digest": action_digest,
            }
            _attach_execution_trace(record, decisions, tools)
            if attempt_trace:
                record["attempt_trace"] = attempt_trace
            if reflection is not None:
                record["sample_reflection"] = reflection
            if strict_agent_contract:
                record["sample_agent_chain"] = _sample_agent_chain_attestation(
                    record,
                    required_remote_roles=required_success_remote_roles,
                )
            records.append(record)
            if failure_history:
                record["failure_history"] = [dict(item) for item in failure_history]
            task_count["succeeded_examples"] += 1
            task_count["retry_count"] += attempts - 1
            executed_row = dict(row)
            executed_row["predicted"] = predicted
            executed_row["sample_id"] = request.sample_id
            executed_row["sample_execution_status"] = "succeeded"
            executed_row["sample_execution_attempts"] = attempts
            executed_row["sample_execution_retry_count"] = attempts - 1
            executed_row["action_digest"] = action_digest
            executed_row["scoring_fallback"] = None
            if reflection is not None:
                executed_row["sample_reflection"] = reflection
            if strict_agent_contract:
                executed_row["sample_agent_chain"] = record[
                    "sample_agent_chain"
                ]
            successful_rows.append(executed_row)
            append_scoring_row(executed_row)

        if strict_origin_contract and origin_publication_rows:
            raise SampleExecutionContractError(
                "strict origin publication ended with an incomplete prediction vector"
            )
        if origin_executor is not None:
            origin_executor.shutdown(wait=True, cancel_futures=False)
            origin_executor = None
        if result_callback is not None and finalized_result_rows:
            result_callback(tuple(finalized_result_rows))
            finalized_result_rows.clear()

        attempted = len(rows)
        succeeded = len(successful_rows)
        failed = attempted - succeeded
        coverage = succeeded / attempted if attempted else 0.0
        trace_digest = digest(records)
        task_summaries = []
        for task_count in task_counts.values():
            task_attempted = int(task_count["attempted_examples"])
            task_succeeded = int(task_count["succeeded_examples"])
            task_summaries.append(
                {
                    **task_count,
                    "coverage": (
                        task_succeeded / task_attempted if task_attempted else 0.0
                    ),
                }
            )
        public_plan = dict(plan) if plan is not None else None
        plan_mode = (
            str(public_plan.get("execution_mode"))
            if public_plan is not None and public_plan.get("execution_mode")
            else "per_sample_host_multi_agent_tool_state_machine"
        )
        remote_sample_agents = bool(
            public_plan is not None and public_plan.get("remote_sample_agents") is True
        )
        feedback_aggregates = _feedback_aggregates(
            records,
            critic_outcome_counts=critic_outcome_counts,
            reason_code_counts=reason_code_counts,
            repair_tool_outcomes=repair_tool_outcomes,
        )
        tool_performance = summarize_tool_performance(records, scoring_rows)
        remote_planner_invocations = 0
        remote_critic_invocations = 0
        remote_reflection_invocations = 0
        host_route_bypass_count = 0
        complete_agent_chains = 0
        counted_origin_invocations: set[str] = set()
        registered_tool_invocation_keys: set[tuple[str, str, str]] = set()
        dsh_agent_tool_event_ids: set[str] = set()
        for record in records:
            for tool in record.get("tool_trace", ()):
                if (
                    isinstance(tool, Mapping)
                    and tool.get("status") == "completed"
                    and tool.get("tool_id") != "physical-range-check"
                ):
                    registered_tool_invocation_keys.add(
                        (
                            str(tool.get("tool_id") or ""),
                            str(tool.get("input_digest") or ""),
                            str(tool.get("output_digest") or ""),
                        )
                    )
                    if tool.get("execution_owner") == "dsh_agent_tool_call":
                        event_id = tool.get("dsh_tool_event_id")
                        if isinstance(event_id, str) and event_id:
                            dsh_agent_tool_event_ids.add(event_id)
            raw_reflection = record.get("sample_reflection")
            record_bundle = origin_bundle_by_sample_id.get(
                str(record.get("sample_id") or "")
            )
            origin_sample_id = (
                str(raw_reflection.get("origin_sample_id"))
                if strict_origin_contract
                and isinstance(raw_reflection, Mapping)
                and raw_reflection.get("origin_sample_id")
                else record_bundle.origin_sample_id
                if strict_origin_contract and record_bundle is not None
                else None
            )
            count_invocations = bool(
                origin_sample_id is None
                or origin_sample_id not in counted_origin_invocations
            )
            if origin_sample_id is not None:
                counted_origin_invocations.add(origin_sample_id)
            attestation = record.get("sample_agent_chain")
            if isinstance(attestation, Mapping):
                if count_invocations:
                    remote_planner_invocations += int(
                        attestation.get("planner_invocations", 0)
                    )
                    remote_critic_invocations += int(
                        attestation.get("critic_invocations", 0)
                    )
                    remote_reflection_invocations += int(
                        attestation.get("reflector_invocations", 0)
                    )
                    host_route_bypass_count += int(
                        attestation.get("host_route_bypass_count", 0)
                    )
                complete_agent_chains += int(
                    attestation.get("complete") is True
                )
                continue
            role_list = [
                str(item.get("role") or "")
                for item in record.get("agent_trace", ())
                if isinstance(item, Mapping)
            ]
            roles = set(role_list)
            if count_invocations:
                remote_planner_invocations += sum(
                    role == "remote_planner_agent" for role in role_list
                )
                remote_critic_invocations += sum(
                    role == "remote_critic_agent" for role in role_list
                )
                remote_reflection_invocations += sum(
                    role == "remote_reflector_agent" for role in role_list
                )
                host_route_bypass_count += sum(
                    role == "host_deterministic_router" for role in role_list
                )
            if (
                "remote_planner_agent" in roles
                and "remote_critic_agent" in roles
                and isinstance(record.get("sample_reflection"), Mapping)
            ):
                complete_agent_chains += 1
        strict_chain_denominator = max(0, attempted - input_failures)
        strict_agent_chain_coverage = (
            complete_agent_chains / strict_chain_denominator
            if strict_chain_denominator
            else 0.0
        )
        strict_agent_chain_pass = bool(
            not strict_agent_contract
            or (
                strict_chain_denominator == attempted
                and complete_agent_chains == attempted
                and host_route_bypass_count == 0
            )
        )
        succeeded_sample_ids = {
            str(row.get("sample_id"))
            for row in successful_rows
            if isinstance(row.get("sample_id"), str)
        }
        complete_chain_sample_ids = {
            str(record.get("sample_id"))
            for record in records
            if (
                isinstance(record.get("sample_agent_chain"), Mapping)
                and record["sample_agent_chain"].get("complete") is True
            )
            or (
                isinstance(record.get("sample_reflection"), Mapping)
                and any(
                    item.get("role") == "remote_planner_agent"
                    for item in record.get("agent_trace", ())
                    if isinstance(item, Mapping)
                )
                and any(
                    item.get("role") == "remote_critic_agent"
                    for item in record.get("agent_trace", ())
                    if isinstance(item, Mapping)
                )
            )
        }
        succeeded_origins = sum(
            all(
                request.sample_id in succeeded_sample_ids
                for request in bundle.requests
            )
            for bundle in origin_bundles
        )
        complete_origin_agent_chains = sum(
            all(
                request.sample_id in complete_chain_sample_ids
                for request in bundle.requests
            )
            for bundle in origin_bundles
        )
        summary = {
            "schema_version": SAMPLE_EXECUTION_SCHEMA_VERSION,
            "mode": plan_mode,
            "eligible_examples": attempted,
            "attempted_examples": attempted,
            "succeeded_examples": succeeded,
            "failed_examples": failed,
            "skipped_examples": failed,
            "scoring_fallback_examples": scoring_fallback_examples,
            "coverage": coverage,
            "minimum_coverage": policy.minimum_coverage,
            "minimum_task_coverage": policy.minimum_task_coverage,
            "coverage_pass": attempted > 0 and coverage >= policy.minimum_coverage,
            "coverage_early_stopped": terminal_reason is not None,
            "early_stop_reason": terminal_reason,
            "retry_count": total_retries,
            "checkpoint_resumed_examples": len(resumed_rows),
            "checkpoint_pending_examples": len(pending_rows),
            "exploration_failures": exploration_failures,
            "recovered_examples": recovered_examples,
            "input_failures": input_failures,
            "failure_counts": failure_counts,
            "repair_count": repair_count,
            **feedback_aggregates,
            "tool_performance": tool_performance,
            "tasks": task_summaries,
            "failure_preview": [
                record for record in records if record["status"] == "failed"
            ][:16],
            "max_attempts": policy.max_attempts,
            "plan_attempts": plan_attempts,
            "adapter": _adapter_identity(self.adapter),
            "algorithm": {"id": algorithm_id, "version": algorithm_version},
            "execution_policy": "feedback_driven_retry_repair_or_skip",
            "remote_sample_agents": remote_sample_agents,
            "strict_agent_contract": strict_agent_contract,
            "remote_planner_invocations": remote_planner_invocations,
            "remote_critic_invocations": remote_critic_invocations,
            "remote_reflection_invocations": remote_reflection_invocations,
            "registered_prediction_tool_invocations": len(
                registered_tool_invocation_keys
            ),
            "dsh_agent_prediction_tool_invocations": len(
                dsh_agent_tool_event_ids
            ),
            "host_route_bypass_count": host_route_bypass_count,
            "complete_agent_chains": complete_agent_chains,
            "complete_origin_agent_chains": (
                complete_origin_agent_chains
                if strict_origin_contract
                else complete_agent_chains
            ),
            "strict_agent_chain_coverage": strict_agent_chain_coverage,
            "strict_agent_chain_pass": strict_agent_chain_pass,
            "prediction_unit": (
                "forecast_origin_with_target_horizon_vector"
                if strict_origin_contract
                else "target_horizon_cell"
            ),
            "attempted_origin_samples": (
                len(origin_bundles) if strict_origin_contract else attempted
            ),
            "succeeded_origin_samples": (
                succeeded_origins if strict_origin_contract else succeeded
            ),
            "failed_origin_samples": (
                len(origin_bundles) - succeeded_origins
                if strict_origin_contract
                else failed
            ),
            "prediction_cell_count": attempted,
            "prediction_cells_per_origin": (
                len(origin_bundles[0].requests) if origin_bundles else 1
            ),
            "remote_roles": (
                list(public_plan.get("remote_roles", []))
                if public_plan is not None
                and isinstance(public_plan.get("remote_roles"), (list, tuple))
                else []
            ),
            "host_roles": (
                list(public_plan.get("host_roles", []))
                if public_plan is not None
                and isinstance(public_plan.get("host_roles"), (list, tuple))
                else []
            ),
            "batch_plan": public_plan,
            "batch_plan_digest": plan_digest,
            "action_catalog": list(action_catalog.values()),
            "trace_digest": trace_digest,
        }
        if strict_agent_contract:
            if callable(resume_setter):
                resume_setter(None)
            if callable(token_budget_setter):
                token_budget_setter(None)
        return SampleExecutionBatch(
            successful_rows=tuple(successful_rows),
            scoring_rows=tuple(scoring_rows),
            records=tuple(records),
            summary=summary,
        )
    def _prepare_plan(
        self,
        context: Mapping[str, Any],
        policy: SampleExecutionPolicy,
    ) -> tuple[
        dict[str, Any] | None,
        int,
        tuple[str, bool, str] | None,
    ]:
        final_failure: tuple[str, bool, str] | None = None
        for attempt in range(1, policy.plan_max_attempts + 1):
            try:
                raw_plan = self.adapter.plan_batch(context)
                if not isinstance(raw_plan, Mapping):
                    raise SampleExecutionContractError(
                        "sample batch plan must be an object"
                    )
                plan = dict(raw_plan)
                encoded = canonical_json(plan).encode("utf-8")
                if len(encoded) > _MAX_PLAN_BYTES:
                    raise SampleExecutionContractError(
                        "sample batch plan exceeds the bounded contract"
                    )
                return plan, attempt, None
            except SampleExecutionControlError:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate third-party planning tools
                final_failure = classify_sample_failure(exc)
                if not final_failure[1] or attempt >= policy.plan_max_attempts:
                    return None, attempt, final_failure
                self._backoff(policy, attempt)
        return None, policy.plan_max_attempts, final_failure

    def _prepare_first_attempt_batch(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        context: Mapping[str, Any],
        target_bounds: Mapping[str, Mapping[str, Any]],
        algorithm_id: str,
        algorithm_version: str,
        plan: Mapping[str, Any] | None,
    ) -> dict[str, SamplePredictionOutcome]:
        predict_samples = getattr(self.adapter, "predict_samples", None)
        if plan is None or not callable(predict_samples):
            return {}

        requests: list[SamplePredictionRequest] = []
        for raw_row in rows:
            try:
                request, _observed = _request_from_row(
                    dict(raw_row),
                    context=context,
                    target_bounds=target_bounds,
                    algorithm_id=algorithm_id,
                    algorithm_version=algorithm_version,
                )
            except (SampleExecutionContractError, TypeError, ValueError):
                continue
            requests.append(request)
        if not requests:
            return {}
        sample_ids = [request.sample_id for request in requests]
        if len(sample_ids) != len(set(sample_ids)):
            # Ambiguous sample identities remain on the compatible single-call path.
            return {}

        try:
            raw_outcomes = predict_samples(
                tuple(requests),
                tuple(dict(plan) for _request in requests),
                attempts=tuple(1 for _request in requests),
            )
        except (SampleResultCallbackError, SampleExecutionControlError):
            raise
        except Exception as exc:  # noqa: BLE001 - isolate remote microbatch failures
            if gateway_error_in_chain(exc, retryable_only=True) is not None:
                raise
            dsh_error = dsh_native_runtime_error_in_chain(exc)
            if dsh_error is not None and dsh_native_runtime_retryable(dsh_error):
                raise
            return {
                request.sample_id: SamplePredictionOutcome(
                    sample_id=request.sample_id,
                    error=exc,
                )
                for request in requests
            }
        if not isinstance(raw_outcomes, (list, tuple)) or len(raw_outcomes) != len(
            requests
        ):
            error = SampleExecutionContractError(
                "batched sample adapter must return one outcome per request"
            )
            return {
                request.sample_id: SamplePredictionOutcome(
                    sample_id=request.sample_id,
                    error=error,
                )
                for request in requests
            }

        outcomes: dict[str, SamplePredictionOutcome] = {}
        for request, raw_outcome in zip(requests, raw_outcomes):
            if (
                not isinstance(raw_outcome, SamplePredictionOutcome)
                or raw_outcome.sample_id != request.sample_id
            ):
                outcome = SamplePredictionOutcome(
                    sample_id=request.sample_id,
                    error=SampleExecutionContractError(
                        "batched sample adapter returned an out-of-order outcome"
                    ),
                )
            else:
                outcome = raw_outcome
            outcomes[request.sample_id] = outcome
        return outcomes

    def _prepare_strict_origin_bundle(
        self,
        bundle: _StrictOriginBundle,
        *,
        context: Mapping[str, Any],
        target_bounds: Mapping[str, Mapping[str, Any]],
        algorithm_id: str,
        algorithm_version: str,
        plan: Mapping[str, Any] | None,
        policy: SampleExecutionPolicy,
    ) -> tuple[
        dict[tuple[str, int], SamplePredictionOutcome],
        dict[str, dict[str, Any]],
        str | None,
    ]:
        """Prepare one complete Planner/tool/Critic/Reflector origin chain."""

        admission = (
            self.origin_admission()
            if self.origin_admission is not None
            else nullcontext()
        )
        with admission:
            first_attempt_outcomes = self._prepare_first_attempt_batch(
                bundle.rows,
                context=context,
                target_bounds=target_bounds,
                algorithm_id=algorithm_id,
                algorithm_version=algorithm_version,
                plan=plan,
            )
            origin_outcomes = self._prepare_retry_attempt_waves(
                bundle.rows,
                context=context,
                target_bounds=target_bounds,
                algorithm_id=algorithm_id,
                algorithm_version=algorithm_version,
                plan=plan,
                policy=policy,
                first_attempt_outcomes=first_attempt_outcomes,
            )
            origin_reflections = (
                self._prepare_origin_reflection(
                    bundle,
                    prefetched_attempt_outcomes=origin_outcomes,
                    policy=policy,
                )
                if getattr(self.adapter, "sample_reflection_enabled", True)
                else {}
            )
            return (
                origin_outcomes,
                origin_reflections,
                _batch_terminal_reason(first_attempt_outcomes.values()),
            )

    def _prepare_retry_attempt_waves(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        context: Mapping[str, Any],
        target_bounds: Mapping[str, Mapping[str, Any]],
        algorithm_id: str,
        algorithm_version: str,
        plan: Mapping[str, Any] | None,
        policy: SampleExecutionPolicy,
        first_attempt_outcomes: Mapping[str, SamplePredictionOutcome],
    ) -> dict[tuple[str, int], SamplePredictionOutcome]:
        """Batch retryable failures by attempt wave for adapters with batch support."""

        predict_samples = getattr(self.adapter, "predict_samples", None)
        if _batch_terminal_reason(first_attempt_outcomes.values()) is not None:
            return {
                (sample_id, 1): outcome
                for sample_id, outcome in first_attempt_outcomes.items()
            }
        if plan is None or not first_attempt_outcomes or not callable(predict_samples):
            return {
                (sample_id, 1): outcome
                for sample_id, outcome in first_attempt_outcomes.items()
            }

        requests: list[SamplePredictionRequest] = []
        for raw_row in rows:
            try:
                request, _observed = _request_from_row(
                    dict(raw_row),
                    context=context,
                    target_bounds=target_bounds,
                    algorithm_id=algorithm_id,
                    algorithm_version=algorithm_version,
                )
            except (SampleExecutionContractError, TypeError, ValueError):
                continue
            if request.sample_id in first_attempt_outcomes:
                requests.append(request)
        if len({request.sample_id for request in requests}) != len(requests):
            return {
                (sample_id, 1): outcome
                for sample_id, outcome in first_attempt_outcomes.items()
            }

        outcomes = {
            (sample_id, 1): outcome
            for sample_id, outcome in first_attempt_outcomes.items()
        }
        histories: dict[str, list[dict[str, Any]]] = {
            request.sample_id: [] for request in requests
        }
        active = list(requests)
        for attempt in range(1, policy.max_attempts):
            retry_requests: list[SamplePredictionRequest] = []
            retry_plans: list[dict[str, Any]] = []
            for request in active:
                outcome = outcomes.get((request.sample_id, attempt))
                if outcome is None:
                    continue
                exc = _prefetched_outcome_failure(outcome, request)
                if exc is None:
                    continue
                category, retryable, error_type = classify_sample_failure(exc)
                feedback, _decisions, _tools = _failure_feedback_from_exception(
                    exc, attempt, failure=(category, retryable, error_type)
                )
                histories[request.sample_id].append(feedback)
                if not retryable:
                    continue
                retry_requests.append(request)
                retry_plan = dict(plan)
                retry_plan["sample_retry_feedback"] = [
                    dict(item) for item in histories[request.sample_id]
                ]
                retry_plans.append(retry_plan)
            if not retry_requests:
                break
            next_attempt = attempt + 1
            self._backoff(policy, attempt)
            try:
                raw_outcomes = predict_samples(
                    tuple(retry_requests),
                    tuple(retry_plans),
                    attempts=tuple(next_attempt for _ in retry_requests),
                )
            except (SampleResultCallbackError, SampleExecutionControlError):
                raise
            except Exception as exc:  # noqa: BLE001 - isolate one retry wave
                if gateway_error_in_chain(exc, retryable_only=True) is not None:
                    raise
                dsh_error = dsh_native_runtime_error_in_chain(exc)
                if dsh_error is not None and dsh_native_runtime_retryable(dsh_error):
                    raise
                raw_outcomes = tuple(
                    SamplePredictionOutcome(sample_id=request.sample_id, error=exc)
                    for request in retry_requests
                )
            if not isinstance(raw_outcomes, (list, tuple)) or len(raw_outcomes) != len(
                retry_requests
            ):
                contract_error = SampleExecutionContractError(
                    "batched sample adapter must return one retry outcome per request"
                )
                raw_outcomes = tuple(
                    SamplePredictionOutcome(
                        sample_id=request.sample_id,
                        error=contract_error,
                    )
                    for request in retry_requests
                )
            active = []
            for request, raw_outcome in zip(retry_requests, raw_outcomes):
                if (
                    not isinstance(raw_outcome, SamplePredictionOutcome)
                    or raw_outcome.sample_id != request.sample_id
                ):
                    outcome = SamplePredictionOutcome(
                        sample_id=request.sample_id,
                        error=SampleExecutionContractError(
                            "batched sample adapter returned an out-of-order retry outcome"
                        ),
                    )
                else:
                    outcome = raw_outcome
                outcomes[(request.sample_id, next_attempt)] = outcome
                active.append(request)
        return outcomes

    def _prepare_origin_reflection(
        self,
        bundle: _StrictOriginBundle,
        *,
        prefetched_attempt_outcomes: Mapping[
            tuple[str, int], SamplePredictionOutcome
        ],
        policy: SampleExecutionPolicy,
    ) -> dict[str, dict[str, Any]]:
        """Host-score a complete vector, then invoke one remote Reflector."""

        reflector = getattr(self.adapter, "reflect_origin", None)
        if not callable(reflector):
            raise SampleExecutionContractError(
                "strict origin protocol requires an origin reflector"
            )
        scored_cells: list[dict[str, Any]] = []
        for request, observed in zip(bundle.requests, bundle.observed):
            predicted: float | None = None
            status = "failed"
            failure_class: str | None = "missing_prediction_outcome"
            agent_decisions: Sequence[Mapping[str, Any]] = ()
            tool_calls: Sequence[Mapping[str, Any]] = ()
            for attempt in range(1, policy.max_attempts + 1):
                outcome = prefetched_attempt_outcomes.get(
                    (request.sample_id, attempt)
                )
                if outcome is None:
                    break
                failure = _prefetched_outcome_failure(outcome, request)
                if failure is None:
                    assert outcome.result is not None
                    result = _host_critic_result(
                        _validated_result(outcome.result), request
                    )
                    predicted = float(result["predicted"])
                    status = "succeeded"
                    failure_class = None
                    agent_decisions = result["agent_decisions"]
                    tool_calls = result["tool_calls"]
                    break
                failure_class, retryable, _error_type = classify_sample_failure(
                    failure
                )
                _feedback, decisions, tools = _failure_feedback_from_exception(
                    failure,
                    attempt,
                    failure=(failure_class, retryable, type(failure).__name__),
                )
                agent_decisions = decisions
                tool_calls = tools
                if not retryable:
                    break
            scored_cells.append(
                {
                    "sample_id": request.sample_id,
                    "target": request.target,
                    "horizon_hours": request.horizon_hours,
                    "status": status,
                    "observed": float(observed),
                    "predicted": predicted,
                    "baseline": float(request.baseline),
                    "failure_class": failure_class,
                    "agent_decision_digests": [
                        str(item.get("response_digest"))
                        for item in agent_decisions
                        if item.get("response_digest")
                    ],
                    "tool_output_digests": [
                        str(item.get("output_digest"))
                        for item in tool_calls
                        if item.get("output_digest")
                    ],
                }
            )
        raw = reflector(bundle.requests, scored_cells=scored_cells)
        reflection = _validated_origin_reflection(raw, bundle)
        return {
            request.sample_id: reflection for request in bundle.requests
        }

    def _backoff(self, policy: SampleExecutionPolicy, failed_attempt: int) -> None:
        if policy.retry_backoff_seconds <= 0:
            return
        self._sleep(
            min(60.0, policy.retry_backoff_seconds * (2 ** (failed_attempt - 1)))
        )


def _feedback_aggregates(
    records: Sequence[Mapping[str, Any]],
    *,
    critic_outcome_counts: Mapping[str, int],
    reason_code_counts: Mapping[str, int],
    repair_tool_outcomes: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, Any]]:
    """Create label-free signals that later generations may safely reuse."""

    recovered_by_failure_class: dict[str, int] = {}
    reflection_outcome_counts: dict[str, int] = {}
    reflection_error_source_counts: dict[str, int] = {}
    reflection_next_action_counts: dict[str, int] = {}
    seen_reflections: set[str] = set()

    def increment(counter: dict[str, int], key: str) -> None:
        counter[key] = counter.get(key, 0) + 1

    for record in records:
        reflection = record.get("sample_reflection")
        if isinstance(reflection, Mapping):
            reflection_identity = str(
                reflection.get("wave_digest")
                or reflection.get("response_digest")
                or record.get("sample_id")
                or ""
            )
            if reflection_identity and reflection_identity not in seen_reflections:
                seen_reflections.add(reflection_identity)
                for field_name, counter, allowed in (
                    (
                        "outcome_class",
                        reflection_outcome_counts,
                        {"improved", "degraded", "neutral", "failed"},
                    ),
                    (
                        "error_source",
                        reflection_error_source_counts,
                        {
                            "model",
                            "feature",
                            "parameter",
                            "tool",
                            "execution",
                            "unknown",
                        },
                    ),
                    (
                        "next_action",
                        reflection_next_action_counts,
                        {"keep", "increase", "decrease", "repair", "inspect", "stop"},
                    ),
                ):
                    value = reflection.get(field_name)
                    if isinstance(value, str) and value in allowed:
                        increment(counter, value)
        if record.get("status") != "succeeded":
            continue
        history = record.get("failure_history")
        if not isinstance(history, (list, tuple)):
            continue
        failure_classes = {
            str(item.get("failure_class") or "").strip()[:100]
            for item in history
            if isinstance(item, Mapping)
            and isinstance(item.get("failure_class"), str)
            and str(item.get("failure_class")).strip()
        }
        for failure_class in failure_classes:
            increment(recovered_by_failure_class, failure_class)

    return {
        "critic_outcome_counts": dict(sorted(critic_outcome_counts.items())),
        "reason_code_counts": dict(sorted(reason_code_counts.items())),
        "repair_tool_outcomes": {
            tool_id: dict(sorted(outcomes.items()))
            for tool_id, outcomes in sorted(repair_tool_outcomes.items())
        },
        "recovered_by_failure_class": dict(
            sorted(recovered_by_failure_class.items())
        ),
        "reflection_outcome_counts": dict(
            sorted(reflection_outcome_counts.items())
        ),
        "reflection_error_source_counts": dict(
            sorted(reflection_error_source_counts.items())
        ),
        "reflection_next_action_counts": dict(
            sorted(reflection_next_action_counts.items())
        ),
    }


def summarize_tool_performance(
    records: Sequence[Mapping[str, Any]],
    scoring_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate selected-tool outcomes after scoring has completed.

    Labels are used only here, after every routing decision for the candidate is
    complete. The returned bounded aggregate is suitable for later generations;
    it is never sent back into the current sample batch.
    """

    scoring_by_id = {
        str(row.get("sample_id")): row
        for row in scoring_rows
        if isinstance(row.get("sample_id"), str)
    }
    aggregates: dict[tuple[str, str, str, int], dict[str, Any]] = {}

    def aggregate_for(
        tool: Mapping[str, Any], row: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        tool_id = tool.get("tool_id")
        version = tool.get("version")
        target = row.get("target")
        horizon = row.get("horizon_hours")
        if (
            not isinstance(tool_id, str)
            or not tool_id.strip()
            or not isinstance(version, str)
            or not version.strip()
            or not isinstance(target, str)
            or not target.strip()
            or isinstance(horizon, bool)
            or not isinstance(horizon, int)
        ):
            return None
        key = (tool_id.strip()[:160], version.strip()[:80], target.strip()[:120], horizon)
        return aggregates.setdefault(
            key,
            {
                "tool_id": key[0],
                "version": key[1],
                "target": key[2],
                "horizon_hours": key[3],
                "selected": 0,
                "completed": 0,
                "failed": 0,
                "rejected": 0,
                "critic_accept": 0,
                "critic_repair": 0,
                "critic_failed": 0,
                "final_accept": 0,
                "recovered": 0,
                "n": 0,
                "_absolute_error": 0.0,
                "_squared_error": 0.0,
                "_baseline_absolute_error": 0.0,
                "_baseline_squared_error": 0.0,
            },
        )

    for record in records:
        trace = record.get("attempt_trace")
        row = scoring_by_id.get(str(record.get("sample_id")))
        if not isinstance(trace, (list, tuple)) or not isinstance(row, Mapping):
            continue
        accepted_entry: Mapping[str, Any] | None = None
        for entry in trace:
            if not isinstance(entry, Mapping):
                continue
            tool = entry.get("selected_tool")
            if not isinstance(tool, Mapping):
                continue
            aggregate = aggregate_for(tool, row)
            if aggregate is None:
                continue
            aggregate["selected"] += 1
            status = str(tool.get("status") or "").casefold()
            if status in {"completed", "failed", "rejected"}:
                aggregate[status] += 1
            critics = entry.get("critic_decisions")
            critic_repair_recorded = False
            if isinstance(critics, (list, tuple)):
                for critic in critics:
                    if not isinstance(critic, Mapping):
                        continue
                    critic_status = str(critic.get("status") or "").casefold()
                    critic_decision = str(critic.get("decision") or "").casefold()
                    if critic_status == "failed":
                        aggregate["critic_failed"] += 1
                    elif "accept" in critic_decision:
                        aggregate["critic_accept"] += 1
                    elif any(
                        marker in critic_decision
                        for marker in ("repair", "reject", "select_registered_tool")
                    ):
                        aggregate["critic_repair"] += 1
                        critic_repair_recorded = True
            if entry.get("requested_repair_tool") and not critic_repair_recorded:
                aggregate["critic_repair"] += 1
            if entry.get("outcome") == "accepted":
                accepted_entry = entry

        if accepted_entry is None or record.get("status") != "succeeded":
            continue
        final_tool = accepted_entry.get("selected_tool")
        if not isinstance(final_tool, Mapping):
            continue
        aggregate = aggregate_for(final_tool, row)
        if aggregate is None:
            continue
        aggregate["final_accept"] += 1
        if int(record.get("attempts", 0)) > 1:
            aggregate["recovered"] += 1
        try:
            predicted = _finite_float(record.get("predicted"), "tool predicted")
            observed = _finite_float(row.get("observed"), "tool observed")
            baseline = _finite_float(row.get("baseline"), "tool baseline")
        except (SampleExecutionContractError, TypeError, ValueError):
            continue
        error = predicted - observed
        baseline_error = baseline - observed
        aggregate["n"] += 1
        aggregate["_absolute_error"] += abs(error)
        aggregate["_squared_error"] += error * error
        aggregate["_baseline_absolute_error"] += abs(baseline_error)
        aggregate["_baseline_squared_error"] += baseline_error * baseline_error

    result: list[dict[str, Any]] = []
    for aggregate in sorted(
        aggregates.values(),
        key=lambda item: (
            str(item["tool_id"]),
            str(item["target"]),
            int(item["horizon_hours"]),
        ),
    )[:64]:
        count = int(aggregate["n"])
        public = {
            key: value for key, value in aggregate.items() if not key.startswith("_")
        }
        if count:
            mae = float(aggregate["_absolute_error"]) / count
            rmse = math.sqrt(float(aggregate["_squared_error"]) / count)
            baseline_mae = float(aggregate["_baseline_absolute_error"]) / count
            baseline_rmse = math.sqrt(
                float(aggregate["_baseline_squared_error"]) / count
            )
            public.update(
                {
                    "mae": mae,
                    "rmse": rmse,
                    "baseline_mae": baseline_mae,
                    "baseline_rmse": baseline_rmse,
                    "rmse_improvement": baseline_rmse - rmse,
                    "skill_score": skill_score(rmse, baseline_rmse),
                }
            )
        result.append(public)
    return result


def _record_feedback_steps(
    decisions: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    critic_outcome_counts: dict[str, int],
    reason_code_counts: dict[str, int],
    repair_tool_outcomes: dict[str, dict[str, int]],
) -> None:
    repair_tools = {
        "bounded-projection-repair",
        "bounded-persistence-fallback",
    }
    for decision in decisions:
        role = str(decision.get("role") or "")
        if "critic" in role.casefold():
            outcome = _critic_outcome(decision)
            critic_outcome_counts[outcome] = critic_outcome_counts.get(outcome, 0) + 1
        reason_code = decision.get("reason_code")
        if isinstance(reason_code, str) and reason_code.strip():
            code = (
                safe_remote_reason_code(reason_code)
                if role.casefold().startswith("remote_")
                else reason_code.strip()[:160]
            )
            reason_code_counts[code] = reason_code_counts.get(code, 0) + 1
    for tool in tools:
        tool_id = tool.get("tool_id")
        status = tool.get("status")
        if tool_id not in repair_tools or not isinstance(status, str):
            continue
        normalized_status = status.strip().casefold()
        if normalized_status not in {"completed", "failed", "rejected"}:
            continue
        outcomes = repair_tool_outcomes.setdefault(str(tool_id), {})
        outcomes[normalized_status] = outcomes.get(normalized_status, 0) + 1


def _critic_outcome(decision: Mapping[str, Any]) -> str:
    """Reduce host or remote critic evidence to a stable public outcome."""

    decision_text = str(decision.get("decision") or "").casefold()
    status = str(decision.get("status") or "").casefold()
    if "reject" in decision_text or status == "rejected":
        return "rejected"
    if any(marker in decision_text for marker in ("accept", "within", "valid")):
        return "accepted"
    return "other"


def classify_sample_failure(exc: BaseException) -> tuple[str, bool, str]:
    """Return a stable public class, retryability, and exception type."""

    if isinstance(exc, SampleExecutionAttemptError):
        return exc.failure_class, exc.retryable, exc.error_type
    error_type = type(exc).__name__[:120]
    if isinstance(exc, DshNativeRuntimeUnavailableError):
        status_code = getattr(exc, "status_code", None)
        error_code = str(getattr(exc, "error_code", "") or "")
        retryable = dsh_native_runtime_retryable(exc)
        if status_code == 429:
            return "rate_limited", True, error_type
        if status_code == 408:
            return "timeout", True, error_type
        if error_code == "dsh_native_runtime_transport_error":
            return "connection", True, error_type
        if retryable:
            return "remote_transient", True, error_type
        if error_code == "dsh_native_runtime_contract_error":
            return "invalid_output", False, error_type
        return "remote_rejected", False, error_type
    gateway_retryable = getattr(exc, "retryable", None)
    gateway_status = getattr(exc, "status_code", None)
    gateway_split_eligible = getattr(exc, "split_eligible", None)
    if (
        isinstance(gateway_retryable, bool)
        and isinstance(gateway_split_eligible, bool)
    ):
        if gateway_split_eligible:
            return "invalid_output", True, error_type
        if gateway_status == 429:
            return "rate_limited", False, error_type
        if gateway_retryable:
            # The model gateway has already exhausted its request-local retry
            # budget. Outer sample repair must not replay the same transport
            # failure as another agent attempt.
            return "remote_transient", False, error_type
        if isinstance(gateway_status, int):
            return "remote_rejected", False, error_type
        return "tool_error", False, error_type
    message = str(exc).lower()
    if isinstance(exc, (TimeoutError, socket.timeout)) or "timed out" in message:
        return "timeout", True, error_type
    if isinstance(exc, HTTPError):
        if exc.code == 429:
            return "rate_limited", True, error_type
        if exc.code in {408, 500, 502, 503, 504}:
            return "remote_transient", True, error_type
        return "remote_rejected", False, error_type
    if "429" in message or "rate limit" in message or "too many requests" in message:
        return "rate_limited", True, error_type
    if isinstance(exc, (ConnectionError, URLError)):
        return "connection", True, error_type
    if isinstance(exc, SampleRepairRequired):
        return "constraint_rejected", True, error_type
    if isinstance(exc, SampleExecutionContractError):
        return "invalid_output", True, error_type
    if isinstance(exc, (ArithmeticError, FloatingPointError)):
        return "numerical", False, error_type
    return "tool_error", False, error_type


def _request_from_row(
    row: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    target_bounds: Mapping[str, Mapping[str, Any]],
    algorithm_id: str,
    algorithm_version: str,
) -> tuple[SamplePredictionRequest, float]:
    target = str(row.get("target") or "").strip()
    try:
        bounds = target_bounds[target]
    except KeyError as exc:
        raise SampleExecutionContractError(
            f"no registered bounds for prediction target {target!r}"
        ) from exc
    origin = row.get("origin_timestamp")
    target_timestamp = row.get("timestamp", row.get("target_timestamp"))
    horizon = row.get("horizon_hours")
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise SampleExecutionContractError("sample horizon_hours must be an integer")
    observed = _finite_float(row.get("observed"), "sample observed")
    raw_label_free_context = row.get("label_free_context")
    if raw_label_free_context is None:
        raw_label_free_context = {
            name: row[name]
            for name in (
                "feature_snapshot_ref",
                "history_window_ref",
                "features",
                "history",
            )
            if name in row
        }
    request = SamplePredictionRequest(
        sample_id=_sample_id(row, context, target=target, horizon=horizon),
        candidate_id=str(context.get("candidate_id") or ""),
        dataset_digest=str(context.get("dataset_digest") or ""),
        partition=str(row.get("partition") or ""),
        target=target,
        unit=str(bounds.get("unit") or "unknown"),
        horizon_hours=horizon,
        origin_timestamp=origin,
        target_timestamp=target_timestamp,
        baseline=_finite_float(row.get("baseline"), "sample baseline"),
        proposed_prediction=(
            _finite_float(row.get("predicted"), "sample proposed_prediction")
            if row.get("predicted") is not None
            else None
        ),
        minimum=_finite_float(bounds.get("minimum"), "sample minimum"),
        maximum=_finite_float(bounds.get("maximum"), "sample maximum"),
        algorithm_id=algorithm_id,
        algorithm_version=algorithm_version,
        label_free_context=raw_label_free_context,
    )
    return request, observed


def _sample_id(
    row: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    target: str,
    horizon: int,
) -> str:
    identity = {
        "candidate_id": str(context.get("candidate_id") or "")[:300],
        "dataset_digest": str(context.get("dataset_digest") or "")[:300],
        "sample_index": row.get("sample_index"),
        "partition": str(row.get("partition") or "")[:120],
        "target": target[:160],
        "horizon_hours": horizon,
        "origin_timestamp": str(row.get("origin_timestamp"))[:160],
        "target_timestamp": str(
            row.get("timestamp", row.get("target_timestamp"))
        )[:160],
    }
    return "prediction-sample:" + digest(identity)[:32]


def forecast_origin_sample_id(
    requests: Sequence[SamplePredictionRequest],
) -> str:
    """Return the stable identity shared by one vector prediction chain."""

    if not requests:
        raise SampleExecutionContractError("forecast origin bundle must not be empty")
    first = requests[0]
    if any(
        request.candidate_id != first.candidate_id
        or request.dataset_digest != first.dataset_digest
        or request.partition != first.partition
        or request.origin_timestamp != first.origin_timestamp
        for request in requests
    ):
        raise SampleExecutionContractError(
            "forecast origin bundle mixes incompatible sample origins"
        )
    sample_ids = [request.sample_id for request in requests]
    if len(sample_ids) != len(set(sample_ids)):
        raise SampleExecutionContractError(
            "forecast origin bundle sample identifiers must be unique"
        )
    return "origin:" + digest(
        {
            "candidate_id": first.candidate_id,
            "dataset_digest": first.dataset_digest,
            "partition": first.partition,
            "origin_timestamp": first.origin_timestamp,
            "sample_ids": sorted(sample_ids),
        }
    )


def _strict_origin_bundles(
    rows: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    target_bounds: Mapping[str, Mapping[str, Any]],
    algorithm_id: str,
    algorithm_version: str,
) -> tuple[_StrictOriginBundle, ...]:
    """Validate and order complete multi-output forecast-origin bundles."""

    grouped: dict[str, list[tuple[dict[str, Any], SamplePredictionRequest, float]]] = {}
    group_order: list[str] = []
    all_tasks: set[tuple[str, int]] = set()
    for raw_row in rows:
        row = dict(raw_row)
        request, observed = _request_from_row(
            row,
            context=context,
            target_bounds=target_bounds,
            algorithm_id=algorithm_id,
            algorithm_version=algorithm_version,
        )
        origin_key = canonical_json(
            {
                "partition": request.partition,
                "origin_timestamp": request.origin_timestamp,
            }
        )
        if origin_key not in grouped:
            grouped[origin_key] = []
            group_order.append(origin_key)
        grouped[origin_key].append((row, request, observed))
        all_tasks.add((request.target, request.horizon_hours))

    bundles: list[_StrictOriginBundle] = []
    raw_horizons = context.get("horizons_hours")
    expected_tasks = (
        {
            (str(target), int(horizon))
            for target in target_bounds
            for horizon in raw_horizons
            if isinstance(horizon, int)
            and not isinstance(horizon, bool)
            and horizon > 0
        }
        if isinstance(raw_horizons, (list, tuple))
        else set(all_tasks)
    )
    if expected_tasks and all_tasks != expected_tasks:
        raise SampleExecutionContractError(
            "strict origin cohort does not cover the configured target/horizon grid"
        )
    for origin_key in group_order:
        values = sorted(
            grouped[origin_key],
            key=lambda item: (item[1].target, item[1].horizon_hours),
        )
        tasks = [(item[1].target, item[1].horizon_hours) for item in values]
        if len(tasks) != len(set(tasks)):
            raise SampleExecutionContractError(
                "strict origin bundle contains duplicate target/horizon cells"
            )
        if set(tasks) != expected_tasks:
            raise SampleExecutionContractError(
                "strict origin bundle must contain every target/horizon cell"
            )
        requests = tuple(item[1] for item in values)
        bundles.append(
            _StrictOriginBundle(
                origin_sample_id=forecast_origin_sample_id(requests),
                rows=tuple(item[0] for item in values),
                requests=requests,
                observed=tuple(item[2] for item in values),
            )
        )
    return tuple(bundles)


def _validated_checkpoint_rows(
    value: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    target_bounds: Mapping[str, Mapping[str, Any]],
    algorithm_id: str,
    algorithm_version: str,
) -> dict[str, dict[str, Any]]:
    """Validate private persisted rows against the complete current cohort."""

    if value is None:
        return {}
    if not isinstance(value, (list, tuple)):
        raise SampleExecutionContractError("sample checkpoint rows must be an array")
    current: dict[
        str, tuple[dict[str, Any], SamplePredictionRequest, float, bool]
    ] = {}
    for raw_row in rows:
        row = dict(raw_row)
        valid_input = True
        try:
            request, observed = _request_from_row(
                row,
                context=context,
                target_bounds=target_bounds,
                algorithm_id=algorithm_id,
                algorithm_version=algorithm_version,
            )
        except (SampleExecutionContractError, TypeError, ValueError):
            # A non-finite proposed prediction is still reduced to a finite
            # host penalty and can therefore have a durable failed row. Rebuild
            # only its label-free identity while keeping it classified invalid.
            try:
                request, observed = _checkpoint_request_for_invalid_row(
                    row,
                    context=context,
                    target_bounds=target_bounds,
                    algorithm_id=algorithm_id,
                    algorithm_version=algorithm_version,
                )
            except (SampleExecutionContractError, TypeError, ValueError):
                continue
            valid_input = False
        if request.sample_id in current:
            raise SampleExecutionContractError(
                "sample checkpoint cohort contains duplicate sample_id values"
            )
        current[request.sample_id] = (row, request, observed, valid_input)

    resumed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise SampleExecutionContractError(
                f"sample checkpoint rows[{index}] must be an object"
            )
        item = dict(raw)
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise SampleExecutionContractError(
                f"sample checkpoint rows[{index}] has no sample_id"
            )
        if sample_id in resumed:
            raise SampleExecutionContractError(
                "sample checkpoint contains duplicate sample_id values"
            )
        source = current.get(sample_id)
        if source is None:
            raise SampleExecutionContractError(
                "sample checkpoint row is outside the current cohort"
            )
        row, request, observed, valid_input = source
        expected = {
            "sample_index": row.get("sample_index"),
            "sample_id": request.sample_id,
            "candidate_id": request.candidate_id,
            "target": request.target,
            # The durable browser row is built from the finalized scoring row,
            # which preserves an explicit source unit and otherwise uses the
            # archive contract's ``unknown`` fallback.
            "unit": str(row.get("unit") or "unknown"),
            "horizon_hours": request.horizon_hours,
            "origin_timestamp": _checkpoint_timestamp(
                row.get("origin_timestamp", row.get("timestamp"))
            ),
            "target_timestamp": _checkpoint_timestamp(
                row.get("target_timestamp", row.get("timestamp"))
            ),
            "observed": observed,
            "baseline": request.baseline,
        }
        actual = {
            "sample_index": item.get("sample_index"),
            "sample_id": sample_id,
            "candidate_id": item.get("candidate_id"),
            "target": item.get("target"),
            "unit": item.get("unit"),
            "horizon_hours": item.get("horizon_hours"),
            "origin_timestamp": _checkpoint_timestamp(
                item.get("origin_timestamp")
            ),
            "target_timestamp": _checkpoint_timestamp(
                item.get("target_timestamp")
            ),
            "observed": _finite_float(
                item.get("observed"), "checkpoint observed"
            ),
            "baseline": _finite_float(
                item.get("model_reference_baseline", item.get("baseline")),
                "checkpoint model reference baseline",
            ),
        }
        if canonical_json(actual) != canonical_json(expected):
            raise SampleExecutionContractError(
                "sample checkpoint row does not match the current cohort"
            )
        status = str(item.get("status") or "").strip().casefold()
        if status not in {"succeeded", "failed"}:
            raise SampleExecutionContractError(
                "sample checkpoint row has an unsupported status"
            )
        if not valid_input and status != "failed":
            raise SampleExecutionContractError(
                "sample checkpoint cannot resume an invalid input as succeeded"
            )
        attempts = item.get("attempts")
        retry_count = item.get("retry_count")
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or not 0 <= attempts <= 8
            or isinstance(retry_count, bool)
            or not isinstance(retry_count, int)
            or not 0 <= retry_count <= 8
        ):
            raise SampleExecutionContractError(
                "sample checkpoint row has invalid attempt counters"
            )
        projected = {
            **item,
            "sample_id": sample_id,
            "status": status,
            "predicted": _finite_float(
                item.get("predicted"), "checkpoint predicted"
            ),
            "attempts": attempts,
            "retry_count": retry_count,
        }
        resumed[sample_id] = projected
    return resumed


def _checkpoint_request_for_invalid_row(
    row: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    target_bounds: Mapping[str, Mapping[str, Any]],
    algorithm_id: str,
    algorithm_version: str,
) -> tuple[SamplePredictionRequest, float]:
    """Recover identity for a row rejected only by its proposed prediction."""

    projected = dict(row)
    projected["predicted"] = None
    return _request_from_row(
        projected,
        context=context,
        target_bounds=target_bounds,
        algorithm_id=algorithm_id,
        algorithm_version=algorithm_version,
    )


def _checkpoint_digest_value(value: Any) -> Any:
    """Make non-finite sample inputs stable without accepting them as valid."""

    if isinstance(value, float) and not math.isfinite(value):
        return {
            "__ecologyrsi_checkpoint_float__": (
                "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"
            )
        }
    if isinstance(value, Mapping):
        return {
            key: _checkpoint_digest_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_checkpoint_digest_value(item) for item in value]
    return value


def _checkpoint_timestamp(value: Any) -> str | int | float:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return int(number) if number.is_integer() else number
    raise SampleExecutionContractError(
        "sample checkpoint timestamp must be finite or non-empty text"
    )


def _checkpoint_scoring_row(
    row: Mapping[str, Any],
    request: SamplePredictionRequest,
    resumed: Mapping[str, Any],
) -> dict[str, Any]:
    projected = dict(row)
    projected.update(
        {
            "sample_id": request.sample_id,
            "predicted": float(resumed["predicted"]),
            "sample_execution_status": str(resumed["status"]),
            "sample_execution_attempts": int(resumed["attempts"]),
            "sample_execution_retry_count": int(resumed["retry_count"]),
            "prediction_source": resumed.get("prediction_source"),
            "scoring_fallback": resumed.get("scoring_fallback"),
            "scoring_fallback_source": resumed.get("scoring_fallback_source"),
        }
    )
    failure_class = resumed.get("failure_class")
    if failure_class is not None:
        projected["sample_execution_failure"] = {
            "class": str(failure_class)[:120],
            "retryable": False,
            "error_type": "CheckpointResumedFailure",
        }
    failure_summary = resumed.get("failure_summary")
    if isinstance(failure_summary, Mapping):
        projected["sample_execution_failure_summary"] = {
            "decisions": [
                dict(item)
                for item in failure_summary.get("decisions", ())
                if isinstance(item, Mapping)
            ],
            "tools": [
                dict(item)
                for item in failure_summary.get("tools", ())
                if isinstance(item, Mapping)
            ],
        }
    agent_chain = resumed.get("sample_agent_chain")
    if isinstance(agent_chain, Mapping):
        projected["sample_agent_chain"] = dict(agent_chain)
    origin_sample_id = resumed.get("origin_sample_id")
    if isinstance(origin_sample_id, str) and origin_sample_id.startswith("origin:"):
        projected["origin_sample_id"] = origin_sample_id
    return projected


def _adapter_resume_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    resumed_rows: Mapping[str, Mapping[str, Any]],
    progress: Any,
    *,
    force_full_cohort: bool = False,
    origin_bundles: Sequence[_StrictOriginBundle] = (),
) -> dict[str, Any] | None:
    task_totals: dict[tuple[str, int], int] = {}
    for row in rows:
        target = str(row.get("target") or "").strip()[:160]
        raw_horizon = row.get("horizon_hours")
        horizon = (
            raw_horizon
            if isinstance(raw_horizon, int) and not isinstance(raw_horizon, bool)
            else 0
        )
        key = (target, horizon)
        task_totals[key] = task_totals.get(key, 0) + 1
    resumed_failed_by_task: dict[tuple[str, int], int] = {}
    succeeded = 0
    for row in resumed_rows.values():
        if row.get("status") == "succeeded":
            succeeded += 1
            continue
        key = (str(row.get("target") or "")[:160], int(row.get("horizon_hours", 0)))
        resumed_failed_by_task[key] = resumed_failed_by_task.get(key, 0) + 1

    prior = dict(progress) if isinstance(progress, Mapping) else {}
    if not resumed_rows and not prior and not force_full_cohort:
        return None

    def prior_count(name: str, default: int = 0) -> int:
        value = prior.get(name, default)
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else 0
        )

    batch_index = (
        len(resumed_rows) if force_full_cohort else prior_count("batch_index")
    )
    progress_id = prior_count("progress_id")
    split_recovered = min(
        prior_count("adaptive_split_recovered_samples"), len(resumed_rows)
    )
    split_failed = min(
        prior_count("adaptive_split_failed_samples"),
        len(resumed_rows) - split_recovered,
    )
    result = {
        "completed_samples": len(resumed_rows),
        "succeeded_samples": succeeded,
        "failed_samples": len(resumed_rows) - succeeded,
        "total_samples": len(rows),
        "batch_index": batch_index,
        "batch_count": max(
            prior_count("batch_count"),
            len(rows) if force_full_cohort else batch_index,
        ),
        "progress_id": progress_id,
        "gateway_request_count": prior_count(
            "gateway_request_count", batch_index
        ),
        "adaptive_split_trigger_count": prior_count(
            "adaptive_split_trigger_count"
        ),
        "adaptive_split_count": prior_count("adaptive_split_count"),
        "adaptive_split_max_depth": prior_count("adaptive_split_max_depth"),
        "adaptive_split_recovered_samples": split_recovered,
        "adaptive_split_failed_samples": split_failed,
        "tasks": [
            {
                "target": target,
                "horizon_hours": horizon,
                "total_samples": total,
                "resumed_failed_samples": resumed_failed_by_task.get(
                    (target, horizon), 0
                ),
            }
            for (target, horizon), total in sorted(task_totals.items())
        ],
    }
    if origin_bundles:
        completed_origins = [
            bundle
            for bundle in origin_bundles
            if all(request.sample_id in resumed_rows for request in bundle.requests)
        ]
        result.update(
            {
                "completed_origin_samples": len(completed_origins),
                "succeeded_origin_samples": sum(
                    all(
                        resumed_rows[request.sample_id].get("status") == "succeeded"
                        for request in bundle.requests
                    )
                    for bundle in completed_origins
                ),
                "total_origin_samples": len(origin_bundles),
            }
        )
    return result


def _invalid_input_record(
    row: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    error_type: str,
    plan_digest: str | None,
) -> dict[str, Any]:
    target = str(row.get("target") or "unknown")[:160]
    raw_horizon = row.get("horizon_hours")
    horizon = (
        raw_horizon
        if isinstance(raw_horizon, int) and not isinstance(raw_horizon, bool)
        else 0
    )
    return {
        "schema_version": SAMPLE_EXECUTION_SCHEMA_VERSION,
        "sample_id": _sample_id(row, context, target=target, horizon=horizon),
        "status": "failed",
        "attempts": 0,
        "retry_count": 0,
        "target": target,
        "horizon_hours": horizon,
        "predicted": None,
        "batch_plan_digest": plan_digest,
        "failure_action": "skip_sample_before_agent_execution",
        "failure": {
            "class": "invalid_sample_input",
            "retryable": False,
            "error_type": str(error_type)[:120],
        },
    }


def _invalid_input_scoring_row(
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    target_bounds: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Apply a bounded worst-case penalty when the adapter request is invalid."""

    try:
        observed = _finite_float(row.get("observed"), "sample observed")
        baseline = _finite_float(row.get("baseline"), "sample baseline")
        bounds = target_bounds[str(row.get("target") or "").strip()]
        minimum = _finite_float(bounds.get("minimum"), "sample minimum")
        maximum = _finite_float(bounds.get("maximum"), "sample maximum")
    except (KeyError, SampleExecutionContractError, TypeError, ValueError):
        return None
    bounded_baseline = min(maximum, max(minimum, baseline))
    penalty_source, penalty_prediction = max(
        (
            ("bounded_persistence", bounded_baseline),
            ("registered_minimum", minimum),
            ("registered_maximum", maximum),
        ),
        key=lambda item: abs(float(item[1]) - observed),
    )
    result = dict(row)
    result["sample_id"] = record["sample_id"]
    result["predicted"] = penalty_prediction
    result["sample_execution_status"] = "failed"
    result["sample_execution_attempts"] = 0
    result["sample_execution_retry_count"] = 0
    result["sample_execution_failure"] = dict(record.get("failure", {}))
    result["scoring_fallback"] = "invalid_input_physical_penalty"
    result["scoring_fallback_source"] = penalty_source
    return result


def _validated_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SampleExecutionContractError("sample prediction response must be an object")
    unknown = set(value) - {"predicted", "agent_decisions", "tool_calls"}
    if unknown:
        raise SampleExecutionContractError(
            "sample prediction response contains unsupported fields"
        )
    predicted = _finite_float(value.get("predicted"), "sample response predicted")
    decisions = _public_steps(
        value.get("agent_decisions"),
        kind="agent",
        id_field="role",
    )
    tools = _public_steps(
        value.get("tool_calls"),
        kind="tool",
        id_field="tool_id",
    )
    if not decisions:
        raise SampleExecutionContractError(
            "sample prediction response requires an agent decision"
        )
    if not any(item.get("status") == "completed" for item in tools):
        raise SampleExecutionContractError(
            "sample prediction response requires a completed tool call"
        )
    return {
        "predicted": predicted,
        "agent_decisions": decisions,
        "tool_calls": tools,
    }


def _validated_origin_reflection(
    value: Any,
    bundle: _StrictOriginBundle,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "sample_id",
        "origin_sample_id",
        "cell_sample_ids",
        "outcome_class",
        "error_source",
        "next_action",
        "confidence",
        "summary",
        "model_id",
        "response_digest",
        "wave_digest",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise SampleExecutionContractError(
            "origin reflection fields do not match the strict contract"
        )
    reflection = dict(value)
    if (
        reflection["schema_version"]
        != "ecologyrsi-dsh.sample-origin-reflection/1"
        or reflection["sample_id"] != bundle.origin_sample_id
        or reflection["origin_sample_id"] != bundle.origin_sample_id
    ):
        raise SampleExecutionContractError("origin reflection identity mismatch")
    expected_ids = [request.sample_id for request in bundle.requests]
    if reflection["cell_sample_ids"] != expected_ids:
        raise SampleExecutionContractError(
            "origin reflection prediction-cell membership mismatch"
        )
    for name in ("response_digest", "wave_digest"):
        value_digest = reflection[name]
        if (
            not isinstance(value_digest, str)
            or len(value_digest) != 64
            or any(character not in "0123456789abcdef" for character in value_digest)
        ):
            raise SampleExecutionContractError(
                f"origin reflection {name} must be a SHA-256 digest"
            )
    return reflection


def _run_sample_reflection(
    adapter: Any,
    request: SamplePredictionRequest,
    *,
    observed: float,
    predicted: float | None,
    status: str,
    failure_class: str | None,
    agent_decisions: Sequence[Mapping[str, Any]],
    tool_calls: Sequence[Mapping[str, Any]],
    prepared_reflection: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Run an optional post-score reflector and return its public evidence."""

    if prepared_reflection is None:
        if getattr(adapter, "sample_reflection_enabled", True) is False:
            return None, None
        reflector = getattr(adapter, "reflect_sample", None)
        if not callable(reflector):
            return None, None
        raw = reflector(
            request,
            observed=observed,
            predicted=predicted,
            status=status,
            failure_class=failure_class,
            agent_decisions=agent_decisions,
            tool_calls=tool_calls,
        )
    else:
        raw = prepared_reflection
    if not isinstance(raw, Mapping):
        raise SampleExecutionContractError(
            "sample reflector must return an object"
        )
    required = {
        "schema_version",
        "sample_id",
        "outcome_class",
        "error_source",
        "next_action",
        "confidence",
        "summary",
        "model_id",
        "response_digest",
        "wave_digest",
    }
    origin_required = required | {"origin_sample_id", "cell_sample_ids"}
    if frozenset(raw) not in {frozenset(required), frozenset(origin_required)}:
        raise SampleExecutionContractError(
            "sample reflection fields do not match the strict contract"
        )
    reflection = dict(raw)
    origin_reflection = (
        reflection["schema_version"]
        == "ecologyrsi-dsh.sample-origin-reflection/1"
    )
    if not origin_reflection and (
        reflection["schema_version"] != "ecology-sample-reflection@1"
    ):
        raise SampleExecutionContractError(
            "sample reflection schema version mismatch"
        )
    if origin_reflection:
        raw_ids = reflection.get("cell_sample_ids")
        if (
            reflection.get("sample_id") != reflection.get("origin_sample_id")
            or not isinstance(raw_ids, list)
            or request.sample_id not in raw_ids
        ):
            raise SampleExecutionContractError(
                "sample origin reflection membership mismatch"
            )
    elif reflection["sample_id"] != request.sample_id:
        raise SampleExecutionContractError("sample reflection identity mismatch")
    for name in ("response_digest", "wave_digest"):
        value = reflection[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise SampleExecutionContractError(
                f"sample reflection {name} must be a SHA-256 digest"
            )
    decision = {
        "role": "remote_reflector_agent",
        "decision": f"reflect_sample:{reflection['outcome_class']}",
        "status": "completed",
        "model_id": str(reflection["model_id"])[:200],
        "reason_code": f"reflection_{reflection['next_action']}"[:160],
        "confidence": float(reflection["confidence"]),
        "response_digest": reflection["response_digest"],
    }
    return reflection, decision


def _host_critic_result(
    result: Mapping[str, Any], request: SamplePredictionRequest
) -> dict[str, Any]:
    """Apply a host-owned physical critic even for injected agent adapters."""

    predicted = float(result["predicted"])
    decisions = [dict(item) for item in result["agent_decisions"]]
    tools = [dict(item) for item in result["tool_calls"]]
    in_range = request.minimum <= predicted <= request.maximum
    if not any(item.get("role") == "constraint_critic" for item in decisions):
        decisions.append(
            {
                "role": "constraint_critic",
                "decision": (
                    "prediction_within_registered_range"
                    if in_range
                    else "reject_prediction_outside_registered_range"
                ),
                "status": "completed",
            }
        )
    if not any(item.get("tool_id") == "physical-range-check" for item in tools):
        tools.append(
            {
                "tool_id": "physical-range-check",
                "version": "1",
                "status": "completed" if in_range else "rejected",
            }
        )
    if not in_range:
        raise SampleRepairRequired(
            "adapter prediction is outside the physical range",
            agent_decisions=_bounded_public_steps(decisions),
            tool_calls=_bounded_public_steps(tools),
            previous_prediction=predicted,
        )
    return {
        "predicted": predicted,
        "agent_decisions": _bounded_public_steps(decisions),
        "tool_calls": _bounded_public_steps(tools),
    }


def _exception_public_steps(
    exc: BaseException,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read only validated operational evidence attached to a repair request."""

    if not isinstance(exc, (SampleRepairRequired, SampleExecutionAttemptError)):
        return [], []
    try:
        decisions = _public_steps(
            exc.agent_decisions,
            kind="agent",
            id_field="role",
        )
        tools = _public_steps(
            exc.tool_calls,
            kind="tool",
            id_field="tool_id",
        )
    except SampleExecutionContractError:
        return [], []
    return decisions, tools


def _failure_feedback_from_exception(
    exc: BaseException,
    attempt: int,
    *,
    failure: tuple[str, bool, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create the same bounded retry evidence for sequential and batched paths."""

    category, retryable, error_type = failure or classify_sample_failure(exc)
    decisions, tools = _exception_public_steps(exc)
    feedback: dict[str, Any] = {
        "attempt": attempt,
        "failure_class": category,
        "retryable": retryable,
        "error_type": error_type,
    }
    tool_ids = [str(item["tool_id"]) for item in tools if item.get("tool_id")]
    reason_codes = [
        str(item["reason_code"])
        for item in decisions
        if item.get("reason_code")
    ]
    critic_response_digests = [
        str(item["response_digest"])
        for item in decisions
        if "critic" in str(item.get("role") or "").casefold()
        and item.get("response_digest")
    ]
    requested_tool_id = (
        exc.requested_tool_id
        if isinstance(exc, (SampleRepairRequired, SampleExecutionAttemptError))
        else None
    )
    if tool_ids:
        feedback["tool_ids"] = tool_ids[-4:]
    if reason_codes:
        feedback["reason_codes"] = reason_codes[-4:]
    if critic_response_digests:
        feedback["critic_response_digests"] = critic_response_digests[-4:]
    if requested_tool_id is not None:
        feedback["requested_tool_id"] = requested_tool_id
    previous_prediction = (
        exc.previous_prediction
        if isinstance(exc, (SampleRepairRequired, SampleExecutionAttemptError))
        else None
    )
    if previous_prediction is not None:
        feedback["previous_prediction"] = previous_prediction
    return feedback, decisions, tools


def _prefetched_outcome_failure(
    outcome: SamplePredictionOutcome,
    request: SamplePredictionRequest,
) -> BaseException | None:
    """Determine whether a prefetched result needs a retry without retaining it."""

    if outcome.error is not None:
        return outcome.error
    if outcome.result is None:
        return SampleExecutionContractError("prefetched sample outcome has no result")
    try:
        _host_critic_result(_validated_result(outcome.result), request)
    except Exception as exc:  # noqa: BLE001 - mirrors the execution boundary
        return exc
    return None


def _batch_terminal_reason(
    outcomes: Sequence[SamplePredictionOutcome] | Any,
) -> str | None:
    reasons = {
        outcome.terminal_reason
        for outcome in outcomes
        if isinstance(outcome, SamplePredictionOutcome)
        and outcome.terminal_reason is not None
    }
    if not reasons:
        return None
    if len(reasons) != 1:
        raise SampleExecutionContractError(
            "batched sample adapter returned conflicting terminal reasons"
        )
    return next(iter(reasons))


def _bounded_public_steps(
    value: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the first cause and latest recovery/adjudication steps."""

    projected: list[dict[str, Any]] = []
    for item in value:
        public_item = dict(item)
        role = public_item.get("role")
        if (
            isinstance(role, str)
            and role.casefold().startswith("remote_")
            and "reason_code" in public_item
        ):
            public_item["reason_code"] = safe_remote_reason_code(
                public_item["reason_code"]
            )
        projected.append(public_item)
    if len(projected) <= _MAX_PUBLIC_STEPS:
        return projected
    head = _MAX_PUBLIC_STEPS // 2
    return projected[:head] + projected[-(_MAX_PUBLIC_STEPS - head) :]


def _sample_action(
    request: SamplePredictionRequest,
    adapter: SamplePredictionAdapter,
    *,
    agent_decisions: Sequence[Mapping[str, Any]],
    tool_calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    public_agents = _bounded_public_steps(agent_decisions)
    public_tools = _bounded_public_steps(tool_calls)
    return {
        "algorithm": {
            "id": request.algorithm_id,
            "version": request.algorithm_version,
        },
        "adapter": _adapter_identity(adapter),
        "agent_decisions": [
            {
                name: item[name]
                for name in ("role", "decision", "status", "model_id", "reason_code")
                if name in item
            }
            for item in public_agents
        ],
        "tool_calls": [
            {
                name: item[name]
                for name in ("tool_id", "version", "status")
                if name in item
            }
            for item in public_tools
        ],
    }


def _public_steps(value: Any, *, kind: str, id_field: str) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= _MAX_PUBLIC_STEPS:
        raise SampleExecutionContractError(
            f"sample response {kind} steps must contain 1-{_MAX_PUBLIC_STEPS} items"
        )
    result: list[dict[str, Any]] = []
    allowed = (
        {
            "role",
            "decision",
            "status",
            "model_id",
            "reason_code",
            "response_digest",
            "confidence",
        }
        if kind == "agent"
        else {
            "tool_id",
            "version",
            "status",
            "input_digest",
            "output_digest",
            "execution_owner",
            "dsh_tool_event_id",
            "dsh_tool_output_digest",
        }
    )
    required = (
        (id_field, "decision", "status")
        if kind == "agent"
        else (id_field, "version", "status")
    )
    for item in value:
        if not isinstance(item, Mapping) or set(item) - allowed:
            raise SampleExecutionContractError(
                f"sample response {kind} step has unsupported fields"
            )
        projected: dict[str, Any] = {}
        for field_name in required:
            raw = item.get(field_name)
            if not isinstance(raw, str) or not raw.strip():
                raise SampleExecutionContractError(
                    f"sample response {kind} step {field_name} must be non-empty text"
                )
            projected[field_name] = raw.strip()[:160]
        for field_name in allowed - set(required) - {"confidence"}:
            if field_name not in item:
                continue
            raw = item[field_name]
            if not isinstance(raw, str) or not raw.strip():
                raise SampleExecutionContractError(
                    f"sample response {kind} step {field_name} must be non-empty text"
                )
            projected[field_name] = raw.strip()[:200]
        if "confidence" in item:
            confidence = item["confidence"]
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1
            ):
                raise SampleExecutionContractError(
                    "sample response agent step confidence must be in [0, 1]"
                )
            projected["confidence"] = float(confidence)
        result.append(projected)
    return result


def _attach_execution_trace(
    record: dict[str, Any],
    agent_decisions: Sequence[Mapping[str, Any]],
    tool_calls: Sequence[Mapping[str, Any]],
) -> None:
    agents = [
        {
            name: item[name]
            for name in (
                "role",
                "model_id",
                "reason_code",
                "response_digest",
                "confidence",
            )
            if name in item
        }
        for item in _bounded_public_steps(agent_decisions)
        if item.get("response_digest")
    ]
    tools = [
        {
            name: item[name]
            for name in (
                "tool_id",
                "version",
                "status",
                "input_digest",
                "output_digest",
                "execution_owner",
                "dsh_tool_event_id",
                "dsh_tool_output_digest",
            )
            if name in item
        }
        for item in _bounded_public_steps(tool_calls)
        if item.get("input_digest") or item.get("output_digest")
    ]
    if agents:
        record["agent_trace"] = agents
        record["agent_trace_digest"] = digest(agents)
    if tools:
        record["tool_trace"] = tools
        record["tool_trace_digest"] = digest(tools)


def _sample_agent_chain_attestation(
    record: Mapping[str, Any],
    *,
    required_remote_roles: Sequence[str],
) -> dict[str, Any]:
    """Bind the minimal proof needed to resume a strict completed sample."""

    agent_trace = record.get("agent_trace")
    agents = (
        [dict(item) for item in agent_trace if isinstance(item, Mapping)]
        if isinstance(agent_trace, (list, tuple))
        else []
    )
    tool_trace = record.get("tool_trace")
    tools = (
        [dict(item) for item in tool_trace if isinstance(item, Mapping)]
        if isinstance(tool_trace, (list, tuple))
        else []
    )
    roles = [str(item.get("role") or "") for item in agents]
    reflection = record.get("sample_reflection")
    reflection = dict(reflection) if isinstance(reflection, Mapping) else {}
    planner_invocations = roles.count("remote_planner_agent")
    critic_invocations = roles.count("remote_critic_agent")
    reflector_invocations = roles.count("remote_reflector_agent")
    host_route_bypass_count = roles.count("host_deterministic_router")
    registered_tool_invocations = sum(
        str(item.get("tool_id") or "") != "physical-range-check"
        and str(item.get("status") or "") == "completed"
        for item in tools
    )
    dsh_agent_tool_invocations = sum(
        item.get("execution_owner") == "dsh_agent_tool_call"
        and isinstance(item.get("dsh_tool_event_id"), str)
        and bool(item.get("dsh_tool_event_id"))
        for item in tools
    )
    response_digest = reflection.get("response_digest")
    wave_digest = reflection.get("wave_digest")
    required_roles = tuple(required_remote_roles)
    role_counts = {
        "planner": planner_invocations,
        "critic": critic_invocations,
        "reflector": reflector_invocations,
    }
    required_roles_complete = all(
        role_counts.get(role, 0) >= 1 for role in required_roles
    )
    reflection_required = "reflector" in required_roles
    body = {
        "schema_version": SAMPLE_AGENT_CHAIN_ATTESTATION_VERSION,
        "sample_id": str(record.get("sample_id") or ""),
        "planner_invocations": planner_invocations,
        "registered_tool_invocations": registered_tool_invocations,
        "dsh_agent_tool_invocations": dsh_agent_tool_invocations,
        "critic_invocations": critic_invocations,
        "reflector_invocations": reflector_invocations,
        "host_route_bypass_count": host_route_bypass_count,
        "agent_trace_digest": record.get("agent_trace_digest"),
        "tool_trace_digest": record.get("tool_trace_digest"),
        "reflection_response_digest": (
            response_digest if isinstance(response_digest, str) else None
        ),
        "reflection_wave_digest": (
            wave_digest if isinstance(wave_digest, str) else None
        ),
        "required_remote_roles": list(required_roles),
        "complete": bool(
            required_roles_complete
            and registered_tool_invocations >= 1
            and dsh_agent_tool_invocations >= 1
            and host_route_bypass_count == 0
            and (not reflection_required or bool(response_digest))
            and (not reflection_required or bool(wave_digest))
        ),
    }
    return {**body, "attestation_digest": digest(body)}


def _attempt_trace_entry(
    attempt: int,
    decisions: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    outcome: str,
    requested_tool_id: Any = None,
) -> dict[str, Any] | None:
    """Project one remote attempt without retaining prompts or tool output bodies."""

    selected = next(
        (
            item
            for item in tools
            if item.get("tool_id") != "physical-range-check"
            and (
                item.get("input_digest") is not None
                or item.get("output_digest") is not None
            )
        ),
        None,
    )
    remote_decisions = [
        item
        for item in decisions
        if item.get("response_digest") is not None
        or str(item.get("role") or "").startswith("remote_")
    ]
    if selected is None and not remote_decisions:
        return None
    result: dict[str, Any] = {
        "attempt": max(0, int(attempt)),
        "outcome": str(outcome)[:80],
        "critic_decisions": [],
    }
    if selected is not None:
        result["selected_tool"] = {
            name: selected[name]
            for name in (
                "tool_id",
                "version",
                "status",
                "input_digest",
                "output_digest",
                "execution_owner",
                "dsh_tool_event_id",
                "dsh_tool_output_digest",
            )
            if name in selected
        }
    critic_decisions = []
    for item in remote_decisions:
        role = str(item.get("role") or "")
        if "critic" not in role.casefold():
            continue
        critic_decisions.append(
            {
                name: item[name]
                for name in (
                    "role",
                    "model_id",
                    "decision",
                    "status",
                    "reason_code",
                    "response_digest",
                )
                if name in item
            }
        )
    result["critic_decisions"] = critic_decisions
    if isinstance(requested_tool_id, str) and requested_tool_id.strip():
        result["requested_repair_tool"] = requested_tool_id.strip()[:160]
    return result


def _failed_record(
    request: SamplePredictionRequest,
    *,
    observed: float,
    attempts: int,
    category: str,
    retryable: bool,
    error_type: str,
    plan_digest: str | None,
    adapter: SamplePredictionAdapter,
) -> dict[str, Any]:
    record = {
        "schema_version": SAMPLE_EXECUTION_SCHEMA_VERSION,
        "sample_id": request.sample_id,
        "status": "failed",
        "attempts": attempts,
        "retry_count": max(0, attempts - 1),
        "target": request.target,
        "horizon_hours": request.horizon_hours,
        "predicted": None,
        "batch_plan_digest": plan_digest,
        "failure_action": (
            "skip_sample_before_agent_execution"
            if attempts == 0
            else "skip_sample_after_retry_budget"
        ),
        "failure": {
            "class": category[:120],
            "retryable": bool(retryable),
            "error_type": error_type[:120],
        },
    }
    record["failure_summary"] = _sample_failure_summary(
        request,
        adapter,
        category=category,
        attempts=attempts,
        failure=None,
    )
    return record


def _sample_failure_summary(
    request: SamplePredictionRequest,
    adapter: SamplePredictionAdapter,
    *,
    category: str,
    attempts: int,
    failure: BaseException | None,
) -> dict[str, Any]:
    """Keep bounded operational evidence without prompts or exception messages."""

    decisions, tools = (
        _exception_public_steps(failure) if failure is not None else ([], [])
    )
    decisions.append(
        {
            "role": "host_adjudicator",
            "decision": (
                "skip_sample_before_agent_execution"
                if attempts == 0
                else "skip_sample_after_retry_budget"
            ),
            "status": "completed",
            "reason_code": str(category)[:160],
        }
    )
    if not tools:
        adapter_identity = _adapter_identity(adapter)
        tools.append(
            {
                "tool_id": adapter_identity["id"],
                "version": adapter_identity["version"],
                "status": "failed",
            }
        )
    action = _sample_action(
        request,
        adapter,
        agent_decisions=decisions,
        tool_calls=tools,
    )
    return {
        "decisions": action["agent_decisions"],
        "tools": action["tool_calls"],
    }


def _fallback_scoring_row(
    row: Mapping[str, Any],
    request: SamplePredictionRequest,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep the cohort without allowing an execution failure to improve score."""

    result = dict(row)
    observed = _finite_float(row.get("observed"), "sample observed")
    alternatives = (
        [
            ("registered_algorithm_prediction", request.proposed_prediction),
            ("persistence_baseline", request.baseline),
        ]
        if request.proposed_prediction is not None
        else [
            ("persistence_baseline", request.baseline),
            ("registered_minimum_bound", request.minimum),
            ("registered_maximum_bound", request.maximum),
        ]
    )
    penalty_source, penalty_prediction = max(
        alternatives,
        key=lambda item: abs(float(item[1]) - observed),
    )
    result["sample_id"] = request.sample_id
    result["predicted"] = penalty_prediction
    result["sample_execution_status"] = "failed"
    result["sample_execution_attempts"] = int(record.get("attempts", 0))
    result["sample_execution_retry_count"] = int(record.get("retry_count", 0))
    result["sample_execution_failure"] = dict(record.get("failure", {}))
    failure_summary = record.get("failure_summary")
    if isinstance(failure_summary, Mapping):
        result["sample_execution_failure_summary"] = {
            "decisions": [
                dict(item)
                for item in failure_summary.get("decisions", ())
                if isinstance(item, Mapping)
            ],
            "tools": [
                dict(item)
                for item in failure_summary.get("tools", ())
                if isinstance(item, Mapping)
            ],
        }
    result["scoring_fallback"] = "failure_non_improvement_penalty"
    result["scoring_fallback_source"] = penalty_source
    return result


def bounded_sample_execution_records(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int = _MAX_PERSISTED_RECORD_PREVIEW,
) -> list[dict[str, Any]]:
    """Return a deterministic success/failure preview for routine projections."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("sample execution record preview limit must be positive")
    selected = [dict(item) for item in records[:limit]]
    selected_ids = {str(item.get("sample_id")) for item in selected}
    failures = [
        dict(item)
        for item in records
        if item.get("status") == "failed"
        and str(item.get("sample_id")) not in selected_ids
    ]
    if failures:
        reserve = min(16, len(failures), limit)
        selected = selected[: max(0, limit - reserve)] + failures[:reserve]
    return selected[:limit]


def encode_sample_execution_trace(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compress the complete per-sample audit without exposing it in list APIs."""

    projected = [dict(item) for item in records]
    raw = canonical_json(projected).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    return {
        "schema_version": SAMPLE_EXECUTION_TRACE_ARCHIVE_VERSION,
        "encoding": "zlib+base64+canonical-json",
        "record_count": len(projected),
        "uncompressed_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "trace_digest": digest(projected),
        "payload": base64.b64encode(compressed).decode("ascii"),
    }


def _adapter_identity(adapter: SamplePredictionAdapter) -> dict[str, str]:
    adapter_id = getattr(adapter, "adapter_id", type(adapter).__name__)
    version = getattr(adapter, "adapter_version", "unversioned")
    return {"id": str(adapter_id)[:160], "version": str(version)[:80]}


def _label_free_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SampleExecutionContractError(f"{name} must be an object")
    normalized = _label_free_value(value, name, depth=0)
    assert isinstance(normalized, dict)
    encoded = canonical_json(normalized).encode("utf-8")
    if len(encoded) > _MAX_SAMPLE_CONTEXT_BYTES:
        raise SampleExecutionContractError(
            f"{name} exceeds {_MAX_SAMPLE_CONTEXT_BYTES} bytes"
        )
    return normalized


def _label_free_value(value: Any, path: str, *, depth: int) -> Any:
    if depth > 8:
        raise SampleExecutionContractError(f"{path} is nested too deeply")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise SampleExecutionContractError(
                    f"{path} keys must be non-empty text"
                )
            key = raw_key.strip()
            normalized_key = key.casefold().replace("-", "_").replace(" ", "_")
            normalized_token = "".join(
                character for character in key.casefold() if character.isalnum()
            )
            if (
                normalized_key in _FORBIDDEN_SAMPLE_CONTEXT_KEYS
                or normalized_token in _FORBIDDEN_SAMPLE_CONTEXT_TOKENS
            ):
                raise SampleExecutionContractError(
                    f"{path} contains forbidden outcome field {key!r}"
                )
            result[key] = _label_free_value(
                raw_item,
                f"{path}.{key}",
                depth=depth + 1,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _label_free_value(item, f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SampleExecutionContractError(f"{path} must contain finite numbers")
        return value
    raise SampleExecutionContractError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SampleExecutionContractError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise SampleExecutionContractError(f"{name} must be finite")
    return number


def _optional_finite_prediction(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name)


__all__ = [
    "COVERAGE_UNREACHABLE_NOT_EXECUTED_FAILURE",
    "COVERAGE_UNREACHABLE_TERMINAL_REASON",
    "DEFAULT_SAMPLE_EXECUTION_MAX_ATTEMPTS",
    "DEFAULT_SAMPLE_EXECUTION_MIN_COVERAGE",
    "SAMPLE_EXECUTION_SCHEMA_VERSION",
    "SAMPLE_EXECUTION_TRACE_ARCHIVE_VERSION",
    "BatchedSamplePredictionAdapter",
    "CollaborativeSampleExecutor",
    "RegisteredToolCollaborationAdapter",
    "SampleExecutionAttemptError",
    "SampleExecutionBatch",
    "SampleExecutionContractError",
    "SampleExecutionPolicy",
    "SamplePredictionAdapter",
    "SamplePredictionOutcome",
    "SamplePredictionRequest",
    "SampleRepairRequired",
    "bounded_sample_execution_records",
    "classify_sample_failure",
    "encode_sample_execution_trace",
]
