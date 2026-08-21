"""Remote sample routing over host-owned prediction tools.

The gateway chooses a registered tool for each label-free sample. The host
executes that tool and retains the final physical constraint critic. Initial
decisions are microbatched; only sparse failed samples enter the repair role.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from queue import Empty, SimpleQueue
from threading import Condition, Event, Lock
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

from ..core.models import canonical_json, digest
from ..core.redaction import safe_remote_reason_code
from ..integrations.model_gateway import GatewayResponseError, ModelGateway
from .sample_execution import (
    COVERAGE_UNREACHABLE_NOT_EXECUTED_FAILURE,
    COVERAGE_UNREACHABLE_TERMINAL_REASON,
    SampleExecutionCancelledError,
    SampleExecutionControlError,
    SampleExecutionControlUnavailableError,
    SampleExecutionAttemptError,
    SampleExecutionContractError,
    SampleExecutionPausedError,
    SamplePredictionOutcome,
    SamplePredictionRequest,
    classify_sample_failure,
)
from .shared_sample_context import (
    build_origin_shared_routing_payload,
    normalized_sample_planner_prompt_profile,
)

_MAX_GATEWAY_BATCH_SIZE = 128
_MAX_GATEWAY_PAYLOAD_BYTES = 4_000_000
_MIN_GATEWAY_SPLIT_BATCH_SIZE = 8
_MAX_GATEWAY_SPLIT_DEPTH = 4
_DEFAULT_FINE_GRAINED_SPLIT_WAVE_SIZE = 16
_MAX_SAMPLE_CONCURRENCY = 8
_DEFAULT_SAMPLE_CONCURRENCY = 4
_MAX_SAMPLE_OUTPUT_TOKENS = 8_192
_TRUNCATION_RETRY_POLICY = "escalate_once@1"
_PROGRESS_HEARTBEAT_SECONDS = 60.0
_RUN_CONTROL_POLL_SECONDS = 1.0
_SAMPLE_OPERATION_MAX_TOKENS = frozenset(
    {"sample.planner", "sample.repair", "sample.critic"}
)
_ALWAYS_CRITIC_POLICY = "always@1"
_UNCERTAIN_OR_FAILURE_CRITIC_POLICY = "uncertain_or_failure@1"
_CAUSAL_PROVENANCE_SCHEMA_VERSION = "ecologyrsi-dsh.causal-sample-provenance/1"
_LEGACY_SPLIT_ELIGIBLE_GATEWAY_ERRORS = frozenset(
    {
        "model response must be a JSON object",
        "model response must contain exactly one choice",
        "model response choice must contain a message",
        "model response message content must be a JSON string",
        "model response content must contain one JSON object",
        "sample decision response.decisions must be an array",
        "sample decision response must contain exactly one decision per sample",
        "sample decision response must use every input sample_id exactly once",
    }
)
_SAMPLE_REPAIR_ELIGIBLE_GATEWAY_ERROR_CODES = frozenset(
    {
        "sample_decision_confidence_invalid",
        "sample_decision_format_invalid",
        "sample_decision_mapping_invalid",
        "sample_decision_tool_invalid",
    }
)
_ACCEPT_TOOL_ID = "accept"
_TERMINATE_TOOL_ID = "terminate"
_PROJECTION_TOOL_ID = "bounded-projection-repair"
_PERSISTENCE_TOOL_ID = "bounded-persistence-fallback"
_RESERVED_TOOL_IDS = frozenset(
    {
        _ACCEPT_TOOL_ID,
        _TERMINATE_TOOL_ID,
        _PROJECTION_TOOL_ID,
        _PERSISTENCE_TOOL_ID,
    }
)
_DECISION_CONTEXT_FIELDS = (
    "run_id",
    "candidate_id",
    "dataset_digest",
    "split_manifest_digest_sha256",
    "partition",
    "algorithm_id",
    "algorithm_version",
    "evaluator_id",
    "horizons_hours",
    "candidate_parameters",
    "derived_execution_plan",
    "sample_agent_mode",
    "sample_agent_batch_size",
    "sample_concurrency",
    "sample_planner_prompt_profile",
    "strategy_model_id",
    "review_model_id",
    "algorithm_artifact_digest",
    "tool_experience",
    "stage_context_digest",
    "candidate_genome_digest",
)
_FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "actual",
        "actual_value",
        "ground_truth",
        "label",
        "labels",
        "observed",
        "observation",
        "target_value",
    }
)
_FORBIDDEN_OUTCOME_TOKENS = frozenset(
    "".join(character for character in name if character.isalnum())
    for name in _FORBIDDEN_OUTCOME_KEYS
)


class SampleDecisionGateway(Protocol):
    """Narrow dependency implemented by a real or fake remote gateway."""

    def sample_decide(
        self,
        model_id: str,
        *,
        role: str,
        samples: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        available_tools: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Return decisions keyed by sample_id for one bounded microbatch."""


SampleToolHandler = Callable[
    [SamplePredictionRequest, Mapping[str, Any]],
    float | Mapping[str, Any],
]
SampleProgressCallback = Callable[[Mapping[str, Any]], None]
SampleModelUsageCallback = Callable[
    [Sequence[Mapping[str, Any]]], Mapping[str, Any] | None
]
SampleRunControlCallback = Callable[[], str]
SampleOutcomeCallback = Callable[
    [
        Sequence[SamplePredictionRequest],
        Sequence[SamplePredictionOutcome],
        Sequence[int],
    ],
    Mapping[str, str] | None,
]


@dataclass(frozen=True, slots=True)
class GatewaySampleTool:
    """One additional host-owned tool exposed to the remote router."""

    tool_id: str
    version: str
    handler: SampleToolHandler
    purpose: str = "registered_sample_prediction"

    def __post_init__(self) -> None:
        for name in ("tool_id", "version", "purpose"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"sample tool {name} must be non-empty text")
            object.__setattr__(self, name, value.strip())
        if self.tool_id in _RESERVED_TOOL_IDS:
            raise ValueError(f"sample tool id is reserved: {self.tool_id}")
        if not callable(self.handler):
            raise TypeError("sample tool handler must be callable")

    def descriptor(self) -> dict[str, str]:
        return {
            "tool_id": self.tool_id,
            "version": self.version,
            "purpose": self.purpose,
        }


@dataclass(slots=True)
class _ChunkRoutingDiagnostics:
    """Aggregate bounded routing recovery without retaining sample content."""

    gateway_request_count: int = 0
    adaptive_split_trigger_count: int = 0
    adaptive_split_count: int = 0
    adaptive_split_max_depth: int = 0
    adaptive_split_recovered_samples: int = 0
    adaptive_split_failed_samples: int = 0
    model_usage_receipts: list[dict[str, Any]] = field(default_factory=list)
    model_usage_receipt_sink: Callable[[Mapping[str, Any]], None] | None = field(
        default=None,
        repr=False,
    )
    model_call_admission: Callable[[str, int], None] | None = field(
        default=None,
        repr=False,
    )
    model_call_release: Callable[[str], None] | None = field(
        default=None,
        repr=False,
    )
    model_call_control: Callable[[], None] | None = field(
        default=None,
        repr=False,
    )


@dataclass(slots=True)
class _QueuedUsageReceipt:
    """One worker receipt awaiting coordinator-thread durability."""

    receipt: dict[str, Any]
    acknowledged: Event = field(default_factory=Event)
    error: BaseException | None = None


class ModelTokenBudgetExhaustedError(SampleExecutionControlError):
    """A durable token budget cannot admit another gateway wave."""

    error_code = "model_token_budget_exhausted"
    retryable = False

    def __init__(
        self,
        *,
        reason: str,
        token_limit: int,
        tokens_used: int,
        reserved_tokens: int,
        token_reservation_per_wave: int,
        missing_usage_call_count: int,
    ) -> None:
        self.reason = reason
        self.token_limit = token_limit
        self.tokens_used = tokens_used
        self.reserved_tokens = reserved_tokens
        self.token_reservation_per_wave = token_reservation_per_wave
        self.missing_usage_call_count = missing_usage_call_count
        super().__init__(
            "model token budget refused another gateway wave "
            f"(reason={reason}, used={tokens_used}, reserved={reserved_tokens}, "
            f"call_reservation={token_reservation_per_wave}, limit={token_limit}, "
            f"missing_usage_calls={missing_usage_call_count})"
        )


class _GatewayDecisionContractError(SampleExecutionContractError):
    """A decoded gateway response failed the local decision-only contract."""

    def __init__(self, message: str, *, repair_eligible: bool) -> None:
        super().__init__(message)
        self.repair_eligible = bool(repair_eligible)


class GatewaySampleCollaborationAdapter:
    """Use a remote model for routing and host tools for numerical execution."""

    adapter_id = "gateway-sample-collaboration"
    adapter_version = "9"

    def __init__(
        self,
        gateway: SampleDecisionGateway,
        *,
        strategy_model_id: str,
        review_model_id: str | None = None,
        remote_review_enabled: bool = False,
        forecast_tool: Callable[
            [SamplePredictionRequest], float | Mapping[str, Any]
        ]
        | None = None,
        tools: Sequence[GatewaySampleTool] = (),
        microbatch_size: int = _MAX_GATEWAY_BATCH_SIZE,
        sample_concurrency: int = _DEFAULT_SAMPLE_CONCURRENCY,
        minimum_split_batch_size: int | None = None,
        max_split_depth: int = _MAX_GATEWAY_SPLIT_DEPTH,
        progress_callback: SampleProgressCallback | None = None,
        model_usage_callback: SampleModelUsageCallback | None = None,
        run_control_callback: SampleRunControlCallback | None = None,
        operation_max_tokens: Mapping[str, int] | None = None,
        remote_critic_policy: Mapping[str, Any] | None = None,
        sample_planner_prompt_profile: Mapping[str, Any] | None = None,
        sample_truncation_retry_policy: Mapping[str, Any] | None = None,
        token_limit: int = 0,
        token_reservation_per_wave: int = 0,
    ) -> None:
        if not callable(getattr(gateway, "sample_decide", None)):
            raise TypeError("sample decision gateway must define sample_decide")
        self.gateway = gateway
        self.strategy_model_id = _text(strategy_model_id, "strategy_model_id")
        self.review_model_id = (
            _text(review_model_id, "review_model_id")
            if review_model_id is not None
            else None
        )
        if not isinstance(remote_review_enabled, bool):
            raise TypeError("remote_review_enabled must be a boolean")
        if remote_review_enabled and self.review_model_id is None:
            raise ValueError("remote review requires review_model_id")
        self.remote_review_enabled = remote_review_enabled
        self.remote_critic_policy = _normalized_remote_critic_policy(
            remote_critic_policy
        )
        self.sample_planner_prompt_profile = (
            normalized_sample_planner_prompt_profile(sample_planner_prompt_profile)
        )
        self.sample_truncation_retry_policy = _normalized_truncation_retry_policy(
            sample_truncation_retry_policy
        )
        if self.remote_critic_policy is not None and not remote_review_enabled:
            raise ValueError("remote critic policy requires remote review")
        if (
            isinstance(microbatch_size, bool)
            or not isinstance(microbatch_size, int)
            or not 1 <= microbatch_size <= _MAX_GATEWAY_BATCH_SIZE
        ):
            raise ValueError("microbatch_size must be between 1 and 128")
        self.microbatch_size = microbatch_size
        if (
            isinstance(sample_concurrency, bool)
            or not isinstance(sample_concurrency, int)
            or not 1 <= sample_concurrency <= _MAX_SAMPLE_CONCURRENCY
        ):
            raise ValueError("sample_concurrency must be between 1 and 8")
        self.sample_concurrency = sample_concurrency
        self._split_default_small_waves = minimum_split_batch_size is None
        if minimum_split_batch_size is None:
            minimum_split_batch_size = _MIN_GATEWAY_SPLIT_BATCH_SIZE
        elif (
            isinstance(minimum_split_batch_size, bool)
            or not isinstance(minimum_split_batch_size, int)
            or not 1 <= minimum_split_batch_size <= _MAX_GATEWAY_BATCH_SIZE
        ):
            raise ValueError("minimum_split_batch_size must be between 1 and 128")
        if (
            isinstance(max_split_depth, bool)
            or not isinstance(max_split_depth, int)
            or not 0 <= max_split_depth <= 8
        ):
            raise ValueError("max_split_depth must be between 0 and 8")
        self.minimum_split_batch_size = minimum_split_batch_size
        self.max_split_depth = max_split_depth
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable")
        self._progress_callback = progress_callback
        if model_usage_callback is not None and not callable(model_usage_callback):
            raise TypeError("model_usage_callback must be callable")
        self._model_usage_callback = model_usage_callback
        if run_control_callback is not None and not callable(run_control_callback):
            raise TypeError("run_control_callback must be callable")
        self._run_control_callback = run_control_callback
        self.operation_max_tokens = _normalized_operation_max_tokens(
            operation_max_tokens
        )
        self.token_limit = _token_budget_count(token_limit, "token_limit")
        self.token_reservation_per_wave = _token_budget_count(
            token_reservation_per_wave,
            "token_reservation_per_wave",
        )
        self.token_budget_enabled = bool(
            self.token_limit > 0 and self.token_reservation_per_wave > 0
        )
        if self.token_budget_enabled:
            # Preserve legacy checkpoint identity unless the frozen task opts
            # into the new hard-budget scheduler semantics.
            self.adapter_version = "11-token-call-budget"
        if self.sample_planner_prompt_profile is not None:
            self.adapter_version = (
                "12-origin-shared-context-token-call-budget"
                if self.token_budget_enabled
                else "12-origin-shared-context"
            )
        if self.sample_truncation_retry_policy is not None:
            self.adapter_version += "-truncation-retry"
        self._token_budget_tokens_used = 0
        self._token_budget_missing_call_count = 0
        # Installed only for the duration of one host executor call. It sees
        # label-free requests and tool outcomes; the host executor owns the
        # later merge with observations.
        self._outcome_callback: SampleOutcomeCallback | None = None
        self._durable_outcome_statuses: dict[str, str] | None = None
        self._planner_progress_session: dict[str, int] | None = None
        # Installed by the host executor only while resuming one persisted
        # planner cohort. Repair calls deliberately keep their own denominator.
        self._resume_checkpoint: dict[str, Any] | None = None
        self._forecast_tool = forecast_tool or _legacy_candidate_forecast
        indexed: dict[str, GatewaySampleTool] = {}
        for tool in tools:
            if not isinstance(tool, GatewaySampleTool):
                raise TypeError("tools must contain GatewaySampleTool values")
            if tool.tool_id in indexed:
                raise ValueError(f"duplicate sample tool id: {tool.tool_id}")
            indexed[tool.tool_id] = tool
        self._tools = indexed

    def set_outcome_callback(
        self, callback: SampleOutcomeCallback | None
    ) -> None:
        """Install a synchronous host-only microbatch outcome observer."""

        if callback is not None and not callable(callback):
            raise TypeError("outcome callback must be callable")
        self._outcome_callback = callback
        self._durable_outcome_statuses = {} if callback is not None else None
        self._planner_progress_session = None

    def set_resume_checkpoint(self, checkpoint: Mapping[str, Any] | None) -> None:
        """Install a host-validated aggregate baseline for one planner call."""

        self._resume_checkpoint = _normalized_resume_checkpoint(checkpoint)

    def set_token_budget_state(self, state: Mapping[str, Any] | None) -> None:
        """Install the durable run-wide usage baseline before gateway work."""

        if state is None:
            self._token_budget_tokens_used = 0
            self._token_budget_missing_call_count = 0
            return
        if not isinstance(state, Mapping):
            raise TypeError("token budget state must be an object")
        snapshot_limit = state.get("token_limit")
        if snapshot_limit is not None:
            normalized_limit = _token_budget_count(
                snapshot_limit,
                "token budget state token_limit",
            )
            if normalized_limit != self.token_limit:
                raise ValueError("token budget state does not match the frozen limit")
        raw_tokens_used = state.get("tokens_used", state.get("total_tokens", 0))
        raw_missing = state.get(
            "missing_call_count",
            state.get("missing_usage_call_count", 0),
        )
        self._token_budget_tokens_used = _token_budget_count(
            raw_tokens_used,
            "token budget state tokens_used",
        )
        self._token_budget_missing_call_count = _token_budget_count(
            raw_missing,
            "token budget state missing_call_count",
        )

    def _commit_persisted_model_usage(
        self,
        receipt: Mapping[str, Any],
        durable_state: Mapping[str, Any] | None,
        *,
        reservation_tokens: int = 0,
    ) -> None:
        """Advance budget counters only after the receipt callback succeeds."""

        if isinstance(durable_state, Mapping) and (
            "tokens_used" in durable_state or "total_tokens" in durable_state
        ):
            self.set_token_budget_state(durable_state)
            return
        if receipt.get("usage_reported") is True:
            self._token_budget_tokens_used += _token_budget_count(
                receipt.get("total_tokens"),
                "model usage receipt total_tokens",
            )
        elif self.token_budget_enabled and reservation_tokens > 0:
            self._token_budget_tokens_used += reservation_tokens
        else:
            self._token_budget_missing_call_count += 1

    def _token_budget_error(
        self,
        reason: str,
        *,
        reserved_tokens: int,
    ) -> ModelTokenBudgetExhaustedError:
        return ModelTokenBudgetExhaustedError(
            reason=reason,
            token_limit=self.token_limit,
            tokens_used=self._token_budget_tokens_used,
            reserved_tokens=reserved_tokens,
            token_reservation_per_wave=self.token_reservation_per_wave,
            missing_usage_call_count=self._token_budget_missing_call_count,
        )

    def _gateway_call_token_upper_bound(
        self,
        model_id: str,
        *,
        role: str,
        samples: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        available_tools: Sequence[Mapping[str, Any]],
        max_tokens: int | None,
    ) -> int:
        """Resolve the frozen conservative bound for one logical API call."""

        estimator = getattr(self.gateway, "sample_decide_token_upper_bound", None)
        if callable(estimator):
            value = estimator(
                model_id,
                role=role,
                samples=samples,
                context=context,
                available_tools=available_tools,
                max_tokens=max_tokens,
            )
        else:
            # Non-ModelGateway implementations are test/integration adapters.
            # Their hard-budget contract is the explicitly frozen per-call cap.
            value = self.token_reservation_per_wave
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise SampleExecutionContractError(
                "sample gateway returned an invalid token upper bound"
            )
        return value

    def _gateway_decide(
        self,
        model_id: str,
        *,
        role: str,
        samples: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        available_tools: Sequence[Mapping[str, Any]],
        allow_format_retry: bool,
        diagnostics: _ChunkRoutingDiagnostics,
    ) -> Mapping[str, Any]:
        if diagnostics.model_call_control is not None:
            diagnostics.model_call_control()
        logical_call_digest = digest(
            {
                "model_id": model_id,
                "role": role,
                "samples": list(samples),
                "context": dict(context),
                "available_tools": list(available_tools),
                "allow_format_retry": allow_format_retry,
            }
        )
        diagnostic_call = getattr(
            self.gateway, "sample_decide_with_diagnostics", None
        )
        configured_max_tokens = self.operation_max_tokens.get(f"sample.{role}")
        # Reasoning-capable providers occasionally consume the complete
        # response budget before emitting the bounded decision object.  A
        # single same-wave escalation avoids replaying the whole adaptive split
        # tree.  It is opt-in and frozen in the task manifest so legacy runs
        # retain their exact request/retry identity.
        max_tokens = configured_max_tokens
        truncation_retry_used = False
        while True:
            call_id = uuid4().hex
            request_limit = (
                {"max_tokens": max_tokens}
                if isinstance(self.gateway, ModelGateway) and max_tokens is not None
                else {}
            )
            admitted = False
            if self.token_budget_enabled:
                if diagnostics.model_call_admission is None:
                    raise self._token_budget_error(
                        "usage_persistence_unavailable",
                        reserved_tokens=0,
                    )
                call_upper_bound = self._gateway_call_token_upper_bound(
                    model_id,
                    role=role,
                    samples=samples,
                    context=context,
                    available_tools=available_tools,
                    max_tokens=max_tokens,
                )
                diagnostics.model_call_admission(call_id, call_upper_bound)
                admitted = True
            try:
                if callable(diagnostic_call):
                    response, raw_receipt = diagnostic_call(
                        model_id,
                        role=role,
                        samples=samples,
                        context=context,
                        available_tools=available_tools,
                        allow_format_retry=allow_format_retry,
                        **request_limit,
                    )
                else:
                    response = self.gateway.sample_decide(
                        model_id,
                        role=role,
                        samples=samples,
                        context=context,
                        available_tools=available_tools,
                        **request_limit,
                    )
                    raw_receipt = {"http_attempts": 1, "usage_reported": False}
            except Exception as exc:
                try:
                    _capture_model_usage_receipt(
                        diagnostics,
                        exc,
                        call_id=call_id,
                        logical_call_digest=logical_call_digest,
                        role=role,
                        model_id=model_id,
                        outcome="failed",
                    )
                finally:
                    if admitted and diagnostics.model_call_release is not None:
                        diagnostics.model_call_release(call_id)
                if (
                    self.sample_truncation_retry_policy is not None
                    and not truncation_retry_used
                    and role in {"planner", "repair", "critic"}
                    and isinstance(exc, GatewayResponseError)
                    and exc.error_code == "output_truncated"
                    and isinstance(max_tokens, int)
                    and max_tokens < int(
                        self.sample_truncation_retry_policy["max_tokens"]
                    )
                ):
                    truncation_retry_used = True
                    max_tokens = int(
                        self.sample_truncation_retry_policy["max_tokens"]
                    )
                    continue
                raise
            try:
                _capture_model_usage_receipt(
                    diagnostics,
                    raw_receipt,
                    call_id=call_id,
                    logical_call_digest=logical_call_digest,
                    role=role,
                    model_id=model_id,
                    outcome="succeeded",
                )
            finally:
                if admitted and diagnostics.model_call_release is not None:
                    diagnostics.model_call_release(call_id)
            return response

    def plan_batch(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        context_strategy = context.get("strategy_model_id")
        if context_strategy is not None and context_strategy != self.strategy_model_id:
            raise SampleExecutionContractError(
                "sample strategy model does not match the frozen task binding"
            )
        context_review = context.get("review_model_id")
        if (
            self.remote_review_enabled
            and context_review is not None
            and context_review != self.review_model_id
        ):
            raise SampleExecutionContractError(
                "sample review model does not match the frozen task binding"
            )
        raw_prompt_profile = context.get("sample_planner_prompt_profile")
        if raw_prompt_profile is not None:
            context_prompt_profile = normalized_sample_planner_prompt_profile(
                raw_prompt_profile
            )
            if context_prompt_profile != self.sample_planner_prompt_profile:
                raise SampleExecutionContractError(
                    "sample planner prompt profile does not match the frozen task binding"
                )
        safe_context_source = {
            name: context[name]
            for name in _DECISION_CONTEXT_FIELDS
            if name in context
        }
        if self.sample_planner_prompt_profile is not None:
            safe_context_source["sample_planner_prompt_profile"] = dict(
                self.sample_planner_prompt_profile
            )
        safe_context = _safe_mapping(
            safe_context_source,
            "sample batch decision context",
        )
        remote_roles = ["planner", "repair"]
        if self.remote_review_enabled:
            remote_roles.append("critic")
        plan: dict[str, Any] = {
            "plan_id": "gateway-route-host-tool-host-critic@1",
            "execution_mode": "per_sample_remote_route_host_tool_loop",
            "remote_sample_agents": True,
            "remote_roles": remote_roles,
            "host_roles": ["constraint_critic", "host_adjudicator"],
            "strategy_model_id": self.strategy_model_id,
            "review_model_id": self.review_model_id,
            "remote_review_enabled": self.remote_review_enabled,
            "microbatch_size": self.microbatch_size,
            "sample_concurrency": self.sample_concurrency,
            "gateway_batch_limit": _MAX_GATEWAY_BATCH_SIZE,
            "adaptive_split_policy": {
                "eligible_failure": "response_format_or_contract",
                "minimum_batch_size": self.minimum_split_batch_size,
                "default_small_wave_floor": (
                    2 if self._split_default_small_waves else None
                ),
                "default_fine_grained_wave_max_size": (
                    _DEFAULT_FINE_GRAINED_SPLIT_WAVE_SIZE
                    if self._split_default_small_waves
                    else None
                ),
                "maximum_depth": self.max_split_depth,
                "transport_retry_owner": "model_gateway",
            },
            "truncation_retry_policy": (
                dict(self.sample_truncation_retry_policy)
                if self.sample_truncation_retry_policy is not None
                else None
            ),
            "causal_batch_policy": {
                "mode": "verified_origin_cutoff_wave",
                "unverifiable_request": "singleton",
                "invalid_explicit_provenance": "reject_sample",
            },
            "decision_context": safe_context,
            "decision_context_digest": digest(safe_context),
            "tools": [
                {
                    "tool_id": str(context.get("algorithm_id") or "candidate-algorithm"),
                    "version": str(context.get("algorithm_version") or "unversioned"),
                    "purpose": "registered_candidate_prediction",
                },
                {
                    "tool_id": _PROJECTION_TOOL_ID,
                    "version": "1",
                    "purpose": "execute_candidate_then_bounded_projection",
                },
                {
                    "tool_id": _PERSISTENCE_TOOL_ID,
                    "version": "1",
                    "purpose": "bounded_persistence_prediction_or_repair",
                },
                *(tool.descriptor() for tool in self._tools.values()),
            ],
            "routing_policy": (
                "remote_planner_then_host_tool_then_remote_critic_then_"
                "host_constraint_critic;remote_failure_critic_selects_"
                "registered_repair_tool_or_terminate;remote_repair_only_"
                "after_critic_request"
            ),
            "forecast_value_source": "agent_selected_host_tool_on_demand",
        }
        if self.remote_critic_policy is not None:
            plan["remote_critic_policy"] = dict(self.remote_critic_policy)
            if self.remote_critic_policy["version"] == _ALWAYS_CRITIC_POLICY:
                plan["routing_policy"] = (
                    "remote_planner_then_host_tool_then_remote_critic_then_"
                    "host_constraint_critic;remote_critic_reviews_every_sample;"
                    "remote_critic_selects_registered_repair_tool_or_terminate"
                )
            else:
                plan["routing_policy"] = (
                    "remote_planner_then_host_tool_then_host_constraint_critic;"
                    "remote_critic_only_for_low_confidence_or_tool_failure;"
                    "remote_critic_selects_registered_repair_tool_or_terminate"
                )
        if self.sample_planner_prompt_profile is not None:
            plan["sample_planner_prompt_profile"] = dict(
                self.sample_planner_prompt_profile
            )
        coverage_stop_policy = _coverage_stop_policy(
            context.get("sample_execution_policy")
        )
        if coverage_stop_policy is not None:
            plan["coverage_stop_policy"] = coverage_stop_policy
        if self.token_budget_enabled:
            plan["token_budget_policy"] = {
                "mode": "hard_gateway_call_reservation",
                "reservation_scope": "logical_gateway_call_with_http_retries",
                "token_limit": self.token_limit,
                "token_reservation_per_wave": self.token_reservation_per_wave,
            }
        raw_execution_plan = context.get("derived_execution_plan")
        if isinstance(raw_execution_plan, Mapping):
            plan["derived_execution_plan"] = dict(raw_execution_plan)
        return plan

    def predict_sample(
        self,
        request: SamplePredictionRequest,
        plan: Mapping[str, Any],
        *,
        attempt: int,
    ) -> Mapping[str, Any]:
        outcome = self.predict_samples(
            (request,),
            (plan,),
            attempts=(attempt,),
        )[0]
        if outcome.error is not None:
            raise outcome.error
        assert outcome.result is not None
        return outcome.result

    def predict_samples(
        self,
        requests: Sequence[SamplePredictionRequest],
        plans: Sequence[Mapping[str, Any]],
        *,
        attempts: Sequence[int],
    ) -> Sequence[SamplePredictionOutcome]:
        if not (len(requests) == len(plans) == len(attempts)):
            raise ValueError("requests, plans, and attempts must have equal lengths")
        if not requests:
            return ()
        sample_ids = [request.sample_id for request in requests]
        if len(sample_ids) != len(set(sample_ids)):
            raise SampleExecutionContractError(
                "sample microbatch requires unique sample identifiers"
            )
        for request, attempt in zip(requests, attempts):
            if not isinstance(request, SamplePredictionRequest):
                raise TypeError("requests must contain SamplePredictionRequest values")
            if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
                raise ValueError("sample attempts must be positive integers")
            _safe_mapping(request.to_dict(), "sample gateway request")

        outcomes: list[SamplePredictionOutcome | None] = [None] * len(requests)
        grouped: dict[tuple[str, str, str, tuple[Any, ...]], list[int]] = defaultdict(
            list
        )
        for index, (plan, attempt) in enumerate(zip(plans, attempts)):
            plan_data = _safe_mapping(plan, "sample gateway plan")
            role = "planner" if attempt == 1 else "repair"
            context_digest = str(
                plan_data.get("decision_context_digest")
                or digest(plan_data.get("decision_context", {}))
            )
            requested_tool = (
                _requested_repair_tool(plan_data) if role == "repair" else None
            )
            try:
                causal_wave = _causal_wave_identity(requests[index], index=index)
            except SampleExecutionContractError as exc:
                outcomes[index] = SamplePredictionOutcome(
                    sample_id=requests[index].sample_id,
                    error=SampleExecutionAttemptError(
                        str(exc),
                        failure_class="invalid_causal_provenance",
                        retryable=False,
                        error_type=type(exc).__name__,
                    ),
                )
                continue
            grouped[
                (role, context_digest, requested_tool or "", causal_wave)
            ].append(index)

        schedules: list[tuple[str, Sequence[int]]] = []
        totals_by_role: dict[str, int] = defaultdict(int)
        batches_by_role: dict[str, int] = defaultdict(int)
        for (
            role,
            _context_digest,
            _requested_tool,
            _causal_wave,
        ), indices in grouped.items():
            chunks = list(_chunks(indices, self.microbatch_size))
            schedules.extend((role, chunk) for chunk in chunks)
            totals_by_role[role] += len(indices)
            batches_by_role[role] += len(chunks)

        completed_by_role: dict[str, int] = defaultdict(int)
        succeeded_by_role: dict[str, int] = defaultdict(int)
        batch_index_by_role: dict[str, int] = defaultdict(int)
        progress_id_by_role: dict[str, int] = defaultdict(int)
        routing_diagnostics_by_role: dict[str, _ChunkRoutingDiagnostics] = defaultdict(
            _ChunkRoutingDiagnostics
        )
        planner_baseline = self._resume_checkpoint
        if planner_baseline is not None and totals_by_role.get("planner", 0):
            completed_by_role["planner"] = int(
                planner_baseline["completed_samples"]
            )
            succeeded_by_role["planner"] = int(
                planner_baseline["succeeded_samples"]
            )
            batch_index_by_role["planner"] = int(planner_baseline["batch_index"])
            progress_id_by_role["planner"] = int(
                planner_baseline.get("progress_id", 0)
            )
            totals_by_role["planner"] = int(planner_baseline["total_samples"])
            batches_by_role["planner"] = max(
                int(planner_baseline["batch_count"]),
                int(planner_baseline["batch_index"])
                + batches_by_role["planner"],
            )
            routing_diagnostics_by_role["planner"] = _ChunkRoutingDiagnostics(
                gateway_request_count=int(
                    planner_baseline["gateway_request_count"]
                ),
                adaptive_split_trigger_count=int(
                    planner_baseline["adaptive_split_trigger_count"]
                ),
                adaptive_split_count=int(
                    planner_baseline["adaptive_split_count"]
                ),
                adaptive_split_max_depth=int(
                    planner_baseline["adaptive_split_max_depth"]
                ),
                adaptive_split_recovered_samples=int(
                    planner_baseline["adaptive_split_recovered_samples"]
                ),
                adaptive_split_failed_samples=int(
                    planner_baseline["adaptive_split_failed_samples"]
                ),
            )
        if self._durable_outcome_statuses is not None:
            session = self._planner_progress_session
            if session is None and totals_by_role.get("planner", 0):
                routing = routing_diagnostics_by_role["planner"]
                session = {
                    "completed_samples": completed_by_role["planner"],
                    "succeeded_samples": succeeded_by_role["planner"],
                    "total_samples": totals_by_role["planner"],
                    "batch_index": batch_index_by_role["planner"],
                    "batch_count": batches_by_role["planner"],
                    "progress_id": progress_id_by_role["planner"],
                    "gateway_request_count": routing.gateway_request_count,
                    "adaptive_split_trigger_count": (
                        routing.adaptive_split_trigger_count
                    ),
                    "adaptive_split_count": routing.adaptive_split_count,
                    "adaptive_split_max_depth": routing.adaptive_split_max_depth,
                    "adaptive_split_recovered_samples": (
                        routing.adaptive_split_recovered_samples
                    ),
                    "adaptive_split_failed_samples": (
                        routing.adaptive_split_failed_samples
                    ),
                }
                self._planner_progress_session = session
            elif session is not None:
                completed_by_role["planner"] = int(session["completed_samples"])
                succeeded_by_role["planner"] = int(session["succeeded_samples"])
                totals_by_role["planner"] = int(session["total_samples"])
                batch_index_by_role["planner"] = int(session["batch_index"])
                batches_by_role["planner"] = max(
                    int(session["batch_count"]),
                    int(session["batch_index"]),
                )
                progress_id_by_role["planner"] = int(session["progress_id"])
                routing_diagnostics_by_role["planner"] = _ChunkRoutingDiagnostics(
                    gateway_request_count=int(session["gateway_request_count"]),
                    adaptive_split_trigger_count=int(
                        session["adaptive_split_trigger_count"]
                    ),
                    adaptive_split_count=int(session["adaptive_split_count"]),
                    adaptive_split_max_depth=int(
                        session["adaptive_split_max_depth"]
                    ),
                    adaptive_split_recovered_samples=int(
                        session["adaptive_split_recovered_samples"]
                    ),
                    adaptive_split_failed_samples=int(
                        session["adaptive_split_failed_samples"]
                    ),
                )
        coverage_policy = _prediction_coverage_stop_policy(plans, attempts)
        usage_receipt_queue: SimpleQueue[
            dict[str, Any] | _QueuedUsageReceipt
        ] = SimpleQueue()
        coordinator_wakeup = Event()
        published_usage_call_ids: set[str] = set()
        unpublished_usage_receipts: dict[str, dict[str, Any]] = {}
        first_usage_callback_error: BaseException | None = None
        coordinator_error: BaseException | None = None
        budget_condition = Condition(Lock())
        control_lock = Lock()
        control_error: SampleExecutionControlError | None = None
        reservations_by_call_id: dict[str, int] = {}
        reserved_tokens = 0
        budget_error: ModelTokenBudgetExhaustedError | None = None

        def poll_run_control() -> SampleExecutionControlError | None:
            """Latch the first non-running control state for this invocation."""

            nonlocal control_error
            newly_latched = False
            with control_lock:
                # A paused invocation stays stopped even after a fast resume,
                # but keep polling so a later terminal cancellation can upgrade
                # the drain policy and suppress outcome/progress publication.
                paused_latched = isinstance(
                    control_error, SampleExecutionPausedError
                )
                if control_error is not None and not paused_latched:
                    return control_error
                if self._run_control_callback is None:
                    return control_error
                try:
                    raw_status = self._run_control_callback()
                except Exception:  # noqa: BLE001 - fail closed before API work
                    if not paused_latched:
                        control_error = SampleExecutionControlUnavailableError(
                            "sample run control status is unavailable"
                        )
                        newly_latched = True
                else:
                    status = str(raw_status).strip().casefold()
                    if status == "running":
                        return control_error
                    if status == "paused":
                        if not paused_latched:
                            control_error = SampleExecutionPausedError(
                                "sample execution paused"
                            )
                            newly_latched = True
                    elif status == "cancelled":
                        control_error = SampleExecutionCancelledError(
                            "sample execution cancelled"
                        )
                        newly_latched = True
                    else:
                        if not paused_latched:
                            control_error = SampleExecutionControlUnavailableError(
                                "sample run control returned an invalid status"
                            )
                            newly_latched = True
            if newly_latched:
                with budget_condition:
                    budget_condition.notify_all()
                coordinator_wakeup.set()
            return control_error

        def raise_if_run_stopped() -> None:
            stopped = poll_run_control()
            if stopped is not None:
                raise stopped

        def raise_if_model_call_stopped() -> None:
            if coordinator_error is not None:
                raise coordinator_error
            if first_usage_callback_error is not None:
                raise first_usage_callback_error
            raise_if_run_stopped()

        def cancelled_control_latched() -> bool:
            return isinstance(control_error, SampleExecutionCancelledError)

        def enqueue_usage_receipt(receipt: Mapping[str, Any]) -> None:
            if not self.token_budget_enabled and self._run_control_callback is None:
                usage_receipt_queue.put(dict(receipt))
                coordinator_wakeup.set()
                return
            queued = _QueuedUsageReceipt(dict(receipt))
            usage_receipt_queue.put(queued)
            coordinator_wakeup.set()
            # A recursive split or critic call must not spend against stale
            # usage. The coordinator remains the sole ledger writer and wakes
            # this worker only after the physical call is durable.
            while not queued.acknowledged.wait(timeout=_PROGRESS_HEARTBEAT_SECONDS):
                coordinator_wakeup.set()
            if queued.error is not None:
                raise queued.error

        def current_reserved_tokens() -> int:
            with budget_condition:
                return reserved_tokens

        def admit_model_call(call_id: str, call_upper_bound: int) -> None:
            nonlocal budget_error, reserved_tokens
            while True:
                raise_if_run_stopped()
                with budget_condition:
                    if coordinator_error is not None:
                        raise coordinator_error
                    if budget_error is not None:
                        raise budget_error
                    if first_usage_callback_error is not None:
                        raise first_usage_callback_error
                    if self._model_usage_callback is None:
                        budget_error = self._token_budget_error(
                            "usage_persistence_unavailable",
                            reserved_tokens=reserved_tokens,
                        )
                        raise budget_error
                    if self._token_budget_missing_call_count > 0:
                        budget_error = self._token_budget_error(
                            "usage_unreported",
                            reserved_tokens=reserved_tokens,
                        )
                        raise budget_error
                    if call_upper_bound > self.token_reservation_per_wave:
                        budget_error = self._token_budget_error(
                            "call_bound_exceeds_frozen_reservation",
                            reserved_tokens=reserved_tokens,
                        )
                        raise budget_error
                    if (
                        self._token_budget_tokens_used
                        + reserved_tokens
                        + call_upper_bound
                        <= self.token_limit
                    ):
                        reservations_by_call_id[call_id] = call_upper_bound
                        reserved_tokens += call_upper_bound
                        return
                    if reservations_by_call_id:
                        budget_condition.wait(timeout=_RUN_CONTROL_POLL_SECONDS)
                        continue
                    budget_error = self._token_budget_error(
                        "insufficient_remaining_budget",
                        reserved_tokens=reserved_tokens,
                    )
                    raise budget_error

        def release_model_call(call_id: str) -> None:
            nonlocal reserved_tokens
            with budget_condition:
                reservation = reservations_by_call_id.pop(call_id, 0)
                reserved_tokens = max(0, reserved_tokens - reservation)
                budget_condition.notify_all()

        def abort_coordinator(exc: BaseException) -> None:
            """Stop new API work and wake every worker blocked on admission."""

            nonlocal coordinator_error
            if coordinator_error is None:
                coordinator_error = exc
            with budget_condition:
                budget_condition.notify_all()
            coordinator_wakeup.set()

        def commit_model_call(
            receipt: Mapping[str, Any],
            durable_state: Mapping[str, Any] | None,
        ) -> None:
            nonlocal budget_error, reserved_tokens
            call_id = str(receipt["call_id"])
            with budget_condition:
                reservation = reservations_by_call_id.pop(call_id, 0)
                if self.token_budget_enabled:
                    self._commit_persisted_model_usage(
                        receipt,
                        durable_state,
                        reservation_tokens=reservation,
                    )
                    reported_total = receipt.get("total_tokens")
                    if (
                        receipt.get("usage_reported") is True
                        and isinstance(reported_total, int)
                        and not isinstance(reported_total, bool)
                        and reported_total > reservation
                        and budget_error is None
                    ):
                        budget_error = self._token_budget_error(
                            "call_usage_exceeded_reservation",
                            reserved_tokens=reserved_tokens,
                        )
                    elif (
                        self._token_budget_missing_call_count > 0
                        and budget_error is None
                    ):
                        budget_error = self._token_budget_error(
                            "usage_unreported",
                            reserved_tokens=reserved_tokens,
                        )
                    elif (
                        self._token_budget_tokens_used > self.token_limit
                        and budget_error is None
                    ):
                        budget_error = self._token_budget_error(
                            "token_limit_exceeded",
                            reserved_tokens=reserved_tokens,
                        )
                reserved_tokens = max(0, reserved_tokens - reservation)
                budget_condition.notify_all()

        def publish_usage_receipts(
            receipts: Sequence[Mapping[str, Any]],
        ) -> None:
            """Persist physical calls once, from the coordinator thread only."""

            nonlocal first_usage_callback_error
            for raw_receipt in receipts:
                receipt = dict(raw_receipt)
                call_id = str(receipt["call_id"])
                if (
                    call_id in published_usage_call_ids
                    or call_id in unpublished_usage_receipts
                ):
                    continue
                if first_usage_callback_error is not None:
                    unpublished_usage_receipts[call_id] = receipt
                    continue
                try:
                    durable_state = None
                    if self._model_usage_callback is not None:
                        durable_state = self._model_usage_callback((receipt,))
                except BaseException as exc:  # noqa: BLE001 - abort persistence
                    unpublished_usage_receipts[call_id] = receipt
                    first_usage_callback_error = exc
                else:
                    published_usage_call_ids.add(call_id)
                    commit_model_call(receipt, durable_state)

        def drain_usage_receipts() -> None:
            while True:
                try:
                    queued = usage_receipt_queue.get_nowait()
                except Empty:
                    return
                if isinstance(queued, _QueuedUsageReceipt):
                    try:
                        publish_usage_receipts((queued.receipt,))
                        queued.error = first_usage_callback_error
                    except BaseException as exc:  # noqa: BLE001 - wake worker
                        queued.error = exc
                        release_model_call(str(queued.receipt.get("call_id", "")))
                    finally:
                        queued.acknowledged.set()
                else:
                    publish_usage_receipts((queued,))

        def sync_planner_progress_session() -> None:
            if (
                self._durable_outcome_statuses is None
                or totals_by_role.get("planner", 0) < 1
            ):
                return
            routing = routing_diagnostics_by_role["planner"]
            self._planner_progress_session = {
                "completed_samples": completed_by_role["planner"],
                "succeeded_samples": succeeded_by_role["planner"],
                "total_samples": totals_by_role["planner"],
                "batch_index": batch_index_by_role["planner"],
                "batch_count": max(
                    batches_by_role["planner"], batch_index_by_role["planner"]
                ),
                "progress_id": progress_id_by_role["planner"],
                "gateway_request_count": routing.gateway_request_count,
                "adaptive_split_trigger_count": (
                    routing.adaptive_split_trigger_count
                ),
                "adaptive_split_count": routing.adaptive_split_count,
                "adaptive_split_max_depth": routing.adaptive_split_max_depth,
                "adaptive_split_recovered_samples": (
                    routing.adaptive_split_recovered_samples
                ),
                "adaptive_split_failed_samples": (
                    routing.adaptive_split_failed_samples
                ),
            }

        def publish_progress(
            role: str,
            *,
            progress_kind: str,
            in_flight_batches: int,
            queued_batches: int,
            batch_size: int,
            state: Mapping[str, int] | None = None,
        ) -> None:
            """Publish actual scheduler state without exposing sample content."""

            stopped = poll_run_control()
            if cancelled_control_latched():
                return
            # A paused run may still be draining physical requests. Waiting
            # heartbeats remain useful during that drain, and the final
            # ``drained`` heartbeat below gives the projection an exact zero
            # rather than leaving the pause-time snapshot on screen.
            if self._progress_callback is None:
                return
            progress_id_by_role[role] += 1
            routing_diagnostics = routing_diagnostics_by_role[role]
            state = state or {
                "batch_index": batch_index_by_role[role],
                "completed_samples": completed_by_role[role],
                "succeeded_samples": succeeded_by_role[role],
                "gateway_request_count": routing_diagnostics.gateway_request_count,
                "adaptive_split_trigger_count": routing_diagnostics.adaptive_split_trigger_count,
                "adaptive_split_count": routing_diagnostics.adaptive_split_count,
                "adaptive_split_max_depth": routing_diagnostics.adaptive_split_max_depth,
                "adaptive_split_recovered_samples": (
                    routing_diagnostics.adaptive_split_recovered_samples
                ),
                "adaptive_split_failed_samples": routing_diagnostics.adaptive_split_failed_samples,
            }
            completed = state["completed_samples"]
            succeeded = state["succeeded_samples"]
            split_recovered = min(
                state["adaptive_split_recovered_samples"], completed
            )
            split_failed = min(
                state["adaptive_split_failed_samples"],
                max(0, completed - split_recovered),
            )
            self._progress_callback(
                {
                    "schema_version": "ecologyrsi-dsh.sample-microbatch-progress/3",
                    "role": role,
                    "model_id": self.strategy_model_id,
                    "progress_id": progress_id_by_role[role],
                    "progress_kind": progress_kind,
                    "batch_index": state["batch_index"],
                    "batch_count": batches_by_role[role],
                    "batch_size": batch_size,
                    "completed_samples": completed,
                    "total_samples": totals_by_role[role],
                    "succeeded_samples": succeeded,
                    "failed_samples": completed - succeeded,
                    "in_flight_batches": in_flight_batches,
                    "queued_batches": queued_batches,
                    "gateway_request_count": state["gateway_request_count"],
                    "adaptive_split_trigger_count": state["adaptive_split_trigger_count"],
                    "adaptive_split_count": state["adaptive_split_count"],
                    "adaptive_split_max_depth": state["adaptive_split_max_depth"],
                    "adaptive_split_recovered_samples": split_recovered,
                    "adaptive_split_failed_samples": split_failed,
                }
            )
            if role == "planner":
                sync_planner_progress_session()

        def progress_state(role: str) -> dict[str, int]:
            routing_diagnostics = routing_diagnostics_by_role[role]
            return {
                "batch_index": batch_index_by_role[role],
                "completed_samples": completed_by_role[role],
                "succeeded_samples": succeeded_by_role[role],
                "gateway_request_count": routing_diagnostics.gateway_request_count,
                "adaptive_split_trigger_count": routing_diagnostics.adaptive_split_trigger_count,
                "adaptive_split_count": routing_diagnostics.adaptive_split_count,
                "adaptive_split_max_depth": routing_diagnostics.adaptive_split_max_depth,
                "adaptive_split_recovered_samples": routing_diagnostics.adaptive_split_recovered_samples,
                "adaptive_split_failed_samples": routing_diagnostics.adaptive_split_failed_samples,
            }

        def run_schedule(
            role: str, chunk: Sequence[int]
        ) -> tuple[
            list[SamplePredictionOutcome | None],
            _ChunkRoutingDiagnostics,
            GatewayResponseError | None,
            bool,
        ]:
            """Run one independent gateway microbatch in a worker.

            Workers never publish outcomes or mutate shared diagnostics.  This
            keeps the gateway calls concurrent while the coordinator below
            remains the sole owner of the ordered result/ledger merge.
            """

            local_outcomes: list[SamplePredictionOutcome | None] = [
                None
            ] * len(requests)
            local_diagnostics = _ChunkRoutingDiagnostics(
                model_usage_receipt_sink=enqueue_usage_receipt,
                model_call_admission=admit_model_call,
                model_call_release=release_model_call,
                model_call_control=raise_if_model_call_stopped,
            )
            try:
                self._route_chunk(
                    requests,
                    plans,
                    attempts,
                    chunk,
                    role=role,
                    outcomes=local_outcomes,
                    diagnostics=local_diagnostics,
                )
            except ModelTokenBudgetExhaustedError:
                # Admission records the shared stop reason. The coordinator
                # still drains every sibling already in flight before raising it.
                return local_outcomes, local_diagnostics, None, True
            except SampleExecutionControlError:
                # A pause/cancel boundary is run-scoped, never a sample failure.
                # The coordinator drains already-admitted siblings before raising.
                return local_outcomes, local_diagnostics, None, True
            except GatewayResponseError as exc:
                if exc.retryable:
                    # Preserve the original exhausted gateway error for the
                    # coordinator. It must drain sibling schedules before
                    # passing this retry boundary to the generation runner.
                    return local_outcomes, local_diagnostics, exc, False
                for index in chunk:
                    if local_outcomes[index] is None:
                        local_outcomes[index] = SamplePredictionOutcome(
                            sample_id=requests[index].sample_id,
                            error=exc,
                        )
            except BaseException as exc:  # noqa: BLE001 - isolate one chunk
                # A transport/client defect must not discard sibling samples.
                for index in chunk:
                    if local_outcomes[index] is None:
                        local_outcomes[index] = SamplePredictionOutcome(
                            sample_id=requests[index].sample_id,
                            error=exc,
                        )
            return local_outcomes, local_diagnostics, None, False

        def merge_diagnostics(
            target: _ChunkRoutingDiagnostics,
            source: _ChunkRoutingDiagnostics,
        ) -> None:
            for field_name in (
                "gateway_request_count",
                "adaptive_split_trigger_count",
                "adaptive_split_count",
                "adaptive_split_recovered_samples",
                "adaptive_split_failed_samples",
            ):
                setattr(
                    target,
                    field_name,
                    getattr(target, field_name) + getattr(source, field_name),
                )
            target.adaptive_split_max_depth = max(
                target.adaptive_split_max_depth,
                source.adaptive_split_max_depth,
            )

        def record_published_statuses(
            role: str,
            statuses: Mapping[str, str],
            *,
            durable_acknowledgement: bool,
            local_diagnostics: _ChunkRoutingDiagnostics,
        ) -> tuple[int, int]:
            """Advance role counters and the durable planner cohort once."""

            planner_increment = 0
            selected = dict(statuses)
            if durable_acknowledgement:
                assert self._durable_outcome_statuses is not None
                selected = {}
                for sample_id, status in statuses.items():
                    prior = self._durable_outcome_statuses.get(sample_id)
                    if prior is not None:
                        if prior != status:
                            raise SampleExecutionContractError(
                                "sample outcome callback changed a durable status"
                            )
                        continue
                    self._durable_outcome_statuses[sample_id] = status
                    selected[sample_id] = status

            completed_by_role[role] += len(selected)
            succeeded_by_role[role] += sum(
                status == "succeeded" for status in selected.values()
            )
            if selected:
                batch_index_by_role[role] += 1

            if (
                durable_acknowledgement
                and role != "planner"
                and totals_by_role.get("planner", 0) > 0
            ):
                merge_diagnostics(
                    routing_diagnostics_by_role["planner"], local_diagnostics
                )
                completed_by_role["planner"] += len(selected)
                succeeded_by_role["planner"] += sum(
                    status == "succeeded" for status in selected.values()
                )
                if selected:
                    batch_index_by_role["planner"] += 1
                    batches_by_role["planner"] = max(
                        batches_by_role["planner"],
                        batch_index_by_role["planner"],
                    )
                planner_increment = len(selected)
            if durable_acknowledgement or role == "planner":
                sync_planner_progress_session()
            return len(selected), planner_increment

        def publish_diagnostics(
            role: str,
            local_diagnostics: _ChunkRoutingDiagnostics,
        ) -> None:
            """Fallback-flush receipts before making a schedule observable."""

            publish_usage_receipts(local_diagnostics.model_usage_receipts)
            merge_diagnostics(
                routing_diagnostics_by_role[role], local_diagnostics
            )

        def publish_schedule(
            role: str,
            chunk: Sequence[int],
            local_outcomes: Sequence[SamplePredictionOutcome | None],
            local_diagnostics: _ChunkRoutingDiagnostics,
        ) -> tuple[int, int] | None:
            """Publish receipts and merge one completed schedule."""

            publish_diagnostics(role, local_diagnostics)
            if first_usage_callback_error is not None:
                return None
            poll_run_control()
            for index in chunk:
                outcomes[index] = local_outcomes[index]
            published_statuses = _legacy_published_outcome_statuses(
                requests,
                outcomes,
                chunk,
            )
            durable_acknowledgement = False
            if (
                self._outcome_callback is not None
                and not cancelled_control_latched()
            ):
                callback_outcomes = tuple(
                    outcomes[index]
                    if outcomes[index] is not None
                    else SamplePredictionOutcome(
                        sample_id=requests[index].sample_id,
                        error=SampleExecutionContractError(
                            "sample gateway omitted an execution outcome"
                        ),
                    )
                    for index in chunk
                )
                raw_published_statuses = self._outcome_callback(
                    tuple(requests[index] for index in chunk),
                    callback_outcomes,
                    tuple(attempts[index] for index in chunk),
                )
                if raw_published_statuses is not None:
                    durable_acknowledgement = True
                    published_statuses = _validated_published_outcome_statuses(
                        raw_published_statuses,
                        requests=requests,
                        indices=chunk,
                    )
            return record_published_statuses(
                role,
                published_statuses,
                durable_acknowledgement=durable_acknowledgement,
                local_diagnostics=local_diagnostics,
            )
        # Keep one long-lived bounded executor. Completed schedules are published
        # immediately, so one slow request cannot hide faster sibling progress.
        # Once coverage is unreachable, stop submitting work but drain and retain
        # every schedule that was already in flight.
        worker_limit = self.sample_concurrency
        if self.token_budget_enabled:
            # Do not submit more initial schedules than the durable budget can
            # reserve. Otherwise all workers race for the budget lock and an
            # arbitrary later sample can displace an earlier sample when only a
            # subset of the configured concurrency is affordable. Keep one
            # worker when no call fits so admission emits the durable, explicit
            # budget-exhausted boundary without making a gateway request.
            with budget_condition:
                remaining_tokens = max(
                    0, self.token_limit - self._token_budget_tokens_used
                )
            worker_limit = min(
                worker_limit,
                max(1, remaining_tokens // self.token_reservation_per_wave),
            )
        worker_count = min(worker_limit, max(1, len(schedules)))
        coverage_stop = (
            _coverage_stop_decision(
                requests,
                outcomes,
                coverage_policy,
                resume_checkpoint=planner_baseline,
            )
            if coverage_policy is not None
            else None
        )
        next_schedule = 0
        recoverable_gateway_error: GatewayResponseError | None = None
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="sample-gateway",
        ) as executor:
            in_flight: dict[Any, tuple[int, str, Sequence[int]]] = {}

            def coverage_refill_blocked(
                prospective_chunk: Sequence[int],
            ) -> bool:
                if coverage_policy is None:
                    return False
                speculative_failures = set(prospective_chunk)
                for _schedule_index, _role, chunk in in_flight.values():
                    speculative_failures.update(chunk)
                return (
                    _coverage_stop_decision(
                        requests,
                        outcomes,
                        coverage_policy,
                        resume_checkpoint=planner_baseline,
                        assumed_permanent_failures=speculative_failures,
                    )
                    is not None
                )

            def token_budget_window_allows(prospective_in_flight: int) -> bool:
                """Admit schedule windows deterministically in input order.

                Physical-call reservations happen inside worker threads.  If
                more schedules are submitted than the frozen budget can
                possibly reserve, thread lock acquisition would otherwise pick
                a nondeterministic subset of samples.  Treat every submitted
                schedule as one conservative call reservation until it leaves
                the executor window.  When the window is empty, the next
                earliest schedule is still allowed to run so its exact call
                bound can either fit or latch the canonical budget error.
                """

                if not self.token_budget_enabled or prospective_in_flight <= 1:
                    return True
                with budget_condition:
                    return (
                        self._token_budget_tokens_used
                        + prospective_in_flight * self.token_reservation_per_wave
                        <= self.token_limit
                    )

            def submit_available() -> None:
                nonlocal next_schedule
                opening_new_window = not in_flight
                while (
                    coverage_stop is None
                    and recoverable_gateway_error is None
                    and first_usage_callback_error is None
                    and coordinator_error is None
                    and budget_error is None
                    and len(in_flight) < worker_count
                    and next_schedule < len(schedules)
                ):
                    if poll_run_control() is not None:
                        return
                    role, chunk = schedules[next_schedule]
                    if not token_budget_window_allows(len(in_flight) + 1):
                        return
                    if (
                        not opening_new_window
                        and coverage_refill_blocked(chunk)
                    ):
                        return
                    future = executor.submit(run_schedule, role, chunk)
                    in_flight[future] = (next_schedule, role, chunk)
                    future.add_done_callback(
                        lambda _future: coordinator_wakeup.set()
                    )
                    next_schedule += 1

            submit_available()
            next_heartbeat_at = monotonic() + _PROGRESS_HEARTBEAT_SECONDS
            try:
                while in_flight:
                    coordinator_wakeup.clear()
                    drain_usage_receipts()
                    poll_run_control()
                    completed = {
                        future for future in in_flight if future.done()
                    }
                    if not completed:
                        if first_usage_callback_error is not None:
                            coordinator_wakeup.wait(
                                timeout=_PROGRESS_HEARTBEAT_SECONDS
                            )
                            continue
                        heartbeat_wait = max(0.0, next_heartbeat_at - monotonic())
                        if coordinator_wakeup.wait(timeout=heartbeat_wait):
                            continue
                        for role in sorted(
                            {
                                scheduled_role
                                for _index, scheduled_role, _chunk in in_flight.values()
                            }
                        ):
                            publish_progress(
                                role,
                                progress_kind="waiting",
                                in_flight_batches=sum(
                                    scheduled_role == role
                                    for _index, scheduled_role, _chunk in in_flight.values()
                                ),
                                queued_batches=sum(
                                    scheduled_role == role
                                    for scheduled_role, _chunk in schedules[next_schedule:]
                                ),
                                batch_size=0,
                            )
                        next_heartbeat_at = (
                            monotonic() + _PROGRESS_HEARTBEAT_SECONDS
                        )
                        continue
                    completed_results = []
                    for future in sorted(
                        completed, key=lambda item: in_flight[item][0]
                    ):
                        _schedule_index, role, chunk = in_flight.pop(future)
                        (
                            local_outcomes,
                            local_diagnostics,
                            schedule_gateway_error,
                            schedule_control_stopped,
                        ) = future.result()
                        completed_results.append(
                            (
                                role,
                                chunk,
                                local_outcomes,
                                local_diagnostics,
                                schedule_gateway_error,
                                schedule_control_stopped,
                            )
                        )

                    # Inspect the whole completed cohort before publishing any
                    # success. Otherwise a lower-index success can refill the
                    # sliding window before a same-wakeup retry boundary is
                    # observed from a sibling future.
                    for (
                        _role,
                        _chunk,
                        _local_outcomes,
                        _local_diagnostics,
                        schedule_gateway_error,
                        _schedule_control_stopped,
                    ) in completed_results:
                        if (
                            schedule_gateway_error is not None
                            and recoverable_gateway_error is None
                        ):
                            recoverable_gateway_error = schedule_gateway_error

                    completed_progress: list[
                        tuple[str, str, int, dict[str, int]]
                    ] = []
                    for (
                        role,
                        chunk,
                        local_outcomes,
                        local_diagnostics,
                        schedule_gateway_error,
                        schedule_control_stopped,
                    ) in completed_results:
                        # The worker enqueues before returning. Drain again so
                        # usage is durable before outcomes or progress publish.
                        drain_usage_receipts()
                        if first_usage_callback_error is not None:
                            continue
                        if schedule_gateway_error is not None:
                            # A split/route can finish some samples before a
                            # retryable gateway boundary is raised. Persist
                            # those outcomes now; the absent samples remain
                            # unresolved for the outer checkpoint retry.
                            partial_indices = tuple(
                                index
                                for index in chunk
                                if local_outcomes[index] is not None
                            )
                            if partial_indices:
                                published_counts = publish_schedule(
                                    role,
                                    partial_indices,
                                    local_outcomes,
                                    local_diagnostics,
                                )
                                if published_counts is not None:
                                    role_increment, planner_increment = (
                                        published_counts
                                    )
                                    if role_increment:
                                        completed_progress.append(
                                            (
                                                role,
                                                role,
                                                role_increment,
                                                progress_state(role),
                                            )
                                        )
                                    if planner_increment:
                                        completed_progress.append(
                                            (
                                                "planner",
                                                role,
                                                planner_increment,
                                                progress_state("planner"),
                                            )
                                        )
                            else:
                                publish_diagnostics(role, local_diagnostics)
                            continue
                        if schedule_control_stopped:
                            publish_diagnostics(role, local_diagnostics)
                            completed_indices = [
                                index
                                for index in chunk
                                if local_outcomes[index] is not None
                            ]
                            poll_run_control()
                            if (
                                completed_indices
                                and self._outcome_callback is not None
                                and not cancelled_control_latched()
                            ):
                                raw_published_statuses = self._outcome_callback(
                                    tuple(requests[index] for index in completed_indices),
                                    tuple(
                                        local_outcomes[index]
                                        for index in completed_indices
                                        if local_outcomes[index] is not None
                                    ),
                                    tuple(attempts[index] for index in completed_indices),
                                )
                                published_statuses = (
                                    _legacy_published_outcome_statuses(
                                        requests,
                                        local_outcomes,
                                        completed_indices,
                                    )
                                    if raw_published_statuses is None
                                    else _validated_published_outcome_statuses(
                                        raw_published_statuses,
                                        requests=requests,
                                        indices=completed_indices,
                                    )
                                )
                                role_increment, planner_increment = (
                                    record_published_statuses(
                                        role,
                                        published_statuses,
                                        durable_acknowledgement=(
                                            raw_published_statuses is not None
                                        ),
                                        local_diagnostics=local_diagnostics,
                                    )
                                )
                                if role_increment:
                                    completed_progress.append(
                                        (
                                            role,
                                            role,
                                            role_increment,
                                            progress_state(role),
                                        )
                                    )
                                if planner_increment:
                                    completed_progress.append(
                                        (
                                            "planner",
                                            role,
                                            planner_increment,
                                            progress_state("planner"),
                                        )
                                    )
                            continue
                        published_counts = publish_schedule(
                            role,
                            chunk,
                            local_outcomes,
                            local_diagnostics,
                        )
                        if published_counts is None:
                            continue
                        role_increment, planner_increment = published_counts
                        if coverage_policy is not None and coverage_stop is None:
                            coverage_stop = _coverage_stop_decision(
                                requests,
                                outcomes,
                                coverage_policy,
                                resume_checkpoint=planner_baseline,
                            )
                        if role_increment:
                            completed_progress.append(
                                (
                                    role,
                                    role,
                                    role_increment,
                                    progress_state(role),
                                )
                            )
                        if planner_increment:
                            completed_progress.append(
                                (
                                    "planner",
                                    role,
                                    planner_increment,
                                    progress_state("planner"),
                                )
                            )

                    # Refill only after every completion observed in this tick
                    # has contributed its stop conditions.  Partial-window
                    # admission also projects each prospective chunk so several
                    # open slots cannot jointly overshoot the failure budget.
                    submit_available()
                    coalesced_progress: dict[
                        tuple[str, str], tuple[int, dict[str, int]]
                    ] = {}
                    for (
                        progress_role,
                        scheduler_role,
                        batch_size,
                        state,
                    ) in completed_progress:
                        key = (progress_role, scheduler_role)
                        previous = coalesced_progress.get(key)
                        coalesced_progress[key] = (
                            batch_size + (previous[0] if previous is not None else 0),
                            state,
                        )
                    # Result batches are durable before progress is emitted. If
                    # several futures complete in one coordinator tick, publish
                    # only the final aggregate so a later telemetry failure
                    # cannot leave an obsolete 8/9 heartbeat after 9/9 rows.
                    for (
                        progress_role,
                        scheduler_role,
                    ), (batch_size, state) in coalesced_progress.items():
                        publish_progress(
                            progress_role,
                            progress_kind="completed_batch",
                            in_flight_batches=sum(
                                scheduled_role == scheduler_role
                                for _index, scheduled_role, _chunk in in_flight.values()
                            ),
                            queued_batches=sum(
                                scheduled_role == scheduler_role
                                for scheduled_role, _chunk in schedules[next_schedule:]
                            ),
                            batch_size=batch_size,
                            state=state,
                        )
                    next_heartbeat_at = (
                        monotonic() + _PROGRESS_HEARTBEAT_SECONDS
                    )
            except BaseException as exc:  # noqa: BLE001 - drain before rethrow
                abort_coordinator(exc)
            finally:
                # A coordinator callback can fail while another physical call
                # is still running. Keep acknowledging late usage receipts so
                # workers cannot deadlock ThreadPoolExecutor.__exit__.
                while in_flight and not all(
                    future.done() for future in in_flight
                ):
                    coordinator_wakeup.clear()
                    try:
                        drain_usage_receipts()
                    except BaseException as exc:  # noqa: BLE001 - keep draining
                        abort_coordinator(exc)
                    if all(future.done() for future in in_flight):
                        break
                    coordinator_wakeup.wait(
                        timeout=_PROGRESS_HEARTBEAT_SECONDS
                    )
                try:
                    drain_usage_receipts()
                except BaseException as exc:  # noqa: BLE001 - preserve first error
                    abort_coordinator(exc)

        # The scheduler owns the physical in-flight state, while the run
        # control event only records the operator boundary.  Once the executor
        # context has joined every worker, persist a terminal drain heartbeat
        # so a paused run cannot be mistaken for one that still has active
        # requests.  Cancellation deliberately remains silent because its
        # outcome stream is terminal and must not publish late progress.
        if (
            isinstance(control_error, SampleExecutionPausedError)
            and self._progress_callback is not None
        ):
            for role in sorted(batches_by_role):
                try:
                    publish_progress(
                        role,
                        progress_kind="drained",
                        in_flight_batches=0,
                        queued_batches=0,
                        batch_size=0,
                        state=progress_state(role),
                    )
                except BaseException as exc:  # noqa: BLE001 - preserve pause
                    if coordinator_error is None:
                        coordinator_error = exc

        if coordinator_error is not None:
            raise coordinator_error

        if first_usage_callback_error is not None:
            raise first_usage_callback_error

        if control_error is not None:
            raise control_error

        if budget_error is not None:
            raise self._token_budget_error(
                budget_error.reason,
                reserved_tokens=current_reserved_tokens(),
            )

        if recoverable_gateway_error is not None:
            raise recoverable_gateway_error

        if coverage_stop is not None:
            final_coverage_stop = _coverage_stop_decision(
                requests,
                outcomes,
                coverage_policy,
                resume_checkpoint=planner_baseline,
            )
            _mark_coverage_terminal_outcomes(
                requests,
                outcomes,
                final_coverage_stop or coverage_stop,
            )

        return tuple(
            outcome
            if outcome is not None
            else SamplePredictionOutcome(
                sample_id=request.sample_id,
                error=SampleExecutionContractError(
                    "sample gateway omitted an execution outcome"
                ),
            )
            for request, outcome in zip(requests, outcomes)
        )

    def _route_chunk(
        self,
        requests: Sequence[SamplePredictionRequest],
        plans: Sequence[Mapping[str, Any]],
        attempts: Sequence[int],
        indices: Sequence[int],
        *,
        role: str,
        outcomes: list[SamplePredictionOutcome | None],
        diagnostics: _ChunkRoutingDiagnostics,
        split_depth: int = 0,
        split_floor: int | None = None,
    ) -> None:
        if split_floor is None:
            split_floor = (
                2
                if self._split_default_small_waves
                and len(indices) <= _DEFAULT_FINE_GRAINED_SPLIT_WAVE_SIZE
                else self.minimum_split_batch_size
            )
        diagnostics.adaptive_split_max_depth = max(
            diagnostics.adaptive_split_max_depth,
            split_depth,
        )
        requested_tools = {
            _requested_repair_tool(plans[index]) for index in indices
        }
        forced_tool = (
            next(iter(requested_tools))
            if role == "repair"
            and len(requested_tools) == 1
            and None not in requested_tools
            else None
        )
        if forced_tool is not None:
            decisions = {
                requests[index].sample_id: {
                    "sample_id": requests[index].sample_id,
                    "next_tool": forced_tool,
                    "reason_code": "critic_requested_tool",
                    "confidence": 1.0,
                    "response_digest": digest(
                        {
                            "role": "critic_repair_router",
                            "sample_id": requests[index].sample_id,
                            "attempt": attempts[index],
                            "requested_tool_id": forced_tool,
                        }
                    ),
                }
                for index in indices
            }
        else:
            if self.sample_planner_prompt_profile is not None:
                sample_payloads, shared_sample_contexts = (
                    build_origin_shared_routing_payload(
                        [requests[index] for index in indices],
                        [attempts[index] for index in indices],
                        [_retry_feedback(plans[index]) for index in indices],
                    )
                )
                context = self._gateway_context(
                    plans[indices[0]],
                    role=role,
                    shared_sample_contexts=shared_sample_contexts,
                )
            else:
                sample_payloads = [
                    self._routing_sample(
                        requests[index], plans[index], attempts[index]
                    )
                    for index in indices
                ]
                context = self._gateway_context(plans[indices[0]], role=role)
            available_tools = _tool_union(
                self._available_tool_catalog(
                    requests[index], plans[index], role=role
                )
                for index in indices
            )
            try:
                raw_response = self._gateway_decide(
                    self.strategy_model_id,
                    role=role,
                    samples=sample_payloads,
                    context=context,
                    available_tools=available_tools,
                    allow_format_retry=split_depth == 0,
                    diagnostics=diagnostics,
                )
                try:
                    decisions = _validated_decisions(
                        raw_response,
                        [requests[index].sample_id for index in indices],
                        model_id=self.strategy_model_id,
                        role=role,
                    )
                except _GatewayDecisionContractError:
                    raise
                except SampleExecutionContractError as exc:
                    raise _GatewayDecisionContractError(
                        "sample gateway decision response violated its contract",
                        repair_eligible=False,
                    ) from exc
            except SampleExecutionControlError:
                raise
            except GatewayResponseError as exc:
                if exc.retryable:
                    raise
                self._handle_routing_failure(
                    exc,
                    requests,
                    plans,
                    attempts,
                    indices,
                    role=role,
                    outcomes=outcomes,
                    diagnostics=diagnostics,
                    split_depth=split_depth,
                    split_floor=split_floor,
                )
                return
            except Exception as exc:  # noqa: BLE001 - isolate remote microbatch
                self._handle_routing_failure(
                    exc,
                    requests,
                    plans,
                    attempts,
                    indices,
                    role=role,
                    outcomes=outcomes,
                    diagnostics=diagnostics,
                    split_depth=split_depth,
                    split_floor=split_floor,
                )
                return

        if split_depth > 0:
            diagnostics.adaptive_split_recovered_samples += len(indices)

        successful: list[dict[str, Any]] = []
        tool_failures: list[dict[str, Any]] = []
        for index in indices:
            request = requests[index]
            attempt = attempts[index]
            decision = decisions[request.sample_id]
            next_tool = str(decision["next_tool"])
            catalog = {
                item["tool_id"]: item
                for item in self._available_tool_catalog(
                    request, plans[index], role=role
                )
            }
            requested_repair_tool = (
                _requested_repair_tool(plans[index]) if role == "repair" else None
            )
            agent_steps = (
                []
                if requested_repair_tool is not None
                else [
                    _agent_step(
                        decision,
                        role=role,
                        model_id=self.strategy_model_id,
                    )
                ]
            )
            if requested_repair_tool is not None:
                next_tool = requested_repair_tool
                agent_steps.append(
                    {
                        "role": "critic_repair_router",
                        "decision": f"execute_requested_tool:{next_tool}",
                        "status": "completed",
                        "reason_code": "critic_requested_tool",
                        "response_digest": decision["response_digest"],
                    }
                )
            if next_tool not in catalog:
                outcomes[index] = SamplePredictionOutcome(
                    sample_id=request.sample_id,
                    error=SampleExecutionContractError(
                        f"remote {role} selected an unavailable tool: {next_tool}"
                    ),
                )
                continue
            tool_input = {
                "sample": request.to_dict(),
                "attempt": attempt,
                "selected_tool": next_tool,
                "reason_code": decision["reason_code"],
                "batch_plan_digest": digest(plans[index]),
                "algorithm_artifact_digest": (
                    plans[index].get("decision_context", {}).get(
                        "algorithm_artifact_digest"
                    )
                    if isinstance(plans[index].get("decision_context"), Mapping)
                    else None
                ),
            }
            input_digest = digest(tool_input)
            try:
                predicted, output_digest = self._invoke_tool(
                    request,
                    next_tool,
                    attempt=attempt,
                    plan=plans[index],
                )
            except Exception as exc:  # noqa: BLE001 - isolate registered sample tool
                failure_class, retryable, error_type = classify_sample_failure(exc)
                failed_tool = {
                    "tool_id": next_tool,
                    "version": str(catalog[next_tool]["version"]),
                    "status": "failed",
                    "input_digest": input_digest,
                }
                tool_failures.append(
                    {
                        "index": index,
                        "request": request,
                        "attempt": attempt,
                        "agent_steps": agent_steps,
                        "tool_step": failed_tool,
                        "failure_class": failure_class,
                        "host_retryable": retryable,
                        "error_type": error_type,
                    }
                )
                continue
            tool_step = {
                "tool_id": next_tool,
                "version": str(catalog[next_tool]["version"]),
                "status": "completed",
                "input_digest": input_digest,
                "output_digest": output_digest,
            }
            successful.append(
                {
                    "index": index,
                    "request": request,
                    "predicted": predicted,
                    "agent_steps": agent_steps,
                    "tool_step": tool_step,
                    "decision": decision,
                }
            )

        if not self.remote_review_enabled:
            for item in successful:
                request = item["request"]
                outcomes[item["index"]] = SamplePredictionOutcome(
                    sample_id=request.sample_id,
                    result={
                        "predicted": item["predicted"],
                        "agent_decisions": list(item["agent_steps"]),
                        "tool_calls": [item["tool_step"]],
                    },
                )
            for item in tool_failures:
                request = item["request"]
                outcomes[item["index"]] = SamplePredictionOutcome(
                    sample_id=request.sample_id,
                    error=SampleExecutionAttemptError(
                        f"registered sample tool failed: {item['error_type']}",
                        failure_class=str(item["failure_class"]),
                        retryable=bool(item["host_retryable"]),
                        error_type=str(item["error_type"]),
                        agent_decisions=tuple(item["agent_steps"]),
                        tool_calls=(item["tool_step"],),
                    ),
                )
            return
        if successful:
            if (
                self.remote_critic_policy is None
                or self.remote_critic_policy["version"] == _ALWAYS_CRITIC_POLICY
            ):
                self._review_successes(successful, plans, outcomes, diagnostics)
            else:
                threshold = self.remote_critic_policy["min_planner_confidence"]
                uncertain = [
                    item
                    for item in successful
                    if float(item["decision"]["confidence"]) < threshold
                ]
                uncertain_indices = {int(item["index"]) for item in uncertain}
                for item in successful:
                    if int(item["index"]) not in uncertain_indices:
                        request = item["request"]
                        outcomes[item["index"]] = SamplePredictionOutcome(
                            sample_id=request.sample_id,
                            result={
                                "predicted": item["predicted"],
                                "agent_decisions": list(item["agent_steps"]),
                                "tool_calls": [item["tool_step"]],
                            },
                        )
                if uncertain:
                    self._review_successes(
                        uncertain,
                        plans,
                        outcomes,
                        diagnostics,
                        compact=True,
                    )
        if tool_failures:
            self._review_tool_failures(
                tool_failures,
                plans,
                outcomes,
                diagnostics,
                compact=self.remote_critic_policy is not None,
            )

    def _handle_routing_failure(
        self,
        exc: BaseException,
        requests: Sequence[SamplePredictionRequest],
        plans: Sequence[Mapping[str, Any]],
        attempts: Sequence[int],
        indices: Sequence[int],
        *,
        role: str,
        outcomes: list[SamplePredictionOutcome | None],
        diagnostics: _ChunkRoutingDiagnostics,
        split_depth: int,
        split_floor: int,
    ) -> None:
        split_eligible = _is_adaptive_split_failure(exc)
        if split_eligible:
            diagnostics.adaptive_split_trigger_count += 1
        if (
            split_eligible
            and split_depth < self.max_split_depth
            and len(indices) > split_floor
        ):
            diagnostics.adaptive_split_count += 1
            midpoint = len(indices) // 2
            for child_indices in (indices[:midpoint], indices[midpoint:]):
                self._route_chunk(
                    requests,
                    plans,
                    attempts,
                    child_indices,
                    role=role,
                    outcomes=outcomes,
                    diagnostics=diagnostics,
                    split_depth=split_depth + 1,
                    split_floor=split_floor,
                )
            return

        failure_class, _retryable, error_type = classify_sample_failure(exc)
        if split_eligible:
            failure_class = "invalid_output"
        if split_depth > 0 or split_eligible:
            diagnostics.adaptive_split_failed_samples += len(indices)
        # Splitting and agent repair have separate budgets. Batch-level syntax,
        # truncation, and completeness errors may benefit from less response
        # context, but replaying their exhausted split tree as another sample
        # wave only multiplies the same provider failure. A single decoded
        # decision error remains actionable, and receives at most one repair
        # wave after the planner attempt.
        repair_eligible = (
            role != "repair" and _is_sample_repair_eligible_failure(exc)
        )
        batch_error = SampleExecutionAttemptError(
            f"remote {role} microbatch failed: {type(exc).__name__}",
            failure_class=f"remote_batch_{failure_class}",
            retryable=repair_eligible,
            error_type=error_type,
        )
        for index in indices:
            outcomes[index] = SamplePredictionOutcome(
                sample_id=requests[index].sample_id,
                error=batch_error,
            )

    def _review_tool_failures(
        self,
        failures: Sequence[Mapping[str, Any]],
        plans: Sequence[Mapping[str, Any]],
        outcomes: list[SamplePredictionOutcome | None],
        diagnostics: _ChunkRoutingDiagnostics,
        *,
        compact: bool = False,
    ) -> None:
        """Let the remote critic choose recovery or termination after host failure."""

        assert self.review_model_id is not None
        for chunk in _chunks(list(range(len(failures))), self.microbatch_size):
            selected = []
            for index in chunk:
                item = dict(failures[index])
                item["recovery_tools"] = self._recovery_tool_catalog(
                    item["request"],
                    plans[item["index"]],
                    failed_tool_id=str(item["tool_step"]["tool_id"]),
                )
                selected.append(item)
            review_samples = (
                [
                    _compact_critic_failure_sample(item)
                    for item in selected
                ]
                if compact
                else [
                    {
                        "sample_id": item["request"].sample_id,
                        "sample": item["request"].to_dict(),
                        "attempt": item["attempt"],
                        "failure_feedback": _retry_feedback(
                            plans[item["index"]]
                        ),
                        "tool_failure": {
                            "failure_class": item["failure_class"],
                            "host_retryable": item["host_retryable"],
                            "error_type": item["error_type"],
                            "selected_tool": item["tool_step"],
                        },
                        "available_recovery_tools": item["recovery_tools"],
                    }
                    for item in selected
                ]
            )
            available_tools = [
                {
                    "tool_id": _TERMINATE_TOOL_ID,
                    "version": "1",
                    "purpose": "terminate_sample_after_reviewed_tool_failure",
                },
                *_tool_union(item["recovery_tools"] for item in selected),
            ]
            try:
                raw_response = self._gateway_decide(
                    self.review_model_id,
                    role="critic",
                    samples=review_samples,
                    context=self._gateway_context(
                        plans[selected[0]["index"]], role="critic", compact=compact
                    ),
                    available_tools=available_tools,
                    allow_format_retry=True,
                    diagnostics=diagnostics,
                )
                decisions = _validated_decisions(
                    raw_response,
                    [item["request"].sample_id for item in selected],
                    model_id=self.review_model_id,
                    role="critic",
                )
            except SampleExecutionControlError:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate critic microbatch
                if isinstance(exc, GatewayResponseError) and exc.retryable:
                    raise
                critic_failure, classified_retryable, _error_type = (
                    classify_sample_failure(exc)
                )
                critic_retryable = (
                    _is_sample_repair_eligible_failure(exc)
                    if _is_adaptive_split_failure(exc)
                    else classified_retryable
                )
                for item, review_sample in zip(selected, review_samples):
                    request = item["request"]
                    critic_step = {
                        "role": "remote_critic_agent",
                        "decision": (
                            "retry_after_critic_review_failure"
                            if critic_retryable
                            else "terminate_after_critic_review_failure"
                        ),
                        "status": "failed",
                        "model_id": self.review_model_id,
                        "reason_code": f"critic_unavailable_{critic_failure}"[:160],
                        "response_digest": digest(
                            {
                                "model_id": self.review_model_id,
                                "role": "critic",
                                "status": "failed",
                                "failure_class": critic_failure,
                                "sample": review_sample,
                            }
                        ),
                    }
                    outcomes[item["index"]] = SamplePredictionOutcome(
                        sample_id=request.sample_id,
                        error=SampleExecutionAttemptError(
                            "registered sample tool failed and remote critic "
                            "review was unavailable",
                            failure_class=str(item["failure_class"]),
                            retryable=critic_retryable,
                            error_type=str(item["error_type"]),
                            agent_decisions=(*item["agent_steps"], critic_step),
                            tool_calls=(item["tool_step"],),
                        ),
                    )
                continue

            for item in selected:
                request = item["request"]
                decision = decisions[request.sample_id]
                critic_step = _agent_step(
                    decision,
                    role="critic",
                    model_id=self.review_model_id,
                )
                next_tool = str(decision["next_tool"])
                if next_tool == _TERMINATE_TOOL_ID:
                    critic_step["decision"] = "reject_and_terminate_after_tool_failure"
                    outcomes[item["index"]] = SamplePredictionOutcome(
                        sample_id=request.sample_id,
                        error=SampleExecutionAttemptError(
                            "remote critic terminated sample after tool failure",
                            failure_class=str(item["failure_class"]),
                            retryable=False,
                            error_type=str(item["error_type"]),
                            agent_decisions=(*item["agent_steps"], critic_step),
                            tool_calls=(item["tool_step"],),
                        ),
                    )
                    continue
                allowed = {tool["tool_id"] for tool in item["recovery_tools"]}
                if next_tool not in allowed:
                    outcomes[item["index"]] = SamplePredictionOutcome(
                        sample_id=request.sample_id,
                        error=SampleExecutionAttemptError(
                            "remote critic selected an unavailable recovery tool",
                            failure_class="invalid_failure_review",
                            retryable=False,
                            error_type="SampleExecutionContractError",
                            agent_decisions=(*item["agent_steps"], critic_step),
                            tool_calls=(item["tool_step"],),
                        ),
                    )
                    continue
                critic_step["decision"] = f"request_repair_tool:{next_tool}"
                outcomes[item["index"]] = SamplePredictionOutcome(
                    sample_id=request.sample_id,
                    error=SampleExecutionAttemptError(
                        "remote critic requested recovery after tool failure",
                        failure_class=str(item["failure_class"]),
                        retryable=True,
                        error_type=str(item["error_type"]),
                        agent_decisions=(*item["agent_steps"], critic_step),
                        tool_calls=(item["tool_step"],),
                        requested_tool_id=next_tool,
                    ),
                )

    def _recovery_tool_catalog(
        self,
        request: SamplePredictionRequest,
        plan: Mapping[str, Any],
        *,
        failed_tool_id: str,
    ) -> list[dict[str, str]]:
        failed_tool_ids = {failed_tool_id}
        for feedback in _retry_feedback(plan):
            raw_tool_ids = feedback.get("tool_ids")
            if not isinstance(raw_tool_ids, (list, tuple)):
                continue
            failed_tool_ids.update(
                str(tool_id).strip()[:160]
                for tool_id in raw_tool_ids
                if isinstance(tool_id, str) and tool_id.strip()
            )
        return [
            tool
            for tool in self._tool_catalog(request)
            if tool["tool_id"] not in failed_tool_ids
        ]

    def _available_tool_catalog(
        self,
        request: SamplePredictionRequest,
        plan: Mapping[str, Any],
        *,
        role: str,
    ) -> list[dict[str, str]]:
        catalog = self._tool_catalog(request)
        if role != "repair":
            return catalog
        tried_tool_ids = {
            str(tool_id)
            for feedback in _retry_feedback(plan)
            for tool_id in (
                feedback.get("tool_ids")
                if isinstance(feedback.get("tool_ids"), (list, tuple))
                else ()
            )
        }
        return [
            tool for tool in catalog if tool["tool_id"] not in tried_tool_ids
        ]

    def _review_successes(
        self,
        successful: Sequence[Mapping[str, Any]],
        plans: Sequence[Mapping[str, Any]],
        outcomes: list[SamplePredictionOutcome | None],
        diagnostics: _ChunkRoutingDiagnostics,
        *,
        compact: bool = False,
    ) -> None:
        assert self.review_model_id is not None
        for chunk in _chunks(list(range(len(successful))), self.microbatch_size):
            selected = [successful[index] for index in chunk]
            recovery_by_sample = {
                item["request"].sample_id: self._recovery_tool_catalog(
                    item["request"],
                    plans[item["index"]],
                    failed_tool_id=str(item["tool_step"]["tool_id"]),
                )
                for item in selected
            }
            review_samples = (
                [
                    _compact_critic_success_sample(
                        item,
                        recovery_by_sample[item["request"].sample_id],
                    )
                    for item in selected
                ]
                if compact
                else [
                    {
                        "sample_id": item["request"].sample_id,
                        "sample": item["request"].to_dict(),
                        "predicted": item["predicted"],
                        "selected_tool": item["tool_step"],
                    }
                    for item in selected
                ]
            )
            available_tools = [
                {
                    "tool_id": _ACCEPT_TOOL_ID,
                    "version": "1",
                    "purpose": "accept_prediction_for_host_constraint_check",
                },
                *_tool_union(
                    recovery_by_sample[item["request"].sample_id]
                    for item in selected
                ),
            ]
            try:
                raw_response = self._gateway_decide(
                    self.review_model_id,
                    role="critic",
                    samples=review_samples,
                    context=self._gateway_context(
                        plans[selected[0]["index"]], role="critic", compact=compact
                    ),
                    available_tools=available_tools,
                    allow_format_retry=True,
                    diagnostics=diagnostics,
                )
                decisions = _validated_decisions(
                    raw_response,
                    [item["request"].sample_id for item in selected],
                    model_id=self.review_model_id,
                    role="critic",
                )
            except SampleExecutionControlError:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate optional remote review
                if isinstance(exc, GatewayResponseError) and exc.retryable:
                    raise
                failure_class, retryable, error_type = classify_sample_failure(exc)
                split_eligible = _is_adaptive_split_failure(exc)
                if split_eligible:
                    failure_class = "invalid_output"
                    retryable = _is_sample_repair_eligible_failure(exc)
                authentication_failed = (
                    isinstance(exc, GatewayResponseError)
                    and exc.status_code in {401, 403}
                )
                for item, review_sample in zip(selected, review_samples):
                    request = item["request"]
                    critic_step = {
                        "role": "remote_critic_agent",
                        "decision": (
                            "terminate_after_critic_authentication_failure"
                            if authentication_failed
                            else "reject_advisory_keep_scientific_prediction"
                        ),
                        "status": "failed",
                        "model_id": self.review_model_id,
                        "reason_code": f"critic_unavailable_{failure_class}"[:160],
                        "response_digest": digest(
                            {
                                "model_id": self.review_model_id,
                                "role": "critic",
                                "status": "failed",
                                "failure_class": failure_class,
                                "sample": review_sample,
                            }
                        ),
                    }
                    if authentication_failed:
                        outcomes[item["index"]] = SamplePredictionOutcome(
                            sample_id=request.sample_id,
                            error=SampleExecutionAttemptError(
                                "remote critic authentication failed",
                                failure_class=failure_class,
                                retryable=False,
                                error_type=error_type,
                                agent_decisions=(*item["agent_steps"], critic_step),
                                tool_calls=(item["tool_step"],),
                                previous_prediction=item["predicted"],
                            ),
                        )
                    else:
                        outcomes[item["index"]] = SamplePredictionOutcome(
                            sample_id=request.sample_id,
                            result={
                                "predicted": item["predicted"],
                                "agent_decisions": [
                                    *item["agent_steps"],
                                    critic_step,
                                ],
                                "tool_calls": [item["tool_step"]],
                            },
                        )
                continue

            for item in selected:
                request = item["request"]
                decision = decisions[request.sample_id]
                critic_step = _agent_step(
                    decision,
                    role="critic",
                    model_id=self.review_model_id,
                )
                agent_steps = [*item["agent_steps"], critic_step]
                if decision["next_tool"] != _ACCEPT_TOOL_ID:
                    allowed = {
                        tool["tool_id"]
                        for tool in recovery_by_sample[request.sample_id]
                    }
                    if decision["next_tool"] not in allowed:
                        critic_step["status"] = "failed"
                        critic_step["reason_code"] = (
                            "critic_requested_unavailable_advisory_tool"
                        )
                    critic_step["decision"] = (
                        "reject_advisory_keep_scientific_prediction:"
                        f"{decision['next_tool']}"
                    )
                    host_range_rejected = not (
                        request.minimum
                        <= float(item["predicted"])
                        <= request.maximum
                    )
                    if decision["next_tool"] in allowed and host_range_rejected:
                        critic_step["decision"] = (
                            f"request_repair_tool:{decision['next_tool']}"
                        )
                        outcomes[item["index"]] = SamplePredictionOutcome(
                            sample_id=request.sample_id,
                            error=SampleExecutionAttemptError(
                                "remote critic requested registered repair for "
                                "a Host-rejected physical-range prediction",
                                failure_class="constraint_rejected",
                                retryable=True,
                                error_type="SampleRepairRequired",
                                agent_decisions=agent_steps,
                                tool_calls=(item["tool_step"],),
                                requested_tool_id=decision["next_tool"],
                                previous_prediction=item["predicted"],
                            ),
                        )
                        continue
                    # A successful registered predictor is scientific evidence.
                    # The remote critic may report policy-quality evidence, but
                    # it cannot replace that prediction with a fallback.  The
                    # Host physical-range critic still runs after this boundary
                    # and remains authoritative for unsafe values.
                    outcomes[item["index"]] = SamplePredictionOutcome(
                        sample_id=request.sample_id,
                        result={
                            "predicted": item["predicted"],
                            "agent_decisions": agent_steps,
                            "tool_calls": [item["tool_step"]],
                        },
                    )
                    continue
                outcomes[item["index"]] = SamplePredictionOutcome(
                    sample_id=request.sample_id,
                    result={
                        "predicted": item["predicted"],
                        "agent_decisions": agent_steps,
                        "tool_calls": [item["tool_step"]],
                    },
                )

    def _routing_sample(
        self,
        request: SamplePredictionRequest,
        plan: Mapping[str, Any],
        attempt: int,
    ) -> dict[str, Any]:
        return _safe_mapping(
            {
                "sample_id": request.sample_id,
                "sample": request.to_dict(),
                "attempt": attempt,
                "failure_feedback": _retry_feedback(plan),
            },
            "sample routing payload",
        )
    def _gateway_context(
        self,
        plan: Mapping[str, Any],
        *,
        role: str,
        compact: bool = False,
        shared_sample_contexts: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if compact:
            return {
                "role": role,
                "critic_prompt_profile": "uncertain_or_failure_compact@1",
                "decision_policy": (
                    "select_accept_or_one_allowed_recovery_tool_using_only_"
                    "the_supplied_trigger_and_bounded_prediction_context"
                ),
            }
        if (
            self.sample_planner_prompt_profile is not None
            and role in {"planner", "repair"}
        ):
            if not shared_sample_contexts:
                raise SampleExecutionContractError(
                    "origin-shared prompt requires shared sample contexts"
                )
            decision_context = plan.get("decision_context", {})
            if not isinstance(decision_context, Mapping):
                decision_context = {}
            evolution_context = {
                name: decision_context[name]
                for name in (
                    "algorithm_id",
                    "algorithm_version",
                    "evaluator_id",
                    "horizons_hours",
                    "candidate_parameters",
                    "derived_execution_plan",
                    "tool_experience",
                )
                if name in decision_context
            }
            return _safe_mapping(
                {
                    "role": role,
                    "sample_planner_prompt_profile": dict(
                        self.sample_planner_prompt_profile
                    ),
                    "batch_plan_digest": digest(plan),
                    "evolution_context": evolution_context,
                    "shared_sample_contexts": {
                        str(key): dict(value)
                        for key, value in shared_sample_contexts.items()
                    },
                    "context_resolution": (
                        "resolve_each_sample_context_ref_then_sample_id;merge_"
                        "sample_defaults_with_sample_variant_and_merge_label_"
                        "free_context_defaults_with_its_variant"
                    ),
                    "routing_policy": plan.get("routing_policy"),
                },
                "sample gateway shared context",
            )
        return _safe_mapping(
            {
                "role": role,
                "batch_plan_digest": digest(plan),
                "decision_context": plan.get("decision_context", {}),
                "routing_policy": plan.get("routing_policy"),
            },
            "sample gateway context",
        )

    def _tool_catalog(
        self, request: SamplePredictionRequest
    ) -> list[dict[str, str]]:
        return [
            {
                "tool_id": request.algorithm_id,
                "version": request.algorithm_version,
                "purpose": "registered_candidate_prediction",
            },
            {
                "tool_id": _PROJECTION_TOOL_ID,
                "version": "1",
                "purpose": "execute_candidate_then_bounded_projection",
            },
            {
                "tool_id": _PERSISTENCE_TOOL_ID,
                "version": "1",
                "purpose": "bounded_persistence_prediction_or_repair",
            },
            *(tool.descriptor() for tool in self._tools.values()),
        ]

    def _invoke_tool(
        self,
        request: SamplePredictionRequest,
        tool_id: str,
        *,
        attempt: int,
        plan: Mapping[str, Any],
    ) -> tuple[float, str]:
        if tool_id == request.algorithm_id:
            raw_output: Any = self._forecast_tool(request)
        elif tool_id == _PROJECTION_TOOL_ID:
            previous_prediction = _latest_previous_prediction(plan)
            if previous_prediction is None:
                source_prediction, source_output = _normalized_tool_output(
                    self._forecast_tool(request),
                    "registered candidate forecast tool output",
                )
            else:
                source_prediction = previous_prediction
                source_output = {
                    "predicted": previous_prediction,
                    "source": "previous_tool_output",
                }
            raw_output = {
                "predicted": min(
                    request.maximum,
                    max(request.minimum, source_prediction),
                ),
                "metadata": {
                    "operation": "bounded_projection",
                    "source_tool_id": request.algorithm_id,
                    "source_output_digest": digest(source_output),
                },
            }
        elif tool_id == _PERSISTENCE_TOOL_ID:
            raw_output = min(
                request.maximum,
                max(request.minimum, request.baseline),
            )
        else:
            try:
                tool = self._tools[tool_id]
            except KeyError as exc:
                raise SampleExecutionContractError(
                    f"unregistered sample tool: {tool_id}"
                ) from exc
            raw_output = tool.handler(
                request,
                {
                    "attempt": attempt,
                    "batch_plan_digest": digest(plan),
                },
            )
        predicted, public_output = _normalized_tool_output(
            raw_output, "sample tool output"
        )
        return predicted, digest(public_output)


def _retry_feedback(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_feedback = plan.get("sample_retry_feedback", [])
    return (
        [dict(item) for item in raw_feedback[-8:] if isinstance(item, Mapping)]
        if isinstance(raw_feedback, (list, tuple))
        else []
    )


def _requested_repair_tool(plan: Mapping[str, Any]) -> str | None:
    """Return the latest critic-selected repair tool from bounded feedback."""

    raw_feedback = plan.get("sample_retry_feedback")
    if not isinstance(raw_feedback, (list, tuple)):
        return None
    for item in reversed(raw_feedback):
        if not isinstance(item, Mapping) or "requested_tool_id" not in item:
            continue
        tool_id = item.get("requested_tool_id")
        if not isinstance(tool_id, str) or not tool_id.strip():
            raise SampleExecutionContractError(
                "sample retry feedback requested_tool_id must be non-empty text"
            )
        return tool_id.strip()[:160]
    return None


def _latest_previous_prediction(plan: Mapping[str, Any]) -> float | None:
    for item in reversed(_retry_feedback(plan)):
        if "previous_prediction" not in item:
            continue
        return _finite_prediction(
            item.get("previous_prediction"), "sample retry previous_prediction"
        )
    return None


def _validated_decisions(
    value: Any,
    expected_sample_ids: Sequence[str],
    *,
    model_id: str,
    role: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"decisions"}:
        raise _GatewayDecisionContractError(
            "sample gateway response must contain only decisions",
            repair_eligible=False,
        )
    raw_decisions = value.get("decisions")
    if not isinstance(raw_decisions, (list, tuple)):
        raise _GatewayDecisionContractError(
            "sample gateway decisions must be an array",
            repair_eligible=False,
        )
    if len(raw_decisions) != len(expected_sample_ids):
        raise _GatewayDecisionContractError(
            "sample gateway must return one decision per sample",
            repair_eligible=False,
        )
    expected = set(expected_sample_ids)
    decisions: dict[str, dict[str, Any]] = {}
    allowed = {"sample_id", "next_tool", "reason_code", "confidence"}
    for raw in raw_decisions:
        if not isinstance(raw, Mapping) or set(raw) != allowed:
            raise _GatewayDecisionContractError(
                "sample gateway decision fields do not match the contract",
                repair_eligible=True,
            )
        try:
            sample_id = _text(raw.get("sample_id"), "decision sample_id")
        except SampleExecutionContractError as exc:
            raise _GatewayDecisionContractError(
                "sample gateway decision sample_id is invalid",
                repair_eligible=True,
            ) from exc
        if sample_id not in expected or sample_id in decisions:
            raise _GatewayDecisionContractError(
                "sample gateway returned an unknown or duplicate sample_id",
                repair_eligible=True,
            )
        try:
            next_tool = _text(raw.get("next_tool"), "decision next_tool")
        except SampleExecutionContractError as exc:
            raise _GatewayDecisionContractError(
                "sample gateway decision next_tool is invalid",
                repair_eligible=True,
            ) from exc
        reason_code = safe_remote_reason_code(raw.get("reason_code"))
        confidence = raw.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            raise _GatewayDecisionContractError(
                "sample gateway decision confidence must be in [0, 1]",
                repair_eligible=True,
            )
        decision = {
            "sample_id": sample_id,
            "next_tool": next_tool,
            "reason_code": reason_code,
            "confidence": float(confidence),
        }
        decision["response_digest"] = digest(
            {"model_id": model_id, "role": role, "decision": decision}
        )
        decisions[sample_id] = decision
    return decisions


def _agent_step(
    decision: Mapping[str, Any],
    *,
    role: str,
    model_id: str,
) -> dict[str, Any]:
    return {
        "role": f"remote_{role}_agent",
        "decision": f"select_registered_tool:{decision['next_tool']}",
        "status": "completed",
        "model_id": model_id,
        "reason_code": decision["reason_code"],
        "confidence": decision["confidence"],
        "response_digest": decision["response_digest"],
    }


def _coverage_stop_policy(value: Any) -> dict[str, Any] | None:
    """Project the host policy needed for fixed-cohort remote-work stopping."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SampleExecutionContractError(
            "sample execution policy must be an object"
        )
    projected: dict[str, Any] = {
        "mode": "fixed_cohort_maximum_reachable",
    }
    for name in ("minimum_coverage", "minimum_task_coverage"):
        raw = value.get(name)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or not 0 < float(raw) <= 1
        ):
            raise SampleExecutionContractError(
                f"sample execution policy {name} must be in (0, 1]"
            )
        projected[name] = float(raw)
    return projected


def _prediction_coverage_stop_policy(
    plans: Sequence[Mapping[str, Any]],
    attempts: Sequence[int],
) -> dict[str, Any] | None:
    # A retry wave contains only a failed subset, so its request list is not a
    # fixed-cohort denominator. Coverage stopping is decided on the full first
    # wave and then propagated to the executor as a terminal outcome.
    if not attempts or any(attempt != 1 for attempt in attempts):
        return None
    policies = [
        _coverage_stop_policy(plan.get("coverage_stop_policy")) for plan in plans
    ]
    if not policies or policies[0] is None:
        return None
    if any(policy != policies[0] for policy in policies[1:]):
        raise SampleExecutionContractError(
            "sample gateway plans disagree on the coverage stop policy"
        )
    return policies[0]


def _legacy_published_outcome_statuses(
    requests: Sequence[SamplePredictionRequest],
    outcomes: Sequence[SamplePredictionOutcome | None],
    indices: Sequence[int],
) -> dict[str, str]:
    """Preserve observer-only callback semantics for non-durable callers."""

    return {
        requests[index].sample_id: (
            "succeeded"
            if outcomes[index] is not None
            and outcomes[index].result is not None
            and outcomes[index].error is None
            else "failed"
        )
        for index in indices
    }


def _validated_published_outcome_statuses(
    value: Any,
    *,
    requests: Sequence[SamplePredictionRequest],
    indices: Sequence[int],
) -> dict[str, str]:
    """Validate the durable subset acknowledged by the host callback."""

    if not isinstance(value, Mapping):
        raise SampleExecutionContractError(
            "sample outcome callback acknowledgement must be an object"
        )
    allowed = {requests[index].sample_id for index in indices}
    result: dict[str, str] = {}
    for sample_id, raw_status in value.items():
        if not isinstance(sample_id, str) or sample_id not in allowed:
            raise SampleExecutionContractError(
                "sample outcome callback acknowledged an unknown sample"
            )
        status = str(raw_status).strip().casefold()
        if status not in {"succeeded", "failed"}:
            raise SampleExecutionContractError(
                "sample outcome callback acknowledged an unsupported status"
            )
        result[sample_id] = status
    return result


def _coverage_stop_decision(
    requests: Sequence[SamplePredictionRequest],
    outcomes: Sequence[SamplePredictionOutcome | None],
    policy: Mapping[str, Any],
    *,
    resume_checkpoint: Mapping[str, Any] | None = None,
    assumed_permanent_failures: Iterable[int] = (),
) -> dict[str, Any] | None:
    """Return a stop decision only when even perfect remaining work cannot pass."""

    permanent_failures = set(assumed_permanent_failures)
    if any(not 0 <= index < len(requests) for index in permanent_failures):
        raise ValueError("assumed permanent failure index is out of range")
    for index, outcome in enumerate(outcomes):
        if outcome is None or outcome.error is None:
            continue
        _failure_class, retryable, _error_type = classify_sample_failure(outcome.error)
        if not retryable:
            permanent_failures.add(index)

    task_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, request in enumerate(requests):
        task_indices[(request.target, request.horizon_hours)].append(index)

    resumed_task_totals: dict[tuple[str, int], int] = {}
    resumed_task_failures: dict[tuple[str, int], int] = {}
    if resume_checkpoint is not None:
        for task in resume_checkpoint.get("tasks", ()):
            key = (str(task["target"]), int(task["horizon_hours"]))
            resumed_task_totals[key] = int(task["total_samples"])
            resumed_task_failures[key] = int(task["resumed_failed_samples"])

    minimum_task_coverage = float(policy["minimum_task_coverage"])
    task_keys = set(task_indices) | set(resumed_task_totals)
    for target, horizon in sorted(task_keys):
        indices = task_indices.get((target, horizon), ())
        task_total = resumed_task_totals.get((target, horizon), len(indices))
        task_permanent_failures = resumed_task_failures.get(
            (target, horizon), 0
        ) + sum(
            index in permanent_failures for index in indices
        )
        maximum_reachable = (
            (task_total - task_permanent_failures) / task_total
            if task_total
            else 0.0
        )
        if maximum_reachable < minimum_task_coverage:
            return {
                "scope": "task",
                "target": target,
                "horizon_hours": horizon,
                "maximum_reachable_coverage": maximum_reachable,
                "minimum_coverage": minimum_task_coverage,
            }

    total = (
        int(resume_checkpoint["total_samples"])
        if resume_checkpoint is not None
        else len(requests)
    )
    resumed_failures = (
        int(resume_checkpoint["failed_samples"])
        if resume_checkpoint is not None
        else 0
    )
    maximum_reachable = (
        (total - resumed_failures - len(permanent_failures)) / total
        if total
        else 0.0
    )
    minimum_coverage = float(policy["minimum_coverage"])
    if maximum_reachable < minimum_coverage:
        return {
            "scope": "overall",
            "maximum_reachable_coverage": maximum_reachable,
            "minimum_coverage": minimum_coverage,
        }
    return None


def _normalized_resume_checkpoint(
    checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the aggregate-only bridge supplied by the host executor."""

    if checkpoint is None:
        return None
    if not isinstance(checkpoint, Mapping):
        raise TypeError("sample resume checkpoint must be an object")

    def count(name: str) -> int:
        value = checkpoint.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"sample resume checkpoint {name} must be non-negative")
        return value

    completed = count("completed_samples")
    succeeded = count("succeeded_samples")
    failed = count("failed_samples")
    total = count("total_samples")
    batch_index = count("batch_index")
    batch_count = count("batch_count")
    if succeeded + failed != completed or completed > total:
        raise ValueError("sample resume checkpoint outcome counts are inconsistent")
    if batch_count < batch_index:
        raise ValueError("sample resume checkpoint batch position is inconsistent")

    diagnostics = {
        name: count(name)
        for name in (
            "gateway_request_count",
            "adaptive_split_trigger_count",
            "adaptive_split_count",
            "adaptive_split_max_depth",
            "adaptive_split_recovered_samples",
            "adaptive_split_failed_samples",
        )
    }
    if diagnostics["adaptive_split_max_depth"] > 8:
        raise ValueError("sample resume checkpoint split depth is invalid")
    if diagnostics["adaptive_split_count"] > diagnostics[
        "adaptive_split_trigger_count"
    ]:
        raise ValueError("sample resume checkpoint split counters are inconsistent")
    # A durable batch is an outcome-publication boundary. It is not a physical
    # request counter: one request may produce multiple planner/repair batches.
    if (
        diagnostics["adaptive_split_recovered_samples"]
        + diagnostics["adaptive_split_failed_samples"]
        > completed
    ):
        raise ValueError("sample resume checkpoint split outcomes are inconsistent")

    raw_tasks = checkpoint.get("tasks", ())
    if not isinstance(raw_tasks, (list, tuple)):
        raise TypeError("sample resume checkpoint tasks must be an array")
    tasks: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    task_total_sum = 0
    task_failed_sum = 0
    for raw_task in raw_tasks:
        if not isinstance(raw_task, Mapping):
            raise TypeError("sample resume checkpoint task must be an object")
        target = raw_task.get("target")
        horizon = raw_task.get("horizon_hours")
        if not isinstance(target, str):
            raise ValueError("sample resume checkpoint task target is invalid")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise ValueError("sample resume checkpoint task horizon is invalid")
        task_total = raw_task.get("total_samples")
        task_failed = raw_task.get("resumed_failed_samples")
        if (
            isinstance(task_total, bool)
            or not isinstance(task_total, int)
            or task_total < 1
            or isinstance(task_failed, bool)
            or not isinstance(task_failed, int)
            or not 0 <= task_failed <= task_total
        ):
            raise ValueError("sample resume checkpoint task counts are invalid")
        key = (target.strip(), horizon)
        if key in seen:
            raise ValueError("sample resume checkpoint contains duplicate tasks")
        seen.add(key)
        task_total_sum += task_total
        task_failed_sum += task_failed
        tasks.append(
            {
                "target": key[0],
                "horizon_hours": horizon,
                "total_samples": task_total,
                "resumed_failed_samples": task_failed,
            }
        )
    if task_total_sum != total or task_failed_sum != failed:
        raise ValueError("sample resume checkpoint task totals are inconsistent")

    return {
        "completed_samples": completed,
        "succeeded_samples": succeeded,
        "failed_samples": failed,
        "total_samples": total,
        "batch_index": batch_index,
        "batch_count": batch_count,
        "progress_id": count("progress_id"),
        **diagnostics,
        "tasks": tasks,
    }


def _normalized_operation_max_tokens(value: Mapping[str, int] | None) -> dict[str, int]:
    """Validate per-operation output limits frozen into a new task manifest."""

    if value is None:
        return {}
    if not isinstance(value, Mapping) or set(value) != _SAMPLE_OPERATION_MAX_TOKENS:
        raise ValueError(
            "operation_max_tokens must define sample.planner, sample.repair, and sample.critic"
        )
    normalized: dict[str, int] = {}
    for operation in sorted(_SAMPLE_OPERATION_MAX_TOKENS):
        max_tokens = value[operation]
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 512 <= max_tokens <= 8_192
        ):
            raise ValueError(
                f"operation_max_tokens.{operation} must be an integer between 512 and 8192"
            )
        normalized[operation] = max_tokens
    return normalized


def _normalized_remote_critic_policy(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the frozen policy controlling independent sample review."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("remote_critic_policy must be an object")
    version = value.get("version")
    if version == _ALWAYS_CRITIC_POLICY:
        if set(value) != {"version"}:
            raise ValueError("always remote critic policy must define only version")
        return {"version": _ALWAYS_CRITIC_POLICY}
    if set(value) != {"version", "min_planner_confidence"}:
        raise ValueError(
            "remote_critic_policy must define version and min_planner_confidence"
        )
    if version != _UNCERTAIN_OR_FAILURE_CRITIC_POLICY:
        raise ValueError(
            "remote_critic_policy.version must be always@1 or uncertain_or_failure@1"
        )
    threshold = value.get("min_planner_confidence")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0 <= float(threshold) <= 1
    ):
        raise ValueError(
            "remote_critic_policy.min_planner_confidence must be in [0, 1]"
        )
    return {
        "version": _UNCERTAIN_OR_FAILURE_CRITIC_POLICY,
        "min_planner_confidence": float(threshold),
    }


def _normalized_truncation_retry_policy(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the opt-in same-wave output escalation policy.

    This policy is deliberately separate from the gateway's transport retry
    budget.  It is only for a provider response that explicitly reports
    ``finish_reason=length``/``max_tokens`` and therefore cannot contain a
    complete decision contract.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("sample_truncation_retry_policy must be an object")
    if set(value) != {"version", "max_tokens"}:
        raise ValueError(
            "sample_truncation_retry_policy must define version and max_tokens"
        )
    if value.get("version") != _TRUNCATION_RETRY_POLICY:
        raise ValueError(
            "sample_truncation_retry_policy.version must be "
            + _TRUNCATION_RETRY_POLICY
        )
    max_tokens = value.get("max_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 512 <= max_tokens <= _MAX_SAMPLE_OUTPUT_TOKENS
    ):
        raise ValueError(
            "sample_truncation_retry_policy.max_tokens must be an integer "
            "between 512 and 8192"
        )
    return {
        "version": _TRUNCATION_RETRY_POLICY,
        "max_tokens": max_tokens,
    }


def _compact_critic_context(request: SamplePredictionRequest) -> dict[str, Any]:
    """Expose only bounded numeric credibility context to a sparse critic."""

    return {
        "sample_id": request.sample_id,
        "target": request.target,
        "unit": request.unit,
        "horizon_hours": request.horizon_hours,
        "baseline": request.baseline,
        "minimum": request.minimum,
        "maximum": request.maximum,
    }


def _compact_critic_success_sample(
    item: Mapping[str, Any],
    recovery_tools: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    request = item["request"]
    if not isinstance(request, SamplePredictionRequest):
        raise TypeError("compact critic success requires a sample request")
    decision = item["decision"]
    tool_step = item["tool_step"]
    return {
        **_compact_critic_context(request),
        "predicted": float(item["predicted"]),
        "selected_tool_id": str(tool_step["tool_id"]),
        "planner_confidence": float(decision["confidence"]),
        "trigger": "planner_low_confidence",
        "allowed_next_tool_ids": [
            _ACCEPT_TOOL_ID,
            *(str(tool["tool_id"]) for tool in recovery_tools),
        ],
    }


def _compact_critic_failure_sample(item: Mapping[str, Any]) -> dict[str, Any]:
    request = item["request"]
    if not isinstance(request, SamplePredictionRequest):
        raise TypeError("compact critic failure requires a sample request")
    tool_step = item["tool_step"]
    recovery_tools = item["recovery_tools"]
    return {
        **_compact_critic_context(request),
        "attempt": int(item["attempt"]),
        "trigger": "tool_failure",
        "tool_failure": {
            "failure_class": str(item["failure_class"]),
            "host_retryable": bool(item["host_retryable"]),
            "error_type": str(item["error_type"]),
            "failed_tool_id": str(tool_step["tool_id"]),
        },
        "allowed_next_tool_ids": [
            _TERMINATE_TOOL_ID,
            *(str(tool["tool_id"]) for tool in recovery_tools),
        ],
    }


def _token_budget_count(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 1_000_000_000_000
    ):
        raise ValueError(f"{name} must be a bounded non-negative integer")
    return value


def _mark_coverage_terminal_outcomes(
    requests: Sequence[SamplePredictionRequest],
    outcomes: list[SamplePredictionOutcome | None],
    decision: Mapping[str, Any],
) -> None:
    reason_code = (
        "fixed_task_maximum_coverage_unreachable"
        if decision.get("scope") == "task"
        else "overall_maximum_coverage_unreachable"
    )
    for index, request in enumerate(requests):
        outcome = outcomes[index]
        if outcome is None:
            outcome = SamplePredictionOutcome(
                sample_id=request.sample_id,
                error=SampleExecutionAttemptError(
                    "sample was not sent because fixed-cohort coverage is unreachable",
                    failure_class=COVERAGE_UNREACHABLE_NOT_EXECUTED_FAILURE,
                    retryable=False,
                    error_type="CoverageUnreachableEarlyStop",
                    attempted=False,
                    agent_decisions=(
                        {
                            "role": "host_coverage_adjudicator",
                            "decision": "stop_unexecuted_sample",
                            "status": "completed",
                            "reason_code": reason_code,
                        },
                    ),
                ),
            )
        outcomes[index] = SamplePredictionOutcome(
            sample_id=outcome.sample_id,
            result=outcome.result,
            error=outcome.error,
            terminal_reason=COVERAGE_UNREACHABLE_TERMINAL_REASON,
        )


def _is_adaptive_split_failure(exc: BaseException) -> bool:
    """Return whether less response context can plausibly repair the failure.

    The model gateway already exhausts bounded retries for transport failures.
    Only decoded response-format/contract failures are replayed with smaller
    batches. Definite HTTP rejection and local configuration/routing errors are
    intentionally excluded.
    """

    if isinstance(exc, _GatewayDecisionContractError):
        return True
    if not isinstance(exc, GatewayResponseError):
        return False
    explicit = getattr(exc, "split_eligible", None)
    if isinstance(explicit, bool):
        return explicit
    if exc.retryable or exc.status_code is not None:
        return False
    return str(exc).strip() in _LEGACY_SPLIT_ELIGIBLE_GATEWAY_ERRORS


def _is_sample_repair_eligible_failure(exc: BaseException) -> bool:
    """Return whether one decoded decision can be corrected by a repair agent."""

    if isinstance(exc, _GatewayDecisionContractError):
        return exc.repair_eligible
    return (
        isinstance(exc, GatewayResponseError)
        and exc.error_code in _SAMPLE_REPAIR_ELIGIBLE_GATEWAY_ERROR_CODES
    )


def _capture_model_usage_receipt(
    diagnostics: _ChunkRoutingDiagnostics,
    source: Any,
    *,
    call_id: str,
    logical_call_digest: str,
    role: str,
    model_id: str,
    outcome: str,
) -> None:
    """Retain and enqueue one physical call receipt from a worker."""

    http_attempts = _gateway_http_attempts(source)
    diagnostics.gateway_request_count += http_attempts
    if http_attempts == 0:
        return

    if isinstance(source, Mapping):
        raw_usage: Any = source
    else:
        raw_usage = getattr(source, "token_usage", None)
        if not isinstance(raw_usage, Mapping):
            raw_usage = getattr(source, "usage", None)
    usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
    token_counts: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(name)
        token_counts[name] = (
            value
            if isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= 1_000_000_000_000
            else 0
        )
    reported = usage.get("usage_reported")
    usage_reported = (
        reported
        if isinstance(reported, bool)
        else all(name in usage for name in token_counts)
    )
    receipt = {
        "call_id": call_id,
        "logical_call_digest": logical_call_digest,
        "role": role,
        "model_id": model_id,
        "outcome": outcome,
        "usage_reported": usage_reported,
        "http_attempts": http_attempts,
        **token_counts,
    }
    diagnostics.model_usage_receipts.append(receipt)
    if diagnostics.model_usage_receipt_sink is not None:
        diagnostics.model_usage_receipt_sink(receipt)


def _gateway_http_attempts(value: Any) -> int:
    """Read a bounded call-local receipt or exception attempt count."""

    if isinstance(value, Mapping):
        raw = value.get("http_attempts", 1)
    else:
        raw = getattr(value, "http_attempts", None)
        if raw is None:
            raw = getattr(value, "attempts", 1)
    if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw <= 8:
        return 1
    return raw


def _tool_union(
    catalogs: Sequence[Sequence[Mapping[str, Any]]] | Any,
) -> list[dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for catalog in catalogs:
        for raw in catalog:
            item = dict(raw)
            tool_id = str(item.get("tool_id") or "")
            if tool_id and tool_id not in indexed:
                indexed[tool_id] = item
    return list(indexed.values())


def _chunks(values: Sequence[int], size: int) -> list[list[int]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _causal_wave_identity(
    request: SamplePredictionRequest,
    *,
    index: int,
) -> tuple[Any, ...]:
    """Return a verified origin wave, or a request-local singleton fallback."""

    singleton = ("singleton", index)
    origin = _normalized_timestamp(request.origin_timestamp)
    target = _normalized_timestamp(request.target_timestamp)
    provenance = request.label_free_context.get("causal_provenance")
    if provenance is None:
        return singleton
    if not isinstance(provenance, Mapping):
        raise SampleExecutionContractError(
            "sample causal_provenance must be an object"
        )
    if origin is None or target is None or origin[0] != target[0]:
        return singleton
    if not origin[1] < target[1]:
        raise SampleExecutionContractError(
            "sample causal provenance requires origin_timestamp before target_timestamp"
        )
    if provenance.get("schema_version") != _CAUSAL_PROVENANCE_SCHEMA_VERSION:
        raise SampleExecutionContractError(
            "sample causal_provenance schema is unsupported"
        )
    cutoff = _normalized_timestamp(provenance.get("origin_cutoff_timestamp"))
    latest = _normalized_timestamp(provenance.get("latest_context_timestamp"))
    if cutoff != origin:
        raise SampleExecutionContractError(
            "sample causal provenance does not match origin_timestamp"
        )
    if latest is None or latest[0] != origin[0] or latest[1] > origin[1]:
        raise SampleExecutionContractError(
            "sample context contains or claims information after its origin cutoff"
        )

    raw_history = request.label_free_context.get("history_window")
    raw_history_timestamps = provenance.get("history_timestamps")
    if raw_history is not None:
        if not isinstance(raw_history, (list, tuple)) or not isinstance(
            raw_history_timestamps, (list, tuple)
        ):
            raise SampleExecutionContractError(
                "sample history requires aligned causal timestamps"
            )
        if len(raw_history) != len(raw_history_timestamps):
            raise SampleExecutionContractError(
                "sample history and causal timestamps must have equal lengths"
            )
    elif raw_history_timestamps not in (None, [], ()):
        raise SampleExecutionContractError(
            "sample causal history timestamps require a history_window"
        )
    for raw_timestamp in raw_history_timestamps or ():
        history_timestamp = _normalized_timestamp(raw_timestamp)
        if (
            history_timestamp is None
            or history_timestamp[0] != origin[0]
            or history_timestamp[1] > origin[1]
        ):
            raise SampleExecutionContractError(
                "sample history contains a timestamp after its origin cutoff"
            )
    return ("verified_origin", *origin)


def _normalized_timestamp(value: Any) -> tuple[str, Any] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return ("number", Decimal(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return ("number", Decimal(str(value)))
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return ("datetime", parsed.astimezone(timezone.utc))


def _safe_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SampleExecutionContractError(f"{name} must be an object")
    normalized = _safe_value(value, name, depth=0)
    assert isinstance(normalized, dict)
    if len(canonical_json(normalized).encode("utf-8")) > _MAX_GATEWAY_PAYLOAD_BYTES:
        raise SampleExecutionContractError(f"{name} exceeds the gateway payload bound")
    return normalized


def _safe_value(value: Any, path: str, *, depth: int) -> Any:
    if depth > 10:
        raise SampleExecutionContractError(f"{path} is nested too deeply")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise SampleExecutionContractError(f"{path} keys must be non-empty text")
            key = raw_key.strip()
            normalized_key = key.casefold().replace("-", "_").replace(" ", "_")
            normalized_token = "".join(
                character for character in key.casefold() if character.isalnum()
            )
            if (
                normalized_key in _FORBIDDEN_OUTCOME_KEYS
                or normalized_token in _FORBIDDEN_OUTCOME_TOKENS
            ):
                raise SampleExecutionContractError(
                    f"{path} contains forbidden outcome field {key!r}"
                )
            result[key] = _safe_value(
                raw_item,
                f"{path}.{key}",
                depth=depth + 1,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _safe_value(item, f"{path}[{index}]", depth=depth + 1)
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


def _finite_prediction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SampleExecutionContractError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise SampleExecutionContractError(f"{name} must be finite")
    return number


def _normalized_tool_output(value: Any, name: str) -> tuple[float, dict[str, Any]]:
    if isinstance(value, Mapping):
        safe_output = _safe_mapping(value, name)
        if set(safe_output) - {"predicted", "metadata"}:
            raise SampleExecutionContractError(
                "sample tool output contains unsupported fields"
            )
        predicted = _finite_prediction(
            safe_output.get("predicted"), f"{name} predicted"
        )
        return predicted, safe_output
    predicted = _finite_prediction(value, f"{name} predicted")
    return predicted, {"predicted": predicted}


def _legacy_candidate_forecast(request: SamplePredictionRequest) -> float:
    if request.proposed_prediction is None:
        raise SampleExecutionContractError(
            "gateway sample requires an explicit registered forecast tool"
        )
    return request.proposed_prediction


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SampleExecutionContractError(f"{name} must be non-empty text")
    return value.strip()[:200]


__all__ = [
    "GatewaySampleCollaborationAdapter",
    "GatewaySampleTool",
    "SampleDecisionGateway",
]
