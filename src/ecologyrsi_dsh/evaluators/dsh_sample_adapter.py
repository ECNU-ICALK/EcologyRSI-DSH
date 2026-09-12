"""Agent-owned numerical inference with optional Host model capabilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from threading import RLock
from typing import Any

from ..core.models import digest
from .sample_contracts import _causal_wave_identity, _safe_mapping, _normalized_operation_max_tokens, _normalized_remote_critic_policy
from .shared_sample_context import normalized_sample_planner_prompt_profile
from .sample_execution import SampleExecutionPausedError, SampleExecutionCancelledError, SampleExecutionControlUnavailableError, classify_sample_failure
from ..core.errors import dsh_native_runtime_error_in_chain, dsh_native_runtime_retryable
from ..core.model_execution_policy import (
    NATIVE_SAMPLE_OPERATION_MAX_TOKENS,
    SAMPLE_OPERATION_MAX_MAX_TOKENS,
    SAMPLE_OPERATION_MIN_MAX_TOKENS,
)
from ..core.errors import dsh_native_runtime_evaluation_fatal
from ..core.agent_prediction import PREDICTION_TOOL_CALL_BUDGET, validate_predictions, validate_agent_review
from .origin_prompt import compact_origin_contexts
from ..core.redaction import REMOTE_REASON_CODES
from ..integrations.dsh_structured_roles import DshStructuredRoleRuntime
from .sample_execution import (
    SampleExecutionContractError,
    SampleExecutionAttemptError,
    SampleExecutionControlError,
    SamplePredictionOutcome,
    forecast_origin_sample_id,
)


def _sample_routing_wave(
    model_id: str,
    *,
    role: str,
    samples: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    available_tools: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "ecologyrsi-dsh.sample-routing-wave/1",
        "role": role,
        "model_route_id": model_id,
        "samples": [dict(item) for item in samples],
        "context": dict(context),
        "available_tools": [dict(item) for item in available_tools],
        "allowed_reason_codes": sorted(REMOTE_REASON_CODES),
    }


class _DshSampleDecisionClient:
    supports_operation_max_tokens = True

    def __init__(
        self,
        *,
        run_id: str,
        runtime_provider: Callable[[], Any],
        revision_provider: Callable[[str], Mapping[str, int]],
        identity_digests: Mapping[str, str],
    ) -> None:
        self.run_id = run_id
        self.runtime_provider = runtime_provider
        self.revision_provider = revision_provider
        self.identity_digests = dict(identity_digests)

    def sample_decide(
        self,
        model_id: str,
        *,
        role: str,
        samples: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        available_tools: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> Mapping[str, Any]:
        if role not in {"planner", "repair", "critic"}:
            raise SampleExecutionContractError("unsupported DSH sample role")
        dsh_role = "sample-critic" if role == "critic" else "sample-planner"
        stage = "sample.critic" if role == "critic" else "sample.plan"
        schema_id = (
            "ecology-sample-review@2"
            if role == "critic"
            else "ecology-sample-predictions@2"
        )
        max_tokens = options.get("max_tokens")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not SAMPLE_OPERATION_MIN_MAX_TOKENS
            <= max_tokens
            <= SAMPLE_OPERATION_MAX_MAX_TOKENS
        ):
            raise SampleExecutionContractError(
                "DSH sample stage requires a bounded max_tokens value"
            )
        wave = _sample_routing_wave(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )
        wave_digest = digest(wave)
        revisions = dict(self.revision_provider(self.run_id))
        runtime = self.runtime_provider()
        if not isinstance(runtime, DshStructuredRoleRuntime):
            runtime = DshStructuredRoleRuntime(runtime)
        structured = runtime.run(
            run_id=self.run_id,
            stage=stage,
            role=dsh_role,
            context={**wave, "wave_digest": wave_digest},
            output_schema_id=schema_id,
            run_state_revision=int(revisions["run_state_revision"]),
            stage_attempt=max(1, int(wave_digest[:12], 16)),
            ledger_expected_revision=int(revisions["ledger_expected_revision"]),
            idempotency_key=f"{self.run_id}:{stage}:{wave_digest}",
            identity_digests=self.identity_digests,
            max_tokens=max_tokens,
        )
        expected_version = schema_id
        if structured.get("schema_version") != expected_version:
            raise SampleExecutionContractError("DSH sample result schema version mismatch")
        if structured.get("wave_digest") != wave_digest:
            raise SampleExecutionContractError("DSH sample result wave digest mismatch")
        decisions = structured.get("decisions")
        if not isinstance(decisions, list):
            raise SampleExecutionContractError("DSH sample result requires decisions")
        return {"decisions": decisions}

    def sample_reflect(
        self,
        model_id: str,
        *,
        sample: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Run one post-score reflection in its own auditable DSH child."""

        sample_id = str(sample.get("sample_id") or "").strip()
        if not sample_id:
            raise SampleExecutionContractError(
                "DSH sample reflection requires sample_id"
            )
        wave = {
            "schema_version": "ecologyrsi-dsh.sample-reflection-context/1",
            "role": "reflector",
            "model_route_id": model_id,
            "sample": dict(sample),
            "outcome": dict(outcome),
        }
        wave_digest = digest(wave)
        revisions = dict(self.revision_provider(self.run_id))
        runtime = self.runtime_provider()
        if not isinstance(runtime, DshStructuredRoleRuntime):
            runtime = DshStructuredRoleRuntime(runtime)
        structured = runtime.run(
            run_id=self.run_id,
            stage="sample.reflect",
            role="sample-critic",
            context={**wave, "wave_digest": wave_digest},
            output_schema_id="ecology-sample-reflection@1",
            run_state_revision=int(revisions["run_state_revision"]),
            stage_attempt=max(1, int(wave_digest[:12], 16)),
            ledger_expected_revision=int(revisions["ledger_expected_revision"]),
            idempotency_key=f"{self.run_id}:sample.reflect:{wave_digest}",
            identity_digests=self.identity_digests,
        )
        if structured.get("schema_version") != "ecology-sample-reflection@1":
            raise SampleExecutionContractError(
                "DSH sample reflection schema version mismatch"
            )
        if structured.get("wave_digest") != wave_digest:
            raise SampleExecutionContractError(
                "DSH sample reflection wave digest mismatch"
            )
        if structured.get("sample_id") != sample_id:
            raise SampleExecutionContractError(
                "DSH sample reflection sample_id mismatch"
            )
        outcome_class = structured.get("outcome_class")
        if outcome_class not in {"improved", "degraded", "neutral", "failed"}:
            raise SampleExecutionContractError(
                "DSH sample reflection outcome_class is invalid"
            )
        error_source = structured.get("error_source")
        if error_source not in {
            "model",
            "feature",
            "parameter",
            "tool",
            "execution",
            "unknown",
        }:
            raise SampleExecutionContractError(
                "DSH sample reflection error_source is invalid"
            )
        next_action = structured.get("next_action")
        if next_action not in {
            "keep",
            "increase",
            "decrease",
            "repair",
            "inspect",
            "stop",
        }:
            raise SampleExecutionContractError(
                "DSH sample reflection next_action is invalid"
            )
        confidence = structured.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            raise SampleExecutionContractError(
                "DSH sample reflection confidence must be in [0, 1]"
            )
        summary = structured.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise SampleExecutionContractError(
                "DSH sample reflection summary must be non-empty"
            )
        return {
            "schema_version": "ecology-sample-reflection@1",
            "sample_id": sample_id,
            "outcome_class": outcome_class,
            "error_source": error_source,
            "next_action": next_action,
            "confidence": float(confidence),
            "summary": summary.strip()[:1000],
            "model_id": model_id,
            "response_digest": digest(structured),
            "wave_digest": wave_digest,
        }


class DshSampleCollaborationAdapter:
    """Run the only supported DSH sample protocol: one vector chain per origin."""

    adapter_id = "dsh-native-sample-collaboration"
    adapter_version = "7-native-origin-agent-policy"
    # Rejected predictions return to the Agent under the ordinary retry budget.
    terminal_constraint_repair_is_local = False

    def __init__(
        self,
        *,
        run_id: str,
        runtime_provider: Callable[[], Any],
        revision_provider: Callable[[str], Mapping[str, int]],
        identity_digests: Mapping[str, str],
        strategy_model_id: str,
        review_model_id: str,
        forecast_bundle_tool: Callable[..., Any],
        prediction_tool_binder: Callable[..., Any],
        prediction_tool_catalog: Sequence[Mapping[str, Any]] = (),
        prediction_tool_executor: Callable[..., Any] | None = None,
        microbatch_size: int = 128,
        sample_concurrency: int = 4,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        admission_snapshot_provider: Callable[[], Mapping[str, int]] | None = None,
        run_control_callback: Callable[[], str] | None = None,
        remote_critic_policy: Mapping[str, Any] | None = None,
        sample_reflection_policy: str | None = None,
        sample_planner_prompt_profile: Mapping[str, Any] | None = None,
        operation_max_tokens: Mapping[str, int] | None = None,
    ) -> None:
        if not callable(forecast_bundle_tool):
            raise TypeError("strict DSH execution requires a vector prediction tool")
        if not callable(prediction_tool_binder):
            raise TypeError("strict DSH execution requires an agent prediction-tool binder")
        if admission_snapshot_provider is not None and not callable(
            admission_snapshot_provider
        ):
            raise TypeError("admission_snapshot_provider must be callable")
        self._agent_tool_catalog = [dict(item) for item in prediction_tool_catalog]
        self._agent_tool_executor = prediction_tool_executor
        self._strict_progress_callback = progress_callback
        self._prediction_tool_binder = prediction_tool_binder
        self._strict_progress_state: dict[str, int] | None = None
        self._strict_progress_id = 0
        self._strict_latest_gateway_progress: dict[str, Any] | None = None
        self._strict_progress_lock = RLock()
        self._strict_max_in_flight = sample_concurrency
        self._admission_snapshot_provider = admission_snapshot_provider
        if sample_reflection_policy not in {
            None,
            "always_remote_post_score@1",
            "candidate_aggregate_post_score@1",
            "disabled_for_independent_evaluation@1",
        }:
            raise ValueError("unsupported DSH sample reflection policy")
        # A missing policy belongs to historical manifests and preserves the
        # original Planner/Critic/Reflector chain on replay.
        self.sample_reflection_enabled = sample_reflection_policy in {
            None,
            "always_remote_post_score@1",
        }
        require_success_critic = bool(
            remote_critic_policy is None
            or (
                isinstance(remote_critic_policy, Mapping)
                and remote_critic_policy.get("version") == "always@1"
            )
        )
        client = _DshSampleDecisionClient(
            run_id=run_id,
            runtime_provider=runtime_provider,
            revision_provider=revision_provider,
            identity_digests=identity_digests,
        )
        self.strategy_model_id = strategy_model_id
        self.review_model_id = review_model_id
        if not strategy_model_id or not review_model_id:
            raise ValueError("DSH sample roles require model routes")
        if type(microbatch_size) is not int or not 1 <= microbatch_size <= 128:
            raise ValueError("microbatch_size must be between 1 and 128")
        if type(sample_concurrency) is not int or not 1 <= sample_concurrency <= 128:
            raise ValueError("sample_concurrency must be between 1 and 128")
        self.microbatch_size, self.sample_concurrency = microbatch_size, sample_concurrency
        self._forecast_bundle_tool = forecast_bundle_tool
        self._run_control_callback = run_control_callback
        self.remote_review_enabled = True
        self.require_remote_planner = True
        self.require_remote_critic = require_success_critic
        self.remote_critic_policy = _normalized_remote_critic_policy(remote_critic_policy)
        self.sample_planner_prompt_profile = normalized_sample_planner_prompt_profile(sample_planner_prompt_profile)
        self.operation_max_tokens = _normalized_operation_max_tokens(operation_max_tokens or NATIVE_SAMPLE_OPERATION_MAX_TOKENS)
        self._decision_client = client

    def _check_control(self):
        if self._run_control_callback is None:
            return
        try:
            status = self._run_control_callback()
        except Exception as exc:
            raise SampleExecutionControlUnavailableError("run control unavailable") from exc
        if status == "paused":
            raise SampleExecutionPausedError("sample Agent paused")
        if status == "cancelled":
            raise SampleExecutionCancelledError("sample Agent cancelled")
        if status != "running":
            raise SampleExecutionControlUnavailableError("invalid run control state")

    def _agent_decide(self, model_id, *, role, samples, context, available_tools, **_unused):
        self._check_control()
        result = self._decision_client.sample_decide(model_id, role=role, samples=samples,
            context=context, available_tools=available_tools, max_tokens=self.operation_max_tokens["sample." + role])
        self._check_control()
        return result

    def predict_sample(self, request, plan, *, attempt):
        outcome = self.predict_samples((request,), (plan,), attempts=(attempt,))[0]
        if outcome.error is not None:
            raise outcome.error
        return outcome.result

    def predict_samples(self, requests, plans, *, attempts):
        """Execute complete origin waves; outer executor owns concurrency/checkpoints.

        DSH owns admission, usage settlement and transport retries. This adapter
        never opens a second scheduler or splits an origin into cell Agents.
        """
        if not len(requests) == len(plans) == len(attempts):
            raise ValueError("requests, plans, attempts must have equal lengths")
        if len({r.sample_id for r in requests}) != len(requests):
            raise SampleExecutionContractError("sample IDs must be unique")
        groups = {}
        outcomes = [None] * len(requests)
        for i, (request, plan, attempt) in enumerate(zip(requests, plans, attempts)):
            if type(attempt) is not int or attempt < 1:
                raise ValueError("sample attempts must be positive integers")
            _safe_mapping(request.to_dict(), "sample Agent request")
            _safe_mapping(plan, "sample Agent plan")
            try:
                causal = _causal_wave_identity(request, index=i)
            except SampleExecutionContractError as exc:
                outcomes[i] = SamplePredictionOutcome(sample_id=request.sample_id, error=SampleExecutionAttemptError(
                    str(exc), failure_class="invalid_causal_provenance", retryable=False, error_type=type(exc).__name__))
                continue
            key = (request.candidate_id, request.dataset_digest, request.partition, causal,
                   plan["decision_context_digest"], "planner" if attempt == 1 else "repair")
            groups.setdefault(key, []).append(i)
        for key, indices in groups.items():
            self._check_control()
            if len(indices) > 128:
                raise SampleExecutionContractError("origin exceeds prediction-cell protocol limit")
            self._route_chunk(requests, plans, attempts, indices, role=key[-1], outcomes=outcomes, diagnostics=None)
        return tuple(outcomes)

    def _prediction_tool_context(
        self,
        *,
        role: str,
        model_id: str,
        requests: Sequence[Any],
        samples: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        available_tools: Sequence[Mapping[str, Any]],
    ) -> Any:
        # A transport/model failure before the Host tool executes advances the
        # sample attempt counter, so the generic adapter labels the next route
        # as ``repair``.  There is no critic-selected repair tool in that case:
        # it is still the same strict DSH Planner -> prediction-tool chain and
        # therefore needs the same ephemeral Host binding as the first attempt.
        sample_ids = tuple(str(request.sample_id) for request in requests)
        wave = _sample_routing_wave(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
        )
        wave_digest = digest(wave)
        stage_attempt = max(1, int(wave_digest[:12], 16))
        idempotency_key = f"{self._decision_client.run_id}:sample.plan:{wave_digest}"
        frozen_requests = tuple(requests)
        return self._prediction_tool_binder(
            run_id=self._decision_client.run_id,
            stage_attempt=stage_attempt,
            idempotency_key=idempotency_key,
            wave_digest=wave_digest,
            catalog=available_tools,
            sample_ids=sample_ids,
            executor=lambda tool_id, parameters: self._execute_agent_tool(
                frozen_requests, tool_id, parameters
            ),
        )

    def set_resume_checkpoint(self, checkpoint: Mapping[str, Any] | None) -> None:
        """Keep cumulative progress while strict chains stream one sample at a time."""

        with self._strict_progress_lock:
            # The base adapter's cumulative session is invocation-local and
            # cannot be shared by concurrent origin chains. Strict progress is
            # aggregated below after each complete origin is durable.
            if checkpoint is None:
                self._strict_progress_state = None
                self._strict_progress_id = 0
                self._strict_latest_gateway_progress = None
                return
            self._strict_progress_state = {
                "completed_samples": int(
                    checkpoint.get(
                        "completed_origin_samples", checkpoint["completed_samples"]
                    )
                ),
                "succeeded_samples": int(
                    checkpoint.get(
                        "succeeded_origin_samples", checkpoint["succeeded_samples"]
                    )
                ),
                "total_samples": int(
                    checkpoint.get("total_origin_samples", checkpoint["total_samples"])
                ),
                "batch_index": int(
                    checkpoint.get(
                        "completed_origin_samples", checkpoint["completed_samples"]
                    )
                ),
                "batch_count": max(
                    int(
                        checkpoint.get(
                            "total_origin_samples", checkpoint["total_samples"]
                        )
                    ),
                    int(
                        checkpoint.get(
                            "completed_origin_samples", checkpoint["completed_samples"]
                        )
                    ),
                ),
            }
            self._strict_progress_id = int(checkpoint.get("progress_id", 0))
            self._strict_latest_gateway_progress = None

    def _strict_active_origins(self) -> int:
        """Return run-level admitted origin chains without guessing capacity."""

        if self._admission_snapshot_provider is None:
            return 0
        snapshot = self._admission_snapshot_provider()
        active = snapshot.get("active", 0)
        if isinstance(active, bool) or not isinstance(active, int) or active < 0:
            raise SampleExecutionContractError(
                "strict admission snapshot active count is invalid"
            )
        return min(self._strict_max_in_flight, active)

    def _handle_strict_gateway_progress(self, progress: Mapping[str, Any]) -> None:
        """Expose gateway activity without counting a pre-reflection sample."""

        with self._strict_progress_lock:
            if self._strict_progress_callback is None:
                return
            state = self._strict_progress_state
            if state is None:
                return
            projected = dict(progress)
        # The gateway routes all cells together, but its adaptive split
        # diagnostics are expressed in cell counts. Public progress is in
        # completed forecast origins, so do not mix those units.
            projected["adaptive_split_recovered_samples"] = 0
            projected["adaptive_split_failed_samples"] = 0
            self._strict_latest_gateway_progress = projected
            self._strict_progress_id += 1
            completed = int(state["completed_samples"])
            succeeded = int(state["succeeded_samples"])
            remaining = max(0, int(state["total_samples"]) - completed)
            progress_kind = str(projected.get("progress_kind") or "waiting")
            if progress_kind == "completed_batch":
                progress_kind = "waiting"
            in_flight = min(remaining, self._strict_active_origins())
            projected.update(
                {
                    "progress_id": self._strict_progress_id,
                    "progress_kind": progress_kind,
                    "batch_index": int(state["batch_index"]),
                    "batch_count": int(state["batch_count"]),
                    "batch_size": 0,
                    "completed_samples": completed,
                    "total_samples": int(state["total_samples"]),
                    "succeeded_samples": succeeded,
                    "failed_samples": completed - succeeded,
                    "in_flight_batches": in_flight,
                    # Capacity that has not been submitted is not a provider
                    # queue.  Strict origin admission happens outside this
                    # per-origin gateway heartbeat, so only durable provider
                    # gate/launch evidence may raise `queued_batches`.
                    "queued_batches": 0,
                    "awaiting_submission_batches": max(
                        0,
                        remaining - in_flight,
                    ),
                }
            )
            self._strict_progress_callback(projected)

    def record_finalized_sample_progress(self, *, status: str) -> None:
        """Advance strict progress only after reflection and durable publication."""

        with self._strict_progress_lock:
            if self._strict_progress_callback is None:
                return
            state = self._strict_progress_state
            if state is None:
                return
            if status not in {"succeeded", "failed"}:
                raise SampleExecutionContractError(
                    "strict finalized sample status is invalid"
                )
            state["completed_samples"] += 1
            state["succeeded_samples"] += int(status == "succeeded")
            state["batch_index"] = state["completed_samples"]
            self._strict_progress_id += 1
            completed = int(state["completed_samples"])
            succeeded = int(state["succeeded_samples"])
            remaining = max(0, int(state["total_samples"]) - completed)
            projected = dict(self._strict_latest_gateway_progress or {})
            in_flight = min(
                remaining,
                self._strict_active_origins(),
            )
            projected.update({
                "schema_version": "ecologyrsi-dsh.sample-microbatch-progress/3",
                "role": "planner",
                "model_id": self.strategy_model_id,
                "progress_id": self._strict_progress_id,
                "progress_kind": "completed_batch",
                "batch_index": int(state["batch_index"]),
                "batch_count": int(state["batch_count"]),
                "batch_size": 1,
                "completed_samples": completed,
                "total_samples": int(state["total_samples"]),
                "succeeded_samples": succeeded,
                "failed_samples": completed - succeeded,
                "in_flight_batches": in_flight,
                "queued_batches": 0,
                "awaiting_submission_batches": max(
                    0,
                    remaining - in_flight,
                ),
            })
            self._strict_progress_callback(projected)

    def record_finalized_origin_progress(
        self,
        *,
        status: str,
        prediction_cell_count: int,
    ) -> None:
        """Publish one durable origin after all of its prediction cells land."""

        if (
            isinstance(prediction_cell_count, bool)
            or not isinstance(prediction_cell_count, int)
            or prediction_cell_count < 1
        ):
            raise SampleExecutionContractError(
                "strict origin prediction_cell_count is invalid"
            )
        self.record_finalized_sample_progress(status=status)

    def plan_batch(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        for field, expected in (("strategy_model_id", self.strategy_model_id), ("review_model_id", self.review_model_id)):
            if context.get(field) is not None and context[field] != expected:
                raise SampleExecutionContractError("sample model route does not match frozen task")
        fields = {"run_id", "candidate_id", "dataset_digest", "partition", "algorithm_id", "algorithm_version",
            "evaluator_id", "horizons_hours", "candidate_parameters", "tool_experience", "agent_policy",
            "stage_context_digest", "candidate_genome_digest", "candidate_agent_profile", "derived_execution_plan", "evaluation_scope"}
        decision_context = _safe_mapping({key: context[key] for key in fields if key in context}, "sample policy context")
        plan = {"plan_id": "native-origin-agent@1", "remote_sample_agents": True,
                "strategy_model_id": self.strategy_model_id, "review_model_id": self.review_model_id,
                "sample_concurrency": self.sample_concurrency, "microbatch_size": self.microbatch_size,
                "require_remote_planner": True, "require_remote_critic": self.require_remote_critic,
                "remote_critic_policy": self.remote_critic_policy,
                "decision_context": decision_context, "decision_context_digest": digest(decision_context)}
        require_success_critic = bool(
            self.require_remote_critic
            or self.remote_critic_policy is None
            or self.remote_critic_policy["version"] == "always@1"
        )
        required_remote_roles = ["planner"]
        if require_success_critic:
            required_remote_roles.append("critic")
        if self.sample_reflection_enabled:
            required_remote_roles.append("reflector")
        plan.update(
            {
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "sample_prompt_batch_size": 1,
                "prediction_unit": "forecast_origin_with_target_horizon_vector",
                "execution_mode": "per_origin_agent_owned_prediction",
                "host_route_bypass_allowed": False,
                "host_prediction_fallback_allowed": False,
                "post_score_reflection_required": self.sample_reflection_enabled,
                "required_success_remote_roles": required_remote_roles,
                "remote_roles": required_remote_roles,
                "routing_policy": "sample_agent_analyzes_and_optionally_calls_tools_then_submits_prediction;host_validates_then_scores",
                "forecast_value_source": "agent_final_structured_prediction",
                "tools": self._available_tool_catalog(None, plan, role="planner"),
            }
        )
        return plan

    def reflect_sample(
        self,
        request: Any,
        *,
        observed: float,
        predicted: float | None,
        status: str,
        failure_class: str | None,
        agent_decisions: Sequence[Mapping[str, Any]],
        tool_calls: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        sample = {
            "sample_id": request.sample_id,
            "candidate_id": request.candidate_id,
            "dataset_digest": request.dataset_digest,
            "partition": request.partition,
            "target": request.target,
            "unit": request.unit,
            "horizon_hours": request.horizon_hours,
            "origin_timestamp": request.origin_timestamp,
            "target_timestamp": request.target_timestamp,
        }
        outcome = {
            "status": status,
            "observed": float(observed),
            "predicted": float(predicted) if predicted is not None else None,
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
        return self._decision_client.sample_reflect(
            self.review_model_id or self.strategy_model_id,
            sample=sample,
            outcome=outcome,
        )

    def reflect_origin(
        self,
        requests: Sequence[Any],
        *,
        scored_cells: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Reflect once over every target/horizon prediction at one origin."""

        if not requests or len(requests) != len(scored_cells):
            raise SampleExecutionContractError(
                "origin reflection requires one scored cell per request"
            )
        first = requests[0]
        if any(
            request.candidate_id != first.candidate_id
            or request.dataset_digest != first.dataset_digest
            or request.partition != first.partition
            or request.origin_timestamp != first.origin_timestamp
            for request in requests
        ):
            raise SampleExecutionContractError(
                "origin reflection requests do not share one forecast origin"
            )
        sample_ids = [request.sample_id for request in requests]
        if len(sample_ids) != len(set(sample_ids)):
            raise SampleExecutionContractError(
                "origin reflection sample identifiers must be unique"
            )
        origin_sample_id = forecast_origin_sample_id(requests)
        sample = {
            "sample_id": origin_sample_id,
            "candidate_id": first.candidate_id,
            "dataset_digest": first.dataset_digest,
            "partition": first.partition,
            "origin_timestamp": first.origin_timestamp,
            "prediction_unit": "forecast_origin_with_target_horizon_vector",
            "prediction_cells": [
                {
                    "sample_id": request.sample_id,
                    "target": request.target,
                    "unit": request.unit,
                    "horizon_hours": request.horizon_hours,
                    "target_timestamp": request.target_timestamp,
                }
                for request in requests
            ],
        }
        outcome_cells: list[dict[str, Any]] = []
        for request, raw_cell in zip(requests, scored_cells):
            cell = dict(raw_cell)
            if cell.get("sample_id") != request.sample_id:
                raise SampleExecutionContractError(
                    "origin reflection scored cell identity mismatch"
                )
            outcome_cells.append(cell)
        outcome = {
            "status": (
                "succeeded"
                if all(cell.get("status") == "succeeded" for cell in outcome_cells)
                else "failed"
            ),
            "prediction_cell_count": len(outcome_cells),
            "cells": outcome_cells,
        }
        reflection = dict(
            self._decision_client.sample_reflect(
                self.review_model_id or self.strategy_model_id,
                sample=sample,
                outcome=outcome,
            )
        )
        return {
            **reflection,
            "schema_version": "ecologyrsi-dsh.sample-origin-reflection/1",
            "origin_sample_id": origin_sample_id,
            "cell_sample_ids": sample_ids,
        }

    def _available_tool_catalog(self, request, plan, *, role):
        if role not in {"planner", "repair"}:
            return []
        return [
            {"tool_id": "candidate-model", "version": "1", "purpose": "Candidate's evolved default model; optional", "parameters": {}},
            {"tool_id": "persistence", "version": "1", "purpose": "Latest causal history value", "parameters": {}},
            *self._agent_tool_catalog,
        ]

    def _execute_agent_tool(self, requests, tool_id, parameters):
        if tool_id in {"candidate-model", "persistence"}:
            if parameters:
                raise ValueError("this tool takes no parameters")
            if tool_id == "candidate-model":
                return self._forecast_bundle_tool(requests)
            outputs = {}
            for request in requests:
                history = request.label_free_context.get("history_window", [])
                times = request.label_free_context.get("causal_provenance", {}).get("history_timestamps", [])
                if not history or len(history) != len(times):
                    raise ValueError("persistence requires timestamped history")
                latest = max(range(len(times)), key=times.__getitem__)
                outputs[request.sample_id] = {"predicted": history[latest], "metadata": {"source": "persistence"}}
            return outputs
        if self._agent_tool_executor is None:
            raise ValueError("unknown Agent prediction capability")
        return self._agent_tool_executor(requests, tool_id, parameters)

    def _route_chunk(self, requests, plans, attempts, indices, *, role, outcomes,
                     diagnostics, split_depth=0, split_floor=None):
        selected = [requests[i] for i in indices]
        samples, origin_contexts = [], {}
        for index in indices:
            request = requests[index]
            visible = dict(request.label_free_context)
            visible.pop("predictor_state", None)
            ref = digest(visible)
            origin_contexts[ref] = visible
            sample = request.to_dict()
            sample.pop("label_free_context")
            samples.append({**sample, "context_ref": ref, "attempt": attempts[index],
                            "failure_feedback": plans[index].get("sample_retry_feedback", [])})
        decision_context = plans[indices[0]].get("decision_context", {})
        context = {
            "role": role, "origin_contexts": origin_contexts,
            "evaluation_scope": decision_context.get("evaluation_scope"),
            "context_resolution": "Each sample.context_ref resolves to origin_contexts with causal numeric history and current inputs",
            "candidate_agent_profile": decision_context.get("candidate_agent_profile"),
            "evolution_context": {key: decision_context[key] for key in
                                  ("candidate_parameters", "tool_experience", "agent_policy") if key in decision_context},
        }
        compact, shared = compact_origin_contexts(origin_contexts)
        if shared:
            context["origin_contexts"] = compact
            context["shared_origin_values"] = shared
            context["context_resolution"] += (
                "; within origin_contexts, {shared_origin_ref: key} means the exact value in "
                "shared_origin_values[key]. Resolve references without repeating the data or its analysis."
            )
        context["prediction_contract"] = {
            "owner": "sample_agent", "max_prediction_tool_calls": PREDICTION_TOOL_CALL_BUDGET,
            "model_fit_cache_capacity": 8,
            "exploration_budget": "Per-attempt tool calls only; cache eviction does not remove capabilities",
            "analysis_stopping_rule": (
                "Use inherited policy and causal observations first. Direct prediction is allowed. "
                "Analyze the origin vector once, not each cell as a separate research task. "
                "Use a second numerical call only for an unresolved discrepancy or uncertainty. "
                "Two is the hard call limit, not a target. Do not manually reconstruct model fits. "
                "Do not repeat equivalent parameter requests. Stop when evidence is sufficient "
                "and submit the full prediction vector with concise reason codes."
            ),
            "final_prediction": "Agent submits numeric predictions; tools are optional evidence",
            "allowed_methods": ["direct", "model", "blend", "adjusted"],
            "training_boundary": "Models fit training_fit only; evaluation labels are unavailable",
        }
        catalog = self._available_tool_catalog(selected[0], plans[indices[0]], role=role)
        try:
            with self._prediction_tool_context(
                role=role, model_id=self.strategy_model_id, requests=selected,
                samples=samples, context=context, available_tools=catalog,
            ) as binding:
                raw = self._agent_decide(
                    self.strategy_model_id, role=role, samples=samples, context=context,
                    available_tools=catalog, allow_format_retry=False, diagnostics=diagnostics,
                )
                structured = {"schema_version": "ecology-sample-predictions@2", "wave_digest": binding.wave_digest, **raw}
                rows = validate_predictions(structured, [r.sample_id for r in selected], wave_digest=binding.wave_digest)

        except SampleExecutionControlError:
            raise
        except Exception as exc:
            runtime_error = dsh_native_runtime_error_in_chain(exc)
            if runtime_error is not None and (dsh_native_runtime_retryable(runtime_error)
                                              or dsh_native_runtime_evaluation_fatal(runtime_error)):
                raise
            failure_class, retryable, error_type = classify_sample_failure(exc)
            for index in indices:
                failure = SampleExecutionAttemptError("sample Agent attempt failed", failure_class=failure_class,
                    retryable=retryable, error_type=error_type,
                    tool_calls=binding.public_trace(requests[index].sample_id) if "binding" in locals() else ())
                failure.__cause__ = exc
                outcomes[index] = SamplePredictionOutcome(sample_id=requests[index].sample_id, error=failure)
            return
        by_id = {row["sample_id"]: row for row in rows}
        successful = []
        for index in indices:
            request = requests[index]
            row = by_id[request.sample_id]
            tool_trace = binding.public_trace(request.sample_id, row["evidence_call_ids"])
            final_step = {
                "tool_id": "agent-final-prediction", "version": "2", "status": "completed",
                "input_digest": binding.wave_digest, "output_digest": digest(structured),
                "execution_owner": "dsh_agent_prediction",
            }
            agent_step = {
                "role": "remote_planner_agent", "decision": "submit_prediction:" + row["method"],
                "status": "completed", "model_id": self.strategy_model_id,
                "reason_code": row["reason_code"], "confidence": row["confidence"],
                "response_digest": digest(row),
            }
            item = {"index": index, "request": request, "predicted": row["predicted"],
                    "agent_steps": [agent_step], "tool_step": final_step,
                    "decision": row, "tool_trace": tool_trace,
                    "causal_context": origin_contexts[samples[indices.index(index)]["context_ref"]]}
            successful.append(item)
            outcomes[index] = SamplePredictionOutcome(sample_id=request.sample_id, result={
                "predicted": row["predicted"], "agent_decisions": [agent_step],
                "tool_calls": [*tool_trace, final_step],
            })
        review = [item for item in successful if self.require_remote_critic or self.remote_critic_policy is None
                  or self.remote_critic_policy["version"] == "always@1"
                  or item["decision"]["confidence"] < self.remote_critic_policy["min_planner_confidence"]]
        if self.remote_review_enabled and review:
            self._review_successes(review, plans, outcomes, diagnostics)

    def _review_successes(self, successful, plans, outcomes, diagnostics, *, compact=False):
        """Review causal evidence; all non-accept decisions return to the Agent."""
        samples = [{
            "sample_id": item["request"].sample_id,
            "target": item["request"].target,
            "horizon_hours": item["request"].horizon_hours,
            "physical_bounds": [item["request"].minimum, item["request"].maximum],
            "baseline": item["request"].baseline,
            "prediction": dict(item["decision"]),
            "causal_context": item["causal_context"],
            "tool_evidence": item["tool_trace"],
        } for item in successful]
        try:
            raw = self._agent_decide(
                self.review_model_id, role="critic", samples=samples,
                context={"review_contract": {
                    "owner": "sample_agent", "phase": "pre_score_label_free",
                    "actions": ["accept", "revise", "uncertain"],
                    "revision": "Return evidence concerns to the Agent under its existing attempt budget; never substitute a tool value",
                    "hard_constraints": "Host validates finite values and physical bounds after Agent execution",
                }}, available_tools=[], allow_format_retry=False, diagnostics=diagnostics,
            )
            reviews = {row["sample_id"]: row for row in validate_agent_review(raw, [s["sample_id"] for s in samples])}
        except SampleExecutionControlError:
            raise
        except Exception as exc:
            runtime_error = dsh_native_runtime_error_in_chain(exc)
            if runtime_error is not None and (dsh_native_runtime_retryable(runtime_error)
                                              or dsh_native_runtime_evaluation_fatal(runtime_error)):
                raise
            # A selected review is part of this attempt's contract. Unavailability
            # must be visible and cannot silently turn into scientific acceptance.
            for item in successful:
                outcomes[item["index"]] = SamplePredictionOutcome(
                    sample_id=item["request"].sample_id,
                    error=SampleExecutionAttemptError(
                        "required Agent review failed", failure_class="invalid_output",
                        retryable=True, error_type=type(exc).__name__,
                        agent_decisions=item["agent_steps"],
                        tool_calls=[*item["tool_trace"], item["tool_step"]],
                        previous_prediction=item["predicted"],
                    ),
                )
            return
        for item in successful:
            row = reviews[item["request"].sample_id]
            step = {"role": "remote_critic_agent", "decision": row["action"],
                    "status": "completed", "model_id": self.review_model_id,
                    "reason_code": row["reason_code"], "confidence": row["confidence"],
                    "response_digest": digest(row)}
            decisions = [*item["agent_steps"], step]
            calls = [*item["tool_trace"], item["tool_step"]]
            if row["action"] == "accept":
                outcome = SamplePredictionOutcome(sample_id=item["request"].sample_id, result={
                    "predicted": item["predicted"], "agent_decisions": decisions, "tool_calls": calls,
                })
            else:
                outcome = SamplePredictionOutcome(sample_id=item["request"].sample_id,
                    error=SampleExecutionAttemptError(
                        "Critic requested Agent reconsideration", failure_class="critic_" + row["action"],
                        retryable=True, error_type="AgentReviewRequestedRevision",
                        agent_decisions=decisions, tool_calls=calls, previous_prediction=item["predicted"],
                    ))
            outcomes[item["index"]] = outcome


__all__ = ["DshSampleCollaborationAdapter"]
