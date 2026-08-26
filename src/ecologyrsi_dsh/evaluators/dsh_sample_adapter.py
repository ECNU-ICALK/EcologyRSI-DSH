"""DSH-owned planner/critic routing over the existing Host prediction tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from threading import RLock
from typing import Any

from ..core.models import digest
from ..core.redaction import REMOTE_REASON_CODES
from ..integrations.dsh_structured_roles import DshStructuredRoleRuntime
from .gateway_sample_adapter import (
    GatewaySampleCollaborationAdapter,
)
from .sample_execution import (
    SampleExecutionContractError,
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
        **_legacy_options: Any,
    ) -> Mapping[str, Any]:
        if role not in {"planner", "repair", "critic"}:
            raise SampleExecutionContractError("unsupported DSH sample role")
        dsh_role = "sample-critic" if role == "critic" else "sample-planner"
        stage = "sample.critic" if role == "critic" else "sample.plan"
        schema_id = (
            "ecology-sample-review@1"
            if role == "critic"
            else "ecology-sample-decisions@1"
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


class DshSampleCollaborationAdapter(GatewaySampleCollaborationAdapter):
    """Run the only supported DSH sample protocol: one vector chain per origin."""

    adapter_id = "dsh-native-sample-collaboration"
    adapter_version = "1"

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
        microbatch_size: int = 128,
        sample_concurrency: int = 4,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        admission_snapshot_provider: Callable[[], Mapping[str, int]] | None = None,
        run_control_callback: Callable[[], str] | None = None,
        remote_critic_policy: Mapping[str, Any] | None = None,
        sample_reflection_policy: str | None = None,
        sample_planner_prompt_profile: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(forecast_bundle_tool):
            raise TypeError("strict DSH execution requires a vector prediction tool")
        if not callable(prediction_tool_binder):
            raise TypeError("strict DSH execution requires an agent prediction-tool binder")
        if admission_snapshot_provider is not None and not callable(
            admission_snapshot_provider
        ):
            raise TypeError("admission_snapshot_provider must be callable")
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
        super().__init__(
            client,
            strategy_model_id=strategy_model_id,
            review_model_id=review_model_id,
            remote_review_enabled=True,
            forecast_bundle_tool=forecast_bundle_tool,
            tools=(),
            # Every target/horizon cell for one verified forecast origin is
            # sent in the same Planner/Critic wave.
            microbatch_size=microbatch_size,
            sample_concurrency=sample_concurrency,
            progress_callback=(
                self._handle_strict_gateway_progress
                if progress_callback is not None
                else progress_callback
            ),
            run_control_callback=run_control_callback,
            remote_critic_policy=remote_critic_policy,
            sample_planner_prompt_profile=sample_planner_prompt_profile,
            require_remote_planner=True,
            require_remote_critic=require_success_critic,
            operation_max_tokens=None,
            token_limit=0,
            token_reservation_per_wave=0,
        )
        # The gateway base derives versions from optional profiles. DSH has a
        # single explicit protocol identity instead.
        self.adapter_id = "dsh-native-sample-collaboration"
        self._decision_client = client
        self.adapter_version = (
            "4-concurrent-origin-bundle"
            if require_success_critic and self.sample_reflection_enabled
            else "5-adaptive-sparse-origin-review"
        )

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
        if role != "planner":
            return super()._prediction_tool_context(
                role=role,
                model_id=model_id,
                requests=requests,
                samples=samples,
                context=context,
                available_tools=available_tools,
            )
        if len(available_tools) != 1:
            raise SampleExecutionContractError(
                "strict DSH Planner requires exactly one frozen prediction tool"
            )
        tool_id = str(available_tools[0].get("tool_id") or "").strip()
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
            tool_id=tool_id,
            sample_ids=sample_ids,
            executor=lambda: self._forecast_bundle_tool(frozen_requests),
        )

    def set_outcome_callback(self, callback: Callable[..., Any] | None) -> None:
        """Delay durable publication until mandatory post-score reflection exists."""

        self._outcome_callback = None
        self._durable_outcome_statuses = None
        self._planner_progress_session = None

    def set_resume_checkpoint(self, checkpoint: Mapping[str, Any] | None) -> None:
        """Keep cumulative progress while strict chains stream one sample at a time."""

        with self._strict_progress_lock:
            super().set_resume_checkpoint(checkpoint)
            # The base adapter's cumulative session is invocation-local and
            # cannot be shared by concurrent origin chains. Strict progress is
            # aggregated below after each complete origin is durable.
            self._resume_checkpoint = None
            self._durable_outcome_statuses = None
            self._planner_progress_session = None
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
        plan = dict(super().plan_batch(context))
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
                "execution_mode": "per_origin_remote_agent_vector_tool_loop",
                "host_route_bypass_allowed": False,
                "host_prediction_fallback_allowed": False,
                "post_score_reflection_required": self.sample_reflection_enabled,
                "required_success_remote_roles": required_remote_roles,
                "remote_roles": required_remote_roles,
                "routing_policy": (
                    "remote_planner_per_origin_then_agent_invoked_registered_vector_"
                    "tool_then_sparse_remote_critic_on_uncertainty_or_failure_then_"
                    "host_cell_scoring_then_candidate_aggregate_reflection;"
                    "no_host_route_bypass;no_prediction_fallback"
                    if required_remote_roles == ["planner"]
                    else
                    "remote_planner_per_origin_then_agent_invoked_registered_vector_"
                    "tool_then_remote_critic_per_origin_then_host_cell_scoring_"
                    "then_remote_reflector_per_origin;no_host_route_bypass;"
                    "no_prediction_fallback"
                ),
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

    def _available_tool_catalog(
        self,
        request: Any,
        plan: Mapping[str, Any],
        *,
        role: str,
    ) -> list[dict[str, str]]:
        """Keep the initial path bound to one registered candidate tool."""

        if role != "planner":
            return super()._available_tool_catalog(request, plan, role=role)
        return [
            {
                "tool_id": request.algorithm_id,
                "version": request.algorithm_version,
                "purpose": "registered_candidate_prediction",
            }
        ]

    def _review_successes(
        self,
        successful: Sequence[Mapping[str, Any]],
        plans: Sequence[Mapping[str, Any]],
        outcomes: list[Any],
        diagnostics: Any,
        *,
        compact: bool = False,
    ) -> None:
        """Keep DSH critic waves bounded while reviewing every selected sample."""

        super()._review_successes(
            successful,
            plans,
            outcomes,
            diagnostics,
            compact=True,
        )

    def _review_tool_failures(
        self,
        failures: Sequence[Mapping[str, Any]],
        plans: Sequence[Mapping[str, Any]],
        outcomes: list[Any],
        diagnostics: Any,
        *,
        compact: bool = False,
    ) -> None:
        super()._review_tool_failures(
            failures,
            plans,
            outcomes,
            diagnostics,
            compact=True,
        )


__all__ = ["DshSampleCollaborationAdapter"]
