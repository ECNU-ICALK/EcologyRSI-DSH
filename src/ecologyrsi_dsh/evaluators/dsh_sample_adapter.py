"""DSH-owned planner/critic routing over the existing Host prediction tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..core.models import digest
from ..core.redaction import REMOTE_REASON_CODES
from ..integrations.dsh_structured_roles import DshStructuredRoleRuntime
from .gateway_sample_adapter import (
    GatewaySampleCollaborationAdapter,
    GatewaySampleTool,
)
from .sample_execution import SampleExecutionContractError


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
        wave = {
            "schema_version": "ecologyrsi-dsh.sample-routing-wave/1",
            "role": role,
            "model_route_id": model_id,
            "samples": [dict(item) for item in samples],
            "context": dict(context),
            "available_tools": [dict(item) for item in available_tools],
            "allowed_reason_codes": sorted(REMOTE_REASON_CODES),
        }
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


class DshSampleCollaborationAdapter(GatewaySampleCollaborationAdapter):
    """Reuse the stable Host tool/retry/scoring loop with DSH as role owner."""

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
        forecast_tool: Callable[..., Any] | None = None,
        tools: Sequence[GatewaySampleTool] = (),
        microbatch_size: int = 128,
        sample_concurrency: int = 4,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        run_control_callback: Callable[[], str] | None = None,
        remote_critic_policy: Mapping[str, Any] | None = None,
        sample_planner_prompt_profile: Mapping[str, Any] | None = None,
    ) -> None:
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
            forecast_tool=forecast_tool,
            tools=tools,
            microbatch_size=microbatch_size,
            sample_concurrency=sample_concurrency,
            progress_callback=progress_callback,
            run_control_callback=run_control_callback,
            remote_critic_policy=remote_critic_policy,
            sample_planner_prompt_profile=sample_planner_prompt_profile,
            operation_max_tokens=None,
            token_limit=0,
            token_reservation_per_wave=0,
        )
        # The legacy base selects its own version based on optional profiles.
        # Restore the explicit DSH protocol identity after initialization.
        self.adapter_id = "dsh-native-sample-collaboration"
        self.adapter_version = "1"

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
