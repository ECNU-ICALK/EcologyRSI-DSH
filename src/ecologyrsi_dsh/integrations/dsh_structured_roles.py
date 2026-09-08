"""Schema-bound DSH stage facade used by native evolution roles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.errors import DshNativeRuntimeUnavailableError
from ..core.model_execution_policy import research_execution_policy
from ..core.models import canonical_json, digest
from .dsh_native_runtime import DshNativeAgentRuntimeClient


class DshStructuredRoleRuntime:
    def __init__(
        self,
        client: DshNativeAgentRuntimeClient,
        *,
        admission: Any = None,
    ) -> None:
        if isinstance(client, DshNativeAgentRuntimeClient) and admission is None:
            raise DshNativeRuntimeUnavailableError(
                "The real DSH native runtime requires a Host-local admission service.",
                error_code="dsh_native_runtime_contract_error",
            )
        if not isinstance(client, DshNativeAgentRuntimeClient) and not hasattr(
            client, "run_stage"
        ):
            raise TypeError("structured role runtime requires the narrow DSH client")
        self.client = client
        self.admission = admission

    def run(
        self,
        *,
        run_id: str,
        stage: str,
        role: str,
        context: Mapping[str, Any],
        output_schema_id: str,
        run_state_revision: int,
        stage_attempt: int,
        ledger_expected_revision: int,
        idempotency_key: str,
        identity_digests: Mapping[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        policy = research_execution_policy(context) if stage == "generation.research-synthesis" else None
        maximum = policy["synthesis_max_output_tokens"] if policy is not None else 8192
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 512 <= max_tokens <= maximum
        ):
            raise ValueError(f"DSH structured max_tokens must be between 512 and {maximum}")
        if policy is not None and max_tokens != maximum:
            raise ValueError("DSH synthesis max_tokens must match the frozen research execution policy")
        # The exact budget is inside the canonical context and therefore the
        # child item digest and role-stage phenotype identity used by replay.
        request = {
            "run_id": run_id,
            "stage": stage,
            "run_state_revision": run_state_revision,
            "stage_attempt": stage_attempt,
            "ledger_expected_revision": ledger_expected_revision,
            "idempotency_key": idempotency_key,
            "request": {
                "role": role,
                "output_schema_id": output_schema_id,
                "context": dict(context),
                "context_canonical_json": canonical_json(context),
                "context_digest": digest(context),
                "identity_digests": dict(identity_digests or {}),
                **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            },
        }
        if self.admission is not None:
            fence = self.admission.open_admission(
                run_id,
                run_state_revision,
                stage_attempt,
                role=role,
                stage=stage,
                idempotency_key=idempotency_key,
            )
            request["admission_id"] = fence.admission_id
        try:
            replay = getattr(self.admission, "replay_structured_result", None)
            if callable(replay):
                prior = replay(
                    run_id=run_id,
                    stage=stage,
                    role=role,
                    stage_attempt=stage_attempt,
                    idempotency_key=idempotency_key,
                    output_schema_id=output_schema_id,
                    identity_digests=dict(identity_digests or {}),
                )
                if prior is not None:
                    return dict(prior)
            response = self.client.run_stage(request)
            structured = dict(response["structured"])
            if response["result_digest"] != digest(structured):
                raise ValueError("DSH structured result digest mismatch")
            return structured
        finally:
            if self.admission is not None:
                self.admission.close_admission(run_id, run_state_revision, stage_attempt)


__all__ = ["DshStructuredRoleRuntime"]
