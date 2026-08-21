"""Schema-bound DSH stage facade used by native evolution roles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.models import canonical_json, digest
from .dsh_native_runtime import DshNativeAgentRuntimeClient


class DshStructuredRoleRuntime:
    def __init__(self, client: DshNativeAgentRuntimeClient, *, admission: Any = None) -> None:
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
    ) -> dict[str, Any]:
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
            },
        }
        if self.admission is not None:
            self.admission.open_admission(run_id, run_state_revision, stage_attempt)
        try:
            response = self.client.run_stage(request)
            structured = dict(response["structured"])
            if response["result_digest"] != digest(structured):
                raise ValueError("DSH structured result digest mismatch")
            return structured
        finally:
            if self.admission is not None:
                self.admission.close_admission(run_id, run_state_revision, stage_attempt)


__all__ = ["DshStructuredRoleRuntime"]
