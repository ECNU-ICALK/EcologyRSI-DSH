"""Resumable evolution execution and command receipt endpoints."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from ..core.ledger import CommandInProgressError, CommandReceipt
from ..core.redaction import safe_error_code
from ..integrations.dsh_native_runtime import DSH_NATIVE_EXECUTION_PROTOCOL
from .command_receipts import compact_command_receipt_payload
from .generation_execution import complete_if_budget_exhausted, execute_generation
from .projection import _control_payload
from .shared import (
    _assert_http_scope,
    _evaluation_partition,
    _parse_steps,
    _public_http_error,
)


_RETRY_CIRCUIT_PAUSE_CODES = frozenset(
    {
        "gateway_retry_circuit_open",
        "dsh_runtime_retry_circuit_open",
        "research_timeout_retry_circuit_open",
        "sample_persistence_retry_circuit_open",
    }
)


class ExecutionEndpointsMixin:
    def _command_status(self, command_key: str) -> dict[str, Any]:
        """Return a redacted, pollable status for an idempotent command.

        Long-running controls are allowed to acknowledge before the remote DSH
        drain finishes.  The receipt is the durable source of truth; callers
        must not infer completion from the HTTP connection that submitted it.
        """

        key = str(command_key or "").strip()
        if not key or len(key) > 512:
            raise ValueError("command id must be non-empty and at most 512 characters")
        receipt = self.server.ledger.command_receipt(key)
        if receipt is None:
            raise KeyError(f"unknown command: {key}")
        payload = compact_command_receipt_payload(receipt)
        if receipt.response is not None:
            payload["response"] = dict(receipt.response)
        else:
            run_id = receipt.resource_run_id or (None if receipt.command_kind == "create_run" else receipt.run_id)
            if run_id:
                try:
                    if receipt.command_kind.startswith("control:"):
                        payload["response"] = _control_payload(
                            self.server.director.replay(run_id),
                            self.server.sample_admission.snapshot(run_id),
                        )
                    else:
                        payload["response"] = self._run_payload(run_id)
                except (KeyError, RuntimeError, TypeError, ValueError):
                    # The command may be in the Host→DSH binding gap.  Keep the
                    # receipt visible without leaking internal exception text.
                    pass
        return payload

    def _advance_run(
        self,
        run_id: str,
        body: dict[str, Any],
        *,
        target_steps: int | None = None,
    ) -> Any:
        """Run bounded whole-generation steps and resume from durable events."""

        generation_lease = self.server.acquire_generation_lease(
            run_id, blocking=False
        )
        if generation_lease is None:
            raise CommandInProgressError(
                "run is already being advanced; wait for the active generation "
                "or pause it before retrying"
            )
        try:
            return self._advance_run_locked(
                run_id, body, target_steps=target_steps
            )
        finally:
            generation_lease.release()

    def _advance_run_locked(
        self,
        run_id: str,
        body: dict[str, Any],
        *,
        target_steps: int | None = None,
    ) -> Any:
        state, requested_steps, _split = self._validate_advance_request(run_id, body)
        if target_steps is not None:
            if (
                isinstance(target_steps, bool)
                or not isinstance(target_steps, int)
                or not 0 <= target_steps <= 32
            ):
                raise ValueError("target_steps must be an integer between 0 and 32")
            requested_steps = target_steps
        # A process may have committed GenerationAdvanced and then exited
        # before RunCompleted.  Repair that durable gap before counting the
        # command's completed generation steps; otherwise a retry with the same
        # idempotency key would be treated as a no-op forever.
        state = complete_if_budget_exhausted(self, run_id, state)
        if state.run.status.value == "completed":
            return state
        completed_steps = self._completed_command_steps(run_id)
        for _ in range(max(0, requested_steps - completed_steps)):
            state = execute_generation(self, run_id)
            if state.run.status.value != "running":
                break
        return self.server.director.state(run_id)

    def _validate_advance_request(
        self, run_id: str, body: dict[str, Any]
    ) -> tuple[Any, int, str]:
        state = self.server.director.state(run_id)
        _assert_http_scope(state)
        active = getattr(self, "_active_command", None)
        if state.run.status.value == "completed" and active is not None:
            return (
                state,
                _parse_steps(body),
                _evaluation_partition(state.task_manifest, body),
            )
        if state.run.status.value != "running":
            raise RuntimeError("run must be running to advance")
        steps = _parse_steps(body)
        split = _evaluation_partition(state.task_manifest, body)
        self._validate_frozen_runtime_bindings(
            state.task_manifest,
            run_id=state.run.run_id,
        )
        return state, steps, split

    def _validate_control_request(self, run_id: str, action: str) -> Any:
        state = self.server.director.state(run_id)
        _assert_http_scope(state)
        if action == "start" and state.run.status.value == "paused":
            paused = next(
                (
                    event
                    for event in reversed(state.events)
                    if event.kind == "RunPaused"
                ),
                None,
            )
            if (
                paused is not None
                and paused.payload.get("code") in _RETRY_CIRCUIT_PAUSE_CODES
            ):
                raise RuntimeError(
                    "retry circuit requires an explicit resume"
                )
        allowed = {
            "start": ("created", "paused"),
            "pause": ("running",),
            "resume": ("paused",),
            "cancel": ("created", "running", "paused"),
            "complete": ("running", "paused"),
        }[action]
        if state.run.status.value not in allowed:
            expected = ", ".join(allowed)
            raise RuntimeError(
                f"run {run_id} is {state.run.status.value}; expected {expected}"
            )
        if (
            action in {"start", "resume"}
            and state.task_manifest.metadata.get("execution_protocol")
            == DSH_NATIVE_EXECUTION_PROTOCOL
        ):
            # A DSH process restart clears its in-memory run registry while the
            # Python scientific ledger remains durable.  Recreate and validate
            # that exact frozen runtime binding before sending start/resume;
            # otherwise the runtime can only reject the control as unknown.
            self._validate_frozen_runtime_bindings(
                state.task_manifest,
                run_id=state.run.run_id,
            )
        return state

    @staticmethod
    def _command_key(run_id: str, body: dict[str, Any]) -> str | None:
        value = body.get("idempotency_key")
        if value is None or not str(value).strip():
            return None
        return f"{run_id}:{str(value).strip()}"

    def _claim_command(
        self,
        key: str | None,
        run_id: str,
        command_kind: str,
        body: dict[str, Any],
    ) -> dict[str, Any] | None:
        if key is None:
            return None
        active = getattr(self, "_active_command", None)
        if active is not None and active["key"] == key:
            return None
        existing = self.server.ledger.command_receipt(key)
        attempt_start_seq = self.server.ledger.latest_seq()
        receipt = self.server.ledger.begin_command(
            key,
            run_id,
            command_kind,
            body,
            resume_pending=True,
        )
        if receipt is None:
            owned = self.server.ledger.command_receipt(key)
            if owned is None:
                raise RuntimeError("命令收据未持久化")
            self._active_command = {
                "key": key,
                "receipt": owned,
                "new_claim": existing is None,
                "attempt_start_seq": attempt_start_seq,
                "request": dict(body),
            }
        return receipt.response if receipt is not None else None

    def _serve_existing_command(
        self,
        key: str | None,
        run_id: str,
        command_kind: str,
        body: dict[str, Any],
    ) -> bool:
        if key is None or self.server.ledger.command_receipt(key) is None:
            return False
        cached = self._claim_command(key, run_id, command_kind, body)
        if cached is None:
            return False
        self._send(HTTPStatus.OK, cached)
        return True

    def _complete_command(self, key: str | None, payload: dict[str, Any]) -> None:
        if key is None:
            return
        self.server.ledger.complete_command(key, payload)
        active = getattr(self, "_active_command", None)
        if active is not None and active["key"] == key:
            self._active_command = None

    def _completed_command_steps(self, run_id: str) -> int:
        active = getattr(self, "_active_command", None)
        if active is None:
            return 0
        receipt: CommandReceipt = active["receipt"]
        return sum(
            1
            for event in self.server.ledger.events(
                run_id, after_seq=receipt.start_seq
            )
            if event.kind == "GenerationAdvanced"
        )

    def _active_command_has_progress(
        self, active: dict[str, Any]
    ) -> bool | None:
        """Return command-scoped progress, or ``None`` when ownership is unclear."""

        receipt = self.server.ledger.command_receipt(active["key"])
        if receipt is None:
            return None
        command_kind = receipt.command_kind
        request = active.get("request")
        request = request if isinstance(request, dict) else {}
        run_id = receipt.resource_run_id

        if command_kind == "create_run" and run_id is None:
            requested_run_id = request.get("run_id")
            if isinstance(requested_run_id, str) and requested_run_id.strip():
                run_id = requested_run_id.strip()
            else:
                idempotency_key = request.get("idempotency_key")
                if not isinstance(idempotency_key, str) or not idempotency_key.strip():
                    return None
                for event in self.server.ledger.events(after_seq=receipt.start_seq):
                    if event.kind != "RunCreated":
                        continue
                    manifest = event.payload.get("task_manifest")
                    metadata = (
                        manifest.get("metadata")
                        if isinstance(manifest, dict)
                        else None
                    )
                    if (
                        isinstance(metadata, dict)
                        and metadata.get("idempotency_key") == idempotency_key.strip()
                    ):
                        return True
                return False
        elif run_id is None:
            run_id = receipt.run_id.strip() or None

        if run_id is None:
            return None
        events = self.server.ledger.events(run_id, after_seq=receipt.start_seq)
        if command_kind in {"advance", "control:advance", "control:step"}:
            # Any same-run write may be a resumable partial generation. The
            # generation lease prevents an explicit command from running a
            # second generation concurrently on this run.
            return bool(events)
        expected_event = {
            "create_run": "RunCreated",
            "control:start": "RunStarted",
            "control:pause": "RunPaused",
            "control:resume": "RunResumed",
            "control:cancel": "RunCancelled",
            "control:complete": "RunCompleted",
            "record_intervention": "HumanInterventionRecorded",
            "answer_expert_consultation": "ExpertConsultationAnswered",
        }.get(command_kind)
        if expected_event is None:
            return None
        return any(event.kind == expected_event for event in events)

    def _send_post_error(self, status: HTTPStatus, exc: BaseException) -> None:
        payload: dict[str, Any] = {"error": _public_http_error(exc)}
        error_code = safe_error_code(getattr(exc, "error_code", None))
        if error_code is not None:
            payload["error_code"] = error_code
        active = getattr(self, "_active_command", None)
        if active is not None:
            has_progress = self._active_command_has_progress(active)
            if active["new_claim"] and has_progress is False:
                self.server.ledger.abandon_command(active["key"])
                payload["retryable_with_same_idempotency_key"] = True
            else:
                payload.update(
                    {
                        "command_status": "等待恢复",
                        "retryable_with_same_idempotency_key": True,
                        "recovery_note": "使用相同幂等键重试将从已写入阶段继续。",
                    }
                )
            self._active_command = None
        self._send(status, payload)
