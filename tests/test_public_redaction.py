from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.api.events import EventEndpointsMixin
from ecologyrsi_dsh.api.projection import _run_failure_projection, _safe_plan_value
from ecologyrsi_dsh.api.shared import _public_http_error
from ecologyrsi_dsh.core.redaction import (
    public_error_summary,
    public_exception_summary,
)
from ecologyrsi_dsh.presentation.training_assets import _training_safe_value
from ecologyrsi_dsh.presentation.trajectory import _safe_value, _stage_event


class PublicRedactionTests(unittest.TestCase):
    def test_all_public_value_sanitizers_reject_folded_secret_keys_deeply(self) -> None:
        payload = {
            "safe": {
                "clientSecret": "client-secret-value",
                "accessToken": "access-token-value",
                "Authorization.Token": "authorization-token-value",
                "nested": [
                    {"DB.Client-Secret": "nested-secret-value"},
                    {"status": "ok", "token_limit": 1000},
                ],
            },
            "inline": "request failed; Authorization.Token=inline-secret-value",
        }

        for sanitizer in (
            _safe_plan_value,
            _training_safe_value,
            _safe_value,
        ):
            with self.subTest(sanitizer=sanitizer.__module__):
                sanitized = sanitizer(payload)
                encoded = json.dumps(sanitized, sort_keys=True)
                for secret in (
                    "client-secret-value",
                    "access-token-value",
                    "authorization-token-value",
                    "nested-secret-value",
                    "inline-secret-value",
                ):
                    self.assertNotIn(secret, encoded)
                self.assertEqual(sanitized["safe"]["nested"][1]["status"], "ok")
                self.assertEqual(
                    sanitized["safe"]["nested"][1]["token_limit"], 1000
                )

    def test_public_exception_summary_uses_only_type_and_bounded_code(self) -> None:
        class CredentialError(RuntimeError):
            error_code = "gateway_timeout"

        summary = public_exception_summary(
            CredentialError("Bearer audit-secret-token at a private endpoint")
        )

        self.assertEqual(summary, "CredentialError [gateway_timeout]")
        self.assertNotIn("audit-secret-token", summary)
        self.assertEqual(
            public_error_summary(
                "GatewayResponseError: Bearer legacy-audit-secret-token"
            ),
            "GatewayResponseError",
        )
        self.assertEqual(
            public_error_summary(
                "ValueError: no timestamp column in /private/var/tmp/audit/meteo.csv"
            ),
            "ValueError",
        )
        self.assertEqual(
            public_error_summary(
                r"OSError: cannot read C:\Users\audit\private.csv"
            ),
            "OSError",
        )

    def test_event_and_run_failure_projections_redact_legacy_bearer_text(self) -> None:
        event = SimpleNamespace(
            seq=1,
            event_id="event:failure",
            run_id="run:failure",
            kind="EvolutionStageRecorded",
            payload={
                "generation": 0,
                "stage": "proposal",
                "status": "failed",
                "attempt": 1,
                "proposal_id": None,
                "candidate_id": None,
                "public_error": (
                    "GatewayResponseError: Bearer public-event-audit-secret"
                ),
            },
            created_at="2026-08-19T00:00:00+00:00",
        )
        projected_event = EventEndpointsMixin._event_json(event)
        self.assertEqual(
            projected_event["payload"]["public_error"], "GatewayResponseError"
        )

        state = SimpleNamespace(
            events=(
                SimpleNamespace(
                    kind="RunFailed",
                    payload={"reason": "run failed: Bearer run-audit-secret"},
                    created_at="2026-08-19T00:00:00+00:00",
                ),
                event,
            )
        )
        reason, failed_stage = _run_failure_projection(state)
        encoded = json.dumps(
            {"reason": reason, "failed_stage": failed_stage}, sort_keys=True
        )
        self.assertNotIn("public-event-audit-secret", encoded)
        self.assertNotIn("run-audit-secret", encoded)
        self.assertEqual(
            _stage_event([event], "proposal")["public_error"],
            "GatewayResponseError",
        )

    def test_http_and_proposal_text_never_reflect_inline_credentials(self) -> None:
        marker = "http-audit-secret-marker"
        self.assertEqual(
            _public_http_error(RuntimeError(f"request failed: Bearer {marker}")),
            "RuntimeError",
        )
        self.assertEqual(
            _public_http_error(
                ValueError("no timestamp column in /private/var/tmp/audit/meteo.csv")
            ),
            "ValueError",
        )
        proposal_event = SimpleNamespace(
            seq=2,
            event_id="event:proposal",
            run_id="run:proposal",
            kind="ProposalSubmitted",
            payload={
                "proposal": {
                    "proposal_id": "proposal:1",
                    "title": f"clientSecret={marker}",
                }
            },
            created_at="2026-08-19T00:00:00+00:00",
        )
        projected = EventEndpointsMixin._event_json(proposal_event)
        self.assertNotIn(marker, json.dumps(projected, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
