from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecologyrsi_dsh.api.dsh_tools import (
    DshToolAdmissionClosedError,
    DshToolAuthorizationError,
    DshToolService,
    ROLE_TOOLS,
)
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.server import EvolutionHTTPServer


def _identity(ledger: EventLedger, **overrides: object) -> dict:
    value = {
        "run_id": "run:tool-test",
        "role": "sample-planner",
        "stage": "sample.plan",
        "run_state_revision": 3,
        "stage_attempt": 2,
        "ledger_expected_revision": ledger.latest_seq(),
        "session_id": "session:planner-1",
        "idempotency_key": "tool-idem-1",
        "child_reservation_id": "reservation-1",
        "activation_lease_id": "lease-1",
        "genome_digest": "a" * 64,
        "compiled_behavior_digest": "b" * 64,
        "phenotype_instance_digest": "c" * 64,
    }
    value.update(overrides)
    return value


class DshToolServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger(":memory:")
        self.ledger.append("run:tool-test", "RunCreated", {"test": True})
        self.service = DshToolService(self.ledger)
        self.service.open_admission("run:tool-test", 3, 2)

    def tearDown(self) -> None:
        self.ledger.close()

    def test_role_surface_and_idempotency_are_fail_closed(self) -> None:
        envelope = {
            "identity": _identity(self.ledger),
            "arguments": {
                "schema_version": "ecology-sample-decisions@1",
                "wave_digest": "d" * 64,
                "decisions": [],
            },
        }
        first = self.service.execute("ecology_submit_sample_decisions", envelope)
        second = self.service.execute("ecology_submit_sample_decisions", envelope)
        self.assertEqual(first, second)
        changed = json.loads(json.dumps(envelope))
        changed["arguments"]["decisions"] = [{"sample_id": "s1", "prediction": 1.0}]
        with self.assertRaises(ValueError):
            self.service.execute("ecology_submit_sample_decisions", changed)
        wrong_role = json.loads(json.dumps(envelope))
        wrong_role["identity"]["role"] = "researcher"
        with self.assertRaises(DshToolAuthorizationError):
            self.service.execute("ecology_submit_sample_decisions", wrong_role)

    def test_model_cannot_override_identity_or_observe_labels(self) -> None:
        for arguments in (
            {"nested": {"run_id": "forged"}},
            {"rows": [{"ground_truth": 21.0}]},
            {"rows": [{"observed_temperature": 21.0}]},
        ):
            with self.assertRaises(DshToolAuthorizationError):
                self.service.execute(
                    "ecology_execute_prediction_tool",
                    {"identity": _identity(self.ledger), "arguments": arguments},
                )

    def test_closed_fence_rejects_late_submission(self) -> None:
        self.service.close_admission("run:tool-test", 3, 2)
        with self.assertRaises(DshToolAdmissionClosedError):
            self.service.execute(
                "ecology_get_run_context",
                {
                    "identity": _identity(self.ledger, role="coordinator"),
                    "arguments": {},
                },
            )

    def test_structured_result_is_durably_idempotent_and_stage_bound(self) -> None:
        identity = _identity(
            self.ledger,
            role="researcher",
            stage="generation.research",
            idempotency_key="research-result-1",
        )
        structured = {
            "schema_version": "ecology-research-result@1",
            "summary": "bounded evidence",
            "evidence": [],
        }
        envelope = {
            "identity": identity,
            "output_schema_id": "ecology-research-result@1",
            "structured": structured,
            "result_digest": digest(structured),
            "session_metrics": {
                "schema_version": "ecologyrsi-dsh.dsh-session-metrics/1",
                "session_id": identity["session_id"],
                "context_pressure": {
                    "available": True,
                    "source": "dsh_token_meter",
                    "measurement": "current_context_pressure",
                    "log_revision": 9,
                    "baseline_kind": "usage",
                    "total_tokens": 120,
                    "surface_tokens": 80,
                },
                "provider_usage": {
                    "available": True,
                    "source": "dsh_session_projection_token_usage",
                    "measurement": "cumulative_provider_reported_usage",
                    "totals": {
                        "uncached_input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_tokens": 30,
                        "cache_write_tokens": 0,
                        "total_tokens": 150,
                    },
                },
            },
        }
        first = self.service.accept_structured(envelope)
        second = self.service.accept_structured(envelope)
        self.assertEqual(first, second)
        self.assertEqual(
            [event.kind for event in self.ledger.events("run:tool-test")].count(
                "DshStructuredResultAccepted"
            ),
            1,
        )
        accepted = next(
            event
            for event in self.ledger.events("run:tool-test")
            if event.kind == "DshStructuredResultAccepted"
        )
        self.assertEqual(
            accepted.payload["session_metrics"]["provider_usage"]["totals"][
                "total_tokens"
            ],
            150,
        )
        wrong = json.loads(json.dumps(envelope))
        wrong["output_schema_id"] = "ecology-generation-review@1"
        with self.assertRaises(DshToolAuthorizationError):
            self.service.accept_structured(wrong)

    def test_cross_language_schemas_are_closed_and_role_sets_match(self) -> None:
        root = Path(__file__).resolve().parents[1] / "integrations" / "dsh_ecology_plugin"
        schemas = sorted((root / "schemas").glob("*.schema.json"))
        self.assertEqual(len(schemas), 8)
        for path in schemas:
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(value["additionalProperties"])
            self.assertTrue(value["$id"].startswith("ecology-"))
        self.assertFalse(any("submit" in item for item in ROLE_TOOLS["researcher"]))
        self.assertFalse(any("submit" in item for item in ROLE_TOOLS["candidate-proposer"]))
        self.assertFalse(any("submit" in item for item in ROLE_TOOLS["generation-judge"]))


class DshToolHTTPAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.environment = patch.dict(
            os.environ, {"ECOLOGYRSI_SIDECAR_TOOL_TOKEN": "tool-secret"}
        )
        self.environment.start()
        self.server = EvolutionHTTPServer(
            ("127.0.0.1", 0), Path(self.directory.name) / "events.sqlite3"
        )
        self.server.ledger.append("run:tool-test", "RunCreated", {"test": True})
        self.server.dsh_tools.open_admission("run:tool-test", 3, 2)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.environment.stop()
        self.directory.cleanup()

    def _request(self, token: str) -> tuple[int, dict]:
        envelope = {
            "identity": _identity(self.server.ledger, role="coordinator"),
            "arguments": {},
        }
        request = Request(
            f"http://127.0.0.1:{self.server.server_address[1]}"
            "/api/ecology-agent-sidecar/v1/tools/ecology_get_run_context",
            data=json.dumps(envelope).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_internal_tool_token_is_separate_and_required(self) -> None:
        status, _ = self._request("wrong")
        self.assertEqual(status, 401)
        status, payload = self._request("tool-secret")
        self.assertEqual(status, 200)
        self.assertTrue(payload["accepted"])


if __name__ == "__main__":
    unittest.main()
