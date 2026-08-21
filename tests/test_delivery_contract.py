from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecologyrsi_dsh.server import EvolutionHTTPServer


class DeliveryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.db = Path(self.directory.name) / "events.sqlite3"
        self.server = EvolutionHTTPServer(("127.0.0.1", 0), self.db)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.directory.cleanup()

    def request(self, path: str, method: str = "GET", body: dict | None = None) -> tuple[int, object]:
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self.base + path,
            data=encoded,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def create(self, *, auto_advance: int | bool = 0, key: str = "create-key") -> tuple[str, dict]:
        status, payload = self.request(
            "/api/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "slot": "residual_water_stress",
                "budget": {"max_generations": 3, "max_candidates": 3},
                "auto_advance": auto_advance,
                "idempotency_key": key,
            },
        )
        self.assertIn(status, (200, 201), payload)
        assert isinstance(payload, dict)
        return payload["projection"]["run_id"], payload

    def test_health_reports_version_and_scientific_boundary(self) -> None:
        status, payload = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertIs(payload["ok"], True)
        self.assertEqual(payload["evaluation_partition"], "visible/validation/demo")
        self.assertEqual(payload["scientific_scope"], "prediction_demo_non_causal")
        self.assertTrue(payload["package_version"])

    def test_create_and_advance_receipts_survive_server_restart(self) -> None:
        body = {
            "domain_pack_id": "crop_soil_water",
            "dataset_id": "generated-toy-series@1",
            "slot": "residual_water_stress",
            "budget": {"max_generations": 3, "max_candidates": 3},
            "auto_advance": 0,
            "idempotency_key": "restart-create",
        }
        status, first = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 201)
        status, repeated = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 200)
        self.assertEqual(repeated, first)

        run_id = first["projection"]["run_id"]
        path = "/api/runs/" + run_id.replace(":", "%3A") + "/advance"
        advance_body = {"steps": 1, "idempotency_key": "restart-advance"}
        status, advanced = self.request(path, "POST", advance_body)
        self.assertEqual(status, 200)

        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.server = EvolutionHTTPServer(("127.0.0.1", 0), self.db)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

        status, replayed = self.request(path, "POST", advance_body)
        self.assertEqual(status, 200)
        self.assertEqual(replayed, advanced)

    def test_http_refuses_test_partition_and_terminal_advance(self) -> None:
        run_id, created = self.create(auto_advance=0, key="boundary-create")
        path = "/api/runs/" + run_id.replace(":", "%3A")
        status, rejected = self.request(
            path + "/advance",
            "POST",
            {"steps": 1, "split": "test"},
        )
        self.assertEqual(status, 400)
        self.assertIn("validation", rejected["error"])
        self.assertEqual(created["projection"]["candidates_count"], 0)

        status, cancelled = self.request(
            path + "/control",
            "POST",
            {"action": "cancel", "reason": "contract test"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(cancelled["projection"]["status"], "cancelled")
        status, terminal_error = self.request(path + "/advance", "POST", {"steps": 1})
        self.assertEqual(status, 400)
        self.assertIn("running", terminal_error["error"])

    def test_http_rejects_unknown_dataset(self) -> None:
        status, payload = self.request(
            "/api/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "field-observations-2026",
                "budget": 1,
                "auto_advance": 0,
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("generated-toy-series", payload["error"])

    def test_create_validates_auto_advance_and_generation_budget_before_writing(self) -> None:
        for index, value in enumerate((33, -1, 1.5, None)):
            status, payload = self.request(
                "/api/runs",
                "POST",
                {
                    "domain_pack_id": "crop_soil_water",
                    "dataset_id": "generated-toy-series@1",
                    "budget": {"max_candidates": 3, "max_generations": 3},
                    "auto_advance": value,
                    "idempotency_key": f"invalid-auto-{index}",
                },
            )
            self.assertEqual(status, 400, payload)
            self.assertIn("auto_advance", payload["error"])

        status, payload = self.request(
            "/api/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "budget": {"max_candidates": 3, "max_generations": "bad"},
                "auto_advance": 0,
                "idempotency_key": "invalid-generation-budget",
            },
        )
        self.assertEqual(status, 400, payload)
        self.assertIn("max_generations", payload["error"])
        status, listing = self.request("/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(listing["runs"], [])

    def test_invalid_idempotent_commands_do_not_leave_pending_receipts(self) -> None:
        run_id, _created = self.create(auto_advance=0, key="preflight-create")
        path = "/api/runs/" + run_id.replace(":", "%3A")

        body = {"split": "hidden", "idempotency_key": "preflight-bad-split"}
        status, payload = self.request(path + "/advance", "POST", body)
        self.assertEqual(status, 400)
        self.assertIn("validation", payload["error"])
        status, repeated = self.request(path + "/advance", "POST", body)
        self.assertEqual(status, 400)
        self.assertIn("validation", repeated["error"])

        status, invalid_action = self.request(
            path + "/control",
            "POST",
            {"action": "rewind", "idempotency_key": "preflight-bad-action"},
        )
        self.assertEqual(status, 400)
        self.assertIn("action must be", invalid_action["error"])
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_completed_advance_receipt_replays_after_run_is_terminal(self) -> None:
        run_id, _created = self.create(auto_advance=0, key="terminal-create")
        path = "/api/runs/" + run_id.replace(":", "%3A")
        body = {"steps": 3, "idempotency_key": "terminal-advance"}
        status, first = self.request(path + "/advance", "POST", body)
        self.assertEqual(status, 200)
        self.assertEqual(first["projection"]["status"], "completed")
        status, replayed = self.request(path + "/advance", "POST", body)
        self.assertEqual(status, 200)
        self.assertEqual(replayed, first)


if __name__ == "__main__":
    unittest.main()
