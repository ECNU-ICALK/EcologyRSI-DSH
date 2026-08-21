from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
from ecologyrsi_dsh.integrations.dsh_native_runtime import (
    DSH_NATIVE_EXECUTION_PROTOCOL,
    DshNativeAgentRuntimeClient,
)
from ecologyrsi_dsh.server import EvolutionHTTPServer


class _RuntimeHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self._reply()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(  # type: ignore[attr-defined]
            (self.command, self.path, self.headers.get("Authorization"), body)
        )
        self._reply()

    def _reply(self) -> None:
        status, payload = self.server.responses.pop(0)  # type: ignore[attr-defined]
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class DshNativeRuntimeClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _RuntimeHandler)
        self.server.responses = []  # type: ignore[attr-defined]
        self.server.requests = []  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = DshNativeAgentRuntimeClient(
            f"http://127.0.0.1:{self.server.server_address[1]}",
            token="runtime-secret",
            timeout=2,
        )

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()

    @staticmethod
    def _capabilities(*, ready: bool = True) -> dict:
        return {
            "schema_version": "ecology-agent-runtime-capabilities/1",
            "ready": ready,
            "root_services": {"required": ["agents"], "missing": [], "declared": True},
            "presets": [
                {
                    "preset_id": "ecology-researcher-v1",
                    "declared": True,
                    "standing_key": "standing:researcher",
                    "preset_mountable": True,
                    "tool_surface_verified": True,
                    "route_resolvable": True,
                    "live_agent_service_ready": True,
                    "first_call_verified": False,
                }
            ],
            "live_agent_service_ready": True,
            "first_call_verified": False,
        }

    @staticmethod
    def _accepted() -> dict:
        return {
            "accepted": True,
            "run_id": "run-1",
            "run_state_revision": 7,
            "stage_attempt": 2,
            "ledger_expected_revision": 11,
            "idempotency_key": "idem-1",
        }

    def test_capabilities_and_mutations_are_strict_and_bearer_authenticated(self) -> None:
        self.server.responses.extend([(200, self._capabilities()), (200, self._accepted())])  # type: ignore[attr-defined]
        capability = self.client.capabilities()
        self.client.require_capabilities(capability, ["ecology-researcher-v1"])
        response = self.client.create_run(
            {
                "run_id": "run-1",
                "run_state_revision": 7,
                "stage_attempt": 2,
                "ledger_expected_revision": 11,
                "idempotency_key": "idem-1",
            }
        )
        self.assertEqual(response["run_state_revision"], 7)
        self.assertEqual(response["stage_attempt"], 2)
        self.assertEqual(response["ledger_expected_revision"], 11)
        self.assertEqual(self.server.requests[0][2], "Bearer runtime-secret")  # type: ignore[attr-defined]
        self.assertEqual(DSH_NATIVE_EXECUTION_PROTOCOL, "dsh_native_plugin_evolution@1")

    def test_default_timeout_allows_dsh_managed_long_context_turns(self) -> None:
        client = DshNativeAgentRuntimeClient(
            "http://127.0.0.1:8848",
            token="runtime-secret",
        )
        # DSH owns the 600-second structured-stage deadline.  The Python HTTP
        # client leaves a response/cleanup margin so DSH can return its 502
        # timeout contract instead of losing the race to a local socket timeout.
        self.assertEqual(client.timeout, 660.0)

    def test_non_loopback_unknown_fields_and_capability_mismatch_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            DshNativeAgentRuntimeClient("http://localhost:8848", token="x")
        with self.assertRaises(ValueError):
            DshNativeAgentRuntimeClient("https://127.0.0.1.example/", token="x")
        self.server.responses.append((200, {**self._accepted(), "unexpected": True}))  # type: ignore[attr-defined]
        with self.assertRaises(DshNativeRuntimeUnavailableError):
            self.client.create_run(
                {
                    "run_id": "run-1",
                    "run_state_revision": 7,
                    "stage_attempt": 2,
                    "ledger_expected_revision": 11,
                    "idempotency_key": "idem-1",
                }
            )
        with self.assertRaises(DshNativeRuntimeUnavailableError):
            self.client.require_capabilities(
                self._capabilities(ready=False), ["ecology-researcher-v1"]
            )

    def test_remote_errors_and_transport_failures_never_disclose_token(self) -> None:
        self.server.responses.append(  # type: ignore[attr-defined]
            (503, {"error_code": "runtime_busy", "error": "runtime-secret unavailable"})
        )
        with self.assertRaises(DshNativeRuntimeUnavailableError) as raised:
            self.client.status("run-1")
        self.assertEqual(raised.exception.error_code, "runtime_busy")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertNotIn("runtime-secret", str(raised.exception))

    def test_caller_cancellation_fails_before_network(self) -> None:
        with self.assertRaises(DshNativeRuntimeUnavailableError) as raised:
            self.client.capabilities(cancelled=lambda: True)
        self.assertEqual(raised.exception.error_code, "dsh_native_runtime_cancelled")
        self.assertEqual(self.server.requests, [])  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()


class _FakeNativeRuntime:
    def __init__(self, *, unavailable: bool = False) -> None:
        self.unavailable = unavailable
        self.created: list[dict] = []
        self.cancelled: list[dict] = []
        self.live = False
        self.run_ids: set[str] = set()

    def capabilities(self) -> dict:
        if self.unavailable:
            raise DshNativeRuntimeUnavailableError()
        presets = []
        for preset_id in (
            "ecology-coordinator-v1",
            "ecology-researcher-v1",
            "ecology-candidate-proposer-v1",
            "ecology-sample-planner-v1",
            "ecology-sample-critic-v1",
            "ecology-generation-judge-v1",
        ):
            presets.append(
                {
                    "preset_id": preset_id,
                    "declared": True,
                    "standing_key": f"standing:{preset_id}",
                    "preset_mountable": True,
                    "tool_surface_verified": True,
                    "route_resolvable": True,
                    "live_agent_service_ready": self.live,
                    "first_call_verified": False,
                }
            )
        return {
            "schema_version": "ecology-agent-runtime-capabilities/1",
            "ready": True,
            "root_services": {"required": ["agents"], "missing": [], "declared": True},
            "presets": presets,
            "live_agent_service_ready": self.live,
            "first_call_verified": False,
        }

    def require_capabilities(self, payload: dict, required: tuple[str, ...], *, require_live: bool = True) -> None:
        if require_live and not payload["live_agent_service_ready"]:
            raise DshNativeRuntimeUnavailableError()
        present = {item["preset_id"] for item in payload["presets"]}
        if not set(required).issubset(present):
            raise DshNativeRuntimeUnavailableError()

    def create_run(self, request: dict) -> dict:
        self.created.append(request)
        self.run_ids.add(request["run_id"])
        self.live = True
        return {"accepted": True, **{key: request[key] for key in (
            "run_id", "run_state_revision", "stage_attempt", "ledger_expected_revision", "idempotency_key"
        )}}

    def status(self, run_id: str) -> dict:
        if run_id not in self.run_ids:
            raise DshNativeRuntimeUnavailableError(
                "runtime run missing after restart",
                error_code="dsh_native_runtime_http_error",
                status_code=404,
            )
        return {"run_id": run_id, "status": "running"}

    def cancel(self, request: dict) -> dict:
        self.cancelled.append(request)
        return {"accepted": True, **request}


class DshNativeHTTPGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.server = EvolutionHTTPServer(
            ("127.0.0.1", 0), Path(self.directory.name) / "events.sqlite3"
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.directory.cleanup()

    def _post(self, body: dict) -> tuple[int, dict]:
        request = Request(
            self.base + "/runs",
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def _get(self, path: str) -> tuple[int, dict]:
        request = Request(self.base + path, method="GET")
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_catalog_reports_native_dsh_agent_execution_when_runtime_is_bound(self) -> None:
        self.server.dsh_native_runtime = _FakeNativeRuntime()
        status, payload = self._get("/catalog")
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["dsh"]["harness_execution"], "dsh_native_agent_runtime")
        self.assertTrue(payload["dsh"]["official_harness_agent_loop"])

    def test_native_setup_precedes_run_created_and_emits_no_token_budget(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        self.server.model_gateway.catalog = lambda: (_ for _ in ()).throw(AssertionError("legacy gateway used"))  # type: ignore[method-assign]
        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": "run:native-test",
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": False,
                "auto_advance": 0,
                "idempotency_key": "native-create-1",
                "budget": {"max_generations": 1, "token_limit": 12345},
            }
        )
        self.assertEqual(status, 201, payload)
        self.assertEqual(len(runtime.created), 1)
        state = self.server.director.state("run:native-test")
        self.assertEqual(
            state.task_manifest.metadata["execution_protocol"],
            DSH_NATIVE_EXECUTION_PROTOCOL,
        )
        self.assertNotIn("token_limit", state.task_manifest.budget)
        self.assertNotIn("token_reservation_per_wave", state.task_manifest.budget)
        self.assertFalse(state.task_manifest.metadata["dsh_first_call_verified"])
        self.server.validate_frozen_runtime_bindings(state.task_manifest)

    def test_frozen_native_run_is_recreated_after_dsh_process_restart(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": "run:native-restart",
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": False,
                "auto_advance": 0,
                "idempotency_key": "native-create-restart",
            }
        )
        self.assertEqual(status, 201, payload)
        state = self.server.director.state("run:native-restart")
        runtime.live = False
        runtime.run_ids.clear()

        self.server.validate_frozen_runtime_bindings(
            state.task_manifest,
            run_id=state.run.run_id,
        )

        self.assertTrue(runtime.live)
        self.assertEqual(runtime.run_ids, {state.run.run_id})
        self.assertEqual(len(runtime.created), 2)
        restored = runtime.created[-1]
        self.assertEqual(
            restored["idempotency_key"],
            f"runtime-restore:{state.run.run_id}",
        )
        self.assertEqual(
            restored["binding"]["task_manifest_digest"],
            state.task_manifest.digest,
        )

    def test_unavailable_native_runtime_returns_503_without_scientific_run(self) -> None:
        self.server.dsh_native_runtime = _FakeNativeRuntime(unavailable=True)
        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": "run:no-runtime",
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": False,
                "auto_advance": 0,
                "idempotency_key": "native-create-unavailable",
            }
        )
        self.assertEqual(status, 503, payload)
        self.assertEqual(payload["error_code"], "dsh_native_runtime_unavailable")
        self.assertEqual(self.server.ledger.run_ids(), ())
