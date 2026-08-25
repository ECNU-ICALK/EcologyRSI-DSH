from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecologyrsi_dsh.core.errors import (
    DshNativeRuntimeUnavailableError,
    FrozenRuntimeBindingDriftError,
)
from ecologyrsi_dsh.api import generation_execution as generation_execution_module
from ecologyrsi_dsh.api.dsh_tools import DshToolAdmissionClosedError
from ecologyrsi_dsh.core.models import TaskManifest, digest
from ecologyrsi_dsh.integrations.dsh_native_runtime import (
    DSH_NATIVE_EXECUTION_PROTOCOL,
    DshNativeAgentRuntimeClient,
)
from ecologyrsi_dsh.api.handler import EvolutionHTTPServer


def _research_skill_evidence() -> dict:
    return {
        "schema_version": "ecologyrsi-dsh.skill-invocation-evidence/1",
        "stage": "generation.research",
        "skill_name": "autonomous-ecology-research",
        "call_count": 1,
        "successful_call_count": 1,
        "call_seq": 1,
        "result_seq": 2,
        "first_tool_call_verified": True,
        "next_tool_name": "structured_output",
        "next_tool_call_seq": 3,
        "order_verified": True,
        "source": "dsh_session_event_log",
    }


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
                    "preset_id": "ecology-researcher-v6",
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
        self.client.require_capabilities(capability, ["ecology-researcher-v6"])
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

    def test_default_timeouts_allow_dsh_managed_long_context_turns(self) -> None:
        client = DshNativeAgentRuntimeClient(
            "http://127.0.0.1:8848",
            token="runtime-secret",
        )
        # Ordinary control requests remain bounded, while a structured stage
        # can span two DSH-owned 30-minute research attempts plus cleanup.
        self.assertEqual(client.timeout, 660.0)
        self.assertEqual(client.stage_timeout, 3_720.0)

    def test_explicit_stage_timeout_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            DshNativeAgentRuntimeClient(
                "http://127.0.0.1:8848",
                token="runtime-secret",
                stage_timeout=0,
            )

    def test_run_stage_uses_the_long_stage_timeout(self) -> None:
        request = {
            "run_id": "run-1",
            "run_state_revision": 7,
            "stage_attempt": 2,
            "ledger_expected_revision": 11,
            "idempotency_key": "idem-1",
        }
        response = {
            **self._accepted(),
            "structured": {"ok": True},
            "result_digest": "a" * 64,
            "session_id": "session-1",
            "first_call_verified": True,
        }
        with patch.object(
            self.client, "_request", return_value=response
        ) as mocked_request:
            self.client.run_stage(request)
        self.assertEqual(
            mocked_request.call_args.kwargs["timeout"],
            self.client.stage_timeout,
        )

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
                self._capabilities(ready=False), ["ecology-researcher-v6"]
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
        self.paused: list[dict] = []
        self.resumed: list[dict] = []
        self.cancelled: list[dict] = []
        self.live = False
        self.run_ids: set[str] = set()

    def capabilities(self) -> dict:
        if self.unavailable:
            raise DshNativeRuntimeUnavailableError()
        presets = []
        for preset_id in (
            "ecology-coordinator-v3",
            "ecology-researcher-v6",
            "ecology-candidate-proposer-v3",
            "ecology-sample-planner-v3",
            "ecology-sample-critic-v3",
            "ecology-generation-judge-v6",
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

    def pause(self, request: dict) -> dict:
        if request["run_id"] not in self.run_ids:
            raise DshNativeRuntimeUnavailableError(
                "runtime run missing after restart",
                error_code="dsh_native_runtime_http_error",
                status_code=404,
            )
        self.paused.append(request)
        return {"accepted": True, **request}

    def resume(self, request: dict) -> dict:
        if request["run_id"] not in self.run_ids:
            raise DshNativeRuntimeUnavailableError(
                "runtime run missing after restart",
                error_code="dsh_native_runtime_http_error",
                status_code=404,
            )
        self.resumed.append(request)
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
        return self._post_path("/runs", body)

    def _post_path(self, path: str, body: dict) -> tuple[int, dict]:
        request = Request(
            self.base + path,
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
                "candidate_concurrency": 1,
                "sample_concurrency": 2,
                "samples_per_update": 169,
                "sample_agent_batch_size": 9,
                "start": False,
                "auto_advance": 0,
                "idempotency_key": "native-create-1",
                "budget": {"max_generations": 1, "token_limit": 12345},
            }
        )
        self.assertEqual(status, 201, payload)
        self.assertEqual(len(runtime.created), 1)
        self.assertEqual(
            runtime.created[0]["binding"]["initial_run_status"], "created"
        )
        state = self.server.director.state("run:native-test")
        self.assertEqual(
            state.task_manifest.metadata["execution_protocol"],
            DSH_NATIVE_EXECUTION_PROTOCOL,
        )
        self.assertNotIn("token_limit", state.task_manifest.budget)
        self.assertNotIn("token_reservation_per_wave", state.task_manifest.budget)
        self.assertFalse(state.task_manifest.metadata["dsh_first_call_verified"])
        self.assertEqual(state.task_manifest.metadata["candidate_concurrency"], 1)
        self.assertEqual(state.task_manifest.metadata["sample_concurrency"], 2)
        self.assertEqual(state.task_manifest.metadata["samples_per_update"], 169)
        self.assertEqual(state.task_manifest.metadata["sample_agent_batch_size"], 9)
        self.assertEqual(
            state.task_manifest.metadata["sample_budget_class"],
            "selection_eligible",
        )
        self.assertEqual(
            state.task_manifest.metadata["sample_remote_critic_policy"],
            {"version": "always@1"},
        )
        self.assertEqual(
            state.task_manifest.metadata["sample_agent_protocol"],
            "dsh-strict-origin-bundle@3",
        )
        self.assertEqual(state.task_manifest.metadata["sample_prompt_batch_size"], 1)
        self.assertEqual(
            state.task_manifest.metadata["sample_reflection_policy"],
            "always_remote_post_score@1",
        )
        self.assertFalse(state.task_manifest.metadata["allow_host_route_bypass"])
        self.assertFalse(
            state.task_manifest.metadata["allow_host_prediction_fallback"]
        )
        self.assertEqual(
            state.task_manifest.metadata["sample_planner_prompt_profile"],
            {"version": "origin_shared_context@1"},
        )
        self.assertEqual(
            state.task_manifest.metadata["fitness_profile_digest"],
            digest(state.task_manifest.metadata["fitness_profile"]),
        )
        self.assertEqual(
            state.task_manifest.metadata["minimum_selection_samples_per_update"],
            169,
        )
        self.server.validate_frozen_runtime_bindings(state.task_manifest)

        tampered_data = state.task_manifest.to_dict()
        tampered_data["metadata"]["fitness_profile_digest"] = "0" * 64
        with self.assertRaises(FrozenRuntimeBindingDriftError):
            self.server.validate_frozen_runtime_bindings(
                TaskManifest.from_dict(tampered_data)
            )

    def test_native_diagnostic_smoke_accepts_one_complete_origin(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime

        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": "run:native-diagnostic",
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "candidate_concurrency": 1,
                "sample_concurrency": 1,
                "samples_per_update": 1,
                "sample_agent_batch_size": 1,
                "start": False,
                "auto_advance": 0,
                "idempotency_key": "native-diagnostic-1",
            }
        )

        self.assertEqual(status, 201, payload)
        configuration = payload["projection"]["configuration"]
        self.assertEqual(configuration["samples_per_update"], 1)
        self.assertEqual(configuration["sample_budget_class"], "diagnostic_smoke")
        self.assertEqual(configuration["minimum_selection_samples_per_update"], 169)

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
        self.assertEqual(restored["binding"]["initial_run_status"], "created")

    def test_resume_restores_a_paused_native_run_after_dsh_restart(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        run_id = "run:native-resume-restart"
        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": run_id,
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": True,
                "auto_advance": 0,
                "idempotency_key": "native-create-resume-restart",
            }
        )
        self.assertEqual(status, 201, payload)
        status, payload = self._post_path(
            f"/runs/{run_id}/control",
            {"action": "pause", "idempotency_key": "pause-before-restart"},
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["projection"]["status"], "paused")

        runtime.live = False
        runtime.run_ids.clear()
        status, payload = self._post_path(
            f"/runs/{run_id}/control",
            {"action": "resume", "idempotency_key": "resume-after-restart"},
        )

        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["projection"]["status"], "running")
        self.assertEqual(runtime.run_ids, {run_id})
        self.assertEqual(runtime.created[-1]["idempotency_key"], f"runtime-restore:{run_id}")
        self.assertEqual(runtime.resumed[-1]["run_id"], run_id)

    def test_started_native_run_opens_runtime_admission_at_creation(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": "run:native-running",
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": True,
                "auto_advance": 0,
                "idempotency_key": "native-create-running",
            }
        )
        self.assertEqual(status, 201, payload)
        self.assertEqual(
            runtime.created[0]["binding"]["initial_run_status"], "running"
        )

    def test_permanent_proposal_failure_quiesces_native_runtime(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        run_id = "run:native-proposal-terminal-cleanup"
        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": run_id,
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": True,
                "auto_advance": 0,
                "idempotency_key": "native-proposal-terminal-cleanup-create",
            }
        )
        self.assertEqual(status, 201, payload)
        endpoint = SimpleNamespace(server=self.server)

        with (
            patch.object(
                generation_execution_module,
                "start_generation_batch",
                return_value=SimpleNamespace(generation=0),
            ),
            patch.object(
                generation_execution_module,
                "_spawn_generation_candidates",
                side_effect=ValueError("invalid mutation"),
            ),
        ):
            state = generation_execution_module.execute_generation(endpoint, run_id)

        self.assertEqual(state.run.status.value, "failed")
        self.assertEqual(len(runtime.cancelled), 1)
        self.assertEqual(runtime.cancelled[0]["run_id"], run_id)

    def test_native_pause_persists_run_boundary_before_runtime_quiescence(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        run_id = "run:native-pause-boundary"
        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": run_id,
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": True,
                "auto_advance": 0,
                "idempotency_key": "native-pause-boundary-create",
            }
        )
        self.assertEqual(status, 201, payload)
        observed_statuses: list[str] = []
        original_pause = runtime.pause

        def observe_pause(request: dict) -> dict:
            observed_statuses.append(
                self.server.director.state(run_id).run.status.value
            )
            return original_pause(request)

        runtime.pause = observe_pause  # type: ignore[method-assign]
        status, payload = self._post_path(
            f"/runs/{run_id}/control",
            {
                "action": "pause",
                "reason": "pause boundary regression",
                "code": "pause_boundary_regression",
                "idempotency_key": "native-pause-boundary-control",
            },
        )

        self.assertEqual(status, 200, payload)
        self.assertEqual(observed_statuses, ["paused"])
        self.assertEqual(payload["projection"]["status"], "paused")

    def test_native_pause_response_waits_for_host_generation_quiescence(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        run_id = "run:native-pause-host-barrier"
        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": run_id,
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": True,
                "auto_advance": 0,
                "idempotency_key": "native-pause-host-barrier-create",
            }
        )
        self.assertEqual(status, 201, payload)
        active_generation = self.server.acquire_generation_lease(run_id)
        self.assertIsNotNone(active_generation)
        runtime_quiesced = threading.Event()
        original_pause = runtime.pause

        def observe_pause(request: dict) -> dict:
            result = original_pause(request)
            runtime_quiesced.set()
            return result

        runtime.pause = observe_pause  # type: ignore[method-assign]
        response: list[tuple[int, dict]] = []
        request_thread = threading.Thread(
            target=lambda: response.append(
                self._post_path(
                    f"/runs/{run_id}/control",
                    {
                        "action": "pause",
                        "idempotency_key": "native-pause-host-barrier-control",
                    },
                )
            )
        )
        request_thread.start()
        try:
            self.assertTrue(runtime_quiesced.wait(2))
            request_thread.join(0.1)
            self.assertTrue(
                request_thread.is_alive(),
                "pause returned before the active Host generation drained",
            )
        finally:
            active_generation.release()
        request_thread.join(2)

        self.assertFalse(request_thread.is_alive())
        self.assertEqual(response[0][0], 200, response)
        self.assertEqual(response[0][1]["projection"]["status"], "paused")

    def test_native_cancel_is_durable_before_runtime_and_host_quiescence(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        run_id = "run:native-cancel-host-boundary"
        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": run_id,
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": True,
                "auto_advance": 0,
                "idempotency_key": "native-cancel-host-boundary-create",
            }
        )
        self.assertEqual(status, 201, payload)
        state = self.server.director.state(run_id)
        run_state_revision = state.events[-1].seq
        self.server.dsh_tools.open_admission(run_id, run_state_revision, 1)
        active_generation = self.server.acquire_generation_lease(run_id)
        self.assertIsNotNone(active_generation)

        runtime_cancel_entered = threading.Event()
        runtime_cancel_finished = threading.Event()
        release_runtime_cancel = threading.Event()
        original_cancel = runtime.cancel

        def blocking_cancel(request: dict) -> dict:
            runtime_cancel_entered.set()
            if not release_runtime_cancel.wait(2):
                raise RuntimeError("test did not release runtime cancellation")
            result = original_cancel(request)
            runtime_cancel_finished.set()
            return result

        runtime.cancel = blocking_cancel  # type: ignore[method-assign]
        response: list[tuple[int, dict]] = []
        request_thread = threading.Thread(
            target=lambda: response.append(
                self._post_path(
                    f"/runs/{run_id}/control",
                    {
                        "action": "cancel",
                        "reason": "cancel boundary regression",
                        "idempotency_key": "native-cancel-host-boundary-control",
                    },
                )
            )
        )
        request_thread.start()
        try:
            self.assertTrue(runtime_cancel_entered.wait(2))
            self.assertEqual(
                self.server.director.state(run_id).run.status.value,
                "cancelled",
            )
            structured = {
                "schema_version": "ecology-research-result@1",
                "summary": "late result",
            }
            with self.assertRaises(DshToolAdmissionClosedError):
                self.server.dsh_tools.accept_structured(
                    {
                        "identity": {
                            "run_id": run_id,
                            "role": "researcher",
                            "stage": "generation.research",
                            "run_state_revision": run_state_revision,
                            "stage_attempt": 1,
                            "ledger_expected_revision": run_state_revision,
                            "session_id": "session:late-result",
                            "idempotency_key": "late-result",
                            "child_reservation_id": "reservation:late-result",
                            "activation_lease_id": "lease:late-result",
                            "genome_digest": "a" * 64,
                            "compiled_behavior_digest": "b" * 64,
                            "phenotype_instance_digest": "c" * 64,
                        },
                        "output_schema_id": "ecology-research-result@1",
                        "structured": structured,
                        "result_digest": digest(structured),
                        "skill_invocation_evidence": _research_skill_evidence(),
                    }
                )
            release_runtime_cancel.set()
            self.assertTrue(runtime_cancel_finished.wait(2))
            request_thread.join(0.1)
            self.assertTrue(
                request_thread.is_alive(),
                "cancel returned before the active Host generation drained",
            )
        finally:
            release_runtime_cancel.set()
            active_generation.release()
        request_thread.join(2)

        self.assertFalse(request_thread.is_alive())
        self.assertEqual(response[0][0], 200, response)
        self.assertEqual(response[0][1]["projection"]["status"], "cancelled")

    def test_native_cancel_runtime_failure_preserves_terminal_boundary_for_retry(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        run_id = "run:native-cancel-runtime-failure"
        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": run_id,
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": True,
                "auto_advance": 0,
                "idempotency_key": "native-cancel-runtime-failure-create",
            }
        )
        self.assertEqual(status, 201, payload)
        original_cancel = runtime.cancel
        observed_statuses: list[str] = []

        def flaky_cancel(request: dict) -> dict:
            observed_statuses.append(
                self.server.director.state(run_id).run.status.value
            )
            if len(observed_statuses) == 1:
                raise DshNativeRuntimeUnavailableError("runtime unavailable")
            return original_cancel(request)

        runtime.cancel = flaky_cancel  # type: ignore[method-assign]
        control = {
            "action": "cancel",
            "idempotency_key": "native-cancel-runtime-failure-control",
        }
        status, failed = self._post_path(f"/runs/{run_id}/control", control)

        self.assertEqual(status, 503, failed)
        self.assertEqual(
            self.server.director.state(run_id).run.status.value,
            "cancelled",
        )
        self.assertEqual(failed["command_status"], "等待恢复")

        status, recovered = self._post_path(f"/runs/{run_id}/control", control)

        self.assertEqual(status, 200, recovered)
        self.assertEqual(recovered["projection"]["status"], "cancelled")
        self.assertEqual(observed_statuses, ["cancelled", "cancelled"])
        self.assertEqual(
            sum(
                event.kind == "RunCancelled"
                for event in self.server.ledger.events(run_id)
            ),
            1,
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
