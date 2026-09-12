from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecologyrsi_dsh.core.errors import (
    DshNativeRuntimeUnavailableError,
    FrozenRuntimeBindingDriftError,
    dsh_native_runtime_retryable,
)
from ecologyrsi_dsh.application import generation_execution as generation_execution_module
from ecologyrsi_dsh.integrations.dsh_tools import DshToolAdmissionClosedError
from ecologyrsi_dsh.core.models import TaskManifest, canonical_json, digest
from ecologyrsi_dsh.core.state import project_run_state
from ecologyrsi_dsh.integrations.dsh_native_runtime import (
    DSH_NATIVE_EXECUTION_PROTOCOL,
    DshNativeAgentRuntimeClient,
    configured_stage_timeout,
)
from ecologyrsi_dsh.api.handler import EvolutionHTTPServer
from ecologyrsi_dsh.presentation.reporting import run_summary
from ecologyrsi_dsh.presentation.training_assets import training_assets
from ecologyrsi_dsh.version import __version__


class _RealNodeRuntime:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> DshNativeAgentRuntimeClient:
        runtime_server = (
            Path(__file__).resolve().parents[1]
            / "integrations"
            / "dsh_ecology_plugin"
            / "test"
            / "fixtures"
            / "runtime_handshake_server.mjs"
        )
        self.process = subprocess.Popen(
            ["node", str(runtime_server)],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=10):
                self.close()
                raise RuntimeError("real Node runtime did not publish its loopback port")
            address_line = self.process.stdout.readline()
        if not address_line:
            assert self.process.stderr is not None
            detail = self.process.stderr.read()
            self.close()
            raise RuntimeError(detail or "real Node runtime exited before startup")
        address = json.loads(address_line)
        return DshNativeAgentRuntimeClient(
            f"http://127.0.0.1:{address['port']}",
            token="runtime-secret",
            timeout=2,
        )

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self.process is None:
            return
        process, self.process = self.process, None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


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
                    "preset_id": "ecology-researcher-v12",
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
        self.client.require_capabilities(capability, ["ecology-researcher-v12"])
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

    def test_python_resume_handshake_accepts_real_node_restored_paused_hosts(self) -> None:
        preset_ids = (
            "ecology-coordinator-v5",
            "ecology-researcher-v12",
            "ecology-candidate-proposer-v4",
            "ecology-sample-planner-v9",
            "ecology-sample-critic-v5",
            "ecology-generation-judge-v8",
        )
        with _RealNodeRuntime() as client:
            cold = client.capabilities()
            client.require_capabilities(cold, preset_ids, require_live=False)
            self.assertFalse(cold["live_agent_service_ready"])
            run_id = "run:python-node-restored-paused"
            client.create_run(
                {
                    "run_id": run_id,
                    "run_state_revision": 7,
                    "stage_attempt": 0,
                    "ledger_expected_revision": 11,
                    "idempotency_key": f"runtime-restore:{run_id}",
                    "binding": {
                        "initial_run_status": "paused",
                        "restore_provenance": {
                            "source": "python_durable_ledger",
                            "status": "paused",
                        },
                        "strategy_model_id": "provider/model",
                        "review_model_id": "provider/model",
                    },
                }
            )
            live = client.capabilities()
            client.require_capabilities(live, preset_ids, require_live=True)
            resumed = client.resume(
                {
                    "run_id": run_id,
                    "run_state_revision": 8,
                    "stage_attempt": 0,
                    "ledger_expected_revision": 12,
                    "idempotency_key": "resume:python-node-restored-paused",
                }
            )
            self.assertEqual(resumed["run_id"], run_id)
            self.assertEqual(client.status(run_id)["status"], "running")

    def test_default_timeouts_allow_dsh_managed_long_context_turns(self) -> None:
        client = DshNativeAgentRuntimeClient(
            "http://127.0.0.1:8848",
            token="runtime-secret",
        )
        # Ordinary control requests remain bounded, while a structured stage
        # can span two DSH-owned 30-minute research attempts plus cleanup.
        self.assertEqual(client.timeout, 660.0)
        self.assertEqual(client.stage_timeout, 3_720.0)

    def test_stage_timeout_can_be_overridden_for_local_deployment(self) -> None:
        with patch.dict(os.environ, {"ECOLOGYRSI_DSH_STAGE_TIMEOUT": "900"}):
            self.assertEqual(configured_stage_timeout(), 900.0)

    def test_http_server_applies_stage_timeout_override_to_native_client(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ECOLOGYRSI_DSH_RUNTIME_URL": "http://127.0.0.1:8848",
                "ECOLOGYRSI_DSH_RUNTIME_TOKEN": "runtime-secret",
                "ECOLOGYRSI_DSH_STAGE_TIMEOUT": "900",
            },
        ):
            with tempfile.TemporaryDirectory() as directory:
                server = EvolutionHTTPServer(
                    ("127.0.0.1", 0), Path(directory) / "events.sqlite3"
                )
                try:
                    self.assertIsNotNone(server.dsh_native_runtime)
                    self.assertEqual(server.dsh_native_runtime.stage_timeout, 900.0)
                finally:
                    server.close()

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
            "admission_id": "admission-runtime-client-1",
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
                self._capabilities(ready=False), ["ecology-researcher-v12"]
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

    def test_bounded_sample_http_failure_is_not_a_runtime_retry_boundary(self) -> None:
        self.server.responses.append(  # type: ignore[attr-defined]
            (422, {"error_code": "structured_child_model_error"})
        )
        with self.assertRaises(DshNativeRuntimeUnavailableError) as raised:
            self.client.status("run-1")

        self.assertEqual(
            raised.exception.error_code, "structured_child_model_error"
        )
        self.assertEqual(raised.exception.status_code, 422)
        self.assertFalse(dsh_native_runtime_retryable(raised.exception))

    def test_provider_contract_preserves_status_and_retry_after(self) -> None:
        for status in (429, 503):
            self.server.responses.append((status, {
                "schema_version": "ecology-runtime-failure/1",
                "error_code": "structured_child_model_error",
                "failure_domain": "provider", "provider_status": status,
                "retry_after_ms": 17000,
            }))
            with self.assertRaises(DshNativeRuntimeUnavailableError) as raised:
                self.client.status("run-1")
            self.assertTrue(dsh_native_runtime_retryable(raised.exception))
            self.assertEqual(raised.exception.provider_status, status)
            self.assertEqual(raised.exception.retry_after_seconds, 17)

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
        self.activated: list[dict] = []
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
            "ecology-coordinator-v5",
            "ecology-researcher-v12",
            "ecology-candidate-proposer-v4",
            "ecology-sample-planner-v9",
            "ecology-sample-critic-v5",
            "ecology-generation-judge-v8",
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

    def activate(self, request: dict) -> dict:
        if request["run_id"] not in self.run_ids:
            raise DshNativeRuntimeUnavailableError(
                "runtime run missing after restart",
                error_code="dsh_native_runtime_http_error",
                status_code=404,
            )
        self.activated.append(request)
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

    def test_native_create_freezes_top2_adaptive_epoch_schedule(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        schedule = {
            "schema_version": "ecologyrsi-dsh.top2-adaptive-epoch-schedule/1",
            "screening_origin_count": 64,
            "finalist_count": 2,
            "formal_origin_count_per_finalist": 500,
            "local_batch_origin_count": 50,
            "max_local_edits_per_batch": 2,
            "selection_holdout_origin_count": 169,
            "local_evaluation_mode": "prequential",
        }

        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "optimization_protocol": "top2_adaptive_epoch@1",
                "optimization_schedule": schedule,
                "run_id": "run:native-adaptive-schedule",
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "candidate_concurrency": 4,
                "sample_concurrency": 64,
                "start": False,
                "auto_advance": 0,
                "idempotency_key": "native-adaptive-schedule-1",
                "budget": {
                    "max_generations": 1,
                    "candidates_per_generation": 4,
                    "max_candidates": 4,
                },
            }
        )

        self.assertEqual(status, 201, payload)
        self.assertEqual(
            payload["projection"]["configuration"]["optimization_schedule"],
            schedule,
        )
        self.assertEqual(
            payload["projection"]["optimization_protocol"],
            "top2_adaptive_epoch@1",
        )
        state = self.server.director.state("run:native-adaptive-schedule")
        metadata = state.task_manifest.metadata
        self.assertEqual(metadata["optimization_protocol"], "top2_adaptive_epoch@1")
        self.assertEqual(metadata["optimization_schedule"], schedule)
        self.assertEqual(
            metadata["derived_execution_budget"],
            {
                "screening_candidate_origins": 256,
                "formal_candidate_origins": 1000,
                "holdout_candidate_origins": 1014,
                "total_candidate_origins": 2270,
                "total_scoring_cells": 2270,
            },
        )
        self.assertNotIn("samples_per_update", metadata)
        self.assertNotIn("token_limit", state.task_manifest.budget)
        self.assertNotIn("dsh_provider_token_budget_policy", metadata)
        self.assertNotIn("token_budget_scope", metadata)

    def test_native_setup_precedes_run_created_without_hidden_provider_budget(self) -> None:
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
                "sample_agent_batch_size": 1,
                "start": False,
                "auto_advance": 0,
                "idempotency_key": "native-create-1",
                "budget": {"max_generations": 1},
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
        self.assertNotIn("dsh_provider_token_budget_policy", state.task_manifest.metadata)
        self.assertNotIn("token_budget_scope", state.task_manifest.metadata)
        self.assertFalse(state.task_manifest.metadata["dsh_first_call_verified"])
        self.assertEqual(state.task_manifest.metadata["candidate_concurrency"], 1)
        self.assertEqual(state.task_manifest.metadata["sample_concurrency"], 2)
        self.assertNotIn("samples_per_update", state.task_manifest.metadata)
        self.assertEqual(state.task_manifest.metadata["sample_agent_batch_size"], 1)
        self.assertEqual(
            state.task_manifest.metadata["sample_budget_class"],
            "selection_eligible",
        )
        self.assertEqual(
            state.task_manifest.metadata["sample_remote_critic_policy"],
            {
                "version": "uncertain_or_failure@1",
                "min_planner_confidence": 0.5,
            },
        )
        self.assertEqual(
            state.task_manifest.metadata["sample_agent_protocol"],
            "dsh-strict-origin-bundle@4",
        )
        self.assertEqual(state.task_manifest.metadata["sample_prompt_batch_size"], 1)
        self.assertEqual(
            state.task_manifest.metadata["sample_reflection_policy"],
            "candidate_aggregate_post_score@1",
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
            40,
        )
        self.assertEqual(
            state.task_manifest.metadata["host_runtime_build"],
            {
                "package_version": __version__,
                "evolution_runtime_schema": (
                    "ecologyrsi-dsh.evolution-runtime/3"
                ),
                "generation_comparison_schema": (
                    "ecologyrsi-dsh.generation-comparison/1"
                ),
                "projection_schema": "ecologyrsi-dsh.execution-projection/2",
            },
        )
        self.server.validate_frozen_runtime_bindings(state.task_manifest)

        runtime_v2_data = state.task_manifest.to_dict()
        runtime_v2_data["metadata"]["host_runtime_build"][
            "evolution_runtime_schema"
        ] = "ecologyrsi-dsh.evolution-runtime/2"
        self.server.validate_frozen_runtime_bindings(
            TaskManifest.from_dict(runtime_v2_data)
        )

        tampered_data = state.task_manifest.to_dict()
        tampered_data["metadata"]["fitness_profile_digest"] = "0" * 64
        with self.assertRaises(FrozenRuntimeBindingDriftError):
            self.server.validate_frozen_runtime_bindings(
                TaskManifest.from_dict(tampered_data)
            )

        incompatible_data = state.task_manifest.to_dict()
        incompatible_data["metadata"]["host_runtime_build"][
            "evolution_runtime_schema"
        ] = "ecologyrsi-dsh.evolution-runtime/999"
        with self.assertRaises(FrozenRuntimeBindingDriftError):
            self.server.validate_frozen_runtime_bindings(
                TaskManifest.from_dict(incompatible_data)
            )

    def test_native_provider_token_limit_is_rejected_before_runtime_creation(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        status, payload = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": "run:native-token-limit",
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": False,
                "auto_advance": 0,
                "idempotency_key": "native-token-limit-1",
                "budget": {"max_generations": 1, "token_limit": 12345},
            }
        )
        self.assertEqual(status, 400, payload)
        self.assertIn("telemetry", payload["error"])
        self.assertEqual(runtime.created, [])

    def test_native_schedule_replaces_diagnostic_sample_count(self) -> None:
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
                "sample_agent_batch_size": 1,
                "start": False,
                "auto_advance": 0,
                "idempotency_key": "native-diagnostic-1",
            }
        )

        self.assertEqual(status, 201, payload)
        state = self.server.director.state("run:native-diagnostic")
        self.assertNotIn("samples_per_update", state.task_manifest.metadata)
        self.assertEqual(
            state.task_manifest.metadata["optimization_schedule"][
                "formal_origin_count_per_finalist"
            ],
            100,
        )
        self.assertEqual(
            state.task_manifest.metadata["sample_budget_class"],
            "selection_eligible",
        )

    def test_native_start_activates_created_runtime_before_host_run(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        run_id = "run:native-explicit-start"
        status, created = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": run_id,
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": False,
                "auto_advance": 0,
                "idempotency_key": "native-explicit-start-create",
            }
        )
        self.assertEqual(status, 201, created)

        status, started = self._post_path(
            f"/runs/{run_id}/action",
            {"action": "start", "idempotency_key": "native-explicit-start-action"},
        )

        self.assertEqual(status, 200, started)
        self.assertEqual(started["projection"]["status"], "running")
        self.assertEqual(len(runtime.activated), 1)
        self.assertEqual(runtime.activated[0]["run_id"], run_id)

    def test_seed_incumbent_control_is_durable_distinct_and_budget_neutral(
        self,
    ) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        run_id = "run:native-seed-incumbent"
        status, created = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": run_id,
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": True,
                "auto_advance": 0,
                "idempotency_key": "native-seed-incumbent-create",
                "budget": {
                    "max_generations": 1,
                    "candidates_per_generation": 4,
                    "max_candidates": 4,
                },
            }
        )
        self.assertEqual(status, 201, created)

        first = self.server.director.ensure_seed_incumbent_control(run_id)
        recovered = self.server.director.ensure_seed_incumbent_control(run_id)

        self.assertEqual(recovered, first)
        self.assertEqual(first.role.value, "incumbent_control")
        state = self.server.director.replay(run_id)
        seed = state.materialized_seed_genome()
        revision = state.initial_revision_for(first.candidate_id)
        self.assertIsNotNone(revision)
        self.assertEqual(revision.genome_digest, seed.genome_digest)
        self.assertEqual(
            canonical_json(revision.identity_dict()["genome"]),
            canonical_json(seed.to_dict()),
        )
        self.assertEqual(
            [
                item.candidate_id
                for item in state.candidates
                if item.role.value == "incumbent_control"
            ],
            [first.candidate_id],
        )
        self.assertEqual(
            [item for item in state.candidates if item.role.value == "search"],
            [],
        )
        self.assertEqual(
            sum(event.kind == "ProposalSubmitted" for event in state.events),
            1,
        )
        self.assertEqual(
            sum(event.kind == "CandidateSpawned" for event in state.events),
            1,
        )
        self.assertEqual(
            sum(event.kind == "CandidateRevisionCreated" for event in state.events),
            1,
        )
        proposal = state.proposal(first.proposal_id)
        with self.assertRaisesRegex(ValueError, "deterministic seed control"):
            self.server.director.spawn_candidate(
                run_id,
                proposal,
                candidate_id="candidate:forged-seed-control",
                role=first.role,
            )
        forged_revision_data = revision.to_dict()
        forged_revision_data["revision_id"] = "revision:forged-seed-control:r0"
        forged_revision_data.pop("revision_digest", None)
        with self.assertRaisesRegex(ValueError, "deterministic seed control R0"):
            self.server.director.create_candidate_revision(
                run_id,
                type(revision).from_dict(forged_revision_data),
            )

        spawn_event = next(
            event
            for event in state.events
            if event.kind == "CandidateSpawned"
            and event.payload["candidate"]["candidate_id"] == first.candidate_id
        )
        forged_candidate_event = replace(
            state.events[-1],
            seq=state.events[-1].seq + 1,
            event_id="event:forged-seed-control",
            kind="CandidateSpawned",
            payload={
                **spawn_event.payload,
                "candidate": replace(
                    first,
                    candidate_id="candidate:forged-seed-control",
                ).to_dict(),
            },
        )
        with self.assertRaisesRegex(ValueError, "deterministic seed control"):
            project_run_state((*state.events, forged_candidate_event))

        forged_revision_event = replace(
            state.events[-1],
            seq=state.events[-1].seq + 1,
            event_id="event:forged-seed-control-r0",
            kind="CandidateRevisionCreated",
            payload={
                "revision": type(revision)
                .from_dict(forged_revision_data)
                .to_dict()
            },
        )
        with self.assertRaisesRegex(ValueError, "deterministic seed control R0"):
            project_run_state((*state.events, forged_revision_event))
        status, projected = self._get(f"/runs/{run_id}")
        self.assertEqual(status, 200, projected)
        projection = projected["projection"]
        self.assertEqual(projection["candidates_count"], 0)
        self.assertEqual(projection["candidates"], [])
        self.assertEqual(projection["rounds"][0]["candidate_count"], 0)
        self.assertEqual(projection["rounds"][0]["candidates"], [])
        self.assertEqual(training_assets(state), [])
        summary = run_summary(state)
        self.assertEqual(summary["candidate_count"], 0)
        self.assertEqual(summary["training_asset_count"], 0)

        with self.assertRaisesRegex(
            RuntimeError,
            "global incumbent decision",
        ):
            self.server.director.advance_generation(run_id)

        self.server.ledger.append(
            run_id,
            "GenerationAdvanced",
            {"generation": 1},
        )
        with self.assertRaisesRegex(
            ValueError,
            "global incumbent decision",
        ):
            self.server.director.replay(run_id)

    def test_native_start_from_paused_resumes_instead_of_reactivating(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        run_id = "run:native-start-from-paused"
        status, created = self._post(
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": run_id,
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": True,
                "auto_advance": 0,
                "idempotency_key": "native-start-from-paused-create",
            }
        )
        self.assertEqual(status, 201, created)
        status, paused = self._post_path(
            f"/runs/{run_id}/action",
            {
                "action": "pause",
                "idempotency_key": "native-start-from-paused-pause",
            },
        )
        self.assertEqual(status, 202, paused)
        self.assertEqual(paused["projection"]["status"], "paused")

        status, started = self._post_path(
            f"/runs/{run_id}/action",
            {
                "action": "start",
                "idempotency_key": "native-start-from-paused-start",
            },
        )

        self.assertEqual(status, 200, started)
        self.assertEqual(started["projection"]["status"], "running")
        self.assertEqual(runtime.activated, [])
        self.assertEqual(len(runtime.resumed), 1)
        self.assertEqual(runtime.resumed[0]["run_id"], run_id)

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
        self.assertEqual(
            restored["binding"]["restore_provenance"],
            {
                "source": "python_durable_ledger",
                "status": "created",
            },
        )

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
        self.assertEqual(status, 202, payload)
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
        self.assertEqual(
            runtime.created[-1]["binding"]["initial_run_status"],
            "paused",
        )
        self.assertEqual(
            runtime.created[-1]["binding"]["restore_provenance"],
            {
                "source": "python_durable_ledger",
                "status": "paused",
            },
        )
        self.assertEqual(runtime.resumed[-1]["run_id"], run_id)

    def test_server_startup_quiesces_native_runs_behind_host_boundaries(self) -> None:
        runtime = _FakeNativeRuntime()
        self.server.dsh_native_runtime = runtime
        run_ids = {
            "paused": "run:native-startup-recover-paused",
            "failed": "run:native-startup-recover-failed",
            "cancelled": "run:native-startup-recover-cancelled",
        }
        for status_name, run_id in run_ids.items():
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
                    "idempotency_key": f"native-startup-recover-{status_name}",
                }
            )
            self.assertEqual(status, 201, payload)

        with self.server.mutation_lock:
            self.server.director.pause_run(run_ids["paused"])
            self.server.director.fail_run(
                run_ids["failed"],
                "simulated Host failure before native drain",
            )
            self.server.director.cancel_run(
                run_ids["cancelled"],
                "simulated Host cancellation before native drain",
            )
        self.assertEqual(runtime.paused, [])
        self.assertEqual(runtime.cancelled, [])

        database_path = Path(self.directory.name) / "events.sqlite3"
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        with (
            patch.dict(
                os.environ,
                {
                    "ECOLOGYRSI_DSH_RUNTIME_URL": "http://127.0.0.1:1",
                    "ECOLOGYRSI_DSH_RUNTIME_TOKEN": "startup-recovery-token",
                },
            ),
            patch(
                "ecologyrsi_dsh.application.runtime.DshNativeAgentRuntimeClient",
                return_value=runtime,
            ),
        ):
            self.server = EvolutionHTTPServer(("127.0.0.1", 0), database_path)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api"

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if len(runtime.paused) == 1 and len(runtime.cancelled) == 2:
                break
            time.sleep(0.001)

        self.assertEqual(
            {request["run_id"] for request in runtime.paused},
            {run_ids["paused"]},
        )
        self.assertEqual(
            {request["run_id"] for request in runtime.cancelled},
            {run_ids["failed"], run_ids["cancelled"]},
        )
        for status_name, run_id in run_ids.items():
            self.assertEqual(
                self.server.director.state(run_id).run.status.value,
                status_name,
            )
            self.assertEqual(
                self.server.dsh_tools._run_admission[run_id],
                "closed",
            )
            self.assertFalse(
                self.server.native_control_inflight_for(run_id, runtime)
            )

    def test_real_node_restart_resume_passes_the_python_handler_live_gate(self) -> None:
        run_id = "run:real-node-resume-restart"
        with _RealNodeRuntime() as runtime:
            self.server.dsh_native_runtime = runtime
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
                    "idempotency_key": "real-node-create-resume-restart",
                }
            )
            self.assertEqual(status, 201, payload)
            status, payload = self._post_path(
                f"/runs/{run_id}/control",
                {"action": "pause", "idempotency_key": "real-node-pause-restart"},
            )
            self.assertEqual(status, 202, payload)
            self.assertEqual(payload["projection"]["status"], "paused")

        with _RealNodeRuntime() as restarted:
            self.server.dsh_native_runtime = restarted
            status, payload = self._post_path(
                f"/runs/{run_id}/control",
                {"action": "resume", "idempotency_key": "real-node-resume-restart"},
            )

            self.assertEqual(status, 200, payload)
            self.assertEqual(payload["projection"]["status"], "running")
            self.assertEqual(restarted.status(run_id)["status"], "running")

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
            state = generation_execution_module.execute_generation((endpoint).server, run_id)

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

        self.assertEqual(status, 202, payload)
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
            self.assertFalse(request_thread.is_alive())
        finally:
            active_generation.release()
        request_thread.join(2)

        self.assertFalse(request_thread.is_alive())
        self.assertEqual(response[0][0], 202, response)
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
        fence = self.server.dsh_tools.open_admission(
            run_id,
            run_state_revision,
            1,
            role="researcher",
            stage="generation.research",
            idempotency_key="late-result",
        )
        self.server.dsh_tools.allocate_child_reservation(
            {
                "request_id": "native-cancel-late-reservation",
                "run_id": run_id,
                "parent_session_id": "session:research-host",
                "role": "researcher",
                "stage": "generation.research",
                "run_state_revision": run_state_revision,
                "stage_attempt": 1,
                "admission_id": fence.admission_id,
                "timeout_ms": 1_000,
                "item_digest": "d" * 64,
                "idempotency_key": "late-result",
            }
        )
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
                        "admission_id": fence.admission_id,
                    }
                )
            release_runtime_cancel.set()
            self.assertTrue(runtime_cancel_finished.wait(2))
            request_thread.join(0.1)
            self.assertFalse(request_thread.is_alive())
        finally:
            release_runtime_cancel.set()
            active_generation.release()
        request_thread.join(2)

        self.assertFalse(request_thread.is_alive())
        self.assertEqual(response[0][0], 202, response)
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
        with patch(
            "ecologyrsi_dsh.api.auto_progress."
            "_NATIVE_QUIESCENCE_RETRY_BASE_SECONDS",
            0.0,
        ):
            status, failed = self._post_path(f"/runs/{run_id}/control", control)

            self.assertEqual(status, 202, failed)
            self.assertEqual(failed["command_status"], "pending")
            self.assertEqual(
                self.server.director.state(run_id).run.status.value,
                "cancelled",
            )
            receipt_key = f"{run_id}:native-cancel-runtime-failure-control"
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                receipt = self.server.ledger.command_receipt(receipt_key)
                if receipt is not None and receipt.status == "completed":
                    break
                time.sleep(0.001)

        status, recovered = self._post_path(f"/runs/{run_id}/control", control)

        self.assertEqual(status, 200, recovered)
        self.assertEqual(recovered["projection"]["status"], "cancelled")
        # The Host terminal boundary remains durable during the transport
        # outage. The drain retries in place, and a same-key HTTP retry then
        # replays the completed receipt without a third runtime mutation.
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
