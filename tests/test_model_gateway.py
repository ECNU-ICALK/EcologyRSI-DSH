from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import socket
import tempfile
import unittest
from unittest.mock import patch
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Lock, Thread
from urllib.error import HTTPError, URLError

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.redaction import REDACTED
from ecologyrsi_dsh.integrations import model_gateway as model_gateway_module
from ecologyrsi_dsh.ledger import EventLedger
from ecologyrsi_dsh.model_gateway import (
    GatewayConfigurationError,
    GatewayResponseError,
    ModelConnection,
    ModelGateway,
)


class _GatewayStubHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append(  # type: ignore[attr-defined]
            {"path": self.path, "authorization": self.headers.get("Authorization"), "body": body}
        )
        status, payload = self.server.responses.pop(0)  # type: ignore[attr-defined]
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


class _RedirectHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.server.requests.append(self.headers.get("Authorization"))  # type: ignore[attr-defined]
        self.send_response(302)
        self.send_header("Location", self.server.location)  # type: ignore[attr-defined]
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def _completion(content: object) -> dict[str, object]:
    rendered = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return {"choices": [{"message": {"role": "assistant", "content": rendered}}]}


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class _RawBodyResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _RawBodyResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class _RawBodySequenceOpener:
    def __init__(self, *payloads: bytes) -> None:
        self.payloads = list(payloads)
        self.timeouts: list[float] = []

    def open(self, request: object, timeout: float) -> _RawBodyResponse:
        self.timeouts.append(timeout)
        return _RawBodyResponse(self.payloads.pop(0))


class _TimeoutOnceOpener:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def open(self, request: object, timeout: float) -> _FakeResponse:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("simulated timeout")
        return _FakeResponse(self.payload)


class _AlwaysTimeoutOpener:
    def __init__(self) -> None:
        self.calls = 0

    def open(self, request: object, timeout: float) -> _FakeResponse:
        self.calls += 1
        raise TimeoutError("simulated timeout")


class _IncompleteReadResponse:
    def __enter__(self) -> _IncompleteReadResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        raise IncompleteRead(b'{"choices":', 128)


class _InterruptedReadOpener:
    def __init__(self, payload: dict[str, object], *, failures: int) -> None:
        self.payload = payload
        self.failures = failures
        self.timeouts: list[float] = []

    def open(
        self, request: object, timeout: float
    ) -> _FakeResponse | _IncompleteReadResponse:
        self.timeouts.append(timeout)
        if len(self.timeouts) <= self.failures:
            return _IncompleteReadResponse()
        return _FakeResponse(self.payload)


class _SequenceOpener:
    def __init__(self, *actions: object) -> None:
        self.actions = list(actions)
        self.timeouts: list[float] = []

    def open(self, request: object, timeout: float) -> _FakeResponse:
        self.timeouts.append(timeout)
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        assert isinstance(action, dict)
        return _FakeResponse(action)


class _DiagnosticSequenceOpener(_SequenceOpener):
    def __init__(self, gateway: ModelGateway, *actions: object) -> None:
        super().__init__(*actions)
        self.gateway = gateway
        self.snapshots: list[dict[str, object]] = []

    def open(self, request: object, timeout: float) -> _FakeResponse:
        diagnostic = self.gateway.connection_status("policy-main")["last_request"]
        self.snapshots.append(dict(diagnostic) if isinstance(diagnostic, dict) else {})
        return super().open(request, timeout)


class _ConcurrentSampleOpener:
    def __init__(self, attempts_by_sample_id: dict[str, int]) -> None:
        self.attempts_by_sample_id = dict(attempts_by_sample_id)
        self.calls_by_sample_id = {
            sample_id: 0 for sample_id in attempts_by_sample_id
        }
        self._lock = Lock()
        self._first_attempts = Barrier(len(attempts_by_sample_id))

    def open(self, request: object, timeout: float) -> _FakeResponse:
        del timeout
        body = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        user_message = json.loads(body["messages"][1]["content"])
        sample_id = user_message["input"]["samples"][0]["sample_id"]
        with self._lock:
            self.calls_by_sample_id[sample_id] += 1
            attempt = self.calls_by_sample_id[sample_id]
        if attempt == 1:
            self._first_attempts.wait(timeout=2)
        if attempt < self.attempts_by_sample_id[sample_id]:
            raise TimeoutError("simulated concurrent timeout")
        tool_id = user_message["input"]["allowed_tool_ids"][0]
        completion = _completion(
            {
                "decisions": [
                    {
                        "sample_id": sample_id,
                        "next_tool": tool_id,
                        "reason_code": "route",
                        "confidence": 1.0,
                    }
                ]
            }
        )
        completion["usage"] = {
            "prompt_tokens": 11 if sample_id == "sample-fast" else 29,
            "completion_tokens": 7 if sample_id == "sample-fast" else 13,
            "total_tokens": 18 if sample_id == "sample-fast" else 42,
        }
        return _FakeResponse(completion)


class ModelGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _GatewayStubHandler)
        self.server.requests = []  # type: ignore[attr-defined]
        self.server.responses = []  # type: ignore[attr-defined]
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def gateway(self, *, verification_store: object | None = None) -> ModelGateway:
        catalog = json.dumps(
            [
                {
                    "model_id": "policy-main",
                    "gateway_url": self.base_url,
                    "model": "local-policy",
                    "api_key_env": "LOCAL_POLICY_TOKEN",
                }
            ]
        )
        return ModelGateway.from_env(
            {
                "ECOLOGYRSI_DSH_MODELS_JSON": catalog,
                "LOCAL_POLICY_TOKEN": "top-secret",
                # Most unit tests validate contracts rather than wall-clock
                # backoff.  Focused retry tests below inject a sleep recorder.
                "ECOLOGYRSI_DSH_MODEL_RETRY_BASE_SECONDS": "0",
                "ECOLOGYRSI_DSH_MODEL_RETRY_MAX_SECONDS": "0",
            },
            timeout=2,
            verification_store=verification_store,
        )

    def test_catalog_is_redacted_and_status_is_readable(self) -> None:
        gateway = self.gateway()
        before = gateway.catalog()
        self.assertEqual(before[0]["connection"]["state"], "configured")
        self.assertEqual(before[0]["label"], "policy-main")
        self.assertEqual(before[0]["roles"], ["propose", "judge"])
        self.assertTrue(before[0]["credential_configured"])
        self.assertFalse(before[0]["authenticated"])
        self.assertFalse(before[0]["authentication_verified"])
        self.assertEqual(before[0]["authentication_state"], "configured_unverified")
        self.assertFalse(before[0]["available"])
        rendered = json.dumps(before)
        self.assertNotIn("top-secret", rendered)
        self.assertNotIn("api_key_env", rendered)
        self.assertNotIn("token", rendered)
        self.assertNotIn("gateway_url", rendered)

        self.server.responses.append(  # type: ignore[attr-defined]
            (200, _completion({"parameters": {"alpha": 0.4}, "guidance": "降低波动"}))
        )
        result = gateway.propose(
            "policy-main",
            {"objective": "predict"},
            {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
        )
        self.assertEqual(result["parameters"], {"alpha": 0.4})
        self.assertEqual(result["guidance"], "降低波动")
        self.assertEqual(gateway.connection_status("policy-main")["state"], "available")
        after = gateway.catalog()[0]
        self.assertFalse(after["available"])
        self.assertFalse(after["authentication_verified"])
        self.assertFalse(after["authenticated"])
        self.assertEqual(after["authentication_state"], "configured_unverified")
        request = self.server.requests[0]  # type: ignore[attr-defined]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["authorization"], "Bearer top-secret")
        self.assertEqual(request["body"]["model"], "local-policy")
        self.assertEqual(request["body"]["response_format"], {"type": "json_object"})

    def test_sample_decide_is_label_free_and_returns_ordered_decisions(self) -> None:
        gateway = self.gateway()
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "decisions": [
                            {
                                "sample_id": "sample-2",
                                "next_tool": "persistence_baseline",
                                "reason_code": "stable_baseline",
                                "confidence": 0.6,
                            },
                            {
                                "sample_id": "sample-1",
                                "next_tool": "registered_algorithm_prediction",
                                "reason_code": "registered_forecast",
                                "confidence": 0.8,
                            },
                        ]
                    }
                ),
            )
        )

        result = gateway.sample_decide(
            "policy-main",
            role="planner",
            samples=[
                {"sample_id": "sample-1", "target": "canopy_temperature", "baseline": 20.0},
                {"sample_id": "sample-2", "target": "canopy_temperature", "baseline": 21.0},
            ],
            context={"candidate_id": "candidate-1", "dataset_digest": "abc"},
            available_tools=[
                {"tool_id": "registered_algorithm_prediction", "version": "1"},
                {"tool_id": "persistence_baseline", "version": "1"},
            ],
        )

        self.assertEqual(
            result,
            {
                "decisions": [
                    {
                        "sample_id": "sample-1",
                        "next_tool": "registered_algorithm_prediction",
                        "reason_code": "registered_forecast",
                        "confidence": 0.8,
                    },
                    {
                        "sample_id": "sample-2",
                        "next_tool": "persistence_baseline",
                        "reason_code": "stable_baseline",
                        "confidence": 0.6,
                    },
                ]
            },
        )
        request = self.server.requests[0]  # type: ignore[attr-defined]
        payload = json.loads(request["body"]["messages"][1]["content"])
        self.assertEqual(payload["operation"], "sample.planner")
        self.assertEqual(
            payload["input"]["allowed_tool_ids"],
            ["registered_algorithm_prediction", "persistence_baseline"],
        )
        self.assertIn(
            "copy_next_tool_exactly_from_allowed_tool_ids",
            payload["input"]["decision_policy"]["tool_id_contract"],
        )
        self.assertIn(
            "must never be appended",
            payload["response_contract"]["decisions"],
        )
        serialized_input = json.dumps(payload["input"], ensure_ascii=False)
        self.assertNotIn("observed", serialized_input)
        self.assertNotIn("label", serialized_input)

    def test_sample_operations_send_explicit_output_limits(self) -> None:
        gateway = self.gateway()
        for role in ("planner", "critic"):
            self.server.responses.append(  # type: ignore[attr-defined]
                (
                    200,
                    _completion(
                        {
                            "decisions": [
                                {
                                    "sample_id": f"sample-{role}",
                                    "next_tool": "persistence_baseline",
                                    "reason_code": "stable_baseline",
                                    "confidence": 1.0,
                                }
                            ]
                        }
                    ),
                )
            )
        gateway.sample_decide(
            "policy-main",
            role="planner",
            samples=[{"sample_id": "sample-planner"}],
            context={},
            available_tools=[{"tool_id": "persistence_baseline", "version": "1"}],
            max_tokens=3072,
        )
        gateway.sample_decide(
            "policy-main",
            role="critic",
            samples=[{"sample_id": "sample-critic"}],
            context={},
            available_tools=[{"tool_id": "persistence_baseline", "version": "1"}],
            max_tokens=2048,
        )

        self.assertEqual(self.server.requests[0]["body"]["max_tokens"], 3072)  # type: ignore[attr-defined]
        self.assertEqual(self.server.requests[1]["body"]["max_tokens"], 2048)  # type: ignore[attr-defined]

    def test_sample_call_token_bound_covers_every_http_attempt(self) -> None:
        gateway = self.gateway()
        arguments = {
            "model_id": "policy-main",
            "role": "planner",
            "samples": [
                {
                    "sample_id": "sample-bound",
                    "sample": {"features": {"history": [1.0, 2.0, 3.0]}},
                }
            ],
            "context": {"role": "planner"},
            "available_tools": [
                {"tool_id": "persistence_baseline", "version": "1"}
            ],
            "max_tokens": 3072,
        }

        attempts = gateway.max_attempts
        aggregate_bound = gateway.sample_decide_token_upper_bound(**arguments)
        gateway.max_attempts = 1
        one_attempt_bound = gateway.sample_decide_token_upper_bound(**arguments)

        self.assertEqual(aggregate_bound, attempts * one_attempt_bound)
        self.assertGreater(one_attempt_bound, 3072)
        self.assertEqual(self.server.requests, [])  # type: ignore[attr-defined]

    def test_compact_critic_profile_uses_sparse_review_prompt(self) -> None:
        gateway = self.gateway()
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "decisions": [
                            {
                                "sample_id": "sample-critic",
                                "next_tool": "accept",
                                "reason_code": "bounded_review_accept",
                                "confidence": 1.0,
                            }
                        ]
                    }
                ),
            )
        )

        gateway.sample_decide(
            "policy-main",
            role="critic",
            samples=[
                {
                    "sample_id": "sample-critic",
                    "target": "air_temperature",
                    "predicted": 20.0,
                    "trigger": "planner_low_confidence",
                }
            ],
            context={"critic_prompt_profile": "uncertain_or_failure_compact@1"},
            available_tools=[{"tool_id": "accept", "version": "1"}],
        )

        request = self.server.requests[0]  # type: ignore[attr-defined]
        payload = json.loads(request["body"]["messages"][1]["content"])
        policy = payload["input"]["decision_policy"]
        self.assertEqual(
            policy["goal"], "review_only_the_triggered_uncertain_or_failed_sample"
        )
        self.assertIn("invent_tools_or_data", policy["recovery"])

    def test_origin_shared_profile_resolves_context_before_network_call(self) -> None:
        gateway = self.gateway()
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "decisions": [
                            {
                                "sample_id": "sample-shared",
                                "next_tool": "persistence_baseline",
                                "reason_code": "initial_registered_route",
                                "confidence": 0.9,
                            }
                        ]
                    }
                ),
            )
        )
        origin_guard = {
            "origin_timestamps": [0],
            "causal_provenance_digests": [],
        }
        variant = {"target": "air_temperature"}
        variant_ref = digest(
            {
                "schema_version": (
                    "ecologyrsi-dsh.origin-shared-sample-context/1"
                ),
                "origin_guard": origin_guard,
                "variant": variant,
            }
        )
        shared_context = {
            "schema_version": (
                "ecologyrsi-dsh.origin-shared-sample-context/1"
            ),
            "origin_guard": origin_guard,
            "sample_defaults": {"origin_timestamp": 0},
            "label_free_context_defaults": {},
            "sample_variants": {variant_ref: variant},
            "sample_variant_refs": {"sample-shared": variant_ref},
            "sample_count": 1,
        }
        context_ref = digest(shared_context)
        context = {
            "sample_planner_prompt_profile": {
                "version": "origin_shared_context@1"
            },
            "shared_sample_contexts": {
                context_ref: shared_context
            },
        }
        samples = [
            {
                "sample_id": "sample-shared",
                "context_ref": context_ref,
                "horizon_hours": 1,
                "target_timestamp": 1,
                "attempt": 1,
                "failure_feedback": [],
            }
        ]

        serialized_bodies = []
        serialize = model_gateway_module._chat_request_body

        def capture_body(*args, **kwargs):
            body = serialize(*args, **kwargs)
            serialized_bodies.append(body)
            return body

        arguments = {
            "model_id": "policy-main",
            "role": "planner",
            "samples": samples,
            "context": context,
            "available_tools": [
                {"tool_id": "persistence_baseline", "version": "1"}
            ],
            "max_tokens": 3072,
        }
        with patch.object(
            model_gateway_module,
            "_chat_request_body",
            side_effect=capture_body,
        ):
            upper_bound = gateway.sample_decide_token_upper_bound(**arguments)
            gateway.sample_decide(**arguments)

        self.assertGreater(upper_bound, len(serialized_bodies[0]))
        self.assertEqual(serialized_bodies[0], serialized_bodies[1])

        request = self.server.requests[0]  # type: ignore[attr-defined]
        payload = json.loads(request["body"]["messages"][1]["content"])
        policy = payload["input"]["decision_policy"]
        self.assertIn("sample_context_resolution", policy)
        self.assertEqual(
            payload["input"]["samples"][0]["context_ref"], context_ref
        )

    def test_origin_shared_profile_rejects_tampered_context_before_network(self) -> None:
        gateway = self.gateway()
        origin_guard = {
            "origin_timestamps": [0],
            "causal_provenance_digests": [],
        }
        variant = {"target": "air_temperature"}
        variant_ref = digest(
            {
                "schema_version": (
                    "ecologyrsi-dsh.origin-shared-sample-context/1"
                ),
                "origin_guard": origin_guard,
                "variant": variant,
            }
        )
        shared_context = {
            "schema_version": "ecologyrsi-dsh.origin-shared-sample-context/1",
            "origin_guard": origin_guard,
            "sample_defaults": {"origin_timestamp": 0},
            "label_free_context_defaults": {},
            "sample_variants": {variant_ref: variant},
            "sample_variant_refs": {"sample-shared": variant_ref},
            "sample_count": 1,
        }
        context_ref = digest(shared_context)
        shared_context["sample_count"] = 2

        with self.assertRaisesRegex(ValueError, "context digest"):
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[
                    {
                        "sample_id": "sample-shared",
                        "context_ref": context_ref,
                    }
                ],
                context={
                    "sample_planner_prompt_profile": {
                        "version": "origin_shared_context@1"
                    },
                    "shared_sample_contexts": {context_ref: shared_context},
                },
                available_tools=[
                    {"tool_id": "persistence_baseline", "version": "1"}
                ],
            )

        shared_context["sample_count"] = 1
        shared_context["sample_variants"] = {
            variant_ref: {"target": "tampered-target"}
        }
        rebound_context_ref = digest(shared_context)
        with self.assertRaisesRegex(ValueError, "variant digest"):
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[
                    {
                        "sample_id": "sample-shared",
                        "context_ref": rebound_context_ref,
                    }
                ],
                context={
                    "sample_planner_prompt_profile": {
                        "version": "origin_shared_context@1"
                    },
                    "shared_sample_contexts": {
                        rebound_context_ref: shared_context
                    },
                },
                available_tools=[
                    {"tool_id": "persistence_baseline", "version": "1"}
                ],
            )

        self.assertEqual(self.server.requests, [])  # type: ignore[attr-defined]

    def test_origin_shared_profile_rejects_unresolved_reference(self) -> None:
        gateway = self.gateway()

        with self.assertRaisesRegex(ValueError, "context_ref is unresolved"):
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[
                    {
                        "sample_id": "sample-shared",
                        "context_ref": "a" * 64,
                    }
                ],
                context={
                    "sample_planner_prompt_profile": {
                        "version": "origin_shared_context@1"
                    },
                    "shared_sample_contexts": {"b" * 64: {}},
                },
                available_tools=[
                    {"tool_id": "persistence_baseline", "version": "1"}
                ],
            )

        self.assertEqual(self.server.requests, [])  # type: ignore[attr-defined]

    def test_sample_decide_rejects_labels_before_the_network_call(self) -> None:
        gateway = self.gateway()

        with self.assertRaisesRegex(ValueError, "observed"):
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[{"sample_id": "sample-1", "metadata": {"observed": 20.0}}],
                context={},
                available_tools=[{"tool_id": "persistence_baseline", "version": "1"}],
            )
        with self.assertRaisesRegex(ValueError, "label"):
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[{"sample_id": "sample-1"}],
                context={"nested": [{"label": 20.0}]},
                available_tools=[{"tool_id": "persistence_baseline", "version": "1"}],
            )
        with self.assertRaisesRegex(ValueError, "Ground-Truth"):
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[
                    {
                        "sample_id": "sample-1",
                        "metadata": {"nested": {"Ground-Truth": 20.0}},
                    }
                ],
                context={},
                available_tools=[{"tool_id": "persistence_baseline", "version": "1"}],
            )
        with self.assertRaisesRegex(ValueError, "actualValue"):
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[{"sample_id": "sample-1"}],
                context={"nested": {"actualValue": 20.0}},
                available_tools=[{"tool_id": "persistence_baseline", "version": "1"}],
            )

        self.assertEqual(self.server.requests, [])  # type: ignore[attr-defined]

    def test_sample_decide_marks_forbidden_response_field_as_split_eligible(
        self,
    ) -> None:
        gateway = self.gateway()
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "decisions": [
                            {
                                "sample_id": "sample-1",
                                "next_tool": "persistence_baseline",
                                "reason_code": "route",
                                "confidence": 0.5,
                                "metadata": {
                                    "nested": {"Ground-Truth": 20.0}
                                },
                            }
                        ]
                    }
                ),
            )
        )

        with self.assertRaises(GatewayResponseError) as raised:
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[{"sample_id": "sample-1"}],
                context={},
                available_tools=[
                    {"tool_id": "persistence_baseline", "version": "1"}
                ],
            )

        self.assertEqual(
            raised.exception.error_code,
            "sample_decisions_forbidden_field",
        )
        self.assertTrue(raised.exception.split_eligible)
        self.assertFalse(raised.exception.retryable)
        self.assertFalse(gateway.catalog()[0]["api_invalid"])

    def test_sample_decide_rejects_invalid_response_and_batch_overflow(self) -> None:
        gateway = self.gateway()
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "decisions": [
                            {
                                "sample_id": "sample-1",
                                "next_tool": "unregistered_tool",
                                "reason_code": "bad_tool",
                                "confidence": 0.5,
                            }
                        ]
                    }
                ),
            )
        )

        with self.assertRaisesRegex(GatewayResponseError, "unavailable tool") as raised:
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[{"sample_id": "sample-1"}],
                context={},
                available_tools=[{"tool_id": "persistence_baseline", "version": "1"}],
            )
        self.assertEqual(raised.exception.error_code, "sample_decision_tool_invalid")
        self.assertTrue(raised.exception.split_eligible)
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(len(self.server.requests), 1)  # type: ignore[attr-defined]
        with self.assertRaisesRegex(ValueError, "at most 128"):
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[{"sample_id": f"sample-{index}"} for index in range(129)],
                context={},
                available_tools=[{"tool_id": "persistence_baseline", "version": "1"}],
            )
        self.assertEqual(len(self.server.requests), 1)  # type: ignore[attr-defined]

    def test_sample_decide_marks_per_decision_contract_errors_as_split_eligible(
        self,
    ) -> None:
        invalid_decisions = (
            (
                {
                    "sample_id": "sample-1",
                    "next_tool": "persistence_baseline",
                    "reason_code": "route",
                    "confidence": 0.5,
                    "extra": "unsupported",
                },
                "sample_decision_format_invalid",
            ),
            (
                {
                    "sample_id": "sample-1",
                    "next_tool": 7,
                    "reason_code": "route",
                    "confidence": 0.5,
                },
                "sample_decision_tool_invalid",
            ),
            (
                {
                    "sample_id": "sample-1",
                    "next_tool": "persistence_baseline",
                    "reason_code": "route",
                    "confidence": 2.0,
                },
                "sample_decision_confidence_invalid",
            ),
        )
        for decision, error_code in invalid_decisions:
            with self.subTest(error_code=error_code):
                gateway = self.gateway()
                self.server.responses.append(  # type: ignore[attr-defined]
                    (200, _completion({"decisions": [decision]}))
                )

                with self.assertRaises(GatewayResponseError) as raised:
                    gateway.sample_decide(
                        "policy-main",
                        role="planner",
                        samples=[{"sample_id": "sample-1"}],
                        context={},
                        available_tools=[
                            {
                                "tool_id": "persistence_baseline",
                                "version": "1",
                            }
                        ],
                    )

                self.assertEqual(raised.exception.error_code, error_code)
                self.assertTrue(raised.exception.split_eligible)
                self.assertFalse(raised.exception.retryable)

    def test_sample_decide_marks_incomplete_decision_count_as_split_eligible(
        self,
    ) -> None:
        gateway = self.gateway()
        self.server.responses.append(  # type: ignore[attr-defined]
            (200, _completion({"decisions": []}))
        )

        with self.assertRaises(GatewayResponseError) as raised:
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[{"sample_id": "sample-1"}],
                context={},
                available_tools=[
                    {"tool_id": "persistence_baseline", "version": "1"}
                ],
            )

        self.assertEqual(
            raised.exception.error_code,
            "sample_decisions_incomplete",
        )
        self.assertTrue(raised.exception.split_eligible)
        self.assertFalse(raised.exception.retryable)

    def test_sample_decide_reuses_retry_policy_and_critic_role(self) -> None:
        gateway = self.gateway()
        opener = _TimeoutOnceOpener(
            _completion(
                {
                    "decisions": [
                        {
                            "sample_id": "sample-1",
                            "next_tool": "persistence_baseline",
                            "reason_code": "fallback",
                            "confidence": 1.0,
                        }
                    ]
                }
            )
        )
        gateway._opener = opener  # type: ignore[assignment, attr-defined]

        result = gateway.sample_decide(
            "policy-main",
            role="critic",
            samples=[{"sample_id": "sample-1"}],
            context={},
            available_tools=[{"tool_id": "persistence_baseline", "version": "1"}],
        )

        self.assertEqual(result["decisions"][0]["next_tool"], "persistence_baseline")
        self.assertEqual(opener.calls, 2)
        diagnostics = gateway.connection_status("policy-main")["last_request"]
        self.assertEqual(diagnostics["operation"], "sample.critic")
        self.assertEqual(diagnostics["attempts"], 2)

    def test_sample_decide_classifies_length_without_reading_private_reasoning(
        self,
    ) -> None:
        gateway = self.gateway()
        private_marker = "PRIVATE-REASONING-MUST-NOT-BE-READ"
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "role": "assistant",
                                "content": '{"decisions":[{"sample_id":"sample-1"',
                                "reasoning_content": private_marker
                                + '{"decisions":[{"sample_id":"sample-1",'
                                '"next_tool":"persistence_baseline",'
                                '"reason_code":"private","confidence":1.0}]}',
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 4096,
                        "total_tokens": 4216,
                        "completion_tokens_details": {"reasoning_tokens": 4000},
                        "private_usage": private_marker,
                    },
                },
            )
        )

        with self.assertRaises(GatewayResponseError) as raised:
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[{"sample_id": "sample-1"}],
                context={},
                available_tools=[
                    {"tool_id": "persistence_baseline", "version": "1"}
                ],
            )

        error = raised.exception
        self.assertEqual(error.error_code, "output_truncated")
        self.assertTrue(error.split_eligible)
        self.assertFalse(error.retryable)
        self.assertEqual(error.finish_reason, "length")
        self.assertEqual(
            error.usage,
            {
                "prompt_tokens": 120,
                "completion_tokens": 4096,
                "total_tokens": 4216,
            },
        )
        status = gateway.connection_status("policy-main")
        self.assertEqual(status["state"], "configured")
        self.assertEqual(status["last_request"]["classification"], "transient")
        self.assertEqual(
            status["last_request"]["response_metadata"],
            {
                "finish_reason": "length",
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 4096,
                    "total_tokens": 4216,
                },
            },
        )
        public_state = json.dumps(status, ensure_ascii=False)
        self.assertNotIn(private_marker, str(error))
        self.assertNotIn(private_marker, public_state)
        self.assertNotIn("reasoning", public_state)
        self.assertFalse(gateway.catalog()[0]["api_invalid"])
        self.assertEqual(len(self.server.requests), 1)  # type: ignore[attr-defined]

    def test_sample_decide_never_falls_back_to_reasoning_content(self) -> None:
        gateway = self.gateway()
        private_marker = "PRIVATE-FINAL-LOOKALIKE"
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": private_marker
                                + '{"decisions":[{"sample_id":"sample-1",'
                                '"next_tool":"persistence_baseline",'
                                '"reason_code":"private","confidence":1.0}]}',
                            },
                        }
                    ]
                },
            )
        )

        with self.assertRaises(GatewayResponseError) as raised:
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[{"sample_id": "sample-1"}],
                context={},
                available_tools=[
                    {"tool_id": "persistence_baseline", "version": "1"}
                ],
            )

        self.assertEqual(raised.exception.error_code, "final_content_missing")
        self.assertTrue(raised.exception.split_eligible)
        self.assertNotIn(private_marker, str(raised.exception))
        self.assertNotIn(
            private_marker,
            json.dumps(gateway.connection_status("policy-main"), ensure_ascii=False),
        )

    def test_finish_reason_is_allowlisted_and_content_filter_is_not_split(self) -> None:
        valid_content = {
            "decisions": [
                {
                    "sample_id": "sample-1",
                    "next_tool": "persistence_baseline",
                    "reason_code": "baseline",
                    "confidence": 1.0,
                }
            ]
        }
        for finish_reason, error_code in (
            ("content_filter", "content_filtered"),
            ("provider-private-reason", "finish_reason_invalid"),
        ):
            with self.subTest(finish_reason=finish_reason):
                gateway = self.gateway()
                self.server.responses.append(  # type: ignore[attr-defined]
                    (
                        200,
                        {
                            "choices": [
                                {
                                    "finish_reason": finish_reason,
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps(valid_content),
                                    },
                                }
                            ]
                        },
                    )
                )

                with self.assertRaises(GatewayResponseError) as raised:
                    gateway.sample_decide(
                        "policy-main",
                        role="planner",
                        samples=[{"sample_id": "sample-1"}],
                        context={},
                        available_tools=[
                            {"tool_id": "persistence_baseline", "version": "1"}
                        ],
                    )

                self.assertEqual(raised.exception.error_code, error_code)
                self.assertFalse(raised.exception.split_eligible)
                self.assertNotIn(finish_reason, str(raised.exception))

    def test_success_diagnostics_only_retain_allowlisted_response_metadata(self) -> None:
        gateway = self.gateway()
        schema = {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}}
        completion = _completion({"parameters": {"alpha": 0.4}})
        completion["choices"][0]["finish_reason"] = "stop"  # type: ignore[index]
        completion["usage"] = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 19},
            "private": "must-not-be-retained",
        }
        self.server.responses.append((200, completion))  # type: ignore[attr-defined]

        self.assertEqual(
            gateway.propose("policy-main", {}, schema),
            {"parameters": {"alpha": 0.4}},
        )

        metadata = gateway.connection_status("policy-main")["last_request"][
            "response_metadata"
        ]
        self.assertEqual(
            metadata,
            {
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
        )
        self.assertNotIn("reasoning", json.dumps(metadata))
        self.assertNotIn("private", json.dumps(metadata))

    def test_legacy_transient_error_is_not_reported_as_api_invalid(self) -> None:
        gateway = self.gateway()
        gateway._set_status("policy-main", "error", "HTTP 503")  # type: ignore[attr-defined]

        model = gateway.catalog()[0]

        self.assertEqual(model["connection_phase"], "transient_error")
        self.assertTrue(model["temporarily_unavailable"])
        self.assertFalse(model["api_invalid"])
        self.assertEqual(model["authentication_state"], "configured_unverified")

    def test_legacy_response_contract_error_is_not_reported_as_api_invalid(self) -> None:
        gateway = self.gateway()
        gateway._set_status(  # type: ignore[attr-defined]
            "policy-main",
            "error",
            "model response content must contain one JSON object",
        )

        model = gateway.catalog()[0]

        self.assertEqual(model["connection_phase"], "transient_error")
        self.assertTrue(model["temporarily_unavailable"])
        self.assertFalse(model["api_invalid"])
        self.assertTrue(model["execution_available"])

    def test_judge_returns_bounded_intervention_shape(self) -> None:
        gateway = self.gateway()
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "accepted": True,
                        "guidance": "继续局部搜索",
                        "parameter_override": {"window": 7},
                    }
                ),
            )
        )
        result = gateway.judge(
            "policy-main",
            {"parameters": {"window": 5}},
            {"rmse": 0.3, "baseline_rmse": 0.5},
        )
        self.assertEqual(
            result,
            {"accepted": True, "guidance": "继续局部搜索", "parameter_override": {"window": 7}},
        )

    def test_judge_ignores_malformed_optional_override(self) -> None:
        gateway = self.gateway()
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "accepted": False,
                        "guidance": "保持当前候选并收窄搜索范围",
                        "parameter_override": "建议将 blend 调整为 0.1",
                    }
                ),
            )
        )
        self.assertEqual(
            gateway.judge(
                "policy-main",
                {"parameters": {"alpha": 0.4}},
                {"rmse": 0.8, "baseline_rmse": 0.6},
            ),
            {"accepted": False, "guidance": "保持当前候选并收窄搜索范围"},
        )

    def test_proposal_contract_failure_is_retryable_without_masking_cause(self) -> None:
        gateway = self.gateway()
        self.server.responses.append(  # type: ignore[attr-defined]
            (200, _completion({"title": "missing parameters"}))
        )

        with self.assertRaises(GatewayResponseError) as raised:
            gateway.propose(
                "policy-main",
                {},
                {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
            )

        error = raised.exception
        self.assertTrue(error.retryable)
        self.assertEqual(error.error_code, "proposal_response_contract_invalid")
        self.assertIsInstance(error.__cause__, GatewayResponseError)
        self.assertIn("must contain parameters", str(error.__cause__))

    def test_judge_contract_failure_is_retryable_without_masking_cause(self) -> None:
        gateway = self.gateway()
        self.server.responses.append(  # type: ignore[attr-defined]
            (200, _completion({"accepted": "yes"}))
        )

        with self.assertRaises(GatewayResponseError) as raised:
            gateway.judge(
                "policy-main",
                {"parameters": {"alpha": 0.4}},
                {"rmse": 0.3},
            )

        error = raised.exception
        self.assertTrue(error.retryable)
        self.assertEqual(error.error_code, "judge_response_contract_invalid")
        self.assertIsInstance(error.__cause__, GatewayResponseError)
        self.assertIn("accepted must be a boolean", str(error.__cause__))

    def test_truncated_proposal_and_judge_responses_remain_retryable(self) -> None:
        calls = (
            (
                "proposal",
                lambda gateway: gateway.propose(
                    "policy-main",
                    {},
                    {
                        "alpha": {
                            "type": "number",
                            "minimum": 0.05,
                            "maximum": 0.95,
                        }
                    },
                ),
                "proposal_response_contract_invalid",
            ),
            (
                "judge",
                lambda gateway: gateway.judge(
                    "policy-main",
                    {"parameters": {"alpha": 0.4}},
                    {"rmse": 0.3},
                ),
                "judge_response_contract_invalid",
            ),
        )
        for operation, call, expected_code in calls:
            with self.subTest(operation=operation):
                gateway = self.gateway()
                gateway.max_attempts = 1
                gateway._opener = _SequenceOpener(  # type: ignore[assignment, attr-defined]
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {
                                    "role": "assistant",
                                    "content": '{"incomplete":',
                                },
                            }
                        ],
                        "usage": {"total_tokens": 100},
                    }
                )

                with self.assertRaises(GatewayResponseError) as raised:
                    call(gateway)

                error = raised.exception
                self.assertTrue(error.retryable)
                self.assertEqual(error.error_code, expected_code)
                self.assertEqual(error.finish_reason, "length")
                self.assertEqual(error.usage, {"total_tokens": 100})
                self.assertIsInstance(error.__cause__, GatewayResponseError)
                self.assertEqual(error.__cause__.error_code, "output_truncated")

    def test_research_plan_projects_model_descriptions_to_host_schema(self) -> None:
        gateway = self.gateway()
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "team": {
                            "name": "温室研究团队",
                            "roles": ["数据分析", "", {"name": "能力编译"}],
                            "extra_notes": "仅供模型说明",
                        },
                        "prediction_model": {
                            "name": "滚动残差模型",
                            "family": "time_series",
                            "hyperparameters": {"window": 24},
                            "outputs": ["temperature"],
                        },
                        "strategy": {
                            "name": "保守局部搜索",
                            "steps": ["先评测", "再比较"],
                            "constraints": ["不得越界"],
                        },
                        "research": [
                            {
                                "title": "时间序列验证",
                                "url": "https://example.test/source",
                                "extra": "忽略",
                            },
                            {
                                "evidence_ref": "greenlight-model",
                                "summary": "温室状态耦合支持分目标建模。",
                                "private_reasoning": "不应持久化",
                            },
                            {
                                "source": "Bearer reflected-secret",
                                "finding": "敏感来源文本必须整体脱敏。",
                            },
                            {"private_reasoning": "空投影不应被持久化"},
                            "malformed source",
                        ],
                        "implementation_notes": "只使用宿主登记能力",
                        "confidence": 0.8,
                    }
                ),
            )
        )
        result = gateway.research_plan("policy-main", {"domain": "greenhouse"})
        self.assertEqual(result["team"], {"name": "温室研究团队", "roles": ["数据分析", "能力编译"]})
        self.assertEqual(
            result["prediction_model"],
            {"name": "滚动残差模型", "family": "time_series"},
        )
        self.assertEqual(result["strategy"], {"name": "保守局部搜索", "steps": ["先评测", "再比较"]})
        self.assertEqual(
            result["research"],
            [
                {
                    "title": "时间序列验证",
                    "url": "https://example.test/source",
                },
                {
                    "source": "greenlight-model",
                    "finding": "温室状态耦合支持分目标建模。",
                },
                {
                    "source": REDACTED,
                    "finding": "敏感来源文本必须整体脱敏。",
                },
            ],
        )
        self.assertNotIn("private_reasoning", json.dumps(result, ensure_ascii=False))
        self.assertNotIn("reflected-secret", json.dumps(result, ensure_ascii=False))
        self.assertEqual(result["confidence"], 0.8)

    def test_research_plan_accepts_only_an_exact_registered_algorithm_blueprint(
        self,
    ) -> None:
        gateway = self.gateway()
        blueprint = {
            "schema_version": "ecologyrsi-dsh.algorithm-blueprint-request/1",
            "pipeline_id": "greenhouse-targetwise-ridge@1",
            "operator_ids": [
                "host.feature.causal-lag-exogenous@1",
                "host.fit.partition-statistics@1",
                "host.fit.closed-form-ridge@1",
                "host.predictor.targetwise-residual-or-persistence@1",
                "host.postprocess.physical-bounds@1",
            ],
            "parameter_names": [
                "history_steps",
                "ridge_alpha",
                "air_temperature_residual_scale",
                "relative_humidity_residual_scale",
                "co2_concentration_residual_scale",
            ],
        }
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "prediction_model": {
                            "id": "greenhouse-targetwise-ridge@1"
                        },
                        "research": [
                            {
                                "title": "target-specific shrinkage",
                                "url": "https://example.test/targetwise-ridge",
                            }
                        ],
                        "algorithm_blueprint": {
                            **blueprint,
                            "evidence_refs": [
                                "knowledge:targetwise-ridge",
                                "knowledge:online-direction",
                            ],
                            "rationale": "keep CO2 on persistence when scale is zero",
                        },
                        "algorithm_synthesis": {
                            "schema_version": "ecologyrsi-dsh.algorithm-synthesis/1",
                            "pipeline_id": "greenhouse-targetwise-ridge@1",
                            "evidence_refs": ["knowledge:online-direction"],
                            "parameter_focus": [
                                "ridge_alpha",
                                "co2_concentration_residual_scale",
                            ],
                            "rationale": (
                                "Translate the retrieved target-specific shrinkage "
                                "direction into the registered operator graph."
                            ),
                        },
                    }
                ),
            )
        )

        result = gateway.research_plan(
            "policy-main",
            {
                "algorithm_blueprint_catalog": [blueprint],
                "knowledge_snapshot": {
                    "evidence_catalog": [
                        {
                            "knowledge_id": "knowledge:targetwise-ridge",
                            "source_url": "https://example.test/targetwise-ridge",
                            "execution_status": "available_not_selected",
                            "capability_kind": "predictor",
                            "capability_ids": ["greenhouse-targetwise-ridge@1"],
                        },
                        {
                            "knowledge_id": "knowledge:online-direction",
                            "source_url": "https://example.test/online-direction",
                            "execution_status": "metadata_only",
                            "capability_kind": None,
                            "capability_ids": [],
                        },
                    ]
                },
            },
        )

        self.assertEqual(
            result["algorithm_blueprint"]["pipeline_id"],
            "greenhouse-targetwise-ridge@1",
        )
        self.assertEqual(
            result["algorithm_blueprint"]["evidence_refs"],
            ["knowledge:targetwise-ridge", "knowledge:online-direction"],
        )
        self.assertEqual(
            result["algorithm_synthesis"]["parameter_focus"],
            ["ridge_alpha", "co2_concentration_residual_scale"],
        )

        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "algorithm_blueprint": {
                            **blueprint,
                            "evidence_refs": ["knowledge:targetwise-ridge"],
                        },
                        "algorithm_synthesis": {
                            "schema_version": "ecologyrsi-dsh.algorithm-synthesis/1",
                            "pipeline_id": "greenhouse-targetwise-ridge@1",
                            "evidence_refs": ["knowledge:targetwise-ridge"],
                            "parameter_focus": ["unregistered_parameter"],
                            "rationale": "must fail before compilation",
                        },
                    }
                ),
            )
        )
        with self.assertRaisesRegex(
            GatewayResponseError,
            "algorithm_synthesis is invalid",
        ):
            gateway.research_plan(
                "policy-main",
                {
                    "algorithm_blueprint_catalog": [blueprint],
                    "knowledge_snapshot": {
                        "evidence_catalog": [
                            {
                                "knowledge_id": "knowledge:targetwise-ridge",
                                "execution_status": "available_not_selected",
                                "capability_kind": "predictor",
                                "capability_ids": ["greenhouse-targetwise-ridge@1"],
                            }
                        ]
                    },
                },
            )

        tampered = {
            **blueprint,
            "operator_ids": ["model.generated.python@1"],
            "evidence_refs": ["knowledge:targetwise-ridge"],
        }
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "algorithm_blueprint": tampered,
                        "algorithm_synthesis": {
                            "schema_version": "ecologyrsi-dsh.algorithm-synthesis/1",
                            "pipeline_id": "greenhouse-targetwise-ridge@1",
                            "evidence_refs": ["knowledge:targetwise-ridge"],
                            "parameter_focus": ["ridge_alpha"],
                            "rationale": "bounded validation fixture",
                        },
                    }
                ),
            )
        )
        with self.assertRaisesRegex(GatewayResponseError, "does not match"):
            gateway.research_plan(
                "policy-main",
                {
                    "algorithm_blueprint_catalog": [blueprint],
                    "knowledge_snapshot": {
                        "evidence_catalog": [
                            {
                                "knowledge_id": "knowledge:targetwise-ridge",
                                "execution_status": "available_not_selected",
                                "capability_kind": "predictor",
                                "capability_ids": ["greenhouse-targetwise-ridge@1"],
                            }
                        ]
                    },
                },
            )

        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "algorithm_blueprint": {
                            **blueprint,
                            "evidence_refs": ["knowledge:not-in-snapshot"],
                        },
                        "algorithm_synthesis": {
                            "schema_version": "ecologyrsi-dsh.algorithm-synthesis/1",
                            "pipeline_id": "greenhouse-targetwise-ridge@1",
                            "evidence_refs": ["knowledge:not-in-snapshot"],
                            "parameter_focus": ["ridge_alpha"],
                            "rationale": "bounded validation fixture",
                        },
                    }
                ),
            )
        )
        with self.assertRaisesRegex(GatewayResponseError, "outside the frozen snapshot"):
            gateway.research_plan(
                "policy-main",
                {
                    "algorithm_blueprint_catalog": [blueprint],
                    "knowledge_snapshot": {
                        "evidence_catalog": [
                            {
                                "knowledge_id": "knowledge:targetwise-ridge",
                                "execution_status": "available_not_selected",
                                "capability_kind": "predictor",
                                "capability_ids": ["greenhouse-targetwise-ridge@1"],
                            }
                        ]
                    },
                },
            )

        unusable_evidence = (
            (
                "metadata-only",
                {
                    "knowledge_id": "knowledge:targetwise-ridge",
                    "execution_status": "metadata_only",
                    "capability_kind": "predictor",
                    "capability_ids": ["greenhouse-targetwise-ridge@1"],
                },
            ),
            (
                "different-predictor",
                {
                    "knowledge_id": "knowledge:targetwise-ridge",
                    "execution_status": "available_not_selected",
                    "capability_kind": "predictor",
                    "capability_ids": ["greenhouse-exogenous-ridge@1"],
                },
            ),
        )
        for label, evidence in unusable_evidence:
            with self.subTest(label=label):
                self.server.responses.append(  # type: ignore[attr-defined]
                    (
                        200,
                        _completion(
                            {
                                "algorithm_blueprint": {
                                    **blueprint,
                                    "evidence_refs": ["knowledge:targetwise-ridge"],
                                },
                                "algorithm_synthesis": {
                                    "schema_version": (
                                        "ecologyrsi-dsh.algorithm-synthesis/1"
                                    ),
                                    "pipeline_id": "greenhouse-targetwise-ridge@1",
                                    "evidence_refs": ["knowledge:targetwise-ridge"],
                                    "parameter_focus": ["ridge_alpha"],
                                    "rationale": "bounded validation fixture",
                                },
                            }
                        ),
                    )
                )
                with self.assertRaisesRegex(
                    GatewayResponseError,
                    "requires frozen executable evidence",
                ):
                    gateway.research_plan(
                        "policy-main",
                        {
                            "algorithm_blueprint_catalog": [blueprint],
                            "knowledge_snapshot": {
                                "evidence_catalog": [
                                    evidence,
                                    {
                                        "knowledge_id": "knowledge:compatible",
                                        "execution_status": "available_not_selected",
                                        "capability_kind": "predictor",
                                        "capability_ids": [
                                            "greenhouse-targetwise-ridge@1"
                                        ],
                                    },
                                ]
                            },
                        },
                    )

    def test_research_plan_accepts_split_executable_and_research_evidence_without_retry(
        self,
    ) -> None:
        gateway = self.gateway()
        blueprint = {
            "schema_version": "ecologyrsi-dsh.algorithm-blueprint-request/1",
            "pipeline_id": "greenhouse-targetwise-ridge@1",
            "operator_ids": [
                "host.feature.causal-lag-exogenous@1",
                "host.fit.partition-statistics@1",
                "host.fit.closed-form-ridge@1",
                "host.predictor.targetwise-residual-or-persistence@1",
                "host.postprocess.physical-bounds@1",
            ],
            "parameter_names": [
                "history_steps",
                "ridge_alpha",
                "air_temperature_residual_scale",
                "relative_humidity_residual_scale",
                "co2_concentration_residual_scale",
            ],
        }
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "algorithm_blueprint": {
                            **blueprint,
                            "evidence_refs": ["knowledge:executable"],
                            "rationale": "bind the design to the registered predictor",
                        },
                        "algorithm_synthesis": {
                            "schema_version": (
                                "ecologyrsi-dsh.algorithm-synthesis/1"
                            ),
                            "pipeline_id": "greenhouse-targetwise-ridge@1",
                            "evidence_refs": ["openalex:W123"],
                            "parameter_focus": ["ridge_alpha"],
                            "rationale": (
                                "apply the frozen research direction to the "
                                "registered ridge parameter"
                            ),
                        },
                    }
                ),
            )
        )

        result = gateway.research_plan(
            "policy-main",
            {
                "algorithm_blueprint_catalog": [blueprint],
                "knowledge_snapshot": {
                    "evidence_catalog": [
                        {
                            "knowledge_id": "knowledge:executable",
                            "execution_status": "available_not_selected",
                            "capability_kind": "predictor",
                            "capability_id": "greenhouse-targetwise-ridge@1",
                            "capability_ids": [],
                        },
                        {
                            "knowledge_id": "openalex:W123",
                            "execution_status": "metadata_only",
                            "capability_kind": None,
                            "capability_ids": [],
                            "evidence_digest": "b" * 64,
                        },
                    ]
                },
            },
        )

        self.assertEqual(
            result["algorithm_blueprint"]["evidence_refs"],
            ["knowledge:executable"],
        )
        self.assertEqual(
            result["algorithm_synthesis"]["evidence_refs"],
            ["openalex:W123"],
        )
        self.assertEqual(len(self.server.requests), 1)  # type: ignore[attr-defined]
        request_message = json.loads(  # type: ignore[attr-defined]
            self.server.requests[0]["body"]["messages"][1]["content"]
        )
        self.assertNotIn(
            "algorithm_synthesis_correction",
            request_message["input"]["context"],
        )

    def test_research_plan_retries_synthesis_that_skips_online_evidence(
        self,
    ) -> None:
        gateway = self.gateway()
        blueprint = {
            "schema_version": "ecologyrsi-dsh.algorithm-blueprint-request/1",
            "pipeline_id": "greenhouse-targetwise-ridge@1",
            "operator_ids": [
                "host.feature.causal-lag-exogenous@1",
                "host.fit.partition-statistics@1",
                "host.fit.closed-form-ridge@1",
                "host.predictor.targetwise-residual-or-persistence@1",
                "host.postprocess.physical-bounds@1",
            ],
            "parameter_names": [
                "history_steps",
                "ridge_alpha",
                "air_temperature_residual_scale",
                "relative_humidity_residual_scale",
                "co2_concentration_residual_scale",
            ],
        }
        complete_blueprint = {
            **blueprint,
            "evidence_refs": ["knowledge:executable", "openalex:W123"],
            "rationale": "translate the retrieved direction into the host graph",
        }
        self.server.responses.extend(  # type: ignore[attr-defined]
            [
                (
                    200,
                    _completion(
                        {
                            "algorithm_blueprint": {
                                **blueprint,
                                "evidence_refs": [
                                    "knowledge:executable",
                                    "knowledge:catalog-direction",
                                ],
                            },
                            "algorithm_synthesis": {
                                "schema_version": (
                                    "ecologyrsi-dsh.algorithm-synthesis/1"
                                ),
                                "pipeline_id": "greenhouse-targetwise-ridge@1",
                                "evidence_refs": [
                                    "knowledge:catalog-direction"
                                ],
                                "parameter_focus": ["ridge_alpha"],
                                "rationale": "uses only the catalog direction",
                            },
                        }
                    ),
                ),
                (
                    200,
                    _completion(
                        {
                            "algorithm_blueprint": complete_blueprint,
                            "algorithm_synthesis": {
                                "schema_version": (
                                    "ecologyrsi-dsh.algorithm-synthesis/1"
                                ),
                                "pipeline_id": "greenhouse-targetwise-ridge@1",
                                "evidence_refs": ["openalex:W123"],
                                "parameter_focus": ["ridge_alpha"],
                                "rationale": (
                                    "apply the frozen online shrinkage finding to "
                                    "the registered ridge parameter"
                                ),
                            },
                        }
                    ),
                ),
            ]
        )

        result = gateway.research_plan(
            "policy-main",
            {
                "algorithm_synthesis_requirement": {
                    "mode": "degradation_required"
                },
                "algorithm_blueprint_catalog": [blueprint],
                "knowledge_snapshot": {
                    "evidence_catalog": [
                        {
                            "knowledge_id": "knowledge:executable",
                            "execution_status": "available_not_selected",
                            "capability_kind": "predictor",
                            "capability_id": "greenhouse-targetwise-ridge@1",
                            "capability_ids": [],
                        },
                        {
                            "knowledge_id": "knowledge:catalog-direction",
                            "execution_status": "research_only",
                            "capability_kind": None,
                            "capability_ids": [],
                            "evidence_digest": "a" * 64,
                        },
                        {
                            "knowledge_id": "openalex:W123",
                            "execution_status": "metadata_only",
                            "capability_kind": None,
                            "capability_ids": [],
                            "evidence_digest": "b" * 64,
                        },
                    ]
                },
            },
        )

        self.assertEqual(
            result["algorithm_synthesis"]["evidence_refs"],
            ["openalex:W123"],
        )
        self.assertEqual(len(self.server.requests), 2)  # type: ignore[attr-defined]
        self.assertEqual(
            [request["body"]["max_tokens"] for request in self.server.requests],  # type: ignore[attr-defined]
            [16384, 16384],
        )
        first_message = json.loads(  # type: ignore[attr-defined]
            self.server.requests[0]["body"]["messages"][1]["content"]
        )
        requirement = first_message["input"]["context"][
            "algorithm_synthesis_requirement"
        ]
        self.assertEqual(requirement["mode"], "synthesis_required")
        self.assertEqual(
            requirement["compatible_options"],
            [
                {
                    "pipeline_id": "greenhouse-targetwise-ridge@1",
                    "evidence_refs": ["knowledge:executable"],
                }
            ],
        )
        self.assertEqual(
            [
                item["knowledge_id"]
                for item in requirement["required_research_evidence_options"]
            ],
            ["openalex:W123"],
        )
        second_message = json.loads(  # type: ignore[attr-defined]
            self.server.requests[1]["body"]["messages"][1]["content"]
        )
        self.assertEqual(
            second_message["input"]["context"][
                "algorithm_synthesis_correction"
            ]["attempt"],
            2,
        )

    def test_research_plan_requires_explicit_degradation_without_compatible_evidence(
        self,
    ) -> None:
        gateway = self.gateway()
        self.server.responses.extend(  # type: ignore[attr-defined]
            [
                (200, _completion({})),
                (
                    200,
                    _completion(
                        {
                            "algorithm_synthesis_degradation": {
                                "schema_version": (
                                    "ecologyrsi-dsh.algorithm-synthesis-degradation/1"
                                ),
                                "reason_code": (
                                    "no_compatible_executable_evidence"
                                ),
                                "rationale": (
                                    "the frozen snapshot has research metadata but no "
                                    "host-executable predictor mapping"
                                ),
                            }
                        }
                    ),
                ),
            ]
        )

        result = gateway.research_plan(
            "policy-main",
            {
                "algorithm_blueprint_catalog": [],
                "knowledge_snapshot": {
                    "evidence_catalog": [
                        {
                            "knowledge_id": "openalex:W999",
                            "execution_status": "metadata_only",
                            "capability_kind": None,
                            "capability_ids": [],
                        }
                    ]
                },
            },
        )

        self.assertEqual(
            result["algorithm_synthesis_degradation"]["reason_code"],
            "no_compatible_executable_evidence",
        )
        self.assertEqual(len(self.server.requests), 2)  # type: ignore[attr-defined]

    def test_research_plan_semantic_retry_exhaustion_is_a_contract_failure(
        self,
    ) -> None:
        gateway = self.gateway()
        blueprint = {
            "schema_version": "ecologyrsi-dsh.algorithm-blueprint-request/1",
            "pipeline_id": "greenhouse-exogenous-ridge@1",
            "operator_ids": [
                "host.feature.causal-lag-exogenous@1",
                "host.fit.partition-statistics@1",
                "host.fit.closed-form-ridge@1",
                "host.predictor.residual-correction@1",
                "host.postprocess.physical-bounds@1",
            ],
            "parameter_names": [
                "history_steps",
                "ridge_alpha",
                "residual_scale",
            ],
        }
        self.server.responses.extend(  # type: ignore[attr-defined]
            [(200, _completion({})), (200, _completion({}))]
        )

        with self.assertRaises(GatewayResponseError) as raised:
            gateway.research_plan(
                "policy-main",
                {
                    "algorithm_blueprint_catalog": [blueprint],
                    "knowledge_snapshot": {
                        "evidence_catalog": [
                            {
                                "knowledge_id": "knowledge:ridge",
                                "execution_status": "adopted",
                                "capability_kind": "predictor",
                                "capability_ids": [
                                    "greenhouse-exogenous-ridge@1"
                                ],
                            }
                        ]
                    },
                },
            )

        self.assertIn("after 2 semantic attempts", str(raised.exception))
        self.assertTrue(raised.exception.retryable)
        self.assertEqual(
            raised.exception.error_code,
            "research_algorithm_contract_invalid",
        )
        self.assertEqual(raised.exception.attempts, 2)
        self.assertEqual(len(self.server.requests), 2)  # type: ignore[attr-defined]

    def test_connection_verification_requires_credentials_and_valid_contract(self) -> None:
        gateway = self.gateway()
        self.server.responses.append((200, _completion({"ok": True})))  # type: ignore[attr-defined]

        status = gateway.verify_connection("policy-main")

        self.assertEqual(status["state"], "available")
        self.assertTrue(gateway.catalog()[0]["authentication_verified"])
        request = self.server.requests[0]  # type: ignore[attr-defined]
        self.assertEqual(request["authorization"], "Bearer top-secret")
        user_message = json.loads(request["body"]["messages"][1]["content"])
        self.assertEqual(user_message["operation"], "connection.verify")

        self.server.responses.append((200, _completion({"ok": False})))  # type: ignore[attr-defined]
        with self.assertRaisesRegex(GatewayResponseError, "must be true"):
            gateway.verify_connection("policy-main")
        self.assertEqual(gateway.connection_status("policy-main")["state"], "error")
        failed = gateway.catalog()[0]
        self.assertFalse(failed["authentication_verified"])
        self.assertFalse(failed["authenticated"])
        self.assertFalse(failed["available"])
        self.assertEqual(failed["authentication_state"], "verification_failed")

        without_token = ModelGateway.from_env(
            {
                "ECOLOGYRSI_DSH_MODELS_JSON": json.dumps(
                    [
                        {
                            "id": "no-token",
                            "gateway_url": self.base_url,
                            "model": "local-policy",
                        }
                    ]
                )
            }
        )
        with self.assertRaisesRegex(GatewayConfigurationError, "no authentication credential"):
            without_token.verify_connection("no-token")

    def test_verification_contract_error_never_persists_remote_field_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = EventLedger(Path(directory) / "verification.sqlite3")
            try:
                gateway = self.gateway(verification_store=ledger)
                secret_marker = "catalog-audit-secret-marker"
                self.server.responses.append(  # type: ignore[attr-defined]
                    (
                        200,
                        _completion(
                            {
                                "ok": True,
                                f"Bearer {secret_marker}": "remote-controlled-field",
                            }
                        ),
                    )
                )

                with self.assertRaises(GatewayResponseError):
                    gateway.verify_connection("policy-main")

                connection = gateway._connection("policy-main")
                persisted = ledger.model_verification(
                    "policy-main",
                    connection.configuration_digest,
                    connection.credential_fingerprint,
                )
                rendered = json.dumps(
                    {"catalog": gateway.catalog(), "persisted": persisted},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                self.assertNotIn(secret_marker, rendered)
                self.assertEqual(
                    gateway.connection_status("policy-main")["last_error"],
                    "GatewayResponseError",
                )
            finally:
                ledger.close()

    def test_response_wrappers_are_removed_before_allowed_parameter_validation(self) -> None:
        gateway = self.gateway()
        schema = {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}}
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    "<think>private reasoning with {\"ignored\": true}</think>\n"
                    "Here is the bounded result:\n```json\n"
                    "{\"parameters\": {\"alpha\": 0.4}}\n```"
                ),
            )
        )
        self.assertEqual(
            gateway.propose("policy-main", {}, schema),
            {"parameters": {"alpha": 0.4}},
        )
        self.assertEqual(gateway.connection_status("policy-main")["state"], "available")

        self.server.responses.append(  # type: ignore[attr-defined]
            (200, _completion("{\"parameters\": {\"alpha\": 0.4}} {\"extra\": true}"))
        )
        gateway.max_attempts = 1
        with self.assertRaisesRegex(GatewayResponseError, "one JSON object"):
            gateway.propose("policy-main", {}, schema)

        self.server.responses.append(  # type: ignore[attr-defined]
            (200, _completion({"parameters": {"alpha": 0.4, "command": "run"}}))
        )
        with self.assertRaisesRegex(GatewayResponseError, "unsupported names"):
            gateway.propose("policy-main", {}, schema)

        self.server.responses.append(  # type: ignore[attr-defined]
            (200, _completion({"parameters": {"alpha": 2.0}}))
        )
        with self.assertRaisesRegex(GatewayResponseError, "outside the allowed range"):
            gateway.propose("policy-main", {}, schema)

    def test_sample_decide_accepts_wrapped_final_content_and_ignores_reasoning(self) -> None:
        gateway = self.gateway()
        gateway.max_attempts = 1
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": (
                                    "Final decision:\n```json\n"
                                    '{"decisions":[{"sample_id":"sample-1",'
                                    '"next_tool":"persistence_baseline",'
                                    '"reason_code":"registered_tool",'
                                    '"confidence":1.0}]}\n```'
                                ),
                                "reasoning_content": (
                                    '{"decisions":[{"sample_id":"sample-1",'
                                    '"next_tool":"forbidden_private_tool",'
                                    '"reason_code":"private_decoy",'
                                    '"confidence":0.0}]}'
                                ),
                            },
                        }
                    ]
                },
            )
        )

        result = gateway.sample_decide(
            "policy-main",
            role="planner",
            samples=[{"sample_id": "sample-1"}],
            context={},
            available_tools=[
                {"tool_id": "persistence_baseline", "version": "1"}
            ],
        )

        self.assertEqual(result["decisions"][0]["reason_code"], "registered_tool")
        self.assertEqual(result["decisions"][0]["next_tool"], "persistence_baseline")
        self.assertEqual(len(self.server.requests), 1)  # type: ignore[attr-defined]

    def test_sample_decide_repairs_only_the_known_redundant_object_prefix(self) -> None:
        gateway = self.gateway()
        gateway.max_attempts = 1
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    '{"{"decisions":[{"sample_id":"sample-1",'
                    '"next_tool":"persistence_baseline",'
                    '"reason_code":"registered_tool",'
                    '"confidence":0.92}]}'
                ),
            )
        )

        result = gateway.sample_decide(
            "policy-main",
            role="planner",
            samples=[{"sample_id": "sample-1"}],
            context={},
            available_tools=[
                {"tool_id": "persistence_baseline", "version": "1"}
            ],
        )

        self.assertEqual(result["decisions"][0]["reason_code"], "registered_tool")
        self.assertEqual(result["decisions"][0]["next_tool"], "persistence_baseline")
        self.assertEqual(len(self.server.requests), 1)  # type: ignore[attr-defined]

    def test_sample_decide_rejects_non_object_final_content_boundaries(self) -> None:
        gateway = self.gateway()
        gateway.max_attempts = 1
        valid_decision = (
            '{"decisions":[{"sample_id":"sample-1",'
            '"next_tool":"persistence_baseline",'
            '"reason_code":"candidate",'
            '"confidence":1.0}]}'
        )
        cases = (
            ("multiple_objects", valid_decision + ' {"extra":true}', None),
            (
                "truncated_outer_object",
                'Result: {"wrapper":' + valid_decision,
                None,
            ),
            ("array_wrapper", "[" + valid_decision + "]", None),
            ("reasoning_only", "not-json", valid_decision),
        )

        for case_name, content, reasoning_content in cases:
            with self.subTest(case=case_name):
                message: dict[str, object] = {
                    "role": "assistant",
                    "content": content,
                }
                if reasoning_content is not None:
                    message["reasoning_content"] = reasoning_content
                self.server.responses.append(  # type: ignore[attr-defined]
                    (
                        200,
                        {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": message,
                                }
                            ]
                        },
                    )
                )

                with self.assertRaises(GatewayResponseError) as raised:
                    gateway.sample_decide(
                        "policy-main",
                        role="planner",
                        samples=[{"sample_id": "sample-1"}],
                        context={},
                        available_tools=[
                            {
                                "tool_id": "persistence_baseline",
                                "version": "1",
                            }
                        ],
                    )

                self.assertEqual(
                    raised.exception.error_code,
                    "response_format_invalid",
                )

    def test_non_sample_output_format_failure_retries_within_gateway_budget(self) -> None:
        gateway = self.gateway()
        gateway.max_attempts = 2
        schema = {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}}
        self.server.responses.extend(  # type: ignore[attr-defined]
            [
                (200, _completion("I could not serialize the requested object.")),
                (200, _completion({"parameters": {"alpha": 0.4}})),
            ]
        )

        result = gateway.propose("policy-main", {}, schema)

        self.assertEqual(result, {"parameters": {"alpha": 0.4}})
        self.assertEqual(len(self.server.requests), 2)  # type: ignore[attr-defined]
        diagnostic = gateway.connection_status("policy-main")["last_request"]
        self.assertEqual(diagnostic["attempts"], 2)
        self.assertEqual(diagnostic["retry_count"], 1)
        self.assertEqual(diagnostic["classification"], "recovered_transient")
        first_system = self.server.requests[0]["body"]["messages"][0]["content"]  # type: ignore[attr-defined]
        retry_system = self.server.requests[1]["body"]["messages"][0]["content"]  # type: ignore[attr-defined]
        self.assertNotIn("complete, compact JSON object", first_system)
        self.assertIn("complete, compact JSON object", retry_system)

    def test_research_plan_escalates_output_budget_after_truncation(self) -> None:
        gateway = self.gateway()
        private_marker = "PRIVATE-RESEARCH-REASONING-MUST-NOT-BE-READ"
        self.server.responses.extend(  # type: ignore[attr-defined]
            [
                (
                    200,
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {
                                    "role": "assistant",
                                    "content": '{"implementation_notes":"cut',
                                    "reasoning_content": private_marker,
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 300,
                            "completion_tokens": 16384,
                            "total_tokens": 16684,
                        },
                    },
                ),
                (200, _completion({"implementation_notes": "compact result"})),
            ]
        )

        result = gateway.research_plan(
            "policy-main",
            {"domain": "greenhouse"},
        )

        self.assertEqual(result, {"implementation_notes": "compact result"})
        self.assertEqual(len(self.server.requests), 2)  # type: ignore[attr-defined]
        self.assertEqual(
            [request["body"]["max_tokens"] for request in self.server.requests],  # type: ignore[attr-defined]
            [16384, 32768],
        )
        first_system = self.server.requests[0]["body"]["messages"][0]["content"]  # type: ignore[attr-defined]
        retry_system = self.server.requests[1]["body"]["messages"][0]["content"]  # type: ignore[attr-defined]
        self.assertNotIn("complete, compact JSON object", first_system)
        self.assertIn("complete, compact JSON object", retry_system)
        self.assertNotIn(private_marker, json.dumps(result, ensure_ascii=False))
        self.assertNotIn(
            private_marker,
            json.dumps(gateway.connection_status("policy-main"), ensure_ascii=False),
        )

    def test_research_plan_does_not_repeat_truncation_at_maximum_cap(self) -> None:
        gateway = self.gateway()
        gateway.max_attempts = 4
        self.server.responses.extend(  # type: ignore[attr-defined]
            [
                (
                    200,
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {
                                    "role": "assistant",
                                    "content": "{",
                                },
                            }
                        ]
                    },
                ),
                (
                    200,
                    {
                        "choices": [
                            {
                                "finish_reason": "max_tokens",
                                "message": {
                                    "role": "assistant",
                                    "content": "{",
                                },
                            }
                        ]
                    },
                ),
            ]
        )

        with self.assertRaises(GatewayResponseError) as raised:
            gateway.research_plan("policy-main", {"domain": "greenhouse"})

        self.assertEqual(raised.exception.error_code, "output_truncated")
        self.assertEqual(raised.exception.finish_reason, "max_tokens")
        self.assertEqual(raised.exception.attempts, 2)
        self.assertEqual(len(self.server.requests), 2)  # type: ignore[attr-defined]
        self.assertEqual(
            [request["body"]["max_tokens"] for request in self.server.requests],  # type: ignore[attr-defined]
            [16384, 32768],
        )
        diagnostic = gateway.connection_status("policy-main")["last_request"]
        self.assertEqual(diagnostic["attempts"], 2)
        self.assertEqual(diagnostic["retry_count"], 1)

    def test_complete_sample_json_syntax_failure_retries_within_gateway(self) -> None:
        gateway = self.gateway()
        gateway.max_attempts = 2
        self.server.responses.extend(  # type: ignore[attr-defined]
            [
                (200, _completion("not-json")),
                (
                    200,
                    _completion(
                        {
                            "decisions": [
                                {
                                    "sample_id": "sample-1",
                                    "next_tool": "persistence_baseline",
                                    "reason_code": "bounded_retry_recovered",
                                    "confidence": 1.0,
                                }
                            ]
                        }
                    ),
                ),
            ]
        )

        result = gateway.sample_decide(
            "policy-main",
            role="planner",
            samples=[{"sample_id": "sample-1"}],
            context={},
            available_tools=[
                {"tool_id": "persistence_baseline", "version": "1"}
            ],
        )

        self.assertEqual(result["decisions"][0]["sample_id"], "sample-1")
        self.assertEqual(len(self.server.requests), 2)  # type: ignore[attr-defined]
        diagnostic = gateway.connection_status("policy-main")["last_request"]
        self.assertEqual(diagnostic["attempts"], 2)
        self.assertEqual(diagnostic["retry_count"], 1)
        self.assertEqual(diagnostic["classification"], "recovered_transient")
        system_message = self.server.requests[-1]["body"]["messages"][0]["content"]  # type: ignore[attr-defined]
        self.assertIn("exact ASCII field names", system_message)

    def test_sample_json_syntax_retry_has_an_independent_hard_limit(self) -> None:
        gateway = self.gateway()
        gateway.max_attempts = 8
        self.server.responses.extend(  # type: ignore[attr-defined]
            [(200, _completion("not-json")) for _index in range(8)]
        )

        with self.assertRaises(GatewayResponseError) as raised:
            gateway.sample_decide(
                "policy-main",
                role="planner",
                samples=[{"sample_id": "sample-1"}],
                context={},
                available_tools=[
                    {"tool_id": "persistence_baseline", "version": "1"}
                ],
            )

        self.assertEqual(raised.exception.error_code, "response_format_invalid")
        self.assertEqual(raised.exception.attempts, 2)
        self.assertEqual(len(self.server.requests), 2)  # type: ignore[attr-defined]

    def test_sample_diagnostics_report_zero_attempts_for_preflight_failure(self) -> None:
        gateway = ModelGateway(
            [
                ModelConnection(
                    model_id="judge-only",
                    gateway_url=self.base_url,
                    model="local-judge",
                    token="top-secret",
                    roles=("judge",),
                )
            ],
            retry_base_seconds=0,
            retry_max_seconds=0,
        )
        opener = _AlwaysTimeoutOpener()
        gateway._opener = opener  # type: ignore[assignment, attr-defined]

        with self.assertRaises(GatewayConfigurationError) as raised:
            gateway.sample_decide_with_diagnostics(
                "judge-only",
                role="planner",
                samples=[{"sample_id": "sample-1"}],
                context={},
                available_tools=[
                    {"tool_id": "persistence_baseline", "version": "1"}
                ],
                allow_format_retry=True,
            )

        self.assertEqual(opener.calls, 0)
        self.assertEqual(raised.exception.http_attempts, 0)  # type: ignore[attr-defined]

    def test_sample_diagnostics_report_transport_attempts_on_success_and_failure(
        self,
    ) -> None:
        success = self.gateway()
        success.max_attempts = 4
        success._opener = _SequenceOpener(  # type: ignore[assignment, attr-defined]
            TimeoutError("first timeout"),
            TimeoutError("second timeout"),
            _completion(
                {
                    "decisions": [
                        {
                            "sample_id": "sample-success",
                            "next_tool": "persistence_baseline",
                            "reason_code": "route",
                            "confidence": 1.0,
                        }
                    ]
                }
            ),
        )
        result, receipt = success.sample_decide_with_diagnostics(
            "policy-main",
            role="planner",
            samples=[{"sample_id": "sample-success"}],
            context={},
            available_tools=[
                {"tool_id": "persistence_baseline", "version": "1"}
            ],
            allow_format_retry=True,
        )
        self.assertEqual(result["decisions"][0]["sample_id"], "sample-success")
        self.assertEqual(
            receipt,
            {
                "http_attempts": 3,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "usage_reported": False,
            },
        )

        failure = self.gateway()
        failure.max_attempts = 4
        failure_opener = _AlwaysTimeoutOpener()
        failure._opener = failure_opener  # type: ignore[assignment, attr-defined]
        with self.assertRaises(GatewayResponseError) as raised:
            failure.sample_decide_with_diagnostics(
                "policy-main",
                role="planner",
                samples=[{"sample_id": "sample-failure"}],
                context={},
                available_tools=[
                    {"tool_id": "persistence_baseline", "version": "1"}
                ],
                allow_format_retry=True,
            )
        self.assertEqual(failure_opener.calls, 4)
        self.assertEqual(raised.exception.http_attempts, 4)  # type: ignore[attr-defined]

    def test_sample_diagnostic_attempt_receipts_are_call_local_under_concurrency(
        self,
    ) -> None:
        gateway = self.gateway()
        gateway.max_attempts = 4
        opener = _ConcurrentSampleOpener(
            {"sample-fast": 1, "sample-retry": 3}
        )
        gateway._opener = opener  # type: ignore[assignment, attr-defined]

        def decide(sample_id: str) -> dict[str, int]:
            _result, receipt = gateway.sample_decide_with_diagnostics(
                "policy-main",
                role="planner",
                samples=[{"sample_id": sample_id}],
                context={},
                available_tools=[
                    {"tool_id": "persistence_baseline", "version": "1"}
                ],
                allow_format_retry=True,
            )
            return receipt

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                sample_id: executor.submit(decide, sample_id)
                for sample_id in opener.attempts_by_sample_id
            }
            receipts = {
                sample_id: future.result(timeout=5)
                for sample_id, future in futures.items()
            }

        self.assertEqual(
            receipts,
            {
                "sample-fast": {
                    "http_attempts": 1,
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "usage_reported": True,
                },
                "sample-retry": {
                    "http_attempts": 3,
                    "prompt_tokens": 29,
                    "completion_tokens": 13,
                    "total_tokens": 42,
                    # Two timed-out attempts have no provider usage envelope,
                    # so the successful response is only a measured lower bound.
                    "usage_reported": False,
                },
            },
        )
        self.assertEqual(
            opener.calls_by_sample_id,
            {
                sample_id: receipt["http_attempts"]
                for sample_id, receipt in receipts.items()
            },
        )

    def test_sample_diagnostics_aggregate_usage_across_format_repair(self) -> None:
        gateway = self.gateway()
        first = _completion('{"decisions": [')
        first["usage"] = {
            "prompt_tokens": 101,
            "completion_tokens": 17,
            "total_tokens": 118,
        }
        second = _completion(
            {
                "decisions": [
                    {
                        "sample_id": "sample-repair",
                        "next_tool": "persistence_baseline",
                        "reason_code": "repair",
                        "confidence": 0.8,
                    }
                ]
            }
        )
        second["usage"] = {
            "prompt_tokens": 103,
            "completion_tokens": 19,
            "total_tokens": 122,
        }
        gateway._opener = _SequenceOpener(first, second)  # type: ignore[assignment, attr-defined]

        _result, receipt = gateway.sample_decide_with_diagnostics(
            "policy-main",
            role="critic",
            samples=[{"sample_id": "sample-repair"}],
            context={},
            available_tools=[
                {"tool_id": "persistence_baseline", "version": "1"}
            ],
            allow_format_retry=True,
        )

        self.assertEqual(
            receipt,
            {
                "http_attempts": 2,
                "prompt_tokens": 204,
                "completion_tokens": 36,
                "total_tokens": 240,
                "usage_reported": True,
            },
        )

    def test_sample_diagnostics_return_normal_response_usage(self) -> None:
        gateway = self.gateway()
        completion = _completion(
            {
                "decisions": [
                    {
                        "sample_id": "sample-usage",
                        "next_tool": "persistence_baseline",
                        "reason_code": "route",
                        "confidence": 1.0,
                    }
                ]
            }
        )
        completion["usage"] = {
            "prompt_tokens": 41,
            "completion_tokens": 23,
            "total_tokens": 64,
            "private_usage": "must not appear in the receipt",
        }
        gateway._opener = _SequenceOpener(completion)  # type: ignore[assignment, attr-defined]

        _result, receipt = gateway.sample_decide_with_diagnostics(
            "policy-main",
            role="planner",
            samples=[{"sample_id": "sample-usage"}],
            context={},
            available_tools=[
                {"tool_id": "persistence_baseline", "version": "1"}
            ],
            allow_format_retry=True,
        )

        self.assertEqual(
            receipt,
            {
                "http_attempts": 1,
                "prompt_tokens": 41,
                "completion_tokens": 23,
                "total_tokens": 64,
                "usage_reported": True,
            },
        )

    def test_sample_decision_reason_is_mapped_to_a_host_owned_code(self) -> None:
        gateway = self.gateway()
        reflected_secret = "Bearer top-secret"
        self.server.responses.append(  # type: ignore[attr-defined]
            (
                200,
                _completion(
                    {
                        "decisions": [
                            {
                                "sample_id": "sample-1",
                                "next_tool": "persistence_baseline",
                                "reason_code": reflected_secret,
                                "confidence": 1.0,
                            }
                        ]
                    }
                ),
            )
        )

        result = gateway.sample_decide(
            "policy-main",
            role="planner",
            samples=[{"sample_id": "sample-1"}],
            context={},
            available_tools=[
                {"tool_id": "persistence_baseline", "version": "1"}
            ],
        )

        self.assertEqual(
            result["decisions"][0]["reason_code"], "remote_reason_invalid"
        )
        self.assertNotIn(reflected_secret, json.dumps(result))

    def test_transient_http_status_retries_once_but_bad_request_does_not(self) -> None:
        gateway = self.gateway()
        schema = {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}}
        self.server.responses.extend(  # type: ignore[attr-defined]
            [
                (503, {"error": "temporary"}),
                (200, _completion({"parameters": {"alpha": 0.4}})),
            ]
        )
        self.assertEqual(
            gateway.propose("policy-main", {}, schema),
            {"parameters": {"alpha": 0.4}},
        )
        self.assertEqual(len(self.server.requests), 2)  # type: ignore[attr-defined]

        self.server.responses.extend(  # type: ignore[attr-defined]
            [
                (400, {"error": "invalid request"}),
                (200, _completion({"parameters": {"alpha": 0.4}})),
            ]
        )
        with self.assertRaisesRegex(GatewayResponseError, "HTTP 400"):
            gateway.propose("policy-main", {}, schema)
        self.assertEqual(len(self.server.requests), 3)  # type: ignore[attr-defined]
        self.assertEqual(
            gateway.connection_status("policy-main")["last_request"]["attempts"],
            1,
        )

    def test_retry_after_and_exponential_backoff_are_bounded_and_audited(self) -> None:
        gateway = self.gateway()
        gateway.max_attempts = 4
        gateway.retry_base_seconds = 1.0
        gateway.retry_max_seconds = 10.0
        delays: list[float] = []
        retry_snapshots: list[dict[str, object]] = []

        def record_sleep(delay: float) -> None:
            delays.append(delay)
            retry_snapshots.append(
                dict(gateway.connection_status("policy-main")["last_request"])
            )

        gateway._sleep = record_sleep  # type: ignore[method-assign, attr-defined]
        gateway._random = lambda: 0.5  # type: ignore[method-assign, attr-defined]
        gateway._opener = _SequenceOpener(  # type: ignore[assignment, attr-defined]
            HTTPError(
                self.base_url,
                429,
                "queued",
                {"Retry-After": "7"},
                None,
            ),
            HTTPError(self.base_url, 507, "temporary", {}, None),
            _completion({"parameters": {"alpha": 0.4}}),
        )

        result = gateway.propose(
            "policy-main",
            {},
            {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
        )

        self.assertEqual(result, {"parameters": {"alpha": 0.4}})
        self.assertEqual(delays, [7.0, 2.0])
        self.assertEqual(
            [item["outcome"] for item in retry_snapshots],
            ["retrying", "retrying"],
        )
        self.assertEqual(
            [item["classification"] for item in retry_snapshots],
            ["transient", "transient"],
        )
        self.assertEqual(
            [item["next_retry_seconds"] for item in retry_snapshots],
            [7.0, 2.0],
        )
        status = gateway.connection_status("policy-main")
        self.assertEqual(status["state"], "available")
        self.assertEqual(status["request_policy"]["timeout_seconds"], 2.0)
        self.assertEqual(status["last_request"]["attempts"], 3)
        self.assertEqual(status["last_request"]["retry_count"], 2)
        self.assertEqual(status["last_request"]["outcome"], "succeeded")
        self.assertEqual(
            status["last_request"]["classification"], "recovered_transient"
        )
        self.assertIsNone(status["last_request"]["next_retry_seconds"])
        self.assertEqual(
            [item["reason"] for item in status["last_request"]["retries"]],
            ["HTTP 429", "HTTP 507"],
        )

    def test_long_retry_after_is_handed_off_without_blocking_direct_caller(
        self,
    ) -> None:
        gateway = self.gateway()
        gateway.max_attempts = 4
        gateway.retry_base_seconds = 1.0
        gateway.retry_max_seconds = 10.0
        delays: list[float] = []
        gateway._sleep = delays.append  # type: ignore[method-assign, attr-defined]
        gateway._random = lambda: 0.5  # type: ignore[method-assign, attr-defined]
        opener = _SequenceOpener(
            HTTPError(
                self.base_url,
                429,
                "queued",
                {"Retry-After": "90"},
                None,
            ),
            _completion({"parameters": {"alpha": 0.4}}),
        )
        gateway._opener = opener  # type: ignore[assignment, attr-defined]

        with self.assertRaises(GatewayResponseError) as raised:
            gateway.propose(
                "policy-main",
                {},
                {
                    "alpha": {
                        "type": "number",
                        "minimum": 0.05,
                        "maximum": 0.95,
                    }
                },
            )

        error = raised.exception
        self.assertTrue(error.retryable)
        self.assertEqual(error.attempts, 1)
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.retry_after_seconds, 90.0)
        self.assertIsInstance(error.__cause__, HTTPError)
        self.assertEqual(delays, [])
        self.assertEqual(opener.timeouts, [2.0])
        diagnostic = gateway.connection_status("policy-main")["last_request"]
        self.assertEqual(diagnostic["outcome"], "failed")
        self.assertEqual(diagnostic["classification"], "transient")
        self.assertEqual(diagnostic["attempts"], 1)
        self.assertEqual(diagnostic["next_retry_seconds"], 90.0)
        self.assertEqual(diagnostic["retry_count"], 0)
        self.assertEqual(diagnostic["retries"], [])
        self.assertEqual(
            gateway.connection_status("policy-main")["request_policy"][
                "inline_retry_after_cap_seconds"
            ],
            15.0,
        )

    def test_authentication_error_never_hands_off_retry_after(self) -> None:
        gateway = self.gateway()
        delays: list[float] = []
        gateway._sleep = delays.append  # type: ignore[method-assign, attr-defined]
        opener = _SequenceOpener(
            HTTPError(
                self.base_url,
                401,
                "unauthorized",
                {"Retry-After": "90"},
                None,
            ),
            _completion({"parameters": {"alpha": 0.4}}),
        )
        gateway._opener = opener  # type: ignore[assignment, attr-defined]

        with self.assertRaises(GatewayResponseError) as raised:
            gateway.propose(
                "policy-main",
                {},
                {
                    "alpha": {
                        "type": "number",
                        "minimum": 0.05,
                        "maximum": 0.95,
                    }
                },
            )

        self.assertFalse(raised.exception.retryable)
        self.assertIsNone(raised.exception.retry_after_seconds)
        self.assertEqual(delays, [])
        self.assertEqual(opener.timeouts, [2.0])
        diagnostic = gateway.connection_status("policy-main")["last_request"]
        self.assertEqual(diagnostic["classification"], "permanent")
        self.assertIsNone(diagnostic["next_retry_seconds"])

    def test_gateway_error_wrapper_preserves_retry_after_hint(self) -> None:
        original = GatewayResponseError(
            "provider queue is busy",
            retryable=True,
            status_code=429,
            retry_after_seconds=90,
        )

        wrapped = model_gateway_module._wrap_gateway_error(
            "bounded wrapper",
            original,
            error_code="wrapped_gateway_error",
        )

        self.assertTrue(wrapped.retryable)
        self.assertEqual(wrapped.status_code, 429)
        self.assertEqual(wrapped.retry_after_seconds, 90.0)

    def test_first_live_request_is_audited_before_network_call(self) -> None:
        gateway = self.gateway()
        opener = _DiagnosticSequenceOpener(
            gateway,
            _completion({"parameters": {"alpha": 0.4}}),
        )
        gateway._opener = opener  # type: ignore[assignment, attr-defined]

        result = gateway.propose(
            "policy-main",
            {"private_prompt": "diagnostic-payload-sentinel"},
            {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
        )

        self.assertEqual(result, {"parameters": {"alpha": 0.4}})
        self.assertEqual(len(opener.snapshots), 1)
        diagnostic = opener.snapshots[0]
        self.assertEqual(
            set(diagnostic),
            {
                "operation",
                "started_at",
                "updated_at",
                "outcome",
                "attempts",
                "retry_count",
                "retries",
                "last_error",
                "next_retry_seconds",
                "classification",
                "timeout_seconds",
            },
        )
        self.assertEqual(diagnostic["operation"], "propose")
        self.assertEqual(diagnostic["outcome"], "in_progress")
        self.assertEqual(diagnostic["attempts"], 0)
        self.assertEqual(diagnostic["retry_count"], 0)
        self.assertEqual(diagnostic["retries"], [])
        self.assertIsNone(diagnostic["last_error"])
        self.assertIsNone(diagnostic["next_retry_seconds"])
        self.assertEqual(diagnostic["classification"], "none")
        self.assertEqual(diagnostic["timeout_seconds"], 2.0)
        self.assertIsInstance(diagnostic["started_at"], str)
        self.assertIsInstance(diagnostic["updated_at"], str)
        rendered = json.dumps(diagnostic)
        self.assertNotIn("diagnostic-payload-sentinel", rendered)
        self.assertNotIn("top-secret", rendered)
        self.assertNotIn(self.base_url, rendered)

    def test_second_live_request_replaces_previous_success_before_network_call(
        self,
    ) -> None:
        gateway = self.gateway()
        opener = _DiagnosticSequenceOpener(
            gateway,
            _completion({"parameters": {"alpha": 0.4}}),
            _completion({"parameters": {"alpha": 0.5}}),
        )
        gateway._opener = opener  # type: ignore[assignment, attr-defined]
        schema = {
            "alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}
        }

        gateway.propose("policy-main", {}, schema)
        first_terminal = gateway.connection_status("policy-main")["last_request"]
        self.assertEqual(first_terminal["outcome"], "succeeded")

        gateway.propose("policy-main", {}, schema)

        self.assertEqual(len(opener.snapshots), 2)
        second_active = opener.snapshots[1]
        self.assertEqual(second_active["operation"], "propose")
        self.assertEqual(second_active["outcome"], "in_progress")
        self.assertEqual(second_active["attempts"], 0)
        self.assertEqual(second_active["retry_count"], 0)
        self.assertEqual(second_active["retries"], [])

    def test_server_retry_after_is_not_shortened_to_client_backoff_cap(self) -> None:
        delayed = model_gateway_module._gateway_retry_delay(
            HTTPError(
                self.base_url,
                429,
                "queued",
                {"Retry-After": "90"},
                None,
            ),
            attempt=1,
            base_seconds=1.0,
            max_seconds=10.0,
            random_unit=0.5,
        )
        capped = model_gateway_module._gateway_retry_delay(
            HTTPError(
                self.base_url,
                429,
                "queued",
                {"Retry-After": "7200"},
                None,
            ),
            attempt=1,
            base_seconds=1.0,
            max_seconds=10.0,
            random_unit=0.5,
        )

        self.assertEqual(delayed, 90.0)
        self.assertEqual(capped, 3600.0)

    def test_retry_jitter_is_bounded(self) -> None:
        low = model_gateway_module._gateway_retry_delay(
            TimeoutError(),
            attempt=2,
            base_seconds=2.0,
            max_seconds=10.0,
            random_unit=0.0,
        )
        high = model_gateway_module._gateway_retry_delay(
            TimeoutError(),
            attempt=2,
            base_seconds=2.0,
            max_seconds=10.0,
            random_unit=1.0,
        )

        self.assertAlmostEqual(low, 3.2)
        self.assertAlmostEqual(high, 4.8)

    def test_timeout_retries_then_succeeds(self) -> None:
        gateway = self.gateway()
        opener = _TimeoutOnceOpener(_completion({"parameters": {"alpha": 0.4}}))
        gateway._opener = opener  # type: ignore[assignment, attr-defined]
        result = gateway.propose(
            "policy-main",
            {},
            {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
        )
        self.assertEqual(result, {"parameters": {"alpha": 0.4}})
        self.assertEqual(opener.calls, 2)

    def test_transient_connection_and_dns_failures_retry_then_succeed(self) -> None:
        gateway = self.gateway()
        opener = _SequenceOpener(
            ConnectionRefusedError("gateway is restarting"),
            URLError(socket.gaierror(socket.EAI_AGAIN, "temporary DNS failure")),
            _completion({"parameters": {"alpha": 0.4}}),
        )
        gateway._opener = opener  # type: ignore[assignment, attr-defined]

        result = gateway.propose(
            "policy-main",
            {},
            {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
        )

        self.assertEqual(result, {"parameters": {"alpha": 0.4}})
        self.assertEqual(len(opener.timeouts), 3)
        diagnostics = gateway.connection_status("policy-main")["last_request"]
        self.assertEqual(diagnostics["attempts"], 3)
        self.assertEqual(diagnostics["classification"], "recovered_transient")
        self.assertEqual(diagnostics["retry_count"], 2)

    def test_interrupted_response_read_retries_locally_and_is_audited(self) -> None:
        schema = {
            "alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}
        }
        gateway = self.gateway()
        gateway.max_attempts = 3
        recovered_opener = _InterruptedReadOpener(
            _completion({"parameters": {"alpha": 0.4}}),
            failures=2,
        )
        gateway._opener = recovered_opener  # type: ignore[assignment, attr-defined]

        result = gateway.propose("policy-main", {}, schema)

        self.assertEqual(result, {"parameters": {"alpha": 0.4}})
        self.assertEqual(recovered_opener.timeouts, [2.0, 2.0, 2.0])
        recovered = gateway.connection_status("policy-main")["last_request"]
        self.assertEqual(recovered["attempts"], 3)
        self.assertEqual(recovered["retry_count"], 2)
        self.assertEqual(recovered["classification"], "recovered_transient")
        self.assertEqual(
            [item["reason"] for item in recovered["retries"]],
            ["IncompleteRead", "IncompleteRead"],
        )

        exhausted_gateway = self.gateway()
        exhausted_gateway.max_attempts = 2
        exhausted_opener = _InterruptedReadOpener(
            _completion({"parameters": {"alpha": 0.4}}),
            failures=2,
        )
        exhausted_gateway._opener = exhausted_opener  # type: ignore[assignment, attr-defined]

        with self.assertRaises(GatewayResponseError) as raised:
            exhausted_gateway.propose("policy-main", {}, schema)

        self.assertTrue(raised.exception.retryable)
        self.assertEqual(raised.exception.attempts, 2)
        self.assertIsInstance(raised.exception.__cause__, IncompleteRead)
        exhausted = exhausted_gateway.connection_status("policy-main")
        self.assertEqual(exhausted["state"], "configured")
        self.assertEqual(exhausted["last_request"]["attempts"], 2)
        self.assertEqual(exhausted["last_request"]["retry_count"], 1)
        self.assertEqual(exhausted["last_request"]["classification"], "transient")

    def test_malformed_response_body_retries_then_recovers(self) -> None:
        gateway = self.gateway()
        gateway.max_attempts = 3
        valid = json.dumps(
            _completion({"parameters": {"alpha": 0.4}}),
            ensure_ascii=False,
        ).encode("utf-8")
        opener = _RawBodySequenceOpener(
            b"\xff",
            b'{"choices": [',
            valid,
        )
        gateway._opener = opener  # type: ignore[assignment, attr-defined]

        result = gateway.propose(
            "policy-main",
            {},
            {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
        )

        self.assertEqual(result, {"parameters": {"alpha": 0.4}})
        self.assertEqual(opener.timeouts, [2.0, 2.0, 2.0])
        diagnostics = gateway.connection_status("policy-main")["last_request"]
        self.assertEqual(diagnostics["attempts"], 3)
        self.assertEqual(diagnostics["retry_count"], 2)
        self.assertEqual(diagnostics["classification"], "recovered_transient")
        self.assertEqual(
            [item["reason"] for item in diagnostics["retries"]],
            ["UnicodeDecodeError", "JSONDecodeError"],
        )

    def test_malformed_response_body_exhaustion_is_retryable_and_audited(self) -> None:
        malformed_bodies = (
            (b"\xff", UnicodeDecodeError),
            (b'{"choices": [', json.JSONDecodeError),
        )
        for body, error_type in malformed_bodies:
            with self.subTest(error_type=error_type.__name__):
                gateway = self.gateway()
                gateway.max_attempts = 2
                opener = _RawBodySequenceOpener(body, body)
                gateway._opener = opener  # type: ignore[assignment, attr-defined]

                with self.assertRaises(GatewayResponseError) as raised:
                    gateway.propose(
                        "policy-main",
                        {},
                        {
                            "alpha": {
                                "type": "number",
                                "minimum": 0.05,
                                "maximum": 0.95,
                            }
                        },
                    )

                self.assertTrue(raised.exception.retryable)
                self.assertEqual(raised.exception.attempts, 2)
                self.assertIsInstance(raised.exception.__cause__, error_type)
                self.assertIn("retry budget exhausted", str(raised.exception))
                self.assertEqual(opener.timeouts, [2.0, 2.0])
                status = gateway.connection_status("policy-main")
                self.assertEqual(status["state"], "configured")
                diagnostics = status["last_request"]
                self.assertEqual(diagnostics["attempts"], 2)
                self.assertEqual(diagnostics["retry_count"], 1)
                self.assertEqual(diagnostics["outcome"], "failed")
                self.assertEqual(diagnostics["classification"], "transient")
                self.assertEqual(diagnostics["last_error"], error_type.__name__)

    def test_decoded_proposal_response_contract_error_retries_then_recovers(self) -> None:
        gateway = self.gateway()
        invalid_contract = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": 7}}]}
        ).encode("utf-8")
        valid = json.dumps(
            _completion({"parameters": {"alpha": 0.4}})
        ).encode("utf-8")
        opener = _RawBodySequenceOpener(invalid_contract, valid)
        gateway._opener = opener  # type: ignore[assignment, attr-defined]

        result = gateway.propose(
            "policy-main",
            {},
            {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
        )

        self.assertEqual(result, {"parameters": {"alpha": 0.4}})
        self.assertEqual(opener.timeouts, [2.0, 2.0])
        diagnostics = gateway.connection_status("policy-main")["last_request"]
        self.assertEqual(diagnostics["attempts"], 2)
        self.assertEqual(diagnostics["retry_count"], 1)
        self.assertEqual(diagnostics["outcome"], "succeeded")
        self.assertEqual(diagnostics["classification"], "recovered_transient")

    def test_permanent_dns_resolution_failure_is_not_retried(self) -> None:
        gateway = self.gateway()
        opener = _SequenceOpener(
            URLError(socket.gaierror(socket.EAI_NONAME, "unknown host")),
            _completion({"parameters": {"alpha": 0.4}}),
        )
        gateway._opener = opener  # type: ignore[assignment, attr-defined]

        with self.assertRaises(GatewayResponseError) as raised:
            gateway.propose(
                "policy-main",
                {},
                {
                    "alpha": {
                        "type": "number",
                        "minimum": 0.05,
                        "maximum": 0.95,
                    }
                },
            )

        self.assertFalse(raised.exception.retryable)
        self.assertEqual(len(opener.timeouts), 1)

    def test_business_timeout_preserves_explicit_verification(self) -> None:
        gateway = self.gateway()
        self.server.responses.append((200, _completion({"ok": True})))  # type: ignore[attr-defined]
        gateway.verify_connection("policy-main")
        self.assertTrue(gateway.catalog()[0]["authentication_verified"])

        opener = _AlwaysTimeoutOpener()
        gateway._opener = opener  # type: ignore[assignment, attr-defined]
        with self.assertRaisesRegex(GatewayResponseError, "TimeoutError"):
            gateway.propose(
                "policy-main",
                {},
                {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
            )

        self.assertEqual(opener.calls, 4)
        status = gateway.connection_status("policy-main")
        self.assertEqual(status["state"], "configured")
        self.assertIn("retry budget exhausted", status["last_error"])
        self.assertEqual(status["last_request"]["attempts"], 4)
        self.assertEqual(status["last_request"]["retry_count"], 3)
        self.assertEqual(status["last_request"]["outcome"], "failed")
        catalog_item = gateway.catalog()[0]
        self.assertTrue(catalog_item["authentication_verified"])
        self.assertTrue(catalog_item["authenticated"])
        self.assertFalse(catalog_item["available"])
        self.assertEqual(catalog_item["authentication_state"], "verified")
        self.assertEqual(catalog_item["connection_phase"], "transient_error")
        self.assertTrue(catalog_item["temporarily_unavailable"])
        self.assertFalse(catalog_item["api_invalid"])

    def test_transient_timeout_is_not_persisted_as_invalid_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = EventLedger(Path(directory) / "verification.sqlite3")
            try:
                gateway = self.gateway(verification_store=ledger)
                self.server.responses.append((200, _completion({"ok": True})))  # type: ignore[attr-defined]
                gateway.verify_connection("policy-main")
                gateway._opener = _AlwaysTimeoutOpener()  # type: ignore[assignment, attr-defined]
                with self.assertRaises(GatewayResponseError):
                    gateway.propose(
                        "policy-main",
                        {},
                        {
                            "alpha": {
                                "type": "number",
                                "minimum": 0.05,
                                "maximum": 0.95,
                            }
                        },
                    )

                restored = self.gateway(verification_store=ledger).catalog()[0]
                self.assertTrue(restored["authentication_verified"])
                self.assertEqual(restored["authentication_state"], "verified")
                self.assertEqual(restored["connection"]["state"], "configured")
                self.assertIn(
                    "retry budget exhausted", restored["connection"]["last_error"]
                )
            finally:
                ledger.close()

    def test_shared_store_catalog_refreshes_verification_without_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "shared-verification.sqlite3"
            ledger_a = EventLedger(database)
            ledger_b = EventLedger(database)
            try:
                gateway_a = self.gateway(verification_store=ledger_a)
                gateway_b = self.gateway(verification_store=ledger_b)
                self.assertFalse(gateway_a.catalog()[0]["authentication_verified"])
                self.assertFalse(gateway_b.catalog()[0]["authentication_verified"])

                self.server.responses.append((200, _completion({"ok": True})))  # type: ignore[attr-defined]
                gateway_a.verify_connection("policy-main")

                observed_by_b = gateway_b.catalog()[0]
                self.assertTrue(observed_by_b["authentication_verified"])
                self.assertTrue(observed_by_b["available"])
                self.assertTrue(observed_by_b["verification_persisted"])
                self.assertEqual(
                    gateway_b.connection_status("policy-main")["state"],
                    "available",
                )
            finally:
                ledger_b.close()
                ledger_a.close()

    def test_stale_business_status_cannot_downgrade_shared_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "shared-verification.sqlite3"
            ledger_a = EventLedger(database)
            ledger_b = EventLedger(database)
            try:
                gateway_a = self.gateway(verification_store=ledger_a)
                stale_gateway_b = self.gateway(verification_store=ledger_b)
                self.assertFalse(
                    stale_gateway_b.catalog()[0]["authentication_verified"]
                )

                self.server.responses.extend(  # type: ignore[attr-defined]
                    [
                        (200, _completion({"ok": True})),
                        (200, _completion({"parameters": {"alpha": 0.4}})),
                    ]
                )
                gateway_a.verify_connection("policy-main")
                stale_gateway_b.propose(
                    "policy-main",
                    {},
                    {
                        "alpha": {
                            "type": "number",
                            "minimum": 0.05,
                            "maximum": 0.95,
                        }
                    },
                )
                self.assertEqual(
                    stale_gateway_b.connection_status("policy-main")["state"],
                    "available",
                )

                ledger_c = EventLedger(database)
                try:
                    gateway_c = self.gateway(verification_store=ledger_c)
                    observed_by_c = gateway_c.catalog()[0]
                    self.assertTrue(observed_by_c["authentication_verified"])
                    self.assertTrue(observed_by_c["available"])
                    self.assertTrue(observed_by_c["verification_persisted"])
                finally:
                    ledger_c.close()

                refreshed_b = stale_gateway_b.catalog()[0]
                self.assertTrue(refreshed_b["authentication_verified"])
                self.assertTrue(refreshed_b["available"])
                self.assertTrue(gateway_a.catalog()[0]["authentication_verified"])
            finally:
                ledger_b.close()
                ledger_a.close()

    def test_model_timeout_can_be_configured_from_environment(self) -> None:
        catalog = json.dumps(
            [
                {
                    "id": "policy-main",
                    "gateway_url": self.base_url,
                    "model": "local-policy",
                    "token": "top-secret",
                }
            ]
        )
        defaults = ModelGateway.from_env({"ECOLOGYRSI_DSH_MODELS_JSON": catalog})
        self.assertEqual(defaults.timeout, 900.0)
        self.assertEqual(defaults.max_attempts, 4)
        self.assertEqual(defaults.retry_max_seconds, 600.0)
        configured = ModelGateway.from_env(
            {
                "ECOLOGYRSI_DSH_MODELS_JSON": catalog,
                "ECOLOGYRSI_DSH_MODEL_TIMEOUT": "75",
                "ECOLOGYRSI_DSH_MODEL_MAX_ATTEMPTS": "6",
                "ECOLOGYRSI_DSH_MODEL_RETRY_BASE_SECONDS": "2.5",
                "ECOLOGYRSI_DSH_MODEL_RETRY_MAX_SECONDS": "45",
            }
        )
        self.assertEqual(configured.timeout, 75.0)
        self.assertEqual(configured.max_attempts, 6)
        self.assertEqual(configured.retry_base_seconds, 2.5)
        self.assertEqual(configured.retry_max_seconds, 45.0)
        explicit = ModelGateway.from_env(
            {"ECOLOGYRSI_DSH_MODELS_JSON": catalog, "ECOLOGYRSI_DSH_MODEL_TIMEOUT": "75"},
            timeout=2,
        )
        self.assertEqual(explicit.timeout, 2.0)
        with self.assertRaisesRegex(GatewayConfigurationError, "positive number"):
            ModelGateway.from_env(
                {"ECOLOGYRSI_DSH_MODELS_JSON": catalog, "ECOLOGYRSI_DSH_MODEL_TIMEOUT": "0"}
            )
        with self.assertRaisesRegex(GatewayConfigurationError, "between 1 and 8"):
            ModelGateway.from_env(
                {
                    "ECOLOGYRSI_DSH_MODELS_JSON": catalog,
                    "ECOLOGYRSI_DSH_MODEL_MAX_ATTEMPTS": "9",
                }
            )
        with self.assertRaisesRegex(GatewayConfigurationError, "greater than or equal"):
            ModelGateway.from_env(
                {
                    "ECOLOGYRSI_DSH_MODELS_JSON": catalog,
                    "ECOLOGYRSI_DSH_MODEL_RETRY_BASE_SECONDS": "5",
                    "ECOLOGYRSI_DSH_MODEL_RETRY_MAX_SECONDS": "2",
                }
            )

    def test_environment_fallback_and_address_policy(self) -> None:
        gateway = ModelGateway.from_env(
            {
                "ECOLOGYRSI_DSH_GATEWAY_URL": self.base_url,
                "ECOLOGYRSI_DSH_MODEL": "fallback-model",
                "ECOLOGYRSI_DSH_TOKEN": "fallback-secret",
            }
        )
        self.assertEqual(gateway.catalog()[0]["model_id"], "fallback-model")
        self.assertEqual(ModelGateway.from_env({}).catalog(), [])
        with self.assertRaisesRegex(GatewayConfigurationError, "HTTPS or loopback HTTP"):
            ModelGateway.from_env(
                {
                    "ECOLOGYRSI_DSH_GATEWAY_URL": "ftp://example.com/v1",
                    "ECOLOGYRSI_DSH_MODEL": "unsafe",
                }
            )
        with self.assertRaisesRegex(GatewayConfigurationError, "loopback"):
            ModelGateway.from_env(
                {
                    "ECOLOGYRSI_DSH_GATEWAY_URL": "http://example.com/v1",
                    "ECOLOGYRSI_DSH_MODEL": "unsafe",
                }
            )

    def test_catalog_rejects_duplicates_and_missing_credentials(self) -> None:
        duplicate = json.dumps(
            [
                {"id": "same", "gateway_url": self.base_url, "model": "one"},
                {"id": "same", "gateway_url": self.base_url, "model": "two"},
            ]
        )
        with self.assertRaisesRegex(GatewayConfigurationError, "duplicate model_id"):
            ModelGateway.from_env({"ECOLOGYRSI_DSH_MODELS_JSON": duplicate})
        missing = json.dumps(
            [{"id": "one", "gateway_url": self.base_url, "model": "one", "api_key_env": "MISSING"}]
        )
        with self.assertRaisesRegex(GatewayConfigurationError, "not set"):
            ModelGateway.from_env({"ECOLOGYRSI_DSH_MODELS_JSON": missing})

    def test_roles_restrict_model_operations(self) -> None:
        catalog = json.dumps(
            [
                {
                    "id": "judge-only",
                    "label": "仅评审模型",
                    "roles": ["judge"],
                    "gateway_url": self.base_url,
                    "model": "local-judge",
                }
            ]
        )
        gateway = ModelGateway.from_env({"ECOLOGYRSI_DSH_MODELS_JSON": catalog})
        with self.assertRaisesRegex(GatewayConfigurationError, "does not allow"):
            gateway.propose(
                "judge-only",
                {},
                {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
            )

    def test_role_aliases_are_case_insensitive(self) -> None:
        catalog = json.dumps(
            [
                {
                    "id": "mixed-case-roles",
                    "roles": ["PROPOSE", "Reviewer"],
                    "gateway_url": self.base_url,
                    "model": "mixed-case-model",
                }
            ]
        )
        gateway = ModelGateway.from_env({"ECOLOGYRSI_DSH_MODELS_JSON": catalog})
        self.assertEqual(gateway.catalog()[0]["roles"], ["propose", "judge"])

    def test_configuration_digest_excludes_secret_and_binds_callable_identity(self) -> None:
        first = ModelConnection(
            model_id="stable-id",
            gateway_url=self.base_url,
            model="remote-a",
            token="secret-one",
            roles=("judge", "propose"),
        )
        rotated_secret = ModelConnection(
            model_id="stable-id",
            gateway_url=self.base_url,
            model="remote-a",
            token="secret-two",
            roles=("propose", "judge"),
        )
        changed_model = ModelConnection(
            model_id="stable-id",
            gateway_url=self.base_url,
            model="remote-b",
            token="secret-one",
            roles=("judge", "propose"),
        )
        allowlisted_loopback = ModelConnection(
            model_id="stable-id",
            gateway_url=self.base_url,
            model="remote-a",
            token="secret-one",
            roles=("judge", "propose"),
            allow_insecure_http=True,
        )
        https = ModelConnection(
            model_id="https-id",
            gateway_url="https://models.example/v1",
            model="remote-a",
            token="secret-one",
        )
        allowlisted_https = ModelConnection(
            model_id="https-id",
            gateway_url="https://models.example/v1",
            model="remote-a",
            token="secret-one",
            allow_insecure_http=True,
        )

        self.assertEqual(first.configuration_digest, rotated_secret.configuration_digest)
        self.assertNotEqual(first.configuration_digest, changed_model.configuration_digest)
        self.assertFalse(allowlisted_loopback.allow_insecure_http)
        self.assertEqual(first.configuration_digest, allowlisted_loopback.configuration_digest)
        self.assertFalse(allowlisted_https.allow_insecure_http)
        self.assertEqual(https.configuration_digest, allowlisted_https.configuration_digest)
        gateway = ModelGateway((first,))
        self.assertEqual(
            gateway.configuration_digest("stable-id"),
            first.configuration_digest,
        )
        rendered = json.dumps(gateway.catalog())
        self.assertNotIn("secret-one", rendered)

    def test_irrelevant_loopback_allowlist_change_preserves_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = EventLedger(Path(directory) / "verification.sqlite3")
            try:
                original = ModelConnection(
                    model_id="policy-main",
                    gateway_url=self.base_url,
                    model="local-policy",
                    token="top-secret",
                )
                first_gateway = ModelGateway(
                    (original,),
                    timeout=2,
                    verification_store=ledger,
                )
                self.server.responses.append(  # type: ignore[attr-defined]
                    (200, _completion({"ok": True}))
                )
                first_gateway.verify_connection("policy-main")

                allowlisted = ModelConnection(
                    model_id="policy-main",
                    gateway_url=self.base_url,
                    model="local-policy",
                    token="top-secret",
                    allow_insecure_http=True,
                )
                restored_gateway = ModelGateway(
                    (allowlisted,),
                    timeout=2,
                    verification_store=ledger,
                )
                restored = restored_gateway.catalog()[0]
                self.assertEqual(
                    original.configuration_digest,
                    allowlisted.configuration_digest,
                )
                self.assertTrue(restored["authentication_verified"])
                self.assertTrue(restored["available"])
            finally:
                ledger.close()

    def test_redirect_never_forwards_model_bearer(self) -> None:
        redirect = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
        redirect.requests = []  # type: ignore[attr-defined]
        redirect.location = self.base_url + "/chat/completions"  # type: ignore[attr-defined]
        thread = Thread(target=redirect.serve_forever, daemon=True)
        thread.start()
        try:
            gateway_url = f"http://127.0.0.1:{redirect.server_address[1]}/v1"
            catalog = json.dumps(
                [
                    {
                        "id": "redirecting-policy",
                        "gateway_url": gateway_url,
                        "model": "local-policy",
                        "token": "redirect-secret",
                    }
                ]
            )
            gateway = ModelGateway.from_env({"ECOLOGYRSI_DSH_MODELS_JSON": catalog})
            with self.assertRaisesRegex(GatewayResponseError, "HTTP 302"):
                gateway.propose(
                    "redirecting-policy",
                    {},
                    {"alpha": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
                )
            self.assertEqual(redirect.requests, ["Bearer redirect-secret"])  # type: ignore[attr-defined]
            self.assertEqual(self.server.requests, [])  # type: ignore[attr-defined]
        finally:
            redirect.shutdown()
            redirect.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
