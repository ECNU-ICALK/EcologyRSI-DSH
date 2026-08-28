from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecologyrsi_dsh.api.dsh_tools import (
    DshToolAdmissionClosedError,
    DshToolAuthorizationError,
    DshStructuredResultPersistenceError,
    DshToolService,
    ROLE_TOOLS,
)
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.api.handler import EvolutionHTTPServer
from ecologyrsi_dsh.integrations.dsh_structured_roles import DshStructuredRoleRuntime


FUTURE_DEADLINE_UNIX_MS = 4_102_444_800_000


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


def _skill_evidence(stage: str) -> dict:
    skill_by_stage = {
        "generation.research": "autonomous-ecology-research",
        "generation.judge": "candidate-scientific-review",
        "sample.plan": "origin-vector-forecasting-balanced",
        "sample.reflect": "origin-vector-review",
    }
    return {
        "schema_version": "ecologyrsi-dsh.skill-invocation-evidence/1",
        "stage": stage,
        "skill_name": skill_by_stage[stage],
        "call_count": 1,
        "successful_call_count": 1,
        "call_seq": 1,
        "result_seq": 2,
        "first_tool_call_verified": True,
        "next_tool_name": (
            "ecology_execute_prediction_tool"
            if stage == "sample.plan"
            else "structured_output"
        ),
        "next_tool_call_seq": 3,
        "order_verified": True,
        "source": "dsh_session_event_log",
    }


def _research_envelope(ledger: EventLedger, *, deadline_unix_ms: int) -> dict:
    structured = {
        "schema_version": "ecology-research-result@1",
        "summary": "bounded evidence",
        "evidence": [],
    }
    return {
        "identity": _identity(
            ledger,
            role="researcher",
            stage="generation.research",
            idempotency_key="deadline-result",
        ),
        "output_schema_id": "ecology-research-result@1",
        "structured": structured,
        "result_digest": digest(structured),
        "skill_invocation_evidence": _skill_evidence("generation.research"),
        "deadline_unix_ms": deadline_unix_ms,
    }


def _reservation_request(
    admission_id: str,
    *,
    request_id: str = "deadline-reservation-1",
    timeout_ms: int = 1_000,
    sample_member_digests: list[str] | None = None,
) -> dict:
    request = {
        "request_id": request_id,
        "run_id": "run:tool-test",
        "parent_session_id": "session:research-host",
        "role": "researcher",
        "stage": "generation.research",
        "run_state_revision": 3,
        "stage_attempt": 2,
        "admission_id": admission_id,
        "timeout_ms": timeout_ms,
        "item_digest": "d" * 64,
        "idempotency_key": "deadline-result",
    }
    if sample_member_digests is not None:
        request["sample_member_digests"] = list(sample_member_digests)
    return request


def _admitted_research_envelope(
    ledger: EventLedger,
    *,
    admission_id: str,
) -> dict:
    envelope = _research_envelope(ledger, deadline_unix_ms=1)
    envelope.pop("deadline_unix_ms")
    envelope["admission_id"] = admission_id
    return envelope


def _arm_structured_envelope(
    service: DshToolService,
    envelope: dict,
    *,
    timeout_ms: int = 1_000,
) -> object:
    identity = envelope["identity"]
    fence = service.open_admission(
        identity["run_id"],
        identity["run_state_revision"],
        identity["stage_attempt"],
        role=identity["role"],
        stage=identity["stage"],
        idempotency_key=identity["idempotency_key"],
    )
    request_id_digest = digest(
        {
            "run_id": identity["run_id"],
            "stage": identity["stage"],
            "idempotency_key": identity["idempotency_key"],
        }
    )
    service.allocate_child_reservation(
        {
            "request_id": f"arm-{request_id_digest}",
            "run_id": identity["run_id"],
            "parent_session_id": identity["session_id"],
            "role": identity["role"],
            "stage": identity["stage"],
            "run_state_revision": identity["run_state_revision"],
            "stage_attempt": identity["stage_attempt"],
            "admission_id": fence.admission_id,
            "timeout_ms": timeout_ms,
            "item_digest": digest({"stage": identity["stage"]}),
            "idempotency_key": identity["idempotency_key"],
        }
    )
    envelope.pop("deadline_unix_ms", None)
    envelope["admission_id"] = fence.admission_id
    return fence


def _append_recorded_planner_result(
    ledger: EventLedger,
    *,
    case: str,
) -> tuple[dict, object]:
    """Append one durable Planner result, optionally corrupting its binding."""

    idempotency_key = f"restart-planner-{case}"
    prediction_event_digest = digest(
        {
            "idempotency_key": idempotency_key,
            "tool_name": "ecology_execute_prediction_tool",
        }
    )
    prediction_event_id = (
        f"run:tool-test:dsh-prediction-tool:"
        f"{prediction_event_digest}"
    )
    prediction_payload = {
        "schema_version": "ecologyrsi-dsh.dsh-prediction-tool-executed/1",
        "stage": "sample.plan",
        "stage_attempt": 2,
        "idempotency_key": idempotency_key,
        "tool_id": "ridge@1",
        "wave_digest": "d" * 64,
        "sample_ids": ["s1"],
        "prediction_count": 1,
        "request_digest": "",
        "output_digest": "f" * 64,
        "execution_owner": "dsh_agent_tool_call",
    }
    if case == "binding":
        prediction_payload["stage_attempt"] = 3
    elif case == "wave":
        prediction_payload["wave_digest"] = "c" * 64
    elif case == "request-digest":
        prediction_payload["request_digest"] = "0" * 64
    if case != "request-digest":
        prediction_payload["request_digest"] = digest(
            {
                "tool_name": "ecology_execute_prediction_tool",
                "run_id": "run:tool-test",
                "stage": "sample.plan",
                "stage_attempt": prediction_payload["stage_attempt"],
                "idempotency_key": prediction_payload["idempotency_key"],
                "arguments": {
                    "tool_id": prediction_payload["tool_id"],
                    "wave_digest": prediction_payload["wave_digest"],
                },
            }
        )
    prediction_event = ledger.append(
        "run:tool-test",
        "DshPredictionToolExecuted",
        prediction_payload,
        event_id=prediction_event_id,
    )

    identity = _identity(ledger, idempotency_key=idempotency_key)
    structured = {
        "schema_version": "ecology-sample-decisions@1",
        "wave_digest": "d" * 64,
        "decisions": [
            {
                "sample_id": "s1",
                "next_tool": "ridge@1",
                "reason_code": "frozen_registered_route",
                "confidence": 0.9,
            }
        ],
    }
    if case == "decision":
        structured["decisions"][0]["next_tool"] = "forged@9"
    receipt = {
        "event_id": prediction_event.event_id,
        "event_seq": prediction_event.seq,
        "request_digest": prediction_payload["request_digest"],
        "output_digest": prediction_payload["output_digest"],
        "execution_owner": "dsh_agent_tool_call",
    }
    if case == "receipt":
        receipt["output_digest"] = "0" * 64
    accepted_payload = {
        "schema_version": "ecologyrsi-dsh.structured-result-accepted/1",
        "identity": identity,
        "output_schema_id": "ecology-sample-decisions@1",
        "result_digest": digest(structured),
        "structured": structured,
        "skill_invocation_evidence": _skill_evidence("sample.plan"),
    }
    if case != "missing-receipt":
        accepted_payload["required_tool_receipt"] = receipt
    accepted_event_digest = digest(
        {
            "stage": "sample.plan",
            "idempotency_key": idempotency_key,
        }
    )
    accepted_event_id = (
        f"run:tool-test:dsh-structured:"
        f"{accepted_event_digest}"
    )
    accepted_event = ledger.append(
        "run:tool-test",
        "DshStructuredResultAccepted",
        accepted_payload,
        event_id=accepted_event_id,
    )
    envelope = {
        "identity": identity,
        "output_schema_id": "ecology-sample-decisions@1",
        "structured": structured,
        "result_digest": digest(structured),
        "skill_invocation_evidence": _skill_evidence("sample.plan"),
        "admission_id": "lost-response-admission",
    }
    return envelope, accepted_event


def _accepted_payload(envelope: dict) -> dict:
    return {
        "schema_version": "ecologyrsi-dsh.structured-result-accepted/1",
        "identity": json.loads(json.dumps(envelope["identity"])),
        "output_schema_id": envelope["output_schema_id"],
        "result_digest": envelope["result_digest"],
        "structured": json.loads(json.dumps(envelope["structured"])),
        "skill_invocation_evidence": json.loads(
            json.dumps(envelope["skill_invocation_evidence"])
        ),
    }


def _race_structured_append_winner(
    service: DshToolService,
    ledger: EventLedger,
    envelope: dict,
    winning_payload: dict,
    *,
    thread_name: str,
) -> object:
    """Block the legal request at append, commit a winner, then resume it."""

    before_append = threading.Event()
    release_append = threading.Event()
    original_append = ledger.append
    outcome: list[object] = []

    def observed_append(*args: object, **kwargs: object) -> object:
        if (
            threading.current_thread().name == thread_name
            and len(args) > 1
            and args[1] == "DshStructuredResultAccepted"
        ):
            before_append.set()
            if not release_append.wait(2):
                raise RuntimeError("test did not release structured append")
        return original_append(*args, **kwargs)

    def accept() -> None:
        try:
            outcome.append(service.accept_structured(envelope))
        except Exception as error:  # noqa: BLE001 - asserted by the caller
            outcome.append(error)

    worker = threading.Thread(target=accept, name=thread_name)
    try:
        with patch.object(ledger, "append", side_effect=observed_append):
            worker.start()
            if not before_append.wait(1):
                raise AssertionError("legal request did not reach its final append")
            identity = envelope["identity"]
            event_digest = digest(
                {
                    "stage": identity["stage"],
                    "idempotency_key": identity["idempotency_key"],
                }
            )
            event_id = (
                f"{identity['run_id']}:dsh-structured:"
                f"{event_digest}"
            )
            original_append(
                identity["run_id"],
                "DshStructuredResultAccepted",
                winning_payload,
                event_id=event_id,
            )
            release_append.set()
            worker.join(2)
    finally:
        release_append.set()
        worker.join(2)

    if worker.is_alive():
        raise AssertionError("structured append worker did not finish")
    if len(outcome) != 1:
        raise AssertionError(f"unexpected structured append outcome: {outcome!r}")
    return outcome[0]


class DshToolServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger(":memory:")
        self.ledger.append("run:tool-test", "RunCreated", {"test": True})
        self.service = DshToolService(self.ledger)
        self.service.open_admission("run:tool-test", 3, 2)

    def tearDown(self) -> None:
        self.ledger.close()

    def test_role_surface_and_idempotency_are_fail_closed(self) -> None:
        executions = 0

        def predict() -> dict:
            nonlocal executions
            executions += 1
            return {"s1": {"predicted": 21.5, "metadata": {"model": "ridge"}}}

        with self.service.bind_prediction_tool(
            run_id="run:tool-test",
            stage_attempt=2,
            idempotency_key="tool-idem-1",
            wave_digest="d" * 64,
            tool_id="ridge@1",
            sample_ids=("s1",),
            executor=predict,
        ) as binding:
            envelope = {
                "identity": _identity(self.ledger),
                "arguments": {"tool_id": "ridge@1", "wave_digest": "d" * 64},
            }
            first = self.service.execute(
                "ecology_execute_prediction_tool", envelope
            )
            self.assertEqual(first["prediction_count"], 1)
            self.assertEqual(executions, 1)
            with self.assertRaises(DshToolAuthorizationError):
                self.service.execute("ecology_execute_prediction_tool", envelope)

            retry = json.loads(json.dumps(envelope))
            retry["identity"]["session_id"] = "session:planner-2"
            retry["identity"]["ledger_expected_revision"] = self.ledger.latest_seq()
            second = self.service.execute(
                "ecology_execute_prediction_tool", retry
            )
            self.assertEqual(second["output_digest"], first["output_digest"])
            self.assertEqual(executions, 1)
            self.assertEqual(
                [event.kind for event in self.ledger.events("run:tool-test")].count(
                    "DshPredictionToolExecuted"
                ),
                1,
            )
            self.assertEqual(
                binding.prediction_bundle()["s1"]["metadata"]["execution_owner"],
                "dsh_agent_tool_call",
            )

            wrong_role = json.loads(json.dumps(retry))
            wrong_role["identity"]["role"] = "researcher"
            with self.assertRaises(DshToolAuthorizationError):
                self.service.execute("ecology_execute_prediction_tool", wrong_role)

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
                "ecology_execute_prediction_tool",
                {
                    "identity": _identity(self.ledger),
                    "arguments": {"tool_id": "ridge@1", "wave_digest": "d" * 64},
                },
            )

    def test_dynamic_retrieval_uses_sufficient_dsh_primary_without_fallback(self) -> None:
        fallback_calls: list[tuple[str, ...]] = []

        def fallback(queries: tuple[str, ...], *, limit: int = 8) -> dict:
            fallback_calls.append(queries)
            return {"content": "unused", "sources": [], "truncated": False}

        service = DshToolService(self.ledger, retrieval_fallback=fallback)
        service.open_admission("run:tool-test", 3, 2)
        identity = _identity(
            self.ledger,
            role="researcher",
            stage="generation.research",
            idempotency_key="research-result-1",
        )
        completed = service.complete_retrieval(
            {
                "identity": identity,
                "arguments": {
                    "queries": ["greenhouse temperature forecasting"],
                    "retrieval_key": "check-temperature-literature",
                },
                "primary_result": {
                    "content": "Two greenhouse temperature forecasting sources.",
                    "sources": [
                        {
                            "url": "https://example.org/greenhouse-temperature",
                            "title": "Greenhouse temperature forecasting",
                            "snippet": "A forecasting benchmark for protected crops.",
                        },
                        {
                            "url": "https://example.net/protected-crop-model",
                            "title": "Protected crop temperature model",
                            "snippet": "Greenhouse forecasting with causal weather inputs.",
                        },
                    ],
                    "truncated": False,
                },
            }
        )

        self.assertEqual(completed["provider_route"], "dsh_primary")
        self.assertIsNone(completed["fallback_reason"])
        self.assertEqual(len(completed["sources"]), 2)
        self.assertEqual(fallback_calls, [])
        event = self.ledger.events("run:tool-test")[-1]
        self.assertEqual(event.kind, "DshRetrievalExecuted")
        self.assertEqual(event.payload["result_digest"], completed["result_digest"])
        self.assertEqual(event.payload["result_digest"], digest(event.payload["result"]))
        self.assertEqual(
            event.payload["primary_quality"]["distinct_source_count"], 2
        )

    def test_dynamic_retrieval_falls_back_on_insufficiency_and_replays(self) -> None:
        fallback_calls: list[tuple[str, ...]] = []

        def fallback(queries: tuple[str, ...], *, limit: int = 8) -> dict:
            fallback_calls.append(queries)
            return {
                "content": "OpenAlex metadata fallback.",
                "sources": [
                    {
                        "url": "https://openalex.org/W1",
                        "title": "Greenhouse temperature forecasting",
                        "snippet": "Metadata-only evidence.",
                    },
                    {
                        "url": "https://openalex.org/W2",
                        "title": "Forecasting protected crop temperature",
                        "snippet": "Metadata-only comparison.",
                    },
                ],
                "truncated": False,
            }

        service = DshToolService(self.ledger, retrieval_fallback=fallback)
        service.open_admission("run:tool-test", 3, 2)
        identity = _identity(
            self.ledger,
            role="candidate-proposer",
            stage="candidate.propose",
            idempotency_key="candidate-result-1",
        )
        request = {
            "identity": identity,
            "arguments": {
                "queries": ["greenhouse temperature forecasting"],
                "retrieval_key": "verify-proposal-evidence",
            },
        }
        completed = service.complete_retrieval(
            {
                **request,
                "primary_result": {
                    "sources": [
                        {
                            "url": "https://example.org/only-one",
                            "title": "Greenhouse temperature forecasting",
                        }
                    ],
                    "truncated": False,
                },
            }
        )

        self.assertEqual(
            completed["provider_route"],
            "dsh_primary_then_openalex_fallback",
        )
        self.assertEqual(
            completed["fallback_reason"], "insufficient_distinct_sources"
        )
        self.assertEqual(len(fallback_calls), 1)
        replay_request = json.loads(json.dumps(request))
        replay_request["identity"]["session_id"] = "session:proposal-retry"
        replay_request["identity"]["ledger_expected_revision"] = (
            self.ledger.latest_seq()
        )
        replayed = service.replay_retrieval(replay_request)
        self.assertEqual(replayed, completed)
        self.assertEqual(len(fallback_calls), 1)

        conflict = json.loads(json.dumps(replay_request))
        conflict["arguments"]["queries"] = ["soil moisture forecasting"]
        with self.assertRaisesRegex(ValueError, "idempotency key"):
            service.replay_retrieval(conflict)

    def test_dynamic_retrieval_technical_failure_falls_back_and_budget_is_bounded(
        self,
    ) -> None:
        fallback_calls = 0

        def fallback(queries: tuple[str, ...], *, limit: int = 8) -> dict:
            nonlocal fallback_calls
            fallback_calls += 1
            return {
                "content": "Fallback metadata.",
                "sources": [
                    {
                        "url": f"https://openalex.org/{fallback_calls}A",
                        "title": f"Greenhouse forecast evidence {fallback_calls}",
                    },
                    {
                        "url": f"https://openalex.org/{fallback_calls}B",
                        "title": f"Temperature forecast evidence {fallback_calls}",
                    },
                ],
                "truncated": False,
            }

        service = DshToolService(self.ledger, retrieval_fallback=fallback)
        service.open_admission("run:tool-test", 3, 2)
        identity = _identity(
            self.ledger,
            role="generation-judge",
            stage="generation.judge",
            idempotency_key="judge-result-1",
        )
        for index in range(3):
            result = service.complete_retrieval(
                {
                    "identity": {
                        **identity,
                        "ledger_expected_revision": self.ledger.latest_seq(),
                    },
                    "arguments": {
                        "queries": [f"greenhouse forecast check {index}"],
                        "retrieval_key": f"judge-search-{index}",
                    },
                    "primary_error_code": "primary_provider_error",
                }
            )
            self.assertEqual(result["fallback_reason"], "primary_provider_error")

        with self.assertRaisesRegex(DshToolAuthorizationError, "budget"):
            service.complete_retrieval(
                {
                    "identity": {
                        **identity,
                        "ledger_expected_revision": self.ledger.latest_seq(),
                    },
                    "arguments": {
                        "queries": ["greenhouse forecast fourth query"],
                        "retrieval_key": "judge-search-3",
                    },
                    "primary_error_code": "primary_timeout",
                }
            )
        next_stage_result = service.complete_retrieval(
            {
                "identity": {
                    **identity,
                    "idempotency_key": "judge-result-next-generation",
                    "ledger_expected_revision": self.ledger.latest_seq(),
                },
                "arguments": {
                    "queries": ["greenhouse forecast new generation"],
                    "retrieval_key": "judge-search-0",
                },
                "primary_error_code": "primary_timeout",
            }
        )
        self.assertEqual(next_stage_result["fallback_reason"], "primary_timeout")
        self.assertEqual(fallback_calls, 4)

    def test_dynamic_retrieval_keeps_partial_primary_when_one_query_fails(self) -> None:
        service = DshToolService(
            self.ledger,
            retrieval_fallback=lambda _queries, limit=8: {
                "sources": [
                    {
                        "url": "https://openalex.org/W-partial",
                        "title": "Greenhouse forecasting fallback",
                    }
                ],
                "truncated": False,
            },
        )
        service.open_admission("run:tool-test", 3, 2)
        result = service.complete_retrieval(
            {
                "identity": _identity(
                    self.ledger,
                    role="researcher",
                    stage="generation.research",
                    idempotency_key="partial-result",
                ),
                "arguments": {
                    "queries": [
                        "greenhouse temperature forecasting",
                        "greenhouse humidity forecasting",
                    ],
                    "retrieval_key": "partial-primary",
                },
                "primary_result": {
                    "sources": [
                        {
                            "url": "https://example.org/temperature",
                            "title": "Greenhouse temperature forecasting",
                        }
                    ],
                    "truncated": False,
                },
                "primary_error_code": "primary_timeout",
            }
        )
        self.assertEqual(result["fallback_reason"], "primary_timeout")
        self.assertEqual(
            [item["url"] for item in result["sources"]],
            [
                "https://example.org/temperature",
                "https://openalex.org/W-partial",
            ],
        )

    def test_dynamic_retrieval_duplicate_completion_is_single_event(self) -> None:
        fallback_calls = 0

        def fallback(_queries: tuple[str, ...], *, limit: int = 8) -> dict:
            nonlocal fallback_calls
            fallback_calls += 1
            return {
                "sources": [
                    {"url": "https://openalex.org/W1", "title": "Greenhouse forecast"},
                    {"url": "https://openalex.org/W2", "title": "Temperature forecast"},
                ],
                "truncated": False,
            }

        service = DshToolService(self.ledger, retrieval_fallback=fallback)
        service.open_admission("run:tool-test", 3, 2)
        base_identity = _identity(
            self.ledger,
            role="researcher",
            stage="generation.research",
            idempotency_key="concurrent-result",
        )
        results: list[dict] = []
        failures: list[BaseException] = []

        def complete(session_id: str) -> None:
            try:
                results.append(
                    service.complete_retrieval(
                        {
                            "identity": {
                                **base_identity,
                                "session_id": session_id,
                            },
                            "arguments": {
                                "queries": ["greenhouse temperature forecasting"],
                                "retrieval_key": "concurrent-search",
                            },
                            "primary_error_code": "primary_provider_error",
                        }
                    )
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                failures.append(exc)

        workers = [
            threading.Thread(target=complete, args=(f"session:concurrent-{index}",))
            for index in range(2)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=2)

        self.assertEqual(failures, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])
        self.assertEqual(fallback_calls, 1)
        self.assertEqual(
            [event.kind for event in self.ledger.events("run:tool-test")].count(
                "DshRetrievalExecuted"
            ),
            1,
        )

    def test_dynamic_retrieval_rejects_unknown_role_and_closed_fence(self) -> None:
        arguments = {
            "queries": ["greenhouse temperature forecasting"],
            "retrieval_key": "authorization-check",
        }
        with self.assertRaises(DshToolAuthorizationError):
            self.service.complete_retrieval(
                {
                    "identity": _identity(
                        self.ledger,
                        role="unregistered-role",
                        stage="generation.research",
                    ),
                    "arguments": arguments,
                    "primary_error_code": "primary_provider_error",
                }
            )
        self.service.close_admission("run:tool-test", 3, 2)
        with self.assertRaises(DshToolAdmissionClosedError):
            self.service.complete_retrieval(
                {
                    "identity": _identity(
                        self.ledger,
                        role="researcher",
                        stage="generation.research",
                    ),
                    "arguments": arguments,
                    "primary_error_code": "primary_provider_error",
                }
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
            "skill_invocation_evidence": _skill_evidence(
                "generation.research"
            ),
            "deadline_unix_ms": FUTURE_DEADLINE_UNIX_MS,
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
        _arm_structured_envelope(self.service, envelope)
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

    def test_generation_judge_requires_candidate_review_skill(self) -> None:
        identity = _identity(
            self.ledger,
            role="generation-judge",
            stage="generation.judge",
            session_id="session:judge-1",
            idempotency_key="generation-judge-1",
        )
        structured = {
            "schema_version": "ecology-generation-review@1",
            "accepted": False,
            "rationale": "The candidate evidence is insufficient.",
            "flags": ["insufficient_evidence"],
        }
        envelope = {
            "identity": identity,
            "output_schema_id": "ecology-generation-review@1",
            "structured": structured,
            "result_digest": digest(structured),
            "skill_invocation_evidence": _skill_evidence("generation.judge"),
            "deadline_unix_ms": FUTURE_DEADLINE_UNIX_MS,
        }
        _arm_structured_envelope(self.service, envelope)

        accepted = self.service.accept_structured(envelope)

        self.assertTrue(accepted["accepted"])
        event = self.ledger.events("run:tool-test")[-1]
        self.assertEqual(
            event.payload["skill_invocation_evidence"]["skill_name"],
            "candidate-scientific-review",
        )

        wrong_skill = json.loads(json.dumps(envelope))
        wrong_skill["identity"]["idempotency_key"] = "generation-judge-old-skill"
        wrong_skill["skill_invocation_evidence"]["skill_name"] = (
            "batch-scientific-reflection"
        )
        with self.assertRaisesRegex(ValueError, "stage contract"):
            self.service.accept_structured(wrong_skill)

    def test_structured_persistence_retries_same_result_without_tool_replay(
        self,
    ) -> None:
        envelope = _research_envelope(
            self.ledger,
            deadline_unix_ms=FUTURE_DEADLINE_UNIX_MS,
        )
        _arm_structured_envelope(self.service, envelope)
        original_append = self.ledger.append
        append_calls = 0

        def flaky_append(*args: object, **kwargs: object) -> object:
            nonlocal append_calls
            append_calls += 1
            if append_calls == 1:
                raise sqlite3.OperationalError("database temporarily busy")
            return original_append(*args, **kwargs)

        self.ledger.append = flaky_append
        try:
            receipt = self.service.accept_structured(envelope)
        finally:
            self.ledger.append = original_append

        self.assertTrue(receipt["accepted"])
        self.assertEqual(append_calls, 2)
        self.assertEqual(
            [event.kind for event in self.ledger.events("run:tool-test")].count(
                "DshStructuredResultAccepted"
            ),
            1,
        )

    def test_completed_planner_is_replayed_without_a_second_agent_call(self) -> None:
        executions = 0

        def predict() -> dict:
            nonlocal executions
            executions += 1
            return {"s1": {"predicted": 21.5, "metadata": {"model": "ridge"}}}

        identity = _identity(self.ledger)
        structured = {
            "schema_version": "ecology-sample-decisions@1",
            "wave_digest": "d" * 64,
            "decisions": [
                {
                    "sample_id": "s1",
                    "next_tool": "ridge@1",
                    "reason_code": "initial_registered_route",
                    "confidence": 0.9,
                }
            ],
        }
        with self.service.bind_prediction_tool(
            run_id="run:tool-test",
            stage_attempt=2,
            idempotency_key="tool-idem-1",
            wave_digest="d" * 64,
            tool_id="ridge@1",
            sample_ids=("s1",),
            executor=predict,
        ):
            self.service.execute(
                "ecology_execute_prediction_tool",
                {
                    "identity": identity,
                    "arguments": {
                        "tool_id": "ridge@1",
                        "wave_digest": "d" * 64,
                    },
                },
            )
            envelope = {
                "identity": identity,
                "output_schema_id": "ecology-sample-decisions@1",
                "structured": structured,
                "result_digest": digest(structured),
                "skill_invocation_evidence": _skill_evidence("sample.plan"),
                "deadline_unix_ms": FUTURE_DEADLINE_UNIX_MS,
            }
            _arm_structured_envelope(self.service, envelope)
            self.service.accept_structured(envelope)

        class NeverCalledRuntime:
            calls = 0

            def run_stage(self, _request: dict) -> dict:
                self.calls += 1
                raise AssertionError("a persisted Planner must be replayed")

        restarted = DshToolService(self.ledger)
        runtime = NeverCalledRuntime()
        with restarted.bind_prediction_tool(
            run_id="run:tool-test",
            stage_attempt=2,
            idempotency_key="tool-idem-1",
            wave_digest="d" * 64,
            tool_id="ridge@1",
            sample_ids=("s1",),
            executor=predict,
        ) as binding:
            replayed = DshStructuredRoleRuntime(runtime, admission=restarted).run(
                run_id="run:tool-test",
                stage="sample.plan",
                role="sample-planner",
                context={"retry_after": "critic_failure"},
                output_schema_id="ecology-sample-decisions@1",
                run_state_revision=self.ledger.latest_seq(),
                stage_attempt=2,
                ledger_expected_revision=self.ledger.latest_seq(),
                idempotency_key="tool-idem-1",
                identity_digests={
                    "genome_digest": "a" * 64,
                    "compiled_behavior_digest": "b" * 64,
                    "phenotype_instance_digest": "c" * 64,
                },
            )
            bundle = binding.prediction_bundle()

        self.assertEqual(replayed, structured)
        self.assertEqual(bundle["s1"]["predicted"], 21.5)
        self.assertEqual(runtime.calls, 0)
        self.assertEqual(executions, 2)
        kinds = [event.kind for event in self.ledger.events("run:tool-test")]
        self.assertEqual(kinds.count("DshPredictionToolExecuted"), 1)
        self.assertEqual(kinds.count("DshStructuredResultAccepted"), 1)

    def test_structured_replay_fails_closed_on_identity_or_digest_drift(self) -> None:
        identity = _identity(
            self.ledger,
            role="researcher",
            stage="generation.research",
            idempotency_key="research-replay",
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
            "skill_invocation_evidence": _skill_evidence(
                "generation.research"
            ),
            "deadline_unix_ms": FUTURE_DEADLINE_UNIX_MS,
        }
        _arm_structured_envelope(self.service, envelope)
        self.service.accept_structured(envelope)
        replay_args = {
            "run_id": "run:tool-test",
            "stage": "generation.research",
            "role": "researcher",
            "stage_attempt": 2,
            "idempotency_key": "research-replay",
            "output_schema_id": "ecology-research-result@1",
            "identity_digests": {
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
        }
        self.assertEqual(
            self.service.replay_structured_result(**replay_args), structured
        )
        drifted = json.loads(json.dumps(replay_args))
        drifted["identity_digests"]["genome_digest"] = "f" * 64
        with self.assertRaises(DshToolAuthorizationError):
            self.service.replay_structured_result(**drifted)

        corrupt_run = "run:corrupt-replay"
        corrupt_key = "corrupt-result"
        self.ledger.append(corrupt_run, "RunCreated", {"test": True})
        corrupt_identity = {
            **identity,
            "run_id": corrupt_run,
            "idempotency_key": corrupt_key,
        }
        corrupt_event_digest = digest(
            {"stage": "generation.research", "idempotency_key": corrupt_key}
        )
        corrupt_event_id = (
            f"{corrupt_run}:dsh-structured:"
            f"{corrupt_event_digest}"
        )
        self.ledger.append(
            corrupt_run,
            "DshStructuredResultAccepted",
            {
                "schema_version": "ecologyrsi-dsh.structured-result-accepted/1",
                "identity": corrupt_identity,
                "output_schema_id": "ecology-research-result@1",
                "result_digest": "0" * 64,
                "structured": structured,
                "skill_invocation_evidence": _skill_evidence(
                    "generation.research"
                ),
            },
            event_id=corrupt_event_id,
        )
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.service.replay_structured_result(
                **{
                    **replay_args,
                    "run_id": corrupt_run,
                    "idempotency_key": corrupt_key,
                }
            )

    def test_sample_reflection_result_is_stage_bound_and_accepted(self) -> None:
        identity = _identity(
            self.ledger,
            role="sample-critic",
            stage="sample.reflect",
            session_id="session:reflector-1",
            idempotency_key="sample-reflection-1",
        )
        structured = {
            "schema_version": "ecology-sample-reflection@1",
            "wave_digest": "d" * 64,
            "sample_id": "sample-1",
            "outcome_class": "degraded",
            "error_source": "parameter",
            "next_action": "decrease",
            "confidence": 0.8,
            "summary": "The completed sample suggests a smaller correction.",
        }

        envelope = {
            "identity": identity,
            "output_schema_id": "ecology-sample-reflection@1",
            "structured": structured,
            "result_digest": digest(structured),
            "skill_invocation_evidence": _skill_evidence("sample.reflect"),
            "deadline_unix_ms": FUTURE_DEADLINE_UNIX_MS,
        }
        _arm_structured_envelope(self.service, envelope)
        accepted = self.service.accept_structured(envelope)

        self.assertTrue(accepted["accepted"])
        event = self.ledger.events("run:tool-test")[-1]
        self.assertEqual(event.kind, "DshStructuredResultAccepted")
        self.assertEqual(event.payload["identity"]["stage"], "sample.reflect")
        self.assertEqual(
            event.payload["output_schema_id"], "ecology-sample-reflection@1"
        )

    def test_structured_result_requires_an_armed_admission(self) -> None:
        envelope = _research_envelope(
            self.ledger,
            deadline_unix_ms=FUTURE_DEADLINE_UNIX_MS,
        )
        envelope.pop("deadline_unix_ms")

        with self.assertRaisesRegex(ValueError, "admission_id"):
            self.service.accept_structured(envelope)

    def test_child_reservation_arms_one_frozen_monotonic_deadline(self) -> None:
        monotonic_ms = [100]
        service = DshToolService(
            self.ledger,
            monotonic_ms=lambda: monotonic_ms[0],
        )
        fence = service.open_admission(
            "run:tool-test",
            3,
            2,
            role="researcher",
            stage="generation.research",
            idempotency_key="deadline-result",
        )

        first = service.allocate_child_reservation(
            _reservation_request(fence.admission_id)
        )
        frozen_deadline = fence.deadline_monotonic_ms
        monotonic_ms[0] = 500
        second = service.allocate_child_reservation(
            _reservation_request(
                fence.admission_id,
                request_id="deadline-reservation-2",
            )
        )

        self.assertEqual(first["admission_id"], fence.admission_id)
        self.assertEqual(first["timeout_ms"], 1_000)
        self.assertEqual(second["admission_id"], fence.admission_id)
        self.assertEqual(frozen_deadline, 1_100)
        self.assertEqual(fence.deadline_monotonic_ms, frozen_deadline)

        changed = _reservation_request(
            fence.admission_id,
            request_id="deadline-reservation-3",
            timeout_ms=1_001,
        )
        with self.assertRaises(Exception) as caught:
            service.allocate_child_reservation(changed)
        self.assertEqual(
            getattr(caught.exception, "error_code", None),
            "structured_role_operational_timeout",
        )
        self.assertEqual(fence.deadline_monotonic_ms, frozen_deadline)

    def test_sample_child_reservation_persists_member_correlation(self) -> None:
        fence = self.service.open_admission(
            "run:tool-test",
            3,
            2,
            role="sample-planner",
            stage="sample.plan",
            idempotency_key="sample-plan-members",
        )
        members = sorted((digest("cell-a-1"), digest("cell-a-2")))
        request = _reservation_request(
            fence.admission_id,
            request_id="sample-member-reservation-1",
            sample_member_digests=members,
        )
        request.update({
            "role": "sample-planner",
            "stage": "sample.plan",
            "idempotency_key": "sample-plan-members",
        })

        allocated = self.service.allocate_child_reservation(request)

        self.assertEqual(allocated["launch"]["sample_member_digests"], members)
        launch_event = next(
            event
            for event in self.ledger.events("run:tool-test")
            if event.kind == "DshChildLaunchReserved"
            and event.payload["request_id"] == "sample-member-reservation-1"
        )
        self.assertEqual(
            launch_event.payload["launch"]["sample_member_digests"],
            members,
        )

    def test_child_failure_settles_latest_reservation_idempotently(self) -> None:
        fence = self.service.open_admission(
            "run:tool-test",
            3,
            2,
            role="sample-planner",
            stage="sample.plan",
            idempotency_key="sample-plan-failure",
        )
        request = _reservation_request(
            fence.admission_id,
            request_id="sample-failure-reservation-1",
        )
        request.update({
            "role": "sample-planner",
            "stage": "sample.plan",
            "idempotency_key": "sample-plan-failure",
        })
        allocated = self.service.allocate_child_reservation(request)
        failure = {
            "run_id": "run:tool-test",
            "stage": "sample.plan",
            "idempotency_key": "sample-plan-failure",
            "error_code": "rate_limit",
        }

        first = self.service.record_child_failure(failure)
        second = self.service.record_child_failure(failure)

        self.assertEqual(first, {"accepted": True, "already_settled": False})
        self.assertEqual(second, {"accepted": True, "already_settled": True})
        failed_events = [
            event
            for event in self.ledger.events("run:tool-test")
            if event.kind == "DshChildExecutionFailed"
        ]
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(
            failed_events[0].payload["identity"]["child_reservation_id"],
            allocated["launch"]["reservation_id"],
        )

    def test_child_failure_rejects_unknown_or_invalid_request(self) -> None:
        missing = self.service.record_child_failure({
            "run_id": "run:tool-test",
            "stage": "sample.plan",
            "idempotency_key": "missing",
            "error_code": "rate_limit",
        })
        self.assertEqual(
            missing,
            {"accepted": False, "reason": "launch_not_found"},
        )
        with self.assertRaisesRegex(ValueError, "normalized"):
            self.service.record_child_failure({
                "run_id": "run:tool-test",
                "stage": "sample.plan",
                "idempotency_key": "missing",
                "error_code": "contains spaces",
            })

    def test_armed_deadline_uses_only_monotonic_time_after_wall_rollback(self) -> None:
        monotonic_ms = [100]
        service = DshToolService(
            self.ledger,
            monotonic_ms=lambda: monotonic_ms[0],
        )
        service._wall_clock_ms = lambda: self.fail(
            "wall time must not participate in armed deadline authorization"
        )
        fence = service.open_admission(
            "run:tool-test",
            3,
            2,
            role="researcher",
            stage="generation.research",
            idempotency_key="deadline-result",
        )
        service.allocate_child_reservation(
            _reservation_request(fence.admission_id)
        )
        monotonic_ms[0] = 1_100

        with self.assertRaises(Exception) as caught:
            service.accept_structured(
                _admitted_research_envelope(
                    self.ledger,
                    admission_id=fence.admission_id,
                )
            )

        self.assertEqual(
            getattr(caught.exception, "error_code", None),
            "structured_role_operational_timeout",
        )
        self.assertFalse(any(
            event.kind == "DshStructuredResultAccepted"
            for event in self.ledger.events("run:tool-test")
        ))

    def test_child_reservation_rejects_timeout_above_protocol_ceiling(self) -> None:
        service = DshToolService(self.ledger, monotonic_ms=lambda: 100)
        fence = service.open_admission(
            "run:tool-test",
            3,
            2,
            role="researcher",
            stage="generation.research",
            idempotency_key="deadline-result",
        )

        with self.assertRaises(Exception) as caught:
            service.allocate_child_reservation(
                _reservation_request(
                    fence.admission_id,
                    timeout_ms=1_800_001,
                )
            )

        self.assertEqual(
            getattr(caught.exception, "error_code", None),
            "structured_role_operational_timeout",
        )
        self.assertFalse(any(
            event.kind == "DshChildLaunchReserved"
            for event in self.ledger.events("run:tool-test")
        ))

    def test_structured_result_must_match_the_armed_admission(self) -> None:
        service = DshToolService(self.ledger, monotonic_ms=lambda: 100)
        fence = service.open_admission(
            "run:tool-test",
            3,
            2,
            role="researcher",
            stage="generation.research",
            idempotency_key="deadline-result",
        )
        service.allocate_child_reservation(
            _reservation_request(fence.admission_id)
        )

        with self.assertRaises(Exception) as caught:
            service.accept_structured(
                _admitted_research_envelope(
                    self.ledger,
                    admission_id="admission-forged",
                )
            )

        self.assertEqual(
            getattr(caught.exception, "error_code", None),
            "structured_role_operational_timeout",
        )

    def test_expired_structured_result_is_rejected_before_arrival_work(self) -> None:
        monotonic_ms = [100]
        self.service._monotonic_ms = lambda: monotonic_ms[0]
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        before = self.ledger.count("run:tool-test")
        monotonic_ms[0] = 1_100

        with self.assertRaises(Exception) as caught:
            self.service.accept_structured(envelope)

        self.assertEqual(
            getattr(caught.exception, "error_code", None),
            "structured_role_operational_timeout",
        )
        self.assertRegex(str(caught.exception), "deadline")
        self.assertEqual(self.ledger.count("run:tool-test"), before)

    def test_structured_result_deadline_is_rechecked_inside_ledger_precommit(self) -> None:
        monotonic_ms = [100]
        monotonic_calls = 0
        precommit_entered = threading.Event()
        release_precommit = threading.Event()

        def monotonic_clock_ms() -> int:
            nonlocal monotonic_calls
            monotonic_calls += 1
            if monotonic_calls == 3:
                precommit_entered.set()
                if not release_precommit.wait(2):
                    raise RuntimeError("test did not release precommit deadline check")
            return monotonic_ms[0]

        self.service._monotonic_ms = lambda: monotonic_ms[0]
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        self.service._monotonic_ms = monotonic_clock_ms
        monotonic_calls = 0
        before = self.ledger.count("run:tool-test")
        outcome: list[object] = []

        def accept() -> None:
            try:
                outcome.append(self.service.accept_structured(envelope))
            except Exception as error:  # noqa: BLE001 - asserted below
                outcome.append(error)

        worker = threading.Thread(target=accept)
        worker.start()
        try:
            self.assertTrue(
                precommit_entered.wait(1),
                "structured result never reached its ledger precommit guard",
            )
            monotonic_ms[0] = 1_100
        finally:
            release_precommit.set()
        worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertEqual(
            getattr(outcome[0], "error_code", None),
            "structured_role_operational_timeout",
        )
        self.assertEqual(self.ledger.count("run:tool-test"), before)
        self.assertFalse(any(
            event.kind == "DshStructuredResultAccepted"
            for event in self.ledger.events("run:tool-test")
        ))

    def test_exact_structured_receipt_replays_after_its_deadline(self) -> None:
        monotonic_ms = [100]
        self.service._monotonic_ms = lambda: monotonic_ms[0]
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)

        first = self.service.accept_structured(envelope)
        monotonic_ms[0] = 1_100
        second = self.service.accept_structured(envelope)

        self.assertEqual(second, first)
        self.assertEqual(
            sum(
                event.kind == "DshStructuredResultAccepted"
                for event in self.ledger.events("run:tool-test")
            ),
            1,
        )

    def test_exact_structured_receipt_replays_after_service_restart(self) -> None:
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        first = self.service.accept_structured(envelope)
        restarted = DshToolService(self.ledger)

        second = restarted.accept_structured(envelope)

        self.assertEqual(second, first)
        self.assertEqual(
            sum(
                event.kind == "DshStructuredResultAccepted"
                for event in self.ledger.events("run:tool-test")
            ),
            1,
        )

    def test_restarted_service_validates_identity_before_durable_lookup(self) -> None:
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        self.service.accept_structured(envelope)
        malformed_fields = {
            "run_state_revision float": ("run_state_revision", 3.0),
            "stage_attempt float": ("stage_attempt", 2.0),
            "ledger_expected_revision float": (
                "ledger_expected_revision",
                float(envelope["identity"]["ledger_expected_revision"]),
            ),
            "bool revision": ("ledger_expected_revision", True),
            "negative revision": ("run_state_revision", -1),
            "empty session": ("session_id", ""),
            "blank reservation": ("child_reservation_id", "  "),
            "non-text lease": ("activation_lease_id", 7),
            "invalid digest": ("genome_digest", "not-a-sha256"),
        }

        for label, (field, value) in malformed_fields.items():
            with self.subTest(label=label):
                retry = json.loads(json.dumps(envelope))
                retry["identity"][field] = value
                restarted = DshToolService(self.ledger)

                def durable_lookup_must_not_run(*_args: object) -> object:
                    self.fail("malformed identity reached durable lookup")

                restarted._event_by_id = durable_lookup_must_not_run
                with self.assertRaises(DshToolAuthorizationError):
                    restarted.accept_structured(retry)

    def test_restarted_service_rejects_same_malformed_identity_from_history(
        self,
    ) -> None:
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        envelope.pop("deadline_unix_ms")
        envelope["admission_id"] = "lost-response-admission"
        envelope["identity"]["stage_attempt"] = 2.0
        payload = {
            "schema_version": "ecologyrsi-dsh.structured-result-accepted/1",
            "identity": dict(envelope["identity"]),
            "output_schema_id": envelope["output_schema_id"],
            "result_digest": envelope["result_digest"],
            "structured": dict(envelope["structured"]),
            "skill_invocation_evidence": dict(
                envelope["skill_invocation_evidence"]
            ),
        }
        event_digest = digest(
            {
                "stage": "generation.research",
                "idempotency_key": "deadline-result",
            }
        )
        event_id = (
            "run:tool-test:dsh-structured:"
            f"{event_digest}"
        )
        self.ledger.append(
            "run:tool-test",
            "DshStructuredResultAccepted",
            payload,
            event_id=event_id,
        )

        with self.assertRaises(DshToolAuthorizationError):
            DshToolService(self.ledger).accept_structured(envelope)

    def test_restarted_service_preserves_non_identity_error_classification(
        self,
    ) -> None:
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        self.service.accept_structured(envelope)
        restarted = DshToolService(self.ledger)

        structured_conflict = json.loads(json.dumps(envelope))
        structured_conflict["structured"]["summary"] = "different result"
        structured_conflict["result_digest"] = digest(
            structured_conflict["structured"]
        )
        with self.assertRaisesRegex(ValueError, "idempotency key was reused"):
            restarted.accept_structured(structured_conflict)

        digest_conflict = json.loads(json.dumps(envelope))
        digest_conflict["result_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "result digest mismatch"):
            restarted.accept_structured(digest_conflict)

        skill_conflict = json.loads(json.dumps(envelope))
        skill_conflict["skill_invocation_evidence"]["skill_name"] = (
            "origin-vector-review"
        )
        with self.assertRaisesRegex(ValueError, "stage contract"):
            restarted.accept_structured(skill_conflict)

        schema_conflict = json.loads(json.dumps(envelope))
        schema_conflict["output_schema_id"] = "ecology-generation-review@1"
        with self.assertRaisesRegex(
            DshToolAuthorizationError,
            "schema is not allowed",
        ):
            restarted.accept_structured(schema_conflict)

        metrics_conflict = json.loads(json.dumps(envelope))
        metrics_conflict["session_metrics"] = {
            "schema_version": "ecologyrsi-dsh.dsh-session-metrics/1",
            "session_id": "session:forged",
            "context_pressure": {
                "available": False,
                "source": "dsh_token_meter",
            },
            "provider_usage": {
                "available": False,
                "source": "dsh_session_projection_token_usage",
            },
        }
        with self.assertRaisesRegex(
            DshToolAuthorizationError,
            "session metrics identity mismatch",
        ):
            restarted.accept_structured(metrics_conflict)

    def test_restarted_planner_exact_receipt_requires_no_live_fence_or_binding(
        self,
    ) -> None:
        envelope, accepted_event = _append_recorded_planner_result(
            self.ledger,
            case="healthy",
        )
        before = self.ledger.count("run:tool-test")

        receipt = DshToolService(self.ledger).accept_structured(envelope)

        self.assertEqual(receipt["event_id"], accepted_event.event_id)
        self.assertEqual(receipt["event_seq"], accepted_event.seq)
        self.assertEqual(self.ledger.count("run:tool-test"), before)

    def test_restarted_planner_explicit_replay_requires_no_live_binding(
        self,
    ) -> None:
        envelope, _accepted_event = _append_recorded_planner_result(
            self.ledger,
            case="healthy",
        )
        identity = envelope["identity"]
        before = self.ledger.count("run:tool-test")

        replayed = DshToolService(self.ledger).replay_structured_result(
            run_id=identity["run_id"],
            stage=identity["stage"],
            role=identity["role"],
            stage_attempt=identity["stage_attempt"],
            idempotency_key=identity["idempotency_key"],
            output_schema_id=envelope["output_schema_id"],
            identity_digests={
                "genome_digest": identity["genome_digest"],
                "compiled_behavior_digest": identity[
                    "compiled_behavior_digest"
                ],
                "phenotype_instance_digest": identity[
                    "phenotype_instance_digest"
                ],
            },
        )

        self.assertEqual(replayed, envelope["structured"])
        self.assertEqual(self.ledger.count("run:tool-test"), before)

    def test_restarted_planner_runtime_replays_without_agent_call(self) -> None:
        envelope, _accepted_event = _append_recorded_planner_result(
            self.ledger,
            case="healthy",
        )
        identity = envelope["identity"]

        class CountingRuntime:
            def __init__(self) -> None:
                self.calls = 0

            def run_stage(self, _request: dict) -> dict:
                self.calls += 1
                unexpected = {"unexpected": True}
                return {
                    "structured": unexpected,
                    "result_digest": digest(unexpected),
                }

        client = CountingRuntime()
        restarted = DshToolService(self.ledger)
        runtime = DshStructuredRoleRuntime(client, admission=restarted)
        before = self.ledger.count("run:tool-test")
        current_revision = self.ledger.latest_seq()

        replayed = runtime.run(
            run_id=identity["run_id"],
            stage=identity["stage"],
            role=identity["role"],
            context={"restart": "new Host invocation"},
            output_schema_id=envelope["output_schema_id"],
            run_state_revision=current_revision + 10,
            stage_attempt=identity["stage_attempt"],
            ledger_expected_revision=current_revision,
            idempotency_key=identity["idempotency_key"],
            identity_digests={
                "genome_digest": identity["genome_digest"],
                "compiled_behavior_digest": identity[
                    "compiled_behavior_digest"
                ],
                "phenotype_instance_digest": identity[
                    "phenotype_instance_digest"
                ],
            },
        )

        self.assertEqual(replayed, envelope["structured"])
        self.assertEqual(client.calls, 0)
        self.assertEqual(self.ledger.count("run:tool-test"), before)

    def test_restarted_planner_revalidates_durable_prediction_binding(self) -> None:
        for case in (
            "missing-receipt",
            "receipt",
            "request-digest",
            "binding",
            "decision",
            "wave",
        ):
            with self.subTest(case=case):
                envelope, _accepted_event = _append_recorded_planner_result(
                    self.ledger,
                    case=case,
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "recorded sample.plan prediction-tool binding is invalid",
                ):
                    DshToolService(self.ledger).accept_structured(envelope)

    def test_restarted_service_rejects_conflicting_structured_replay(self) -> None:
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        self.service.accept_structured(envelope)
        restarted = DshToolService(self.ledger)
        conflicting = json.loads(json.dumps(envelope))
        conflicting["structured"]["summary"] = "conflicting restarted payload"
        conflicting["result_digest"] = digest(conflicting["structured"])

        with self.assertRaisesRegex(ValueError, "idempotency key was reused"):
            restarted.accept_structured(conflicting)

        self.assertEqual(
            sum(
                event.kind == "DshStructuredResultAccepted"
                for event in self.ledger.events("run:tool-test")
            ),
            1,
        )

    def test_restarted_service_rejects_conflicting_identity_replay(self) -> None:
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        self.service.accept_structured(envelope)
        restarted = DshToolService(self.ledger)
        conflicting = json.loads(json.dumps(envelope))
        conflicting["identity"]["session_id"] = "session:conflicting-retry"

        with self.assertRaisesRegex(ValueError, "idempotency key was reused"):
            restarted.accept_structured(conflicting)

        self.assertEqual(
            sum(
                event.kind == "DshStructuredResultAccepted"
                for event in self.ledger.events("run:tool-test")
            ),
            1,
        )

    def test_expired_conflicting_structured_replay_still_rejects_idempotency(self) -> None:
        monotonic_ms = [100]
        self.service._monotonic_ms = lambda: monotonic_ms[0]
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        self.service.accept_structured(envelope)
        monotonic_ms[0] = 1_100
        conflicting = json.loads(json.dumps(envelope))
        conflicting["structured"]["summary"] = "conflicting late payload"
        conflicting["result_digest"] = digest(conflicting["structured"])

        with self.assertRaisesRegex(ValueError, "idempotency key was reused"):
            self.service.accept_structured(conflicting)

        self.assertEqual(
            sum(
                event.kind == "DshStructuredResultAccepted"
                for event in self.ledger.events("run:tool-test")
            ),
            1,
        )

    def test_concurrent_exact_receipt_replays_after_early_lookup_expires(self) -> None:
        replay_after_empty_lookup = threading.Event()
        release_replay = threading.Event()
        monotonic_ms = [100]
        self.service._monotonic_ms = lambda: monotonic_ms[0]
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        original_event_by_id = self.service._event_by_id
        blocked_once = False

        def observed_event_by_id(run_id: str, event_id: str) -> object:
            nonlocal blocked_once
            result = original_event_by_id(run_id, event_id)
            if (
                threading.current_thread().name == "late-exact-replay"
                and ":dsh-structured:" in event_id
                and not blocked_once
            ):
                blocked_once = True
                replay_after_empty_lookup.set()
                if not release_replay.wait(2):
                    raise RuntimeError("test did not release late exact replay")
            return result

        self.service._event_by_id = observed_event_by_id
        replay_outcome: list[object] = []

        def replay() -> None:
            try:
                replay_outcome.append(self.service.accept_structured(envelope))
            except Exception as error:  # noqa: BLE001 - asserted below
                replay_outcome.append(error)

        replay_worker = threading.Thread(
            target=replay,
            name="late-exact-replay",
        )
        replay_worker.start()
        self.assertTrue(
            replay_after_empty_lookup.wait(1),
            "replay did not finish its early receipt lookup",
        )
        first = self.service.accept_structured(envelope)
        monotonic_ms[0] = 1_100
        release_replay.set()
        replay_worker.join(2)

        self.assertFalse(replay_worker.is_alive())
        self.assertEqual(replay_outcome, [first])
        self.assertEqual(
            sum(
                event.kind == "DshStructuredResultAccepted"
                for event in self.ledger.events("run:tool-test")
            ),
            1,
        )

    def test_insert_ignore_exact_replay_skips_expired_commit_guard(self) -> None:
        before_append = threading.Event()
        release_append = threading.Event()
        monotonic_ms = [100]
        monotonic_calls = 0

        def replay_monotonic_ms() -> int:
            nonlocal monotonic_calls
            monotonic_calls += 1
            return monotonic_ms[0]

        self.service._monotonic_ms = replay_monotonic_ms
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        original_event_by_id = self.service._event_by_id
        original_append = self.ledger.append
        hid_early_receipt = False

        def observed_event_by_id(run_id: str, event_id: str) -> object:
            nonlocal hid_early_receipt
            result = original_event_by_id(run_id, event_id)
            if (
                threading.current_thread().name == "insert-ignore-replay"
                and ":dsh-structured:" in event_id
                and not hid_early_receipt
            ):
                hid_early_receipt = True
                return None
            return result

        def observed_append(*args: object, **kwargs: object) -> object:
            if (
                threading.current_thread().name == "insert-ignore-replay"
                and len(args) > 1
                and args[1] == "DshStructuredResultAccepted"
            ):
                before_append.set()
                if not release_append.wait(2):
                    raise RuntimeError("test did not release exact replay append")
            return original_append(*args, **kwargs)

        self.service._event_by_id = observed_event_by_id
        self.ledger.append = observed_append
        replay_outcome: list[object] = []

        def replay() -> None:
            try:
                replay_outcome.append(self.service.accept_structured(envelope))
            except Exception as error:  # noqa: BLE001 - asserted below
                replay_outcome.append(error)

        replay_worker = threading.Thread(target=replay, name="insert-ignore-replay")
        replay_worker.start()
        self.assertTrue(before_append.wait(1))

        first = self.service.accept_structured(envelope)
        monotonic_ms[0] = 1_100
        calls_before_insert_ignore = monotonic_calls
        release_append.set()
        replay_worker.join(2)

        self.assertFalse(replay_worker.is_alive())
        self.assertEqual(replay_outcome, [first])
        self.assertEqual(
            monotonic_calls,
            calls_before_insert_ignore,
            "an INSERT OR IGNORE replay must not enter the new-row deadline guard",
        )
        self.assertEqual(
            sum(
                event.kind == "DshStructuredResultAccepted"
                for event in self.ledger.events("run:tool-test")
            ),
            1,
        )

    def test_insert_ignore_race_never_acknowledges_type_invalid_winner(self) -> None:
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        invalid_payload = _accepted_payload(envelope)
        invalid_payload["identity"]["stage_attempt"] = 2.0

        outcome = _race_structured_append_winner(
            self.service,
            self.ledger,
            envelope,
            invalid_payload,
            thread_name="type-invalid-structured-winner",
        )

        self.assertIsInstance(outcome, DshToolAuthorizationError)
        self.assertIn("invalid Host revision field: stage_attempt", str(outcome))
        with self.assertRaisesRegex(
            DshToolAuthorizationError,
            "invalid Host revision field: stage_attempt",
        ):
            DshToolService(self.ledger).accept_structured(envelope)
        self.assertEqual(
            sum(
                event.kind == "DshStructuredResultAccepted"
                for event in self.ledger.events("run:tool-test")
            ),
            1,
        )

    def test_insert_ignore_race_preserves_structured_conflict_classification(
        self,
    ) -> None:
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        conflicting_payload = _accepted_payload(envelope)
        conflicting_payload["structured"]["summary"] = "concurrent conflict"
        conflicting_payload["result_digest"] = digest(
            conflicting_payload["structured"]
        )

        outcome = _race_structured_append_winner(
            self.service,
            self.ledger,
            envelope,
            conflicting_payload,
            thread_name="conflicting-structured-winner",
        )

        self.assertIsInstance(outcome, ValueError)
        self.assertRegex(str(outcome), "idempotency key was reused")
        with self.assertRaisesRegex(ValueError, "idempotency key was reused"):
            DshToolService(self.ledger).accept_structured(envelope)
        self.assertEqual(
            sum(
                event.kind == "DshStructuredResultAccepted"
                for event in self.ledger.events("run:tool-test")
            ),
            1,
        )

    def test_close_fence_cannot_overtake_structured_result_commit(self) -> None:
        monotonic_calls = 0
        precommit_entered = threading.Event()
        release_precommit = threading.Event()
        close_blocked = threading.Event()
        close_finished = threading.Event()
        fence = self.service._fences[("run:tool-test", 3, 2)]
        inner_lock = fence.lock

        class ObservedFenceLock:
            def __enter__(self) -> ObservedFenceLock:
                if threading.current_thread().name == "close-admission":
                    if not inner_lock.acquire(blocking=False):
                        close_blocked.set()
                        inner_lock.acquire()
                else:
                    inner_lock.acquire()
                return self

            def __exit__(self, *_error: object) -> None:
                inner_lock.release()

        fence.lock = ObservedFenceLock()

        def monotonic_clock_ms() -> int:
            nonlocal monotonic_calls
            monotonic_calls += 1
            if monotonic_calls == 3:
                precommit_entered.set()
                if not release_precommit.wait(2):
                    raise RuntimeError("test did not release guarded commit")
            return 100

        self.service._monotonic_ms = lambda: 100
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        self.service._monotonic_ms = monotonic_clock_ms
        monotonic_calls = 0
        outcome: list[object] = []

        def accept() -> None:
            try:
                outcome.append(self.service.accept_structured(envelope))
            except Exception as error:  # noqa: BLE001 - asserted below
                outcome.append(error)

        accept_worker = threading.Thread(target=accept)
        accept_worker.start()
        self.assertTrue(precommit_entered.wait(1))

        def close() -> None:
            self.service.close_admission("run:tool-test", 3, 2)
            close_finished.set()

        close_worker = threading.Thread(target=close, name="close-admission")
        close_worker.start()
        try:
            self.assertTrue(
                close_blocked.wait(1),
                "close did not contend on the guarded commit's fence",
            )
            self.assertFalse(close_finished.is_set())
        finally:
            release_precommit.set()
        accept_worker.join(2)
        close_worker.join(2)

        self.assertFalse(accept_worker.is_alive())
        self.assertFalse(close_worker.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], dict)
        self.assertTrue(outcome[0]["accepted"])
        self.assertTrue(close_finished.is_set())
        self.assertEqual(
            self.service._fences[("run:tool-test", 3, 2)].state,
            "closed",
        )

    def test_close_run_cannot_overtake_structured_result_commit(self) -> None:
        monotonic_calls = 0
        precommit_entered = threading.Event()
        release_precommit = threading.Event()
        close_blocked = threading.Event()
        close_finished = threading.Event()
        fence = self.service._fences[("run:tool-test", 3, 2)]
        inner_lock = fence.lock

        class ObservedFenceLock:
            def __enter__(self) -> ObservedFenceLock:
                if threading.current_thread().name == "close-run-admissions":
                    if not inner_lock.acquire(blocking=False):
                        close_blocked.set()
                        inner_lock.acquire()
                else:
                    inner_lock.acquire()
                return self

            def __exit__(self, *_error: object) -> None:
                inner_lock.release()

        fence.lock = ObservedFenceLock()

        def monotonic_clock_ms() -> int:
            nonlocal monotonic_calls
            monotonic_calls += 1
            if monotonic_calls == 3:
                precommit_entered.set()
                if not release_precommit.wait(2):
                    raise RuntimeError("test did not release guarded commit")
            return 100

        self.service._monotonic_ms = lambda: 100
        envelope = _research_envelope(self.ledger, deadline_unix_ms=1_000)
        _arm_structured_envelope(self.service, envelope)
        self.service._monotonic_ms = monotonic_clock_ms
        monotonic_calls = 0
        outcome: list[object] = []

        accept_worker = threading.Thread(
            target=lambda: outcome.append(
                self.service.accept_structured(envelope)
            )
        )
        accept_worker.start()
        self.assertTrue(precommit_entered.wait(1))

        def close_run() -> None:
            self.service.close_run_admissions("run:tool-test")
            close_finished.set()

        close_worker = threading.Thread(
            target=close_run,
            name="close-run-admissions",
        )
        close_worker.start()
        try:
            self.assertTrue(close_blocked.wait(1))
            self.assertFalse(close_finished.is_set())
        finally:
            release_precommit.set()
        accept_worker.join(2)
        close_worker.join(2)

        self.assertFalse(accept_worker.is_alive())
        self.assertFalse(close_worker.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertTrue(outcome[0]["accepted"])
        self.assertTrue(close_finished.is_set())
        self.assertEqual(fence.state, "closed")

    def test_cross_language_schemas_are_closed_and_role_sets_match(self) -> None:
        root = Path(__file__).resolve().parents[1] / "integrations" / "dsh_ecology_plugin"
        schemas = sorted((root / "schemas").glob("*.schema.json"))
        self.assertEqual(len(schemas), 13)
        self.assertTrue((root / "schemas" / "sample-reflection.schema.json").is_file())
        for path in schemas:
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(value["additionalProperties"])
            self.assertTrue(value["$id"].startswith("ecology-"))
        self.assertEqual(
            ROLE_TOOLS["sample-planner"],
            frozenset({"web_search", "ecology_execute_prediction_tool"}),
        )
        self.assertTrue(
            all(
                tools == frozenset({"web_search"})
                for role, tools in ROLE_TOOLS.items()
                if role != "sample-planner"
            )
        )


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
            "identity": _identity(self.server.ledger),
            "arguments": {"tool_id": "ridge@1", "wave_digest": "d" * 64},
        }
        request = Request(
            f"http://127.0.0.1:{self.server.server_address[1]}"
            "/api/ecology-agent-sidecar/v1/tools/ecology_execute_prediction_tool",
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

    def _post(self, path: str, body: dict, token: str = "tool-secret") -> tuple[int, dict]:
        request = Request(
            f"http://127.0.0.1:{self.server.server_address[1]}{path}",
            data=json.dumps(body).encode(),
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
        with self.server.dsh_tools.bind_prediction_tool(
            run_id="run:tool-test",
            stage_attempt=2,
            idempotency_key="tool-idem-1",
            wave_digest="d" * 64,
            tool_id="ridge@1",
            sample_ids=("s1",),
            executor=lambda: {
                "s1": {"predicted": 21.5, "metadata": {"model": "ridge"}}
            },
        ):
            status, payload = self._request("tool-secret")
        self.assertEqual(status, 200)
        self.assertTrue(payload["accepted"])

    def test_tool_admission_failure_exposes_stable_machine_code(self) -> None:
        original_execute = self.server.dsh_tools.execute

        def reject_tool(_tool_name: str, _envelope: dict) -> dict:
            raise DshToolAdmissionClosedError("prediction binding is closed")

        self.server.dsh_tools.execute = reject_tool
        try:
            status, payload = self._request("tool-secret")
        finally:
            self.server.dsh_tools.execute = original_execute

        self.assertEqual(status, 409)
        self.assertEqual(payload["error_code"], "dsh_tool_admission_closed")

    def test_dynamic_retrieval_completion_and_replay_have_dedicated_endpoints(
        self,
    ) -> None:
        identity = _identity(
            self.server.ledger,
            role="researcher",
            stage="generation.research",
            idempotency_key="research-result-http",
        )
        request = {
            "identity": identity,
            "arguments": {
                "queries": ["greenhouse temperature forecasting"],
                "retrieval_key": "http-search",
            },
        }
        status, replay = self._post(
            "/api/ecology-agent-sidecar/v1/retrievals/replay",
            request,
        )
        self.assertEqual(status, 200)
        self.assertEqual(replay, {"found": False, "result": None})

        status, completed = self._post(
            "/api/ecology-agent-sidecar/v1/retrievals/complete",
            {
                **request,
                "primary_result": {
                    "sources": [
                        {
                            "url": "https://example.org/temperature",
                            "title": "Greenhouse temperature forecasting",
                        },
                        {
                            "url": "https://example.net/forecasting",
                            "title": "Protected crop forecasting model",
                        },
                    ],
                    "truncated": False,
                },
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(completed["provider_route"], "dsh_primary")

        request["identity"] = {
            **identity,
            "session_id": "session:http-replay",
            "ledger_expected_revision": self.server.ledger.latest_seq(),
        }
        status, replay = self._post(
            "/api/ecology-agent-sidecar/v1/retrievals/replay",
            request,
        )
        self.assertEqual(status, 200)
        self.assertTrue(replay["found"])
        self.assertEqual(replay["result"], completed)

    def test_expired_structured_result_exposes_a_stable_machine_code(self) -> None:
        monotonic_ms = [100]
        self.server.dsh_tools._monotonic_ms = lambda: monotonic_ms[0]
        fence = self.server.dsh_tools.open_admission(
            "run:tool-test",
            3,
            2,
            role="researcher",
            stage="generation.research",
            idempotency_key="deadline-result",
        )
        status, receipt = self._post(
            "/api/ecology-agent-sidecar/v1/child-reservations",
            _reservation_request(fence.admission_id),
        )
        self.assertEqual(status, 200, receipt)
        monotonic_ms[0] = 1_100

        status, payload = self._post(
            "/api/ecology-agent-sidecar/v1/structured-results",
            _admitted_research_envelope(
                self.server.ledger,
                admission_id=fence.admission_id,
            ),
        )

        self.assertEqual(status, 409)
        self.assertEqual(
            payload["error_code"],
            "structured_role_operational_timeout",
        )

    def test_structured_sidecar_omits_unapproved_machine_code(self) -> None:
        class PrivateSidecarError(RuntimeError):
            error_code = "private_internal_code"

        original_accept = self.server.dsh_tools.accept_structured

        def reject_structured(_envelope: dict) -> dict:
            raise PrivateSidecarError("bounded internal failure")

        self.server.dsh_tools.accept_structured = reject_structured
        try:
            status, payload = self._post(
                "/api/ecology-agent-sidecar/v1/structured-results",
                {},
            )
        finally:
            self.server.dsh_tools.accept_structured = original_accept

        self.assertEqual(status, 409)
        self.assertIn("error", payload)
        self.assertNotIn("error_code", payload)

    def test_structured_persistence_unavailable_is_retryable_service_error(self) -> None:
        original_accept = self.server.dsh_tools.accept_structured

        def unavailable(_envelope: dict) -> dict:
            raise DshStructuredResultPersistenceError("private sqlite detail")

        self.server.dsh_tools.accept_structured = unavailable
        try:
            status, payload = self._post(
                "/api/ecology-agent-sidecar/v1/structured-results",
                {},
            )
        finally:
            self.server.dsh_tools.accept_structured = original_accept

        self.assertEqual(status, 503)
        self.assertEqual(
            payload["error_code"],
            "structured_result_persistence_unavailable",
        )
        self.assertNotIn("private sqlite detail", str(payload))


if __name__ == "__main__":
    unittest.main()
