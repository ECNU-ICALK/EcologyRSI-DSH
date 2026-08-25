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
from ecologyrsi_dsh.api.handler import EvolutionHTTPServer
from ecologyrsi_dsh.integrations.dsh_structured_roles import DshStructuredRoleRuntime


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
                    "idempotency_key": "check-temperature-literature",
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
                "idempotency_key": "verify-proposal-evidence",
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
                        "idempotency_key": f"judge-search-{index}",
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
                        "idempotency_key": "judge-search-3",
                    },
                    "primary_error_code": "primary_timeout",
                }
            )
        self.assertEqual(fallback_calls, 3)

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
        }

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
            self.service.accept_structured(
                {
                    "identity": identity,
                    "output_schema_id": "ecology-sample-decisions@1",
                    "structured": structured,
                    "result_digest": digest(structured),
                    "skill_invocation_evidence": _skill_evidence("sample.plan"),
                }
            )

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
        self.service.accept_structured(
            {
                "identity": identity,
                "output_schema_id": "ecology-research-result@1",
                "structured": structured,
                "result_digest": digest(structured),
                "skill_invocation_evidence": _skill_evidence(
                    "generation.research"
                ),
            }
        )
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
        corrupt_event_id = (
            f"{corrupt_run}:dsh-structured:"
            f"{digest({'stage': 'generation.research', 'idempotency_key': corrupt_key})}"
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

        accepted = self.service.accept_structured(
            {
                "identity": identity,
                "output_schema_id": "ecology-sample-reflection@1",
                "structured": structured,
                "result_digest": digest(structured),
                "skill_invocation_evidence": _skill_evidence("sample.reflect"),
            }
        )

        self.assertTrue(accepted["accepted"])
        event = self.ledger.events("run:tool-test")[-1]
        self.assertEqual(event.kind, "DshStructuredResultAccepted")
        self.assertEqual(event.payload["identity"]["stage"], "sample.reflect")
        self.assertEqual(
            event.payload["output_schema_id"], "ecology-sample-reflection@1"
        )

    def test_cross_language_schemas_are_closed_and_role_sets_match(self) -> None:
        root = Path(__file__).resolve().parents[1] / "integrations" / "dsh_ecology_plugin"
        schemas = sorted((root / "schemas").glob("*.schema.json"))
        self.assertEqual(len(schemas), 12)
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


if __name__ == "__main__":
    unittest.main()
