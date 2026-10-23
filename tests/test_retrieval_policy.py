from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ecologyrsi_dsh.api.handler import EvolutionRequestHandler
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import canonical_json, digest
from ecologyrsi_dsh.core.state import _validate_dsh_retrieval_event
from ecologyrsi_dsh.integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ecologyrsi_dsh.integrations.dsh_tools import (
    DshRetrievalBudgetError,
    DshRetrievalDisabledError,
    DshToolAdmissionClosedError,
    DshToolAuthorizationError,
    DshToolService,
)
from ecologyrsi_dsh.knowledge.retrieval_policy import (
    resolve_retrieval_policy,
    validate_retrieval_policy,
)
from tests.test_dsh_tool_contracts import _identity


def _sources(count: int, *, prefix: str = "source") -> dict:
    return {
        "sources": [
            {"url": f"https://example.org/{prefix}-{index}",
             "title": f"greenhouse forecasting source {index}"}
            for index in range(count)
        ],
        "truncated": False,
    }


class RetrievalPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        configured = patch(
            "ecologyrsi_dsh.knowledge.deepseek_search.is_deepseek_search_configured",
            return_value=False,
        )
        configured.start()
        self.addCleanup(configured.stop)

    def service(self, *, online=True, policy=None, additional=None):
        ledger = EventLedger(":memory:")
        self.addCleanup(ledger.close)
        ledger.append("run:tool-test", "RunCreated", {
            "task_manifest": {"metadata": {
                "knowledge_online_enabled": online, "retrieval_policy": policy or {},
            }},
        })
        fallback = Mock(return_value=_sources(2, prefix="fallback"))
        service = DshToolService(
            ledger, retrieval_fallback=fallback, retrieval_additional=additional
        )
        service.open_admission("run:tool-test", 3, 2)
        return ledger, service, fallback

    @staticmethod
    def request(ledger, *, key="lookup", stage="generation.search-plan", queries=None):
        return {
            "identity": _identity(ledger, role="researcher", stage=stage),
            "arguments": {"queries": queries or ["greenhouse forecasting"], "retrieval_key": key},
        }

    def test_shared_switch_and_strict_bounded_configuration(self):
        default = resolve_retrieval_policy({}, environ={})
        self.assertFalse(default.online_enabled)
        self.assertEqual(default.dynamic_max_calls_per_stage, 8)
        self.assertEqual(default.max_results, 16)
        self.assertFalse(resolve_retrieval_policy(
            {"knowledge_online_enabled": True}, environ={"ECOLOGYRSI_KNOWLEDGE_ONLINE": "off"}
        ).online_enabled)
        self.assertTrue(resolve_retrieval_policy(
            {}, environ={"ECOLOGYRSI_KNOWLEDGE_ONLINE": "true"}
        ).online_enabled)
        for value in (None, [], {"unknown": 2}, {"max_results": True},
                      {"max_results": 33}, {"max_queries_per_call": 0},
                      {"max_query_chars": 601}, {"dynamic_max_calls_per_stage": 25}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_retrieval_policy(value)

    def test_offline_rejects_preflight_and_direct_completion_before_provider(self):
        ledger, service, fallback = self.service(online=False)
        request = self.request(ledger)
        with self.assertRaises(DshRetrievalDisabledError):
            service.prepare_retrieval(request)
        with self.assertRaises(DshRetrievalDisabledError):
            service.complete_retrieval({**request, "primary_error_code": "primary_timeout"})
        fallback.assert_not_called()
        self.assertEqual(ledger.retrieval_stage_count(request["identity"]), 0)

    def test_budget_is_reserved_before_concurrent_network_calls(self):
        ledger, service, fallback = self.service(policy={"dynamic_max_calls_per_stage": 1})
        def prepare(index):
            try:
                return service.prepare_retrieval(self.request(ledger, key=f"lookup-{index}"))
            except DshRetrievalBudgetError:
                return None
        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(prepare, range(6)))
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(ledger.events_by_kind("run:tool-test", "DshRetrievalExecuted"), ())
        fallback.assert_not_called()

    def test_closing_bound_stage_or_run_releases_abandoned_reservations(self):
        ledger, service, _ = self.service()
        request = self.request(ledger)
        service.open_admission(
            "run:tool-test", 3, 2, role="researcher", stage="generation.search-plan",
            idempotency_key=request["identity"]["idempotency_key"],
        )
        service.prepare_retrieval(request)
        self.assertTrue(service._retrieval_reservations)
        service.close_admission("run:tool-test", 3, 2)
        self.assertFalse(service._retrieval_reservations)
        ledger, service, _ = self.service()
        service.prepare_retrieval(self.request(ledger))
        service.close_run_admissions("run:tool-test")
        self.assertFalse(service._retrieval_reservations)

    def test_completed_cache_replays_when_offline_and_budget_is_exhausted(self):
        ledger, service, _ = self.service(policy={"dynamic_max_calls_per_stage": 1})
        request = self.request(ledger)
        policy = service.prepare_retrieval(request)["policy"]
        self.assertEqual(policy["dynamic_max_calls_per_stage"], 1)
        completed = service.complete_retrieval({**request, "primary_result": _sources(2)})
        with self.assertRaises(DshRetrievalBudgetError):
            service.prepare_retrieval(self.request(ledger, key="another"))
        with patch.dict(os.environ, {"ECOLOGYRSI_KNOWLEDGE_ONLINE": "false"}):
            replay = service.prepare_retrieval(request)
        self.assertTrue(replay["found"])
        self.assertEqual(replay["result"], completed)
        self.assertFalse(replay["policy"]["online_enabled"])

    def test_reserved_key_cannot_change_queries_and_policy_bounds_apply_before_network(self):
        ledger, service, _ = self.service(policy={
            "max_queries_per_call": 1, "max_query_chars": 30,
        })
        request = self.request(ledger)
        service.prepare_retrieval(request)
        with self.assertRaisesRegex(ValueError, "idempotency"):
            service.prepare_retrieval(self.request(ledger, queries=["soil moisture"]))
        for queries in (["a", "b"], ["x" * 31]):
            with self.subTest(queries=queries), self.assertRaises(DshToolAuthorizationError):
                service.prepare_retrieval(self.request(ledger, key="bounds", queries=queries))

    def test_synthesis_and_search_plan_both_allow_expanded_queries_and_result_replay(self):
        for stage in ("generation.search-plan", "generation.research-synthesis"):
            with self.subTest(stage=stage):
                ledger, service, _ = self.service(policy={"max_results": 32})
                request = self.request(
                    ledger, stage=stage,
                    queries=[f"greenhouse forecasting model {index} " + "x" * 200 for index in range(8)],
                )
                service.prepare_retrieval(request)
                completed = service.complete_retrieval({**request, "primary_result": _sources(32)})
                self.assertEqual(len(completed["sources"]), 32)
                event = ledger.events_by_kind("run:tool-test", "DshRetrievalExecuted")[0]
                _validate_dsh_retrieval_event(event.payload, run_id="run:tool-test")

    def test_additional_source_runs_even_when_primary_sufficient(self):
        additional = Mock(return_value=_sources(2, prefix="deepseek"))
        ledger, service, fallback = self.service(additional=additional)
        completed = service.complete_retrieval({
            **self.request(ledger), "primary_result": _sources(2),
        })
        additional.assert_called_once_with(("greenhouse forecasting",), limit=16)
        fallback.assert_not_called()
        self.assertEqual(len(completed["sources"]), 4)
        self.assertEqual(completed["provider_route"], "dsh_primary_with_deepseek")
        _validate_dsh_retrieval_event(
            ledger.events_by_kind("run:tool-test", "DshRetrievalExecuted")[0].payload,
            run_id="run:tool-test",
        )

    def test_full_primary_result_does_not_crowd_out_supplemental_sources(self):
        ledger, service, _ = self.service(additional=Mock(return_value=_sources(2, prefix="deepseek")))
        completed = service.complete_retrieval({
            **self.request(ledger), "primary_result": _sources(16),
        })
        self.assertEqual(len(completed["sources"]), 16)
        self.assertEqual(sum("deepseek-" in source["url"] for source in completed["sources"]), 2)

    def test_additional_evidence_can_satisfy_failed_primary_without_openalex(self):
        ledger, service, fallback = self.service(additional=Mock(return_value=_sources(2)))
        completed = service.complete_retrieval({
            **self.request(ledger), "primary_error_code": "primary_timeout",
        })
        fallback.assert_not_called()
        self.assertIsNone(completed["fallback_reason"])
        self.assertEqual(completed["provider_route"], "dsh_primary_with_deepseek")
        self.assertEqual(completed["provider_diagnostics"]["primary_error_code"], "primary_timeout")

    def test_additional_failure_retains_safe_diagnostic_without_exception_details(self):
        ledger, service, fallback = self.service(
            additional=Mock(side_effect=RuntimeError("secret credential and remote body"))
        )
        completed = service.complete_retrieval({
            **self.request(ledger), "primary_result": _sources(2),
        })
        fallback.assert_not_called()
        self.assertEqual(completed["provider_diagnostics"]["additional_error_code"], "deepseek_provider_error")
        self.assertNotIn("secret", str(completed))
        _validate_dsh_retrieval_event(
            ledger.events_by_kind("run:tool-test", "DshRetrievalExecuted")[0].payload,
            run_id="run:tool-test",
        )

    def test_provider_work_does_not_hold_global_admission_lock(self):
        entered = threading.Event()
        release = threading.Event()
        def additional(_queries, *, limit):
            entered.set()
            if not release.wait(5):
                raise AssertionError("test did not release provider")
            return _sources(2)
        ledger, service, _ = self.service(additional=additional)
        with ThreadPoolExecutor(max_workers=2) as executor:
            running = executor.submit(service.complete_retrieval, {
                **self.request(ledger), "primary_result": _sources(2),
            })
            self.assertTrue(entered.wait(2))
            try:
                other = executor.submit(service.prepare_retrieval, self.request(ledger, key="other"))
                self.assertFalse(other.result(timeout=2)["found"])
            finally:
                release.set()
            running.result(timeout=2)

    def test_closed_stage_cannot_persist_in_flight_provider_result(self):
        ledger, service, fallback = self.service()
        def additional(_queries, *, limit):
            service.close_admission("run:tool-test", 3, 2)
            return _sources(2)
        service._retrieval_additional = additional
        with self.assertRaises(DshToolAdmissionClosedError):
            service.complete_retrieval({**self.request(ledger), "primary_result": _sources(2)})
        self.assertEqual(ledger.events_by_kind("run:tool-test", "DshRetrievalExecuted"), ())
        fallback.assert_not_called()

    def test_insufficient_combined_evidence_uses_openalex(self):
        ledger, service, fallback = self.service(additional=Mock(return_value=_sources(1)))
        completed = service.complete_retrieval({
            **self.request(ledger), "primary_result": _sources(0),
        })
        fallback.assert_called_once()
        self.assertEqual(completed["provider_route"], "dsh_primary_with_deepseek_then_openalex_fallback")
        _validate_dsh_retrieval_event(
            ledger.events_by_kind("run:tool-test", "DshRetrievalExecuted")[0].payload,
            run_id="run:tool-test",
        )

    def test_policy_is_included_in_stage_context_before_canonical_hashing(self):
        ledger, service, _ = self.service(policy={"dynamic_max_calls_per_stage": 12})
        client = SimpleNamespace(run_stage=Mock(return_value={
            "structured": {"ok": True}, "result_digest": digest({"ok": True}),
        }))
        runtime = DshStructuredRoleRuntime(client, admission=service)
        runtime.run(
            run_id="run:tool-test", stage="generation.search-plan", role="researcher",
            context={"retrieval_policy": {"online_enabled": False}},
            output_schema_id="ecology-research-search-plan@1", run_state_revision=3,
            stage_attempt=2, ledger_expected_revision=ledger.latest_seq(), idempotency_key="stage-policy",
        )
        request = client.run_stage.call_args.args[0]["request"]
        self.assertTrue(request["context"]["retrieval_policy"]["online_enabled"])
        self.assertEqual(request["context"]["retrieval_policy"]["dynamic_max_calls_per_stage"], 12)
        self.assertEqual(request["context_canonical_json"], canonical_json(request["context"]))
        self.assertEqual(request["context_digest"], digest(request["context"]))

    def test_stage_policy_is_frozen_while_network_switch_remains_live(self):
        _, service, _ = self.service()
        frozen = service.stage_retrieval_policy_for_run("run:tool-test")
        with patch.dict(os.environ, {"ECOLOGYRSI_KNOWLEDGE_ONLINE": "false"}):
            self.assertEqual(service.stage_retrieval_policy_for_run("run:tool-test"), frozen)
            self.assertFalse(service.retrieval_policy_for_run("run:tool-test").online_enabled)

    def test_legacy_stage_context_remains_unchanged_for_accepted_result_replay(self):
        ledger = EventLedger(":memory:")
        self.addCleanup(ledger.close)
        ledger.append("run:tool-test", "RunCreated", {"task_manifest": {"metadata": {}}})
        service = DshToolService(ledger)
        client = SimpleNamespace(run_stage=Mock(return_value={
            "structured": {"ok": True}, "result_digest": digest({"ok": True}),
        }))
        DshStructuredRoleRuntime(client, admission=service).run(
            run_id="run:tool-test", stage="generation.search-plan", role="researcher",
            context={"legacy": True}, output_schema_id="ecology-research-search-plan@1",
            run_state_revision=3, stage_attempt=2, ledger_expected_revision=ledger.latest_seq(),
            idempotency_key="legacy-stage",
        )
        request = client.run_stage.call_args.args[0]["request"]
        self.assertEqual(request["context"], {"legacy": True})
        self.assertEqual(request["context_digest"], digest({"legacy": True}))

    def test_api_rejects_malformed_compact_and_manifest_policies_before_binding(self):
        handler = object.__new__(EvolutionRequestHandler)
        for body in (
            {"retrieval_policy": {"max_results": 1000}},
            {"task_manifest": {"metadata": {"retrieval_policy": {"unknown": 1}}}},
        ):
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, "retrieval_policy"):
                handler._task_from_request(body)

    def test_new_api_runs_freeze_defaults_and_partial_overrides_in_manifest(self):
        handler = object.__new__(EvolutionRequestHandler)
        handler._bind_runtime_task = Mock(side_effect=lambda task, **_kwargs: task)
        for manifest in (False, True):
            for custom in (None, {"dynamic_max_calls_per_stage": 12}):
                body = {
                    "dataset_id": "generated-toy-series@1",
                    **({"retrieval_policy": custom} if custom is not None else {}),
                }
                if manifest:
                    body["task_manifest"] = {
                        "task_id": "retrieval-policy-api", "objective": "forecast",
                        "domain_pack": "crop_soil_water",
                        "visible_datasets": ["generated-toy-series@1"], "budget": {"max_candidates": 1},
                    }
                with self.subTest(manifest=manifest, custom=custom):
                    task = handler._task_from_request(body)
                    self.assertEqual(task.metadata["retrieval_policy"], {
                        "dynamic_max_calls_per_stage": 12 if custom else 8,
                        "max_queries_per_call": 8, "max_query_chars": 320, "max_results": 16,
                    })


if __name__ == "__main__":
    unittest.main()
