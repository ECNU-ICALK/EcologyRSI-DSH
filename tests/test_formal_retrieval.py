from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import canonical_json, digest
from ecologyrsi_dsh.integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ecologyrsi_dsh.integrations.dsh_tools import (
    DshFormalRetrievalIsolationError, DshRetrievalBudgetError,
    DshToolAdmissionClosedError, DshToolService,
)
from ecologyrsi_dsh.knowledge.formal_retrieval import (
    build_formal_retrieval_snapshot, search_formal_retrieval_snapshot,
    validate_formal_retrieval_snapshot,
)
from tests.test_dsh_tool_contracts import _identity


RUN_ID = "run:tool-test"


class FormalRetrievalTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        configured = patch("ecologyrsi_dsh.knowledge.deepseek_search.is_deepseek_search_configured",
                           return_value=False)
        configured.start()
        self.addCleanup(configured.stop)
        self.ledger = EventLedger(":memory:")
        self.addCleanup(self.ledger.close)
        self.metadata = {"knowledge_online_enabled": True, "retrieval_policy": {
            "dynamic_max_calls_per_stage": 2, "max_results": 4,
        }}
        self.ledger.append(RUN_ID, "RunCreated", {"task_manifest": {"metadata": self.metadata}})
        self.fallback = Mock(side_effect=AssertionError("network fallback was called"))
        self.additional = Mock(side_effect=AssertionError("network additional was called"))
        self.last_resort = Mock(side_effect=AssertionError("Crossref network was called"))
        self.service = DshToolService(self.ledger, retrieval_fallback=self.fallback,
                                      retrieval_additional=self.additional,
                                      retrieval_last_resort=self.last_resort)
        self.service.open_admission(RUN_ID, 3, 2)

    def freeze(self):
        snapshot = build_formal_retrieval_snapshot(self.metadata)
        self.ledger.append(RUN_ID, "FormalRetrievalSnapshotFrozen", {"snapshot": snapshot})
        return snapshot

    def request(self, key="lookup", queries=None, **identity):
        return {"identity": _identity(self.ledger, **identity),
                "arguments": {"queries": queries or ["ridge regularization"],
                              "retrieval_key": key}}

    def assert_no_providers(self):
        self.fallback.assert_not_called()
        self.additional.assert_not_called()
        self.last_resort.assert_not_called()

    def test_builder_reads_only_curated_methods_and_frozen_budget(self):
        first = build_formal_retrieval_snapshot(self.metadata)
        with patch.dict(os.environ, {"ECOLOGYRSI_KNOWLEDGE_ONLINE": "true"}):
            second = build_formal_retrieval_snapshot({
                **self.metadata, "model_output": "ground_truth=99",
                "online_snippets": [{"observed": 99}], "sample_results": [99],
            })
        self.assertEqual(first, second)
        self.assertEqual(len(first["documents"]), 8)
        self.assertFalse(first["retrieval_policy"]["online_enabled"])
        self.assertNotIn("ground_truth", str(first))
        self.assertEqual(first["snapshot_digest"],
                         digest({k: v for k, v in first.items() if k != "snapshot_digest"}))

    def test_snapshot_rejects_tampering_extra_fields_and_online_policy(self):
        original = build_formal_retrieval_snapshot(self.metadata)
        for mutate, rehash in (
            (lambda s: s["documents"][0].update(summary="poisoned"), False),
            (lambda s: s.update(online_snippets=[]), True),
            (lambda s: s["documents"][0].update(observed=123), True),
            (lambda s: s["retrieval_policy"].update(online_enabled=True), True),
            (lambda s: s["documents"].pop(), True),
        ):
            snapshot = deepcopy(original)
            mutate(snapshot)
            if rehash:
                snapshot["snapshot_digest"] = digest({k: v for k, v in snapshot.items()
                                                       if k != "snapshot_digest"})
            with self.subTest(snapshot=snapshot.keys()), self.assertRaises(ValueError):
                validate_formal_retrieval_snapshot(snapshot)

    def test_search_reuses_content_without_rereading_catalog_or_fetching_sources(self):
        snapshot = build_formal_retrieval_snapshot(self.metadata)
        with patch("ecologyrsi_dsh.knowledge.formal_retrieval.files",
                   side_effect=AssertionError("live catalog reread")):
            result = search_formal_retrieval_snapshot(snapshot, ["ridge"], limit=4)
            empty = search_formal_retrieval_snapshot(snapshot, ["zzzzunknown987"], limit=4)
        self.assertEqual(result["sources"][0]["title"], "岭回归残差预测")
        self.assertEqual(empty["sources"], [])
        self.assert_no_providers()

    def test_local_query_always_returns_found_and_snapshot_identity(self):
        snapshot = self.freeze()
        with self.service.bind_formal_retrieval(RUN_ID, snapshot):
            context = self.service.formal_retrieval_context_for_run(RUN_ID)
            with patch.dict(os.environ, {"ECOLOGYRSI_KNOWLEDGE_ONLINE": "true"}):
                reply = self.service.prepare_retrieval(self.request())
                empty = self.service.prepare_retrieval(self.request("none", ["zzzzunknown987"]))
            self.assertEqual(context, {"mode": "frozen_snapshot",
                "snapshot_digest": snapshot["snapshot_digest"], "document_count": 8,
                "online_enabled": False})
            self.assertTrue(reply["found"])
            self.assertFalse(reply["policy"]["online_enabled"])
            self.assertEqual(reply["result"]["provider_route"], "frozen_snapshot")
            self.assertEqual(reply["result"]["snapshot_digest"], snapshot["snapshot_digest"])
            self.assertTrue(reply["result"]["sources"])
            self.assertTrue(empty["found"])
            self.assertEqual(empty["result"]["sources"], [])
        events = self.ledger.events_by_kind(RUN_ID, "DshRetrievalExecuted")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].payload["frozen_snapshot_digest"], snapshot["snapshot_digest"])
        self.assertNotIn("snapshot_digest", events[0].payload["result"])
        self.assert_no_providers()

    def test_completed_local_replay_is_idempotent_and_budget_stays_frozen(self):
        snapshot = self.freeze()
        with self.service.bind_formal_retrieval(RUN_ID, snapshot):
            original = self.service.prepare_retrieval(self.request())
            self.service.prepare_retrieval(self.request("second", ["soil water"]))
            with self.assertRaises(DshRetrievalBudgetError):
                self.service.prepare_retrieval(self.request("third"))
            replay = self.service.prepare_retrieval(self.request())
            self.assertEqual(original, replay)
            self.assertEqual(self.service.stage_retrieval_policy_for_run(RUN_ID).max_results, 4)
        self.assertEqual(len(self.ledger.events_by_kind(RUN_ID, "DshRetrievalExecuted")), 2)
        self.assert_no_providers()

    def test_local_lookup_works_when_global_online_is_disabled(self):
        snapshot = self.freeze()
        with self.service.bind_formal_retrieval(RUN_ID, snapshot):
            with patch.dict(os.environ, {"ECOLOGYRSI_KNOWLEDGE_ONLINE": "false"}):
                self.assertTrue(self.service.prepare_retrieval(self.request())["found"])
        self.assert_no_providers()

    def test_structured_request_hashes_formal_context_without_changing_identity(self):
        snapshot = self.freeze()
        client = SimpleNamespace(run_stage=Mock(return_value={
            "structured": {"ok": True}, "result_digest": digest({"ok": True}),
        }))
        identity = {"genome_digest": "a" * 64, "compiled_behavior_digest": "b" * 64,
                    "phenotype_instance_digest": "c" * 64}
        with self.service.bind_formal_retrieval(RUN_ID, snapshot):
            with patch.dict(os.environ, {"ECOLOGYRSI_KNOWLEDGE_ONLINE": "true"}):
                DshStructuredRoleRuntime(client, admission=self.service).run(
                    run_id=RUN_ID, stage="sample.plan", role="sample-planner",
                    context={"evaluation_scope": {"phase": "validation"}},
                    output_schema_id="ecology-sample-predictions@2",
                    run_state_revision=3, stage_attempt=2,
                    ledger_expected_revision=self.ledger.latest_seq(),
                    idempotency_key="formal-wave", identity_digests=identity,
                )
        request = client.run_stage.call_args.args[0]["request"]
        context = request["context"]
        self.assertEqual(context["formal_retrieval"]["snapshot_digest"], snapshot["snapshot_digest"])
        self.assertFalse(context["retrieval_policy"]["online_enabled"])
        self.assertEqual(context["retrieval_policy"], snapshot["retrieval_policy"])
        self.assertEqual(request["context_canonical_json"], canonical_json(context))
        self.assertEqual(request["context_digest"], digest(context))
        self.assertEqual(request["identity_digests"], identity)
        self.assert_no_providers()

    def test_completion_cannot_submit_external_result_even_after_local_replay(self):
        snapshot = self.freeze()
        with self.service.bind_formal_retrieval(RUN_ID, snapshot):
            self.service.prepare_retrieval(self.request())
            for arguments in (
                {"primary_result": {"sources": [], "truncated": False}},
                {"primary_error_code": "primary_timeout"},
            ):
                with self.assertRaises(DshFormalRetrievalIsolationError):
                    self.service.complete_retrieval({**self.request(), **arguments})
        self.assertEqual(len(self.ledger.events_by_kind(RUN_ID, "DshRetrievalExecuted")), 1)
        self.assert_no_providers()

    def test_durable_formal_marker_without_binding_fails_closed(self):
        self.ledger.append(RUN_ID, "FormalStageFrozen", {"stage": "validation"})
        with patch.dict(os.environ, {"ECOLOGYRSI_KNOWLEDGE_ONLINE": "true"}):
            with self.assertRaises(DshFormalRetrievalIsolationError):
                self.service.prepare_retrieval(self.request())
            with self.assertRaises(DshFormalRetrievalIsolationError):
                self.service.complete_retrieval({**self.request(), "primary_error_code": "primary_timeout"})
        self.assert_no_providers()

    def test_unbound_or_restarted_formal_worker_cannot_reenter_online_mode(self):
        snapshot = self.freeze()
        with self.service.bind_formal_retrieval(RUN_ID, snapshot):
            self.service.prepare_retrieval(self.request())
        restarted = DshToolService(self.ledger, retrieval_fallback=self.fallback,
                                  retrieval_additional=self.additional)
        restarted.open_admission(RUN_ID, 3, 2)
        for service in (self.service, restarted):
            with self.assertRaises(DshFormalRetrievalIsolationError):
                service.prepare_retrieval(self.request())
            with self.assertRaises(DshFormalRetrievalIsolationError):
                service.formal_retrieval_context_for_run(RUN_ID)
        self.assert_no_providers()

    def test_binding_requires_exact_persisted_snapshot(self):
        snapshot = build_formal_retrieval_snapshot(self.metadata)
        with self.assertRaises(DshFormalRetrievalIsolationError):
            with self.service.bind_formal_retrieval(RUN_ID, snapshot):
                self.fail("unpersisted snapshot admitted")
        self.freeze()
        snapshot["documents"][0]["summary"] = "different method text"
        snapshot["snapshot_digest"] = digest({k: v for k, v in snapshot.items() if k != "snapshot_digest"})
        with self.assertRaises(DshFormalRetrievalIsolationError):
            with self.service.bind_formal_retrieval(RUN_ID, snapshot):
                self.fail("different snapshot admitted")

    def test_sealed_stage_rejects_queries_while_bound(self):
        snapshot = self.freeze()
        self.ledger.append(RUN_ID, "FormalStageFrozen", {"stage": "validation"})
        with self.service.bind_formal_retrieval(RUN_ID, snapshot):
            self.ledger.append(RUN_ID, "FormalStageSealed", {"stage": "validation"})
            with self.assertRaises(DshFormalRetrievalIsolationError):
                self.service.prepare_retrieval(self.request())
        self.assert_no_providers()

    def test_same_snapshot_can_bind_validation_and_final_test_replicas(self):
        snapshot = self.freeze()
        for stage in ("validation", "final_test"):
            self.ledger.append(RUN_ID, "FormalStageFrozen", {"stage": stage})
            with self.service.bind_formal_retrieval(RUN_ID, snapshot):
                for replica in (0, 1):
                    reply = self.service.prepare_retrieval(
                        self.request(idempotency_key=f"{stage}-{replica}"))
                    self.assertEqual(reply["result"]["snapshot_digest"], snapshot["snapshot_digest"])
            self.ledger.append(RUN_ID, "FormalStageSealed", {"stage": stage})
        self.assertEqual(len(self.ledger.events_by_kind(RUN_ID, "DshRetrievalExecuted")), 4)
        self.assert_no_providers()

    def test_concurrent_replays_persist_only_one_local_execution(self):
        snapshot = self.freeze()
        request = self.request()
        with self.service.bind_formal_retrieval(RUN_ID, snapshot):
            with ThreadPoolExecutor(max_workers=6) as pool:
                replies = list(pool.map(lambda _: self.service.prepare_retrieval(request), range(6)))
        self.assertTrue(all(reply == replies[0] for reply in replies))
        self.assertEqual(len(self.ledger.events_by_kind(RUN_ID, "DshRetrievalExecuted")), 1)
        self.assert_no_providers()

    def test_pause_during_lookup_prevents_durable_result(self):
        snapshot = self.freeze()
        real_search = search_formal_retrieval_snapshot
        def close_during_search(*args, **kwargs):
            result = real_search(*args, **kwargs)
            self.service.close_run_admissions(RUN_ID)
            return result
        with self.service.bind_formal_retrieval(RUN_ID, snapshot):
            with patch("ecologyrsi_dsh.integrations.dsh_tools.search_formal_retrieval_snapshot",
                       side_effect=close_during_search):
                with self.assertRaises(DshToolAdmissionClosedError):
                    self.service.prepare_retrieval(self.request())
        self.assertEqual(self.ledger.events_by_kind(RUN_ID, "DshRetrievalExecuted"), ())
        self.assert_no_providers()

    def test_old_online_cache_cannot_be_replayed_into_formal_run(self):
        request = self.request()
        # A valid historical online event predates the formal boundary.
        self.service._retrieval_additional = None
        self.service._retrieval_fallback = Mock(return_value={
            "sources": [{"url": "https://example.org/old", "title": "old online evidence"}],
            "truncated": False,
        })
        self.service.complete_retrieval({**request, "primary_error_code": "primary_timeout"})
        snapshot = self.freeze()
        with self.service.bind_formal_retrieval(RUN_ID, snapshot):
            with self.assertRaises(DshFormalRetrievalIsolationError):
                self.service.prepare_retrieval(request)
            with self.assertRaises(DshFormalRetrievalIsolationError):
                self.service.complete_retrieval({**request, "primary_error_code": "primary_timeout"})


if __name__ == "__main__":
    unittest.main()
