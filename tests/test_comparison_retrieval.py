"""Comparison evidence is frozen without disabling research or certification."""
from copy import deepcopy
from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ecologyrsi_dsh.core.immutable import freeze_json, thaw_json
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import canonical_json, digest
from ecologyrsi_dsh.core.state import RunStateReducer
from ecologyrsi_dsh.integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ecologyrsi_dsh.integrations.dsh_tools import DshFormalRetrievalIsolationError, DshToolService
from ecologyrsi_dsh.knowledge.comparison_retrieval import (
    COMPARISON_RETRIEVAL_POLICY, COMPARISON_RETRIEVAL_STAGES,
    comparison_retrieval_snapshot, freeze_comparison_retrieval_metadata,
)
from tests import test_formal_retrieval as formal_tests
from tests import test_formal_retrieval_replay as replay_tests

RUN_ID = formal_tests.RUN_ID


class ComparisonRetrievalTests(unittest.TestCase):
    request = formal_tests.FormalRetrievalTests.request
    assert_no_providers = formal_tests.FormalRetrievalTests.assert_no_providers

    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start(); self.addCleanup(env.stop)
        self.metadata = freeze_comparison_retrieval_metadata({
            "knowledge_online_enabled": True,
            "comparison_retrieval_policy": COMPARISON_RETRIEVAL_POLICY,
            "retrieval_policy": {"dynamic_max_calls_per_stage": 3, "max_results": 4},
        })
        self.snapshot = self.metadata["comparison_retrieval_snapshot"]
        self.ledger = EventLedger(":memory:")
        self.addCleanup(self.ledger.close)
        self.ledger.append(RUN_ID, "RunCreated", {"task_manifest": {"metadata": self.metadata}})
        self.fallback = Mock(side_effect=AssertionError("network fallback"))
        self.additional = Mock(side_effect=AssertionError("network additional"))
        self.last_resort = Mock(side_effect=AssertionError("Crossref network"))
        self.service = DshToolService(self.ledger, retrieval_fallback=self.fallback,
                                      retrieval_additional=self.additional,
                                      retrieval_last_resort=self.last_resort)
        self.service.open_admission(RUN_ID, 3, 2)

    def test_pair_arms_query_same_snapshot_at_multiple_steps_without_network(self):
        replies = []
        with patch.dict(os.environ, {"ECOLOGYRSI_KNOWLEDGE_ONLINE": "true"}):
            for arm in ("champion", "challenger"):
                for key, query in (("first", "ridge"), ("later", "time series split")):
                    replies.append(self.service.prepare_retrieval(self.request(
                        key, [query], idempotency_key=arm,
                        genome_digest=("a" if arm == "champion" else "b") * 64)))
        self.assertTrue(all(reply["found"] for reply in replies))
        self.assertTrue(all(reply["result"]["snapshot_digest"] == self.snapshot["snapshot_digest"]
                            for reply in replies))
        self.assertEqual(replies[0]["result"]["sources"], replies[2]["result"]["sources"])
        self.assertEqual(replies[1]["result"]["sources"], replies[3]["result"]["sources"])
        self.assert_no_providers()

    def test_research_and_local_edit_still_admit_online_at_any_step(self):
        for stage, role in (("generation.search-plan", "researcher"),
                            ("generation.research-synthesis", "researcher"),
                            ("candidate.propose", "candidate-proposer"),
                            ("candidate.local_edit", "candidate-proposer"),
                            ("generation.reflect", "generation-judge")):
            with self.subTest(stage=stage):
                self.assertIsNone(self.service.comparison_retrieval_context_for_stage(RUN_ID, stage))
                for key in ("initial", "follow-up"):
                    reply = self.service.prepare_retrieval(self.request(key, stage=stage, role=role,
                                                                       idempotency_key=stage))
                    self.assertFalse(reply["found"])
                    self.assertTrue(reply["policy"]["online_enabled"])
        self.assert_no_providers()

    def test_comparison_rejects_external_completion_and_restarts_from_manifest(self):
        for stage, role in (("sample.plan", "sample-planner"), ("sample.critic", "sample-critic"),
                            ("sample.reflect", "sample-critic"), ("generation.judge", "generation-judge")):
            request = self.request(stage=stage, role=role, idempotency_key=stage)
            with self.assertRaises(DshFormalRetrievalIsolationError):
                self.service.complete_retrieval({**request, "primary_error_code": "primary_timeout"})
        first = self.service.prepare_retrieval(self.request())
        restarted = DshToolService(self.ledger, retrieval_fallback=self.fallback,
                                   retrieval_additional=self.additional)
        restarted.open_admission(RUN_ID, 3, 2)
        with patch("ecologyrsi_dsh.knowledge.formal_retrieval.files", side_effect=AssertionError("catalog reread")):
            self.assertEqual(first, restarted.prepare_retrieval(self.request()))
        self.assert_no_providers()

    def test_context_hashes_snapshot_only_for_comparison_and_formal_has_precedence(self):
        client = SimpleNamespace(run_stage=Mock(return_value={"structured": {"ok": True},
                                                               "result_digest": digest({"ok": True})}))
        runtime = DshStructuredRoleRuntime(client, admission=self.service)
        for attempt, stage in enumerate(("sample.critic", "generation.research-synthesis"), start=3):
            runtime.run(run_id=RUN_ID, stage=stage, role="sample-critic" if stage.startswith("sample") else "researcher",
                        context={}, output_schema_id="test", run_state_revision=3, stage_attempt=attempt,
                        ledger_expected_revision=self.ledger.latest_seq(), idempotency_key=stage)
            request = client.run_stage.call_args.args[0]["request"]
            context = request["context"]
            self.assertEqual(request["context_canonical_json"], canonical_json(context))
            self.assertEqual(request["context_digest"], digest(context))
            if stage.startswith("sample"):
                self.assertEqual(context["comparison_retrieval"]["snapshot_digest"], self.snapshot["snapshot_digest"])
                self.assertEqual(context["comparison_retrieval_scope"], "methods_only_no_dataset_results")
                self.assertFalse(context["retrieval_policy"]["online_enabled"])
                self.assertNotIn("formal_retrieval", context)
            else:
                self.assertNotIn("comparison_retrieval", context)
                self.assertTrue(context["retrieval_policy"]["online_enabled"])
        self.ledger.append(RUN_ID, "FormalRetrievalSnapshotFrozen", {"snapshot": self.snapshot})
        with self.assertRaises(DshFormalRetrievalIsolationError):
            self.service.prepare_retrieval(self.request())
        with self.service.bind_formal_retrieval(RUN_ID, self.snapshot):
            self.assertIsNone(self.service.comparison_retrieval_context_for_stage(RUN_ID, "sample.plan"))
            self.assertEqual(self.service.formal_retrieval_context_for_run(RUN_ID)["snapshot_digest"],
                             self.snapshot["snapshot_digest"])

    def test_metadata_legacy_missing_unknown_tamper_and_frozen_copy(self):
        self.assertEqual(freeze_comparison_retrieval_metadata({"seed": 0}), {"seed": 0})
        self.assertIsNone(comparison_retrieval_snapshot({}))
        self.assertEqual(comparison_retrieval_snapshot(freeze_json(self.metadata)), self.snapshot)
        for metadata in ({"comparison_retrieval_policy": "unknown@1"},
                         {"comparison_retrieval_policy": COMPARISON_RETRIEVAL_POLICY},
                         {"comparison_retrieval_snapshot": self.snapshot}):
            with self.assertRaises(ValueError):
                comparison_retrieval_snapshot(metadata)
        altered = deepcopy(self.metadata)
        altered["comparison_retrieval_snapshot"]["documents"][0]["summary"] = "changed"
        with self.assertRaises(ValueError):
            comparison_retrieval_snapshot(altered)
        self.assertEqual(freeze_comparison_retrieval_metadata({**self.metadata, "online_snippets": ["observed=123"]})
                         ["comparison_retrieval_snapshot"], self.snapshot)

    def test_replay_uses_manifest_snapshot_and_rejects_wrong_route_digest_or_results(self):
        fixture = replay_tests.FormalRetrievalReplayTests()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        payload = thaw_json(fixture.prefix[0].payload)
        payload["task_manifest"]["metadata"].update(self.metadata)
        created = replace(fixture.prefix[0], payload=payload)
        fixture.snapshot = self.snapshot
        def reducer():
            return RunStateReducer(created)
        fixture.apply(reducer(), "DshRetrievalExecuted", fixture.retrieval())
        for change in ("digest", "results", "online"):
            event = fixture.retrieval()
            if change == "digest":
                event["frozen_snapshot_digest"] = "0" * 64
            elif change == "results":
                event["result"]["sources"] = []
                event["result_digest"] = digest(event["result"])
            else:
                event["provider_route"] = "dsh_primary"
                event.pop("frozen_snapshot_digest")
            with self.subTest(change=change), self.assertRaises(ValueError):
                fixture.apply(reducer(), "DshRetrievalExecuted", event)


if __name__ == "__main__":
    unittest.main()
