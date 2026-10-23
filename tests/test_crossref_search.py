from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.state import _validate_dsh_retrieval_event
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
from ecologyrsi_dsh.knowledge import crossref_search as crossref
from ecologyrsi_dsh.knowledge.retrieval import retrieve_generation_knowledge
from tests.test_knowledge_retrieval import _task, _JsonResponse
from tests.test_retrieval_policy import RetrievalPolicyTests, _sources


class CrossrefSearchTests(unittest.TestCase):
    def test_metadata_preserves_absent_abstract_and_labels_jats_abstract(self):
        payload = {"status": "ok", "message": {"items": [
            {"DOI": "10.1234/ABC", "title": ["A title"]},
            {"DOI": "10.1234/XYZ", "title": ["B title"],
             "abstract": "<jats:p>Observed &amp; bounded evidence.</jats:p>"},
            {"DOI": "not-a-doi", "title": ["Reject"]},
        ]}}
        with patch.object(crossref, "urlopen", return_value=_JsonResponse(payload)) as transport:
            cards = crossref.crossref_cards(("greenhouse climate",), limit=4)
            result = crossref.search_crossref_metadata(("greenhouse climate",), limit=4)
        self.assertEqual(len(cards), 2)
        self.assertIsNone(cards[0].abstract_summary)
        self.assertIsNone(cards[0].content_digest)
        self.assertIn("未提供摘要", result["sources"][0]["snippet"])
        self.assertIn("Observed & bounded evidence.", cards[1].abstract_summary)
        self.assertNotIn("<jats", cards[1].abstract_summary)
        self.assertTrue(all(not card.executable for card in cards))
        self.assertTrue(all(card.execution_status == "metadata_only" for card in cards))
        request = transport.call_args.args[0]
        self.assertEqual(request.host, "api.crossref.org")
        self.assertIsNone(request.get_header("Authorization"))

    def test_invalid_input_never_reaches_network(self):
        with patch.object(crossref, "urlopen") as transport:
            for queries, limit in [("query", 2), ([], 2), ([""], 2), (["x\n"], 2),
                                   (["x" * 601], 2), (["x"] * 13, 2), (["x"], True),
                                   (["x"], 0), (["x"], 33)]:
                with self.subTest(queries=queries, limit=limit), self.assertRaises(crossref.CrossrefSearchError):
                    crossref.crossref_cards(queries, limit=limit)
            transport.assert_not_called()

    def test_http_failure_is_bounded_and_sanitized(self):
        error = HTTPError("https://api.crossref.org/works", 429, "sensitive", {}, None)
        with patch.object(crossref, "urlopen", side_effect=error) as transport:
            with self.assertRaisesRegex(crossref.CrossrefSearchError, "^http_429$"):
                crossref.crossref_cards(("greenhouse",))
        self.assertEqual(transport.call_count, 1)

    def test_malformed_response_fails_safely(self):
        for payload in [[], {"status": "ok", "message": {}}, {"status": "failed"}]:
            with self.subTest(payload=payload), patch.object(crossref, "urlopen", return_value=_JsonResponse(payload)):
                with self.assertRaisesRegex(crossref.CrossrefSearchError, "invalid_response"):
                    crossref.crossref_cards(("greenhouse",))

    def test_query_budget_deduplication_and_partial_failure(self):
        a = SimpleNamespace(knowledge_id="a")
        b = SimpleNamespace(knowledge_id="b")
        def response(query, limit):
            if query == "first":
                raise crossref.CrossrefSearchError("http_503")
            return [a, b] if query == "second" else [a]
        with patch.object(crossref, "_cards", side_effect=response) as transport:
            cards = crossref.crossref_cards(["first", "second", "third", "fourth"])
        self.assertEqual(transport.call_count, 3)
        self.assertEqual([c.knowledge_id for c in cards], ["a", "b"])

    def test_host_uses_crossref_only_when_other_online_sources_have_no_cards(self):
        ledger = EventLedger(); self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, StrategyRouterDSHAdapter())
        director.start_evolution(_task(online=True), run_id="run:crossref")
        card = crossref.KnowledgeCard(
            knowledge_id="crossref:10.1234/a", title="Paper", summary="Metadata only",
            source_url="https://doi.org/10.1234/a", source_kind="论文元数据",
            source_authority="Crossref", execution_status="metadata_only", selection_reason="research")
        with patch("ecologyrsi_dsh.knowledge.deepseek_search.is_deepseek_search_configured", return_value=False), \
             patch("ecologyrsi_dsh.knowledge.retrieval._openalex_cards_for_queries", side_effect=TimeoutError), \
             patch.object(crossref, "crossref_cards", return_value=[card]) as fallback:
            snapshot = retrieve_generation_knowledge(director.state("run:crossref"))
        fallback.assert_called_once()
        self.assertEqual(snapshot.retrieval_status, "catalog_and_online")
        self.assertIn("Crossref", snapshot.provider)
        self.assertTrue(snapshot.warnings)
        self.assertIn(card, snapshot.cards)
        with patch("ecologyrsi_dsh.knowledge.deepseek_search.is_deepseek_search_configured", return_value=False), \
             patch("ecologyrsi_dsh.knowledge.retrieval._openalex_cards_for_queries", return_value=[card]), \
             patch.object(crossref, "crossref_cards") as fallback:
            retrieve_generation_knowledge(director.state("run:crossref"))
        fallback.assert_not_called()


class CrossrefDynamicTests(unittest.TestCase):
    setUp = RetrievalPolicyTests.setUp
    service = RetrievalPolicyTests.service
    request = staticmethod(RetrievalPolicyTests.request)

    def test_dynamic_all_online_empty_uses_last_resort_and_replays_without_network(self):
        ledger, service, fallback = self.service(additional=Mock(return_value=_sources(0)))
        fallback.side_effect = TimeoutError
        last = service._retrieval_last_resort = Mock(return_value=_sources(2, prefix="crossref"))
        request = self.request(ledger)
        result = service.complete_retrieval({**request, "primary_error_code": "primary_provider_error"})
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["provider_route"], "dsh_primary_with_deepseek_then_openalex_then_crossref_fallback")
        event = ledger.events_by_kind("run:tool-test", "DshRetrievalExecuted")[0]
        _validate_dsh_retrieval_event(event.payload, run_id="run:tool-test")
        self.assertEqual(service.prepare_retrieval(request)["result"], result)
        last.assert_called_once()

    def test_partial_existing_online_evidence_does_not_trigger_crossref(self):
        ledger, service, fallback = self.service(additional=Mock(return_value=_sources(1)))
        fallback.return_value = _sources(0)
        service._retrieval_last_resort = Mock()
        service.complete_retrieval({**self.request(ledger), "primary_error_code": "primary_provider_error"})
        service._retrieval_last_resort.assert_not_called()

    def test_last_resort_failure_retains_empty_results_and_safe_diagnostic(self):
        ledger, service, fallback = self.service()
        fallback.return_value = _sources(0)
        service._retrieval_last_resort = Mock(side_effect=RuntimeError("private detail"))
        result = service.complete_retrieval({**self.request(ledger), "primary_error_code": "primary_provider_error"})
        self.assertEqual(result["sources"], [])
        event = ledger.events_by_kind("run:tool-test", "DshRetrievalExecuted")[0]
        self.assertEqual(event.payload["provider_diagnostics"]["last_resort_error_code"], "crossref_provider_error")
        self.assertNotIn("private detail", json.dumps(event.payload))
        _validate_dsh_retrieval_event(event.payload, run_id="run:tool-test")

    def test_legacy_sample_without_frozen_policy_does_not_expand_to_crossref(self):
        ledger, service, fallback = self.service()
        fallback.return_value = _sources(0)
        service._retrieval_last_resort = Mock()
        request = self.request(ledger, stage="sample.plan")
        request["identity"]["role"] = "sample-planner"
        service.complete_retrieval({**request, "primary_error_code": "primary_provider_error"})
        service._retrieval_last_resort.assert_not_called()
