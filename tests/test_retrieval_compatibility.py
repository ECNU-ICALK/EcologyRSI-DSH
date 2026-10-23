"""Retrieval expansion preserves frozen identities and usable evidence breadth."""

from __future__ import annotations

from copy import deepcopy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
from ecologyrsi_dsh.knowledge import retrieval
from ecologyrsi_dsh.knowledge.models import (
    KnowledgeCard,
    KnowledgeSnapshot,
    validate_knowledge_context,
)


_LEGACY_SNAPSHOT = {
    "schema_version": "ecologyrsi-dsh.knowledge-snapshot/1",
    "run_id": "run:historical-evidence",
    "generation": 1,
    "query_terms": ["soil moisture forecasting"],
    "cards": [{
        "knowledge_id": "openalex:W123",
        "title": "Historical paper",
        "summary": "Frozen research clue.",
        "source_url": "https://example.org/paper",
        "source_kind": "paper metadata",
        "source_authority": "OpenAlex",
        "execution_status": "metadata_only",
        "executable": False,
        "selection_reason": "Research only.",
        "algorithm_tags": ["online_metadata"],
        "capability_kind": None,
        "capability_id": None,
        "parameter_hints": [],
        "publication_year": 2025,
        "cited_by_count": 3,
    }],
    "online_enabled": True,
    "provider": "内置核验目录 + OpenAlex 元数据",
    "retrieval_status": "catalog_and_online",
    "warnings": [],
    "retrieved_at": "2026-08-25T00:00:00+00:00",
    # Fixed using the historical /1 snapshot wire format, before this expansion.
    "snapshot_digest": "2bd3bdc62e222d58f2df2d57ae1c011a9bf09bd04999b6c5c1d21cfa39284a68",
}


def _state(**policy):
    return SimpleNamespace(
        task_manifest=SimpleNamespace(
            domain_pack="greenhouse_environment@1",
            metadata={
                "domain": "greenhouse",
                "knowledge_online_enabled": True,
                "retrieval_policy": policy,
            },
        ),
        run=SimpleNamespace(run_id="run:retrieval-compatibility", generation=0),
    )


def _paper(index: int) -> KnowledgeCard:
    return KnowledgeCard(
        knowledge_id=f"openalex:W{index:02}",
        title=f"Paper {index}",
        summary="Short paper evidence.",
        source_url=f"https://papers.example/{index}",
        source_kind="paper metadata",
        source_authority="OpenAlex",
        execution_status="metadata_only",
        selection_reason="Research only.",
    )


class RetrievalCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        for guard in (
            patch.dict(os.environ, {
                "ECOLOGYRSI_DSH_CREDENTIALS_FILE": "/nonexistent/ecology-test-credentials",
                "ECOLOGYRSI_DSH_SETTINGS_FILE": "/nonexistent/ecology-test-settings",
            }),
            patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network is forbidden")),
            patch(
                "ecologyrsi_dsh.knowledge.deepseek_search.is_deepseek_search_configured",
                return_value=False,
            ),
        ):
            guard.start()
            self.addCleanup(guard.stop)

    def _identity(self, metadata: dict) -> dict:
        parent = SimpleNamespace(genome_digest="1" * 64)
        task = SimpleNamespace(metadata=metadata)
        run = SimpleNamespace(run_id="run:retrieval-compatibility", generation=2)
        context = {"objective": "compare forecasting models", "generation": 2}
        before = deepcopy(context)
        with patch(
            "ecologyrsi_dsh.evolution.strategies.compile_plugin_behavior",
            return_value=SimpleNamespace(compiled_behavior_digest="2" * 64),
        ):
            result = StrategyRouterDSHAdapter._native_stage_identity(
                parent, task, run, "generation.research-synthesis", context
            )
        self.assertEqual(context, before)
        return result

    def test_legacy_stage_identity_remains_the_historical_digest(self) -> None:
        # A run created before retrieval_policy existed must resume with the old
        # context identity, even if the deployment's online override changes.
        for online_override in ("0", "1"):
            with patch.dict(os.environ, {"ECOLOGYRSI_KNOWLEDGE_ONLINE": online_override}):
                identity = self._identity({"knowledge_online_enabled": True})
            self.assertEqual(
                identity["phenotype_instance_digest"],
                "f65c0c6cbe0ca64178300cb3f07d1fd411fd94bb42148e7d363817064d08eebe",
            )
            self.assertEqual(identity["genome_digest"], "1" * 64)
            self.assertEqual(identity["compiled_behavior_digest"], "2" * 64)

    def test_explicit_policy_is_bound_to_new_stage_identity(self) -> None:
        metadata = {"knowledge_online_enabled": True, "retrieval_policy": {}}
        for online_override in ("0", "1"):
            with patch.dict(os.environ, {"ECOLOGYRSI_KNOWLEDGE_ONLINE": online_override}):
                identity = self._identity(metadata)
            self.assertEqual(
                identity["phenotype_instance_digest"],
                "d6a20ff75b4002cba05caf06a780e8b708a5d42d68024e17f8b36c73d8e5001c",
            )
        changed = self._identity({
            **metadata, "retrieval_policy": {"max_results": 24},
        })
        self.assertNotEqual(
            changed["phenotype_instance_digest"], identity["phenotype_instance_digest"]
        )

    def test_bootstrap_bounds_host_hints_for_both_providers(self) -> None:
        state = _state(max_query_chars=20, max_queries_per_call=3)
        with (
            patch.dict(os.environ, {"ECOLOGYRSI_KNOWLEDGE_ONLINE": "1"}),
            patch.object(retrieval, "_openalex_cards_for_queries", return_value=[]) as papers,
            patch(
                "ecologyrsi_dsh.knowledge.deepseek_search.is_deepseek_search_configured",
                return_value=True,
            ),
            patch(
                "ecologyrsi_dsh.knowledge.deepseek_search.search_deepseek_metadata",
                return_value={"sources": [], "truncated": False},
            ) as web,
        ):
            snapshot = retrieval.retrieve_generation_knowledge(
                state, query_terms=("soil water",)
            )
        self.assertTrue(snapshot.query_terms)
        self.assertEqual(snapshot.query_terms[0], "soil water")
        self.assertLessEqual(len(snapshot.query_terms), 3)
        self.assertTrue(all(len(query) <= 20 for query in snapshot.query_terms))
        self.assertEqual(papers.call_args.args[0], snapshot.query_terms)
        self.assertEqual(web.call_args.args[0], snapshot.query_terms)

    def test_authored_queries_above_policy_are_rejected_before_provider_calls(self) -> None:
        state = _state(max_query_chars=20, max_queries_per_call=1)
        with patch.object(retrieval, "_openalex_cards_for_queries") as papers:
            for queries in (("x" * 21,), ("soil", "water")):
                with self.subTest(queries=queries):
                    with self.assertRaisesRegex(ValueError, "configured retrieval policy"):
                        retrieval.retrieve_generation_knowledge(state, query_terms=queries)
            papers.assert_not_called()

    def test_thirty_two_cards_reach_context_with_provider_interleaving(self) -> None:
        state = _state(max_results=32)
        paper_cards = [_paper(index) for index in range(32)]
        web_sources = [{
            "url": f"https://web.example/{index}",
            "title": f"Web source {index}",
            "snippet": "Short web evidence.",
        } for index in range(32)]
        with (
            patch.dict(os.environ, {"ECOLOGYRSI_KNOWLEDGE_ONLINE": "1"}),
            patch.object(retrieval, "_openalex_cards_for_queries", return_value=paper_cards),
            patch(
                "ecologyrsi_dsh.knowledge.deepseek_search.is_deepseek_search_configured",
                return_value=True,
            ),
            patch(
                "ecologyrsi_dsh.knowledge.deepseek_search.search_deepseek_metadata",
                return_value={"sources": web_sources, "truncated": False},
            ),
        ):
            snapshot = retrieval.retrieve_generation_knowledge(
                state, query_terms=("forecast model alternatives",)
            )
        self.assertEqual(len(snapshot.cards), 32)
        online = [
            card for card in snapshot.cards
            if card.knowledge_id.startswith(("deepseek:", "openalex:"))
        ]
        self.assertGreater(len(online), 16)
        for index, card in enumerate(online):
            expected_host = "web.example" if index % 2 == 0 else "papers.example"
            self.assertEqual(card.source_url, f"https://{expected_host}/{index // 2}")
            self.assertFalse(card.executable)
            self.assertIsNone(card.capability_id)
        context = snapshot.proposal_context()
        self.assertEqual(len(context["evidence_catalog"]), 32)
        self.assertEqual(
            [item["knowledge_id"] for item in context["evidence_catalog"]],
            [card.knowledge_id for card in snapshot.cards],
        )
        self.assertEqual(validate_knowledge_context(context), context)

    def test_legacy_snapshot_roundtrip_keeps_its_original_digest(self) -> None:
        historical = deepcopy(_LEGACY_SNAPSHOT)
        snapshot = KnowledgeSnapshot.from_dict(historical)
        self.assertEqual(snapshot.snapshot_digest, _LEGACY_SNAPSHOT["snapshot_digest"])
        self.assertEqual(snapshot.to_dict(), _LEGACY_SNAPSHOT)
        self.assertEqual(historical, _LEGACY_SNAPSHOT)
        self.assertEqual(
            validate_knowledge_context(snapshot.proposal_context()),
            snapshot.proposal_context(),
        )


if __name__ == "__main__":
    unittest.main()
