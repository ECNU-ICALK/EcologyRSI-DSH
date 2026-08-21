from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace
from unittest import mock
from urllib.error import HTTPError, URLError
from urllib.parse import unquote

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import TaskManifest, canonical_json, digest
from ecologyrsi_dsh.evolution.analysis import GenerationAnalysis
from ecologyrsi_dsh.evolution.batches import start_generation_batch
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
from ecologyrsi_dsh.knowledge import retrieval as knowledge_retrieval
from ecologyrsi_dsh.knowledge.mapping import map_catalog_entry
from ecologyrsi_dsh.knowledge.models import (
    KnowledgeCard,
    KnowledgeSnapshot,
    validate_knowledge_context,
)
from ecologyrsi_dsh.knowledge.retrieval import (
    assess_generation_knowledge,
    retrieve_generation_knowledge,
)


def _task(*, online: bool = False) -> TaskManifest:
    return TaskManifest(
        task_id="knowledge-loop",
        objective="改进作物土壤水分预测",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={
            "max_candidates": 3,
            "max_generations": 1,
            "candidates_per_generation": 3,
        },
        metadata={
            "domain": "toy",
            "strategy_id": "adaptive_local@1",
            "prediction_model_id": "toy-rolling-water@1",
            "evaluator_id": "toy_time_forward@1",
            "knowledge_online_enabled": online,
        },
    )


class _JsonResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")
        self.read_limits: list[int] = []

    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        self.read_limits.append(limit)
        return self._payload


def _evidence_snapshot() -> KnowledgeSnapshot:
    abstract = "Ordered abstract content"
    cards = (
        KnowledgeCard(
            knowledge_id="knowledge:ridge",
            title="Ridge evidence",
            summary="Generic ridge summary",
            source_url="https://example.test/ridge",
            source_kind="official documentation",
            source_authority="Example",
            execution_status="adopted",
            selection_reason="mapped",
            capability_kind="predictor",
            capability_id="greenhouse-exogenous-ridge@1",
            capability_ids=(
                "greenhouse-exogenous-ridge@1",
                "greenhouse-targetwise-ridge@1",
            ),
            parameter_hints=("ridge_alpha",),
            abstract_summary=(
                "OpenAlex 摘要（metadata_only；未经全文核验）：" + abstract
            ),
            content_digest=digest(abstract),
        ),
        KnowledgeCard(
            knowledge_id="knowledge:metadata",
            title="Metadata evidence",
            summary="Metadata fallback summary",
            source_url="https://example.test/metadata",
            source_kind="paper metadata",
            source_authority="Example",
            execution_status="metadata_only",
            selection_reason="research only",
        ),
    )
    return KnowledgeSnapshot(
        run_id="run:evidence",
        generation=0,
        query_terms=("ridge forecasting",),
        cards=cards,
        online_enabled=True,
        provider="test",
        retrieval_status="catalog_and_online",
        retrieved_at="2026-08-18T00:00:00+00:00",
    )


def _refresh_evidence_digest(item: dict) -> None:
    identity = {key: value for key, value in item.items() if key != "evidence_digest"}
    item["evidence_digest"] = digest(identity)


class KnowledgeRetrievalTests(unittest.TestCase):
    def test_tool_performance_without_sample_failures_preserves_metric_focus(
        self,
    ) -> None:
        analysis = GenerationAnalysis(
            run_id="run:knowledge-assessment",
            generation=0,
            candidate_count=1,
            eligible_count=0,
            outcome="no_eligible_candidate",
            next_generation_focus="优先改善 室内二氧化碳浓度 的 1 小时时距",
            sample_failures=(
                {
                    "attempted": 9,
                    "succeeded": 9,
                    "failed": 0,
                    "skipped": 0,
                    "coverage_pass": True,
                    "tool_performance": [{"tool_id": "ridge", "failed": 0}],
                },
            ),
        )
        state = SimpleNamespace(
            run=SimpleNamespace(run_id="run:knowledge-assessment"),
            evaluation_for=lambda _candidate_id: None,
        )

        assessment = assess_generation_knowledge(
            state,
            analysis,
            _evidence_snapshot(),
        )

        self.assertEqual(
            assessment.next_action,
            "优先改善 室内二氧化碳浓度 的 1 小时时距",
        )

    def test_openalex_abstract_is_reconstructed_and_frozen_as_metadata(self) -> None:
        response = _JsonResponse(
            {
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "display_name": "Soil moisture forecasting with ridge models",
                        "publication_year": 2025,
                        "cited_by_count": 7,
                        "primary_location": None,
                        "abstract_inverted_index": {
                            "models": [4],
                            "Forecasting": [0],
                            "with": [3],
                            "moisture": [2],
                            "soil": [1],
                        },
                    }
                ]
            }
        )
        with (
            mock.patch.dict(
                knowledge_retrieval.os.environ,
                {"ECOLOGYRSI_OPENALEX_TIMEOUT": ""},
                clear=False,
            ),
            mock.patch.object(
                knowledge_retrieval,
                "urlopen",
                return_value=response,
            ) as urlopen,
        ):
            cards = knowledge_retrieval._openalex_cards(
                "soil moisture ridge",
                limit=1,
                required_title_markers=("soil moisture",),
            )

        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertEqual(card.execution_status, "metadata_only")
        self.assertIn("metadata_only", card.abstract_summary or "")
        self.assertIn("未经全文核验", card.abstract_summary or "")
        self.assertIn("Forecasting soil moisture with models", card.abstract_summary or "")
        self.assertEqual(
            card.content_digest,
            digest("Forecasting soil moisture with models"),
        )
        requested_url = urlopen.call_args.args[0].full_url
        self.assertIn("abstract_inverted_index", unquote(requested_url))
        self.assertEqual(
            urlopen.call_args.kwargs["timeout"],
            knowledge_retrieval._OPENALEX_DEFAULT_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            response.read_limits,
            [knowledge_retrieval._OPENALEX_MAX_RESPONSE_BYTES],
        )
        self.assertEqual(KnowledgeCard.from_dict(card.to_dict()), card)

    def test_openalex_transient_errors_retry_with_short_exponential_backoff(
        self,
    ) -> None:
        response = _JsonResponse({"results": []})
        transient_failures = [
            TimeoutError("queued"),
            URLError("connection reset"),
            HTTPError(
                "https://api.openalex.org/works",
                429,
                "rate limited",
                {},
                None,
            ),
            response,
        ]
        with (
            mock.patch.dict(
                knowledge_retrieval.os.environ,
                {"ECOLOGYRSI_OPENALEX_TIMEOUT": "37.5"},
                clear=False,
            ),
            mock.patch.object(
                knowledge_retrieval,
                "urlopen",
                side_effect=transient_failures,
            ) as urlopen,
            mock.patch.object(knowledge_retrieval.time, "sleep") as sleep,
        ):
            cards = knowledge_retrieval._openalex_cards("soil water")

        self.assertEqual(cards, [])
        self.assertEqual(urlopen.call_count, 4)
        self.assertEqual(
            [call.kwargs["timeout"] for call in urlopen.call_args_list],
            [37.5, 37.5, 37.5, 37.5],
        )
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [0.25, 0.5, 1.0],
        )

    def test_openalex_retry_after_is_respected_and_bounded(self) -> None:
        response = _JsonResponse({"results": []})
        for raw_retry_after, expected_delay in (
            ("3.5", 3.5),
            ("3600", knowledge_retrieval._OPENALEX_RETRY_AFTER_MAX_SECONDS),
        ):
            with (
                self.subTest(retry_after=raw_retry_after),
                mock.patch.object(
                    knowledge_retrieval,
                    "urlopen",
                    side_effect=[
                        HTTPError(
                            "https://api.openalex.org/works",
                            429,
                            "rate limited",
                            {"Retry-After": raw_retry_after},
                            None,
                        ),
                        response,
                    ],
                ),
                mock.patch.object(knowledge_retrieval.time, "sleep") as sleep,
            ):
                self.assertEqual(
                    knowledge_retrieval._openalex_cards("soil water"),
                    [],
                )

            sleep.assert_called_once_with(expected_delay)

    def test_openalex_retry_after_supports_http_date(self) -> None:
        error = HTTPError(
            "https://api.openalex.org/works",
            503,
            "unavailable",
            {"Retry-After": "Thu, 01 Jan 1970 00:16:44 GMT"},
            None,
        )
        with mock.patch.object(knowledge_retrieval.time, "time", return_value=1000.0):
            self.assertEqual(
                knowledge_retrieval._openalex_retry_after_seconds(error),
                4.0,
            )
            self.assertEqual(
                knowledge_retrieval._openalex_retry_delay(error, 0),
                4.0,
            )

    def test_openalex_retries_all_transient_http_status_classes(self) -> None:
        for status in (408, 425, 429, 500, 503, 599):
            with self.subTest(status=status):
                error = HTTPError(
                    "https://api.openalex.org/works",
                    status,
                    "transient",
                    {},
                    None,
                )
                self.assertTrue(knowledge_retrieval._retryable_openalex_error(error))
        for status in (400, 401, 403, 404, 499):
            with self.subTest(status=status):
                error = HTTPError(
                    "https://api.openalex.org/works",
                    status,
                    "permanent",
                    {},
                    None,
                )
                self.assertFalse(knowledge_retrieval._retryable_openalex_error(error))

    def test_openalex_retry_budget_is_three_retries(self) -> None:
        with (
            mock.patch.object(
                knowledge_retrieval,
                "urlopen",
                side_effect=TimeoutError("still queued"),
            ) as urlopen,
            mock.patch.object(knowledge_retrieval.time, "sleep") as sleep,
            self.assertRaises(TimeoutError),
        ):
            knowledge_retrieval._openalex_cards("soil water")

        self.assertEqual(urlopen.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [0.25, 0.5, 1.0],
        )

    def test_openalex_non_retryable_http_error_falls_back_immediately(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, StrategyRouterDSHAdapter())
        director.start_evolution(_task(online=True), run_id="run:http-fallback")
        error = HTTPError(
            "https://api.openalex.org/works",
            404,
            "not found with sensitive upstream detail",
            {},
            None,
        )
        with (
            mock.patch.object(
                knowledge_retrieval,
                "urlopen",
                side_effect=error,
            ) as urlopen,
            mock.patch.object(knowledge_retrieval.time, "sleep") as sleep,
        ):
            snapshot = retrieve_generation_knowledge(
                director.state("run:http-fallback")
            )

        self.assertEqual(snapshot.retrieval_status, "catalog_online_fallback")
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()
        self.assertNotIn("sensitive upstream detail", " ".join(snapshot.warnings))
        ledger.close()

    def test_openalex_abstract_reconstruction_is_position_and_size_bounded(self) -> None:
        inverted = {f"term{index}": [index] for index in reversed(range(700))}

        abstract = knowledge_retrieval._reconstruct_openalex_abstract(inverted)

        self.assertIsNotNone(abstract)
        terms = (abstract or "").split()
        self.assertEqual(len(terms), 512)
        self.assertEqual(terms[0], "term0")
        self.assertEqual(terms[-1], "term511")
        summary = knowledge_retrieval._openalex_abstract_summary(abstract or "")
        self.assertLessEqual(len(summary), 1800)
        self.assertTrue(summary.endswith("..."))

    def test_openalex_query_plan_is_short_deterministic_and_deduplicated(self) -> None:
        long_query = "soil " + "water " * 100
        queries = (
            "  soil   moisture forecasting  ",
            "SOIL MOISTURE FORECASTING",
            long_query,
            "irrigation calibration",
            "crop water prediction",
            "evapotranspiration model",
            "time series validation",
            "unused seventh query",
        )

        first = knowledge_retrieval._openalex_query_plan(queries)
        second = knowledge_retrieval._openalex_query_plan(queries)

        self.assertEqual(first, second)
        self.assertEqual(first[-1], "soil moisture forecasting")
        self.assertEqual(len(first), knowledge_retrieval._OPENALEX_MAX_QUERIES)
        self.assertEqual(len({item.casefold() for item in first}), len(first))
        self.assertTrue(
            all(
                len(item) <= knowledge_retrieval._OPENALEX_QUERY_MAX_CHARS
                for item in first
            )
        )

    def test_openalex_short_queries_fall_back_and_deduplicate_work_ids(self) -> None:
        def metadata_card(work_id: str) -> KnowledgeCard:
            return KnowledgeCard(
                knowledge_id=f"openalex:{work_id}",
                title=f"Soil moisture study {work_id}",
                summary="metadata only",
                source_url=f"https://openalex.org/{work_id}",
                source_kind="paper metadata",
                source_authority="OpenAlex",
                execution_status="metadata_only",
                selection_reason="not mapped",
                algorithm_tags=("online_metadata",),
            )

        calls: list[str] = []

        def search(
            query: str,
            *,
            limit: int,
            required_title_markers: tuple[str, ...],
        ) -> list[KnowledgeCard]:
            calls.append(query)
            self.assertEqual(required_title_markers, ("soil moisture",))
            results = {
                "broad domain query": [metadata_card("W1"), metadata_card("W2")],
                "weak target query": [],
                "horizon query": [metadata_card("W1")],
            }[query]
            return results[:limit]

        with mock.patch.object(
            knowledge_retrieval,
            "_openalex_cards",
            side_effect=search,
        ):
            cards = knowledge_retrieval._openalex_cards_for_queries(
                (
                    "broad domain query",
                    "weak target query",
                    "horizon query",
                ),
                limit=2,
                required_title_markers=("soil moisture",),
            )

        self.assertEqual(calls, ["weak target query", "horizon query", "broad domain query"])
        self.assertEqual([card.knowledge_id for card in cards], ["openalex:W1", "openalex:W2"])

    def test_proposal_context_exposes_digest_bound_frozen_evidence(self) -> None:
        context = _evidence_snapshot().proposal_context()

        self.assertEqual(
            context["retrieval"],
            {
                "online_enabled": True,
                "status": "catalog_and_online",
                "provider": "test",
                "warnings": [],
            },
        )
        catalog = context["evidence_catalog"]
        self.assertEqual(len(catalog), 2)
        self.assertEqual(
            set(catalog[0]),
            {
                "knowledge_id",
                "title",
                "summary",
                "source_url",
                "source_kind",
                "source_authority",
                "execution_status",
                "capability_kind",
                "capability_id",
                "capability_ids",
                "parameter_hints",
                "evidence_digest",
            },
        )
        self.assertIn("Ordered abstract content", catalog[0]["summary"])
        self.assertEqual(
            catalog[0]["capability_ids"],
            [
                "greenhouse-exogenous-ridge@1",
                "greenhouse-targetwise-ridge@1",
            ],
        )
        self.assertEqual(catalog[1]["summary"], "Metadata fallback summary")
        for item in catalog:
            identity = {
                key: value for key, value in item.items() if key != "evidence_digest"
            }
            self.assertEqual(item["evidence_digest"], digest(identity))
        self.assertEqual(validate_knowledge_context(context), context)

    def test_evidence_catalog_validator_rejects_tampering_and_invalid_shapes(self) -> None:
        original = _evidence_snapshot().proposal_context()
        mutations = {}

        duplicate = copy.deepcopy(original)
        duplicate["evidence_catalog"].append(
            copy.deepcopy(duplicate["evidence_catalog"][0])
        )
        mutations["duplicate id"] = duplicate

        insecure_url = copy.deepcopy(original)
        insecure_url["evidence_catalog"][0]["source_url"] = "http://example.test/ridge"
        _refresh_evidence_digest(insecure_url["evidence_catalog"][0])
        mutations["http url"] = insecure_url

        changed_digest = copy.deepcopy(original)
        changed_digest["evidence_catalog"][0]["summary"] = "tampered"
        mutations["digest mismatch"] = changed_digest

        unknown_field = copy.deepcopy(original)
        unknown_field["evidence_catalog"][0]["unexpected"] = "value"
        mutations["unknown field"] = unknown_field

        missing_field = copy.deepcopy(original)
        missing_field["evidence_catalog"][0].pop("source_kind")
        mutations["missing field"] = missing_field

        blank_capability = copy.deepcopy(original)
        blank_capability["evidence_catalog"][0]["capability_ids"] = [""]
        _refresh_evidence_digest(blank_capability["evidence_catalog"][0])
        mutations["blank capability"] = blank_capability

        unknown_status = copy.deepcopy(original)
        unknown_status["evidence_catalog"][0]["execution_status"] = "executable"
        _refresh_evidence_digest(unknown_status["evidence_catalog"][0])
        mutations["unknown status"] = unknown_status

        unknown_retrieval_status = copy.deepcopy(original)
        unknown_retrieval_status["retrieval"]["status"] = "online_ok"
        mutations["unknown retrieval status"] = unknown_retrieval_status

        malformed_retrieval = copy.deepcopy(original)
        malformed_retrieval["retrieval"]["unexpected"] = True
        mutations["malformed retrieval"] = malformed_retrieval

        too_many = copy.deepcopy(original)
        too_many["evidence_catalog"] = []
        for index in range(25):
            item = copy.deepcopy(original["evidence_catalog"][0])
            item["knowledge_id"] = f"knowledge:{index}"
            _refresh_evidence_digest(item)
            too_many["evidence_catalog"].append(item)
        mutations["too many"] = too_many

        for label, context in mutations.items():
            with self.subTest(label=label), self.assertRaises((TypeError, ValueError)):
                validate_knowledge_context(context)

    def test_evidence_catalog_and_total_context_have_encoded_size_limits(self) -> None:
        original = _evidence_snapshot().proposal_context()
        oversized_catalog = copy.deepcopy(original)
        oversized_catalog["query_terms"] = []
        oversized_catalog["adopted_knowledge"] = []
        oversized_catalog["research_only_knowledge"] = []
        oversized_catalog["evidence_catalog"] = []
        for index in range(24):
            item = copy.deepcopy(original["evidence_catalog"][0])
            item["knowledge_id"] = f"knowledge:{index}"
            item["summary"] = "s" * 1200
            item["source_authority"] = "a" * 400
            _refresh_evidence_digest(item)
            oversized_catalog["evidence_catalog"].append(item)
        catalog_bytes = len(
            canonical_json(oversized_catalog["evidence_catalog"]).encode("utf-8")
        )
        context_bytes = len(canonical_json(oversized_catalog).encode("utf-8"))
        self.assertGreater(catalog_bytes, 48_000)
        self.assertLess(context_bytes, 64_000)
        with self.assertRaisesRegex(ValueError, "evidence_catalog is too large"):
            validate_knowledge_context(oversized_catalog)

        oversized_context = copy.deepcopy(original)
        oversized_context["query_terms"] = ["q" * 64_000]
        with self.assertRaisesRegex(ValueError, "knowledge_snapshot is too large"):
            validate_knowledge_context(oversized_context)

    def test_legacy_cards_and_context_keep_their_original_shape(self) -> None:
        card = KnowledgeCard(
            knowledge_id="legacy:card",
            title="Legacy card",
            summary="Legacy summary",
            source_url="https://example.test/legacy",
            source_kind="documentation",
            source_authority="Example",
            execution_status="research_only",
            selection_reason="legacy",
            capability_kind="predictor",
            capability_id="greenhouse-exogenous-ridge@1",
        )
        serialized_card = card.to_dict()
        self.assertNotIn("abstract_summary", serialized_card)
        self.assertNotIn("content_digest", serialized_card)
        self.assertNotIn("capability_ids", serialized_card)
        identity = {
            "schema_version": "ecologyrsi-dsh.knowledge-snapshot/1",
            "run_id": "run:legacy",
            "generation": 0,
            "query_terms": ["legacy"],
            "cards": [serialized_card],
            "online_enabled": False,
            "provider": "legacy",
            "retrieval_status": "catalog_only",
            "warnings": [],
            "retrieved_at": "2026-08-18T00:00:00+00:00",
        }
        payload = {**identity, "snapshot_digest": digest(identity)}

        restored = KnowledgeSnapshot.from_dict(payload)

        self.assertEqual(restored.to_dict(), payload)
        legacy_context = restored.proposal_context()
        legacy_context.pop("evidence_catalog")
        self.assertEqual(validate_knowledge_context(legacy_context), legacy_context)

    def test_ridge_evidence_maps_to_all_registered_ridge_predictors(self) -> None:
        entry = next(
            item
            for item in knowledge_retrieval._catalog()
            if item["knowledge_id"] == "sklearn-ridge"
        )
        card = map_catalog_entry(
            entry,
            {
                "prediction_model_id": "greenhouse-targetwise-ridge@1",
                "strategy_id": "adaptive_local@1",
                "evaluator_id": "greenhouse_time_forward@1",
            },
        )

        self.assertEqual(card.execution_status, "adopted")
        self.assertEqual(card.capability_id, "greenhouse-targetwise-ridge@1")
        self.assertEqual(
            card.capability_ids,
            (
                "greenhouse-exogenous-ridge@1",
                "greenhouse-targetwise-ridge@1",
                "greenhouse-horizon-targetwise-ridge@1",
            ),
        )

    def test_failure_and_horizon_terms_are_sanitized_into_online_query(self) -> None:
        task_data = _task(online=True).to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "strategy_id": "autonomous_model@1",
        }
        task = TaskManifest.from_dict(task_data)
        analysis = GenerationAnalysis(
            run_id="run:query-feedback",
            generation=0,
            candidate_count=1,
            eligible_count=0,
            outcome="no_eligible_candidate",
            target_weaknesses=(
                {"target": "soil_moisture", "horizon_hours": 24},
            ),
            horizon_weaknesses=({"horizon_hours": 168},),
            algorithm_failures=(
                {"failure_code": "adapter_timeout'); DROP TABLE runs; --"},
            ),
            sample_failures=(
                {"failure_counts": {"tool_timeout<script>": 3}},
            ),
        )

        class _State:
            task_manifest = task
            run = SimpleNamespace(run_id="run:query-feedback", generation=1)

            @staticmethod
            def analysis_for(generation: int) -> GenerationAnalysis | None:
                return analysis if generation == 0 else None

        with mock.patch(
            "ecologyrsi_dsh.knowledge.retrieval._openalex_cards",
            return_value=[],
        ) as online_search:
            snapshot = retrieve_generation_knowledge(_State())

        terms = " ".join(snapshot.query_terms)
        online_queries = [call.args[0] for call in online_search.call_args_list]
        self.assertEqual(
            online_queries,
            list(knowledge_retrieval._openalex_query_plan(snapshot.query_terms)),
        )
        for expected in (
            "168 hour horizon",
            "adapter timeout drop table runs",
            "tool timeout script",
        ):
            self.assertIn(expected, terms)
            self.assertTrue(any(expected in query for query in online_queries))
        for unsafe in ("'", ";", "<", ">", ")"):
            self.assertNotIn(unsafe, terms)
            self.assertNotIn(unsafe, " ".join(online_queries))
        self.assertNotIn(" ".join(snapshot.query_terms), online_queries)

    def test_autonomous_strategy_is_a_registered_executable_capability(self) -> None:
        task = _task()
        task = TaskManifest(
            **{
                **task.to_dict(),
                "metadata": {
                    **task.metadata,
                    "strategy_id": "autonomous_model@1",
                },
            }
        )
        card = map_catalog_entry(
            {
                "knowledge_id": "local:autonomous",
                "title_zh": "自主策略",
                "summary_zh": "已登记的自主策略能力",
                "source_url": "https://example.invalid/autonomous",
                "source_kind": "内置目录",
                "source_authority": "EcologyRSI",
                "algorithm_tags": ["strategy"],
                "execution_mapping": {
                    "kind": "strategy",
                    "capability_id": "autonomous_model@1",
                },
            },
            task.metadata,
        )
        self.assertEqual(card.execution_status, "adopted")
        self.assertEqual(card.capability_id, "autonomous_model@1")
        self.assertEqual(card.capability_ids, ("autonomous_model@1",))

    def test_batch_freezes_one_shared_knowledge_snapshot(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, StrategyRouterDSHAdapter())
        director.start_evolution(_task(), run_id="run:knowledge")

        first = start_generation_batch(director, "run:knowledge")
        second = start_generation_batch(director, "run:knowledge")
        state = director.state("run:knowledge")

        self.assertEqual(first, second)
        self.assertEqual(len(state.knowledge_snapshots), 1)
        self.assertEqual(
            first.knowledge_snapshot_digest,
            state.knowledge_snapshots[0].snapshot_digest,
        )
        self.assertTrue(state.knowledge_snapshots[0].executable_cards)
        self.assertEqual(
            sum(
                event.kind == "GenerationKnowledgeRetrieved"
                for event in state.events
            ),
            1,
        )
        ledger.close()

    def test_online_failure_falls_back_without_stopping_the_round(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, StrategyRouterDSHAdapter())
        director.start_evolution(_task(online=True), run_id="run:fallback")
        with mock.patch(
            "ecologyrsi_dsh.knowledge.retrieval._openalex_cards",
            side_effect=TimeoutError("offline"),
        ):
            snapshot = retrieve_generation_knowledge(director.state("run:fallback"))

        self.assertEqual(snapshot.retrieval_status, "catalog_online_fallback")
        self.assertTrue(snapshot.warnings)
        self.assertTrue(snapshot.executable_cards)
        retrieval = snapshot.proposal_context()["retrieval"]
        self.assertTrue(retrieval["online_enabled"])
        self.assertEqual(retrieval["status"], "catalog_online_fallback")
        self.assertTrue(retrieval["warnings"])
        ledger.close()

    def test_online_metadata_never_becomes_executable(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, StrategyRouterDSHAdapter())
        director.start_evolution(_task(online=True), run_id="run:metadata")
        online_card = {
            "knowledge_id": "openalex:W1",
            "title": "Unverified algorithm",
            "summary": "metadata only",
            "source_url": "https://openalex.org/W1",
            "source_kind": "论文元数据",
            "source_authority": "OpenAlex",
            "execution_status": "metadata_only",
            "selection_reason": "not mapped",
            "algorithm_tags": ["online_metadata"],
        }
        with mock.patch(
            "ecologyrsi_dsh.knowledge.retrieval._openalex_cards",
            return_value=[KnowledgeCard.from_dict(online_card)],
        ):
            snapshot = retrieve_generation_knowledge(director.state("run:metadata"))

        result = KnowledgeSnapshot.from_dict(snapshot.to_dict())
        card = next(item for item in result.cards if item.knowledge_id == "openalex:W1")
        self.assertFalse(card.executable)
        self.assertEqual(card.execution_status, "metadata_only")
        self.assertFalse(result.proposal_context()["safety_boundary"]["external_code_execution"])
        ledger.close()


if __name__ == "__main__":
    unittest.main()
