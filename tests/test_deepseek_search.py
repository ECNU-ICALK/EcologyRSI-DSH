from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request

from ecologyrsi_dsh.knowledge import deepseek_search as search


def _row(url: str, **extra: object) -> dict:
    return {
        "title": "Soil moisture forecasting",
        "url": url,
        "snippet": "Short excerpt",
        "summary": "Long research summary",
        "content": "",
        "date": "2026-10-20T00:00:00Z",
        "authority": 3,
        "rerank_score": 2.4,
        **extra,
    }


class _Response:
    def __init__(self, payload: object) -> None:
        self.data = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, count):
        return self.data[:count]


class DeepSeekSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = mock.patch.dict(
            os.environ, {"ECOLOGYRSI_DEEPSEEK_SEARCH_API_KEY": "test-official-key"}, clear=True
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.original_private_file_reader = search._private_file_api_key
        private_file = mock.patch.object(search, "_private_file_api_key", return_value=None)
        private_file.start()
        self.addCleanup(private_file.stop)

    @contextmanager
    def transport(self, callback):
        with mock.patch.object(search, "build_opener") as build:
            build.return_value.open.side_effect = callback
            yield build

    def test_official_wire_contract_and_long_summary_mapping(self) -> None:
        calls = []

        def respond(request, *, timeout):
            calls.append((request, timeout))
            return _Response({"results": [_row("https://EXAMPLE.org/paper?utm_source=x#part")]})

        with self.transport(respond) as build:
            result = search.search_deepseek_metadata(("soil moisture",), limit=20)
        request, timeout = calls[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/v1/tools/search")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-official-key")
        self.assertEqual(
            json.loads(request.data),
            {"query": "soil moisture", "count": 20, "need_content": False},
        )
        self.assertEqual(timeout, 20)
        self.assertIsInstance(build.call_args.args[0], search._NoRedirectHandler)
        self.assertEqual(
            result["sources"],
            [{
                "url": "https://example.org/paper",
                "title": "Soil moisture forecasting",
                "snippet": "Long research summary",
                "publishedAt": "2026-10-20T00:00:00Z",
            }],
        )
        self.assertNotIn("test-official-key", json.dumps(result))
        self.assertFalse(result["truncated"])

    def test_configuration_prefers_dedicated_key_and_official_directory_only(self) -> None:
        directory = "ecologyrsi_dsh.integrations.dsh_directory.discover_model_entries"
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "general"}):
            self.assertEqual(search._api_key(), "test-official-key")
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(directory) as discover:
            discover.return_value = [
                {"gateway_url": "https://proxy.example/v1", "token": "third-party"},
                {"gateway_url": "https://api.deepseek.com.evil.test/v1", "token": "wrong"},
                {"gateway_url": "http://api.deepseek.com/v1", "token": "insecure"},
                {"gateway_url": "https://api.deepseek.com:8443/v1", "token": "wrong-port"},
            ]
            self.assertFalse(search.is_deepseek_search_configured())
            discover.return_value.append(
                {"gateway_url": "https://api.deepseek.com/v1", "token": "official"}
            )
            self.assertTrue(search.is_deepseek_search_configured())
            self.assertEqual(search._api_key(), "official")

    def test_general_key_works_without_directory_discovery(self) -> None:
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "general"}, clear=True):
            self.assertEqual(search._api_key(), "general")

    def test_private_dsh_file_uses_only_named_official_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            credentials = Path(directory) / ".credentials.yaml"
            with mock.patch.dict(os.environ, {
                "ECOLOGYRSI_DSH_CREDENTIALS_FILE": str(credentials)
            }, clear=True):
                credentials.write_text("THIRDPARTY_DEEPSEEK_KEY: third-party\n")
                credentials.chmod(0o600)
                self.assertIsNone(self.original_private_file_reader())
                credentials.write_text(
                    "THIRDPARTY_DEEPSEEK_KEY: third-party\n"
                    "DEEPSEEK_API_KEY: official-general\n"
                )
                self.assertEqual(self.original_private_file_reader(), "official-general")
                credentials.write_text(
                    "DEEPSEEK_API_KEY: official-general\n"
                    "ECOLOGYRSI_DEEPSEEK_SEARCH_API_KEY: official-search\n"
                )
                self.assertEqual(self.original_private_file_reader(), "official-search")
                credentials.chmod(0o644)
                self.assertIsNone(self.original_private_file_reader())

    def test_named_private_key_works_without_an_official_model_entry(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            search, "_private_file_api_key", return_value="official-file"
        ), mock.patch(
            "ecologyrsi_dsh.integrations.dsh_directory.discover_model_entries"
        ) as discover:
            self.assertTrue(search.is_deepseek_search_configured())
            self.assertEqual(search._api_key(), "official-file")
            discover.assert_not_called()

    def test_missing_key_fails_before_network(self) -> None:
        with mock.patch.object(search, "_api_key", return_value=None), mock.patch.object(
            search, "build_opener"
        ) as build:
            with self.assertRaisesRegex(search.DeepSeekSearchError, "no configured"):
                search.search_deepseek_metadata(("soil",))
            build.assert_not_called()

    def test_long_plain_queries_preserve_all_words_and_cjk(self) -> None:
        query = " ".join(f"word{number}" for number in range(45))
        fragments = search._query_fragments(query)
        self.assertLessEqual(len(fragments), 3)
        self.assertTrue(all(len(fragment) <= 128 for fragment in fragments))
        self.assertEqual(" ".join(fragments), query)
        chinese = "温室作物土壤水分预测模型" * 20
        self.assertEqual("".join(search._query_fragments(chinese)), chinese)
        syntax = '"soil moisture" site:example.org OR site:example.edu'
        self.assertEqual(search._query_fragments(syntax), (syntax,))

    def test_long_syntax_is_not_silently_changed_and_partial_feedback_is_actionable(self) -> None:
        long_query = "soil " * 30 + "site:example.org"
        with self.transport(lambda *_args, **_kwargs: _Response({"results": []})) as build:
            with self.assertRaisesRegex(search.DeepSeekSearchError, "128"):
                search.search_deepseek_metadata((long_query,))
            build.assert_not_called()
            result = search.search_deepseek_metadata((long_query, "valid"))
        self.assertIn("128 Unicode", result["content"])
        self.assertIn("shorten", result["content"])
        self.assertTrue(result["truncated"])
        self.assertNotIn(long_query, result["content"])

    def test_round_robin_merge_keeps_later_query_results(self) -> None:
        def respond(request, **_kwargs):
            query = json.loads(request.data)["query"]
            return _Response({"results": [
                _row(f"https://example.org/{query}/{rank}") for rank in range(4)
            ]})

        with self.transport(respond):
            result = search.search_deepseek_metadata(("first", "second"), limit=4)
        self.assertEqual(
            [row["url"] for row in result["sources"]],
            ["https://example.org/first/0", "https://example.org/second/0",
             "https://example.org/first/1", "https://example.org/second/1"],
        )
        self.assertTrue(result["truncated"])

    def test_https_dedup_and_summary_fallback(self) -> None:
        rows = [
            _row("http://example.org/insecure"),
            _row("https://user:pass@example.org/private"),
            _row("https://example.org/a?utm_source=one", summary="", date=None),
            _row("https://example.org/a?utm_source=two"),
            _row("https://example.org/has space"),
            _row("https://[::1]:443/"),
        ]
        with self.transport(lambda *_args, **_kwargs: _Response({"results": rows})):
            result = search.search_deepseek_metadata(("soil",))
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["sources"][0]["snippet"], "Short excerpt")
        self.assertNotIn("publishedAt", result["sources"][0])
        self.assertEqual(result["sources"][1]["url"], "https://[::1]/")

    def test_partial_failure_retains_success_without_remote_error_details(self) -> None:
        def respond(request, **_kwargs):
            if json.loads(request.data)["query"] == "fail":
                raise HTTPError(request.full_url, 401, "secret-provider-error", {}, None)
            return _Response({"results": [_row("https://example.org/success")]})

        with self.transport(respond):
            result = search.search_deepseek_metadata(("fail", "success"))
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("1 of 2", result["content"])
        self.assertNotIn("secret-provider-error", str(result))
        self.assertNotIn("test-official-key", str(result))

    def test_nonretryable_status_and_error_body_never_escape(self) -> None:
        for status, code in ((401, "unauthorized"), (402, "insufficient_balance"), (400, "request_rejected")):
            with self.subTest(status=status):
                with self.transport(lambda request, **_kwargs: (_ for _ in ()).throw(
                    HTTPError(request.full_url, status, "secret-response", {}, None)
                )) as build:
                    with self.assertRaises(search.DeepSeekSearchError) as caught:
                        search.search_deepseek_metadata(("soil",))
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn("secret-response", str(caught.exception))
                self.assertEqual(build.return_value.open.call_count, 1)

    def test_retryable_failure_recovers_with_bounded_retries(self) -> None:
        responses = [
            HTTPError(search._ENDPOINT, 429, "private error", {}, None),
            _Response({"results": [_row("https://example.org/retry")]}),
        ]
        with mock.patch.object(search, "build_opener") as build, mock.patch.object(
            search.time, "sleep"
        ) as sleep:
            build.return_value.open.side_effect = responses
            result = search.search_deepseek_metadata(("soil",))
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(build.return_value.open.call_count, 2)
        sleep.assert_called_once()

    def test_total_http_attempts_include_retries_and_never_exceed_twelve(self) -> None:
        with self.transport(lambda request, **_kwargs: (_ for _ in ()).throw(
            HTTPError(request.full_url, 503, "private error", {}, None)
        )) as build, mock.patch.object(search.time, "sleep"):
            with self.assertRaises(search.DeepSeekSearchError):
                search.search_deepseek_metadata(tuple(f"query {number}" for number in range(8)))
        self.assertEqual(build.return_value.open.call_count, 12)

    def test_at_most_four_concurrent_requests(self) -> None:
        lock = threading.Lock()
        barrier = threading.Barrier(4, timeout=5)
        active = 0
        peak = 0

        def respond(_request, **_kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait()
            with lock:
                active -= 1
            return _Response({"results": []})

        with self.transport(respond):
            search.search_deepseek_metadata(tuple(f"query {number}" for number in range(8)))
        self.assertEqual(peak, 4)

    def test_source_limit_is_capped_at_thirty_two(self) -> None:
        counts = []

        def respond(request, **_kwargs):
            counts.append(json.loads(request.data)["count"])
            return _Response({"results": [_row(f"https://example.org/{i}") for i in range(50)]})

        with self.transport(respond):
            result = search.search_deepseek_metadata(("soil",), limit=100)
        self.assertEqual(counts, [32])
        self.assertEqual(len(result["sources"]), 32)
        self.assertTrue(result["truncated"])

    def test_redirect_is_refused_instead_of_forwarding_authorization(self) -> None:
        request = Request(search._ENDPOINT, headers={"Authorization": "Bearer private-key"})
        with self.assertRaises(search.DeepSeekSearchError) as caught:
            search._NoRedirectHandler().redirect_request(
                request, None, 302, "private-key", {}, "https://attacker.example"
            )
        self.assertEqual(caught.exception.code, "redirect_rejected")
        self.assertNotIn("private-key", str(caught.exception))
        self.assertNotIn("attacker", str(caught.exception))

    def test_invalid_payload_and_configuration_fail_safely(self) -> None:
        with self.transport(lambda *_args, **_kwargs: _Response({"private": "secret"})):
            with self.assertRaises(search.DeepSeekSearchError) as caught:
                search.search_deepseek_metadata(("soil",))
        self.assertEqual(caught.exception.code, "invalid_response")
        for value in ("nan", "0", "121", "secret"):
            with mock.patch.dict(os.environ, {"ECOLOGYRSI_DEEPSEEK_SEARCH_TIMEOUT": value}):
                with self.assertRaises(search.DeepSeekSearchError) as caught:
                    search.search_deepseek_metadata(("soil",))
                self.assertEqual(caught.exception.code, "invalid_timeout")
        for value in (0, -1, True, 2.5):
            with self.assertRaises(search.DeepSeekSearchError):
                search.search_deepseek_metadata(("soil",), limit=value)


if __name__ == "__main__":
    unittest.main()
