"""Bounded access to the official DeepSeek Search API.

Search is a retrieval request, not a model completion. Credentials are resolved
locally and are only sent to the fixed official endpoint. Remote text remains
untrusted evidence; this module never executes or fetches returned URLs.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


_ENDPOINT = "https://api.deepseek.com/v1/tools/search"
_MAX_QUERY_CHARS = 128
_MAX_QUERY_FRAGMENTS = 3
_MAX_HTTP_REQUESTS = 12
_MAX_WORKERS = 4
_MAX_SOURCES = 32
_MAX_RESPONSE_BYTES = 2_000_000
_MAX_RETRIES = 2
_TRACKING_PARAMETERS = frozenset(
    {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src"}
)
_SEARCH_SYNTAX = re.compile(
    r'["“”]|(?:^|\s)(?:-?(?:site|lang|after|before):|OR(?:\s|$)|-\S)',
    re.IGNORECASE,
)


class DeepSeekSearchError(RuntimeError):
    """A bounded local error; remote response bodies and credentials are omitted."""

    def __init__(self, code: str) -> None:
        messages = {
            "not_configured": "DeepSeek Search has no configured official API credential.",
            "invalid_queries": "DeepSeek Search requires one to twelve nonempty queries.",
            "query_requires_shortening": (
                "DeepSeek Search queries with operators or quotes must fit 128 Unicode "
                "characters; shorten the query. Natural-language queries allow at most "
                "three fragments of 128 characters."
            ),
            "invalid_limit": "DeepSeek Search result limit must be a positive integer.",
            "invalid_timeout": "DeepSeek Search timeout must be between 1 and 120 seconds.",
            "invalid_response": "DeepSeek Search returned an invalid or oversized response.",
            "redirect_rejected": "DeepSeek Search rejected an HTTP redirect.",
            "unauthorized": "DeepSeek Search rejected the configured API credential.",
            "insufficient_balance": "DeepSeek Search account balance is insufficient.",
            "request_rejected": "DeepSeek Search rejected the request.",
            "temporarily_unavailable": "DeepSeek Search is temporarily unavailable.",
        }
        self.code = code if code in messages else "temporarily_unavailable"
        super().__init__(messages[self.code])


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise DeepSeekSearchError("redirect_rejected")


def _official_gateway(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme == "https"
            and parsed.hostname == "api.deepseek.com"
            and parsed.port in (None, 443)
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


def _api_key() -> str | None:
    for name in ("ECOLOGYRSI_DEEPSEEK_SEARCH_API_KEY", "DEEPSEEK_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    private_key = _private_file_api_key()
    if private_key:
        return private_key
    # Reuse the existing private-file reader. Never infer an official credential
    # from a provider/model name or reuse a third-party gateway's token.
    from ..integrations.dsh_directory import discover_model_entries

    try:
        entries = discover_model_entries()
    except Exception:  # noqa: BLE001 - optional local credential discovery
        return None
    for entry in entries:
        if _official_gateway(entry.get("gateway_url")):
            value = entry.get("token")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _private_file_api_key() -> str | None:
    """Read only the two explicitly named DeepSeek keys from a private DSH file."""

    from ..integrations.dsh_directory import _flat_credentials, _secure_text

    path = Path(
        os.environ.get("ECOLOGYRSI_DSH_CREDENTIALS_FILE")
        or "~/.dsh/.credentials.yaml"
    ).expanduser()
    text = _secure_text(path)
    if text is None:
        return None
    credentials = _flat_credentials(text)
    for name in ("ECOLOGYRSI_DEEPSEEK_SEARCH_API_KEY", "DEEPSEEK_API_KEY"):
        value = credentials.get(name, "").strip()
        if value:
            return value
    return None


def is_deepseek_search_configured() -> bool:
    """Check availability without returning or logging the resolved credential."""

    return _api_key() is not None


def _timeout_seconds() -> float:
    try:
        value = float(os.environ.get("ECOLOGYRSI_DEEPSEEK_SEARCH_TIMEOUT", "20"))
    except ValueError:
        raise DeepSeekSearchError("invalid_timeout") from None
    if not math.isfinite(value) or not 1 <= value <= 120:
        raise DeepSeekSearchError("invalid_timeout")
    return value


def _query_fragments(query: str) -> tuple[str, ...]:
    query = " ".join(query.split())
    if not query:
        raise DeepSeekSearchError("invalid_queries")
    if len(query) <= _MAX_QUERY_CHARS:
        return (query,)
    # Splitting operators, quoted phrases, or Boolean expressions can silently
    # change their scope. Return actionable feedback instead of altering them.
    if _SEARCH_SYNTAX.search(query):
        raise DeepSeekSearchError("query_requires_shortening")
    fragments: list[str] = []
    remaining = query
    while remaining:
        if len(fragments) >= _MAX_QUERY_FRAGMENTS:
            raise DeepSeekSearchError("query_requires_shortening")
        if len(remaining) <= _MAX_QUERY_CHARS:
            fragments.append(remaining)
            break
        boundary = remaining.rfind(" ", 0, _MAX_QUERY_CHARS + 1)
        if boundary < 1:
            # CJK queries commonly have no spaces. A long Latin token/URL is
            # not safely separable and must be rewritten by the caller.
            head = remaining[:_MAX_QUERY_CHARS]
            if not re.search(r"[\u3400-\u9fff]", head):
                raise DeepSeekSearchError("query_requires_shortening")
            boundary = _MAX_QUERY_CHARS
        fragments.append(remaining[:boundary])
        remaining = remaining[boundary:].strip()
    return tuple(fragments)


def _query_plan(queries: tuple[str, ...]) -> tuple[list[str], list[str]]:
    if (
        not isinstance(queries, (tuple, list))
        or not 1 <= len(queries) <= _MAX_HTTP_REQUESTS
        or any(not isinstance(query, str) or not query.strip() for query in queries)
    ):
        raise DeepSeekSearchError("invalid_queries")
    groups: list[tuple[str, ...]] = []
    notes: list[str] = []
    for query in queries:
        try:
            groups.append(_query_fragments(query))
        except DeepSeekSearchError:
            notes.append(
                "A query was skipped: DeepSeek permits 128 Unicode characters for "
                "search syntax or quotes; shorten it, or use at most three "
                "128-character natural-language fragments."
            )
    if not groups:
        raise DeepSeekSearchError("query_requires_shortening")
    plan: list[str] = []
    seen: set[str] = set()
    # Keep each original query represented before adding its longer fragments.
    for rank in range(max(map(len, groups))):
        for group in groups:
            if rank >= len(group) or group[rank].casefold() in seen:
                continue
            seen.add(group[rank].casefold())
            plan.append(group[rank])
    if len(plan) > _MAX_HTTP_REQUESTS:
        notes.append("The twelve-request search budget omitted additional query fragments.")
        plan = plan[:_MAX_HTTP_REQUESTS]
    return plan, notes


def _canonical_url(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 2048:
        return None
    try:
        parsed = urlsplit(value.strip())
        if (
            parsed.scheme.casefold() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or any(ord(char) < 33 for char in value.strip())
        ):
            return None
        port = parsed.port
        hostname = parsed.hostname.casefold()
        host = f"[{hostname}]" if ":" in hostname else hostname
        netloc = host if port in (None, 443) else f"{host}:{port}"
    except ValueError:
        return None
    query = urlencode(
        sorted(
            (name, item)
            for name, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not name.casefold().startswith("utm_")
            and name.casefold() not in _TRACKING_PARAMETERS
        )
    )
    return urlunsplit(("https", netloc, parsed.path or "/", query, ""))


def _text(value: Any, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.split())[:maximum] or None


def _normalize_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("results"), list):
        raise DeepSeekSearchError("invalid_response")
    sources: list[dict[str, str]] = []
    raw_sources = payload["results"]
    seen: set[str] = set()
    for item in raw_sources[:50]:
        if not isinstance(item, Mapping):
            continue
        url = _canonical_url(item.get("url"))
        if url is None or url in seen:
            continue
        title = _text(item.get("title"), 300)
        snippet = _text(item.get("summary"), 1200) or _text(item.get("snippet"), 1200)
        if title is None and snippet is None:
            continue
        seen.add(url)
        source = {"url": url}
        if title:
            source["title"] = title
        if snippet:
            source["snippet"] = snippet
        published = _text(item.get("date"), 100)
        if published:
            source["publishedAt"] = published
        sources.append(source)
    if raw_sources and not sources:
        raise DeepSeekSearchError("invalid_response")
    return {"sources": sources, "truncated": len(raw_sources) > 50}


class _RetryBudget:
    """Reserve one initial request per query before sharing remaining retries."""

    def __init__(self, initial_queries: int) -> None:
        self.remaining = _MAX_HTTP_REQUESTS - initial_queries
        self.lock = Lock()

    def take(self) -> bool:
        with self.lock:
            if self.remaining <= 0:
                return False
            self.remaining -= 1
            return True


def _safe_http_error(status: int) -> DeepSeekSearchError:
    if status == 401:
        return DeepSeekSearchError("unauthorized")
    if status == 402:
        return DeepSeekSearchError("insufficient_balance")
    if 300 <= status < 400:
        return DeepSeekSearchError("redirect_rejected")
    if status in {408, 429} or status >= 500:
        return DeepSeekSearchError("temporarily_unavailable")
    return DeepSeekSearchError("request_rejected")


def _search_one(
    query: str, *, key: str, count: int, timeout: float, retry_budget: _RetryBudget
) -> dict[str, Any]:
    request = Request(
        _ENDPOINT,
        data=json.dumps(
            {"query": query, "count": count, "need_content": False},
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    opener = build_opener(_NoRedirectHandler())
    for attempt in range(_MAX_RETRIES + 1):
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise DeepSeekSearchError("invalid_response")
            return _normalize_response(json.loads(raw.decode("utf-8")))
        except HTTPError as exc:
            status = exc.code
            exc.close()
            error = _safe_http_error(status)
            retryable = status in {408, 429} or 500 <= status <= 599
        except (TimeoutError, URLError, OSError):
            error = DeepSeekSearchError("temporarily_unavailable")
            retryable = True
        except (UnicodeError, ValueError):
            raise DeepSeekSearchError("invalid_response") from None
        except DeepSeekSearchError:
            raise
        except Exception:  # noqa: BLE001 - never surface remote transport details
            raise DeepSeekSearchError("temporarily_unavailable") from None
        if not retryable or attempt >= _MAX_RETRIES or not retry_budget.take():
            raise error from None
        time.sleep(min(0.25 * 2**attempt, 1.0))
    raise DeepSeekSearchError("temporarily_unavailable")


def search_deepseek_metadata(
    queries: tuple[str, ...], *, limit: int = 16
) -> dict[str, Any]:
    """Return bounded web evidence with summary snippets and deterministic order.

    At most twelve HTTP requests (including retries) run with four workers.
    Partial failures retain successful results; complete failure raises only a
    safe local error so the caller can choose another retrieval source.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise DeepSeekSearchError("invalid_limit")
    limit = min(limit, _MAX_SOURCES)
    plan, notes = _query_plan(queries)
    key = _api_key()
    if key is None:
        raise DeepSeekSearchError("not_configured")
    timeout = _timeout_seconds()
    budget = _RetryBudget(len(plan))
    results: list[dict[str, Any]] = []
    failures: list[DeepSeekSearchError] = []
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(plan))) as pool:
        futures = [
            pool.submit(
                _search_one,
                query,
                key=key,
                count=limit,
                timeout=timeout,
                retry_budget=budget,
            )
            for query in plan
        ]
        for future in futures:
            try:
                results.append(future.result())
            except DeepSeekSearchError as exc:
                failures.append(exc)
    if not results:
        raise failures[0] from None
    if failures:
        notes.append(f"{len(failures)} of {len(plan)} search requests failed; successful results are retained.")
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    truncated = bool(notes) or any(result["truncated"] for result in results)
    for rank in range(max((len(result["sources"]) for result in results), default=0)):
        for result in results:
            if rank >= len(result["sources"]):
                continue
            source = result["sources"][rank]
            if source["url"] in seen:
                continue
            seen.add(source["url"])
            if len(sources) >= limit:
                truncated = True
                continue
            sources.append(source)
    return {
        "sources": sources,
        "content": (
            f"DeepSeek Search returned {len(sources)} sources. "
            "External search summaries are unverified evidence. "
            + " ".join(dict.fromkeys(notes))
        ).strip(),
        "truncated": truncated,
    }


__all__ = [
    "DeepSeekSearchError",
    "is_deepseek_search_configured",
    "search_deepseek_metadata",
]
