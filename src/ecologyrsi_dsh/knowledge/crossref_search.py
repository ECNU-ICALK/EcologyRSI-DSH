"""Bounded, anonymous Crossref metadata fallback; never fetches paper full text."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ..core.models import digest
from .models import KnowledgeCard


class CrossrefSearchError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _text(value, limit):
    if not isinstance(value, str):
        return None
    parser = _Text()
    parser.feed(value[:16_000])
    return " ".join(" ".join(parser.parts).split())[:limit] or None


def _cards(query: str, limit: int) -> list[KnowledgeCard]:
    request = Request(
        "https://api.crossref.org/works?" + urlencode({
            "query.bibliographic": query, "rows": min(limit, 16),
            "select": "DOI,title,abstract,published",
        }),
        headers={"Accept": "application/json",
                 "User-Agent": "EcologyRSI-DSH (research metadata fallback)"},
    )
    try:
        with urlopen(request, timeout=8) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise CrossrefSearchError("response_too_large")
        payload = json.loads(raw.decode("utf-8"))
    except HTTPError as exc:
        code = exc.code
        exc.close()
        raise CrossrefSearchError(f"http_{code}") from None
    except (TimeoutError, URLError, OSError):
        raise CrossrefSearchError("transport_error") from None
    except (ValueError, UnicodeError):
        raise CrossrefSearchError("invalid_response") from None
    message = payload.get("message") if isinstance(payload, dict) else None
    items = message.get("items") if isinstance(message, dict) else None
    if not isinstance(payload, dict) or payload.get("status") != "ok" or not isinstance(items, list):
        raise CrossrefSearchError("invalid_response")
    cards = []
    for item in items[:32]:
        if not isinstance(item, dict):
            continue
        doi = item.get("DOI")
        titles = item.get("title")
        title = _text(titles[0], 300) if isinstance(titles, list) and titles else None
        if not isinstance(doi, str) or not re.fullmatch(r"10\.\d{4,9}/\S{1,400}", doi) or not title:
            continue
        doi = doi.casefold()
        abstract = _text(item.get("abstract"), 1_700)
        summary = ("Crossref 论文元数据：仅题名和 DOI，来源未提供摘要；不是全文证据。"
                   if not abstract else "Crossref 论文元数据及出版者提供的摘要；尚未经全文方法核验。")
        cards.append(KnowledgeCard(
            knowledge_id="crossref:" + doi, title=title, summary=summary,
            source_url="https://doi.org/" + quote(doi, safe="/"),
            source_kind="论文元数据", source_authority="Crossref",
            execution_status="metadata_only", algorithm_tags=("online_metadata",),
            selection_reason="仅补充研究线索；不能授权执行论文代码或证明方法有效。",
            abstract_summary=("Crossref 摘要（metadata_only；未经全文核验）：" + abstract
                              if abstract else None),
            content_digest=digest(abstract) if abstract else None,
        ))
    return cards


def crossref_cards(queries: tuple[str, ...] | list[str], *, limit: int = 16) -> list[KnowledgeCard]:
    """At most three concurrent requests, no retries, deduplicated across queries."""
    if (not isinstance(queries, (tuple, list)) or not 1 <= len(queries) <= 12
            or any(not isinstance(q, str) or not q.strip() or len(q) > 600
                   or any(ord(c) < 32 for c in q) for q in queries)
            or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 32):
        raise CrossrefSearchError("invalid_arguments")
    plan = list(dict.fromkeys(" ".join(q.split()) for q in queries))[:3]
    batches, errors = [], []
    with ThreadPoolExecutor(max_workers=len(plan)) as executor:
        futures = [executor.submit(_cards, query, limit) for query in plan]
        for future in futures:
            try:
                batches.append(future.result())
            except CrossrefSearchError as exc:
                errors.append(exc)
                batches.append([])
    if len(errors) == len(plan):
        raise errors[0]
    result, seen = [], set()
    for rank in range(max(map(len, batches), default=0)):
        for batch in batches:
            if rank < len(batch) and batch[rank].knowledge_id not in seen:
                seen.add(batch[rank].knowledge_id)
                result.append(batch[rank])
                if len(result) >= limit:
                    return result
    return result


def search_crossref_metadata(queries: tuple[str, ...], *, limit: int = 16) -> dict:
    from .retrieval import normalize_dynamic_search_result

    cards = crossref_cards(queries, limit=limit)
    return normalize_dynamic_search_result({
        "content": "Crossref metadata fallback; titles/DOIs and supplied abstracts only; no full text verified.",
        "sources": [{"url": c.source_url, "title": c.title,
                     "snippet": c.abstract_summary or c.summary} for c in cards],
        "truncated": len(cards) >= limit,
    }, max_sources=limit)
