"""Offline method references frozen before independent data is opened.

Only the packaged, Host-curated catalog is an input to the builder. Online
snippets, run evidence, model output, and dataset rows are never consulted.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from importlib.resources import files
import json
import re
from typing import Any
from urllib.parse import urlsplit

from ..core.models import digest
from .retrieval import normalize_dynamic_search_result
from .retrieval_policy import resolve_retrieval_policy, validate_retrieval_policy


FORMAL_RETRIEVAL_SNAPSHOT_EVENT = "FormalRetrievalSnapshotFrozen"
FORMAL_RETRIEVAL_SNAPSHOT_SCHEMA = "ecologyrsi-dsh.formal-retrieval-snapshot/1"
_METHOD_IDS = frozenset({
    "sklearn-time-series-split", "sklearn-ridge", "host-bounded-parameter-sweep",
    "host-adaptive-local-search", "optuna-tpe", "pcse-wofost", "aquacrop-ospy",
    "greenlight-model",
})
_DOCUMENT_FIELDS = frozenset({
    "document_id", "title", "summary", "source_url", "source_authority", "tags",
})
_SNAPSHOT_FIELDS = frozenset({
    "schema_version", "source", "scope", "catalog_digest", "documents",
    "retrieval_policy", "snapshot_digest",
})


def build_formal_retrieval_snapshot(metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Copy complete curated method summaries and a frozen local query budget."""
    catalog = json.loads(files("ecologyrsi_dsh.knowledge")
                         .joinpath("ecology_algorithms.json").read_text(encoding="utf-8"))
    if not isinstance(catalog, list):
        raise ValueError("formal method catalog is invalid")
    selected = [item for item in catalog
                if isinstance(item, dict) and item.get("knowledge_id") in _METHOD_IDS]
    documents = sorted(({
        "document_id": item["knowledge_id"], "title": item["title_zh"],
        "summary": item["summary_zh"], "source_url": item["source_url"],
        "source_authority": item["source_authority"],
        "tags": list(item["algorithm_tags"]),
    } for item in selected), key=lambda item: item["document_id"])
    policy = replace(resolve_retrieval_policy(metadata or {}, environ={}), online_enabled=False)
    body = {
        "schema_version": FORMAL_RETRIEVAL_SNAPSHOT_SCHEMA,
        "source": "packaged_ecology_algorithms_catalog",
        "scope": "methods_only_no_dataset_results",
        "catalog_digest": digest(selected), "documents": documents,
        "retrieval_policy": policy.to_dict(),
    }
    return validate_formal_retrieval_snapshot({**body, "snapshot_digest": digest(body)})


def validate_formal_retrieval_snapshot(value: Any) -> dict[str, Any]:
    """Validate persisted content without rebuilding it from a newer catalog."""
    if not isinstance(value, Mapping) or set(value) != _SNAPSHOT_FIELDS:
        raise ValueError("formal retrieval snapshot has an invalid shape")
    value = deepcopy(dict(value))
    if (value["schema_version"] != FORMAL_RETRIEVAL_SNAPSHOT_SCHEMA
            or value["source"] != "packaged_ecology_algorithms_catalog"
            or value["scope"] != "methods_only_no_dataset_results"):
        raise ValueError("formal retrieval snapshot provenance is invalid")
    for key in ("catalog_digest", "snapshot_digest"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise ValueError("formal retrieval snapshot digest is invalid")
    if digest({key: item for key, item in value.items() if key != "snapshot_digest"}) != value["snapshot_digest"]:
        raise ValueError("formal retrieval snapshot digest mismatch")
    policy = value["retrieval_policy"]
    if not isinstance(policy, Mapping) or policy.get("online_enabled") is not False:
        raise ValueError("formal retrieval snapshot must forbid online access")
    budgets = validate_retrieval_policy({key: item for key, item in policy.items()
                                       if key != "online_enabled"})
    if set(policy) != {"online_enabled", *budgets} or len(budgets) != 4:
        raise ValueError("formal retrieval snapshot requires a complete frozen policy")
    documents = value["documents"]
    if not isinstance(documents, list) or len(documents) != len(_METHOD_IDS):
        raise ValueError("formal retrieval snapshot must contain the trusted method directory")
    identities = []
    for document in documents:
        if not isinstance(document, Mapping) or set(document) != _DOCUMENT_FIELDS:
            raise ValueError("formal method document has an invalid shape")
        for field in _DOCUMENT_FIELDS - {"tags"}:
            if not isinstance(document[field], str) or not 1 <= len(document[field]) <= 4000:
                raise ValueError("formal method document field is invalid")
        parsed = urlsplit(document["source_url"])
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError("formal method reference URL is invalid")
        tags = document["tags"]
        if (not isinstance(tags, list) or not 1 <= len(tags) <= 32
                or any(not isinstance(tag, str) or not re.fullmatch(r"[a-z0-9_]{1,80}", tag)
                       for tag in tags)):
            raise ValueError("formal method tags are invalid")
        identities.append(document["document_id"])
    if len(set(identities)) != len(identities) or set(identities) != _METHOD_IDS:
        raise ValueError("formal method directory identity is invalid")
    if identities != sorted(identities):
        raise ValueError("formal method directory must be ordered")
    return value


def _tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
    for part in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.update(part[index:index + 2] for index in range(max(1, len(part) - 1)))
    return tokens


def search_formal_retrieval_snapshot(
    snapshot: Mapping[str, Any], queries: Sequence[str], *, limit: int,
) -> dict[str, Any]:
    """Deterministic local lexical lookup; links are citations, never fetched."""
    snapshot = validate_formal_retrieval_snapshot(snapshot)
    if isinstance(queries, (str, bytes)) or not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError("formal retrieval queries are invalid")
    if type(limit) is not int or not 1 <= limit <= snapshot["retrieval_policy"]["max_results"]:
        raise ValueError("formal retrieval result limit is invalid")
    query_terms = set().union(*(_tokens(query) for query in queries))
    ranked = []
    for document in snapshot["documents"]:
        terms = _tokens(" ".join(
            [document["document_id"], document["title"], document["summary"],
             document["source_url"], *document["tags"]]))
        score = len(query_terms & terms)
        if score:
            ranked.append((score, document["document_id"], document))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    result = {
        "sources": [{"url": document["source_url"], "title": document["title"],
                     "snippet": document["summary"]}
                    for _, _, document in ranked[:limit]],
        "content": "Frozen Host-curated method summaries only; no online requests or evaluation results.",
        "truncated": len(ranked) > limit,
    }
    return normalize_dynamic_search_result(result, max_sources=limit)
