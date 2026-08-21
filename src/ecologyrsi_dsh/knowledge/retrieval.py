"""Curated catalog and optional OpenAlex metadata retrieval."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Mapping
from datetime import timezone
from email.utils import parsedate_to_datetime
from importlib.resources import files
from itertools import islice
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..core.models import digest
from .mapping import map_catalog_entry
from .models import KnowledgeAssessment, KnowledgeCard, KnowledgeSnapshot

_TARGET_TERMS = {
    "air_temperature": "greenhouse air temperature",
    "relative_humidity": "greenhouse relative humidity",
    "co2_concentration": "greenhouse carbon dioxide concentration",
    "soil_moisture": "soil moisture",
    "soil_water": "soil water",
}
_OPENALEX_ABSTRACT_MAX_UNIQUE_TERMS = 1_024
_OPENALEX_ABSTRACT_MAX_SCANNED_POSITIONS = 4_096
_OPENALEX_ABSTRACT_MAX_POSITIONS = 512
_OPENALEX_ABSTRACT_MAX_TERM_CHARS = 200
_OPENALEX_ABSTRACT_SUMMARY_MAX_CHARS = 1_800
_OPENALEX_DEFAULT_TIMEOUT_SECONDS = 20.0
_OPENALEX_TIMEOUT_ENV = "ECOLOGYRSI_OPENALEX_TIMEOUT"
_OPENALEX_MAX_RETRIES = 3
_OPENALEX_RETRY_BASE_SECONDS = 0.25
_OPENALEX_RETRY_MAX_SECONDS = 1.0
_OPENALEX_RETRY_AFTER_MAX_SECONDS = 5.0
_OPENALEX_MAX_QUERIES = 6
_OPENALEX_QUERY_MAX_CHARS = 180
_OPENALEX_MAX_RESPONSE_BYTES = 1_000_000
_OPENALEX_RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429})


def _safe_failure_token(value: Any) -> str | None:
    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value or "").strip().casefold(),
    )
    result = " ".join(normalized.split())[:80]
    return result or None


def _safe_horizon(value: Any) -> int | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    result = round(float(value))
    return result if 1 <= result <= 8760 else None


def _reconstruct_openalex_abstract(value: Any) -> str | None:
    """Rebuild a bounded abstract from OpenAlex's word-to-position mapping."""

    if not isinstance(value, Mapping):
        return None
    positioned_terms: list[tuple[int, str]] = []
    scanned_positions = 0
    for raw_term, raw_positions in islice(
        value.items(), _OPENALEX_ABSTRACT_MAX_UNIQUE_TERMS
    ):
        if scanned_positions >= _OPENALEX_ABSTRACT_MAX_SCANNED_POSITIONS:
            break
        if not isinstance(raw_term, str) or not isinstance(raw_positions, list):
            continue
        term = " ".join(raw_term.split())
        if not term or len(term) > _OPENALEX_ABSTRACT_MAX_TERM_CHARS:
            continue
        for raw_position in raw_positions:
            if scanned_positions >= _OPENALEX_ABSTRACT_MAX_SCANNED_POSITIONS:
                break
            scanned_positions += 1
            if (
                isinstance(raw_position, bool)
                or not isinstance(raw_position, int)
                or raw_position < 0
                or raw_position > 100_000
            ):
                continue
            positioned_terms.append((raw_position, term))
    if not positioned_terms:
        return None
    positioned_terms.sort(key=lambda item: (item[0], item[1]))
    abstract_terms: list[str] = []
    seen_positions: set[int] = set()
    for position, term in positioned_terms:
        if position in seen_positions:
            continue
        seen_positions.add(position)
        abstract_terms.append(term)
        if len(abstract_terms) >= _OPENALEX_ABSTRACT_MAX_POSITIONS:
            break
    return " ".join(abstract_terms) or None


def _openalex_abstract_summary(abstract: str) -> str:
    prefix = "OpenAlex 摘要（metadata_only；未经全文核验）："
    value = prefix + abstract
    if len(value) <= _OPENALEX_ABSTRACT_SUMMARY_MAX_CHARS:
        return value
    return value[: _OPENALEX_ABSTRACT_SUMMARY_MAX_CHARS - 3].rstrip() + "..."


def _catalog() -> list[dict[str, Any]]:
    path = files("ecologyrsi_dsh.knowledge").joinpath("ecology_algorithms.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError("knowledge catalog must be an array")
    return [dict(item) for item in value]


def _query_terms(state: Any) -> tuple[str, ...]:
    metadata = state.task_manifest.metadata
    domain = str(metadata.get("domain") or state.task_manifest.domain_pack).casefold()
    research_domain = str(metadata.get("research_domain") or "").strip()
    base = (
        "greenhouse indoor climate crop production temperature prediction machine learning"
        if "greenhouse" in domain
        else "soil moisture crop water prediction model irrigation calibration"
    )
    terms = [base]
    if research_domain and research_domain.casefold() not in domain:
        terms.append(f"{research_domain} ecological forecasting algorithms")
    previous = state.analysis_for(state.run.generation - 1) if state.run.generation else None
    if previous is not None and previous.target_weaknesses:
        target = str(previous.target_weaknesses[0].get("target") or "")
        target_term = _TARGET_TERMS.get(target)
        horizon = previous.target_weaknesses[0].get("horizon_hours")
        if target_term:
            terms.append(
                f"{target_term} forecasting"
                + (f" {int(horizon)} hour horizon" if isinstance(horizon, int) else "")
            )
    if previous is not None and previous.horizon_weaknesses:
        weakest_horizon = _safe_horizon(
            previous.horizon_weaknesses[0].get("horizon_hours")
        )
        if weakest_horizon is not None:
            terms.append(
                f"time series forecasting {weakest_horizon} hour horizon error reduction"
            )
    if previous is not None:
        algorithm_codes = [
            token
            for row in previous.algorithm_failures[:8]
            if isinstance(row, Mapping)
            if (token := _safe_failure_token(row.get("failure_code"))) is not None
        ]
        if algorithm_codes:
            terms.append(
                "registered prediction algorithm compile debug repair "
                + " ".join(dict.fromkeys(algorithm_codes))
            )
        sample_classes: list[str] = []
        for row in previous.sample_failures[:8]:
            if not isinstance(row, Mapping):
                continue
            counts = row.get("failure_counts")
            if not isinstance(counts, Mapping):
                continue
            for raw_name, raw_count in list(counts.items())[:16]:
                if (
                    isinstance(raw_count, int)
                    and not isinstance(raw_count, bool)
                    and raw_count > 0
                ):
                    token = _safe_failure_token(raw_name)
                    if token is not None:
                        sample_classes.append(token)
        if sample_classes:
            terms.append(
                "multi agent prediction tool retry repair "
                + " ".join(dict.fromkeys(sample_classes))
            )
    strategy_id = str(metadata.get("strategy_id") or "")
    if strategy_id == "adaptive_local@1":
        terms.append("bounded adaptive parameter optimization time series validation")
    elif strategy_id == "parameter_sweep@1":
        terms.append("bounded parameter sensitivity search ecological model calibration")
    elif strategy_id == "autonomous_model@1":
        terms.append(
            "ecological forecasting model architecture team design strategy benchmark"
        )
    return tuple(dict.fromkeys(terms))[:6]


def _entry_matches_domain(entry: Mapping[str, Any], domain: str) -> bool:
    domains = entry.get("domains", [])
    return isinstance(domains, list) and (
        "all" in domains
        or ("greenhouse" in domain and "greenhouse" in domains)
        or ("toy" in domain and "crop_soil_water" in domains)
        or ("crop-soil-water" in domain and "crop_soil_water" in domains)
    )


def _online_enabled(metadata: Mapping[str, Any]) -> bool:
    configured = metadata.get("knowledge_online_enabled", False)
    if not isinstance(configured, bool):
        configured = False
    environment = os.environ.get("ECOLOGYRSI_KNOWLEDGE_ONLINE")
    if environment is not None:
        return environment.strip().casefold() in {"1", "true", "yes", "on"}
    return configured


def _openalex_timeout_seconds() -> float:
    raw = str(os.environ.get(_OPENALEX_TIMEOUT_ENV, "")).strip()
    if not raw:
        return _OPENALEX_DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(f"{_OPENALEX_TIMEOUT_ENV} must be a positive number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{_OPENALEX_TIMEOUT_ENV} must be a positive number")
    return timeout


def _retryable_openalex_error(exc: BaseException) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in _OPENALEX_RETRYABLE_HTTP_STATUS or 500 <= exc.code <= 599
    return isinstance(exc, (TimeoutError, URLError))


def _openalex_retry_after_seconds(exc: BaseException) -> float | None:
    if not isinstance(exc, HTTPError):
        return None
    headers = getattr(exc, "headers", None)
    raw = headers.get("Retry-After") if headers is not None else None
    value = str(raw or "").strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = retry_at.timestamp() - time.time()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _openalex_retry_delay(exc: BaseException, retry_count: int) -> float:
    exponential = min(
        _OPENALEX_RETRY_BASE_SECONDS * (2**retry_count),
        _OPENALEX_RETRY_MAX_SECONDS,
    )
    retry_after = _openalex_retry_after_seconds(exc)
    if retry_after is None:
        return exponential
    return min(
        _OPENALEX_RETRY_AFTER_MAX_SECONDS,
        max(exponential, retry_after),
    )


def _read_openalex_payload(request: Request) -> Any:
    timeout = _openalex_timeout_seconds()
    for retry_count in range(_OPENALEX_MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read(_OPENALEX_MAX_RESPONSE_BYTES)
            return json.loads(raw.decode("utf-8"))
        except (TimeoutError, URLError) as exc:
            if not _retryable_openalex_error(exc) or retry_count >= _OPENALEX_MAX_RETRIES:
                raise
            time.sleep(_openalex_retry_delay(exc, retry_count))
    raise AssertionError("unreachable OpenAlex retry state")


def _bounded_openalex_query(value: Any) -> str:
    query = " ".join(str(value or "").split())
    if len(query) <= _OPENALEX_QUERY_MAX_CHARS:
        return query
    bounded = query[:_OPENALEX_QUERY_MAX_CHARS]
    if " " in bounded:
        bounded = bounded.rsplit(" ", 1)[0]
    return bounded.strip()


def _openalex_query_plan(queries: tuple[str, ...]) -> tuple[str, ...]:
    if not queries:
        return ()
    planned: list[str] = []
    seen: set[str] = set()
    base_query = _bounded_openalex_query(queries[0])
    base_normalized = base_query.casefold()

    def add(raw_query: str) -> None:
        query = _bounded_openalex_query(raw_query)
        normalized = query.casefold()
        if not query or normalized in seen:
            return
        seen.add(normalized)
        planned.append(query)

    # ``_query_terms`` keeps the broad domain query first for the audit
    # snapshot. Search generation-specific terms first and retain that broad
    # query as the deterministic final fallback.
    for raw_query in queries[1:]:
        if len(planned) >= _OPENALEX_MAX_QUERIES - 1:
            break
        if _bounded_openalex_query(raw_query).casefold() == base_normalized:
            continue
        add(raw_query)
    add(base_query)
    return tuple(planned)


def _openalex_cards(
    query: str,
    *,
    limit: int = 5,
    required_title_markers: tuple[str, ...] = (),
) -> list[KnowledgeCard]:
    query = _bounded_openalex_query(query)
    if not query or limit <= 0:
        return []
    params = urlencode(
        {
            "search": query,
            "per-page": min(max(limit * 4, 10), 25),
            "select": (
                "id,display_name,publication_year,doi,primary_location,"
                "cited_by_count,abstract_inverted_index"
            ),
        }
    )
    url = "https://api.openalex.org/works?" + params
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "EcologyRSI-DSH/0.2 (metadata-only knowledge retrieval)",
        },
    )
    payload = _read_openalex_payload(request)
    results = payload.get("results", []) if isinstance(payload, Mapping) else []
    cards: list[KnowledgeCard] = []
    for item in results:
        if not isinstance(item, Mapping):
            continue
        work_id = str(item.get("id") or "").strip()
        title = str(item.get("display_name") or "").strip()
        if not work_id.startswith("https://openalex.org/") or not title:
            continue
        normalized_title = title.casefold()
        if required_title_markers and not any(
            marker in normalized_title for marker in required_title_markers
        ):
            continue
        location = item.get("primary_location")
        source_name = "OpenAlex"
        landing_url = work_id
        if isinstance(location, Mapping):
            source = location.get("source")
            if isinstance(source, Mapping) and source.get("display_name"):
                source_name = str(source["display_name"])
            candidate_url = location.get("landing_page_url")
            if isinstance(candidate_url, str) and candidate_url.startswith("https://"):
                landing_url = candidate_url
        cited = item.get("cited_by_count")
        abstract = _reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
        cards.append(
            KnowledgeCard(
                knowledge_id="openalex:" + work_id.rsplit("/", 1)[-1],
                title=title,
                summary=(
                    "OpenAlex 返回的论文元数据线索；尚未完成全文方法核验、数据条件比对"
                    "和本地实现适配，不能直接进入候选执行。"
                ),
                source_url=landing_url,
                source_kind="论文元数据",
                source_authority=source_name,
                execution_status="metadata_only",
                selection_reason=(
                    "在线结果仅用于扩展研究线索；未映射到冻结运行的本地能力，"
                    "因此不执行其代码或参数。"
                ),
                algorithm_tags=("online_metadata",),
                publication_year=(
                    int(item["publication_year"])
                    if isinstance(item.get("publication_year"), int)
                    else None
                ),
                cited_by_count=int(cited) if isinstance(cited, int) and cited >= 0 else None,
                abstract_summary=(
                    _openalex_abstract_summary(abstract) if abstract is not None else None
                ),
                content_digest=digest(abstract) if abstract is not None else None,
            )
        )
        if len(cards) >= limit:
            break
    return cards


def _openalex_cards_for_queries(
    queries: tuple[str, ...],
    *,
    limit: int = 5,
    required_title_markers: tuple[str, ...] = (),
) -> list[KnowledgeCard]:
    plan = _openalex_query_plan(queries)
    if not plan or limit <= 0:
        return []
    cards: list[KnowledgeCard] = []
    seen_ids: set[str] = set()
    for query in plan:
        query_cards = _openalex_cards(
            query,
            limit=limit,
            required_title_markers=required_title_markers,
        )
        for card in query_cards:
            if card.knowledge_id in seen_ids:
                continue
            seen_ids.add(card.knowledge_id)
            cards.append(card)
            if len(cards) >= limit:
                return cards
    return cards


def retrieve_generation_knowledge(state: Any) -> KnowledgeSnapshot:
    """Collect and freeze evidence for the current generation."""

    metadata = state.task_manifest.metadata
    domain = str(metadata.get("domain") or state.task_manifest.domain_pack).casefold()
    queries = _query_terms(state)
    entries = [item for item in _catalog() if _entry_matches_domain(item, domain)]
    cards = [map_catalog_entry(item, metadata) for item in entries]
    warnings: list[str] = []
    online = _online_enabled(metadata)
    online_cards: list[KnowledgeCard] = []
    status = "catalog_only"
    if online:
        try:
            markers = (
                ("greenhouse", "controlled environment", "plant factor")
                if "greenhouse" in domain
                else (
                    "soil moisture",
                    "soil water",
                    "crop water",
                    "irrigation",
                    "evapotranspiration",
                    "aquacrop",
                )
            )
            online_cards = _openalex_cards_for_queries(
                queries,
                limit=5,
                required_title_markers=markers,
            )
            status = "catalog_and_online" if online_cards else "catalog_online_empty"
        except Exception as exc:  # noqa: BLE001 - optional metadata provider isolation
            status = "catalog_online_fallback"
            warnings.append(
                f"OpenAlex 元数据检索失败，已使用内置目录：{type(exc).__name__}"
            )
    cards.extend(online_cards)
    cards.sort(
        key=lambda item: (
            {"adopted": 0, "available_not_selected": 1, "research_only": 2, "metadata_only": 3}[
                item.execution_status
            ],
            item.knowledge_id,
        )
    )
    return KnowledgeSnapshot(
        run_id=state.run.run_id,
        generation=state.run.generation,
        query_terms=queries,
        cards=tuple(cards[:24]),
        online_enabled=online,
        provider="内置核验目录 + OpenAlex 元数据" if online else "内置核验目录",
        retrieval_status=status,
        warnings=tuple(warnings),
    )


def assess_generation_knowledge(
    state: Any,
    analysis: Any,
    snapshot: KnowledgeSnapshot,
) -> KnowledgeAssessment:
    """Describe observed round outcome without claiming knowledge caused it."""

    incumbent = (
        state.evaluation_for(analysis.incumbent_before_candidate_id)
        if analysis.incumbent_before_candidate_id
        else None
    )
    selected = (
        state.evaluation_for(analysis.selected_candidate_id)
        if analysis.selected_candidate_id
        else None
    )
    delta = (
        float(selected.score - incumbent.score)
        if selected is not None and incumbent is not None
        else None
    )
    adopted_count = len(snapshot.executable_cards)
    algorithm_failures = tuple(getattr(analysis, "algorithm_failures", ()))
    sample_failures = tuple(getattr(analysis, "sample_failures", ()))
    if analysis.outcome == "promoted":
        outcome = "observed_progress"
        conclusion = (
            f"本轮采用 {adopted_count} 条可执行知识映射，并产生严格改善的新冠军；"
            "该结果是联合搜索观察，不能归因为单条知识。"
        )
        next_action = "保留已采用映射，围绕新冠军缩小步长，并根据新弱点更新检索词。"
    elif analysis.outcome == "no_improvement":
        outcome = "no_observed_improvement"
        conclusion = (
            f"本轮采用 {adopted_count} 条可执行知识映射，但候选未严格改善当前最优方案。"
        )
        next_action = "保留当前最优方案，修订检索词并更换一个可执行参数方向。"
    else:
        outcome = "no_eligible_candidate"
        conclusion = (
            f"本轮采用 {adopted_count} 条可执行知识映射，但没有候选同时通过科学门禁和独立评审。"
        )
        next_action = "回到当前最优方案附近，优先检索约束修复和更保守的已注册方法。"
    observed_focus = str(getattr(analysis, "next_generation_focus", "") or "").strip()
    if observed_focus:
        next_action = observed_focus[:1000]
    has_sample_execution_failure = any(
        int(row.get(name, 0) or 0) > 0
        for row in sample_failures
        for name in ("failed", "skipped", "input_failures")
    ) or any(row.get("coverage_pass") is False for row in sample_failures)
    if algorithm_failures:
        next_action = "优先修复上一轮算法编译/调试失败码，仅更换已登记适配器配置。"
    elif has_sample_execution_failure:
        next_action = "优先降低样本工具调用失败率并提高覆盖率，不要求每个样本成功。"
    return KnowledgeAssessment(
        run_id=state.run.run_id,
        generation=analysis.generation,
        snapshot_digest=snapshot.snapshot_digest,
        outcome=outcome,
        conclusion=conclusion,
        next_action=next_action,
        selected_candidate_id=analysis.selected_candidate_id,
        score_delta=delta,
    )


__all__ = ["assess_generation_knowledge", "retrieve_generation_knowledge"]
