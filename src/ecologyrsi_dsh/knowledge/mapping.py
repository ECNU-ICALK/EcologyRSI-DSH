"""Map knowledge claims to capabilities already registered in the host."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .algorithms import registered_capability_ids
from .models import KnowledgeCard


def _active_capabilities(metadata: Mapping[str, Any]) -> dict[str, str]:
    return {
        "strategy": str(metadata.get("strategy_id") or "parameter_sweep@1"),
        "predictor": str(metadata.get("prediction_model_id") or ""),
        "evaluator": str(metadata.get("evaluator_id") or ""),
    }


def map_catalog_entry(
    entry: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> KnowledgeCard:
    """Make an explicit execution decision against the frozen run binding."""

    raw_mapping = entry.get("execution_mapping")
    capability_kind = None
    capability_id = None
    capability_ids: tuple[str, ...] = ()
    parameter_hints: tuple[str, ...] = ()
    status = "research_only"
    reason = "当前宿主没有与该算法对应的已注册执行适配器，仅保留为研究线索。"
    if isinstance(raw_mapping, Mapping):
        capability_kind = str(raw_mapping.get("kind") or "") or None
        raw_ids = raw_mapping.get("capability_ids")
        if isinstance(raw_ids, list):
            configured_ids = tuple(
                dict.fromkeys(str(item).strip() for item in raw_ids if str(item).strip())
            )
        else:
            single_id = str(raw_mapping.get("capability_id") or "").strip()
            configured_ids = (single_id,) if single_id else ()
        capability_id = configured_ids[0] if configured_ids else None
        hints = raw_mapping.get("parameter_hints", [])
        if isinstance(hints, list):
            parameter_hints = tuple(str(item) for item in hints if str(item).strip())
    if capability_kind and capability_id:
        capability_ids = tuple(
            item
            for item in configured_ids
            if item in registered_capability_ids(capability_kind)
        )
        active = _active_capabilities(metadata).get(capability_kind)
        if active in capability_ids:
            capability_id = active
            status = "adopted"
            reason = (
                f"已映射到本运行冻结的{capability_kind}能力 {capability_id}；"
                "只提供结构化参数与评测方法依据。"
            )
        elif capability_ids:
            capability_id = capability_ids[0]
            status = "available_not_selected"
            reason = (
                f"宿主已登记 {capability_id}，但本运行冻结的是 {active or '未选择'}；"
                "不能在轮次中途切换实现。"
            )
    return KnowledgeCard(
        knowledge_id=str(entry["knowledge_id"]),
        title=str(entry["title_zh"]),
        summary=str(entry["summary_zh"]),
        source_url=str(entry["source_url"]),
        source_kind=str(entry["source_kind"]),
        source_authority=str(entry["source_authority"]),
        execution_status=status,
        selection_reason=reason,
        algorithm_tags=tuple(str(item) for item in entry.get("algorithm_tags", [])),
        capability_kind=capability_kind,
        capability_id=capability_id,
        capability_ids=capability_ids,
        parameter_hints=parameter_hints,
        publication_year=entry.get("publication_year"),
    )


def knowledge_focus_parameter(
    context: Mapping[str, Any] | None,
    schemas: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Choose only a host-registered parameter named by adopted knowledge."""

    if not isinstance(context, Mapping):
        return None
    adopted = context.get("adopted_knowledge")
    if not isinstance(adopted, list):
        return None
    for item in adopted:
        if not isinstance(item, Mapping):
            continue
        hints = item.get("parameter_hints")
        if not isinstance(hints, list):
            continue
        for name in hints:
            if isinstance(name, str) and name in schemas:
                return name
    return None


__all__ = ["knowledge_focus_parameter", "map_catalog_entry"]
