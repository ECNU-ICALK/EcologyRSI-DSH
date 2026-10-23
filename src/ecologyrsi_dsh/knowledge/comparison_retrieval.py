"""Shared method evidence for evolution comparisons, separate from certification.

Research stages retain online retrieval. Comparison stages query the same
Host-curated library frozen in RunCreated; labels and online/model snippets
never become inputs to this library. Legacy manifests remain unchanged.
"""
from collections.abc import Mapping
from typing import Any
from ..core.immutable import thaw_json

from .formal_retrieval import build_formal_retrieval_snapshot, validate_formal_retrieval_snapshot

COMPARISON_RETRIEVAL_POLICY_KEY = "comparison_retrieval_policy"
COMPARISON_RETRIEVAL_POLICY = "paired_method_snapshot@1"
COMPARISON_RETRIEVAL_SNAPSHOT_KEY = "comparison_retrieval_snapshot"
COMPARISON_RETRIEVAL_STAGES = frozenset({
    "sample.plan", "sample.critic", "sample.reflect", "generation.judge",
})


def comparison_retrieval_enabled(metadata: Mapping[str, Any]) -> bool:
    value = metadata.get(COMPARISON_RETRIEVAL_POLICY_KEY)
    if value not in (None, COMPARISON_RETRIEVAL_POLICY):
        raise ValueError("unsupported comparison retrieval policy")
    if value is None and COMPARISON_RETRIEVAL_SNAPSHOT_KEY in metadata:
        raise ValueError("comparison snapshot requires its frozen policy")
    return value == COMPARISON_RETRIEVAL_POLICY


def comparison_retrieval_snapshot(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    if not comparison_retrieval_enabled(metadata):
        return None
    # Never rebuild a missing/old snapshot from the current catalog on resume.
    return validate_formal_retrieval_snapshot(thaw_json(metadata.get(COMPARISON_RETRIEVAL_SNAPSHOT_KEY)))


def comparison_retrieval_snapshot_digest(metadata: Mapping[str, Any]) -> str | None:
    snapshot = comparison_retrieval_snapshot(metadata)
    return snapshot["snapshot_digest"] if snapshot else None


def freeze_comparison_retrieval_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Creation-time helper. Reuse an existing frozen snapshot without refresh."""
    result = dict(metadata)
    if not comparison_retrieval_enabled(result):
        return result
    if COMPARISON_RETRIEVAL_SNAPSHOT_KEY not in result:
        result[COMPARISON_RETRIEVAL_SNAPSHOT_KEY] = build_formal_retrieval_snapshot(result)
    result[COMPARISON_RETRIEVAL_SNAPSHOT_KEY] = comparison_retrieval_snapshot(result)
    return result


def comparison_retrieval_context(metadata: Mapping[str, Any], stage: str) -> dict[str, Any] | None:
    if stage not in COMPARISON_RETRIEVAL_STAGES:
        return None
    snapshot = comparison_retrieval_snapshot(metadata)
    if snapshot is None:
        return None
    return {
        "comparison_retrieval": {
            "mode": "frozen_snapshot", "snapshot_digest": snapshot["snapshot_digest"],
            "document_count": len(snapshot["documents"]), "online_enabled": False,
        },
        "comparison_retrieval_scope": snapshot["scope"],
        "comparison_retrieval_policy": COMPARISON_RETRIEVAL_POLICY,
        "retrieval_policy": dict(snapshot["retrieval_policy"]),
    }
