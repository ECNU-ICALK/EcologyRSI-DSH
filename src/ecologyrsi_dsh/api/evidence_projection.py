"""Detail-only evidence projection."""

from __future__ import annotations

from typing import Any


def build_evidence_projection(state: Any) -> dict[str, object]:
    from .projection import _evolution_evidence_projection

    return dict(_evolution_evidence_projection(state))
