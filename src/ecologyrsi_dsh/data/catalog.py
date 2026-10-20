"""Dataset storage and frozen-snapshot identities."""

from __future__ import annotations

import os
from pathlib import Path

from ..core.errors import FrozenRuntimeBindingDriftError


def _assert_snapshot_digest(
    label: str,
    actual: str,
    expected: str | None,
) -> None:
    if expected is None:
        return
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError(f"{label}的冻结校验值必须是非空字符串")
    normalized = expected.strip()
    if normalized != actual:
        raise FrozenRuntimeBindingDriftError(label, expected_digest=normalized, current_digest=actual)


def _default_data_root() -> Path:
    configured = os.environ.get("ECOLOGYRSI_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    # Downloads and every normal service start share one durable location.
    # Do not change roots based on which old checkout happens to exist.
    return (Path.home() / ".ecologyrsi-dsh" / "data" / "greenhouse").resolve()
