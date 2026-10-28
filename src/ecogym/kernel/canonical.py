"""Canonical JSON and content digests.

Every identity in ecogym (dataset, task, partitions, genome, evaluation,
ledger event) is the SHA-256 of one canonical JSON encoding: sorted keys, no
whitespace, UTF-8, and no NaN or infinity. ``jsonable`` maps the few
non-JSON values the framework produces (dataclasses, tuples, numpy scalars
and arrays, non-finite floats) onto that encoding.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    """Return ``value`` as plain JSON data; non-finite floats become ``None``."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON data")


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value: Any) -> str:
    """SHA-256 hex digest of the canonical JSON encoding of ``value``."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


__all__ = ["canonical_json", "digest", "jsonable"]
