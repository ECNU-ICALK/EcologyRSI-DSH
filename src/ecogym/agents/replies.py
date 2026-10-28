"""The JSON reply protocol: every agent reply is one JSON object."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .clients import AgentError


def extract_json(text: str) -> dict[str, Any]:
    """The first complete JSON object in ``text`` (code fences and prose around it are ignored)."""

    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    raise AgentError("the reply contains no JSON object")


def require(value: Mapping[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    if key not in value or not isinstance(value[key], kind):
        names = kind.__name__ if isinstance(kind, type) else "/".join(item.__name__ for item in kind)
        raise AgentError(f"the reply needs '{key}' as {names}")
    return value[key]


__all__ = ["extract_json", "require"]
