"""Canonical JSON identities for immutable scientific and research inputs."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any


def normalized_json(value: Any) -> Any:
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        return unicodedata.normalize("NFC", value)
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite_json_number")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("nonstring_json_key")
            key = unicodedata.normalize("NFC", key)
            if key in result:
                raise ValueError("duplicate_normalized_json_key")
            result[key] = normalized_json(item)
        return result
    if isinstance(value, (tuple, list)):
        return [normalized_json(item) for item in value]
    raise TypeError("unsupported_json_type")


def canonical_text(value: Any) -> str:
    return json.dumps(normalized_json(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def content_id(namespace: str, value: Any) -> str:
    if type(namespace) is not str or not namespace or "\0" in namespace:
        raise ValueError("invalid_hash_namespace")
    payload = namespace.encode("utf-8") + b"\0" + canonical_text(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        key = unicodedata.normalize("NFC", key)
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(text: str) -> None:
    raise ValueError("invalid_json_constant:" + text)


def parse_object(text: str) -> dict[str, Any]:
    if type(text) is not str:
        raise TypeError("json_text_required")
    value = json.loads(text, object_pairs_hook=_unique_pairs,
                       parse_constant=_reject_constant)
    if type(value) is not dict:
        raise TypeError("json_object_required")
    return normalized_json(value)


@dataclass(frozen=True, slots=True)
class FrozenObject:
    canonical: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical", canonical_text(parse_object(self.canonical)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenObject":
        if not isinstance(value, Mapping):
            raise TypeError("mapping_required")
        return cls(canonical_text(value))

    def to_dict(self) -> dict[str, Any]:
        return parse_object(self.canonical)

    def digest(self, namespace: str) -> str:
        return content_id(namespace, self.to_dict())
