"""Strict parsing of TOML configuration.

Specs and task cards are plain TOML read with ``tomllib``. ``Fields`` wraps
one table, hands out typed values with defaults and bounds, and rejects any
key nobody asked for, so a typo in a task card fails loudly instead of
silently falling back to a default.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

_MISSING = object()


class ConfigError(ValueError):
    """A spec, task card or run configuration is invalid."""


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        try:
            return tomllib.load(stream)
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{path}: {error}") from None


class Fields:
    """Typed, exhaustive access to one configuration table."""

    def __init__(self, mapping: Mapping[str, Any], where: str) -> None:
        if not isinstance(mapping, Mapping):
            raise ConfigError(f"{where} must be a table")
        self._mapping = dict(mapping)
        self._used: set[str] = set()
        self.where = where

    def __contains__(self, key: str) -> bool:
        return key in self._mapping

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._mapping))

    def _take(self, key: str, default: Any) -> Any:
        self._used.add(key)
        if key in self._mapping:
            return self._mapping[key]
        if default is _MISSING:
            raise ConfigError(f"{self.where}.{key} is required")
        return default

    def raw(self, key: str, default: Any = _MISSING) -> Any:
        return self._take(key, default)

    def str(self, key: str, default: Any = _MISSING, *, choices: tuple[str, ...] = ()) -> str:
        value = self._take(key, default)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{self.where}.{key} must be a non-empty string")
        if choices and value not in choices:
            raise ConfigError(f"{self.where}.{key} must be one of {', '.join(choices)}")
        return value

    def bool(self, key: str, default: Any = _MISSING) -> bool:
        value = self._take(key, default)
        if not isinstance(value, bool):
            raise ConfigError(f"{self.where}.{key} must be true or false")
        return value

    def int(self, key: str, default: Any = _MISSING, *, lo: int | None = None, hi: int | None = None) -> int:
        value = self._take(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{self.where}.{key} must be an integer")
        _check_bounds(self.where, key, value, lo, hi)
        return value

    def float(
        self, key: str, default: Any = _MISSING, *, lo: float | None = None, hi: float | None = None
    ) -> float:
        value = self._take(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigError(f"{self.where}.{key} must be a finite number")
        _check_bounds(self.where, key, float(value), lo, hi)
        return float(value)

    def strs(self, key: str, default: Any = _MISSING) -> tuple[str, ...]:
        value = self._take(key, default)
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and item for item in value):
            raise ConfigError(f"{self.where}.{key} must be a list of strings")
        return tuple(value)

    def ints(self, key: str, default: Any = _MISSING, *, lo: int | None = None) -> tuple[int, ...]:
        value = self._take(key, default)
        if not isinstance(value, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in value
        ):
            raise ConfigError(f"{self.where}.{key} must be a list of integers")
        for item in value:
            _check_bounds(self.where, key, item, lo, None)
        return tuple(value)

    def table(self, key: str, *, required: bool = False) -> Fields:
        """A sub-table; an absent optional table reads as empty."""

        value = self._take(key, _MISSING if required else {})
        return Fields(value, f"{self.where}.{key}")

    def tables(self, key: str, default: Any = _MISSING) -> list[Fields]:
        value = self._take(key, default)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{self.where}.{key} must be an array of tables")
        return [Fields(item, f"{self.where}.{key}[{index}]") for index, item in enumerate(value)]

    def done(self) -> None:
        unknown = sorted(set(self._mapping) - self._used)
        if unknown:
            raise ConfigError(f"{self.where} has unknown keys: {', '.join(unknown)}")


def _check_bounds(where: str, key: str, value: float, lo: float | None, hi: float | None) -> None:
    if lo is not None and value < lo:
        raise ConfigError(f"{where}.{key} must be >= {lo}")
    if hi is not None and value > hi:
        raise ConfigError(f"{where}.{key} must be <= {hi}")


__all__ = ["ConfigError", "Fields", "load_toml"]
