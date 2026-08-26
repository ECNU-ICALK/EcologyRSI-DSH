"""Regression tests for bounded exception graph traversal."""

from __future__ import annotations

import builtins
import unittest

from ecologyrsi_dsh.core.errors import find_exception, walk_exception_graph


class MarkerError(RuntimeError):
    """A test-only exception used to identify the nested leaf."""


class _ExceptionGroupCompat(RuntimeError):
    """Python 3.10-compatible stand-in for an exception group."""

    def __init__(self, message: str, exceptions: list[BaseException]) -> None:
        super().__init__(message)
        self.exceptions = tuple(exceptions)


class ExceptionGraphTests(unittest.TestCase):
    def test_exception_graph_is_cycle_safe_and_reads_groups(self) -> None:
        """A graph traversal must find a grouped leaf without revisiting cycles."""

        leaf = MarkerError("leaf")
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = leaf
        wrapper.__context__ = wrapper
        exception_group = getattr(
            builtins,
            "ExceptionGroup",
            _ExceptionGroupCompat,
        )
        grouped = exception_group("group", [wrapper, ValueError("peer")])

        self.assertIs(find_exception(grouped, MarkerError), leaf)
        self.assertLessEqual(len(tuple(walk_exception_graph(grouped))), 4)
