"""Contract tests for strict origin protocol capabilities."""

from __future__ import annotations

import unittest

from ecologyrsi_dsh.core.protocols import (
    is_strict_origin_protocol,
    supports_concurrent_origins,
    supports_two_stage_screening,
)


class ProtocolContractTests(unittest.TestCase):
    def test_legacy_and_current_strict_protocols_keep_distinct_capabilities(self) -> None:
        self.assertTrue(is_strict_origin_protocol("dsh-strict-origin-bundle@3"))
        self.assertTrue(is_strict_origin_protocol("dsh-strict-origin-bundle@4"))
        self.assertFalse(supports_concurrent_origins("dsh-strict-origin-bundle@3"))
        self.assertTrue(supports_concurrent_origins("dsh-strict-origin-bundle@4"))
        self.assertTrue(supports_two_stage_screening("dsh-strict-origin-bundle@4"))
