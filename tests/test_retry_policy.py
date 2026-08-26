"""Regression tests for shared gateway retry contracts."""

from __future__ import annotations

import unittest

from ecologyrsi_dsh.core.retry_policy import retry_policy


class GatewayRetryPolicyTests(unittest.TestCase):
    def test_retry_policy_keeps_public_contract(self) -> None:
        """The DSH outage contract must retain its published circuit and action."""

        policy = retry_policy("dsh_native_runtime")
        self.assertEqual(policy.circuit_code, "dsh_runtime_retry_circuit_open")
        self.assertEqual(policy.suggested_action, "check_dsh_runtime_then_resume")
