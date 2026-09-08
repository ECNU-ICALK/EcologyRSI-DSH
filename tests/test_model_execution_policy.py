import unittest

from ecologyrsi_dsh.core.model_execution_policy import (
    RESEARCH_EXECUTION_POLICY, research_execution_policy,
)
from ecologyrsi_dsh.core.models import TaskManifest


class ModelExecutionPolicyTests(unittest.TestCase):
    def test_old_report_policy_remains_readable_without_silent_upgrade(self):
        policy = {**RESEARCH_EXECUTION_POLICY, "synthesis_report_format": "concise@1"}
        original = self.task({"research_execution_policy": policy})
        restored = TaskManifest.from_dict(original.to_dict())
        self.assertEqual(restored.digest, original.digest)
        self.assertEqual(research_execution_policy(restored.metadata), policy)
        self.assertEqual(RESEARCH_EXECUTION_POLICY["synthesis_report_format"], "concise@2")

    def task(self, metadata):
        return TaskManifest(task_id="policy", objective="bounded research", domain_pack="test", metadata=metadata)

    def test_policy_is_frozen_in_task_digest_without_upgrading_history(self):
        original = self.task({})
        self.assertIsNone(research_execution_policy(original.metadata))
        supplied = dict(RESEARCH_EXECUTION_POLICY)
        current = self.task({"research_execution_policy": supplied})
        self.assertNotEqual(original.digest, current.digest)
        supplied["synthesis_max_output_tokens"] = 999999
        self.assertEqual(research_execution_policy(current.metadata)["synthesis_max_output_tokens"], 16384)
        self.assertIsNone(research_execution_policy(original.metadata))
        self.assertEqual(TaskManifest.from_dict(current.to_dict()).digest, current.digest)

    def test_unbounded_or_ambiguous_policy_is_rejected(self):
        for key, value in (
            ("synthesis_max_output_tokens", 32768),
            ("synthesis_max_output_tokens", 16384.0),
            ("retry_identical_exhausted_request", True),
            ("retry_identical_exhausted_request", 0),
            ("synthesis_context_format", "unknown"),
            ("synthesis_report_format", "concise@999"),
            ("extra", "ignored"),
        ):
            with self.subTest(key=key, value=value):
                policy = {**RESEARCH_EXECUTION_POLICY, key: value}
                with self.assertRaisesRegex(ValueError, "research execution policy"):
                    self.task({"research_execution_policy": policy})
        for value in (None, {}, [], "compact"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "research execution policy"):
                self.task({"research_execution_policy": value})
