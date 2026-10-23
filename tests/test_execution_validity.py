"""A terminal critique rejects the candidate; a broken execution stays paused."""
from copy import deepcopy
import unittest

from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
from ecologyrsi_dsh.evaluators.execution_validity import (
    EXECUTION_OUTCOME_POLICY, EXECUTION_OUTCOME_POLICY_KEY,
    complete_origin_coverage, execution_outcome_policy_enabled, require_valid_execution,
)
from ecologyrsi_dsh.evolution.champion_challenger import local_challenger_safety_reason


POLICY = {EXECUTION_OUTCOME_POLICY_KEY: EXECUTION_OUTCOME_POLICY}


def rejected_metrics():
    return {"scientific_pass": False, "sample_execution_coverage_pass": False,
            "constraint_violations": 0,
            "sample_execution": {"attempted_origin_samples": 50, "succeeded_origin_samples": 49,
                "failed_origin_samples": 1, "attempted_examples": 450, "succeeded_examples": 449,
                "failed_examples": 1, "failure_counts": {"critic_revise": 1},
                "coverage_pass": False, "successful_agent_provenance_pass": True}}


class ExecutionValidityTests(unittest.TestCase):
    def test_terminal_critique_settles_but_cannot_win(self):
        for reason in ("critic_revise", "critic_uncertain"):
            metrics = rejected_metrics()
            metrics["sample_execution"]["failure_counts"] = {reason: 1}
            before = deepcopy(metrics)
            require_valid_execution(metrics, 50, phase="holdout", metadata=POLICY)
            self.assertEqual(metrics, before)  # No substituted prediction, count, or score.
            self.assertEqual(local_challenger_safety_reason(metrics), "coverage_guardrail_failed")
            self.assertFalse(complete_origin_coverage(metrics["sample_execution"], POLICY))

    def test_old_run_remains_paused_and_unknown_policy_rejected(self):
        with self.assertRaises(DshNativeRuntimeUnavailableError):
            require_valid_execution(rejected_metrics(), 50, phase="holdout")
        self.assertFalse(execution_outcome_policy_enabled({}))
        with self.assertRaises(ValueError):
            execution_outcome_policy_enabled({EXECUTION_OUTCOME_POLICY_KEY: "unknown"})

    def test_transport_unknown_and_mixed_failures_stay_paused(self):
        for failures in ({"remote_rejected": 1}, {"invalid_output": 1}, {"critic_revise": 1, "timeout": 1}, {}):
            metrics = rejected_metrics()
            metrics["sample_execution"]["failure_counts"] = failures
            with self.assertRaises(DshNativeRuntimeUnavailableError):
                require_valid_execution(metrics, 50, phase="holdout", metadata=POLICY)

    def test_missing_inconsistent_counts_or_success_flags_fail_closed(self):
        changes = [("sample_execution", "failed_examples", 2),
                   ("sample_execution", "attempted_origin_samples", 49),
                   ("sample_execution", "failed_origin_samples", True),
                   ("sample_execution", "attempted_examples", 449),
                   ("sample_execution", "coverage_pass", True),
                   (None, "scientific_pass", True),
                   (None, "sample_execution_coverage_pass", True)]
        for scope, key, value in changes:
            metrics = rejected_metrics()
            (metrics[scope] if scope else metrics)[key] = value
            with self.assertRaises(DshNativeRuntimeUnavailableError):
                require_valid_execution(metrics, 50, phase="holdout", metadata=POLICY)

    def test_complete_cohort_and_partial_or_missing_evidence(self):
        summary = {"attempted_origin_samples": 50, "succeeded_origin_samples": 50}
        self.assertTrue(complete_origin_coverage(summary, POLICY))
        require_valid_execution({"sample_execution": summary}, 50, phase="holdout", metadata=POLICY)
        self.assertFalse(complete_origin_coverage({}, POLICY))
        self.assertFalse(complete_origin_coverage({"attempted_origin_samples": True,
                                                  "succeeded_origin_samples": True}, POLICY))
        with self.assertRaises(DshNativeRuntimeUnavailableError):
            require_valid_execution({}, 50, phase="holdout", metadata=POLICY)
