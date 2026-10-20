"""Native local-edit validation and restart-safe, bounded correction."""

from copy import deepcopy
import unittest
from unittest.mock import Mock

from ecologyrsi_dsh.application.formal_trajectory import _validated_local_edit_response
from ecologyrsi_dsh.core.errors import (
    DshNativeRuntimeUnavailableError,
    dsh_native_runtime_retryable,
)
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.evolution.local_edits import LocalEditProposal


def proposal(length=400):
    return {
        "schema_version": "ecology-local-edit@1",
        "decision": "mutate",
        "operations": [{"op": "set_bounded_parameter", "name": "ridge_alpha", "value": 0.2}],
        "evidence_refs": ["batch:score"],
        "expected_effect_cells": ["air_temperature@1h"],
        "risk_cells": [],
        "revisit": {"reason": "new_batch_evidence", "justification": "证" * length},
    }


class LocalEditRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.request = {
            "run_id": "run:local-edit-repair",
            "stage": "candidate.local_edit",
            "role": "candidate-proposer",
            "context": {"batch_evidence": {"score": -0.02}, "maximum_operations": 2},
            "output_schema_id": "ecology-local-edit@1",
            "run_state_revision": 41,
            "stage_attempt": 1,
            "ledger_expected_revision": 41,
            "idempotency_key": "run:local-edit-repair:candidate:one:batch:4:local-edit",
            "identity_digests": {"genome_digest": "a" * 64},
        }

    def test_live_shaped_549_character_revisit_is_rejected_without_truncation(self):
        invalid = proposal(549)
        with self.assertRaisesRegex(ValueError, "at most 400 characters"):
            LocalEditProposal.from_dict(invalid)
        self.assertEqual(len(invalid["revisit"]["justification"]), 549)
        LocalEditProposal.from_dict(proposal(400))
        with self.assertRaises(ValueError):
            LocalEditProposal.from_dict(proposal(401))

    def test_one_correction_preserves_evidence_and_uses_a_new_stable_identity(self):
        rejected, corrected = proposal(549), proposal(90)
        runtime = Mock()
        runtime.run.side_effect = [rejected, corrected]
        before = deepcopy(self.request)
        result = _validated_local_edit_response(runtime, **self.request)
        self.assertEqual(result.to_dict(), corrected)
        self.assertEqual(self.request, before)
        first, second = [c.kwargs for c in runtime.run.call_args_list]
        self.assertEqual(first, self.request)
        self.assertEqual(second["stage_attempt"], 2)
        self.assertEqual(second["idempotency_key"], first["idempotency_key"] + ":contract-repair:2")
        for name in ("run_id", "stage", "role", "identity_digests", "output_schema_id"):
            self.assertEqual(first[name], second[name])
        feedback = second["context"].pop("host_validation_feedback")
        self.assertEqual(second["context"], first["context"])
        self.assertEqual(feedback["rejected_output_digest"], digest(rejected))
        self.assertIn("400 characters", feedback["validation_detail"])
        self.assertNotIn(rejected["revisit"]["justification"], str(feedback))

    def test_valid_output_needs_no_correction(self):
        runtime = Mock()
        runtime.run.return_value = proposal()
        self.assertEqual(_validated_local_edit_response(runtime, **self.request).to_dict(), proposal())
        runtime.run.assert_called_once_with(**self.request)

    def test_output_exhaustion_uses_existing_correction_with_unchanged_budget(self):
        error = DshNativeRuntimeUnavailableError(
            "private-provider-diagnostic", error_code="structured_child_output_budget_exhausted",
            failure_domain="execution_contract", status_code=422,
        )
        request = {**self.request, "max_tokens": 32768}
        runtime = Mock()
        runtime.run.side_effect = [error, proposal(90)]
        self.assertEqual(_validated_local_edit_response(runtime, **request).to_dict(), proposal(90))
        first, second = [c.kwargs for c in runtime.run.call_args_list]
        self.assertEqual(first, request)
        self.assertEqual(second["stage_attempt"], 2)
        self.assertEqual(second["idempotency_key"], first["idempotency_key"] + ":contract-repair:2")
        feedback = second["context"].pop("host_validation_feedback")
        self.assertEqual(feedback["rejection_code"], "output_budget_exhausted_before_structured_output")
        self.assertEqual(feedback["output_token_budget"], 32768)
        self.assertNotIn("private-provider-diagnostic", str(feedback))
        self.assertNotIn("rejected_output_digest", feedback)
        second["stage_attempt"] = first["stage_attempt"]
        second["idempotency_key"] = first["idempotency_key"]
        self.assertEqual(second, first)

    def test_output_exhaustion_and_schema_rejection_share_one_correction(self):
        for failures in ("budget-budget", "schema-budget", "budget-schema"):
            with self.subTest(failures=failures):
                error = DshNativeRuntimeUnavailableError(
                    error_code="structured_child_output_budget_exhausted", status_code=422,
                    failure_domain="execution_contract",
                )
                runtime = Mock()
                runtime.run.side_effect = [error if kind == "budget" else proposal(549)
                                           for kind in failures.split("-")]
                with self.assertRaises(DshNativeRuntimeUnavailableError) as caught:
                    _validated_local_edit_response(runtime, **self.request)
                self.assertEqual(runtime.run.call_count, 2)
                self.assertFalse(dsh_native_runtime_retryable(caught.exception))
                if failures.endswith("budget"):
                    self.assertIs(caught.exception, error)
                else:
                    self.assertEqual(caught.exception.error_code, "structured_child_output_schema_invalid")

    def test_other_native_contract_failures_do_not_consume_correction(self):
        for code in ("structured_child_execution_budget_exhausted", "structured_result_missing",
                     "structured_child_output_schema_invalid", "structured_role_operational_timeout"):
            with self.subTest(code=code):
                error = DshNativeRuntimeUnavailableError(error_code=code, status_code=422)
                runtime = Mock()
                runtime.run.side_effect = error
                with self.assertRaises(DshNativeRuntimeUnavailableError) as caught:
                    _validated_local_edit_response(runtime, **self.request)
                self.assertIs(caught.exception, error)
                runtime.run.assert_called_once_with(**self.request)

    def test_output_repair_resume_reuses_failed_and_successful_identities(self):
        receipts, requests = {}, []
        def run(**request):
            key = request["idempotency_key"]
            if key not in receipts:
                requests.append(deepcopy(request))
                receipts[key] = (DshNativeRuntimeUnavailableError(
                    error_code="structured_child_output_budget_exhausted", status_code=422)
                    if request["stage_attempt"] == 1 else proposal(100))
            if isinstance(receipts[key], Exception):
                raise receipts[key]
            return deepcopy(receipts[key])
        runtime = Mock(run=run)
        first = _validated_local_edit_response(runtime, **self.request)
        resumed = _validated_local_edit_response(runtime, **self.request)
        self.assertEqual(first, resumed)
        self.assertEqual(len(requests), 2)
        self.assertNotIn("output_token_budget", requests[1]["context"]["host_validation_feedback"])

    def test_two_invalid_results_fail_as_contract_error_not_host_fault_or_keep(self):
        runtime = Mock()
        runtime.run.return_value = proposal(549)
        with self.assertRaises(DshNativeRuntimeUnavailableError) as caught:
            _validated_local_edit_response(runtime, **self.request)
        self.assertEqual(runtime.run.call_count, 2)
        self.assertEqual(caught.exception.error_code, "structured_child_output_schema_invalid")
        self.assertEqual(caught.exception.failure_domain, "execution_contract")
        self.assertFalse(dsh_native_runtime_retryable(caught.exception))

    def test_provider_errors_propagate_to_recovery_without_fake_keep(self):
        for preceding in ([], [proposal(549)]):
            with self.subTest(after_rejected_response=bool(preceding)):
                error = DshNativeRuntimeUnavailableError(
                    "provider unavailable", failure_domain="provider", status_code=503,
                )
                runtime = Mock()
                runtime.run.side_effect = preceding + [error]
                with self.assertRaises(DshNativeRuntimeUnavailableError) as caught:
                    _validated_local_edit_response(runtime, **self.request)
                self.assertIs(caught.exception, error)
                self.assertTrue(dsh_native_runtime_retryable(error))
                self.assertEqual(runtime.run.call_count, len(preceding) + 1)

    def test_resume_reuses_both_persisted_attempts(self):
        receipts, requests = {}, []

        def run(**request):
            key = request["idempotency_key"]
            if key not in receipts:
                requests.append(deepcopy(request))
                receipts[key] = proposal(549 if request["stage_attempt"] == 1 else 100)
            return deepcopy(receipts[key])

        runtime = Mock(run=run)
        first = _validated_local_edit_response(runtime, **self.request)
        resumed = _validated_local_edit_response(runtime, **self.request)
        self.assertEqual(first, resumed)
        self.assertEqual(len(requests), 2)


if __name__ == "__main__":
    unittest.main()
