from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError, dsh_native_runtime_retryable
from ecologyrsi_dsh.core.model_execution_policy import RESEARCH_EXECUTION_POLICY
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ecologyrsi_dsh.api.auto_progress import _progress_failure_irrecoverable, _progress_failure_retryable


class StructuredStageBudgetTests(unittest.TestCase):
    def request(self, stage, context, max_tokens):
        requests = []
        native = SimpleNamespace(run_stage=lambda value: requests.append(value) or {
            "structured": {"accepted": True}, "result_digest": digest({"accepted": True}),
        })
        runtime = DshStructuredRoleRuntime(native)
        result = runtime.run(run_id="run:budget", stage=stage, role="researcher",
            context=context, output_schema_id="ecology-research-synthesis@1",
            run_state_revision=1, stage_attempt=1, ledger_expected_revision=1,
            idempotency_key="research:budget", max_tokens=max_tokens)
        return result, requests[0]["request"]

    def test_new_synthesis_budget_is_explicit_and_covered_by_context_digest(self):
        context = {"research_execution_policy": dict(RESEARCH_EXECUTION_POLICY)}
        result, request = self.request("generation.research-synthesis", context, 16384)
        self.assertTrue(result["accepted"])
        self.assertEqual(request["max_tokens"], 16384)
        self.assertEqual(request["context_digest"], digest(context))
        altered = {"research_execution_policy": {**RESEARCH_EXECUTION_POLICY, "synthesis_max_output_tokens": 8192}}
        self.assertNotEqual(request["context_digest"], digest(altered))

    def test_extended_budget_requires_exact_policy_stage_and_value(self):
        policy = {"research_execution_policy": dict(RESEARCH_EXECUTION_POLICY)}
        for stage, context, tokens in [
            ("generation.research-synthesis", {}, 16384),
            ("generation.search-plan", policy, 16384),
            ("sample.plan", policy, 16384),
            ("generation.research-synthesis", policy, None),
            ("generation.research-synthesis", policy, 8192),
            ("generation.research-synthesis", policy, 16385),
            ("generation.research-synthesis", policy, True),
            ("generation.research-synthesis", {"research_execution_policy": {}}, 16384),
        ]:
            with self.subTest(stage=stage, tokens=tokens):
                with self.assertRaises(ValueError): self.request(stage, context, tokens)
        _result, request = self.request("generation.research-synthesis", {}, 8192)
        self.assertEqual(request["max_tokens"], 8192)

    def test_fatal_cause_wins_over_wrapper_and_cancelled_sibling(self):
        from ecologyrsi_dsh.core.errors import dsh_native_runtime_error_in_chain, preferred_execution_failure
        from ecologyrsi_dsh.api.auto_progress import _retry_later_error
        fatal = DshNativeRuntimeUnavailableError(error_code="structured_child_tool_protocol_error", status_code=422)
        outage = DshNativeRuntimeUnavailableError(status_code=502)
        outage.__cause__ = fatal
        self.assertIs(dsh_native_runtime_error_in_chain(outage), fatal)
        self.assertFalse(_progress_failure_retryable(outage))
        self.assertTrue(_progress_failure_irrecoverable(outage))
        self.assertIsNone(_retry_later_error(outage))
        earlier = DshNativeRuntimeUnavailableError(status_code=503)
        self.assertIs(preferred_execution_failure(earlier, fatal), fatal)
        self.assertIs(preferred_execution_failure(fatal, earlier), fatal)
        self.assertIs(preferred_execution_failure(earlier, RuntimeError()), earlier)

    def test_deterministic_failures_override_http_503_retry_classification(self):
        for code in ("structured_child_output_budget_exhausted", "structured_child_tool_protocol_error",
                     "structured_child_output_schema_invalid", "dsh_native_runtime_contract_error"):
            with self.subTest(code=code):
                error = DshNativeRuntimeUnavailableError(error_code=code, status_code=503)
                self.assertFalse(dsh_native_runtime_retryable(error))
                wrapped = RuntimeError("research stage failed")
                wrapped.__cause__ = error
                self.assertFalse(_progress_failure_retryable(wrapped, stage="research"))
                self.assertTrue(_progress_failure_irrecoverable(wrapped, stage="research"))
        self.assertTrue(dsh_native_runtime_retryable(DshNativeRuntimeUnavailableError(
            error_code="dsh_native_runtime_transport_error", status_code=503)))
        self.assertTrue(dsh_native_runtime_retryable(DshNativeRuntimeUnavailableError(status_code=429)))


if __name__ == "__main__": unittest.main()
