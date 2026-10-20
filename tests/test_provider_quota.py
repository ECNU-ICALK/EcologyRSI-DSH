"""Upstream quota rejection must not become failed predictions or retry bursts."""
import unittest

from ecologyrsi_dsh.api.auto_progress import _retry_later_error
from ecologyrsi_dsh.core.errors import (
    DshNativeRuntimeUnavailableError, dsh_native_runtime_evaluation_fatal,
    dsh_native_runtime_retryable,
)
from ecologyrsi_dsh.core.redaction import public_exception_summary
from ecologyrsi_dsh.core.runtime_failure import validate_runtime_failure
from tests.test_dsh_sample_execution import _SampleRuntime
from tests.test_native_output_repair import execute

CODE = "model_provider_quota_exhausted"


class QuotaPlanner(_SampleRuntime):
    def run_stage(self, request):
        self.requests.append(request)
        raise DshNativeRuntimeUnavailableError(
            "private provider payload", error_code=CODE,
            status_code=429, failure_domain="provider", provider_status=429,
        )


class ProviderQuotaTests(unittest.TestCase):
    def test_quota_propagates_before_publishing_failed_scoring_rows(self):
        runtime, publications = QuotaPlanner(), []
        with self.assertRaises(DshNativeRuntimeUnavailableError) as caught:
            execute(runtime, max_attempts=3, publications=publications)
        error = caught.exception
        self.assertEqual(error.error_code, CODE)
        self.assertEqual(len(runtime.requests), 1)
        self.assertEqual(publications, [])
        self.assertTrue(dsh_native_runtime_evaluation_fatal(error))
        self.assertFalse(dsh_native_runtime_retryable(error))
        self.assertIsNone(_retry_later_error(error))
        self.assertIn("不是本地 Token 预算限制", public_exception_summary(error))
        self.assertNotIn("配额已耗尽", public_exception_summary(error))
        self.assertNotIn("private", public_exception_summary(error))

    def test_quota_metadata_is_the_only_non_retryable_provider_exception(self):
        failure = dict(schema_version="ecology-runtime-failure/1", error_code=CODE,
                       failure_domain="provider", retryable=False, provider_status=429,
                       retry_after_ms=None, affected_scope="model_route")
        self.assertEqual(validate_runtime_failure(failure, error_code=CODE), failure)
        for change in ({"retryable": True}, {"provider_status": 503},
                       {"retry_after_ms": 1000}, {"failure_domain": "execution_contract"},
                       {"error_code": "structured_child_model_error"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                value = {**failure, **change}
                validate_runtime_failure(value, error_code=value["error_code"])

    def test_request_throttling_remains_retryable(self):
        error = DshNativeRuntimeUnavailableError(
            "rate limited", error_code="structured_child_model_error",
            status_code=429, failure_domain="provider", provider_status=429,
        )
        self.assertTrue(dsh_native_runtime_retryable(error))
        self.assertFalse(dsh_native_runtime_evaluation_fatal(error))
