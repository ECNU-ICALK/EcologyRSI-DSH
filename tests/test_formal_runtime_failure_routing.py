"""Fatal sample failures must drain the owning runtime generation."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ecologyrsi_dsh.application.runtime import ApplicationRuntime
from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
from ecologyrsi_dsh.core.ledger import EventLedger


class FormalRuntimeFailureRoutingTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            "ECOLOGYRSI_DSH_MODELS_JSON": "[]",
            "ECOLOGYRSI_DSH_RUNTIME_URL": "",
            "ECOLOGYRSI_DSH_RUNTIME_TOKEN": "",
        })
        environment.start()
        self.addCleanup(environment.stop)
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        self.runtime = ApplicationRuntime(ledger)
        self.runtime.auto_progress = SimpleNamespace(abort_native_evaluation=Mock())
        self.runtime.dsh_tools.close_run_admissions = Mock()

    def fail_sample(self):
        failure = DshNativeRuntimeUnavailableError(
            error_code="structured_child_tool_protocol_error", status_code=422)
        with self.assertRaises(DshNativeRuntimeUnavailableError) as caught:
            with self.runtime.sample_admission.admit("run:failure", 2):
                raise failure
        self.assertIs(caught.exception, failure)
        self.runtime.dsh_tools.close_run_admissions.assert_called_once_with("run:failure")
        return failure

    def test_formal_failure_uses_formal_cleanup_without_training_control(self):
        abort = Mock(return_value=True)
        self.runtime.independent_evaluation = SimpleNamespace(abort_native_evaluation=abort)
        self.fail_sample()
        abort.assert_called_once_with("run:failure")
        self.runtime.auto_progress.abort_native_evaluation.assert_not_called()

    def test_training_failure_still_uses_its_owner(self):
        abort = Mock(return_value=False)
        self.runtime.independent_evaluation = SimpleNamespace(abort_native_evaluation=abort)
        failure = self.fail_sample()
        abort.assert_called_once_with("run:failure")
        self.runtime.auto_progress.abort_native_evaluation.assert_called_once_with(
            "run:failure", error=failure)

    def test_runtime_without_formal_service_preserves_training_cleanup(self):
        failure = self.fail_sample()
        self.runtime.auto_progress.abort_native_evaluation.assert_called_once_with(
            "run:failure", error=failure)

    def test_formal_cleanup_failure_never_falls_through_to_training_control(self):
        abort = Mock(side_effect=RuntimeError("formal cleanup unavailable"))
        self.runtime.independent_evaluation = SimpleNamespace(abort_native_evaluation=abort)
        self.fail_sample()
        abort.assert_called_once_with("run:failure")
        self.runtime.auto_progress.abort_native_evaluation.assert_not_called()
