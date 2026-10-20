"""Real native HTTP routes -> Python decoding -> origin admission, no model calls."""
from __future__ import annotations

import json
from pathlib import Path
import selectors
import shutil
import subprocess
import unittest

from ecologyrsi_dsh.core.errors import (
    DshNativeRuntimeUnavailableError, dsh_native_runtime_congestion_signal,
    dsh_native_runtime_evaluation_fatal, dsh_native_runtime_retryable,
)
from ecologyrsi_dsh.execution.sample_admission import RunSampleAdmission
from ecologyrsi_dsh.integrations.dsh_native_runtime import DshNativeAgentRuntimeClient


@unittest.skipUnless(shutil.which('node'), 'Node.js required for native HTTP boundary')
class PersistenceFailureBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        cls.process=subprocess.Popen(
            [shutil.which('node'),str(root/'tests/fixtures/persistence_boundary_server.mjs')],
            cwd=root,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,text=True,
        )
        with selectors.DefaultSelector() as selector:
            selector.register(cls.process.stdout,selectors.EVENT_READ)
            if not selector.select(10):
                cls.tearDownClass()
                raise AssertionError('synthetic native HTTP fixture failed to start')
        line=cls.process.stdout.readline()
        if not line:
            cls.tearDownClass()
            raise AssertionError('synthetic native HTTP fixture exited')
        port=json.loads(line)['port']
        cls.client=DshNativeAgentRuntimeClient(f'http://127.0.0.1:{port}',
                                              token='synthetic-loopback-token',timeout=2)

    @classmethod
    def tearDownClass(cls):
        if cls.process.poll() is None:
            cls.process.terminate()
            try:cls.process.wait(5)
            except subprocess.TimeoutExpired:
                cls.process.kill();cls.process.wait(5)
        cls.process.stdout.close()

    def invoke(self,stage):
        admission=RunSampleAdmission()
        with self.assertRaises(DshNativeRuntimeUnavailableError) as caught:
            with admission.admit('run:boundary',4):
                self.client._request('POST','/api/ecology-agent-runtime/v1/runs/run%3Aboundary/stages',body={
                    'run_id':'run:boundary','run_state_revision':1,'stage_attempt':1,
                    'ledger_expected_revision':1,'idempotency_key':'synthetic-boundary',
                    'stage':stage,'admission_id':'synthetic-admission','request':{},
                })
        return caught.exception,admission

    def test_deterministic_host_rejections_remain_contract_failures(self):
        cases={'sidecar_rejected':'host_rejected',
               'dsh_tool_authorization_failed':'authorization',
               'dsh_prediction_binding_closed':'binding_closed',
               'dsh_tool_admission_closed':'admission_closed',
               'dsh_skill_evidence_invalid':'tool_evidence',
               'invalid_response':'invalid_response'}
        for cause,suffix in cases.items():
            with self.subTest(cause=cause):
                error,admission=self.invoke(cause)
                self.assertEqual(error.error_code,'structured_result_persist_'+suffix)
                self.assertEqual(error.status_code,422)
                self.assertEqual(error.failure_domain,'execution_contract')
                self.assertIsNone(error.provider_status)
                self.assertFalse(dsh_native_runtime_retryable(error))
                self.assertTrue(dsh_native_runtime_evaluation_fatal(error))
                self.assertFalse(dsh_native_runtime_congestion_signal(error))
                self.assertIs(admission.execution_failure_for('run:boundary'),error)
                self.assertEqual(admission.snapshot('run:boundary')['adaptive_limit'],4)
                self.assertEqual(admission.snapshot('run:boundary')['congestion_events'],0)
                # No silent sample failure: further admission is closed by this fault.
                with self.assertRaises(DshNativeRuntimeUnavailableError):
                    with admission.admit('run:boundary',4):self.fail('contract failure swallowed')

    def test_transient_host_faults_retry_without_claiming_provider_congestion(self):
        for cause,suffix in {'dsh_session_projection_not_ready':'projection_lag',
                             'structured_result_persistence_unavailable':'host_unavailable',
                             'unavailable':'transport','timeout':'timeout'}.items():
            with self.subTest(cause=cause):
                error,admission=self.invoke(cause)
                self.assertEqual(error.error_code,'structured_result_persist_'+suffix)
                self.assertEqual(error.status_code,502)
                self.assertIsNone(error.failure_domain)
                self.assertIsNone(error.provider_status)
                self.assertTrue(dsh_native_runtime_retryable(error))
                self.assertFalse(dsh_native_runtime_evaluation_fatal(error))
                self.assertFalse(dsh_native_runtime_congestion_signal(error))
                self.assertIsNone(admission.execution_failure_for('run:boundary'))
                self.assertEqual(admission.snapshot('run:boundary')['adaptive_limit'],4)
                self.assertEqual(admission.snapshot('run:boundary')['congestion_events'],0)

    def test_real_provider_status_still_reduces_admission(self):
        for status in (408,425,429,502,503,504):
            with self.subTest(status=status):
                error,admission=self.invoke('provider:'+str(status))
                self.assertEqual(error.provider_status,status)
                self.assertEqual(error.failure_domain,'provider')
                self.assertTrue(dsh_native_runtime_retryable(error))
                self.assertTrue(dsh_native_runtime_congestion_signal(error))
                self.assertEqual(admission.snapshot('run:boundary')['adaptive_limit'],2)
                self.assertEqual(admission.snapshot('run:boundary')['congestion_events'],1)

    def test_untrusted_failure_cannot_forge_contract_or_provider_evidence(self):
        error,admission=self.invoke('untrusted')
        self.assertEqual(error.error_code,'dsh_native_runtime_unavailable')
        self.assertEqual(error.status_code,502)
        self.assertIsNone(error.failure_domain)
        self.assertTrue(dsh_native_runtime_retryable(error))
        self.assertFalse(dsh_native_runtime_congestion_signal(error))
        self.assertEqual(admission.snapshot('run:boundary')['adaptive_limit'],4)
        self.assertNotIn('private',str(error))

    def test_known_contract_code_takes_precedence_over_legacy_503(self):
        for code in ('structured_result_persist_host_rejected','structured_result_persist_tool_evidence'):
            error=DshNativeRuntimeUnavailableError(error_code=code,status_code=503)
            self.assertFalse(dsh_native_runtime_retryable(error))
            self.assertTrue(dsh_native_runtime_evaluation_fatal(error))
            self.assertFalse(dsh_native_runtime_congestion_signal(error))

    def test_missing_provider_metadata_never_proves_capacity(self):
        for domain,status in ((None,None),('provider',None),(None,503)):
            error=DshNativeRuntimeUnavailableError(error_code='structured_child_model_error',
                status_code=503,failure_domain=domain,provider_status=status)
            self.assertTrue(dsh_native_runtime_retryable(error))
            self.assertFalse(dsh_native_runtime_congestion_signal(error))

    def test_explicit_contract_domain_blocks_retry_even_with_generic_503(self):
        error=DshNativeRuntimeUnavailableError(error_code='unknown_contract',
            status_code=503,failure_domain='execution_contract')
        self.assertFalse(dsh_native_runtime_retryable(error))
        self.assertTrue(dsh_native_runtime_evaluation_fatal(error))
        self.assertFalse(dsh_native_runtime_congestion_signal(error))
