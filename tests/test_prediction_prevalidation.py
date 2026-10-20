"""Host semantic validation before native capture: no science or budget writes."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecologyrsi_dsh.api.handler import EvolutionHTTPServer
from ecologyrsi_dsh.core.agent_prediction import PredictionContractViolation, validate_predictions
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import canonical_json, digest
from ecologyrsi_dsh.integrations.dsh_tools import (
    DshPredictionBindingClosedError, DshToolAdmissionClosedError,
    DshToolAuthorizationError, DshToolOperationalTimeoutError, DshToolService,
    DshStructuredResultPersistenceError,
)
from tests.test_dsh_tool_contracts import _identity, _skill_evidence


@contextmanager
def fixture(*, service=None, allowed_methods=None, mean=False):
    owned = service is None
    ledger = EventLedger(":memory:") if owned else service.ledger
    if owned:
        service = DshToolService(ledger)
    clock = [100.0]
    service._monotonic_ms = lambda: clock[0]
    ledger.append("run:tool-test", "RunCreated", {"test": True})
    identity = _identity(ledger)
    fence = service.open_admission("run:tool-test", 3, 2, role="sample-planner",
                                   stage="sample.plan", idempotency_key=identity["idempotency_key"])
    reservation = service.allocate_child_reservation({
        "request_id": "prevalidation-test", "run_id": identity["run_id"], "parent_session_id": "parent",
        "role": "sample-planner", "stage": "sample.plan", "run_state_revision": 3, "stage_attempt": 2,
        "admission_id": fence.admission_id, "timeout_ms": 1000, "item_digest": "d" * 64,
        "idempotency_key": identity["idempotency_key"],
    })
    identity["child_reservation_id"] = reservation["launch"]["reservation_id"]
    identity["ledger_expected_revision"] = ledger.latest_seq()
    ids = [f"cell-{i}" for i in range(9)]
    executions = []
    activations = {identity["session_id"]: deepcopy(identity)}
    attestation_calls = []
    def attest(supplied, *, parent_session_id, timeout):
        attestation_calls.append({"identity": deepcopy(supplied), "parent": parent_session_id, "timeout": timeout})
        expected = activations.get(supplied["session_id"])
        return parent_session_id == "parent" and expected is not None and all(
            supplied[name] == value for name, value in expected.items() if name != "ledger_expected_revision")
    previous_attestor = service._activation_attestor
    service._activation_attestor = attest
    def activate(session_id):
        new_identity = {**identity, "session_id": session_id, "activation_lease_id": "lease:" + session_id}
        request = {"request_id": "prevalidation:" + session_id, "run_id": identity["run_id"], "parent_session_id": "parent",
            "role": "sample-planner", "stage": "sample.plan", "run_state_revision": 3, "stage_attempt": 2,
            "admission_id": fence.admission_id, "timeout_ms": 1000, "item_digest": "d" * 64,
            "idempotency_key": identity["idempotency_key"]}
        reservation = service.allocate_child_reservation(request)
        new_identity["child_reservation_id"] = reservation["launch"]["reservation_id"]
        activations[session_id] = new_identity
        return deepcopy(new_identity)
    def executor(tool_id, _parameters):
        executions.append(tool_id)
        if tool_id == "failed":
            raise ValueError("synthetic capability failure")
        return {sample: {"predicted": 20.0 if tool_id == "first" else 22.0} for sample in ids}
    with service.bind_prediction_tool(run_id=identity["run_id"], stage_attempt=2,
            idempotency_key=identity["idempotency_key"], wave_digest="a" * 64, sample_ids=ids,
            catalog=[{"tool_id": name, "version": "1"} for name in ("first", "second", "failed")],
            executor=executor, prediction_tool_call_budget=4, allowed_methods=allowed_methods,
            identity_digests={name: identity[name] for name in ("genome_digest", "compiled_behavior_digest", "phenotype_instance_digest")},
            mean_blend_required=mean) as binding:
        def envelope(method="direct", refs=(), predicted=21.0):
            structured = {"schema_version": "ecology-sample-predictions@2", "wave_digest": "a" * 64,
                          "decisions": [{"sample_id": sample, "predicted": predicted, "confidence": .9,
                            "reason_code": "agent_" + method, "method": method, "evidence_call_ids": list(refs)} for sample in ids]}
            return {"identity": deepcopy(identity), "output_schema_id": "ecology-sample-predictions@2",
                    "structured": structured, "result_digest": digest(structured), "admission_id": fence.admission_id}
        def call(call_id, tool_id="first", session_id=None):
            return service.execute("ecology_execute_prediction_tool", {"identity":
                deepcopy(activations[session_id or identity["session_id"]]), "arguments": {
                "tool_id": tool_id, "call_id": call_id, "parameters": {}, "wave_digest": "a" * 64}})
        yield SimpleNamespace(service=service, ledger=ledger, binding=binding, identity=identity,
              fence=fence, clock=clock, ids=ids, executions=executions, envelope=envelope, call=call,
              activate=activate, activations=activations, attestation_calls=attestation_calls)
    service._activation_attestor = previous_attestor
    if owned:
        ledger.close()


def snapshot(f):
    return (f.ledger.latest_seq(), deepcopy(f.binding.calls), deepcopy(f.binding._predictions),
            deepcopy(f.binding._session_calls), f.binding._spent(), list(f.executions),
            f.fence.state, f.fence.deadline_monotonic_ms)


class PredictionPrevalidationTests(unittest.TestCase):
    def test_actual_S_failure_shape_reports_all_nine_and_does_not_mutate(self):
        with fixture() as f:
            f.call("cm-1")
            envelope = f.envelope("blend", ["cm-1"])
            before = snapshot(f)
            for _ in range(3):
                result = f.service.validate_structured(envelope)
                self.assertFalse(result["valid"])
                self.assertEqual(result["result_digest"], envelope["result_digest"])
                self.assertEqual(result["diagnostics"], [{"code": "blend_requires_two_results",
                    "path": f"value.decisions[{i}].evidence_call_ids"} for i in range(9)])
                self.assertEqual(snapshot(f), before)
            with self.assertRaisesRegex(PredictionContractViolation, "at least two"):
                f.binding.final_receipt(envelope["structured"], session_id=f.identity["session_id"])
            # Model-authored second output is tested independently, never a Host rewrite.
            corrected = f.envelope("adjusted", ["cm-1"])
            self.assertTrue(f.service.validate_structured(corrected)["valid"])
            self.assertEqual(snapshot(f), before)
            accepted = f.service.accept_structured({**corrected, "skill_invocation_evidence": _skill_evidence("sample.plan")})
            self.assertTrue(accepted["accepted"])
            committed = f.ledger.latest_seq()
            self.assertEqual(f.service.accept_structured({**corrected, "skill_invocation_evidence": _skill_evidence("sample.plan")}), accepted)
            self.assertEqual(f.ledger.latest_seq(), committed)
            self.assertEqual(f.binding._spent(), 1)

    def test_visible_successful_receipts_are_session_scoped(self):
        with fixture() as f:
            f.call("one"); f.call("bad", "failed")
            second_identity = f.activate("other-session")
            before = snapshot(f)
            for refs, session in ((["not-returned"], None), (["bad"], None), (["one"], "other-session")):
                envelope = f.envelope("model", refs)
                if session: envelope["identity"] = deepcopy(second_identity)
                result = f.service.validate_structured(envelope)
                self.assertFalse(result["valid"])
                self.assertTrue(all(x["code"] == "evidence_unavailable" for x in result["diagnostics"]))
                self.assertEqual(snapshot(f), before)
            # A real idempotent reacquire makes exactly that receipt visible to B.
            f.call("one", session_id="other-session")
            envelope = f.envelope("model", ["one"]); envelope["identity"] = deepcopy(second_identity)
            self.assertTrue(f.service.validate_structured(envelope)["valid"])
            self.assertEqual(f.binding._spent(), 2)
            self.assertEqual(f.executions, ["first", "failed"])

    def test_frozen_methods_and_mean_formula_use_shared_final_validator(self):
        methods = ["direct", "model"]
        with fixture(allowed_methods=methods) as f:
            methods.append("adjusted")
            f.call("one")
            envelope = f.envelope("adjusted", ["one"])
            self.assertFalse(f.service.validate_structured(envelope)["valid"])
            self.assertEqual(f.binding.allowed_methods, frozenset({"direct", "model"}))
            with self.assertRaisesRegex(PredictionContractViolation, "candidate's directive"):
                f.service.accept_structured({**envelope, "skill_invocation_evidence": _skill_evidence("sample.plan")})
        with fixture(mean=True) as f:
            f.call("one"); f.call("two", "second")
            before = snapshot(f)
            invalid = f.service.validate_structured(f.envelope("blend", ["one", "two"], predicted=23.0))
            self.assertEqual([x["code"] for x in invalid["diagnostics"]], ["mean_blend_mismatch"] * 9)
            self.assertTrue(f.service.validate_structured(f.envelope("blend", ["one", "two"]))["valid"])
            self.assertEqual(snapshot(f), before)

    def test_shape_errors_are_value_free_bounded_diagnostics(self):
        with fixture() as f:
            base = f.envelope()["structured"]
            for field, value in (("method", []), ("reason_code", {}), ("confidence", True),
                                 ("predicted", None), ("evidence_call_ids", ["one", "one"])):
                malformed = deepcopy(base); malformed["decisions"][0][field] = value
                with self.subTest(field=field), self.assertRaises(PredictionContractViolation) as caught:
                    validate_predictions(malformed, f.ids, wave_digest="a" * 64)
                self.assertEqual(caught.exception.diagnostics[0]["path"], f"value.decisions[0].{field}")
            for value in (float('inf'), float('nan'), 10 ** 400):
                malformed = deepcopy(base); malformed["decisions"][0]["predicted"] = value
                with self.assertRaises(PredictionContractViolation):
                    validate_predictions(malformed, f.ids, wave_digest="a" * 64)
            malicious = f.envelope(); malicious["structured"]["decisions"][0]["observed_secret"] = "do-not-echo-private-value"
            malicious["result_digest"] = digest(malicious["structured"])
            result = f.service.validate_structured(malicious)
            self.assertFalse(result["valid"])
            self.assertNotIn("do-not-echo", json.dumps(result))
            self.assertLessEqual(len(canonical_json(result).encode()), 4096)

    def test_identity_wave_reservation_and_admission_faults_are_not_invalid_args(self):
        with fixture() as f:
            for field, value in (("run_id", "unknown"), ("stage", "sample.critic"),
                ("child_reservation_id", "wrong-reservation"), ("ledger_expected_revision", 10000),
                ("idempotency_key", "foreign-wave"), ("session_id", "forged-session"),
                ("activation_lease_id", "forged-lease"), ("genome_digest", "e" * 64),
                ("compiled_behavior_digest", "e" * 64), ("phenotype_instance_digest", "e" * 64)):
                envelope = f.envelope(); envelope["identity"][field] = value
                before = snapshot(f)
                with self.subTest(field=field), self.assertRaises((DshToolAuthorizationError, DshToolAdmissionClosedError)):
                    f.service.validate_structured(envelope)
                self.assertEqual(snapshot(f), before)
            for field, value in (("admission_id", "other"), ("result_digest", "b" * 64)):
                envelope = f.envelope(); envelope[field] = value
                with self.assertRaises(DshToolAuthorizationError): f.service.validate_structured(envelope)
            envelope = f.envelope(); envelope["structured"]["wave_digest"] = "b" * 64
            envelope["result_digest"] = digest(envelope["structured"])
            with self.assertRaises(DshToolAuthorizationError): f.service.validate_structured(envelope)

    def test_close_timeout_and_internal_fault_keep_operational_type(self):
        with fixture() as f:
            with patch.object(f.binding, "final_receipt", side_effect=RuntimeError("private-internal-detail")):
                with self.assertRaisesRegex(RuntimeError, "private-internal-detail"):
                    f.service.validate_structured(f.envelope())
            f.clock[0] = f.fence.deadline_monotonic_ms
            with self.assertRaises(DshToolOperationalTimeoutError): f.service.validate_structured(f.envelope())
            f.clock[0] = 100
            f.service.close_admission("run:tool-test", 3, 2)
            with self.assertRaises(DshToolAdmissionClosedError): f.service.validate_structured(f.envelope())

    def test_races_are_rechecked_and_validation_is_not_persistence_authority(self):
        for change in ("close", "deadline", "binding"):
            with self.subTest(change=change), fixture() as f:
                original = f.binding.final_receipt
                def race(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if change == "close": f.service.close_admission("run:tool-test", 3, 2)
                    elif change == "deadline": f.clock[0] = f.fence.deadline_monotonic_ms
                    else: f.service._prediction_bindings.clear()
                    return result
                with patch.object(f.binding, "final_receipt", side_effect=race):
                    with self.assertRaises((DshToolAdmissionClosedError, DshToolOperationalTimeoutError)):
                        f.service.validate_structured(f.envelope())
        with fixture() as f:
            self.assertTrue(f.service.validate_structured(f.envelope())["valid"])
            envelope = f.envelope("blend", ["fake"])
            with self.assertRaises(PredictionContractViolation):
                f.service.accept_structured({**envelope, "skill_invocation_evidence": _skill_evidence("sample.plan")})
            f.service.close_admission("run:tool-test", 3, 2)
            with self.assertRaises(DshToolAdmissionClosedError):
                f.service.accept_structured({**f.envelope(), "skill_invocation_evidence": _skill_evidence("sample.plan")})

    def test_authority_is_required_again_at_accept_and_exact_replay_is_independent(self):
        with fixture() as f:
            envelope = {**f.envelope(), "skill_invocation_evidence": _skill_evidence("sample.plan")}
            self.assertTrue(f.service.validate_structured(f.envelope())["valid"])
            original = f.activations.pop(f.identity["session_id"])
            before = snapshot(f)
            with self.assertRaises(DshToolAuthorizationError): f.service.accept_structured(envelope)
            self.assertEqual(snapshot(f), before)
            f.activations[f.identity["session_id"]] = original
            receipt = f.service.accept_structured(envelope)
            calls = len(f.attestation_calls)
            f.activations.clear(); f.service._prediction_bindings.clear()
            f.service.close_admission("run:tool-test", 3, 2)
            self.assertEqual(f.service.accept_structured(envelope), receipt)
            self.assertEqual(len(f.attestation_calls), calls)
            forged = deepcopy(envelope); forged["identity"]["session_id"] = "other"
            with self.assertRaisesRegex(ValueError, "idempotency"): f.service.accept_structured(forged)
        with fixture() as f:
            before = snapshot(f)
            f.service._activation_attestor = None
            with self.assertRaises(DshStructuredResultPersistenceError): f.service.validate_structured(f.envelope())
            self.assertEqual(snapshot(f), before)

    def test_attestation_uses_remaining_deadline_and_rechecks_after_network(self):
        from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
        with fixture() as f:
            before = snapshot(f)
            for error in (DshNativeRuntimeUnavailableError("unavailable", error_code="dsh_native_runtime_transport_error"),):
                with patch.object(f.service, "_activation_attestor", side_effect=error):
                    with self.assertRaises(DshStructuredResultPersistenceError): f.service.validate_structured(f.envelope())
                self.assertEqual(snapshot(f), before)
            with patch.object(f.service, "_activation_attestor", return_value="true"):
                with self.assertRaisesRegex(RuntimeError, "invalid native"): f.service.validate_structured(f.envelope())
            original = f.service._activation_attestor
            def expired(*args, **kwargs):
                self.assertLessEqual(kwargs["timeout"], .1)
                f.clock[0] = f.fence.deadline_monotonic_ms
                return original(*args, **kwargs)
            f.clock[0] = f.fence.deadline_monotonic_ms - 100
            with patch.object(f.service, "_activation_attestor", side_effect=expired):
                with self.assertRaises(DshToolOperationalTimeoutError): f.service.validate_structured(f.envelope())

    def test_native_attestation_client_preserves_transport_bound_and_is_readonly(self):
        from ecologyrsi_dsh.integrations.dsh_native_runtime import DshNativeAgentRuntimeClient
        from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
        client = DshNativeAgentRuntimeClient("http://127.0.0.1:1", token="synthetic-control", timeout=17)
        response = {"schema_version": "ecologyrsi-dsh.child-activation-attestation/1", "valid": True}
        with patch.object(client, "_request", return_value=response) as request:
            for remaining, expected in ((.05, .05), (300, 17)):
                self.assertTrue(client.attest_child_activation({"session_id": "synthetic"}, parent_session_id="parent", timeout=remaining))
                self.assertEqual(request.call_args.kwargs["timeout"], expected)
                self.assertEqual(request.call_args.args, ("POST", "/api/ecology-agent-runtime/v1/child-activations/attest"))
            self.assertEqual(request.call_count, 2)
        for malformed in ({**response, "valid": "true"}, {**response, "extra": True}, {"valid": True}):
            with patch.object(client, "_request", return_value=malformed):
                with self.assertRaises(DshNativeRuntimeUnavailableError):
                    client.attest_child_activation({}, parent_session_id="parent", timeout=1)

    def test_authenticated_http_is_readonly_for_valid_invalid_and_faults(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ECOLOGYRSI_SIDECAR_TOOL_TOKEN": "test-only-secret"}):
            server = EvolutionHTTPServer(("127.0.0.1", 0), Path(directory)/"events.sqlite3")
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            def post(envelope, token="test-only-secret"):
                request = Request(f"http://127.0.0.1:{server.server_address[1]}/api/ecology-agent-sidecar/v1/structured-results/validate",
                    data=json.dumps(envelope).encode(), headers={"Authorization":"Bearer "+token,"Content-Type":"application/json"})
                try:
                    with urlopen(request, timeout=3) as response: return response.status, json.load(response)
                except HTTPError as error: return error.code, json.load(error)
            try:
                with fixture(service=server.dsh_tools) as f:
                    before = snapshot(f)
                    self.assertEqual(post(f.envelope())[0], 200)
                    status, result = post(f.envelope("blend", ["missing"]))
                    self.assertEqual(status, 200); self.assertFalse(result["valid"])
                    self.assertEqual(post(f.envelope(), token="wrong")[0], 401)
                    for key, value in (("session_id", "forged"), ("activation_lease_id", "forged"),
                        ("genome_digest", "e" * 64), ("compiled_behavior_digest", "e" * 64),
                        ("phenotype_instance_digest", "e" * 64)):
                        forged = f.envelope(); forged["identity"][key] = value
                        self.assertEqual(post(forged)[0], 403)
                    with patch.object(f.service, "_activation_attestor", side_effect=DshStructuredResultPersistenceError("private")):
                        status, result = post(f.envelope()); self.assertEqual(status, 503)
                        self.assertEqual(result["error_code"], "structured_result_persistence_unavailable")
                        self.assertNotIn("private", json.dumps(result))
                    wrong = f.envelope(); wrong["admission_id"] = "wrong"
                    self.assertEqual(post(wrong)[0], 403)
                    with patch.object(f.binding, "final_receipt", side_effect=RuntimeError("private-internal-detail")):
                        status, result = post(f.envelope())
                        self.assertEqual(status, 409); self.assertNotIn("private", json.dumps(result))
                    self.assertEqual(snapshot(f), before)
                    self.assertFalse(list(Path(directory).rglob("*rejections*")))
            finally:
                server.shutdown(); thread.join(2); server.close()


if __name__ == "__main__":
    unittest.main()
