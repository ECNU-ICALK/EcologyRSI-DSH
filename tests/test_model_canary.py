from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.integrations.dsh_native_runtime import DshNativeAgentRuntimeClient
from ecologyrsi_dsh.integrations.model_canary import (
    CanaryBounds, ModelCanaryReceiptStore, RECEIPT_SCHEMA, SCOPE,
    canary_request, required_canary_identities, require_model_preflight,
    run_preflight, validate_canary_receipt,
)


def metadata():
    return {"strategy_model_id": "provider/strategy", "review_model_id": "provider/reviewer",
            "preset_content_digest": "a" * 64, "standing_tool_surface_digest": "b" * 64,
            "resolved_policy_route_config_digest": "c" * 64, "resolved_review_route_config_digest": "d" * 64}


def receipt(request, *, passed=True):
    identity = request["identity"]
    now = datetime.now(timezone.utc)
    return {"schema_version": RECEIPT_SCHEMA, "scope": SCOPE, "receipt_id": "canary:test",
            "identity": identity, "identity_digest": digest(identity), "bounds": request["bounds"],
            "started_at": (now - timedelta(seconds=1)).isoformat(), "completed_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=request["bounds"]["ttl_seconds"])).isoformat(),
            "elapsed_ms": 1000, "passed": passed, "output_schema_digest": "e" * 64,
            "result_digest": "f" * 64 if passed else None,
            "route_binding": "role_host_explicit_provider_and_model",
            "tool_evidence": {"first_tool_call_verified": True, "order_verified": True,
                              "successful_call_count": 1, "next_tool_name": "structured_output",
                              "stage": identity["stage"], "source": "dsh_session_event_log"} if passed else None,
            "attempts": [{"attempt": 1, "session_id": "child:1", "elapsed_ms": 1000, "accepted": passed}],
            "failure": None if passed else {"code": "structured_child_output_budget_exhausted", "boundary": "tool_and_schema_contract", "retry": False},
            "usage": {"complete": True, "reported_tokens": 500, "observed_session_count": 1, "launched_session_count": 1, "complete_session_count": 1}}


class ModelCanaryTests(unittest.TestCase):
    def test_identity_derived_from_exact_host_metadata_and_scope(self):
        a, b, critic, planner = required_canary_identities(metadata())
        self.assertEqual(critic["stage"], "sample.critic")
        self.assertEqual((a["role"], a["preset_id"], a["output_schema_id"]), ("researcher", "ecology-researcher-v12", "ecology-research-search-plan@1"))
        self.assertEqual(b["model_id"], "reviewer")
        self.assertNotEqual(a["route_config_digest"], b["route_config_digest"])
        for key in ("strategy_model_id", "preset_content_digest", "resolved_review_route_config_digest"):
            m = metadata(); del m[key]
            with self.assertRaises(ValueError): required_canary_identities(m)
        self.assertEqual(planner["stage"], "sample.plan")
        a["stage"] = "unsupported.stage"
        with self.assertRaisesRegex(ValueError, "unsupported"): canary_request(a)

    def test_bounds_reject_noninteger_and_excess_budget(self):
        for kwargs in ({"max_attempts": 5}, {"max_output_tokens": 8192}, {"max_reported_tokens": 50001}, {"total_timeout_ms": 180001}, {"ttl_seconds": 0}, {"max_attempts": True}):
            with self.assertRaises(ValueError): CanaryBounds(**kwargs)

    def test_receipt_requires_host_evidence_instead_of_self_reported_success(self):
        identity = required_canary_identities(metadata())[0]
        good = receipt(canary_request(identity))
        validate_canary_receipt(good, identity, require_passed=True)
        for mutate in (
            lambda r: r.update(tool_evidence=None),
            lambda r: r["tool_evidence"].update(source="model_self_report"),
            lambda r: r["tool_evidence"].update(next_tool_name="plain_text"),
            lambda r: r["usage"].update(complete=False),
            lambda r: r["usage"].update(reported_tokens=30000),
            lambda r: r.update(identity_digest="a" * 64),
            lambda r: r.update(scope="scientific_qualification"),
            lambda r: r.update(attempts=[]),
        ):
            bad = deepcopy(good); mutate(bad)
            with self.assertRaises(ValueError): validate_canary_receipt(bad, identity, require_passed=True)

    def test_incomplete_billable_usage_is_distinct_from_observed_meter_coverage(self):
        identity = required_canary_identities(metadata())[0]
        value = receipt(canary_request(identity))
        value["usage"].update(complete=False, complete_session_count=0)
        validate_canary_receipt(value, identity, require_passed=True)
        value["usage"].update(observed_session_count=0)
        with self.assertRaises(ValueError): validate_canary_receipt(value, identity, require_passed=True)

    def test_receipt_store_ttl_identity_and_failed_receipts_fail_closed(self):
        identity = required_canary_identities(metadata())[0]
        good = receipt(canary_request(identity))
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelCanaryReceiptStore(tmp)
            path = store.save(good)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(store.require_fresh(identity)["identity_digest"], digest(identity))
            with self.assertRaises(ValueError): store.require_fresh(identity, now=datetime.now(timezone.utc) + timedelta(hours=2))
            changed = dict(identity, model_id="changed")
            with self.assertRaises(FileNotFoundError): store.require_fresh(changed)
            failed = receipt(canary_request(identity), passed=False)
            store.save(failed)
            with self.assertRaisesRegex(ValueError, "not passed"): store.require_fresh(identity)
            future = deepcopy(good)
            future["completed_at"] = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
            with self.assertRaises(ValueError): store.save(future)

    def test_real_client_route_uses_bounded_deadline_and_validates_response_identity(self):
        client = DshNativeAgentRuntimeClient("http://127.0.0.1:8848", token="private-not-printed")
        request = canary_request(required_canary_identities(metadata())[0])
        with patch.object(client, "_request", return_value=receipt(request)) as api:
            self.assertTrue(client.run_canary(request)["passed"])
            args, kwargs = api.call_args
            self.assertEqual(args, ("POST", "/api/ecology-agent-runtime/v1/canaries"))
            self.assertEqual(kwargs["timeout"], 130)
        wrong = receipt(request); wrong["identity"] = dict(request["identity"], model_id="wrong")
        with patch.object(client, "_request", return_value=wrong):
            with self.assertRaises(ValueError): client.run_canary(request)

    def test_preflight_covers_four_roles_caches_and_first_failure_stops(self):
        class Runtime:
            fail = False
            def __init__(self): self.requests = []
            def run_canary(self, request):
                self.requests.append(request)
                return receipt(request, passed=not self.fail)
        client = Runtime()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_preflight(client, metadata=metadata(), receipt_directory=tmp)
            self.assertTrue(result["passed"]); self.assertEqual(len(client.requests), 4)
            self.assertEqual(len(require_model_preflight(metadata(), tmp)), 4)
            run_preflight(client, metadata=metadata(), receipt_directory=tmp)
            self.assertEqual(len(client.requests), 4)
            client.fail = True
            failed = run_preflight(client, metadata=metadata(), receipt_directory=tmp, force=True)
            self.assertFalse(failed["passed"]); self.assertEqual(len(client.requests), 5)
            with self.assertRaises(ValueError): require_model_preflight(metadata(), tmp)
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 4)
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])


class ModelPreflightHTTPTests(unittest.TestCase):
    from tests.test_auto_progress import AutoProgressHTTPTests as _HTTP
    setUp, tearDown, request = _HTTP.setUp, _HTTP.tearDown, _HTTP.request

    def test_timeout_reconciliation_is_read_only_and_requires_all_fresh_receipts(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from ecologyrsi_dsh.api.handler import EvolutionRequestHandler
        self.server.dsh_native_runtime = Mock()
        directory = Path(self.server.ledger.path).parent / "model-preflight"
        store = ModelCanaryReceiptStore(directory)
        identities = required_canary_identities(metadata())
        body = {"execution_protocol": "dsh_native_plugin_evolution@1", "check_only": True}
        with patch.object(EvolutionRequestHandler, "_task_from_request", return_value=SimpleNamespace(metadata=metadata())):
            status, result = self.request("/model-preflight", "POST", body)
            self.assertEqual(status, 200)
            self.assertEqual(result, {"passed": False, "pending": False})
            store.save(receipt(canary_request(identities[0])))
            self.assertFalse(self.request("/model-preflight", "POST", body)[1]["passed"])
            store.save(receipt(canary_request(identities[1])))
            self.assertFalse(self.request("/model-preflight", "POST", body)[1]["passed"])
            store.save(receipt(canary_request(identities[2])))
            self.assertFalse(self.request("/model-preflight", "POST", body)[1]["passed"])
            store.save(receipt(canary_request(identities[3])))
            self.assertTrue(self.request("/model-preflight", "POST", body)[1]["passed"])
            # While a check is in flight, an earlier receipt cannot settle it.
            with self.server.model_preflight_lock:
                status, result = self.request("/model-preflight", "POST", body)
            self.assertEqual((status, result), (202, {"passed": False, "pending": True}))
            store.save(receipt(canary_request(identities[1]), passed=False))
            self.assertFalse(self.request("/model-preflight", "POST", body)[1]["passed"])
        self.server.dsh_native_runtime.run_canary.assert_not_called()

    def test_normal_preflight_reuses_valid_receipts_without_model_calls(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from ecologyrsi_dsh.api.handler import EvolutionRequestHandler
        self.server.dsh_native_runtime = Mock()
        store = ModelCanaryReceiptStore(Path(self.server.ledger.path).parent / "model-preflight")
        for identity in required_canary_identities(metadata()):
            store.save(receipt(canary_request(identity)))
        with patch.object(EvolutionRequestHandler, "_task_from_request", return_value=SimpleNamespace(metadata=metadata())):
            status, result = self.request("/model-preflight", "POST", {"execution_protocol": "dsh_native_plugin_evolution@1"})
        self.assertEqual(status, 200)
        self.assertTrue(result["passed"])
        self.server.dsh_native_runtime.run_canary.assert_not_called()


if __name__ == "__main__": unittest.main()
