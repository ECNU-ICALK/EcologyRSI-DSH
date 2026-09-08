from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.api import handler as handler_module
from ecologyrsi_dsh.api.projection import _monitor_payload, _state_payload
from ecologyrsi_dsh.core.model_preflight import (
    AUDIT_EVENT, AUDIT_METADATA_KEY, AUDIT_SCHEMA,
    build_preflight_audit, model_preflight_projection, validate_preflight_audit,
)
from ecologyrsi_dsh.core.state import RunStateReducer
from ecologyrsi_dsh.integrations.model_canary import canary_request, required_canary_identities, SCOPE
from tests import test_dsh_native_runtime as native_tests
from tests import test_model_canary as canary_tests


def receipts_for(metadata, *_args, **_kwargs):
    result = []
    for index, identity in enumerate(required_canary_identities(metadata)):
        receipt = canary_tests.receipt(canary_request(identity))
        receipt["receipt_id"] = f"canary:role-{index}"
        receipt["raw_prompt"] = "must-never-be-persisted"
        receipt["attempts"][0]["diagnostic"] = "must-never-be-persisted"
        result.append(receipt)
    return tuple(result)


class ModelPreflightAuditTests(unittest.TestCase):
    setUp = native_tests.DshNativeHTTPGateTests.setUp
    tearDown = native_tests.DshNativeHTTPGateTests.tearDown
    _post = native_tests.DshNativeHTTPGateTests._post
    _post_path = native_tests.DshNativeHTTPGateTests._post_path

    def body(self, *, start=False, required=True):
        self.server.dsh_native_runtime = native_tests._FakeNativeRuntime()
        return {"execution_protocol": "dsh_native_plugin_evolution@1",
                "run_id": "run:preflight-audit", "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1", "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review", "start": start, "auto_advance": 0,
                "require_model_contract_preflight": required,
                "idempotency_key": "preflight-audit-create"}

    def create(self, *, start=False, required=True):
        body = self.body(start=start, required=required)
        with patch.object(handler_module, "require_model_preflight", side_effect=receipts_for):
            status, payload = self._post(body)
        self.assertEqual(status, 201, payload)
        return body, self.server.director.state(body["run_id"])

    def test_new_create_records_safe_run_bound_evidence_before_start_and_selected_run_views_agree(self):
        _body, state = self.create(start=True)
        events = list(state.events)
        audit = next(event for event in events if event.kind == AUDIT_EVENT)
        self.assertLess(events.index(audit), next(i for i, event in enumerate(events) if event.kind == "RunStarted"))
        self.assertEqual(state.task_manifest.metadata[AUDIT_METADATA_KEY], AUDIT_SCHEMA)
        self.assertNotIn("receipts", state.task_manifest.metadata)
        self.assertNotIn("must-never-be-persisted", json.dumps(audit.payload))
        evidence = model_preflight_projection(state)
        self.assertEqual((evidence["status"], evidence["scope"]), ("verified", SCOPE))
        self.assertEqual([role["model_id"] for role in evidence["roles"]], ["dsh/strategy", "dsh/review"])
        self.assertEqual(len(evidence["roles"]), 2)
        for projection in (_state_payload(state)["projection"], _monitor_payload(state)["projection"]):
            self.assertEqual(projection["model_contract_preflight"], evidence)

    def test_completed_create_replay_does_not_require_current_receipt_or_duplicate_audit(self):
        body, state = self.create()
        self.server.dsh_native_runtime.unavailable = True
        with patch.object(handler_module, "require_model_preflight", side_effect=AssertionError("fresh gate on idempotent replay")):
            status, payload = self._post(body)
        self.assertEqual(status, 200, payload)
        after = self.server.director.state(body["run_id"])
        self.assertEqual(after.task_manifest.digest, state.task_manifest.digest)
        self.assertEqual(sum(event.kind == AUDIT_EVENT for event in after.events), 1)

    def test_transport_verified_preserves_incomplete_usage_without_claiming_billing_coverage(self):
        body = self.body()

        def incomplete_receipts(metadata, *_args, **_kwargs):
            receipts = receipts_for(metadata)
            for receipt in receipts:
                receipt["usage"].update(complete=False, complete_session_count=0)
            return receipts

        with patch.object(handler_module, "require_model_preflight", side_effect=incomplete_receipts):
            status, payload = self._post(body)
        self.assertEqual(status, 201, payload)
        state = self.server.director.state(body["run_id"])
        audit = next(event for event in state.events if event.kind == AUDIT_EVENT)
        self.assertTrue(all(entry["receipt"]["usage"]["complete"] is False
                            for entry in audit.payload["receipts"]))
        evidence = model_preflight_projection(state)
        self.assertEqual((evidence["status"], evidence["scope"]), ("verified", SCOPE))
        self.assertNotIn("usage_complete", evidence)

    def test_interrupted_creation_recovers_missing_audit_using_fresh_receipts(self):
        body = self.body(start=True)
        director = self.server.director
        with patch.object(handler_module, "require_model_preflight", side_effect=receipts_for), patch.object(
            director, "record_model_contract_preflight", side_effect=RuntimeError("simulated write interruption")
        ):
            status, _payload = self._post(body)
        self.assertEqual(status, 400)
        before = director.state(body["run_id"])
        self.assertEqual(before.run.status.value, "created")
        self.assertEqual(model_preflight_projection(before)["status"], "missing")
        with patch.object(handler_module, "require_model_preflight", side_effect=receipts_for) as gate:
            status, payload = self._post(body)
        self.assertEqual(status, 200, payload)
        gate.assert_called_once()
        after = director.state(body["run_id"])
        self.assertEqual(after.run.status.value, "running")
        self.assertEqual(before.task_manifest.digest, after.task_manifest.digest)
        self.assertEqual(sum(event.kind == AUDIT_EVENT for event in after.events), 1)
        self.assertEqual(len(self.server.dsh_native_runtime.activated), 1)

    def test_seed_materialization_interruption_binds_resource_and_recovers_original_seed(self):
        body = self.body(start=True)
        director = self.server.director
        with patch.object(handler_module, "require_model_preflight", side_effect=receipts_for), patch.object(
            director, "recover_run_initialization", side_effect=RuntimeError("seed materialization interrupted")
        ):
            status, _payload = self._post(body)
        self.assertEqual(status, 400)
        before = director.state(body["run_id"])
        self.assertEqual([event.kind for event in before.events], ["RunCreated"])
        command = self.server.ledger.command_receipt("create:preflight-audit-create")
        self.assertEqual(command.resource_run_id, body["run_id"])
        self.assertEqual(self.server.dsh_native_runtime.cancelled, [])
        with patch.object(handler_module, "require_model_preflight", side_effect=receipts_for):
            status, payload = self._post(body)
        self.assertEqual(status, 200, payload)
        after = director.state(body["run_id"])
        self.assertEqual(after.run.status.value, "running")
        self.assertEqual(before.task_manifest.digest, after.task_manifest.digest)
        self.assertEqual(after.materialized_seed_genome().genome_digest,
                         before.events[0].payload["genome_initialization"]["expected_seed_genome_digest"])
        for kind in ("RunCreated", "RunSeedGenomeMaterialized", AUDIT_EVENT, "RunStarted"):
            self.assertEqual(sum(event.kind == kind for event in after.events), 1)

    def test_interruption_after_audit_recovers_without_rechecking_ttl(self):
        body = self.body(start=True)
        with patch.object(handler_module, "require_model_preflight", side_effect=receipts_for), patch.object(
            self.server.director, "start_run", side_effect=RuntimeError("simulated start interruption")
        ):
            status, _payload = self._post(body)
        self.assertEqual(status, 400)
        with patch.object(handler_module, "require_model_preflight", side_effect=AssertionError("historical audit needs no fresh gate")):
            status, payload = self._post(body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["projection"]["status"], "running")
        self.assertEqual(payload["projection"]["model_contract_preflight"]["status"], "verified")

    def test_missing_audit_control_start_rechecks_before_native_activation(self):
        body = self.body()
        with patch.object(handler_module, "require_model_preflight", side_effect=receipts_for), patch.object(
            self.server.director, "record_model_contract_preflight", side_effect=RuntimeError("write interruption")
        ):
            self._post(body)
        with patch.object(handler_module, "require_model_preflight", side_effect=receipts_for):
            status, payload = self._post_path(f"/runs/{body['run_id']}/action", {
                "action": "start", "idempotency_key": "recover-start",
            })
        self.assertEqual(status, 200, payload)
        state = self.server.director.state(body["run_id"])
        audit = next(event for event in state.events if event.kind == AUDIT_EVENT)
        self.assertEqual(self.server.dsh_native_runtime.activated[0]["run_state_revision"], audit.seq)

    def test_opt_in_missing_audit_blocks_director_and_replayed_start(self):
        body = self.body()
        with patch.object(handler_module, "require_model_preflight", side_effect=receipts_for), patch.object(
            self.server.director, "record_model_contract_preflight", side_effect=RuntimeError("write interruption")
        ):
            self._post(body)
        state = self.server.director.state(body["run_id"])
        with self.assertRaisesRegex(RuntimeError, "preflight audit"):
            self.server.director.start_run(body["run_id"])
        reducer = RunStateReducer(state.events[0])
        reducer.apply(state.events[1:])
        forged = replace(state.events[-1], seq=state.events[-1].seq + 1, event_id="forged:start",
                         kind="RunStarted", payload={"session_id": "session:forged"})
        with self.assertRaisesRegex(ValueError, "preflight audit"):
            reducer.apply((forged,))

    def test_legacy_no_opt_in_remains_startable_without_claiming_verified(self):
        _body, state = self.create(required=False)
        self.server.director.start_run(state.run.run_id)
        after = self.server.director.state(state.run.run_id)
        self.assertNotIn(AUDIT_METADATA_KEY, after.task_manifest.metadata)
        self.assertEqual(model_preflight_projection(after)["status"], "not_required")
        legacy = replace(after.task_manifest, metadata={**dict(after.task_manifest.metadata),
                                                      "require_model_contract_preflight": True})
        legacy_state = replace(after, task_manifest=legacy)
        self.assertEqual(model_preflight_projection(legacy_state)["status"], "missing")
        self.assertFalse(model_preflight_projection(legacy_state)["audit_required"])

    def test_recorded_ttl_uses_historical_time_and_rejects_tampering(self):
        _body, state = self.create()
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        receipts = receipts_for(state.task_manifest.metadata)
        for receipt in receipts:
            receipt.update(started_at=(now - timedelta(seconds=1)).isoformat(), completed_at=now.isoformat(),
                           expires_at=(now + timedelta(hours=1)).isoformat())
        payload = build_preflight_audit(state.task_manifest, state.run.run_id, receipts,
                                        checked_at=(now + timedelta(seconds=1)).isoformat())
        kwargs = {"recorded_at": (now + timedelta(seconds=2)).isoformat(),
                  "created_at": now.isoformat()}
        validate_preflight_audit(payload, state.task_manifest, state.run.run_id, **kwargs)
        with self.assertRaises(ValueError):
            validate_preflight_audit(payload, state.task_manifest, state.run.run_id,
                                     **{**kwargs, "recorded_at": (now + timedelta(hours=2)).isoformat()})
        for mutate in (
            lambda p: p.update(run_id="run:other"),
            lambda p: p.update(task_manifest_digest="0" * 64),
            lambda p: p["receipts"][0].update(receipt_digest="0" * 64),
            lambda p: p["receipts"].reverse(),
            lambda p: p["receipts"][0]["receipt"].update(raw_prompt="forged extra data"),
        ):
            bad = deepcopy(payload); mutate(bad)
            with self.assertRaises(ValueError):
                validate_preflight_audit(bad, state.task_manifest, state.run.run_id, **kwargs)

    def test_reducer_rejects_duplicate_or_post_start_audit(self):
        _body, state = self.create(start=True)
        audit = next(event for event in state.events if event.kind == AUDIT_EVENT)
        reducer = RunStateReducer(state.events[0]); reducer.apply(state.events[1:])
        duplicate = replace(audit, seq=state.events[-1].seq + 1, event_id="duplicate:audit")
        with self.assertRaisesRegex(ValueError, "once before start"):
            reducer.apply((duplicate,))

    def test_unknown_audit_policy_cannot_disable_required_gate(self):
        _body, state = self.create()
        for metadata in (
            {**dict(state.task_manifest.metadata), AUDIT_METADATA_KEY: "unknown/2"},
            {**dict(state.task_manifest.metadata), "require_model_contract_preflight": False},
        ):
            with self.assertRaisesRegex(ValueError, "preflight audit policy"):
                replace(state.task_manifest, metadata=metadata)
