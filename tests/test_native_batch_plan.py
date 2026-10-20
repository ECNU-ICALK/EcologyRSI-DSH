"""Native Host planning faults must not manufacture scientific observations."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.models import canonical_json, digest
from ecologyrsi_dsh.evaluators.sample_execution import (
    CollaborativeSampleExecutor, SampleExecutionContractError, SampleExecutionPolicy,
)
from tests import test_agent_owned_prediction as agent_fixture
from tests.test_dsh_sample_execution import _request


class NativeBatchPlanTests(unittest.TestCase):
    def fixture(self):
        fixture = agent_fixture.AgentOwnedPredictionTests()
        self.addCleanup(fixture.doCleanups)
        adapter, plan, native, _ = fixture.setup_agent(lambda context, call: agent_fixture.result(context))
        return fixture, adapter, plan, native

    def test_large_catalog_plan_keeps_full_tool_contract_at_actual_prediction_boundary(self):
        fixture, adapter, original, native = self.fixture()
        adapter._agent_tool_catalog = [
            {"tool_id": f"registered-tool-{i}", "version": "1", "purpose": "p" * 900,
             "parameters": {"scale": {"type": "number", "default": 0.1, "minimum": 0, "maximum": 1}}}
            for i in range(12)
        ]
        context = deepcopy(original["decision_context"])
        context["tool_experience"] = [{"summary": "s" * 1000} for _ in range(6)]
        before = deepcopy(context)
        plan, attempts, failure = CollaborativeSampleExecutor(adapter)._prepare_plan(context, SampleExecutionPolicy())
        self.assertIsNone(failure)
        self.assertEqual(attempts, 1)
        self.assertEqual(plan["decision_context"], before)
        self.assertEqual(context, before)
        catalog = adapter._available_tool_catalog(None, plan, role="planner")
        self.assertGreater(len(canonical_json({**plan, "tools": catalog}).encode()), 16384)
        self.assertLess(len(canonical_json(plan).encode()), 16384)
        self.assertEqual(plan["tool_catalog_digest"], digest(catalog))
        self.assertEqual([entry["definition_digest"] for entry in plan["tools"]], [digest(tool) for tool in catalog])
        outcome = fixture.predict(adapter, plan)
        self.assertIsNone(outcome.error)
        wire_catalog = native.requests[0]["request"]["context"]["available_tools"]
        self.assertEqual(wire_catalog, catalog)
        self.assertEqual(wire_catalog[2]["parameters"]["scale"]["maximum"], 1)

    def test_catalog_drift_is_rejected_and_historical_full_catalog_plan_remains_readable(self):
        _, adapter, plan, _ = self.fixture()
        original_catalog = adapter._available_tool_catalog(None, plan, role="planner")
        historical = {k:deepcopy(v) for k,v in plan.items() if k not in {"tool_catalog_digest", "tool_catalog_source"}}
        historical["tools"] = original_catalog
        self.assertEqual(adapter._available_tool_catalog(None, historical, role="planner"), original_catalog)
        adapter._agent_tool_catalog.append({"tool_id":"extra-tool", "version":"1", "parameters":{}})
        with self.assertRaisesRegex(SampleExecutionContractError, "catalog changed"):
            adapter._available_tool_catalog(None, plan, role="planner")

    def test_host_plan_contract_errors_publish_no_failed_cells_and_make_no_model_call(self):
        for oversized in (False, True):
            with self.subTest(oversized=oversized):
                _, adapter, plan, native = self.fixture()
                publications, checkpoints = [], []
                kwargs = ({"return_value": {**plan, "oversized_host_field":"x" * 16384}} if oversized
                          else {"side_effect":SampleExecutionContractError("invalid native Host plan")})
                with patch.object(adapter, "plan_batch", **kwargs), self.assertRaises(SampleExecutionContractError):
                    CollaborativeSampleExecutor(adapter).execute(
                        ({**_request("plan-fault").to_dict(), "observed":21.0},),
                        context=plan["decision_context"],
                        target_bounds={"air_temperature":{"minimum":-20.,"maximum":80.}},
                        algorithm_id="registered-predictor", algorithm_version="1",
                        result_callback=publications.extend, checkpoint_callback=checkpoints.append,
                    )
                self.assertEqual(publications, [])
                self.assertEqual(checkpoints, [])
                self.assertEqual(native.requests, [])
