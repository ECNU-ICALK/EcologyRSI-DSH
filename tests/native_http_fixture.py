"""Shared loopback HTTP fixture for current frozen native binding tests."""
import unittest

from tests.test_dsh_native_runtime import DshNativeHTTPGateTests, _FakeNativeRuntime


class NativeBindingFixture(unittest.TestCase):
    tearDown = DshNativeHTTPGateTests.tearDown
    _post_path = DshNativeHTTPGateTests._post_path
    _get = DshNativeHTTPGateTests._get

    def setUp(self):
        DshNativeHTTPGateTests.setUp(self)
        self.native = _FakeNativeRuntime()
        self.server.dsh_native_runtime = self.native

    @staticmethod
    def body(key, **overrides):
        return {
            "execution_protocol": "dsh_native_plugin_evolution@1",
            "optimization_protocol": "joint_plugin_epoch@2",
            "dataset_id": "agc_cucumber_2018", "episode_id": "agc_cucumber_2018:TeamA",
            "strategy_model_id": "dsh/strategy", "review_model_id": "dsh/review",
            "autonomous_mode": True, "knowledge_online_enabled": False,
            "budget": {"max_generations": 1, "max_candidates": 4, "candidates_per_generation": 4},
            "start": False, "auto_advance": 0, "idempotency_key": key,
            **overrides,
        }

    def create(self, key, **overrides):
        status, result = self._post_path("/runs", self.body(key, **overrides))
        self.assertEqual(status, 201, result)
        return self.server.director.state(result["projection"]["run_id"]), result
