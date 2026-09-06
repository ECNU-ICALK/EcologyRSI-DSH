import unittest

from ecologyrsi_dsh.evolution_lab.capabilities import CapabilityRegistry, CapabilitySpec
from ecologyrsi_dsh.evolution_lab.genome import Mutation, MutationPlanner, PluginGenome


class PluginGenomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = CapabilityRegistry([
            CapabilitySpec("skill", "planner-a@1", ("sample-planner",), ("greenhouse",), {"threshold": {"minimum": 0, "maximum": 1}}),
            CapabilitySpec("skill", "planner-b@1", ("sample-planner",), ("greenhouse",)),
            CapabilitySpec("tool", "predict@1", ("sample-planner",), ("greenhouse",)),
            CapabilitySpec("tool", "repair@1", ("sample-planner",), ("greenhouse",)),
            CapabilitySpec("workflow", "wave@1", ("sample-planner",), ("greenhouse",), {"wave_size": {"minimum": 1, "maximum": 16, "integer": True}}),
            CapabilitySpec("algorithm", "ridge@1", (), ("greenhouse",), {"alpha": {"minimum": 0, "maximum": 10}}),
        ])
        self.parent = PluginGenome(
            domain="greenhouse",
            skills={"sample-planner": {"capability_id": "planner-a@1", "parameters": {"threshold": 0.7}}},
            tools={"sample-planner": ("predict@1",)},
            workflow={"capability_id": "wave@1", "parameters": {"wave_size": 4}},
            algorithm={"capability_id": "ridge@1", "parameters": {"alpha": 1}},
        )

    def test_digest_is_order_independent_and_round_trips(self) -> None:
        reversed_parent = PluginGenome(
            domain="greenhouse",
            skills={"sample-planner": {"parameters": {"threshold": 0.7}, "capability_id": "planner-a@1"}},
            tools={"sample-planner": ["predict@1"]},
            workflow={"parameters": {"wave_size": 4}, "capability_id": "wave@1"},
            algorithm={"parameters": {"alpha": 1}, "capability_id": "ridge@1"},
        )
        self.assertEqual(self.parent.digest, reversed_parent.digest)
        self.assertEqual(PluginGenome.from_dict(self.parent.to_dict()).digest, self.parent.digest)

    def test_planner_applies_one_safe_axis(self) -> None:
        planner = MutationPlanner(self.registry)
        child = planner.apply(self.parent, Mutation("skill", "sample-planner", {"capability_id": "planner-b@1", "parameters": {}}))
        self.assertNotEqual(child.digest, self.parent.digest)
        self.assertEqual(child.algorithm, self.parent.algorithm)
        with self.assertRaises(ValueError):
            planner.apply(self.parent, Mutation("unknown", "x", {}))

    def test_tool_policy_cannot_escape_registered_role(self) -> None:
        planner = MutationPlanner(self.registry)
        with self.assertRaises(ValueError):
            planner.apply(self.parent, Mutation("tool", "sample-planner", {"enabled": ["missing@1"]}))


if __name__ == "__main__":
    unittest.main()
