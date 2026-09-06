import unittest

from ecologyrsi_dsh.evolution_lab.capabilities import (
    CapabilityRegistry,
    CapabilitySpec,
    CapabilityProposal,
)


class CapabilityRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = CapabilityRegistry(
            [
                CapabilitySpec(
                    kind="skill",
                    capability_id="planner-balanced@1",
                    roles=("sample-planner",),
                    domains=("greenhouse",),
                    parameters={"confidence_threshold": {"minimum": 0.0, "maximum": 1.0}},
                ),
                CapabilitySpec(
                    kind="tool",
                    capability_id="prediction@1",
                    roles=("sample-planner",),
                    domains=("greenhouse",),
                ),
                CapabilitySpec(
                    kind="workflow",
                    capability_id="sample-wave@1",
                    roles=("sample-planner",),
                    domains=("greenhouse",),
                    parameters={"wave_size": {"minimum": 1, "maximum": 32, "integer": True}},
                ),
            ]
        )

    def test_registry_validates_role_domain_and_parameter(self) -> None:
        self.assertTrue(self.registry.compatible("planner-balanced@1", "skill", "sample-planner", "greenhouse"))
        self.registry.validate_parameters("planner-balanced@1", {"confidence_threshold": 0.8})
        with self.assertRaises(ValueError):
            self.registry.validate_parameters("planner-balanced@1", {"confidence_threshold": 2})
        with self.assertRaises(ValueError):
            self.registry.require("planner-balanced@1", "skill", role="generation-judge")

    def test_unknown_or_executable_capability_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.registry.require("missing@1", "tool")
        with self.assertRaises(ValueError):
            CapabilityProposal(
                kind="tool",
                capability_id="generated@1",
                roles=("sample-planner",),
                artifact={"code": "import os; os.system('bad')"},
            ).validate()

    def test_digest_is_stable_and_proposals_are_pending(self) -> None:
        first = self.registry.require("planner-balanced@1", "skill")
        second = CapabilitySpec.from_dict(first.to_dict())
        self.assertEqual(first.digest, second.digest)
        proposal = CapabilityProposal(
            kind="skill",
            capability_id="researcher-generated@1",
            roles=("researcher",),
            artifact={"instruction": "compare evidence and state uncertainty"},
        )
        self.assertEqual(proposal.validate().status, "pending")


if __name__ == "__main__":
    unittest.main()
