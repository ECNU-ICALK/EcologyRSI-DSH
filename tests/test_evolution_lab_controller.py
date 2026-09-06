import tempfile
import unittest
from pathlib import Path

from ecologyrsi_dsh.evolution_lab.capabilities import CapabilityRegistry, CapabilitySpec
from ecologyrsi_dsh.evolution_lab.controller import EvolutionController
from ecologyrsi_dsh.evolution_lab.evaluator import EvaluationReport
from ecologyrsi_dsh.evolution_lab.genome import Mutation, PluginGenome
from ecologyrsi_dsh.evolution_lab.store import EvolutionStore


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = CapabilityRegistry([
            CapabilitySpec("skill", "planner-a@1", ("sample-planner",), ("greenhouse",)),
            CapabilitySpec("skill", "planner-b@1", ("sample-planner",), ("greenhouse",)),
            CapabilitySpec("tool", "predict@1", ("sample-planner",), ("greenhouse",)),
            CapabilitySpec("workflow", "wave@1", (), ("greenhouse",)),
            CapabilitySpec("algorithm", "ridge@1", (), ("greenhouse",)),
        ])
        self.parent = PluginGenome(
            domain="greenhouse",
            skills={"sample-planner": {"capability_id": "planner-a@1", "parameters": {}}},
            tools={"sample-planner": ("predict@1",)},
            workflow={"capability_id": "wave@1", "parameters": {}},
            algorithm={"capability_id": "ridge@1", "parameters": {}},
        )

    def test_candidate_is_recorded_and_eligible_version_can_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = EvolutionStore(Path(temp) / "evolution.sqlite")
            controller = EvolutionController(store, self.registry)
            controller.seed(self.parent)

            def backend(genome, cohort_id):
                score = 0.21 if genome.skills["sample-planner"]["capability_id"] == "planner-b@1" else 0.20
                return EvaluationReport(cohort_id, score, {"temperature@1h": score}, 0, 1, 1, 1, 1, True, 64)

            decision = controller.evaluate_and_record(
                self.parent.digest,
                Mutation("skill", "sample-planner", {"capability_id": "planner-b@1"}),
                "cohort-a",
                backend,
            )
            self.assertEqual(decision.status, "certification_eligible")
            controller.promote(controller.latest_candidate_digest())
            self.assertEqual(controller.current_incumbent().digest, controller.latest_candidate_digest())

    def test_rollback_is_durable_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = EvolutionStore(Path(temp) / "evolution.sqlite")
            controller = EvolutionController(store, self.registry)
            controller.seed(self.parent)
            controller.rollback(self.parent.digest, idempotency_key="rollback-1")
            controller.rollback(self.parent.digest, idempotency_key="rollback-1")
            self.assertEqual(controller.current_incumbent().digest, self.parent.digest)


if __name__ == "__main__":
    unittest.main()
