from __future__ import annotations

import unittest

from ecologyrsi_dsh import (
    EventLedger,
    EvolutionDirector,
    FakeDSHAdapter,
    TaskManifest,
    ToyCropSoilWater,
)


class ScientificBoundaryTests(unittest.TestCase):
    def test_toy_dataset_digest_is_stable_and_seed_scoped(self) -> None:
        fixed = ToyCropSoilWater(seed=0)
        first = ToyCropSoilWater(seed=7)
        replay = ToyCropSoilWater(seed=7)
        different = ToyCropSoilWater(seed=8)

        self.assertEqual(
            fixed.dataset_digest,
            "af2534320547eefabca79865a3bd22f63a56c6791bcda4f5d971a8cccffbc5a2",
        )
        self.assertEqual(first.dataset_digest, replay.dataset_digest)
        self.assertNotEqual(first.dataset_digest, different.dataset_digest)

    def test_evaluation_records_visible_demo_scope_and_data_digest(self) -> None:
        ledger = EventLedger()
        try:
            task = TaskManifest(
                task_id="toy-boundary",
                objective="predict soil water",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("toy-dataset@1",),
                budget={"max_candidates": 1},
                seed=7,
            )
            director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=1))
            director.start_evolution(task, run_id="run:scientific-boundary")
            candidate = director.propose_and_spawn("run:scientific-boundary")
            state = director.state("run:scientific-boundary")
            toy = ToyCropSoilWater(seed=task.seed)
            evaluation = toy.evaluate_candidate(
                state.run.run_id,
                candidate,
                state.proposal(candidate.proposal_id),
            )

            self.assertEqual(evaluation.partition, "validation")
            self.assertEqual(evaluation.metrics["dataset_digest"], toy.dataset_digest)
            self.assertEqual(evaluation.metrics["evaluation_scope"], "visible/validation/demo")
            self.assertIs(evaluation.metrics["causal_interpretation"], False)
        finally:
            ledger.close()


if __name__ == "__main__":
    unittest.main()
