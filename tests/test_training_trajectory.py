from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ecologyrsi_dsh import (
    EventLedger,
    EvolutionDirector,
    FakeDSHAdapter,
    ModelArtifact,
    TaskManifest,
    ToyCropSoilWater,
)
from ecologyrsi_dsh.models import Evaluation
from ecologyrsi_dsh.reporting import training_assets


class TrainingTrajectoryContractTests(unittest.TestCase):
    def _task(self, max_candidates: int = 1) -> TaskManifest:
        return TaskManifest(
            task_id="trajectory-contract",
            objective="预测土壤水分并根据反馈优化参数",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={
                "max_candidates": max_candidates,
                "max_generations": max_candidates,
            },
            seed=3,
        )

    def _finish_candidate(
        self,
        director: EvolutionDirector,
        run_id: str,
        candidate: object,
    ) -> None:
        # Keep the training artifact explicit so the trajectory can distinguish
        # host compilation/fitting from the later feedback prediction pass.
        candidate_id = candidate.candidate_id  # type: ignore[attr-defined]
        proposal = director.state(run_id).proposal(candidate.proposal_id)  # type: ignore[attr-defined]
        artifact = ModelArtifact(
            artifact_id=f"artifact:{candidate_id}",
            run_id=run_id,
            candidate_id=candidate_id,
            model_id="toy-rolling-water@1",
            dataset_digest="dataset:trajectory",
            training_partition="training_fit",
            training_rows=18,
            parameters=proposal.changes,
            learned_parameters={"bias": 0.01},
            metrics={"fit_rmse": 0.12},
        )
        director.record_artifact(artifact)
        evaluation = ToyCropSoilWater(seed=3).evaluate_candidate(
            run_id, candidate, proposal  # type: ignore[arg-type]
        )
        metrics = dict(evaluation.metrics)
        metrics.update(
            {
                "prediction_preview": [
                    {
                        "timestamp": 20,
                        "origin_timestamp": 19,
                        "target_timestamp": 20,
                        "target": "soil_water",
                        "unit": "fraction",
                        "horizon_hours": 24,
                        "observed": 0.4,
                        "predicted": 0.42,
                        "baseline": 0.38,
                        "raw_rows": "must not leak",
                        "private_reasoning": "must not leak",
                    }
                ],
                "api_key": "secret",
            }
        )
        evaluation = Evaluation.from_dict(
            {
                **evaluation.to_dict(),
                "metrics": metrics,
                "artifact_digest": artifact.digest,
            }
        )
        director.evaluate_and_decide(evaluation)

    def test_complete_trajectory_has_all_stages_and_safe_prediction_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with EventLedger(Path(directory) / "events.sqlite3") as ledger:
                director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=4))
                run = director.start_evolution(self._task(), run_id="run:trajectory")
                candidate = director.propose_and_spawn(run.run.run_id)
                self._finish_candidate(director, run.run.run_id, candidate)

                asset = training_assets(director.state(run.run.run_id))[0]
                trajectory = asset["trajectory"]
                expected = {
                    "input_context",
                    "agent_research",
                    "agent_proposal",
                    "host_compile",
                    "training_prediction",
                    "agent_feedback",
                    "agent_optimization",
                    "final_result",
                }
                self.assertTrue(expected.issubset(trajectory))
                self.assertEqual(
                    trajectory["training_prediction"]["sample_count"],
                    17,
                )
                self.assertEqual(trajectory["training_prediction"]["shown_count"], 1)
                record = trajectory["training_prediction"]["prediction_records"][0]
                self.assertEqual(record["predicted_value"], 0.42)
                self.assertEqual(record["observed_value"], 0.4)
                self.assertEqual(
                    record["input_reference"]["partition"],
                    "validation",
                )
                self.assertEqual(record["reference"], record["observed_value"])
                self.assertAlmostEqual(record["error"], 0.02)
                self.assertEqual(
                    trajectory["final_result"]["prediction_ref"],
                    "training_prediction.prediction_records",
                )
                self.assertEqual(
                    asset["episode"]["trajectory"]["schema_version"],
                    trajectory["schema_version"],
                )
                serialized = json.dumps(asset, ensure_ascii=False)
                for forbidden in (
                    "prediction_preview",
                    '"rows"',
                    "raw_rows",
                    "private_reasoning",
                    "api_key",
                    "must not leak",
                ):
                    self.assertNotIn(forbidden, serialized)

    def test_child_trajectory_carries_parent_feedback_and_parent_asset_stays_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with EventLedger(Path(directory) / "events.sqlite3") as ledger:
                director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=4))
                run_id = director.start_evolution(
                    self._task(max_candidates=2), run_id="run:parent-child"
                ).run.run_id
                parent = director.propose_and_spawn(run_id)
                self._finish_candidate(director, run_id, parent)
                director.advance_generation(run_id)
                parent_asset_before = training_assets(director.state(run_id))[0]

                child = director.propose_and_spawn(
                    run_id, parent_candidate_id=parent.candidate_id
                )
                self._finish_candidate(director, run_id, child)
                assets = training_assets(director.state(run_id))
                parent_asset_after = next(
                    item for item in assets if item["candidate_id"] == parent.candidate_id
                )
                child_asset = next(
                    item for item in assets if item["candidate_id"] == child.candidate_id
                )
                self.assertEqual(parent_asset_before, parent_asset_after)
                child_trajectory = child_asset["trajectory"]
                self.assertEqual(
                    child_trajectory["input_context"]["parent_candidate_id"],
                    parent.candidate_id,
                )
                self.assertEqual(
                    child_trajectory["agent_feedback"]["parent_candidate"][
                        "candidate_id"
                    ],
                    parent.candidate_id,
                )
                self.assertTrue(
                    any(
                        item["parameter"]
                        for item in child_trajectory["agent_optimization"][
                            "parameter_changes"
                        ]
                    )
                )
                self.assertEqual(
                    child_trajectory["agent_optimization"]["feedback_source"],
                    "parent_evaluation",
                )


if __name__ == "__main__":
    unittest.main()
