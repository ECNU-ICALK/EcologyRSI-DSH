from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ecologyrsi_dsh import (
    Evaluation,
    EventLedger,
    EvolutionDirector,
    FakeDSHAdapter,
    TaskManifest,
    ToyCropSoilWater,
)
from ecologyrsi_dsh.cli import main
from ecologyrsi_dsh.models import digest
from ecologyrsi_dsh.reporting import export_errors, run_export


class ExportContractTests(unittest.TestCase):
    def _make_run(self, db: Path, run_id: str) -> None:
        task = TaskManifest(
            task_id="export-task",
            objective="predict soil water",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={"max_candidates": 1},
            seed=5,
        )
        with EventLedger(db) as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=1))
            state = director.start_evolution(task, run_id=run_id)
            candidate = director.propose_and_spawn(run_id)
            proposal = director.state(run_id).proposal(candidate.proposal_id)
            evaluation = ToyCropSoilWater(seed=5).evaluate_candidate(
                run_id, candidate, proposal
            )
            evaluation_data = evaluation.to_dict()
            evaluation_data["metrics"] = {
                **evaluation_data["metrics"],
                "prediction_preview": [{"observed": 0.4, "predicted": 0.41}],
                "rows": [{"raw": "must-not-leak"}],
                "private_reasoning": "must-not-leak",
                "api_key": "training-secret",
                "capability_token": "capability-secret",
                "nested_credentials": {
                    "session-token": "session-secret",
                },
                "public_note": "capability_token=inline-secret",
                "operator_note": "Authorization: Bearer training-secret",
            }
            director.evaluate_and_decide(Evaluation.from_dict(evaluation_data))
            director.complete_run(state.run.run_id)

    def test_export_digest_and_import_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_db = root / "source.sqlite3"
            target_db = root / "target.sqlite3"
            bundle = root / "run.json"
            self._make_run(source_db, "run:export")
            with EventLedger(source_db) as ledger:
                payload = run_export(EvolutionDirector(ledger).replay("run:export"))
            self.assertEqual(export_errors(payload), [])
            self.assertEqual(payload["summary"]["training_asset_count"], 1)
            self.assertEqual(payload["summary"]["round_count"], 1)
            self.assertEqual(
                payload["summary"]["selection_scope"],
                "iterative_training_feedback_only",
            )
            self.assertEqual(
                payload["summary"]["formal_validation_status"], "not_run"
            )
            self.assertEqual(
                payload["summary"]["best_candidate_scope"],
                "iterative_training_feedback_only",
            )
            completed_event = next(
                item for item in payload["events"] if item["kind"] == "RunCompleted"
            )
            self.assertEqual(
                completed_event["payload"]["outcome"],
                "completed_with_search_retained_candidate",
            )
            self.assertEqual(len(payload["rounds"]), 1)
            self.assertEqual(payload["rounds"][0]["stages"]["decision"], "completed")
            asset = payload["training_assets"][0]
            self.assertEqual(
                asset["schema_version"],
                "ecologyrsi-dsh.evolution-training-sample/1",
            )
            self.assertEqual(asset["run_id"], "run:export")
            self.assertEqual(asset["generation"], 1)
            self.assertIsNone(asset["parent_candidate_id"])
            self.assertEqual(asset["input"]["dataset_id"], "generated-toy-series@1")
            self.assertEqual(asset["decision"]["status"], "approved")
            self.assertEqual(asset["admission"]["tier"], "iterative_positive")
            self.assertFalse(asset["admission"]["formal_training_ready"])
            self.assertTrue(asset["admission"]["requires_governance_review"])
            episode = asset["episode"]
            self.assertEqual(
                list(episode["stages"]),
                [
                    "strategy_input",
                    "proposal_response",
                    "training",
                    "evaluation",
                    "decision",
                ],
            )
            self.assertEqual(episode["lineage"]["candidate_id"], asset["candidate_id"])
            self.assertEqual(
                episode["event_chain_digest"], digest(episode["event_receipts"])
            )
            unsigned_episode = dict(episode)
            episode_digest = unsigned_episode.pop("episode_digest_sha256")
            self.assertEqual(episode_digest, digest(unsigned_episode))
            self.assertTrue(
                all(
                    set(receipt) == {"seq", "event_id", "kind", "payload_digest"}
                    for receipt in episode["event_receipts"]
                )
            )
            self.assertIn("rmse", episode["stages"]["evaluation"]["metrics"])
            self.assertTrue(episode["reproducibility"]["policy_model_digest"])
            self.assertTrue(episode["reproducibility"]["judge_model_digest"])
            serialized_asset = json.dumps(asset, ensure_ascii=False)
            for redline in (
                "prediction_preview",
                "must-not-leak",
                "private_reasoning",
                '"rows"',
                "training-secret",
                "capability-secret",
                "session-secret",
                "inline-secret",
            ):
                self.assertNotIn(redline, serialized_asset)
            bundle.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["verify", str(bundle)]), 0)
                self.assertEqual(main(["import", str(bundle), "--db", str(target_db)]), 0)
            with EventLedger(target_db) as ledger:
                state = EvolutionDirector(ledger).replay("run:export")
                self.assertEqual(len(state.events), len(payload["events"]))
                self.assertEqual(state.task_manifest.digest, payload["summary"]["manifest_digest"])

    def test_tampered_export_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "source.sqlite3"
            bundle = root / "run.json"
            self._make_run(db, "run:tamper")
            with EventLedger(db) as ledger:
                payload = run_export(EvolutionDirector(ledger).replay("run:tamper"))
            payload["summary"]["status"] = "tampered"
            bundle.write_text(json.dumps(payload), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["verify", str(bundle)]), 1)


if __name__ == "__main__":
    unittest.main()
