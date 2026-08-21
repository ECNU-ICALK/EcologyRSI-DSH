from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ecologyrsi_dsh.director import EvolutionDirector
from ecologyrsi_dsh.dsh import FakeDSHAdapter
from ecologyrsi_dsh.ledger import EventLedger
from ecologyrsi_dsh.models import (
    Evaluation,
    HumanIntervention,
    InterventionKind,
    ModelArtifact,
    Proposal,
    TaskManifest,
)
from ecologyrsi_dsh.server import EvolutionRequestHandler, _intervention_projection


class HumanInterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.ledger = EventLedger(Path(self.directory.name) / "events.sqlite3")
        self.director = EvolutionDirector(self.ledger, FakeDSHAdapter())
        self.task = TaskManifest(
            task_id="human-loop",
            objective="test append-only human guidance",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={"max_candidates": 2, "max_generations": 2},
            metadata={"strategy_id": "parameter_sweep@1"},
        )
        self.run_id = self.director.start_evolution(
            self.task, run_id="run:human-loop"
        ).run.run_id

    def tearDown(self) -> None:
        self.ledger.close()
        self.directory.cleanup()

    def test_paused_intervention_is_applied_to_exactly_one_proposal(self) -> None:
        self.director.pause_run(self.run_id)
        intervention = HumanIntervention(
            intervention_id="human-1",
            run_id=self.run_id,
            kind=InterventionKind.PARAMETER_OVERRIDE,
            message="下一轮提高平滑权重",
            created_by="研究者",
            parameter_overrides={"alpha": 0.8},
        )
        self.director.record_intervention(intervention)
        self.director.resume_run(self.run_id)

        candidate = self.director.propose_and_spawn(self.run_id)
        state = self.director.state(self.run_id)
        proposal = state.proposal(candidate.proposal_id)
        self.assertEqual(proposal.changes["alpha"], 0.8)
        self.assertIn("下一轮提高平滑权重", proposal.rationale)
        self.assertEqual(
            state.interventions[0].applied_proposal_id,
            proposal.proposal_id,
        )
        receipt = next(
            event.payload
            for event in state.events
            if event.kind == "HumanInterventionApplied"
        )
        self.assertEqual(receipt["application_status"], "enforced")
        self.assertTrue(receipt["enforced"])
        self.assertEqual(state.pending_interventions, ())

    def test_training_artifact_is_digest_bound_to_evaluation(self) -> None:
        candidate = self.director.propose_and_spawn(self.run_id)
        artifact = ModelArtifact(
            artifact_id="artifact-1",
            run_id=self.run_id,
            candidate_id=candidate.candidate_id,
            model_id="rolling-residual@1",
            dataset_digest="dataset-sha256",
            training_partition="training_fit",
            training_rows=24,
            parameters={"window": 3},
            learned_parameters={"bias": 0.02},
            metrics={"training_rmse": 0.1},
        )
        self.director.record_artifact(artifact)
        evaluation = Evaluation(
            evaluation_id="evaluation-1",
            run_id=self.run_id,
            candidate_id=candidate.candidate_id,
            score=0.8,
            passed=True,
            metrics={"rmse": 0.2},
            partition="validation",
            evaluator_digest="test-evaluator@1",
            artifact_digest=artifact.digest,
        )
        self.director.record_evaluation(evaluation)

        replayed = EvolutionDirector(self.ledger).replay(self.run_id)
        self.assertEqual(replayed.artifact_for(candidate.candidate_id), artifact)
        self.assertEqual(
            replayed.evaluation_for(candidate.candidate_id).artifact_digest,
            artifact.digest,
        )

    def test_guidance_constraint_and_unparsed_text_have_explicit_receipts(self) -> None:
        self.director.pause_run(self.run_id)
        for intervention in (
            HumanIntervention(
                intervention_id="human-guidance",
                run_id=self.run_id,
                kind=InterventionKind.GUIDANCE,
                message="缩短时间窗口",
                created_by="研究者",
            ),
            HumanIntervention(
                intervention_id="human-constraint",
                run_id=self.run_id,
                kind=InterventionKind.CONSTRAINT,
                message="window<=1",
                created_by="研究者",
            ),
            HumanIntervention(
                intervention_id="human-free-text",
                run_id=self.run_id,
                kind=InterventionKind.GUIDANCE,
                message="请综合考虑模型稳定性",
                created_by="研究者",
            ),
        ):
            self.director.record_intervention(intervention)
        self.director.resume_run(self.run_id)

        proposal = self.director.request_proposal(self.run_id)
        self.assertEqual(proposal.changes["window"], 1)
        state = self.director.state(self.run_id)
        receipts = {
            event.payload["intervention_id"]: event.payload
            for event in state.events
            if event.kind == "HumanInterventionApplied"
        }
        self.assertEqual(receipts["human-guidance"]["application_status"], "applied")
        self.assertTrue(receipts["human-guidance"]["applied"])
        self.assertFalse(receipts["human-guidance"]["enforced"])
        self.assertEqual(receipts["human-constraint"]["application_status"], "enforced")
        self.assertTrue(receipts["human-constraint"]["enforced"])
        self.assertEqual(receipts["human-free-text"]["application_status"], "recorded")
        self.assertFalse(receipts["human-free-text"]["applied"])
        self.assertIn("未唯一识别", receipts["human-free-text"]["reason"])
        self.assertEqual(state.pending_interventions, ())

    def test_legacy_application_events_are_projected_conservatively(self) -> None:
        self.director.pause_run(self.run_id)
        controls = (
            HumanIntervention(
                intervention_id="legacy-override",
                run_id=self.run_id,
                kind=InterventionKind.PARAMETER_OVERRIDE,
                message="旧版参数覆盖",
                created_by="研究者",
                parameter_overrides={"alpha": 0.5},
            ),
            HumanIntervention(
                intervention_id="legacy-guidance",
                run_id=self.run_id,
                kind=InterventionKind.GUIDANCE,
                message="旧版方向建议",
                created_by="研究者",
            ),
        )
        for control in controls:
            self.director.record_intervention(control)
        self.director.resume_run(self.run_id)
        proposal = Proposal(
            proposal_id="proposal:legacy-receipts",
            run_id=self.run_id,
            generation=0,
            title="旧版提案",
            changes={"alpha": 0.5, "window": 3, "water_threshold": 0.35},
        )
        self.ledger.append(
            self.run_id,
            "ProposalSubmitted",
            {"proposal": proposal.to_dict()},
        )
        for control in controls:
            self.ledger.append(
                self.run_id,
                "HumanInterventionApplied",
                {
                    "intervention_id": control.intervention_id,
                    "proposal_id": proposal.proposal_id,
                },
            )

        state = self.director.state(self.run_id)
        projected = {
            item.intervention_id: _intervention_projection(state, item)
            for item in state.interventions
        }
        self.assertEqual(projected["legacy-override"]["application_status"], "enforced")
        self.assertEqual(projected["legacy-override"]["status"], "已强制执行")
        self.assertEqual(projected["legacy-guidance"]["application_status"], "recorded")
        self.assertEqual(projected["legacy-guidance"]["status"], "仅记录（未执行）")

        kinds = {item.intervention_id: item.kind.value for item in state.interventions}
        rendered = [
            EvolutionRequestHandler._event_json(
                event,
                intervention_kinds=kinds,
            )
            for event in state.events
            if event.kind == "HumanInterventionApplied"
        ]
        self.assertEqual(rendered[0]["payload"]["application_status"], "enforced")
        self.assertEqual(rendered[1]["payload"]["application_status"], "recorded")

    def test_constraint_is_enforced_after_parameter_override(self) -> None:
        self.director.pause_run(self.run_id)
        self.director.record_intervention(
            HumanIntervention(
                intervention_id="human-override-before-constraint",
                run_id=self.run_id,
                kind=InterventionKind.PARAMETER_OVERRIDE,
                message="先设置较长窗口",
                created_by="研究者",
                parameter_overrides={"window": 6},
            )
        )
        self.director.record_intervention(
            HumanIntervention(
                intervention_id="human-final-constraint",
                run_id=self.run_id,
                kind=InterventionKind.CONSTRAINT,
                message="window<=4",
                created_by="研究者",
            )
        )
        self.director.resume_run(self.run_id)

        proposal = self.director.request_proposal(self.run_id)
        self.assertEqual(proposal.changes["window"], 4)
        receipts = {
            event.payload["intervention_id"]: event.payload
            for event in self.director.state(self.run_id).events
            if event.kind == "HumanInterventionApplied"
        }
        self.assertEqual(
            receipts["human-override-before-constraint"]["application_status"],
            "applied",
        )
        self.assertFalse(receipts["human-override-before-constraint"]["enforced"])
        self.assertEqual(
            receipts["human-final-constraint"]["application_status"],
            "enforced",
        )

    def test_intervention_is_rejected_while_running(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected paused"):
            self.director.record_intervention(
                HumanIntervention(
                    intervention_id="human-running",
                    run_id=self.run_id,
                    kind="guidance",
                    message="不应在运行中写入",
                    created_by="研究者",
                )
            )


if __name__ == "__main__":
    unittest.main()
