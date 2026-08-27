from __future__ import annotations

import json
import unittest

from ecologyrsi_dsh.api.events import EventEndpointsMixin
from ecologyrsi_dsh.core.ledger import Event


class TrajectoryPublicEventTests(unittest.TestCase):
    def _project(self, kind: str, payload: dict) -> dict:
        event = Event(
            seq=1,
            event_id=f"event:{kind}",
            run_id="run:public-trajectory",
            kind=kind,
            payload=payload,
            created_at="2026-08-27T00:00:00+00:00",
        )
        return EventEndpointsMixin._event_json(event)["payload"]

    def test_revision_batch_and_local_edit_events_are_redacted(self) -> None:
        revision = self._project(
            "CandidateRevisionCreated",
            {
                "revision": {
                    "generation": 0,
                    "candidate_id": "candidate:1",
                    "revision_id": "revision:1",
                    "parent_revision_id": None,
                    "source_batch_index": None,
                    "status": "active",
                    "revision_digest": "revision-digest",
                    "genome_digest": "genome-digest",
                    "behavior_digest": "behavior-digest",
                    "mutation_digest": "mutation-digest",
                    "genome": {"private_prompt": "SECRET-GENOME"},
                }
            },
        )
        evaluation = self._project(
            "FormalBatchEvaluated",
            {
                "evaluation": {
                    "evaluation_id": "evaluation:1",
                    "scope": {
                        "generation": 0,
                        "candidate_id": "candidate:1",
                        "candidate_revision_id": "revision:1",
                        "batch_index": 0,
                        "origin_count": 50,
                        "cohort_digest": "cohort-digest",
                    },
                    "score": 0.75,
                    "passed": True,
                    "evaluator_digest": "evaluator-digest",
                    "metrics": {"private_trace": "SECRET-METRICS"},
                }
            },
        )
        proposal = self._project(
            "LocalEditProposalRecorded",
            {
                "proposal_id": "local-proposal:1",
                "candidate_id": "candidate:1",
                "batch_index": 0,
                "evidence_scope_digest": "scope-digest",
                "decision": "edit",
                "operations": [{"private_code": "SECRET-OPERATION"}],
            },
        )

        projected = json.dumps(
            [revision, evaluation, proposal], ensure_ascii=False, sort_keys=True
        )
        self.assertNotIn("SECRET-GENOME", projected)
        self.assertNotIn("SECRET-METRICS", projected)
        self.assertNotIn("SECRET-OPERATION", projected)
        self.assertNotIn("genome", revision)
        self.assertNotIn("metrics", evaluation)
        self.assertNotIn("operations", proposal)
        self.assertEqual(revision["revision_id"], "revision:1")
        self.assertEqual(evaluation["origin_count"], 50)
        self.assertEqual(proposal["operation_count"], 1)

    def test_holdout_and_champion_events_expose_only_audit_summary(self) -> None:
        holdout = self._project(
            "GenerationHoldoutFrozen",
            {
                "holdout": {
                    "holdout_id": "holdout:0",
                    "generation": 0,
                    "cohort_digest": "cohort-digest",
                    "origin_count": 169,
                    "arm_bindings": {
                        "finalist_1": {
                            "candidate_id": "candidate:1",
                            "candidate_revision_id": "revision:1",
                        },
                        "finalist_2": {
                            "candidate_id": "candidate:2",
                            "candidate_revision_id": "revision:2",
                        },
                        "incumbent": {
                            "candidate_id": "candidate:0",
                            "candidate_revision_id": "revision:0",
                        },
                    },
                }
            },
        )
        comparison = self._project(
            "GenerationComparisonRecorded",
            {
                "comparison": {
                    "comparison_id": "comparison:0",
                    "comparison_digest": "comparison-digest",
                    "generation": 0,
                    "cohort_digest": "cohort-digest",
                    "selected_candidate_id": "candidate:1",
                    "selected_revision_id": "revision:1",
                    "gate_results": {"private_reasoning": "SECRET-GATE"},
                    "holdout_evaluations": [{"metrics": "SECRET-HOLDOUT"}],
                }
            },
        )
        champion = self._project(
            "GenerationChampionSelected",
            {
                "generation": 0,
                "selected_candidate_id": "candidate:1",
                "selected_revision_id": "revision:1",
            },
        )

        projected = json.dumps(
            [holdout, comparison, champion], ensure_ascii=False, sort_keys=True
        )
        self.assertNotIn("SECRET-GATE", projected)
        self.assertNotIn("SECRET-HOLDOUT", projected)
        self.assertNotIn("gate_results", comparison)
        self.assertNotIn("holdout_evaluations", comparison)
        self.assertEqual(holdout["arm_count"], 3)
        self.assertEqual(champion["selected_revision_id"], "revision:1")


if __name__ == "__main__":
    unittest.main()
