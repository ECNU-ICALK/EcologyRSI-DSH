"""Formal retrieval evidence remains fixed and independently replayable."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.ledger import Event
from ecologyrsi_dsh.core.models import CandidateStatus, digest
from ecologyrsi_dsh.core.state import RunStateReducer, project_run_state
from ecologyrsi_dsh.knowledge.formal_retrieval import (
    build_formal_retrieval_snapshot,
    search_formal_retrieval_snapshot,
)
from tests import test_artifact_revision_identity as artifact_fixture
from tests import test_genome_replay as native_fixture


class FormalRetrievalReplayTests(unittest.TestCase):
    def setUp(self):
        network = patch("socket.create_connection", side_effect=AssertionError("network forbidden"))
        network.start()
        self.addCleanup(network.stop)
        self.native = native_fixture.GenomeReplayTests()
        self.native.setUp()
        self.addCleanup(self.native.tearDown)
        self.run_id = self.native._start_new("run:formal-retrieval-replay")
        self.prefix = self.native.ledger.events(self.run_id)
        self.snapshot = build_formal_retrieval_snapshot()

    def reducer(self):
        reducer = RunStateReducer(self.prefix[0])
        reducer.apply(self.prefix[1:])
        return reducer

    def apply(self, reducer, kind, payload):
        event = Event(
            seq=reducer.events[-1].seq + 1,
            event_id=f"event:{kind}:{reducer.events[-1].seq + 1}",
            run_id=reducer.run.run_id,
            kind=kind,
            payload=payload,
            created_at="2026-10-20T00:00:00+00:00",
        )
        reducer.apply((event,))
        return event

    def freeze(self, reducer, snapshot=None):
        return self.apply(
            reducer, "FormalRetrievalSnapshotFrozen",
            {"snapshot": self.snapshot if snapshot is None else snapshot},
        )

    def retrieval(self, *, snapshot=None, queries=None, run_id=None):
        snapshot = self.snapshot if snapshot is None else snapshot
        queries = ["ridge forecasting"] if queries is None else queries
        result = search_formal_retrieval_snapshot(
            snapshot, tuple(queries), limit=snapshot["retrieval_policy"]["max_results"]
        )
        return {
            "schema_version": "ecologyrsi-dsh.retrieval-executed/1",
            "execution_owner": "dsh_agent_web_search",
            "identity": {
                "run_id": run_id or self.run_id,
                "role": "sample-planner",
                "stage": "sample.plan",
                "run_state_revision": 3,
                "stage_attempt": 1,
                "ledger_expected_revision": 3,
                "session_id": "session:formal-retrieval",
                "idempotency_key": "sample:formal-retrieval",
                "child_reservation_id": "reservation:formal-retrieval",
                "activation_lease_id": "lease:formal-retrieval",
                "genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64,
                "phenotype_instance_digest": "c" * 64,
            },
            "retrieval_idempotency_key": "formal-methods",
            "queries": queries,
            "query_digest": digest(queries),
            "provider_route": "frozen_snapshot",
            "frozen_snapshot_digest": snapshot["snapshot_digest"],
            "fallback_reason": None,
            "primary_quality": {
                "distinct_source_count": len(result["sources"]),
                "evidence_source_count": len(result["sources"]),
                "query_token_count": 0,
                "overlap_token_count": 0,
                "sufficient": True,
                "fallback_reason": None,
            },
            "result": result,
            "result_digest": digest(result),
        }

    def test_snapshot_and_local_results_replay_including_no_matches(self):
        for queries in (["ridge forecasting"], ["zzzzqqlocalnomatch"]):
            with self.subTest(queries=queries):
                reducer = self.reducer()
                self.freeze(reducer)
                payload = self.retrieval(queries=queries)
                self.apply(reducer, "DshRetrievalExecuted", payload)
                replay = project_run_state(reducer.events)
                self.assertEqual(replay.events[-1].payload["result"], payload["result"])
                self.assertEqual(reducer.formal_retrieval_snapshot, self.snapshot)
                if queries == ["zzzzqqlocalnomatch"]:
                    self.assertEqual(payload["result"]["sources"], [])

    def test_duplicate_snapshot_is_rejected_even_when_identical(self):
        reducer = self.reducer()
        self.freeze(reducer)
        with self.assertRaisesRegex(ValueError, "multiple formal retrieval snapshot"):
            self.freeze(reducer)

    def test_snapshot_validator_rejects_tampering_and_payload_extensions(self):
        altered = deepcopy(self.snapshot)
        altered["documents"][0]["summary"] += " modified after freezing"
        invalid_payloads = (
            {"snapshot": altered},
            {"snapshot": self.snapshot, "stage": "validation"},
            {"snapshot": {"snapshot_digest": self.snapshot["snapshot_digest"]}},
        )
        for payload in invalid_payloads:
            with self.subTest(payload_keys=tuple(payload)), self.assertRaises(ValueError):
                self.apply(self.reducer(), "FormalRetrievalSnapshotFrozen", payload)

    def test_frozen_route_requires_a_preceding_same_run_snapshot(self):
        with self.assertRaisesRegex(ValueError, "snapshot binding"):
            self.apply(self.reducer(), "DshRetrievalExecuted", self.retrieval())
        reducer = self.reducer()
        self.freeze(reducer)
        payload = self.retrieval()
        payload["frozen_snapshot_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "snapshot binding"):
            self.apply(reducer, "DshRetrievalExecuted", payload)

    def test_frozen_route_requires_canonical_digest_and_no_fallback(self):
        invalid = []
        for value in (None, "", "A" * 64, "a" * 63):
            payload = self.retrieval()
            payload["frozen_snapshot_digest"] = value
            invalid.append(payload)
        missing = self.retrieval()
        missing.pop("frozen_snapshot_digest")
        invalid.append(missing)
        fallback = self.retrieval()
        fallback["primary_quality"]["sufficient"] = False
        invalid.append(fallback)
        for payload in invalid:
            reducer = self.reducer()
            self.freeze(reducer)
            with self.subTest(digest=payload.get("frozen_snapshot_digest")), self.assertRaises(ValueError):
                self.apply(reducer, "DshRetrievalExecuted", payload)

    def test_online_route_cannot_claim_snapshot_but_legacy_online_event_replays(self):
        payload = self.retrieval()
        payload["provider_route"] = "dsh_primary"
        with self.assertRaisesRegex(ValueError, "online route"):
            self.apply(self.reducer(), "DshRetrievalExecuted", payload)
        payload.pop("frozen_snapshot_digest")
        reducer = self.reducer()
        self.apply(reducer, "DshRetrievalExecuted", payload)
        self.assertEqual(project_run_state(reducer.events).events[-1].payload, payload)

    def test_snapshot_freezing_immediately_forbids_online_retrieval(self):
        reducer = self.reducer()
        self.freeze(reducer)
        self.assertFalse(reducer.formal_stage_started)
        payload = self.retrieval()
        payload["provider_route"] = "dsh_primary"
        payload.pop("frozen_snapshot_digest")
        with self.assertRaisesRegex(ValueError, "online evidence after snapshot freezing"):
            self.apply(reducer, "DshRetrievalExecuted", payload)

    def test_recomputed_result_digest_cannot_hide_forged_snapshot_content(self):
        reducer = self.reducer()
        self.freeze(reducer)
        payload = self.retrieval()
        payload["result"]["sources"] = [
            {"url": "https://example.org/foreign", "snippet": "Unfrozen external evidence"}
        ]
        payload["result_digest"] = digest(payload["result"])
        with self.assertRaisesRegex(ValueError, "snapshot result mismatch"):
            self.apply(reducer, "DshRetrievalExecuted", payload)

    def test_response_only_snapshot_digest_is_not_accepted_in_event_result(self):
        reducer = self.reducer()
        self.freeze(reducer)
        payload = self.retrieval()
        payload["result"]["snapshot_digest"] = self.snapshot["snapshot_digest"]
        payload["result_digest"] = digest(payload["result"])
        with self.assertRaisesRegex(ValueError, "result is invalid"):
            self.apply(reducer, "DshRetrievalExecuted", payload)

    def test_queries_obey_frozen_limits_even_when_global_limits_allow_them(self):
        snapshot = build_formal_retrieval_snapshot({"retrieval_policy": {
            "max_queries_per_call": 1, "max_query_chars": 32,
        }})
        for queries in (["ridge", "forecasting"], ["r" * 33]):
            reducer = self.reducer()
            self.freeze(reducer, snapshot)
            with self.subTest(queries=queries), self.assertRaisesRegex(ValueError, "frozen snapshot policy"):
                self.apply(reducer, "DshRetrievalExecuted", self.retrieval(snapshot=snapshot, queries=queries))

    def test_formal_stage_reuses_snapshot_and_legacy_stage_can_replay_without_one(self):
        for with_snapshot in (False, True):
            with self.subTest(with_snapshot=with_snapshot):
                fixture = artifact_fixture.ArtifactRevisionIdentityTests()
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                fixture._record()
                fixture.reducer.run = replace(
                    fixture.reducer.run,
                    selection_incumbent_id=fixture.candidate.candidate_id,
                    validated_candidate_id=fixture.candidate.candidate_id,
                )
                fixture.reducer.candidates[fixture.candidate.candidate_id] = replace(
                    fixture.reducer.candidates[fixture.candidate.candidate_id],
                    status=CandidateStatus.PROMOTED,
                )
                fixture.reducer.task = replace(fixture.reducer.task, metadata={
                    **fixture.reducer.task.metadata, "episode_id": "episode:identity",
                })
                if with_snapshot:
                    fixture.ledger.append(fixture.run_id, "FormalRetrievalSnapshotFrozen", {"snapshot": self.snapshot})
                with patch.object(fixture.director, "state", side_effect=fixture._state):
                    for stage in ("validation", "final_test"):
                        fixture.director.reserve_formal_stage(
                            fixture.run_id, stage=stage,
                            candidate_id=fixture.candidate.candidate_id,
                            objective_family_digest="a" * 64, analysis_plan_digest="b" * 64,
                            partition_digest=digest(stage), idempotency_key=f"formal:{stage}",
                        )
                        fixture._state(fixture.run_id)
                        if with_snapshot:
                            fixture.ledger.append(
                                fixture.run_id, "DshRetrievalExecuted",
                                self.retrieval(run_id=fixture.run_id),
                            )
                            fixture._state(fixture.run_id)
                self.assertTrue(fixture.reducer.formal_stage_started)
                snapshots = [e for e in fixture.reducer.events if e.kind == "FormalRetrievalSnapshotFrozen"]
                self.assertEqual(len(snapshots), int(with_snapshot))
                with self.assertRaisesRegex(ValueError, "after formal evaluation starts"):
                    self.freeze(fixture.reducer)


if __name__ == "__main__":
    unittest.main()
