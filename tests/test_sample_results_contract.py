from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zlib

import ecologyrsi_dsh.core.sample_results as sample_results_module
from ecologyrsi_dsh import (
    Evaluation,
    EventLedger,
    EvolutionDirector,
    Proposal,
    TaskManifest,
)
from ecologyrsi_dsh.core.sample_results import (
    build_sample_results,
    decode_sample_results,
    sample_result_batch_event_payload,
    sample_results_event_payload,
)
from ecologyrsi_dsh.evaluators.sample_execution import (
    CollaborativeSampleExecutor,
    SampleExecutionPolicy,
)
from ecologyrsi_dsh.presentation.reporting import run_export
from ecologyrsi_dsh.server import EvolutionHTTPServer


RUN_ID = "run:sample-contract"
CANDIDATE_ID = "candidate:sample-contract"


def _task(task_id: str = "sample-contract") -> TaskManifest:
    return TaskManifest(
        task_id=task_id,
        objective="test private per-sample result persistence",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={"max_candidates": 4, "max_generations": 2},
        seed=17,
    )


def _source_row(index: int, *, failed: bool = False) -> dict:
    row = {
        "sample_index": index,
        "sample_id": f"sample:{index}",
        "target": "soil_water",
        "unit": "fraction",
        "horizon_hours": 24,
        "origin_timestamp": index,
        "target_timestamp": index + 1,
        "observed": 0.5,
        "predicted": 0.45 if not failed else 0.9,
        "baseline": 0.7,
        "sample_execution_status": "failed" if failed else "succeeded",
        "sample_execution_attempts": 3 if failed else 1,
        "sample_execution_retry_count": 2 if failed else 0,
    }
    if failed:
        row.update(
            {
                "scoring_fallback": "failure_non_improvement_penalty",
                "scoring_fallback_source": "registered_algorithm_prediction",
                "sample_execution_failure": {"class": "tool_timeout"},
                "sample_execution_failure_summary": {
                    "decisions": [
                        {
                            "role": "host_adjudicator",
                            "decision": "skip_sample_after_retry_budget",
                            "status": "completed",
                            "reason_code": "tool_timeout",
                        }
                    ],
                    "tools": [
                        {
                            "tool_id": "ridge-tool",
                            "version": "1",
                            "status": "failed",
                        }
                    ],
                },
            }
        )
    return row


def _checkpoint(
    *, cohort: str = "a", execution: str = "b", sample_count: int = 2
) -> dict:
    return {
        "schema_version": "ecologyrsi-dsh.sample-checkpoint/1",
        "cohort_digest": cohort * 64,
        "execution_context_digest": execution * 64,
        "sample_count": sample_count,
    }


def _seed_candidate(
    director: EvolutionDirector,
    run_id: str,
    candidate_id: str = CANDIDATE_ID,
) -> tuple[Proposal, object]:
    director.start_evolution(_task(run_id), run_id=run_id)
    proposal = Proposal(
        proposal_id=f"proposal:{run_id}",
        run_id=run_id,
        generation=0,
        title="sample result contract",
        changes={"alpha": 0.2},
    )
    director.submit_proposal(proposal)
    candidate = director.spawn_candidate(
        run_id, proposal, candidate_id=candidate_id
    )
    return proposal, candidate


class _TwoSampleAdapter:
    adapter_id = "two-sample-adapter"
    adapter_version = "1"

    def __init__(self) -> None:
        self.sample_calls = 0

    def plan_batch(self, context):
        return {"plan_id": "two-sample@1"}

    def predict_sample(self, request, plan, *, attempt):
        del plan, attempt
        self.sample_calls += 1
        return {
            "predicted": request.proposed_prediction,
            "agent_decisions": [
                {"role": "planner", "decision": "use_registered_tool", "status": "completed"}
            ],
            "tool_calls": [
                {"tool_id": "ridge-tool", "version": "1", "status": "completed"}
            ],
        }


class SampleResultCodecTests(unittest.TestCase):
    def test_fit_selected_baseline_provenance_and_normalized_reward_round_trip(self) -> None:
        source = _source_row(1)
        source.update(
            {
                "baseline": 0.8,
                "model_reference_baseline": 0.7,
                "baseline_id": "seasonal_24h",
                "baseline_profile_digest": "b" * 64,
                "normalization_scale": 0.2,
            }
        )
        rows = build_sample_results(CANDIDATE_ID, [source])

        self.assertEqual(rows[0]["baseline_id"], "seasonal_24h")
        self.assertEqual(rows[0]["model_reference_baseline"], 0.7)
        self.assertAlmostEqual(rows[0]["reward"], 0.25)
        self.assertAlmostEqual(rows[0]["normalized_reward"], 1.0)

        evaluation = Evaluation(
            evaluation_id="evaluation:baseline-v2",
            run_id=RUN_ID,
            candidate_id=CANDIDATE_ID,
            score=0.1,
            passed=True,
            partition="validation",
        )
        payload = sample_results_event_payload(evaluation, rows, revision="revision:1")
        self.assertEqual(decode_sample_results(payload), rows)

        legacy = deepcopy(payload)
        legacy["reward_definition"] = sample_results_module.SAMPLE_REWARD_DEFINITION_V1
        self.assertEqual(decode_sample_results(legacy), rows)

    def test_reward_status_unit_and_fallback_source_round_trip(self) -> None:
        rows = build_sample_results(
            CANDIDATE_ID, [_source_row(1), _source_row(2, failed=True)]
        )
        self.assertAlmostEqual(rows[0]["reward"], 0.15)
        self.assertEqual(rows[0]["prediction_source"], "agent_tool_prediction")
        self.assertEqual(rows[1]["prediction_source"], "scoring_fallback")
        self.assertEqual(rows[1]["scoring_fallback_source"], "registered_algorithm_prediction")
        self.assertEqual(rows[1]["failure_class"], "tool_timeout")
        self.assertEqual(
            rows[1]["failure_summary"]["decisions"][0]["reason_code"],
            "tool_timeout",
        )
        self.assertEqual(
            rows[1]["failure_summary"]["tools"][0]["tool_id"],
            "ridge-tool",
        )
        self.assertEqual(rows[1]["unit"], "fraction")

        evaluation = Evaluation(
            evaluation_id="evaluation:codec",
            run_id=RUN_ID,
            candidate_id=CANDIDATE_ID,
            score=0.1,
            passed=False,
            partition="validation",
        )
        payload = sample_results_event_payload(evaluation, rows, revision="revision:1")
        self.assertEqual(decode_sample_results(payload), rows)

    def test_corrupt_metadata_digest_base64_and_decompression_limit_fail_closed(self) -> None:
        rows = build_sample_results(CANDIDATE_ID, [_source_row(1)])
        evaluation = Evaluation(
            evaluation_id="evaluation:corrupt",
            run_id=RUN_ID,
            candidate_id=CANDIDATE_ID,
            score=0.1,
            passed=True,
            partition="validation",
        )
        original = sample_results_event_payload(evaluation, rows, revision="revision:1")

        bad_count = deepcopy(original)
        bad_count["record_count"] = 2
        with self.assertRaisesRegex(ValueError, "metadata"):
            decode_sample_results(bad_count)

        bad_digest = deepcopy(original)
        bad_digest["result_digest"] = "wrong"
        with self.assertRaisesRegex(ValueError, "metadata"):
            decode_sample_results(bad_digest)

        bad_base64 = deepcopy(original)
        bad_base64["archive"]["payload"] = "not base64!"
        with self.assertRaisesRegex(ValueError, "base64"):
            decode_sample_results(bad_base64)

        bomb = deepcopy(original)
        raw = b"[" + (b" " * 128) + b"]"
        compressed = zlib.compress(raw)
        bomb["archive"].update(
            {
                "payload": base64.b64encode(compressed).decode("ascii"),
                "compressed_bytes": len(compressed),
                "uncompressed_bytes": 16,
            }
        )
        with patch.object(sample_results_module, "MAX_SAMPLE_RESULTS_UNCOMPRESSED_BYTES", 16):
            with self.assertRaisesRegex(ValueError, "decompression limit"):
                decode_sample_results(bomb)

    def test_executor_assigns_one_global_order_across_two_result_batches(self) -> None:
        adapter = _TwoSampleAdapter()
        published: list[tuple[dict, ...]] = []

        def record(rows):
            published.append(tuple(dict(row) for row in rows))
            self.assertEqual(adapter.sample_calls, len(published))

        source_rows = []
        for index in (1, 2):
            row = _source_row(index)
            row.pop("sample_index")
            row.pop("sample_id")
            row["partition"] = "validation"
            source_rows.append(row)
        batch = CollaborativeSampleExecutor(adapter).execute(
            source_rows,
            context={
                "candidate_id": CANDIDATE_ID,
                "dataset_digest": "dataset:fixed",
            },
            target_bounds={
                "soil_water": {"unit": "fraction", "minimum": 0.0, "maximum": 1.0}
            },
            algorithm_id="ridge-as-agent-tool",
            algorithm_version="1",
            policy=SampleExecutionPolicy(max_attempts=1, retry_backoff_seconds=0),
            result_callback=record,
            result_batch_size=1,
        )
        self.assertEqual([[row["sample_index"] for row in item] for item in published], [[1], [2]])
        self.assertEqual([row["sample_index"] for row in batch.scoring_rows], [1, 2])


class SampleResultDirectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.director = EvolutionDirector(self.ledger)
        self.proposal, self.candidate = _seed_candidate(self.director, RUN_ID)

    def tearDown(self) -> None:
        self.ledger.close()

    def _start(self, revision: str) -> None:
        self.director.start_evaluation_sample_results(
            RUN_ID,
            generation=0,
            proposal_id=self.proposal.proposal_id,
            candidate_id=CANDIDATE_ID,
            revision=revision,
        )

    def _batch(self, revision: str, batch_index: int, indices: tuple[int, ...]):
        rows = build_sample_results(CANDIDATE_ID, [_source_row(index) for index in indices])
        payload = sample_result_batch_event_payload(
            RUN_ID,
            CANDIDATE_ID,
            rows,
            revision=revision,
            batch_index=batch_index,
        )
        return rows, self.director.record_evaluation_sample_result_batch(RUN_ID, payload)

    def test_latest_revision_fences_old_writer_and_restart_resumes_idempotently(self) -> None:
        self._start("old")
        old_rows, _event = self._batch("old", 1, (1,))
        self._start("new")
        old_payload = sample_result_batch_event_payload(
            RUN_ID, CANDIDATE_ID, old_rows, revision="old", batch_index=1
        )
        with self.assertRaisesRegex(ValueError, "superseded"):
            self.director.record_evaluation_sample_result_batch(RUN_ID, old_payload)

        first_rows, first_event = self._batch("new", 1, (1,))
        restarted = EvolutionDirector(self.ledger)
        first_payload = sample_result_batch_event_payload(
            RUN_ID, CANDIDATE_ID, first_rows, revision="new", batch_index=1
        )
        self.assertEqual(
            restarted.record_evaluation_sample_result_batch(RUN_ID, first_payload),
            first_event,
        )
        second_rows = build_sample_results(CANDIDATE_ID, [_source_row(2)])
        restarted.record_evaluation_sample_result_batch(
            RUN_ID,
            sample_result_batch_event_payload(
                RUN_ID, CANDIDATE_ID, second_rows, revision="new", batch_index=2
            ),
        )

    def test_cross_run_payload_and_cross_batch_duplicate_are_rejected_before_append(self) -> None:
        self._start("revision:shared")
        first_rows, _event = self._batch("revision:shared", 1, (1,))
        count = self.ledger.count(RUN_ID)
        with self.assertRaisesRegex(ValueError, "duplicate sample"):
            self.director.record_evaluation_sample_result_batch(
                RUN_ID,
                sample_result_batch_event_payload(
                    RUN_ID,
                    CANDIDATE_ID,
                    first_rows,
                    revision="revision:shared",
                    batch_index=2,
                ),
            )
        self.assertEqual(self.ledger.count(RUN_ID), count)

        other_run = "run:sample-contract-other"
        other_director = EvolutionDirector(self.ledger)
        other_proposal, _candidate = _seed_candidate(
            other_director, other_run, CANDIDATE_ID
        )
        other_director.start_evaluation_sample_results(
            other_run,
            generation=0,
            proposal_id=other_proposal.proposal_id,
            candidate_id=CANDIDATE_ID,
            revision="revision:shared",
        )
        payload = sample_result_batch_event_payload(
            RUN_ID,
            CANDIDATE_ID,
            first_rows,
            revision="revision:shared",
            batch_index=1,
        )
        with self.assertRaisesRegex(ValueError, "another run"):
            other_director.record_evaluation_sample_result_batch(other_run, payload)

    def test_completion_is_atomic_order_independent_and_exactly_idempotent(self) -> None:
        self._start("revision:complete")
        first, _ = self._batch("revision:complete", 1, (1,))
        second, _ = self._batch("revision:complete", 2, (2,))
        evaluation = Evaluation(
            evaluation_id="evaluation:complete",
            run_id=RUN_ID,
            candidate_id=CANDIDATE_ID,
            score=0.2,
            passed=True,
            partition="validation",
        )
        payload = sample_results_event_payload(
            evaluation, [*second, *first], revision="revision:complete"
        )
        self.assertEqual(
            self.director.record_evaluation(evaluation, sample_results=payload),
            evaluation,
        )
        count = self.ledger.count(RUN_ID)
        self.assertEqual(
            self.director.record_evaluation(evaluation, sample_results=payload),
            evaluation,
        )
        self.assertEqual(self.ledger.count(RUN_ID), count)

    def test_open_revision_cannot_be_silently_completed_without_sample_results(self) -> None:
        self._start("revision:open")
        evaluation = Evaluation(
            evaluation_id="evaluation:open",
            run_id=RUN_ID,
            candidate_id=CANDIDATE_ID,
            score=0.0,
            passed=False,
            partition="validation",
        )
        with self.assertRaisesRegex(ValueError, "sealed atomically"):
            self.director.record_evaluation(evaluation)

    def test_matching_checkpoint_resumes_after_sqlite_reopen(self) -> None:
        run_id = "run:sample-checkpoint-restart"
        candidate_id = "candidate:sample-checkpoint-restart"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            ledger = EventLedger(path)
            director = EvolutionDirector(ledger)
            proposal, _candidate = _seed_candidate(
                director, run_id, candidate_id
            )
            prepared = director.prepare_evaluation_sample_checkpoint(
                run_id,
                generation=0,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate_id,
                checkpoint=_checkpoint(),
            )
            rows = build_sample_results(candidate_id, [_source_row(1)])
            director.record_evaluation_sample_result_batch(
                run_id,
                sample_result_batch_event_payload(
                    run_id,
                    candidate_id,
                    rows,
                    revision=prepared["revision"],
                    batch_index=1,
                ),
            )
            director.record_evaluation_progress(
                run_id,
                generation=0,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate_id,
                progress={
                    "role": "planner",
                    "model_id": "strategy-model",
                    "batch_index": 1,
                    "batch_count": 2,
                    "batch_size": 1,
                    "completed_samples": 1,
                    "total_samples": 2,
                    "succeeded_samples": 1,
                    "failed_samples": 0,
                },
            )
            ledger.close()

            restarted_ledger = EventLedger(path)
            try:
                restarted = EvolutionDirector(restarted_ledger)
                resumed = restarted.prepare_evaluation_sample_checkpoint(
                    run_id,
                    generation=0,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate_id,
                    checkpoint=_checkpoint(),
                )
                self.assertTrue(resumed["resumed"])
                self.assertEqual(resumed["revision"], prepared["revision"])
                self.assertEqual(resumed["rows"], rows)
                self.assertEqual(resumed["next_batch_index"], 2)
                self.assertEqual(resumed["progress"]["completed_samples"], 1)
                resume_events = [
                    event
                    for event in restarted.state(run_id).events
                    if event.kind == "EvaluationSampleResultsResumed"
                ]
                self.assertEqual(len(resume_events), 1)
                self.assertEqual(resume_events[0].payload["record_count"], 1)
            finally:
                restarted_ledger.close()

    def test_checkpoint_completion_rejects_missing_rows_without_progress(self) -> None:
        prepared = self.director.prepare_evaluation_sample_checkpoint(
            RUN_ID,
            generation=0,
            proposal_id=self.proposal.proposal_id,
            candidate_id=CANDIDATE_ID,
            checkpoint=_checkpoint(sample_count=2),
        )
        rows, _event = self._batch(prepared["revision"], 1, (1,))
        evaluation = Evaluation(
            evaluation_id="evaluation:checkpoint-incomplete",
            run_id=RUN_ID,
            candidate_id=CANDIDATE_ID,
            score=0.1,
            passed=False,
            partition="validation",
        )
        payload = sample_results_event_payload(
            evaluation,
            rows,
            revision=prepared["revision"],
        )

        with self.assertRaisesRegex(ValueError, "checkpoint cohort"):
            self.director.record_evaluation(evaluation, sample_results=payload)

    def test_checkpoint_completion_rejects_mismatched_v3_progress_total(self) -> None:
        prepared = self.director.prepare_evaluation_sample_checkpoint(
            RUN_ID,
            generation=0,
            proposal_id=self.proposal.proposal_id,
            candidate_id=CANDIDATE_ID,
            checkpoint=_checkpoint(sample_count=2),
        )
        rows, _event = self._batch(prepared["revision"], 1, (1, 2))
        self.director.record_evaluation_progress(
            RUN_ID,
            generation=0,
            proposal_id=self.proposal.proposal_id,
            candidate_id=CANDIDATE_ID,
            progress={
                "role": "planner",
                "model_id": "strategy-model",
                "batch_index": 1,
                "batch_count": 1,
                "batch_size": 2,
                "completed_samples": 2,
                "total_samples": 3,
                "succeeded_samples": 2,
                "failed_samples": 0,
                "gateway_request_count": 1,
                "progress_id": 1,
                "progress_kind": "completed_batch",
                "in_flight_batches": 0,
                "queued_batches": 0,
            },
            revision=prepared["revision"],
        )
        evaluation = Evaluation(
            evaluation_id="evaluation:checkpoint-progress-mismatch",
            run_id=RUN_ID,
            candidate_id=CANDIDATE_ID,
            score=0.2,
            passed=True,
            partition="validation",
        )
        payload = sample_results_event_payload(
            evaluation,
            rows,
            revision=prepared["revision"],
        )

        with self.assertRaisesRegex(ValueError, "progress total"):
            self.director.record_evaluation(evaluation, sample_results=payload)

    def test_checkpoint_completion_ignores_unrevisioned_progress_total(self) -> None:
        prepared = self.director.prepare_evaluation_sample_checkpoint(
            RUN_ID,
            generation=0,
            proposal_id=self.proposal.proposal_id,
            candidate_id=CANDIDATE_ID,
            checkpoint=_checkpoint(sample_count=2),
        )
        rows, _event = self._batch(prepared["revision"], 1, (1, 2))
        self.director.record_evaluation_progress(
            RUN_ID,
            generation=0,
            proposal_id=self.proposal.proposal_id,
            candidate_id=CANDIDATE_ID,
            progress={
                "role": "planner",
                "model_id": "legacy-strategy-model",
                "batch_index": 1,
                "batch_count": 1,
                "batch_size": 2,
                "completed_samples": 2,
                "total_samples": 3,
                "succeeded_samples": 2,
                "failed_samples": 0,
                "gateway_request_count": 1,
            },
        )
        evaluation = Evaluation(
            evaluation_id="evaluation:checkpoint-legacy-progress",
            run_id=RUN_ID,
            candidate_id=CANDIDATE_ID,
            score=0.2,
            passed=True,
            partition="validation",
        )
        payload = sample_results_event_payload(
            evaluation,
            rows,
            revision=prepared["revision"],
        )

        self.assertEqual(
            self.director.record_evaluation(evaluation, sample_results=payload),
            evaluation,
        )

    def test_changed_checkpoint_digests_and_legacy_revision_start_fresh(self) -> None:
        prepared = self.director.prepare_evaluation_sample_checkpoint(
            RUN_ID,
            generation=0,
            proposal_id=self.proposal.proposal_id,
            candidate_id=CANDIDATE_ID,
            checkpoint=_checkpoint(),
        )
        self._batch(prepared["revision"], 1, (1,))

        cohort_changed = self.director.prepare_evaluation_sample_checkpoint(
            RUN_ID,
            generation=0,
            proposal_id=self.proposal.proposal_id,
            candidate_id=CANDIDATE_ID,
            checkpoint=_checkpoint(cohort="c"),
        )
        self.assertFalse(cohort_changed["resumed"])
        self.assertNotEqual(cohort_changed["revision"], prepared["revision"])
        self.assertEqual(cohort_changed["rows"], ())

        execution_changed = self.director.prepare_evaluation_sample_checkpoint(
            RUN_ID,
            generation=0,
            proposal_id=self.proposal.proposal_id,
            candidate_id=CANDIDATE_ID,
            checkpoint=_checkpoint(cohort="c", execution="d"),
        )
        self.assertFalse(execution_changed["resumed"])
        self.assertNotEqual(
            execution_changed["revision"], cohort_changed["revision"]
        )
        self.assertEqual(execution_changed["rows"], ())

        legacy_run = "run:sample-checkpoint-legacy"
        legacy_candidate = "candidate:sample-checkpoint-legacy"
        proposal, _candidate = _seed_candidate(
            self.director, legacy_run, legacy_candidate
        )
        self.director.start_evaluation_sample_results(
            legacy_run,
            generation=0,
            proposal_id=proposal.proposal_id,
            candidate_id=legacy_candidate,
            revision="legacy-revision",
        )
        fresh = self.director.prepare_evaluation_sample_checkpoint(
            legacy_run,
            generation=0,
            proposal_id=proposal.proposal_id,
            candidate_id=legacy_candidate,
            checkpoint=_checkpoint(),
        )
        self.assertFalse(fresh["resumed"])
        self.assertNotEqual(fresh["revision"], "legacy-revision")
        latest_start = [
            event
            for event in self.director.state(legacy_run).events
            if event.kind == "EvaluationSampleResultsStarted"
        ][-1]
        self.assertEqual(
            latest_start.payload["resume_disposition"],
            "legacy_revision_without_checkpoint",
        )


class SampleResultHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.server = EvolutionHTTPServer(
            ("127.0.0.1", 0), Path(self.directory.name) / "events.sqlite3"
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.directory.cleanup()

    def request(self, path: str) -> tuple[int, dict]:
        try:
            with urlopen(Request(self.base + path), timeout=3) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def _candidate(self, run_id: str, candidate_id: str):
        proposal, candidate = _seed_candidate(
            self.server.director, run_id, candidate_id
        )
        return proposal, candidate

    def _sample_path(self, run_id: str, candidate_id: str, suffix: str = "") -> str:
        return (
            f"/api/runs/{quote(run_id, safe='')}/samples?"
            f"candidate_id={quote(candidate_id, safe='')}{suffix}"
        )

    def test_pending_legacy_aborted_and_completed_pagination_contract(self) -> None:
        pending_run = "run:http-pending"
        _proposal, pending_candidate = self._candidate(pending_run, "candidate:http-pending")
        status, pending = self.request(
            self._sample_path(pending_run, pending_candidate.candidate_id)
        )
        self.assertEqual(status, 200)
        self.assertEqual(pending["status"], "pending")
        self.assertIs(pending["supported"], True)
        self.assertIs(pending["complete"], False)

        self.server.director.fail_candidate(
            pending_run, pending_candidate.candidate_id, "legacy terminal candidate"
        )
        status, legacy = self.request(
            self._sample_path(pending_run, pending_candidate.candidate_id)
        )
        self.assertEqual(status, 200)
        self.assertEqual(legacy["status"], "legacy")
        self.assertIs(legacy["legacy"], True)
        self.assertIs(legacy["supported"], False)

        aborted_run = "run:http-aborted"
        proposal, candidate = self._candidate(aborted_run, "candidate:http-aborted")
        director = self.server.director
        director.start_evaluation_sample_results(
            aborted_run,
            generation=0,
            proposal_id=proposal.proposal_id,
            candidate_id=candidate.candidate_id,
            revision="revision:aborted",
        )
        one = build_sample_results(candidate.candidate_id, [_source_row(1)])
        director.record_evaluation_sample_result_batch(
            aborted_run,
            sample_result_batch_event_payload(
                aborted_run,
                candidate.candidate_id,
                one,
                revision="revision:aborted",
                batch_index=1,
            ),
        )
        director.record_evaluation_progress(
            aborted_run,
            generation=0,
            proposal_id=proposal.proposal_id,
            candidate_id=candidate.candidate_id,
            progress={
                "role": "planner",
                "model_id": "test-model",
                "batch_index": 1,
                "batch_count": 2,
                "batch_size": 1,
                "completed_samples": 1,
                "total_samples": 2,
                "succeeded_samples": 1,
                "failed_samples": 0,
            },
        )
        director.fail_candidate(aborted_run, candidate.candidate_id, "stopped")
        status, aborted = self.request(
            self._sample_path(aborted_run, candidate.candidate_id)
        )
        self.assertEqual(status, 200)
        self.assertEqual(aborted["status"], "aborted")
        self.assertEqual(aborted["available_count"], 1)
        self.assertEqual(aborted["expected_count"], 2)
        self.assertIs(aborted["partial"], True)

        completed_run = "run:http-completed"
        proposal, candidate = self._candidate(completed_run, "candidate:http-completed")
        director.start_evaluation_sample_results(
            completed_run,
            generation=0,
            proposal_id=proposal.proposal_id,
            candidate_id=candidate.candidate_id,
            revision="revision:completed",
        )
        first = build_sample_results(candidate.candidate_id, [_source_row(1)])
        second = build_sample_results(candidate.candidate_id, [_source_row(2)])
        for batch_index, rows in ((1, first), (2, second)):
            director.record_evaluation_sample_result_batch(
                completed_run,
                sample_result_batch_event_payload(
                    completed_run,
                    candidate.candidate_id,
                    rows,
                    revision="revision:completed",
                    batch_index=batch_index,
                ),
            )
        evaluation = Evaluation(
            evaluation_id="evaluation:http-completed",
            run_id=completed_run,
            candidate_id=candidate.candidate_id,
            score=0.2,
            passed=True,
            partition="validation",
        )
        director.record_evaluation(
            evaluation,
            sample_results=sample_results_event_payload(
                evaluation, [*second, *first], revision="revision:completed"
            ),
        )
        path = self._sample_path(
            completed_run, candidate.candidate_id, "&offset=0&limit=1"
        )
        status, first_page = self.request(path)
        self.assertEqual(status, 200)
        self.assertEqual(first_page["status"], "completed")
        self.assertEqual(first_page["rows"][0]["sample_index"], 1)
        self.assertEqual(first_page["next_offset"], 1)
        status, second_page = self.request(
            self._sample_path(
                completed_run, candidate.candidate_id, "&offset=1&limit=1"
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(second_page["rows"][0]["sample_index"], 2)
        self.assertIsNone(second_page["next_offset"])

    def test_open_sample_revision_reports_paused_run_state(self) -> None:
        run_id = "run:http-paused"
        proposal, candidate = self._candidate(run_id, "candidate:http-paused")
        director = self.server.director
        director.start_evaluation_sample_results(
            run_id,
            generation=0,
            proposal_id=proposal.proposal_id,
            candidate_id=candidate.candidate_id,
            revision="revision:paused",
        )
        rows = build_sample_results(
            candidate.candidate_id, [_source_row(1, failed=True)]
        )
        director.record_evaluation_sample_result_batch(
            run_id,
            sample_result_batch_event_payload(
                run_id,
                candidate.candidate_id,
                rows,
                revision="revision:paused",
                batch_index=1,
            ),
        )
        director.pause_run(
            run_id,
            reason="operator acceptance checkpoint",
            code="acceptance_checkpoint",
        )

        status, payload = self.request(
            self._sample_path(run_id, candidate.candidate_id)
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "paused")
        self.assertEqual(payload["available_count"], 1)
        self.assertIs(payload["partial"], True)
        self.assertIs(payload["complete"], False)
        failed = payload["rows"][0]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["observed"], 0.5)
        self.assertEqual(failed["baseline"], 0.7)
        self.assertEqual(failed["predicted"], 0.9)
        self.assertAlmostEqual(failed["reward"], -0.2)
        self.assertEqual(failed["attempts"], 3)
        self.assertEqual(failed["failure_class"], "tool_timeout")
        self.assertEqual(
            failed["failure_summary"]["decisions"][0]["role"],
            "host_adjudicator",
        )
        self.assertEqual(
            failed["failure_summary"]["tools"][0]["tool_id"],
            "ridge-tool",
        )

    def test_public_events_and_projection_export_never_include_private_archives(self) -> None:
        run_id = "run:http-private-events"
        proposal, candidate = self._candidate(run_id, "candidate:http-private-events")
        director = self.server.director
        director.start_evaluation_sample_results(
            run_id,
            generation=0,
            proposal_id=proposal.proposal_id,
            candidate_id=candidate.candidate_id,
            revision="revision:private",
        )
        rows = build_sample_results(candidate.candidate_id, [_source_row(1)])
        director.record_evaluation_sample_result_batch(
            run_id,
            sample_result_batch_event_payload(
                run_id,
                candidate.candidate_id,
                rows,
                revision="revision:private",
                batch_index=1,
            ),
        )
        status, events = self.request(
            f"/api/runs/{quote(run_id, safe='')}/events"
        )
        self.assertEqual(status, 200)
        serialized_events = json.dumps(events, ensure_ascii=False)
        for marker in (
            "EvaluationSampleResultsStarted",
            "EvaluationSampleResultBatchRecorded",
            "observed",
            "predicted",
            "reward",
            "archive",
            "sample:1",
        ):
            self.assertNotIn(marker, serialized_events)
        self.assertEqual(
            int(events["next_cursor"]), self.server.ledger.events(run_id)[-1].seq
        )

        exported = run_export(director.state(run_id))
        serialized_export_events = json.dumps(exported["events"], ensure_ascii=False)
        self.assertNotIn("EvaluationSampleResult", serialized_export_events)
        self.assertNotIn("sample:1", serialized_export_events)


if __name__ == "__main__":
    unittest.main()
