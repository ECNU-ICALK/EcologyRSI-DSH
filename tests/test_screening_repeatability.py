"""A/A must resume its own scope and never replace the original race score."""
import base64
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zlib

from ecologyrsi_dsh import EvolutionDirector, FakeDSHAdapter
from ecologyrsi_dsh.application import generation_execution as execution
from ecologyrsi_dsh.application.screening_repeatability import ensure_screening_repeatability
from ecologyrsi_dsh.core.immutable import thaw_json
from ecologyrsi_dsh.core.models import digest, canonical_json
from ecologyrsi_dsh.core.sample_results import build_sample_results
from ecologyrsi_dsh.core.state import project_run_state
from ecologyrsi_dsh.core.trajectory import EvaluationScope, EvaluationPhase
from ecologyrsi_dsh.evaluators import repeatability as aa
from ecologyrsi_dsh.evaluators.sample_execution import SampleExecutionPausedError
from tests import test_evidence_guided_epoch as race


def trace(predictions):
    records = [{"sample_id": f"aa:{i}", "predicted": value, "status": "succeeded"}
               for i, value in enumerate(predictions)]
    return {"encoding": "zlib+base64+canonical-json", "record_count": len(records),
            "trace_digest": digest(records),
            "payload": base64.b64encode(zlib.compress(canonical_json(records).encode())).decode()}


class ScreeningRepeatabilityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = race.EvidenceGuidedRaceTests()
        self.addCleanup(self.fixture.doCleanups)
        original = race._native_task()
        with patch.object(race, "_native_task", return_value=replace(original,
                metadata={**original.metadata, aa.POLICY_KEY: aa.POLICY})):
            self.fixture.setUp()
        self.services = self.fixture.services
        self.run_id = self.fixture.run_id
        self.control = self.fixture.control
        self.metrics = None

    def screen(self, candidates=None):
        for candidate in (candidates if candidates is not None else (*self.fixture.candidates, self.control)):
            self.fixture._screen(candidate, extra_metrics={"sample_execution_trace_archive": trace([1.] * 10)})
        reference = self.fixture.director.state(self.run_id).screening_for(0, self.control.candidate_id)
        if reference is not None:
            self.metrics = thaw_json(reference.payload["metrics"])

    def scope(self):
        cohort = self.fixture.cohorts.screening
        return EvaluationScope(run_id=self.run_id, generation=0, candidate_id=self.control.candidate_id,
            candidate_revision_id=self.fixture.cohorts.revision_bindings[self.control.candidate_id],
            phase=EvaluationPhase.SCREENING, cohort_digest=cohort.cohort_digest,
            origin_count=10, inference_replica=1)

    def revised_metrics(self):
        metrics = deepcopy(self.metrics)
        metrics["objective_score"] = .35
        metrics["targets"][0]["skill_score"] = .35
        metrics["sample_execution_trace_archive"] = trace([1.2] * 10)
        return metrics

    def test_repeat_requires_all_five_sealed_before_any_model_call(self):
        self.screen([self.control, *self.fixture.candidates[:3]])
        evaluator = Mock()
        self.services.evaluators = evaluator
        with self.assertRaisesRegex(ValueError, "all five"):
            ensure_screening_repeatability(self.services, self.run_id, 0)
        evaluator.evaluate_scientific.assert_not_called()
        self.assertIsNone(self.fixture.director.state(self.run_id).formal_selection_for(0))

    def test_diagnostic_has_one_observation_multiplier_and_rejects_tampering(self):
        self.screen()
        state = self.fixture.director.state(self.run_id)
        payload = aa.build_payload(state, self.scope(), self.revised_metrics(), .35)
        summary = payload["summary"]
        self.assertEqual(summary["paired_prediction_count"], 10)
        self.assertEqual(summary["changed_prediction_count"], 10)
        self.assertEqual(summary["independent_observation_count_multiplier"], 1)
        self.assertFalse(summary["selection_score_replaced"])
        aa.validate_payload(state, payload)
        for field in ("reference_record_digest", "score", "summary", "scope"):
            changed = deepcopy(payload)
            if field == "score": changed[field] = .9
            elif field == "summary": changed[field]["selection_score_replaced"] = True
            elif field == "scope": changed[field]["inference_replica"] = 0
            else: changed[field] = "0" * 64
            changed["evidence_digest"] = digest({k:v for k,v in changed.items() if k != "evidence_digest"})
            with self.subTest(field=field), self.assertRaises(ValueError):
                aa.validate_payload(state, changed)
        for key in ("screening_evaluator_digest", "baseline_metrics_digest", "dataset_digest", "comparison_retrieval_snapshot_digest"):
            changed = self.revised_metrics(); changed[key] = "x" * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "inputs differ"):
                aa.build_payload(state, self.scope(), changed, .35)
        changed = self.revised_metrics(); changed["sample_execution_trace_archive"] = trace([1.2] * 9)
        with self.assertRaisesRegex(ValueError, "every frozen"):
            aa.build_payload(state, self.scope(), changed, .35)

    def test_pause_resume_distinct_checkpoint_atomic_seal_and_no_better_repeat_selection(self):
        self.screen()
        reference = self.fixture.director.state(self.run_id).screening_for(0, self.control.candidate_id)
        rows = [{"sample_id": f"aa:{i}", "sample_index": i + 1,
                 "target": "air_temperature", "horizon_hours": 1,
                 "origin_timestamp": i, "target_timestamp": i + 1,
                 "observed": 2., "predicted": 1.2, "baseline": 1.} for i in range(10)]
        restored_counts = []
        def evaluate(_task, candidate, _proposal, *, scope, **kwargs):
            self.assertEqual(scope.inference_replica, 1)
            self.assertNotEqual(scope.scope_key, replace(scope, inference_replica=0).scope_key)
            restored = kwargs["on_sample_checkpoint"]({
                "schema_version": "ecologyrsi-dsh.sample-checkpoint/2",
                "candidate_revision_id": scope.candidate_revision_id, "evaluation_phase": "screening",
                "formal_batch_index": None, "holdout_arm": None, "cohort_digest": scope.cohort_digest,
                "execution_scope_digest": scope.scope_key, "inference_replica": 1,
                "sample_cohort_digest": "a" * 64, "execution_context_digest": "b" * 64,
                "sample_count": 10})
            restored_counts.append(len(restored["rows"]))
            if not restored["rows"]:
                kwargs["on_sample_results"](rows[:3])
                raise SampleExecutionPausedError("paused during A/A")
            kwargs["on_sample_results"](rows[3:])
            return SimpleNamespace(evaluation=SimpleNamespace(score=.35, metrics=self.revised_metrics(),
                                                              evaluator_digest="race-fixture@1"),
                sample_results=build_sample_results(candidate.candidate_id, rows))
        self.services.evaluators = SimpleNamespace(evaluate_scientific=evaluate)
        proposal = SimpleNamespace(proposal_id=self.control.proposal_id)
        with patch.object(execution, "_holdout_replay_inputs", return_value=(self.control, proposal, object())):
            chosen = execution._prepare_formal_finalists(self.services, self.run_id, self.fixture.candidates,
                                                        max_concurrency=1)
            self.assertEqual(chosen, ())
            self.assertIsNone(self.fixture.director.state(self.run_id).formal_selection_for(0))
            self.services.director = EvolutionDirector(self.fixture.ledger, FakeDSHAdapter())
            chosen = execution._prepare_formal_finalists(self.services, self.run_id, self.fixture.candidates,
                                                        max_concurrency=1)
            self.assertEqual([c.candidate_id for c in chosen], [self.fixture.candidates[0].candidate_id])
            self.assertTrue(ensure_screening_repeatability(self.services, self.run_id, 0))
        self.assertEqual(restored_counts, [0, 3])
        state = self.services.director.replay(self.run_id)
        self.assertEqual(state.screening_for(0, self.control.candidate_id), reference)
        events = [e for e in state.events if e.kind == aa.EVENT]
        self.assertEqual(len(events), 1)
        self.assertEqual(aa.summary_for(state, 0)["reference_score"], .2)
        self.assertEqual(aa.summary_for(state, 0)["repeat_score"], .35)
        self.assertTrue(any(e.kind == "EvaluationSampleResultsRecorded" and e.seq == events[0].seq + 1
                            for e in state.events))
        # Altering a sealed diagnostic, even with a newly hashed body, fails replay.
        tampered = thaw_json(events[0].payload); tampered["summary"]["changed_prediction_count"] = 0
        tampered["evidence_digest"] = digest({k:v for k,v in tampered.items() if k != "evidence_digest"})
        with self.assertRaisesRegex(ValueError, "sealed sources"):
            project_run_state(tuple(replace(e, payload=tampered) if e.event_id == events[0].event_id else e for e in state.events))

    def test_selection_cannot_be_sealed_without_repeatability_and_legacy_summary_is_unchanged(self):
        self.screen()
        with patch("ecologyrsi_dsh.application.screening_repeatability.ensure_screening_repeatability", return_value=True):
            with self.assertRaisesRegex(ValueError, "sealed repeatability"):
                execution._prepare_formal_finalists(self.services, self.run_id, self.fixture.candidates,
                                                    max_concurrency=1)
        self.assertIsNone(aa.summary_for(SimpleNamespace(task_manifest=SimpleNamespace(metadata={})), 0))


if __name__ == "__main__":
    unittest.main()
