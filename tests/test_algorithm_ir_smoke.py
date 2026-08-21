from __future__ import annotations

from dataclasses import replace
import unittest
from types import SimpleNamespace

from ecologyrsi_dsh.api.generation_execution import (
    _ensure_candidate_algorithm_ready,
)
from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import CandidateStatus, Proposal, TaskManifest, digest
from ecologyrsi_dsh.data.toy import ToyCropSoilWater
from ecologyrsi_dsh.dsh import FakeDSHAdapter
from ecologyrsi_dsh.evolution.batches import start_generation_batch
from ecologyrsi_dsh.knowledge.algorithm_smoke import (
    ALGORITHM_SMOKE_VERSION,
    AlgorithmSmokeError,
    smoke_test_algorithm_spec,
)
from ecologyrsi_dsh.knowledge.algorithms import (
    AlgorithmAttempt,
    AlgorithmSpec,
    compile_algorithm_spec,
)
from tests.test_evaluation import (
    DATASET_ID,
    SPLIT_DIGEST,
    _DatasetStub,
    _cohort_series,
    _series,
)


def _toy_task() -> TaskManifest:
    toy = ToyCropSoilWater(seed=0)
    return TaskManifest(
        task_id="algorithm-ir-smoke-toy",
        objective="smoke a restricted toy predictor",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={"max_candidates": 1, "max_generations": 1},
        metadata={
            "domain": "toy",
            "dataset_seed": 0,
            "dataset_digest": toy.dataset_digest,
            "split_manifest_digest": "s" * 64,
            "prediction_model_id": "toy-rolling-water@1",
            "evaluator_id": "toy_time_forward@1",
        },
    )


def _toy_proposal(*, run_id: str = "run:algorithm-ir-smoke") -> Proposal:
    return Proposal(
        proposal_id=f"proposal:{run_id}",
        run_id=run_id,
        generation=0,
        title="restricted toy algorithm",
        changes={"alpha": 0.4, "window": 5, "water_threshold": 0.4},
    )


def _greenhouse_task(predictor_id: str) -> TaskManifest:
    return TaskManifest(
        task_id="algorithm-ir-smoke-greenhouse",
        objective="smoke a registered greenhouse predictor",
        domain_pack="greenhouse_cucumber_2018",
        visible_datasets=(DATASET_ID,),
        budget={"max_candidates": 1},
        metadata={
            "prediction_model_id": predictor_id,
            "evaluator_id": (
                "greenhouse_multihorizon_time_forward@2"
                if predictor_id == "greenhouse-horizon-targetwise-ridge@1"
                else "greenhouse_time_forward@1"
            ),
            "episode_id": f"{DATASET_ID}:TeamA",
            "dataset_digest": "d" * 64,
            "split_manifest_digest": SPLIT_DIGEST,
        },
    )


def _passing_smoke_evidence(spec: AlgorithmSpec) -> dict:
    assert isinstance(spec.algorithm_ir, dict)
    evidence = {
        "schema_version": ALGORITHM_SMOKE_VERSION,
        "status": "passed",
        "source_partition": "training_fit",
        "restricted_partition_access": False,
        "algorithm_ir_digest": spec.algorithm_ir["ir_digest"],
    }
    evidence["smoke_digest"] = digest(evidence)
    return evidence


class AlgorithmIRSmokeTests(unittest.TestCase):
    def test_compiler_emits_registered_ir_and_rejects_operator_tampering(self) -> None:
        task = _toy_task()
        proposal = _toy_proposal()
        spec = compile_algorithm_spec(task, proposal, None)
        self.assertIsNotNone(spec.algorithm_ir)
        assert spec.algorithm_ir is not None
        operators = spec.algorithm_ir["operators"]
        self.assertTrue(operators)
        self.assertTrue(
            all(str(item["operator_id"]).startswith("host.") for item in operators)
        )
        self.assertFalse(
            spec.algorithm_ir["security_boundary"][
                "model_generated_code_execution"
            ]
        )

        tampered = spec.to_dict()
        tampered["algorithm_ir"]["operators"][0]["operator_id"] = (
            "model.generated.python@1"
        )
        with self.assertRaisesRegex(ValueError, "operator graph is not host registered"):
            AlgorithmSpec.from_dict(tampered)

    def test_synthetic_and_observed_smoke_use_training_fit_only(self) -> None:
        toy_task = _toy_task()
        toy_spec = compile_algorithm_spec(toy_task, _toy_proposal(), None)
        toy_evidence = smoke_test_algorithm_spec(toy_spec, toy_task)
        self.assertEqual(toy_evidence["dataset_kind"], "synthetic")
        self.assertEqual(toy_evidence["source_partition"], "training_fit")
        self.assertFalse(toy_evidence["restricted_partition_access"])
        self.assertGreater(toy_evidence["usable_predictions"], 0)

        for predictor_id, changes in (
            (
                "greenhouse-rolling-residual@1",
                {"blend": 0.5, "window": 3, "bias_scale": 0.5},
            ),
            (
                "greenhouse-exogenous-ridge@1",
                {
                    "history_steps": 3,
                    "ridge_alpha": 0.1,
                    "residual_scale": 0.5,
                },
            ),
            (
                "greenhouse-targetwise-ridge@1",
                {
                    "history_steps": 3,
                    "ridge_alpha": 0.1,
                    "air_temperature_residual_scale": 0.8,
                    "relative_humidity_residual_scale": 0.8,
                    "co2_concentration_residual_scale": 0.0,
                },
            ),
            (
                "greenhouse-horizon-targetwise-ridge@1",
                {
                    "history_steps": 3,
                    "ridge_alpha": 0.1,
                    "air_temperature_1h_residual_scale": 0.8,
                    "air_temperature_6h_residual_scale": 0.8,
                    "air_temperature_24h_residual_scale": 0.8,
                    "relative_humidity_1h_residual_scale": 0.8,
                    "relative_humidity_6h_residual_scale": 0.8,
                    "relative_humidity_24h_residual_scale": 0.8,
                    "co2_concentration_1h_residual_scale": 0.0,
                    "co2_concentration_6h_residual_scale": 0.8,
                    "co2_concentration_24h_residual_scale": 0.8,
                },
            ),
        ):
            with self.subTest(predictor_id=predictor_id):
                task = _greenhouse_task(predictor_id)
                proposal = Proposal(
                    proposal_id=f"proposal:smoke:{predictor_id}",
                    run_id=f"run:smoke:{predictor_id}",
                    generation=0,
                    title="observed training-fit smoke",
                    changes=changes,
                )
                spec = compile_algorithm_spec(task, proposal, None)
                if predictor_id == "greenhouse-horizon-targetwise-ridge@1":
                    ordinary_series = _cohort_series()
                    feedback = ordinary_series.partitions["training_feedback"]
                    changed_values = {
                        name: tuple(
                            999999.0 if feedback.start <= index < feedback.end else value
                            for index, value in enumerate(values)
                        )
                        for name, values in ordinary_series.values.items()
                    }
                    changed_feedback_series = replace(
                        ordinary_series, values=changed_values
                    )
                else:
                    ordinary_series = _series(tail_marker=0.0)
                    changed_feedback_series = _series(tail_marker=999999.0)
                ordinary = smoke_test_algorithm_spec(
                    spec,
                    task,
                    _DatasetStub(ordinary_series),
                )
                changed_feedback = smoke_test_algorithm_spec(
                    spec,
                    task,
                    _DatasetStub(changed_feedback_series),
                )
                self.assertEqual(ordinary, changed_feedback)
                self.assertEqual(ordinary["dataset_kind"], "observed")
                self.assertEqual(ordinary["source_partition"], "training_fit")
                self.assertGreater(ordinary["usable_predictions"], 0)

    def test_retryable_smoke_failure_feeds_the_next_attempt(self) -> None:
        calls: list[dict] = []

        def runner(spec, task, proposal, *, attempt, failure_feedback):
            del task, proposal
            calls.append(
                {
                    "attempt": attempt,
                    "failure_feedback": [dict(item) for item in failure_feedback],
                }
            )
            if attempt == 1:
                raise AlgorithmSmokeError(
                    "smoke_remote_tool_timeout",
                    "registered smoke tool timed out",
                    retryable=True,
                    evidence={"tool_stage": "fit"},
                )
            return _passing_smoke_evidence(spec)

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            task = _toy_task()
            director.start_evolution(task, run_id="run:smoke-retry")
            start_generation_batch(director, "run:smoke-retry")
            proposal = director.submit_proposal(
                _toy_proposal(run_id="run:smoke-retry")
            )
            candidate = director.spawn_candidate(
                "run:smoke-retry", proposal, slot_index=0
            )
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=SimpleNamespace(),
                    algorithm_smoke_runner=runner,
                )
            )

            ready = _ensure_candidate_algorithm_ready(
                endpoint,
                director.state("run:smoke-retry"),
                proposal,
                candidate,
            )

            self.assertTrue(ready)
            self.assertEqual([item["attempt"] for item in calls], [1, 2])
            self.assertEqual(
                calls[1]["failure_feedback"][0]["failure_code"],
                "smoke_remote_tool_timeout",
            )
            attempts = director.state("run:smoke-retry").algorithm_attempts_for(
                candidate.candidate_id
            )
            self.assertEqual(
                [(item.phase, item.attempt, item.status) for item in attempts],
                [
                    ("compile", 1, "passed"),
                    ("debug", 1, "failed"),
                    ("debug", 2, "passed"),
                ],
            )

    def test_builtin_smoke_binds_only_transient_retry_feedback(self) -> None:
        task = _toy_task()
        spec = compile_algorithm_spec(task, _toy_proposal(), None)
        feedback = (
            {
                "attempt": 1,
                "failure_code": "smoke_remote_tool_timeout",
                "retryable": True,
                "exception_type": "TimeoutError",
                "public_error": "TimeoutError: queued tool request",
            },
        )

        evidence = smoke_test_algorithm_spec(
            spec,
            task,
            attempt=2,
            failure_feedback=feedback,
        )

        context = evidence["retry_context"]
        self.assertEqual(context["attempt"], 2)
        self.assertEqual(context["prior_failure_count"], 1)
        self.assertEqual(context["prior_failure_digest"], digest(list(feedback)))
        self.assertEqual(
            context["retry_mode"],
            "same_immutable_ir_after_transient_runtime_failure",
        )
        self.assertEqual(context["algorithm_revision_policy"], "new_ir_requires_new_proposal")

        invalid_feedback = ({**feedback[0], "retryable": False},)
        with self.assertRaisesRegex(
            AlgorithmSmokeError,
            "only sequential transient smoke failures",
        ):
            smoke_test_algorithm_spec(
                spec,
                task,
                attempt=2,
                failure_feedback=invalid_feedback,
            )

    def test_tampered_smoke_digest_is_rejected(self) -> None:
        def tampered_runner(spec, task, proposal, *, attempt, failure_feedback):
            del task, proposal, attempt, failure_feedback
            evidence = _passing_smoke_evidence(spec)
            evidence["status"] = "compatibility_skipped"
            return evidence

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            task = _toy_task()
            director.start_evolution(task, run_id="run:smoke-digest-tampered")
            start_generation_batch(director, "run:smoke-digest-tampered")
            proposal = director.submit_proposal(
                _toy_proposal(run_id="run:smoke-digest-tampered")
            )
            candidate = director.spawn_candidate(
                "run:smoke-digest-tampered", proposal, slot_index=0
            )
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=SimpleNamespace(),
                    algorithm_smoke_runner=tampered_runner,
                )
            )

            ready = _ensure_candidate_algorithm_ready(
                endpoint,
                director.state("run:smoke-digest-tampered"),
                proposal,
                candidate,
            )

            self.assertFalse(ready)
            state = director.state("run:smoke-digest-tampered")
            self.assertIs(
                state.candidate(candidate.candidate_id).status,
                CandidateStatus.FAILED,
            )
            debug_attempt = state.algorithm_attempts_for(candidate.candidate_id)[-1]
            self.assertEqual(debug_attempt.status, "failed")
            self.assertEqual(debug_attempt.failure_code, "smoke_digest_mismatch")

    def test_restart_resumes_after_persisted_smoke_failure(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            task = _toy_task()
            director.start_evolution(task, run_id="run:smoke-resume")
            start_generation_batch(director, "run:smoke-resume")
            proposal = director.submit_proposal(
                _toy_proposal(run_id="run:smoke-resume")
            )
            candidate = director.spawn_candidate(
                "run:smoke-resume", proposal, slot_index=0
            )
            spec = compile_algorithm_spec(task, proposal, None)
            director.record_algorithm_attempt(
                AlgorithmAttempt(
                    run_id=candidate.run_id,
                    generation=0,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    phase="compile",
                    attempt=1,
                    status="passed",
                    algorithm_spec_digest=spec.spec_digest,
                    algorithm_spec=spec.to_dict(),
                    evidence={"registered_adapters_only": True},
                )
            )
            persisted_feedback = {
                "attempt": 1,
                "failure_code": "smoke_remote_tool_timeout",
                "retryable": True,
                "exception_type": "AlgorithmSmokeError",
                "public_error": "AlgorithmSmokeError: timeout",
            }
            director.record_algorithm_attempt(
                AlgorithmAttempt(
                    run_id=candidate.run_id,
                    generation=0,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    phase="debug",
                    attempt=1,
                    status="failed",
                    algorithm_spec_digest=spec.spec_digest,
                    evidence={
                        "stage": "training_fit_smoke",
                        "failure_feedback": persisted_feedback,
                    },
                    failure_code="smoke_remote_tool_timeout",
                    public_error="AlgorithmSmokeError: timeout",
                )
            )
            calls = []

            def resumed_runner(
                current_spec,
                task,
                proposal,
                *,
                attempt,
                failure_feedback,
            ):
                del task, proposal
                calls.append((attempt, tuple(failure_feedback)))
                return _passing_smoke_evidence(current_spec)

            restarted = EvolutionDirector(ledger, FakeDSHAdapter())
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=restarted,
                    evaluators=SimpleNamespace(),
                    algorithm_smoke_runner=resumed_runner,
                )
            )
            ready = _ensure_candidate_algorithm_ready(
                endpoint,
                restarted.state("run:smoke-resume"),
                proposal,
                candidate,
            )

            self.assertTrue(ready)
            self.assertEqual(calls[0][0], 2)
            self.assertEqual(
                calls[0][1][0]["failure_code"], "smoke_remote_tool_timeout"
            )

    def test_retry_budget_exhaustion_fails_only_the_candidate(self) -> None:
        def always_timeout(spec, task, proposal, *, attempt, failure_feedback):
            del spec, task, proposal, attempt, failure_feedback
            raise AlgorithmSmokeError(
                "smoke_remote_tool_timeout",
                "registered smoke tool timed out",
                retryable=True,
            )

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            task = _toy_task()
            director.start_evolution(task, run_id="run:smoke-exhausted")
            start_generation_batch(director, "run:smoke-exhausted")
            proposal = director.submit_proposal(
                _toy_proposal(run_id="run:smoke-exhausted")
            )
            candidate = director.spawn_candidate(
                "run:smoke-exhausted", proposal, slot_index=0
            )
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=SimpleNamespace(),
                    algorithm_smoke_runner=always_timeout,
                )
            )

            ready = _ensure_candidate_algorithm_ready(
                endpoint,
                director.state("run:smoke-exhausted"),
                proposal,
                candidate,
            )

            self.assertFalse(ready)
            state = director.state("run:smoke-exhausted")
            self.assertEqual(
                state.candidate(candidate.candidate_id).status,
                CandidateStatus.FAILED,
            )
            debug_attempts = [
                item
                for item in state.algorithm_attempts_for(candidate.candidate_id)
                if item.phase == "debug"
            ]
            self.assertEqual(len(debug_attempts), 3)
            self.assertTrue(all(item.status == "failed" for item in debug_attempts))


if __name__ == "__main__":
    unittest.main()
