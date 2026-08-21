from __future__ import annotations

import unittest
from types import SimpleNamespace

from ecologyrsi_dsh import EventLedger, EvolutionDirector, FakeDSHAdapter, TaskManifest
from ecologyrsi_dsh.core.models import (
    Evaluation,
    ModelArtifact,
    Promotion,
    PromotionDecision,
    Proposal,
    digest,
)
from ecologyrsi_dsh.core.sample_results import (
    build_sample_results,
    sample_result_batch_event_payload,
)
from ecologyrsi_dsh.data.toy import ToyCropSoilWater
from ecologyrsi_dsh.reporting import run_completion_outcome, run_summary
from ecologyrsi_dsh.api.generation_execution import _model_token_budget_state
from ecologyrsi_dsh.api.projection import (
    _dsh_runtime_projection,
    _evaluation_progress_rates,
    _model_usage_summary,
    _public_evaluation_metrics,
)
from ecologyrsi_dsh.server import _projection_json


def _task(*, max_candidates: int = 1, max_generations: int = 1) -> TaskManifest:
    return TaskManifest(
        task_id="projection-trace",
        objective="检查逐样本预测轨迹",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={
            "max_candidates": max_candidates,
            "max_generations": max_generations,
            "candidates_per_generation": max_candidates,
        },
        seed=3,
    )


class ExecutionProjectionTests(unittest.TestCase):
    def test_structured_result_with_dsh_session_metrics_replays(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(), run_id="run:projection-dsh-session-metrics"
            ).run.run_id
            structured = {
                "schema_version": "ecology-research-result@1",
                "summary": "bounded evidence",
                "evidence": [],
            }
            ledger.append(
                run_id,
                "DshStructuredResultAccepted",
                {
                    "schema_version": "ecologyrsi-dsh.structured-result-accepted/1",
                    "identity": {
                        "run_id": run_id,
                        "role": "researcher",
                        "stage": "generation.research",
                        "session_id": "dsh-child-replay-1",
                    },
                    "output_schema_id": "ecology-research-result@1",
                    "result_digest": digest(structured),
                    "structured": structured,
                    "session_metrics": {
                        "schema_version": "ecologyrsi-dsh.dsh-session-metrics/1",
                        "session_id": "dsh-child-replay-1",
                        "context_pressure": {
                            "available": True,
                            "source": "dsh_token_meter",
                            "measurement": "current_context_pressure",
                            "log_revision": 27,
                            "baseline_kind": "usage",
                            "total_tokens": 13_073,
                            "surface_tokens": 6_087,
                        },
                        "provider_usage": {
                            "available": True,
                            "source": "dsh_session_projection_token_usage",
                            "measurement": "cumulative_provider_reported_usage",
                            "totals": {
                                "uncached_input_tokens": 6_774,
                                "output_tokens": 680,
                                "cache_read_tokens": 0,
                                "cache_write_tokens": 0,
                                "total_tokens": 7_454,
                            },
                        },
                    },
                },
            )

            state = director.state(run_id)

        self.assertEqual(state.events[-1].kind, "DshStructuredResultAccepted")

    def test_dsh_runtime_projection_aggregates_real_session_usage(self) -> None:
        state = SimpleNamespace(
            run=SimpleNamespace(session_id="dsh-native:run:usage"),
            events=(
                SimpleNamespace(
                    seq=1,
                    kind="DshRuntimeBound",
                    payload={
                        "execution_protocol": "dsh_native_plugin_evolution@1",
                        "capabilities_digest": "a" * 64,
                        "preset_ids": ["ecology-sample-planner-v1"],
                    },
                ),
                SimpleNamespace(
                    seq=2,
                    kind="DshStructuredResultAccepted",
                    payload={
                        "identity": {"session_id": "dsh-child-1"},
                        "session_metrics": {
                            "schema_version": "ecologyrsi-dsh.dsh-session-metrics/1",
                            "session_id": "dsh-child-1",
                            "context_pressure": {
                                "available": True,
                                "total_tokens": 120,
                                "surface_tokens": 80,
                            },
                            "provider_usage": {
                                "available": True,
                                "totals": {
                                    "uncached_input_tokens": 100,
                                    "output_tokens": 20,
                                    "cache_read_tokens": 30,
                                    "cache_write_tokens": 0,
                                    "total_tokens": 150,
                                },
                            },
                        },
                    },
                ),
            ),
        )

        projected = _dsh_runtime_projection(state)

        self.assertTrue(projected["provider_usage"]["available"])
        self.assertEqual(projected["provider_usage"]["total_tokens"], 150)
        self.assertEqual(projected["provider_usage"]["session_count"], 1)
        self.assertTrue(projected["context_pressure"]["available"])
        self.assertEqual(projected["context_pressure"]["maximum_total_tokens"], 120)

    def test_public_metrics_hide_private_sample_and_promotion_evidence(self) -> None:
        public = _public_evaluation_metrics(
            {
                "score": 0.2,
                "sample_execution_records": [{"sample_id": "private"}],
                "sample_execution_trace_archive": "private",
                "promotion_block_evidence": {"blocks": [{"cells": []}]},
            }
        )

        self.assertEqual(public, {"score": 0.2})

    def test_candidate_execution_uses_the_adopted_prediction_model(self) -> None:
        task = TaskManifest(
            task_id="projection-adopted-predictor",
            objective="project the candidate's adopted predictor",
            domain_pack="greenhouse_environment@1",
            visible_datasets=("agc_cucumber_2018",),
            budget={"max_candidates": 1},
            metadata={
                "prediction_model_id": "greenhouse-exogenous-ridge@1",
                "evaluator_id": "greenhouse_multihorizon_time_forward@2",
            },
        )
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                task, run_id="run:projection-adopted-predictor"
            ).run.run_id
            proposal = director.submit_proposal(
                Proposal(
                    proposal_id="proposal:projection-adopted-predictor",
                    run_id=run_id,
                    generation=0,
                    title="adopt a registered predictor",
                    changes={"history_steps": 6},
                    metadata={
                        "prediction_model_adoption": {
                            "status": "adopted",
                            "adopted_id": "greenhouse-horizon-targetwise-ridge@1",
                        }
                    },
                )
            )
            director.spawn_candidate(run_id, proposal)

            projection = _projection_json(director.state(run_id))

        self.assertEqual(
            projection["candidates"][0]["execution"]["prediction_model_id"],
            "greenhouse-horizon-targetwise-ridge@1",
        )

    def test_greenhouse_projection_uses_v2_only_for_missing_evaluator_binding(
        self,
    ) -> None:
        cases = (
            ("default", {}, "greenhouse_multihorizon_time_forward@2"),
            (
                "legacy",
                {"evaluator_id": "greenhouse_multihorizon_time_forward@1"},
                "greenhouse_multihorizon_time_forward@1",
            ),
        )
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            for suffix, metadata, expected_evaluator in cases:
                with self.subTest(suffix=suffix):
                    task = TaskManifest(
                        task_id=f"projection-greenhouse-{suffix}",
                        objective="project the frozen greenhouse evaluator",
                        domain_pack="greenhouse_environment@1",
                        visible_datasets=("agc_cucumber_2018",),
                        budget={"max_candidates": 1},
                        metadata=metadata,
                    )
                    run_id = director.start_evolution(
                        task, run_id=f"run:projection-greenhouse-{suffix}"
                    ).run.run_id
                    proposal = director.submit_proposal(
                        Proposal(
                            proposal_id=f"proposal:projection-greenhouse-{suffix}",
                            run_id=run_id,
                            generation=0,
                            title="greenhouse projection binding",
                            changes={"history_steps": 3},
                        )
                    )
                    director.spawn_candidate(run_id, proposal)

                    projection = _projection_json(director.state(run_id))

                    self.assertEqual(
                        projection["configuration"]["evaluator_id"],
                        expected_evaluator,
                    )
                    self.assertEqual(
                        projection["candidates"][0]["inference_trace"]["evaluator_id"],
                        expected_evaluator,
                    )

    def test_projection_exposes_frozen_sample_agent_configuration(self) -> None:
        remote_task = TaskManifest(
            task_id="projection-sample-agents",
            objective="检查远程样本微批配置",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={
                "max_candidates": 1,
                "token_limit": 20_000,
            },
            seed=7,
            metadata={
                "sample_agent_mode": "gateway_microbatch",
                "sample_agent_batch_size": 128,
                "samples_per_update": 500,
                "sample_concurrency": 6,
                "strategy_model_id": "strategy-model",
                "review_model_id": "review-model",
                "sample_operation_max_tokens": {
                    "sample.planner": 3072,
                    "sample.repair": 3072,
                    "sample.critic": 2048,
                },
                # Existing runs may predate the explicit scope marker while
                # already using the current sample hard-budget policy.
                "sample_token_budget_policy": "hard_gateway_call_reservation@1",
            },
        )
        legacy_task = _task()
        no_budget_task = TaskManifest(
            task_id="projection-sample-policy-without-hard-budget",
            objective="检查仅计量任务不公开硬预算范围",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget=1,
            seed=8,
            metadata={
                "sample_token_budget_policy": "hard_gateway_call_reservation@1",
            },
        )

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            remote_run_id = director.start_evolution(
                remote_task,
                run_id="run:projection-sample-agents",
            ).run.run_id
            legacy_run_id = director.start_evolution(
                legacy_task,
                run_id="run:projection-legacy-sample-agents",
            ).run.run_id
            no_budget_run_id = director.start_evolution(
                no_budget_task,
                run_id="run:projection-sample-policy-without-hard-budget",
            ).run.run_id

            remote_projection = _projection_json(director.state(remote_run_id))
            legacy_projection = _projection_json(director.state(legacy_run_id))
            no_budget_projection = _projection_json(director.state(no_budget_run_id))
            remote = remote_projection["configuration"]
            legacy = legacy_projection["configuration"]

        self.assertEqual(remote["sample_agent_mode"], "gateway_microbatch")
        self.assertEqual(remote["sample_agent_batch_size"], 128)
        self.assertEqual(remote["samples_per_update"], 500)
        self.assertEqual(remote["sample_concurrency"], 6)
        self.assertEqual(remote_projection["samples_per_update"], 500)
        self.assertEqual(remote_projection["sample_agent_batch_size"], 128)
        self.assertEqual(remote_projection["sample_concurrency"], 6)
        self.assertEqual(remote["strategy_model_id"], "strategy-model")
        self.assertEqual(remote["review_model_id"], "review-model")
        self.assertEqual(remote["sample_operation_max_tokens"]["sample.planner"], 3072)
        self.assertEqual(remote["sample_operation_max_tokens"]["sample.critic"], 2048)
        self.assertEqual(
            remote_projection["token_budget_scope"],
            "sample_agent_gateway_calls_only@1",
        )
        self.assertFalse(remote_projection["run_wide_accounting_complete"])
        self.assertEqual(
            remote["token_budget_scope"],
            "sample_agent_gateway_calls_only@1",
        )
        self.assertFalse(remote["run_wide_accounting_complete"])
        self.assertEqual(
            legacy["sample_agent_mode"], "host_feedback_state_machine"
        )
        self.assertIsNone(legacy["sample_agent_batch_size"])
        self.assertIsNone(legacy["samples_per_update"])
        self.assertEqual(
            remote_projection["execution_diagnostics"]["partition_scan_policy"],
            "full_training_fit_rotating_bounded_training_feedback_per_generation",
        )
        self.assertEqual(
            remote_projection["execution_diagnostics"]["samples_per_update"], 500
        )
        self.assertEqual(
            legacy_projection["execution_diagnostics"]["partition_scan_policy"],
            "full_frozen_partition_per_candidate",
        )
        self.assertIsNone(legacy_projection["token_budget_scope"])
        self.assertFalse(legacy_projection["run_wide_accounting_complete"])
        self.assertIsNone(no_budget_projection["token_budget_scope"])
        self.assertFalse(no_budget_projection["run_wide_accounting_complete"])

    def test_projection_aggregates_durable_model_usage_not_static_budget(self) -> None:
        from ecologyrsi_dsh.api.events import EventEndpointsMixin

        usage_task = TaskManifest(
            task_id="projection-model-usage",
            objective="检查真实模型用量账本",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={
                "max_candidates": 1,
                "max_generations": 1,
                "candidates_per_generation": 1,
                # A legacy static estimate must not be projected as actual use.
                "tokens_used": 999_999,
                "token_limit": 2_000,
            },
            seed=9,
        )
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                usage_task, run_id="run:projection-model-usage"
            ).run.run_id
            candidate = director.propose_and_spawn(run_id)
            before = _projection_json(director.state(run_id))
            self.assertFalse(before["token_usage_available"])
            self.assertEqual(before["tokens_used"], 0)
            director.record_model_usage(
                run_id,
                generation=0,
                candidate_id=candidate.candidate_id,
                role="planner",
                model_id="newapi/glm-5.2",
                usage={
                    "prompt_tokens": 101,
                    "completion_tokens": 19,
                    "total_tokens": 120,
                },
                gateway_request_count=2,
                revision="revision:usage",
                usage_index=0,
            )
            director.record_model_usage(
                run_id,
                generation=0,
                candidate_id=candidate.candidate_id,
                role="critic",
                model_id="newapi/deepseek-flash",
                usage={
                    "prompt_tokens": 31,
                    "completion_tokens": 9,
                    "total_tokens": 40,
                },
                gateway_request_count=1,
                revision="revision:usage",
                usage_index=1,
            )
            state = director.replay(run_id)
            projection = _projection_json(state)
            event = next(
                item for item in state.events if item.kind == "ModelUsageRecorded"
            )
            public_event = EventEndpointsMixin._event_json(event)

        self.assertTrue(projection["token_usage_available"])
        self.assertEqual(projection["tokens_used"], 160)
        self.assertEqual(projection["token_limit"], 2_000)
        # A token_limit alone is not enough to relabel a legacy task as using
        # the current sample-agent hard-budget admission policy.
        self.assertIsNone(projection["token_budget_scope"])
        self.assertFalse(projection["run_wide_accounting_complete"])
        self.assertEqual(
            projection["model_usage"],
            {
                "schema_version": "ecologyrsi-dsh.model-usage-summary/1",
                "available": True,
                "call_count": 2,
                "prompt_tokens": 132,
                "completion_tokens": 28,
                "total_tokens": 160,
                "gateway_request_count": 3,
                "by_role": {
                    "critic": {
                        "prompt_tokens": 31,
                        "completion_tokens": 9,
                        "total_tokens": 40,
                        "gateway_request_count": 1,
                        "call_count": 1,
                    },
                    "planner": {
                        "prompt_tokens": 101,
                        "completion_tokens": 19,
                        "total_tokens": 120,
                        "gateway_request_count": 2,
                        "call_count": 1,
                    },
                },
            },
        )
        self.assertEqual(public_event["type"], "model.usage_recorded")
        self.assertEqual(public_event["payload"]["total_tokens"], 120)
        self.assertEqual(public_event["payload"]["model_id"], "newapi/glm-5.2")

    def test_v3_waiting_heartbeat_continues_progress_id_after_restart(self) -> None:
        from ecologyrsi_dsh.api.events import EventEndpointsMixin

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(), run_id="run:progress-v3-restart"
            ).run.run_id
            candidate = director.propose_and_spawn(run_id)
            revision = "revision:progress-v3"
            director.start_evaluation_sample_results(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=revision,
                checkpoint={
                    "schema_version": "ecologyrsi-dsh.sample-checkpoint/1",
                    "cohort_digest": "a" * 64,
                    "execution_context_digest": "b" * 64,
                    "sample_count": 4,
                },
            )
            waiting = {
                "role": "planner",
                "model_id": "newapi/glm-5.2",
                "batch_index": 0,
                "batch_count": 2,
                "batch_size": 0,
                "completed_samples": 0,
                "total_samples": 4,
                "succeeded_samples": 0,
                "failed_samples": 0,
                "gateway_request_count": 0,
                "adaptive_split_trigger_count": 0,
                "adaptive_split_count": 0,
                "adaptive_split_max_depth": 0,
                "adaptive_split_recovered_samples": 0,
                "adaptive_split_failed_samples": 0,
                "progress_id": 1,
                "progress_kind": "waiting",
                "in_flight_batches": 2,
                "queued_batches": 0,
            }
            first = director.record_evaluation_progress(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                progress=waiting,
                revision=revision,
            )
            restarted = EvolutionDirector(ledger, FakeDSHAdapter())
            second = restarted.record_evaluation_progress(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                progress={**waiting, "progress_id": 2, "queued_batches": 1},
                revision=revision,
            )
            projection = _projection_json(restarted.state(run_id))[
                "execution_progress"
            ]["stage_progress"]
            public_event = EventEndpointsMixin._event_json(second)

        self.assertNotEqual(first.seq, second.seq)
        self.assertEqual(projection["progress_id"], 2)
        self.assertEqual(projection["in_flight_batches"], 2)
        self.assertEqual(projection["queued_batches"], 1)
        self.assertEqual(public_event["payload"]["revision"], revision)
        self.assertEqual(public_event["payload"]["progress_kind"], "waiting")
        self.assertEqual(public_event["payload"]["in_flight_batches"], 2)
        self.assertEqual(public_event["payload"]["queued_batches"], 1)

    def test_v3_restart_heartbeat_overrides_larger_prior_attempt_count(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(), run_id="run:progress-v3-restart-count"
            ).run.run_id
            candidate = director.propose_and_spawn(run_id)
            revision = "revision:progress-v3-count"
            director.start_evaluation_sample_results(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=revision,
                checkpoint={
                    "schema_version": "ecologyrsi-dsh.sample-checkpoint/1",
                    "cohort_digest": "a" * 64,
                    "execution_context_digest": "b" * 64,
                    "sample_count": 100,
                },
            )
            completed = {
                "role": "planner",
                "model_id": "newapi/glm-5.2",
                "batch_index": 8,
                "batch_count": 10,
                "batch_size": 9,
                "completed_samples": 72,
                "total_samples": 100,
                "succeeded_samples": 70,
                "failed_samples": 2,
                "gateway_request_count": 8,
                "adaptive_split_trigger_count": 0,
                "adaptive_split_count": 0,
                "adaptive_split_max_depth": 0,
                "adaptive_split_recovered_samples": 0,
                "adaptive_split_failed_samples": 0,
                "progress_id": 1,
                "progress_kind": "completed_batch",
                "in_flight_batches": 2,
                "queued_batches": 0,
            }
            director.record_evaluation_progress(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                progress=completed,
                revision=revision,
            )
            restarted = EvolutionDirector(ledger, FakeDSHAdapter())
            restarted.record_evaluation_progress(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                progress={
                    **completed,
                    "batch_index": 7,
                    "batch_size": 0,
                    "completed_samples": 70,
                    "succeeded_samples": 70,
                    "failed_samples": 0,
                    "progress_id": 2,
                    "progress_kind": "waiting",
                    "in_flight_batches": 3,
                    "queued_batches": 1,
                },
                revision=revision,
            )
            stage_progress = _projection_json(restarted.state(run_id))[
                "execution_progress"
            ]["stage_progress"]

        self.assertEqual(stage_progress["progress_id"], 2)
        self.assertEqual(stage_progress["progress_kind"], "waiting")
        self.assertEqual(stage_progress["completed_samples"], 70)
        self.assertEqual(stage_progress["in_flight_batches"], 3)
        self.assertEqual(stage_progress["queued_batches"], 1)

    def test_v2_usage_summary_charges_missing_call_reservation_once(self) -> None:
        task = TaskManifest(
            task_id="projection-partial-usage",
            objective="检查缺失用量回执的保守计费",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={
                "max_candidates": 1,
                "max_generations": 1,
                "candidates_per_generation": 1,
                "token_limit": 1_000,
                "token_reservation_per_wave": 100,
            },
            seed=3,
        )
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                task, run_id="run:projection-partial-usage"
            ).run.run_id
            candidate = director.propose_and_spawn(run_id)
            revision = "revision:partial-usage"
            director.start_evaluation_sample_results(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=revision,
                checkpoint={
                    "schema_version": "ecologyrsi-dsh.sample-checkpoint/1",
                    "cohort_digest": "a" * 64,
                    "execution_context_digest": "b" * 64,
                    "sample_count": 2,
                },
            )
            director.record_model_usage_batch(
                run_id,
                generation=candidate.generation,
                candidate_id=candidate.candidate_id,
                revision=revision,
                receipts=(
                    {
                        "call_id": "call:reported",
                        "logical_call_digest": "c" * 64,
                        "role": "planner",
                        "model_id": "newapi/glm-5.2",
                        "outcome": "succeeded",
                        "usage_reported": True,
                        "http_attempts": 1,
                        "prompt_tokens": 17,
                        "completion_tokens": 3,
                        "total_tokens": 20,
                    },
                    {
                        "call_id": "call:partial",
                        "logical_call_digest": "d" * 64,
                        "role": "critic",
                        "model_id": "newapi/deepseek-flash",
                        "outcome": "failed",
                        "usage_reported": False,
                        "http_attempts": 1,
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "total_tokens": 6,
                    },
                ),
            )
            state = director.state(run_id)
            projection = _projection_json(state)
            usage = projection["model_usage"]
            runtime_budget = _model_token_budget_state(state)

        self.assertTrue(usage["available"])
        self.assertFalse(usage["complete"])
        self.assertEqual(usage["reported_call_count"], 1)
        self.assertEqual(usage["missing_call_count"], 1)
        # Raw diagnostics retain the provider-side estimate, but budget
        # accounting replaces it with the frozen reservation rather than
        # charging both values.
        self.assertEqual(usage["total_tokens"], 26)
        self.assertEqual(usage["budget_accounted_tokens"], 120)
        self.assertEqual(usage["missing_call_reservation"], 100)
        self.assertEqual(projection["tokens_used"], 120)
        self.assertEqual(runtime_budget["tokens_used"], 120)

    def test_v2_usage_receipts_without_reported_tokens_remain_visible(self) -> None:
        state = SimpleNamespace(
            events=(
                SimpleNamespace(
                    kind="ModelUsageRecorded",
                    payload={
                        "schema_version": "ecologyrsi-dsh.model-usage/2",
                        "candidate_id": "candidate:partial",
                        "revision": "revision:partial",
                        "role": "planner",
                        "logical_call_digest": "a" * 64,
                        "outcome": "failed",
                        "usage_reported": False,
                        "http_attempts": 2,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                ),
            )
        )

        usage = _model_usage_summary(state)

        self.assertTrue(usage["available"])
        self.assertFalse(usage["complete"])
        self.assertEqual(usage["missing_call_count"], 1)
        self.assertEqual(usage["total_tokens"], 0)
        self.assertEqual(usage["physical_call_count"], 1)
        self.assertEqual(usage["logical_call_count"], 1)
        self.assertEqual(usage["replayed_call_count"], 0)
        self.assertEqual(usage["outcome_counts"], {"failed": 1})

    def test_v2_usage_summary_separates_logical_replays_from_legacy_calls(
        self,
    ) -> None:
        candidate_id = "candidate:usage-scope"
        revision = "revision:usage-scope"

        def usage_event(payload: dict[str, object]) -> SimpleNamespace:
            return SimpleNamespace(kind="ModelUsageRecorded", payload=payload)

        common = {
            "schema_version": "ecologyrsi-dsh.model-usage/2",
            "candidate_id": candidate_id,
            "revision": revision,
            "role": "planner",
            "usage_reported": True,
            "http_attempts": 1,
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
        }
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(token_reservation_per_wave=100),
            events=(
                usage_event(
                    {
                        **common,
                        "logical_call_digest": "a" * 64,
                        "outcome": "failed",
                    }
                ),
                usage_event(
                    {
                        **common,
                        "logical_call_digest": "a" * 64,
                        "outcome": "succeeded",
                    }
                ),
                usage_event(
                    {
                        **common,
                        "logical_call_digest": "b" * 64,
                        "outcome": "succeeded",
                    }
                ),
                usage_event(
                    {
                        "schema_version": "ecologyrsi-dsh.model-usage/1",
                        "candidate_id": candidate_id,
                        "revision": revision,
                        "role": "critic",
                        "gateway_request_count": 2,
                        "prompt_tokens": 16,
                        "completion_tokens": 4,
                        "total_tokens": 20,
                    }
                ),
            ),
        )

        usage = _model_usage_summary(state)

        self.assertEqual(usage["schema_version"], "ecologyrsi-dsh.model-usage-summary/2")
        self.assertEqual(usage["call_count"], 4)
        self.assertEqual(usage["physical_call_count"], 3)
        self.assertEqual(usage["logical_call_count"], 2)
        self.assertEqual(usage["replayed_call_count"], 1)
        self.assertEqual(usage["outcome_counts"], {"failed": 1, "succeeded": 2})
        self.assertEqual(usage["total_tokens"], 50)
        self.assertEqual(usage["budget_accounted_tokens"], 50)
        self.assertEqual(usage["scope_candidate_id"], candidate_id)
        self.assertEqual(usage["scope_revision"], revision)

    def test_v2_progress_rejects_batch_larger_than_completed_samples(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(), run_id="run:projection-v2-progress-validation"
            ).run.run_id
            candidate = director.propose_and_spawn(run_id)
            with self.assertRaisesRegex(ValueError, "batch size"):
                director.record_evaluation_progress(
                    run_id,
                    generation=candidate.generation,
                    proposal_id=candidate.proposal_id,
                    candidate_id=candidate.candidate_id,
                    progress={
                        "role": "planner",
                        "model_id": "newapi/glm-5.2",
                        "batch_index": 1,
                        "batch_count": 2,
                        "batch_size": 2,
                        "completed_samples": 1,
                        "total_samples": 4,
                        "succeeded_samples": 1,
                        "failed_samples": 0,
                    },
                )

    def test_failed_run_exposes_reason_and_latest_failed_stage(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(max_candidates=1), run_id="run:projection-failed"
            ).run.run_id
            director.record_evolution_stage(
                run_id,
                generation=0,
                stage="proposal",
                status="failed",
                public_error="TimeoutError: strategy request timed out",
            )
            director.fail_run(run_id, "remote strategy API timed out")

            projection = _projection_json(director.state(run_id))

            self.assertEqual(projection["status"], "failed")
            self.assertEqual(
                projection["failure_reason"], "remote strategy API timed out"
            )
            self.assertEqual(projection["failed_stage"]["generation"], 0)
            self.assertEqual(projection["failed_stage"]["stage"], "proposal")
            self.assertIn("TimeoutError", projection["failed_stage"]["public_error"])

    def test_partial_projection_exposes_stage_evidence_and_safe_plan(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(max_candidates=1), run_id="run:projection-partial"
            ).run.run_id
            proposal = director.submit_proposal(
                Proposal(
                    proposal_id="proposal:projection-partial",
                    run_id=run_id,
                    generation=0,
                    title="局部参数方案",
                    changes={"alpha": 0.4},
                    metadata={
                        "plan": {"strategy": {"name": "局部搜索"}},
                        "api_key": "should-not-be-visible",
                        "apiKey": "should-not-be-visible",
                        "private_reasoning": "should-not-be-visible",
                    },
                )
            )
            candidate = director.spawn_candidate(run_id, proposal)
            director.record_evolution_stage(
                run_id,
                generation=0,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
                stage="training",
                status="started",
            )

            projection = _projection_json(director.state(run_id))
            candidate_view = projection["candidates"][0]
            self.assertEqual(candidate_view["execution"]["current_stage"], "training")
            self.assertEqual(candidate_view["execution"]["evidence"], "stage_events")
            self.assertEqual(candidate_view["inference_trace"]["status"], "pending")
            self.assertEqual(projection["execution_progress"]["phase"], "training")
            self.assertGreater(projection["execution_progress"]["progress_percent"], 0)
            self.assertLess(projection["execution_progress"]["progress_percent"], 100)
            self.assertNotIn("api_key", str(candidate_view["model_plan"]))
            self.assertNotIn("private_reasoning", str(candidate_view["model_plan"]))

    def test_projection_uses_durable_sample_microbatch_progress(self) -> None:
        from ecologyrsi_dsh.api.events import EventEndpointsMixin

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(max_candidates=1), run_id="run:projection-microbatch-progress"
            ).run.run_id
            proposal = director.submit_proposal(
                Proposal(
                    proposal_id="proposal:projection-microbatch-progress",
                    run_id=run_id,
                    generation=0,
                    title="真实样本微批",
                    changes={"alpha": 0.4},
                )
            )
            candidate = director.spawn_candidate(run_id, proposal)
            director.record_evolution_stage(
                run_id,
                generation=0,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
                stage="training",
                status="completed",
            )
            director.record_evolution_stage(
                run_id,
                generation=0,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
                stage="evaluation",
                status="started",
            )
            director.record_evaluation_progress(
                run_id,
                generation=0,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
                progress={
                    "role": "planner",
                    "model_id": "strategy-model",
                    "batch_index": 3,
                    "batch_count": 6,
                    "batch_size": 128,
                    "completed_samples": 384,
                    "total_samples": 768,
                    "succeeded_samples": 380,
                    "failed_samples": 4,
                    "gateway_request_count": 9,
                    "adaptive_split_trigger_count": 3,
                    "adaptive_split_count": 3,
                    "adaptive_split_max_depth": 2,
                    "adaptive_split_recovered_samples": 120,
                    "adaptive_split_failed_samples": 4,
                },
            )

            projection = _projection_json(director.state(run_id))
            progress_event = next(
                event
                for event in director.state(run_id).events
                if event.kind == "EvaluationProgressRecorded"
            )
            public_event = EventEndpointsMixin._event_json(progress_event)

            artifact = ModelArtifact(
                artifact_id="artifact:projection-microbatch-progress",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                model_id="toy-rolling-water@1",
                dataset_digest="dataset:projection",
                training_partition="train",
                training_rows=10,
                metrics={
                    "training_partition_rows": 10,
                    "training_eligible_examples": 10,
                    "training_used_examples": 10,
                    "training_skipped_examples": 0,
                    "fit_passes_completed": 1,
                },
            )
            director.record_artifact(artifact)
            director.record_evaluation(
                Evaluation(
                    evaluation_id="evaluation:projection-microbatch-progress",
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    score=0.7,
                    passed=True,
                    partition="validation",
                    metrics={
                        "evaluation_partition_rows": 760,
                        "evaluation_eligible_examples": 768,
                        "evaluation_used_examples": 760,
                        "evaluation_skipped_examples": 8,
                    },
                    artifact_digest=artifact.digest,
                )
            )
            finalized_projection = _projection_json(director.state(run_id))

        candidate_progress = projection["candidates"][0]["execution"]
        run_progress = projection["execution_progress"]
        self.assertEqual(candidate_progress["stage_progress"]["batch_index"], 3)
        self.assertEqual(run_progress["stage_progress"]["completed_samples"], 384)
        self.assertEqual(run_progress["stage_progress"]["progress_percent"], 50.0)
        self.assertEqual(run_progress["stage_progress"]["gateway_request_count"], 9)
        self.assertEqual(run_progress["stage_progress"]["adaptive_split_count"], 3)
        self.assertEqual(
            run_progress["stage_progress"]["causal_wave_sample_count"], 128
        )
        self.assertEqual(
            run_progress["stage_progress"]["configured_concurrency"], 4
        )
        self.assertEqual(
            run_progress["stage_progress"]["adaptive_split_recovered_samples"],
            120,
        )
        self.assertEqual(public_event["payload"]["adaptive_split_max_depth"], 2)
        self.assertEqual(public_event["payload"]["adaptive_split_failed_samples"], 4)
        self.assertEqual(run_progress["last_event_at"], candidate_progress["stage_progress"]["updated_at"])
        self.assertGreater(run_progress["progress_percent"], 50.0)
        self.assertLess(run_progress["progress_percent"], 66.7)
        self.assertNotIn("observed", str(run_progress["stage_progress"]))
        self.assertNotIn("sample_id", str(run_progress["stage_progress"]))
        live_diagnostics = projection["execution_diagnostics"]
        self.assertEqual(live_diagnostics["evaluation_used_examples"], 0)
        self.assertEqual(live_diagnostics["live_evaluation_completed_examples"], 384)
        self.assertEqual(live_diagnostics["live_evaluation_total_examples"], 768)
        self.assertEqual(live_diagnostics["candidate_work_items"], 384)
        self.assertEqual(live_diagnostics["execution_evidence_status"], "partial_live")
        finalized_diagnostics = finalized_projection["execution_diagnostics"]
        self.assertEqual(finalized_diagnostics["evaluation_used_examples"], 760)
        self.assertEqual(finalized_diagnostics["live_evaluation_completed_examples"], 0)
        self.assertEqual(finalized_diagnostics["candidate_work_items"], 770)
        self.assertEqual(finalized_diagnostics["execution_evidence_status"], "recorded")

    def test_diagnostics_fall_back_to_active_result_batches_and_relabel_terminal_evidence(
        self,
    ) -> None:
        def source_row(index: int) -> dict[str, object]:
            return {
                "sample_index": index,
                "sample_id": f"sample:{index}",
                "target": "soil_water",
                "unit": "fraction",
                "horizon_hours": 24,
                "origin_timestamp": index,
                "target_timestamp": index + 1,
                "observed": 0.5,
                "predicted": 0.45,
                "baseline": 0.7,
                "sample_execution_status": "succeeded",
                "sample_execution_attempts": 1,
                "sample_execution_retry_count": 0,
            }

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(max_candidates=2), run_id="run:projection-batch-fallback"
            ).run.run_id
            proposal = director.submit_proposal(
                Proposal(
                    proposal_id="proposal:projection-batch-fallback",
                    run_id=run_id,
                    generation=0,
                    title="持久批次回退",
                    changes={"alpha": 0.4},
                )
            )
            candidate = director.spawn_candidate(run_id, proposal)
            revision = "revision:projection-batch-fallback"
            director.start_evaluation_sample_results(
                run_id,
                generation=0,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=revision,
                checkpoint={
                    "schema_version": "ecologyrsi-dsh.sample-checkpoint/1",
                    "cohort_digest": "a" * 64,
                    "execution_context_digest": "b" * 64,
                    "sample_count": 2,
                },
            )

            def record_batch(batch_index: int) -> None:
                rows = build_sample_results(
                    candidate.candidate_id, [source_row(batch_index)]
                )
                director.record_evaluation_sample_result_batch(
                    run_id,
                    sample_result_batch_event_payload(
                        run_id,
                        candidate.candidate_id,
                        rows,
                        revision=revision,
                        batch_index=batch_index,
                    ),
                )

            record_batch(1)
            director.record_evaluation_progress(
                run_id,
                generation=0,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=revision,
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
                    "progress_id": 1,
                    "progress_kind": "completed_batch",
                    "in_flight_batches": 0,
                    "queued_batches": 0,
                },
            )
            # The second host-finalized batch survives even if its progress
            # heartbeat cannot be appended.
            record_batch(2)
            running = _projection_json(director.state(run_id))["execution_diagnostics"]
            director.pause_run(run_id, reason="operator pause")
            paused = _projection_json(director.state(run_id))["execution_diagnostics"]
            director.resume_run(run_id)
            director.fail_candidate(run_id, candidate.candidate_id, "evaluation stopped")
            failed = _projection_json(director.state(run_id))["execution_diagnostics"]
            next_proposal = director.submit_proposal(
                Proposal(
                    proposal_id="proposal:projection-batch-fallback:next",
                    run_id=run_id,
                    generation=0,
                    title="后续候选",
                    changes={"alpha": 0.5},
                )
            )
            next_candidate = director.spawn_candidate(run_id, next_proposal)
            director.record_evaluation_progress(
                run_id,
                generation=0,
                proposal_id=next_proposal.proposal_id,
                candidate_id=next_candidate.candidate_id,
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
            mixed = _projection_json(director.state(run_id))["execution_diagnostics"]
            director.cancel_run(run_id, "cancel remaining work")
            cancelled = _projection_json(director.state(run_id))["execution_diagnostics"]

        self.assertEqual(running["evaluation_used_examples"], 0)
        self.assertEqual(running["live_evaluation_completed_examples"], 2)
        self.assertEqual(running["live_evaluation_total_examples"], 2)
        self.assertEqual(running["candidate_work_items"], 2)
        self.assertIsNone(running["live_evaluation_succeeded_examples"])
        self.assertEqual(
            running["partial_evaluation_sources"],
            ["durable_sample_result_batches"],
        )
        self.assertEqual(running["execution_evidence_status"], "partial_live")
        self.assertEqual(paused["execution_evidence_status"], "retained_partial")
        self.assertEqual(failed["execution_evidence_status"], "aborted_partial")
        self.assertEqual(mixed["execution_evidence_status"], "mixed_partial")
        self.assertEqual(mixed["live_evaluation_completed_examples"], 3)
        self.assertEqual(mixed["partial_evaluation_active_candidate_count"], 1)
        self.assertEqual(mixed["partial_evaluation_aborted_candidate_count"], 1)
        self.assertEqual(cancelled["execution_evidence_status"], "aborted_partial")

    def test_stopped_runs_do_not_project_unfinished_candidates_as_running(
        self,
    ) -> None:
        cases = (
            ("paused", "paused", "paused", "retained_partial"),
            ("cancelled", "cancelled", "not_recorded", "aborted_partial"),
            ("failed", "failed", "not_recorded", "aborted_partial"),
            ("completed", "completed", "not_recorded", "aborted_partial"),
        )
        for label, expected_phase, expected_stage, expected_evidence in cases:
            with self.subTest(status=label), EventLedger() as ledger:
                director = EvolutionDirector(ledger, FakeDSHAdapter())
                run_id = f"run:stopped-stage-projection:{label}"
                director.start_evolution(_task(), run_id=run_id)
                candidate = director.propose_and_spawn(run_id)
                director.record_evolution_stage(
                    run_id,
                    generation=0,
                    proposal_id=candidate.proposal_id,
                    candidate_id=candidate.candidate_id,
                    stage="evaluation",
                    status="started",
                )
                director.record_evaluation_progress(
                    run_id,
                    generation=0,
                    proposal_id=candidate.proposal_id,
                    candidate_id=candidate.candidate_id,
                    progress={
                        "role": "planner",
                        "model_id": "projection-test-model",
                        "batch_index": 1,
                        "batch_count": 2,
                        "batch_size": 1,
                        "completed_samples": 1,
                        "total_samples": 2,
                        "succeeded_samples": 1,
                        "failed_samples": 0,
                    },
                )
                live = _projection_json(director.state(run_id))
                self.assertEqual(live["execution_progress"]["phase"], "evaluation")
                self.assertEqual(
                    live["candidates"][0]["execution"]["stages"]["evaluation"],
                    "running",
                )

                if label == "paused":
                    director.pause_run(run_id, reason="operator pause")
                elif label == "cancelled":
                    director.cancel_run(run_id, "operator cancel")
                elif label == "failed":
                    director.fail_run(run_id, "runtime failed")
                else:
                    director.complete_run(run_id)
                projection = _projection_json(director.state(run_id))

                candidate_view = projection["candidates"][0]
                expected_candidate_status = (
                    "paused" if label == "paused" else "aborted"
                )
                self.assertEqual(candidate_view["status"], expected_candidate_status)
                self.assertEqual(
                    candidate_view["execution"]["status"],
                    expected_candidate_status,
                )
                self.assertEqual(
                    candidate_view["execution"]["stages"]["evaluation"],
                    expected_stage,
                )
                self.assertNotIn(
                    "running",
                    candidate_view["execution"]["stages"].values(),
                )
                self.assertEqual(
                    projection["execution_progress"]["phase"],
                    expected_phase,
                )
                self.assertEqual(
                    projection["execution_progress"]["current_stage"],
                    "evaluation" if label == "paused" else None,
                )
                self.assertEqual(
                    projection["execution_diagnostics"]["execution_evidence_status"],
                    expected_evidence,
                )
                round_view = projection["rounds"][0]
                self.assertNotIn("running", round_view["stages"].values())
                self.assertNotIn(
                    "running",
                    round_view["candidates"][0]["stages"].values(),
                )
                if label == "paused":
                    director.resume_run(run_id)
                    resumed = _projection_json(director.state(run_id))
                    self.assertEqual(resumed["execution_progress"]["phase"], "evaluation")
                    self.assertEqual(resumed["candidates"][0]["status"], "evaluating")
                    self.assertEqual(
                        resumed["candidates"][0]["execution"]["stages"][
                            "evaluation"
                        ],
                        "running",
                    )

    def test_recent_evaluation_progress_rates_use_a_bounded_window(self) -> None:
        events = [
            SimpleNamespace(
                seq=index,
                created_at=f"2026-08-19T00:{index:02d}:00+00:00",
                payload={
                    "completed_samples": index * 10,
                    "gateway_request_count": index * 2,
                },
            )
            for index in range(1, 13)
        ]

        sample_rate, gateway_rate = _evaluation_progress_rates(
            events, events[-1]
        )

        self.assertEqual(sample_rate, 10.0)
        self.assertEqual(gateway_rate, 2.0)

    def test_evaluation_progress_rates_accept_a_short_history(self) -> None:
        events = [
            SimpleNamespace(
                seq=index,
                created_at=f"2026-08-19T00:00:0{index}+00:00",
                payload={
                    "completed_samples": index,
                    "gateway_request_count": index,
                },
            )
            for index in range(1, 4)
        ]

        sample_rate, gateway_rate = _evaluation_progress_rates(
            events, events[-1]
        )

        self.assertEqual(sample_rate, 60.0)
        self.assertEqual(gateway_rate, 60.0)

    def test_progress_uses_latest_sample_result_revision_after_recovery(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(max_candidates=1), run_id="run:projection-latest-revision"
            ).run.run_id
            proposal = director.submit_proposal(
                Proposal(
                    proposal_id="proposal:projection-latest-revision",
                    run_id=run_id,
                    generation=0,
                    title="重启后进度",
                    changes={"alpha": 0.4},
                )
            )
            candidate = director.spawn_candidate(run_id, proposal)

            def start_revision(revision: str) -> None:
                director.start_evaluation_sample_results(
                    run_id,
                    generation=0,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    revision=revision,
                )

            def record_progress(completed: int) -> None:
                director.record_evaluation_progress(
                    run_id,
                    generation=0,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    progress={
                        "role": "planner",
                        "model_id": "strategy-model",
                        "batch_index": completed // 9,
                        "batch_count": 802,
                        "batch_size": 9,
                        "completed_samples": completed,
                        "total_samples": 7125,
                        "succeeded_samples": completed,
                        "failed_samples": 0,
                    },
                )

            start_revision("revision:before-restart")
            record_progress(306)
            start_revision("revision:after-restart")
            record_progress(36)

            projection = _projection_json(director.state(run_id))

        stage_progress = projection["candidates"][0]["execution"]["stage_progress"]
        self.assertEqual(stage_progress["completed_samples"], 36)
        self.assertEqual(stage_progress["batch_index"], 4)

    def test_projection_exposes_isolated_legacy_revision_aggregate(self) -> None:
        """An isolated revision is visible without reviving its counters."""

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(max_candidates=1), run_id="run:projection-superseded-revision"
            ).run.run_id
            candidate = director.propose_and_spawn(run_id)
            director.record_evolution_stage(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                stage="evaluation",
                status="started",
            )
            legacy_revision = "revision:legacy-uncheckpointed"
            director.start_evaluation_sample_results(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=legacy_revision,
            )
            director.record_evaluation_progress(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                progress={
                    "role": "planner",
                    "model_id": "newapi/glm-5.2",
                    "batch_index": 8,
                    "batch_count": 12,
                    "batch_size": 9,
                    "completed_samples": 72,
                    "total_samples": 100,
                    "succeeded_samples": 70,
                    "failed_samples": 2,
                },
            )
            active_revision = "revision:checkpointed-replacement"
            director.start_evaluation_sample_results(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=active_revision,
                checkpoint={
                    "schema_version": "ecologyrsi-dsh.sample-checkpoint/1",
                    "cohort_digest": "a" * 64,
                    "execution_context_digest": "b" * 64,
                    "sample_count": 100,
                },
                supersedes_revision=legacy_revision,
                resume_disposition="legacy_revision_without_checkpoint",
            )
            director.record_evaluation_progress(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=active_revision,
                progress={
                    "role": "planner",
                    "model_id": "newapi/glm-5.2",
                    "batch_index": 0,
                    "batch_count": 12,
                    "batch_size": 0,
                    "completed_samples": 0,
                    "total_samples": 100,
                    "succeeded_samples": 0,
                    "failed_samples": 0,
                    "gateway_request_count": 0,
                    "adaptive_split_trigger_count": 0,
                    "adaptive_split_count": 0,
                    "adaptive_split_max_depth": 0,
                    "adaptive_split_recovered_samples": 0,
                    "adaptive_split_failed_samples": 0,
                    "progress_id": 1,
                    "progress_kind": "waiting",
                    "in_flight_batches": 2,
                    "queued_batches": 0,
                },
            )

            projection = _projection_json(director.state(run_id))

        candidate_execution = projection["candidates"][0]["execution"]
        active_progress = candidate_execution["stage_progress"]
        superseded = candidate_execution["superseded_sample_revision"]
        self.assertEqual(active_progress["revision"], active_revision)
        self.assertEqual(active_progress["completed_samples"], 0)
        self.assertEqual(superseded["revision"], legacy_revision)
        self.assertEqual(superseded["resume_disposition"], "legacy_revision_without_checkpoint")
        self.assertEqual(superseded["completed_samples"], 72)
        self.assertEqual(superseded["succeeded_samples"], 70)
        self.assertEqual(superseded["failed_samples"], 2)
        self.assertEqual(superseded["total_samples"], 100)
        self.assertEqual(
            projection["execution_progress"]["superseded_sample_revision"],
            superseded,
        )
        self.assertNotIn("sample_id", str(superseded))
        self.assertNotIn("observed", str(superseded))

    def test_projection_omits_superseded_summary_for_normal_resume(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(max_candidates=1), run_id="run:projection-normal-resume"
            ).run.run_id
            candidate = director.propose_and_spawn(run_id)
            revision = "revision:matching-checkpoint"
            checkpoint = {
                "schema_version": "ecologyrsi-dsh.sample-checkpoint/1",
                "cohort_digest": "a" * 64,
                "execution_context_digest": "b" * 64,
                "sample_count": 4,
            }
            director.start_evaluation_sample_results(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=revision,
                checkpoint=checkpoint,
            )
            director.prepare_evaluation_sample_checkpoint(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                checkpoint=checkpoint,
            )

            execution = _projection_json(director.state(run_id))["candidates"][0][
                "execution"
            ]

        self.assertIsNone(execution["superseded_sample_revision"])

    def test_historical_v1_evaluation_progress_remains_replayable(self) -> None:
        from ecologyrsi_dsh.api.events import EventEndpointsMixin

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(max_candidates=1), run_id="run:projection-progress-v1"
            ).run.run_id
            proposal = director.submit_proposal(
                Proposal(
                    proposal_id="proposal:projection-progress-v1",
                    run_id=run_id,
                    generation=0,
                    title="历史微批进度",
                    changes={"alpha": 0.4},
                )
            )
            candidate = director.spawn_candidate(run_id, proposal)
            ledger.append(
                run_id,
                "EvaluationProgressRecorded",
                {
                    "schema_version": "ecologyrsi-dsh.evaluation-progress/1",
                    "generation": 0,
                    "proposal_id": proposal.proposal_id,
                    "candidate_id": candidate.candidate_id,
                    "role": "planner",
                    "model_id": "strategy-model",
                    "batch_index": 2,
                    "batch_count": 4,
                    "batch_size": 64,
                    "completed_samples": 128,
                    "total_samples": 256,
                    "succeeded_samples": 126,
                    "failed_samples": 2,
                },
                event_id="event:historical-evaluation-progress-v1",
            )

            projection = _projection_json(director.state(run_id))
            progress_event = next(
                event
                for event in director.state(run_id).events
                if event.kind == "EvaluationProgressRecorded"
            )
            public_event = EventEndpointsMixin._event_json(progress_event)

        stage_progress = projection["candidates"][0]["execution"]["stage_progress"]
        self.assertEqual(
            stage_progress["schema_version"],
            "ecologyrsi-dsh.evaluation-progress/1",
        )
        self.assertEqual(stage_progress["gateway_request_count"], 2)
        self.assertEqual(stage_progress["adaptive_split_count"], 0)
        self.assertEqual(stage_progress["adaptive_split_recovered_samples"], 0)
        self.assertEqual(public_event["payload"]["gateway_request_count"], 2)
        self.assertEqual(public_event["payload"]["adaptive_split_count"], 0)

    def test_completed_projection_contains_bounded_sample_trace(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(max_candidates=1), run_id="run:projection-complete"
            ).run.run_id
            proposal = director.submit_proposal(
                Proposal(
                    proposal_id="proposal:projection-complete",
                    run_id=run_id,
                    generation=0,
                    title="完整预测方案",
                    changes={"alpha": 0.4},
                    metadata={"proposal_source": "host_strategy"},
                )
            )
            candidate = director.spawn_candidate(run_id, proposal)
            artifact = ModelArtifact(
                artifact_id="artifact:projection-complete",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                model_id="toy-rolling-water@1",
                dataset_digest="dataset:projection",
                training_partition="train",
                training_rows=10,
                metrics={
                    "execution_mode": "registered_lightweight",
                    "fit_method": "toy_score",
                    "epochs_completed": 1,
                    "training_partition_rows": 10,
                    "training_eligible_examples": 18,
                    "training_used_examples": 16,
                    "training_skipped_examples": 2,
                },
            )
            director.record_artifact(artifact)
            evaluation = Evaluation(
                evaluation_id="evaluation:projection-complete",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                score=0.8,
                passed=True,
                partition="validation",
                metrics={
                    "n": 2,
                    "evaluation_partition_rows": 2,
                    "evaluation_eligible_examples": 3,
                    "evaluation_used_examples": 2,
                    "evaluation_skipped_examples": 1,
                    "scientific_pass": True,
                    "prediction_preview": [
                        {
                            "timestamp": 10,
                            "origin_timestamp": 9,
                            "target_timestamp": 10,
                            "target": "soil_water",
                            "unit": "fraction",
                            "horizon_hours": 24,
                            "observed": 0.4,
                            "predicted": 0.42,
                            "baseline": 0.38,
                            "raw_rows": "must not be copied",
                        }
                    ],
                },
                artifact_digest=artifact.digest,
            )
            director.record_evaluation(evaluation)
            director.decide_promotion(
                Promotion(
                    promotion_id="promotion:projection-complete",
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    decision=PromotionDecision.APPROVED,
                    reason="通过测试",
                )
            )
            director.complete_run(run_id, outcome="accepted")
            state = director.state(run_id)
            completed_event = next(
                event for event in state.events if event.kind == "RunCompleted"
            )
            self.assertEqual(
                completed_event.payload["outcome"],
                "completed_with_search_retained_candidate",
            )
            self.assertEqual(
                completed_event.payload["selection_scope"],
                "iterative_training_feedback_only",
            )
            self.assertEqual(
                completed_event.payload["formal_validation_status"], "not_run"
            )
            projection = _projection_json(state)
            candidate_view = projection["candidates"][0]
            trace = candidate_view["inference_trace"]
            self.assertEqual(trace["status"], "completed")
            self.assertEqual(trace["sample_count"], 2)
            self.assertEqual(trace["shown_count"], 1)
            self.assertTrue(trace["truncated"])
            row = trace["rows"][0]
            self.assertEqual(row["error"], 0.019999999999999962)
            self.assertEqual(row["baseline_error"], -0.020000000000000018)
            self.assertEqual(
                row["reward"],
                abs(0.38 - 0.4) - abs(0.42 - 0.4),
            )
            self.assertEqual(
                trace["reward_definition"],
                "absolute_error_improvement_vs_persistence@1",
            )
            self.assertTrue(trace["positive_is_better"])
            self.assertNotIn("raw_rows", row)
            self.assertEqual(candidate_view["execution"]["progress_percent"], 100.0)
            self.assertEqual(projection["execution_progress"]["progress_percent"], 100.0)
            self.assertEqual(
                projection["outcome"], "completed_with_acceptable_candidate"
            )
            self.assertEqual(
                projection["selection_scope"], "iterative_training_feedback_only"
            )
            self.assertEqual(projection["formal_validation_status"], "not_run")
            self.assertEqual(
                projection["best_candidate_scope"],
                "iterative_training_feedback_only",
            )
            self.assertEqual(projection["best_candidate_id"], candidate.candidate_id)
            self.assertEqual(
                projection["best_observed_candidate_id"], candidate.candidate_id
            )
            self.assertEqual(projection["best_observed_score"], 0.8)
            diagnostics = projection["execution_diagnostics"]
            self.assertEqual(diagnostics["training_partition_rows"], 10)
            self.assertEqual(diagnostics["training_eligible_examples"], 18)
            self.assertEqual(diagnostics["training_used_examples"], 16)
            self.assertEqual(diagnostics["training_skipped_examples"], 2)
            self.assertEqual(diagnostics["evaluation_partition_rows"], 2)
            self.assertEqual(diagnostics["evaluation_eligible_examples"], 3)
            self.assertEqual(diagnostics["evaluation_used_examples"], 2)
            self.assertEqual(diagnostics["evaluation_skipped_examples"], 1)
            self.assertEqual(diagnostics["candidate_work_items"], 18)
            self.assertEqual(diagnostics["fit_passes_completed"], 1)
            self.assertFalse(diagnostics["iterative_epoch_training"])
            self.assertTrue(diagnostics["legacy_workload_estimate_used"])
            self.assertEqual(
                diagnostics["proposal_sources"], {"host_strategy": 1}
            )
            self.assertEqual(diagnostics["remote_strategy_status"], "not_called")
            round_diagnostics = projection["rounds"][0]["execution_diagnostics"]
            self.assertEqual(round_diagnostics["proposal_attempts"], 1)
            self.assertEqual(round_diagnostics["unique_candidates"], 1)
            self.assertEqual(round_diagnostics["training_used_examples"], 16)
            self.assertEqual(round_diagnostics["evaluation_used_examples"], 2)
            self.assertEqual(round_diagnostics["candidate_work_items"], 18)
            self.assertEqual(
                round_diagnostics["proposal_sources"], {"host_strategy": 1}
            )

    def test_exhausted_projection_separates_observed_best_from_acceptable_best(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                _task(max_candidates=2), run_id="run:projection-exhausted"
            ).run.run_id
            candidates = []
            for slot, score, marker in (
                (0, 0.9, "highest-observed"),
                (1, 0.1, "last-evaluated"),
            ):
                proposal = director.submit_proposal(
                    Proposal(
                        proposal_id=f"proposal:projection-exhausted:{slot}",
                        run_id=run_id,
                        generation=0,
                        title=f"候选 {slot}",
                        changes={"alpha": score},
                    )
                )
                candidate = director.spawn_candidate(
                    run_id, proposal, slot_index=slot
                )
                candidates.append(candidate)
                director.evaluate_and_decide(
                    Evaluation(
                        evaluation_id=f"evaluation:projection-exhausted:{slot}",
                        run_id=run_id,
                        candidate_id=candidate.candidate_id,
                        score=score,
                        passed=False,
                        partition="validation",
                        metrics={
                            "marker": marker,
                            "scientific_pass": False,
                            "judge_status": "unavailable" if slot == 0 else "completed",
                        },
                    )
                )
            director.complete_run(
                run_id,
                termination_reason="candidate_budget_exhausted",
                outcome="budget_exhausted_without_acceptable_candidate",
            )

            state = director.state(run_id)
            projection = _projection_json(state)
            summary = run_summary(state)

            self.assertIsNone(projection["best_candidate_id"])
            self.assertIsNone(projection["best_candidate_score"])
            self.assertEqual(
                projection["best_observed_candidate_id"], candidates[0].candidate_id
            )
            self.assertEqual(projection["best_observed_score"], 0.9)
            self.assertIsNone(projection["metrics_candidate_id"])
            self.assertEqual(projection["metrics"], {})
            self.assertEqual(
                projection["metrics_scope"], "no_current_search_incumbent"
            )
            self.assertEqual(
                projection["best_observed_score_scope"],
                "observation_only_full_cohort",
            )
            self.assertFalse(projection["best_observed_drives_current_metrics"])
            candidate_views = {item["id"]: item for item in projection["candidates"]}
            self.assertEqual(
                candidate_views[candidates[0].candidate_id]["execution"]["stages"][
                    "judge"
                ],
                "failed",
            )
            self.assertEqual(
                projection["outcome"],
                "budget_exhausted_without_acceptable_candidate",
            )
            self.assertEqual(
                projection["termination_reason"], "candidate_budget_exhausted"
            )
            progress = projection["execution_progress"]
            self.assertEqual(progress["phase"], "completed")
            self.assertIsNone(progress["current_stage"])
            self.assertLess(progress["completed_steps"], progress["total_steps"])
            self.assertLess(progress["progress_percent"], 100.0)
            self.assertEqual(
                [point["best_observed_score"] for point in projection["trajectory"]],
                [0.9, 0.9],
            )

            self.assertIsNone(summary["best_candidate_id"])
            self.assertIsNone(summary["best_score"])
            self.assertEqual(
                summary["best_observed_candidate_id"], candidates[0].candidate_id
            )
            self.assertEqual(summary["best_observed_score"], 0.9)
            self.assertEqual(summary["outcome"], projection["outcome"])
            self.assertEqual(
                summary["selection_scope"], "iterative_training_feedback_only"
            )
            self.assertEqual(summary["formal_validation_status"], "not_run")
            self.assertIsNone(summary["best_candidate_scope"])

    def test_cross_cohort_lower_score_does_not_replace_current_projection(self) -> None:
        task_data = _task(max_candidates=2, max_generations=2).to_dict()
        task_data["metadata"] = {"samples_per_update": 500}
        task = TaskManifest.from_dict(task_data)
        first_digest = "a" * 64
        second_digest = "b" * 64

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = director.start_evolution(
                task, run_id="run:projection-cross-cohort-incumbent"
            ).run.run_id

            first_proposal = director.submit_proposal(
                Proposal(
                    proposal_id="proposal:projection-cross-cohort:first",
                    run_id=run_id,
                    generation=0,
                    title="first cohort champion",
                    changes={"alpha": 0.8},
                )
            )
            first = director.spawn_candidate(run_id, first_proposal)
            first_promotion = director.evaluate_and_decide(
                Evaluation(
                    evaluation_id="evaluation:projection-cross-cohort:first",
                    run_id=run_id,
                    candidate_id=first.candidate_id,
                    score=0.8,
                    passed=True,
                    partition="validation",
                    metrics={
                        "marker": "old-window-incumbent",
                        "evaluation_index_digest": first_digest,
                    },
                )
            )
            self.assertIs(first_promotion.decision, PromotionDecision.APPROVED)
            director.advance_generation(run_id)

            second_proposal = director.submit_proposal(
                Proposal(
                    proposal_id="proposal:projection-cross-cohort:second",
                    run_id=run_id,
                    generation=1,
                    title="second cohort champion",
                    changes={"alpha": 0.4},
                )
            )
            second = director.spawn_candidate(run_id, second_proposal)
            second_promotion = director.evaluate_and_decide(
                Evaluation(
                    evaluation_id="evaluation:projection-cross-cohort:second",
                    run_id=run_id,
                    candidate_id=second.candidate_id,
                    score=0.4,
                    passed=True,
                    partition="validation",
                    metrics={
                        "marker": "current-incumbent",
                        "evaluation_index_digest": second_digest,
                    },
                )
            )
            self.assertIs(second_promotion.decision, PromotionDecision.REJECTED)
            projection = _projection_json(director.state(run_id))

        self.assertEqual(projection["best_candidate_id"], first.candidate_id)
        self.assertEqual(projection["best_candidate_score"], 0.8)
        self.assertEqual(projection["metrics_candidate_id"], first.candidate_id)
        self.assertEqual(projection["metrics_scope"], "current_search_incumbent")
        self.assertEqual(projection["metrics"]["score"], 0.8)
        self.assertEqual(projection["metrics"]["marker"], "old-window-incumbent")
        self.assertEqual(
            projection["best_observed_candidate_id"], first.candidate_id
        )
        self.assertEqual(projection["best_observed_score"], 0.8)
        self.assertEqual(
            projection["best_observed_score_scope"],
            "observation_only_cross_cohort_not_comparable",
        )
        self.assertFalse(projection["best_observed_drives_current_metrics"])

        first_point, second_point = projection["trajectory"]
        self.assertEqual(first_point["incumbent_score"], 0.8)
        self.assertEqual(second_point["incumbent_score"], 0.8)
        self.assertEqual(first_point["evaluation_cohort_digest"], first_digest)
        self.assertEqual(second_point["evaluation_cohort_digest"], second_digest)
        self.assertEqual(
            second_point["incumbent_before_cohort_digest"], first_digest
        )
        self.assertEqual(
            second_point["score_comparison_boundary"], "different_cohort"
        )
        self.assertFalse(second_point["score_comparable_to_incumbent_before"])

    def test_legacy_accepted_completion_outcome_keeps_public_contract(self) -> None:
        state = type(
            "LegacyState",
            (),
            {
                "run": type(
                    "LegacyRun",
                    (),
                    {
                        "status": type("LegacyStatus", (), {"value": "completed"})(),
                        "best_candidate_id": "candidate:legacy",
                    },
                )(),
                "events": [
                    type(
                        "LegacyEvent",
                        (),
                        {
                            "kind": "RunCompleted",
                            "payload": {
                                "outcome": "accepted",
                                "termination_reason": "legacy_completion",
                            },
                        },
                    )()
                ],
            },
        )()

        self.assertEqual(
            run_completion_outcome(state),
            ("completed_with_acceptable_candidate", "legacy_completion"),
        )

    def test_toy_evaluator_records_prediction_preview(self) -> None:
        toy = ToyCropSoilWater(seed=4)
        candidate = type("CandidateStub", (), {"run_id": "run:toy", "candidate_id": "candidate:toy"})()
        proposal = type("ProposalStub", (), {"run_id": "run:toy", "changes": {"alpha": 0.4, "window": 5, "water_threshold": 0.4}})()
        evaluation = toy.evaluate_candidate("run:toy", candidate, proposal)
        preview = evaluation.metrics["prediction_preview"]
        self.assertTrue(preview)
        self.assertIn("observed", preview[0])
        self.assertIn("predicted", preview[0])
        self.assertIn("baseline", preview[0])
        self.assertLessEqual(len(preview), 48)


if __name__ == "__main__":
    unittest.main()
