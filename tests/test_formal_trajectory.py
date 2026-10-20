from __future__ import annotations

import base64
import json
import unittest
import zlib
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock, patch

from ecologyrsi_dsh import (
    EvolutionDirector,
)
from tests.native_fixtures import NativeRoleStub
from ecologyrsi_dsh.application import formal_trajectory
from ecologyrsi_dsh.application.formal_trajectory import (
    _agent_tool_usage,
    _durable_batch_metrics,
    _inactive_axis_rejection_reason,
    _inert_mutation_axes,
    _local_edit_bundle_signature,
    _local_edit_context,
    _local_edit_current_state,
    _legal_parameter_neighborhoods,
    _local_edit_evidence_metrics,
    _local_edit_policy_rejection_reason,
    _local_edit_proposal,
    _recent_local_edit_history,
    _INACTIVE_AXIS_REJECTION_REASON,
    _TOOL_DEPENDENT_MUTATION_AXES,
)
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.trajectory import (
    FormalBatchComparisonDecision,
    TrajectoryStatus,
)
from ecologyrsi_dsh.evaluators.sample_execution import (
    encode_sample_execution_trace,
)
from ecologyrsi_dsh.evolution.local_edits import (
    LocalEditContext,
    LocalEditProposal,
)
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evolution.effect_contracts import hard_effect_failure


class _NoopScopedCallbacks:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def evaluation_kwargs(self) -> dict:
        return {}

    def completion_payload(self, _evaluation_id, _sample_results):
        return None


class _PairedLaneEvaluator:
    def __init__(self, initial_revision_id: str) -> None:
        self.initial_revision_id = initial_revision_id
        self.calls: list[tuple[int, str, str]] = []

    def evaluate_scientific(self, _task, _candidate, _proposal, *, scope, **_kwargs):
        revision_id = scope.candidate_revision_id
        batch_index = int(scope.batch_index)
        arm = scope.formal_batch_arm.value if scope.formal_batch_arm else "prequential"
        self.calls.append((batch_index, arm, revision_id))
        if batch_index == 0:
            score = 0.2
        elif batch_index == 1:
            score = 0.4 if revision_id == self.initial_revision_id else 0.3
        else:
            score = 0.4 if revision_id == self.initial_revision_id else 0.55
        from ecologyrsi_dsh.evaluators.objectives import OBJECTIVE_AGGREGATION_VERSION, DEFAULT_TARGET_WEIGHTS
        from ecologyrsi_dsh.evolution.promotion import build_promotion_block_evidence
        cohort = _kwargs.get("cohort")
        timestamps = ([o.origin_timestamp for o in cohort.origins] if cohort is not None
                      else list(range(scope.origin_count)))
        rows = [dict(target=target, horizon_hours=horizon, origin_timestamp=t,
            observed=0., predicted=1.-score, baseline=1., normalization_scale=1.,
            sample_execution_status="succeeded") for t in timestamps for target in DEFAULT_TARGET_WEIGHTS for horizon in (1,6,24)]
        metrics = {
            "objective_score": score,
            "objective_aggregation_version": OBJECTIVE_AGGREGATION_VERSION,
            "objective_target_weights": DEFAULT_TARGET_WEIGHTS,
            "objective_horizons": [1,6,24], "objective_weight_coverage": 1.,
            "baseline_profile_digest": "b" * 64,
            "evaluation_index_digest": digest({"batch": batch_index, "timestamps": timestamps}),
            "dataset_digest": "d" * 64, "split_manifest_digest_sha256": "e" * 64,
            "promotion_block_evidence": build_promotion_block_evidence(rows,
                horizons=(1,6,24), target_weights=DEFAULT_TARGET_WEIGHTS,
                dataset_digest="d"*64, split_manifest_digest_sha256="e"*64),
            "constraint_violations": 0, "sample_execution_coverage_pass": True,
            "sample_execution": {
                "attempted_origin_samples": scope.origin_count,
                "succeeded_origin_samples": scope.origin_count,
                "complete_origin_agent_chains": scope.origin_count,
                "failed_origin_samples": 0, "failed_examples": 0, "scoring_fallback_examples": 0,
                "coverage": 1., "coverage_pass": True, "minimum_coverage": .95,
                "strict_agent_chain_pass": True,
            },
            "targets": [{"target": target, "horizon_hours": horizon,
                "skill_score": score, "n": scope.origin_count,
                "eligible_rows": scope.origin_count, "sample_execution_coverage": 1.}
                for target in DEFAULT_TARGET_WEIGHTS for horizon in (1,6,24)],
        }
        evaluation = SimpleNamespace(
            score=score,
            passed=True,
            metrics=metrics,
            evaluator_digest="paired-lane-evaluator@1",
        )
        return SimpleNamespace(evaluation=evaluation, sample_results=())


class FormalTrajectoryTests(unittest.TestCase):

    def test_durable_batch_metrics_keep_one_auditable_compressed_trace(self) -> None:
        trace_records = [
            {
                "sample_id": f"sample:{index}",
                "status": "succeeded",
                "prediction": {"air_temperature": 20.0 + index / 1000},
                "observation": {"air_temperature": 20.5 + index / 1000},
            }
            for index in range(500)
        ]
        trace_archive = encode_sample_execution_trace(trace_records)
        metrics = {
            "objective_score": 0.2,
            "constraint_violations": 0,
            "sample_execution_coverage_pass": True,
            "sample_execution_trace_digest": trace_archive["trace_digest"],
            "dataset_digest": "b" * 64,
            "sample_execution": {
                "attempted_origin_samples": 50,
                "succeeded_origin_samples": 49,
                "coverage": 0.98,
                "coverage_pass": True,
                "failure_counts": {"provider": 1},
                "batch_plan": {"origins": ["private"] * 500},
                "action_catalog": [{"private": "action"}] * 100,
            },
            "targets": [
                {
                    "target": "air_temperature",
                    "horizons": [{"hours": 1, "skill_score": 0.1}],
                }
            ],
            "sample_execution_records": trace_records,
            "sample_execution_trace_archive": trace_archive,
            "prediction_preview": [{"observed": 42}] * 500,
        }

        durable = _durable_batch_metrics(metrics)

        self.assertEqual(durable["dataset_digest"], "b" * 64)
        self.assertEqual(
            durable["sample_execution_trace_digest"],
            trace_archive["trace_digest"],
        )
        self.assertEqual(
            durable["sample_execution_trace_archive"]["record_count"],
            500,
        )
        self.assertEqual(
            durable["sample_execution_trace_archive"]["payload"],
            metrics["sample_execution_trace_archive"]["payload"],
        )
        restored_trace = json.loads(
            zlib.decompress(
                base64.b64decode(
                    durable["sample_execution_trace_archive"]["payload"]
                )
            ).decode("utf-8")
        )
        self.assertEqual(restored_trace, trace_records)
        self.assertEqual(
            durable["sample_execution_trace_archive"]["trace_digest"],
            trace_archive["trace_digest"],
        )
        encoded = json.dumps(durable, ensure_ascii=False)
        self.assertNotIn("sample_execution_records", durable)
        self.assertNotIn("prediction_preview", durable)
        self.assertLess(
            len(encoded.encode("utf-8")),
            len(json.dumps(metrics, ensure_ascii=False).encode("utf-8")),
        )

    def test_local_editor_receives_current_values_and_bounded_outcome_history(self) -> None:
        from tests.test_local_edits import _parent

        parent = _parent()
        current = _local_edit_current_state(
            SimpleNamespace(
                revision_id="revision:local-edit:r0",
                genome=parent.to_dict(),
            )
        )
        self.assertEqual(
            current["scientific_program"]["predictor_id"],
            parent.scientific_program["predictor_ref"]["id"],
        )
        self.assertEqual(
            current["scientific_program"]["parameter_values"]["ridge_alpha"],
            parent.scientific_program["parameter_overrides"]["ridge_alpha"],
        )
        neighborhoods = _legal_parameter_neighborhoods(
            SimpleNamespace(
                revision_id="revision:local-edit:r0",
                genome=parent.to_dict(),
            ),
            SimpleNamespace(
                allowed_mutation_targets={
                    "scientific_parameter": (
                        "co2_concentration_1h_residual_scale",
                    )
                },
                parameter_schemas={
                    "co2_concentration_1h_residual_scale": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    }
                }
            ),
        )
        self.assertEqual(
            neighborhoods["co2_concentration_1h_residual_scale"]["maximum"],
            0.15,
        )

        proposals = {
            index: {
                "proposal": {
                    "decision": "mutate",
                    "operations": [
                        {
                            "op": "set_bounded_parameter",
                            "name": "ridge_alpha",
                            "value": index + 1,
                        }
                    ],
                }
            }
            for index in range(10)
        }
        state = SimpleNamespace(
            formal_batches=tuple(
                SimpleNamespace(
                    candidate_id="candidate:history",
                    batch_index=index,
                    revision_id=f"revision:history:{index}",
                )
                for index in range(10)
            ),
            local_edit_outcomes=tuple(
                {
                    "candidate_id": "candidate:history",
                    "batch_index": index,
                    "outcome": "rejected",
                }
                for index in range(10)
            ),
            local_edit_proposal_for=lambda _candidate, index: proposals[index],
        )
        history = _recent_local_edit_history(state, "candidate:history", 10)
        self.assertEqual(len(history), 8)
        self.assertEqual(history[0]["batch_index"], 2)
        self.assertEqual(
            history[0]["candidate_revision_id"], "revision:history:2"
        )
        self.assertEqual(history[-1]["operations"][0]["value"], 10)

    def test_rejected_bundle_dedup_is_order_stable_and_revision_scoped(self) -> None:
        first = {
            "op": "set_bounded_parameter",
            "name": "ridge_alpha",
            "value": 0.5,
        }
        second = {
            "op": "set_bounded_parameter",
            "name": "history_steps",
            "value": 7,
        }
        state = SimpleNamespace(
            formal_batches=(
                SimpleNamespace(
                    candidate_id="candidate:dedup",
                    batch_index=0,
                    revision_id="revision:dedup:r0",
                ),
            ),
            local_edit_outcomes=(
                {
                    "candidate_id": "candidate:dedup",
                    "batch_index": 0,
                    "outcome": "rejected",
                },
            ),
            local_edit_proposal_for=lambda _candidate, index: (
                {
                    "proposal": {
                        "decision": "mutate",
                        "operations": [first, second],
                    }
                }
                if index == 0
                else None
            ),
        )
        reordered = LocalEditProposal(
            decision="mutate",
            operations=(second, first),
            evidence_refs=(),
            expected_effect_cells=(),
            risk_cells=(),
        )

        self.assertEqual(
            _local_edit_bundle_signature((first, second)),
            _local_edit_bundle_signature((second, first)),
        )
        self.assertEqual(
            _local_edit_policy_rejection_reason(
                state,
                "candidate:dedup",
                2,
                "revision:dedup:r0",
                reordered,
            ),
            "duplicate_recent_rejected_bundle",
        )
        self.assertIsNone(
            _local_edit_policy_rejection_reason(
                state,
                "candidate:dedup",
                2,
                "revision:dedup:r1",
                reordered,
            )
        )
        changed = LocalEditProposal(
            decision="mutate",
            operations=(second, {**first, "value": 0.6}),
            evidence_refs=(),
            expected_effect_cells=(),
            risk_cells=(),
        )
        self.assertIsNone(
            _local_edit_policy_rejection_reason(
                state,
                "candidate:dedup",
                2,
                "revision:dedup:r0",
                changed,
            )
        )


    def test_batch_local_catalog_excludes_whole_predictor_replacement(self) -> None:
        from tests.test_local_edits import _parent

        parent = _parent()
        revision = SimpleNamespace(
            revision_id="revision:bounded-local:r0",
            genome=parent.to_dict(),
            genome_digest=parent.genome_digest,
        )
        state = SimpleNamespace(
            run=SimpleNamespace(run_id="run:bounded-local"),
            proposal=lambda _id: SimpleNamespace(metadata={}),
            task_manifest=SimpleNamespace(
                metadata={
                    "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                    "fitness_profile": {"expected_targets": ["air_temperature", "relative_humidity", "co2_concentration"],
                                        "expected_horizons": [1, 6, 24]},
                }
            ),
            batch_evaluation_for=lambda *_args: SimpleNamespace(scope=SimpleNamespace(scope_key="a" * 64)),
        )
        candidate = SimpleNamespace(
            candidate_id="candidate:bounded-local",
            proposal_id="proposal:bounded-local",
            generation=0,
        )
        batch = SimpleNamespace(batch_index=0)

        with (
            patch(
                "ecologyrsi_dsh.application.formal_trajectory._registered_mutation_targets",
                return_value={
                    "scientific_parameter": ("ridge_alpha",),
                    "registered_predictor": (
                        "greenhouse-horizon-targetwise-ridge@1",
                    ),
                },
            ),
            patch(
                "ecologyrsi_dsh.application.formal_trajectory._genome_parameter_boundary",
                return_value=("test", {}),
            ),
        ):
            context = _local_edit_context(
                state,
                candidate,
                revision,
                batch,
            )

        self.assertNotIn("registered_predictor", context.allowed_mutation_targets)
        self.assertIn("scientific_parameter", context.allowed_mutation_targets)

    def test_zero_local_edit_budget_returns_keep_without_calling_dsh(self) -> None:
        state = SimpleNamespace(candidate_identity_binding=Mock())
        endpoint = SimpleNamespace(
            server=SimpleNamespace(dsh_native_runtime=Mock(), dsh_tools=Mock())
        )
        context = LocalEditContext(
            run_id="run:zero-edit",
            generation=0,
            candidate_id="candidate:zero-edit",
            candidate_revision_id="revision:zero-edit:r0",
            batch_index=0,
            evidence_scope_digest="a" * 64,
            parent_genome_digest="b" * 64,
            maximum_operations=0,
            allowed_mutation_targets={},
            allowed_evidence_refs=("batch:score",),
            allowed_effect_cells=(),
            parameter_schemas={},
        )

        proposal = _local_edit_proposal(
            (endpoint).server,
            state,
            SimpleNamespace(candidate_id="candidate:zero-edit"),
            SimpleNamespace(batch_index=0),
            context,
        )

        self.assertEqual(proposal.decision.value, "keep")
        self.assertEqual(proposal.operations, ())
        state.candidate_identity_binding.assert_not_called()

    def test_large_effect_receipt_is_preserved_for_gates_but_summarized_for_editor(self) -> None:
        for semantics, status in (("soft", "inconclusive"), ("hard", "failed")):
            with self.subTest(semantics=semantics, status=status):
                checks = [
                    {
                        "predicate_id": predicate,
                        "semantics": semantics,
                        "status": status,
                        "eligible": 225,
                        "matched": 0,
                        "evidence_refs": [
                            f"sample:{index}:evidence:{digest({'cell': index})}"
                            for index in range(225)
                        ],
                    }
                    for predicate in ("guidance_behavior_changed", "skill_triggered")
                ]
                body = {
                    "schema_version": "ecologyrsi-dsh.edit-effect-receipt/1",
                    "contract_id": "contract:child",
                    "scope_digest": "a" * 64,
                    "child_genome_digest": "b" * 64,
                    "status": status,
                    "qualification": "behavior_only",
                    "checks": checks,
                }
                receipt = {**body, "receipt_digest": digest(body)}
                metrics = {
                    "objective_score": 0.25,
                    "sample_execution": {
                        "coverage": 1.0,
                        "mutation_effect_receipt": receipt,
                    },
                    "targets": [{"target": "air_temperature", "horizons": [
                        {"hours": 1, "skill_score": 0.1}
                    ]}],
                    "sample_execution_trace_archive": encode_sample_execution_trace([
                        {"sample_id": "sample:1", "status": "succeeded"}
                    ]),
                }
                original = json.dumps(metrics, sort_keys=True)
                self.assertGreater(len(json.dumps(receipt).encode()), 16 * 1024)

                editor = _local_edit_evidence_metrics(metrics)
                durable = _durable_batch_metrics(metrics)

                self.assertLess(len(json.dumps(editor).encode()), 16 * 1024)
                self.assertNotIn("mutation_effect_receipt", editor["sample_execution"])
                summary = editor["sample_execution"]["mutation_effect_summary"]
                self.assertEqual(summary["source_receipt_digest"], receipt["receipt_digest"])
                self.assertEqual(summary["status"], status)
                for check in summary["checks"]:
                    self.assertEqual(check["semantics"], semantics)
                    self.assertEqual(check["status"], status)
                    self.assertEqual(check["eligible"], 225)
                    self.assertEqual(check["matched"], 0)
                    self.assertEqual(check["evidence_ref_count"], 225)
                    self.assertEqual(check["evidence_refs_digest"], digest(checks[0]["evidence_refs"]))
                    self.assertNotIn("evidence_refs", check)
                self.assertEqual(durable["sample_execution"]["mutation_effect_receipt"], receipt)
                self.assertNotIn("mutation_effect_summary", durable["sample_execution"])
                self.assertEqual(durable["targets"], metrics["targets"])
                self.assertEqual(durable["sample_execution_trace_archive"], metrics["sample_execution_trace_archive"])
                self.assertEqual(hard_effect_failure(durable), hard_effect_failure(metrics))
                self.assertEqual(hard_effect_failure(durable),
                                 "hard_mutation_effect_failed" if semantics == "hard" else None)
                self.assertEqual(json.dumps(metrics, sort_keys=True), original)

    def test_local_edit_evidence_is_deeply_json_safe_and_aggregate_only(self) -> None:
        metrics = MappingProxyType(
            {
                "objective_score": 0.25,
                "sample_execution": MappingProxyType(
                    {
                        "coverage": 0.95,
                        "coverage_pass": True,
                        "minimum_coverage": 0.8,
                        "strict_agent_chain_pass": False,
                        "strict_agent_chain_coverage": 0.86,
                        "complete_origin_agent_chains": 43,
                        "failure_counts": MappingProxyType(
                            {"constraint_rejected": 2}
                        ),
                    }
                ),
                "targets": (
                    MappingProxyType(
                        {
                            "target": "air_temperature",
                            "horizons": (
                                MappingProxyType({"hours": 1, "skill_score": 0.1}),
                            ),
                        }
                    ),
                ),
                "sample_execution_records": (MappingProxyType({"sample": 1}),),
                "sample_execution_trace_archive": MappingProxyType({"raw": "trace"}),
                "prediction_preview": (MappingProxyType({"truth": 1.0}),),
                "failure_preview": {"sample": "SECRET-FAILURE"},
                "action_catalog": [{"sample": "SECRET-ACTION"}],
                "batch_plan": {"origins": [{"sample": "SECRET-PLAN"}]},
            }
        )

        evidence = _local_edit_evidence_metrics(metrics)

        self.assertEqual(evidence["targets"][0]["horizons"][0]["hours"], 1)
        self.assertEqual(
            evidence["sample_execution"]["failure_counts"][
                "constraint_rejected"
            ],
            2,
        )
        self.assertIs(
            evidence["sample_execution"]["strict_agent_chain_pass"],
            False,
        )
        self.assertNotIn("sample_execution_records", evidence)
        self.assertNotIn("sample_execution_trace_archive", evidence)
        self.assertNotIn("prediction_preview", evidence)
        self.assertNotIn("failure_preview", evidence)
        self.assertNotIn("action_catalog", evidence)
        self.assertNotIn("batch_plan", evidence)
        self.assertLess(len(json.dumps(evidence, ensure_ascii=False).encode("utf-8")), 16 * 1024)
        json.dumps(evidence, allow_nan=False)




    @staticmethod
    def _scored_state(
        scores: dict[int, float],
        candidate_id: str = "candidate:regress",
        **overrides,
    ) -> SimpleNamespace:
        """A prequential lane whose observable per batch is one clean score."""

        evaluations = {
            index: SimpleNamespace(
                score=score,
                metrics={
                    "constraint_violations": 0,
                    "sample_execution": {"coverage_pass": True},
                },
            )
            for index, score in scores.items()
        }
        fields: dict[str, object] = {
            "batch_evaluation_for": lambda _candidate, index: evaluations.get(index),
            "formal_batches": tuple(
                SimpleNamespace(
                    candidate_id=candidate_id,
                    batch_index=index,
                    revision_id=f"revision:regress:batch:{index}",
                )
                for index in sorted(scores)
            ),
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)


    def test_recent_edit_history_reports_realized_batch_scores(self) -> None:
        """The editor must be able to read what its own last operation did."""

        proposals = {
            index: {
                "proposal": {
                    "decision": "mutate",
                    "operations": [
                        {
                            "op": "set_bounded_parameter",
                            "name": "residual_scale_6h",
                            "value": 0.0375 * (index + 1),
                        }
                    ],
                }
            }
            for index in range(3)
        }
        state = self._scored_state(
            {0: 0.00944, 1: 0.04395, 2: 0.04607},
            local_edit_outcomes=tuple(
                {
                    "candidate_id": "candidate:regress",
                    "batch_index": index,
                    "outcome": "applied",
                }
                for index in range(3)
            ),
            local_edit_proposal_for=lambda _candidate, index: proposals.get(index),
        )

        history = _recent_local_edit_history(state, "candidate:regress", 2)

        self.assertEqual(
            [
                (row["batch_index"], row["batch_score"], row["next_batch_score"])
                for row in history
            ],
            [(0, 0.00944, 0.04395), (1, 0.04395, 0.04607)],
        )


    def test_zero_tool_usage_closes_the_axes_it_makes_unmeasurable(self) -> None:
        """A batch the Agent answered from context cannot score a model edit."""

        def metrics(rate: float | None, cells: int = 90) -> dict[str, object]:
            sample: dict[str, object] = {"agent_prediction_cells": cells}
            if rate is not None:
                sample["agent_tool_usage_rate"] = rate
                sample["agent_tool_invocation_cells"] = int(round(rate * cells))
                sample["agent_tool_invocations"] = int(round(rate * cells))
            return {"sample_execution": sample}

        parameter_edit = LocalEditProposal(
            decision="mutate",
            operations=(
                {"op": "set_bounded_parameter", "name": "ridge_alpha", "value": 0.15},
            ),
            evidence_refs=("batch:metrics",),
            expected_effect_cells=(),
            risk_cells=(),
        )
        instruction_edit = LocalEditProposal(
            decision="mutate",
            operations=(
                {
                    "op": "select_instruction_template",
                    "role": "sample-planner",
                    "instruction_template_id": "sample-planner-model-anchored@1",
                },
            ),
            evidence_refs=("batch:metrics",),
            expected_effect_cells=(),
            risk_cells=(),
        )

        self.assertEqual(
            _inert_mutation_axes(_agent_tool_usage(metrics(0.0))),
            _TOOL_DEPENDENT_MUTATION_AXES,
        )
        self.assertEqual(
            _inactive_axis_rejection_reason(metrics(0.0), parameter_edit),
            _INACTIVE_AXIS_REJECTION_REASON,
        )
        # The instruction axis is exactly how the lane escapes, so it stays open.
        self.assertIsNone(
            _inactive_axis_rejection_reason(metrics(0.0), instruction_edit)
        )
        # Any measured usage reopens the parameter axis.
        self.assertEqual(_inert_mutation_axes(_agent_tool_usage(metrics(0.05))), ())
        self.assertIsNone(
            _inactive_axis_rejection_reason(metrics(0.05), parameter_edit)
        )
        # An unmeasured batch may not close an axis on suspicion alone.
        self.assertIsNone(_agent_tool_usage(metrics(None)))
        self.assertEqual(_inert_mutation_axes(None), ())
        self.assertIsNone(
            _inactive_axis_rejection_reason(metrics(None), parameter_edit)
        )


class PairedFormalTrajectoryTests(unittest.TestCase):
    def setUp(self):
        from tests.test_evidence_guided_epoch import EvidenceGuidedRaceTests
        from ecologyrsi_dsh.application import generation_execution
        EvidenceGuidedRaceTests.setUp(self)
        for candidate in (*self.candidates, self.control):
            EvidenceGuidedRaceTests._screen(self, candidate)
        self.finalist = generation_execution._prepare_formal_finalists(
            self.services, self.run_id, self.candidates, max_concurrency=1)[0]
        self.cid = self.finalist.candidate_id
        self.initial_id = self.director.state(self.run_id).initial_revision_for(self.cid).revision_id
        self.evaluator.initial_revision_id = self.initial_id
        self.evaluator.calls.clear()

    def patches(self, proposal=None):
        from contextlib import ExitStack
        from dataclasses import replace
        stack = ExitStack()
        stack.enter_context(patch.object(formal_trajectory, 'EvaluationSession', _NoopScopedCallbacks))
        stack.enter_context(patch.object(formal_trajectory, '_local_edit_context', side_effect=self._local_context))
        stack.enter_context(patch.object(formal_trajectory, '_local_edit_proposal', return_value=proposal or replace(
            self._mutate_proposal(), operations=({'op':'set_bounded_parameter','name':'ridge_alpha','value':.7},))))
        return stack

    def run_formal(self):
        return formal_trajectory.execute_next_formal_batch(self.services, self.run_id, self.cid)

    def test_reused_screening_then_paired_child_regression_retains_parent(self):
        with self.patches():
            self.assertTrue(self.run_formal())
            self.assertTrue(self.run_formal())
            self.assertEqual(self.evaluator.calls, [])
            self.assertTrue(formal_trajectory.execute_next_local_edit(self.services, self.run_id, self.cid))
            for _ in range(3):
                self.assertTrue(self.run_formal())
        state = self.director.state(self.run_id)
        comparison = state.batch_comparison_for(self.cid, 1)
        self.assertEqual(comparison.decision, FormalBatchComparisonDecision.CHAMPION_RETAINED)
        self.assertEqual(comparison.champion_after_revision_id, self.initial_id)
        self.assertEqual(len(self.evaluator.calls), 2)
        self.assertEqual(state.trajectory_for(self.cid).status, TrajectoryStatus.COMPLETED)
        self.assertFalse(formal_trajectory.execute_next_local_edit(self.services, self.run_id, self.cid))
        self.assertIsNone(state.revision_activation_for(self.cid, 1))

    def test_incomplete_chain_is_rejected_before_scientific_seal(self):
        with self.patches():
            self.run_formal()
            self.run_formal()
            formal_trajectory.execute_next_local_edit(self.services, self.run_id, self.cid)
            evaluate = self.evaluator.evaluate_scientific
            def invalid(*args, **kwargs):
                result = evaluate(*args, **kwargs)
                result.evaluation.metrics['sample_execution'].update(
                    strict_agent_chain_pass=False, complete_origin_agent_chains=24,
                    failed_origin_samples=1, succeeded_origin_samples=24, failed_examples=1)
                return result
            with patch.object(self.evaluator, 'evaluate_scientific', side_effect=invalid):
                from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
                with self.assertRaises(DshNativeRuntimeUnavailableError):
                    self.run_formal()
        state = self.director.state(self.run_id)
        self.assertIsNone(state.batch_evaluation_for(self.cid, 1))
        self.assertIsNone(state.batch_comparison_for(self.cid, 1))

    def test_keep_revision_reuses_the_same_paired_evaluation(self):
        keep = LocalEditProposal(decision='keep', operations=(), evidence_refs=('batch:score',),
            expected_effect_cells=(), risk_cells=())
        with self.patches(keep):
            self.run_formal()
            self.run_formal()
            formal_trajectory.execute_next_local_edit(self.services, self.run_id, self.cid)
            for _ in range(3): self.run_formal()
        comparison = self.director.state(self.run_id).batch_comparison_for(self.cid, 1)
        self.assertEqual(len(self.evaluator.calls), 1)
        self.assertEqual(comparison.champion_evaluation_id, comparison.challenger_evaluation_id)

    def test_crash_after_each_sealed_arm_resumes_only_missing_work(self):
        with self.patches():
            self.run_formal()
            self.run_formal()
            formal_trajectory.execute_next_local_edit(self.services, self.run_id, self.cid)
            real = formal_trajectory._director_mutation
            for expected_calls in (1, 2):
                def lost_response(services, method, *args, **kwargs):
                    value = real(services, method, *args, **kwargs)
                    if method == 'record_formal_batch_evaluation': raise ConnectionError('lost response')
                    return value
                with patch.object(formal_trajectory, '_director_mutation', side_effect=lost_response):
                    with self.assertRaises(ConnectionError): self.run_formal()
                self.assertEqual(len(self.evaluator.calls), expected_calls)
                self.director = EvolutionDirector(self.ledger, NativeRoleStub())
                self.services.director = self.director
            self.run_formal()
            self.assertFalse(self.run_formal())
        self.assertEqual(len(self.evaluator.calls), 2)
        self.assertEqual(self.director.state(self.run_id).trajectory_for(self.cid).status, TrajectoryStatus.COMPLETED)

    def test_local_edit_lost_response_preserves_full_authored_proposal(self):
        with self.patches():
            self.run_formal(); self.run_formal()
            real = formal_trajectory._director_mutation
            def lost(services, method, *args, **kwargs):
                result = real(services, method, *args, **kwargs)
                if method == 'record_local_edit_proposal': raise ConnectionError('lost after commit')
                return result
            with patch.object(formal_trajectory, '_director_mutation', side_effect=lost):
                with self.assertRaises(ConnectionError):
                    formal_trajectory.execute_next_local_edit(self.services, self.run_id, self.cid)
            recorded = self.director.state(self.run_id).local_edit_proposal_for(self.cid, 0)
            with patch.object(formal_trajectory, '_local_edit_proposal', side_effect=AssertionError('must not reauthor')):
                self.assertTrue(formal_trajectory.execute_next_local_edit(self.services, self.run_id, self.cid))
            self.assertEqual(self.director.state(self.run_id).local_edit_proposal_for(self.cid, 0), recorded)

    @staticmethod
    def _local_context(state, candidate, revision, batch):
        from ecologyrsi_dsh.evolution.local_edits import LocalEditContext

        evaluation = state.batch_evaluation_for(
            candidate.candidate_id,
            batch.batch_index,
        )
        return LocalEditContext(
            run_id=state.run.run_id,
            generation=candidate.generation,
            candidate_id=candidate.candidate_id,
            candidate_revision_id=revision.revision_id,
            batch_index=batch.batch_index,
            evidence_scope_digest=evaluation.scope.scope_key,
            parent_genome_digest=revision.genome_digest,
            maximum_operations=1,
            allowed_mutation_targets={
                "scientific_parameter": ("ridge_alpha",),
            },
            allowed_evidence_refs=("batch:score",),
            allowed_effect_cells=tuple(f"{target}@{hour}h" for target in
                ("air_temperature", "relative_humidity", "co2_concentration") for hour in (1, 6, 24)),
            parameter_schemas={
                "ridge_alpha": {"minimum": 0.0001, "maximum": 1.0},
            },
        )


    @staticmethod
    def _mutate_proposal() -> LocalEditProposal:
        return LocalEditProposal(
            decision="mutate",
            operations=(
                {
                    "op": "set_bounded_parameter",
                    "name": "ridge_alpha",
                    "value": 0.2,
                },
            ),
            evidence_refs=("batch:score",),
            expected_effect_cells=("air_temperature@1h",),
            risk_cells=(),
        )




if __name__ == "__main__":
    unittest.main()
