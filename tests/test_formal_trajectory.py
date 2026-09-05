from __future__ import annotations

import base64
import json
import unittest
import zlib
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock, patch

from ecologyrsi_dsh import (
    EventLedger,
    EvolutionDirector,
    FakeDSHAdapter,
    TaskManifest,
)
from ecologyrsi_dsh.api import formal_trajectory
from ecologyrsi_dsh.api.formal_trajectory import (
    _durable_batch_metrics,
    _local_edit_bundle_signature,
    _local_edit_context,
    _local_edit_current_state,
    _legal_parameter_neighborhoods,
    _local_edit_evidence_metrics,
    _local_edit_policy_rejection_reason,
    _local_edit_proposal,
    _local_challenger_policy,
    _prequential_safety_reason,
    _recent_local_edit_history,
    _safety_requires_rollback,
    execute_next_local_edit,
)
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.screening import screening_cohort_digest
from ecologyrsi_dsh.core.trajectory import (
    CandidateRevision,
    FormalBatchArm,
    FormalBatchComparisonDecision,
    LocalEditOutcome,
    RevisionAdvanceReason,
    RevisionStatus,
    TrajectoryStatus,
)
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.epoch_cohorts import (
    plan_generation_selection_cohorts,
    plan_run_adaptation_cohort,
)
from ecologyrsi_dsh.evaluators.sample_execution import (
    encode_sample_execution_trace,
)
from ecologyrsi_dsh.evolution.local_edits import (
    LocalEditContext,
    LocalEditProposal,
    LocalEditResult,
)
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule


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
        arm = scope.formal_batch_arm.value
        self.calls.append((batch_index, arm, revision_id))
        if batch_index == 0:
            score = 0.2
        elif batch_index == 1:
            score = 0.4 if revision_id == self.initial_revision_id else 0.3
        else:
            score = 0.4 if revision_id == self.initial_revision_id else 0.55
        metrics = {
            "objective_score": score,
            "objective_aggregation_version": "paired-test-objective@1",
            "objective_target_weights": {"air_temperature": 1.0},
            "objective_horizons": [1],
            "baseline_profile_digest": "b" * 64,
            "evaluation_index_digest": digest({"batch": batch_index}),
            "dataset_digest": "d" * 64,
            "split_manifest_digest_sha256": "e" * 64,
            "constraint_violations": 0,
            "sample_execution_coverage_pass": True,
            "sample_execution": {
                "attempted_origin_samples": scope.origin_count,
                "succeeded_origin_samples": scope.origin_count,
                "failed_origin_samples": 0,
                "coverage": 1.0,
                "coverage_pass": True,
                "minimum_coverage": 0.95,
                "strict_agent_chain_pass": True,
            },
            "targets": [
                {
                    "target": "air_temperature",
                    "horizon_hours": 1,
                    "skill_score": score,
                }
            ],
        }
        evaluation = SimpleNamespace(
            score=score,
            passed=True,
            metrics=metrics,
            evaluator_digest="paired-lane-evaluator@1",
        )
        return SimpleNamespace(evaluation=evaluation, sample_results=())


class FormalTrajectoryTests(unittest.TestCase):
    def test_runtime_v3_selects_positive_delta_local_policy(self) -> None:
        task = TaskManifest(
            task_id="runtime-v3-local-policy",
            objective="verify local search policy",
            domain_pack="crop-soil-water@toy",
            metadata={
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "host_runtime_build": {
                    "evolution_runtime_schema": (
                        "ecologyrsi-dsh.evolution-runtime/3"
                    )
                },
            },
        )
        state = SimpleNamespace(task_manifest=task)
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state)
            )
        )

        self.assertEqual(
            _local_challenger_policy(endpoint, "run:v3"),
            (1e-12, False),
        )

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

    def test_same_revision_repeated_rejected_bundle_never_creates_child(self) -> None:
        candidate = SimpleNamespace(candidate_id="candidate:dedup", generation=0)
        revision = SimpleNamespace(
            revision_id="revision:dedup:r0",
            genome={},
            parent_revision_id=None,
        )
        prior = SimpleNamespace(
            candidate_id=candidate.candidate_id,
            batch_index=0,
            revision_id=revision.revision_id,
        )
        pending = SimpleNamespace(
            candidate_id=candidate.candidate_id,
            batch_index=2,
            revision_id=revision.revision_id,
        )
        evaluation = SimpleNamespace(
            metrics={
                "constraint_violations": 0,
                "sample_execution_coverage_pass": True,
            }
        )
        operation = {
            "op": "set_bounded_parameter",
            "name": "ridge_alpha",
            "value": 0.5,
        }
        prior_record = {
            "proposal_id": "local-edit:candidate:dedup:0",
            "candidate_id": candidate.candidate_id,
            "batch_index": 0,
            "proposal": {
                "decision": "mutate",
                "operations": [operation],
            },
        }
        state = SimpleNamespace(
            candidate=lambda _id: candidate,
            trajectory_for=lambda _id: SimpleNamespace(batch_count=4),
            formal_batches=(prior, pending),
            batch_evaluation_for=lambda *_args: evaluation,
            revision_activation_for=lambda _candidate, index: (
                object() if index == 0 else None
            ),
            local_edit_outcomes=(
                {
                    "candidate_id": candidate.candidate_id,
                    "batch_index": 0,
                    "outcome": "rejected",
                },
            ),
            local_edit_proposal_for=lambda _candidate, index: (
                prior_record if index == 0 else None
            ),
            revision=lambda _id: revision,
        )
        endpoint = SimpleNamespace(
            server=SimpleNamespace(director=SimpleNamespace(state=lambda _id: state))
        )
        proposal = LocalEditProposal(
            decision="mutate",
            operations=(operation,),
            evidence_refs=("batch:score",),
            expected_effect_cells=(),
            risk_cells=(),
        )
        context = SimpleNamespace(
            candidate_revision_id=revision.revision_id,
            evidence_scope_digest="a" * 64,
        )
        calls = []

        with (
            patch(
                "ecologyrsi_dsh.api.formal_trajectory._local_edit_context",
                return_value=context,
            ),
            patch(
                "ecologyrsi_dsh.api.formal_trajectory._local_edit_proposal",
                return_value=proposal,
            ),
            patch(
                "ecologyrsi_dsh.api.formal_trajectory.apply_or_reject_local_edit_bundle"
            ) as apply_edit,
            patch(
                "ecologyrsi_dsh.api.formal_trajectory._director_mutation",
                side_effect=lambda *args: calls.append(args),
            ),
        ):
            self.assertTrue(
                execute_next_local_edit(
                    endpoint, "run:dedup", candidate.candidate_id
                )
            )

        apply_edit.assert_not_called()
        self.assertEqual(
            [call[1] for call in calls],
            [
                "record_local_edit_proposal",
                "decide_local_edit",
                "advance_trajectory_revision",
            ],
        )
        self.assertEqual(calls[0][3]["proposal"], proposal.to_dict())
        self.assertEqual(calls[1][3]["outcome"], LocalEditOutcome.REJECTED.value)
        self.assertEqual(
            calls[1][3]["reason"], "duplicate_recent_rejected_bundle"
        )
        self.assertEqual(calls[1][3]["active_revision_id"], revision.revision_id)
        self.assertEqual(
            calls[2][6], RevisionAdvanceReason.LOCAL_EDIT_REJECTED
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
            task_manifest=SimpleNamespace(
                metadata={
                    "optimization_schedule": OptimizationSchedule.default().to_dict()
                }
            ),
            batch_evaluation_for=lambda *_args: SimpleNamespace(scope=SimpleNamespace(scope_key="a" * 64)),
        )
        candidate = SimpleNamespace(
            candidate_id="candidate:bounded-local",
            generation=0,
        )
        batch = SimpleNamespace(batch_index=0)

        with (
            patch(
                "ecologyrsi_dsh.api.formal_trajectory._registered_mutation_targets",
                return_value={
                    "scientific_parameter": ("ridge_alpha",),
                    "registered_predictor": (
                        "greenhouse-horizon-targetwise-ridge@1",
                    ),
                },
            ),
            patch(
                "ecologyrsi_dsh.api.formal_trajectory._genome_parameter_boundary",
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
            endpoint,
            state,
            SimpleNamespace(candidate_id="candidate:zero-edit"),
            SimpleNamespace(batch_index=0),
            context,
        )

        self.assertEqual(proposal.decision.value, "keep")
        self.assertEqual(proposal.operations, ())
        state.candidate_identity_binding.assert_not_called()

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

    def test_prequential_guard_blocks_edit_for_coverage_or_constraints(self) -> None:
        self.assertEqual(
            _prequential_safety_reason({"sample_execution_coverage_pass": False}),
            "coverage_guardrail_failed",
        )
        self.assertEqual(
            _prequential_safety_reason(
                {
                    "sample_execution_coverage_pass": False,
                    "sample_execution": {
                        "coverage_pass": False,
                        "failure_counts": {"constraint_rejected": 3},
                    },
                }
            ),
            "sample_constraint_guardrail_failed",
        )
        self.assertEqual(
            _prequential_safety_reason(
                {
                    "constraint_violations": 0,
                    "sample_execution_coverage_pass": True,
                    "sample_execution": {
                        "attempted_origin_samples": 50,
                        "succeeded_origin_samples": 49,
                        "minimum_coverage": 0.8,
                        "coverage_pass": True,
                        "strict_agent_chain_pass": False,
                        "failure_counts": {},
                    },
                }
            ),
            "strict_origin_chain_guardrail_failed",
        )
        self.assertFalse(
            _safety_requires_rollback("strict_origin_chain_guardrail_failed")
        )
        self.assertEqual(
            _prequential_safety_reason(
                {
                    "constraint_violations": 0,
                    "sample_execution_coverage_pass": True,
                    "sample_execution": {
                        "attempted_origin_samples": 50,
                        "succeeded_origin_samples": 39,
                        "minimum_coverage": 0.8,
                        "coverage_pass": True,
                        "strict_agent_chain_pass": True,
                        "failure_counts": {},
                    },
                }
            ),
            "origin_coverage_guardrail_failed",
        )
        self.assertEqual(
            _prequential_safety_reason(
                {
                    "constraint_violations": 0,
                    "sample_execution_coverage_pass": True,
                    "sample_execution": {
                        "attempted_origin_samples": 50,
                        "succeeded_origin_samples": 43,
                        "minimum_coverage": 0.8,
                        "coverage_pass": True,
                        "strict_agent_chain_pass": False,
                        "failure_counts": {"constraint_rejected": 7},
                    },
                }
            ),
            "sample_constraint_guardrail_failed",
        )
        self.assertEqual(
            _prequential_safety_reason({"constraint_violations": 1}),
            "constraint_guardrail_failed",
        )
        self.assertEqual(
            _prequential_safety_reason({"constraint_violations": float("nan")}),
            "constraint_guardrail_invalid",
        )
        self.assertEqual(
            _prequential_safety_reason({"constraint_violations": 0}),
            "coverage_guardrail_missing",
        )
        self.assertFalse(_safety_requires_rollback("coverage_guardrail_failed"))
        self.assertTrue(
            _safety_requires_rollback("sample_constraint_guardrail_failed")
        )
        self.assertIsNone(
            _prequential_safety_reason(
                {"constraint_violations": 0, "sample_execution": {"coverage_pass": True}}
            )
        )

    def test_recovery_replays_full_durable_proposal_before_child_decision(self) -> None:
        """A crash after proposal persistence must not re-author a lossy edit."""

        candidate = SimpleNamespace(candidate_id="candidate:replay", generation=0)
        revision = SimpleNamespace(
            revision_id="revision:replay:r0",
            genome={},
            parent_revision_id=None,
        )
        batch = SimpleNamespace(
            candidate_id=candidate.candidate_id,
            batch_index=0,
            revision_id=revision.revision_id,
        )
        evaluation = SimpleNamespace(metrics={"constraint_violations": 0})
        proposal = {
            "schema_version": "ecology-local-edit@1",
            "decision": "keep",
            "operations": [],
            "evidence_refs": ["batch:score", "batch:metrics"],
            "expected_effect_cells": ["air_temperature@1h"],
            "risk_cells": ["co2_concentration@24h"],
        }
        state = SimpleNamespace(
            run=SimpleNamespace(run_id="run:replay"),
            candidate=lambda _id: candidate,
            trajectory_for=lambda _id: SimpleNamespace(batch_count=2),
            formal_batches=(batch,),
            batch_evaluation_for=lambda *_args: evaluation,
            revision_activation_for=lambda *_args: None,
            local_edit_outcomes=(),
            local_edit_proposal_for=lambda *_args: {
                "proposal_id": "local-edit:candidate:replay:0",
                "candidate_id": candidate.candidate_id,
                "batch_index": 0,
                "evidence_scope_digest": "a" * 64,
                "proposal": proposal,
                "decision": "keep",
                "operations": [],
            },
            revision=lambda _id: revision,
        )
        endpoint = SimpleNamespace(
            server=SimpleNamespace(director=SimpleNamespace(state=lambda _id: state))
        )
        context = SimpleNamespace(
            evidence_scope_digest="a" * 64,
            candidate_revision_id=revision.revision_id,
        )
        mutations = []

        with (
            patch("ecologyrsi_dsh.api.formal_trajectory._local_edit_context", return_value=context),
            patch("ecologyrsi_dsh.api.formal_trajectory._local_edit_proposal") as authored,
            patch("ecologyrsi_dsh.api.formal_trajectory.EcologyEvolutionPluginGenome.from_dict", return_value=object()),
            patch(
                "ecologyrsi_dsh.api.formal_trajectory.apply_or_reject_local_edit_bundle",
                return_value=LocalEditResult(LocalEditOutcome.KEPT, (), None, "b" * 64),
            ) as apply_edit,
            patch("ecologyrsi_dsh.api.formal_trajectory._director_mutation", side_effect=lambda *_args: mutations.append(_args[1])),
        ):
            self.assertTrue(execute_next_local_edit(endpoint, "run:replay", candidate.candidate_id))

        authored.assert_not_called()
        recovered = apply_edit.call_args.args[1]
        self.assertEqual(recovered.evidence_refs, ("batch:score", "batch:metrics"))
        self.assertEqual(recovered.expected_effect_cells, ("air_temperature@1h",))
        self.assertEqual(recovered.risk_cells, ("co2_concentration@24h",))
        self.assertEqual(mutations, ["decide_local_edit", "advance_trajectory_revision"])

    def test_prequential_guard_durably_rolls_back_to_parent_revision(self) -> None:
        candidate = SimpleNamespace(candidate_id="candidate:guard", generation=0)
        revision = SimpleNamespace(
            revision_id="revision:guard:r1",
            genome={},
            parent_revision_id="revision:guard:r0",
        )
        batch = SimpleNamespace(
            candidate_id=candidate.candidate_id,
            batch_index=1,
            revision_id=revision.revision_id,
        )
        evaluation = SimpleNamespace(metrics={"constraint_violations": 1})
        state = SimpleNamespace(
            run=SimpleNamespace(run_id="run:guard"),
            candidate=lambda _id: candidate,
            trajectory_for=lambda _id: SimpleNamespace(batch_count=3),
            formal_batches=(batch,),
            batch_evaluation_for=lambda *_args: evaluation,
            revision_activation_for=lambda *_args: None,
            local_edit_outcomes=(),
            local_edit_proposal_for=lambda *_args: None,
            revision=lambda _id: revision,
        )
        endpoint = SimpleNamespace(
            server=SimpleNamespace(director=SimpleNamespace(state=lambda _id: state))
        )
        calls = []
        with (
            patch("ecologyrsi_dsh.api.formal_trajectory._local_edit_context", return_value=SimpleNamespace(evidence_scope_digest="a" * 64)),
            patch("ecologyrsi_dsh.api.formal_trajectory._local_edit_proposal") as authored,
            patch("ecologyrsi_dsh.api.formal_trajectory._director_mutation", side_effect=lambda *args: calls.append(args)),
        ):
            self.assertTrue(execute_next_local_edit(endpoint, "run:guard", candidate.candidate_id))

        authored.assert_not_called()
        self.assertEqual(calls[0][1], "record_local_edit_proposal")
        self.assertEqual(calls[1][1], "decide_local_edit")
        self.assertEqual(calls[1][3]["outcome"], LocalEditOutcome.ROLLED_BACK.value)
        self.assertEqual(calls[1][3]["active_revision_id"], "revision:guard:r0")
        self.assertEqual(calls[2][1], "advance_trajectory_revision")
        self.assertEqual(calls[2][6], RevisionAdvanceReason.PREQUENTIAL_SAFETY_ROLLBACK)


class PairedFormalTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_local_edits import _parent

        self.ledger = EventLedger()
        self.director = EvolutionDirector(
            self.ledger,
            FakeDSHAdapter(max_proposals=20),
        )
        self.schedule = OptimizationSchedule.from_dict(
            {
                **OptimizationSchedule.default().to_dict(),
                "formal_origin_count_per_finalist": 30,
                "local_batch_origin_count": 10,
            }
        )
        task = TaskManifest(
            task_id="paired-formal-trajectory",
            objective="exercise champion challenger state machine",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={
                "max_generations": 1,
                "candidates_per_generation": 4,
                "max_candidates": 4,
            },
            seed=23,
            metadata={
                "episode_id": "episode:paired-formal",
                "optimization_protocol": "top2_adaptive_epoch@1",
                "optimization_schedule": self.schedule.to_dict(),
                "prediction_cells_per_origin": 1,
            },
        )
        self.run_id = "run:paired-formal"
        self.director.start_evolution(task, run_id=self.run_id)
        self.candidates = tuple(
            self.director.propose_and_spawn(self.run_id) for _ in range(4)
        )
        parent = _parent()
        self.revisions: dict[str, CandidateRevision] = {}
        for index, candidate in enumerate(self.candidates):
            revision = CandidateRevision(
                revision_id=f"revision:paired-formal:{index}:r0",
                run_id=self.run_id,
                generation=0,
                candidate_id=candidate.candidate_id,
                genome=parent.to_dict(),
                genome_digest=parent.genome_digest,
                behavior_digest=parent.behavior_digest,
                mutation_digest=digest({"seed-mutation": index}),
                status=RevisionStatus.ACTIVE,
            )
            self.director.create_candidate_revision(self.run_id, revision)
            self.revisions[candidate.candidate_id] = revision
        dataset = SimpleNamespace(
            dataset_id="generated-toy-series@1",
            episode_id="episode:paired-formal",
            timestamps=tuple(range(1200)),
            partitions={"model_selection": IndexRange(0, 1200)},
        )
        adaptation = plan_run_adaptation_cohort(
            dataset,
            schedule=self.schedule,
            seed=23,
        )
        cohorts = plan_generation_selection_cohorts(
            dataset,
            schedule=self.schedule,
            generation=0,
            adaptation=adaptation,
            seed=23,
        )
        self.director.freeze_run_adaptation_cohort(self.run_id, adaptation)
        self.director.freeze_generation_selection_cohorts(
            self.run_id,
            cohorts,
        )
        for candidate in self.candidates:
            self.director.record_candidate_screening(
                self.run_id,
                candidate_id=candidate.candidate_id,
                generation=0,
                score=1.0 - candidate.slot_index * 0.1,
                passed=True,
                constraint_violations=0,
                origin_count=64,
                prediction_cell_count=64,
                cohort_digest=cohorts.screening.cohort_digest,
            )
        screening = [
            event.payload
            for event in self.director.state(self.run_id).candidate_screening_events
        ]
        self.finalist = self.candidates[0]
        self.director.freeze_formal_selection_cohort(
            self.run_id,
            generation=0,
            selected_candidate_ids=[
                self.candidates[0].candidate_id,
                self.candidates[1].candidate_id,
            ],
            screening_digest=screening_cohort_digest(screening),
        )
        initial = self.revisions[self.finalist.candidate_id]
        self.evaluator = _PairedLaneEvaluator(initial.revision_id)
        self.endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=self.director,
                ledger=self.ledger,
                evaluators=self.evaluator,
            )
        )

    def tearDown(self) -> None:
        self.ledger.close()

    @staticmethod
    def _revision_inputs(state, candidate, revision_id, _task):
        return (
            state.revision(revision_id),
            SimpleNamespace(proposal_id=candidate.proposal_id),
            object(),
        )

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
            allowed_effect_cells=("air_temperature@1h",),
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

    def _execution_patches(self, proposal: LocalEditProposal):
        return (
            patch.object(
                formal_trajectory,
                "_phase_task_manifest",
                return_value=object(),
            ),
            patch.object(
                formal_trajectory,
                "_revision_evaluation_inputs",
                side_effect=self._revision_inputs,
            ),
            patch.object(
                formal_trajectory,
                "_ScopedEvaluationCallbacks",
                _NoopScopedCallbacks,
            ),
            patch.object(
                formal_trajectory,
                "_local_edit_context",
                side_effect=self._local_context,
            ),
            patch.object(
                formal_trajectory,
                "_local_edit_proposal",
                return_value=proposal,
            ),
        )

    def test_retained_champion_parents_next_challenger_and_final_has_no_edit(self) -> None:
        candidate_id = self.finalist.candidate_id
        initial_id = self.revisions[candidate_id].revision_id
        patches = self._execution_patches(self._mutate_proposal())
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            self.assertTrue(
                formal_trajectory.execute_next_formal_batch(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )
            self.assertTrue(
                formal_trajectory.execute_next_formal_batch(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )
            state = self.director.state(self.run_id)
            warmup = state.batch_comparison_for(candidate_id, 0)
            self.assertEqual(
                warmup.decision,
                FormalBatchComparisonDecision.INITIAL_CHAMPION,
            )
            self.assertEqual(len(self.evaluator.calls), 1)

            self.assertTrue(
                formal_trajectory.execute_next_local_edit(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )
            first_challenger = self.director.state(self.run_id).revision_activation_for(
                candidate_id,
                0,
            ).to_revision_id
            self.assertNotEqual(first_challenger, initial_id)

            for _ in range(3):
                self.assertTrue(
                    formal_trajectory.execute_next_formal_batch(
                        self.endpoint,
                        self.run_id,
                        candidate_id,
                    )
                )
            state = self.director.state(self.run_id)
            retained = state.batch_comparison_for(candidate_id, 1)
            self.assertEqual(
                retained.decision,
                FormalBatchComparisonDecision.CHAMPION_RETAINED,
            )
            self.assertEqual(retained.champion_after_revision_id, initial_id)

            self.assertTrue(
                formal_trajectory.execute_next_local_edit(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )
            state = self.director.state(self.run_id)
            second_challenger_id = state.revision_activation_for(
                candidate_id,
                1,
            ).to_revision_id
            second_challenger = state.revision(second_challenger_id)
            self.assertEqual(second_challenger.parent_revision_id, initial_id)

            for _ in range(3):
                self.assertTrue(
                    formal_trajectory.execute_next_formal_batch(
                        self.endpoint,
                        self.run_id,
                        candidate_id,
                    )
                )
            state = self.director.state(self.run_id)
            trajectory = state.trajectory_for(candidate_id)
            final_comparison = state.batch_comparison_for(candidate_id, 2)
            self.assertIs(trajectory.status, TrajectoryStatus.COMPLETED)
            self.assertEqual(
                trajectory.final_revision_id,
                final_comparison.champion_after_revision_id,
            )
            self.assertEqual(trajectory.final_revision_id, second_challenger_id)
            self.assertFalse(
                formal_trajectory.execute_next_local_edit(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )
            self.assertEqual(
                len(
                    [
                        item
                        for item in state.local_edit_outcomes
                        if item["candidate_id"] == candidate_id
                    ]
                ),
                2,
            )
            self.assertIsNone(state.revision_activation_for(candidate_id, 2))
            self.assertFalse(
                any(
                    item.candidate_id == candidate_id
                    and item.source_batch_index == 2
                    for item in state.candidate_revisions
                )
            )

    def test_paired_challenger_missing_strict_chain_evidence_fails_closed(self) -> None:
        candidate_id = self.finalist.candidate_id
        evaluate = self.evaluator.evaluate_scientific

        def evaluate_without_challenger_chain(*args, scope, **kwargs):
            bundle = evaluate(*args, scope=scope, **kwargs)
            if (
                scope.batch_index == 1
                and scope.formal_batch_arm is FormalBatchArm.CHALLENGER
            ):
                bundle.evaluation.score = 0.6
                bundle.evaluation.metrics["objective_score"] = 0.6
                bundle.evaluation.metrics["targets"][0]["skill_score"] = 0.6
                del bundle.evaluation.metrics["sample_execution"][
                    "strict_agent_chain_pass"
                ]
            return bundle

        patches = self._execution_patches(self._mutate_proposal())
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patch.object(
                self.evaluator,
                "evaluate_scientific",
                side_effect=evaluate_without_challenger_chain,
            ),
        ):
            self.assertTrue(
                formal_trajectory.execute_next_formal_batch(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )
            self.assertTrue(
                formal_trajectory.execute_next_formal_batch(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )
            self.assertTrue(
                formal_trajectory.execute_next_local_edit(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )
            for _ in range(3):
                self.assertTrue(
                    formal_trajectory.execute_next_formal_batch(
                        self.endpoint,
                        self.run_id,
                        candidate_id,
                    )
                )

        comparison = self.director.state(self.run_id).batch_comparison_for(
            candidate_id,
            1,
        )
        self.assertEqual(comparison.challenger_score, 0.6)
        self.assertFalse(comparison.safety_gate_passed)
        self.assertEqual(
            comparison.decision,
            FormalBatchComparisonDecision.CHAMPION_RETAINED,
        )
        self.assertEqual(comparison.reason, "challenger_safety_gate_failed")

    def test_resume_after_each_boundary_executes_only_missing_work(self) -> None:
        """Every durable paired boundary is an idempotent resume point."""

        class _InjectedCrash(RuntimeError):
            pass

        candidate_id = self.finalist.candidate_id
        initial_id = self.revisions[candidate_id].revision_id
        proposal = self._mutate_proposal()
        authored_batch_indexes: list[int] = []
        real_mutation = formal_trajectory._director_mutation

        def author_local_edit(_endpoint, _state, _candidate, batch, _context):
            authored_batch_indexes.append(batch.batch_index)
            return proposal

        def restart_from_ledger() -> None:
            self.director = EvolutionDirector(
                self.ledger,
                FakeDSHAdapter(max_proposals=20),
            )
            self.endpoint.server.director = self.director

        def crash_after(boundary: str, action) -> None:
            boundary_reached = False

            def commit_then_crash(endpoint, method_name, *args, **kwargs):
                nonlocal boundary_reached
                result = real_mutation(
                    endpoint,
                    method_name,
                    *args,
                    **kwargs,
                )
                if method_name == boundary:
                    boundary_reached = True
                    raise _InjectedCrash(boundary)
                return result

            with patch.object(
                formal_trajectory,
                "_director_mutation",
                side_effect=commit_then_crash,
            ):
                with self.assertRaisesRegex(_InjectedCrash, boundary):
                    action()
            self.assertTrue(boundary_reached)
            restart_from_ledger()

        run_formal = lambda: formal_trajectory.execute_next_formal_batch(
            self.endpoint,
            self.run_id,
            candidate_id,
        )
        run_local = lambda: formal_trajectory.execute_next_local_edit(
            self.endpoint,
            self.run_id,
            candidate_id,
        )
        execution_patches = (
            patch.object(
                formal_trajectory,
                "_phase_task_manifest",
                return_value=object(),
            ),
            patch.object(
                formal_trajectory,
                "_revision_evaluation_inputs",
                side_effect=self._revision_inputs,
            ),
            patch.object(
                formal_trajectory,
                "_ScopedEvaluationCallbacks",
                _NoopScopedCallbacks,
            ),
            patch.object(
                formal_trajectory,
                "_local_edit_context",
                side_effect=self._local_context,
            ),
            patch.object(
                formal_trajectory,
                "_local_edit_proposal",
                side_effect=author_local_edit,
            ),
        )

        with (
            execution_patches[0],
            execution_patches[1],
            execution_patches[2],
            execution_patches[3],
            execution_patches[4],
        ):
            # Warmup evaluation and comparison both survive a lost response.
            crash_after("record_formal_batch_evaluation", run_formal)
            self.assertEqual(
                self.evaluator.calls,
                [(0, FormalBatchArm.CHAMPION.value, initial_id)],
            )
            crash_after("record_formal_batch_comparison", run_formal)
            initial_comparison = self.director.state(
                self.run_id
            ).batch_comparison_for(candidate_id, 0)
            initial_comparison_payload = initial_comparison.to_dict()
            self.assertFalse(run_formal())
            self.assertEqual(
                self.director.state(self.run_id)
                .batch_comparison_for(candidate_id, 0)
                .to_dict(),
                initial_comparison_payload,
            )
            self.assertEqual(len(self.evaluator.calls), 1)

            # Proposal, child, decision, and activation are separate durable
            # points. Each retry must consume the recorded work instead of
            # asking the editor again or creating another child identity.
            crash_after("record_local_edit_proposal", run_local)
            self.assertEqual(authored_batch_indexes, [0])
            self.assertIsNotNone(
                self.director.state(self.run_id).local_edit_proposal_for(
                    candidate_id,
                    0,
                )
            )
            crash_after("create_candidate_revision", run_local)
            first_child_id = f"revision:{candidate_id}:batch:1"
            first_child_payload = self.director.state(self.run_id).revision(
                first_child_id
            ).identity_dict()
            self.assertEqual(authored_batch_indexes, [0])
            crash_after("decide_local_edit", run_local)
            self.assertEqual(
                self.director.state(self.run_id)
                .revision(first_child_id)
                .identity_dict(),
                first_child_payload,
            )
            self.assertEqual(authored_batch_indexes, [0])
            crash_after("advance_trajectory_revision", run_local)
            first_activation = self.director.state(
                self.run_id
            ).revision_activation_for(candidate_id, 0)
            self.assertEqual(first_activation.to_revision_id, first_child_id)
            self.assertFalse(run_local())
            self.assertEqual(authored_batch_indexes, [0])

            # The next same-cohort pair independently resumes after each arm,
            # then preserves the exact comparison when its response is lost.
            crash_after("record_formal_batch_evaluation", run_formal)
            self.assertEqual(
                self.evaluator.calls[-1],
                (1, FormalBatchArm.CHAMPION.value, initial_id),
            )
            crash_after("record_formal_batch_evaluation", run_formal)
            self.assertEqual(
                self.evaluator.calls[-1],
                (1, FormalBatchArm.CHALLENGER.value, first_child_id),
            )
            crash_after("record_formal_batch_comparison", run_formal)
            retained_comparison = self.director.state(
                self.run_id
            ).batch_comparison_for(candidate_id, 1)
            retained_comparison_payload = retained_comparison.to_dict()
            self.assertFalse(run_formal())
            self.assertEqual(
                self.director.state(self.run_id)
                .batch_comparison_for(candidate_id, 1)
                .to_dict(),
                retained_comparison_payload,
            )
            self.assertEqual(len(self.evaluator.calls), 3)

            self.assertTrue(run_local())
            second_child_id = f"revision:{candidate_id}:batch:2"
            second_child_payload = self.director.state(self.run_id).revision(
                second_child_id
            ).identity_dict()
            self.assertEqual(authored_batch_indexes, [0, 1])

            # The final comparison may complete the trajectory, but schema v2
            # must never author or activate an unvalidated post-final child.
            self.assertTrue(run_formal())
            self.assertTrue(run_formal())
            crash_after("complete_formal_trajectory", run_formal)
            completed_state = self.director.state(self.run_id)
            final_comparison = completed_state.batch_comparison_for(
                candidate_id,
                2,
            )
            final_comparison_payload = final_comparison.to_dict()
            self.assertIs(
                completed_state.trajectory_for(candidate_id).status,
                TrajectoryStatus.COMPLETED,
            )
            self.assertEqual(
                completed_state.trajectory_for(candidate_id).final_revision_id,
                second_child_id,
            )
            self.assertFalse(run_formal())
            self.assertFalse(run_local())

        final_state = self.director.state(self.run_id)
        self.assertEqual(
            self.evaluator.calls,
            [
                (0, FormalBatchArm.CHAMPION.value, initial_id),
                (1, FormalBatchArm.CHAMPION.value, initial_id),
                (1, FormalBatchArm.CHALLENGER.value, first_child_id),
                (2, FormalBatchArm.CHAMPION.value, initial_id),
                (2, FormalBatchArm.CHALLENGER.value, second_child_id),
            ],
        )
        self.assertEqual(authored_batch_indexes, [0, 1])
        self.assertEqual(
            final_state.revision(second_child_id).identity_dict(),
            second_child_payload,
        )
        self.assertEqual(
            final_state.batch_comparison_for(candidate_id, 2).to_dict(),
            final_comparison_payload,
        )
        self.assertEqual(
            [
                item.comparison_id
                for item in final_state.formal_batch_comparisons
                if item.candidate_id == candidate_id
            ],
            [
                f"comparison:{candidate_id}:0",
                f"comparison:{candidate_id}:1",
                f"comparison:{candidate_id}:2",
            ],
        )
        self.assertIsNone(final_state.local_edit_proposal_for(candidate_id, 2))
        self.assertIsNone(final_state.revision_activation_for(candidate_id, 2))
        self.assertFalse(
            any(
                revision.candidate_id == candidate_id
                and revision.source_batch_index == 2
                for revision in final_state.candidate_revisions
            )
        )
        candidate_events = self.ledger.events(self.run_id)
        self.assertEqual(
            sum(
                event.kind == "CandidateRevisionCreated"
                and event.payload["revision"]["revision_id"] == first_child_id
                for event in candidate_events
            ),
            1,
        )
        self.assertEqual(
            sum(
                event.kind == "CandidateRevisionCreated"
                and event.payload["revision"]["revision_id"] == second_child_id
                for event in candidate_events
            ),
            1,
        )
        self.assertEqual(
            sum(
                event.kind == "FormalTrajectoryCompleted"
                for event in candidate_events
            ),
            1,
        )

    def test_same_revision_pair_reuses_one_evaluation(self) -> None:
        candidate_id = self.finalist.candidate_id
        keep = LocalEditProposal(
            decision="keep",
            operations=(),
            evidence_refs=("batch:score",),
            expected_effect_cells=(),
            risk_cells=(),
        )
        patches = self._execution_patches(keep)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            self.assertTrue(
                formal_trajectory.execute_next_formal_batch(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )
            self.assertTrue(
                formal_trajectory.execute_next_formal_batch(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )
            self.assertTrue(
                formal_trajectory.execute_next_local_edit(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )
            self.assertTrue(
                formal_trajectory.execute_next_formal_batch(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )
            call_count = len(self.evaluator.calls)
            self.assertTrue(
                formal_trajectory.execute_next_formal_batch(
                    self.endpoint,
                    self.run_id,
                    candidate_id,
                )
            )

        comparison = self.director.state(self.run_id).batch_comparison_for(
            candidate_id,
            1,
        )
        self.assertEqual(len(self.evaluator.calls), call_count)
        self.assertEqual(
            comparison.champion_evaluation_id,
            comparison.challenger_evaluation_id,
        )
        self.assertEqual(
            comparison.decision,
            FormalBatchComparisonDecision.CHAMPION_RETAINED,
        )


if __name__ == "__main__":
    unittest.main()
