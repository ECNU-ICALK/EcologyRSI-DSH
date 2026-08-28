from __future__ import annotations

import base64
import json
import unittest
import zlib
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock, patch

from ecologyrsi_dsh.api.formal_trajectory import (
    _durable_batch_metrics,
    _local_edit_context,
    _local_edit_current_state,
    _local_edit_evidence_metrics,
    _local_edit_proposal,
    _prequential_safety_reason,
    _recent_local_edit_history,
    _safety_requires_rollback,
    execute_next_local_edit,
)
from ecologyrsi_dsh.core.trajectory import LocalEditOutcome, RevisionAdvanceReason
from ecologyrsi_dsh.evaluators.sample_execution import (
    encode_sample_execution_trace,
)
from ecologyrsi_dsh.evolution.local_edits import LocalEditContext, LocalEditResult
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule


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
        self.assertEqual(history[-1]["operations"][0]["value"], 10)

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
        context = SimpleNamespace(evidence_scope_digest="a" * 64)
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


if __name__ == "__main__":
    unittest.main()
