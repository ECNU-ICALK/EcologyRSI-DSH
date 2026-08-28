from __future__ import annotations

import json
import unittest
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock

from ecologyrsi_dsh.api.formal_trajectory import (
    _local_edit_evidence_metrics,
    _local_edit_proposal,
)
from ecologyrsi_dsh.evolution.local_edits import LocalEditContext


class FormalTrajectoryTests(unittest.TestCase):
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
            }
        )

        evidence = _local_edit_evidence_metrics(metrics)

        self.assertEqual(evidence["targets"][0]["horizons"][0]["hours"], 1)
        self.assertNotIn("sample_execution_records", evidence)
        self.assertNotIn("sample_execution_trace_archive", evidence)
        self.assertNotIn("prediction_preview", evidence)
        json.dumps(evidence, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
