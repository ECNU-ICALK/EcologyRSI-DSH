from __future__ import annotations

import json
import unittest
from types import MappingProxyType

from ecologyrsi_dsh.api.formal_trajectory import _local_edit_evidence_metrics


class FormalTrajectoryTests(unittest.TestCase):
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
