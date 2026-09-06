import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ecologyrsi_dsh.evolution.analysis import _parameter_effects
from ecologyrsi_dsh.presentation.reporting import _search_design_audit


class ParameterCoverageTests(unittest.TestCase):
    def test_parameter_effects_report_unidentifiable_constant_axes(self) -> None:
        state = SimpleNamespace(task_manifest=SimpleNamespace(metadata={}))
        scored = [
            (SimpleNamespace(changes={"history_steps": 3, "residual_scale": 0.5}), -0.1),
            (SimpleNamespace(changes={"history_steps": 6, "residual_scale": 0.5}), -0.05),
            (SimpleNamespace(changes={"history_steps": 9, "residual_scale": 0.5}), -0.02),
        ]
        with patch(
            "ecologyrsi_dsh.evolution.analysis._historical_scored_configurations",
            return_value=scored,
        ):
            effects = _parameter_effects(state, 0)

        residual = next(item for item in effects if item["parameter"] == "residual_scale")
        self.assertEqual(residual["unique_value_count"], 1)
        self.assertEqual(residual["direction"], "unidentifiable")
        self.assertIsNone(residual["association_with_score"])

    def test_search_design_audit_is_bounded(self) -> None:
        audit = _search_design_audit(
            {
                "policy": "bounded_identifiable_parameter_design@1",
                "adopted_parameter": "residual_scale",
                "shared_reference_parameters": {
                    "residual_scale": 0.5,
                    "history_steps": 3,
                },
                "private_error": "must not be projected",
            }
        )
        self.assertEqual(audit["adopted_parameter"], "residual_scale")
        self.assertNotIn("private_error", audit)
        self.assertEqual(audit["shared_reference_parameters"]["history_steps"], 3)


if __name__ == "__main__":
    unittest.main()
