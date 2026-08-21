from __future__ import annotations

from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.evolution.analysis import (
    GenerationAnalysis,
    _metric_weaknesses,
    build_cross_generation_experience,
)
from ecologyrsi_dsh.evolution.context import analysis_focus_parameter


class HorizonFeedbackTests(unittest.TestCase):
    def test_all_target_horizon_cells_retain_normalized_reward(self) -> None:
        targets = ("air_temperature", "relative_humidity", "co2_concentration")
        horizons = (1, 6, 24)
        metric_rows = [
            {
                "target": target,
                "horizon_hours": horizon,
                "unit": "unit",
                "skill_score": 0.2 - horizon / 100,
                "normalized_mean_reward": (
                    -0.25
                    if target in {"relative_humidity", "co2_concentration"}
                    and horizon == 24
                    else 0.1
                ),
            }
            for target in targets
            for horizon in horizons
        ]
        evaluation = SimpleNamespace(metrics={"targets": metric_rows})
        state = SimpleNamespace(evaluation_for=lambda _candidate_id: evaluation)

        rows = _metric_weaknesses(
            state,
            [SimpleNamespace(candidate_id="candidate:one")],
            "targets",
            ("target", "horizon_hours", "unit"),
        )

        self.assertEqual(len(rows), 9)
        by_identity = {
            (row["target"], row["horizon_hours"]): row for row in rows
        }
        self.assertEqual(
            by_identity[("co2_concentration", 24)][
                "median_normalized_mean_reward"
            ],
            -0.25,
        )
        self.assertEqual(
            by_identity[("relative_humidity", 24)]["reward_evidence_count"],
            1,
        )

        reward_only_evaluation = SimpleNamespace(
            metrics={
                "targets": [
                    {
                        "target": "co2_concentration",
                        "horizon_hours": 24,
                        "unit": "ppm",
                        "normalized_mean_reward": -0.3,
                    }
                ]
            }
        )
        reward_only_state = SimpleNamespace(
            evaluation_for=lambda _candidate_id: reward_only_evaluation
        )
        reward_only = _metric_weaknesses(
            reward_only_state,
            [SimpleNamespace(candidate_id="candidate:reward-only")],
            "targets",
            ("target", "horizon_hours", "unit"),
        )
        self.assertEqual(len(reward_only), 1)
        self.assertNotIn("median_skill_score", reward_only[0])
        self.assertEqual(reward_only[0]["median_normalized_mean_reward"], -0.3)

    def test_negative_reward_is_active_history_even_when_rmse_skill_is_positive(
        self,
    ) -> None:
        identity = {
            "target": "co2_concentration",
            "horizon_hours": 24,
            "unit": "ppm",
        }
        negative_reward = GenerationAnalysis(
            run_id="run:reward-history",
            generation=0,
            candidate_count=1,
            eligible_count=0,
            outcome="no_eligible_candidate",
            target_weaknesses=(
                {
                    **identity,
                    "median_skill_score": 0.05,
                    "median_normalized_mean_reward": -0.2,
                    "evidence_count": 1,
                    "reward_evidence_count": 1,
                },
            ),
        )
        state = SimpleNamespace(
            generation_analyses=(negative_reward,),
            research_iterations=(),
            proposals=(),
        )

        experience = build_cross_generation_experience(state, 1)

        issue = next(
            item
            for item in experience["active_unresolved"]
            if item["kind"] == "weak_target"
        )
        self.assertEqual(issue["identity"]["target"], "co2_concentration")
        self.assertEqual(
            issue["latest_observation"]["weakness_basis"], "mean_reward"
        )
        self.assertEqual(
            experience["generations"][0]["weak_targets"][0][
                "median_normalized_mean_reward"
            ],
            -0.2,
        )

        recovered = GenerationAnalysis(
            run_id="run:reward-history",
            generation=1,
            candidate_count=1,
            eligible_count=1,
            outcome="promoted",
            target_weaknesses=(
                {
                    **identity,
                    "median_skill_score": 0.1,
                    "median_normalized_mean_reward": 0.1,
                    "evidence_count": 1,
                    "reward_evidence_count": 1,
                },
            ),
        )
        state.generation_analyses = (negative_reward, recovered)
        resolved = build_cross_generation_experience(state, 2)
        self.assertFalse(resolved["active_unresolved"])
        self.assertEqual(resolved["resolved_archived"][0]["resolved_generation"], 1)

    def test_targetwise_focus_uses_the_degraded_targets_residual_scale(self) -> None:
        schemas = {
            "history_steps": {"type": "integer", "minimum": 1, "maximum": 12},
            "ridge_alpha": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
            "air_temperature_residual_scale": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "relative_humidity_residual_scale": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "co2_concentration_residual_scale": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        }
        co2_analysis = {
            "target_weaknesses": [
                {"target": "co2_concentration", "horizon_hours": 24}
            ]
        }
        humidity_analysis = {
            "target_weaknesses": [
                {"target": "relative_humidity", "horizon_hours": 24}
            ]
        }

        self.assertEqual(
            analysis_focus_parameter(co2_analysis, schemas),
            "co2_concentration_residual_scale",
        )
        self.assertEqual(
            analysis_focus_parameter(humidity_analysis, schemas),
            "relative_humidity_residual_scale",
        )

    def test_horizon_targetwise_focus_uses_the_degraded_cell(self) -> None:
        schemas = {
            "history_steps": {"type": "integer", "minimum": 1, "maximum": 12},
            "ridge_alpha": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
            **{
                f"{target}_{horizon}h_residual_scale": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                }
                for target in (
                    "air_temperature",
                    "relative_humidity",
                    "co2_concentration",
                )
                for horizon in (1, 6, 24)
            },
        }

        self.assertEqual(
            analysis_focus_parameter(
                {
                    "target_weaknesses": [
                        {"target": "co2_concentration", "horizon_hours": 1}
                    ]
                },
                schemas,
            ),
            "co2_concentration_1h_residual_scale",
        )
        self.assertEqual(
            analysis_focus_parameter(
                {
                    "target_weaknesses": [
                        {"target": "co2_concentration", "horizon_hours": 24}
                    ]
                },
                schemas,
            ),
            "co2_concentration_24h_residual_scale",
        )


if __name__ == "__main__":
    unittest.main()
