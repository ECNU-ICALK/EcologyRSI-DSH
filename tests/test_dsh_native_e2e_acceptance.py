from __future__ import annotations

import unittest

from scripts import dsh_native_e2e_acceptance as acceptance


class DshNativeAcceptanceScriptTests(unittest.TestCase):
    def test_current_native_directory_allows_one_route_for_two_independent_roles(self):
        item = {"model_id": "fixture/model", "configured": True,
                "execution_available": True, "roles": ["propose", "judge"]}
        catalog = {"dsh_models": [item], "models": [{"model_id": "retired"}]}
        self.assertEqual(acceptance.configured_role_models(catalog, "propose"), [item])
        self.assertEqual(acceptance.configured_role_models(catalog, "judge"), [item])
        self.assertEqual(acceptance.configured_role_models({"models": [item]}, "propose"), [])
        self.assertEqual(acceptance.configured_role_models({"dsh_models": [{**item, "execution_available": False}]}, "judge"), [])

    def test_current_complete_vector_totals_require_known_counts(self):
        def projection(origins, cells):
            return {"status": "completed", "configuration": {"optimization_protocol": "joint_plugin_epoch@2"},
                    "execution_diagnostics": {"adaptive_all_settled_origins": origins, "adaptive_scored_cells": cells}}
        for origins in (200, 250, 300):
            self.assertEqual(acceptance._current_execution_evidence(projection(origins, origins * 9), 1)["scoring_cells"], origins * 9)
        for origins, cells in ((199, 1791), (301, 2709), (200, 1799), (None, 1800), (200, None)):
            with self.subTest(origins=origins, cells=cells), self.assertRaises(RuntimeError):
                acceptance._current_execution_evidence(projection(origins, cells), 1)
        with self.assertRaises(RuntimeError):
            acceptance._current_execution_evidence({**projection(200, 1800), "status": "paused"}, 1)

    def test_skill_evidence_uses_ledger_backed_public_projection(self) -> None:
        runtime = {
            "first_call_verified": True,
            "skill_invocation": {
                "all_verified": True,
                "verified_call_count": 4,
                "skills": ["sample.plan", "sample.review"],
            },
        }

        skill_runtime = acceptance._verified_skill_runtime(  # noqa: SLF001
            runtime,
            structured_event_count=4,
        )

        self.assertEqual(skill_runtime["verified_call_count"], 4)

    def test_skill_evidence_rejects_projection_count_mismatch(self) -> None:
        runtime = {
            "first_call_verified": True,
            "skill_invocation": {
                "all_verified": True,
                "verified_call_count": 3,
                "skills": ["sample.plan"],
            },
        }

        with self.assertRaisesRegex(RuntimeError, "projection is incomplete"):
            acceptance._verified_skill_runtime(  # noqa: SLF001
                runtime,
                structured_event_count=4,
            )

    def test_skill_evidence_rejects_unverified_first_calls(self) -> None:
        runtime = {
            "first_call_verified": False,
            "skill_invocation": {
                "all_verified": False,
                "verified_call_count": 1,
                "skills": ["sample.plan"],
            },
        }

        with self.assertRaisesRegex(RuntimeError, "projection is incomplete"):
            acceptance._verified_skill_runtime(  # noqa: SLF001
                runtime,
                structured_event_count=1,
            )


if __name__ == "__main__":
    unittest.main()
