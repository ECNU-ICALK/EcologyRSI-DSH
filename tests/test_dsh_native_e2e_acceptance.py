from __future__ import annotations

import unittest

from scripts import dsh_native_e2e_acceptance as acceptance


class DshNativeAcceptanceScriptTests(unittest.TestCase):
    def test_failed_candidate_tool_event_remains_valid_durable_evidence(self) -> None:
        self.assertTrue(
            acceptance._prediction_tool_event_count_is_valid(  # noqa: SLF001
                durable_event_count=2,
                completed_origin_summaries=[
                    {"dsh_agent_prediction_tool_invocations": 1}
                ],
                candidate_count=2,
                prediction_cell_budget_per_candidate=9,
            )
        )

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
