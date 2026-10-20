from __future__ import annotations
import unittest
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.evaluators.shared_sample_context import (
 ORIGIN_SHARED_CONTEXT_PROFILE, normalized_sample_planner_prompt_profile,
 sibling_stage_context_digest,
)
class SharedSampleContextTests(unittest.TestCase):
    def test_profile_validation_is_exact(self) -> None:
        self.assertEqual(
            normalized_sample_planner_prompt_profile(
                {"version": ORIGIN_SHARED_CONTEXT_PROFILE}
            ),
            {"version": ORIGIN_SHARED_CONTEXT_PROFILE},
        )
        with self.assertRaises(ValueError):
            normalized_sample_planner_prompt_profile(
                {"version": ORIGIN_SHARED_CONTEXT_PROFILE, "extra": True}
            )
        with self.assertRaises(ValueError):
            normalized_sample_planner_prompt_profile({"version": "future@2"})


    def test_sibling_stage_digest_excludes_candidate_genome(self) -> None:
        contracts = {
            "data_protocol_digest": "a" * 64,
            "evaluation_cohort_digest": "b" * 64,
            "fitness_profile_digest": "c" * 64,
        }
        first = sibling_stage_context_digest(
            task_manifest_digest="d" * 64,
            generation=2,
            frozen_contract_digests=contracts,
        )
        second = sibling_stage_context_digest(
            task_manifest_digest="d" * 64,
            generation=2,
            frozen_contract_digests=contracts,
        )
        self.assertEqual(first, second)
        self.assertNotEqual(
            digest({"stage_context_digest": first, "genome_digest": "1" * 64}),
            digest({"stage_context_digest": second, "genome_digest": "2" * 64}),
        )
