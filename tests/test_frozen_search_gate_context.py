"""Research feedback must describe the gates the Host actually enforces."""
from copy import deepcopy
from dataclasses import replace
import unittest

from ecologyrsi_dsh.core.search_policy import SEARCH_GUARD_POLICY, local_challenger_policy
from ecologyrsi_dsh.evaluators.fitness import FitnessProfile
from ecologyrsi_dsh.evolution.genome import materialize_seed_genome
from ecologyrsi_dsh.evolution.strategies import _mutation_contract_catalog
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_autonomous_search_reflection_cycle import _task
from tests.test_evolution_genome import _initialization


class FrozenSearchGateContextTests(unittest.TestCase):
    def catalog(self, task):
        parent = materialize_seed_genome(current_program_registry().seed_template("greenhouse-default@1"),
                                         _initialization(task_manifest_digest=task.digest))
        return _mutation_contract_catalog(task, parent)

    def test_custom_frozen_thresholds_reach_model_contract_without_mutation(self):
        profile = FitnessProfile(selection_minimum_score_delta=0.007,
                                 selection_cell_regression_tolerance=0.02,
                                 selection_minimum_coverage=0.97)
        base = _task()
        task = replace(base, metadata={**base.metadata, "fitness_profile": profile.to_dict(),
            "fitness_profile_digest": profile.profile_digest, "search_guard_policy": SEARCH_GUARD_POLICY})
        before = deepcopy(task.to_dict())
        gates = self.catalog(task)["evaluation_evidence_contract"]["frozen_search_gates"]
        self.assertEqual(gates["minimum_score_delta"], 0.007)
        self.assertEqual(gates["cell_regression_tolerance"], 0.02)
        self.assertEqual(gates["minimum_coverage"], 0.97)
        self.assertEqual(gates["local_comparison"], local_challenger_policy(task.metadata))
        self.assertEqual(gates["fitness_profile_digest"], profile.profile_digest)
        self.assertEqual(gates["scope"], "search_selection_not_independent_certification")
        self.assertTrue(gates["additional_host_gates_required"])
        self.assertEqual(task.to_dict(), before)

    def test_partial_legacy_profile_does_not_receive_current_defaults(self):
        contract = self.catalog(_task())["evaluation_evidence_contract"]
        self.assertNotIn("frozen_search_gates", contract)

    def test_unbound_thresholds_are_not_presented_as_frozen_facts(self):
        profile = FitnessProfile()
        task = replace(_task(), metadata={**_task().metadata, "fitness_profile": profile.to_dict(),
                                         "fitness_profile_digest": "0" * 64})
        with self.assertRaisesRegex(ValueError, "fitness profile binding mismatch"):
            self.catalog(task)


if __name__ == "__main__":
    unittest.main()
