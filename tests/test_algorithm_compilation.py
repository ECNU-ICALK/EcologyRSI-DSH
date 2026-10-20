"""The native Genome compiler is the sole executable algorithm authority."""
from dataclasses import replace
import unittest

from ecologyrsi_dsh.core.models import Proposal
from ecologyrsi_dsh.knowledge.algorithms import AlgorithmSpec, compile_algorithm_spec, debug_algorithm_spec
from ecologyrsi_dsh.evolution.workflow_ir import compile_plugin_behavior
from ecologyrsi_dsh.core.state import persisted_genome_from_proposal
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from ecologyrsi_dsh.evolution.genome import deep_thaw_json
from tests.native_candidate_fixture import seed_proposal
from tests.native_fixtures import native_manifest


class AlgorithmCompilationTests(unittest.TestCase):
    def setUp(self):
        self.task = native_manifest()
        self.proposal = seed_proposal(self.task)

    def test_current_compilation_matches_shared_genome_template(self):
        spec = compile_algorithm_spec(self.task, self.proposal, None)
        compiled = compile_plugin_behavior(persisted_genome_from_proposal(self.proposal), self.task,
                                           None, current_program_registry())
        expected = AlgorithmSpec(run_id=self.proposal.run_id, generation=0,
                                proposal_id=self.proposal.proposal_id,
                                **deep_thaw_json(compiled.algorithm_spec_template))
        self.assertEqual(spec, expected)
        self.assertTrue(debug_algorithm_spec(spec, self.task, self.proposal)["passed"])
        self.assertEqual(AlgorithmSpec.from_dict(spec.to_dict()), spec)

    def test_no_genome_cannot_fall_back_to_old_research_plan(self):
        proposal = Proposal(proposal_id="old", run_id=self.proposal.run_id, generation=0,
                            title="old", changes=dict(self.proposal.changes),
                            metadata={"plan": {"prediction_model": {"id": "greenhouse-exogenous-ridge@1"}}})
        with self.assertRaisesRegex(ValueError, "Genome"):
            compile_algorithm_spec(self.task, proposal, None)

    def test_parameters_cannot_disagree_with_genome(self):
        proposal = replace(self.proposal, changes={**self.proposal.changes, "ridge_alpha": .987})
        with self.assertRaisesRegex(ValueError, "parameters differ"):
            compile_algorithm_spec(self.task, proposal, None)

    def test_operator_tampering_is_rejected(self):
        data = compile_algorithm_spec(self.task, self.proposal, None).to_dict()
        data["algorithm_ir"]["operators"][0]["operator_id"] = "shell.execute@1"
        with self.assertRaisesRegex(ValueError, "operator graph"):
            AlgorithmSpec.from_dict(data)

    def test_forbidden_partition_is_rejected_before_compilation(self):
        for partition in ("validation", "final_test", "external_holdout"):
            task = replace(self.task, metadata={**self.task.metadata, "evaluation_partition": partition})
            with self.assertRaisesRegex(ValueError, "partition"):
                compile_algorithm_spec(task, self.proposal, None)

    def test_frozen_snapshot_change_is_rejected(self):
        task = replace(self.task, metadata={**self.task.metadata, "split_manifest_digest": "e" * 64})
        with self.assertRaisesRegex(ValueError, "binding"):
            compile_algorithm_spec(task, self.proposal, None)
