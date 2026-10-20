"""The editor should see only operations the frozen targeted policy permits."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from ecologyrsi_dsh.application.formal_trajectory import (
    _local_edit_context, _legal_parameter_neighborhoods,
)
from ecologyrsi_dsh.evolution.local_edits import validate_local_edit_proposal
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from tests.test_local_edits import _parent


class TargetedEditorContractTests(unittest.TestCase):
    def context(self, *, targeted, budget=2):
        parent = _parent()
        values = parent.scientific_program["parameter_overrides"]
        schemas = {name: {"type": "number", "minimum": 0., "maximum": 1.}
                   for name in values if name.endswith("h_residual_scale")}
        schemas.update(history_steps={"type": "integer", "minimum": 1, "maximum": 12},
                       ridge_alpha={"type": "number", "minimum": .0001, "maximum": 1.})
        metadata = {
            "optimization_schedule": replace(OptimizationSchedule.default(), max_local_edits_per_batch=budget).to_dict(),
            "fitness_profile": {"expected_targets": ["air_temperature", "relative_humidity", "co2_concentration"],
                                "expected_horizons": [1, 6, 24]},
        }
        if targeted:
            metadata["targeted_search_policy"] = "cellwise_training_evidence@1"
        state = SimpleNamespace(
            run=SimpleNamespace(run_id="run:targeted-editor"),
            task_manifest=SimpleNamespace(metadata=metadata),
            proposal=lambda _: SimpleNamespace(metadata={}),
            batch_evaluation_for=lambda *_: SimpleNamespace(scope=SimpleNamespace(scope_key="a" * 64)),
        )
        candidate = SimpleNamespace(candidate_id="candidate:targeted", proposal_id="proposal:targeted", generation=0)
        revision = SimpleNamespace(revision_id="revision:targeted:r0", genome=parent.to_dict(), genome_digest=parent.genome_digest)
        with patch("ecologyrsi_dsh.application.formal_trajectory._registered_mutation_targets",
                   return_value={"scientific_parameter": tuple(schemas)}), patch(
                "ecologyrsi_dsh.application.formal_trajectory._genome_parameter_boundary", return_value=("test", schemas)):
            context = _local_edit_context(state, candidate, revision, SimpleNamespace(batch_index=0))
        return context, revision

    def test_advertised_targets_and_ranges_all_validate_as_single_cell_edits(self):
        context, revision = self.context(targeted=True)
        targets = context.allowed_mutation_targets["scientific_parameter"]
        self.assertEqual(len(targets), 9)
        self.assertEqual(context.maximum_operations, 1)
        neighborhoods = _legal_parameter_neighborhoods(revision, context)
        self.assertEqual(set(neighborhoods), set(targets))
        for name in targets:
            target, horizon = name.removesuffix("_residual_scale").rsplit("_", 1)
            proposal = {"schema_version": "ecology-local-edit@1", "decision": "mutate",
                        "operations": [{"op": "set_bounded_parameter", "name": name, "value": .1}],
                        "evidence_refs": ["batch:metrics"], "expected_effect_cells": [f"{target}@{horizon}"], "risk_cells": []}
            validate_local_edit_proposal(proposal, context)
        self.assertNotIn("ridge_alpha", neighborhoods)
        self.assertNotIn("history_steps", neighborhoods)

    def test_old_mode_and_explicit_zero_budget_are_preserved(self):
        context, revision = self.context(targeted=False)
        self.assertEqual(context.maximum_operations, 2)
        self.assertIn("ridge_alpha", _legal_parameter_neighborhoods(revision, context))
        self.assertIn("history_steps", context.allowed_mutation_targets["scientific_parameter"])
        disabled, _ = self.context(targeted=True, budget=0)
        self.assertEqual(disabled.maximum_operations, 0)
