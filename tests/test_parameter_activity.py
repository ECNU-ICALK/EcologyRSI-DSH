"""Zero-correction search explores observable forecast changes first."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.application.formal_trajectory import _local_edit_current_state
from ecologyrsi_dsh.core.models import Run, TaskManifest
from ecologyrsi_dsh.core.trajectory import LocalEditOutcome
from ecologyrsi_dsh.evaluators.greenhouse_prediction import (
    BASELINE_ALIGNED_RIDGE_MODEL_ID, BaselineAlignedRidgeConfig, fit_predict_exogenous_ridge,
)
from ecologyrsi_dsh.evolution.genome import (
    TRUST_REGION_MUTATION_OPERATOR_ID, apply_genome_mutation, materialize_seed_genome,
)
from ecologyrsi_dsh.evolution.local_edits import (
    LocalEditContext, LocalEditProposal, apply_or_reject_local_edit_bundle,
)
from ecologyrsi_dsh.evolution.parameter_activity import (
    parameter_activity_contract,
)
from ecologyrsi_dsh.evolution.strategies import (
    _genome_parameter_boundary, _mutation_contract_catalog, _predictor_semantics,
    _registered_mutation_targets, _validate_candidate_direction_realizability,
)
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_baseline_aligned_ridge import periodic_series
from tests.test_candidate_direction_contract import _direction_payload
from tests.test_evolution_genome import _initialization, _mutation_context
from tests.test_local_edits import _context


def aligned_seed():
    registry = current_program_registry()
    return materialize_seed_genome(
        registry.seed_template("greenhouse-baseline-aligned-default@1"), _initialization()
    )


def mutate(parent, operation):
    return apply_genome_mutation(
        parent, {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": [operation]},
        _mutation_context(parent, mutation_operator_id=TRUST_REGION_MUTATION_OPERATOR_ID),
        current_program_registry(),
    )


class ParameterActivityTests(unittest.TestCase):
    def setUp(self):
        self.parent = aligned_seed()
        self.task = TaskManifest(task_id="aligned", objective="test", domain_pack="greenhouse_environment@1",
                                 metadata={"prediction_model_id": BASELINE_ALIGNED_RIDGE_MODEL_ID,
                                           "evaluator_id": "greenhouse_multihorizon_time_forward@3"})

    def test_zero_scale_hyperparameters_have_identical_fixed_cohort_predictions(self):
        series = periodic_series()
        reference = None
        for history, alpha in ((6, .1), (7, .1), (6, .2), (1, 1.), (12, .0001)):
            result = fit_predict_exogenous_ridge(
                series, targets=("air_temperature", "relative_humidity", "co2_concentration"),
                horizons=(1, 6, 24), config=BaselineAlignedRidgeConfig(history, alpha, 0.),
                evaluation_history_steps=12,
            )
            rows = [(row["origin_timestamp"], row["target"], row["horizon_hours"], row["predicted"])
                    for row in result["prediction_rows"] if row["partition"] == "training_feedback"]
            self.assertTrue(rows)
            if reference is not None:
                self.assertEqual(rows, reference)
            reference = rows


    def test_zero_scale_tool_advice_does_not_prune_agent_search(self):
        targets = _registered_mutation_targets(self.task, self.parent)
        self.assertIn("history_steps", targets["scientific_parameter"])
        self.assertIn("ridge_alpha", targets["scientific_parameter"])
        activity = parameter_activity_contract(self.parent)
        self.assertEqual(activity["scope"], "optional_candidate_model_only")
        self.assertEqual(activity["zero_effect_default_tool_parameters"], ["history_steps", "ridge_alpha"])
        self.assertFalse(activity["agent_policy_parameters_pruned"])
        self.assertIn("final numerical predictions", _mutation_contract_catalog(self.task, self.parent)["mutation_axis_effects"]["instruction_profile"])

    def test_local_edit_can_change_agent_visible_default_parameters(self):
        context = _context(self.parent)
        proposal = LocalEditProposal(decision="mutate", operations=[
            {"op": "set_bounded_parameter", "name": "history_steps", "value": 7}],
            evidence_refs=["metric:overall"], expected_effect_cells=["air_temperature@1h"], risk_cells=[])
        outcome = apply_or_reject_local_edit_bundle(self.parent, proposal, context, current_program_registry())
        self.assertIs(outcome.outcome, LocalEditOutcome.APPLIED)
        self.assertNotEqual(self.parent.behavior_digest, outcome.child.behavior_digest)
