from __future__ import annotations

import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.models import TaskManifest, digest
from ecologyrsi_dsh.core.state import validate_generation_comparison_binding
from ecologyrsi_dsh.core.trajectory import (
    EvaluationPhase,
    EvaluationScope,
    GenerationComparison,
    GenerationHoldout,
    HoldoutArm,
    HoldoutEvaluation,
)
from ecologyrsi_dsh.evaluators.objectives import OBJECTIVE_AGGREGATION_VERSION
from ecologyrsi_dsh.evolution.promotion import (
    PROMOTION_BLOCK_EVIDENCE_VERSION,
    PROMOTION_SCORE_DEFINITION,
)
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from ecologyrsi_dsh.evaluators.fitness import FitnessProfile, SelectionAssessment


TARGETS = ("air_temperature", "relative_humidity", "co2_concentration")
HORIZONS = (1, 6, 24)


def _block_evidence(skill: float) -> dict:
    blocks = []
    for index in range(8):
        cells = [
            {
                "target": target,
                "horizon_hours": horizon,
                "eligible": 1,
                "succeeded": 1,
                "candidate_squared_error_sum": (1.0 - skill) ** 2,
                "baseline_squared_error_sum": 1.0,
                "normalized_reward_sum": skill,
            }
            for target in TARGETS
            for horizon in HORIZONS
        ]
        blocks.append(
            {
                "block_id": digest({"block": index}),
                "origin_block_index": index,
                "cells": cells,
            }
        )
    body = {
        "schema_version": PROMOTION_BLOCK_EVIDENCE_VERSION,
        "block_hours": 24,
        "objective_aggregation_version": OBJECTIVE_AGGREGATION_VERSION,
        "score_definition": PROMOTION_SCORE_DEFINITION,
        "target_weights": {target: 1 / 3 for target in TARGETS},
        "horizons": list(HORIZONS),
        "block_count": len(blocks),
        "blocks": blocks,
    }
    return {**body, "evidence_digest": digest(body)}


def _evaluation(
    arm: HoldoutArm,
    candidate: str,
    revision: str,
    score: float,
    passed: bool = True,
    skill: float = 0.2,
):
    scope = EvaluationScope(
        run_id="run:comparison",
        generation=0,
        candidate_id=candidate,
        candidate_revision_id=revision,
        phase=EvaluationPhase.HOLDOUT,
        cohort_digest=digest({"cohort": "holdout"}),
        origin_count=169,
        holdout_arm=arm,
    )
    return HoldoutEvaluation(
        evaluation_id=f"holdout:{arm.value}",
        scope=scope,
        score=score,
        passed=passed,
        metrics={
            "constraint_violations": 0,
            "sample_execution_coverage_pass": True,
            "objective_weight_coverage": 1.0,
            "sample_execution": {
                "attempted_origin_samples": 169,
                "succeeded_origin_samples": 169,
                "minimum_coverage": 0.95,
                "coverage_pass": True,
                "strict_agent_chain_pass": True,
            },
            "objective_aggregation_version": OBJECTIVE_AGGREGATION_VERSION,
            "objective_target_weights": {target: 1 / 3 for target in TARGETS},
            "objective_horizons": list(HORIZONS),
            "baseline_profile_digest": "b" * 64,
            "evaluation_index_digest": "c" * 64,
            "dataset_digest": "d" * 64,
            "split_manifest_digest_sha256": "e" * 64,
            "targets": [
                {
                    "target": target,
                    "horizon_hours": horizon,
                    "skill_score": skill,
                    "sample_execution_coverage": 1.0,
                }
                for target in TARGETS
                for horizon in HORIZONS
            ],
            "promotion_block_evidence": _block_evidence(skill),
        },
        evaluator_digest=digest({"evaluator": "test"}),
    )


class GenerationComparisonTests(unittest.TestCase):
    def pair(self, gain=.2):
        from tests.test_run_parameter_consistency import small_holdout
        return (small_holdout(HoldoutArm.FINALIST_1,gain),small_holdout(HoldoutArm.INCUMBENT,.1))

    def compare(self, values, **kwargs):
        return build_generation_comparison(run_id=values[0].scope.run_id,generation=0,
            cohort_digest=values[0].scope.cohort_digest,holdout_evaluations=values,
            require_paired_strict_chain=True, **kwargs)

    def test_selects_complete_paired_gain_without_claiming_certification(self):
        result=self.compare(self.pair())
        self.assertEqual(result.selected_candidate_id,'candidate:finalist_1')
        self.assertAlmostEqual(result.gate_results['delta_to_incumbent'],.1)
        self.assertFalse(result.gate_results['arms']['finalist_1']['certification_eligible'])
        self.assertIsNone(result.gate_results['certification_selected_arm'])
        self.assertEqual(GenerationComparison.from_dict(result.to_dict()),result)

    def test_retain_incumbent_for_equal_negative_or_below_practical_gain(self):
        for gain in (.1,-.2,.104):
            with self.subTest(gain=gain):
                result=self.compare(self.pair(gain))
                self.assertEqual(result.selected_candidate_id,'candidate:incumbent')
                self.assertFalse(result.gate_results['arms']['finalist_1']['search_eligible'])

    def test_advisory_flag_cannot_claim_scientific_certification(self):
        result=self.compare(self.pair(),challenger_promotion_allowed=False)
        self.assertEqual(result.selected_candidate_id,'candidate:finalist_1')
        self.assertFalse(result.gate_results['arms']['finalist_1']['certification_eligible'])
        self.assertIn('screening_exploration_only', result.gate_results['arms']['finalist_1']['certification_failures'])

    def test_malformed_constraint_count_and_missing_chain_fail_closed(self):
        from dataclasses import replace
        for field,value in [('constraint_violations',None),('constraint_violations',-.1),
                            ('constraint_violations',True),('constraint_violations',.5),
                            ('sample_execution',{'strict_agent_chain_pass':False})]:
            items=list(self.pair());metrics=items[0].to_dict()['metrics'];metrics[field]=value
            items[0]=replace(items[0],metrics=metrics)
            with self.subTest(field=field,value=value):
                gate=self.compare(items).gate_results['arms']['finalist_1']
                self.assertFalse(gate['search_eligible'])

    def test_duplicate_missing_or_incomplete_grid_cannot_overwrite_evidence(self):
        from dataclasses import replace
        for mutate in (lambda rows:rows[:1],lambda rows:[*rows,rows[0]],
                       lambda rows:[{**row,'skill_score':-.2} if i==0 else row for i,row in enumerate(rows)]):
            items=[]
            for item in self.pair():
                metrics=item.to_dict()['metrics']
                if item.scope.holdout_arm is HoldoutArm.FINALIST_1:
                    metrics['targets']=mutate(metrics['targets'])
                items.append(replace(item,metrics=metrics))
            self.assertFalse(self.compare(items).gate_results['arms']['finalist_1']['search_eligible'])
        items=[]
        for item in self.pair():
            metrics=item.to_dict()['metrics'];metrics['targets']=metrics['targets'][:1]
            items.append(replace(item,metrics=metrics))
        self.assertFalse(self.compare(items).gate_results['arms']['finalist_1']['complete_objective_grid'])

    def test_wrong_scope_extra_arms_and_old_protocol_are_rejected(self):
        from dataclasses import replace
        items=self.pair()
        for altered in ((items[0],),(*items,items[0]),
            (replace(items[0],scope=replace(items[0].scope,cohort_digest='f'*64),metrics=items[0].to_dict()['metrics']),items[1])):
            with self.assertRaises(ValueError):self.compare(altered)
        with self.assertRaises(ValueError):self.compare(items,experiment_protocol='joint_plugin_epoch@1')
