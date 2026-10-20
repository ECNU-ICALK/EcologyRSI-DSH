"""Operational availability alone cannot qualify as scientific improvement."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.application.formal_trajectory import _durable_batch_metrics
from ecologyrsi_dsh.core.models import TaskManifest, digest, thaw_json
from ecologyrsi_dsh.data.adapters import CUCUMBER_2018
from ecologyrsi_dsh.core.search_policy import (
    PAIRED_EXECUTION_QUALIFICATION, SEARCH_GUARD_POLICY,
    local_challenger_policy, paired_execution_qualification_required,
)
from ecologyrsi_dsh.core.state import (
    validate_generation_comparison_binding,
)
from ecologyrsi_dsh.core.trajectory import (
    BatchEvaluation, EvaluationPhase, EvaluationScope, FormalBatchArm,
    FormalBatchComparison, GenerationHoldout, HoldoutArm, HoldoutEvaluation,
    _deep_freeze_json,
)
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from ecologyrsi_dsh.evaluators.objectives import (
    DEFAULT_TARGET_WEIGHTS, OBJECTIVE_AGGREGATION_VERSION,
)
from ecologyrsi_dsh.evolution.champion_challenger import (
    assess_local_challenger, local_challenger_safety_reason,
    validate_formal_batch_comparison,
)
from ecologyrsi_dsh.evolution.promotion import (
    _resampled_objective, _validated_evidence, assess_promotion_improvement,
    build_promotion_block_evidence,
)
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evolution.execution_qualification import paired_scoring_evidence_complete


GUARD_METADATA = {
    "search_guard_policy": SEARCH_GUARD_POLICY,
    "execution_protocol": "dsh_native_plugin_evolution@1",
    "host_runtime_build": {
        "evolution_runtime_schema": "ecologyrsi-dsh.evolution-runtime/3"
    },
    "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
}


def evaluation(arm, *, failed=False, skill=.1):
    """Synthetic complete grid; failed arm loses one whole origin per day."""
    local = isinstance(arm, FormalBatchArm)
    days = 3 if local else 8
    rows = [
        dict(target=t, horizon_hours=h, origin_timestamp=day * 24 + hour,
             observed=0., predicted=1. - skill, baseline=1., normalization_scale=1.,
             sample_execution_status="failed" if failed and hour == 0 else "succeeded",
             scoring_fallback="failure_non_improvement_penalty" if failed and hour == 0 else None)
        for day in range(days) for hour in range(24)
        for t in DEFAULT_TARGET_WEIGHTS for h in (1, 6, 24)
    ]
    evidence = build_promotion_block_evidence(
        rows, horizons=(1, 6, 24), target_weights=DEFAULT_TARGET_WEIGHTS,
        dataset_digest="d" * 64, split_manifest_digest_sha256="e" * 64,
    )
    q = 23 / 24 if failed else 1.
    metrics = dict(
        objective_aggregation_version=OBJECTIVE_AGGREGATION_VERSION,
        objective_target_weights=DEFAULT_TARGET_WEIGHTS, objective_horizons=[1, 6, 24],
        baseline_profile_digest="b" * 64, evaluation_index_digest="c" * 64,
        dataset_digest="d" * 64, split_manifest_digest_sha256="e" * 64,
        promotion_block_evidence=evidence, constraint_violations=0,
        sample_execution_coverage_pass=True, objective_weight_coverage=1.,
        sample_execution=dict(attempted_origin_samples=days * 24,
            succeeded_origin_samples=days * (23 if failed else 24),
            failed_origin_samples=days if failed else 0,
            failed_examples=days * 9 if failed else 0,
            scoring_fallback_examples=days * 9 if failed else 0,
            complete_origin_agent_chains=days * (23 if failed else 24),
            minimum_coverage=.95, coverage_pass=True, strict_agent_chain_pass=not failed),
        targets=[dict(target=t, horizon_hours=h, skill_score=skill,
            sample_execution_coverage=q, n=days * (23 if failed else 24), eligible_rows=days * 24)
            for t in DEFAULT_TARGET_WEIGHTS for h in (1, 6, 24)],
    )
    parsed = _validated_evidence(SimpleNamespace(metrics=metrics))
    score = _resampled_objective(parsed, list(parsed["blocks"]))
    scope = EvaluationScope(
        run_id="run:synthetic-operation-review", generation=0,
        candidate_id="candidate:local" if local else "candidate:" + arm.value,
        candidate_revision_id="revision:" + arm.value,
        phase=EvaluationPhase.FORMAL_BATCH if local else EvaluationPhase.HOLDOUT,
        cohort_digest=digest({"cohort": days}), origin_count=days * 24,
        **(dict(batch_index=1, formal_batch_arm=arm) if local else dict(holdout_arm=arm)),
    )
    return (BatchEvaluation if local else HoldoutEvaluation)(
        evaluation_id="eval:" + arm.value, scope=scope, score=score,
        passed=not failed, metrics=metrics, evaluator_digest="f" * 64,
    )


def local_comparison(champion, challenger, *, marked):
    policy = local_challenger_policy(GUARD_METADATA)
    policy["require_paired_strict_chain"] = marked
    assessment = assess_local_challenger(
        champion, challenger,
        challenger_safety_gate_passed=local_challenger_safety_reason(challenger.metrics) is None,
        **policy,
    )
    return FormalBatchComparison(
        comparison_id="comparison:synthetic", run_id=champion.scope.run_id,
        generation=0, candidate_id=champion.scope.candidate_id, batch_index=1,
        cohort_digest=champion.scope.cohort_digest,
        champion_before_revision_id=champion.scope.candidate_revision_id,
        challenger_revision_id=challenger.scope.candidate_revision_id,
        champion_evaluation_id=champion.evaluation_id, challenger_evaluation_id=challenger.evaluation_id,
        champion_evaluation_digest=champion.evaluation_digest,
        challenger_evaluation_digest=challenger.evaluation_digest,
        champion_score=champion.score, challenger_score=challenger.score,
        score_delta=assessment.score_delta, comparison_contract_digest=assessment.comparison_contract_digest,
        safety_gate_passed=assessment.safety_gate_passed,
        cell_regression_gate_passed=assessment.cell_regression_gate_passed,
        minimum_score_delta=.005, decision=assessment.decision,
        champion_after_revision_id=assessment.champion_after_revision_id, reason=assessment.reason,
        paired_execution_qualification=PAIRED_EXECUTION_QUALIFICATION if marked else None,
    )


def holdout_comparison(*, marked, failed_incumbent=True):
    items = tuple(evaluation(arm, failed=failed_incumbent and arm is HoldoutArm.INCUMBENT,
        skill=.1 if arm is HoldoutArm.INCUMBENT or failed_incumbent else .2) for arm in (HoldoutArm.FINALIST_1, HoldoutArm.INCUMBENT))
    return build_generation_comparison(
        run_id=items[0].scope.run_id, generation=0, cohort_digest=items[0].scope.cohort_digest,
        holdout_evaluations=items, incumbent_candidate_id="candidate:incumbent",
        require_paired_strict_chain=marked,
    )


def complete_chain(evaluation):
    metrics = evaluation.to_dict()["metrics"]
    metrics["sample_execution"]["strict_agent_chain_pass"] = True
    metrics["sample_execution"]["complete_origin_agent_chains"] = evaluation.scope.origin_count
    return replace(evaluation, metrics=metrics)


def with_dataset_task(item, adapter=CUCUMBER_2018):
    """Carry the frozen dataset contract every real run binds to its metrics."""
    contract = adapter.contract()
    metrics = item.to_dict()["metrics"]
    metrics["dataset_task"] = contract
    evidence = metrics["promotion_block_evidence"]
    evidence["dataset_task"] = contract
    evidence["evidence_digest"] = digest(
        {key: value for key, value in evidence.items() if key != "evidence_digest"}
    )
    return replace(item, metrics=metrics)


class PairedExecutionQualificationTests(unittest.TestCase):
    def test_local_availability_only_gain_is_retained_without_changing_score(self):
        champion = evaluation(FormalBatchArm.CHAMPION, failed=True)
        challenger = evaluation(FormalBatchArm.CHALLENGER)
        comparison = local_comparison(champion, challenger, marked=True)
        self.assertAlmostEqual(comparison.score_delta, .0458333333333333)
        self.assertTrue(comparison.cell_regression_gate_passed)
        self.assertEqual(comparison.decision.value, "champion_retained")
        self.assertEqual(comparison.reason, "paired_strict_agent_chain_failed")
        validate_formal_batch_comparison(comparison, champion, challenger,
                                        **local_challenger_policy(GUARD_METADATA))

    def test_local_both_arms_require_explicit_true_even_if_caller_safety_is_true(self):
        originals = (evaluation(FormalBatchArm.CHAMPION), evaluation(FormalBatchArm.CHALLENGER, skill=.2))
        for index in (0, 1):
            for value in (False, None, 1):
                with self.subTest(arm=index, value=value):
                    items = list(originals)
                    metrics = items[index].to_dict()["metrics"]
                    metrics["sample_execution"]["strict_agent_chain_pass"] = value
                    items[index] = replace(items[index], metrics=metrics)
                    result = assess_local_challenger(*items, challenger_safety_gate_passed=True,
                                                     **local_challenger_policy(GUARD_METADATA))
                    self.assertEqual(result.reason, "paired_strict_agent_chain_failed")

    def test_local_scientific_gain_with_complete_chains_still_promotes(self):
        result = local_comparison(evaluation(FormalBatchArm.CHAMPION),
            evaluation(FormalBatchArm.CHALLENGER, skill=.2), marked=True)
        self.assertEqual(result.decision.value, "challenger_promoted")
        original = evaluation(FormalBatchArm.CHAMPION)
        durable = replace(original, metrics=_durable_batch_metrics(original.metrics))
        self.assertTrue(paired_scoring_evidence_complete(durable, durable))

    def test_local_complete_chain_with_failed_scoring_still_cannot_promote(self):
        champion = complete_chain(evaluation(FormalBatchArm.CHAMPION, failed=True))
        challenger = evaluation(FormalBatchArm.CHALLENGER)
        comparison = local_comparison(champion, challenger, marked=True)
        self.assertEqual(comparison.reason, "paired_scoring_evidence_incomplete")
        self.assertEqual(comparison.decision.value, "champion_retained")
        old = local_comparison(champion, challenger, marked=False)
        self.assertEqual(old.decision.value, "challenger_promoted")
        validate_formal_batch_comparison(old, champion, challenger, **local_challenger_policy(GUARD_METADATA))

    def test_durable_local_evidence_preserves_real_dataset_contract_and_gates(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                champion = with_dataset_task(evaluation(FormalBatchArm.CHAMPION))
                challenger = with_dataset_task(evaluation(
                    FormalBatchArm.CHALLENGER, skill=.2, failed=failed))
                originals = (champion, challenger)
                durable = tuple(replace(item, metrics=_durable_batch_metrics(item.metrics))
                                for item in originals)
                self.assertEqual(paired_scoring_evidence_complete(*durable), not failed)
                self.assertEqual(paired_scoring_evidence_complete(*durable),
                                 paired_scoring_evidence_complete(*originals))
                for before, after in zip(originals, durable):
                    self.assertEqual(after.metrics["dataset_task"], before.metrics["dataset_task"])
                    self.assertEqual(after.metrics["promotion_block_evidence"],
                                     before.metrics["promotion_block_evidence"])
                expected = local_comparison(*originals, marked=True)
                actual = local_comparison(*durable, marked=True)
                self.assertEqual(actual.decision, expected.decision)
                self.assertEqual(actual.reason, expected.reason)
                validate_formal_batch_comparison(actual, *durable,
                                                **local_challenger_policy(GUARD_METADATA))


    def test_holdout_max_t_availability_only_gain_cannot_qualify(self):
        comparison = holdout_comparison(marked=True)
        self.assertEqual(comparison.selected_candidate_id, "candidate:incumbent")
        incumbent = comparison.gate_results["arms"]["incumbent"]
        self.assertTrue(incumbent["search_eligible"])
        self.assertFalse(incumbent["certification_eligible"])
        self.assertIn("paired_strict_agent_chain_failed", incumbent["certification_failures"])
        for arm in ("finalist_1",):
            gate = comparison.gate_results["arms"][arm]
            self.assertAlmostEqual(gate["stability_lower_bound"], .0458333333333333)
            self.assertFalse(gate["certification_eligible"])
            self.assertFalse(gate["search_eligible"])
            self.assertIn("paired_strict_agent_chain_failed", gate["certification_failures"])

    def test_holdout_scientific_gain_with_complete_chains_still_qualifies(self):
        comparison = holdout_comparison(marked=True, failed_incumbent=False)
        self.assertEqual(comparison.selected_candidate_id, "candidate:finalist_1")
        self.assertTrue(comparison.gate_results["arms"]["finalist_1"]["search_eligible"])
        self.assertFalse(comparison.gate_results["arms"]["finalist_1"]["certification_eligible"])

    def test_frozen_dataset_contract_still_matches_its_own_block_evidence(self):
        # Every registered dataset contract holds arrays, and `HoldoutEvaluation`
        # deep-freezes `metrics` into mappingproxies and real tuples while
        # `_validated_evidence` thaws its side into dicts and lists. Comparing
        # the two raw can never match, so complete 50-origin evidence was
        # reported `paired_scoring_evidence_incomplete` for both arms of
        # run:e4332050-18c1-4562-8f03-3c4c8ee3a8bf. The whole suite missed it
        # because no other fixture carries a `dataset_task` on a frozen
        # trajectory evaluation.
        contract = CUCUMBER_2018.contract()
        self.assertNotEqual(
            _deep_freeze_json(contract, "dataset_task"),
            contract,
            "fixture no longer exercises the frozen/thawed boundary",
        )

        incumbent, finalist = (
            with_dataset_task(evaluation(arm, skill=skill))
            for arm, skill in (
                (HoldoutArm.INCUMBENT, 0.1),
                (HoldoutArm.FINALIST_1, 0.2),
            )
        )
        for item in (incumbent, finalist):
            self.assertEqual(thaw_json(item.metrics["dataset_task"]), contract)
            self.assertIsNotNone(_validated_evidence(item))
            self.assertTrue(paired_scoring_evidence_complete(item, item))
        self.assertTrue(paired_scoring_evidence_complete(incumbent, finalist))

        report = assess_promotion_improvement(finalist, incumbent)
        self.assertTrue(report["comparable"])
        self.assertNotEqual(report["reason_code"], "incompatible_block_configuration")
        self.assertGreater(report["score_delta"], 0)


    def test_holdout_complete_chain_with_failed_scoring_still_cannot_qualify(self):
        base = holdout_comparison(marked=False)
        items = tuple(complete_chain(item) for item in base.holdout_evaluations)
        for marked in (False, True):
            comparison = build_generation_comparison(run_id=base.run_id, generation=0,
                cohort_digest=base.cohort_digest, holdout_evaluations=items,
                require_paired_strict_chain=marked)
            gate = comparison.gate_results["arms"]["finalist_1"]
            self.assertTrue(gate["strict_agent_chain_pass"])
            self.assertAlmostEqual(gate["stability_lower_bound"], .0458333333333333)
            self.assertFalse(gate["certification_eligible"])
            if marked:
                self.assertFalse(gate["paired_scoring_evidence_complete"])
                self.assertIn("paired_scoring_evidence_incomplete", gate["certification_failures"])
                self.assertNotIn("paired_strict_agent_chain_failed", gate["certification_failures"])

    def test_complete_scoring_qualification_rejects_missing_or_inconsistent_counts(self):
        original = evaluation(FormalBatchArm.CHAMPION)
        self.assertTrue(paired_scoring_evidence_complete(original, original))
        for field in ("attempted_origin_samples", "succeeded_origin_samples", "complete_origin_agent_chains",
                      "failed_origin_samples", "failed_examples", "scoring_fallback_examples"):
            for value in (None, True, -1, 1.5, 999):
                with self.subTest(field=field, value=value):
                    metrics = original.to_dict()["metrics"]
                    if value is None:
                        metrics["sample_execution"].pop(field)
                    else:
                        metrics["sample_execution"][field] = value
                    changed = replace(original, metrics=metrics)
                    self.assertFalse(paired_scoring_evidence_complete(original, changed))
                    self.assertFalse(paired_scoring_evidence_complete(changed, original))
        wrong_scope = replace(original, scope=replace(original.scope, origin_count=73),
                              metrics=original.to_dict()["metrics"])
        self.assertFalse(paired_scoring_evidence_complete(wrong_scope, wrong_scope))

    def test_complete_scoring_qualification_requires_bound_nine_cell_daily_totals(self):
        original = evaluation(FormalBatchArm.CHAMPION)
        def changed(mutator):
            metrics = original.to_dict()["metrics"]
            evidence = metrics["promotion_block_evidence"]
            mutator(evidence)
            evidence["evidence_digest"] = digest({k: v for k, v in evidence.items() if k != "evidence_digest"})
            return replace(original, metrics=metrics)
        mutations = {
            "missing_cell": lambda e: e["blocks"][0]["cells"].pop(),
            "duplicate_cell": lambda e: e["blocks"][0]["cells"].append(dict(e["blocks"][0]["cells"][0])),
            "failed_cell": lambda e: e["blocks"][0]["cells"][0].update(succeeded=23),
            "unequal_cell_day_count": lambda e: e["blocks"][0]["cells"][0].update(eligible=25, succeeded=25),
            "scope_total_mismatch": lambda e: [cell.update(eligible=23, succeeded=23) for cell in e["blocks"][0]["cells"]],
            "empty_day": lambda e: [cell.update(eligible=0, succeeded=0) for cell in e["blocks"][0]["cells"]],
            "duplicate_day_index": lambda e: e["blocks"][1].update(origin_block_index=e["blocks"][0]["origin_block_index"]),
            "missing_day_index": lambda e: e["blocks"][0].pop("origin_block_index"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                altered = changed(mutate)
                self.assertFalse(paired_scoring_evidence_complete(altered, altered))
        different_day = changed(lambda e: e["blocks"][0].update(origin_block_index=99))
        self.assertTrue(paired_scoring_evidence_complete(different_day, different_day))
        self.assertFalse(paired_scoring_evidence_complete(original, different_day))



    def test_unknown_and_non_guard_contracts_fail_closed(self):
        for metadata, marker in ((GUARD_METADATA, "unknown@1"), ({}, PAIRED_EXECUTION_QUALIFICATION)):
            with self.assertRaises(ValueError):
                paired_execution_qualification_required(metadata, marker)
        for metadata in (GUARD_METADATA, {}):
            for new_decision in (False, True):
                with self.subTest(metadata=metadata, new_decision=new_decision), self.assertRaises(ValueError):
                    paired_execution_qualification_required(metadata, None, new_decision=new_decision)



if __name__ == "__main__":
    unittest.main()
