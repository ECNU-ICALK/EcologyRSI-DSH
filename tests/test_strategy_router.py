from __future__ import annotations

import json
import unittest

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.dsh import FakeDSHAdapter, StrategyRouterDSHAdapter
from ecologyrsi_dsh.evolution.batches import start_generation_batch
from ecologyrsi_dsh.evolution.context import safe_aggregate_feedback
from ecologyrsi_dsh.integrations.model_gateway import GatewayResponseError
from ecologyrsi_dsh.knowledge.algorithms import compile_algorithm_spec
from ecologyrsi_dsh.models import Run, TaskManifest


class _PolicyGatewayStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict]] = []
        self.research_calls: list[tuple[str, dict]] = []

    def catalog(self) -> list[dict]:
        return [{"model_id": "policy-main", "connection": {"state": "available"}}]

    def propose(self, model_id: str, context: dict, allowed_parameters: dict) -> dict:
        self.calls.append((model_id, context, allowed_parameters))
        return {
            "parameters": {"blend": 0.72, "window": 10, "bias_scale": 0.9},
            "rationale": "聚合评测指标显示该区域更稳定",
            "guidance": "保持偏差缩放系数在 1 以下",
        }

    def research_plan(self, model_id: str, context: dict) -> dict:
        self.research_calls.append((model_id, context))
        return {"status": "model_generated", "research": []}


class AggregateFeedbackBoundaryTests(unittest.TestCase):
    def test_sample_execution_detail_is_removed_before_agent_feedback(self) -> None:
        projected = safe_aggregate_feedback(
            {
                "scientific_pass": False,
                "sample_execution": {
                    "failure_preview": [
                        {
                            "attempt_trace": [
                                {"critic_decisions": [{"reason_code": "private"}]}
                            ]
                        }
                    ]
                },
                "sample_execution_trace_archive": "private-archive",
                "sample_execution_coverage": 0.0,
            },
            name="generation judge aggregate metrics",
        )

        self.assertEqual(
            projected,
            {"scientific_pass": False, "sample_execution_coverage": 0.0},
        )


class _UnavailablePolicyGateway(_PolicyGatewayStub):
    def propose(self, model_id: str, context: dict, allowed_parameters: dict) -> dict:
        self.calls.append((model_id, context, allowed_parameters))
        raise GatewayResponseError(
            "policy model request failed: Bearer strategy-audit-secret-token"
        )


class _UnavailableResearchGateway(_PolicyGatewayStub):
    def research_plan(self, model_id: str, context: dict) -> dict:
        self.research_calls.append((model_id, context))
        raise GatewayResponseError(
            "research model queue is saturated",
            retryable=True,
            attempts=4,
            status_code=429,
        )


class _PredictorSelectingGateway:
    def __init__(self, requested_id: str, *, fail_research: bool = False) -> None:
        self.requested_id = requested_id
        self.fail_research = fail_research
        self.research_calls: list[tuple[str, dict]] = []
        self.calls: list[tuple[str, dict, dict]] = []

    def catalog(self) -> list[dict]:
        return []

    def research_plan(self, model_id: str, context: dict) -> dict:
        self.research_calls.append((model_id, context))
        if self.fail_research:
            raise AssertionError("frozen predictor binding must prevent replanning")
        return {"prediction_model": {"id": self.requested_id}}

    def propose(self, model_id: str, context: dict, allowed_parameters: dict) -> dict:
        self.calls.append((model_id, context, allowed_parameters))
        if "blend" in allowed_parameters:
            parameters = {"blend": 0.72, "window": 10, "bias_scale": 0.9}
        else:
            parameters = {
                "history_steps": 6,
                "ridge_alpha": 0.05,
                "residual_scale": 0.75,
            }
        return {"parameters": parameters}


class _MixedSingleParameterSweepGateway(_PredictorSelectingGateway):
    def research_plan(self, model_id: str, context: dict) -> dict:
        self.research_calls.append((model_id, context))
        return {
            "prediction_model": {"id": self.requested_id},
            "strategy": {"name": "bounded_single_parameter_sweep"},
        }

    def propose(self, model_id: str, context: dict, allowed_parameters: dict) -> dict:
        self.calls.append((model_id, context, allowed_parameters))
        if context["slot_index"] == 0:
            parameters = {
                "history_steps": 6,
                "ridge_alpha": 0.05,
                "residual_scale": 0.75,
            }
        else:
            parameters = {
                "history_steps": 3,
                "ridge_alpha": 0.1,
                "residual_scale": 0.9,
            }
        return {"parameters": parameters}


def _task(*, metadata: dict | None = None, domain_pack: str = "crop-soil-water@toy") -> TaskManifest:
    return TaskManifest(
        task_id="strategy-test",
        objective="生成可评测参数",
        domain_pack=domain_pack,
        visible_datasets=("visible@1",),
        budget={"max_candidates": 8},
        seed=5,
        metadata=metadata or {},
    )


def _run(generation: int = 0) -> Run:
    return Run(
        run_id="run:strategy",
        task_id="strategy-test",
        task_manifest_digest="digest",
        generation=generation,
    )


def _autonomous_switch_task(
    *,
    include_rolling: bool = True,
    candidates_per_generation: int = 1,
) -> TaskManifest:
    dataset_id = "agc_cucumber_2018"
    rolling = {
        "id": "greenhouse-rolling-residual@1",
        "dataset_ids": [dataset_id],
        "configuration_digest": "rolling-digest",
        "parameter_schemas": {
            "blend": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "window": {"type": "integer", "minimum": 1, "maximum": 48},
            "bias_scale": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        },
    }
    ridge = {
        "id": "greenhouse-exogenous-ridge@1",
        "dataset_ids": [dataset_id],
        "configuration_digest": "ridge-digest",
        "parameter_schemas": {
            "history_steps": {"type": "integer", "minimum": 1, "maximum": 12},
            "ridge_alpha": {
                "type": "number",
                "minimum": 0.0001,
                "maximum": 1.0,
            },
            "residual_scale": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
    }
    predictors = [ridge, rolling] if include_rolling else [ridge]
    return TaskManifest(
        task_id="predictor-adoption",
        objective="select a registered greenhouse predictor",
        domain_pack="greenhouse_environment@1",
        visible_datasets=(dataset_id,),
        budget={
            "max_candidates": candidates_per_generation,
            "max_generations": 1,
            "candidates_per_generation": candidates_per_generation,
        },
        metadata={
            "domain": "greenhouse",
            "autonomous_mode": True,
            "strategy_id": "autonomous_model@1",
            "strategy_model_id": "policy-main",
            "prediction_model_id": "greenhouse-exogenous-ridge@1",
            "prediction_model_digest": "ridge-digest",
            "evaluator_id": "greenhouse_time_forward@1",
            "dataset_digest": "d" * 64,
            "split_manifest_digest": "s" * 64,
            "runtime_component_catalog": {
                "prediction_models": predictors,
                "evaluators": [],
                "selected_prediction_model_id": "greenhouse-exogenous-ridge@1",
                "selected_evaluator_id": "greenhouse_time_forward@1",
            },
        },
    )


def _greenhouse_parent() -> dict:
    return {
        "candidate_id": "candidate:parent",
        "status": "promoted",
        "proposal_id": "proposal:parent",
        "proposal_parameters": {
            "blend": 0.88,
            "window": 30,
            "bias_scale": 0.2,
        },
        "artifact": None,
        "evaluation": {"score": 0.41, "passed": True},
        "judge": {
            "accepted": True,
            "guidance": "从建议参数继续探索。",
            "parameter_override": {
                "blend": 0.86,
                "window": 25,
                "bias_scale": 0.0,
            },
        },
    }


class StrategyRouterTests(unittest.TestCase):
    def test_autonomous_single_parameter_siblings_share_seed_and_change_one_axis(
        self,
    ) -> None:
        gateway = _MixedSingleParameterSweepGateway(
            "greenhouse-exogenous-ridge@1"
        )
        task = _autonomous_switch_task(
            include_rolling=False,
            candidates_per_generation=2,
        )
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            director.start_evolution(task, run_id="run:single-axis-siblings")
            batch = start_generation_batch(director, "run:single-axis-siblings")
            proposals = [
                director.request_proposal(
                    "run:single-axis-siblings",
                    generation_batch=batch,
                    slot_index=slot_index,
                    consume_interventions=False,
                )
                for slot_index in range(2)
            ]

        seeds = [
            call[1]["search_design"]["host_seed_parameters"]
            for call in gateway.calls
        ]
        self.assertEqual(seeds[0], seeds[1])
        for proposal in proposals:
            changed = [
                name
                for name, value in proposal.changes.items()
                if value != seeds[0][name]
            ]
            self.assertEqual(len(changed), 1)
            audit = proposal.metadata["search_design_audit"]
            self.assertEqual(audit["shared_reference_parameters"], seeds[0])
            self.assertTrue(audit["host_projection_applied"])
        self.assertNotEqual(proposals[0].changes, proposals[1].changes)

    def test_parameter_sweep_generates_chinese_toy_proposal(self) -> None:
        adapter = StrategyRouterDSHAdapter(max_proposals=4)
        run = _run()
        task = _task(metadata={"strategy_id": "parameter_sweep@1", "domain": "toy"})
        session = adapter.open_session(run, task)
        proposal = adapter.propose(run, task, session)
        self.assertEqual(
            proposal.changes,
            {"alpha": 0.2, "window": 3, "water_threshold": 0.35},
        )
        self.assertIn("作物土壤水分", proposal.title)
        self.assertIn("参数扫描策略", proposal.rationale)

    def test_greenhouse_domain_uses_explicit_parameter_names(self) -> None:
        adapter = StrategyRouterDSHAdapter(max_proposals=2)
        run = _run(generation=1)
        task = _task(
            metadata={"strategy_id": "parameter_sweep@1", "domain": "greenhouse"},
            domain_pack="greenhouse-climate@1",
        )
        proposal = adapter.propose(run, task, adapter.open_session(run, task))
        self.assertEqual(set(proposal.changes), {"blend", "window", "bias_scale"})
        self.assertEqual(proposal.changes["window"], 6)
        self.assertIn("温室环境", proposal.title)

    def test_exogenous_ridge_uses_its_own_bounded_parameter_space(self) -> None:
        adapter = StrategyRouterDSHAdapter(max_proposals=2)
        run = _run(generation=1)
        task = _task(
            metadata={
                "strategy_id": "parameter_sweep@1",
                "domain": "greenhouse",
                "prediction_model_id": "greenhouse-exogenous-ridge@1",
            },
            domain_pack="greenhouse-climate@1",
        )
        session = adapter.open_session(run, task)
        proposal = adapter.propose(
            run,
            task,
            session,
            interventions={
                "guidance": "降低岭回归强度",
                "parameter_override": {"history_steps": 8},
            },
        )
        self.assertEqual(
            set(proposal.changes),
            {"history_steps", "ridge_alpha", "residual_scale"},
        )
        self.assertEqual(proposal.changes["history_steps"], 8)
        self.assertAlmostEqual(proposal.changes["ridge_alpha"], 0.0001)
        self.assertIn("外生变量预测", proposal.title)

    def test_horizon_targetwise_ridge_exposes_the_registered_cell_grid(self) -> None:
        adapter = StrategyRouterDSHAdapter(max_proposals=2)
        run = _run(generation=0)
        task = _task(
            metadata={
                "strategy_id": "parameter_sweep@1",
                "domain": "greenhouse",
                "prediction_model_id": "greenhouse-horizon-targetwise-ridge@1",
            },
            domain_pack="greenhouse-climate@1",
        )

        schemas = adapter.parameter_schemas_for_task(task)
        proposal = adapter.propose(run, task, adapter.open_session(run, task))

        expected_scales = {
            f"{target}_{horizon}h_residual_scale"
            for target in (
                "air_temperature",
                "relative_humidity",
                "co2_concentration",
            )
            for horizon in (1, 6, 24)
        }
        self.assertEqual(set(schemas), {"history_steps", "ridge_alpha", *expected_scales})
        self.assertEqual(set(proposal.changes), set(schemas))
        self.assertEqual(proposal.changes["co2_concentration_1h_residual_scale"], 0.0)
        self.assertGreater(
            proposal.changes["co2_concentration_6h_residual_scale"], 0.0
        )
        self.assertGreater(
            proposal.changes["co2_concentration_24h_residual_scale"], 0.0
        )

    def test_generation_one_predictor_switch_keeps_metrics_but_resets_parameters(
        self,
    ) -> None:
        ridge_parameters = {
            "history_steps": 6,
            "ridge_alpha": 0.05,
            "residual_scale": 0.75,
        }
        targetwise_parameters = {
            "history_steps": 6,
            "ridge_alpha": 0.05,
            "air_temperature_residual_scale": 0.75,
            "relative_humidity_residual_scale": 0.75,
            "co2_concentration_residual_scale": 0.0,
        }
        cases = (
            (
                "greenhouse-targetwise-ridge@1",
                ridge_parameters,
                set(targetwise_parameters),
            ),
            (
                "greenhouse-exogenous-ridge@1",
                targetwise_parameters,
                set(ridge_parameters),
            ),
        )
        for predictor_id, parent_parameters, expected_names in cases:
            with self.subTest(predictor_id=predictor_id):
                adapter = StrategyRouterDSHAdapter(max_proposals=2)
                run = _run(generation=1)
                task = _task(
                    metadata={
                        "strategy_id": "parameter_sweep@1",
                        "domain": "greenhouse",
                        "prediction_model_id": predictor_id,
                    },
                    domain_pack="greenhouse-climate@1",
                )
                parent = {
                    "candidate_id": "candidate:parent",
                    "status": "promoted",
                    "proposal_id": "proposal:parent",
                    "proposal_parameters": parent_parameters,
                    "artifact": {"model_id": "previous-registered-predictor"},
                    "evaluation": {"score": 0.31, "passed": False},
                    "judge": {
                        "accepted": False,
                        "parameter_override": dict(parent_parameters),
                    },
                }

                proposal = adapter.propose(
                    run,
                    task,
                    adapter.open_session(run, task),
                    parent_candidate_id="candidate:parent",
                    parent_context=parent,
                )

                self.assertEqual(set(proposal.changes), expected_names)
                self.assertIn("参数空间已切换", proposal.rationale)
                self.assertNotEqual(proposal.changes, parent_parameters)

    def test_predictor_parameter_spaces_cannot_be_mixed(self) -> None:
        adapter = StrategyRouterDSHAdapter(max_proposals=2)
        run = _run()
        ridge_task = _task(
            metadata={
                "strategy_id": "parameter_sweep@1",
                "domain": "greenhouse",
                "prediction_model_id": "greenhouse-exogenous-ridge@1",
            },
            domain_pack="greenhouse-climate@1",
        )
        with self.assertRaisesRegex(ValueError, "unsupported parameters"):
            adapter.propose(
                run,
                ridge_task,
                adapter.open_session(run, ridge_task),
                interventions={"parameter_override": {"window": 6}},
            )

    def test_adaptive_strategy_uses_incumbent_and_intervention(self) -> None:
        adapter = StrategyRouterDSHAdapter(max_proposals=2)
        run = _run()
        task = _task(
            metadata={
                "strategy_id": "adaptive_local@1",
                "incumbent_parameters": {"alpha": 0.5, "window": 8, "water_threshold": 0.4},
            }
        )
        proposal = adapter.propose(
            run,
            task,
            adapter.open_session(run, task),
            interventions={"guidance": "缩短时间窗口", "parameter_override": {"window": 4}},
        )
        self.assertEqual(proposal.changes["window"], 4)
        self.assertNotEqual(proposal.changes["alpha"], 0.5)
        self.assertIn("缩短时间窗口", proposal.rationale)
        with self.assertRaisesRegex(ValueError, "outside the allowed range"):
            adapter.propose(
                run,
                task,
                adapter.open_session(run, task),
                interventions={"parameter_override": {"window": 100}},
            )

    def test_guidance_and_constraint_have_bounded_parameter_effects(self) -> None:
        adapter = StrategyRouterDSHAdapter(max_proposals=2)
        run = _run(generation=2)
        task = _task(
            metadata={"strategy_id": "parameter_sweep@1", "domain": "greenhouse"},
            domain_pack="greenhouse-climate@1",
        )
        proposal = adapter.propose(
            run,
            task,
            adapter.open_session(run, task),
            interventions={
                "guidance": "请降低混合权重",
                "constraints": ["时间窗口在 6 以下", "偏差缩放系数不超过 1"],
            },
        )
        self.assertEqual(proposal.changes["blend"], 0.5)
        self.assertEqual(proposal.changes["window"], 6)
        self.assertEqual(proposal.changes["bias_scale"], 1.0)

    def test_ambiguous_or_negated_guidance_is_not_executed(self) -> None:
        adapter = StrategyRouterDSHAdapter(max_proposals=3)
        run = _run()
        task = _task(metadata={"strategy_id": "parameter_sweep@1", "domain": "toy"})
        session = adapter.open_session(run, task)
        ambiguous = adapter.propose(
            run,
            task,
            session,
            interventions={"guidance": "同时提高 alpha 并缩短时间窗口"},
        )
        negated = adapter.propose(
            run,
            task,
            session,
            interventions={"guidance": "不要提高 alpha"},
        )
        elaborated_negation = adapter.propose(
            run,
            task,
            session,
            interventions={"guidance": "不要大幅提高 alpha"},
        )
        self.assertEqual(ambiguous.changes["alpha"], 0.2)
        self.assertEqual(ambiguous.changes["window"], 3)
        self.assertEqual(negated.changes["alpha"], 0.35)
        self.assertEqual(elaborated_negation.changes["alpha"], 0.5)

    def test_negated_constraint_is_not_executed(self) -> None:
        adapter = StrategyRouterDSHAdapter(max_proposals=2)
        run = _run()
        task = _task(metadata={"strategy_id": "parameter_sweep@1", "domain": "toy"})
        proposal = adapter.propose(
            run,
            task,
            adapter.open_session(run, task),
            interventions={"constraints": ["不要让 window<=1"]},
        )
        self.assertEqual(proposal.changes["window"], 3)

    def test_authenticated_strategy_calls_gateway_and_applies_override(self) -> None:
        gateway = _PolicyGatewayStub()
        adapter = StrategyRouterDSHAdapter(gateway=gateway, max_proposals=2)  # type: ignore[arg-type]
        run = _run()
        task = _task(
            metadata={
                "strategy_id": "dsh_authenticated@1",
                "policy_model_id": "policy-main",
                "domain": "greenhouse",
            },
            domain_pack="greenhouse-climate@1",
        )
        proposal = adapter.propose(
            run,
            task,
            adapter.open_session(run, task),
            interventions={"parameter_override": {"window": 8}},
        )
        self.assertEqual(proposal.changes, {"blend": 0.72, "window": 8, "bias_scale": 0.9})
        self.assertIn("认证模型", proposal.title)
        self.assertEqual(gateway.calls[0][0], "policy-main")
        self.assertEqual(gateway.calls[0][1]["domain"], "greenhouse")
        self.assertEqual(set(gateway.calls[0][2]), {"blend", "window", "bias_scale"})
        self.assertEqual(adapter.connection_catalog[0]["model_id"], "policy-main")

    def test_parent_judge_override_seeds_but_does_not_replace_remote_proposal(self) -> None:
        gateway = _PolicyGatewayStub()
        adapter = StrategyRouterDSHAdapter(gateway=gateway, max_proposals=2)  # type: ignore[arg-type]
        run = _run(generation=1)
        task = _task(
            metadata={
                "strategy_id": "dsh_authenticated@1",
                "policy_model_id": "policy-main",
                "domain": "greenhouse",
            },
            domain_pack="greenhouse-climate@1",
        )
        proposal = adapter.propose(
            run,
            task,
            adapter.open_session(run, task),
            parent_candidate_id="candidate:parent",
            parent_context=_greenhouse_parent(),
        )

        self.assertEqual(
            gateway.calls[0][1]["search_design"]["host_seed_parameters"],
            {"blend": 0.86, "window": 25, "bias_scale": 0.0},
        )
        self.assertEqual(
            proposal.changes,
            {"blend": 0.72, "window": 10, "bias_scale": 0.9},
        )
        self.assertIn("作为本轮策略起点", proposal.rationale)

    def test_parent_context_redacts_case_and_separator_label_aliases(self) -> None:
        gateway = _PolicyGatewayStub()
        adapter = StrategyRouterDSHAdapter(gateway=gateway, max_proposals=2)  # type: ignore[arg-type]
        run = _run(generation=1)
        task = _task(
            metadata={
                "strategy_id": "dsh_authenticated@1",
                "policy_model_id": "policy-main",
                "domain": "greenhouse",
            },
            domain_pack="greenhouse-climate@1",
        )
        parent = _greenhouse_parent()
        parent["evaluation"].update(
            {
                "Observed": 0.7,
                "actualValue": 0.71,
                "nested": {
                    "Ground-Truth": 0.72,
                    "safe_aggregate_count": 3,
                },
            }
        )

        adapter.propose(
            run,
            task,
            adapter.open_session(run, task),
            parent_candidate_id="candidate:parent",
            parent_context=parent,
        )

        remote_parent = gateway.calls[0][1]["parent"]
        serialized = str(remote_parent)
        self.assertNotIn("Observed", serialized)
        self.assertNotIn("actualValue", serialized)
        self.assertNotIn("Ground-Truth", serialized)
        self.assertEqual(
            remote_parent["evaluation"]["nested"]["safe_aggregate_count"], 3
        )

    def test_parent_judge_override_seeds_local_sweep_and_adaptive_search(self) -> None:
        run = _run(generation=1)
        for strategy_id in ("parameter_sweep@1", "adaptive_local@1"):
            with self.subTest(strategy_id=strategy_id):
                adapter = StrategyRouterDSHAdapter(max_proposals=2)
                task = _task(
                    metadata={"strategy_id": strategy_id, "domain": "greenhouse"},
                    domain_pack="greenhouse-climate@1",
                )
                proposal = adapter.propose(
                    run,
                    task,
                    adapter.open_session(run, task),
                    parent_candidate_id="candidate:parent",
                    parent_context=_greenhouse_parent(),
                )

                self.assertNotEqual(
                    proposal.changes,
                    {"blend": 0.86, "window": 25, "bias_scale": 0.0},
                )
                self.assertEqual(proposal.changes["window"], 25 if strategy_id == "parameter_sweep@1" else 24)
                self.assertIn("作为本轮策略起点", proposal.rationale)

    def test_parent_judge_override_seeds_remote_host_fallback(self) -> None:
        gateway = _UnavailablePolicyGateway()
        adapter = StrategyRouterDSHAdapter(gateway=gateway, max_proposals=2)  # type: ignore[arg-type]
        run = _run(generation=1)
        task = _task(
            metadata={
                "strategy_id": "dsh_authenticated@1",
                "policy_model_id": "policy-main",
                "domain": "greenhouse",
            },
            domain_pack="greenhouse-climate@1",
        )

        proposal = adapter.propose(
            run,
            task,
            adapter.open_session(run, task),
            parent_candidate_id="candidate:parent",
            parent_context=_greenhouse_parent(),
        )

        self.assertEqual(
            proposal.changes,
            {"blend": 0.76, "window": 25, "bias_scale": 0.0},
        )
        self.assertEqual(
            proposal.metadata["host_fallback"]["parameter_source"],
            "bounded_parent_sweep",
        )

    def test_greenhouse_batch_reserves_anchor_and_explains_parameter_direction(self) -> None:
        gateway = _PolicyGatewayStub()
        adapter = StrategyRouterDSHAdapter(gateway=gateway, max_proposals=4)  # type: ignore[arg-type]
        run = _run()
        task = _task(
            metadata={
                "strategy_id": "dsh_authenticated@1",
                "policy_model_id": "policy-main",
                "domain": "greenhouse",
            },
            domain_pack="greenhouse-climate@1",
        )
        session = adapter.open_session(run, task)
        shared = {
            "generation": 0,
            "batch_size": 3,
            "round_parent_candidate_id": None,
            "previous_generation_analysis": None,
            "context_digest": "batch-context-digest",
        }
        anchor = adapter.propose(
            run,
            task,
            session,
            batch_context={**shared, "slot_index": 0},
        )
        recovery_seed = adapter.propose(
            run,
            task,
            session,
            batch_context={**shared, "slot_index": 1},
        )
        exploratory = adapter.propose(
            run,
            task,
            session,
            batch_context={**shared, "slot_index": 2},
        )

        self.assertEqual(
            anchor.changes,
            {"blend": 1.0, "window": 24, "bias_scale": 0.0},
        )
        self.assertEqual(anchor.metadata["proposal_source"], "host_reserved_seed")
        self.assertFalse(anchor.metadata["remote_strategy_called"])
        self.assertIn("持续性诊断锚点", anchor.rationale)
        self.assertEqual(
            recovery_seed.changes,
            {"blend": 0.93, "window": 25, "bias_scale": 0.0},
        )
        self.assertEqual(
            recovery_seed.metadata["proposal_source"], "host_reserved_seed"
        )
        self.assertIn("保守恢复种子", recovery_seed.rationale)
        self.assertEqual(
            exploratory.changes,
            {"blend": 0.72, "window": 10, "bias_scale": 0.9},
        )
        self.assertEqual(exploratory.metadata["proposal_source"], "remote_model")
        self.assertTrue(exploratory.metadata["remote_strategy_succeeded"])
        self.assertEqual(len(gateway.calls), 1)
        context = gateway.calls[0][1]
        self.assertEqual(
            context["search_design"]["host_seed_parameters"],
            {"blend": 0.87, "window": 24, "bias_scale": 0.0},
        )
        self.assertIn(
            "Increasing blend gives more weight to the latest observation",
            context["parameter_semantics"]["blend"],
        )
        self.assertIn(
            "Zero disables the learned bias correction",
            context["parameter_semantics"]["bias_scale"],
        )

    def test_small_greenhouse_batches_always_include_a_remote_proposal(self) -> None:
        for batch_size, expected_sources in (
            (1, ["remote_model"]),
            (2, ["host_reserved_seed", "remote_model"]),
        ):
            with self.subTest(batch_size=batch_size):
                gateway = _PolicyGatewayStub()
                adapter = StrategyRouterDSHAdapter(  # type: ignore[arg-type]
                    gateway=gateway, max_proposals=batch_size
                )
                run = _run()
                task = _task(
                    metadata={
                        "strategy_id": "dsh_authenticated@1",
                        "policy_model_id": "policy-main",
                        "domain": "greenhouse",
                    },
                    domain_pack="greenhouse-climate@1",
                )
                session = adapter.open_session(run, task)
                shared = {
                    "generation": 0,
                    "batch_size": batch_size,
                    "round_parent_candidate_id": None,
                    "previous_generation_analysis": None,
                    "context_digest": "small-batch-context",
                }
                proposals = [
                    adapter.propose(
                        run,
                        task,
                        session,
                        batch_context={**shared, "slot_index": slot_index},
                    )
                    for slot_index in range(batch_size)
                ]
                self.assertEqual(
                    [item.metadata["proposal_source"] for item in proposals],
                    expected_sources,
                )
                self.assertEqual(len(gateway.calls), 1)

    def test_remote_gateway_failure_uses_audited_host_fallback_for_slot(self) -> None:
        gateway = _UnavailablePolicyGateway()
        adapter = StrategyRouterDSHAdapter(gateway=gateway, max_proposals=3)  # type: ignore[arg-type]
        run = _run()
        task = _task(
            metadata={
                "strategy_id": "dsh_authenticated@1",
                "policy_model_id": "policy-main",
                "domain": "greenhouse",
            },
            domain_pack="greenhouse-climate@1",
        )
        session = adapter.open_session(run, task)
        shared = {
            "generation": 0,
            "batch_size": 3,
            "round_parent_candidate_id": None,
            "previous_generation_analysis": None,
            "context_digest": "batch-context-digest",
        }
        adapter.propose(run, task, session, batch_context={**shared, "slot_index": 0})
        adapter.propose(run, task, session, batch_context={**shared, "slot_index": 1})
        fallback = adapter.propose(
            run,
            task,
            session,
            batch_context={**shared, "slot_index": 2},
        )

        self.assertEqual(
            fallback.changes,
            {"blend": 0.87, "window": 24, "bias_scale": 0.0},
        )
        self.assertIn("宿主有界回退参数", fallback.rationale)
        audit = fallback.metadata["host_fallback"]
        self.assertTrue(audit["applied"])
        self.assertEqual(audit["reason"], "remote_strategy_gateway_error")
        self.assertEqual(audit["error_type"], "GatewayResponseError")
        self.assertEqual(
            audit["public_error"],
            "GatewayResponseError [gateway_response_error]",
        )
        self.assertNotIn("strategy-audit-secret-token", json.dumps(fallback.to_dict()))
        self.assertEqual(audit["parameter_source"], "host_seed_parameters")
        self.assertEqual(audit["slot_index"], 2)
        self.assertEqual(audit["batch_size"], 3)
        self.assertEqual(fallback.metadata["proposal_source"], "host_fallback")
        self.assertTrue(fallback.metadata["remote_strategy_called"])
        self.assertFalse(fallback.metadata["remote_strategy_succeeded"])
        self.assertEqual(len(gateway.calls), 1)

    def test_research_gateway_failure_uses_same_audited_fallback_policy(self) -> None:
        gateway = _UnavailableResearchGateway()
        adapter = StrategyRouterDSHAdapter(gateway=gateway, max_proposals=1)  # type: ignore[arg-type]
        run = _run()
        task = _task(
            metadata={
                "autonomous_mode": True,
                "strategy_id": "autonomous_model@1",
                "policy_model_id": "policy-main",
                "domain": "toy",
                "remote_fallback_policy": "record_and_continue",
            }
        )

        proposal = adapter.propose(
            run,
            task,
            adapter.open_session(run, task),
        )

        self.assertEqual(proposal.metadata["proposal_source"], "host_fallback")
        audit = proposal.metadata["host_fallback"]
        self.assertEqual(audit["reason"], "remote_research_plan_gateway_error")
        self.assertEqual(audit["operation"], "research_plan")
        self.assertEqual(audit["error_type"], "GatewayResponseError")
        self.assertEqual(len(gateway.research_calls), 1)
        self.assertEqual(gateway.calls, [])

    def test_retryable_research_failure_is_rethrown_for_run_retry(self) -> None:
        gateway = _UnavailableResearchGateway()
        adapter = StrategyRouterDSHAdapter(  # type: ignore[arg-type]
            gateway=gateway,
            max_proposals=1,
        )
        run = _run()
        task = _task(
            metadata={
                "autonomous_mode": True,
                "strategy_id": "autonomous_model@1",
                "policy_model_id": "policy-main",
                "domain": "toy",
                "remote_fallback_policy": "fail_run",
            }
        )

        with self.assertRaises(GatewayResponseError) as caught:
            adapter.propose(run, task, adapter.open_session(run, task))

        self.assertTrue(caught.exception.retryable)
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(gateway.calls, [])

    def test_research_plan_receives_runtime_boundary_and_hard_gates(self) -> None:
        gateway = _PolicyGatewayStub()
        adapter = StrategyRouterDSHAdapter(gateway=gateway)  # type: ignore[arg-type]
        objective_profile = {
            "id": "greenhouse-skill@1",
            "hard_gates": ["all_target_skills_non_negative"],
        }
        component_catalog = {
            "prediction_models": [{"id": "greenhouse-rolling-residual@1"}],
            "selected_prediction_model_id": "greenhouse-rolling-residual@1",
        }
        task = _task(
            metadata={
                "domain": "greenhouse",
                "objective_profile": objective_profile,
                "runtime_component_catalog": component_catalog,
            },
            domain_pack="greenhouse-climate@1",
        )
        schemas = adapter.parameter_schemas_for_task(task)
        schemas["blend"]["maximum"] = 0.5
        self.assertEqual(
            adapter.parameter_schemas_for_task(task)["blend"]["maximum"],
            1.0,
        )

        adapter.research_plan(
            "policy-main",
            run=_run(),
            task=task,
            parameter_schemas=adapter.parameter_schemas_for_task(task),
        )
        context = gateway.research_calls[0][1]
        self.assertEqual(context["runtime_component_catalog"], component_catalog)
        self.assertEqual(context["objective_profile"], objective_profile)
        self.assertEqual(
            context["hard_gates"],
            ["all_target_skills_non_negative"],
        )
        self.assertEqual(
            set(context["allowed_parameter_schemas"]),
            {"blend", "window", "bias_scale"},
        )

    def test_deferred_plan_adopts_compatible_predictor_before_proposal(self) -> None:
        gateway = _PredictorSelectingGateway("greenhouse-rolling-residual@1")
        adapter = StrategyRouterDSHAdapter(gateway=gateway)  # type: ignore[arg-type]
        task = _autonomous_switch_task()
        run = Run(
            run_id="run:predictor-switch",
            task_id=task.task_id,
            task_manifest_digest=task.digest,
        )

        proposal = adapter.propose(
            run,
            task,
            adapter.open_session(run, task),
        )

        self.assertEqual(
            set(gateway.research_calls[0][1]["allowed_parameter_schemas"]),
            {"history_steps", "ridge_alpha", "residual_scale"},
        )
        self.assertEqual(
            set(gateway.calls[0][2]),
            {"blend", "window", "bias_scale"},
        )
        self.assertEqual(
            gateway.calls[0][1]["prediction_model_id"],
            "greenhouse-rolling-residual@1",
        )
        adoption = proposal.metadata["prediction_model_adoption"]
        self.assertEqual(adoption["status"], "adopted")
        self.assertEqual(
            adoption["adopted_id"], "greenhouse-rolling-residual@1"
        )
        spec = compile_algorithm_spec(task, proposal, None)
        self.assertEqual(spec.adapter_id, "greenhouse-rolling-residual@1")
        self.assertEqual(spec.predictor_adoption, adoption)

    def test_unknown_or_incompatible_plan_predictor_is_research_only(self) -> None:
        for requested_id, include_rolling in (
            ("model-generated-python@1", True),
            ("greenhouse-rolling-residual@1", False),
        ):
            with self.subTest(requested_id=requested_id):
                gateway = _PredictorSelectingGateway(requested_id)
                adapter = StrategyRouterDSHAdapter(  # type: ignore[arg-type]
                    gateway=gateway
                )
                task = _autonomous_switch_task(include_rolling=include_rolling)
                run = Run(
                    run_id=f"run:rejected:{include_rolling}",
                    task_id=task.task_id,
                    task_manifest_digest=task.digest,
                )
                proposal = adapter.propose(
                    run,
                    task,
                    adapter.open_session(run, task),
                )

                adoption = proposal.metadata["prediction_model_adoption"]
                self.assertEqual(adoption["status"], "research_only")
                self.assertEqual(
                    adoption["adopted_id"], "greenhouse-exogenous-ridge@1"
                )
                self.assertEqual(
                    set(gateway.calls[0][2]),
                    {"history_steps", "ridge_alpha", "residual_scale"},
                )
                self.assertEqual(
                    compile_algorithm_spec(task, proposal, None).adapter_id,
                    "greenhouse-exogenous-ridge@1",
                )

    def test_restart_replays_first_proposal_predictor_adoption(self) -> None:
        task = _autonomous_switch_task(candidates_per_generation=2)
        first_gateway = _PredictorSelectingGateway(
            "greenhouse-rolling-residual@1"
        )
        with EventLedger() as ledger:
            first_director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=first_gateway),  # type: ignore[arg-type]
            )
            first_director.start_evolution(task, run_id="run:adoption-replay")
            batch = start_generation_batch(first_director, "run:adoption-replay")
            first = first_director.request_proposal(
                "run:adoption-replay",
                generation_batch=batch,
                slot_index=0,
                consume_interventions=False,
            )
            first_director.spawn_candidate(
                "run:adoption-replay", first, slot_index=0
            )

            replay_gateway = _PredictorSelectingGateway(
                "greenhouse-exogenous-ridge@1",
                fail_research=True,
            )
            replayed_director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=replay_gateway),  # type: ignore[arg-type]
            )
            second = replayed_director.request_proposal(
                "run:adoption-replay",
                generation_batch=batch,
                slot_index=1,
            )

            self.assertEqual(replay_gateway.research_calls, [])
            self.assertEqual(
                second.metadata["prediction_model_adoption"],
                first.metadata["prediction_model_adoption"],
            )
            self.assertEqual(second.metadata["plan"], first.metadata["plan"])
            self.assertEqual(
                set(second.changes), {"blend", "window", "bias_scale"}
            )

    def test_behavior_changes_bump_strategy_configuration_digests(self) -> None:
        implementations = {
            "parameter_sweep@1": ("bounded-parent-sweep/6", "bounded-parent-sweep/5"),
            "adaptive_local@1": (
                "bounded-feedback-local-search/6",
                "bounded-feedback-local-search/5",
            ),
            "dsh_authenticated@1": (
                "authenticated-structured-proposal/7",
                "authenticated-structured-proposal/6",
            ),
            "autonomous_model@1": (
                "per-generation-research-runtime-adoption/12",
                "per-generation-research-runtime-adoption/11",
            ),
        }
        for strategy_id, (
            implementation,
            previous_implementation,
        ) in implementations.items():
            with self.subTest(strategy_id=strategy_id):
                current_digest = digest(
                    {
                        "strategy_id": strategy_id,
                        "implementation": implementation,
                        "host_parameter_boundary": "prediction-model-specific/1",
                    }
                )
                previous_digest = digest(
                    {
                        "strategy_id": strategy_id,
                        "implementation": previous_implementation,
                        "host_parameter_boundary": "prediction-model-specific/1",
                    }
                )
                self.assertEqual(
                    StrategyRouterDSHAdapter.configuration_digest(strategy_id),
                    current_digest,
                )
                self.assertNotEqual(
                    StrategyRouterDSHAdapter.configuration_digest(strategy_id),
                    previous_digest,
                )

    def test_invalid_strategy_domain_and_session_fail_closed(self) -> None:
        adapter = StrategyRouterDSHAdapter()
        run = _run()
        with self.assertRaisesRegex(RuntimeError, "not open"):
            adapter.propose(run, _task(), "wrong-session")
        task = _task(metadata={"strategy_id": "unbounded@1"})
        with self.assertRaisesRegex(ValueError, "unsupported strategy_id"):
            adapter.propose(run, task, adapter.open_session(run, task))
        task = _task(metadata={"domain": "ocean"})
        with self.assertRaisesRegex(ValueError, "unsupported task domain"):
            adapter.propose(run, task, adapter.open_session(run, task))

    def test_fake_adapter_remains_call_compatible(self) -> None:
        adapter = FakeDSHAdapter(max_proposals=1)
        run = _run()
        task = _task()
        proposal = adapter.propose(
            run,
            task,
            adapter.open_session(run, task),
            interventions={"guidance": "ignored by compatibility adapter"},
        )
        self.assertEqual(proposal.changes["alpha"], 0.2)


if __name__ == "__main__":
    unittest.main()
