from __future__ import annotations

import unittest
from types import SimpleNamespace

import ecologyrsi_dsh.knowledge.algorithms as algorithms_module
from ecologyrsi_dsh.api.generation_execution import _evaluate_candidate
from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import Evaluation, Proposal, TaskManifest, digest
from ecologyrsi_dsh.evolution.analysis import build_generation_analysis
from ecologyrsi_dsh.evolution.batches import (
    finalize_generation_batch,
    start_generation_batch,
)
from ecologyrsi_dsh.evolution.context import batch_context
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
from ecologyrsi_dsh.knowledge.algorithm_ir import registered_algorithm_blueprint
from ecologyrsi_dsh.knowledge.algorithms import (
    AlgorithmCompileError,
    AlgorithmSpec,
    compile_algorithm_spec,
    debug_algorithm_spec,
    resolve_predictor_adoption,
)
from ecologyrsi_dsh.knowledge.models import KnowledgeCard, KnowledgeSnapshot


class _CapturingGateway:
    def __init__(self) -> None:
        self.contexts: list[dict] = []

    def propose(self, _model_id: str, context: dict, _schemas: dict) -> dict:
        self.contexts.append(context)
        return {
            "parameters": {"alpha": 0.4, "window": 5, "water_threshold": 0.4},
            "rationale": "repair from bounded failure feedback",
        }


def _task(*, strategy_id: str = "adaptive_local@1") -> TaskManifest:
    return TaskManifest(
        task_id="algorithm-compile",
        objective="predict soil water",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={
            "max_candidates": 2,
            "max_generations": 2,
            "candidates_per_generation": 1,
        },
        metadata={
            "domain": "toy",
            "strategy_id": strategy_id,
            "strategy_model_id": "strategy-model"
            if "authenticated" in strategy_id
            else None,
            "prediction_model_id": "toy-rolling-water@1",
            "evaluator_id": "toy_time_forward@1",
        },
    )


class AlgorithmCompilationTests(unittest.TestCase):
    def test_greenhouse_missing_binding_defaults_to_multihorizon_v2(self):
        task = TaskManifest(
            task_id="greenhouse-default-v2",
            objective="predict greenhouse climate",
            domain_pack="greenhouse_environment@1",
            visible_datasets=("agc_cucumber_2018",),
            budget={"max_candidates": 1},
        )
        proposal = Proposal(
            proposal_id="proposal:greenhouse-default-v2",
            run_id="run:greenhouse-default-v2",
            generation=0,
            title="default registered ridge",
            changes={
                "history_steps": 3,
                "ridge_alpha": 0.1,
                "residual_scale": 0.5,
            },
        )

        adoption = resolve_predictor_adoption(task, {})
        compiled = compile_algorithm_spec(task, proposal, None)

        self.assertEqual(
            adoption.evaluator_id, "greenhouse_multihorizon_time_forward@2"
        )
        self.assertEqual(
            compiled.evaluator_id, "greenhouse_multihorizon_time_forward@2"
        )

        legacy_data = task.to_dict()
        legacy_data["metadata"] = {
            "prediction_model_id": "greenhouse-exogenous-ridge@1",
            "evaluator_id": "greenhouse_multihorizon_time_forward@1",
        }
        legacy = compile_algorithm_spec(
            TaskManifest.from_dict(legacy_data), proposal, None
        )
        self.assertEqual(
            legacy.evaluator_id, "greenhouse_multihorizon_time_forward@1"
        )

    def test_capability_versions_match_the_current_runtime_boundaries(self):
        capabilities = algorithms_module._CAPABILITIES
        self.assertEqual(
            {
                key: value["version"]
                for key, value in capabilities["strategy"].items()
            },
            {
                "parameter_sweep@1": "bounded-parent-sweep/6",
                "adaptive_local@1": "bounded-feedback-local-search/6",
                "dsh_authenticated@1": "authenticated-structured-proposal/7",
                "autonomous_model@1": "per-generation-research-runtime-adoption/8",
            },
        )
        self.assertEqual(
            {
                key: value["version"]
                for key, value in capabilities["evaluator"].items()
            },
            {
                "toy_time_forward@1": "toy-time-forward/3",
                "greenhouse_time_forward@1": "greenhouse-time-forward/5",
                "greenhouse_multihorizon_time_forward@1": (
                    "greenhouse-multihorizon-time-forward/4"
                ),
                "greenhouse_multihorizon_time_forward@2": (
                    "greenhouse-multihorizon-time-forward/5"
                ),
            },
        )

    def test_horizon_targetwise_predictor_compiles_to_registered_operator_ir(self):
        task_data = _task().to_dict()
        task_data["visible_datasets"] = ["agc_cucumber_2018"]
        task_data["domain_pack"] = "greenhouse_environment@1"
        task_data["metadata"] = {
            **task_data["metadata"],
            "domain": "greenhouse",
            "prediction_model_id": "greenhouse-horizon-targetwise-ridge@1",
            "evaluator_id": "greenhouse_multihorizon_time_forward@2",
            "dataset_digest": "d" * 64,
            "split_manifest_digest": "s" * 64,
        }
        task = TaskManifest.from_dict(task_data)
        changes = {
            "history_steps": 6,
            "ridge_alpha": 0.05,
            "air_temperature_1h_residual_scale": 0.8,
            "air_temperature_6h_residual_scale": 0.8,
            "air_temperature_24h_residual_scale": 0.8,
            "relative_humidity_1h_residual_scale": 0.7,
            "relative_humidity_6h_residual_scale": 0.7,
            "relative_humidity_24h_residual_scale": 0.7,
            "co2_concentration_1h_residual_scale": 0.0,
            "co2_concentration_6h_residual_scale": 0.8,
            "co2_concentration_24h_residual_scale": 0.8,
        }
        proposal = Proposal(
            proposal_id="proposal:horizon-targetwise-blueprint",
            run_id="run:horizon-targetwise-blueprint",
            generation=0,
            title="registered horizon-targetwise ridge",
            changes=changes,
        )

        spec = compile_algorithm_spec(task, proposal, None)

        self.assertEqual(spec.adapter_id, "greenhouse-horizon-targetwise-ridge@1")
        self.assertEqual(spec.evaluator_id, "greenhouse_multihorizon_time_forward@2")
        assert spec.algorithm_ir is not None
        self.assertEqual(dict(spec.algorithm_ir["parameters"]), changes)
        self.assertIn(
            "host.predictor.horizon-targetwise-residual-or-persistence@1",
            [item["operator_id"] for item in spec.algorithm_ir["operators"]],
        )

    def test_research_blueprint_lowers_to_registered_targetwise_operator_ir(self):
        task_data = _task().to_dict()
        task_data["visible_datasets"] = ["agc_cucumber_2018"]
        task_data["domain_pack"] = "greenhouse_environment@1"
        task_data["metadata"] = {
            **task_data["metadata"],
            "domain": "greenhouse",
            "prediction_model_id": "greenhouse-targetwise-ridge@1",
            "evaluator_id": "greenhouse_multihorizon_time_forward@1",
            "dataset_digest": "d" * 64,
            "split_manifest_digest": "s" * 64,
        }
        task = TaskManifest.from_dict(task_data)
        blueprint = {
            **registered_algorithm_blueprint("greenhouse-targetwise-ridge@1"),
            "evidence_refs": ["knowledge:targetwise-ridge"],
            "rationale": "retain persistence for a weak CO2 correction",
        }
        synthesis = {
            "schema_version": "ecologyrsi-dsh.algorithm-synthesis/1",
            "pipeline_id": "greenhouse-targetwise-ridge@1",
            "evidence_refs": ["knowledge:metadata-only"],
            "parameter_focus": [
                "ridge_alpha",
                "co2_concentration_residual_scale",
            ],
            "rationale": (
                "Translate the frozen research direction into target-specific "
                "shrinkage using only the registered operator graph."
            ),
        }
        snapshot = KnowledgeSnapshot(
            run_id="run:targetwise-blueprint",
            generation=0,
            query_terms=("target-specific ridge shrinkage",),
            cards=(
                KnowledgeCard(
                    knowledge_id="knowledge:targetwise-ridge",
                    title="Target-specific ridge shrinkage",
                    summary="Use target-specific shrinkage when residual quality differs.",
                    source_url="https://example.test/targetwise-ridge",
                    source_kind="curated",
                    source_authority="Example Research Registry",
                    execution_status="available_not_selected",
                    selection_reason="Frozen research evidence for this generation.",
                    capability_kind="predictor",
                    capability_id="greenhouse-targetwise-ridge@1",
                ),
                KnowledgeCard(
                    knowledge_id="knowledge:metadata-only",
                    title="Unverified metadata result",
                    summary="A search result without an executable host mapping.",
                    source_url="https://example.test/metadata-only",
                    source_kind="online_metadata",
                    source_authority="Metadata Index",
                    execution_status="metadata_only",
                    selection_reason="Metadata only; not mapped to a predictor.",
                ),
            ),
            online_enabled=False,
            provider="test registry",
            retrieval_status="catalog_only",
        )
        changes = {
            "history_steps": 6,
            "ridge_alpha": 0.05,
            "air_temperature_residual_scale": 0.8,
            "relative_humidity_residual_scale": 0.7,
            "co2_concentration_residual_scale": 0.0,
        }
        proposal = Proposal(
            proposal_id="proposal:targetwise-blueprint",
            run_id="run:targetwise-blueprint",
            generation=0,
            title="registered targetwise ridge blueprint",
            changes=changes,
            metadata={
                "plan": {
                    "prediction_model": {
                        "id": "greenhouse-targetwise-ridge@1"
                    },
                    "research": [
                        {
                            "title": "target-specific shrinkage",
                            "url": "https://example.test/targetwise-ridge",
                        }
                    ],
                    "algorithm_blueprint": blueprint,
                    "algorithm_synthesis": synthesis,
                }
            },
        )

        spec = compile_algorithm_spec(task, proposal, snapshot)
        debug_evidence = debug_algorithm_spec(spec, task, proposal, snapshot)

        self.assertEqual(spec.adapter_id, "greenhouse-targetwise-ridge@1")
        self.assertTrue(debug_evidence["passed"])
        assert spec.algorithm_ir is not None
        self.assertTrue(spec.algorithm_ir["source_blueprint_digest"])
        self.assertEqual(
            spec.algorithm_ir["source_synthesis_digest"],
            digest(synthesis),
        )
        self.assertIn("+synthesis/", spec.algorithm_version)
        self.assertEqual(
            [item["operator_id"] for item in spec.algorithm_ir["operators"]],
            blueprint["operator_ids"],
        )
        adopted_sources = {
            item["source_url"]
            for item in spec.knowledge_mappings
            if item["decision"] == "adopted"
        }
        self.assertIn("https://example.test/targetwise-ridge", adopted_sources)
        synthesis_mapping = next(
            item
            for item in spec.knowledge_mappings
            if item["knowledge_id"] == "knowledge:metadata-only"
        )
        self.assertEqual(synthesis_mapping["decision"], "research_only")
        self.assertEqual(
            synthesis_mapping["evidence_role"],
            "algorithm_synthesis_source",
        )

        invalid_synthesis_plan = dict(proposal.metadata["plan"])
        invalid_synthesis_plan["algorithm_synthesis"] = {
            **synthesis,
            "parameter_focus": ["model_generated_learning_rate"],
        }
        invalid_synthesis_data = proposal.to_dict()
        invalid_synthesis_data["metadata"] = {"plan": invalid_synthesis_plan}
        with self.assertRaises(AlgorithmCompileError) as invalid_synthesis:
            compile_algorithm_spec(
                task,
                Proposal.from_dict(invalid_synthesis_data),
                snapshot,
            )
        self.assertEqual(
            invalid_synthesis.exception.code,
            "invalid_algorithm_synthesis",
        )

        tampered_plan = dict(proposal.metadata["plan"])
        tampered_plan["algorithm_blueprint"] = {
            **blueprint,
            "operator_ids": ["model.generated.python@1"],
        }
        tampered_data = proposal.to_dict()
        tampered_data["metadata"] = {"plan": tampered_plan}
        with self.assertRaises(AlgorithmCompileError) as caught:
            compile_algorithm_spec(
                task,
                Proposal.from_dict(tampered_data),
                snapshot,
            )
        self.assertEqual(caught.exception.code, "invalid_algorithm_blueprint")

        unknown_ref_plan = dict(proposal.metadata["plan"])
        unknown_ref_plan["algorithm_blueprint"] = {
            **blueprint,
            "evidence_refs": ["knowledge:not-in-snapshot"],
        }
        unknown_ref_plan["algorithm_synthesis"] = {
            **synthesis,
            "evidence_refs": ["knowledge:not-in-snapshot"],
        }
        unknown_ref_data = proposal.to_dict()
        unknown_ref_data["metadata"] = {"plan": unknown_ref_plan}
        with self.assertRaises(AlgorithmCompileError) as unknown_ref:
            compile_algorithm_spec(
                task,
                Proposal.from_dict(unknown_ref_data),
                snapshot,
            )
        self.assertEqual(
            unknown_ref.exception.code,
            "unknown_algorithm_blueprint_evidence",
        )

        unknown_synthesis_plan = dict(proposal.metadata["plan"])
        unknown_synthesis_plan["algorithm_synthesis"] = {
            **synthesis,
            "evidence_refs": ["knowledge:not-in-snapshot"],
        }
        unknown_synthesis_data = proposal.to_dict()
        unknown_synthesis_data["metadata"] = {"plan": unknown_synthesis_plan}
        with self.assertRaises(AlgorithmCompileError) as unknown_synthesis:
            compile_algorithm_spec(
                task,
                Proposal.from_dict(unknown_synthesis_data),
                snapshot,
            )
        self.assertEqual(
            unknown_synthesis.exception.code,
            "invalid_algorithm_synthesis",
        )

        metadata_only_plan = dict(proposal.metadata["plan"])
        metadata_only_plan["algorithm_blueprint"] = {
            **blueprint,
            "evidence_refs": ["knowledge:metadata-only"],
        }
        metadata_only_data = proposal.to_dict()
        metadata_only_data["metadata"] = {"plan": metadata_only_plan}
        with self.assertRaises(AlgorithmCompileError) as metadata_only:
            compile_algorithm_spec(
                task,
                Proposal.from_dict(metadata_only_data),
                snapshot,
            )
        self.assertEqual(
            metadata_only.exception.code,
            "algorithm_blueprint_has_no_executable_evidence",
        )

    def test_compiler_matches_registered_greenhouse_predictor_evaluator_pairs(self):
        ridge_task = _task()
        ridge_task_data = ridge_task.to_dict()
        ridge_task_data["metadata"] = {
            **ridge_task_data["metadata"],
            "prediction_model_id": "greenhouse-exogenous-ridge@1",
            "evaluator_id": "greenhouse_time_forward@1",
        }
        ridge_task = TaskManifest.from_dict(ridge_task_data)
        ridge_proposal = Proposal(
            proposal_id="proposal:ridge-one-hour",
            run_id="run:ridge-one-hour",
            generation=0,
            title="ridge one hour",
            changes={
                "history_steps": 3,
                "ridge_alpha": 0.1,
                "residual_scale": 0.5,
            },
            metadata={
                "plan": {
                    "prediction_model": {
                        "id": "greenhouse-exogenous-ridge@1",
                        "name": "ridge residual forecast",
                    },
                    "research": [
                        {
                            "source": "scikit-learn",
                            "title": "Ridge regression",
                            "url": "https://scikit-learn.org/stable/modules/linear_model.html",
                        }
                    ],
                }
            },
        )
        compiled = compile_algorithm_spec(ridge_task, ridge_proposal, None)
        self.assertEqual(compiled.adapter_id, "greenhouse-exogenous-ridge@1")
        self.assertEqual(compiled.evaluator_id, "greenhouse_time_forward@1")
        plan_mapping = next(
            item
            for item in compiled.knowledge_mappings
            if item["knowledge_id"].startswith("model-plan:predictor:")
        )
        self.assertEqual(plan_mapping["decision"], "adopted")
        self.assertTrue(
            any(
                item["knowledge_id"].startswith("model-research:")
                and item["decision"] == "research_only"
                for item in compiled.knowledge_mappings
            )
        )

        rolling_data = ridge_task.to_dict()
        rolling_data["metadata"] = {
            **rolling_data["metadata"],
            "prediction_model_id": "greenhouse-rolling-residual@1",
            "evaluator_id": "greenhouse_multihorizon_time_forward@1",
        }
        rolling_task = TaskManifest.from_dict(rolling_data)
        rolling_proposal = Proposal(
            proposal_id="proposal:rolling-multihorizon",
            run_id="run:rolling-multihorizon",
            generation=0,
            title="rolling multihorizon",
            changes={"blend": 0.5, "window": 12, "bias_scale": 0.5},
        )
        with self.assertRaisesRegex(
            AlgorithmCompileError, "predictor and evaluator are incompatible"
        ):
            compile_algorithm_spec(rolling_task, rolling_proposal, None)

    def test_multi_card_snapshot_compiles_only_active_registered_bindings(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, StrategyRouterDSHAdapter())
            director.start_evolution(_task(), run_id="run:algorithm-spec")
            batch = start_generation_batch(director, "run:algorithm-spec")
            proposal = director.request_proposal(
                "run:algorithm-spec", generation_batch=batch, slot_index=0
            )
            state = director.state("run:algorithm-spec")
            spec = compile_algorithm_spec(
                state.task_manifest, proposal, state.knowledge_for(0)
            )
            replayed = AlgorithmSpec.from_dict(spec.to_dict())

            self.assertEqual(replayed.spec_digest, spec.spec_digest)
            self.assertTrue(all(item.startswith("host.") for item in spec.tool_ids))
            self.assertTrue(spec.dataset_digest)
            self.assertTrue(spec.split_manifest_digest)
            self.assertEqual(
                spec.allowed_partitions,
                ("training_fit", "training_feedback"),
            )
            decisions = {item["decision"] for item in spec.knowledge_mappings}
            self.assertIn("adopted", decisions)
            self.assertIn("not_selected", decisions)
            self.assertIn("research_only", decisions)
            self.assertFalse(
                spec.to_dict()["security_boundary"]["external_code_execution"]
            )

    def test_compiler_rejects_hidden_or_final_partition_requests(self) -> None:
        for partition in ("hidden", "final", "test", "development", "gate", "external"):
            with self.subTest(partition=partition):
                task_data = _task().to_dict()
                task_data["metadata"] = {
                    **task_data["metadata"],
                    "evaluation_partition": partition,
                }
                proposal = Proposal(
                    proposal_id=f"proposal:forbidden:{partition}",
                    run_id="run:forbidden-partition",
                    generation=0,
                    title="forbidden partition",
                    changes={
                        "alpha": 0.4,
                        "window": 5,
                        "water_threshold": 0.4,
                    },
                )
                with self.assertRaises(AlgorithmCompileError) as caught:
                    compile_algorithm_spec(
                        TaskManifest.from_dict(task_data), proposal, None
                    )
                self.assertEqual(caught.exception.code, "forbidden_data_partition")

    def test_compile_failure_blocks_training_and_reaches_next_model_context(
        self,
    ) -> None:
        gateway = _CapturingGateway()
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger, StrategyRouterDSHAdapter(gateway=gateway)
            )
            task = _task(strategy_id="dsh_authenticated@1")
            director.start_evolution(task, run_id="run:algorithm-repair")
            start_generation_batch(director, "run:algorithm-repair")
            proposal = director.submit_proposal(
                Proposal(
                    proposal_id="proposal:invalid-contract",
                    run_id="run:algorithm-repair",
                    generation=0,
                    title="invalid host contract",
                    changes={
                        "alpha": 0.4,
                        "window": 5,
                        "water_threshold": 0.4,
                        "source_code": 1,
                    },
                )
            )
            candidate = director.spawn_candidate(
                "run:algorithm-repair", proposal, slot_index=0
            )

            class _MustNotTrain:
                calls = 0

                def evaluate_scientific(
                    self, *_args: object, **_kwargs: object
                ) -> object:
                    self.calls += 1
                    raise AssertionError("training must not run before debug passes")

            evaluator = _MustNotTrain()
            endpoint = SimpleNamespace(
                server=SimpleNamespace(director=director, evaluators=evaluator)
            )
            _evaluate_candidate(
                endpoint, "run:algorithm-repair", candidate.candidate_id
            )
            self.assertEqual(evaluator.calls, 0)

            analysis = finalize_generation_batch(director, "run:algorithm-repair")
            self.assertEqual(
                analysis.algorithm_failures[0]["failure_code"],
                "parameter_contract_mismatch",
            )
            algorithm_failure = analysis.algorithm_failures[0]
            self.assertEqual(algorithm_failure["phase"], "compile")
            self.assertEqual(algorithm_failure["stage"], "compile")
            self.assertFalse(algorithm_failure["retryable"])
            self.assertEqual(algorithm_failure["error_type"], "AlgorithmCompileError")
            self.assertEqual(algorithm_failure["attempt_count"], 1)
            self.assertEqual(
                algorithm_failure["details"], {"registered_adapters_only": True}
            )
            self.assertNotIn("public_error", str(algorithm_failure))
            director.advance_generation("run:algorithm-repair")
            next_batch = start_generation_batch(director, "run:algorithm-repair")
            director.request_proposal(
                "run:algorithm-repair", generation_batch=next_batch, slot_index=0
            )
            feedback = gateway.contexts[-1]["previous_generation_analysis"]
            self.assertEqual(
                feedback["algorithm_failures"][0]["failure_code"],
                "parameter_contract_mismatch",
            )
            self.assertNotIn("source_code", str(feedback))

    def test_sample_failure_counts_are_feedback_but_raw_records_are_blocked(
        self,
    ) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, StrategyRouterDSHAdapter())
            director.start_evolution(_task(), run_id="run:sample-feedback")
            batch = start_generation_batch(director, "run:sample-feedback")
            proposal = director.request_proposal(
                "run:sample-feedback", generation_batch=batch, slot_index=0
            )
            candidate = director.spawn_candidate("run:sample-feedback", proposal)
            director.record_evaluation(
                Evaluation(
                    evaluation_id="evaluation:sample-feedback",
                    run_id="run:sample-feedback",
                    candidate_id=candidate.candidate_id,
                    score=0.2,
                    passed=False,
                    metrics={
                        "scientific_pass": False,
                        "sample_execution": {
                            "eligible_examples": 10,
                            "attempted_examples": 10,
                            "succeeded_examples": 8,
                            "failed_examples": 2,
                            "skipped_examples": 2,
                            "coverage": 0.8,
                            "minimum_coverage": 0.7,
                            "coverage_pass": True,
                            "retry_count": 3,
                            "repair_count": 1,
                            "exploration_failures": 2,
                            "recovered_examples": 1,
                            "input_failures": 0,
                            "failure_counts": {"tool_timeout": 2},
                            "critic_outcome_counts": {"rejected": 2, "accepted": 1},
                            "reason_code_counts": {"bounded_repair": 1},
                            "repair_tool_outcomes": {
                                "bounded-projection-repair": {"completed": 1}
                            },
                            "recovered_by_failure_class": {"tool_timeout": 1},
                            "tool_performance": [
                                {
                                    "tool_id": "greenhouse-exogenous-ridge",
                                    "version": "1",
                                    "target": "air_temperature",
                                    "horizon_hours": 1,
                                    "selected": 8,
                                    "completed": 8,
                                    "failed": 0,
                                    "rejected": 0,
                                    "critic_accept": 8,
                                    "critic_repair": 0,
                                    "critic_failed": 0,
                                    "final_accept": 8,
                                    "recovered": 1,
                                    "n": 8,
                                    "mae": 0.2,
                                    "rmse": 0.3,
                                    "baseline_mae": 0.4,
                                    "baseline_rmse": 0.5,
                                    "rmse_improvement": 0.2,
                                    "skill_score": 0.4,
                                }
                            ],
                        },
                        "sample_execution_records": [
                            {"sample_id": "secret-sample", "observed": 0.7}
                        ],
                    },
                )
            )
            analysis = build_generation_analysis(
                director.state("run:sample-feedback"), batch
            )
            self.assertEqual(analysis.sample_failures[0]["failed"], 2)
            self.assertEqual(analysis.sample_failures[0]["repair_count"], 1)
            self.assertEqual(
                analysis.sample_failures[0]["failure_counts"], {"tool_timeout": 2}
            )
            self.assertEqual(analysis.sample_failures[0]["exploration_failures"], 2)
            self.assertEqual(analysis.sample_failures[0]["recovered_examples"], 1)
            self.assertEqual(
                analysis.sample_failures[0]["critic_outcome_counts"],
                {"accepted": 1, "rejected": 2},
            )
            self.assertEqual(
                analysis.sample_failures[0]["repair_tool_outcomes"],
                {"bounded-projection-repair": {"completed": 1}},
            )
            self.assertEqual(
                analysis.ranking[0]["tool_performance"][0]["final_accept"], 8
            )
            self.assertEqual(
                analysis.sample_failures[0]["tool_performance"][0]["skill_score"],
                0.4,
            )
            self.assertNotIn("secret-sample", str(analysis.sample_failures))

            context = batch_context(
                {
                    "generation": 1,
                    "slot_index": 0,
                    "batch_size": 1,
                    "round_parent_candidate_id": None,
                    "previous_generation_analysis": {
                        **analysis.to_dict(),
                        "sample_execution_records": [
                            {"sample_id": "secret-sample", "observed": 0.7}
                        ],
                        "Observed": 0.71,
                        "actual_value": 0.72,
                        "nested": {
                            "Ground-Truth": 0.73,
                            "safe_aggregate_count": 2,
                        },
                    },
                    "knowledge_snapshot": None,
                    "knowledge_snapshot_digest": None,
                    "context_digest": "context:test",
                },
                type("RunStub", (), {"generation": 1})(),
            )
            serialized = str(context)
            self.assertIn("tool_timeout", serialized)
            self.assertNotIn("secret-sample", serialized)
            self.assertNotIn("Observed", serialized)
            self.assertNotIn("actual_value", serialized)
            self.assertNotIn("Ground-Truth", serialized)
            self.assertIn("safe_aggregate_count", serialized)


if __name__ == "__main__":
    unittest.main()
