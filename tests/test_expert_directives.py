"""The expert -> model directive chain under the DSH-native protocol.

Before these tests existed, the native dispatch dropped `interventions` on the
floor: an expert paused a run, wrote a sentence, the web UI printed 已应用, and
the proposer never saw a word of it. The two halves pinned here are the fix and
its honesty condition -- the text reaches the model's stage context, and the
receipt stops claiming an execution that the native protocol cannot perform.
"""

from __future__ import annotations

import unittest

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import (
    HumanIntervention,
    InterventionKind,
    TaskManifest,
    canonical_json,
    digest,
)
from ecologyrsi_dsh.evolution.batches import start_generation_batch
from ecologyrsi_dsh.evolution.interventions import (
    _guidance_step,
    _parse_constraint,
    _parse_guidance,
    advisory_only_receipts,
    expert_directive_context,
)
from ecologyrsi_dsh.evolution.strategies import (
    StrategyRouterDSHAdapter,
    _native_expert_directives,
)
from ecologyrsi_dsh.api.projection import _intervention_projection
from tests.test_autonomous_search_reflection_cycle import (
    _CycleRuntime,
    _task as _autonomous_task,
)


def _native_task() -> TaskManifest:
    return TaskManifest(
        task_id="expert-directives",
        objective="predict greenhouse climate",
        domain_pack="greenhouse_environment@1",
        visible_datasets=("agc_cucumber_2018",),
        budget={"max_candidates": 4, "candidates_per_generation": 1},
        metadata={
            "execution_protocol": "dsh_native_plugin_evolution@1",
            "seed_genome_template_id": "greenhouse-default@1",
            "prediction_model_id": "greenhouse-horizon-targetwise-ridge@1",
            "evaluator_id": "greenhouse_multihorizon_time_forward@2",
            "strategy_id": "autonomous_model@1",
            "dataset_digest": "2" * 64,
            "dataset_snapshot_set_digest": "2" * 64,
            "split_manifest_digest": "3" * 64,
            "data_protocol_digest": "4" * 64,
            "stage_policy_digest": "5" * 64,
            "evaluator_digest": "6" * 64,
            "fitness_profile_digest": "7" * 64,
            "security_kernel_digest": "8" * 64,
            "selection_reviewer_program_digest": "9" * 64,
            "required_capability_digest": "a" * 64,
            "resolved_policy_route_digest": "b" * 64,
            "resolved_review_route_digest": "c" * 64,
            "resolved_policy_route_config_digest": "b" * 64,
            "resolved_review_route_config_digest": "c" * 64,
            "preset_content_digest": "d" * 64,
            "standing_tool_surface_digest": "e" * 64,
            "evaluation_cohort_digest": "f" * 64,
        },
    )


class _MutationRuntime:
    """Return one fixed bounded mutation and keep every stage context."""

    def __init__(self, *, ridge_alpha: float = 0.2) -> None:
        self.structured = {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [
                {
                    "op": "set_bounded_parameter",
                    "name": "ridge_alpha",
                    "value": ridge_alpha,
                }
            ],
        }
        self.requests: list[dict] = []

    def run_stage(self, request: dict) -> dict:
        self.requests.append(request)
        return {
            "structured": self.structured,
            "result_digest": digest(self.structured),
        }

    def contexts(self, stage: str) -> list[dict]:
        return [
            item["request"]["context"]
            for item in self.requests
            if item["stage"] == stage
        ]


class _NativeProposalAdapter:
    """The native proposer without the remote research planner.

    `start_generation_batch` calls `research_iteration` whenever the adapter
    exposes one, which would need a live gateway. Withholding the attribute
    takes the host-fallback research path and leaves `candidate.propose` --
    the stage this module is about -- on the real `_propose_native` code.
    """

    def __init__(self, runtime: _MutationRuntime) -> None:
        self._inner = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )

    def open_session(self, run, task) -> str:
        return self._inner.open_session(run, task)

    def close_session(self, session_id: str) -> None:
        self._inner.close_session(session_id)

    def propose(self, *args, **kwargs):
        return self._inner.propose(*args, **kwargs)


class NativeExpertDirectiveChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.addCleanup(self.ledger.close)
        self.runtime = _MutationRuntime()
        self.adapter = _NativeProposalAdapter(self.runtime)
        self.director = EvolutionDirector(self.ledger, self.adapter)
        self.run_id = "run:expert-directives"
        self.director.create_run(_native_task(), run_id=self.run_id)
        self.director.start_run(self.run_id)

    def _record(self, intervention: HumanIntervention) -> None:
        self.director.pause_run(self.run_id)
        self.director.record_intervention(intervention)
        self.director.resume_run(self.run_id)

    def _propose(self):
        batch = start_generation_batch(self.director, self.run_id)
        return self.director.request_proposal(
            self.run_id,
            generation_batch=batch,
            slot_index=0,
        )

    def test_guidance_reaches_the_native_proposer_as_an_advisory_directive(
        self,
    ) -> None:
        self._record(
            HumanIntervention(
                intervention_id="expert-guidance-1",
                run_id=self.run_id,
                kind=InterventionKind.GUIDANCE,
                message="夜间湿度误差偏大，建议提高岭回归强度",
                created_by="温室专家",
            )
        )

        proposal = self._propose()

        directives = self.runtime.contexts("candidate.propose")[0][
            "expert_directives"
        ]
        self.assertEqual(
            directives["schema_version"], "ecologyrsi-dsh.expert-directives/1"
        )
        self.assertEqual(
            [row["message"] for row in directives["directives"]],
            ["夜间湿度误差偏大，建议提高岭回归强度"],
        )
        self.assertEqual(directives["directives"][0]["kind"], "guidance")
        self.assertTrue(
            directives["policy"]["advisory_directives_are_not_host_enforced"]
        )
        # The genome stays the single source of truth: the Host must not
        # rewrite `changes` to a human-stepped number the run never executes.
        self.assertEqual(proposal.changes["ridge_alpha"], 0.2)

    def test_a_run_without_interventions_omits_the_directive_key_entirely(
        self,
    ) -> None:
        self._propose()

        context = self.runtime.contexts("candidate.propose")[0]
        self.assertNotIn("expert_directives", context)

    def test_domain_knowledge_is_delivered_without_any_parameter_claim(self) -> None:
        self._record(
            HumanIntervention(
                intervention_id="expert-knowledge-1",
                run_id=self.run_id,
                kind=InterventionKind.DOMAIN_KNOWLEDGE,
                message="黄瓜夜间蒸腾滞后约两小时，湿度预测应考虑更长滞后结构",
                created_by="温室专家",
            )
        )

        proposal = self._propose()

        directives = self.runtime.contexts("candidate.propose")[0][
            "expert_directives"
        ]
        self.assertEqual(
            [row["kind"] for row in directives["directives"]], ["domain_knowledge"]
        )
        self.assertEqual(
            directives["directives"][0]["host_enforcement"],
            "advisory_only_host_did_not_enforce",
        )

        state = self.director.state(self.run_id)
        receipt = next(
            event.payload
            for event in state.events
            if event.kind == "HumanInterventionApplied"
        )
        self.assertEqual(receipt["application_status"], "recorded")
        self.assertFalse(receipt["applied"])
        self.assertTrue(receipt["model_context_delivered"])
        self.assertNotIn("parameter", receipt)
        self.assertEqual(proposal.changes["ridge_alpha"], 0.2)

    def test_a_parseable_guidance_receipt_stops_claiming_host_enforcement(
        self,
    ) -> None:
        self._record(
            HumanIntervention(
                intervention_id="expert-guidance-2",
                run_id=self.run_id,
                kind=InterventionKind.GUIDANCE,
                message="提高岭回归强度",
                created_by="温室专家",
            )
        )

        self._propose()

        state = self.director.state(self.run_id)
        receipt = next(
            event.payload
            for event in state.events
            if event.kind == "HumanInterventionApplied"
        )
        self.assertEqual(receipt["application_status"], "recorded")
        self.assertTrue(receipt["model_context_delivered"])
        # What the parser read is preserved as a record, not as an execution.
        self.assertEqual(receipt["host_did_not_enforce"]["parameter"], "ridge_alpha")
        self.assertEqual(receipt["host_did_not_enforce"]["direction"], "increase")
        self.assertIn("宿主未改写", receipt["reason"])

        projected = _intervention_projection(state, state.interventions[0])
        self.assertEqual(projected["status"], "模型已读取（未强制执行）")
        self.assertTrue(projected["model_context_delivered"])

    def test_parent_selection_keeps_its_enforcement_receipt(self) -> None:
        # The batch really does pick the parent, so that receipt was never a
        # claim about proposal parameters and must survive the downgrade.
        receipts = advisory_only_receipts(
            [
                {"kind": "parent_selection", "intervention_id": "p-1"},
                {"kind": "guidance", "intervention_id": "g-1"},
            ],
            [
                {
                    "intervention_id": "p-1",
                    "kind": "parent_selection",
                    "application_status": "enforced",
                    "applied": True,
                    "enforced": True,
                    "reason": "人工选择的父候选已用于本轮提案",
                },
                {
                    "intervention_id": "g-1",
                    "kind": "guidance",
                    "application_status": "applied",
                    "applied": True,
                    "enforced": False,
                    "reason": "已按固定步长应用人工调整指引",
                    "parameter": "ridge_alpha",
                },
            ],
        )
        self.assertEqual(receipts[0]["application_status"], "enforced")
        self.assertNotIn("model_context_delivered", receipts[0])
        self.assertEqual(receipts[1]["application_status"], "recorded")
        self.assertEqual(
            receipts[1]["host_did_not_enforce"], {"parameter": "ridge_alpha"}
        )


class ResearchStageDirectiveTests(unittest.TestCase):
    """The research and search stages must see the expert's reasoning too.

    `candidate.propose` only picks the numeric move inside an already-chosen
    direction. The stages that choose *what to investigate* -- the search plan
    and the research synthesis -- ran with no view of a pending intervention at
    all, so the planner kept proposing directions the expert had argued against.
    """

    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.addCleanup(self.ledger.close)
        self.runtime = _CycleRuntime()
        self.director = EvolutionDirector(
            self.ledger,
            StrategyRouterDSHAdapter(
                gateway=object(),
                native_runtime_provider=lambda: self.runtime,
            ),
        )
        self.run_id = "run:research-directives"
        self.director.create_run(_autonomous_task(), run_id=self.run_id)
        self.director.start_run(self.run_id)

    def _contexts(self, stage: str) -> list[dict]:
        return [
            item["request"]["context"]
            for item in self.runtime.requests
            if item["stage"] == stage
        ]

    def test_both_direction_choosing_stages_receive_the_directives(self) -> None:
        self.director.pause_run(self.run_id)
        self.director.record_intervention(
            HumanIntervention(
                intervention_id="expert-knowledge-1",
                run_id=self.run_id,
                kind=InterventionKind.DOMAIN_KNOWLEDGE,
                message="黄瓜夜间蒸腾滞后约两小时，优先研究更长滞后结构",
                created_by="温室专家",
            )
        )
        self.director.resume_run(self.run_id)

        start_generation_batch(self.director, self.run_id)

        search_context = self._contexts("generation.search-plan")[0]
        self.assertEqual(
            [row["message"] for row in search_context["expert_directives"]["directives"]],
            ["黄瓜夜间蒸腾滞后约两小时，优先研究更长滞后结构"],
        )
        synthesis_context = self._contexts("generation.research-synthesis")[0]
        collaboration = synthesis_context["expert_collaboration"]
        self.assertEqual(
            [row["kind"] for row in collaboration["expert_directives"]["directives"]],
            ["domain_knowledge"],
        )
        self.assertTrue(
            collaboration["expert_directives"]["policy"][
                "advisory_directives_are_not_host_enforced"
            ]
        )

    def test_a_run_without_interventions_omits_the_key_at_both_stages(self) -> None:
        start_generation_batch(self.director, self.run_id)

        self.assertNotIn(
            "expert_directives", self._contexts("generation.search-plan")[0]
        )
        collaboration = self._contexts("generation.research-synthesis")[0][
            "expert_collaboration"
        ]
        self.assertNotIn("expert_directives", collaboration)


class BoundedParserBoundaryTests(unittest.TestCase):
    """The enforcement parser must follow the run's own parameter boundary.

    It used to index a domain-keyed alias table, so every predictor boundary
    added after that table was written raised `KeyError` out of
    `apply_bounded_interventions` -- a run with any pending guidance could not
    produce a proposal at all.
    """

    def test_every_live_boundary_parses_without_a_lookup_error(self) -> None:
        from ecologyrsi_dsh.evolution.strategies import (
            StrategyRouterDSHAdapter as _Adapter,
        )

        boundaries = (
            _Adapter._TOY_SCHEMAS,
            _Adapter._GREENHOUSE_SCHEMAS,
            _Adapter._GREENHOUSE_RIDGE_SCHEMAS,
            _Adapter._GREENHOUSE_ALIGNED_RIDGE_SCHEMAS,
            _Adapter._GREENHOUSE_TARGETWISE_RIDGE_SCHEMAS,
            _Adapter._GREENHOUSE_HORIZON_TARGETWISE_RIDGE_SCHEMAS,
            _Adapter._GREENHOUSE_RECIPE_RIDGE_SCHEMAS,
        )
        for schemas in boundaries:
            with self.subTest(parameters=sorted(schemas)):
                self.assertIsInstance(
                    _parse_guidance("提高岭回归强度", schemas), (str, tuple)
                )
                self.assertIsInstance(
                    _parse_constraint("ridge_alpha<=0.5", schemas), (str, tuple)
                )

    def test_a_boundary_only_parameter_gets_a_span_derived_step(self) -> None:
        # `air_temperature_1h_residual_scale` exists only in the horizon
        # boundary and has no hand-written step, so the step comes off its span.
        schemas = {
            "air_temperature_1h_residual_scale": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            }
        }
        parsed = _parse_guidance("提高 air_temperature_1h_residual_scale", schemas)
        self.assertEqual(
            parsed, ("air_temperature_1h_residual_scale", "increase")
        )
        self.assertEqual(
            _guidance_step(
                "air_temperature_1h_residual_scale",
                schemas["air_temperature_1h_residual_scale"],
            ),
            0.1,
        )


class ExpertDirectiveContextTests(unittest.TestCase):
    def test_numeric_and_structural_kinds_are_not_advisory_text(self) -> None:
        context = expert_directive_context(
            [
                {
                    "intervention_id": "a",
                    "kind": "parameter_override",
                    "message": "alpha=0.8",
                },
                {
                    "intervention_id": "b",
                    "kind": "parent_selection",
                    "message": "继续演化这个候选",
                },
                {"intervention_id": "c", "kind": "constraint", "message": "window<=4"},
            ]
        )
        self.assertEqual(
            [row["intervention_id"] for row in context["directives"]], ["c"]
        )

    def test_blank_text_and_over_limit_rows_are_dropped(self) -> None:
        controls = [
            {"intervention_id": f"k-{index}", "kind": "domain_knowledge", "message": "x"}
            for index in range(12)
        ]
        controls.insert(0, {"intervention_id": "blank", "kind": "guidance", "message": "  "})
        context = expert_directive_context(controls)
        self.assertEqual(len(context["directives"]), 7)
        self.assertNotIn(
            "blank", [row["intervention_id"] for row in context["directives"]]
        )

    def test_receipts_are_reflected_in_the_host_enforcement_field(self) -> None:
        context = expert_directive_context(
            [{"intervention_id": "g", "kind": "guidance", "message": "提高岭回归强度"}],
            receipts=[
                {
                    "application_status": "enforced",
                    "parameter": "ridge_alpha",
                    "reason": "参数约束已由宿主在最终提案边界强制执行",
                }
            ],
        )
        row = context["directives"][0]
        self.assertEqual(row["host_enforcement"], "host_enforced")
        self.assertEqual(row["host_enforced_parameter"], "ridge_alpha")
        self.assertIn("强制执行", row["host_reason"])


class AdapterInterventionTranslationTests(unittest.TestCase):
    def test_the_adapter_mapping_becomes_ordered_advisory_rows(self) -> None:
        context = _native_expert_directives(
            {
                "host_applies_interventions": True,
                "guidance": "提高岭回归强度 | 缩短历史窗口",
                "constraints": ["ridge_alpha<=0.5", "  "],
                "domain_knowledge": ["夜间蒸腾滞后约两小时"],
                "parameter_override": {"ridge_alpha": 0.3},
            }
        )
        assert context is not None
        self.assertEqual(
            [row["kind"] for row in context["directives"]],
            ["guidance", "constraint", "domain_knowledge"],
        )

    def test_an_empty_or_absent_mapping_yields_no_context(self) -> None:
        self.assertIsNone(_native_expert_directives(None))
        self.assertIsNone(_native_expert_directives({}))
        self.assertIsNone(
            _native_expert_directives(
                {"host_applies_interventions": True, "guidance": "   "}
            )
        )
        self.assertIsNone(
            _native_expert_directives(
                {"parameter_override": {"ridge_alpha": 0.3}}
            )
        )


class DirectiveContextDigestStabilityTests(unittest.TestCase):
    def test_the_directive_block_is_the_only_stage_context_difference(self) -> None:
        # `_native_stage_identity` folds `digest(context)` into the phenotype
        # instance digest. An unconditional `expert_directives: None` key would
        # therefore change every archived native run's identity, so the key must
        # be absent -- not null -- when no expert has written anything.
        contexts = []
        for intervention in (None, "提高岭回归强度"):
            ledger = EventLedger()
            self.addCleanup(ledger.close)
            runtime = _MutationRuntime()
            director = EvolutionDirector(ledger, _NativeProposalAdapter(runtime))
            run_id = "run:digest-stability"
            director.create_run(_native_task(), run_id=run_id)
            director.start_run(run_id)
            if intervention is not None:
                director.pause_run(run_id)
                director.record_intervention(
                    HumanIntervention(
                        intervention_id="expert-1",
                        run_id=run_id,
                        kind=InterventionKind.GUIDANCE,
                        message=intervention,
                        created_by="温室专家",
                    )
                )
                director.resume_run(run_id)
            batch = start_generation_batch(director, run_id)
            director.request_proposal(
                run_id, generation_batch=batch, slot_index=0
            )
            contexts.append(runtime.contexts("candidate.propose")[0])

        plain, guided = contexts
        self.assertEqual(set(guided) - set(plain), {"expert_directives"})
        self.assertEqual(set(plain) - set(guided), set())
        # Recording an intervention is itself a ledger event, so the knowledge
        # snapshot and everything digesting it legitimately move. Those are
        # provenance, not context shape; every other key must be byte-identical
        # so the directive block is the only *content* the feature introduced.
        expected_provenance_drift = {
            "expert_directives",
            "diagnostic_report",
            "mutation_context",
            "research_iteration",
            "generation_context_digest",
        }
        drifted = {
            key
            for key in guided
            if canonical_json(guided[key]) != canonical_json(plain.get(key))
        }
        self.assertEqual(drifted, expected_provenance_drift)


if __name__ == "__main__":
    unittest.main()
