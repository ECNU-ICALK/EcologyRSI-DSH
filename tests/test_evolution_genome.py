from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

from ecologyrsi_dsh.core.models import Proposal, TaskManifest
from ecologyrsi_dsh.evolution.genome import (
    EcologyEvolutionPluginGenome,
    FrozenRunInitialization,
    GenomeMutationContextV1,
    apply_genome_mutation,
    deep_freeze_json,
    deep_thaw_json,
    legacy_genome_from_proposal,
    materialize_seed_genome,
    migrate_legacy_seed,
)
from ecologyrsi_dsh.knowledge.models import KnowledgeSnapshot
from ecologyrsi_dsh.knowledge.algorithms import compile_algorithm_spec
from ecologyrsi_dsh.knowledge.program_registry import (
    LEGACY_PROGRAM_CATALOG_0_2_2,
    current_program_registry,
)


_FIXTURES = Path(__file__).with_name("fixtures")


def _initialization(**changes: object) -> FrozenRunInitialization:
    values: dict[str, object] = {
        "run_id": "run:genome-test",
        "task_manifest_digest": "1" * 64,
        "dataset_snapshot_set_digest": "2" * 64,
        "split_manifest_digest": "3" * 64,
        "data_protocol_digest": "4" * 64,
        "stage_policy_digest": "5" * 64,
        "evaluator_digest": "6" * 64,
        "fitness_profile_digest": "7" * 64,
        "security_kernel_digest": "8" * 64,
        "selection_reviewer_program_digest": "9" * 64,
        "protocol": "dsh_native_plugin_evolution@1",
        "required_capability_digest": "a" * 64,
        "resolved_policy_route_digest": "b" * 64,
        "resolved_review_route_digest": "c" * 64,
        "registry_catalog_digest": current_program_registry().catalog_digest,
        "compiler_digest": "d" * 64,
    }
    values.update(changes)
    return FrozenRunInitialization(**values)


def _seed_genome() -> EcologyEvolutionPluginGenome:
    registry = current_program_registry()
    return materialize_seed_genome(
        registry.seed_template("greenhouse-default@1"),
        _initialization(registry_catalog_digest=registry.catalog_digest),
    )


class EvolutionGenomeTests(unittest.TestCase):
    def test_genome_nested_state_cannot_mutate_after_validation(self) -> None:
        genome = _seed_genome()

        with self.assertRaises(TypeError):
            genome.scientific_program["parameter_overrides"] = {"ridge_alpha": 9}
        with self.assertRaises(TypeError):
            genome.agent_program["candidate_execution_program"]["role_profiles"][0][
                "enabled_tool_ids"
            ] = ()

    def test_to_dict_returns_detached_deep_copy(self) -> None:
        genome = _seed_genome()
        first = genome.to_dict()
        first["scientific_program"]["parameter_overrides"]["ridge_alpha"] = 999
        first["agent_program"]["candidate_execution_program"]["role_profiles"][0][
            "enabled_tool_ids"
        ].append("host.shell")

        second = genome.to_dict()
        self.assertNotEqual(
            second["scientific_program"]["parameter_overrides"]["ridge_alpha"],
            999,
        )
        self.assertNotIn(
            "host.shell",
            second["agent_program"]["candidate_execution_program"]["role_profiles"][
                0
            ]["enabled_tool_ids"],
        )

    def test_deep_freeze_normalizes_unicode_keys_strings_and_negative_zero(self) -> None:
        frozen = deep_freeze_json({"e\u0301": ["e\u0301", -0.0]})
        self.assertEqual(deep_thaw_json(frozen), {"é": ["é", 0.0]})

    def test_digest_normalization_rejects_bool_nan_infinity_and_duplicate_set_items(
        self,
    ) -> None:
        registry = current_program_registry()
        template = registry.seed_template("greenhouse-default@1").to_dict()
        template.pop("template_digest")
        template["scientific_program"]["parameter_overrides"]["history_steps"] = True
        with self.assertRaisesRegex((TypeError, ValueError), "bool|number|integer"):
            type(registry.seed_template("greenhouse-default@1")).from_dict(template)

        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                deep_freeze_json({"value": bad})

        genome = _seed_genome().to_dict()
        tools = genome["agent_program"]["candidate_execution_program"][
            "role_profiles"
        ][0]["enabled_tool_ids"]
        tools.append(tools[0])
        genome.pop("genome_id")
        genome.pop("genome_digest")
        genome.pop("behavior_digest")
        with self.assertRaisesRegex(ValueError, "unique"):
            EcologyEvolutionPluginGenome.from_dict(genome)

    def test_domain_separated_deterministic_genome_identity(self) -> None:
        one = _seed_genome()
        two = EcologyEvolutionPluginGenome.from_dict(one.to_dict())

        self.assertEqual(one.genome_digest, two.genome_digest)
        self.assertEqual(one.genome_id, f"genome:{one.genome_digest[:24]}")
        self.assertNotEqual(one.genome_digest, one.behavior_digest)

    def test_workflow_graph_cannot_be_supplied_when_template_is_authoritative(
        self,
    ) -> None:
        value = _seed_genome().to_dict()
        value["agent_program"]["candidate_execution_program"]["nodes"] = []
        value.pop("genome_id")
        value.pop("genome_digest")
        value.pop("behavior_digest")
        with self.assertRaisesRegex(ValueError, "unsupported|nodes"):
            EcologyEvolutionPluginGenome.from_dict(value)

    def test_arbitrary_prompt_script_and_command_fields_are_rejected(self) -> None:
        for field_name in ("prompt", "system_script", "shell_command", "module_url"):
            with self.subTest(field_name=field_name):
                value = _seed_genome().to_dict()
                profile = value["agent_program"]["candidate_execution_program"][
                    "role_profiles"
                ][0]
                profile["instruction_parameters"][field_name] = "execute arbitrary text"
                value.pop("genome_id")
                value.pop("genome_digest")
                value.pop("behavior_digest")
                with self.assertRaisesRegex(ValueError, "instruction|unsupported"):
                    EcologyEvolutionPluginGenome.from_dict(value)

    def test_candidate_cannot_select_its_own_proposer_critic_or_judge(self) -> None:
        parent = _seed_genome()
        context = GenomeMutationContextV1(
            run_id="run:genome-test",
            generation=0,
            slot_index=0,
            slot_seed=42,
            parent_candidate_id=None,
            parent_genome_digest=parent.genome_digest,
            generation_batch_digest="e" * 64,
            research_iteration_digest="f" * 64,
            knowledge_snapshot_digest="0" * 64,
            mutation_budget_digest="1" * 64,
            mutation_operator_id="bounded-single-parent-mutation@1",
        )
        for role in ("candidate-proposer", "sample-critic", "generation-judge"):
            with self.subTest(role=role), self.assertRaisesRegex(
                ValueError, "reviewer|role|reproduction"
            ):
                apply_genome_mutation(
                    parent,
                    {
                        "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                        "operations": [
                            {
                                "op": "select_instruction_template",
                                "role": role,
                                "instruction_template_id": "sample-planner@1",
                            }
                        ],
                    },
                    context,
                    current_program_registry(),
                )

    def test_mutation_digest_covers_accepted_mutation_and_full_context(self) -> None:
        parent = _seed_genome()
        base = dict(
            run_id="run:genome-test",
            generation=0,
            slot_index=0,
            slot_seed=42,
            parent_candidate_id=None,
            parent_genome_digest=parent.genome_digest,
            generation_batch_digest="e" * 64,
            research_iteration_digest="f" * 64,
            knowledge_snapshot_digest="0" * 64,
            mutation_budget_digest="1" * 64,
            mutation_operator_id="bounded-single-parent-mutation@1",
        )
        mutation = {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [
                {
                    "op": "set_bounded_parameter",
                    "name": "ridge_alpha",
                    "value": 0.2,
                }
            ],
        }
        first = apply_genome_mutation(
            parent, mutation, GenomeMutationContextV1(**base), current_program_registry()
        )
        replay = apply_genome_mutation(
            parent, mutation, GenomeMutationContextV1(**base), current_program_registry()
        )
        changed = apply_genome_mutation(
            parent,
            mutation,
            GenomeMutationContextV1(**{**base, "slot_seed": 43}),
            current_program_registry(),
        )

        self.assertEqual(first.genome_digest, replay.genome_digest)
        self.assertEqual(first.lineage["mutation_digest"], replay.lineage["mutation_digest"])
        self.assertNotEqual(
            first.lineage["mutation_digest"], changed.lineage["mutation_digest"]
        )

    def test_first_generation_lineage_uses_materialized_seed_parent_and_null_parent_candidate(
        self,
    ) -> None:
        parent = _seed_genome()
        context = GenomeMutationContextV1(
            run_id="run:genome-test",
            generation=0,
            slot_index=1,
            slot_seed=7,
            parent_candidate_id=None,
            parent_genome_digest=parent.genome_digest,
            generation_batch_digest="e" * 64,
            research_iteration_digest="f" * 64,
            knowledge_snapshot_digest="0" * 64,
            mutation_budget_digest="1" * 64,
            mutation_operator_id="bounded-single-parent-mutation@1",
        )
        child = apply_genome_mutation(
            parent,
            {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": []},
            context,
            current_program_registry(),
        )
        self.assertIsNone(child.lineage["parent_candidate_id"])
        self.assertEqual(child.lineage["parent_genome_digest"], parent.genome_digest)
        self.assertEqual(child.lineage["generation"], 0)
        self.assertEqual(child.lineage["slot_index"], 1)

    def test_seed_template_cannot_contain_run_specific_bindings(self) -> None:
        template_type = type(
            current_program_registry().seed_template("greenhouse-default@1")
        )
        value = current_program_registry().seed_template(
            "greenhouse-default@1"
        ).to_dict()
        value["runtime_binding"] = {"protocol": "forbidden"}
        with self.assertRaisesRegex(ValueError, "run-specific|unsupported"):
            template_type.from_dict(value)

    def test_materialized_seed_contains_all_frozen_run_bindings(self) -> None:
        initialization = _initialization()
        genome = materialize_seed_genome(
            current_program_registry().seed_template("greenhouse-default@1"),
            initialization,
        )
        result = genome.to_dict()

        self.assertEqual(result["lineage"]["origin_kind"], "seed_catalog")
        self.assertIsNone(result["lineage"]["generation"])
        self.assertEqual(result["runtime_binding"], initialization.bindings.runtime_binding)
        self.assertEqual(
            result["frozen_contract_refs"], initialization.bindings.frozen_contract_refs
        )
        self.assertNotIn("compiler_digest", result)
        self.assertNotIn("run_id", result)

    def test_tool_policy_can_only_narrow_registered_base_policy(self) -> None:
        parent = _seed_genome()
        context = GenomeMutationContextV1(
            run_id="run:genome-test",
            generation=1,
            slot_index=0,
            slot_seed=11,
            parent_candidate_id="candidate:parent",
            parent_genome_digest=parent.genome_digest,
            generation_batch_digest="e" * 64,
            research_iteration_digest="f" * 64,
            knowledge_snapshot_digest="0" * 64,
            mutation_budget_digest="1" * 64,
            mutation_operator_id="bounded-single-parent-mutation@1",
        )
        with self.assertRaisesRegex(ValueError, "subset|tool"):
            apply_genome_mutation(
                parent,
                {
                    "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                    "operations": [
                        {
                            "op": "narrow_role_tool_policy",
                            "role": "sample-planner",
                            "enabled_tool_ids": [
                                "ecology_execute_prediction_tool",
                                "host.shell",
                            ],
                        }
                    ],
                },
                context,
                current_program_registry(),
            )

    def test_legacy_no_snapshot_projection_preserves_none_and_full_ir(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "legacy_program_0_2_2_no_snapshot.json").read_text(
                encoding="utf-8"
            )
        )
        projected = legacy_genome_from_proposal(
            Proposal.from_dict(fixture["proposal"]),
            TaskManifest.from_dict(fixture["task"]),
            None,
        )
        result = projected.to_dict()

        self.assertTrue(result["projected"])
        self.assertIsNone(result["knowledge_snapshot_digest"])
        self.assertEqual(result["legacy_algorithm_ir"], fixture["legacy_algorithm_ir"])
        with self.assertRaises(TypeError):
            projected.legacy_algorithm_ir["predictor_id"] = "changed"

    def test_legacy_projection_ignores_current_registry_changes(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "legacy_program_0_2_2.json").read_text(encoding="utf-8")
        )
        snapshot = KnowledgeSnapshot.from_dict(fixture["knowledge_snapshot"])
        before = legacy_genome_from_proposal(
            Proposal.from_dict(fixture["proposal"]),
            TaskManifest.from_dict(fixture["task"]),
            snapshot,
        )
        changed_registry = current_program_registry().with_program_override(
            "predictors",
            "toy-rolling-water@1",
            {"version": "future-incompatible/999"},
        )
        after = legacy_genome_from_proposal(
            Proposal.from_dict(fixture["proposal"]),
            TaskManifest.from_dict(fixture["task"]),
            snapshot,
            legacy_catalog=LEGACY_PROGRAM_CATALOG_0_2_2,
            current_registry=changed_registry,
        )

        self.assertEqual(before.projection_digest, after.projection_digest)
        self.assertEqual(before.to_dict(), after.to_dict())
        self.assertEqual(before.legacy_algorithm_ir, fixture["legacy_algorithm_ir"])
        historical = compile_algorithm_spec(
            TaskManifest.from_dict(fixture["task"]),
            Proposal.from_dict(fixture["proposal"]),
            snapshot,
        )
        self.assertEqual(historical.algorithm_ir, fixture["legacy_algorithm_ir"])

    def test_projected_legacy_requires_explicit_migration_seed(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "legacy_program_0_2_2_no_snapshot.json").read_text(
                encoding="utf-8"
            )
        )
        projected = legacy_genome_from_proposal(
            Proposal.from_dict(fixture["proposal"]),
            TaskManifest.from_dict(fixture["task"]),
            None,
        )
        with self.assertRaisesRegex(TypeError, "projected|migration"):
            apply_genome_mutation(
                projected,  # type: ignore[arg-type]
                {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": []},
                object(),  # type: ignore[arg-type]
                current_program_registry(),
            )

        migrated = migrate_legacy_seed(
            projected,
            _initialization(),
            current_program_registry().migration_template("legacy-dsh-native@1"),
        )
        self.assertEqual(migrated.lineage["origin_kind"], "legacy_migration")
        self.assertEqual(
            migrated.lineage["migration_source"]["projection_digest"],
            projected.projection_digest,
        )


if __name__ == "__main__":
    unittest.main()
