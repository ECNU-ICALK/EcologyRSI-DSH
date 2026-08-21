from __future__ import annotations

import json
from pathlib import Path
import unittest

from ecologyrsi_dsh.core.models import Proposal, TaskManifest
from ecologyrsi_dsh.evolution.genome import (
    FrozenRunInitialization,
    GenomeMutationContextV1,
    apply_genome_mutation,
    legacy_genome_from_proposal,
    materialize_seed_genome,
)
from ecologyrsi_dsh.evolution.workflow_ir import (
    CompilationInstanceContext,
    bind_phenotype_instance,
    compile_dsh_workflow_spec,
    compile_legacy_algorithm_ir,
    compile_plugin_behavior,
)
from ecologyrsi_dsh.knowledge.models import KnowledgeSnapshot
from ecologyrsi_dsh.knowledge.program_registry import (
    ProgramRegistrySnapshot,
    current_program_registry,
)


_FIXTURES = Path(__file__).with_name("fixtures")


def _task() -> TaskManifest:
    return TaskManifest(
        task_id="workflow-compiler",
        objective="predict greenhouse climate",
        domain_pack="greenhouse_environment@1",
        visible_datasets=("agc_cucumber_2018",),
        budget={"max_candidates": 2},
        metadata={
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
        },
    )


def _initialization(task: TaskManifest) -> FrozenRunInitialization:
    metadata = task.metadata
    registry = current_program_registry()
    return FrozenRunInitialization(
        run_id="run:workflow-compiler",
        task_manifest_digest=task.digest,
        dataset_snapshot_set_digest=metadata["dataset_snapshot_set_digest"],
        split_manifest_digest=metadata["split_manifest_digest"],
        data_protocol_digest=metadata["data_protocol_digest"],
        stage_policy_digest=metadata["stage_policy_digest"],
        evaluator_digest=metadata["evaluator_digest"],
        fitness_profile_digest=metadata["fitness_profile_digest"],
        security_kernel_digest=metadata["security_kernel_digest"],
        selection_reviewer_program_digest=metadata[
            "selection_reviewer_program_digest"
        ],
        protocol="dsh_native_plugin_evolution@1",
        required_capability_digest="a" * 64,
        resolved_policy_route_digest="b" * 64,
        resolved_review_route_digest="c" * 64,
        registry_catalog_digest=registry.catalog_digest,
        compiler_digest="d" * 64,
    )


def _seed(task: TaskManifest | None = None):
    task = task or _task()
    registry = current_program_registry()
    return materialize_seed_genome(
        registry.seed_template("greenhouse-default@1"), _initialization(task)
    )


def _child(task: TaskManifest, *, slot_index: int, slot_seed: int):
    parent = _seed(task)
    context = GenomeMutationContextV1(
        run_id="run:workflow-compiler",
        generation=0,
        slot_index=slot_index,
        slot_seed=slot_seed,
        parent_candidate_id=None,
        parent_genome_digest=parent.genome_digest,
        generation_batch_digest="e" * 64,
        research_iteration_digest="f" * 64,
        knowledge_snapshot_digest="0" * 64,
        mutation_budget_digest="1" * 64,
        mutation_operator_id="bounded-single-parent-mutation@1",
    )
    return apply_genome_mutation(
        parent,
        {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": []},
        context,
        current_program_registry(),
    )


def _instance_context(
    task: TaskManifest,
    *,
    candidate_id: str,
    slot_index: int,
    policy_route: str = "b" * 64,
) -> CompilationInstanceContext:
    metadata = task.metadata
    return CompilationInstanceContext(
        run_id="run:workflow-compiler",
        proposal_id=f"proposal:{slot_index}",
        candidate_id=candidate_id,
        generation=0,
        slot_index=slot_index,
        task_manifest_digest=task.digest,
        dataset_snapshot_set_digest=metadata["dataset_snapshot_set_digest"],
        split_manifest_digest=metadata["split_manifest_digest"],
        data_protocol_digest=metadata["data_protocol_digest"],
        stage_policy_digest=metadata["stage_policy_digest"],
        evaluator_digest=metadata["evaluator_digest"],
        evaluation_cohort_digest="a" * 64,
        required_capability_digest="a" * 64,
        resolved_policy_route_config_digest=policy_route,
        resolved_review_route_config_digest="c" * 64,
        preset_content_digest="d" * 64,
        standing_tool_surface_digest="e" * 64,
        security_kernel_digest=metadata["security_kernel_digest"],
    )


class WorkflowIRTests(unittest.TestCase):
    def test_one_resolved_behavior_has_one_compiled_behavior_digest(self) -> None:
        task = _task()
        genome = _child(task, slot_index=0, slot_seed=10)
        registry = current_program_registry()

        first = compile_plugin_behavior(genome, task, None, registry)
        replay = compile_plugin_behavior(genome, task, None, registry)

        self.assertEqual(first.compiled_behavior_digest, replay.compiled_behavior_digest)
        self.assertEqual(first.to_dict(), replay.to_dict())
        self.assertEqual(
            first.algorithm_behavior["predictor_id"],
            "greenhouse-horizon-targetwise-ridge@1",
        )

    def test_behavior_identical_siblings_share_compiled_digest_but_not_instance_digest(
        self,
    ) -> None:
        task = _task()
        first_genome = _child(task, slot_index=0, slot_seed=10)
        second_genome = _child(task, slot_index=1, slot_seed=11)

        first_behavior = compile_plugin_behavior(
            first_genome, task, None, current_program_registry()
        )
        second_behavior = compile_plugin_behavior(
            second_genome, task, None, current_program_registry()
        )
        first_bound = bind_phenotype_instance(
            first_behavior,
            _instance_context(task, candidate_id="candidate:first", slot_index=0),
        )
        second_bound = bind_phenotype_instance(
            second_behavior,
            _instance_context(task, candidate_id="candidate:second", slot_index=1),
        )

        self.assertNotEqual(first_genome.genome_digest, second_genome.genome_digest)
        self.assertEqual(
            first_behavior.compiled_behavior_digest,
            second_behavior.compiled_behavior_digest,
        )
        self.assertNotEqual(
            first_bound.phenotype_instance_digest,
            second_bound.phenotype_instance_digest,
        )

    def test_instance_identifiers_never_enter_compiled_behavior_digest(self) -> None:
        task = _task()
        behavior = compile_plugin_behavior(
            _child(task, slot_index=0, slot_seed=10),
            task,
            None,
            current_program_registry(),
        )
        encoded = json.dumps(behavior.behavior_identity_dict(), sort_keys=True)
        for forbidden in (
            "run:workflow-compiler",
            "proposal:",
            "candidate:",
            "slot_index",
            "genome_id",
            "genome_digest",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_same_genome_under_different_compiler_has_distinct_compiled_behavior_digest(
        self,
    ) -> None:
        task = _task()
        genome = _child(task, slot_index=0, slot_seed=10)
        registry = current_program_registry()
        first = compile_plugin_behavior(
            genome, task, None, registry, compiler_semantic_digest="1" * 64
        )
        second = compile_plugin_behavior(
            genome, task, None, registry, compiler_semantic_digest="2" * 64
        )
        self.assertNotEqual(first.compiled_behavior_digest, second.compiled_behavior_digest)

    def test_resolved_route_config_is_covered_by_runtime_execution_digest(self) -> None:
        task = _task()
        behavior = compile_plugin_behavior(
            _child(task, slot_index=0, slot_seed=10),
            task,
            None,
            current_program_registry(),
        )
        first = bind_phenotype_instance(
            behavior,
            _instance_context(task, candidate_id="candidate:first", slot_index=0),
        )
        second = bind_phenotype_instance(
            behavior,
            _instance_context(
                task,
                candidate_id="candidate:first",
                slot_index=0,
                policy_route="f" * 64,
            ),
        )
        self.assertEqual(
            first.compiled_behavior_digest, second.compiled_behavior_digest
        )
        self.assertNotEqual(
            first.runtime_execution_digest, second.runtime_execution_digest
        )

    def test_omitted_workflow_defaults_equal_explicit_defaults(self) -> None:
        task = _task()
        explicit = _child(task, slot_index=0, slot_seed=10)
        omitted_value = explicit.to_dict()
        omitted_value["agent_program"]["candidate_execution_program"][
            "workflow_overrides"
        ] = {}
        omitted_value.pop("genome_id")
        omitted_value.pop("genome_digest")
        omitted_value.pop("behavior_digest")
        omitted = type(explicit).from_dict(omitted_value)

        explicit_behavior = compile_plugin_behavior(
            explicit, task, None, current_program_registry()
        )
        omitted_behavior = compile_plugin_behavior(
            omitted, task, None, current_program_registry()
        )
        self.assertEqual(
            explicit_behavior.compiled_behavior_digest,
            omitted_behavior.compiled_behavior_digest,
        )

    def test_workflow_registry_rejects_cycles_and_mixed_reviewer_privilege(self) -> None:
        programs = current_program_registry().to_dict()["programs"]
        candidate = programs["workflow_templates"]["candidate-sample-execution@1"]
        candidate["graph"]["edges"] = [
            {"from": "sample-plan", "to": "sample-plan"}
        ]
        with self.assertRaisesRegex(ValueError, "cycle"):
            ProgramRegistrySnapshot.from_programs(programs)

        programs = current_program_registry().to_dict()["programs"]
        candidate = programs["workflow_templates"]["candidate-sample-execution@1"]
        candidate["graph"]["nodes"].append(
            {
                "id": "judge",
                "role": "generation-judge",
                "script_id": "fixed-generation-judge@1",
            }
        )
        candidate["graph"]["allowed_roles"].append("generation-judge")
        with self.assertRaisesRegex(ValueError, "reviewer|privilege|mixed"):
            ProgramRegistrySnapshot.from_programs(programs)

    def test_workflow_compile_rejects_ref_drift_and_tool_expansion(self) -> None:
        genome = _seed().to_dict()
        execution = genome["agent_program"]["candidate_execution_program"]
        registry = current_program_registry()
        bad_ref = dict(execution["workflow_template_ref"])
        bad_ref["catalog_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest|reference"):
            compile_dsh_workflow_spec(
                bad_ref,
                execution["workflow_overrides"],
                execution["role_profiles"],
                registry,
            )

        expanded = json.loads(json.dumps(execution["role_profiles"]))
        expanded[0]["enabled_tool_ids"].append("host.shell")
        with self.assertRaisesRegex(ValueError, "tool|subset"):
            compile_dsh_workflow_spec(
                execution["workflow_template_ref"],
                execution["workflow_overrides"],
                expanded,
                registry,
            )

    def test_bind_fails_closed_on_data_stage_or_security_drift(self) -> None:
        task = _task()
        behavior = compile_plugin_behavior(
            _child(task, slot_index=0, slot_seed=10),
            task,
            None,
            current_program_registry(),
        )
        base = _instance_context(
            task, candidate_id="candidate:first", slot_index=0
        ).to_dict()
        for field in (
            "dataset_snapshot_set_digest",
            "stage_policy_digest",
            "security_kernel_digest",
        ):
            with self.subTest(field=field):
                changed = {**base, field: "0" * 64}
                with self.assertRaisesRegex(ValueError, "mismatch|binding"):
                    bind_phenotype_instance(
                        behavior, CompilationInstanceContext.from_dict(changed)
                    )

    def test_legacy_adapter_and_new_compiler_equal_old_full_algorithm_ir_dict(
        self,
    ) -> None:
        fixture = json.loads(
            (_FIXTURES / "legacy_program_0_2_2.json").read_text(encoding="utf-8")
        )
        task = TaskManifest.from_dict(fixture["task"])
        proposal = Proposal.from_dict(fixture["proposal"])
        snapshot = KnowledgeSnapshot.from_dict(fixture["knowledge_snapshot"])
        projected = legacy_genome_from_proposal(proposal, task, snapshot)

        self.assertEqual(
            compile_legacy_algorithm_ir(projected), fixture["legacy_algorithm_ir"]
        )


if __name__ == "__main__":
    unittest.main()
