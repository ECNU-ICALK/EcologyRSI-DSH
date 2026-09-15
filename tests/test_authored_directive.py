"""The `authored_directive@1` axis: a planner writes its strategy, not picks one.

Covers the three things that make this axis different from the axes around it:
the grammar is a whitelist rather than free text, the authored value lives in the
genome (so it is part of candidate identity), and leaving it absent has to keep
every archived genome byte-identical -- the same guarantee `feature_recipe` gets
from the `optional=` projection it is modelled on.
"""

from __future__ import annotations

import unittest

from ecologyrsi_dsh.evaluators.authored_directive import (
    ALLOWED_DIRECTIVE_ANCHORS,
    AUTHORED_DIRECTIVE_POLICY_ID,
    AUTHORED_DIRECTIVE_SCHEMA_VERSION,
    BLEND_RULE_METHODS,
    DIRECTIVE_TOP_LEVEL_KEYS,
    MAX_DIRECTIVE_RATIONALE_LENGTH,
    MAX_DIRECTIVE_TOOL_PLAN_STEPS,
    directive_grammar,
    render_authored_directive,
    validate_authored_directive,
)
from ecologyrsi_dsh.evolution.genome import (
    EcologyEvolutionPluginGenome,
    FrozenRunInitialization,
    GenomeMutationContextV1,
    apply_genome_mutation,
    materialize_seed_genome,
)
from ecologyrsi_dsh.evolution.local_edits import _operation_target
from ecologyrsi_dsh.evolution.workflow_ir import resolve_candidate_agent_profile
from ecologyrsi_dsh.knowledge.autonomous_cycle import (
    CANDIDATE_MUTATION_AXES,
    MUTATION_OPERATION_BY_AXIS,
)
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry


PLANNER_TOOL_ID = "ecology_execute_prediction_tool"


def _directive(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "anchor": "persistence",
        "blend_rule": "none",
        "tool_plan": [
            {"tool_id": PLANNER_TOOL_ID, "purpose": "candidate_model_baseline"}
        ],
        "rationale": "Persistence beat the model last batch; test it as the anchor.",
    }
    body.update(changes)
    return body


def _initialization() -> FrozenRunInitialization:
    return FrozenRunInitialization(
        run_id="run:authored-directive",
        task_manifest_digest="1" * 64,
        dataset_snapshot_set_digest="2" * 64,
        split_manifest_digest="3" * 64,
        data_protocol_digest="4" * 64,
        stage_policy_digest="5" * 64,
        evaluator_digest="6" * 64,
        fitness_profile_digest="7" * 64,
        security_kernel_digest="8" * 64,
        selection_reviewer_program_digest="9" * 64,
        protocol="dsh_native_plugin_evolution@1",
        required_capability_digest="a" * 64,
        resolved_policy_route_digest="b" * 64,
        resolved_review_route_digest="c" * 64,
        registry_catalog_digest=current_program_registry().catalog_digest,
        compiler_digest="d" * 64,
    )


def _seed_genome() -> EcologyEvolutionPluginGenome:
    registry = current_program_registry()
    return materialize_seed_genome(
        registry.seed_template("greenhouse-default@1"), _initialization()
    )


def _mutation_context(
    parent: EcologyEvolutionPluginGenome, *, slot_index: int = 0
) -> GenomeMutationContextV1:
    return GenomeMutationContextV1(
        run_id="run:authored-directive",
        generation=0,
        slot_index=slot_index,
        slot_seed=42,
        parent_candidate_id=None,
        parent_genome_digest=parent.genome_digest,
        generation_batch_digest="e" * 64,
        research_iteration_digest="f" * 64,
        knowledge_snapshot_digest="0" * 64,
        mutation_budget_digest="1" * 64,
        mutation_operator_id="bounded-single-parent-mutation@1",
    )


def _author(
    parent: EcologyEvolutionPluginGenome,
    directive: dict[str, object],
    *,
    role: str = "sample-planner",
    slot_index: int = 0,
) -> EcologyEvolutionPluginGenome:
    return apply_genome_mutation(
        parent,
        {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [
                {
                    "op": "author_role_directive",
                    "role": role,
                    "authored_directive": directive,
                }
            ],
        },
        _mutation_context(parent, slot_index=slot_index),
        current_program_registry(),
    )


class AuthoredDirectiveGrammarTests(unittest.TestCase):
    def test_a_legal_directive_normalizes_and_is_content_addressed(self) -> None:
        frozen = validate_authored_directive(_directive())

        self.assertEqual(frozen.anchor, "persistence")
        self.assertEqual(frozen.allowed_methods, BLEND_RULE_METHODS["none"])
        self.assertEqual(frozen.tool_ids, (PLANNER_TOOL_ID,))
        self.assertEqual(
            frozen.to_dict()["schema_version"], AUTHORED_DIRECTIVE_SCHEMA_VERSION
        )
        # Whitespace is collapsed before the bound is applied, so two directives
        # that differ only in line wrapping share one digest and the length
        # ceiling cannot be widened with padding.
        padded = validate_authored_directive(
            _directive(rationale="  ".join(str(frozen.rationale).split()))
        )
        self.assertEqual(padded.digest, frozen.digest)

    def test_out_of_grammar_clauses_are_refused(self) -> None:
        cases = (
            ("anchor", _directive(anchor="whatever_feels_right"), "anchor"),
            ("blend_rule", _directive(blend_rule="kalman"), "blend_rule"),
            (
                "unknown_clause",
                {**_directive(), "escalation": {"min_confidence": 0.1}},
                "unsupported fields",
            ),
            (
                "empty_plan",
                _directive(tool_plan=[]),
                f"1..{MAX_DIRECTIVE_TOOL_PLAN_STEPS} steps",
            ),
            (
                "plan_over_budget",
                _directive(
                    tool_plan=[
                        {"tool_id": f"tool_{index}", "purpose": "sensitivity_probe"}
                        for index in range(MAX_DIRECTIVE_TOOL_PLAN_STEPS + 1)
                    ]
                ),
                f"1..{MAX_DIRECTIVE_TOOL_PLAN_STEPS} steps",
            ),
            (
                "duplicate_step",
                _directive(
                    tool_plan=[
                        {"tool_id": PLANNER_TOOL_ID, "purpose": "discrepancy_check"},
                        {"tool_id": PLANNER_TOOL_ID, "purpose": "discrepancy_check"},
                    ]
                ),
                "duplicates an earlier step",
            ),
            (
                "unknown_purpose",
                _directive(
                    tool_plan=[{"tool_id": PLANNER_TOOL_ID, "purpose": "vibes"}]
                ),
                "purpose",
            ),
            (
                "rationale_too_long",
                _directive(rationale="x" * (MAX_DIRECTIVE_RATIONALE_LENGTH + 1)),
                f"at most {MAX_DIRECTIVE_RATIONALE_LENGTH}",
            ),
            ("empty_rationale", _directive(rationale="   "), "non-empty text"),
            (
                "wrong_schema",
                _directive(schema_version="ecologyrsi-dsh.authored-directive/9"),
                "schema version",
            ),
        )
        for name, body, message in cases:
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                validate_authored_directive(body)

    def test_a_plan_cannot_name_a_tool_the_policy_withheld(self) -> None:
        # The same directive is legal or not depending on the role's grant, which
        # is why the tool check belongs to the binding-aware layer and not to the
        # static grammar.
        body = _directive(
            tool_plan=[{"tool_id": "some_other_tool", "purpose": "sensitivity_probe"}]
        )
        validate_authored_directive(body)
        with self.assertRaisesRegex(ValueError, "plans unavailable tools"):
            validate_authored_directive(body, allowed_tool_ids=[PLANNER_TOOL_ID])

    def test_rendering_is_deterministic_and_states_the_enforced_threshold(
        self,
    ) -> None:
        frozen = validate_authored_directive(_directive())
        text = render_authored_directive(frozen, escalation_threshold=0.5)

        self.assertEqual(text, render_authored_directive(frozen.to_dict(), escalation_threshold=0.5))
        self.assertIn("persistence", text)
        self.assertIn("0.50", text)
        self.assertIn(str(frozen.rationale), text)
        # Every anchor and rule has prose; a clause the renderer cannot speak
        # would reach the planner as a KeyError at compile time.
        for anchor in sorted(ALLOWED_DIRECTIVE_ANCHORS):
            for rule in sorted(BLEND_RULE_METHODS):
                self.assertTrue(
                    render_authored_directive(
                        validate_authored_directive(
                            _directive(anchor=anchor, blend_rule=rule)
                        )
                    )
                )

    def test_the_published_grammar_matches_what_the_validator_enforces(self) -> None:
        grammar = directive_grammar()

        self.assertEqual(grammar["anchor"]["choices"], sorted(ALLOWED_DIRECTIVE_ANCHORS))
        self.assertEqual(grammar["blend_rule"]["choices"], sorted(BLEND_RULE_METHODS))
        self.assertEqual(
            grammar["tool_plan"]["maximum_steps"], MAX_DIRECTIVE_TOOL_PLAN_STEPS
        )
        self.assertEqual(
            grammar["rationale"]["maximum_length"], MAX_DIRECTIVE_RATIONALE_LENGTH
        )
        # Every clause the validator accepts is published, and nothing else --
        # a clause missing here is one the editor has no way to discover.
        self.assertEqual(
            set(grammar["clauses"]), DIRECTIVE_TOP_LEVEL_KEYS - {"schema_version"}
        )
        # Zero-argument and stable, because the registry hashes it into
        # catalog_digest: the grammar an Agent was shown must be recoverable
        # from the genome's digest alone.
        self.assertEqual(grammar, directive_grammar())

    def test_the_registry_publishes_the_policy_with_the_same_bounds(self) -> None:
        policy = current_program_registry().program(
            "directive_policies", AUTHORED_DIRECTIVE_POLICY_ID
        )

        self.assertEqual(
            policy["parameters"]["max_tool_plan_steps"]["maximum"],
            MAX_DIRECTIVE_TOOL_PLAN_STEPS,
        )
        self.assertEqual(
            policy["parameters"]["max_rationale_length"]["maximum"],
            MAX_DIRECTIVE_RATIONALE_LENGTH,
        )
        self.assertEqual(policy["grammar"], directive_grammar())


class AuthoredDirectiveGenomeTests(unittest.TestCase):
    def test_authoring_changes_behavior_and_reaches_the_delivered_directive(
        self,
    ) -> None:
        registry = current_program_registry()
        parent = _seed_genome()
        child = _author(parent, _directive())

        profile = next(
            item
            for item in child.agent_program["candidate_execution_program"][
                "role_profiles"
            ]
            if item["role"] == "sample-planner"
        )
        self.assertEqual(
            profile["directive_policy_ref"]["id"], AUTHORED_DIRECTIVE_POLICY_ID
        )
        self.assertNotEqual(child.behavior_digest, parent.behavior_digest)

        resolved = resolve_candidate_agent_profile(child, registry)
        self.assertEqual(resolved["directive_policy_id"], AUTHORED_DIRECTIVE_POLICY_ID)
        # The rendered text lands in the same slot a selected template fills, so
        # nothing on the delivery path has to know this axis exists.
        self.assertIn("Anchor every cell on persistence", resolved["instruction_directive"])
        self.assertEqual(
            resolved["allowed_prediction_methods"], list(BLEND_RULE_METHODS["none"])
        )
        self.assertEqual(
            resolved["authored_directive_digest"],
            validate_authored_directive(_directive()).digest,
        )
        # Round-trips: from_dict re-derives behavior_digest, so a directive that
        # did not survive normalization would fail here rather than silently.
        self.assertEqual(
            EcologyEvolutionPluginGenome.from_dict(child.to_dict()).genome_digest,
            child.genome_digest,
        )

    def test_an_absent_directive_leaves_the_archived_projection_untouched(self) -> None:
        # The compatibility guarantee of the whole step. A candidate that never
        # authored one keeps the historical role_profile key set, so every
        # archived genome_digest, behavior_digest and genome_id stays replayable.
        parent = _seed_genome()
        profile = next(
            item
            for item in parent.agent_program["candidate_execution_program"][
                "role_profiles"
            ]
            if item["role"] == "sample-planner"
        )
        self.assertNotIn("authored_directive", profile)
        self.assertNotIn("directive_policy_ref", profile)

        resolved = resolve_candidate_agent_profile(parent, current_program_registry())
        for key in (
            "authored_directive",
            "authored_directive_digest",
            "directive_policy_id",
            "allowed_prediction_methods",
        ):
            self.assertNotIn(key, resolved)

    def test_re_authoring_the_identical_directive_is_refused(self) -> None:
        # Mirrors the "does not change" guard the other operators carry: without
        # it a local edit could burn its one operation and still be accepted,
        # which is exactly what the behavior_digest guardrail cannot catch.
        child = _author(_seed_genome(), _directive())
        with self.assertRaisesRegex(ValueError, "does not change"):
            _author(child, _directive())

        revised = _author(child, _directive(blend_rule="mean"), slot_index=1)
        self.assertNotEqual(revised.behavior_digest, child.behavior_digest)
        self.assertEqual(
            resolve_candidate_agent_profile(revised, current_program_registry())[
                "allowed_prediction_methods"
            ],
            list(BLEND_RULE_METHODS["mean"]),
        )

    def test_only_the_sample_planner_may_author_a_directive(self) -> None:
        # Named roles rather than "every other profile in the seed": the seed
        # carries one role today, so iterating it would assert nothing. The
        # restriction exists because rendering assumes the planner's prediction
        # contract -- `blend_rule` narrows prediction methods, which no other
        # role submits.
        parent = _seed_genome()
        for role in ("sample-critic", "candidate-proposer"):
            with self.subTest(role=role), self.assertRaisesRegex(
                ValueError, "only the sample-planner role can author a directive"
            ):
                _author(parent, _directive(), role=role)

    def test_the_axis_is_registered_everywhere_an_operator_has_to_be(self) -> None:
        # Three registration sites, all of which fail differently when missed:
        # the axis map (KeyError in the proposer), the operator chain (the
        # mutation is refused), and the local-edit target resolver.
        self.assertIn("instruction_directive", CANDIDATE_MUTATION_AXES)
        self.assertEqual(
            MUTATION_OPERATION_BY_AXIS["instruction_directive"],
            "author_role_directive",
        )
        self.assertEqual(
            _operation_target(
                {"op": "author_role_directive", "role": "sample-planner"}
            ),
            (
                "instruction_directive",
                AUTHORED_DIRECTIVE_POLICY_ID,
                "instruction-directive:sample-planner",
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
