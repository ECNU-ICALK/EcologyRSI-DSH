"""Every advertised mutation axis must reach something that reads it.

The axes in this file were all present before: the genome accepted the edit, the
digest moved, the guardrail passed -- and nothing at run time differed, because
no host component read the field. `behavior_digest` covers the whole agent
program, so it cannot tell a real behaviour change from a decorative one; only a
test that follows the value to its consumer can. That is what these are.
"""
from __future__ import annotations

import json
import unittest
from dataclasses import replace

from ecologyrsi_dsh.core.agent_prediction import (
    PREDICTION_METHODS,
    validate_predictions,
)
from ecologyrsi_dsh.core.models import Proposal, TaskManifest
from ecologyrsi_dsh.evaluators.authored_directive import (
    ALLOWED_DIRECTIVE_ANCHORS,
    AUTHORED_DIRECTIVE_POLICY_ID,
    MAX_DIRECTIVE_TOOL_PLAN_STEPS,
)
from ecologyrsi_dsh.evaluators.dsh_sample_adapter import DshSampleCollaborationAdapter
from ecologyrsi_dsh.evaluators.registry import (
    TOY_DATASET_ID,
    EvaluatorRegistry,
    _candidate_remote_critic_policy,
)
from ecologyrsi_dsh.evaluators.sample_contracts import _normalized_remote_critic_policy
from ecologyrsi_dsh.evolution.agent_policy import (
    allowed_prediction_methods as _allowed_prediction_methods,
)
from ecologyrsi_dsh.evolution.execution_plan import DerivedExecutionPlan
from ecologyrsi_dsh.evolution.strategies import (
    MUTATION_AXIS_EFFECTS,
    MUTATION_DIRECTIONS_BY_AXIS,
    MUTATION_OPERATION_BY_AXIS,
    mutation_axis_contract,
)
from ecologyrsi_dsh.evolution.workflow_ir import (
    EXECUTABLE_WORKFLOW_PARAMETERS,
    resolve_candidate_agent_profile,
)
from ecologyrsi_dsh.evolution.genome import (
    CANDIDATE_WORKFLOW_TEMPLATE_ID,
    EcologyEvolutionPluginGenome,
    apply_genome_mutation,
    materialize_seed_genome,
)
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_evolution_genome import _initialization, _mutation_context


def _task(**metadata) -> TaskManifest:
    return TaskManifest(
        task_id="axis-wiring",
        objective="axis wiring",
        domain_pack="toy_water@1",
        visible_datasets=(TOY_DATASET_ID,),
        budget={"max_candidates": 1},
        metadata=metadata,
    )


# The run-level policy the greenhouse runs with today
# (`_STRICT_SAMPLE_REMOTE_CRITIC_POLICY`).
_STRICT_POLICY = {"version": "uncertain_or_failure@1", "min_planner_confidence": 0.5}


def _adapter(policy) -> DshSampleCollaborationAdapter:
    """A collaboration adapter built only far enough to answer one question.

    None of the providers are ever called: the escalation predicate reads the
    frozen policy and the decision, and nothing else.
    """

    def _unavailable(*_args, **_kwargs):
        raise AssertionError("axis wiring test must not reach the runtime")

    return DshSampleCollaborationAdapter(
        run_id="run:axis-wiring",
        runtime_provider=_unavailable,
        revision_provider=_unavailable,
        identity_digests={
            "genome_digest": "a" * 64,
            "compiled_behavior_digest": "b" * 64,
            "phenotype_instance_digest": "c" * 64,
        },
        strategy_model_id="dsh/strategy",
        review_model_id="dsh/review",
        forecast_bundle_tool=_unavailable,
        prediction_tool_binder=_unavailable,
        remote_critic_policy=policy,
    )


def _proposal(profile: dict | None) -> Proposal:
    metadata = {} if profile is None else {"candidate_agent_profile": profile}
    return Proposal(
        proposal_id="proposal:axis-wiring",
        run_id="run:axis-wiring",
        generation=0,
        title="axis wiring",
        changes={},
        rationale="axis wiring",
        metadata=metadata,
    )


def _seed_genome() -> EcologyEvolutionPluginGenome:
    return materialize_seed_genome(
        current_program_registry().seed_template(
            "greenhouse-baseline-aligned-default@1"
        ),
        _initialization(),
    )


class WorkflowParameterAxisTests(unittest.TestCase):
    def test_the_candidates_requested_retry_depth_reaches_the_sample_policy(self):
        """The genome value has to change execution, not just the digest."""

        registry = current_program_registry()
        genome = _seed_genome()
        deeper = apply_genome_mutation(
            genome,
            {"schema_version": "ecologyrsi-dsh.genome-mutation/1",
             "operations": [{"op": "set_bounded_workflow_parameter",
                             "name": "max_attempts", "value": 7}]},
            _mutation_context(genome),
            registry,
        )
        seed_profile = resolve_candidate_agent_profile(genome, registry)
        deeper_profile = resolve_candidate_agent_profile(deeper, registry)
        self.assertEqual(seed_profile["workflow_parameters"]["max_attempts"], 3)
        self.assertEqual(deeper_profile["workflow_parameters"]["max_attempts"], 7)

        policy_for = EvaluatorRegistry._sample_execution_policy
        plan = DerivedExecutionPlan(source_generation=None, source_analysis_digest=None)
        self.assertEqual(plan.sample_max_attempts, 3)
        self.assertEqual(
            policy_for(_task(), plan, _proposal(deeper_profile)).max_attempts, 7
        )
        self.assertEqual(
            policy_for(_task(), plan, _proposal(seed_profile)).max_attempts, 3
        )

    def test_the_derived_floor_only_ever_raises_the_requested_depth(self):
        """Floor and request are different claims, so the synthesis is a max().

        The derived value is the host's reliability floor -- it exists because
        the previous generation saw transient failures -- and the candidate's is
        this generation's requested depth. Letting the floor lower a request
        would be an unattributable score change; letting a request lower the
        floor would trade the host's recovery margin for the candidate's guess.
        """

        registry = current_program_registry()
        profile = resolve_candidate_agent_profile(_seed_genome(), registry)
        policy_for = EvaluatorRegistry._sample_execution_policy
        raised_floor = DerivedExecutionPlan(
            source_generation=1,
            source_analysis_digest="a" * 64,
            sample_max_attempts=6,
        )
        # Request 3 (the seed) under a floor of 6: the floor wins.
        self.assertEqual(
            policy_for(_task(), raised_floor, _proposal(profile)).max_attempts, 6
        )
        # Request 8 under the same floor: the request wins.
        deep = {**profile, "workflow_parameters": {
            **profile["workflow_parameters"], "max_attempts": 8}}
        self.assertEqual(
            policy_for(_task(), raised_floor, _proposal(deep)).max_attempts, 8
        )

    def test_a_profile_archived_before_the_field_existed_keeps_its_scored_depth(self):
        """Rescoring must not import today's defaults into yesterday's score."""

        registry = current_program_registry()
        profile = resolve_candidate_agent_profile(_seed_genome(), registry)
        archived = {
            key: value
            for key, value in profile.items()
            if key != "workflow_parameters"
        }
        plan = DerivedExecutionPlan(
            source_generation=1,
            source_analysis_digest="a" * 64,
            sample_max_attempts=5,
        )
        policy_for = EvaluatorRegistry._sample_execution_policy
        self.assertEqual(policy_for(_task(), plan, _proposal(archived)).max_attempts, 5)
        self.assertEqual(policy_for(_task(), plan, _proposal(None)).max_attempts, 5)
        self.assertEqual(policy_for(_task(), plan).max_attempts, 5)

    def test_no_legal_requested_depth_is_below_the_hosts_own_floor(self):
        """A value the floor always overrules is an unobservable mutation.

        The derived floor is at least ``min(8, 3 + 0)``, so a contract that
        allowed 1 or 2 advertised two values whose effect could never appear in
        any evaluation record.
        """

        contract = current_program_registry().program(
            "workflow_templates", "candidate-sample-execution@1"
        )["parameters"]["max_attempts"]
        floor = DerivedExecutionPlan(
            source_generation=None, source_analysis_digest=None
        ).sample_max_attempts
        self.assertEqual(contract["minimum"], floor)
        self.assertEqual(contract["default"], floor)

    def test_only_workflow_parameters_with_a_consumer_are_advertised(self):
        """`max_concurrent` and `wave_size` are declared but read by nobody."""

        declared = set(
            current_program_registry().program(
                "workflow_templates", "candidate-sample-execution@1"
            )["parameters"]
        )
        self.assertEqual(declared, {"max_concurrent", "wave_size", "max_attempts"})
        self.assertEqual(EXECUTABLE_WORKFLOW_PARAMETERS, {"max_attempts"})
        self.assertTrue(EXECUTABLE_WORKFLOW_PARAMETERS <= declared)


class InstructionParameterAxisTests(unittest.TestCase):
    """`confidence_threshold` has to change how much review the run buys."""

    def _threshold_of(self, genome) -> float:
        profile = resolve_candidate_agent_profile(genome, current_program_registry())
        policy = _candidate_remote_critic_policy(
            _task(sample_remote_critic_policy=_STRICT_POLICY), _proposal(profile)
        )
        return _normalized_remote_critic_policy(policy)["min_planner_confidence"]

    def test_the_candidates_threshold_reaches_the_critic_escalation_rate(self):
        registry = current_program_registry()
        genome = _seed_genome()
        eager = apply_genome_mutation(
            genome,
            {"schema_version": "ecologyrsi-dsh.genome-mutation/1",
             "operations": [{"op": "set_instruction_parameter",
                             "role": "sample-planner",
                             "name": "confidence_threshold", "value": 0.75}]},
            _mutation_context(genome),
            registry,
        )
        self.assertEqual(self._threshold_of(genome), 0.5)
        self.assertEqual(self._threshold_of(eager), 0.75)

        # The threshold is only worth anything if it moves the escalation set.
        # Production planner confidences cluster in 0.70-0.85, so these are the
        # decisions the two thresholds actually have to disagree about.
        confidences = (0.70, 0.72, 0.78, 0.85)
        seed_adapter = _adapter({**_STRICT_POLICY, "min_planner_confidence": 0.5})
        eager_adapter = _adapter({**_STRICT_POLICY, "min_planner_confidence": 0.75})
        self.assertEqual(
            [seed_adapter._requires_remote_review({"confidence": value})
             for value in confidences],
            [False, False, False, False],
        )
        self.assertEqual(
            [eager_adapter._requires_remote_review({"confidence": value})
             for value in confidences],
            [True, True, False, False],
        )

    def test_generation_zero_costs_what_the_run_already_costs(self):
        """Connecting a knob must not be a silent price increase.

        The seed's threshold has to equal the run-level value the host used while
        the parameter had no consumer, or generation 0 would escalate at a
        different rate than the baseline every candidate is compared against.
        """

        self.assertEqual(
            self._threshold_of(_seed_genome()),
            _STRICT_POLICY["min_planner_confidence"],
        )

    def test_a_candidate_cannot_switch_independent_review_off(self):
        """The two policy-shaped escapes stay the host's, not the candidate's."""

        profile = resolve_candidate_agent_profile(
            _seed_genome(), current_program_registry()
        )
        # No run-level policy means "review everything" downstream. Synthesising
        # one from the candidate's threshold would hand it an opt-out.
        self.assertIsNone(_candidate_remote_critic_policy(_task(), _proposal(profile)))
        self.assertTrue(_adapter(None)._requires_remote_review({"confidence": 0.99}))
        # `always@1` carries no threshold at all -- `_normalized_remote_critic_policy`
        # rejects the key -- so there is nothing to overlay.
        always = {"version": "always@1"}
        self.assertEqual(
            _candidate_remote_critic_policy(
                _task(sample_remote_critic_policy=always), _proposal(profile)
            ),
            always,
        )
        self.assertTrue(_adapter(always)._requires_remote_review({"confidence": 0.99}))

    def test_a_profile_archived_before_the_field_was_read_keeps_its_threshold(self):
        task = _task(sample_remote_critic_policy=_STRICT_POLICY)
        profile = resolve_candidate_agent_profile(
            _seed_genome(), current_program_registry()
        )
        archived = {
            key: value
            for key, value in profile.items()
            if key != "instruction_parameters"
        }
        for proposal in (_proposal(archived), _proposal(None), None):
            with self.subTest(proposal=proposal):
                self.assertEqual(
                    _candidate_remote_critic_policy(task, proposal), _STRICT_POLICY
                )

    def test_the_overlay_keeps_the_policy_contract_shape(self):
        """The 2-key contract is validated again downstream; do not break it."""

        overlaid = _candidate_remote_critic_policy(
            _task(sample_remote_critic_policy=_STRICT_POLICY),
            _proposal(
                {"instruction_parameters": {"confidence_threshold": 0.625}}
            ),
        )
        self.assertEqual(set(overlaid), {"version", "min_planner_confidence"})
        self.assertEqual(
            _normalized_remote_critic_policy(overlaid),
            {"version": "uncertain_or_failure@1", "min_planner_confidence": 0.625},
        )

    def test_every_planner_template_advertises_the_same_cost_window(self):
        """A template switch must not move cost, or nothing is attributable.

        `select_instruction_template` clears the candidate's overrides, so each
        template's default becomes the live threshold the moment it is selected.
        Different defaults would make the `instruction_profile` axis change both
        strategy and spend at once.
        """

        registry = current_program_registry()
        contracts = {}
        for program_id in registry.program_ids("instruction_templates"):
            parameters = registry.program("instruction_templates", program_id)[
                "parameters"
            ]
            if "confidence_threshold" in parameters:
                contracts[program_id] = parameters["confidence_threshold"]
        self.assertEqual(len(contracts), 7)
        self.assertEqual(
            {tuple(sorted(contract.items())) for contract in contracts.values()},
            {tuple(sorted({"minimum": 0.5, "maximum": 0.75, "default": 0.5,
                           "integer": False}.items()))},
        )

    def test_the_advertised_window_is_the_cost_window_not_the_arithmetic_one(self):
        """[0, 1] is legal for the policy; it is not legal for the budget.

        A threshold near 0.9 escalates nearly every cell at the confidences the
        planner actually reports, which is a cost cliff rather than a strategy.
        """

        contract = current_program_registry().program(
            "instruction_templates", "sample-planner-balanced@1"
        )["parameters"]["confidence_threshold"]
        self.assertLess(contract["maximum"], 0.9)
        for value in (0.4, 0.9):
            with self.assertRaises(ValueError):
                apply_genome_mutation(
                    _seed_genome(),
                    {"schema_version": "ecologyrsi-dsh.genome-mutation/1",
                     "operations": [{"op": "set_instruction_parameter",
                                     "role": "sample-planner",
                                     "name": "confidence_threshold",
                                     "value": value}]},
                    _mutation_context(_seed_genome()),
                    current_program_registry(),
                )


class ClosedAxisTests(unittest.TestCase):
    """An axis with no usable edit must be closed, not advertised and broken."""

    def _local_edit_targets(self) -> dict:
        from types import SimpleNamespace
        from unittest.mock import patch

        from ecologyrsi_dsh.application.formal_trajectory import _local_edit_context
        from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
        from tests.test_local_edits import _parent

        parent = _parent()
        revision = SimpleNamespace(
            revision_id="revision:axis-wiring:r0",
            genome=parent.to_dict(),
            genome_digest=parent.genome_digest,
        )
        state = SimpleNamespace(
            run=SimpleNamespace(run_id="run:axis-wiring"),
            task_manifest=SimpleNamespace(
                metadata={
                    "optimization_schedule": OptimizationSchedule.default().to_dict(),
                    "fitness_profile": {
                        "expected_targets": [
                            "air_temperature",
                            "relative_humidity",
                            "co2_concentration",
                        ],
                        "expected_horizons": [1, 6, 24],
                    },
                }
            ),
            batch_evaluation_for=lambda *_args: SimpleNamespace(
                scope=SimpleNamespace(scope_key="a" * 64)
            ),
        )
        # `_registered_mutation_targets` needs a whole run's task manifest and
        # contributes only the two scientific axes. The axes under test here are
        # the ones `_local_edit_context` adds on top of it, so it is stubbed for
        # the same reason `test_formal_trajectory` stubs it.
        with (
            patch(
                "ecologyrsi_dsh.application.formal_trajectory._registered_mutation_targets",
                return_value={
                    "scientific_parameter": ("ridge_alpha",),
                    "registered_predictor": (
                        "greenhouse-horizon-targetwise-ridge@1",
                    ),
                },
            ),
            patch(
                "ecologyrsi_dsh.application.formal_trajectory._genome_parameter_boundary",
                return_value=("test", {}),
            ),
        ):
            context = _local_edit_context(
                state,
                SimpleNamespace(candidate_id="candidate:axis-wiring", generation=0),
                revision,
                SimpleNamespace(batch_index=0),
            )
        return dict(context.allowed_mutation_targets)

    def test_the_tool_policy_axis_offers_nothing_while_narrowing_can_only_harm(self):
        """The one legal narrowing switches the prediction tool off entirely.

        `sample-planner-tools@1` holds a single tool and the operator only ever
        narrows, so the axis could express exactly one edit: remove the path by
        which the scientific parameters reach a prediction. Offering it spent the
        batch's single operation on a self-inflicted regression.
        """

        targets = self._local_edit_targets()
        self.assertIn("instruction_tool_policy", targets)
        self.assertEqual(tuple(targets["instruction_tool_policy"]), ())
        policy = current_program_registry().tool_policy("sample-planner-tools@1")
        self.assertEqual(len(policy), 1)

    def test_the_axes_that_are_open_advertise_exactly_their_live_knobs(self):
        targets = self._local_edit_targets()
        self.assertEqual(tuple(targets["workflow_parameter"]), ("max_attempts",))
        self.assertEqual(
            tuple(targets["instruction_parameter"]), ("confidence_threshold",)
        )
        # The authoring axis is targeted by the policy that bounds the text, not
        # by the text, so its one target is the registered grammar itself.
        self.assertEqual(
            tuple(targets["instruction_directive"]), (AUTHORED_DIRECTIVE_POLICY_ID,)
        )

    def test_an_uncompilable_workflow_class_is_rejected_at_mutation_time(self):
        """Rejecting one proposal, rather than failing the run a batch later.

        `compile_dsh_workflow_spec` pins the candidate workflow class. A genome
        that cites another registered template passes `apply_genome_mutation` and
        `from_dict` and then raises from `_revision_evaluation_inputs` on the next
        batch, which is a run failure rather than a rejected edit.
        """

        registry = current_program_registry()
        other = [
            program_id
            for program_id in registry.program_ids("workflow_templates")
            if program_id != CANDIDATE_WORKFLOW_TEMPLATE_ID
        ]
        self.assertTrue(other, "fixture needs a second registered workflow template")
        genome = _seed_genome()
        with self.assertRaisesRegex(ValueError, "registered workflow class"):
            apply_genome_mutation(
                genome,
                {"schema_version": "ecologyrsi-dsh.genome-mutation/1",
                 "operations": [{"op": "select_registered_workflow_template",
                                 "workflow_template_id": other[0]}]},
                _mutation_context(genome),
                registry,
            )


class LegalBoundsPayloadTests(unittest.TestCase):
    """An advertised knob has to arrive with its range and its current value.

    A target name alone makes every proposal a guess, and a guess outside the
    registered contract costs the batch its one operation. These two payloads are
    the only place the local editor learns either fact.
    """

    def _payloads(self) -> tuple[dict, dict]:
        from types import SimpleNamespace

        from ecologyrsi_dsh.application.formal_trajectory import (
            _legal_instruction_parameter_bounds,
            _legal_workflow_parameter_bounds,
        )
        from tests.test_local_edits import _context, _parent

        parent = _parent()
        revision = SimpleNamespace(
            revision_id="revision:axis-wiring:bounds",
            genome=parent.to_dict(),
            genome_digest=parent.genome_digest,
        )
        context = replace(
            _context(parent),
            allowed_mutation_targets={
                **_context(parent).allowed_mutation_targets,
                "workflow_parameter": ("max_attempts",),
                "instruction_parameter": ("confidence_threshold",),
            },
        )
        return (
            _legal_workflow_parameter_bounds(revision, context),
            _legal_instruction_parameter_bounds(revision, context),
        )

    def test_the_workflow_knob_arrives_with_its_contract_and_live_value(self):
        workflow, _instruction = self._payloads()
        self.assertEqual(set(workflow), {"max_attempts"})
        bounds = workflow["max_attempts"]
        self.assertEqual(bounds["minimum"], 3)
        self.assertEqual(bounds["maximum"], 8)
        self.assertTrue(bounds["integer"])
        # The seed's own override, not the contract default -- this is the value a
        # `decrease` would have to move away from.
        self.assertEqual(bounds["current_value"], 3)

    def test_the_instruction_knob_arrives_keyed_by_the_role_that_owns_it(self):
        _workflow, instruction = self._payloads()
        self.assertEqual(set(instruction), {"sample-planner"})
        bounds = instruction["sample-planner"]["confidence_threshold"]
        self.assertEqual(bounds["minimum"], 0.5)
        self.assertEqual(bounds["maximum"], 0.75)
        self.assertEqual(bounds["current_value"], 0.5)

    def test_a_knob_that_is_not_offered_is_not_published(self):
        """Only the advertised targets, so the payload cannot widen the axis.

        `candidate-sample-execution@1` also declares `max_concurrent` and
        `wave_size`; publishing their bounds would read as an invitation to edit
        two parameters nothing consumes.
        """

        workflow, _instruction = self._payloads()
        self.assertNotIn("max_concurrent", workflow)
        self.assertNotIn("wave_size", workflow)

    def test_the_authoring_axis_arrives_with_its_grammar_and_its_tool_grant(self):
        """The same problem as the bounds above, for a clause-shaped payload.

        A number needs its range; a structure needs its clause whitelist. And the
        allowed tool ids are the *intersection* of the grammar and this
        candidate's own grant, so a directive that follows the published contract
        cannot be refused by the compiler for planning a withheld tool.
        """

        from ecologyrsi_dsh.application.formal_trajectory import (
            _legal_directive_grammar,
        )
        from types import SimpleNamespace

        from tests.test_local_edits import _parent

        parent = _parent()
        grammar = _legal_directive_grammar(
            SimpleNamespace(
                revision_id="revision:axis-wiring:grammar",
                genome=parent.to_dict(),
                genome_digest=parent.genome_digest,
            )
        )
        self.assertEqual(grammar["policy_id"], AUTHORED_DIRECTIVE_POLICY_ID)
        self.assertEqual(grammar["anchor"]["choices"], sorted(ALLOWED_DIRECTIVE_ANCHORS))
        self.assertEqual(
            grammar["tool_plan.allowed_tool_ids"],
            list(
                next(
                    item
                    for item in parent.agent_program["candidate_execution_program"][
                        "role_profiles"
                    ]
                    if item["role"] == "sample-planner"
                )["enabled_tool_ids"]
            ),
        )
        # `author` versus `revise` is decided by this flag, and the seed has
        # authored nothing yet.
        self.assertFalse(grammar["current_directive_present"])
        self.assertEqual(
            grammar["effective_parameters"]["max_tool_plan_steps"],
            MAX_DIRECTIVE_TOOL_PLAN_STEPS,
        )
        # JSON-safe: this payload is serialized into the editor's prompt.
        self.assertEqual(json.loads(json.dumps(grammar)), grammar)


class InstructionDirectiveAxisTests(unittest.TestCase):
    """An authored `blend_rule` has to narrow what the host accepts.

    This axis was added precisely because the axes above are scalars: it lets a
    candidate rewrite its own strategy text. But a directive whose clauses only
    reached the prompt would be the same decoration this file exists to prevent,
    so `blend_rule` is followed here from the genome to the contract shown to the
    Agent and on to the validator that judges its answer.
    """

    def _authored(self, blend_rule: str) -> EcologyEvolutionPluginGenome:
        genome = _seed_genome()
        profile = next(
            item
            for item in genome.agent_program["candidate_execution_program"][
                "role_profiles"
            ]
            if item["role"] == "sample-planner"
        )
        return apply_genome_mutation(
            genome,
            {"schema_version": "ecologyrsi-dsh.genome-mutation/1",
             "operations": [{"op": "author_role_directive",
                             "role": "sample-planner",
                             "authored_directive": {
                                 "anchor": "candidate_model",
                                 "blend_rule": blend_rule,
                                 "tool_plan": [{
                                     "tool_id": profile["enabled_tool_ids"][0],
                                     "purpose": "candidate_model_baseline"}],
                                 "rationale": (
                                     "Lead with the candidate's own model and "
                                     "check it against the observations.")}}]},
            _mutation_context(genome),
            current_program_registry(),
        )

    def test_the_authored_blend_rule_reaches_the_prediction_validator(self):
        registry = current_program_registry()
        strict = resolve_candidate_agent_profile(self._authored("none"), registry)
        wide = resolve_candidate_agent_profile(
            self._authored("confidence_weighted"), registry
        )
        self.assertEqual(strict["allowed_prediction_methods"], ["direct", "model"])
        self.assertEqual(_allowed_prediction_methods(strict), ("direct", "model"))
        self.assertEqual(
            _allowed_prediction_methods(wide),
            ("adjusted", "blend", "direct", "model"),
        )

        # The clause is a rejection, not a suggestion: a method the directive
        # withheld is refused even though it is a globally valid method.
        context = {"wave_digest": "d" * 64, "samples": [{"sample_id": "one"}]}
        answer = {
            "schema_version": "ecology-sample-predictions@2",
            "wave_digest": context["wave_digest"],
            "decisions": [{"sample_id": "one", "predicted": 21.5, "confidence": 0.8,
                           "reason_code": "agent_blend", "method": "blend",
                           "evidence_call_ids": ["one", "two"]}],
        }
        validate_predictions(
            answer, ["one"], wave_digest=context["wave_digest"],
            allowed_methods=_allowed_prediction_methods(wide),
        )
        with self.assertRaisesRegex(ValueError, "outside the candidate's directive"):
            validate_predictions(
                answer, ["one"], wave_digest=context["wave_digest"],
                allowed_methods=_allowed_prediction_methods(strict),
            )

    def test_the_rendered_directive_replaces_the_selected_templates_sentence(self):
        registry = current_program_registry()
        seed_profile = resolve_candidate_agent_profile(_seed_genome(), registry)
        authored = resolve_candidate_agent_profile(self._authored("mean"), registry)

        self.assertNotEqual(
            authored["instruction_directive"], seed_profile["instruction_directive"]
        )
        self.assertIn("Anchor every cell on this candidate", authored["instruction_directive"])
        # Delivery is unchanged: same slot, same skill, same template identity,
        # so nothing on the JavaScript side has to know this axis exists.
        self.assertEqual(authored["skill_name"], seed_profile["skill_name"])
        self.assertEqual(
            authored["instruction_template_id"], seed_profile["instruction_template_id"]
        )
        # The escalation number stated to the planner is the one the host will
        # enforce, so the prose and the policy cannot describe two rules.
        threshold = authored["instruction_parameters"]["confidence_threshold"]
        self.assertIn(f"{threshold:.2f}", authored["instruction_directive"])

    def test_a_profile_archived_before_authoring_existed_keeps_every_method(self):
        """The compatibility guarantee, checked at the consumer.

        An archived profile carries no `allowed_prediction_methods` key, so the
        reader has to fall back to the full set -- otherwise rescoring would
        judge yesterday's answers against a rule that did not exist.
        """

        archived = resolve_candidate_agent_profile(
            _seed_genome(), current_program_registry()
        )
        self.assertNotIn("allowed_prediction_methods", archived)
        for profile in (archived, {}, None, {"allowed_prediction_methods": []}):
            with self.subTest(profile=profile):
                self.assertEqual(
                    _allowed_prediction_methods(profile),
                    tuple(sorted(PREDICTION_METHODS)),
                )


class MutationAxisContractTests(unittest.TestCase):
    def test_every_advertised_axis_comes_with_the_operation_that_writes_it(self):
        """An axis name without its operation id is not a usable offer.

        The local editor was shown six axes and the operation names for three,
        because only those three are in the shared Skill. The other three could
        be reached only by guessing an operation id, and a wrong guess costs the
        batch its single operation.
        """

        axes = tuple(MUTATION_OPERATION_BY_AXIS)
        contract = mutation_axis_contract(axes)
        for key in ("operation_by_axis", "mutation_directions_by_axis",
                    "mutation_axis_effects"):
            self.assertEqual(tuple(contract[key]), axes, key)
        self.assertEqual(set(MUTATION_DIRECTIONS_BY_AXIS), set(axes))
        self.assertEqual(set(MUTATION_AXIS_EFFECTS), set(axes))
        self.assertTrue(all(contract["mutation_axis_effects"].values()))

    def test_an_axis_with_no_registered_operation_is_refused_not_omitted(self):
        with self.assertRaises(ValueError):
            mutation_axis_contract(("scientific_parameter", "not_an_axis"))

    def test_the_projection_is_ordered_by_the_axes_the_caller_advertises(self):
        contract = mutation_axis_contract(("instruction_profile", "workflow_parameter"))
        self.assertEqual(
            contract["operation_by_axis"],
            {"instruction_profile": "select_instruction_template",
             "workflow_parameter": "set_bounded_workflow_parameter"},
        )


if __name__ == "__main__":
    unittest.main()
