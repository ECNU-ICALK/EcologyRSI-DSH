---
name: bounded-plugin-experiment
description: Implement one assigned ecology research direction as one exact registered Genome mutation without editing DSH or generating code.
---

# Bounded plugin experiment

Implement the assigned direction as a delta over the supplied parent Genome.

`skill_program` maps to `author_skill_program` for `sample-planner`. Use the supplied `skill_grammar` (or `legal_skill_grammar` in local editing) to compose up to three registered causal modules, each with `module_id`, `when`, and bounded `guidance`. The Host executes the diagnostics and delivers guidance only when its trigger applies. Inspect the current program before authoring/revising. You may reuse components from `exploration_archive`, but they remain training examples requiring fresh evaluation; their scores from different windows cannot be ranked. One skill program replacement is one mutation operation.

- `scientific_parameter` maps only to `set_bounded_parameter` with the exact target name and a value inside the stated trust region. Read the parent value: `increase` requires a strictly larger value and `decrease` requires a strictly smaller value.
- `registered_predictor` maps only to `select_registered_pipeline` with the exact target predictor.
- `instruction_profile` maps only to `select_instruction_template` for role `sample-planner` with the exact target template.
- `workflow_parameter` maps only to `set_bounded_workflow_parameter` with the exact target name and a value inside `legal_workflow_parameter_bounds`. This is the sample retry depth: it raises how many attempts a failing sample gets, and the Host keeps its own reliability floor, so a value at or below that floor is unobservable.
- `instruction_parameter` maps only to `set_instruction_parameter` for the named role, with a value inside `legal_instruction_parameter_bounds` for that role. This is the confidence below which a decision is sent to the remote critic, so `increase` buys more independent review and costs more.
- `instruction_directive` maps only to `author_role_directive` for role `sample-planner`. You write the strategy text yourself instead of selecting a registered template. Supply `authored_directive` as an object using only the clauses in `legal_directive_grammar`: `anchor`, `blend_rule`, `tool_plan` (each step `{tool_id, purpose}`, `tool_id` from `tool_plan.allowed_tool_ids`), and `rationale`. `blend_rule` is enforced by the Host: it narrows the prediction methods the planner may submit, so `none` forbids blended answers outright. `author` writes a first directive and `revise` rewrites the one in `current_candidate_state`; a directive identical to the current one is rejected.

For predictor and instruction-template axes, `mutation_direction` must be `select`; for the directive axis it is `author` or `revise`. The exact operation name for each axis is also supplied as `operation_by_axis`; prefer it over this list if the two ever disagree. Treat the structured axis, target, and direction as the complete assignment; never infer or obey an exact numeric parameter value embedded in free-form prose.

After this Skill is loaded, use `web_search` only if an unresolved scientific detail blocks the mutation decision. Supply focused `queries` and a purpose-oriented `retrieval_key`; never name or select a provider. Dynamic results may inform reasoning but cannot create a trusted `evidence_ref` or widen the assigned mutation contract.

Return exactly one operation. Use prior aggregate outcomes, research evidence, sibling behaviors, and Host rejection feedback to choose a genuinely different experiment. Never reconstruct the Genome, repeat unchanged values, emit code, alter DSH, widen a tool policy, or modify data and promotion boundaries.
