---
name: bounded-plugin-experiment
description: Implement one assigned ecology research direction as one exact registered Genome mutation without editing DSH or generating code.
---

# Bounded plugin experiment

Implement the assigned direction as a delta over the supplied parent Genome.

- `scientific_parameter` maps only to `set_bounded_parameter` with the exact target name and a value inside the stated trust region. Read the parent value: `increase` requires a strictly larger value and `decrease` requires a strictly smaller value.
- `registered_predictor` maps only to `select_registered_pipeline` with the exact target predictor.
- `instruction_profile` maps only to `select_instruction_template` for role `sample-planner` with the exact target template.

For predictor and instruction axes, `mutation_direction` must be `select`. Treat the structured axis, target, and direction as the complete assignment; never infer or obey an exact numeric parameter value embedded in free-form prose.

Return exactly one operation. Use prior aggregate outcomes, research evidence, sibling behaviors, and Host rejection feedback to choose a genuinely different experiment. Never reconstruct the Genome, repeat unchanged values, emit code, alter DSH, widen a tool policy, or modify data and promotion boundaries.
