import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { prepareStageOutputSchema } from "../lib/runtime/stage-runner.js";
import { researchMissingFieldFeedback, researchNestedRequiredFields } from "../lib/runtime/research-schema-feedback.js";

const synthesis = JSON.parse(await readFile(new URL("../schemas/research-synthesis.schema.json", import.meta.url)));
const reflection = JSON.parse(await readFile(new URL("../schemas/generation-reflection.schema.json", import.meta.url)));
const portfolio = {
  policy: "mechanism_evidence_search@1",
  mechanism_catalog: {
    mechanism_by_axis: { scientific_parameter: "parameter_adjustment", instruction_profile: "agent_decision" },
    input_refs: ["history:air_temperature"], target_cells: ["air_temperature@24h"],
  },
};
const base = {
  knowledge_snapshot: { evidence_catalog: [{ knowledge_id: "paper:one" }] },
  required_candidate_direction_count: 4,
  synthesis_contract: {
    allowed_mutation_targets: { scientific_parameter: ["ridge_alpha"], instruction_profile: ["balanced@1"] },
    mutation_directions_by_axis: { scientific_parameter: ["increase", "decrease"], instruction_profile: ["select"] },
  },
};

test("new synthesis binds mechanism to each actual mutation branch and frozen inputs", () => {
  const context = structuredClone(base);
  context.synthesis_contract.candidate_portfolio = portfolio;
  const schema = prepareStageOutputSchema("generation.research-synthesis", synthesis, context);
  const variants = schema.properties.candidate_directions.items.oneOf;
  for (const variant of variants) {
    assert.ok(variant.required.includes("mechanism_hypothesis"));
    const hypothesis = variant.properties.mechanism_hypothesis.properties;
    assert.deepEqual(hypothesis.mechanism_id.enum, [portfolio.mechanism_catalog.mechanism_by_axis[variant.properties.mutation_axis.const]]);
    assert.deepEqual(hypothesis.input_refs.items.enum, portfolio.mechanism_catalog.input_refs);
  }
  const feedback = JSON.parse(researchMissingFieldFeedback(schema, {
    candidate_directions: [{ mutation_axis: "scientific_parameter", mechanism_hypothesis: {} }],
  }));
  assert.ok(feedback.missing_required_fields.includes("value.candidate_directions[0].mechanism_hypothesis.mechanism_id"));
  assert.ok(researchNestedRequiredFields("generation.research-synthesis", schema)["candidate_directions[*].mechanism_hypothesis"].includes("input_refs"));
});

test("old runs retain their vocabulary and reflection receives the same new contract", () => {
  const old = prepareStageOutputSchema("generation.research-synthesis", synthesis, base);
  for (const item of old.properties.candidate_directions.items.oneOf) {
    assert.equal(item.properties.mechanism_hypothesis, undefined);
    assert.ok(!item.required.includes("mechanism_hypothesis"));
  }
  const fresh = prepareStageOutputSchema("generation.reflect", reflection, { candidate_portfolio: portfolio });
  assert.ok(fresh.properties.candidate_directions.items.required.includes("mechanism_hypothesis"));
  const historical = prepareStageOutputSchema("generation.reflect", reflection, {});
  assert.equal(historical.properties.candidate_directions.items.properties.mechanism_hypothesis, undefined);
});
