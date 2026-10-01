// Bind the model's output vocabulary to the same immutable catalog the Host
// validates. Use only schema primitives supported by the DSH output compiler.
import { researchExecutionPolicy } from "./research-execution-policy.js";

function boundText(field, maximum) {
  field.description = `Aim for at most ${maximum} characters; this is a concise writing target. ${field.description || ""}`.trim();
}

export function specializeResearchOutputSchema(stage, schema, context) {
  if (stage !== "generation.research-synthesis") return;
  const catalog = context?.knowledge_snapshot?.evidence_catalog;
  const targets = context?.synthesis_contract?.allowed_mutation_targets;
  const directionsByAxis = context?.synthesis_contract?.mutation_directions_by_axis;
  const count = context?.required_candidate_direction_count;
  if (!Array.isArray(catalog) || !targets || !directionsByAxis || !Number.isSafeInteger(count) || count < 1 || count > 8) {
    throw new Error("research synthesis requires a frozen evidence and mutation catalog");
  }
  const refs = new Set();
  for (const card of catalog) {
    if (!card || typeof card !== "object") continue;
    for (const value of [card.knowledge_id, card.evidence_digest, card.capability_id,
      ...(Array.isArray(card.capability_ids) ? card.capability_ids : [])]) {
      if (typeof value === "string" && value.trim()) refs.add(value.trim());
    }
  }
  if (!refs.size) throw new Error("research synthesis frozen evidence catalog is empty");
  const referenceSchema = { type: "string", enum: [...refs].sort() };
  schema.properties.evidence.items.properties.evidence_ref = structuredClone(referenceSchema);
  const directions = schema.properties.candidate_directions;
  directions.description = `Return exactly ${count} distinct directions, one per candidate slot.`;
  const item = structuredClone(directions.items);
  item.properties.evidence_refs.items = structuredClone(referenceSchema);
  if (researchExecutionPolicy(context)) {
    boundText(schema.properties.summary, 1200);
    schema.properties.evidence.description = "Prefer at most 8 relevant evidence items; do not summarize the entire catalog.";
    boundText(schema.properties.evidence.items.properties.finding, 400);
    boundText(schema.properties.evidence.items.properties.relevance, 240);
    for (const [field, maximum] of Object.entries({
      title: 120, hypothesis: 400, target_weakness: 240, capability_focus: 160,
      expected_tradeoff: 240, success_criterion: 240,
    })) boundText(item.properties[field], maximum);
    item.properties.evidence_refs.description = "Prefer at most 4 relevant frozen identifiers for this direction.";
  }
  const variants = [];
  for (const [axis, values] of Object.entries(targets)) {
    const allowedDirections = directionsByAxis[axis];
    if (!item.properties.mutation_axis.enum.includes(axis)
      || !Array.isArray(values) || values.some(value => typeof value !== "string" || !value.trim())
      || !Array.isArray(allowedDirections) || !allowedDirections.length
      || allowedDirections.some(value => !item.properties.mutation_direction.enum.includes(value))) {
      throw new Error("research synthesis mutation catalog is invalid");
    }
    if (!values.length) continue;
    const variant = structuredClone(item);
    variant.properties.mutation_axis = { type: "string", const: axis };
    variant.properties.mutation_target = { type: "string", enum: [...new Set(values)].sort() };
    variant.properties.mutation_direction = { type: "string", enum: allowedDirections };
    variants.push(variant);
  }
  if (!variants.length) throw new Error("research synthesis has no legal mutation targets");
  const schedule = context.synthesis_contract.diversity_schedule;
  for (const family of schedule?.required_family_by_slot || []) {
    if (!schedule.axes_by_family?.[family]?.some(axis => targets[axis]?.length)) {
      throw new Error("research synthesis diversity schedule has no legal mutation targets");
    }
  }
  // DSH drops JSON Schema if/then/allOf. A oneOf of complete objects preserves
  // the axis/target/direction relationship in the actual submitted schema.
  directions.items = variants.length === 1 ? variants[0] : { oneOf: variants };
}
