import { sessionEvents } from "./session-events.js";

const STAGE = "generation.research-synthesis";
const MAX_FEEDBACK_BYTES = 4096;
const MAX_MISSING_PATHS = 32;
const MAX_INSPECTED_ITEMS = 8;

function object(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function commonRequired(schema) {
  const variants = Array.isArray(schema?.oneOf) ? schema.oneOf : [schema];
  const first = variants[0]?.required;
  if (!Array.isArray(first)) return [];
  return first.filter(name => typeof name === "string"
    && variants.every(branch => branch?.required?.includes(name)));
}

// Derive presentation hints from the same submitted schema, never a second
// acceptance contract. In particular, every oneOf branch requires its title.
export function researchNestedRequiredFields(stage, schema) {
  if (stage !== STAGE) return {};
  return Object.fromEntries(["evidence", "candidate_directions"].map(name => [
    `${name}[*]`, commonRequired(schema?.properties?.[name]?.items),
  ]).filter(([, fields]) => fields.length));
}

export function researchMissingFieldFeedback(schema, args) {
  if (!object(args) || !Array.isArray(args.candidate_directions)) return null;
  const itemSchema = schema?.properties?.candidate_directions?.items;
  const variants = Array.isArray(itemSchema?.oneOf) ? itemSchema.oneOf : [itemSchema];
  const common = commonRequired(itemSchema);
  const missing = [];
  for (const [index, item] of args.candidate_directions.slice(0, MAX_INSPECTED_ITEMS).entries()) {
    if (!object(item)) continue;
    const matching = variants.filter(branch => typeof item.mutation_axis === "string"
      && branch?.properties?.mutation_axis?.const === item.mutation_axis);
    // An unknown/ambiguous discriminator cannot select a branch. Its common
    // required fields remain unambiguous; never guess a scientific axis.
    const required = matching.length === 1 ? matching[0].required : common;
    for (const name of required || []) {
      if (typeof name === "string" && !Object.hasOwn(item, name)) {
        missing.push(`value.candidate_directions[${index}].${name}`);
      }
    }
  }
  if (!missing.length) return null;
  const feedback = {
    schema_version: "ecologyrsi-dsh.research-schema-feedback/1",
    instruction: "Add each named required field to its existing direction object, preserving other valid fields. These diagnostics supplement INVALID_ARGS; all original schema constraints and the existing output-attempt limit still apply.",
    missing_required_fields: missing.slice(0, MAX_MISSING_PATHS),
    omitted_missing_field_count: Math.max(0, missing.length - MAX_MISSING_PATHS),
    uninspected_direction_count: Math.max(0, args.candidate_directions.length - MAX_INSPECTED_ITEMS),
  };
  while (Buffer.byteLength(JSON.stringify(feedback), "utf8") > MAX_FEEDBACK_BYTES
    && feedback.missing_required_fields.length) {
    feedback.missing_required_fields.pop();
    feedback.omitted_missing_field_count++;
  }
  return feedback.missing_required_fields.length ? JSON.stringify(feedback) : null;
}

// Harness's public post-execute waterfall can enrich failed tool content while
// retaining its original error/code/isError. It runs before the next request,
// including when the first tool call precedes the subagent start promise.
export function installResearchSchemaFeedback(ctx, { stage, schema, parentSessionId, label }) {
  if (stage !== STAGE || !parentSessionId || !label || typeof ctx?.on !== "function") {
    return { dispose() {} };
  }
  const frozenSchema = structuredClone(schema);
  let disposed = false;
  const unsubscribe = ctx.on("tools/post-execute", async (exec, result, next) => {
    const decision = await next();
    // The public tools hook receives normalized errors under `info`; session
    // tool/result events use the legacy flat projection. Never let a flat
    // fallback override present (including malformed) normalized metadata.
    const error = object(result?.error) ? result.error : null;
    const info = error && Object.hasOwn(error, "info") ? error.info : error;
    if (disposed || exec?.signal?.aborted || exec?.name !== "structured_output"
      || result?.isError !== true || info?.name !== "ToolArgsError"
      || info?.code !== "INVALID_ARGS" || decision?.kind !== "accept"
      || Object.hasOwn(decision, "value")) return decision;
    const session = exec.agent?.session;
    if (typeof session?.id !== "string" || session.header?.parentSession !== parentSessionId) return decision;
    const descriptor = sessionEvents(session)?.findLast(event => event?.type === "subagent/descriptor");
    if (descriptor?.data?.label !== label) return decision;
    const content = decision.content ?? result.content;
    if (!Array.isArray(content)) return decision;
    const feedback = researchMissingFieldFeedback(frozenSchema, exec.arguments);
    if (!feedback) return decision;
    return { ...decision, content: [
      ...content,
      { type: "text", text: feedback },
    ] };
  }, { global: true });
  return { dispose() {
    if (disposed) return;
    disposed = true;
    try { unsubscribe?.(); } catch { /* cleanup must not replace the stage outcome */ }
  } };
}
