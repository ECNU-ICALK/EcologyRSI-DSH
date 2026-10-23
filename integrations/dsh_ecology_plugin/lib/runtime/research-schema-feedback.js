import { sessionEvents } from "./session-events.js";

const STAGE = "generation.research-synthesis";
const MAX_FEEDBACK_BYTES = 4096;
const MAX_MISSING_PATHS = 32;
const MAX_INSPECTED_ITEMS = 8;
const MAX_INSPECTED_EVIDENCE = 32;
const MAX_INSPECTED_REFS = 16;

function oneEditApart(left, right) {
  if (Math.abs(left.length - right.length) > 1) return false;
  let i = 0, j = 0, edits = 0;
  while (i < left.length && j < right.length) {
    if (left[i] === right[j]) { i++; j++; continue; }
    if (++edits > 1) return false;
    if (left.length >= right.length) i++;
    if (right.length >= left.length) j++;
  }
  return edits + (i < left.length || j < right.length ? 1 : 0) === 1;
}

function invalidReference(schema, value, path) {
  const allowed = schema?.enum;
  if (typeof value !== "string" || !Array.isArray(allowed) || allowed.includes(value)) return null;
  // Suggestions are diagnostics, never an identity substitution. Only exact
  // members of this submitted schema may be offered to the model for review.
  const nearby = value.length <= 160
    ? allowed.filter(ref => typeof ref === "string" && ref.length <= 160 && oneEditApart(value, ref))
    : [];
  return { path, submitted_value: value.slice(0, 160),
    submitted_value_truncated: value.length > 160,
    allowed_identifiers_one_edit_away: nearby.slice(0, 3),
    omitted_nearby_identifier_count: Math.max(0, nearby.length - 3) };
}

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
  const result = Object.fromEntries(["evidence", "candidate_directions"].map(name => [
    `${name}[*]`, commonRequired(schema?.properties?.[name]?.items),
  ]).filter(([, fields]) => fields.length));
  const item = schema?.properties?.candidate_directions?.items;
  const nested = commonRequired((item?.oneOf?.[0] || item)?.properties?.mechanism_hypothesis);
  if (nested.length) result["candidate_directions[*].mechanism_hypothesis"] = nested;
  return result;
}

export function researchMissingFieldFeedback(schema, args) {
  if (!object(args) || !Array.isArray(args.candidate_directions)) return null;
  const itemSchema = schema?.properties?.candidate_directions?.items;
  const variants = Array.isArray(itemSchema?.oneOf) ? itemSchema.oneOf : [itemSchema];
  const common = commonRequired(itemSchema);
  const missing = [];
  const invalidReferences = [];
  for (const [index, item] of (Array.isArray(args.evidence) ? args.evidence : []).slice(0, MAX_INSPECTED_EVIDENCE).entries()) {
    const diagnostic = invalidReference(schema?.properties?.evidence?.items?.properties?.evidence_ref,
      item?.evidence_ref, `value.evidence[${index}].evidence_ref`);
    if (diagnostic) invalidReferences.push(diagnostic);
  }
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
    const branch = matching.length === 1 ? matching[0] : variants[0];
    if (matching.length === 1 && Array.isArray(item.evidence_refs)) {
      for (const [refIndex, ref] of item.evidence_refs.slice(0, MAX_INSPECTED_REFS).entries()) {
        const diagnostic = invalidReference(branch?.properties?.evidence_refs?.items, ref,
          `value.candidate_directions[${index}].evidence_refs[${refIndex}]`);
        if (diagnostic) invalidReferences.push(diagnostic);
      }
    }
    if (object(item.mechanism_hypothesis)) {
      for (const name of commonRequired(branch?.properties?.mechanism_hypothesis)) {
        if (!Object.hasOwn(item.mechanism_hypothesis, name)) {
          missing.push(`value.candidate_directions[${index}].mechanism_hypothesis.${name}`);
        }
      }
    }
  }
  if (!missing.length && !invalidReferences.length) return null;
  const feedback = {
    schema_version: "ecologyrsi-dsh.research-schema-feedback/1",
    instruction: "Add each named required field. For every invalid evidence reference, verify the cited source in the frozen catalog and copy its exact allowed identifier at every listed path; nearby identifiers are suggestions, not verified source equivalences. If no catalog source supports a claim, remove its unsupported citation/claim. These diagnostics supplement INVALID_ARGS; all original schema constraints and the existing output-attempt limit still apply. Arguments are not changed by the Host.",
    missing_required_fields: missing.slice(0, MAX_MISSING_PATHS),
    omitted_missing_field_count: Math.max(0, missing.length - MAX_MISSING_PATHS),
    uninspected_direction_count: Math.max(0, args.candidate_directions.length - MAX_INSPECTED_ITEMS),
    invalid_evidence_references: invalidReferences.slice(0, MAX_MISSING_PATHS),
    omitted_invalid_reference_count: Math.max(0, invalidReferences.length - MAX_MISSING_PATHS),
    uninspected_evidence_count: Math.max(0, (Array.isArray(args.evidence) ? args.evidence.length : 0) - MAX_INSPECTED_EVIDENCE),
  };
  while (Buffer.byteLength(JSON.stringify(feedback), "utf8") > MAX_FEEDBACK_BYTES
    && (feedback.missing_required_fields.length || feedback.invalid_evidence_references.length)) {
    if (feedback.invalid_evidence_references.length) {
      feedback.invalid_evidence_references.pop();
      feedback.omitted_invalid_reference_count++;
    } else {
      feedback.missing_required_fields.pop();
      feedback.omitted_missing_field_count++;
    }
  }
  return feedback.missing_required_fields.length || feedback.invalid_evidence_references.length ? JSON.stringify(feedback) : null;
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
