// Versioned, lossless model-facing views. Authoritative Host values, receipts
// and their digests are never modified by these presentation-only functions.
import { createHash } from "node:crypto";

export const DISPLAY_PROJECTION_SCHEMA = "ecologyrsi-dsh.display-projection/1";
const FIELD = "_display_projection";
const SAMPLE_FIELDS = new Set([
  "algorithm_id", "algorithm_version", "attempt", "candidate_id", "dataset_digest",
  "failure_feedback", "origin_timestamp", "partition",
]);
const SAMPLE_RULE = "For every samples row, insert shared_sample_fields into that row; no row may override a shared field. All original sample fields remain available after expansion. Use the unchanged sample_id.";
const TOOL_RULE = "For each parameter_refs entry i:p, outputs[i].metadata.parameters is exactly shared_parameters[p]. Resolve the reference without changing values; all other fields and original output_digest are unchanged.";

const record = value => value !== null && typeof value === "object" && !Array.isArray(value);
function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (record(value)) return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  return JSON.stringify(value);
}
export function displaySourceDigest(value) {
  return createHash("sha256").update(canonical(value)).digest("hex");
}
const bytes = value => Buffer.byteLength(JSON.stringify(value), "utf8");
function invalid() { throw new Error("invalid lossless display projection"); }
function sourceAllowed(value) {
  if (record(value) && Object.hasOwn(value, FIELD)) invalid();
}
function marker(value, kind, fields, rule) {
  return { schema_version: DISPLAY_PROJECTION_SCHEMA, kind,
    source_digest: displaySourceDigest(value), expansion: rule, ...fields };
}
function checkedMarker(view, kind, fields, rule) {
  const info = view[FIELD];
  if (!record(info) || info.schema_version !== DISPLAY_PROJECTION_SCHEMA || info.kind !== kind
    || info.expansion !== rule || !/^[0-9a-f]{64}$/.test(info.source_digest)
    || Object.keys(info).sort().join() !== ["schema_version", "kind", "source_digest", "expansion", ...fields].sort().join()) invalid();
  return info;
}
function restored(view, info) {
  delete view[FIELD];
  if (displaySourceDigest(view) !== info.source_digest) invalid();
  return view;
}

export function projectSampleContext(context) {
  sourceAllowed(context);
  if (!record(context) || !Array.isArray(context.samples) || context.samples.length < 2
    || !context.samples.every(record)) return context;
  const common = Object.fromEntries(Object.entries(context.samples[0]).filter(([key, value]) =>
    SAMPLE_FIELDS.has(key) && context.samples.every(row => Object.hasOwn(row, key)
      && canonical(row[key]) === canonical(value))));
  if (Object.keys(common).length === 0) return context;
  const view = structuredClone(context);
  view[FIELD] = marker(context, "sample-context", { shared_sample_fields: structuredClone(common) }, SAMPLE_RULE);
  for (const row of view.samples) for (const key of Object.keys(common)) delete row[key];
  // Include the version, digest and expansion instructions in the size test.
  return bytes(view) < bytes(context) ? view : context;
}

export function expandSampleContext(view) {
  if (!record(view) || !Object.hasOwn(view, FIELD)) return view;
  const info = checkedMarker(view, "sample-context", ["shared_sample_fields"], SAMPLE_RULE);
  const shared = info.shared_sample_fields;
  if (!record(shared) || Object.keys(shared).length === 0
    || Object.keys(shared).some(key => !SAMPLE_FIELDS.has(key))
    || !Array.isArray(view.samples) || view.samples.length < 2
    || !view.samples.every(row => record(row) && Object.keys(shared).every(key => !Object.hasOwn(row, key)))) invalid();
  const full = structuredClone(view);
  full.samples = full.samples.map(row => ({ ...structuredClone(shared), ...row }));
  return restored(full, info);
}

export function projectPredictionToolResult(value) {
  sourceAllowed(value);
  // Failed/error results and all unrelated tools keep their exact rendering.
  if (!record(value) || value.accepted !== true || value.status !== "completed"
    || !Array.isArray(value.outputs) || value.outputs.length < 2) return value;
  const groups = new Map();
  for (const [index, row] of value.outputs.entries()) {
    const parameters = row?.metadata?.parameters;
    if (!record(row?.metadata) || !record(parameters) || bytes(parameters) < 128) continue;
    const key = canonical(parameters);
    if (!groups.has(key)) groups.set(key, { parameters, indices: [] });
    groups.get(key).indices.push(index);
  }
  const shared = {}, refs = {};
  for (const { parameters, indices } of groups.values()) {
    if (indices.length < 2) continue;
    const ref = `p${Object.keys(shared).length + 1}`;
    shared[ref] = structuredClone(parameters);
    for (const index of indices) refs[index] = ref;
  }
  if (Object.keys(refs).length === 0) return value;
  const view = structuredClone(value);
  view[FIELD] = marker(value, "prediction-tool-parameters", {
    shared_parameters: shared, parameter_refs: refs,
  }, TOOL_RULE);
  for (const index of Object.keys(refs)) delete view.outputs[index].metadata.parameters;
  return bytes(view) < bytes(value) ? view : value;
}

export function expandPredictionToolResult(view) {
  if (!record(view) || !Object.hasOwn(view, FIELD)) return view;
  const info = checkedMarker(view, "prediction-tool-parameters", ["shared_parameters", "parameter_refs"], TOOL_RULE);
  const shared = info.shared_parameters, refs = info.parameter_refs;
  if (view.accepted !== true || view.status !== "completed" || !Array.isArray(view.outputs)
    || !record(shared) || !record(refs) || Object.keys(refs).length === 0
    || Object.keys(shared).length === 0
    || !Object.entries(shared).every(([key, value]) => /^p[1-9][0-9]*$/.test(key) && record(value))) invalid();
  const counts = new Map();
  for (const [index, ref] of Object.entries(refs)) {
    if (!/^(0|[1-9][0-9]*)$/.test(index) || !Number.isSafeInteger(Number(index))
      || Number(index) >= view.outputs.length || typeof ref !== "string" || !Object.hasOwn(shared, ref)
      || !record(view.outputs[index]?.metadata) || Object.hasOwn(view.outputs[index].metadata, "parameters")) invalid();
    counts.set(ref, (counts.get(ref) || 0) + 1);
  }
  if (Object.keys(shared).some(key => (counts.get(key) || 0) < 2)) invalid();
  const full = structuredClone(view);
  for (const [index, ref] of Object.entries(refs)) full.outputs[index].metadata.parameters = structuredClone(shared[ref]);
  return restored(full, info);
}

export function modelContextView(stage, context) {
  // Keep critic/reflection and all non-planning contracts exactly as before.
  return stage === "sample.plan" ? projectSampleContext(context) : context;
}
