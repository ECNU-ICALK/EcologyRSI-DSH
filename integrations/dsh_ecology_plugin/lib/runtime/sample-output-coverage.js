import { sessionEvents } from "./session-events.js";

const MAX_FEEDBACK_BYTES = 4096;
const MAX_DIAGNOSTICS = 32;

function freezeSampleIds(samples) {
  if (!Array.isArray(samples) || samples.length < 1 || samples.length > 128) {
    throw new Error("sample output coverage requires 1 to 128 frozen samples");
  }
  const ids = samples.map(sample => sample?.sample_id);
  if (ids.some(id => typeof id !== "string" || !id.trim() || id.length > 240)
    || new Set(ids).size !== ids.length) {
    throw new Error("sample output coverage requires unique frozen sample ids");
  }
  return Object.freeze(ids);
}

// This is the existing Host coverage rule, applied before the one-shot tool
// captures an output. Values, evidence, schema and final Host validation are
// untouched. Arrays may be reordered, but their sample-id multiset is exact.
export function sampleCoverageFeedback(sampleIds, args) {
  if (!args || typeof args !== "object" || !Array.isArray(args.decisions)) return null;
  const expected = new Set(sampleIds), seen = new Set();
  const invalidPaths = [], duplicatePaths = [];
  for (const [index, decision] of args.decisions.entries()) {
    const id = decision?.sample_id;
    if (!expected.has(id)) invalidPaths.push(`value.decisions[${index}].sample_id`);
    else if (seen.has(id)) duplicatePaths.push(`value.decisions[${index}].sample_id`);
    else seen.add(id);
  }
  const missing = sampleIds.filter(id => !seen.has(id));
  if (args.decisions.length === sampleIds.length && !invalidPaths.length
    && !duplicatePaths.length && !missing.length) return null;
  const feedback = {
    schema_version: "ecologyrsi-dsh.sample-output-coverage/1",
    instruction: "Submit exactly one decision for each frozen context.samples sample_id, in any order. Remove extra or duplicate decisions and supply missing decisions yourself. Preserve valid predictions. The original schema, Host validation and existing two-output-attempt limit still apply.",
    expected_decision_count: sampleIds.length,
    received_decision_count: args.decisions.length,
    duplicate_sample_id_paths: duplicatePaths.slice(0, MAX_DIAGNOSTICS),
    unknown_or_missing_sample_id_paths: invalidPaths.slice(0, MAX_DIAGNOSTICS),
    missing_sample_ids: missing.slice(0, MAX_DIAGNOSTICS),
    omitted_diagnostic_count: Math.max(0, duplicatePaths.length - MAX_DIAGNOSTICS)
      + Math.max(0, invalidPaths.length - MAX_DIAGNOSTICS) + Math.max(0, missing.length - MAX_DIAGNOSTICS),
  };
  const lists = [feedback.missing_sample_ids, feedback.unknown_or_missing_sample_id_paths,
    feedback.duplicate_sample_id_paths];
  while (Buffer.byteLength(JSON.stringify(feedback), "utf8") > MAX_FEEDBACK_BYTES) {
    const list = lists.find(values => values.length);
    if (!list) throw new Error("sample output coverage feedback limit is invalid");
    list.pop(); feedback.omitted_diagnostic_count++;
  }
  return JSON.stringify(feedback);
}

export function installSampleOutputCoverage(ctx, { stage, samples, parentSessionId, label }) {
  if (stage !== "sample.plan") return { dispose() {} };
  const sampleIds = freezeSampleIds(samples);
  if (typeof ctx?.on !== "function" || !parentSessionId || !label) {
    throw new Error("sample output coverage requires the scoped native tools hook");
  }
  let disposed = false;
  const unsubscribe = ctx.on("tools/pre-execute", async (exec, next) => {
    const decision = await next();
    if (disposed || exec?.signal?.aborted || exec?.name !== "structured_output"
      || decision?.kind !== "allow") return decision;
    const session = exec.agent?.session;
    if (typeof session?.id !== "string" || session.header?.parentSession !== parentSessionId) return decision;
    const descriptor = sessionEvents(session)?.findLast(event => event?.type === "subagent/descriptor");
    if (descriptor?.data?.label !== label) return decision;
    const feedback = sampleCoverageFeedback(sampleIds, exec.arguments);
    if (!feedback) return decision;
    // rc.2's public pre-execute deny materializes a normal failed tool result
    // with error.info. The existing synchronous output-attempt guard sees it.
    return { kind: "deny", reason: feedback, info: { name: "ToolArgsError", code: "INVALID_ARGS" } };
  }, { global: true });
  return { dispose() {
    if (disposed) return;
    disposed = true;
    try { unsubscribe?.(); } catch { /* cleanup cannot replace the stage outcome */ }
  } };
}
