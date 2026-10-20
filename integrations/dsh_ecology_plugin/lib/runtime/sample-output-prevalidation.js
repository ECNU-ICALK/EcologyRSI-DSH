import { SidecarError } from "../sidecar/client.js";
import { isTrustedStructuredPhase } from "./structured-stage-errors.js";
import { sessionEvents } from "./session-events.js";

const SCHEMA = "ecologyrsi-dsh.prediction-prevalidation/1";
const MAX_BYTES = 4096;
const RESPONSE_KEYS = ["diagnostics", "result_digest", "schema_version", "valid"];
const DIAGNOSTIC_PATH = /^value(?:\.(?:schema_version|wave_digest)|\.decisions(?:\[\d{1,3}\](?:\.(?:sample_id|predicted|confidence|method|reason_code|evidence_call_ids(?:\[\d{1,3}\])?))?)?)?$/;
// Only this scoped hook can mint repair authority after an authenticated Host
// response. A model-written error string or lookalike object is not authority.
const repairAuthorities = new WeakMap();

function structuredCall(session, callId) {
  const calls = sessionEvents(session)?.filter(event => event.type === "tool/call"
    && event.data?.name === "structured_output") || [];
  const matches = calls.filter(event => event.data?.callId === callId);
  return typeof callId === "string" && callId.length > 0 && matches.length === 1
    && Number.isInteger(matches[0].seq) && Number.isInteger(matches[0].data?.turn)
    && Number.isInteger(matches[0].data?.step)
    ? { event: matches[0], count: calls.length } : null;
}

function callIdentity(event) {
  return { seq: event.seq, callId: event.data?.callId,
    turn: event.data?.turn, step: event.data?.step };
}

function sameCall(event, expected) {
  return expected && Object.entries(expected).every(([key, value]) => (
    key === "seq" ? event.seq : event.data?.[key]
  ) === value);
}

export function trustedSamplePredictionRepair(authority, rawEvents, {
  stage, sessionId, resultDigest,
} = {}) {
  const state = repairAuthorities.get(authority);
  if (stage !== "sample.plan" || !state || state.disposed || state.cancelled() || !state.first
    || !state.last || state.sessionId !== sessionId
    || state.resultDigest !== resultDigest || !Array.isArray(rawEvents)) return false;
  const calls = rawEvents.filter(event => event.type === "tool/call"
    && event.data?.name === "structured_output");
  return calls.length === 2 && sameCall(calls[0], state.first)
    && sameCall(calls[1], state.last) && state.first.turn === state.last.turn;
}

export function predictionPrevalidationFeedback(response, expectedDigest, { allowPredictionRepair = false } = {}) {
  if (!response || typeof response !== "object" || Array.isArray(response)
    || JSON.stringify(Object.keys(response).sort()) !== JSON.stringify(RESPONSE_KEYS)
    || response.schema_version !== SCHEMA || typeof response.valid !== "boolean"
    || !/^[a-f0-9]{64}$/.test(expectedDigest) || response.result_digest !== expectedDigest
    || !Array.isArray(response.diagnostics) || response.diagnostics.length > 32
    || Buffer.byteLength(JSON.stringify(response), "utf8") > MAX_BYTES
    || response.valid !== (response.diagnostics.length === 0)
    || response.diagnostics.some(item => !item || typeof item !== "object" || Array.isArray(item)
      || JSON.stringify(Object.keys(item).sort()) !== '["code","path"]'
      || typeof item.code !== "string" || !/^[a-z][a-z0-9_]{0,63}$/.test(item.code)
      || typeof item.path !== "string" || item.path.length > 160 || !DIAGNOSTIC_PATH.test(item.path))) {
    throw new SidecarError("invalid_response");
  }
  if (response.valid) return null;
  // Only fixed codes and field paths are exposed. No Host error string, tool
  // value, sample payload, credentials, labels or model text is forwarded.
  const feedback = { schema_version: SCHEMA, diagnostics: response.diagnostics,
    instruction: allowPredictionRepair
      ? "Correct the named fields yourself. For this first blend_requires_two_results rejection only, you may call the existing ecology_execute_prediction_tool within the unchanged frozen tool budget, remaining_calls and deadline to obtain missing successful evidence, or choose another frozen allowed method using existing evidence. Use unique call_ids and wait for each result. Do not call any other tool, invent references or change Host metadata. Submit structured_output once more; the existing two-output-attempt limit still applies."
      : "Correct the named fields yourself using the frozen prediction contract and successful tool evidence. Do not invent references, change Host metadata or request extra tools. The existing two-output-attempt limit still applies." };
  while (Buffer.byteLength(JSON.stringify(feedback), "utf8") > MAX_BYTES) feedback.diagnostics = feedback.diagnostics.slice(0, -1);
  return JSON.stringify(feedback);
}

async function cancellable(operation, signal) {
  if (signal.aborted) throw new SidecarError("aborted");
  let abort;
  const cancelled = new Promise((_resolve, reject) => {
    abort = () => reject(new SidecarError("aborted"));
    signal.addEventListener("abort", abort, { once: true });
  });
  try { return await Promise.race([Promise.resolve().then(() => { if (signal.aborted) throw new SidecarError("aborted"); return operation(); }), cancelled]); }
  finally { signal.removeEventListener("abort", abort); }
}

// A public pre-execute waterfall runs before the one-shot capture tool body.
// Contract feedback becomes INVALID_ARGS; infrastructure failure cancels the
// child and is rethrown by the stage boundary, never offered as model repair.
export function installSampleOutputPrevalidation(ctx, {
  stage, parentSessionId, label, bindOutput, digest, validate, signal = null,
}) {
  const controller = new AbortController();
  let disposed = false, firstError = null, claimedSession = null;
  const repairAuthority = Object.freeze({});
  const repairState = { disposed: false, first: null, last: null, sessionId: null, resultDigest: null,
    cancelled: () => controller.signal.aborted || signal?.aborted === true };
  repairAuthorities.set(repairAuthority, repairState);
  const handle = { signal: controller.signal, repairAuthority, get error() { return firstError; },
    dispose() { disposed = true; repairState.disposed = true; } };
  if (stage !== "sample.plan") return handle;
  if (typeof ctx?.on !== "function" || !parentSessionId || !label
    || [bindOutput, digest, validate].some(fn => typeof fn !== "function")) {
    throw new Error("sample output prevalidation requires the scoped native tools hook");
  }
  const unsubscribe = ctx.on("tools/pre-execute", async (exec, next) => {
    const decision = await next();
    if (disposed || exec?.name !== "structured_output" || decision?.kind !== "allow") return decision;
    const session = exec.agent?.session;
    if (typeof session?.id !== "string" || session.header?.parentSession !== parentSessionId) return decision;
    const descriptor = sessionEvents(session)?.findLast(event => event?.type === "subagent/descriptor");
    if (descriptor?.data?.label !== label) return decision;
    const combined = AbortSignal.any([controller.signal, ...(signal ? [signal] : []), ...(exec.signal ? [exec.signal] : [])]);
    if (combined.aborted) return { kind: "cancel" };
    try {
      if (claimedSession !== null && claimedSession !== session.id) throw new SidecarError("dsh_tool_authorization_failed");
      claimedSession = session.id;
      let structured;
      try { structured = bindOutput(structuredClone(exec.arguments)); }
      catch (error) {
        // The native schema owns primitive/array shape and explicit const
        // mismatches. Preserve its existing corrective INVALID_ARGS path.
        if (isTrustedStructuredPhase(error, "output_schema")) return decision;
        throw error;
      }
      const expectedDigest = digest(structured);
      const response = await cancellable(() => validate({ structured, resultDigest: expectedDigest,
        sessionId: session.id, signal: combined }), combined);
      if (disposed || combined.aborted) return { kind: "cancel" };
      // Do not authorize an object changed while the Host check was in flight.
      if (digest(bindOutput(exec.arguments)) !== expectedDigest) throw new SidecarError("invalid_response");
      // Validate the full wire contract before granting any exception. The
      // exact call comes from the native event log, not model arguments.
      let feedback = predictionPrevalidationFeedback(response, expectedDigest);
      const call = structuredCall(session, exec.callId);
      if (feedback && !repairState.first && call?.count === 1
        && response.diagnostics.every(item => item.code === "blend_requires_two_results")) {
        repairState.first = callIdentity(call.event);
        repairState.sessionId = session.id;
        feedback = predictionPrevalidationFeedback(response, expectedDigest, { allowPredictionRepair: true });
      } else if (!feedback && repairState.first && call?.count === 2
        && repairState.sessionId === session.id) {
        repairState.last = callIdentity(call.event);
        repairState.resultDigest = expectedDigest;
      }
      if (!feedback) return decision;
      return { kind: "deny", reason: feedback, info: { name: "ToolArgsError", code: "INVALID_ARGS" } };
    } catch (error) {
      if (disposed || combined.aborted) return { kind: "cancel" };
      firstError ??= error;
      controller.abort(firstError);
      return { kind: "deny", reason: "Host prevalidation could not complete.",
        info: { name: "HostPrevalidationError", code: "HOST_PREVALIDATION_FAILED" } };
    }
  }, { global: true });
  handle.dispose = () => {
    if (disposed) return;
    disposed = true;
    repairState.disposed = true;
    controller.abort();
    try { unsubscribe?.(); } catch { /* cleanup cannot replace the stage outcome */ }
  };
  return handle;
}
