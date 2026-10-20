import { sessionEvents, toolResultIdentity } from "./session-events.js";
import { structuredPhaseError } from "./structured-stage-errors.js";

export function maximumStructuredOutputAttempts(stage) {
  return stage === "generation.reflect" ? 3 : 2;
}

// This is an early stop, never evidence authorizing a result. The existing
// terminal verifier still checks the full tool sequence and unique success.
export function structuredOutputAttemptsExhausted(events, stage) {
  if (!Array.isArray(events)) return false;
  const calls = events.filter(event => event?.type === "tool/call"
    && event.data?.name === "structured_output");
  const maximum = maximumStructuredOutputAttempts(stage);
  if (calls.length > maximum) return true;
  if (calls.length !== maximum) return false;
  const last = calls.at(-1), data = last.data;
  if (!Number.isSafeInteger(last.seq) || typeof data.callId !== "string" || !data.callId
    || new Set(calls.map(call => call.data.callId)).size !== calls.length) return false;
  const results = events.filter(event => event?.type === "tool/result"
    && toolResultIdentity(event)?.callId === data.callId);
  if (results.length !== 1) return false;
  const result = results[0];
  return result.seq > last.seq
    && result.data?.turn === data.turn && result.data?.step === data.step
    && result.sourceEventSeqs?.length === 1 && result.sourceEventSeqs[0] === last.seq
    && toolResultIdentity(result)?.isError === true
    && result.data?.error?.name === "ToolArgsError"
    && result.data.error.code === "INVALID_ARGS";
}

export function watchStructuredOutputBudget(ctx, { stage, parentSessionId, label }) {
  const controller = new AbortController();
  let sessionId = null, error = null, disposed = false, unsubscribe = null;
  let rejectFailure;
  const failure = new Promise((_resolve, reject) => { rejectFailure = reject; });
  failure.catch(() => {});
  const stop = cause => {
    if (disposed || error) return;
    error = cause;
    // Session.append publishes observers synchronously. Abort here, before its
    // tool/result append returns and the Harness can start the next step.
    controller.abort(error);
    rejectFailure(error);
  };
  const inspect = session => {
    if (!session || disposed || error) return;
    const events = sessionEvents(session);
    if (sessionId === null) {
      const descriptor = events?.findLast(event => event?.type === "subagent/descriptor");
      if (!parentSessionId || session.header?.parentSession !== parentSessionId
        || descriptor?.data?.label !== label || typeof session.id !== "string") return;
      sessionId = session.id;
    }
    if (session.id === sessionId && structuredOutputAttemptsExhausted(events, stage)) {
      stop(structuredPhaseError("output_attempts"));
    }
  };
  if (typeof ctx?.on === "function") {
    unsubscribe = ctx.on("session/event", (session, event) => {
      if (["subagent/descriptor", "tool/call", "tool/result"].includes(event?.type)) inspect(session);
    }, { global: true });
  }
  return {
    signal: controller.signal,
    failure,
    bindChild(child) {
      const id = String(child?.id || "");
      if (sessionId !== null && sessionId !== id) {
        stop(structuredPhaseError("child_session"));
        return;
      }
      sessionId = id;
      inspect(ctx?.sessions?.get?.(id));
    },
    throwIfExceeded() { if (error) throw error; },
    dispose() {
      if (disposed) return;
      disposed = true;
      try { unsubscribe?.(); } catch { /* observer cleanup cannot replace the stage outcome */ }
    },
  };
}
