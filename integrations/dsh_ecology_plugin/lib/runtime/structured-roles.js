import {
  createStructuredDeadline,
  remainingStructuredDeadlineMs,
  structuredDeadlineExpired,
  validateStructuredDeadline,
  validateStructuredTimeoutMs,
} from "./structured-deadline.js";
import {
  isTrustedStructuredPhase,
  structuredPhaseError,
} from "./structured-stage-errors.js";
import { validateStructuredStageBudget } from "./research-execution-policy.js";

function structuredResult(run) {
  if (typeof run?.result === "function") return run.result();
  if (run?.result && typeof run.result.then === "function") return run.result;
  if (run?.result !== undefined) return run.result;
  return run;
}

function operationalTimeoutError() {
  const error = new Error("structured role operational timeout");
  error.code = "structured_role_operational_timeout";
  return error;
}

const STRUCTURED_PERSISTENCE_MAX_ATTEMPTS = 2;
const RETRYABLE_PERSISTENCE_CODES = new Set([
  "structured_result_persistence_unavailable",
  "unavailable",
  "timeout",
  "dsh_session_projection_not_ready",
]);

function retryablePersistenceError(error) {
  // These are the only transient states safe to replay with the exact frozen
  // envelope: a Host append/control-plane fault, a loopback transport fault,
  // or eventual consistency in the child Session projection. Contract,
  // authorization, admission and cancellation failures remain fail-closed.
  return RETRYABLE_PERSISTENCE_CODES.has(error?.code);
}

function detachCleanup(operation) {
  try {
    Promise.resolve(operation?.()).catch(() => {});
  } catch {
    // Cleanup must not replace the structured-role outcome or wait forever.
  }
}

export async function runStructuredRole(
  roleHost,
  reservedBinding,
  request,
  {
    pendingStarts,
    admission,
    persist,
    timeoutMs,
    deadline,
    classifyMissingCapture,
    observeChild,
    signal = null,
  } = {},
) {
  if (!roleHost?.agent) throw new Error("structured role requires a retained role-host Agent");
  if (!reservedBinding?.label) throw new Error("structured role requires a pre-registered child label");
  if (!request?.outputSchema || typeof request.outputSchema !== "object") {
    throw new Error("structured role requires an output schema");
  }
  validateStructuredStageBudget(
    reservedBinding.binding?.stage, request.maxTokens,
    request.researchExecutionPolicy === undefined ? null
      : { research_execution_policy: request.researchExecutionPolicy },
    "structured role maxTokens",
  );
  if (!pendingStarts?.start || typeof persist !== "function") {
    throw new Error("structured role lifecycle services are required");
  }
  if (
    timeoutMs !== undefined
  ) {
    validateStructuredTimeoutMs(timeoutMs);
  }
  if (deadline !== undefined) {
    validateStructuredDeadline(deadline);
    if (timeoutMs !== undefined && deadline.timeoutMs !== timeoutMs) {
      throw new Error("structured role deadline timeout mismatch");
    }
  }
  let pending = null;
  let run;
  let usageObserver;
  let timeout = null;
  let resultPersisted = false;
  const hardDeadline = deadline === undefined
    ? (timeoutMs === undefined ? null : createStructuredDeadline(timeoutMs))
    : deadline;
  const deadlineAt = hardDeadline?.deadlineAt ?? null;
  let deadlinePromise = null;
  let timedOut = false;
  const timeoutError = operationalTimeoutError();
  const abortFromCaller = () => {
    pending?.controller?.abort(signal?.reason);
  };
  signal?.addEventListener("abort", abortFromCaller, { once: true });
  const expireDeadline = () => {
    if (!timedOut) {
      timedOut = true;
      pending?.controller.abort();
    }
    return timeoutError;
  };
  const deadlineExpired = () => (
    hardDeadline !== null && (timedOut || structuredDeadlineExpired(hardDeadline))
  );
  const requireBeforeDeadline = () => {
    if (deadlineExpired()) throw expireDeadline();
  };
  const withinDeadline = async (operation) => {
    if (deadlinePromise === null) return await operation();
    requireBeforeDeadline();
    try {
      const value = await Promise.race([
        Promise.resolve().then(operation),
        deadlinePromise,
      ]);
      requireBeforeDeadline();
      return value;
    } catch (error) {
      if (deadlineExpired() || error === timeoutError) throw expireDeadline();
      throw error;
    }
  };
  const remainingTimeoutMs = () => {
    requireBeforeDeadline();
    return Math.max(1, remainingStructuredDeadlineMs(hardDeadline));
  };
  if (deadlineAt !== null) {
    deadlinePromise = new Promise((_resolve, reject) => {
      timeout = setTimeout(
        () => reject(expireDeadline()),
        remainingStructuredDeadlineMs(hardDeadline),
      );
    });
    deadlinePromise.catch(() => {});
  }
  try {
    requireBeforeDeadline();
    const outputSchema = structuredClone(request.outputSchema);
    requireBeforeDeadline();
    try {
      pending = pendingStarts.start("one-shot", {
        provider: "spawn",
        parent: roleHost.agent,
        label: reservedBinding.label,
        prompt: [{ type: "text", text: request.prompt }],
        outputSchema,
        // DSH accepts child generation overrides through
        // SubagentStartRequest.agentOptions. Provider and model deliberately
        // remain inherited from the retained role-host. A larger synthesis
        // ceiling requires the exact Host-frozen stage policy above.
        ...(request.maxTokens === undefined
          ? {}
          : { agentOptions: { maxTokens: request.maxTokens } }),
      }, {
        roleHostAgent: roleHost.agent,
        runId: reservedBinding?.launch?.run_id || reservedBinding?.binding?.run_id,
      });
      requireBeforeDeadline();
      run = await withinDeadline(() => pending.promise);
      // Failure to observe usage must not invent or replace a model outcome.
      try { usageObserver = observeChild?.(run); } catch { /* reflected as missing coverage */ }
    } catch (error) {
      if (deadlineExpired() || error === timeoutError) throw expireDeadline();
      throw structuredPhaseError(
        error?.code === "provider_stage_admission_closed" ? "control"
          : error?.code === "UNSUPPORTED_SCHEMA" ? "output_schema" : "start",
        error,
      );
    }
    let result;
    try {
      result = await withinDeadline(() => structuredResult(run));
    } catch (error) {
      if (deadlineExpired() || error === timeoutError) throw expireDeadline();
      throw structuredPhaseError("result", error);
    }
    requireBeforeDeadline();
    const stopReason = result?.stopReason;
    requireBeforeDeadline();
    // DSH's one-shot result preserves the consumed turn's native terminal
    // reason. A direct max-tokens result must not fall through to the generic
    // model failure, even before Session projection has caught up.
    if (stopReason === "max-tokens") throw structuredPhaseError("output_budget");
    const structured = result?.structured;
    requireBeforeDeadline();
    const validStructured = structured
      && typeof structured === "object"
      && !Array.isArray(structured);
    const hasCaptureClassifier = typeof classifyMissingCapture === "function";
    let captureDisposition = null;
    if (!validStructured && ["error", "completed"].includes(stopReason) && hasCaptureClassifier) {
      try {
        const classificationDeadline = deadlineAt === null ? null : Object.freeze({
          signal: pending.controller.signal,
          throwIfExpired: requireBeforeDeadline,
          remainingTimeoutMs,
        });
        captureDisposition = await withinDeadline(
          () => classifyMissingCapture({ run, result }, classificationDeadline),
        );
      } catch {
        captureDisposition = "non-missing";
      }
    }
    requireBeforeDeadline();
    if (captureDisposition === "tool-protocol") {
      throw structuredPhaseError("tool_protocol");
    }
    if (captureDisposition === "output-budget") {
      throw structuredPhaseError("output_budget");
    }
    // Preserve existing completed-without-result handling for other shapes.
    if (stopReason === "completed") captureDisposition = null;
    if (stopReason && stopReason !== "completed") {
      if (stopReason === "aborted") throw structuredPhaseError("aborted");
      const captureKind = typeof captureDisposition === "object"
        ? captureDisposition?.kind
        : captureDisposition;
      if (captureKind === "missing") throw structuredPhaseError("capture");
      if (captureKind === "retryable-model") {
        throw structuredPhaseError("model");
      }
      if (captureKind === "retryable-provider") {
        throw structuredPhaseError("model", null, {
          providerRateLimit: captureDisposition?.limitKind !== "concurrency",
          retryAfterMs: captureDisposition?.retryAfterMs,
        });
      }
      if (stopReason === "error" && captureDisposition !== null) {
        throw structuredPhaseError("model_terminal");
      }
      throw structuredPhaseError("model");
    }
    if (!validStructured) {
      if (hasCaptureClassifier && captureDisposition !== "missing") {
        throw structuredPhaseError("model_terminal");
      }
      throw structuredPhaseError("capture");
    }
    let admissionOpen = true;
    if (admission?.isOpen) {
      try {
        admissionOpen = await withinDeadline(
          () => admission.isOpen(reservedBinding),
        );
      } catch (error) {
        if (deadlineExpired() || error === timeoutError) throw expireDeadline();
        throw structuredPhaseError("admission", error);
      }
    }
    requireBeforeDeadline();
    if (!admissionOpen) {
      const error = structuredPhaseError("admission_closed");
      // Keep the public message stable for callers that surface this safe state.
      error.message = "structured result admission is closed";
      throw error;
    }
    // rc.6 SubagentRun publishes the real child Session identity as `id`.
    // `childId` belongs to the continuable-start seam.
    const sessionId = String(run?.id || "");
    requireBeforeDeadline();
    if (!sessionId) throw structuredPhaseError("child_session");
    const persistedStructured = structuredClone(structured);
    requireBeforeDeadline();
    const persistenceDeadline = deadlineAt === null ? null : Object.freeze({
      signal: pending.controller.signal,
      throwIfExpired: requireBeforeDeadline,
      remainingTimeoutMs,
    });
    let accepted;
    for (let persistenceAttempt = 1;
      persistenceAttempt <= STRUCTURED_PERSISTENCE_MAX_ATTEMPTS;
      persistenceAttempt += 1) {
      try {
        // Re-submit the exact immutable output from this child session. In
        // particular, do not begin another planner turn or prediction-tool
        // call merely because the Host ledger was briefly unavailable.
        accepted = await withinDeadline(() => persist({
          binding: reservedBinding,
          structured: persistedStructured,
          session_id: sessionId,
        }, persistenceDeadline));
        break;
      } catch (error) {
        if (deadlineExpired() || error === timeoutError) throw expireDeadline();
        if (isTrustedStructuredPhase(error, "capture")) throw error;
        if (
          retryablePersistenceError(error)
          && persistenceAttempt < STRUCTURED_PERSISTENCE_MAX_ATTEMPTS
        ) {
          continue;
        }
        throw structuredPhaseError("persistence", error);
      }
    }
    requireBeforeDeadline();
    const receiptAccepted = accepted?.accepted;
    requireBeforeDeadline();
    if (!accepted || receiptAccepted !== true) {
      throw structuredPhaseError("not_accepted");
    }
    resultPersisted = true;
    const returnedStructured = structuredClone(structured);
    requireBeforeDeadline();
    const response = Object.freeze({
      structured: returnedStructured,
      receipt: accepted,
      session_id: sessionId,
    });
    requireBeforeDeadline();
    return response;
  } finally {
    try {
      await usageObserver?.settle(resultPersisted ? "succeeded"
        : timedOut ? "timed_out" : signal?.aborted ? "cancelled" : "failed");
    } catch { /* missing accounting is exposed by the Host reservation count */ }
    try {
      if (pending !== null) {
        if (deadlineExpired()) expireDeadline();
        if (resultPersisted) {
          // Persistence is the durable completion boundary. DSH child-session
          // disposal is private resource cleanup and has been observed to
          // remain pending after the accepted result was already committed.
          // Never hold the HTTP response (and its Python worker) behind that
          // cleanup; remove admission bookkeeping immediately and dispose in
          // the background.
          detachCleanup(() => typeof pendingStarts.dispose === "function"
            ? pendingStarts.dispose(pending)
            : run?.dispose?.());
          detachCleanup(() => pendingStarts.finish?.(pending));
        } else if (timedOut) {
          if (run === undefined) {
            pending.promise.then(
              (lateRun) => {
                try {
                  Promise.resolve(structuredResult(lateRun)).catch(() => {});
                } catch {
                  // A late result accessor is observational cleanup only.
                }
                detachCleanup(() => typeof pendingStarts.dispose === "function"
                  ? pendingStarts.dispose(pending)
                  : lateRun?.dispose?.());
              },
              () => {},
            );
          } else {
            detachCleanup(() => typeof pendingStarts.dispose === "function"
              ? pendingStarts.dispose(pending)
              : run.dispose?.());
          }
          detachCleanup(() => pendingStarts.finish?.(pending));
        } else {
          try {
            if (run !== undefined) {
              await withinDeadline(() => typeof pendingStarts.dispose === "function"
                ? pendingStarts.dispose(pending)
                : run.dispose?.());
            }
          } finally {
            if (deadlineExpired()) {
              expireDeadline();
              detachCleanup(() => pendingStarts.finish?.(pending));
            } else {
              await withinDeadline(() => pendingStarts.finish?.(pending));
            }
          }
        }
      }
    } catch (error) {
      if (deadlineExpired() || error === timeoutError) throw expireDeadline();
      // Normal cleanup is awaited for quiescence, but its private failure must
      // not replace either a successful result or the primary phase outcome.
    } finally {
      if (timeout !== null) clearTimeout(timeout);
      signal?.removeEventListener("abort", abortFromCaller);
    }
    if (deadlineExpired()) throw expireDeadline();
  }
}

export { structuredResult };
