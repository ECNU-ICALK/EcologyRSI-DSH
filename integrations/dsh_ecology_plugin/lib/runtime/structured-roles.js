import {
  createStructuredDeadline,
  remainingStructuredDeadlineMs,
  structuredDeadlineExpired,
  validateStructuredDeadline,
  validateStructuredTimeoutMs,
} from "./structured-deadline.js";
import { structuredPhaseError } from "./structured-stage-errors.js";

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
    signal = null,
  } = {},
) {
  if (!roleHost?.agent) throw new Error("structured role requires a retained role-host Agent");
  if (!reservedBinding?.label) throw new Error("structured role requires a pre-registered child label");
  if (!request?.outputSchema || typeof request.outputSchema !== "object") {
    throw new Error("structured role requires an output schema");
  }
  if (
    request.maxTokens !== undefined
    && (!Number.isSafeInteger(request.maxTokens)
      || request.maxTokens < 512
      || request.maxTokens > 8192)
  ) {
    throw new Error("structured role maxTokens must be between 512 and 8192");
  }
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
        ...(request.maxTokens === undefined
          ? {}
          : { maxTokens: request.maxTokens }),
      }, {
        roleHostAgent: roleHost.agent,
        runId: reservedBinding?.launch?.run_id || reservedBinding?.binding?.run_id,
      });
      requireBeforeDeadline();
      run = await withinDeadline(() => pending.promise);
    } catch (error) {
      if (deadlineExpired() || error === timeoutError) throw expireDeadline();
      throw structuredPhaseError(
        error?.code === "provider_stage_admission_closed" ? "control" : "start",
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
    const structured = result?.structured;
    requireBeforeDeadline();
    const validStructured = structured
      && typeof structured === "object"
      && !Array.isArray(structured);
    const hasCaptureClassifier = typeof classifyMissingCapture === "function";
    let captureDisposition = null;
    if (!validStructured && stopReason === "error" && hasCaptureClassifier) {
      try {
        captureDisposition = classifyMissingCapture({ run, result });
      } catch {
        captureDisposition = "non-missing";
      }
    }
    requireBeforeDeadline();
    if (stopReason && stopReason !== "completed") {
      if (stopReason === "aborted") throw structuredPhaseError("aborted");
      if (captureDisposition === "missing") throw structuredPhaseError("capture");
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
      // Preserve the legacy message for callers that surface this safe state.
      error.message = "structured result admission is closed";
      throw error;
    }
    // rc.6 SubagentRun publishes the real child Session identity as `id`.
    // `childId` belongs to the continuable-start and Workflow event seams.
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
    try {
      accepted = await withinDeadline(() => persist({
        binding: reservedBinding,
        structured: persistedStructured,
        session_id: sessionId,
      }, persistenceDeadline));
    } catch (error) {
      if (deadlineExpired() || error === timeoutError) throw expireDeadline();
      throw structuredPhaseError("persistence", error);
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
