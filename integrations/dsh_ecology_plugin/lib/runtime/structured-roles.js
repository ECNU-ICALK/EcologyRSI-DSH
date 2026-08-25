function structuredResult(run) {
  if (typeof run?.result === "function") return run.result();
  if (run?.result && typeof run.result.then === "function") return run.result;
  if (run?.result !== undefined) return run.result;
  return run;
}

function phaseError(code, cause) {
  if (cause?.code) return cause;
  const error = new Error(code, cause ? { cause } : undefined);
  error.code = code;
  return error;
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
  { pendingStarts, admission, persist, timeoutMs } = {},
) {
  if (!roleHost?.agent) throw new Error("structured role requires a retained role-host Agent");
  if (!reservedBinding?.label) throw new Error("structured role requires a pre-registered child label");
  if (!request?.outputSchema || typeof request.outputSchema !== "object") {
    throw new Error("structured role requires an output schema");
  }
  if (!pendingStarts?.start || typeof persist !== "function") {
    throw new Error("structured role lifecycle services are required");
  }
  if (
    timeoutMs !== undefined
    && (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1)
  ) {
    throw new Error("structured role timeout must be a positive integer");
  }
  let pending = null;
  let run;
  let timeout = null;
  const deadlineAt = timeoutMs === undefined
    ? null
    : performance.now() + timeoutMs;
  const deadlineUnixMs = timeoutMs === undefined
    ? null
    : Date.now() + timeoutMs;
  let deadlinePromise = null;
  let timedOut = false;
  const timeoutError = operationalTimeoutError();
  const expireDeadline = () => {
    if (!timedOut) {
      timedOut = true;
      pending?.controller.abort();
    }
    return timeoutError;
  };
  const deadlineExpired = () => (
    deadlineAt !== null && (timedOut || performance.now() >= deadlineAt)
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
    return Math.max(1, Math.ceil(deadlineAt - performance.now()));
  };
  if (deadlineAt !== null) {
    deadlinePromise = new Promise((_resolve, reject) => {
      timeout = setTimeout(() => reject(expireDeadline()), timeoutMs);
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
      }, {
        roleHostAgent: roleHost.agent,
        runId: reservedBinding?.launch?.run_id || reservedBinding?.binding?.run_id,
      });
      requireBeforeDeadline();
      run = await withinDeadline(() => pending.promise);
    } catch (error) {
      if (deadlineExpired() || error === timeoutError) throw expireDeadline();
      throw phaseError("structured_child_start_failed", error);
    }
    let result;
    try {
      result = await withinDeadline(() => structuredResult(run));
    } catch (error) {
      if (deadlineExpired() || error === timeoutError) throw expireDeadline();
      throw phaseError("structured_child_result_failed", error);
    }
    requireBeforeDeadline();
    const stopReason = result?.stopReason;
    requireBeforeDeadline();
    if (stopReason && stopReason !== "completed") {
      throw phaseError(
        stopReason === "aborted"
          ? "structured_child_aborted"
          : "structured_child_model_error",
      );
    }
    const structured = result?.structured;
    requireBeforeDeadline();
    if (!structured || typeof structured !== "object" || Array.isArray(structured)) {
      throw phaseError("structured_result_missing");
    }
    let admissionOpen = true;
    if (admission?.isOpen) {
      admissionOpen = await withinDeadline(
        () => admission.isOpen(reservedBinding),
      );
    }
    requireBeforeDeadline();
    if (!admissionOpen) {
      const error = phaseError("structured_result_admission_closed");
      // Preserve the legacy message for callers that surface this safe state.
      error.message = "structured result admission is closed";
      throw error;
    }
    // rc.6 SubagentRun publishes the real child Session identity as `id`.
    // `childId` belongs to the continuable-start and Workflow event seams.
    const sessionId = String(run?.id || "");
    requireBeforeDeadline();
    if (!sessionId) throw phaseError("structured_child_session_missing");
    const persistedStructured = structuredClone(structured);
    requireBeforeDeadline();
    const persistenceDeadline = deadlineAt === null ? null : Object.freeze({
      deadlineUnixMs,
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
      throw phaseError("structured_result_persist_failed", error);
    }
    requireBeforeDeadline();
    const receiptAccepted = accepted?.accepted;
    requireBeforeDeadline();
    if (!accepted || receiptAccepted !== true) {
      throw phaseError("structured_result_not_accepted");
    }
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
        if (timedOut) {
          if (run === undefined) {
            pending.promise.then(
              (lateRun) => {
                try {
                  Promise.resolve(structuredResult(lateRun)).catch(() => {});
                } catch {
                  // A late result accessor is observational cleanup only.
                }
                detachCleanup(() => lateRun?.dispose?.());
              },
              () => {},
            );
          } else {
            detachCleanup(() => run.dispose?.());
          }
          detachCleanup(() => pendingStarts.finish?.(pending));
        } else {
          try {
            if (run !== undefined) {
              await withinDeadline(() => run.dispose?.());
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
      throw error;
    } finally {
      if (timeout !== null) clearTimeout(timeout);
    }
    if (deadlineExpired()) throw expireDeadline();
  }
}

export { structuredResult };
