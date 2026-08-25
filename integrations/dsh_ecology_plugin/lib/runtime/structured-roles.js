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
  const pending = pendingStarts.start("one-shot", {
    provider: "spawn",
    parent: roleHost.agent,
    label: reservedBinding.label,
    prompt: [{ type: "text", text: request.prompt }],
    outputSchema: structuredClone(request.outputSchema),
  }, {
    roleHostAgent: roleHost.agent,
    runId: reservedBinding?.launch?.run_id || reservedBinding?.binding?.run_id,
  });
  let run;
  let timeout = null;
  let deadlineAt = null;
  let deadlinePromise = null;
  let timedOut = false;
  const timeoutError = operationalTimeoutError();
  const expireDeadline = () => {
    if (!timedOut) {
      timedOut = true;
      pending.controller.abort();
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
    return await Promise.race([
      Promise.resolve().then(operation),
      deadlinePromise,
    ]);
  };
  if (timeoutMs !== undefined) {
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1) {
      pending.controller.abort();
      detachCleanup(() => pendingStarts.finish?.(pending));
      throw new Error("structured role timeout must be a positive integer");
    }
    deadlineAt = performance.now() + timeoutMs;
    deadlinePromise = new Promise((_resolve, reject) => {
      timeout = setTimeout(() => reject(expireDeadline()), timeoutMs);
    });
    // A synchronous phase failure may leave the deadline unraced until finally.
    deadlinePromise.catch(() => {});
  }
  try {
    try {
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
    if (result?.stopReason && result.stopReason !== "completed") {
      throw phaseError(
        result.stopReason === "aborted"
          ? "structured_child_aborted"
          : "structured_child_model_error",
      );
    }
    const structured = result?.structured;
    if (!structured || typeof structured !== "object" || Array.isArray(structured)) {
      throw phaseError("structured_result_missing");
    }
    if (admission?.isOpen && !await withinDeadline(
      () => admission.isOpen(reservedBinding),
    )) {
      const error = phaseError("structured_result_admission_closed");
      // Preserve the legacy message for callers that surface this safe state.
      error.message = "structured result admission is closed";
      throw error;
    }
    // rc.6 SubagentRun publishes the real child Session identity as `id`.
    // `childId` belongs to the continuable-start and Workflow event seams.
    const sessionId = String(run?.id || "");
    if (!sessionId) throw phaseError("structured_child_session_missing");
    let accepted;
    try {
      accepted = await withinDeadline(() => persist({
        binding: reservedBinding,
        structured: structuredClone(structured),
        session_id: sessionId,
      }));
    } catch (error) {
      if (deadlineExpired() || error === timeoutError) throw expireDeadline();
      throw phaseError("structured_result_persist_failed", error);
    }
    requireBeforeDeadline();
    if (!accepted || accepted.accepted !== true) {
      throw phaseError("structured_result_not_accepted");
    }
    return Object.freeze({
      structured: structuredClone(structured),
      receipt: accepted,
      session_id: sessionId,
    });
  } finally {
    if (timeout !== null) clearTimeout(timeout);
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
  }
}

export { structuredResult };
