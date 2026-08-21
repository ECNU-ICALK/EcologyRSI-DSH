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
  let timedOut = false;
  let timeout = null;
  if (timeoutMs !== undefined) {
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1) {
      pending.controller.abort();
      pendingStarts.finish?.(pending);
      throw new Error("structured role timeout must be a positive integer");
    }
    timeout = setTimeout(() => {
      timedOut = true;
      pending.controller.abort();
    }, timeoutMs);
  }
  try {
    try {
      run = await pending.promise;
    } catch (error) {
      if (timedOut) {
        throw new Error("structured role operational timeout", { cause: error });
      }
      throw phaseError("structured_child_start_failed", error);
    }
    let result;
    try {
      result = await structuredResult(run);
    } catch (error) {
      throw phaseError("structured_child_result_failed", error);
    }
    const structured = result?.structured;
    if (!structured || typeof structured !== "object" || Array.isArray(structured)) {
      throw phaseError("structured_result_missing");
    }
    if (admission?.isOpen && !await admission.isOpen(reservedBinding)) {
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
      accepted = await persist({
        binding: reservedBinding,
        structured: structuredClone(structured),
        session_id: sessionId,
      });
    } catch (error) {
      throw phaseError("structured_result_persist_failed", error);
    }
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
    await run?.dispose?.();
    pendingStarts.finish?.(pending);
  }
}

export { structuredResult };
