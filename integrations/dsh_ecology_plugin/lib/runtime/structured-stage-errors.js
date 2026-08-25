const PHASE_CODES = Object.freeze({
  start: "structured_child_start_failed",
  result: "structured_child_result_failed",
  control: "provider_stage_admission_closed",
  aborted: "structured_child_aborted",
  model: "structured_child_model_error",
  capture: "structured_result_missing",
  admission: "structured_result_admission_failed",
  admission_closed: "structured_result_admission_closed",
  child_session: "structured_child_session_missing",
  persistence: "structured_result_persist_failed",
  not_accepted: "structured_result_not_accepted",
});

const trustedStructuredErrors = new WeakMap();

export function structuredPhaseError(phase, cause = null) {
  const code = PHASE_CODES[phase];
  if (!code) throw new Error("unknown structured stage error phase");
  const error = new Error(code);
  error.code = code;
  trustedStructuredErrors.set(error, { phase, cause });
  return error;
}

export function isTrustedStructuredPhase(error, phase) {
  return trustedStructuredErrors.get(error)?.phase === phase;
}
