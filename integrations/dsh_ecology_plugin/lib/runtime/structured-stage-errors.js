const PHASE_CODES = Object.freeze({
  start: "structured_child_start_failed",
  output_schema: "structured_child_output_schema_invalid",
  result: "structured_child_result_failed",
  control: "provider_stage_admission_closed",
  aborted: "structured_child_aborted",
  model: "structured_child_model_error",
  model_terminal: "structured_child_model_error",
  route_cooldown: "provider_route_cooling_down",
  tool_protocol: "structured_child_tool_protocol_error",
  output_budget: "structured_child_output_budget_exhausted",
  execution_budget: "structured_child_execution_budget_exhausted",
  capture: "structured_result_missing",
  admission: "structured_result_admission_failed",
  admission_closed: "structured_result_admission_closed",
  child_session: "structured_child_session_missing",
  persistence: "structured_result_persist_failed",
  not_accepted: "structured_result_not_accepted",
});

const trustedStructuredErrors = new WeakMap();

function persistencePublicDetail(cause) {
  if (cause?.code === "dsh_session_projection_not_ready") {
    return "dsh_session_projection_not_ready";
  }
  // SidecarClient only retains this field after the Python boundary has
  // applied its credential-redacting public error policy. Preserve that
  // bounded diagnostic in the local service log while the HTTP response and
  // durable failure event continue to expose only the stable phase code.
  if (cause?.name === "SidecarError" && typeof cause.publicDetail === "string") {
    const normalized = cause.publicDetail
      .replace(/[\u0000-\u001f\u007f]/g, " ")
      .trim();
    if (normalized) return normalized.slice(0, 256);
  }
  if (cause?.name === "SidecarError" && typeof cause.code === "string") {
    return `sidecar_${cause.code}`.slice(0, 256);
  }
  return null;
}

export function structuredPhaseError(phase, cause = null, metadata = null) {
  const code = PHASE_CODES[phase];
  if (!code) throw new Error("unknown structured stage error phase");
  const error = new Error(code);
  error.code = code;
  const publicDetail = phase === "persistence"
    ? persistencePublicDetail(cause)
    : null;
  if (publicDetail !== null) error.publicDetail = publicDetail;
  trustedStructuredErrors.set(error, { phase, cause, metadata });
  return error;
}

export function isTrustedStructuredPhase(error, phase) {
  return trustedStructuredErrors.get(error)?.phase === phase;
}

export function structuredRetryAfterMs(error) {
  const value = trustedStructuredErrors.get(error)?.metadata?.retryAfterMs;
  return Number.isSafeInteger(value) && value > 0 && value <= 3_600_000
    ? value
    : null;
}

export function isStructuredProviderRateLimit(error) {
  return trustedStructuredErrors.get(error)?.metadata?.providerRateLimit === true;
}

// Only errors created at the trusted child boundary can supply HTTP semantics.
// Never forward provider messages or accept metadata attached by a tool.
export function structuredFailureContract(error) {
  const trusted = trustedStructuredErrors.get(error);
  if (!trusted || !["model", "model_terminal", "route_cooldown", "tool_protocol",
    "output_budget", "execution_budget", "output_schema", "capture"].includes(trusted.phase)) return null;
  // A malformed tool envelope is a failed provider attempt, not a scientific
  // result. Mixed upstream replicas can alternate valid calls and DSML prose;
  // allow bounded Host recovery without accepting or executing that prose.
  const retryable = ["model", "route_cooldown", "tool_protocol"].includes(trusted.phase);
  const supplied = trusted.metadata?.providerStatus;
  const providerStatus = Number.isSafeInteger(supplied) && supplied >= 400 && supplied <= 599
    ? supplied : null;
  return {
    schema_version: "ecology-runtime-failure/1",
    error_code: structuredFailureCode(error),
    failure_domain: retryable ? "provider" : "execution_contract",
    retryable,
    provider_status: providerStatus,
    retry_after_ms: structuredRetryAfterMs(error),
    affected_scope: retryable ? "model_route" : "stage",
    http_status: retryable ? (providerStatus === 429 ? 429 : 503) : 422,
  };
}

const PERSISTENCE_BOUNDARIES = Object.freeze({
  dsh_session_projection_not_ready: "structured_result_persist_projection_lag",
  structured_result_persistence_unavailable: "structured_result_persist_host_unavailable",
  unavailable: "structured_result_persist_transport",
  timeout: "structured_result_persist_timeout",
  dsh_tool_admission_closed: "structured_result_persist_admission_closed",
  dsh_tool_authorization_failed: "structured_result_persist_authorization",
  dsh_prediction_binding_closed: "structured_result_persist_binding_closed",
  sidecar_rejected: "structured_result_persist_host_rejected",
  invalid_response: "structured_result_persist_invalid_response",
  dsh_skill_evidence_invalid: "structured_result_persist_tool_evidence",
});

// Only trusted phase causes map to public machine codes. Never copy an error
// message, request payload, arbitrary cause.code or credentials to the ledger.
export function structuredFailureCode(error) {
  const trusted = trustedStructuredErrors.get(error);
  return trusted?.phase === "persistence"
    ? PERSISTENCE_BOUNDARIES[trusted.cause?.code] || PHASE_CODES.persistence
    : error?.code;
}
