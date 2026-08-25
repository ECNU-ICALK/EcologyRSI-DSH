export const MAX_STRUCTURED_STAGE_TIMEOUT_MS = 1_800_000;

export function validateStructuredTimeoutMs(value, name = "structured role timeout") {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  if (value > MAX_STRUCTURED_STAGE_TIMEOUT_MS) {
    throw new Error(`${name} must be at most ${MAX_STRUCTURED_STAGE_TIMEOUT_MS}`);
  }
  return value;
}

export function createStructuredDeadline(timeoutMs) {
  const validated = validateStructuredTimeoutMs(timeoutMs);
  return Object.freeze({
    timeoutMs: validated,
    deadlineAt: performance.now() + validated,
  });
}

export function validateStructuredDeadline(deadline) {
  if (
    !deadline
    || typeof deadline !== "object"
    || !Number.isFinite(deadline.deadlineAt)
    || deadline.deadlineAt < 0
    || !Number.isSafeInteger(deadline.timeoutMs)
  ) {
    throw new Error("structured role deadline is invalid");
  }
  validateStructuredTimeoutMs(deadline.timeoutMs);
  return deadline;
}

export function structuredDeadlineExpired(deadline) {
  return performance.now() >= validateStructuredDeadline(deadline).deadlineAt;
}

export function remainingStructuredDeadlineMs(deadline) {
  const checked = validateStructuredDeadline(deadline);
  return Math.max(0, Math.ceil(checked.deadlineAt - performance.now()));
}
