import { isTrustedStructuredPhase, structuredPhaseError, structuredRetryAfterMs } from "./structured-stage-errors.js";

// Route health is separate from provider-wide physical/RPM admission. A sick
// model must not close other models, and queued work must recheck after admission.
export class RouteHealth {
  constructor({ now = Date.now, cooldownMs = 60_000 } = {}) {
    this.now = now;
    this.cooldownMs = cooldownMs;
    this.routes = new Map();
  }

  snapshot(route) {
    const state = this.routes.get(route);
    return {
      state: state?.state || "closed",
      consecutive_failures: state?.consecutive || 0,
      retry_after_ms: Math.max(0, (state?.until || 0) - this.now()),
    };
  }

  async run(route, operation) {
    let state = this.routes.get(route);
    if (!state) {
      state = { state: "closed", until: 0, consecutive: 0, recent: [], epoch: 0 };
      this.routes.set(route, state);
    }
    if (state.state === "half_open" || (state.state === "open" && this.now() < state.until)) {
      throw structuredPhaseError("route_cooldown", null, {
        providerStatus: 503, retryAfterMs: Math.max(1, state.until - this.now()),
      });
    }
    const probe = state.state === "open";
    if (probe) state.state = "half_open";
    const epoch = state.epoch;
    try {
      const result = await operation();
      // Late successes from the pre-outage wave cannot close an open circuit.
      if (probe && epoch === state.epoch) {
        Object.assign(state, { state: "closed", consecutive: 0, recent: [], until: 0 });
      } else if (state.state === "closed" && epoch === state.epoch) {
        state.consecutive = 0;
        state.recent.push(false);
        state.recent = state.recent.slice(-10);
      }
      return result;
    } catch (error) {
      const serviceFailure = isTrustedStructuredPhase(error, "model")
        || isTrustedStructuredPhase(error, "tool_protocol");
      if (serviceFailure) {
        state.consecutive += 1;
        state.recent.push(true);
        state.recent = state.recent.slice(-10);
      }
      if (probe || (serviceFailure && (state.consecutive >= 3
          || (state.recent.length >= 10 && state.recent.filter(Boolean).length >= 3)))) {
        state.state = "open";
        state.epoch += 1;
        state.until = Math.max(state.until, this.now() + Math.max(
          this.cooldownMs, structuredRetryAfterMs(error) || 0,
        ));
      }
      throw error;
    }
  }
}
