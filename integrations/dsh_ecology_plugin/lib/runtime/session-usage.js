import { dshSessionMetrics } from "./agents.js";

export function sessionUsageComplete(events, { settlement, freshUsage } = {}) {
  const messages = Array.isArray(events) ? events.filter(e => e?.type === "assistant/message") : [];
  const endings = Array.isArray(events) ? events.filter(e => e?.type === "turn/end") : [];
  return !["active", "cancelled", "timed_out"].includes(settlement) && freshUsage === true && messages.length > 0
    && messages.every(e => e.data?.usage && typeof e.data.usage === "object")
    && endings.length > 0 && endings.every(e => e.data?.reason?.kind === "completed");
}

// Observe the public usage projection, never assistant text or reasoning.
// Accounting is independent of schema validity and result admission.
export function observeSessionUsage(ctx, sidecar, identity, { intervalMs = 5000 } = {}) {
  let queue = Promise.resolve();
  let lastAvailable = null;
  let lastAcknowledged = null;
  let stopped = false;
  const capture = (settlement) => {
    let metrics = dshSessionMetrics(ctx, identity.session_id);
    const freshUsage = metrics.provider_usage.available === true;
    if (freshUsage) lastAvailable = metrics;
    else if (lastAvailable) metrics = lastAvailable;
    const events = ctx?.sessions?.get?.(identity.session_id)?.events;
    const complete = sessionUsageComplete(events, { settlement, freshUsage });
    const body = {
      schema_version: "ecologyrsi-dsh.session-usage/1", identity: { ...identity },
      session_metrics: structuredClone(metrics), settlement, usage_complete: complete,
    };
    // Pressure changes alone do not need another ledger event.
    const key = JSON.stringify([body.session_metrics.provider_usage, settlement, complete]);
    queue = queue.then(async () => {
      if (key === lastAcknowledged) return;
      for (let attempt = 0; attempt < 2; attempt += 1) {
        try {
          const receipt = await sidecar.request("/api/ecology-agent-sidecar/v1/session-usage", {
            method: "POST", timeoutMs: 2000, body,
          });
          if (receipt?.accepted !== true) throw new Error("usage was not accepted");
          lastAcknowledged = key;
          return;
        } catch {
          // An exact replay or later cumulative snapshot recovers transient
          // delivery errors. Missing records remain visible in Host coverage.
        }
      }
    }).catch(() => {});
    return queue;
  };
  capture("active");
  const timer = setInterval(() => { if (!stopped) capture("active"); }, intervalMs);
  timer.unref?.();
  return {
    async settle(settlement) {
      if (stopped) return;
      stopped = true;
      clearInterval(timer);
      await capture(settlement);
    },
  };
}
