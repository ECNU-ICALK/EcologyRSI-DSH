function boundedDelay(value, name) {
  if (!Number.isSafeInteger(value) || value < 0 || value > 3_600_000) {
    throw new Error(`${name} must be an integer between 0 and 3600000`);
  }
  return value;
}

function sleep(milliseconds, signal) {
  if (milliseconds <= 0) return Promise.resolve();
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    function onAbort() {
      clearTimeout(timer);
      resolve();
    }
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function admissionClosedError() {
  const error = new Error("provider stage admission is closed");
  error.code = "provider_stage_admission_closed";
  return error;
}

async function abortable(operation, signal) {
  if (signal.aborted) throw admissionClosedError();
  let onAbort;
  const aborted = new Promise((_resolve, reject) => {
    onAbort = () => reject(admissionClosedError());
    signal.addEventListener("abort", onAbort, { once: true });
  });
  try {
    return await Promise.race([operation, aborted]);
  } finally {
    signal.removeEventListener("abort", onAbort);
  }
}

/**
 * Provider-scoped admission for model-backed stages.
 *
 * Calls are serialized and spaced within one run/provider pair. Independent
 * evolution runs may therefore use the same provider concurrently, while a
 * single run still cannot make every sibling hit the same RPM window.
 */
export class ProviderStageGate {
  constructor({
    minimumIntervalMs = 30_000,
    failureCooldownMs = 30_000,
    now = Date.now,
    delay = sleep,
  } = {}) {
    this.minimumIntervalMs = boundedDelay(minimumIntervalMs, "minimumIntervalMs");
    this.failureCooldownMs = boundedDelay(failureCooldownMs, "failureCooldownMs");
    this.now = now;
    this.delay = delay;
    this.tails = new Map();
    this.nextAllowedAt = new Map();
    this.records = new Set();
  }

  async run(provider, operation, { runId = null } = {}) {
    const providerKey = String(provider || "default");
    const key = `${providerKey}\u0000${String(runId || "unscoped")}`;
    if (typeof operation !== "function") throw new Error("provider stage operation is required");
    const previous = this.tails.get(key) || Promise.resolve();
    const controller = new AbortController();
    const record = { runId, controller, promise: null };
    const current = abortable(previous.catch(() => {}), controller.signal).then(async () => {
      const wait = Math.max(
        0,
        (this.nextAllowedAt.get(key) || 0) - this.now(),
        (this.nextAllowedAt.get(providerKey) || 0) - this.now(),
      );
      if (wait > 0) {
        await abortable(
          Promise.resolve(this.delay(wait, controller.signal)),
          controller.signal,
        );
      }
      if (controller.signal.aborted) throw admissionClosedError();
      try {
        return await operation();
      } finally {
        this.nextAllowedAt.set(
          key,
          Math.max(
            this.nextAllowedAt.get(key) || 0,
            this.now() + this.minimumIntervalMs,
          ),
        );
      }
    });
    // Keep queue ordering independent from the caller-facing promise. A queued
    // run may be cancelled immediately, but its successor must still wait for
    // the predecessor that the cancelled run was originally queued behind.
    const barrier = Promise.allSettled([previous, current]).then(() => undefined);
    record.promise = current;
    this.records.add(record);
    this.tails.set(key, barrier);
    void barrier.then(() => {
      if (this.tails.get(key) === barrier) this.tails.delete(key);
    });
    try {
      return await current;
    } finally {
      this.records.delete(record);
    }
  }

  penalize(provider, milliseconds = this.failureCooldownMs) {
    const key = String(provider || "default");
    const cooldown = boundedDelay(milliseconds, "provider failure cooldown");
    this.nextAllowedAt.set(
      key,
      Math.max(this.nextAllowedAt.get(key) || 0, this.now() + cooldown),
    );
  }

  cancelRun(runId) {
    for (const record of this.records) {
      if (record.runId === runId) record.controller.abort();
    }
  }

  async drainRun(runId) {
    const selected = [...this.records].filter((record) => record.runId === runId);
    await Promise.allSettled(selected.map((record) => record.promise));
  }
}
