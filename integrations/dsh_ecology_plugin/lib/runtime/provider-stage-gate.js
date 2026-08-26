export const MAX_STRUCTURED_STAGE_IN_FLIGHT = 128;

function boundedDelay(value, name) {
  if (!Number.isSafeInteger(value) || value < 0 || value > 3_600_000) {
    throw new Error(`${name} must be an integer between 0 and 3600000`);
  }
  return value;
}

function boundedConcurrency(value) {
  if (
    !Number.isSafeInteger(value)
    || value < 1
    || value > MAX_STRUCTURED_STAGE_IN_FLIGHT
  ) {
    throw new Error(
      `maxInFlight must be between 1 and ${MAX_STRUCTURED_STAGE_IN_FLIGHT}`,
    );
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
 * Provider-wide FIFO admission for model-backed stages.
 *
 * The concurrency bound is physical, not multiplied by candidate or run
 * count. Optional RPM spacing is measured between starts. Successful
 * completions never create a new cooldown; only an explicit penalty does.
 */
export class ProviderStageGate {
  constructor({
    minimumIntervalMs = 0,
    failureCooldownMs = 30_000,
    maxInFlight = MAX_STRUCTURED_STAGE_IN_FLIGHT,
    now = Date.now,
    delay = sleep,
  } = {}) {
    this.minimumIntervalMs = boundedDelay(minimumIntervalMs, "minimumIntervalMs");
    this.failureCooldownMs = boundedDelay(failureCooldownMs, "failureCooldownMs");
    this.maxInFlight = boundedConcurrency(maxInFlight);
    this.now = now;
    this.delay = delay;
    this.queues = new Map();
    this.active = new Map();
    this.nextAllowedAt = new Map();
    this.pumping = new Set();
    this.records = new Set();
    this.closedRuns = new Set();
  }

  async run(provider, operation, { runId = null } = {}) {
    const providerKey = String(provider || "default");
    if (typeof operation !== "function") throw new Error("provider stage operation is required");
    this.assertRunOpen(runId);
    const controller = new AbortController();
    let resolveCaller;
    let rejectCaller;
    const caller = new Promise((resolve, reject) => {
      resolveCaller = resolve;
      rejectCaller = reject;
    });
    const record = {
      runId,
      providerKey,
      operation,
      controller,
      resolveCaller,
      rejectCaller,
      promise: caller,
      completion: null,
      status: "queued",
    };
    this.records.add(record);
    const queue = this.queues.get(providerKey) || [];
    queue.push(record);
    this.queues.set(providerKey, queue);
    void this.pump(providerKey);
    return await caller;
  }

  async pump(providerKey) {
    if (this.pumping.has(providerKey)) return;
    this.pumping.add(providerKey);
    try {
      while ((this.active.get(providerKey) || 0) < this.maxInFlight) {
        const queue = this.queues.get(providerKey);
        if (!queue?.length) {
          this.queues.delete(providerKey);
          return;
        }
        const record = queue[0];
        if (record.controller.signal.aborted || this.closedRuns.has(record.runId)) {
          queue.shift();
          record.status = "cancelled";
          record.rejectCaller(admissionClosedError());
          this.records.delete(record);
          continue;
        }
        const wait = Math.max(
          0,
          (this.nextAllowedAt.get(providerKey) || 0) - this.now(),
        );
        if (wait > 0) {
          try {
            await abortable(
              Promise.resolve(this.delay(wait, record.controller.signal)),
              record.controller.signal,
            );
          } catch (error) {
            queue.shift();
            record.status = "cancelled";
            record.rejectCaller(error);
            this.records.delete(record);
            continue;
          }
        }
        if (record.controller.signal.aborted || this.closedRuns.has(record.runId)) {
          continue;
        }
        queue.shift();
        record.status = "active";
        this.active.set(providerKey, (this.active.get(providerKey) || 0) + 1);
        this.nextAllowedAt.set(
          providerKey,
          Math.max(
            this.nextAllowedAt.get(providerKey) || 0,
            this.now() + this.minimumIntervalMs,
          ),
        );
        const operationPromise = Promise.resolve().then(record.operation);
        record.completion = operationPromise;
        void abortable(operationPromise, record.controller.signal).then(
          record.resolveCaller,
          record.rejectCaller,
        );
        const release = () => {
          record.status = "completed";
          this.records.delete(record);
          const remaining = Math.max(0, (this.active.get(providerKey) || 1) - 1);
          if (remaining === 0) this.active.delete(providerKey);
          else this.active.set(providerKey, remaining);
          void this.pump(providerKey);
        };
        void operationPromise.then(release, release);
      }
    } finally {
      this.pumping.delete(providerKey);
      if (
        (this.queues.get(providerKey)?.length || 0) > 0
        && (this.active.get(providerKey) || 0) < this.maxInFlight
      ) {
        queueMicrotask(() => { void this.pump(providerKey); });
      }
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
    for (const providerKey of this.queues.keys()) void this.pump(providerKey);
  }

  closeRun(runId) {
    this.closedRuns.add(runId);
    this.cancelRun(runId);
  }

  openRun(runId) {
    this.closedRuns.delete(runId);
  }

  assertRunOpen(runId) {
    if (this.closedRuns.has(runId)) throw admissionClosedError();
  }

  async drainRun(runId) {
    while (true) {
      const selected = [...this.records].filter((record) => record.runId === runId);
      if (selected.length === 0) return;
      await Promise.allSettled(
        selected.map((record) => record.completion || record.promise),
      );
    }
  }
}
