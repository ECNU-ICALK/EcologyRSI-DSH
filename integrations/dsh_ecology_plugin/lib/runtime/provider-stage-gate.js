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

function queueTimeoutError() {
  const error = new Error("provider stage queue deadline exceeded");
  error.code = "provider_queue_timeout";
  return error;
}

function deadlineRemainingMs(deadline, now = Date.now) {
  if (deadline == null) return Infinity;
  if (Number.isSafeInteger(deadline)) return Math.max(0, deadline);
  if (typeof deadline.remainingTimeoutMs === "function") {
    return Math.max(0, Number(deadline.remainingTimeoutMs()) || 0);
  }
  if (typeof deadline !== "object" || !Number.isFinite(deadline.deadlineAt)) return Infinity;
  // structured deadlines use performance.now(); wall-clock deadlines use now().
  const clock = deadline.deadlineAt < 100_000_000_000 ? performance.now() : now();
  return Math.max(0, Math.ceil(deadline.deadlineAt - clock));
}

async function abortable(operation, signal) {
  if (signal.aborted) throw signal.reason || admissionClosedError();
  let onAbort;
  const aborted = new Promise((_resolve, reject) => {
    onAbort = () => reject(signal.reason || admissionClosedError());
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

  async run(provider, operation, {
    runId = null,
    deadline = null,
    signal = null,
  } = {}) {
    const providerKey = String(provider || "default");
    if (typeof operation !== "function") throw new Error("provider stage operation is required");
    this.assertRunOpen(runId);
    const controller = new AbortController();
    const externalAbort = () => controller.abort(signal.reason || admissionClosedError());
    if (signal) {
      if (signal.aborted) externalAbort();
      else signal.addEventListener("abort", externalAbort, { once: true });
    }
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
      deadline,
      queueTimer: null,
      deadlineTimer: null,
    };
    const remaining = deadlineRemainingMs(deadline, this.now);
    if (remaining <= 0) {
      record.status = "cancelled";
      rejectCaller(queueTimeoutError());
      if (signal) signal.removeEventListener("abort", externalAbort);
      return await caller;
    }
    if (Number.isFinite(remaining)) {
      record.queueTimer = setTimeout(() => this.#expireQueued(record), remaining);
    }
    this.records.add(record);
    const queue = this.queues.get(providerKey) || [];
    queue.push(record);
    this.queues.set(providerKey, queue);
    void this.pump(providerKey);
    try {
      return await caller;
    } finally {
      if (signal) signal.removeEventListener("abort", externalAbort);
    }
  }

  #expireQueued(record) {
    if (record.status !== "queued") return;
    const queue = this.queues.get(record.providerKey);
    const index = queue?.indexOf(record) ?? -1;
    if (index >= 0) queue.splice(index, 1);
    record.status = "cancelled";
    record.controller.abort(queueTimeoutError());
    record.rejectCaller(queueTimeoutError());
    this.records.delete(record);
    if (queue && queue.length === 0) this.queues.delete(record.providerKey);
    void this.pump(record.providerKey);
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
          if (queue[0] === record) queue.shift();
          if (record.queueTimer !== null) clearTimeout(record.queueTimer);
          record.status = "cancelled";
          record.rejectCaller(record.controller.signal.reason || admissionClosedError());
          this.records.delete(record);
          continue;
        }
        const remaining = deadlineRemainingMs(record.deadline, this.now);
        if (remaining <= 0) {
          if (queue[0] === record) queue.shift();
          this.#expireQueued(record);
          continue;
        }
        const wait = Math.max(
          0,
          (this.nextAllowedAt.get(providerKey) || 0) - this.now(),
        );
        if (wait > 0) {
          try {
            await abortable(
              Promise.race([
                Promise.resolve(this.delay(wait, record.controller.signal)),
                Number.isFinite(remaining)
                  ? new Promise((resolve) => setTimeout(resolve, remaining))
                  : new Promise(() => {}),
              ]),
              record.controller.signal,
            );
          } catch (error) {
            if (queue[0] === record) queue.shift();
            if (record.queueTimer !== null) clearTimeout(record.queueTimer);
            record.status = "cancelled";
            record.rejectCaller(error);
            this.records.delete(record);
            continue;
          }
        }
        if (record.controller.signal.aborted || this.closedRuns.has(record.runId)) {
          continue;
        }
        if (deadlineRemainingMs(record.deadline, this.now) <= 0) {
          if (queue[0] === record) queue.shift();
          this.#expireQueued(record);
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
        if (record.queueTimer !== null) clearTimeout(record.queueTimer);
        const activeRemaining = deadlineRemainingMs(record.deadline, this.now);
        if (Number.isFinite(activeRemaining)) {
          record.deadlineTimer = setTimeout(() => {
            if (record.status === "active") {
              record.status = "draining";
              record.controller.abort(queueTimeoutError());
            }
          }, activeRemaining);
        }
        const operationPromise = Promise.resolve().then(() => record.operation(record.controller.signal));
        record.completion = operationPromise;
        void abortable(operationPromise, record.controller.signal).then(
          record.resolveCaller,
          record.rejectCaller,
        );
        const release = () => {
          if (record.deadlineTimer !== null) clearTimeout(record.deadlineTimer);
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
      if (record.runId === runId) record.controller.abort(admissionClosedError());
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

  async drainRun(runId, deadline = null) {
    const timeoutMs = deadline == null
      ? 30_000
      : Number.isSafeInteger(deadline)
        ? deadline
        : Number.isSafeInteger(deadline?.timeoutMs)
          ? deadline.timeoutMs
          : deadlineRemainingMs(deadline, this.now);
    const bounded = Math.max(0, Math.min(3_600_000, timeoutMs));
    const startedAt = this.now();
    while (true) {
      const selected = [...this.records].filter((record) => record.runId === runId);
      if (selected.length === 0) return { status: "drained", remaining: 0 };
      const remainingMs = Math.max(0, bounded - (this.now() - startedAt));
      if (remainingMs <= 0) {
        return { status: "deadline_exceeded", remaining: selected.length };
      }
      let timer;
      const timeout = new Promise((resolve) => {
        timer = setTimeout(() => resolve(false), remainingMs);
      });
      const settled = Promise.allSettled(selected.map((record) => record.completion || record.promise))
        .then(() => true);
      const done = await Promise.race([settled, timeout]);
      clearTimeout(timer);
      if (!done) {
        return {
          status: "deadline_exceeded",
          remaining: [...this.records].filter((record) => record.runId === runId).length,
        };
      }
    }
  }
}
