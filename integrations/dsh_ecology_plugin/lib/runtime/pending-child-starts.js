function launchAdmissionClosedError() {
  const error = new Error("provider stage admission is closed");
  error.code = "provider_stage_admission_closed";
  return error;
}

export class PendingChildStarts {
  constructor(ctx, { launchFence = null } = {}) {
    this.ctx = ctx;
    this.launchFence = launchFence;
    this.pending = new Set();
    this.closedRuns = new Set();
  }

  get size() { return this.pending.size; }

  finish(record) {
    if (!record) return;
    if (!record.finalized) {
      record.finalized = true;
      record.resolveFinalization();
    }
    this.pending.delete(record);
  }

  hasRun(runId) {
    return [...this.pending].some(
      (record) => runId === undefined || record.binding.runId === runId,
    );
  }

  closeRun(runId) { this.closedRuns.add(runId); }

  openRun(runId) { this.closedRuns.delete(runId); }

  // See ProviderStageGate.forgetRun: terminal runs cannot start another child, so
  // their closed-run marker is pure growth.
  forgetRun(runId) { this.closedRuns.delete(runId); }

  assertRunOpen(runId) {
    if (this.closedRuns.has(runId)) throw launchAdmissionClosedError();
    this.launchFence?.assertRunOpen?.(runId);
  }

  #record(kind, binding, operation) {
    this.assertRunOpen(binding.runId);
    const controller = new AbortController();
    let resolveStart;
    let rejectStart;
    const start = new Promise((resolve, reject) => {
      resolveStart = resolve;
      rejectStart = reject;
    });
    let resolveFinalization;
    const finalization = new Promise((resolve) => { resolveFinalization = resolve; });
    const record = {
      kind,
      binding,
      controller,
      signal: controller.signal,
      result: null,
      error: null,
      promise: null,
      finalization,
      resolveFinalization,
      finalized: false,
      quiescence: null,
      disposePromise: null,
    };
    record.promise = start.then(
      (result) => { record.result = result; return result; },
      (error) => { record.error = error; throw error; },
    );
    record.promise.catch(() => {});
    this.pending.add(record);
    try {
      const value = operation(controller.signal);
      resolveStart(value);
    } catch (error) {
      record.error = error;
      rejectStart(error);
    }
    return record;
  }

  start(kind, request, binding = {}) {
    if (!new Set(["one-shot", "continuable"]).has(kind)) throw new Error("unsupported child start kind");
    return this.#record(kind, binding, (signal) => {
      if (kind === "one-shot") {
        const { provider, ...childRequest } = request;
        return this.ctx.subagents.start(provider, {
          ...childRequest,
          signal,
        });
      }
      return this.ctx.subagents.startContinuable({
        ...request,
        signal,
      });
    });
  }

  dispose(record) {
    if (!record?.result?.dispose) return Promise.resolve();
    return this.#memoizedCleanup(
      record,
      "disposePromise",
      () => record.result.dispose(),
    );
  }

  #memoizedCleanup(record, field, operation) {
    if (record[field] !== null) return record[field];
    let resolveCleanup;
    let rejectCleanup;
    const inFlight = new Promise((resolve, reject) => {
      resolveCleanup = resolve;
      rejectCleanup = reject;
    });
    inFlight.catch(() => {});
    record[field] = inFlight;
    try {
      const external = operation();
      if (external === inFlight) {
        resolveCleanup();
      } else {
        Promise.resolve(external).then(resolveCleanup, rejectCleanup);
      }
    } catch (error) {
      rejectCleanup(error);
    }
    return inFlight;
  }

  quiesce(record) {
    if (!record) return Promise.resolve();
    if (record.quiescence !== null) return record.quiescence;
    record.controller.abort();
    record.quiescence = this.#quiesceRecord(record);
    record.quiescence.catch(() => {});
    return record.quiescence;
  }

  async #settledOrFinalized(record, operation) {
    if (record.finalized) return { kind: "finalized" };
    const observed = Promise.resolve(operation).then(
      (value) => ({ kind: "settled", status: "fulfilled", value }),
      (error) => ({ kind: "settled", status: "rejected", error }),
    );
    return await Promise.race([
      observed,
      record.finalization.then(() => ({ kind: "finalized" })),
    ]);
  }

  async #quiesceRecord(record) {
    try {
      const started = await this.#settledOrFinalized(record, record.promise);
      if (started.kind === "finalized") return;
      if (started.status === "rejected") return;

      if (record.kind === "one-shot") {
        await this.#settledOrFinalized(record, this.dispose(record));
        return;
      }

      if (record.kind === "continuable") {
        if (record.result?.childId) {
          let interruption;
          try {
            interruption = this.ctx.subagents.interrupt(record.result.childId, {
              kind: "ancestor",
              agent: record.binding.roleHostAgent,
            });
          } catch (error) {
            interruption = Promise.reject(error);
          }
          const interrupted = await this.#settledOrFinalized(record, interruption);
          if (interrupted.kind === "finalized") return;
        }
        if (record.binding.roleHostAgent) {
          let descendants;
          try {
            descendants = this.ctx.subagents.drainContinuableDescendants([
              record.binding.roleHostAgent,
            ]);
          } catch (error) {
            descendants = Promise.reject(error);
          }
          await this.#settledOrFinalized(record, descendants);
        }
        return;
      }

    } finally {
      this.finish(record);
    }
  }

  async cancelAndQuiesce({ runId } = {}) {
    if (runId !== undefined) this.closeRun(runId);
    while (true) {
      const records = [...this.pending].filter(
        (record) => runId === undefined || record.binding.runId === runId,
      );
      if (records.length === 0) return;
      await Promise.allSettled(records.map((record) => this.quiesce(record)));
    }
  }
}
