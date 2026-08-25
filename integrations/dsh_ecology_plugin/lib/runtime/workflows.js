const WORKFLOW_SCRIPTS = Object.freeze({
  "ecology-one-shot-v1": `
const results = [];
for (let offset = 0; offset < args.items.length; offset += args.maxConcurrent) {
  const batch = args.items.slice(offset, offset + args.maxConcurrent);
  const values = await parallel(batch.map((item) => () => agent(item.prompt, {
    label: item.label,
    schema: item.schema,
  })));
  results.push(...values);
}
return results;
`.trim(),
});

function positiveInteger(value, name, fallback) {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved < 1) throw new Error(`${name} must be a positive integer`);
  return resolved;
}

function launchAdmissionClosedError() {
  const error = new Error("provider stage admission is closed");
  error.code = "provider_stage_admission_closed";
  return error;
}

export function startHomogeneousWorkflow(roleHost, compiledSpec, items) {
  const script = WORKFLOW_SCRIPTS[compiledSpec.template_id];
  if (!script) throw new Error("unknown fixed workflow template");
  const maxItems = positiveInteger(compiledSpec.max_items, "max_items", 32);
  const maxTotalAgents = positiveInteger(compiledSpec.max_total_agents, "max_total_agents", maxItems);
  const maxConcurrent = positiveInteger(compiledSpec.max_concurrent, "max_concurrent", Math.min(4, maxTotalAgents));
  const syncTimeoutMs = positiveInteger(compiledSpec.sync_timeout_ms, "sync_timeout_ms", 120000);
  if (!Array.isArray(items) || items.length > maxItems || items.length > maxTotalAgents) {
    throw new Error("workflow items exceed bounded limits");
  }
  for (const item of items) {
    if (
      !item || typeof item !== "object" || Array.isArray(item)
      || typeof item.label !== "string" || !item.label
      || typeof item.prompt !== "string" || !item.prompt
      || !item.schema || typeof item.schema !== "object" || Array.isArray(item.schema)
    ) throw new Error("workflow item is outside the structured contract");
  }
  const encoded = JSON.stringify(items);
  if (Buffer.byteLength(encoded, "utf8") > 256 * 1024) throw new Error("workflow structured args are too large");
  const engine = roleHost?.services?.workflowEngine;
  if (!engine?.start) throw new Error("role-host Workflow service is unavailable");
  if (!roleHost?.agent) throw new Error("workflow requires a retained role-host Agent");
  const workflowName = compiledSpec.workflow_name ?? compiledSpec.template_id;
  if (typeof workflowName !== "string" || !/^[a-z0-9][a-z0-9-]*$/.test(workflowName)) {
    throw new Error("workflow_name must be a normalized kebab-case identifier");
  }
  return engine.start({
    script,
    meta: {
      name: workflowName,
      description: "Execute one bounded EcologyRSI structured agent wave.",
    },
    args: {
      items: structuredClone(items),
      maxConcurrent,
      operationalTimeoutMs: syncTimeoutMs,
    },
    maxTotalAgents,
    parent: roleHost.agent,
  });
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
      cancelPromise: null,
      disposePromise: null,
      workflowResultPromise: null,
    };
    record.promise = start.then(
      (result) => { record.result = result; return result; },
      (error) => { record.error = error; throw error; },
    );
    record.promise.catch(() => {});
    this.pending.add(record);
    try {
      const value = operation(controller.signal);
      if (kind === "workflow") record.result = value;
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

  startWorkflow(operation, binding = {}) {
    if (typeof operation !== "function") throw new Error("workflow start operation is required");
    const record = this.#record("workflow", binding, operation);
    if (record.error) {
      this.finish(record);
      throw record.error;
    }
    if (!record.result || typeof record.result !== "object" || typeof record.result.then === "function") {
      this.finish(record);
      throw new Error("workflow start must return a synchronous run handle");
    }
    return record;
  }

  cancel(record, reason = "run quiescing") {
    if (!record) return Promise.resolve();
    record.controller.abort();
    if (record.kind !== "workflow" || !record.result?.cancel) return Promise.resolve();
    if (record.cancelPromise === null) {
      try {
        record.cancelPromise = Promise.resolve(record.result.cancel(reason));
      } catch (error) {
        record.cancelPromise = Promise.reject(error);
      }
      record.cancelPromise.catch(() => {});
    }
    return record.cancelPromise;
  }

  dispose(record) {
    if (!record?.result?.dispose) return Promise.resolve();
    if (record.disposePromise === null) {
      try {
        record.disposePromise = Promise.resolve(record.result.dispose());
      } catch (error) {
        record.disposePromise = Promise.reject(error);
      }
      record.disposePromise.catch(() => {});
    }
    return record.disposePromise;
  }

  quiesce(record, reason = "run quiescing") {
    if (!record) return Promise.resolve();
    if (record.quiescence !== null) return record.quiescence;
    record.controller.abort();
    record.quiescence = this.#quiesceRecord(record, reason);
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

  #workflowResult(record) {
    if (record.workflowResultPromise !== null) return record.workflowResultPromise;
    try {
      const value = typeof record.result?.result === "function"
        ? record.result.result()
        : record.result?.result;
      record.workflowResultPromise = Promise.resolve(value);
    } catch (error) {
      record.workflowResultPromise = Promise.reject(error);
    }
    record.workflowResultPromise.catch(() => {});
    return record.workflowResultPromise;
  }

  async #quiesceRecord(record, reason) {
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

      const cancelled = await this.#settledOrFinalized(
        record,
        this.cancel(record, reason),
      );
      if (cancelled.kind === "finalized") return;
      const settled = await this.#settledOrFinalized(
        record,
        this.#workflowResult(record),
      );
      if (settled.kind === "finalized") return;
      await this.#settledOrFinalized(record, this.dispose(record));
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

export { WORKFLOW_SCRIPTS };
