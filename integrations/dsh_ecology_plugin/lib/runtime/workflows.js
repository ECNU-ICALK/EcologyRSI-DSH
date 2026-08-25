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

  finish(record) { this.pending.delete(record); }

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
    const record = {
      kind,
      binding,
      controller,
      signal: controller.signal,
      result: null,
      error: null,
      promise: null,
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
      this.pending.delete(record);
      throw record.error;
    }
    if (!record.result || typeof record.result !== "object" || typeof record.result.then === "function") {
      this.pending.delete(record);
      throw new Error("workflow start must return a synchronous run handle");
    }
    return record;
  }

  async cancelAndQuiesce({ runId } = {}) {
    if (runId !== undefined) this.closeRun(runId);
    while (true) {
      const records = [...this.pending].filter(
        (record) => runId === undefined || record.binding.runId === runId,
      );
      if (records.length === 0) return;
      for (const record of records) record.controller.abort();
      await Promise.allSettled(records.map((record) => record.promise));
      await Promise.allSettled(records.map(async (record) => {
        if (record.kind === "one-shot") {
          await record.result?.dispose?.();
        } else if (record.kind === "continuable" && record.result?.childId) {
          await this.ctx.subagents.interrupt(record.result.childId, {
            kind: "ancestor",
            agent: record.binding.roleHostAgent,
          });
        } else if (record.kind === "workflow") {
          await record.result?.cancel?.("run quiescing");
        }
      }));
      const parents = [...new Set(records
        .filter((record) => record.kind === "continuable" && record.binding.roleHostAgent)
        .map((record) => record.binding.roleHostAgent))];
      if (parents.length) {
        await Promise.allSettled([
          this.ctx.subagents.drainContinuableDescendants(parents),
        ]);
      }
      for (const record of records) this.pending.delete(record);
    }
  }
}

export { WORKFLOW_SCRIPTS };
