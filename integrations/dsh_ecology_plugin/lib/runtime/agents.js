import { randomUUID } from "node:crypto";

function bindingKey(binding) {
  return `${binding.run_id}\u0000${binding.role}`;
}

function agentOptionsFor(modelRoute) {
  const route = String(modelRoute || "");
  const separator = route.indexOf("/");
  if (separator <= 0 || separator === route.length - 1) return { model: route };
  return {
    provider: route.slice(0, separator),
    model: route.slice(separator + 1),
  };
}

function presetSetup(ctx, presetId) {
  return async (agentCtx) => {
    const mounted = await ctx.agentPresets.mount(agentCtx, presetId);
    if (mounted?.id && mounted.id !== presetId) {
      throw new Error(`DSH preset mount drifted: expected ${presetId}, got ${mounted.id}`);
    }
  };
}

async function persistPresetBoundary(agent, presetId) {
  const session = agent?.session;
  if (!session) throw new Error("created DSH role-host has no Session");
  if (typeof session.append === "function") {
    await session.append("agent-preset/selected", { agentPreset: presetId });
  }
  if (typeof session.flush === "function") await session.flush();
}

export class RoleAgentManager {
  constructor(ctx) {
    this.ctx = ctx;
    this.pending = new Map();
    this.handles = new Map();
  }

  createRoleAgent(binding) {
    const key = bindingKey(binding);
    if (this.handles.has(key)) return Promise.resolve(this.handles.get(key));
    if (this.pending.has(key)) return this.pending.get(key);
    const promise = this.#create(binding, key).finally(() => this.pending.delete(key));
    this.pending.set(key, promise);
    return promise;
  }

  get(runId, role) {
    return this.handles.get(bindingKey({ run_id: runId, role })) || null;
  }

  async quiesceRun(runId, { dispose = false } = {}) {
    const selected = [...this.handles.entries()].filter(
      ([_key, handle]) => handle.binding.run_id === runId,
    );
    for (const [_key, handle] of selected) {
      await handle.agent?.waitForIdle?.();
      await handle.agent?.session?.flush?.();
    }
    if (dispose) {
      await Promise.all(selected.map(([_key, handle]) => handle.dispose()));
      for (const [key] of selected) this.handles.delete(key);
    }
  }

  async resumeRoleAgent(binding) {
    const key = bindingKey(binding);
    if (this.handles.has(key)) return this.handles.get(key);
    const sessionId = String(binding.session_id || "");
    const presetId = String(binding.preset_id || "");
    if (!sessionId || !/^[a-z0-9][a-z0-9-]*$/.test(presetId)) {
      throw new Error("invalid persisted DSH role-host identity");
    }
    const inspected = await this.ctx.sessionPersistence.inspect(sessionId);
    const header = inspected?.header || inspected;
    const meta = header?.meta;
    if (!header || !meta || (header.id && header.id !== sessionId)) {
      throw new Error("persisted DSH role-host header is invalid");
    }
    const expectedMeta = {
      agentPreset: presetId,
      ecologyRunId: binding.run_id,
      ecologyRole: binding.role,
      ecologyPresetContentDigest: binding.preset_content_digest,
      ecologyToolSurfaceDigest: binding.standing_tool_surface_digest,
      ecologyRouteConfigDigest: binding.route_config_digest,
    };
    for (const [name, value] of Object.entries(expectedMeta)) {
      if (meta[name] !== value) throw new Error(`persisted DSH role-host ${name} drifted`);
    }
    const standingKey = await this.ctx.agentPresets.standingKeyFor(presetId);
    if (!standingKey) throw new Error(`DSH preset is not mountable: ${presetId}`);
    const rawHandle = await this.ctx.agents.resume({
      resumeSessionId: sessionId,
      agentOptions: agentOptionsFor(binding.model),
      setup: presetSetup(this.ctx, presetId),
    });
    const agent = rawHandle?.agent || rawHandle;
    try {
      const compaction = await this.ctx.agentPresets.serviceFor(agent, "compaction");
      if (!compaction) throw new Error(`preset ${presetId} has no Compaction service`);
      const workflowEngine = binding.require_workflow
        ? await this.ctx.agentPresets.serviceFor(agent, "workflowEngine")
        : null;
      if (binding.require_workflow && !workflowEngine) {
        throw new Error(`preset ${presetId} has no Workflow service`);
      }
      const handle = Object.freeze({
        agent,
        rawHandle,
        sessionId,
        standingKey,
        binding: Object.freeze({ ...binding }),
        services: Object.freeze({ compaction, workflowEngine }),
        async dispose() { await rawHandle.dispose(); },
      });
      this.handles.set(key, handle);
      return handle;
    } catch (error) {
      await rawHandle?.dispose?.();
      throw error;
    }
  }

  async #create(binding, key) {
    const presetId = String(binding.preset_id || "");
    if (!/^[a-z0-9][a-z0-9-]*$/.test(presetId)) throw new Error("invalid DSH preset id");
    const standingKey = await this.ctx.agentPresets.standingKeyFor(presetId);
    if (!standingKey) throw new Error(`DSH preset is not mountable: ${presetId}`);
    const sessionId = `ecology-role-${randomUUID()}`;
    const options = {
      sessionId,
      meta: {
        cwd: binding.cwd,
        agentPreset: presetId,
        ecologyRunId: binding.run_id,
        ecologyRole: binding.role,
        ecologyPresetContentDigest: binding.preset_content_digest,
        ecologyToolSurfaceDigest: binding.standing_tool_surface_digest,
        ecologyRouteConfigDigest: binding.route_config_digest,
      },
      agentOptions: agentOptionsFor(binding.model),
      setup: presetSetup(this.ctx, presetId),
    };
    const rawHandle = await this.ctx.agents.create(options);
    const agent = rawHandle?.agent || rawHandle;
    try {
      await persistPresetBoundary(agent, presetId);
      const compaction = await this.ctx.agentPresets.serviceFor(agent, "compaction");
      if (!compaction) throw new Error(`preset ${presetId} has no Compaction service`);
      const workflowEngine = binding.require_workflow
        ? await this.ctx.agentPresets.serviceFor(agent, "workflowEngine")
        : null;
      if (binding.require_workflow && !workflowEngine) {
        throw new Error(`preset ${presetId} has no Workflow service`);
      }
      const handle = Object.freeze({
        agent,
        rawHandle,
        sessionId,
        standingKey,
        binding: Object.freeze({ ...binding }),
        services: Object.freeze({ compaction, workflowEngine }),
        async dispose() { await rawHandle.dispose(); },
      });
      this.handles.set(key, handle);
      return handle;
    } catch (error) {
      await rawHandle?.dispose?.();
      throw error;
    }
  }

  async dispose() {
    const settled = await Promise.allSettled(this.pending.values());
    const handles = new Set(this.handles.values());
    for (const item of settled) if (item.status === "fulfilled") handles.add(item.value);
    this.pending.clear();
    this.handles.clear();
    await Promise.allSettled([...handles].map((handle) => handle.dispose()));
  }
}

export function tokenSemantics(ctx, session) {
  let contextPressure = null;
  let providerUsage = null;
  try {
    contextPressure = ctx?.tokenMeter?.measure?.(session) ?? null;
  } catch {
    // Observability is intentionally fail-soft. A cold/disposed child Session
    // can race the final metrics snapshot, but must never reject a valid,
    // schema-bound scientific result.
  }
  try {
    const snapshot = ctx?.sessionProjections?.snapshot?.(session);
    providerUsage = snapshot?.values?.tokenUsage ?? null;
  } catch {
    // Preserve an available TokenMeter reading even when the cumulative
    // projection cache is temporarily unavailable, and vice versa.
  }
  return {
    context_pressure: contextPressure,
    provider_usage: providerUsage,
  };
}

function nonnegativeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0 ? value : 0;
}

export function dshSessionMetrics(ctx, sessionId) {
  const session = ctx?.sessions?.get?.(sessionId);
  if (!session) {
    return {
      schema_version: "ecologyrsi-dsh.dsh-session-metrics/1",
      session_id: String(sessionId),
      context_pressure: { available: false, source: "dsh_token_meter" },
      provider_usage: {
        available: false,
        source: "dsh_session_projection_token_usage",
      },
    };
  }
  const semantics = tokenSemantics(ctx, session);
  const pressure = semantics.context_pressure;
  const usage = semantics.provider_usage;
  // @deepseek-ai/dsh-token-meter exposes tokenUsage as the four cumulative
  // buckets directly.  It is not wrapped in a `totals` member.
  const rawTotals = usage;
  const totals = rawTotals && typeof rawTotals === "object"
    ? {
      uncached_input_tokens: nonnegativeInteger(rawTotals.uncachedInputTokens),
      output_tokens: nonnegativeInteger(rawTotals.outputTokens),
      cache_read_tokens: nonnegativeInteger(rawTotals.cacheReadTokens),
      cache_write_tokens: nonnegativeInteger(rawTotals.cacheWriteTokens),
    }
    : null;
  if (totals) {
    totals.total_tokens = (
      totals.uncached_input_tokens
      + totals.output_tokens
      + totals.cache_read_tokens
      + totals.cache_write_tokens
    );
  }
  return {
    schema_version: "ecologyrsi-dsh.dsh-session-metrics/1",
    session_id: String(sessionId),
    context_pressure: pressure && typeof pressure === "object"
      ? {
        available: true,
        source: "dsh_token_meter",
        measurement: "current_context_pressure",
        log_revision: nonnegativeInteger(pressure.logRevision),
        baseline_kind: String(pressure.baseline?.kind || "none"),
        total_tokens: nonnegativeInteger(pressure.totalTokens),
        surface_tokens: nonnegativeInteger(pressure.surfaceTokens),
      }
      : { available: false, source: "dsh_token_meter" },
    provider_usage: totals
      ? {
        available: true,
        source: "dsh_session_projection_token_usage",
        measurement: "cumulative_provider_reported_usage",
        totals,
      }
      : {
        available: false,
        source: "dsh_session_projection_token_usage",
      },
  };
}

export { agentOptionsFor, persistPresetBoundary };
