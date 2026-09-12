import { createHash, randomUUID } from "node:crypto";

// DSH 0.1.5 narrowed the persisted session header to a closed shape: SessionStore
// builds it from an explicit key allowlist (cwd, parentSession, isSeeded, origin,
// delegationDepth, agentPreset), and CreateAgentOptions.meta now declares exactly
// those keys. Plugin-authored meta is therefore dropped before persistence sees
// it, and a plugin-defined session event is refused on read unless it carries
// `ignorable: true`, which Session.append cannot set. So the role-host provenance
// moves into the one field DSH stores verbatim and revalidates on every read: the
// session id. A drifted preset digest, tool surface, or model route fingerprints
// differently, so it cannot address the session it would have to resume.
const ROLE_SESSION_PROVENANCE_VERSION = "ecologyrsi-dsh.role-session-provenance/1";
const ROLE_SESSION_ID_PREFIX = "ecology-role-";
const ROLE_SESSION_FINGERPRINT_LENGTH = 16;

// Frozen order: the fingerprint is an identity, so its pre-image must not depend
// on key insertion order, and a field appended here changes every session id.
const ROLE_SESSION_PROVENANCE_FIELDS = Object.freeze([
  "run_id",
  "role",
  "preset_id",
  "preset_content_digest",
  "standing_tool_surface_digest",
  "route_config_digest",
]);

export function roleSessionFingerprint(binding) {
  const preimage = JSON.stringify([
    ROLE_SESSION_PROVENANCE_VERSION,
    ...ROLE_SESSION_PROVENANCE_FIELDS.map((field) => String(binding?.[field] ?? "")),
  ]);
  return createHash("sha256").update(preimage, "utf8")
    .digest("hex").slice(0, ROLE_SESSION_FINGERPRINT_LENGTH);
}

export function roleSessionId(binding) {
  return `${ROLE_SESSION_ID_PREFIX}${roleSessionFingerprint(binding)}-${randomUUID()}`;
}

function assertRoleSessionProvenance(sessionId, binding) {
  const expected = roleSessionFingerprint(binding);
  if (sessionId.startsWith(`${ROLE_SESSION_ID_PREFIX}${expected}-`)) return;
  const carried = sessionId.startsWith(ROLE_SESSION_ID_PREFIX)
    ? sessionId.slice(
      ROLE_SESSION_ID_PREFIX.length,
      ROLE_SESSION_ID_PREFIX.length + ROLE_SESSION_FINGERPRINT_LENGTH,
    )
    : "";
  // A pre-0.1.5 role host minted `ecology-role-<uuid>` and kept its provenance in
  // header meta. Its log is also an older session format that this DSH refuses
  // outright, so there is no resume path to offer: name the cut-over rather than
  // report a digest mismatch the operator cannot act on.
  if (!/^[0-9a-f]{16}$/.test(carried)) {
    throw new Error(
      "persisted DSH role-host predates "
      + `${ROLE_SESSION_PROVENANCE_VERSION} and cannot be resumed: ${sessionId}`,
    );
  }
  throw new Error(
    `persisted DSH role-host provenance drifted: session id carries ${carried}, `
    + `binding fingerprints to ${expected} `
    + `over (${ROLE_SESSION_PROVENANCE_FIELDS.join(", ")})`,
  );
}

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

// Model metadata is a small, local-ish catalog read, not a generation call: a
// route that cannot answer within this bound falls back to the declared default
// effort rather than stalling role-host creation.
const MODEL_INFO_TIMEOUT_MS = 10_000;

async function roleRequestOptions(ctx, binding) {
  const options = agentOptionsFor(binding.model);
  if (!options.provider || typeof ctx.llm?.resolveModelInfo !== "function") return options;
  const info = await ctx.llm.resolveModelInfo(
    options.provider, options.model, AbortSignal.timeout(MODEL_INFO_TIMEOUT_MS),
  );
  const reasoning = info?.reasoning;
  const efforts = new Set(reasoning?.efforts?.map((effort) => effort.id) || []);
  // Numerical sample work still uses the real Agent and its tools. Disable
  // optional deep thinking only when the route explicitly advertises it;
  // research/proposal work retains the declared default (or high reasoning).
  const sampleRole = ["sample-planner", "sample-critic"].includes(binding.role);
  const effort = sampleRole && efforts.has("off") ? "off"
    : binding.role === "generation-judge" && efforts.has("low") ? "low"
      : reasoning?.defaultEffort || (efforts.has("high") ? "high" : undefined);
  return effort === undefined ? options : { ...options, reasoningEffort: effort };
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
    const failures = [];
    const processed = new Set();
    const prefix = `${runId}\u0000`;
    while (true) {
      const pending = [...this.pending.entries()]
        .filter(([key]) => key.startsWith(prefix))
        .map(([_key, promise]) => promise);
      if (pending.length) await Promise.allSettled(pending);

      const selected = [...this.handles.entries()].filter(
        ([_key, handle]) => (
          handle.binding.run_id === runId
          && !processed.has(handle)
        ),
      );
      for (const [_key, handle] of selected) processed.add(handle);
      const quiesced = await Promise.allSettled(selected.map(async ([_key, handle]) => {
        const idle = await Promise.allSettled([
          Promise.resolve().then(() => handle.agent?.waitForIdle?.()),
        ]);
        const flushed = await Promise.allSettled([
          Promise.resolve().then(() => handle.agent?.session?.flush?.()),
        ]);
        const failed = [...idle, ...flushed].find((item) => item.status === "rejected");
        if (failed) throw failed.reason;
      }));
      for (const item of quiesced) {
        if (item.status === "rejected") failures.push(item.reason);
      }
      if (dispose) {
        const disposed = await Promise.allSettled(
          selected.map(([_key, handle]) => Promise.resolve().then(() => handle.dispose())),
        );
        for (const [key] of selected) this.handles.delete(key);
        for (const item of disposed) {
          if (item.status === "rejected") failures.push(item.reason);
        }
      }

      const hasPending = [...this.pending.keys()].some((key) => key.startsWith(prefix));
      const hasUnprocessed = [...this.handles.values()].some(
        (handle) => handle.binding.run_id === runId && !processed.has(handle),
      );
      if (!hasPending && !hasUnprocessed) break;
    }
    if (failures.length) throw failures[0];
  }

  async resumeRoleAgent(binding) {
    const key = bindingKey(binding);
    if (this.handles.has(key)) return this.handles.get(key);
    const sessionId = String(binding.session_id || "");
    const presetId = String(binding.preset_id || "");
    if (!sessionId || !/^[a-z0-9][a-z0-9-]*$/.test(presetId)) {
      throw new Error("invalid persisted DSH role-host identity");
    }
    // DSH 0.1.5 removed SessionPersistence.inspect; the equivalent full logical
    // read now lives on the app-layer SessionController. This guard only needs
    // detached header metadata, so `stat` is both the surviving primitive and the
    // cheaper one: it never reads the event log. Log integrity stays where it
    // belongs, in ctx.agents.resume.
    const stored = await this.ctx.sessionPersistence.stat(sessionId);
    const header = stored?.header;
    if (!header || header.id !== sessionId) {
      throw new Error("persisted DSH role-host header is invalid");
    }
    // agentPreset is the one provenance field DSH itself persists, so keep its
    // own message; everything else is covered by the session-id fingerprint.
    if (header.agentPreset !== presetId) {
      throw new Error("persisted DSH role-host agentPreset drifted");
    }
    assertRoleSessionProvenance(sessionId, binding);
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
      const handle = Object.freeze({
        agent,
        rawHandle,
        sessionId,
        standingKey,
        binding: Object.freeze({ ...binding }),
        services: Object.freeze({ compaction }),
        async dispose() { await rawHandle.dispose(); },
      });
      this.handles.set(key, handle);
      return handle;
    } catch (error) {
      try { await rawHandle?.dispose?.(); } catch {}
      throw error;
    }
  }

  async #create(binding, key) {
    const presetId = String(binding.preset_id || "");
    if (!/^[a-z0-9][a-z0-9-]*$/.test(presetId)) throw new Error("invalid DSH preset id");
    const standingKey = await this.ctx.agentPresets.standingKeyFor(presetId);
    if (!standingKey) throw new Error(`DSH preset is not mountable: ${presetId}`);
    const sessionId = roleSessionId(binding);
    const options = {
      sessionId,
      // Closed shape as of DSH 0.1.5; the run/role/digest provenance rides in
      // sessionId because anything else here is dropped before persistence.
      meta: {
        cwd: binding.cwd,
        agentPreset: presetId,
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
      const handle = Object.freeze({
        agent,
        rawHandle,
        sessionId,
        standingKey,
        binding: Object.freeze({ ...binding }),
        services: Object.freeze({ compaction }),
        async dispose() { await rawHandle.dispose(); },
      });
      this.handles.set(key, handle);
      return handle;
    } catch (error) {
      try { await rawHandle?.dispose?.(); } catch {}
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
  const totals = usage && typeof usage === "object"
    ? {
      uncached_input_tokens: nonnegativeInteger(usage.uncachedInputTokens),
      output_tokens: nonnegativeInteger(usage.outputTokens),
      cache_read_tokens: nonnegativeInteger(usage.cacheReadTokens),
      cache_write_tokens: nonnegativeInteger(usage.cacheWriteTokens),
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

export { agentOptionsFor, persistPresetBoundary, roleRequestOptions };
