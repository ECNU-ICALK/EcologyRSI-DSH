const ROOT_SERVICES = Object.freeze([
  "agents",
  "sessions",
  "tokenMeter",
  "subagents",
  "tools",
  "sessionPersistence",
  "sessionProjections",
  "agentPresets",
  "llm",
]);

function schemaName(value) {
  return value?.name || value?.function?.name || value?.schema?.name || null;
}

export async function runtimeCapabilities(ctx, presetCatalog = []) {
  const missing = ROOT_SERVICES.filter((name) => ctx?.[name] == null);
  const presets = [];
  for (const raw of presetCatalog) {
    const presetId = String(raw?.preset_id || "");
    const requiredTools = Array.isArray(raw?.required_tools)
      ? [...new Set(raw.required_tools.map(String))].sort()
      : [];
    let standingKey = null;
    let presetMountable = false;
    let toolSurfaceVerified = false;
    let routeResolvable = false;
    try {
      standingKey = await ctx?.agentPresets?.standingKeyFor?.(presetId);
      presetMountable = (
        (typeof standingKey === "string" && standingKey.length > 0)
        || (standingKey != null && typeof standingKey === "object")
      );
    } catch {}
    if (presetMountable) {
      try {
        const schemas = await ctx?.tools?.schemas?.(standingKey);
        const names = new Set(Array.isArray(schemas) ? schemas.map(schemaName) : []);
        toolSurfaceVerified = requiredTools.every((name) => names.has(name));
      } catch {}
    }
    const resolveCallConfig = ctx?.llm?.resolveCallConfig;
    if (typeof raw?.model !== "string" || !raw.model) {
      routeResolvable = typeof resolveCallConfig === "function";
    } else {
      try {
        const resolved = await resolveCallConfig.call(ctx.llm, { model: raw.model });
        routeResolvable = Boolean(resolved && typeof resolved === "object");
      } catch {}
    }
    presets.push({
      preset_id: presetId,
      declared: Boolean(presetId),
      standing_key: standingKey,
      preset_mountable: presetMountable,
      tool_surface_verified: toolSurfaceVerified,
      route_resolvable: routeResolvable,
      // These require a real role-host or a successful real stage call.
      live_agent_service_ready: false,
      first_call_verified: false,
    });
  }
  return {
    schema_version: "ecology-agent-runtime-capabilities/1",
    ready: missing.length === 0
      && presets.every((item) => item.preset_mountable
        && item.tool_surface_verified
        && item.route_resolvable),
    root_services: {
      required: [...ROOT_SERVICES],
      missing,
      declared: missing.length === 0,
    },
    presets,
    live_agent_service_ready: false,
    first_call_verified: false,
  };
}

export { ROOT_SERVICES };
