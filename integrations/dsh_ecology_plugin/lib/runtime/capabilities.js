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
  "web",
]);

function schemaName(value) {
  return value?.name || value?.function?.name || value?.schema?.name || null;
}

export async function runtimeCapabilities(ctx, presetCatalog = []) {
  deployedContent ||= runtimeContentDigest();
  const settings = ctx.get?.("settings") ?? ctx.settings;
  const config = settings?.get?.("llm-pi-ai");
  const contentDigest = createHash("sha256")
    .update(await deployedContent).update("\0").update(inferenceConfigDigest(config)).digest("hex");
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
        const expected = new Set(requiredTools);
        toolSurfaceVerified = (
          names.size === expected.size
          && [...expected].every((name) => names.has(name))
        );
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
      content_digest: contentDigest,
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
import { createHash } from "node:crypto";
import { readdir, readFile } from "node:fs/promises";

export async function runtimeContentDigest(root = new URL("../../", import.meta.url)) {
  const hash = createHash("sha256");
  async function visit(path) {
    const entries = await readdir(new URL(path, root), { withFileTypes: true });
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name, "en"))) {
      const relative = path + entry.name;
      if (entry.isDirectory()) await visit(relative + "/");
      else if (entry.isFile() && /\.(js|json|yml|md)$/.test(entry.name)) {
        hash.update(relative + "\0").update(await readFile(new URL(relative, root))).update("\0");
      }
    }
  }
  for (const folder of ["lib/", "presets/", "schemas/"]) await visit(folder);
  return hash.digest("hex");
}

let deployedContent;

export function inferenceConfigDigest(config) {
  const fields = ["api", "models", "modelOverrides", "compat", "reasoning", "thinkingBudgets",
    "defaultContextWindow", "defaultMaxTokens", "defaultInput"];
  const contracts = Object.fromEntries(Object.entries(config?.providers || {}).map(([id, provider]) => {
    let endpoint = null;
    if (provider.baseURL) {
      const url = new URL(provider.baseURL);
      endpoint = url.origin + url.pathname;
    }
    return [id, { endpoint, ...Object.fromEntries(fields.filter(k => provider[k] !== undefined).map(k => [k, provider[k]])) }];
  }));
  const canonical = value => value && typeof value === "object"
    ? Array.isArray(value) ? value.map(canonical)
      : Object.fromEntries(Object.keys(value).sort().map(k => [k, canonical(value[k])])) : value;
  // Never publish endpoints, headers or credentials. Only the contract digest
  // enters the existing frozen identity and invalidates stale canary receipts.
  return createHash("sha256").update(JSON.stringify(canonical(contracts))).digest("hex");
}
