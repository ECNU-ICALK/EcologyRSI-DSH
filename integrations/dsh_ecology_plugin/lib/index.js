import { fileURLToPath } from "node:url";

import { resolvePluginConfig } from "./config.js";
import { RuntimeController } from "./runtime/controller.js";
import { registerRuntimeRoutes, RUNTIME_API_BASE } from "./runtime/routes.js";
import { registerApiProxy, API_BASE } from "./web/proxy.js";
import { registerStaticRoute, STATIC_BASE } from "./web/static.js";

export const name = "ecologyrsi-dsh-evolution";
export const inject = [
  "webServer", "agents", "sessions", "tokenMeter", "subagents", "tools",
  "sessionPersistence", "sessionProjections", "agentPresets", "llm", "web",
];

const DEFAULT_STATIC_ROOT = fileURLToPath(
  new URL("../../../plugins/ecology_evolution/", import.meta.url),
);

export function apply(ctx, rawConfig = {}) {
  const config = resolvePluginConfig(rawConfig, { defaultStaticRoot: DEFAULT_STATIC_ROOT });
  ctx.effect(
    () => registerStaticRoute(ctx, config.staticRoot),
    "ecologyrsi: static workbench route",
  );
  ctx.effect(
    () => registerApiProxy(ctx, config),
    "ecologyrsi: loopback API proxy",
  );
  if (config.runtimeToken) {
    const controller = rawConfig.controller || new RuntimeController(ctx);
    if (!rawConfig.controller && config.sidecarToolToken) {
      const runner = controller.configureStageRunner(config);
      ctx.provide("ecologyAgentTools", runner.bridge);
      // A controller this plugin created owns live role-host Sessions, provider
      // admission state and usage timers. Tie them to the plugin's own lifetime
      // so a reload disposes them instead of stranding them in the Node host.
      ctx.effect(
        () => () => { void controller.dispose(); },
        "ecologyrsi: native runtime controller",
      );
    }
    ctx.effect(
      () => registerRuntimeRoutes(ctx, controller, config),
      "ecologyrsi: loopback agent runtime control API",
    );
  }
}

export const routes = Object.freeze({
  api: API_BASE,
  runtime: RUNTIME_API_BASE,
  static: STATIC_BASE,
});
