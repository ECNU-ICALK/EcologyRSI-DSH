import { assertModelArgumentsSafe, TOOL_DEFINITIONS } from "./definitions.js";
import { executeDynamicRetrieval } from "./retrieval.js";

export const DYNAMIC_RETRIEVAL_TOOL_PROFILE = "dynamic-retrieval-v1";

const LEGACY_ROLE_PLUGIN_TOOL_NAMES = Object.freeze({
  coordinator: [],
  researcher: [],
  "candidate-proposer": [],
  "sample-planner": ["ecology_execute_prediction_tool"],
  "sample-critic": [],
  "generation-judge": [],
});

const LEGACY_ROLE_TOOL_NAMES = Object.freeze({
  coordinator: ["skill"],
  researcher: ["skill"],
  "candidate-proposer": ["skill"],
  "sample-planner": ["skill", "ecology_execute_prediction_tool"],
  "sample-critic": ["skill"],
  "generation-judge": ["skill"],
});

export const ROLE_PLUGIN_TOOL_NAMES = Object.freeze({
  coordinator: ["web_search"],
  researcher: ["web_search"],
  "candidate-proposer": ["web_search"],
  "sample-planner": ["web_search", "ecology_execute_prediction_tool"],
  "sample-critic": ["web_search"],
  "generation-judge": ["web_search"],
});

export const ROLE_TOOL_NAMES = Object.freeze({
  coordinator: ["skill", "web_search"],
  researcher: ["skill", "web_search"],
  "candidate-proposer": ["skill", "web_search"],
  "sample-planner": ["skill", "web_search", "ecology_execute_prediction_tool"],
  "sample-critic": ["skill", "web_search"],
  "generation-judge": ["skill", "web_search"],
});

const STRUCTURED_OUTPUT_TOOL = "structured_output";

function profileUsesDynamicRetrieval(toolProfile) {
  if (toolProfile == null || toolProfile === "") return false;
  if (toolProfile !== DYNAMIC_RETRIEVAL_TOOL_PROFILE) {
    throw new Error(`unsupported ecology role tool profile: ${String(toolProfile)}`);
  }
  return true;
}

export function roleToolNames(role, toolProfile = null) {
  const names = (profileUsesDynamicRetrieval(toolProfile)
    ? ROLE_TOOL_NAMES
    : LEGACY_ROLE_TOOL_NAMES)[role];
  if (!names) throw new Error(`unknown ecology role: ${role}`);
  return names;
}

export function rolePluginToolNames(role, toolProfile = null) {
  const names = (profileUsesDynamicRetrieval(toolProfile)
    ? ROLE_PLUGIN_TOOL_NAMES
    : LEGACY_ROLE_PLUGIN_TOOL_NAMES)[role];
  if (!names) throw new Error(`unknown ecology role: ${role}`);
  return names;
}

export function registerRoleToolGuard(ctx, role, { toolProfile = null } = {}) {
  const names = roleToolNames(role, toolProfile);
  if (!names) throw new Error(`unknown ecology role: ${role}`);
  if (typeof ctx?.tools?.guard !== "function") {
    throw new Error("DSH tools.guard is required in the agent standing scope");
  }
  const allowed = new Set([...names, STRUCTURED_OUTPUT_TOOL]);
  return ctx.tools.guard((exec) => (
    allowed.has(exec?.name)
      ? undefined
      : `tool ${String(exec?.name || "<unknown>")} is outside the frozen ${role} role surface`
  ));
}

export function registerRoleTools(ctx, config = {}) {
  const role = String(config.role || "");
  const names = rolePluginToolNames(role, config.toolProfile);
  const register = ctx?.tools?.register || ctx?.tools?.define;
  if (typeof register !== "function") throw new Error("DSH tool registration service is required");
  const bridge = config.bridge || ctx.ecologyAgentTools;
  const disposers = [];
  for (const name of names) {
    const definition = TOOL_DEFINITIONS[name];
    const handler = name === "web_search"
      ? async (args, exec) => executeDynamicRetrieval({ ctx, bridge, role, args, exec })
      : async (args, exec) => {
        assertModelArgumentsSafe(args, { labelFree: role === "sample-planner" });
        if (!bridge?.bindingFor || !bridge?.sidecar?.request) throw new Error("ecology role tool bridge is unavailable");
        const binding = await bridge.bindingFor(exec, { role, toolName: name });
        if (!binding || binding.role !== role) throw new Error("role tool authorization failed");
        const result = await bridge.sidecar.request(`/api/ecology-agent-sidecar/v1/tools/${name}`, {
          body: { identity: binding, arguments: structuredClone(args) },
          signal: exec?.signal,
        });
        return result;
      };
    const disposer = register.call(ctx.tools, {
      ...definition,
      output: {
        schema: { type: "object", additionalProperties: true },
        render: (_args, value) => [{ type: "text", text: JSON.stringify(value) }],
      },
      execute: handler,
    });
    if (typeof disposer === "function") disposers.push(disposer);
  }
  return () => { for (const dispose of disposers.reverse()) dispose(); };
}
