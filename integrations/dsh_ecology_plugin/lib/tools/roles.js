import { ROLE_CONTRACTS } from "../runtime/contracts.js";
import { assertModelArgumentsSafe, TOOL_DEFINITIONS } from "./definitions.js";
import { executeDynamicRetrieval } from "./retrieval.js";
import { projectPredictionToolResult } from "../runtime/display-projection.js";
import { SidecarError } from "../sidecar/client.js";

export const DYNAMIC_RETRIEVAL_TOOL_PROFILE = "dynamic-retrieval-v1";

export const ROLE_TOOL_NAMES = Object.freeze(Object.fromEntries(
  Object.entries(ROLE_CONTRACTS).map(([role, contract]) => [role, contract.required_tools]),
));
export const ROLE_PLUGIN_TOOL_NAMES = Object.freeze(Object.fromEntries(
  Object.entries(ROLE_TOOL_NAMES).map(([role, names]) =>
    [role, Object.freeze(names.filter(name => name !== "skill"))]),
));
const STRUCTURED_OUTPUT_TOOL = "structured_output";

export function roleToolNames(role, toolProfile = DYNAMIC_RETRIEVAL_TOOL_PROFILE) {
  if (toolProfile !== DYNAMIC_RETRIEVAL_TOOL_PROFILE) {
    throw new Error(`unsupported ecology role tool profile: ${String(toolProfile)}`);
  }
  const names = ROLE_TOOL_NAMES[role];
  if (!names) throw new Error(`unknown ecology role: ${role}`);
  return names;
}

export function rolePluginToolNames(role, toolProfile = DYNAMIC_RETRIEVAL_TOOL_PROFILE) {
  roleToolNames(role, toolProfile);
  return ROLE_PLUGIN_TOOL_NAMES[role];
}

export function registerRoleToolGuard(ctx, role, { toolProfile = DYNAMIC_RETRIEVAL_TOOL_PROFILE } = {}) {
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
  const register = ctx?.tools?.register;
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
        try {
          return await bridge.sidecar.request(`/api/ecology-agent-sidecar/v1/tools/${name}`, {
            body: { identity: binding, arguments: structuredClone(args) },
            signal: exec?.signal,
          });
        } catch (error) {
          // Explain this exact Host diagnostic without exposing arbitrary
          // response text. Recovery replays an existing request, never grants
          // additional computation or makes unseen evidence citeable.
          if (name === "ecology_execute_prediction_tool" && error instanceof SidecarError
            && error.code === "sidecar_rejected"
            && error.publicDetail === "prediction tool call budget exhausted") {
            throw new SidecarError(error.code,
              "Prediction computation budget is exhausted. If prediction_contract.recoverable_call_requests is present, replay an exact listed request, including its original call_id, to receive already-computed evidence without a new computation. Otherwise use successful evidence already received. Do not invent call IDs or predictions; new computations remain unavailable.");
          }
          throw error;
        }
      };
    const disposer = register.call(ctx.tools, {
      ...definition,
      output: {
        schema: { type: "object", additionalProperties: true },
        render: (_args, value) => [{ type: "text", text: JSON.stringify(
          name === "ecology_execute_prediction_tool" ? projectPredictionToolResult(value) : value,
        ) }],
      },
      execute: handler,
    });
    if (typeof disposer === "function") disposers.push(disposer);
  }
  return () => { for (const dispose of disposers.reverse()) dispose(); };
}
