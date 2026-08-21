export const name = "ecologyrsi-dsh-agent-plane";
export const inject = ["tools", "ecologyAgentTools"];

export function apply(ctx, config = {}) {
  if (!ctx?.tools?.restrict) {
    throw new Error("DSH tools.restrict is required in the agent standing scope");
  }
  // An empty standing allowlist removes every root/global model-facing tool.
  // Role-local tools registered by this same preset scope remain available.
  ctx.tools.restrict({ allow: [] });
  const dispose = registerRoleTools(ctx, config);
  if (typeof ctx.effect === "function") {
    ctx.effect(() => dispose, `ecologyrsi: ${config.role} role tools`);
  }
  return dispose;
}
import { registerRoleTools } from "./roles.js";
