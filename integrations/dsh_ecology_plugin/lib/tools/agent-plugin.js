import { registerRoleToolGuard, registerRoleTools, roleToolNames } from "./roles.js";
import { roleRequestOptions } from "../runtime/agents.js";

export const name = "ecologyrsi-dsh-agent-plane";
export const inject = ["tools", "web", "ecologyAgentTools", "llm"];

export function installRoleReasoning(ctx, role) {
  if (typeof ctx.on !== "function") return () => {};
  const policies = new Map();
  // DSH 0.1.5 added AgentOptions.reasoningEffort, but that only seeds the host's
  // own default; it is not inherited. The preset's request waterfall is, by both
  // its role host AND its spawned children, so per-role effort stays here — and
  // DSH records the resolved effort in each child's native request/header.
  return ctx.on("agent/request", async (_payload, next) => {
    const resolved = await next();
    const model = `${resolved.provider}/${resolved.model}`;
    if (!policies.has(model)) {
      const policy = roleRequestOptions(ctx, { model, role });
      policies.set(model, policy);
      policy.catch(() => policies.delete(model));
    }
    const { reasoningEffort } = await policies.get(model);
    return reasoningEffort === undefined ? resolved : { ...resolved, reasoningEffort };
  });
}

export function apply(ctx, config = {}) {
  const role = String(config.role || "");
  roleToolNames(role, config.toolProfile);

  // A preset is a standing ancestor of both its role host and spawned children.
  // A standing `restrict({ allow: [] })` therefore also masks tools registered
  // by that same preset when a descendant resolves its inherited surface.  The
  // Host readiness probe verifies the exact visible schema set instead; this
  // guard remains the execution-time authorization boundary.
  const disposeTools = registerRoleTools(ctx, { ...config, role });
  const disposeGuard = registerRoleToolGuard(ctx, role, {
    toolProfile: config.toolProfile,
  });
  const disposeReasoning = installRoleReasoning(ctx, role);
  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    disposeReasoning?.();
    disposeGuard?.();
    disposeTools?.();
  };
  if (typeof ctx.effect === "function") {
    ctx.effect(() => dispose, `ecologyrsi: ${role} role tools`);
  }
  return dispose;
}
