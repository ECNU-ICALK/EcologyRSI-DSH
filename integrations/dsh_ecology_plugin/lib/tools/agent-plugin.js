import { registerRoleToolGuard, registerRoleTools, roleToolNames } from "./roles.js";

export const name = "ecologyrsi-dsh-agent-plane";
export const inject = ["tools", "web", "ecologyAgentTools"];

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
  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    disposeGuard?.();
    disposeTools?.();
  };
  if (typeof ctx.effect === "function") {
    ctx.effect(() => dispose, `ecologyrsi: ${role} role tools`);
  }
  return dispose;
}
