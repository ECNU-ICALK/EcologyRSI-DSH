import {
  bearerMatches,
  isLoopbackAddress,
  readBoundedJson,
  safeJsonError,
} from "../security.js";

export const RUNTIME_API_BASE = "/api/ecology-agent-runtime/v1";

const IDENTITY_FIELDS = new Set([
  "run_id",
  "run_state_revision",
  "stage_attempt",
  "ledger_expected_revision",
  "idempotency_key",
]);

function validIdentityBody(body, extraFields = new Set()) {
  const allowed = new Set([...IDENTITY_FIELDS, ...extraFields]);
  if (Object.keys(body).some((name) => !allowed.has(name))) return false;
  if ([...IDENTITY_FIELDS].some((name) => !(name in body))) return false;
  if (typeof body.run_id !== "string" || !body.run_id) return false;
  if (typeof body.idempotency_key !== "string" || !body.idempotency_key) return false;
  return ["run_state_revision", "stage_attempt", "ledger_expected_revision"]
    .every((name) => Number.isSafeInteger(body[name]) && body[name] >= 0);
}

function sendJson(res, status, value) {
  const body = Buffer.from(JSON.stringify(value));
  res.writeHead(status, {
    "cache-control": "no-store",
    "content-length": String(body.length),
    "content-type": "application/json; charset=utf-8",
    "x-content-type-options": "nosniff",
  });
  res.end(body);
}

export function registerRuntimeRoutes(ctx, controller, config) {
  const runtimeToken = String(config?.runtimeToken || "");
  if (!runtimeToken) throw new Error("runtimeToken is required");
  const maxBodyBytes = Number(config?.maxBodyBytes || 1024 * 1024);
  return ctx.webServer.register({
    kind: "prefix",
    path: RUNTIME_API_BASE,
    async handler(req, res) {
      if (!isLoopbackAddress(req.socket?.remoteAddress)) {
        safeJsonError(res, 403, "loopback_required");
        return;
      }
      if (req.headers?.origin || req.headers?.["sec-fetch-site"] === "cross-site") {
        safeJsonError(res, 403, "browser_request_forbidden");
        return;
      }
      if (!bearerMatches(req.headers?.authorization, runtimeToken)) {
        safeJsonError(res, 401, "invalid_control_token");
        return;
      }
      let parsed;
      try {
        parsed = new URL(req.url || RUNTIME_API_BASE, "http://dsh.local");
      } catch {
        safeJsonError(res, 404, "unknown_runtime_route");
        return;
      }
      if (parsed.origin !== "http://dsh.local") {
        safeJsonError(res, 404, "unknown_runtime_route");
        return;
      }
      const relative = parsed.pathname.slice(RUNTIME_API_BASE.length);
      if (req.method === "GET" && relative === "/capabilities") {
        try { sendJson(res, 200, await controller.capabilities()); }
        catch { safeJsonError(res, 502, "runtime_controller_failed"); }
        return;
      }
      const statusMatch = relative.match(/^\/runs\/([^/]+)$/);
      if (req.method === "GET" && statusMatch) {
        try { sendJson(res, 200, await controller.status(decodeURIComponent(statusMatch[1]))); }
        catch { safeJsonError(res, 404, "runtime_run_not_found"); }
        return;
      }
      const create = req.method === "POST" && ["/runs", "/runs/start"].includes(relative);
      const mutation = req.method === "POST"
        ? relative.match(/^\/runs\/([^/]+)\/(stages|pause|cancel|resume)$/)
        : null;
      if (!create && !mutation) {
        safeJsonError(res, 404, "unknown_runtime_route");
        return;
      }
      let body;
      try {
        body = await readBoundedJson(req, maxBodyBytes);
      } catch {
        safeJsonError(res, 400, "invalid_request_body");
        return;
      }
      const extra = create
        ? new Set(["binding"])
        : mutation?.[2] === "stages" ? new Set(["stage", "request"]) : new Set();
      if (!validIdentityBody(body, extra)
        || (mutation && decodeURIComponent(mutation[1]) !== body.run_id)
        || (mutation?.[2] === "stages" && (typeof body.stage !== "string" || !body.stage))) {
        safeJsonError(res, 400, "invalid_start_contract");
        return;
      }
      try {
        let result;
        if (create) result = await controller.startRun(body);
        else if (mutation[2] === "stages") result = await controller.runStage(body);
        else if (mutation[2] === "pause") result = await controller.pause(body);
        else if (mutation[2] === "cancel") result = await controller.cancel(body);
        else result = await controller.resume(body);
        sendJson(res, 200, result);
      } catch (error) {
        const diagnostic = error?.publicDetail || error?.code || error?.name || "unknown";
        console.warn(`[ecologyrsi] runtime controller failed: ${diagnostic}`);
        safeJsonError(res, 502, "runtime_controller_failed");
      }
    },
  });
}
