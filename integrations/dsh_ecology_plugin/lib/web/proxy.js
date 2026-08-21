import { request as httpRequest } from "node:http";

export const API_BASE = "/api/ecology-evolution";

const HOP = new Set([
  "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
  "te", "trailer", "transfer-encoding", "upgrade",
]);
const REQUEST_BLOCKED = new Set([...HOP, "authorization", "cookie"]);
const RESPONSE_BLOCKED = new Set([...HOP, "set-cookie"]);

function copyHeaders(source, { host, blocked = HOP } = {}) {
  const result = {};
  for (const [key, value] of Object.entries(source || {})) {
    if (!blocked.has(key.toLowerCase()) && value !== undefined) result[key] = value;
  }
  if (host) result.host = host;
  return result;
}

function requestTarget(requestUrl, origin) {
  let parsed;
  try { parsed = new URL(requestUrl || API_BASE, "http://dsh.local"); } catch { return undefined; }
  if (parsed.origin !== "http://dsh.local"
    || (parsed.pathname !== API_BASE && !parsed.pathname.startsWith(`${API_BASE}/`))) return undefined;
  const target = new URL(origin.origin);
  target.pathname = parsed.pathname;
  target.search = parsed.search;
  return target;
}

function errorResponse(res, status, message) {
  const body = Buffer.from(JSON.stringify({ error: message }));
  res.writeHead(status, {
    "cache-control": "no-store", "content-length": String(body.length),
    "content-type": "application/json; charset=utf-8",
  });
  res.end(body);
}

export function registerApiProxy(ctx, config) {
  return ctx.webServer.register({
    kind: "prefix", path: API_BASE,
    handler(req, res) {
      const target = requestTarget(req.url, config.backendOrigin);
      if (!target || !new Set(["GET", "HEAD", "POST", "PATCH", "DELETE"]).has(req.method)) {
        errorResponse(res, 400, "无效的生态模型进化代理路径"); return Promise.resolve();
      }
      const declared = Number(req.headers?.["content-length"] || 0);
      if (declared > config.maxBodyBytes) {
        errorResponse(res, 413, "请求过大"); return Promise.resolve();
      }
      return new Promise((resolveRequest) => {
        let responseBytes = 0;
        const upstream = httpRequest(target, {
          method: req.method,
          headers: {
            ...copyHeaders(req.headers, { host: target.host, blocked: REQUEST_BLOCKED }),
            ...(config.serviceToken ? { authorization: `Bearer ${config.serviceToken}` } : {}),
          },
        }, (upstreamResponse) => {
          const declaredResponseBytes = Number(upstreamResponse.headers["content-length"]);
          if (Number.isFinite(declaredResponseBytes)
            && declaredResponseBytes > config.maxResponseBytes) {
            upstreamResponse.resume();
            errorResponse(res, 502, "生态模型进化响应过大");
            upstreamResponse.once("end", resolveRequest);
            upstreamResponse.once("close", resolveRequest);
            return;
          }
          res.writeHead(
            upstreamResponse.statusCode || 502,
            copyHeaders(upstreamResponse.headers, { blocked: RESPONSE_BLOCKED }),
          );
          upstreamResponse.on("data", (chunk) => {
            responseBytes += chunk.length;
            if (responseBytes > config.maxResponseBytes) {
              upstreamResponse.destroy(); res.destroy();
            }
          });
          upstreamResponse.pipe(res);
          upstreamResponse.once("end", resolveRequest);
          upstreamResponse.once("close", resolveRequest);
        });
        upstream.setTimeout(config.totalTimeoutMs, () => upstream.destroy(new Error("timeout")));
        upstream.once("error", () => {
          if (!res.headersSent) errorResponse(res, 502, "生态模型进化服务不可用");
          else res.destroy();
          resolveRequest();
        });
        req.once("aborted", () => upstream.destroy());
        req.pipe(upstream);
      });
    },
  });
}
