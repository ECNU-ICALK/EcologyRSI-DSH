import { readFile } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";

export const STATIC_BASE = "/plugins/ecology/evolution";

const MIME_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

function targetFor(staticRoot, pathname) {
  let relative;
  try {
    relative = decodeURIComponent(pathname.slice(STATIC_BASE.length)).replace(/^\/+/, "") || "index.html";
  } catch { return undefined; }
  const allowed = relative === "index.html" || relative === "app.js"
    || relative === "styles.css" || /^assets\/js\/[A-Za-z0-9_.-]+\.js$/.test(relative);
  if (!allowed) return undefined;
  const root = resolve(staticRoot);
  const target = resolve(root, relative);
  return target === root || target.startsWith(`${root}${sep}`) ? target : undefined;
}

function headers(target) {
  return {
    "cache-control": "no-store",
    "content-security-policy": [
      "default-src 'self'", "script-src 'self'", "style-src 'self'",
      "img-src 'self' data:", "connect-src 'self'", "frame-ancestors 'self'",
      "object-src 'none'", "base-uri 'none'",
    ].join("; "),
    "content-type": MIME_TYPES[extname(target)] || "application/octet-stream",
    "referrer-policy": "no-referrer",
    "x-content-type-options": "nosniff",
  };
}

export function registerStaticRoute(ctx, staticRoot) {
  return ctx.webServer.register({
    kind: "prefix", path: STATIC_BASE,
    async handler(req, res) {
      if (!new Set(["GET", "HEAD"]).has(req.method)) {
        res.writeHead(405, { allow: "GET, HEAD" }); res.end(); return;
      }
      const parsed = new URL(req.url || STATIC_BASE, "http://dsh.local");
      if (parsed.pathname === STATIC_BASE) {
        res.writeHead(308, { location: `${STATIC_BASE}/${parsed.search}` }); res.end(); return;
      }
      const target = targetFor(staticRoot, parsed.pathname);
      if (!target) { res.writeHead(404); res.end(); return; }
      try {
        const body = await readFile(target);
        res.writeHead(200, { ...headers(target), "content-length": String(body.length) });
        res.end(req.method === "HEAD" ? undefined : body);
      } catch (error) {
        if (error?.code === "ENOENT") { res.writeHead(404); res.end(); return; }
        throw error;
      }
    },
  });
}
