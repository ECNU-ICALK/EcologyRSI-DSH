// @ecogym/dsh-plugin, host half.
//
// Starts the ecogym service (`python -m ecogym serve --port 0 --exit-with-stdin`) when DSH starts and stops it
// when DSH stops, and serves its JSON API to the browser at /api/ecogym/* (same origin as the DSH web app).
// Requests pass DSH's own /api checks first (Host/Origin fence and the browser login cookie); the bearer token
// the service needs never leaves this process. The browser half (client.js) adds the 生态进化 panel.
//
// Settings, from the plugin row's config if it names a python, else from ~/.ecogym/dsh-plugin.json
// ($ECOGYM_DSH_SETTINGS overrides the path), which `ecogym dsh install` writes; read again on every start:
//   python   interpreter with ecogym installed (required)
//   home     ECOGYM_HOME: ledger, harness store, trajectories (default: ecogym's own default, ~/.ecogym)
//   ledger   ledger file (default: <home>/ledger.sqlite3)
//   env      extra environment for the service, e.g. ECOGYM_DATA_ROOT

import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { request } from "node:http";
import { homedir } from "node:os";
import { delimiter, dirname, join } from "node:path";

export const name = "ecogym";
export const inject = ["webServer"];

export const API = "/api/ecogym";
const LOG_LINES = 300;
const BOOT_FAILURES = 3; // consecutive exits before the service ever got ready: stop retrying until asked
const RESTART_DELAYS_MS = [2_000, 5_000, 15_000, 30_000, 60_000];
const STOP_GRACE_MS = 40_000; // the service stops its agents first (up to ~30 s)
const PROXY_TIMEOUT_MS = 120_000;
const MAX_BODY_BYTES = 2 * 1024 * 1024;
const HOP = ["connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
  "transfer-encoding", "upgrade"];
const REQUEST_DROP = new Set([...HOP, "authorization", "content-length", "cookie", "expect", "host", "origin",
  "referer"]);
const RESPONSE_DROP = new Set([...HOP, "set-cookie"]);

function now() {
  return new Date().toISOString();
}

function stringEnv(value) {
  const env = {};
  if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) && item !== undefined && item !== null) env[key] = String(item);
    }
  }
  return env;
}

function eachLine(stream, onLine) {
  let rest = "";
  stream.setEncoding("utf8");
  stream.on("data", (chunk) => {
    rest += chunk;
    const lines = rest.split(/\r?\n/);
    rest = lines.pop() ?? "";
    for (const line of lines) onLine(line);
  });
  stream.on("end", () => {
    if (rest) onLine(rest);
    rest = "";
  });
}

export function settingsPath(env = process.env) {
  return env.ECOGYM_DSH_SETTINGS || join(homedir(), ".ecogym", "dsh-plugin.json");
}

function readSettings(path) {
  try {
    const value = JSON.parse(readFileSync(path, "utf8"));
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

/** The ecogym service as a child process of DSH: started, watched, restarted after crashes, stopped. */
export class Sidecar {
  constructor(config = {}) {
    this.config = config && typeof config === "object" && typeof config.python === "string" ? config : null;
    this.settings = this.config ? "plugin config" : settingsPath();
    this.configure(this.config || readSettings(this.settings));
    this.token = randomBytes(32).toString("base64url");
    this.state = "stopped"; // starting | ready | exited | failed | unconfigured | stopped
    this.child = null;
    this.port = null;
    this.log = [];
    this.bootFailures = 0;
    this.restarts = 0;
    this.startedAt = null;
    this.readyAt = null;
    this.lastExit = null;
    this.timer = null;
    this.stopping = null; // the old service while a restart waits for it to exit
    this.disposed = false;
  }

  configure(values) {
    this.python = typeof values.python === "string" ? values.python : "";
    this.home = typeof values.home === "string" ? values.home : "";
    this.ledger = typeof values.ledger === "string" ? values.ledger : "";
    this.env = stringEnv(values.env);
  }

  note(line) {
    this.log.push(line.length > 2000 ? `${line.slice(0, 2000)}…` : line);
    if (this.log.length > LOG_LINES) this.log.splice(0, this.log.length - LOG_LINES);
  }

  start() {
    if (this.disposed || this.child || this.stopping) return;
    if (!this.config) this.configure(readSettings(this.settings));
    if (!this.python || !existsSync(this.python)) {
      this.state = "unconfigured";
      this.note(`[ecogym] 找不到 Python 解释器：${this.python || "（未配置）"}（设置来自 ${this.settings}）。`
        + "在装有 ecogym 的环境中运行 ecogym dsh install。");
      return;
    }
    const env = { ...process.env, ...this.env, ECOGYM_SERVICE_TOKEN: this.token, PYTHONUNBUFFERED: "1" };
    if (this.home) env.ECOGYM_HOME = this.home;
    // Agents run `dsh`; it sits next to the node binary that runs this DSH.
    env.PATH = [dirname(process.execPath), env.PATH || ""].filter(Boolean).join(delimiter);
    const args = ["-m", "ecogym", "serve", "--host", "127.0.0.1", "--port", "0", "--exit-with-stdin"];
    if (this.ledger) args.push("--ledger", this.ledger);
    this.state = "starting";
    this.port = null;
    this.startedAt = now();
    this.readyAt = null;
    this.note(`[ecogym] ${now()} 启动 ${this.python} ${args.join(" ")}`);
    let child;
    try {
      child = spawn(this.python, args, { env, stdio: ["pipe", "pipe", "pipe"], cwd: this.home && existsSync(this.home) ? this.home : undefined });
    } catch (error) {
      this.note(`[ecogym] 无法启动：${error.message}`);
      this.exited(null, null);
      return;
    }
    this.child = child;
    child.stdin.on("error", () => {});
    eachLine(child.stdout, (line) => {
      this.note(line);
      const match = /http:\/\/127\.0\.0\.1:(\d+)\//.exec(line);
      if (match && this.child === child && this.state === "starting") {
        this.port = Number(match[1]);
        this.state = "ready";
        this.readyAt = now();
        this.bootFailures = 0;
        console.log(`ecogym: service ready on 127.0.0.1:${this.port} (pid ${child.pid})`);
      }
    });
    eachLine(child.stderr, (line) => this.note(line));
    child.once("error", (error) => {
      this.note(`[ecogym] 进程错误：${error.message}`);
      if (this.child === child) this.exited(null, null);
    });
    child.once("exit", (code, signal) => {
      if (this.child === child) this.exited(code, signal);
    });
  }

  exited(code, signal) {
    const wasReady = this.state === "ready";
    this.child = null;
    this.port = null;
    this.lastExit = { code, signal, at: now() };
    this.note(`[ecogym] ${now()} 后端退出（code ${code ?? "–"}${signal ? `，signal ${signal}` : ""}）`);
    if (this.disposed) {
      this.state = "stopped";
      return;
    }
    if (!wasReady) this.bootFailures += 1;
    if (this.bootFailures >= BOOT_FAILURES) {
      this.state = "failed";
      console.warn(`ecogym: the service failed to start ${this.bootFailures} times; see the 生态进化 panel`);
      return;
    }
    this.state = "exited";
    const delay = RESTART_DELAYS_MS[Math.min(this.restarts, RESTART_DELAYS_MS.length - 1)];
    this.restarts += 1;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.start();
    }, delay);
    this.timer.unref?.();
  }

  terminate(child) {
    try {
      child.stdin.end();
    } catch {
      // already closed
    }
    child.kill("SIGTERM");
    const hard = setTimeout(() => child.kill("SIGKILL"), STOP_GRACE_MS);
    hard.unref?.();
    child.once("exit", () => clearTimeout(hard));
  }

  /** Restart on request: the running service stops its runs (they can be resumed) and a new one starts. */
  restart() {
    clearTimeout(this.timer);
    this.timer = null;
    this.bootFailures = 0;
    if (this.stopping) return; // a restart already waits for the old service; it starts the new one
    const child = this.child;
    if (!child) {
      this.start();
      return;
    }
    this.child = null;
    this.port = null;
    this.state = "starting";
    this.stopping = child;
    child.once("exit", () => {
      this.stopping = null;
      this.start();
    });
    this.terminate(child);
  }

  stop() {
    this.disposed = true;
    clearTimeout(this.timer);
    this.timer = null;
    const child = this.child;
    this.child = null;
    this.port = null;
    this.state = "stopped";
    if (child) this.terminate(child);
  }

  status() {
    return {
      state: this.state, pid: this.child?.pid ?? null, python: this.python, home: this.home || null,
      ledger: this.ledger || null, settings: this.settings, started_at: this.startedAt, ready_at: this.readyAt,
      restarts: this.restarts, last_exit: this.lastExit, log: this.log.slice(-80),
    };
  }
}

function sendJson(res, status, value) {
  const body = Buffer.from(JSON.stringify(value));
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8", "content-length": String(body.length),
    "cache-control": "no-store", "x-content-type-options": "nosniff",
  });
  res.end(body);
}

/** DSH's own /api checks: Host/Origin fence plus the browser login. Without them the API stays closed. */
function rejection(ctx, req) {
  const connection = ctx.get?.("connection");
  if (!connection || typeof connection.requestRejection !== "function") return 503;
  return connection.requestRejection(req);
}

/** The request body, or null once it passes MAX_BODY_BYTES (the rest is drained, not kept). */
function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) resolve(null);
      else chunks.push(chunk);
    });
    req.once("end", () => resolve(Buffer.concat(chunks)));
    req.once("error", reject);
    req.once("close", () => reject(new Error("the request was aborted"))); // no-op after "end"
  });
}

async function forward(sidecar, req, res, path) {
  let body; // POST bodies are small JSON: read whole, so the service always gets a Content-Length
  if (req.method === "POST") {
    try {
      body = await readBody(req);
    } catch {
      return undefined; // the browser went away mid-upload
    }
    if (body === null) return sendJson(res, 413, { error: "请求体太大" });
  }
  return new Promise((done) => {
    const headers = {};
    for (const [key, value] of Object.entries(req.headers)) {
      if (!REQUEST_DROP.has(key.toLowerCase()) && value !== undefined) headers[key] = value;
    }
    headers.host = `127.0.0.1:${sidecar.port}`;
    headers.authorization = `Bearer ${sidecar.token}`;
    if (body !== undefined) headers["content-length"] = String(body.length);
    const upstream = request({ host: "127.0.0.1", port: sidecar.port, method: req.method, path, headers }, (answer) => {
      const out = {};
      for (const [key, value] of Object.entries(answer.headers)) {
        if (!RESPONSE_DROP.has(key.toLowerCase()) && value !== undefined) out[key] = value;
      }
      res.writeHead(answer.statusCode || 502, out);
      answer.pipe(res);
      answer.once("end", done);
      answer.once("close", done);
    });
    upstream.setTimeout(PROXY_TIMEOUT_MS, () => upstream.destroy(new Error("timeout")));
    upstream.once("error", (error) => {
      if (!res.headersSent) sendJson(res, 502, { error: `生态进化后端没有响应：${error.message}` });
      else res.destroy();
      done();
    });
    res.once("close", () => {
      if (!res.writableFinished) upstream.destroy(); // the browser went away: do not keep the backend busy
      done();
    });
    upstream.end(body);
  });
}

async function handle(ctx, sidecar, req, res) {
  const url = new URL(req.url || API, "http://dsh.local");
  const rest = url.pathname.slice(API.length);
  if (rest !== "" && !rest.startsWith("/")) return sendJson(res, 404, { error: "not found" });
  const denied = rejection(ctx, req);
  if (denied !== undefined) {
    const reason = { 401: "unauthorized", 503: "DSH 的登录检查不可用，生态进化接口保持关闭" }[denied] || "forbidden";
    return sendJson(res, denied, { error: reason });
  }
  if (req.method !== "GET" && req.method !== "POST") return sendJson(res, 405, { error: "method not allowed" });
  if (Number(req.headers["content-length"] || 0) > MAX_BODY_BYTES) return sendJson(res, 413, { error: "请求体太大" });
  if (rest === "/_status") return sendJson(res, 200, sidecar.status());
  if (rest === "/_restart") {
    if (req.method !== "POST" || req.headers["x-ecogym"] !== "1") return sendJson(res, 403, { error: "POST with X-Ecogym: 1" });
    sidecar.restart();
    return sendJson(res, 202, sidecar.status());
  }
  if (sidecar.state !== "ready" || !sidecar.port) {
    return sendJson(res, 503, { error: "生态进化后端未就绪", backend: sidecar.status() });
  }
  return forward(sidecar, req, res, `/api${rest || "/"}${url.search}`);
}

export function apply(ctx, config = {}) {
  const sidecar = new Sidecar(config || {});
  ctx.effect(() => {
    sidecar.start();
    return () => sidecar.stop();
  }, "ecogym: service");
  ctx.effect(() => ctx.webServer.register({
    kind: "prefix", path: API, handler: (req, res) => handle(ctx, sidecar, req, res),
  }), "ecogym: /api/ecogym");
}
