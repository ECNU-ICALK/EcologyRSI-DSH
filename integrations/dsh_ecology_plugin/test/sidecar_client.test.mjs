import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";

import { SidecarClient, SidecarError } from "../lib/sidecar/client.js";

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
}
function close(server) { return new Promise((resolve) => server.close(resolve)); }

test("sidecar client bounds response size and never exposes its token", async () => {
  const server = createServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ data: "x".repeat(500) }));
  });
  const port = await listen(server);
  try {
    const client = new SidecarClient({
      origin: `http://127.0.0.1:${port}`,
      token: "top-secret-token",
      maxResponseBytes: 100,
    });
    await assert.rejects(
      client.request("/api/ecology-agent-sidecar/v1/context", { method: "GET" }),
      (error) => error instanceof SidecarError
        && error.code === "response_too_large"
        && !String(error).includes("top-secret-token"),
    );
  } finally { await close(server); }
});

test("sidecar timeout and caller abort have stable public codes", async () => {
  const server = createServer((_req, _res) => {});
  const port = await listen(server);
  try {
    const client = new SidecarClient({
      origin: `http://127.0.0.1:${port}`, token: "token", totalTimeoutMs: 30,
    });
    await assert.rejects(
      client.request("/api/ecology-agent-sidecar/v1/context", { method: "GET" }),
      (error) => error.code === "timeout",
    );
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      client.request("/api/ecology-agent-sidecar/v1/context", {
        method: "GET", signal: controller.signal,
      }),
      (error) => error.code === "aborted",
    );
  } finally { await close(server); }
});

test("sidecar rejection retains only the sidecar's already-redacted public detail", async () => {
  const server = createServer((_req, res) => {
    res.writeHead(409, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "stage admission is closed" }));
  });
  const port = await listen(server);
  try {
    const client = new SidecarClient({
      origin: `http://127.0.0.1:${port}`,
      token: "top-secret-token",
    });
    await assert.rejects(
      client.request("/api/ecology-agent-sidecar/v1/structured-results", {
        body: { result: true },
      }),
      (error) => error instanceof SidecarError
        && error.code === "sidecar_rejected"
        && error.publicDetail === "stage admission is closed"
        && !JSON.stringify(error).includes("top-secret-token"),
    );
  } finally { await close(server); }
});

test("sidecar origin and paths are literal loopback allowlists", () => {
  assert.throws(() => new SidecarClient({ origin: "http://example.com", token: "x" }), /loopback/);
  const client = new SidecarClient({ origin: "http://127.0.0.1:8777", token: "x" });
  assert.rejects(client.request("http://attacker.example/private"), /path/);
});
