import assert from "node:assert/strict";
import { Readable, Writable } from "node:stream";
import test from "node:test";

import { registerRuntimeRoutes } from "../lib/runtime/routes.js";

class Response extends Writable {
  constructor() { super(); this.chunks = []; this.statusCode = null; this.headers = {}; }
  _write(chunk, _encoding, callback) { this.chunks.push(Buffer.from(chunk)); callback(); }
  writeHead(code, headers = {}) { this.statusCode = code; this.headers = headers; return this; }
  json() { return JSON.parse(Buffer.concat(this.chunks).toString("utf8") || "{}"); }
}

function request(path, { address = "127.0.0.1", token = "secret", body = {}, headers = {} } = {}) {
  const data = body === null ? [] : [Buffer.from(typeof body === "string" ? body : JSON.stringify(body))];
  const req = Readable.from(data);
  req.url = path;
  req.method = "POST";
  req.headers = {
    authorization: `Bearer ${token}`,
    "content-type": "application/json",
    ...headers,
  };
  req.socket = { remoteAddress: address };
  return req;
}

function route() {
  let registered;
  const ctx = { webServer: { register(value) { registered = value; return () => {}; } } };
  registerRuntimeRoutes(ctx, {
    async startRun(body) { return { accepted: true, run_state_revision: body.run_state_revision }; },
  }, { runtimeToken: "secret", maxBodyBytes: 128 });
  return registered;
}

test("runtime API is loopback and bearer-token only", async () => {
  for (const options of [
    { address: "10.0.0.2" },
    { token: "wrong" },
    { headers: { origin: "https://attacker.example" } },
    { headers: { "sec-fetch-site": "cross-site" } },
  ]) {
    const res = new Response();
    await route().handler(request("/api/ecology-agent-runtime/v1/runs/start", options), res);
    assert.ok([401, 403].includes(res.statusCode));
    assert.doesNotMatch(Buffer.concat(res.chunks).toString(), /secret/);
  }
});

test("runtime API rejects overflow, unknown fields, and conflated revisions", async () => {
  const valid = {
    run_id: "run:1", run_state_revision: 2, stage_attempt: 1,
    ledger_expected_revision: 3, idempotency_key: "idem:1",
  };
  for (const body of [
    { ...valid, unknown: true },
    { ...valid, stage_attempt: undefined },
    "x".repeat(200),
  ]) {
    const res = new Response();
    await route().handler(request("/api/ecology-agent-runtime/v1/runs/start", { body }), res);
    assert.equal(res.statusCode, 400);
  }
  const res = new Response();
  await route().handler(request("/api/ecology-agent-runtime/v1/runs/start", { body: valid }), res);
  assert.equal(res.statusCode, 200);
  assert.equal(res.json().run_state_revision, 2);
});

test("runtime prefix cannot proxy arbitrary methods or paths", async () => {
  const res = new Response();
  const req = request("/api/ecology-agent-runtime/v1/private", { body: {} });
  req.method = "DELETE";
  await route().handler(req, res);
  assert.equal(res.statusCode, 404);
});
