import assert from "node:assert/strict";
import { Readable, Writable } from "node:stream";
import { createServer } from "node:http";
import { once } from "node:events";
import test from "node:test";

import { registerRuntimeRoutes } from "../lib/runtime/routes.js";
import { structuredPhaseError } from "../lib/runtime/structured-stage-errors.js";

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

function route(controller = {
  async startRun(body) { return { accepted: true, run_state_revision: body.run_state_revision }; },
}, maxBodyBytes = 128) {
  let registered;
  const ctx = { webServer: { register(value) { registered = value; return () => {}; } } };
  registerRuntimeRoutes(ctx, controller, { runtimeToken: "secret", maxBodyBytes });
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

test("runtime API distinguishes bounded sample failure from runtime outage", async () => {
  const valid = {
    run_id: "run:1", run_state_revision: 2, stage_attempt: 1,
    ledger_expected_revision: 3, idempotency_key: "idem:1",
  };
  for (const code of [
    "structured_child_model_error",
    "structured_child_tool_protocol_error",
    "structured_child_output_budget_exhausted",
    "structured_child_output_schema_invalid",
    "structured_result_missing",
  ]) {
    const res = new Response();
    await route({
      async startRun() {
        const error = new Error("private provider detail");
        error.code = code;
        throw error;
      },
    }).handler(request("/api/ecology-agent-runtime/v1/runs/start", { body: valid }), res);
    assert.equal(res.statusCode, 422);
    assert.deepEqual(res.json(), { error: "runtime_stage_failed", error_code: code });
    assert.doesNotMatch(Buffer.concat(res.chunks).toString(), /private provider detail/);
  }

  const persistence = new Response();
  await route({
    async startRun() {
      const error = new Error("private persistence detail");
      error.code = "structured_result_persist_failed";
      throw error;
    },
  }).handler(request("/api/ecology-agent-runtime/v1/runs/start", { body: valid }), persistence);
  assert.equal(persistence.statusCode, 502);
  assert.deepEqual(persistence.json(), {
    error: "runtime_controller_failed",
    error_code: "dsh_native_runtime_unavailable",
  });
  assert.doesNotMatch(Buffer.concat(persistence.chunks).toString(), /private persistence detail/);

  const res = new Response();
  await route({
    async startRun() { throw new Error("private runtime detail"); },
  }).handler(request("/api/ecology-agent-runtime/v1/runs/start", { body: valid }), res);
  assert.equal(res.statusCode, 502);
  assert.deepEqual(res.json(), {
    error: "runtime_controller_failed",
    error_code: "dsh_native_runtime_unavailable",
  });
  assert.doesNotMatch(Buffer.concat(res.chunks).toString(), /private runtime detail/);
});

test("real HTTP structured POST preserves schema failure code without exposing private causes", async () => {
  let knownSchemaFailure = true;
  let calls = 0;
  const registered = route({
    async runStage(body) {
      calls += 1;
      assert.equal(body.stage, "generation.research-synthesis");
      const cause = new Error("private_api_key=fake-private-value; private schema details");
      if (knownSchemaFailure) throw structuredPhaseError("output_schema", cause);
      throw cause;
    },
  }, 4096);
  const server = createServer(registered.handler);
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  try {
    const body = {
      run_id: "run:schema-http", run_state_revision: 2, stage_attempt: 1,
      ledger_expected_revision: 3, idempotency_key: "schema-http",
      stage: "generation.research-synthesis", admission_id: "admission-schema-http", request: {},
    };
    const url = `http://127.0.0.1:${server.address().port}/api/ecology-agent-runtime/v1/runs/run%3Aschema-http/stages`;
    for (const expectedStatus of [422, 502]) {
      const response = await fetch(url, { method: "POST",
        headers: { authorization: "Bearer secret", "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      const text = await response.text();
      assert.equal(response.status, expectedStatus);
      assert.deepEqual(JSON.parse(text), knownSchemaFailure
        ? { error: "runtime_stage_failed", error_code: "structured_child_output_schema_invalid",
          schema_version: "ecology-runtime-failure/1", failure_domain: "execution_contract",
          retryable: false, provider_status: null, retry_after_ms: null, affected_scope: "stage" }
        : { error: "runtime_controller_failed", error_code: "dsh_native_runtime_unavailable" });
      assert.doesNotMatch(text, /private|fake-private-value|schema details|secret/);
      knownSchemaFailure = false;
    }
    assert.equal(calls, 2);
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
});

test("trusted provider failures cross HTTP with status and bounded recovery metadata", async () => {
  const body = { run_id: "run:1", run_state_revision: 2, stage_attempt: 1,
    ledger_expected_revision: 3, idempotency_key: "idem:1" };
  for (const status of [429, 503]) {
    const res = new Response();
    await route({ async startRun() {
      throw structuredPhaseError("model", new Error("private token"),
        { providerStatus: status, retryAfterMs: 17000 });
    } }).handler(request("/api/ecology-agent-runtime/v1/runs/start", { body }), res);
    assert.equal(res.statusCode, status);
    assert.equal(res.json().provider_status, status);
    assert.equal(res.json().retry_after_ms, 17000);
    assert.equal(res.json().retryable, true);
    assert.doesNotMatch(JSON.stringify(res.json()), /private|token/);
  }
});

test("model canary route reuses loopback browser and bearer boundaries", async () => {
  let called = 0;
  const controller = { async runCanary(body) { called += 1; return { passed: true, scope: "tool_and_schema_transport_only", identity: body.identity }; } };
  for (const options of [{address:"10.0.0.2"},{token:"wrong"},{headers:{origin:"https://attacker.example"}},{headers:{"sec-fetch-site":"cross-site"}}]) {
    const res=new Response();await route(controller).handler(request("/api/ecology-agent-runtime/v1/canaries",options),res);
    assert.ok([401,403].includes(res.statusCode));
  }
  assert.equal(called,0);
  const res=new Response();await route(controller).handler(request("/api/ecology-agent-runtime/v1/canaries",{body:{identity:{role:"researcher"}}}),res);
  assert.equal(res.statusCode,200);assert.equal(called,1);
});

test("canary API failures never echo arbitrary runtime diagnostics",async()=>{
  const res=new Response();await route({async runCanary(){const e=new Error("secret credential");e.code="secret_token";throw e;}})
    .handler(request("/api/ecology-agent-runtime/v1/canaries"),res);
  assert.equal(res.statusCode,422);assert.equal(res.json().error_code,"model_canary_failed");
  assert.doesNotMatch(JSON.stringify(res.json()),/secret/);
});
