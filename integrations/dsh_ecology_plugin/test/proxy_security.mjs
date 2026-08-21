import assert from "node:assert/strict";
import { createServer } from "node:http";
import { Readable, Writable } from "node:stream";

import { apply } from "../lib/index.js";

class TestResponse extends Writable {
  constructor() {
    super();
    this.body = [];
    this.headers = {};
    this.headersSent = false;
    this.statusCode = null;
  }

  _write(chunk, _encoding, callback) {
    this.body.push(Buffer.from(chunk));
    callback();
  }

  writeHead(statusCode, headers = {}) {
    this.statusCode = statusCode;
    this.headers = headers;
    this.headersSent = true;
    return this;
  }
}

function registeredRoutes(config = {}) {
  const registrations = [];
  const context = {
    webServer: {
      register(route) {
        registrations.push(route);
        return () => {};
      },
    },
    effect(register) {
      register();
    },
  };
  apply(context, config);
  return registrations;
}

function request(url, { method = "GET", headers = {}, body = null } = {}) {
  const stream = Readable.from(body == null ? [] : [Buffer.from(body)]);
  stream.url = url;
  stream.method = method;
  stream.headers = headers;
  return stream;
}

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
}

function close(server) {
  return new Promise((resolve, reject) => {
    server.close((error) => error ? reject(error) : resolve());
  });
}

for (const backendOrigin of [
  "https://127.0.0.1:8777",
  "http://example.com:8777",
  "http://user:password@127.0.0.1:8777",
  "http://127.0.0.1:8777/private",
  "http://127.0.0.1:8777/?query=1",
]) {
  assert.throws(
    () => registeredRoutes({ backendOrigin }),
    /loopback HTTP backend|loopback host|must be an origin/,
  );
}

const invalidRoutes = registeredRoutes({ backendOrigin: "http://127.0.0.1:8777" });
const invalidApi = invalidRoutes.find((route) => route.path === "/api/ecology-evolution");
for (const url of [
  "http://attacker.example/api/ecology-evolution/health",
  "//attacker.example/api/ecology-evolution/health",
  "/api/ecology-evolution/../private",
  "/api/other",
]) {
  const response = new TestResponse();
  await invalidApi.handler(request(url), response);
  assert.equal(response.statusCode, 400, url);
}

const malformedStatic = invalidRoutes.find((route) => route.path === "/plugins/ecology/evolution");
const malformedResponse = new TestResponse();
await malformedStatic.handler(
  request("/plugins/ecology/evolution/%E0%A4%A"),
  malformedResponse,
);
assert.equal(malformedResponse.statusCode, 404);

let captured;
const upstreamServer = createServer((upstreamRequest, upstreamResponse) => {
  const chunks = [];
  upstreamRequest.on("data", (chunk) => chunks.push(chunk));
  upstreamRequest.on("end", () => {
    captured = {
      body: Buffer.concat(chunks).toString("utf8"),
      headers: upstreamRequest.headers,
      method: upstreamRequest.method,
      url: upstreamRequest.url,
    };
    const body = upstreamRequest.url === "/api/ecology-evolution/runs/large"
      ? Buffer.alloc(3 * 1024 * 1024, "x")
      : Buffer.from("{}");
    upstreamResponse.writeHead(200, {
      "content-type": "application/json",
      "content-length": String(body.length),
      "set-cookie": "sidecar-cookie=must-not-escape",
      "x-sidecar-result": "ok",
    });
    upstreamResponse.end(body);
  });
});
const upstreamPort = await listen(upstreamServer);
try {
  const routes = registeredRoutes({
    backendOrigin: `http://127.0.0.1:${upstreamPort}`,
    serviceToken: "test-service-token",
  });
  const api = routes.find((route) => route.path === "/api/ecology-evolution");
  const response = new TestResponse();
  await api.handler(request(
    "/api/ecology-evolution/runs?include_archived=true",
    {
      method: "POST",
      headers: {
        authorization: "Bearer browser-token",
        cookie: "dsh-session=must-not-leak",
        "content-length": "2",
        "x-dsh-request": "kept",
      },
      body: "{}",
    },
  ), response);

  assert.equal(captured.method, "POST");
  assert.equal(captured.url, "/api/ecology-evolution/runs?include_archived=true");
  assert.equal(captured.body, "{}");
  assert.equal(captured.headers.authorization, "Bearer test-service-token");
  assert.equal(captured.headers.cookie, undefined);
  assert.equal(captured.headers["x-dsh-request"], "kept");
  assert.equal(response.statusCode, 200);
  assert.equal(response.headers["set-cookie"], undefined);
  assert.equal(response.headers["x-sidecar-result"], "ok");

  const largeResponse = new TestResponse();
  await api.handler(
    request("/api/ecology-evolution/runs/large"),
    largeResponse,
  );
  assert.equal(largeResponse.statusCode, 200);
  assert.equal(Buffer.concat(largeResponse.body).length, 3 * 1024 * 1024);
} finally {
  await close(upstreamServer);
}

console.log("dsh proxy security test: ok");
