import test from "node:test";
import assert from "node:assert/strict";
import { RouteHealth } from "../lib/runtime/route-health.js";
import { structuredPhaseError } from "../lib/runtime/structured-stage-errors.js";
import { NativeStageRunner } from "../lib/runtime/stage-runner.js";
import { resolvePluginConfig } from "../lib/config.js";

test("production request backoff cannot shorten route outage cooling", async () => {
  const config = resolvePluginConfig({}, { env: {} });
  const runner = new NativeStageRunner({}, { ...config, sidecar: {} });
  for (let i = 0; i < 3; i++) {
    await assert.rejects(runner.routeHealth.run("provider/a", () => {
      throw structuredPhaseError("model");
    }));
  }
  assert.ok(runner.routeHealth.snapshot("provider/a").retry_after_ms > 59_000);
});

test("route circuit stops dispatch, isolates models, and permits one half-open probe", async () => {
  let now = 0;
  const health = new RouteHealth({ now: () => now, cooldownMs: 100 });
  let calls = 0;
  const fail = () => { calls++; throw structuredPhaseError("model"); };
  for (let i = 0; i < 3; i++) await assert.rejects(health.run("provider/a", fail));
  await assert.rejects(health.run("provider/a", fail), { code: "provider_route_cooling_down" });
  assert.equal(calls, 3);
  assert.equal(await health.run("provider/b", () => 42), 42);
  now = 100;
  let resolve;
  const probe = health.run("provider/a", () => new Promise(r => { resolve = r; }));
  await assert.rejects(health.run("provider/a", fail), { code: "provider_route_cooling_down" });
  resolve(1);
  await probe;
  assert.equal(health.snapshot("provider/a").state, "closed");
});

test("late success cannot close an open route; malformed provider envelopes also cool down", async () => {
  const health = new RouteHealth();
  let resolve;
  const late = health.run("a", () => new Promise(r => { resolve = r; }));
  for (let i = 0; i < 3; i++) await assert.rejects(health.run("a", () => { throw structuredPhaseError("model"); }));
  resolve(1); await late;
  assert.equal(health.snapshot("a").state, "open");
  for (let i = 0; i < 3; i++) await assert.rejects(health.run("b", () => { throw structuredPhaseError("tool_protocol"); }));
  assert.equal(health.snapshot("b").state, "open");
  for (let i = 0; i < 3; i++) await assert.rejects(health.run("c", () => { throw structuredPhaseError("output_schema"); }));
  assert.equal(health.snapshot("c").state, "closed");
});
