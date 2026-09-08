import assert from "node:assert/strict";
import test from "node:test";

import { ProviderStageGate } from "../lib/runtime/provider-stage-gate.js";

const tick = () => new Promise((resolve) => setImmediate(resolve));

for (const cancellation of ["external", "run", "already-aborted"]) {
  test(`queued ${cancellation} cancellation settles while another run occupies the provider`, async () => {
    const gate = new ProviderStageGate({ maxInFlight: 1 });
    let release;
    const active = gate.run("provider", () => new Promise((resolve) => { release = resolve; }), {
      runId: "run:active",
    });
    await tick();
    const controller = new AbortController();
    if (cancellation === "already-aborted") controller.abort(new Error("caller cancelled"));
    let started = false;
    let outcome = null;
    const queued = gate.run("provider", () => { started = true; }, {
      runId: "run:queued",
      signal: controller.signal,
    }).then(
      () => { outcome = "fulfilled"; },
      () => { outcome = "cancelled"; },
    );
    if (cancellation === "external") controller.abort(new Error("caller cancelled"));
    if (cancellation === "run") gate.cancelRun("run:queued");
    await tick();
    try {
      assert.equal(outcome, "cancelled");
      assert.equal(started, false);
      assert.equal(gate.snapshot("provider").queued, 0);
      assert.equal(gate.snapshot("provider").active, 1);
      assert.deepEqual(await gate.drainRun("run:queued", { timeoutMs: 1 }), {
        status: "drained", remaining: 0,
      });
    } finally {
      release();
      await Promise.all([active, queued]);
    }
  });
}

test("completed spacing waits leave no deadline timers behind", async (t) => {
  const realSetTimeout = globalThis.setTimeout;
  const realClearTimeout = globalThis.clearTimeout;
  const timers = new Set();
  t.mock.method(globalThis, "setTimeout", (callback, milliseconds, ...args) => {
    const timer = realSetTimeout(() => {
      timers.delete(timer);
      callback(...args);
    }, milliseconds);
    timers.add(timer);
    return timer;
  });
  t.mock.method(globalThis, "clearTimeout", (timer) => {
    timers.delete(timer);
    return realClearTimeout(timer);
  });
  let now = Date.now();
  const gate = new ProviderStageGate({
    minimumIntervalMs: 10,
    maxInFlight: 1,
    now: () => now,
    delay: async (milliseconds) => { now += milliseconds; },
  });
  const deadline = { deadlineAt: now + 60_000 };
  try {
    for (let index = 0; index < 3; index += 1) {
      await gate.run("provider", async () => index, { runId: "run:spacing", deadline });
    }
    await tick();
    assert.equal(timers.size, 0, "all queue and active deadlines must be cancelled after completion");
  } finally {
    for (const timer of timers) realClearTimeout(timer);
  }
});

test("cancelling active work keeps its physical slot until it settles", async () => {
  const gate = new ProviderStageGate({ maxInFlight: 1 });
  let release;
  const active = gate.run("provider", () => new Promise((resolve) => { release = resolve; }), {
    runId: "run:active",
  }).catch(() => {});
  await tick();
  gate.cancelRun("run:active");
  let nextStarted = false;
  const next = gate.run("provider", () => { nextStarted = true; }, { runId: "run:next" });
  await tick();
  assert.equal(nextStarted, false);
  assert.equal(gate.snapshot("provider").active, 1);
  assert.equal(gate.snapshot("provider").draining, 1);
  assert.equal(gate.snapshot("provider").lifecycle_counts.provider_active, 0);
  release();
  await Promise.all([active, next]);
  assert.equal(nextStarted, true);
  assert.equal(gate.snapshot("provider").active, 0);
});

test("an RPM burst slows future starts across runs without reducing physical concurrency", async () => {
  let now = 0;
  const starts = [];
  const gate = new ProviderStageGate({ minimumIntervalMs: 3000, maxInFlight: 64,
    now: () => now, delay: async milliseconds => { now += milliseconds; } });
  gate.penalize('pjlab', 13000, { reduceConcurrency: false });
  for (let i = 0; i < 20; i++) gate.penalize('pjlab', 3000, { reduceConcurrency: false });
  assert.equal(gate.snapshot('pjlab').effectiveMinimumIntervalMs, 6000);
  assert.equal(gate.snapshot('pjlab').effectiveMaxInFlight, 64);
  for (const runId of ['one', 'two', 'one']) await gate.run('pjlab', () => starts.push(now), {runId});
  assert.deepEqual(starts, [13000, 19000, 25000]);
  assert.equal(gate.snapshot('other').effectiveMinimumIntervalMs, 3000);
  now = 60000;
  gate.penalize('pjlab', 3000, { reduceConcurrency: false });
  assert.equal(gate.snapshot('pjlab').effectiveMinimumIntervalMs, 12000);
  assert.equal(gate.snapshot('pjlab').effectiveMaxInFlight, 64);
});
