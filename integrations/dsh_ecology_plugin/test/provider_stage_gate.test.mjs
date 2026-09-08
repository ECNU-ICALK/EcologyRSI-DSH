import assert from "node:assert/strict";
import test from "node:test";

import { ProviderStageGate } from "../lib/runtime/provider-stage-gate.js";

test("provider stage gate admits the configured 128-request provider window immediately", async () => {
  const gate = new ProviderStageGate({ minimumIntervalMs: 0 });
  let active = 0;
  let maximum = 0;
  const releases = [];
  const jobs = Array.from({ length: 129 }, (_, index) => gate.run(
    "pjlab",
    async () => {
      active += 1;
      maximum = Math.max(maximum, active);
      await new Promise((resolve) => { releases[index] = resolve; });
      active -= 1;
      return index;
    },
    { runId: "run-128" },
  ));

  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(maximum, 128);
  assert.equal(releases.filter(Boolean).length, 128);
  assert.deepEqual(gate.snapshot("pjlab"), {
    maxInFlight: 128,
    effectiveMaxInFlight: 128,
    effectiveMinimumIntervalMs: 0,
    active: 128,
    queued: 1,
    draining: 0,
    planned: 129,
    lifecycle_counts: {
      provider_queued: 1,
      provider_active: 128,
      draining: 0,
      planned: 129,
    },
    cooldownRemainingMs: 0,
  });

  for (let index = 0; index < 128; index += 1) releases[index]();
  await new Promise((resolve) => setImmediate(resolve));
  releases[128]();
  await Promise.all(jobs);

  assert.throws(
    () => new ProviderStageGate({ maxInFlight: 129 }),
    /maxInFlight must be between 1 and 128/,
  );
  assert.throws(
    () => new ProviderStageGate({ adaptiveInitialInFlight: 129 }),
    /maxInFlight must be between 1 and 128/,
  );
});

test("provider stage gate does not learn burst capacity from idle successes", async () => {
  const gate = new ProviderStageGate({ minimumIntervalMs: 0 });
  for (let index = 0; index < 100; index += 1) {
    await gate.run("pjlab", async () => index, { runId: "run-idle" });
  }
  assert.equal(gate.snapshot("pjlab").effectiveMaxInFlight, 128);
});

test("provider stage gate does not queue a default 64-origin Host wave", async () => {
  const gate = new ProviderStageGate({ minimumIntervalMs: 0 });
  let active = 0;
  let maximum = 0;
  let release;
  const hold = new Promise((resolve) => { release = resolve; });
  const jobs = Array.from({ length: 64 }, () => gate.run(
    "pjlab",
    async () => {
      active += 1;
      maximum = Math.max(maximum, active);
      await hold;
      active -= 1;
    },
    { runId: "run-host-64" },
  ));

  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(maximum, 64);
  assert.equal(gate.snapshot("pjlab").active, 64);
  assert.equal(gate.snapshot("pjlab").queued, 0);
  release();
  await Promise.all(jobs);
});

test("two default 64-origin runs share one 128-request provider ceiling", async () => {
  const gate = new ProviderStageGate({ minimumIntervalMs: 0 });
  let active = 0;
  let maximum = 0;
  let release;
  const hold = new Promise((resolve) => { release = resolve; });
  const jobs = ["run-a", "run-b"].flatMap((runId) => (
    Array.from({ length: 64 }, () => gate.run(
      "pjlab",
      async () => {
        active += 1;
        maximum = Math.max(maximum, active);
        await hold;
        active -= 1;
      },
      { runId },
    ))
  ));

  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(maximum, 128);
  assert.equal(gate.snapshot("pjlab").active, 128);
  assert.equal(gate.snapshot("pjlab").queued, 0);
  release();
  await Promise.all(jobs);
});

test("provider stage gate enforces the configured provider-wide concurrency", async () => {
  const gate = new ProviderStageGate({ minimumIntervalMs: 0, maxInFlight: 2 });
  let releaseFirst;
  let releaseSecond;
  let active = 0;
  let maximum = 0;
  const first = gate.run("pjlab", async () => {
    active += 1;
    maximum = Math.max(maximum, active);
    await new Promise((resolve) => { releaseFirst = resolve; });
    active -= 1;
    return "first";
  }, { runId: "run-one" });
  await new Promise((resolve) => setImmediate(resolve));
  const second = gate.run("pjlab", async () => {
    active += 1;
    maximum = Math.max(maximum, active);
    await new Promise((resolve) => { releaseSecond = resolve; });
    active -= 1;
    return "second";
  }, { runId: "run-one" });
  let thirdStarted = false;
  const third = gate.run("pjlab", async () => {
    thirdStarted = true;
    return "third";
  }, { runId: "run-two" });

  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(maximum, 2);
  assert.equal(thirdStarted, false);
  releaseFirst();
  assert.equal(await third, "third");
  releaseSecond();
  assert.deepEqual(await Promise.all([first, second]), ["first", "second"]);
  assert.equal(maximum, 2);
});

test("provider stage gate allows independent runs on the same provider", async () => {
  const gate = new ProviderStageGate({ minimumIntervalMs: 0, maxInFlight: 2 });
  let releaseFirst;
  let active = 0;
  let maximum = 0;
  const first = gate.run("pjlab", async () => {
    active += 1;
    maximum = Math.max(maximum, active);
    await new Promise((resolve) => { releaseFirst = resolve; });
    active -= 1;
  }, { runId: "run-one" });
  await new Promise((resolve) => setImmediate(resolve));

  const second = gate.run("pjlab", async () => {
    active += 1;
    maximum = Math.max(maximum, active);
    active -= 1;
  }, { runId: "run-two" });

  await second;
  assert.equal(maximum, 2);
  releaseFirst();
  await first;
});

test("successful work does not add a completion-time cooldown", async () => {
  let now = 1_000;
  const waits = [];
  const gate = new ProviderStageGate({
    minimumIntervalMs: 10,
    maxInFlight: 1,
    now: () => now,
    delay: async (milliseconds) => {
      waits.push(milliseconds);
      now += milliseconds;
    },
  });

  await gate.run("pjlab", async () => { now += 100; }, { runId: "run-one" });
  await gate.run("pjlab", async () => {}, { runId: "run-one" });

  assert.deepEqual(waits, []);
});

test("provider stage gate applies a bounded cooldown after a failed turn", async () => {
  let now = 1_000;
  const waits = [];
  const gate = new ProviderStageGate({
    minimumIntervalMs: 10,
    maxInFlight: 1,
    failureCooldownMs: 30,
    now: () => now,
    delay: async (milliseconds) => {
      waits.push(milliseconds);
      now += milliseconds;
    },
  });

  await gate.run("pjlab", async () => {
    gate.penalize("pjlab");
  }, { runId: "run-one" });
  await gate.run("pjlab", async () => {}, { runId: "run-two" });

  assert.deepEqual(waits, [30]);
});

test("a Retry-After extends an in-progress spacing wait without one early start", async () => {
  let now = 1_000;
  const waits = [];
  let gate;
  gate = new ProviderStageGate({
    minimumIntervalMs: 10,
    maxInFlight: 1,
    now: () => now,
    delay: async (milliseconds) => {
      waits.push(milliseconds);
      if (waits.length === 1) {
        gate.penalize("pjlab", 100, { reduceConcurrency: false });
      }
      now += milliseconds;
    },
  });

  await gate.run("pjlab", async () => {}, { runId: "run-first" });
  let secondStartedAt = null;
  await gate.run("pjlab", async () => { secondStartedAt = now; }, {
    runId: "run-second",
  });

  assert.deepEqual(waits, [10, 990]);
  assert.equal(secondStartedAt, 2_000);
});

test("an RPM cooldown does not rewrite the provider concurrency capacity", async () => {
  const gate = new ProviderStageGate({
    minimumIntervalMs: 0,
    maxInFlight: 8,
    adaptiveFloor: 2,
  });

  gate.penalize("pjlab", 30_000, { reduceConcurrency: false });

  assert.equal(gate.snapshot("pjlab").effectiveMaxInFlight, 8);
  assert.ok(gate.snapshot("pjlab").cooldownRemainingMs > 0);
});

test("provider stage gate reduces burst concurrency once per cooldown and recovers slowly", async () => {
  let now = 1_000;
  const gate = new ProviderStageGate({
    minimumIntervalMs: 0,
    failureCooldownMs: 30,
    maxInFlight: 8,
    adaptiveFloor: 2,
    adaptiveRecoverySuccesses: 2,
    now: () => now,
    delay: async (milliseconds) => { now += milliseconds; },
  });
  let release;
  const hold = new Promise((resolve) => { release = resolve; });
  const active = Array.from({ length: 8 }, () => gate.run(
    "pjlab",
    async () => { await hold; },
    { runId: "run-adaptive" },
  ));
  await new Promise((resolve) => setImmediate(resolve));

  gate.penalize("pjlab");
  assert.equal(gate.snapshot("pjlab").effectiveMaxInFlight, 4);
  gate.penalize("pjlab");
  assert.equal(gate.snapshot("pjlab").effectiveMaxInFlight, 4);

  release();
  await Promise.all(active);
  now += 30;
  const releases = [];
  const recovery = Array.from({ length: 9 }, (_, index) => gate.run(
    "pjlab",
    () => new Promise((resolve) => { releases[index] = resolve; }),
    { runId: "run-adaptive" },
  ));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(releases.filter(Boolean).length, 4);
  for (let index = 0; index < 4; index += 1) {
    releases[index]();
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.equal(gate.snapshot("pjlab").effectiveMaxInFlight, 5);
  for (let index = 4; index < releases.length; index += 1) releases[index]();
  await Promise.all(recovery);
});

test("provider stage gate remembers a failed congestion point before probing again", async () => {
  let now = 1_000;
  const gate = new ProviderStageGate({
    minimumIntervalMs: 0,
    failureCooldownMs: 0,
    maxInFlight: 8,
    adaptiveFloor: 2,
    adaptiveRecoverySuccesses: 1,
    adaptiveProbeCooldownMs: 30,
    now: () => now,
  });
  let releaseInitial;
  const initialHold = new Promise((resolve) => { releaseInitial = resolve; });
  const initial = Array.from({ length: 8 }, () => gate.run(
    "pjlab",
    () => initialHold,
    { runId: "run-congestion-memory" },
  ));
  await new Promise((resolve) => setImmediate(resolve));
  gate.penalize("pjlab");
  assert.equal(gate.snapshot("pjlab").effectiveMaxInFlight, 4);
  releaseInitial();
  await Promise.all(initial);

  const releases = [];
  let draining = false;
  const recovery = Array.from({ length: 50 }, (_, index) => gate.run(
    "pjlab",
    () => new Promise((resolve) => {
      releases[index] = resolve;
      if (draining) resolve();
    }),
    { runId: "run-congestion-memory" },
  ));
  await new Promise((resolve) => setImmediate(resolve));

  // Four, five, then six saturated successes restore 4 -> 5 -> 6 -> 7.
  for (let index = 0; index < 15; index += 1) {
    releases[index]();
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.equal(gate.snapshot("pjlab").effectiveMaxInFlight, 7);

  // More sustained success cannot immediately retry the failed limit of 8.
  for (let index = 15; index < 29; index += 1) {
    releases[index]();
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.equal(gate.snapshot("pjlab").effectiveMaxInFlight, 7);

  now += 30;
  for (let index = 29; index < 36; index += 1) {
    releases[index]();
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.equal(gate.snapshot("pjlab").effectiveMaxInFlight, 8);
  draining = true;
  for (let index = 36; index < releases.length; index += 1) releases[index]();
  await Promise.all(recovery);
});

test("provider stage gate revokes a run waiting for its own interval", async () => {
  let now = 1_000;
  const gate = new ProviderStageGate({
    minimumIntervalMs: 10,
    maxInFlight: 1,
    now: () => now,
    delay: () => new Promise(() => {}),
  });
  await gate.run("pjlab", async () => {}, { runId: "run-target" });
  let targetStarted = false;
  const target = gate.run(
    "pjlab",
    async () => { targetStarted = true; },
    { runId: "run-target" },
  );
  await new Promise((resolve) => setImmediate(resolve));

  gate.cancelRun("run-target");
  await gate.drainRun("run-target");
  await assert.rejects(
    target,
    (error) => error.code === "provider_stage_admission_closed",
  );
  assert.equal(targetStarted, false);
});

test("a closed provider launch fence rejects future work until explicitly reopened", async () => {
  const gate = new ProviderStageGate({ minimumIntervalMs: 0 });
  let operations = 0;

  gate.closeRun("run-target");
  await gate.drainRun("run-target");
  await assert.rejects(
    gate.run(
      "pjlab",
      async () => { operations += 1; },
      { runId: "run-target" },
    ),
    (error) => error.code === "provider_stage_admission_closed",
  );
  assert.equal(operations, 0);

  gate.openRun("run-target");
  await gate.run(
    "pjlab",
    async () => { operations += 1; },
    { runId: "run-target" },
  );
  assert.equal(operations, 1);
});

test("provider queue wait is bounded by the stage deadline", async () => {
  const gate = new ProviderStageGate({ maxInFlight: 1, minimumIntervalMs: 0 });
  let release;
  const first = gate.run("pjlab", () => new Promise((resolve) => { release = resolve; }), {
    runId: "run-timeout",
  });
  await new Promise((resolve) => setImmediate(resolve));
  let started = false;
  const second = gate.run("pjlab", () => {
    started = true;
    return "unexpected";
  }, {
    runId: "run-timeout",
    deadline: { deadlineAt: performance.now() + 20, timeoutMs: 20 },
  });
  const outcome = await Promise.race([
    second.then(() => ({ status: "fulfilled" }), (error) => ({ status: "rejected", error })),
    new Promise((resolve) => setTimeout(() => resolve({ status: "guard" }), 100)),
  ]);
  release();
  await first;
  assert.equal(outcome.status, "rejected");
  assert.equal(outcome.error?.code, "provider_queue_timeout");
  assert.equal(started, false);
});

test("active cancellation forwards AbortSignal and drain reports completion", async () => {
  const gate = new ProviderStageGate({ maxInFlight: 1, minimumIntervalMs: 0 });
  let resolveStarted;
  let receivedSignal = false;
  const started = new Promise((resolve) => { resolveStarted = resolve; });
  const run = gate.run("pjlab", (signal) => new Promise((resolveOperation) => {
    receivedSignal = signal instanceof AbortSignal;
    resolveStarted();
    signal?.addEventListener("abort", () => resolveOperation("aborted"), { once: true });
  }), { runId: "run-cancel" });
  run.catch(() => {});
  await started;
  gate.cancelRun("run-cancel");
  const drain = await gate.drainRun("run-cancel", { timeoutMs: 100 });
  assert.equal(receivedSignal, true);
  assert.equal(drain.status, "drained");
  assert.equal(drain.remaining, 0);
});

test("drainRun returns a bounded deadline status for non-interruptible work", async () => {
  const gate = new ProviderStageGate({ maxInFlight: 1, minimumIntervalMs: 0 });
  const never = gate.run("pjlab", () => new Promise((resolve) => setTimeout(resolve, 100)), {
    runId: "run-draining",
  });
  await new Promise((resolve) => setImmediate(resolve));
  const drain = await gate.drainRun("run-draining", { timeoutMs: 10 });
  assert.equal(drain.status, "deadline_exceeded");
  assert.equal(drain.remaining, 1);
  gate.cancelRun("run-draining");
  await never.catch(() => {});
});
