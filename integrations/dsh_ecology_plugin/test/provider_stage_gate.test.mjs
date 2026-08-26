import assert from "node:assert/strict";
import test from "node:test";

import { ProviderStageGate } from "../lib/runtime/provider-stage-gate.js";

test("provider stage gate defaults to and accepts exactly 128 in flight", async () => {
  const gate = new ProviderStageGate({ minimumIntervalMs: 0 });
  let active = 0;
  let maximum = 0;
  let release;
  const hold = new Promise((resolve) => { release = resolve; });
  const firstWave = Array.from({ length: 128 }, (_, index) => gate.run(
    "pjlab",
    async () => {
      active += 1;
      maximum = Math.max(maximum, active);
      await hold;
      active -= 1;
      return index;
    },
    { runId: "run-128" },
  ));
  let overflowStarted = false;
  const overflow = gate.run(
    "pjlab",
    async () => { overflowStarted = true; },
    { runId: "run-128" },
  );

  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(maximum, 128);
  assert.equal(overflowStarted, false);
  release();
  await Promise.all([...firstWave, overflow]);

  assert.throws(
    () => new ProviderStageGate({ maxInFlight: 129 }),
    /maxInFlight must be between 1 and 128/,
  );
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
