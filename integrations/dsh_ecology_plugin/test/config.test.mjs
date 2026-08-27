import assert from "node:assert/strict";
import test from "node:test";

import { resolvePluginConfig } from "../lib/config.js";


test("research stages have an independent long-running timeout", () => {
  const defaults = resolvePluginConfig({}, {
    defaultStaticRoot: "/tmp/ecologyrsi-static",
    env: {},
  });
  assert.equal(defaults.structuredStageTimeoutMs, 600_000);
  assert.equal(defaults.researchStageTimeoutMs, 1_800_000);
  assert.equal(defaults.sampleCriticStageTimeoutMs, 600_000);
  assert.equal(defaults.structuredStageMinIntervalMs, 500);
  assert.equal(defaults.structuredStageMaxInFlight, 128);
  assert.equal(defaults.structuredStageFailureCooldownMs, 15_000);

  const configured = resolvePluginConfig({
    researchStageTimeoutMs: 900_000,
    sampleCriticStageTimeoutMs: 120_000,
  }, {
    defaultStaticRoot: "/tmp/ecologyrsi-static",
    env: {},
  });
  assert.equal(configured.researchStageTimeoutMs, 900_000);
  assert.equal(configured.sampleCriticStageTimeoutMs, 120_000);
});

test("provider admission accepts zero start interval and bounds concurrency", () => {
  const configured = resolvePluginConfig({
    structuredStageMinIntervalMs: 0,
    structuredStageMaxInFlight: 128,
  }, {
    defaultStaticRoot: "/tmp/ecologyrsi-static",
    env: {},
  });
  assert.equal(configured.structuredStageMinIntervalMs, 0);
  assert.equal(configured.structuredStageMaxInFlight, 128);
  for (const value of [0, 129]) {
    assert.throws(
      () => resolvePluginConfig({ structuredStageMaxInFlight: value }, {
        defaultStaticRoot: "/tmp/ecologyrsi-static",
        env: {},
      }),
      /structuredStageMaxInFlight must be between 1 and 128/,
    );
  }
});

test("sample critic stage timeout must be positive", () => {
  assert.throws(
    () => resolvePluginConfig({ sampleCriticStageTimeoutMs: 0 }, {
      defaultStaticRoot: "/tmp/ecologyrsi-static",
      env: {},
    }),
    /sampleCriticStageTimeoutMs must be positive/,
  );
});

test("research stage timeout must be positive", () => {
  assert.throws(
    () => resolvePluginConfig({ researchStageTimeoutMs: 0 }, {
      defaultStaticRoot: "/tmp/ecologyrsi-static",
      env: {},
    }),
    /researchStageTimeoutMs must be positive/,
  );
});

test("all structured stage timeouts respect the protocol ceiling", () => {
  for (const name of [
    "structuredStageTimeoutMs",
    "researchStageTimeoutMs",
    "sampleCriticStageTimeoutMs",
  ]) {
    for (const value of [1_800_001, 2_147_483_648]) {
      assert.throws(
        () => resolvePluginConfig({ [name]: value }, {
          defaultStaticRoot: "/tmp/ecologyrsi-static",
          env: {},
        }),
        new RegExp(`${name} must be at most 1800000`),
      );
    }
  }
});
