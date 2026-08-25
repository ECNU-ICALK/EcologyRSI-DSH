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

  const configured = resolvePluginConfig({
    researchStageTimeoutMs: 900_000,
  }, {
    defaultStaticRoot: "/tmp/ecologyrsi-static",
    env: {},
  });
  assert.equal(configured.researchStageTimeoutMs, 900_000);
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
