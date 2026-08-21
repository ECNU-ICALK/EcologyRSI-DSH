import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("package exposes an isolated agent-plane entry and pins DSH rc dependencies", async () => {
  const pkg = JSON.parse(await readFile(new URL("package.json", root), "utf8"));
  assert.equal(pkg.exports["./agent-plugin"], "./lib/tools/agent-plugin.js");
  assert.ok(pkg.files.includes("presets/**/*.yml"));
  assert.deepEqual(pkg.dependencies, {});
  for (const [name, version] of Object.entries({
    ...pkg.dependencies,
    ...pkg.peerDependencies,
  })) {
    if (name.startsWith("@deepseek-ai/dsh-")) {
      assert.match(version, /^\d+\.\d+\.\d+-rc\.\d+$/);
      assert.equal(pkg.peerDependenciesMeta[name]?.optional, true);
    }
  }
  const module = await import("../lib/tools/agent-plugin.js");
  assert.equal(typeof module.apply, "function");
  assert.deepEqual(module.inject, ["tools", "ecologyAgentTools"]);
});
