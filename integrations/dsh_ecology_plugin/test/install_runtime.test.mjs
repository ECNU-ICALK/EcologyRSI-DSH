import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, realpath, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { installPresetTree, managedPatchText, resolveDshHome } from "../../../scripts/install_dsh_ecology_runtime.mjs";

const source = new URL("../presets/", import.meta.url);

test("preset installation is exact, idempotent, and refuses drift", async () => {
  const tmp = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-dsh-install-")));
  const dshHome = path.join(tmp, "dsh-home");
  await mkdir(dshHome);
  await installPresetTree({ sourceRoot: source, dshHome });
  await installPresetTree({ sourceRoot: source, dshHome });
  const target = path.join(dshHome, ".agent-presets", "ecology-researcher-v1", "preset.yml");
  assert.match(await readFile(target, "utf8"), /Ecology Researcher/);
  await writeFile(target, "drift\n");
  await assert.rejects(installPresetTree({ sourceRoot: source, dshHome }), /drift/);
});

test("DSH home resolver rejects a symlink target", async () => {
  const tmp = await mkdtemp(path.join(os.tmpdir(), "ecology-dsh-link-"));
  const actual = path.join(tmp, "actual");
  const linked = path.join(tmp, "linked");
  await mkdir(actual);
  await symlink(actual, linked);
  await assert.rejects(resolveDshHome({ env: { DSH_HOME: linked }, homeDir: tmp }), /symlink/);
});

test("managed Host patch has the exact DSH service injection and no embedded credentials", () => {
  const text = managedPatchText({ staticRoot: "/safe/static" });
  for (const name of [
    "webServer", "agents", "sessions", "tokenMeter", "subagents", "tools",
    "sessionPersistence", "sessionProjections", "agentPresets", "llm",
  ]) assert.match(text, new RegExp(`\\b${name}\\b`));
  assert.doesNotMatch(text, /credentials|serviceToken|runtimeToken|secret/i);
});
