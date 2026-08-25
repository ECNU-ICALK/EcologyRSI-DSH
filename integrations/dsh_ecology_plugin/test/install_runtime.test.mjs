import assert from "node:assert/strict";
import {
  access,
  chmod,
  cp,
  mkdtemp,
  mkdir,
  readFile,
  realpath,
  symlink,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  installManagedPatch,
  installPresetTree,
  managedPatchText,
  resolveDshHome,
} from "../../../scripts/install_dsh_ecology_runtime.mjs";

const source = new URL("../presets/", import.meta.url);

async function installedDshBin() {
  const candidate = process.env.DSH_BIN || path.join(
    os.homedir(),
    ".local",
    "share",
    "ecologyrsi-dsh-dsh-cli",
    "node_modules",
    ".bin",
    process.platform === "win32" ? "dsh.cmd" : "dsh",
  );
  try {
    await access(candidate);
    return candidate;
  } catch {
    return null;
  }
}

test("preset installation is exact, idempotent, and refuses drift", async () => {
  const tmp = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-dsh-install-")));
  const dshHome = path.join(tmp, "dsh-home");
  await mkdir(dshHome);
  const obsolete = [
    "ecology-researcher-v1",
    "ecology-researcher-v2",
    "ecology-researcher-v3",
    "ecology-generation-judge-v3",
    "ecology-researcher-v4",
    "ecology-generation-judge-v4",
    "ecology-researcher-v5",
    "ecology-generation-judge-v5",
  ].map(
    (id) => path.join(dshHome, ".agent-presets", id),
  );
  for (const target of obsolete) {
    await mkdir(target, { recursive: true });
    await writeFile(path.join(target, "preset.yml"), "obsolete\n");
  }
  await installPresetTree({ sourceRoot: source, dshHome });
  await installPresetTree({ sourceRoot: source, dshHome });
  for (const target of obsolete) {
    await assert.rejects(readFile(path.join(target, "preset.yml")), { code: "ENOENT" });
  }
  const target = path.join(dshHome, ".agent-presets", "ecology-researcher-v6", "preset.yml");
  assert.match(await readFile(target, "utf8"), /Ecology Researcher/);
  await writeFile(target, "drift\n");
  await assert.rejects(installPresetTree({ sourceRoot: source, dshHome }), /drift/);
});

test("preset installation rejects a composition that DSH cannot parse", async (t) => {
  const dshBin = await installedDshBin();
  if (dshBin == null) {
    t.skip("a local DSH CLI is required for parser parity");
    return;
  }
  const tmp = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-dsh-invalid-preset-")));
  const sourceRoot = path.join(tmp, "presets");
  const dshHome = path.join(tmp, "dsh-home");
  await cp(source, sourceRoot, { recursive: true });
  await mkdir(dshHome);
  await writeFile(
    path.join(sourceRoot, "ecology-researcher-v6", "agent.cordis.yml"),
    "- id: persona\n  name: '@deepseek-ai/dsh-persona'\n  config:\n    text: invalid plain scalar: parsed as a mapping\n",
  );

  await assert.rejects(
    installPresetTree({ sourceRoot, dshHome, dshBin }),
    /ecology-researcher-v6[\s\S]*not valid YAML|not valid YAML[\s\S]*ecology-researcher-v6/i,
  );
});

test("preset installation supports a DSH test double without packaged parser modules", async () => {
  const tmp = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-dsh-test-double-")));
  const dshBin = path.join(tmp, "fake-dsh");
  const dshHome = path.join(tmp, "dsh-home");
  await writeFile(dshBin, "#!/bin/sh\nexit 0\n");
  await chmod(dshBin, 0o755);
  await mkdir(dshHome);

  await installPresetTree({ sourceRoot: source, dshHome, dshBin });

  assert.match(
    await readFile(path.join(dshHome, ".agent-presets", "ecology-researcher-v6", "preset.yml"), "utf8"),
    /Ecology Researcher/,
  );
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

test("managed Host patch replaces a freshly initialized empty patch document", async () => {
  const tmp = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-dsh-patch-")));
  const profileRoot = path.join(tmp, "profiles", "web");
  const target = path.join(profileRoot, "cordis.patch.yml");
  await mkdir(profileRoot, { recursive: true });
  await writeFile(target, "# DSH generated profile overlay\n[]\n");

  await installManagedPatch({ dshHome: tmp, staticRoot: "/safe/static", profile: "web" });
  const first = await readFile(target, "utf8");
  assert.doesNotMatch(first, /^\s*\[\]\s*$/m);
  assert.match(first, /# BEGIN ECOLOGYRSI DSH RUNTIME/);
  assert.match(first, /- insert:/);

  await installManagedPatch({ dshHome: tmp, staticRoot: "/safe/static", profile: "web" });
  assert.equal(await readFile(target, "utf8"), first);
});
