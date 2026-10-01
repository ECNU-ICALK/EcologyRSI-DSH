import assert from "node:assert/strict";
import {
  access,
  chmod,
  cp,
  mkdtemp,
  mkdir,
  readFile,
  readdir,
  rm,
  realpath,
  symlink,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import {
  PRESET_IDS,
  installManagedPatch,
  installPresetTree,
  managedPatchText,
  resolveDshHome,
  validatePluginArchive,
  installRuntime,
} from "../../../scripts/install_dsh_ecology_runtime.mjs";

const source = new URL("../presets/", import.meta.url);

test("bundled archive includes exact runtime and rejects missing visibility before changing a profile", async (t) => {
  const pluginRoot = fileURLToPath(new URL("../", import.meta.url));
  const manifest = JSON.parse(await readFile(path.join(pluginRoot, "package.json"), "utf8"));
  const archive = path.join(pluginRoot, "dist", `ecologyrsi-dsh-evolution-plugin-${manifest.version}.tgz`);
  await validatePluginArchive({ archive, pluginRoot });
  const tmp = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-archive-review-")));
  t.after(() => rm(tmp, { recursive: true, force: true }));
  assert.equal(spawnSync("tar", ["-xzf", archive, "-C", tmp]).status, 0);
  await rm(path.join(tmp, "package/lib/runtime/session-visibility.js"));
  const broken = path.join(tmp, "broken.tgz");
  assert.equal(spawnSync("tar", ["-czf", broken, "-C", tmp, "package"]).status, 0);
  const dshHome = path.join(tmp, "profile-must-stay-absent");
  await assert.rejects(installRuntime({ packageArchive: broken, dshHome, pluginRoot,
    projectRoot: path.resolve(pluginRoot, "../..") }), /missing package\/lib\/runtime\/session-visibility.js/);
  await assert.rejects(access(dshHome), { code: "ENOENT" });
});

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
  const stale = [
    "ecology-coordinator-v3",
    "ecology-researcher-v6",
    "ecology-candidate-proposer-v3",
    "ecology-sample-planner-v3",
    "ecology-sample-critic-v3",
    "ecology-generation-judge-v6",
    "ecology-local-editor-v1",
    "ecology-researcher-v129",
  ].map(
    (id) => path.join(dshHome, ".agent-presets", id),
  );
  for (const target of stale) {
    await mkdir(target, { recursive: true });
    await writeFile(path.join(target, "preset.yml"), "stale\n");
  }
  const unmanagedId = "ecology-user-specialist-v1";
  const unmanagedTarget = path.join(dshHome, ".agent-presets", unmanagedId);
  await mkdir(unmanagedTarget, { recursive: true });
  await writeFile(path.join(unmanagedTarget, "preset.yml"), "user managed\n");

  await installPresetTree({ sourceRoot: source, dshHome });
  await installPresetTree({ sourceRoot: source, dshHome });
  for (const target of stale) {
    await assert.rejects(readFile(path.join(target, "preset.yml")), { code: "ENOENT" });
  }
  assert.equal(await readFile(path.join(unmanagedTarget, "preset.yml"), "utf8"), "user managed\n");
  const installedIds = (await readdir(path.join(dshHome, ".agent-presets"), { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
  assert.deepEqual(installedIds, [...PRESET_IDS, unmanagedId].sort());

  const target = path.join(dshHome, ".agent-presets", "ecology-researcher-v13", "preset.yml");
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
    path.join(sourceRoot, "ecology-researcher-v13", "agent.cordis.yml"),
    "- id: persona\n  name: '@deepseek-ai/dsh-persona'\n  config:\n    text: invalid plain scalar: parsed as a mapping\n",
  );

  await assert.rejects(
    installPresetTree({ sourceRoot, dshHome, dshBin }),
    /ecology-researcher-v13[\s\S]*not valid YAML|not valid YAML[\s\S]*ecology-researcher-v13/i,
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
    await readFile(path.join(dshHome, ".agent-presets", "ecology-researcher-v13", "preset.yml"), "utf8"),
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
    "sessionPersistence", "sessionProjections", "agentPresets", "llm", "web",
  ]) assert.match(text, new RegExp(`\\b${name}\\b`));
  assert.doesNotMatch(text, /credentials|serviceToken|runtimeToken|secret/i);
  assert.match(text, /id: ecologyrsi-session-visibility/);
  assert.match(text, /name: '@ecologyrsi\/dsh-evolution-plugin\/session-visibility'/);
  assert.match(text, /inject: \[sessions, sessionPersistence\]/);
});

test("managed Host patch replaces a freshly initialized empty patch document", async () => {
  const tmp = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-dsh-patch-")));
  const profileRoot = path.join(tmp, "profiles", "web");
  const target = path.join(profileRoot, "cordis.patch.yml");
  await mkdir(profileRoot, { recursive: true });
  await writeFile(target, "# DSH generated profile overlay\n[]\n");

  await installManagedPatch({ dshHome: tmp, staticRoot: "/safe/static", profile: "web" });
  const first = await readFile(target, "utf8");
  assert.ok(first.includes(path.join(profileRoot, "node_modules", "@ecologyrsi",
    "dsh-evolution-plugin", "lib", "runtime", "session-visibility.js")));
  assert.doesNotMatch(first, /^\s*\[\]\s*$/m);
  assert.match(first, /# BEGIN ECOLOGYRSI DSH RUNTIME/);
  assert.match(first, /- insert:/);

  await installManagedPatch({ dshHome: tmp, staticRoot: "/safe/static", profile: "web" });
  assert.equal(await readFile(target, "utf8"), first);
});

test("validated runtime installation works fresh, upgrades an old profile, and is idempotent", async (t) => {
  const pluginRoot = fileURLToPath(new URL("../", import.meta.url));
  const manifest = JSON.parse(await readFile(path.join(pluginRoot, "package.json"), "utf8"));
  const archive = path.join(pluginRoot, "dist", `ecologyrsi-dsh-evolution-plugin-${manifest.version}.tgz`);
  const tmp = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-full-install-")));
  t.after(() => rm(tmp, { recursive: true, force: true }));
  const fakeDsh = path.join(tmp, "fake-dsh.cjs");
  await writeFile(fakeDsh, `#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const args = process.argv.slice(2);
if (args[0] === 'plugin') {
  const profile = args[args.indexOf('--profile') + 1];
  const target = path.join(process.cwd(), 'profiles', profile, 'node_modules', '@ecologyrsi', 'dsh-evolution-plugin');
  fs.mkdirSync(target, {recursive: true});
  execFileSync('tar', ['-xzf', args.at(-1).slice(5), '-C', target, '--strip-components', '1']);
}
`);
  await chmod(fakeDsh, 0o755);
  const previous = process.env.DSH_BIN;
  process.env.DSH_BIN = fakeDsh;
  t.after(() => { if (previous === undefined) delete process.env.DSH_BIN; else process.env.DSH_BIN = previous; });
  for (const scenario of ["fresh", "upgrade"]) {
    const dshHome = path.join(tmp, scenario);
    const installed = path.join(dshHome, "profiles/web/node_modules/@ecologyrsi/dsh-evolution-plugin");
    if (scenario === "upgrade") {
      await mkdir(installed, { recursive: true });
      await writeFile(path.join(installed, "package.json"), JSON.stringify({version: "0.7.10"}));
      await mkdir(path.join(dshHome, "plugin-cache/ecologyrsi"), {recursive: true});
      await writeFile(path.join(dshHome, "plugin-cache/ecologyrsi/old-version.tgz"), "old immutable cache");
    }
    const options = {projectRoot: path.resolve(pluginRoot, "../.."), pluginRoot,
      staticRoot: path.join(tmp, "static"), packageArchive: archive, dshHome};
    await installRuntime(options);
    const patchPath = path.join(dshHome, "profiles/web/cordis.patch.yml");
    const first = await readFile(patchPath, "utf8");
    await installRuntime(options);
    assert.equal(await readFile(patchPath, "utf8"), first);
    assert.equal(JSON.parse(await readFile(path.join(installed, "package.json"), "utf8")).version, manifest.version);
    assert.deepEqual(await readFile(path.join(installed, "lib/runtime/session-visibility.js")),
      await readFile(path.join(pluginRoot, "lib/runtime/session-visibility.js")));
    if (scenario === "upgrade") {
      assert.equal(await readFile(path.join(dshHome, "plugin-cache/ecologyrsi/old-version.tgz"), "utf8"), "old immutable cache");
    }
  }
});
