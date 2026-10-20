#!/usr/bin/env node
// Explicit offline upgrade regression. The caller supplies the immutable 0.10.3
// archive; every write is beneath a temporary home. Package transport is a local
// copy test double, while preset validation uses the real rc.2 parser/installer.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { chmod, cp, mkdir, mkdtemp, readFile, readdir, realpath, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { dshPackageJson, installPresetTree, installRuntime, validatePluginArchive } from "../../../scripts/install_dsh_ecology_runtime.mjs";

const baseline = path.resolve(process.argv[2] || "");
assert.ok(process.argv[2], "pass the immutable 0.10.3 archive path explicitly");
const evidence = process.argv[3] ? path.resolve(process.argv[3]) : null;
const pluginRoot = fileURLToPath(new URL("../", import.meta.url));
const projectRoot = path.resolve(pluginRoot, "../..");
const hostBin = process.env.DSH_BIN || "dsh";
const hostPackage = await dshPackageJson(hostBin);
assert.ok(hostPackage, "DSH_BIN must identify a real packaged Harness");
assert.equal(JSON.parse(await readFile(hostPackage, "utf8")).version, "0.2.0-rc.2");
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const baselineDigest = sha(await readFile(baseline));
const tar = args => {
  const result = spawnSync("tar", args, { encoding: "utf8", maxBuffer: 32 * 1024 * 1024 });
  assert.equal(result.status, 0, result.stderr);
  return result.stdout;
};
const members = tar(["-tzf", baseline]).trim().split("\n");
assert.ok(members.every(member => member.startsWith("package/") && !member.split("/").includes("..")));
assert.equal(new Set(members).size, members.length);
const tmp = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-display-upgrade-")));
const report = { schema_version: "ecologyrsi-dsh.display-projection-upgrade/1", passed: false,
  baseline_version: "0.10.3", baseline_archive_sha256: baselineDigest, harness_version: "0.2.0-rc.2",
  remote_model_calls: 0, remote_package_calls: 0, temporary_home_only: true,
  package_transport: "local extraction test double", preset_validation: "real rc.2 parser and production installer" };
const snapshot = async root => {
  const files = {};
  async function walk(relative = "") {
    for (const entry of await readdir(path.join(root, relative), { withFileTypes: true })) {
      const name = path.posix.join(relative, entry.name);
      assert.ok(!entry.isSymbolicLink());
      if (entry.isDirectory()) await walk(name);
      else files[name] = sha(await readFile(path.join(root, name)));
    }
  }
  await walk(); return files;
};
const previousBin = process.env.DSH_BIN;
try {
  const oldRoot = path.join(tmp, "old"); await mkdir(oldRoot);
  tar(["-xzf", baseline, "-C", oldRoot]);
  const oldPackage = path.join(oldRoot, "package");
  assert.equal(JSON.parse(await readFile(path.join(oldPackage, "package.json"), "utf8")).version, "0.10.3");
  await validatePluginArchive({ archive: baseline, pluginRoot: oldPackage });
  const newParent = path.join(tmp, "new"), newPackage = path.join(newParent, "package");
  await mkdir(newPackage, { recursive: true });
  for (const name of ["package.json", "lib", "presets", "schemas"]) {
    await cp(path.join(pluginRoot, name), path.join(newPackage, name), { recursive: true });
  }
  const manifest = JSON.parse(await readFile(path.join(newPackage, "package.json"), "utf8"));
  manifest.version += "-display-upgrade-test";
  await writeFile(path.join(newPackage, "package.json"), JSON.stringify(manifest));
  const candidateArchive = path.join(tmp, `candidate-${manifest.version}.tgz`);
  tar(["-czf", candidateArchive, "-C", newParent, "package"]);
  await validatePluginArchive({ archive: candidateArchive, pluginRoot: newPackage });
  const dshHome = path.join(tmp, "home"); await mkdir(dshHome);
  // The first installation uses actual released 0.10.3 preset bytes, not a copy
  // of the new source. This is the case a same-id Skill edit used to break.
  await installPresetTree({ sourceRoot: path.join(oldPackage, "presets"), dshHome, dshBin: hostBin });
  const before = await snapshot(path.join(dshHome, ".agent-presets"));
  await installPresetTree({ sourceRoot: path.join(newPackage, "presets"), dshHome, dshBin: hostBin });
  assert.deepEqual(await snapshot(path.join(dshHome, ".agent-presets")), before);
  const fakeDsh = path.join(tmp, "copy-package.cjs");
  await writeFile(fakeDsh, `#!/usr/bin/env node
const fs = require('node:fs'), path = require('node:path'), {execFileSync} = require('node:child_process');
const args = process.argv.slice(2);
if (args[0] === 'plugin') {
  const dest = path.join(process.cwd(), 'profiles', args[args.indexOf('--profile') + 1], 'node_modules/@ecologyrsi/dsh-evolution-plugin');
  fs.mkdirSync(dest, {recursive:true});
  execFileSync('tar', ['-xzf', args.at(-1).slice(5), '-C', dest, '--strip-components', '1']);
}
`);
  await chmod(fakeDsh, 0o755); process.env.DSH_BIN = fakeDsh;
  const common = { projectRoot, dshHome, staticRoot: path.join(projectRoot, "plugins/ecology_evolution") };
  await installRuntime({ ...common, packageArchive: baseline, pluginRoot: oldPackage });
  const installed = path.join(dshHome, "profiles/web/node_modules/@ecologyrsi/dsh-evolution-plugin");
  assert.equal(JSON.parse(await readFile(path.join(installed, "package.json"), "utf8")).version, "0.10.3");
  await installRuntime({ ...common, packageArchive: candidateArchive, pluginRoot: newPackage });
  await installRuntime({ ...common, packageArchive: candidateArchive, pluginRoot: newPackage });
  assert.equal(JSON.parse(await readFile(path.join(installed, "package.json"), "utf8")).version, manifest.version);
  assert.deepEqual(await readFile(path.join(installed, "lib/runtime/display-projection.js")),
    await readFile(path.join(newPackage, "lib/runtime/display-projection.js")));
  assert.deepEqual(await snapshot(path.join(dshHome, ".agent-presets")), before);
  assert.equal(sha(await readFile(path.join(dshHome, "plugin-cache/ecologyrsi", path.basename(baseline)))), baselineDigest);
  // Same-id drift must remain rejected; do not relax the installer's protection.
  const skill = "ecology-sample-planner-v14/skills/origin-vector-forecasting/SKILL.md";
  await writeFile(path.join(newPackage, "presets", skill), "changed under the same preset id\n");
  await assert.rejects(installPresetTree({ sourceRoot: path.join(newPackage, "presets"), dshHome, dshBin: hostBin }),
    /refusing drifting preset: ecology-sample-planner-v14/);
  assert.deepEqual(await snapshot(path.join(dshHome, ".agent-presets")), before);
  assert.equal(sha(await readFile(baseline)), baselineDigest);
  Object.assign(report, { passed: true, candidate_version: manifest.version, immutable_preset_files: Object.keys(before).length,
    preset_sha256: before, upgrade_idempotent: true, same_id_drift_rejected: true, baseline_archive_unchanged: true });
} finally {
  if (previousBin === undefined) delete process.env.DSH_BIN; else process.env.DSH_BIN = previousBin;
  await rm(tmp, { recursive: true, force: true });
  report.temporary_home_removed = true;
  if (evidence) await writeFile(evidence, JSON.stringify(report, null, 2) + "\n");
}
console.log(JSON.stringify({ passed: report.passed, immutable_preset_files: report.immutable_preset_files,
  upgrade_idempotent: report.upgrade_idempotent, same_id_drift_rejected: report.same_id_drift_rejected,
  remote_model_calls: 0, remote_package_calls: 0 }));
