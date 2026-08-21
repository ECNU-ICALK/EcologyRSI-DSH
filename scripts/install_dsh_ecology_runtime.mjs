#!/usr/bin/env node
import { createHash } from "node:crypto";
import {
  cp, lstat, mkdir, mkdtemp, open, readFile, readdir, realpath, rename, rm, stat, writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

export const PRESET_IDS = Object.freeze([
  "ecology-coordinator-v1",
  "ecology-researcher-v1",
  "ecology-candidate-proposer-v1",
  "ecology-sample-planner-v1",
  "ecology-sample-critic-v1",
  "ecology-generation-judge-v1",
]);

const BEGIN = "# BEGIN ECOLOGYRSI DSH RUNTIME (managed)";
const END = "# END ECOLOGYRSI DSH RUNTIME (managed)";

async function exists(target) {
  try { await stat(target); return true; } catch (error) { if (error.code === "ENOENT") return false; throw error; }
}

async function assertNoSymlink(target, stopAt = path.parse(target).root) {
  let current = path.resolve(target);
  const stop = path.resolve(stopAt);
  while (current.startsWith(stop)) {
    try {
      const metadata = await lstat(current);
      if (metadata.isSymbolicLink()) throw new Error(`refusing symlink target: ${current}`);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
    if (current === stop) break;
    current = path.dirname(current);
  }
}

export async function resolveDshHome({ env = process.env, homeDir = os.homedir() } = {}) {
  const candidate = path.resolve(env.DSH_HOME || path.join(homeDir, ".dsh"));
  try {
    if ((await lstat(candidate)).isSymbolicLink()) throw new Error(`refusing symlink target: ${candidate}`);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  await mkdir(candidate, { recursive: true, mode: 0o700 });
  return await realpath(candidate);
}

async function directoryDigest(root) {
  const digest = createHash("sha256");
  async function walk(current, relative = "") {
    const entries = await readdir(current, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const nextRelative = path.posix.join(relative, entry.name);
      const absolute = path.join(current, entry.name);
      const metadata = await lstat(absolute);
      if (metadata.isSymbolicLink()) throw new Error(`preset source contains symlink: ${nextRelative}`);
      digest.update(`${entry.isDirectory() ? "d" : "f"}:${nextRelative}\0`);
      if (entry.isDirectory()) await walk(absolute, nextRelative);
      else if (entry.isFile()) digest.update(await readFile(absolute));
      else throw new Error(`unsupported preset entry: ${nextRelative}`);
    }
  }
  await walk(root);
  return digest.digest("hex");
}

async function fileDigest(target) {
  return createHash("sha256").update(await readFile(target)).digest("hex");
}

async function fsyncFile(target) {
  const handle = await open(target, "r");
  try { await handle.sync(); } finally { await handle.close(); }
}

async function fsyncTree(root) {
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const target = path.join(root, entry.name);
    if (entry.isDirectory()) await fsyncTree(target);
    else if (entry.isFile()) await fsyncFile(target);
  }
}

export async function installPresetTree({ sourceRoot, dshHome }) {
  const sourcePath = fileURLToPath(sourceRoot instanceof URL ? sourceRoot : pathToFileURL(path.resolve(sourceRoot)));
  const destinationRoot = path.join(path.resolve(dshHome), ".agent-presets");
  await assertNoSymlink(dshHome);
  await mkdir(destinationRoot, { recursive: true, mode: 0o700 });
  await assertNoSymlink(destinationRoot, dshHome);
  for (const id of PRESET_IDS) {
    if (!/^[a-z0-9][a-z0-9-]*$/.test(id)) throw new Error(`invalid preset id: ${id}`);
    const source = path.join(sourcePath, id);
    const target = path.join(destinationRoot, id);
    const sourceDigest = await directoryDigest(source);
    if (await exists(target)) {
      await assertNoSymlink(target, dshHome);
      const installedDigest = await directoryDigest(target);
      if (sourceDigest !== installedDigest) throw new Error(`refusing drifting preset: ${id}`);
      continue;
    }
    const temporary = `${target}.tmp-${process.pid}-${Date.now()}`;
    await cp(source, temporary, { recursive: true, errorOnExist: true, force: false });
    await fsyncTree(temporary);
    await rename(temporary, target);
  }
}

export function managedPatchText({ staticRoot }) {
  const safeRoot = String(staticRoot).replaceAll("'", "''");
  return `${BEGIN}\n- insert:\n    - id: ecologyrsi-evolution\n      name: '@ecologyrsi/dsh-evolution-plugin'\n      inject: [webServer, agents, sessions, tokenMeter, subagents, tools, sessionPersistence, sessionProjections, agentPresets, llm]\n      config:\n        staticRoot: '${safeRoot}'\n        backendOrigin: 'http://127.0.0.1:8777'\n${END}\n`;
}

async function atomicWrite(target, content) {
  await mkdir(path.dirname(target), { recursive: true });
  await assertNoSymlink(path.dirname(target));
  const temporary = `${target}.tmp-${process.pid}-${Date.now()}`;
  await writeFile(temporary, content, { encoding: "utf8", mode: 0o600, flag: "wx" });
  await fsyncFile(temporary);
  await rename(temporary, target);
}

export async function installManagedPatch({ dshHome, staticRoot, profile = "web" }) {
  if (!/^[a-z0-9][a-z0-9-]*$/.test(profile)) throw new Error("invalid DSH profile name");
  const target = path.join(dshHome, "profiles", profile, "cordis.patch.yml");
  await assertNoSymlink(target, dshHome);
  const managed = managedPatchText({ staticRoot });
  let previous = "";
  if (await exists(target)) previous = await readFile(target, "utf8");
  const begin = previous.indexOf(BEGIN);
  const end = previous.indexOf(END);
  const legacyPattern = /- insert:\n    - id: ecologyrsi-evolution\n      name: '@ecologyrsi\/dsh-evolution-plugin'\n      inject: \[webServer\]\n      config:\n        staticRoot: '[^'\n]+'\n        backendOrigin: 'http:\/\/127\.0\.0\.1:8777'\n?/;
  const legacyMatch = begin < 0 && end < 0 ? previous.match(legacyPattern) : null;
  const outsideManaged = previous.replace(
    previous.slice(Math.max(begin, 0), end >= 0 ? end + END.length : 0),
    "",
  );
  if (
    (begin >= 0) !== (end >= 0)
    || (/id:\s*ecologyrsi-evolution/.test(outsideManaged) && legacyMatch == null)
  ) {
    throw new Error("refusing unmanaged or malformed ecologyrsi Host patch");
  }
  let next;
  if (begin >= 0) next = `${previous.slice(0, begin)}${managed}${previous.slice(end + END.length).replace(/^\n/, "")}`;
  else if (legacyMatch != null && legacyMatch.index != null) {
    next = `${previous.slice(0, legacyMatch.index)}${managed}${previous.slice(legacyMatch.index + legacyMatch[0].length)}`;
  }
  else next = `${previous}${previous && !previous.endsWith("\n") ? "\n" : ""}${managed}`;
  if (next !== previous) await atomicWrite(target, next);
  return target;
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { stdio: "inherit", ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} exited with status ${result.status}`);
}

export async function installRuntime({
  projectRoot, pluginRoot: suppliedPluginRoot, staticRoot: suppliedStaticRoot,
  packageArchive, dshHome, profile = "web",
}) {
  const pluginRoot = path.resolve(suppliedPluginRoot || path.join(projectRoot, "integrations", "dsh_ecology_plugin"));
  const staticRoot = path.resolve(suppliedStaticRoot || path.join(projectRoot, "plugins", "ecology_evolution"));
  const temporary = await mkdtemp(path.join(os.tmpdir(), "ecology-dsh-pack-"));
  try {
    let packed;
    if (packageArchive) {
      packed = path.resolve(packageArchive);
      if (!(await exists(packed)) || path.extname(packed) !== ".tgz") throw new Error("bundled DSH plugin archive is missing");
    } else {
      run("npm", ["pack", "--pack-destination", temporary], { cwd: pluginRoot });
      const archives = (await readdir(temporary)).filter((name) => name.endsWith(".tgz"));
      if (archives.length !== 1) throw new Error("npm pack did not produce exactly one archive");
      packed = path.join(temporary, archives[0]);
    }
    const cache = path.join(dshHome, "plugin-cache", "ecologyrsi");
    await mkdir(cache, { recursive: true, mode: 0o700 });
    const stable = path.join(cache, path.basename(packed));
    if (await exists(stable)) {
      if (await fileDigest(stable) !== await fileDigest(packed)) throw new Error("refusing drifting cached DSH plugin archive");
    } else {
      await cp(packed, stable, { errorOnExist: true, force: false });
      await fsyncFile(stable);
    }
    const dshBin = process.env.DSH_BIN || "dsh";
    run(
      dshBin,
      ["plugin", "--profile", profile, "add", "--save-exact", `file:${stable}`],
      { cwd: dshHome },
    );
    await installPresetTree({ sourceRoot: path.join(pluginRoot, "presets"), dshHome });
    await installManagedPatch({ dshHome, staticRoot, profile });
    run(dshBin, ["--profile", profile, "--dump-config"], { cwd: dshHome });
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

async function main() {
  const argumentsMap = new Map();
  for (let index = 2; index < process.argv.length; index += 2) {
    const name = process.argv[index];
    const value = process.argv[index + 1];
    if (!name?.startsWith("--") || value == null) throw new Error("installer arguments must be --name value pairs");
    argumentsMap.set(name, value);
  }
  const projectRoot = argumentsMap.has("--project-root")
    ? path.resolve(argumentsMap.get("--project-root"))
    : path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const dshHome = await resolveDshHome();
  await installRuntime({
    projectRoot,
    pluginRoot: argumentsMap.get("--plugin-root"),
    staticRoot: argumentsMap.get("--static-root"),
    packageArchive: argumentsMap.get("--tgz"),
    profile: argumentsMap.get("--profile") || "web",
    dshHome,
  });
  process.stdout.write(`EcologyRSI DSH runtime installed in ${dshHome}\n`);
}

async function isMainModule() {
  if (!process.argv[1]) return false;
  return await realpath(path.resolve(process.argv[1])) === await realpath(fileURLToPath(import.meta.url));
}

if (await isMainModule()) {
  main().catch((error) => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
}
