#!/usr/bin/env node
import { createHash } from "node:crypto";
import {
  cp, lstat, mkdir, mkdtemp, open, readFile, readdir, realpath, rename, rm, stat, writeFile,
} from "node:fs/promises";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

import { PRESET_MANIFEST } from "../integrations/dsh_ecology_plugin/lib/runtime/contracts.js";
export const PRESET_IDS = Object.freeze(PRESET_MANIFEST.presets.map(item => item.preset_id));
const MANAGED_PRESET_ID = /^ecology-(?:coordinator|researcher|candidate-proposer|sample-planner|sample-critic|generation-judge|local-editor)-v[0-9]+$/;

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

export async function validatePluginArchive({ archive, pluginRoot }) {
  // Inspect bytes without extracting untrusted paths or touching a DSH profile.
  const tar = (args) => {
    const result = spawnSync("tar", args, { maxBuffer: 32 * 1024 * 1024 });
    if (result.error || result.status !== 0) throw new Error("invalid DSH plugin archive");
    return result.stdout;
  };
  const entries = tar(["-tzf", archive]).toString("utf8").trim().split("\n");
  if (new Set(entries).size !== entries.length || entries.some((entry) =>
    !entry.startsWith("package/") || entry.split("/").includes(".."))) {
    throw new Error("unsafe or duplicate DSH plugin archive paths");
  }
  const required = ["package.json", "lib/index.js", "lib/runtime/session-visibility.js"];
  async function collect(relative) {
    for (const entry of await readdir(path.join(pluginRoot, relative), { withFileTypes: true })) {
      const next = path.posix.join(relative, entry.name);
      if (entry.isDirectory()) await collect(next);
      else if (entry.isFile()) required.push(next);
      else throw new Error(`unsupported plugin source entry: ${next}`);
    }
  }
  await collect("lib");
  await collect("presets");
  await collect("schemas");
  for (const relative of new Set(required)) {
    const member = `package/${relative}`;
    if (!entries.includes(member)) throw new Error(`DSH plugin archive is missing ${member}`);
    const contents = tar(["-xOf", archive, member]);
    if (!contents.equals(await readFile(path.join(pluginRoot, relative)))) {
      throw new Error(`DSH plugin archive does not match source: ${member}`);
    }
  }
  const manifest = JSON.parse(await readFile(path.join(pluginRoot, "package.json"), "utf8"));
  if (manifest.exports?.["./session-visibility"] !== "./lib/runtime/session-visibility.js") {
    throw new Error("DSH plugin session-visibility export is missing");
  }
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

async function resolveExecutable(command, env = process.env) {
  if (path.isAbsolute(command) || path.dirname(command) !== ".") {
    return await realpath(path.resolve(command));
  }
  const extensions = process.platform === "win32"
    ? (env.PATHEXT || ".EXE;.CMD;.BAT;.COM").split(";")
    : [""];
  for (const directory of String(env.PATH || "").split(path.delimiter)) {
    if (!directory) continue;
    for (const extension of extensions) {
      const candidate = path.join(directory, `${command}${extension}`);
      try {
        if ((await stat(candidate)).isFile()) return await realpath(candidate);
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
    }
  }
  throw new Error(`DSH executable is not available: ${command}`);
}

export async function dshPackageJson(dshBin) {
  const executable = await resolveExecutable(dshBin);
  let directory = path.dirname(executable);
  while (true) {
    const candidate = path.join(directory, "package.json");
    if (await exists(candidate)) {
      const manifest = JSON.parse(await readFile(candidate, "utf8"));
      if (manifest.name === "@deepseek-ai/dsh") return candidate;
    }
    const parent = path.dirname(directory);
    if (parent === directory) return null; // Standalone test doubles have no package.
    directory = parent;
  }
}

async function validatePresetTreeWithDsh(sourcePath, dshBin) {
  const packageJson = await dshPackageJson(dshBin);
  if (packageJson == null) return;
  const manifest = JSON.parse(await readFile(packageJson, "utf8"));
  if (manifest.version !== "0.2.0-rc.2") {
    throw new Error(`this plugin requires DSH 0.2.0-rc.2; found ${manifest.version}`);
  }
  const requireFromDsh = createRequire(packageJson);
  // Use exactly the host's parser and row validator, including !!js expressions.
  // Missing modules in a real installation are an incompatibility, not a skip.
  const { entryListSchema } = await import(pathToFileURL(
    requireFromDsh.resolve("@deepseek-ai/cordis-plugin-include"),
  ).href);
  const { entryListProblem } = await import(pathToFileURL(
    requireFromDsh.resolve("@deepseek-ai/dsh-agent-preset-registry"),
  ).href);
  const yaml = requireFromDsh("js-yaml");
  for (const id of PRESET_IDS) {
    let rows;
    try {
      rows = yaml.load(await readFile(path.join(sourcePath, id, "agent.cordis.yml"), "utf8"),
        { schema: entryListSchema });
    } catch (error) {
      throw new Error(`DSH preset ${id} is not valid YAML`, { cause: error });
    }
    const problem = entryListProblem(rows, id);
    if (problem) throw new Error(`DSH preset ${id} is invalid: ${problem}`);
  }
}

export async function installPresetTree({ sourceRoot, dshHome, dshBin = null }) {
  const sourcePath = fileURLToPath(sourceRoot instanceof URL ? sourceRoot : pathToFileURL(path.resolve(sourceRoot)));
  const destinationRoot = path.join(path.resolve(dshHome), ".agent-presets");
  if (dshBin != null) await validatePresetTreeWithDsh(sourcePath, dshBin);
  await assertNoSymlink(dshHome);
  await mkdir(destinationRoot, { recursive: true, mode: 0o700 });
  await assertNoSymlink(destinationRoot, dshHome);
  const currentPresetIds = new Set(PRESET_IDS);
  for (const entry of await readdir(destinationRoot, { withFileTypes: true })) {
    const id = entry.name;
    if (!MANAGED_PRESET_ID.test(id) || currentPresetIds.has(id)) continue;
    const target = path.join(destinationRoot, id);
    await assertNoSymlink(target, dshHome);
    await rm(target, { recursive: true, force: false });
  }
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

export function managedPatchText({
  staticRoot,
  sessionVisibilityModule = "@ecologyrsi/dsh-evolution-plugin/session-visibility",
  presetRoot = fileURLToPath(new URL("../integrations/dsh_ecology_plugin/presets/", import.meta.url)),
}) {
  const quote = value => `'${String(value).replaceAll("'", "''")}'`;
  // DSH 0.2 discovers declarations in the composition, not .agent-presets folders.
  // Includes keep !!js baseUrl relative to each installed preset's own skills.
  const declarations = PRESET_IDS.map((id, order) => `    - id: preset-${id}
      name: '@deepseek-ai/dsh-agent-preset'
      config:
        id: ${id}
        name: ${quote(id)}
        order: ${100 + order}
        plugins:
          - id: composition
            name: '@deepseek-ai/cordis-plugin-include'
            config:
              path: ${quote(pathToFileURL(path.join(presetRoot, id, "agent.cordis.yml")).href)}
`).join("");
  return `${BEGIN}
- insert:
    - id: ecologyrsi-evolution
      name: '@ecologyrsi/dsh-evolution-plugin'
      inject: [webServer, agents, sessions, tokenMeter, subagents, tools, sessionPersistence, sessionProjections, agentPresets, llm, web]
      config:
        staticRoot: ${quote(staticRoot)}
        backendOrigin: 'http://127.0.0.1:8777'
    - id: ecologyrsi-session-visibility
      name: ${quote(sessionVisibilityModule)}
      inject: [sessions, sessionPersistence]
${declarations}${END}
`;
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
  // A running DSH may have cached the previous package.json exports. Resolve
  // this new, independent entry by installed path so live config reload can
  // add it without restarting the evolution controller and its active Agents.
  const managed = managedPatchText({
    staticRoot,
    presetRoot: path.join(path.resolve(dshHome), "profiles", profile,
      "node_modules", "@ecologyrsi", "dsh-evolution-plugin", "presets"),
    sessionVisibilityModule: path.join(
      path.resolve(dshHome), "profiles", profile, "node_modules", "@ecologyrsi",
      "dsh-evolution-plugin", "lib", "runtime", "session-visibility.js",
    ),
  });
  let previous = "";
  if (await exists(target)) previous = await readFile(target, "utf8");
  const begin = previous.indexOf(BEGIN);
  const end = previous.indexOf(END);
  const meaningfulLines = previous
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"));
  const emptyDocumentMatch = (
    begin < 0
    && end < 0
    && meaningfulLines.length === 1
    && meaningfulLines[0] === "[]"
  )
    ? previous.match(/^[ \t]*\[\][ \t]*$/m)
    : null;
  const outsideManaged = previous.replace(
    previous.slice(Math.max(begin, 0), end >= 0 ? end + END.length : 0),
    "",
  );
  if (
    (begin >= 0) !== (end >= 0)
    || /id:\s*ecologyrsi-evolution/.test(outsideManaged)
  ) {
    throw new Error("refusing unmanaged or malformed ecologyrsi Host patch");
  }
  let next;
  if (begin >= 0) next = `${previous.slice(0, begin)}${managed}${previous.slice(end + END.length).replace(/^\n/, "")}`;
  else if (emptyDocumentMatch != null && emptyDocumentMatch.index != null) {
    next = `${previous.slice(0, emptyDocumentMatch.index)}${managed}${previous
      .slice(emptyDocumentMatch.index + emptyDocumentMatch[0].length)
      .replace(/^\r?\n/, "")}`;
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
    await validatePluginArchive({ archive: packed, pluginRoot });
    const dshBin = process.env.DSH_BIN || "dsh";
    // Refuse an incompatible host or invalid composition before changing its
    // package installation, cache, or managed patch.
    await validatePresetTreeWithDsh(path.join(pluginRoot, "presets"), dshBin);
    const cache = path.join(dshHome, "plugin-cache", "ecologyrsi");
    await mkdir(cache, { recursive: true, mode: 0o700 });
    const stable = path.join(cache, path.basename(packed));
    if (await exists(stable)) {
      if (await fileDigest(stable) !== await fileDigest(packed)) throw new Error("refusing drifting cached DSH plugin archive");
    } else {
      await cp(packed, stable, { errorOnExist: true, force: false });
      await fsyncFile(stable);
    }
    run(
      dshBin,
      ["plugin", "--profile", profile, "add", "--save-exact", `file:${stable}`],
      { cwd: dshHome },
    );
    await installPresetTree({
      sourceRoot: path.join(pluginRoot, "presets"),
      dshHome,
      dshBin,
    });
    await installManagedPatch({ dshHome, staticRoot, profile });
    run(dshBin, ["--profile", profile, "--dump-config"], {
      cwd: dshHome,
      stdio: ["ignore", "ignore", "inherit"],
    });
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
