import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmod, mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const installer = fileURLToPath(new URL("../../../scripts/install_dsh_ecology_runtime.mjs", import.meta.url));

test("installer rejects unknown home and duplicate options before any filesystem or CLI mutation", async t => {
  const tmp = await mkdtemp(path.join(os.tmpdir(), "ecology-install-cli-"));
  t.after(() => rm(tmp, { recursive: true, force: true }));
  const fakeDsh = path.join(tmp, "must-not-run.cjs");
  await writeFile(fakeDsh, "#!/usr/bin/env node\nrequire('node:fs').writeFileSync(process.env.INSTALL_MUTATION_MARKER, 'called');\nprocess.exit(99);\n");
  await chmod(fakeDsh, 0o755);
  const original = await readdir(tmp);
  for (const [args, error] of [
    [["--dsh-home", path.join(tmp, "ignored-home")], /unknown installer argument: --dsh-home/],
    [["--profile", "web", "--typo", "value"], /unknown installer argument: --typo/],
    [["--profile", "web", "--profile", "other"], /duplicate installer argument: --profile/],
    [["--project-root", tmp, "--project-root", tmp], /duplicate installer argument: --project-root/],
    [["--profile"], /installer arguments must be --name value pairs/],
  ]) {
    const result = spawnSync(process.execPath, [installer, ...args], { cwd: tmp, encoding: "utf8",
      env: { ...process.env, DSH_HOME: path.join(tmp, "must-stay-absent"), DSH_BIN: fakeDsh,
        INSTALL_MUTATION_MARKER: path.join(tmp, "command-ran") } });
    assert.equal(result.status, 1, JSON.stringify(args));
    assert.match(result.stderr, error);
    assert.equal(result.stdout, "");
    assert.deepEqual(await readdir(tmp), original, "invalid CLI must not create home, profile, cache, or start DSH");
  }
});
