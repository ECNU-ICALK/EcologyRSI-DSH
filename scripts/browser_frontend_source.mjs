// Bind browser evidence to the actual static response bytes, excluding API data.
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const digest = bytes => createHash("sha256").update(bytes).digest("hex");

export function captureFrontendSource(page, baseUrl, sourceRoot) {
  const base = new URL(baseUrl);
  const prefix = base.pathname.endsWith("/") ? base.pathname : base.pathname + "/";
  const root = path.resolve(sourceRoot);
  const pending = [];
  const captured = new Map();
  const errors = [];
  const listener = response => {
    const url = new URL(response.url());
    if (url.origin !== base.origin || !url.pathname.startsWith(prefix)
        || response.request().method() !== "GET" || response.status() !== 200) return;
    const relative = decodeURIComponent(url.pathname.slice(prefix.length)) || "index.html";
    if (!/\.(?:html|css|js)$/.test(relative)) return;
    pending.push((async () => {
      assert.ok(!relative.split("/").includes("..") && !relative.includes("\\"), "Unsafe static path");
      const file = path.resolve(root, relative);
      assert.ok(file.startsWith(root + path.sep), "Static path leaves frontend source");
      const [loaded, expected] = await Promise.all([response.body(), fs.readFile(file)]);
      const hash = digest(loaded);
      assert.equal(hash, digest(expected), "Loaded frontend differs from source: " + relative);
      if (captured.has(relative)) assert.equal(captured.get(relative), hash, "Static response changed during review");
      captured.set(relative, hash);
    })().catch(error => errors.push(error.message)));
  };
  page.on("response", listener);
  return async () => {
    page.off("response", listener);
    await Promise.all(pending);
    assert.deepEqual(errors, [], "Cannot bind browser evidence to frontend source");
    const html = await fs.readFile(path.join(root, "index.html"), "utf8");
    const linked = [...html.matchAll(/(?:src|href)=["']([^"']+\.(?:js|css))["']/g)].map(match => match[1]);
    const required = [...new Set(["index.html", ...linked])].sort();
    assert.deepEqual([...captured.keys()].sort(), required, "Browser did not load the complete frontend entry set");
    for (const relative of required) {
      assert.equal(captured.get(relative), digest(await fs.readFile(path.join(root, relative))),
        "Frontend source changed during browser review: " + relative);
    }
    return {frontend_source_verified: true,
      frontend_source_sha256: Object.fromEntries([...captured.entries()].sort(([a], [b]) => a.localeCompare(b)))};
  };
}
