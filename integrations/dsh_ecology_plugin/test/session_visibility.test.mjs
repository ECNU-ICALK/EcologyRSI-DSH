import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { evolutionSessionIds, installSessionVisibility } from "../lib/runtime/session-visibility.js";

test("identity and ancestry hide old/new evolution sessions, never a matching title or cwd", () => {
  const headers = [
    { id: "grandchild", parentSession: "child" },
    { id: "child", parentSession: "ecology-role-old" },
    { id: "preset-only", agentPreset: "ecology-researcher-v1" },
    { id: "ecology-role-new" },
    { id: "user", cwd: "/EcologyRSI-DSH", title: "EcologyRSI-DSH" },
    { id: "user-child", parentSession: "user", origin: "subagent" },
    { id: "other-preset", agentPreset: "ecology-user-custom-v1" },
  ];
  assert.deepEqual([...evolutionSessionIds(headers)].sort(),
    ["grandchild", "child", "preset-only", "ecology-role-new"].sort());
});

function fixture() {
  const headers = new Map([
    ["ecology-role-old", { id: "ecology-role-old" }],
    ["child", { id: "child", parentSession: "ecology-role-old" }],
    ["grandchild", { id: "grandchild", parentSession: "child" }],
    ["normal", { id: "normal", cwd: "/EcologyRSI-DSH" }],
  ]);
  const live = new Map();
  const reads = [], emitted = [], listeners = new Map();
  const persistence = {
    async listSessionDirs() { return [...headers.keys()].map((id) => `/store/${id}`); },
    async list() {
      const dirs = await this.listSessionDirs();
      return dirs.map((directory) => {
        const id = path.basename(directory); reads.push(id);
        return { header: headers.get(id) };
      });
    },
    async stat(id) { return { header: headers.get(id) }; },
    async open(id) { return { header: headers.get(id), log: "durable events" }; },
  };
  const query = { async listSessions(signal) {
    signal?.throwIfAborted();
    return [...await persistence.list(), ...[...live.values()].map((session) => ({ header: session.header }))];
  } };
  const emitter = { emit: (...args) => emitted.push(args) };
  const ctx = {
    sessionPersistence: persistence, sessionQuery: query,
    sessionController: { ctx: emitter },
    sessions: { get: (id) => live.get(id) },
    inject(_names, callback) {
      const cleanups = [];
      callback({ ...ctx, effect: (effect) => cleanups.push(effect()) });
      return { dispose: () => cleanups.reverse().forEach((cleanup) => cleanup()) };
    },
    on(event, callback) { listeners.set(event, callback); return () => listeners.delete(event); },
  };
  return { ctx, headers, live, persistence, query, reads, emitted, emitter, listeners };
}

test("workspace scans skip known log directories; normal refreshes and direct resume stay intact", async () => {
  const f = fixture();
  const originalList = f.query.listSessions, originalEmit = f.emitter.emit;
  const dispose = installSessionVisibility(f.ctx);
  try {
    assert.deepEqual((await f.query.listSessions()).map((r) => r.header.id), ["normal"]);
    f.reads.length = 0;
    assert.deepEqual((await f.query.listSessions()).map((r) => r.header.id), ["normal"]);
    assert.deepEqual(f.reads, ["normal"], "hidden logs must not even be read");
    f.headers.set("new-user", { id: "new-user" });
    assert.deepEqual((await f.query.listSessions()).map((r) => r.header.id), ["normal", "new-user"]);
    const [ui, all] = await Promise.all([f.query.listSessions(), f.persistence.list()]);
    assert.equal(ui.length, 2);
    assert.equal(all.length, 5, "concurrent non-UI reads retain the full corpus");
    assert.equal((await f.persistence.stat("child")).header.parentSession, "ecology-role-old");
    assert.equal((await f.persistence.open("child")).log, "durable events");
    await assert.rejects(f.query.listSessions(AbortSignal.abort()), { name: "AbortError" });
  } finally { await dispose(); }
  assert.equal(f.query.listSessions, originalList);
  assert.equal(f.emitter.emit, originalEmit);
  assert.equal((await f.query.listSessions()).length, 5);
});

test("new role hosts and descendants never emit UI list updates while runtime events remain", async () => {
  const f = fixture();
  const dispose = installSessionVisibility(f.ctx);
  try {
    const host = { id: "ecology-role-live", header: { id: "ecology-role-live" } };
    const child = { id: "live-child", header: { id: "live-child", parentSession: host.id } };
    f.live.set(host.id, host); f.live.set(child.id, child);
    f.listeners.get("session/created")(child);
    for (const session of [host, child]) {
      f.emitter.emit("api-session/added", { sessionId: session.id, parentSessionId: session.header.parentSession });
      f.emitter.emit("api-session/status", session.id, true);
      f.emitter.emit("api-session/activity", session.id, 1);
    }
    f.live.delete(child.id);
    f.emitter.emit("api-session/removed", child.id);
    assert.equal(f.emitted.length, 0);
    f.emitter.emit("session/event", child, { type: "assistant/message" });
    f.emitter.emit("api-session/added", { sessionId: "user" });
    assert.deepEqual(f.emitted.map(([event]) => event), ["session/event", "api-session/added"]);
    assert.deepEqual((await f.query.listSessions()).map((r) => r.header.id), ["normal"]);
  } finally { await dispose(); }
});

test("restart reuses the local index and discovers newly persisted children", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "ecology-session-visibility-"));
  const cacheFile = path.join(directory, "index.json");
  try {
    const first = fixture();
    const dispose = installSessionVisibility(first.ctx, { cacheFile });
    await first.query.listSessions(); await dispose();
    const cached = JSON.parse(await readFile(cacheFile, "utf8"));
    assert.ok(cached.ids.includes("child"));
    const second = fixture();
    second.headers.set("new-child", { id: "new-child", parentSession: "child" });
    const cleanup = installSessionVisibility(second.ctx, { cacheFile });
    try {
      assert.deepEqual((await second.query.listSessions()).map((r) => r.header.id), ["normal"]);
      assert.deepEqual(second.reads, ["normal", "new-child"]);
    } finally { await cleanup(); }
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("non-JSONL backends still filter correctly without an enumeration optimization", async () => {
  const f = fixture();
  f.persistence.list = async () => [...f.headers.values()].map((header) => ({ header }));
  delete f.persistence.listSessionDirs;
  const dispose = installSessionVisibility(f.ctx);
  try { assert.deepEqual((await f.query.listSessions()).map((r) => r.header.id), ["normal"]); }
  finally { await dispose(); }
});
