import { AsyncLocalStorage } from "node:async_hooks";
import { readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

const ROLE_PREFIX = "ecology-role-";
const PRESET = /^ecology-(?:coordinator|researcher|candidate-proposer|sample-planner|sample-critic|generation-judge|local-editor)-v\d+$/;
const CACHE_VERSION = 1;
export const name = "ecologyrsi-session-visibility";
export const inject = ["sessions", "sessionPersistence"];

export function apply(ctx) {
  ctx.effect(() => installSessionVisibility(ctx), "ecologyrsi: local-only evolution sessions");
}

const LIST_EVENTS = new Set([
  "api-session/added", "api-session/removed", "api-session/status", "api-session/activity",
]);

// Only immutable provenance identifies an internal session. A user can have a
// normal conversation with the same cwd or title as an evolution role host.
export function evolutionSessionIds(headers, known = new Set()) {
  const children = new Map();
  const pending = [];
  for (const header of headers) {
    if (!header?.id) continue;
    if (header.id.startsWith(ROLE_PREFIX) || PRESET.test(header.agentPreset || "")
      || known.has(header.id) || String(header.parentSession || "").startsWith(ROLE_PREFIX)
      || known.has(header.parentSession)) {
      pending.push(header.id);
    }
    if (header.parentSession) {
      const siblings = children.get(header.parentSession) || [];
      siblings.push(header.id);
      children.set(header.parentSession, siblings);
    }
  }
  const visited = new Set();
  for (let cursor = 0; cursor < pending.length; cursor++) {
    const id = pending[cursor];
    if (visited.has(id)) continue;
    visited.add(id);
    known.add(id);
    pending.push(...(children.get(id) || []));
  }
  return known;
}

function replaceMethod(target, key, wrap) {
  const descriptor = Object.getOwnPropertyDescriptor(target, key);
  const original = target[key];
  const replacement = wrap(original);
  Object.defineProperty(target, key, { configurable: true, writable: true, value: replacement });
  return () => {
    if (target[key] !== replacement) return;
    if (descriptor) Object.defineProperty(target, key, descriptor);
    else delete target[key];
  };
}

// DSH 0.1.5-rc.2 has no list-visibility hook. Keep the policy in this plugin:
// filter the public query list BEFORE ApiSessionList builds projections, and
// scope the JSONL enumeration optimization to that list's async call chain.
// stat/open, full persistence listings, resume, and concurrent runtime work
// continue to see the complete local log corpus.
export function installSessionVisibility(ctx, { cacheFile } = {}) {
  const hidden = new Set();
  const listing = new AsyncLocalStorage();
  const persistence = ctx.sessionPersistence;
  const disposers = [];
  let disposed = false;
  let timer;
  let savedSize = 0;
  let saving = Promise.resolve();
  const cachePath = cacheFile ?? (typeof persistence?.root === "string"
    ? path.join(persistence.root, ".ecology-workspace-hidden.json") : null);
  const warn = (error) => console.warn(`[ecologyrsi] session visibility cache: ${error.code || error.name}`);
  const ready = cachePath ? readFile(cachePath, "utf8").then((text) => {
    const data = JSON.parse(text);
    if (data.version !== CACHE_VERSION || !Array.isArray(data.ids)
      || !data.ids.every((id) => typeof id === "string" && id.length > 0)) return;
    for (const id of data.ids) hidden.add(id);
    savedSize = data.ids.length;
  }).catch((error) => { if (error.code !== "ENOENT") warn(error); }) : Promise.resolve();

  function save() {
    saving = saving.then(async () => {
      await ready;
      if (!cachePath || hidden.size === savedSize) return;
      const ids = [...hidden];
      const temporary = `${cachePath}.${process.pid}.tmp`;
      await writeFile(temporary, JSON.stringify({ version: CACHE_VERSION, ids }), { mode: 0o600 });
      await rename(temporary, cachePath);
      savedSize = ids.length;
    }).catch(warn);
    return saving;
  }

  function remember(headers) {
    const before = hidden.size;
    evolutionSessionIds(headers, hidden);
    if (hidden.size !== before && !disposed && !timer) {
      timer = setTimeout(() => { timer = null; void save(); }, 5000);
      timer.unref?.();
    }
  }

  function isHidden(id, suppliedHeader) {
    if (!id) return false;
    if (hidden.has(id) || id.startsWith(ROLE_PREFIX)) return true;
    const chain = [];
    const visited = new Set();
    let header = suppliedHeader || ctx.sessions?.get?.(id)?.header;
    while (header && !visited.has(header.id)) {
      visited.add(header.id);
      chain.push(header);
      header = ctx.sessions?.get?.(header.parentSession)?.header;
    }
    remember(chain);
    return hidden.has(id);
  }

  if (typeof persistence?.listSessionDirs === "function") {
    disposers.push(replaceMethod(persistence, "listSessionDirs", (original) => async function (...args) {
      const directories = await original.apply(this, args);
      if (!listing.getStore()) return directories;
      return directories.filter((directory) => {
        // DSH's reversible path-segment encoding (UUIDs stay literal).
        const id = path.basename(directory).replace(/~([0-9A-F]{4})/g,
          (_, hex) => String.fromCharCode(parseInt(hex, 16)));
        return !hidden.has(id) && !id.startsWith(ROLE_PREFIX);
      });
    }));
  }

  const queryFiber = ctx.inject(["sessionQuery"], (queryCtx) => {
    queryCtx.effect(() => replaceMethod(queryCtx.sessionQuery, "listSessions", (original) => async function (...args) {
      await ready;
      const records = await listing.run(true, () => original.apply(this, args));
      remember(records.map((record) => record.header));
      const visible = records.filter((record) => !hidden.has(record.header.id));
      await save();
      return visible;
    }), "ecologyrsi: exclude internal sessions from workspace queries");
  });
  disposers.push(() => queryFiber.dispose());

  // The controller emits incremental list updates from its own context. Filter
  // only those UI notifications; Session lifecycle and event-log subscribers
  // (including persistence and usage accounting) are left intact.
  const controllerFiber = ctx.inject(["sessionController"], (controllerCtx) => {
    const emitter = controllerCtx.sessionController.ctx;
    controllerCtx.effect(() => replaceMethod(emitter, "emit", (original) => function (event, ...args) {
      if (LIST_EVENTS.has(event)) {
        const item = args[0];
        const id = typeof item === "string" ? item : item?.sessionId;
        const header = typeof item === "object" ? {
          id, parentSession: item.parentSessionId,
        } : undefined;
        if (isHidden(id, ctx.sessions?.get?.(id)?.header || header)) return;
      }
      return original.call(this, event, ...args);
    }), "ecologyrsi: exclude internal session list notifications");
  });
  disposers.push(() => controllerFiber.dispose());
  disposers.push(ctx.on("session/created", (session) => {
    isHidden(session.id, session.header);
  }, { prepend: true }));

  return () => {
    disposed = true;
    clearTimeout(timer);
    for (const dispose of disposers.reverse()) dispose();
    return save();
  };
}
