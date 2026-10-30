// @ecogym/dsh-plugin, browser half: the 生态进化 panel of the DSH web app.
//
// A row in the sidebar's global panel list opens the panel in DSH's main column (the slot keyed "ecogym").
// Everything shown comes from the ecogym service through the host half's same-origin proxy at /api/ecogym/.
// Rendered with DSH's own primitives and design tokens; plain React (no build step), so it stays one file.
//
// Sections, in order: style · data (the Api table of every call, api(), useResource polling, catalog, model
// routes) · toasts · formatting and status labels · primitives · backend state · evolution runs (list, run
// page and its cards) · new evolution · episodes · experts · training data · tasks · about (the 介绍 page) ·
// panel (top bar, navigation, registration). docs/architecture.md describes the layers behind it.
window.__ModuleLoader__.load({
  id: "@ecogym/dsh-plugin",
  factory: (require) => {
    const module = { exports: {} };
    const exports = module.exports;
    Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });

    const React = require("react");
    let P = {};
    try {
      P = require("@deepseek-ai/dsh-client-ui-primitives") || {};
    } catch {
      P = {};
    }
    const { useState, useEffect, useRef, useMemo, useCallback } = React;
    const h = React.createElement;
    const Fragment = React.Fragment;

    const PANEL = "ecogym";
    const API = "/api/ecogym/";
    const MINUS = "−";
    let host = null; // the client plugin context (DSH remote services for the model list)

    /* ---- style ------------------------------------------------------------------------------------------- */

    const css = `
.eg{--eg-fill:var(--dsw-alias-bg-module-platform,var(--dsw-alias-bg-layer-1));height:100%;display:flex;flex-direction:column;min-height:0;background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary);font-family:var(--dsw-font-family,inherit);font-size:14px;line-height:1.5;position:relative}
.eg *{box-sizing:border-box}
.eg a{color:var(--dsw-alias-link,var(--dsw-alias-state-business-primary));text-decoration:none}
.eg a:hover{text-decoration:underline}
.eg-bar{flex:none;height:56px;display:flex;align-items:center;gap:12px;padding:0 20px 0 24px;border-bottom:.5px solid var(--dsw-alias-border-l2)}
.eg-brand{display:inline-flex;align-items:center;gap:8px;font-size:15px;font-weight:600;white-space:nowrap}
.eg-tabs{display:flex;gap:2px;margin-left:12px;min-width:0;overflow-x:auto}
.eg-tab{height:30px;padding:0 12px;border:0;border-radius:15px;background:transparent;color:var(--dsw-alias-label-secondary);font:inherit;font-size:13px;cursor:pointer;white-space:nowrap;display:inline-flex;align-items:center;gap:6px}
.eg-tab:hover{background:var(--dsw-alias-interactive-bg-hover)}
.eg-tab[aria-current=page]{background:var(--dsw-alias-interactive-bg-active);color:var(--dsw-alias-label-primary);font-weight:500}
.eg-count{min-width:18px;height:18px;padding:0 5px;border-radius:9px;background:var(--dsw-alias-state-warn-primary);color:#fff;font-size:11px;line-height:18px;text-align:center}
.eg-spacer{flex:1}
.eg-pill{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--dsw-alias-label-tertiary);white-space:nowrap}
.eg-scroll{flex:1;min-height:0;overflow:auto;--dsh-scrollbar-thumb:var(--dsw-alias-scrollbar-bg-l2)}
.eg-page{max-width:1120px;margin:0 auto;padding:24px 32px 72px}
.eg-head{display:flex;align-items:flex-start;gap:16px;margin-bottom:16px}
.eg-head h1{margin:0;font-size:20px;font-weight:600;line-height:28px}
.eg-head p{margin:4px 0 0;color:var(--dsw-alias-label-secondary);font-size:13px;max-width:780px}
.eg-head-main{flex:1;min-width:0}
.eg-title-line{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.eg-actions{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.eg-back{border:0;background:transparent;color:var(--dsw-alias-label-secondary);font:inherit;font-size:13px;cursor:pointer;padding:4px 8px;margin:-4px 0 8px -8px;border-radius:8px}
.eg-back:hover{background:var(--dsw-alias-interactive-bg-hover);color:var(--dsw-alias-label-primary)}
.eg-card{border:.5px solid var(--dsw-alias-border-l4);border-radius:16px;padding:16px 18px;margin-top:12px;background:transparent}
.eg-card-head{display:flex;align-items:flex-start;gap:12px;margin-bottom:10px}
.eg-card-head h2{margin:0;font-size:15px;font-weight:600;line-height:22px}
.eg-card-head p{margin:2px 0 0;color:var(--dsw-alias-label-secondary);font-size:12.5px}
.eg-hint{color:var(--dsw-alias-label-secondary);font-size:13px}
.eg-faint{color:var(--dsw-alias-label-tertiary)}
.eg-mono{font-family:var(--ds-font-family-code,ui-monospace,SFMono-Regular,Menlo,monospace);font-size:12px}
.eg-good{color:var(--dsw-alias-state-success-primary)}
.eg-bad{color:var(--dsw-alias-state-error-primary)}
.eg-warn{color:var(--dsw-alias-state-warn-primary)}
.eg-stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-top:12px}
.eg-stat{background:var(--eg-fill);border-radius:12px;padding:12px 14px;min-width:0}
.eg-stat-label{font-size:12px;color:var(--dsw-alias-label-tertiary)}
.eg-stat-value{font-size:20px;font-weight:600;line-height:28px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.eg-stat-sub{font-size:12px;color:var(--dsw-alias-label-secondary);overflow-wrap:anywhere}
.eg-progress{height:4px;border-radius:2px;background:var(--dsw-alias-border-l2);margin-top:8px;overflow:hidden}
.eg-progress i{display:block;height:100%;background:var(--dsw-alias-state-business-primary)}
.eg-list{display:flex;flex-direction:column;gap:8px}
.eg-row{display:flex;align-items:flex-start;gap:12px;width:100%;padding:12px 14px;border:.5px solid var(--dsw-alias-border-l4);border-radius:14px;background:transparent;color:inherit;font:inherit;text-align:left;cursor:pointer}
.eg-row:hover{background:var(--dsw-alias-interactive-bg-hover)}
.eg-row-dot{padding-top:6px;flex:none;display:inline-flex}
.eg-row-main{flex:1;min-width:0}
.eg-row-title{font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.eg-row-meta{font-size:12.5px;color:var(--dsw-alias-label-secondary);overflow-wrap:anywhere}
.eg-row-side{flex:none;display:flex;flex-direction:column;align-items:flex-end;gap:4px;font-size:12px;color:var(--dsw-alias-label-tertiary)}
.eg-tag{display:inline-flex;align-items:center;height:20px;padding:0 7px;border-radius:6px;font-size:12px;white-space:nowrap;background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-secondary)}
.eg-tag[data-tone=success]{color:var(--dsw-alias-state-success-primary)}
.eg-tag[data-tone=danger]{color:var(--dsw-alias-state-error-primary)}
.eg-tag[data-tone=warning]{color:var(--dsw-alias-state-warn-primary)}
.eg-tag[data-tone=info]{color:var(--dsw-alias-state-business-primary)}
.eg-dot{width:8px;height:8px;border-radius:50%;display:inline-block;background:var(--dsw-alias-label-tertiary)}
.eg-dot[data-state=ongoing]{background:var(--dsw-alias-state-business-primary)}
.eg-dot[data-state=done]{background:var(--dsw-alias-state-success-primary)}
.eg-dot[data-state=error]{background:var(--dsw-alias-state-error-primary)}
.eg-dot[data-state=warning]{background:var(--dsw-alias-state-warn-primary)}
.eg-btn{height:32px;padding:0 14px;border-radius:16px;border:.5px solid var(--dsw-alias-border-l4);background:transparent;color:var(--dsw-alias-label-primary);font:inherit;font-size:13px;cursor:pointer;white-space:nowrap}
.eg-btn:hover:not(:disabled){background:var(--dsw-alias-interactive-bg-hover)}
.eg-btn:disabled{opacity:.5;cursor:default}
.eg-btn[data-variant=primary]{background:var(--dsw-alias-button-primary-fill,var(--dsw-alias-brand-primary));border-color:transparent;color:var(--dsw-alias-label-primary-inverted,#fff)}
.eg-table-wrap{overflow-x:auto}
.eg-table{width:100%;border-collapse:collapse;font-size:13px}
.eg-table th{font-size:12px;font-weight:500;color:var(--dsw-alias-label-tertiary);text-align:left;padding:6px 10px;border-bottom:.5px solid var(--dsw-alias-border-l2);white-space:nowrap}
.eg-table td{padding:7px 10px;border-bottom:.5px solid var(--dsw-alias-border-l2);vertical-align:top}
.eg-table tr:last-child td{border-bottom:0}
.eg-table .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.eg-table tr.eg-click{cursor:pointer}
.eg-table tr.eg-click:hover td{background:var(--dsw-alias-interactive-bg-hover)}
.eg-table tfoot td{border-top:.5px solid var(--dsw-alias-border-l3);font-weight:600}
.eg-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.eg-chip{display:inline-flex;align-items:center;gap:4px;height:22px;padding:0 8px;border-radius:11px;background:var(--eg-fill);font-size:12px;color:var(--dsw-alias-label-secondary);white-space:nowrap}
.eg-kv{display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;margin:0;font-size:13px}
.eg-kv dt{color:var(--dsw-alias-label-tertiary)}
.eg-kv dd{margin:0;min-width:0;overflow-wrap:anywhere}
.eg-meta{display:flex;flex-wrap:wrap;gap:4px 18px;margin-top:6px;font-size:12.5px;color:var(--dsw-alias-label-secondary)}
.eg-meta b{font-weight:400;color:var(--dsw-alias-label-tertiary);margin-right:6px}
.eg-alert{border-radius:12px;padding:10px 14px;margin-top:12px;font-size:13px;background:var(--eg-fill)}
.eg-alert[data-tone=danger]{background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 10%,transparent)}
.eg-alert[data-tone=warning]{background:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 12%,transparent)}
.eg-alert[data-tone=info]{background:color-mix(in srgb,var(--dsw-alias-state-business-primary) 10%,transparent)}
.eg-alert b{font-weight:600}
.eg-pre{margin:8px 0 0;padding:10px 12px;border-radius:10px;background:var(--eg-fill);font-family:var(--ds-font-family-code,ui-monospace,monospace);font-size:12px;line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto}
.eg-diff{margin:8px 0 0;padding:8px 0;border-radius:10px;background:var(--eg-fill);font-family:var(--ds-font-family-code,ui-monospace,monospace);font-size:12px;line-height:1.5;max-height:420px;overflow:auto}
.eg-diff span{display:block;padding:0 12px;white-space:pre-wrap;overflow-wrap:anywhere}
.eg-diff .add{background:color-mix(in srgb,var(--dsw-alias-state-success-primary) 14%,transparent)}
.eg-diff .del{background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 12%,transparent)}
.eg-diff .meta{color:var(--dsw-alias-label-tertiary)}
.eg-md{font-size:13.5px;overflow-wrap:anywhere}
.eg-details{margin-top:8px}
.eg-details>summary{cursor:pointer;font-size:13px;color:var(--dsw-alias-label-secondary);list-style:none;display:inline-flex;align-items:center;gap:6px;padding:2px 0}
.eg-details>summary::-webkit-details-marker{display:none}
.eg-details>summary::before{content:"";width:6px;height:6px;border-right:1.5px solid currentColor;border-bottom:1.5px solid currentColor;transform:rotate(-45deg);transition:transform .15s}
.eg-details[open]>summary::before{transform:rotate(45deg)}
.eg-round{border:.5px solid var(--dsw-alias-border-l4);border-radius:14px;margin-top:8px}
.eg-round>summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:10px;padding:11px 14px;border-radius:14px}
.eg-round>summary::-webkit-details-marker{display:none}
.eg-round>summary:hover{background:var(--dsw-alias-interactive-bg-hover)}
.eg-round-no{font-weight:600;white-space:nowrap}
.eg-round-sum{flex:1;min-width:0;font-size:12.5px;color:var(--dsw-alias-label-secondary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.eg-round-body{padding:4px 14px 14px}
.eg-cand{border-radius:12px;background:var(--eg-fill);padding:12px 14px;margin-top:10px}
.eg-cand[data-adopted=true]{box-shadow:inset 3px 0 0 var(--dsw-alias-state-success-primary)}
.eg-cand-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.eg-cand-summary{margin-top:6px;font-size:13px;overflow-wrap:anywhere}
.eg-line{margin-top:6px;font-size:13px}
.eg-forest{display:grid;grid-template-columns:max-content minmax(160px,1fr) max-content;gap:8px 14px;align-items:center;font-size:12.5px}
.eg-track{position:relative;height:22px}
.eg-track i{position:absolute;top:50%}
.eg-track .zero{top:0;bottom:0;width:1px;background:var(--dsw-alias-border-l3)}
.eg-track .gate{top:0;bottom:0;width:0;border-left:1px dashed var(--dsw-alias-state-warn-primary)}
.eg-track .ci{height:4px;margin-top:-2px;border-radius:2px;background:var(--dsw-alias-label-tertiary)}
.eg-track .dot{width:10px;height:10px;margin:-5px 0 0 -5px;border-radius:50%;background:var(--dsw-alias-label-secondary)}
.eg-track[data-adopted=true] .ci,.eg-track[data-adopted=true] .dot{background:var(--dsw-alias-state-success-primary)}
.eg-axis{position:relative;height:16px;font-size:11px;color:var(--dsw-alias-label-tertiary)}
.eg-axis span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.eg-panel{border-radius:12px;background:var(--eg-fill);padding:12px 14px;min-width:0}
.eg-panel h3{margin:0 0 6px;font-size:13.5px;font-weight:600}
.eg-matrix{display:grid;gap:4px;font-size:12px}
.eg-mc{border-radius:8px;padding:8px 10px;background:var(--eg-fill);font-variant-numeric:tabular-nums}
.eg-mc b{display:block;font-size:15px;font-weight:600}
.eg-mh{color:var(--dsw-alias-label-tertiary);padding:2px 10px;align-self:end}
.eg-form{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px 16px}
.eg-field{display:flex;flex-direction:column;gap:5px;min-width:0}
.eg-field>span{font-size:12.5px;color:var(--dsw-alias-label-secondary)}
.eg-field small{font-size:12px;color:var(--dsw-alias-label-tertiary)}
.eg-input{box-sizing:border-box;width:100%;border:.5px solid var(--dsw-alias-border-l4);background:var(--dsw-alias-bg-layer-1);color:var(--dsw-alias-label-primary);font:inherit;font-size:13px;border-radius:10px;padding:8px 12px;outline:none}
.eg-input:focus{border-color:var(--dsw-alias-brand-primary)}
textarea.eg-input{resize:vertical;min-height:72px;line-height:1.5}
.eg-check{display:inline-flex;align-items:center;gap:8px;font-size:13px;cursor:pointer}
.eg-empty{text-align:center;padding:56px 24px;color:var(--dsw-alias-label-secondary)}
.eg-empty h2{margin:12px 0 6px;font-size:16px;font-weight:600;color:var(--dsw-alias-label-primary)}
.eg-empty p{margin:0 auto 16px;max-width:560px;font-size:13px}
.eg-log{margin-top:10px;max-height:260px}
.eg-toasts{position:absolute;right:20px;bottom:20px;display:flex;flex-direction:column;gap:8px;z-index:5;pointer-events:none}
.eg-toast{pointer-events:auto;max-width:420px;padding:10px 14px;border-radius:12px;font-size:13px;background:var(--dsw-alias-bg-layer-3,var(--dsw-alias-bg-layer-2));box-shadow:var(--dsw-elevation-panel,0 4px 16px rgba(0,0,0,.18));border:.5px solid var(--dsw-alias-border-l3)}
.eg-toast[data-tone=danger]{border-color:var(--dsw-alias-state-error-primary)}
.eg-q{font-weight:500;overflow-wrap:anywhere}
.eg-note{padding:8px 0;border-bottom:.5px solid var(--dsw-alias-border-l2);font-size:13px;overflow-wrap:anywhere}
.eg-note:last-child{border-bottom:0}
.eg-two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}
.eg-flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px}
.eg-step{border-radius:12px;background:var(--eg-fill);padding:10px 12px;min-width:0}
.eg-step b{display:flex;align-items:center;gap:7px;font-size:13.5px;font-weight:600}
.eg-step i{font-style:normal;flex:none;display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;background:var(--dsw-alias-state-business-primary);color:#fff;font-size:11px}
.eg-step p{margin:5px 0 0;font-size:12.5px;color:var(--dsw-alias-label-secondary)}
.eg-layers{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
.eg-layers p{margin:0}
.eg-timeline{position:relative;height:30px;border-radius:8px;background:var(--dsw-alias-bg-layer-2);overflow:hidden}
.eg-seg{position:absolute;top:0;bottom:0;display:flex;align-items:center;padding:0 6px;font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.eg-seg[data-kind=visible],.eg-swatch[data-kind=visible]{background:color-mix(in srgb,var(--dsw-alias-state-business-primary) 22%,transparent)}
.eg-seg[data-kind=local],.eg-swatch[data-kind=local]{background:color-mix(in srgb,var(--dsw-alias-state-business-primary) 45%,transparent)}
.eg-seg[data-kind=hidden],.eg-swatch[data-kind=hidden]{background:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 38%,transparent)}
.eg-seg[data-kind=sealed],.eg-swatch[data-kind=sealed]{background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 24%,transparent)}
.eg-swatch{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px;vertical-align:-1px}
.eg-stack{display:flex;flex-direction:column}
.eg-tier{border-radius:12px;background:var(--eg-fill);padding:10px 14px}
.eg-tier-head{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.eg-tier-head b{font-weight:600;font-size:13.5px}
.eg-tier p{margin:4px 0 0;font-size:12.5px;color:var(--dsw-alias-label-secondary)}
.eg-tier-arrow{text-align:center;color:var(--dsw-alias-label-tertiary);line-height:20px}
@media (max-width:900px){.eg-two{grid-template-columns:1fr}.eg-page{padding:20px 18px 64px}.eg-bar{padding:0 12px}}
`;
    const styleId = "@ecogym/dsh-plugin/client";
    if (typeof document !== "undefined" && !document.querySelector(`style[data-plugin-css="${styleId}"]`)) {
      const tag = document.createElement("style");
      tag.dataset.plugin = "@ecogym/dsh-plugin";
      tag.dataset.pluginCss = styleId;
      tag.textContent = css;
      document.head.append(tag);
    }

    /* ---- data --------------------------------------------------------------------------------------------- */

    // Every backend call the panel makes. "_status" and "_restart" are answered by the plugin's host half
    // (lib/index.js); the rest is forwarded to the ecogym service, whose ROUTES table (service.py, also
    // GET routes) lists every endpoint. Pages read with useResource (polling) and write with api(path, { body }).
    const enc = encodeURIComponent;
    const Api = {
      status: "_status", restart: "_restart",
      catalog: "catalog", routes: "routes", stages: (task) => `tasks/${enc(task)}/stages`,
      evolutions: "evolutions", evolution: (id) => `evolutions/${enc(id)}`,
      evolutionAction: (id, action) => `evolutions/${enc(id)}/${action}`,
      episodes: "episodes", episode: (id) => `episodes/${enc(id)}`,
      runningEpisodes: (evolution) => `episodes?evolution=${enc(evolution)}&status=running`,
      harness: (ref) => `harnesses/${enc(ref)}`, harnessDiff: (before, after) => `harnesses/${enc(before)}/diff/${enc(after)}`,
      modelCheck: "models/check",
      expert: "expert", expertAnswer: "expert/answer", expertNote: "expert/note",
      exports: "exports", exportDetail: (id) => `exports/${enc(id)}`,
    };

    async function api(path, options = {}) {
      const init = { method: options.method || "GET", headers: { accept: "application/json" }, credentials: "same-origin" };
      if (options.body !== undefined) {
        init.method = options.method || "POST";
        init.headers["content-type"] = "application/json";
        init.headers["x-ecogym"] = "1";
        init.body = JSON.stringify(options.body);
      }
      const response = await fetch(API + path, init);
      const text = await response.text();
      let value = null;
      try {
        value = text ? JSON.parse(text) : null;
      } catch {
        value = text;
      }
      if (!response.ok) {
        const error = new Error((value && value.error) || `HTTP ${response.status}`);
        error.status = response.status;
        error.body = value;
        throw error;
      }
      return value;
    }

    /** Poll a JSON resource; re-renders only when the payload changes. */
    function useResource(path, { interval = 0, enabled = true } = {}) {
      const [state, setState] = useState({ data: undefined, error: null, path });
      const [tick, setTick] = useState(0);
      const last = useRef({ path: null, text: "" });
      useEffect(() => {
        if (!enabled || !path) return undefined;
        if (last.current.path !== path) {
          last.current = { path, text: "" };
          setState({ data: undefined, error: null, path });
        }
        let alive = true;
        let timer = null;
        const load = async () => {
          if (document.hidden && last.current.text) {
            timer = setTimeout(load, interval || 5000);
            return;
          }
          try {
            const value = await api(path);
            if (!alive) return;
            const text = JSON.stringify(value);
            if (text !== last.current.text) {
              last.current.text = text;
              setState({ data: value, error: null, path });
            } else {
              setState((old) => (old.error ? { ...old, error: null } : old));
            }
          } catch (error) {
            if (alive) setState((old) => ({ ...old, error }));
          }
          if (alive && interval) timer = setTimeout(load, interval);
        };
        load();
        return () => {
          alive = false;
          clearTimeout(timer);
        };
      }, [path, interval, enabled, tick]);
      const reload = useCallback(() => setTick((value) => value + 1), []);
      return { data: state.path === path ? state.data : undefined, error: state.error, reload };
    }

    /** Re-render every few seconds while something runs, so elapsed times keep moving. */
    function useTick(enabled, ms = 5000) {
      const [, setTick] = useState(0);
      useEffect(() => {
        if (!enabled) return undefined;
        const timer = setInterval(() => setTick((value) => value + 1), ms);
        return () => clearInterval(timer);
      }, [enabled, ms]);
    }

    let catalogCache = null;
    const CATALOG_RETRY_MS = [2000, 5000, 15000, 30000];
    /** The task catalog, read once; a failure is shown and retried with backoff. */
    function useCatalog(ready) {
      const [state, setState] = useState({ catalog: catalogCache, error: null });
      useEffect(() => {
        if (!ready || catalogCache) return undefined;
        let alive = true;
        let timer = null;
        let attempt = 0;
        const load = () => api(Api.catalog).then((value) => {
          catalogCache = value;
          if (alive) setState({ catalog: value, error: null });
        }).catch((error) => {
          if (!alive) return;
          setState({ catalog: null, error });
          timer = setTimeout(load, CATALOG_RETRY_MS[Math.min(attempt++, CATALOG_RETRY_MS.length - 1)]);
        });
        load();
        return () => {
          alive = false;
          clearTimeout(timer);
        };
      }, [ready]);
      return state;
    }

    function CatalogPending({ error, text = "正在读取…" }) {
      return error ? h(Alert, { tone: "warning" }, `读不到任务目录：${error.message}。稍后会自动重试。`)
        : h("p", { className: "eg-hint" }, text);
    }

    /** DSH's Session Remote, read without an inject requirement: the panel still works on a host without it. */
    function dshSession() {
      try {
        const session = host && typeof host.get === "function" ? host.get("remote.session") : null;
        return session && typeof session.modelCatalog === "function" ? session : null;
      } catch {
        return null;
      }
    }

    function withTimeout(promise, ms) {
      return Promise.race([promise, new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), ms))]);
    }

    /**
     * DSH's model catalog ({ok, value}) over the same-origin RPC route, the typed Remote client as fallback. DSH
     * builds the catalog by asking every configured provider, so it can take long when one is unreachable.
     */
    async function dshModelCatalog() {
      const abort = new AbortController();
      const timer = setTimeout(() => abort.abort(), 8000);
      try {
        const rpcId = globalThis.crypto?.randomUUID?.() || `ecogym-${Date.now()}`;
        const response = await fetch("/api/session/modelCatalog", {
          method: "POST", credentials: "same-origin", headers: { "content-type": "application/json" }, signal: abort.signal,
          body: JSON.stringify({ type: "client-request", rpcId, method: "session/modelCatalog", payload: { args: {} } }),
        });
        const body = response.ok ? await response.json() : null;
        const result = body?.type === "server-response" ? body.result : body;
        if (result?.ok) return result;
      } catch (error) {
        if (error?.name === "AbortError") throw new Error("DSH did not answer within 8 s");
      } finally {
        clearTimeout(timer);
      }
      const session = dshSession();
      return session ? withTimeout(Promise.resolve().then(() => session.modelCatalog()), 8000) : null;
    }

    const DEEPSEEK = "deepseek-official";
    const DEEPSEEK_ID = /^deepseek-[a-z0-9.-]{1,40}$/;
    /** A model reference as the pickers store it: <provider>/<model>; a bare DeepSeek id is DeepSeek's own. */
    function modelRef(value) {
      const text = String(value || "").trim();
      return DEEPSEEK_ID.test(text) ? `${DEEPSEEK}/${text}` : text;
    }

    /**
     * Model routes for the pickers, from the service (it reads DSH's provider settings and knows which routes it
     * can run and whether their keys exist). DeepSeek's own model list comes from DSH's catalog when DSH answers.
     */
    function useModelRoutes(catalog) {
      const [deepseek, setDeepseek] = useState([]);
      useEffect(() => {
        let alive = true;
        dshModelCatalog().then((response) => {
          if (!response?.ok) throw new Error(response?.error?.message || "no catalog");
          const group = (response.value?.groups || []).find((item) => item.id === DEEPSEEK);
          const models = (group?.models || []).filter((model) => DEEPSEEK_ID.test(model.id));
          if (alive) setDeepseek(models.map((model) => ({ id: model.id, name: model.name || model.id })));
        }).catch((error) => console.warn("[ecogym] no DSH model catalog; DeepSeek shows ecogym's defaults:", error?.message));
        return () => {
          alive = false;
        };
      }, []);
      return useMemo(() => (catalog?.models?.routes || []).map((route) => {
        const models = route.provider === DEEPSEEK && deepseek.length ? deepseek : route.models.map((id) => ({ id, name: id }));
        return { ...route, options: models.map((model) => ({ value: `${route.provider}/${model.id}`, // provider shown: ids repeat across routes
          label: `${model.name && model.name !== model.id ? `${model.id}（${model.name}）` : model.id} · ${route.name}` })) };
      }), [catalog, deepseek]);
    }

    function routeNote(route) {
      const api = route.api === "deepseek" ? "DeepSeek 官方接口" : route.api;
      return `${route.name} · ${api} · ${route.host} · 密钥 ${route.key_env}${route.key_configured ? " 已配置" : " 未配置"}`;
    }

    const modelChecks = new Map();
    /** The service's answer to "can this model run now?" (it sends one tiny request); only "yes" is kept here. */
    function useModelCheck(ref) {
      const [state, setState] = useState(() => modelChecks.get(ref) || null);
      useEffect(() => {
        if (!ref) {
          setState(null);
          return undefined;
        }
        if (modelChecks.has(ref)) {
          setState(modelChecks.get(ref));
          return undefined;
        }
        let alive = true;
        setState({ pending: true });
        const timer = setTimeout(() => api(Api.modelCheck, { body: { model: ref } }).then((value) => {
          if (value?.ok) modelChecks.set(ref, value); // a refusal is asked again next time (a key may be added)
          if (alive) setState(value);
        }).catch((error) => {
          if (alive) setState({ ok: false, error: `测试失败：${error.message}` });
        }), 300);
        return () => {
          alive = false;
          clearTimeout(timer);
        };
      }, [ref]);
      return state;
    }

    /** One model picker: every route grouped by provider; routes that cannot run are shown but not selectable. */
    function ModelSelect({ label, value, onChange, routes, empty, note }) {
      const ref = modelRef(value);
      const route = routes.find((item) => ref.startsWith(`${item.provider}/`));
      const listed = routes.some((item) => item.options.some((option) => option.value === ref));
      const check = useModelCheck(listed && route?.runnable ? ref : "");
      let hint = note;
      if (ref && !listed) hint = "这个模型不在 DSH 当前的设置里，开始运行时会被拒绝；请换一个。";
      else if (ref && route) hint = route.runnable ? routeNote(route) : `不可用：${route.problem}`;
      const tested = !check ? null : check.pending ? h("span", { className: "eg-faint" }, "正在测试能否调用…")
        : check.ok ? h("span", { className: "eg-good" }, check.note && check.note !== "ok" ? `测试请求：${check.note}` : "已测试：可以调用")
          : h("span", { className: "eg-bad" }, check.error);
      return h("label", { className: "eg-field" }, h("span", null, label),
        h("select", { className: "eg-input", value: ref, onChange },
          empty ? h("option", { value: "" }, empty) : null,
          ref && !listed ? h("option", { value: ref }, `${ref}（当前不可用）`) : null,
          routes.map((item) => h("optgroup", { key: item.provider,
            label: `${item.name}${item.host ? ` · ${item.host}` : ""}${item.runnable ? "" : `（不可用：${item.problem}）`}` },
          item.options.map((option) => h("option", { key: option.value, value: option.value, disabled: !item.runnable }, option.label))))),
        hint ? h("small", null, hint) : null,
        tested ? h("small", null, tested) : null);
    }

    /* ---- toasts and confirmation ---------------------------------------------------------------------------- */

    const toastListeners = new Set();
    function toast(text, tone = "neutral") {
      for (const listener of toastListeners) listener({ id: Math.random(), text, tone });
    }
    function Toasts() {
      const [items, setItems] = useState([]);
      useEffect(() => {
        const add = (item) => {
          setItems((old) => [...old.slice(-3), item]);
          setTimeout(() => setItems((old) => old.filter((entry) => entry.id !== item.id)), 4500);
        };
        toastListeners.add(add);
        return () => toastListeners.delete(add);
      }, []);
      return h("div", { className: "eg-toasts", "aria-live": "polite" },
        items.map((item) => h("div", { key: item.id, className: "eg-toast", "data-tone": item.tone }, item.text)));
    }

    function useConfirm() {
      const [request, setRequest] = useState(null);
      const ask = useCallback((options) => {
        if (typeof P.Modal !== "function") return Promise.resolve(window.confirm(`${options.title}\n\n${options.text || ""}`));
        return new Promise((resolve) => setRequest({ ...options, resolve }));
      }, []);
      const close = (answer) => {
        request?.resolve(answer);
        setRequest(null);
      };
      const element = request && typeof P.Modal === "function" ? h(P.Modal, {
        open: true, onClose: () => close(false), title: request.title, closeLabel: "关闭", description: request.text,
        footer: h(Fragment, null,
          h(Button, { variant: "outline", onClick: () => close(false) }, "取消"),
          h(Button, { variant: "primary", onClick: () => close(true) }, request.confirm || "确定")),
      }, request.body || null) : null;
      return [element, ask];
    }

    /* ---- formatting ------------------------------------------------------------------------------------------ */

    const isNum = (value) => typeof value === "number" && Number.isFinite(value);
    function signed(value, digits = 4) {
      if (!isNum(value)) return "–";
      const text = Math.abs(value).toFixed(digits);
      if (Number(text) === 0) return (0).toFixed(digits);
      return (value > 0 ? "+" : MINUS) + text;
    }
    function plain(value, digits = 3) {
      return isNum(value) ? (value < 0 ? MINUS : "") + Math.abs(value).toFixed(digits) : "–";
    }
    function toneClass(value) {
      if (!isNum(value) || Math.abs(value) < 5e-5) return "";
      return value > 0 ? "eg-good" : "eg-bad";
    }
    function timeOf(value) {
      if (isNum(value)) return value * 1000;
      const time = Date.parse(value ?? "");
      return Number.isNaN(time) ? null : time;
    }
    const pad = (n) => String(n).padStart(2, "0");
    function when(value, year = false) {
      const time = timeOf(value);
      if (time === null) return "–";
      const d = new Date(time);
      return `${year ? `${d.getFullYear()}-` : ""}${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
    }
    function duration(seconds) {
      if (!isNum(seconds) || seconds < 0) return "–";
      const s = Math.round(seconds);
      if (s < 60) return `${s} 秒`;
      const m = Math.floor(s / 60);
      if (m < 60) return `${m} 分`;
      return m % 60 ? `${Math.floor(m / 60)} 小时 ${m % 60} 分` : `${Math.floor(m / 60)} 小时`;
    }
    function tokens(value) {
      if (!isNum(value) || value <= 0) return "–";
      if (value >= 1e6) return `${(value / 1e6).toFixed(value >= 1e7 ? 0 : 1)}M`;
      return value >= 1e3 ? `${Math.round(value / 1e3)}K` : String(value);
    }
    function bytes(value) {
      if (!isNum(value)) return "–";
      if (value >= 1 << 20) return `${(value / (1 << 20)).toFixed(1)} MB`;
      return value >= 1024 ? `${Math.round(value / 1024)} KB` : `${value} B`;
    }
    const short = (digest) => String(digest || "").slice(0, 12);
    const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : null);
    function copy(text) {
      navigator.clipboard?.writeText(text).then(() => toast(`已复制 ${text}`), () => toast(text));
    }
    function mix(value, scale) {
      if (!isNum(value) || value === 0) return undefined;
      const share = Math.round(6 + Math.min(Math.abs(value) / scale, 1) * 26);
      const color = value > 0 ? "var(--dsw-alias-state-success-primary)" : "var(--dsw-alias-state-error-primary)";
      return `color-mix(in srgb, ${color} ${share}%, transparent)`;
    }

    const EVO_STATUS = {
      running: ["运行中", "info", "ongoing"], finished: ["已完成", "success", "done"], failed: ["失败", "danger", "error"],
      stopped: ["已停止", "warning", "warning"], interrupted: ["已中断", "warning", "warning"],
      elsewhere: ["运行中（其他进程）", "info", "ongoing"],
    };
    const EP_STATUS = {
      running: ["运行中", "info", "ongoing"], finished: ["已完成", "success", "done"], timeout: ["到时结束", "success", "done"],
      infra_failed: ["服务故障", "danger", "error"], interrupted: ["已中断", "warning", "warning"],
      unfinished: ["未结束", "warning", "warning"],
    };
    const CAND_STATUS = {
      proposed: ["已提出", "neutral"], adopted: ["已采纳", "success"], passed: ["通过门槛", "success"],
      not_adopted: ["未采纳", "neutral"], invalid: ["未通过检查", "danger"], unchanged: ["没有改动", "neutral"],
      duplicate: ["与以往版本相同", "warning"], rejected: ["被拒绝", "danger"],
    };
    const ROLE = { incumbent: "在任", candidate: "候选", holdout: "留出验证" };
    const SCORED = ["finished", "timeout"];

    function evoKey(run) {
      if (run.active) return "running";
      if (run.status === "running") return run.stale ? "interrupted" : "elsewhere";
      return EVO_STATUS[run.status] ? run.status : "failed";
    }
    function epKey(item) {
      if (item.status !== "running") return EP_STATUS[item.status] ? item.status : "finished";
      if (item.orphaned) return "unfinished"; // its run is not running here and it started before this backend
      const started = timeOf(item.started_at);
      return started !== null && Date.now() - started > ((item.minutes || 30) + 20) * 60000 ? "unfinished" : "running";
    }

    function taskInfo(catalog, id) {
      const card = String(id || "").split(":")[0];
      return (catalog?.tasks || []).find((task) => task.id === card) || null;
    }
    function taskShort(catalog, id) {
      const [card, episode] = String(id || "").split(":");
      const task = taskInfo(catalog, card);
      if (!task) return String(id || "");
      return `${episode || task.episodes?.[0] || ""}（${task.dataset}）`;
    }
    /** A task in a table cell: its episode, with the dataset in a quieter tone. */
    function TaskLabel({ catalog, id }) {
      const [card, episode] = String(id || "").split(":");
      const task = taskInfo(catalog, card);
      if (!task) return h("span", { title: id }, String(id || ""));
      return h("span", { title: id }, episode || task.episodes?.[0] || "", h("span", { className: "eg-faint" }, ` ${task.dataset}`));
    }
    function reasonText(catalog, reason) {
      if (!reason) return "";
      const fixed = { improved: "通过全部门槛", below_practical_delta: "平均增益没有超过实用门槛",
        interval_crosses_zero: "增益的置信区间包含 0", better: "更好", worse: "更差", within_uncertainty: "在不确定范围内",
        credible_regression: "可信地变差", unconfirmed_regression: "变差，但只有一个回合，不作否决" };
      if (fixed[reason]) return fixed[reason];
      const list = (text) => text.split(",").map((task) => taskShort(catalog, task)).join("、");
      if (reason.startsWith("task_regression:")) return `有任务可信变差：${list(reason.slice(16))}`;
      if (reason.startsWith("missing_tasks:")) return `这些任务缺少可用回合：${list(reason.slice(14))}`;
      return reason;
    }
    function horizonText(task, horizon) {
      const match = /^(\d+)(min|h|d)$/.exec(task?.step || "");
      if (!match) return `${horizon} 步`;
      const n = Number(match[1]) * horizon;
      return match[2] === "d" ? `${n} 天` : `${n} ${match[2]}`;
    }
    const UNITS = { degC: "°C", percent: "%" };

    /* ---- primitives (DSH's own when present) -------------------------------------------------------------- */

    function Button({ variant = "outline", onClick, disabled, title, children, type = "button" }) {
      if (typeof P.Button === "function") return h(P.Button, { variant, onClick, disabled, title, type }, children);
      return h("button", { type, className: "eg-btn", "data-variant": variant, onClick, disabled, title }, children);
    }
    function Tag({ tone = "neutral", children, title }) {
      if (typeof P.Tag === "function") return h(P.Tag, { tone, title }, children);
      return h("span", { className: "eg-tag", "data-tone": tone, title }, children);
    }
    function Dot({ state }) {
      if (typeof P.StateDot === "function") return h(P.StateDot, { state });
      return h("span", { className: "eg-dot", "data-state": state });
    }
    function Markdown({ text }) {
      if (!text) return null;
      if (typeof P.MarkdownText === "function") {
        return h("div", { className: "eg-md" }, h(P.MarkdownText, {
          text, labels: { code: { copyLabel: "复制", copiedLabel: "已复制" }, footnotes: "脚注" },
        }));
      }
      return h("pre", { className: "eg-pre" }, text);
    }
    function StatusTag({ map, keyName }) {
      const [text, tone] = map[keyName] || [keyName, "neutral"];
      return h(Tag, { tone }, text);
    }
    function Card({ title, hint, actions, children }) {
      return h("section", { className: "eg-card" },
        title || actions ? h("div", { className: "eg-card-head" },
          h("div", { className: "eg-head-main" }, title ? h("h2", null, title) : null, hint ? h("p", null, hint) : null),
          actions ? h("div", { className: "eg-actions" }, actions) : null) : null,
        children);
    }
    function Stat({ label, value, sub, progress }) {
      return h("div", { className: "eg-stat" },
        h("div", { className: "eg-stat-label" }, label),
        h("div", { className: "eg-stat-value" }, value),
        sub ? h("div", { className: "eg-stat-sub" }, sub) : null,
        isNum(progress) ? h("div", { className: "eg-progress" }, h("i", { style: { width: `${Math.min(100, progress * 100)}%` } })) : null);
    }
    function Alert({ tone = "info", children }) {
      return h("div", { className: "eg-alert", "data-tone": tone }, children);
    }
    function Empty({ title, text, action }) {
      return h("div", { className: "eg-empty" }, h(Leaf, { size: 40 }), h("h2", null, title), text ? h("p", null, text) : null, action || null);
    }
    function Details({ summary, children, open = false }) {
      const [shown, setShown] = useState(open);
      return h("details", { className: "eg-details", open: shown, onToggle: (event) => setShown(event.currentTarget.open) },
        h("summary", null, summary), shown ? children : null);
    }
    function Leaf({ size = 16 }) {
      return h("svg", { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
        strokeWidth: 1.7, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": "true" },
      h("path", { d: "M5 19c0-8 5.5-13.5 14-14-.4 8.6-6 14-13 14.2" }),
      h("path", { d: "M5 19.5 13 11.5" }));
    }
    function Diff({ patch }) {
      const lines = String(patch || "").split("\n");
      return h("div", { className: "eg-diff" }, lines.map((line, index) => {
        const kind = line.startsWith("+++") || line.startsWith("---") || line.startsWith("@@") ? "meta"
          : line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : "";
        return h("span", { key: index, className: kind }, line || " ");
      }));
    }
    function KV({ pairs }) {
      return h("dl", { className: "eg-kv" }, pairs.flatMap(([key, value]) => [h("dt", { key: `k${key}` }, key), h("dd", { key: `v${key}` }, value)]));
    }
    function Mono({ children, title }) {
      return h("span", { className: "eg-mono", title }, children);
    }

    /* ---- backend state ------------------------------------------------------------------------------------- */

    function BackendCard({ status, error, onRestart }) {
      let title = "正在连接生态进化后端…";
      let text = "DSH 启动时会一并启动 ecogym 服务。";
      let tone = "info";
      if (error) {
        title = "无法连接生态进化插件";
        text = error.status === 401 ? "DSH 登录已失效，请刷新页面。" : `插件的服务端没有响应（${error.message}）。请确认 DSH 是以安装了插件的 profile 启动的。`;
        tone = "danger";
      } else if (status?.state === "unconfigured") {
        title = "插件还没有配置 Python";
        text = "在装有 ecogym 的环境里运行 ecogym dsh install，再点重新启动后端。";
        tone = "warning";
      } else if (status?.state === "failed") {
        title = "生态进化后端多次启动失败";
        text = "下面是后端的最后输出。修复后点重新启动。";
        tone = "danger";
      } else if (status?.state === "exited") {
        title = "生态进化后端已退出，正在自动重启";
        text = "进行中的进化运行会停在最后一个完成的代，后端恢复后可以继续。";
        tone = "warning";
      }
      return h(Card, { title, hint: text, actions: status && ["failed", "exited", "unconfigured"].includes(status.state)
        ? h(Button, { variant: "primary", onClick: onRestart }, "重新启动后端") : null },
      h(Alert, { tone }, h(KV, { pairs: [
        ["状态", status?.state || "–"], ["Python", h(Mono, null, status?.python || "–")],
        ["ECOGYM_HOME", h(Mono, null, status?.home || "默认（~/.ecogym）")],
        ["设置", h(Mono, null, status?.settings || "–")],
      ] })),
      status?.log?.length ? h("pre", { className: "eg-pre eg-log" }, status.log.slice(-40).join("\n")) : null);
    }

    /* ---- evolution runs ------------------------------------------------------------------------------------ */

    function EvolutionList({ go, catalog }) {
      const { data, error } = useResource(Api.evolutions, { interval: 8000 });
      const runs = data || [];
      return h(Fragment, null,
        h("div", { className: "eg-head" },
          h("div", { className: "eg-head-main" }, h("h1", null, "Harness 进化"),
            h("p", null, "DSH 研究智能体按一版 Harness 独立解每个训练任务，宿主在隐藏时期逐时刻打分；评审智能体读成绩和完整轨迹，改指令、技能、代码库、DSH 设置和经验，提出候选。候选只有在多任务配对比较中可信地变好才被采纳，最后在留出任务和封存时期各验证一次。",
              " ", h("a", { href: "#", onClick: (event) => { event.preventDefault(); go({ tab: "about", page: null }); } }, "整个流程见「介绍」")))),
        error ? h(Alert, { tone: "danger" }, `读取失败：${error.message}`) : null,
        data && !runs.length ? h(Empty, { title: "还没有进化运行", text: "选择任务组和预算，开始第一次 Harness 进化。运行在后台进行，可以随时离开这个页面。",
          action: h(Button, { variant: "primary", onClick: () => go({ page: "new" }) }, "新建进化") }) : null,
        h("div", { className: "eg-list" }, runs.map((run) => {
          const key = evoKey(run);
          const [text, tone, dot] = EVO_STATUS[key];
          const verified = [run.verified?.holdout ? "留出" : "", ...(run.verified?.sealed || [])].filter(Boolean);
          return h("button", { key: run.run_id, className: "eg-row", onClick: () => go({ page: "evolution", id: run.run_id }) },
            h("span", { className: "eg-row-dot" }, h(Dot, { state: dot })),
            h("span", { className: "eg-row-main" },
              h("div", { className: "eg-row-title" }, run.label || run.run_id),
              h("div", { className: "eg-row-meta" }, [run.suite?.title || run.suite?.id, `第 ${run.rounds_finished ?? 0} / ${run.generations ?? "?"} 代`,
                `采纳 ${run.adoptions ?? 0} 次`, `${run.episodes ?? 0} 个回合`, run.tokens ? `${tokens(run.tokens)} token` : null].filter(Boolean).join(" · ")),
              h("div", { className: "eg-row-meta eg-faint" }, h(Mono, null, `${short(run.seed?.digest)} → ${short(run.incumbent)}`),
                ` · ${when(run.started_at)}`, verified.length ? ` · 已验证 ${verified.join("、")}` : "")),
            h("span", { className: "eg-row-side" }, h(Tag, { tone }, text), run.verifying ? h(Tag, { tone: "info" }, "验证中") : null));
        })));
    }

    function phaseOf(view) {
      const rounds = view.rounds || [];
      const round = rounds[rounds.length - 1];
      if (!round) return "准备中：读取任务数据和参考成绩";
      if (round.finished) return `第 ${round.generation} 代已完成，准备下一代`;
      const since = timeOf(round.started_at);
      const elapsed = since === null ? "" : `（这一代已进行 ${duration((Date.now() - since) / 1000)}）`;
      if (round.generation === 0) return `第 0 代基线：起点 Harness 正在各训练任务上运行${elapsed}`;
      if (!round.evolver) return `第 ${round.generation} 代：评审智能体正在阅读证据、编写候选 Harness${elapsed}`;
      if (!(round.episodes || []).length) return `第 ${round.generation} 代：候选与在任版本正在各训练任务上运行回合${elapsed}`;
      return `第 ${round.generation} 代：正在比较候选与在任版本${elapsed}`;
    }

    function harnessNames(view) {
      const names = new Map([[view.seed?.digest, `起点 ${view.seed?.name || "seed"}`]]);
      for (const round of view.rounds || []) {
        for (const candidate of round.candidates || []) {
          if (candidate.digest) names.set(candidate.digest, `第 ${round.generation} 代候选 ${candidate.name}`);
        }
      }
      return names;
    }

    function EvolutionPage({ id, go, catalog }) {
      const [confirmEl, ask] = useConfirm();
      const [live, setLive] = useState(true);
      const { data: view, error, reload } = useResource(Api.evolution(id), { interval: live ? 4000 : 30000 });
      const running = view ? view.active || view.verifying || view.status === "running" : true;
      useEffect(() => setLive(Boolean(running)), [running]);
      useTick(Boolean(view?.active));
      const { data: episodes } = useResource(Api.runningEpisodes(id),
        { interval: 6000, enabled: Boolean(view?.active || view?.verifying) });
      if (error && !view) return h(Alert, { tone: "danger" }, `无法读取进化运行 ${id}：${error.message}`);
      if (!view) return h("p", { className: "eg-hint" }, "正在读取进化运行…");
      const key = evoKey(view);
      const [text, tone] = EVO_STATUS[key];
      const names = harnessNames(view);
      const act = async (path, body, message) => {
        try {
          await api(Api.evolutionAction(view.run_id, path), { body: body || {} });
          toast(message);
          reload();
        } catch (failure) {
          toast(`操作失败：${failure.message}`, "danger");
        }
      };
      const stop = async () => {
        if (await ask({ title: "在本代结束后停止？", text: "当前这一代会跑完；已完成的代保留在账本里，之后可以继续。", confirm: "本代结束后停止" })) {
          act("stop", {}, "将在当前这一代结束后停止");
        }
      };
      const abort = async () => {
        if (await ask({ title: "立即中止？", text: "正在运行的智能体会被停止，这一代的工作作废；继续运行时从这一代重新开始。", confirm: "立即中止" })) {
          act("abort", {}, "正在中止");
        }
      };
      const resume = async () => {
        const elsewhere = key === "elsewhere";
        if (await ask({ title: "继续这次进化？", text: elsewhere
          ? "账本显示它仍在运行，但不是由这个后端运行的（可能在命令行里运行，也可能已经中断）。确认它已不在运行再继续，否则会重复运行。"
          : "从最后一个完成的代继续；中断的那一代会重新运行。", confirm: "继续运行" })) {
          act("resume", { force: elsewhere }, "已继续运行");
        }
      };
      const notice = key === "failed" ? h(Alert, { tone: "danger" }, h("b", null, "运行失败："), view.error || "没有记录原因")
        : key === "stopped" ? h(Alert, { tone: "warning" }, h("b", null, "已停止："), view.error || "按请求停止", "。已完成的代保留在账本中，可以继续运行。")
          : key === "interrupted" ? h(Alert, { tone: "warning" }, h("b", null, "已中断："), "运行它的进程已经退出（或很久没有新记录），可以继续运行。")
            : view.error ? h(Alert, { tone: "warning" }, view.error) : null;
      return h(Fragment, null, confirmEl,
        h("button", { className: "eg-back", onClick: () => go({ page: null }) }, "← 全部进化运行"),
        h("div", { className: "eg-head" },
          h("div", { className: "eg-head-main" },
            h("div", { className: "eg-title-line" }, h("h1", null, view.label || view.run_id), h(Tag, { tone }, text),
              view.verifying ? h(Tag, { tone: "info" }, "验证进行中") : null),
            h("div", { className: "eg-meta" },
              h("span", null, h("b", null, "任务组"), view.suite?.title || view.suite?.id),
              h("span", null, h("b", null, "起点 → 在任"), h(Mono, null, `${short(view.seed?.digest)} → ${short(view.incumbent)}`), ` ${names.get(view.incumbent) || ""}`),
              h("span", null, h("b", null, "开始"), when(view.started_at, true)),
              h("span", null, h("b", null, "运行"), h(Mono, null, view.run_id)))),
          h("div", { className: "eg-actions" },
            view.active ? h(Button, { onClick: stop }, "本代结束后停止") : null,
            view.active ? h(Button, { onClick: abort }, "立即中止") : null,
            !view.active && ["stopped", "failed", "interrupted", "elsewhere"].includes(key) ? h(Button, { variant: "primary", onClick: resume }, "继续运行") : null,
            h(Button, { onClick: () => copy(view.run_id) }, "复制 ID"))),
        notice,
        h(RunStats, { view }),
        view.active ? h(LiveCard, { view, episodes, catalog, go }) : null,
        h(Scoreboard, { view, catalog, names }),
        h(GainChart, { view }),
        h(Verification, { view, catalog, ask, act }),
        h(Rounds, { view, catalog, go }),
        h(HarnessCard, { view, names }),
        h(AdviceCard, { view, catalog }),
        h(ConfigCard, { view, catalog }));
    }

    function RunStats({ view }) {
      const rounds = view.rounds || [];
      const finished = rounds.filter((round) => round.generation > 0 && round.finished).length;
      const total = view.config?.generations ?? 0;
      const episodes = rounds.flatMap((round) => round.episodes || []);
      const failures = episodes.filter((item) => !SCORED.includes(item.status)).length;
      const candidates = rounds.flatMap((round) => round.candidates || []);
      const ran = candidates.filter((item) => !["invalid", "unchanged", "duplicate", "rejected"].includes(item.status)).length;
      const sessions = [...episodes, ...rounds.map((round) => round.evolver).filter(Boolean)];
      const used = sessions.reduce((sum, item) => sum + (item.tokens || 0), 0);
      const metered = sessions.reduce((sum, item) => sum + (item.metered_tokens || 0), 0);
      const start = timeOf(view.started_at);
      const end = view.active || view.status === "running" ? Date.now() : timeOf(view.finished_at || view.updated_at);
      return h("div", { className: "eg-stats" },
        h(Stat, { label: "进度", value: `${finished} / ${total} 代`, sub: rounds[0]?.finished ? "基线已完成" : "基线进行中",
          progress: view.status === "running" && total ? finished / total : undefined }),
        h(Stat, { label: "采纳", value: `${(view.adoptions || []).length} 次`,
          sub: (view.adoptions || []).map((item) => `第 ${item.generation} 代 ${item.name}`).join("、") || "还没有" }),
        h(Stat, { label: "候选", value: `${candidates.length} 个`, sub: `${ran} 个通过检查并运行` }),
        h(Stat, { label: "回合", value: `${episodes.length} 个`, sub: failures ? h("span", { className: "eg-bad" }, `${failures} 个故障或中断`) : "全部正常结束" }),
        h(Stat, { label: "Token", value: tokens(used), sub: metered ? `计费 ${tokens(metered)}（缓存按 1/10）` : "智能体与评审合计" }),
        h(Stat, { label: "用时", value: duration(start !== null && end !== null ? (end - start) / 1000 : null),
          sub: view.finished_at ? `结束于 ${when(view.finished_at)}` : "进行中" }));
    }

    function LiveCard({ view, episodes, catalog, go }) {
      const live = (episodes || []).filter((item) => item.context?.evolution === view.run_id && item.status === "running");
      return h(Card, { title: "正在进行", hint: phaseOf(view) },
        live.length ? h("div", { className: "eg-table-wrap" }, h("table", { className: "eg-table" },
          h("thead", null, h("tr", null, h("th", null, "任务"), h("th", null, "版本"), h("th", null, "模型"), h("th", { className: "num" }, "已运行"), h("th", null, "回合"))),
          h("tbody", null, live.map((item) => h("tr", { key: item.episode, className: "eg-click", onClick: () => go({ page: "episode", id: item.episode }) },
            h("td", null, h(TaskLabel, { catalog, id: item.task_id })),
            h("td", null, `${ROLE[item.context?.role] || item.context?.role || ""}${item.context?.candidate ? ` ${item.context.candidate}` : ""}`),
            h("td", null, item.model || "–"),
            h("td", { className: "num" }, duration((Date.now() - (timeOf(item.started_at) || Date.now())) / 1000)),
            h("td", null, h(Mono, null, item.episode))))))) : h("p", { className: "eg-hint" }, "现在没有运行中的回合。"));
    }

    function Scoreboard({ view, catalog, names }) {
      const tasks = view.suite?.train || [];
      const table = new Map();
      for (const round of view.rounds || []) {
        for (const entry of round.episodes || []) {
          if (!isNum(entry.score) || !SCORED.includes(entry.status)) continue;
          if (!table.has(entry.harness)) table.set(entry.harness, new Map());
          const byTask = table.get(entry.harness);
          if (!byTask.has(entry.task)) byTask.set(entry.task, []);
          byTask.get(entry.task).push(entry.score);
        }
      }
      if (!table.size) return null;
      const refs = [...new Set(Object.values(view.task_references || {}).flatMap((item) => Object.keys(item)))].filter((name) => name !== "naive");
      const averagedRow = (key, label) => {
        const byTask = view.ensembles?.[key];
        if (!byTask || !Object.keys(byTask).length) return null;
        const values = tasks.map((task) => byTask[task]?.score);
        return h("tr", { key: `avg-${key}`, className: "eg-faint" }, h("td", null, label),
          tasks.map((task) => {
            const item = byTask[task];
            return h("td", { key: task, className: "num", title: item ? `${item.n} 个回合的预报逐时刻平均；这些回合各自的平均 ${signed(item.mean, 3)}${item.left_out ? `；另有 ${item.left_out} 个回合缺预报，未计入` : ""}` : "" },
              item ? h(Fragment, null, signed(item.score, 3), ` (${item.n})`) : "–");
          }),
          h("td", { className: "num" }, values.every(isNum) ? signed(mean(values), 3) : "–"));
      };
      return h(Card, { title: "训练任务上的隐藏期成绩", hint: "每个版本在每个训练任务上各回合的隐藏期总技能（相对朴素基线）均值，括号里是回合数。“平均各回合的预报”把几个回合的隐藏期预报逐时刻平均后计分（集成），只作参考，采纳仍按单个回合比较。参考行是宿主内置方法在同一批起点上的成绩。" },
        h("div", { className: "eg-table-wrap" }, h("table", { className: "eg-table" },
          h("thead", null, h("tr", null, h("th", null, "版本"), tasks.map((task) => h("th", { key: task, className: "num", title: task }, taskShort(catalog, task))), h("th", { className: "num" }, "平均"))),
          h("tbody", null,
            [...table.entries()].map(([digest, byTask]) => {
              const means = [];
              return h(Fragment, { key: digest },
                h("tr", null,
                  h("td", null, names.get(digest) || "", " ", h(Mono, null, short(digest)), digest === view.incumbent ? h(Fragment, null, " ", h(Tag, { tone: "success" }, "在任")) : null),
                  tasks.map((task) => {
                    const values = byTask.get(task) || [];
                    if (!values.length) return h("td", { key: task, className: "num eg-faint" }, "–");
                    const value = mean(values);
                    means.push(value);
                    return h("td", { key: task, className: "num", style: { background: mix(value, 0.3) }, title: values.map((item) => signed(item)).join("、") },
                      signed(value, 3), h("span", { className: "eg-faint" }, ` (${values.length})`));
                  }),
                  h("td", { className: "num" }, h("b", null, signed(mean(means), 3)))),
                averagedRow(digest, "　平均各回合的预报"));
            }),
            averagedRow("all", "全部回合的预报平均"),
            refs.map((name) => {
              const values = tasks.map((task) => view.task_references?.[task]?.[name]);
              return h("tr", { key: `ref-${name}`, className: "eg-faint" }, h("td", null, `参考：宿主内置 ${name}`),
                values.map((value, index) => h("td", { key: tasks[index], className: "num" }, signed(value, 3))),
                h("td", { className: "num" }, signed(mean(values.filter(isNum)), 3)));
            })))));
    }

    function niceTicks(lo, hi, count = 5) {
      const raw = (hi - lo) / count;
      const magnitude = 10 ** Math.floor(Math.log10(raw));
      const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) || raw;
      const ticks = [];
      for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-12; t += step) ticks.push(Number(t.toFixed(10)));
      return { ticks, digits: Math.max(0, -Math.floor(Math.log10(step)) + (step / magnitude === 2.5 ? 1 : 0)) };
    }

    function GainChart({ view }) {
      const rows = [];
      for (const round of view.rounds || []) {
        for (const candidate of round.candidates || []) {
          if (candidate.decision && isNum(candidate.decision.delta)) rows.push({ round, candidate, decision: candidate.decision });
        }
      }
      if (!rows.length) return null;
      const gate = view.gates?.min_delta;
      const values = [0, ...(isNum(gate) ? [gate] : [])];
      for (const { decision } of rows) values.push(decision.delta, ...decision.interval.filter(isNum));
      let lo = Math.min(...values);
      let hi = Math.max(...values);
      const span = Math.max(hi - lo, 0.004);
      lo -= span * 0.08;
      hi += span * 0.08;
      const x = (value) => ((value - lo) / (hi - lo)) * 100;
      const { ticks, digits } = niceTicks(lo, hi);
      const tasks = (view.suite?.train || []).length;
      const bases = new Set(rows.map(({ decision }) => decision.basis).filter(Boolean)); // as each comparison ran
      const tInterval = bases.size ? bases.has("tasks_t") : tasks >= 3;
      const basis = bases.size > 1 ? "各任务增益的 t 区间（可比的训练任务少于 3 个的那几代用分层自助区间）"
        : tInterval ? "各任务增益的 t 区间" : "分层自助区间（依次重抽任务、回合和天；任务少于 3 个时只反映任务内的随机差异）";
      return h(Card, { title: "各代候选的增益", hint: `候选 − 在任：训练任务上的平均总技能增益，与${basis}，多个候选时区间按 Bonferroni 加宽。采纳门槛：增益大于 ${signed(gate, 3)}、区间下限大于 0（单侧检验）、没有任务可信变差超过 ${signed(view.gates?.task_cap, 3)}。` },
        h("div", { className: "eg-forest" },
          rows.flatMap(({ round, candidate, decision }) => {
            const [low, high] = decision.interval;
            const adopted = candidate.status === "adopted";
            return [
              h("div", { key: `l${round.generation}${candidate.name}` }, `第 ${round.generation} 代 ${candidate.name}`),
              h("div", { key: `t${round.generation}${candidate.name}`, className: "eg-track", "data-adopted": String(adopted) },
                h("i", { className: "zero", style: { left: `${x(0)}%` } }),
                isNum(gate) ? h("i", { className: "gate", style: { left: `${x(gate)}%` } }) : null,
                h("i", { className: "ci", style: { left: `${x(low)}%`, width: `${Math.max(0.5, x(high) - x(low))}%` } }),
                h("i", { className: "dot", style: { left: `${x(decision.delta)}%` }, title: `增益 ${signed(decision.delta)}` })),
              h("div", { key: `v${round.generation}${candidate.name}` }, h("b", { className: toneClass(decision.delta) }, signed(decision.delta)),
                ` [${signed(low)}, ${signed(high)}]`, adopted ? h("span", { className: "eg-good" }, " · 采纳") : null),
            ];
          }),
          h("div", { key: "axis-l" }),
          h("div", { key: "axis", className: "eg-axis" }, ticks.map((tick) => h("span", { key: tick, style: { left: `${x(tick)}%` } }, signed(tick, digits)))),
          h("div", { key: "axis-r" })));
    }

    function VersionsTable({ body, field, catalog }) {
      const versions = body.versions || {};
      const labels = ["seed", "final"].filter((label) => versions[label]);
      const tasks = [...new Set(labels.flatMap((label) => Object.keys(versions[label][field] || {})))].sort();
      return h("div", { className: "eg-table-wrap" }, h("table", { className: "eg-table" },
        h("thead", null, h("tr", null, h("th", null, "任务"), labels.map((label) => h("th", { key: label, className: "num" }, label === "seed" ? "起点" : "最终")),
          labels.length > 1 ? h("th", { className: "num" }, "最终 − 起点") : null, h("th", null, "回合"))),
        h("tbody", null, tasks.map((task) => {
          const entries = labels.map((label) => versions[label][field]?.[task] || {});
          const scores = entries.map((entry) => entry.score);
          const change = labels.length > 1 && isNum(scores[0]) && isNum(scores[1]) ? scores[1] - scores[0] : null;
          return h("tr", { key: task }, h("td", null, h(TaskLabel, { catalog, id: task })),
            scores.map((value, index) => h("td", { key: labels[index], className: `num ${toneClass(value)}` }, signed(value, 3))),
            labels.length > 1 ? h("td", { className: `num ${toneClass(change)}` }, signed(change, 3)) : null,
            h("td", { className: "eg-faint" }, entries.map((entry) => entry.episode ? `${short(entry.episode)}${entry.reused ? "（复用）" : ""}` : "").filter(Boolean).join("、")));
        }))));
    }

    function Verification({ view, catalog, ask, act }) {
      const finished = view.status === "finished";
      const sealed = view.verification?.sealed || {};
      const count = view.suite?.holdout_tasks ?? 0;
      const busy = view.verifying;
      const holdout = view.verification?.holdout;
      const unfinished = holdout ? (holdout.tasks || []).filter((task) => Object.values(holdout.versions || {})
        .some((version) => !SCORED.includes(version.episodes?.[task]?.status))) : [];
      const run = async (kind, stage) => {
        const ok = await ask(kind === "holdout" && unfinished.length
          ? { title: `补跑 ${unfinished.length} 个留出任务？`, text: "只重跑上次没有成绩的回合（服务故障或被中断）；已有成绩的回合直接复用。", confirm: "开始补跑" }
          : kind === "holdout"
            ? { title: `在 ${count} 个留出任务上验证？`, text: "起点版本和最终版本在每个留出任务上各运行一个回合（按这次运行的回合预算）。每个版本在每个留出任务上只运行一次，跨运行有效。", confirm: "开始验证" }
            : { title: `在封存时期 ${stage} 上评测？`, text: "把两个版本提交过的预测程序在更晚的封存时期上重新拟合、逐时刻预报。每个版本每个任务只评一次，已评完的任务立刻写入账本；中途中断后再评只补没有结果的任务。", confirm: "开始评测" });
        if (ok) act("verify", { kind, stage }, "验证已开始，结果会显示在这里");
      };
      const rows = [
        { key: "holdout", label: `留出任务（${count} 个）`, body: view.verification?.holdout, field: "episodes",
          hint: "起点和最终版本在进化中没见过的任务上各跑一个回合",
          button: h(Button, { variant: "primary", disabled: !finished || busy || !count, onClick: () => run("holdout") }, "运行留出验证") },
        { key: "validation", label: "封存时期 validation", body: sealed.validation, field: "tasks",
          hint: "在更晚的封存时期上重判两个版本提交过的程序",
          button: h(Button, { variant: "primary", disabled: !finished || busy, onClick: () => run("sealed", "validation") }, "评测 validation") },
        { key: "final_test", label: "封存时期 final_test", body: sealed.final_test, field: "tasks",
          hint: sealed.validation ? "最后一道检验，每个版本每个任务只评一次" : "须在 validation 之后",
          button: h(Button, { variant: "primary", disabled: !finished || busy || !sealed.validation, onClick: () => run("sealed", "final_test") }, "评测 final_test") },
      ];
      return h(Card, { title: "独立验证", hint: finished ? "最终版本相对起点版本的提升，还要在进化中没见过的任务和更晚的时期上各验证一次。区间的算法和采纳门槛相同（3 个及以上任务时是各任务增益的 t 区间）；每个版本在每个任务上只有一个结果，所以不做单任务否决。" : "运行完成后可以验证。",
        actions: busy ? h(Tag, { tone: "info" }, "验证进行中") : null },
      h("div", { className: "eg-table-wrap" }, h("table", { className: "eg-table" },
        h("thead", null, h("tr", null, h("th", null, "检验"), h("th", { className: "num" }, "起点"), h("th", { className: "num" }, "最终"),
          h("th", { className: "num" }, "最终 − 起点"), h("th", null, "95% 区间 → 结论"))),
        h("tbody", null, rows.map((row) => {
          if (!row.body) {
            return h("tr", { key: row.key }, h("td", null, row.label), h("td", { colSpan: 3, className: "eg-faint" }, row.hint), h("td", null, row.button));
          }
          const decision = row.body.decision;
          const means = row.body.means || {};
          const change = isNum(decision?.delta) ? decision.delta : isNum(means.seed) && isNum(means.final) ? means.final - means.seed : null;
          return h("tr", { key: row.key },
            h("td", null, row.label, h("div", { className: "eg-faint" }, when(row.body.at))),
            h("td", { className: "num" }, signed(means.seed, 3)),
            h("td", { className: "num" }, signed(means.final, 3)),
            h("td", { className: `num ${toneClass(change)}` }, h("b", null, signed(change, 3))),
            h("td", null, decision ? h(Fragment, null, `[${signed(decision.interval[0], 3)}, ${signed(decision.interval[1], 3)}] → `,
              h("span", { className: decision.reason === "improved" ? "eg-good" : "" }, reasonText(catalog, decision.reason)))
              : h("span", { className: "eg-faint" }, Object.keys(row.body.versions || {}).length < 2 ? "只有一个版本（没有采纳过候选），没有可比较的对象" : "没有可比较的任务"),
              row.key === "holdout" && unfinished.length ? h("div", { style: { marginTop: "6px" } },
                h("span", { className: "eg-bad" }, `${unfinished.length} 个任务没有成绩 `),
                h(Button, { disabled: busy, onClick: () => run("holdout") }, "补跑")) : null));
        })))),
      rows.filter((row) => row.body).map((row) => h(Details, { key: row.key, summary: `${row.label}：逐任务成绩` },
        h(VersionsTable, { body: row.body, field: row.field, catalog }))));
    }

    function CandidateCard({ view, round, candidate, catalog }) {
      const decision = candidate.decision;
      const changes = candidate.changes || [];
      return h("div", { className: "eg-cand", "data-adopted": String(candidate.status === "adopted") },
        h("div", { className: "eg-cand-head" }, h("b", null, `候选 ${candidate.name}`), h(StatusTag, { map: CAND_STATUS, keyName: candidate.status }), h(Mono, { title: candidate.digest }, short(candidate.digest))),
        candidate.summary ? h("div", { className: "eg-cand-summary" }, candidate.summary) : null,
        decision && isNum(decision.delta) ? h(Fragment, null,
          h("div", { className: "eg-line" }, "增益 ", h("b", { className: toneClass(decision.delta) }, signed(decision.delta)),
            `，${((decision.confidence || 0.95) * 100).toFixed(1)}% 区间 [${signed(decision.interval[0])}, ${signed(decision.interval[1])}] → ${reasonText(catalog, decision.reason)}`),
          h("div", { className: "eg-chips" }, (decision.tasks || []).map((item) => h("span", { key: item.task, className: "eg-chip",
            title: `区间 [${signed(item.interval?.[0])}, ${signed(item.interval?.[1])}]，回合数 候选 ${item.replicates?.[0] ?? "–"} / 在任 ${item.replicates?.[1] ?? "–"}，${reasonText(catalog, item.reason)}` },
            taskShort(catalog, item.task), " ", h("span", { className: toneClass(item.delta) }, signed(item.delta, 3)),
            item.reason === "credible_regression" ? h("span", { className: "eg-bad" }, " · 否决") : null))))
          : decision ? h("div", { className: "eg-line" }, reasonText(catalog, decision.reason)) : null,
        (candidate.problems || []).length ? h(Alert, { tone: "warning" }, candidate.problems.map((item, index) => h("div", { key: index }, item))) : null,
        changes.length ? h("div", { className: "eg-chips" }, changes.map((item) => h("span", { key: item.path, className: "eg-chip" }, item.path, " ",
          h("span", { className: "eg-good" }, `+${item.added ?? 0}`), h("span", { className: "eg-bad" }, `${MINUS}${item.removed ?? 0}`)))) : null,
        candidate.proposal || changes.length ? h(Details, { summary: `提案与改动（${changes.length} 个文件）` },
          h(Markdown, { text: candidate.proposal }),
          candidate.smoke ? h("p", { className: "eg-hint" }, `模板试运行（本地检验期）：${candidate.smoke.status}，总分 ${signed(candidate.smoke.score)}，覆盖率 ${Math.round((candidate.smoke.coverage || 0) * 100)}%`) : null,
          changes.map((item) => h(Diff, { key: item.path, patch: item.patch || `${item.status} ${item.path}` }))) : null);
    }

    function Rounds({ view, catalog, go }) {
      const rounds = view.rounds || [];
      if (!rounds.length) return null;
      return h(Card, { title: "各代详情", hint: "评审意见、候选的提案与改动、比较结果和每个回合。" },
        rounds.map((round, index) => {
          let outcome;
          if (!round.finished) outcome = view.active ? h(Tag, { tone: "info" }, "进行中") : h(Tag, { tone: "warning" }, "未完成");
          else if (round.generation === 0) outcome = h(Tag, null, "基线");
          else if (round.adopted) outcome = h(Tag, { tone: "success" }, `采纳 ${round.adopted}`);
          else if (round.outcome === "no_candidates") outcome = h(Tag, { tone: "warning" }, "没有可运行的候选");
          else outcome = h(Tag, null, "保留在任");
          const scores = (round.episodes || []).filter((item) => isNum(item.score)).map((item) => item.score);
          const evolver = round.evolver;
          return h("details", { key: `${round.generation}-${round.started_at}`, className: "eg-round", open: index === rounds.length - 1 },
            h("summary", null, h("span", { className: "eg-round-no" }, `第 ${round.generation} 代`), outcome,
              h("span", { className: "eg-round-sum" }, `在任 ${short(round.incumbent)} · ${(round.candidates || []).length} 个候选 · ${(round.episodes || []).length} 个回合${scores.length ? ` · 平均 ${signed(mean(scores), 3)}` : ""}`)),
            h("div", { className: "eg-round-body" },
              round.generation === 0 ? h("p", { className: "eg-hint" }, "基线：起点版本在每个训练任务上运行，作为第 1 代比较的在任成绩。")
                : !evolver ? h("p", { className: "eg-hint" }, view.active ? "评审智能体正在阅读证据、编写候选…" : "这一代没有评审记录。")
                  : evolver.status === "error" ? h(Alert, { tone: "danger" }, h("b", null, "评审会话出错："), evolver.error || "")
                  : evolver.status === "infra_failed" ? h(Alert, { tone: "danger" }, h("b", null, "评审会话没有得到模型回复："), evolver.upstream_error || "服务故障")
                    : h("p", { className: "eg-hint" }, `评审会话：${evolver.status}，${duration(evolver.seconds)}，${tokens(evolver.tokens)} token，${evolver.llm_calls ?? "–"} 次模型调用`),
              (round.notes || []).length ? h("div", null, round.notes.map((note, i) => h("div", { key: i, className: "eg-note" }, h("span", { className: "eg-chip" }, note.kind), " ", note.text))) : null,
              evolver?.final_text ? h(Details, { summary: "评审的总结" }, h(Markdown, { text: evolver.final_text })) : null,
              (round.candidates || []).map((candidate) => h(CandidateCard, { key: candidate.name, view, round, candidate, catalog })),
              (round.episodes || []).length ? h("div", { className: "eg-table-wrap", style: { marginTop: "10px" } }, h("table", { className: "eg-table" },
                h("thead", null, h("tr", null, h("th", null, "版本"), h("th", null, "任务"), h("th", null, "状态"), h("th", { className: "num" }, "隐藏期总分"), h("th", { className: "num" }, "Token"), h("th", null, "回合"))),
                h("tbody", null, round.episodes.map((entry, i) => h("tr", { key: entry.episode || i, className: entry.episode ? "eg-click" : "", onClick: entry.episode ? () => go({ page: "episode", id: entry.episode }) : undefined },
                  h("td", null, entry.role === "incumbent" ? "在任" : `候选 ${entry.candidate}`),
                  h("td", null, h(TaskLabel, { catalog, id: entry.task })),
                  h("td", null, h(StatusTag, { map: EP_STATUS, keyName: entry.status })),
                  h("td", { className: `num ${toneClass(entry.score)}` }, signed(entry.score)),
                  h("td", { className: "num" }, tokens(entry.tokens)),
                  h("td", null, entry.episode ? h(Mono, null, entry.episode) : entry.error || "")))))) : null));
        }));
    }

    function HarnessCard({ view, names }) {
      const [open, setOpen] = useState(false);
      const [tab, setTab] = useState("diff");
      const changed = view.incumbent && view.incumbent !== view.seed?.digest;
      const { data: harness, error } = useResource(open ? Api.harness(view.incumbent) : null);
      const { data: diff } = useResource(open && changed ? Api.harnessDiff(view.seed.digest, view.incumbent) : null);
      const tabs = [["diff", "与起点的差异"], ["instructions", "指令"], ["lessons", "经验"], ["agent", "DSH 设置"], ["files", "文件"]];
      const texts = harness?.texts || {};
      let body = null;
      if (error) body = h(Alert, { tone: "warning" }, `读不到这个版本：${error.message}`);
      else if (!harness) body = h("p", { className: "eg-hint" }, "正在读取…");
      else if (tab === "diff") body = !changed ? h("p", { className: "eg-hint" }, "在任版本就是起点版本。") : !diff ? h("p", { className: "eg-hint" }, "正在读取…")
        : diff.length ? diff.map((item) => h(Diff, { key: item.path, patch: item.patch || `${item.status} ${item.path}` })) : h("p", { className: "eg-hint" }, "没有差异。");
      else if (tab === "instructions") body = h(Markdown, { text: texts["instructions.md"] || "（没有 instructions.md）" });
      else if (tab === "lessons") body = h(Markdown, { text: texts["memory/lessons.md"] || "（还没有经验记录）" });
      else if (tab === "agent") body = h("pre", { className: "eg-pre" }, texts["harness.toml"] || JSON.stringify(harness.agent || {}, null, 2));
      else body = h("div", { className: "eg-chips" }, (harness.files || []).map((file) => h("span", { key: file, className: "eg-chip eg-mono" }, file)));
      return h(Card, { title: "在任 Harness", hint: `${names.get(view.incumbent) || ""} ${short(view.incumbent)}：智能体的指令、技能、代码库模板、经验和 DSH 设置。`,
        actions: h(Button, { onClick: () => setOpen(!open) }, open ? "收起" : "查看内容") },
      open ? h(Fragment, null,
        h("div", { className: "eg-tabs", style: { margin: "0 0 10px" } }, tabs.map(([id, label]) => h("button", {
          key: id, className: "eg-tab", "aria-current": tab === id ? "page" : undefined, onClick: () => setTab(id) }, label))),
        body) : null);
    }

    function AdviceCard({ view, catalog }) {
      const datasets = [...new Set((view.suite?.train || []).map((task) => taskInfo(catalog, task)?.dataset).filter(Boolean))];
      const [scope, setScope] = useState("*");
      const [text, setText] = useState("");
      const [busy, setBusy] = useState(false);
      const save = async () => {
        if (!text.trim()) return toast("请先写下建议", "danger");
        setBusy(true);
        try {
          await api(Api.expertNote, { body: { scope, text: text.trim() } });
          setText("");
          toast("已保存；之后的回合和下一代评审会读到它");
        } catch (failure) {
          toast(`保存失败：${failure.message}`, "danger");
        }
        setBusy(false);
      };
      return h(Card, { title: "专家建议", hint: "给方向性的提示，不必实时：写入账本后，之后开始的回合会在 data/expert.md 里读到，下一代的评审智能体也会看到。" },
        h("div", { className: "eg-form", style: { gridTemplateColumns: "220px 1fr" } },
          h("label", { className: "eg-field" }, h("span", null, "适用范围"),
            h("select", { className: "eg-input", value: scope, onChange: (event) => setScope(event.target.value) },
              h("option", { value: "*" }, "全部任务"), datasets.map((item) => h("option", { key: item, value: item }, item)))),
          h("label", { className: "eg-field" }, h("span", null, "建议"),
            h("textarea", { className: "eg-input", rows: 2, value: text, placeholder: "例如：夜间通风关闭时，CO₂ 主要由补光下的光合吸收决定", onChange: (event) => setText(event.target.value) }))),
        h("div", { className: "eg-actions", style: { marginTop: "10px" } }, h(Button, { variant: "primary", disabled: busy, onClick: save }, "保存建议")));
    }

    function ConfigCard({ view, catalog }) {
      const config = view.config || {};
      const gates = view.gates || {};
      return h(Card, { title: "运行配置" }, h(Details, { summary: "展开" }, h(KV, { pairs: [
        ["代数 × 每代候选", `${config.generations} × ${config.candidates}`],
        ["每个版本每个任务的回合", String(config.repeats ?? "–")],
        ["回合预算", `${config.minutes} 分钟，${Number(config.tokens || 0).toLocaleString()} 计费 token`],
        ["内循环模型", config.model || "由 Harness 决定"],
        ["评审智能体", `${config.evolver_model}，${config.evolver_minutes} 分钟，${Number(config.evolver_tokens || 0).toLocaleString()} 计费 token`],
        ["并发 / 重试", `${config.concurrency} / ${config.retries}`],
        ["门槛", `平均增益 > ${gates.min_delta}，${Math.round((gates.confidence || 0) * 100)}% 区间下限 > 0，任务损失上限 ${gates.task_cap}`],
        ["训练任务", (view.suite?.train || []).map((task) => taskShort(catalog, task)).join("、")],
        ["任务组", h(Mono, null, config.suite || "")],
        ["ecogym", view.ecogym || "–"],
      ] })));
    }

    /* ---- new evolution -------------------------------------------------------------------------------------- */

    const SAVED = "ecogym.new-evolution";
    function NewEvolution({ go, catalog, catalogError }) {
      if (!catalog) return h(CatalogPending, { error: catalogError, text: "正在读取任务组…" });
      return h(NewEvolutionForm, { go, catalog });
    }

    function NewEvolutionForm({ go, catalog }) {
      const suites = catalog?.suites || [];
      const defaults = catalog?.evolve_defaults || {};
      const routes = useModelRoutes(catalog);
      const [values, setValues] = useState(() => {
        let saved = {};
        try {
          saved = JSON.parse(localStorage.getItem(SAVED) || "{}") || {};
        } catch {
          saved = {};
        }
        const start = { suite: suites.some((item) => item.id === saved.suite) ? saved.suite : suites[0]?.id || "", label: "",
          ...Object.fromEntries(Object.keys(defaults).map((key) => [key, saved[key] ?? defaults[key] ?? ""])) };
        return { ...start, model: modelRef(start.model), evolver_model: modelRef(start.evolver_model) };
      });
      const [busy, setBusy] = useState(false);
      const [problem, setProblem] = useState("");
      const set = (key) => (event) => {
        const value = event.target.value;
        setValues((old) => ({ ...old, [key]: value }));
      };
      const suite = suites.find((item) => item.id === values.suite);
      const runnable = routes.filter((item) => item.runnable).length;
      const fallback = catalog?.models?.dsh_default;
      const modelsHint = `可选 DSH 设置里的全部提供方（${catalog?.models?.settings || "~/.dsh/settings.yaml"}），共 ${runnable} 个可用。`
        + "模型调用经 ecogym 的本机网关转发：真实密钥留在本机，沙箱里只有一次性令牌，每次调用都记入轨迹。"
        + (fallback ? ` DSH 自己的默认模型是 ${fallback.provider}/${fallback.model}。` : "");
      const n = suite?.train?.length || 0;
      const g = Number(values.generations) || 0;
      const k = Number(values.candidates) || 0;
      const r = Number(values.repeats) || 0;
      const c = Math.max(1, Number(values.concurrency) || 1);
      const m = Number(values.minutes) || 0;
      const episodes = n * r * (1 + g * (k + 1));
      const hours = (Math.ceil((n * r) / c) * m + g * ((Number(values.evolver_minutes) || 0) + Math.ceil((n * r * (k + 1)) / c) * m)) / 60;
      const budget = episodes * (Number(values.tokens) || 0) + g * (Number(values.evolver_tokens) || 0);
      const field = (key, label, attrs = {}, note = "") => h("label", { className: "eg-field", key },
        h("span", null, label), h("input", { className: "eg-input", value: values[key] ?? "", onChange: set(key), ...attrs }),
        note ? h("small", null, note) : null);
      const submit = async (event) => {
        event?.preventDefault?.();
        if (busy) return;
        setProblem("");
        if (!values.suite) return setProblem("请选择任务组");
        const body = { suite: values.suite, label: values.label.trim() };
        for (const key of Object.keys(defaults)) {
          const value = String(values[key] ?? "").trim();
          if (value !== "") body[key] = typeof defaults[key] === "number" ? Number(value) : value;
        }
        localStorage.setItem(SAVED, JSON.stringify({ ...body, label: undefined }));
        setBusy(true);
        try {
          const started = await api(Api.evolutions, { body });
          toast(`已开始进化 ${started.run_id}`);
          go({ page: "evolution", id: started.run_id });
        } catch (failure) {
          setProblem(failure.message);
          setBusy(false);
        }
      };
      return h("form", { onSubmit: submit },
        h("button", { type: "button", className: "eg-back", onClick: () => go({ page: null }) }, "← 全部进化运行"),
        h("div", { className: "eg-head" }, h("div", { className: "eg-head-main" }, h("h1", null, "新建 Harness 进化"),
          h("p", null, "第 0 代先让起点 Harness 在每个训练任务上运行。之后每一代，评审智能体阅读各回合的成绩、轨迹、笔记和以往候选，提出若干候选 Harness；候选与在任版本在每个训练任务上同时运行，只有在多任务配对比较中可信地变好才会被采纳。"))),
        h(Card, { title: "任务组", hint: suite ? `训练任务 ${n} 个：${suite.train.map((task) => taskShort(catalog, task)).join("、")}；留出任务 ${suite.holdout_tasks} 个，只在验证时运行。` : "没有可用的任务组。" },
          h("div", { className: "eg-form", style: { gridTemplateColumns: "minmax(0, 560px)" } }, h("label", { className: "eg-field" }, h("span", null, "任务组"),
            h("select", { className: "eg-input", value: values.suite, onChange: set("suite") }, suites.map((item) => h("option", { key: item.id, value: item.id }, `${item.title}（${item.id}）`))))),
          suite?.notes ? h("p", { className: "eg-hint", style: { whiteSpace: "pre-wrap" } }, suite.notes) : null),
        h(Card, { title: "起点与预算" }, h("div", { className: "eg-form" },
          field("harness", "起点 Harness", { autoComplete: "off" }, "seed，或已保存版本的摘要前缀"),
          field("generations", "代数", { type: "number", min: 1, max: 20, step: 1 }),
          field("candidates", "每代候选数", { type: "number", min: 1, max: 6, step: 1 }, "多个候选时置信度按 Bonferroni 调整"),
          field("repeats", "每个版本每个任务的回合数", { type: "number", min: 1, max: 5, step: 1 }, "多跑几次能看清回合间的随机差异"),
          field("minutes", "每个回合的分钟数", { type: "number", min: 1, step: 1 }),
          field("tokens", "每个回合的计费 token", { type: "number", min: 10000, step: 10000 }, "命中缓存的输入 token 按 1/10 计"),
          field("concurrency", "同时运行的回合", { type: "number", min: 1, max: 16, step: 1 }))),
        h(Card, { title: "模型（API）", hint: modelsHint },
          h("div", { className: "eg-form", style: { gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))" } },
            h(ModelSelect, { label: "内循环智能体的模型", value: values.model, onChange: set("model"), routes,
              empty: `由 Harness 决定（起点版本用 ${catalog?.harness_model || "DeepSeek"}）`,
              note: "固定后，每个回合都用这个模型；不固定时由各版本 Harness 的 [agent] 决定（只能是 DeepSeek 模型）。" }),
            h(ModelSelect, { label: "评审智能体的模型", value: values.evolver_model, onChange: set("evolver_model"), routes,
              note: "每一代读证据、写候选的评审会话用这个模型。" })),
          catalog?.models?.error ? h(Alert, { tone: "warning" }, `读取 DSH 的模型设置失败：${catalog.models.error}`) : null),
        h(Card, { title: "评审智能体" }, h("div", { className: "eg-form" },
          field("evolver_minutes", "每代评审分钟数", { type: "number", min: 1, step: 1 }),
          field("evolver_tokens", "每代评审计费 token", { type: "number", min: 10000, step: 10000 }))),
        h(Card, null, h("div", { className: "eg-form", style: { gridTemplateColumns: "1fr" } },
          h("label", { className: "eg-field" }, h("span", null, "标签"), h("input", { className: "eg-input", maxLength: 200, value: values.label, onChange: set("label"), placeholder: "说明这次运行，例如：温室任务组，3 代 × 2 个候选" }))),
          h("p", { className: "eg-hint" }, `最多 ${episodes} 个回合和 ${g} 次评审会话，约 ${hours.toFixed(1)} 小时，计费 token 上限约 ${(budget / 1e6).toFixed(0)}M。运行在后台进行，关闭页面不影响；DSH 退出时会停在最后完成的代，之后可以继续。`)),
        problem ? h(Alert, { tone: "danger" }, problem) : null,
        h("div", { className: "eg-actions", style: { marginTop: "16px" } },
          h(Button, { variant: "primary", disabled: busy, onClick: () => submit() }, busy ? "正在启动…" : "开始进化"),
          h(Button, { onClick: () => go({ page: null }) }, "取消")));
    }

    /* ---- episodes ------------------------------------------------------------------------------------------ */

    function EpisodeList({ go, catalog }) {
      const { data, error } = useResource(Api.episodes, { interval: 8000 });
      const items = data || [];
      return h(Fragment, null,
        h("div", { className: "eg-head" }, h("div", { className: "eg-head-main" }, h("h1", null, "回合"),
          h("p", null, "一个回合是 DSH 智能体在隔离工作区里独立解一个任务的完整过程：读数据、查资料、写代码、做实验、提交预测程序，再由宿主在隐藏时期上逐时刻打分。完整轨迹都会保存。"))),
        error ? h(Alert, { tone: "danger" }, `读取失败：${error.message}`) : null,
        data && !items.length ? h(Empty, { title: "还没有回合", text: "进化运行和验证会产生回合，也可以用 ecogym episode run 单独运行。" }) : null,
        items.length ? h(Card, null, h("div", { className: "eg-table-wrap" }, h("table", { className: "eg-table" },
          h("thead", null, h("tr", null, h("th", null, "任务"), h("th", null, "来源"), h("th", null, "状态"), h("th", { className: "num" }, "隐藏期总分"),
            h("th", { className: "num" }, "本地最好"), h("th", null, "模型"), h("th", { className: "num" }, "Token"), h("th", null, "开始"))),
          h("tbody", null, items.map((item) => h("tr", { key: item.episode, className: "eg-click", onClick: () => go({ page: "episode", id: item.episode }) },
            h("td", null, h(TaskLabel, { catalog, id: item.task_id })),
            h("td", null, item.context?.role ? `${ROLE[item.context.role] || item.context.role}${item.context.candidate ? ` ${item.context.candidate}` : ""}${item.context.generation !== undefined ? ` · 第 ${item.context.generation} 代` : ""}` : "单独运行"),
            h("td", null, h(StatusTag, { map: EP_STATUS, keyName: epKey(item) })),
            h("td", { className: `num ${toneClass(item.score)}` }, signed(item.score)),
            h("td", { className: "num" }, signed(item.best_local)),
            h("td", null, item.model || "–"),
            h("td", { className: "num" }, tokens(item.tokens)),
            h("td", null, when(item.started_at)))))))) : null);
    }

    function CellMatrix({ task, cells }) {
      const byKey = new Map((cells || []).map((row) => [row.cell, row]));
      const targets = task ? task.targets : [...new Set((cells || []).map((row) => row.cell.split("@")[0]))].map((name) => ({ name }));
      const horizons = task ? task.horizons : [...new Set((cells || []).map((row) => Number(row.cell.split("@")[1])))].sort((a, b) => a - b);
      return h("div", { className: "eg-matrix", style: { gridTemplateColumns: `minmax(120px,max-content) repeat(${horizons.length}, minmax(110px,1fr))` } },
        h("div", { className: "eg-mh" }),
        horizons.map((hz) => h("div", { key: `h${hz}`, className: "eg-mh" }, horizonText(task, hz))),
        targets.flatMap((target) => [
          h("div", { key: `t${target.name}`, className: "eg-mh", style: { alignSelf: "center" } }, target.label || target.name,
            h("div", { className: "eg-faint" }, `${target.name}${target.unit ? ` · ${UNITS[target.unit] || target.unit}` : ""}`)),
          ...horizons.map((hz) => {
            const row = byKey.get(`${target.name}@${hz}`);
            if (!row) return h("div", { key: `${target.name}${hz}`, className: "eg-mc eg-faint" }, "–");
            return h("div", { key: `${target.name}${hz}`, className: "eg-mc", style: { background: mix(row.skill, 0.45) },
              title: `${row.n} 个预测 · MAE ${plain(row.mae)} · 偏差 ${signed(row.bias, 3)}` },
            h("b", { className: toneClass(row.skill) }, signed(row.skill, 3)),
            h("span", { className: "eg-faint" }, `RMSE ${plain(row.rmse)} · 基线 ${plain(row.baseline_rmse)}`));
          }),
        ]));
    }

    function EpisodePage({ id, go, catalog }) {
      const [running, setRunning] = useState(true);
      const { data: record, error } = useResource(Api.episode(id), { interval: running ? 5000 : 0 });
      useEffect(() => {
        if (record) setRunning(epKey(record) === "running");
      }, [record]);
      useTick(running && Boolean(record));
      if (error && !record) return h(Alert, { tone: "danger" }, `无法读取回合 ${id}：${error.message}`);
      if (!record) return h("p", { className: "eg-hint" }, "正在读取回合…");
      const task = taskInfo(catalog, record.task_id);
      const scored = record.scored;
      const finished = record.finished || {};
      const context = record.context || {};
      const best = Math.max(...(record.submissions || []).filter((item) => item.accepted).map((item) => item.local?.score).filter(isNum));
      const start = timeOf(record.started_at);
      const timeline = [...(record.evaluations || []).map((item) => ["检验", item]), ...(record.submissions || []).map((item) => ["提交", item])]
        .sort((a, b) => (timeOf(a[1].at) || 0) - (timeOf(b[1].at) || 0));
      const activity = finished.activity || {};
      const refs = scored?.references ? Object.entries(scored.references).filter(([name]) => name !== "naive").map(([name, value]) => `${name} ${signed(value, 3)}`).join("，") : "";
      return h(Fragment, null,
        h("button", { className: "eg-back", onClick: () => go({ page: null }) }, "← 返回"),
        h("div", { className: "eg-head" }, h("div", { className: "eg-head-main" },
          h("div", { className: "eg-title-line" }, h("h1", null, record.label || record.episode), h(StatusTag, { map: EP_STATUS, keyName: epKey(record) })),
          h("div", { className: "eg-meta" },
            h("span", null, h("b", null, "任务"), taskShort(catalog, record.task_id)),
            h("span", null, h("b", null, "Harness"), h(Mono, null, short(record.harness?.digest)), ` ${record.harness?.name || ""}`),
            h("span", null, h("b", null, "模型"), record.model || "–"),
            h("span", null, h("b", null, "开始"), when(record.started_at, true)),
            context.evolution ? h("span", null, h("b", null, "来自"), h("a", { href: "#", onClick: (event) => {
              event.preventDefault();
              go({ tab: "evolutions", page: "evolution", id: context.evolution });
            } }, context.evolution), ` ${ROLE[context.role] || context.role || ""}${context.candidate ? ` ${context.candidate}` : ""}`) : null)),
        h("div", { className: "eg-actions" }, h(Button, { onClick: () => copy(record.episode) }, "复制 ID"))),
        record.status === "infra_failed" ? h(Alert, { tone: "danger" }, h("b", null, "模型服务没有回复："),
          finished.upstream_error || "没有一次模型调用得到回复（服务故障，不计为 Harness 的成绩）") : null,
        scored?.errors?.length ? h(Alert, { tone: "warning" }, h("b", null, "程序在隐藏期上的错误："), scored.errors.slice(0, 3).join("；")) : null,
        h("div", { className: "eg-stats" },
          h(Stat, { label: "隐藏期总分", value: scored ? h("span", { className: toneClass(scored.score) }, signed(scored.score)) : "–",
            sub: scored ? `95% 区间 [${signed(scored.interval?.[0])}, ${signed(scored.interval?.[1])}] · ${scored.origins} 个起点` : "运行结束后计算" }),
          h(Stat, { label: "最好的本地成绩", value: Number.isFinite(best) ? signed(best) : "–", sub: "本地检验期，已接受的提交" }),
          h(Stat, { label: "提交 / 检验", value: `${(record.submissions || []).length} / ${(record.evaluations || []).length}`, sub: scored?.auto ? "没有被接受的提交：按当时的程序评分" : scored?.n ? `评分用第 ${scored.n} 次提交` : "" }),
          h(Stat, { label: "笔记 / 问题", value: `${(record.notes || []).length} / ${(record.questions || []).length}` }),
          h(Stat, { label: "Token", value: tokens(finished.tokens), sub: finished.llm_calls ? `${finished.llm_calls} 次模型调用` : "" }),
          h(Stat, { label: "用时", value: duration(finished.seconds), sub: finished.status ? `状态 ${finished.status}` : "进行中" })),
        scored ? h(Card, { title: "隐藏期逐格技能", hint: `宿主用隐藏时期之前的全部数据拟合提交的程序，再逐个时间步送入新观测、在每个计分起点要求预报；技能相对朴素基线，大于 0 更好。同一批起点上的参考：${refs || "–"}` },
          h(CellMatrix, { task, cells: scored.cells })) : null,
        timeline.length ? h(Card, { title: "时间线", hint: "智能体在本地检验期上的检验和提交（隐藏期成绩只在结束后计算）。" }, h("div", { className: "eg-table-wrap" }, h("table", { className: "eg-table" },
          h("thead", null, h("tr", null, h("th", { className: "num" }, "分钟"), h("th", null, "动作"), h("th", { className: "num" }, "本地总分"), h("th", null, "状态"), h("th", null, "说明"))),
          h("tbody", null, timeline.map(([action, item], index) => h("tr", { key: index },
            h("td", { className: "num" }, start !== null && timeOf(item.at) !== null ? ((timeOf(item.at) - start) / 60000).toFixed(1) : "–"),
            h("td", null, action, action === "提交" ? h(Fragment, null, " ", item.accepted ? h(Tag, { tone: "success" }, "已接受") : h(Tag, { tone: "danger" }, "未接受")) : null),
            h("td", { className: `num ${toneClass(item.local?.score)}` }, signed(item.local?.score)),
            h("td", null, item.local?.status || ""),
            h("td", null, item.note || ""))))))) : null,
        (record.notes || []).length ? h(Card, { title: "智能体的笔记", hint: "回合中记下的发现；导出训练数据时进入经验库。" },
          record.notes.map((note, index) => h("div", { key: index, className: "eg-note" }, h("span", { className: "eg-chip" }, note.kind), " ", note.text,
            (note.sources || []).length ? h("div", { className: "eg-faint" }, `来源：${note.sources.join("；")}`) : null))) : null,
        (record.questions || []).length ? h(Card, { title: "给专家的问题" }, record.questions.map((item) => h("div", { key: item.id, className: "eg-note" }, item.question))) : null,
        finished.final_text ? h(Card, { title: "智能体的最终回复" }, h(Markdown, { text: finished.final_text })) : null,
        Object.keys(activity.tool_calls || {}).length ? h(Card, { title: "工具使用" },
          h("div", { className: "eg-chips" }, Object.entries(activity.tool_calls).map(([name, count]) => h("span", { key: name, className: "eg-chip" }, `${name.replace(/^mcp__ecogym__/, "ecogym.")} ${count}`))),
          (activity.web_queries || []).length ? h("p", { className: "eg-hint" }, `网页搜索：${activity.web_queries.slice(0, 20).join("；")}`) : null,
          h("p", { className: "eg-hint" }, `子智能体 ${activity.subagents || 0} 个 · 轨迹 `, h(Mono, null, finished.trajectory?.dir || "–"))) : null);
    }

    /* ---- experts ------------------------------------------------------------------------------------------- */

    function ExpertPage({ go, catalog }) {
      const { data, error, reload } = useResource(Api.expert, { interval: 15000 });
      const [answers, setAnswers] = useState({});
      const [note, setNote] = useState({ scope: "*", text: "", by: "" });
      const editNote = (key, event) => {
        const value = event.target.value;
        setNote((old) => ({ ...old, [key]: value }));
      };
      const questions = data?.questions || [];
      const open = questions.filter((item) => !item.answers.length);
      const answered = questions.filter((item) => item.answers.length);
      const send = async (path, body, done) => {
        try {
          await api(path, { body });
          toast("已保存");
          done();
          reload();
        } catch (failure) {
          toast(`保存失败：${failure.message}`, "danger");
        }
      };
      const source = (item) => item.task_id ? `回合 ${item.run_id} · ${taskShort(catalog, item.task_id)}` : `进化运行 ${item.run_id} 的评审智能体`;
      const scopes = ["*", ...(catalog?.datasets || []), ...(catalog?.tasks || []).map((task) => task.id)];
      return h(Fragment, null,
        h("div", { className: "eg-head" }, h("div", { className: "eg-head-main" }, h("h1", null, "专家参与"),
          h("p", null, "智能体遇到需要领域判断的问题会留言，然后继续工作，不等待回答。专家的回答和主动给出的提示写入账本；之后匹配任务的回合会在 data/expert.md 里读到，评审智能体在证据里也能看到。"))),
        error ? h(Alert, { tone: "danger" }, `读取失败：${error.message}`) : null,
        h(Card, { title: `待回答（${open.length}）` }, open.length ? open.map((item) => h("div", { key: item.id, className: "eg-note" },
          h("div", { className: "eg-faint" }, `${when(item.at)} · ${source(item)}`),
          h("div", { className: "eg-q" }, item.question), item.context ? h("div", { className: "eg-hint" }, item.context) : null,
          h("textarea", { className: "eg-input", rows: 3, style: { marginTop: "8px" }, value: answers[item.id] || "", placeholder: "写下回答",
            onChange: (event) => {
              const value = event.target.value;
              setAnswers((old) => ({ ...old, [item.id]: value }));
            } }),
          h("div", { className: "eg-actions", style: { marginTop: "8px" } }, h(Button, { variant: "primary", disabled: !(answers[item.id] || "").trim(),
            onClick: () => send(Api.expertAnswer, { question: item.id, text: answers[item.id].trim() }, () => setAnswers((old) => ({ ...old, [item.id]: "" }))) }, "提交回答"),
          item.task_id ? h(Button, { onClick: () => go({ tab: "episodes", page: "episode", id: item.run_id }) }, "查看回合") : null))) : h("p", { className: "eg-hint" }, "现在没有待回答的问题。")),
        h(Card, { title: "主动提示", hint: "不必等问题：给某个数据集或任务写下领域知识，之后的回合都会读到。" },
          h("div", { className: "eg-form" },
            h("label", { className: "eg-field" }, h("span", null, "适用范围"), h("select", { className: "eg-input", value: note.scope, onChange: (event) => editNote("scope", event) },
              scopes.map((scope) => h("option", { key: scope, value: scope }, scope === "*" ? "全部任务" : scope)))),
            h("label", { className: "eg-field" }, h("span", null, "署名（可选）"), h("input", { className: "eg-input", maxLength: 200, value: note.by, onChange: (event) => editNote("by", event) }))),
          h("label", { className: "eg-field", style: { marginTop: "12px" } }, h("span", null, "提示"),
            h("textarea", { className: "eg-input", rows: 3, value: note.text, placeholder: "例如：夜间通风关闭时，CO₂ 主要由补光下的光合吸收决定", onChange: (event) => editNote("text", event) })),
          h("div", { className: "eg-actions", style: { marginTop: "10px" } }, h(Button, { variant: "primary", disabled: !note.text.trim(),
            onClick: () => send(Api.expertNote, { scope: note.scope, text: note.text.trim(), by: note.by.trim() }, () => setNote((old) => ({ ...old, text: "" }))) }, "保存提示")),
          (data?.notes || []).length ? h("div", { style: { marginTop: "12px" } }, data.notes.slice().reverse().map((item, index) => h("div", { key: index, className: "eg-note" },
            h("span", { className: "eg-chip" }, item.scope === "*" ? "全部任务" : item.scope), " ", item.text, h("span", { className: "eg-faint" }, ` · ${when(item.at)}${item.by ? ` · ${item.by}` : ""}`)))) : null),
        answered.length ? h(Card, { title: `已回答（${answered.length}）` }, answered.map((item) => h("div", { key: item.id, className: "eg-note" },
          h("div", { className: "eg-q" }, item.question),
          item.answers.map((answer, index) => h("div", { key: index }, `答：${answer.text}`, h("span", { className: "eg-faint" }, ` · ${when(answer.at)}${answer.by ? ` · ${answer.by}` : ""}`))),
          h("div", { className: "eg-faint" }, source(item))))) : null);
    }

    /* ---- training data --------------------------------------------------------------------------------------- */

    function ExportPreview({ id }) {
      const { data, error } = useResource(Api.exportDetail(id));
      if (error) return h(Alert, { tone: "warning" }, error.message);
      if (!data) return h("p", { className: "eg-hint" }, "正在读取…");
      const sft = data.sft_preview;
      const pair = data.preference_preview;
      return h(Fragment, null,
        sft ? h("div", { className: "eg-panel", style: { marginTop: "10px" } }, h("h3", null, `第一条 SFT 样本：${sft.messages} 条消息，${sft.tools} 个工具`),
          h("div", { className: "eg-hint" }, `${sft.meta?.kind === "evolver" ? "评审会话" : "回合"} ${sft.meta?.session || ""} · ${sft.meta?.task || ""} · 隐藏期 ${signed(sft.meta?.score)}`),
          sft.head.map((message, index) => h("div", { key: index, className: "eg-note" },
            h("span", { className: "eg-chip" }, message.role), message.tool_calls?.length ? h("span", { className: "eg-faint" }, ` 调用 ${message.tool_calls.join("、")}`) : null,
            message.reasoning_chars ? h("span", { className: "eg-faint" }, ` · 推理 ${message.reasoning_chars} 字`) : null,
            message.text ? h("div", { className: "eg-mono", style: { whiteSpace: "pre-wrap", marginTop: "4px" } }, message.text + (message.chars > message.text.length ? " …" : "")) : null))) : null,
        pair ? h("div", { className: "eg-panel", style: { marginTop: "10px" } }, h("h3", null, "第一个偏好对"),
          h("div", { className: "eg-hint" }, `任务 ${pair.meta?.task} · 胜方 ${pair.meta?.chosen?.session} (${signed(pair.meta?.chosen?.score)}) · 负方 ${pair.meta?.rejected?.session} (${signed(pair.meta?.rejected?.score)}) · 差 ${signed(pair.meta?.delta)}`)) : null);
    }

    function DataPage() {
      const [fast, setFast] = useState(false);
      const { data, error, reload } = useResource(Api.exports, { interval: fast ? 3000 : 20000 });
      const [options, setOptions] = useState({ min_score: "0", evolver: true, include_failed: false, margin: "0.02", cross_prompt: false });
      const [posting, setPosting] = useState(false);
      const [preview, setPreview] = useState(null);
      const items = data || [];
      useEffect(() => setFast(items.some((item) => item.running)), [items]);
      const option = (key, read) => (event) => {
        const value = read(event.target);
        setOptions((old) => ({ ...old, [key]: value }));
      };
      const start = async () => {
        setPosting(true);
        try {
          const body = { evolver: options.evolver, include_failed: options.include_failed, margin: Number(options.margin) || 0.02,
            cross_prompt: options.cross_prompt, min_score: options.min_score.trim() === "" ? null : Number(options.min_score) };
          const started = await api(Api.exports, { body });
          toast(`正在导出到 ${started.path}`);
          reload();
        } catch (failure) {
          toast(`导出失败：${failure.message}`, "danger");
        } finally {
          setPosting(false);
        }
      };
      const COUNT = [["sft", "SFT 线程"], ["preferences", "偏好对"], ["episodes", "回合"], ["notes", "笔记"], ["sessions", "会话"]];
      return h(Fragment, null,
        h("div", { className: "eg-head" }, h("div", { className: "eg-head-main" }, h("h1", null, "训练数据"),
          h("p", null, "每个回合和评审会话的完整模型调用都保存在轨迹里。导出时按对话线程重建成多轮 SFT 样本（含推理和工具调用），同一任务上隐藏期成绩可信地拉开差距的两个回合组成偏好对；笔记汇成经验库。"))),
        h(Card, { title: "导出", hint: "写到账本旁的 exports/ 目录：sft.jsonl、preferences.jsonl、episodes.jsonl、notes.jsonl 和 manifest.json。工作区路径会替换成 /workspace，凭据不会进入数据。" },
          h("div", { className: "eg-form" },
            h("label", { className: "eg-field" }, h("span", null, "SFT 的隐藏期总分须高于"), h("input", { className: "eg-input", value: options.min_score, onChange: option("min_score", (input) => input.value) }), h("small", null, "0 = 朴素基线；留空则不按分数过滤。只收有被接受提交的回合")),
            h("label", { className: "eg-field" }, h("span", null, "偏好对的最小分差"), h("input", { className: "eg-input", value: options.margin, onChange: option("margin", (input) => input.value) })),
            h("label", { className: "eg-check" }, h("input", { type: "checkbox", checked: options.evolver, onChange: option("evolver", (input) => input.checked) }), "包括评审智能体的会话"),
            h("label", { className: "eg-check" }, h("input", { type: "checkbox", checked: options.include_failed, onChange: option("include_failed", (input) => input.checked) }), "也导出不合格的会话（meta 里有标记）"),
            h("label", { className: "eg-check" }, h("input", { type: "checkbox", checked: options.cross_prompt, onChange: option("cross_prompt", (input) => input.checked) }), "也导出提示不同的偏好对（不同 Harness，附 rejected_prompt）")),
          h("div", { className: "eg-actions", style: { marginTop: "12px" } }, h(Button, { variant: "primary", disabled: posting || items.some((item) => item.running), onClick: start }, posting ? "正在开始…" : "导出训练数据"))),
        error ? h(Alert, { tone: "danger" }, `读取失败：${error.message}`) : null,
        items.map((item) => h(Card, { key: item.id, title: item.running ? `正在导出 ${item.id}…` : `导出 ${item.id}`,
          hint: item.created_at ? `${when(item.created_at, true)} · ecogym ${item.ecogym || ""}` : item.error || "",
          actions: item.running || item.error ? null : h(Fragment, null, h(Button, { onClick: () => copy(item.path) }, "复制路径"),
            h(Button, { onClick: () => setPreview(preview === item.id ? null : item.id) }, preview === item.id ? "收起预览" : "预览样本")) },
        item.error ? h(Alert, { tone: "danger" }, item.error) : null,
        item.counts ? h("div", { className: "eg-stats" }, COUNT.map(([key, label]) => h(Stat, { key, label, value: String(item.counts[key] ?? 0) }))) : null,
        item.files ? h("div", { className: "eg-chips" }, Object.entries(item.files).map(([name, size]) => h("span", { key: name, className: "eg-chip eg-mono" }, `${name} ${bytes(size)}`))) : null,
        item.skipped && Object.keys(item.skipped).length ? h("p", { className: "eg-hint" }, `跳过：${Object.entries(item.skipped).map(([reason, count]) => `${reason} ${count}`).join("；")}`) : null,
        item.path ? h("p", { className: "eg-hint" }, h(Mono, null, item.path)) : null,
        preview === item.id ? h(ExportPreview, { id: item.id }) : null)));
    }

    /* ---- tasks --------------------------------------------------------------------------------------------- */

    function TasksPage({ catalog, catalogError }) {
      if (!catalog) return h(CatalogPending, { error: catalogError });
      return h(Fragment, null,
        h("div", { className: "eg-head" }, h("div", { className: "eg-head-main" }, h("h1", null, "任务"),
          h("p", null, "任务组把同一类预测任务分成训练任务（进化时使用）和留出任务（只在验证时运行）。每个任务给定数据集、预测目标和时距；宿主保管隐藏时期和封存时期，按逐时刻预报打分。新领域的接入方法见 docs/new-domain.md。"))),
        (catalog.suites || []).map((suite) => h(Card, { key: suite.id, title: suite.title, hint: suite.id },
          h(KV, { pairs: [
            ["训练任务", suite.train.map((task) => taskShort(catalog, task)).join("、")],
            ["留出任务", `${suite.holdout_tasks} 个`],
            ["采纳门槛", `平均增益 > ${suite.gates?.min_delta}，${Math.round((suite.gates?.confidence || 0) * 100)}% 区间下限 > 0，任务损失上限 ${suite.gates?.task_cap}`],
          ] }), suite.notes ? h("p", { className: "eg-hint", style: { whiteSpace: "pre-wrap" } }, suite.notes) : null)),
        (catalog.tasks || []).map((task) => h(Card, { key: task.id, title: task.title, hint: task.id },
          h(KV, { pairs: [
            ["数据集", `${task.dataset}（步长 ${task.step}）`],
            ["预测目标", task.targets.map((target) => `${target.label || target.name}（${UNITS[target.unit] || target.unit || "–"}）`).join("、")],
            ["时距", task.horizons.map((hz) => horizonText(task, hz)).join("、")],
            ["默认 episode", (task.episodes || []).join("、") || "–"],
          ] }))),
        h("p", { className: "eg-hint", style: { marginTop: "16px" } }, `ecogym ${catalog.version} · 账本 `, h(Mono, null, catalog.ledger)));
    }

    /* ---- about: what the project does and how the parts fit ---------------------------------------------------- */

    const FLOW = [
      ["任务与数据", "任务卡给定数据集、预测目标和时距。数据按时间切成几段（见下），智能体只拿到隐藏期之前的部分。"],
      ["回合", "一个 DSH 智能体按一版 Harness 独立解一个任务：看数据、查资料、写预测程序，用本地检验挑版本，然后提交。"],
      ["裁判", "宿主在智能体看不到的隐藏期上逐时刻运行最后一次被接受的提交，算出每格相对朴素基线的技能。"],
      ["评审", "评审智能体读所有回合的成绩、代码、笔记和以往的候选，写出下一代的候选 Harness，宿主逐个检查。"],
      ["比较与采纳", "候选和在任版本在每个训练任务上各跑几个回合，做多任务配对比较；通过门槛的候选成为新的在任版本。"],
      ["独立验证", "进化结束后，起点和最终版本在留出隔间和更晚的封存时期上各验证一次。"],
    ];
    const STAGE_INFO = {
      calibration_fit: ["拟合期", "visible", "智能体可见"],
      calibration_uq: ["本地检验期", "local", "智能体可见；evaluate_local 和 submit 在这里打分"],
      model_selection: ["隐藏期", "hidden", "只有宿主能看到；比较 Harness 版本只用这一段的成绩"],
      validation: ["封存 validation", "sealed", "进化中没有人看过；验证时每个版本每个任务只评一次"],
      final_test: ["封存 final_test", "sealed", "同上，必须在 validation 之后"],
    };
    const EPISODE_FACTS = [
      ["工作区", "TASK.md（目标、评分、接口、时限、预算），AGENTS.md（Harness 的说明和经验），data/（可见数据、列说明、领域背景、专家意见），submission/（Harness 的起始代码）"],
      ["工具", "task_info（剩余时间和预算）、evaluate_local（本地检验）、submit（提交，以最后一次被接受的为准）、note（记下发现和教训）、ask_expert（给专家留言，不等回答）"],
      ["沙箱", "只能读写自己的回合目录，只能连本机的模型网关和工具服务"],
      ["模型网关", "真实密钥留在本机；按成本计 token，到预算或截止时间就拒绝调用；每次请求和回复都记进轨迹"],
      ["续跑", "智能体在剩余时间超过四分之一时就结束的，宿主开一段新会话请它继续（最多 3 段）；一段续跑既没有检验也没有提交就停"],
      ["结束后", "裁判在隐藏期上评分；模型调用、工具调用、工作区和 DSH 会话存进轨迹，之后可以导出为训练数据"],
    ];
    const PAGES = [
      ["evolutions", "进化", "运行列表；运行页有进度和用量、正在运行的回合、成绩表、增益图、独立验证、各代详情和在任 Harness"],
      ["new", "新建进化", "选任务组、起点 Harness、代数、候选数、每个版本的回合数、预算和模型"],
      ["episodes", "回合", "全部回合；单个回合的逐格技能、本地检验与提交的时间线、笔记、最终回复和工具使用"],
      ["expert", "专家", "回答智能体提给专家的问题，或主动写下领域提示；之后的回合和评审都会读到"],
      ["data", "训练数据", "把回合和评审会话导出为多轮 SFT 线程和偏好对"],
      ["tasks", "任务", "任务组的训练与留出任务、采纳门槛，各任务的数据集、目标和时距"],
    ];
    const TIERS = [
      ["浏览器：生态进化面板", "lib/client.js", "React，用 DSH 的组件和主题。页面按需轮询接口（3–30 秒一次，运行中的更频繁），目录只读一次；面板调用的全部接口列在 client.js 的 Api 表里。"],
      ["DSH 网页服务：插件宿主端", "lib/index.js", "随 DSH 启动和停止 ecogym 服务，崩溃后自动重启；同源代理 /api/ecogym/*：先过 DSH 的登录检查，再换上服务令牌转发。_status 和 _restart 由它自己回答。"],
      ["ecogym 服务（Python，只监听本机）", "service.py", "ROUTES 表把每个接口对应到一个方法。页面看到的视图都在请求时从账本折叠出来，进化、验证和导出在服务的后台线程里运行。"],
      ["进化循环、回合与裁判、导出", "evolution/、episodes/", "回合和评审会话是沙箱里的 DSH 无界面会话，经本机模型网关调用模型提供方；裁判在宿主上运行提交的程序并打分。"],
      ["存储", "$ECOGYM_HOME", "账本 ledger.sqlite3（只追加的事件，所有状态都能从它重算）、Harness 库、轨迹、训练数据导出。"],
    ];
    const TERMS = [
      ["Harness", "智能体带进每个任务的全部内容：说明、技能、起始代码库、经验和 DSH 设置。进化的就是它。"],
      ["回合", "一个智能体按一版 Harness 独立解一个任务的一次运行。"],
      ["在任 / 候选", "当前被采纳的版本 / 评审新写的、等待比较的版本。"],
      ["代", "一次评审，加上候选和在任版本在每个训练任务上的回合，再加一次比较。第 0 代只跑起点版本。"],
      ["训练任务 / 留出任务", "进化中比较版本用的任务 / 评审看不到、只在最终验证时运行的任务。"],
      ["技能", "每格 = (RMSE基线 − RMSE) / max(RMSE, RMSE基线)；总分是逐格加权平均，0 等于朴素基线。"],
      ["平均各回合的预报", "同一版本几个回合的隐藏期预报逐时刻平均后的成绩，只作参考，不进入门槛。"],
      ["账本", "只追加的事件记录；运行中断后从它恢复，面板上的每个数字都从它算出。"],
      ["SFT 线程 / 偏好对", "从回合轨迹导出的多轮训练样本 / 同一任务上成绩可信地不同的两个回合。"],
    ];

    function gateText(gates) {
      if (!gates) return "";
      return `平均增益大于 ${gates.min_delta}；区间下限大于 0（3 个及以上训练任务时是各任务增益的 t 区间，置信度 ${Math.round((gates.confidence || 0) * 100)}%，多个候选时按 Bonferroni 调整）；没有任务的损失超过 ${gates.task_cap} 且它自己的区间整体低于 0（两个版本在这个任务上都至少有 2 个回合时才判断）。`;
    }

    function Timeline({ task }) {
      const { data, error } = useResource(task ? Api.stages(task) : null);
      if (error) return h(Alert, { tone: "warning" }, `读不到时间分段：${error.message}`);
      if (!data) return h("p", { className: "eg-hint" }, "正在读取…");
      const total = data.days || 1;
      return h(Fragment, null,
        h("div", { className: "eg-timeline", role: "img", "aria-label": "时间分段" }, data.stages.map((item) => {
          const [label, kind] = STAGE_INFO[item.stage] || [item.stage, "visible"];
          return h("div", { key: item.stage, className: "eg-seg", "data-kind": kind, title: `${label}：${item.start} → ${item.end}`,
            style: { left: `${(item.from_days / total) * 100}%`, width: `${(item.days / total) * 100}%` } }, label);
        })),
        h("div", { className: "eg-table-wrap", style: { marginTop: "8px" } }, h("table", { className: "eg-table" },
          h("thead", null, h("tr", null, h("th", null, "时期"), h("th", null, "时间"), h("th", { className: "num" }, "天数"), h("th", null, "谁能看到"))),
          h("tbody", null, data.stages.map((item) => {
            const [label, kind, who] = STAGE_INFO[item.stage] || [item.stage, "visible", ""];
            return h("tr", { key: item.stage },
              h("td", null, h("span", { className: "eg-swatch", "data-kind": kind }), label),
              h("td", null, h(Mono, null, `${item.start.replace("T", " ")} → ${item.end.replace("T", " ")}`)),
              h("td", { className: "num" }, plain(item.days, 1)), h("td", null, who));
          })))),
        h("p", { className: "eg-hint" }, "相邻两段之间留出一小段间隔（embargo），预报时距不会跨进下一段。智能体的数据文件到隐藏期开始前为止；裁判评隐藏期时用它之前的全部行拟合，再逐个时间步送入新观测。"));
    }

    function AboutPage({ go, catalog, catalogError }) {
      const suites = catalog?.suites || [];
      const tasks = [...new Set(suites.flatMap((suite) => suite.train))];
      const [task, setTask] = useState(null);
      const [suiteId, setSuiteId] = useState(null);
      const { data: routes } = useResource(Api.routes);
      if (!catalog) return h(CatalogPending, { error: catalogError });
      const shown = task || tasks[0];
      const suite = suites.find((item) => item.id === suiteId) || suites[0];
      const groups = [...new Set((routes || []).map((route) => route.group))];
      return h(Fragment, null,
        h("div", { className: "eg-head" }, h("div", { className: "eg-head-main" }, h("h1", null, "介绍"),
          h("p", null, "ecogym 让大模型研究智能体在生态预测任务上自己做研究：看数据、查资料、写预测程序、检验和提交。它带进任务的工作方式（Harness）由评审智能体根据成绩和完整轨迹不断改进，只有在多任务比较中可信地变好才被采纳。过程中积累可复用的知识（技能、代码库、经验、专家意见）和完整的轨迹，轨迹可以导出为训练数据。DSH 本身不改，打分全部由宿主上的确定性代码完成，不交给大模型。"))),
        h(Card, { title: "整体流程", hint: "第 2–5 步每一代重复一次；最后一代结束后做第 6 步。" },
          h("div", { className: "eg-flow" }, FLOW.map(([title, text], index) => h("div", { key: title, className: "eg-step" },
            h("b", null, h("i", null, index + 1), title), h("p", null, text))))),
        h(Card, { title: "数据怎么切、谁能看到", hint: "每个任务的数据按时间切成五段；下面是一个训练任务的实际分段。",
          actions: tasks.length > 1 ? h("select", { className: "eg-input", style: { width: "auto" }, value: shown, onChange: (event) => setTask(event.target.value) },
            tasks.map((id) => h("option", { key: id, value: id }, taskShort(catalog, id)))) : null },
          shown ? h(Timeline, { task: shown }) : h("p", { className: "eg-hint" }, "还没有任务组。")),
        h(Card, { title: "三层评测", hint: "智能体能看到的检验和决定版本去留的检验是分开的。" },
          h("div", { className: "eg-layers" },
            h("div", { className: "eg-panel" }, h("h3", null, "1. 本地检验"), h("p", { className: "eg-hint" }, "智能体自己调用：用本地检验期之前的行拟合，在本地检验期上按裁判的规则打分，用来在回合里挑版本。")),
            h("div", { className: "eg-panel" }, h("h3", null, "2. 隐藏期"), h("p", { className: "eg-hint" }, "宿主打分，智能体看不到：最后一次被接受的提交用隐藏期之前的全部行拟合，再逐时刻预报隐藏期。比较 Harness 版本只用这个成绩。")),
            h("div", { className: "eg-panel" }, h("h3", null, "3. 独立验证"), h("p", { className: "eg-hint" }, "进化结束后各一次：起点和最终版本在留出隔间上各跑一个回合，它们的程序在封存时期上重新评分。"))),
          suite ? h("div", { style: { marginTop: "12px" } },
            h("div", { className: "eg-actions" }, h("b", null, "采纳门槛"), suites.length > 1 ? h("select", { className: "eg-input", style: { width: "auto" }, value: suite.id,
              onChange: (event) => setSuiteId(event.target.value) }, suites.map((item) => h("option", { key: item.id, value: item.id }, item.title))) : h("span", { className: "eg-faint" }, suite.title)),
            h("p", { className: "eg-hint" }, `${gateText(suite.gates)} 这个任务组有 ${suite.train.length} 个训练任务、${suite.holdout_tasks} 个留出任务。`)) : null),
        h(Card, { title: "一个回合里有什么" }, h(KV, { pairs: EPISODE_FACTS })),
        h(Card, { title: "面板各页" }, h("div", { className: "eg-list" }, PAGES.map(([tab, title, text]) => h("button", { key: tab, className: "eg-row",
          onClick: () => go(tab === "new" ? { tab: "evolutions", page: "new" } : { tab, page: null }) },
          h("span", { className: "eg-row-main" }, h("div", { className: "eg-row-title" }, title), h("div", { className: "eg-row-meta" }, text)))))),
        h(Card, { title: "前后端怎么配合", hint: "从上到下是一次请求经过的各层。关闭页面不影响运行；重启后端时，进化停在最后一个完成的代，之后可以继续。" },
          h("div", { className: "eg-stack" }, TIERS.map(([title, file, text], index) => h(Fragment, { key: title },
            index ? h("div", { className: "eg-tier-arrow", "aria-hidden": "true" }, "↓") : null,
            h("div", { className: "eg-tier" }, h("div", { className: "eg-tier-head" }, h("b", null, title), h(Mono, null, file)), h("p", null, text))))),
          routes ? h(Details, { summary: `服务的全部接口（${routes.length} 个，来自 GET /api/routes）` },
            h("div", { className: "eg-table-wrap", style: { marginTop: "8px" } }, h("table", { className: "eg-table" },
              h("thead", null, h("tr", null, h("th", null, "分组"), h("th", null, "方法"), h("th", null, "路径"), h("th", null, "作用"))),
              h("tbody", null, groups.flatMap((group) => routes.filter((route) => route.group === group).map((route) => h("tr", { key: `${route.method} ${route.path}` },
                h("td", { className: "eg-faint" }, group), h("td", null, h(Mono, null, route.method)), h("td", null, h(Mono, null, route.path)), h("td", null, route.text))))))),
            h("p", { className: "eg-hint" }, "浏览器经 DSH 访问时路径前缀是 /api/ecogym/（代理去掉 ecogym 再转发）。“方法竞技”一组面板不调用，留给直接使用服务的程序；命令行不经服务，直接运行方法竞技。")) : null),
        h(Card, { title: "术语" }, h(KV, { pairs: TERMS })));
    }

    /* ---- panel ----------------------------------------------------------------------------------------------- */

    // Top-bar pages. A page inside a tab (a run, an episode, the new-evolution form) is nav.page with nav.id.
    const TABS = [["evolutions", "进化"], ["episodes", "回合"], ["expert", "专家"], ["data", "训练数据"], ["tasks", "任务"], ["about", "介绍"]];
    let remembered = { tab: "evolutions", page: null, id: null };

    /** Keeps a page's error inside the panel (DSH would otherwise replace the whole main column). */
    class Boundary extends React.Component {
      constructor(props) {
        super(props);
        this.state = { error: null };
      }

      static getDerivedStateFromError(error) {
        return { error };
      }

      componentDidUpdate(previous) {
        if (previous.page !== this.props.page && this.state.error) this.setState({ error: null });
      }

      render() {
        if (!this.state.error) return this.props.children;
        return h(Alert, { tone: "danger" }, h("b", null, "这个页面出错了："), String(this.state.error?.message || this.state.error),
          h("div", { style: { marginTop: "8px" } }, h(Button, { onClick: () => this.setState({ error: null }) }, "重试")));
      }
    }

    function EcogymPanel() {
      const [nav, setNav] = useState(remembered);
      const status = useResource(Api.status, { interval: 4000 });
      const ready = status.data?.state === "ready";
      const { catalog, error: catalogError } = useCatalog(ready);
      const expert = useResource(Api.expert, { interval: 30000, enabled: ready });
      const openQuestions = (expert.data?.questions || []).filter((item) => !item.answers.length).length;
      const scroller = useRef(null);
      const go = useCallback((next) => {
        setNav((old) => {
          const value = { tab: next.tab || old.tab, page: next.page ?? null, id: next.id ?? null };
          remembered = value;
          return value;
        });
        if (scroller.current) scroller.current.scrollTop = 0;
      }, []);
      const restart = async () => {
        try {
          await api(Api.restart, { body: {} });
          toast("正在重新启动后端");
          status.reload();
        } catch (failure) {
          toast(`重启失败：${failure.message}`, "danger");
        }
      };
      let page;
      if (!ready) page = h(BackendCard, { status: status.data, error: status.data ? null : status.error, onRestart: restart });
      else if (nav.page === "new") page = h(NewEvolution, { go, catalog, catalogError });
      else if (nav.page === "evolution") page = h(EvolutionPage, { key: nav.id, id: nav.id, go, catalog });
      else if (nav.page === "episode") page = h(EpisodePage, { key: nav.id, id: nav.id, go, catalog });
      else if (nav.tab === "episodes") page = h(EpisodeList, { go, catalog });
      else if (nav.tab === "expert") page = h(ExpertPage, { go, catalog });
      else if (nav.tab === "data") page = h(DataPage, null);
      else if (nav.tab === "tasks") page = h(TasksPage, { catalog, catalogError });
      else if (nav.tab === "about") page = h(AboutPage, { go, catalog, catalogError });
      else page = h(EvolutionList, { go, catalog });
      const state = status.data?.state;
      const [pillText, pillDot] = ready ? ["后端运行中", "done"] : state === "starting" ? ["后端启动中", "ongoing"]
        : state === "exited" ? ["后端重启中", "warning"] : status.error ? ["插件未连接", "error"] : ["后端未运行", "error"];
      return h("div", { className: "eg" },
        h("header", { className: "eg-bar" },
          h("span", { className: "eg-brand" }, h(Leaf, { size: 18 }), "生态进化"),
          h("nav", { className: "eg-tabs", "aria-label": "生态进化" }, TABS.map(([id, label]) => h("button", {
            key: id, type: "button", className: "eg-tab", "aria-current": nav.tab === id && !nav.page ? "page" : undefined,
            onClick: () => go({ tab: id, page: null }),
          }, label, id === "expert" && openQuestions ? h("span", { className: "eg-count" }, openQuestions) : null))),
          h("span", { className: "eg-spacer" }),
          h("span", { className: "eg-pill", title: status.data?.pid ? `pid ${status.data.pid}` : "" }, h(Dot, { state: pillDot }), pillText),
          ready && nav.page !== "new" ? h(Button, { variant: "primary", onClick: () => go({ tab: "evolutions", page: "new" }) }, "新建进化") : null),
        h("main", { className: "eg-scroll", ref: scroller }, h("div", { className: "eg-page" },
          h(Boundary, { page: `${nav.tab}/${nav.page}/${nav.id}` }, page))),
        h(Toasts, null));
    }

    function PanelIcon({ size = 16 }) {
      return h(Leaf, { size });
    }

    const inject = ["slots"];
    /** A failed registration costs this panel, never DSH's boot or the slot owner that declared it. */
    function guarded(ctx, slot, register) {
      try {
        ctx.slots.inject(slot, () => {
          try {
            return register();
          } catch (error) {
            console.error(`[ecogym] could not register in ${slot}`, error);
            return () => {};
          }
        });
      } catch (error) {
        console.error(`[ecogym] could not wait for ${slot}`, error);
      }
    }
    function apply(ctx) {
      host = ctx;
      guarded(ctx, "sidebar.panellist", () => ctx.slots.register({
        name: "sidebar.panellist", id: PANEL, order: 10, label: "生态进化",
      }, PanelIcon));
      guarded(ctx, "main", () => ctx.slots.register({ name: "main", key: PANEL }, EcogymPanel));
    }

    exports.apply = apply;
    exports.inject = inject;
    return module.exports;
  },
});
