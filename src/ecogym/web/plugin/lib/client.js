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
.eg-facts{color:var(--dsw-alias-label-primary)}
.eg-flowline{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;margin:10px 0 6px;font-size:12.5px;color:var(--dsw-alias-label-secondary)}
.eg-flowline span{display:inline-flex;align-items:center;gap:6px;background:var(--eg-fill);border-radius:10px;padding:5px 10px}
.eg-flowline i{font-style:normal;flex:none;display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;background:var(--dsw-alias-state-business-primary);color:#fff;font-size:11px}
.eg-flowline em{font-style:normal;color:var(--dsw-alias-label-tertiary)}
.eg-section-title{margin:22px 0 8px;font-size:14px;font-weight:600}
.eg-sections{display:flex;gap:4px;margin:14px 0 2px;padding-bottom:8px;border-bottom:.5px solid var(--dsw-alias-border-l2);overflow-x:auto}
.eg-live{display:flex;align-items:center;gap:12px;flex-wrap:wrap;font-size:13px;margin-bottom:10px}
.eg-live .eg-progress{flex:0 0 220px;margin-top:0}
.eg-checks{display:flex;flex-direction:column;gap:8px;margin-top:14px}
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
.eg .eg-md h1,.eg .eg-md h2{font-size:15px;line-height:22px;margin:14px 0 6px}
.eg .eg-md h3,.eg .eg-md h4,.eg .eg-md h5{font-size:14px;line-height:21px;margin:12px 0 4px}
.eg .eg-md>div>:first-child{margin-top:0}
.eg-details{margin-top:8px}
.eg-subsection{margin-top:16px;padding-top:12px;border-top:.5px solid var(--dsw-alias-border-l2)}
.eg-subsection h3{margin:0 0 4px;font-size:14px;font-weight:600;line-height:21px}
.eg-subsection>.eg-hint{margin:0 0 8px}
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
.eg-flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px}
.eg-step{border-radius:12px;background:var(--eg-fill);padding:10px 12px;min-width:0}
.eg-step b{display:flex;align-items:center;gap:7px;font-size:13.5px;font-weight:600}
.eg-step i{font-style:normal;flex:none;display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;background:var(--dsw-alias-state-business-primary);color:#fff;font-size:11px}
.eg-step p{margin:5px 0 0;font-size:12.5px;color:var(--dsw-alias-label-secondary)}
.eg-timeline{position:relative;height:30px;border-radius:8px;background:var(--dsw-alias-bg-layer-2);overflow:hidden}
.eg-seg{position:absolute;top:0;bottom:0;display:flex;align-items:center;padding:0 6px;font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.eg-seg[data-kind=visible],.eg-swatch[data-kind=visible]{background:color-mix(in srgb,var(--dsw-alias-state-business-primary) 22%,transparent)}
.eg-seg[data-kind=local],.eg-swatch[data-kind=local]{background:color-mix(in srgb,var(--dsw-alias-state-business-primary) 45%,transparent)}
.eg-seg[data-kind=hidden],.eg-swatch[data-kind=hidden]{background:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 38%,transparent)}
.eg-seg[data-kind=sealed],.eg-swatch[data-kind=sealed]{background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 24%,transparent)}
.eg-swatch{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px;vertical-align:-1px}
.eg-summary{font-size:13.5px;line-height:1.75}
.eg-summary-head{font-size:16px;font-weight:600;line-height:24px;margin-bottom:8px}
.eg-summary p{margin:6px 0 0;overflow-wrap:anywhere}
.eg-summary p b{font-weight:600}
.eg-summary .eg-read{color:var(--dsw-alias-label-secondary);font-size:12.5px}
.eg-chart{overflow-x:auto}
.eg-chart>svg{display:block;width:100%;min-width:720px;height:auto}
.eg-legend.eg-chart svg{flex:none}
.eg-chart text{fill:var(--dsw-alias-label-secondary);font-size:11.5px;font-family:inherit}
.eg-chart text.strong{fill:var(--dsw-alias-label-primary);font-size:12px;font-weight:600}
.eg-chart text.note{fill:var(--dsw-alias-state-success-primary);font-size:11.5px}
.eg-chart text.ref-t{fill:var(--dsw-alias-label-tertiary)}
.eg-chart .grid{stroke:var(--dsw-alias-border-l2);stroke-width:1}
.eg-chart .zero{stroke:var(--dsw-alias-border-l3);stroke-width:1}
.eg-chart .sep{stroke:var(--dsw-alias-border-l3);stroke-dasharray:2 4}
.eg-chart .ref{stroke:var(--dsw-alias-label-tertiary);stroke-width:1.2;stroke-dasharray:2 4}
.eg-chart .trunk{stroke:var(--dsw-alias-label-primary);stroke-width:3;stroke-linecap:round;cursor:pointer}
.eg-chart .before{fill:none;stroke:var(--dsw-alias-state-success-primary);stroke-width:2;stroke-dasharray:6 5}
.eg-chart .before-dot{fill:var(--dsw-alias-state-success-primary)}
.eg-chart .adopt{fill:none;stroke:var(--dsw-alias-state-success-primary);stroke-width:2.5}
.eg-chart .cand{cursor:pointer}
.eg-chart .cand line{stroke:var(--dsw-alias-label-tertiary);stroke-width:1.6}
.eg-chart .cand circle{fill:var(--dsw-alias-bg-base);stroke:var(--dsw-alias-label-secondary);stroke-width:2}
.eg-chart .cand[data-side=above] line,.eg-chart .cand[data-side=above] circle{stroke:var(--dsw-alias-state-success-primary)}
.eg-chart .cand[data-side=below] line,.eg-chart .cand[data-side=below] circle{stroke:var(--dsw-alias-state-error-primary)}
.eg-chart .cand[data-adopted=true] circle{fill:var(--dsw-alias-state-success-primary)}
.eg-chart .cand:hover circle{stroke-width:3}
.eg-chart .cand text{fill:var(--dsw-alias-label-secondary);font-size:11px}
.eg-graph{overflow-x:auto;padding-bottom:4px}
.eg-graph svg{display:block}
.eg-graph text{fill:var(--dsw-alias-label-primary);font-size:12px;font-family:inherit}
.eg-graph text.sub{fill:var(--dsw-alias-label-secondary);font-size:11.5px}
.eg-graph text.head{fill:var(--dsw-alias-label-tertiary)}
.eg-graph .good{fill:var(--dsw-alias-state-success-primary)}
.eg-graph .bad{fill:var(--dsw-alias-state-error-primary)}
.eg-graph .node rect{fill:var(--eg-fill);stroke:var(--dsw-alias-border-l3);stroke-width:1}
.eg-graph .node[data-pick=true]{cursor:pointer}
.eg-graph .node[data-pick=true]:hover rect,.eg-graph .node:focus-visible rect{stroke:var(--dsw-alias-label-secondary)}
.eg-graph .node:focus{outline:none}
.eg-graph .node[data-kind=trunk] rect{fill:var(--dsw-alias-bg-layer-2)}
.eg-graph .node[data-kind=external] rect{fill:transparent;stroke-dasharray:3 3}
.eg-graph .node[data-tone=adopted] rect{fill:color-mix(in srgb,var(--dsw-alias-state-success-primary) 16%,transparent);stroke:var(--dsw-alias-state-success-primary);stroke-width:1.5}
.eg-graph .node[data-tone=success] rect{stroke:var(--dsw-alias-state-success-primary);stroke-width:1.5}
.eg-graph .node[data-tone=info] rect{stroke:var(--dsw-alias-state-business-primary);stroke-width:1.5}
.eg-graph .node[data-tone=danger] rect{stroke:var(--dsw-alias-state-error-primary);stroke-dasharray:4 3}
.eg-graph .node[data-tone=warning] rect{stroke:var(--dsw-alias-state-warn-primary);stroke-dasharray:4 3}
.eg-graph .edge{fill:none;stroke:var(--dsw-alias-border-l3);stroke-width:1.5}
.eg-graph .edge[data-kind=trunk]{stroke:var(--dsw-alias-label-tertiary);stroke-width:2}
.eg-graph .edge[data-kind=base]{stroke:var(--dsw-alias-state-business-primary);stroke-dasharray:5 4;opacity:.7}
.eg-graph .edge[data-kind=external]{stroke:var(--dsw-alias-state-business-primary);stroke-dasharray:5 4;opacity:.7}
.eg-graph .edge[data-kind=adopt]{stroke:var(--dsw-alias-state-success-primary);stroke-width:2.5}
.eg-legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:10px;font-size:12px;color:var(--dsw-alias-label-secondary)}
.eg-legend span{display:inline-flex;align-items:center;gap:6px}
.eg-legend svg{flex:none}
.eg-steps{margin-top:6px}
.eg-story{position:relative;padding:0 0 18px 36px}
.eg-story::before{content:"";position:absolute;left:11px;top:26px;bottom:0;width:1px;background:var(--dsw-alias-border-l3)}
.eg-story:last-child{padding-bottom:4px}
.eg-story:last-child::before{display:none}
.eg-story-no{position:absolute;left:0;top:0;width:23px;height:23px;border-radius:50%;background:var(--dsw-alias-state-business-primary);color:#fff;font-size:12px;font-weight:600;display:flex;align-items:center;justify-content:center}
.eg-story h3{margin:0;font-size:14px;font-weight:600;line-height:23px}
.eg-story-sub{margin-top:2px;font-size:12.5px;color:var(--dsw-alias-label-secondary)}
.eg-story-body{margin-top:8px}
.eg-label{font-size:12px;color:var(--dsw-alias-label-tertiary);margin:10px 0 2px}
.eg-reads{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:6px 12px;align-items:baseline;font-size:12.5px;margin-top:6px}
.eg-reads>dt{color:var(--dsw-alias-label-tertiary);white-space:nowrap}
.eg-reads>dd{margin:0;display:flex;flex-wrap:wrap;gap:4px 6px}
.eg-reads .eg-chip{margin:0}
button.eg-chip{border:0;font:inherit;font-size:12px;cursor:pointer}
button.eg-chip:hover{color:var(--dsw-alias-label-primary);outline:.5px solid var(--dsw-alias-border-l3)}
.eg-clamp[data-open=false]{max-height:var(--eg-clamp,240px);overflow:hidden;-webkit-mask-image:linear-gradient(to bottom,#000 72%,transparent);mask-image:linear-gradient(to bottom,#000 72%,transparent)}
.eg-more{border:0;background:transparent;color:var(--dsw-alias-link,var(--dsw-alias-state-business-primary));font:inherit;font-size:12.5px;cursor:pointer;padding:2px 0}
.eg-more:hover{text-decoration:underline}
.eg-cand[data-focus=true]{outline:1.5px solid var(--dsw-alias-state-business-primary);outline-offset:2px}
.eg-switch{display:inline-flex;border:.5px solid var(--dsw-alias-border-l4);border-radius:8px;overflow:hidden;vertical-align:middle}
.eg-switch button{border:0;background:transparent;color:var(--dsw-alias-label-secondary);font:inherit;font-size:12px;padding:2px 9px;cursor:pointer}
.eg-switch button[aria-pressed=true]{background:var(--dsw-alias-interactive-bg-active);color:var(--dsw-alias-label-primary)}
.eg-cell-chips{display:flex;flex-wrap:wrap;gap:4px}
.eg-score{height:22px;padding:0 7px;border:0;border-radius:6px;background:var(--eg-fill);color:var(--dsw-alias-label-primary);font:inherit;font-size:12px;font-variant-numeric:tabular-nums;cursor:pointer}
.eg-score:hover{outline:1px solid var(--dsw-alias-border-l3)}
.eg-verdict{margin-top:8px;font-size:13px}
.eg-act{display:grid;grid-template-columns:76px minmax(0,1fr);gap:4px 10px;padding:7px 0;border-top:.5px solid var(--dsw-alias-border-l2);font-size:13px}
.eg-act:first-child{border-top:0}
.eg-act-main{min-width:0;overflow-wrap:anywhere}
.eg-where{border:0;background:transparent;padding:0;font:inherit;font-size:12px;color:var(--dsw-alias-link,var(--dsw-alias-state-business-primary));cursor:pointer;text-align:left;white-space:nowrap;font-variant-numeric:tabular-nums}
.eg-where:hover{text-decoration:underline}
.eg-kind{display:inline-flex;align-items:center;height:19px;padding:0 6px;margin-right:6px;border-radius:5px;background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-secondary);font-size:11.5px;white-space:nowrap;vertical-align:1px}
.eg-kind[data-tool=ecogym]{color:var(--dsw-alias-state-business-primary)}
.eg-kind[data-tool=write]{color:var(--dsw-alias-state-warn-primary)}
.eg-thread-head{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin:16px 0 4px;font-size:13.5px;font-weight:600}
.eg-thread-head span{font-weight:400;font-size:12.5px;color:var(--dsw-alias-label-secondary)}
.eg-turn{display:grid;grid-template-columns:44px minmax(0,1fr);gap:0 8px;padding:8px 4px;border-top:.5px solid var(--dsw-alias-border-l2);border-radius:6px;transition:background .6s}
.eg-turn[data-flash=true]{background:color-mix(in srgb,var(--dsw-alias-state-business-primary) 12%,transparent)}
.eg-turn-no{font-size:12px;color:var(--dsw-alias-label-tertiary);font-variant-numeric:tabular-nums;padding-top:3px;white-space:nowrap}
.eg-turn-body{min-width:0}
.eg-turn-body>.eg-details:first-child{margin-top:0}
.eg-ctx{font-size:12px;color:var(--dsw-alias-label-tertiary);padding:2px 0}
.eg-prompt{border-radius:10px;background:var(--eg-fill);padding:8px 12px;margin:6px 0;font-size:12.5px}
.eg-prompt pre{margin:4px 0 0;white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;font-size:12.5px}
.eg-think>summary{display:flex;align-items:baseline;min-width:0;max-width:100%}
.eg-think>summary b{flex:none;font-weight:500;margin-right:6px;color:var(--dsw-alias-label-secondary)}
.eg-think>summary em{font-style:normal;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--dsw-alias-label-tertiary)}
.eg-call{margin-top:3px}
.eg-call>summary{display:flex;align-items:baseline;gap:6px;min-width:0;max-width:100%;color:var(--dsw-alias-label-primary)}
.eg-call>summary code{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:var(--ds-font-family-code,ui-monospace,monospace);font-size:12px}
.eg-call>summary small{flex:none;margin-left:auto;padding-left:8px;font-size:12px;color:var(--dsw-alias-label-tertiary)}
.eg-call>summary small.eg-bad{color:var(--dsw-alias-state-error-primary)}
.eg-call .eg-pre{max-height:360px}
.eg-call-result{font-size:12px;color:var(--dsw-alias-label-tertiary);margin-top:6px}
@media (max-width:900px){.eg-page{padding:20px 18px 64px}.eg-bar{padding:0 12px}}
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
      catalog: "catalog", stages: (task) => `tasks/${enc(task)}/stages`,
      evolutions: "evolutions", evolution: (id) => `evolutions/${enc(id)}`,
      evolutionAction: (id, action) => `evolutions/${enc(id)}/${action}`,
      evolverSession: (id, generation, steps = true) => `evolutions/${enc(id)}/evolver/${enc(generation)}${steps ? "" : "?steps=0"}`,
      episodes: "episodes", episode: (id) => `episodes/${enc(id)}`, episodeTranscript: (id) => `episodes/${enc(id)}/transcript`,
      roundEpisodes: (evolution, generation) => `episodes?evolution=${enc(evolution)}&generation=${enc(generation)}`,
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
        : hint(text);
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
      let help = note;
      if (ref && !listed) help = "这个模型不在 DSH 当前的设置里，开始运行时会被拒绝；请换一个。";
      else if (ref && route) help = route.runnable ? routeNote(route) : `不可用：${route.problem}`;
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
        help ? h("small", null, help) : null,
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
    /** Green above 0, red below; none for a value that shows as zero with ``digits`` decimals. */
    function toneClass(value, digits = 3) {
      if (!isNum(value) || Math.abs(value) < 0.5 * 10 ** -digits) return "";
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
    /** A cell's background: green above 0, red below, stronger up to ``scale``; none for what shows as 0.000. */
    function mix(value, scale) {
      if (!isNum(value) || Math.abs(value) < 5e-4) return undefined;
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
    /** Which version an episode ran for its run (its ``context``): 在任, 候选 a, 留出验证. */
    const roleText = (context) => `${ROLE[context?.role] || context?.role || ""}${context?.candidate ? ` ${context.candidate}` : ""}`;
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
    /** A task as a column of a Table's head: its episode with the dataset below, the task id on hover. */
    function taskHead(catalog, id, kind = "num") {
      const [card, episode] = String(id || "").split(":");
      const task = taskInfo(catalog, card);
      return [task ? h(Fragment, null, episode || task.episodes?.[0] || "", h("div", { className: "eg-faint" }, task.dataset)) : String(id || ""), kind, id];
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

    /** A component DSH exports: a function, or a memo / forwardRef object (MarkdownText is a memo). */
    const isComponent = (value) => typeof value === "function" || (typeof value === "object" && value !== null && "$$typeof" in value);
    function Button({ variant = "outline", onClick, disabled, title, children, type = "button" }) {
      if (isComponent(P.Button)) return h(P.Button, { variant, onClick, disabled, title, type }, children);
      return h("button", { type, className: "eg-btn", "data-variant": variant, onClick, disabled, title }, children);
    }
    function Tag({ tone = "neutral", children, title }) {
      if (isComponent(P.Tag)) return h(P.Tag, { tone, title }, children);
      return h("span", { className: "eg-tag", "data-tone": tone, title }, children);
    }
    function Dot({ state }) {
      if (isComponent(P.StateDot)) return h(P.StateDot, { state });
      return h("span", { className: "eg-dot", "data-state": state });
    }
    const MARKDOWN_LABELS = { code: { copyLabel: "复制", copiedLabel: "已复制" }, footnotes: "脚注" }; // one object: a new one re-parses the text
    function Markdown({ text }) {
      if (!text) return null;
      if (isComponent(P.MarkdownText)) return h("div", { className: "eg-md" }, h(P.MarkdownText, { text, labels: MARKDOWN_LABELS }));
      return h("pre", { className: "eg-pre" }, text);
    }
    function StatusTag({ map, keyName }) {
      const [text, tone] = map[keyName] || [keyName, "neutral"];
      return h(Tag, { tone }, text);
    }
    /** A page's title and what the page is for; ``children`` go below the text. */
    function PageHead({ title, text, children }) {
      return h("div", { className: "eg-head" }, h("div", { className: "eg-head-main" }, h("h1", null, title),
        text ? h("p", null, text) : null, children));
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
    /** A quiet paragraph: explanations, empty and loading states. */
    const hint = (...children) => h("p", { className: "eg-hint" }, ...children);
    function Alert({ tone = "info", children }) {
      return h("div", { className: "eg-alert", "data-tone": tone }, children);
    }
    function Empty({ title, text, action }) {
      return h("div", { className: "eg-empty" }, h(Leaf, { size: 40 }), h("h2", null, title), text ? h("p", null, text) : null, action || null);
    }
    function Details({ summary, children, open = false, className = "eg-details" }) {
      const [shown, setShown] = useState(open);
      return h("details", { className, open: shown, onToggle: (event) => setShown(event.currentTarget.open) },
        h("summary", null, summary), shown ? children : null);
    }
    /** Long content cut to a few lines, with a button to show all of it (only when it is longer). */
    function Clamp({ children, height = 240 }) {
      const box = useRef(null);
      const [open, setOpen] = useState(false);
      const [long, setLong] = useState(false);
      useEffect(() => {
        const element = box.current;
        if (element && !open) setLong(element.scrollHeight > element.clientHeight + 4);
      });
      return h(Fragment, null,
        h("div", { ref: box, className: "eg-clamp", "data-open": String(open), style: { "--eg-clamp": `${height}px` } }, children),
        long || open ? h("button", { type: "button", className: "eg-more", onClick: () => setOpen(!open) }, open ? "收起" : "展开全部") : null);
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
    /** A table in its scroll wrapper. ``head``: column titles; ``[title, "num", tip]`` right-aligns a number column. */
    function Table({ head, children, style }) {
      return h("div", { className: "eg-table-wrap", style }, h("table", { className: "eg-table" },
        h("thead", null, h("tr", null, head.map((item, index) => {
          const [title, kind, tip] = Array.isArray(item) ? item : [item];
          return h("th", { key: index, className: kind === "num" ? "num" : undefined, title: tip }, title);
        }))),
        h("tbody", null, children)));
    }
    /** A link that moves within the panel. */
    function Link({ onClick, children }) {
      return h("a", { href: "#", onClick: (event) => {
        event.preventDefault();
        onClick();
      } }, children);
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

    /** The method in five steps: [name, its line on the landing page, what it does (the 介绍 page)]. */
    const FLOW = [
      ["回合", "研究智能体按一份工作说明（Harness）独立解每个预测任务",
        "一个 DSH 智能体按一版 Harness 独立解一个任务：看数据、查资料、写预测程序，用本地检验挑版本，然后提交。"],
      ["裁判", "宿主在智能体看不到的隐藏期上打分",
        "宿主在隐藏期上逐时刻运行被评分的程序，算出每格相对朴素基线的技能。被评分的是最后一次在本地明显好过起始程序（模板）的提交，没有就是模板本身。"],
      ["评审", "评审智能体读成绩和完整过程，改出新的 Harness（候选）",
        "评审智能体读各回合的成绩、代码、笔记和会话记录，以及以往的候选和运行，从在任版本或存档里的版本出发写出候选，宿主逐个检查。"],
      ["比较与采纳", "候选在多个任务上可信地更好，才换上",
        "候选和在任版本在每个训练任务上各跑几个回合，做多任务配对比较；通过门槛的候选成为新的在任版本。"],
      ["独立验证", "最后在没见过的任务和更晚的时期上验证",
        "进化结束后，起点和最终版本在留出任务上各跑一个回合，它们的程序再在更晚的封存时期上重新评分。"],
    ];
    const CHECK_NAMES = { holdout: "留出任务", validation: "封存 validation", final_test: "封存 final_test" };

    /** A run's independent verifications (its summary's ``verified``) as small tags: each gain and whether its
     * interval clears 0, or just "已验证" when there was nothing to compare (one version); null when none was done. */
    function GainTags({ verified, style }) {
      const gains = verified?.gains || {};
      const done = Object.keys(CHECK_NAMES).filter((name) => gains[name] || (name === "holdout" ? verified?.holdout : (verified?.sealed || []).includes(name)));
      return done.length ? h("div", { className: "eg-chips", style }, done.map((name) => {
        const gain = gains[name];
        if (!isNum(gain?.delta)) return h(Tag, { key: name, title: "只有一个版本（没有采纳过候选），或者没有可比较的任务" }, `${CHECK_NAMES[name]} 已验证`);
        const side = credibility(gain.interval);
        return h(Tag, { key: name, tone: side === "above" ? "success" : side === "below" ? "danger" : "neutral",
          title: `${CHECK_NAMES[name]}：最终 − 起点 ${signed(gain.delta, 4)}，区间 [${signed(gain.interval?.[0], 3)}, ${signed(gain.interval?.[1], 3)}]，${CREDIBLE_TEXT[side]}` },
        `${CHECK_NAMES[name]} ${signed(gain.delta, 3)}`);
      })) : null;
    }

    function EvolutionList({ go }) {
      const { data, error } = useResource(Api.evolutions, { interval: 8000 });
      const runs = data || [];
      const latest = runs.find((run) => run.status === "finished" && run.adoptions);
      return h(Fragment, null,
        h(PageHead, { title: "Harness 进化", text: "让研究智能体的工作方式自己变好：每一代改出几个新版本，只有在多个任务上可信地更好才换上。" },
          h("div", { className: "eg-flowline" }, FLOW.map(([, text], index) => h(Fragment, { key: text },
            index ? h("em", null, "→") : null, h("span", null, h("i", null, index + 1), text))),
          h(Link, { onClick: () => go({ tab: "about", page: null }) }, "详细流程"))),
        latest ? h(Card, { title: "最近一次有结果的进化", hint: latest.label || latest.run_id,
          actions: h(Button, { onClick: () => go({ page: "evolution", id: latest.run_id }) }, "查看") },
        h("p", { style: { margin: 0 } }, `从 ${short(latest.seed?.digest)} 进化到 ${short(latest.incumbent)}：采纳 ${latest.adoptions} 次`,
          (latest.adoption_gains || []).length ? `（训练任务上每次 ${latest.adoption_gains.map((value) => signed(value, 3)).join("、")}）` : "", "。",
          Object.keys(latest.verified?.gains || {}).length ? "独立验证（最终 − 起点）：" : "还没有独立验证。"),
        h(GainTags, { verified: latest.verified })) : null,
        error ? h(Alert, { tone: "danger" }, `读取失败：${error.message}`) : null,
        data && !runs.length ? h(Empty, { title: "还没有进化运行", text: "选择任务组和预算，开始第一次 Harness 进化。运行在后台进行，可以随时离开这个页面。",
          action: h(Button, { variant: "primary", onClick: () => go({ page: "new" }) }, "新建进化") }) : null,
        runs.length ? h("h2", { className: "eg-section-title" }, `全部运行（${runs.length}）`) : null,
        h("div", { className: "eg-list" }, runs.map((run) => {
          const key = evoKey(run);
          const [text, tone, dot] = EVO_STATUS[key];
          return h("button", { key: run.run_id, className: "eg-row", onClick: () => go({ page: "evolution", id: run.run_id }) },
            h("span", { className: "eg-row-dot" }, h(Dot, { state: dot })),
            h("span", { className: "eg-row-main" },
              h("div", { className: "eg-row-title" }, run.label || run.run_id),
              h("div", { className: "eg-row-meta" }, [run.suite?.title || run.suite?.id, `第 ${run.rounds_finished ?? 0} / ${run.generations ?? "?"} 代`,
                `采纳 ${run.adoptions ?? 0} 次`, `${run.episodes ?? 0} 个回合`, run.tokens ? `${tokens(run.tokens)} token` : null, when(run.started_at)].filter(Boolean).join(" · ")),
              h(GainTags, { verified: run.verified, style: { marginTop: "4px" } })),
            h("span", { className: "eg-row-side" }, h(Tag, { tone }, text), run.ended_early ? h(Tag, { title: "停下后就此结束，剩下的代没有运行" }, "提前结束") : null,
              run.verifying ? h(Tag, { tone: "info" }, "验证中") : null));
        })));
    }

    /** What the service will do about a stopped run by itself (its ``resume_plan``), in words; "" when nothing. */
    function resumeText(plan) {
      if (!plan) return "";
      if (!plan.enabled) return "这个后端没有开启自动继续（ecogym serve --no-auto-resume），需要手动继续。";
      if (plan.left <= 0) return "自动继续的次数已经用完，需要手动继续。";
      const every = Math.round((plan.every_seconds || 300) / 60);
      if (plan.waiting_for?.length) return `后端在等 ${plan.waiting_for.join("、")} 恢复回复，恢复后自动继续（每 ${every} 分钟看一次）。`;
      if (plan.error) return `后端上次自动继续没有成功（${plan.error}），${every} 分钟后再试。`;
      return `后端会在 ${every} 分钟内自动继续这次运行（还能自动继续 ${plan.left} 次）。`;
    }

    function phaseOf(view) {
      const rounds = view.rounds || [];
      const round = rounds[rounds.length - 1];
      if (!round) return "准备中：读取任务数据和参考成绩";
      if (view.waiting) {
        const down = timeOf(view.waiting.since);
        return `第 ${round.generation} 代：模型服务 ${view.waiting.model} 没有回复，等它恢复${down === null ? "" : `（已等 ${duration((Date.now() - down) / 1000)}，`
          + `最多 ${view.waiting.limit_minutes ?? "?"} 分钟）`}；这段时间不开始新回合，被打断的回合恢复后重跑，不算失败`;
      }
      if (round.finished) return `第 ${round.generation} 代已完成，准备下一代`;
      const since = timeOf(round.started_at);
      const elapsed = since === null ? "" : `（这一代已进行 ${duration((Date.now() - since) / 1000)}）`;
      if (round.generation === 0) return `第 0 代基线：起点 Harness 正在各训练任务上运行${elapsed}`;
      if (!round.evolver || round.evolver.rerun) return `第 ${round.generation} 代：评审智能体正在阅读证据、编写候选 Harness${round.evolver?.rerun ? "（服务故障后重新评审）" : ""}${elapsed}`;
      if (!(round.episodes || []).length) return `第 ${round.generation} 代：候选与在任版本正在各训练任务上运行回合${elapsed}`;
      return `第 ${round.generation} 代：正在比较候选与在任版本${elapsed}`;
    }

    /** The run's numbers in one line: progress, adoptions, episodes, tokens, time. */
    function runFacts(view) {
      const rounds = view.rounds || [];
      const done = rounds.filter((round) => round.generation > 0 && round.finished).length;
      const episodes = rounds.flatMap((round) => round.episodes || []);
      const failures = episodes.filter((item) => !SCORED.includes(item.status)).length;
      const used = [...episodes, ...rounds.flatMap((round) => [...(round.earlier_evolvers || []), round.evolver].filter(Boolean))]
        .reduce((sum, item) => sum + (item.tokens || 0), 0);
      const start = timeOf(view.started_at);
      const end = view.status === "running" ? Date.now() : timeOf(view.finished_at || view.updated_at);
      return [`第 ${done} / ${view.config?.generations ?? "?"} 代`, `采纳 ${(view.adoptions || []).length} 次`,
        `${episodes.length} 个回合${failures ? `（${failures} 个故障或中断）` : ""}`, used ? `${tokens(used)} token` : null,
        `用时 ${duration(start !== null && end !== null ? (end - start) / 1000 : null)}`].filter(Boolean);
    }

    const RUN_SECTIONS = [["overview", "概览"], ["process", "每一代"], ["versions", "版本与成绩"], ["settings", "设置"]];

    function EvolutionPage({ id, go, catalog, focus: initialFocus }) {
      const [confirmEl, ask] = useConfirm();
      const [live, setLive] = useState(true);
      const [focus, setFocus] = useState(initialFocus || null);
      const [section, setSection] = useState(initialFocus ? "process" : "overview");
      const pick = useCallback((target) => {
        setFocus({ ...target, at: Date.now() });
        setSection("process");
      }, []);
      const { data: view, error, reload } = useResource(Api.evolution(id), { interval: live ? 5000 : 30000 });
      const running = view ? view.active || view.verifying || view.status === "running" : true;
      useEffect(() => setLive(Boolean(running)), [running]);
      useTick(Boolean(view?.active));
      if (error && !view) return h(Alert, { tone: "danger" }, `无法读取进化运行 ${id}：${error.message}`);
      if (!view) return hint("正在读取进化运行…");
      const key = evoKey(view);
      const [text, tone] = EVO_STATUS[key];
      const act = async (path, body, message) => {
        try {
          await api(Api.evolutionAction(view.run_id, path), { body: body || {} });
          toast(message);
          reload();
        } catch (failure) {
          toast(`操作失败：${failure.message}`, "danger");
        }
      };
      // a button's action: ask first, then post it
      const confirmed = (question, path, body, message) => async () => {
        if (await ask(question)) act(path, body, message);
      };
      const stop = confirmed({ title: "在本代结束后停止？", text: "当前这一代会跑完；已完成的代保留在账本里，之后可以继续。", confirm: "本代结束后停止" },
        "stop", {}, "将在当前这一代结束后停止");
      const abort = confirmed({ title: "立即中止？", text: "正在运行的智能体会被停止，这一代的工作作废；继续运行时从这一代重新开始。", confirm: "立即中止" },
        "abort", {}, "正在中止");
      const elsewhere = key === "elsewhere";
      const resume = confirmed({ title: "继续这次进化？", confirm: "继续运行", text: elsewhere
        ? "账本显示它仍在运行，但不是由这个后端运行的（可能在命令行里运行，也可能已经中断）。确认它已不在运行再继续，否则会重复运行。"
        : "从最后一个完成的代继续；中断的那一代会重新运行。" }, "resume", { force: elsewhere }, "已继续运行");
      const lastDone = lastAttempts(view).filter((round) => round.finished).pop();
      const left = lastDone ? (view.config?.generations ?? lastDone.generation) - lastDone.generation : 0;
      const finish = confirmed({ title: "就此结束这次进化？", confirm: "结束运行", text: lastDone ? `第 ${lastDone.generation} 代结束后的在任版本`
        + `（${versionAlias(view, lastDone.incumbent_after)} ${short(lastDone.incumbent_after)}）就是最终版本，`
        + `${left > 0 ? `剩下的 ${left} 代不再运行` : "没有剩下的代"}。结束后可以做独立验证，但不能再继续运行。` : "" }, "finish", {}, "已结束，可以验证了");
      const auto = resumeText(view.resume_plan);
      const choice = auto + (lastDone ? "可以继续运行，也可以就此结束、验证现在的在任版本。" : "可以继续运行。");
      const notice = key === "failed" ? h(Alert, { tone: "danger" }, h("b", null, "运行失败："), view.error || "没有记录原因", `。${choice}`)
        : key === "stopped" ? h(Alert, { tone: "warning" }, h("b", null, "已停止："), view.error || "按请求停止", `。已完成的代保留在账本中，${choice}`)
          : key === "interrupted" ? h(Alert, { tone: "warning" }, h("b", null, "已中断："), `运行它的进程已经退出（或很久没有新记录），${choice}`)
            : view.waiting && key === "running" ? h(Alert, { tone: "warning" }, h("b", null, "等待模型服务："), `${view.waiting.error || view.waiting.probe || "没有回复"}。`
              + `每分钟试一次，恢复后接着运行；等满 ${view.waiting.limit_minutes ?? "?"} 分钟还没恢复，运行停下${view.config?.auto_resume ? "，服务恢复后自动继续" : "，之后可以继续"}。`)
            : view.ended_early ? h(Alert, { tone: "info" }, h("b", null, "提前结束："),
              `停在第 ${view.result?.last_generation ?? "?"} 代之后（计划 ${view.result?.planned_generations ?? "?"} 代），最终版本是那时的在任版本。`)
              : view.error ? h(Alert, { tone: "warning" }, view.error) : null;
      return h(Fragment, null, confirmEl,
        h("button", { className: "eg-back", onClick: () => go({ page: null }) }, "← 全部进化运行"),
        h("div", { className: "eg-head" },
          h("div", { className: "eg-head-main" },
            h("div", { className: "eg-title-line" }, h("h1", null, view.label || view.run_id), h(Tag, { tone }, text),
              view.verifying ? h(Tag, { tone: "info" }, "验证进行中") : null),
            h("div", { className: "eg-meta" },
              h("span", null, h("b", null, "任务组"), view.suite?.title || view.suite?.id),
              h("span", null, h("b", null, "起点 → 在任"), h(Mono, null, `${short(view.seed?.digest)} → ${short(view.incumbent)}`), ` ${versionAlias(view, view.incumbent)}`),
              h("span", null, h("b", null, "开始"), when(view.started_at, true)),
              (view.resumes || []).length ? h("span", null, h("b", null, "继续过"), `${view.resumes.length} 次`
                + (view.resumes.some((item) => item.by === "auto") ? `（后端自动 ${view.resumes.filter((item) => item.by === "auto").length} 次）` : "")) : null,
              h("span", null, h("b", null, "运行"), h(Mono, null, view.run_id))),
            h("div", { className: "eg-meta eg-facts" }, runFacts(view).map((item) => h("span", { key: item }, item)))),
          h("div", { className: "eg-actions" },
            view.active ? h(Button, { onClick: stop }, "本代结束后停止") : null,
            view.active ? h(Button, { onClick: abort }, "立即中止") : null,
            !view.active && ["stopped", "failed", "interrupted", "elsewhere"].includes(key) ? h(Button, { variant: "primary", onClick: resume }, "继续运行") : null,
            !view.active && !view.verifying && lastDone && ["stopped", "failed", "interrupted"].includes(key)
              ? h(Button, { onClick: finish, title: "以现在的在任版本作为最终版本结束运行，然后就能验证" }, "结束运行") : null,
            h(Button, { onClick: () => copy(view.run_id) }, "复制 ID"))),
        notice,
        h("div", { className: "eg-sections", role: "tablist", "aria-label": "运行的各部分" }, RUN_SECTIONS.map(([value, label]) => h("button", {
          key: value, type: "button", role: "tab", className: "eg-tab", "aria-selected": String(section === value),
          "aria-current": section === value ? "page" : undefined, onClick: () => setSection(value) }, label))),
        // every section stays mounted (hidden when not chosen), so opened generations and scroll targets survive a switch
        h("div", { hidden: section !== "overview" },
          view.active ? h(LiveCard, { view, catalog, go }) : null,
          h(RunSummary, { view, go }),
          h(ScoreCard, { view, catalog, onPick: pick }),
          h(Verification, { view, catalog, ask, act })),
        h("div", { hidden: section !== "process" }, h(Process, { view, catalog, go, focus })),
        h("div", { hidden: section !== "versions" },
          h(LineageCard, { view, catalog, onPick: pick }),
          h(Scoreboard, { view, catalog }),
          h(HarnessCard, { view })),
        h("div", { hidden: section !== "settings" }, h(ConfigCard, { view, catalog, go })));
    }

    /** What runs now: the current generation's episodes (polled while the run is active), with its progress. */
    function LiveCard({ view, catalog, go }) {
      const round = (view.rounds || [])[(view.rounds || []).length - 1];
      const { data: episodes } = useResource(round ? Api.roundEpisodes(view.run_id, round.generation) : null, { interval: 8000 });
      const since = timeOf(round?.started_at) ?? 0;
      const mine = (episodes || []).filter((item) => (timeOf(item.started_at) ?? 0) >= since - 1000); // this attempt of the round
      const live = mine.filter((item) => item.status === "running");
      // one slot per version, task and replicate: an episode tried again after a provider failure counts once, at its end
      const slot = (item) => [item.context?.role, item.context?.candidate, item.task_id, item.context?.replicate].join("|");
      const busy = new Set(live.map(slot));
      const done = new Set(mine.filter((item) => item.status !== "running").map(slot).filter((key) => !busy.has(key))).size;
      const reviewing = Boolean(round?.generation) && (!round.evolver || round.evolver.rerun); // no episodes until it ends
      const runnable = (round?.candidates || []).filter((candidate) => candidate.status === "proposed" || (candidate.digest && candidate.decision)).length;
      const idle = Boolean(round?.generation) && !reviewing && !runnable; // no candidate can run: the loop runs no episodes
      const total = round && !reviewing && !idle ? (view.suite?.train || []).length * (view.config?.repeats || 1) * (1 + runnable) : 0;
      const quiet = reviewing ? "评审智能体正在读证据、写候选；它结束后回合才开始。" : idle ? "这一代没有可以运行的候选，不跑回合。" : "现在没有运行中的回合。";
      return h(Card, { title: "正在进行", hint: phaseOf(view) },
        total ? h("div", { className: "eg-live" },
          h("div", { className: "eg-progress" }, h("i", { style: { width: `${Math.min(100, (done / total) * 100)}%` } })),
          h("span", null, `这一代 ${Math.min(done, total)} / ${total} 个回合已结束${live.length ? `，${live.length} 个正在运行` : ""}`
            + `（同时最多 ${view.config?.concurrency ?? "?"} 个）`)) : null,
        live.length ? h(Table, { head: ["任务", "版本", "模型", ["已运行", "num"], "回合"] },
          live.map((item) => h("tr", { key: item.episode, className: "eg-click",
            onClick: () => go({ page: "episode", id: item.episode, back: { page: "evolution", id: view.run_id } }) },
            h("td", null, h(TaskLabel, { catalog, id: item.task_id })),
            h("td", null, roleText(item.context)),
            h("td", null, item.model || "–"),
            h("td", { className: "num" }, duration((Date.now() - (timeOf(item.started_at) || Date.now())) / 1000)),
            h("td", null, h(Mono, null, item.episode))))) : hint(quiet));
    }

    function Scoreboard({ view, catalog }) {
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
      const start = table.get(view.seed?.digest);
      const base = (task) => (start?.get(task) ? mean(start.get(task)) : null);
      const refs = [...new Set(Object.values(view.task_references || {}).flatMap((item) => Object.keys(item)))].filter((name) => name !== "naive");
      return h(Card, { title: "各版本在每个训练任务上的成绩", hint: "隐藏期总技能（相对朴素基线，大于 0 更好）的各回合均值，括号里是回合数，悬停看每个回合。"
        + "底色表示比起点版本在这个任务上高（绿）还是低（红）。对照行是宿主内置的方法在同一批起点上的成绩。" },
        h(Table, { head: ["版本", ...tasks.map((task) => taskHead(catalog, task)), ["平均", "num"]] },
          [...table.entries()].map(([digest, byTask]) => {
            const means = [];
            return h("tr", { key: digest },
              h("td", { style: { whiteSpace: "nowrap" } }, `${versionAlias(view, digest)} `, h(Mono, null, short(digest)),
                digest === view.incumbent ? h(Fragment, null, " ", h(Tag, { tone: "success" }, "在任")) : null),
              tasks.map((task) => {
                const values = byTask.get(task) || [];
                if (!values.length) return h("td", { key: task, className: "num eg-faint" }, "–");
                const value = mean(values);
                means.push(value);
                return h("td", { key: task, className: "num", style: { background: isNum(base(task)) ? mix(value - base(task), 0.03) : undefined },
                  title: values.map((item) => signed(item)).join("、") }, signed(value, 3), h("span", { className: "eg-faint" }, ` (${values.length})`));
              }),
              h("td", { className: "num" }, h("b", null, means.length === tasks.length ? signed(mean(means), 3) : "–")));
          }),
          refs.map((name) => {
            const values = tasks.map((task) => view.task_references?.[task]?.[name]);
            return h("tr", { key: `ref-${name}`, className: "eg-faint" }, h("td", { title: `宿主内置的预报方法 ${name}` }, `对照：${REFERENCE_NAMES[name] || name}`),
              values.map((value, index) => h("td", { key: tasks[index], className: "num" }, signed(value, 3))),
              h("td", { className: "num" }, signed(mean(values.filter(isNum)), 3)));
          })));
    }

    function niceTicks(lo, hi, count = 5) {
      const raw = (hi - lo) / count;
      const magnitude = 10 ** Math.floor(Math.log10(raw));
      const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) || raw;
      const ticks = [];
      for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-12; t += step) ticks.push(Number(t.toFixed(10)));
      return { ticks, digits: Math.max(0, -Math.floor(Math.log10(step)) + (step / magnitude === 2.5 ? 1 : 0)) };
    }

    function VersionsTable({ body, field, catalog }) {
      const versions = body.versions || {};
      const labels = ["seed", "final"].filter((label) => versions[label]);
      const tasks = [...new Set(labels.flatMap((label) => Object.keys(versions[label][field] || {})))].sort();
      return h(Table, { head: ["任务", ...labels.map((label) => [label === "seed" ? "起点" : "最终", "num"]), ...(labels.length > 1 ? [["最终 − 起点", "num"]] : []), "回合"] },
        tasks.map((task) => {
          const entries = labels.map((label) => versions[label][field]?.[task] || {});
          const scores = entries.map((entry) => entry.score);
          const change = labels.length > 1 && isNum(scores[0]) && isNum(scores[1]) ? scores[1] - scores[0] : null;
          return h("tr", { key: task }, h("td", null, h(TaskLabel, { catalog, id: task })),
            scores.map((value, index) => h("td", { key: labels[index], className: `num ${toneClass(value)}` }, signed(value, 3))),
            labels.length > 1 ? h("td", { className: `num ${toneClass(change)}` }, signed(change, 3)) : null,
            h("td", { className: "eg-faint" }, entries.map((entry) => entry.episode ? `${short(entry.episode)}${entry.reused ? "（复用）" : ""}` : "").filter(Boolean).join("、")));
        }));
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
            ? { title: `在 ${count} 个留出任务上验证？`, text: "起点版本和最终版本在每个留出任务上各运行一个回合（按这次运行的模型和回合预算）。同一版本在同样的模型和预算下，每个留出任务只运行一次，跨运行有效；模型或预算不同时重新运行。", confirm: "开始验证" }
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
      const live = view.active || (view.status === "running" && !view.stale);
      return h(Card, { title: "独立验证", hint: finished ? "最终版本相对起点版本的提升，还要在进化中没见过的任务和更晚的时期上各验证一次。"
        + `区间的算法和采纳门槛相同（${view.gates?.test === "signed_rank" ? "任务太少时改用 t 区间或分层自助区间" : "任务少于 3 个时改用分层自助区间"}）；`
        + "每个版本在每个任务上只有一个结果，所以不做单任务否决。"
        : live ? "运行完成后可以验证。" : "运行还没有结束：要验证现在的在任版本，先点上面的「结束运行」（结束后不能再继续）；也可以继续运行，跑完再验证。",
        actions: busy ? h(Tag, { tone: "info" }, "验证进行中") : null },
      h(Table, { head: ["检验", ["起点", "num"], ["最终", "num"], ["最终 − 起点", "num"], "95% 区间 → 结论"] },
        rows.map((row) => {
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
        })),
      rows.filter((row) => row.body).map((row) => h(Details, { key: row.key, summary: `${row.label}：逐任务成绩` },
        h(VersionsTable, { body: row.body, field: row.field, catalog }))),
      h(AdoptionChecks, { view, ask, act, busy, live }));
    }

    /** The check of one adoption on the training tasks' sealed period, in words (null when there is none). */
    function checkText(check) {
      if (!check) return null;
      if (check.error) return `复核没有完成（${String(check.error).slice(0, 160)}）；运行再继续时、或在「独立验证」里可以补做。`;
      const decision = check.decision;
      if (!decision) return "复核时没有两个版本都有程序的训练任务，无从比较。";
      const { better, total } = tasksBetter(decision);
      return `在训练任务更晚、谁都没看过的封存时期 validation 上复核：被替换的版本 ${signed(check.means?.parent, 3)}，它 ${signed(check.means?.adopted, 3)}，`
        + `差 ${signed(decision.delta, 3)}（区间 [${signed(decision.interval[0], 3)}, ${signed(decision.interval[1], 3)}]，${CREDIBLE_TEXT[credibility(decision.interval)]}），`
        + `${total} 个任务里 ${better} 个更好。`;
    }

    /** Every adoption of the run: its gain on the hidden period when it was adopted, and on the sealed period. */
    function AdoptionChecks({ view, ask, act, busy, live }) {
      const adoptions = view.adoptions || [];
      if (!adoptions.length) return null;
      // the same rule as the service (verify.unchecked): a check without an error counts, a later error does not undo it
      const checks = new Map();
      for (const item of view.adoption_checks || []) {
        if (!item.error || !checks.has(item.generation) || checks.get(item.generation).error) checks.set(item.generation, item);
      }
      const pending = adoptions.filter((item) => !checks.get(item.generation) || checks.get(item.generation).error);
      const run = async () => {
        if (await ask({ title: `复核 ${pending.length} 次采纳？`, text: "把每次采纳时比较用的程序（被采纳的候选和被它替换的版本）在训练任务的封存时期 validation 上重新拟合、逐时刻预报，"
          + "不运行智能体。每个版本每个任务在这个时期只评一次，已有的结果直接复用；final_test 留给最终验证。", confirm: "开始复核" })) {
          act("verify", { kind: "adoptions" }, "复核已开始，结果会显示在这里");
        }
      };
      return h("div", { className: "eg-subsection" },
        h("h3", null, "各次采纳的复核"),
        hint("每次采纳都是在训练任务的隐藏期上选出来的小提升，选得多了，一部分可能只是碰上了隐藏期的运气。"
          + "所以被采纳的候选和被它替换的版本提交过的程序，还要在训练任务更晚、谁都没看过的封存时期 validation 上再评一次"
          + "（每个版本每个任务只评一次，评过的直接复用）。评审智能体看不到这些结果，它们只给人判断这次进化。"),
        h(Table, { head: ["采纳", ["采纳时（隐藏期）", "num"], ["复核（validation）", "num"], "区间", ["更好的任务", "num"]] },
          adoptions.map((item) => {
            const check = checks.get(item.generation);
            const decision = check?.decision;
            const { better, total } = tasksBetter(decision);
            return h("tr", { key: item.generation },
              h("td", null, `第 ${item.generation} 代 候选 ${item.name}`, h("div", { className: "eg-faint" }, h(Mono, null, short(item.digest)))),
              h("td", { className: `num ${toneClass(item.delta)}` }, signed(item.delta, 3)),
              check && !check.error && decision
                ? h(Fragment, null,
                  h("td", { className: `num ${toneClass(decision.delta)}` }, h("b", null, signed(decision.delta, 3))),
                  h("td", null, `[${signed(decision.interval[0], 3)}, ${signed(decision.interval[1], 3)}]`),
                  h("td", { className: "num" }, `${better} / ${total}`))
                : h("td", { colSpan: 3, className: check?.error ? "eg-bad" : "eg-faint" },
                  check?.error ? `没有完成：${String(check.error).slice(0, 120)}` : check ? "没有可比较的任务" : live ? "运行会在这一代之后复核" : "还没有复核"));
          })),
        pending.length && !live ? h("div", { style: { marginTop: "8px" } },
          h(Button, { disabled: busy, onClick: run }, `复核 ${pending.length} 次采纳`)) : null);
    }

    /* ---- the evolution process: lineage graph, each generation's story ------------------------------------ */

    const CAND_TONE = { adopted: "adopted", passed: "success", proposed: "info", invalid: "danger", rejected: "danger",
      duplicate: "warning", unchanged: "warning" };
    const NOTE_KIND = { finding: "发现", lesson: "经验", dead_end: "无效方向" };

    /** The last attempt of each generation (a round cut short and run again appears twice in the record). */
    function lastAttempts(view) {
      const byGeneration = new Map();
      for (const round of view.rounds || []) byGeneration.set(round.generation, round);
      return [...byGeneration.values()].sort((a, b) => a.generation - b.generation);
    }

    /** A version's hidden-period score in one round: the mean over training tasks of its episodes' mean. */
    function versionMean(round, digest) {
      const byTask = new Map();
      for (const entry of round?.episodes || []) {
        if (entry.harness !== digest || !isNum(entry.score) || !SCORED.includes(entry.status)) continue;
        if (!byTask.has(entry.task)) byTask.set(entry.task, []);
        byTask.get(entry.task).push(entry.score);
      }
      return byTask.size ? mean([...byTask.values()].map(mean)) : null;
    }

    function tasksBetter(decision) {
      const tasks = decision?.tasks || [];
      return { better: tasks.filter((item) => isNum(item.delta) && item.delta > 0).length, total: tasks.length };
    }

    /** Where a candidate's content came from, in words. */
    function baseText(view, candidate) {
      const base = candidate.base;
      if (!base?.digest) return "从在任版本出发";
      if (base.digest === view.seed?.digest) return `从起点版本出发（存档 ${base.label || "seed"}）`;
      for (const round of view.rounds || []) {
        const found = (round.candidates || []).find((item) => item.digest === base.digest);
        if (found) return `从本次第 ${round.generation} 代的候选 ${found.name} 出发（存档 ${base.label}）`;
      }
      return `从其他运行的候选 ${base.label || short(base.digest)} 出发（存档）`;
    }

    /** Text cut to a width in px at the graph's font size (a CJK character is about twice as wide as a Latin one). */
    function fit(text, width, size = 12) {
      let used = 0;
      let out = "";
      for (const char of String(text || "")) {
        used += /[\u2e80-\uffff]/.test(char) ? size : size * 0.6;
        if (used > width) return `${out}…`;
        out += char;
      }
      return out;
    }

    const G = { node: 164, height: 46, indent: 18, column: 212, row: 60, top: 30, left: 2 };

    /**
     * The run as a graph: a column per generation, its incumbent on top (the trunk) and its candidates below; an
     * edge from where each candidate started (the incumbent, an archived version of this run, or another run's)
     * and from an adopted candidate to the next generation's incumbent.
     */
    function Lineage({ view, catalog, onPick }) {
      const rounds = lastAttempts(view);
      if (!rounds.length) return null;
      const own = new Map(); // a candidate's digest -> its node
      rounds.forEach((round, index) => (round.candidates || []).forEach((candidate) => {
        if (candidate.digest && !own.has(candidate.digest)) own.set(candidate.digest, `c${index}-${candidate.name}`);
      }));
      const externals = [];
      for (const round of rounds) {
        for (const candidate of round.candidates || []) {
          const digest = candidate.base?.digest;
          if (digest && !own.has(digest) && digest !== view.seed?.digest && !externals.some((item) => item.digest === digest)) {
            externals.push({ digest, label: candidate.base.label });
          }
        }
      }
      const shift = externals.length ? 1 : 0;
      const colX = (index) => G.left + (index + shift) * G.column;
      const rowY = (row) => G.top + row * G.row;
      const nodes = new Map();
      const edges = [];
      const add = (node) => nodes.set(node.id, node);
      const level = versionLevels(view);
      const origin = seedOrigin(view);
      rounds.forEach((round, index) => {
        const score = versionMean(round, round.incumbent);
        add({ id: `t${index}`, kind: "trunk", x: colX(index), y: rowY(0), tone: "neutral",
          title: round.generation === 0 ? `起点：${origin}` : `在任：${versionAlias(view, round.incumbent)}`,
          sub: isNum(score) ? [{ text: `这一代回合均分 ${signed(score, 3)}` }] : [{ text: short(round.incumbent).slice(0, 6) }],
          tip: `${round.generation === 0 ? "起点版本" : "这一代开始时的在任版本（当前在用的版本）"} ${short(round.incumbent)}`
            + (round.generation === 0 && (view.ancestry || [])[0] ? `：${day(view.ancestry[0].started_at)} 那次运行（${view.ancestry[0].run_id}）采纳的候选` : "")
            + (isNum(score) ? `\n这一代它的回合在各训练任务上的均分 ${signed(score, 3)}` : ""),
          pick: () => onPick({ generation: round.generation }) });
      });
      const end = rounds.length;
      const done = view.status === "finished";
      const changed = view.incumbent !== view.seed?.digest;
      const finalScore = level(view.incumbent);
      add({ id: `t${end}`, kind: "trunk", x: colX(end), y: rowY(0), tone: done && changed ? "success" : "neutral",
        title: `${done ? "最终" : "现在"}：${changed ? versionAlias(view, view.incumbent) : "仍是起点"}`,
        sub: [{ text: `${changed ? "换了新版本" : "没有换版本"}${isNum(finalScore) ? ` · ${signed(finalScore, 3)}` : ""}` }],
        tip: `${done ? "进化结束时" : "现在"}的在任版本 ${short(view.incumbent)}${isNum(finalScore) ? `\n成绩 ${signed(finalScore, 3)}：它作为在任版本的全部回合在各训练任务上的平均` : ""}` });
      for (let index = 0; index < end; index += 1) edges.push({ kind: "trunk", from: `t${index}`, to: `t${index + 1}` });
      externals.forEach((item, index) => add({ id: `x${index}`, kind: "external", x: G.left, y: rowY(index + 1), tone: "neutral",
        title: item.label || short(item.digest).slice(0, 6), sub: [{ text: "以前运行的候选（未采纳）" }],
        tip: `另一次运行（训练任务都在本次之内）里的候选 ${item.label}（${short(item.digest)}）：本次有候选从它出发` }));
      let rows = externals.length;
      rounds.forEach((round, index) => {
        (round.candidates || []).forEach((candidate, row) => {
          const decision = candidate.decision;
          const [status] = CAND_STATUS[candidate.status] || [candidate.status];
          const measured = decision && isNum(decision.delta);
          const { better, total } = tasksBetter(decision);
          const id = `c${index}-${candidate.name}`;
          const waiting = candidate.status === "proposed" ? (view.active && !round.finished ? "回合进行中" : "没有比较") : "没有运行";
          const side = measured ? credibility(decision.interval) : null;
          const gap = measured ? `${decision.delta >= 0 ? "高" : "低"} ${plain(Math.abs(decision.delta), 3)}` : "";
          add({ id, kind: "candidate", x: colX(index) + G.indent, y: rowY(row + 1), tone: CAND_TONE[candidate.status] || "neutral",
            title: `g${round.generation}${candidate.name} · ${status}`,
            sub: measured ? [{ text: `比在任${gap}`, tone: side === "above" ? "eg-good" : side === "below" ? "eg-bad" : undefined },
              { text: ` · ${CREDIBLE_TEXT[side]}` }] : [{ text: waiting }],
            tip: [candidate.summary, baseText(view, candidate), measured ? `比在任版本${gap}（${total} 个训练任务里 ${better} 个更好），误差范围 `
              + `[${signed(decision.interval?.[0], 3)}, ${signed(decision.interval?.[1], 3)}]：${reasonText(catalog, decision.reason)}` : ""].filter(Boolean).join("\n"),
            pick: () => onPick({ generation: round.generation, name: candidate.name }) });
          rows = Math.max(rows, row + 1);
          const base = candidate.base?.digest;
          if (!base) edges.push({ kind: "fork", from: `t${index}`, to: id });
          else if (own.has(base)) edges.push({ kind: "base", from: own.get(base), to: id });
          else if (base === view.seed?.digest) edges.push({ kind: "base", from: "t0", to: id });
          else edges.push({ kind: "external", from: `x${externals.findIndex((item) => item.digest === base)}`, to: id });
          if (round.finished && round.adopted === candidate.name) edges.push({ kind: "adopt", from: id, to: `t${index + 1}` });
        });
      });
      const path = (edge) => {
        const a = nodes.get(edge.from);
        const b = nodes.get(edge.to);
        if (!a || !b) return null;
        if (edge.kind === "fork") return `M${a.x + G.indent / 2},${a.y + G.height} V${b.y + G.height / 2} H${b.x}`;
        const [x1, y1, x2, y2] = [a.x + G.node, a.y + G.height / 2, b.x, b.y + G.height / 2];
        if (edge.kind === "trunk") return `M${x1},${y1} H${x2}`;
        const dx = Math.max(24, (x2 - x1) / 2);
        return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
      };
      const heads = [
        ...(shift ? [{ x: G.left, text: "以前运行里的候选" }] : []),
        ...rounds.map((round, index) => {
          const [outcome] = roundOutcome(view, round);
          return { x: colX(index), text: `第 ${round.generation} 代 · ${outcome}` };
        }),
        { x: colX(end), text: done ? "结果" : "现在" },
      ];
      const width = colX(end) + G.node + G.left;
      const height = rowY(rows) + G.height + 6;
      const order = { trunk: 0, fork: 1, external: 2, base: 3, adopt: 4 };
      // scaled down to the card's width, but not below 80% so its text stays readable (a wider graph scrolls);
      // never drawn larger than its size
      return h("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "进化谱系图",
        style: { width: "100%", minWidth: `${Math.round(width * 0.8)}px`, maxWidth: `${width}px`, height: "auto" } },
        heads.map((item) => h("text", { key: `h${item.x}`, x: item.x + 2, y: 14, className: "head" }, item.text)),
        [...edges].sort((a, b) => order[a.kind] - order[b.kind]).map((edge, index) => {
          const d = path(edge);
          return d ? h("path", { key: `e${index}`, className: "edge", "data-kind": edge.kind, d }) : null;
        }),
        [...nodes.values()].map((node) => h("g", { key: node.id, className: "node", "data-kind": node.kind, "data-tone": node.tone,
          "data-pick": String(Boolean(node.pick)), transform: `translate(${node.x},${node.y})`, tabIndex: node.pick ? 0 : undefined,
          role: node.pick ? "button" : undefined, onClick: node.pick || undefined,
          onKeyDown: node.pick ? (event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              node.pick();
            }
          } : undefined },
        h("title", null, node.tip || node.title),
        h("rect", { width: G.node, height: G.height, rx: 9 }),
        h("text", { x: 10, y: 18 }, fit(node.title, G.node - 18)),
        h("text", { x: 10, y: 36, className: "sub" }, node.sub.map((part, index) => h("tspan", { key: index,
          className: part.tone === "eg-good" ? "good" : part.tone === "eg-bad" ? "bad" : undefined }, part.text))))));
    }

    function Swatch({ tone, kind }) {
      return h("svg", { width: 22, height: 14, "aria-hidden": "true" },
        h("g", { className: "node", "data-tone": tone, "data-kind": kind }, h("rect", { x: 1, y: 1, width: 20, height: 12, rx: 3 })));
    }
    function Line({ kind }) {
      return h("svg", { width: 28, height: 10, "aria-hidden": "true" }, h("path", { className: "edge", "data-kind": kind, d: "M1,5 H27" }));
    }

    function LineageCard({ view, catalog, onPick }) {
      if (!(view.rounds || []).length) return null;
      const item = (icon, text) => h("span", { key: text }, icon, text);
      const kinds = new Set();
      for (const round of view.rounds || []) {
        for (const candidate of round.candidates || []) kinds.add(candidate.base?.digest ? "base" : "fork");
      }
      return h(Card, { title: "版本从哪里来", hint: "每一列是一代。最上面一行是在任版本（当前在用的版本）：一路平着连下去，表示一直没有换。下面是这一代评审改出的候选，连线指向它是在哪个版本上改的。候选被采纳时，有一条绿线连到下一代的在任版本。点一个版本，跳到「每一代」里那一代的过程。" },
        h("div", { className: "eg-graph" }, h(Lineage, { view, catalog, onPick })),
        h("div", { className: "eg-legend eg-graph" },
          item(h(Line, { kind: "trunk" }), "在任版本延续到下一代"),
          kinds.has("fork") ? item(h(Line, { kind: "fork" }), "从在任版本改出") : null,
          kinds.has("base") ? item(h(Line, { kind: "external" }), "在以前没采纳的候选上接着改") : null,
          item(h(Line, { kind: "adopt" }), "采纳：成为下一代的在任"),
          item(h(Swatch, { tone: "adopted" }), "采纳"),
          item(h(Swatch, { tone: "danger" }), "没通过检查，没运行")),
        hint("“比在任高 0.016 · 在误差内”：候选的平均成绩比在任版本高 0.016，但这个差别比误差范围小，分不清是真的更好还是运气。"
          + "只有“可信地更好”且高得足够多的候选才会被采纳。"));
    }

    function roundOutcome(view, round) {
      if (!round.finished) return view.active ? ["进行中", "info"] : ["未完成", "warning"];
      if (round.generation === 0) return ["基线", "neutral"];
      if (round.adopted) return [`采纳 ${round.adopted}`, "success"];
      if (round.outcome === "no_candidates") return ["没有可运行的候选", "warning"];
      return ["保留在任", "neutral"];
    }

    /** A generation in one line: what came of it, in numbers (and why its best candidate was not adopted). */
    function roundLine(view, round, tries, catalog) {
      const parts = [];
      const compared = (round.candidates || []).filter((candidate) => isNum(candidate.decision?.delta));
      const describe = (candidate) => {
        const { better, total } = tasksBetter(candidate.decision);
        return `${signed(candidate.decision.delta, 4)}（${total} 个任务里 ${better} 个更好）`;
      };
      if (round.generation === 0) {
        const score = versionMean(round, round.incumbent);
        parts.push(`起点 ${short(round.incumbent).slice(0, 6)}`, isNum(score) ? `均分 ${signed(score, 3)}` : null);
      } else if (round.adopted) {
        const chosen = compared.find((candidate) => candidate.name === round.adopted);
        parts.push(chosen ? `比在任 ${describe(chosen)}` : null);
        const check = round.sealed_check?.error ? null : round.sealed_check?.decision;
        if (check) parts.push(`封存期复核 ${signed(check.delta, 3)}`);
      } else if (compared.length) {
        const best = compared.reduce((top, candidate) => (candidate.decision.delta > top.decision.delta ? candidate : top));
        parts.push(`最好的候选 ${best.name} ${describe(best)}：${reasonText(catalog, best.decision.reason)}`);
      } else if ((round.candidates || []).length) {
        const counts = {};
        for (const candidate of round.candidates) counts[candidate.status] = (counts[candidate.status] || 0) + 1;
        parts.push(Object.entries(counts).map(([status, count]) => `${count} 个${(CAND_STATUS[status] || [status])[0]}`).join("、"));
      }
      parts.push(`${(round.episodes || []).length} 个回合`, round.evolver ? `评审 ${duration(round.evolver.seconds)}` : null);
      if (tries > 1) parts.push(`跑了 ${tries} 次（前 ${tries - 1} 次被中断，不算数）`);
      return parts.filter(Boolean).join(" · ");
    }

    function Process({ view, catalog, go, focus }) {
      const rounds = lastAttempts(view); // a generation run again after an interruption: the attempt that counts
      const [open, setOpen] = useState(() => new Set([rounds.length - 1]));
      useEffect(() => setOpen((old) => (old.has(rounds.length - 1) ? old : new Set([...old, rounds.length - 1]))), [rounds.length]);
      useEffect(() => {
        if (!focus) return undefined;
        let index = -1;
        rounds.forEach((round, i) => {
          if (round.generation === focus.generation) index = i;
        });
        if (index < 0) return undefined;
        setOpen((old) => (old.has(index) ? old : new Set([...old, index])));
        const timer = setTimeout(() => {
          const target = (focus.name && document.getElementById(`eg-cand-${index}-${focus.name}`)) || document.getElementById(`eg-gen-${index}`);
          target?.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 120);
        return () => clearTimeout(timer);
      }, [focus]);
      if (!rounds.length) return null;
      const toggle = (index, value) => setOpen((old) => {
        if (old.has(index) === value) return old;
        const next = new Set(old);
        if (value) next.add(index);
        else next.delete(index);
        return next;
      });
      const tries = (generation) => (view.rounds || []).filter((item) => item.generation === generation).length;
      return h(Card, { title: "每一代的过程", hint: "每一行是一代，点开按顺序看：评审读了什么、找到什么问题；候选从哪里出发、改了什么、为什么；宿主的检查；候选和在任版本在各训练任务上的回合；比较和决定。" },
        rounds.map((round, index) => h(GenerationStory, { key: round.generation, view, round, index, catalog, go, tries: tries(round.generation),
          open: open.has(index), onToggle: toggle, focus: focus?.generation === round.generation ? focus : null })));
    }

    function GenerationStory({ view, round, index, catalog, go, open, onToggle, focus, tries }) {
      const [text, tone] = roundOutcome(view, round);
      const line = roundLine(view, round, tries, catalog);
      const back = { page: "evolution", id: view.run_id, focus: { generation: round.generation } };
      const steps = round.generation === 0 ? [
        ["起点版本在每个训练任务上运行", h(EpisodeGrid, { view, round, catalog, go, back })],
        ["结果", h(Outcome, { view, round })],
      ] : [
        ["评审：读证据，找问题", h(EvolverStep, { view, round, go, back })],
        ["候选：从哪里出发，改了什么，为什么", h(CandidatesStep, { view, round, index, catalog, focus })],
        ["回合：候选和在任版本在每个训练任务上各跑几次", h(EpisodeGrid, { view, round, catalog, go, back })],
        ["比较（候选 − 在任）与结果", h(Fragment, null, h(DecisionStep, { view, round, catalog }), h(Outcome, { view, round }))],
      ];
      return h("details", { id: `eg-gen-${index}`, className: "eg-round", open, onToggle: (event) => onToggle(index, event.currentTarget.open) },
        h("summary", null, h("span", { className: "eg-round-no" }, `第 ${round.generation} 代`), h(Tag, { tone }, text),
          h("span", { className: "eg-round-sum", title: line }, line)),
        open ? h("div", { className: "eg-round-body eg-steps" }, steps.map(([title, body], i) => h("div", { key: title, className: "eg-story" },
          h("span", { className: "eg-story-no" }, i + 1), h("h3", null, title), h("div", { className: "eg-story-body" }, body)))) : null);
    }

    function EvolverStep({ view, round, go, back }) {
      const evolver = round.evolver;
      const recorded = Boolean(evolver?.trajectory?.dir);
      const { data: session, error } = useResource(recorded ? Api.evolverSession(view.run_id, round.generation, false) : null);
      if (!evolver) {
        return hint(view.active && !round.finished ? "评审智能体正在阅读证据、编写候选…" : "这一代没有评审记录。");
      }
      const page = () => go({ page: "evolver", id: `${view.run_id}|${round.generation}`, back });
      const trouble = sessionTrouble(evolver);
      const earlier = (round.earlier_evolvers || []).filter((item) => item.rerun).length;
      const redone = earlier && !evolver.rerun ? hint(`这一代的第一次评审会话被模型服务故障打断，下面是服务恢复后重新做的评审（之前那次用了 ${duration(round.earlier_evolvers[0].seconds)}）。`) : null;
      const steps = recorded && session?.recorded !== false; // a session whose model never answered leaves no steps
      const final = String(evolver.final_text || "").trim();
      return h(Fragment, null,
        h("div", { className: "eg-story-sub" }, [evolver.status === "finished" ? null : SESSION_STATUS[evolver.status]?.[0] || evolver.status,
          duration(evolver.seconds), evolver.tokens ? `${tokens(evolver.tokens)} token` : null, `${evolver.llm_calls ?? "–"} 次模型调用`,
          session?.turns ? `${session.turns} 步` : null, evolver.continuations ? `含 ${evolver.continuations} 段续跑` : null].filter(Boolean).join(" · "),
        steps ? h(Fragment, null, " · ", h(Link, { onClick: page }, "看评审的每一步 →")) : null),
        trouble ? h(Alert, { tone: "danger" }, h("b", null, trouble[0]), trouble[1]) : null,
        redone,
        error ? h(Alert, { tone: "warning" }, `读不到评审会话：${error.message}`) : null,
        session?.recorded ? h(Reads, { reads: session.reads, go, back, label: "看过的材料（按第一次看的先后；悬停看是第几步）" })
          : session ? hint("没有存下这次会话的模型调用，看不到它读了什么。")
            : recorded && !error ? hint("正在读取评审会话…") : null,
        final ? h(Fragment, null, h("div", { className: "eg-label" }, "评审的判断（会话最后的回复）"),
          h(Clamp, { height: 280 }, h(Markdown, { text: final }))) : null,
        (round.notes || []).length ? h(Fragment, null, h("div", { className: "eg-label" }, "评审记下的笔记（以后各代的证据里会出现）"),
          round.notes.map((note, i) => h("div", { key: i, className: "eg-note" }, h("span", { className: "eg-chip" }, NOTE_KIND[note.kind] || note.kind), " ", note.text))) : null);
    }

    const READ_GROUPS = { brief: "说明", guide: "说明", evidence: "证据", summary: "回合摘要", transcript: "回合会话记录", program: "回合的程序",
      episode: "回合", archive: "存档版本", incumbent: "在任版本", candidate: "候选", proposal: "提案", other: "其他" };
    const EVIDENCE_FILES = { "overview.md": "证据总览", "archive.md": "以往候选", "lessons.md": "内循环笔记", "contrasts.md": "回合对照",
      "prior_runs.md": "以往运行", "expert.md": "专家意见" };
    const whereText = (where) => (where[0] ? `线程 ${where[0] + 1} 第 ${where[1]} 步` : `第 ${where[1]} 步`);

    function readText(item) {
      if (item.kind === "brief") return "任务说明 EVOLVE.md";
      if (item.kind === "guide") return "评审指南 AGENTS.md";
      if (item.kind === "evidence") return EVIDENCE_FILES[item.key] ? `${EVIDENCE_FILES[item.key]} ${item.key}` : item.key;
      if (["summary", "transcript", "program", "episode"].includes(item.kind)) return item.key === "*" ? "全部回合" : short(item.key.replace(/^ep:/, ""));
      if (item.kind === "archive" && item.key === "README.md") return "存档说明 README.md";
      if (item.kind === "candidate") return `候选 ${item.key}`;
      if (item.kind === "proposal") return `${item.key}.md`;
      return item.key;
    }

    /** What a session looked at in its workspace, grouped by kind; a chip opens the episode or jumps to the step. */
    function Reads({ reads, go, back, onStep, label }) {
      const items = reads || [];
      const groups = new Map();
      for (const item of items) {
        const name = READ_GROUPS[item.kind] || item.kind;
        if (!groups.has(name)) groups.set(name, []);
        groups.get(name).push(item);
      }
      const chip = (item) => {
        const key = `${item.kind}:${item.key}`;
        const title = `${item.paths.join("\n")}\n${item.steps.map(whereText).join("、")}`;
        if (onStep) return h("button", { key, type: "button", className: "eg-chip", title, onClick: () => onStep(item.steps[0]) }, readText(item));
        if (go && /^ep:/.test(item.key)) {
          return h("button", { key, type: "button", className: "eg-chip", title: `${title}\n点开这个回合`, onClick: () => go({ page: "episode", id: item.key, back }) }, readText(item));
        }
        return h("span", { key, className: "eg-chip", title }, readText(item));
      };
      const transcripts = items.filter((item) => item.kind === "transcript").length;
      return h(Fragment, null,
        label ? h("div", { className: "eg-label" }, label) : null,
        items.length ? h("dl", { className: "eg-reads" }, [...groups.entries()].flatMap(([name, list]) => [
          h("dt", { key: `t${name}` }, name), h("dd", { key: `d${name}` }, list.map(chip))]))
          : hint("会话里没有读工作区文件的记录。"),
        items.length && !transcripts ? h("p", { className: "eg-hint eg-warn" }, "没有打开任何回合的会话记录（evidence/episodes/…/transcript.md）。") : null);
    }

    function CandidatesStep({ view, round, index, catalog, focus }) {
      const candidates = round.candidates || [];
      if (!candidates.length) {
        return hint(round.evolver ? "评审没有留下候选。" : "评审会话结束后出现。");
      }
      return candidates.map((candidate) => h(CandidateStory, { key: candidate.name, view, round, index, candidate, catalog,
        focused: focus?.name === candidate.name }));
    }

    function CandidateStory({ view, index, candidate, catalog, focused }) {
      const base = candidate.base?.digest ? candidate.base : null;
      const [against, setAgainst] = useState(base ? "base" : "incumbent");
      const fromBase = against === "base" && base && candidate.digest;
      const { data: baseDiff, error } = useResource(fromBase ? Api.harnessDiff(base.digest, candidate.digest) : null);
      const changes = fromBase ? baseDiff : candidate.changes || [];
      return h("div", { id: `eg-cand-${index}-${candidate.name}`, className: "eg-cand", "data-adopted": String(candidate.status === "adopted"),
        "data-focus": String(Boolean(focused)) },
      h("div", { className: "eg-cand-head" }, h("b", null, `候选 ${candidate.name}`), h(StatusTag, { map: CAND_STATUS, keyName: candidate.status }),
        h(Mono, { title: candidate.digest }, short(candidate.digest)), h("span", { className: "eg-faint" }, baseText(view, candidate))),
      candidate.summary ? h("div", { className: "eg-cand-summary" }, candidate.summary) : null,
      h("div", { className: "eg-label" }, base ? h(Fragment, null, "改了什么　",
        h("span", { className: "eg-switch" }, [["base", `相对出发的版本 ${base.label || short(base.digest)}`], ["incumbent", "相对在任版本（比较的对象）"]]
          .map(([key, text]) => h("button", { key, type: "button", "aria-pressed": String(against === key), onClick: () => setAgainst(key) }, text))))
        : "改了什么（相对在任版本）"),
      error ? h(Alert, { tone: "warning" }, `读不到两个版本的差异：${error.message}`)
        : !changes ? hint("正在读取…")
          : changes.length ? h(Fragment, null,
            h("div", { className: "eg-chips" }, changes.map((item) => h("span", { key: item.path, className: "eg-chip" }, item.path, " ",
              h("span", { className: "eg-good" }, `+${item.added ?? 0}`), h("span", { className: "eg-bad" }, `${MINUS}${item.removed ?? 0}`)))),
            h(Details, { key: against, summary: `看改动（${changes.length} 个文件）` },
              changes.map((item) => h(Diff, { key: item.path, patch: item.patch || `${item.status} ${item.path}` }))))
            : hint(fromBase ? "和出发的版本相同。" : "没有改动。"),
      h("div", { className: "eg-label" }, "提案：为什么这样改（评审写的）"),
      written(candidate.proposal) ? h(Clamp, { height: 200 }, h(Markdown, { text: candidate.proposal }))
        : hint("没有写提案（只有宿主给的空模板）。"),
      h(CheckView, { candidate, catalog }));
    }

    /** A proposal with content, not only the headings of the skeleton the host puts in proposals/. */
    const written = (text) => String(text || "").split("\n").some((line) => line.trim() && !line.trim().startsWith("#"));

    /** The host's own check of a candidate after the session: problems, warnings, the template's local scores. */
    function CheckView({ candidate, catalog }) {
      const smoke = candidate.smoke;
      const tasks = Object.entries(smoke?.tasks || {});
      const problems = candidate.problems || [];
      const warnings = candidate.warnings || [];
      let body;
      if (tasks.length) {
        body = h(Table, { head: ["训练任务", ["候选的模板", "num"], ["在任的模板", "num"], ["差", "num"]] },
          tasks.map(([task, item]) => {
            const change = item.status === "ok" && isNum(item.score) && isNum(item.incumbent_score) ? item.score - item.incumbent_score : null;
            return h("tr", { key: task }, h("td", null, h(TaskLabel, { catalog, id: task })),
              h("td", { className: "num" }, item.status === "ok" ? signed(item.score, 3) : h("span", { className: "eg-bad" }, item.status)),
              h("td", { className: "num" }, signed(item.incumbent_score, 3)),
              h("td", { className: `num ${toneClass(change)}` }, signed(change, 3)));
          }));
      } else if (smoke) {
        body = hint(`模板试运行：${smoke.status}，本地 ${signed(smoke.score, 3)}`
          + `${isNum(smoke.incumbent_score) ? `（在任的模板 ${signed(smoke.incumbent_score, 3)}）` : ""}，覆盖率 ${Math.round((smoke.coverage || 0) * 100)}%`);
      } else if (candidate.status !== "invalid") {
        body = hint("模板代码没有改动，不需要试运行。");
      }
      return h(Fragment, null,
        h("div", { className: "eg-label" }, "宿主的检查：能否加载、是否超限、有没有任务的专有名称；模板改了就在本地检验期上实际运行（不是隐藏期）"),
        body || null,
        problems.length ? h(Alert, { tone: "warning" }, h("b", null, "没有通过的检查："), problems.map((item, i) => h("div", { key: i }, item))) : null,
        warnings.length ? h(Alert, { tone: "warning" }, h("b", null, "提醒（不阻止运行）："), warnings.map((item, i) => h("div", { key: i }, item))) : null);
    }

    /** The incumbent's episodes from earlier rounds that a round's comparison also used (``evolution.loop``'s pool):
     * finished rounds after the incumbent was adopted, under the same expert guidance. */
    function earlierIncumbent(view, round) {
      if (round.generation === 0) return [];
      const adoptedAt = Math.max(-1, ...(view.adoptions || []).filter((item) => item.digest === round.incumbent).map((item) => item.generation));
      return (view.rounds || []).filter((other) => other !== round && other.finished && other.generation > adoptedAt && other.generation < round.generation)
        .flatMap((other) => (other.episodes || []).filter((entry) => entry.role === "incumbent" && entry.harness === round.incumbent
          && SCORED.includes(entry.status) && (entry.guidance ?? null) === (round.guidance?.[entry.task] ?? null))
          .map((entry) => ({ ...entry, generation: other.generation })));
    }

    /** A round's episodes: a row per version, a column per training task, one chip per episode (click opens it). */
    function EpisodeGrid({ view, round, catalog, go, back }) {
      const tasks = view.suite?.train || [];
      const entries = round.episodes || [];
      if (!entries.length) {
        return hint(!round.finished && view.active ? "回合正在运行（见「概览」里的「正在进行」），全部结束后出现在这里。" : "这一代没有回合记录。");
      }
      const earlier = earlierIncumbent(view, round);
      const rows = [{ key: "incumbent", label: round.generation === 0 ? "起点版本" : "在任", match: (entry) => entry.role === "incumbent" }];
      for (const candidate of round.candidates || []) {
        if (entries.some((entry) => entry.role === "candidate" && entry.candidate === candidate.name)) {
          rows.push({ key: candidate.name, label: `候选 ${candidate.name}`, match: (entry) => entry.role === "candidate" && entry.candidate === candidate.name });
        }
      }
      const scores = (list) => list.filter((entry) => SCORED.includes(entry.status) && isNum(entry.score)).map((entry) => entry.score);
      const bases = new Map(tasks.map((task) => [task, mean(scores([...entries, ...earlier].filter((entry) => entry.role === "incumbent" && entry.task === task)))]));
      const chip = (entry, faded) => {
        const ok = SCORED.includes(entry.status) && isNum(entry.score);
        const base = round.generation ? bases.get(entry.task) : null; // the colour: above or below the incumbent's mean
        return h("button", { key: entry.episode || `${entry.task}-${entry.replicate}-${entry.generation ?? ""}`, type: "button", className: "eg-score",
          disabled: !entry.episode, style: { background: ok && isNum(base) ? mix(entry.score - base, 0.03) : undefined, opacity: faded ? 0.5 : undefined },
          title: `${entry.episode || "回合没能开始"}${faded ? `：第 ${entry.generation} 代的在任回合，比较时一并计入` : ""}\n`
            + `${EP_STATUS[entry.status]?.[0] || entry.status} · ${tokens(entry.tokens)} token · ${duration(entry.seconds)}`
            + `${entry.error ? `\n${String(entry.error).slice(0, 300)}` : ""}`,
          onClick: () => go({ page: "episode", id: entry.episode, back }) },
        ok ? signed(entry.score, 3) : h("span", { className: "eg-bad" }, EP_STATUS[entry.status]?.[0] || entry.status));
      };
      const pooled = earlier.length ? Math.round((earlier.length + entries.filter((entry) => entry.role === "incumbent").length) / Math.max(1, tasks.length)) : 0;
      return h(Fragment, null,
        h(Table, { head: ["版本", ...tasks.map((task) => taskHead(catalog, task, "")), ["平均", "num", "各任务回合均值的平均（在任一行含淡色的早先回合）"]] },
          rows.map((row) => {
            const means = [];
            return h("tr", { key: row.key }, h("td", { style: { whiteSpace: "nowrap" } }, row.label),
              tasks.map((task) => {
                const own = entries.filter((entry) => row.match(entry) && entry.task === task);
                const before = row.key === "incumbent" ? earlier.filter((entry) => entry.task === task) : [];
                const values = scores([...own, ...before]);
                if (values.length) means.push(mean(values));
                return h("td", { key: task }, h("div", { className: "eg-cell-chips" }, own.map((entry) => chip(entry, false)), before.map((entry) => chip(entry, true))));
              }),
              h("td", { className: "num" }, h("b", null, means.length === tasks.length ? signed(mean(means), 3) : "–")));
          })),
        hint("每个格子是一个回合的隐藏期总技能（相对朴素基线，大于 0 更好），点开看那个回合（悬停看状态、token 和用时）。"
          + (round.generation ? "底色表示比在任版本在这个任务上的均值高（绿）还是低（红）。" : "")
          + (earlier.length ? `淡色是在任版本在之前各代跑过的回合（专家意见相同），比较时和本代的一起算，每个任务约 ${pooled} 个。` : "")
          + (round.generation ? "同一个版本的几个回合之间差别很大，所以要多跑几次、在多个任务上比较。" : "")));
    }

    /** Why a comparison came out as it did, in words. */
    function verdict(decision, gates, catalog) {
      const range = `[${signed(decision.interval?.[0], 3)}, ${signed(decision.interval?.[1], 3)}]`;
      const { better, total } = tasksBetter(decision);
      const head = `平均增益 ${signed(decision.delta, 4)}，${total} 个训练任务里 ${better} 个更好。`;
      const reason = String(decision.reason || "");
      if (reason === "improved") {
        return h(Fragment, null, head, h("b", { className: "eg-good" }, " 通过全部门槛："),
          `增益超过实用门槛 ${signed(gates?.min_delta, 3)}，区间 ${range} 整个在 0 以上，没有任务可信变差。`);
      }
      if (reason === "interval_crosses_zero") {
        return h(Fragment, null, head, h("b", null, " 不采纳："),
          `区间 ${range} 包含 0。按回合之间的随机差异，这个增益还可能来自运气；要分清，需要更多回合或更多训练任务。`);
      }
      if (reason === "below_practical_delta") {
        return h(Fragment, null, head, h("b", null, " 不采纳："), `增益没有超过实用门槛 ${signed(gates?.min_delta, 3)}，差别太小，不值得换版本。`);
      }
      if (reason.startsWith("task_regression:")) {
        return h(Fragment, null, head, h("b", { className: "eg-bad" }, " 不采纳："), `${reasonText(catalog, reason)}：有任务可信变差时一票否决。`);
      }
      if (reason.startsWith("missing_tasks:")) return h(Fragment, null, h("b", null, "没有比较："), `${reasonText(catalog, reason)}。`);
      return h(Fragment, null, head, " ", reasonText(catalog, reason));
    }

    function DecisionStep({ view, round, catalog }) {
      const compared = (round.candidates || []).filter((candidate) => candidate.decision);
      if (!compared.length) {
        return hint(round.finished ? "没有候选进入比较。" : view.active ? "回合全部结束后比较。" : unfinishedText(view));
      }
      const gate = view.gates?.min_delta;
      const values = [0, ...(isNum(gate) ? [gate] : [])];
      for (const { decision } of compared) {
        values.push(...[decision.delta, ...(decision.interval || [])].filter(isNum));
        for (const item of decision.tasks || []) values.push(...[item.delta, ...(item.interval || [])].filter(isNum));
      }
      let lo = Math.min(...values);
      let hi = Math.max(...values);
      const span = Math.max(hi - lo, 0.004);
      lo -= span * 0.06;
      hi += span * 0.06;
      const x = (value) => ((value - lo) / (hi - lo)) * 100;
      const { ticks, digits } = niceTicks(lo, hi);
      const track = (delta, interval, strong) => h("div", { className: "eg-track", "data-adopted": String(Boolean(strong)) },
        h("i", { className: "zero", style: { left: `${x(0)}%` } }),
        isNum(gate) ? h("i", { className: "gate", style: { left: `${x(gate)}%` } }) : null,
        isNum(interval?.[0]) && isNum(interval?.[1]) ? h("i", { className: "ci", style: { left: `${x(interval[0])}%`, width: `${Math.max(0.5, x(interval[1]) - x(interval[0]))}%` } }) : null,
        isNum(delta) ? h("i", { className: "dot", style: { left: `${x(delta)}%` } }) : null);
      const range = (interval) => `[${signed(interval?.[0], 3)}, ${signed(interval?.[1], 3)}]`;
      const first = compared[0].decision;
      const level = ((first.confidence || 0.95) * 100).toFixed(1).replace(/\.0$/, "");
      const taskLevel = ((first.task_confidence || 0.95) * 100).toFixed(0);
      const blocks = compared.map((candidate) => {
        const decision = candidate.decision;
        return h("div", { key: candidate.name, className: "eg-cand" },
          h("div", { className: "eg-cand-head" }, h("b", null, `候选 ${candidate.name}`), h(StatusTag, { map: CAND_STATUS, keyName: candidate.status })),
          h("div", { className: "eg-forest", style: { marginTop: "8px" } },
            (decision.tasks || []).flatMap((item) => [
              h("div", { key: `l${item.task}` }, h(TaskLabel, { catalog, id: item.task })),
              h("div", { key: `t${item.task}` }, track(item.delta, item.interval, false)),
              h("div", { key: `v${item.task}`, title: `候选 ${signed(item.candidate, 3)}（${item.replicates?.[0] ?? "–"} 个回合）− 在任 ${signed(item.incumbent, 3)}（${item.replicates?.[1] ?? "–"} 个回合）` },
                h("span", { className: toneClass(item.delta) }, signed(item.delta, 3)), ` ${range(item.interval)} `,
                h("span", { className: item.reason === "credible_regression" ? "eg-bad" : "eg-faint" }, reasonText(catalog, item.reason))),
            ]),
            ...(isNum(decision.delta) ? [
              h("div", { key: "lm" }, h("b", null, "平均")),
              h("div", { key: "tm" }, track(decision.delta, decision.interval, decision.adopted)),
              h("div", { key: "vm" }, h("b", { className: toneClass(decision.delta) }, signed(decision.delta, 3)), ` ${range(decision.interval)}`),
            ] : []),
            h("div", { key: "axis-l" }),
            h("div", { key: "axis", className: "eg-axis" }, ticks.map((tick) => h("span", { key: tick, style: { left: `${x(tick)}%` } }, signed(tick, digits)))),
            h("div", { key: "axis-r" })),
          h("div", { className: "eg-verdict" }, verdict(decision, view.gates, catalog)));
      });
      return h(Fragment, null, blocks,
        hint(`每个任务一行：候选回合的均值 − 在任回合的均值，点是增益，横条是 ${taskLevel}% 区间（悬停看两边的均值和回合数）。`
          + `“平均”一行是 ${level}% 区间${compared.length > 1 ? "（按这一代的候选数加宽）" : ""}。竖线是 0，虚线是实用门槛 ${signed(gate, 3)}。`
          + "采纳要同时满足：平均增益超过门槛、区间下限大于 0、没有任务可信变差。"));
    }

    /** A generation that did not finish: run again on resume, or left out when the run was ended there. */
    const unfinishedText = (view) => (view.status === "finished" ? "运行在这一代完成之前结束了，这一代不算数。" : "这一代没有完成；继续运行时会重跑这一代。");

    function Outcome({ view, round }) {
      if (!round.finished) return round.generation ? null : hint(view.active ? "这一代还在进行。" : unfinishedText(view));
      if (round.generation === 0) {
        return h("p", null, `起点版本 ${short(round.incumbent)} 在各训练任务上的均分 ${signed(versionMean(round, round.incumbent), 3)}。`
          + "这些回合是第 1 代比较时在任版本的成绩；之后每一代，在任版本还会和候选同时再跑。");
      }
      if (round.adopted) {
        const check = checkText(round.sealed_check);
        return h(Fragment, null, h("p", null, h("b", { className: "eg-good" }, `采纳候选 ${round.adopted}：`),
          `从下一代起，在任版本由 ${short(round.incumbent)} 换成 ${short(round.incumbent_after)}。`),
        check ? hint(check) : null);
      }
      if (round.outcome === "no_candidates") return h("p", null, `没有可以运行的候选，在任版本仍是 ${short(round.incumbent)}。`);
      return h("p", null, `没有候选被采纳，在任版本仍是 ${short(round.incumbent)}。`
        + "比较过的候选留在存档里：以后各代的评审能看到它们的改动和结果，也可以从它们出发再改。");
    }

    /* ---- the run at a glance: what it did, in words; its score over the generations --------------------- */

    /** Where a difference stands against zero: credibly above, credibly below, or within the noise. */
    function credibility(interval) {
      const [low, high] = interval || [];
      if (isNum(low) && low > 0) return "above";
      if (isNum(high) && high < 0) return "below";
      return "noise";
    }
    const CREDIBLE_TEXT = { above: "可信地更好", below: "可信地更差", noise: "在误差内" };
    const runShort = (id) => String(id || "").replace(/^evo:/, "").slice(0, 6);
    const day = (value) => when(value).slice(0, 5);

    /** A version's short name in this run: 起点, or the candidate that became it (g2b). */
    function versionAlias(view, digest) {
      if (digest === view.seed?.digest) return "起点";
      for (const round of lastAttempts(view)) {
        const found = (round.candidates || []).find((candidate) => candidate.digest === digest);
        if (found) return `g${round.generation}${found.name}`;
      }
      return short(digest).slice(0, 6);
    }

    /** The starting version's own name: the earlier run's candidate it is (9eca52-g2b), or the shipped seed. */
    function seedOrigin(view) {
      const step = (view.ancestry || [])[0];
      if (step) return `${runShort(step.run_id)}-g${step.generation}${step.name}`;
      return view.seed?.name === "seed" ? "初始版本" : view.seed?.name || "";
    }

    /** digest -> score: an incumbent's mean over training tasks of its means in its last comparison (all its
     * episodes as the incumbent), else of its own episodes; an adopted candidate's own mean when it never ran as
     * the incumbent. */
    function versionLevels(view) {
      const levels = new Map();
      const adopted = new Map();
      for (const round of lastAttempts(view)) {
        const compared = (round.candidates || []).find((candidate) => (candidate.decision?.tasks || []).length);
        const pooled = compared ? mean(compared.decision.tasks.map((item) => item.incumbent).filter(isNum)) : null;
        const own = versionMean(round, round.incumbent);
        if (isNum(pooled)) levels.set(round.incumbent, pooled);
        else if (!levels.has(round.incumbent) && isNum(own)) levels.set(round.incumbent, own);
        const chosen = (round.candidates || []).find((candidate) => candidate.status === "adopted");
        const score = chosen ? mean((chosen.decision?.tasks || []).map((item) => item.candidate).filter(isNum)) : null;
        if (isNum(score)) adopted.set(chosen.digest, score);
      }
      return (digest) => (levels.has(digest) ? levels.get(digest) : adopted.get(digest));
    }

    /** The comparisons of a run (a candidate with a measured gain), with the means the gate compared. */
    function comparisons(view) {
      return lastAttempts(view).filter((round) => round.generation > 0).flatMap((round) => (round.candidates || [])
        .filter((candidate) => isNum(candidate.decision?.delta) && (candidate.decision.interval || []).every(isNum))
        .map((candidate) => {
          const tasks = candidate.decision.tasks || [];
          return { round, candidate, decision: candidate.decision, side: credibility(candidate.decision.interval),
            before: mean(tasks.map((item) => item.incumbent).filter(isNum)), after: mean(tasks.map((item) => item.candidate).filter(isNum)) };
        }));
    }

    /** "从 a 提到 b", "从 a 降到 b", or "从 a 到 b（差别在误差内）", by the interval of the change. */
    function move(from, to, interval) {
      const side = credibility(interval);
      if (side === "above") return `从 ${signed(from, 3)} 提到 ${signed(to, 3)}`;
      if (side === "below") return `从 ${signed(from, 3)} 降到 ${signed(to, 3)}`;
      return `从 ${signed(from, 3)} 到 ${signed(to, 3)}（差别在误差内）`;
    }

    const NOT_ADOPTED = [["passed", "也通过了门槛（同一代只采纳增益最大的一个）"], ["veto", "有训练任务可信地变差（一票否决）"],
      ["small", "可信地更好但高得不够多（没超过门槛）"], ["noise", "在误差内（分不清好坏）"], ["below", "可信地更差"]];

    /** The compared candidates that were not adopted, in one sentence: why, and on how many tasks they were better. */
    function notAdoptedText(rest, adoptions) {
      const kind = (item) => (item.candidate.status === "passed" ? "passed" : item.side === "below" ? "below"
        : String(item.decision.reason || "").startsWith("task_regression") ? "veto" : item.side === "above" ? "small" : "noise");
      const counts = {};
      for (const item of rest) counts[kind(item)] = (counts[kind(item)] || 0) + 1;
      const listed = NOT_ADOPTED.filter(([name]) => counts[name]);
      const many = rest.length > 1;
      const head = adoptions ? `另外 ${rest.length} 个候选没有被采纳`
        : `${new Set(rest.map((item) => item.round.generation)).size} 代共 ${rest.length} 个候选进入比较，${many ? "没有一个" : "没有"}被采纳`;
      const why = listed.length === 1 ? `，${many ? "都" : ""}${listed[0][1]}` : `：${listed.map(([name, text]) => `${counts[name]} 个${text}`).join("，")}`;
      const better = rest.map((item) => tasksBetter(item.decision));
      const [lo, hi] = [Math.min(...better.map((item) => item.better)), Math.max(...better.map((item) => item.better))];
      return `${head}${why}；${many ? "它们" : "它"}在 ${Math.max(...better.map((item) => item.total))} 个训练任务里有 ${lo === hi ? lo : `${lo}～${hi}`} 个更好。`;
    }

    /** The run in plain words: what came out and what that means; what it does, where it started and how to read a
     * score are folded away. */
    function RunSummary({ view, go }) {
      const rounds = lastAttempts(view);
      const tasks = view.suite?.train || [];
      const config = view.config || {};
      const level = versionLevels(view);
      const compared = comparisons(view);
      const adopted = compared.filter((item) => item.candidate.status === "adopted");
      const rest = compared.filter((item) => item.candidate.status !== "adopted");
      const origin = (view.ancestry || [])[0];
      const key = evoKey(view);
      const finished = view.status === "finished";
      const live = key === "running" || key === "elsewhere";
      const finalScore = level(view.incumbent);
      const done = rounds.filter((round) => round.generation > 0 && round.finished).length;
      const link = (step) => h(Link, { onClick: () => go({ page: "evolution", id: step.run_id }) }, step.run_id);

      let headline;
      if (live) headline = `正在进行：${phaseOf(view)}。`;
      else if (adopted.length) {
        headline = `进化了 ${adopted.length} 次：训练任务上的成绩从 ${signed(adopted[0].before, 3)} 提到 ${signed(adopted[adopted.length - 1].after, 3)}。`;
      } else if (compared.length) headline = "这次没有找到可信的提升，最终仍用起点版本。";
      else headline = "这次没有产出能比较的候选，最终仍用起点版本。";
      if (!finished && !live) headline = `运行${EVO_STATUS[key][0]}（完成了 ${done} / ${config.generations ?? "?"} 代）。${adopted.length ? headline : ""}`;
      if (origin && !adopted.length) headline += `起点本身是 ${day(origin.started_at)} 那次运行进化出来的版本。`;

      const what = `研究智能体按一份工作说明（Harness：指令、技能、代码模板和经验），独立去做「${view.suite?.title || view.suite?.id}」里的 `
        + `${tasks.length} 个训练任务，每个任务是一个预报问题。每一代，评审智能体读完各回合的成绩和过程，改出 ${config.candidates ?? "几"} 个新版本（候选）；`
        + `候选和当前在用的版本（在任版本）在每个训练任务上各跑 ${config.repeats ?? "几"} 次，只有比在任版本可信地更好，才换上。共 ${config.generations ?? "?"} 代。`;

      let start;
      if (origin) {
        start = [`它不是从零开始：它是 ${day(origin.started_at)} 那次运行（`, link(origin), `）第 ${origin.generation} 代采纳的候选 ${origin.name}。`];
        if (origin.suite === view.suite?.id && isNum(origin.before) && isNum(origin.after)) {
          start.push(`那次把训练任务上的成绩${move(origin.before, origin.after, origin.interval)}`);
        } else start.push("那次的训练任务和这次不同，成绩不能直接比");
        const holdout = origin.holdout;
        if (holdout && isNum(holdout.seed) && isNum(holdout.final)) {
          start.push(`；在 ${holdout.tasks} 个没参与进化的任务上${move(holdout.seed, holdout.final, holdout.interval)}`);
        }
        const sealed = origin.sealed?.final_test || origin.sealed?.validation;
        if (sealed && isNum(sealed.seed) && isNum(sealed.final)) start.push(`；在更晚、谁都没见过的时期上${move(sealed.seed, sealed.final, sealed.interval)}`);
        start.push("。这次是在这个已经改进过、验证过的版本上继续找提升。");
        if ((view.ancestry || []).length > 1) start.push(`再往前还有 ${view.ancestry.length - 1} 次运行进化过它。`);
      } else {
        start = [view.seed?.name === "seed" ? "起点是项目自带的初始版本。" : `起点是版本 ${view.seed?.name || short(view.seed?.digest)}。`];
      }

      const endWord = finished ? "最终版本" : "现在在任的版本";
      let result;
      if (!compared.length) {
        const failed = rounds.filter((round) => round.generation > 0 && round.evolver && !SCORED.includes(round.evolver.status));
        const counts = {};
        for (const round of rounds) {
          for (const candidate of round.candidates || []) {
            if (!isNum(candidate.decision?.delta)) counts[candidate.status] = (counts[candidate.status] || 0) + 1;
          }
        }
        const listed = Object.entries(counts).map(([status, count]) => `${count} 个${(CAND_STATUS[status] || [status])[0]}`).join("、");
        const why = [...new Set(failed.map((round) => (SESSION_STATUS[round.evolver.status] || [round.evolver.status])[0]))].join("、");
        const details = [failed.length ? `${failed.length} 代的评审会话没有正常结束（${why}）` : "", listed ? `候选里 ${listed}` : ""].filter(Boolean).join("；");
        result = live && !done ? `第 1 代的比较还没有完成${details ? `（${details}）` : ""}。`
          : `没有候选进入比较${details ? `：${details}` : ""}。${endWord}仍是起点。`;
      } else {
        const held = (round) => {
          const decision = round.sealed_check?.error ? null : round.sealed_check?.decision;
          if (!decision) return "";
          const { better, total } = tasksBetter(decision);
          return `在训练任务更晚、谁都没看过的封存时期上复核，它比被替换的版本${decision.delta >= 0 ? "高" : "低"} ${plain(Math.abs(decision.delta), 3)}`
            + `（${CREDIBLE_TEXT[credibility(decision.interval)]}，${total} 个任务里 ${better} 个更好）。`;
        };
        const parts = adopted.map(({ round, candidate, before, after, decision }) => `第 ${round.generation} 代采纳了候选 ${candidate.name}：`
          + `训练任务上的成绩${move(before, after, decision.interval)}，提升了 ${plain(decision.delta, 3)}（这个提升的误差范围 [${signed(decision.interval[0], 3)}, ${signed(decision.interval[1], 3)}]，整段在 0 以上）。`
          + held(round));
        if (rest.length) parts.push(notAdoptedText(rest, adopted.length));
        parts.push(`${endWord}${adopted.length ? `是 ${versionAlias(view, view.incumbent)}` : "仍是起点"}。`);
        result = parts.join("");
      }

      let next = "";
      if (finished && !adopted.length && compared.length && compared.every((item) => item.side === "noise" && item.candidate.status !== "passed"
        && !String(item.decision.reason || "").startsWith("task_regression"))) {
        next = "这些候选还算不上一致的改进：它们和在任版本的差别比误差范围小，或者在一部分训练任务上更好、另一部分更差。"
          + "没过门槛的候选留在存档里，以后的评审可以在它们上面接着改；要分清更小的差别，需要更多训练任务。";
      } else if (adopted.length && finished) {
        const holdout = view.verification?.holdout;
        const sealed = view.verification?.sealed?.final_test || view.verification?.sealed?.validation;
        const checks = [];
        if (holdout && isNum(holdout.means?.seed) && isNum(holdout.means?.final)) {
          checks.push(`在 ${(holdout.tasks || []).length} 个没参与进化的任务上${move(holdout.means.seed, holdout.means.final, holdout.decision?.interval)}`);
        }
        if (sealed && isNum(sealed.means?.seed) && isNum(sealed.means?.final)) {
          checks.push(`在更晚、谁都没见过的时期上${move(sealed.means.seed, sealed.means.final, sealed.decision?.interval)}`);
        }
        next = checks.length ? `提升不只出现在训练任务上。最终版本和起点版本又各比了一次：${checks.join("；")}。`
          : "采纳是在训练任务上决定的；最终版本还要在没参与进化的任务和更晚的时期上各验证一次（见下面的「独立验证」）。";
      }

      const read = "成绩是预报误差比朴素预报低多少：" + (isNum(finalScore) && Math.abs(finalScore) >= 0.005
        ? `比如${finished ? "最终" : "在任"}版本的 ${signed(finalScore, 3)}，大约是误差比朴素预报${finalScore > 0 ? "低" : "高"} ${Math.round(Math.abs(finalScore) * 100)}%；` : "")
        + "0 是和朴素预报一样好，负数是更差。朴素预报是照搬现在的值，或上一个周期同一时刻的值（取更准的一个）。所有成绩都在研究智能体看不到的“隐藏期”上计算。";

      return h(Card, { title: "结论" }, h("div", { className: "eg-summary" },
        h("div", { className: "eg-summary-head" }, headline),
        h("p", null, h("b", null, live ? "到目前为止　" : "结果　"), result),
        next ? h("p", null, h("b", null, "这说明什么　"), next) : null,
        h(Details, { summary: "背景：这次在做什么、从哪里开始、成绩怎么读" },
          h("p", null, h("b", null, "在做什么　"), what),
          h("p", null, h("b", null, "从哪里开始　"), ...start),
          h("p", { className: "eg-read" }, h("b", null, "怎么读成绩　"), read))));
    }

    const S = { width: 960, height: 290, left: 54, right: 184, top: 26, bottom: 46 };
    const REFERENCE_NAMES = { expert: "专家模型", recipe_seed: "recipe 种子" };

    /**
     * The run's score over its generations: a flat line for the incumbent (a step up is an adoption), a dot per
     * candidate with the interval of its gain drawn around the line, the earlier runs that evolved the starting
     * version (dashed) and the host's best built-in method (dotted).
     */
    function ScoreChart({ view, catalog, onPick }) {
      const rounds = lastAttempts(view);
      const tasks = view.suite?.train || [];
      const level = versionLevels(view);
      const before = [];
      for (const step of view.ancestry || []) {
        if (step.suite !== view.suite?.id || !isNum(step.before) || !isNum(step.after)) break;
        before.unshift(step);
      }
      const columns = [...before.map((step) => ({ kind: "before", step, weight: 0.8 })),
        ...rounds.map((round) => ({ kind: "round", round, weight: 1 })), { kind: "end", weight: 0.5 }];
      let carried = null;
      for (const column of columns) {
        if (column.kind === "before") continue;
        column.digest = column.kind === "end" ? view.incumbent : column.round.incumbent;
        const value = level(column.digest);
        column.level = isNum(value) ? value : carried;
        if (isNum(column.level)) carried = column.level;
      }
      const placed = columns.filter((column) => column.kind !== "before" && isNum(column.level));
      if (!placed.length) return null;
      const byCandidate = new Map(comparisons(view).map((item) => [item.candidate, item]));
      for (const column of columns) {
        if (column.kind !== "round" || !isNum(column.level)) continue;
        column.points = (column.round.candidates || []).filter((candidate) => byCandidate.has(candidate)).map((candidate) => byCandidate.get(candidate));
      }
      let reference = null;
      const names = new Set(tasks.flatMap((task) => Object.keys(view.task_references?.[task] || {})));
      for (const name of names) {
        if (name === "naive") continue;
        const values = tasks.map((task) => view.task_references?.[task]?.[name]);
        if (values.length && values.every(isNum) && (!reference || mean(values) > reference.value)) reference = { name, value: mean(values) };
      }
      const values = [...placed.map((column) => column.level), ...before.map((step) => step.before), ...(reference ? [reference.value] : [])];
      for (const column of columns) {
        for (const point of column.points || []) values.push(column.level + point.decision.interval[0], column.level + point.decision.interval[1]);
      }
      let lo = Math.min(...values);
      let hi = Math.max(...values);
      const pad = Math.max((hi - lo) * 0.08, 0.004);
      lo -= pad;
      hi += pad;
      const { ticks, digits } = niceTicks(lo, hi, 4);
      const plotH = S.height - S.top - S.bottom;
      const y = (value) => S.top + (1 - (value - lo) / (hi - lo)) * plotH;
      const total = columns.reduce((sum, column) => sum + column.weight, 0);
      const width = Math.max(S.width, S.left + S.right + Math.round(total * 84)); // a long run: wider, and it scrolls
      const plotW = width - S.left - S.right;
      let cursor = S.left;
      for (const column of columns) {
        column.x0 = cursor;
        column.x1 = cursor + (column.weight / total) * plotW;
        column.cx = (column.x0 + column.x1) / 2;
        cursor = column.x1;
      }
      const end = columns[columns.length - 1];
      const first = columns.find((column) => column.kind !== "before" && isNum(column.level));
      for (const column of columns) { // a candidate's dot: its gain above or below the incumbent's line
        const points = column.points || [];
        const room = Math.min(30, (column.x1 - column.x0 - 40) / Math.max(1, points.length));
        points.forEach((point, i) => {
          point.x = column.cx + 10 + (i - (points.length - 1) / 2) * room;
          point.y = y(column.level + point.decision.delta);
        });
      }
      const parts = [];
      ticks.forEach((tick) => parts.push(h("line", { key: `g${tick}`, className: "grid", x1: S.left, x2: end.x1, y1: y(tick), y2: y(tick) }),
        h("text", { key: `t${tick}`, x: S.left - 8, y: y(tick) + 4, textAnchor: "end" }, signed(tick, digits))));
      parts.push(h("text", { key: "axis", x: 4, y: 12 }, "成绩（训练任务平均）"));
      if (lo < 0 && hi > 0) {
        parts.push(h("line", { key: "zero", className: "zero", x1: S.left, x2: end.x1, y1: y(0), y2: y(0) }),
          h("text", { key: "zero-t", x: S.left + 4, y: y(0) - 4 }, "0 = 朴素预报"));
      }
      const labels = [];
      if (reference) {
        parts.push(h("line", { key: "ref", className: "ref", x1: S.left, x2: end.x1, y1: y(reference.value), y2: y(reference.value) },
          h("title", null, `宿主内置方法 ${reference.name}${reference.name === "expert" ? "（2026-10 研究中人工设计的专家模型）" : ""}在同一批起点上的成绩（各训练任务平均），只作对照`)));
        labels.push({ key: "ref", y: y(reference.value), text: `${REFERENCE_NAMES[reference.name] || reference.name} ${signed(reference.value, 3)}`, className: "ref-t" });
      }
      // what came before this run: the earlier runs' starting scores, rising to this run's start
      before.forEach((step, index) => {
        const column = columns[index];
        const next = index + 1 < before.length ? { x: columns[index + 1].cx, y: y(before[index + 1].before) } : { x: first.x0, y: y(first.level) };
        parts.push(h("path", { key: `b${index}`, className: "before", d: `M${column.cx},${y(step.before)} L${next.x},${next.y}` },
          h("title", null, `${day(step.started_at)} 那次运行（${step.run_id}）第 ${step.generation} 代采纳候选 ${step.name}：训练任务上的成绩 ${signed(step.before, 3)} → ${signed(step.after, 3)}`)),
        h("circle", { key: `bc${index}`, className: "before-dot", cx: column.cx, cy: y(step.before), r: 4 }),
        h("text", { key: `bt${index}`, x: column.cx, y: y(step.before) + 17, textAnchor: "middle" }, signed(step.before, 3)),
        h("text", { key: `bl${index}`, x: (column.cx + next.x) / 2 - 8, y: (y(step.before) + next.y) / 2 + 4, textAnchor: "end", className: "note" },
          `采纳 g${step.generation}${step.name}`));
      });
      // the incumbent: one flat segment per generation; an adoption steps up to the next segment
      let previous = null;
      columns.forEach((column, index) => {
        if (column.kind === "before" || !isNum(column.level)) return;
        const alias = versionAlias(view, column.digest);
        const changed = !previous || previous.digest !== column.digest;
        parts.push(h("line", { key: `s${index}`, className: "trunk", x1: column.x0, x2: column.x1, y1: y(column.level), y2: y(column.level),
          onClick: column.kind === "round" ? () => onPick({ generation: column.round.generation }) : undefined },
        h("title", null, `${column.kind === "end" ? (view.status === "finished" ? "最终版本" : "现在在任的版本") : `第 ${column.round.generation} 代的在任版本`}：`
          + `${alias === "起点" ? `起点（${seedOrigin(view)}）` : alias} ${short(column.digest)}\n成绩 ${signed(column.level, 3)}：它作为在任版本的全部回合在各训练任务上的平均`)));
        if (changed && column.kind === "round") { // after an adoption only the name: the column's candidates sit beside it
          parts.push(h("text", { key: `sl${index}`, x: column.x0 + 4, y: y(column.level) - 7, className: "strong" },
            column.round.generation ? alias : `${alias} ${signed(column.level, 3)}`));
        }
        if (previous && previous.digest !== column.digest) {
          const chosen = (previous.points || []).find((point) => point.candidate.digest === column.digest);
          if (chosen) {
            parts.push(h("path", { key: `a${index}`, className: "adopt", d: `M${chosen.x},${chosen.y} L${column.x0},${y(column.level)}` },
              h("title", null, `采纳 g${previous.round.generation}${chosen.candidate.name}：从下一代起由它做在任版本`)));
          }
        }
        previous = column;
      });
      labels.push({ key: "end", y: y(end.level), className: "strong",
        text: `${view.status === "finished" ? "最终" : "现在"}：${end.digest === view.seed?.digest ? "仍是起点" : versionAlias(view, end.digest)} ${signed(end.level, 3)}` });
      // candidates: a dot at incumbent + gain, the interval of the gain around the incumbent's line
      columns.forEach((column, index) => {
        (column.points || []).forEach((point) => {
          const { candidate, decision, side, before: was, after: now } = point;
          const top = y(column.level + decision.interval[1]);
          const bottom = y(column.level + decision.interval[0]);
          const name = `g${column.round.generation}${candidate.name}`;
          parts.push(h("g", { key: `c${index}-${candidate.name}`, className: "cand", "data-side": side, "data-adopted": String(candidate.status === "adopted"),
            onClick: () => onPick({ generation: column.round.generation, name: candidate.name }) },
          h("title", null, `${name} · ${(CAND_STATUS[candidate.status] || [candidate.status])[0]}\n成绩 ${signed(now, 3)}，这一代的在任版本 ${signed(was, 3)}`
            + `\n差 ${signed(decision.delta, 4)}，误差范围 [${signed(decision.interval[0], 3)}, ${signed(decision.interval[1], 3)}]：${CREDIBLE_TEXT[side]}`
            + `\n${reasonText(catalog, decision.reason)}${candidate.summary ? `\n${candidate.summary}` : ""}`),
          h("line", { x1: point.x, x2: point.x, y1: top, y2: bottom }),
          h("line", { className: "cap", x1: point.x - 4, x2: point.x + 4, y1: top, y2: top }),
          h("line", { className: "cap", x1: point.x - 4, x2: point.x + 4, y1: bottom, y2: bottom }),
          h("circle", { cx: point.x, cy: point.y, r: candidate.status === "adopted" ? 6 : 4.5 }),
          // the name above the bar, or below it when the bar's top would sit on the incumbent's line
          h("text", { x: point.x, y: Math.abs(top - y(column.level)) < 14 ? bottom + 14 : top - 6, textAnchor: "middle" }, name)));
        });
      });
      // the right margin: the final score and the reference, kept apart
      labels.sort((a, b) => a.y - b.y);
      for (let i = 1; i < labels.length; i += 1) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 16);
      labels.forEach((label) => parts.push(h("text", { key: `m${label.key}`, x: end.x1 + 8, y: label.y + 4, className: label.className }, label.text)));
      columns.forEach((column, index) => {
        const head = column.kind === "before" ? `此前 ${day(column.step.started_at)}` : column.kind === "end" ? "结果" : `第 ${column.round.generation} 代`;
        const sub = column.kind === "before" ? `运行 ${runShort(column.step.run_id)}` : column.kind === "end" ? "" : roundOutcome(view, column.round)[0];
        parts.push(h("text", { key: `x${index}`, x: column.cx, y: S.height - S.bottom + 20, textAnchor: "middle", className: "strong" }, head));
        if (sub) {
          parts.push(h("text", { key: `xs${index}`, x: column.cx, y: S.height - S.bottom + 36, textAnchor: "middle" },
            h("title", null, sub), fit(sub, column.x1 - column.x0 - 6, 11.5)));
        }
        if (index && column.kind !== columns[index - 1].kind && columns[index - 1].kind === "before") {
          parts.push(h("line", { key: `sep${index}`, className: "sep", x1: column.x0, x2: column.x0, y1: S.top - 10, y2: S.height - S.bottom + 4 }));
        }
      });
      return h("svg", { viewBox: `0 0 ${width} ${S.height}`, role: "img", "aria-label": "各代成绩的变化",
        style: { minWidth: `${Math.round(width * 0.8)}px` } }, parts);
    }

    function ScoreCard({ view, catalog, onPick }) {
      const chart = h(ScoreChart, { view, catalog, onPick });
      const level = versionLevels(view);
      if (!lastAttempts(view).some((round) => isNum(level(round.incumbent)))) return null; // nothing scored yet
      const icon = (body) => h("svg", { width: 28, height: 14, "aria-hidden": "true" }, body);
      return h(Card, { title: "成绩怎样变化", hint: "横线是在任版本（当前在用的版本）的成绩：平着表示没有换版本，往上跳一次就是换上了更好的版本。每个圆点是一个候选，竖线是它和在任版本之差的误差范围。点一个候选，看那一代的过程。" },
        h("div", { className: "eg-chart" }, chart),
        h("div", { className: "eg-legend eg-chart" },
          h("span", null, icon(h("line", { className: "trunk", x1: 1, x2: 27, y1: 7, y2: 7 })), "在任版本的成绩"),
          h("span", null, icon(h("g", { className: "cand", "data-side": "above" }, h("line", { x1: 14, x2: 14, y1: 1, y2: 13 }), h("circle", { cx: 14, cy: 7, r: 4 }))),
            "候选：整段竖线在横线上方，才算可信地更好"),
          h("span", null, icon(h("g", { className: "cand", "data-side": "noise" }, h("line", { x1: 14, x2: 14, y1: 1, y2: 13 }), h("circle", { cx: 14, cy: 7, r: 4 }))),
            "竖线跨过横线：差别在误差内，分不清好坏"),
          h("span", null, icon(h("g", { className: "cand", "data-side": "below" }, h("line", { x1: 14, x2: 14, y1: 1, y2: 13 }), h("circle", { cx: 14, cy: 7, r: 4 }))),
            "整段在下方：可信地更差"),
          h("span", null, icon(h("line", { className: "adopt", x1: 1, x2: 27, y1: 12, y2: 2 })), "采纳"),
          (view.ancestry || []).length ? h("span", null, icon(h("line", { className: "before", x1: 1, x2: 27, y1: 12, y2: 2 })), "此前的运行（起点是怎么来的）") : null,
          h("span", null, icon(h("line", { className: "ref", x1: 1, x2: 27, y1: 7, y2: 7 })), "对照：宿主内置的预报方法（不参与进化）")));
    }

    function HarnessCard({ view }) {
      const [open, setOpen] = useState(false);
      const [tab, setTab] = useState("diff");
      const changed = view.incumbent && view.incumbent !== view.seed?.digest;
      const { data: harness, error } = useResource(open ? Api.harness(view.incumbent) : null);
      const { data: diff } = useResource(open && changed ? Api.harnessDiff(view.seed.digest, view.incumbent) : null);
      const tabs = [["diff", "与起点的差异"], ["instructions", "指令"], ["lessons", "经验"], ["agent", "DSH 设置"], ["files", "文件"]];
      const texts = harness?.texts || {};
      let body = null;
      if (error) body = h(Alert, { tone: "warning" }, `读不到这个版本：${error.message}`);
      else if (!harness) body = hint("正在读取…");
      else if (tab === "diff") body = !changed ? hint("在任版本就是起点版本。") : !diff ? hint("正在读取…")
        : diff.length ? diff.map((item) => h(Diff, { key: item.path, patch: item.patch || `${item.status} ${item.path}` })) : hint("没有差异。");
      else if (tab === "instructions") body = h(Markdown, { text: texts["instructions.md"] || "（没有 instructions.md）" });
      else if (tab === "lessons") body = h(Markdown, { text: texts["memory/lessons.md"] || "（还没有经验记录）" });
      else if (tab === "agent") body = h("pre", { className: "eg-pre" }, texts["harness.toml"] || JSON.stringify(harness.agent || {}, null, 2));
      else body = h("div", { className: "eg-chips" }, (harness.files || []).map((file) => h("span", { key: file, className: "eg-chip eg-mono" }, file)));
      return h(Card, { title: "在任 Harness", hint: `${versionAlias(view, view.incumbent)} ${short(view.incumbent)}：智能体的指令、技能、代码库模板、经验和 DSH 设置。`,
        actions: h(Button, { onClick: () => setOpen(!open) }, open ? "收起" : "查看内容") },
      open ? h(Fragment, null,
        h("div", { className: "eg-tabs", style: { margin: "0 0 10px" } }, tabs.map(([id, label]) => h("button", {
          key: id, className: "eg-tab", "aria-current": tab === id ? "page" : undefined, onClick: () => setTab(id) }, label))),
        body) : null);
    }

    /** The run's settings; advice for its agents is written on the 专家 page (one form for every run). */
    function ConfigCard({ view, catalog, go }) {
      const config = view.config || {};
      return h(Card, { title: "运行配置", actions: h(Button, { onClick: () => go({ tab: "expert", page: null }) }, "写专家建议") },
        h(KV, { pairs: [
          ["代数 × 每代候选", `${config.generations} × ${config.candidates}`],
          ["每个版本每个任务的回合", String(config.repeats ?? "–")],
          ["回合预算", `${config.minutes} 分钟，${Number(config.tokens || 0).toLocaleString()} 计费 token`],
          ["内循环模型", config.model || "由 Harness 决定"],
          ["评审智能体", `${config.evolver_model}，${config.evolver_minutes} 分钟，${Number(config.evolver_tokens || 0).toLocaleString()} 计费 token`],
          ["并发 / 重试", `${config.concurrency} / ${config.retries}`],
          ["模型服务故障时", `${!isNum(config.outage_minutes) ? "–" : config.outage_minutes > 0 ? `最多等 ${config.outage_minutes} 分钟` : "不等"}`
            + `${config.auto_resume ? "；停下后后端自动继续" : ""}`],
          ["采纳后自动复核", config.check_adoptions ? "每次采纳后在训练任务的封存时期 validation 上复核" : "不复核（可以在「概览」的独立验证里补做）"],
          ["采纳门槛", gateText(view.gates)],
          ["训练任务", (view.suite?.train || []).map((task) => taskShort(catalog, task)).join("、")],
          ["任务组", h(Mono, null, config.suite || "")],
          ["ecogym", view.ecogym || "–"],
        ] }),
        h("p", { className: "eg-hint", style: { marginBottom: 0 } }, "专家建议写进账本后，之后开始的回合会在 data/expert.md 里读到，下一代的评审智能体也会看到。"));
    }

    /* ---- new evolution -------------------------------------------------------------------------------------- */

    const SAVED = "ecogym.new-evolution";

    /** The critical count of a signed-rank suite for k candidates a round (−1: the test cannot pass); null otherwise. */
    function rankCritical(suite, k) {
      const value = suite?.signed_rank_critical?.[String(Math.min(6, Math.max(1, Number(k) || 1)))];
      return Number.isInteger(value) ? value : null;
    }

    /** A signed-rank suite's gate in words for k candidates a round; "" for a t-interval suite or one that cannot pass. */
    function rankRule(suite, k) {
      const c = rankCritical(suite, k);
      if (c === null || c < 0) return "";
      const n = suite.train.length;
      return c === 0 ? `${n} 个训练任务全部更好`
        : `${n} 个训练任务全部更好，或者变差的只是变化最小的一两个（把各任务的变化按大小从小到大排名，变差任务的名次加起来不超过 ${c}）`;
    }

    /** What a suite's design means for the adoption gate, with k candidates a round and r episodes per version and task. */
    function designText(suite, k, r) {
      const rule = rankRule(suite, k);
      const repeats = Number(r) || suite.repeats || 2;
      const head = `按每个版本每个训练任务 ${suite.repeats ?? 2} 个回合设计。`;
      if (rule) {
        return `${head}采纳看各训练任务是否一致地变好（符号秩检验）：${k || 1} 个候选时，${rule}，平均增益也超过门槛，才算可信。`
          + "它只看各任务是变好还是变差、变化大小排第几，所以一个任务上特别大的增益不会单独决定结果；反过来，一个任务的变差比别处的增益还明显时就通不过。"
          + (repeats >= 2 ? `每个版本每个任务有 ${repeats} 个回合，所以还要没有任务可信地变差超过 ${plain(suite.gates?.task_cap ?? 0.03, 2)}（单任务否决）。`
            : "每个版本每个任务只有 1 个回合时，单任务否决不起作用。");
      }
      if (rankCritical(suite, k) !== null) return `${head}${k || 1} 个候选时，这个任务组的训练任务太少，符号秩检验通不过；请减少每代的候选数。`;
      if (suite.train.length < 3) return `${head}训练任务少于 3 个时，区间只反映任务内的随机差异，算不上真正的检验。`;
      return `${head}采纳看各训练任务增益的 t 区间：区间下限大于 0、平均增益也超过门槛，才算可信。在一个已经不错的版本附近，`
        + "各任务增益的差别主要来自任务本身，所以训练任务越多，越小的增益也能确认；同一任务多跑几次帮助不大。";
    }
    function NewEvolution({ go, catalog, catalogError }) {
      if (!catalog) return h(CatalogPending, { error: catalogError, text: "正在读取任务组…" });
      return h(NewEvolutionForm, { go, catalog });
    }

    function NewEvolutionForm({ go, catalog }) {
      const suites = catalog?.suites || [];
      const defaults = catalog?.evolve_defaults || {};
      const routes = useModelRoutes(catalog);
      const recommended = suites.find((item) => item.recommended) || suites[0];
      const [values, setValues] = useState(() => {
        let saved = {};
        try {
          saved = JSON.parse(localStorage.getItem(SAVED) || "{}") || {};
        } catch {
          saved = {};
        }
        const chosen = suites.find((item) => item.id === saved.suite) || recommended;
        const start = { suite: chosen?.id || "", label: "",
          ...Object.fromEntries(Object.keys(defaults).map((key) => [key, saved[key] ?? defaults[key] ?? ""])) };
        // the episodes per task the suite is designed for, unless the saved form was for this suite
        if (chosen?.repeats && !(saved.suite === chosen.id && saved.repeats)) start.repeats = chosen.repeats;
        return { ...start, model: modelRef(start.model), evolver_model: modelRef(start.evolver_model) };
      });
      const [busy, setBusy] = useState(false);
      const [problem, setProblem] = useState("");
      const set = (key) => (event) => {
        const value = event.target.value;
        setValues((old) => {
          const next = { ...old, [key]: value };
          const picked = key === "suite" ? suites.find((item) => item.id === value) : null;
          if (picked?.repeats) next.repeats = picked.repeats; // another suite: its own design
          return next;
        });
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
          if (typeof defaults[key] === "boolean") {
            body[key] = values[key] === true || values[key] === "true";
            continue;
          }
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
      const checked = (key) => values[key] === true || values[key] === "true";
      const check = (key, label) => h("label", { className: "eg-check", key },
        h("input", { type: "checkbox", checked: checked(key), onChange: (event) => setValues((old) => ({ ...old, [key]: event.target.checked })) }), label);
      const estimate = `最多 ${episodes} 个回合和 ${g} 次评审会话，约 ${hours.toFixed(1)} 小时，计费 token 上限约 ${(budget / 1e6).toFixed(0)}M。`;
      return h("form", { onSubmit: submit },
        h("button", { type: "button", className: "eg-back", onClick: () => go({ page: null }) }, "← 全部进化运行"),
        h(PageHead, { title: "新建 Harness 进化", text: "第 0 代先让起点 Harness 在每个训练任务上跑一遍，作为基线。之后每一代，评审智能体读完成绩和过程，改出几个候选；候选和在任版本在每个训练任务上同时跑，只有可信地更好才换上。运行在后台进行，关闭页面不影响。" }),
        h(Card, { title: "1. 进化什么", hint: suite ? `训练任务 ${n} 个：${suite.train.map((task) => taskShort(catalog, task)).join("、")}；留出任务 ${suite.holdout_tasks} 个，只在最后验证时运行。` : "没有可用的任务组。" },
          h("div", { className: "eg-form" },
            h("label", { className: "eg-field", style: { gridColumn: "span 2" } }, h("span", null, "任务组"),
              h("select", { className: "eg-input", value: values.suite, onChange: set("suite") }, suites.map((item) => h("option", { key: item.id, value: item.id },
                `${item.title}（${item.id}）${item.recommended ? " · 推荐" : ""}`)))),
            field("harness", "起点 Harness", { autoComplete: "off" }, "seed（自带的初始版本），或以前采纳的版本的摘要前缀"),
            h("label", { className: "eg-field", style: { gridColumn: "1 / -1" } }, h("span", null, "标签（可选）"),
              h("input", { className: "eg-input", maxLength: 200, value: values.label, onChange: set("label"), placeholder: "说明这次运行，例如：温室任务组，10 代 × 2 个候选，低负载" }))),
          suite ? h(Details, { summary: "这个任务组怎样决定采纳" }, hint(designText(suite, k, r)),
            suite.notes ? h("p", { className: "eg-hint", style: { whiteSpace: "pre-wrap" } }, suite.notes) : null) : null),
        h(Card, { title: "2. 进化多久、多快" },
          h("div", { className: "eg-form" },
            field("generations", "代数", { type: "number", min: 1, max: 30, step: 1 }, "代数越多，找到改进的机会越多"),
            field("candidates", "每代候选数", { type: "number", min: 1, max: 6, step: 1 }, "多个候选时门槛按 Bonferroni 加严")),
          h("div", { className: "eg-field", style: { marginTop: "14px" } }, h("span", null, "电脑负载：同时运行几个回合"),
            h("div", { className: "eg-actions" }, LOADS.map(([count, name, note]) => h(Button, { key: count, variant: c === count ? "primary" : "outline",
              onClick: () => setValues((old) => ({ ...old, concurrency: String(count) })) }, `${name}（${count} 个）${note ? ` · ${note}` : ""}`))),
            h("small", null, "每个回合都在本机运行智能体的实验和程序评分。同时运行越多，结束得越快、电脑也越热；模型调用在提供方，不占本机。")),
          h("p", { className: "eg-hint", style: { marginBottom: 0 } }, estimate)),
        h(Card, { title: "3. 用哪个模型", hint: modelsHint },
          h("div", { className: "eg-form", style: { gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))" } },
            h(ModelSelect, { label: "研究智能体（解任务）", value: values.model, onChange: set("model"), routes,
              empty: `由 Harness 决定（起点版本用 ${catalog?.harness_model || "DeepSeek"}）`,
              note: "固定后，每个回合都用这个模型；不固定时由各版本 Harness 的 [agent] 决定（只能是 DeepSeek 模型）。" }),
            h(ModelSelect, { label: "评审智能体（改 Harness）", value: values.evolver_model, onChange: set("evolver_model"), routes,
              note: "每一代读证据、写候选的评审会话用这个模型。" })),
          catalog?.models?.error ? h(Alert, { tone: "warning" }, `读取 DSH 的模型设置失败：${catalog.models.error}`) : null),
        h(Card, null, h(Details, { summary: "高级设置：回合与评审的预算、服务故障时怎么办、复核" },
          h("div", { className: "eg-form", style: { marginTop: "10px" } },
            field("repeats", "每个版本每个任务的回合数", { type: "number", min: 1, max: 5, step: 1 }, "多跑几次能看清回合间的随机差异"),
            field("minutes", "每个回合的分钟数", { type: "number", min: 1, step: 1 }, "按机器醒着的时间计"),
            field("tokens", "每个回合的计费 token", { type: "number", min: 10000, step: 10000 }, "命中缓存的输入 token 按 1/10 计"),
            field("concurrency", "同时运行的回合", { type: "number", min: 1, max: 16, step: 1 }),
            field("evolver_minutes", "每代评审分钟数", { type: "number", min: 1, step: 1 }),
            field("evolver_tokens", "每代评审计费 token", { type: "number", min: 10000, step: 10000 }),
            field("outage_minutes", "模型服务故障时最多等几分钟", { type: "number", min: 0, step: 10 }, "测试请求也没有回复时不开始新回合、每分钟试一次；等满还没恢复，运行停下；0 = 不等")),
          h("div", { className: "eg-checks" },
            check("auto_resume", "自动继续：服务恢复后，或后端（DSH）重启后，后端自己接着运行"),
            check("check_adoptions", "每次采纳后，在训练任务更晚的封存时期上复核一次（只重评程序，评审看不到结果）")))),
        problem ? h(Alert, { tone: "danger" }, problem) : null,
        h("div", { className: "eg-actions", style: { marginTop: "16px" } },
          h(Button, { variant: "primary", disabled: busy, onClick: () => submit() }, busy ? "正在启动…" : "开始进化"),
          h(Button, { onClick: () => go({ page: null }) }, "取消"),
          h("span", { className: "eg-hint" }, `${estimate}${checked("auto_resume") ? "停下后会自动继续。" : ""}`)));
    }

    const LOADS = [[2, "低", "适合笔记本、长时间运行"], [3, "中", ""], [6, "高", "最快、最热"]];

    /* ---- episodes ------------------------------------------------------------------------------------------ */

    function EpisodeList({ go, catalog }) {
      const { data, error } = useResource(Api.episodes, { interval: 8000 });
      const items = data || [];
      return h(Fragment, null,
        h(PageHead, { title: "回合", text: "一个回合是 DSH 智能体在隔离工作区里独立解一个任务的完整过程：读数据、查资料、写代码、做实验、提交预测程序，再由宿主在隐藏时期上逐时刻打分。完整轨迹都会保存。" }),
        error ? h(Alert, { tone: "danger" }, `读取失败：${error.message}`) : null,
        data && !items.length ? h(Empty, { title: "还没有回合", text: "进化运行和验证会产生回合，也可以用 ecogym episode run 单独运行。" }) : null,
        items.length ? h(Card, null, h(Table, { head: ["任务", "来源", "状态", ["隐藏期总分", "num"], ["本地最好", "num"], "模型", ["Token", "num"], "开始"] },
          items.map((item) => h("tr", { key: item.episode, className: "eg-click", onClick: () => go({ page: "episode", id: item.episode }) },
            h("td", null, h(TaskLabel, { catalog, id: item.task_id })),
            h("td", null, item.context?.role ? `${roleText(item.context)}${item.context.generation !== undefined ? ` · 第 ${item.context.generation} 代` : ""}` : "单独运行"),
            h("td", null, h(StatusTag, { map: EP_STATUS, keyName: epKey(item) })),
            h("td", { className: `num ${toneClass(item.score, 4)}` }, signed(item.score)),
            h("td", { className: "num" }, signed(item.best_local)),
            h("td", null, item.model || "–"),
            h("td", { className: "num" }, tokens(item.tokens)),
            h("td", null, when(item.started_at)))))) : null);
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

    function EpisodePage({ id, go, catalog, back }) {
      const [running, setRunning] = useState(true);
      const { data: record, error } = useResource(Api.episode(id), { interval: running ? 5000 : 0 });
      useEffect(() => {
        if (record) setRunning(epKey(record) === "running");
      }, [record]);
      useTick(running && Boolean(record));
      if (error && !record) return h(Alert, { tone: "danger" }, `无法读取回合 ${id}：${error.message}`);
      if (!record) return hint("正在读取回合…");
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
        h("button", { className: "eg-back", onClick: () => go(back || { page: null }) },
          back?.focus ? `← 返回运行的第 ${back.focus.generation} 代` : "← 返回"),
        h("div", { className: "eg-head" }, h("div", { className: "eg-head-main" },
          h("div", { className: "eg-title-line" }, h("h1", null, record.label || record.episode), h(StatusTag, { map: EP_STATUS, keyName: epKey(record) })),
          h("div", { className: "eg-meta" },
            h("span", null, h("b", null, "任务"), taskShort(catalog, record.task_id)),
            h("span", null, h("b", null, "Harness"), h(Mono, null, short(record.harness?.digest)), ` ${record.harness?.name || ""}`),
            h("span", null, h("b", null, "模型"), record.model || "–"),
            h("span", null, h("b", null, "开始"), when(record.started_at, true)),
            context.evolution ? h("span", null, h("b", null, "来自"),
              h(Link, { onClick: () => go({ tab: "evolutions", page: "evolution", id: context.evolution }) }, context.evolution), ` ${roleText(context)}`) : null)),
        h("div", { className: "eg-actions" }, h(Button, { onClick: () => copy(record.episode) }, "复制 ID"))),
        record.status === "infra_failed" && finished.stalled ? h(Alert, { tone: "danger" }, h("b", null, "模型服务故障拖住了智能体："),
          `失败的模型调用让智能体有 ${stalledFor(finished)}没法工作（本回合共 ${isNum(record.minutes) ? `${record.minutes * 60} 秒` : "几分钟"}；`
          + "包括调用一直失败、会话提前结束后剩下的时间），超过一半，所以记为服务故障：成绩照常给出，但不计为 Harness 的成绩；"
          + `进化运行还有重试次数时会重跑这个回合。${finished.upstream_error ? `最后一次失败：${finished.upstream_error}` : ""}`)
          : record.status === "infra_failed" ? h(Alert, { tone: "danger" }, h("b", null, "模型服务没有回复："),
            finished.upstream_error || "没有一次模型调用得到回复（服务故障，不计为 Harness 的成绩）") : null,
        scored?.errors?.length ? h(Alert, { tone: "warning" }, h("b", null, "程序在隐藏期上的错误："), scored.errors.slice(0, 3).join("；")) : null,
        h("div", { className: "eg-stats" },
          h(Stat, { label: "隐藏期总分", value: scored ? h("span", { className: toneClass(scored.score, 4) }, signed(scored.score)) : "–",
            sub: scored ? `95% 区间 [${signed(scored.interval?.[0])}, ${signed(scored.interval?.[1])}] · ${scored.origins} 个起点` : "运行结束后计算" }),
          h(Stat, { label: "最好的本地成绩", value: Number.isFinite(best) ? signed(best) : "–", sub: isNum(record.start?.local?.score)
            ? `已接受的提交；起始程序 ${signed(record.start.local.score)}` : "本地检验期，已接受的提交" }),
          h(Stat, { label: "提交 / 检验", value: `${(record.submissions || []).length} / ${(record.evaluations || []).length}`, sub: scored?.auto ? "没有被接受的提交：按当时的程序评分"
            : scored?.start ? `评分用起始程序：没有提交在本地比它高 ${record.replace_margin ?? ""} 以上` : scored?.n ? `评分用第 ${scored.n} 次提交` : "" }),
          h(Stat, { label: "笔记 / 问题", value: `${(record.notes || []).length} / ${(record.questions || []).length}` }),
          h(Stat, { label: "Token", value: tokens(finished.tokens), sub: finished.llm_calls ? `${finished.llm_calls} 次模型调用` : "" }),
          h(Stat, { label: "用时", value: duration(finished.seconds), sub: finished.status
            ? `状态 ${finished.status}${finished.asleep_seconds >= 60 ? `；其中休眠 ${duration(finished.asleep_seconds)}，不计入时限` : ""}` : "进行中" })),
        scored ? h(Card, { title: "隐藏期逐格技能", hint: `宿主用隐藏时期之前的全部数据拟合提交的程序，再逐个时间步送入新观测、在每个计分起点要求预报；技能相对朴素基线，大于 0 更好。同一批起点上的参考：${refs || "–"}` },
          h(CellMatrix, { task, cells: scored.cells })) : null,
        timeline.length ? h(Card, { title: "时间线", hint: "智能体在本地检验期上的检验和提交（隐藏期成绩只在结束后计算）。" },
          h(Table, { head: [["分钟", "num"], "动作", ["本地总分", "num"], "状态", "说明"] }, timeline.map(([action, item], index) => h("tr", { key: index },
            h("td", { className: "num" }, start !== null && timeOf(item.at) !== null ? ((timeOf(item.at) - start) / 60000).toFixed(1) : "–"),
            h("td", null, action, action === "提交" ? h(Fragment, null, " ", !item.accepted ? h(Tag, { tone: "danger" }, "未接受")
              : item.replaces_start === false ? h(Tag, { tone: "neutral" }, "已接受，不替换模板") : h(Tag, { tone: "success" }, "已接受")) : null),
            h("td", { className: `num ${toneClass(item.local?.score, 4)}` }, signed(item.local?.score)),
            h("td", null, item.local?.status || ""),
            h("td", null, item.note || ""))))) : null,
        (record.notes || []).length ? h(Card, { title: "智能体的笔记", hint: "回合中记下的发现；导出训练数据时进入经验库。" },
          record.notes.map((note, index) => h("div", { key: index, className: "eg-note" }, h("span", { className: "eg-chip" }, note.kind), " ", note.text,
            (note.sources || []).length ? h("div", { className: "eg-faint" }, `来源：${note.sources.join("；")}`) : null))) : null,
        (record.questions || []).length ? h(Card, { title: "给专家的问题" }, record.questions.map((item) => h("div", { key: item.id, className: "eg-note" }, item.question))) : null,
        finished.final_text ? h(Card, { title: "智能体的最终回复" }, h(Markdown, { text: finished.final_text })) : null,
        Object.keys(activity.tool_calls || {}).length ? h(Card, { title: "工具使用" },
          h("div", { className: "eg-chips" }, Object.entries(activity.tool_calls).map(([name, count]) => h("span", { key: name, className: "eg-chip" }, `${name.replace(/^mcp__ecogym__/, "ecogym.")} ${count}`))),
          (activity.web_queries || []).length ? hint(`网页搜索：${activity.web_queries.slice(0, 20).join("；")}`) : null,
          hint(`子智能体 ${activity.subagents || 0} 个 · 轨迹 `, h(Mono, null, finished.trajectory?.dir || "–"))) : null,
        h(TranscriptCard, { record }));
    }

    /* ---- sessions: an agent's steps; the evolver's page ------------------------------------------------------ */

    const TOOL_NAMES = { read: "读文件", bash: "命令行", edit: "编辑", multi_edit: "编辑", write: "写文件", grep: "搜索", glob: "找文件",
      ls: "列目录", skill: "技能", todo_write: "待办", job_output: "后台输出", job_kill: "停后台任务", task: "子智能体",
      web_search: "网页搜索", web_fetch: "读网页" };
    const ECOGYM_TOOLS = { check_candidate: "检查候选", screen_templates: "批量试模板", reset_candidate: "重置候选", task_info: "剩余预算", note: "记笔记",
      ask_expert: "问专家", evaluate_local: "本地检验", submit: "提交" };
    const ECOGYM_PREFIX = "mcp__ecogym__";
    const WRITE_TOOLS = ["edit", "multi_edit", "write"];

    function toolName(name) {
      const text = String(name || "");
      if (text.startsWith(ECOGYM_PREFIX)) {
        const tool = text.slice(ECOGYM_PREFIX.length);
        return { group: "ecogym", tool, label: ECOGYM_TOOLS[tool] || tool };
      }
      return { group: WRITE_TOOLS.includes(text) ? "write" : "dsh", tool: text, label: TOOL_NAMES[text] || text || "工具" };
    }

    const workspace = (text) => String(text || "").split("/workspace/work/").join("");

    /** What a call works on, for its one-line summary. */
    function callTarget(call, tool) {
      if (!call.name.startsWith(ECOGYM_PREFIX)) return workspace(call.target);
      let args = null;
      try {
        args = JSON.parse(call.target || "{}");
      } catch {
        return call.target;
      }
      if (tool === "reset_candidate") return `候选 ${args.name} ← ${args.base === "incumbent" ? "在任版本" : `存档 ${args.base}`}`;
      if (tool === "screen_templates") return `${(args.variants || []).length} 个变体：${(args.variants || []).join("、")}`;
      if (tool === "note") return `[${NOTE_KIND[args.kind] || args.kind}] ${args.text || ""}`;
      if (tool === "ask_expert") return args.question || "";
      if (args.name) return `候选 ${args.name}`;
      return Object.keys(args).length ? call.target : "";
    }

    /** One line on what an ecogym tool answered. */
    function resultBrief(tool, data) {
      if (!data || typeof data !== "object") return "";
      if (tool === "check_candidate") {
        const smoke = data.smoke || {};
        const local = isNum(smoke.score) ? ` · 模板本地 ${signed(smoke.score, 3)}${isNum(smoke.incumbent_score) ? `（在任 ${signed(smoke.incumbent_score, 3)}）` : ""}` : "";
        const warned = (data.warnings || []).length ? ` · ${data.warnings.length} 条提醒` : "";
        return `${data.ok ? "通过" : `没通过：${(CAND_STATUS[data.status] || [data.status])[0]}`}${local}${warned}`;
      }
      if (tool === "reset_candidate") return data.reset ? "已重置" : "";
      if (tool === "screen_templates") {
        const best = (data.variants || []).find((item) => isNum(item.gain));
        return best ? `最好的 ${best.name}：本地平均 ${signed(best.gain, 4)}，${best.better} 个任务更好、${best.worse} 个更差`
          : `${(data.variants || []).length} 个变体都没有正常完成`;
      }
      if (tool === "task_info") return isNum(data.minutes_left) ? `剩 ${data.minutes_left} 分钟` : "";
      if (tool === "evaluate_local") return `本地 ${signed(data.score, 3)} · ${data.status}`;
      if (tool === "submit") return `${data.accepted ? "已接受" : "未接受"} · 本地 ${signed(data.score, 3)}`;
      if (tool === "note" || tool === "ask_expert") return data.recorded ? "已记下" : "";
      return "";
    }

    function oneLine(text, limit = 160) {
      const line = String(text || "").replace(/\s+/g, " ").trim();
      return line.length > limit ? `${line.slice(0, limit)}…` : line;
    }

    function CallView({ call, result }) {
      const { group, tool, label } = toolName(call.name);
      const brief = group === "ecogym" ? resultBrief(tool, result?.data) : "";
      const failed = group === "ecogym" && result && !result.data && result.text.trim();
      return h(Details, { className: "eg-details eg-call", summary: h(Fragment, null,
        h("span", { className: "eg-kind", "data-tool": group === "dsh" ? undefined : group }, group === "ecogym" ? `ecogym · ${label}` : label),
        h("code", { title: workspace(call.target) }, callTarget(call, tool)),
        brief ? h("small", null, brief) : failed ? h("small", { className: "eg-bad" }, "出错") : !result ? h("small", null, "没有结果") : null) },
      h("div", { className: "eg-call-result" }, "调用的参数"), h("pre", { className: "eg-pre" }, workspace(call.text) || "（没有参数）"),
      result ? h(Fragment, null, h("div", { className: "eg-call-result" }, `结果（${result.text.length.toLocaleString()} 字）`),
        h("pre", { className: "eg-pre" }, workspace(result.text) || "（空）")) : null);
    }

    function Turn({ id, turn, items, results, flash }) {
      return h("div", { id, className: "eg-turn", "data-flash": String(Boolean(flash)) },
        h("div", { className: "eg-turn-no" }, turn === null ? "" : `第 ${turn} 步`),
        h("div", { className: "eg-turn-body" }, items.map((item, index) => {
          if (item.kind === "system") return h("div", { key: index, className: "eg-ctx" }, `系统提示，${item.chars.toLocaleString()} 字（不显示）`);
          if (item.kind === "context") return h("div", { key: index, className: "eg-ctx" }, `DSH 注入的上下文，${item.chars.toLocaleString()} 字：${item.text}`);
          if (item.kind === "prompt") {
            return h("div", { key: index, className: "eg-prompt" }, h("div", { className: "eg-faint" }, "宿主（或上级智能体）的消息"),
              h(Clamp, { height: 150 }, h("pre", null, workspace(item.text))));
          }
          if (item.kind === "reasoning") {
            return h(Details, { key: index, className: "eg-details eg-think", summary: h(Fragment, null, h("b", null, "思考"), h("em", null, oneLine(item.text))) },
              h("pre", { className: "eg-pre" }, item.text));
          }
          if (item.kind === "reply") return h("div", { key: index, style: { marginTop: "4px" } }, h(Markdown, { text: workspace(item.text) }));
          if (item.kind === "call") return h(CallView, { key: index, call: item, result: results.get(item.id) });
          return null;
        })));
    }

    /** A recorded session (``episodes.transcripts``) step by step; ``flash`` marks the step jumped to. */
    function SessionSteps({ threads, flash }) {
      if (!(threads || []).length) return hint("没有存下这个会话的模型调用记录。");
      return threads.map((thread, index) => {
        const results = new Map();
        const blocks = [];
        for (const item of thread.steps || []) {
          if (item.kind === "result") {
            results.set(item.id, item);
            continue;
          }
          const turn = item.step ?? null;
          const last = blocks[blocks.length - 1];
          if (last && last.turn === turn) last.items.push(item);
          else blocks.push({ turn, items: [item] });
        }
        const turns = blocks.filter((block) => block.turn !== null).length;
        return h("div", { key: index },
          h("div", { className: "eg-thread-head" }, threads.length > 1 ? `线程 ${index + 1}：${thread.label}` : thread.label,
            h("span", null, `${thread.calls} 次模型调用 · ${turns} 步`)),
          blocks.map((block, i) => h(Turn, { key: i, id: block.turn === null ? undefined : `eg-turn-${index}-${block.turn}`, turn: block.turn,
            items: block.items, results, flash: block.turn !== null && flash === `${index}-${block.turn}` })));
      });
    }

    function smokeLine(smoke, catalog) {
      const tasks = Object.entries(smoke?.tasks || {});
      if (!tasks.length) {
        return ` · 模板本地 ${signed(smoke?.score, 3)}${isNum(smoke?.incumbent_score) ? `（在任 ${signed(smoke.incumbent_score, 3)}）` : ""}`;
      }
      return ` · 模板本地：${tasks.map(([task, item]) => `${taskShort(catalog, task)} ${item.status === "ok" ? signed(item.score, 3) : item.status}`
        + `${isNum(item.incumbent_score) ? `（在任 ${signed(item.incumbent_score, 3)}）` : ""}`).join("、")}`;
    }

    function ToolAction({ item, catalog }) {
      const args = item.arguments || {};
      const result = item.result || {};
      const head = h("span", { className: "eg-kind", "data-tool": "ecogym" }, toolName(ECOGYM_PREFIX + item.tool).label);
      if (result.error) return h(Fragment, null, head, h("span", { className: "eg-bad" }, `出错：${result.error}`));
      if (item.tool === "check_candidate") {
        return h(Fragment, null, head, `候选 ${args.name ?? result.name}：`,
          h("b", { className: result.ok ? "eg-good" : "eg-bad" }, result.ok ? "通过" : `没有通过（${(CAND_STATUS[result.status] || [result.status])[0]}）`),
          result.smoke ? smokeLine(result.smoke, catalog) : " · 模板没有改动，不试运行",
          (result.problems || []).length ? h("div", { className: "eg-bad" }, result.problems.join("；")) : null,
          (result.warnings || []).length ? h("div", { className: "eg-warn" }, `提醒：${result.warnings.join("；")}`) : null,
          (result.changes || []).length ? h("div", { className: "eg-faint" }, `相对在任版本改了：${result.changes.map((change) => `${change.path} +${change.added ?? 0}/${MINUS}${change.removed ?? 0}`).join("，")}`) : null);
      }
      if (item.tool === "reset_candidate") {
        return h(Fragment, null, head, `候选 ${args.name} 重置为${args.base === "incumbent" ? "在任版本" : `存档里的 ${args.base}`}`,
          result.summary ? h("div", { className: "eg-faint" }, `它的摘要：${result.summary}`) : null);
      }
      if (item.tool === "screen_templates") {
        const rows = result.variants || [];
        return h(Fragment, null, head, `在本地检验期上试了 ${rows.length} 个模板变体（在任版本的模板平均 ${signed(result.incumbent_mean, 3)}`
          + `${isNum(result.seconds) ? `，用时 ${duration(result.seconds)}` : ""}）：`,
          h("div", { className: "eg-faint" }, rows.map((row) => (row.status === "ok"
            ? `${row.name} ${signed(row.gain, 4)}（${row.better} 个任务更好、${row.worse} 个更差）`
            : `${row.name} ${{ missing: "找不到", invalid: "不能用", failed: "没有正常完成" }[row.status] || row.status}${row.problem ? `：${row.problem}` : ""}`)).join("；")));
      }
      if (item.tool === "task_info") {
        const states = Object.entries(result.candidates || {}).map(([name, state]) => `${name} ${typeof state === "string" ? "未检查" : state.ok ? "通过" : "没通过"}`);
        return h(Fragment, null, head, `剩 ${result.minutes_left ?? "–"} 分钟，已用 ${tokens(result.tokens_used)} token`, states.length ? `；候选 ${states.join("、")}` : "");
      }
      if (item.tool === "note") return h(Fragment, null, head, h("span", { className: "eg-chip" }, NOTE_KIND[args.kind] || args.kind), " ", args.text);
      if (item.tool === "ask_expert") return h(Fragment, null, head, args.question);
      return h(Fragment, null, head, h(Mono, null, JSON.stringify(args)));
    }

    function WriteAction({ item }) {
      return h(Fragment, null,
        h("span", { className: "eg-kind", "data-tool": "write" }, item.certain ? toolName(item.tool).label : "命令行（推断）"),
        item.paths.map((path, index) => h(Fragment, { key: path }, index ? "、" : "", h(Mono, null, path))),
        item.count > 1 ? h("span", { className: "eg-faint" }, ` ×${item.count}`) : null,
        item.description ? h("span", { className: "eg-faint" }, ` · ${item.description}`) : null);
    }

    function Actions({ items, onStep, catalog }) {
      if (!(items || []).length) return hint("没有记录到宿主工具的调用或候选文件的改动。");
      return h("div", null, items.map((item, index) => h("div", { key: index, className: "eg-act" },
        h("button", { type: "button", className: "eg-where", onClick: () => onStep(item.where) }, whereText(item.where)),
        h("div", { className: "eg-act-main" }, item.kind === "tool" ? h(ToolAction, { item, catalog }) : h(WriteAction, { item })))));
    }

    const SESSION_STATUS = { ...EP_STATUS, error: ["出错", "danger", "error"] };
    /** How long failed model calls kept a session (an episode or an evolver session) from working. */
    const stalledFor = (item) => (isNum(item.stalled_seconds) ? `${Math.round(item.stalled_seconds)} 秒` : "大半时间");

    /** Why an evolver session did not end normally, as [title, text]; null when it did. */
    function sessionTrouble(session) {
      const then = session.rerun ? "这一代不算数，等模型服务能回复后重新评审。" : "这一代不算数：运行停下，继续时这一代重跑。";
      if (session.status === "error") return ["评审会话出错：", session.error || ""];
      if (session.status !== "infra_failed") return null;
      if (!session.stalled) return ["评审会话没有得到模型回复：", `${session.upstream_error || "服务故障"}。${then}`];
      return ["模型服务故障让评审会话没法工作：", `${stalledFor(session)}都在等失败的模型调用，也没有留下能比较的候选。${then}`
        + `${session.upstream_error ? `最后一次失败：${session.upstream_error}` : ""}`];
    }

    function EvolverPage({ id, go, catalog, back }) {
      const text = String(id || "");
      const run = text.slice(0, text.lastIndexOf("|"));
      const generation = text.slice(text.lastIndexOf("|") + 1);
      const { data, error } = useResource(Api.evolverSession(run, generation, true));
      const [flash, setFlash] = useState(null);
      useEffect(() => {
        if (!flash) return undefined;
        const timer = setTimeout(() => setFlash(null), 1800);
        return () => clearTimeout(timer);
      }, [flash]);
      const jump = useCallback((where) => {
        document.getElementById(`eg-turn-${where[0]}-${where[1]}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
        setFlash(`${where[0]}-${where[1]}`);
      }, []);
      const toRun = back || { page: "evolution", id: run, focus: { generation: Number(generation) } };
      const returning = h("button", { className: "eg-back", onClick: () => go(toRun) }, `← 返回运行的第 ${generation} 代`);
      if (error && !data) return h(Fragment, null, returning, h(Alert, { tone: "danger" }, `读不到评审会话：${error.message}`));
      if (!data) return h(Fragment, null, returning, hint("正在读取评审会话…"));
      const session = data.session || {};
      const trouble = sessionTrouble(session);
      const calls = session.activity?.tool_calls || {};
      const used = Object.values(calls).reduce((sum, count) => sum + count, 0);
      return h(Fragment, null, returning,
        h("div", { className: "eg-head" }, h("div", { className: "eg-head-main" },
          h("div", { className: "eg-title-line" }, h("h1", null, `第 ${data.generation} 代的评审会话`), h(StatusTag, { map: SESSION_STATUS, keyName: session.status })),
          h("p", null, "评审智能体在自己的工作区里读证据、在候选目录里改 Harness，用宿主的工具检查。下面依次是它看过的材料、关键动作、产出的候选、最后的总结和完整的每一步。"),
          h("div", { className: "eg-meta" },
            h("span", null, h("b", null, "运行"), h(Link, { onClick: () => go(toRun) }, data.run_id)),
            h("span", null, h("b", null, "这一代的在任"), h(Mono, null, short(data.incumbent))),
            h("span", null, h("b", null, "开始"), when(data.started_at, true))))),
        trouble ? h(Alert, { tone: "danger" }, h("b", null, trouble[0]), trouble[1]) : null,
        h("div", { className: "eg-stats" },
          h(Stat, { label: "用时", value: duration(session.seconds) }),
          h(Stat, { label: "Token", value: tokens(session.tokens), sub: session.metered_tokens ? `计费 ${tokens(session.metered_tokens)}` : "" }),
          h(Stat, { label: "步数", value: String(data.turns ?? "–"), sub: `${session.llm_calls ?? "–"} 次模型调用${session.failed_calls ? `，失败 ${session.failed_calls} 次` : ""}`
            + `${session.continuations ? `；含 ${session.continuations} 段续跑` : ""}` }),
          h(Stat, { label: "工具调用", value: `${used} 次`, sub: Object.entries(calls).map(([name, count]) => `${toolName(name).label} ${count}`).join("、") })),
        h(Card, { title: "看过的材料", hint: "按第一次看的先后排列；点一项跳到那一步。来自读文件的调用和命令行里出现的路径（命令行部分是推断）。" },
          h(Reads, { reads: data.reads, onStep: jump })),
        h(Card, { title: "关键动作", hint: "调用宿主工具（检查、重置候选、记笔记、看剩余预算）和改动候选或提案文件的步骤。命令行改了哪些文件是从命令里推断的。" },
          h(Actions, { items: data.timeline, onStep: jump, catalog })),
        h(Card, { title: "产出的候选", hint: "会话结束后宿主再检查一遍；回合、比较和结果在运行页这一代的过程里。",
          actions: h(Button, { onClick: () => go(toRun) }, "看这一代的比较和结果") },
        (data.candidates || []).length ? data.candidates.map((candidate) => h("div", { key: candidate.name, className: "eg-note" },
          h("b", null, `候选 ${candidate.name}`), " ", h(StatusTag, { map: CAND_STATUS, keyName: candidate.status }), " ", h(Mono, null, short(candidate.digest)),
          h("span", { className: "eg-faint" }, candidate.base?.label ? ` · 从存档 ${candidate.base.label} 出发` : " · 从在任版本出发"),
          candidate.summary ? h("div", null, candidate.summary) : null)) : hint("没有候选。")),
        data.final_text ? h(Card, { title: "评审的总结（会话最后的回复）" }, h(Markdown, { text: data.final_text })) : null,
        (data.notes || []).length ? h(Card, { title: "记下的笔记", hint: "以后各代的评审在证据里会读到。" },
          data.notes.map((note, index) => h("div", { key: index, className: "eg-note" }, h("span", { className: "eg-chip" }, NOTE_KIND[note.kind] || note.kind), " ", note.text))) : null,
        h(Card, { title: "每一步", hint: "每一步的思考、工具调用和结果；点开一行看完整的参数和结果（过长处保留首尾）。系统提示和 DSH 注入的上下文只列出名字，工作区写作 /workspace。" },
          h(SessionSteps, { threads: data.threads, flash })));
    }

    /** An episode's sessions step by step, read when asked for (the log can be long). */
    function TranscriptCard({ record }) {
      const [shown, setShown] = useState(false);
      const running = epKey(record) === "running";
      const { data, error } = useResource(shown ? Api.episodeTranscript(record.episode) : null);
      let body = null;
      if (shown) {
        body = error ? h(Alert, { tone: "warning" }, `读不到会话记录：${error.message}`) : !data ? hint("正在读取会话记录…")
          : h(SessionSteps, { threads: data.threads });
      }
      return h(Card, { title: "会话记录", hint: running ? "回合结束后可以看到智能体每一步的思考、工具调用和结果。"
        : "智能体每一步的思考、调用的工具和得到的结果。系统提示和 DSH 注入的上下文只列出名字，工作区写作 /workspace。",
      actions: h(Button, { disabled: running, onClick: () => setShown(!shown) }, shown ? "收起" : "显示每一步") }, body);
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
        h(PageHead, { title: "专家参与", text: "智能体遇到需要领域判断的问题会留言，然后继续工作，不等待回答。专家的回答和主动给出的提示写入账本；之后匹配任务的回合会在 data/expert.md 里读到，评审智能体在证据里也能看到。" }),
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
          item.task_id ? h(Button, { onClick: () => go({ tab: "episodes", page: "episode", id: item.run_id }) }, "查看回合") : null))) : hint("现在没有待回答的问题。")),
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
      if (!data) return hint("正在读取…");
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
        h(PageHead, { title: "训练数据", text: "每个回合和评审会话的完整模型调用都保存在轨迹里。导出时按对话线程重建成多轮 SFT 样本（含推理和工具调用），同一任务上隐藏期成绩可信地拉开差距的两个回合组成偏好对；笔记汇成经验库。" }),
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
        item.skipped && Object.keys(item.skipped).length ? hint(`跳过：${Object.entries(item.skipped).map(([reason, count]) => `${reason} ${count}`).join("；")}`) : null,
        item.path ? hint(h(Mono, null, item.path)) : null,
        preview === item.id ? h(ExportPreview, { id: item.id }) : null)));
    }

    /* ---- tasks --------------------------------------------------------------------------------------------- */

    /** A suite's adoption gate in words (the suite's ``gates``; also shown with a run's settings). */
    function gateText(gates) {
      if (!gates) return "–";
      const interval = gates.test === "signed_rank" ? "各任务增益的符号秩区间（Wilcoxon 符号秩检验的 Hodges–Lehmann 区间）" : "各任务增益的 t 区间";
      return `平均增益大于 ${gates.min_delta}；区间下限大于 0（3 个及以上训练任务时是${interval}，置信度 ${Math.round((gates.confidence || 0) * 100)}%，多个候选时按 Bonferroni 调整）；没有任务的损失超过 ${gates.task_cap} 且它自己的区间整体低于 0（两个版本在这个任务上都至少有 2 个回合时才判断）。`;
    }

    function TasksPage({ catalog, catalogError }) {
      if (!catalog) return h(CatalogPending, { error: catalogError });
      return h(Fragment, null,
        h(PageHead, { title: "任务", text: "任务组把同一类预测任务分成训练任务（进化时使用）和留出任务（只在验证时运行）。每个任务给定数据集、预测目标和时距；宿主保管隐藏时期和封存时期，按逐时刻预报打分。新领域的接入方法见 docs/new-domain.md。" }),
        (catalog.suites || []).map((suite) => h(Card, { key: suite.id, title: suite.title, hint: suite.id },
          h(KV, { pairs: [
            ["训练任务", suite.train.map((task) => taskShort(catalog, task)).join("、")],
            ["留出任务", `${suite.holdout_tasks} 个`],
            ["采纳门槛", gateText(suite.gates)],
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

    const STAGE_INFO = {
      calibration_fit: ["拟合期", "visible", "智能体可见，用来拟合它的程序"],
      calibration_uq: ["本地检验期", "local", "智能体可见：evaluate_local 和 submit 在这里打分，智能体用它在回合里挑版本"],
      model_selection: ["隐藏期", "hidden", "只有宿主能看到：被评分的程序用它之前的全部行拟合，再逐时刻预报；比较 Harness 版本只用这个成绩"],
      validation: ["封存 validation", "sealed", "进化中谁都没看过：独立验证和各次采纳的复核用它，每个版本每个任务只评一次"],
      final_test: ["封存 final_test", "sealed", "最后一道检验，在 validation 之后，每个版本每个任务只评一次"],
    };
    const EPISODE_FACTS = [
      ["工作区", "TASK.md（目标、评分、接口、时限、预算），AGENTS.md（Harness 的说明和经验），data/（可见数据、列说明、领域背景、专家意见），submission/（Harness 的起始代码）"],
      ["起始程序", "智能体开始前，宿主先在本地检验期上运行 submission/ 里原有的程序（模板）；一次提交只有在本地总分比它高至少 0.02 时才替换它，否则评分用模板"],
      ["工具", "task_info（剩余时间、预算、现在评分用哪个程序）、evaluate_local（本地检验）、submit（提交，告诉你是否替换了起始程序）、note（记下发现和教训）、ask_expert（给专家留言，不等回答）"],
      ["沙箱", "只能读写自己的回合目录，只能连本机的模型网关和工具服务"],
      ["模型网关", "真实密钥留在本机；按成本计 token，到预算或截止时间就拒绝调用；每次请求和回复都记进轨迹"],
      ["续跑", "智能体在剩余时间超过四分之一时就结束的，宿主开一段新会话请它继续（最多 3 段）；一段续跑既没有检验也没有提交就停"],
      ["结束后", "裁判在隐藏期上评分；模型调用、工具调用、工作区和 DSH 会话存进轨迹，之后可以导出为训练数据"],
    ];
    const TERMS = [
      ["Harness", "智能体带进每个任务的全部内容：说明、技能、起始代码库、经验和 DSH 设置。进化的就是它。"],
      ["回合", "一个智能体按一版 Harness 独立解一个任务的一次运行。"],
      ["在任 / 候选", "当前被采纳的版本 / 评审新写的、等待比较的版本。"],
      ["代", "一次评审，加上候选和在任版本在每个训练任务上的回合，再加一次比较。第 0 代只跑起点版本。"],
      ["训练任务 / 留出任务", "进化中比较版本用的任务 / 评审看不到、只在最终验证时运行的任务。"],
      ["技能", "每格 = (RMSE基线 − RMSE) / max(RMSE, RMSE基线)；总分是逐格加权平均，0 等于朴素基线。"],
      ["隐藏期 / 封存时期", "隐藏期：智能体看不到、宿主用来比较版本的一段时间。封存时期：更晚的一段，智能体和评审都看不到，只用来验证（最终验证和各次采纳的复核），每个版本每个任务只评一次。"],
      ["采纳门槛", "候选比在任版本平均高得足够多（默认 0.005），各训练任务增益的区间整段在 0 以上，而且没有任务可信地变差太多，才换上；各任务组的完整规则在「任务」页。"],
      ["账本", "只追加的事件记录；运行中断后从它恢复，面板上的每个数字都从它算出。"],
      ["SFT 线程 / 偏好对", "从回合轨迹导出的多轮训练样本 / 同一任务上成绩可信地不同的两个回合。"],
    ];

    function Timeline({ task }) {
      const { data, error } = useResource(task ? Api.stages(task) : null);
      if (error) return h(Alert, { tone: "warning" }, `读不到时间分段：${error.message}`);
      if (!data) return hint("正在读取…");
      const total = data.days || 1;
      const info = (item) => STAGE_INFO[item.stage] || [item.stage, "visible", ""];
      return h(Fragment, null,
        h("div", { className: "eg-timeline", role: "img", "aria-label": "时间分段" }, data.stages.map((item) => {
          const [label, kind] = info(item);
          return h("div", { key: item.stage, className: "eg-seg", "data-kind": kind, title: `${label}：${item.start} → ${item.end}`,
            style: { left: `${(item.from_days / total) * 100}%`, width: `${(item.days / total) * 100}%` } }, label);
        })),
        h(Table, { head: ["时期", "时间", ["天数", "num"], "谁能看到、做什么用"], style: { marginTop: "8px" } }, data.stages.map((item) => {
          const [label, kind, who] = info(item);
          return h("tr", { key: item.stage },
            h("td", { style: { whiteSpace: "nowrap" } }, h("span", { className: "eg-swatch", "data-kind": kind }), label),
            h("td", { style: { whiteSpace: "nowrap" } }, h(Mono, null, `${item.start.replace("T", " ").slice(0, 16)} → ${item.end.replace("T", " ").slice(0, 16)}`)),
            h("td", { className: "num" }, plain(item.days, 1)), h("td", null, who));
        })),
        hint("相邻两段之间留出一小段间隔（embargo），预报时距不会跨进下一段。智能体的数据文件到隐藏期开始前为止；裁判评隐藏期时用它之前的全部行拟合，再逐个时间步送入新观测。"));
    }

    function AboutPage({ catalog, catalogError }) {
      const tasks = [...new Set((catalog?.suites || []).flatMap((suite) => suite.train))];
      const [task, setTask] = useState(null);
      if (!catalog) return h(CatalogPending, { error: catalogError });
      const shown = task || tasks[0];
      return h(Fragment, null,
        h(PageHead, { title: "介绍", text: "ecogym 让大模型研究智能体在生态预测任务上自己做研究：看数据、查资料、写预测程序、检验和提交。它带进任务的工作方式（Harness）由评审智能体根据成绩和完整轨迹不断改进，只有在多任务比较中可信地变好才被采纳。过程中积累可复用的知识（技能、代码库、经验、专家意见）和完整的轨迹，轨迹可以导出为训练数据。DSH 本身不改，打分全部由宿主上的确定性代码完成，不交给大模型。" }),
        h(Card, { title: "整体流程", hint: "第 1–4 步每一代重复一次（第 0 代只让起点版本跑第 1、2 步，作为基线）；最后一代结束后做第 5 步。" },
          h("div", { className: "eg-flow" }, FLOW.map(([title, , text], index) => h("div", { key: title, className: "eg-step" },
            h("b", null, h("i", null, index + 1), title), h("p", null, text))))),
        h(Card, { title: "术语" }, h(KV, { pairs: TERMS })),
        h(Card, { title: "数据怎么切、谁能看到", hint: "任务卡给定数据集、预测目标和时距。每个任务的数据按时间切成五段，智能体只拿到隐藏期之前的部分；下面是一个训练任务的实际分段。",
          actions: tasks.length > 1 ? h("select", { className: "eg-input", style: { width: "auto" }, value: shown, onChange: (event) => setTask(event.target.value) },
            tasks.map((id) => h("option", { key: id, value: id }, taskShort(catalog, id)))) : null },
          shown ? h(Timeline, { task: shown }) : hint("还没有任务组。")),
        h(Card, { title: "一个回合里有什么" }, h(KV, { pairs: EPISODE_FACTS })),
        h("p", { className: "eg-hint", style: { marginTop: "16px" } }, "关闭页面不影响运行；重启后端时，进化停在最后一个完成的代，之后可以继续（勾选了自动继续的运行会自己继续）。"
          + "面板、DSH 插件、ecogym 服务和账本怎么配合，见 docs/architecture.md；服务的全部接口可以从 GET /api/routes 读到。"));
    }

    /* ---- panel ----------------------------------------------------------------------------------------------- */

    // Top-bar pages, with what each is for (its tooltip). A page inside a tab (a run, an episode, the new-evolution
    // form) is nav.page with nav.id.
    const TABS = [
      ["evolutions", "进化", "进化运行；一次运行分四部分看：概览、每一代、版本与成绩、设置"],
      ["episodes", "回合", "全部回合：逐格技能、本地检验与提交、笔记、会话记录"],
      ["expert", "专家", "回答智能体提给专家的问题，或主动写下领域提示；之后的回合和评审都会读到"],
      ["data", "训练数据", "把回合和评审会话导出为多轮 SFT 线程和偏好对"],
      ["tasks", "任务", "任务组的训练与留出任务、采纳门槛，各任务的数据集、目标和时距"],
      ["about", "介绍", "这个项目在做什么：流程、术语、数据怎么切"],
    ];
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
          // back: where the page's back button leads (a run page opened at one generation: focus)
          const value = { tab: next.tab || old.tab, page: next.page ?? null, id: next.id ?? null, back: next.back ?? null, focus: next.focus ?? null };
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
      else if (nav.page === "evolution") page = h(EvolutionPage, { key: nav.id, id: nav.id, go, catalog, focus: nav.focus });
      else if (nav.page === "evolver") page = h(EvolverPage, { key: nav.id, id: nav.id, go, catalog, back: nav.back });
      else if (nav.page === "episode") page = h(EpisodePage, { key: nav.id, id: nav.id, go, catalog, back: nav.back });
      else if (nav.tab === "episodes") page = h(EpisodeList, { go, catalog });
      else if (nav.tab === "expert") page = h(ExpertPage, { go, catalog });
      else if (nav.tab === "data") page = h(DataPage, null);
      else if (nav.tab === "tasks") page = h(TasksPage, { catalog, catalogError });
      else if (nav.tab === "about") page = h(AboutPage, { catalog, catalogError });
      else page = h(EvolutionList, { go });
      const state = status.data?.state;
      const [pillText, pillDot] = ready ? ["后端运行中", "done"] : state === "starting" ? ["后端启动中", "ongoing"]
        : state === "exited" ? ["后端重启中", "warning"] : status.error ? ["插件未连接", "error"] : ["后端未运行", "error"];
      return h("div", { className: "eg" },
        h("header", { className: "eg-bar" },
          h("span", { className: "eg-brand" }, h(Leaf, { size: 18 }), "生态进化"),
          h("nav", { className: "eg-tabs", "aria-label": "生态进化" }, TABS.map(([id, label, tip]) => h("button", {
            key: id, type: "button", className: "eg-tab", title: tip, "aria-current": nav.tab === id && !nav.page ? "page" : undefined,
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
