window.__ModuleLoader__.load({
  id: "@ecologyrsi/dsh-evolution-plugin",
  factory: (require) => {
    const module = { exports: {} };
    const exports = module.exports;
    Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });

    const React = require("react");
    const {
      IconCloseOutline16,
      IconEnhanceOutline16,
    } = require("@deepseek-ai/dsh-client-ui-primitives");

    const PLUGIN_URL = "/plugins/ecology/evolution/?api=/api/ecology-evolution";
    const OPEN_EVENT = "ecologyrsi/open-evolution";
    const capabilities = [
      "evolution.catalog.read",
      "evolution.run.create",
      "evolution.run.advance",
      "evolution.projection.read",
      "evaluation.samples.read",
      "training.data.read",
      "run.control",
      "run.archive",
      "run.delete",
      "intervention.write",
    ];
    let hostPluginContext = null;

    function boundedText(value, maximum = 200) {
      if (typeof value !== "string") return "";
      const normalized = value.trim();
      return normalized ? normalized.slice(0, maximum) : "";
    }

    function hostModelId(provider, model) {
      return `${provider}/${model}`;
    }

    function flattenHostModelDirectory(response) {
      const result = response && response.result;
      if (!result || result.ok !== true || !result.value) return [];
      const groups = Array.isArray(result.value.groups) ? result.value.groups : [];
      const seen = new Set();
      const models = [];
      for (const group of groups) {
        const provider = boundedText(group && group.id, 120);
        const providerLabel = boundedText(group && group.name, 160) || provider;
        if (!provider || !Array.isArray(group.models)) continue;
        for (const model of group.models) {
          const modelName = boundedText(model && model.id, 200);
          if (!modelName) continue;
          const id = hostModelId(provider, modelName);
          if (seen.has(id)) continue;
          seen.add(id);
          const modelLabel = boundedText(model && model.name, 160) || modelName;
          models.push({
            id,
            model_id: id,
            label: `${providerLabel} · ${modelLabel}`.slice(0, 260),
            provider,
            model: modelName,
            aliases: [modelName, `${provider}:${modelName}`],
            roles: ["propose", "judge"],
            model_source: "dsh_host_directory",
            authentication_state: "dsh_authenticated",
            credential_configured: true,
            authentication_verified: true,
            authenticated: true,
            available: true,
          });
        }
      }
      return models.slice(0, 100);
    }

    function requestHostModelDirectoryOverHttp() {
      const rpcId = globalThis.crypto && typeof globalThis.crypto.randomUUID === "function"
        ? globalThis.crypto.randomUUID()
        : `ecologyrsi-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      return fetch("/api/llm.models", {
        method: "POST",
        headers: { "content-type": "application/json", "accept": "application/json" },
        body: JSON.stringify({
          type: "client-request",
          rpcId,
          method: "llm.models",
          payload: {},
        }),
      }).then((response) => response.ok ? response.json() : null).then(flattenHostModelDirectory).catch(() => []);
    }

    function readHostModelDirectory(ctx) {
      try {
        const connection = ctx.get("connection");
        if (!connection || !connection.api || !connection.api.llm) {
          console.warn("[ecologyrsi] DSH connection API unavailable; using same-origin model directory");
          return requestHostModelDirectoryOverHttp();
        }
        return connection.api.llm.models({}).then((response) => {
          const models = flattenHostModelDirectory(response);
          if (!models.length) console.warn("[ecologyrsi] DSH connection model directory empty; using same-origin fallback");
          return models.length ? models : requestHostModelDirectoryOverHttp();
        }).catch((error) => {
          console.warn("[ecologyrsi] DSH connection model directory failed; using same-origin fallback", error && error.message);
          return requestHostModelDirectoryOverHttp();
        });
      } catch (_error) {
        console.warn("[ecologyrsi] DSH connection lookup threw; using same-origin fallback");
        return requestHostModelDirectoryOverHttp();
      }
    }

    const css = `
      .ecology-dsh-entry {
        box-sizing: border-box;
        flex: 1 1 auto;
        min-width: 36px;
        height: 42px;
        margin: 4px 0 0;
        display: flex;
        align-items: center;
      }
      .ecology-dsh-entry[data-rail="true"] {
        flex: 0 0 36px;
        width: 36px;
        height: 36px;
        margin: 0;
      }
      .ecology-dsh-entry__button {
        width: 100%;
        height: 38px;
        padding: 0 10px;
        display: inline-flex;
        align-items: center;
        justify-content: flex-start;
        gap: 8px;
        overflow: hidden;
        border: 0;
        border-radius: 8px;
        color: var(--dsw-alias-label-primary);
        background: transparent;
        cursor: pointer;
        font: inherit;
        font-size: 13px;
      }
      .ecology-dsh-entry__button:hover {
        background: var(--dsw-alias-interactive-bg-hover-solid);
      }
      .ecology-dsh-entry[data-rail="true"] .ecology-dsh-entry__button {
        width: 36px;
        height: 36px;
        padding: 0;
        justify-content: center;
        border-radius: 50%;
      }
      .ecology-dsh-entry__label {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .ecology-dsh-overlay {
        position: fixed;
        inset: 0;
        z-index: 1000;
        display: flex;
        flex-direction: column;
        background: var(--dsw-alias-bg-base, #fff);
        color: var(--dsw-alias-label-primary, #1f2328);
      }
      .ecology-dsh-overlay__bar {
        box-sizing: border-box;
        flex: 0 0 48px;
        min-height: 48px;
        padding: 0 14px 0 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 1px solid var(--dsw-alias-border-l2, #e5e7eb);
        background: var(--dsw-alias-bg-base, #fff);
      }
      .ecology-dsh-overlay__title {
        min-width: 0;
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 14px;
        font-weight: 600;
      }
      .ecology-dsh-overlay__status {
        color: var(--dsw-alias-label-tertiary, #667085);
        font-size: 12px;
      }
      .ecology-dsh-overlay__close {
        width: 32px;
        height: 32px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 0;
        border-radius: 50%;
        color: inherit;
        background: transparent;
        cursor: pointer;
      }
      .ecology-dsh-overlay__close:hover {
        background: var(--dsw-alias-interactive-bg-hover);
      }
      .ecology-dsh-overlay__frame {
        flex: 1 1 auto;
        width: 100%;
        min-height: 0;
        border: 0;
        background: #fff;
      }
    `;

    const styleId = "@ecologyrsi/dsh-evolution-plugin/client";
    if (!document.querySelector(`style[data-plugin-css="${styleId}"]`)) {
      const style = document.createElement("style");
      style.dataset.plugin = "@ecologyrsi/dsh-evolution-plugin";
      style.dataset.pluginCss = styleId;
      style.textContent = css;
      document.head.append(style);
    }

    function openWorkbench() {
      window.dispatchEvent(new CustomEvent(OPEN_EVENT));
    }

    function EvolutionEntry({ wide }) {
      return React.createElement(
        "div",
        { className: "ecology-dsh-entry", "data-rail": String(!wide) },
        React.createElement(
          "button",
          {
            type: "button",
            className: "ecology-dsh-entry__button",
            title: "生态模型进化工作台",
            "aria-label": "打开生态模型进化工作台",
            onClick: openWorkbench,
          },
          React.createElement(IconEnhanceOutline16, { size: wide ? 16 : 18 }),
          wide && React.createElement(
            "span",
            { className: "ecology-dsh-entry__label" },
            "生态模型进化",
          ),
        ),
      );
    }

    function EvolutionOverlay() {
      const [visible, setVisible] = React.useState(false);
      const iframeRef = React.useRef(null);

      React.useEffect(() => {
        const open = () => setVisible(true);
        window.addEventListener(OPEN_EVENT, open);
        return () => window.removeEventListener(OPEN_EVENT, open);
      }, []);

      React.useEffect(() => {
        if (!visible) return undefined;
        const closeOnEscape = (event) => {
          if (event.key === "Escape") setVisible(false);
        };
        window.addEventListener("keydown", closeOnEscape);
        return () => window.removeEventListener("keydown", closeOnEscape);
      }, [visible]);

      React.useEffect(() => {
        if (!visible) return undefined;
        let disposed = false;
        const sendContext = async () => {
          const target = iframeRef.current && iframeRef.current.contentWindow;
          if (!target) return;
          const models = await readHostModelDirectory(hostPluginContext);
          if (disposed || target !== (iframeRef.current && iframeRef.current.contentWindow)) return;
          target.postMessage({
            type: "dsh.context",
            api_base: "/api/ecology-evolution",
            capability_token: "dsh-local-loopback",
            identity: {
              subject_id: "dsh-local-user",
              display_name: "DSH 本地研究员",
            },
            capabilities,
            models,
          }, window.location.origin);
        };
        const onMessage = (event) => {
          const target = iframeRef.current && iframeRef.current.contentWindow;
          if (
            event.origin === window.location.origin
            && event.source === target
            && event.data
            && event.data.type === "plugin.ready"
          ) void sendContext();
        };
        window.addEventListener("message", onMessage);
        return () => {
          disposed = true;
          window.removeEventListener("message", onMessage);
        };
      }, [visible]);

      if (!visible) return null;
      return React.createElement(
        "section",
        {
          className: "ecology-dsh-overlay",
          role: "dialog",
          "aria-modal": "true",
          "aria-label": "生态模型进化工作台",
        },
        React.createElement(
          "header",
          { className: "ecology-dsh-overlay__bar" },
          React.createElement(IconEnhanceOutline16, { size: 18 }),
          React.createElement(
            "strong",
            { className: "ecology-dsh-overlay__title" },
            "生态模型进化工作台",
          ),
          React.createElement(
            "span",
            { className: "ecology-dsh-overlay__status" },
            "DSH 插件",
          ),
          React.createElement(
            "button",
            {
              type: "button",
              className: "ecology-dsh-overlay__close",
              title: "关闭",
              "aria-label": "关闭生态模型进化工作台",
              onClick: () => setVisible(false),
            },
            React.createElement(IconCloseOutline16, { size: 18 }),
          ),
        ),
        React.createElement("iframe", {
          ref: iframeRef,
          className: "ecology-dsh-overlay__frame",
          src: PLUGIN_URL,
          title: "生态模型进化工作台",
        }),
      );
    }

    const inject = ["slots", "connection"];
    function apply(ctx) {
      hostPluginContext = ctx;
      ctx.slots.inject("sidebar.footer.action", () => ctx.slots.register({
        name: "sidebar.footer.action",
        id: "ecologyrsi-evolution",
        order: -100,
      }, EvolutionEntry));
      ctx.slots.inject("shell.overlay", () => ctx.slots.register({
        name: "shell.overlay",
        id: "ecologyrsi-evolution",
        order: 100,
      }, EvolutionOverlay));
    }

    exports.apply = apply;
    exports.inject = inject;
    return module.exports;
  },
});
