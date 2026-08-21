"use strict";

window.EcologyDSHHost = (function () {
  var pluginId = "ecologyrsi.evolution";
  var pluginVersion = "0.3.15";
  var contextProtocol = "ecology-evolution.host-context/1";
  var supportedApiPaths = [
    "/api",
    "/api/v1",
    "/api/ecology-evolution",
    "/api/ecology-evolution/v1"
  ];
  var query = new URLSearchParams(window.location.search);

  function normalizeOrigin(value) {
    try { return new URL(String(value || ""), window.location.href).origin; } catch (error) { return null; }
  }
  function configuredOrigins(name) {
    return query.getAll(name).map(normalizeOrigin).filter(function (value, index, values) {
      return value && value !== "null" && values.indexOf(value) === index;
    });
  }

  var explicitParentOrigins = configuredOrigins("parent_origin");
  var explicitApiOrigins = configuredOrigins("api_origin");
  var capabilityToken = null;
  var hostContextReceived = false;
  var hostCapabilities = null;
  var hostIdentity = null;
  var hostModels = [];

  function normalizeBase(value) {
    var raw = String(value || "").trim();
    if (!raw) { throw new Error("API 地址不能为空。"); }
    var parsed;
    try { parsed = new URL(raw, window.location.origin); } catch (error) { throw new Error("API 地址格式无效。"); }
    var path = parsed.pathname.length > 1 && parsed.pathname.charAt(parsed.pathname.length - 1) === "/" ? parsed.pathname.slice(0, -1) : parsed.pathname;
    if (supportedApiPaths.indexOf(path) < 0 || parsed.search || parsed.hash) {
      throw new Error("API 地址不在插件允许的同源代理路径范围内。");
    }
    var allowedOrigins = [window.location.origin].concat(explicitApiOrigins);
    if (allowedOrigins.indexOf(parsed.origin) < 0) { throw new Error("API 地址来源未获授权。"); }
    return parsed.origin === window.location.origin ? path : parsed.origin + path;
  }

  var apiBase;
  try { apiBase = normalizeBase(query.get("api") || "/api"); } catch (error) { apiBase = "/api"; }

  function isTrustedParentOrigin(origin) {
    return origin === window.location.origin || explicitParentOrigins.indexOf(origin) >= 0;
  }
  function parentTargetOrigin() { return explicitParentOrigins[0] || window.location.origin; }
  function stringValue(value, maximumLength) {
    if (typeof value !== "string") { return null; }
    var normalized = value.trim();
    return normalized ? normalized.slice(0, maximumLength) : null;
  }
  function normalizeIdentity(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) { return null; }
    var subjectId = stringValue(value.subject_id || value.subjectId || value.id, 200);
    var displayName = stringValue(value.display_name || value.displayName || value.name, 120);
    if (!subjectId && !displayName) { return null; }
    return { subjectId: subjectId, displayName: displayName };
  }
  function normalizeCapabilities(value) {
    if (!Array.isArray(value)) { throw new Error("DSH 能力范围必须是字符串数组。"); }
    return value.map(function (item) { return stringValue(item, 160); }).filter(function (item, index, values) {
      return item && values.indexOf(item) === index;
    }).slice(0, 200);
  }
  function normalizeModels(value) {
    if (value == null) { return []; }
    if (!Array.isArray(value)) { throw new Error("DSH 模型目录必须是数组。"); }
    return value.map(function (item) {
      if (!item || typeof item !== "object" || Array.isArray(item)) { return null; }
      var id = stringValue(item.model_id || item.id, 200);
      if (!id) { return null; }
      var roleAliases = { strategy: "propose", policy: "propose", proposer: "propose", review: "judge", reviewer: "judge" };
      var roles = Array.isArray(item.roles) ? item.roles.map(function (role) {
        var normalized = stringValue(role, 40);
        var lowered = normalized && normalized.toLowerCase();
        var canonical = lowered && (roleAliases[lowered] || lowered);
        return normalized ? (canonical === "propose" || canonical === "judge" ? canonical : normalized) : null;
      }).filter(function (role, index, values) { return role && values.indexOf(role) === index; }) : [];
      var provider = stringValue(item.provider, 120);
      var model = stringValue(item.model, 200);
      var aliases = Array.isArray(item.aliases) ? item.aliases.map(function (alias) {
        return stringValue(alias, 220);
      }).filter(function (alias, index, values) {
        return alias && values.indexOf(alias) === index;
      }).slice(0, 8) : [];
      return {
        id: id,
        label: stringValue(item.label || item.display_name, 160),
        provider: provider,
        model: model,
        aliases: aliases,
        roles: roles
      };
    }).filter(Boolean).slice(0, 100);
  }
  function publicContext() {
    return {
      apiBase: apiBase,
      embedded: window.parent && window.parent !== window,
      connected: hostContextReceived,
      identity: hostIdentity,
      capabilities: hostCapabilities,
      models: hostModels
    };
  }

  function acceptContextMessage(event) {
    if (event.source !== window.parent || !event.data || event.data.type !== "dsh.context") {
      return { accepted: false, ignored: true };
    }
    if (!isTrustedParentOrigin(event.origin)) {
      return { accepted: false, error: "已拒绝来源未获授权的 DSH 宿主上下文。" };
    }
    var nested = event.data.context;
    var payload = nested && typeof nested === "object" && !Array.isArray(nested) ? Object.assign({}, event.data, nested) : event.data;
    var token = stringValue(payload.capability_token, 8192);
    var suppliedBase = payload.api_base || payload.apiBase || payload.proxy_path;
    if (!token || typeof suppliedBase !== "string" || !suppliedBase.trim()) {
      return { accepted: false, error: "DSH 宿主上下文缺少同源代理地址或能力令牌。" };
    }
    try {
      var nextBase = normalizeBase(suppliedBase);
      var hasCapabilities = Object.prototype.hasOwnProperty.call(payload, "capabilities") || Object.prototype.hasOwnProperty.call(payload, "capability_scopes");
      var nextCapabilities = hasCapabilities ? normalizeCapabilities(payload.capabilities || payload.capability_scopes) : null;
      var nextModels = normalizeModels(payload.models || payload.model_catalog);
      apiBase = nextBase;
      capabilityToken = token;
      hostCapabilities = nextCapabilities;
      hostIdentity = normalizeIdentity(payload.identity || payload.user);
      hostModels = nextModels;
      hostContextReceived = true;
      return { accepted: true, context: publicContext() };
    } catch (error) {
      return { accepted: false, error: error.message };
    }
  }

  function preferredModelId(role, availableIds, excludedId) {
    var available = Array.isArray(availableIds) ? availableIds : [];
    var aliases = { strategy: "propose", policy: "propose", planner: "propose", proposer: "propose", review: "judge", reviewer: "judge", critic: "judge" };
    var requiredRole = String(role || "").toLowerCase();
    requiredRole = aliases[requiredRole] || requiredRole;
    var match = hostModels.find(function (item) {
      var roles = Array.isArray(item.roles) ? item.roles.map(function (value) {
        var normalized = String(value || "").toLowerCase();
        return aliases[normalized] || normalized;
      }) : [];
      // Older DSH hosts did not attach role metadata.  Treat those entries as
      // shared candidates for preselection; the service remains authoritative
      // for the final role check.
      return item.id !== excludedId && (!roles.length || roles.indexOf(requiredRole) >= 0) && available.indexOf(item.id) >= 0;
    });
    return match ? match.id : "";
  }

  function request(path, options) {
    if (typeof path !== "string" || path.charAt(0) !== "/" || path.indexOf("//") === 0) {
      return Promise.reject(new Error("插件请求路径无效。"));
    }
    var opts = options || {};
    var controller = new AbortController();
    var timeout = window.setTimeout(function () { controller.abort(); }, opts.timeout || 8000);
    var headers = Object.assign({ Accept: "application/json" }, opts.headers || {});
    if (capabilityToken) { headers.Authorization = "Bearer " + capabilityToken; }
    if (opts.body && typeof opts.body !== "string") {
      headers["Content-Type"] = "application/json";
      opts = Object.assign({}, opts, { body: JSON.stringify(opts.body) });
    }
    return fetch(apiBase + path, Object.assign({}, opts, { headers: headers, signal: controller.signal })).then(function (response) {
      return response.text().then(function (text) {
        var payload = {};
        if (text) { try { payload = JSON.parse(text); } catch (error) { payload = { message: text }; } }
        if (!response.ok) {
          var message = payload.error || payload.message || "HTTP " + response.status;
          var requestError = new Error(typeof message === "string" ? message : JSON.stringify(message));
          requestError.status = response.status;
          requestError.errorCode = typeof payload.error_code === "string" ? payload.error_code : null;
          throw requestError;
        }
        return payload;
      });
    }).finally(function () { window.clearTimeout(timeout); });
  }

  function postReady() {
    if (!window.parent || window.parent === window) { return false; }
    window.parent.postMessage({
      type: "plugin.ready",
      plugin_id: pluginId,
      version: pluginVersion,
      context_protocol: contextProtocol,
      supported_api_bases: supportedApiPaths.slice()
    }, parentTargetOrigin());
    return true;
  }

  return {
    acceptContextMessage: acceptContextMessage,
    getPublicContext: publicContext,
    normalizeBase: normalizeBase,
    postReady: postReady,
    preferredModelId: preferredModelId,
    request: request,
    supportedApiPaths: supportedApiPaths.slice()
  };
})();
