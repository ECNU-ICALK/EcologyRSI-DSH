"""Model routes: which API answers an agent's model calls.

A model is named ``<provider>/<model>``: a provider route of the user's DSH
(the ``llm-pi-ai`` section of ``$DSH_HOME/settings.yaml``, as DSH's own model
picker lists it) and one of its models, or ``deepseek-official`` for
DeepSeek's own API. A bare DeepSeek model id (``deepseek-v4-pro``) means
``deepseek-official/<id>``.

Agents never call a provider themselves. The session's gateway forwards to
the route's ``baseURL`` with the key its ``apiKeyEnv`` names, resolved on the
host (``ecogym.kernel.secrets``: the environment, then DSH's credential
store), and adds the route's ``headers`` there too; the sandbox only ever
holds a per-session token. Routes on the ``openai-completions`` protocol
with a ``baseURL`` and a ``models`` list can run, because their replies are
assembled into trajectories; other routes are listed with the reason they
cannot. Settings are read with DSH's own YAML parser (a ``node`` next to
``dsh``), again whenever the file changes. Settings can list a model the key
may not use, so ``probe`` sends one tiny request before a run starts.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..kernel.secrets import has_secret, resolve_secret

DEEPSEEK = "deepseek-official"
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_KEY = "DEEPSEEK_API_KEY"
DEEPSEEK_MODELS = ("deepseek-v4-pro", "deepseek-v4-flash")  # DSH lists the rest itself
DEEPSEEK_MODEL = re.compile(r"^deepseek-[a-z0-9.-]{1,40}$")
SUPPORTED_API = "openai-completions"
PROVIDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+-]{0,127}$")
SESSION_FIELDS_DROPPED = ("apiKeyEnv", "baseURL", "headers", "models", "modelOverrides")
READ_SCRIPT = r"""
const [root, file] = process.argv.slice(1);
const fs = require("node:fs");
let text;
try {
  text = fs.readFileSync(file, "utf8");
} catch (error) {
  if (error.code === "ENOENT") { process.stdout.write("{}"); process.exit(0); }
  throw error;
}
const YAML = require("node:module").createRequire(require("node:path").join(root, "package.json"))("yaml");
const value = YAML.parse(text) || {};
process.stdout.write(JSON.stringify({ providers: (value["llm-pi-ai"] || {}).providers || {},
  default: value["agent-default-model"] || null }));
"""


class ModelError(ValueError):
    """A model reference that cannot run: unknown route or model, unsupported protocol, missing key."""


@dataclass(frozen=True, slots=True)
class Route:
    provider: str
    name: str
    api: str
    base_url: str
    key_env: str
    models: tuple[str, ...]  # empty for deepseek-official: any DeepSeek model id
    profile: dict[str, Any] = field(default_factory=dict)  # the user's settings for the route
    headers: dict[str, str] = field(default_factory=dict)  # sent upstream by the gateway, never to the sandbox
    problem: str = ""  # why the route cannot run, or ""

    def session_profile(self, model: str, effort: str, base_url: str, key_env: str) -> dict[str, Any]:
        """The route as one agent session sees it: only ``model``, through the gateway, with a session key."""

        entries = [entry for entry in self.profile.get("models") or [] if isinstance(entry, dict)
                   and entry.get("id") == model] or [{"id": model}]
        body = {key: value for key, value in self.profile.items() if key not in SESSION_FIELDS_DROPPED}
        body.update({"displayName": self.name, "api": self.api, "baseURL": base_url, "apiKeyEnv": key_env,
                     "models": entries[:1]})
        efforts = entries[0].get("reasoningEfforts")
        if isinstance(efforts, dict) and effort in efforts:  # pi-ai refuses a level the model does not declare
            body["reasoning"] = effort
        return body


def settings_path() -> Path:
    return Path(os.environ.get("DSH_HOME") or Path.home() / ".dsh").expanduser() / "settings.yaml"


def _dsh_tools() -> tuple[str, Path]:
    """The node that runs dsh, and dsh's package root (its node_modules hold the YAML parser)."""

    dsh = os.environ.get("ECOGYM_DSH_BIN") or shutil.which("dsh")
    if not dsh:
        raise ModelError("dsh not found; install DeepSeek Harness or set ECOGYM_DSH_BIN")
    script = Path(os.path.realpath(dsh))
    root = next((parent for parent in script.parents if (parent / "package.json").is_file()), None)
    node = shutil.which("node", path=str(Path(dsh).expanduser().parent)) or shutil.which("node")
    if root is None or node is None:
        raise ModelError(f"cannot find node or the dsh package next to {dsh}")
    return node, root


_CACHE: dict[str, Any] = {}


def dsh_settings() -> dict[str, Any]:
    """``{"providers": {...}, "default": {...} | None}`` from DSH's settings file (cached until it changes)."""

    path = settings_path()
    try:
        stat = path.stat()
        stamp = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return {"providers": {}, "default": None}
    if _CACHE.get("stamp") != stamp:
        node, root = _dsh_tools()
        done = subprocess.run([node, "-e", READ_SCRIPT, str(root), str(path)], capture_output=True, text=True,
                              timeout=60, stdin=subprocess.DEVNULL)
        if done.returncode != 0:
            raise ModelError(f"cannot read {path}: {done.stderr.strip()[-500:]}")
        _CACHE.update(stamp=stamp, value=json.loads(done.stdout or "{}"))
    return _CACHE["value"]


def _route(provider: str, profile: Any) -> Route:
    if not isinstance(profile, dict):
        return Route(provider, provider, "", "", "", (), problem="DSH 设置里这个提供方的配置不是映射")
    models = tuple(str(entry["id"]) for entry in profile.get("models") or []
                   if isinstance(entry, dict) and isinstance(entry.get("id"), str))
    headers = {str(key): str(value) for key, value in (profile.get("headers") or {}).items()} \
        if isinstance(profile.get("headers"), dict) else {}
    api, base_url = str(profile.get("api") or ""), str(profile.get("baseURL") or "")
    problem = ""
    if api != SUPPORTED_API:
        problem = f"接口是 {api or 'pi-ai 内置目录'}，ecogym 目前只能转发 {SUPPORTED_API}"
    elif urlparse(base_url).scheme not in ("http", "https") or not urlparse(base_url).hostname:
        problem = "DSH 设置里没有 baseURL"
    elif not models:
        problem = "DSH 设置里没有列出模型"
    elif not profile.get("apiKeyEnv"):
        problem = "DSH 设置里没有 apiKeyEnv"
    return Route(provider, str(profile.get("displayName") or provider), api, base_url,
                 str(profile.get("apiKeyEnv") or ""), models, profile, headers, problem)


def routes() -> dict[str, Route]:
    """deepseek-official and every provider route of the DSH settings, runnable or not."""

    found = {DEEPSEEK: Route(DEEPSEEK, "DeepSeek", "deepseek", DEEPSEEK_URL, DEEPSEEK_KEY, ())}
    for provider, profile in sorted((dsh_settings().get("providers") or {}).items()):
        if provider != DEEPSEEK and PROVIDER.match(str(provider)):
            found[str(provider)] = _route(str(provider), profile)
    return found


def split(ref: str) -> tuple[str, str]:
    """``(provider, model)`` of a reference; a bare DeepSeek id is deepseek-official's."""

    ref = str(ref).strip()
    provider, slash, model = ref.partition("/")
    if not slash:
        if DEEPSEEK_MODEL.match(ref):
            return DEEPSEEK, ref
        raise ModelError(f"model {ref!r}: write <provider>/<model>, e.g. deepseek-official/deepseek-v4-pro")
    if not PROVIDER.match(provider) or not MODEL.match(model):
        raise ModelError(f"model {ref!r} is not <provider>/<model>")
    return provider, model


def resolve(ref: str) -> tuple[Route, str]:
    """The route and model id of a reference that can run here; ``ModelError`` says why not."""

    provider, model = split(ref)
    if provider == DEEPSEEK:
        if not DEEPSEEK_MODEL.match(model):
            raise ModelError(f"{model!r} is not a DeepSeek model id")
        return routes()[DEEPSEEK], model
    route = routes().get(provider)
    if route is None:
        raise ModelError(f"no provider {provider!r} in {settings_path()} (llm-pi-ai.providers)")
    if route.problem:
        raise ModelError(f"{provider}: {route.problem}")
    if model not in route.models:
        raise ModelError(f"{provider} has no model {model!r}; it lists {', '.join(route.models)}")
    return route, model


def check(ref: str) -> str:
    """Fail unless ``ref`` can run now (route, model and key); returns the canonical ``provider/model``."""

    route, model = resolve(ref)
    if not has_secret(route.key_env):
        raise ModelError(f"{route.provider}: no {route.key_env} in the environment or DSH's credential store "
                         "(~/.dsh/.credentials.yaml)")
    return f"{route.provider}/{model}"


PROBE_REFUSED = (401, 403, 404)  # bad key, no access to the model, unknown model: no session could get a reply
PROBE_TOOL = {"type": "function", "function": {"name": "echo", "description": "Repeat a word.",
                                               "parameters": {"type": "object",
                                                              "properties": {"word": {"type": "string"}},
                                                              "required": ["word"]}}}


def _message(text: str) -> str:
    try:
        body = json.loads(text)
        error = body.get("error") if isinstance(body, dict) else None
        found = error.get("message") if isinstance(error, dict) else error if isinstance(error, str) else None
        if found:
            return str(found)[:300]
    except ValueError:
        pass
    return text.strip()[:300]


def probe(ref: str, *, timeout: float = 20.0) -> str:
    """Send the model one tiny request from the host, as the gateway would; ``ModelError`` if it is refused.

    Settings can list a model the key may not use (the provider answers 403 to every call), which only a
    real request reveals. Refusals (401, 403, 404) raise with the provider's message; anything else (rate
    limits, outages, parameters a provider dislikes) passes with a note, since sessions retry those. The
    request carries one tool, as every agent call does: a provider whose tool calling is misconfigured (on
    2026-11-02 one answered 400 "auto tool choice requires --enable-auto-tool-choice" for an hour) shows it
    only then.
    """

    route, model = resolve(ref)
    if not has_secret(route.key_env):
        raise ModelError(f"{route.provider}: no {route.key_env} in the environment or DSH's credential store")
    # streamed: the status arrives before any (possibly slow) reasoning, and the status is all this asks for
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 16,
                       "stream": True, "tools": [PROBE_TOOL]}).encode()
    request = urllib.request.Request(route.base_url.rstrip("/") + "/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json", **route.headers,
                                              "Authorization": f"Bearer {resolve_secret(route.key_env)}"})
    proxy = os.environ.get("ECOGYM_UPSTREAM_PROXY")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {}))
    try:
        with opener.open(request, timeout=timeout):
            pass  # accepted: closing drops the rest of the stream
        return "ok"
    except urllib.error.HTTPError as error:
        detail = _message(error.read(4000).decode("utf-8", "replace"))
        if error.code in PROBE_REFUSED:
            raise ModelError(f"{route.provider}/{model}: the provider refused a test request ({error.code} "
                             f"{detail}); choose another model or check the key's access") from None
        return f"test request answered {error.code}: {detail}"
    except (OSError, ValueError) as error:
        return f"test request failed: {type(error).__name__}: {error}"


def describe() -> dict[str, Any]:
    """Every route for a model picker: names, protocol, endpoint host, key state, models; no secrets."""

    try:
        found, error = routes(), ""
    except ModelError as problem:
        found, error = {DEEPSEEK: Route(DEEPSEEK, "DeepSeek", "deepseek", DEEPSEEK_URL, DEEPSEEK_KEY, ())}, str(problem)
    items = []
    for route in found.values():
        key = bool(route.key_env) and has_secret(route.key_env)
        problem = route.problem or ("" if key else f"缺少 {route.key_env or 'API key'}")
        items.append({"provider": route.provider, "name": route.name, "api": route.api,
                      "host": urlparse(route.base_url).hostname or "", "key_env": route.key_env,
                      "key_configured": key, "runnable": not problem, "problem": problem,
                      "models": list(route.models or DEEPSEEK_MODELS)})
    default = None if error else dsh_settings().get("default")
    return {"routes": items, "dsh_default": default, "settings": str(settings_path()), "error": error}


__all__ = ["DEEPSEEK", "DEEPSEEK_KEY", "DEEPSEEK_URL", "ModelError", "Route", "check", "describe", "dsh_settings",
           "probe", "resolve", "routes", "settings_path", "split"]
