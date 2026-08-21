"""Read a DSH local model directory without exposing its credentials.

The DSH web host can display a richer model directory than the standalone
EcologyRSI service.  This adapter mirrors the provider/model entries that are
configured in ``~/.dsh/settings.yaml`` and resolves their credentials from
the DSH credential file or the current process environment.  Only the server
uses the resulting tokens; callers receive ordinary ``ModelGateway`` entries
which are redacted before reaching the browser.

The parser deliberately supports the small YAML subset emitted by DSH and
uses PyYAML when it is already installed.  PyYAML is optional so the runtime
package keeps its zero-dependency installation contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from .gateway_url_policy import GatewayUrlAssessment, assess_gateway_url
from .model_bindings import canonical_model_roles


_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
_UNSAFE_HTTP_VALUES = {"1", "true", "yes", "on", "enabled"}
_KEY_VALUE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*(?P<value>.*)$")


def insecure_http_allowed_for_provider(
    provider: str,
    environ: Mapping[str, str],
) -> bool:
    """Return whether one exact DSH provider may use non-loopback HTTP.

    The provider list is deliberately comma-delimited and exact-match only:
    no prefix, case folding, or wildcard expansion can widen an authorization.
    The legacy all-provider switch remains supported for existing deployments.
    """

    legacy_allow_all = str(
        environ.get("ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP", "")
    ).strip().casefold() in _UNSAFE_HTTP_VALUES
    if legacy_allow_all:
        return True
    allowed_providers = {
        item.strip()
        for item in str(
            environ.get("ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP_PROVIDERS", "")
        ).split(",")
        if item.strip()
    }
    return provider in allowed_providers


def _scalar(value: Any) -> Any:
    """Normalize the scalar forms used in DSH's settings files."""

    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text[0:1] in {"'", '"'}:
        try:
            if text[0] == '"':
                return json.loads(text)
            return text[1:-1].replace("''", "'")
        except (TypeError, ValueError):
            return text.strip("'\"")
    return text


def _secure_text(path: Path) -> str | None:
    """Read a DSH file only when it is private to the current user."""

    try:
        mode = path.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _yaml_load(text: str) -> Any:
    """Use an installed YAML parser, with a narrow fallback for DSH files."""

    try:
        import yaml  # type: ignore[import-not-found]

        return yaml.safe_load(text)
    except (ImportError, AttributeError, TypeError, ValueError):
        return _fallback_settings_load(text)


def _fallback_settings_load(text: str) -> dict[str, Any]:
    """Parse the provider/model subset if PyYAML is unavailable."""

    providers: dict[str, dict[str, Any]] = {}
    in_llm = False
    in_providers = False
    provider_name: str | None = None
    provider: dict[str, Any] | None = None
    model: dict[str, Any] | None = None
    in_models = False

    def flush_model() -> None:
        nonlocal model
        if provider is not None and model and model.get("id"):
            provider.setdefault("models", []).append(model)
        model = None

    def flush_provider() -> None:
        nonlocal provider_name, provider, in_models
        flush_model()
        if provider_name and provider is not None:
            providers[provider_name] = provider
        provider_name = None
        provider = None
        in_models = False

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        content = raw_line.strip()
        if content.startswith("#"):
            continue
        if indent == 0:
            flush_provider()
            in_llm = content.startswith("llm-pi-ai:")
            in_providers = False
            continue
        if not in_llm:
            continue
        if indent == 2 and content == "providers:":
            in_providers = True
            continue
        if not in_providers:
            continue
        if indent == 4 and content.endswith(":") and not content.startswith("-"):
            flush_provider()
            provider_name = content[:-1].strip()
            provider = {"models": []}
            continue
        if provider is None:
            continue
        if indent == 6 and content == "models:":
            flush_model()
            in_models = True
            continue
        if indent == 6:
            match = _KEY_VALUE.match(content)
            if match:
                in_models = False
                provider[match.group("key")] = _scalar(match.group("value"))
            continue
        if in_models and indent == 8 and content.startswith("-"):
            flush_model()
            model = {}
            inline = content[1:].strip()
            match = _KEY_VALUE.match(inline)
            if match:
                model[match.group("key")] = _scalar(match.group("value"))
            continue
        if in_models and indent >= 10 and model is not None:
            match = _KEY_VALUE.match(content)
            if match:
                model[match.group("key")] = _scalar(match.group("value"))
    flush_provider()
    return {"llm-pi-ai": {"providers": providers}}


def _flat_credentials(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        match = _KEY_VALUE.match(raw_line.strip())
        if not match:
            continue
        value = _scalar(match.group("value"))
        if isinstance(value, str) and value:
            values[match.group("key")] = value
    return values


def _discovery_enabled(env: Mapping[str, str], *, real_environment: bool) -> bool:
    value = env.get("ECOLOGYRSI_DSH_DISCOVERY")
    if value is not None:
        return str(value).strip().casefold() not in _FALSE_VALUES
    # Explicit mapping arguments are used by tests and embedding callers; do
    # not unexpectedly read the operator's home directory in that mode.
    return real_environment


def _directory_roles(
    raw_model: Mapping[str, Any] | None,
    raw_provider: Mapping[str, Any],
) -> list[str]:
    raw_roles: Any = None
    if raw_model is not None:
        raw_roles = raw_model.get("roles", raw_model.get("role"))
    if raw_roles in (None, ""):
        raw_roles = raw_provider.get("roles", raw_provider.get("role"))
    if raw_roles in (None, ""):
        return ["propose", "judge"]
    if isinstance(raw_roles, str):
        text = raw_roles.strip().strip("[]")
        raw_roles = [item.strip().strip("'\"") for item in text.split(",")]
    if not isinstance(raw_roles, (list, tuple)):
        return ["propose", "judge"]
    try:
        return list(canonical_model_roles(raw_roles))
    except ValueError:
        # DSH role metadata is descriptive and older hosts used unrelated
        # names.  Preserve the historical shared-directory behavior instead
        # of silently dropping the model from both EcologyRSI selectors.
        return ["propose", "judge"]


def _provider_unavailable_reason(
    base_url: str,
    api: str,
    *,
    allow_insecure_http: bool,
    assessment: GatewayUrlAssessment | None = None,
) -> dict[str, str] | None:
    url_assessment = assessment or assess_gateway_url(
        base_url,
        allow_insecure_http=allow_insecure_http,
    )
    if url_assessment.reason_code == "missing_gateway_url":
        return url_assessment.directory_reason
    if api not in {"openai-completions", "openai-chat", "openai"}:
        return {
            "code": "unsupported_provider_api",
            "message": f"The DSH provider API {api or 'unknown'} is not OpenAI-compatible.",
        }
    return url_assessment.directory_reason


def discover_model_entries(
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return DSH-configured OpenAI-compatible model entries.

    Unsafe non-loopback HTTP endpoints remain visible as unavailable directory
    entries unless the operator adds the exact provider ID to
    ``ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP_PROVIDERS``.  The legacy global switch
    remains compatible but is broader than the recommended provider allowlist.
    This keeps the gateway transport invariant while explaining why a
    DSH-registered model cannot be executed.
    """

    env = os.environ if environ is None else environ
    if not _discovery_enabled(env, real_environment=environ is None):
        return []
    settings_path = Path(
        env.get("ECOLOGYRSI_DSH_SETTINGS_FILE", "~/.dsh/settings.yaml")
    ).expanduser()
    credentials_path = Path(
        env.get("ECOLOGYRSI_DSH_CREDENTIALS_FILE", "~/.dsh/.credentials.yaml")
    ).expanduser()
    settings_text = _secure_text(settings_path)
    if settings_text is None:
        return []
    credentials_text = _secure_text(credentials_path) or ""
    credentials = _flat_credentials(credentials_text)
    loaded = _yaml_load(settings_text)
    if not isinstance(loaded, Mapping):
        return []
    llm = loaded.get("llm-pi-ai")
    providers = llm.get("providers") if isinstance(llm, Mapping) else None
    if not isinstance(providers, Mapping):
        return []
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for provider_id, raw_provider in providers.items():
        if not isinstance(raw_provider, Mapping):
            continue
        provider = str(provider_id).strip()
        base_url = str(
            raw_provider.get("baseURL", raw_provider.get("base_url", ""))
        ).strip()
        api = str(raw_provider.get("api", "openai-completions")).strip().casefold()
        models = raw_provider.get("models", [])
        if not provider or not isinstance(models, list):
            continue
        provider_http_allowed = insecure_http_allowed_for_provider(provider, env)
        url_assessment = assess_gateway_url(
            base_url,
            allow_insecure_http=provider_http_allowed,
        )
        unavailable_reason = _provider_unavailable_reason(
            base_url,
            api,
            allow_insecure_http=provider_http_allowed,
            assessment=url_assessment,
        )
        key_env = str(
            raw_provider.get("apiKeyEnv", raw_provider.get("api_key_env", ""))
        ).strip()
        token = str(env.get(key_env, "")).strip() if key_env else ""
        if not token and key_env:
            token = credentials.get(key_env, "")
        display_provider = str(
            raw_provider.get("displayName", raw_provider.get("display_name", provider))
        ).strip() or provider
        for raw_model in models:
            if isinstance(raw_model, str):
                model_id = raw_model.strip()
                model_name = model_id
            elif isinstance(raw_model, Mapping):
                model_id = str(raw_model.get("id", raw_model.get("model", ""))).strip()
                model_name = str(raw_model.get("name", model_id)).strip() or model_id
            else:
                continue
            if not model_id:
                continue
            entry_id = f"{provider}/{model_id}"
            if entry_id in seen:
                continue
            seen.add(entry_id)
            entry: dict[str, Any] = {
                "id": entry_id,
                "provider": provider,
                "label": f"{display_provider} · {model_name}",
                "model": model_id,
                "roles": _directory_roles(
                    raw_model if isinstance(raw_model, Mapping) else None,
                    raw_provider,
                ),
                "directory_available": unavailable_reason is None,
            }
            if unavailable_reason is None:
                entry["gateway_url"] = url_assessment.url
                entry["allow_insecure_http"] = (
                    url_assessment.insecure_http_exception
                )
            else:
                entry["unavailable_reason"] = unavailable_reason
            if token:
                entry["token"] = token
            entries.append(entry)
    return entries
