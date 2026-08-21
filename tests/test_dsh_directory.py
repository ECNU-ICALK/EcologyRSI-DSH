from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from ecologyrsi_dsh.integrations.dsh_directory import discover_model_entries
from ecologyrsi_dsh.integrations.model_gateway import ModelGateway


class DshDirectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.settings = root / "settings.yaml"
        self.credentials = root / "credentials.yaml"
        self.settings.write_text(
            """
llm-pi-ai:
  providers:
    secure:
      displayName: Secure DSH
      apiKeyEnv: SECURE_KEY
      api: openai-completions
      baseURL: https://models.example/v1
      role: reviewer
      models:
        - id: glm-5.2
          name: GLM 5.2
          role: proposer
        - id: kimi-k3
    newapi:
      apiKeyEnv: NEWAPI_KEY
      baseURL: http://203.0.113.10/v1
      models:
        - id: blocked
    other-http:
      apiKeyEnv: OTHER_HTTP_KEY
      baseURL: http://198.51.100.20/v1
      models:
        - id: also-blocked
""".lstrip(),
            encoding="utf-8",
        )
        self.credentials.write_text(
            "SECURE_KEY: dsh-secret\n"
            "NEWAPI_KEY: newapi-secret\n"
            "OTHER_HTTP_KEY: other-http-secret\n",
            encoding="utf-8",
        )
        # DSH intentionally keeps these files private.  The adapter fails
        # closed for a looser mode so a copied test fixture cannot leak keys.
        os.chmod(self.settings, 0o600)
        os.chmod(self.credentials, 0o600)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _env(self, **extra: str) -> dict[str, str]:
        return {
            "ECOLOGYRSI_DSH_DISCOVERY": "1",
            "ECOLOGYRSI_DSH_SETTINGS_FILE": str(self.settings),
            "ECOLOGYRSI_DSH_CREDENTIALS_FILE": str(self.credentials),
            **extra,
        }

    def test_discovery_mirrors_secure_dsh_models_and_redacts_at_gateway(self) -> None:
        entries = discover_model_entries(self._env())
        self.assertEqual(
            [item["id"] for item in entries],
            [
                "secure/glm-5.2",
                "secure/kimi-k3",
                "newapi/blocked",
                "other-http/also-blocked",
            ],
        )
        self.assertEqual(entries[0]["token"], "dsh-secret")

        gateway = ModelGateway.from_env(self._env())
        catalog = gateway.catalog()
        self.assertEqual(
            {item["model_id"] for item in catalog},
            {
                "secure/glm-5.2",
                "secure/kimi-k3",
                "newapi/blocked",
                "other-http/also-blocked",
            },
        )
        self.assertNotIn("dsh-secret", json.dumps(catalog))
        self.assertNotIn("newapi-secret", json.dumps(catalog))
        self.assertNotIn("other-http-secret", json.dumps(catalog))

    def test_mapping_call_does_not_read_home_directory_without_opt_in(self) -> None:
        env = {
            "ECOLOGYRSI_DSH_SETTINGS_FILE": str(self.settings),
            "ECOLOGYRSI_DSH_CREDENTIALS_FILE": str(self.credentials),
        }
        self.assertEqual(discover_model_entries(env), [])
        self.assertEqual(ModelGateway.from_env(env).catalog(), [])

    def test_insecure_http_is_blocked_by_default(self) -> None:
        entries = discover_model_entries(self._env())
        for model_id in ("newapi/blocked", "other-http/also-blocked"):
            blocked = next(item for item in entries if item["id"] == model_id)
            self.assertFalse(blocked["directory_available"])
            self.assertEqual(
                blocked["unavailable_reason"]["code"],
                "insecure_http_blocked",
            )
            self.assertNotIn("gateway_url", blocked)

    def test_provider_allowlist_only_enables_exact_provider(self) -> None:
        allowed_env = self._env(
            ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP_PROVIDERS=" ,newapi,unknown, "
        )
        entries = discover_model_entries(allowed_env)
        allowed = next(item for item in entries if item["id"] == "newapi/blocked")
        self.assertTrue(allowed["directory_available"])
        self.assertEqual(allowed["gateway_url"], "http://203.0.113.10/v1")
        self.assertNotIn("unavailable_reason", allowed)
        still_blocked = next(
            item for item in entries if item["id"] == "other-http/also-blocked"
        )
        self.assertFalse(still_blocked["directory_available"])

        catalog = {
            item["model_id"]: item
            for item in ModelGateway.from_env(allowed_env).catalog()
        }
        runnable = catalog["newapi/blocked"]
        self.assertTrue(runnable["directory_available"])
        self.assertTrue(runnable["configured"])
        self.assertFalse(catalog["other-http/also-blocked"]["directory_available"])

    def test_unknown_empty_and_wrong_case_allowlist_items_do_not_enable_http(self) -> None:
        env = self._env(
            ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP_PROVIDERS=" ,unknown,NEWAPI,, "
        )
        entries = {
            item["id"]: item for item in discover_model_entries(env)
        }
        self.assertFalse(entries["newapi/blocked"]["directory_available"])
        self.assertFalse(entries["other-http/also-blocked"]["directory_available"])

    def test_allowlist_does_not_change_https_or_loopback_bindings(self) -> None:
        baseline = {
            item["model_id"]: item["configuration_digest"]
            for item in ModelGateway.from_env(self._env()).catalog()
        }
        https_listed = {
            item["model_id"]: item["configuration_digest"]
            for item in ModelGateway.from_env(
                self._env(
                    ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP_PROVIDERS="secure"
                )
            ).catalog()
        }
        self.assertEqual(
            baseline["secure/glm-5.2"],
            https_listed["secure/glm-5.2"],
        )

        self.settings.write_text(
            """
llm-pi-ai:
  providers:
    loopback-v4:
      baseURL: http://127.0.0.2:8000/v1
      models:
        - id: local-v4
    loopback-v6:
      baseURL: http://[0:0:0:0:0:0:0:1]:8001/v1
      models:
        - id: local-v6
""".lstrip(),
            encoding="utf-8",
        )
        unlisted_entries = discover_model_entries(self._env())
        listed_env = self._env(
            ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP_PROVIDERS=(
                "loopback-v4,loopback-v6"
            )
        )
        listed_entries = discover_model_entries(listed_env)
        self.assertTrue(
            all(item["directory_available"] for item in listed_entries)
        )
        self.assertTrue(
            all(item["allow_insecure_http"] is False for item in listed_entries)
        )
        self.assertEqual(
            {
                item["model_id"]: item["configuration_digest"]
                for item in ModelGateway.from_env(self._env()).catalog()
            },
            {
                item["model_id"]: item["configuration_digest"]
                for item in ModelGateway.from_env(listed_env).catalog()
            },
        )
        self.assertEqual(
            [item["id"] for item in unlisted_entries],
            [item["id"] for item in listed_entries],
        )

    def test_malformed_allowlisted_urls_are_structured_unavailable_items(self) -> None:
        self.settings.write_text(
            """
llm-pi-ai:
  providers:
    embedded-credential:
      baseURL: http://user:pass@203.0.113.10/v1
      models:
        - id: blocked
    query-url:
      baseURL: "http://203.0.113.11/v1?tenant=a"
      models:
        - id: blocked
    fragment-url:
      baseURL: "http://203.0.113.12/v1#section"
      models:
        - id: blocked
    valid-loopback:
      baseURL: http://127.0.0.2:8002/v1
      models:
        - id: runnable
""".lstrip(),
            encoding="utf-8",
        )
        env = self._env(
            ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP_PROVIDERS=(
                "embedded-credential,query-url,fragment-url"
            )
        )
        entries = {item["id"]: item for item in discover_model_entries(env)}
        for model_id in (
            "embedded-credential/blocked",
            "query-url/blocked",
            "fragment-url/blocked",
        ):
            self.assertFalse(entries[model_id]["directory_available"])
            self.assertEqual(
                entries[model_id]["unavailable_reason"]["code"],
                "invalid_gateway_url",
            )
            self.assertNotIn("gateway_url", entries[model_id])
        self.assertTrue(entries["valid-loopback/runnable"]["directory_available"])

        # One malformed provider must not prevent the valid provider or the
        # structured unavailable entries from reaching the redacted catalog.
        catalog = {
            item["model_id"]: item for item in ModelGateway.from_env(env).catalog()
        }
        self.assertEqual(set(catalog), set(entries))
        self.assertTrue(catalog["valid-loopback/runnable"]["directory_available"])
        self.assertFalse(catalog["query-url/blocked"]["directory_available"])

    def test_legacy_global_insecure_http_switch_remains_compatible(self) -> None:
        env = self._env(ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP="1")
        entries = {
            item["id"]: item for item in discover_model_entries(env)
        }
        self.assertTrue(entries["newapi/blocked"]["directory_available"])
        self.assertTrue(entries["other-http/also-blocked"]["directory_available"])

        catalog = {
            item["model_id"]: item for item in ModelGateway.from_env(env).catalog()
        }
        self.assertTrue(catalog["newapi/blocked"]["configured"])
        self.assertTrue(catalog["other-http/also-blocked"]["configured"])

    def test_provider_and_model_role_aliases_are_canonicalized(self) -> None:
        entries = {
            item["id"]: item
            for item in discover_model_entries(self._env())
        }

        self.assertEqual(entries["secure/glm-5.2"]["roles"], ["propose"])
        self.assertEqual(entries["secure/kimi-k3"]["roles"], ["judge"])

        catalog = {
            item["model_id"]: item
            for item in ModelGateway.from_env(self._env()).catalog()
        }
        self.assertEqual(catalog["secure/glm-5.2"]["roles"], ["propose"])
        self.assertEqual(catalog["secure/kimi-k3"]["roles"], ["judge"])
