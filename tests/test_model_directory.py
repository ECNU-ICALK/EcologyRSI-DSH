"""Directory identity and authorization remain separate from native preflight."""
import json
import unittest
from ecologyrsi_dsh.integrations.model_directory import ModelConnection, ModelDirectory, ModelConfigurationError
from ecologyrsi_dsh.integrations.model_bindings import model_supports_role


class ModelDirectoryTests(unittest.TestCase):
    def connection(self, **changes):
        return ModelConnection(**{"model_id": "provider/model", "model": "model",
                                  "gateway_url": "https://models.example/v1", "token": "private-token", **changes})

    def test_catalog_does_not_store_or_expose_credentials(self):
        connection = self.connection()
        directory = ModelDirectory([connection])
        self.assertNotIn("private-token", repr(connection))
        self.assertNotIn("private-token", repr(vars(directory)))
        self.assertNotIn("private-token", json.dumps(directory.catalog()))
        self.assertNotIn("models.example", json.dumps(directory.catalog()))
        self.assertFalse(directory.catalog()[0]["authentication_verified"])
        self.assertTrue(directory.catalog()[0]["configured"])
        directory.catalog()[0]["roles"].clear()
        self.assertEqual(directory.catalog()[0]["roles"], ["propose", "judge"])

    def test_configuration_changes_invalidate_but_secret_rotation_does_not(self):
        first = self.connection()
        self.assertEqual(first.configuration_digest, self.connection(token="rotated").configuration_digest)
        for changes in ({"roles": ("propose",)}, {"native_configuration_digest": "a" * 64}, {"model": "model-2"}):
            self.assertNotEqual(first.configuration_digest, self.connection(**changes).configuration_digest)

    def test_same_model_can_hold_both_roles(self):
        item = ModelDirectory([self.connection()]).catalog()[0]
        self.assertTrue(model_supports_role(item, "propose"))
        self.assertTrue(model_supports_role(item, "judge"))
        restricted = ModelDirectory([self.connection(roles=("propose",))]).catalog()[0]
        self.assertFalse(model_supports_role(restricted, "judge"))

    def test_unsafe_or_ambiguous_connections_fail_closed(self):
        for url in ("http://public.example/v1", "https://user:secret@example.com", "https://example.com?secret=1"):
            with self.assertRaises(ModelConfigurationError):
                self.connection(gateway_url=url)
        with self.assertRaises(ModelConfigurationError):
            ModelDirectory([self.connection(), self.connection()])
        with self.assertRaises(ModelConfigurationError):
            ModelDirectory().configuration_digest("unknown")

    def test_no_direct_network_inference_interface(self):
        directory = ModelDirectory([self.connection()])
        for operation in ("propose", "judge", "research_plan", "sample_decide", "verify_connection"):
            self.assertFalse(hasattr(directory, operation))
