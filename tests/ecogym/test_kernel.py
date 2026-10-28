from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ecogym.kernel import ConfigError, Fields, canonical_json, digest
from ecogym.kernel.ledger import Ledger, LedgerError


class CanonicalTests(unittest.TestCase):
    def test_sorted_compact_and_nan_free(self):
        self.assertEqual(canonical_json({"b": 1, "a": [1.5, float("nan")]}), '{"a":[1.5,null],"b":1}')
        self.assertEqual(canonical_json({"x": np.int64(3), "y": np.array([1.0, 2.0])}), '{"x":3,"y":[1.0,2.0]}')

    def test_digest_ignores_key_order(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        self.assertNotEqual(digest({"a": 1}), digest({"a": 1.5}))


class FieldsTests(unittest.TestCase):
    def test_unknown_keys_fail(self):
        fields = Fields({"a": 1, "typo": 2}, "card")
        self.assertEqual(fields.int("a"), 1)
        with self.assertRaises(ConfigError):
            fields.done()

    def test_types_and_bounds(self):
        fields = Fields({"n": 5, "f": 0.5, "s": "x"}, "card")
        with self.assertRaises(ConfigError):
            fields.int("n", hi=4)
        with self.assertRaises(ConfigError):
            fields.int("f")
        self.assertEqual(fields.str("s", choices=("x", "y")), "x")
        self.assertEqual(fields.float("missing", 2.0), 2.0)
        with self.assertRaises(ConfigError):
            fields.str("absent")


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "ledger.sqlite3"

    def tearDown(self):
        self.directory.cleanup()

    def test_append_read_and_verify(self):
        with Ledger(self.path) as ledger:
            ledger.append("run:a", "run_started", {"x": 1})
            ledger.append("run:b", "run_started", {"x": 2})
            ledger.append("run:a", "run_finished", {"ok": True})
            self.assertEqual(ledger.runs(), ["run:a", "run:b"])
            self.assertEqual([event.kind for event in ledger.events("run:a")], ["run_started", "run_finished"])
            self.assertEqual(ledger.verify(), 3)

    def test_tampering_is_detected(self):
        with Ledger(self.path) as ledger:
            ledger.append("run:a", "run_started", {"x": 1})
            ledger.append("run:a", "run_finished", {"ok": True})
        database = sqlite3.connect(self.path)
        database.execute("UPDATE events SET body = '{\"x\":2}' WHERE seq = 1")
        database.commit()
        database.close()
        with Ledger(self.path) as ledger, self.assertRaises(LedgerError):
            ledger.verify()

    def test_exposure_claimed_once(self):
        with Ledger(self.path) as ledger:
            self.assertTrue(ledger.claim_exposure("k", task="t", stage="validation", method="m", run_id="r"))
            self.assertFalse(ledger.claim_exposure("k", task="t", stage="validation", method="m", run_id="r2"))
            self.assertEqual(len(ledger.exposures(method="m")), 1)


if __name__ == "__main__":
    unittest.main()
