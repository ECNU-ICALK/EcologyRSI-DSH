from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from ecologyrsi_dsh.core.exposure_registry import ScientificExposureRegistry
from ecologyrsi_dsh.core.ledger import EventLedger, SCHEMA_VERSION


class ScientificExposureRegistryTests(unittest.TestCase):
    def test_reading_selection_reward_before_assessment_never_opens_a_formal_path(self) -> None:
        ledger = EventLedger()
        registry = ScientificExposureRegistry(ledger)
        record = registry.record_adaptive_evidence(
            run_id="run:a",
            evidence_digest="a" * 64,
            fitness_profile_digest="b" * 64,
        )
        self.assertEqual(record["evidence_class"], "exploratory_adaptive_data")
        self.assertFalse(record["formal_confirmation"])
        self.assertIsNone(registry.formal_exposure("a" * 64))
        ledger.close()


if __name__ == "__main__":
    unittest.main()
