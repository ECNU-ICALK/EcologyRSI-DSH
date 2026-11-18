"""How many times one slot of a round runs when its episodes end in service failures or host errors."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ecogym.evolution.loop import OUTAGE_RERUNS, SERVICE_RERUNS, Evolution, EvolveConfig, Snapshot

from .helpers import synthetic_bench


class Gate:
    """A model service whose test request goes unanswered (``down``) or is answered."""

    def __init__(self, *, down: bool) -> None:
        self.down = down
        self.confirmed = 0

    def wait(self, cancel) -> None:
        pass

    def confirm(self, error: str) -> bool:
        self.confirmed += 1
        return self.down


class EpisodeRetryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        bench = synthetic_bench()
        self.task = bench.task
        suite = SimpleNamespace(id="synthetic", train=[bench.task.id], digest="suite",
                                gates=SimpleNamespace(test="t", confidence=0.95), tasks=lambda _: [bench.task])
        with patch("ecogym.evolution.loop.suite_spec", return_value=suite):
            self.evolution = Evolution(EvolveConfig(suite="synthetic", repeats=1, retries=1, concurrency=1),
                                       Path(directory.name) / "ledger.sqlite3")
        self.evolution.benches[bench.task.id] = bench
        self.harness = SimpleNamespace(digest="seed", agent=SimpleNamespace(model=None))

    def run_slot(self, outcomes, gate):
        """One slot whose attempts end as ``outcomes`` say (a status, or an exception the host raised)."""

        configs = []

        def run_episode(config, *args, **kwargs):
            configs.append(config)
            outcome = outcomes[len(configs) - 1]
            if isinstance(outcome, Exception):
                raise outcome
            return {"episode": f"ep:{len(configs)}", "status": outcome, "score": 0.1, "predictions": None,
                    "scored": {}, "tokens": 1, "metered_tokens": 1, "seconds": 1.0,
                    "upstream_error": "503: Service is temporarily unable" if outcome == "infra_failed" else None}

        with (patch("ecogym.evolution.loop.run_episode", side_effect=run_episode),
              patch.object(self.evolution, "_gate", return_value=gate)):
            entry = self.evolution._episode("evo:test", 1, self.harness, "incumbent", None, self.task, 0,
                                            Snapshot(0, {self.task.id: None}), "plan", 0)
        return entry, [config.context["attempt"] for config in configs]

    def test_an_unstable_service_does_not_use_up_the_retries(self):
        """The test request is answered, yet the episodes end infra_failed (2026-11-17/18: pjlab failing most
        calls for hours): the slot runs again instead of being lost after ``retries``."""

        gate = Gate(down=False)
        entry, attempts = self.run_slot(["infra_failed"] * SERVICE_RERUNS + ["finished"], gate)
        self.assertEqual(entry["status"], "finished")
        self.assertEqual(attempts, list(range(SERVICE_RERUNS + 1)))
        self.assertEqual((entry["attempt"], entry["service_reruns"]), (SERVICE_RERUNS, SERVICE_RERUNS))
        self.assertNotIn("upstream_error", entry)
        self.assertEqual(gate.confirmed, SERVICE_RERUNS)

    def test_reruns_after_service_failures_are_bounded(self):
        entry, attempts = self.run_slot(["infra_failed"] * 10, Gate(down=False))
        self.assertEqual(len(attempts), 1 + SERVICE_RERUNS + self.evolution.config.retries)
        self.assertEqual((entry["status"], entry["service_reruns"]), ("infra_failed", SERVICE_RERUNS))
        self.assertIn("503", entry["upstream_error"])

    def test_a_confirmed_outage_reruns_first(self):
        entry, attempts = self.run_slot(["infra_failed"] * (OUTAGE_RERUNS + 1) + ["finished"], Gate(down=True))
        self.assertEqual(entry["status"], "finished")
        self.assertEqual((entry["outage_reruns"], entry["service_reruns"]), (OUTAGE_RERUNS, 1))
        self.assertEqual(attempts, list(range(OUTAGE_RERUNS + 2)))

    def test_host_errors_still_count_against_retries(self):
        entry, attempts = self.run_slot([RuntimeError("judge crashed")] * 3 + ["finished"], Gate(down=False))
        self.assertEqual(entry["status"], "error")
        self.assertEqual(len(attempts), 1 + self.evolution.config.retries)
        self.assertNotIn("service_reruns", entry)


if __name__ == "__main__":
    unittest.main()
