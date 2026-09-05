from __future__ import annotations

import unittest
from types import SimpleNamespace

from ecologyrsi_dsh import TaskManifest
from ecologyrsi_dsh.api.evidence_projection import build_evidence_projection
from ecologyrsi_dsh.api.progress_projection import build_progress_projection
from ecologyrsi_dsh.api.run_projection import build_configuration


class ProjectionContractTests(unittest.TestCase):
    def _state(self):
        task = TaskManifest(
            task_id="contract-task",
            objective="projection contract",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={"max_candidates": 2, "max_generations": 1},
            seed=1,
        )
        return SimpleNamespace(
            task_manifest=task,
            run=SimpleNamespace(run_id="run:contract", status=SimpleNamespace(value="running")),
            events=(),
            candidates=(),
            evaluations=(),
        )

    def test_full_and_summary_configuration_have_identical_keys(self):
        state = self._state()
        full = build_configuration(state.task_manifest, state, profile="full")
        summary = build_configuration(state.task_manifest, state, profile="summary")
        self.assertEqual(set(full), set(summary))

    def test_progress_planned_equals_lifecycle_total(self):
        progress = build_progress_projection(
            SimpleNamespace(
                task_manifest=SimpleNamespace(metadata={}, max_generations=1, candidates_per_generation=1, max_candidates=1),
                run=SimpleNamespace(status=SimpleNamespace(value="running"), generation=0, created_at="now"),
                events=(),
            ),
            admission=None,
        )
        lifecycle = (
            "local_waiting", "admission_waiting", "provider_queued", "provider_active",
            "workflow_running", "persisting", "completed", "failed", "retry_waiting",
            "cancelled", "draining",
        )
        self.assertEqual(progress["planned"], sum(progress.get(key, 0) for key in lifecycle))

    def test_evidence_projection_is_separate_from_compact_monitor(self):
        state = self._state()
        evidence = build_evidence_projection(state)
        self.assertIsInstance(evidence, dict)


if __name__ == "__main__":
    unittest.main()
