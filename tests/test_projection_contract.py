from __future__ import annotations

import unittest
from types import SimpleNamespace

from ecologyrsi_dsh import TaskManifest
from ecologyrsi_dsh.api.projection import _evolution_evidence_projection, _run_execution_progress
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
        self.assertEqual(full["execution_protocol"], "dsh_native_plugin_evolution@1")

    def test_progress_planned_equals_lifecycle_total(self):
        progress = _run_execution_progress(
            SimpleNamespace(
                task_manifest=SimpleNamespace(metadata={}, max_generations=1, candidates_per_generation=1, max_candidates=1),
                run=SimpleNamespace(status=SimpleNamespace(value="running"), generation=0, created_at="now"),
                events=(),
            ),
            None,
        )
        self.assertEqual(progress["total_candidates"], 1)
        self.assertEqual(progress["completed_candidates"], 0)

    def test_evidence_projection_is_separate_from_compact_monitor(self):
        state = self._state()
        evidence = _evolution_evidence_projection(state)
        self.assertIsInstance(evidence, dict)


if __name__ == "__main__":
    unittest.main()
