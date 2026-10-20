from __future__ import annotations

import unittest
from dataclasses import replace

from tests.native_fixtures import native_manifest, start_native_run, NativeRoleStub
from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.api.projection import _evolution_evidence_projection, _run_execution_progress
from ecologyrsi_dsh.api.run_projection import build_configuration


class ProjectionContractTests(unittest.TestCase):
    def _state(self):
        task = native_manifest(task_id="contract-task")
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            return start_native_run(director, task=task, run_id="run:contract")

    def test_full_and_summary_configuration_have_identical_keys(self):
        state = self._state()
        full = build_configuration(state.task_manifest, state, profile="full")
        summary = build_configuration(state.task_manifest, state, profile="summary")
        self.assertEqual(set(full), set(summary))
        self.assertEqual(full["execution_protocol"], "dsh_native_plugin_evolution@1")

    def test_progress_uses_frozen_round_budget_without_inventing_completion(self):
        state = self._state()
        progress = _run_execution_progress(state)
        self.assertEqual(progress["total_generations"], state.task_manifest.max_generations)
        self.assertEqual(progress["completed_generations"], 0)
        self.assertIsNone(progress["overall_progress_percent"])
        self.assertEqual(progress["phase"], "queued")

    def test_feature_activation_comes_only_from_frozen_metadata(self):
        state = self._state()
        keys = ("targeted_search_policy", "behavior_diagnostics_policy", "candidate_novelty_policy")
        old_task = state.task_manifest
        for profile in ("full", "summary"):
            historical = build_configuration(old_task, state, profile)
            self.assertTrue(all(historical[key] is None for key in keys))
        policies = dict(zip(keys, ("cellwise_training_evidence@1", "prediction_contribution@1", "run_behavior_novelty@1")))
        new_task = replace(old_task, metadata={**old_task.metadata, **policies})
        self.assertEqual({key: build_configuration(new_task, state)[key] for key in keys}, policies)
        self.assertFalse(any(key in old_task.metadata for key in keys))

    def test_evidence_projection_is_separate_from_compact_monitor(self):
        state = self._state()
        evidence = _evolution_evidence_projection(state)
        self.assertIsInstance(evidence, dict)


if __name__ == "__main__":
    unittest.main()
