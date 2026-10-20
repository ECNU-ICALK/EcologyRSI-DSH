"""Public plugin discovery must describe the live, single create contract."""

from http import HTTPStatus
from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.api.handler import EvolutionRequestHandler, PLUGIN_MANIFEST
from ecologyrsi_dsh.core.prediction_policy import (
    CELLWISE_EVALUATOR_ID, RUNTIME_PREDICTION_POLICY, TARGETED_SEARCH_POLICY,
)
from ecologyrsi_dsh.core.search_policy import SEARCH_GUARD_POLICY
from ecologyrsi_dsh.evolution.novelty import CANDIDATE_NOVELTY_POLICY
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule


class PluginManifestContractTests(unittest.TestCase):
    def setUp(self) -> None:
        # Only exercise request parsing; runtime binding performs data/model I/O.
        self.handler = SimpleNamespace(_bind_runtime_task=lambda task, **_kwargs: task)
        self.required = {
            "dataset_id": "greenhouse-public-fixture",
            "execution_protocol": "dsh_native_plugin_evolution@1",
            "strategy_model_id": "provider/same-model",
            "review_model_id": "provider/same-model",
        }

    def parse(self, body: dict):
        return EvolutionRequestHandler._task_from_request(self.handler, body)

    def test_required_fields_form_a_parseable_request_and_are_actually_required(self) -> None:
        self.assertEqual(set(PLUGIN_MANIFEST["request_schema"]["required"]), set(self.required))
        self.assertEqual(self.parse(self.required).dataset, self.required["dataset_id"])
        for field in self.required:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.parse({key: value for key, value in self.required.items() if key != field})

    def test_every_advertised_optional_field_is_accepted_without_legacy_aliases(self) -> None:
        optional = {
            "episode_id": "episode-public-fixture",
            "optimization_protocol": "joint_plugin_epoch@2",
            "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
            "autonomous_mode": True,
            "rounds": 1,
            "budget": {"max_generations": 1, "candidates_per_generation": 4, "max_candidates": 4},
            "candidates_per_generation": 4,
            "max_candidates": 4,
            "model_workflow": "research_compile_evolve@1",
            "knowledge_online_enabled": True,
            "candidate_concurrency": 2,
            "sample_concurrency": 4,
            "sample_agent_batch_size": 1,
            "seed_policy": "fixed",
            "seed": 0,
            "requested_mode": "evolution",
            "auto_advance": "continuous",
            "auto_progress": True,
            "start": True,
            "idempotency_key": "public-fixture",
            "run_id": "run:public-fixture",
            "task_id": "task:public-fixture",
            "objective": "greenhouse multihorizon prediction",
            "prediction_selection_policy": RUNTIME_PREDICTION_POLICY,
            "targeted_search_policy": TARGETED_SEARCH_POLICY,
            "prediction_model_id": "greenhouse-baseline-aligned-ridge@2",
            "evaluator_id": CELLWISE_EVALUATOR_ID,
            "seed_genome_template_id": "greenhouse-recipe-default@2",
            "search_guard_policy": SEARCH_GUARD_POLICY,
            "require_model_contract_preflight": True,
            "allow_host_fallback": False,
            "research_domain": "greenhouse_environment@1",
            "candidate_novelty_policy": CANDIDATE_NOVELTY_POLICY,
        }
        advertised = PLUGIN_MANIFEST["request_schema"]["optional"]
        self.assertEqual(len(advertised), len(set(advertised)))
        self.assertEqual(set(advertised), set(optional))
        self.assertFalse(set(advertised) & set(self.required))
        for field, value in optional.items():
            with self.subTest(field=field):
                self.parse({**self.required, "rounds": 1, field: value})
        self.parse({**self.required, **optional})
        for field in ("domain_pack_id", "domain", "dataset", "task_manifest"):
            with self.subTest(retired=field), self.assertRaisesRegex(ValueError, "unsupported create fields"):
                self.parse({**self.required, field: "legacy"})

    def test_retired_manual_routes_are_neither_advertised_nor_dispatched(self) -> None:
        self.assertNotIn("write_intervention", PLUGIN_MANIFEST["capabilities"])
        self.assertNotIn("expert_consultation_answer", PLUGIN_MANIFEST)
        responses = []
        handler = SimpleNamespace(_send=lambda status, body: responses.append((status, body)))
        for path in (
            ["runs", "run:fixture", "expert-consultations", "consultation:fixture", "answer"],
            ["runs", "run:fixture", "interventions"],
            ["runs", "run:fixture", "advance"],
        ):
            EvolutionRequestHandler._dispatch_post(handler, path, {})
            self.assertEqual(responses[-1][0], HTTPStatus.NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
