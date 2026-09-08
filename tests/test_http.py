from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
from http import HTTPStatus
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ecologyrsi_dsh.api import handler as handler_module
from ecologyrsi_dsh.api import shared as api_shared
from ecologyrsi_dsh.application.config import bind_toy_dataset
from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
from ecologyrsi_dsh.core.models import Evaluation, TaskManifest
from ecologyrsi_dsh.api.handler import EvolutionHTTPServer
from ecologyrsi_dsh.data.toy import ToyCropSoilWater
from ecologyrsi_dsh.evolution.schedule import (
    LEGACY_SCHEDULE_SCHEMA_VERSION,
    OPTIMIZATION_PROTOCOL,
    PREQUENTIAL_LOCAL_EVALUATION_MODE,
    SCHEDULE_SCHEMA_VERSION,
    ISOLATED_SCHEDULE_SCHEMA_VERSION,
    PAIRED_LOCAL_EVALUATION_MODE,
    OptimizationSchedule,
)
from ecologyrsi_dsh.integrations.dsh_native_runtime import DSH_NATIVE_EXECUTION_PROTOCOL


class _ImmediateNativeControlRuntime:
    def __init__(self) -> None:
        self.paused: list[dict] = []
        self.cancelled: list[dict] = []

    def capabilities(self) -> dict:
        return {"ready": True}

    def require_capabilities(
        self,
        _payload: dict,
        _required: object,
        **_kwargs: object,
    ) -> None:
        return

    def create_run(self, request: dict) -> dict:
        return {"accepted": True, **request}

    def pause(self, request: dict) -> dict:
        self.paused.append(dict(request))
        return {"accepted": True, **request}

    def cancel(self, request: dict) -> dict:
        self.cancelled.append(dict(request))
        return {"accepted": True, **request}

    def resume(self, request: dict) -> dict:
        return {"accepted": True, **request}


class HTTPServerErrorHandlingTests(unittest.TestCase):
    def test_listen_backlog_covers_maximum_sample_concurrency(self) -> None:
        self.assertGreaterEqual(EvolutionHTTPServer.request_queue_size, 128)

    def test_client_disconnect_errors_do_not_reach_default_traceback_handler(self) -> None:
        server = EvolutionHTTPServer.__new__(EvolutionHTTPServer)
        request = object()
        client_address = ("127.0.0.1", 43210)

        with patch.object(ThreadingHTTPServer, "handle_error") as parent_handler:
            for error_type in (
                ConnectionResetError,
                BrokenPipeError,
                ConnectionAbortedError,
            ):
                try:
                    raise error_type("client disconnected")
                except error_type:
                    server.handle_error(request, client_address)

        parent_handler.assert_not_called()

    def test_unexpected_request_error_reaches_default_traceback_handler(self) -> None:
        server = EvolutionHTTPServer.__new__(EvolutionHTTPServer)
        request = object()
        client_address = ("127.0.0.1", 43210)

        with patch.object(ThreadingHTTPServer, "handle_error") as parent_handler:
            try:
                raise RuntimeError("unexpected request failure")
            except RuntimeError:
                server.handle_error(request, client_address)

        parent_handler.assert_called_once_with(request, client_address)


class HTTPContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.server = EvolutionHTTPServer(
            ("127.0.0.1", 0), Path(self._directory.name) / "events.sqlite3"
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self._directory.cleanup()

    def request(self, path: str, method: str = "GET", body: dict | None = None) -> tuple[int, object]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=3) as response:
                raw = response.read()
                if response.headers.get_content_type() == "application/json":
                    return response.status, json.loads(raw)
                return response.status, raw.decode("utf-8")
        except HTTPError as exc:
            raw = exc.read()
            return exc.code, json.loads(raw)

    def test_evolution_capacity_uses_server_cohort_planner_truth(self) -> None:
        status, payload = self.request(
            "/api/evolution-capacity",
            method="POST",
            body={
                "dataset_id": "generated-toy-series@1",
                "episode_id": "generated-toy-series@1:seed-0",
                "optimization_schedule": OptimizationSchedule.default().to_dict(),
                "planned_generations": 5,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["required_unique_origins"], 1665)
        self.assertEqual(
            payload["candidate_origin_executions_per_generation"], 2663
        )
        self.assertEqual(payload["scoring_cells_per_generation"], 23967)
        # The synthetic toy series has no complete 1/6/24-hour origin
        # population; it remains intentionally ineligible for the strict
        # cohort planner even though toy run creation is not capacity-gated.
        self.assertFalse(payload["sufficient"])
        self.assertEqual(payload["cohort_reuse_policy"], "cycle_after_exhaustion@1")
        self.assertGreaterEqual(payload["reused_origin_occurrences"], 0)
        self.assertFalse(payload["capacity_enforced_for_run_creation"])

    def test_create_defaults_to_v3_and_explicit_legacy_schedule_round_trips(self) -> None:
        base = {
            "dataset_id": "generated-toy-series@1",
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "rounds": 1,
            "candidates_per_generation": 4,
            "max_candidates": 4,
            "auto_advance": 0,
        }
        status, created = self.request(
            "/api/runs",
            method="POST",
            body={**base, "idempotency_key": "default-paired-schedule"},
        )

        self.assertEqual(status, 201, created)
        default_schedule = created["projection"]["configuration"][
            "optimization_schedule"
        ]
        self.assertEqual(default_schedule["schema_version"], ISOLATED_SCHEDULE_SCHEMA_VERSION)
        self.assertEqual(
            default_schedule["local_evaluation_mode"],
            PAIRED_LOCAL_EVALUATION_MODE,
        )

        legacy_schedule = OptimizationSchedule.default().to_dict()
        legacy_schedule.update(
            schema_version=LEGACY_SCHEDULE_SCHEMA_VERSION,
            local_evaluation_mode=PREQUENTIAL_LOCAL_EVALUATION_MODE,
        )
        status, created = self.request(
            "/api/runs",
            method="POST",
            body={
                **base,
                "optimization_schedule": legacy_schedule,
                "idempotency_key": "explicit-legacy-schedule",
            },
        )

        self.assertEqual(status, 201, created)
        self.assertEqual(
            created["projection"]["configuration"]["optimization_schedule"],
            legacy_schedule,
        )

    def _seed_test_partition_run(self, *, manifest_partition: str = "validation") -> str:
        """Write a deliberately out-of-scope run directly to the local ledger."""

        manifest = bind_toy_dataset(
            TaskManifest(
                task_id="http-scope-test",
                objective="exercise HTTP scope redaction",
                domain_pack="crop_soil_water",
                visible_datasets=("generated-toy-series@1",),
                budget=1,
                seed=7,
                seed_policy="fixed",
                metadata={"evaluation_partition": manifest_partition},
            ),
            required=True,
        )
        run_id = "run:http-scope-test"
        state = self.server.director.start_evolution(manifest, run_id=run_id)
        if manifest_partition == "validation":
            candidate = self.server.director.propose_and_spawn(run_id)
            current = self.server.director.state(run_id)
            proposal = current.proposal(candidate.proposal_id)
            evaluation = ToyCropSoilWater(seed=manifest.seed).evaluate_candidate(
                run_id,
                candidate,
                proposal,
                split="test",
            )
            self.server.director.evaluate_and_decide(evaluation)
            self.server.director.advance_generation(run_id)
            self.server.director.complete_run(run_id)
        return quote(state.run.run_id, safe="")

    def test_new_run_contract_rejects_samples_per_update(self) -> None:
        status, payload = self.request(
            "/api/runs",
            "POST",
            {
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "samples_per_update": 4500,
                "auto_advance": 0,
                "idempotency_key": "samples-per-update-is-obsolete",
            },
        )

        self.assertEqual(status, 400, payload)
        self.assertIn("samples_per_update is not supported", payload["error"])

    def test_new_run_contract_rejects_legacy_model_workflow(self) -> None:
        status, payload = self.request(
            "/api/runs",
            "POST",
            {
                "dataset_id": "generated-toy-series@1",
                "model_workflow": "legacy_component_search@1",
                "rounds": 1,
                "candidates_per_generation": 1,
                "max_candidates": 1,
                "auto_advance": 0,
            },
        )
        self.assertEqual(status, 400, payload)
        self.assertIn("research_compile_evolve@1", payload["error"])

    def test_projection_cursor_and_generation_step(self) -> None:
        status, html = self.request("/plugins/ecology/evolution/")
        self.assertEqual(status, 200)
        self.assertIn("生态模型进化工作台", html)

        status, plugin_manifest = self.request("/api/plugin/ecology_evolution")
        self.assertEqual(status, 200)
        self.assertEqual(plugin_manifest["display_name"], "生态模型进化工作台")

        status, created = self.request(
            "/api/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "slot": "residual_water_stress",
                "budget": {"max_generations": 2, "max_candidates": 3},
                "auto_advance": 1,
                "idempotency_key": "http-test-run",
            },
        )
        self.assertEqual(status, 201)
        projection = created["projection"]
        self.assertEqual(projection["generation"], 1)
        self.assertEqual(projection["candidates_count"], 1)
        self.assertEqual(projection["task"]["slot"], "residual_water_stress")
        self.assertEqual(projection["seed_policy"], "fixed")
        self.assertEqual(projection["evaluation_partition"], "validation")
        self.assertIs(projection["token_usage_available"], False)
        self.assertTrue(projection["dataset_digest"])

        run_path = quote(projection["run_id"], safe="")
        status, events = self.request(f"/api/runs/{run_path}/events")
        self.assertEqual(status, 200)
        self.assertTrue(events["events"])
        self.assertEqual(events["total_public_events"], len(events["events"]))
        self.assertIs(events["truncated"], False)
        status, tail = self.request(f"/api/runs/{run_path}/events?tail=1")
        self.assertEqual(status, 200)
        self.assertEqual(len(tail["events"]), 1)
        self.assertGreaterEqual(tail["total_public_events"], 1)
        self.assertEqual(
            tail["next_cursor"],
            events["next_cursor"],
        )
        self.assertEqual(
            tail["truncated"],
            tail["total_public_events"] > 1,
        )
        cursor = events["next_cursor"]
        status, no_events = self.request(f"/api/runs/{run_path}/events?after={cursor}")
        self.assertEqual(status, 200)
        self.assertEqual(no_events["events"], [])

        status, stepped = self.request(
            f"/api/runs/{run_path}/advance", "POST", {"steps": 1}
        )
        self.assertEqual(status, 200)
        self.assertEqual(stepped["projection"]["generation"], 2)
        self.assertEqual(stepped["projection"]["status"], "completed")

    def test_compact_manifest_uses_candidate_budget_for_generation_limit(self) -> None:
        status, created = self.request(
            "/api/runs",
            "POST",
            {
                "run_id": "run:compact-budget",
                "task_manifest": {
                    "task_id": "compact-budget",
                    "objective": "exercise compact manifest defaults",
                    "domain_pack": "crop-soil-water@toy",
                    "visible_datasets": ["generated-toy-series@1"],
                    "budget": {"max_candidates": 3},
                    "seed": 7,
                    "seed_policy": "fixed",
                    "policy_version": "policy@1",
                    "metadata": {},
                },
                "auto_advance": 1,
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(created["projection"]["total_generations"], 3)
        self.assertEqual(created["projection"]["generation"], 1)
        self.assertEqual(created["projection"]["status"], "running")

    def test_dataset_only_request_derives_domain_pack(self) -> None:
        status, created = self.request(
            "/api/runs",
            "POST",
            {
                "dataset_id": "generated-toy-series@1",
                "rounds": 1,
                "max_candidates": 1,
                "auto_advance": 0,
                "idempotency_key": "dataset-only-domain-inference",
            },
        )
        self.assertEqual(status, 201, created)
        configuration = created["projection"]["configuration"]
        self.assertEqual(configuration["dataset_id"], "generated-toy-series@1")
        self.assertEqual(configuration["domain_pack_id"], "crop_soil_water")
        self.assertEqual(
            configuration["episode_id"], "generated-toy-series@1:seed-0"
        )

    def test_dataset_and_explicit_domain_mismatch_is_rejected(self) -> None:
        status, rejected = self.request(
            "/api/runs",
            "POST",
            {
                "dataset_id": "generated-toy-series@1",
                "domain": "greenhouse_environment@1",
                "rounds": 1,
            },
        )
        self.assertEqual(status, 400, rejected)
        self.assertIn("领域与数据集不一致", rejected["error"])

    def test_visible_gate_reports_failed_after_a_rejected_evaluation(self) -> None:
        manifest = bind_toy_dataset(
            TaskManifest(
                task_id="visible-gate-failure",
                objective="verify visible gate status",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("generated-toy-series@1",),
                budget=1,
                seed=7,
                metadata={"evaluation_partition": "validation"},
            ),
            required=True,
        )
        run_id = "run:visible-gate-failure"
        self.server.director.start_evolution(manifest, run_id=run_id)
        candidate = self.server.director.propose_and_spawn(run_id)
        self.server.director.evaluate_and_decide(
            Evaluation(
                evaluation_id="evaluation:visible-gate-failure",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                score=-1.0,
                passed=False,
                partition="validation",
                evaluator_digest="toy_time_forward@1",
            )
        )

        status, payload = self.request(
            f"/api/runs/{quote(run_id, safe='')}"
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["projection"]["gate"]["visible"], "未通过")

    def test_generation_only_budget_infers_candidate_limit(self) -> None:
        status, created = self.request(
            "/api/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "budget": {"max_generations": 2},
                "auto_advance": 1,
                "idempotency_key": "generation-only-budget",
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(created["projection"]["total_generations"], 2)
        self.assertEqual(created["projection"]["max_candidates"], 2)
        self.assertEqual(created["projection"]["generation"], 1)
        self.assertEqual(created["projection"]["status"], "running")

    def test_conflicting_budget_reserves_every_generation_slot(self) -> None:
        status, created = self.request(
            "/api/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "rounds": 10,
                "candidates_per_generation": 3,
                "max_candidates": 3,
                "auto_advance": 0,
                "idempotency_key": "complete-generation-budget",
            },
        )

        self.assertEqual(status, 201, created)
        projection = created["projection"]
        self.assertEqual(projection["total_generations"], 10)
        self.assertEqual(projection["candidates_per_generation"], 3)
        self.assertEqual(projection["max_candidates"], 30)
        self.assertEqual(projection["budget"]["max_candidates"], 30)

    def test_plugin_response_has_browser_security_headers(self) -> None:
        with urlopen(self.base + "/plugins/ecology/evolution/", timeout=3) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")

    def test_plugin_manifest_entrypoint_resolves_relative_assets(self) -> None:
        entrypoint = "/plugins/ecology/evolution"
        with urlopen(self.base + entrypoint + "?api=/api/ecology-evolution", timeout=3) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(
                response.geturl(),
                self.base + "/plugins/ecology/evolution/?api=/api/ecology-evolution",
            )
            self.assertIn('href="styles.css"', response.read().decode("utf-8"))

        for asset, content_type in (
            ("styles.css", "text/css"),
            ("app.js", "text/javascript"),
            ("assets/js/host.js", "text/javascript"),
        ):
            with urlopen(
                self.base + f"/plugins/ecology/evolution/{asset}", timeout=3
            ) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers.get_content_type(), content_type)

    def test_canonical_proxy_is_the_only_public_api_base(self) -> None:
        status, health = self.request("/api/ecology-evolution/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["ok"])

        for prefix in ("/api/v1", "/api/ecology-evolution/v1"):
            status, _payload = self.request(prefix + "/health")
            self.assertEqual(status, 404)

        status, manifest = self.request("/api/ecology-evolution/plugin/ecology_evolution")
        self.assertEqual(status, 200)
        self.assertEqual(manifest["recommended_dsh_proxy_base"], "/api/ecology-evolution")
        self.assertEqual(manifest["supported_bases"], ["/api/ecology-evolution"])

    def test_catalog_exposes_only_canonical_workflow(self) -> None:
        status, catalog = self.request("/api/ecology-evolution/catalog")
        self.assertEqual(status, 200)
        self.assertEqual(
            [item["id"] for item in catalog["model_workflows"]],
            ["research_compile_evolve@1"],
        )

    def test_health_live_and_ready_expose_operational_checks(self) -> None:
        status, live = self.request("/health/live")
        self.assertEqual(status, 200, live)
        self.assertEqual(live["status"], "live")
        self.assertTrue(live["ok"])

        status, ready = self.request("/health/ready")
        self.assertEqual(status, 200, ready)
        self.assertEqual(ready["status"], "ready")
        self.assertTrue(ready["ok"])
        self.assertEqual(ready["checks"]["sqlite_integrity"], "ok")
        self.assertTrue(ready["checks"]["plugin_manifest"])

    def test_plugin_root_uses_install_data_directory_as_fallback(self) -> None:
        with (
            patch.dict(os.environ, {"ECOLOGYRSI_PLUGIN_DIR": ""}),
            patch.object(api_shared.Path, "is_dir", return_value=False),
            patch.object(api_shared.sysconfig, "get_path", return_value="/python-data"),
        ):
            root = api_shared._plugin_root()
        self.assertEqual(
            root,
            Path("/python-data/share/ecologyrsi-dsh/plugins/ecology_evolution"),
        )

    def test_http_rejects_out_of_scope_evaluation_from_run_events_and_list(self) -> None:
        run_path = self._seed_test_partition_run()

        for path in (
            f"/api/runs/{run_path}",
            f"/api/runs/{run_path}/events",
            "/api/runs",
        ):
            status, payload = self.request(path)
            self.assertEqual(status, 400, (path, payload))
            self.assertIn("validation partition", payload["error"])

    def test_http_rejects_out_of_scope_task_manifest_partition(self) -> None:
        run_path = self._seed_test_partition_run(manifest_partition="test")

        for path in (
            f"/api/runs/{run_path}",
            f"/api/runs/{run_path}/events",
            "/api/runs",
        ):
            status, payload = self.request(path)
            self.assertEqual(status, 400, (path, payload))
            self.assertIn("validation partition", payload["error"])

    def test_http_rejects_out_of_scope_manifest_before_writing(self) -> None:
        status, payload = self.request(
            "/api/runs",
            "POST",
            {
                "run_id": "run:scope-create-rejected",
                "task_manifest": {
                    "task_id": "scope-create-rejected",
                    "objective": "scope boundary",
                    "domain_pack": "crop_soil_water",
                    "visible_datasets": ["generated-toy-series@1"],
                    "budget": 1,
                    "seed": 7,
                    "seed_policy": "fixed",
                    "policy_version": "policy@1",
                    "metadata": {"evaluation_partition": "test"},
                },
                "auto_advance": 0,
                "idempotency_key": "scope-create-rejected-key",
            },
        )
        self.assertEqual(status, 400, payload)
        self.assertIn("validation", payload["error"])
        self.assertEqual(self.server.ledger.run_ids(), ())
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_duplicate_explicit_run_id_does_not_claim_new_receipt(self) -> None:
        body = {
            "run_id": "run:duplicate-explicit",
            "domain_pack_id": "crop_soil_water",
            "dataset_id": "generated-toy-series@1",
            "budget": 1,
            "auto_advance": 0,
            "idempotency_key": "duplicate-first",
        }
        status, first = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 201, first)

        repeated = dict(body, idempotency_key="duplicate-second")
        status, payload = self.request("/api/runs", "POST", repeated)
        self.assertEqual(status, 400, payload)
        self.assertIn("run already exists", payload["error"])
        self.assertEqual(self.server.ledger.pending_command_keys(), ())
        self.assertIsNone(self.server.ledger.command_receipt("create:duplicate-second"))
        self.assertEqual(self.server.ledger.run_ids(), (first["projection"]["run_id"],))

    def test_compact_manifest_rejects_truncated_integer_inputs(self) -> None:
        base = {
            "domain_pack_id": "crop_soil_water",
            "dataset_id": "generated-toy-series@1",
            "auto_advance": 0,
        }
        for index, value in enumerate((1.5, True, "1")):
            status, payload = self.request(
                "/api/runs",
                "POST",
                dict(base, budget=value, idempotency_key=f"bad-budget-{index}"),
            )
            self.assertEqual(status, 400, payload)
            self.assertIn("budget", payload["error"])

        status, payload = self.request(
            "/api/runs",
            "POST",
            dict(base, budget=1, seed=1.5, idempotency_key="bad-seed"),
        )
        self.assertEqual(status, 400, payload)
        self.assertIn("seed", payload["error"])
        self.assertEqual(self.server.ledger.run_ids(), ())
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_percent_in_run_id_is_decoded_once_for_projection_and_events(self) -> None:
        run_id = "run:percent%marker"
        status, created = self.request(
            "/api/runs",
            "POST",
            {
                "run_id": run_id,
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "budget": 1,
                "auto_advance": 0,
                "idempotency_key": "percent-run",
            },
        )
        self.assertEqual(status, 201, created)
        encoded = quote(run_id, safe="")
        status, projected = self.request(f"/api/runs/{encoded}")
        self.assertEqual(status, 200, projected)
        self.assertEqual(projected["projection"]["run_id"], run_id)
        status, events = self.request(f"/api/runs/{encoded}/events")
        self.assertEqual(status, 200, events)
        self.assertEqual(events["run_id"], run_id)

    def test_monitor_view_is_compact_and_shallow_merge_safe(self) -> None:
        run_id, _created = self._create_running_run_for_boundary("monitor-view")
        encoded = quote(run_id, safe="")
        status, detail = self.request(f"/api/runs/{encoded}")
        self.assertEqual(status, 200, detail)
        status, monitor = self.request(f"/api/runs/{encoded}?view=monitor")
        self.assertEqual(status, 200, monitor)
        self.assertEqual(
            monitor["schema_version"],
            "ecologyrsi-dsh.browser-run-monitor/1",
        )
        projection = monitor["projection"]
        self.assertEqual(projection["run_id"], run_id)
        for required in (
            "projection_revision",
            "updated_at",
            "status",
            "generation",
            "candidates_count",
            "execution_progress",
            "execution_scheduler",
            "adaptive_trajectories",
            "run_wide_usage",
            "dsh_runtime",
        ):
            self.assertIn(required, projection)
        for omitted in ("candidates", "rounds", "trajectory", "training_assets"):
            self.assertNotIn(omitted, projection)
        self.assertLess(len(json.dumps(monitor)), len(json.dumps(detail)))

    def test_control_responses_and_command_receipts_remain_compact(self) -> None:
        run_id, _created = self._create_running_run_for_boundary(
            "large-control-response"
        )
        encoded_run_id = quote(run_id, safe="")
        status, detail = self.request(f"/api/runs/{encoded_run_id}")
        self.assertEqual(status, 200, detail)

        controls = (
            ("pause", "paused", "large-control-pause"),
            ("resume", "running", "large-control-resume"),
        )
        for action, expected_status, idempotency_key in controls:
            with self.subTest(action=action):
                status, payload = self.request(
                    f"/api/runs/{encoded_run_id}/control",
                    "POST",
                    {
                        "action": action,
                        "idempotency_key": idempotency_key,
                    },
                )
                self.assertEqual(status, 200, payload)
                self.assertEqual(
                    payload["schema_version"],
                    "ecologyrsi-dsh.browser-run-control/1",
                )
                self.assertEqual(payload["projection"]["status"], expected_status)
                for omitted in (
                    "candidates",
                    "rounds",
                    "trajectory",
                    "training_assets",
                ):
                    self.assertNotIn(omitted, payload["projection"])
                self.assertLess(len(json.dumps(payload)), len(json.dumps(detail)))

                command_id = f"{run_id}:{idempotency_key}"
                status, receipt = self.request(
                    "/api/commands/" + quote(command_id, safe="")
                )
                self.assertEqual(status, 200, receipt)
                self.assertEqual(receipt["status"], "completed")
                self.assertEqual(
                    receipt["response"]["schema_version"],
                    "ecologyrsi-dsh.browser-run-control/1",
                )
                for omitted in (
                    "candidates",
                    "rounds",
                    "trajectory",
                    "training_assets",
                ):
                    self.assertNotIn(omitted, receipt["response"]["projection"])

    def test_create_command_receipt_remains_compact_after_completion(self) -> None:
        run_id, detail = self._create_running_run_for_boundary(
            "large-create-receipt"
        )
        status, receipt = self.request(
            "/api/commands/" + quote("create:large-create-receipt", safe="")
        )
        self.assertEqual(status, 200, receipt)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["run_id"], run_id)
        self.assertEqual(
            receipt["response"]["schema_version"],
            "ecologyrsi-dsh.browser-run-monitor/1",
        )
        for omitted in (
            "candidates",
            "rounds",
            "trajectory",
            "training_assets",
        ):
            self.assertNotIn(omitted, receipt["response"]["projection"])
        self.assertLess(len(json.dumps(receipt)), len(json.dumps(detail)))

    def test_negative_cursor_and_non_integer_steps_are_rejected_before_claim(self) -> None:
        run_id, _created = self._create_running_run_for_boundary("cursor-step-boundary")
        path = "/api/runs/" + quote(run_id, safe="")
        status, payload = self.request(f"{path}/events?after=-1")
        self.assertEqual(status, 400, payload)
        self.assertIn("non-negative", payload["error"])
        for tail in (0, 501, "many"):
            status, payload = self.request(f"{path}/events?tail={tail}")
            self.assertEqual(status, 400, payload)
            self.assertIn("tail", payload["error"])

        for value, key in ((1.5, "float-steps"), ("1", "string-steps")):
            status, payload = self.request(
                f"{path}/advance",
                "POST",
                {"steps": value, "idempotency_key": key},
            )
            self.assertEqual(status, 400, payload)
            self.assertIn("steps", payload["error"])
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_unrelated_run_event_does_not_leave_failed_control_pending(self) -> None:
        run_id, _created = self._create_running_run_for_boundary(
            "scoped-command-target"
        )
        other_run_id, _other = self._create_running_run_for_boundary(
            "scoped-command-other"
        )

        def fail_after_unrelated_write(_run_id: str) -> None:
            self.server.director.record_evolution_stage(
                other_run_id,
                generation=0,
                stage="proposal",
                status="started",
            )
            raise RuntimeError("injected pause failure")

        key = "scoped-command-error"
        path = "/api/runs/" + quote(run_id, safe="") + "/control"
        with patch.object(
            self.server.director,
            "pause_run",
            side_effect=fail_after_unrelated_write,
        ):
            status, failed = self.request(
                path,
                "POST",
                {"action": "pause", "idempotency_key": key},
            )

        self.assertEqual(status, 400, failed)
        self.assertTrue(failed["retryable_with_same_idempotency_key"])
        self.assertNotIn("command_status", failed)
        self.assertIsNone(self.server.ledger.command_receipt(f"{run_id}:{key}"))
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_model_preflight_wait_does_not_hold_global_mutation_lock(self) -> None:
        lock_available = []
        def probe_preflight(handler, _body):
            def probe():
                acquired = handler.server.mutation_lock.acquire(timeout=.2)
                lock_available.append(acquired)
                if acquired: handler.server.mutation_lock.release()
            thread = threading.Thread(target=probe)
            thread.start()
            thread.join(timeout=1)
            handler._send(HTTPStatus.OK, {"passed": True})
        with patch.object(handler_module.EvolutionRequestHandler, "_model_preflight", new=probe_preflight):
            status, payload = self.request("/api/model-preflight", "POST", {})
        self.assertEqual(status, 200, payload)
        self.assertEqual(lock_available, [True])

    def test_advance_does_not_hold_global_mutation_lock(self) -> None:
        run_id, _created = self._create_running_run_for_boundary(
            "advance-lock-boundary"
        )
        lock_available: list[bool] = []

        def probe_advance(handler, target_run_id, _body, **_kwargs):
            def probe() -> None:
                acquired = handler.server.mutation_lock.acquire(timeout=0.2)
                lock_available.append(acquired)
                if acquired:
                    handler.server.mutation_lock.release()

            thread = threading.Thread(target=probe)
            thread.start()
            thread.join(timeout=1)
            return handler.server.director.state(target_run_id)

        path = "/api/runs/" + quote(run_id, safe="") + "/advance"
        with patch.object(
            handler_module.EvolutionRequestHandler,
            "_advance_run",
            new=probe_advance,
        ):
            status, payload = self.request(path, "POST", {"steps": 1})

        self.assertEqual(status, 200, payload)
        self.assertEqual(lock_available, [True])

    def test_native_control_drain_keeps_unrelated_mutations_responsive(self) -> None:
        class BlockingNativeRuntime:
            def __init__(self) -> None:
                self.entered = threading.Event()

            def capabilities(self) -> dict:
                return {"ready": True}

            def require_capabilities(self, _payload: dict, _required: object, **_kwargs: object) -> None:
                return

            def create_run(self, request: dict) -> dict:
                return {"accepted": True, **request}

            def status(self, request: str) -> dict:
                return {"run_id": request, "status": "running"}

            def pause(self, request: dict) -> dict:
                self.entered.set()
                return {"accepted": True, **request}

            def cancel(self, request: dict) -> dict:
                self.entered.set()
                return {"accepted": True, **request}

            def resume(self, request: dict) -> dict:
                return {"accepted": True, **request}

        for action in ("pause", "cancel"):
            with self.subTest(action=action):
                run_id = f"run:native-{action}-drain-lock"
                runtime = BlockingNativeRuntime()
                self.server.dsh_native_runtime = runtime
                status, created = self.request(
                    "/api/runs",
                    "POST",
                    {
                        "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                        "run_id": run_id,
                        "domain_pack_id": "crop_soil_water",
                        "dataset_id": "generated-toy-series@1",
                        "strategy_model_id": "dsh/strategy",
                        "review_model_id": "dsh/review",
                        "start": True,
                        "auto_advance": 0,
                        "idempotency_key": f"native-{action}-drain-create",
                    },
                )
                self.assertEqual(status, 201, created)
                archivable_run_id = f"run:unrelated-archive-during-{action}"
                status, archived_run = self.request(
                    "/api/runs",
                    "POST",
                    {
                        "run_id": archivable_run_id,
                        "domain_pack_id": "crop_soil_water",
                        "dataset_id": "generated-toy-series@1",
                        "budget": 1,
                        "auto_advance": 0,
                        "idempotency_key": f"unrelated-archive-during-{action}-create",
                    },
                )
                self.assertEqual(status, 201, archived_run)
                status, cancelled = self.request(
                    f"/api/runs/{quote(archivable_run_id, safe='')}/control",
                    "POST",
                    {
                        "action": "cancel",
                        "idempotency_key": f"unrelated-archive-during-{action}-cancel",
                    },
                )
                self.assertEqual(status, 200, cancelled)
                active_generation = self.server.acquire_generation_lease(run_id)
                self.assertIsNotNone(active_generation)

                control_response: list[tuple[int, object]] = []
                control_thread = threading.Thread(
                    target=lambda: control_response.append(
                        self.request(
                            f"/api/runs/{quote(run_id, safe='')}/control",
                            "POST",
                            {
                                "action": action,
                                "idempotency_key": f"native-{action}-drain-control",
                            },
                        )
                    )
                )
                control_thread.start()
                create_response: list[tuple[int, object]] = []
                create_thread: threading.Thread | None = None
                archive_response: list[tuple[int, object]] = []
                archive_thread: threading.Thread | None = None
                resume_response: list[tuple[int, object]] = []
                resume_thread: threading.Thread | None = None
                try:
                    self.assertTrue(runtime.entered.wait(1))
                    receipt = self.server.ledger.command_receipt(
                        f"{run_id}:native-{action}-drain-control"
                    )
                    self.assertIsNotNone(receipt)
                    self.assertEqual(receipt.status, "pending")
                    self.assertTrue(control_thread.is_alive())

                    create_thread = threading.Thread(
                        target=lambda: create_response.append(
                            self.request(
                                "/api/runs",
                                "POST",
                                {
                                    "run_id": f"run:unrelated-create-during-{action}",
                                    "domain_pack_id": "crop_soil_water",
                                    "dataset_id": "generated-toy-series@1",
                                    "budget": 1,
                                    "auto_advance": 0,
                                    "idempotency_key": f"unrelated-create-during-{action}",
                                },
                            )
                        )
                    )
                    create_thread.start()
                    create_thread.join(0.5)
                    self.assertFalse(
                        create_thread.is_alive(),
                        "an unrelated create waited for native control drain",
                    )
                    self.assertEqual(create_response[0][0], 201, create_response)

                    archive_thread = threading.Thread(
                        target=lambda: archive_response.append(
                            self.request(
                                f"/api/runs/{quote(archivable_run_id, safe='')}/archive",
                                "POST",
                                {},
                            )
                        )
                    )
                    archive_thread.start()
                    archive_thread.join(0.5)
                    self.assertFalse(
                        archive_thread.is_alive(),
                        "an unrelated archive waited for native control drain",
                    )
                    self.assertEqual(archive_response[0][0], 200, archive_response)

                    if action == "pause":
                        resume_thread = threading.Thread(
                            target=lambda: resume_response.append(
                                self.request(
                                    f"/api/runs/{quote(run_id, safe='')}/control",
                                    "POST",
                                    {
                                        "action": "resume",
                                        "idempotency_key": "native-pause-drain-resume",
                                    },
                                )
                            )
                        )
                        resume_thread.start()
                        resume_thread.join(0.5)
                        self.assertFalse(
                            resume_thread.is_alive(),
                            "resume control should be rejected while pause is draining",
                        )
                        self.assertEqual(
                            self.server.director.state(run_id).run.status.value,
                            "paused",
                        )
                finally:
                    active_generation.release()
                control_thread.join(2)
                self.assertFalse(control_thread.is_alive())
                self.assertEqual(control_response[0][0], 202, control_response)
                if resume_thread is not None:
                    resume_thread.join(2)
                    self.assertFalse(resume_thread.is_alive())
                    self.assertEqual(resume_response[0][0], 409, resume_response)

    def test_native_pause_reconciles_lifecycle_race_without_orphan_marker(self) -> None:
        for outcome in ("target", "other_safe"):
            with self.subTest(outcome=outcome):
                runtime = _ImmediateNativeControlRuntime()
                self.server.dsh_native_runtime = runtime
                run_id = f"run:native-pause-race-{outcome}"
                status, created = self.request(
                    "/api/runs",
                    "POST",
                    {
                        "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                        "run_id": run_id,
                        "domain_pack_id": "crop_soil_water",
                        "dataset_id": "generated-toy-series@1",
                        "strategy_model_id": "dsh/strategy",
                        "review_model_id": "dsh/review",
                        "start": True,
                        "auto_advance": 0,
                        "idempotency_key": f"native-pause-race-{outcome}-create",
                    },
                )
                self.assertEqual(status, 201, created)
                original_pause = self.server.director.pause_run

                def race_pause(target_run_id: str, **kwargs: object) -> None:
                    if outcome == "target":
                        original_pause(target_run_id, **kwargs)
                    else:
                        self.server.director.cancel_run(target_run_id, "race winner")
                    raise RuntimeError("injected lifecycle race")

                with patch.object(
                    self.server.director,
                    "pause_run",
                    side_effect=race_pause,
                ):
                    status, payload = self.request(
                        f"/api/runs/{quote(run_id, safe='')}/control",
                        "POST",
                        {
                            "action": "pause",
                            "idempotency_key": f"native-pause-race-{outcome}-control",
                        },
                    )

                if outcome == "target":
                    self.assertEqual(status, 202, payload)
                    deadline = time.monotonic() + 2
                    while (
                        self.server.native_control_inflight_for(run_id, runtime)
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.001)
                    self.assertEqual(len(runtime.paused), 1)
                else:
                    # This request owned the only native drain before the
                    # concurrent Host cancellation won. It must still quiesce
                    # DSH instead of merely dropping its marker.
                    self.assertEqual(status, 202, payload)
                    self.assertEqual(
                        self.server.director.state(run_id).run.status.value,
                        "cancelled",
                    )
                    deadline = time.monotonic() + 2
                    while (
                        self.server.native_control_inflight_for(run_id, runtime)
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.001)
                    self.assertEqual(len(runtime.paused), 1)
                self.assertFalse(
                    self.server.native_control_inflight_for(run_id, runtime)
                )

    def test_native_control_thread_launch_failure_uses_inline_takeover(self) -> None:
        runtime = _ImmediateNativeControlRuntime()
        self.server.dsh_native_runtime = runtime
        run_id = "run:native-thread-launch-fallback"
        status, created = self.request(
            "/api/runs",
            "POST",
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": run_id,
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": True,
                "auto_advance": 0,
                "idempotency_key": "native-thread-launch-fallback-create",
            },
        )
        self.assertEqual(status, 201, created)

        with patch.object(
            handler_module.EvolutionRequestHandler,
            "_start_native_control_thread",
            side_effect=RuntimeError("injected thread start failure"),
        ) as start_thread:
            status, paused = self.request(
                f"/api/runs/{quote(run_id, safe='')}/control",
                "POST",
                {
                    "action": "pause",
                    "idempotency_key": "native-thread-launch-fallback-control",
                },
            )

        self.assertEqual(status, 200, paused)
        self.assertTrue(paused["remote_quiesced"])
        self.assertEqual(len(runtime.paused), 1)
        self.assertEqual(start_thread.call_count, 2)
        self.assertFalse(self.server.native_control_inflight_for(run_id, runtime))

    def test_manual_native_pause_retries_transient_drain_before_resume(self) -> None:
        class TransientPauseRuntime:
            def __init__(self) -> None:
                self.run_ids: set[str] = set()
                self.pause_calls = 0
                self.first_failed = threading.Event()
                self.retry_entered = threading.Event()
                self.release_retry = threading.Event()

            def capabilities(self) -> dict:
                return {"ready": True}

            def require_capabilities(
                self,
                _payload: dict,
                _required: object,
                **_kwargs: object,
            ) -> None:
                return

            def create_run(self, request: dict) -> dict:
                self.run_ids.add(request["run_id"])
                return {"accepted": True, **request}

            def status(self, run_id: str) -> dict:
                return {"run_id": run_id, "status": "running"}

            def pause(self, request: dict) -> dict:
                self.pause_calls += 1
                if self.pause_calls == 1:
                    self.first_failed.set()
                    raise DshNativeRuntimeUnavailableError(
                        "temporary pause transport outage",
                        error_code="dsh_native_runtime_transport_error",
                    )
                self.retry_entered.set()
                self.release_retry.wait(timeout=3)
                return {"accepted": True, **request}

            def cancel(self, request: dict) -> dict:
                return {"accepted": True, **request}

            def resume(self, request: dict) -> dict:
                return {"accepted": True, **request}

        runtime = TransientPauseRuntime()
        self.server.dsh_native_runtime = runtime
        run_id = "run:manual-native-transient-pause"
        status, created = self.request(
            "/api/runs",
            "POST",
            {
                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "run_id": run_id,
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_model_id": "dsh/strategy",
                "review_model_id": "dsh/review",
                "start": True,
                "auto_advance": 0,
                "idempotency_key": "manual-native-transient-create",
            },
        )
        self.assertEqual(status, 201, created)

        try:
            with patch(
                "ecologyrsi_dsh.api.auto_progress._NATIVE_QUIESCENCE_RETRY_BASE_SECONDS",
                0.0,
            ):
                pause_status, paused = self.request(
                    f"/api/runs/{quote(run_id, safe='')}/control",
                    "POST",
                    {
                        "action": "pause",
                        "idempotency_key": "manual-native-transient-pause",
                    },
                )
                self.assertEqual(pause_status, 202, paused)
                self.assertTrue(runtime.first_failed.wait(timeout=1))
                self.assertTrue(runtime.retry_entered.wait(timeout=1))
                self.assertTrue(
                    self.server.native_control_inflight_for(run_id, runtime)
                )
                resume_status, resume_payload = self.request(
                    f"/api/runs/{quote(run_id, safe='')}/control",
                    "POST",
                    {
                        "action": "resume",
                        "idempotency_key": "manual-native-resume-too-early",
                    },
                )
                self.assertEqual(resume_status, 409, resume_payload)
        finally:
            runtime.release_retry.set()

        receipt_key = f"{run_id}:manual-native-transient-pause"
        deadline = time.monotonic() + 2
        receipt = None
        while time.monotonic() < deadline:
            receipt = self.server.ledger.command_receipt(receipt_key)
            if receipt is not None and receipt.status == "completed":
                break
            time.sleep(0.001)
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.status, "completed")
        self.assertTrue(receipt.response["remote_quiesced"])
        self.assertEqual(runtime.pause_calls, 2)
        self.assertFalse(
            self.server.native_control_inflight_for(run_id, runtime)
        )

    def test_pause_control_persists_operator_reason_and_code(self) -> None:
        run_id, _created = self._create_running_run_for_boundary(
            "pause-cause-projection"
        )
        path = "/api/runs/" + quote(run_id, safe="") + "/control"
        status, payload = self.request(
            path,
            "POST",
            {
                "action": "pause",
                "reason": "等待上游队列降压",
                "code": "operator_backpressure",
                "idempotency_key": "pause-cause",
            },
        )
        self.assertEqual(status, 200, payload)
        projection = payload["projection"]
        self.assertEqual(projection["status"], "paused")
        self.assertEqual(projection["pause_reason"], "等待上游队列降压")
        self.assertEqual(projection["pause_code"], "operator_backpressure")
        status, events = self.request(f"{path[:-8]}/events")
        self.assertEqual(status, 200, events)
        paused = [item for item in events["events"] if item["kind"] == "RunPaused"]
        self.assertEqual(paused[-1]["payload"]["reason"], "等待上游队列降压")
        self.assertEqual(paused[-1]["payload"]["code"], "operator_backpressure")

    def _create_running_run_for_boundary(self, key: str) -> tuple[str, dict]:
        status, payload = self.request(
            "/api/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "budget": {"max_generations": 2, "max_candidates": 2},
                "auto_advance": 0,
                "idempotency_key": key,
            },
        )
        self.assertEqual(status, 201, payload)
        return payload["projection"]["run_id"], payload


if __name__ == "__main__":
    unittest.main()
