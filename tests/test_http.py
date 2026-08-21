from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ecologyrsi_dsh import server as server_module
from ecologyrsi_dsh.api import shared as api_shared
from ecologyrsi_dsh.config import bind_toy_dataset
from ecologyrsi_dsh.models import Evaluation, TaskManifest
from ecologyrsi_dsh.server import EvolutionHTTPServer
from ecologyrsi_dsh.toy import ToyCropSoilWater


class HTTPServerErrorHandlingTests(unittest.TestCase):
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
        with urlopen(self.base + entrypoint + "?api=/api/v1", timeout=3) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(
                response.geturl(),
                self.base + "/plugins/ecology/evolution/?api=/api/v1",
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

    def test_dsh_same_origin_proxy_aliases_resolve(self) -> None:
        for prefix in ("/api/ecology-evolution", "/api/ecology-evolution/v1"):
            status, health = self.request(prefix + "/health")
            self.assertEqual(status, 200)
            self.assertTrue(health["ok"])

            status, manifest = self.request(prefix + "/plugin/ecology_evolution")
            self.assertEqual(status, 200)
            self.assertEqual(manifest["recommended_dsh_proxy_base"], "/api/ecology-evolution")

    def test_plugin_root_uses_install_data_directory_as_fallback(self) -> None:
        with (
            patch.dict(os.environ, {"ECOLOGYRSI_PLUGIN_DIR": ""}),
            patch.object(api_shared.Path, "is_dir", return_value=False),
            patch.object(api_shared.sysconfig, "get_path", return_value="/python-data"),
        ):
            root = server_module._plugin_root()
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

    def test_negative_cursor_and_non_integer_steps_are_rejected_before_claim(self) -> None:
        run_id, _created = self._create_running_run_for_boundary("cursor-step-boundary")
        path = "/api/runs/" + quote(run_id, safe="")
        status, payload = self.request(f"{path}/events?after=-1")
        self.assertEqual(status, 400, payload)
        self.assertIn("non-negative", payload["error"])

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
            server_module.EvolutionRequestHandler,
            "_advance_run",
            new=probe_advance,
        ):
            status, payload = self.request(path, "POST", {"steps": 1})

        self.assertEqual(status, 200, payload)
        self.assertEqual(lock_available, [True])

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
