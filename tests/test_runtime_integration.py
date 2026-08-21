from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ecologyrsi_dsh.integrations.model_gateway import GatewayResponseError
from ecologyrsi_dsh.server import EvolutionHTTPServer


class _AuthenticatedModelStubHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append(  # type: ignore[attr-defined]
            {
                "authorization": self.headers.get("Authorization"),
                "path": self.path,
                "body": body,
            }
        )
        payload = self.server.responses.pop(0)  # type: ignore[attr-defined]
        encoded = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


class RuntimeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.directory.name) / "events.sqlite3"
        self.server = EvolutionHTTPServer(
            ("127.0.0.1", 0), self.db_path
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.directory.cleanup()

    def request(
        self, path: str, method: str = "GET", body: dict | None = None
    ) -> tuple[int, dict]:
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self.base + path,
            data=encoded,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_catalog_and_training_page_are_redacted_and_partition_bounded(self) -> None:
        status, catalog = self.request("/catalog")
        self.assertEqual(status, 200)
        # The shared DSH directory is distinct from the two local fallback
        # implementations.  Keep this contract explicit so a browser cannot
        # accidentally present host components as remote API choices.
        self.assertIn("dsh_models", catalog)
        self.assertIn("authenticated_models", catalog)
        self.assertIn("dsh_strategy_models", catalog)
        self.assertIn("dsh_review_models", catalog)
        if not catalog["dsh_models"]:
            self.assertEqual(catalog["authenticated_models"], [])
            self.assertEqual(catalog["dsh_strategy_models"], [])
            self.assertEqual(catalog["dsh_review_models"], [])
        self.assertTrue(
            all(item.get("model_source") == "dsh_gateway" for item in catalog["dsh_models"])
        )
        self.assertTrue(
            all(item.get("local_model") is not True for item in catalog["dsh_models"])
        )
        self.assertIn("generated-toy-series@1", {item["id"] for item in catalog["datasets"]})
        self.assertEqual(
            [item["id"] for item in catalog["models"][:2]],
            ["host_parameter_generator@1", "rule_judge@1"],
        )
        self.assertEqual(catalog["models"][0]["authentication_state"], "local")
        self.assertEqual(catalog["models"][1]["authentication_state"], "local")
        self.assertNotIn("晋级", catalog["models"][1]["description"])
        predictors = {item["id"]: item for item in catalog["prediction_models"]}
        self.assertIn("toy-rolling-water@1", predictors)
        self.assertEqual(
            predictors["toy-rolling-water@1"]["dataset_ids"],
            ["generated-toy-series@1"],
        )
        evaluators = {item["id"]: item for item in catalog["evaluators"]}
        if "greenhouse_multihorizon_time_forward@1" in evaluators:
            self.assertEqual(
                evaluators["greenhouse_multihorizon_time_forward@1"][
                    "prediction_task_count"
                ],
                9,
            )
            self.assertEqual(
                evaluators["greenhouse_multihorizon_time_forward@1"][
                    "minimum_samples_per_update"
                ],
                9,
            )
        self.assertEqual(
            evaluators["toy_time_forward@1"]["prediction_task_count"],
            1,
        )
        self.assertEqual(
            evaluators["toy_time_forward@1"]["minimum_samples_per_update"],
            1,
        )
        serialized = json.dumps(catalog)
        self.assertNotIn("api_key_env", serialized)
        self.assertNotIn("token", serialized.casefold())

        status, page = self.request(
            "/datasets/generated-toy-series%401?partition=training_fit&offset=0&limit=3"
        )
        self.assertEqual(status, 200)
        self.assertEqual(page["partition"], "training_fit")
        self.assertEqual(len(page["page"]["rows"]), 3)
        self.assertEqual(page["schema"][0]["label"], "时间索引")

        for partition in ("development", "gate", "external", "hidden", "test", "final"):
            status, payload = self.request(
                "/datasets/generated-toy-series%401?partition=" + partition
            )
            self.assertEqual(status, 403, (partition, payload))

    def test_frozen_configuration_artifacts_trajectory_and_intervention(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_id": "parameter_sweep@1",
                "evaluator_id": "toy_time_forward@1",
                "policy_model_id": "host_parameter_generator@1",
                "judge_model_id": "rule_judge@1",
                "slot": "bounded_residual_predictor",
                "budget": {"max_generations": 2, "max_candidates": 2},
                "auto_advance": 1,
                "idempotency_key": "integrated-create",
            },
        )
        self.assertEqual(status, 201, created)
        projection = created["projection"]
        self.assertEqual(projection["configuration"]["strategy_id"], "parameter_sweep@1")
        self.assertEqual(
            projection["configuration"]["prediction_model_id"],
            "toy-rolling-water@1",
        )
        self.assertEqual(
            projection["configuration"]["evaluator_id"], "toy_time_forward@1"
        )
        self.assertEqual(projection["configuration"]["judge_model_id"], "rule_judge@1")
        self.assertEqual(len(projection["configuration"]["strategy_digest"]), 64)
        self.assertEqual(
            len(projection["configuration"]["prediction_model_digest"]), 64
        )
        self.assertEqual(len(projection["configuration"]["evaluator_digest"]), 64)
        self.assertEqual(len(projection["configuration"]["policy_model_digest"]), 64)
        self.assertEqual(len(projection["configuration"]["judge_model_digest"]), 64)
        self.assertEqual(len(projection["artifacts"]), 1)
        self.assertEqual(len(projection["trajectory"]), 1)
        self.assertEqual(len(projection["training_assets"]), 1)
        self.assertEqual(len(projection["rounds"]), 1)
        first_asset = projection["training_assets"][0]
        self.assertEqual(
            first_asset["schema_version"],
            "ecologyrsi-dsh.evolution-training-sample/1",
        )
        self.assertFalse(first_asset["admission"]["formal_training_ready"])
        self.assertTrue(first_asset["admission"]["requires_governance_review"])
        self.assertEqual(first_asset["output"]["artifact"]["training_partition"], "train")
        first_episode = first_asset["episode"]
        self.assertEqual(
            first_episode["reproducibility"]["policy_model_digest"],
            projection["configuration"]["policy_model_digest"],
        )
        self.assertEqual(
            first_episode["reproducibility"]["judge_model_digest"],
            projection["configuration"]["judge_model_digest"],
        )
        self.assertEqual(first_episode["stages"]["training"]["status"], "completed")
        self.assertEqual(
            first_episode["stages"]["training"]["artifact_digest"],
            first_asset["output"]["artifact"]["artifact_digest"],
        )
        self.assertEqual(first_episode["stages"]["evaluation"]["status"], "completed")
        self.assertTrue(first_episode["event_receipts"])
        self.assertEqual(
            projection["rounds"][0]["stages"],
            {
                "proposal": "completed",
                "candidate": "completed",
                "training": "completed",
                "evaluation": "completed",
                "judge": "completed",
                "decision": "completed",
            },
        )
        run_path = "/runs/" + quote(projection["run_id"], safe="")

        status, _paused = self.request(run_path + "/control", "POST", {"action": "pause"})
        self.assertEqual(status, 200)
        intervention = {
            "kind": "parameter_override",
            "message": "将平滑系数固定为人工审定值。",
            "created_by": "集成测试研究员",
            "parameter_overrides": {"alpha": 0.42},
            "idempotency_key": "integrated-human-input",
        }
        status, recorded = self.request(run_path + "/interventions", "POST", intervention)
        self.assertEqual(status, 201, recorded)
        self.assertEqual(recorded["projection"]["interventions"][0]["status"], "等待下一轮")
        status, replayed = self.request(run_path + "/interventions", "POST", intervention)
        self.assertEqual(status, 200)
        self.assertEqual(replayed, recorded)

        status, _resumed = self.request(run_path + "/control", "POST", {"action": "resume"})
        self.assertEqual(status, 200)
        status, advanced = self.request(run_path + "/advance", "POST", {"steps": 1})
        self.assertEqual(status, 200, advanced)
        projection = advanced["projection"]
        self.assertEqual(projection["status"], "completed")
        self.assertEqual(projection["candidates"][0]["changes"]["alpha"], 0.42)
        projected_intervention = projection["interventions"][0]
        self.assertEqual(projected_intervention["status"], "已强制执行")
        self.assertEqual(projected_intervention["application_status"], "enforced")
        self.assertTrue(projected_intervention["recorded"])
        self.assertTrue(projected_intervention["applied"])
        self.assertTrue(projected_intervention["enforced"])
        self.assertEqual(projected_intervention["result_values"], {"alpha": 0.42})
        self.assertEqual(len(projection["artifacts"]), 2)
        self.assertEqual(len(projection["trajectory"]), 2)
        self.assertEqual(len(projection["training_assets"]), 2)
        self.assertEqual(len(projection["rounds"]), 2)
        self.assertEqual(projection["training_assets"][0], first_asset)
        second_asset = projection["training_assets"][1]
        self.assertEqual(second_asset["input"]["parent_parameters"]["alpha"], 0.2)
        self.assertEqual(
            second_asset["input"]["applied_interventions"][0]["kind"],
            "parameter_override",
        )
        self.assertEqual(
            second_asset["input"]["applied_interventions"][0]["parameter_overrides"],
            {"alpha": 0.42},
        )
        self.assertEqual(
            second_asset["input"]["applied_interventions"][0]["application_status"],
            "enforced",
        )
        self.assertTrue(
            second_asset["input"]["applied_interventions"][0]["enforced"]
        )
        self.assertEqual(projection["rounds"][1]["applied_intervention_count"], 1)
        self.assertEqual(
            projection["rounds"][1]["parent_candidate_id"],
            projection["training_assets"][0]["candidate_id"],
        )
        serialized_assets = json.dumps(projection["training_assets"], ensure_ascii=False)
        for redline in ("prediction_preview", '"rows"', "private_reasoning"):
            self.assertNotIn(redline, serialized_assets)

        status, events = self.request(run_path + "/events")
        self.assertEqual(status, 200)
        self.assertTrue(events["events"])
        self.assertNotIn("task_manifest", json.dumps(events))
        application_event = next(
            item
            for item in events["events"]
            if item["type"] == "intervention.applied"
        )
        self.assertEqual(application_event["payload"]["application_status"], "enforced")
        self.assertEqual(
            application_event["payload"]["message"],
            "人工意见已由宿主边界强制执行。",
        )
        self.assertEqual(application_event["payload"]["result_values"], {"alpha": 0.42})
        promotion_events = {
            item["payload"]["candidate_id"]: item["payload"]
            for item in events["events"]
            if item["type"] == "promotion.decided"
        }
        for candidate in projection["candidates"]:
            promotion = candidate.get("promotion")
            if promotion is None:
                continue
            event_payload = promotion_events[candidate["candidate_id"]]
            self.assertEqual(event_payload["decision"], promotion["decision"])
            self.assertEqual(event_payload["reason"], promotion["reason"])
            if promotion["decision"] == "rejected":
                self.assertNotIn("保留为搜索候选", promotion["reason"])

    def test_creation_rejects_incompatible_prediction_model_before_writing(self) -> None:
        status, payload = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_id": "parameter_sweep@1",
                "prediction_model_id": "greenhouse-exogenous-ridge@1",
                "evaluator_id": "toy_time_forward@1",
                "budget": 1,
                "auto_advance": 0,
                "idempotency_key": "incompatible-prediction-binding",
            },
        )

        self.assertEqual(status, 400, payload)
        self.assertIn("incompatible", payload["error"])
        self.assertEqual(self.server.ledger.run_ids(), ())

    def test_decision_failure_is_recorded_as_a_visible_stage_event(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "budget": 1,
                "auto_advance": 0,
                "idempotency_key": "decision-stage-failure",
            },
        )
        self.assertEqual(status, 201, created)
        run_path = "/runs/" + quote(created["projection"]["run_id"], safe="")

        with patch.object(
            self.server.director,
            "decide_promotion",
            side_effect=RuntimeError(
                "决策写入测试失败；Bearer decision-audit-secret-token"
            ),
        ):
            status, rejected = self.request(
                run_path + "/advance", "POST", {"steps": 1}
            )
        self.assertEqual(status, 400, rejected)

        status, events = self.request(run_path + "/events")
        self.assertEqual(status, 200, events)
        failed = [
            item
            for item in events["events"]
            if item["type"] == "stage.recorded"
            and item["payload"].get("stage") == "decision"
            and item["payload"].get("status") == "failed"
        ]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["payload"]["public_error"], "RuntimeError")
        self.assertNotIn("decision-audit-secret-token", json.dumps(events))

    @unittest.skipUnless(
        os.environ.get("ECOLOGYRSI_TEST_REAL_DATA") == "1",
        "需设置 ECOLOGYRSI_TEST_REAL_DATA=1",
    )
    def test_real_greenhouse_run_uses_training_feedback(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "greenhouse_environment@1",
                "dataset_id": "agc_cucumber_2018",
                "episode_id": "agc_cucumber_2018:Croperators",
                "strategy_id": "parameter_sweep@1",
                "policy_model_id": "host_parameter_generator@1",
                "judge_model_id": "rule_judge@1",
                "slot": "bounded_residual_predictor",
                "budget": {"max_generations": 1, "max_candidates": 1},
                "auto_advance": 1,
                "idempotency_key": "integrated-real-greenhouse",
            },
        )
        self.assertEqual(status, 201, created)
        projection = created["projection"]
        self.assertEqual(projection["status"], "completed")
        self.assertEqual(
            projection["dataset"]["episode_id"],
            "agc_cucumber_2018:Croperators",
        )
        self.assertEqual(projection["evaluation_partition"], "training_feedback")
        self.assertEqual(projection["artifacts"][0]["training_partition"], "training_fit")
        self.assertGreater(projection["candidates"][0]["metrics"]["n"], 100)
        self.assertFalse(projection["candidates"][0]["metrics"]["causal_interpretation"])
        asset = projection["training_assets"][0]
        self.assertEqual(asset["input"]["episode_id"], "agc_cucumber_2018:Croperators")
        self.assertEqual(asset["output"]["model_id"], "greenhouse-exogenous-ridge@1")
        self.assertEqual(
            {
                item["horizon_hours"]
                for item in projection["candidates"][0]["metrics"]["horizons"]
            },
            {1, 6, 24},
        )
        self.assertGreater(
            projection["execution_diagnostics"]["training_used_examples"], 1000
        )
        self.assertEqual(asset["evaluation"]["partition"], "training_feedback")
        self.assertEqual(asset["evaluation"]["judge"]["model_id"], "rule_judge@1")
        self.assertNotIn("prediction_preview", json.dumps(asset))
        self.assertEqual(projection["rounds"][0]["stages"]["training"], "completed")


class AuthenticatedModelRuntimeTests(RuntimeIntegrationTests):
    def setUp(self) -> None:
        self.model_server = ThreadingHTTPServer(
            ("127.0.0.1", 0), _AuthenticatedModelStubHandler
        )
        self.model_server.requests = []  # type: ignore[attr-defined]
        self.model_server.responses = []  # type: ignore[attr-defined]
        self.model_thread = threading.Thread(
            target=self.model_server.serve_forever, daemon=True
        )
        self.model_thread.start()
        base = f"http://127.0.0.1:{self.model_server.server_address[1]}/v1"
        catalog = json.dumps(
            [
                {
                    "id": "dsh-policy",
                    "label": "DSH 候选生成模型",
                    "roles": ["propose"],
                    "gateway_url": base,
                    "model": "policy-model",
                    "api_key_env": "RUNTIME_POLICY_TOKEN",
                },
                {
                    "id": "dsh-judge",
                    "label": "DSH 独立评审模型",
                    "roles": ["judge"],
                    "gateway_url": base,
                    "model": "judge-model",
                    "api_key_env": "RUNTIME_JUDGE_TOKEN",
                },
            ]
        )
        self.environment = patch.dict(
            os.environ,
            {
                "ECOLOGYRSI_DSH_MODELS_JSON": catalog,
                "RUNTIME_POLICY_TOKEN": "policy-secret",
                "RUNTIME_JUDGE_TOKEN": "judge-secret",
            },
        )
        self.environment.start()
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()
        self.environment.stop()
        self.model_server.shutdown()
        self.model_server.server_close()
        self.model_thread.join(timeout=2)

    def test_configured_policy_and_independent_judge_are_called_without_preflight(self) -> None:
        self.model_server.responses.extend(  # type: ignore[attr-defined]
            [
                {
                    "parameters": {
                        "alpha": 0.44,
                        "window": 5,
                        "water_threshold": 0.41,
                    },
                    "rationale": "根据可见训练摘要提出有界修改。",
                },
                {"accepted": True, "guidance": "保持固定科学门禁。"},
            ]
        )
        status, catalog = self.request("/catalog")
        self.assertEqual(status, 200)
        self.assertEqual(
            catalog["schema_version"], "ecologyrsi-dsh.runtime-catalog/5"
        )
        remote = {item["id"]: item for item in catalog["models"]}
        self.assertTrue(remote["dsh-policy"]["configured"])
        self.assertFalse(remote["dsh-policy"]["verified"])
        self.assertFalse(remote["dsh-policy"]["available"])
        self.assertFalse(remote["dsh-policy"]["connection_available"])
        self.assertTrue(remote["dsh-policy"]["execution_available"])
        self.assertEqual(
            {item["id"] for item in catalog["dsh_models"]},
            {"dsh-policy", "dsh-judge"},
        )
        # Legacy authentication diagnostics remain empty until a real model
        # call succeeds, but they are not a run precondition.
        self.assertEqual(catalog["authenticated_models"], [])
        self.assertEqual(
            {item["id"] for item in catalog["dsh_strategy_models"]},
            {"dsh-policy"},
        )
        self.assertEqual(
            {item["id"] for item in catalog["dsh_review_models"]},
            {"dsh-judge"},
        )
        self.assertNotIn("policy-secret", json.dumps(catalog))
        self.assertTrue(catalog["dsh"]["connected"])
        self.assertTrue(catalog["dsh"]["roles_ready"])
        self.assertEqual(catalog["dsh"]["authenticated_model_count"], 0)
        self.assertNotIn("model.connection.verify", catalog["dsh"]["capabilities"])
        self.assertTrue(
            all(item["execution_available"] for item in catalog["dsh_models"])
        )

        run_request = {
            "domain_pack_id": "crop_soil_water",
            "dataset_id": "generated-toy-series@1",
            "strategy_id": "dsh_authenticated@1",
            "evaluator_id": "toy_time_forward@1",
            "policy_model_id": "dsh-policy",
            "judge_model_id": "dsh-judge",
            "slot": "bounded_residual_predictor",
            "budget": {"max_generations": 1, "max_candidates": 1},
            "auto_advance": 1,
            "idempotency_key": "authenticated-model-run",
        }
        status, created = self.request("/runs", "POST", run_request)
        self.assertEqual(status, 201, created)
        projection = created["projection"]
        self.assertEqual(projection["configuration"]["policy_model_id"], "dsh-policy")
        self.assertEqual(projection["configuration"]["judge_model_id"], "dsh-judge")
        self.assertEqual(projection["candidates"][0]["changes"]["alpha"], 0.44)
        requests = self.model_server.requests  # type: ignore[attr-defined]
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            [item["authorization"] for item in requests],
            [
                "Bearer policy-secret",
                "Bearer judge-secret",
            ],
        )
        self.assertEqual(
            [item["body"]["model"] for item in requests],
            ["policy-model", "judge-model"],
        )

    def test_run_creation_rejects_non_executable_directory_routes(self) -> None:
        current = {
            item["model_id"]: item for item in self.server.model_gateway.catalog()
        }
        for index, reason in enumerate(
            ("insecure_http_blocked", "host_route_not_available_to_sidecar")
        ):
            unsafe_policy = {
                **current["dsh-policy"],
                "configured": False,
                "directory_available": False,
                "execution_available": False,
                "credential_configured": True,
                "unavailable_reason": {"code": reason},
            }
            with (
                self.subTest(reason=reason),
                patch.object(
                    self.server.model_gateway,
                    "catalog",
                    return_value=[unsafe_policy, current["dsh-judge"]],
                ),
            ):
                status, rejected = self.request(
                    "/runs",
                    "POST",
                    {
                        "domain_pack_id": "crop_soil_water",
                        "dataset_id": "generated-toy-series@1",
                        "strategy_id": "dsh_authenticated@1",
                        "evaluator_id": "toy_time_forward@1",
                        "policy_model_id": "dsh-policy",
                        "judge_model_id": "dsh-judge",
                        "budget": {"max_generations": 1, "max_candidates": 1},
                        "auto_advance": 0,
                        "idempotency_key": f"unsafe-model-route-{index}",
                    },
                )
            self.assertEqual(status, 400, rejected)
            self.assertIn("后端调用配置不可执行", rejected["error"])
        self.assertEqual(self.model_server.requests, [])  # type: ignore[attr-defined]

    def test_explicit_model_verification_endpoint_is_removed(self) -> None:
        status, payload = self.request("/models/dsh-policy/verify", "POST", {})
        self.assertEqual(status, 404, payload)
        self.assertEqual(payload["error"], "not found")
        self.assertEqual(self.model_server.requests, [])  # type: ignore[attr-defined]

    def test_autonomous_create_receives_exact_runtime_catalog_and_gates(self) -> None:
        self.model_server.responses.append(  # type: ignore[attr-defined]
            {
                "prediction_model": {
                    "id": "greenhouse-exogenous-ridge@1",
                    "name": "外生变量岭回归",
                }
            }
        )

        description = {
            "descriptor": {
                "runnable": True,
                "display_name_zh": "测试温室序列",
                "domain_id": "greenhouse_test",
                "adapter_id": "greenhouse_test",
            },
            "readiness": {"ready": True},
        }
        series = SimpleNamespace(
            digest="d" * 64,
            split_manifest_digest_sha256="s" * 64,
            episode_id="agc_cucumber_2018:test",
        )
        with (
            patch.object(self.server.datasets, "describe", return_value=description),
            patch.object(self.server.datasets, "series", return_value=series),
        ):
            status, created = self.request(
                "/runs",
                "POST",
                {
                    "domain_pack_id": "greenhouse_environment@1",
                    "dataset_id": "agc_cucumber_2018",
                    "strategy_model_id": "dsh-policy",
                    "review_model_id": "dsh-judge",
                    "autonomous_mode": True,
                    "prediction_model_id": "greenhouse-exogenous-ridge@1",
                    "evaluator_id": "greenhouse_multihorizon_time_forward@1",
                    "budget": {
                        "max_generations": 1,
                        "candidates_per_generation": 1,
                        "max_candidates": 1,
                    },
                    "auto_advance": 0,
                    "idempotency_key": "autonomous-runtime-context",
                },
            )
            explicit_status, explicit_created = self.request(
                "/runs",
                "POST",
                {
                    "domain_pack_id": "greenhouse_environment@1",
                    "dataset_id": "agc_cucumber_2018",
                    "strategy_model_id": "dsh-policy",
                    "review_model_id": "dsh-judge",
                    "autonomous_mode": True,
                    "prediction_model_id": "greenhouse-exogenous-ridge@1",
                    "evaluator_id": "greenhouse_multihorizon_time_forward@1",
                    "budget": {
                        "max_generations": 1,
                        "candidates_per_generation": 1,
                        "max_candidates": 1,
                    },
                    "samples_per_update": 321,
                    "sample_concurrency": 3,
                    "sample_agent_batch_size": 16,
                    "auto_advance": 0,
                    "idempotency_key": "autonomous-runtime-explicit-sampling",
                },
            )
            rejected_sampling_controls = []
            for index, (name, value) in enumerate(
                (
                    ("samples_per_update", 8),
                    ("samples_per_update", 100_001),
                    ("sample_concurrency", 9),
                    ("sample_agent_batch_size", 129),
                )
            ):
                rejected_sampling_controls.append(
                    self.request(
                        "/runs",
                        "POST",
                        {
                            "domain_pack_id": "greenhouse_environment@1",
                            "dataset_id": "agc_cucumber_2018",
                            "strategy_model_id": "dsh-policy",
                            "review_model_id": "dsh-judge",
                            "autonomous_mode": True,
                            "prediction_model_id": "greenhouse-exogenous-ridge@1",
                            "evaluator_id": "greenhouse_multihorizon_time_forward@1",
                            "budget": {"max_generations": 1, "max_candidates": 1},
                            name: value,
                            "auto_advance": 0,
                            "idempotency_key": f"invalid-sampling-control-{index}",
                        },
                    )
                )
        self.assertEqual(status, 201, created)
        self.assertEqual(explicit_status, 201, explicit_created)
        self.assertTrue(
            all(status == 400 for status, _payload in rejected_sampling_controls),
            rejected_sampling_controls,
        )
        self.assertEqual(self.model_server.requests, [])  # type: ignore[attr-defined]
        self.assertEqual(
            created["projection"]["configuration"]["sample_agent_batch_size"],
            64,
        )
        self.assertEqual(
            created["projection"]["configuration"]["samples_per_update"], 500
        )
        self.assertEqual(
            created["projection"]["configuration"]["sample_concurrency"], 2
        )
        explicit_configuration = explicit_created["projection"]["configuration"]
        self.assertEqual(explicit_configuration["samples_per_update"], 321)
        self.assertEqual(explicit_configuration["sample_concurrency"], 3)
        self.assertEqual(explicit_configuration["sample_agent_batch_size"], 16)
        self.assertEqual(
            created["projection"]["configuration"]["sample_operation_max_tokens"],
            {
                "sample.planner": 6144,
                "sample.repair": 3072,
                "sample.critic": 2048,
            },
        )
        self.assertEqual(
            created["projection"]["configuration"]["sample_remote_critic_policy"],
            {"version": "always@1"},
        )
        self.assertEqual(
            created["projection"]["configuration"][
                "sample_planner_prompt_profile"
            ],
            {"version": "origin_shared_context@1"},
        )
        self.assertEqual(created["projection"]["token_limit"], 100_000_000)
        self.assertEqual(
            created["projection"]["token_reservation_per_wave"], 262_144
        )
        self.assertEqual(
            created["projection"]["configuration"]["sample_token_budget_policy"],
            "hard_gateway_call_reservation@1",
        )
        self.assertEqual(
            created["projection"]["token_budget_scope"],
            "sample_agent_gateway_calls_only@1",
        )
        self.assertFalse(created["projection"]["run_wide_accounting_complete"])
        self.assertEqual(
            created["projection"]["configuration"]["token_budget_scope"],
            "sample_agent_gateway_calls_only@1",
        )
        self.assertFalse(
            created["projection"]["configuration"]["run_wide_accounting_complete"]
        )

        state = self.server.director.state(created["projection"]["run_id"])
        self.assertEqual(state.task_manifest.metadata["samples_per_update"], 500)
        self.assertEqual(state.task_manifest.metadata["sample_concurrency"], 2)
        self.assertEqual(state.task_manifest.metadata["sample_agent_batch_size"], 64)
        self.assertEqual(
            state.task_manifest.metadata["sample_planner_prompt_profile"],
            {"version": "origin_shared_context@1"},
        )
        self.server.strategy_router.research_plan(
            "dsh-policy",
            run=state.run,
            task=state.task_manifest,
            parameter_schemas=self.server.strategy_router.parameter_schemas_for_task(
                state.task_manifest
            ),
        )

        research_request = self.model_server.requests[0]  # type: ignore[attr-defined]
        user_message = json.loads(research_request["body"]["messages"][1]["content"])
        context = user_message["input"]["context"]
        component_catalog = context["runtime_component_catalog"]
        self.assertEqual(
            [item["id"] for item in component_catalog["prediction_models"]],
            [
                "greenhouse-exogenous-ridge@1",
                "greenhouse-targetwise-ridge@1",
            ],
        )
        predictor = component_catalog["prediction_models"][0]
        self.assertEqual(
            set(predictor["parameter_schemas"]),
            {"history_steps", "ridge_alpha", "residual_scale"},
        )
        self.assertIn(
            "L2 regularization strength",
            predictor["parameter_semantics"]["ridge_alpha"],
        )
        targetwise_predictor = component_catalog["prediction_models"][1]
        self.assertEqual(
            set(targetwise_predictor["parameter_schemas"]),
            {
                "history_steps",
                "ridge_alpha",
                "air_temperature_residual_scale",
                "relative_humidity_residual_scale",
                "co2_concentration_residual_scale",
            },
        )
        self.assertEqual(
            [item["pipeline_id"] for item in context["algorithm_blueprint_catalog"]],
            [
                "greenhouse-exogenous-ridge@1",
                "greenhouse-targetwise-ridge@1",
            ],
        )
        self.assertEqual(
            component_catalog["selected_evaluator_id"],
            "greenhouse_multihorizon_time_forward@1",
        )
        gates = context["hard_gates"]
        self.assertEqual(
            [(item["metric"], item["operator"]) for item in gates],
            [
                ("objective_score", ">"),
                ("normalized_rmse", "<="),
                ("constraint_violations", "<="),
                ("sample_execution_coverage", ">="),
            ],
        )
        self.assertEqual(gates[0]["threshold"], 1e-9)
        self.assertEqual(gates[1]["tolerance"], 1e-12)
        self.assertEqual(gates[2]["threshold"], 0)
        self.assertEqual(gates[3]["threshold"], 0.8)
        self.assertEqual(context["objective_profile"]["hard_gates"], gates)
        self.assertEqual(
            set(context["allowed_parameter_schemas"]),
            {"history_steps", "ridge_alpha", "residual_scale"},
        )

    def test_continuous_autonomous_create_keeps_background_research_retry_alive(
        self,
    ) -> None:
        planner_started = threading.Event()
        release_planner = threading.Event()
        planner_threads: list[str] = []

        def fail_research_plan(*_args: object, **_kwargs: object) -> object:
            planner_threads.append(threading.current_thread().name)
            planner_started.set()
            if not release_planner.wait(timeout=5):
                raise AssertionError("test did not release deferred research planner")
            raise GatewayResponseError(
                "simulated research-plan timeout",
                retryable=True,
                attempts=4,
                status_code=503,
            )

        with patch.object(
            self.server.strategy_router,
            "research_plan",
            side_effect=fail_research_plan,
        ) as research_plan:
            status, created = self.request(
                "/runs",
                "POST",
                {
                    "domain_pack_id": "crop_soil_water",
                    "dataset_id": "generated-toy-series@1",
                    "strategy_model_id": "dsh-policy",
                    "review_model_id": "dsh-judge",
                    "rounds": 2,
                    "candidates_per_generation": 1,
                    "max_candidates": 2,
                    "knowledge_online_enabled": False,
                    "auto_progress": True,
                    "auto_advance": 0,
                    "idempotency_key": "strict-research-plan-failure",
                },
            )
            self.assertEqual(status, 201, created)
            initial = created["projection"]
            self.assertEqual(initial["status"], "running")
            self.assertEqual(initial["generation"], 0)
            self.assertEqual(initial["candidates"], [])
            self.assertEqual(
                initial["configuration"]["autonomous_plan"], {}
            )
            self.assertTrue(planner_started.wait(timeout=2))
            self.assertEqual(planner_threads, ["ecologyrsi-auto-progress"])

            run_id = initial["run_id"]
            release_planner.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = self.server.director.state(run_id)
                if any(event.kind == "GatewayRetryScheduled" for event in state.events):
                    break
                time.sleep(0.01)

            state = self.server.director.state(run_id)
            self.assertEqual(state.run.status.value, "running")
            retry = next(
                event
                for event in reversed(state.events)
                if event.kind == "GatewayRetryScheduled"
            )
            self.assertEqual(retry.payload["generation"], 0)
            self.assertGreaterEqual(float(retry.payload["delay_seconds"]), 0.0)
            self.assertFalse(any(event.kind == "RunFailed" for event in state.events))
            research_plan.assert_called_once()

    def test_continuous_autonomous_create_ignores_stale_connection_error(self) -> None:
        with (
            patch.object(
                self.server.model_gateway,
                "connection_status",
                return_value={"state": "error", "last_error": "HTTP 503"},
            ) as connection_status,
            patch.object(self.server.strategy_router, "research_plan") as research_plan,
        ):
            status, created = self.request(
                "/runs",
                "POST",
                {
                    "domain_pack_id": "crop_soil_water",
                    "dataset_id": "generated-toy-series@1",
                    "strategy_model_id": "dsh-policy",
                    "review_model_id": "dsh-judge",
                    "rounds": 2,
                    "candidates_per_generation": 1,
                    "max_candidates": 2,
                    "auto_advance": "continuous",
                    "start": False,
                    "idempotency_key": "known-connection-error",
                },
            )

        self.assertEqual(status, 201, created)
        self.assertEqual(created["projection"]["status"], "created")
        self.assertTrue(created["projection"]["configuration"]["auto_progress"])
        connection_status.assert_not_called()
        research_plan.assert_not_called()
        self.assertEqual(len(self.server.ledger.run_ids()), 1)
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_continuous_autonomous_create_defers_explicit_plan_fallback(self) -> None:
        with patch.object(
            self.server.strategy_router,
            "research_plan",
            side_effect=TimeoutError("simulated research-plan timeout"),
        ) as research_plan:
            status, created = self.request(
                "/runs",
                "POST",
                {
                    "domain_pack_id": "crop_soil_water",
                    "dataset_id": "generated-toy-series@1",
                    "strategy_model_id": "dsh-policy",
                    "review_model_id": "dsh-judge",
                    "rounds": 2,
                    "candidates_per_generation": 1,
                    "max_candidates": 2,
                    "auto_progress": True,
                    "auto_advance": 0,
                    "allow_host_fallback": True,
                    "start": False,
                    "idempotency_key": "allowed-research-plan-fallback",
                },
            )

        self.assertEqual(status, 201, created)
        projection = created["projection"]
        configuration = projection["configuration"]
        plan = configuration["autonomous_plan"]
        self.assertEqual(projection["status"], "created")
        self.assertTrue(configuration["auto_progress"])
        self.assertTrue(configuration["allow_host_fallback"])
        self.assertEqual(
            configuration["remote_fallback_policy"], "record_and_continue"
        )
        self.assertEqual(plan, {})
        self.assertEqual(
            configuration["autonomous_plan_execution"],
            "deferred_to_first_generation",
        )
        research_plan.assert_not_called()

    def test_restarted_service_rejects_same_id_with_changed_remote_model(self) -> None:
        run_request = {
            "domain_pack_id": "crop_soil_water",
            "dataset_id": "generated-toy-series@1",
            "strategy_id": "dsh_authenticated@1",
            "evaluator_id": "toy_time_forward@1",
            "policy_model_id": "dsh-policy",
            "judge_model_id": "rule_judge@1",
            "budget": {"max_generations": 1, "max_candidates": 1},
            "auto_advance": 0,
            "idempotency_key": "remote-model-drift",
        }
        status, created = self.request("/runs", "POST", run_request)
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        frozen_digest = created["projection"]["configuration"][
            "policy_model_digest"
        ]

        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        changed_catalog = json.loads(os.environ["ECOLOGYRSI_DSH_MODELS_JSON"])
        changed_catalog[0]["model"] = "policy-model-v2"
        os.environ["ECOLOGYRSI_DSH_MODELS_JSON"] = json.dumps(changed_catalog)
        self.server = EvolutionHTTPServer(("127.0.0.1", 0), self.db_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api"

        current_digest = self.server.model_gateway.configuration_digest("dsh-policy")
        self.assertNotEqual(current_digest, frozen_digest)
        status, catalog = self.request("/catalog")
        self.assertEqual(status, 200, catalog)
        changed_policy = next(
            item for item in catalog["dsh_models"] if item["id"] == "dsh-policy"
        )
        self.assertFalse(changed_policy["authentication_verified"])
        self.assertFalse(changed_policy["available"])
        self.assertFalse(changed_policy["verification_persisted"])
        status, rejected = self.request(
            "/runs/" + quote(run_id, safe="") + "/advance",
            "POST",
            {"steps": 1},
        )
        self.assertEqual(status, 400, rejected)
        self.assertIn("候选生成模型配置发生漂移", rejected["error"])
        self.assertEqual(
            rejected["error_code"], "frozen_runtime_binding_drift"
        )
        encoded_rejection = json.dumps(rejected, ensure_ascii=False)
        self.assertNotIn(frozen_digest, encoded_rejection)
        self.assertNotIn(current_digest, encoded_rejection)
        self.assertEqual(self.server.ledger.pending_command_keys(), ())
        self.assertEqual(
            len(self.model_server.requests), 0  # type: ignore[attr-defined]
        )

    def test_restarted_service_continues_configured_models_without_preflight(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_id": "dsh_authenticated@1",
                "evaluator_id": "toy_time_forward@1",
                "policy_model_id": "dsh-policy",
                "judge_model_id": "dsh-judge",
                "budget": {"max_generations": 1, "max_candidates": 1},
                "auto_advance": 0,
                "idempotency_key": "remote-model-reauthentication",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        frozen_policy_digest = created["projection"]["configuration"][
            "policy_model_digest"
        ]
        frozen_judge_digest = created["projection"]["configuration"][
            "judge_model_digest"
        ]

        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.server = EvolutionHTTPServer(("127.0.0.1", 0), self.db_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api"

        self.assertEqual(
            self.server.model_gateway.configuration_digest("dsh-policy"),
            frozen_policy_digest,
        )
        self.assertEqual(
            self.server.model_gateway.configuration_digest("dsh-judge"),
            frozen_judge_digest,
        )
        status, catalog = self.request("/catalog")
        self.assertEqual(status, 200, catalog)
        restored = {item["id"]: item for item in catalog["dsh_models"]}
        for model_id in ("dsh-policy", "dsh-judge"):
            self.assertFalse(restored[model_id]["authentication_verified"])
            self.assertFalse(restored[model_id]["verification_persisted"])
            self.assertFalse(restored[model_id]["available"])
            self.assertTrue(restored[model_id]["execution_available"])

        self.model_server.responses.extend(  # type: ignore[attr-defined]
            [
                {
                    "parameters": {
                        "alpha": 0.44,
                        "window": 5,
                        "water_threshold": 0.41,
                    }
                },
                {"accepted": True, "guidance": "保持固定科学门禁。"},
            ]
        )
        advance_path = "/runs/" + quote(run_id, safe="") + "/advance"
        status, advanced = self.request(advance_path, "POST", {"steps": 1})
        self.assertEqual(status, 200, advanced)
        self.assertEqual(advanced["projection"]["status"], "completed")
        self.assertEqual(len(self.model_server.requests), 2)  # type: ignore[attr-defined]

    def test_restarted_service_accepts_rotated_credential_without_preflight(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_id": "dsh_authenticated@1",
                "evaluator_id": "toy_time_forward@1",
                "policy_model_id": "dsh-policy",
                "judge_model_id": "rule_judge@1",
                "budget": {"max_generations": 1, "max_candidates": 1},
                "auto_advance": 0,
                "idempotency_key": "rotated-credential-no-preflight",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        configuration_digest = self.server.model_gateway.configuration_digest(
            "dsh-policy"
        )

        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        os.environ["RUNTIME_POLICY_TOKEN"] = "rotated-policy-secret"
        self.server = EvolutionHTTPServer(("127.0.0.1", 0), self.db_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api"

        self.assertEqual(
            self.server.model_gateway.configuration_digest("dsh-policy"),
            configuration_digest,
        )
        status, catalog = self.request("/catalog")
        self.assertEqual(status, 200, catalog)
        rotated_policy = next(
            item for item in catalog["dsh_models"] if item["id"] == "dsh-policy"
        )
        self.assertFalse(rotated_policy["authentication_verified"])
        self.assertFalse(rotated_policy["available"])
        self.assertFalse(rotated_policy["verification_persisted"])
        self.assertEqual(
            rotated_policy["authentication_state"], "configured_unverified"
        )
        self.assertTrue(rotated_policy["execution_available"])

        self.model_server.responses.append(  # type: ignore[attr-defined]
            {
                "parameters": {
                    "alpha": 0.44,
                    "window": 5,
                    "water_threshold": 0.41,
                }
            }
        )
        status, advanced = self.request(
            "/runs/" + quote(run_id, safe="") + "/advance",
            "POST",
            {"steps": 1},
        )
        self.assertEqual(status, 200, advanced)
        self.assertEqual(advanced["projection"]["status"], "completed")
        self.assertEqual(len(self.model_server.requests), 1)  # type: ignore[attr-defined]
        self.assertEqual(
            self.model_server.requests[0]["authorization"],  # type: ignore[attr-defined]
            "Bearer rotated-policy-secret",
        )

    def test_policy_timeout_is_recorded_and_uses_host_fallback_without_preflight(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_id": "dsh_authenticated@1",
                "evaluator_id": "toy_time_forward@1",
                "policy_model_id": "dsh-policy",
                "judge_model_id": "rule_judge@1",
                "budget": {"max_generations": 1, "max_candidates": 1},
                "auto_advance": 0,
                "idempotency_key": "verified-timeout-host-fallback",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        self.assertEqual(len(self.model_server.requests), 0)  # type: ignore[attr-defined]

        with patch.object(
            self.server.model_gateway._opener,  # type: ignore[attr-defined]
            "open",
            side_effect=TimeoutError("simulated proposal timeout"),
        ) as open_mock:
            status, advanced = self.request(
                "/runs/" + quote(run_id, safe="") + "/advance",
                "POST",
                {"steps": 1},
            )

        self.assertEqual(status, 200, advanced)
        self.assertEqual(
            open_mock.call_count,
            self.server.model_gateway.max_attempts,  # type: ignore[attr-defined]
        )
        self.assertEqual(advanced["projection"]["status"], "completed")
        status, catalog = self.request("/catalog")
        self.assertEqual(status, 200, catalog)
        policy = next(
            item for item in catalog["dsh_models"] if item["id"] == "dsh-policy"
        )
        self.assertFalse(policy["authentication_verified"])
        self.assertFalse(policy["available"])
        self.assertEqual(policy["connection"]["state"], "configured")

        status, events = self.request(
            "/runs/" + quote(run_id, safe="") + "/events"
        )
        self.assertEqual(status, 200, events)
        proposal_failures = [
            item
            for item in events["events"]
            if item["type"] == "stage.recorded"
            and item["payload"].get("stage") == "proposal"
            and item["payload"].get("status") == "failed"
        ]
        self.assertEqual(len(proposal_failures), 1)
        self.assertEqual(
            proposal_failures[0]["payload"]["public_error"],
            "GatewayResponseError [gateway_response_error]",
        )
        self.assertNotIn(
            "simulated proposal timeout",
            json.dumps(proposal_failures, ensure_ascii=False),
        )

    def test_restarted_service_does_not_require_reverification_for_builtin_models(self) -> None:
        status, created = self.request(
            "/runs",
            "POST",
            {
                "domain_pack_id": "crop_soil_water",
                "dataset_id": "generated-toy-series@1",
                "strategy_id": "parameter_sweep@1",
                "evaluator_id": "toy_time_forward@1",
                "policy_model_id": "host_parameter_generator@1",
                "judge_model_id": "rule_judge@1",
                "budget": {"max_generations": 1, "max_candidates": 1},
                "auto_advance": 0,
                "idempotency_key": "builtin-model-restart",
            },
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]

        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.server = EvolutionHTTPServer(("127.0.0.1", 0), self.db_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api"

        status, advanced = self.request(
            "/runs/" + quote(run_id, safe="") + "/advance",
            "POST",
            {"steps": 1},
        )
        self.assertEqual(status, 200, advanced)
        self.assertEqual(advanced["projection"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
