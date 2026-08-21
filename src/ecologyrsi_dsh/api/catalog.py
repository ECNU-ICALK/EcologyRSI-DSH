"""Catalog and dataset query endpoints for the HTTP handler."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

from ..integrations.model_bindings import (
    HOST_PARAMETER_GENERATOR_ID,
    RULE_JUDGE_ID,
    builtin_model_configuration_digest,
    model_supports_role,
)
from ..evaluators.registry import (
    TOY_DATASET_ID,
    TOY_EVALUATOR_ID,
    TOY_PREDICTOR_MODEL_ID,
)
from .projection import _projection_json, _run_summary_projection, _state_payload


class CatalogEndpointsMixin:
    def _catalog_payload(self) -> dict[str, Any]:
        raw_catalog = self.server.datasets.catalog()
        ready_datasets: list[dict[str, Any]] = []
        unavailable_datasets: list[dict[str, Any]] = []
        for raw in raw_catalog["datasets"]:
            readiness = raw["readiness"]
            item = {
                "id": raw["dataset_id"],
                "dataset_id": raw["dataset_id"],
                "label": raw["display_name_zh"],
                "display_name": raw["display_name_zh"],
                "description": " ".join(raw.get("notes_zh", [])),
                "domain_id": raw["domain_id"],
                "domain_pack_id": (
                    "crop_soil_water"
                    if raw["dataset_id"] == TOY_DATASET_ID
                    else "greenhouse_environment@1"
                ),
                "license": raw["license"],
                "ready": readiness["ready"],
                "readiness": readiness,
            }
            if raw["runnable"] and readiness["ready"]:
                item["episodes"] = self.server.datasets.episodes(raw["dataset_id"])
                ready_datasets.append(item)
            else:
                unavailable_datasets.append(item)
        ready_datasets.sort(
            key=lambda item: (item["id"] == TOY_DATASET_ID, item["id"])
        )
        has_greenhouse = any(
            item["domain_pack_id"] == "greenhouse_environment@1"
            for item in ready_datasets
        )
        domain_packs = []
        if has_greenhouse:
            domain_packs.append(
                {
                    "id": "greenhouse_environment@1",
                    "label": "温室环境历史回放模型包",
                    "description": "针对温度、相对湿度和二氧化碳浓度开展时间前向预测进化。",
                }
            )
        domain_packs.append(
            {
                "id": "crop_soil_water",
                "label": "作物—土壤—水分合成模型包",
                "description": "用于验证运行、事件账本和交互流程的确定性工程样例。",
            }
        )
        remote_models = self.server.model_gateway.catalog()
        models = [
            {
                "id": HOST_PARAMETER_GENERATOR_ID,
                "model_id": HOST_PARAMETER_GENERATOR_ID,
                "label": "内置有界参数生成器",
                "description": "由宿主按固定参数范围生成候选，不调用外部模型。",
                "roles": ["propose"],
                "configuration_digest": builtin_model_configuration_digest(
                    HOST_PARAMETER_GENERATOR_ID
                ),
                "binding_source": "builtin_implementation",
                "authentication_state": "local",
                "local_model": True,
                "available": True,
            },
            {
                "id": RULE_JUDGE_ID,
                "model_id": RULE_JUDGE_ID,
                "label": "内置规则独立评审",
                "description": "只依据固定科学指标作出独立搜索保留判断。",
                "roles": ["judge"],
                "configuration_digest": builtin_model_configuration_digest(
                    RULE_JUDGE_ID
                ),
                "binding_source": "builtin_implementation",
                "authentication_state": "local",
                "local_model": True,
                "available": True,
            },
        ]
        for raw in remote_models:
            configured = bool(
                raw.get(
                    "configured",
                    raw.get("credential_configured")
                    and raw.get("directory_available", True),
                )
            )
            verified = bool(raw.get("authentication_verified"))
            available = bool(raw.get("available"))
            directory_available = raw.get("directory_available", True) is not False
            execution_available = configured and directory_available
            models.append(
                {
                    **raw,
                    "id": raw["model_id"],
                    "model_source": "dsh_gateway",
                    "description": (
                        "该模型已在 DSH 主目录登记，但当前后端配置不可执行；请查看不可用原因。"
                        if not directory_available
                        else "通过服务端 Bearer 凭据连接兼容模型网关；连接在实际运行请求中检查。"
                    ),
                    "configured": configured,
                    "verified": verified,
                    "connection_available": available,
                    "execution_available": execution_available,
                    "available": available,
                }
            )
        # Keep one explicit, shared DSH model directory in addition to the
        # role-filtered compatibility keys below.  The browser may use these
        # fields to show the same configured inventory for both selectors;
        # execution still enforces role, credential, and frozen-binding checks.
        dsh_models = [
            item for item in models if item.get("model_source") == "dsh_gateway"
        ]
        authenticated_models = [
            item
            for item in dsh_models
            if item.get("authentication_verified") is True
        ]
        available_models = [item for item in dsh_models if item.get("available") is True]
        dsh_strategy_models = [
            item for item in dsh_models if model_supports_role(item, "propose")
        ]
        dsh_review_models = [
            item for item in dsh_models if model_supports_role(item, "judge")
        ]
        available_ids = {item["id"] for item in ready_datasets}
        evaluators = [
            item
            for item in self.server.evaluators.catalog()
            if available_ids.intersection(item["dataset_ids"])
        ]
        evaluators.sort(
            key=lambda item: (item["id"] == TOY_EVALUATOR_ID, item["id"])
        )
        prediction_models = [
            item
            for item in self.server.evaluators.predictor_catalog()
            if available_ids.intersection(item["dataset_ids"])
        ]
        prediction_models.sort(
            key=lambda item: (item["id"] == TOY_PREDICTOR_MODEL_ID, item["id"])
        )
        configured_count = sum(
            1 for item in models if item.get("model_source") == "dsh_gateway" and item.get("configured")
        )
        verified_count = sum(
            1 for item in dsh_models if item.get("authentication_verified")
        )
        available_count = sum(1 for item in dsh_models if item.get("available"))
        executable_count = sum(
            1 for item in dsh_models if item.get("execution_available")
        )
        # A strategy is runnable only when a credentialed model explicitly
        # exposes the proposal role.  Counting any remote model here used to
        # make a judge-only DSH setup look ready for autonomous evolution.
        configured_strategy_count = sum(
            1
            for item in dsh_models
            if item.get("configured") and model_supports_role(item, "propose")
        )
        configured_review_count = sum(
            1
            for item in dsh_models
            if item.get("configured") and model_supports_role(item, "judge")
        )
        verified_strategy_count = sum(
            1
            for item in dsh_models
            if item.get("authentication_verified")
            and model_supports_role(item, "propose")
        )
        verified_review_count = sum(
            1
            for item in dsh_models
            if item.get("authentication_verified")
            and model_supports_role(item, "judge")
        )
        available_strategy_count = sum(
            1
            for item in dsh_models
            if item.get("available") and model_supports_role(item, "propose")
        )
        available_review_count = sum(
            1
            for item in dsh_models
            if item.get("available") and model_supports_role(item, "judge")
        )
        executable_strategy_count = sum(
            1
            for item in dsh_models
            if item.get("execution_available")
            and model_supports_role(item, "propose")
        )
        executable_review_count = sum(
            1
            for item in dsh_models
            if item.get("execution_available")
            and model_supports_role(item, "judge")
        )
        return {
            "schema_version": "ecologyrsi-dsh.runtime-catalog/5",
            "domain_packs": domain_packs,
            "datasets": ready_datasets,
            "unavailable_datasets": unavailable_datasets,
            "strategies": [
                {
                    **item,
                    **(
                        {"available": executable_strategy_count > 0}
                        if item["requires_authenticated_model"]
                        else {"available": True}
                    ),
                }
                for item in self.server.strategy_router.catalog()
            ],
            "model_workflows": [
                {
                    "id": "research_compile_evolve@1",
                    "label": "模型调研—编译—进化",
                    "description": "策略模型先形成研究计划，再在宿主有界参数空间内逐轮生成和改进候选。",
                    "autonomous": True,
                },
                {
                    "id": "legacy_component_search@1",
                    "label": "固定组件搜索（兼容）",
                    "description": "沿用旧版由请求显式冻结预测模型、策略和评测器的流程。",
                    "autonomous": False,
                },
            ],
            "prediction_models": prediction_models,
            "evaluators": evaluators,
            "models": models,
            "dsh_models": dsh_models,
            "authenticated_models": authenticated_models,
            "available_models": available_models,
            "dsh_strategy_models": dsh_strategy_models,
            "dsh_review_models": dsh_review_models,
            "policy_models": [item for item in models if model_supports_role(item, "propose")],
            "judge_models": [item for item in models if model_supports_role(item, "judge")],
            # Canonical names for the autonomous workflow.  Keep the legacy
            # policy/judge keys above so older plugin builds continue to load.
            "strategy_models": [item for item in models if model_supports_role(item, "propose")],
            "review_models": [item for item in models if model_supports_role(item, "judge")],
            "dsh": {
                # Native mode delegates Session/context management, model
                # routing, structured roles, subagents and Workflow execution
                # to the mounted DSH runtime.  The Python process remains the
                # durable scientific-state sidecar only.
                "harness_execution": (
                    "dsh_native_agent_runtime"
                    if self.server.dsh_native_runtime is not None
                    else "sidecar_openai_compatible_gateway"
                ),
                "official_harness_agent_loop": self.server.dsh_native_runtime
                is not None,
                "connected": (
                    self.server.dsh_native_runtime is not None
                    or executable_count > 0
                ),
                "configured": bool(remote_models),
                "environment": (
                    "dsh_native"
                    if self.server.dsh_native_runtime is not None
                    else "configured"
                    if configured_count
                    else "local"
                ),
                "configured_model_count": configured_count,
                "authenticated_model_count": verified_count,
                "verified_model_count": verified_count,
                "available_model_count": available_count,
                "executable_model_count": executable_count,
                "dsh_model_count": len(dsh_models),
                "authenticated_dsh_model_count": len(authenticated_models),
                "available_dsh_model_count": len(available_models),
                "dsh_strategy_model_count": len(dsh_strategy_models),
                "dsh_review_model_count": len(dsh_review_models),
                "configured_strategy_model_count": configured_strategy_count,
                "configured_review_model_count": configured_review_count,
                "authenticated_strategy_model_count": verified_strategy_count,
                "authenticated_review_model_count": verified_review_count,
                "available_strategy_model_count": available_strategy_count,
                "available_review_model_count": available_review_count,
                "executable_strategy_model_count": executable_strategy_count,
                "executable_review_model_count": executable_review_count,
                # These names are retained for old clients. They now describe
                # safe, credentialed execution configuration; request health
                # remains available in each model's ``connection`` object.
                "strategy_connected": executable_strategy_count > 0,
                "review_connected": executable_review_count > 0,
                "roles_verified": verified_strategy_count > 0 and verified_review_count > 0,
                "roles_ready": executable_strategy_count > 0 and executable_review_count > 0,
                "capabilities": [
                    "training.data.read",
                    "evolution.run.create",
                    "evolution.run.advance",
                    "evolution.projection.read",
                    "evaluation.samples.read",
                    "run.control",
                    "run.archive",
                    "run.delete",
                    "intervention.write",
                ],
            },
        }

    def _dataset_payload(self, dataset_id: str) -> dict[str, Any]:
        query = parse_qs(urlparse(self.path).query)
        if not query:
            description = self.server.datasets.describe(dataset_id)
            descriptor = description["descriptor"]
            return {
                **description,
                "dataset": {
                    "id": descriptor["dataset_id"],
                    "display_name": descriptor["display_name_zh"],
                    "description": " ".join(descriptor.get("notes_zh", [])),
                    "license": descriptor["license"],
                },
            }
        return self._dataset_sample_payload(dataset_id)

    def _dataset_sample_payload(self, dataset_id: str) -> dict[str, Any]:
        query = parse_qs(urlparse(self.path).query)
        partition = query.get("partition", ["training_fit"])[0]
        episode_id = query.get("episode_id", [None])[0]
        expected_dataset_digest = query.get("expected_dataset_digest", [None])[0]
        expected_split_digest = query.get(
            "expected_split_manifest_digest", [None]
        )[0]
        try:
            offset = int(query.get("offset", ["0"])[0])
            limit = int(query.get("limit", ["20"])[0])
        except (TypeError, ValueError) as exc:
            raise ValueError("offset 和 limit 必须是整数") from exc
        description = self.server.datasets.describe(dataset_id)
        sample = self.server.datasets.sample(
            dataset_id,
            partition=partition,
            episode_id=episode_id,
            offset=offset,
            limit=limit,
            expected_dataset_digest=expected_dataset_digest,
            expected_split_manifest_digest=expected_split_digest,
        )
        descriptor = description["descriptor"]
        schema = [
            {
                "name": "timestamp",
                "label": "时间索引",
                "type": "整数",
                "unit": "小时",
            }
        ]
        for feature in sample["features"].values():
            schema.append(
                {
                    "name": feature["name"],
                    "label": feature["display_name_zh"],
                    "type": "数值",
                    "unit": feature["unit"],
                    "role": feature["role"],
                }
            )
        rows = [
            {"timestamp": item["timestamp"], **item["values"]}
            for item in sample["rows"]
        ]
        series = self.server.datasets.series(
            dataset_id,
            episode_id,
            expected_dataset_digest=expected_dataset_digest,
            expected_split_manifest_digest=expected_split_digest,
        )
        return {
            "schema_version": "ecologyrsi-dsh.browser-dataset/2",
            "dataset": {
                "id": dataset_id,
                "display_name": descriptor["display_name_zh"],
                "description": " ".join(descriptor.get("notes_zh", [])),
                "license": descriptor["license"],
                "digest": sample["dataset_digest_sha256"],
                "episode_id": sample["episode_id"],
            },
            "schema": schema,
            "features": sample["features"],
            "partitions": {
                name: {"count": value.size}
                for name, value in series.partitions.items()
                if name in {"training_fit", "training_feedback"}
            },
            "page": {
                "rows": rows,
                "offset": sample["offset"],
                "limit": sample["limit"],
                "total": sample["total"],
                "next_offset": sample["next_offset"],
            },
            "profile": description["profile"],
            "readiness": description["readiness"],
            "source_integrity": description["readiness"].get("source_integrity"),
            "partition": sample["partition"],
            "dataset_digest_sha256": sample["dataset_digest_sha256"],
            "split_manifest_digest": sample["split_manifest_digest_sha256"],
            "split_manifest_digest_sha256": sample["split_manifest_digest_sha256"],
            "offset": sample["offset"],
            "limit": sample["limit"],
            "total": sample["total"],
            "next_offset": sample["next_offset"],
            "rows": rows,
        }

    def _list_runs(self) -> dict[str, Any]:
        query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
        unknown = set(query) - {"include_archived", "view"}
        if unknown:
            raise ValueError("unknown run list query: " + ", ".join(sorted(unknown)))
        values = query.get("include_archived", ["false"])
        if len(values) != 1 or values[0].strip().casefold() not in {"true", "false"}:
            raise ValueError("include_archived must be true or false")
        include_archived = values[0].strip().casefold() == "true"
        view_values = query.get("view", ["detail"])
        if len(view_values) != 1 or view_values[0].strip().casefold() not in {
            "detail",
            "summary",
        }:
            raise ValueError("view must be detail or summary")
        view = view_values[0].strip().casefold()
        projector = _run_summary_projection if view == "summary" else _projection_json
        runs = []
        for run_id in self.server.ledger.run_ids(include_archived=include_archived):
            try:
                state = self.server.director.replay(run_id)
                runs.append(self._decorate_run_projection(projector(state)))
            except (KeyError, ValueError):
                # A list response must fail closed rather than silently
                # presenting a partial view that could be mistaken for the
                # complete local ledger.
                raise
        return {
            "runs": runs,
            "view": view,
            "include_archived": include_archived,
            "archived_count": self.server.ledger.archived_count(),
        }

    def _decorate_run_projection(self, projection: dict[str, Any]) -> dict[str, Any]:
        item = dict(projection)
        archived_at = self.server.ledger.archived_at(str(item["run_id"]))
        item["archived"] = archived_at is not None
        item["archived_at"] = archived_at
        scheduler = getattr(self.server, "auto_progress", None)
        if scheduler is not None:
            item["execution_scheduler"] = scheduler.diagnostics(
                str(item["run_id"])
            )
        return item

    def _run_payload(self, run_id: str) -> dict[str, Any]:
        payload = _state_payload(self.server.director.replay(run_id))
        payload["projection"] = self._decorate_run_projection(payload["projection"])
        return payload
