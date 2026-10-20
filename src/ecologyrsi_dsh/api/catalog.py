"""Catalog and dataset query endpoints for the HTTP handler."""

from __future__ import annotations

from ..core.prediction_policy import RUNTIME_EVALUATOR_ID, RUNTIME_PREDICTION_POLICY

from typing import Any
from ..application.queries import RunQueries
from urllib.parse import parse_qs, urlparse

from ..integrations.model_bindings import model_supports_role
from .projection import (
    _admission_progress_fields,
    _assert_http_scope,
    _monitor_payload,
    _projection_json,
    _run_summary_projection,
    _state_payload,
)
from .workspaces import WORKSPACE_VIEWS, overview_projection, workspace_section
from .projection import _workspace_revisions


class CatalogEndpointsMixin:
    def _catalog_payload(self) -> dict[str, Any]:
        from ..evolution.parameters import run_parameter_contract
        ready_datasets, unavailable_datasets = [], []
        for raw in self.server.datasets.catalog()["datasets"]:
            adapter = raw.get("task_adapter")
            item = {
                "id": raw["dataset_id"], "dataset_id": raw["dataset_id"],
                "label": raw["display_name_zh"], "display_name": raw["display_name_zh"],
                "description": " ".join(raw.get("notes_zh", [])),
                "domain_id": raw["domain_id"],
                "domain_pack_id": adapter["domain_pack_id"] if adapter else None,
                "training_selectable": raw["training_selectable"], "task_adapter": adapter,
                "license": raw["license"], "ready": raw["readiness"]["ready"],
                "readiness": raw["readiness"],
            }
            if adapter:
                item["evaluation"] = next(e for e in self.server.evaluators.catalog(raw["dataset_id"])
                    if e["id"] == adapter["evaluator_id"])
            if raw["runnable"] and item["ready"]:
                item["episodes"] = self.server.datasets.episodes(raw["dataset_id"])
                ready_datasets.append(item)
            else:
                unavailable_datasets.append(item)
        ready_datasets.sort(key=lambda item: item["id"])
        available_ids = {item["id"] for item in ready_datasets}
        dsh_models = [{**raw, "id": raw["model_id"], "model_source": "dsh_directory",
            "execution_available": bool(raw.get("configured")) and raw.get("directory_available") is not False}
            for raw in self.server.model_directory.catalog()]
        roles_ready = all(any(item["execution_available"] and model_supports_role(item, role)
            for item in dsh_models) for role in ("propose", "judge"))
        return {
            "schema_version": "ecologyrsi-dsh.runtime-catalog/5",
            "run_parameters": run_parameter_contract(),
            "domain_packs": [{"id": "greenhouse_environment@1", "label": "温室环境历史回放模型包",
                "description": "针对温度、相对湿度和二氧化碳浓度开展时间前向预测进化。"}],
            "datasets": ready_datasets, "unavailable_datasets": unavailable_datasets,
            "strategies": [{**item, "available": roles_ready} for item in self.server.strategy_router.catalog()],
            "model_workflows": [{"id": "research_compile_evolve@1", "label": "模型调研—编译—进化",
                "description": "研究计划与候选程序均由原生角色生成，科学评分由宿主独立执行。", "autonomous": True}],
            "prediction_selection_policy": RUNTIME_PREDICTION_POLICY,
            "runtime_evaluator_id": RUNTIME_EVALUATOR_ID,
            "prediction_models": sorted((item for item in self.server.evaluators.predictor_catalog()
                if available_ids.intersection(item["dataset_ids"])), key=lambda item: item["id"]),
            "evaluators": sorted((item for item in self.server.evaluators.catalog()
                if available_ids.intersection(item["dataset_ids"])), key=lambda item: item["id"]),
            "dsh_models": dsh_models,
            "dsh": {"harness_execution": "dsh_native_agent_runtime",
                "official_harness_agent_loop": self.server.dsh_native_runtime is not None,
                "connected": self.server.dsh_native_runtime is not None,
                "configured": bool(dsh_models), "environment": "dsh_native", "roles_ready": roles_ready,
                "capabilities": ["training.data.read", "evolution.run.create", "evolution.projection.read",
                    "evaluation.samples.read", "run.control", "run.archive", "run.delete"]},
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
        expected_protocol_digest = query.get("expected_data_protocol_digest", [None])[0]
        protocol_args = ({"expected_data_protocol_digest": expected_protocol_digest}
                         if expected_protocol_digest is not None else {})
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
            **protocol_args,
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
        reader = self.server.datasets.selection_view
        series = reader(
            dataset_id,
            episode_id,
            expected_dataset_digest=expected_dataset_digest,
            expected_split_manifest_digest=expected_split_digest,
            **protocol_args,
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
        unknown = set(query) - {"include_archived", "view", "limit", "before"}
        if unknown:
            raise ValueError("unknown run list query: " + ", ".join(sorted(unknown)))
        values = query.get("include_archived", ["false"])
        if len(values) != 1 or values[0].strip().casefold() not in {"true", "false"}:
            raise ValueError("include_archived must be true or false")
        include_archived = values[0].strip().casefold() == "true"
        view_values = query.get("view", ["summary"])
        if len(view_values) != 1 or view_values[0].strip().casefold() not in {
            "summary",
        }:
            raise ValueError("view must be summary")
        view = view_values[0].strip().casefold()
        projector = _run_summary_projection
        def page_integer(name: str, default: int | None) -> int | None:
            values = query.get(name)
            if values is None:
                return default
            if len(values) != 1 or not values[0].isascii() or not values[0].isdigit():
                raise ValueError(name + " must be a positive integer")
            return int(values[0])
        limit = page_integer("limit", 50)
        queries = RunQueries(self.server.director)
        before = page_integer("before", None)
        result = queries.summaries(projector, include_archived=include_archived, limit=limit, before=before)
        return {
            **result,
            "runs": [self._decorate_run_projection(item) for item in result['runs']],
            "view": view, "include_archived": include_archived, "limit": limit,
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
        query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
        view = query.get("view", ["overview"])[0]
        if (set(query) - {"view", "candidate_id"}
                or any(len(values) != 1 for values in query.values())
                or view not in {"monitor"} | WORKSPACE_VIEWS
                or ('candidate_id' in query and view not in {'candidate'})):
            raise ValueError("invalid run view query")
        admission_snapshot = self.server.sample_admission.snapshot(run_id)
        if view == 'overview':
            base = RunQueries(self.server.director).completed_projection(run_id, overview_projection)
            progress = base.get('execution_progress', {}).get('stage_progress')
            if isinstance(progress, dict) and str(progress.get('schema_version', '')).startswith('ecologyrsi-dsh.adaptive-progress/'):
                progress.update(_admission_progress_fields(admission_snapshot))
            return {'schema_version': 'ecologyrsi-dsh.browser-run-workspace/1',
                    'view': view, 'projection': self._decorate_run_projection(base)}
        if view in WORKSPACE_VIEWS:
            candidate_id = query.get('candidate_id', [None])[0]
            def build_workspace(state):
                _assert_http_scope(state)
                revision = state.events[-1].seq if state.events else 0
                section = self.server.workspace_cache.get_or_build(
                    (run_id, _workspace_revisions(state)[view], view, candidate_id),
                    lambda: workspace_section(state, view, candidate_id),
                )
                base = ({'run_id': run_id, 'projection_revision': revision,
                         'workspace_revisions': _workspace_revisions(state),
                         'status': state.run.status.value}
                        if view in {'candidate'} else _projection_json(state, admission_snapshot, overview_only=True))
                return {**base, **section}
            projection = RunQueries(self.server.director).completed_projection(
                run_id, build_workspace,
                cache_key=f'{workspace_section.__module__}.{workspace_section.__qualname__}:{view}:{candidate_id or ""}',
            )
            # Only completed projections persist across restarts. Mutable
            # admission and scheduler fields still come from the live host.
            progress = projection.get('execution_progress', {}).get('stage_progress')
            if isinstance(progress, dict) and str(progress.get('schema_version', '')).startswith('ecologyrsi-dsh.adaptive-progress/'):
                progress.update(_admission_progress_fields(admission_snapshot))
            return {
                'schema_version': 'ecologyrsi-dsh.browser-run-workspace/1',
                'view': view,
                'projection': self._decorate_run_projection(projection),
            }
        state = RunQueries(self.server.director).state(run_id)
        payload = _monitor_payload(state, admission_snapshot)
        payload["projection"] = self._decorate_run_projection(payload["projection"])
        return payload
