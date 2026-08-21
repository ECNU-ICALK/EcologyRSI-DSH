#!/usr/bin/env python3
"""Run one bounded DSH-native evolution through the DSH and sidecar ports."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROTOCOL = "dsh_native_plugin_evolution@1"


def request_json(
    origin: str,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    token: str = "",
) -> dict[str, Any]:
    encoded = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    headers = {"Accept": "application/json"}
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(
            Request(origin.rstrip("/") + path, data=encoded, method=method, headers=headers),
            timeout=30,
        ) as response:
            value = json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {error.code}: {detail}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"non-object response from {path}")
    return value


def item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("dataset_id") or item.get("model_id") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsh-origin", default="http://127.0.0.1:8848")
    parser.add_argument("--sidecar-origin", default="http://127.0.0.1:8777")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if args.generations < 1 or args.candidates < 1:
        parser.error("generations and candidates must be positive")

    runtime_token = os.environ.get("ECOLOGYRSI_DSH_RUNTIME_TOKEN", "")
    service_token = os.environ.get("ECOLOGYRSI_SERVICE_TOKEN", "")
    capabilities = request_json(
        args.dsh_origin,
        "/api/ecology-agent-runtime/v1/capabilities",
        token=runtime_token,
    )
    if capabilities.get("ready") is not True:
        raise RuntimeError("DSH runtime capabilities are not ready")
    health = request_json(args.sidecar_origin, "/api/health", token=service_token)
    if health.get("ok") is not True:
        raise RuntimeError("Python sidecar is not healthy")
    catalog = request_json(args.sidecar_origin, "/api/catalog", token=service_token)
    datasets = [
        item
        for item in catalog.get("datasets", [])
        if isinstance(item, dict)
        and (
            item.get("ready") is True
            or (
                isinstance(item.get("readiness"), dict)
                and item["readiness"].get("ready") is True
            )
        )
        and item_id(item) != "generated-toy-series@1"
    ]
    configured_models = [
        item
        for item in catalog.get("dsh_models", catalog.get("models", []))
        if isinstance(item, dict)
        and item.get("configured", True) is not False
        and item.get("credential_configured", True) is not False
    ]
    available_models = [
        item
        for item in configured_models
        if item.get("available") is True or item.get("authenticated") is True
    ]
    models = available_models if len(available_models) >= 2 else configured_models
    if not datasets or len(models) < 2:
        raise RuntimeError("one runnable real dataset and two configured DSH models are required")
    dataset_id = item_id(datasets[0])
    models = sorted(
        models,
        key=lambda item: (
            "glm" not in item_id(item).casefold(),
            "gpt-5.6-sol" not in item_id(item).casefold(),
            "flash" not in item_id(item).casefold(),
            item_id(item),
        ),
    )
    strategy_model_id, review_model_id = item_id(models[0]), item_id(models[1])
    if not dataset_id or not strategy_model_id or not review_model_id:
        raise RuntimeError("catalog identities are incomplete")
    created = request_json(
        args.sidecar_origin,
        "/api/runs",
        method="POST",
        token=service_token,
        body={
            "execution_protocol": PROTOCOL,
            "dataset_id": dataset_id,
            "strategy_model_id": strategy_model_id,
            "review_model_id": review_model_id,
            "autonomous_mode": True,
            "rounds": args.generations,
            "candidates_per_generation": args.candidates,
            "max_candidates": args.generations * args.candidates,
            "samples_per_update": 9,
            "sample_agent_batch_size": 9,
            "sample_concurrency": 2,
            "auto_progress": True,
            "auto_advance": "continuous",
            "allow_host_fallback": False,
            "idempotency_key": f"dsh-native-e2e-{int(time.time())}",
        },
    )
    projection = created.get("projection", created)
    run_id = str(projection.get("run_id") or "")
    if not run_id:
        raise RuntimeError("run creation returned no run_id")
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        current = request_json(
            args.sidecar_origin,
            f"/api/runs/{quote(run_id, safe='')}",
            token=service_token,
        )
        projection = current.get("projection", current)
        if projection.get("status") in {"completed", "failed", "cancelled"}:
            break
        time.sleep(2.0)
    else:
        raise TimeoutError("DSH-native smoke timed out")
    if projection.get("status") != "completed":
        raise RuntimeError(
            "DSH-native smoke did not complete: "
            + str(projection.get("failure_reason") or projection.get("status"))
        )
    candidates = projection.get("candidates", [])
    genomes = sum(
        1
        for item in candidates
        if isinstance(item, dict) and item.get("genome", {}).get("available") is True
    )
    runtime = projection.get("dsh_runtime", {})
    model_usage = projection.get("model_usage", {})
    print(f"protocol={runtime.get('execution_protocol')}")
    print(f"genomes={genomes}")
    print(f"python_model_requests={model_usage.get('call_count', 0)}")
    print(f"dsh_agent_sessions>={len(runtime.get('preset_ids', []))}")
    print("workflow_reconciliation=separately_verified_by_fault_injection")
    print("reward_contract=unchanged")
    print("scientific_result=selection_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
