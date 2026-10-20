#!/usr/bin/env python3
"""Run one bounded DSH-native evolution through the DSH and sidecar ports."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any
from uuid import uuid4
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
    timeout: float = 30.0,
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
            timeout=timeout,
        ) as response:
            value = json.loads(response.read())
    except HTTPError as error:
        raise RuntimeError(f"{path} returned HTTP {error.code}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"non-object response from {path}")
    return value


def item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("dataset_id") or item.get("episode_id") or item.get("model_id") or "")


def _verified_skill_runtime(
    runtime: dict[str, Any], *, structured_event_count: int
) -> dict[str, Any]:
    """Validate ledger-backed Skill evidence exposed by the public projection.

    Individual structured-result payloads are intentionally redacted by the
    public event endpoint.  The run projection derives this aggregate from the
    unredacted append-only ledger, so it is the public contract the acceptance
    client can safely verify.
    """

    skill_runtime = runtime.get("skill_invocation", {})
    if (
        structured_event_count < 1
        or runtime.get("first_call_verified") is not True
        or not isinstance(skill_runtime, dict)
        or skill_runtime.get("all_verified") is not True
        or skill_runtime.get("verified_call_count") != structured_event_count
    ):
        raise RuntimeError("DSH Skill invocation projection is incomplete")
    return skill_runtime


def all_run_events(origin: str, run_id: str, *, token: str) -> list[dict[str, Any]]:
    """Read the complete paginated public event stream for one run."""

    events: list[dict[str, Any]] = []
    after = "0"
    while True:
        page = request_json(
            origin,
            f"/api/runs/{quote(run_id, safe='')}/events?after={quote(after, safe='')}",
            token=token,
        )
        rows = page.get("events", [])
        if not isinstance(rows, list) or any(not isinstance(item, dict) for item in rows):
            raise RuntimeError("run event endpoint returned an invalid page")
        events.extend(rows)
        if page.get("has_more") is not True:
            return events
        next_cursor = str(page.get("next_cursor") or "").strip()
        if not next_cursor or next_cursor == after:
            raise RuntimeError("run event pagination did not advance")
        after = next_cursor


def configured_role_models(catalog: dict[str, Any], role: str) -> list[dict[str, Any]]:
    """Use only the authoritative DSH directory and explicit role binding."""
    return [item for item in catalog.get("dsh_models", [])
            if isinstance(item, dict) and item.get("configured") is True
            and item.get("execution_available") is True and role in item.get("roles", [])]


def _current_execution_evidence(projection: dict[str, Any], generations: int) -> dict[str, Any]:
    if projection.get("status") != "completed":
        raise RuntimeError("native evolution did not complete")
    config = projection.get("configuration", {})
    if config.get("optimization_protocol") != "joint_plugin_epoch@2":
        raise RuntimeError("acceptance requires the current joint protocol")
    diagnostics = projection.get("execution_diagnostics", {})
    origins = diagnostics.get("adaptive_all_settled_origins")
    cells = diagnostics.get("adaptive_scored_cells")
    # Five screen arms, two formal arms and two holdout arms; one optional
    # paired recheck. Replans are not new origins; failed predictions cannot
    # turn an incomplete run into a passing engineering acceptance.
    if (type(origins) is not int or not 200 * generations <= origins <= 300 * generations
            or type(cells) is not int or cells != origins * 9):
        raise RuntimeError("complete current vector execution evidence is missing")
    return {"candidate_origin_executions": origins, "scoring_cells": cells}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsh-origin", default="http://127.0.0.1:8848")
    parser.add_argument("--sidecar-origin", default="http://127.0.0.1:8777")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=4, choices=(4,))
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--episode-id", default="")
    parser.add_argument("--strategy-model-id", default="")
    parser.add_argument("--review-model-id", default="")
    parser.add_argument("--run-id", default="", help="verify an existing current run; no new model work")
    parser.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()
    if not 1 <= args.generations <= 50 or args.timeout <= 0:
        parser.error("generations must be 1–50 and timeout must be positive")
    runtime_token = os.environ.get("ECOLOGYRSI_DSH_RUNTIME_TOKEN", "")
    service_token = os.environ.get("ECOLOGYRSI_SERVICE_TOKEN", "")
    capabilities = request_json(args.dsh_origin,
        "/api/ecology-agent-runtime/v1/capabilities", token=runtime_token)
    if capabilities.get("ready") is not True:
        raise RuntimeError("DSH runtime capabilities are not ready")
    health = request_json(args.sidecar_origin, "/api/health", token=service_token)
    if health.get("ok") is not True:
        raise RuntimeError("Python sidecar is not healthy")
    run_id = args.run_id.strip()
    if not run_id:
        catalog = request_json(args.sidecar_origin, "/api/catalog", token=service_token)
        datasets = [item for item in catalog.get("datasets", [])
                    if isinstance(item, dict) and item.get("training_selectable") is True
                    and item.get("readiness", {}).get("ready") is True]
        dataset_id = args.dataset_id.strip() or (item_id(datasets[0]) if datasets else "")
        dataset = next((item for item in datasets if item_id(item) == dataset_id), None)
        if dataset is None:
            raise RuntimeError("a runnable observed greenhouse dataset is required")
        episodes = dataset.get("episodes", [])
        episode_id = args.episode_id.strip() or (item_id(episodes[0]) if episodes else "")
        if not episode_id:
            raise RuntimeError("an explicit available episode is required")
        selected = {}
        for role, supplied in (("propose", args.strategy_model_id), ("judge", args.review_model_id)):
            models = configured_role_models(catalog, role)
            model_id = supplied.strip() or (item_id(models[0]) if models else "")
            if not any(item_id(item) == model_id for item in models):
                raise RuntimeError(f"an executable configured {role} model is required")
            selected[role] = model_id
        # Separate role identities may intentionally use the same model route.
        body = {"execution_protocol": PROTOCOL, "optimization_protocol": "joint_plugin_epoch@2",
                "dataset_id": dataset_id, "episode_id": episode_id,
                "strategy_model_id": selected["propose"], "review_model_id": selected["judge"],
                "autonomous_mode": True, "knowledge_online_enabled": False,
                "budget": {"max_generations": args.generations,
                           "candidates_per_generation": 4, "max_candidates": 4 * args.generations},
                "candidate_concurrency": 2, "sample_concurrency": 4,
                "auto_progress": True, "auto_advance": "continuous",
                "idempotency_key": f"dsh-native-e2e-{uuid4().hex}"}
        preflight = request_json(args.sidecar_origin, "/api/model-preflight", method="POST",
                                 token=service_token, body=body, timeout=args.timeout)
        if preflight.get("passed") is not True:
            raise RuntimeError("standard four-role native preflight did not pass")
        created = request_json(args.sidecar_origin, "/api/runs", method="POST", token=service_token, body=body)
        run_id = str(created.get("projection", {}).get("run_id") or "")
        if not run_id:
            raise RuntimeError("run creation returned no run_id")
    print(f"run_id={run_id}", flush=True)
    deadline = time.monotonic() + args.timeout
    route = f"/api/runs/{quote(run_id, safe='')}"
    while time.monotonic() < deadline:
        current = request_json(args.sidecar_origin, route + "?view=monitor", token=service_token)
        projection = current.get("projection", {})
        if projection.get("status") in {"completed", "failed", "cancelled", "paused"}:
            break
        time.sleep(5.0)
    else:
        raise TimeoutError("native acceptance observation timed out; run state was not changed")
    overview = request_json(args.sidecar_origin, route + "?view=overview", token=service_token)
    projection.update(overview.get("projection", {}))
    generations = projection.get("total_generations")
    if type(generations) is not int or not 1 <= generations <= 50:
        raise RuntimeError("run projection has no valid frozen generation count")
    evidence = _current_execution_evidence(projection, generations)
    configuration = projection["configuration"]
    dataset_id = configuration["dataset_id"]
    strategy_model_id = configuration["strategy_model_id"]
    review_model_id = configuration["review_model_id"]
    if configuration.get("execution_protocol") != PROTOCOL:
        raise RuntimeError("existing run is not native")
    events = all_run_events(args.sidecar_origin, run_id, token=service_token)
    analyzed_generations = {
        int(event.get("payload", {}).get("generation"))
        for event in events
        if event.get("kind") == "GenerationAnalyzed"
        and isinstance(event.get("payload"), dict)
        and isinstance(event["payload"].get("generation"), int)
        and not isinstance(event["payload"].get("generation"), bool)
    }
    expected_generations = set(range(generations))
    if analyzed_generations != expected_generations:
        raise RuntimeError(
            "generation-level analysis coverage mismatch: "
            f"expected {sorted(expected_generations)}, "
            f"got {sorted(analyzed_generations)}"
        )
    for kind in (
        "GenerationSearchPlanned",
        "GenerationKnowledgeRetrieved",
        "GenerationResearchIterated",
        "GenerationAnalyzed",
        "GenerationReflected",
        "GenerationKnowledgeAssessed",
    ):
        recorded_generations = {
            int(event.get("payload", {}).get("generation"))
            for event in events
            if event.get("kind") == kind
            and isinstance(event.get("payload"), dict)
            and isinstance(event["payload"].get("generation"), int)
            and not isinstance(event["payload"].get("generation"), bool)
        }
        if recorded_generations != analyzed_generations:
            raise RuntimeError(
                f"{kind} does not cover every completed generation"
            )
    generation_events: dict[str, dict[int, dict[str, Any]]] = {}
    for kind in (
        "GenerationSearchPlanned",
        "GenerationResearchIterated",
        "GenerationAnalyzed",
        "GenerationReflected",
    ):
        by_generation: dict[int, dict[str, Any]] = {}
        for event in events:
            payload = event.get("payload")
            if event.get("kind") != kind or not isinstance(payload, dict):
                continue
            generation = payload.get("generation")
            if isinstance(generation, bool) or not isinstance(generation, int):
                continue
            if generation in by_generation:
                raise RuntimeError(f"{kind} has duplicate generation events")
            by_generation[generation] = payload
        generation_events[kind] = by_generation
    for generation in sorted(expected_generations):
        search = generation_events["GenerationSearchPlanned"][generation]
        research = generation_events["GenerationResearchIterated"][generation]
        analysis = generation_events["GenerationAnalyzed"][generation]
        reflection = generation_events["GenerationReflected"][generation]
        analysis_digest = analysis.get("analysis_digest")
        reflection_digest = reflection.get("reflection_digest")
        if (
            not isinstance(analysis_digest, str)
            or len(analysis_digest) != 64
            or not isinstance(reflection_digest, str)
            or len(reflection_digest) != 64
            or reflection.get("analysis_digest") != analysis_digest
        ):
            raise RuntimeError(
                f"generation {generation} analysis/reflection digest chain is invalid"
            )
        expected_analysis_source = (
            generation_events["GenerationAnalyzed"][generation - 1].get(
                "analysis_digest"
            )
            if generation > 0
            else None
        )
        expected_reflection_source = (
            generation_events["GenerationReflected"][generation - 1].get(
                "reflection_digest"
            )
            if generation > 0
            else None
        )
        if (
            search.get("source_analysis_digest") != expected_analysis_source
            or search.get("source_reflection_digest")
            != expected_reflection_source
            or research.get("source_analysis_digest")
            != expected_analysis_source
        ):
            raise RuntimeError(
                f"generation {generation} did not consume the preceding "
                "analysis/reflection evidence"
            )
    reflected = [
        event
        for event in events
        if event.get("kind") == "GenerationReflected"
    ]
    if any(
        not isinstance(event.get("payload"), dict)
        or int(event["payload"].get("direction_count") or 0) < 2
        or not isinstance(event["payload"].get("reflection_digest"), str)
        or len(event["payload"]["reflection_digest"]) != 64
        for event in reflected
    ):
        raise RuntimeError("generation reflection evidence is incomplete")
    structured_events = [event for event in events if event.get("kind") == "DshStructuredResultAccepted"]
    runtime = projection.get("dsh_runtime", {})
    skill_runtime = _verified_skill_runtime(runtime, structured_event_count=len(structured_events))
    usage = runtime.get("provider_usage", {})
    print(json.dumps({"protocol": PROTOCOL, "dataset_id": dataset_id,
        "strategy_model_id": strategy_model_id, "review_model_id": review_model_id,
        **evidence, "skill_verified_calls": skill_runtime["verified_call_count"],
        "reported_tokens": usage.get("total_tokens"), "accounting_coverage": usage.get("accounting_coverage"),
        "prediction_tool_events": sum(event.get("kind") == "DshPredictionToolExecuted" for event in events),
        "analyzed_generations": len(analyzed_generations), "scientific_result": "selection_only"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
