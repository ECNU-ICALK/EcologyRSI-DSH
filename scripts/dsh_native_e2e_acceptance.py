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


def _prediction_tool_event_count_is_valid(
    *,
    durable_event_count: int,
    completed_origin_summaries: list[dict[str, Any]],
    candidate_count: int,
    prediction_cell_budget_per_candidate: int,
) -> bool:
    """Return whether the public tool-event count matches sample evidence."""

    completed_event_floor = sum(
        int(summary.get("dsh_agent_prediction_tool_invocations") or 0)
        for summary in completed_origin_summaries
    )
    if (
        durable_event_count < 0
        or candidate_count < 1
        or prediction_cell_budget_per_candidate < 9
    ):
        return False
    origin_budget_per_candidate = prediction_cell_budget_per_candidate // 9
    frozen_budget_ceiling = candidate_count * origin_budget_per_candidate
    return completed_event_floor <= durable_event_count <= frozen_budget_ceiling


def _validate_prediction_tool_event_count(
    *,
    durable_event_count: int,
    completed_origin_summaries: list[dict[str, Any]],
    candidate_count: int,
    prediction_cell_budget_per_candidate: int,
) -> None:
    """Verify the public durable tool-event count against sample evidence."""

    if not _prediction_tool_event_count_is_valid(
        durable_event_count=durable_event_count,
        completed_origin_summaries=completed_origin_summaries,
        candidate_count=candidate_count,
        prediction_cell_budget_per_candidate=prediction_cell_budget_per_candidate,
    ):
        raise RuntimeError(
            "durable DSH prediction-tool event count is outside completed-to-budget bounds"
        )


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsh-origin", default="http://127.0.0.1:8848")
    parser.add_argument("--sidecar-origin", default="http://127.0.0.1:8777")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--strategy-model-id", default="")
    parser.add_argument("--review-model-id", default="")
    parser.add_argument(
        "--run-id",
        default="",
        help="verify an existing run instead of creating a new one",
    )
    parser.add_argument(
        "--samples-per-update",
        type=int,
        default=1521,
        help="prediction-cell budget; use 9 for a one-origin diagnostic smoke",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="maximum seconds to wait; real provider pacing can exceed 20 minutes",
    )
    args = parser.parse_args()
    if (
        args.generations < 1
        or args.candidates < 1
        or args.samples_per_update < 1
    ):
        parser.error("generations, candidates, and samples-per-update must be positive")

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
    run_id = args.run_id.strip()
    if run_id:
        existing = request_json(
            args.sidecar_origin,
            f"/api/runs/{quote(run_id, safe='')}",
            token=service_token,
        )
        projection = existing.get("projection", existing)
        configuration = projection.get("configuration", {})
        runtime = projection.get("dsh_runtime", {})
        if (
            configuration.get("execution_protocol") != PROTOCOL
            or runtime.get("execution_protocol") != PROTOCOL
        ):
            raise RuntimeError("existing run is not a DSH-native evolution")
        dataset_id = str(configuration.get("dataset_id") or "")
        strategy_model_id = str(configuration.get("strategy_model_id") or "")
        review_model_id = str(configuration.get("review_model_id") or "")
    else:
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
            raise RuntimeError(
                "one runnable real dataset and two configured DSH models are required"
            )
        dataset_by_id = {item_id(item): item for item in datasets}
        dataset_id = args.dataset_id.strip() or item_id(datasets[0])
        if dataset_id not in dataset_by_id:
            raise RuntimeError(f"requested dataset is not runnable: {dataset_id}")
        models = sorted(
            models,
            key=lambda item: (
                "glm" not in item_id(item).casefold(),
                "gpt-5.6-sol" not in item_id(item).casefold(),
                "flash" not in item_id(item).casefold(),
                item_id(item),
            ),
        )
        model_by_id = {item_id(item): item for item in models}
        strategy_model_id = args.strategy_model_id.strip() or item_id(models[0])
        review_model_id = args.review_model_id.strip() or item_id(models[1])
        for role, model_id in (
            ("strategy", strategy_model_id),
            ("review", review_model_id),
        ):
            if model_id not in model_by_id:
                raise RuntimeError(
                    f"requested {role} model is not executable: {model_id}"
                )
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
                # The real greenhouse evaluator needs 169 complete origins × nine
                # target/horizon cells before a generation is selection-eligible.
                "samples_per_update": args.samples_per_update,
                "sample_agent_batch_size": 9,
                "sample_concurrency": 2,
                "auto_progress": True,
                "auto_advance": "continuous",
                "allow_host_fallback": False,
                "idempotency_key": f"dsh-native-e2e-{uuid4().hex}",
            },
        )
        projection = created.get("projection", created)
        run_id = str(projection.get("run_id") or "")
        if not run_id:
            raise RuntimeError("run creation returned no run_id")
    print(f"run_id={run_id}", flush=True)
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
    expected_candidate_count = args.generations * args.candidates
    if len(candidates) != expected_candidate_count:
        raise RuntimeError(
            "completed run has an unexpected candidate count: "
            f"expected {expected_candidate_count}, got {len(candidates)}"
        )
    genomes = sum(
        1
        for item in candidates
        if isinstance(item, dict) and item.get("genome", {}).get("available") is True
    )
    origin_summaries = [
        item["metrics"]["sample_execution"]
        for item in candidates
        if isinstance(item, dict)
        and isinstance(item.get("metrics"), dict)
        and isinstance(item["metrics"].get("sample_execution"), dict)
    ]
    if genomes != len(candidates):
        raise RuntimeError("one or more candidates have no durable Genome")
    cohort_digests_by_generation: dict[int, set[str]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(
            candidate.get("metrics"), dict
        ):
            continue
        cohort_digest = candidate["metrics"].get("feedback_update_cohort_digest")
        if not isinstance(cohort_digest, str) or len(cohort_digest) != 64:
            raise RuntimeError("candidate has no verifiable feedback cohort digest")
        generation = int(candidate.get("generation") or 0)
        cohort_digests_by_generation.setdefault(generation, set()).add(cohort_digest)
    if any(len(digests) != 1 for digests in cohort_digests_by_generation.values()):
        raise RuntimeError("sibling candidates did not share one frozen feedback cohort")
    events = all_run_events(args.sidecar_origin, run_id, token=service_token)
    analyzed_generations = {
        int(event.get("payload", {}).get("generation"))
        for event in events
        if event.get("kind") == "GenerationAnalyzed"
        and isinstance(event.get("payload"), dict)
        and isinstance(event["payload"].get("generation"), int)
        and not isinstance(event["payload"].get("generation"), bool)
    }
    expected_generations = set(range(args.generations))
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
    prediction_tool_events = [
        event
        for event in events
        if event.get("kind") == "DshPredictionToolExecuted"
    ]
    structured_events = [
        event
        for event in events
        if event.get("kind") == "DshStructuredResultAccepted"
    ]
    if projection.get("sample_agent_protocol") == "dsh-strict-origin-bundle@3":
        if not origin_summaries:
            raise RuntimeError("strict origin smoke produced no sample evidence")
        for summary in origin_summaries:
            origins = int(summary.get("attempted_origin_samples") or 0)
            cells = int(summary.get("prediction_cell_count") or 0)
            cells_per_origin = int(summary.get("prediction_cells_per_origin") or 0)
            if (
                origins < 1
                or cells_per_origin != 9
                or cells != origins * cells_per_origin
                or summary.get("strict_agent_chain_pass") is not True
            ):
                raise RuntimeError("strict origin vector evidence is incomplete")
            if any(
                int(summary.get(name) or 0) != origins
                for name in (
                    "remote_planner_invocations",
                    "registered_prediction_tool_invocations",
                    "dsh_agent_prediction_tool_invocations",
                    "remote_reflection_invocations",
                )
            ) or int(summary.get("remote_critic_invocations") or 0) < origins:
                raise RuntimeError(
                    "strict origin run did not execute one complete agent/tool chain per origin"
                )
        _validate_prediction_tool_event_count(
            durable_event_count=len(prediction_tool_events),
            completed_origin_summaries=origin_summaries,
            candidate_count=len(candidates),
            prediction_cell_budget_per_candidate=int(
                projection.get("configuration", {}).get("samples_per_update")
                or args.samples_per_update
            ),
        )
    runtime = projection.get("dsh_runtime", {})
    skill_runtime = _verified_skill_runtime(
        runtime,
        structured_event_count=len(structured_events),
    )
    model_usage = projection.get("model_usage", {})
    sample_budget_class = projection.get("configuration", {}).get(
        "sample_budget_class"
    )
    if sample_budget_class == "diagnostic_smoke":
        champion_events = [
            event
            for event in events
            if event.get("kind") == "GenerationChampionSelected"
            and isinstance(event.get("payload"), dict)
        ]
        if (
            projection.get("outcome") != "diagnostic_smoke_completed"
            or projection.get("termination_reason")
            != "diagnostic_smoke_completed_no_promotion"
            or projection.get("best_candidate_id") is not None
            or any(
                candidate.get("status") == "promoted"
                for candidate in candidates
                if isinstance(candidate, dict)
            )
            or any(
                event["payload"].get("champion_candidate_id") is not None
                for event in champion_events
            )
        ):
            raise RuntimeError(
                "diagnostic run crossed the formal promotion boundary"
            )
    print(f"protocol={runtime.get('execution_protocol')}")
    print(f"dataset_id={dataset_id}")
    print(f"strategy_model_id={strategy_model_id}")
    print(f"review_model_id={review_model_id}")
    print(f"genomes={genomes}")
    print(f"python_model_requests={model_usage.get('call_count', 0)}")
    print(f"dsh_agent_sessions>={len(runtime.get('preset_ids', []))}")
    print(
        "skill_first_stages="
        f"{skill_runtime.get('verified_call_count', 0)} verified calls/"
        f"{len(skill_runtime.get('skills', []))} skills"
    )
    print(
        "autonomous_generation_contract="
        f"{len(analyzed_generations)} analyzed/search/research/reflection cycles"
    )
    if origin_summaries:
        print(
            "origin_vector_contract="
            + ",".join(
                (
                    f"{int(summary.get('attempted_origin_samples') or 0)} origins/"
                    f"{int(summary.get('prediction_cell_count') or 0)} cells/"
                    f"{int(summary.get('dsh_agent_prediction_tool_invocations') or 0)} agent tool calls"
                )
                for summary in origin_summaries
            )
        )
        print(f"durable_prediction_tool_events={len(prediction_tool_events)}")
    print("workflow_reconciliation=separately_verified_by_fault_injection")
    print("reward_contract=unchanged")
    print(f"sample_budget_class={sample_budget_class}")
    print(
        "scientific_result="
        + (
            "diagnostic_only_no_promotion"
            if sample_budget_class == "diagnostic_smoke"
            else "selection_only"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
