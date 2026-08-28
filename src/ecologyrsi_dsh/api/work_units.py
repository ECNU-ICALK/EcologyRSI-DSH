"""Lane-bounded scheduler for the Top-2 adaptive epoch protocol."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from ..core.models import RunStatus
from ..core.trajectory import TrajectoryStatus


def _formal_lane_priority(
    state: Any,
    candidate_id: str,
    selection_index: int,
) -> tuple[int, int, int] | None:
    """Return a deterministic priority for one finalist lane.

    A formal batch and its following local-edit/activation boundary form one
    pair.  A half-finished pair is resumed first; otherwise the lane with the
    fewest completed pairs advances.  The frozen Top-2 order is only the final
    tie-breaker, so replay produces the same choice without new scheduler
    state.
    """

    trajectory = state.trajectory_for(candidate_id)
    if trajectory is not None and trajectory.status is TrajectoryStatus.COMPLETED:
        return None
    completed_pairs = 0
    if trajectory is not None:
        for batch_index in range(trajectory.batch_count):
            if state.revision_activation_for(candidate_id, batch_index) is None:
                break
            completed_pairs += 1
    in_flight = (
        trajectory is not None
        and completed_pairs < trajectory.batch_count
        and state.formal_batch_for(candidate_id, completed_pairs) is not None
    )
    return (0 if in_flight else 1, completed_pairs, selection_index)


def execute_next_adaptive_work_unit(endpoint: Any, run_id: str) -> bool:
    """Advance one durable boundary and return whether work was committed.

    Screening is kept as the existing four-candidate operation. Once Top-2 is
    frozen, each invocation advances one durable boundary per admitted finalist
    lane. Candidate concurrency of one preserves deterministic rotation;
    concurrency of two lets both 50-origin evaluations share the run-level
    sample permits. A started batch remains highest priority until its
    local-edit boundary is activated. Every worker is drained before returning,
    keeping pause/cancel and replay boundaries observable.
    """

    from .formal_trajectory import (
        execute_next_formal_batch,
        execute_next_local_edit,
        ensure_formal_trajectory,
    )
    from .generation_execution import (
        _finalize_adaptive_generation,
        _freeze_adaptive_generation_inputs,
        _prepare_formal_finalists,
        _spawn_generation_candidates,
        _two_stage_screening_enabled,
    )
    from ..evolution.batches import start_generation_batch

    state = endpoint.server.director.state(run_id)
    if state.run.status is not RunStatus.RUNNING:
        return False
    batch = state.batch_for(state.run.generation)
    if batch is None:
        batch = start_generation_batch(endpoint.server.director, run_id)
        if not _spawn_generation_candidates(endpoint, run_id, batch):
            return False
        _freeze_adaptive_generation_inputs(endpoint, run_id, batch.generation)
        return True
    current = tuple(
        sorted(
            (
                item
                for item in state.candidates
                if item.generation == batch.generation
            ),
            key=lambda item: item.slot_index,
        )
    )
    if not _two_stage_screening_enabled(state, current):
        return False
    formal = state.formal_selection_for(batch.generation)
    if formal is None:
        finalists = _prepare_formal_finalists(
            endpoint,
            run_id,
            current,
            max_concurrency=int(state.task_manifest.metadata.get("candidate_concurrency") or 1),
        )
        return bool(finalists)

    selected_candidate_ids = tuple(formal.payload["selected_candidate_ids"])
    ranked_lanes = sorted(
        (
            (priority, candidate_id)
            for selection_index, candidate_id in enumerate(selected_candidate_ids)
            if (
                priority := _formal_lane_priority(
                    state, candidate_id, selection_index
                )
            )
            is not None
        ),
        key=lambda item: item[0],
    )
    def advance_lane(candidate_id: str, local_edit_lock: Lock | None) -> bool:
        state = endpoint.server.director.state(run_id)
        if state.run.status is not RunStatus.RUNNING:
            return False
        trajectory = ensure_formal_trajectory(endpoint, run_id, candidate_id)
        if trajectory.status.value == "completed":
            return False
        if execute_next_formal_batch(endpoint, run_id, candidate_id):
            return True
        if local_edit_lock is None:
            return execute_next_local_edit(endpoint, run_id, candidate_id)
        # Local-edit stages currently use one run/revision admission identity.
        # Keep those short DSH boundaries mutually exclusive while allowing
        # the expensive, independent 50-origin evaluations to overlap. Recheck
        # control after waiting so a queued edit never starts after pause or
        # cancellation.
        with local_edit_lock:
            state = endpoint.server.director.state(run_id)
            if state.run.status is not RunStatus.RUNNING:
                return False
            return execute_next_local_edit(endpoint, run_id, candidate_id)

    candidate_concurrency = int(
        state.task_manifest.metadata.get("candidate_concurrency") or 1
    )
    lane_limit = min(2, candidate_concurrency, len(ranked_lanes))
    changed = False
    if lane_limit >= 2:
        selected_lanes = tuple(ranked_lanes[:lane_limit])
        local_edit_lock = Lock()
        # Both finalists receive one lane-local durable boundary per scheduler
        # turn. Their origin workers still acquire the existing shared
        # RunSampleAdmission permits, so two 50-origin batches use at most the
        # frozen run-level sample_concurrency rather than 2x that limit.
        with ThreadPoolExecutor(
            max_workers=lane_limit,
            thread_name_prefix="adaptive-finalist",
        ) as executor:
            futures = tuple(
                (
                    candidate_id,
                    executor.submit(advance_lane, candidate_id, local_edit_lock),
                )
                for _priority, candidate_id in selected_lanes
            )
            results: list[bool] = []
            failures: list[Exception] = []
            # Observe results in deterministic frozen-lane order. The executor
            # drains every sibling before an error is re-raised, so no worker
            # can outlive the scheduler turn and mutate the ledger later.
            for _candidate_id, future in futures:
                try:
                    results.append(bool(future.result()))
                except Exception as exc:  # noqa: BLE001
                    failures.append(exc)
            if failures:
                raise failures[0]
            changed = any(results)
    else:
        for _priority, candidate_id in ranked_lanes:
            if advance_lane(candidate_id, None):
                changed = True
                # With one candidate lane permit, preserve the historical
                # half-pair-first, one-boundary scheduler behavior.
                break
    if changed:
        return True
    state = endpoint.server.director.state(run_id)
    if state.run.status is not RunStatus.RUNNING:
        return False
    analysis = _finalize_adaptive_generation(endpoint, run_id, batch)
    if analysis is None:
        return False
    endpoint.server.director.advance_generation(run_id)
    return True


__all__ = ["execute_next_adaptive_work_unit"]
