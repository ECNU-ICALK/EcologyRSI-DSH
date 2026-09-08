"""Lane-bounded scheduler for the Top-2 adaptive epoch protocol."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from threading import Lock
from typing import Any
from .ports import GenerationRuntime

from ..core.models import CandidateRole, RunStatus
from ..core.trajectory import TrajectoryStatus
from ..evolution.schedule import (
    OPTIMIZATION_PROTOCOL,
    PAIRED_LOCAL_EVALUATION_MODE,
    OptimizationSchedule,
)


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
    raw_schedule = getattr(state.task_manifest, "metadata", {}).get(
        "optimization_schedule"
    )
    paired = False
    if isinstance(raw_schedule, Mapping):
        paired = (
            OptimizationSchedule.from_dict(raw_schedule).local_evaluation_mode
            == PAIRED_LOCAL_EVALUATION_MODE
        )
    completed_pairs = 0
    if trajectory is not None:
        transition_count = (
            trajectory.batch_count - 1 if paired else trajectory.batch_count
        )
        for batch_index in range(transition_count):
            if state.revision_activation_for(candidate_id, batch_index) is None:
                break
            completed_pairs += 1
    current_batch_index = completed_pairs
    in_flight = (
        trajectory is not None
        and current_batch_index < trajectory.batch_count
        and state.formal_batch_for(candidate_id, current_batch_index) is not None
    )
    return (0 if in_flight else 1, completed_pairs, selection_index)


def execute_next_adaptive_work_unit(services: GenerationRuntime, run_id: str) -> bool:
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

    state = services.director.state(run_id)
    if state.run.status is not RunStatus.RUNNING:
        return False
    batch = state.batch_for(state.run.generation)
    if batch is None:
        batch = start_generation_batch(services.director, run_id)
        if not _spawn_generation_candidates(services, run_id, batch):
            return False
        _freeze_adaptive_generation_inputs(services, run_id, batch.generation)
        return True
    if (
        state.task_manifest.metadata.get("optimization_protocol")
        == OPTIMIZATION_PROTOCOL
        and state.task_manifest.metadata.get("cohort_capacity_enforced") is True
    ):
        generation_candidates = tuple(
            item
            for item in state.candidates
            if item.generation == batch.generation
            and getattr(item, "role", CandidateRole.SEARCH)
            in {CandidateRole.SEARCH, CandidateRole.SEARCH.value}
        )
        inputs_complete = (
            len(generation_candidates) == batch.batch_size
            and all(
                state.initial_revision_for(item.candidate_id) is not None
                for item in generation_candidates
            )
            and state.run_adaptation_cohort is not None
            and state.generation_cohort_for(batch.generation) is not None
        )
        if not inputs_complete:
            # GenerationBatchStarted is the first durable boundary of an epoch.
            # A crash immediately after it (or midway through spawning R0/cohort
            # inputs) must resume that same batch rather than falling through to
            # the screening gate and reporting a false no-progress condition.
            if not _spawn_generation_candidates(services, run_id, batch):
                return False
            _freeze_adaptive_generation_inputs(services, run_id, batch.generation)
            return True
    current = tuple(
        sorted(
            (
                item
                for item in state.candidates
                if item.generation == batch.generation
                and getattr(item, "role", CandidateRole.SEARCH)
                in {CandidateRole.SEARCH, CandidateRole.SEARCH.value}
            ),
            key=lambda item: item.slot_index,
        )
    )
    if not _two_stage_screening_enabled(state, current):
        return False
    formal = state.formal_selection_for(batch.generation)
    if formal is None:
        finalists = _prepare_formal_finalists(
            services,
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
        state = services.director.state(run_id)
        if state.run.status is not RunStatus.RUNNING:
            return False
        trajectory = ensure_formal_trajectory(services, run_id, candidate_id)
        if trajectory.status.value == "completed":
            return False
        if execute_next_formal_batch(services, run_id, candidate_id):
            return True
        if local_edit_lock is None:
            return execute_next_local_edit(services, run_id, candidate_id)
        # Local-edit stages currently use one run/revision admission identity.
        # Keep those short DSH boundaries mutually exclusive while allowing
        # the expensive, independent 50-origin evaluations to overlap. Recheck
        # control after waiting so a queued edit never starts after pause or
        # cancellation.
        with local_edit_lock:
            state = services.director.state(run_id)
            if state.run.status is not RunStatus.RUNNING:
                return False
            return execute_next_local_edit(services, run_id, candidate_id)

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
    state = services.director.state(run_id)
    if state.run.status is not RunStatus.RUNNING:
        return False
    analysis = _finalize_adaptive_generation(services, run_id, batch)
    if analysis is None:
        return False
    services.director.advance_generation(run_id)
    return True


__all__ = ["execute_next_adaptive_work_unit"]
