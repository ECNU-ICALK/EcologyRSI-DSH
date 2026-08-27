"""One-step scheduler for the Top-2 adaptive epoch protocol."""

from __future__ import annotations

from typing import Any

from ..core.models import RunStatus


def execute_next_adaptive_work_unit(endpoint: Any, run_id: str) -> bool:
    """Advance one durable boundary and return whether work was committed.

    Screening is kept as the existing four-candidate operation. Once Top-2 is
    frozen, each invocation advances at most one 50-origin batch (or its local
    edit) for each finalist, then yields to the auto-progress FIFO. This keeps
    pause/cancel responsive and makes every scheduler turn observable.
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

    changed = False
    for candidate_id in tuple(formal.payload["selected_candidate_ids"]):
        state = endpoint.server.director.state(run_id)
        trajectory = ensure_formal_trajectory(endpoint, run_id, candidate_id)
        if trajectory.status.value == "completed":
            continue
        if execute_next_formal_batch(endpoint, run_id, candidate_id):
            changed = True
            # A single lane boundary is the work unit. The next FIFO turn
            # advances the sibling lane and prevents one finalist monopolising
            # the run-level sample permit.
            break
        if execute_next_local_edit(endpoint, run_id, candidate_id):
            changed = True
            break
    if changed:
        return True
    state = endpoint.server.director.state(run_id)
    analysis = _finalize_adaptive_generation(endpoint, run_id, batch)
    if analysis is None:
        return False
    endpoint.server.director.advance_generation(run_id)
    return True


__all__ = ["execute_next_adaptive_work_unit"]
