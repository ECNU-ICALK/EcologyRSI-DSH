"""Read use cases shared by HTTP and CLI; projections never mutate runs."""
from __future__ import annotations
from dataclasses import dataclass
from ..core.director import EvolutionDirector
from ..core.state import RunState


@dataclass(frozen=True)
class RunPage:
    states: tuple[RunState, ...]
    next_cursor: int | None
    archived_count: int


@dataclass(frozen=True)
class RunQueries:
    director: EvolutionDirector

    def state(self, run_id: str) -> RunState:
        return self.director.state(run_id)

    def completed_projection(self, run_id, projector, *, cache_key=None):
        """Read a validated completed-run snapshot without restoring all events.

        Only a completed run is persisted. A changed ledger revision, projector
        or source version causes normal validated replay before rebuilding it.
        Runtime-only fields are added by the caller after this immutable read.
        """
        cache = self.director._projection_checkpoints
        identity = cache_key or projector.__module__ + '.' + projector.__qualname__
        revision = self.director.ledger.latest_run_seq(run_id)
        payload = cache.summary(run_id, revision, identity)
        if (payload is not None and payload.get('status') == 'completed'
                and payload.get('run_id') == run_id
                and payload.get('projection_revision') == revision):
            return payload
        state = self.state(run_id)
        payload = projector(state)
        if payload.get('status') == 'completed':
            cache.save_summary(run_id, state.events[-1].seq, identity, payload)
        return payload

    def completed_read(self, run_id, reader, *, cache_key):
        """Cache a bounded public read while keeping run identity outside its value."""
        def project(state):
            return {'run_id': run_id, 'status': state.run.status.value,
                    'projection_revision': state.events[-1].seq, 'value': reader(state)}
        return self.completed_projection(run_id, project, cache_key=cache_key)['value']

    def page(self, *, include_archived: bool = False, limit: int = 50,
             before: int | None = None) -> RunPage:
        ids, cursor = self.director.ledger.run_page(include_archived=include_archived, limit=limit, before=before)
        return RunPage(tuple(self.state(run_id) for run_id in ids), cursor,
                       self.director.ledger.archived_count())

    def summaries(self, projector, *, include_archived: bool = False, limit: int = 50,
                  before: int | None = None) -> dict:
        facts, cursor = self.director.ledger.run_index(
            include_archived=include_archived, limit=limit, before=before)
        return {'runs': [projector(row) for row in facts], 'next_cursor': cursor,
                'archived_count': self.director.ledger.archived_count()}
