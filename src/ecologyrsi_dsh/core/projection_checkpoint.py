"""Disposable, versioned JSON checkpoints for the native event projection.

Only explicitly registered domain dataclasses/enums can be decoded. No pickle,
module loading from stored names, or executable serialization is accepted.
The event ledger remains authoritative; missing or corrupt caches are ignored.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile

from . import state as state_module
from .trajectory import (
    EvaluationPhase, EvaluationScope, FormalBatchComparisonDecision, RevisionStatus,
)
from ..evaluators.epoch_cohorts import PlannedBatch, PlannedCohort, PlannedOrigin
from ..knowledge.models import KnowledgeCard
from ..knowledge.autonomous_cycle import CandidateDirection


@lru_cache(maxsize=1)
def _types():
    # Nested records are not necessarily imported by state_module itself.
    # Keep the allowlist explicit; stored names never trigger imports.
    values = (*vars(state_module).values(), EvaluationScope, PlannedBatch,
              PlannedCohort, PlannedOrigin, KnowledgeCard, EvaluationPhase,
              FormalBatchComparisonDecision, RevisionStatus, CandidateDirection)
    return {value.__module__ + '.' + value.__qualname__: value for value in values
            if isinstance(value, type) and (is_dataclass(value) or issubclass(value, Enum))}


@lru_cache(maxsize=1)
def _version():
    root = Path(__file__).resolve().parents[1]
    hashes = [(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*.py'))]
    return hashlib.sha256(json.dumps(hashes).encode()).hexdigest()


def _encode(value):
    if isinstance(value, Enum):
        return {'type': 'enum', 'class': value.__class__.__module__ + '.' + value.__class__.__qualname__, 'value': value.value}
    if is_dataclass(value):
        name = value.__class__.__module__ + '.' + value.__class__.__qualname__
        if name not in _types():
            raise ValueError('unregistered_checkpoint_type')
        return {'type': 'record', 'class': name, 'fields': {f.name: _encode(getattr(value, f.name)) for f in fields(value) if f.init}}
    if isinstance(value, Mapping):
        return {'type': 'map', 'items': [[_encode(k), _encode(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list, set, frozenset)):
        kind = 'list' if isinstance(value, list) else type(value).__name__
        return {'type': kind, 'items': [_encode(item) for item in value]}
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError('unsupported_checkpoint_value')


def _decode(value):
    if not isinstance(value, dict):
        if value is None or type(value) in (str, bool, int, float):
            return value
        raise ValueError('invalid_checkpoint_value')
    kind = value['type']
    if kind in ('enum', 'record'):
        cls = _types()[value['class']]
        if kind == 'enum' and issubclass(cls, Enum):
            return cls(value['value'])
        if kind == 'record' and is_dataclass(cls):
            return cls(**{key: _decode(item) for key, item in value['fields'].items()})
        raise ValueError('checkpoint_type_mismatch')
    if kind == 'map':
        return {_decode(k): _decode(v) for k, v in value['items']}
    if kind in ('tuple', 'list', 'set', 'frozenset'):
        cls = {'tuple': tuple, 'list': list, 'set': set, 'frozenset': frozenset}[kind]
        return cls(_decode(item) for item in value['items'])
    raise ValueError('invalid_checkpoint_tag')


class ProjectionCheckpoints:
    interval = 128

    def __init__(self, ledger):
        self.ledger = ledger
        self.root = None if ledger.path == ':memory:' else Path(ledger.path).resolve().with_name(Path(ledger.path).name + '.read-models')
        self.saved = {}

    def _path(self, run_id):
        return self.root / (hashlib.sha256(run_id.encode()).hexdigest() + '.json')

    def load(self, run_id, latest_seq):
        if self.root is None:
            return None
        try:
            envelope = json.loads(self._path(run_id).read_text(encoding='utf-8'))
            payload = envelope['payload']
            if envelope['version'] != _version() or hashlib.sha256(payload.encode()).hexdigest() != envelope['sha256']:
                return None
            raw = json.loads(payload)
            if raw['run_id'] != run_id or raw['last_seq'] > latest_seq:
                return None
            reducer = state_module.RunStateReducer.__new__(state_module.RunStateReducer)
            reducer.__dict__.update(_decode(raw['state']))
            reducer.created = reducer.events[0]
            reducer.events_by_seq = {event.seq: event for event in reducer.events}
            if reducer.events[-1].seq != raw['last_seq']:
                return None
            if not self.ledger.checkpoint_boundary_matches(reducer.created, reducer.events[-1]):
                return None
            reducer.snapshot()  # Domain invariants still apply to restored summaries.
            self.saved[run_id] = len(reducer.events)
            return reducer
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
            return None

    def capture(self, reducer):
        run_id = reducer.created.run_id
        if self.root is None or self.ledger._connection.in_transaction or len(reducer.events) - self.saved.get(run_id, 0) < self.interval:
            return None
        self.saved[run_id] = len(reducer.events)
        try:
            state = {key: value for key, value in reducer.__dict__.items() if key not in ('created', 'events_by_seq')}
            payload = json.dumps({'run_id': run_id, 'last_seq': reducer.events[-1].seq, 'state': _encode(state)}, allow_nan=False, separators=(',', ':'))
            return reducer.created, reducer.events[-1], payload
        except (ValueError, TypeError, RecursionError):
            return None

    def save(self, capture):
        if capture is None:
            return
        first, last, payload = capture
        run_id = first.run_id
        if not self.ledger.checkpoint_boundary_matches(first, last):
            return
        path = None
        try:
            envelope = {'version': _version(), 'sha256': hashlib.sha256(payload.encode()).hexdigest(), 'payload': payload}
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile('w', dir=self.root, encoding='utf-8', delete=False) as handle:
                path = Path(handle.name)
                json.dump(envelope, handle, allow_nan=False)
                handle.flush()
            os.replace(path, self._path(run_id))
        except (OSError, ValueError, TypeError, RecursionError):
            # This derived cache must never turn a committed command into failure.
            pass
        finally:
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _summary_path(self, run_id, projector):
        # Different read models must coexist rather than overwrite the run
        # list's cache each time a browser opens its overview.
        identity = hashlib.sha256(projector.encode()).hexdigest()
        return self._path(run_id).with_suffix('.' + identity + '.summary.json')

    def summary(self, run_id, revision, projector):
        if self.root is None:
            return None
        try:
            raw = json.loads(self._summary_path(run_id, projector).read_text(encoding='utf-8'))
            if raw['revision'] != revision or raw['version'] != _version() or raw['projector'] != projector:
                return None
            payload = raw['payload']
            if hashlib.sha256(payload.encode()).hexdigest() != raw['sha256']:
                return None
            result = json.loads(payload)
            return result if isinstance(result, dict) else None
        except (OSError, KeyError, TypeError, ValueError):
            return None

    def save_summary(self, run_id, revision, projector, value):
        if self.root is None:
            return
        path = None
        try:
            payload = json.dumps(value, allow_nan=False, separators=(',', ':'))
            if len(payload.encode('utf-8')) > 16 * 1024 * 1024:
                return
            raw = {'revision': revision, 'version': _version(), 'projector': projector,
                   'payload': payload, 'sha256': hashlib.sha256(payload.encode()).hexdigest()}
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile('w', dir=self.root, encoding='utf-8', delete=False) as handle:
                path = Path(handle.name)
                json.dump(raw, handle, allow_nan=False)
            os.replace(path, self._summary_path(run_id, projector))
            # Parameterized audit pages are disposable: bound their disk
            # footprint rather than retaining every pagination combination.
            pages = sorted(self.root.glob(self._path(run_id).stem + '.*.summary.json'),
                           key=lambda item: item.stat().st_mtime, reverse=True)
            total_bytes = 0
            for index, page in enumerate(pages):
                total_bytes += page.stat().st_size
                if index >= 64 or total_bytes > 32 * 1024 * 1024:
                    page.unlink(missing_ok=True)
        except (OSError, ValueError, TypeError):
            pass
        finally:
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
