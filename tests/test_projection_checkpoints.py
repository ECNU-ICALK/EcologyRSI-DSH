import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ecologyrsi_dsh import EventLedger, EvolutionDirector
from ecologyrsi_dsh.core.state import project_run_state
from tests.test_core import manifest


class ProjectionCheckpointTests(unittest.TestCase):
    def test_parameterized_public_page_cache_is_bounded(self):
        from ecologyrsi_dsh.core.projection_checkpoint import ProjectionCheckpoints
        with tempfile.TemporaryDirectory() as tmp, EventLedger() as ledger:
            cache = ProjectionCheckpoints(ledger)
            cache.root = Path(tmp)
            for page in range(70):
                cache.save_summary('run', 1, f'page:{page}', {'page': page})
            self.assertLessEqual(len(list(cache.root.glob('*.summary.json'))), 64)
            self.assertIsNone(cache.summary('run', 1, 'page:0'))
            self.assertEqual(cache.summary('run', 1, 'page:69'), {'page': 69})

    def test_restart_loads_checkpoint_then_only_new_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            with EventLedger(Path(tmp)/'state.sqlite') as ledger:
                writer=EvolutionDirector(ledger)
                writer._projection_checkpoints.interval=1
                writer.create_run(manifest(1),run_id='checkpoint')
                first=writer.state('checkpoint')
                self.assertTrue(list((Path(tmp)/'state.sqlite.read-models').glob('*.json')))
                writer._projection_checkpoints.interval = 10000
                writer.start_run('checkpoint')
                restarted=EvolutionDirector(ledger)
                with patch.object(ledger,'events',wraps=ledger.events) as read:
                    state=restarted.state('checkpoint')
                read.assert_called_once_with('checkpoint',after_seq=first.events[-1].seq)
                self.assertEqual(state,project_run_state(ledger.events('checkpoint')))

    def test_corrupt_or_unknown_version_cache_falls_back_to_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            with EventLedger(Path(tmp)/'state.sqlite') as ledger:
                writer=EvolutionDirector(ledger);writer._projection_checkpoints.interval=1
                writer.create_run(manifest(1),run_id='checkpoint');writer.state('checkpoint')
                path=next((Path(tmp)/'state.sqlite.read-models').glob('*.json'))
                for corruption in ('invalid JSON',json.dumps({'version':'unknown','payload':'{}','sha256':'bad'})):
                    path.write_text(corruption)
                    restarted=EvolutionDirector(ledger)
                    with patch.object(ledger,'events',wraps=ledger.events) as read:
                        state=restarted.state('checkpoint')
                    read.assert_called_once_with('checkpoint',after_seq=0)
                    self.assertEqual(state,project_run_state(ledger.events('checkpoint')))

    def test_nested_formal_and_knowledge_records_round_trip(self):
        from ecologyrsi_dsh.core.projection_checkpoint import _encode, _decode
        from ecologyrsi_dsh.core.trajectory import (
            EvaluationScope, EvaluationPhase, FormalBatchComparisonDecision, RevisionStatus,
        )
        from ecologyrsi_dsh.evaluators.epoch_cohorts import PlannedOrigin, PlannedCohort, PlannedBatch
        from ecologyrsi_dsh.knowledge.models import KnowledgeCard
        from ecologyrsi_dsh.knowledge.autonomous_cycle import CandidateDirection
        origin = PlannedOrigin('a' * 64, 'dataset', 'episode', 10, 10, 34, 'b' * 64)
        cohort = PlannedCohort('adaptation_batch', (origin,), 24)
        values = (
            PlannedBatch(0, cohort),
            EvaluationScope('run', 0, 'candidate', 'revision', EvaluationPhase.SCREENING, 'c' * 64, 1),
            KnowledgeCard('knowledge', 'Title', 'Summary', 'https://example.org/evidence',
                          'catalog', 'curated', 'research_only', 'context'),
            FormalBatchComparisonDecision.CHAMPION_RETAINED, RevisionStatus.FINAL,
            CandidateDirection('next', 'Regularization', 'Test more shrinkage',
                'Short horizon error', 'Residual fit', 'scientific_parameter',
                'ridge_alpha', ('d' * 64,), 'May reduce long horizon gain',
                'Improve all scored cells', 'increase'),
        )
        self.assertEqual(_decode(json.loads(json.dumps(_encode(values)))), values)
        with self.assertRaises(KeyError):
            _decode({'type': 'record', 'class': 'untrusted.module.Record', 'fields': {}})
