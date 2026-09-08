"""Behavior tests for the migrated application, scientific and query boundaries."""
import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from ecologyrsi_dsh import EventLedger, EvolutionDirector
from ecologyrsi_dsh.application.cli import main, _demo_manifest
from ecologyrsi_dsh.application.queries import RunQueries
from ecologyrsi_dsh.application.runtime import ApplicationRuntime
from ecologyrsi_dsh.data.registry import DatasetRegistry
from ecologyrsi_dsh.core.state import project_run_state
from ecologyrsi_dsh.science.context import EvaluationContext
from ecologyrsi_dsh.core.identity import FrozenObject
from tests.test_core import manifest


class QueryBoundaryTests(unittest.TestCase):
    def test_incremental_tail_and_full_replay_match(self):
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger)
            director.create_run(manifest(1), run_id='query')
            first = director.state('query'); seq = first.events[-1].seq
            with patch.object(ledger, 'events', wraps=ledger.events) as events:
                director.start_run('query')
                updated = director.state('query')
            self.assertTrue(events.call_args_list)
            self.assertTrue(all(call.kwargs.get('after_seq') == seq for call in events.call_args_list))
            self.assertEqual(updated, project_run_state(ledger.events('query')))
            self.assertEqual(first.run.status.value, 'created')

    def test_keyset_page_does_not_duplicate_when_new_run_arrives(self):
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger)
            for i in range(6): director.create_run(manifest(1), run_id=f'run-{i}')
            queries = RunQueries(director)
            first = queries.page(limit=2)
            director.create_run(manifest(1), run_id='new-arrival')
            second = queries.page(limit=2, before=first.next_cursor)
            third = queries.page(limit=2, before=second.next_cursor)
            self.assertEqual([s.run.run_id for page in (first, second, third) for s in page.states], ['run-5','run-4','run-3','run-2','run-1','run-0'])
            self.assertIsNone(third.next_cursor)
            with self.assertRaises(ValueError): queries.page(limit=201)
            with self.assertRaises(ValueError): queries.page(before=0)

    def test_cli_advances_using_application_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'main.sqlite'
            with EventLedger(path) as ledger:
                director=ApplicationRuntime(ledger).director
                task = _demo_manifest(SimpleNamespace(candidates=1, seed=7))
                series = DatasetRegistry().series('generated-toy-series@1')
                task = replace(task, metadata={**task.metadata, 'split_manifest_digest': series.split_manifest_digest_sha256})
                director.start_evolution(task, run_id='cli-advance')
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(['advance','--db',str(path),'--run-id','cli-advance']),0)
            with EventLedger(path) as ledger:
                self.assertTrue(any(event.kind=='EvaluationRecorded' for event in ledger.events('cli-advance')))


class MeasurementContextTests(unittest.TestCase):
    def test_grid_is_immutable_and_checks_time_and_coverage(self):
        context = EvaluationContext(FrozenObject.from_mapping({'metric':'native_bounded_rmse_skill'}), (('temperature',1,100,101),))
        valid={'target':'temperature','horizon_hours':1,'origin_timestamp':100,'timestamp':101}
        self.assertEqual(context.validate_rows([valid]),0)
        with self.assertRaisesRegex(ValueError,'duplicate'): context.validate_rows([valid,valid])
        with self.assertRaisesRegex(ValueError,'incomplete'): context.validate_rows([])
        self.assertEqual(context.validate_rows([],allow_missing=True),1)
        with self.assertRaisesRegex(ValueError,'identity'): context.validate_rows([{**valid,'timestamp':102}])
        changed=replace(context,contract=FrozenObject.from_mapping({'metric':'normalized_MAE'}))
        self.assertNotEqual(changed.context_id,context.context_id)
