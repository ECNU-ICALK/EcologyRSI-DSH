"""Usage identity checks remain complete without decoding unrelated children."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.dsh_usage import check_usage_binding
from ecologyrsi_dsh.integrations.dsh_tools import DshToolService
from tests.test_final_delivery import usage


class ScopedUsageTests(TestCase):
    def launch(self, ledger, reservation, run='r'):
        return ledger.append(run, 'DshChildLaunchReserved', {
            'launch': {'reservation_id': reservation, 'run_id': run,
                       'stage': 'generation.research', 'idempotency_key': 'key'}})

    def body(self, reservation='reservation', session='session', total=10):
        result = usage(total)
        result['identity'].update(child_reservation_id=reservation, session_id=session,
                                  stage='generation.research', idempotency_key='key')
        result['session_metrics']['session_id'] = session
        return result

    def test_query_keeps_both_identity_axes_and_decodes_only_matches(self):
        with EventLedger() as ledger:
            own = self.launch(ledger, 'reservation')
            unrelated = self.launch(ledger, 'other')
            previous = ledger.append('r', 'DshSessionUsageRecorded', self.body(total=5))
            accepted = ledger.append('r', 'DshStructuredResultAccepted', {
                'identity': {'child_reservation_id': 'other', 'session_id': 'session'}})
            ledger.append('another-run', 'DshSessionUsageRecorded', self.body(total=99))
            for index in range(100):
                ledger.append('r', 'DshSessionUsageRecorded', self.body(f'r{index}', f's{index}'))
            with patch.object(ledger, '_row_to_event', wraps=ledger._row_to_event) as decode:
                events = ledger.dsh_usage_binding_events('r', 'reservation', 'session')
            self.assertEqual([e.seq for e in events], [own.seq, previous.seq, accepted.seq])
            self.assertEqual(decode.call_count, 3)
            with self.assertRaisesRegex(ValueError, 'another reservation'):
                check_usage_binding(self.body(), events)
            self.assertNotIn(unrelated, events)

    def test_concurrent_reservations_cannot_claim_one_session(self):
        with EventLedger() as ledger:
            self.launch(ledger, 'one'); self.launch(ledger, 'two')
            service = DshToolService(ledger)
            def record(reservation):
                try:
                    return service.record_session_usage(self.body(reservation))['accepted']
                except ValueError:
                    return False
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(sorted(pool.map(record, ('one', 'two'))), [False, True])
            self.assertEqual(len(ledger.events_by_kind('r', 'DshSessionUsageRecorded')), 1)

    def test_all_prior_cumulative_constraints_survive_scoping(self):
        with EventLedger() as ledger:
            self.launch(ledger, 'reservation')
            service = DshToolService(ledger)
            first = self.body(total=10)
            self.assertTrue(service.record_session_usage(first)['accepted'])
            lower = self.body(total=9)
            with self.assertRaisesRegex(ValueError, 'cannot decrease'):
                service.record_session_usage(lower)
            changed = self.body(session='changed')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                service.record_session_usage(changed)
            complete = self.body(total=20)
            complete.update(settlement='succeeded', usage_complete=True)
            service.record_session_usage(complete)
            reopened = self.body(total=30)
            with self.assertRaisesRegex(ValueError, 'outcome'):
                service.record_session_usage(reopened)
            self.assertTrue(service.record_session_usage(complete)['already_recorded'])
