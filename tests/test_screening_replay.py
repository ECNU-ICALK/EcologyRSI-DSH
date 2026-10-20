"""Current five-arm screening identities, selection and tamper rejection."""
from dataclasses import replace
import unittest
from ecologyrsi_dsh.application import generation_execution
from ecologyrsi_dsh.core.models import thaw_json
from ecologyrsi_dsh.core.screening import screening_record_digest, screening_cohort_digest
from ecologyrsi_dsh.core.state import project_run_state
from tests import test_evidence_guided_epoch as race

class ScreeningReplayTests(unittest.TestCase):
    def setUp(self):
        race.EvidenceGuidedRaceTests.setUp(self)
        for candidate in (*self.candidates, self.control):
            race.EvidenceGuidedRaceTests._screen(self, candidate)
        generation_execution._prepare_formal_finalists(self.services,self.run_id,self.candidates,max_concurrency=1)
        state = self.director.state(self.run_id)
        self.events = state.events
        self.screening_events = state.candidate_screening_events
        self.formal_event = state.formal_selection_for(0)
        self.selected_ids = tuple(self.formal_event.payload['selected_candidate_ids'])

    def changed(self, event, payload):
        return tuple(replace(e,payload=payload) if e.event_id==event.event_id else e for e in self.events)

    def test_current_five_arm_replay_has_one_finalist_and_three_screened_out(self):
        state = project_run_state(self.events)
        self.assertEqual(len(state.candidate_screening_events),5)
        self.assertEqual(len(state.screened_out_events),3)
        self.assertEqual(len(self.selected_ids),1)
        self.assertFalse(state.screening_for(0,self.selected_ids[0]).payload['passed'])

    def test_digest_is_order_independent_and_ignores_display_fields(self):
        rows=[thaw_json(e.payload) for e in self.screening_events]
        self.assertEqual(screening_cohort_digest(rows),screening_cohort_digest(rows[::-1]))
        self.assertEqual(screening_record_digest(rows[0]),screening_record_digest({**rows[0],'display':'ignored'}))

    def test_replay_rejects_forged_generation_revision_cohort_and_count(self):
        event=self.screening_events[0]
        for change in ({'generation':1},{'candidate_revision_id':'revision:other'},
                       {'cohort_digest':'a'*64},{'origin_count':9},{'prediction_cell_count':89}):
            payload={**thaw_json(event.payload),**change}
            payload['record_digest']=screening_record_digest(payload)
            with self.subTest(change=change), self.assertRaises(ValueError):
                project_run_state(self.changed(event,payload))

    def test_replay_rejects_candidate_ownership_forgery(self):
        event=next(e for e in self.events if e.kind=='CandidateSpawned')
        payload=thaw_json(event.payload);payload['candidate']['run_id']='run:foreign'
        with self.assertRaisesRegex(ValueError,'ownership'):
            project_run_state(self.changed(event,payload))

    def test_formal_selection_requires_every_frozen_arm(self):
        for event in self.screening_events:
            with self.subTest(candidate=event.payload['candidate_id']), self.assertRaises(ValueError):
                project_run_state(tuple(e for e in self.events if e.event_id!=event.event_id))

    def test_selected_candidate_cannot_be_screened_out(self):
        event=next(e for e in self.events if e.kind=='CandidateScreenedOut')
        with self.assertRaises(ValueError):
            project_run_state(self.changed(event,{**thaw_json(event.payload),'candidate_id':self.selected_ids[0]}))

    def test_duplicate_screening_payload_cannot_change_score(self):
        event=self.screening_events[0];payload={**thaw_json(event.payload),'score':.5}
        payload['record_digest']=screening_record_digest(payload)
        conflicting=replace(event,seq=self.events[-1].seq+1,event_id='forged',payload=payload)
        with self.assertRaises(ValueError):project_run_state((*self.events,conflicting))

    def test_exact_screening_and_selection_retry_preserves_first_event(self):
        before=self.ledger.count()
        for event in self.screening_events:
            payload=thaw_json(event.payload)
            args={k:payload[k] for k in ('candidate_id','candidate_revision_id','generation','score',
                'passed','constraint_violations','origin_count','prediction_cell_count','cohort_digest','metrics')}
            self.assertEqual(self.director.record_candidate_screening(self.run_id,**args),event)
            with self.assertRaises(ValueError):
                self.director.record_candidate_screening(self.run_id,**{**args,'score':.5})
        self.assertEqual(self.director.freeze_formal_selection_cohort(self.run_id,generation=0,
            selected_candidate_ids=self.selected_ids,screening_digest=self.formal_event.payload['screening_digest']),self.formal_event)
        self.assertEqual(self.ledger.count(),before)

    def test_selection_cannot_switch_to_another_revision_after_freeze(self):
        before=self.ledger.count()
        with self.assertRaises(ValueError):
            self.director.freeze_formal_selection_cohort(self.run_id,generation=0,
                selected_candidate_ids=[self.candidates[1].candidate_id],
                screening_digest=self.formal_event.payload['screening_digest'])
        self.assertEqual(self.ledger.count(),before)
