"""Frozen current schedule, budget reservations and strict deserialization."""
import unittest
from dataclasses import replace
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule

class OptimizationScheduleTests(unittest.TestCase):
    def test_current_budget_separates_base_conditional_and_unique_origins(self):
        schedule=OptimizationSchedule.for_new_run()
        self.assertEqual(OptimizationSchedule.from_dict(schedule.to_dict()), schedule)
        self.assertEqual([schedule.batch_origin_count(i) for i in (0,1)], [10,25])
        self.assertEqual(schedule.max_local_edit_decisions_per_finalist,1)
        self.assertEqual(schedule.max_local_edits_per_finalist,2)
        plan=schedule.execution_plan(2,cells_per_origin=9)
        self.assertEqual(plan['required_unique_origins'],170)
        self.assertEqual(plan['run_budget']['base_candidate_origins'],400)
        self.assertEqual(plan['run_budget']['recheck_candidate_origins'],200)
        self.assertEqual(plan['run_budget']['total_candidate_origins'],600)
        self.assertEqual(plan['run_budget']['total_scoring_cells'],5400)
        self.assertTrue(plan['budget_includes_recheck_reserve'])
        self.assertFalse(plan['training_replay'])
    def test_every_frozen_field_rejects_missing_extra_or_changed_values(self):
        current=OptimizationSchedule.for_new_run().to_dict()
        for key,value in current.items():
            for changed in (None,True,0,'legacy@1'):
                if changed==value and type(changed)==type(value):continue
                with self.subTest(key=key,value=changed),self.assertRaises(ValueError):
                    OptimizationSchedule.from_dict({**current,key:changed})
            with self.subTest(missing=key),self.assertRaises(ValueError):
                OptimizationSchedule.from_dict({k:v for k,v in current.items() if k!=key})
        with self.assertRaises(ValueError):OptimizationSchedule.from_dict({**current,'samples_per_update':9})
    def test_budget_and_batch_boundaries_reject_bool_or_invalid_values(self):
        schedule=OptimizationSchedule.for_new_run()
        for value in (True,0,-1,None,1.5):
            with self.subTest(value=value),self.assertRaises(ValueError):schedule.run_execution_budget(value,cells_per_origin=9)
            with self.subTest(value=value),self.assertRaises(ValueError):schedule.generation_execution_budget(cells_per_origin=value)
        for index in (True,-1,2,1.5):
            with self.assertRaises(ValueError):schedule.batch_origin_count(index)
        with self.assertRaises(ValueError):schedule.execution_plan(1,cells_per_origin=9,native=False)
