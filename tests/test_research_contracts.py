import unittest
from dataclasses import FrozenInstanceError
from ecologyrsi_dsh.core.research import DiagnosticReport, HypothesisProposal


class ResearchContractTests(unittest.TestCase):
    def test_diagnostic_report_is_immutable_and_canonical(self):
        report = DiagnosticReport(
            task_id="task-1", program_id="program-1", comparison_id=None,
            weak_cells=("temperature@24h",), failure_patterns=("regime_change",),
            evidence_refs=("eval-1",), possible_causes=("lagged response",),
            unresolved_questions=("does feature availability permit it?",),
        )
        with self.assertRaises(FrozenInstanceError):
            report.task_id = "other"
        self.assertEqual(report.report_id, report.report_id)
        self.assertEqual(report.to_dict()["weak_cells"], ["temperature@24h"])

    def test_hypothesis_requires_evidence_and_applicability(self):
        with self.assertRaises(ValueError):
            HypothesisProposal(
                "p", "parent", "problem", "hypothesis", (), "feature", "change-1",
                "lower error", "error unchanged", (),
            )
