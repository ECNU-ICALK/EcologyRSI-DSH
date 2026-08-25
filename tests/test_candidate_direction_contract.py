from __future__ import annotations

import unittest

from ecologyrsi_dsh.knowledge.autonomous_cycle import CandidateDirection


def _direction_payload() -> dict[str, object]:
    return {
        "direction_id": "history-increase",
        "title": "Increase bounded history",
        "hypothesis": "A longer history may improve temporal context.",
        "target_weakness": "long-horizon skill",
        "capability_focus": "registered greenhouse predictor",
        "mutation_axis": "scientific_parameter",
        "mutation_target": "history_steps",
        "mutation_direction": "increase",
        "evidence_refs": [],
        "expected_tradeoff": "May increase fitting cost.",
        "success_criterion": "Improve the frozen diagnostic score.",
    }


class CandidateDirectionContractTests(unittest.TestCase):
    def test_mutation_direction_is_required(self) -> None:
        payload = _direction_payload()
        del payload["mutation_direction"]

        with self.assertRaisesRegex(
            ValueError, "candidate direction fields do not match the contract"
        ):
            CandidateDirection.from_dict(payload)

    def test_mutation_direction_cannot_be_null(self) -> None:
        payload = _direction_payload()
        payload["mutation_direction"] = None

        with self.assertRaisesRegex(ValueError, "mutation_direction"):
            CandidateDirection.from_dict(payload)

    def test_mutation_direction_is_digest_bound(self) -> None:
        increasing = CandidateDirection.from_dict(_direction_payload())
        decreasing_payload = _direction_payload()
        decreasing_payload["mutation_direction"] = "decrease"
        decreasing = CandidateDirection.from_dict(decreasing_payload)

        self.assertIn("mutation_direction", increasing.to_dict())
        self.assertNotEqual(increasing.direction_digest, decreasing.direction_digest)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
