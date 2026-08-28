from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ecologyrsi_dsh.api.handler import (
    _dsh_revision_snapshot,
    _ValidatedCandidateIdentityCache,
)
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.state import DSH_NATIVE_EVOLUTION_PROTOCOL


def _identity(character: str = "a") -> dict[str, str]:
    return {
        "execution_protocol": DSH_NATIVE_EVOLUTION_PROTOCOL,
        "genome_digest": character * 64,
        "behavior_digest": character * 64,
        "compiled_behavior_digest": character * 64,
        "phenotype_instance_digest": character * 64,
        "compiler_semantic_digest": character * 64,
        "registry_catalog_digest": character * 64,
        "security_semantic_digest": character * 64,
        "runtime_execution_digest": character * 64,
        "evaluation_cohort_digest": character * 64,
    }


def _state(
    run_id: str,
    seq: int,
    bindings: dict[str, dict[str, str]],
) -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(run_id=run_id),
        events=(SimpleNamespace(seq=seq),),
        candidate_identity_bindings=tuple(
            {
                "candidate_id": candidate_id,
                "identity_binding": binding,
            }
            for candidate_id, binding in bindings.items()
        ),
    )


class DshProviderHotPathTests(unittest.TestCase):
    def test_revision_snapshot_never_replays_the_run(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        ledger.append("run:target", "Example", {"value": 1})
        target_tail = ledger.append("run:target", "Example", {"value": 2})
        global_tail = ledger.append("run:other", "Example", {"value": 3})

        with patch.object(
            ledger,
            "events",
            side_effect=AssertionError("revision hot path must not replay"),
        ):
            for _ in range(256):
                revision = _dsh_revision_snapshot(ledger, "run:target")

        self.assertEqual(revision["run_state_revision"], target_tail.seq)
        self.assertEqual(revision["ledger_expected_revision"], global_tail.seq)

    def test_identity_cache_replays_once_for_concurrent_sample_workers(self) -> None:
        ledger = Mock(spec=EventLedger)
        ledger.latest_run_seq.return_value = 11
        state_provider = Mock(
            return_value=_state(
                "run:hot",
                11,
                {"candidate:a": _identity()},
            )
        )
        cache = _ValidatedCandidateIdentityCache(ledger, state_provider)

        with ThreadPoolExecutor(max_workers=64) as executor:
            results = tuple(
                executor.map(
                    lambda _index: cache.get("run:hot", "candidate:a"),
                    range(512),
                )
            )

        self.assertTrue(all(item == _identity() for item in results))
        self.assertEqual(state_provider.call_count, 1)
        self.assertEqual(ledger.latest_run_seq.call_count, 1)
        assert results[0] is not None
        results[0]["genome_digest"] = "f" * 64
        self.assertEqual(
            cache.get("run:hot", "candidate:a")["genome_digest"],
            "a" * 64,
        )

    def test_identity_cache_refreshes_for_a_later_generation_candidate(self) -> None:
        ledger = Mock(spec=EventLedger)
        ledger.latest_run_seq.side_effect = (10, 20)
        state_provider = Mock(
            side_effect=(
                _state("run:refresh", 10, {"candidate:a": _identity("a")}),
                _state(
                    "run:refresh",
                    20,
                    {
                        "candidate:a": _identity("a"),
                        "candidate:b": _identity("b"),
                    },
                ),
            )
        )
        cache = _ValidatedCandidateIdentityCache(ledger, state_provider)

        self.assertEqual(
            cache.get("run:refresh", "candidate:a")["genome_digest"],
            "a" * 64,
        )
        self.assertEqual(
            cache.get("run:refresh", "candidate:b")["genome_digest"],
            "b" * 64,
        )
        self.assertEqual(state_provider.call_count, 2)
        self.assertEqual(ledger.latest_run_seq.call_count, 2)

    def test_invalid_identity_is_not_cached_after_failed_validation(self) -> None:
        ledger = Mock(spec=EventLedger)
        ledger.latest_run_seq.return_value = 12
        invalid = _identity()
        invalid["genome_digest"] = "tampered"
        state_provider = Mock(
            side_effect=(
                _state("run:tamper", 12, {"candidate:a": invalid}),
                _state("run:tamper", 12, {"candidate:a": _identity()}),
            )
        )
        cache = _ValidatedCandidateIdentityCache(ledger, state_provider)

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            cache.get("run:tamper", "candidate:a")
        self.assertEqual(
            cache.get("run:tamper", "candidate:a")["genome_digest"],
            "a" * 64,
        )
        self.assertEqual(state_provider.call_count, 2)


if __name__ == "__main__":
    unittest.main()
