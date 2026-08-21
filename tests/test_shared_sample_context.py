from __future__ import annotations

import unittest

from ecologyrsi_dsh.core.models import canonical_json, digest
from ecologyrsi_dsh.evaluators.sample_execution import SamplePredictionRequest
from ecologyrsi_dsh.evaluators.shared_sample_context import (
    ORIGIN_SHARED_CONTEXT_PROFILE,
    build_origin_shared_routing_payload,
    expand_origin_shared_routing_payload,
    normalized_sample_planner_prompt_profile,
    sibling_stage_context_digest,
)


class SharedSampleContextTests(unittest.TestCase):
    def test_origin_wave_encoding_is_lossless_and_omits_host_identifiers(self) -> None:
        requests = _origin_wave()
        samples, shared_contexts = build_origin_shared_routing_payload(
            requests,
            (1,) * len(requests),
            ((),) * len(requests),
        )

        self.assertEqual(len(shared_contexts), 1)
        self.assertTrue(
            all(
                set(sample)
                == {
                    "sample_id",
                    "context_ref",
                    "horizon_hours",
                    "target_timestamp",
                    "attempt",
                    "failure_feedback",
                }
                for sample in samples
            )
        )
        expanded = expand_origin_shared_routing_payload(samples, shared_contexts)
        for request in requests:
            reconstructed = expanded[request.sample_id]
            expected = request.to_dict()
            for field in (
                "sample_id",
                "target",
                "unit",
                "horizon_hours",
                "origin_timestamp",
                "target_timestamp",
                "baseline",
                "minimum",
                "maximum",
                "label_free_context",
            ):
                self.assertEqual(reconstructed[field], expected[field])

        encoded = canonical_json(
            {"samples": samples, "shared_sample_contexts": shared_contexts}
        )
        self.assertNotIn("candidate:private", encoded)
        self.assertNotIn("dataset:private", encoded)
        self.assertNotIn("partition:private", encoded)

    def test_origin_wave_encoding_reduces_repeated_payload_bytes(self) -> None:
        requests = _origin_wave()
        legacy = [
            {
                "sample_id": request.sample_id,
                "sample": request.to_dict(),
                "attempt": 1,
                "failure_feedback": [],
            }
            for request in requests
        ]
        samples, shared_contexts = build_origin_shared_routing_payload(
            requests,
            (1,) * len(requests),
            ((),) * len(requests),
        )
        legacy_bytes = len(canonical_json({"samples": legacy}).encode("utf-8"))
        compact_bytes = len(
            canonical_json(
                {
                    "samples": samples,
                    "shared_sample_contexts": shared_contexts,
                }
            ).encode("utf-8")
        )

        self.assertLess(compact_bytes, legacy_bytes * 0.70)

    def test_context_digest_never_reuses_a_different_origin(self) -> None:
        first = _origin_wave()[0]
        second = _request(
            target=first.target,
            horizon=first.horizon_hours,
            origin=200,
            ordinal=99,
        )
        first_samples, first_contexts = build_origin_shared_routing_payload(
            (first,), (1,), ((),)
        )
        second_samples, second_contexts = build_origin_shared_routing_payload(
            (second,), (1,), ((),)
        )

        self.assertNotEqual(
            first_samples[0]["context_ref"], second_samples[0]["context_ref"]
        )
        self.assertNotEqual(set(first_contexts), set(second_contexts))

    def test_expansion_rejects_tampered_context_and_variant_digests(self) -> None:
        samples, shared_contexts = build_origin_shared_routing_payload(
            _origin_wave(), (1,) * 9, ((),) * 9
        )
        context_ref = samples[0]["context_ref"]
        tampered_context = dict(shared_contexts[context_ref])
        tampered_context["sample_count"] = 10
        with self.assertRaisesRegex(ValueError, "context digest"):
            expand_origin_shared_routing_payload(
                samples, {context_ref: tampered_context}
            )

        tampered_variant_context = dict(shared_contexts[context_ref])
        tampered_variant_context["sample_variants"] = {
            key: dict(value)
            for key, value in tampered_variant_context["sample_variants"].items()
        }
        variant_ref = next(iter(tampered_variant_context["sample_variants"]))
        tampered_variant_context["sample_variants"][variant_ref]["target"] = (
            "tampered-target"
        )
        new_context_ref = digest(tampered_variant_context)
        rebound_samples = [
            {**sample, "context_ref": new_context_ref} for sample in samples
        ]
        with self.assertRaisesRegex(ValueError, "variant digest"):
            expand_origin_shared_routing_payload(
                rebound_samples, {new_context_ref: tampered_variant_context}
            )

    def test_profile_validation_is_exact(self) -> None:
        self.assertEqual(
            normalized_sample_planner_prompt_profile(
                {"version": ORIGIN_SHARED_CONTEXT_PROFILE}
            ),
            {"version": ORIGIN_SHARED_CONTEXT_PROFILE},
        )
        with self.assertRaises(ValueError):
            normalized_sample_planner_prompt_profile(
                {"version": ORIGIN_SHARED_CONTEXT_PROFILE, "extra": True}
            )
        with self.assertRaises(ValueError):
            normalized_sample_planner_prompt_profile({"version": "future@2"})

    def test_sibling_stage_digest_excludes_candidate_genome(self) -> None:
        contracts = {
            "data_protocol_digest": "a" * 64,
            "evaluation_cohort_digest": "b" * 64,
            "fitness_profile_digest": "c" * 64,
        }
        first = sibling_stage_context_digest(
            task_manifest_digest="d" * 64,
            generation=2,
            frozen_contract_digests=contracts,
        )
        second = sibling_stage_context_digest(
            task_manifest_digest="d" * 64,
            generation=2,
            frozen_contract_digests=contracts,
        )
        self.assertEqual(first, second)
        self.assertNotEqual(
            digest({"stage_context_digest": first, "genome_digest": "1" * 64}),
            digest({"stage_context_digest": second, "genome_digest": "2" * 64}),
        )


def _origin_wave() -> tuple[SamplePredictionRequest, ...]:
    return tuple(
        _request(target=target, horizon=horizon, origin=100, ordinal=ordinal)
        for ordinal, (target, horizon) in enumerate(
            (target, horizon)
            for target in ("air_temperature", "relative_humidity", "co2_concentration")
            for horizon in (1, 6, 24)
        )
    )


def _request(
    *, target: str, horizon: int, origin: int, ordinal: int
) -> SamplePredictionRequest:
    target_index = {
        "air_temperature": 1,
        "relative_humidity": 2,
        "co2_concentration": 3,
    }[target]
    feature_snapshot = [
        {"name": f"feature:{index}", "value": float(index + target_index)}
        for index in range(32)
    ]
    return SamplePredictionRequest(
        sample_id=f"sample:{origin}:{ordinal}",
        candidate_id="candidate:private",
        dataset_digest="dataset:private",
        partition="partition:private",
        target=target,
        unit="unit",
        horizon_hours=horizon,
        origin_timestamp=origin,
        target_timestamp=origin + horizon,
        baseline=float(target_index),
        proposed_prediction=None,
        minimum=-10.0,
        maximum=5000.0,
        algorithm_id="registered-algorithm",
        algorithm_version="1",
        label_free_context={
            "schema_version": "ecologyrsi-dsh.label-free-sample-context/1",
            "history_window": [float(target_index + index) for index in range(6)],
            "feature_snapshot": feature_snapshot,
            "causal_provenance": {
                "schema_version": "ecologyrsi-dsh.causal-sample-provenance/1",
                "origin_cutoff_timestamp": origin,
                "latest_context_timestamp": origin,
                "history_timestamps": [origin - index for index in range(6)],
            },
            "predictor_state": {
                "predictor_id": "greenhouse-exogenous-ridge@1",
                "history_steps": 6,
                "ridge_alpha": 0.1,
                "residual_scale": 0.5,
            },
        },
    )


if __name__ == "__main__":
    unittest.main()
