from __future__ import annotations

from dataclasses import dataclass
import unittest

from ecologyrsi_dsh.core.exposure_registry import (
    FormalExposureAlreadyUsed,
    ScientificExposureRegistry,
    raw_holdout_exposure_key,
)
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.data.contracts import DatasetSeries, SelectionDatasetView
from ecologyrsi_dsh.data.greenhouse import FeatureSpec
from ecologyrsi_dsh.data.splits import (
    EpisodeSplit,
    IndexRange,
    build_four_stage_data_protocol,
)


@dataclass(frozen=True)
class _Episode:
    episode_id: str = "dataset:TeamA"
    timestamps: tuple[int, ...] = tuple(range(1100))
    content_sha256: str = "a" * 64


def _legacy_split() -> EpisodeSplit:
    return EpisodeSplit(
        dataset_id="dataset",
        episode_id="dataset:TeamA",
        role="optimization",
        row_count=1100,
        training_fit=IndexRange(0, 650),
        training_feedback=IndexRange(674, 850),
        development=IndexRange(874, 950),
        gate=IndexRange(974, 1100),
        content_sha256="a" * 64,
    )


def _protocol():
    return build_four_stage_data_protocol(
        "dataset",
        _Episode(),
        _legacy_split(),
        dataset_digest="b" * 64,
        split_manifest_digest="c" * 64,
        target_names=("air_temperature", "relative_humidity", "co2_concentration"),
        horizons=(1, 6, 24),
        history_steps=3,
    )


def _series(formal_offset: float = 0.0) -> DatasetSeries:
    timestamps = tuple(range(1100))
    values = tuple(float(index) for index in timestamps)
    changed = values[:874] + tuple(item + formal_offset for item in values[874:])
    return DatasetSeries(
        schema="ecologyrsi-dsh.dataset-series/1",
        dataset_id="dataset",
        domain_id="greenhouse",
        episode_id="dataset:TeamA",
        digest="a" * 64,
        timestamps=timestamps,
        values={"air_temperature": changed},
        partitions={
            "training_fit": IndexRange(0, 650),
            "training_feedback": IndexRange(674, 850),
            "development": IndexRange(874, 950),
        },
        features={
            "air_temperature": FeatureSpec(
                "air_temperature", "air", "environment", "degC"
            )
        },
        split_manifest_digest_sha256="c" * 64,
    )


class FourStageDataProtocolTests(unittest.TestCase):
    def test_exact_seventy_percent_cut_and_both_twenty_four_hour_embargoes(self) -> None:
        protocol = _protocol()
        self.assertEqual(protocol.source_timezone, "unspecified-naive-local")
        self.assertEqual(protocol.calendar_encoding, "excel-serial-hour-fixed-24h@1")
        self.assertEqual(protocol.calibration_fit, IndexRange(0, 455))
        self.assertEqual(protocol.fit_to_uq_embargo, IndexRange(455, 478))
        self.assertEqual(protocol.calibration_uq, IndexRange(478, 650))
        self.assertEqual(protocol.uq_to_selection_embargo, IndexRange(650, 674))
        self.assertEqual(protocol.model_selection, IndexRange(674, 850))
        self.assertEqual(protocol.validation, IndexRange(874, 950))
        self.assertEqual(protocol.final_test, IndexRange(974, 1100))
        self.assertEqual(set(protocol.partition_digests), {
            "calibration_fit", "fit_to_uq_embargo", "calibration_uq",
            "uq_to_selection_embargo", "model_selection", "validation", "final_test",
        })

    def test_target_timestamp_membership_is_half_open(self) -> None:
        protocol = _protocol()
        self.assertTrue(protocol.contains_target("calibration_fit", 454))
        self.assertFalse(protocol.contains_target("calibration_fit", 455))
        self.assertTrue(protocol.contains_target("calibration_uq", 478))
        self.assertFalse(protocol.contains_target("calibration_uq", 650))

    def test_target_membership_uses_timestamps_not_row_indices(self) -> None:
        episode = _Episode(timestamps=tuple(range(10_000, 11_100)))
        protocol = build_four_stage_data_protocol(
            "dataset",
            episode,
            _legacy_split(),
            dataset_digest="b" * 64,
            split_manifest_digest="c" * 64,
            target_names=("air_temperature", "relative_humidity", "co2_concentration"),
            horizons=(1, 6, 24),
            history_steps=3,
        )
        self.assertTrue(protocol.contains_target("calibration_fit", 10_454))
        self.assertFalse(protocol.contains_target("calibration_fit", 10_455))
        self.assertIn("objective_grid", protocol.identity_dict())
        self.assertEqual(protocol.objective_grid["horizons"], [1, 6, 24])

    def test_selection_view_structurally_excludes_formal_rows(self) -> None:
        view = SelectionDatasetView.from_series(_series(), _protocol())
        self.assertNotIn("development", view.partitions)
        self.assertNotIn("gate", view.partitions)
        self.assertEqual(len(view.timestamps), 850)
        self.assertEqual(max(view.partitions[name].end for name in view.partitions), 850)

    def test_changing_formal_values_cannot_change_selection_view_digest(self) -> None:
        first = SelectionDatasetView.from_series(_series(0.0), _protocol())
        changed = SelectionDatasetView.from_series(_series(999.0), _protocol())
        self.assertEqual(first.selection_view_digest, changed.selection_view_digest)
        self.assertEqual(first.values, changed.values)

    def test_short_protocol_fails_closed_without_moving_cut(self) -> None:
        short = _Episode(timestamps=tuple(range(220)))
        split = EpisodeSplit(
            dataset_id="dataset", episode_id=short.episode_id, role="optimization",
            row_count=220, training_fit=IndexRange(0, 120),
            training_feedback=IndexRange(144, 170),
            development=IndexRange(194, 200), gate=IndexRange(210, 220),
            content_sha256=short.content_sha256,
        )
        with self.assertRaisesRegex(ValueError, "insufficient|calibration"):
            build_four_stage_data_protocol(
                "dataset", short, split,
                dataset_digest="b" * 64,
                split_manifest_digest="c" * 64,
                target_names=("air_temperature",), horizons=(1, 6, 24),
                history_steps=3,
            )


class FormalExposureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.registry = ScientificExposureRegistry(self.ledger)
        self.raw_key = raw_holdout_exposure_key(
            dataset_digest="a" * 64,
            split_manifest_digest="b" * 64,
            episode_id="dataset:TeamA",
            stage="validation",
            stage_partition_digest="c" * 64,
        )

    def tearDown(self) -> None:
        self.ledger.close()

    def _reserve(self, **overrides):
        values = {
            "raw_holdout_key": self.raw_key,
            "objective_family_digest": "d" * 64,
            "plan_digest": "e" * 64,
            "idempotency_key": "formal:one",
            "run_id": "run:one",
            "stage": "validation",
            "candidate_id": "candidate:one",
            "artifact_digest": "f" * 64,
            "genome_digest": "1" * 64,
            "partition_digest": "c" * 64,
        }
        values.update(overrides)
        return self.registry.reserve_formal_stage(**values)

    def test_raw_holdout_key_excludes_objective_family(self) -> None:
        self.assertEqual(
            self.raw_key,
            raw_holdout_exposure_key(
                dataset_digest="a" * 64,
                split_manifest_digest="b" * 64,
                episode_id="dataset:TeamA",
                stage="validation",
                stage_partition_digest="c" * 64,
            ),
        )

    def test_look_is_reserved_before_any_current_cohort_metric_is_read(self) -> None:
        token = self._reserve()
        observed = self.registry.with_formal_stage(
            token,
            lambda: self.registry.formal_exposure(self.raw_key)["state"],
        )
        self.assertEqual(observed, "opened")
        self.registry.seal_formal_stage(token, outcome="passed")
        self.assertEqual(self.registry.formal_exposure(self.raw_key)["state"], "sealed")

    def test_stage_token_is_single_use_and_artifact_run_scoped(self) -> None:
        token = self._reserve()
        with self.assertRaisesRegex(ValueError, "binding"):
            self.registry.open_formal_stage(token, artifact_digest="0" * 64)
        self.registry.open_formal_stage(token)
        with self.assertRaisesRegex(ValueError, "opened|single-use"):
            self.registry.open_formal_stage(token)

    def test_objective_change_cannot_reopen_same_development_partition(self) -> None:
        self._reserve()
        with self.assertRaises(FormalExposureAlreadyUsed):
            self._reserve(
                objective_family_digest="9" * 64,
                plan_digest="8" * 64,
                idempotency_key="formal:changed-objective",
                run_id="run:two",
            )

    def test_fitness_or_statistical_plan_change_cannot_reopen_same_gate_partition(self) -> None:
        self._reserve()
        with self.assertRaises(FormalExposureAlreadyUsed):
            self._reserve(
                plan_digest=digest({"different_fitness": True}),
                idempotency_key="formal:changed-plan",
            )


if __name__ == "__main__":
    unittest.main()
