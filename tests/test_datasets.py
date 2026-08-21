from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from ecologyrsi_dsh.cli import build_parser
from ecologyrsi_dsh.datasets import DatasetRegistry, DatasetSeries
from ecologyrsi_dsh.greenhouse import GreenhouseDatasetAdapter
from ecologyrsi_dsh.splits import build_split_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "datasets" / "autonomous_greenhouse.json"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _touch_required_files(dataset_dir: Path, patterns: tuple[str, ...], team: str) -> None:
    for pattern in patterns:
        relative = pattern.replace("*/", f"{team}/").replace("**/", "nested/")
        target = dataset_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()


def _bind_fixture_zip(
    registry: DatasetRegistry,
    data_root: Path,
    members: list[tuple[str | zipfile.ZipInfo, bytes]],
    *,
    required_globs: tuple[str, ...] = ("TeamA/Greenhouse_climate.csv",),
) -> Path:
    dataset_id = "agc_cucumber_2018"
    archive = data_root / dataset_id / "_archives" / "fixture.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in members:
            bundle.writestr(name, content)
    descriptor = replace(
        registry._descriptors[dataset_id],
        required_globs=required_globs,
        source_files=(
            {
                "name": archive.name,
                "download_url": "https://example.invalid/fixture.zip",
                "size_bytes": archive.stat().st_size,
                "md5": hashlib.md5(archive.read_bytes()).hexdigest(),
                "archive_format": "zip",
            },
        ),
    )
    registry._descriptors[dataset_id] = descriptor
    return archive


@dataclass(frozen=True)
class _Episode:
    episode_id: str
    timestamps: tuple[int, ...]
    content_sha256: str


class SplitManifestTests(unittest.TestCase):
    def test_exact_time_forward_ranges_and_public_gate_redaction(self) -> None:
        episode = _Episode("team-a", tuple(range(100)), "a" * 64)
        manifest = build_split_manifest("dataset-a", (episode,))
        split = manifest.split_for("team-a")

        self.assertEqual((split.training_fit.start, split.training_fit.end), (0, 30))
        self.assertEqual((split.training_feedback.start, split.training_feedback.end), (31, 60))
        self.assertEqual((split.development.start, split.development.end), (80, 80))
        self.assertEqual((split.gate.start, split.gate.end), (100, 100))
        public_record = manifest.to_dict()["episode_records"][0]
        self.assertNotIn("gate", public_record)
        self.assertIn("gate", manifest.to_dict(include_restricted=True)["episode_records"][0])

    def test_reference_episode_is_external_holdout(self) -> None:
        episodes = (
            _Episode("team-a", tuple(range(100)), "a" * 64),
            _Episode("Reference", tuple(range(100)), "b" * 64),
        )
        manifest = build_split_manifest("dataset-a", episodes)
        self.assertEqual(manifest.split_for("team-a").role, "optimization")
        self.assertEqual(manifest.split_for("Reference").role, "external_holdout")


class GreenhouseAdapterTests(unittest.TestCase):
    def test_cucumber_loader_aggregates_to_hourly_values(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_csv(
                root / "TeamA" / "Greenhouse_climate.csv",
                ["GHtime", "Tair", "RHair", "CO2air"],
                [
                    {"GHtime": "1.0000", "Tair": 20, "RHair": 70, "CO2air": 700},
                    {"GHtime": "1.0208", "Tair": 22, "RHair": 72, "CO2air": 740},
                    {"GHtime": "1.0417", "Tair": 24, "RHair": 74, "CO2air": 760},
                ],
            )
            adapter = GreenhouseDatasetAdapter("cucumber", "greenhouse_cucumber_2018", root)
            episode = adapter.load().episodes[0]

        self.assertEqual(episode.episode_id, "cucumber:TeamA")
        self.assertEqual(episode.timestamps, (24, 25))
        self.assertEqual(episode.values["air_temperature"], (21.0, 24.0))
        self.assertEqual(episode.features["air_temperature"].display_name_zh, "室内气温")

    def test_tomato_loader_derives_electricity_and_net_water(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            team = root / "TeamA"
            _write_csv(
                team / "GreenhouseClimate.csv",
                ["%time", "Tair", "RHair", "CO2air"],
                [{"%time": "1.0", "Tair": 21, "RHair": 68, "CO2air": 720}],
            )
            _write_csv(
                team / "Resources.csv",
                ["%time", "ElecHigh", "ElecLow", "Irr", "Drain"],
                [{"%time": "1.0", "ElecHigh": 2.5, "ElecLow": 1.5, "Irr": 8, "Drain": 3}],
            )
            adapter = GreenhouseDatasetAdapter("tomato", "greenhouse_tomato_2019", root)
            episode = adapter.load().episodes[0]

        self.assertEqual(episode.values["electricity_use"], (4.0,))
        self.assertEqual(episode.values["water_use"], (5.0,))
        self.assertEqual(episode.features["water_use"].display_name_zh, "净用水量")

    def test_missing_required_feature_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_csv(
                root / "TeamA" / "Greenhouse_climate.csv",
                ["GHtime", "Tair", "RHair"],
                [{"GHtime": "1.0", "Tair": 20, "RHair": 70}],
            )
            adapter = GreenhouseDatasetAdapter("cucumber", "greenhouse_cucumber_2018", root)
            with self.assertRaisesRegex(ValueError, "co2_concentration"):
                adapter.load()


class DatasetRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = DatasetRegistry(catalog_path=CATALOG_PATH, data_root=PROJECT_ROOT / "missing-test-data")

    def test_catalog_has_chinese_names_and_readiness(self) -> None:
        catalog = self.registry.catalog()
        by_id = {item["dataset_id"]: item for item in catalog["datasets"]}
        self.assertIn("agc_cucumber_2018", by_id)
        self.assertEqual(by_id["agc_cucumber_2018"]["readiness"]["status"], "missing")
        cucumber_source = by_id["agc_cucumber_2018"]["readiness"]["source_integrity"]
        tomato_source = by_id["agc_tomato_2019"]["readiness"]["source_integrity"]
        self.assertEqual(cucumber_source["status"], "missing")
        self.assertEqual(cucumber_source["sources"][0]["expected_size_bytes"], 8_975_944)
        self.assertEqual(
            cucumber_source["sources"][0]["expected_md5"],
            "243eaa9041da23d0c4bf99576715aa44",
        )
        self.assertEqual(tomato_source["sources"][0]["expected_size_bytes"], 8_418_715)
        self.assertEqual(
            tomato_source["sources"][0]["expected_md5"],
            "2a0c7f3332881caef54ca8f4dc60c9a3",
        )
        lettuce_readiness = by_id["agc_lettuce_online_rgbd_2021"]["readiness"]
        self.assertEqual(lettuce_readiness["status"], "catalog_only")
        self.assertEqual(lettuce_readiness["source_integrity"]["status"], "not_checked")
        self.assertTrue(by_id["generated-toy-series@1"]["display_name_zh"])

    def test_source_archive_is_audited_without_changing_extracted_data_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            data_root = Path(raw)
            descriptor = self.registry._descriptors["agc_cucumber_2018"]
            _touch_required_files(
                data_root / descriptor.dataset_id,
                descriptor.required_globs,
                "TeamA",
            )
            registry = DatasetRegistry(catalog_path=CATALOG_PATH, data_root=data_root)

            readiness = registry.describe(descriptor.dataset_id)["readiness"]

        self.assertEqual(readiness["status"], "ready")
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["source_integrity"]["status"], "missing")
        self.assertFalse(readiness["source_integrity"]["verified"])
        self.assertIn("运行就绪状态不受影响", readiness["source_integrity"]["message_zh"])
        self.assertEqual(readiness["provenance"]["source_integrity_status"], "missing")

    def test_source_archive_reports_verified_and_mismatched_checksums_without_paths(self) -> None:
        archive_content = b"source archive fixture"
        expected_md5 = hashlib.md5(archive_content).hexdigest()
        with tempfile.TemporaryDirectory() as raw:
            data_root = Path(raw)
            dataset_id = "agc_cucumber_2018"
            registry = DatasetRegistry(catalog_path=CATALOG_PATH, data_root=data_root)
            descriptor = replace(
                registry._descriptors[dataset_id],
                source_files=(
                    {
                        "name": "fixture.zip",
                        "size_bytes": len(archive_content),
                        "md5": expected_md5,
                        "archive_format": "zip",
                    },
                ),
            )
            registry._descriptors[dataset_id] = descriptor
            archive = data_root / dataset_id / "_archives" / "fixture.zip"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(archive_content)

            verified = registry.describe(dataset_id)["readiness"]["source_integrity"]
            archive.write_bytes(b"tampered archive fixture")
            mismatched = registry.describe(dataset_id)["readiness"]["source_integrity"]

        self.assertEqual(verified["status"], "verified")
        self.assertTrue(verified["verified"])
        self.assertTrue(verified["sources"][0]["size_matches"])
        self.assertTrue(verified["sources"][0]["md5_matches"])
        self.assertEqual(mismatched["status"], "mismatch")
        self.assertFalse(mismatched["verified"])
        self.assertFalse(mismatched["sources"][0]["size_matches"])
        self.assertFalse(mismatched["sources"][0]["md5_matches"])
        self.assertNotIn(str(data_root), json.dumps(mismatched, ensure_ascii=False))
        self.assertNotIn("path", mismatched["sources"][0])

    def test_fetch_reuses_verified_archive_and_ready_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            data_root = Path(raw)
            registry = DatasetRegistry(catalog_path=CATALOG_PATH, data_root=data_root)
            _bind_fixture_zip(
                registry,
                data_root,
                [("TeamA/Greenhouse_climate.csv", b"time,value\n1,20\n")],
            )

            first = registry.fetch_data(("agc_cucumber_2018",))
            extracted = (
                data_root
                / "agc_cucumber_2018"
                / "TeamA"
                / "Greenhouse_climate.csv"
            )
            first_mtime = extracted.stat().st_mtime_ns
            second = registry.fetch_data(("agc_cucumber_2018",))
            extracted_content = extracted.read_bytes()
            second_mtime = extracted.stat().st_mtime_ns

        self.assertEqual(first["prepared"][0]["archives"][0]["action"], "reused")
        self.assertEqual(first["prepared"][0]["extraction"], "extracted")
        self.assertEqual(second["prepared"][0]["extraction"], "reused")
        self.assertEqual(extracted_content, b"time,value\n1,20\n")
        self.assertEqual(second_mtime, first_mtime)

    def test_fetch_rejects_zip_traversal_and_symbolic_links(self) -> None:
        link = zipfile.ZipInfo("TeamA/Greenhouse_climate.csv")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        cases = (
            ([("../outside.csv", b"escape")], "不安全路径"),
            ([(link, b"../../outside.csv")], "符号链接"),
        )
        for members, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as raw:
                data_root = Path(raw)
                registry = DatasetRegistry(catalog_path=CATALOG_PATH, data_root=data_root)
                _bind_fixture_zip(registry, data_root, members)
                with self.assertRaisesRegex(ValueError, message):
                    registry.fetch_data(("agc_cucumber_2018",))
                self.assertFalse((data_root / "outside.csv").exists())

    def test_fetch_does_not_overwrite_conflicting_extracted_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            data_root = Path(raw)
            registry = DatasetRegistry(catalog_path=CATALOG_PATH, data_root=data_root)
            _bind_fixture_zip(
                registry,
                data_root,
                [
                    ("TeamA/Greenhouse_climate.csv", b"archive"),
                    ("meteo.csv", b"weather"),
                ],
                required_globs=("TeamA/Greenhouse_climate.csv", "meteo.csv"),
            )
            existing = (
                data_root
                / "agc_cucumber_2018"
                / "TeamA"
                / "Greenhouse_climate.csv"
            )
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"local")

            with self.assertRaisesRegex(FileExistsError, "内容不一致"):
                registry.fetch_data(("agc_cucumber_2018",))

            self.assertEqual(existing.read_bytes(), b"local")
            self.assertFalse((data_root / "agc_cucumber_2018" / "meteo.csv").exists())

    def test_data_cli_parses_audit_and_archive_only_fetch(self) -> None:
        parser = build_parser()
        audit = parser.parse_args(["data", "audit", "agc_cucumber_2018"])
        fetch = parser.parse_args(
            [
                "data",
                "fetch",
                "agc_tomato_2019",
                "--data-root",
                "/tmp/agc-data",
                "--archive-only",
            ]
        )
        self.assertEqual(audit.dataset_ids, ["agc_cucumber_2018"])
        self.assertEqual(fetch.dataset_ids, ["agc_tomato_2019"])
        self.assertEqual(fetch.data_root, "/tmp/agc-data")
        self.assertTrue(fetch.archive_only)

    def test_toy_series_contract_and_feedback_profile(self) -> None:
        description = self.registry.describe("generated-toy-series@1")
        series = self.registry.series("generated-toy-series@1")
        self.assertIsInstance(series, DatasetSeries)
        self.assertEqual(series.schema, "ecologyrsi-dsh.dataset-series/1")
        self.assertEqual(series.evaluation_partition, "training_feedback")
        self.assertEqual(description["profile"]["evaluation_partition"], "training_feedback")
        self.assertEqual(description["visible_partitions"], ["training_fit", "training_feedback"])
        self.assertIn("development", description["restricted_partitions"])
        self.assertEqual(len(series.timestamps), series.partitions["development"].end)

    def test_new_protocol_cannot_call_legacy_full_series(self) -> None:
        with self.assertRaisesRegex(PermissionError, "selection_view|legacy"):
            self.registry.series(
                "generated-toy-series@1",
                execution_protocol="dsh_native_plugin_evolution@1",
            )

    def test_sample_pagination_and_limit(self) -> None:
        first = self.registry.sample("generated-toy-series@1", limit=3)
        second = self.registry.sample(
            "generated-toy-series@1", offset=first["next_offset"], limit=3
        )
        self.assertEqual(len(first["rows"]), 3)
        self.assertEqual(first["rows"][0]["index"], 0)
        self.assertEqual(second["rows"][0]["index"], 3)
        self.assertIn("降雨", {item["display_name_zh"] for item in first["features"].values()})
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            self.registry.sample("generated-toy-series@1", limit=101)

    def test_series_and_samples_reject_frozen_digest_drift(self) -> None:
        series = self.registry.series("generated-toy-series@1")
        replay = self.registry.series(
            "generated-toy-series@1",
            expected_dataset_digest=series.digest,
            expected_split_manifest_digest=series.split_manifest_digest_sha256,
        )
        self.assertEqual(replay.digest, series.digest)
        with self.assertRaisesRegex(ValueError, "数据集快照发生漂移") as raised:
            self.registry.sample(
                "generated-toy-series@1",
                expected_dataset_digest="0" * 64,
                expected_split_manifest_digest=series.split_manifest_digest_sha256,
            )
        self.assertEqual(
            raised.exception.error_code, "frozen_runtime_binding_drift"
        )
        self.assertNotIn("0" * 64, str(raised.exception))
        self.assertNotIn(series.digest, str(raised.exception))
        with self.assertRaisesRegex(ValueError, "时间分区快照发生漂移"):
            self.registry.sample(
                "generated-toy-series@1",
                expected_dataset_digest=series.digest,
                expected_split_manifest_digest="0" * 64,
            )

    def test_new_registry_rejects_changed_csv_against_old_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            data_root = Path(raw)
            dataset_dir = data_root / "agc_cucumber_2018"
            descriptor = self.registry._descriptors["agc_cucumber_2018"]
            _touch_required_files(dataset_dir, descriptor.required_globs, "TeamA")
            climate = dataset_dir / "TeamA" / "Greenhouse_climate.csv"
            rows = [
                {
                    "GHtime": str(index / 24),
                    "Tair": 20 + index / 100,
                    "RHair": 70,
                    "CO2air": 700,
                }
                for index in range(100)
            ]
            _write_csv(
                climate,
                ["GHtime", "Tair", "RHair", "CO2air"],
                rows,
            )
            first_registry = DatasetRegistry(
                catalog_path=CATALOG_PATH,
                data_root=data_root,
            )
            frozen = first_registry.series(
                "agc_cucumber_2018", "agc_cucumber_2018:TeamA"
            )

            rows[0]["Tair"] = 35
            _write_csv(
                climate,
                ["GHtime", "Tair", "RHair", "CO2air"],
                rows,
            )
            restarted_registry = DatasetRegistry(
                catalog_path=CATALOG_PATH,
                data_root=data_root,
            )
            with self.assertRaisesRegex(ValueError, "数据集快照发生漂移"):
                restarted_registry.series(
                    "agc_cucumber_2018",
                    "agc_cucumber_2018:TeamA",
                    expected_dataset_digest=frozen.digest,
                    expected_split_manifest_digest=(
                        frozen.split_manifest_digest_sha256
                    ),
                )

    def test_development_gate_and_external_partitions_fail_closed(self) -> None:
        for partition in ("development", "gate", "external", "external_holdout", "hidden", "test", "final"):
            with self.subTest(partition=partition):
                with self.assertRaises(PermissionError):
                    self.registry.sample("generated-toy-series@1", partition=partition)

    def test_reference_episode_is_rejected_by_registry(self) -> None:
        reference_id = "agc_cucumber_2018:Reference"
        with tempfile.TemporaryDirectory() as raw:
            data_root = Path(raw)
            dataset_dir = data_root / "agc_cucumber_2018"
            descriptor = next(
                item for item in self.registry._descriptors.values() if item.dataset_id == "agc_cucumber_2018"
            )
            _touch_required_files(dataset_dir, descriptor.required_globs, "Reference")
            _write_csv(
                dataset_dir / "Reference" / "Greenhouse_climate.csv",
                ["GHtime", "Tair", "RHair", "CO2air"],
                [{"GHtime": str(index / 24), "Tair": 20, "RHair": 70, "CO2air": 700} for index in range(100)],
            )
            registry = DatasetRegistry(catalog_path=CATALOG_PATH, data_root=data_root)
            with self.assertRaises(PermissionError):
                registry.series("agc_cucumber_2018", reference_id)

    @unittest.skipUnless(os.environ.get("ECOLOGYRSI_TEST_REAL_DATA") == "1", "需设置 ECOLOGYRSI_TEST_REAL_DATA=1")
    def test_optional_real_greenhouse_data(self) -> None:
        registry = DatasetRegistry(catalog_path=CATALOG_PATH)
        ready = [item for item in registry.catalog()["datasets"] if item["readiness"]["ready"]]
        greenhouse = [item for item in ready if item["adapter_id"] == "greenhouse_timeseries"]
        self.assertTrue(greenhouse, "未找到已就绪的真实温室数据")
        series = registry.series(greenhouse[0]["dataset_id"])
        self.assertGreater(len(series.timestamps), 0)


if __name__ == "__main__":
    unittest.main()
