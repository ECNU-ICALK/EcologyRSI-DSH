"""Dataset registry and read-only sample/query service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .catalog import (
    _assert_snapshot_digest,
    _default_catalog_path,
    _default_data_root,
    _toy_descriptor,
    _toy_series,
)
from .contracts import (
    _PREPARABLE_DATASET_IDS,
    _RESTRICTED_PARTITIONS,
    _TOY_DATASET_ID,
    _VISIBLE_PARTITIONS,
    DatasetDescriptor,
    DatasetSeries,
    SelectionDatasetView,
)
from .greenhouse import CanonicalEpisode, CanonicalSeries, GreenhouseDatasetAdapter
from .preparation import (
    _audit_source_archive,
    _download_verified_source,
    _extract_verified_source,
    _missing_required_files,
    _provenance_summary,
    _source_integrity_result,
    _unchecked_source_archive,
    _validated_source,
)
from .splits import (
    SplitManifest,
    build_four_stage_data_protocol,
    build_split_manifest,
)


class DatasetRegistry:
    """Resolve catalogue metadata and expose only non-restricted sample rows."""

    def __init__(
        self,
        *,
        catalog_path: str | Path | None = None,
        data_root: str | Path | None = None,
    ) -> None:
        self.catalog_path = _default_catalog_path() if catalog_path is None else Path(catalog_path).expanduser().resolve()
        self.data_root = _default_data_root() if data_root is None else Path(data_root).expanduser().resolve()
        self._descriptors = self._load_catalog(self.catalog_path)
        self._descriptors[_TOY_DATASET_ID] = _toy_descriptor()
        self._series_cache: dict[str, CanonicalSeries] = {}
        self._split_cache: dict[str, SplitManifest] = {}

    def catalog(self) -> dict[str, Any]:
        return {
            "schema_version": "ecologyrsi-dsh.dataset-catalog-response/1",
            "datasets": [
                {
                    **item.to_dict(),
                    "readiness": self._readiness(item),
                }
                for item in sorted(self._descriptors.values(), key=lambda value: value.dataset_id)
            ],
        }

    def audit_data(self, dataset_ids: tuple[str, ...] | None = None) -> dict[str, Any]:
        """Audit the two locally preparable AGC datasets without changing files."""

        descriptors = self._preparable_descriptors(dataset_ids)
        return {
            "schema_version": "ecologyrsi-dsh.dataset-audit/1",
            "data_root": str(self.data_root),
            "datasets": [
                {
                    "dataset_id": descriptor.dataset_id,
                    "display_name_zh": descriptor.display_name_zh,
                    **self._readiness(descriptor),
                }
                for descriptor in descriptors
            ],
        }

    def fetch_data(
        self,
        dataset_ids: tuple[str, ...],
        *,
        extract: bool = True,
    ) -> dict[str, Any]:
        """Download verified AGC archives and safely prepare their extracted files."""

        descriptors = self._preparable_descriptors(dataset_ids)
        self.data_root.mkdir(parents=True, exist_ok=True)
        prepared: list[dict[str, Any]] = []
        for descriptor in descriptors:
            dataset_dir = self._dataset_dir(descriptor)
            if dataset_dir.is_symlink():
                raise ValueError(f"数据集目录不得是符号链接：{descriptor.dataset_id}")
            archives_dir = dataset_dir / "_archives"
            if archives_dir.is_symlink():
                raise ValueError(f"归档目录不得是符号链接：{descriptor.dataset_id}")
            archives_dir.mkdir(parents=True, exist_ok=True)
            archive_results: list[dict[str, Any]] = []
            for raw_source in descriptor.source_files:
                source = _validated_source(raw_source)
                archive = archives_dir / source["name"]
                archive_action = _download_verified_source(source, archive)
                archive_results.append(
                    {
                        "name": source["name"],
                        "action": archive_action,
                        "size_bytes": source["size_bytes"],
                        "md5": source["md5"],
                    }
                )

            extraction_action = "not_requested"
            if extract:
                if not _missing_required_files(dataset_dir, descriptor.required_globs):
                    extraction_action = "reused"
                else:
                    for raw_source in descriptor.source_files:
                        source = _validated_source(raw_source)
                        _extract_verified_source(
                            source,
                            archives_dir / source["name"],
                            dataset_dir,
                        )
                    missing = _missing_required_files(
                        dataset_dir, descriptor.required_globs
                    )
                    if missing:
                        raise RuntimeError(
                            f"数据集 {descriptor.dataset_id} 解压后仍缺少必需文件："
                            + ", ".join(missing)
                        )
                    extraction_action = "extracted"

            prepared.append(
                {
                    "dataset_id": descriptor.dataset_id,
                    "archives": archive_results,
                    "extraction": extraction_action,
                }
            )
        return {
            "schema_version": "ecologyrsi-dsh.dataset-fetch/1",
            "data_root": str(self.data_root),
            "prepared": prepared,
            "audit": self.audit_data(tuple(item.dataset_id for item in descriptors)),
        }

    def episodes(self, dataset_id: str) -> list[dict[str, Any]]:
        """Return browser-safe optimization episode choices."""

        descriptor = self._descriptor(dataset_id)
        if descriptor.dataset_id == _TOY_DATASET_ID:
            return [
                {
                    "id": "generated-toy-series@1:seed-0",
                    "episode_id": "generated-toy-series@1:seed-0",
                    "label": "固定 seed-0 序列",
                    "row_count": 60,
                }
            ]
        if descriptor.adapter_id == "greenhouse_timeseries":
            filename = (
                "Greenhouse_climate.csv"
                if descriptor.domain_id == "greenhouse_cucumber_2018"
                else "GreenhouseClimate.csv"
            )
            result = []
            for path in sorted(self._dataset_dir(descriptor).glob(f"*/{filename}")):
                team = path.parent.name
                if "reference" in team.casefold():
                    continue
                episode_id = f"{descriptor.dataset_id}:{team}"
                result.append(
                    {
                        "id": episode_id,
                        "episode_id": episode_id,
                        "label": team,
                        "row_count": None,
                    }
                )
            return result
        canonical = self._load_series(descriptor)
        manifest = self._split_manifest(descriptor, canonical)
        result: list[dict[str, Any]] = []
        for episode in canonical.episodes:
            split = manifest.split_for(episode.episode_id)
            if split.role != "optimization":
                continue
            team = episode.episode_id.rsplit(":", 1)[-1]
            result.append(
                {
                    "id": episode.episode_id,
                    "episode_id": episode.episode_id,
                    "label": team,
                    "row_count": len(episode.timestamps),
                }
            )
        return result

    def describe(self, dataset_id: str) -> dict[str, Any]:
        descriptor = self._descriptor(dataset_id)
        return {
            "schema_version": "ecologyrsi-dsh.dataset-description/1",
            "descriptor": descriptor.to_dict(),
            "readiness": self._readiness(descriptor),
            "profile": self._profile(descriptor),
            "visible_partitions": list(_VISIBLE_PARTITIONS),
            "restricted_partitions": sorted(_RESTRICTED_PARTITIONS),
        }

    def series(
        self,
        dataset_id: str,
        episode_id: str | None = None,
        *,
        expected_dataset_digest: str | None = None,
        expected_split_manifest_digest: str | None = None,
        execution_protocol: str | None = None,
    ) -> DatasetSeries:
        if execution_protocol == "dsh_native_plugin_evolution@1":
            raise PermissionError(
                "DSH-native execution must use selection_view; full series is legacy-only"
            )
        descriptor = self._descriptor(dataset_id)
        canonical = self._load_series(descriptor)
        split_manifest = self._split_manifest(descriptor, canonical)
        episode = self._select_episode(canonical, split_manifest, episode_id)
        split = split_manifest.split_for(episode.episode_id)
        if split.role == "external_holdout":
            raise PermissionError("external holdout episodes are not available")

        # Keep the development range for evaluator integration, but omit the
        # post-development embargo and gate tail. ``sample`` separately denies
        # development rows so the browser can show only aggregate readiness.
        visible_end = split.development.end
        result = DatasetSeries(
            schema="ecologyrsi-dsh.dataset-series/1",
            dataset_id=episode.dataset_id,
            domain_id=episode.domain_id,
            episode_id=episode.episode_id,
            digest=episode.content_sha256,
            timestamps=episode.timestamps[:visible_end],
            values={name: items[:visible_end] for name, items in episode.values.items()},
            partitions={
                "training_fit": split.training_fit,
                "training_feedback": split.training_feedback,
                "development": split.development,
            },
            features=episode.features,
            split_manifest_digest_sha256=split_manifest.split_manifest_digest_sha256,
        )
        _assert_snapshot_digest(
            "数据集快照",
            result.digest,
            expected_dataset_digest,
        )
        _assert_snapshot_digest(
            "时间分区快照",
            result.split_manifest_digest_sha256,
            expected_split_manifest_digest,
        )
        return result

    def selection_view(
        self,
        dataset_id: str,
        episode_id: str | None = None,
        *,
        expected_dataset_digest: str | None = None,
        expected_split_manifest_digest: str | None = None,
        expected_data_protocol_digest: str | None = None,
        target_names: tuple[str, ...] = (
            "air_temperature",
            "relative_humidity",
            "co2_concentration",
        ),
        horizons: tuple[int, ...] = (1, 6, 24),
        history_steps: int = 3,
    ) -> SelectionDatasetView:
        """Return the only raw-row view allowed to a new adaptive evaluator."""

        descriptor = self._descriptor(dataset_id)
        canonical = self._load_series(descriptor)
        manifest = self._split_manifest(descriptor, canonical)
        episode = self._select_episode(canonical, manifest, episode_id)
        split = manifest.split_for(episode.episode_id)
        if split.role != "optimization":
            raise PermissionError("external holdout episodes are not available")
        legacy = self.series(
            dataset_id,
            episode.episode_id,
            expected_dataset_digest=expected_dataset_digest,
            expected_split_manifest_digest=expected_split_manifest_digest,
        )
        protocol = build_four_stage_data_protocol(
            dataset_id,
            episode,
            split,
            dataset_digest=legacy.digest,
            split_manifest_digest=legacy.split_manifest_digest_sha256,
            target_names=target_names,
            horizons=horizons,
            history_steps=history_steps,
        )
        _assert_snapshot_digest(
            "四阶段数据协议",
            protocol.protocol_digest,
            expected_data_protocol_digest,
        )
        return SelectionDatasetView.from_series(legacy, protocol)

    def sample(
        self,
        dataset_id: str,
        partition: str = "training_fit",
        episode_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
        *,
        expected_dataset_digest: str | None = None,
        expected_split_manifest_digest: str | None = None,
    ) -> dict[str, Any]:
        normalized_partition = str(partition).strip().casefold().replace("-", "_")
        if normalized_partition in _RESTRICTED_PARTITIONS:
            raise PermissionError(f"restricted dataset partition: {normalized_partition}")
        if normalized_partition not in _VISIBLE_PARTITIONS:
            raise ValueError(f"unknown or unsupported dataset partition: {normalized_partition}")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")

        series = self.series(
            dataset_id,
            episode_id,
            expected_dataset_digest=expected_dataset_digest,
            expected_split_manifest_digest=expected_split_manifest_digest,
        )
        selected = series.partitions[normalized_partition]
        start = min(selected.start + offset, selected.end)
        end = min(start + limit, selected.end)
        rows = []
        for index in range(start, end):
            rows.append(
                {
                    "index": index,
                    "timestamp": series.timestamps[index],
                    "values": {name: values[index] for name, values in series.values.items()},
                }
            )
        consumed = end - selected.start
        return {
            "schema_version": "ecologyrsi-dsh.dataset-sample-page/1",
            "dataset_id": series.dataset_id,
            "domain_id": series.domain_id,
            "episode_id": series.episode_id,
            "dataset_digest_sha256": series.digest,
            "split_manifest_digest_sha256": series.split_manifest_digest_sha256,
            "partition": normalized_partition,
            "evaluation_partition": series.evaluation_partition,
            "offset": offset,
            "limit": limit,
            "total": selected.size,
            "next_offset": consumed if end < selected.end else None,
            "features": {name: item.to_dict() for name, item in series.features.items()},
            "rows": rows,
        }

    @staticmethod
    def _load_catalog(path: Path) -> dict[str, DatasetDescriptor]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"dataset catalog does not exist: {path}") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid dataset catalog JSON: {exc.msg}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
            raise ValueError("dataset catalog root must contain a datasets list")
        descriptors: dict[str, DatasetDescriptor] = {}
        for raw in payload["datasets"]:
            if not isinstance(raw, dict):
                raise ValueError("dataset catalog items must be objects")
            descriptor = DatasetDescriptor.from_dict(raw)
            if descriptor.dataset_id in descriptors:
                raise ValueError(f"duplicate dataset_id: {descriptor.dataset_id}")
            descriptors[descriptor.dataset_id] = descriptor
        return descriptors

    def _descriptor(self, dataset_id: str) -> DatasetDescriptor:
        normalized = str(dataset_id).strip()
        try:
            return self._descriptors[normalized]
        except KeyError:
            raise KeyError(f"unknown dataset_id: {normalized}") from None

    def _dataset_dir(self, descriptor: DatasetDescriptor) -> Path:
        return self.data_root / descriptor.dataset_id

    def _preparable_descriptors(
        self, dataset_ids: tuple[str, ...] | None
    ) -> tuple[DatasetDescriptor, ...]:
        selected = tuple(sorted(_PREPARABLE_DATASET_IDS)) if dataset_ids is None else dataset_ids
        if not selected:
            raise ValueError("至少需要指定一个可准备的数据集")
        if len(set(selected)) != len(selected):
            raise ValueError("数据集标识不能重复")
        unsupported = sorted(set(selected) - _PREPARABLE_DATASET_IDS)
        if unsupported:
            raise ValueError(
                "当前仅支持准备 agc_cucumber_2018 和 agc_tomato_2019；不支持："
                + ", ".join(unsupported)
            )
        return tuple(self._descriptor(dataset_id) for dataset_id in selected)

    def _readiness(self, descriptor: DatasetDescriptor) -> dict[str, Any]:
        source_integrity = self._source_integrity(descriptor)
        if descriptor.dataset_id == _TOY_DATASET_ID:
            return {
                "status": "ready",
                "ready": True,
                "missing_globs": [],
                "provenance": _provenance_summary(descriptor, source_integrity),
                "source_integrity": source_integrity,
            }
        data_dir = self._dataset_dir(descriptor)
        missing = [pattern for pattern in descriptor.required_globs if not any(data_dir.glob(pattern))]
        if not descriptor.runnable:
            status = "catalog_only"
        elif missing:
            status = "missing"
        else:
            status = "ready"
        return {
            "status": status,
            "ready": status == "ready",
            "missing_globs": missing,
            "provenance": _provenance_summary(descriptor, source_integrity),
            "source_integrity": source_integrity,
        }

    def _source_integrity(self, descriptor: DatasetDescriptor) -> dict[str, Any]:
        """Audit declared source archives without exposing local filesystem paths."""

        if not descriptor.source_files:
            return _source_integrity_result(
                status="not_applicable",
                verified=None,
                sources=[],
                message_zh="该数据集没有需要校验的本地来源归档。",
            )
        if not descriptor.runnable:
            return _source_integrity_result(
                status="not_checked",
                verified=None,
                sources=[_unchecked_source_archive(source) for source in descriptor.source_files],
                message_zh="该数据集尚未提供运行适配器，因此未读取本地归档进行 MD5 校验。",
            )

        sources = [
            _audit_source_archive(self._dataset_dir(descriptor), source)
            for source in descriptor.source_files
        ]
        statuses = {item["status"] for item in sources}
        if statuses == {"verified"}:
            status = "verified"
            verified: bool | None = True
            message = f"{len(sources)} 个来源归档已通过文件大小和 MD5 校验。"
        elif "unreadable" in statuses:
            status = "unreadable"
            verified = False
            message = "部分来源归档无法读取，未通过完整性校验。"
        elif "mismatch" in statuses:
            status = "mismatch"
            verified = False
            message = "部分来源归档的文件大小或 MD5 与目录记录不一致。"
        elif "missing" in statuses:
            status = "missing"
            verified = False
            message = "部分来源归档缺失；已解压数据的运行就绪状态不受影响。"
        else:
            status = "unverifiable"
            verified = False
            message = "来源目录元数据不完整，无法完成文件大小和 MD5 校验。"
        return _source_integrity_result(
            status=status,
            verified=verified,
            sources=sources,
            message_zh=message,
        )

    def _profile(self, descriptor: DatasetDescriptor) -> dict[str, Any]:
        if descriptor.dataset_id == _TOY_DATASET_ID:
            profile = {
                "schema_version": "ecologyrsi-dsh.dataset-profile/1",
                "adapter_id": "toy_crop_soil_water",
                "domain_id": descriptor.domain_id,
                "evaluation_mode": "generated_fixture",
                "sampling": "daily",
                "required_features": ["rainfall", "evapotranspiration", "soil_water"],
                "scientific_limits_zh": ["仅用于工程测试，不代表真实农业观测或因果结论。"],
            }
        elif descriptor.adapter_id == "greenhouse_timeseries":
            profile = GreenhouseDatasetAdapter(
                descriptor.dataset_id,
                descriptor.domain_id,
                self._dataset_dir(descriptor),
            ).profile()
        else:
            profile = {
                "schema_version": "ecologyrsi-dsh.dataset-profile/1",
                "adapter_id": descriptor.adapter_id,
                "domain_id": descriptor.domain_id,
                "evaluation_mode": "not_implemented",
                "scientific_limits_zh": ["当前仅登记来源和许可，不能启动运行。"],
            }
        return {
            **profile,
            "evaluation_partition": "training_feedback",
            "split_policy": {
                "version": "time-forward-embargo/1",
                "train_fraction": 0.6,
                "training_feedback_fraction": 0.5,
                "development_fraction": 0.2,
                "training_feedback_embargo_hours": 1,
                "embargo_hours": 24,
            },
        }

    def _load_series(self, descriptor: DatasetDescriptor) -> CanonicalSeries:
        cached = self._series_cache.get(descriptor.dataset_id)
        if cached is not None:
            return cached
        readiness = self._readiness(descriptor)
        if not descriptor.runnable:
            raise ValueError(f"dataset is catalog-only and cannot run: {descriptor.dataset_id}")
        if not readiness["ready"]:
            raise FileNotFoundError(
                f"dataset is not ready: {descriptor.dataset_id}; missing {', '.join(readiness['missing_globs'])}"
            )
        if descriptor.dataset_id == _TOY_DATASET_ID:
            loaded = _toy_series()
        elif descriptor.adapter_id == "greenhouse_timeseries":
            loaded = GreenhouseDatasetAdapter(
                descriptor.dataset_id,
                descriptor.domain_id,
                self._dataset_dir(descriptor),
            ).load()
        else:  # pragma: no cover - runnable descriptors are currently explicit
            raise ValueError(f"unsupported dataset adapter: {descriptor.adapter_id}")
        self._series_cache[descriptor.dataset_id] = loaded
        return loaded

    def _split_manifest(self, descriptor: DatasetDescriptor, series: CanonicalSeries) -> SplitManifest:
        cached = self._split_cache.get(descriptor.dataset_id)
        if cached is None:
            cached = build_split_manifest(descriptor.dataset_id, series.episodes)
            self._split_cache[descriptor.dataset_id] = cached
        return cached

    @staticmethod
    def _select_episode(
        series: CanonicalSeries,
        manifest: SplitManifest,
        episode_id: str | None,
    ) -> CanonicalEpisode:
        if episode_id is not None:
            selected = next((item for item in series.episodes if item.episode_id == episode_id), None)
            if selected is None:
                raise KeyError(f"unknown episode_id: {episode_id}")
            if manifest.split_for(selected.episode_id).role == "external_holdout":
                raise PermissionError("external holdout episodes are not available")
            return selected
        for item in series.episodes:
            if manifest.split_for(item.episode_id).role == "optimization":
                return item
        raise PermissionError("dataset contains no accessible optimization episode")


__all__ = ["DatasetDescriptor", "DatasetRegistry", "DatasetSeries"]
