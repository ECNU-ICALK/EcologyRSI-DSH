"""Verified download and safe archive preparation helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Mapping
import urllib.request
from urllib.parse import urlsplit
import zipfile

from .contracts import DatasetDescriptor


def _missing_required_files(dataset_dir: Path, patterns: tuple[str, ...]) -> list[str]:
    return [
        pattern
        for pattern in patterns
        if not any(path.is_file() and not path.is_symlink() for path in dataset_dir.glob(pattern))
    ]


def _validated_source(source: Mapping[str, Any]) -> dict[str, Any]:
    name = source.get("name")
    size = source.get("size_bytes")
    checksum = source.get("md5")
    url = source.get("download_url")
    archive_format = source.get("archive_format")
    if (
        not isinstance(name, str)
        or not name.strip()
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("来源归档名称必须是不含路径的文件名")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"来源归档 {name} 的 size_bytes 无效")
    if (
        not isinstance(checksum, str)
        or len(checksum.strip()) != 32
        or any(character not in "0123456789abcdefABCDEF" for character in checksum.strip())
    ):
        raise ValueError(f"来源归档 {name} 的 MD5 无效")
    if not isinstance(url, str) or urlsplit(url).scheme != "https":
        raise ValueError(f"来源归档 {name} 必须使用 HTTPS 下载地址")
    if archive_format not in {"zip", "7z"}:
        raise ValueError(f"来源归档 {name} 的格式不受支持：{archive_format}")
    return {
        "name": name,
        "size_bytes": size,
        "md5": checksum.strip().casefold(),
        "download_url": url,
        "archive_format": archive_format,
    }

def _source_matches(path: Path, source: Mapping[str, Any]) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == source["size_bytes"]
        and _file_md5(path) == source["md5"]
    )


def _download_verified_source(source: Mapping[str, Any], destination: Path) -> str:
    if destination.exists() or destination.is_symlink():
        if _source_matches(destination, source):
            return "reused"
        raise FileExistsError(
            f"现有归档 {destination.name} 与目录记录不一致；"
            "为避免覆盖，请先人工移走该文件"
        )

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            request = urllib.request.Request(
                source["download_url"],
                headers={"User-Agent": "EcologyRSI-DSH/0.2"},
            )
            checksum = hashlib.md5()
            downloaded = 0
            with urllib.request.urlopen(request, timeout=120) as response:
                final_url = response.geturl()
                if urlsplit(final_url).scheme != "https":
                    raise ValueError("数据下载重定向不得降级为非 HTTPS 地址")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > source["size_bytes"]:
                        raise ValueError(
                            f"下载文件 {source['name']} 超过目录记录的大小"
                        )
                    checksum.update(chunk)
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if downloaded != source["size_bytes"]:
            raise ValueError(
                f"下载文件 {source['name']} 大小不匹配："
                f"{downloaded} != {source['size_bytes']}"
            )
        if checksum.hexdigest() != source["md5"]:
            raise ValueError(f"下载文件 {source['name']} 未通过 MD5 校验")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"归档目标已被其他文件占用：{destination.name}")
        temporary.replace(destination)
        temporary = None
        return "downloaded"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _safe_archive_member(name: str) -> PurePosixPath:
    if not name or "\x00" in name:
        raise ValueError("归档包含空名称或 NUL 字符")
    normalized = PurePosixPath(name.replace("\\", "/"))
    if (
        normalized.is_absolute()
        or ".." in normalized.parts
        or not normalized.parts
        or normalized.parts[0].endswith(":")
        or normalized.parts[0] == "_archives"
    ):
        raise ValueError(f"归档包含不安全路径：{name}")
    return normalized


def _extract_verified_source(
    source: Mapping[str, Any], archive: Path, dataset_dir: Path
) -> None:
    if not _source_matches(archive, source):
        raise ValueError(f"归档 {source['name']} 在解压前未通过完整性校验")
    with tempfile.TemporaryDirectory(
        prefix=f".{dataset_dir.name}.extract-", dir=dataset_dir.parent
    ) as raw_staging:
        staging = Path(raw_staging)
        if source["archive_format"] == "zip":
            _extract_safe_zip(archive, staging)
        else:
            _extract_safe_7z(archive, staging)
        _publish_staging(staging, dataset_dir)


def _extract_safe_zip(archive: Path, staging: Path) -> None:
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = _safe_archive_member(member.filename)
            folded = relative.as_posix().casefold()
            if folded in seen and not member.is_dir():
                raise ValueError(f"ZIP 包含重复成员：{member.filename}")
            seen.add(folded)
            mode = member.external_attr >> 16
            member_type = stat.S_IFMT(mode)
            if member_type == stat.S_IFLNK:
                raise ValueError(f"ZIP 包含符号链接：{member.filename}")
            if member_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError(f"ZIP 包含特殊文件：{member.filename}")
            target = staging.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"ZIP 包含重复目标：{member.filename}")
            with bundle.open(member) as source_stream, target.open("xb") as output:
                shutil.copyfileobj(source_stream, output, length=1024 * 1024)


def _extract_safe_7z(archive: Path, staging: Path) -> None:
    bsdtar = shutil.which("bsdtar")
    if bsdtar is None:
        raise RuntimeError("解压番茄 7z 数据需要系统提供 bsdtar")
    environment = {**os.environ, "LC_ALL": "C"}
    try:
        names = subprocess.run(
            [bsdtar, "-tf", str(archive)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.splitlines()
        verbose = subprocess.run(
            [bsdtar, "-tvf", str(archive)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.splitlines()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"无法读取 7z 归档目录：{archive.name}") from exc
    for name in names:
        _safe_archive_member(name)
    for line in verbose:
        if line and line[0] not in {"-", "d"}:
            raise ValueError("7z 归档包含符号链接、硬链接或特殊文件")
    try:
        subprocess.run(
            [bsdtar, "-xf", str(archive), "-C", str(staging)],
            check=True,
            capture_output=True,
            env=environment,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"无法解压 7z 归档：{archive.name}") from exc


def _publish_staging(staging: Path, dataset_dir: Path) -> None:
    entries = sorted(staging.rglob("*"), key=lambda item: (len(item.parts), str(item)))
    for entry in entries:
        mode = entry.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(f"解压结果包含符号链接或特殊文件：{entry.name}")
        relative = entry.relative_to(staging)
        target = dataset_dir / relative
        for parent in (target, *target.parents):
            if parent == dataset_dir.parent:
                break
            if parent.is_symlink():
                raise ValueError(f"目标目录包含符号链接：{relative}")
        if target.exists():
            if entry.is_dir() and target.is_dir():
                continue
            if entry.is_file() and target.is_file() and _files_equal(entry, target):
                continue
            raise FileExistsError(f"解压目标已存在且内容不一致：{relative}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        relative = entry.relative_to(staging)
        target = dataset_dir / relative
        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with entry.open("rb") as source_stream, target.open("xb") as output:
                shutil.copyfileobj(source_stream, output, length=1024 * 1024)


def _files_equal(left: Path, right: Path) -> bool:
    return left.stat().st_size == right.stat().st_size and _file_md5(left) == _file_md5(right)


def _audit_source_archive(
    dataset_dir: Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    raw_name = source.get("name")
    name = str(raw_name).strip() if isinstance(raw_name, str) else ""
    expected_size = source.get("size_bytes")
    expected_md5 = source.get("md5")
    expected_md5 = expected_md5.strip().casefold() if isinstance(expected_md5, str) else None
    metadata_valid = (
        bool(name)
        and Path(name).name == name
        and isinstance(expected_size, int)
        and not isinstance(expected_size, bool)
        and expected_size >= 0
        and isinstance(expected_md5, str)
        and len(expected_md5) == 32
        and all(character in "0123456789abcdef" for character in expected_md5)
    )
    record: dict[str, Any] = {
        "name": name or "未命名来源归档",
        "status": "unverifiable",
        "exists": False,
        "expected_size_bytes": (
            expected_size
            if isinstance(expected_size, int) and not isinstance(expected_size, bool)
            else None
        ),
        "actual_size_bytes": None,
        "size_matches": None,
        "expected_md5": expected_md5,
        "actual_md5": None,
        "md5_matches": None,
        "message_zh": "来源目录元数据不完整或格式无效。",
    }
    if not metadata_valid:
        return record

    archive_path = dataset_dir / "_archives" / name
    try:
        actual_size = archive_path.stat().st_size
    except FileNotFoundError:
        record.update(
            status="missing",
            message_zh=f"来源归档 {name} 不存在。",
        )
        return record
    except OSError:
        record.update(
            status="unreadable",
            message_zh=f"来源归档 {name} 无法读取。",
        )
        return record
    if not archive_path.is_file():
        record.update(
            status="unreadable",
            exists=True,
            message_zh=f"来源归档 {name} 不是普通文件。",
        )
        return record

    try:
        actual_md5 = _file_md5(archive_path)
    except OSError:
        record.update(
            status="unreadable",
            exists=True,
            actual_size_bytes=actual_size,
            size_matches=actual_size == expected_size,
            message_zh=f"来源归档 {name} 无法完整读取。",
        )
        return record

    size_matches = actual_size == expected_size
    md5_matches = actual_md5 == expected_md5
    record.update(
        status="verified" if size_matches and md5_matches else "mismatch",
        exists=True,
        actual_size_bytes=actual_size,
        size_matches=size_matches,
        actual_md5=actual_md5,
        md5_matches=md5_matches,
        message_zh=(
            f"来源归档 {name} 已通过文件大小和 MD5 校验。"
            if size_matches and md5_matches
            else f"来源归档 {name} 的文件大小或 MD5 与目录记录不一致。"
        ),
    )
    return record


def _unchecked_source_archive(source: Mapping[str, Any]) -> dict[str, Any]:
    raw_name = source.get("name")
    name = str(raw_name).strip() if isinstance(raw_name, str) else ""
    expected_size = source.get("size_bytes")
    expected_md5 = source.get("md5")
    return {
        "name": name or "未命名来源归档",
        "status": "not_checked",
        "exists": None,
        "expected_size_bytes": (
            expected_size
            if isinstance(expected_size, int) and not isinstance(expected_size, bool)
            else None
        ),
        "actual_size_bytes": None,
        "size_matches": None,
        "expected_md5": (
            expected_md5.strip().casefold() if isinstance(expected_md5, str) else None
        ),
        "actual_md5": None,
        "md5_matches": None,
        "message_zh": "数据集尚不可运行，未读取本地来源归档。",
    }


def _file_md5(path: Path) -> str:
    checksum = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _source_integrity_result(
    *,
    status: str,
    verified: bool | None,
    sources: list[dict[str, Any]],
    message_zh: str,
) -> dict[str, Any]:
    return {
        "schema_version": "ecologyrsi-dsh.source-integrity/1",
        "status": status,
        "verified": verified,
        "source_count": len(sources),
        "verified_count": sum(item["status"] == "verified" for item in sources),
        "missing_count": sum(item["status"] == "missing" for item in sources),
        "mismatch_count": sum(item["status"] == "mismatch" for item in sources),
        "unverifiable_count": sum(
            item["status"] in {"unreadable", "unverifiable"} for item in sources
        ),
        "message_zh": message_zh,
        "sources": sources,
    }


def _provenance_summary(
    descriptor: DatasetDescriptor,
    source_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "publisher": descriptor.publisher,
        "doi": descriptor.doi,
        "license": descriptor.license,
        "source_integrity_status": source_integrity["status"],
    }
