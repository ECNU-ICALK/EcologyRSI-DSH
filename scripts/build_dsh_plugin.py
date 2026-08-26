#!/usr/bin/env python3
"""Pack the DSH plugin from a staging tree containing root legal files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    from .release_safety import checked_lstat, require_directory, require_regular_file
except ImportError:
    from release_safety import checked_lstat, require_directory, require_regular_file

LEGAL_FILES = ("LICENSE", "NOTICE")


def build_plugin(root: Path, output_dir: Path) -> Path:
    package_root = root / "integrations/dsh_ecology_plugin"
    checked_lstat(root, package_root, "DSH plugin source directory")
    package_json = package_root / "package.json"
    selected_symlinks = [
        path for path in package_root.rglob("*") if path.is_symlink()
    ]
    if selected_symlinks:
        raise RuntimeError(
            f"symlink is not allowed in DSH plugin source: {selected_symlinks[0]}"
        )
    require_regular_file(root, package_json, "package.json")
    package = json.loads(package_json.read_text(encoding="utf-8"))
    patterns = package.get("files")
    if not isinstance(patterns, list) or not all(
        isinstance(pattern, str) and pattern for pattern in patterns
    ):
        raise RuntimeError("DSH plugin package files must be a list of paths")
    for legal_name in LEGAL_FILES:
        if legal_name not in patterns:
            raise RuntimeError(f"DSH plugin files must include {legal_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ecologyrsi-dsh-plugin-") as directory:
        staging = Path(directory) / "package"
        staging.mkdir()
        shutil.copy2(package_json, staging / "package.json")
        for pattern in patterns:
            if pattern in LEGAL_FILES:
                continue
            for source in package_root.glob(pattern):
                source_stat = checked_lstat(root, source, "DSH plugin source")
                if source_stat is None or not source.is_file():
                    continue
                relative = source.relative_to(package_root)
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        for legal_name in LEGAL_FILES:
            legal_source = root / legal_name
            require_regular_file(root, legal_source, "legal source")
            shutil.copy2(legal_source, staging / legal_name)

        result = subprocess.run(
            ["npm", "pack", str(staging), "--pack-destination", str(output_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        archive_name = result.stdout.strip().splitlines()[-1]

    archive = output_dir / archive_name
    if not archive.is_file():
        raise RuntimeError(f"npm did not create expected plugin archive: {archive}")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.absolute()
    output_dir = args.output_dir.absolute()
    require_directory(Path(root.anchor), root, "build root")
    require_directory(
        Path(output_dir.anchor),
        output_dir,
        "plugin output directory",
        allow_missing=True,
    )
    archive = build_plugin(root, output_dir)
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
