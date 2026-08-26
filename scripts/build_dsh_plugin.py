#!/usr/bin/env python3
"""Pack the DSH plugin from a staging tree containing root legal files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

LEGAL_FILES = ("LICENSE", "NOTICE")


def build_plugin(root: Path, output_dir: Path) -> Path:
    package_root = root / "integrations/dsh_ecology_plugin"
    package_json = package_root / "package.json"
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
                if not source.is_file() or source.is_symlink():
                    continue
                relative = source.relative_to(package_root)
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        for legal_name in LEGAL_FILES:
            shutil.copy2(root / legal_name, staging / legal_name)

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
    archive = build_plugin(args.root.resolve(), args.output_dir.resolve())
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
