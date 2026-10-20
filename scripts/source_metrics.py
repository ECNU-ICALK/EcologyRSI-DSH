#!/usr/bin/env python3
"""Compare production code against a Git baseline without counting tests/docs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ROOTS = {"src", "plugins", "integrations", "scripts"}
SUFFIXES = {".py", ".js", ".mjs", ".css", ".html", ".sh"}
EXCLUDED = {"dist", "node_modules", "__pycache__", "test", "tests", ".runtime"}


def selected(name: str) -> bool:
    path = Path(name)
    return (path.parts[0] in ROOTS and path.suffix in SUFFIXES
            and not EXCLUDED.intersection(path.parts))


def measure(files: dict[str, bytes]) -> dict:
    groups = {}
    for name, raw in files.items():
        group = groups.setdefault(Path(name).parts[0], {"files": 0, "lines": 0, "nonblank": 0, "bytes": 0})
        lines = raw.decode("utf-8").splitlines()
        group["files"] += 1
        group["lines"] += len(lines)
        group["nonblank"] += sum(bool(line.strip()) for line in lines)
        group["bytes"] += len(raw)
    return {"groups": groups, "total": {key: sum(group[key] for group in groups.values())
                                      for key in ("files", "lines", "nonblank", "bytes")}}


def compare(baseline: str) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "--verify", baseline + "^{commit}"],
                                     cwd=ROOT, text=True).strip()
    names = subprocess.check_output(["git", "ls-tree", "-rz", "--name-only", commit], cwd=ROOT).decode().split("\0")
    before = measure({name: subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
                      for name in names if name and selected(name)})
    current = {}
    for folder in ROOTS:
        for path in (ROOT / folder).rglob("*"):
            name = path.relative_to(ROOT).as_posix()
            if path.is_file() and selected(name):
                current[name] = path.read_bytes()
    after = measure(current)
    return {"baseline_commit": commit, "scope": "production Python/JS/HTML/CSS/shell; tests, docs, JSON and archives excluded",
            "before": before, "after": after,
            "retained_percent": {key: round(after["total"][key] / before["total"][key] * 100, 2)
                                 for key in ("lines", "nonblank", "bytes")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="3365016f45ce13cd0df73adb78ff6c618f80b14a")
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline), ensure_ascii=False, indent=2))
