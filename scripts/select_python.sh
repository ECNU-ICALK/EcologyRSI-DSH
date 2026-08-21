#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

candidate="${PYTHON:-}"
if [ -z "$candidate" ] && command -v uv >/dev/null 2>&1; then
  candidate="$(uv python find --no-project --system '>=3.10' 2>/dev/null || true)"
fi
if [ -z "$candidate" ] && [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  candidate="$ROOT_DIR/.venv/bin/python"
fi
if [ -z "$candidate" ]; then
  candidate="$(command -v python3 || true)"
fi
if [ -z "$candidate" ]; then
  echo "Python >= 3.10 is required" >&2
  exit 1
fi

if ! "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "selected interpreter must be Python >= 3.10: $candidate" >&2
  exit 1
fi

printf '%s\n' "$candidate"
