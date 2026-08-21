#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$("$SCRIPT_DIR/select_python.sh")"
DIST_DIR="$ROOT_DIR/dist"
PLUGIN_ROOT="$ROOT_DIR/integrations/dsh_ecology_plugin"
PLUGIN_DIST="$PLUGIN_ROOT/dist"

cd "$ROOT_DIR"

mkdir -p "$PLUGIN_DIST"
npm pack "$PLUGIN_ROOT" --pack-destination "$PLUGIN_DIST"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to build release artifacts" >&2
  exit 1
fi

PYTHON="$PYTHON_BIN" "$SCRIPT_DIR/verify_delivery.sh" --source-only

mkdir -p "$DIST_DIR"
uv build \
  --clear \
  --python "$PYTHON_BIN" \
  --out-dir "$DIST_DIR" \
  "$ROOT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/create_delivery_archive.py" \
  --root "$ROOT_DIR" \
  --dist "$DIST_DIR"

PYTHON="$PYTHON_BIN" "$SCRIPT_DIR/verify_delivery.sh" --artifacts

echo "release artifacts:"
find "$DIST_DIR" -maxdepth 1 -type f -print | sort
