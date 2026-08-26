#!/usr/bin/env bash
set -euo pipefail

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONUTF8=1

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$("$SCRIPT_DIR/select_python.sh")"
DIST_DIR="$ROOT_DIR/dist"
PLUGIN_ROOT="$ROOT_DIR/integrations/dsh_ecology_plugin"
PLUGIN_DIST="$PLUGIN_ROOT/dist"

cd "$ROOT_DIR"

if [ "${ECOLOGYRSI_ALLOW_DIRTY_BUILD:-0}" != "1" ] && \
  [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
  echo "release builds require a clean worktree; set ECOLOGYRSI_ALLOW_DIRTY_BUILD=1 for a non-final candidate" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to build release artifacts" >&2
  exit 1
fi

PYTHON="$PYTHON_BIN" "$SCRIPT_DIR/verify_delivery.sh" --source-only

mkdir -p "$PLUGIN_DIST"
find "$PLUGIN_DIST" -maxdepth 1 -type f -name '*.tgz' -delete
"$PYTHON_BIN" "$SCRIPT_DIR/build_dsh_plugin.py" \
  --root "$ROOT_DIR" \
  --output-dir "$PLUGIN_DIST"

mkdir -p "$DIST_DIR"
uv build \
  --clear \
  --python "$PYTHON_BIN" \
  --out-dir "$DIST_DIR" \
  "$ROOT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/create_delivery_archive.py" \
  --root "$ROOT_DIR" \
  --dist "$DIST_DIR"

PYTHON="$PYTHON_BIN" "$SCRIPT_DIR/verify_delivery.sh" --artifacts-only

echo "release artifacts:"
find "$DIST_DIR" -maxdepth 1 -type f -print | sort
