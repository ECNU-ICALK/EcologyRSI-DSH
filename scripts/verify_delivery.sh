#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$("$SCRIPT_DIR/select_python.sh")"
MODE="${1:---source-only}"

case "$MODE" in
  --source-only|--artifacts) ;;
  *)
    echo "usage: $0 [--source-only|--artifacts]" >&2
    exit 2
    ;;
esac

cd "$ROOT_DIR"

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("EcologyRSI-DSH delivery verification requires Python >= 3.10")
print(f"python preflight: {sys.version.split()[0]} ({sys.executable})")
PY

required_files="
README.md
CHANGELOG.md
LICENSE
NOTICE
RELEASE-CHECKLIST.md
MANIFEST.in
Makefile
pyproject.toml
examples/minimal_run.py
examples/task-manifest.json
examples/local-config.json
scripts/real_api_agent_tool_acceptance.py
scripts/dsh_native_e2e_acceptance.py
scripts/install_dsh_ecology_runtime.mjs
datasets/autonomous_greenhouse.json
plugins/ecology_evolution/index.html
plugins/ecology_evolution/styles.css
plugins/ecology_evolution/app.js
plugins/ecology_evolution/assets/js/host.js
plugins/ecology_evolution/assets/js/core.js
plugins/ecology_evolution/assets/js/commands.js
plugins/ecology_evolution/plugin.json
plugins/ecology_evolution/test/smoke.mjs
docs/screenshots/01-run-settings.jpg
docs/screenshots/02-parameter-design.jpg
docs/screenshots/03-training-data.jpg
docs/screenshots/04-evolution-process.jpg
docs/screenshots/05-candidate-evaluation.jpg
docs/screenshots/06-human-governance.jpg
integrations/dsh_ecology_plugin/package.json
integrations/dsh_ecology_plugin/lib/index.js
integrations/dsh_ecology_plugin/lib/client.js
integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js
integrations/dsh_ecology_plugin/lib/runtime/reconciliation.js
integrations/dsh_ecology_plugin/lib/tools/agent-plugin.js
integrations/dsh_ecology_plugin/schemas/genome-mutation.schema.json
integrations/dsh_ecology_plugin/presets/ecology-coordinator-v1/preset.yml
integrations/dsh_ecology_plugin/presets/ecology-generation-judge-v1/agent.cordis.yml
integrations/dsh_ecology_plugin/dist/ecologyrsi-dsh-evolution-plugin-0.3.15.tgz
integrations/dsh_ecology_plugin/test/proxy_security.mjs
"

for file in $required_files; do
  if [ ! -f "$file" ]; then
    echo "missing delivery file: $file" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import re
import sys

root = Path.cwd()
project_text = (root / "pyproject.toml").read_text(encoding="utf-8")
match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', project_text)
if match is None:
    raise SystemExit("project version is missing")
project_version = match.group(1)
package_text = (root / "src/ecologyrsi_dsh/version.py").read_text(encoding="utf-8")
package_match = re.search(r'(?m)^__version__\s*=\s*"([^"]+)"\s*$', package_text)
if package_match is None or package_match.group(1) != project_version:
    raise SystemExit("package and project versions differ")

version_sources = {
    "browser host adapter": (
        root / "plugins/ecology_evolution/assets/js/host.js",
        r'(?m)^\s*var pluginVersion\s*=\s*"([^"]+)";\s*$',
    ),
    "browser footer": (
        root / "plugins/ecology_evolution/index.html",
        r'插件版本\s+([^\s<]+)',
    ),
    "browser plugin README": (
        root / "plugins/ecology_evolution/README.md",
        r'"type":"plugin\.ready","plugin_id":"ecologyrsi\.evolution","version":"([^"]+)"',
    ),
    "NOTICE": (
        root / "NOTICE",
        r'(?m)^EcologyRSI-DSH\s+([^\s]+)\s*$',
    ),
}
for label, (path, pattern) in version_sources.items():
    source_match = re.search(pattern, path.read_text(encoding="utf-8"))
    if source_match is None:
        raise SystemExit(f"{label} version is missing")
    if source_match.group(1) != project_version:
        raise SystemExit(
            f"{label} version {source_match.group(1)!r} != project version {project_version!r}"
        )
plugin = json.loads((root / "plugins/ecology_evolution/plugin.json").read_text(encoding="utf-8"))
if plugin.get("version") != project_version:
    raise SystemExit(f"plugin version {plugin.get('version')!r} != project version {project_version!r}")
host_plugin = json.loads(
    (root / "integrations/dsh_ecology_plugin/package.json").read_text(encoding="utf-8")
)
if host_plugin.get("version") != project_version:
    raise SystemExit(
        f"host plugin version {host_plugin.get('version')!r} != project version {project_version!r}"
    )
if host_plugin.get("private") is not True or host_plugin.get("license") != "UNLICENSED":
    raise SystemExit("the proprietary host plugin must be private and UNLICENSED")
if plugin.get("development_only") is not False:
    raise SystemExit("delivery candidate must set development_only=false")
if plugin.get("release_stage") != "delivery-candidate":
    raise SystemExit("plugin release_stage must be delivery-candidate")
integrity = plugin.get("integrity", {})
if integrity.get("status") != "delivery-candidate-unsigned":
    raise SystemExit("unsigned candidate integrity status is missing")
denied = set(plugin.get("denied_capabilities", ()))
for capability in ("hidden.read", "final.read", "release.write", "physical.actuate"):
    if capability not in denied:
        raise SystemExit(f"plugin must deny {capability}")
for path in (
    sorted((root / "src").rglob("*.py"))
    + sorted((root / "examples").rglob("*.py"))
    + sorted((root / "scripts").glob("*.py"))
):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
manifest = json.loads((root / "examples/task-manifest.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(root / "src"))
from ecologyrsi_dsh import ToyCropSoilWater
expected_digest = ToyCropSoilWater(seed=0).dataset_digest
if manifest.get("metadata", {}).get("dataset_digest") != expected_digest:
    raise SystemExit("example dataset digest does not match the fixed seed-0 snapshot")
if manifest.get("metadata", {}).get("dataset_seed") != 0:
    raise SystemExit("example must distinguish dataset_seed from search seed")
print(f"source metadata and syntax: ok ({project_version})")
PY

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ecologyrsi-dsh-source-verify.XXXXXX")"
trap 'rm -rf "$TMP_ROOT"' EXIT HUP INT TERM

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT_DIR/src" \
  "$PYTHON_BIN" examples/minimal_run.py \
  --db "$TMP_ROOT/example.sqlite3" \
  --run-id "run:source-verification" > "$TMP_ROOT/example.json"

"$PYTHON_BIN" - "$TMP_ROOT/example.json" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(Path("examples/task-manifest.json").read_text(encoding="utf-8"))
expected_candidates = int(manifest["budget"]["max_candidates"])
assert payload["status"] == "completed", payload
assert payload["candidate_count"] == payload["evaluation_count"] == expected_candidates, payload
assert payload["event_count"] >= 10, payload
assert payload["best_candidate_id"], payload
print("source evolution and replay example: ok")
PY

DATA_ROOT="${ECOLOGYRSI_DATA_ROOT:-$ROOT_DIR/../EcologyRSI/data/greenhouse}"
REAL_DATA_TESTS=0
if ECOLOGYRSI_DATA_ROOT="$DATA_ROOT" PYTHONPATH="$ROOT_DIR/src" \
  "$PYTHON_BIN" - <<'PY'
from ecologyrsi_dsh.datasets import DatasetRegistry

required = {"agc_cucumber_2018", "agc_tomato_2019"}
catalog = DatasetRegistry().catalog()["datasets"]
ready = {item["dataset_id"] for item in catalog if item["readiness"]["ready"]}
raise SystemExit(0 if required <= ready else 1)
PY
then
  REAL_DATA_TESTS=1
  echo "real AGC delivery tests: enabled ($DATA_ROOT)"
else
  echo "real AGC delivery tests: skipped (required extracted files are not ready under $DATA_ROOT)"
fi

ECOLOGYRSI_DATA_ROOT="$DATA_ROOT" \
ECOLOGYRSI_TEST_REAL_DATA="$REAL_DATA_TESTS" \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT_DIR/src" \
  "$PYTHON_BIN" -m unittest discover -s tests -v

if ! command -v node >/dev/null 2>&1; then
  echo "node is required to validate the browser plugin" >&2
  exit 1
fi
find plugins/ecology_evolution -name '*.js' -exec node --check {} \;
node plugins/ecology_evolution/test/smoke.mjs
find integrations/dsh_ecology_plugin -name '*.js' -exec node --check {} \;
node integrations/dsh_ecology_plugin/test/proxy_security.mjs

if [ "$MODE" = "--artifacts" ]; then
  "$PYTHON_BIN" scripts/verify_artifacts.py dist
fi

echo "delivery verification: ok ($MODE)"
