# EcologyRSI-DSH 0.3.33 Delivery Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a replay-safe, concurrency-bounded, self-contained EcologyRSI-DSH `0.3.33` release and deploy it only on `127.0.0.1:8777/8848` without losing the current SQLite run history.

**Architecture:** Keep the existing event-sourced runtime and v3/v4 compatibility boundaries. Add small host-owned contract modules for protocol capabilities, complete-origin budgets, screening replay, run-scoped sample admission, exception traversal, and retry policy; then make the existing large orchestration modules consume those contracts. Build artifacts from one clean commit, embed the exact DSH plugin in source and delivery layouts, and validate the production database on a read-only copy before replacing the two running services.

**Tech Stack:** Python 3.10+, `unittest`, SQLite event ledger, threaded candidate/sample execution, Node.js built-in test runner, DSH npm plugin, vanilla browser plugin, `uv`, setuptools, deterministic tar archives.

**Spec:** `docs/superpowers/specs/2026-08-26-delivery-cleanup-0.3.33-design.md`

## Global Constraints

- Final public version is exactly `0.3.33` in Python metadata, browser plugin metadata, DSH package metadata, documentation, artifacts, and verification scripts.
- The browser default is 500 complete forecast origins; for the current 3-target × 3-horizon profile this freezes `samples_per_update=4500` scoring cells.
- New strict-origin runs reject scoring-cell budgets that are not divisible by `prediction_cells_per_origin`; historical manifests are replayed without rewriting.
- Per-run sample admission is exactly the frozen `sample_concurrency` value, default and maximum 8; the DSH provider gate remains the cross-run final physical cap of 8.
- `passed` in 64-origin screening remains diagnostic evidence and is not a hard finalist filter.
- v3/v6 and v4/v7 presets, `dsh-strict-origin-bundle@3`, `dsh-strict-origin-bundle@4`, and `evaluators/uncertainty.py` remain supported.
- Existing SQLite files, `.runtime` state, active run events, and `0.3.32` artifacts are never deleted during implementation.
- Only the project services on `127.0.0.1:8777` and `127.0.0.1:8848` may be replaced during deployment.
- Every behavior change follows red-green-refactor; a test must fail for the intended reason before production code changes.
- Final release requires a clean Git commit, full Python and Node verification, clean-source installation, read-only production database replay, and live web verification.

---

### Task 1: Freeze Protocol Capabilities and Complete-Origin Budgets

**Files:**
- Create: `src/ecologyrsi_dsh/core/protocols.py`
- Create: `src/ecologyrsi_dsh/core/sample_budget.py`
- Create: `tests/test_protocol_contracts.py`
- Create: `tests/test_sample_budget.py`
- Modify: `src/ecologyrsi_dsh/api/handler.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py`
- Modify: `src/ecologyrsi_dsh/api/projection.py`
- Modify: `src/ecologyrsi_dsh/core/director.py`
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py`
- Modify: `src/ecologyrsi_dsh/evaluators/sample_execution.py`
- Modify: `tests/test_runtime_integration.py`
- Modify: `tests/test_candidate_parallel_evaluation.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `is_strict_origin_protocol(value: object) -> bool`, `supports_concurrent_origins(value: object) -> bool`, `supports_two_stage_screening(value: object) -> bool`.
- Produces: `complete_origin_count(scoring_cells: int, cells_per_origin: int) -> int` and `scoring_cell_budget(origin_count: int, cells_per_origin: int) -> int`.
- Produces: new strict-run default `500 * prediction_cells_per_origin`; callers that explicitly submit a non-multiple receive `ValueError` and HTTP 400.
- Consumed by: Tasks 2 and 3 for screening and concurrent-origin gates.

- [ ] **Step 1: Write failing protocol and budget tests**

```python
class ProtocolContractTests(unittest.TestCase):
    def test_legacy_and_current_strict_protocols_keep_distinct_capabilities(self) -> None:
        self.assertTrue(is_strict_origin_protocol("dsh-strict-origin-bundle@3"))
        self.assertTrue(is_strict_origin_protocol("dsh-strict-origin-bundle@4"))
        self.assertFalse(supports_concurrent_origins("dsh-strict-origin-bundle@3"))
        self.assertTrue(supports_concurrent_origins("dsh-strict-origin-bundle@4"))
        self.assertTrue(supports_two_stage_screening("dsh-strict-origin-bundle@4"))


class SampleBudgetTests(unittest.TestCase):
    def test_complete_origin_budget_rejects_silent_truncation(self) -> None:
        self.assertEqual(complete_origin_count(4_500, 9), 500)
        with self.assertRaisesRegex(ValueError, "4,501.*9.*4,500"):
            complete_origin_count(4_501, 9)

    def test_scoring_budget_is_derived_from_complete_origins(self) -> None:
        self.assertEqual(scoring_cell_budget(500, 9), 4_500)
        self.assertEqual(scoring_cell_budget(169, 9), 1_521)
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest \
  tests.test_protocol_contracts tests.test_sample_budget -v
```

Expected: import errors for the two new contract modules.

- [ ] **Step 3: Implement the pure capability and budget contracts**

```python
# core/protocols.py
STRICT_ORIGIN_PROTOCOLS = frozenset({
    "dsh-strict-origin-bundle@3",
    "dsh-strict-origin-bundle@4",
})
CONCURRENT_ORIGIN_PROTOCOL = "dsh-strict-origin-bundle@4"

def is_strict_origin_protocol(value: object) -> bool:
    return isinstance(value, str) and value in STRICT_ORIGIN_PROTOCOLS

def supports_concurrent_origins(value: object) -> bool:
    return value == CONCURRENT_ORIGIN_PROTOCOL

def supports_two_stage_screening(value: object) -> bool:
    return value == CONCURRENT_ORIGIN_PROTOCOL
```

```python
# core/sample_budget.py
def scoring_cell_budget(origin_count: int, cells_per_origin: int) -> int:
    if isinstance(origin_count, bool) or not isinstance(origin_count, int) or origin_count < 1:
        raise ValueError("origin_count must be a positive integer")
    if isinstance(cells_per_origin, bool) or not isinstance(cells_per_origin, int) or cells_per_origin < 1:
        raise ValueError("cells_per_origin must be a positive integer")
    return origin_count * cells_per_origin

def complete_origin_count(scoring_cells: int, cells_per_origin: int) -> int:
    if isinstance(scoring_cells, bool) or not isinstance(scoring_cells, int) or scoring_cells < 1:
        raise ValueError("scoring_cells must be a positive integer")
    if isinstance(cells_per_origin, bool) or not isinstance(cells_per_origin, int) or cells_per_origin < 1:
        raise ValueError("cells_per_origin must be a positive integer")
    origin_count, remainder = divmod(scoring_cells, cells_per_origin)
    if remainder:
        nearest = origin_count * cells_per_origin
        raise ValueError(
            f"scoring-cell budget {scoring_cells:,} must be divisible by "
            f"{cells_per_origin}; nearest lower complete budget is {nearest:,}"
        )
    return origin_count
```

- [ ] **Step 4: Replace literal protocol sets and budget truncation**

Use the helpers in every production check currently spelling `{"dsh-strict-origin-bundle@3", "dsh-strict-origin-bundle@4"}` or comparing only v4 capabilities. In `_phase_task_manifest`, replace floor division with:

```python
formal_origins = complete_origin_count(formal_cells, cells_per_origin)
```

In run creation, set an omitted strict-native budget to:

```python
samples_per_update = scoring_cell_budget(500, prediction_cells_per_origin)
```

and validate every explicit strict-native value with `complete_origin_count`. Keep the non-native compatibility default isolated as `1_600`, and never recalculate a persisted task manifest during replay.

- [ ] **Step 5: Add HTTP regression coverage and make it GREEN**

Extend `test_autonomous_runtime_context_is_frozen` so the default strict run is 4,500 cells/500 origins, an explicit 4,500 request is accepted, and 4,501 returns 400. Replace the prior accepted non-multiple `321` fixture with `450` so the explicit diagnostic run remains 50 complete origins.

Run:

```bash
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest \
  tests.test_protocol_contracts tests.test_sample_budget \
  tests.test_runtime_integration tests.test_candidate_parallel_evaluation \
  tests.test_evaluation -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/ecologyrsi_dsh/core/protocols.py \
  src/ecologyrsi_dsh/core/sample_budget.py \
  src/ecologyrsi_dsh/api/handler.py \
  src/ecologyrsi_dsh/api/generation_execution.py \
  src/ecologyrsi_dsh/api/projection.py \
  src/ecologyrsi_dsh/core/director.py \
  src/ecologyrsi_dsh/evaluators/registry.py \
  src/ecologyrsi_dsh/evaluators/sample_execution.py \
  tests/test_protocol_contracts.py tests/test_sample_budget.py \
  tests/test_runtime_integration.py tests/test_candidate_parallel_evaluation.py \
  tests/test_evaluation.py
git commit -m "fix: enforce complete-origin budgets"
```

---

### Task 2: Make Two-Stage Screening a Validated Replay State Machine

**Files:**
- Create: `src/ecologyrsi_dsh/core/screening.py`
- Create: `tests/test_screening_replay.py`
- Modify: `src/ecologyrsi_dsh/core/state.py`
- Modify: `src/ecologyrsi_dsh/core/director.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py`
- Modify: `tests/test_candidate_parallel_evaluation.py`
- Modify: `tests/test_core.py`

**Interfaces:**
- Consumes: Task 1 `supports_two_stage_screening` and `complete_origin_count`.
- Produces: `screening_record_digest(payload) -> str` and `screening_cohort_digest(records) -> str`.
- Produces: `RunState.candidate_screening_events`, `RunState.formal_selection_events`, `RunState.screened_out_events`, plus generation/candidate lookup methods.
- Produces: v2 screening/formal events for new writes while accepting and validating existing v1 events.
- Consumed by: Task 3 projection and final production database replay.

- [ ] **Step 1: Write forged-event replay tests**

Create one valid strict-v4 run fixture with four spawned generation-0 candidates. Append valid screening records and a formal Top-2 event through the director, then copy and mutate event streams to assert:

```python
with self.assertRaisesRegex(ValueError, "screening generation"):
    project_run_state(events_with_wrong_generation)

with self.assertRaisesRegex(ValueError, "SHA-256"):
    project_run_state(events_with_malformed_cohort_digest)

with self.assertRaisesRegex(ValueError, "missing screening"):
    project_run_state(events_selecting_unscreened_candidate)

with self.assertRaisesRegex(ValueError, "selected candidate"):
    project_run_state(events_screening_out_selected_candidate)

with self.assertRaisesRegex(ValueError, "conflicting screening"):
    project_run_state(events_with_same_id_and_different_payload)
```

Also assert a normal replay returns the same two selected candidate ids and preserves a diagnostic `passed=False` record without filtering it.

- [ ] **Step 2: Run the replay tests and confirm RED**

Run:

```bash
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest \
  tests.test_screening_replay -v
```

Expected: forged streams are currently accepted or the new structured state fields are missing.

- [ ] **Step 3: Implement canonical screening contracts**

`core/screening.py` defines exact schemas and digest helpers. New records use:

```python
SCREENING_SCHEMA_V2 = "ecologyrsi-dsh.candidate-screening/2"
FORMAL_SELECTION_SCHEMA_V2 = "ecologyrsi-dsh.formal-selection-cohort/2"
SCREENED_OUT_SCHEMA_V1 = "ecologyrsi-dsh.candidate-screened-out/1"

def screening_record_digest(payload: Mapping[str, Any]) -> str:
    normalized = {
        name: payload[name]
        for name in (
            "generation", "candidate_id", "score", "passed",
            "constraint_violations", "origin_count",
            "prediction_cell_count", "cohort_digest",
        )
    }
    return digest(normalized)

def screening_cohort_digest(records: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(records, key=lambda item: str(item["candidate_id"]))
    return digest(ordered)
```

New v2 screening payloads include `record_digest`. Existing v1 payloads retain their original digest semantics and are validated without rewriting.

- [ ] **Step 4: Project and validate screening state**

In `project_run_state`, keep dictionaries keyed by `(generation, candidate_id)` and `generation`. Validate candidate ownership, exact 64-origin/`64 * prediction_cells_per_origin` evidence for strict-v4 selection runs, SHA-256 format, v2 record digest, formal selected count 2, formal digest, idempotency, and screened-out membership. Store the original validated `Event` objects in `RunState` so `formal_selection_event_id` remains auditable.

Add lookup methods:

```python
def screening_for(self, generation: int, candidate_id: str) -> Event | None: ...
def formal_selection_for(self, generation: int) -> Event | None: ...
```

Update `_screening_records` and `_formal_selection_event` to use those methods instead of rescanning unvalidated raw events.

- [ ] **Step 5: Verify writer idempotency and replay compatibility**

Add tests showing exact duplicate director calls replay the original event, changed payloads conflict, and v1 fixtures still replay. Run:

```bash
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest \
  tests.test_screening_replay tests.test_candidate_parallel_evaluation \
  tests.test_core -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/ecologyrsi_dsh/core/screening.py \
  src/ecologyrsi_dsh/core/state.py src/ecologyrsi_dsh/core/director.py \
  src/ecologyrsi_dsh/api/generation_execution.py \
  tests/test_screening_replay.py tests/test_candidate_parallel_evaluation.py \
  tests/test_core.py
git commit -m "fix: validate two-stage screening replay"
```

---

### Task 3: Enforce Run-Scoped Sample Admission and Honest Queue Progress

**Files:**
- Create: `src/ecologyrsi_dsh/api/sample_admission.py`
- Create: `tests/test_sample_admission.py`
- Modify: `src/ecologyrsi_dsh/api/handler.py`
- Modify: `src/ecologyrsi_dsh/api/projection.py`
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py`
- Modify: `src/ecologyrsi_dsh/evaluators/sample_execution.py`
- Modify: `tests/test_sample_execution.py`
- Modify: `tests/test_execution_projection.py`
- Modify: `plugins/ecology_evolution/assets/js/render_process.js`
- Modify: `plugins/ecology_evolution/test/smoke.mjs`

**Interfaces:**
- Consumes: Task 1 concurrent-origin capability helper.
- Produces: `RunSampleAdmission.admit(run_id: str, limit: int)` context manager and `snapshot(run_id: str) -> dict[str, int]`.
- Produces: optional `origin_admission` factory on `CollaborativeSampleExecutor`; the factory is host-only and never enters canonical execution context or request payloads.
- Produces: screening projection fields `awaiting_submission_batches`, `in_flight_batches`, `queued_batches`, `completed_samples`, `succeeded_samples`, and `failed_samples` with non-overlapping meanings.

- [ ] **Step 1: Write the concurrency RED test**

Use three worker groups sharing one `RunSampleAdmission`, set `limit=8`, block admitted bodies on an event, and start 9 calls. Assert only 8 enter before release, the ninth is locally waiting, and cancellation/error bodies release all permits:

```python
snapshot = admission.snapshot("run:test")
self.assertEqual(snapshot["active"], 8)
self.assertEqual(snapshot["waiting"], 1)
self.assertEqual(maximum_active, 8)

release.set()
for thread in threads:
    thread.join(3)
self.assertEqual(admission.snapshot("run:test"), {
    "limit": 8, "active": 0, "waiting": 0,
})
```

Add a second test for `limit=3` with eight callers and a mismatch test proving the same run cannot change its frozen limit.

- [ ] **Step 2: Run the admission tests and confirm RED**

Run:

```bash
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest \
  tests.test_sample_admission -v
```

Expected: `api.sample_admission` does not exist.

- [ ] **Step 3: Implement the run-scoped controller**

Use a lock-protected per-run state containing one `threading.BoundedSemaphore`, immutable `limit`, `active`, and `waiting`. The context manager increments waiting before blocking, transfers one count from waiting to active after acquisition, and always decrements active/releases in `finally`. `snapshot` returns zeros for an unseen run and never exposes semaphore internals.

```python
@contextmanager
def admit(self, run_id: str, limit: int) -> Iterator[None]:
    state = self._state_for(run_id, limit)
    with self._lock:
        state.waiting += 1
    state.semaphore.acquire()
    with self._lock:
        state.waiting -= 1
        state.active += 1
    try:
        yield
    finally:
        with self._lock:
            state.active -= 1
        state.semaphore.release()
```

- [ ] **Step 4: Integrate admission before each strict origin chain**

Create one controller on `EvolutionHTTPServer`. Pass a provider to `EvaluatorRegistry`, then pass `lambda: provider(run_id, raw_concurrency)` into each DSH-native `CollaborativeSampleExecutor`. Wrap `_prepare_strict_origin_bundle` in that context. The callback must not be copied into `context_data`, hashed, serialized, or sent to DSH.

Retain the DSH `ProviderStageGate(maxInFlight=8)` unchanged. This yields a run-level exact limit before the provider-wide cross-run limit.

- [ ] **Step 5: Correct progress semantics and write GREEN tests**

For screening projection, calculate:

```python
completed = len(completed_reflections)
failed = sum(
    event.payload.get("structured", {}).get("outcome_class") == "failed"
    for event in completed_reflections.values()
)
succeeded = sum(
    event.payload.get("structured", {}).get("outcome_class")
    in {"improved", "degraded", "neutral"}
    for event in completed_reflections.values()
)
in_flight = min(outstanding, configured_concurrency or 8)
provider_queued = max(0, outstanding - in_flight)
awaiting_submission = max(0, total - completed - outstanding)
```

Unknown reflection outcomes count only as completed. Set `queued_batches=provider_queued`, add `awaiting_submission_batches`, and change `queue_semantics` to `provider_admission_only`. Update the web renderer to label not-yet-submitted work as “待提交” rather than “排队”.

Run:

```bash
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest \
  tests.test_sample_admission tests.test_sample_execution \
  tests.test_execution_projection -v
node plugins/ecology_evolution/test/smoke.mjs
```

Expected: all tests pass; the fixture with one failed reflection reports completed 1, succeeded 0, failed 1, provider queued 0, and awaiting submission 253.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/ecologyrsi_dsh/api/sample_admission.py \
  src/ecologyrsi_dsh/api/handler.py src/ecologyrsi_dsh/api/projection.py \
  src/ecologyrsi_dsh/evaluators/registry.py \
  src/ecologyrsi_dsh/evaluators/sample_execution.py \
  tests/test_sample_admission.py tests/test_sample_execution.py \
  tests/test_execution_projection.py \
  plugins/ecology_evolution/assets/js/render_process.js \
  plugins/ecology_evolution/test/smoke.mjs
git commit -m "fix: bound run sample admission"
```

---

### Task 4: Remove Proven Duplicate Core Logic and Python Compatibility Debt

**Files:**
- Create: `src/ecologyrsi_dsh/core/retry_policy.py`
- Create: `tests/test_exception_graph.py`
- Create: `tests/test_retry_policy.py`
- Modify: `src/ecologyrsi_dsh/core/errors.py`
- Modify: `src/ecologyrsi_dsh/core/director.py`
- Modify: `src/ecologyrsi_dsh/core/state.py`
- Modify: `src/ecologyrsi_dsh/integrations/model_gateway.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py`
- Modify: `src/ecologyrsi_dsh/api/auto_progress.py`
- Modify: `src/ecologyrsi_dsh/data/preparation.py`
- Modify: `tests/test_dsh_tool_contracts.py`
- Modify: `tests/test_autonomous_search_reflection_cycle.py`
- Delete: `.superpowers/sdd/2026-08-25-evolution-safety-followup/task-2-report.md`
- Delete: `.superpowers/sdd/2026-08-25-evolution-safety-followup/task-3-report.md`
- Delete: `.superpowers/sdd/2026-08-25-evolution-safety-followup/task-4-report.md`
- Delete: `.superpowers/sdd/2026-08-25-evolution-safety-followup/task-5-report.md`

**Interfaces:**
- Produces: `walk_exception_graph(exc, max_depth=32, max_nodes=256)` and `find_exception(exc, expected_type, predicate=None)`.
- Produces: `GatewayRetryPolicy` values and `retry_policy(retry_class: str) -> GatewayRetryPolicy` shared by director and replay state.
- Preserves: every existing public retry code, action, message, epoch, delay, and v2 event schema.

- [ ] **Step 1: Write exception graph and retry-policy RED tests**

```python
def test_exception_graph_is_cycle_safe_and_reads_groups(self) -> None:
    leaf = MarkerError("leaf")
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = leaf
    wrapper.__context__ = wrapper
    grouped = ExceptionGroup("group", [wrapper, ValueError("peer")])
    self.assertIs(find_exception(grouped, MarkerError), leaf)
    self.assertLessEqual(len(tuple(walk_exception_graph(grouped))), 4)

def test_retry_policy_keeps_public_contract(self) -> None:
    policy = retry_policy("dsh_native_runtime")
    self.assertEqual(policy.circuit_code, "dsh_runtime_retry_circuit_open")
    self.assertEqual(policy.suggested_action, "check_dsh_runtime_then_resume")
```

- [ ] **Step 2: Run new tests and confirm RED**

Run:

```bash
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest \
  tests.test_exception_graph tests.test_retry_policy -v
```

Expected: missing public helpers/modules.

- [ ] **Step 3: Implement and consume shared exception traversal**

`walk_exception_graph` yields each exception once, follows cause before context, follows `exceptions` for Python 3.11 groups without importing `ExceptionGroup` on Python 3.10, and enforces both depth and node bounds. Replace local stacks in model gateway, generation execution, auto progress, and DSH error lookup with `find_exception`/`walk_exception_graph`.

- [ ] **Step 4: Move retry policy constants into a dependency-free module**

Use an immutable dataclass:

```python
@dataclass(frozen=True, slots=True)
class GatewayRetryPolicy:
    circuit_code: str
    suggested_action: str
    public_reason: str
```

Export schema version, classes, policy map, circuit codes, epoch seconds, and maximum delay. Both writer and replay import from `core.retry_policy`; neither imports the other.

- [ ] **Step 5: Fix Python 3.10 source compatibility and MD5 intent**

Move dict literals used inside multiline f-strings in `tests/test_dsh_tool_contracts.py` into precomputed JSON strings. Remove the unused `batch` local in `tests/test_autonomous_search_reflection_cycle.py`. Change upstream checksum calls to:

```python
hashlib.md5(usedforsecurity=False)
```

Run the exact compatibility and security gates:

```bash
uv run --python 3.10 --no-project python -m compileall -q src tests scripts
uvx ruff check src tests scripts --select F
uvx bandit -r src scripts -ll
```

Expected: compileall and Ruff exit 0; Bandit reports no unexplained high-severity item.

- [ ] **Step 6: Delete only proven tracked SDD reports and run regression tests**

Delete the four exact tracked report paths listed above; do not remove `.runtime`, another plan's ledger, or broad workspace directories.

Run:

```bash
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest \
  tests.test_exception_graph tests.test_retry_policy \
  tests.test_gateway_retry_circuit tests.test_auto_progress \
  tests.test_dsh_tool_contracts -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add -u .superpowers/sdd/2026-08-25-evolution-safety-followup \
  src tests
git add src/ecologyrsi_dsh/core/retry_policy.py \
  tests/test_exception_graph.py tests/test_retry_policy.py
git commit -m "refactor: centralize runtime contracts"
```

---

### Task 5: Make Source and Release Artifacts Self-Contained

**Files:**
- Create: `scripts/build_dsh_plugin.py`
- Modify: `.gitignore`
- Modify: `MANIFEST.in`
- Modify: `Makefile`
- Modify: `pyproject.toml`
- Modify: `scripts/build_delivery.sh`
- Modify: `scripts/create_delivery_archive.py`
- Modify: `scripts/verify_delivery.sh`
- Modify: `scripts/verify_artifacts.py`
- Modify: `integrations/dsh_ecology_plugin/package.json`
- Modify: `tests/test_delivery_scripts.py`

**Interfaces:**
- Produces: `build_dsh_plugin.py --root ROOT --output-dir DIR`, packing a temporary staging tree with root LICENSE/NOTICE.
- Produces: `verify_delivery.sh --artifacts-only`, which never repeats the source suite.
- Produces: public source allowlist containing screenshots but excluding `docs/superpowers/**` and `docs/项目整体Review与方案B优化报告.md`.
- Produces: external and embedded `BUILD-INFO.json` with version, commit, `dirty=false`, source epoch, Python, Node, npm, and uv versions.

- [ ] **Step 1: Write failing source/delivery contract tests**

Add behavior tests that call `included_source_files(ROOT)` and assert:

```python
self.assertIn(
    "integrations/dsh_ecology_plugin/dist/"
    "ecologyrsi-dsh-evolution-plugin-0.3.33.tgz",
    included,
)
self.assertFalse(any(name.startswith("docs/superpowers/") for name in included))
self.assertNotIn("docs/项目整体Review与方案B优化报告.md", included)
```

Add a temporary fake dist test for `create_archive` asserting the nested plugin path, `artifacts/` copy, `BUILD-INFO.json`, and internal checksums exist. Add a test that executes `bash scripts/verify_delivery.sh --help` or an isolated argument parser path and proves `--artifacts-only` is accepted independently.

- [ ] **Step 2: Run delivery-script tests and confirm RED**

Run:

```bash
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest \
  tests.test_delivery_scripts -v
```

Expected: internal docs are included, current nested tgz is excluded, or artifact-only mode is absent.

- [ ] **Step 3: Implement explicit public source selection**

Replace recursive `docs` inclusion with `docs/screenshots`. Keep the exact versioned nested DSH tgz even though one path component is `dist`. Update MANIFEST and wheel data files to include the current tgz and remove `plugins/ecology_evolution/test/smoke.mjs` from wheel data only; the smoke file remains in sdist/delivery source tests.

Use a generic `.gitignore` exception for `ecologyrsi-dsh-evolution-plugin-*.tgz`; verification still rejects zero, multiple, stale, or wrong-version tgz files.

- [ ] **Step 4: Build the DSH package through a legal-file staging tree**

`build_dsh_plugin.py` copies package.json plus the paths selected by npm's `files` contract to a temporary directory, copies root `LICENSE` and `NOTICE`, and runs `npm pack STAGING --pack-destination OUTPUT`. Extend package `files` with `LICENSE` and `NOTICE`. `verify_npm_plugin` checks both members and byte equality with root files.

- [ ] **Step 5: Split source and artifact verification and bind build provenance**

Accept exactly `--source-only`, `--artifacts`, and `--artifacts-only`; keep `--artifacts` as compatibility shorthand for source plus artifacts. `build_delivery.sh` runs source once, builds artifacts, then runs artifact-only once.

Before final release, reject a dirty worktree unless `ECOLOGYRSI_ALLOW_DIRTY_BUILD=1`. Generate:

```json
{
  "schema_version": "ecologyrsi-dsh.build-info/1",
  "version": "0.3.33",
  "commit": "<40 lowercase hex characters>",
  "dirty": false,
  "source_date_epoch": 0,
  "tools": {"python": "...", "node": "...", "npm": "...", "uv": "..."}
}
```

The implementation substitutes the real command outputs for the shown tool values and validates their types and bounds.

- [ ] **Step 6: Run package gates and commit source changes**

Run:

```bash
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest \
  tests.test_delivery_scripts -v
node --test integrations/dsh_ecology_plugin/test/*.test.mjs \
  integrations/dsh_ecology_plugin/test/proxy_security.mjs
node plugins/ecology_evolution/test/smoke.mjs
```

Expected: Python delivery tests, all Node tests, and browser smoke pass.

```bash
git add .gitignore MANIFEST.in Makefile pyproject.toml \
  scripts/build_dsh_plugin.py scripts/build_delivery.sh \
  scripts/create_delivery_archive.py scripts/verify_delivery.sh \
  scripts/verify_artifacts.py integrations/dsh_ecology_plugin/package.json \
  tests/test_delivery_scripts.py
git commit -m "build: make delivery artifacts self-contained"
```

---

### Task 6: Align Version, Documentation, Browser Copy, and Screenshots

**Files:**
- Modify: `src/ecologyrsi_dsh/version.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `NOTICE`
- Modify: `CHANGELOG.md`
- Modify: `README.md`
- Modify: `RELEASE-CHECKLIST.md`
- Modify: `plugins/ecology_evolution/plugin.json`
- Modify: `plugins/ecology_evolution/index.html`
- Modify: `plugins/ecology_evolution/assets/js/host.js`
- Modify: `plugins/ecology_evolution/README.md`
- Modify: `plugins/ecology_evolution/test/smoke.mjs`
- Modify: `integrations/dsh_ecology_plugin/package.json`
- Modify: `integrations/dsh_ecology_plugin/README.md`
- Modify: `MANIFEST.in`
- Modify: `scripts/verify_delivery.sh`
- Modify: `docs/screenshots/01-run-settings.jpg`
- Modify: `docs/screenshots/02-parameter-design.jpg`
- Modify: `docs/screenshots/03-training-data.jpg`
- Modify: `docs/screenshots/04-evolution-process.jpg`
- Modify: `docs/screenshots/05-candidate-evaluation.jpg`
- Modify: `docs/screenshots/06-human-governance.jpg`

**Interfaces:**
- Consumes: Tasks 1–5 runtime terminology and artifact layout.
- Produces: every active version reference at `0.3.33` and current operational documentation.
- Produces: screenshots captured from the deployed `8777/8848` UI after smoke data is available.

- [ ] **Step 1: Add/adjust executable documentation assertions**

Update smoke tests to assert `0.3.33`, default 500 complete origins, 4,500 scoring cells, candidate concurrency 4, sample concurrency 8, provider queue versus awaiting-submission labels, and twelve installed preset ids. Update delivery tests so the HTML defaults and current plugin manifest are behavioral sources of truth; do not grep human prose as a substitute for runtime tests.

- [ ] **Step 2: Run smoke/version tests and confirm RED**

Run:

```bash
node plugins/ecology_evolution/test/smoke.mjs
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest \
  tests.test_delivery_scripts tests.test_runtime_integration -v
```

Expected: version assertions still observe `0.3.32` or old documentation/default fixtures.

- [ ] **Step 3: Bump all active version locations to 0.3.33**

Update Python, browser, DSH package, NOTICE, manifest, wheel data path, verifier path, and lockfile. Add a `0.3.33 - 2026-08-26` CHANGELOG entry grouped into runtime correctness, replay safety, cleanup, delivery, and deployment.

- [ ] **Step 4: Replace stale operational wording**

Document exactly:

```text
默认每次更新：500 个完整预测时点 / 4,500 个评分单元
筛选：每候选 64 个完整预测时点
正式评估：Top 2，各 500 个完整预测时点
候选并发：4
逐样本并发：8
同 provider 的 DSH stage 全局在飞上限：8
```

Remove active instructions claiming concurrency 2, 1,600 per round, six v2 presets, provider serialization, or a UI token cap. Historical changelog entries retain their original version facts.

- [ ] **Step 5: Run text, browser, and version gates**

Run:

```bash
node plugins/ecology_evolution/test/smoke.mjs
node --test integrations/dsh_ecology_plugin/test/*.test.mjs \
  integrations/dsh_ecology_plugin/test/proxy_security.mjs
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest \
  tests.test_delivery_scripts tests.test_runtime_integration -v
```

Expected: all pass and active documentation has no stale operational defaults.

- [ ] **Step 6: Build and track the exact plugin tgz, then commit Task 6**

Run the DSH package builder, ensure exactly one `0.3.33` tgz exists, and stage it with the source/version changes:

```bash
uv run --no-project python scripts/build_dsh_plugin.py \
  --root "$PWD" \
  --output-dir "$PWD/integrations/dsh_ecology_plugin/dist"
git add src/ecologyrsi_dsh/version.py pyproject.toml uv.lock NOTICE CHANGELOG.md \
  README.md RELEASE-CHECKLIST.md MANIFEST.in scripts/verify_delivery.sh \
  plugins/ecology_evolution integrations/dsh_ecology_plugin \
  docs/screenshots
git commit -m "release: prepare ecologyrsi dsh 0.3.33"
```

Screenshots may remain unchanged in this commit only if their visible values already match the executable UI; otherwise capture them during Task 7 and amend with a separate documentation commit.

---

### Task 7: Verify Production Replay, Build Final Artifacts, and Deploy 8777/8848

**Files:**
- Generate: `dist/ecologyrsi_dsh-0.3.33-py3-none-any.whl`
- Generate: `dist/ecologyrsi_dsh-0.3.33.tar.gz`
- Generate: `dist/ecologyrsi-dsh-evolution-plugin-0.3.33.tgz`
- Generate: `dist/ecologyrsi-dsh-0.3.33-delivery.tar.gz`
- Generate: `dist/SHA256SUMS`
- Generate: `dist/BUILD-INFO.json`
- Possibly modify: `docs/screenshots/*.jpg` after live capture

**Interfaces:**
- Consumes: all source tasks and the existing production SQLite path.
- Produces: final clean artifacts and live services on only ports 8777/8848.
- Preserves: original database bytes except for normal post-resume event appends by the running application.

- [ ] **Step 1: Run the full source gate on a clean commit**

```bash
git diff --quiet
git diff --cached --quiet
PYTHON="$(uv python find --no-project --system '>=3.10')" \
  ./scripts/verify_delivery.sh --source-only
```

Expected: exit 0; Python suite reports no failures, all Node tests pass, browser smoke passes, Python 3.10 compile and Ruff gates pass.

- [ ] **Step 2: Replay a read-only copy of the production database**

Record the active database path and latest sequence, copy it to a `mktemp -d` directory, and run a script that opens every run id through `EvolutionDirector(...).state(run_id)`. Assert the current run `run:75f6393f-c4f2-41c6-93c8-d47400080093` replays, its latest sequence is not lower than the recorded value, and its generation/candidate counts match the source database query.

The script must never open the production path for writes.

- [ ] **Step 3: Build and verify final artifacts**

```bash
SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" make release
make verify-artifacts
```

Expected: exactly the six Task 7 deliverables exist; artifact verification installs wheel/sdist/delivery in isolation, installs the DSH runtime, checks LICENSE/NOTICE, and validates checksums and build provenance.

- [ ] **Step 4: Record deployment boundary and pause safely**

Record PIDs listening on 8777/8848, health payloads, active run id, latest event sequence, and auto-progress state. Pause automatic progression through the existing control endpoint and wait until DSH in-flight work reaches its safe boundary. Do not archive, purge, or delete the run.

- [ ] **Step 5: Replace only 8777/8848 with final artifacts**

Stop only the processes positively identified as this project's 8777/8848 services. Start the Python backend from the final wheel against the existing database and start the DSH/browser host from the final plugin on 8848. Do not start other project ports.

- [ ] **Step 6: Verify live health, replay, queue semantics, and one origin wave**

Verify:

```text
GET http://127.0.0.1:8777/api/health -> ok=true, package_version=0.3.33
GET http://127.0.0.1:8848/ -> HTTP 200
current run remains selectable and has the same run id
event sequence never decreases
provider in-flight never exceeds 8
unsubmitted origins are displayed as 待提交, not provider 排队
one complete origin reaches plan, critic, score, and reflect
```

Resume auto-progress only after these checks pass. Observe at least one complete origin wave and confirm no duplicate sample ids or conflicting event ids.

- [ ] **Step 7: Capture final screenshots if visible values changed**

Use the live browser to capture the six README views at their existing dimensions and replace only the exact six screenshot files. Re-run the browser smoke and delivery-script tests, then commit screenshot changes separately:

```bash
git add docs/screenshots README.md
git commit -m "docs: refresh 0.3.33 workbench screenshots"
```

- [ ] **Step 8: Run final verification and record evidence**

```bash
git diff --check
PYTHONPATH="$PWD/src" uv run --no-project python -m unittest discover -s tests -v
node --test integrations/dsh_ecology_plugin/test/*.test.mjs \
  integrations/dsh_ecology_plugin/test/proxy_security.mjs
node plugins/ecology_evolution/test/smoke.mjs
make verify-artifacts
```

Expected: all commands exit 0. Record test counts, skipped tests, artifact SHA-256 values, production replay result, live PIDs, URLs, current event sequence, and the observed origin-wave result in the final handoff.
