# Reward and Scoring Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the Python evaluator and deliver a shared, baseline-aware,
coverage-safe scoring kernel with reliable same-cohort promotion decisions.

**Architecture:** Move deterministic greenhouse objective and baseline logic
into focused pure modules while keeping `EvaluatorRegistry` as the public
orchestrator. Persist additive versioned evidence in existing JSON metrics and
route every promotion path through one compatibility-aware decision helper.

**Tech Stack:** Python 3.10+ standard library, `unittest`, append-only JSON
events in SQLite, existing Node.js smoke tests.

**Spec:** `docs/superpowers/specs/2026-08-20-reward-scoring-reliability-design.md`

## Global Constraints

- Use `training_fit` only for scale and baseline selection.
- Use `training_feedback` only for visible candidate scoring.
- Never read `development`, `gate`, `external_holdout`, hidden, test, or final
  labels in the evolution loop.
- Keep persistence as the predictor's residual reference; apply a scoring
  baseline only after candidate prediction.
- Keep target weights exactly `1/3` each and horizon weights exactly `1/H`.
- Keep normalized reward, skill, and missing-evidence penalties in `[-1, 1]`.
- Use missing penalty `-1.0`, minimum passing score `1e-9`, no-regression
  tolerance `1e-12`, and practical promotion delta `0.005`.
- Use deterministic 24-hour paired blocks, 1,000 bootstrap resamples, and a
  95% percentile interval when at least four paired blocks exist.
- Add fields without removing established API fields; old reward archives and
  legacy evaluations must remain replayable.
- Bump evaluator implementation and objective aggregation versions whenever
  new-run score semantics change.
- Add no third-party runtime dependency and no SQLite schema migration.
- All production behavior changes follow RED → GREEN → REFACTOR; run the
  specified failing test before implementation.

---

### Task 1: Restore source validity and establish a clean baseline

**Files:**
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py:2970-2995`
- Test: `scripts/verify_delivery.sh`

**Interfaces:**
- Consumes: the existing `evaluation_metrics` mapping construction.
- Produces: an importable evaluator module with one
  `mean_raw_normalized_reward` field and unchanged intended values.

- [ ] **Step 1: Re-run the existing failing syntax verification**

Run:

```bash
LC_ALL=en_US.UTF-8 "$PYTHON" -S -c '
from pathlib import Path
bad = []
for path in list(Path("src").rglob("*.py")) + list(Path("tests").rglob("*.py")):
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except Exception as exc:
        bad.append((str(path), str(exc)))
print(bad)
raise SystemExit(bool(bad))
'
```

Expected: exit 1 naming `registry.py` near line 2985.

- [ ] **Step 2: Apply the minimal syntax repair**

The metrics mapping must contain exactly these two aggregate fields:

```python
"mean_normalized_reward": (
    fmean(normalized_mean_rewards) if normalized_mean_rewards else None
),
"mean_raw_normalized_reward": (
    fmean(raw_normalized_mean_rewards)
    if raw_normalized_mean_rewards
    else None
),
```

Remove the malformed duplicate conditional and duplicate dictionary key; do
not change scoring semantics in this task.

- [ ] **Step 3: Verify parsing, import, and test collection**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest --collect-only -q
```

Expected: exit 0 and a non-zero test count.

- [ ] **Step 4: Run the focused existing evaluation tests**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest tests.test_evaluation -v
```

Expected: all tests pass. If they expose an independent pre-existing failure,
record it before proceeding and diagnose it with systematic debugging.

### Task 2: Extract and test the shared objective kernel

**Files:**
- Create: `src/ecologyrsi_dsh/evaluators/objectives.py`
- Create: `tests/test_objectives.py`
- Modify: `src/ecologyrsi_dsh/evaluators/metrics.py`
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py`

**Interfaces:**
- Consumes: per-cell dictionaries with `target`, `horizon_hours`, `n`,
  `eligible_rows`, `skill_score`, `normalized_mean_reward`, and optional
  `objective_quality`.
- Produces:
  - `normalized_absolute_error_reward(baseline_errors, candidate_errors, scale) -> tuple[float, float]`
  - `skill_score(candidate_nrmse, baseline_nrmse) -> float`
  - `aggregate_greenhouse_objective(task_results, horizons, *, target_weights, missing_skill_penalty=-1.0, missing_reward_penalty=-1.0) -> dict[str, Any]`
  - compatibility wrappers `_skill_score`, `_clip_normalized_objective`, and
    `_normalized_absolute_error_reward` from `metrics.py`.

- [ ] **Step 1: Write failing objective behavior tests**

Create literal, hand-checked tests including:

```python
class ObjectiveKernelTests(unittest.TestCase):
    def test_reward_sign_and_bounds(self) -> None:
        raw, bounded = normalized_absolute_error_reward([2.0], [1.0], 2.0)
        self.assertEqual(raw, 0.5)
        self.assertEqual(bounded, 0.5)
        raw, bounded = normalized_absolute_error_reward([1.0], [5.0], 2.0)
        self.assertEqual(raw, -2.0)
        self.assertEqual(bounded, -1.0)

    def test_partial_coverage_is_penalized_once(self) -> None:
        rows = [
            {
                "target": target,
                "horizon_hours": 1,
                "n": 8,
                "eligible_rows": 10,
                "skill_score": 0.5,
                "normalized_mean_reward": 0.25,
                "objective_quality": 0.8,
            }
            for target in ("air_temperature", "relative_humidity", "co2_concentration")
        ]
        result = aggregate_greenhouse_objective(rows, (1,), target_weights=DEFAULT_TARGET_WEIGHTS)
        self.assertAlmostEqual(result["weighted_skill_score"], 0.2)
        self.assertAlmostEqual(result["weighted_normalized_mean_reward"], 0.0)
        self.assertAlmostEqual(result["objective_effective_weight_coverage"], 0.8)

    def test_missing_cell_keeps_its_denominator_weight(self) -> None:
        rows = [
            {
                "target": "air_temperature",
                "horizon_hours": 1,
                "n": 1,
                "eligible_rows": 1,
                "skill_score": 1.0,
                "normalized_mean_reward": 1.0,
            }
        ]
        result = aggregate_greenhouse_objective(rows, (1,), target_weights=DEFAULT_TARGET_WEIGHTS)
        self.assertAlmostEqual(result["weighted_skill_score"], -1.0 / 3.0)
        self.assertEqual(result["objective_missing_task_count"], 2)
```

Also test duplicate cell rejection, unknown target rejection, invalid target
weights, duplicate horizons, row-order invariance, and unit-rescaling
invariance of `skill_score`.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest tests.test_objectives -v
```

Expected: import failure because `evaluators.objectives` does not exist.

- [ ] **Step 3: Implement the pure kernel**

Move the established bounded formulas from `metrics.py` and the aggregation
loop from `registry.py` into `objectives.py`. Define these exact constants:

```python
OBJECTIVE_COMPONENT_BOUND = 1.0
OBJECTIVE_MISSING_PENALTY = -1.0
DEFAULT_TARGET_WEIGHTS = {
    "air_temperature": 1 / 3,
    "relative_humidity": 1 / 3,
    "co2_concentration": 1 / 3,
}
OBJECTIVE_AGGREGATION_VERSION = "weighted_task_skill_reward@2"
```

Reject duplicate rows rather than allowing the last row to win. Derive
quality from explicit `objective_quality` when finite, otherwise from
`n / eligible_rows` when both are valid, otherwise use `1.0` for a usable
legacy row. Clamp quality to `[0, 1]`.

- [ ] **Step 4: Integrate both evaluator paths**

Keep the existing private function name in `registry.py` as a thin wrapper for
compatibility with tests/imports, but make it call
`aggregate_greenhouse_objective`. Make `metrics.py` wrappers delegate to the
new pure functions so existing imports continue to work.

Change no hard-gate threshold in this task.

- [ ] **Step 5: Run focused and integration tests**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest \
  tests.test_objectives tests.test_evaluation tests.test_horizon_feedback -v
```

Expected: all tests pass.

### Task 3: Add fit-selected causal scoring baselines and reward v2

**Files:**
- Create: `src/ecologyrsi_dsh/evaluators/baselines.py`
- Create: `tests/test_baselines.py`
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py`
- Modify: `src/ecologyrsi_dsh/core/sample_results.py`
- Modify: `tests/test_sample_results_contract.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `DatasetSeries`, registered target names, registered horizons,
  `training_fit`, and completed candidate scoring rows.
- Produces:
  - `BaselineCellSelection(target, horizon_hours, baseline_id, fit_rmse, persistence_rmse, seasonal_24h_rmse, fit_count)`
  - `BaselineProfile(schema_version, cells, profile_digest)`
  - `select_baseline_profile(series, targets, horizons) -> BaselineProfile`
  - `apply_scoring_baselines(rows, series, profile, normalization_scales) -> tuple[dict[str, Any], ...]`

- [ ] **Step 1: Write failing baseline tests**

Use a deterministic hourly `DatasetSeries` fixture. Hand-check these behaviors:

```python
class BaselineSelectionTests(unittest.TestCase):
    def test_seasonal_baseline_uses_only_values_available_at_origin(self) -> None:
        profile = select_baseline_profile(series, ("air_temperature",), (1, 6, 24))
        rows = apply_scoring_baselines(feedback_rows, series, profile, scales)
        for row in rows:
            if row["baseline_id"] == "seasonal_24h":
                self.assertLessEqual(row["target_timestamp"] - 24, row["origin_timestamp"])

    def test_tie_prefers_persistence(self) -> None:
        profile = select_baseline_profile(constant_series, ("air_temperature",), (1,))
        self.assertEqual(profile.cell("air_temperature", 1).baseline_id, "persistence")

    def test_scoring_baseline_does_not_replace_model_reference(self) -> None:
        row = apply_scoring_baselines(rows, series, profile, scales)[0]
        self.assertEqual(row["model_reference_baseline"], rows[0]["baseline"])
```

Also test fit-only selection by changing feedback labels without changing the
profile digest, deterministic digest under row order, seasonal-history
fallback, and failed-row reward non-positivity.

- [ ] **Step 2: Run baseline tests and verify RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest tests.test_baselines -v
```

Expected: import failure because `evaluators.baselines` does not exist.

- [ ] **Step 3: Implement baseline selection**

Use exact timestamp lookup. For each fit label timestamp `u` and horizon `h`:

```text
origin = u - h
persistence prediction = y(origin)
seasonal prediction = y(u - 24)
```

Accept a candidate only when its value and the observed fit label are finite
and its source timestamp is no later than the origin. Compare candidate RMSE
on the intersection of rows where both baselines exist. Choose seasonal only
when its RMSE is lower by more than `1e-12`; otherwise choose persistence.
Use schema version `ecologyrsi-dsh.baseline-profile/1` and digest the canonical
cell list plus dataset and split digests.

- [ ] **Step 4: Apply the profile after candidate execution**

In both greenhouse evaluator paths:

1. compute the profile before feedback scoring;
2. leave generated/sample request `baseline` unchanged during prediction;
3. after sample execution, copy it to `model_reference_baseline`;
4. replace scoring `baseline` with the selected comparator;
5. add `baseline_id`, `baseline_fallback`, `baseline_profile_digest`, and
   `normalization_scale`;
6. if a failed row's fallback would beat the scoring baseline on that row,
   replace its displayed fallback with the scoring baseline so reward is zero,
   while the objective coverage penalty still applies.

Persist the profile and its digest in evaluation and artifact metrics.

- [ ] **Step 5: Version and extend sample reward archives**

Define:

```python
SAMPLE_REWARD_DEFINITION_V1 = "absolute_error_improvement_vs_persistence@1"
SAMPLE_REWARD_DEFINITION_V2 = "absolute_error_improvement_vs_fit_selected_baseline@2"
SAMPLE_REWARD_DEFINITION = SAMPLE_REWARD_DEFINITION_V2
SUPPORTED_SAMPLE_REWARD_DEFINITIONS = frozenset({
    SAMPLE_REWARD_DEFINITION_V1,
    SAMPLE_REWARD_DEFINITION_V2,
})
```

Accept both definitions while decoding. New projected rows add optional
`model_reference_baseline`, `baseline_id`, `baseline_profile_digest`,
`normalization_scale`, and
`normalized_reward = clip(reward / normalization_scale, -1, 1)`. Old rows
without these fields rebuild byte-for-byte under the old archive contract.

- [ ] **Step 6: Verify baseline, reward, replay, and evaluator integration**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest \
  tests.test_baselines tests.test_sample_results_contract \
  tests.test_evaluation tests.test_recovery_security -v
```

Expected: all tests pass.

### Task 4: Add compatibility-aware practical and paired promotion gates

**Files:**
- Create: `src/ecologyrsi_dsh/evolution/promotion.py`
- Create: `tests/test_promotion_reliability.py`
- Modify: `src/ecologyrsi_dsh/evaluators/objectives.py`
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py`
- Modify: `src/ecologyrsi_dsh/core/director.py`
- Modify: `src/ecologyrsi_dsh/evolution/analysis.py`
- Modify: `tests/test_director_invariants.py`
- Modify: `tests/test_evolution_feedback_loop.py`

**Interfaces:**
- Consumes: same-cohort candidate/incumbent evaluations containing compatible
  evaluator digest, objective aggregation version, baseline profile digest,
  and bounded block evidence.
- Produces:
  - `build_block_evidence(scoring_rows, normalization_scales, *, block_hours=24) -> dict[str, Any]`
  - `paired_block_bootstrap(candidate_evidence, incumbent_evidence, *, resamples=1000, seed_material) -> dict[str, Any]`
  - `promotion_comparison(task, candidate_evaluation, incumbent_evaluation) -> PromotionComparison`
  - `PromotionComparison.comparable`, `.improves`, `.point_delta`,
    `.minimum_delta`, `.confidence_status`, `.lower_bound`, and `.reason_code`.

- [ ] **Step 1: Write failing promotion reliability tests**

Cover these literal cases:

```python
class PromotionReliabilityTests(unittest.TestCase):
    def test_new_objective_requires_practical_delta(self) -> None:
        comparison = promotion_comparison(task, evaluation(0.104), evaluation(0.100))
        self.assertTrue(comparison.comparable)
        self.assertFalse(comparison.improves)
        self.assertEqual(comparison.reason_code, "below_minimum_practical_delta")

    def test_legacy_objective_keeps_legacy_tolerance(self) -> None:
        comparison = promotion_comparison(legacy_task, legacy_evaluation(0.1001), legacy_evaluation(0.1))
        self.assertTrue(comparison.improves)

    def test_non_positive_paired_lower_bound_blocks_promotion(self) -> None:
        comparison = promotion_comparison(task, uncertain_candidate, incumbent)
        self.assertFalse(comparison.improves)
        self.assertEqual(comparison.reason_code, "paired_confidence_not_positive")
```

Also test positive confidence, fewer than four blocks, mismatched block IDs,
different cohort, different evaluator digest, different objective version,
different baseline digest, deterministic resampling, and parity between direct
and batch decisions.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest tests.test_promotion_reliability -v
```

Expected: import failure because `evolution.promotion` does not exist.

- [ ] **Step 3: Build bounded block evidence**

Group completed scoring rows by `origin_timestamp // 24` and target/horizon.
Each block cell stores only counts and normalized sufficient statistics:

```python
{
    "target": target,
    "horizon_hours": horizon,
    "eligible": eligible_count,
    "succeeded": succeeded_count,
    "candidate_squared_error_sum": sum((p - y) ** 2 / scale ** 2),
    "baseline_squared_error_sum": sum((b - y) ** 2 / scale ** 2),
    "normalized_reward_sum": sum(clipped_normalized_rewards),
}
```

Bound evidence to 128 ordered blocks and include schema version, block hours,
objective version, target weights, horizons, and a digest. Do not include raw
values, predictions, sample IDs, or unbounded timestamps in remote contexts.

- [ ] **Step 4: Implement deterministic paired bootstrap**

Require identical block identities and objective configuration. Resample the
ordered blocks with replacement using `random.Random` seeded from the SHA-256
digest of `seed_material`. Reconstruct each evaluation's objective score from
the resampled sufficient statistics and collect 1,000 paired deltas. Use
sorted indices `int(0.025 * (n - 1))` and `int(0.975 * (n - 1))` for the
percentile interval.

With fewer than four paired blocks, return `insufficient_blocks` without a
confidence interval. Invalid/mismatched evidence returns an explicit
incompatibility code and never an unpaired interval.

- [ ] **Step 5: Implement the shared promotion comparison**

Legacy evaluations without `weighted_task_skill_reward@2` keep the existing
`1e-12` tolerance. New evaluations require:

```text
same verified cohort
same evaluator digest
same objective aggregation version
same baseline profile digest
candidate score > incumbent score + 0.005
paired lower bound > 0 when confidence status is completed
```

Use this helper in `Director.decide_promotion`,
`Director.evaluate_and_decide`, and `build_generation_analysis`. Include its
reason code and bounded summary in selection reasoning without changing the
meaning of `Evaluation.passed`.

- [ ] **Step 6: Prove no private evidence reaches model contexts**

Extend existing context/redaction tests so `objective_block_evidence`, block
identities, squared-error sums, and bootstrap seed material are absent from
strategy and judge payloads.

- [ ] **Step 7: Run promotion and evolution tests**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest \
  tests.test_promotion_reliability tests.test_director_invariants \
  tests.test_evolution_feedback_loop tests.test_strategy_router \
  tests.test_judge_persistence -v
```

Expected: all tests pass.

### Task 5: Version contracts, update documentation, and verify delivery

**Files:**
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `RELEASE-CHECKLIST.md`
- Modify: `RELEASE-CHECKLIST.md`
- Test: `scripts/verify_delivery.sh`

**Interfaces:**
- Consumes: final objective, baseline, reward, and promotion constants.
- Produces: matching runtime catalog/digests, user documentation, and a source
  delivery that passes all local checks.

- [ ] **Step 1: Add contract regression tests before version edits**

Extend catalog/runtime tests to assert that a changed objective implementation
changes evaluator configuration digest and that the objective profile exposes:

```python
{
    "minimum_practical_score_delta": 0.005,
    "confidence_method": "paired_moving_block_bootstrap@1",
    "confidence_level": 0.95,
    "block_hours": 24,
    "bootstrap_resamples": 1000,
}
```

Run the focused tests and confirm they fail because these fields/version
changes are not yet present.

- [ ] **Step 2: Bump executable versions and freeze the profile**

Increment the implementation string of every greenhouse evaluator whose
scoring semantics changed. Expose the objective aggregation version, scoring
baseline policy, reward definition, hard gates, practical delta, and
confidence method in `objective_profile()` and therefore in the frozen run
metadata/digest contract.

- [ ] **Step 3: Update documentation from the implemented constants**

Document separately:

- model reference baseline versus scoring baseline;
- raw and normalized sample reward;
- selection score versus hard gates;
- legacy and new promotion behavior;
- confidence limitations and unchanged formal validation boundary;
- the locale-safe verification command for paths containing non-ASCII text.

Do not claim cross-episode validation, uncertainty calibration, causal effect,
or production control.

- [ ] **Step 4: Run source syntax and full Python verification**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHON="$PYTHON" \
  ./scripts/verify_delivery.sh --source-only
```

Expected: source metadata/syntax, example, unittest suite, JavaScript syntax,
browser smoke, and proxy security all pass. The script may explicitly skip
real AGC tests only when the datasets are not locally ready.

- [ ] **Step 5: Run independent targeted checks**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest -q
find plugins/ecology_evolution -name '*.js' -exec node --check {} \;
node plugins/ecology_evolution/test/smoke.mjs
find integrations/dsh_ecology_plugin -name '*.js' -exec node --check {} \;
node integrations/dsh_ecology_plugin/test/proxy_security.mjs
```

Expected: all commands exit 0.

- [ ] **Step 6: Verify artifact status without destructive overwrite**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHON="$PYTHON" \
  ./scripts/verify_delivery.sh --artifacts
```

If existing `dist/` artifacts fail source-binding verification because they
predate this change, report them as stale and do not delete or overwrite them
without an explicit release request. Source completion does not imply that a
new release archive was published.
