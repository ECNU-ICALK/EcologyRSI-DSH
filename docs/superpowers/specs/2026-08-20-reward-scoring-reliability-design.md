# Reward and Scoring Reliability Design

**Date:** 2026-08-20

**Status:** Approved for direct execution by the user

## 1. Context

EcologyRSI-DSH is a local, replayable research workbench for bounded model
evolution on greenhouse time-series data. Its current scientific decision is
not physical control: it selects a prediction candidate for continued search
using `training_fit` for fitting and a frozen, time-forward
`training_feedback` cohort for visible evaluation. Formal development, gate,
external-holdout validation, causal claims, and physical actuation remain
outside this change.

The repository already contains the right safety foundations: frozen dataset
and split digests, target/horizon metrics, persistence comparison, physical
range gates, coverage gates, independent review, append-only replay, and
same-cohort promotion rules. The current working source also contains an
unfinished reward/scoring revision. A malformed duplicate conditional in
`evaluators/registry.py` makes the evaluator module syntactically invalid, so
the Python test suite cannot currently be collected.

The scoring implementation is duplicated between the one-hour and
multi-horizon evaluators. This duplication is the root engineering condition
that allowed the current syntax defect and creates a high risk of semantic
drift between evaluators.

## 2. Goals

1. Restore an importable, testable source tree before changing behavior.
2. Make sample reward, selection score, hard gates, and promotion evidence
   separate, named, versioned concepts.
3. Use one pure scoring kernel for one-hour and multi-horizon greenhouse
   evaluation.
4. Compare candidates to a stronger, fit-selected causal naive baseline while
   retaining persistence as the model's residual reference.
5. Prevent missing, failed, or selectively dropped samples from improving a
   candidate's score.
6. Require a practically meaningful score improvement for formal promotion;
   when compatible paired block evidence exists, also require its confidence
   lower bound to be positive.
7. Preserve old ledger replay and public API compatibility through additive
   fields and explicit version changes.
8. Keep the implementation dependency-free beyond the Python standard
   library.

## 3. Non-goals

- Add crop growth, yield, irrigation, nutrient, plant–soil–microbe, or full
  greenhouse energy-balance models.
- Read or tune against `development`, `gate`, `external_holdout`, hidden, or
  final partitions.
- Claim statistical independence, causality, cross-site generalization, or
  production readiness.
- Replace the current registered predictors or permit generated source code.
- Introduce SciPy, NumPy, a distributed queue, or a database migration.

## 4. Architecture

### 4.1 New scoring module

Create `src/ecologyrsi_dsh/evaluators/objectives.py`. It owns only pure,
deterministic calculations:

- bounded normalized objective components;
- per-cell error/reward metrics;
- target/horizon weighting;
- coverage-aware missing-evidence penalties;
- deterministic block summaries and paired improvement confidence;
- validation of objective configuration and input rows.

It must not train models, call a gateway, read the ledger, mutate rows, or
construct domain events.

### 4.2 New baseline module

Create `src/ecologyrsi_dsh/evaluators/baselines.py`. It owns causal naive
baseline selection and application:

- `persistence`: predict the last target value available at the forecast
  origin;
- `seasonal_24h`: predict the target value at `target_timestamp - 24`, which
  is never later than the origin for the registered 1/6/24-hour horizons;
- select the lower-RMSE baseline independently for each target/horizon using
  only rows whose labels are inside `training_fit`;
- break ties in favor of persistence;
- fall back to persistence when the selected seasonal value is unavailable on
  an evaluation row;
- emit a frozen, digest-bound baseline profile.

The candidate predictors continue to use persistence as their internal
residual reference. Baseline application occurs after candidate prediction so
changing the scoring comparator cannot silently change model output.

### 4.3 Registry orchestration

`evaluators/registry.py` remains the public evaluator registry and orchestrator.
It delegates metric calculation and baseline selection rather than containing
separate copies. Existing public methods and evaluator identifiers remain
callable. Evaluator implementation versions and the objective aggregation
version are bumped because new runs have different scoring semantics.

### 4.4 Promotion logic

`core/director.py` remains the authority for formal promotion. It receives
additive reliability evidence through `Evaluation.metrics` and enforces the
promotion policy. `evolution/analysis.py` uses the same helper so automatic
generation analysis and direct/manual promotion cannot disagree.

## 5. Data flow

1. Fit the candidate model on `training_fit` only.
2. Build candidate-independent baseline candidates and select a baseline for
   each target/horizon using `training_fit` only.
3. Freeze scale method, target/horizon weights, baseline profile, objective
   version, evaluator digest, and random seed in the run/evaluation evidence.
4. Select the generation's label-free `training_feedback` cohort before any
   candidate-specific model execution.
5. Produce candidate predictions using the existing predictor and sample-tool
   workflow.
6. Apply the frozen scoring baseline after prediction. Preserve the original
   persistence value as `model_reference_baseline`.
7. Compute per-sample reward and per-cell metrics.
8. Apply coverage and physical gates, then aggregate the selection score.
9. Run the independent judge without allowing it to change scientific fields.
10. Rank same-cohort siblings. Formal promotion additionally applies the
    practical-effect and confidence policy below.

## 6. Reward and objective definitions

For observed value `y`, candidate prediction `p`, frozen scoring baseline `b`,
and target scale `s_t` estimated from `training_fit`:

```text
raw_reward = |b - y| - |p - y|
normalized_reward = clip(raw_reward / s_t, -1, 1)
```

Positive reward means the candidate reduced absolute error relative to the
frozen baseline. `raw_reward` remains in physical target units for audit.
`normalized_reward` is the cross-target learning signal. The scale is the
population standard deviation of finite `training_fit` target values with the
existing `1e-6` floor; its method and source partition are recorded.

For each target/horizon cell:

```text
candidate_nrmse = RMSE(p - y) / s_t
baseline_nrmse  = RMSE(b - y) / s_t
cell_skill      = clip(1 - candidate_nrmse / baseline_nrmse, -1, 1)
```

When baseline nRMSE is effectively zero, skill is zero only if the candidate
is also effectively perfect; otherwise it is `-1`.

Let `q` be successful model-evidence rows divided by eligible rows in the
cell. Operational failure is incorporated once:

```text
effective_cell_skill  = q * cell_skill + (1 - q) * -1
effective_cell_reward = q * mean(normalized_reward) + (1 - q) * -1
```

Failed sample rows remain in the private archive. Their displayed fallback
prediction is made no better than the frozen scoring baseline, so a failed row
cannot show a positive raw reward. Failed fallbacks do not enter candidate
RMSE as if they were model outputs; their effect is the explicit missing
evidence penalty and coverage gate above.

The canonical selection score is the weighted mean of
`effective_cell_skill`. The learning reward is separately the weighted mean
of `effective_cell_reward`; it does not replace the selection score.

Default weights are exactly:

- target: `1/3` for air temperature, relative humidity, and CO2;
- horizon: `1/H` across the evaluator's registered horizons;
- missing cell: `-1` without removing its denominator weight.

Weights, bounds, and penalties must be finite, non-negative where applicable,
cover exactly the registered targets, and sum to a positive value.

## 7. Hard gates

A greenhouse candidate is scientifically passing only if all conditions hold:

1. aggregate selection score is greater than `1e-9`;
2. every target/horizon candidate nRMSE is no greater than its frozen baseline
   nRMSE plus `1e-12`;
3. total physical-range violations are zero;
4. overall sample execution coverage is at least the frozen policy threshold;
5. every target/horizon cell reaches its frozen minimum coverage;
6. all objective cells and the baseline profile are present and digest-valid.

The judge may turn a scientific pass into a rejection but cannot turn a
scientific failure into a pass.

## 8. Promotion reliability

Point-estimate equality is not improvement. Replace the current effective
promotion margin of `1e-12` with a frozen practical effect:

```text
MINIMUM_PRACTICAL_SCORE_DELTA = 0.005
```

This is a 0.5 percentage-point improvement on the bounded skill scale. A new
run records this value in its objective profile.

For two evaluations on the same verified cohort and compatible evaluator,
objective, baseline-profile, and scoring versions:

- compute deterministic paired moving-block bootstrap evidence from ordered
  24-hour block sufficient statistics;
- use 1,000 resamples and a deterministic seed derived from the cohort and
  evaluation digests;
- report mean delta and the 2.5%/97.5% percentile interval;
- require `candidate_score > incumbent_score + 0.005`;
- when at least four paired blocks are available, also require the 95% lower
  bound of the paired delta to be greater than zero;
- when fewer than four blocks exist, mark confidence as
  `insufficient_blocks` and apply only the practical-effect rule;
- never compare different or unverifiable cohorts.

The bootstrap input is aggregate block sufficient statistics, not raw
observations, predictions, or model reasoning. It is excluded from remote
strategy and judge contexts and bounded in size. Direct promotion, automatic
generation analysis, and manual approval use the same promotion helper.

## 9. Compatibility and versioning

- Keep `absolute_error_improvement_vs_persistence@1` readable for historical
  sample archives.
- New runs use a new sample reward definition that names the selected scoring
  baseline while retaining the same sign convention.
- Add fields such as `model_reference_baseline`, `baseline_id`,
  `baseline_profile_digest`, `normalized_reward`, `objective_score`,
  `overall_reward`, and `promotion_reliability` without removing established
  fields.
- Bump `GREENHOUSE_OBJECTIVE_AGGREGATION_VERSION` and the relevant evaluator
  implementation strings so restart validation fails closed on semantic
  drift.
- Existing persisted evaluations without reliability evidence remain
  replayable. They may use the legacy `1e-12` rule only inside a run whose
  frozen evaluator digest and objective semantics are also legacy; new and old
  objective versions are never directly compared.
- No SQLite schema change is required because all additions live in existing
  JSON event payloads.

## 10. Error handling

- Invalid weights, duplicate cells, non-finite metrics, invalid horizons, or
  mismatched baseline profiles fail before evaluation persistence.
- Missing seasonal history deterministically falls back to persistence and is
  counted in baseline diagnostics.
- A missing cell produces the fixed `-1` penalty and a failed hard gate.
- Bootstrap incompatibility produces explicit non-comparability evidence; it
  never silently falls back to unpaired confidence.
- Syntax/source verification is the first delivery check so import failures
  stop before long tests or artifact construction.

## 11. Testing strategy

### Pure objective tests

- persistence-equivalent candidate has zero reward and zero skill;
- lower error is positive and higher error is negative;
- target-unit rescaling leaves normalized selection unchanged;
- row order and unequal sample counts do not change explicit cell weights;
- missing and partial-coverage cells receive exactly one penalty;
- extreme rewards and skills remain within `[-1, 1]`;
- duplicate/unknown target-horizon cells and invalid weights fail closed.

### Baseline tests

- seasonal values never come from after the forecast origin;
- selection reads `training_fit` only;
- lower fit RMSE wins and ties choose persistence;
- missing evaluation history falls back to persistence;
- profile digest is deterministic and candidate-independent.

### Promotion tests

- a delta at or below `0.005` cannot promote;
- a larger delta promotes when compatible evidence has positive lower bound;
- a non-positive lower bound blocks promotion;
- insufficient blocks use the explicit practical-effect-only status;
- different cohort/objective/evaluator/baseline digests remain incomparable;
- direct, batch, and manual promotion paths agree.

### Integration and regression tests

- one-hour and multi-horizon evaluators expose the same objective contract;
- failed sample fallback cannot produce positive reward;
- strategy/judge contexts do not receive block evidence or raw sample fields;
- frozen runtime binding rejects changed implementation digests;
- old events remain replayable;
- full unittest suite, source verification, examples, JavaScript syntax,
  browser smoke, proxy security, and artifact verification pass.

## 12. Delivery phases

1. **Restore:** reproduce and repair the syntax defect; verify all Python files
   parse and tests collect.
2. **Unify:** extract and integrate the pure objective kernel without changing
   externally observed score semantics beyond the already-started versioned
   reward revision.
3. **Strengthen:** add fit-selected scoring baselines and additive evidence.
4. **Harden promotion:** add the practical-effect and paired block-confidence
   policy with a single shared decision helper.
5. **Close delivery:** update documentation, run full verification, rebuild
   artifacts only after source verification is clean, and report any skipped
   real-data or external-network checks explicitly.

## 13. Assumptions and failure boundaries

- Canonical dataset timestamps are integer hours in a consistent local-series
  frame; `timestamp - 24` therefore represents the preceding daily phase.
- The registered horizons remain 1, 6, and 24 hours for multi-horizon
  greenhouse evaluation.
- Training-fit statistics and baseline selection are predictive calibration,
  not causal identification.
- A confidence interval on visible feedback does not establish external
  validity. Cross-episode, cross-season, development/gate, interval coverage,
  and mechanism validation remain required before broader claims.
- The smallest executable next step is the syntax-regression test and repair;
  no behavioral refactor proceeds until that gate is green.
