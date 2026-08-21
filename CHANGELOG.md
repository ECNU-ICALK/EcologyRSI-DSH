# Changelog

All notable changes to EcologyRSI-DSH are recorded in this file.

## 0.3.15 - 2026-08-21

DSH candidate-proposer convergence patch.

### Fixed

- Define genome operations as a concise mutation delta and explicitly forbid
  reconstruction of unchanged parent configuration, preventing the proposer
  from repeatedly reasoning through a full copied genome.
- Keep DSH-native context management and schema-bound structured output as the
  sole agent execution path; no per-sample Token hard cap was introduced.

## 0.3.14 - 2026-08-21

DSH transient-outage recovery patch.

### Fixed

- Keep an autonomous run alive and schedule a visible delayed retry when the
  DSH transport, service, or bounded structured-stage call is temporarily
  unavailable, instead of terminally failing after three rapid replays.
- Give each research-stage retry a distinct deterministic idempotency key while
  preserving the durable generation and attempt identity.
- Distinguish retryable DSH service errors from schema, identity, route, and
  response-contract violations so permanent protocol faults still fail closed.
- Leave a 60-second client response margin beyond DSH's own 600-second
  structured-stage deadline, allowing DSH to own cancellation and return its
  bounded error contract rather than racing a local socket timeout.

### Verified

- Recover a real DSH-native research stage across a Python sidecar restart and
  accept the second durable child launch without Host/model-gateway fallback.
- Pass the complete Python regression suite plus focused native-runtime,
  structured-role, cancellation, and sample-agent tests.

## 0.3.13 - 2026-08-21

Workbench completion-language clarification.

### Changed

- Replace the misleading user-facing “budget exhausted” completion wording
  with “completed the configured evolution scale”, including completed
  generation and candidate counts when available.
- Explain that no candidate passed all evaluation gates while retaining the
  original machine-readable outcome and termination reason for audit and
  replay compatibility.

## 0.3.12 - 2026-08-21

DSH multi-generation workbench reliability patch.

### Fixed

- Require one-shot DSH roles to call `structured_output` in their first response,
  avoiding provider-side analysis narration that can exhaust a stage timeout.
- Raise the bounded DSH web projection proxy response ceiling from 2 MiB to
  16 MiB and reject declared oversize responses cleanly, so real
  multi-generation projections remain refreshable in the workbench.

### Verified

- Retain the 0.3.11 parent-genome continuity fix and validate it through a
  real three-generation DSH-native evolution run.

## 0.3.11 - 2026-08-21

Multi-generation DSH parent-genome continuity patch.

### Fixed

- Resolve a new generation's research parent from the previous generation's
  persisted `search_parent_candidate_id` before the new generation batch has
  been created.
- Preserve the batch-frozen parent genome as the authority after batch
  creation, so restart and replay semantics remain unchanged.
- Add a regression test covering the exact generation-advance gap found by the
  three-generation web run.

## 0.3.10 - 2026-08-21

DSH Host-bound structured-response patch.

### Fixed

- Send the concise first-response `structured_output` instruction in every
  Host-generated one-shot and Workflow child prompt, so the constraint reaches
  both existing and newly installed frozen role presets.
- Restore the `v1` preset definitions to their frozen content instead of
  silently changing an already published preset identity.
- Add a request-boundary regression test that inspects the actual child prompt
  delivered to DSH.

## 0.3.9 - 2026-08-21

Preset-only DSH structured-role response hint (superseded by 0.3.10).

### Changed

- Require researcher, candidate-proposer, sample-planner, sample-critic, and
  generation-judge presets to submit the requested schema in their first
  response without narrating long analysis.
- Add a preset contract test that keeps the concise first-response instruction
  present across every structured role.

## 0.3.8 - 2026-08-21

DSH proposal provenance visibility patch.

### Fixed

- Mark candidate mutations produced by the native DSH candidate-proposer as
  `dsh_native_agent` instead of projecting them as `legacy_unknown`.
- Count native DSH proposal calls and successes in run and round diagnostics.
- Show an explicit DSH-native proposal label in process and candidate views.

## 0.3.7 - 2026-08-21

DSH native child-session metrics replay patch.

### Fixed

- Accept the optional, ingress-validated `session_metrics` field while replaying
  `DshStructuredResultAccepted` events.
- Revalidate exact context-pressure and provider-reported token-usage semantics
  during replay, including Session identity and token-total consistency.
- Add a regression test for the real write-success/read-failure boundary found
  by the web evolution run.

## 0.3.6 - 2026-08-21

DSH rc.6 one-shot child identity compatibility patch.

### Fixed

- Read the published one-shot child Session id from `SubagentRun.id`, matching
  the rc.6 public contract. `childId` remains limited to continuable starts and
  Workflow lifecycle events.

## 0.3.5 - 2026-08-21

Bounded DSH structured-stage diagnostics.

### Added

- Added stable phase codes for child start/result, schema-bound output,
  admission, child Session identity and durable-result persistence failures.
  Provider error text remains behind the redaction boundary.

## 0.3.4 - 2026-08-21

DSH Session observability fail-soft patch.

### Fixed

- Made final TokenMeter and Session Projection snapshots independent and
  fail-soft so a cold/disposed child Session cannot reject an otherwise valid
  structured scientific result.

## 0.3.3 - 2026-08-21

DSH-native runtime observability and catalog correction.

### Fixed

- Read the real flat four-bucket DSH Token Usage projection so cumulative
  provider usage is no longer reported as unavailable.
- Retained the Python sidecar's already-redacted bounded rejection detail in
  DSH runtime diagnostics without exposing tokens or response headers.
- Corrected the catalog and workbench banner to report native DSH
  Agent/Session/Workflow execution instead of the legacy compatibility path.

## 0.3.2 - 2026-08-21

DSH-native closed-loop evolution hardening patch.

### Fixed

- Routed sample-planner waves through the real DSH Workflow Engine and
  persisted real child Session identities and DSH provider-usage projections.
- Added actionable cross-generation reflection and rejected exact replays of
  previously failed compiled behavior.
- Kept raw scientific fitness separate from execution-policy quality while
  preserving Host-owned physical-range repair and terminal authentication
  failures.
- Closed the cancellation race that allowed stale workers to append stage
  observations after a run became terminal.
- Aligned sample reason-code schemas with the Host enum and exposed DSH context
  pressure without introducing a Token hard cap.

## 0.3.1 - 2026-08-21

DSH-native runtime reliability patch.

### Fixed

- Bounded DSH critic inputs to aggregate evidence while preserving independent
  review of every selected sample.
- Removed private sample execution archives from generation-judge context and
  classified deterministic judge contract failures as permanent.
- Prevented unavailable judges from entering an unbounded retry loop or the
  candidate decision path.
- Normalized event payloads through canonical JSON before idempotency checks,
  preventing tuple/list representation drift after a successful append.
- Added a DSH child operational timeout without imposing any Token, context, or
  output-length cap.

## 0.3.0 - 2026-08-21

DSH-native plugin evolution runtime.

### Added

- Immutable plugin genomes, deterministic compilation, DSH role presets,
  structured Agent stages, durable child reservations, restart reconciliation,
  and pause/cancel admission barriers.
- DSH-native sample planning and review with Host-owned numerical tools while
  preserving the existing reward contract for identical predictions.
- Cellwise calibrated residual uncertainty artifacts and formal point/UQ gates.

### Changed

- Agent context, compaction, subagents, workflows, and output length are owned
  by DSH. The workbench no longer accepts a per-sample Token hard limit.
- Public projections distinguish search, validation, runtime, genome, and
  fitness identities without exposing private Agent messages.

## 0.2.2 - 2026-08-20

Reward, baseline, and promotion reliability hardening.

### Added

- A pure, versioned objective kernel with fixed target/horizon denominators,
  bounded skill and reward components, explicit coverage penalties, and strict
  rejection of duplicate or unknown task cells.
- Leakage-safe baseline fitting that selects persistence or a causal 24-hour
  seasonal comparator per target and horizon using only `training_fit`.
- Per-sample baseline provenance, normalized rewards, and compatibility-aware
  decoding for historical persistence-reward archives.
- A shared promotion policy with a practical score delta of `0.005` and a
  deterministic 1,000-resample paired 24-hour origin-block bootstrap that
  reconstructs coverage-penalized RMSE skill from bounded sufficient statistics.

### Changed

- Greenhouse objective aggregation is now `weighted_task_skill_reward@2` and
  reward is `absolute_error_improvement_vs_fit_selected_baseline@2`.
- The one-hour and multi-horizon evaluators apply the selected scoring baseline
  only after model prediction, preserving the original persistence input as
  `model_reference_baseline`.
- Direct, automatic, and manual promotion paths now use the same version-aware
  improvement assessment. Legacy evaluations retain their historical `1e-12`
  rule only under a matching common contract and are not compared directly
  with v2 scoring contracts.
- Evaluator configuration digests now freeze objective, reward, baseline,
  hard-gate, and promotion-confidence constants.

### Fixed

- Repaired a malformed duplicate objective metric expression that prevented the
  Python evaluator registry from importing.
- Repaired stale one-hour objective variable references that failed evaluation
  after the earlier score-field rename.
- Prevented failed sample executions from receiving positive reward against the
  stronger frozen scoring baseline.
- Made evaluation cohort identity independent of candidate execution failures,
  rejected tampered baseline profiles, and removed failed fallbacks and private
  bootstrap evidence from public prediction projections.
- Required complete SHA-256 scoring contracts for v2 comparisons, bound block
  weights and horizons to the frozen objective, and seeded confidence evidence
  from the cohort and both evaluations.

## 0.2.1 - 2026-08-20

Delivery hardening and observable autonomous evolution.

### Added

- Per-sample execution evidence for observed and predicted values, reward,
  agent/tool attempts, retry counts, failure categories, and conservative
  scoring fallback, exposed through bounded APIs and candidate evaluation UI.
- A versioned multi-horizon greenhouse evaluator and a registered ridge
  predictor with independent residual scales for every target at 1, 6, and
  24 hours.  The original evaluator contract and digest remain available for
  replaying historical runs, while new greenhouse runs use the expanded v2
  contract.
- A frozen research-to-execution chain covering literature evidence, research
  plan, algorithm blueprint and synthesis, restricted IR compilation, static
  debug, training smoke, real-sample execution, and cross-generation feedback.
- Feedback-driven planner, predictor, critic, repair-tool, and host-adjudication
  sample loops with bounded retries and microbatch remote calls. Individual
  sample failures are penalized without aborting an otherwise viable candidate.
- Continuous generation progression with bounded workers, durable recovery,
  per-run generation locks, and explicit archive, restore, and delete controls.
- A dedicated parameter-design workspace for generation count, candidates per
  generation, feedback samples per update, gateway microbatch size, sample
  concurrency, candidate and token budgets, seed policy, and knowledge retrieval.
- Release-bound real API acceptance reports with a per-file source manifest and
  verified SHA-256 identities for the wheel, sdist, and complete delivery archive;
  artifact/source equality is checked both before and after the live API run.

### Changed

- Remote model startup no longer depends on a separate browser API probe.
  Runtime calls use longer bounded timeouts, backoff, and retry classification.
- Research-plan calls explicitly reserve up to 8,192 output tokens, and the
  workbench uses one projection monitor for server-managed continuous runs
  with slower polling outside evaluation stages.
- Real API acceptance no longer auto-allows a non-loopback plaintext HTTP
  provider; release evidence defaults to configured HTTPS model routes.
- Real API acceptance now reads the selected evolution ledger in read-only mode
  and binds evidence to its latest matching GLM 5.2 and DeepSeek Flash
  `RunCreated` configuration instead of a stale hard-coded run identity.
- Candidate evaluation now separates live, paused, cancelled, partial, and
  completed sample states and preserves the last coherent snapshot on errors.
- New real autonomous runs use a deterministic, target/horizon-interleaved
  `training_feedback` window of 500 samples per update by default. Sibling
  candidates share the same window, while registered predictors such as ridge
  regression continue to fit on the full `training_fit` partition. Historical
  manifests without `samples_per_update` retain their full-feedback behavior.
- Promotion and cross-generation experience are cohort-aware: scores are
  compared strictly only for identical evaluation cohorts; after a window
  change, the current cohort's eligible champion advances without treating its
  raw score as an improvement over the prior window.
- Release verification now compares wheel, sdist, and complete delivery archive
  bytes with the current source tree, and cross-checks every user-visible and
  handshake version marker, preventing stale artifacts from passing.
- The DSH host bridge is explicitly private and unlicensed for redistribution,
  matching the proprietary root package.
- Isolated release builds pin their setuptools and wheel versions, and source
  distributions plus complete delivery archives include the workspace lockfile.

### Fixed

- Kept terminal run views on the latest persisted round and candidate evidence,
  including completed, failed, cancelled, paused, empty-round, and partially
  evaluated states. Candidate and process views now converge on the same latest
  sample revision without overriding an explicit operator selection.
- Made truncated or malformed proposal and independent-review responses enter
  durable cooldown recovery. Completed scientific evaluations are reused when
  only the final judge is temporarily unavailable, while authentication,
  permission, role, routing, and configuration errors remain terminal.
- Prevented candidate-budget exhaustion from completing a run before the final
  candidate evaluation is sealed, and saturated persistent retry backoff before
  exponentiation so very large retry histories cannot overflow.
- Added a bounded run-list summary projection for browser polling and removed
  the global mutation lock from long-running manual generation advances. This
  keeps pause, cancel, and unrelated runs responsive during remote calls while
  retaining the per-run generation lease.
- Ignored expected client disconnect exceptions at the HTTP server boundary so
  browser refreshes and bounded CLI reads do not emit misleading sidecar
  tracebacks; unexpected request exceptions still use the standard handler.
- Rendered a temporarily unavailable independent judge as an automatic retry
  or unavailable state instead of a rejection, and withheld the provisional
  `judge_accepted=false` metric until a completed judge response exists.
- Derived each evaluator's minimum feedback window from its complete
  target-by-horizon task count (1 for toy, 3 for single-horizon greenhouse,
  and 9 for multi-horizon greenhouse), and reject smaller runs in both the API
  and browser command layer. This prevents deterministic scientific-gate
  failures caused by a feedback window that can never cover every task.
- Kept research-stage timeouts, truncated responses, non-definitive gateway
  response failures, and exhausted semantic corrections alive as durable
  15-to-300-second cooldown retries.  Definite authentication and other 4xx
  failures remain terminal, and a retry releases the autonomous worker instead
  of replaying or failing the complete generation.
- Separated executable-blueprint evidence from literature-synthesis evidence:
  blueprints remain bound to registered predictors, while synthesis may cite
  any item in the frozen generation snapshot. Repeated semantic-contract
  failures now enter a durable cooldown and retry instead of terminating the
  complete evolution run.
- Kept long-running research stages visibly timed from their durable start
  event, restored the create action after asynchronous submission, and exposed
  pending, answered, applied, and audit-only expert-consultation states without
  requiring synchronous expert feedback.
- Unified public-data redaction across event, run, training-asset, trajectory,
  and strategy projections. Credential-like keys are matched after
  case/punctuation folding, and persisted exception diagnostics expose only a
  bounded exception type and error code rather than raw exception messages.
  Legacy model-health rows are scrubbed on open, and HTTP errors cannot expose
  credential-like text, private URLs, or local absolute paths.
- Removed API facade import-order dependencies so events, execution, catalog,
  projection, transport, and handler modules can each be imported in a fresh
  process without importing `server` first.
- Requeued failed terminal-event writes without replaying the completed
  generation, preventing a transient ledger failure from leaving a run marked
  `running` but absent from the autonomous worker queue.
- Limited complete sample-response JSON syntax recovery to one extra top-level
  request per wave; split descendants do not inherit the configurable transport
  retry budget, and progress records the real HTTP attempt count.
- Made the default adaptive splitter reduce malformed causal waves of eight or
  fewer decisions to singletons while retaining the existing bounded split
  floor for larger batches and explicit caller overrides.
- Prevented exhausted planner or critic transport retries from being replayed by
  the sample repair layer, while preserving decoded contract failures as
  eligible for adaptive splitting and repair.
- Mapped remote sample `reason_code` values to a finite host-owned vocabulary so
  a model cannot reflect Authorization data into metrics or compressed traces.
- Prevented stale or out-of-order projection and event responses from replacing
  newer UI state; fixed run-switch and sample-pagination rollback behavior.
- Accepted the observed GLM JSON-mode corruption only when an exact redundant
  object prefix can be removed into one complete object; all nearby malformed,
  multi-object, array-wrapped, truncated, and private-reasoning forms stay rejected.
- Made run selection transactional: an unreadable target no longer clears the
  current run's creation state, sample view, timing, or automatic progression.
- Cleared run-scoped failure banners after a successful context switch and
  kept the run selector readable when status filters and actions share the bar.
- Preserved the complete run outcome and formal-validation boundary on narrow
  screens instead of truncating it behind an ellipsis.
- Hardened the DSH loopback proxy against absolute-URL, path traversal, prefix
  escape, credential forwarding, cookie reflection, and upstream error leaks.
- Added state-recovery and command/idempotency checks for interrupted autonomous
  runs, while keeping historical event schemas replayable.
- Kept public metrics, the retained-score summary, and the green trajectory tied
  to the actual incumbent after every promotion, including a lower raw score on
  a different feedback cohort. The cross-run raw maximum is now observation-only,
  and experience summaries report the formal comparison score separately from
  `batch_highest_observed_score` so an unaccepted candidate cannot imply progress.
- Made hard Token-budget admission deterministic when the configured sample
  concurrency exceeds the number of calls that can be reserved. The coordinator
  now starts the earliest affordable schedules first instead of letting worker
  lock timing choose which samples consume the remaining budget.
- Restored isolated PEP 517 builds so release creation installs the declared
  build backend instead of depending on undeclared packages in the project venv.

### Boundaries

- This remains an unsigned, single-host research delivery candidate. Formal
  development/gate/holdout evaluation, production isolation, plugin signing,
  and physical greenhouse actuation remain outside the release.

## 0.2.0 - 2026-08-16

Deliverable ecology evolution workbench.

### Changed

- Reorganized the Python implementation into application, core, data,
  evolution, evaluator, integration, presentation, and API subpackages while
  retaining thin compatibility modules for the original public imports.
- Split the browser workbench into small dependency-free feature scripts under
  `assets/js`; `app.js` now contains only event binding and startup.

### Added

- Runnable AGC 2018 cucumber and AGC 2019 tomato historical datasets with
  hourly normalization, time-forward splits, embargoes, and safe sample APIs.
- Source-archive provenance auditing with declared size and MD5 verification,
  exposed separately from extracted-data readiness without local path leakage.
- `data audit` and `data fetch` CLI commands for HTTPS download, bounded
  size/MD5 verification, conflict-safe reuse, and guarded ZIP/7z extraction.
- Selectable rolling-residual and exogenous-ridge prediction models, plus a
  frozen compatibility contract between datasets, predictors, and evaluators.
- Leakage-bounded exogenous ridge fitting and 1/6/24-hour greenhouse evaluation
  across temperature, relative humidity, and CO2, including horizon summaries,
  persistence baselines, missing rows, physical-range violations, and previews.
- Bounded parameter sweep, local adaptive, and authenticated DSH proposal
  strategies with separate policy and judge model roles.
- Training artifacts, three-target greenhouse evaluation, baseline comparison,
  score trajectories, redacted event projections, and append-only human input.
- Five Chinese workspaces for setup, training data, evolution progress,
  candidate evaluation, and collaboration/governance.
- Strict incumbent retention: the first passing candidate may establish the
  incumbent; later candidates must improve its score by more than `1e-12`.
  Passing without improvement is rejected, and manual approval cannot bypass
  the invariant.
- One redacted `projection.training_assets` record per candidate, restricted to
  `iterative_positive`, `iterative_negative`, `quarantine`, or `pending`, plus
  a six-stage `projection.rounds` view from proposal through retention decision.
- Chinese training-asset and per-round stage views in the webview workbench,
  with sample browsing for both `training_fit` and `training_feedback` partitions.
- Explicit authenticated-model connection verification; remote policy and judge
  models must pass a minimal Bearer JSON probe before a run can start.
- Unified DSH model directory for the strategy and independent-review API
  selectors, with authenticated and pending-verification subsets, host-session
  ID intersection, and case-insensitive role aliases.  The DSH Web Profile host
  now reads its sanitized `llm.models` directory and passes provider/model
  aliases through the handshake; the server-side directory remains authoritative.
- Deterministic, bounded execution receipts for guidance, parameter overrides,
  numeric constraints, and parent selection; unparsed text is recorded-only.
- Complete redacted evolution episodes with five stages, lineage,
  reproducibility metadata, event receipts, and self-verifying digests.
- Live progress projection, dataset retry and frozen-run context, expandable
  event history, model connection status, and responsive mobile charts.
- Durable started/completed/failed events for proposal, candidate, training,
  evaluation, judge, and decision stages, including restart-safe event IDs.
- A real fit-to-evaluation callback boundary, so live stage polling and failure
  attribution switch from training to feedback scoring at the actual boundary.
- Chinese source-integrity, multi-horizon process, learned-model, and prediction
  detail views with predictor-specific human parameter guidance.
- Observable first-round startup, frozen remote-model re-verification after a
  service restart, and a wrapped training-data toolbar down to narrow viewports.
- Per-generation knowledge snapshots with a curated offline algorithm catalog,
  optional OpenAlex metadata retrieval, explicit host-capability mapping,
  non-executable research-only isolation, and non-causal round-end assessment.
- Chinese knowledge-source, execution-status, snapshot, and next-action views,
  with the same snapshot digest carried into candidate training-asset lineage.
- A local DSH Web Profile Cordis plugin that registers a Chinese sidebar entry,
  embeds the workbench in a DSH overlay, serves its static assets, and proxies
  the EcologyRSI API through the DSH origin.

### Boundaries

- Real greenhouse runs are historical replay and prediction only; they do not
  support counterfactual control or causal claims.
- Lettuce datasets remain catalog-only until their multimodal adapters exist.
- Development, gate, hidden, final, and external holdout samples remain outside
  browser access and the adaptive loop.
- Evolution training assets always declare `formal_training_ready=false` and
  require governance review; they are not formal SFT or DPO datasets.
- DSH integration includes a local Web Profile host plugin and a server-side
  OpenAI-compatible Bearer gateway, but remains unsigned. It is not official
  DSH OAuth, account authentication, plugin signing, or marketplace publication.

## 0.1.0 - 2026-08-16

Initial runnable MVP.

### Added

- Dependency-free Python evolution core with immutable run contracts.
- Append-only SQLite event ledger and deterministic replay projection.
- Narrow DSH adapter protocol with a deterministic local fake adapter.
- Structured parameter proposals; arbitrary candidate code is not executed.
- Toy crop-soil-water fixture with time-forward train, validation, and test splits.
- Local CLI and standard-library HTTP API for create, advance, control, and replay.
- CLI preflight (`doctor`), compact summaries, projection-only export/verify, and
  append-only import/replay of run bundles.
- Browser-native Ecology Evolution plugin shell and development manifest.
- Wheel, source distribution, and complete delivery-archive build checks.

### Boundaries

- The toy domain is an engineering fixture, not a validated scientific model.
- The plugin manifest is SDK-neutral and development-only.
- Hidden/final evaluation, production isolation, signed release governance,
  arbitrary-code evolution, and real DSH integration are not implemented.
