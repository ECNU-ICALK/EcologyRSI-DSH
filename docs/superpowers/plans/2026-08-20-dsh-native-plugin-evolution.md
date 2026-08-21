# DSH-native Ecology Plugin Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a versioned Ecology plugin genome—not DSH—the evolvable object, execute every new autonomous model/agent action through DSH-native Agent/Session/Compaction/Subagent/Workflow services, and promote candidates with non-compensatory, statistically reliable scientific fitness.

**Architecture:** Keep DSH and the Python scientific/safety kernel stable. A deterministic compiler lowers a bounded `EcologyEvolutionPluginGenome` into the existing registered `AlgorithmIR`, registered feature/training policies, and role-isolated DSH workflow specs. A Cordis Host Controller runs role-specific DSH presets and calls the Python sidecar only through structured, authenticated ecology tools. Existing reward and primary score remain the scientific core; validity, reliability, robustness, uncertainty, and efficiency become lexicographic fitness layers.

**Tech Stack:** Python 3.10–3.12 standard library, append-only SQLite event ledger, `pytest`/`unittest`, Node.js ESM and `node:test`, Cordis 4.0.1, DSH 0.1.0-rc.6, HTML/CSS/vanilla JavaScript workbench.

**Spec:** `docs/superpowers/specs/2026-08-20-dsh-native-plugin-evolution-design.md`

## Global Constraints

- DSH is a stable runtime dependency and is never mutated, patched by a genome, or used as the fitness target.
- The Cordis plugin source is a stable executable shell. Runtime evolution changes only a validated declarative genome.
- New autonomous runs use `dsh_native_plugin_evolution@1`; Python model HTTP calls and silent fallback are forbidden.
- New Agent creation omits `maxTokens`; new run requests omit Token hard-budget fields. DSH owns context and output management.
- Keep the existing per-sample reward and coverage-penalized target × horizon primary score unchanged while runtime migration is underway.
- Fitness is lexicographic. Efficiency can break ties only after validity, primary science, reliability, robustness, and required UQ pass.
- Genomes cannot contain executable source, shell commands, dynamic imports, arbitrary URLs/prompts, unrestricted tools, credentials, labels, rewards, or fitness.
- Genome source refs are recursively immutable and single-source; AlgorithmIR and complete Workflow graphs exist only as compiler output.
- A candidate controls only its candidate-execution program. Its reproduction program is recorded/inherited but mutation-disabled in V1; proposer/critic/judge selection for the current candidate is fixed by parent/security state.
- New-protocol genome, compiled-behavior, and phenotype-instance binding fields are mandatory; optional/null compatibility applies only to historical protocols.
- A role tool policy may only narrow the host allowlist. Enforce visibility with `restrict()` and authorization with an execution-time `guard()`.
- DSH Worker Workflow scripts are host-authored fixed templates. Never execute model- or sidecar-provided JavaScript in the Workflow VM.
- Each Workflow fans out one homogeneous role because rc.6 children inherit the parent preset and do not support per-call tool filters/personas.
- Sibling candidates receive the same frozen stage-context envelope in fresh Sessions and cannot read sibling intermediate results.
- Reviewer Sessions are fresh and independent from proposer/planner Sessions.
- Workflow handles are in-memory and cannot resume after process restart; recovery reconciles idempotent items and starts a new Workflow.
- `training_fit` and `training_feedback` are the only evolution-visible label partitions. Validation/final-test results never feed search.
- Adaptive selection evidence is explicitly exploratory. Raw development/gate exposure and its locked analysis plan are durable across runs; changing objectives cannot reopen a holdout.
- Public projections and exports never expose private sample rows, block sufficient statistics, credentials, full system instructions, or complete DSH Sessions.
- DSH 0.1.0-rc.6 packages are pinned exactly during this migration; Session format v0 is treated as release-sensitive.
- The workspace is not a Git repository. Use test-verified filesystem checkpoints/release archives rather than commit steps.
- Every behavior change follows RED → GREEN → REFACTOR. Run the named failing test before editing production code.

---

### Task 0: Re-establish the implementation baseline

**Files:**
- Read: `pyproject.toml`
- Read: `integrations/dsh_ecology_plugin/package.json`
- Read: `scripts/verify_delivery.sh`
- Verify: all current Python and Node tests

**Interfaces:**
- Consumes: the current `0.2.2` source tree and installed `.venv`.
- Produces: a recorded clean baseline before protocol/schema changes.

- [ ] **Step 1: Verify Python syntax and test collection**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest --collect-only -q
```

Expected: exit 0 and a non-zero test count.

- [ ] **Step 2: Run the current full Python suite**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest -q
```

Expected: all existing tests pass. Diagnose any failure before beginning Task 1.

- [ ] **Step 3: Run current Node contracts**

Run:

```bash
node integrations/dsh_ecology_plugin/test/proxy_security.mjs
node plugins/ecology_evolution/test/smoke.mjs
```

Expected: both commands exit 0.

- [ ] **Step 4: Capture a recoverable delivery checkpoint**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 \
  bash scripts/build_delivery.sh
```

Expected: the current source/wheel/archive verification succeeds. Retain the generated `0.2.2` delivery artifact as the pre-migration recovery point.

### Task 1: Freeze canonicalization, registries, immutable genome, and legacy identity

**Files:**
- Create: `src/ecologyrsi_dsh/knowledge/program_registry.py`
- Create: `src/ecologyrsi_dsh/knowledge/legacy_program_catalog_0_2_2.json`
- Create: `src/ecologyrsi_dsh/evolution/genome.py`
- Create: `tests/fixtures/legacy_program_0_2_2.json`
- Create: `tests/fixtures/legacy_program_0_2_2_no_snapshot.json`
- Create: `tests/test_evolution_genome.py`
- Modify: `src/ecologyrsi_dsh/evolution/__init__.py`
- Modify: `src/ecologyrsi_dsh/knowledge/__init__.py`

**Interfaces:**
- `ProgramRegistrySnapshot` and immutable `LEGACY_PROGRAM_CATALOG_0_2_2`.
- `SeedGenomeTemplate`, `FrozenRunInitialization`, its schema-listed `GenomeBindingSubset`, and `materialize_seed_genome(template, initialization)`; compiler digest stays outside the genome.
- `deep_freeze_json(value) -> FrozenJson` and detached `deep_thaw_json(value)`.
- `EcologyEvolutionPluginGenome.from_dict(value) -> EcologyEvolutionPluginGenome`.
- `GenomeMutationContextV1` with run/generation/slot/seed/parent/batch/research/knowledge/budget/operator identity.
- `apply_genome_mutation(parent, accepted_mutation, context, registry) -> EcologyEvolutionPluginGenome`.
- `legacy_genome_from_proposal(proposal, task, knowledge_snapshot: KnowledgeSnapshot | None, legacy_catalog=LEGACY_PROGRAM_CATALOG_0_2_2)`.
- `migrate_legacy_seed(projected_legacy, bindings, frozen_migration_template) -> EcologyEvolutionPluginGenome`.

- [ ] **Step 1: Write RED tests for immutable identity and one truth source**

Add tests for nested post-validation mutation, detached `to_dict()`, Unicode/number/set canonicalization, bool/NaN/Infinity rejection, domain-separated digests, deterministic content-derived genome IDs, origin lineage, mutation context replay, parent/batch/slot consistency, tool narrowing, fixed reviewer ownership, and arbitrary code/prompt/script rejection.

Explicit tests include:

- `test_genome_nested_state_cannot_mutate_after_validation`;
- `test_to_dict_returns_detached_deep_copy`;
- `test_digest_normalization_rejects_bool_nan_infinity_and_duplicate_set_items`;
- `test_workflow_graph_cannot_be_supplied_when_template_is_authoritative`;
- `test_candidate_cannot_select_its_own_proposer_critic_or_judge`;
- `test_mutation_digest_covers_accepted_mutation_and_full_context`;
- `test_first_generation_lineage_uses_materialized_seed_parent_and_null_parent_candidate`;
- `test_seed_template_cannot_contain_run_specific_bindings`;
- `test_materialized_seed_contains_all_frozen_run_bindings`.

- [ ] **Step 2: Write legacy golden RED tests**

Create two fixtures: one with a real-shaped frozen `KnowledgeSnapshot`, and one with historical `KnowledgeSnapshot=None`. Both include algorithm blueprint, synthesis, prediction-model adoption, proposal changes, and the old full `AlgorithmIR.to_dict()`. Test that current-registry changes do not alter either projected legacy genome, the no-snapshot projection preserves `knowledge_snapshot_digest=None`, and every projection is read-only.

Required compatibility tests include `test_legacy_no_snapshot_projection_preserves_none_and_full_ir`, `test_legacy_projection_ignores_current_registry_changes`, and `test_projected_legacy_requires_explicit_migration_seed`.

- [ ] **Step 3: Run RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_evolution_genome.py -q
```

Expected: import failures for the registry/genome modules.

- [ ] **Step 4: Implement canonical immutable JSON and registry snapshots**

Use typed frozen child contracts plus an internal recursively immutable JSON representation. Normalize Unicode NFC, sorted object keys, declared set fields, `-0.0`, and finite numbers exactly as the spec requires. Use domain-separated SHA-256; do not reuse the generic object digest for genome identity.

Register the current V1 predictor/operator/feature/fit/UQ/workflow/instruction/tool catalogs and permanently ship `legacy-program-catalog@0.2.2`. The legacy catalog is never generated from the current registry at replay time.

The catalog stores only `SeedGenomeTemplate`, never a complete task-bound genome. A template omits lineage, task/run/data/route/security binding and cannot be used as a GenerationBatch parent until materialized.

- [ ] **Step 5: Implement the single-source genome**

Store only registered source refs plus bounded overrides. Reject candidate-supplied `AlgorithmIR`, Workflow nodes/edges, reviewer program, model credentials, labels, rewards, fitness or arbitrary prompt text. Keep the V1 reproduction-program mutation mask false until a separately versioned lineage-level offspring fitness exists. Use `genome_revision`; reserve `run_state_revision`, `stage_attempt`, and `ledger_expected_revision` for runtime/ledger contracts.

- [ ] **Step 6: Implement deterministic mutation lineage**

Require `GenomeMutationContextV1`. Derive `genome_digest` without `genome_id`, then set `genome_id="genome:" + genome_digest[:24]`. Validate lineage against parent, GenerationBatch, Proposal and Candidate coordinates. Tool sets are sorted unique and must be subsets of the resolved base policy.

- [ ] **Step 7: Implement seed materialization and explicit legacy migration**

`materialize_seed_genome()` combines one catalog template with complete frozen run bindings and returns a canonical full parent genome. `migrate_legacy_seed()` combines the immutable legacy scientific projection with a versioned DSH-native migration/agent template and new-run bindings. Neither may read a mutable catalog after materialization; the complete canonical result is suitable for event persistence. A projected legacy view is never itself inheritable.

- [ ] **Step 8: Implement the frozen legacy adapter**

Use the supplied frozen knowledge snapshot or the versioned no-snapshot sentinel and the legacy catalog. The sentinel participates only in adapter identity: emitted legacy `knowledge_snapshot_digest` stays `None`. Treat old blueprint/synthesis/adoption/changes as legacy compiler source and the old `DerivedExecutionPlan` as phenotype evidence. Never inspect labels, evaluations or current registry entries.

- [ ] **Step 9: Run focused identity/legacy regressions**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_evolution_genome.py \
  tests/test_research_iteration.py \
  tests/test_algorithm_compilation.py -q
```

Expected: all pass without changing current execution behavior.

### Task 2: Compile source genomes into closed behavior and instance identities

**Files:**
- Create: `src/ecologyrsi_dsh/evolution/workflow_ir.py`
- Create: `tests/test_workflow_ir.py`
- Modify: `src/ecologyrsi_dsh/knowledge/program_registry.py`
- Modify: `src/ecologyrsi_dsh/knowledge/algorithm_ir.py`
- Modify: `src/ecologyrsi_dsh/knowledge/algorithms.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py`
- Modify: `tests/test_algorithm_compilation.py`
- Modify: `tests/test_algorithm_ir_smoke.py`

**Interfaces:**
- `compile_dsh_workflow_spec(template_ref, overrides, role_profiles, registry) -> CompiledDshWorkflowSpec`.
- `compile_plugin_behavior(genome, task, knowledge_snapshot, registry) -> CompiledEcologyBehaviorSpec`.
- `bind_phenotype_instance(compiled_behavior, CompilationInstanceContext) -> BoundEcologyPluginSpec` after Candidate ID exists.
- `CompiledEcologyBehaviorSpec` includes full scientific/feature/fit/UQ/workflow specs plus compiler/registry/security/runtime semantic digests and `compiled_behavior_digest`; the bound spec adds runtime-execution and `phenotype_instance_digest`.

- [ ] **Step 1: Write closed-compiler RED tests**

Test unknown/mismatched source refs, candidate-supplied AlgorithmIR or graph, conflicting refs/overrides, registry templates containing cycles or mixed-role privilege, reviewer Session reuse, expanded partitions, tool-policy expansion, unresolved model route, and mismatched data/stage/security digests. Freeze `algorithm_behavior_projection@1` as a positive field allowlist; tests prove rationale/evidence/source-plan/instance changes do not alter it, effective parameter/operator/feature/workflow/tool changes do, and omitted defaults equal explicit defaults.

Add:

- `test_one_resolved_behavior_has_one_compiled_behavior_digest`;
- `test_behavior_identical_siblings_share_compiled_digest_but_not_instance_digest`;
- `test_instance_identifiers_never_enter_compiled_behavior_digest`;
- `test_same_genome_under_different_compiler_has_distinct_compiled_behavior_digest`;
- `test_resolved_route_config_is_covered_by_runtime_execution_digest`;
- `test_legacy_adapter_and_new_compiler_equal_old_full_algorithm_ir_dict`.

- [ ] **Step 2: Run RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_workflow_ir.py tests/test_algorithm_compilation.py -q
```

Expected: missing workflow/compiler interfaces.

- [ ] **Step 3: Validate host-authored registry templates**

Validate the complete DAG only when loading the trusted registry. A candidate genome can select a template and bounded overrides but cannot submit nodes/edges. Fixed reviewer templates are resolved from the security/fitness kernel, not the candidate.

- [ ] **Step 4: Lower the scientific and agent programs**

Produce the existing behavioral `AlgorithmIR`, registered feature/fit/UQ spec, candidate-execution Workflow spec, and next-generation reproduction spec. Do not generate JavaScript. The behavior compiler has no candidate ID and strips all instance/display/evidence-only fields. After `CandidateSpawned`, bind run/proposal/candidate/generation/slot and exact route/config, preset content/tool surface, data protocol, stage and evaluation context into the bound `AlgorithmSpec` and `runtime_execution_digest`.

- [ ] **Step 5: Compute separate behavior and instance digests**

First build an explicit behavior projection that strips run/proposal/candidate/generation/slot, lineage and genome identity from `AlgorithmSpec` and all agent specs. Domain-separate `compiled_behavior_digest` over resolved scientific/agent behavior plus compiler/registry/security/runtime semantic inputs. Then compute `phenotype_instance_digest` over compiled behavior + genome digest + complete task/run/proposal/candidate/slot and frozen execution/evaluation bindings. Use `compiled_behavior_digest + evaluation_cohort_digest` for duplicate/cache identity; use phenotype-instance identity only for anti-crosswire event/artifact binding.

- [ ] **Step 6: Preserve legacy compilation exactly**

The compatibility wrapper calls the frozen legacy adapter with its historical knowledge snapshot (including `None`) and catalog. Assert complete `AlgorithmIR.to_dict()` equality, not merely predictor/parameter equality.

- [ ] **Step 7: Run compile, smoke, and security tests**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_workflow_ir.py \
  tests/test_algorithm_compilation.py \
  tests/test_algorithm_ir_smoke.py \
  tests/test_scientific_boundary.py -q
```

Expected: all pass, including the legacy golden equality test.

### Task 3: Persist seed materialization and bind behavior/instance identity through replay

**Files:**
- Modify: `src/ecologyrsi_dsh/core/models.py`
- Modify: `src/ecologyrsi_dsh/core/director.py`
- Modify: `src/ecologyrsi_dsh/core/state.py`
- Modify: `src/ecologyrsi_dsh/knowledge/algorithms.py`
- Modify: `src/ecologyrsi_dsh/evolution/analysis.py`
- Modify: `src/ecologyrsi_dsh/evolution/batches.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py`
- Modify: `tests/test_director_invariants.py`
- Modify: `tests/test_evolution_feedback_loop.py`
- Modify: `tests/test_training_trajectory.py`
- Create: `tests/test_genome_replay.py`

**Interfaces:**
- `RunSeedGenomeMaterialized` persists the complete canonical seed/migration genome before the first GenerationBatch.
- New protocol requires genome/source-behavior/compiled-behavior/phenotype-instance/compiler/registry/security/runtime-execution digests through Candidate, AlgorithmSpec/attempt, Artifact, Evaluation and Promotion.
- Evaluation also binds the exact `artifact_digest`.
- Historical protocols may omit all new fields.
- `RunState.persisted_genome_for(candidate_id)`, `materialized_seed_genome()`, `parent_genome_for_generation(generation)`, and read-only `projected_legacy_genome_for(candidate_id)` have non-overlapping semantics.

- [ ] **Step 1: Write protocol-aware replay and binding RED tests**

Cover historical missing-field replay, stage-aware required fields, canonical genome metadata, nested metadata detachment, same-genome/wrong-phenotype-instance rejection, behavior-identical sibling duplicate identity, wrong-artifact evaluation rejection, promotion binding, parent genome context, lineage/Proposal/Candidate coordinate checks, and immutable parent assets. `ProposalSubmitted` must not require candidate-dependent identity; after `CandidateSpawned`, the first algorithm attempt must. Add run creation/restart cases proving the full materialized seed—not only its digest—is persisted before GenerationBatch and a later catalog change cannot alter the first-generation parent. Add explicit legacy migration-seed cases; a projected legacy view alone cannot be a parent.

Required tests include:

- `test_new_protocol_rejects_all_none_genome_chain`;
- `test_legacy_protocol_accepts_missing_genome_fields`;
- `test_promotion_rejects_matching_genome_but_wrong_phenotype_instance`;
- `test_behavior_identical_siblings_dedupe_by_compiled_behavior_and_cohort`;
- `test_materialized_seed_canonical_json_precedes_first_generation_batch`;
- `test_restart_uses_persisted_seed_after_catalog_change`;
- `test_crash_between_run_created_and_seed_materialized_recovers_identically`;
- `test_partial_initialization_cannot_start_generation`;
- `test_proposal_does_not_require_candidate_dependent_instance_digest`;
- `test_projected_legacy_genome_cannot_be_promoted_or_resumed_without_migration_seed`.

- [ ] **Step 2: Run RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_genome_replay.py tests/test_director_invariants.py -q
```

Expected: missing fields/protocol enforcement.

- [ ] **Step 3: Add compatibility fields with protocol-aware requirements**

Serializers accept missing fields for historical execution protocols. New DSH-native requirements are event-specific: Proposal requires genome/source behavior; CandidateSpawned enables instance binding; the first AlgorithmAttempt and all later artifact/evaluation/promotion transitions require compiled behavior plus phenotype instance. Do not treat `None == None` as a valid binding, require a future-dependent field early, or rewrite historical events.

- [ ] **Step 4: Persist canonical genome identity safely**

Store `Proposal.metadata["evolution_genome_canonical_json"]` as an immutable canonical string, plus explicit genome/behavior digests. Parsing returns a detached immutable object. The proposal digest covers the canonical string. Before append, materialize the seed as a pure function. `RunCreated` stores the canonical template, all materialization inputs/version, expected full materialized canonical JSON/digest, and enters INITIALIZING. Append exactly one idempotent `RunSeedGenomeMaterialized` before READY/GenerationBatch. A crash in between is repaired only from the RunCreated payload; never from the current catalog. Duplicate/different seed events fail replay.

- [ ] **Step 5: Enforce the complete binding chain**

Before each transition verify:

```text
Proposal/Candidate genome
→ compiled behavior + phenotype instance/compiler/registry/security/runtime execution
→ artifact genome/compiled-behavior/phenotype-instance/artifact digest
→ evaluation genome/compiled-behavior/phenotype-instance/artifact digest
→ promotion evaluation/artifact/compiled-behavior/phenotype-instance/genome digest
```

Fail before training on compile identity mismatch and before promotion on any later mismatch.

- [ ] **Step 6: Extend GenerationBatch, lineage, and state accessors**

Implement `materialized_seed_genome()` and `parent_genome_for_generation()`. Freeze the complete parent genome plus stage context digests. Generation 0 must reference the one materialized seed event; later generations reference a persisted candidate genome frozen in their batch. Validate RunCreated expected seed, seed event, batch parent digest and returned canonical genome as one chain. Keep persisted, projected legacy, and migrated-seed access separate. Only persisted DSH-native genomes may be inherited; keep search parent distinct from formal incumbent.

- [ ] **Step 7: Run replay/lifecycle regressions**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_genome_replay.py \
  tests/test_director_invariants.py \
  tests/test_evolution_feedback_loop.py \
  tests/test_training_trajectory.py -q
```

Expected: new and historical event streams both pass with no implicit legacy promotion.

### Task 4: Make adaptive selection honest and add hierarchical fitness

**Files:**
- Create: `src/ecologyrsi_dsh/evaluators/fitness.py`
- Create: `src/ecologyrsi_dsh/core/exposure_registry.py`
- Modify: `src/ecologyrsi_dsh/core/ledger.py`
- Create: `tests/test_fitness.py`
- Create: `tests/test_scientific_exposure_registry.py`
- Modify: `src/ecologyrsi_dsh/evolution/promotion.py`
- Modify: `src/ecologyrsi_dsh/evolution/analysis.py`
- Modify: `src/ecologyrsi_dsh/core/models.py`
- Modify: `src/ecologyrsi_dsh/core/director.py`
- Modify: `tests/test_promotion.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- `FitnessProfile.from_task(task) -> FitnessProfile`.
- `build_fitness_assessment(evaluation, incumbent, runtime_metrics, profile) -> FitnessAssessment`.
- `assess_generation_selection(candidates, incumbent, profile) -> tuple[SelectionAssessment, ...]` whose evidence class is always `exploratory_adaptive_data`.
- `ScientificExposureRegistry` uses append-only, unique-keyed tables created by the versioned `EventLedger` schema; formal holdout APIs are completed in Task 5.
- Existing `assess_promotion_improvement()` remains a compatibility wrapper; under the new protocol it always emits `exploratory_adaptive_data`, and legacy `confidence_pass` is display-only and can never set validated/confirmed state.

- [ ] **Step 1: Write RED tests for known statistical failures**

Add deterministic fixtures with these named tests:

- `test_three_or_seven_blocks_are_insufficient_and_fail_closed`;
- `test_point_estimate_and_interval_use_identical_block_ids`;
- `test_bootstrap_draws_contiguous_three_day_blocks`;
- `test_total_blocks_without_four_legal_three_day_starts_are_insufficient`;
- `test_max_t_prevents_noise_winner_from_passing`;
- `test_selection_bound_is_labeled_exploratory_not_confidence`;
- `test_adaptively_reused_selection_rows_cannot_produce_confirmatory_promotion`;
- `test_reading_selection_reward_before_assessment_never_opens_a_formal_path`;
- `test_legacy_confidence_pass_cannot_validate_a_new_protocol_candidate`;
- `test_schema_five_database_migrates_without_changing_existing_events`;
- `test_efficiency_cannot_outrank_scientific_regression`;
- `test_average_improvement_with_one_bad_cell_fails_robustness_gate`;
- `test_existing_sample_reward_values_are_unchanged`.

- [ ] **Step 2: Run RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_fitness.py tests/test_promotion.py -q
```

Expected: failures for fail-open evidence, non-contiguous resampling, and missing fitness interfaces.

- [ ] **Step 3: Freeze one host-owned fitness profile**

Move effective coverage, minimum counts, practical delta, block method, exploratory max-T resample count/quantile, UQ requirements, efficiency references and evidence-class labels into one normalized `fitness_profile@1`. Freeze production `B=10_000` and the empirical 0.95 quantile index, but explicitly deny p-value, confidence, alpha-spending or confirmatory semantics for adaptive selection. Bind the profile digest into TaskManifest/evaluator contracts. Remove duplicate effective thresholds from loosely coupled metadata paths.

- [ ] **Step 4: Implement validity and robustness layers**

Validity requires matching digests/stage, complete finite target × horizon evidence, zero physical violations, overall and per-cell coverage, minimum per-cell samples/blocks, label isolation, and independent reviewer identity.

Selection robustness records minimum cell delta/exploratory stability floor and lower-quartile cell behavior. Formal stages use separately pre-registered LCBs. Preserve current per-cell no-regression as the initial hard rule.

- [ ] **Step 5: Replace the bootstrap implementation**

Use every paired 24h block in private computation. A valid three-day start is three consecutive calendar-day block IDs in the frozen calendar encoding with no intervening timestamp gap. Selection needs at least four valid starts. Draw non-circular valid sequences, concatenate until the original block count and truncate; never wrap or treat rotating-window neighbors as temporal neighbors. Use the same sequence indices for candidate, incumbent and all siblings. Report the exact block-ID/resample-policy digest. Remove the first-128-block computation truncation; bound only public serialization.

- [ ] **Step 6: Add sibling max-T as exploratory stability ranking only**

Include all predeclared same-cohort siblings with computable primary paired scores in one family; invalid/non-computable siblings deterministically rank below computable candidates and cannot be replaced after labels are seen. For each resample compute `M_r=max_k(delta_star[r,k]-delta_hat[k])`, `q=sorted(M)[ceil(0.95*B)-1]`, and `selection_stability_floor_k=delta_hat[k]-q`. Use `delta_hat>0.005` plus a positive stability floor only as an internal selection rule. Persist `evidence_class=exploratory_adaptive_data`; this transition may update search parent/selection incumbent but can never set validated/confirmed status. Fewer than eight paired day blocks returns `insufficient_evidence`. Reusing rows or creating a new run never changes the evidence class.

- [ ] **Step 7: Use a lexicographic ranking key**

Replace the current scalar-first `_rank_key()` with the ordered layers from the spec. Runtime efficiency fields are read only after all required scientific layers pass.

- [ ] **Step 8: Run focused scientific regressions**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_fitness.py \
  tests/test_promotion.py \
  tests/test_objectives.py \
  tests/test_baselines.py \
  tests/test_evaluation.py \
  tests/test_evolution_feedback_loop.py \
  tests/test_sample_results_contract.py \
  tests/test_sample_execution.py \
  tests/test_horizon_feedback.py \
  tests/test_director_invariants.py \
  tests/test_scientific_exposure_registry.py \
  tests/test_core.py -q
```

Expected: all pass and hand-checked legacy reward tests remain unchanged.

### Task 5: Enforce stage-scoped data views and the four-stage protocol

**Files:**
- Modify: `src/ecologyrsi_dsh/data/splits.py`
- Modify: `src/ecologyrsi_dsh/data/registry.py`
- Modify: `src/ecologyrsi_dsh/data/contracts.py`
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py`
- Modify: `src/ecologyrsi_dsh/core/models.py`
- Modify: `src/ecologyrsi_dsh/core/director.py`
- Modify: `src/ecologyrsi_dsh/core/exposure_registry.py`
- Create: `tests/test_formal_stage_protocol.py`
- Modify: `tests/test_datasets.py`
- Modify: `tests/test_delivery_contract.py`

**Interfaces:**
- `DatasetRegistry.selection_view(...) -> SelectionDatasetView`.
- Legacy-only `DatasetRegistry.series(..., execution_protocol=...)` rejects new protocols.
- `DataProtocol@time-forward-four-stage@2` with calibration-fit/calibration-UQ/model-selection/validation/final-test/external roles and digest.
- Single-use `FormalStageToken` bound to run/stage/candidate/artifact/genome/statistical-plan/partition digests.
- `ScientificExposureRegistry.reserve_formal_stage(raw_holdout_key, objective_family_digest, plan_digest, idempotency_key)` where the raw unique key excludes objective identity.
- Run-level `selection_incumbent_id`, `validated_candidate_id`, `final_test_candidate_id`, and irreversible stage seals.

- [ ] **Step 1: Write leakage and seal RED tests**

Prove that the typed selection view cannot contain development/gate rows; new protocol cannot call legacy full `series()`; changing development/gate values leaves all selection outputs/digests unchanged; stage tokens are single-use and artifact/run scoped; development cannot open before one candidate/artifact/statistical plan is locked; development results cannot enter new proposals; gate cannot open twice; external reference cannot be selected as an evolution episode; and a new run cannot reuse an exposed holdout as independent. Changing target, horizon, baseline, score/statistical plan, fitness profile, or objective-family digest must still collide on the same raw holdout exposure key.

Add deterministic split tests for exact 70% calibration cut, half-open target-timestamp membership, both 24h embargo ranges, fixed calendar/timezone basis, every partition/embargo digest, and fail-closed short data. Add `test_adaptively_reused_selection_rows_cannot_produce_confirmatory_promotion` at the stage boundary.

Required exposure tests include `test_look_is_reserved_before_any_current_cohort_metric_is_read` (for the locked formal holdout), `test_objective_change_cannot_reopen_same_development_partition`, `test_fitness_or_statistical_plan_change_cannot_reopen_same_gate_partition`, and `test_raw_holdout_key_excludes_objective_family`.

- [ ] **Step 2: Run RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_formal_stage_protocol.py tests/test_datasets.py -q
```

Expected: missing typed stage views, exposure registry rules, and formal-stage state.

- [ ] **Step 3: Raise the fit→feedback embargo to 24 hours**

Create split/data protocol `time-forward-four-stage@2`; keep historical `@1` with its one-hour embargo replayable. Implement the spec literally: distinguish source timezone `unspecified-naive-local` from deterministic `excel-serial-hour-fixed-24h@1` calendar encoding and never call the source UTC; use `raw_cut=a+floor(0.70*(b-a))`; half-open calibration-fit; lower-bound 24h embargo start for calibration-UQ; another lower-bound 24h start for model-selection; target-timestamp membership; and explicit embargo ranges/digests. Before execution require calibration-fit `>=80` eligible, label-complete, history-constructible rows/`>=14` day blocks and calibration-UQ `>=40` such rows/`>=8` day blocks in every target × horizon cell; prediction-success coverage is checked only after execution. Any missing boundary or short cell fails run creation; never search for a more favorable cut.

- [ ] **Step 4: Implement minimal stage views**

All new evaluators require a typed stage view. Point models, baselines and normalization scales use calibration-fit. Calibration-UQ exposes only locked-artifact residual calibration inputs. Selection receives no formal rows. Validation/final-test raw rows are available only inside the formal evaluator under a single-use token; normal code receives aggregates.

- [ ] **Step 5: Implement irreversible formal-stage events**

Add structured stage-frozen/completed/sealed evidence. Use a unique raw holdout key over dataset/split/episode/stage/partition only, plus a separate analysis-family key that adds objective-family digest and full plan digest. Reserve both atomically, but let only the raw key authorize opening. Objective/target/horizon/baseline/profile/statistical-plan changes can never reopen the same raw partition. Failure or insufficient evidence seals the exposure; a new run using the same exposure is exploratory only. A successful gate changes report status only and never modifies a genome or incumbent.

- [ ] **Step 6: Run data, leakage, and replay regressions**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_formal_stage_protocol.py \
  tests/test_datasets.py \
  tests/test_scientific_boundary.py \
  tests/test_delivery_contract.py \
  tests/test_core.py \
  tests/test_evaluation.py \
  tests/test_greenhouse_prediction.py \
  tests/test_execution_projection.py \
  tests/test_scientific_exposure_registry.py -q
```

Expected: all pass; selection is structurally unable to access later partitions.

### Task 6: Refactor and secure the Cordis plugin runtime foundation

**Files:**
- Modify: `integrations/dsh_ecology_plugin/lib/index.js`
- Create: `integrations/dsh_ecology_plugin/lib/config.js`
- Create: `integrations/dsh_ecology_plugin/lib/security.js`
- Create: `integrations/dsh_ecology_plugin/lib/sidecar/client.js`
- Create: `integrations/dsh_ecology_plugin/lib/web/static.js`
- Create: `integrations/dsh_ecology_plugin/lib/web/proxy.js`
- Create: `integrations/dsh_ecology_plugin/lib/runtime/capabilities.js`
- Create: `integrations/dsh_ecology_plugin/lib/runtime/controller.js`
- Create: `integrations/dsh_ecology_plugin/lib/runtime/routes.js`
- Create: `integrations/dsh_ecology_plugin/lib/runtime/run-registry.js`
- Modify: `integrations/dsh_ecology_plugin/package.json`
- Modify: `integrations/dsh_ecology_plugin/test/proxy_security.mjs`
- Create: `integrations/dsh_ecology_plugin/test/capabilities.test.mjs`
- Create: `integrations/dsh_ecology_plugin/test/runtime_api_security.test.mjs`
- Create: `integrations/dsh_ecology_plugin/test/sidecar_client.test.mjs`

**Interfaces:**
- `runtimeCapabilities(ctx, presetCatalog) -> RuntimeCapabilities`.
- Loopback-only `/api/ecology-agent-runtime/v1/*` control API.
- Bounded `SidecarClient` with `AbortSignal`, explicit timeouts, size limits, and stable error mapping.

- [ ] **Step 1: Write Node RED tests**

Use `node:test` and fake Cordis services. Test missing root services, advisory vs resolved model status, non-loopback requests, missing/wrong control token, distinct revision fields, body overflow, timeout, unknown fields, forbidden methods, Origin/Sec-Fetch rejection, runtime-prefix proxy rejection, and token scans of bodies/headers/errors.

- [ ] **Step 2: Run RED**

Run:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/capabilities.test.mjs \
  integrations/dsh_ecology_plugin/test/runtime_api_security.test.mjs \
  integrations/dsh_ecology_plugin/test/sidecar_client.test.mjs
```

Expected: module import failures.

- [ ] **Step 3: Split static, proxy, config, and security code**

Preserve existing public routes. Add strict method/path allowlists, byte limits, connect/response/total timeouts, abort propagation, response-header filtering, and literal loopback hosts. Keep the current proxy test green after each extraction.

- [ ] **Step 4: Implement truthful capability reporting**

Probe root services `agents`, `sessions`, `tokenMeter`, `subagents`, `tools`, `sessionPersistence`, `sessionProjections`, `agentPresets`, and `llm`. Without creating a probe Agent, report `declared`, fake/tested `preset_mountable`, standing-key `tool_surface_verified`, and `route_resolvable`. Define but do not synthesize `live_agent_service_ready` or `first_call_verified`; Task 7/real stage calls are their only authorities.

- [ ] **Step 5: Add the internal control API**

Use `ECOLOGYRSI_DSH_RUNTIME_TOKEN` only for Python→DSH control. Reserve a separate `ECOLOGYRSI_SIDECAR_TOOL_TOKEN` for DSH→Python ecology tools. Neither token may use the browser proxy, client bundle, query string, or public context. Every write schema uses distinct `run_state_revision`, `stage_attempt`, `ledger_expected_revision`, and `idempotency_key` fields.

- [ ] **Step 6: Pin direct runtime dependencies**

Move required imported helpers to exact direct dependencies:

```json
{
  "@deepseek-ai/dsh-llm": "0.1.0-rc.6",
  "@deepseek-ai/dsh-subagent": "0.1.0-rc.6",
  "@deepseek-ai/dsh-tools": "0.1.0-rc.6",
  "@deepseek-ai/schemastery": "3.18.1"
}
```

Pin every imported DSH prerelease package at `0.1.0-rc.6`; do not use caret ranges for rc packages. Keep peer dependencies as compatibility declarations only.
Declare exact peer contracts for the public services actually injected, including Host Webserver, Agent, Session, Token Meter, Session Projections, Subagent, Tools, Session Persistence and Agent Presets; role preset packages provide Compaction and Workflow in their isolated realms.

- [ ] **Step 7: Run Node security regressions**

Run:

```bash
node integrations/dsh_ecology_plugin/test/proxy_security.mjs
node --test integrations/dsh_ecology_plugin/test/*.test.mjs
```

Expected: all pass without a real model call.

### Task 7: Install role-isolated DSH presets and lifecycle control

**Files:**
- Create: `integrations/dsh_ecology_plugin/presets/ecology-coordinator-v1/preset.yml`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-coordinator-v1/agent.cordis.yml`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-researcher-v1/preset.yml`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-researcher-v1/agent.cordis.yml`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-candidate-proposer-v1/preset.yml`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-candidate-proposer-v1/agent.cordis.yml`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-sample-planner-v1/preset.yml`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-sample-planner-v1/agent.cordis.yml`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-sample-critic-v1/preset.yml`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-sample-critic-v1/agent.cordis.yml`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-generation-judge-v1/preset.yml`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-generation-judge-v1/agent.cordis.yml`
- Create: `integrations/dsh_ecology_plugin/lib/runtime/agents.js`
- Create: `integrations/dsh_ecology_plugin/lib/runtime/child-bindings.js`
- Create: `integrations/dsh_ecology_plugin/lib/runtime/workflows.js`
- Create: `integrations/dsh_ecology_plugin/lib/tools/agent-plugin.js`
- Create: `scripts/install_dsh_ecology_runtime.mjs`
- Create: `integrations/dsh_ecology_plugin/test/agent_lifecycle.test.mjs`
- Create: `integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs`
- Create: `integrations/dsh_ecology_plugin/test/child_binding_race.test.mjs`
- Create: `integrations/dsh_ecology_plugin/test/package_contract.test.mjs`
- Create: `integrations/dsh_ecology_plugin/test/preset_realms.test.mjs`
- Create: `integrations/dsh_ecology_plugin/test/install_runtime.test.mjs`
- Create: `integrations/dsh_ecology_plugin/test/token_semantics.test.mjs`

**Interfaces:**
- `createRoleAgent(binding) -> AgentHandle`.
- `allocateChildLaunches(stageBinding, boundedItems) -> tuple[DurableChildLaunch, ...]` uses a Python-ledger CAS to persist monotonic launch attempts before Host start.
- `reserveChildBinding(parentSessionId, durableLaunch, frozenBinding) -> ChildBindingReservation` installs the ledger-issued, never-reused safe label in the Host before child start.
- `claimChildBinding(execAgent) -> FrozenChildBinding` derives parent from `execAgent.session.header.parentSession` and label from `foldSubagentDescriptor(session events)` on the first guarded tool call.
- `openStageActivationLease(childIdentity, revision, stageAttempt, waveDigest, idempotencyKey) -> StageActivationLease` is durable and single-turn; continuable followups rotate it only after idle/flush.
- `startTrackedSubagent(kind, request, binding) -> PendingChildStart` synchronously registers an `AbortController` and start Promise before awaiting DSH.
- `startHomogeneousWorkflow(roleHost, compiledSpec, items) -> WorkflowRun`.
- `cancelAndQuiesce(runBinding, mode) -> Promise<void>`.
- Installer atomically installs a packed plugin dependency closure, exact versioned presets, and an id-targeted Web Profile patch.

- [ ] **Step 1: Write lifecycle RED tests**

Assert atomic setup before Agent publication, unpredictable Session IDs, exact preset/model/preset-content/tool-surface binding, single-flight creation, standing-scope `restrict({allow: []})`, no tool-workflow/general subagent/Bash/FS/Web/MCP, fixed-script non-interpolation, structured args size/schema, `maxTotalAgents`/`maxConcurrent`/`maxItems`/sync timeout, context-pressure vs projected-usage semantics, and handle cleanup on Cordis teardown. Add a race fake where the child calls a tool before the start Promise/lifecycle event returns: a pre-registered binding must authorize exactly once using `session.header.parentSession` plus folded descriptor label. Test that same business idempotency with a new durable launch attempt gets a different label, process loss cannot reset the counter or reproduce any old label, and missing/wrong/reused/revoked/terminal labels fail closed while tombstones are never overwritten. Add cancellation while one-shot and continuable start Promises are unresolved: pending controllers abort, Promises settle, returned one-shot runs dispose, accepted continuables interrupt/drain, and clean completion is impossible while any pending start remains.

- [ ] **Step 2: Run RED**

Run:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/agent_lifecycle.test.mjs \
  integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs \
  integrations/dsh_ecology_plugin/test/child_binding_race.test.mjs \
  integrations/dsh_ecology_plugin/test/package_contract.test.mjs \
  integrations/dsh_ecology_plugin/test/preset_realms.test.mjs \
  integrations/dsh_ecology_plugin/test/install_runtime.test.mjs \
  integrations/dsh_ecology_plugin/test/token_semantics.test.mjs
```

Expected: missing preset and lifecycle modules.

- [ ] **Step 3: Add exact role presets**

Use the exact legal preset IDs and matching directory basenames `ecology-coordinator-v1`, `ecology-researcher-v1`, `ecology-candidate-proposer-v1`, `ecology-sample-planner-v1`, `ecology-sample-critic-v1`, and `ecology-generation-judge-v1`. rc.6 preset IDs match `[a-z0-9][a-z0-9-]*`. Create and export a minimal agent-only plugin subpath that clears inherited tools; Task 9 adds role tools to it. Each preset loads only this agent subpath, its persona, Compaction, and when needed `dsh-workflow-worker-thread`; it never loads the Host web entry, `dsh-tool-workflow`, a general subagent tool, Bash, FS, Web, arbitrary MCP or Ask User.

- [ ] **Step 4: Implement Agent creation and cancellation**

Retain every `AgentHandle`. Do not equate `ctx.agents.get()` with a disposable handle. Agent creation omits `maxTokens`. After preset mount, materialize and flush a public rc.6-known preset-selection Session boundary so an otherwise empty role-host is resumable; never append an unknown custom event. Verify `serviceFor(agent, "compaction")` and required `serviceFor(agent, "workflowEngine")` before setting `live_agent_service_ready=true`.

- [ ] **Step 5: Implement fixed homogeneous Workflow templates**

Use host-authored scripts selected by compiled template ID. Never interpolate structured args into source. Subscribe to child lifecycle events before start, use ledger-issued item/idempotency/launch-attempt labels, store `WorkflowRun` handles in memory, persist child Session mappings promptly, and always call `dispose()`.

For a one-shot, ask the Python ledger to CAS-allocate one durable launch attempt plus one single-turn activation lease before `start()`. The domain-separated reservation digest includes run, stage, role, item, business idempotency and launch attempt. Before calling either asynchronous `subagents.start()` or `startContinuable()`, synchronously register `PendingChildStart` with an `AbortController`, pass its signal in the request, then attach/await the Promise; do not wait for a run/childId to begin tracking. `SubagentRun` has no abort method: cancellation aborts the pending signal, awaits settlement, then disposes a returned one-shot run; an accepted continuable is interrupted/drained by childId. Workflow Worker has no Host per-child pre-start hook, so before synchronous `workflowEngine.start()` allocate attempts, identity reservations, and activation leases for the **entire bounded batch** in one operation; the fixed script receives only those labels. The standing guard claims immutable child identity from parent header + folded label; every tool call also needs the current activation lease. Persist recovery mappings as `(parentSessionId,label)→childSessionId`. Start failure/unpublished items/cancel revoke durable reservations/leases; normal completion marks terminal, and tombstones remain permanent. Retain every Workflow and settled one-shot SubagentRun; for continuable children retain `{childId,messageId}` plus the role-host `AgentHandle` because rc.6 continuation manager owns the child handle.

- [ ] **Step 6: Implement a safe installer**

The installer resolves a trusted non-symlink DSH home from DSH configuration/`DSH_HOME`/`os.homedir()` and never hardcodes a user path. It packs the plugin to a versioned npm tgz, copies it to a stable DSH-local cache, and runs `dsh plugin --profile web add --save-exact file:<absolute-tgz>` so package/lock contain real dependencies. It backs up patch/package/lock and same-name presets, uses temporary files + fsync + atomic rename, rolls back every target on failure, refuses a drifting same-name preset, is idempotent, and never reads or scans `.credentials.yaml`.

The patch injection is exactly:

```text
webServer, agents, sessions, tokenMeter, subagents,
tools, sessionPersistence, sessionProjections, agentPresets, llm
```

After installation verify plugin Host and agent-plugin subpath imports, `dsh --profile web --dump-config`, exact preset basename/ID, standing tool schemas, live realm services, and absence of forbidden tools.

- [ ] **Step 7: Run all lifecycle/package tests**

Run:

```bash
node --test integrations/dsh_ecology_plugin/test/*.test.mjs
```

Expected: all pass with fake services.

### Task 8: Add the Python DSH-native client and hard no-fallback gate

**Files:**
- Create: `src/ecologyrsi_dsh/integrations/dsh_native_runtime.py`
- Create: `tests/test_dsh_native_runtime.py`
- Modify: `src/ecologyrsi_dsh/api/handler.py`
- Modify: `src/ecologyrsi_dsh/api/transport.py`
- Modify: `src/ecologyrsi_dsh/api/auto_progress.py`
- Modify: `src/ecologyrsi_dsh/evolution/strategies.py`
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py`
- Modify: `src/ecologyrsi_dsh/integrations/__init__.py`
- Modify: `src/ecologyrsi_dsh/core/errors.py`
- Modify: `tests/test_runtime_integration.py`
- Modify: `tests/test_http.py`

**Interfaces:**
- `DshNativeAgentRuntimeClient.capabilities()`.
- `create_run(binding)` / `run_stage(request)` / `cancel(run_id)` / `resume(run_id)` / `status(run_id)`.
- No completion/chat/sample method exists.

- [ ] **Step 1: Write client and no-fallback RED tests**

Use a local fake HTTP server. Test strict response schemas, distinct run-state/stage-attempt/ledger revisions, idempotency fields, timeout/abort, stable error codes, token redaction, and capability mismatch. Monkeypatch `ModelGateway` network entry points to raise and prove a new DSH-native run never invokes them.

- [ ] **Step 2: Run RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_dsh_native_runtime.py tests/test_runtime_integration.py -q
```

Expected: missing client/protocol failures.

- [ ] **Step 3: Implement the narrow client**

Use Python standard-library HTTP only. Require literal loopback host, control token, bounded JSON, timeouts, stable error decoding, and response redaction. Do not import model binding credentials or `ModelGateway`.

- [ ] **Step 4: Freeze the new runtime protocol during run creation**

New autonomous requests require declared/mountable/tool-verified presets, live role-host Compaction/Workflow services, resolvable frozen policy/review route configs, matching preset/tool/registry/security/data/stage digests, and a resumable flushed role-host boundary. `first_call_verified` starts false and is set only by a successful real stage. Add a typed runtime-unavailable error and map it to HTTP `503 dsh_native_runtime_unavailable`; create no scientific run if setup fails.

- [ ] **Step 5: Remove new-run Token budgets**

The new request/TaskManifest path does not emit `token_limit`, `token_reservation_per_wave`, operation `max_tokens`, or truncation retry Token configuration. Keep old properties/parsers for historical event replay only.

- [ ] **Step 6: Enforce import/path separation**

New DSH-native strategies/evaluators receive only `DshNativeAgentRuntimeClient`. Legacy gateway adapters remain under explicit `legacy_execution_protocol` guards and cannot be selected for a new run.

- [ ] **Step 7: Run runtime/API regressions**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_dsh_native_runtime.py \
  tests/test_runtime_integration.py \
  tests/test_http.py \
  tests/test_model_gateway.py -q
```

Expected: DSH-native tests pass and legacy read/diagnostic tests remain green.

### Task 9: Register role tools and structured stage contracts

**Files:**
- Create: `integrations/dsh_ecology_plugin/lib/tools/definitions.js`
- Create: `integrations/dsh_ecology_plugin/lib/tools/roles.js`
- Modify: `integrations/dsh_ecology_plugin/lib/tools/agent-plugin.js`
- Create: `integrations/dsh_ecology_plugin/schemas/stage-context.schema.json`
- Create: `integrations/dsh_ecology_plugin/schemas/research-result.schema.json`
- Create: `integrations/dsh_ecology_plugin/schemas/genome-mutation.schema.json`
- Create: `integrations/dsh_ecology_plugin/schemas/sample-wave.schema.json`
- Create: `integrations/dsh_ecology_plugin/schemas/sample-decisions.schema.json`
- Create: `integrations/dsh_ecology_plugin/schemas/sample-review.schema.json`
- Create: `integrations/dsh_ecology_plugin/schemas/generation-summary.schema.json`
- Create: `integrations/dsh_ecology_plugin/schemas/generation-review.schema.json`
- Create: `integrations/dsh_ecology_plugin/test/role_tools.test.mjs`
- Create: `integrations/dsh_ecology_plugin/test/tool_authorization.test.mjs`
- Create: `src/ecologyrsi_dsh/api/dsh_tools.py`
- Create: `tests/test_dsh_tool_contracts.py`
- Modify: `src/ecologyrsi_dsh/server.py`
- Modify: `src/ecologyrsi_dsh/api/handler.py`
- Modify: `integrations/dsh_ecology_plugin/package.json`

**Interfaces:**
- Versioned schemas for stage context, research result, genome mutation, sample wave/decision/review, generation summary/review.
- DSH→Python internal tool endpoint authenticated only by `ECOLOGYRSI_SIDECAR_TOOL_TOKEN`.
- Every sidecar envelope carries Host-bound Agent/Session, child-identity reservation, current activation-lease ID/epoch, run, role, stage, `run_state_revision`, `stage_attempt`, `ledger_expected_revision`, idempotency key, genome, compiled-behavior, and phenotype-instance digests; these identity fields are not model arguments.

- [ ] **Step 1: Write cross-language schema RED tests**

Load the same JSON schemas in Node and Python. Test exact role tool sets, unknown fields, cross-run/role/Session/revision calls, repeated idempotency keys, output validation, blocked label field names, `restrict({allow: []})` behavior, scope-local role tool visibility, direct execute denial, and Host-bound identity that model arguments cannot override.

- [ ] **Step 2: Run RED**

Run:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/role_tools.test.mjs \
  integrations/dsh_ecology_plugin/test/tool_authorization.test.mjs
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_dsh_tool_contracts.py -q
```

Expected: missing schemas/endpoints.

- [ ] **Step 3: Add versioned schemas with `additionalProperties: false`**

Research/proposal/judge use one-shot DSH output schemas and have no model-callable submit tools. Planner/critic schemas support bounded wave submissions. Schema IDs and digests enter the compiler registry.

- [ ] **Step 4: Implement role registration plus execution guard**

Export `lib/tools/agent-plugin.js` as an independent package subpath and load it from presets; never load Host `lib/index.js` in a role realm. In the preset standing scope, clear inherited tools with `restrict({allow: []})`, register only role-local tools, and add an execution guard. For child Agents the guard derives parent from `exec.agent.session.header.parentSession`, folds the known `subagent/descriptor` events for the never-reused launch label, atomically claims the pending reservation to `exec.agent.id`, and then checks the Host closure binding; lifecycle start events are too late to authorize the first call. Call `exec.concludeTurn()` only for planner/critic after Python durable acceptance.

- [ ] **Step 5: Implement Python state/idempotency validation**

Validate the current ledger state before every operation. A repeated identical idempotency key returns the original digest; a repeated key with different input fails deterministically. Planner-facing payload construction excludes observed/label/ground truth recursively.

- [ ] **Step 6: Run role/security contracts**

Run:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/role_tools.test.mjs \
  integrations/dsh_ecology_plugin/test/tool_authorization.test.mjs
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_dsh_tool_contracts.py \
  tests/test_scientific_boundary.py \
  tests/test_recovery_security.py -q
```

Expected: all pass.

### Task 10: Migrate research, genome proposal, and generation judge to DSH

**Files:**
- Modify: `src/ecologyrsi_dsh/evolution/strategies.py`
- Modify: `src/ecologyrsi_dsh/knowledge/research_iteration.py`
- Modify: `src/ecologyrsi_dsh/evolution/batches.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py`
- Modify: `src/ecologyrsi_dsh/core/director.py`
- Create: `integrations/dsh_ecology_plugin/lib/runtime/structured-roles.js`
- Create: `integrations/dsh_ecology_plugin/test/structured_roles.test.mjs`
- Create: `tests/test_dsh_structured_roles.py`
- Modify: `tests/test_research_iteration.py`
- Modify: `tests/test_strategy_router.py`
- Modify: `tests/test_judge_persistence.py`

**Interfaces:**
- DSH stages `generation.research`, `candidate.propose`, and `generation.judge`.
- `runStructuredRole(roleHost, reservedBinding, requestWithOutputSchema) -> SubagentResult.structured` registers a pending start/AbortController before calling DSH, retains and disposes the settled `SubagentRun`, and performs Host-owned idempotent persistence through the same admission fence as tool submissions.
- Proposer returns `GenomeMutation@1`, not arbitrary `Proposal.changes` or code.
- Python applies/compiles the mutation and mirrors compatible scientific parameters to legacy `Proposal.changes` for old projections.

- [ ] **Step 1: Write structured-role RED tests**

Fake the DSH control API and return valid/invalid structured results. In Node, test that an undriven role-host pre-registers a child binding, starts a one-shot spawn child with output schema, authorizes a tool call made before the start Promise resolves, retains/disposes the `SubagentRun`, reads only `SubagentResult.structured`, exposes no `submit_*` tool, ignores free text, and persists idempotently only while admission remains open. Wrong descriptor labels and cancellation between child completion and Host persistence must fail closed. In Python test schema failures, parent genome/mutation-context mismatch, reviewer Session reuse, route drift, retry idempotency, child genome attempting to select its own proposer/judge, and absence of any ModelGateway call.

- [ ] **Step 2: Run RED**

Run:

```bash
node --test integrations/dsh_ecology_plugin/test/structured_roles.test.mjs
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_dsh_structured_roles.py -q
```

Expected: current strategies still call the gateway and tests fail.

- [ ] **Step 3: Migrate shared generation research**

Use the frozen parent genome reproduction program to configure the researcher role-host, then start a fresh one-shot child with `outputSchema`. For generation 0 the parent must be the canonical materialized seed loaded from `RunSeedGenomeMaterialized`, never the current seed catalog. Pass one frozen redacted generation context, persist its structured result through the Host, and keep it shared across sibling candidates. The child genome does not yet exist and cannot configure this stage.

- [ ] **Step 4: Migrate candidate proposal to mutation**

Use the same frozen parent reproduction program for every proposer child. Each slot receives the same parent/context envelope plus `GenomeMutationContextV1` slot/seed/budget identity. Host persists the structured mutation, Python applies/compiles it, and duplicate `compiled_behavior_digest + evaluation_cohort_digest` candidates are rejected before training even when their lineage/genome/phenotype-instance digests differ. A child's reproduction program becomes active only if it is next generation's search parent.

- [ ] **Step 5: Migrate independent generation judge**

Create a fresh one-shot judge child under the fixed `selection_reviewer_program`, never a candidate program. It receives only aggregate redacted scientific evidence. Its structured result is Host-persisted; it can reject a scientific pass but cannot override a scientific failure or modify fitness.

- [ ] **Step 6: Persist DSH stage evidence**

Append `DshRuntimeBound`, stage started/submission/completed/paused/failed events with stable IDs, digests, stop reasons, and Session references. Do not store complete conversation content.

- [ ] **Step 7: Run research/proposal/judge regressions**

Run:

```bash
node --test integrations/dsh_ecology_plugin/test/structured_roles.test.mjs
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_dsh_structured_roles.py \
  tests/test_research_iteration.py \
  tests/test_strategy_router.py \
  tests/test_judge_persistence.py \
  tests/test_genome_replay.py -q
```

Expected: all new-run role actions use the fake DSH runtime only.

### Task 11: Migrate sample planner/critic and enforce sibling context fairness

**Files:**
- Create: `src/ecologyrsi_dsh/evaluators/dsh_sample_adapter.py`
- Create: `tests/test_dsh_sample_execution.py`
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py`
- Modify: `src/ecologyrsi_dsh/evaluators/sample_execution.py`
- Modify: `src/ecologyrsi_dsh/evaluators/shared_sample_context.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py`
- Modify: `tests/test_sample_execution.py`
- Modify: `tests/test_shared_sample_context.py`
- Modify: `tests/test_recoverable_evaluation_integration.py`

**Interfaces:**
- `DshSampleCollaborationAdapter` implements the existing sample collaboration boundary without model gateway methods.
- Planner may use a continuable Session across waves; each critic wave uses a fresh independent role Session.
- Every sibling starts from the same `stage_context_digest` with candidate-specific genome as the only intentional difference.

- [ ] **Step 1: Write sample and fairness RED tests**

Test valid planner→registered tool→critic behavior, no-label recursive scans, planner Session continuity, critic freshness, identical sibling context digest, no sibling result lookup, completion-order invariance, cancellation, and no `max_tokens`/Token reservation fields. Multi-wave tests require immutable child identity plus a distinct durable `StageActivationLease` per followup, previous-turn idle/flush before rotation, pause closing the old lease, resume issuing a new lease, and late old-turn tools/results being rejected.

- [ ] **Step 2: Run RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_dsh_sample_execution.py tests/test_shared_sample_context.py -q
```

Expected: missing DSH sample adapter and context checks.

- [ ] **Step 3: Implement planner wave execution**

Use a role-host Agent or continuable subagent for planner continuity. Execute only registered prediction tools. Descriptor-claim one immutable child identity before its first tool call. Before every followup, after proving the previous turn idle and flushing its Session, CAS-open one single-use `StageActivationLease` bound to candidate/genome/compiled-behavior/phenotype-instance/cohort, wave digest, idempotency key, run-state revision and stage attempt. The Host closure supplies the active lease identity and stores structured action/tool digests; never mutate/reclaim the descriptor binding to advance a wave.

- [ ] **Step 4: Implement independent critic execution**

Start a fresh critic role Session under the fixed reviewer program for each bounded review unit. It receives predictions and constraint summaries but not observed labels. Candidate genomes cannot change critic preset/instruction/tools/model. Host adjudication remains deterministic.

- [ ] **Step 5: Implement homogeneous fan-out and bounded concurrency**

Planner fan-out and critic fan-out are separate Workflows/role hosts. Host Controller sequences them. Concurrency comes from the frozen execution plan and DSH Workflow limits, never from a Token budget.

- [ ] **Step 6: Freeze and verify sibling stage contexts**

Compute one candidate-independent context digest from GenerationBatch, parent genome, knowledge/research, cohort, fitness profile, compiler and registry. Inject it into fresh Sessions. Reject any sibling request whose digest differs.

- [ ] **Step 7: Run sample and evaluation regressions**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_dsh_sample_execution.py \
  tests/test_sample_execution.py \
  tests/test_shared_sample_context.py \
  tests/test_recoverable_evaluation_integration.py \
  tests/test_sample_results_contract.py -q
```

Expected: all pass; reward outputs remain byte-for-byte compatible for the same predictions.

### Task 12: Implement restart reconciliation, cancellation, and protocol recovery

**Files:**
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/controller.js`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/run-registry.js`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/child-bindings.js`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/workflows.js`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/structured-roles.js`
- Create: `integrations/dsh_ecology_plugin/test/resume.test.mjs`
- Create: `integrations/dsh_ecology_plugin/test/workflow_reconciliation.test.mjs`
- Create: `integrations/dsh_ecology_plugin/test/cancel_race.test.mjs`
- Modify: `src/ecologyrsi_dsh/integrations/dsh_native_runtime.py`
- Modify: `src/ecologyrsi_dsh/api/dsh_tools.py`
- Modify: `src/ecologyrsi_dsh/api/auto_progress.py`
- Modify: `src/ecologyrsi_dsh/core/state.py`
- Create: `tests/test_dsh_reconciliation.py`
- Create: `tests/test_dsh_cancel_race.py`
- Modify: `tests/test_run_cleanup.py`

**Interfaces:**
- `resumeRoleAgent(session_id, frozen_binding) -> AgentHandle` after persistence inspection.
- `reconcileStage(stage_binding, durable_accepted_items, session_items) -> ReconciliationPlan`.
- `closeStageAdmission(run_id, run_state_revision, stage_attempt) -> AdmissionFence` before DSH cancellation; it gates both role-tool submissions and Host-owned structured-result acceptance.
- `DshWorkflowReconciled` event binds old interrupted Workflow ID, new Workflow ID, completed idempotency keys, and remaining item digest.

- [ ] **Step 1: Write crash/restart RED tests**

Simulate process loss after zero, some, and all Python-durable submissions. Test exact Session/preset/model/preset-content/tool-surface lineage validation, corrupt persistence, missing frozen route, duplicate submission, an empty role-host crash/resume boundary, DSH-only child completion, pre-start reservation/descriptor claim and safe child labels/mappings, process-loss-safe launch attempts, cancel during compaction, late tool submissions, structured result completing just after admission closure, one-shot/continuable cancellation while start Promises are unresolved, continuable planner interruption, stage-activation rotation across waves and pause/resume, descendant draining, wedged start/disposal timeout, and absence of a recoverable Workflow handle.

- [ ] **Step 2: Run RED**

Run:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/resume.test.mjs \
  integrations/dsh_ecology_plugin/test/workflow_reconciliation.test.mjs \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_dsh_reconciliation.py tests/test_dsh_cancel_race.py -q
```

Expected: missing reconciliation behavior.

- [ ] **Step 3: Resume Agents from exact persisted identity**

Inspect `sessionPersistence` header/lineage/preset/log boundary, including the rc.6-known preset-selection event flushed at creation, then call `ctx.agents.resume()` with the same setup and exact frozen route/preset/tool digests. Never append unknown Session event types or substitute a default/same-name model.

- [ ] **Step 4: Reconcile—not resume—Workflows**

Before a one-shot start, or before `workflowEngine.start()` for **all** bounded Workflow items, ask the Python ledger to CAS-allocate fresh durable launch attempts and install every never-reused parentSession+safe-label reservation; the Worker has no per-child Host hook. Subscribe to child start/end and persist the claimed `(parentSessionId,label)→childSessionId` mapping for recovery. Mark the old Workflow interrupted and tombstone all old reservations and activation leases. Treat Python durable accepted submissions as the only completion authority; a DSH-only completed child is rerun. Start a new fixed Workflow for every item without a matching accepted digest, using the same business idempotency key but a fresh ledger-issued launch-attempt/reservation label. Never authorize from a late lifecycle event alone, reset a counter after process loss, or overwrite a tombstone.

- [ ] **Step 5: Make pause/cancel a quiescence barrier**

First use Python CAS to move active→cancelling/pausing and atomically close **all stage admission/activation leases**, including Host structured-result persistence, for the exact run-state revision/stage attempt. Abort every registered pending start controller and await every start Promise settlement; dispose any returned one-shot `SubagentRun` (there is no run.abort). Cancel/dispose every Workflow. For pause, interrupt any continuable whose start was accepted, then wait child idle/flush; close the old activation lease permanently. Resume may retain child identity but allocates a new lease only after idle/flush, so old-turn results cannot enter the new stage. For terminal cancel/teardown, call `await subagents.drainContinuableDescendants([roleHost])`. Only then cancel role-host Agents, wait idle, flush Sessions, and reconcile durable acceptance; pause retains the role-host handle, while terminal cancel/teardown awaits every retained `AgentHandle.dispose()` before recording clean state. Any pending-start/Promise/dispose/drain/idle/flush/handle-disposal timeout records a stable failed/paused cleanup state, keeps admission/reservations/leases closed, and never claims clean quiescence.

- [ ] **Step 6: Run recovery and cleanup regressions**

Run:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/resume.test.mjs \
  integrations/dsh_ecology_plugin/test/workflow_reconciliation.test.mjs \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_dsh_reconciliation.py \
  tests/test_dsh_cancel_race.py \
  tests/test_run_cleanup.py \
  tests/test_recovery_security.py \
  tests/test_auto_progress.py -q
```

Expected: all pass with no orphaned live handles in fakes.

### Task 13: Add predictive uncertainty and formal scientific validation

**Files:**
- Create: `src/ecologyrsi_dsh/evaluators/uncertainty.py`
- Create: `tests/test_uncertainty.py`
- Modify: `src/ecologyrsi_dsh/evaluators/greenhouse_prediction.py`
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py`
- Modify: `src/ecologyrsi_dsh/evaluators/fitness.py`
- Modify: `src/ecologyrsi_dsh/evaluators/baselines.py`
- Modify: `src/ecologyrsi_dsh/data/splits.py`
- Modify: `src/ecologyrsi_dsh/data/contracts.py`
- Modify: `src/ecologyrsi_dsh/data/registry.py`
- Modify: `src/ecologyrsi_dsh/core/director.py`
- Modify: `src/ecologyrsi_dsh/core/exposure_registry.py`
- Modify: `tests/test_formal_stage_protocol.py`
- Modify: `tests/test_fitness.py`
- Modify: `tests/test_datasets.py`
- Modify: `tests/test_baselines.py`
- Modify: `tests/test_promotion.py`
- Modify: `tests/test_sample_results_contract.py`
- Modify: `tests/test_greenhouse_prediction.py`
- Modify: `tests/test_execution_projection.py`

**Interfaces:**
- Registered `cellwise_time_block_calibrated_residual@1` UQ policy.
- Frozen `BaselineUqArtifact` containing the calibration-fit-selected point baseline and its calibration-UQ residual/quantile/scale bindings.
- Per prediction optional lower/upper interval, calibrated only on calibration data.
- Aggregate PICP, normalized interval width/score, Bonferroni-corrected block coverage bounds, and paired interval-score non-inferiority.
- `FormalFitnessAssessment` compares one locked candidate to the frozen calibration-fit baseline, never to the selection incumbent.

- [ ] **Step 1: Write UQ RED tests**

Test disjoint/digest-bound calibration-fit and calibration-UQ, exact deterministic cutpoint replay, locked point artifacts never training on residual rows, baseline/scale using calibration-fit only, cell-specific finite-sample quantile `ceil((n+1)(1-alpha))`, timestamp gaps breaking blocks, nominal 90% intervals, serially correlated hits using day-block bounds, nine-cell Bonferroni correction, undercoverage, excessively wide intervals, and paired interval-score non-inferiority with `delta_IS=0.05`. Freeze `formal-time-block-bootstrap@1`: 10,000 replicates, at least ten legal three-day starts, SHA-256 counter-derived draws, exact row/cell weighting, quantile indices, primary LCB, coverage LCB and interval-score UCB; cross-language fixtures must match byte-for-byte. The baseline tests must prove its point predictor is selected only on calibration-fit, candidate and baseline use the same policy/alpha/cell/calendar/scale but separate calibration-UQ residual quantiles, the exact normalized interval-score formula is used on identical formal rows/day blocks, and missing either side cannot be dropped. Missing intervals are allowed only in selection.

Formal-stage tests prove UQ cannot rescue negative point skill, the frozen baseline—not selection incumbent—is the comparator, coverage `>=0.95`, per-cell `n>=80`, blocks `>=14`, every cell skill nonnegative, formal score/LCB positive, insufficient evidence is sealed/inconclusive, and final-test results never modify candidate/incumbent/genome.

- [ ] **Step 2: Run RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_uncertainty.py \
  tests/test_formal_stage_protocol.py \
  tests/test_datasets.py -q
```

Expected: missing UQ module/policy.

- [ ] **Step 3: Finalize the calibration-fit/calibration-UQ data boundary**

Point models, preprocessing, fit-selected baselines and normalization scales use calibration-fit only. Use Task 5's exact 70% cut, half-open target-timestamp membership, fixed-hour calendar and 24h embargo—no runtime recutting. After the embargo, calibration-UQ computes nonconformity for the locked candidate point artifact and the frozen point baseline, and never retrains either. Never use model-selection, validation or final-test labels. Preserve the reward/score formulas; do not assert historical numeric equality after the training subset changes.

- [ ] **Step 4: Implement the time-block calibrated residual policy**

Calibrate each target × horizon separately with the finite-sample quantile index. Freeze positive `sigma_c` from calibration-fit and use normalized absolute residuals. Candidate and fit-selected persistence/24h-seasonal baseline run the identical registered interval policy and nominal alpha over the same calibration-UQ rows, but each stores its own residual vector and quantile. Sort in the frozen calendar basis and split temporal blocks at gaps. Do not claim exchangeability/distribution-free coverage; report empirical time-block evidence. Selection may set `require_predictive_intervals=false` during migration.

- [ ] **Step 5: Add formal point and UQ gates**

For validation/final require overall/per-cell coverage 0.95, 80 successes and 14 day blocks per cell, at least ten legal continuous three-day starts, `S_formal>0`, every `s_c>=0`, and the exact `formal-time-block-bootstrap@1` one-sided `LCB(S_formal)>0` versus the frozen baseline. Then require its Bonferroni-corrected per-cell empirical coverage LCB at least 0.85. Compute the normalized Winkler interval score for candidate and `BaselineUqArtifact` on the same formal rows/day-block draws, and require its paired `UCB(IS_candidate-IS_baseline)<=0.05`. Point or UQ failure rejects; insufficient evidence/continuity is inconclusive; both seal the exposed stage.

- [ ] **Step 6: Execute one locked validation and one sealed final test**

Only aggregate preregistered metrics/CIs leave the formal evaluator. Consult the cross-run exposure registry before opening. Validation/final evidence cannot enter strategy context, genome mutation, parent selection or cross-generation experience. Final-test status changes only the report conclusion.

- [ ] **Step 7: Run UQ/formal regressions**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_uncertainty.py \
  tests/test_formal_stage_protocol.py \
  tests/test_fitness.py \
  tests/test_evaluation.py \
  tests/test_datasets.py \
  tests/test_baselines.py \
  tests/test_promotion.py \
  tests/test_sample_results_contract.py \
  tests/test_greenhouse_prediction.py \
  tests/test_execution_projection.py -q
```

Expected: all pass; reward/score mathematical contracts remain unchanged and the new split/UQ protocol is explicitly versioned.

### Task 14: Update projections, workbench, exports, and Token semantics

**Files:**
- Modify: `src/ecologyrsi_dsh/api/projection.py`
- Modify: `src/ecologyrsi_dsh/api/events.py`
- Modify: `src/ecologyrsi_dsh/presentation/reporting.py`
- Modify: `src/ecologyrsi_dsh/presentation/training_assets.py`
- Modify: `src/ecologyrsi_dsh/presentation/trajectory.py`
- Modify: `plugins/ecology_evolution/index.html`
- Modify: `plugins/ecology_evolution/app.js`
- Modify: `plugins/ecology_evolution/assets/js/commands.js`
- Modify: `plugins/ecology_evolution/assets/js/render_shell.js`
- Modify: `plugins/ecology_evolution/assets/js/render_process.js`
- Modify: `plugins/ecology_evolution/assets/js/render_candidates.js`
- Modify: `plugins/ecology_evolution/test/smoke.mjs`
- Modify: `tests/test_execution_projection.py`
- Modify: `tests/test_stage_projection.py`
- Modify: `tests/test_export_contract.py`
- Modify: `tests/test_public_redaction.py`

**Interfaces:**
- Public candidate projection adds redacted genome/lineage/compiled-behavior/phenotype-instance/compiler/runtime/workflow and fitness layers.
- Run projection distinguishes selection incumbent, search parent, validated candidate, and final-test candidate.
- DSH runtime projection exposes Agent/Session/Workflow/context pressure without secrets or full conversations.

- [ ] **Step 1: Write projection/UI RED tests**

Assert genome diff and fitness layers render; four candidate statuses are distinct; historical candidates without genomes still render; runtime readiness is truthful; private prompts/Sessions/block rows do not appear; Token input and `token_budget` request field no longer exist.

- [ ] **Step 2: Run RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_execution_projection.py \
  tests/test_stage_projection.py \
  tests/test_export_contract.py \
  tests/test_public_redaction.py -q
node plugins/ecology_evolution/test/smoke.mjs
```

Expected: missing fields and obsolete Token control assertions.

- [ ] **Step 3: Add redacted genome and fitness projections**

Expose IDs, source/compiled behavior and phenotype-instance digests, registered template/policy names, bounded mutation summaries, gates, evidence-class labels, formal confidence summaries and efficiency metrics. Selection resampling must render as `探索性自适应证据`, never as confidence/p-value. Never expose instruction template bodies, runtime tokens, full tool arguments, private block evidence, or complete DSH messages.

- [ ] **Step 4: Remove Token hard-budget UX**

Delete the input, validation, form serialization, summary text and create-request field. Replace it with read-only context pressure from `tokenMeter.measure()` and cumulative provider usage only from `sessionProjections.snapshot(...).values.tokenUsage`, each labeled as reported or estimated. Never derive cumulative usage from TokenMeter pressure.

- [ ] **Step 5: Add DSH-native readiness and lifecycle UI**

Show required capabilities, root/role Sessions, active/reconciled Workflow state, compaction status, frozen routes and stable failure codes. No fallback switch is offered.

- [ ] **Step 6: Preserve historical projection/export compatibility**

Historical gateway runs display `旧执行协议（只读/可回放）`. Legacy genome adapters may provide a derived read-only summary, clearly labeled as projected.

- [ ] **Step 7: Run UI/projection/export regressions**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest \
  tests/test_execution_projection.py \
  tests/test_stage_projection.py \
  tests/test_export_contract.py \
  tests/test_public_redaction.py \
  tests/test_training_trajectory.py -q
node plugins/ecology_evolution/test/smoke.mjs
```

Expected: all pass.

### Task 15: Package, install, and verify the complete DSH-native release

**Files:**
- Modify: `pyproject.toml`
- Modify: `MANIFEST.in`
- Modify: `integrations/dsh_ecology_plugin/package.json`
- Modify: `integrations/dsh_ecology_plugin/README.md`
- Modify: `plugins/ecology_evolution/README.md`
- Modify: `plugins/ecology_evolution/plugin.json`
- Modify: `README.md`
- Modify: `scripts/verify_delivery.sh`
- Modify: `scripts/verify_artifacts.py`
- Modify: `scripts/build_delivery.sh`
- Modify: `scripts/create_delivery_archive.py`
- Modify: `scripts/real_api_agent_tool_acceptance.py`
- Create: `scripts/dsh_native_e2e_acceptance.py`
- Modify: `src/ecologyrsi_dsh/cli.py`
- Modify: `src/ecologyrsi_dsh/application/cli.py`
- Modify: `src/ecologyrsi_dsh/version.py`

**Interfaces:**
- Delivery includes a reproducible Cordis npm tgz with real dependency/exports/files closure, nested runtime modules, schemas, all role presets, installer, workbench assets, and Python modules.
- Installed CLI `ecologyrsi-dsh install-dsh-runtime --profile web` locates the bundled tgz/installer without relying on the source checkout.
- DSH-native acceptance exercises one small real generation through port 8848 with Python on loopback 8777 and no separate frontend port.

- [ ] **Step 1: Write package-content RED tests**

Require every nested JS module, JSON schema, preset YAML, installer, immutable legacy catalog, workbench asset and Python genome/fitness/runtime module in sdist/wheel/release archive. Inspect the npm tgz for Host/agent-plugin exports, exact dependency closure and no secret/profile/session files. Test installation from the built wheel in a temporary DSH home.

- [ ] **Step 2: Run delivery verification and observe RED**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 \
  bash scripts/verify_delivery.sh
```

Expected: failure naming newly required artifacts before packaging lists are updated.

- [ ] **Step 3: Update package manifests and version**

Set the project, plugin and manifest version to `0.3.0`. Extend `MANIFEST.in` for JS/MJS/JSON/MD/YML and explicitly enumerate every nested setuptools wheel data-file directory. Build the plugin npm tgz during `build_delivery.sh`, include it in wheel/sdist/archive, and teach the installed CLI to locate it. Do not package any DSH home, database, Session, credential, log, cache or generated run artifact.

- [ ] **Step 4: Install presets/profile patch and restart DSH**

Run from the installed CLI path:

```bash
ecologyrsi-dsh install-dsh-runtime --profile web
```

Expected: exact version check, npm tgz installed through `dsh plugin --profile web add --save-exact`, atomic backup/rollback path printed, idempotent preset/profile installation, import/dump-config/preset mount verification, and no credential access.

- [ ] **Step 5: Run the complete automated suite**

Run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest -q
node integrations/dsh_ecology_plugin/test/proxy_security.mjs
node --test integrations/dsh_ecology_plugin/test/*.test.mjs
node plugins/ecology_evolution/test/smoke.mjs
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 \
  bash scripts/verify_delivery.sh
```

Expected: every command exits 0.

- [ ] **Step 6: Run a fake-provider DSH integration acceptance**

Exercise Agent creation, role preset mounting, pre-start child reservation/descriptor claim (including first-tool-before-start-return), structured research/mutation/judge, planner/critic workflows, compaction, cancel/resume, pending-start signal abort plus settled one-shot disposal, continuable interruption/descendant drain, Host-result admission races, empty-role-host recovery, zero/some/all accepted Workflow reconciliation, DSH-only completion replay, event replay, and no-fallback checks without external provider cost. This fault-injection suite—not the ordinary real smoke—is the authority for reconciliation.

- [ ] **Step 7: Run one bounded real DSH smoke**

With DSH on `127.0.0.1:8848` and the Python sidecar on `127.0.0.1:8777`, run:

```bash
LC_ALL=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python scripts/dsh_native_e2e_acceptance.py \
  --dsh-origin http://127.0.0.1:8848 \
  --sidecar-origin http://127.0.0.1:8777 \
  --generations 1 \
  --candidates 2
```

Expected:

```text
protocol=dsh_native_plugin_evolution@1
genomes=2
python_model_requests=0
dsh_agent_sessions>=5
workflow_reconciliation=separately_verified_by_fault_injection
reward_contract=unchanged
scientific_result=selection_only
```

- [ ] **Step 8: Perform the final acceptance audit**

Verify all completion-definition items in the spec using artifacts, event replay, DSH Session references, network observations, public redaction scans and formal-stage seals. Do not claim causal/control performance or formal validation from the one-generation smoke.

## Execution Checkpoints

Run a full Python + Node regression after Tasks 3, 5, 9, 12, and 14. Keep the delivery artifact from Task 0 until Task 15 passes. If a checkpoint fails, revert only the current task's file set from the retained filesystem snapshot; never reset the whole workspace or delete user data.

Recommended execution groups:

1. Tasks 0–3: genome identity and deterministic compiler, with no runtime behavior change.
2. Tasks 4–5: scientific/statistical correctness and structural data isolation.
3. Tasks 6–9: DSH runtime foundation, presets, client, and tools.
4. Tasks 10–12: role migration, sample workflows, and recovery.
5. Tasks 13–15: UQ/formal validation, UI, packaging, and end-to-end acceptance.
