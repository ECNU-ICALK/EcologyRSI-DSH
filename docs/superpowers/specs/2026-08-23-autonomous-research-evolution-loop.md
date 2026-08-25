# Autonomous Research and Evolution Loop

## Objective

Implement autonomous research, proposal, execution, evaluation, reflection, and continued evolution entirely as an EcologyRSI DSH plugin workflow. DSH owns Agent sessions, context, model routing, subagents, and workflow execution. The Python sidecar owns immutable data boundaries, registered numerical tools, scoring, selection, persistence, and security gates. DSH core is not modified.

## Generation protocol

Every new native run freezes `autonomous_research_protocol=dsh-model-search-reflect@1` and executes this sequence:

```text
previous aggregate result + reflection + parent Genome
                         |
                         v
             generation.search-plan (LLM)
                         |
                         v
       Host catalog/OpenAlex metadata retrieval
                         |
                         v
        generation.research-synthesis (LLM)
                         |
              N distinct directions
                         |
                         v
       candidate.propose × N (LLM, one slot each)
                         |
                         v
       Host validation, compilation, smoke checks
                         |
                         v
    same frozen feedback cohort evaluation × N
                         |
                         v
      Host ranking, stability test, incumbent gate
                         |
                         v
              generation.reflect (LLM)
                         |
                         +----> next generation search-plan
```

The search plan is persisted before retrieval. Search results are frozen before synthesis. Every candidate stores its direction ID/digest, mutation, resulting Genome digest, compiled behavior digest, cohort digest, and result. The batch reflection stores the source analysis digest. These identities make pause/resume and event replay deterministic.

## Per-origin prediction protocol

The default greenhouse objective has three targets (`air_temperature`, `relative_humidity`, `co2_concentration`) and three horizons (1 h, 6 h, 24 h). One forecast origin is therefore one Agent sample containing nine scoring cells.

For every origin and candidate:

1. DSH Sample Planner receives only causal, label-free history and all nine requested cells, then explicitly selects the registered candidate tool.
2. The Host invokes the closed-form ridge predictor once as a vector tool and requires exactly nine sample-ID keyed outputs.
3. DSH Sample Critic reviews the vector; it may accept it or request an allowlisted repair after a Host-detectable failure, but cannot invent prediction values.
4. The Host applies physical checks and scores each cell against the frozen fit-selected baseline.
5. DSH Sample Reflector receives the completed historical outcome, classifies it, and recommends a bounded next action. It cannot change the just-scored vector.
6. The checkpoint becomes durable only when all nine cells and the reflection are complete. A partial origin is rerun after restart.

The generation summary deduplicates the shared origin reflection and carries `reflection_outcome_counts`, `reflection_error_source_counts`, and `reflection_next_action_counts` into batch reflection and future research. There is no deterministic single-tool routing bypass and no Host prediction fallback in this strict protocol.

## Why 1521 scoring cells

The default fitness profile requires at least 8 paired 24-hour blocks. Eight blocks need `(8 - 1) × 24 + 1 = 169` consecutive forecast origins so that block starts and the three-day moving-block test are available. Each origin has 9 target/horizon cells, so the minimum selection-eligible budget is `169 × 9 = 1521` cells.

This is not 1521 LLM prediction calls. Under `dsh-strict-origin-bundle@3`, it is 169 origin chains per candidate: normally 169 Planner calls, 169 vector-tool calls, 169 Critic calls, and 169 post-score Reflector calls. Smaller budgets remain useful diagnostic smoke runs but cannot promote a candidate.

## Selection and data boundaries

- `training_fit` fits the registered predictor and selects the scoring baseline.
- A frozen rotating window from `training_feedback` is the adaptive selection cohort. Every sibling and required control uses the same cohort.
- `development`, `gate`, external holdout, hidden, and final test remain outside the adaptive loop. Reusing a final test every generation would leak information and invalidate it as a final test.
- The Host ranks only evaluated candidates, requires all scientific and independent-review gates, a score delta greater than 0.005, per-cell non-regression, and centered max-T moving-block stability before replacing the incumbent.
- If no candidate improves, the incumbent is retained. A safe evaluated search parent may be retained only under the explicit exploratory policy and is never presented as a formal improvement.

## Search and implementation boundary

The strategy model controls queries, synthesis, direction choice, and bounded mutations. The Host controls network destinations, response limits, evidence freezing, parameter schemas, program registries, compilation, evaluation, and promotion. OpenAlex output is metadata-only and never executed.

The generation-start search plan remains the bootstrap path that freezes citation-authoritative evidence. It is not the only retrieval opportunity: the dynamic routing extension in `2026-08-25-dynamic-dsh-retrieval-routing.md` permits every EcologyRSI stage to call one provider-neutral `web_search` after Skill and before its terminal tool. DSH internal search is primary; the sidecar invokes OpenAlex only after a technical failure or deterministic evidence-insufficiency check. These dynamic results are durable advisory context and cannot create a trusted `evidence_ref` or bypass any Host gate.

“Implementing a proposal” means compiling it into an allowlisted plugin Genome operation such as a bounded parameter, registered policy, registered tool choice, or registered workflow setting. Arbitrary source generation, dynamic imports, shell commands, dependency installation, evaluator changes, and DSH framework mutation are rejected. A genuinely new algorithm must first be developed and registered as a reviewed EcologyRSI plugin capability.

## Failure and recovery rules

- Invalid, no-op, overlarge, duplicate, or previously failed mutations are rejected before candidate creation. A bounded validation detail is returned to the next proposer attempt.
- Research synthesis and generation reflection each receive one fresh model retry when their schema-shaped output fails Host semantic validation. The retry carries only a bounded validation detail and rejected-output digest; it never relaxes the contract.
- Every synthesized/reflected direction names exactly one Host-registered mutation axis and target (`scientific_parameter`, `registered_predictor`, or `instruction_profile`) for the frozen parent Genome, and every accepted candidate mutation must implement that exact assignment.
- New native runs select a seed Genome template whose predictor matches the frozen task predictor. Restored historical runs derive mutation boundaries from their persisted parent Genome rather than potentially stale task/UI metadata.
- OpenAlex failure falls back to the built-in verified catalog and is recorded as a warning.
- Missing required Planner, Critic, or Reflector evidence fails the strict chain closed.
- Candidate evaluations may execute in bounded parallel workers, but each run keeps a single generation decision barrier.
- Search plan, evidence snapshot, synthesis, candidate results, generation analysis, and reflection are append-only events. Existing runs without the new protocol replay with their historical semantics.

## Acceptance criteria

- One origin with 3 × 3 cells produces exactly one Planner, one vector-tool, one Critic, and one Reflector invocation and nine scoring rows.
- Model-authored retrieval queries execute before Host fallbacks.
- Synthesis returns exactly one distinct direction per candidate slot and cites only frozen evidence.
- Every accepted mutation is one effective trust-region operation; retry receives an actionable Host validation detail.
- Reflection is required before generation advance and seeds the next search plan.
- Reflection aggregates, candidate direction lineage, model-call evidence, and cohort identity survive replay.
- Python, DSH plugin Node tests, browser smoke, static compilation, and a local sidecar/native smoke all pass, or an unavailable external DSH/provider dependency is reported explicitly.
