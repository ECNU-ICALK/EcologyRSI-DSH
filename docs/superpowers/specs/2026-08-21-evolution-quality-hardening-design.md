# Evolution Quality Hardening Design

## Goal

Make each generation a small, auditable plugin-genome change; prevent wasted sibling candidates; expose completed DSH research and truthful provider usage; and require a complete model-backed decision and reflection chain for every effective sample.

## Frozen decisions

- A candidate mutation contains exactly one effective operation within the Host trust region.
- The Python Host is authoritative for mutation validation. DSH output schemas and prompts are advisory constraints because the rc.6 schema projection cannot preserve every JSON Schema keyword.
- An operation that leaves its targeted value unchanged is invalid. A sequence that produces the original genome is invalid.
- Duplicate and failed candidates have no scientific rank. Only candidates with a finite evaluation score are ranked.
- Candidate proposal remains sequential within a generation so each later slot can receive bounded digests for already accepted sibling behaviors. Candidate evaluation remains bounded-parallel.
- Public research status is derived from the native fields `dsh_research_summary`, `dsh_research_evidence`, and `dsh_evolution_reflection` when they are present.
- Provider usage must come from DSH SessionProjection data. TokenMeter remains context pressure only. Workflow metrics are captured while the child Session is still available.
- Every effective forecast origin invokes a remote Planner even when only one registered tool is legal. The traditional predictor remains a Host vector tool and never replaces the Planner call; one invocation returns all target/horizon cells for that origin.
- Every origin-level tool result invokes a remote Critic, and every Host-scored prediction vector invokes a separate remote Reflector. New real runs use `dsh-strict-origin-bundle@3`; historical manifests retain their frozen policy.
- Generation 1 and later re-evaluate the search parent and formal elite on the current cohort with the same strict chain. Missing controls or incomplete chain evidence fails closed.
- Candidate concurrency stays bounded at four by default and eight at the API hard maximum.

## Data flow

1. DSH proposes a mutation delta.
2. Python validates operation count and effective changes, applies the delta, compiles behavior, and rejects prohibited or sibling duplicate behavior.
3. Each later proposal receives only bounded sibling digests and effective parameter summaries, never evaluation labels or holdout data.
4. Every forecast origin uses one DSH Planner call, then exactly one selected Host-registered vector prediction tool for all target/horizon cells.
5. Every origin vector uses one DSH Critic call; after deterministic Host cell scoring, the origin uses one DSH Reflector call that cannot change the current prediction. Its bounded outcome/error/action counts enter the next generation's aggregate context.
6. Strict checkpoints are published only after reflection and include a digest-bound chain attestation. Unproven interrupted samples are executed again.
7. Parent/elite controls run on the same cohort. Promotion and next-parent replacement require the frozen minimum delta and paired stability gate; otherwise the prior elite is retained.
8. DSH child usage is snapshotted before workflow disposal and stored with the durable structured result.
9. Reporting maps native research/reflection fields and separates operational routing diagnostics from scientific fitness.

## Error handling

- Invalid or no-op mutations fail closed before candidate creation.
- Exact sibling behavior triggers a bounded re-proposal; exhausted retries produce an explicit duplicate candidate rather than silently evaluating it.
- Missing SessionProjection remains observability-unavailable and never fabricates usage from context pressure.
- Host deterministic routing is forbidden by the strict sample protocol, including the single-tool path.
- Existing runs replay with their original manifest values.

## Verification

- Unit tests reproduce the 22-operation/no-op mutation failure and observe rejection.
- Mixed evaluated/duplicate ranking tests keep duplicates unranked.
- Proposal tests prove later slots receive prior sibling digests and retry an exact sibling duplicate.
- Reporting tests use the actual native DSH research shape.
- Workflow tests prove provider usage is available even when the child is removed after completion.
- Sample execution tests prove the single-tool path still calls Planner and Critic, returns all nine default greenhouse cells in one vector tool invocation, then calls Reflector after Host scoring; Critic failure never returns a successful prediction.
- Runtime tests prove new runs freeze strict one-sample prompts and `always@1` critic policy while historical runs remain unchanged.
- Selection tests prove parent/elite controls use the current cohort and no-improvement retains the prior search parent and incumbent.
- Full Python, Node, browser smoke, diff-check, and a bounded local native evolution smoke are run before completion.
