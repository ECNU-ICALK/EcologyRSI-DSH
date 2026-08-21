# DSH Evolution Closed-Loop Hardening Design

## Goal

Turn the current DSH-backed one-round candidate evaluation into an auditable closed-loop plugin evolution system. DSH owns model context, role Agents, Subagents, and bounded workflows. Python remains the durable authority for the evolution ledger, registered scientific tools, deterministic fitness, and promotion gates.

## Scope

This hardening covers five observed failures from the 2026-08-21 native runs:

1. Cross-generation reflection reached the researcher but not the candidate proposer.
2. A stochastic critic could replace a successful candidate prediction and thereby change authoritative fitness.
3. The runtime declared `dsh_native_workflow` while only using one-shot DSH Subagents.
4. DSH token and child-session evidence existed but was not projected into the run ledger.
5. DSH roles emitted reason codes that the Host later reduced to `remote_reason_invalid`.

Legacy autonomous execution is outside the normal web path. A newly created autonomous run using a non-native execution protocol must fail closed at the server boundary.

## Frozen Boundaries

- DSH itself never evolves.
- Evolution changes only the plugin genome: registered scientific parameters, registered tool choices, bounded workflow settings, and role instruction parameters.
- Dataset partitions, evaluator, objective profile, promotion policy, registry allowlists, and security digests remain immutable.
- There is no per-sample Agent token hard limit. DSH context pressure and compaction remain authoritative.
- Repaired or fallback predictions never overwrite the deterministic raw-candidate fitness record.
- Python durable acceptance remains the completion authority after restart.

## Reflection Contract

The Host builds `ecologyrsi-dsh.evolution-reflection/1` from the previous generation or compatible historical runs. The bounded contract contains:

- prior analysis and knowledge-assessment digests;
- target/horizon priorities;
- scientific and execution failure codes;
- Judge flags and bounded guidance;
- raw-candidate and repaired-path score summaries;
- prohibited mutation digests and previously rejected parameter sets;
- evidence-backed parameter directions with confidence and non-causal labels;
- requested workflow or role-policy changes;
- diversity requirements for the next proposal batch.

The DSH researcher receives this reflection and returns grounded research evidence. The candidate proposer receives the validated reflection object, the research result, and the parent genome—not only their digests. The Host rejects an exact repeat of a prohibited mutation unless the reflection explicitly supplies new evidence for reevaluation.

## Fitness Contract

Each evaluated sample maintains two tracks:

1. `raw_candidate`: output from the candidate's registered prediction tool and deterministic Host constraint check.
2. `execution_policy`: DSH planner/critic decisions, retries, repairs, fallback results, cost, and reliability.

`raw_candidate` is authoritative for the scientific objective. A critic reviewing a successful registered prediction may recommend another tool, but that recommendation is advisory and cannot replace the raw prediction. Tool execution failures may still enter the bounded repair loop, but the raw failure and repair penalty remain durable and cannot become a neutral baseline score.

The public projection exposes raw score, repaired score, fallback rate, repair count, and the score source. Promotion uses raw scientific fitness plus separate execution-policy gates.

## DSH Workflow Contract

The coordinator role-host owns a fixed Host-authored Workflow template. Before a bounded batch starts, the Host reserves every child label durably. The DSH Workflow Engine starts the batch with structured arguments and bounded `maxItems`, `maxTotalAgents`, `maxConcurrent`, and timeout values. Dynamic scripts and model-authored workflow code remain forbidden.

The initial implementation uses the Workflow Engine for homogeneous structured role batches. Single researcher, proposer, and judge calls may remain one-shot DSH Subagents. Sample planner and critic waves must use the coordinator or sample-planner Workflow service rather than Python-only wave orchestration.

## Reason-Code Contract

Planner and critic schemas expose only Host-owned reason-code enums. The stage context repeats the allowed codes and semantic predicates. The Host additionally rejects contradictory decisions, including a low-confidence reason when confidence is above the frozen threshold or a repair request without a supported failure/review predicate.

## Observability Contract

Every accepted result stores the actual DSH child UUID and the reservation label separately. The runtime aggregates DSH `sessionProjections.snapshot(session).values.tokenUsage` across child sessions and reports provider usage as observed—not estimated. `tokenMeter.measure()` remains current pressure only. Capabilities and run projection set `first_call_verified` after a real stage call.

## Recovery and Cancellation

Admission closes atomically before cancellation. Pending starts are aborted and settled, returned one-shot runs are disposed, workflow runs are disposed, and role-hosts are flushed before terminal status. No stage failure, retry schedule, structured result, or usage event may be admitted after terminal cancellation. Runtime cleanup status is reported separately from scientific run outcome.

## Tests

- Reflection content changes the proposer request and exact failed mutations are rejected.
- A critic recommendation cannot change a successful raw prediction or reward.
- Actual tool failures retain raw failure evidence and repair penalties.
- Sample batches exercise the DSH Workflow Engine with fixed scripts and bounded arguments.
- Actual child UUID, label, provider usage, and context pressure survive projection and restart.
- Reason-code enums prevent `remote_reason_invalid` on conforming roles and contradictory outputs fail closed.
- Cancellation prevents every late mutation event.
- Native browser E2E runs at least two generations and demonstrates that generation 2 consumes generation 1 reflection.
