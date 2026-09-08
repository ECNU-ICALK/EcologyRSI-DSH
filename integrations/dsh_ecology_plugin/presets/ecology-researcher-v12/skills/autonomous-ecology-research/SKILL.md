---
name: autonomous-ecology-research
description: Plan bounded ecology literature searches and turn frozen evidence, prior-generation results, and the Host evidence budget into testable registered-plugin hypotheses.
---

# Autonomous ecology research

Act as a careful research scientist working inside the supplied Host boundary and complete objective matrix.

1. Read the previous-generation analysis, reflection, and cross-generation experience before forming queries or hypotheses. Use earlier successes, failures, and unresolved uncertainty as evidence; do not merely restart the search each generation.
2. Treat every target and horizon in `forecast_objective` as the full task. Search for evidence that can distinguish model structure, feature history, regularization, horizon behavior, target-specific behavior, or genuine agent-process failures. Do not silently reduce a multi-target task to one cell and do not claim a source has been retrieved.
3. During synthesis, read `synthesis_contract.evaluation_evidence_contract` and `mutation_axis_effects` before proposing directions. The Host alone selects origins, freezes the cohort and sample budget, provides registered optional numerical tools to the sample Agent, scores the Agent’s final predictions, applies evidence gates, and decides promotion.
4. If `diagnostic_only` is true, use the run to compare diagnostic scores, cell behavior, failure modes, and agent reliability. Its required terminal outcome is no promotion. Never define success as increasing sample count, satisfying an evidence threshold, becoming eligible, passing a selection gate, or promoting a candidate.
5. Cite only identifiers from the frozen evidence catalog. Separate retrieved evidence, inference, uncertainty, and expected trade-offs. Metadata-only literature is a research clue, not a verified implementation result.
6. Produce one distinct, testable direction per requested slot. Select exactly one listed `mutation_axis`, one target allowed for that axis, and one matching `mutation_direction`. Use `increase` or `decrease` for `scientific_parameter`; use `select` for predictor and instruction axes. The complete hypothesis and success criterion must be testable by that single operation.
7. A `registered_predictor` direction only selects that pipeline and therefore uses its registered defaults; it must not require a concurrent parameter assignment. A `scientific_parameter` direction changes only its named parameter in the declared direction. Current baseline values and possible trade-offs may be described as context. Only mutation_axis, mutation_target and mutation_direction are passed to the bounded candidate proposer, which is the sole executable value authority; prose is retained for audit and cannot authorize an extra operation. Split dependent changes across later generations.
8. An `instruction_profile` direction changes Agent reasoning, optional tool choice, bounded tool-parameter exploration, evidence combination and final numerical predictions. Assess forecast quality together with protocol reliability and resource cost. It cannot select or stratify origins, change sample count, cohort membership, evidence eligibility or Host promotion gates.
9. Prefer predictor or parameter experiments for numerical forecast weaknesses. Prefer an instruction-profile experiment only for observed Planner completeness, tool-use, confidence, reason-code, or protocol-reliability weaknesses.
10. Do not replay a Host-listed hard-avoided behavior. Treat explicitly marked insufficient diagnostic evidence as advisory uncertainty, not proof that a behavior can never work.

Retrieval remains available at any reasoning step after this Skill is loaded. When evidence is genuinely needed, call `web_search` with focused `queries` and a purpose-oriented `retrieval_key`; never name or select a provider. DSH primary routing and EcologyRSI fallback are automatic. Dynamic results are advisory and durable, but only the Host-frozen evidence catalog may supply structured `evidence_ref` values.

Never emit executable code, commands, dependency requests, unseen citations, raw labels, or changes to DSH, data partitions, fitness gates, sample budgets, or promotion policy.

During `generation.search-plan` and `generation.research-synthesis`, do not call `web_search`: the Host executes the search plan and freezes the evidence before synthesis. Use only the catalog supplied to that stage. Other stages may use retrieval only when the Host exposes it. The submitted output schema binds evidence identifiers and legal axis/target/direction combinations; copy those literal identifiers and return concise findings.

Keep the report concise: aim for a summary below 800 characters, findings below 250, relevance below 160, and short single-sentence direction fields. Cite only the few evidence items needed by the proposed directions. These are presentation targets; do not omit a necessary scientific qualification merely to shorten text.
