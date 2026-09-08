---
name: batch-scientific-reflection
description: Judge aggregate candidate evidence under the frozen evidence budget, extract reusable scientific lessons, and propose distinct registered next-generation experiments.
---

# Batch scientific reflection over the full objective

Treat the completed candidate batch as a controlled comparison over one frozen cohort. Read earlier-generation analysis and reflection so lessons, failed assumptions, and unresolved questions accumulate rather than reset.

Use `candidate_outcomes` as the only authoritative candidate-level mapping. It is Host-generated and sorted by ascending scientific `rank`, with unscored candidates last. Bind every observation through the explicit `candidate_id`, `direction_id`, and `direction_digest` fields. A rank is an outcome position, not a direction identifier: never infer that rank 1 means direction `d1`, that rank 2 means `d2`, or that array position encodes a direction. Do not reconstruct candidate identities from `generation_analysis` or prose.

Judge the complete Cartesian product in `forecast_objective`: for the greenhouse task, temperature, relative humidity, and CO2 at 1h, 6h, and 24h. Separate aggregate predictive skill, every target-horizon cell, worst-cell behavior, agent/tool reliability, efficiency, and uncertainty. A direction may focus on a weak cell, but it must state how the other cells are preserved and must never redefine one cell as the whole task. Explain supported improvements and failures without causal overclaiming.

Read `host_boundary.evaluation_evidence_contract` before judging success. A diagnostic-only batch can rank hypotheses and create reusable lessons, but cannot satisfy formal evidence thresholds or promote a candidate. Treat its required no-promotion terminal outcome as correct governance, not as an experimental failure. Never recommend changing the frozen cohort, origin count, sample budget, evaluator, gates, or promotion policy.

If an unresolved methodological question blocks reflection, use `web_search` after loading this Skill with focused `queries` and a purpose-oriented `retrieval_key`. Never name or select a provider. Treat dynamic results as advisory background only: cite structured directions solely with identifiers from the frozen knowledge snapshot.

Propose exactly the requested number of distinct next directions. Each direction must choose one allowed `mutation_axis` and one exact listed target, cite only frozen evidence identifiers, state a falsifiable success criterion within the frozen evidence class, and avoid replaying Host-listed hard-failed effective behaviors. The experiment must be achievable by exactly one operation:

- A `scientific_parameter` direction must set `mutation_direction` to `increase` or `decrease`. Describe only that direction of change; never write an exact parameter assignment in free text.
- A `registered_predictor` direction must set `mutation_direction` to `select`. Selecting it uses its registered defaults and cannot also set parameters.
- An `instruction_profile` direction must set `mutation_direction` to `select`. It can change Agent reasoning, optional model use, bounded model-parameter exploration, evidence combination and final numerical predictions. Assess its numerical effect on the same frozen cohort alongside protocol reliability and resource cost. It cannot change Host sampling, evidence eligibility or promotion gates.

Split dependent changes across later generations. Evidence marked insufficient is a diagnostic clue, not a permanent behavior ban. Search recommendations should target unresolved uncertainty and distinguish target-specific, horizon-specific, model-structure, and agent-process hypotheses. Every proposed direction is advisory and must pass the next generation's Host preflight before it can be assigned or executed. The stop recommendation is advisory; Host policy alone controls continuation and selection.
