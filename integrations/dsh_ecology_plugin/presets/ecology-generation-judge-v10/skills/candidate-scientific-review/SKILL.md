---
name: candidate-scientific-review
description: Review one ecology candidate from its supplied scientific evaluation without making selection decisions or proposing future experiments.
---

# Candidate scientific review

Review exactly one candidate and only the evidence supplied in the `generation.judge` context:

- `candidate_id`, `proposal_id`, and `generation` identify the candidate under review.
- `scientific_evaluation` supplies the Host-computed `score`, `passed` result, aggregate `metrics`, and evidence digests.
- `review_policy` binds the review purpose, evidence class, exact candidate revision and scope, and Host decision authority. Follow it before interpreting `passed`: exploratory evidence is not an independent certification attempt.
- `review_policy.host_gate_summary` supplies the sealed same-cohort incumbent comparison when available: score delta, worst-cell delta, coverage, practical threshold, and each applicable Host gate. Cite these values when discussing relative improvement or regression; never reconstruct a comparison from an unrelated score.
- `fitness_profile_digest` and `evaluation_cohort_digest` bind the review to the frozen scoring profile and cohort; they are identifiers, not additional scientific evidence.

Assess whether the supplied candidate evidence supports acceptance for the stated review purpose, explain uncertainty or limitations, and avoid causal claims that the aggregate evidence cannot establish. Do not infer raw samples, hidden labels, omitted metrics, sibling outcomes, or facts from a digest.

You may use `web_search` after loading this Skill only to clarify general methodology. Supply focused `queries` and a purpose-oriented `retrieval_key`; never name or select a provider. Search results cannot substitute for missing candidate evidence and cannot change the Host-computed score, gates, or cohort.

Return only one `ecology-generation-review@1` object with exactly these fields:

- `schema_version`
- `accepted`
- `rationale`
- `flags`

Candidate-review acceptance is an advisory assessment of this evidence only. It is not selection, ranking, promotion, or authorization to advance the candidate; never claim that any of those occurred. Never propose next-generation directions, experiments, searches, mutations, or policy changes.

For exploratory review, a missing formal time-block minimum, independent inference replica, or a false scientific certification gate is a limitation, not a reason to reject exploration. Report concrete physical, coverage, identity, or evidence inconsistencies with evidence references; the Host verifies them and owns every blocking decision. Never waive a Host cell-regression or practical-improvement requirement. Your `accepted` value and free-text flags are advisory and cannot create a promotion veto.
