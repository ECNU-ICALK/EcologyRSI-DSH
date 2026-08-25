---
name: candidate-scientific-review
description: Review one ecology candidate from its supplied scientific evaluation without making selection decisions or proposing future experiments.
---

# Candidate scientific review

Review exactly one candidate and only the evidence supplied in the `generation.judge` context:

- `candidate_id`, `proposal_id`, and `generation` identify the candidate under review.
- `scientific_evaluation` supplies the Host-computed `score`, `passed` result, aggregate `metrics`, and evidence digests.
- `fitness_profile_digest` and `evaluation_cohort_digest` bind the review to the frozen scoring profile and cohort; they are identifiers, not additional scientific evidence.

Assess whether the supplied candidate evidence supports acceptance, explain uncertainty or limitations, and avoid causal claims that the aggregate evidence cannot establish. Do not infer raw samples, hidden labels, omitted metrics, sibling outcomes, or facts from a digest.

Return only one `ecology-generation-review@1` object with exactly these fields:

- `schema_version`
- `accepted`
- `rationale`
- `flags`

Candidate-review acceptance is an advisory assessment of this evidence only. It is not selection, ranking, promotion, or authorization to advance the candidate; never claim that any of those occurred. Never propose next-generation directions, experiments, searches, mutations, or policy changes.
