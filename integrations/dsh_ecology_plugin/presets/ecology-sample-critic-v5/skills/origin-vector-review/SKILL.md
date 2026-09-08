---
name: origin-vector-review
description: Review Agent forecasts against causal evidence and return bounded revision requests.
---

# Origin-vector review

For pre-score review, inspect the Agent's final numeric vector, its method and evidence call references, causal history, supplied model outputs and physical bounds. Return one `action` per requested sample: `accept` when evidence is adequate, `revise` when evidence conflicts with the forecast, or `uncertain` when context is insufficient. Use only the supplied reason codes and a self-assessed confidence. A revise or uncertain decision sends the forecast back to the Agent within the existing attempt budget. Do not provide substitute numbers or choose a Host recovery tool. Criticism is not a statistical improvement test: hidden evaluation labels are unavailable and Host gates alone decide promotion.

For post-score reflection, compare the immutable historical outcome across cells, distinguish final Agent error, raw tool error and adjustment effect, and recommend one bounded next-generation action. Do not change past predictions or infer causation from observational errors.
