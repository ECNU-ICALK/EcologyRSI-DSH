---
name: origin-vector-review
description: Review Agent forecasts against causal evidence and return bounded revision requests.
---

# Origin-vector review

For pre-score review, inspect the Agent's final numeric vector, its method and evidence call references, causal history, supplied model outputs and physical bounds. Return one `action` per requested sample: `accept` when evidence is adequate, `revise` when evidence conflicts with the forecast, or `uncertain` when context is insufficient. Use only the supplied reason codes and a self-assessed confidence. A revise or uncertain decision sends the forecast back to the Agent within the existing attempt budget. Do not provide substitute numbers or choose a Host recovery tool. Criticism is not a statistical improvement test: hidden evaluation labels are unavailable and Host gates alone decide promotion.

For post-score reflection, compare the immutable historical outcome across cells, distinguish final Agent error, raw tool error and adjustment effect, and recommend one bounded next-generation action. Do not change past predictions or infer causation from observational errors.

For pre-score review, make one bounded pass over the supplied evidence and submit the complete decision array promptly. Check evidence references, physical consistency, and whether the stated method is supported; do not refit a model, reconstruct the prediction vector, or repeatedly derive the time series. Reserve output budget for every requested sample. Submit only `sample_id`, `action`, `confidence`, and one allowed `reason_code` per decision, without narrative fields. Insufficient evidence warrants `uncertain`, never assumed acceptance. This does not relax any evidence requirement or permit skipping a sample.

When the Host supplies `formal_retrieval`, `web_search` queries only its frozen evidence library. You may search that library at any reasoning step after Skill and before final output, and refine your queries as needed. The library stays fixed; returned URLs are citations only and must never be read remotely. Do not request live-network retrieval or external fallback.
