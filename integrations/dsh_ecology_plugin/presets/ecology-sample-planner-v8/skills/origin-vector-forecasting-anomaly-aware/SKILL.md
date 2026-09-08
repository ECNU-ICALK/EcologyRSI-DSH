---
name: origin-vector-forecasting-anomaly-aware
description: Agent-owned ecology forecasts using optional tools and bounded model experiments.
---

Check missing values and unusual current observations; consider an alternative model or a conservative adjustment when the evidence supports it.

Use only information available at the forecast origin. You own the final numerical predictions. Decide whether a model is useful; `candidate-model` is an optional default, not a mandatory route. The capability catalog lists other models, defaults and parameter bounds. Call `ecology_execute_prediction_tool` zero to six times, supplying a unique `call_id`, `tool_id`, exact `wave_digest`, and `parameters` (an empty object selects defaults). Wait for results, assess them, and decide whether another call is needed. Tool errors count toward the budget and may inform your next choice. Do not retry the same call unless retrieving the same idempotent result.

This task is inference at one origin, not an open-ended research or calibration session. The six-call limit is a ceiling, not a target: stop gathering evidence once you can submit the full prediction vector. Keep comparisons concise and reserve output room for every sample. Do not rerun the same predictor with the same parameters through another tool name merely to confirm an existing result. Calibration and cross-sample model search belong to the outer evolution loop.

You may directly predict, use a model output, blend multiple model results, or make an evidence-based adjustment. For non-direct methods cite the successful call_ids you actually used. For blends cite at least two. You are not required to copy a tool output. Use `web_search` only for label-free scientific evidence, at most three calls, before final output; never search for the evaluation episode's future measurements.

Call `structured_output` once with one decision per exact sample_id: finite `predicted`, `confidence` in [0,1], `method` (direct/model/blend/adjusted), an allowed `reason_code`, and `evidence_call_ids`. Physical bounds and operational budgets remain Host-enforced. If structured_output rejects its arguments, correct the arguments once using that feedback. Do not call any other tool after the first submission or repeat a successful output; if correction fails, end the turn for Host-owned retries. Never read evaluation labels, change the scoring rules, or claim unsupported certainty.
