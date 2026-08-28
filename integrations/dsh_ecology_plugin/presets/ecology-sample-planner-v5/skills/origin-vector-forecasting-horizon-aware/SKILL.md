---
name: origin-vector-forecasting-horizon-aware
description: Route one complete forecast origin through its registered vector predictor with explicit awareness of target and horizon coverage.
---

# Horizon-aware origin-vector forecasting

Confirm that the supplied origin contains the complete target-by-horizon grid and keep all cells together. Pay attention to the greater uncertainty of longer horizons when assigning confidence, but never change the registered prediction. Finish those checks first, then call the sole vector prediction tool exactly once with the exact `tool_id` and `wave_digest`.

After that tool returns, do not analyze its numeric values, physical bounds, repair sequence, or fallback policy; Host validation owns those decisions. Immediately call `structured_output` exactly once with one decision per `sample_id` and the same registered tool. Emit no explanation. If the prediction tool or `structured_output` is rejected, terminate without repeating either call; the Host owns bounded retries.

If a label-free scientific question blocks routing, use `web_search` before the prediction tool with focused `queries` and a purpose-oriented `retrieval_key`; never name or select a provider. Search is optional and cannot replace, repeat, or change the registered prediction call.

Use only reason codes supplied by the Host. Do not extrapolate values yourself, inspect labels, or split the origin into independent model calls.
