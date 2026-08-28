---
name: origin-vector-forecasting-balanced
description: Route one complete multi-target multi-horizon forecast origin through its single registered vector predictor with balanced attention to all cells.
---

# Balanced origin-vector forecasting

Treat the supplied samples as one inseparable forecast-origin vector. Check that every target and horizon is present, and decide confidence before calling the sole registered prediction tool once with the exact `tool_id` and `wave_digest`.

After that tool returns, do not analyze its numeric values, physical bounds, repair sequence, or fallback policy; Host validation owns those decisions. Immediately call `structured_output` exactly once with one decision per supplied `sample_id`, the same registered tool for every cell, and a context-listed reason code. Emit no explanation. If the prediction tool or `structured_output` is rejected, terminate without repeating either call; the Host owns bounded retries.

If a label-free scientific question blocks routing, use `web_search` before the prediction tool with focused `queries` and a purpose-oriented `retrieval_key`; never name or select a provider. Search is optional and cannot replace, repeat, or change the registered prediction call.

Use only reason codes explicitly listed in the context. Do not calculate, replace, smooth, or invent predictions, and do not use labels or future observations.
