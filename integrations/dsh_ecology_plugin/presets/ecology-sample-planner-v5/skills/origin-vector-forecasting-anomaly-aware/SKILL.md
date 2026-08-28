---
name: origin-vector-forecasting-anomaly-aware
description: Route a full forecast-origin vector through its registered predictor while carefully checking unusual inputs and tool-result completeness.
---

# Anomaly-aware origin-vector forecasting

Before prediction, inspect only the label-free context for missing, extreme, or internally inconsistent observations. These checks affect confidence and the allowed reason code, never the predicted value. Finish those checks first, then call the sole registered vector prediction tool exactly once with the exact `tool_id` and `wave_digest`.

After that tool returns, do not analyze its numeric values, physical bounds, repair sequence, or fallback policy; Host validation owns those decisions. Immediately call `structured_output` exactly once with one decision for every supplied `sample_id` and the same registered tool. Emit no explanation. If the prediction tool or `structured_output` is rejected, terminate without repeating either call; the Host owns bounded retries.

If a label-free scientific question blocks routing, use `web_search` before the prediction tool with focused `queries` and a purpose-oriented `retrieval_key`; never name or select a provider. Search is optional and cannot replace, repeat, or change the registered prediction call.

Use only context-listed reason codes. Never infer labels, repair inputs, calculate forecasts, or replace a tool output.
