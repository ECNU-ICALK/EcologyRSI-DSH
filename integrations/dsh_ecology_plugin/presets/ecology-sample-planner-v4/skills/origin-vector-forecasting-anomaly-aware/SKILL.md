---
name: origin-vector-forecasting-anomaly-aware
description: Route a full forecast-origin vector through its registered predictor while carefully checking unusual inputs and tool-result completeness.
---

# Anomaly-aware origin-vector forecasting

Before prediction, inspect only the label-free context for missing, extreme, or internally inconsistent observations. These checks affect confidence and the allowed reason code, never the predicted value. Call the sole registered vector prediction tool exactly once with the exact `tool_id` and `wave_digest`; verify that its result covers every supplied `sample_id`; then return one decision for every cell using that same tool.

If a label-free scientific question blocks routing, use `web_search` before the prediction tool with focused `queries` and a purpose-oriented `retrieval_key`; never name or select a provider. Search is optional and cannot replace, repeat, or change the registered prediction call.

Use only context-listed reason codes. Never infer labels, repair inputs, calculate forecasts, or replace a tool output.
