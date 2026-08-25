---
name: origin-vector-forecasting-balanced
description: Route one complete multi-target multi-horizon forecast origin through its single registered vector predictor with balanced attention to all cells.
---

# Balanced origin-vector forecasting

Treat the supplied samples as one inseparable forecast-origin vector. Check that every target and horizon is present, then call the sole registered prediction tool once with the exact `tool_id` and `wave_digest`. After the result, return exactly one decision per supplied `sample_id`, preserving the same tool choice for every cell.

Use only reason codes explicitly listed in the context. Do not calculate, replace, smooth, or invent predictions, and do not use labels or future observations.
