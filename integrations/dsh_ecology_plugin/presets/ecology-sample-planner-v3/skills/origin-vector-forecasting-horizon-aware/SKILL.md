---
name: origin-vector-forecasting-horizon-aware
description: Route one complete forecast origin through its registered vector predictor with explicit awareness of target and horizon coverage.
---

# Horizon-aware origin-vector forecasting

Confirm that the supplied origin contains the complete target-by-horizon grid and keep all cells together. Pay attention to the greater uncertainty of longer horizons when assigning confidence, but never change the registered prediction. Call the sole vector prediction tool exactly once with the exact `tool_id` and `wave_digest`, wait for all outputs, then return one decision per `sample_id` with the same tool.

Use only reason codes supplied by the Host. Do not extrapolate values yourself, inspect labels, or split the origin into independent model calls.
