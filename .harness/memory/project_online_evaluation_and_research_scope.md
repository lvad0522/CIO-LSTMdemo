---
name: Online evaluation and research scope
description: Confirmed U850-CIO-LSTM research contract, focus-region evaluation direction, and visualization priorities.
type: project
status: verified
module: online-inference
---

- Research chain is U850 anomaly → CIO projection → pre-trained LSTM → summer precipitation anomaly; CIO is an indispensable LSTM input, not an optional competing module.
- Historical evaluation is leave-one-year-out over 2000–2019: hold out one target year and train on the other 19 years. New U850 data are supported as inference inputs, not a request to retrain.
- Prediction domain is 100–125°E, 20–40°N; the paper focus region is 112–121°E, 23–27°N. U850-only is the intentional checkpoint-compatible configuration; SST is out of scope.
- Keep the pointwise Pearson-r heatmap as the main spatial skill view. Add task-specific focus-region evaluation: regional mean prediction/truth series, regional-series r/RMSE/MAE, mean defined pointwise r, dates/sample count, and CSV export. Do not mix this with the historical 2003 S2S chart.
- Fix scientific labels: call both maps precipitation anomaly; distinguish r=0.5 contour from any precipitation contour; do not claim negative r alone proves worse-than-climatology. The stripe-source investigation is paused until this evaluation change is complete, then resume.
- Stripe investigation resumed 2026-10-01 on the real pre1/2000 diagnostic artifact: raw/processed U850 fields are longitudinally smooth, but `e_u850.reshape(41,81,order='F')` has strong longitude-wise striping and so does `|processed × weight|`. The displayed `mode_U` map is a smooth geographic layout and is not the same array (correlation −0.3007) as the calculation weight map. Preserve the legacy F-order computation because it reproduces official CIO, but do not present its fence-like contribution map as physical geographic evidence; resolve the UI semantics before paper use.
