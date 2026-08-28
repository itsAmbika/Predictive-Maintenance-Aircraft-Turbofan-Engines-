---
title: Aircraft Engine RUL Prognostics
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Predict remaining useful life of turbofan engines from NASA C-MAPSS sensor data
---

# Aircraft Engine RUL Prognostics

Predicts the **Remaining Useful Life (RUL)** of turbofan engines from raw sensor
telemetry, using NASA's C-MAPSS run-to-failure dataset (Saxena, Goebel, Simon &
Eklund, PHM08).

Click **Try sample data** to score a real 100-engine fleet, or drop in your own
whitespace-delimited C-MAPSS file (26 columns: engine id, cycle, 3 operating
settings, 21 sensors).

## What you get per engine

- **Predicted RUL** in cycles, with an 80% prediction interval
- **Health score** and a LOW / MEDIUM / HIGH maintenance risk category
- **P(failure ≤ 20 cycles)** from a dedicated classifier
- **SHAP attributions** — which sensors drove this specific prediction
- The engine's **health-indicator trajectory** over its observed life

## Model

XGBoost over **290 features** engineered from 21 raw sensors (lag, rolling
mean/std/min/max, EMA, first differences, an operating-condition cluster, and a
PCA health indicator). Six constant sensors are dropped; every transform is fit
on training engines only, with engine-grouped splits so no engine's cycles
appear on both sides.

| Metric | Validation (4,010 rows) | Official PHM08 test (100 engines) |
|---|---|---|
| MAE | 23.19 cycles | **21.86 cycles** |
| RMSE | 32.06 | 30.20 |
| R² | 0.756 | 0.472 |
| NASA/PHM08 score | 1,194,469 | 28,761 |

The two columns measure different things: the official protocol scores only the
*last* cycle of each test engine, which has far less target variance than
scoring every row. MAE on near-failure engines (RUL ≤ 20) is **4.0 cycles** —
the model is most accurate exactly where maintenance decisions are made.

## How it's built

The whole pipeline is reproducible from raw data with one command
(`python -m src.pipeline.run_all`): config-driven stages, MLflow tracking, a
quality gate on the official test set, 54 tests, and CI. Training and serving
share one feature-building function, so the API cannot drift from the model.

Source and full MLOps write-up:
**https://github.com/itsAmbika/Predictive-Maintenance-Aircraft-Turbofan-Engines-**

## Caveats

Risk thresholds and the health score are application-level choices made for this
project, not certified maintenance standards. The model is trained on FD001
(one operating condition, one fault mode); FD002/FD004 have six operating
conditions and would need a retrain.
