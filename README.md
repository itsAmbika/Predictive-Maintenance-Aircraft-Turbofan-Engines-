# Aircraft Engine RUL Prognostics — ML Pipeline

Remaining Useful Life (RUL) prognostics for turbofan engines, built on NASA's C-MAPSS
degradation simulation dataset (Saxena, Goebel, Simon & Eklund, PHM08 — see
`docs/Damage Propagation Modeling.pdf`), following the phased plan in
`docs/project_plan.pdf`.

This repo covers the ML pipeline (data understanding through sequence models) plus a
FastAPI serving layer and a React frontend on top of the trained model. **See
[`HANDOVER.md`](HANDOVER.md) for the current, authoritative status** — what's been
executed with real data, what's still left, and suggested next steps.
Short version: notebooks 01–06 have been run for real (FD001; best served model:
XGBoost, R² 0.76; GRU is close behind and wins on MAE/RMSE); the API additionally
serves failure probabilities, RUL prediction intervals, and live SHAP explanations
(fitted/persisted via `scripts/fit_serving_extras.py` rather than re-running notebooks
07/08); a React app (`frontend-react/`) is the primary UI, with the older static-JS
dashboard (`frontend/`) still present but superseded.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
cd aircraft-rul-prognostics
uv sync                     # creates .venv/ and installs every pinned dependency
```

Register the environment as a Jupyter kernel (needed once, so notebooks can select it):

```bash
uv run ipykernel install --user --name aircraft-rul-prognostics --display-name "Python (aircraft-rul-prognostics)"
```

Then open any notebook under `notebooks/` (VS Code Jupyter extension, JupyterLab via
`uv run jupyter lab`, or plain `jupyter notebook`) and select the
**"Python (aircraft-rul-prognostics)"** kernel.

## Data

`data/raw/` already contains the extracted C-MAPSS files (`train_FD00x.txt`,
`test_FD00x.txt`, `RUL_FD00x.txt`, `readme.txt`) for all four subsets. The pipeline
builds on **FD001** (single operating condition, single fault mode — the simplest
subset); the code is written so the same steps generalize to FD002/FD003/FD004, but
that re-run is left to you.

## Notebooks (run in order)

| Notebook | Plan phase(s) | What it does |
|---|---|---|
| `01_data_understanding.ipynb` | 0, 1 | Load raw data, physical column meanings, generate RUL, train-vs-test structure |
| `02_eda.ipynb` | 2 | Data quality, sensor variance, distributions, degradation curves, smoothing, correlation, outliers, operating conditions |
| `03_feature_engineering.ipynb` | 3, 4, 5 | Engine-grouped split, leakage-safe scaling, lag/rolling/diff/EMA features, operating-condition clustering, PCA health indicator, RUL capping |
| `04_baseline_models.ipynb` | 6, 8, 9, 10 | Constant / Linear Regression / Decision Tree / Random Forest baselines, raw-vs-capped RUL comparison, error analysis |
| `05_tree_models.ipynb` | 7, 10 | XGBoost, LightGBM, optional `GroupKFold` hyperparameter search, deeper error analysis |
| `06_lstm_gru.ipynb` | 11, 12 | LSTM and GRU sequence models on raw scaled sensor windows, compared against every tabular model |
| `07_explainability.ipynb` | 15 | SHAP global/local explanations on the best tree model, mapped to physical sensor meanings |
| `08_failure_probability.ipynb` | 16, 17, 18 | Health score, failure-probability classifiers, prediction intervals, final per-engine maintenance table |

All 8 notebooks have been executed for real on FD001 (see `HANDOVER.md` for exact
metrics). Each notebook reads its inputs from `data/interim/` or `data/processed/`
(written by the notebook before it) and writes its own outputs there — run them in
order if you re-run from scratch. Note: the API doesn't depend on 07/08 having been
*re-run* to serve their functionality — `scripts/fit_serving_extras.py` fits and
persists the failure-probability and RUL-interval models directly from training data,
and SHAP explanations are computed live at request time via `shap.TreeExplainer`.

## API + frontend

A FastAPI app serves the trained model saved by notebook 05/06 with predictions, RUL
intervals, failure probabilities, and live SHAP explanations, plus a React frontend on top:

```bash
# backend
uv sync                                          # picks up fastapi/uvicorn/shap
uv run uvicorn api.main:app --reload --port 8000
# http://localhost:8000/docs  -> interactive API docs

# frontend (separate terminal)
cd frontend-react
npm install
npm run dev -- --port 5173
# http://localhost:5173/  -> the React app (proxies /api to :8000)
```

Upload a raw C-MAPSS file (e.g. `data/raw/test_FD001.txt`) to get real per-engine RUL
predictions with an 80% interval, health scores, failure probability, risk categories,
and top SHAP factors driving each prediction. See `api/main.py` / `api/inference.py`
for the endpoints and how serving-time features are kept identical to training-time
features, and `HANDOVER.md` for the full done/not-done breakdown, including the older
static `frontend/` dashboard, still present but superseded by `frontend-react/`.

## Project structure

```
data/
  raw/         original C-MAPSS txt files
  interim/     intermediate tables (RUL added, scaled-but-not-lagged sensors)
  processed/   final model-ready feature tables
src/           reusable pipeline code, imported by every notebook
  data_loader.py        column layout, sensor/op-setting metadata, file loading
  preprocessing.py      RUL construction, sensor variance triage, scaling, engine-grouped split
  feature_engineering.py  lag/rolling/diff/EMA features, operating-condition KMeans, PCA health indicator
  sequence.py            sliding-window sequence construction for LSTM/GRU
  evaluate.py            MAE/RMSE/R2 + the PHM08/NASA asymmetric scoring function, RUL-bucketed error analysis
  health.py              health score, failure-probability labels, risk categorization
  torch_models.py        LSTM/GRU model definition + shared training loop
notebooks/     01-08, run in order (see table above)
models/        saved model artifacts (created by 04/05/06)
artifacts/     saved scalers, PCA/KMeans objects, feature-column manifests (created by 03)
reports/       leaderboard CSV, final maintenance table, figures
docs/          the PHM08 paper and the project plan this pipeline follows
api/           FastAPI serving layer (main.py, inference.py, schemas.py)
frontend/      old static dashboard (index.html, app.js, style.css) — superseded, still works
frontend-react/  React + TypeScript + Tailwind + Framer Motion + Recharts UI (primary frontend)
scripts/       fit_serving_extras.py — persists notebook 08's failure-prob/quantile models
```

## Design principles followed throughout

- **No leakage**: every scaler/KMeans/PCA is fit on training engines only, then applied
  unchanged to validation/test engines. Engine IDs never appear on both sides of a split
  (`src/preprocessing.assert_no_engine_overlap`).
- **Engine-grouped splitting**: validation is a `GroupShuffleSplit` on `unit_number`, not
  a random row split — otherwise cycles from the same engine leak across train/val.
- **The official test set stays untouched** until final evaluation in each notebook —
  it is never used for fitting, tuning, or model selection.
- **Application-level outputs are labeled as such**: the PCA health indicator, RUL cap,
  health score formula, and LOW/MEDIUM/HIGH risk thresholds are all modeling decisions
  made in this project, not values published by NASA or the PHM08 paper.

## Explicitly out of scope for now

Transformer/TCN models, anomaly detection, engine-similarity search, sensitivity
analysis, authentication, automated tests, model versioning/retraining pipelines, and
Docker packaging. A FastAPI serving layer and a React frontend now exist (see above).
Full detail on what's done vs. left is in `HANDOVER.md`.
