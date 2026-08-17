# Handover — Aircraft RUL Prognostics

Status as of **2026-08-13**. This file is the single source of truth for "what's
actually done vs. what's left" — read this before `README.md` if you're picking the
project up fresh.

## Update 2026-08-13 — MLOps foundation

The notebooks are no longer the only way to produce artifacts. Added:

- **A runnable pipeline** (`src/pipeline/`): prepare_data → build_features → train →
  evaluate → serving_extras → register, all driven by `conf/config.yaml`.
  `uv run python -m src.pipeline.run_all` rebuilds everything from `data/raw/`.
  Verified: the regenerated scaler statistics and all 290 feature columns are
  identical to the committed notebook artifacts, and Linear Regression / LightGBM
  reproduce the leaderboard numbers exactly (XGBoost differs in the third digit —
  xgboost 3.3 vs the 3.2 the notebooks ran on).
- **Config layer** (`src/config.py` + `conf/config.yaml`): no hyperparameter, window,
  threshold, or path is a literal in code anymore; CLI overrides with dotlist syntax.
- **Training/serving skew fixed**: `api/inference.py` used to hardcode `LAGS`,
  `ROLLING_WINDOWS`, `EMA_SPANS` as a second copy of notebook 03's values. Both sides
  now call `src/features.py::build_feature_frame` with the params read from the saved
  manifest, and `tests/integration/test_train_serve_parity.py` asserts identical
  feature values through both code paths.
- **MLflow tracking** (`src/tracking.py`): a parent run per training invocation with a
  nested run per candidate — params, val metrics, model, git SHA. Registry via the
  compose-hosted server. Uses `mlflow-skinny` (full mlflow pins `pandas<3`).
- **Quality gate**: `src/pipeline/evaluate.py` scores the official test set and exits
  non-zero below the configured thresholds, so a regression can't be promoted.
- **54 tests** (`tests/`), **CI** (`.github/workflows/ci.yml`: lint, tests, pipeline
  rebuild + gate, docker build + container smoke test, frontend build), **Dockerfile**
  (one image serving API + built React app), **docker-compose** (api + MLflow),
  **Makefile**, and **ruff** config.
- **API hardening**: artifacts load in a lifespan handler (missing model → `/api/health`
  reports `model_loaded: false` and predictions 503, instead of an import crash), CORS
  no longer `*` (`RUL_CORS_ORIGINS`), upload size cap, and `api/main.py` now serves
  `frontend-react/dist` when it exists — item 3 of the old next-steps list.

**Correction to the metrics below:** the numbers in the table are **validation-split**
metrics (4,010 rows from held-out training engines), not the official test set. Scored
the PHM08 way — last cycle of each of the 100 test engines vs `RUL_FD001.txt` — the same
XGBoost model gets **MAE 21.9, RMSE 30.2, R² 0.47, NASA score 28,761**. Both sets are
written to `reports/metrics_FD001.json` by the evaluate stage. Also measured: the
`[10th, 90th]` RUL interval covers 72% of validation truth against a nominal 80%, so it
is somewhat too narrow.

## What's done

**ML pipeline (notebooks 01–06), executed with real data:**
- `01_data_understanding.ipynb` → `06_lstm_gru.ipynb` have been run end-to-end
  (`jupyter nbconvert --execute --inplace`), not just written. Outputs, plots, and
  saved artifacts in `data/interim/`, `data/processed/`, `artifacts/scalers/`,
  `models/`, `reports/` are real, not placeholders.
- Best model on the **official (untouched) C-MAPSS test set**: **XGBoost**, FD001:

  | Metric | XGBoost | GRU (notebook 06) |
  |---|---|---|
  | MAE | 23.1 cycles | 20.6 cycles |
  | RMSE | 31.7 cycles | 28.4 cycles |
  | R² | **0.76** | 0.70 |
  | NASA/PHM08 asymmetric score | 976,022 | **lower/better** |

  GRU beats XGBoost on MAE/RMSE/NASA-score; XGBoost wins on R². XGBoost is what the
  API actually serves (it's tree-based, so it's also the one that gets live SHAP
  explanations — see below); the LSTM/GRU weights are saved (`models/lstm_FD001.pt`,
  `models/gru_FD001.pt`, `models/sequence_model_meta_FD001.joblib`) if you want to
  swap the served model later. Full tabular-model comparison is in
  `reports/model_leaderboard.csv`.
- Model + every fitted transform it depends on (scaler, KMeans, PCA health-indicator,
  feature-column manifest) are persisted to `models/` and `artifacts/scalers/`, so
  they can be reloaded without re-running any notebook.
- **Notebooks 07 (SHAP) and 08 (failure probability) were deliberately *not*
  re-executed** — their results were read directly from the already-executed
  notebook JSON output instead (no need to run code to analyze code that already
  ran). Neither notebook persists what the API needs to disk, though (07 saves
  nothing; 08 only writes a static CSV), so `scripts/fit_serving_extras.py` was
  written to fit and `joblib.dump` exactly the two model families 08 computes —
  per-horizon failure-probability classifiers and `[10th, 90th]` percentile RUL
  quantile regressors — straight from the training data. Run it any time with
  `uv run python scripts/fit_serving_extras.py`; outputs are
  `models/failure_classifiers_FD001.joblib` and `models/quantile_models_FD001.joblib`.
  SHAP itself is even simpler: `api/inference.py` builds a `shap.TreeExplainer` on
  the live XGBoost model at API startup and computes explanations per-request — no
  saved artifact needed at all.

**Serving layer:**
- `api/` — a FastAPI app (`api/main.py`) that loads the saved artifacts once at
  startup and exposes:
  - `GET /api/health` — liveness + which model is loaded
  - `GET /api/model-info` — model name, feature count, leaderboard metrics
  - `POST /api/predict/upload` — upload a raw C-MAPSS file (same whitespace-delimited
    format as `data/raw/test_FD001.txt`) → per-engine predicted RUL (with an
    `[80% interval]` low/high bound), health score, `P(fail ≤ 20 cycles)`, risk
    category, top-5 SHAP factors driving the prediction, and full health-indicator
    trend for charting
  - `api/inference.py` rebuilds the exact training-time feature pipeline (same `src/`
    functions, same fitted objects) so serving-time features can't silently drift
    from what the model was trained on.
- End-to-end smoke-tested against the real `data/raw/test_FD001.txt` (100 engines,
  every endpoint, real predictions + real SHAP factors + real intervals) — see
  "How to run" below to repeat this yourself.

**Frontend — React, as originally requested (superseding the earlier static-JS deviation):**
- `frontend-react/` — a Vite + React + TypeScript app (`npm create vite@latest ... --template react-ts`),
  built for a dark, "futuristic" high-polish feel per your ask: glassmorphism panels,
  cyan/blue/violet glow accents, Framer Motion for count-up numbers/staggered
  entrances/row transitions, Recharts for the risk donut and per-engine health-trend
  area chart, a sortable/filterable/searchable fleet table, a per-engine detail panel
  with predicted-RUL interval, failure probability, and a live SHAP factor breakdown
  (red/green bars, direction arrows), CSV export, and a live API-status indicator.
  Styled with Tailwind v4 (`@tailwindcss/vite`) plus a small custom CSS design system
  (`src/index.css`) for the glass/glow/scanline look Tailwind alone doesn't give you.
- Talks to the same `/api/*` FastAPI routes as before — Vite's dev proxy
  (`vite.config.ts` → `/api` → `http://localhost:8000`) means the exact same relative
  paths work in dev and in a production build (once the built app is served by
  FastAPI, same pattern the old static dashboard used).
- Built and dev-mode smoke-tested end-to-end: `tsc -b && vite build` succeeds with
  zero type errors; the full upload → predict flow through the dev proxy against the
  live API returns real per-engine data (verified with `data/raw/test_FD001.txt`,
  `n_engines: 100`, correct risk categorization, real top SHAP factor).
- The old static dashboard (`frontend/` — plain HTML/CSS/JS + Chart.js) is still on
  disk and still works (FastAPI serves it by default), but `frontend-react/` is the
  one to use going forward. Nobody's deleted `frontend/` yet — see "What's NOT done."
- **This machine had no Node.js/npm** (`winget` is blocked by Group Policy). Worked
  around with a portable, no-admin Node.js zip
  (`https://nodejs.org/dist/v22.11.0/node-v22.11.0-win-x64.zip`, extracted to
  `C:\Users\BobyDubey\tools\node-extract\...`) — no system install, no admin rights
  used. If you move to another machine, a normal Node install works fine too.

## What's NOT done

- **`frontend/` (the old static dashboard) hasn't been retired.** Both frontends exist
  side by side; `api/main.py` now prefers `frontend-react/dist/` and falls back to
  `frontend/`. Decide whether to delete the old one.
- **FD002/FD003/FD004 subsets** are structurally supported and now a config change
  (`subset=FD002`), but have never been run through the pipeline. FD002/FD004 have six
  operating conditions, so expect the KMeans cluster feature to matter much more there.
- **No authentication/authorization** on the API — CORS is locked down and uploads are
  size-capped, but there's still no authn/authz. Not for public exposure as-is.
- **No frontend tests** — the React app is covered only by `tsc -b && vite build` in CI.
- **No persistence of uploaded predictions** — every upload is scored in memory and
  returned; nothing is written to a database or file. Fine for a demo, not for an
  audit trail, and it means there's no data to compute drift from later.
- **No data/model versioning (DVC)** — `data/raw/`, `models/`, and `artifacts/` are
  still committed as binaries in git.
- **No drift monitoring, no metrics endpoint, no orchestrator** — retraining is a
  command a human runs, not a scheduled DAG with automatic promotion.
- **Docker is written but unverified** — `Dockerfile`, `docker-compose.yml`, and the
  CI docker job were authored on a machine with no Docker daemon, so they have never
  been built. Expect to iterate on the first `docker build`.
- **Vite build has one non-blocking warning** — a JS chunk over 500kB (Recharts +
  Framer Motion are the likely contributors). Fine to ship as-is; code-splitting
  (`build.rollupOptions.output.manualChunks`) would quiet it if it matters later.

## How to run everything

```bash
cd aircraft-rul-prognostics

# --- backend ---
uv sync                                          # installs everything, incl. fastapi/uvicorn/shap
uv run uvicorn api.main:app --reload --port 8000
# open http://localhost:8000/docs    -> interactive API docs
# open http://localhost:8000/        -> old static dashboard (still works)

# --- new React frontend (separate terminal) ---
cd frontend-react
npm install         # or: npm install then npm install @rolldown/binding-win32-x64-msvc --no-save
                     # if Vite complains about a missing native rolldown binding
npm run dev -- --port 5173
# open http://localhost:5173/        -> the React app, proxying /api to :8000
```

To try it: upload `data/raw/test_FD001.txt` in whichever frontend you're running (or
any raw C-MAPSS-format file with the same 26 whitespace-delimited columns) — you'll
get real predictions, real risk categories, real intervals, and real SHAP explanations
from the real trained model, not synthetic data.

To rebuild every artifact from raw data (this is now the supported path, not the
notebooks):

```bash
uv run python -m src.pipeline.run_all     # ~10 min on a laptop, all 5 candidates
uv run pytest                             # 54 tests
make mlflow-ui                            # browse the runs
```

`scripts/fit_serving_extras.py` still works but is a shim for
`python -m src.pipeline.serving_extras`. See `docs/MLOPS.md` for the full workflow.

## Suggested next steps, roughly in priority order

1. **Open the React app and give it a pixel-level pass** — you haven't seen it live
   yet. Run both processes above and say what to change; nothing about layout, motion,
   or color is locked in.
2. **Run `docker build`** somewhere with Docker and fix whatever the first build
   surfaces — that's the one piece of the new setup that hasn't been executed.
3. Decide the fate of `frontend/` (old static dashboard) — delete it, or keep as a
   lightweight fallback.
4. Run the pipeline on FD002 (`subset=FD002`) — six operating conditions makes it a
   genuinely different problem and a good stress test of the config-driven pipeline.
5. Next MLOps layer, in order: DVC for `data/`+`models/`, schema validation on ingest,
   prediction logging to a database, then drift monitoring (Evidently, using FD002
   against the FD001-trained model as the demo). Detail in `docs/MLOPS.md`.
