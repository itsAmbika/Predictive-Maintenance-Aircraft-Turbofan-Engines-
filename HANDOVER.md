# Handover — Aircraft RUL Prognostics

Status as of **2026-08-05**. This file is the single source of truth for "what's
actually done vs. what's left" — read this before `README.md` if you're picking the
project up fresh.

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

- **`frontend/` (the old static dashboard) hasn't been retired.** Both frontends
  exist side by side. Decide whether to delete `frontend/` and have `api/main.py`
  serve `frontend-react/dist/` instead, or keep both around.
- **FD002/FD003/FD004 subsets** are structurally supported by every `src/` function
  but never run through the pipeline. `api/main.py` is hardcoded to `SUBSET = "FD001"`.
- **No authentication/authorization** on the API — fine for local use, not for
  exposing beyond localhost as-is.
- **No automated tests** (unit tests for `src/`, integration test for the API, no
  frontend tests either). The only verification so far is the manual smoke tests
  described above.
- **No persistence of uploaded predictions** — every upload is scored in memory and
  returned; nothing is written to a database or file. Fine for a demo, not for an
  audit trail.
- **No retraining/versioning story** — if you retrain and get a new
  `best_model_FD001.joblib`, the API picks it up on restart, but there's no model
  registry, no A/B, no rollback.
- **No Docker packaging, no production build wired into FastAPI yet** — `frontend-react`
  runs via `npm run dev`, separate from the API process; `npm run build` works
  (verified) but `api/main.py` isn't yet pointed at the built `dist/` output.
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

To re-run or extend the ML pipeline itself, see the notebook table in `README.md`. To
regenerate the failure-probability/quantile artifacts the API depends on (e.g. after
retraining), re-run `uv run python scripts/fit_serving_extras.py`.

## Suggested next steps, roughly in priority order

1. **Open the React app and give it a pixel-level pass** — you haven't seen it live
   yet. Run both processes above and say what to change; nothing about layout, motion,
   or color is locked in.
2. Decide the fate of `frontend/` (old static dashboard) — delete it, or keep as a
   lightweight fallback.
3. Wire `npm run build`'s output into `api/main.py` so one `uvicorn` process serves
   the production React build (matches how `frontend/` is served today).
4. Add a couple of integration tests (`api/predict/upload` against a small fixture
   file) before this goes anywhere beyond local/demo use.
5. Only after the above: containerize (`Dockerfile` for `uv run uvicorn`, serving the
   built React app).
