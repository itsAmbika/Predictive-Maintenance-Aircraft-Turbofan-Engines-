# MLOps guide

How this project is built, trained, tested, and shipped. `README.md` covers the
modeling; this file covers the machinery around it.

## The pipeline

The notebooks are narrative and EDA. **The pipeline is the source of truth for
every artifact** — anything in `models/`, `artifacts/`, or `reports/` can be
regenerated from `data/raw/` with one command:

```bash
uv run python -m src.pipeline.run_all
```

| Stage | Module | Reads | Writes |
|---|---|---|---|
| 1. prepare_data | `src/pipeline/prepare_data.py` | `data/raw/` | `data/interim/*.parquet`, `reports/data_stats_<subset>.json` |
| 2. build_features | `src/pipeline/build_features.py` | `data/interim/` | `data/processed/*.parquet`, `artifacts/scalers/*.joblib` |
| 3. train | `src/pipeline/train.py` | `data/processed/` | `models/best_model_<subset>*.joblib`, `reports/model_leaderboard.csv`, MLflow runs |
| 4. evaluate | `src/pipeline/evaluate.py` | `models/`, `data/processed/` | `reports/metrics_<subset>.json`, `reports/error_by_rul_bin_<subset>.csv` |
| 5. serving_extras | `src/pipeline/serving_extras.py` | `data/processed/` | `models/failure_classifiers_*.joblib`, `models/quantile_models_*.joblib` |
| 6. register | `src/pipeline/register.py` | `models/`, `reports/` | MLflow Model Registry version + `@champion` alias |

Each stage runs standalone (`python -m src.pipeline.build_features`) and every
stage takes the same config overrides.

## Configuration

`conf/config.yaml` holds every parameter that can change a model — split sizes,
lags and rolling windows, hyperparameters, the RUL cap, risk thresholds, the
quality gate. Code reads it through `src/config.py`, which validates the YAML
against a dataclass schema (a typo like `featurs.lags` fails at load, not
silently).

Override anything from the CLI with dotlist syntax:

```bash
uv run python -m src.pipeline.run_all subset=FD003 target.train_on=RUL_capped models.candidates=[xgboost]
```

The resolved config is written into the feature manifest and the model meta, so
every artifact records exactly which settings produced it.

## No training/serving skew

Both sides call the same function:

```
src/features.py::build_feature_frame
        ↑                        ↑
src/pipeline/build_features.py   api/inference.py
```

The feature params (lags, windows, EMA spans) live in
`artifacts/scalers/feature_manifest_<subset>.joblib`, so serving reads back the
values training actually used instead of keeping its own copy. Retrain with
different windows and the API follows on restart.

`tests/integration/test_train_serve_parity.py` asserts this: it runs the same raw
rows through both paths and requires identical values for all 290 feature columns.

## Experiment tracking

Runs go to MLflow. With no configuration, that's a local `mlruns/` file store —
no server, works on a fresh clone:

```bash
make train
make mlflow-ui        # http://localhost:5000
```

For the Model Registry (which the file store doesn't support), start the server
and point the pipeline at it:

```bash
docker compose up mlflow
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m src.pipeline.run_all mlflow.register=true
```

The train stage logs one parent run per invocation with a nested run per
candidate model: params, validation metrics, fit time, the serialized model, and
tags for git SHA and subset. The evaluate stage logs official test-set metrics
and the gate result against the same experiment.

> The project depends on `mlflow-skinny`, not `mlflow`: the full package pins
> `pandas<3`, which conflicts with this project's pandas 3. Skinny is the same
> client; the server runs from its own image in `docker-compose.yml`.

## Quality gate

`src/pipeline/evaluate.py` scores the **official** test set — the last cycle of
each of the 100 test engines, compared against `RUL_FD001.txt` — and applies the
thresholds in `evaluation.gate`. The stage exits non-zero on failure, so CI and
retraining jobs stop before a regression reaches the registry.

Two metric sets are reported, because they are easy to confuse:

| | what it measures | current (XGBoost, FD001) |
|---|---|---|
| `official_test_last_cycle` | 100 engines, last cycle each — the PHM08 setup | MAE 21.9, RMSE 30.2, **R² 0.47** |
| validation split | 4,010 rows from held-out training engines | MAE 23.2, RMSE 32.1, **R² 0.76** |

The R² 0.76 quoted in `README.md`/`HANDOVER.md` is the **validation** number.
`/api/model-info` also serves the validation leaderboard row. Both are in
`reports/metrics_<subset>.json`.

## Tests

```bash
make test-fast     # no artifacts needed — what CI runs on every push
make test          # everything
```

- `tests/unit/` — leakage rules (scaler fit on train only, engine-grouped splits,
  lag features never crossing an engine boundary), the asymmetric NASA score,
  risk/health rules, config loading.
- `tests/integration/test_pipeline_stages.py` — the whole DAG on a synthetic
  6-engine dataset in a tmp dir; seconds, not minutes.
- `tests/integration/test_api.py` — endpoint contracts against `TestClient`.
- `tests/integration/test_train_serve_parity.py` — the skew test above.

Tests needing real trained artifacts are marked `slow` and skip when absent.

## CI

`.github/workflows/ci.yml` runs five jobs: `lint` (ruff), `test`, `pipeline`
(rebuilds artifacts from the committed raw data, runs the gate, then the slow
tests), `docker` (builds the image, boots it, scores the real test file through
the running container), and `frontend` (typecheck + Vite build).

## Container

One image serves the API and the production React build:

```bash
make docker-build
docker run --rm -p 8000:8000 rul-api:local     # http://localhost:8000
```

The image installs with `--no-dev` (no mlflow/pytest/jupyter) and skips torch —
only the LSTM/GRU training code imports it, and the served model is XGBoost.
Mount `models/` and `artifacts/` to swap in a retrained model without rebuilding.

## Deploying to Hugging Face Spaces

The image runs as-is on a Docker Space: it listens on `PORT` (default 7860, what
Spaces expects), runs as uid 1000 with a writable `HOME`, and points matplotlib's
and numba's cache dirs at `/tmp` -- the usual reasons a container works locally and
fails on Spaces.

1. Create the Space at <https://huggingface.co/new-space> -- **SDK: Docker**,
   **Template: Blank**. Note the id, e.g. `your-name/aircraft-rul-prognostics`.
2. Create a **write** token at <https://huggingface.co/settings/tokens>.
3. Push the deploy tree:

```bash
HF_TOKEN=hf_xxx ./deploy/huggingface/sync.sh your-name/aircraft-rul-prognostics
```

The script pushes only what the image needs -- no notebooks, tests, docs, or the
~30MB of raw C-MAPSS files (one 2.2MB sample is kept so the demo works), and it
swaps in `deploy/huggingface/README.md`, which carries the YAML card metadata
Spaces require. The GitHub README stays clean.

The first build takes ~10-15 minutes (npm build + the Python dependency install);
watch the Space's **Logs** tab. Afterwards, `.github/workflows/deploy-hf.yml`
redeploys on demand -- set the `HF_TOKEN` secret and `HF_SPACE_ID` variable in the
repo, then uncomment its `push` trigger to redeploy on every merge to main.

Free CPU Spaces sleep after inactivity and cold-start in roughly a minute, which
is fine for a portfolio demo. Note the API has **no authentication** -- anyone who
finds the Space can post files at the inference endpoint (capped at 32MB by
`RUL_MAX_UPLOAD_BYTES`). Acceptable for a public demo, not for anything else.

## Free hosting options

Measured footprint of the serving process: **~272MB peak** while scoring 100
engines (imports 239MB, model + SHAP explainer 24MB, a full request 9MB). That
fits a 512MB free tier, so the same Docker image works everywhere -- no need to
trade away the React frontend for a Python-native UI.

| Option | Free? | Keeps this stack? | Catch |
|---|---|---|---|
| **Render** (`render.yaml`) | yes, no card | yes -- same Dockerfile | sleeps after ~15 min idle, ~50s cold start |
| Hugging Face Spaces, Docker SDK | yes on CPU basic | yes | see note below |
| Google Cloud Run | effectively $0 at demo traffic | yes | requires a card on file |
| Fly.io / Koyeb / Railway | trial credit only | yes | card required, credit expires |
| HF Spaces, Gradio or Streamlit SDK | yes, no card | no -- rewrite UI in Python | loses the React app |
| Vercel / Netlify functions | yes | no | 250MB bundle cap; xgboost + shap + scipy won't fit |

Render is the default recommendation: it reads `render.yaml`, builds the same
Dockerfile, injects `$PORT` (which the CMD honours), and needs no payment method.

> On Hugging Face: the **Docker SDK itself is free on CPU basic hardware**. What
> costs money there is upgraded hardware, persistent storage, and Dev Mode -- if
> you hit a paywall while creating the Space, check which of those got switched
> on rather than assuming Docker is the paid part.

## Operational notes

- **CORS** is no longer `*`. Set `RUL_CORS_ORIGINS` (comma-separated) for
  deployment; it defaults to the Vite dev server.
- **Uploads** are capped by `RUL_MAX_UPLOAD_BYTES` (32 MB default).
- **Startup**: artifacts load in the FastAPI lifespan handler. A missing model
  makes `/api/health` report `model_loaded: false` and prediction return 503,
  instead of crashing the process at import.
- **Subset**: `RUL_SUBSET` env var, or `subset` in the config.
- **Sample data**: `GET /api/sample` serves `data/raw/test_<subset>.txt`, which
  backs the frontend's "Try sample data" button so a first-time visitor can score
  a real fleet without downloading the NASA dataset. Returns 404 if absent.

## What's deliberately not here yet

Tier 2+ of the roadmap: DVC for data/model versioning, schema validation
(pandera) at ingest, prediction logging to a database, drift monitoring with
Evidently (FD002 against an FD001-trained model is the natural demo),
Prometheus/Grafana observability, an orchestrator (Prefect/Airflow) for scheduled
retraining, API authentication, and a model card.
