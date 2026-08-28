"""
FastAPI serving layer for the trained C-MAPSS RUL model.

Run from the project root with:

    uv run uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000/ for the dashboard (the built React app if
``frontend-react/dist`` exists, otherwise the legacy static one) or
http://localhost:8000/docs for the interactive API docs.

Artifacts load in the lifespan handler rather than at import time, so importing
this module (in tests, or to inspect the routes) does not require model files on
disk, and a failed load surfaces as a clean 503 instead of an import crash.
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import inference, schemas
from src import data_loader as dl
from src.config import PROJECT_ROOT, load_config

logger = logging.getLogger("api")

CFG = load_config()
SUBSET = os.environ.get("RUL_SUBSET", CFG.subset)
# Uploads are read fully into memory before parsing; cap them so a stray large
# file can't exhaust the process.
MAX_UPLOAD_BYTES = int(os.environ.get("RUL_MAX_UPLOAD_BYTES", 32 * 1024 * 1024))
# Comma-separated origins; defaults to the Vite dev server. Not "*", so the API
# stays safe to expose beyond localhost once auth is added.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("RUL_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]

STATE: dict = {"artifacts": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["artifacts"] = inference.load_artifacts(SUBSET, cfg=CFG)
        logger.info("loaded model '%s' for subset %s", STATE["artifacts"]["meta"]["model_name"], SUBSET)
    except Exception as exc:  # noqa: BLE001 - reported through /api/health instead of crashing
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("model artifacts failed to load: %s", STATE["error"])
    yield
    STATE.clear()


app = FastAPI(title="Aircraft RUL Prognostics API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_artifacts() -> dict:
    if STATE.get("artifacts") is None:
        raise HTTPException(
            503,
            f"model artifacts unavailable: {STATE.get('error') or 'not loaded yet'}. "
            "Run `python -m src.pipeline.run_all` to build them.",
        )
    return STATE["artifacts"]


@app.get("/api/health")
def health_check() -> dict:
    """Liveness + whether the model actually loaded (readiness)."""
    artifacts = STATE.get("artifacts")
    return {
        "status": "ok" if artifacts else "degraded",
        "subset": SUBSET,
        "model_loaded": artifacts is not None,
        "model_name": artifacts["meta"]["model_name"] if artifacts else None,
        "error": STATE.get("error"),
    }


@app.get("/api/model-info", response_model=schemas.ModelInfo)
def model_info() -> schemas.ModelInfo:
    artifacts = get_artifacts()
    meta = artifacts["meta"]
    model_name = meta["model_name"]

    leaderboard_path = PROJECT_ROOT / str(CFG.paths.reports) / "model_leaderboard.csv"
    leaderboard = pd.read_csv(leaderboard_path, index_col=0)
    if model_name not in leaderboard.index:
        raise HTTPException(500, f"model '{model_name}' not found in {leaderboard_path.name}")
    row = leaderboard.loc[model_name]

    return schemas.ModelInfo(
        subset=SUBSET,
        model_name=model_name,
        n_features=len(meta["feature_cols"]),
        metrics={
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "R2": float(row["R2"]),
            "NASA_score": float(row["NASA_score"]),
        },
        trained_at=meta.get("trained_at"),
        git_sha=meta.get("git_sha"),
        mlflow_run_id=meta.get("mlflow_run_id"),
    )


@app.get("/api/sample")
def sample_file() -> FileResponse:
    """One real C-MAPSS file, so a visitor can try the demo without the dataset.

    Shipped in the image (see Dockerfile); absent in checkouts that never
    downloaded the raw data, which is reported as a 404 rather than a crash.
    """
    path = PROJECT_ROOT / str(CFG.paths.raw) / f"test_{SUBSET}.txt"
    if not path.exists():
        raise HTTPException(404, f"sample file not available for subset {SUBSET}")
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.post("/api/predict/upload", response_model=schemas.FleetPredictionResponse)
async def predict_upload(file: UploadFile = File(...)) -> schemas.FleetPredictionResponse:
    """Accepts a raw C-MAPSS file (whitespace-delimited, 26 columns, no header) --
    the same format as data/raw/test_FD001.txt -- and returns per-engine predictions."""
    artifacts = get_artifacts()

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit")

    try:
        raw_df = pd.read_csv(io.BytesIO(raw_bytes), sep=r"\s+", header=None, names=dl.ALL_COLS)
    except Exception as exc:
        raise HTTPException(400, f"could not parse '{file.filename}' as a raw C-MAPSS file: {exc}") from exc

    missing = set(dl.ALL_COLS) - set(raw_df.columns)
    if missing or raw_df.empty or raw_df.isna().all(axis=None):
        raise HTTPException(400, "file does not look like a raw C-MAPSS sensor file (26 whitespace-delimited columns)")
    if raw_df[dl.ALL_COLS].isna().any(axis=None):
        raise HTTPException(400, "file has missing values in the 26 expected columns")

    last_rows, trends = inference.predict_fleet(raw_df, artifacts)

    engines = [
        schemas.EnginePrediction(
            unit_number=int(row.unit_number),
            last_cycle=int(row.cycle),
            predicted_rul=round(float(row.RUL_pred), 1),
            rul_low=round(float(row.rul_pred_low), 1) if row.rul_pred_low is not None else None,
            rul_high=round(float(row.rul_pred_high), 1) if row.rul_pred_high is not None else None,
            health_score=round(float(row.health_score), 1),
            risk=row.risk,
            risk_action=row.risk_action,
            fail_within_20_proba=round(float(row.fail_within_20_proba), 3)
            if row.fail_within_20_proba is not None
            else None,
            top_factors=[schemas.ShapFactor(**f) for f in row.top_factors],
            health_trend=[schemas.HealthPoint(**pt) for pt in trends[int(row.unit_number)]],
        )
        for row in last_rows.itertuples(index=False)
    ]
    engines.sort(key=lambda e: e.predicted_rul)

    return schemas.FleetPredictionResponse(
        subset=SUBSET,
        n_engines=len(engines),
        engines=engines,
        failure_horizon=int(CFG.serving.failure_horizon),
    )


def _frontend_dir() -> Path:
    """Prefer the production React build; fall back to the legacy static dashboard."""
    dist = PROJECT_ROOT / "frontend-react" / "dist"
    return dist if (dist / "index.html").exists() else PROJECT_ROOT / "frontend"


app.mount("/", StaticFiles(directory=str(_frontend_dir()), html=True), name="frontend")
