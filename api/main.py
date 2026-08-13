"""
FastAPI serving layer for the trained C-MAPSS RUL model.

Run from the project root with:

    uv run uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000/ for the dashboard (served as static files),
or http://localhost:8000/docs for the interactive API docs.
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import inference, schemas
from api.inference import PROJECT_ROOT
from src import data_loader as dl

SUBSET = "FD001"

app = FastAPI(title="Aircraft RUL Prognostics API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS = inference.load_artifacts(SUBSET)


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok", "subset": SUBSET, "model_name": ARTIFACTS["meta"]["model_name"]}


@app.get("/api/model-info", response_model=schemas.ModelInfo)
def model_info() -> schemas.ModelInfo:
    model_name = ARTIFACTS["meta"]["model_name"]
    leaderboard = pd.read_csv(PROJECT_ROOT / "reports" / "model_leaderboard.csv", index_col=0)
    if model_name not in leaderboard.index:
        raise HTTPException(500, f"model '{model_name}' not found in reports/model_leaderboard.csv")
    row = leaderboard.loc[model_name]
    return schemas.ModelInfo(
        subset=SUBSET,
        model_name=model_name,
        n_features=len(ARTIFACTS["meta"]["feature_cols"]),
        metrics={
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "R2": float(row["R2"]),
            "NASA_score": float(row["NASA_score"]),
        },
    )


@app.post("/api/predict/upload", response_model=schemas.FleetPredictionResponse)
async def predict_upload(file: UploadFile = File(...)) -> schemas.FleetPredictionResponse:
    """Accepts a raw C-MAPSS file (whitespace-delimited, 26 columns, no header) —
    the same format as data/raw/test_FD001.txt — and returns per-engine predictions."""
    raw_bytes = await file.read()
    try:
        raw_df = pd.read_csv(io.BytesIO(raw_bytes), sep=r"\s+", header=None, names=dl.ALL_COLS)
    except Exception as exc:
        raise HTTPException(400, f"could not parse '{file.filename}' as a raw C-MAPSS file: {exc}") from exc

    missing = set(dl.ALL_COLS) - set(raw_df.columns)
    if missing or raw_df.empty:
        raise HTTPException(400, "file does not look like a raw C-MAPSS sensor file (26 whitespace-delimited columns)")

    last_rows, trends = inference.predict_fleet(raw_df, ARTIFACTS)

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
            fail_within_20_proba=round(float(row.fail_within_20_proba), 3) if row.fail_within_20_proba is not None else None,
            top_factors=[schemas.ShapFactor(**f) for f in row.top_factors],
            health_trend=[schemas.HealthPoint(**pt) for pt in trends[int(row.unit_number)]],
        )
        for row in last_rows.itertuples(index=False)
    ]
    engines.sort(key=lambda e: e.predicted_rul)

    return schemas.FleetPredictionResponse(subset=SUBSET, n_engines=len(engines), engines=engines)


app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "frontend"), html=True), name="frontend")
