"""Pydantic response models for the RUL prognostics API."""

from __future__ import annotations

from pydantic import BaseModel


class HealthPoint(BaseModel):
    cycle: int
    health_indicator: float


class ShapFactor(BaseModel):
    feature: str
    impact: float
    direction: str


class EnginePrediction(BaseModel):
    unit_number: int
    last_cycle: int
    predicted_rul: float
    rul_low: float | None = None
    rul_high: float | None = None
    health_score: float
    risk: str
    risk_action: str
    fail_within_20_proba: float | None = None
    top_factors: list[ShapFactor] = []
    health_trend: list[HealthPoint]


class FleetPredictionResponse(BaseModel):
    subset: str
    n_engines: int
    engines: list[EnginePrediction]
    # Horizon behind `fail_within_20_proba`, so the UI can label it from the
    # response instead of assuming the configured default.
    failure_horizon: int = 20


class ModelInfo(BaseModel):
    subset: str
    model_name: str
    n_features: int
    metrics: dict[str, float]
    # Provenance -- filled from the model meta written by the training stage.
    trained_at: str | None = None
    git_sha: str | None = None
    mlflow_run_id: str | None = None
