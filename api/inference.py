"""
Serving-time inference pipeline for a trained C-MAPSS RUL model.

Rebuilds the exact same feature pipeline fit in notebooks/03_feature_engineering.ipynb
(scale -> lag/rolling/diff/EMA -> operating-condition cluster -> PCA health indicator)
by reusing the same src/ functions and the fitted artifacts they saved, so serving-time
features are identical to training-time features. Only the *last* observed cycle of
each engine is scored, matching how RUL is evaluated in the official C-MAPSS test set.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src import data_loader as dl
from src import feature_engineering as fe
from src import health
from src import preprocessing as pp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "scalers"
MODELS_DIR = PROJECT_ROOT / "models"

LAGS = [1, 2, 3]
ROLLING_WINDOWS = [5, 10, 20]
EMA_SPANS = [5, 10]

# How the top SHAP-attributed feature columns map back to a human-readable sensor
# name, mirroring the mapping notebooks/07_explainability.ipynb builds from
# src.data_loader.SENSOR_INFO. Falls back to the raw column name if it isn't a
# recognized sensor-derived feature.
_SENSOR_NUMS_IN_COL = {int(num): info for num, info in ((k.split("_")[1], v) for k, v in dl.SENSOR_INFO.items())}


def _readable_feature_name(col: str) -> str:
    for token in col.split("_"):
        if token.isdigit() and int(token) in _SENSOR_NUMS_IN_COL:
            info = _SENSOR_NUMS_IN_COL[int(token)]
            return f"{info['symbol']} - {info['description']} ({col})"
    return col


def load_artifacts(subset: str = "FD001") -> dict:
    """Load every fitted transform + the trained model for one C-MAPSS subset."""
    manifest = joblib.load(ARTIFACTS_DIR / f"feature_manifest_{subset}.joblib")
    health_bundle = joblib.load(ARTIFACTS_DIR / f"health_indicator_pca_{subset}.joblib")

    failure_clf_path = MODELS_DIR / f"failure_classifiers_{subset}.joblib"
    quantile_path = MODELS_DIR / f"quantile_models_{subset}.joblib"
    model = joblib.load(MODELS_DIR / f"best_model_{subset}.joblib")

    return {
        "subset": subset,
        "scaler": joblib.load(ARTIFACTS_DIR / f"sensor_scaler_{subset}.joblib"),
        "kmeans": joblib.load(ARTIFACTS_DIR / f"operating_condition_kmeans_{subset}.joblib"),
        "health_scaler": health_bundle["scaler"],
        "health_pca": health_bundle["pca"],
        "health_sign_flipped": health_bundle["sign_flipped"],
        "manifest": manifest,
        "model": model,
        "meta": joblib.load(MODELS_DIR / f"best_model_{subset}_meta.joblib"),
        # Optional extras -- present once scripts/fit_serving_extras.py has been run;
        # callers must handle None (health/failure-probability/interval columns are
        # simply omitted from the response rather than raising).
        "failure_classifiers": joblib.load(failure_clf_path) if failure_clf_path.exists() else None,
        "quantile_models": joblib.load(quantile_path) if quantile_path.exists() else None,
        # TreeExplainer is exact and cheap for tree ensembles (XGBoost/LightGBM/RF) --
        # built once at startup and reused for every prediction request, exactly like
        # notebooks/07_explainability.ipynb does, just computed live instead of only
        # inside a notebook cell.
        "shap_explainer": shap.TreeExplainer(model),
    }


def build_features(raw_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Apply the fitted scaler/KMeans/PCA + temporal features to raw sensor rows.

    ``raw_df`` must have the same columns as a raw C-MAPSS file
    (``src.data_loader.ALL_COLS``): unit_number, cycle, 3 op settings, 21 sensors.
    """
    feature_sensors = artifacts["manifest"]["feature_sensors"]

    out = pp.apply_scaler(raw_df, feature_sensors, artifacts["scaler"])
    out = fe.add_lag_features(out, feature_sensors, lags=LAGS)
    out = fe.add_rolling_features(out, feature_sensors, windows=ROLLING_WINDOWS)
    out = fe.add_diff_features(out, feature_sensors, periods=[1])
    out = fe.add_ema_features(out, feature_sensors, spans=EMA_SPANS)
    out = fe.add_operating_condition_cluster(out, dl.OP_SETTING_COLS, artifacts["kmeans"])
    out = fe.add_health_indicator(out, feature_sensors, artifacts["health_scaler"], artifacts["health_pca"])
    if artifacts["health_sign_flipped"]:
        out["health_indicator"] = -out["health_indicator"]

    # Lag/diff features are undefined for an engine's first few cycles. We only ever
    # score the LAST row of each engine below, so this only matters for engines with
    # fewer cycles than the largest lag -- fill defensively rather than drop rows.
    out = out.fillna(0.0)
    return out


def _top_shap_factors(shap_row: np.ndarray, feature_cols: list[str], top_n: int = 5) -> list[dict]:
    """Rank one row's SHAP values by |impact| and return the top few, human-readable.

    Positive shap_value here means "pushed the predicted RUL up" (healthier); negative
    means "pushed it down" (closer to failure) -- same sign convention as SHAP itself.
    """
    order = np.argsort(-np.abs(shap_row))[:top_n]
    return [
        {
            "feature": _readable_feature_name(feature_cols[i]),
            "impact": round(float(shap_row[i]), 3),
            "direction": "lowers_rul" if shap_row[i] < 0 else "raises_rul",
        }
        for i in order
    ]


def predict_fleet(raw_df: pd.DataFrame, artifacts: dict) -> tuple[pd.DataFrame, dict]:
    """Score the most recent cycle of every engine in ``raw_df``.

    Returns (one row per engine with RUL_pred/health_score/risk/failure-probability/
    RUL-interval/top SHAP factors, {unit_number: health trend over all observed
    cycles}) for the fleet table + per-engine detail chart.
    """
    feature_cols = artifacts["meta"]["feature_cols"]
    model = artifacts["model"]

    fe_df = build_features(raw_df, artifacts).sort_values(["unit_number", "cycle"])

    trends = {
        int(uid): g[["cycle", "health_indicator"]].round(4).to_dict("records")
        for uid, g in fe_df.groupby("unit_number")
    }

    last_rows = fe_df.groupby("unit_number", as_index=False).tail(1).reset_index(drop=True)
    X_last = last_rows[feature_cols]
    last_rows["RUL_pred"] = model.predict(X_last)
    last_rows["health_score"] = health.health_score(last_rows["RUL_pred"])
    last_rows["risk"] = health.risk_category(last_rows["RUL_pred"]).values
    last_rows["risk_action"] = last_rows["risk"].map(health.risk_action)

    # Optional extras -- only populated once scripts/fit_serving_extras.py has been run.
    if artifacts["quantile_models"] is not None:
        last_rows["rul_pred_low"] = artifacts["quantile_models"]["low"].predict(X_last)
        last_rows["rul_pred_high"] = artifacts["quantile_models"]["high"].predict(X_last)
    else:
        last_rows["rul_pred_low"] = None
        last_rows["rul_pred_high"] = None

    if artifacts["failure_classifiers"] is not None:
        last_rows["fail_within_20_proba"] = artifacts["failure_classifiers"][20].predict_proba(X_last)[:, 1]
    else:
        last_rows["fail_within_20_proba"] = None

    shap_values = artifacts["shap_explainer"].shap_values(X_last)
    top_factors_by_row = [_top_shap_factors(shap_values[i], feature_cols) for i in range(len(last_rows))]
    last_rows["top_factors"] = top_factors_by_row

    return last_rows, trends
