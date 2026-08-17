"""
Serving-time inference pipeline for a trained C-MAPSS RUL model.

Features are built by :func:`src.features.build_feature_frame` -- the same
function ``src/pipeline/build_features.py`` uses at training time -- with the
fitted transforms *and* the feature params (lags, windows, EMA spans) read back
from the saved manifest. Serving therefore cannot drift from training: retrain
with different windows and the API follows automatically.

Only the *last* observed cycle of each engine is scored, matching how RUL is
evaluated on the official C-MAPSS test set.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap
from omegaconf import DictConfig

from src import data_loader as dl
from src import health
from src.config import PROJECT_ROOT, load_config
from src.features import FeatureParams, FittedTransforms, build_feature_frame

__all__ = ["PROJECT_ROOT", "load_artifacts", "build_features", "predict_fleet"]

# How a SHAP-attributed feature column maps back to a human-readable sensor name,
# mirroring notebooks/07_explainability.ipynb. Falls back to the raw column name.
_SENSOR_NUMS_IN_COL = {int(num): info for num, info in ((k.split("_")[1], v) for k, v in dl.SENSOR_INFO.items())}


def _readable_feature_name(col: str) -> str:
    for token in col.split("_"):
        if token.isdigit() and int(token) in _SENSOR_NUMS_IN_COL:
            info = _SENSOR_NUMS_IN_COL[int(token)]
            return f"{info['symbol']} - {info['description']} ({col})"
    return col


def load_artifacts(subset: str | None = None, cfg: DictConfig | None = None) -> dict:
    """Load every fitted transform + the trained model for one C-MAPSS subset."""
    cfg = cfg or load_config()
    subset = subset or cfg.subset
    artifacts_dir = PROJECT_ROOT / str(cfg.paths.artifacts)
    models_dir = PROJECT_ROOT / str(cfg.paths.models)

    manifest = joblib.load(artifacts_dir / f"feature_manifest_{subset}.joblib")
    model = joblib.load(models_dir / f"best_model_{subset}.joblib")

    failure_clf_path = models_dir / f"failure_classifiers_{subset}.joblib"
    quantile_path = models_dir / f"quantile_models_{subset}.joblib"

    return {
        "subset": subset,
        "cfg": cfg,
        "fitted": FittedTransforms.load(artifacts_dir, subset),
        "feature_params": FeatureParams.from_manifest(manifest),
        "manifest": manifest,
        "model": model,
        "meta": joblib.load(models_dir / f"best_model_{subset}_meta.joblib"),
        # Optional extras -- present once `python -m src.pipeline.serving_extras` has
        # run; callers must handle None (those response fields are simply omitted).
        "failure_classifiers": joblib.load(failure_clf_path) if failure_clf_path.exists() else None,
        "quantile_models": joblib.load(quantile_path) if quantile_path.exists() else None,
        # TreeExplainer is exact and cheap for tree ensembles -- built once at
        # startup and reused for every request.
        "shap_explainer": shap.TreeExplainer(model),
    }


def build_features(raw_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Apply the fitted scaler/KMeans/PCA + temporal features to raw sensor rows.

    ``raw_df`` must have the same columns as a raw C-MAPSS file
    (``src.data_loader.ALL_COLS``): unit_number, cycle, 3 op settings, 21 sensors.

    Lag/diff features are undefined for an engine's first few cycles; since only
    the last row per engine is scored, those are filled rather than dropped so an
    engine shorter than the largest lag still gets a prediction.
    """
    return build_feature_frame(raw_df, artifacts["fitted"], artifacts["feature_params"], na_policy="fill")


def _top_shap_factors(shap_row: np.ndarray, feature_cols: list[str], top_n: int = 5) -> list[dict]:
    """Rank one row's SHAP values by |impact| and return the top few, human-readable.

    Positive means "pushed the predicted RUL up" (healthier); negative means
    "pushed it down" (closer to failure) -- SHAP's own sign convention.
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
    serving = artifacts["cfg"].serving
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
    last_rows["health_score"] = health.health_score(last_rows["RUL_pred"], rul_healthy=serving.rul_healthy)
    last_rows["risk"] = health.risk_category(
        last_rows["RUL_pred"],
        high_below=serving.risk_high_below,
        medium_below=serving.risk_medium_below,
    ).values
    last_rows["risk_action"] = last_rows["risk"].map(health.risk_action)

    if artifacts["quantile_models"] is not None:
        last_rows["rul_pred_low"] = artifacts["quantile_models"]["low"].predict(X_last)
        last_rows["rul_pred_high"] = artifacts["quantile_models"]["high"].predict(X_last)
    else:
        last_rows["rul_pred_low"] = None
        last_rows["rul_pred_high"] = None

    classifiers = artifacts["failure_classifiers"]
    horizon = int(serving.failure_horizon)
    if classifiers is not None and horizon in classifiers:
        last_rows["fail_within_20_proba"] = classifiers[horizon].predict_proba(X_last)[:, 1]
    else:
        last_rows["fail_within_20_proba"] = None

    shap_values = artifacts["shap_explainer"].shap_values(X_last)
    last_rows["top_factors"] = [
        _top_shap_factors(shap_values[i], feature_cols, top_n=int(serving.shap_top_n)) for i in range(len(last_rows))
    ]

    return last_rows, trends
