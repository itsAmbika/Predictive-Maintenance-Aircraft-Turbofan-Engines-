"""Stage 3 -- train every candidate model, log to MLflow, persist the best one.

Replaces the model-fitting halves of notebooks 04 and 05. Each candidate becomes
a nested MLflow run under one parent run, with its params, validation metrics,
and the model itself logged; the winner (by ``models.selection_metric``) is also
written to ``models/best_model_<subset>.joblib`` -- the file the API loads.

    uv run python -m src.pipeline.train
    uv run python -m src.pipeline.train models.candidates=[xgboost] models.xgboost.max_depth=8
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any

import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src import evaluate as ev
from src import tracking
from src.config import config_to_dict, load_config, resolve, set_seeds

# Keys in conf/config.yaml -> the display names used in reports/model_leaderboard.csv
# (which the API reads back in /api/model-info, so these names are a contract).
DISPLAY_NAMES = {
    "linear": "Linear Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def _build_estimator(key: str, params: dict) -> tuple[Any, str]:
    """Return (unfitted estimator, mlflow flavor module name)."""
    if key == "linear":
        return LinearRegression(**params), "sklearn"
    if key == "decision_tree":
        return DecisionTreeRegressor(**params), "sklearn"
    if key == "random_forest":
        return RandomForestRegressor(**params), "sklearn"
    if key == "xgboost":
        import xgboost as xgb

        return xgb.XGBRegressor(**params), "xgboost"
    if key == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(**params), "lightgbm"
    raise ValueError(f"unknown model '{key}' -- add it to _build_estimator")


def _fit(key: str, model: Any, X_train, y_train, X_val, y_val) -> Any:
    """Fit, wiring up early stopping for the boosters that support it."""
    if key == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif key == "lightgbm":
        import lightgbm as lgb

        rounds = model.get_params().get("early_stopping_rounds")
        callbacks = [lgb.early_stopping(stopping_rounds=rounds, verbose=False)] if rounds else []
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="mae", callbacks=callbacks)
    else:
        model.fit(X_train, y_train)
    return model


def _log_model(mlflow, flavor: str, model: Any, X_sample) -> None:
    """Log with the right MLflow flavor, tolerating the 2.x/3.x arg rename."""
    from mlflow.models import infer_signature

    module = getattr(mlflow, flavor)
    log = module.log_model
    kwargs: dict[str, Any] = {"signature": infer_signature(X_sample, model.predict(X_sample))}
    if flavor in {"sklearn", "lightgbm"}:
        # MLflow 3 serializes sklearn-API estimators with skops, an optional extra
        # we don't install; cloudpickle ships with mlflow-skinny and handles them.
        # (LightGBM's sklearn wrapper goes through the same code path.)
        kwargs["serialization_format"] = "cloudpickle"
    arg = {"sklearn": "sk_model", "xgboost": "xgb_model", "lightgbm": "lgb_model"}[flavor]
    try:  # MLflow >= 2.20 / 3.x
        log(**{arg: model}, name="model", **kwargs)
    except TypeError:  # older signature
        log(**{arg: model}, artifact_path="model", **kwargs)


def train(cfg: DictConfig) -> dict:
    subset = cfg.subset
    processed = resolve(cfg, "processed")
    models_dir = resolve(cfg, "models")
    reports = resolve(cfg, "reports")
    artifacts = resolve(cfg, "artifacts")

    manifest = joblib.load(artifacts / f"feature_manifest_{subset}.joblib")
    feature_cols = manifest["feature_cols"]

    train_df = pd.read_parquet(processed / f"train_{subset}_features.parquet")
    val_df = pd.read_parquet(processed / f"val_{subset}_features.parquet")

    target = cfg.target.train_on
    X_train, y_train = train_df[feature_cols], train_df[target]
    # Always scored against the true (uncapped) RUL, whatever the model was fit on.
    X_val, y_val = val_df[feature_cols], val_df[cfg.target.name]

    results: dict[str, dict] = {}
    fitted: dict[str, Any] = {}

    with tracking.run(cfg, stage="train") as parent:
        mlflow = tracking.setup(cfg)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_rows", len(train_df))

        for key in cfg.models.candidates:
            name = DISPLAY_NAMES.get(key, key)
            params = dict(config_to_dict(cfg).get("models", {}).get(key, {}) or {})
            estimator, flavor = _build_estimator(key, params)

            with mlflow.start_run(run_name=name, nested=True):
                t0 = time.perf_counter()
                model = _fit(key, estimator, X_train, y_train, X_val, y_val)
                fit_seconds = time.perf_counter() - t0

                report = ev.regression_report(y_val, model.predict(X_val))
                results[name] = {**report, "fit_seconds": fit_seconds}
                fitted[name] = model

                mlflow.set_tags({"model_key": key, "model_name": name})
                mlflow.log_params({f"model.{k}": v for k, v in params.items()})
                mlflow.log_metrics({f"val_{k}": float(v) for k, v in report.items()})
                mlflow.log_metric("fit_seconds", fit_seconds)
                _log_model(mlflow, flavor, model, X_val.head(5))
                print(f"[train] {name:18s} MAE={report['MAE']:.3f} RMSE={report['RMSE']:.3f} R2={report['R2']:.3f}")

        table = ev.metrics_table(results)
        best_name = (
            table[cfg.models.selection_metric].idxmin()
            if cfg.models.selection_direction == "min"
            else table[cfg.models.selection_metric].idxmax()
        )
        best_model = fitted[best_name]

        # The leaderboard keeps rows for models not retrained in this run, so a
        # single-candidate run doesn't wipe the comparison table.
        path = reports / "model_leaderboard.csv"
        leaderboard = pd.read_csv(path, index_col=0) if path.exists() else pd.DataFrame(columns=table.columns)
        leaderboard = pd.concat([leaderboard, table[~table.index.isin(leaderboard.index)]])
        leaderboard.loc[table.index, table.columns] = table
        leaderboard.to_csv(path)

        meta = {
            "model_name": best_name,
            "feature_cols": feature_cols,
            "target": target,
            "subset": subset,
            "trained_at": dt.datetime.now(dt.UTC).isoformat(),
            "git_sha": tracking.git_sha(),
            "mlflow_run_id": parent.info.run_id,
            "val_metrics": results[best_name],
            "feature_params": manifest.get("feature_params"),
            "config": config_to_dict(cfg),
        }
        joblib.dump(best_model, models_dir / f"best_model_{subset}.joblib")
        joblib.dump(meta, models_dir / f"best_model_{subset}_meta.joblib")

        mlflow.set_tag("best_model", best_name)
        mlflow.log_metrics({f"best_val_{k}": float(v) for k, v in results[best_name].items()})
        mlflow.log_artifact(str(path))

    print(f"[train] best on val {cfg.models.selection_metric}: {best_name} -> models/best_model_{subset}.joblib")
    return {"best_model": best_name, "results": results, "run_id": parent.info.run_id}


def main() -> None:
    cfg = load_config(cli_overrides=True)
    set_seeds(cfg.project.seed)
    train(cfg)


if __name__ == "__main__":
    main()
