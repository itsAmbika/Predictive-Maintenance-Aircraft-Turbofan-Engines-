"""Stage 5 -- fit the two extra model families the API serves.

Notebook 08 computes these in memory and only ever writes a static CSV, which
can't score a newly uploaded file. This stage fits and persists the actual model
objects:

  * per-horizon failure-probability classifiers -- P(RUL <= h) for each h
  * quantile regressors -- the [low, high] RUL prediction interval

    uv run python -m src.pipeline.serving_extras
"""

from __future__ import annotations

import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import health as hl
from src import tracking
from src.config import config_to_dict, load_config, resolve, set_seeds


def fit_extras(cfg: DictConfig) -> dict:
    subset = cfg.subset
    processed = resolve(cfg, "processed")
    models_dir = resolve(cfg, "models")
    extras = cfg.serving_extras

    meta = joblib.load(models_dir / f"best_model_{subset}_meta.joblib")
    feature_cols = meta["feature_cols"]

    train = pd.read_parquet(processed / f"train_{subset}_features.parquet")
    val = pd.read_parquet(processed / f"val_{subset}_features.parquet")
    X_train, X_val = train[feature_cols], val[feature_cols]

    thresholds = list(extras.failure_thresholds)
    labeled_train = hl.add_failure_labels(train, thresholds=tuple(thresholds), rul_col=cfg.target.name)
    labeled_val = hl.add_failure_labels(val, thresholds=tuple(thresholds), rul_col=cfg.target.name)

    clf_params = dict(config_to_dict(cfg)["serving_extras"]["classifier_params"])
    classifiers, aucs = {}, {}
    for h in thresholds:
        clf = LogisticRegression(**clf_params)
        clf.fit(X_train, labeled_train[f"fail_within_{h}"])
        classifiers[h] = clf
        aucs[f"fail_within_{h}_auc"] = float(
            roc_auc_score(labeled_val[f"fail_within_{h}"], clf.predict_proba(X_val)[:, 1])
        )
        print(f"[serving_extras] fail_within_{h}: val ROC-AUC {aucs[f'fail_within_{h}_auc']:.3f}")

    q_params = dict(config_to_dict(cfg)["serving_extras"]["quantile_params"])
    low = GradientBoostingRegressor(loss="quantile", alpha=float(extras.quantile_low), **q_params)
    high = GradientBoostingRegressor(loss="quantile", alpha=float(extras.quantile_high), **q_params)
    low.fit(X_train, train[cfg.target.name])
    high.fit(X_train, train[cfg.target.name])

    # Empirical coverage on validation: what fraction of true RULs land inside the
    # predicted interval. Should sit near (high - low), i.e. 80% by default.
    lo_pred, hi_pred = low.predict(X_val), high.predict(X_val)
    coverage = float(((val[cfg.target.name] >= lo_pred) & (val[cfg.target.name] <= hi_pred)).mean())
    nominal = float(extras.quantile_high - extras.quantile_low)
    print(f"[serving_extras] interval coverage on val: {coverage:.3f} (nominal {nominal:.2f})")

    joblib.dump(classifiers, models_dir / f"failure_classifiers_{subset}.joblib")
    joblib.dump({"low": low, "high": high}, models_dir / f"quantile_models_{subset}.joblib")

    metrics = {**aucs, "interval_coverage": coverage, "interval_nominal": nominal}
    with tracking.run(cfg, stage="serving_extras"):
        mlflow = tracking.setup(cfg)
        mlflow.log_metrics(metrics)
    return metrics


def main() -> None:
    cfg = load_config(cli_overrides=True)
    set_seeds(cfg.project.seed)
    fit_extras(cfg)


if __name__ == "__main__":
    main()
