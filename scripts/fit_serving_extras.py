"""
Fits and persists the two model families that notebooks/08_failure_probability.ipynb
computes but never saves to disk: the per-horizon failure-probability classifiers
and the [10th, 90th] percentile RUL-interval regressors.

Notebook 08, as written, fits `classifiers` (dict of LogisticRegression) and
`gbr_low`/`gbr_high` (GradientBoostingRegressor) in memory and only ever writes out
a static CSV snapshot (`final_maintenance_table_FD001.csv`) for the fixed official
test set. That CSV can't be reused to score a newly-uploaded file. This script does
the identical fit (same features, same hyperparameters, same train split) and saves
the actual model objects, so the API can apply them to any uploaded engine at
request time -- matching how `best_model_FD001.joblib` is already used.

Run once with: uv run python scripts/fit_serving_extras.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from src import health as hl

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SUBSET = "FD001"
THRESHOLDS = (10, 20, 30)


def main() -> None:
    meta = joblib.load(MODELS_DIR / f"best_model_{SUBSET}_meta.joblib")
    feature_cols = meta["feature_cols"]

    train = pd.read_parquet(PROCESSED_DIR / f"train_{SUBSET}_features.parquet")
    for col in feature_cols:
        if col not in train.columns:
            train[col] = 0.0
    X_train = train[feature_cols]

    print("Fitting failure-probability classifiers for thresholds:", THRESHOLDS)
    train_labeled = hl.add_failure_labels(train, thresholds=THRESHOLDS, rul_col="RUL")
    classifiers = {}
    for h in THRESHOLDS:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train, train_labeled[f"fail_within_{h}"])
        classifiers[h] = clf
        print(f"  fail_within_{h}: fit on {len(X_train)} rows")

    print("Fitting [10th, 90th] percentile RUL-interval regressors")
    gbr_low = GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=300, max_depth=3, random_state=42)
    gbr_high = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=300, max_depth=3, random_state=42)
    gbr_low.fit(X_train, train["RUL"])
    gbr_high.fit(X_train, train["RUL"])

    joblib.dump(classifiers, MODELS_DIR / f"failure_classifiers_{SUBSET}.joblib")
    joblib.dump({"low": gbr_low, "high": gbr_high}, MODELS_DIR / f"quantile_models_{SUBSET}.joblib")
    print(f"Saved failure_classifiers_{SUBSET}.joblib and quantile_models_{SUBSET}.joblib to {MODELS_DIR}")


if __name__ == "__main__":
    main()
