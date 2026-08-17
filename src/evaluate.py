"""
Evaluation metrics for RUL regression (PHASE 9-10), including the PHM08 /
NASA asymmetric scoring function used in the original competition.

NASA score (Saxena et al., 2008):

    d = RUL_pred - RUL_true
    s = sum( exp(-d/13) - 1 )   for d < 0   (early prediction, under-estimated RUL)
    s = sum( exp( d/10) - 1 )   for d >= 0  (late prediction, over-estimated RUL)

Late predictions (telling maintainers an engine has more life left than it
truly does) are penalized far more heavily than early ones - this is the whole
point of the score, and why it should always be reported alongside MAE/RMSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true
    early = d < 0
    score = np.where(early, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_report(y_true, y_pred) -> dict:
    """MAE, RMSE, R2, and NASA score in one dict - the row you'd log per model."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "NASA_score": nasa_score(y_true, y_pred),
        "n": len(y_true),
    }


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """results: {"Linear Regression": {"MAE":..., "RMSE":..., ...}, "XGBoost": {...}, ...}"""
    return pd.DataFrame(results).T[["MAE", "RMSE", "R2", "NASA_score", "n"]]


# ---------------------------------------------------------------------------
# Error analysis by remaining-life bucket (PHASE 10)
# ---------------------------------------------------------------------------

DEFAULT_BINS = [
    (-np.inf, 20, "RUL <= 20"),
    (20, 50, "20 < RUL <= 50"),
    (50, 100, "50 < RUL <= 100"),
    (100, np.inf, "RUL > 100"),
]


def evaluate_by_rul_bin(y_true, y_pred, bins: list[tuple[float, float, str]] = DEFAULT_BINS) -> pd.DataFrame:
    """Report MAE/RMSE/NASA score separately per RUL range.

    The low-RUL bucket ("close to failure") is operationally the most important:
    it answers "does the model still perform when it matters most?".
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rows = []
    for lo, hi, label in bins:
        mask = (y_true > lo) & (y_true <= hi)
        if mask.sum() == 0:
            rows.append({"bucket": label, "n": 0, "MAE": np.nan, "RMSE": np.nan, "NASA_score": np.nan})
            continue
        rows.append(
            {
                "bucket": label,
                "n": int(mask.sum()),
                "MAE": mean_absolute_error(y_true[mask], y_pred[mask]),
                "RMSE": rmse(y_true[mask], y_pred[mask]),
                "NASA_score": nasa_score(y_true[mask], y_pred[mask]),
            }
        )
    return pd.DataFrame(rows).set_index("bucket")
