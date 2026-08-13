"""
Application-level health score, failure-probability targets, and maintenance
risk categories (PHASE 16-17, 26).

Everything here is derived transparently from a model's predicted RUL. None of
it is the paper's internal health index (that index came from normalized
operational margins on flow/efficiency and was never released to challenge
participants) and none of it is a NASA-prescribed maintenance threshold - these
are our own application decision rules, clearly labeled as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def health_score(rul_pred, rul_healthy: float = 125) -> np.ndarray:
    """Health = 100 * min(1, RUL_pred / rul_healthy). 100 = fully healthy, 0 = failed."""
    rul_pred = np.asarray(rul_pred, dtype=np.float64)
    return 100.0 * np.minimum(1.0, np.maximum(0.0, rul_pred) / rul_healthy)


def add_failure_labels(df: pd.DataFrame, thresholds: tuple[int, ...] = (10, 20, 30), rul_col: str = "RUL") -> pd.DataFrame:
    """Add binary classification targets: fail_within_{h} = 1 if RUL <= h."""
    out = df.copy()
    for h in thresholds:
        out[f"fail_within_{h}"] = (out[rul_col] <= h).astype(int)
    return out


RISK_RULES = [
    (60, np.inf, "LOW", "Continue monitoring"),
    (30, 60, "MEDIUM", "Increase inspection frequency"),
    (-np.inf, 30, "HIGH", "Schedule maintenance inspection"),
]


def risk_category(rul_pred) -> pd.Series:
    """Map predicted RUL to a LOW/MEDIUM/HIGH application risk bucket.

    These cut points (30, 60 cycles) are an application-level modeling choice
    made for this project, not a NASA-issued maintenance standard.
    """
    rul_pred = pd.Series(rul_pred)
    labels = pd.Series(index=rul_pred.index, dtype=object)
    for lo, hi, label, _action in RISK_RULES:
        mask = (rul_pred > lo) & (rul_pred <= hi)
        labels[mask] = label
    return labels


def risk_action(risk_label: str) -> str:
    for _lo, _hi, label, action in RISK_RULES:
        if label == risk_label:
            return action
    return "Unknown"
