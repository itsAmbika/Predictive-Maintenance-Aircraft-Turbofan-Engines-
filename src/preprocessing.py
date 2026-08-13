"""
Preprocessing utilities: RUL target construction, constant-sensor detection,
leakage-safe scaling, and engine-grouped train/validation splitting.

Design rule enforced throughout this module: any statistic used to transform data
(a scaler's mean/std, a "which sensors are constant" decision, cluster centroids,
...) must be *fit* on the training engines only, then *applied* to validation/test
engines. See PHASE 3 / PHASE 5 of the project plan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# RUL target
# ---------------------------------------------------------------------------


def add_rul(df: pd.DataFrame, id_col: str = "unit_number", time_col: str = "cycle") -> pd.DataFrame:
    """Append a ``RUL`` column for TRAINING data (full run-to-failure trajectories).

    RUL_i,t = (last cycle observed for engine i) - t
    """
    out = df.copy()
    max_cycle = out.groupby(id_col)[time_col].transform("max")
    out["RUL"] = max_cycle - out[time_col]
    return out


def add_rul_test(
    test_df: pd.DataFrame,
    rul_at_end: pd.Series,
    id_col: str = "unit_number",
    time_col: str = "cycle",
) -> pd.DataFrame:
    """Append a ``RUL`` column for TEST data.

    Test trajectories are truncated before failure. ``rul_at_end`` (indexed by
    unit_number) is the number of cycles remaining *after* the last observed
    cycle of each engine, as supplied in RUL_FD00x.txt. For any earlier cycle t
    within the same engine: RUL_t = rul_at_end + (last_cycle - t).
    """
    out = test_df.copy()
    max_cycle = out.groupby(id_col)[time_col].transform("max")
    tail_rul = out[id_col].map(rul_at_end)
    out["RUL"] = tail_rul + (max_cycle - out[time_col])
    return out


def cap_rul(rul: pd.Series, cap: int | None = 125) -> pd.Series:
    """Clip RUL at ``cap`` cycles (PHASE 8 modeling assumption, not a physical constant).

    Pass ``cap=None`` to disable capping (raw RUL).
    """
    if cap is None:
        return rul
    return rul.clip(upper=cap)


# ---------------------------------------------------------------------------
# Sensor variance triage (PHASE 2.2) - classification only, no dropping decision
# ---------------------------------------------------------------------------


def classify_sensor_variance(
    df: pd.DataFrame, sensor_cols: list[str], near_constant_std_threshold: float = 1e-3
) -> pd.DataFrame:
    """Return a small report classifying each sensor as constant / near_constant / variable.

    A sensor is "constant" if it has a single unique value across the whole
    (training) dataset. "near_constant" uses a normalized std-dev threshold
    (std / (mean of abs values), guarding against div-by-zero) so that sensors
    with tiny but real variance are not confused with pure constants.
    """
    rows = []
    for col in sensor_cols:
        series = df[col]
        std = series.std()
        mean_abs = series.abs().mean()
        norm_std = std / mean_abs if mean_abs > 0 else std
        n_unique = series.nunique()
        if n_unique <= 1:
            category = "constant"
        elif norm_std < near_constant_std_threshold:
            category = "near_constant"
        else:
            category = "variable"
        rows.append(
            {
                "sensor": col,
                "n_unique": n_unique,
                "std": std,
                "normalized_std": norm_std,
                "min": series.min(),
                "max": series.max(),
                "category": category,
            }
        )
    return pd.DataFrame(rows).set_index("sensor")


# ---------------------------------------------------------------------------
# Leakage-safe scaling
# ---------------------------------------------------------------------------


def fit_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    """Fit a StandardScaler using TRAINING engines only."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler) -> pd.DataFrame:
    """Transform ``feature_cols`` in-place (returns a copy) using an already-fit scaler."""
    out = df.copy()
    out[feature_cols] = scaler.transform(out[feature_cols])
    return out


# ---------------------------------------------------------------------------
# Engine-grouped splitting (PHASE 5)
# ---------------------------------------------------------------------------


def group_train_val_split(
    df: pd.DataFrame,
    id_col: str = "unit_number",
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split TRAINING engines into train/validation, keeping every cycle of an
    engine on the same side of the split (no engine appears in both).

    The official C-MAPSS test set (with its own RUL_FD00x.txt) remains the
    untouched final-evaluation held-out set; this function only splits the
    training file into train/validation for model selection & tuning.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    groups = df[id_col].values
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def assert_no_engine_overlap(*dfs: pd.DataFrame, id_col: str = "unit_number") -> None:
    """Sanity check: raise if any engine id appears in more than one of the given frames."""
    id_sets = [set(d[id_col].unique()) for d in dfs]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            overlap = id_sets[i] & id_sets[j]
            if overlap:
                raise ValueError(f"Engine id overlap between split {i} and {j}: {sorted(overlap)}")
