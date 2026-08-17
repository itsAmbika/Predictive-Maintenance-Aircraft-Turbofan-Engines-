"""
Temporal feature engineering for C-MAPSS (PHASE 4).

Every function groups by ``id_col`` before computing lags / rolling stats / diffs
so that no information ever crosses an engine boundary. Functions that fit a
model (KMeans for operating-condition clusters, PCA for the health indicator)
accept an optional pre-fit object so you fit once on training engines and reuse
it, unchanged, on validation/test engines - the same leakage-safe pattern as
``preprocessing.fit_scaler`` / ``apply_scaler``.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def add_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int], id_col: str = "unit_number") -> pd.DataFrame:
    """Add sensor(t - lag) columns, e.g. sensor_11_lag1, sensor_11_lag5."""
    out = df.copy()
    grouped = out.groupby(id_col)
    for col in cols:
        for lag in lags:
            out[f"{col}_lag{lag}"] = grouped[col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    cols: list[str],
    windows: list[int],
    stats: tuple[str, ...] = ("mean", "std", "min", "max"),
    id_col: str = "unit_number",
    time_col: str = "cycle",
) -> pd.DataFrame:
    """Add rolling window statistics, e.g. sensor_11_mean_10, sensor_11_std_10.

    Rows are sorted by (id_col, time_col) first so the rolling window walks
    forward in time within each engine.
    """
    out = df.sort_values([id_col, time_col]).copy()
    grouped = out.groupby(id_col)
    for col in cols:
        for window in windows:
            roll = grouped[col].rolling(window=window, min_periods=1)
            for stat in stats:
                out[f"{col}_{stat}_{window}"] = roll.agg(stat).reset_index(level=0, drop=True)
    return out


def add_diff_features(
    df: pd.DataFrame, cols: list[str], periods: list[int] = (1,), id_col: str = "unit_number"
) -> pd.DataFrame:
    """Add one (or multi-) step differences: sensor_t - sensor_{t-period}."""
    out = df.copy()
    grouped = out.groupby(id_col)
    for col in cols:
        for period in periods:
            out[f"{col}_diff{period}"] = grouped[col].diff(period)
    return out


def add_ema_features(df: pd.DataFrame, cols: list[str], spans: list[int], id_col: str = "unit_number") -> pd.DataFrame:
    """Add exponential moving averages: sensor_11_ema10."""
    out = df.copy()
    grouped = out.groupby(id_col)
    for col in cols:
        for span in spans:
            out[f"{col}_ema{span}"] = grouped[col].transform(lambda s, sp=span: s.ewm(span=sp, adjust=False).mean())
    return out


# ---------------------------------------------------------------------------
# Operating-condition clustering (PHASE 4.6) - mainly useful for FD002 / FD004
# ---------------------------------------------------------------------------


def fit_operating_condition_kmeans(
    df: pd.DataFrame, op_cols: list[str], n_clusters: int = 6, random_state: int = 42
) -> KMeans:
    """Fit KMeans on operating settings using TRAINING rows only."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(df[op_cols])
    return km


def add_operating_condition_cluster(df: pd.DataFrame, op_cols: list[str], kmeans: KMeans) -> pd.DataFrame:
    """Assign each row to an already-fit operating-condition cluster."""
    out = df.copy()
    out["operating_condition_cluster"] = kmeans.predict(out[op_cols])
    return out


# ---------------------------------------------------------------------------
# Data-driven health indicator (PHASE 4.7)
#
# NOTE: this is NOT the paper's internal health index (which was derived from
# normalized operational margins on flow/efficiency and was never exposed to
# challenge participants). This is our own 1-D degradation proxy, learned
# purely from the available sensors via PCA.
# ---------------------------------------------------------------------------


def fit_health_indicator_pca(df: pd.DataFrame, sensor_cols: list[str]) -> tuple[StandardScaler, PCA]:
    """Fit a scaler + 1-component PCA on TRAINING rows for the selected sensors."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[sensor_cols])
    pca = PCA(n_components=1, random_state=42)
    pca.fit(scaled)
    return scaler, pca


def add_health_indicator(df: pd.DataFrame, sensor_cols: list[str], scaler: StandardScaler, pca: PCA) -> pd.DataFrame:
    """Project sensors onto the fitted 1-D PCA axis -> ``health_indicator`` column.

    Sign is not guaranteed to point in any particular direction; orient it (e.g.
    by correlating with RUL on the training set) before interpreting it as
    "higher = healthier".
    """
    out = df.copy()
    scaled = scaler.transform(out[sensor_cols])
    out["health_indicator"] = pca.transform(scaled)[:, 0]
    return out
