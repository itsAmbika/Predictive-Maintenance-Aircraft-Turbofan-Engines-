"""The one place features are built.

Training (``src/pipeline/build_features.py``) and serving (``api/inference.py``)
both call :func:`build_feature_frame` with the same :class:`FeatureParams` and the
same :class:`FittedTransforms`, so serving-time features cannot drift from
training-time features. The params travel *inside* the saved feature manifest, so
retraining with different lags automatically changes what the API computes -- no
second copy of the numbers to keep in sync.

Composition order (identical on both sides):

    scale sensors -> lag/rolling/diff/EMA -> operating-condition cluster
    -> PCA health indicator (+ optional sign flip) -> NaN policy
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src import data_loader as dl
from src import feature_engineering as fe
from src import preprocessing as pp


@dataclass(frozen=True)
class FeatureParams:
    """Everything that decides *which* feature columns exist."""

    lags: list[int] = field(default_factory=lambda: [1, 2, 3])
    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    rolling_stats: list[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])
    ema_spans: list[int] = field(default_factory=lambda: [5, 10])
    diff_periods: list[int] = field(default_factory=lambda: [1])

    @classmethod
    def from_cfg(cls, features_cfg: Any) -> FeatureParams:
        return cls(
            lags=list(features_cfg.lags),
            rolling_windows=list(features_cfg.rolling_windows),
            rolling_stats=list(features_cfg.rolling_stats),
            ema_spans=list(features_cfg.ema_spans),
            diff_periods=list(features_cfg.diff_periods),
        )

    @classmethod
    def from_manifest(cls, manifest: dict) -> FeatureParams:
        """Read the params the model was actually trained with.

        Manifests written before feature params were persisted fall back to the
        historical defaults above -- which are exactly the values notebook 03 used.
        """
        stored = manifest.get("feature_params")
        return cls(**stored) if stored else cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def max_lag(self) -> int:
        return max([*self.lags, *self.diff_periods], default=0)


@dataclass
class FittedTransforms:
    """The fitted objects a feature frame needs, fit on training engines only."""

    feature_sensors: list[str]
    scaler: Any
    kmeans: Any
    health_scaler: Any
    health_pca: Any
    health_sign_flipped: bool

    @classmethod
    def load(cls, artifacts_dir: str | Path, subset: str) -> FittedTransforms:
        d = Path(artifacts_dir)
        manifest = joblib.load(d / f"feature_manifest_{subset}.joblib")
        health = joblib.load(d / f"health_indicator_pca_{subset}.joblib")
        return cls(
            feature_sensors=list(manifest["feature_sensors"]),
            scaler=joblib.load(d / f"sensor_scaler_{subset}.joblib"),
            kmeans=joblib.load(d / f"operating_condition_kmeans_{subset}.joblib"),
            health_scaler=health["scaler"],
            health_pca=health["pca"],
            health_sign_flipped=bool(health["sign_flipped"]),
        )

    def save(self, artifacts_dir: str | Path, subset: str) -> None:
        d = Path(artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, d / f"sensor_scaler_{subset}.joblib")
        joblib.dump(self.kmeans, d / f"operating_condition_kmeans_{subset}.joblib")
        joblib.dump(
            {
                "scaler": self.health_scaler,
                "pca": self.health_pca,
                "sign_flipped": self.health_sign_flipped,
            },
            d / f"health_indicator_pca_{subset}.joblib",
        )


def add_temporal_features(df: pd.DataFrame, sensors: list[str], params: FeatureParams) -> pd.DataFrame:
    """Lag / rolling / diff / EMA columns, grouped by engine (never across engines)."""
    out = fe.add_lag_features(df, sensors, lags=params.lags)
    out = fe.add_rolling_features(out, sensors, windows=params.rolling_windows, stats=tuple(params.rolling_stats))
    out = fe.add_diff_features(out, sensors, periods=params.diff_periods)
    out = fe.add_ema_features(out, sensors, spans=params.ema_spans)
    return out


def build_feature_frame(
    df: pd.DataFrame,
    fitted: FittedTransforms,
    params: FeatureParams,
    na_policy: str = "fill",
) -> pd.DataFrame:
    """Raw C-MAPSS rows -> model-ready feature frame, using already-fit transforms.

    ``na_policy``:
      * ``"drop"`` (training) -- drop the leading rows of each engine where lag /
        diff features are undefined.
      * ``"fill"`` (serving) -- keep every row and fill those NaNs with 0.0. Only
        the last cycle of each engine is scored, so this matters only for engines
        shorter than the largest lag, which would otherwise vanish entirely.
      * ``"keep"`` -- leave NaNs in place (used by tests comparing both paths).
    """
    sensors = fitted.feature_sensors

    # Normalize row order ONCE, here. Lag/diff/EMA are all defined relative to the
    # previous row within an engine and none of them sort internally (only
    # add_rolling_features does), so an uploaded file in any other order -- shuffled,
    # or the common "newest first" export -- would silently get wrong temporal
    # features. For already-sorted input (every raw C-MAPSS file) this is a no-op.
    ordered = df.sort_values(["unit_number", "cycle"], kind="stable").reset_index(drop=True)

    out = pp.apply_scaler(ordered, sensors, fitted.scaler)
    out = add_temporal_features(out, sensors, params)
    out = fe.add_operating_condition_cluster(out, dl.OP_SETTING_COLS, fitted.kmeans)
    out = fe.add_health_indicator(out, sensors, fitted.health_scaler, fitted.health_pca)
    if fitted.health_sign_flipped:
        out["health_indicator"] = -out["health_indicator"]

    if na_policy == "drop":
        out = out.dropna(subset=[f"{sensors[0]}_lag{params.max_lag}"]).reset_index(drop=True)
    elif na_policy == "fill":
        out = out.fillna(0.0)
    elif na_policy != "keep":
        raise ValueError(f"unknown na_policy: {na_policy!r}")
    return out


def feature_columns(df: pd.DataFrame, feature_sensors: list[str]) -> tuple[list[str], list[str]]:
    """Split a built frame's columns into (engineered_cols, full model feature list).

    Column order is taken from the frame itself, so it always matches the order
    :func:`build_feature_frame` produced -- the model is fit on positional columns
    and would silently mispredict if serving reordered them.
    """
    engineered = [c for c in df.columns if any(c.startswith(f"{s}_") for s in feature_sensors)]
    feature_cols = [
        *feature_sensors,
        *engineered,
        *dl.OP_SETTING_COLS,
        "operating_condition_cluster",
        "health_indicator",
    ]
    return engineered, feature_cols
