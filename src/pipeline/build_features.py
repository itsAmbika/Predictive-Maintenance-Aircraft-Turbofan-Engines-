"""Stage 2 -- interim parquet -> model-ready features + every fitted transform.

Replaces notebooks/03_feature_engineering.ipynb. Leakage rules enforced here:

  * the train/val split is by *engine* (GroupShuffleSplit on unit_number)
  * scaler / KMeans / PCA are fit on training engines only, then applied unchanged
  * the official test file is transformed, never fit on

Everything the serving layer needs to rebuild identical features -- the fitted
objects *and* the feature params that produced them -- is written to
``artifacts/scalers/feature_manifest_<subset>.joblib``.

    uv run python -m src.pipeline.build_features
"""

from __future__ import annotations

import joblib
import pandas as pd
from omegaconf import DictConfig

from src import data_loader as dl
from src import feature_engineering as fe
from src import preprocessing as pp
from src.config import config_to_dict, load_config, resolve, set_seeds
from src.features import (
    FeatureParams,
    FittedTransforms,
    add_temporal_features,
    build_feature_frame,
    feature_columns,
)


def _fit_transforms(train_split: pd.DataFrame, cfg: DictConfig, params: FeatureParams) -> FittedTransforms:
    """Fit scaler -> KMeans -> health PCA on training engines, in that order.

    The KMeans/PCA fits need scaled + temporally-featured rows, so this does one
    throwaway pass to produce them. The *saved* feature tables are then rebuilt
    through :func:`build_feature_frame`, the same function serving calls.
    """
    variance = pp.classify_sensor_variance(
        train_split, dl.SENSOR_COLS, near_constant_std_threshold=cfg.features.near_constant_std_threshold
    )
    feature_sensors = variance[variance["category"] != "constant"].index.tolist()
    constant_sensors = variance[variance["category"] == "constant"].index.tolist()
    print(f"[build_features] dropping {len(constant_sensors)} constant sensors: {constant_sensors}")
    print(f"[build_features] keeping {len(feature_sensors)} sensors as features")

    scaler = pp.fit_scaler(train_split, feature_sensors)
    scaled = pp.apply_scaler(train_split, feature_sensors, scaler)
    temporal = add_temporal_features(scaled, feature_sensors, params)
    temporal = temporal.dropna(subset=[f"{feature_sensors[0]}_lag{params.max_lag}"]).reset_index(drop=True)

    n_clusters = dl.SUBSET_INFO[cfg.subset].conditions
    kmeans = fe.fit_operating_condition_kmeans(
        temporal, dl.OP_SETTING_COLS, n_clusters=n_clusters, random_state=cfg.project.seed
    )

    health_scaler, health_pca = fe.fit_health_indicator_pca(temporal, feature_sensors)
    hi = fe.add_health_indicator(temporal, feature_sensors, health_scaler, health_pca)
    corr = hi[["health_indicator", cfg.target.name]].corr().iloc[0, 1]
    # Orient the axis so higher = more degraded, whichever way PCA happened to point.
    sign_flipped = bool(corr < 0)
    print(f"[build_features] corr(health_indicator, {cfg.target.name}) = {corr:.3f}  sign_flipped={sign_flipped}")

    return FittedTransforms(
        feature_sensors=feature_sensors,
        scaler=scaler,
        kmeans=kmeans,
        health_scaler=health_scaler,
        health_pca=health_pca,
        health_sign_flipped=sign_flipped,
    )


def build(cfg: DictConfig) -> dict:
    subset = cfg.subset
    interim = resolve(cfg, "interim")
    processed = resolve(cfg, "processed")
    artifacts = resolve(cfg, "artifacts")
    params = FeatureParams.from_cfg(cfg.features)

    train_all = pd.read_parquet(interim / f"train_{subset}_with_rul.parquet")
    test_all = pd.read_parquet(interim / f"test_{subset}_with_rul.parquet")

    train_split, val_split = pp.group_train_val_split(
        train_all, val_size=cfg.split.val_size, random_state=cfg.split.random_state
    )
    # test_all comes from a separate file whose unit_numbers restart at 1 -- a
    # different ID space, so it is deliberately not part of this overlap check.
    pp.assert_no_engine_overlap(train_split, val_split)

    fitted = _fit_transforms(train_split, cfg, params)

    frames = {}
    for name, raw in (("train", train_split), ("val", val_split), ("test", test_all)):
        frames[name] = build_feature_frame(raw, fitted, params, na_policy="drop")
        print(f"[build_features] {name}: {len(raw)} rows -> {frames[name].shape}")

    engineered, feature_cols = feature_columns(frames["train"], fitted.feature_sensors)

    for name, df in frames.items():
        if df.isna().sum().sum():
            raise ValueError(f"NaNs remain in the {name} feature table")
        # Test RUL is ground truth we score against, so it is never capped.
        if name != "test":
            df["RUL_capped"] = pp.cap_rul(df[cfg.target.name], cap=cfg.target.cap)
        df.to_parquet(processed / f"{name}_{subset}_features.parquet")

    fitted.save(artifacts, subset)
    manifest = {
        "subset": subset,
        "feature_sensors": fitted.feature_sensors,
        "engineered_cols": engineered,
        "feature_cols": feature_cols,
        "rul_cap": cfg.target.cap,
        # Serving reads these back so its feature columns cannot drift from training's.
        "feature_params": params.to_dict(),
        "config": config_to_dict(cfg),
    }
    joblib.dump(manifest, artifacts / f"feature_manifest_{subset}.joblib")

    summary = {
        "n_features": len(feature_cols),
        "n_feature_sensors": len(fitted.feature_sensors),
        "train_rows": int(len(frames["train"])),
        "val_rows": int(len(frames["val"])),
        "test_rows": int(len(frames["test"])),
    }
    print(f"[build_features] {summary['n_features']} feature columns -> {artifacts}")
    return summary


def main() -> None:
    cfg = load_config(cli_overrides=True)
    set_seeds(cfg.project.seed)
    build(cfg)


if __name__ == "__main__":
    main()
