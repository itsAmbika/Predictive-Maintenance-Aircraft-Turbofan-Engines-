"""Unit tests for feature engineering -- especially that nothing crosses an engine boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import feature_engineering as fe
from src import preprocessing as pp
from src.features import FeatureParams, FittedTransforms, build_feature_frame, feature_columns

SENSORS = ["sensor_2", "sensor_3", "sensor_4"]


def test_lag_features_do_not_leak_across_engines(raw_frame):
    out = fe.add_lag_features(raw_frame, SENSORS, lags=[1, 2])
    for _, g in out.sort_values(["unit_number", "cycle"]).groupby("unit_number"):
        # the first rows of every engine must be undefined, not borrowed from the previous engine
        assert g["sensor_2_lag1"].iloc[:1].isna().all()
        assert g["sensor_2_lag2"].iloc[:2].isna().all()
        assert np.allclose(g["sensor_2_lag1"].iloc[1:], g["sensor_2"].iloc[:-1])


def test_rolling_features_are_causal(raw_frame):
    """A rolling mean must never include a future cycle."""
    out = fe.add_rolling_features(raw_frame, ["sensor_2"], windows=[3], stats=("mean",))
    g = out[out["unit_number"] == 1].sort_values("cycle")
    expected = g["sensor_2"].iloc[:3].mean()
    assert np.isclose(g["sensor_2_mean_3"].iloc[2], expected)
    assert np.isclose(g["sensor_2_mean_3"].iloc[0], g["sensor_2"].iloc[0])


def test_diff_and_ema_respect_engine_groups(raw_frame):
    out = fe.add_diff_features(raw_frame, ["sensor_2"], periods=[1])
    out = fe.add_ema_features(out, ["sensor_2"], spans=[5])
    firsts = out.sort_values(["unit_number", "cycle"]).groupby("unit_number").head(1)
    assert firsts["sensor_2_diff1"].isna().all()
    # EMA seeds from the engine's own first value
    assert np.allclose(firsts["sensor_2_ema5"], firsts["sensor_2"])


def test_feature_params_round_trip_through_a_manifest():
    params = FeatureParams(lags=[1, 4], rolling_windows=[7], rolling_stats=["mean"], ema_spans=[3], diff_periods=[2])
    restored = FeatureParams.from_manifest({"feature_params": params.to_dict()})
    assert restored == params
    assert restored.max_lag == 4


def test_feature_params_from_legacy_manifest_uses_notebook_defaults():
    """Manifests written before params were persisted must still serve correctly."""
    restored = FeatureParams.from_manifest({"feature_sensors": SENSORS})
    assert restored.lags == [1, 2, 3]
    assert restored.rolling_windows == [5, 10, 20]
    assert restored.ema_spans == [5, 10]


def _fitted(df: pd.DataFrame) -> FittedTransforms:
    scaler = pp.fit_scaler(df, SENSORS)
    scaled = pp.apply_scaler(df, SENSORS, scaler)
    kmeans = fe.fit_operating_condition_kmeans(scaled, dl.OP_SETTING_COLS, n_clusters=1, random_state=0)
    hscaler, hpca = fe.fit_health_indicator_pca(scaled, SENSORS)
    return FittedTransforms(SENSORS, scaler, kmeans, hscaler, hpca, health_sign_flipped=False)


def test_build_feature_frame_column_count_follows_the_params(raw_frame):
    fitted = _fitted(raw_frame)
    small = FeatureParams(lags=[1], rolling_windows=[3], rolling_stats=["mean"], ema_spans=[2], diff_periods=[1])
    big = FeatureParams(
        lags=[1, 2], rolling_windows=[3, 5], rolling_stats=["mean", "std"], ema_spans=[2], diff_periods=[1]
    )

    _, small_cols = feature_columns(build_feature_frame(raw_frame, fitted, small), SENSORS)
    _, big_cols = feature_columns(build_feature_frame(raw_frame, fitted, big), SENSORS)
    assert len(big_cols) > len(small_cols)
    # 3 sensors x (1 lag + 1 rolling stat x 1 window + 1 diff + 1 ema) + base + op cols + cluster + HI
    assert len(small_cols) == 3 * (1 + 1 + 1 + 1) + 3 + 3 + 2


def test_na_policies_agree_on_the_rows_they_share(raw_frame):
    """`fill` (serving) and `drop` (training) must produce identical values for
    every row that survives the drop -- only the leading rows differ."""
    fitted = _fitted(raw_frame)
    params = FeatureParams(lags=[1, 2], rolling_windows=[3], rolling_stats=["mean"], ema_spans=[2], diff_periods=[1])

    dropped = build_feature_frame(raw_frame, fitted, params, na_policy="drop")
    filled = build_feature_frame(raw_frame, fitted, params, na_policy="fill")

    _, cols = feature_columns(dropped, SENSORS)
    key = ["unit_number", "cycle"]
    merged = dropped[key + cols].merge(filled[key + cols], on=key, suffixes=("_train", "_serve"))
    assert len(merged) == len(dropped)
    for col in cols:
        assert np.allclose(merged[f"{col}_train"], merged[f"{col}_serve"]), f"{col} differs between na policies"


def test_build_feature_frame_rejects_unknown_na_policy(raw_frame):
    with pytest.raises(ValueError, match="na_policy"):
        build_feature_frame(raw_frame, _fitted(raw_frame), FeatureParams(), na_policy="whatever")


def test_health_indicator_tracks_degradation(raw_frame):
    """The PCA indicator should move monotonically-ish with engine age."""
    fitted = _fitted(raw_frame)
    out = build_feature_frame(raw_frame, fitted, FeatureParams(), na_policy="fill")
    g = out[out["unit_number"] == 1]
    corr = np.corrcoef(g["cycle"], g["health_indicator"])[0, 1]
    assert abs(corr) > 0.8


def test_feature_columns_keeps_frame_order(raw_frame):
    fitted = _fitted(raw_frame)
    built = build_feature_frame(raw_frame, fitted, FeatureParams(), na_policy="fill")
    engineered, cols = feature_columns(built, SENSORS)
    assert cols[: len(SENSORS)] == SENSORS
    assert cols[-2:] == ["operating_condition_cluster", "health_indicator"]
    assert engineered == [c for c in built.columns if c in engineered], "engineered order must match frame order"
    # sensor_2's features must not be attributed to sensor_20-style neighbours
    assert all(c.startswith(("sensor_2_", "sensor_3_", "sensor_4_")) for c in engineered)
