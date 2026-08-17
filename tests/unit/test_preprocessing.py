"""Unit tests for the target construction, scaling, and leakage guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import preprocessing as pp
from tests.conftest import make_raw_frame


def test_add_rul_counts_down_to_zero_per_engine(raw_frame):
    out = pp.add_rul(raw_frame)
    for _, g in out.groupby("unit_number"):
        assert g["RUL"].min() == 0, "the last observed cycle must have RUL 0"
        assert g.sort_values("cycle")["RUL"].is_monotonic_decreasing
        assert g["RUL"].max() == g["cycle"].max() - 1


def test_add_rul_test_offsets_by_the_supplied_tail_rul(raw_frame):
    tail = pd.Series({1: 10, 2: 20, 3: 30}, name="RUL")
    tail.index.name = "unit_number"
    out = pp.add_rul_test(raw_frame, tail)
    for unit, expected in tail.items():
        g = out[out["unit_number"] == unit]
        last = g.loc[g["cycle"].idxmax()]
        assert last["RUL"] == expected
        # one cycle earlier is one cycle more of remaining life
        prev = g[g["cycle"] == last["cycle"] - 1].iloc[0]
        assert prev["RUL"] == expected + 1


def test_cap_rul_clips_only_the_top():
    rul = pd.Series([0, 50, 125, 200])
    assert list(pp.cap_rul(rul, cap=125)) == [0, 50, 125, 125]
    assert list(pp.cap_rul(rul, cap=None)) == [0, 50, 125, 200]


def test_classify_sensor_variance_finds_constant_sensors(raw_frame):
    report = pp.classify_sensor_variance(raw_frame, [f"sensor_{i}" for i in range(1, 22)])
    constants = set(report[report["category"] == "constant"].index)
    assert {"sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"} <= constants


def test_scaler_is_fit_on_train_only_and_applied_unchanged():
    """The whole leakage rule in one assertion: transforming val must use train stats."""
    train = make_raw_frame(n_engines=3, seed=1)
    val = make_raw_frame(n_engines=2, seed=2)
    val["sensor_2"] += 100.0  # a val-only shift the scaler must NOT absorb

    cols = ["sensor_2", "sensor_3"]
    scaler = pp.fit_scaler(train, cols)
    scaled_val = pp.apply_scaler(val, cols, scaler)

    assert not np.isclose(scaled_val["sensor_2"].mean(), 0.0, atol=0.1), (
        "val standardized to mean 0 means the scaler saw val data"
    )
    scaled_train = pp.apply_scaler(train, cols, scaler)
    assert np.isclose(scaled_train["sensor_2"].mean(), 0.0, atol=1e-9)


def test_group_split_keeps_every_cycle_of_an_engine_together():
    df = pp.add_rul(make_raw_frame(n_engines=10))
    train, val = pp.group_train_val_split(df, val_size=0.3, random_state=42)

    assert set(train["unit_number"]) & set(val["unit_number"]) == set()
    assert len(train) + len(val) == len(df)
    for unit in val["unit_number"].unique():
        assert (val["unit_number"] == unit).sum() == (df["unit_number"] == unit).sum()


def test_group_split_is_deterministic_for_a_fixed_seed():
    df = pp.add_rul(make_raw_frame(n_engines=10))
    a, _ = pp.group_train_val_split(df, val_size=0.3, random_state=42)
    b, _ = pp.group_train_val_split(df, val_size=0.3, random_state=42)
    assert sorted(a["unit_number"].unique()) == sorted(b["unit_number"].unique())


def test_assert_no_engine_overlap_raises_on_overlap():
    df = make_raw_frame(n_engines=4)
    pp.assert_no_engine_overlap(df[df.unit_number <= 2], df[df.unit_number > 2])
    with pytest.raises(ValueError, match="overlap"):
        pp.assert_no_engine_overlap(df[df.unit_number <= 3], df[df.unit_number >= 3])
