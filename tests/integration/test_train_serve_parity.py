"""The test that matters most: training-time and serving-time features agree.

Training builds features in ``src/pipeline/build_features.py``; serving builds
them in ``api/inference.py``. Both go through ``src.features.build_feature_frame``
with the params stored in the manifest -- these tests prove the two paths really
do produce identical numbers for the same raw rows, and that the model's expected
input matches what serving hands it.

All marked slow: they need artifacts from `python -m src.pipeline.run_all`.
"""

from __future__ import annotations

import numpy as np
import pytest

from api import inference
from src.features import FeatureParams, FittedTransforms, build_feature_frame

pytestmark = pytest.mark.slow


def test_serving_features_match_training_features(artifacts, raw_cmapss_file, cfg, project_root):
    """Same raw rows through both code paths -> identical feature values."""
    fitted = FittedTransforms.load(project_root / str(cfg.paths.artifacts), cfg.subset)
    params = FeatureParams.from_manifest(artifacts["manifest"])

    training_side = build_feature_frame(raw_cmapss_file, fitted, params, na_policy="drop")
    serving_side = inference.build_features(raw_cmapss_file, artifacts)

    feature_cols = artifacts["meta"]["feature_cols"]
    key = ["unit_number", "cycle"]
    merged = training_side[key + feature_cols].merge(
        serving_side[key + feature_cols], on=key, suffixes=("_train", "_serve")
    )
    assert len(merged) == len(training_side) > 0

    for col in feature_cols:
        assert np.allclose(merged[f"{col}_train"], merged[f"{col}_serve"], equal_nan=True), (
            f"training/serving skew in feature '{col}'"
        )


def test_manifest_feature_params_are_persisted(artifacts):
    """A manifest without params silently falls back to defaults -- catch that here."""
    assert artifacts["manifest"].get("feature_params"), (
        "feature_manifest is missing feature_params; rebuild with `python -m src.pipeline.build_features`"
    )


def test_model_input_width_matches_the_manifest(artifacts):
    model = artifacts["model"]
    expected = len(artifacts["meta"]["feature_cols"])
    n_in = getattr(model, "n_features_in_", None)
    if n_in is None:
        pytest.skip("estimator does not expose n_features_in_")
    assert n_in == expected


def test_serving_produces_every_manifest_column(artifacts, raw_cmapss_file):
    built = inference.build_features(raw_cmapss_file, artifacts)
    missing = set(artifacts["meta"]["feature_cols"]) - set(built.columns)
    assert not missing, f"serving is missing {len(missing)} training columns, e.g. {sorted(missing)[:5]}"


def test_predict_fleet_scores_the_last_cycle_of_each_engine(artifacts, raw_cmapss_file):
    last_rows, trends = inference.predict_fleet(raw_cmapss_file, artifacts)

    assert len(last_rows) == raw_cmapss_file["unit_number"].nunique()
    for row in last_rows.itertuples():
        expected_last = raw_cmapss_file[raw_cmapss_file["unit_number"] == row.unit_number]["cycle"].max()
        assert row.cycle == expected_last
        assert len(trends[int(row.unit_number)]) == int((raw_cmapss_file["unit_number"] == row.unit_number).sum())


def test_feature_params_changing_would_change_the_columns(artifacts, raw_cmapss_file, project_root, cfg):
    """Guards the fix this refactor was for: if serving hardcoded its windows, a
    manifest with different params would produce the same columns anyway."""
    fitted = FittedTransforms.load(project_root / str(cfg.paths.artifacts), cfg.subset)
    stored = FeatureParams.from_manifest(artifacts["manifest"])
    other = FeatureParams(
        lags=[1],
        rolling_windows=[4],
        rolling_stats=["mean"],
        ema_spans=stored.ema_spans,
        diff_periods=stored.diff_periods,
    )
    a = build_feature_frame(raw_cmapss_file, fitted, stored, na_policy="fill")
    b = build_feature_frame(raw_cmapss_file, fitted, other, na_policy="fill")
    assert set(a.columns) != set(b.columns)


def test_predictions_do_not_depend_on_row_order(artifacts, raw_cmapss_file):
    """An uploaded file may arrive in any order -- shuffled, or "newest first",
    a common export convention. Lag/diff/EMA are defined relative to the previous
    row within an engine, so the feature builder must impose the order itself
    rather than trusting the file. Before this was fixed, shuffling the rows moved
    predictions by up to 2 cycles."""
    canonical = raw_cmapss_file.sort_values(["unit_number", "cycle"]).reset_index(drop=True)
    base, _ = inference.predict_fleet(canonical, artifacts)
    base = base.set_index("unit_number")["RUL_pred"].sort_index()

    variants = {
        "shuffled": canonical.sample(frac=1, random_state=0).reset_index(drop=True),
        "cycles_descending": canonical.sort_values(["unit_number", "cycle"], ascending=[True, False]).reset_index(
            drop=True
        ),
        "interleaved_by_cycle": canonical.sort_values(["cycle", "unit_number"]).reset_index(drop=True),
    }
    for name, df in variants.items():
        got, _ = inference.predict_fleet(df, artifacts)
        got = got.set_index("unit_number")["RUL_pred"].sort_index()
        assert np.allclose(got, base), f"row order '{name}' changed the predictions"
