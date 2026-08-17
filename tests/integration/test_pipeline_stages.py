"""End-to-end pipeline test on a miniature dataset.

Runs prepare_data -> build_features -> train against a synthetic 6-engine subset
written into a tmp_path, with the config pointed at those directories. Fast
enough for CI (seconds), and it exercises the real stage code -- so a broken
stage fails here rather than in a 20-minute nightly run.
"""

from __future__ import annotations

import joblib
import pandas as pd
import pytest

from src import data_loader as dl
from src.config import load_config
from src.pipeline import build_features, prepare_data, train
from tests.conftest import make_raw_frame

SUBSET = "FD001"  # SUBSET_INFO drives the KMeans cluster count, so use a real name


@pytest.fixture
def mini_project(tmp_path, monkeypatch):
    """A tiny but structurally real C-MAPSS project rooted at tmp_path."""
    raw = tmp_path / "raw"
    raw.mkdir()

    train_df = make_raw_frame(n_engines=6, n_cycles=60, seed=7)
    # Test engines are truncated before failure, as in the real data.
    test_df = make_raw_frame(n_engines=4, n_cycles=40, seed=8)
    test_df = pd.concat([g.head(len(g) - 12) for _, g in test_df.groupby("unit_number")], ignore_index=True)[
        dl.ALL_COLS
    ]

    train_df.to_csv(raw / f"train_{SUBSET}.txt", sep=" ", header=False, index=False)
    test_df.to_csv(raw / f"test_{SUBSET}.txt", sep=" ", header=False, index=False)
    pd.DataFrame({"RUL": [10, 20, 30, 40]}).to_csv(raw / f"RUL_{SUBSET}.txt", sep=" ", header=False, index=False)

    # Every stage resolves paths through src.config.resolve(), which reads
    # PROJECT_ROOT from its own module globals -- so pointing that at tmp_path
    # redirects the whole pipeline without touching the real project directories.
    import src.config as config_module

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)

    cfg = load_config(
        overrides=[
            f"subset={SUBSET}",
            "paths.raw=raw",
            "paths.interim=interim",
            "paths.processed=processed",
            "paths.artifacts=artifacts",
            "paths.models=models",
            "paths.reports=reports",
            "features.lags=[1,2]",
            "features.rolling_windows=[3]",
            "features.rolling_stats=[mean]",
            "features.ema_spans=[3]",
            "models.candidates=[linear,random_forest]",
            "models.random_forest.n_estimators=10",
            "models.random_forest.max_depth=4",
            f"mlflow.tracking_uri=file:{(tmp_path / 'mlruns').as_posix()}",
        ]
    )
    return cfg, tmp_path


def test_full_pipeline_produces_a_servable_model(mini_project):
    cfg, root = mini_project

    stats = prepare_data.prepare(cfg)
    assert stats["train_engines"] == 6
    assert (root / "interim" / f"train_{SUBSET}_with_rul.parquet").exists()

    summary = build_features.build(cfg)
    assert summary["n_features"] > 0
    assert summary["train_rows"] > 0

    manifest = joblib.load(root / "artifacts" / f"feature_manifest_{SUBSET}.joblib")
    assert manifest["feature_params"]["lags"] == [1, 2]
    assert manifest["feature_cols"][-2:] == ["operating_condition_cluster", "health_indicator"]
    # constant sensors must have been dropped
    assert "sensor_1" not in manifest["feature_sensors"]

    result = train.train(cfg)
    assert result["best_model"] in {"Linear Regression", "Random Forest"}

    model = joblib.load(root / "models" / f"best_model_{SUBSET}.joblib")
    meta = joblib.load(root / "models" / f"best_model_{SUBSET}_meta.joblib")
    assert meta["model_name"] == result["best_model"]
    assert meta["mlflow_run_id"]
    assert meta["feature_cols"] == manifest["feature_cols"]

    leaderboard = pd.read_csv(root / "reports" / "model_leaderboard.csv", index_col=0)
    assert list(leaderboard.columns) == ["MAE", "RMSE", "R2", "NASA_score", "n"]
    assert result["best_model"] in leaderboard.index

    test_features = pd.read_parquet(root / "processed" / f"test_{SUBSET}_features.parquet")
    preds = model.predict(test_features[meta["feature_cols"]])
    assert len(preds) == len(test_features)


def test_no_engine_appears_in_both_train_and_val(mini_project):
    cfg, root = mini_project
    prepare_data.prepare(cfg)
    build_features.build(cfg)

    train_df = pd.read_parquet(root / "processed" / f"train_{SUBSET}_features.parquet")
    val_df = pd.read_parquet(root / "processed" / f"val_{SUBSET}_features.parquet")
    assert set(train_df["unit_number"]) & set(val_df["unit_number"]) == set()


def test_test_split_is_never_capped(mini_project):
    """The official test RUL is ground truth -- capping it would flatter the metrics."""
    cfg, root = mini_project
    prepare_data.prepare(cfg)
    build_features.build(cfg)

    test_df = pd.read_parquet(root / "processed" / f"test_{SUBSET}_features.parquet")
    assert "RUL_capped" not in test_df.columns
    train_df = pd.read_parquet(root / "processed" / f"train_{SUBSET}_features.parquet")
    assert train_df["RUL_capped"].max() <= cfg.target.cap


def test_raw_columns_survive_the_pipeline(mini_project):
    cfg, root = mini_project
    prepare_data.prepare(cfg)
    interim = pd.read_parquet(root / "interim" / f"train_{SUBSET}_with_rul.parquet")
    assert set(dl.ALL_COLS) <= set(interim.columns)
    assert interim["RUL"].min() == 0
