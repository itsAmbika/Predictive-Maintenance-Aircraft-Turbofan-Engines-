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


def test_truncated_validation_mimics_the_official_protocol(mini_project):
    """Candidates are ranked on simulated truncated engines, not all val rows.

    Validation engines run to failure, so their last row always has RUL 0 and
    ranking on it would be degenerate; ranking on every row measures a population
    the model is never asked about. This samples one pre-failure cut per engine.
    """
    cfg, root = mini_project
    prepare_data.prepare(cfg)
    build_features.build(cfg)

    val = pd.read_parquet(root / "processed" / f"val_{SUBSET}_features.parquet")
    n_rep = int(cfg.evaluation.selection.n_truncations)
    sel = train.truncated_validation(val, cfg)

    assert len(sel) == n_rep * val["unit_number"].nunique()
    assert set(sel["unit_number"]) == set(val["unit_number"]), "every engine must appear"
    # never a degenerate end-of-life row, and never the very start of an engine
    assert (sel["RUL"] > 0).all(), "a truncation point must have life remaining"
    for unit, g in sel.groupby("unit_number"):
        earliest = val[val["unit_number"] == unit]["cycle"].min()
        assert g["cycle"].min() > earliest

    # deterministic for a fixed seed
    assert train.truncated_validation(val, cfg)["cycle"].tolist() == sel["cycle"].tolist()


def test_unknown_selection_protocol_is_rejected(mini_project):
    cfg, _ = mini_project
    prepare_data.prepare(cfg)
    build_features.build(cfg)
    cfg.evaluation.selection.protocol = "vibes"
    with pytest.raises(ValueError, match="selection.protocol"):
        train.train(cfg)


def test_group_kfold_scores_every_candidate_over_all_engines(mini_project):
    """CV folds are grouped by engine and every candidate is scored on every fold."""
    cfg, root = mini_project
    prepare_data.prepare(cfg)
    build_features.build(cfg)

    tr = pd.read_parquet(root / "processed" / f"train_{SUBSET}_features.parquet")
    va = pd.read_parquet(root / "processed" / f"val_{SUBSET}_features.parquet")
    pool = pd.concat([tr, va], ignore_index=True)
    manifest = joblib.load(root / "artifacts" / f"feature_manifest_{SUBSET}.joblib")

    cfg.evaluation.selection.n_splits = 2
    scores = train.group_kfold_scores(pool, cfg, manifest["feature_cols"])

    assert set(scores) == {"Linear Regression", "Random Forest"}
    for name, r in scores.items():
        assert r["MAE"] > 0, name
        # the fold-to-fold spread is what says whether a ranking is trustworthy
        assert "MAE_std" in r and r["MAE_std"] >= 0
        assert r["n"] > 0


def test_group_kfold_never_scores_an_engine_it_trained_on(mini_project, monkeypatch):
    """The leakage guarantee of grouped CV: held-out engines are truly held out."""
    cfg, root = mini_project
    prepare_data.prepare(cfg)
    build_features.build(cfg)

    tr = pd.read_parquet(root / "processed" / f"train_{SUBSET}_features.parquet")
    va = pd.read_parquet(root / "processed" / f"val_{SUBSET}_features.parquet")
    pool = pd.concat([tr, va], ignore_index=True)
    manifest = joblib.load(root / "artifacts" / f"feature_manifest_{SUBSET}.joblib")
    cfg.evaluation.selection.n_splits = 2

    seen = []
    real_fit = train._fit

    def spy(key, model, X_train, y_train, X_val, y_val):
        seen.append((set(pool.loc[X_train.index, "unit_number"]), set(pool.loc[X_val.index, "unit_number"])))
        return real_fit(key, model, X_train, y_train, X_val, y_val)

    monkeypatch.setattr(train, "_fit", spy)
    train.group_kfold_scores(pool, cfg, manifest["feature_cols"])

    assert seen, "no folds ran"
    for fit_engines, stop_engines in seen:
        # inner early-stopping engines must also be disjoint from the fitted ones
        assert fit_engines & stop_engines == set()
