"""Unit tests for the application-level decision rules and the config layer."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from src import health as hl
from src.config import Config, load_config, set_seeds

# --- health / risk rules ---------------------------------------------------


def test_health_score_is_bounded_and_saturates_at_healthy():
    scores = hl.health_score([-10, 0, 62.5, 125, 500], rul_healthy=125)
    assert scores.min() >= 0 and scores.max() <= 100
    assert scores[1] == 0
    assert scores[2] == pytest.approx(50.0)
    assert scores[3] == scores[4] == 100.0


def test_health_score_scales_with_the_configured_horizon():
    assert hl.health_score([50], rul_healthy=50)[0] == 100.0
    assert hl.health_score([50], rul_healthy=100)[0] == pytest.approx(50.0)


def test_risk_categories_use_the_default_cut_points():
    risks = list(hl.risk_category([10, 30, 31, 60, 61, 200]))
    assert risks == ["HIGH", "HIGH", "MEDIUM", "MEDIUM", "LOW", "LOW"]


def test_risk_categories_follow_configured_thresholds():
    risks = list(hl.risk_category([15, 25, 45], high_below=20, medium_below=40))
    assert risks == ["HIGH", "MEDIUM", "LOW"]


def test_every_risk_label_has_an_action():
    for label in ("LOW", "MEDIUM", "HIGH"):
        assert hl.risk_action(label) != "Unknown"
    assert hl.risk_action("NONSENSE") == "Unknown"


def test_failure_labels_are_binary_and_nested():
    import pandas as pd

    df = pd.DataFrame({"RUL": [5, 15, 25, 100]})
    out = hl.add_failure_labels(df, thresholds=(10, 20, 30))
    assert list(out["fail_within_10"]) == [1, 0, 0, 0]
    assert list(out["fail_within_30"]) == [1, 1, 1, 0]
    # a shorter horizon implies every longer one
    assert (out["fail_within_10"] <= out["fail_within_30"]).all()


# --- config ----------------------------------------------------------------


def test_default_config_matches_the_schema():
    cfg = load_config()
    assert cfg.subset.startswith("FD")
    assert cfg.target.name == "RUL"
    assert cfg.models.selection_metric in {"MAE", "RMSE", "R2", "NASA_score"}
    assert cfg.models.selection_direction in {"min", "max"}


def test_overrides_apply_with_dotlist_syntax():
    cfg = load_config(overrides=["subset=FD003", "features.lags=[1,5]", "models.xgboost.max_depth=9"])
    assert cfg.subset == "FD003"
    assert list(cfg.features.lags) == [1, 5]
    assert cfg.models.xgboost.max_depth == 9


def test_unknown_key_is_rejected_by_the_schema(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("featurs:\n  lags: [1]\n")  # typo'd section
    with pytest.raises(OmegaConfBaseException):
        load_config(config_path=bad)


def test_config_defaults_match_the_shipped_yaml():
    """conf/config.yaml must stay loadable against the dataclass schema."""
    merged = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load("conf/config.yaml"))
    assert merged.serving.risk_high_below < merged.serving.risk_medium_below
    assert merged.evaluation.gate.min_r2 <= 1.0


def test_set_seeds_makes_numpy_deterministic():
    set_seeds(123)
    a = np.random.rand(5)
    set_seeds(123)
    assert np.allclose(a, np.random.rand(5))
