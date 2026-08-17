"""Unit tests for the metrics -- above all the asymmetric NASA/PHM08 score."""

from __future__ import annotations

import numpy as np
import pytest

from src import evaluate as ev


def test_nasa_score_is_zero_for_perfect_predictions():
    y = np.array([10.0, 50.0, 100.0])
    assert ev.nasa_score(y, y) == pytest.approx(0.0)


def test_nasa_score_punishes_late_predictions_harder_than_early_ones():
    """The whole point of the score: saying an engine has more life left than it
    does (late) must cost more than the same error in the safe direction."""
    y_true = np.array([50.0])
    late = ev.nasa_score(y_true, y_true + 10)
    early = ev.nasa_score(y_true, y_true - 10)
    assert late > early
    assert late == pytest.approx(np.exp(10 / 10) - 1)
    assert early == pytest.approx(np.exp(10 / 13) - 1)


def test_nasa_score_grows_superlinearly_with_error():
    y = np.array([50.0])
    small = ev.nasa_score(y, y + 5)
    big = ev.nasa_score(y, y + 10)
    assert big > 2 * small


def test_regression_report_has_the_leaderboard_schema():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    report = ev.regression_report(y_true, y_true + 1)
    assert set(report) == {"MAE", "RMSE", "R2", "NASA_score", "n"}
    assert report["MAE"] == pytest.approx(1.0)
    assert report["RMSE"] == pytest.approx(1.0)
    assert report["n"] == 4


def test_metrics_table_column_order_matches_the_csv_contract():
    """reports/model_leaderboard.csv is read back by /api/model-info."""
    table = ev.metrics_table({"A": ev.regression_report([1.0, 2.0], [1.0, 2.0])})
    assert list(table.columns) == ["MAE", "RMSE", "R2", "NASA_score", "n"]


def test_evaluate_by_rul_bin_splits_by_true_rul():
    y_true = np.array([5.0, 30.0, 80.0, 150.0])
    result = ev.evaluate_by_rul_bin(y_true, y_true)
    assert list(result["n"]) == [1, 1, 1, 1]
    assert result["MAE"].sum() == pytest.approx(0.0)


def test_evaluate_by_rul_bin_handles_empty_buckets():
    y_true = np.array([5.0, 6.0])
    result = ev.evaluate_by_rul_bin(y_true, y_true)
    assert result.loc["RUL > 100", "n"] == 0
    assert np.isnan(result.loc["RUL > 100", "MAE"])
