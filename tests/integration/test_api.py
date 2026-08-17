"""Integration tests for the FastAPI serving layer.

The API must import and answer /api/health even with no model on disk -- that is
the point of loading artifacts in the lifespan handler rather than at import
time. Everything that needs real predictions is marked slow.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src import data_loader as dl


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` runs the lifespan handler
        yield c


def _upload(client, df, filename="test_FD001.txt"):
    buf = io.BytesIO(df.to_csv(sep=" ", header=False, index=False).encode())
    return client.post("/api/predict/upload", files={"file": (filename, buf, "text/plain")})


def test_health_always_answers(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert "model_loaded" in body


def test_openapi_schema_is_served(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/api/predict/upload" in r.json()["paths"]


def test_rejects_a_file_that_is_not_cmapss(client):
    buf = io.BytesIO(b"this is not a sensor file\n")
    r = client.post("/api/predict/upload", files={"file": ("notes.txt", buf, "text/plain")})
    # 400 when the model is loaded, 503 when it isn't -- never a 500
    assert r.status_code in {400, 503}


def test_rejects_an_empty_file(client):
    r = client.post("/api/predict/upload", files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")})
    assert r.status_code in {400, 503}


@pytest.mark.slow
def test_model_info_reports_provenance(client):
    if not client.get("/api/health").json()["model_loaded"]:
        pytest.skip("no trained model on disk")
    body = client.get("/api/model-info").json()
    assert body["n_features"] > 0
    assert set(body["metrics"]) == {"MAE", "RMSE", "R2", "NASA_score"}
    assert body["model_name"]


@pytest.mark.slow
def test_upload_scores_every_engine_once(client, raw_cmapss_file):
    if not client.get("/api/health").json()["model_loaded"]:
        pytest.skip("no trained model on disk")

    r = _upload(client, raw_cmapss_file)
    assert r.status_code == 200, r.text
    body = r.json()

    expected_units = sorted(raw_cmapss_file["unit_number"].unique())
    assert body["n_engines"] == len(expected_units)
    assert sorted(e["unit_number"] for e in body["engines"]) == expected_units

    ruls = [e["predicted_rul"] for e in body["engines"]]
    assert ruls == sorted(ruls), "engines must come back sorted by predicted RUL"

    for e in body["engines"]:
        assert e["predicted_rul"] > -50
        assert 0 <= e["health_score"] <= 100
        assert e["risk"] in {"LOW", "MEDIUM", "HIGH"}
        assert e["risk_action"]
        assert len(e["health_trend"]) == int((raw_cmapss_file["unit_number"] == e["unit_number"]).sum())
        if e["rul_low"] is not None:
            assert e["rul_low"] <= e["rul_high"]
        if e["fail_within_20_proba"] is not None:
            assert 0.0 <= e["fail_within_20_proba"] <= 1.0
        if e["top_factors"]:
            assert all(f["direction"] in {"raises_rul", "lowers_rul"} for f in e["top_factors"])


@pytest.mark.slow
def test_predictions_are_deterministic(client, raw_cmapss_file):
    if not client.get("/api/health").json()["model_loaded"]:
        pytest.skip("no trained model on disk")
    first = _upload(client, raw_cmapss_file).json()
    second = _upload(client, raw_cmapss_file).json()
    assert [e["predicted_rul"] for e in first["engines"]] == [e["predicted_rul"] for e in second["engines"]]


@pytest.mark.slow
def test_a_single_short_engine_still_gets_a_prediction(client, raw_cmapss_file):
    """Engines shorter than the largest lag must not vanish (the `fill` NaN policy)."""
    if not client.get("/api/health").json()["model_loaded"]:
        pytest.skip("no trained model on disk")

    short = raw_cmapss_file[raw_cmapss_file["unit_number"] == raw_cmapss_file["unit_number"].min()].head(2)
    r = _upload(client, short[dl.ALL_COLS])
    assert r.status_code == 200, r.text
    assert r.json()["n_engines"] == 1
