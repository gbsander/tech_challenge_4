"""Testes da API. Requer artefatos em artifacts/ — pula automaticamente se ausentes."""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.config import settings


def _artifacts_present() -> bool:
    return (
        Path(settings.model_path).exists()
        and Path(settings.scaler_path).exists()
        and Path(settings.metadata_path).exists()
    )


pytestmark = pytest.mark.skipif(
    not _artifacts_present(),
    reason="Artefatos não treinados — rode `python -m src.train_pipeline` antes.",
)


@pytest.fixture(scope="module")
def client():
    from api.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_info(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"]
    assert body["window"] >= 1
    assert "metrics_test" in body


def test_predict_happy_path(client):
    r0 = client.get("/")
    window = r0.json()["window"]
    rng = np.random.default_rng(0)
    closes = (30 + rng.normal(scale=0.5, size=window)).tolist()
    r = client.post("/predict", json={"closes": closes})
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], float)


def test_predict_validation_too_few(client):
    r0 = client.get("/")
    window = r0.json()["window"]
    closes = [30.0] * (window - 1)
    r = client.post("/predict", json={"closes": closes})
    assert r.status_code == 422


def test_predict_validation_negative(client):
    closes = [30.0] * 60 + [-1.0]
    r = client.post("/predict", json={"closes": closes})
    assert r.status_code == 422


def test_metrics_endpoint(client):
    client.get("/health")  # gera métricas
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "http_requests_total" in body
    assert "http_request_duration_seconds" in body
