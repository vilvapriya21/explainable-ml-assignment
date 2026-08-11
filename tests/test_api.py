"""Integration tests for FastAPI model endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from src.api.main import app
from src.config import MODEL_REVIEW_PATH


def _valid_payload() -> dict:
    """Build a valid request using the first benchmark record.

    Returns:
        Feature values keyed by original sklearn feature names.
    """
    dataset = load_breast_cancer()
    return {
        name: float(value)
        for name, value in zip(dataset.feature_names, dataset.data[0])
    }


def test_health_endpoint_returns_200_and_expected_shape() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert set(response.json()) == {"status", "model_loaded", "model_version"}


def test_predict_with_valid_input_returns_200_and_valid_class() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 200
    assert response.json()["predicted_class"] in {0, 1}
    assert response.json()["class_label"] in {"malignant", "benign"}


def test_predict_with_missing_field_returns_422() -> None:
    payload = _valid_payload()
    payload.pop("mean radius")

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_with_invalid_type_returns_422() -> None:
    payload = _valid_payload()
    payload["mean radius"] = "not-a-number"

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_explain_with_valid_input_returns_top_five_features() -> None:
    with TestClient(app) as client:
        response = client.post("/explain", json=_valid_payload())

    assert response.status_code == 200
    assert len(response.json()["top_features"]) == 5


def test_model_review_returns_200_even_when_file_missing(tmp_path, monkeypatch) -> None:
    missing_review = tmp_path / "missing_review.md"
    monkeypatch.setattr("src.api.main.MODEL_REVIEW_PATH", str(missing_review))

    with TestClient(app) as client:
        response = client.get("/model-review")

    assert response.status_code == 200
    assert response.json() == {"content": "", "generated": False}
