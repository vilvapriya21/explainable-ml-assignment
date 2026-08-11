"""Tests for containerized selected-model training artifacts."""

import json
from types import SimpleNamespace

import numpy as np

import src.train as train


class _FakePipeline:
    """Provide deterministic predictions without fitting a real model."""

    def fit(self, X: object, y: object) -> "_FakePipeline":
        """Accept training inputs and return the fitted fake pipeline."""
        del X, y
        return self

    def predict(self, X: object) -> np.ndarray:
        """Return deterministic binary predictions."""
        del X
        return np.array([0, 1])

    def predict_proba(self, X: object) -> np.ndarray:
        """Return deterministic positive-class probabilities."""
        del X
        return np.array([[0.9, 0.1], [0.1, 0.9]])


def test_train_preserves_existing_comparison_entries(monkeypatch, tmp_path) -> None:
    """Refresh only XGBoost metrics when a comparison artifact already exists."""
    metrics_path = tmp_path / "metrics.json"
    existing_metrics = {
        "XGBoost": {"accuracy": 0.1, "business_cost": 99.0},
        "LightGBM": {"accuracy": 0.8, "business_cost": 4.0},
        "Random Forest": {"accuracy": 0.7, "business_cost": 5.0},
        "selected_model": "LightGBM",
        "selection_reason": "Existing comparison selection.",
    }
    metrics_path.write_text(json.dumps(existing_metrics, indent=2), encoding="utf-8")
    preserved_lightgbm = json.dumps(existing_metrics["LightGBM"], sort_keys=True)
    preserved_random_forest = json.dumps(
        existing_metrics["Random Forest"], sort_keys=True
    )
    preserved_selection = json.dumps(
        {
            "selected_model": existing_metrics["selected_model"],
            "selection_reason": existing_metrics["selection_reason"],
        },
        sort_keys=True,
    )

    monkeypatch.setattr(train, "METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(train, "MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setattr(train, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(
        train,
        "load_breast_cancer",
        lambda: SimpleNamespace(data=np.zeros((4, 2)), target=np.array([0, 1, 0, 1])),
    )
    monkeypatch.setattr(
        train,
        "train_test_split",
        lambda *args, **kwargs: (
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.array([0, 1]),
            np.array([0, 1]),
        ),
    )
    monkeypatch.setattr(train, "build_selected_pipeline", _FakePipeline)
    monkeypatch.setattr(train.joblib, "dump", lambda *args, **kwargs: None)

    train.main()

    merged_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert merged_metrics["XGBoost"] != existing_metrics["XGBoost"]
    assert json.dumps(merged_metrics["LightGBM"], sort_keys=True) == preserved_lightgbm
    assert (
        json.dumps(merged_metrics["Random Forest"], sort_keys=True)
        == preserved_random_forest
    )
    assert (
        json.dumps(
            {
                "selected_model": merged_metrics["selected_model"],
                "selection_reason": merged_metrics["selection_reason"],
            },
            sort_keys=True,
        )
        == preserved_selection
    )
