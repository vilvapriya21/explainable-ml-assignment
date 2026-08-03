"""Tests for the scikit-learn trainer pipeline."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.trainers.sklearn_trainer import SklearnModelTrainer


def test_sklearn_trainer_train_predict_and_save_load_round_trip(tmp_path) -> None:
    X, y = make_classification(
        n_samples=40,
        n_features=4,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=7,
    )
    trainer = SklearnModelTrainer(
        LogisticRegression(random_state=7),
        name="logistic_regression",
        hyperparameters={"max_iter": 100},
        random_state=7,
    )

    trainer.train(X, y)
    predictions = trainer.predict(X)
    probabilities = trainer.predict_proba(X)
    model_path = tmp_path / "model.joblib"
    trainer.save(str(model_path))

    restored_trainer = SklearnModelTrainer(
        LogisticRegression(), "placeholder", {}, random_state=0
    ).load(str(model_path))

    assert predictions.shape == (40,)
    assert probabilities.shape == (40, 2)
    assert np.array_equal(restored_trainer.predict(X), predictions)


def test_sklearn_trainer_rejects_prediction_before_training() -> None:
    trainer = SklearnModelTrainer(
        LogisticRegression(), "logistic_regression", {}, random_state=0
    )

    with pytest.raises(RuntimeError, match="not fitted"):
        trainer.predict([[0.0, 1.0]])
