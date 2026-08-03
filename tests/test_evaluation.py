"""Tests for model evaluation and registry behavior."""

import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation import ModelEvaluator
from src.trainers.base import ModelRegistry
from src.trainers.sklearn_trainer import SklearnModelTrainer


def test_model_evaluator_returns_expected_metrics_and_business_cost() -> None:
    evaluator = ModelEvaluator(fn_weight=5.0, fp_weight=1.0)

    result = evaluator.evaluate(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 0, 1],
        y_proba=[0.1, 0.2, 0.8, 0.9],
    )

    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["business_cost"] == 6.0
    assert result["roc_auc"] == 1.0


def test_model_evaluator_rejects_mismatched_prediction_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        ModelEvaluator().evaluate([0, 1], [0])


def test_model_registry_registers_and_retrieves_trainers() -> None:
    registry = ModelRegistry()
    trainer = SklearnModelTrainer(
        LogisticRegression(), "logistic_regression", {}, random_state=0
    )

    registry.register("baseline", trainer)

    assert registry.get("baseline") is trainer
    assert registry.list_models() == ["baseline"]
    with pytest.raises(KeyError, match="No model is registered"):
        registry.get("missing")
