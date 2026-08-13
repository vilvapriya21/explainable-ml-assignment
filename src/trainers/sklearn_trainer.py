"""Scikit-learn implementation of the model trainer interface."""

from pathlib import Path
from typing import Any, Dict

import joblib

from src.python_advanced import measure_execution_time
from src.trainers.base import BaseModelTrainer


class SklearnModelTrainer(BaseModelTrainer):
    """Train and persist a scikit-learn-compatible estimator.

    Args:
        estimator: Estimator implementing scikit-learn's fit and predict API.
        name: A descriptive name for the model.
        hyperparameters: Model-specific hyperparameter values.
        random_state: Random seed used when training the model.
    """

    def __init__(
        self,
        estimator: Any,
        name: str,
        hyperparameters: Dict[str, Any],
        random_state: int,
    ) -> None:
        super().__init__(name, hyperparameters, random_state)
        self._estimator = estimator
        self._is_fitted = False

    def train(self, X: Any, y: Any) -> None:
        """Fit the wrapped estimator and mark it as fitted.

        Args:
            X: Feature data accepted by the estimator.
            y: Target values aligned with X.
        """
        timed_fit = measure_execution_time(self._estimator.fit)
        timed_fit(X, y)
        self._is_fitted = True

    def predict(self, X: Any) -> Any:
        """Generate predictions using the fitted estimator.

        Args:
            X: Feature data accepted by the estimator.

        Returns:
            Estimator predictions.

        Raises:
            RuntimeError: If the estimator has not been trained or loaded.
        """
        self._ensure_fitted()
        return self._estimator.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Generate class probabilities using the fitted estimator.

        Args:
            X: Feature data accepted by the estimator.

        Returns:
            Estimator class probabilities.

        Raises:
            RuntimeError: If the estimator has not been trained or loaded.
            AttributeError: If the estimator does not support predict_proba.
        """
        self._ensure_fitted()
        if not hasattr(self._estimator, "predict_proba"):
            raise AttributeError(
                f"{type(self._estimator).__name__} does not support predict_proba."
            )
        return self._estimator.predict_proba(X)

    def save(self, path: str) -> None:
        """Persist the estimator and its configuration with joblib.

        Args:
            path: Destination joblib file path.

        Raises:
            RuntimeError: If the estimator has not been trained or loaded.
        """
        self._ensure_fitted()
        joblib.dump(
            {"estimator": self._estimator, "config": self.config, "is_fitted": True},
            path,
        )

    def load(self, path: str) -> "SklearnModelTrainer":
        """Restore estimator state and configuration from a joblib file.

        Args:
            path: Source joblib file path.

        Returns:
            This restored trainer instance.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Saved model file does not exist: {path!r}.")

        saved_state = joblib.load(source)
        config = saved_state["config"]
        self._estimator = saved_state["estimator"]
        self._name = config["name"]
        self._hyperparameters = dict(config["hyperparameters"])
        self._random_state = config["random_state"]
        self._is_fitted = saved_state["is_fitted"]
        return self

    def _ensure_fitted(self) -> None:
        """Raise an error when inference or persistence is requested too early.

        Raises:
            RuntimeError: If the estimator has not been trained or loaded.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "The model is not fitted. Call train() or load() before using it."
            )
