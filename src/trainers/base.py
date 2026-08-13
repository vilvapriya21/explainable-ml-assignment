"""Abstract interfaces and registry support for model trainers."""

import abc
from pathlib import Path
from typing import Any, Dict, List


class BaseModelTrainer(abc.ABC):
    """Define the common interface for model training implementations.

    This abstract class cannot be instantiated directly; subclasses must
    implement all lifecycle methods. It defines the stable contract shared by
    training backends without claiming to know how a particular model is fit,
    queried, or persisted. Configuration is held in private attributes; the
    read-only config property returns copied values so callers cannot mutate
    the configuration controlling a trainer.

    Args:
        name: A descriptive name for the model.
        hyperparameters: Model-specific hyperparameter values.
        random_state: Random seed used when training the model.
    """

    def __init__(
        self,
        name: str,
        hyperparameters: Dict[str, Any],
        random_state: int,
    ) -> None:
        self._name = name
        self._hyperparameters = dict(hyperparameters)
        self._random_state = random_state

    @property
    def config(self) -> Dict[str, Any]:
        """Return a copy of this trainer's configuration.

        Returns:
            The model name, copied hyperparameters, and random state.
        """
        return {
            "name": self._name,
            "hyperparameters": dict(self._hyperparameters),
            "random_state": self._random_state,
        }

    @abc.abstractmethod
    def train(self, X: Any, y: Any) -> None:
        """Fit the model using feature data and targets.

        Args:
            X: Feature data accepted by the underlying model.
            y: Target values aligned with X.
        """

    @abc.abstractmethod
    def predict(self, X: Any) -> Any:
        """Generate predictions for feature data.

        Args:
            X: Feature data accepted by the underlying model.

        Returns:
            Predictions produced by the fitted model.
        """

    @abc.abstractmethod
    def predict_proba(self, X: Any) -> Any:
        """Generate class probabilities for feature data.

        Args:
            X: Feature data accepted by the underlying model.

        Returns:
            Class probabilities produced by the fitted model.
        """

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Persist the trainer state.

        Args:
            path: Destination file path.
        """

    @abc.abstractmethod
    def load(self, path: str) -> "BaseModelTrainer":
        """Restore the trainer state from persistent storage.

        Args:
            path: Source file path.

        Returns:
            The restored trainer instance.
        """


class ModelRegistry:
    """Manage named trainer instances through composition.

    The registry holds trainers without extending their training interface: it
    coordinates model objects but is not itself a model and cannot meaningfully
    train or predict. Evaluation remains separate because it is a cross-cutting
    concern that can compare predictions from any trainer or external source.
    """

    def __init__(self) -> None:
        self._models: Dict[str, BaseModelTrainer] = {}

    def register(self, name: str, trainer: BaseModelTrainer) -> None:
        """Register a trainer under a unique name.

        Args:
            name: Name used to retrieve the trainer.
            trainer: Trainer instance to store.

        Raises:
            TypeError: If trainer does not implement BaseModelTrainer.
        """
        if not isinstance(trainer, BaseModelTrainer):
            raise TypeError(
                "trainer must be an instance of BaseModelTrainer, "
                f"got {type(trainer).__name__}."
            )
        self._models[name] = trainer

    def get(self, name: str) -> BaseModelTrainer:
        """Return a registered trainer.

        Args:
            name: Registered trainer name.

        Returns:
            The matching trainer.

        Raises:
            KeyError: If no trainer is registered under name.
        """
        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(f"No model is registered with the name {name!r}.") from exc

    def list_models(self) -> List[str]:
        """Return registered trainer names.

        Returns:
            Names of all registered trainers in registration order.
        """
        return list(self._models)

    def save_all(self, directory: str) -> None:
        """Save every registered trainer to a directory.

        Args:
            directory: Directory in which trainer files are saved.
        """
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        for name, trainer in self._models.items():
            trainer.save(str(destination / f"{name}.joblib"))

    def load(self, name: str, directory: str) -> BaseModelTrainer:
        """Load a registered trainer from its saved file.

        Args:
            name: Registered trainer name.
            directory: Directory containing the saved trainer files.

        Returns:
            The restored trainer.
        """
        trainer = self.get(name)
        self._models[name] = trainer.load(str(Path(directory) / f"{name}.joblib"))
        return self._models[name]
