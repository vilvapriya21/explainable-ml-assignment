"""Model evaluation utilities."""

from typing import Any, Dict, Optional

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.trainers.base import BaseModelTrainer


class ModelEvaluator:
    """Evaluate predictions from a trainer or another prediction source.

    Args:
        fn_weight: Business cost assigned to each false negative.
        fp_weight: Business cost assigned to each false positive.
    """

    def __init__(self, fn_weight: float = 5.0, fp_weight: float = 1.0) -> None:
        self._fn_weight = fn_weight
        self._fp_weight = fp_weight

    def evaluate(
        self,
        y_true: Any,
        y_pred: Any,
        y_proba: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Calculate classification metrics and configurable business cost.

        Args:
            y_true: Actual binary class labels.
            y_pred: Predicted binary class labels.
            y_proba: Optional positive-class probabilities for ROC AUC.

        Returns:
            Accuracy, precision, recall, F1, confusion matrix, business cost,
            and ROC AUC when probabilities are supplied.

        Raises:
            ValueError: If y_true and y_pred have different lengths.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                "y_true and y_pred must have the same length, "
                f"got {len(y_true)} and {len(y_pred)}."
            )

        matrix = confusion_matrix(y_true, y_pred)
        binary_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        false_positives = int(binary_matrix[0, 1])
        false_negatives = int(binary_matrix[1, 0])
        results: Dict[str, Any] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": matrix.tolist(),
            "business_cost": self._fn_weight * false_negatives
            + self._fp_weight * false_positives,
        }
        if y_proba is not None:
            results["roc_auc"] = roc_auc_score(y_true, y_proba)
        return results
