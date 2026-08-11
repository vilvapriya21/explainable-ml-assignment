"""Single-record model explainability helpers."""

from typing import Any, Dict, List, Sequence

import numpy as np
import shap


def get_top_feature_contributions(
    classifier: Any,
    feature_row: Any,
    feature_names: Sequence[str],
    top_n: int = 5,
) -> List[Dict[str, float | str]]:
    """Return the largest SHAP contributions for one transformed feature row.

    Args:
        classifier: Fitted tree-based classifier to explain.
        feature_row: One preprocessed feature row accepted by the classifier.
        feature_names: Feature names aligned with feature_row columns.
        top_n: Number of largest absolute contributions to return.

    Returns:
        Feature and SHAP contribution dictionaries, sorted by absolute value.

    Raises:
        ValueError: If feature_row does not contain exactly one observation.
    """
    values = np.asarray(feature_row)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError("feature_row must contain exactly one transformed record.")

    explainer = shap.TreeExplainer(classifier)
    shap_values = np.asarray(explainer.shap_values(values))
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    contributions = shap_values[0]
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return [
        {
            "feature": str(feature_names[index]),
            "contribution": float(contributions[index]),
        }
        for index in order
    ]
