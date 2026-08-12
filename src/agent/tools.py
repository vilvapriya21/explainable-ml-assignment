"""CrewAI tools for reading and validating model-review artifacts."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from langchain_core.tools import tool

from src.config import FN_COST, FP_COST
from src.recommendation_chain import (
    BusinessConstraint,
    RecommendationRequest,
    get_model_recommendation,
)

logger = logging.getLogger(__name__)

_REQUIRED_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "roc_auc",
    "business_cost",
}


def _read_json_file(path: str, label: str) -> Dict[str, Any]:
    """Read one JSON artifact with contextual error handling.

    Args:
        path: Artifact file path.
        label: Human-readable artifact name for errors.

    Returns:
        Parsed JSON object.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the file contains malformed or non-object JSON.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path!r}.")
    try:
        contents = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} file contains malformed JSON: {path!r}.") from exc
    if not isinstance(contents, dict):
        raise ValueError(f"{label} file must contain a JSON object: {path!r}.")
    return contents


def _model_entries(metrics_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract named metric dictionaries from supported artifact shapes.

    Args:
        metrics_data: Parsed metrics artifact.

    Returns:
        Model names mapped to their metric dictionaries.

    Raises:
        ValueError: If no valid model metric entries are present.
    """
    entries: Dict[str, Dict[str, Any]] = {
        name: values
        for name, values in metrics_data.items()
        if isinstance(values, dict) and _REQUIRED_METRICS.issubset(values)
    }
    if entries:
        return entries
    comparison = metrics_data.get("comparison")
    if isinstance(comparison, list):
        for item in comparison:
            if isinstance(item, dict) and isinstance(item.get("model"), str):
                entries[item["model"]] = item
    if not entries:
        raise ValueError(
            "metrics.json must include at least one named model entry with "
            "accuracy, precision, recall, f1, confusion_matrix, roc_auc, and "
            "business_cost."
        )
    return entries


def _validate_metrics_artifact(metrics_data: Dict[str, Any]) -> None:
    """Validate required metrics artifact keys and metric fields.

    Args:
        metrics_data: Parsed metrics artifact.

    Raises:
        ValueError: If required top-level keys or model fields are missing.
    """
    if "selected_model" not in metrics_data:
        raise ValueError("metrics.json is missing required key 'selected_model'.")
    entries = _model_entries(metrics_data)
    for name, values in entries.items():
        missing = _REQUIRED_METRICS - set(values)
        if missing:
            raise ValueError(
                f"metrics.json model {name!r} is missing required metrics: "
                f"{sorted(missing)}."
            )


@tool("Read Metrics File")
def read_metrics_file(path: str = "artifacts/metrics.json") -> dict:
    """Read and validate the metrics artifact.

    Args:
        path: Path to metrics.json.

    Returns:
        Parsed and validated metrics data.

    Raises:
        FileNotFoundError: If the metrics artifact is missing.
        ValueError: If JSON is malformed or lacks required metrics.
    """
    metrics_data = _read_json_file(path, "Metrics")
    _validate_metrics_artifact(metrics_data)
    logger.info("Read validated metrics artifact from %s.", path)
    return metrics_data


@tool("Audit Model Comparison")
def audit_model_comparison(path: str = "artifacts/metrics.json") -> dict:
    """Read a model comparison and independently verify every business cost.

    Args:
        path: Path to the metrics artifact.

    Returns:
        A compact per-model audit with reported and recomputed business costs.
    """
    metrics_data = _read_json_file(path, "Metrics")
    _validate_metrics_artifact(metrics_data)
    entries = _model_entries(metrics_data)
    audit: Dict[str, Dict[str, Any]] = {}
    for name, values in entries.items():
        cost_check = compute_weighted_business_cost.func(
            metrics=values,
            fn_cost=FN_COST,
            fp_cost=FP_COST,
        )
        audit[name] = {
            "accuracy": values["accuracy"],
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1"],
            "roc_auc": values["roc_auc"],
            "confusion_matrix": values["confusion_matrix"],
            "business_cost": values["business_cost"],
            **cost_check,
        }
    logger.info("Audited %d model comparison entries from %s.", len(audit), path)
    return {
        "selected_model": metrics_data["selected_model"],
        "selection_reason": metrics_data.get("selection_reason", ""),
        "models": audit,
    }


@tool("Read Explainability Summaries")
def read_explainability_summaries(
    shap_path: str = "artifacts/shap_summary.json",
    lime_path: str = "artifacts/lime_summary.json",
) -> dict:
    """Read and validate SHAP and LIME summary artifacts.

    Args:
        shap_path: Path to the SHAP summary JSON.
        lime_path: Path to the LIME summary JSON.

    Returns:
        A dictionary containing validated SHAP and LIME summaries.

    Raises:
        FileNotFoundError: If either required summary file is missing.
        ValueError: If either summary is malformed or lacks required keys.
    """
    shap_data = _read_json_file(shap_path, "SHAP summary")
    lime_data = _read_json_file(lime_path, "LIME summary")
    shap_required = {
        "global_importance",
        "top_5_features",
        "local_correct_example",
        "local_false_negative_example",
    }
    lime_required = {"correct_example", "incorrect_example"}
    shap_missing = shap_required - set(shap_data)
    lime_missing = lime_required - set(lime_data)
    if shap_missing:
        raise ValueError(
            f"SHAP summary is missing required keys: {sorted(shap_missing)}."
        )
    if lime_missing:
        raise ValueError(
            f"LIME summary is missing required keys: {sorted(lime_missing)}."
        )
    logger.info("Read validated SHAP and LIME summaries.")
    return {
        "top_5_features": shap_data["top_5_features"],
        "false_negative": {
            "record_index": shap_data["local_false_negative_example"]["record_index"],
            "true_label": shap_data["local_false_negative_example"]["true_label"],
            "predicted_label": shap_data["local_false_negative_example"]["predicted_label"],
            "top_shap_contributions": shap_data["local_false_negative_example"][
                "feature_contributions"
            ][:5],
            "top_lime_contributions": lime_data["incorrect_example"][
                "feature_contributions"
            ][:5],
        },
    }


@tool("Compute Weighted Business Cost")
def compute_weighted_business_cost(
    metrics: dict,
    fn_cost: float,
    fp_cost: float,
) -> dict:
    """Independently recompute cost from a binary confusion matrix.

    Args:
        metrics: Model metrics containing confusion_matrix and reported cost.
        fn_cost: Cost assigned to one false negative.
        fp_cost: Cost assigned to one false positive.

    Returns:
        Recomputed and reported costs plus their equality check.

    Raises:
        ValueError: If confusion_matrix is missing or malformed.
    """
    matrix = metrics.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or not all(isinstance(row, list) and len(row) == 2 for row in matrix)
    ):
        raise ValueError(
            "metrics must contain confusion_matrix as [[TN, FP], [FN, TP]]."
        )
    false_positives = float(matrix[0][1])
    false_negatives = float(matrix[1][0])
    recomputed_cost = fn_cost * false_negatives + fp_cost * false_positives
    reported_cost = float(metrics.get("business_cost", recomputed_cost))
    return {
        "recomputed_cost": float(recomputed_cost),
        "reported_cost": reported_cost,
        "matches_reported": bool(abs(recomputed_cost - reported_cost) < 1e-9),
    }


@tool("Get Model Recommendation")
def get_model_recommendation_tool(
    model_metrics: dict,
    fn_cost: float,
    fp_cost: float,
) -> dict:
    """Return a local structured recommendation as a plain dictionary.

    Args:
        model_metrics: Candidate model metrics keyed by model name.
        fn_cost: Cost assigned to one false negative.
        fp_cost: Cost assigned to one false positive.

    Returns:
        A validated model recommendation serialized to a dictionary.
    """
    request = RecommendationRequest(
        model_metrics=model_metrics,
        business_constraint=BusinessConstraint(
            false_negative_cost=fn_cost,
            false_positive_cost=fp_cost,
        ),
    )
    return get_model_recommendation(request).model_dump()
