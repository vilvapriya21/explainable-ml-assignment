"""Containerized equivalent of notebook 03's selected-model training steps.

Notebooks are not intended to execute inside a container, so this module
recreates the selected pipeline for Docker and CI artifact generation. When a
notebook comparison artifact already exists, the trainer merges its refreshed
XGBoost metrics instead of replacing the comparison so Docker can rerun
independently without destroying the reference comparison from notebook 03.
"""

import logging
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    ARTIFACTS_DIR,
    FN_COST,
    FP_COST,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
)
from src.evaluation import ModelEvaluator
from src.utils.io import load_json, save_json
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)

# Selected by notebook 03's controlled RandomizedSearchCV run.
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
MODEL_NAME = "XGBoost"
NOTEBOOK_MODEL_NAME = "XGBoost (tuned)"


def build_selected_pipeline() -> Pipeline:
    """Build the selected XGBoost preprocessing and classification pipeline.

    Returns:
        An unfitted pipeline matching notebook 03's selected configuration.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=XGB_N_ESTIMATORS,
                    max_depth=XGB_MAX_DEPTH,
                    learning_rate=XGB_LEARNING_RATE,
                    subsample=XGB_SUBSAMPLE,
                    eval_metric="logloss",
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _merged_metrics(existing_metrics: dict, metrics: dict) -> dict:
    """Update the retrained model while preserving comparison metadata.

    Args:
        existing_metrics: Existing metrics artifact content.
        metrics: Fresh metrics from the retrained XGBoost pipeline.

    Returns:
        The existing artifact with only the corresponding XGBoost metrics
        updated.
    """
    merged_metrics = existing_metrics.copy()
    if MODEL_NAME in merged_metrics:
        merged_metrics[MODEL_NAME] = metrics
        return merged_metrics

    comparison = merged_metrics.get("comparison")
    if isinstance(comparison, list):
        for index, record in enumerate(comparison):
            if isinstance(record, dict) and record.get("model") == NOTEBOOK_MODEL_NAME:
                updated_record = record.copy()
                updated_record.update(metrics)
                comparison[index] = updated_record
                return merged_metrics

    merged_metrics[MODEL_NAME] = metrics
    return merged_metrics


def _metrics_payload(metrics: dict) -> dict:
    """Build the artifact payload without replacing an existing comparison.

    Args:
        metrics: Fresh metrics from the retrained XGBoost pipeline.

    Returns:
        A merged comparison artifact or a clearly marked single-model artifact.
    """
    if Path(METRICS_PATH).is_file():
        return _merged_metrics(load_json(METRICS_PATH), metrics)
    return {
        MODEL_NAME: metrics,
        "selected_model": MODEL_NAME,
        "selection_reason": (
            "Single-model trainer-service run; this artifact is not a full "
            "notebook 03 comparison."
        ),
    }


def main() -> None:
    """Train the selected pipeline and persist model and metric artifacts."""
    configure_logging()
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        stratify=dataset.target,
        random_state=RANDOM_SEED,
    )
    pipeline = build_selected_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = ModelEvaluator(fn_weight=FN_COST, fp_weight=FP_COST).evaluate(
        y_test,
        predictions,
        probabilities,
    )

    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    save_json(_metrics_payload(metrics), METRICS_PATH)
    logger.info("Saved trained pipeline to %s and metrics to %s.", MODEL_PATH, METRICS_PATH)


if __name__ == "__main__":
    main()
