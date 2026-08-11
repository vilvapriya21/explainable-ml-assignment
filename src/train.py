"""Containerized equivalent of notebook 03's selected-model training steps.

Notebooks are not intended to execute inside a container, so this module
recreates the selected pipeline for Docker and CI artifact generation.
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
from src.utils.io import save_json
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)

# Selected by notebook 03's controlled RandomizedSearchCV run.
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8


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
    save_json(
        {
            "XGBoost": metrics,
            "selected_model": "XGBoost",
            "selection_reason": (
                "Selected notebook 03 XGBoost configuration retrained for the "
                "containerized evaluation workflow."
            ),
        },
        METRICS_PATH,
    )
    logger.info("Saved trained pipeline to %s and metrics to %s.", MODEL_PATH, METRICS_PATH)


if __name__ == "__main__":
    main()
