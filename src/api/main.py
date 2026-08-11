"""FastAPI application for trained-model prediction and explanation."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Tuple

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sklearn.datasets import load_breast_cancer

from src.api.schemas import (
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    ModelReviewResponse,
    PredictRequest,
    PredictResponse,
)
from src.config import API_TITLE, API_VERSION, MODEL_PATH, MODEL_REVIEW_PATH
from src.explainability import get_top_feature_contributions
from src.trainers.sklearn_trainer import SklearnModelTrainer
from src.utils.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

_DATASET = load_breast_cancer()
_FEATURE_NAMES = list(_DATASET.feature_names)
_CLASS_LABELS = {index: str(label) for index, label in enumerate(_DATASET.target_names)}


def _load_persisted_model(path: str) -> Any:
    """Load either supported project persistence format.

    Args:
        path: Saved model artifact path.

    Returns:
        A fitted trainer or full sklearn pipeline.
    """
    persisted = joblib.load(path)
    if isinstance(persisted, dict) and {"estimator", "config", "is_fitted"}.issubset(
        persisted
    ):
        config = persisted["config"]
        trainer = SklearnModelTrainer(
            estimator=persisted["estimator"],
            name=config["name"],
            hyperparameters=config["hyperparameters"],
            random_state=config["random_state"],
        )
        return trainer.load(path)
    return persisted


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the saved model once while allowing health checks on failure.

    Args:
        app: FastAPI application receiving loaded state.

    Yields:
        Control to the running application.
    """
    app.state.model = None
    try:
        app.state.model = _load_persisted_model(MODEL_PATH)
        logger.info("Loaded model artifact from %s.", MODEL_PATH)
    except Exception:
        logger.exception("Unable to load model artifact from %s.", MODEL_PATH)
    yield


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Prevent unhandled internal exceptions from leaking to API clients.

    Args:
        request: Request that triggered the exception.
        exc: Internal exception to log.

    Returns:
        A generic client-safe error response.
    """
    logger.exception("Unhandled API exception for %s %s.", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _get_model() -> Any:
    """Return the startup-loaded model or a client-safe availability error.

    Returns:
        Loaded trainer or pipeline.

    Raises:
        HTTPException: If no model was loaded during application startup.
    """
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return model


def _prediction_frame(request: PredictRequest) -> pd.DataFrame:
    """Build one feature row in the original sklearn feature order.

    Args:
        request: Validated request feature values.

    Returns:
        A single-row DataFrame ordered by original feature names.
    """
    values = request.model_dump(by_alias=True)
    return pd.DataFrame([[values[name] for name in _FEATURE_NAMES]], columns=_FEATURE_NAMES)


def _predict(request: PredictRequest) -> Tuple[Any, pd.DataFrame, int, float]:
    """Run prediction and obtain predicted-class probability once.

    Args:
        request: Validated prediction request.

    Returns:
        Loaded model, input DataFrame, predicted class, and its probability.

    Raises:
        HTTPException: If model is unavailable or inference fails.
    """
    model = _get_model()
    frame = _prediction_frame(request)
    try:
        predicted_class = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0][predicted_class])
        return model, frame, predicted_class, probability
    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed")


def _explain_components(model: Any, frame: pd.DataFrame) -> Tuple[Any, Any]:
    """Extract a tree classifier and transformed row for local SHAP values.

    Args:
        model: Loaded trainer or sklearn pipeline.
        frame: Original-feature input DataFrame.

    Returns:
        Fitted classifier and a one-row transformed feature matrix.

    Raises:
        ValueError: If the loaded model cannot supply tree explanation inputs.
    """
    estimator = getattr(model, "_estimator", model)
    if not hasattr(estimator, "named_steps") or "classifier" not in estimator.named_steps:
        raise ValueError("Loaded model does not expose a pipeline classifier step.")
    return estimator.named_steps["classifier"], estimator[:-1].transform(frame)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health even when the model artifact failed to load.

    Returns:
        Current model availability and API version.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=getattr(app.state, "model", None) is not None,
        model_version=API_VERSION,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict one benchmark record using the startup-loaded model.

    Args:
        request: Validated feature values.

    Returns:
        Predicted class, label, probability, and API version.
    """
    _, _, predicted_class, probability = _predict(request)
    return PredictResponse(
        predicted_class=predicted_class,
        class_label=_CLASS_LABELS[predicted_class],
        probability=probability,
        model_version=API_VERSION,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(request: PredictRequest) -> ExplainResponse:
    """Predict one record and return only its five largest SHAP contributions.

    Args:
        request: Validated feature values.

    Returns:
        Prediction details and top local feature contributions.
    """
    model, frame, predicted_class, probability = _predict(request)
    try:
        classifier, transformed_row = _explain_components(model, frame)
        contributions = get_top_feature_contributions(
            classifier,
            transformed_row,
            _FEATURE_NAMES,
            top_n=5,
        )
        return ExplainResponse(
            predicted_class=predicted_class,
            class_label=_CLASS_LABELS[predicted_class],
            probability=probability,
            top_features=[FeatureContribution(**item) for item in contributions],
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Explanation failed.")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/model-review", response_model=ModelReviewResponse)
def model_review() -> ModelReviewResponse:
    """Return generated review markdown or an empty not-yet-generated state.

    Returning an empty successful response when the report is absent lets clients
    distinguish an unrun review workflow from a failed API endpoint.

    Returns:
        Persisted report content and whether it has been generated.
    """
    review_path = Path(MODEL_REVIEW_PATH)
    if not review_path.is_file():
        return ModelReviewResponse(content="", generated=False)
    try:
        return ModelReviewResponse(
            content=review_path.read_text(encoding="utf-8"),
            generated=True,
        )
    except Exception:
        logger.exception("Unable to read model review artifact.")
        raise HTTPException(status_code=500, detail="Model review unavailable")
