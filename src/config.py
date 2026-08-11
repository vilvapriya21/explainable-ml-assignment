"""Environment-configurable application constants and artifact paths."""

import os

RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
FN_COST: float = float(os.getenv("FN_COST", "5.0"))
FP_COST: float = float(os.getenv("FP_COST", "1.0"))
ARTIFACTS_DIR: str = os.getenv("ARTIFACTS_DIR", "artifacts")
MODEL_PATH: str = os.getenv("MODEL_PATH", f"{ARTIFACTS_DIR}/model.joblib")
METRICS_PATH: str = os.getenv("METRICS_PATH", f"{ARTIFACTS_DIR}/metrics.json")
SHAP_SUMMARY_PATH: str = os.getenv(
    "SHAP_SUMMARY_PATH",
    f"{ARTIFACTS_DIR}/shap_summary.json",
)
LIME_SUMMARY_PATH: str = os.getenv(
    "LIME_SUMMARY_PATH",
    f"{ARTIFACTS_DIR}/lime_summary.json",
)
MODEL_REVIEW_PATH: str = os.getenv(
    "MODEL_REVIEW_PATH",
    f"{ARTIFACTS_DIR}/model_review.md",
)
LABEL_MAPPING_PATH: str = os.getenv(
    "LABEL_MAPPING_PATH",
    f"{ARTIFACTS_DIR}/label_mapping.json",
)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
AGENT_MAX_ITER: int = int(os.getenv("AGENT_MAX_ITER", "5"))
AGENT_TIMEOUT_SECONDS: int = int(os.getenv("AGENT_TIMEOUT_SECONDS", "300"))
API_TITLE: str = os.getenv("API_TITLE", "Explainable ML Model Evaluation API")
API_VERSION: str = os.getenv("API_VERSION", "1.0.0")
