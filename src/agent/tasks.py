"""Factory functions for Part G CrewAI tasks."""

import logging

from crewai import Agent, Task

from src.config import FN_COST, FP_COST

logger = logging.getLogger(__name__)


def build_metrics_analysis_task(agent: Agent) -> Task:
    """Build the metrics analysis task.

    Args:
        agent: Agent responsible for metrics analysis.

    Returns:
        A configured metrics analysis task.
    """
    return Task(
        description=(
            "Read artifacts/metrics.json, validate every model has accuracy, "
            "precision, recall, F1, confusion matrix, ROC-AUC, and business cost. "
            f"Compare all models and independently recompute each business cost "
            f"using false-negative cost={FN_COST} and false-positive "
            f"cost={FP_COST} from src.config."
        ),
        expected_output=(
            "A per-model metric summary with comparison findings and explicit "
            "discrepancy flags for every reported versus recomputed business cost."
        ),
        agent=agent,
    )


def build_explainability_review_task(agent: Agent) -> Task:
    """Build the independent explainability review task.

    Args:
        agent: Agent responsible for explainability review.

    Returns:
        A configured explainability review task.
    """
    return Task(
        description=(
            "Independently read SHAP and LIME summaries, identify the top five "
            "features, flag SHAP/LIME disagreement or surprising false-negative "
            "drivers, and summarize the false-negative record in plain language."
        ),
        expected_output=(
            "Structured findings listing top features, local false-negative "
            "evidence, method agreement or disagreement, and reliability concerns."
        ),
        agent=agent,
    )


def build_recommendation_task(agent: Agent) -> Task:
    """Build the final recommendation task.

    Args:
        agent: Agent responsible for final recommendation content.

    Returns:
        A configured recommendation task without crew-level context wiring.
    """
    # TODO: workflow.py supplies metrics and explainability task context at assembly.
    return Task(
        description=(
            "Combine the Metrics Analyst and Explainability Reviewer outputs, "
            "recommend one model, explain performance trade-offs, suggest concrete "
            "production monitoring metrics, and produce markdown for "
            "artifacts/model_review.md."
        ),
        expected_output=(
            "Well-formed markdown with sections: Recommendation, Reasoning, Risk, "
            "Trade-off, and Monitoring Suggestions."
        ),
        agent=agent,
    )
