"""Factory functions for Part G CrewAI tasks."""

import json
import logging
from typing import Any

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
            "Call Audit Model Comparison exactly once with path 'artifacts/metrics.json'. "
            "It verifies every business cost using false-negative cost="
            f"{FN_COST} and false-positive cost={FP_COST}. Then produce a final "
            "analysis that validates accuracy, precision, recall, F1, confusion "
            "matrix, ROC-AUC, and reported business cost for all three models. "
            "State the lowest-cost model and any discrepancy between reported and "
            "recomputed cost. Do not perform another tool call. Your next response must be plain text beginning "
            "with 'Final Answer:'; do not emit an Action or Action Input."
        ),
        expected_output=(
            "A per-model metric summary with comparison findings and explicit "
            "discrepancy flags for every reported versus recomputed business cost. "
            "The response must begin with 'Final Answer:'."
        ),
        agent=agent,
    )


def build_explainability_review_task(agent: Agent, evidence: dict[str, Any]) -> Task:
    """Build the independent explainability review task.

    Args:
        agent: Agent responsible for explainability review.
        evidence: Validated, compact SHAP and LIME evidence.

    Returns:
        A configured explainability review task.
    """
    return Task(
        description=(
            "Review the validated SHAP and LIME evidence below. Identify the top five "
            "features, describe the false-negative record in plain language, and flag "
            "agreement, disagreement, or uncertainty between SHAP and LIME. Do not "
            "invent facts beyond this evidence. Return plain text beginning with "
            "'Final Answer:'.\n\nValidated evidence:\n"
            f"{json.dumps(evidence, indent=2)}"
        ),
        expected_output=(
            "Structured findings listing top features, local false-negative "
            "evidence, method agreement or disagreement, and reliability concerns. "
            "The response must begin with 'Final Answer:'."
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
            "artifacts/model_review.md. Use only the provided task context; do not call tools. Return markdown "
            "beginning exactly with '## Recommendation' and name the selected model "
            "on the next line. The verified costs are XGBoost (tuned)=4, LightGBM "
            "(tuned)=8, and Random Forest baseline=14; never state the reverse. "
            "XGBoost and LightGBM have equal accuracy (0.972), while XGBoost has "
            "the slightly higher F1 and lower cost. Include only these sections: "
            "Recommendation, Reasoning, Risk, Trade-off, and Monitoring Suggestions. "
            "Do not include code fences, a second report, or a file-path heading."
        ),
        expected_output=(
            "Well-formed markdown with sections: Recommendation, Reasoning, Risk, "
            "Trade-off, and Monitoring Suggestions."
        ),
        agent=agent,
    )
