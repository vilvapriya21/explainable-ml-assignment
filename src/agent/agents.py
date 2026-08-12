"""Factory functions for the Part G CrewAI agents."""

import logging

from crewai import Agent

from src.agent.llm import build_review_llm
from src.agent.tools import (
    audit_model_comparison,
)
from src.config import AGENT_MAX_ITER

logger = logging.getLogger(__name__)


def _build_agent(**kwargs: object) -> Agent:
    """Create an agent with the explicitly configured non-OpenAI LLM.

    Args:
        **kwargs: CrewAI agent configuration values.

    Returns:
        An agent using the selected local LLM provider.
    """
    return Agent(llm=build_review_llm(), **kwargs)


def build_metrics_analyst_agent() -> Agent:
    """Build an agent that audits model performance and business cost.

    Returns:
        A configured Metrics Analyst agent.
    """
    return _build_agent(
        role="Metrics Analyst",
        goal=(
            "Read metrics.json, validate every model's required metrics, compare "
            "performance, and independently verify weighted business cost."
        ),
        backstory=(
            "You are an ML performance auditor who checks reported metrics against "
            "raw confusion matrices. You identify discrepancies and explain "
            "performance trade-offs using evidence from artifacts."
        ),
        tools=[audit_model_comparison],
        verbose=True,
        max_iter=AGENT_MAX_ITER,
        allow_delegation=False,
    )


def build_explainability_reviewer_agent() -> Agent:
    """Build an agent that reviews SHAP and LIME evidence.

    Returns:
        A configured Explainability Reviewer agent.
    """
    return _build_agent(
        role="Explainability Reviewer",
        goal=(
            "Read SHAP and LIME summaries, identify influential features, review "
            "the false-negative explanation, and flag reliability concerns."
        ),
        backstory=(
            "You are an ML explainability reviewer focused on whether local and "
            "global evidence agree. You report feature-level reliability concerns "
            "without treating learned associations as causal facts."
        ),
        tools=[],
        verbose=True,
        max_iter=AGENT_MAX_ITER,
        allow_delegation=False,
    )


def build_recommendation_agent() -> Agent:
    """Build an agent that produces the final model review recommendation.

    Returns:
        A configured Recommendation Agent.
    """
    return _build_agent(
        role="Recommendation Agent",
        goal=(
            "Combine metrics and explainability findings, recommend one model, "
            "explain trade-offs, suggest monitoring, and produce final review "
            "content for artifacts/model_review.md."
        ),
        backstory=(
            "You are a decision-focused ML reviewer who turns audited metrics and "
            "explainability findings into clear operational guidance. You prioritize "
            "business constraints and transparent risks over unsupported claims."
        ),
        tools=[],
        verbose=True,
        max_iter=AGENT_MAX_ITER,
        allow_delegation=False,
    )
