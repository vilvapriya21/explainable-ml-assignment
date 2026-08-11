"""Reusable LangChain recommendation chain for model-selection decisions."""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class BusinessConstraint(BaseModel):
    """Represent the relative cost of binary-classification errors.

    Args:
        false_negative_cost: Cost assigned to one false negative.
        false_positive_cost: Cost assigned to one false positive.
    """

    false_negative_cost: float
    false_positive_cost: float


class RecommendationRequest(BaseModel):
    """Represent candidate model metrics and business constraints.

    Args:
        model_metrics: Metrics keyed by candidate model name.
        business_constraint: Costs used to compare classification errors.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_metrics: Dict[str, Dict[str, Any]]
    business_constraint: BusinessConstraint


class ModelRecommendation(BaseModel):
    """Represent a validated model-selection recommendation.

    Args:
        recommended_model: Model selected for the stated constraints.
        main_reason: Primary evidence supporting the selection.
        important_risk: Tradeoff requiring attention.
        suggested_next_action: Practical follow-up action.
    """

    recommended_model: str
    main_reason: str
    important_risk: str
    suggested_next_action: str


class DeterministicMockLLM(Runnable[Any, str]):
    """Generate data-driven structured recommendations without external APIs."""

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        """Return a JSON recommendation derived from the rendered prompt.

        Args:
            input: Rendered LangChain prompt containing request JSON.
            config: Optional runnable configuration.
            **kwargs: Additional runnable invocation options.

        Returns:
            JSON text compatible with ModelRecommendation.

        Raises:
            ValueError: If request JSON or required error counts are invalid.
        """
        del config, kwargs
        prompt_text = self._prompt_text(input)
        request_data = self._request_from_prompt(prompt_text)
        return self._recommend(request_data).model_dump_json()

    @staticmethod
    def _prompt_text(prompt: Any) -> str:
        """Extract text from a rendered prompt value.

        Args:
            prompt: Prompt value supplied by LCEL.

        Returns:
            Concatenated prompt message content.
        """
        if hasattr(prompt, "to_messages"):
            return "\n".join(str(message.content) for message in prompt.to_messages())
        return str(prompt)

    @staticmethod
    def _request_from_prompt(prompt_text: str) -> RecommendationRequest:
        """Parse the delimited request JSON embedded in the prompt.

        Args:
            prompt_text: Rendered reviewer prompt.

        Returns:
            Validated recommendation request.

        Raises:
            ValueError: If the prompt does not contain valid request JSON.
        """
        match = re.search(
            r"REQUEST_JSON_START\s*(\{.*?\})\s*REQUEST_JSON_END",
            prompt_text,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError("Recommendation prompt does not contain request JSON.")
        try:
            return RecommendationRequest.model_validate(json.loads(match.group(1)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "Recommendation prompt contains invalid request JSON."
            ) from exc

    def _recommend(self, request: RecommendationRequest) -> ModelRecommendation:
        """Build a recommendation using actual candidate error counts.

        Args:
            request: Validated metrics and business constraints.

        Returns:
            A data-driven model recommendation.

        Raises:
            ValueError: If no candidates or required error counts are present.
        """
        if not request.model_metrics:
            raise ValueError("model_metrics is empty; at least one model is required.")

        costs: Dict[str, float] = {}
        error_counts: Dict[str, Tuple[float, float]] = {}
        for name, metrics in request.model_metrics.items():
            false_negatives, false_positives = self._error_counts(name, metrics)
            error_counts[name] = (false_negatives, false_positives)
            costs[name] = (
                request.business_constraint.false_negative_cost * false_negatives
                + request.business_constraint.false_positive_cost * false_positives
            )

        recommended_name = min(costs, key=costs.get)
        selected_metrics = request.model_metrics[recommended_name]
        selected_fn, selected_fp = error_counts[recommended_name]
        selected_precision = self._metric_value(selected_metrics, "precision")
        selected_recall = self._metric_value(selected_metrics, "recall")
        main_reason = (
            f"{recommended_name} has the lowest business cost "
            f"({costs[recommended_name]:.2f}) with {selected_fn:.0f} false "
            f"negatives and {selected_fp:.0f} false positives; its precision is "
            f"{selected_precision:.3f} and recall is {selected_recall:.3f}."
        )
        return ModelRecommendation(
            recommended_model=recommended_name,
            main_reason=main_reason,
            important_risk=self._risk(
                recommended_name,
                request.model_metrics,
                selected_precision,
                selected_fp,
            ),
            suggested_next_action=(
                "Re-validate this choice on a larger holdout set and monitor the "
                "false-negative rate after deployment."
            ),
        )

    @staticmethod
    def _metric_value(metrics: Dict[str, Any], name: str) -> float:
        """Return a numeric metric or zero when it is absent.

        Args:
            metrics: Metrics for one candidate model.
            name: Metric name to retrieve.

        Returns:
            The metric value, or zero when unavailable.
        """
        return float(metrics.get(name, 0.0))

    @staticmethod
    def _error_counts(name: str, metrics: Dict[str, Any]) -> Tuple[float, float]:
        """Extract false-negative and false-positive counts from model metrics.

        Args:
            name: Candidate model name for error context.
            metrics: Metrics for one candidate model.

        Returns:
            False-negative and false-positive counts, respectively.

        Raises:
            ValueError: If metrics lacks usable error-count information.
        """
        false_negatives = metrics.get("false_negatives", metrics.get("FN"))
        false_positives = metrics.get("false_positives", metrics.get("FP"))
        if false_negatives is not None and false_positives is not None:
            return float(false_negatives), float(false_positives)

        matrix = metrics.get("confusion_matrix")
        if (
            isinstance(matrix, list)
            and len(matrix) == 2
            and all(isinstance(row, list) and len(row) == 2 for row in matrix)
        ):
            return float(matrix[1][0]), float(matrix[0][1])
        raise ValueError(
            f"Model {name!r} is missing false-negative/false-positive counts or "
            "a 2x2 confusion_matrix."
        )

    def _risk(
        self,
        recommended_name: str,
        metrics_by_model: Dict[str, Dict[str, Any]],
        selected_precision: float,
        selected_false_positives: float,
    ) -> str:
        """Describe a concrete tradeoff visible among candidate metrics.

        Args:
            recommended_name: Selected model name.
            metrics_by_model: Metrics for every candidate.
            selected_precision: Precision for the selected model.
            selected_false_positives: False-positive count for the selection.

        Returns:
            A tradeoff statement based on candidate data.
        """
        highest_precision_name = max(
            metrics_by_model,
            key=lambda name: self._metric_value(metrics_by_model[name], "precision"),
        )
        highest_precision = self._metric_value(
            metrics_by_model[highest_precision_name],
            "precision",
        )
        if highest_precision_name != recommended_name and highest_precision > selected_precision:
            return (
                f"{recommended_name} has lower precision ({selected_precision:.3f}) "
                f"than {highest_precision_name} ({highest_precision:.3f}), so its "
                "positive predictions may require closer review."
            )
        return (
            f"Even at the lowest cost, {recommended_name} produces "
            f"{selected_false_positives:.0f} false positives that should be monitored."
        )


def get_model_recommendation(
    request: RecommendationRequest,
    llm: Optional[Any] = None,
) -> ModelRecommendation:
    """Return a validated, constraint-aware model recommendation.

    Args:
        request: Candidate metrics and business error costs.
        llm: Optional LangChain-compatible runnable replacing the local mock.

    Returns:
        A validated structured recommendation.

    Raises:
        ValueError: If candidates are absent, metrics are incomplete, or output
            cannot be parsed into ModelRecommendation.
    """
    if not request.model_metrics:
        raise ValueError("model_metrics is empty; there is nothing to recommend.")

    output_parser = PydanticOutputParser(pydantic_object=ModelRecommendation)
    prompt = ChatPromptTemplate.from_template(
        "You are an ML reviewer. Assess candidate models using the business "
        "constraints and return exactly the requested structured fields.\n"
        "REQUEST_JSON_START\n{request_json}\nREQUEST_JSON_END\n"
        "{format_instructions}"
    )
    chain = prompt | (llm or DeterministicMockLLM()) | output_parser
    try:
        result = chain.invoke(
            {
                "request_json": request.model_dump_json(),
                "format_instructions": output_parser.get_format_instructions(),
            }
        )
    except ValueError:
        raise
    except Exception as exc:
        model_names = ", ".join(request.model_metrics)
        raise ValueError(
            "Unable to parse a model recommendation for candidates: "
            f"{model_names}."
        ) from exc

    if not isinstance(result, ModelRecommendation):
        raise ValueError("Recommendation output was not a ModelRecommendation.")

    logger.info("Generated recommendation for %s.", result.recommended_model)
    return result
