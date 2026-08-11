"""Tests for the local structured recommendation chain."""

import pytest

from src.recommendation_chain import (
    BusinessConstraint,
    ModelRecommendation,
    RecommendationRequest,
    get_model_recommendation,
)


def test_recommendation_selects_lowest_business_cost_model() -> None:
    request = RecommendationRequest(
        model_metrics={
            "model_a": {"FN": 4, "FP": 1, "precision": 0.94, "recall": 0.80},
            "model_b": {"FN": 1, "FP": 3, "precision": 0.88, "recall": 0.93},
            "model_c": {"FN": 2, "FP": 5, "precision": 0.90, "recall": 0.88},
        },
        business_constraint=BusinessConstraint(
            false_negative_cost=5,
            false_positive_cost=1,
        ),
    )

    result = get_model_recommendation(request)

    assert result.recommended_model == "model_b"
    assert "model_b" in result.main_reason


def test_recommendation_returns_complete_validated_schema() -> None:
    request = RecommendationRequest(
        model_metrics={
            "model_a": {
                "confusion_matrix": [[90, 4], [2, 104]],
                "precision": 0.96,
                "recall": 0.98,
            }
        },
        business_constraint=BusinessConstraint(
            false_negative_cost=5,
            false_positive_cost=1,
        ),
    )

    result = get_model_recommendation(request)

    assert isinstance(result, ModelRecommendation)
    assert all(
        isinstance(value, str) and value
        for value in (
            result.recommended_model,
            result.main_reason,
            result.important_risk,
            result.suggested_next_action,
        )
    )


def test_recommendation_rejects_empty_model_metrics() -> None:
    request = RecommendationRequest(
        model_metrics={},
        business_constraint=BusinessConstraint(
            false_negative_cost=5,
            false_positive_cost=1,
        ),
    )

    with pytest.raises(ValueError, match="nothing to recommend"):
        get_model_recommendation(request)


def test_recommendation_rejects_missing_error_count_data() -> None:
    request = RecommendationRequest(
        model_metrics={
            "incomplete_model": {"precision": 0.90, "recall": 0.85},
        },
        business_constraint=BusinessConstraint(
            false_negative_cost=5,
            false_positive_cost=1,
        ),
    )

    with pytest.raises(ValueError, match="missing false-negative"):
        get_model_recommendation(request)
