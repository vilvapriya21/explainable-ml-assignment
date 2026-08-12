"""Tests for compact, verified CrewAI artifact tools."""

from src.agent.tools import audit_model_comparison, read_explainability_summaries


def test_audit_model_comparison_verifies_all_models() -> None:
    """The one-shot audit retains every model and verifies its cost."""
    result = audit_model_comparison.invoke({"path": "artifacts/metrics.json"})

    assert set(result["models"]) == {
        "XGBoost (tuned)",
        "LightGBM (tuned)",
        "Random Forest baseline",
    }
    assert all(item["matches_reported"] for item in result["models"].values())


def test_explainability_tool_returns_compact_evidence() -> None:
    """The reviewer receives concise evidence instead of full raw artifacts."""
    result = read_explainability_summaries.invoke(
        {
            "shap_path": "artifacts/shap_summary.json",
            "lime_path": "artifacts/lime_summary.json",
        }
    )

    assert len(result["top_5_features"]) == 5
    assert len(result["false_negative"]["top_shap_contributions"]) == 5
    assert len(result["false_negative"]["top_lime_contributions"]) == 5
