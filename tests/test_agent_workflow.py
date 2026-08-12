"""Tests for workflow guardrails without real CrewAI execution."""

import json

import pytest

from src.agent import workflow


def test_run_model_review_returns_missing_file_failure(tmp_path) -> None:
    missing_metrics = tmp_path / "missing_metrics.json"

    result = workflow.run_model_review(
        metrics_path=str(missing_metrics),
        shap_path=str(tmp_path / "shap.json"),
        lime_path=str(tmp_path / "lime.json"),
    )

    assert result["status"] == "failed"
    assert str(missing_metrics) in result["missing_files"]


def test_run_model_review_returns_malformed_json_failure(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    shap_path = tmp_path / "shap.json"
    lime_path = tmp_path / "lime.json"
    metrics_path.write_text("{invalid json", encoding="utf-8")
    shap_path.write_text(json.dumps({}), encoding="utf-8")
    lime_path.write_text(json.dumps({}), encoding="utf-8")

    result = workflow.run_model_review(
        metrics_path=str(metrics_path),
        shap_path=str(shap_path),
        lime_path=str(lime_path),
    )

    assert result["status"] == "failed"
    assert "Malformed JSON" in result["reason"]


def test_run_model_review_writes_mocked_crew_output(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.json"
    shap_path = tmp_path / "shap.json"
    lime_path = tmp_path / "lime.json"
    output_path = tmp_path / "model_review.md"
    for path in (metrics_path, shap_path, lime_path):
        path.write_text(json.dumps({}), encoding="utf-8")

    class MockCrew:
        def kickoff(self) -> str:
            return "## Recommendation\nMock Model\n\nReasoning text."

    monkeypatch.setattr(
        workflow,
        "_build_crew",
        lambda max_crew_iterations: (MockCrew(), object()),
    )

    result = workflow.run_model_review(
        metrics_path=str(metrics_path),
        shap_path=str(shap_path),
        lime_path=str(lime_path),
        output_path=str(output_path),
    )

    assert result == {
        "status": "success",
        "output_path": str(output_path),
        "recommended_model": "Mock Model",
    }
    assert output_path.read_text(encoding="utf-8").startswith("## Recommendation")


def test_iteration_limited_agent_output_is_rejected() -> None:
    """A partial CrewAI result must not be written as a model review."""

    class MockTaskOutput:
        raw = "Agent stopped due to iteration limit or time limit."

    class MockResult:
        raw = "## Recommendation\nXGBoost"
        tasks_output = [MockTaskOutput()]

    with pytest.raises(ValueError, match="reached its iteration limit"):
        workflow._assert_no_agent_iteration_limit(MockResult())


def test_markdown_output_removes_duplicate_file_section() -> None:
    """Repeated fenced report content is not persisted in the final artifact."""
    result = (
        "## Recommendation\nXGBoost\n\n## artifacts/model_review.md\n```markdown\n"
        "## Recommendation\nXGBoost\n```"
    )

    assert workflow._markdown_output(result) == "## Recommendation\nXGBoost"
