"""Guarded sequential workflow for the three-agent model review crew.

The workflow is sequential because it is a fixed evidence pipeline: metrics
and explainability review precede one final recommendation. It does not need
the dynamic delegation behavior of a hierarchical crew.
"""

import json
import logging
import re
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Tuple

from src.config import AGENT_MAX_ITER, AGENT_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = AGENT_TIMEOUT_SECONDS


def _failure(reason: str, missing_files: List[str] | None = None) -> Dict[str, Any]:
    """Build a consistent workflow failure result.

    Args:
        reason: Human-readable failure reason.
        missing_files: Input paths that could not be found.

    Returns:
        Structured failure information for the caller.
    """
    return {
        "status": "failed",
        "reason": reason,
        "missing_files": missing_files or [],
    }


def _validate_json_inputs(paths: List[str]) -> Tuple[bool, str]:
    """Confirm that all existing workflow inputs contain valid JSON.

    Args:
        paths: Artifact paths whose JSON syntax must be checked.

    Returns:
        Whether validation succeeded and a contextual error message when it did not.
    """
    for path in paths:
        try:
            json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return False, f"Malformed JSON in {path!r}: {exc.msg}."
        except OSError as exc:
            return False, f"Unable to read {path!r}: {exc}."
    return True, ""


def _build_crew(max_crew_iterations: int) -> Tuple[Any, Any]:
    """Construct the ordered crew and retain the recommendation task.

    Args:
        max_crew_iterations: Explicit maximum iteration count per agent.

    Returns:
        The configured Crew instance and its final recommendation task.
    """
    from crewai import Crew, Process

    from src.agent.agents import (
        build_explainability_reviewer_agent,
        build_metrics_analyst_agent,
        build_recommendation_agent,
    )
    from src.agent.tasks import (
        build_explainability_review_task,
        build_metrics_analysis_task,
        build_recommendation_task,
    )
    from src.agent.tools import read_explainability_summaries

    metrics_agent = build_metrics_analyst_agent()
    explainability_agent = build_explainability_reviewer_agent()
    recommendation_agent = build_recommendation_agent()
    for agent in (metrics_agent, explainability_agent, recommendation_agent):
        agent.max_iter = max_crew_iterations

    metrics_task = build_metrics_analysis_task(metrics_agent)
    explainability_evidence = read_explainability_summaries.func(
        shap_path="artifacts/shap_summary.json",
        lime_path="artifacts/lime_summary.json",
    )
    explainability_task = build_explainability_review_task(
        explainability_agent,
        explainability_evidence,
    )
    recommendation_task = build_recommendation_task(recommendation_agent)
    recommendation_task.context = [metrics_task, explainability_task]
    crew = Crew(
        agents=[metrics_agent, explainability_agent, recommendation_agent],
        tasks=[metrics_task, explainability_task, recommendation_task],
        process=Process.sequential,
        verbose=True,
    )
    return crew, recommendation_task


def _kickoff_with_timeout(crew: Any, timeout_seconds: int) -> Tuple[bool, Any]:
    """Run crew kickoff in a bounded worker thread.

    Args:
        crew: Crew instance to execute.
        timeout_seconds: Maximum number of seconds to wait.

    Returns:
        Whether kickoff completed and either its result or its exception.
    """
    outcome: Dict[str, Any] = {}

    def kickoff() -> None:
        try:
            outcome["result"] = crew.kickoff()
        except Exception as exc:
            outcome["exception"] = exc

    worker = Thread(target=kickoff, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        return False, None
    if "exception" in outcome:
        raise outcome["exception"]
    return True, outcome.get("result")


def _markdown_output(result: Any) -> str:
    """Extract markdown text from a CrewAI kickoff result.

    Args:
        result: Crew kickoff result.

    Returns:
        Final recommendation markdown text.

    Raises:
        ValueError: If result does not contain non-empty text.
    """
    text = getattr(result, "raw", result)
    markdown = str(text).strip()
    if not markdown:
        raise ValueError("Crew returned empty recommendation output.")
    duplicate_marker = "\n## artifacts/model_review.md"
    if duplicate_marker in markdown:
        markdown = markdown.split(duplicate_marker, maxsplit=1)[0].rstrip()
    return markdown


def _assert_no_agent_iteration_limit(result: Any) -> None:
    """Reject a crew result when an agent did not complete its assigned work.

    Args:
        result: Crew kickoff result, including any individual task outputs.

    Raises:
        ValueError: If a task stopped because its iteration or time limit was met.
    """
    task_outputs = getattr(result, "tasks_output", [])
    output_texts = [str(getattr(output, "raw", output)) for output in task_outputs]
    output_texts.append(str(getattr(result, "raw", result)))
    if any("stopped due to iteration limit or time limit" in text.lower() for text in output_texts):
        raise ValueError(
            "A CrewAI agent reached its iteration limit before completing its task; "
            "no model review was written."
        )


def _recommended_model(markdown: str) -> str:
    """Extract the recommended model name from final markdown.

    Args:
        markdown: Final report content.

    Returns:
        Recommended model name.

    Raises:
        ValueError: If the final report has no discernible recommendation.
    """
    patterns = (
        r"recommended(?:\s+model)?\s*[:\-]\s*\**([^\n*]+)",
        r"^#{1,6}\s*recommendation\s*\n\s*\**([^\n*]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, markdown, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            model_name = match.group(1).strip(" #:-")
            if model_name:
                return model_name
    raise ValueError("Final recommendation output does not name a recommended model.")


def run_model_review(
    metrics_path: str = "artifacts/metrics.json",
    shap_path: str = "artifacts/shap_summary.json",
    lime_path: str = "artifacts/lime_summary.json",
    output_path: str = "artifacts/model_review.md",
    max_crew_iterations: int = AGENT_MAX_ITER,
) -> dict:
    """Run the three-agent review crew and write the final report.

    Args:
        metrics_path: Path to the model metrics artifact.
        shap_path: Path to the SHAP summary artifact.
        lime_path: Path to the LIME summary artifact.
        output_path: Destination markdown report path.
        max_crew_iterations: Explicit maximum iterations for each agent.

    Returns:
        A structured success or failure result.
    """
    if max_crew_iterations <= 0:
        reason = "max_crew_iterations must be a positive integer."
        logger.error(reason)
        return _failure(reason)

    input_paths = [metrics_path, shap_path, lime_path]
    missing_files = [path for path in input_paths if not Path(path).is_file()]
    if missing_files:
        reason = f"Required review artifact is missing: {missing_files[0]!r}."
        logger.error(reason)
        return _failure(reason, missing_files)

    valid_json, validation_reason = _validate_json_inputs(input_paths)
    if not valid_json:
        logger.error("Model review pre-flight validation failed: %s", validation_reason)
        return _failure(validation_reason)

    try:
        crew, recommendation_task = _build_crew(max_crew_iterations)
        del recommendation_task
        completed, result = _kickoff_with_timeout(crew, _DEFAULT_TIMEOUT_SECONDS)
        if not completed:
            reason = "execution timeout exceeded"
            logger.error("Model review crew %s after %s seconds.", reason, _DEFAULT_TIMEOUT_SECONDS)
            return _failure(reason)

        _assert_no_agent_iteration_limit(result)
        markdown = _markdown_output(result)
        recommended_model = _recommended_model(markdown)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown, encoding="utf-8")
        logger.info("Wrote model review report to %s.", output_path)
        return {
            "status": "success",
            "output_path": output_path,
            "recommended_model": recommended_model,
        }
    except ValueError as exc:
        logger.error("Model review artifact validation failed: %s", exc)
        return _failure(str(exc))
    except Exception as exc:
        logger.exception("Model review crew execution failed.")
        return _failure(str(exc))


if __name__ == "__main__":
    print(run_model_review())
