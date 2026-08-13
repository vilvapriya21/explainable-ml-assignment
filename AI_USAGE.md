# AI Usage Disclosure

## Tools Used

OpenAI Codex and Claude were used as implementation and review assistants
during development. Claude assisted with early project scaffolding, the A1-A3
work, and iterative fix cycles. OpenAI Codex assisted with implementation,
testing, notebook execution, dependency configuration, and Docker debugging.

## What the Tools Assisted With

- Drafting source modules, tests, notebooks, and documentation from the
  assessment requirements.
- Suggesting fixes after local test, notebook, PyTorch, CrewAI, and Docker
  failures.
- Reviewing dependency imports and helping separate runtime dependencies from
  notebook and test dependencies.
- Generating candidate explanations and report wording that were checked
  against the computed artifacts.

## Maintainer Review and Changes

The maintainer ran the local virtual environment, notebooks, Docker workflow,
and CrewAI workflow, then reported the observed repeated tool calls, artifact
path mismatch, and Docker trainer permission failure for correction. The
comparison artifact verified the final cost ordering as XGBoost=4, LightGBM=8,
and Random Forest=14, which was used to check the final model-review wording.
The maintainer retained responsibility for selecting the environment, running
the project, and accepting the submitted code and written conclusions.
